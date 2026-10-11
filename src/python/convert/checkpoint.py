# The checkpoint's format as the converter asks for it: where every tensor is and how many bytes it takes in each dtype
# (engine/layout.py's rows, engine/dtypes.py's kinds), and the Writer that puts the tensors where they belong, a piece
# at a time.
import struct

import numpy as np

from engine.dtypes import DTYPES, dtype_of
from engine.layout import FORM, MATRIX, TABLE, VECTOR, Row, check_suited, file_size, form_of, placed, tensor_rows

# what layout() says of a row: True for what int8 quantizes, False for the norm weights, None for the RoPE tables
IS_MATRIX = {VECTOR: False, TABLE: None}


def layout(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, **form):
    """(shape, is a matrix) of every tensor, in file order: the rows of engine/layout.py (tensor_rows(), which says
    what each is and takes the same header and form, FORM's keys), as the pairs quantize.py and the tests read.

    is a matrix: True for what int8 quantizes, False for the norm weights, None for the RoPE tables.
    """
    if set(form) - set(FORM):
        # (as when the form was keyword arguments: a key that is misspelt is no form, and form_of() would drop it unheard)
        raise TypeError(f"layout() got an unexpected keyword argument {sorted(set(form) - set(FORM))[0]!r}")
    header = (dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
    return [(row.shape, IS_MATRIX.get(row.role, True)) for row in tensor_rows(header, form)]


def tensor_bytes(shape, is_matrix, dtype):
    """How many bytes a tensor of layout() takes in a checkpoint of that dtype."""
    row = Row("", TABLE if is_matrix is None else MATRIX if is_matrix else VECTOR, tuple(shape))
    return placed([row], dtype_of(dtype))[0].size


def checkpoint_size(header, dtype, form=None):
    """The bytes of a checkpoint with this header, dtype and form (FORM: what the rows take besides the header)."""
    return file_size(tensor_rows(header, form), dtype_of(dtype))


class Writer:
    """Puts pieces of the tensors of layout(), in any order, where they belong in the checkpoint buffer."""

    def __init__(self, out, header, dtype, form=None, sink=None, quantize_rows=None):
        """out: a buffer of the checkpoint's size, or None with sink: an object with open(size, header, dtype,
        form) and write(offset, array of bytes), for a checkpoint that lives outside Python (T93: the WebAssembly
        memory of public/forward.js, which the header and the rest size, T115). Pyodide's own memory never shrinks,
        so a converted model that went through a Python buffer on its way there would keep taking its size twice.
        form (llama2_numpy.FORM, see checkpoint_form()): what the file does not say, which lays out its tensors and
        sizes the forward pass besides the header; the sink gets it whole."""
        # quantize_rows: the dtypes' packing on the SIMD kernels (llama2_numpy.kernel_quantizer), the same bytes
        # several times faster, for rows of whole groups of 32; NumPy's (the dtype's pack) for anything else, and
        # where there are no kernels
        self.dtype, self.sink, self.quantize_rows = dtype_of(dtype), sink, quantize_rows
        form = form_of(form)
        rows = tensor_rows(header, form)
        check_suited(rows, self.dtype)
        size = file_size(rows, self.dtype)
        if sink is not None:
            self.out = None
            sink.open(size, list(header), self.dtype, form)
        else:
            self.out = np.frombuffer(out, dtype=np.uint8)
            assert self.out.size == size, "the buffer has not the size of the checkpoint"
        self.put(0, np.frombuffer(struct.pack("<7i", *header), dtype=np.uint8))
        # where every row is and how it is stored, by its number in the file
        self.places = placed(rows, self.dtype)

    def put(self, offset, array):
        raw = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
        if self.sink is not None:
            self.sink.write(offset, raw)
        else:
            self.out[offset:offset + raw.size] = raw

    def write(self, index, first, values):
        """values: whole rows of tensor number index, beginning at its element number first. They are stored as the
        row's kind (its place's: engine/layout.py's kind_of()), whatever the file's dtype is: the vectors of a
        quantized file as float32, a row the form gives a kind of its own as that, and a row the file leaves out not
        at all."""
        place = self.places[index]
        if place.kind is None:
            return
        kind = DTYPES[place.kind]
        if not kind.bits:
            self.put(place.offset + first * kind.itemsize, np.asarray(values).astype(kind.name, copy=False))
            return
        rows = np.asarray(values, dtype=np.float32).reshape(-1, place.row.shape[-1])
        # (the kernels pack whole groups of 32 or more: an int8 row of smaller groups is NumPy's. T230: nothing of a
        # ternary row is rounded, the values are ternary already or this raises, on the kernel as in NumPy)
        fast = self.quantize_rows is not None and place.group % 32 == 0
        packed, scales = self.quantize_rows(rows, kind.name) if fast else kind.pack(rows)
        self.put(place.offset + kind.stored_bytes(first, place.group)[0], packed)
        self.put(place.scales + 4 * (first // place.group), scales)
