# The checkpoint's format: where every tensor is and how many bytes it takes in each dtype, the quantizer, and the
# Writer that puts the tensors where they belong, a piece at a time.
import math
import struct

import numpy as np

from engine.layout import FORM, MATRIX, QUANTIZED, TABLE, VECTOR, Row, check_suited, file_size, form_of, placed, tensor_rows
from engine.packing import TERNARY_GROUP, pack6, quantize6, ternary


def group_size(row_length):
    size = 32
    while row_length % size:
        size //= 2
    return size


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


# the dtypes with groups and scales; int6 is T98's, see llama2_numpy.pack6; ternary T230's (pack_ternary): the weights
# of a ternary model as they are, two bits each, which no other model can be written as (Writer refuses):
# engine/layout.py's QUANTIZED
# the two a model converted with no dtype asked for may get (Stream's callable dtype, T115)
EITHER = ("int8", "int6")


def dtype_name(dtype):
    """"float32", "float16", "int8", "int6" or "ternary" from a name or a NumPy dtype (NumPy has neither of the last
    two)."""
    return str(dtype) if str(dtype) in ("int6", "ternary") else np.dtype(dtype).name


def check_dtype(dtype):
    if str(dtype) not in ("int6", "ternary") and np.dtype(dtype) not in (np.float32, np.float16, np.int8):
        raise ValueError(f"dtype must be float32, float16, int8, int6 or ternary, not {dtype}.")


def tensor_bytes(shape, is_matrix, dtype):
    """How many bytes a tensor of layout() takes in a checkpoint of that dtype."""
    row = Row("", TABLE if is_matrix is None else MATRIX if is_matrix else VECTOR, tuple(shape))
    return placed([row], dtype_name(dtype))[0].size


def checkpoint_size(header, dtype, form=None):
    """The bytes of a checkpoint with this header, dtype and form (FORM: what the rows take besides the header)."""
    return file_size(tensor_rows(header, form), dtype_name(dtype))


def quantize(values):
    """float32 values, whole rows -> (int8 values, float32 scales), one scale per group of the row."""
    groups = values.reshape(-1, group_size(values.shape[-1]))
    scales = (np.abs(groups).max(axis=1) / 127.0).astype(np.float32)
    inverse = np.divide(1.0, scales, out=np.zeros_like(scales), where=scales > 0)
    return np.rint(groups * inverse[:, None]).astype(np.int8), scales


class Writer:
    """Puts pieces of the tensors of layout(), in any order, where they belong in the checkpoint buffer."""

    def __init__(self, out, header, dtype, form=None, sink=None, quantize_rows=None):
        """out: a buffer of the checkpoint's size, or None with sink: an object with open(size, header, dtype,
        form) and write(offset, array of bytes), for a checkpoint that lives outside Python (T93: the WebAssembly
        memory of public/forward.js, which the header and the rest size, T115). Pyodide's own memory never shrinks,
        so a converted model that went through a Python buffer on its way there would keep taking its size twice.
        form (llama2_numpy.FORM, see checkpoint_form()): what the file does not say, which lays out its tensors and
        sizes the forward pass besides the header; the sink gets it whole."""
        # quantize_rows: quantize() on the SIMD kernels (llama2_numpy.kernel_quantizer), the same bytes six times
        # faster, for rows of whole groups of 32; NumPy's quantize() for anything else, and where there are no kernels
        self.dtype, self.sink, self.quantize_rows = dtype_name(dtype), sink, quantize_rows
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
        # (where it begins, its shape, layout()'s "is a matrix") of every row, by its number in the file
        self.tensors = [(place.offset, place.row.shape, IS_MATRIX.get(place.row.role, True)) for place in placed(rows, self.dtype)]

    def put(self, offset, array):
        raw = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
        if self.sink is not None:
            self.sink.write(offset, raw)
        else:
            self.out[offset:offset + raw.size] = raw

    def write(self, index, first, values):
        """values: whole rows of tensor number index, beginning at its element number first."""
        offset, shape, is_matrix = self.tensors[index]
        if self.dtype not in QUANTIZED:
            self.put(offset + first * np.dtype(self.dtype).itemsize, np.asarray(values).astype(self.dtype, copy=False))
        elif is_matrix and self.dtype == "int6":
            rows = np.asarray(values, dtype=np.float32).reshape(-1, shape[-1])
            if self.quantize_rows is not None:
                packed, scales = self.quantize_rows(rows, "int6")  # the same bytes on the kernel (T98)
            else:
                quantized, scales = quantize6(rows)
                packed = pack6(quantized)
            self.put(offset + first * 3 // 4, packed)
            self.put(offset + math.prod(shape) * 3 // 4 + 4 * (first // 32), scales)
        elif is_matrix and self.dtype == "ternary":
            # T230: nothing is rounded, the values are ternary already or this raises (the same bytes on the kernel)
            rows = np.asarray(values, dtype=np.float32).reshape(-1, shape[-1])
            packed, scales = self.quantize_rows(rows, "ternary") if self.quantize_rows is not None else ternary(rows)
            self.put(offset + first // 4, packed)
            self.put(offset + math.prod(shape) // 4 + 4 * (first // TERNARY_GROUP), scales)
        elif is_matrix:
            fast = self.quantize_rows is not None and shape[-1] % 32 == 0
            quantized, scales = (self.quantize_rows if fast else quantize)(np.asarray(values, dtype=np.float32).reshape(-1, shape[-1]))
            self.put(offset + first, quantized)
            self.put(offset + math.prod(shape) + 4 * (first // group_size(shape[-1])), scales)
        elif is_matrix is False:
            self.put(offset + 4 * first, np.asarray(values, dtype=np.float32))
