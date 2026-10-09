# The checkpoint's format: where every tensor is and how many bytes it takes in each dtype, the quantizer, and the
# Writer that puts the tensors where they belong, a piece at a time.
import math
import struct

import numpy as np

from llama2_numpy import (TERNARY_GROUP, convolution_form, form_of, linear_form, linear_widths, pack6, quantize6,
                          ternary)
from convert.config import head_size


def group_size(row_length):
    size = 32
    while row_length % size:
        size //= 2
    return size


def layout(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, bias=False, arch="llama", qk_norm=False,
           head_dim=0, linear=None, rotated=None, convolution=None):
    """(shape, is a matrix) of every tensor, in file order. llama2_numpy.py reads the same order.

    is a matrix: True for what int8 quantizes, False for the norm weights, None for the RoPE tables.
    bias: the model adds a bias after the q, k and v projections (Qwen2). Those three vectors per layer go last,
    so that a checkpoint without them is byte for byte the file it always was.
    qk_norm: the model normalizes every head of q and k before RoPE (Qwen3, T124): the two weights of a head's size
    per layer go after the biases, for the same reason.
    head_dim: the size of a head where it is not dim / n_heads (0: it is). Then q and the attention's output are
    n_heads * head_dim wide, not dim (Qwen3 0.6B: 16 heads of 128 in a dim of 1024, T124).
    linear: the linear-attention layers of arch "qwen35" (T229, llama2_numpy.linear_form()). Its tensors are stacked
    by the kind of the layer: those of the full-attention layers (q, its gate, k, v, o, the norms of the heads of q
    and k), those of the linear-attention layers (q, k and v in one matrix, z, the two small matrices of the gates,
    which are never quantized, the taps of the convolution, dt_bias, the decay, the norm of a value head, the output),
    and the FFN of every layer.
    rotated: the form's rotated basis (T237), which moves no tensor: the same ones are stored in another basis.
    convolution: the convolution layers of arch "lfm2" (T260, llama2_numpy.convolution_form()). Its tensors are stacked
    by the kind of the layer too: those of the attention layers (q, k, v, o, the norms of the heads of q and k), those
    of the convolution layers (the matrix in, of 3 dim rows; the taps, which are never quantized; the matrix out),
    and the FFN of every layer.
    """
    head_size = head_dim or dim // n_heads
    q_dim, kv_dim = n_heads * head_size, n_kv_heads * head_size
    if arch == "lfm2":
        convolution = convolution_form(convolution, n_layers)
        short = convolution["layers"].count("c")
        full = n_layers - short
        tensors = [((abs(vocab_size), dim), True), ((n_layers, dim), False),
                   ((full, q_dim, dim), True), ((full, kv_dim, dim), True), ((full, kv_dim, dim), True),
                   ((full, dim, q_dim), True), ((full, head_size), False), ((full, head_size), False),
                   ((short, 3 * dim, dim), True), ((short, convolution["taps"], dim), False), ((short, dim, dim), True),
                   ((n_layers, dim), False),
                   ((n_layers, hidden_dim, dim), True), ((n_layers, dim, hidden_dim), True), ((n_layers, hidden_dim, dim), True),
                   ((dim,), False), ((seq_len, head_size // 2), None), ((seq_len, head_size // 2), None)]
        if vocab_size < 0:
            tensors.append(((abs(vocab_size), dim), True))
        return tensors
    if arch == "qwen35":
        linear = linear_form(linear)
        mixed, _, read = linear_widths(linear)
        full = n_layers // linear["every"]
        lines, values = n_layers - full, linear["value_heads"]
        tensors = [((abs(vocab_size), dim), True), ((n_layers, dim), False),
                   ((full, q_dim, dim), True), ((full, q_dim, dim), True),
                   ((full, kv_dim, dim), True), ((full, kv_dim, dim), True), ((full, dim, q_dim), True),
                   ((full, head_size), False), ((full, head_size), False),
                   ((lines, mixed, dim), True), ((lines, read, dim), True),
                   ((lines, values, dim), False), ((lines, values, dim), False),
                   ((lines, linear["conv"], mixed), False), ((lines, values), False), ((lines, values), False),
                   ((lines, linear["value_dim"]), False), ((lines, dim, read), True),
                   ((n_layers, dim), False),
                   ((n_layers, hidden_dim, dim), True), ((n_layers, dim, hidden_dim), True), ((n_layers, hidden_dim, dim), True),
                   ((dim,), False), ((seq_len, head_size // 2), None), ((seq_len, head_size // 2), None)]
        if vocab_size < 0:
            tensors.append(((abs(vocab_size), dim), True))
        return tensors
    if arch in ("gpt2", "neox"):
        # GPT-2: LayerNorm (a weight and a bias), a bias after every projection, learned positions instead of
        # RoPE, and an FFN of two matrices instead of three (no gate). Same attention.
        # GPT-NeoX is the same, except that it rotates part of each head (so it keeps the RoPE tables of the
        # Llama layout in place of the table of positions).
        vector = lambda n=dim: ((n_layers, n), False)
        positions = [((seq_len, head_size // 2), None), ((seq_len, head_size // 2), None)] if arch == "neox" \
            else [((seq_len, dim), True)]
        tensors = [((abs(vocab_size), dim), True), *positions,
                   vector(), vector(),
                   ((n_layers, dim, dim), True), ((n_layers, dim, dim), True), ((n_layers, dim, dim), True),
                   vector(), vector(), vector(),
                   ((n_layers, dim, dim), True), vector(),
                   vector(), vector(),
                   ((n_layers, hidden_dim, dim), True), vector(hidden_dim),
                   ((n_layers, dim, hidden_dim), True), vector(),
                   ((dim,), False), ((dim,), False)]
        if vocab_size < 0:
            tensors.append(((abs(vocab_size), dim), True))
        return tensors
    tensors = [((abs(vocab_size), dim), True), ((n_layers, dim), False),
               ((n_layers, q_dim, dim), True), ((n_layers, kv_dim, dim), True), ((n_layers, kv_dim, dim), True),
               ((n_layers, dim, q_dim), True), ((n_layers, dim), False),
               ((n_layers, hidden_dim, dim), True), ((n_layers, dim, hidden_dim), True), ((n_layers, hidden_dim, dim), True),
               ((dim,), False), ((seq_len, head_size // 2), None), ((seq_len, head_size // 2), None)]
    if vocab_size < 0:
        tensors.append(((abs(vocab_size), dim), True))
    if bias:
        tensors += [((n_layers, q_dim), False), ((n_layers, kv_dim), False), ((n_layers, kv_dim), False)]
    if qk_norm:
        tensors += [((n_layers, head_size), False), ((n_layers, head_size), False)]
    return tensors


# the dtypes with groups and scales; int6 is T98's, see llama2_numpy.pack6; ternary T230's (pack_ternary): the weights
# of a ternary model as they are, two bits each, which no other model can be written as (Writer refuses)
QUANTIZED = ("int8", "int6", "ternary")
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
    count, dtype = math.prod(shape), dtype_name(dtype)
    if dtype not in QUANTIZED:
        return count * np.dtype(dtype).itemsize
    if is_matrix is None:
        return 0  # int8 and int6 checkpoints leave the RoPE tables out
    if dtype == "int6":
        # 24 bytes of values and a float32 scale per group of 32; the norm weights stay float32
        return count // 32 * 28 if is_matrix else 4 * count
    if dtype == "ternary":
        # 32 bytes of values and a float32 scale per group of 128
        return count // TERNARY_GROUP * 36 if is_matrix else 4 * count
    # int8 values and one float32 scale per group; the norm weights stay float32
    return count + 4 * (count // group_size(shape[-1])) if is_matrix else 4 * count


def checkpoint_size(header, dtype, form=None):
    """The bytes of a checkpoint with this header, dtype and form (llama2_numpy.FORM: what layout() takes besides
    the header)."""
    return 28 + sum(tensor_bytes(shape, is_matrix, dtype) for shape, is_matrix in layout(*header, **form_of(form)))


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
        tensors = layout(*header, **form)
        if self.dtype == "int6" and any(is_matrix and shape[-1] % 32 for shape, is_matrix in tensors):
            raise ValueError("Six bits a weight needs rows of whole groups of 32, and this model has other rows.")
        if self.dtype == "ternary" and any(is_matrix and shape[-1] % TERNARY_GROUP for shape, is_matrix in tensors):
            raise ValueError("Ternary weights need rows of whole groups of 128, and this model has other rows.")
        size = checkpoint_size(header, dtype, form)
        if sink is not None:
            self.out = None
            sink.open(size, list(header), self.dtype, form)
        else:
            self.out = np.frombuffer(out, dtype=np.uint8)
            assert self.out.size == size, "the buffer has not the size of the checkpoint"
        self.put(0, np.frombuffer(struct.pack("<7i", *header), dtype=np.uint8))
        self.tensors, offset = [], 28
        for shape, is_matrix in tensors:
            self.tensors.append((offset, shape, is_matrix))
            offset += tensor_bytes(shape, is_matrix, dtype)

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
