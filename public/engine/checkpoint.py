# A checkpoint's tensors where the weights live outside Python (Tensor, outside(), external_tensors()), the dtype read
# from its size, and the outlier channels of the final norm. What the file holds is engine/layout.py's.
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is
# derived from two works under the MIT License, whose notice follows: tairov/llama2.py
# (https://github.com/tairov/llama2.py; its LICENSE names no copyright holder) and karpathy/llama2.c
# (https://github.com/karpathy/llama2.c), Copyright (c) 2023 Andrej.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import struct

import numpy as np

from engine.tokenizer import CHARSMAP
from engine.layout import (TABLE, check_suited, convolution_form, file_size, form_of, linear_form, placed, suited,
                           tensor_rows)
from engine.packing import PACKED

# The classifier's input has a few channels that the final norm's weight blows up (openai-community/gpt2: 12 to 17
# times, 316 against a median of 0.3). With the int8 kernels the activations are quantized in groups of 32, so one
# such channel sets the scale of its group and the other 31 round to nothing: GPT-2 lost 17% of perplexity to
# that (T92). These many channels, the largest of the norm's weight, are taken out of the vector before it is
# quantized and multiplied in float32 by their own columns of the classifier (vocab_size * 8 multiply-adds next to
# vocab_size * dim). With them GPT-2 is back to +0.35%. Only a norm whose largest weight is OUTLIER_RATIO times
# its median gets this (GPT-2: 13.9; tiny-lm 1.1, llm-jp-3 150M 1.3, Pythia 160M 1.3, rinna GPT-2 1.4, SmolLM2 135M
# 1.9): the others gain nothing from it (tiny-lm stays at +0.31%) and would pay about 3% of speed.
OUTLIER_CHANNELS = 8
OUTLIER_RATIO = 4.0


class Tensor:
    """Where a tensor of the checkpoint is, when the weights live outside Python (T93: the forward pass runs in
    public/forward.js on its own WebAssembly memory). kind: "int8" (values, then one float32 scale per group of
    the last dimension at scales), "int6" (the same with the values packed, see pack6), "ternary" (T230: two bits a
    value and a scale per group of 128, see pack_ternary), "f32" or "f16". Offsets count
    from the start of the checkpoint file."""

    __slots__ = ("kind", "offset", "shape", "group", "scales")

    def __init__(self, kind, offset, shape, group=0, scales=0):
        self.kind, self.offset, self.shape, self.group, self.scales = kind, offset, tuple(shape), group, scales

    def plan(self):
        return {"kind": self.kind, "offset": self.offset, "shape": list(self.shape), "group": self.group,
                "scales": self.scales}


# what forward.js calls the two kinds that are not quantized (Tensor's kind)
SHORT = {"float32": "f32", "float16": "f16"}


def dtype_of(dtype):
    """A checkpoint's dtype by its name, from a name or a NumPy dtype (NumPy has neither int6 nor ternary)."""
    return str(dtype) if str(dtype) in PACKED else np.dtype(dtype).name


def outside(rows, dtype):
    """{name: Tensor} of the rows of a checkpoint of this dtype (by name) that the engine reads from the file: where
    Llama(external=) says they are. The RoPE tables are the file's in float32 alone (half precision is too coarse for
    the angles, and a quantized file leaves them out): the engine computes the others. ValueError where no file of
    this dtype can hold these rows."""
    check_suited(rows, dtype)
    return {place.row.name: Tensor(SHORT.get(place.kind, place.kind), place.offset, place.row.shape, place.group, place.scales)
            for place in placed(rows, dtype) if place.kind is not None and (place.row.role != TABLE or dtype == "float32")}


def external_tensors(header, dtype, form=None):
    """Where every tensor of a checkpoint with this header, dtype and form (FORM) is, {name: Tensor.plan()}, as
    Llama(external=) hands them to public/forward.js, before any of its bytes are there (T156: the worker sends the
    layers' matrices to the GPU as they come, and keeps the rest). A model whose embedding is its classifier has it
    under both names."""
    tensors = outside(tensor_rows(header, form), dtype_of(dtype))
    tensors.setdefault("wcls", tensors["token_embedding_table"])
    return {name: tensor.plan() for name, tensor in tensors.items()}


def outlier_channels(weight, count=OUTLIER_CHANNELS, ratio=OUTLIER_RATIO):
    """The count channels with the largest final-norm weight, in order, or none when the weight has no outliers
    (its largest is less than ratio times its median)."""
    size = np.abs(np.asarray(weight, dtype=np.float32))
    if size.max() < ratio * np.median(size):
        return np.zeros(0, dtype=np.intp)
    return np.sort(np.argsort(-size)[:count])


def outlier_columns(classifier, channels):
    """The columns of the int8 classifier for these channels, as float32, one column per row (len(channels),
    vocab_size): what the add_columns kernel multiplies (see OUTLIER_CHANNELS)."""
    values, scales = classifier  # (vocab_size, dim / group, group) int8 and (vocab_size, dim / group, 1) float32
    group = values.shape[-1]
    columns = np.stack([values[:, c // group, c % group].astype(np.float32) * scales[:, c // group, 0] for c in channels])
    return np.ascontiguousarray(columns)


def checkpoint_dtype(header, size, form=None):
    """"float32", "float16", "int8", "int6" or "ternary": what a checkpoint file of size bytes with this header (7
    ints) holds.

    The legacy format does not say, but the header fixes the size of each variant. Anything else is no checkpoint
    this engine can read, and the ValueError says so before hundreds of megabytes are read for nothing.
    form: what the file cannot say either (FORM, taken out of a model's options): the tensors differ with it.
    """
    form = form_of(form)
    arch = form["arch"]
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = (int(value) for value in header)
    head_size = int(form["head_dim"]) or (dim // n_heads if n_heads and dim % n_heads == 0 else 0)
    limit = 1 << 24
    if not (0 < dim < limit and 0 < hidden_dim < limit and 0 < n_layers < 4096 and 0 < n_kv_heads <= n_heads <= dim
            and 0 < abs(vocab_size) < limit and 0 < seq_len < limit and 0 < head_size < limit and n_heads % n_kv_heads == 0):
        raise ValueError("This is not a llama2.c checkpoint: the header makes no sense.")
    linear = linear_form(form["linear"]) if arch == "qwen35" else None
    if arch == "qwen35" and (linear is None or n_layers < linear["every"]):
        raise ValueError("This is not a llama2.c checkpoint: a hybrid model has to say its linear layers.")
    convolution = convolution_form(form["convolution"]) if arch == "lfm2" else None
    if arch == "lfm2" and (convolution is None or len(convolution["layers"]) != n_layers):
        raise ValueError("This is not a llama2.c checkpoint: an LFM2 has to say its convolution layers.")
    rows = tensor_rows(header, form)
    # quantize.py: int8 values and a float32 scale per group; the vectors stay float32, the RoPE tables are left out
    sizes = {file_size(rows, name): name for name in ("float32", "float16", "int8")}
    for name in PACKED:
        # T98: 24 bytes and a float32 scale per group of 32 (only rows of whole groups can be int6).
        # T230: 32 bytes and a float32 scale per group of 128 (only rows of whole groups can be ternary). No other
        # dtype of the same header has this size: with M values in the matrices and V in the vectors it is 0.28125 M
        # + 4 V, int6 0.875 M + 4 V, int8 1.125 M + 4 V, float32 more than 4 M + 4 V, and float16 (2 M + 2 V and the
        # RoPE tables) would need 2 V > 1.7 M, vectors as large as the matrices (tests/test_ternary.py tries shapes)
        if suited(rows, name):
            sizes.setdefault(file_size(rows, name), name)
    if size not in sizes:
        raise ValueError(f"This is not a llama2.c checkpoint: its header asks for {file_size(rows, 'float32')} bytes as "
                         f"float32, {file_size(rows, 'float16')} as float16 or {file_size(rows, 'int8')} as int8, and "
                         f"the file has {size}.")
    return sizes[size]


def check_tokenizer(tokenizer, header):
    """ValueError unless tokenizer (a tokenizer.bin) holds exactly the vocabulary of the checkpoint with this header.

    The engine would read the first pieces of a larger vocabulary without complaint, and write nonsense.
    """
    vocab_size = abs(int(list(header)[5]))
    offset, pieces = 4, 0
    while offset + 8 <= len(tokenizer):
        if pieces == vocab_size and bytes(tokenizer[offset:offset + len(CHARSMAP)]) == CHARSMAP:
            # T216: a sentencepiece model's normalizer after the pieces, to the end
            (size,) = struct.unpack_from("<I", tokenizer, offset + len(CHARSMAP))
            offset += len(CHARSMAP) + 4 + size
            break
        _, length = struct.unpack_from("<fi", tokenizer, offset)
        if length < 0 or offset + 8 + length > len(tokenizer):
            raise ValueError("The smaller file is not a llama2.c tokenizer.bin.")
        offset += 8 + length
        pieces += 1
    if offset != len(tokenizer) or pieces == 0:
        raise ValueError("The smaller file is not a llama2.c tokenizer.bin.")
    if pieces != vocab_size:
        raise ValueError(f"This tokenizer.bin holds {pieces} pieces, but the checkpoint has a vocabulary of "
                         f"{vocab_size}: they do not belong together.")
