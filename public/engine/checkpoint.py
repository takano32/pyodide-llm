# A checkpoint's tensors: where each is (Tensor, Places), the form a file does not say (FORM), the dtype read from
# its size, and the outlier channels of the final norm.
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
import math
import struct

import numpy as np

from engine.tokenizer import CHARSMAP
from engine.layers import convolution_form, linear_form, linear_widths, rope
from engine.packing import TERNARY_GROUP, group_of, stored_bytes

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


class Places:
    """Where the tensors of a checkpoint are, taken in file order (Llama's take() with external=, T93): a Tensor for
    each, from the header's 28 bytes on. dtype: the checkpoint's as numpy has it (int6 and ternary are int8 with that
    packing, PACKED)."""

    def __init__(self, dtype, packing=None):
        self.dtype, self.packing, self.offset = np.dtype(dtype), packing, 28

    def take(self, *shape, matrix=True, widen=True):
        count = math.prod(shape)
        if self.dtype == np.int8 and matrix:
            group, stored = group_of(shape[-1], self.packing), stored_bytes(count, self.packing)
            tensor = Tensor(self.packing or "int8", self.offset, shape, group, self.offset + stored)
            self.offset += stored + 4 * (count // group)
            return tensor
        tensor = Tensor("f16" if self.dtype == np.float16 else "f32", self.offset, shape)
        self.offset += count * (2 if self.dtype == np.float16 else 4)
        return tensor


# the attributes of Llama that are tensors of the file, in no particular order
TENSOR_NAMES = ("token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo", "rms_ffn_weight", "w1", "w2", "w3",
                "rms_final_weight", "freq_cis_real", "freq_cis_imag", "wcls", "bq", "bk", "bv", "positions",
                "ln_att_bias", "ln_ffn_bias", "ln_final_bias", "bo", "b1", "b2", "q_norm", "k_norm",
                "wg", "wqkv", "wz", "wb", "wa", "conv", "dt_bias", "decay", "delta_norm", "wout", "win")


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


# The form of a checkpoint: what sets its tensors and sizes its forward pass besides the 7 ints of the header, which
# the legacy file cannot say (see Llama.__init__), with the value of a file that says nothing. The converter writes
# them into the options, and one dict of these names goes to everything that lays the file out or sizes it
# (llama2_convert.layout(), checkpoint_size() and Writer, checkpoint_dtype() below, forward.js's footprint()), so
# that another one is added where it is used, not along the way (T144).
# linear (T229): the linear-attention layers of arch "qwen35", see linear_form(); None where there are none.
# convolution (T260): the convolution layers of arch "lfm2", see convolution_form(); None where there are none.
FORM = {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0, "linear": None, "rotated": None,
        "convolution": None}


def form_of(options=None):
    """The form (FORM's keys, each with its default where options has none) out of options, a dict with those and
    any others (the options of a model, a manifest's). Dicts from JavaScript are read too (a JsProxy). A key given as
    None (a JSON null, or JavaScript's undefined) says nothing, as head_size() reads "head_dim": null: it had
    checkpoint_dtype() fail on int(None) while footprint() counted dim / heads (the review of T144)."""
    options = options.to_py() if hasattr(options, "to_py") else (options or {})
    return {key: default if options.get(key) is None else options[key] for key, default in FORM.items()}


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
    q_dim, kv_dim = n_heads * head_size, n_kv_heads * head_size
    rope = 2 * seq_len * (head_size // 2)
    if arch in ("gpt2", "neox"):
        # the same tensors in the same order as gpt2_tensors() and llama2_convert.layout(arch=): q, k, v, o, the two
        # FFN matrices (no gate), and for GPT-2 the table of positions in place of the RoPE tables
        matrices = [(abs(vocab_size), dim)] + [(n_layers * dim, dim)] * 4 + [(n_layers * hidden_dim, dim), (n_layers * dim, hidden_dim)]
        if arch == "gpt2":
            matrices.append((seq_len, dim))
            rope = 0
        # LayerNorm weights and biases (two per layer, one at the end), the biases of q, k, v, o and the FFN
        vectors = n_layers * (4 * dim + 3 * dim + dim + hidden_dim + dim) + 2 * dim
    elif arch == "qwen35":
        # the same tensors in the same order as qwen35_tensors() and llama2_convert.layout(arch=): the full-attention
        # layers' (q, its gate, k, v, o), the linear-attention layers' (q, k and v in one, z, the output), the FFN
        linear = linear_form(form["linear"])
        if linear is None or n_layers < linear["every"]:
            raise ValueError("This is not a llama2.c checkpoint: a hybrid model has to say its linear layers.")
        mixed, _, read = linear_widths(linear)
        full = n_layers // linear["every"]
        lines = n_layers - full
        matrices = [(abs(vocab_size), dim), (full * q_dim, dim), (full * q_dim, dim), (full * kv_dim, dim),
                    (full * kv_dim, dim), (full * dim, q_dim), (lines * mixed, dim), (lines * read, dim),
                    (lines * dim, read), (n_layers * hidden_dim, dim), (n_layers * dim, hidden_dim),
                    (n_layers * hidden_dim, dim)]
        # the norms of the layers and of the heads of q and k, and of a linear layer: the two small matrices of its
        # gates (float32 whatever the file), the taps, dt_bias, decay and the norm of a value head
        vectors = 2 * n_layers * dim + dim + 2 * full * head_size \
            + lines * (2 * linear["value_heads"] * dim + linear["conv"] * mixed + 2 * linear["value_heads"] + linear["value_dim"])
    elif arch == "lfm2":
        # the same tensors in the same order as lfm2_tensors() and llama2_convert.layout(arch=): the attention
        # layers' (q, k, v, o), the convolution layers' (the matrix in, the matrix out), the FFN
        convolution = convolution_form(form["convolution"])
        if convolution is None or len(convolution["layers"]) != n_layers:
            raise ValueError("This is not a llama2.c checkpoint: an LFM2 has to say its convolution layers.")
        short = convolution["layers"].count("c")
        full = n_layers - short
        matrices = [(abs(vocab_size), dim), (full * q_dim, dim), (full * kv_dim, dim), (full * kv_dim, dim),
                    (full * dim, q_dim), (short * 3 * dim, dim), (short * dim, dim), (n_layers * hidden_dim, dim),
                    (n_layers * dim, hidden_dim), (n_layers * hidden_dim, dim)]
        # the norms of the layers and of the heads of q and k, and a convolution layer's taps
        vectors = 2 * n_layers * dim + dim + 2 * full * head_size + short * convolution["taps"] * dim
    else:
        # the same tensors in the same order as llama_tensors() and quantize.py: (rows, row length) of the matrices
        matrices = [(abs(vocab_size), dim), (n_layers * q_dim, dim), (n_layers * kv_dim, dim), (n_layers * kv_dim, dim),
                    (n_layers * dim, q_dim), (n_layers * hidden_dim, dim), (n_layers * dim, hidden_dim),
                    (n_layers * hidden_dim, dim)]
        vectors = 2 * n_layers * dim + dim + (n_layers * (q_dim + 2 * kv_dim) if form["bias"] else 0) \
            + (2 * n_layers * head_size if form["qk_norm"] else 0)
    if vocab_size < 0:
        matrices.append((abs(vocab_size), dim))
    floats = sum(rows * length for rows, length in matrices) + vectors + rope

    def group(length):
        size = 32
        while length % size:
            size //= 2
        return size

    # quantize.py: int8 values and a float32 scale per group; the vectors stay float32, the RoPE tables are left out
    int8 = sum(rows * length + 4 * (rows * length // group(length)) for rows, length in matrices) + 4 * vectors
    sizes = {28 + 4 * floats: "float32", 28 + 2 * floats: "float16", 28 + int8: "int8"}
    if all(length % 32 == 0 for _, length in matrices):
        # T98: 24 bytes and a float32 scale per group of 32 (only rows of whole groups can be int6)
        sizes.setdefault(28 + sum(rows * length // 32 * 28 for rows, length in matrices) + 4 * vectors, "int6")
    if all(length % TERNARY_GROUP == 0 for _, length in matrices):
        # T230: 32 bytes and a float32 scale per group of 128 (only rows of whole groups can be ternary). No other
        # dtype of the same header has this size: with M values in the matrices and V in the vectors it is 0.28125 M
        # + 4 V, int6 0.875 M + 4 V, int8 1.125 M + 4 V, float32 more than 4 M + 4 V, and float16 (2 M + 2 V and the
        # RoPE tables) would need 2 V > 1.7 M, vectors as large as the matrices (tests/test_ternary.py tries shapes)
        sizes.setdefault(28 + sum(rows * length // TERNARY_GROUP * 36 for rows, length in matrices) + 4 * vectors, "ternary")
    if size not in sizes:
        raise ValueError(f"This is not a llama2.c checkpoint: its header asks for {28 + 4 * floats} bytes as float32, "
                         f"{28 + 2 * floats} as float16 or {28 + int8} as int8, and the file has {size}.")
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
