# What a model's file holds: the tensors of a checkpoint as rows (name, role, shape, which layers a stack covers), for
# each of the layouts, in the order of the file; the form a file does not say (FORM); how each row is stored in a file
# of a dtype, and where. Everything that needs the order of the tensors reads it here (T359): the converter's layout()
# and Writer, checkpoint_dtype(), the engine's attributes, external_tensors(), and conversion_plan() by the rows' names.
# This file imports nothing of the engine or of the converter: both read it.
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
from typing import NamedTuple

# ---- the form
# The form of a checkpoint: what sets its tensors and sizes its forward pass besides the 7 ints of the header, which
# the legacy file cannot say (see Llama.__init__), with the value of a file that says nothing. The converter writes
# them into the options, and one dict of these names goes to everything that lays the file out or sizes it
# (tensor_rows() below and what reads it, forward.js's footprint()), so that another one is added where it is used,
# not along the way (T144).
# linear (T229): the linear-attention layers of arch "qwen35", see linear_form(); None where there are none.
# rotated (T237): a rotated basis, which moves no tensor: the same ones are stored in another basis.
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


# (what a Gated DeltaNet layer computes with these numbers: the comment on Qwen3.5 in engine/layers.py)
LINEAR = ("every", "key_heads", "value_heads", "key_dim", "value_dim", "conv")


def linear_form(linear):
    """The numbers of a hybrid model's linear-attention layers, FORM's "linear", as a dict of ints (LINEAR's keys:
    every "every"-th layer is a full-attention one, the heads and their sizes, the taps of the convolution), from a
    dict of Python or of JavaScript. None for a model without such layers."""
    if linear is None:
        return None
    linear = linear.to_py() if hasattr(linear, "to_py") else linear
    numbers = {key: int(linear[key]) for key in LINEAR}
    if min(numbers.values()) < 1 or numbers["every"] < 2 or numbers["value_heads"] % numbers["key_heads"]:
        raise ValueError(f"These are not the numbers of linear-attention layers: {numbers}.")
    return numbers


def linear_widths(linear):
    """(the values the convolution runs over: q, k and v; those of q or of k; those of v) of a linear-attention layer."""
    keys, values = linear["key_heads"] * linear["key_dim"], linear["value_heads"] * linear["value_dim"]
    return 2 * keys + values, keys, values


# (what a convolution layer computes: the comment on LFM2 in engine/layers.py)
CONVOLUTION = ("layers", "taps")


def convolution_form(convolution, n_layers=None):
    """The convolution layers of an LFM2, FORM's "convolution", as {"layers": a letter for every layer, "c" for a
    convolution layer and "a" for one that attends, "taps": how many tokens the convolution reads}, from a dict of
    Python or of JavaScript. None for a model without such layers. n_layers: the header's, which the letters have to be
    as many as."""
    if convolution is None:
        return None
    convolution = convolution.to_py() if hasattr(convolution, "to_py") else convolution
    layers, taps = str(convolution["layers"]), int(convolution["taps"])
    if not layers or set(layers) - set("ac") or taps < 2 or (n_layers is not None and len(layers) != n_layers):
        raise ValueError(f"These are not the convolution layers of a model: {layers!r} with {taps} taps.")
    return {"layers": layers, "taps": taps}


def layer_slots(n_layers, linear, convolution=None):
    """For every layer: (whether it keeps a state in place of keys and values: a Qwen3.5's linear-attention layer or an
    LFM2's convolution layer, its place among the layers of its kind), which is where its tensors are in the file's
    stacks: a model whose layers all attend has (False, l) for layer l."""
    if convolution is not None:
        kinds = [kind == "c" for kind in convolution["layers"]]
    else:
        kinds = [linear is not None and (l + 1) % linear["every"] != 0 for l in range(n_layers)]
    slots, counts = [], [0, 0]
    for kind in kinds:
        slots.append((kind, counts[kind]))
        counts[kind] += 1
    return slots


# ---- the rows
# What a row is to the engine and to a quantized file (kind_of() below):
#   MATRIX      what an activation is multiplied by; a quantized file holds it quantized
#   EMBEDDING   the table of the tokens' rows, a matrix to the file (and the classifier, where the model has no other)
#   CLASSIFIER  the classifier of a model whose embedding is not it (a negative vocabulary in the header)
#   POSITIONS   GPT-2's learned table of positions, a matrix to the file
#   VECTOR      the weights of a norm, a bias, and whatever else is small and never quantized (the two small matrices of
#               a linear-attention layer's gates, the taps of a convolution): float32 in a quantized file
#   TABLE       a RoPE table: left out of a quantized file (the engine computes it)
MATRIX, EMBEDDING, CLASSIFIER, POSITIONS, VECTOR, TABLE = "matrix", "embedding", "classifier", "positions", "vector", "table"
# Which layers the first axis of a row's shape stacks, in their order (None: one tensor for the model):
#   EVERY       every layer
#   ATTENDING   the layers that keep keys and values (all of them, but in a hybrid model)
#   STATEFUL    the layers that keep a state in their place: a Qwen3.5's linear-attention layers, an LFM2's convolutions
EVERY, ATTENDING, STATEFUL = "every", "attending", "stateful"


class Row(NamedTuple):
    """A tensor of the file. name: the attribute the engine holds it as (and its key in the plan forward.js gets)."""
    name: str
    role: str
    shape: tuple
    per: str = None


class Dims:
    """The numbers the rows' shapes are made of: the header's 7 ints, and what the form (FORM) adds to them."""

    def __init__(self, header, form=None):
        form = form_of(form)
        (self.dim, self.hidden_dim, self.n_layers, self.n_heads, self.n_kv_heads, vocab_size,
         self.seq_len) = (int(value) for value in header)
        # (a negative vocabulary in the header: the classifier is a tensor of its own)
        self.vocab_size, self.shared = abs(vocab_size), vocab_size > 0
        self.arch, self.bias, self.qk_norm = form["arch"], bool(form["bias"]), bool(form["qk_norm"])
        # the size of a head where it is not dim / n_heads (T124: Qwen3 0.6B has 16 heads of 128 in a dim of 1024);
        # q and the attention's output are n_heads * head_size wide
        self.head_size = int(form["head_dim"]) or self.dim // self.n_heads
        self.q_dim, self.kv_dim = self.n_heads * self.head_size, self.n_kv_heads * self.head_size
        self.linear = linear_form(form["linear"]) if self.arch == "qwen35" else None
        self.convolution = convolution_form(form["convolution"], self.n_layers) if self.arch == "lfm2" else None
        if (self.arch == "qwen35" and self.linear is None) or (self.arch == "lfm2" and self.convolution is None):
            raise ValueError("A hybrid model (qwen35) has to say its linear layers, and an LFM2 its convolution layers.")
        self.slots = layer_slots(self.n_layers, self.linear, self.convolution)

    def layers(self, per):
        """The layers a stack covers, in the order of the stack."""
        return [layer for layer, (stateful, _) in enumerate(self.slots)
                if per == EVERY or stateful == (per == STATEFUL)]

    def stack(self, name, role, per, *shape):
        """A row with one tensor of this shape for each of the layers of per."""
        return Row(name, role, (len(self.layers(per)), *shape), per)


# The parts the layouts are made of. A layout is a function of the Dims that returns the rows in file order, and a
# new one is written from these and from the others (after() puts rows behind a row of another layout).
def embedding(d):
    return [Row("token_embedding_table", EMBEDDING, (d.vocab_size, d.dim))]


def attention(d, q_dim=None, kv_dim=None):
    """q, k, v and o of the layers that attend."""
    q_dim, kv_dim = q_dim or d.q_dim, kv_dim or d.kv_dim
    return [d.stack("wq", MATRIX, ATTENDING, q_dim, d.dim), d.stack("wk", MATRIX, ATTENDING, kv_dim, d.dim),
            d.stack("wv", MATRIX, ATTENDING, kv_dim, d.dim), d.stack("wo", MATRIX, ATTENDING, d.dim, q_dim)]


def head_norms(d):
    """The RMSNorm weights of q and k, of a head's size (Qwen3, T124)."""
    return [d.stack("q_norm", VECTOR, ATTENDING, d.head_size), d.stack("k_norm", VECTOR, ATTENDING, d.head_size)]


def gated_ffn(d):
    """The norm before the FFN and its three matrices (gate, down, up)."""
    return [d.stack("rms_ffn_weight", VECTOR, EVERY, d.dim), d.stack("w1", MATRIX, EVERY, d.hidden_dim, d.dim),
            d.stack("w2", MATRIX, EVERY, d.dim, d.hidden_dim), d.stack("w3", MATRIX, EVERY, d.hidden_dim, d.dim)]


def rope_tables(d):
    return [Row("freq_cis_real", TABLE, (d.seq_len, d.head_size // 2)), Row("freq_cis_imag", TABLE, (d.seq_len, d.head_size // 2))]


def classifier(d):
    return [] if d.shared else [Row("wcls", CLASSIFIER, (d.vocab_size, d.dim))]


def after(rows, name, *more):
    """rows with more put behind the row called name."""
    at = [row.name for row in rows].index(name) + 1
    return [*rows[:at], *more, *rows[at:]]


def llama(d):
    """A Llama (llama2.c's file), and what was added behind it so that a checkpoint without is the file it always
    was: the q, k and v biases of a Qwen2, then the norms of the heads of q and k of a Qwen3."""
    rows = [*embedding(d), d.stack("rms_att_weight", VECTOR, EVERY, d.dim), *attention(d), *gated_ffn(d),
            Row("rms_final_weight", VECTOR, (d.dim,)), *rope_tables(d), *classifier(d)]
    if d.bias:
        rows += [d.stack("bq", VECTOR, ATTENDING, d.q_dim), d.stack("bk", VECTOR, ATTENDING, d.kv_dim),
                 d.stack("bv", VECTOR, ATTENDING, d.kv_dim)]
    return rows + head_norms(d) if d.qk_norm else rows


def qwen35(d):
    """A Qwen3.5 (T229), its tensors stacked by the kind of the layer: those of the full-attention layers (q, its gate,
    k, v, o, the norms of the heads of q and k), those of the linear-attention layers (q, k and v in one matrix, z, the
    two small matrices of the gates, which are never quantized: they feed a sigmoid and an exp; the taps of the
    convolution, dt_bias, the decay, the norm of a value head, the output), and the FFN of every layer."""
    mixed, _, read = linear_widths(d.linear)
    values = d.linear["value_heads"]
    return [*embedding(d), d.stack("rms_att_weight", VECTOR, EVERY, d.dim),
            *after(attention(d), "wq", d.stack("wg", MATRIX, ATTENDING, d.q_dim, d.dim)), *head_norms(d),
            d.stack("wqkv", MATRIX, STATEFUL, mixed, d.dim), d.stack("wz", MATRIX, STATEFUL, read, d.dim),
            d.stack("wb", VECTOR, STATEFUL, values, d.dim), d.stack("wa", VECTOR, STATEFUL, values, d.dim),
            d.stack("conv", VECTOR, STATEFUL, d.linear["conv"], mixed),
            d.stack("dt_bias", VECTOR, STATEFUL, values), d.stack("decay", VECTOR, STATEFUL, values),
            d.stack("delta_norm", VECTOR, STATEFUL, d.linear["value_dim"]), d.stack("wout", MATRIX, STATEFUL, d.dim, read),
            *gated_ffn(d), Row("rms_final_weight", VECTOR, (d.dim,)), *rope_tables(d), *classifier(d)]


def lfm2(d):
    """An LFM2 (T260), stacked by the kind of the layer too: those of the attention layers (q, k, v, o, the norms of the
    heads of q and k), those of the convolution layers (the matrix in, whose 3 dim rows are B, C and what B multiplies;
    the taps, which are never quantized; the matrix out), and the FFN of every layer."""
    return [*embedding(d), d.stack("rms_att_weight", VECTOR, EVERY, d.dim), *attention(d), *head_norms(d),
            d.stack("win", MATRIX, STATEFUL, 3 * d.dim, d.dim), d.stack("conv", VECTOR, STATEFUL, d.convolution["taps"], d.dim),
            d.stack("wout", MATRIX, STATEFUL, d.dim, d.dim),
            *gated_ffn(d), Row("rms_final_weight", VECTOR, (d.dim,)), *rope_tables(d), *classifier(d)]


def gpt2(d):
    """A GPT-2: LayerNorm (a weight and a bias), a bias after every projection, a learned table of positions instead
    of RoPE, and an FFN of two matrices instead of three (no gate). Same attention, of heads that fill dim.
    A GPT-NeoX is the same, except that it rotates part of each head (so it keeps the RoPE tables of the Llama layout
    in place of the table of positions)."""
    vector = lambda name, width=d.dim: d.stack(name, VECTOR, EVERY, width)
    q, k, v, o = attention(d, d.dim, d.dim)
    return [*embedding(d), *(rope_tables(d) if d.arch == "neox" else [Row("positions", POSITIONS, (d.seq_len, d.dim))]),
            vector("rms_att_weight"), vector("ln_att_bias"), q, k, v, vector("bq"), vector("bk"), vector("bv"),
            o, vector("bo"),
            vector("rms_ffn_weight"), vector("ln_ffn_bias"),
            d.stack("w1", MATRIX, EVERY, d.hidden_dim, d.dim), vector("b1", d.hidden_dim),
            d.stack("w2", MATRIX, EVERY, d.dim, d.hidden_dim), vector("b2"),
            Row("rms_final_weight", VECTOR, (d.dim,)), Row("ln_final_bias", VECTOR, (d.dim,)), *classifier(d)]


# the layout of an architecture (FORM's "arch"); any other name is read as a Llama, as the forward pass does
LAYOUTS = {"llama": llama, "gpt2": gpt2, "neox": gpt2, "qwen35": qwen35, "lfm2": lfm2}


def tensor_rows(header, form=None):
    """The tensors of a checkpoint with this header (7 ints) and form (FORM), as rows in file order."""
    d = Dims(header, form)
    return LAYOUTS.get(d.arch, llama)(d)


# ---- how a row is stored, and where
# A quantized dtype holds a matrix as values and one float32 scale for every group of a row: (the bits of a value, the
# values of a group; 0: 32, or the largest power of two below it that divides the row). int6 is T98's (pack6), ternary
# T230's (pack_ternary). (T359: the registry of the dtypes, step 2, takes these.)
PACKINGS = {"int8": (8, 0), "int6": (6, 32), "ternary": (2, 128)}
QUANTIZED = tuple(PACKINGS)
UNSUITED = {"int6": "Six bits a weight needs rows of whole groups of 32, and this model has other rows.",
            "ternary": "Ternary weights need rows of whole groups of 128, and this model has other rows."}


def kind_of(row, dtype):
    """How a row is stored in a file of this dtype ("float32", "float16", "int8", "int6", "ternary"), by the same
    names; None: it is left out. Today a file has one dtype and a row follows it by its role: every matrix is the
    file's, the vectors of a quantized file are float32 and it has no RoPE tables. This is the one place that chooses,
    so that a file whose tensors are of several kinds (T363) changes this function and nothing that reads it."""
    if dtype not in QUANTIZED:
        return dtype
    return "float32" if row.role == VECTOR else None if row.role == TABLE else dtype


def group_of(length, kind):
    """The values of a group of a row of this length, of a quantized kind."""
    group = PACKINGS[kind][1]
    if not group:
        group = 32
        while length % group:
            group //= 2
    return group


class Place(NamedTuple):
    """Where a row is in the file: its kind (kind_of(); None: nowhere, and size is 0), the offset of its values and
    all its bytes, and of a quantized kind the values of a group and the offset of the scales (0 and 0 otherwise)."""
    row: Row
    kind: str
    offset: int
    size: int
    group: int = 0
    scales: int = 0


def placed(rows, dtype, offset=28):
    """A Place for every row, in file order: the header's 28 bytes, then one tensor after the other."""
    places = []
    for row in rows:
        kind, count = kind_of(row, dtype), math.prod(row.shape)
        if kind in QUANTIZED:
            group = group_of(row.shape[-1], kind)
            values = count // group * (group * PACKINGS[kind][0] // 8)
            place = Place(row, kind, offset, values + 4 * (count // group), group, offset + values)
        else:
            place = Place(row, kind, offset, 0 if kind is None else count * (2 if kind == "float16" else 4))
        places.append(place)
        offset += place.size
    return places


def file_size(rows, dtype):
    """The bytes of a checkpoint of these rows and this dtype."""
    return 28 + sum(place.size for place in placed(rows, dtype))


def suited(rows, dtype):
    """Whether a file of this dtype can hold these rows: a packed kind needs rows of whole groups."""
    return all(row.shape[-1] % place.group == 0 for row, place in zip(rows, placed(rows, dtype)) if place.group)


def check_suited(rows, dtype):
    if not suited(rows, dtype):
        raise ValueError(UNSUITED[dtype])
