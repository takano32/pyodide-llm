# What a model's file holds: the tensors of a checkpoint as rows (name, role, shape, which layers a stack covers), for
# each of the layouts, in the order of the file; the form a file does not say (FORM); how each row is stored in a file
# of a dtype, and where. Everything that needs the order of the tensors reads it here (T.359): the converter's layout()
# and Writer, checkpoint_dtype(), the engine's attributes, external_tensors(), and conversion_plan() by the rows' names.
# What a kind of storage is (its bytes, its groups) is engine/dtypes.py's, which this file reads, and nothing else of
# the engine or of the converter: both read it.
# The architectures are one table here too (LAYOUTS, T.359.6): what the file of each holds and what its forward pass is,
# as facts that every part of the engine reads in place of comparing the architecture's name.
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
from typing import Callable, NamedTuple, Optional

from engine.dtypes import BLOCKS, BYTES, CODEBOOKS, KINDS, QUANTIZED, kind_named

# ---- the form
# The form of a checkpoint: what sets its tensors and sizes its forward pass besides the 7 ints of the header, which
# the legacy file cannot say (see Llama.__init__), with the value of a file that says nothing. The converter writes
# them into the options, and one dict of these names goes to everything that lays the file out or sizes it
# (tensor_rows() below and what reads it, forward.js's footprint()), so that another one is added where it is used,
# not along the way (T.144).
# arch: the architecture, a name of LAYOUTS (below).
# linear (T.229): the linear-attention layers of arch "qwen35", see linear_form(); None where there are none.
# rotated (T.237): a rotated basis, which moves no tensor: the same ones are stored in another basis.
# convolution (T.260): the convolution layers of arch "lfm2", see convolution_form(); None where there are none.
# (linear and convolution are the kinds of layers that keep a state: each such kind has its numbers under its own
# name here, Stateful below, and stateful_form() holds them to the architecture)
# kinds (T.359): the rows that are not stored the way the file's dtype stores a row of their role, see kinds_form();
# None where every row is, which is every file a conversion writes unless it is asked to keep a GGUF's blocks (T.363).
# tiled (T.363): a linear-attention layer's output matrix has its columns in llama.cpp's order of the value heads (every
# key head's first one, then every key head's second, ...), which moves no tensor: its input is put in that order.
# wide (T.331): the layers whose heads are of a size of their own (a Gemma 4's full-attention layers: 512 where the
# others' are 256), see wide_form(): their q, k, v, o and the norms of their heads are stacks of their own. None: none.
# double (T.331): the layers whose FFN is twice as wide (a Gemma 4 E2B's last 20), see double_form(): its three
# matrices are stacks of their own. None: none.
# kv_shared (T.330): the layers that read the keys and values of a layer before them in place of their own, see
# shared_form(): they have no k, v and norm of k's heads. None: none.
# per_layer (T.329): the width of the input every layer takes of its own table (a Gemma 4's per-layer embedding:
# hidden_size_per_layer_input, 256); 0: none.
FORM = {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0, "linear": None, "rotated": None,
        "convolution": None, "kinds": None, "tiled": None, "wide": None, "double": None, "kv_shared": None,
        "per_layer": 0}


def form_of(options=None):
    """The form (FORM's keys, each with its default where options has none) out of options, a dict with those and
    any others (the options of a model, a manifest's). Dicts from JavaScript are read too (a JsProxy). A key given as
    None (a JSON null, or JavaScript's undefined) says nothing, as head_size() reads "head_dim": null: it had
    checkpoint_dtype() fail on int(None) while footprint() counted dim / heads (the review of T.144)."""
    options = options.to_py() if hasattr(options, "to_py") else (options or {})
    return {key: default if options.get(key) is None else options[key] for key, default in FORM.items()}


# (what a Gated DeltaNet layer computes with these numbers: the comment on Qwen3.5 in engine/layers.py)
LINEAR = ("every", "key_heads", "value_heads", "key_dim", "value_dim", "conv")


def linear_form(linear, n_layers=None):
    """The numbers of a hybrid model's linear-attention layers, FORM's "linear", as a dict of ints (LINEAR's keys:
    every "every"-th layer is a full-attention one, the heads and their sizes, the taps of the convolution), from a
    dict of Python or of JavaScript. None for a model without such layers. n_layers: the header's, which have to be
    enough for one full-attention layer."""
    if linear is None:
        return None
    linear = linear.to_py() if hasattr(linear, "to_py") else linear
    numbers = {key: int(linear[key]) for key in LINEAR}
    if min(numbers.values()) < 1 or numbers["every"] < 2 or numbers["value_heads"] % numbers["key_heads"]:
        raise ValueError(f"These are not the numbers of linear-attention layers: {numbers}.")
    if n_layers is not None and n_layers < numbers["every"]:
        raise ValueError(f"A model of {n_layers} layers has no full-attention layer where every {numbers['every']}th is one.")
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


def kinds_form(kinds):
    """The rows of a file that are stored in a kind of their own, FORM's "kinds", as {the row's name: the kind's (a
    dtype's or a block's), or a tuple of them, one for each layer of its stack}, from a dict of Python or of
    JavaScript (a list for the tuple): a file whose tensors are of several kinds (T.363) says here which, since its
    size cannot. None where there are none."""
    kinds = kinds.to_py() if hasattr(kinds, "to_py") else kinds
    named = lambda kind: kind_named(kind) if isinstance(kind, str) else tuple(kind_named(one) for one in kind)
    return {str(name): named(kind) for name, kind in kinds.items()} if kinds else None


def layer_list(layers, n_layers):
    """Layers by their numbers, sorted and each once, from a list of Python's or of JavaScript's; a ValueError for one
    that a model of n_layers layers has not (n_layers None: any)."""
    layers = layers.to_py() if hasattr(layers, "to_py") else layers
    layers = tuple(sorted({int(layer) for layer in layers or ()}))
    if layers and (layers[0] < 0 or (n_layers is not None and layers[-1] >= n_layers)):
        raise ValueError(f"A model of {n_layers} layers has no layer {layers[0] if layers[0] < 0 else layers[-1]}.")
    return layers


def wide_form(wide, n_layers=None):
    """T.331: the layers whose heads are of a size of their own, FORM's "wide", as {"layers": which they are, in order,
    "head_dim": the size of their heads}, from a dict of Python or of JavaScript. None for a model without any (a
    Gemma 4's full-attention layers: transformers' global_head_dim)."""
    wide = wide.to_py() if hasattr(wide, "to_py") else wide
    if not wide:
        return None
    layers, head = layer_list(wide["layers"], n_layers), int(wide["head_dim"])
    if head < 2 or head % 2:
        raise ValueError(f"A head of {head} values is no head that RoPE can turn.")
    return {"layers": layers, "head_dim": head} if layers else None


def double_form(double, n_layers=None):
    """T.331: the layers whose FFN is twice as wide as hidden_dim, FORM's "double", in order, from a list of Python or
    of JavaScript. None for a model without any (transformers' use_double_wide_mlp: a Gemma 4 E2B's layers that read
    the keys and values of another)."""
    return layer_list(double, n_layers) or None


def shared_form(shared, n_layers=None):
    """T.330: the layers that read another's keys and values, FORM's "kv_shared", as {"layers": which they are, in order,
    "from": the layer each reads, in the same order}, from a dict of Python or of JavaScript. None for a model without
    any (a Gemma 4's last num_kv_shared_layers). A layer is read from that comes before it and reads its own: what the
    shapes of the two have to agree on is the Dims' to see."""
    shared = shared.to_py() if hasattr(shared, "to_py") else shared
    if not shared:
        return None
    pairs = sorted({int(layer): int(source) for layer, source in zip(shared["layers"], shared["from"])}.items())
    if len(pairs) != len(shared["layers"]) or len(shared["layers"]) != len(shared["from"]):
        raise ValueError("The layers that read another's keys and values are each said once, with the layer each reads.")
    layers = layer_list([layer for layer, _ in pairs], n_layers)
    if any(not 0 <= source < layer or source in layers for layer, source in pairs):
        raise ValueError("A layer reads the keys and values of a layer before it that has its own.")
    return {"layers": layers, "from": tuple(source for _, source in pairs)} if layers else None


class Stateful(NamedTuple):
    """A kind of layer that keeps a state in place of keys and values, as a layout has it (Layout's stateful): the
    rows of STATEFUL are its tensors. What such a layer computes is the forward pass's own (engine/model.py has
    NumPy's, by this name)."""
    # what a plan calls a layer of this kind (engine/plan.py's layer_facts()), and the key of its numbers in the form
    # (FORM's), which the Dims of a layout that has it holds under this name too
    name: str
    # (the form's value, the header's layers) -> the numbers as the layouts read them, None for None, and a ValueError
    # for what are no numbers of such layers, or of none a model of that many layers can have
    parse: Callable
    # (the numbers, the header's layers) -> for every layer, whether it is of this kind
    layers: Callable


GATED_DELTA = Stateful("linear", linear_form, lambda linear, n_layers: [(l + 1) % linear["every"] != 0 for l in range(n_layers)])
SHORT_CONVOLUTION = Stateful("convolution", convolution_form, lambda convolution, n_layers: [kind == "c" for kind in convolution["layers"]])


def slots_of(stateful):
    """For every layer: (whether it keeps a state in place of keys and values, its place among the layers of its
    kind), which is where its tensors are in the file's stacks, from whether each layer keeps a state: a model whose
    layers all attend has (False, l) for layer l."""
    slots, counts = [], [0, 0]
    for kind in stateful:
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
#   CODES       the codebooks of llama.cpp's blocks (T.363), bytes: of a file with rows of such kinds alone
#   ROWS        a table that is only ever read a row at a time and multiplies nothing (T.329: a Gemma 4's embedding of
#               every layer), a matrix to the file, which the NumPy forward keeps as it is stored
MATRIX, EMBEDDING, CLASSIFIER, POSITIONS, VECTOR, TABLE = "matrix", "embedding", "classifier", "positions", "vector", "table"
CODES, ROWS = "codes", "rows"
# Which layers the first axis of a row's shape stacks, in their order (None: one tensor for the model):
#   EVERY       every layer
#   ATTENDING   the layers that keep keys and values (all of them, but in a hybrid model)
#   STATEFUL    the layers that keep a state in their place: a Qwen3.5's linear-attention layers, an LFM2's convolutions
EVERY, ATTENDING, STATEFUL = "every", "attending", "stateful"
# T.331: of the layers that attend, those whose heads are of the size of the form's "wide" (WIDE) and the others
# (NARROW, every one that attends where the form says none); of every layer, those whose FFN is of the form's
# "double" (DOUBLE) and the others (SINGLE)
NARROW, WIDE, SINGLE, DOUBLE = "narrow", "wide", "single", "double"
# T.330: of NARROW and WIDE, the layers that compute their own keys and values (every one where the form's "kv_shared"
# says none): the stacks of k, v and the norm of k's heads
NARROW_KEPT, WIDE_KEPT = "narrow kept", "wide kept"


class Row(NamedTuple):
    """A tensor of the file. name: the attribute the engine holds it as (and its key in the plan forward.js gets).
    kind: how it is stored where that is not the file's way with a row of its role (the form's "kinds"), see
    kind_of(): a kind's name, or one for each layer of a stack (a tuple)."""
    name: str
    role: str
    shape: tuple
    per: str = None
    kind: str = None


class Dims:
    """The numbers the rows' shapes are made of: the header's 7 ints, and what the form (FORM) adds to them. form: the
    form whole, for what a layout alone reads of it."""

    def __init__(self, header, form=None):
        self.form = form = form_of(form)
        (self.dim, self.hidden_dim, self.n_layers, self.n_heads, self.n_kv_heads, vocab_size,
         self.seq_len) = (int(value) for value in header)
        # (a negative vocabulary in the header: the classifier is a tensor of its own)
        self.vocab_size, self.shared = abs(vocab_size), vocab_size > 0
        self.arch, self.layout = form["arch"], layout_of(form["arch"])
        # the size of a head where it is not dim / n_heads (T.124: Qwen3 0.6B has 16 heads of 128 in a dim of 1024);
        # q and the attention's output are n_heads * head_size wide
        self.head_size = int(form["head_dim"]) or self.dim // self.n_heads
        self.q_dim, self.kv_dim = self.n_heads * self.head_size, self.n_kv_heads * self.head_size
        # the layers that keep a state, where the layout has such: their numbers under the kind's name (linear,
        # convolution: None in a model of another kind, or of none), and the kind's name, None where every layer attends
        kind, numbers = self.layout.stateful, stateful_form(self.arch, form, self.n_layers)
        for name in stateful_kinds():
            setattr(self, name, numbers if kind is not None and name == kind.name else None)
        self.stateful_kind = kind and kind.name
        self.slots = slots_of(kind.layers(numbers, self.n_layers) if kind is not None else [False] * self.n_layers)
        self.kinds = kinds_form(form["kinds"])
        # T.331: the layers whose heads are of a size of their own, and those whose FFN is twice as wide (None: none)
        self.wide, self.double = wide_form(form["wide"], self.n_layers), double_form(form["double"], self.n_layers)
        if (self.wide or self.double) and not self.layout.wide:
            raise ValueError(f"The architecture {self.arch!r} has no layers of heads or of an FFN of another width.")
        self.wide_head = self.wide["head_dim"] if self.wide else 0
        # T.329: the width of every layer's input of its own table (0: none)
        self.per_layer = int(form["per_layer"] or 0)
        if self.per_layer < 0 or (self.per_layer and not self.layout.wide):
            raise ValueError(f"The architecture {self.arch!r} has no embedding of every layer of {self.per_layer}.")
        # T.330: the layers that read another's keys and values (None: none), of that one's size of heads and an attending
        # one
        # (sharing: shared is the classifier's, above)
        self.sharing = shared_form(form["kv_shared"], self.n_layers)
        if self.sharing:
            if not self.layout.wide:
                raise ValueError(f"The architecture {self.arch!r} has no layers that read another's keys and values.")
            attending = self.layers(ATTENDING)
            for layer, source in zip(self.sharing["layers"], self.sharing["from"]):
                if layer not in attending or source not in attending or self.head_of(layer) != self.head_of(source):
                    raise ValueError(f"Layer {layer} cannot read the keys and values of layer {source}.")

    def rows(self):
        """The tensors of the checkpoint, as rows in file order: its architecture's layout, each with the kind the
        form gives it where it gives one."""
        rows = self.layout.rows(self)
        if self.kinds is None:
            return rows
        unknown = sorted(set(self.kinds) - {row.name for row in rows})
        if unknown:
            raise ValueError(f"This model has no tensor called {unknown[0]!r} to store in a kind of its own.")
        rows = [row._replace(kind=self.kinds.get(row.name)) for row in rows]
        for row in rows:
            if isinstance(row.kind, tuple) and (row.per is None or len(row.kind) != row.shape[0]):
                raise ValueError(f"The tensor {row.name!r} is given {len(row.kind)} kinds, one for a layer, and it is "
                                 f"{'no stack of layers' if row.per is None else f'a stack of {row.shape[0]}'}.")
        # (T.363) the codebooks of llama.cpp's blocks, last, where a row is of such a kind
        blocks = any(kind in BLOCKS for row in rows for kind in kinds_of(row))
        return rows + [Row("codebooks", CODES, (CODEBOOKS,), kind=BYTES.name)] if blocks else rows

    def layers(self, per):
        """The layers a stack covers, in the order of the stack."""
        if per in (NARROW, WIDE):
            wide = self.wide["layers"] if self.wide else ()
            return [layer for layer in self.layers(ATTENDING) if (layer in wide) == (per == WIDE)]
        if per in (NARROW_KEPT, WIDE_KEPT):
            shared = self.sharing["layers"] if self.sharing else ()
            return [layer for layer in self.layers(NARROW if per == NARROW_KEPT else WIDE) if layer not in shared]
        if per in (SINGLE, DOUBLE):
            return [layer for layer in range(self.n_layers) if (layer in (self.double or ())) == (per == DOUBLE)]
        return [layer for layer, (stateful, _) in enumerate(self.slots)
                if per == EVERY or stateful == (per == STATEFUL)]

    def head_of(self, layer):
        """T.331: the size of a layer's heads: the form's "wide" one where the layer is of those."""
        return self.wide_head if self.wide and layer in self.wide["layers"] else self.head_size

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
    """The RMSNorm weights of q and k, of a head's size (Qwen3, T.124)."""
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
    if d.form["bias"]:
        rows += [d.stack("bq", VECTOR, ATTENDING, d.q_dim), d.stack("bk", VECTOR, ATTENDING, d.kv_dim),
                 d.stack("bv", VECTOR, ATTENDING, d.kv_dim)]
    return rows + head_norms(d) if d.form["qk_norm"] else rows


def qwen35(d):
    """A Qwen3.5 (T.229), its tensors stacked by the kind of the layer: those of the full-attention layers (q, its gate,
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
    """An LFM2 (T.260), stacked by the kind of the layer too: those of the attention layers (q, k, v, o, the norms of the
    heads of q and k), those of the convolution layers (the matrix in, whose 3 dim rows are B, C and what B multiplies;
    the taps, which are never quantized; the matrix out), and the FFN of every layer."""
    return [*embedding(d), d.stack("rms_att_weight", VECTOR, EVERY, d.dim), *attention(d), *head_norms(d),
            d.stack("win", MATRIX, STATEFUL, 3 * d.dim, d.dim), d.stack("conv", VECTOR, STATEFUL, d.convolution["taps"], d.dim),
            d.stack("wout", MATRIX, STATEFUL, d.dim, d.dim),
            *gated_ffn(d), Row("rms_final_weight", VECTOR, (d.dim,)), *rope_tables(d), *classifier(d)]


def gemma3(d):
    """A Gemma 3 (T.261): a Llama's tensors with the norms of the heads of q and k (a Qwen3's), and a second norm in each
    half of a layer, of what the attention and the FFN give before it joins the stream. One pair of RoPE tables, of the
    layers that attend over all the positions: those of its local layers the engine makes (local_layers())."""
    return [*embedding(d), d.stack("rms_att_weight", VECTOR, EVERY, d.dim), *attention(d), *head_norms(d),
            d.stack("rms_att_post_weight", VECTOR, EVERY, d.dim), *gated_ffn(d),
            d.stack("rms_ffn_post_weight", VECTOR, EVERY, d.dim),
            Row("rms_final_weight", VECTOR, (d.dim,)), *rope_tables(d), *classifier(d)]


def gemma4(d):
    """A Gemma 4 (T.331): a Gemma 3's tensors, but stacked by the width of the layer: the attention of the layers whose
    heads are of the form's "wide" size (q, k, v, o and the norms of their heads, each name with "_wide") after the
    others', and the FFN of the layers of the form's "double" (twice hidden_dim: each name with "_double") after the
    others'; and a number every layer multiplies the stream by at its end (layer_scale). Its RoPE tables are of the
    layers that attend over all the positions, which are the wide ones where there are such."""
    rows, head = [*embedding(d)], d.wide_head or d.head_size
    if d.per_layer:
        # T.329: the embedding of every layer (a row of it is P values for each layer, P its width), the projection of
        # the embedding's row into the same, and the norm of that projection (of P values, each layer's alike)
        width = d.n_layers * d.per_layer
        rows += [Row("per_layer_table", ROWS, (d.vocab_size, width)), Row("per_layer_projection", MATRIX, (width, d.dim)),
                 Row("per_layer_norm", VECTOR, (d.per_layer,))]
    rows.append(d.stack("rms_att_weight", VECTOR, EVERY, d.dim))
    # (T.330: k, v and the norm of k's heads of the layers that compute their own keys and values)
    for per, kept, suffix, size in ((NARROW, NARROW_KEPT, "", d.head_size), (WIDE, WIDE_KEPT, "_wide", d.wide_head)):
        if d.layers(per):
            q, kv = d.n_heads * size, d.n_kv_heads * size
            rows += [d.stack("wq" + suffix, MATRIX, per, q, d.dim), d.stack("wk" + suffix, MATRIX, kept, kv, d.dim),
                     d.stack("wv" + suffix, MATRIX, kept, kv, d.dim), d.stack("wo" + suffix, MATRIX, per, d.dim, q),
                     d.stack("q_norm" + suffix, VECTOR, per, size), d.stack("k_norm" + suffix, VECTOR, kept, size)]
    rows += [d.stack("rms_att_post_weight", VECTOR, EVERY, d.dim), d.stack("rms_ffn_weight", VECTOR, EVERY, d.dim)]
    for per, suffix, hidden in ((SINGLE, "", d.hidden_dim), (DOUBLE, "_double", 2 * d.hidden_dim)):
        if d.layers(per):
            rows += [d.stack("w1" + suffix, MATRIX, per, hidden, d.dim), d.stack("w2" + suffix, MATRIX, per, d.dim, hidden),
                     d.stack("w3" + suffix, MATRIX, per, hidden, d.dim)]
    rows.append(d.stack("rms_ffn_post_weight", VECTOR, EVERY, d.dim))
    if d.per_layer:
        # (T.329: what every layer makes of its input: a gate of its own, the projection back, and its norm)
        rows += [d.stack("ple_gate", MATRIX, EVERY, d.per_layer, d.dim), d.stack("ple_proj", MATRIX, EVERY, d.dim, d.per_layer),
                 d.stack("ple_post", VECTOR, EVERY, d.dim)]
    return rows + [d.stack("layer_scale", VECTOR, EVERY, 1),
                   Row("rms_final_weight", VECTOR, (d.dim,)), Row("freq_cis_real", TABLE, (d.seq_len, head // 2)),
                   Row("freq_cis_imag", TABLE, (d.seq_len, head // 2)), *classifier(d)]


def spark(d):
    """A Spark-X2.5 (T.342): a Llama's tensors and a gate of each head of the attention's output (g_proj: a value a head,
    whose sigmoid multiplies that head's output before wo), behind wo. q, k and v are one matrix in the original, which
    the converter cuts. Its RoPE tables are of the layers that attend over all the positions, which turn a part of
    each head; those of its local layers, which turn the whole of it, the engine makes (local_layers())."""
    return after(llama(d), "wo", d.stack("head_gate", MATRIX, ATTENDING, d.n_heads, d.dim))


def gpt2(d):
    """A GPT-2: LayerNorm (a weight and a bias), a bias after every projection, a learned table of positions instead
    of RoPE, and an FFN of two matrices instead of three (no gate). Same attention, of heads that fill dim.
    A GPT-NeoX is the same, except that it rotates part of each head (so it keeps the RoPE tables of the Llama layout
    in place of the table of positions)."""
    vector = lambda name, width=d.dim: d.stack(name, VECTOR, EVERY, width)
    q, k, v, o = attention(d, d.dim, d.dim)
    return [*embedding(d), *(rope_tables(d) if d.layout.rope else [Row("positions", POSITIONS, (d.seq_len, d.dim))]),
            vector("rms_att_weight"), vector("ln_att_bias"), q, k, v, vector("bq"), vector("bk"), vector("bv"),
            o, vector("bo"),
            vector("rms_ffn_weight"), vector("ln_ffn_bias"),
            d.stack("w1", MATRIX, EVERY, d.hidden_dim, d.dim), vector("b1", d.hidden_dim),
            d.stack("w2", MATRIX, EVERY, d.dim, d.hidden_dim), vector("b2"),
            Row("rms_final_weight", VECTOR, (d.dim,)), Row("ln_final_bias", VECTOR, (d.dim,)), *classifier(d)]


# ---- the architectures
# How much of a head RoPE turns (Layout's rope; None: none of it, the model has a learned table of positions)
# (T.331: PROPORTIONAL, a Gemma 4's: the first rotary values of a head, by the angles of the whole head's width where
# PARTLY's are of the width it turns)
WHOLE, PARTLY, PROPORTIONAL = "whole", "partly", "proportional"
# The activation of an FFN (Layout's activation): a SiLU, or a GELU (its tanh form, which GPT-2 calls gelu_new), or
# (T.342) a GELU of its exact form, by erf (transformers' "gelu": a Spark-X2.5's)
SILU, GELU, GELU_ERF = "silu", "gelu", "gelu_erf"


class Layout(NamedTuple):
    """An architecture: the layout of its file, and what of its forward pass is not the same in all of them, as
    facts. Whatever in the engine differs by the architecture reads one of these, and nothing compares the name: a
    new architecture is a record of LAYOUTS, the function of its rows where no other's are its own, and, where its
    layers are of a new kind, that kind (a Stateful, and its steps in the forward passes). One that is another but
    for a fact is that one's record with the fact replaced (LAYOUTS["llama"]._replace(...)).
    What the rows already say is no fact here: a bias is added where the file has one (bq, bo), the heads of q and k
    are normalized where it has those norms, a position's row is added where it has the table, and what the attention
    and the FFN give is normalized where it has the norms for that (rms_att_post_weight, rms_ffn_post_weight)."""
    # (the Dims) -> the tensors of the file as rows, in file order
    rows: Callable
    # the kind of its layers that keep a state in place of keys and values, where it has such layers
    stateful: Optional[Stateful] = None
    # how much of each head of q and k RoPE turns: WHOLE, PARTLY or PROPORTIONAL (the options say how many values:
    # "rotary"; T.331: a local layer turns its whole head all the same), or None
    rope: Optional[str] = WHOLE
    # yarn's magnitude is in the RoPE tables the engine makes for a file that has none (rope_magnitude())
    yarn_magnitude: bool = False
    # its norms are LayerNorms, of a weight and a bias (RMSNorms, of a weight, otherwise)
    layer_norm: bool = False
    # its FFN is gated: three matrices, w2(act(w1 x) * w3 x) (otherwise GPT-2's: two and their biases, w2 act(w1 x + b1) + b2)
    gated_ffn: bool = True
    # its matrices may be in a rotated basis (T.237)
    rotatable: bool = False
    # some of its layers may be ones RoPE leaves alone (T.255)
    unturned: bool = False
    # the activation of its FFN, SILU, GELU or GELU_ERF: of the gate of a gated one (a Llama's is a SiLU, T.261: a
    # Gemma's a GELU, T.342: a Spark-X2.5's the exact GELU), of the one matrix's output of GPT-2's. Another fact than
    # gated_ffn: a Gemma's FFN is gated as a Llama's is
    activation: str = SILU
    # some of its layers may be local ones (T.261): they attend a window of the last positions, and RoPE turns their q
    # and k by a base of their own (local_layers())
    local: bool = False
    # T.331: what its attention reads of v is normalized head by head (an RMSNorm without a weight)
    value_norm: bool = False
    # T.331: some of its layers may have heads of another size, or an FFN twice as wide (the form's "wide", "double")
    wide: bool = False


# by the architecture's name (FORM's "arch")
LAYOUTS = {
    "llama": Layout(llama, yarn_magnitude=True, rotatable=True, unturned=True),
    "gpt2": Layout(gpt2, rope=None, layer_norm=True, gated_ffn=False, activation=GELU),
    "neox": Layout(gpt2, rope=PARTLY, layer_norm=True, gated_ffn=False, activation=GELU),
    "qwen35": Layout(qwen35, stateful=GATED_DELTA, rope=PARTLY, rotatable=True),
    "lfm2": Layout(lfm2, stateful=SHORT_CONVOLUTION),
    "gemma3": Layout(gemma3, activation=GELU, local=True),
    "gemma4": Layout(gemma4, rope=PROPORTIONAL, activation=GELU, local=True, value_norm=True, wide=True),
    # T.342: local layers that turn their whole heads, and the others a quarter of theirs (a Gemma 3's local layers and a
    # GPT-NeoX's RoPE), a gate of each head's output and the exact GELU
    "spark": Layout(spark, rope=PARTLY, activation=GELU_ERF, local=True),
}


def layout_of(arch):
    """The architecture of this name (FORM's "arch"), and a ValueError for a name that is none's: a mistyped one was
    read as a Llama until T.359.6, and ran as another model without a word."""
    layout = LAYOUTS.get(arch) if isinstance(arch, str) else None
    if layout is None:
        raise ValueError(f"There is no architecture called {arch!r}: {', '.join(LAYOUTS)}.")
    return layout


def unturned_layers(arch, n_layers, layers):
    """The layers RoPE leaves alone (T.255) of a model of this architecture with n_layers layers, in order, from what
    its options say of them (a list of Python's or of JavaScript's, or nothing): a ValueError for an architecture
    that has no such layers, or for a layer the model has not."""
    layers = layers.to_py() if hasattr(layers, "to_py") else layers
    layers = tuple(sorted({int(layer) for layer in layers or ()}))
    if layers and (not layout_of(arch).unturned or not 0 <= layers[0] <= layers[-1] < n_layers):
        raise ValueError("The layers RoPE leaves alone are layers of a Llama, and none of another architecture.")
    return layers


# (the keys of a model's local layers, as its options say them)
LOCAL = ("layers", "window", "rope_theta")


def local_layers(arch, n_layers, local):
    """The local layers (T.261) of a model of this architecture with n_layers layers, from what its options say of them
    (a dict of Python's or of JavaScript's, or nothing): {"layers": which they are, in order, "window": how many of the
    last positions each attends, its own among them, "rope_theta": the base RoPE turns their q and k by}; None for a
    model without any. A Gemma 3's "sliding_attention" layers: five of every six, a window of 512 and a base of 1e4
    where the others' is 1e6. The file is the same whichever layers they are, so the options say it, as they say the
    layers RoPE leaves alone. A ValueError for an architecture that has no such layers, for a layer the model has not,
    and for what is no window or no base."""
    local = local.to_py() if hasattr(local, "to_py") else local
    if not local:
        return None
    layers = tuple(sorted({int(layer) for layer in local["layers"]}))
    window, theta = int(local["window"]), float(local["rope_theta"])
    if not layout_of(arch).local or (layers and not 0 <= layers[0] <= layers[-1] < n_layers) or window < 1 \
            or not (math.isfinite(theta) and theta > 0):
        raise ValueError("The local layers are layers of a Gemma 3 with a window and a base of RoPE, and none of "
                         "another architecture.")
    return {"layers": layers, "window": window, "rope_theta": theta} if layers else None


def stateful_kinds():
    """The kinds of layers that keep a state, of all the architectures, by name."""
    return {layout.stateful.name: layout.stateful for layout in LAYOUTS.values() if layout.stateful is not None}


def stateful_form(arch, form, n_layers):
    """The numbers of the layers that keep a state of a model of this architecture and form (FORM, as form_of() gives
    it) with n_layers layers, as the kind's parse() reads them; None for an architecture whose layers all attend.
    The one place that holds a form to its architecture: a ValueError where the form does not say the layers the
    architecture has, says layers of a kind it has not, or says what are no numbers of them."""
    kind = layout_of(arch).stateful
    said = [name for name in stateful_kinds() if form[name] is not None]
    if said != ([kind.name] if kind is not None else []):
        raise ValueError(f"The architecture {arch!r} and the layers of its form go together: it has "
                         f"{kind.name if kind is not None else 'none'} that keep a state, and the form says "
                         f"{', '.join(said) or 'none'}.")
    return kind.parse(form[kind.name], n_layers) if kind is not None else None


def tensor_rows(header, form=None):
    """The tensors of a checkpoint with this header (7 ints) and form (FORM), as rows in file order."""
    return Dims(header, form).rows()


# ---- how a row is stored, and where
def kinds_of(row):
    """The kinds a form gave a row, as a tuple: none, the one, or its layers'."""
    return row.kind if isinstance(row.kind, tuple) else () if row.kind is None else (row.kind,)


def kind_of(row, dtype):
    """How a row is stored in a file of this dtype, by a name of KINDS; None: it is left out. The row's own kind
    where the form gave it one (FORM's "kinds"; a tuple, one for each layer of a stack). Otherwise it follows the file's dtype by its role: every matrix is
    the file's, the vectors of a quantized file are float32 and it has no RoPE tables. This is the one place that
    chooses: whatever writes, reads or sizes a row asks for its kind here and goes by that, not by the file's dtype, so
    that a file whose tensors are of several kinds (T.363) is the form's to say and nobody else's to know."""
    if row.kind is not None:
        return row.kind
    if dtype not in QUANTIZED:
        return dtype
    return "float32" if row.role == VECTOR else None if row.role == TABLE else dtype


class Place(NamedTuple):
    """Where a row is in the file: its kind (kind_of(); None: nowhere, and size is 0), the offset of its values and
    all its bytes, and of a kind in groups the values of a group and the offset of the scales (0 and 0 otherwise).
    layers: of a stack with a kind for each layer (kind is the tuple), each layer's Place, one after the other.
    parts: the places of one kind each, where bytes are read and written: the layers', or this one."""
    row: Row
    kind: str
    offset: int
    size: int
    group: int = 0
    scales: int = 0
    layers: tuple = ()

    @property
    def parts(self):
        return self.layers or (self,)


def placed(rows, dtype, offset=28):
    """A Place for every row, in file order: the header's 28 bytes, then one tensor after the other."""
    places = []
    for row in rows:
        kind, count = kind_of(row, dtype), math.prod(row.shape)
        if kind is None:
            place = Place(row, None, offset, 0)
        elif isinstance(kind, tuple):
            layers = tuple(placed([Row(row.name, row.role, row.shape[1:], kind=one) for one in kind], dtype, offset))
            place = Place(row, kind, offset, sum(layer.size for layer in layers), layers=layers)
        elif KINDS[kind].group:
            group = KINDS[kind].group(row.shape[-1])
            values, scales = KINDS[kind].stored_bytes(count, group)
            place = Place(row, kind, offset, values + scales, group, offset + values)
        else:
            place = Place(row, kind, offset, count * KINDS[kind].itemsize)
        places.append(place)
        offset += place.size
    return places


def file_size(rows, dtype):
    """The bytes of a checkpoint of these rows and this dtype."""
    return 28 + sum(place.size for place in placed(rows, dtype))


def unsuited(rows, dtype):
    """The kinds of a file of this dtype that cannot hold their rows: a packed kind needs rows of whole groups."""
    return [part.kind for place in placed(rows, dtype) if place.kind for part in place.parts
            if not KINDS[part.kind].suits(part.row.shape[-1])]


def suited(rows, dtype):
    """Whether a file of this dtype can hold these rows."""
    return not unsuited(rows, dtype)


def check_suited(rows, dtype):
    for kind in unsuited(rows, dtype):
        raise ValueError(KINDS[kind].unsuited)
