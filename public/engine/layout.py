# What a model's file holds: the tensors of a checkpoint as rows (name, role, shape, which layers a stack covers), for
# each of the layouts, in the order of the file; the form a file does not say (FORM); how each row is stored in a file
# of a dtype, and where. Everything that needs the order of the tensors reads it here (T359): the converter's layout()
# and Writer, checkpoint_dtype(), the engine's attributes, external_tensors(), and conversion_plan() by the rows' names.
# What a kind of storage is (its bytes, its groups) is engine/dtypes.py's, which this file reads, and nothing else of
# the engine or of the converter: both read it.
# The architectures are one table here too (LAYOUTS, T359.6): what the file of each holds and what its forward pass is,
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

from engine.dtypes import DTYPES, QUANTIZED, dtype_of

# ---- the form
# The form of a checkpoint: what sets its tensors and sizes its forward pass besides the 7 ints of the header, which
# the legacy file cannot say (see Llama.__init__), with the value of a file that says nothing. The converter writes
# them into the options, and one dict of these names goes to everything that lays the file out or sizes it
# (tensor_rows() below and what reads it, forward.js's footprint()), so that another one is added where it is used,
# not along the way (T144).
# arch: the architecture, a name of LAYOUTS (below).
# linear (T229): the linear-attention layers of arch "qwen35", see linear_form(); None where there are none.
# rotated (T237): a rotated basis, which moves no tensor: the same ones are stored in another basis.
# convolution (T260): the convolution layers of arch "lfm2", see convolution_form(); None where there are none.
# (linear and convolution are the kinds of layers that keep a state: each such kind has its numbers under its own
# name here, Stateful below, and stateful_form() holds them to the architecture)
# kinds (T359): the rows that are not stored the way the file's dtype stores a row of their role, see kinds_form();
# None where every row is, which is every file a conversion writes today.
FORM = {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0, "linear": None, "rotated": None,
        "convolution": None, "kinds": None}


def form_of(options=None):
    """The form (FORM's keys, each with its default where options has none) out of options, a dict with those and
    any others (the options of a model, a manifest's). Dicts from JavaScript are read too (a JsProxy). A key given as
    None (a JSON null, or JavaScript's undefined) says nothing, as head_size() reads "head_dim": null: it had
    checkpoint_dtype() fail on int(None) while footprint() counted dim / heads (the review of T144)."""
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
    """The rows of a file that are stored in a kind of their own, FORM's "kinds", as {the row's name: the kind's
    (DTYPES')}, from a dict of Python or of JavaScript: a file whose tensors are of several kinds (T363) says here
    which, since its size cannot. None where there are none."""
    kinds = kinds.to_py() if hasattr(kinds, "to_py") else kinds
    return {str(name): dtype_of(kind) for name, kind in kinds.items()} if kinds else None


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


def layer_slots(n_layers, linear, convolution=None):
    """slots_of() for the numbers of a Qwen3.5's linear-attention layers or of an LFM2's convolution layers (None and
    None: every layer attends), for whoever has those two and no Dims (a Dims has its slots)."""
    kind, numbers = (SHORT_CONVOLUTION, convolution) if convolution is not None else (GATED_DELTA, linear)
    return slots_of(kind.layers(numbers, n_layers) if numbers is not None else [False] * n_layers)


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
    """A tensor of the file. name: the attribute the engine holds it as (and its key in the plan forward.js gets).
    kind: how it is stored where that is not the file's way with a row of its role (the form's "kinds"), see
    kind_of()."""
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
        # the size of a head where it is not dim / n_heads (T124: Qwen3 0.6B has 16 heads of 128 in a dim of 1024);
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

    def rows(self):
        """The tensors of the checkpoint, as rows in file order: its architecture's layout, each with the kind the
        form gives it where it gives one."""
        rows = self.layout.rows(self)
        if self.kinds is None:
            return rows
        unknown = sorted(set(self.kinds) - {row.name for row in rows})
        if unknown:
            raise ValueError(f"This model has no tensor called {unknown[0]!r} to store in a kind of its own.")
        return [row._replace(kind=self.kinds.get(row.name)) for row in rows]

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
    if d.form["bias"]:
        rows += [d.stack("bq", VECTOR, ATTENDING, d.q_dim), d.stack("bk", VECTOR, ATTENDING, d.kv_dim),
                 d.stack("bv", VECTOR, ATTENDING, d.kv_dim)]
    return rows + head_norms(d) if d.form["qk_norm"] else rows


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
    return [*embedding(d), *(rope_tables(d) if d.layout.rope else [Row("positions", POSITIONS, (d.seq_len, d.dim))]),
            vector("rms_att_weight"), vector("ln_att_bias"), q, k, v, vector("bq"), vector("bk"), vector("bv"),
            o, vector("bo"),
            vector("rms_ffn_weight"), vector("ln_ffn_bias"),
            d.stack("w1", MATRIX, EVERY, d.hidden_dim, d.dim), vector("b1", d.hidden_dim),
            d.stack("w2", MATRIX, EVERY, d.dim, d.hidden_dim), vector("b2"),
            Row("rms_final_weight", VECTOR, (d.dim,)), Row("ln_final_bias", VECTOR, (d.dim,)), *classifier(d)]


# ---- the architectures
# How much of a head RoPE turns (Layout's rope; None: none of it, the model has a learned table of positions)
WHOLE, PARTLY = "whole", "partly"


class Layout(NamedTuple):
    """An architecture: the layout of its file, and what of its forward pass is not the same in all of them, as
    facts. Whatever in the engine differs by the architecture reads one of these, and nothing compares the name: a
    new architecture is a record of LAYOUTS, the function of its rows where no other's are its own, and, where its
    layers are of a new kind, that kind (a Stateful, and its steps in the forward passes). One that is another but
    for a fact is that one's record with the fact replaced (LAYOUTS["llama"]._replace(...)).
    What the rows already say is no fact here: a bias is added where the file has one (bq, bo), the heads of q and k
    are normalized where it has those norms, a position's row is added where it has the table."""
    # (the Dims) -> the tensors of the file as rows, in file order
    rows: Callable
    # the kind of its layers that keep a state in place of keys and values, where it has such layers
    stateful: Optional[Stateful] = None
    # how much of each head of q and k RoPE turns: WHOLE, PARTLY (the options say how many values: "rotary"), or None
    rope: Optional[str] = WHOLE
    # yarn's magnitude is in the RoPE tables the engine makes for a file that has none (rope_magnitude())
    yarn_magnitude: bool = False
    # its norms are LayerNorms, of a weight and a bias (RMSNorms, of a weight, otherwise)
    layer_norm: bool = False
    # its FFN is gated: three matrices, w2(silu(w1 x) * w3 x) (otherwise GPT-2's: two and their biases, with a GELU)
    gated_ffn: bool = True
    # its matrices may be in a rotated basis (T237)
    rotatable: bool = False
    # some of its layers may be ones RoPE leaves alone (T255)
    unturned: bool = False


# by the architecture's name (FORM's "arch")
LAYOUTS = {
    "llama": Layout(llama, yarn_magnitude=True, rotatable=True, unturned=True),
    "gpt2": Layout(gpt2, rope=None, layer_norm=True, gated_ffn=False),
    "neox": Layout(gpt2, rope=PARTLY, layer_norm=True, gated_ffn=False),
    "qwen35": Layout(qwen35, stateful=GATED_DELTA, rope=PARTLY, rotatable=True),
    "lfm2": Layout(lfm2, stateful=SHORT_CONVOLUTION),
}


def layout_of(arch):
    """The architecture of this name (FORM's "arch"), and a ValueError for a name that is none's: a mistyped one was
    read as a Llama until T359.6, and ran as another model without a word."""
    layout = LAYOUTS.get(arch) if isinstance(arch, str) else None
    if layout is None:
        raise ValueError(f"There is no architecture called {arch!r}: {', '.join(LAYOUTS)}.")
    return layout


def unturned_layers(arch, n_layers, layers):
    """The layers RoPE leaves alone (T255) of a model of this architecture with n_layers layers, in order, from what
    its options say of them (a list of Python's or of JavaScript's, or nothing): a ValueError for an architecture
    that has no such layers, or for a layer the model has not."""
    layers = layers.to_py() if hasattr(layers, "to_py") else layers
    layers = tuple(sorted({int(layer) for layer in layers or ()}))
    if layers and (not layout_of(arch).unturned or not 0 <= layers[0] <= layers[-1] < n_layers):
        raise ValueError("The layers RoPE leaves alone are layers of a Llama, and none of another architecture.")
    return layers


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
def kind_of(row, dtype):
    """How a row is stored in a file of this dtype, by a name of DTYPES; None: it is left out. The row's own kind
    where the form gave it one (FORM's "kinds"). Otherwise it follows the file's dtype by its role: every matrix is
    the file's, the vectors of a quantized file are float32 and it has no RoPE tables. This is the one place that
    chooses: whatever writes, reads or sizes a row asks for its kind here and goes by that, not by the file's dtype, so
    that a file whose tensors are of several kinds (T363) is the form's to say and nobody else's to know."""
    if row.kind is not None:
        return row.kind
    if dtype not in QUANTIZED:
        return dtype
    return "float32" if row.role == VECTOR else None if row.role == TABLE else dtype


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
        if kind is None:
            place = Place(row, None, offset, 0)
        elif DTYPES[kind].bits:
            group = DTYPES[kind].group(row.shape[-1])
            values, scales = DTYPES[kind].stored_bytes(count, group)
            place = Place(row, kind, offset, values + scales, group, offset + values)
        else:
            place = Place(row, kind, offset, count * DTYPES[kind].itemsize)
        places.append(place)
        offset += place.size
    return places


def file_size(rows, dtype):
    """The bytes of a checkpoint of these rows and this dtype."""
    return 28 + sum(place.size for place in placed(rows, dtype))


def unsuited(rows, dtype):
    """The kinds of a file of this dtype that cannot hold their rows: a packed kind needs rows of whole groups."""
    return [place.kind for place in placed(rows, dtype) if place.kind and not DTYPES[place.kind].suits(place.row.shape[-1])]


def suited(rows, dtype):
    """Whether a file of this dtype can hold these rows."""
    return not unsuited(rows, dtype)


def check_suited(rows, dtype):
    for kind in unsuited(rows, dtype):
        raise ValueError(DTYPES[kind].unsuited)
