# The plan Python hands public/forward.js (T93, T359.5): what a model is to an engine outside Python, as one dict made in
# one place from the facts of the file (its header and form: engine/layout.py's Dims and rows), its dtype and how it is
# to be run. No model is needed to make one, and nothing here reads one: Llama gathers what it has and calls
# forward_plan(), and ExternalForward (engine/external.py) is handed the result.
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

from typing import Any, NamedTuple

import numpy as np

from engine.checkpoint import OUTLIER_CHANNELS, SEVERAL_KINDS, outlier_channels, outside
from engine.layers import RMS_EPS, rotated_widths
from engine.layout import TABLE, linear_widths, unturned_layers

# the kind of a layer that attends over all the positions (and keeps keys and values); a layer that keeps a state in
# their place is of the kind its architecture has (engine/layout.py's Stateful, by its name: "linear", "convolution")
ATTENTION = "attention"


class Settings(NamedTuple):
    """What a plan is made of besides the file (forward_plan()), as one value: a setting that is added is a field
    here and a line where it is read, and no argument of the functions in between.
    How the model is run: int8 (the matrices stay int8: the kernels compute on them), kv_start (the positions the keys
    and values begin with room for) and disable (T52's switches, of which "relaxed" and "kv16" are the plan's).
    What the file cannot say, the model's options: rotary (how many values of a head RoPE turns; 0: all),
    parallel_residual, rms_norm_eps, unturned (T255: the layers RoPE leaves alone), rotated (T237: rotated_form()'s
    {"block", "signs"}, None for a model in its own basis) and tables (the RoPE tables (cos, sin) of a file that has
    none of its own: rope_tables()'s; None where the file has them)."""
    int8: bool
    kv_start: int
    disable: tuple = ()
    rotary: int = 0
    parallel_residual: bool = False
    rms_norm_eps: float = RMS_EPS
    unturned: tuple = ()
    rotated: Any = None
    tables: Any = None


def layer_facts(dims, settings):
    """For every layer, what an engine has to know of it: {"kind": ATTENTION, or the layout's kind of a layer that
    keeps a state, "place": its place among the layers of its kind, which is where its tensors are in the file's stacks
    and its keys and values or its state in the engine, "rope": whether RoPE turns its q and k}. All of it is the
    layout's: the slots and the kind are the Dims', a model has RoPE where its file has (or leaves out) the tables,
    and only a layer that attends is turned, but the layers the settings say RoPE leaves alone (T255: the Dims refuses
    those that are none of this model's).
    A fact of a layer that an engine needs goes here, as another key of every layer's dict (and where it is a model's
    option, as another field of Settings)."""
    unturned = unturned_layers(dims.arch, dims.n_layers, settings.unturned)
    turning = any(row.role == TABLE for row in dims.rows())
    return [{"kind": dims.stateful_kind if state else ATTENTION, "place": place,
             "rope": turning and not state and layer not in unturned}
            for layer, (state, place) in enumerate(dims.slots)]


def plan_widths(dims):
    """The widths an engine sizes its buffers by, which are not in the header: "q" (q and the attention's output: the
    heads together), "kv" (k and v), "linear" (of a linear-attention layer: "mixed", what the convolution runs over;
    "keys", q or k; "read", v, which the output matrix reads; None where there are no such layers) and "rotated" (the
    widths of what the model's matrices read, each of which a rotated basis has its signs for, T237)."""
    linear = dims.linear and dict(zip(("mixed", "keys", "read"), linear_widths(dims.linear)))
    return {"q": dims.q_dim, "kv": dims.kv_dim, "linear": linear,
            "rotated": rotated_widths(dims.dim, dims.q_dim, dims.hidden_dim, dims.linear)}


def forward_plan(dims, dtype, read, settings=None, **said):
    """The plan of a model for an engine outside Python (forward.js's createForward()), a dict.

    What the file is: dims (engine/layout.py's Dims: the header and the form), dtype (a name of DTYPES), and read(offset,
    length), the bytes of the file, of which the final norm's weight is looked at for its outlier channels (T92).
    Everything else, the model's options and how it is run: settings (a Settings), or its fields by their names.

    The keys: the header's numbers; "tensors", where every row of the file is ({name: Tensor.plan()}, the classifier of
    a model without one of its own under both names); "derived", the bytes of the few arrays Python computes (the RoPE
    tables, the signs of a rotated basis); "outliers"; the form's "linear" and "convolution"; and, from T359.5,
    "layers" (layer_facts()) and "widths" (plan_widths()), so that an engine need not work out again which layer is of
    which kind (forward.js reads them from T375)."""
    settings = Settings(**said) if settings is None else settings
    int8, disable, rotated, tables = settings.int8, settings.disable, settings.rotated, settings.tables
    if dims.kinds is not None:
        raise ValueError(SEVERAL_KINDS)
    layers = layer_facts(dims, settings)
    held = outside(dims.rows(), dtype)
    # (the classifier of a model that has no other is its embedding, under both names)
    held.setdefault("wcls", held["token_embedding_table"])
    tensors = {name: tensor.plan() for name, tensor in held.items()}
    derived = {}
    if tables is not None:
        derived = {name: np.ascontiguousarray(table, dtype=np.float32).tobytes()
                   for name, table in zip(("freq_cis_real", "freq_cis_imag"), tables)}
    if rotated is not None:
        # T237: the signs of every width, with the transform's 1 / sqrt(block) in them (what the kernel multiplies by)
        scale = np.float32(1.0 / math.sqrt(rotated["block"]))
        derived.update({f"signs.{width}": (signs * scale).tobytes() for width, signs in rotated["signs"].items()})
    channels = []
    # (T237: not in a rotated basis, where the classifier reads R of its input: the rotation spreads a channel
    # over its block, and a column of the stored matrix is no channel's)
    if int8 and rotated is None:
        raw = read(held["rms_final_weight"].offset, dims.dim * 4)
        weight = np.frombuffer(bytes(raw.to_py() if hasattr(raw, "to_py") else raw), dtype=np.float32)
        channels = [int(c) for c in outlier_channels(weight, min(OUTLIER_CHANNELS, dims.dim))]
    return {"arch": dims.arch, "dim": dims.dim, "hidden_dim": dims.hidden_dim, "n_layers": dims.n_layers,
            "n_heads": dims.n_heads, "n_kv_heads": dims.n_kv_heads, "head_size": dims.head_size,
            "vocab_size": dims.vocab_size, "seq_len": dims.seq_len, "rotary": int(settings.rotary) or dims.head_size,
            "parallel_residual": bool(settings.parallel_residual), "kv_start": settings.kv_start,
            "rms_norm_eps": settings.rms_norm_eps,
            "shared_classifier": dims.shared, "int8": bool(int8),
            "relaxed": "relaxed" not in disable, "tensors": tensors, "derived": derived, "outliers": channels,
            # T229: a Qwen3.5's linear-attention layers (None: none)
            "linear": dims.linear,
            # T255: the layers RoPE leaves alone
            "unturned": list(unturned_layers(dims.arch, dims.n_layers, settings.unturned)),
            # T260: an LFM2's convolution layers (None: none)
            "convolution": dims.convolution,
            # T237: the block of a rotated basis (0: the model's own basis); its signs are in derived
            "rotated": rotated["block"] if rotated else 0,
            # T110: the keys and values of an int8 model may be float16 (forward.js uses that on a shared memory)
            "half_kv": bool(int8) and "kv16" not in disable,
            # T359.5: every layer's kind, its place among its kind and whether RoPE turns it; and the widths
            "layers": layers, "widths": plan_widths(dims)}
