"""engine/plan.py (T359.5): the plan Python hands forward.js, made from a header, a form, a dtype and how the model is
run, with no model and no weights: every test here but the last makes no Llama. The last holds Llama to handing
ExternalForward what forward_plan() makes of the same facts. (That the plan's keys are the ones forward.js reads is
tests/plan-keys-check.mjs's, and that their values are what they were is the net's kind "layouts".)"""
import math
import struct

import numpy as np
import pytest

import engine.model
from engine.layout import STATEFUL_KINDS
from engine.plan import ATTENTION
from engine.tensors import rope_tables
from llama2_numpy import (ATTENDING, EVERY, OUTLIER_CHANNELS, QUANTIZED, SEVERAL_KINDS, STATEFUL, TABLE, Dims, Llama,
                          external_tensors, file_size, forward_plan, layer_facts, linear_widths, outlier_channels,
                          plan_widths, rope_frequencies, rotated_form, rotated_widths, sign_bits, tensor_rows)

LINEAR = {"every": 4, "key_heads": 2, "value_heads": 4, "key_dim": 64, "value_dim": 128, "conv": 4}
# (name, header, form, the model's options): every dimension of a header differs from the others, and every row is
# whole groups of 128, so that a file of each of the five dtypes can hold it
MODELS = [
    ("llama", [256, 768, 4, 8, 4, 2048, 320], {}, {}),
    ("llama, its own classifier", [384, 1024, 6, 12, 4, -2304, 288], {}, {}),
    ("qwen2", [256, 768, 4, 8, 4, 2048, 320], {"bias": True}, {}),
    ("qwen3", [256, 768, 4, 4, 2, 2048, 320], {"qk_norm": True, "head_dim": 128}, {"rms_norm_eps": 1e-6}),
    ("qwen2 and qwen3 at once", [256, 768, 4, 4, 2, -2048, 320], {"bias": True, "qk_norm": True, "head_dim": 128}, {}),
    ("smollm3", [256, 768, 5, 8, 4, 2048, 320], {}, {"unturned": (1, 4)}),
    ("gpt2", [256, 1024, 3, 8, 8, -2304, 320], {"arch": "gpt2"}, {}),
    ("neox", [256, 1024, 3, 8, 8, 2048, 320], {"arch": "neox"}, {"rotary": 8, "parallel_residual": True}),
    ("qwen35", [256, 768, 8, 4, 2, -2048, 320], {"arch": "qwen35", "head_dim": 128, "linear": LINEAR}, {"rotary": 32}),
    ("qwen35, every second", [256, 768, 5, 4, 2, 2048, 320],
     {"arch": "qwen35", "head_dim": 128, "linear": {**LINEAR, "every": 2}}, {"rotary": 32}),
    ("lfm2", [256, 768, 8, 8, 4, 2048, 320], {"arch": "lfm2", "qk_norm": True, "convolution": {"layers": "ccaccaca", "taps": 3}}, {}),
]
DTYPES = ("float32", "float16", "int8", "int6", "ternary")
cases = pytest.mark.parametrize("name, header, form, more", MODELS, ids=[model[0] for model in MODELS])
# the keys forward.js is handed: the 24 it reads today, and the two of T359.5
KEYS = {"arch", "dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "head_size", "vocab_size", "seq_len", "rotary",
        "parallel_residual", "kv_start", "rms_norm_eps", "shared_classifier", "int8", "relaxed", "tensors", "derived",
        "outliers", "linear", "unturned", "convolution", "rotated", "half_kv", "layers", "widths"}


def no_bytes(offset, length):
    raise AssertionError("nothing of the file is read for this plan")


def flat(offset, length):
    """A final norm's weight with no outliers."""
    return np.ones(length // 4, dtype=np.float32).tobytes()


@cases
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_plan_is_made_of_a_header_a_form_and_a_dtype(name, header, form, more, dtype):
    dims = Dims(header, form)
    plan = forward_plan(dims, dtype, no_bytes, int8=False, kv_start=64, **more)
    assert set(plan) == KEYS
    # the header's numbers, and what the form adds to them
    dim, hidden, layers, heads, kv_heads, vocab, seq_len = header
    head = form.get("head_dim") or dim // heads
    assert [plan[key] for key in ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len")] == \
        [dim, hidden, layers, heads, kv_heads, abs(vocab), seq_len]
    assert plan["arch"] == form.get("arch", "llama") and plan["head_size"] == head
    assert plan["shared_classifier"] is (vocab > 0)
    assert plan["linear"] == form.get("linear") and plan["convolution"] == form.get("convolution")
    # where every tensor is: what the worker is told before the bytes are there (T156), the classifier under both names
    assert plan["tensors"] == external_tensors(header, dtype, form)
    assert list(plan["tensors"]) == list(external_tensors(header, dtype, form))
    assert (plan["tensors"]["wcls"] == plan["tensors"]["token_embedding_table"]) is (vocab > 0)
    tables = [row.name for row in tensor_rows(header, form) if row.role == TABLE]
    assert all((table in plan["tensors"]) is (dtype == "float32") for table in tables)
    # the model's options, as they were given or as a model without them has them
    assert plan["rotary"] == more.get("rotary", head) and plan["unturned"] == list(more.get("unturned", ()))
    assert plan["parallel_residual"] is more.get("parallel_residual", False)
    assert plan["rms_norm_eps"] == more.get("rms_norm_eps", 1e-5)
    # how it is run
    assert plan["kv_start"] == 64 and plan["int8"] is False and plan["half_kv"] is False and plan["relaxed"] is True
    assert plan["derived"] == {} and plan["outliers"] == [] and plan["rotated"] == 0


@cases
def test_the_layers_of_a_plan_are_the_layouts(name, header, form, more):
    dims, rows = Dims(header, form), tensor_rows(header, form)
    layers = forward_plan(dims, "int8", no_bytes, int8=False, kv_start=64, **more)["layers"]
    assert layers == layer_facts(dims, more.get("unturned", ())) and len(layers) == header[2]
    assert all(set(layer) == {"kind", "place", "rope"} for layer in layers)
    # a layer's kind is the stack its tensors are in, and its place is its place in that stack
    attending, stateful = dims.layers(ATTENDING), dims.layers(STATEFUL)
    assert [l for l, layer in enumerate(layers) if layer["kind"] == ATTENTION] == attending
    assert [layers[l]["place"] for l in attending] == list(range(len(attending)))
    assert [layers[l]["place"] for l in stateful] == list(range(len(stateful)))
    for row in rows:
        if row.per in (ATTENDING, STATEFUL):
            kinds = {layer["kind"] for layer in layers if (layer["kind"] == ATTENTION) == (row.per == ATTENDING)}
            assert len(kinds) == 1 and row.shape[0] == sum(layer["kind"] in kinds for layer in layers), row
            assert row.shape[0] == 1 + max(layer["place"] for layer in layers if layer["kind"] in kinds), row
        elif row.per == EVERY:
            assert row.shape[0] == len(layers)
    # the kind that keeps a state is the one whose tensors the layout has
    names = {row.name for row in rows}
    assert {layer["kind"] for layer in layers} - {ATTENTION} == \
        ({"linear"} if "wqkv" in names else {"convolution"} if "win" in names else set())
    assert ATTENTION not in STATEFUL_KINDS.values() and dims.stateful_kind == STATEFUL_KINDS.get(dims.arch)
    # RoPE turns the layers that attend, of a model whose file has (or leaves out) the tables, but those left alone
    turning = any(row.role == TABLE for row in rows)
    assert [layer["rope"] for layer in layers] == \
        [turning and layer["kind"] == ATTENTION and l not in more.get("unturned", ()) for l, layer in enumerate(layers)]


def test_which_layers_are_of_which_kind_and_which_rope_turns():
    def facts(header, form, unturned=()):
        return [(layer["kind"][0], layer["place"], layer["rope"]) for layer in layer_facts(Dims(header, form), unturned)]

    header = [256, 768, 5, 8, 4, 2048, 320]
    assert facts(header, {}) == [("a", l, True) for l in range(5)]
    assert facts(header, {}, (1, 4)) == [("a", 0, True), ("a", 1, False), ("a", 2, True), ("a", 3, True), ("a", 4, False)]
    assert facts(header, {"arch": "gpt2"}) == [("a", l, False) for l in range(5)]
    assert facts(header, {"arch": "neox"}) == [("a", l, True) for l in range(5)]
    assert facts(header, {"arch": "qwen35", "linear": {**LINEAR, "every": 2}}) == \
        [("l", 0, False), ("a", 0, True), ("l", 1, False), ("a", 1, True), ("l", 2, False)]
    assert facts(header, {"arch": "lfm2", "convolution": {"layers": "cacca", "taps": 3}}) == \
        [("c", 0, False), ("a", 0, True), ("c", 1, False), ("c", 2, False), ("a", 1, True)]


@cases
def test_the_widths_of_a_plan(name, header, form, more):
    dims = Dims(header, form)
    widths = forward_plan(dims, "float32", no_bytes, int8=False, kv_start=64, **more)["widths"]
    assert widths == plan_widths(dims) and set(widths) == {"q", "kv", "linear", "rotated"}
    dim, hidden, _, heads, kv_heads = header[:5]
    head = form.get("head_dim") or dim // heads
    assert widths["q"] == heads * head and widths["kv"] == kv_heads * head
    # q's width is a row of wq's, and kv's of wk's, where the heads are the header's
    shapes = {row.name: row.shape for row in tensor_rows(header, form)}
    if form.get("arch") not in ("gpt2", "neox"):
        assert shapes["wq"][1] == widths["q"] and shapes["wk"][1] == widths["kv"] and shapes["wo"][2] == widths["q"]
    if "linear" in form:
        mixed, keys, read = linear_widths(dims.linear)
        assert widths["linear"] == {"mixed": mixed, "keys": keys, "read": read}
        assert shapes["wqkv"][1] == mixed and shapes["wz"][1] == read and shapes["wout"][2] == read
        assert read in widths["rotated"]
    else:
        assert widths["linear"] is None
    assert widths["rotated"] == rotated_widths(dim, heads * head, hidden, dims.linear) == sorted(set(widths["rotated"]))
    assert {dim, heads * head, hidden} <= set(widths["rotated"])


def test_how_a_model_is_run_is_the_plans():
    dims = Dims([256, 768, 4, 8, 4, 2048, 320])
    asked = []

    def read(offset, length):
        asked.append((offset, length))
        weight = np.ones(length // 4, dtype=np.float32)
        weight[[5, 77, 130]] = (50.0, 60.0, 70.0)
        return weight.tobytes()

    plan = forward_plan(dims, "int8", read, int8=True, kv_start=256)
    # (T92) the outlier channels of the final norm, which is read where the plan says it is
    assert asked == [(plan["tensors"]["rms_final_weight"]["offset"], 256 * 4)]
    weight = np.frombuffer(read(0, 256 * 4), dtype=np.float32)
    assert plan["outliers"] == [int(c) for c in outlier_channels(weight, OUTLIER_CHANNELS)]
    assert {5, 77, 130} <= set(plan["outliers"]) and len(plan["outliers"]) == OUTLIER_CHANNELS
    assert all(type(c) is int for c in plan["outliers"])
    assert plan["int8"] is True and plan["half_kv"] is True and plan["relaxed"] is True and plan["kv_start"] == 256
    assert forward_plan(dims, "int8", flat, int8=True, kv_start=256)["outliers"] == []
    # T52's switches
    without = forward_plan(dims, "int8", flat, int8=True, disable=("relaxed", "kv16", "sampler"), kv_start=256)
    assert without["relaxed"] is False and without["half_kv"] is False and without["int8"] is True
    # a model whose matrices are widened: no outliers are looked for, and its keys and values are float32
    widened = forward_plan(dims, "int8", no_bytes, int8=False, disable=("int8",), kv_start=256)
    assert widened["int8"] is False and widened["half_kv"] is False and widened["outliers"] == []
    # the bytes may be JavaScript's (a proxy with to_py)
    proxy = type("Proxy", (), {"to_py": lambda self: memoryview(read(0, 256 * 4))})
    assert forward_plan(dims, "int8", lambda offset, length: proxy(), int8=True, kv_start=256)["outliers"] == plan["outliers"]
    # rotary: 0 is all of a head
    assert forward_plan(dims, "int8", flat, int8=True, kv_start=1, rotary=0)["rotary"] == 32
    assert forward_plan(dims, "int8", flat, int8=True, kv_start=1, rotary=8)["rotary"] == 8


def test_the_tables_python_computes_are_the_plans_bytes():
    header = [256, 768, 4, 8, 4, 2048, 320]
    dims = Dims(header)
    cos, sin = rope_tables("llama", 320, 32, 32, 1.0, "int8", lambda width: rope_frequencies(width, 10000.0, None))
    plan = forward_plan(dims, "int8", flat, int8=True, kv_start=256, tables=(cos, sin))
    assert set(plan["derived"]) == {"freq_cis_real", "freq_cis_imag"}
    assert plan["derived"]["freq_cis_real"] == cos.astype(np.float32).tobytes() and len(plan["derived"]["freq_cis_real"]) == 320 * 16 * 4
    assert plan["derived"]["freq_cis_imag"] == sin.astype(np.float32).tobytes()
    # (a table that is not float32, or not one piece of memory, is made so)
    wide = forward_plan(dims, "int8", flat, int8=True, kv_start=256, tables=(cos.astype(np.float64), sin[::-1][::-1]))
    assert wide["derived"] == plan["derived"]
    # T237: a rotated basis: its block, the signs of every width with the transform's scale in them, and no outliers
    signs = {width: np.where(np.arange(width) % 7 == 3, -1.0, 1.0) for width in plan["widths"]["rotated"]}
    rotated = rotated_form({"block": 16, "signs": {str(width): sign_bits(values) for width, values in signs.items()}},
                           plan["widths"]["rotated"])
    turned = forward_plan(dims, "int8", no_bytes, int8=True, kv_start=256, rotated=rotated, tables=(cos, sin))
    assert turned["rotated"] == 16 and turned["outliers"] == []
    assert set(turned["derived"]) == {"freq_cis_real", "freq_cis_imag", *(f"signs.{width}" for width in signs)}
    for width, values in signs.items():
        got = np.frombuffer(turned["derived"][f"signs.{width}"], dtype=np.float32)
        assert got.size == width and np.array_equal(got, (values * np.float32(1 / math.sqrt(16))).astype(np.float32))


def test_a_file_no_engine_outside_can_take_has_no_plan():
    header = [256, 768, 4, 8, 4, 2048, 320]
    with pytest.raises(ValueError) as refused:
        forward_plan(Dims(header, {"kinds": {"wq": "float16"}}), "int8", flat, int8=True, kv_start=256)
    assert str(refused.value) == SEVERAL_KINDS
    # (a packed kind holds rows of whole groups alone)
    with pytest.raises(ValueError, match="128"):
        forward_plan(Dims([200, 440, 3, 4, 2, 1000, 96]), "ternary", flat, int8=True, kv_start=256)


# ------------------------------------------------------------------------------------------- and what Llama hands on
class Handed(Exception):
    """The plan is in hand: nothing of the engine is built (and the tokenizer is never read)."""


class Outside:
    """forward.js as far as the plan, for a file that is nowhere: its header, its size, a final norm's weight with three
    channels far above the rest, and start(plan), which keeps the plan and ends."""

    def __init__(self, header, size):
        self.header, self.size, self.plan = header, size, None

    def read(self, offset, length):
        if offset == 0:
            return struct.pack("<7i", *self.header)
        weight = 1.0 + np.arange(length // 4, dtype=np.float32) / 1024
        weight[[5, 77, 130]] = (50.0, 60.0, 70.0)
        return weight.tobytes()

    def start(self, plan):
        self.plan = plan
        raise Handed()


@cases
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("disable", [(), ("relaxed", "kv16"), ("int8",)], ids=["", "no relaxed, no kv16", "no int8"])
def test_a_llama_hands_on_the_plan_of_its_file(name, header, form, more, dtype, disable, monkeypatch):
    """Llama gathers what it has (the options it was given, the tables it computes, whether the kernels take the
    matrices as int8) and the plan is forward_plan()'s of the same."""
    monkeypatch.setattr(engine.model, "KV_START", 48)
    dims = Dims(header, form)
    outside = Outside(header, file_size(dims.rows(), dtype))
    with pytest.raises(Handed):
        Llama(None, b"", dtype=dtype, external=outside, disable=disable, **form, **more)
    rotary = more.get("rotary") or dims.head_size
    tables = rope_tables(dims.arch, dims.seq_len, dims.head_size, rotary, 1.0, dtype,
                         lambda width: rope_frequencies(width, 10000.0, None))
    # (every width of these models is whole groups of 32: the int8 kernels take the matrices of a quantized file)
    int8 = dtype in QUANTIZED and "int8" not in disable
    alone = forward_plan(dims, dtype, outside.read, int8=int8, disable=disable, kv_start=48, tables=tables, **more)
    assert outside.plan == alone and list(outside.plan) == list(alone)
    assert bool(alone["outliers"]) is int8 and alone["half_kv"] is (int8 and not disable)
    assert set(alone["derived"]) == (set() if tables is None else {"freq_cis_real", "freq_cis_imag"})
