"""T359.6: the table of architectures (engine/layout.py's LAYOUTS), a record at a time and with no model: the rows of
each for a header, how its form is read and refused, its facts (against a second copy of them, here); that a name the
table has not is refused wherever a form comes in, in one sentence; that a form and its architecture are held together
in one place, whose one sentence every entrance says; that a record made of another's and a new kind of layer are read
by every part without a line in them; and the settings of a plan as one value."""
import struct

import numpy as np
import pytest

import engine.model
import llama2_convert
from conftest import pack_tokenizer, tiny_vocab
from engine.layout import GATED_DELTA, SHORT_CONVOLUTION
from llama2_numpy import (ATTENDING, EVERY, FORM, LAYOUTS, MATRIX, PARTLY, STATEFUL, TABLE, WHOLE, Dims, Layout, Llama,
                          Settings, Stateful, after, checkpoint_dtype, external_tensors, file_size, form_of,
                          forward_plan, layer_facts, layer_slots, layout_of, rope_frequencies, slots_of,
                          stateful_form, stateful_kinds, tensor_rows, unturned_layers)
from engine.tensors import rope_tables

LINEAR = {"every": 2, "key_heads": 2, "value_heads": 4, "key_dim": 16, "value_dim": 32, "conv": 4}
CONVOLUTION = {"layers": "ccac", "taps": 3}
# the facts of every architecture, a second time: a record that is changed has to be changed here too
FACTS = {
    "llama": dict(rows="llama", stateful=None, rope=WHOLE, yarn_magnitude=True, layer_norm=False, gated_ffn=True,
                  rotatable=True, unturned=True),
    "gpt2": dict(rows="gpt2", stateful=None, rope=None, yarn_magnitude=False, layer_norm=True, gated_ffn=False,
                 rotatable=False, unturned=False),
    "neox": dict(rows="gpt2", stateful=None, rope=PARTLY, yarn_magnitude=False, layer_norm=True, gated_ffn=False,
                 rotatable=False, unturned=False),
    "qwen35": dict(rows="qwen35", stateful="linear", rope=PARTLY, yarn_magnitude=False, layer_norm=False, gated_ffn=True,
                   rotatable=True, unturned=False),
    "lfm2": dict(rows="lfm2", stateful="convolution", rope=WHOLE, yarn_magnitude=False, layer_norm=False, gated_ffn=True,
                 rotatable=False, unturned=False),
}
# (a header and the form each architecture needs to be laid out)
HEADER = [64, 128, 4, 4, 2, 320, 48]
FORMS = {"llama": {}, "gpt2": {"arch": "gpt2"}, "neox": {"arch": "neox"},
         "qwen35": {"arch": "qwen35", "head_dim": 16, "linear": LINEAR},
         "lfm2": {"arch": "lfm2", "qk_norm": True, "convolution": CONVOLUTION}}
# what an engine is told of a model of each, besides the form (Llama's arguments)
OPTIONS = {"qwen35": {"rotary": 4}, "neox": {"rotary": 4}}
every = pytest.mark.parametrize("arch", list(FACTS))


def made_up(header, form=None, dtype="float32", seed=6):
    """A checkpoint of this header and form with small random numbers for every tensor."""
    size = file_size(tensor_rows(header, form), dtype) - 28
    width = {"float32": 4, "float16": 2}[dtype]
    values = np.random.default_rng(seed).uniform(-0.2, 0.2, size // width).astype(dtype)
    return struct.pack("<7i", *header) + values.tobytes()


def model(arch, dtype="float32", **more):
    # (the heads of a GPT-2 and of a GPT-NeoX fill dim, and their keys and values are as many)
    header = [*HEADER[:4], HEADER[3] if FACTS[arch]["rows"] == "gpt2" else HEADER[4], *HEADER[5:]]
    form = {**FORMS[arch], "arch": arch}
    return Llama(made_up(header, form, dtype), pack_tokenizer(tiny_vocab(320)), dtype=dtype,
                 **{**form, **OPTIONS.get(arch, {}), **more})


def refusal(call, *arguments, **named):
    with pytest.raises(ValueError) as caught:
        call(*arguments, **named)
    return str(caught.value)


# ---- a record at a time, with no model
def test_the_table_is_these_architectures_and_each_has_these_facts():
    assert list(LAYOUTS) == list(FACTS)
    for arch, facts in FACTS.items():
        layout = layout_of(arch)
        assert layout is LAYOUTS[arch] and isinstance(layout, Layout)
        said = {**layout._asdict(), "rows": layout.rows.__name__, "stateful": layout.stateful and layout.stateful.name}
        assert said == facts, arch
    # (every fact of a record is in the copy: a fact that is added is added there)
    assert set(Layout._fields) == set(FACTS["llama"])
    # a GPT-NeoX is a GPT-2 but for what RoPE turns, and nothing else
    assert LAYOUTS["neox"] == LAYOUTS["gpt2"]._replace(rope=PARTLY)


@every
def test_the_rows_of_a_record_are_what_its_facts_say_the_forward_pass_reads(arch):
    """The facts are the forward pass's and the rows are the file's: a record whose two halves disagree would have the
    reference read a tensor the file has not."""
    layout, dims = LAYOUTS[arch], Dims(HEADER, FORMS[arch])
    rows = layout.rows(dims)
    assert rows == dims.rows() == tensor_rows(HEADER, FORMS[arch])
    names = {row.name for row in rows}
    # RoPE tables, or a learned table of positions
    assert any(row.role == TABLE for row in rows) == (layout.rope is not None) == ("positions" not in names)
    # a LayerNorm has a bias, and GPT-2's FFN has two matrices and their biases where a gated one has three matrices
    assert {"ln_att_bias", "ln_ffn_bias", "ln_final_bias"} <= names if layout.layer_norm else not names & {"ln_att_bias", "ln_ffn_bias", "ln_final_bias"}
    assert ("w3" in names) == layout.gated_ffn == (not names & {"b1", "b2"})
    # the tensors of the layers that keep a state, where the architecture has such layers, and as many as there are
    kept = [row for row in rows if row.per == STATEFUL]
    assert bool(kept) == (layout.stateful is not None)
    count = sum(state for state, _ in dims.slots)
    assert all(row.shape[0] == count for row in kept)
    assert all(row.shape[0] == HEADER[2] - count for row in rows if row.per == ATTENDING)
    assert all(row.shape[0] == HEADER[2] for row in rows if row.per == EVERY)


@every
def test_a_dims_has_the_numbers_of_its_kind_of_layer_and_of_no_other(arch):
    dims, kind = Dims(HEADER, FORMS[arch]), LAYOUTS[arch].stateful
    assert dims.layout is LAYOUTS[arch] and dims.arch == arch
    assert dims.stateful_kind == (kind.name if kind else None)
    numbers = {name: getattr(dims, name) for name in stateful_kinds()}
    assert set(numbers) == {"linear", "convolution"}
    assert {name for name, value in numbers.items() if value is not None} == ({kind.name} if kind else set())
    if kind:
        assert numbers[kind.name] == kind.parse(FORMS[arch][kind.name], HEADER[2])
        assert dims.slots == slots_of(kind.layers(numbers[kind.name], HEADER[2]))
    else:
        assert dims.slots == [(False, layer) for layer in range(HEADER[2])]


def test_the_kinds_of_layers_that_keep_a_state():
    assert stateful_kinds() == {"linear": GATED_DELTA, "convolution": SHORT_CONVOLUTION}
    assert set(stateful_kinds()) <= set(FORM) and all(FORM[name] is None for name in stateful_kinds())
    # which layers are of the kind: all but every second of a Qwen3.5's, an LFM2's by their letters
    linear = GATED_DELTA.parse(LINEAR, 5)
    assert GATED_DELTA.layers(linear, 5) == [True, False, True, False, True]
    assert GATED_DELTA.layers({**linear, "every": 4}, 9) == [True, True, True, False, True, True, True, False, True]
    convolution = SHORT_CONVOLUTION.parse(CONVOLUTION, 4)
    assert SHORT_CONVOLUTION.layers(convolution, 4) == [True, True, False, True]
    assert slots_of([True, True, False, True, False]) == [(True, 0), (True, 1), (False, 0), (True, 2), (False, 1)]
    assert slots_of([]) == [] and slots_of([False] * 3) == [(False, 0), (False, 1), (False, 2)]
    # (the function of the two forms, which tools without a Dims call, says the same)
    assert layer_slots(5, linear) == slots_of(GATED_DELTA.layers(linear, 5))
    assert layer_slots(4, None, convolution) == slots_of(SHORT_CONVOLUTION.layers(convolution, 4))
    assert layer_slots(3, None) == slots_of([False] * 3)
    # nothing said is nothing
    assert GATED_DELTA.parse(None, 4) is None and SHORT_CONVOLUTION.parse(None, 4) is None


def test_a_kind_refuses_numbers_that_are_none_of_a_model_of_these_layers():
    for wrong in ({"every": 1}, {"conv": 0}, {"value_heads": 3}):
        assert "not the numbers of linear-attention layers" in refusal(GATED_DELTA.parse, {**LINEAR, **wrong}, 4)
    # (a model too short for one full-attention layer: Llama and checkpoint_dtype() each had a check of their own)
    assert "no full-attention layer" in refusal(GATED_DELTA.parse, {**LINEAR, "every": 5}, 4)
    assert GATED_DELTA.parse({**LINEAR, "every": 4}, 4)["every"] == 4
    for wrong in ({"layers": "ccacc"}, {"layers": "cca"}, {"layers": "ccxc"}, {"taps": 1}):
        assert "not the convolution layers" in refusal(SHORT_CONVOLUTION.parse, {**CONVOLUTION, **wrong}, 4)


# ---- a name the table has not
ENTRANCES = {
    "layout_of()": lambda arch: layout_of(arch),
    "Dims": lambda arch: Dims(HEADER, {"arch": arch}),
    "tensor_rows()": lambda arch: tensor_rows(HEADER, {"arch": arch}),
    "checkpoint_dtype()": lambda arch: checkpoint_dtype(HEADER, 1 << 20, {"arch": arch}),
    "external_tensors()": lambda arch: external_tensors(HEADER, "int8", {"arch": arch}),
    "the converter's layout()": lambda arch: llama2_convert.layout(*HEADER, arch=arch),
    "the converter's checkpoint_size()": lambda arch: llama2_convert.checkpoint_size(HEADER, "int8", {"arch": arch}),
    "the converter's Writer": lambda arch: llama2_convert.Writer(None, HEADER, "float32", {"arch": arch}, sink=type(
        "Sink", (), {"open": lambda *_: None, "write": lambda *_: None})()),
    "Llama": lambda arch: Llama(struct.pack("<7i", *HEADER), None, arch=arch),
    "Llama, its weights outside": lambda arch: Llama(None, None, arch=arch, external=type("Outside", (), {
        "size": 1 << 20, "read": staticmethod(lambda offset, length: struct.pack("<7i", *HEADER))})()),
    "stateful_form()": lambda arch: stateful_form(arch, form_of({"arch": arch}), 4),
    "unturned_layers()": lambda arch: unturned_layers(arch, 4, [1]),
    "rope_tables()": lambda arch: rope_tables(arch, 48, 16, 16, 1.0, "int8", lambda width: rope_frequencies(width, 10000.0, None)),
}


@pytest.mark.parametrize("entrance", list(ENTRANCES))
def test_a_name_that_is_no_architectures_is_refused_wherever_it_comes_in(entrance):
    """It was read as a Llama until T359.6: a mistyped name ran another model's forward pass without a word."""
    for arch in ("llama3", "LLAMA", "qwen3", "", "gpt-2"):
        assert refusal(ENTRANCES[entrance], arch) == f"There is no architecture called {arch!r}: llama, gpt2, neox, qwen35, lfm2."
    for arch in FACTS:  # (and a name it has is not what is refused there)
        try:
            ENTRANCES[entrance](arch)
        except ValueError as other:
            assert "no architecture called" not in str(other)


def test_a_name_that_is_no_string_is_no_architecture():
    for arch in (0, ("llama",), 1.5):
        assert "no architecture called" in refusal(layout_of, arch)
    # (nothing said is the form's default, a Llama: form_of() reads None so)
    assert Dims(HEADER, {"arch": None}).arch == "llama" == form_of(None)["arch"]


# ---- a form and its architecture, held together in one place
TOGETHER = [
    ("qwen35", {}, "The architecture 'qwen35' and the layers of its form go together: it has linear that keep a state, "
                   "and the form says none."),
    ("lfm2", {}, "The architecture 'lfm2' and the layers of its form go together: it has convolution that keep a state, "
                 "and the form says none."),
    ("llama", {"linear": LINEAR}, "The architecture 'llama' and the layers of its form go together: it has none that "
                                  "keep a state, and the form says linear."),
    ("gpt2", {"convolution": CONVOLUTION}, "The architecture 'gpt2' and the layers of its form go together: it has none "
                                           "that keep a state, and the form says convolution."),
    ("qwen35", {"convolution": CONVOLUTION}, "The architecture 'qwen35' and the layers of its form go together: it has "
                                             "linear that keep a state, and the form says convolution."),
    ("lfm2", {"linear": LINEAR, "convolution": CONVOLUTION},
     "The architecture 'lfm2' and the layers of its form go together: it has convolution that keep a state, and the "
     "form says linear, convolution."),
]


@pytest.mark.parametrize("arch, said, sentence", TOGETHER, ids=[f"{arch} with {', '.join(said) or 'nothing'}" for arch, said, _ in TOGETHER])
def test_a_form_without_its_architectures_layers_or_with_anothers_is_refused_in_one_sentence(arch, said, sentence):
    """Three places had this check, each with a sentence of its own (the Dims, checkpoint_dtype() and Llama), and the
    Dims' let another architecture's layers through unheard."""
    form = {"arch": arch, **said}
    assert refusal(stateful_form, arch, form_of(form), 4) == sentence
    assert refusal(Dims, HEADER, form) == sentence
    assert refusal(tensor_rows, HEADER, form) == sentence
    assert refusal(external_tensors, HEADER, "int8", form) == sentence
    assert refusal(llama2_convert.layout, *HEADER, **form) == sentence
    assert refusal(Llama, struct.pack("<7i", *HEADER), None, **form) == sentence
    # (checkpoint_dtype() says of whatever it refuses that it is no checkpoint, in front)
    assert refusal(checkpoint_dtype, HEADER, 1 << 20, form) == f"This is not a llama2.c checkpoint: {sentence}"


def test_numbers_that_are_no_layers_of_this_model_are_refused_the_same_way_everywhere():
    for form, words in (({"arch": "qwen35", "linear": {**LINEAR, "every": 5}}, "A model of 4 layers has no full-attention layer where every 5th is one."),
                        ({"arch": "lfm2", "convolution": {"layers": "ccacc", "taps": 3}}, "These are not the convolution layers of a model: 'ccacc' with 3 taps.")):
        assert refusal(Dims, HEADER, form) == words
        assert refusal(external_tensors, HEADER, "int8", form) == words
        assert refusal(Llama, struct.pack("<7i", *HEADER), None, **form) == words
        assert refusal(checkpoint_dtype, HEADER, 1 << 20, form) == f"This is not a llama2.c checkpoint: {words}"


# ---- the facts, as each part reads them
@every
def test_the_layers_rope_leaves_alone_are_an_architectures_that_has_such(arch):
    if FACTS[arch]["unturned"]:
        assert unturned_layers(arch, 4, [3, 1, 3]) == (1, 3) and unturned_layers(arch, 4, None) == ()
        for layers in ([4], [-1], [0, 9]):
            assert "layers RoPE leaves alone" in refusal(unturned_layers, arch, 4, layers)
    else:
        assert unturned_layers(arch, 4, ()) == () and "layers RoPE leaves alone" in refusal(unturned_layers, arch, 4, [1])


@every
def test_the_tables_the_engine_makes_are_what_the_facts_of_rope_say(arch):
    frequencies = lambda width: rope_frequencies(width, 10000.0, None)
    make = lambda dtype, magnitude=1.0, rotary=16: rope_tables(arch, 48, 16, rotary, magnitude, dtype, frequencies)
    angles = np.arange(48)[:, None] * frequencies(16)
    if FACTS[arch]["rope"] is None:
        # no turn: tables of zeros that nothing reads, whatever the file's dtype
        assert all(not table.any() and table.shape == (48, 8) for dtype in ("float32", "int8") for table in make(dtype))
        return
    assert make("float32") is None  # the file has them
    cos, sin = make("int8", 1.5)
    if FACTS[arch]["rope"] == PARTLY:
        part = np.arange(48)[:, None] * frequencies(4)
        cos, sin = make("int8", 1.5, 4)
        assert np.array_equal(cos[:, :2], np.cos(part).astype(np.float32)) and not cos[:, 2:].any() and not sin[:, 2:].any()
    else:
        magnitude = 1.5 if FACTS[arch]["yarn_magnitude"] else 1.0
        assert np.array_equal(cos, (np.cos(angles) * magnitude).astype(np.float32))
        assert np.array_equal(sin, (np.sin(angles) * magnitude).astype(np.float32))


@every
def test_a_model_has_its_architectures_facts_from_the_time_it_is_made(arch):
    """What the reference reads at every token is the record's, resolved once: attributes, and no lookup in a step."""
    llama, facts = model(arch), FACTS[arch]
    assert (llama.layer_norm, llama.gated_ffn) == (facts["layer_norm"], facts["gated_ffn"])
    assert (llama.turning, llama.partly) == (facts["rope"] is not None, facts["rope"] == PARTLY)
    # the step and the states of its layers that keep a state, by their names
    assert (llama.stateful, llama.states) == {"linear": ("linear_attention", ("delta_state", "conv_state")),
                                              "convolution": ("short_convolution", ("conv_state",))}.get(facts["stateful"], (None, ()))
    assert set(engine.model.STATEFUL_LAYERS) == set(stateful_kinds())
    kept = sum(state for state, _ in llama.slots)
    assert all(getattr(llama, name).shape[0] == kept and not getattr(llama, name).any() for name in llama.states)
    logits = [llama.forward(token, pos) for pos, token in enumerate((1, 7, 9))]
    assert all(np.isfinite(step).all() and step.shape == (320,) for step in logits)
    if llama.states:
        assert any(getattr(llama, name).any() for name in llama.states)
        # position 0 clears every one of them, and the run is the same again
        again = llama.forward(1, 0)
        assert np.array_equal(again, logits[0])
        with pytest.raises(ValueError, match="position 1 comes next"):
            llama.forward(9, 2)
    # a rotated basis and the layers RoPE leaves alone, of the architectures that take them
    rotated = {"block": 32, "signs": {"64": "0" * 16, "128": "0" * 32, "32": "0" * 8}}
    if facts["rotatable"]:
        assert model(arch, rotated=rotated).rotated["block"] == 32
    else:
        with pytest.raises(ValueError, match="rotated basis"):
            model(arch, rotated=rotated)
    if facts["unturned"]:
        assert model(arch, unturned=[2]).unturned == (2,)
    else:
        with pytest.raises(ValueError, match="layers RoPE leaves alone"):
            model(arch, unturned=[2])


# ---- adding to the table: a record and nothing in the parts that read it
def test_an_architecture_that_is_anothers_record_but_for_a_fact(monkeypatch):
    """A made-up "plain": a Llama whose RoPE tables have no yarn magnitude. One record, and every part reads it."""
    monkeypatch.setitem(LAYOUTS, "plain", LAYOUTS["llama"]._replace(yarn_magnitude=False))
    form = {"arch": "plain"}
    assert tensor_rows(HEADER, form) == tensor_rows(HEADER)
    assert checkpoint_dtype(HEADER, file_size(tensor_rows(HEADER), "float16"), form) == "float16"
    assert external_tensors(HEADER, "int8", form) == external_tensors(HEADER, "int8")
    assert llama2_convert.layout(*HEADER, **form) == llama2_convert.layout(*HEADER)
    yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 12}
    checkpoint, tokenizer = made_up(HEADER, form, "float16"), pack_tokenizer(tiny_vocab(320))
    plain = Llama(checkpoint, tokenizer, dtype="float16", arch="plain", rope_scaling=yarn)
    llama = Llama(checkpoint, tokenizer, dtype="float16", rope_scaling=yarn)
    assert llama.rope_magnitude == plain.rope_magnitude > 1.1
    assert np.allclose(plain.freq_cis_real * llama.rope_magnitude, llama.freq_cis_real, rtol=1e-6)
    theirs, ours = [[each.forward(token, pos) for pos, token in enumerate((1, 7, 9))] for each in (llama, plain)]
    assert np.array_equal(theirs[0], ours[0])  # (one token attends to itself: no turn shows)
    assert np.isfinite(ours[2]).all() and not np.allclose(theirs[2], ours[2], atol=1e-5)
    # the plan of it, with no model: a Llama's, under its name
    plan = forward_plan(Dims(HEADER, form), "int8", lambda offset, length: bytes(length), Settings(int8=False, kv_start=8))
    assert plan["arch"] == "plain" and {layer["kind"] for layer in plan["layers"]} == {"attention"}


def test_the_kind_of_the_norms_and_of_the_ffn_are_two_facts(monkeypatch):
    """Every architecture of the table has LayerNorms with GPT-2's FFN or RMSNorms with a gated one, so nothing of
    theirs tells one fact from the other: a made-up GPT-2 with RMSNorms (the file's biases of the norms unread) does."""
    monkeypatch.setitem(LAYOUTS, "rms gpt2", LAYOUTS["gpt2"]._replace(layer_norm=False))
    monkeypatch.setitem(FACTS, "rms gpt2", {**FACTS["gpt2"], "layer_norm": False})
    monkeypatch.setitem(FORMS, "rms gpt2", {"arch": "rms gpt2"})
    ours, gpt2 = model("rms gpt2"), model("gpt2")
    assert (ours.layer_norm, ours.gated_ffn) == (False, False) and (gpt2.layer_norm, gpt2.gated_ffn) == (True, False)
    assert ours.rows == gpt2.rows and ours.w3 is None and ours.b1 is not None
    theirs, mine = gpt2.forward(5, 0), ours.forward(5, 0)
    assert np.isfinite(mine).all() and not np.allclose(theirs, mine, atol=1e-5)


def test_a_new_kind_of_layer_that_keeps_a_state_is_a_record_and_its_rows(monkeypatch):
    """A made-up "rnn": a Llama some of whose layers are recurrent (one matrix each, and a state). The kind, the rows
    and the record are all there is to write for the file's side: the Dims, the sizes, the places, the refusals and
    the plan's layers have it without a line of theirs."""
    def parse(recurrent, n_layers):
        if recurrent is not None and len(recurrent["layers"]) != n_layers:
            raise ValueError("These are not the recurrent layers of a model.")
        return recurrent and {"layers": str(recurrent["layers"]), "width": int(recurrent["width"])}

    recurrent = Stateful("recurrent", parse, lambda numbers, n_layers: [kind == "r" for kind in numbers["layers"]])
    rnn = lambda d: after(LAYOUTS["llama"].rows(d), "wo", d.stack("wr", MATRIX, STATEFUL, d.recurrent["width"], d.dim))
    monkeypatch.setitem(FORM, "recurrent", None)
    monkeypatch.setitem(LAYOUTS, "rnn", Layout(rnn, stateful=recurrent))
    form = {"arch": "rnn", "recurrent": {"layers": "rara", "width": 96}}
    assert stateful_kinds()["recurrent"] is recurrent
    dims = Dims(HEADER, form)
    assert dims.recurrent == form["recurrent"] and dims.stateful_kind == "recurrent"
    assert dims.linear is None and dims.convolution is None and Dims(HEADER).recurrent is None
    assert dims.slots == [(True, 0), (False, 0), (True, 1), (False, 1)]
    rows = {row.name: row for row in tensor_rows(HEADER, form)}
    assert rows["wr"].shape == (2, 96, 64) and rows["wq"].shape[0] == 2 and rows["w1"].shape[0] == 4
    # the sizes and the places
    size = file_size(tensor_rows(HEADER, form), "int8")
    assert checkpoint_dtype(HEADER, size, form) == "int8" and llama2_convert.checkpoint_size(HEADER, "int8", form) == size
    assert external_tensors(HEADER, "int8", form)["wr"]["shape"] == [2, 96, 64]
    # the plan's layers, with no model
    layers = layer_facts(dims, Settings(int8=True, kv_start=8))
    assert [(layer["kind"], layer["place"], layer["rope"]) for layer in layers] == \
        [("recurrent", 0, False), ("attention", 0, True), ("recurrent", 1, False), ("attention", 1, True)]
    # the refusals: its own, and the one sentence of a form that does not go with its architecture
    assert refusal(Dims, HEADER, {**form, "recurrent": {"layers": "rar", "width": 96}}) == "These are not the recurrent layers of a model."
    assert refusal(Dims, HEADER, {"arch": "rnn"}) == ("The architecture 'rnn' and the layers of its form go together: it "
                                                      "has recurrent that keep a state, and the form says none.")
    assert refusal(checkpoint_dtype, HEADER, size, {**form, "arch": "llama"}) == (
        "This is not a llama2.c checkpoint: The architecture 'llama' and the layers of its form go together: it has "
        "none that keep a state, and the form says recurrent.")
    assert "layers RoPE leaves alone" in refusal(unturned_layers, "rnn", 4, [1])


# ---- the settings of a plan, one value
def test_the_settings_of_a_plan_are_one_value_or_its_fields_by_name():
    dims, read = Dims([256, 768, 4, 8, 4, 2048, 320]), lambda offset, length: np.ones(length // 4, dtype=np.float32).tobytes()
    settings = Settings(int8=True, kv_start=48, disable=("kv16",), rotary=8, parallel_residual=True, rms_norm_eps=1e-6,
                        unturned=(2, 0))
    plan = forward_plan(dims, "int8", read, settings)
    assert plan == forward_plan(dims, "int8", read, **settings._asdict())
    assert (plan["rotary"], plan["parallel_residual"], plan["rms_norm_eps"], plan["kv_start"], plan["half_kv"],
            plan["unturned"]) == (8, True, 1e-6, 48, False, [0, 2])
    assert [layer["rope"] for layer in plan["layers"]] == [False, True, False, True]
    # what is run has to be said, and a name that is no setting's is refused
    assert Settings._fields[:2] == ("int8", "kv_start") and Settings._field_defaults.keys() == set(Settings._fields[2:])
    for wrong in ({}, {"int8": True}, {"int8": True, "kv_start": 8, "unturnd": (1,)}):
        with pytest.raises(TypeError):
            forward_plan(dims, "int8", read, **wrong)


def test_the_plan_refuses_the_layers_rope_leaves_alone_that_a_model_refuses():
    """T359.5's review: forward_plan() alone let a layer the model has not through (it showed in no layer's facts)."""
    read = lambda offset, length: np.ones(length // 4, dtype=np.float32).tobytes()
    settings = lambda **more: Settings(int8=True, kv_start=8, **more)
    for layers in ((4,), (-1,), (0, 9)):
        assert "layers RoPE leaves alone" in refusal(forward_plan, Dims(HEADER), "int8", read, settings(unturned=layers))
        assert "layers RoPE leaves alone" in refusal(layer_facts, Dims(HEADER), settings(unturned=layers))
    for arch in ("gpt2", "neox", "qwen35", "lfm2"):
        assert "layers RoPE leaves alone" in refusal(forward_plan, Dims(HEADER, FORMS[arch]), "int8", read, settings(unturned=(1,)))
    assert forward_plan(Dims(HEADER), "int8", read, settings(unturned=[3, 3, 1]))["unturned"] == [1, 3]
