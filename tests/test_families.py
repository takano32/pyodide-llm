"""convert/families/ (T359): what differs between one model_type of config.json and another is one record in one
table, which the converter's shared flow looks up. These hold the table to what the flow asks of it; what each family
converts to is its own tests' (test_granite.py, test_smollm3.py, test_qwen35.py, test_lfm2.py, test_gpt2.py,
test_neox.py, test_gguf.py), and the net's kinds "layouts" and "python" compare the answers between two trees."""
import json
import math

import numpy as np
import pytest
from conftest import synthetic_weights
from test_convert import converted, hugging_face, reader, safetensors_file, streamed
from test_gguf import fed, gguf_file, unigram, with_original

import llama2_convert
import llama2_numpy
from convert import families
from convert.families.llama import GRANITE, LLAMA, MISTRAL, QWEN2, QWEN3, SMOLLM3, gguf_config
from llama2_convert import (FAMILIES, Family, Safetensors, architecture, check_config, conversion_plan, family_of,
                            gguf_model, normalize)
from llama2_numpy import FORM, TABLE, Dims, tensor_rows

# The table as it stands, written a second time on purpose (as the order of the rows is in test_layout.py): a
# family's fact changed by mistake changes which models convert and how, and few of the facts have a test of their
# own. model_type: (title, arch, GGUF architecture, classifier, the facts that are true of it)
FACTS = {
    "llama": ("Llama", "llama", "llama", "lm_head.weight", {"free_heads", "scaled", "rotatable", "rms_norm"}),
    "mistral": ("Mistral", "llama", None, "lm_head.weight", {"renamed", "free_heads", "scaled", "rotatable", "rms_norm"}),
    "granite": ("Granite", "llama", "granite", "lm_head.weight", {"free_heads", "scaled", "rotatable", "rms_norm"}),
    "smollm3": ("SmolLM3", "llama", "smollm3", "lm_head.weight", {"free_heads", "scaled", "rotatable", "rms_norm"}),
    "qwen2": ("Qwen2", "llama", "qwen2", "lm_head.weight", {"free_heads", "scaled", "rotatable", "rms_norm"}),
    "qwen3": ("Qwen3", "llama", "qwen3", "lm_head.weight", {"free_heads", "scaled", "rotatable", "rms_norm"}),
    "qwen3_5_text": ("Qwen3.5", "qwen35", "qwen35", "lm_head.weight", {"free_heads", "rotatable", "rms_norm", "partly"}),
    "qwen3_5": ("Qwen3.5", "qwen35", None, "lm_head.weight", {"renamed", "free_heads", "rotatable", "rms_norm", "partly"}),
    "lfm2": ("LFM2", "lfm2", "lfm2", "lm_head.weight", {"rms_norm"}),
    "gpt2": ("GPT-2", "gpt2", "gpt2", "lm_head.weight", {"fixed_context"}),
    "gpt_neox": ("GPT-NeoX", "neox", "gptneox", "embed_out.weight", {"partly"}),
}
FLAGS = ("renamed", "fixed_context", "free_heads", "scaled", "rotatable", "rms_norm", "partly")
LINEAR = {"every": 4, "key_heads": 2, "value_heads": 4, "key_dim": 64, "value_dim": 128, "conv": 4}
# a header and a form of every layout, with every row a layout can have (a classifier of its own, a Qwen2's biases
# and a Qwen3's norms at once), and how much of a head turns
LAYOUTS = {
    "llama": ([256, 768, 4, 4, 2, -2048, 320], {"bias": True, "qk_norm": True, "head_dim": 128}, 0),
    "gpt2": ([256, 1024, 3, 8, 8, -2304, 320], {"arch": "gpt2"}, 0),
    "neox": ([256, 1024, 3, 8, 8, -2048, 320], {"arch": "neox"}, 8),
    "qwen35": ([256, 768, 8, 4, 2, -2048, 320], {"arch": "qwen35", "head_dim": 128, "linear": LINEAR}, 32),
    "lfm2": ([256, 768, 8, 8, 4, -2048, 320], {"arch": "lfm2", "convolution": {"layers": "ccaccaca", "taps": 3}}, 0),
}
BASE = {"hidden_size": 256, "intermediate_size": 768, "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 2048,
        "max_position_embeddings": 4096}
by_type = pytest.mark.parametrize("model_type", list(FACTS))


def test_the_table_is_the_model_types_the_converter_takes():
    assert list(FAMILIES) == list(FACTS)
    assert all(isinstance(family, Family) for family in FAMILIES.values())


@by_type
def test_a_model_type_resolves_to_its_family_and_the_family_says_what_it_is(model_type):
    title, arch, gguf, classifier, true = FACTS[model_type]
    family = family_of({"model_type": model_type})
    assert family is FAMILIES[model_type]
    assert (family.title, family.arch, family.gguf, family.classifier) == (title, arch, gguf, classifier)
    assert {flag for flag in FLAGS if getattr(family, flag)} == true
    assert architecture({"model_type": model_type}) == arch
    assert all(getattr(family, flag) in (True, False) for flag in FLAGS)


def test_an_unknown_model_type_is_refused_in_the_words_it_was():
    for model_type, said in (("bert", "a bert"), (None, "a model of unknown type")):
        config = {**BASE, **({"model_type": model_type} if model_type else {})}
        with pytest.raises(ValueError) as refusal:
            check_config(normalize(config))
        assert str(refusal.value) == (
            f"This model cannot be converted: it is {said}, and only Llama, Mistral, Granite, SmolLM3, Qwen2, Qwen3, "
            f"Qwen3.5, LFM2, GPT-2 and GPT-NeoX models are supported.")
        # (its layout and its plan are a Llama's until then: nothing else is asked of an unknown name)
        assert family_of(config) is LLAMA and architecture(config) == "llama"


def test_a_name_that_normalize_gives_another_is_taken_under_the_other_only():
    """check_config() reads a normalize()d config: a Mistral is a "llama" there and a Qwen3.5 with its vision model
    a "qwen3_5_text", and the name config.json had is refused as it was before there was a table."""
    mistral = {**BASE, "model_type": "mistral", "sliding_window": 1024}
    assert normalize(mistral)["model_type"] == "llama" and normalize(mistral)["max_position_embeddings"] == 1024
    check_config(normalize(mistral))
    with pytest.raises(ValueError, match="it is a mistral, and only Llama"):
        check_config(mistral)
    whole = {"model_type": "qwen3_5", "text_config": {**BASE, "num_hidden_layers": 8, "num_key_value_heads": 2, "eos_token_id": 7}}
    text = normalize(whole)
    # the family after the lift is the language model's: its defaults are filled in (head_dim, rotary_pct)
    assert (text["model_type"], text["bos_token_id"], text["head_dim"], text["rotary_pct"]) == ("qwen3_5_text", 7, 256, 0.25)
    check_config(text)
    with pytest.raises(ValueError, match="it is a qwen3_5, and only Llama"):
        check_config(whole)
    # one without a text_config stays what it is, defaults and all, and is refused by its name
    assert normalize({**BASE, "model_type": "qwen3_5"}) == {**BASE, "model_type": "qwen3_5"}
    assert normalize(normalize(whole)) == normalize(whole) and normalize(normalize(mistral)) == normalize(mistral)


def test_a_familys_own_names_are_read_before_the_ones_of_rope_that_all_share():
    """An LFM2's theta is 1e6 where its config.json says none, also where it has a rope_parameters that says none:
    read after the shared names, the Llama's 10000 would be there already."""
    lfm2 = {"model_type": "lfm2", "hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 2048,
            "max_position_embeddings": 4096, "block_ff_dim": 768, "layer_types": ["conv", "conv", "full_attention", "full_attention"]}
    assert normalize(lfm2)["rope_theta"] == normalize({**lfm2, "rope_parameters": {}})["rope_theta"] == 1000000.0
    assert normalize({**lfm2, "rope_parameters": {"rope_theta": 5e5}})["rope_theta"] == 5e5
    assert normalize({**BASE, "model_type": "llama", "rope_parameters": {}})["rope_theta"] == 10000.0


def test_only_a_qwen2_may_say_its_attention_has_biases():
    """transformers' attention_bias: q, k and v of a Qwen2 have one (the file's own tensors say so to the layout);
    any other Llama that says it has tensors the engine would leave out."""
    biased = {**BASE, "attention_bias": True}
    check_config({**biased, "model_type": "qwen2"})
    for model_type in ("llama", "qwen3", "granite", "smollm3", "qwen3_5_text", "lfm2"):
        with pytest.raises(ValueError, match="its layers have biases"):
            check_config(normalize({**biased, "model_type": model_type, "num_hidden_layers": 8,
                                    "layer_types": None if model_type != "lfm2" else ["conv", "full_attention"] * 4}))
    with pytest.raises(ValueError, match="its layers have biases"):
        check_config({**BASE, "model_type": "qwen2", "mlp_bias": True})


@by_type
def test_a_family_has_a_source_for_every_row_of_its_layout_and_no_other(model_type):
    family = FAMILIES[model_type]
    header, form, rotary = LAYOUTS[family.arch]
    rows = tensor_rows(header, form)
    sources = family.sources(Dims(header, form), "", rotary)
    wanted = {row.name for row in rows if row.role != TABLE}
    assert set(sources) == wanted
    assert all(isinstance(name, str) and (transform is None or isinstance(transform, tuple))
               for name, transform in sources.values())
    # and the plan of the layout is made of them: a form says its layout, and the layout's family is this one's
    assert families.of_layout(family.arch).sources is family.sources
    d = Dims(header, form)
    plan, shapes = conversion_plan(header, form, "", rotary)
    for row, parts in zip(rows, plan):
        if row.role != TABLE:
            name, transform = sources[row.name]
            assert parts == ([(name.format(layer), transform) for layer in d.layers(row.per)] if row.per else [(name, transform)])
    # the classifier's name is the one checkpoint_header() looks for
    assert sources["wcls"][0] == family.classifier


def test_every_layout_of_the_engine_has_a_family_and_the_first_of_the_table_stands_for_it():
    assert set(families.LAYOUTS) == set(LAYOUTS) == {family.arch for family in FAMILIES.values()}
    assert set(families.LAYOUTS) == set(llama2_numpy.LAYOUTS)
    assert families.LAYOUTS["llama"] is LLAMA and families.LAYOUTS["qwen35"] is FAMILIES["qwen3_5_text"]
    assert families.of_layout("a layout nobody wrote") is LLAMA


def gguf_targets(family):
    """What a GGUF's names become by a family's table: the whole names, and the beginnings of a layer's."""
    names, layer, layers = family.gguf_names
    return set(names.values()), {layer + value for value in layers.values()}


@by_type
def test_a_gguf_of_a_family_feeds_every_row_and_names_nothing_the_plan_has_not(model_type):
    """The table of llama.cpp's names and the sources are written side by side in a family and say the same tensors:
    by Hugging Face's names without what stands in front of them in a file ("transformer." of a GPT-2; a Qwen3.5's GGUF
    is the language model saved alone, "model.")."""
    family = FAMILIES[model_type]
    if family.gguf is None:
        assert model_type in ("mistral", "qwen3_5") and family.renamed
        return
    assert families.GGUF[family.gguf] == (model_type, family)
    header, form, rotary = LAYOUTS[family.arch]
    form = {**form, "bias": False} if model_type == "qwen3" else {**form, "qk_norm": False} if model_type in ("llama", "qwen2", "granite", "smollm3") else form
    rows = tensor_rows(header, form)
    sources = family.sources(Dims(header, form), "model." if family.arch == "qwen35" else "", rotary)
    used = {sources[row.name][0] for row in rows if row.role != TABLE}
    whole, begun = gguf_targets(family)
    reached = lambda name: name in whole or any(name == begin or name.startswith(begin + ".") for begin in begun)
    assert all(reached(name) for name in used), sorted(name for name in used if not reached(name))
    every = {name for name, _ in sources.values()}
    assert whole <= every
    assert all(any(name == begin or name.startswith(begin + ".") for name in every) for begin in begun)


def test_the_refusal_of_a_gguf_names_the_families_that_have_one():
    with pytest.raises(ValueError) as refusal:
        gguf_model({"general.architecture": "mamba"}, {}, 0)
    assert str(refusal.value) == ("This GGUF holds a mamba: only Llama, Granite, SmolLM3, Qwen2, Qwen3, Qwen3.5, LFM2, "
                                  "GPT-2 and GPT-NeoX ones are supported.")
    assert list(families.GGUF) == ["llama", "granite", "smollm3", "qwen2", "qwen3", "qwen35", "lfm2", "gpt2", "gptneox"]


def test_a_llama_with_a_little_more_is_the_llamas_record_but_for_that():
    """A Granite, a SmolLM3, a Qwen2, a Qwen3 and a Mistral are no layouts of their own: what they do not say is the
    Llama's very own."""
    differs = lambda family: {field for field in Family._fields if getattr(family, field) is not getattr(LLAMA, field)
                              and getattr(family, field) != getattr(LLAMA, field)}
    assert differs(GRANITE) == {"title", "check", "scale", "gguf", "gguf_config", "agrees"}
    assert differs(SMOLLM3) == {"title", "check", "options", "gguf", "gguf_config", "agrees"}
    assert differs(QWEN2) == {"title", "check", "gguf", "gguf_stored"}
    assert differs(QWEN3) == {"title", "gguf", "gguf_names", "gguf_stored"}
    assert differs(MISTRAL) == {"title", "renamed", "after", "gguf"}


@by_type
def test_what_only_a_family_says_is_asked_of_no_other(model_type):
    """q's scale, the layers RoPE leaves alone and the two forms, by the names the window always had for them."""
    config = {**BASE, "model_type": model_type, "attention_multiplier": 0.5, "no_rope_layers": [1, 0, 1, 0],
              "layer_types": ["conv", "conv", "full_attention", "full_attention"], "use_parallel_residual": False}
    if family_of(config).arch == "qwen35":
        del config["layer_types"]  # (an LFM2's kinds of layers are none of a Qwen3.5's)
    family = family_of(config)
    assert (llama2_convert.query_scale(config) != 1.0) == (model_type == "granite")
    assert llama2_convert.query_scale(config) == (0.5 * math.sqrt(32) if model_type == "granite" else 1.0)
    assert llama2_convert.unturned_layers(config) == ([1, 3] if model_type == "smollm3" else [])
    assert (llama2_convert.linear_layers(config) is not None) == (family.arch == "qwen35")
    assert (llama2_convert.convolution_layers(config) is not None) == (family.arch == "lfm2")
    assert set(family.form) == {"qwen35": {"linear"}, "lfm2": {"convolution"}}.get(family.arch, set()) <= set(FORM)
    assert set(family.options(config)) == {"smollm3": {"unturned"}, "gpt_neox": {"parallel_residual"}}.get(model_type, set())


# ---- a family added to the table is a family everywhere: a made-up one that is a Llama but for one number of its
# config.json (what q is multiplied by, as a Granite's), from safetensors and from its GGUF
def twice(settings, weights, by):
    tensors, published = hugging_face(settings, weights, settings["shared"])
    return tensors, {**published, "model_type": "twice", "q_multiplier": by, "rms_norm_eps": 1e-5}


@pytest.fixture
def made_up(monkeypatch):
    family = LLAMA._replace(
        title="Twice", scale=lambda config: float(config["q_multiplier"]), gguf="twice",
        gguf_config=lambda key, tensors: {**gguf_config(key, tensors), "q_multiplier": key("q_scale")},
        agrees=lambda own, config: [("multiplier of q", own["q_multiplier"], config.get("q_multiplier"))])
    monkeypatch.setitem(FAMILIES, "twice", family)
    monkeypatch.setitem(families.GGUF, "twice", ("twice", family))
    return family


def test_a_family_put_into_the_table_converts(made_up):
    settings, weights = synthetic_weights(n_kv_heads=2, shared=False)
    tensors, published = twice(settings, weights, 0.5)
    file = safetensors_file(tensors)
    llama = {name: tensor * np.float32(0.5) if ".q_proj." in name else tensor for name, tensor in tensors.items()}
    expected = converted(Safetensors(reader(safetensors_file(llama))), {**published, "model_type": "llama"}, "float32")
    assert converted(Safetensors(reader(file)), published, "float32") == expected
    assert streamed(file, published, "float32", 4096)[0] == expected
    assert expected != converted(Safetensors(reader(file)), {**published, "model_type": "llama"}, "float32")
    # and the refusal of another model names it
    with pytest.raises(ValueError, match="only Llama, Mistral, Granite, SmolLM3, Qwen2, Qwen3, Qwen3.5, LFM2, GPT-2, GPT-NeoX and Twice models"):
        check_config({**BASE, "model_type": "bert"})


def test_a_family_put_into_the_table_reads_its_gguf(made_up):
    settings, weights = synthetic_weights(n_kv_heads=2, shared=False)
    tensors, published = twice(settings, weights, 0.5)
    more = [("twice.attention.layer_norm_rms_epsilon", 6, 1e-5), ("twice.q_scale", 6, 0.5)]
    file, same = gguf_file(tensors, published, settings["vocab_size"], "twice", more=more, turned=True)
    expected = converted(Safetensors(reader(safetensors_file(same))), published, "int8")
    assert bytes(fed(file, "int8").checkpoint) == expected
    got = with_original(file, published, unigram(settings["vocab_size"]), "tokenizer.json", "int8")
    assert bytes(got.checkpoint) == expected and got.options["arch"] == "llama"
    # what it must agree on with the original's config.json is asked too
    with pytest.raises(ValueError, match="its multiplier of q is 0.5 here and 0.25 there"):
        llama2_convert.gguf_weights(file, json.dumps({**published, "q_multiplier": 0.25}))
    with pytest.raises(ValueError, match="its architecture is twice here and llama there"):
        llama2_convert.gguf_weights(file, json.dumps({**published, "model_type": "llama"}))


def test_a_model_type_that_is_no_name_is_refused_as_an_unknown_one():
    """A list or a dict as model_type is hashable by no table: the refusal, not a TypeError from the lookup."""
    for model_type in (["llama"], {"a": 1}):
        config = {**BASE, "model_type": model_type}
        assert family_of(config) is LLAMA and architecture(config) == "llama"
        assert normalize(config)["model_type"] == model_type
        with pytest.raises(ValueError, match="only Llama, Mistral"):
            check_config(config)


def test_two_families_with_one_gguf_name_are_refused_at_the_table():
    """A family that is another's _replace() has that one's GGUF name unless it says its own: the later would win in
    the table GGUF is made from, and the earlier would be unreachable without a word."""
    with pytest.raises(ValueError, match="both the GGUF architecture granite"):
        families.by_gguf({"granite": GRANITE, "granite_again": GRANITE._replace(title="Again")})
    assert families.by_gguf({"llama": LLAMA, "mistral": MISTRAL}) == {"llama": ("llama", LLAMA)}
