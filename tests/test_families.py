"""convert/families/ (T359): what differs between one model_type of config.json and another is one record in one
table, which the converter's shared flow looks up. These hold the table to what the flow asks of it; what each family
converts to is its own tests' (test_granite.py, test_smollm3.py, test_qwen35.py, test_lfm2.py, test_gpt2.py,
test_neox.py, test_gguf.py), and the net's kinds "layouts" and "python" compare the answers between two trees."""
import json
import math
from typing import NamedTuple

import numpy as np
import pytest
from conftest import synthetic_weights
from test_convert import converted, hugging_face, reader, safetensors_file, streamed
from test_gguf import fed, gguf_file, unigram, with_original

import llama2_convert
import llama2_numpy
from convert import families
from convert.families.llama import GRANITE, LLAMA, MISTRAL, QWEN2, QWEN3, SMOLLM3, gguf_config
from convert.gguf import MARKS
from llama2_convert import (FAMILIES, Family, Safetensors, check_config, conversion_plan, family_of,
                            gguf_model, normalize)
from llama2_numpy import FORM, TABLE, Dims, tensor_rows

# T640.3: what llama.cpp stores of a family otherwise than Hugging Face has it, marked in the tensor's entry of the
# header, written a second time on purpose: a tensor's name after its layer (or its whole name outside the layers) and
# the marks of its entry. A tensor of a family's table that is not in its family's marks has none
TURNED = {f"{name}.{part}": {"turned": heads} for name, heads in (("attn_q", 8), ("attn_k", 2)) for part in ("weight", "bias")}
QWEN35_MARKS = {**{name: {"done": "one"} for name in ("output_norm.weight", "attn_norm.weight", "ffn_norm.weight",
                                                     "post_attention_norm.weight", "attn_q_norm.weight",
                                                     "attn_k_norm.weight")},
                "ssm_a": {"done": "decay", "tiled": (0, 2, 2, 1, 0)},  # (a whole name, as ssm_dt.bias)
                "ssm_conv1d.weight": {"shape": [64, 1, 8], "tiled": (16, 2, 2, 4, 0)},
                "attn_qkv.weight": {"tiled": (16, 2, 2, 4, 0)}, "attn_gate.weight": {"tiled": (0, 2, 2, 4, 0)},
                "ssm_alpha.weight": {"tiled": (0, 2, 2, 1, 0)}, "ssm_beta.weight": {"tiled": (0, 2, 2, 1, 0)},
                "ssm_dt.bias": {"tiled": (0, 2, 2, 1, 0)}, "ssm_out.weight": {"tiled": (0, 2, 2, 4, 1)}}
LFM2_MARKS = {"shortconv.conv.weight": {"shape": [64, 1, 8]}}
GEMMA3_MARKS = {name: {"done": "one"} for name in ("output_norm.weight", "attn_norm.weight", "post_attention_norm.weight",
                                                   "ffn_norm.weight", "post_ffw_norm.weight", "attn_q_norm.weight",
                                                   "attn_k_norm.weight")}
# T649.4: a PLaMo 3's norms with what the model adds (the two behind by their whole names, without a ".weight")
PLAMO3_MARKS = {**{name: {"done": "one"} for name in ("output_norm.weight", "attn_norm.weight", "ffn_norm.weight",
                                                    "attn_q_norm.weight", "attn_k_norm.weight")},
                "post_attention_norm": {"done": "plus"}, "post_ffw_norm": {"done": "plus"}}
GPT2_MARKS = {f"{name}.weight": {"shape": [8, 64], "transposed": True} for name in ("attn_qkv", "attn_output", "ffn_up", "ffn_down")}
NEOX_MARKS = {f"attn_qkv.{part}": {"split": 8} for part in ("weight", "bias")}


class Facts(NamedTuple):
    title: str
    arch: str
    gguf: str | None  # the GGUF architecture
    classifier: str
    true: set  # the facts that are true of it (FLAGS)
    # T640.5, of its GGUF (none where it has no GGUF): what stands before the Hugging Face names its table of
    # llama.cpp's names gives, the rows of its layout that table has no name for, and the marks above
    prefix: str | None = None
    unnamed: set | None = None
    marked: dict | None = None


LLAMAS = {"free_heads", "scaled", "rotatable", "rms_norm"}
# the families that are no other's name and whose GGUF the converter does not read (yet)
NO_GGUF_YET = set()
# The table as it stands, written a second time on purpose (as the order of the rows is in test_layout.py): a
# family's fact changed by mistake changes which models convert and how, and few of the facts have a test of their
# own. By model_type, in the order of the table (the order the refusals name the titles in)
FACTS = {
    "llama": Facts("Llama", "llama", "llama", "lm_head.weight", LLAMAS, "", {"qk_norm"}, TURNED),
    "mistral": Facts("Mistral", "llama", None, "lm_head.weight", {"renamed", *LLAMAS}),
    "granite": Facts("Granite", "llama", "granite", "lm_head.weight", LLAMAS, "", {"qk_norm"}, TURNED),
    "smollm3": Facts("SmolLM3", "llama", "smollm3", "lm_head.weight", LLAMAS, "", {"qk_norm"}, TURNED),
    "qwen2": Facts("Qwen2", "llama", "qwen2", "lm_head.weight", LLAMAS, "", {"qk_norm"}, {}),
    "qwen3": Facts("Qwen3", "llama", "qwen3", "lm_head.weight", LLAMAS, "", {"bias"}, {}),
    "qwen3_5_text": Facts("Qwen3.5", "qwen35", "qwen35", "lm_head.weight", {"free_heads", "rotatable", "rms_norm", "partly"},
                          "model.", set(), QWEN35_MARKS),
    "qwen3_5": Facts("Qwen3.5", "qwen35", None, "lm_head.weight", {"renamed", "free_heads", "rotatable", "rms_norm", "partly"}),
    "lfm2": Facts("LFM2", "lfm2", "lfm2", "lm_head.weight", {"rms_norm"}, "", set(), LFM2_MARKS),
    "gemma3_text": Facts("Gemma 3", "gemma3", "gemma3", "lm_head.weight", {"free_heads", "rms_norm"}, "", set(), GEMMA3_MARKS),
    "gemma4_text": Facts("Gemma 4", "gemma4", "gemma4", "lm_head.weight", {"free_heads", "rms_norm"}, "model.", set(), {}),
    "gemma4": Facts("Gemma 4", "gemma4", None, "lm_head.weight", {"renamed", "free_heads", "rms_norm"}),
    "spark2_5": Facts("Spark-X2.5", "spark", "spark2_5", "lm_head.weight", {"free_heads", "rms_norm", "partly"}, "", set(), {}),
    "plamo3": Facts("PLaMo 3", "plamo3", "plamo3", "lm_head.weight", {"free_heads", "scaled", "rms_norm"}, "", set(),
                    PLAMO3_MARKS),
    "gpt2": Facts("GPT-2", "gpt2", "gpt2", "lm_head.weight", {"fixed_context"}, "", set(), GPT2_MARKS),
    "gpt_neox": Facts("GPT-NeoX", "neox", "gptneox", "embed_out.weight", {"partly"}, "", set(), NEOX_MARKS),
}


def listed(titles):
    """Titles as the refusals list them, "A, B and C", each once and in the order of FACTS (T640.5: the sentences are
    made of the table's second copy here, not copied out with the titles in them)."""
    titles = list(dict.fromkeys(titles))
    return f"{', '.join(titles[:-1])} and {titles[-1]}"


TITLES = [facts.title for facts in FACTS.values()]
# the refusals: of a config.json no family takes, and of a GGUF no family reads
SUPPORTED = listed(TITLES)
GGUF_SUPPORTED = listed(facts.title for facts in FACTS.values() if facts.gguf)
# the families that have one, by their GGUF architecture
BY_GGUF = {facts.gguf: facts for facts in FACTS.values() if facts.gguf}
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
    "gemma3": ([256, 768, 6, 4, 1, -2048, 320], {"arch": "gemma3", "head_dim": 128}, 0),
    "gemma4": ([256, 768, 6, 4, 1, -2048, 320], {"arch": "gemma4", "head_dim": 128,
                                                 "wide": {"layers": [2, 5], "head_dim": 256}, "double": [4, 5],
                                                 "kv_shared": {"layers": [4, 5], "from": [3, 2]}, "per_layer": 16}, 0),
    # (T341: a number for each layer, RoPE turning a part of some layers' heads and all of the others')
    "spark": ([256, 768, 4, 4, 2, -2048, 320], {"arch": "spark", "head_dim": 128}, (128, 128, 128, 32)),
    "plamo3": ([256, 768, 8, 4, 1, -2048, 320], {"arch": "plamo3", "head_dim": 128}, 0),
}
BASE = {"hidden_size": 256, "intermediate_size": 768, "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 2048,
        "max_position_embeddings": 4096}
by_type = pytest.mark.parametrize("model_type", list(FACTS))


def test_the_table_is_the_model_types_the_converter_takes():
    assert list(FAMILIES) == list(FACTS)
    assert all(isinstance(family, Family) for family in FAMILIES.values())


@by_type
def test_a_model_type_resolves_to_its_family_and_the_family_says_what_it_is(model_type):
    title, arch, gguf, classifier, true, *_ = FACTS[model_type]
    family = family_of({"model_type": model_type})
    assert family is FAMILIES[model_type]
    assert (family.title, family.arch, family.gguf, family.classifier) == (title, arch, gguf, classifier)
    assert {flag for flag in FLAGS if getattr(family, flag)} == true
    assert family_of({"model_type": model_type}).arch == arch
    assert all(getattr(family, flag) in (True, False) for flag in FLAGS)


def test_an_unknown_model_type_is_refused_in_the_words_it_was():
    for model_type, said in (("bert", "a bert"), (None, "a model of unknown type")):
        config = {**BASE, **({"model_type": model_type} if model_type else {})}
        with pytest.raises(ValueError) as refusal:
            check_config(normalize(config))
        assert str(refusal.value) == f"This model cannot be converted: it is {said}, and only {SUPPORTED} models are supported."
        # (its layout and its plan are a Llama's until then: nothing else is asked of an unknown name)
        assert family_of(config) is LLAMA and family_of(config).arch == "llama"


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
    assert all(isinstance(name, str) and (transform is None or isinstance(transform, tuple) or callable(transform))
               for name, transform in sources.values())
    # and the plan of the layout is made of them: a form says its layout, and the layout's family is this one's
    assert families.of_layout(family.arch).sources is family.sources
    d = Dims(header, form)
    plan, shapes = conversion_plan(header, form, "", rotary)
    for row, parts in zip(rows, plan):
        if row.role != TABLE:
            name, transform = sources[row.name]
            of = transform if callable(transform) else lambda layer: transform
            assert parts == ([(name.format(layer), of(layer)) for layer in d.layers(row.per)] if row.per else [(name, transform)])
    # the classifier's name is the one checkpoint_header() looks for
    assert sources["wcls"][0] == family.classifier


def test_a_source_of_a_stack_may_be_made_otherwise_in_each_layer(monkeypatch):
    """T341: a transform that is a function of the layer's number gives each layer of the stack its own (a
    Spark-X2.5's q and k are interleaved over the width RoPE turns of that layer); a tuple is every layer's."""
    header, form, rotary = LAYOUTS["llama"]
    name, transform = LLAMA.sources(Dims(header, form), "", rotary)["wq"]
    assert transform == ("permute", 4)
    every = lambda d, prefix, rotary: {**LLAMA.sources(d, prefix, rotary), "wq": (name, lambda layer: transform if layer % 2 else None)}
    monkeypatch.setitem(families.LAYOUTS, "llama", LLAMA._replace(sources=every))
    plan, _ = conversion_plan(header, form, "", rotary)
    wq, wk = ([row.name for row in tensor_rows(header, form)].index(name) for name in ("wq", "wk"))
    assert plan[wq] == [(name.format(layer), transform if layer % 2 else None) for layer in range(header[2])]
    assert plan[wk] == [(f"model.layers.{layer}.self_attn.k_proj.weight", ("permute", 2)) for layer in range(header[2])]


def test_every_layout_of_the_engine_has_a_family_and_the_first_of_the_table_stands_for_it():
    assert set(families.LAYOUTS) == set(LAYOUTS) == {family.arch for family in FAMILIES.values()}
    assert set(families.LAYOUTS) == set(llama2_numpy.LAYOUTS)
    assert families.LAYOUTS["llama"] is LLAMA and families.LAYOUTS["qwen35"] is FAMILIES["qwen3_5_text"]
    # (T393: a name that is no layout's is refused in the engine's words, not read as a Llama: name_prefix() reached it)
    with pytest.raises(ValueError, match="^There is no architecture called 'a layout nobody wrote': llama, "):
        families.of_layout("a layout nobody wrote")
    with pytest.raises(ValueError, match="^There is no architecture called None: "):
        families.of_layout(None)


def gguf_targets(family):
    """What a GGUF's names become by a family's table: the whole names, and the beginnings of a layer's."""
    names, layer, layers = family.gguf_names
    return set(names.values()), {layer + value for value in layers.values()}


@by_type
def test_a_gguf_of_a_family_feeds_every_row_and_names_nothing_the_plan_has_not(model_type):
    """The table of llama.cpp's names and the sources are written side by side in a family and say the same tensors:
    by Hugging Face's names without what stands in front of them in a file ("transformer." of a GPT-2; a Qwen3.5's GGUF
    and a Gemma 4's are the language model saved alone, "model.")."""
    family = FAMILIES[model_type]
    *_, prefix, unnamed, marked = FACTS[model_type]
    if family.gguf is None:
        # (a Gemma 4's whole model, T529: its GGUF is its language model's, gemma4_text)
        assert (family.renamed or model_type in NO_GGUF_YET) and prefix is unnamed is marked is None
        return
    assert families.GGUF[family.gguf] == (model_type, family)
    header, form, rotary = LAYOUTS[family.arch]
    form = {**form, **{row: False for row in unnamed}}
    rows = tensor_rows(header, form)
    sources = family.sources(Dims(header, form), prefix, rotary)
    used = {sources[row.name][0] for row in rows if row.role != TABLE}
    whole, begun = gguf_targets(family)
    reached = lambda name: name in whole or any(name == begin or name.startswith(begin + ".") for begin in begun)
    assert all(reached(name) for name in used), sorted(name for name in used if not reached(name))
    every = {name for name, _ in sources.values()}
    assert whole <= every
    assert all(any(name == begin or name.startswith(begin + ".") for name in every) for begin in begun)


def gguf_of_every_name(family):
    """Every tensor a family's table of llama.cpp's names reads, of layer 0, as F32 (64, 8): a layer's name alone
    (a whole one: a Qwen3.5's ssm_a, or ssm_dt.bias with its dot), and with a weight and a bias after it where it has
    no dot."""
    names, _, layers = family.gguf_names
    every = [*names, *(f"blk.0.{name}{part}" for name in layers for part in ("", *(() if "." in name else (".weight", ".bias"))))]
    return {name: {"type": 0, "shape": [64, 8], "offset": 0} for name in every}


@pytest.mark.parametrize("arch", list(families.GGUF))
def test_a_gguf_of_a_family_is_marked_where_llama_cpp_stores_it_otherwise(arch):
    """The marks of the header (the GGUF's shape put back, and what the plan undoes or leaves out: turned, split,
    transposed, tiled, done) by the second copy of FACTS: a mark a family's table spells wrongly, or a norm it leaves
    unmarked, changes a GGUF's numbers and is caught here, not only by a model's test of its own."""
    _, family = families.GGUF[arch]
    key = lambda name: f"{arch}.{name}"
    metadata = {"general.architecture": arch, key("block_count"): 4, key("embedding_length"): 64,
                key("attention.head_count"): 8, key("attention.head_count_kv"): [0, 0, 2, 2] if arch == "lfm2" else 2,
                key("feed_forward_length"): 128, key("context_length"): 256, key("attention.key_length"): 8,
                key("ssm.time_step_rank"): 4, key("ssm.group_count"): 2, key("ssm.state_size"): 4,
                key("ssm.inner_size"): 16, key("ssm.conv_kernel"): 4, key("full_attention_interval"): 4}
    tensors = gguf_of_every_name(family)
    header, _ = gguf_model(metadata, tensors, 0)
    names, _, _ = family.gguf_names
    after = lambda name: name if name in names else ".".join(name.split(".")[2:])
    marks = {after(name): {mark: value for mark, value in header[target].items() if mark not in ("dtype", "data_offsets")
                           and not (mark == "shape" and value == tensors[name]["shape"])}
             for name in tensors for target in [family_target(family, name)]}
    assert {name: marked for name, marked in marks.items() if marked} == BY_GGUF[arch].marked


@pytest.mark.parametrize("arch", list(families.GGUF))
def test_the_marks_of_a_family_name_tensors_of_its_table_and_marks_there_are(arch):
    """T640.3: a name of gguf_marks() that gguf_names has not (a spelling of it gone wrong) would mark nothing and say
    nothing; a mark MARKS has not would be a KeyError of the first GGUF that has the tensor only. Said here of every
    family, whichever GGUF its own tests write. (A Qwen3.5 of two value heads to a key head: the marks of the tiles.)"""
    _, family = families.GGUF[arch]
    config = {**BASE, "num_key_value_heads": 2, "linear_num_key_heads": 2, "linear_num_value_heads": 4}
    names, _, layers = family.gguf_names
    read = lambda name: name in names or name in layers or (name.rsplit(".", 1)[0] in layers and name.rsplit(".", 1)[-1]
                                                             in ("weight", "bias"))
    marks = family.gguf_marks(config)
    assert all(read(name) for name in marks), sorted(name for name in marks if not read(name))
    assert {mark for marked in marks.values() for mark in marked} <= set(MARKS)


def family_target(family, name):
    """Hugging Face's name of a GGUF's tensor by a family's table, as gguf_model() looks it up."""
    names, layer, layers = family.gguf_names
    if name in names:
        return names[name]
    parts = name.split(".")
    whole = ".".join(parts[2:])
    return layer.format(parts[1]) + (layers[whole] if whole in layers else f"{layers[parts[2]]}.{parts[3]}")


def test_the_refusal_of_a_gguf_names_the_families_that_have_one():
    with pytest.raises(ValueError) as refusal:
        gguf_model({"general.architecture": "mamba"}, {}, 0)
    assert str(refusal.value) == f"This GGUF holds a mamba: only {GGUF_SUPPORTED} ones are supported."
    assert list(families.GGUF) == list(BY_GGUF)


def test_a_llama_with_a_little_more_is_the_llamas_record_but_for_that():
    """A Granite, a SmolLM3, a Qwen2, a Qwen3 and a Mistral are no layouts of their own: what they do not say is the
    Llama's very own."""
    differs = lambda family: {field for field in Family._fields if getattr(family, field) is not getattr(LLAMA, field)
                              and getattr(family, field) != getattr(LLAMA, field)}
    assert differs(GRANITE) == {"title", "check", "scale", "gguf", "gguf_config", "agrees"}
    assert differs(SMOLLM3) == {"title", "check", "options", "gguf", "gguf_config", "agrees"}
    assert differs(QWEN2) == {"title", "check", "gguf", "gguf_marks"}
    assert differs(QWEN3) == {"title", "gguf", "gguf_names", "gguf_marks"}
    assert differs(MISTRAL) == {"title", "renamed", "after", "gguf"}


@by_type
def test_what_only_a_family_says_is_asked_of_no_other(model_type):
    """q's scale, the layers RoPE leaves alone and the two forms, as the family answers them."""
    config = {**BASE, "model_type": model_type, "attention_multiplier": 0.5, "no_rope_layers": [1, 0, 1, 0],
              "layer_types": ["conv", "conv", "full_attention", "full_attention"], "use_parallel_residual": False}
    if family_of(config).arch == "qwen35":
        del config["layer_types"]  # (an LFM2's kinds of layers are none of a Qwen3.5's)
    family = family_of(config)
    assert (family.scale(config) != 1.0) == (model_type == "granite")
    assert family.scale(config) == (0.5 * math.sqrt(32) if model_type == "granite" else 1.0)
    assert family.options(config).get("unturned", []) == ([1, 3] if model_type == "smollm3" else [])
    assert ("linear" in family.form and family.form["linear"](config) is not None) == (family.arch == "qwen35")
    assert ("convolution" in family.form and family.form["convolution"](config) is not None) == (family.arch == "lfm2")
    assert set(family.form) == {"qwen35": {"linear"}, "lfm2": {"convolution"},
                                "gemma4": {"wide", "double", "kv_shared", "per_layer"}}.get(family.arch, set()) <= set(FORM)
    assert set(family.options(config)) == {"smollm3": {"unturned"}, "gpt_neox": {"parallel_residual"},
                                           "gemma4_text": {"rotary"}, "gemma4": {"rotary"}}.get(model_type, set())


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
    with pytest.raises(ValueError) as refusal:
        check_config({**BASE, "model_type": "bert"})
    assert str(refusal.value).endswith(f"only {listed([*TITLES, 'Twice'])} models are supported.")


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
        assert family_of(config) is LLAMA and family_of(config).arch == "llama"
        assert normalize(config)["model_type"] == model_type
        with pytest.raises(ValueError, match="only Llama, Mistral"):
            check_config(config)


def test_two_families_with_one_gguf_name_are_refused_at_the_table():
    """A family that is another's _replace() has that one's GGUF name unless it says its own: the later would win in
    the table GGUF is made from, and the earlier would be unreachable without a word."""
    with pytest.raises(ValueError, match="both the GGUF architecture granite"):
        families.by_gguf({"granite": GRANITE, "granite_again": GRANITE._replace(title="Again")})
    assert families.by_gguf({"llama": LLAMA, "mistral": MISTRAL}) == {"llama": ("llama", LLAMA)}


BIASES, NORMS = "model.layers.0.self_attn.q_proj.bias", "model.layers.0.self_attn.q_norm.weight"


@by_type
def test_the_biases_and_the_norms_of_the_heads_are_said_by_the_llamas_alone(model_type):
    """T369 (T359.3's finding): FORM's "bias" and "qk_norm" are rows a Llama's layout has where the file has their
    tensors, and the form of no other layout reads them. They were asked of every source by a Llama's names: a Qwen3.5
    saved as a language model has a tensor of the norm's name, and its options said qk_norm where those of the same
    model saved with its vision model (the names begin otherwise) said nothing."""
    family = FAMILIES[model_type]
    for source in ({}, {BIASES: 1}, {NORMS: 1}, {BIASES: 1, NORMS: 1}, {f"model.language_model.{NORMS[6:]}": 1}):
        found = family.found(source)
        if family.arch == "llama":
            assert found == {"bias": BIASES in source, "qk_norm": NORMS in source}
        else:
            assert found == {}
    assert (family.found is LLAMA.found) == (family.arch == "llama")


def test_the_form_of_a_qwen35_and_of_an_lfm2_says_no_qk_norm_whatever_the_names():
    qwen35 = {**BASE, "model_type": "qwen3_5_text", "num_hidden_layers": 8, "num_key_value_heads": 2, "head_dim": 128}
    lfm2 = {**BASE, "model_type": "lfm2", "num_hidden_layers": 8, "num_key_value_heads": 4, "layer_types": ["conv"] * 4 + ["full_attention"] * 4,
            "block_ff_dim": 768, "block_auto_adjust_ff_dim": False}
    for config in (qwen35, lfm2):
        for source in ({}, {NORMS: 1, BIASES: 1}):
            form = llama2_convert.checkpoint_form(normalize(config), source)
            assert (form["bias"], form["qk_norm"]) == (False, False), (config["model_type"], source)
    # and a Llama's is the source's, as it was
    for source in ({}, {NORMS: 1}, {BIASES: 1}):
        form = llama2_convert.checkpoint_form(normalize({**BASE, "model_type": "qwen3"}), source)
        assert (form["bias"], form["qk_norm"]) == (BIASES in source, NORMS in source)
    assert list(llama2_convert.checkpoint_form(normalize({**BASE, "model_type": "llama"}), {})) == \
        ["bias", "arch", "qk_norm", "head_dim", "linear", "rotated", "convolution"]


# ---- T387: three rows of the families that no real config steps on (a real config.json writes the key)
def test_a_gpt2_config_without_n_inner_has_an_ffn_four_times_its_width():
    from convert.families.gpt2 import gpt2_names
    assert gpt2_names({"n_embd": 64, "n_layer": 2, "n_head": 4})["intermediate_size"] == 256
    assert gpt2_names({"n_embd": 64, "n_layer": 2, "n_head": 4, "n_inner": 100})["intermediate_size"] == 100
    assert gpt2_names({"n_embd": 64, "n_layer": 2, "n_head": 4, "n_inner": None})["intermediate_size"] == 256


def test_the_tying_of_the_classifier_where_a_config_does_not_say():
    from convert.families.gpt2 import gpt2_names, neox_names
    # GPT-2 ties it to the embedding unless it says otherwise, GPT-NeoX does not
    assert gpt2_names({"n_embd": 64})["tie_word_embeddings"] is True
    assert neox_names({})["tie_word_embeddings"] is False
    assert neox_names({"tie_word_embeddings": True})["tie_word_embeddings"] is True


def test_a_neox_config_that_says_not_how_much_of_a_head_turns_turns_a_quarter():
    """T589: as transformers' GPTNeoXConfig (rotary_pct 0.25), where rotary_dim() alone would read the whole head."""
    from convert.families.family import rotary_dim
    from convert.families.gpt2 import neox_names
    config = {"hidden_size": 64, "num_attention_heads": 4}
    assert rotary_dim(neox_names(config)) == 4
    assert rotary_dim(neox_names({**config, "rotary_pct": 1.0})) == 16
    assert rotary_dim(neox_names({**config, "partial_rotary_factor": 0.5})) == 8


def test_a_neox_that_rotates_none_of_a_head_is_refused():
    from convert.families.gpt2 import neox_check
    assert FAMILIES["gpt_neox"].check is neox_check  # (the table's row is the function tested here)
    config = {"hidden_size": 64, "num_attention_heads": 4, "hidden_act": "gelu"}
    neox_check({**config, "rotary_pct": 0.25})  # (4 of a head's 16: taken)
    with pytest.raises(ValueError, match="it rotates none of each head"):
        neox_check({**config, "rotary_pct": 0.0})
    with pytest.raises(ValueError, match="it rotates none of each head"):
        neox_check({**config, "rotary_pct": 0.05})  # (0.8 of a value: none)
