"""T331: Gemma 4's layers (Google's gemma-4-E2B-it and E4B-it, their language model), through the converter and the
engine's NumPy forward pass, against a reference written from the Hugging Face tensors the way transformers computes
(conftest.naive_gemma4_logits): heads of their own size in the full-attention layers, the proportional RoPE of those,
v normalized without a weight, the scores not scaled, every layer's number it multiplies the stream by, the norms of
their weight, the cap of the logits. Layers that read another's keys and values (T330) and an embedding of every layer
(T329) are not yet: the converter refuses a model that has them, which every published Gemma 4 has."""
import json
import struct

import numpy as np
import pytest
from conftest import naive_gemma4_logits, pack_tokenizer, tiny_vocab
from made_up import checkpoint, gemma4_model, uniform
from test_gemma3 import TOKENS, close
from test_convert import safetensors_file
from test_gguf import unigram

import llama2_convert
from llama2_convert import check_config, checkpoint_form, checkpoint_size, normalize
from llama2_numpy import FORM, GELU, LAYOUTS, Llama, form_of, tensor_rows
from engine.layout import DOUBLE, NARROW, PROPORTIONAL, SINGLE, WIDE, Dims, double_form, wide_form

MODELS = {
    "small": dict(),
    "a classifier of its own": dict(shared=False),
    "the whole model's checkpoint": dict(whole=True),
    "every layer attends all": dict(kinds="lll"),
    "the last two layers alone attend all": dict(kinds="ssssll", window=3),
    "a window past the text": dict(window=20),
    "keys for every head": dict(n_kv_heads=4, kinds="lssl"),
    "heads of one size": dict(wide_head=16),
    "half of a wide head turns": dict(proportion=0.5, kinds="lssl"),
    "no cap of the logits": dict(softcap=None),
    "a cap that bites": dict(softcap=0.5),
    "another epsilon, the bases the other way round": dict(eps=1e-5, theta=50.0, local_theta=1000000.0),
}
PASSED = ("dtype", "arch", "bias", "head_dim", "local", "rope_theta", "rms_norm_eps", "rotary", "softcap", "wide",
          "double")


def text(config):
    """the language model's config.json, the reference's"""
    return config.get("text_config", config)


def conversion(tensors, config, dtype="float32"):
    """The page's way: the file in its own order, with a tokenizer.json, and the options that come out."""
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    made = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), unigram(text(config)["vocab_size"]),
                                     "tokenizer.json", dtype=dtype, max_seq_len=text(config)["max_position_embeddings"],
                                     start=8 + size)
    made.stream.feed(file[8 + size:])
    made.stream.finish()
    return made


def engine(tensors, config, dtype="float32"):
    """The page's way: the file in its own order, and a model of exactly the options that come out."""
    made = conversion(tensors, config, dtype)
    passed = {key: value for key, value in made.options.items() if key in PASSED}
    return Llama(bytes(made.stream.out), pack_tokenizer(tiny_vocab(text(config)["vocab_size"])), **passed), made


@pytest.mark.parametrize("model", MODELS)
def test_a_gemma4_converts_and_runs_like_transformers(model):
    tensors, config = gemma4_model(**MODELS[model])
    llama, made = engine(tensors, config)
    want = naive_gemma4_logits(tensors, text(config), TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos
    # the form that sizes the file is what the options say
    assert checkpoint_size(made.stream.header, "float32", made.options) == len(made.stream.out)


# every tensor of a layer: changed in one layer, the engine must follow the reference to other numbers
KINDS = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
         "self_attn.q_norm.weight", "self_attn.k_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
         "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "mlp.gate_proj.weight",
         "mlp.up_proj.weight", "mlp.down_proj.weight", "layer_scalar"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which", [0, 2, -1])
def test_every_tensor_is_read_where_transformers_reads_it(kind, which):
    if (kind, which) == ("layer_scalar", -1):
        pytest.skip("the last layer's number scales what the last norm takes, which a norm does not see")
    tensors, config = gemma4_model()
    name = f"model.layers.{which % config['num_hidden_layers']}.{kind}"  # a local layer, a wide one, the last
    changed = dict(tensors)
    rng = np.random.default_rng(7)
    changed[name] = (tensors[name] + rng.standard_normal(tensors[name].shape) * 0.3).astype(np.float32)
    before, after = engine(tensors, config)[0], engine(changed, config)[0]
    want = naive_gemma4_logits(changed, config, TOKENS)
    moved = False
    for pos, token in enumerate(TOKENS):
        was, got = before.forward(token, pos).copy(), after.forward(token, pos)
        assert close(got, want[pos]), pos
        moved |= not np.allclose(got, was, rtol=1e-3, atol=1e-3)
    assert moved, name


def test_the_conversion_tells_the_engine_what_the_file_cannot():
    tensors, config = gemma4_model()
    made = conversion(tensors, config)
    options = made.options
    assert options["arch"] == "gemma4" and options["head_dim"] == 16 and options["rms_norm_eps"] == 1e-6
    assert options["wide"] == {"layers": [2, 5], "head_dim": 32} and "double" not in options
    assert options["local"] == {"layers": [0, 1, 3, 4], "window": 4, "rope_theta": 100.0}
    assert options["rope_theta"] == 1e6 and options["rotary"] == 8 and options["softcap"] == 30.0
    assert form_of(options) == {**FORM, "arch": "gemma4", "head_dim": 16, "wide": {"layers": [2, 5], "head_dim": 32}}
    # heads of one size: no wide layers; no cap: no softcap
    assert "wide" not in conversion(*gemma4_model(wide_head=16)).options
    assert "softcap" not in conversion(*gemma4_model(softcap=None)).options
    # the wide layers' rows are of their own, after the others'
    names = [row.name for row in tensor_rows(made.stream.header, form_of(options))]
    assert names.index("k_norm") < names.index("wq_wide") and "w1_double" not in names and "layer_scale" in names


def test_the_root_of_a_heads_size_is_in_the_norm_of_q_and_the_root_of_the_width_in_the_table():
    """The model scales no score and the engine divides by the root of a head's size: the root goes into the weight of
    q's norm. And the embedding's sqrt(dim) into the table and out of the last norm, as a Gemma 3's."""
    tensors, config = gemma4_model()
    llama = engine(tensors, config)[0]
    assert np.allclose(llama.q_norm[0], tensors["model.layers.0.self_attn.q_norm.weight"].reshape(2, 8).T.reshape(-1) * 4.0)
    wide_norm = tensors["model.layers.2.self_attn.q_norm.weight"].reshape(2, 16).T.reshape(-1)
    assert np.allclose(llama.q_norm_wide[0], wide_norm * np.float32(32 ** 0.5))
    assert np.allclose(llama.k_norm[0], tensors["model.layers.0.self_attn.k_norm.weight"].reshape(2, 8).T.reshape(-1))
    assert np.allclose(llama.token_embedding_table, tensors["model.embed_tokens.weight"] * np.float32(32 ** 0.5))
    assert np.allclose(llama.rms_final_weight, tensors["model.norm.weight"] / np.float32(32 ** 0.5))
    assert llama.layer_scale.shape == (6, 1)


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_a_quantized_gemma4_is_close(dtype):
    """The smaller files hold the same model: the most likely token at every position, logits near."""
    tensors, config = gemma4_model()
    llama = engine(tensors, config, dtype)[0]
    want = naive_gemma4_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        got = llama.forward(token, pos)
        assert np.abs(got - want[pos]).max() < (0.05 if dtype == "float16" else 1.0), pos


# config.json of google/gemma-4-E2B-it at 3e22461f (its text_config, by the keys the converter reads)
E2B = dict(hidden_size=1536, intermediate_size=6144, num_hidden_layers=35, num_attention_heads=8, num_key_value_heads=1,
           head_dim=256, global_head_dim=512, hidden_size_per_layer_input=256, num_kv_shared_layers=20,
           use_double_wide_mlp=True, vocab_size=262144, vocab_size_per_layer_input=262144, max_position_embeddings=131072,
           final_logit_softcapping=30.0, sliding_window=512, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh",
           attention_bias=False, attention_k_eq_v=False, enable_moe_block=False, num_global_key_value_heads=None,
           tie_word_embeddings=True, model_type="gemma4_text",
           layer_types=["full_attention" if layer % 5 == 4 else "sliding_attention" for layer in range(35)],
           rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                               "rope_type": "proportional"},
                            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}})


def refusal(config):
    with pytest.raises(ValueError) as caught:
        check_config(normalize(config))
    return str(caught.value)


def test_what_the_engine_has_not_yet_is_refused_in_words():
    """Every published Gemma 4 reads another layer's keys and values and has an embedding of every layer (T330, T329):
    refused, the first of those first; and what a Gemma 4 of other sizes has (the 12B's keys that are its values, the
    26B's experts)."""
    assert refusal(E2B) == "This model cannot be converted: its layers read the keys and values of other layers."
    assert refusal({**E2B, "num_kv_shared_layers": 0}) == "This model cannot be converted: it has an embedding for every layer."
    whole = {"model_type": "gemma4", "text_config": E2B}
    assert refusal(whole) == refusal(E2B)
    for more in ({"attention_k_eq_v": True}, {"enable_moe_block": True}, {"num_global_key_value_heads": 4}):
        assert refusal({**E2B, **more}) == "This model cannot be converted: its layers are not the ones of a Gemma 4 E2B or E4B."
    assert "the yarn kind" in refusal({**E2B, "rope_parameters": {**E2B["rope_parameters"],
                                                                  "full_attention": {"rope_type": "yarn", "factor": 2.0}}})
    # what it reads of the real one: the form, and the options, past what is refused
    said = normalize({**E2B, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0})
    check_config(said)
    form = checkpoint_form(said, {})
    assert form["wide"] == {"layers": list(range(4, 35, 5)), "head_dim": 512} and form["double"] is None
    assert form["head_dim"] == 256


def test_the_last_layer_attends_all_and_a_size_of_heads_has_two_layers_or_none():
    """transformers makes the last layer a full-attention one whatever config.json says; and a stack of one layer is
    read as no stack, so that a model with one layer of a size of heads is refused (as an LFM2 of one such layer)."""
    said = normalize({**E2B, "layer_types": ["sliding_attention"] * 35})
    assert said["layer_types"][-1] == "full_attention" and said["layer_types"][:-1] == ["sliding_attention"] * 34
    one = {**E2B, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0}
    assert refusal({**one, "layer_types": ["sliding_attention"] * 35}) == \
        "This model cannot be converted: it has fewer than two layers of each size of its heads."
    check_config(normalize({**one, "layer_types": ["sliding_attention"] * 35, "global_head_dim": 256}))
    check_config(normalize({**one, "layer_types": ["full_attention"] * 35}))


def test_the_wide_layers_and_the_double_ffn_are_the_forms_and_a_gemma4s():
    assert wide_form(None) is None and wide_form({"layers": [], "head_dim": 8}) is None
    assert wide_form({"layers": [3, 1, 3], "head_dim": 8}, 4) == {"layers": (1, 3), "head_dim": 8}
    assert double_form(None) is None and double_form([]) is None and double_form([2, 0], 3) == (0, 2)
    for wrong in ({"layers": [4], "head_dim": 8}, {"layers": [-1], "head_dim": 8}, {"layers": [0], "head_dim": 7}):
        with pytest.raises(ValueError):
            wide_form(wrong, 4)
    with pytest.raises(ValueError):
        double_form([4], 4)
    with pytest.raises(ValueError):  # a Gemma 3 has no such layers
        Dims([32, 64, 4, 4, 1, 320, 24], {"arch": "gemma3", "head_dim": 16, "double": [1]})
    dims = Dims([32, 64, 4, 4, 1, 320, 24], {"arch": "gemma4", "head_dim": 16, "wide": {"layers": [1, 3], "head_dim": 32},
                                              "double": [2, 3]})
    assert (dims.layers(NARROW), dims.layers(WIDE), dims.layers(SINGLE), dims.layers(DOUBLE)) == ([0, 2], [1, 3], [0, 1], [2, 3])
    assert [dims.head_of(layer) for layer in range(4)] == [16, 32, 16, 32]
    shapes = {row.name: row.shape for row in dims.rows()}
    assert shapes["wq"] == (2, 64, 32) and shapes["wq_wide"] == (2, 128, 32) and shapes["wo_wide"] == (2, 32, 128)
    assert shapes["w1"] == (2, 64, 32) and shapes["w2_double"] == (2, 32, 128) and shapes["layer_scale"] == (4, 1)
    assert shapes["freq_cis_real"] == (24, 16)
    assert LAYOUTS["gemma4"].rope == PROPORTIONAL and LAYOUTS["gemma4"].activation == GELU
    assert LAYOUTS["gemma4"].value_norm and LAYOUTS["gemma4"].wide
    assert not any(layout.value_norm or layout.wide for arch, layout in LAYOUTS.items() if arch != "gemma4")


def doubled(seed, double, poked=False):
    """A Gemma 4 file of 3 layers (the last a wide one) whose layers of double have their FFN twice as wide, with the
    rows past hidden_dim of the gate and up matrices and the columns past it of the down matrix 0 (what they add is
    then GELU(0) * 0, nothing): the same model as the file without, whatever else. poked: one of those rows not 0."""
    header, form = [32, 48, 3, 4, 1, 320, 24], {"arch": "gemma4", "head_dim": 8, "wide": {"layers": [2], "head_dim": 16}}
    plain = checkpoint(header, form, tensor=uniform(seed))
    single = Llama(plain, pack_tokenizer(tiny_vocab(320)), **form, rotary=4)
    rows = Dims(header, {**form, "double": double}).rows()
    values = {}
    for row in rows:
        if row.name.endswith("_double"):
            narrow = getattr(single, row.name[:2])[list(double)]
            values[row.name] = np.zeros(row.shape, dtype=np.float32)
            values[row.name][(slice(None), slice(None, narrow.shape[1]), slice(None, narrow.shape[2]))] = narrow
        elif row.name in ("w1", "w2", "w3"):
            values[row.name] = getattr(single, row.name)[[l for l in range(3) if l not in double]]
        else:
            values[row.name] = getattr(single, row.name)
    if poked:  # (the last of the rows past hidden_dim: of the gate and up matrices, and the column of down)
        values["w1_double"][0][-1], values["w3_double"][0][-1], values["w2_double"][0][:, -1] = 0.5, 0.5, 0.5
    wide = checkpoint(header, {**form, "double": double}, tensor=lambda row: values[row.name])
    return single, Llama(wide, pack_tokenizer(tiny_vocab(320)), **form, rotary=4, double=double)


@pytest.mark.parametrize("double", [[1], [0, 2], [0, 1, 2]])
def test_an_ffn_twice_as_wide_is_read_where_its_layers_are(double):
    single, wide = doubled(3, double)
    for pos, token in enumerate(TOKENS):
        assert np.allclose(wide.forward(token, pos), single.forward(token, pos), rtol=1e-5, atol=1e-5), pos
    # and the rows past hidden_dim are read: one of them made other moves the logits
    single, poked = doubled(3, double, poked=True)
    assert not np.allclose(poked.forward(TOKENS[0], 0), single.forward(TOKENS[0], 0), rtol=1e-4, atol=1e-4)


def test_no_other_architecture_is_changed_by_the_layers_of_another_width():
    """The NumPy forward pass of every other architecture keeps one array of keys and values for its layers, and a
    Gemma 4 one for each (of the size of its heads)."""
    tensors, config = gemma4_model()
    llama = engine(tensors, config)[0]
    assert [keys.shape[-1] for keys in llama.key_cache] == [16, 16, 32, 16, 16, 32]
    assert isinstance(engine(*gemma4_model(wide_head=16))[0].key_cache, list) is False


def test_the_pages_engine_is_handed_no_gemma4_until_it_reads_one():
    """T331.2 has forward.js read what a Gemma 4's layers are: until then no plan of one is made, where it would run as
    another model without a word."""
    from engine.plan import Settings, forward_plan
    dims = Dims([32, 64, 4, 4, 1, 320, 24], {"arch": "gemma4", "head_dim": 16})
    with pytest.raises(ValueError, match="does not run a Gemma 4 yet"):
        forward_plan(dims, "int8", lambda offset, length: bytes(length), Settings(int8=True, kv_start=256))
