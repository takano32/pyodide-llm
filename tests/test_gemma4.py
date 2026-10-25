"""T.331: Gemma 4's layers (Google's gemma-4-E2B-it and E4B-it, their language model), through the converter and the
engine's NumPy forward pass, against a reference written from the Hugging Face tensors the way transformers computes
(conftest.naive_gemma4_logits): heads of their own size in the full-attention layers, the proportional RoPE of those,
v normalized without a weight, the scores not scaled, every layer's number it multiplies the stream by, the norms of
their weight, the cap of the logits. Layers that read another's keys and values (T.330) and an embedding of every layer
(T.329) are not yet: the converter refuses a model that has them, which every published Gemma 4 has."""
import json
import struct

import numpy as np
import pytest
from conftest import naive_gemma4_logits, pack_tokenizer, tiny_vocab
from made_up import checkpoint, gemma4_model, uniform
from test_gemma3 import TOKENS, close, safetensors_conversion
from test_convert import safetensors_file
from test_gguf import metadata_value, q8_0_blocks, unigram, with_original

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
    # T.330: the last layers read the keys and values of the last of their kind before them
    "the last four share": dict(kinds="ssllsssl", sharing=4),
    "the last four share, their FFN twice as wide": dict(kinds="ssllsssl", sharing=4, double=True),
    "all but the first two of a kind share": dict(kinds="slslssssl", sharing=5, double=True),
    "sharing, heads of one size": dict(kinds="ssllsl", sharing=2, wide_head=16, double=True),
    # T.329: every layer's input of its own
    "an embedding of every layer": dict(per_layer=8),
    "all of it, as E2B has it": dict(kinds="sslsslsslssl", sharing=5, double=True, per_layer=8, whole=True),
    "all of it, as E4B has it, a classifier of its own": dict(kinds="sslsslsslssl", sharing=5, per_layer=16, shared=False),
}
PASSED = ("dtype", "arch", "bias", "head_dim", "local", "rope_theta", "rms_norm_eps", "rotary", "softcap", "wide",
          "double", "kv_shared", "per_layer")


@pytest.fixture(autouse=True)
def page_reads_embeddings(monkeypatch):
    """(T.329.2 is when the page's forward pass reads them: the converter refuses them until then, but here)"""
    import convert.families.gemma4
    monkeypatch.setattr(convert.families.gemma4, "PAGE_READS_EMBEDDINGS", True)


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


def test_what_the_engine_has_not_yet_is_refused_in_words(monkeypatch):
    """Every published Gemma 4 has an embedding of every layer, which the page's forward pass reads from T.329.2:
    refused until then; and what a Gemma 4 of other sizes has (the 12B's keys that are its values, the 26B's
    experts), and a table of another vocabulary."""
    import convert.families.gemma4
    check_config(normalize(E2B))
    assert refusal({**E2B, "vocab_size_per_layer_input": 1000}) == "This model cannot be converted: it has an embedding for every layer."
    monkeypatch.setattr(convert.families.gemma4, "PAGE_READS_EMBEDDINGS", False)
    assert refusal(E2B) == "This model cannot be converted: it has an embedding for every layer."
    monkeypatch.setattr(convert.families.gemma4, "PAGE_READS_EMBEDDINGS", True)
    whole = {"model_type": "gemma4", "text_config": {**E2B, "vocab_size_per_layer_input": 1000}}
    assert refusal(whole) == refusal({**E2B, "vocab_size_per_layer_input": 1000})
    for more in ({"attention_k_eq_v": True}, {"enable_moe_block": True}, {"num_global_key_value_heads": 4}):
        assert refusal({**E2B, **more}) == "This model cannot be converted: its layers are not the ones of a Gemma 4 E2B or E4B."
    assert "the yarn kind" in refusal({**E2B, "rope_parameters": {**E2B["rope_parameters"],
                                                                  "full_attention": {"rope_type": "yarn", "factor": 2.0}}})
    # what it reads of the real one: the form, and the options, past what is refused
    said = normalize({**E2B, "hidden_size_per_layer_input": 0})
    check_config(said)
    form = checkpoint_form(said, {})
    assert form["wide"] == {"layers": list(range(4, 35, 5)), "head_dim": 512} and form["double"] == list(range(15, 35))
    assert form["head_dim"] == 256
    # T.330: the last 20 read layer 13 (the local ones) and 14 (those that attend all), as transformers' layers do
    assert form["kv_shared"] == {"layers": list(range(15, 35)), "from": [14 if layer % 5 == 4 else 13 for layer in range(15, 35)]}
    # and E4B's: 18 of 42, from 22 and 23, an FFN of one width
    e4b = normalize({**E2B, "hidden_size": 2560, "intermediate_size": 10240, "num_hidden_layers": 42, "num_key_value_heads": 2,
                     "num_kv_shared_layers": 18, "use_double_wide_mlp": False, "hidden_size_per_layer_input": 0,
                     "layer_types": ["full_attention" if layer % 6 == 5 else "sliding_attention" for layer in range(42)]})
    check_config(e4b)
    form = checkpoint_form(e4b, {})
    assert form["double"] is None and form["kv_shared"]["from"] == [23 if layer % 6 == 5 else 22 for layer in range(24, 42)]
    for wrong, said in ((dict(num_kv_shared_layers=35), "of layers it has not"), (dict(num_kv_shared_layers=33), "of layers it has not"), (dict(num_kv_shared_layers=30), "of each size of its heads"),
                        (dict(num_kv_shared_layers=34), "of layers it has not")):
        assert said in refusal({**E2B, "hidden_size_per_layer_input": 0, **wrong}), wrong


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


def test_the_shared_layers_are_the_forms_and_read_what_they_can():
    from engine.layout import NARROW_KEPT, WIDE_KEPT, shared_form
    assert shared_form(None) is None and shared_form({"layers": [], "from": []}) is None
    assert shared_form({"layers": [5, 4], "from": [2, 1]}, 6) == {"layers": (4, 5), "from": (1, 2)}
    for wrong in ({"layers": [4], "from": [4]}, {"layers": [4], "from": [5]}, {"layers": [4, 5], "from": [1, 4]},
                  {"layers": [4, 4], "from": [1, 2]}, {"layers": [4, 5], "from": [1]}):
        with pytest.raises(ValueError):
            shared_form(wrong, 6)
    form = {"arch": "gemma4", "head_dim": 16, "wide": {"layers": [1, 3, 5], "head_dim": 32},
            "kv_shared": {"layers": [4, 5], "from": [2, 3]}}
    dims = Dims([32, 64, 6, 4, 1, 320, 24], form)
    assert (dims.layers(NARROW_KEPT), dims.layers(WIDE_KEPT)) == ([0, 2], [1, 3])
    shapes = {row.name: row.shape for row in dims.rows()}
    assert shapes["wq"] == (3, 64, 32) and shapes["wk"] == (2, 16, 32) and shapes["k_norm_wide"] == (2, 32)
    # a layer reads one of its own size of heads
    with pytest.raises(ValueError, match="cannot read"):
        Dims([32, 64, 6, 4, 1, 320, 24], {**form, "kv_shared": {"layers": [4, 5], "from": [3, 2]}})


def test_a_shared_layer_attends_the_window_of_the_layer_it_reads():
    """T.518: a layer that reads another's keys and values attends a window where that one does and none where it does
    not (the CPU keeps a window's keys and values in a ring, which the reader takes as the layer it reads keeps it):
    refused otherwise, by the plan and by NumPy's model"""
    from engine.layout import check_shared_windows
    from engine.plan import Settings, layer_facts
    form = {"arch": "gemma4", "head_dim": 16, "wide": {"layers": [1, 3, 5], "head_dim": 32},
            "kv_shared": {"layers": [4, 5], "from": [2, 3]}}
    dims = Dims([32, 64, 6, 4, 1, 320, 24], form)
    same = {"layers": [0, 2, 4], "window": 4, "rope_theta": 100.0}
    check_shared_windows(dims.sharing, same)
    check_shared_windows(dims.sharing, None)
    assert len(layer_facts(dims, Settings(int8=True, kv_start=8, local=same))) == 6
    for other in ({"layers": [0, 2], "window": 4, "rope_theta": 100.0}, {"layers": [0, 4], "window": 4, "rope_theta": 100.0}):
        with pytest.raises(ValueError, match="Layer 4 cannot read the keys and values of layer 2: one of them attends a window"):
            layer_facts(dims, Settings(int8=True, kv_start=8, local=other))
    # (NumPy's model the same: a made-up Gemma 4 whose shared layers read their kind's, given local layers of another sort)
    tensors, config = gemma4_model(kinds="ssllsssl", sharing=4)
    made = conversion(tensors, config, "float32")
    passed = {key: value for key, value in made.options.items() if key in PASSED}
    vocabulary = pack_tokenizer(tiny_vocab(text(config)["vocab_size"]))
    Llama(bytes(made.stream.out), vocabulary, **passed)
    local = dict(passed["local"])
    with pytest.raises(ValueError, match="one of them attends a window"):
        Llama(bytes(made.stream.out), vocabulary, **{**passed, "local": {**local, "layers": [layer for layer in local["layers"] if layer < 4]}})


def test_a_shared_layer_keeps_no_keys_and_values_and_reads_its_sources():
    tensors, config = gemma4_model(kinds="ssllsssl", sharing=4, double=True)
    llama = engine(tensors, config)[0]
    for pos, token in enumerate(TOKENS):
        llama.forward(token, pos)
    assert [keys is None for keys in llama.key_cache] == [False] * 4 + [True] * 4
    assert [held for *_, held in llama.heads] == [0, 1, 2, 3, 1, 1, 1, 3]
    # what a shared layer's own k and v would make is read nowhere: changed, nothing moves
    changed = {**tensors, "model.layers.6.self_attn.k_proj.weight": tensors["model.layers.6.self_attn.k_proj.weight"] + 1.0,
               "model.layers.7.self_attn.v_proj.weight": tensors["model.layers.7.self_attn.v_proj.weight"] * 3.0}
    other = engine(changed, config)[0]
    fresh = engine(tensors, config)[0]
    for pos, token in enumerate(TOKENS):
        assert np.array_equal(other.forward(token, pos), fresh.forward(token, pos)), pos


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


def test_the_pages_engine_is_handed_a_gemma4_with_all_of_its_layers():
    """T.331.2 has forward.js read what a Gemma 4's layers are; T.330.2 and T.329.2 its layers that share keys and values
    and its embedding of every layer: a plan of each is made, and one with the embedding of every layer has its table,
    the projection, its norm and every layer's three among its tensors (forward.js reads them by their names)."""
    from engine.plan import Settings, forward_plan
    no_bytes = lambda offset, length: bytes(length)
    for form in ({}, {"kv_shared": {"layers": [3], "from": [1]}}, {"per_layer": 8}):
        dims = Dims([32, 64, 4, 4, 1, 320, 24], {"arch": "gemma4", "head_dim": 16, **form})
        tensors = forward_plan(dims, "int8", no_bytes, Settings(int8=True, kv_start=256))["tensors"]
        own = {"per_layer_table", "per_layer_projection", "per_layer_norm", "ple_gate", "ple_proj", "ple_post"}
        assert own <= set(tensors) if form.get("per_layer") else not own & set(tensors), form
        if form.get("per_layer"):
            assert tensors["per_layer_table"]["shape"] == [320, 4 * 8] and tensors["per_layer_norm"]["shape"] == [8]


ALL = dict(kinds="sslsslsslssl", sharing=5, double=True, per_layer=8)
# T.329: the tensors of every layer's input, of the model and of a layer (one that keeps its keys and values, one that
# reads another's, the last)
OWN = ["model.embed_tokens_per_layer.weight", "model.per_layer_model_projection.weight",
       "model.per_layer_projection_norm.weight"] + [f"model.layers.{layer}.{name}.weight" for layer in (0, 8, 11)
                                                    for name in ("per_layer_input_gate", "per_layer_projection",
                                                                 "post_per_layer_input_norm")]


@pytest.mark.parametrize("name", OWN)
def test_every_tensor_of_the_inputs_of_the_layers_is_read_where_transformers_reads_it(name):
    tensors, config = gemma4_model(**ALL)
    changed = dict(tensors)
    changed[name] = (tensors[name] + np.random.default_rng(7).standard_normal(tensors[name].shape) * 0.3).astype(np.float32)
    before, after = engine(tensors, config)[0], engine(changed, config)[0]
    want = naive_gemma4_logits(changed, config, TOKENS)
    moved = False
    for pos, token in enumerate(TOKENS):
        was, got = before.forward(token, pos).copy(), after.forward(token, pos)
        assert close(got, want[pos]), pos
        moved |= not np.allclose(got, was, rtol=1e-3, atol=1e-3)
    assert moved, name


@pytest.mark.parametrize("dtype, width", [("float16", 8), ("int8", 8), ("int6", 32)])
def test_the_table_of_every_layer_is_held_as_it_is_stored(dtype, width):
    """The table is read a row at a time and multiplies nothing: NumPy keeps it as the file stores it (a quantized file's
    values and scales, a float16 file's halves), and the model is close to the reference all the same. (Six bits need
    rows of whole groups of 32: a layer's projection back reads width values.)"""
    tensors, config = gemma4_model(**{**ALL, "per_layer": width})
    llama = engine(tensors, config, dtype)[0]
    if dtype == "float16":
        assert llama.per_layer_table.dtype == np.float16
    else:
        assert isinstance(llama.per_layer_table, tuple)
    want = naive_gemma4_logits(tensors, config, TOKENS)
    # (a quantized made-up model drifts from the reference past the first token by its noise alone, 3.6 at int8 with a
    # table of 32 a layer, of logits up to 6.6: the first token's row is what this asks of them)
    for pos, token in enumerate(TOKENS if dtype == "float16" else TOKENS[:1]):
        assert np.abs(llama.forward(token, pos) - want[pos]).max() < (0.05 if dtype == "float16" else 1.0), pos


def test_the_projection_is_scaled_before_its_norm_where_the_epsilon_shows_it():
    """The projection of the embedding's row is multiplied by 1 / sqrt(dim) before an RMSNorm, which only its epsilon
    lets see a scale: a projection small enough for the epsilon to matter shows whether the scale is where the
    model has it (the made-up projections of the other tests are too large for that: the review of T.329.1's breaks)."""
    tensors, config = gemma4_model(**ALL)
    name = "model.per_layer_model_projection.weight"
    small = {**tensors, name: tensors[name] * np.float32(2e-3)}
    llama = engine(small, config)[0]
    want = naive_gemma4_logits(small, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos


# ---- a GGUF (T.529)
# llama.cpp's names by transformers' of the language model saved alone (conversion/gemma.py's Gemma4Model, gguf-py's
# constants.py and tensor_mapping.py at b76b1337), written out here rather than taken from the converter's family
GGUF_NAMES = {"embed_tokens.weight": "token_embd.weight", "norm.weight": "output_norm.weight",
              "embed_tokens_per_layer.weight": "per_layer_token_embd.weight",
              "per_layer_model_projection.weight": "per_layer_model_proj.weight",
              "per_layer_projection_norm.weight": "per_layer_proj_norm.weight"}
GGUF_LAYER = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight",
              "pre_feedforward_layernorm.weight": "ffn_norm.weight", "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
              "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
              "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
              "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
              "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
              "mlp.down_proj.weight": "ffn_down.weight", "per_layer_input_gate.weight": "inp_gate.weight",
              "per_layer_projection.weight": "proj.weight", "post_per_layer_input_norm.weight": "post_norm.weight",
              "layer_scalar": "layer_output_scale.weight"}
# rows of whole groups of 32 (Q8_0); E2B's names of the whole model, its layers and its embedding of every layer
GGUF_E2B = dict(ALL, per_layer=32, whole=True)


def bfloat16(values):
    """llama.cpp's BF16 (round to nearest even, as torch's): the bytes, and the values they hold."""
    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    kept = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return kept.tobytes(), (kept.astype(np.uint32) << 16).view(np.float32).reshape(values.shape)


def gemma4_gguf(table_proportion=None, more=(), leave=(), **shape):
    """A GGUF v3 of a small Gemma 4 the way llama.cpp writes one (as ggml-org's gemma-4-E2B-it Q8_0 is): Q8_0 matrices
    (F32 where a row is not of whole groups of 32), the projection of every layer's embedding as BF16, the norms and
    every layer's number F32 and as they are (llama.cpp adds nothing to a Gemma 4's), q and k as Hugging Face holds
    them, rope_freqs after the table of every layer (1 for the pairs that turn, 1e30 for the others; table_proportion:
    of another part than the model's), the FFN a width a layer where use_double_wide_mlp. Returns the config.json, the
    file, and under the Hugging Face names the values it stands for."""
    tensors, config = gemma4_model(**shape)
    model = text(config)
    prefix = "model.language_model." if "text_config" in config else "model."
    stored, same = {}, {}
    for name, tensor in tensors.items():
        short = name[len(prefix):]
        if short in GGUF_NAMES:
            gguf = GGUF_NAMES[short]
        elif name == "lm_head.weight":
            gguf = "output.weight"
        else:
            _, layer, rest = short.split(".", 2)
            gguf = f"blk.{layer}.{GGUF_LAYER[rest]}"
        if short == "per_layer_model_projection.weight":
            blob, held = bfloat16(tensor)
            stored[gguf], same[name] = [blob, 30, tensor.shape], held
        elif tensor.ndim == 2 and tensor.shape[-1] % 32 == 0:
            blob, held = q8_0_blocks(tensor)
            stored[gguf], same[name] = [blob, 8, tensor.shape], held
        else:
            stored[gguf], same[name] = [tensor.astype(np.float32).tobytes(), 0, tensor.shape], tensor
        if short == "embed_tokens_per_layer.weight":
            wide = model["global_head_dim"]
            turned = int(wide * (table_proportion or model["rope_parameters"]["full_attention"]["partial_rotary_factor"]) / 2)
            table = np.array([1.0] * turned + [1e30] * (wide // 2 - turned), dtype=np.float32)
            stored["rope_freqs.weight"] = [table.tobytes(), 0, table.shape]
    layers, sharing = model["num_hidden_layers"], model["num_kv_shared_layers"]
    widths = [model["intermediate_size"] * (2 if model["use_double_wide_mlp"] and layer >= layers - sharing else 1)
              for layer in range(layers)]
    arch = "gemma4"
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, layers),
                (f"{arch}.context_length", 4, model["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, model["hidden_size"]),
                (f"{arch}.feed_forward_length", *((9, (4, widths)) if model["use_double_wide_mlp"] else (4, widths[0]))),
                (f"{arch}.attention.head_count", 4, model["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 4, model["num_key_value_heads"]),
                (f"{arch}.rope.freq_base", 6, model["rope_parameters"]["full_attention"]["rope_theta"]),
                (f"{arch}.rope.freq_base_swa", 6, model["rope_parameters"]["sliding_attention"]["rope_theta"]),
                (f"{arch}.attention.layer_norm_rms_epsilon", 6, model["rms_norm_eps"]),
                (f"{arch}.attention.key_length", 4, model["global_head_dim"]),
                (f"{arch}.attention.value_length", 4, model["global_head_dim"]),
                (f"{arch}.attention.sliding_window", 4, model["sliding_window"]),
                (f"{arch}.attention.shared_kv_layers", 4, sharing),
                (f"{arch}.embedding_length_per_layer_input", 4, model["hidden_size_per_layer_input"]),
                (f"{arch}.attention.sliding_window_pattern", 9, (7, [kind == "sliding_attention" for kind in model["layer_types"]])),
                (f"{arch}.attention.key_length_swa", 4, model["head_dim"]),
                (f"{arch}.attention.value_length_swa", 4, model["head_dim"]),
                (f"{arch}.rope.dimension_count", 4, model["global_head_dim"]),
                (f"{arch}.rope.dimension_count_swa", 4, model["head_dim"]),
                ("tokenizer.ggml.model", 8, "gemma4"), ("tokenizer.ggml.bos_token_id", 4, 2),
                ("tokenizer.ggml.eos_token_id", 4, 1)]
    if model["final_logit_softcapping"]:
        metadata.append((f"{arch}.final_logit_softcapping", 6, model["final_logit_softcapping"]))
    metadata = [entry for entry in metadata if entry[0] not in leave] + list(more)
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    out = [b"GGUF", struct.pack("<IQQ", 3, len(stored), len(metadata))]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind) + metadata_value(kind, value))
    blobs, offset = [], 0
    for gguf, (blob, type_, tensor_shape) in stored.items():
        out.append(string(gguf) + struct.pack("<I", len(tensor_shape)) + struct.pack(f"<{len(tensor_shape)}Q", *reversed(tensor_shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    return config, head + b"".join(blobs), same


GGUF_SHAPES = {"as E2B is": GGUF_E2B,
               "as E4B is: a classifier of its own, one width of FFN": dict(kinds="sslsslsslssl", sharing=5, per_layer=32,
                                                                           n_kv_heads=2, shared=False),
               "the language model saved alone, no cap, half of a head turns": dict(GGUF_E2B, whole=False, softcap=None,
                                                                                   proportion=0.5)}


@pytest.mark.parametrize("dtype", ["int8", "float32", "int6"])
@pytest.mark.parametrize("shape", GGUF_SHAPES)
def test_a_gemma4_gguf_with_the_originals_files_is_the_safetensors_conversion(shape, dtype, monkeypatch):
    """The list's way in (T.529): the checkpoint, tokenizer.bin and options of the safetensors of the same values, to the
    byte, the table of every layer among them (the row the page reads from the keeper, T.329.2.2's tableHole()). The
    rope_freqs table passes as it streams by, held to the original's quarter."""
    monkeypatch.setattr("convert.stream.PIECE", 700)
    config, file, same = gemma4_gguf(**GGUF_SHAPES[shape])
    vocabulary = unigram(text(config)["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
    expected = safetensors_conversion(same, config, vocabulary, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options and got.options["arch"] == "gemma4" and got.options["per_layer"] == 32
    if dtype == "float32":
        passed = {key: value for key, value in got.options.items() if key in PASSED}
        llama = Llama(bytes(got.checkpoint), pack_tokenizer(tiny_vocab(text(config)["vocab_size"])), **passed)
        want = naive_gemma4_logits(same, text(config), TOKENS)
        for pos, token in enumerate(TOKENS):
            assert close(llama.forward(token, pos), want[pos]), pos


@pytest.mark.parametrize("change, what", [
    ({"num_kv_shared_layers": 4}, "layers that read another's keys and values"),
    ({"sliding_window": 6}, "sliding-attention layers"),
    ({"layer_types": ["sliding_attention"] * 2 + ["full_attention"] * 10}, "sliding-attention layers"),
    ({"final_logit_softcapping": None}, "cap of the logits"),
    ({"rms_norm_eps": 1e-5}, "RMSNorm epsilon"),
    ({"head_dim": 32}, "size of a head"),
    ({"num_key_value_heads": 2}, "number of key-value heads"),
])
def test_a_gemma4_gguf_that_is_not_the_originals_is_refused(change, what):
    """What no tensor shows: which layers read another's keys and values, which are local, their window, the cap."""
    config, file, _ = gemma4_gguf(**GGUF_E2B)
    llama2_convert.gguf_weights(file, json.dumps(config))  # its own config goes through
    changed = {**config, "text_config": {**config["text_config"], **change}}
    with pytest.raises(ValueError, match=f"its {what} is "):
        llama2_convert.gguf_weights(file, json.dumps(changed))


def test_a_gemma4_gguf_that_turns_another_part_of_a_head_is_refused_as_it_streams_past():
    """How much of a head turns is the rope_freqs table's alone (rope.dimension_count is the whole head): held to the
    original's as it passes, the header being no different."""
    config, file, _ = gemma4_gguf(table_proportion=0.5, **GGUF_E2B)
    with pytest.raises(ValueError, match="scales its RoPE otherwise than the original"):
        with_original(file, config, unigram(text(config)["vocab_size"]), "tokenizer.json", "int8")


def test_a_gemma4_gguf_alone_is_refused_by_its_vocabulary():
    """Its vocabulary is llama.cpp's gemma4 one, which the engine does not read: refused by that, and not by the table
    (which is not the first tensor, and is no scaling)."""
    _, file, _ = gemma4_gguf(**GGUF_E2B)
    with pytest.raises(ValueError, match="This GGUF has a gemma4 vocabulary"):
        llama2_convert.Conversion.from_gguf(file, dtype="int8")
