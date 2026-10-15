"""Google's Gemma 3, the text model (T261): a Qwen3's attention and a gated FFN, with a norm behind each as well as in
front, a GELU in the FFN's gate, an embedding multiplied by the root of its width, and layers of two kinds: most attend
a window of the last positions and turn their q and k by a base of RoPE of their own.

The reference (conftest.naive_gemma3_logits) is written from the Hugging Face tensors the way transformers computes
them, so it checks the converter (the 1 of every norm, q and k turned, the root folded into the table and out of the
last norm, which layers are local ones) and the engine together. tests/reference_gemma3.py holds that reference, and
the engine, to transformers itself (CI: PyTorch is not on the development machine)."""
import json
import math
import struct

import numpy as np
import pytest
from conftest import gemma3_model, naive_gemma3_logits, pack_tokenizer, tiny_vocab
from test_convert import converted, reader, safetensors_file, streamed
from test_gguf import metadata_value, q8_0_blocks, unigram, with_original

import llama2_convert
from llama2_convert import Safetensors, check_config, checkpoint_form, checkpoint_size, normalize
from llama2_numpy import FORM, GELU, LAYOUTS, SILU, Llama, checkpoint_dtype, form_of, local_layers, tensor_rows
from convert.families.gemma3 import local

# more than the window (4), and a token that comes again
TOKENS = [1, 5, 7, 9, 11, 5, 5, 300, 2, 17, 4]
MODELS = {
    "small": dict(),
    "a classifier of its own": dict(shared=False),
    "the kinds by a pattern": dict(kinds="sslssl", listed=False, pattern=3),
    "the 1B's pattern, cut short": dict(kinds="sssss", listed=False, pattern=6),
    "the bases as transformers 5 writes them": dict(parameters=True, kinds="lsl"),
    "every layer attends all": dict(kinds="lll"),
    "every layer is local": dict(kinds="sss", window=3),
    "a window of one": dict(window=1, kinds="sls"),
    "a window past the text": dict(window=20),
    "keys for every head": dict(n_kv_heads=4, kinds="lssl"),
    "heads that fill the width": dict(head_dim=8, n_kv_heads=2),
    "another epsilon, the bases the other way round": dict(eps=1e-5, theta=50.0, local_theta=1000000.0),
}


def read(config):
    """config.json as transformers' Gemma3TextConfig would have read it, for the reference: the kinds of the layers
    listed and the bases by their old names (by hand here, not by the converter's normalize())."""
    read = dict(config)
    if "layer_types" not in read:
        pattern = read["sliding_window_pattern"]
        read["layer_types"] = ["sliding_attention" if (layer + 1) % pattern else "full_attention"
                               for layer in range(read["num_hidden_layers"])]
    if "rope_parameters" in read:
        read["rope_theta"] = read["rope_parameters"]["full_attention"]["rope_theta"]
        read["rope_local_base_freq"] = read["rope_parameters"]["sliding_attention"]["rope_theta"]
    return read


def options_of(config):
    """What the caller passes to Llama() for a file of this config.json (the converter's options say the same)."""
    config = normalize(config)
    form = checkpoint_form(config, {})
    return {"arch": "gemma3", "head_dim": form["head_dim"], "rope_theta": config["rope_theta"],
            "rms_norm_eps": config["rms_norm_eps"], "local": local(config)}


def engine(tensors, config, dtype="float32"):
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), config, dtype)
    return Llama(checkpoint, pack_tokenizer(tiny_vocab(config["vocab_size"])), dtype=dtype, **options_of(config)), checkpoint


def conversion(tensors, config, dtype="float32"):
    """The page's way: the file in its own order, with a tokenizer.json, and the options that come out."""
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    made = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), unigram(config["vocab_size"]),
                                     "tokenizer.json", dtype=dtype, max_seq_len=config["max_position_embeddings"], start=8 + size)
    made.stream.feed(file[8 + size:])
    made.stream.finish()
    return made


def close(got, want):
    return np.allclose(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("model", MODELS)
def test_a_gemma3_converts_and_runs_like_transformers(model):
    tensors, config = gemma3_model(**MODELS[model])
    llama, checkpoint = engine(tensors, config)
    header = struct.unpack_from("<7i", checkpoint, 0)
    assert header[:5] == (32, 64, config["num_hidden_layers"], 4, config["num_key_value_heads"])
    assert checkpoint_dtype(header, len(checkpoint), options_of(config)) == "float32"
    want = naive_gemma3_logits(tensors, read(config), TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos


# every tensor of a layer: changed in one layer, the engine must follow the reference to other numbers, or the tensor
# is not read where transformers reads it
KINDS = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
         "self_attn.q_norm.weight", "self_attn.k_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
         "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "mlp.gate_proj.weight",
         "mlp.up_proj.weight", "mlp.down_proj.weight"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which", [0, 2, -1])
def test_every_tensor_is_read_where_transformers_reads_it(kind, which):
    tensors, config = gemma3_model()
    name = f"model.layers.{which % config['num_hidden_layers']}.{kind}"  # a local layer, one that is not, the last
    changed = dict(tensors)
    rng = np.random.default_rng(7)
    changed[name] = (tensors[name] + rng.standard_normal(tensors[name].shape) * 0.3).astype(np.float32)
    before, after = engine(tensors, config)[0], engine(changed, config)[0]
    want = naive_gemma3_logits(changed, read(config), TOKENS)
    moved = False
    for pos, token in enumerate(TOKENS):
        was, got = before.forward(token, pos).copy(), after.forward(token, pos)
        assert close(got, want[pos]), pos
        moved |= not np.allclose(got, was, rtol=1e-3, atol=1e-3)
    assert moved, name


def test_a_part_of_one_tensor_that_is_changed_shows():
    """The reference is not blind to where a value sits in its tensor: one value of each norm of a layer, of a head's
    norm (which the converter turns as it turns the rows of q), and of the last norm."""
    tensors, config = gemma3_model()
    want = naive_gemma3_logits(tensors, read(config), TOKENS)
    spots = [("model.layers.0.input_layernorm.weight", 3), ("model.layers.1.post_attention_layernorm.weight", 31),
             ("model.layers.2.pre_feedforward_layernorm.weight", 0), ("model.layers.5.post_feedforward_layernorm.weight", 17),
             ("model.layers.0.self_attn.q_norm.weight", 1), ("model.layers.4.self_attn.k_norm.weight", 9),
             ("model.norm.weight", 5)]
    for name, spot in spots:
        changed = {**tensors, name: tensors[name].copy()}
        changed[name][spot] += 2.0
        other = naive_gemma3_logits(changed, read(config), TOKENS)
        assert not np.allclose(other[-1], want[-1], rtol=1e-3, atol=1e-3), (name, spot)
        llama = engine(changed, config)[0]
        for pos, token in enumerate(TOKENS):
            assert close(llama.forward(token, pos), other[pos]), (name, spot, pos)


@pytest.mark.parametrize("window", [1, 2, 4, 5])
def test_a_local_layer_attends_its_window_and_nothing_before_it(window):
    """Two layers, both local ones (the converter takes no model of one layer): the logits at a position are a function
    of the tokens two windows reach back to, 2 (window - 1) before this one. The token just past that changes nothing;
    the oldest one within it does (a window read one position too long, or too short, fails one of the two)."""
    tensors, config = gemma3_model(kinds="ss", window=window)
    llama = engine(tensors, config)[0]
    text = [3, 8, 20, 31, 44, 59, 61, 70, 82, 97, 101, 113]
    last, reach = len(text) - 1, 2 * (window - 1)

    def logits(model, text):
        return [model.forward(token, pos).copy() for pos, token in enumerate(text)][-1]

    plain = logits(llama, text)
    outside = list(text)
    outside[last - reach - 1] = 250  # the newest token the windows leave out
    assert np.array_equal(logits(llama, outside), plain)
    if window > 1:
        inside = list(text)
        inside[last - reach] = 250  # the oldest they have
        assert not np.allclose(logits(llama, inside), plain, rtol=1e-4, atol=1e-4)
    # (and layers that are not local read every token)
    whole = engine(*gemma3_model(kinds="ll", window=window))[0]
    assert not np.allclose(logits(whole, outside), logits(whole, text), rtol=1e-3, atol=1e-3)


def test_a_local_layer_turns_by_its_own_base_and_the_others_by_theirs():
    """The two pairs of tables: the file's (float32) or the engine's from rope_theta, for the layers that attend all the
    positions, and the engine's own for the local ones. One base for both kinds is another model."""
    tensors, config = gemma3_model()
    llama = engine(tensors, config)[0]
    positions = np.arange(config["max_position_embeddings"])[:, None]
    angles = lambda theta: positions / theta ** (np.arange(0, 16, 2) / 16)
    assert np.allclose(llama.freq_cis_real, np.cos(angles(1e6)), atol=1e-6)
    assert np.allclose(llama.local_cis[1], np.sin(angles(100.0)), atol=1e-6)
    want = naive_gemma3_logits(tensors, read(config), TOKENS)
    for base in ("rope_theta", "rope_local_base_freq"):
        other = Llama(engine(tensors, config)[1], pack_tokenizer(tiny_vocab(320)),
                      **{**options_of(config), "local": {**options_of(config)["local"], "rope_theta": 1e6}}) \
            if base == "rope_local_base_freq" else \
            Llama(converted(Safetensors(reader(safetensors_file(tensors))), {**config, "rope_theta": 100.0}, "float32"),
                  pack_tokenizer(tiny_vocab(320)), **options_of(config))
        got = [other.forward(token, pos).copy() for pos, token in enumerate(TOKENS)]
        assert close(got[0], want[0]) and not np.allclose(got[-1], want[-1], rtol=1e-3, atol=1e-3), base
    # a model without local layers has no such tables
    assert engine(*gemma3_model(kinds="lll"))[0].local_cis is None


def test_the_root_of_the_width_is_in_the_table_and_out_of_the_last_norm():
    """What the converter folds: the embedding's rows times sqrt(dim) (in float32), and where the table is the
    classifier too, the last norm's weight (1 + w) over sqrt(dim); with a classifier of its own the norm stays 1 + w.
    Every other norm is 1 + w, the heads' turned as the rows of q are."""
    for shared in (True, False):
        tensors, config = gemma3_model(shared=shared)
        llama = engine(tensors, config)[0]
        root = np.float32(math.sqrt(32))
        assert np.array_equal(llama.token_embedding_table, tensors["model.embed_tokens.weight"] * root)
        final = tensors["model.norm.weight"] + np.float32(1.0)
        assert np.array_equal(llama.rms_final_weight, final * np.float32(1.0 / math.sqrt(32)) if shared else final)
        assert np.array_equal(llama.wcls, llama.token_embedding_table if shared else tensors["lm_head.weight"])
        assert np.array_equal(llama.rms_att_post_weight[3], tensors["model.layers.3.post_attention_layernorm.weight"] + np.float32(1.0))
        assert np.array_equal(llama.rms_ffn_post_weight[5], tensors["model.layers.5.post_feedforward_layernorm.weight"] + np.float32(1.0))
        turned = (tensors["model.layers.2.self_attn.q_norm.weight"] + np.float32(1.0)).reshape(2, 8).T.reshape(-1)
        assert np.array_equal(llama.q_norm[2], turned)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8", "int6"])
@pytest.mark.parametrize("shared", [True, False])
def test_the_file_in_its_own_order_gives_the_same_checkpoint(dtype, shared, monkeypatch):
    monkeypatch.setattr("convert.stream.PIECE", 700)  # several pieces per tensor, and not a multiple of a row
    tensors, config = gemma3_model(shared=shared)
    file = safetensors_file(tensors)
    expected = converted(Safetensors(reader(file)), config, dtype)
    got, progress = streamed(file, config, dtype, 4096)
    assert got == expected and progress[-1][0] == progress[-1][1]
    assert checkpoint_dtype(struct.unpack_from("<7i", got, 0), len(got), options_of(config)) == dtype
    llama = Llama(got, pack_tokenizer(tiny_vocab(320)), dtype=dtype, **options_of(config))
    want = naive_gemma3_logits(tensors, read(config), TOKENS)
    got = [llama.forward(token, pos).copy() for pos, token in enumerate(TOKENS)]
    if dtype == "float32":
        assert close(got, want)
    else:
        assert np.corrcoef(np.ravel(got), np.ravel(want))[0, 1] > (0.999 if dtype == "float16" else 0.9)


def test_the_table_goes_piece_by_piece_and_is_the_table_taken_whole(monkeypatch):
    """The embedding is the one tensor with a transform that is not given whole (262,144 rows of the real one): the
    pieces it comes in, of the reader's choosing or of the stream's, make the bytes of the whole tensor multiplied at
    once, and every other tensor with a transform is still asked for whole."""
    tensors, config = gemma3_model()
    file = safetensors_file(tensors, "BF16")
    whole = streamed(file, config, "int8", 1 << 20)[0]
    asked = []
    for piece in (64, 700, 5000):
        monkeypatch.setattr("convert.stream.PIECE", piece)
        log = []
        assert converted(Safetensors(reader(file, log)), config, "int8") == whole
        assert streamed(file, config, "int8", 333)[0] == whole
        asked.append(len(log))
    assert asked[0] > asked[1] > asked[2]  # (the smaller the piece, the more reads: the table is not read at once)
    from convert.stream import piecewise
    assert piecewise(None) and piecewise(("scale", 2.0))
    assert not any(piecewise(transform) for transform in (("one",), ("permute", 4), (("permute", 4), ("scale", 2.0)),
                                                           (("one",), ("scale", 0.5)), ("transpose",), ("taps",)))


@pytest.mark.parametrize("model", ["small", "a classifier of its own", "the kinds by a pattern", "every layer attends all",
                                   "the bases as transformers 5 writes them", "heads that fill the width"])
def test_the_conversion_tells_the_engine_what_the_file_cannot(model):
    """The lesson of T72: the engine built from exactly the options the conversion gives runs like the reference."""
    shape = MODELS[model]
    tensors, config = gemma3_model(**shape)
    made = conversion(tensors, config)
    kinds = shape.get("kinds", "sslssl")
    assert made.options["arch"] == "gemma3" and made.options["bias"] is False
    if "s" in kinds:
        assert made.options["local"] == {"layers": [layer for layer, kind in enumerate(kinds) if kind == "s"],
                                         "window": 4, "rope_theta": 100.0}
    else:
        assert "local" not in made.options
    assert made.options["rope_theta"] == 1e6 and made.options["rms_norm_eps"] == 1e-6
    assert made.options.get("head_dim") == (None if shape.get("head_dim") == 8 else 16)
    assert not {"qk_norm", "linear", "convolution", "rotary", "rotated", "unturned", "rope_scaling"} & set(made.options)
    assert made.options["bos"] == 2 and set(made.options["stop_tokens"]) == {2, 7}
    passed = {key: value for key, value in made.options.items()
              if key in ("dtype", "arch", "bias", "head_dim", "local", "rope_theta", "rms_norm_eps")}
    llama = Llama(bytes(made.stream.out), pack_tokenizer(tiny_vocab(config["vocab_size"])), **passed)
    want = naive_gemma3_logits(tensors, read(config), TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos
    # and the form that sizes the file is what the options say
    assert checkpoint_size(made.stream.header, "float32", made.options) == len(made.stream.out)
    assert form_of(made.options) == {**FORM, "arch": "gemma3", "head_dim": made.options.get("head_dim", 0)}
    assert (made.stream.header[5] < 0) is (not shape.get("shared", True))


def test_the_other_models_keep_the_options_they_always_had():
    """No "local" where there are no such layers: a saved conversion is used again only while its options are what the
    converter gives (kept.js's CONVERTER)."""
    from conftest import lfm2_model, qwen35_model, synthetic_weights
    from test_convert import hugging_face
    from test_lfm2 import conversion as convolved
    from test_qwen3 import conversion as plain
    from test_qwen35 import conversion as hybrid
    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    assert set(plain(tensors, published, settings).options) == {"tokenizer_kind", "nfkc", "dtype", "rope_theta", "bos",
                                                               "stop_tokens", "bias", "arch"}
    assert "local" not in hybrid(*qwen35_model()).options and "local" not in convolved(*lfm2_model()).options


def test_the_facts_of_the_architecture():
    """What no other architecture has: a gated FFN whose activation is a GELU (a Llama's is a SiLU, and GPT-2's GELU is
    of an FFN without a gate), and layers that may be local ones. And two norms no other has, as rows of its file."""
    assert LAYOUTS["gemma3"].activation == GELU and LAYOUTS["gemma3"].gated_ffn and LAYOUTS["gemma3"].local
    assert all(not (layout.gated_ffn and layout.activation == GELU) and not layout.local
               for arch, layout in LAYOUTS.items() if arch not in ("gemma3", "gemma4"))
    assert {arch: layout.activation for arch, layout in LAYOUTS.items()} == {
        "llama": SILU, "gpt2": GELU, "neox": GELU, "qwen35": SILU, "lfm2": SILU, "gemma3": GELU, "gemma4": GELU}
    names = [row.name for row in tensor_rows((32, 64, 6, 4, 1, 320, 24), {"arch": "gemma3", "head_dim": 16})]
    assert {"rms_att_post_weight", "rms_ffn_post_weight", "q_norm", "k_norm"} <= set(names)
    for arch in ("llama", "qwen35", "lfm2", "gpt2", "neox"):
        form = {"arch": arch, "linear": {"every": 2, "key_heads": 2, "value_heads": 4, "key_dim": 8, "value_dim": 8, "conv": 4}
                if arch == "qwen35" else None, "convolution": {"layers": "ccacca", "taps": 3} if arch == "lfm2" else None}
        heads = 4
        assert not {"rms_att_post_weight", "rms_ffn_post_weight"} & {row.name for row in tensor_rows((32, 64, 6, 4, heads, 320, 24), form)}


def test_the_local_layers_are_a_gemmas_and_layers_of_the_model():
    said = {"layers": [4, 0, 1, 1], "window": 512, "rope_theta": 10000}
    assert local_layers("gemma3", 6, said) == {"layers": (0, 1, 4), "window": 512, "rope_theta": 10000.0}
    assert local_layers("gemma3", 6, None) is None and local_layers("llama", 6, None) is None
    assert local_layers("gemma3", 6, {**said, "layers": []}) is None
    refusal = "The local layers are layers of a Gemma 3 with a window and a base of RoPE, and none of another architecture."
    for arch, layers, more in (("llama", 6, {}), ("lfm2", 6, {}), ("gemma3", 4, {}), ("gemma3", 6, {"layers": [-1]}),
                               ("gemma3", 6, {"window": 0}), ("gemma3", 6, {"rope_theta": 0.0}),
                               ("gemma3", 6, {"rope_theta": float("inf")})):
        with pytest.raises(ValueError) as error:
            local_layers(arch, layers, {**said, **more})
        assert str(error.value) == refusal
    # and Llama() refuses them the same way
    tensors, config = gemma3_model()
    checkpoint = engine(tensors, config)[1]
    with pytest.raises(ValueError, match="The local layers are layers of a Gemma 3"):
        Llama(checkpoint, pack_tokenizer(tiny_vocab(320)), **{**options_of(config), "local": {**said, "layers": [6]}})


def test_the_pages_engine_is_handed_what_a_gemma_is():
    """T261.2: the plan of a model whose weights are outside Python says the three things forward.js runs a Gemma 3 by:
    the tensors of the two norms behind, the FFN's activation, and of every layer its window and whether the local
    layers' tables turn it, with those tables. (T261.1 let go of the engine and refused the model here.)"""
    tensors, config = gemma3_model()
    checkpoint = engine(tensors, config, "int8")[1]
    handed = {}

    class Engine:
        backend = "test"
        bind = staticmethod(lambda logits: None)
        forward = staticmethod(lambda *args: None)

    class External:
        size = len(checkpoint)
        read = staticmethod(lambda at, length: checkpoint[at:at + length])

        @staticmethod
        def start(plan):
            handed.update(plan)
            return Engine

    llama = Llama(None, pack_tokenizer(tiny_vocab(320)), dtype="int8", external=External, **options_of(config))
    assert llama.backend == "test" and handed["arch"] == "gemma3" and handed["activation"] == "gelu"
    assert [(layer["window"], layer["local"]) for layer in handed["layers"]] == \
        [(4, True), (4, True), (0, False), (4, True), (4, True), (0, False)]
    assert {"rms_att_post_weight", "rms_ffn_post_weight"} <= set(handed["tensors"])
    positions = np.arange(24)[:, None]
    assert np.allclose(np.frombuffer(handed["derived"]["local_cis_real"], dtype=np.float32).reshape(24, 8),
                       np.cos(positions / 100.0 ** (np.arange(0, 16, 2) / 16)), atol=1e-6)
    assert np.allclose(np.frombuffer(handed["derived"]["freq_cis_imag"], dtype=np.float32).reshape(24, 8),
                       np.sin(positions / 1e6 ** (np.arange(0, 16, 2) / 16)), atol=1e-6)


def test_the_keys_and_values_of_every_position_are_kept_past_the_window():
    """The cache grows as any model's does (a local layer reads the last of it), and a long text stays finite."""
    tensors, config = gemma3_model(seq_len=600, window=8)
    llama = engine(tensors, config)[0]
    for pos in range(300):
        llama.forward(5 + pos % 7, pos, need_logits=False)
    assert llama.key_cache.shape[0] == 6 and llama.key_cache.shape[2] > 256
    assert np.all(np.any(llama.key_cache[:, :, :300], axis=(1, 3)))
    assert np.isfinite(llama.forward(3, 300)).all()


# ---- config.json
# the config.json of the published models (unsloth/gemma-3-270m-it at 23cf460f and unsloth/gemma-3-1b-it at 5b11413a),
# what is read of them
PUBLISHED = {
    "gemma-3-270m-it": (dict(_sliding_window_pattern=6, hidden_size=640, intermediate_size=2048, num_hidden_layers=18,
                             layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 3, rope_local_base_freq=10000.0,
                             rope_theta=1000000.0), [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]),
    "gemma-3-1b-it": (dict(sliding_window_pattern=6, hidden_size=1152, intermediate_size=6912, num_hidden_layers=26,
                           rope_local_base_freq=10000, rope_theta=1000000, cache_implementation="hybrid"),
                      [layer for layer in range(26) if (layer + 1) % 6]),
}


@pytest.mark.parametrize("model", PUBLISHED)
def test_the_published_configs_read_as_transformers_reads_them(model):
    said, layers = PUBLISHED[model]
    common = dict(model_type="gemma3_text", architectures=["Gemma3ForCausalLM"], attention_bias=False, attention_dropout=0.0,
                  attn_logit_softcapping=None, bos_token_id=2, eos_token_id=106, final_logit_softcapping=None, head_dim=256,
                  hidden_activation="gelu_pytorch_tanh", max_position_embeddings=32768, num_attention_heads=4,
                  num_key_value_heads=1, pad_token_id=0, query_pre_attn_scalar=256, rms_norm_eps=1e-06, rope_scaling=None,
                  sliding_window=512, vocab_size=262144)
    config = normalize({**common, **said})
    check_config(config)
    assert normalize(config) == config  # twice is once
    assert local(config) == {"layers": layers, "window": 512, "rope_theta": 10000.0}
    assert config["rope_theta"] == 1e6 and "rope_scaling" not in config and config["tie_word_embeddings"] is True
    form = checkpoint_form(config, {})
    assert form_of(form) == {**FORM, "arch": "gemma3", "head_dim": 256}
    header = llama2_convert.checkpoint_header(config, {}, 4096)
    assert header == (said["hidden_size"], said["intermediate_size"], said["num_hidden_layers"], 4, 1, 262144, 4096)


REFUSED = [
    ({"hidden_activation": "silu"}, "its activation is silu, not gelu_pytorch_tanh"),
    ({"attention_bias": True}, "its layers have biases"),
    ({"attn_logit_softcapping": 50.0}, "it caps its attention's scores or its logits"),
    ({"final_logit_softcapping": 30.0}, "it caps its attention's scores or its logits"),
    ({"use_bidirectional_attention": True}, "it attends in both directions"),
    ({"layer_types": ["sliding_attention"] * 5}, "its layers are not sliding-attention layers and full-attention layers, one kind for every layer"),
    ({"layer_types": ["sliding_attention"] * 5 + ["conv"]}, "its layers are not sliding-attention layers and full-attention layers, one kind for every layer"),
    ({"layer_types": None, "sliding_window_pattern": 0}, "its layers are not sliding-attention layers and full-attention layers, one kind for every layer"),
    ({"sliding_window": None}, "its config.json has no usable sliding_window"),
    ({"sliding_window": 0}, "its config.json has no usable sliding_window"),
    ({"rope_local_base_freq": 0}, "its config.json has no usable base of RoPE"),
    ({"rope_theta": "1e6"}, "its config.json has no usable base of RoPE"),
    ({"query_pre_attn_scalar": 168}, "it scales its attention's scores by 168, not by the size of its heads"),
    ({"rope_scaling": {"rope_type": "yarn", "factor": 2.0}}, "it uses RoPE scaling of the yarn kind"),
    ({"rope_parameters": {"full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1e6},
                          "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}}}, "it uses RoPE scaling of the linear kind"),
    ({"head_dim": 15}, "its attention heads do not divide the hidden size the way llama2.c expects"),
]


@pytest.mark.parametrize("change, reason", REFUSED)
def test_a_gemma_the_engine_cannot_run_is_refused(change, reason):
    _, config = gemma3_model()
    check_config(normalize(config))
    with pytest.raises(ValueError) as error:
        check_config(normalize({**config, **change}))
    assert str(error.value) == f"This model cannot be converted: {reason}."


def test_a_model_without_a_window_has_no_local_layers_and_needs_no_window():
    _, config = gemma3_model(kinds="lll")
    config = normalize({**config, "sliding_window": None})
    check_config(config)
    assert local(config) is None


def test_the_engine_reads_the_tensors_where_the_layout_puts_them():
    """Every row of the file as the engine holds it against the Hugging Face tensor it is made of, in every dtype's
    file: the sizes agree (checkpoint_dtype()), and the float32 file's rows are the tensors."""
    tensors, config = gemma3_model(shared=False)
    for dtype in ("float32", "float16", "int8", "int6"):
        llama, checkpoint = engine(tensors, config, dtype)
        header = struct.unpack_from("<7i", checkpoint, 0)
        assert checkpoint_dtype(header, len(checkpoint), options_of(config)) == dtype
        assert llama.wo.shape == (6, 32, 64) and llama.wq.shape == (6, 64, 32) and llama.wk.shape == (6, 16, 32)
        tolerance = {"float32": 0, "float16": 1e-3, "int8": 0.02, "int6": 0.05}[dtype]
        for ours, theirs in (("wv", "self_attn.v_proj"), ("wo", "self_attn.o_proj"), ("w1", "mlp.gate_proj"),
                             ("w2", "mlp.down_proj"), ("w3", "mlp.up_proj")):
            for layer in (0, 5):
                assert np.allclose(getattr(llama, ours)[layer], tensors[f"model.layers.{layer}.{theirs}.weight"], atol=tolerance), (dtype, ours)


# ---- a GGUF
GGUF_NAMES = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
              "lm_head.weight": "output.weight"}
GGUF_LAYER = {"input_layernorm": "attn_norm", "post_attention_layernorm": "post_attention_norm",
              "pre_feedforward_layernorm": "ffn_norm", "post_feedforward_layernorm": "post_ffw_norm",
              "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
              "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
# rows of whole groups of 32 (Q8_0)
GEMMA = dict(dim=64, hidden_dim=128, head_dim=32)


def gemma3_gguf(more=(), leave=(), change=None, **shape):
    """A GGUF v3 of a small Gemma 3 the way llama.cpp writes one (as unsloth's gemma-3-270m-it Q8_0 is): Q8_0 matrices,
    F32 norms with the 1 added, q and k as Hugging Face holds them, the embedding as it is, the norm behind the
    attention as post_attention_norm and the one behind the FFN as post_ffw_norm, the window and the full-attention
    layers' base in the metadata and neither the pattern nor the other base (llama.cpp has 6 and 1e4 for those: the
    made-up model's are given in more where they are others). The last norm first, as the real file has it.
    Returns the config.json, the file, and under the Hugging Face names the values it stands for."""
    tensors, config = gemma3_model(**{**GEMMA, **shape})
    stored, same = {}, {}
    for name in sorted(tensors, key=lambda name: name != "model.norm.weight"):
        tensor = tensors[name]
        if name in GGUF_NAMES:
            gguf = GGUF_NAMES[name]
        else:
            _, _, layer, *rest = name.split(".")
            gguf = f"blk.{layer}.{GGUF_LAYER['.'.join(rest[:-1])]}.{rest[-1]}"
        if tensor.ndim == 2:
            blob, held = q8_0_blocks(tensor)
            stored[gguf], same[name] = [blob, 8, tensor.shape], held
        else:
            stored[gguf], same[name] = [(tensor + np.float32(1.0)).astype(np.float32).tobytes(), 0, tensor.shape], tensor
    if change:
        change(stored)
    arch = "gemma3"
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, config["num_hidden_layers"]),
                (f"{arch}.context_length", 4, config["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, config["hidden_size"]),
                (f"{arch}.feed_forward_length", 4, config["intermediate_size"]),
                (f"{arch}.attention.head_count", 4, config["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 4, config["num_key_value_heads"]),
                (f"{arch}.attention.layer_norm_rms_epsilon", 6, config["rms_norm_eps"]),
                (f"{arch}.attention.key_length", 4, config["head_dim"]),
                (f"{arch}.attention.value_length", 4, config["head_dim"]),
                (f"{arch}.rope.freq_base", 6, float(config["rope_theta"])),
                (f"{arch}.attention.sliding_window", 4, config["sliding_window"]),
                ("tokenizer.ggml.model", 8, "llama"), ("tokenizer.ggml.pre", 8, "default"),
                ("tokenizer.ggml.bos_token_id", 4, 2), ("tokenizer.ggml.eos_token_id", 4, 7)]
    metadata = [entry for entry in metadata if entry[0] not in leave] + list(more)
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    tokens = [f"w{i}" for i in range(config["vocab_size"])]
    out = [b"GGUF", struct.pack("<IQQ", 3, len(stored), len(metadata) + 3)]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind))
        out.append(metadata_value(kind, value))
    out.append(string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens)) + b"".join(map(string, tokens)))
    out.append(string("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, len(tokens)) + struct.pack(f"<{len(tokens)}i", *[1] * len(tokens)))
    out.append(string("tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, 0))
    blobs, offset = [], 0
    for gguf, (blob, type_, shape) in stored.items():
        out.append(string(gguf) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *reversed(shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    return config, head + b"".join(blobs), same


def safetensors_conversion(same, config, vocabulary, dtype):
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = llama2_convert.Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(config), vocabulary,
                                         "tokenizer.json", dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    return expected


# the real file's metadata: a window, llama.cpp's own pattern (6) and its own base of the local layers (1e4)
AS_PUBLISHED = dict(kinds="ssssslssss", local_theta=10000.0, window=5)
GGUF_SHAPES = {"as the published one says it": (AS_PUBLISHED, ()),
               "a classifier of its own": ({**AS_PUBLISHED, "shared": False}, ()),
               "a pattern and a base of its own": (dict(kinds="sslssl", local_theta=100.0),
                                                   (("gemma3.attention.sliding_window_pattern", 4, 3), ("gemma3.rope.freq_base_swa", 6, 100.0))),
               "the layers one by one": (dict(kinds="slsslsl", local_theta=10000.0),
                                         (("gemma3.attention.sliding_window_pattern", 9, (7, [True, False, True, True, False, True, False])),))}


@pytest.mark.parametrize("dtype", ["int8", "float32", "float16", "int6"])
@pytest.mark.parametrize("shape", GGUF_SHAPES)
def test_a_gemma3_gguf_with_the_originals_files_is_the_safetensors_conversion(shape, dtype, monkeypatch):
    """The list's way in: the checkpoint, tokenizer.bin and options of the safetensors of the same values, to the byte.
    The norms come with their 1 added and are not given another; the last norm comes before the table."""
    monkeypatch.setattr("convert.stream.PIECE", 700)
    model, more = GGUF_SHAPES[shape]
    config, file, same = gemma3_gguf(more=more, **model)
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
    expected = safetensors_conversion(same, config, vocabulary, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert got.options["arch"] == "gemma3" and got.options["local"]["layers"] == [l for l, kind in enumerate(model["kinds"]) if kind == "s"]
    if dtype == "float32":
        llama = Llama(bytes(got.checkpoint), pack_tokenizer(tiny_vocab(config["vocab_size"])), **options_of(config))
        want = naive_gemma3_logits(same, read(config), TOKENS)
        for pos, token in enumerate(TOKENS):
            assert close(llama.forward(token, pos), want[pos]), pos


def test_a_gguf_without_a_window_is_a_model_whose_layers_all_attend_everything():
    config, file, _ = gemma3_gguf(leave=("gemma3.attention.sliding_window",), kinds="lll")
    header, base = llama2_convert.gguf_weights(file, json.dumps({**config, "sliding_window": None}))
    assert "model.layers.2.post_feedforward_layernorm.weight" in json.loads(header)


@pytest.mark.parametrize("change, what", [
    ({"layer_types": ["sliding_attention"] * 10}, "sliding-attention layers"),
    ({"sliding_window": 6}, "sliding-attention layers"),
    ({"rope_local_base_freq": 100.0}, "sliding-attention layers"),
    ({"rope_theta": 10000.0}, "RoPE theta"),
    ({"num_key_value_heads": 2, "num_attention_heads": 2}, "number of heads"),
    ({"rms_norm_eps": 1e-5}, "RMSNorm epsilon"),
    ({"head_dim": 16}, "size of a head"),
    ({"model_type": "llama"}, "architecture"),
])
def test_a_gemma3_gguf_that_is_not_the_originals_is_refused(change, what):
    """What no tensor shows: which layers are local, their window, the two bases."""
    config, file, _ = gemma3_gguf(**AS_PUBLISHED)
    llama2_convert.gguf_weights(file, json.dumps(config))  # its own config goes through
    with pytest.raises(ValueError, match=f"its {what} is "):
        llama2_convert.gguf_weights(file, json.dumps({**config, **change}))


def test_a_gguf_that_caps_its_logits_is_refused():
    """A cap is no tensor either: a GGUF that has one where the original has none is another model."""
    config, file, _ = gemma3_gguf(more=[("gemma3.final_logit_softcapping", 6, 30.0)], **AS_PUBLISHED)
    with pytest.raises(ValueError, match="its cap of the logits is 30.0 here and None there"):
        with_original(file, config, unigram(config["vocab_size"]), "tokenizer.json", "int8")
