"""Qwen3.5's hybrid attention (T229): a Qwen3 most of whose layers are Gated DeltaNet ("linear attention") layers,
which keep a state of a fixed size where the others keep keys and values, and whose full-attention layers gate
their output.

The reference (conftest.naive_qwen35_logits) is written from the Hugging Face tensors the way transformers computes
them, so it checks the converter (q and its gate in one matrix, RoPE over the first part of a head, norms stored
around zero, the taps of the convolution, A_log) and the engine together. tests/reference_qwen35.py holds that
reference, and the engine, to transformers itself (CI: PyTorch is not on the development machine)."""
import json
import struct

import numpy as np
import pytest
from conftest import naive_qwen35_logits, pack_tokenizer, qwen35_model, tiny_vocab
from test_convert import converted, reader, safetensors_file, streamed

import llama2_convert
from llama2_convert import Safetensors, check_config, checkpoint_form, checkpoint_size, normalize, rotary_dim, transformed
from llama2_numpy import FORM, Llama, checkpoint_dtype, form_of, layer_slots, linear_form

TOKENS = [1, 5, 7, 9, 11, 5, 5, 300, 2]  # more than the taps of the convolution, and a token that comes again
# value heads on their own key heads and three to a key head, heads that fill dim and heads that do not, a classifier
# of its own, the language model saved alone, every second, third and fourth layer a full-attention one, and a head
# that turns whole
MODELS = {
    "small": dict(),
    "one value head a key head": dict(value_heads=2, value_dim=8),
    "three value heads a key head": dict(key_heads=1, value_heads=3, key_dim=4, value_dim=10, shared=False),
    "every third": dict(n_layers=7, every=3, prefix="model."),
    "every fourth": dict(n_layers=8, every=4, n_kv_heads=4, head_dim=8, rotary=0.5),
    "whole heads turn": dict(rotary=1.0, conv=2, n_kv_heads=1),
    # (the review) heads of 256 with a quarter of them turned, as every real Qwen3.5 has: 64 of 256, and q's matrix of
    # 1024 rows in a dim of 32
    "heads of 256": dict(n_heads=2, n_kv_heads=1, head_dim=256, rotary=0.25, key_heads=2, value_heads=4, key_dim=32, value_dim=32),
}


def options_of(config):
    """What the caller passes to Llama() for a file of this config.json (the converter's options say the same)."""
    config = normalize(config)
    form = checkpoint_form(config, {})
    return {"arch": "qwen35", "linear": form["linear"], "head_dim": form["head_dim"], "rotary": rotary_dim(config),
            "rope_theta": config["rope_theta"], "rms_norm_eps": config["rms_norm_eps"]}


def engine(tensors, config, dtype="float32"):
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), config, dtype)
    vocab_size = config["text_config"]["vocab_size"]
    return Llama(checkpoint, pack_tokenizer(tiny_vocab(vocab_size)), dtype=dtype, **options_of(config)), checkpoint


def conversion(tensors, config, dtype="float32"):
    """The page's way: the file in its own order, with a tokenizer.json, and the options that come out."""
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    text = config["text_config"]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(text["vocab_size"])]}}).encode()
    made = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), vocabulary,
                                     "tokenizer.json", dtype=dtype, max_seq_len=text["max_position_embeddings"], start=8 + size)
    made.stream.feed(file[8 + size:])
    made.stream.finish()
    return made


@pytest.mark.parametrize("model", MODELS)
def test_a_qwen35_converts_and_runs_like_transformers(model):
    tensors, config = qwen35_model(**MODELS[model])
    llama, checkpoint = engine(tensors, config)
    text = config["text_config"]
    assert struct.unpack_from("<7i", checkpoint, 0)[:5] == (
        text["hidden_size"], text["intermediate_size"], text["num_hidden_layers"], text["num_attention_heads"],
        text["num_key_value_heads"])
    assert checkpoint_dtype(struct.unpack_from("<7i", checkpoint, 0), len(checkpoint), options_of(config)) == "float32"
    want = naive_qwen35_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=2e-4, atol=2e-4), pos


# every tensor of the file that is not a Llama's too, and the norms (stored around zero): changed in one layer, the
# engine must follow the reference to other numbers, or the tensor is not read where transformers reads it
KINDS = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
         "self_attn.q_norm.weight", "self_attn.k_norm.weight", "linear_attn.in_proj_qkv.weight",
         "linear_attn.in_proj_z.weight", "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
         "linear_attn.conv1d.weight", "linear_attn.dt_bias", "linear_attn.A_log", "linear_attn.norm.weight",
         "linear_attn.out_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which", [0, -1])
def test_every_tensor_is_read_where_transformers_reads_it(kind, which):
    tensors, config = qwen35_model(n_layers=6, every=3)
    names = sorted((name for name in tensors if name.endswith(kind) and "layers." in name),
                   key=lambda name: int(name.split("layers.")[1].split(".")[0]))
    name = names[which]  # the first and the last layer of its kind
    changed = dict(tensors)
    rng = np.random.default_rng(7)
    changed[name] = (tensors[name] + rng.standard_normal(tensors[name].shape) * 0.3).astype(np.float32)
    before, after = engine(tensors, config)[0], engine(changed, config)[0]
    want = naive_qwen35_logits(changed, config, TOKENS)
    moved = False
    for pos, token in enumerate(TOKENS):
        was, got = before.forward(token, pos).copy(), after.forward(token, pos)
        assert np.allclose(got, want[pos], rtol=2e-4, atol=2e-4), pos
        moved |= not np.allclose(got, was, rtol=1e-3, atol=1e-3)
    assert moved, name


def test_a_part_of_one_tensor_that_is_changed_shows():
    """The reference is not blind to where a value sits in its tensor: one row of q's gate, one tap of one channel of
    the convolution, the decay of one head."""
    tensors, config = qwen35_model()
    text = config["text_config"]
    want = naive_qwen35_logits(tensors, config, TOKENS)
    spots = [("model.language_model.layers.1.self_attn.q_proj.weight", (2 * text["head_dim"] - 1, slice(None))),
             ("model.language_model.layers.0.linear_attn.conv1d.weight", (5, 0, 0)),
             ("model.language_model.layers.2.linear_attn.A_log", (3,)),
             ("model.language_model.layers.2.linear_attn.in_proj_qkv.weight", (-1, slice(None)))]
    for name, spot in spots:
        changed = {**tensors, name: tensors[name].copy()}
        changed[name][spot] += 2.0
        other = naive_qwen35_logits(changed, config, TOKENS)
        assert not np.allclose(other[-1], want[-1], rtol=1e-3, atol=1e-3), name
        llama = engine(changed, config)[0]
        for pos, token in enumerate(TOKENS):
            assert np.allclose(llama.forward(token, pos), other[pos], rtol=2e-4, atol=2e-4), (name, pos)


def test_a_run_begins_at_position_0_and_goes_on_in_order():
    """The linear-attention layers keep a state where the others keep keys and values: position 0 clears it, and a
    position out of turn is refused (keys and values would simply be written again)."""
    tensors, config = qwen35_model()
    llama = engine(tensors, config)[0]
    want = naive_qwen35_logits(tensors, config, TOKENS)
    for _ in range(2):  # the second run begins with the state the first one left, and clears it
        for pos, token in enumerate(TOKENS[:6]):
            assert np.allclose(llama.forward(token, pos), want[pos], rtol=2e-4, atol=2e-4)
    assert np.any(llama.delta_state) and np.any(llama.conv_state)
    for pos in (3, 5, 7, 1):
        with pytest.raises(ValueError, match="position 6 comes next"):
            llama.forward(5, pos)
    assert np.allclose(llama.forward(TOKENS[6], 6), want[6], rtol=2e-4, atol=2e-4)  # a refusal changed nothing
    # the prompt's tokens, which make no logits, move the state on as the others do
    for pos, token in enumerate(TOKENS[:-1]):
        assert llama.forward(token, pos, need_logits=False) is None
    assert np.allclose(llama.forward(TOKENS[-1], len(TOKENS) - 1), want[-1], rtol=2e-4, atol=2e-4)


def test_the_state_has_a_size_the_context_does_not_change():
    """Keys and values only for the full-attention layers; for the others a matrix a value head and the last
    conv - 1 tokens, whatever the position."""
    tensors, config = qwen35_model(n_layers=6, every=3, seq_len=600)
    llama = engine(tensors, config)[0]
    text = config["text_config"]
    assert llama.key_cache.shape[0] == 2 and llama.delta_state.shape == (4, 4, 8, 6)
    assert llama.conv_state.shape == (4, text["linear_conv_kernel_dim"] - 1, 2 * 16 + 24)
    for pos in range(300):  # past KV_START: the caches grow, the state does not
        llama.forward(5 + pos % 7, pos, need_logits=False)
    assert llama.key_cache.shape[2] > 256 and llama.delta_state.shape == (4, 4, 8, 6)
    assert np.isfinite(llama.forward(3, 300)).all()


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8", "int6"])
def test_the_file_in_its_own_order_gives_the_same_checkpoint(dtype, monkeypatch):
    monkeypatch.setattr(llama2_convert, "PIECE", 700)  # several pieces per tensor, and not a multiple of a row
    # int6 needs rows of whole groups of 32: value heads that make 32 values
    tensors, config = qwen35_model(value_dim=8, shared=False)
    file = safetensors_file(tensors)
    expected = converted(Safetensors(reader(file)), config, dtype)
    got, progress = streamed(file, config, dtype, 4096)
    assert got == expected and progress[-1][0] == progress[-1][1]
    assert checkpoint_dtype(struct.unpack_from("<7i", got, 0), len(got), options_of(config)) == dtype
    # and the engine reads exactly that many bytes, and something like the float32 model's numbers
    llama = Llama(got, pack_tokenizer(tiny_vocab(320)), dtype=dtype, **options_of(config))
    want = naive_qwen35_logits(tensors, config, TOKENS)
    got = [llama.forward(token, pos).copy() for pos, token in enumerate(TOKENS)]
    if dtype == "float32":
        assert np.allclose(got, want, rtol=2e-4, atol=2e-4)
    else:
        assert np.corrcoef(np.ravel(got), np.ravel(want))[0, 1] > (0.999 if dtype == "float16" else 0.9)


def test_bfloat16_is_read_as_the_real_file_stores_it():
    """Qwen/Qwen3.5-0.8B stores everything in bfloat16 but A_log and the norm of the value heads."""
    tensors, config = qwen35_model()
    truncated = {name: (np.ascontiguousarray(tensor).view(np.uint32) >> 16 << 16).view(np.float32) for name, tensor in tensors.items()}
    file = safetensors_file(tensors, "BF16")
    assert streamed(file, config, "float32", 4096)[0] == converted(llama2_convert.Arrays(truncated), config, "float32")


def test_the_gates_of_a_linear_layer_are_never_quantized():
    """wb and wa feed a sigmoid and an exp that scale a whole head's state: float32 in an int8 file, like the norms."""
    tensors, config = qwen35_model()
    llama, _ = engine(tensors, config, "int8")
    for ours, theirs in (("wb", "in_proj_b"), ("wa", "in_proj_a")):
        for a, layer in enumerate((0, 2)):
            assert np.array_equal(getattr(llama, ours)[a], tensors[f"model.language_model.layers.{layer}.linear_attn.{theirs}.weight"])
    assert not np.array_equal(llama.wz[0], tensors["model.language_model.layers.0.linear_attn.in_proj_z.weight"])


@pytest.mark.parametrize("model", ["small", "three value heads a key head", "every fourth"])
def test_the_conversion_tells_the_engine_what_the_file_cannot(model):
    """The lesson of T72: the engine built from exactly the options the conversion gives runs like the reference."""
    tensors, config = qwen35_model(**MODELS[model])
    made = conversion(tensors, config)
    text = config["text_config"]
    assert made.options["arch"] == "qwen35" and made.options["bias"] is False
    assert made.options["linear"] == {"every": text["full_attention_interval"], "key_heads": text["linear_num_key_heads"],
                                      "value_heads": text["linear_num_value_heads"], "key_dim": text["linear_key_head_dim"],
                                      "value_dim": text["linear_value_head_dim"], "conv": text["linear_conv_kernel_dim"]}
    assert made.options["rotary"] == int(text["head_dim"] * text["rope_parameters"]["partial_rotary_factor"])
    assert made.options["rope_theta"] == 1e7 and made.options["rms_norm_eps"] == 1e-6
    assert made.options.get("head_dim") == (None if text["head_dim"] * text["num_attention_heads"] == text["hidden_size"] else text["head_dim"])
    assert "qk_norm" not in made.options  # the architecture has them: nothing to say
    # the BOS the file does not name is the end-of-text token (normalize())
    assert made.options["bos"] == 7 and set(made.options["stop_tokens"]) == {7}
    passed = {key: value for key, value in made.options.items()
              if key in ("dtype", "arch", "bias", "linear", "head_dim", "rotary", "rope_theta", "rms_norm_eps")}
    llama = Llama(bytes(made.stream.out), pack_tokenizer(tiny_vocab(text["vocab_size"])), **passed)
    want = naive_qwen35_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=2e-4, atol=2e-4), pos
    # and the form that sizes the file is what the options say
    assert checkpoint_size(made.stream.header, "float32", made.options) == len(made.stream.out)
    assert form_of(made.options) == {**FORM, "arch": "qwen35", "linear": made.options["linear"],
                                     "head_dim": made.options.get("head_dim", 0)}


def test_the_other_models_keep_the_options_they_always_had():
    """No "linear" where there are no such layers: a saved conversion is used again only while its options are what the
    converter gives (kept.js's CONVERTER)."""
    from conftest import synthetic_weights
    from test_convert import hugging_face
    from test_qwen3 import conversion as plain
    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    assert set(plain(tensors, published, settings).options) == {"tokenizer_kind", "nfkc", "dtype", "rope_theta", "bos",
                                                               "stop_tokens", "bias", "arch"}


def test_the_config_is_the_language_models():
    tensors, config = qwen35_model()
    lifted = normalize(config)
    assert lifted["model_type"] == "qwen3_5_text" and lifted["hidden_size"] == 32
    assert lifted["rope_theta"] == 10000000 and lifted["rotary_pct"] == 0.25 and "rope_scaling" not in lifted
    assert lifted["bos_token_id"] == 7  # none named: the end-of-text token
    assert normalize(lifted) == lifted  # and the text model's own config.json reads the same
    assert normalize({**config, "text_config": {**config["text_config"], "bos_token_id": 3}})["bos_token_id"] == 3
    assert normalize({**config, "text_config": {**config["text_config"], "eos_token_id": [9, 7]}})["bos_token_id"] == 9
    check_config(lifted)


@pytest.mark.parametrize("change, reason", [
    (dict(layer_types=["linear_attention", "full_attention", "full_attention", "linear_attention"]), "not every 2th"),
    (dict(layer_types=["linear_attention"] * 4, full_attention_interval=None), "no usable linear-attention layers|not every"),
    (dict(full_attention_interval=5, layer_types=None), "no full-attention layer"),
    (dict(linear_num_value_heads=3), "numbers of linear-attention layers"),
    (dict(linear_conv_kernel_dim=0), "no usable linear-attention layers"),
    (dict(mlp_only_layers=[1]), "not the ones of a Qwen3.5"),
    (dict(attn_output_gate=False), "not the ones of a Qwen3.5"),
    (dict(hidden_act="gelu"), "activation"),
    (dict(attention_bias=True), "biases"),
    (dict(rope_parameters={"rope_theta": 1e7, "partial_rotary_factor": 0.01}), "rotates none"),
    (dict(rope_parameters={"rope_type": "yarn", "factor": 2.0, "rope_theta": 1e7}), "RoPE scaling"),
])
def test_a_qwen35_the_engine_cannot_run_is_refused(change, reason):
    _, config = qwen35_model()
    text = {key: value for key, value in {**config["text_config"], **change}.items() if value is not None}
    with pytest.raises(ValueError, match=reason):
        check_config(normalize({**config, "text_config": text}))


def test_the_rotation_and_the_heads_follow_transformers_where_the_config_leaves_them_out():
    """Qwen3_5TextConfig: partial_rotary_factor 0.25 (at the top of a config that has no rope_parameters, or left out) and
    heads of 256. The published ones say both; one that did not must not turn whole heads."""
    _, config = qwen35_model()
    text = {key: value for key, value in config["text_config"].items() if key not in ("head_dim", "rope_parameters")}
    text["rope_theta"] = 10000000
    for extra, rotary in (({}, 0.25), ({"partial_rotary_factor": 0.5}, 0.5)):
        lifted = normalize({**config, "text_config": {**text, **extra}})
        assert lifted["rotary_pct"] == rotary and lifted["head_dim"] == 256 and lifted["rope_theta"] == 10000000
        assert rotary_dim(lifted) == int(256 * rotary)
    # a config that says it keeps what it says
    said = normalize(config)
    assert said["rotary_pct"] == 0.25 and said["head_dim"] == 16


def test_the_layers_follow_transformers_where_the_config_leaves_them_out():
    """transformers makes layer_types from full_attention_interval (4 where there is none), and has its own numbers
    for the linear layers."""
    _, config = qwen35_model(n_layers=8, every=4)
    text = {key: value for key, value in config["text_config"].items()
            if not key.startswith("linear_") and key not in ("layer_types", "full_attention_interval")}
    assert checkpoint_form(normalize({**config, "text_config": text}), {})["linear"] == {
        "every": 4, "key_heads": 16, "value_heads": 32, "key_dim": 128, "value_dim": 128, "conv": 4}
    text["layer_types"] = ["linear_attention", "linear_attention", "full_attention"] * 2 + ["linear_attention"] * 2
    assert checkpoint_form(normalize({**config, "text_config": text}), {})["linear"]["every"] == 3


def test_the_layers_slots():
    assert layer_slots(3, None) == [(False, 0), (False, 1), (False, 2)]
    linear = linear_form({"every": 3, "key_heads": 1, "value_heads": 2, "key_dim": 4, "value_dim": 4, "conv": 4})
    assert layer_slots(7, linear) == [(True, 0), (True, 1), (False, 0), (True, 2), (True, 3), (False, 1), (True, 4)]
    assert linear_form(None) is None
    for wrong in ({"every": 1}, {"value_heads": 3, "key_heads": 2}, {"conv": 0}):
        with pytest.raises(ValueError, match="numbers of linear-attention layers"):
            linear_form({**linear, **wrong})


def test_an_engine_that_is_told_half_of_it_refuses():
    tensors, config = qwen35_model()
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), config, "float32")
    options = options_of(config)
    tokenizer = pack_tokenizer(tiny_vocab(320))
    with pytest.raises(ValueError, match="go together"):
        Llama(checkpoint, tokenizer, **{**options, "linear": None})
    with pytest.raises(ValueError, match="go together"):
        Llama(checkpoint, tokenizer, **{**options, "arch": "llama"})
    header = struct.unpack_from("<7i", checkpoint, 0)
    with pytest.raises(ValueError, match="not a llama2.c checkpoint"):
        checkpoint_dtype(header, len(checkpoint), {**options, "linear": None})
    with pytest.raises(ValueError, match="not a llama2.c checkpoint"):
        checkpoint_dtype(header, len(checkpoint), {**options, "linear": {**options["linear"], "conv": 3}})


def test_the_transforms_of_a_qwen35():
    """What the converter does to the tensors that are not stored as the engine reads them."""
    rows = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)
    # two heads of (q: 4 rows, gate: 4 rows); RoPE turns q's first 2 rows, which are already a pair
    q = transformed(rows, ("heads", 2, 0, 2, 2), 0)
    assert q[:, 0].tolist() == [0, 3, 6, 9, 24, 27, 30, 33]
    assert transformed(rows, ("heads", 2, 1, 2, 0), 0)[:, 0].tolist() == [12, 15, 18, 21, 36, 39, 42, 45]
    # a head of 8 whose first 4 rows turn: [first halves, second halves] of those become adjacent pairs
    k = transformed(rows, ("heads", 1, 0, 2, 4), 0)
    assert (k[:, 0] / 3).tolist() == [0, 2, 1, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15]
    norm = transformed(np.arange(8, dtype=np.float32), (("heads", 1, 0, 1, 4), ("one",)), 0)
    assert norm.tolist() == [1, 3, 2, 4, 5, 6, 7, 8]
    taps = transformed(np.arange(6, dtype=np.float32).reshape(3, 1, 2), ("taps",), 0)
    assert taps.tolist() == [[0, 2, 4], [1, 3, 5]]
    assert np.allclose(transformed(np.array([0.0, 1.0], dtype=np.float32), ("decay",), 0), [-1.0, -np.e])
    for wrong in (("heads", 3, 0, 2, 0), ("heads", 2, 0, 2, 6)):
        with pytest.raises(ValueError, match="rows are not"):
            transformed(rows, wrong, 0)
