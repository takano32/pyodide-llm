"""Liquid AI's LFM2 and LFM2.5 (T260): a Qwen3 some of whose layers are convolution layers, which look at this token
and the two before it and keep those in place of keys and values.

The reference (conftest.naive_lfm2_logits) is written from the Hugging Face tensors the way transformers computes
them, so it checks the converter (which layers are which, q and k turned, the taps of the convolution, the three parts
of the matrix in, the FFN's inside from the config's rule) and the engine together. tests/reference_lfm2.py holds that
reference, and the engine, to transformers itself (CI: PyTorch is not on the development machine)."""
import json
import struct

import numpy as np
import pytest
from conftest import lfm2_model, naive_lfm2_logits, pack_tokenizer, tiny_vocab
from test_convert import converted, reader, safetensors_file, streamed
from test_gguf import fed, metadata_value, q8_0_blocks, unigram, with_original

import llama2_convert
from llama2_convert import (Safetensors, architecture, check_config, checkpoint_form, checkpoint_size, convolution_layers,
                            head_size, layout, normalize)
from llama2_numpy import FORM, Llama, checkpoint_dtype, convolution_form, form_of, layer_slots

TOKENS = [1, 5, 7, 9, 11, 5, 5, 300, 2]  # more than the taps of the convolution, and a token that comes again
# the 350M's order of layers in small, a classifier of its own, two taps and four, an FFN whose size the config says
# as it is (the 230M), keys and values for every head, the 230M's order, and convolution layers at both ends
MODELS = {
    "small": dict(),
    "a classifier of its own": dict(shared=False, kinds="ccacca"),
    "two taps": dict(taps=2, kinds="acca"),
    "four taps": dict(taps=4, kinds="cacac"),
    "the size of the FFN as it is": dict(adjust=False, block_ff_dim=64),
    "keys for every head": dict(n_kv_heads=4, kinds="accacc"),
    "the 230M's order": dict(kinds="ccacacacacacac"),
    "attention first and last": dict(kinds="accca", eps=1e-6),
}


def options_of(config):
    """What the caller passes to Llama() for a file of this config.json (the converter's options say the same)."""
    config = normalize(config)
    form = checkpoint_form(config, {})
    return {"arch": "lfm2", "convolution": form["convolution"], "rope_theta": config["rope_theta"],
            "rms_norm_eps": config["rms_norm_eps"]}


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


@pytest.mark.parametrize("model", MODELS)
def test_an_lfm2_converts_and_runs_like_transformers(model):
    tensors, config = lfm2_model(**MODELS[model])
    llama, checkpoint = engine(tensors, config)
    header = struct.unpack_from("<7i", checkpoint, 0)
    assert header[:5] == (config["hidden_size"], 64, config["num_hidden_layers"], config["num_attention_heads"],
                          config["num_key_value_heads"])
    assert checkpoint_dtype(header, len(checkpoint), options_of(config)) == "float32"
    want = naive_lfm2_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=2e-4, atol=2e-4), pos


# every tensor of a layer: changed in one layer, the engine must follow the reference to other numbers, or the tensor
# is not read where transformers reads it
KINDS = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.out_proj.weight",
         "self_attn.q_layernorm.weight", "self_attn.k_layernorm.weight", "conv.in_proj.weight", "conv.conv.weight",
         "conv.out_proj.weight", "operator_norm.weight", "ffn_norm.weight", "feed_forward.w1.weight",
         "feed_forward.w2.weight", "feed_forward.w3.weight"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which", [0, -1])
def test_every_tensor_is_read_where_transformers_reads_it(kind, which):
    tensors, config = lfm2_model(kinds="ccacac")
    names = sorted((name for name in tensors if name.endswith(kind) and "layers." in name),
                   key=lambda name: int(name.split("layers.")[1].split(".")[0]))
    name = names[which]  # the first and the last layer of its kind
    changed = dict(tensors)
    rng = np.random.default_rng(7)
    changed[name] = (tensors[name] + rng.standard_normal(tensors[name].shape) * 0.3).astype(np.float32)
    before, after = engine(tensors, config)[0], engine(changed, config)[0]
    want = naive_lfm2_logits(changed, config, TOKENS)
    moved = False
    for pos, token in enumerate(TOKENS):
        was, got = before.forward(token, pos).copy(), after.forward(token, pos)
        assert np.allclose(got, want[pos], rtol=2e-4, atol=2e-4), pos
        moved |= not np.allclose(got, was, rtol=1e-3, atol=1e-3)
    assert moved, name


def test_a_part_of_one_tensor_that_is_changed_shows():
    """The reference is not blind to where a value sits in its tensor: each tap of one channel of the convolution, a
    row of each of the three parts of the matrix in, a value of a head's norm."""
    tensors, config = lfm2_model()
    dim = config["hidden_size"]
    want = naive_lfm2_logits(tensors, config, TOKENS)
    spots = [("model.layers.0.conv.conv.weight", (5, 0, 0)), ("model.layers.0.conv.conv.weight", (5, 0, 1)),
             ("model.layers.3.conv.conv.weight", (dim - 1, 0, 2)), ("model.layers.1.conv.in_proj.weight", (3, slice(None))),
             ("model.layers.1.conv.in_proj.weight", (dim + 3, slice(None))),
             ("model.layers.1.conv.in_proj.weight", (3 * dim - 1, slice(None))),
             ("model.layers.2.self_attn.q_layernorm.weight", (1,)), ("model.layers.5.self_attn.k_layernorm.weight", (7,))]
    for name, spot in spots:
        changed = {**tensors, name: tensors[name].copy()}
        changed[name][spot] += 2.0
        other = naive_lfm2_logits(changed, config, TOKENS)
        assert not np.allclose(other[-1], want[-1], rtol=1e-3, atol=1e-3), (name, spot)
        llama = engine(changed, config)[0]
        for pos, token in enumerate(TOKENS):
            assert np.allclose(llama.forward(token, pos), other[pos], rtol=2e-4, atol=2e-4), (name, spot, pos)


def test_the_first_tokens_see_zeros_before_them():
    """The convolution of the first token reads its last tap alone, the second token's its last two: a model whose
    older taps are far larger than the newest shows a state that was not zero, or a tap read at the wrong token."""
    tensors, config = lfm2_model(kinds="ccaca")
    for name in tensors:
        if name.endswith("conv.conv.weight"):
            tensors[name] = tensors[name] * np.float32([8.0, 4.0, 1.0])
    llama = engine(tensors, config)[0]
    want = naive_lfm2_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=2e-4, atol=2e-4), pos


def test_a_run_begins_at_position_0_and_goes_on_in_order():
    """The convolution layers keep a state where the others keep keys and values: position 0 clears it, and a position
    out of turn is refused (keys and values would simply be written again)."""
    tensors, config = lfm2_model()
    llama = engine(tensors, config)[0]
    want = naive_lfm2_logits(tensors, config, TOKENS)
    for _ in range(2):  # the second run begins with the state the first one left, and clears it
        for pos, token in enumerate(TOKENS[:6]):
            assert np.allclose(llama.forward(token, pos), want[pos], rtol=2e-4, atol=2e-4)
    assert np.all(np.any(llama.conv_state, axis=2))
    for pos in (3, 5, 7, 1):
        with pytest.raises(ValueError, match="position 6 comes next"):
            llama.forward(5, pos)
    assert np.allclose(llama.forward(TOKENS[6], 6), want[6], rtol=2e-4, atol=2e-4)  # a refusal changed nothing
    # the prompt's tokens, which make no logits, move the state on as the others do
    for pos, token in enumerate(TOKENS[:-1]):
        assert llama.forward(token, pos, need_logits=False) is None
    assert np.allclose(llama.forward(TOKENS[-1], len(TOKENS) - 1), want[-1], rtol=2e-4, atol=2e-4)


def test_the_state_has_a_size_the_context_does_not_change():
    """Keys and values only for the attention layers; for the others the last taps - 1 tokens, whatever the position."""
    tensors, config = lfm2_model(kinds="ccacca", seq_len=600)
    llama = engine(tensors, config)[0]
    assert llama.key_cache.shape[0] == 2 and llama.conv_state.shape == (4, 2, 32)
    for pos in range(300):  # past KV_START: the caches grow, the state does not
        llama.forward(5 + pos % 7, pos, need_logits=False)
    assert llama.key_cache.shape[2] > 256 and llama.conv_state.shape == (4, 2, 32)
    assert np.isfinite(llama.forward(3, 300)).all()


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8", "int6"])
@pytest.mark.parametrize("shared", [True, False])
def test_the_file_in_its_own_order_gives_the_same_checkpoint(dtype, shared, monkeypatch):
    monkeypatch.setattr("convert.stream.PIECE", 700)  # several pieces per tensor, and not a multiple of a row
    tensors, config = lfm2_model(shared=shared)
    file = safetensors_file(tensors)
    expected = converted(Safetensors(reader(file)), config, dtype)
    got, progress = streamed(file, config, dtype, 4096)
    assert got == expected and progress[-1][0] == progress[-1][1]
    assert checkpoint_dtype(struct.unpack_from("<7i", got, 0), len(got), options_of(config)) == dtype
    # the three places the order of the tensors lives in agree on the size: layout() wrote it, checkpoint_dtype() named
    # it, and the engine reads exactly that many bytes, and something like the float32 model's numbers
    llama = Llama(got, pack_tokenizer(tiny_vocab(320)), dtype=dtype, **options_of(config))
    want = naive_lfm2_logits(tensors, config, TOKENS)
    got = [llama.forward(token, pos).copy() for pos, token in enumerate(TOKENS)]
    if dtype == "float32":
        assert np.allclose(got, want, rtol=2e-4, atol=2e-4)
    else:
        assert np.corrcoef(np.ravel(got), np.ravel(want))[0, 1] > (0.999 if dtype == "float16" else 0.9)


def test_the_engine_reads_the_tensors_where_the_layout_puts_them():
    """Places (what forward.js gets, external=) against layout(): the same shapes in the same order at the same
    offsets, for every dtype."""
    from llama2_numpy import Places, TENSOR_NAMES
    from llama2_convert import tensor_bytes
    tensors, config = lfm2_model(shared=False)
    form = checkpoint_form(normalize(config), {})
    for dtype in ("float32", "float16", "int8", "int6"):
        checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), config, dtype)
        header = struct.unpack_from("<7i", checkpoint, 0)
        offset, expected = 28, []
        for shape, is_matrix in layout(*header, **form):
            size = tensor_bytes(shape, is_matrix, dtype)
            # (the RoPE tables are the file's in float32 alone: float16's are too coarse and the engine makes its own)
            if size and (is_matrix is not None or dtype == "float32"):
                expected.append((offset, tuple(shape)))
            offset += size
        assert offset == len(checkpoint)

        class External:
            size = len(checkpoint)
            read = staticmethod(lambda at, length: checkpoint[at:at + length])

            @staticmethod
            def start(plan):
                External.plan = plan
                return type("Engine", (), {"backend": "test", "bind": staticmethod(lambda logits: None),
                                           "forward": staticmethod(lambda *args: None)})

        Llama(None, pack_tokenizer(tiny_vocab(320)), dtype=dtype, external=External, **options_of(config))
        placed = sorted({(t["offset"], tuple(t["shape"])) for t in External.plan["tensors"].values()})
        assert placed == expected, dtype
        assert External.plan["convolution"] == form["convolution"] and External.plan["arch"] == "lfm2"
        assert set(External.plan["tensors"]) <= set(TENSOR_NAMES)


def test_bfloat16_is_read_as_the_real_file_stores_it():
    """LiquidAI/LFM2.5-350M stores everything in bfloat16."""
    tensors, config = lfm2_model()
    truncated = {name: (np.ascontiguousarray(tensor).view(np.uint32) >> 16 << 16).view(np.float32) for name, tensor in tensors.items()}
    file = safetensors_file(tensors, "BF16")
    assert streamed(file, config, "float32", 4096)[0] == converted(llama2_convert.Arrays(truncated), config, "float32")


def test_the_taps_are_never_quantized():
    """Three numbers a channel, float32 in an int8 file like the norms; the two matrices around them are int8."""
    tensors, config = lfm2_model()
    llama, _ = engine(tensors, config, "int8")
    for a, layer in enumerate((0, 1, 3, 4, 6)):
        assert np.array_equal(llama.conv[a], tensors[f"model.layers.{layer}.conv.conv.weight"][:, 0, :].T)
    assert not np.array_equal(llama.win[0], tensors["model.layers.0.conv.in_proj.weight"])
    assert np.allclose(llama.win[0], tensors["model.layers.0.conv.in_proj.weight"], atol=0.02)
    assert np.allclose(llama.wout[4], tensors["model.layers.6.conv.out_proj.weight"], atol=0.02)


@pytest.mark.parametrize("model", ["small", "a classifier of its own", "four taps", "attention first and last"])
def test_the_conversion_tells_the_engine_what_the_file_cannot(model):
    """The lesson of T72: the engine built from exactly the options the conversion gives runs like the reference."""
    shape = MODELS[model]
    tensors, config = lfm2_model(**shape)
    made = conversion(tensors, config)
    assert made.options["arch"] == "lfm2" and made.options["bias"] is False
    assert made.options["convolution"] == {"layers": shape.get("kinds", "ccaccaca"), "taps": shape.get("taps", 3)}
    assert made.options["rope_theta"] == 1e6
    assert made.options.get("rms_norm_eps") == (1e-6 if shape.get("eps") else None)  # only where it is not 1e-5
    assert not {"qk_norm", "head_dim", "linear", "rotary", "rotated"} & set(made.options)
    assert made.options["bos"] == 1 and set(made.options["stop_tokens"]) == {1, 7}
    passed = {key: value for key, value in made.options.items()
              if key in ("dtype", "arch", "bias", "convolution", "rope_theta", "rms_norm_eps")}
    llama = Llama(bytes(made.stream.out), pack_tokenizer(tiny_vocab(config["vocab_size"])), **passed)
    want = naive_lfm2_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=2e-4, atol=2e-4), pos
    # and the form that sizes the file is what the options say
    assert checkpoint_size(made.stream.header, "float32", made.options) == len(made.stream.out)
    assert form_of(made.options) == {**FORM, "arch": "lfm2", "convolution": made.options["convolution"]}
    assert (made.stream.header[5] < 0) is (not shape.get("shared", True))


def test_the_other_models_keep_the_options_they_always_had():
    """No "convolution" where there are no such layers: a saved conversion is used again only while its options are
    what the converter gives (kept.js's CONVERTER)."""
    from conftest import qwen35_model, synthetic_weights
    from test_convert import hugging_face
    from test_qwen3 import conversion as plain
    from test_qwen35 import conversion as hybrid
    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    assert set(plain(tensors, published, settings).options) == {"tokenizer_kind", "nfkc", "dtype", "rope_theta", "bos",
                                                               "stop_tokens", "bias", "arch"}
    assert "convolution" not in hybrid(*qwen35_model()).options
    assert FORM["convolution"] is None and form_of({})["convolution"] is None


# the config.json of the published models, what the FFN's rule and the layers read of them (the files at the revisions
# docs and TODO.md name: LFM2.5-350M 9e6c6ccf, LFM2.5-230M 40cb2ad3, LFM2.5-1.2B-JP-202606 52b8b447, LFM2-700M 86f49fc9,
# LFM2.5-2.6B 654f9463), and the size of w1 in their safetensors (the 350M's and the 1.2B's: 4608 and 8192)
PUBLISHED = {
    "LFM2.5-350M": (dict(hidden_size=1024, block_ff_dim=6656, intermediate_size=6656, block_auto_adjust_ff_dim=True,
                         block_ffn_dim_multiplier=1.0, block_multiple_of=256, num_hidden_layers=16, num_attention_heads=16,
                         num_key_value_heads=8, tie_embedding=True, rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
                         layer_types=["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv",
                                      "full_attention", "conv", "full_attention", "conv", "full_attention", "conv",
                                      "full_attention", "conv"]), 4608, "ccaccaccacacacac"),
    "LFM2.5-230M": (dict(hidden_size=1024, block_ff_dim=2560, intermediate_size=2560, block_auto_adjust_ff_dim=False,
                         block_ffn_dim_multiplier=1.0, block_multiple_of=256, num_hidden_layers=14, num_attention_heads=16,
                         num_key_value_heads=8, tie_embedding=True, tie_word_embeddings=True,
                         rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
                         layer_types=["conv", "conv", "full_attention"] + ["conv", "full_attention"] * 5 + ["conv"]),
                    2560, "ccacacacacacac"),
    "LFM2.5-1.2B-JP-202606": (dict(hidden_size=2048, block_ff_dim=12288, intermediate_size=12288, block_auto_adjust_ff_dim=True,
                                   block_ffn_dim_multiplier=1.0, block_multiple_of=256, num_hidden_layers=16,
                                   num_attention_heads=32, num_key_value_heads=8, tie_embedding=True, rope_theta=1000000.0,
                                   rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
                                   layer_types=["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv",
                                                "conv", "full_attention", "conv", "full_attention", "conv", "full_attention",
                                                "conv", "full_attention", "conv"]), 8192, "ccaccaccacacacac"),
    # the older spelling: the layers that attend by their numbers, theta at the top, no intermediate_size
    "LFM2-700M": (dict(hidden_size=1536, block_ff_dim=10240, block_auto_adjust_ff_dim=True, block_ffn_dim_multiplier=1.0,
                       block_multiple_of=256, num_hidden_layers=16, num_attention_heads=24, num_key_value_heads=8,
                       full_attn_idxs=[2, 5, 8, 10, 12, 14], rope_theta=1000000.0), 6912, "ccaccaccacacacac"),
    "LFM2.5-2.6B": (dict(hidden_size=2048, intermediate_size=10752, block_auto_adjust_ff_dim=False, block_ffn_dim_multiplier=1.0,
                         block_multiple_of=256, num_hidden_layers=30, num_attention_heads=32, num_key_value_heads=8,
                         tie_word_embeddings=True, rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"},
                         layer_types=["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv",
                                      "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv",
                                      "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv",
                                      "full_attention", "conv", "conv", "full_attention", "conv", "conv"]),
                    10752, "ccaccacccacccacccacccaccaccacc"),
}


@pytest.mark.parametrize("model", PUBLISHED)
def test_the_published_configs_read_as_transformers_reads_them(model):
    said, hidden, layers = PUBLISHED[model]
    common = dict(model_type="lfm2", conv_L_cache=3, conv_bias=False, norm_eps=1e-05, vocab_size=65536,
                  max_position_embeddings=128000, bos_token_id=1, eos_token_id=7)
    config = normalize({**common, **said})
    check_config(config)
    assert architecture(config) == "lfm2" and config["intermediate_size"] == hidden
    assert convolution_layers(config) == {"layers": layers, "taps": 3}
    assert config["rope_theta"] == said.get("rope_parameters", said).get("rope_theta") and "rope_scaling" not in config
    assert config["rms_norm_eps"] == 1e-5 and config["tie_word_embeddings"] is True
    assert head_size(config) == 64
    assert normalize(config) == config  # twice is once: the two thirds are not taken again


def test_the_size_of_the_ffn_follows_transformers():
    """Lfm2MLP: block_ff_dim wins over intermediate_size; two thirds of it, times the multiplier, up to a multiple, unless
    block_auto_adjust_ff_dim is off; no rounding up where the multiplier is null."""
    base = dict(model_type="lfm2", num_hidden_layers=4)
    size = lambda **said: normalize({**base, **said})["intermediate_size"]
    assert size(block_ff_dim=6656, intermediate_size=100) == 4608
    assert size(intermediate_size=6656) == 4608
    assert size(intermediate_size=6656, block_auto_adjust_ff_dim=False) == 6656
    assert size(block_ff_dim=6656, block_ffn_dim_multiplier=1.5) == 6656  # int(1.5 * 4437) = 6655, up to 256 * 26
    assert size(block_ff_dim=6656, block_multiple_of=100) == 4500
    assert size(block_ff_dim=6656, block_ffn_dim_multiplier=None) == 4437
    assert size() is None  # nothing said: refused, not guessed
    with pytest.raises(ValueError, match="no usable intermediate_size"):
        check_config(normalize({**lfm2_model()[1], "block_ff_dim": None, "intermediate_size": None}))
    # the layers where the config has no layer_types: every layer attends unless full_attn_idxs names the ones that do
    assert normalize({**base, "full_attn_idxs": [1, 2]})["layer_types"] == ["conv", "full_attention", "full_attention", "conv"]
    assert normalize(base)["layer_types"] == ["full_attention"] * 4
    # theta 1e6 where none is said, the one of rope_parameters where both are; tie_embedding over tie_word_embeddings
    assert normalize(base)["rope_theta"] == 1e6
    assert normalize({**base, "rope_theta": 5.0, "rope_parameters": {"rope_theta": 7.0}})["rope_theta"] == 7.0
    assert normalize({**base, "tie_embedding": False, "tie_word_embeddings": True})["tie_word_embeddings"] is False
    assert normalize({**base, "norm_eps": 1e-6})["rms_norm_eps"] == 1e-6


@pytest.mark.parametrize("change, reason", [
    (dict(conv_bias=True), "convolution layers have biases"),
    (dict(layer_types=["conv", "full_attention"]), "one kind for every layer"),
    (dict(layer_types=["conv", "conv", "full_attention", "conv", "conv", "linear_attention", "conv", "full_attention"]), "one kind for every layer"),
    (dict(layer_types=["conv"] * 7 + ["full_attention"]), "fewer than two"),
    (dict(layer_types=["full_attention"] * 7 + ["conv"]), "fewer than two"),
    (dict(layer_types=None), "fewer than two"),  # transformers: every layer attends
    (dict(conv_L_cache=1), "no usable conv_L_cache"),
    (dict(conv_L_cache=True), "no usable conv_L_cache"),
    (dict(hidden_act="gelu"), "activation"),
    (dict(num_key_value_heads=3), "heads do not divide"),
    (dict(head_dim=16), "heads do not divide"),
    (dict(rope_parameters={"rope_type": "yarn", "factor": 2.0, "rope_theta": 1e6}), "RoPE scaling"),
    (dict(model_type="lfm2_moe"), "it is a lfm2_moe"),
])
def test_an_lfm2_the_engine_cannot_run_is_refused(change, reason):
    _, config = lfm2_model()
    changed = {key: value for key, value in {**config, **change}.items() if value is not None}
    with pytest.raises(ValueError, match=reason):
        check_config(normalize(changed))


def test_the_layers_slots():
    form = convolution_form({"layers": "ccaca", "taps": 3}, 5)
    assert layer_slots(5, None, form) == [(True, 0), (True, 1), (False, 0), (True, 2), (False, 1)]
    assert layer_slots(3, None, None) == [(False, 0), (False, 1), (False, 2)]
    assert convolution_form(None) is None
    for wrong, layers in (({"layers": "ccax"}, None), ({"layers": ""}, None), ({"taps": 1}, None), ({}, 4)):
        with pytest.raises(ValueError, match="not the convolution layers"):
            convolution_form({"layers": "ccaca", "taps": 3, **wrong}, layers)


def test_an_engine_that_is_told_half_of_it_refuses():
    tensors, config = lfm2_model()
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), config, "float32")
    options = options_of(config)
    tokenizer = pack_tokenizer(tiny_vocab(320))
    with pytest.raises(ValueError, match="go together"):
        Llama(checkpoint, tokenizer, **{**options, "convolution": None})
    with pytest.raises(ValueError, match="go together"):
        Llama(checkpoint, tokenizer, **{**options, "arch": "llama"})
    with pytest.raises(ValueError, match="not the convolution layers"):
        Llama(checkpoint, tokenizer, **{**options, "convolution": {"layers": "ccac", "taps": 3}})
    with pytest.raises(ValueError, match="rotated basis"):
        Llama(checkpoint, tokenizer, **options, rotated={"block": 32, "signs": {"32": "00000000", "64": "0" * 16}})
    header = struct.unpack_from("<7i", checkpoint, 0)
    for wrong in (None, {"layers": "ccaccacc", "taps": 3}, {"layers": "ccaccaca", "taps": 4}, {"layers": "ccacca", "taps": 3}):
        with pytest.raises(ValueError, match="not a llama2.c checkpoint"):
            checkpoint_dtype(header, len(checkpoint), {**options, "convolution": wrong})
    # the same layers in another order are the same sizes: the options are what says which layer is which
    assert checkpoint_dtype(header, len(checkpoint), {**options, "convolution": {"layers": "acaccacc", "taps": 3}}) == "float32"


# the headers of the published models (dim, the FFN's inside, layers, heads, key-value heads, vocabulary, a context)
REAL = {"230M": ((1024, 2560, 14, 16, 8, 65536, 4096), "ccacacacacacac"),
        "350M": ((1024, 4608, 16, 16, 8, 65536, 4096), "ccaccaccacacacac"),
        "700M": ((1536, 6912, 16, 24, 8, 65536, 4096), "ccaccaccacacacac"),
        "1.2B": ((2048, 8192, 16, 32, 8, 65536, 4096), "ccaccaccacacacac"),
        "2.6B": ((2048, 10752, 30, 32, 8, 128000, 4096), "ccaccacccacccacccacccaccaccacc")}


@pytest.mark.parametrize("model", REAL)
def test_no_other_form_or_dtype_has_the_size_of_a_real_lfm2(model):
    """The legacy file says neither its dtype nor its form: its size does, with the header. A real LFM2's file of any
    dtype has the size of no other dtype of the same form, and of no dtype of another form of the same header."""
    header, layers = REAL[model]
    form = {"arch": "lfm2", "convolution": {"layers": layers, "taps": 3}}
    sizes = {dtype: checkpoint_size(header, dtype, form) for dtype in ("float32", "float16", "int8", "int6", "ternary")}
    assert len(set(sizes.values())) == 5
    others = [{}, {"bias": True}, {"qk_norm": True}, {"bias": True, "qk_norm": True}, {"arch": "gpt2"}, {"arch": "neox"},
              {"arch": "qwen35", "linear": {"every": 4, "key_heads": 16, "value_heads": 16, "key_dim": 128, "value_dim": 128, "conv": 4}}]
    for dtype, size in sizes.items():
        assert checkpoint_dtype(header, size, form) == dtype
        for other in others:
            with pytest.raises(ValueError, match="not a llama2.c checkpoint"):
                checkpoint_dtype(header, size, other)


# ------------------------------------------------------------------------------------------------ the GGUF
LFM2_NAMES = {"model.embed_tokens.weight": "token_embd.weight", "model.embedding_norm.weight": "token_embd_norm.weight",
              "lm_head.weight": "output.weight"}
LFM2_LAYER = {"operator_norm": "attn_norm", "ffn_norm": "ffn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
              "self_attn.v_proj": "attn_v", "self_attn.out_proj": "attn_output", "self_attn.q_layernorm": "attn_q_norm",
              "self_attn.k_layernorm": "attn_k_norm", "feed_forward.w1": "ffn_gate", "feed_forward.w3": "ffn_up",
              "feed_forward.w2": "ffn_down", "conv.in_proj": "shortconv.in_proj", "conv.conv": "shortconv.conv",
              "conv.out_proj": "shortconv.out_proj"}
# rows of whole groups of 32 (Q8_0)
LFM2 = dict(dim=64, block_ff_dim=192)


def lfm2_gguf(more=(), bos=1, eos=7, change=None, leave=(), **shape):
    """A GGUF v3 of a small LFM2 the way llama.cpp writes one (as LiquidAI's LFM2.5-350M Q8_0 is): Q8_0 matrices, F32
    norms and taps, the convolution's taps (channels, taps) without their axis of one, q and k as Hugging Face holds
    them, the last norm as token_embd_norm, the key-value heads a number a layer (0 for a convolution layer), the
    FFN's inside as it is. Returns the config.json, the file, and under the Hugging Face names the values it stands
    for: the matrices as Q8_0 rounds them, everything else as the original has it.
    more: further metadata; leave: metadata keys left out; change(stored): alters what is written, {GGUF name:
    [bytes, ggml type, shape]}."""
    tensors, config = lfm2_model(**{**LFM2, **shape})
    stored, same = {}, {}
    for name, tensor in tensors.items():
        if name in LFM2_NAMES:
            gguf = LFM2_NAMES[name]
        else:
            _, _, layer, *rest = name.split(".")
            gguf = f"blk.{layer}.{LFM2_LAYER['.'.join(rest[:-1])]}.{rest[-1]}"
        if tensor.ndim == 2:
            blob, held = q8_0_blocks(tensor)
            stored[gguf], same[name] = [blob, 8, tensor.shape], held
        else:
            value = tensor.reshape(tensor.shape[0], tensor.shape[-1]) if tensor.ndim == 3 else tensor
            stored[gguf], same[name] = [np.ascontiguousarray(value, np.float32).tobytes(), 0, value.shape], tensor
    if change:
        change(stored)
    arch, kinds = "lfm2", config["layer_types"]
    hidden = tensors["model.layers.0.feed_forward.w1.weight"].shape[0]
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, config["num_hidden_layers"]),
                (f"{arch}.context_length", 4, config["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, config["hidden_size"]),
                (f"{arch}.feed_forward_length", 4, hidden),
                (f"{arch}.attention.head_count", 4, config["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 9, (4, [config["num_key_value_heads"] if kind == "full_attention" else 0 for kind in kinds])),
                (f"{arch}.attention.layer_norm_rms_epsilon", 6, config["norm_eps"]),
                (f"{arch}.rope.freq_base", 6, float(config["rope_parameters"]["rope_theta"])),
                (f"{arch}.vocab_size", 4, config["vocab_size"]),
                (f"{arch}.shortconv.l_cache", 4, config["conv_L_cache"]),
                ("tokenizer.ggml.model", 8, "gpt2"), ("tokenizer.ggml.pre", 8, "lfm2"),
                *[(f"tokenizer.ggml.{key}_token_id", 4, id) for key, id in (("bos", bos), ("eos", eos)) if id is not None]]
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


LFM2_SHAPES = {"the 350M's order": dict(kinds="ccaccaccacacacac"), "a classifier of its own": dict(shared=False, kinds="ccacca"),
               "four taps, the size of the FFN as it is": dict(taps=4, adjust=False, block_ff_dim=96, kinds="accacc"),
               "another epsilon": dict(eps=1e-6, kinds="caca")}


@pytest.mark.parametrize("dtype", ["int8", "float32", "float16", "int6"])
@pytest.mark.parametrize("shape", LFM2_SHAPES)
def test_an_lfm2_gguf_with_the_originals_files_is_the_safetensors_conversion(shape, dtype, monkeypatch):
    """The list's way in: the checkpoint, tokenizer.bin and options of the safetensors of the same values, to the byte.
    The convolution comes without its axis of one, the last norm and the convolution's tensors under llama.cpp's
    names, and nothing in another order. Fed 4096 bytes at a time, in pieces of a few rows."""
    monkeypatch.setattr("convert.stream.PIECE", 700)
    config, file, same = lfm2_gguf(**LFM2_SHAPES[shape])
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
    expected = safetensors_conversion(same, config, vocabulary, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert got.options["arch"] == "lfm2" and got.options["convolution"]["layers"] == LFM2_SHAPES[shape]["kinds"]
    # and it is the model: the engine on the GGUF's values runs like the reference on them
    if dtype == "float32":
        llama = Llama(bytes(got.checkpoint), pack_tokenizer(tiny_vocab(config["vocab_size"])), **options_of(config))
        want = naive_lfm2_logits(same, config, TOKENS)
        for pos, token in enumerate(TOKENS):
            assert np.allclose(llama.forward(token, pos), want[pos], rtol=2e-4, atol=2e-4), pos


def test_an_lfm2_gguf_alone_converts_to_the_checkpoint_of_the_same_values():
    """Such a GGUF alone: the layers, the taps, the FFN's size and theta come from its own metadata, by config.json's
    names, and its pre-tokenizer is the one of an LFM2's tokenizer.json (Llama 3's pattern, and no piece taken whole
    past its merges: tokenizer.json's ignore_merges is false, whatever llama.cpp does with the name)."""
    for shape in ({}, dict(eps=1e-6, taps=2, kinds="acca")):
        config, file, same = lfm2_gguf(**shape)
        made = fed(file, "int8")
        expected = safetensors_conversion(same, config, unigram(config["vocab_size"]), "int8")
        assert bytes(made.checkpoint) == bytes(expected.checkpoint)
        for key in ("arch", "convolution", "rope_theta", "bos", "stop_tokens", "bias"):
            assert made.options[key] == expected.options[key], key
        assert made.options.get("rms_norm_eps") == expected.options.get("rms_norm_eps") == shape.get("eps")
        assert made.options["pretokenizer"] == "llama3" and made.options["ignore_merges"] is False
        assert made.options["nfc"] is False and "head_dim" not in made.options
    with pytest.raises(ValueError, match="names no BOS token"):
        fed(lfm2_gguf(bos=None)[1], "int8")
    # the taps where the GGUF does not say them are transformers' 3
    assert fed(lfm2_gguf(leave=("lfm2.shortconv.l_cache",))[1], "int8").options["convolution"]["taps"] == 3


@pytest.mark.parametrize("change, what", [
    (dict(layer_types=["conv", "full_attention"] * 4), "convolution layers"),
    (dict(layer_types=["conv", "conv", "full_attention", "conv", "full_attention", "conv", "conv", "full_attention"]), "convolution layers"),
    (dict(conv_L_cache=4), "convolution layers"),
    (dict(block_ff_dim=96), "size of the FFN"),
    (dict(block_auto_adjust_ff_dim=False), "size of the FFN"),
    (dict(rope_parameters={"rope_type": "default", "rope_theta": 10000.0}), "RoPE theta"),
    (dict(num_attention_heads=8, num_key_value_heads=4), "number of heads"),
    (dict(num_key_value_heads=4), "number of key-value heads"),
    (dict(norm_eps=1e-6), "RMSNorm epsilon"),
    (dict(num_hidden_layers=7, layer_types=["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv"]), "number of layers"),
    (dict(tie_embedding=False), "classifier of its own"),
    (dict(model_type="qwen3"), "architecture"),
])
def test_an_lfm2_gguf_that_is_not_the_originals_is_refused(change, what):
    """What the tensors do not say (or say only as they stream past): which layers are convolution layers, their taps,
    heads of the same product, theta, the epsilon."""
    config, file, _ = lfm2_gguf()
    llama2_convert.gguf_weights(file, json.dumps(config))  # its own config goes through
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(file, json.dumps({**config, **change}))


def test_an_lfm2_gguf_whose_layers_have_other_heads_is_refused():
    """llama.cpp says the key-value heads a layer; one number for all the attention layers is what the engine has."""
    kinds = lfm2_model(**LFM2)[1]["layer_types"]
    counts = [2 if kind == "full_attention" else 0 for kind in kinds]
    counts[2] = 4
    for said, why in ((counts, r"\[2, 4\] key-value heads"), ([0] * len(kinds), "no key-value heads")):
        _, file, _ = lfm2_gguf(leave=("lfm2.attention.head_count_kv",), more=[("lfm2.attention.head_count_kv", 9, (4, said))])
        with pytest.raises(ValueError, match=why):
            fed(file, "int8")


def test_a_convolution_stored_another_way_is_refused():
    """The taps as (taps, channels), or a matrix in of another size: the shape says so before a value is converted."""
    def turned(stored):
        blob, kind, shape = stored["blk.0.shortconv.conv.weight"]
        stored["blk.0.shortconv.conv.weight"] = [blob, kind, tuple(reversed(shape))]
    config, file, _ = lfm2_gguf(change=turned)
    with pytest.raises(ValueError, match=r"conv.conv.weight is \(3, 1, 64\), not \(64, 1, 3\)"):
        with_original(file, config, unigram(config["vocab_size"]), "tokenizer.json", "int8")
