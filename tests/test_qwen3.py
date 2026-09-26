"""Qwen3 is a Llama that normalizes every head of q and k before RoPE (T124): the converter writes the two RMSNorm
weights (one head's size) per layer after everything else, and the engine applies them. The file keeps the 7 int
header, so the caller passes qk_norm=True, as it passes bias=True for a Qwen2. Some Qwen3 (0.6B, 4B) and some
Llamas (MiniCPM5 1B) have heads of another size than dim / n_heads: head_dim, which the caller passes too."""
import inspect
import json
import struct

import numpy as np
import pytest
from conftest import naive_logits, pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab
from test_convert import converted, hugging_face, reader, safetensors_file, streamed

import llama2_convert
from llama2_convert import Safetensors, has_qk_norm, permute_heads
from llama2_numpy import FORM, Llama, checkpoint_dtype, form_of

# head_size 16: q twice as wide as dim (Qwen3 0.6B's 16 heads of 128 in 1024); 4: half as wide
CONFIGS = [dict(n_kv_heads=4), dict(n_kv_heads=2), dict(n_kv_heads=1, shared=False), dict(n_kv_heads=2, head_size=16),
           dict(n_kv_heads=4, head_size=4, shared=False)]


def options_of(settings):
    """What the caller passes to Llama() for a file of these settings (the converter's options say the same)."""
    return {"qk_norm": True, **({"head_dim": settings["head_size"]} if settings["q_dim"] != settings["dim"] else {})}


def qwen3(config, weights, shared, seed=11):
    """What Hugging Face would publish for a Qwen3 with these weights: the norms of q and k, in Hugging Face's order
    of a head (first halves, then second halves), which the converter interleaves like the rows of wq and wk."""
    tensors, published = hugging_face(config, weights, shared)
    head_size = config["head_size"]
    rng = np.random.default_rng(seed)
    for name in ("q_norm", "k_norm"):
        weights[name] = (1.0 + rng.standard_normal((config["n_layers"], head_size)) * 0.3).astype(np.float32)
        for layer in range(config["n_layers"]):
            ours = weights[name][layer]
            tensors[f"model.layers.{layer}.self_attn.{name}.weight"] = \
                np.ascontiguousarray(ours.reshape(head_size // 2, 2).T.reshape(-1))
    return tensors, {**published, "model_type": "qwen3"}


def conversion(tensors, published, settings):
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(settings["vocab_size"])]}}).encode()
    made = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary,
                                     "tokenizer.json", dtype="float32", max_seq_len=settings["seq_len"], start=8 + size)
    made.stream.feed(file[8 + size:])
    made.stream.finish()
    return made


@pytest.mark.parametrize("config", CONFIGS)
def test_a_qwen3_converts_and_runs_like_the_reference(config):
    shared = config.pop("shared", True)
    settings, weights = synthetic_weights(shared=shared, **config)
    tensors, published = qwen3(settings, weights, shared)
    source = Safetensors(reader(safetensors_file(tensors)))
    assert has_qk_norm(source)
    checkpoint = converted(source, published, "float32")
    # two vectors of a head's size per layer more than the same model without them, and the header says nothing
    plain = pack_checkpoint(settings, weights)
    assert len(checkpoint) == len(plain) + 4 * settings["n_layers"] * 2 * settings["head_size"]
    assert checkpoint[:28] == plain[:28]
    options = options_of(settings)
    assert checkpoint_dtype(struct.unpack_from("<7i", checkpoint, 0), len(checkpoint), options) == "float32"

    llama = Llama(checkpoint, pack_tokenizer(tiny_vocab(settings["vocab_size"])), **options)
    tokens = [1, 5, 7, 9]
    want = naive_logits(settings, weights, tokens)
    for pos, token in enumerate(tokens):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=1e-4, atol=1e-4)


def test_the_norms_matter():
    """Without them the numbers are another model's: the reference is not blind to them (at a later position: the
    first attends only to itself, whatever q and k are)."""
    settings, weights = synthetic_weights()
    tensors, published = qwen3(settings, weights, True)
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), published, "float32")
    llama = Llama(checkpoint, pack_tokenizer(tiny_vocab(settings["vocab_size"])), qk_norm=True)
    plain = {name: value for name, value in weights.items() if name not in ("q_norm", "k_norm")}
    llama.forward(5, 0)
    assert not np.allclose(llama.forward(7, 1), naive_logits(settings, plain, [5, 7])[1], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("head_size", [0, 16])
@pytest.mark.parametrize("dtype", ["float32", "int8", "int6"])
def test_the_file_in_its_own_order_gives_the_same_checkpoint(dtype, head_size):
    settings, weights = synthetic_weights(head_size=head_size)
    tensors, published = qwen3(settings, weights, True)
    file = safetensors_file(tensors)
    expected = converted(Safetensors(reader(file)), published, dtype)
    got, progress = streamed(file, published, dtype, 4096)
    assert got == expected and progress[-1][0] == progress[-1][1]
    assert checkpoint_dtype(struct.unpack_from("<7i", got, 0), len(got), options_of(settings)) == dtype


@pytest.mark.parametrize("head_size", [0, 16])
def test_the_conversion_tells_the_engine_about_the_norms_and_the_heads(head_size):
    """The file cannot say it has them, nor the size of a head: the options must (the lesson of T72). And the engine
    built from exactly those options runs like the reference."""
    settings, weights = synthetic_weights(head_size=head_size)
    tensors, published = qwen3(settings, weights, True)
    made = conversion(tensors, published, settings)
    assert made.options["qk_norm"] is True and made.options["arch"] == "llama"
    assert made.options.get("head_dim") == (head_size or None)
    options = {key: value for key, value in made.options.items() if key in ("dtype", "bias", "arch", "qk_norm", "head_dim")}
    llama = Llama(bytes(made.stream.out), pack_tokenizer(tiny_vocab(settings["vocab_size"])), **options)
    want = naive_logits(settings, weights, [5, 7])
    for pos, token in enumerate([5, 7]):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=1e-4, atol=1e-4)


def test_a_model_without_them_keeps_the_options_it_always_had():
    """No qk_norm key where there are no norms: what a conversion of every model before T124 gives stays the same
    (a saved conversion is used again only while its options are what the converter gives, kept.js's CONVERTER)."""
    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    assert "qk_norm" not in conversion(tensors, published, settings).options


def test_a_llama_with_heads_of_another_size_needs_no_norms():
    """MiniCPM5 1B is such a Llama (128 against dim / heads = 96): head_dim alone, without the norms of Qwen3."""
    settings, weights = synthetic_weights(n_kv_heads=2, head_size=16)
    tensors, published = hugging_face(settings, weights, True)
    made = conversion(tensors, published, settings)
    assert "qk_norm" not in made.options and made.options["head_dim"] == 16
    llama = Llama(bytes(made.stream.out), pack_tokenizer(tiny_vocab(settings["vocab_size"])), head_dim=16)
    want = naive_logits(settings, weights, [5, 7, 9])
    for pos, token in enumerate([5, 7, 9]):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=1e-4, atol=1e-4)


def test_heads_that_do_not_fill_dim_are_said_where_dim_over_heads_rounds_to_them():
    """T144: head_dim goes into the options where heads * head_dim is not dim, not only where dim // heads is not
    head_dim. 4 heads of 8 in a dim of 34 round to 8: without head_dim, checkpoint_dtype() took such a file for no
    checkpoint at all and footprint() counted heads of 8.5 (no real model of the kind is known)."""
    settings, weights = synthetic_weights(dim=34, n_kv_heads=2, head_size=8)
    tensors, published = hugging_face(settings, weights, True)
    published["head_dim"] = 8  # hugging_face() writes it only where it is not dim // heads
    made = conversion(tensors, published, settings)
    assert made.options["head_dim"] == 8
    checkpoint = bytes(made.stream.out)
    assert checkpoint_dtype(struct.unpack_from("<7i", checkpoint, 0), len(checkpoint), made.options) == "float32"
    llama = Llama(checkpoint, pack_tokenizer(tiny_vocab(settings["vocab_size"])), **form_of(made.options))
    want = naive_logits(settings, weights, [5, 7, 9])
    for pos, token in enumerate([5, 7, 9]):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=1e-4, atol=1e-4)


def test_the_form_has_one_set_of_defaults():
    """T144: FORM is what the file cannot say, with the value of a form that says nothing. Everything that takes it by
    its names must mean the same by nothing, or a model whose options leave one out is laid out two ways."""
    for function in (llama2_convert.layout, Llama.__init__):
        parameters = inspect.signature(function).parameters
        assert {key: parameters[key].default for key in FORM} == FORM, function.__qualname__
    assert form_of(None) == form_of({"tokenizer_kind": "bpe"}) == FORM


def test_heads_of_the_wrong_size_are_never_interleaved():
    """permute_heads() reshaped rows of another head size without a word and moved rows across heads (the review of
    T124: half the rows of a Qwen3 0.6B's wq). Now it refuses."""
    rows = np.arange(64 * 3, dtype=np.float32).reshape(64, 3)
    permute_heads(rows, 4, 16)
    with pytest.raises(ValueError, match="not 4 heads of 8"):
        permute_heads(rows, 4, 8)
    with pytest.raises(ValueError):
        permute_heads(np.zeros(16, dtype=np.float32), 1, 8)


def test_a_head_size_that_is_not_the_one_of_the_file_is_refused():
    """The size check sees a head_dim that is not the file's (and the engine then never reads the wrong rows)."""
    settings, weights = synthetic_weights(head_size=16)
    tensors, published = qwen3(settings, weights, True)
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), published, "float32")
    header = struct.unpack_from("<7i", checkpoint, 0)
    with pytest.raises(ValueError, match="not a llama2.c checkpoint"):
        checkpoint_dtype(header, len(checkpoint), {"qk_norm": True})


def test_gpt2_and_neox_keep_heads_of_dim_over_heads():
    """Their fused q, k and v are cut into heads of dim / heads: another head_dim is refused, not guessed."""
    from llama2_convert import check_config
    settings, weights = synthetic_weights()
    _, published = hugging_face(settings, weights, True)
    for model_type in ("gpt_neox",):
        with pytest.raises(ValueError, match="heads"):
            check_config({**published, "model_type": model_type, "head_dim": 16, "hidden_act": "gelu"})
    with pytest.raises(ValueError, match="heads"):
        check_config({**published, "num_attention_heads": 5})
    check_config({**published, "num_attention_heads": 5, "num_key_value_heads": 5, "head_dim": 8})


def test_the_epsilon_of_the_config_reaches_the_engine():
    """rms_norm_eps goes from config.json through the options to every RMSNorm (T124: Qwen3's 1e-6 against the 1e-5
    the engine always had moved its perplexity by 0.12%). A large one, so that the reference sees it."""
    settings, weights = synthetic_weights(n_kv_heads=2, head_size=16)
    tensors, published = qwen3(settings, weights, True)
    made = conversion(tensors, {**published, "rms_norm_eps": 0.5}, settings)
    assert made.options["rms_norm_eps"] == 0.5
    options = {key: made.options[key] for key in ("qk_norm", "head_dim", "rms_norm_eps")}
    llama = Llama(bytes(made.stream.out), pack_tokenizer(tiny_vocab(settings["vocab_size"])), **options)
    want, other = (naive_logits({**settings, "eps": eps}, weights, [5, 7]) for eps in (0.5, 1e-5))
    for pos, token in enumerate([5, 7]):
        got = llama.forward(token, pos)
        assert np.allclose(got, want[pos], rtol=1e-4, atol=1e-4) and not np.allclose(got, other[pos], rtol=1e-3, atol=1e-3)
    # the engine's own 1e-5 is not written down: the options of every model of 1e-5 stay what they were
    assert "rms_norm_eps" not in conversion(tensors, {**published, "rms_norm_eps": 1e-5}, settings).options
