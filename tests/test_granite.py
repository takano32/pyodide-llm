"""A Granite (T253: IBM's, transformers' model_type "granite") is a Llama whose attention multiplies its scores by
config.json's attention_multiplier where a Llama divides them by the root of a head's size. The converter multiplies q
by attention_multiplier * sqrt(head) (llama2_convert.query_scale), and the file, the options and the engine are a
Llama's. Its other three multipliers (the embedding's, the branches', the logits') the engine has not: a model whose
are not 1 is refused. The real model against transformers: tests/reference_llama.py, in CI."""
import json
import math
import struct

import numpy as np
import pytest
from conftest import naive_logits, pack_tokenizer, synthetic_weights, tiny_vocab
from test_convert import converted, hugging_face, reader, safetensors_file, streamed
from test_gguf import fed, gguf_file, unigram, with_original

import llama2_convert
from llama2_convert import Conversion, Safetensors, check_config, query_scale
from llama2_numpy import Llama

ONES = dict(embedding_multiplier=1.0, residual_multiplier=1.0, logits_scaling=1.0)
# dim 32 in 4 heads of 8: 0.0625 * sqrt(8) is no power of two (Granite 4.2 8B's kind); dim 64 in 4 heads of 16:
# 1/64 * 4 is one (Granite 4.2 3B's kind, 1/64 * 8); grouped queries and a classifier of its own, as both have
MODELS = [(dict(), 0.0625), (dict(n_kv_heads=2, shared=False), 0.3), (dict(dim=64, n_kv_heads=1, shared=False), 1 / 64)]


def granite(settings, weights, multiplier, **more):
    """What Hugging Face would publish for a Granite of these weights (a Llama's tensors, by a Llama's names)."""
    tensors, published = hugging_face(settings, weights, settings["shared"])
    return tensors, {**published, "model_type": "granite", "attention_multiplier": multiplier, **ONES, **more}


def scaled_llama(settings, weights, multiplier):
    """The Llama that computes what the Granite does: the same tensors but q, which is multiplied in float32."""
    tensors, published = hugging_face(settings, weights, settings["shared"])
    scale = np.float32(multiplier * math.sqrt(settings["head_size"]))
    return {name: tensor * scale if ".q_proj." in name else tensor for name, tensor in tensors.items()}, published


def conversion(tensors, published, dtype="float32"):
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    made = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), unigram(published["vocab_size"]),
                      "tokenizer.json", dtype=dtype, max_seq_len=1 << 20, start=8 + size)
    made.feed(file[8 + size:])
    made.finish()
    return made


@pytest.mark.parametrize("config, multiplier", MODELS)
def test_a_granite_converts_and_runs_like_the_reference(config, multiplier):
    settings, weights = synthetic_weights(**config)
    tensors, published = granite(settings, weights, multiplier)
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), published, "float32")
    llama = Llama(checkpoint, pack_tokenizer(tiny_vocab(settings["vocab_size"])))
    tokens = [1, 5, 7, 9, 5, 11]
    want = naive_logits({**settings, "attention_multiplier": multiplier}, weights, tokens)
    as_a_llama = naive_logits(settings, weights, tokens)
    for pos, token in enumerate(tokens):
        got = llama.forward(token, pos)
        assert np.allclose(got, want[pos], rtol=1e-4, atol=1e-4)
        # and the reference is not blind to the multiplier (past the first position, which attends to itself alone)
        assert pos == 0 or not np.allclose(got, as_a_llama[pos], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("config, multiplier", MODELS)
@pytest.mark.parametrize("dtype, stored", [("float32", "F32"), ("int8", "BF16"), ("int6", "F32"), ("float16", "F16")])
def test_it_is_the_llama_whose_q_is_scaled_and_nothing_else_knows(config, multiplier, dtype, stored):
    """The very checkpoint and the very options of the Llama with q multiplied (in float32, after the stored type is
    widened): no key of the options and no byte outside q says it was a Granite, so the engine, forward.js and the GPU
    run it as the Llama it is."""
    settings, weights = synthetic_weights(**config)
    if stored != "F32":  # the values the stored type holds, so that both sides start from the same float32
        widen = {"BF16": lambda w: (w.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32),
                 "F16": lambda w: w.astype(np.float16).astype(np.float32)}[stored]
        weights = {name: widen(np.ascontiguousarray(value, dtype=np.float32)) for name, value in weights.items()}
    tensors, published = granite(settings, weights, multiplier)
    plain, llama = scaled_llama(settings, weights, multiplier)
    file = safetensors_file(tensors, stored)
    expected = converted(Safetensors(reader(safetensors_file(plain))), llama, dtype)
    assert converted(Safetensors(reader(file)), published, dtype) == expected
    # the file in its own order (the page's way) is the same checkpoint
    assert streamed(file, published, dtype, 4096)[0] == expected
    assert conversion(tensors, published, dtype).options == conversion(plain, llama, dtype).options


def test_a_multiplier_of_one_over_the_root_of_a_head_is_a_llama_to_the_byte():
    settings, weights = synthetic_weights(dim=64)  # heads of 16: 0.25
    tensors, published = granite(settings, weights, 0.25)
    assert query_scale(published) == 1.0
    _, llama = hugging_face(settings, weights, True)
    for dtype in ("float32", "int8"):
        assert converted(Safetensors(reader(safetensors_file(tensors))), published, dtype) == \
            converted(Safetensors(reader(safetensors_file(tensors))), llama, dtype)


def test_only_a_granite_is_scaled():
    """The multiplier is read where the model is a Granite: a Llama's config.json that carried the same keys (which
    transformers' Llama does not read) converts as before, to the byte."""
    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    carried = {**published, "attention_multiplier": 0.0625, "embedding_multiplier": 12.0}
    assert query_scale(carried) == 1.0 and query_scale({**carried, "model_type": "qwen3"}) == 1.0
    assert query_scale({**carried, "model_type": "granite"}) == pytest.approx(0.0625 * math.sqrt(8))
    source = lambda: Safetensors(reader(safetensors_file(tensors)))
    assert converted(source(), carried, "int8") == converted(source(), published, "int8")


def test_a_granite_that_names_no_attention_multiplier_has_transformers_default_of_one():
    """transformers' GraniteConfig leaves attention_multiplier 1.0 where config.json names none, so its scores are not divided
    by the root of the head's size at all (the review of T253: a default of one over the root, a Llama's, passed every test):
    q is multiplied by the whole root, as where config.json says 1.0 itself. llama.cpp writes no attention.scale then and
    divides by the root: test_a_granite_gguf_that_names_no_scale_scores_as_a_llama refuses that pair."""
    settings, weights = synthetic_weights()
    tensors, published = granite(settings, weights, 1.0)
    without = {key: value for key, value in published.items() if key != "attention_multiplier"}
    assert query_scale(without) == query_scale(published) == pytest.approx(math.sqrt(settings["head_size"]))
    check_config(without)
    source = lambda: Safetensors(reader(safetensors_file(tensors)))
    assert converted(source(), without, "float32") == converted(source(), published, "float32")


@pytest.mark.parametrize("change, reason", [
    (dict(embedding_multiplier=12.0), "embedding_multiplier is 12.0"), (dict(residual_multiplier=0.22), "residual_multiplier is 0.22"),
    (dict(logits_scaling=10.0), "logits_scaling is 10.0"), (dict(logits_scaling=None), "logits_scaling is None"),
    (dict(attention_multiplier=0), "attention_multiplier"), (dict(attention_multiplier=-0.5), "attention_multiplier"),
    (dict(attention_multiplier=None), "attention_multiplier"), (dict(attention_multiplier="0.5"), "attention_multiplier"),
    (dict(attention_multiplier=float("inf")), "attention_multiplier"), (dict(attention_multiplier=True), "attention_multiplier"),
    (dict(head_dim=16), "heads do not divide"), (dict(attention_bias=True), "biases"), (dict(hidden_act="gelu"), "gelu")])
def test_what_the_engine_has_not_of_a_granite_is_refused(change, reason):
    """Granite 3.x and 4.1 scale the embedding, the branches and the logits (12, 0.22, 10 or so): converted without
    them they would write nonsense without a word. transformers' Granite has no head_dim either."""
    settings, weights = synthetic_weights()
    _, published = granite(settings, weights, 0.0625)
    check_config(published)
    with pytest.raises(ValueError, match=reason):
        check_config({**published, **change})


@pytest.mark.parametrize("what", ["norm", "bias"])
def test_a_scale_of_q_under_a_norm_or_beside_a_bias_is_refused(what):
    """A norm of q's heads (Qwen3's) undoes whatever q was multiplied by, and a bias of q (Qwen2's) would have to be
    multiplied with it: tensors no Granite has, and a file that had them would otherwise go through as another model."""
    from test_bias import qwen2
    from test_qwen3 import qwen3
    settings, weights = synthetic_weights()
    tensors, published = (qwen3 if what == "norm" else qwen2)(settings, weights, True)
    published = {**published, "model_type": "granite", "attention_multiplier": 0.0625, **ONES}
    file = safetensors_file(tensors)
    with pytest.raises(ValueError, match="a bias or a norm on its queries"):
        converted(Safetensors(reader(file)), published, "float32")
    with pytest.raises(ValueError, match="a bias or a norm on its queries"):
        streamed(file, published, "float32", 4096)


# ---- a Granite's GGUF: llama.cpp keeps the multiplier in the metadata (granite.attention.scale) and leaves q as it
# is, turned like a Llama's: the conversion scales it once, whichever file the weights came from
def granite_gguf(multiplier=0.0625, config=None, scales=None, **keys):
    settings, weights = synthetic_weights(**{"n_kv_heads": 2, "shared": False, **(config or {})})
    tensors, published = granite(settings, weights, multiplier)
    published["rms_norm_eps"] = 1e-5
    said = {"attention.scale": multiplier, "embedding_scale": 1.0, "residual_scale": 1.0, "logit_scale": 1.0, **(scales or {})}
    more = [("granite.attention.layer_norm_rms_epsilon", 6, 1e-5),
            *[(f"granite.{key}", 6, value) for key, value in said.items() if value is not None]]
    file, same = gguf_file(tensors, published, settings["vocab_size"], "granite", pre="granite-docling", more=more, **keys)
    return settings, published, file, same


@pytest.mark.parametrize("dtype", ["int8", "float32", "int6"])
@pytest.mark.parametrize("multiplier", [0.0625, 0.3])
def test_a_granite_gguf_with_the_originals_files_is_the_safetensors_conversion(multiplier, dtype):
    settings, published, file, same = granite_gguf(multiplier)
    vocabulary = unigram(settings["vocab_size"])
    got = with_original(file, published, vocabulary, "tokenizer.json", dtype)
    expected = conversion(same, published, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer) and got.options == expected.options
    # q was turned in the GGUF and is scaled once: the Llama of the same values would be another checkpoint
    assert bytes(got.checkpoint) != bytes(conversion(same, {**published, "model_type": "llama"}, dtype).checkpoint)


def test_a_granite_gguf_alone_converts_to_the_checkpoint_of_the_same_values():
    """Its own metadata: the architecture's name, the scale of the scores (a float32, which 0.0625 is to the bit) and
    llama.cpp's name of its pre-tokenizer, granite-docling, which splits as GPT-2's pattern does."""
    settings, published, file, same = granite_gguf()
    got = fed(file, "int8")
    assert bytes(got.checkpoint) == converted(Safetensors(reader(safetensors_file(same))), published, "int8")
    assert got.options["arch"] == "llama" and got.options["pretokenizer"] == "gpt2"
    assert not {"attention_multiplier", "scale", "model_type"} & set(got.options)


def test_a_granite_gguf_that_names_no_scale_scores_as_a_llama():
    """llama.cpp: where the GGUF names no attention.scale the scores are divided by the root of a head's size."""
    settings, published, file, same = granite_gguf(1 / math.sqrt(8), scales={"attention.scale": None})
    got = fed(file, "float32")
    llama = {**published, "model_type": "llama"}
    assert np.allclose(np.frombuffer(bytes(got.checkpoint), np.float32, offset=28),
                       np.frombuffer(converted(Safetensors(reader(safetensors_file(same))), llama, "float32"), np.float32, offset=28),
                       rtol=1e-6, atol=0)
    with pytest.raises(ValueError, match="Granite's multipliers"):  # transformers takes 1.0 where config.json names none
        llama2_convert.gguf_weights(file, json.dumps({key: value for key, value in published.items() if key != "attention_multiplier"}))


@pytest.mark.parametrize("change, scales, what", [
    (dict(attention_multiplier=0.125), None, "Granite's multipliers"),
    (dict(), {"embedding_scale": 12.0}, "Granite's multipliers"),
    (dict(), {"residual_scale": 0.22}, "Granite's multipliers"),
    (dict(), {"logit_scale": 10.0}, "Granite's multipliers"),
    (dict(model_type="llama"), None, "architecture"),
    (dict(num_hidden_layers=1), None, "number of layers"),
])
def test_a_granite_gguf_that_is_not_the_originals_is_refused(change, scales, what):
    settings, published, file, _ = granite_gguf(scales=scales)
    if not scales:
        llama2_convert.gguf_weights(file, json.dumps(published))  # its own config goes through
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(file, json.dumps({**published, **change}))


def test_a_llama_gguf_is_no_granites_weights_and_a_granite_gguf_of_other_multipliers_is_refused_alone():
    settings, weights = synthetic_weights(n_kv_heads=2, shared=False)
    tensors, published = hugging_face(settings, weights, False)
    file, _ = gguf_file(tensors, published, settings["vocab_size"])
    with pytest.raises(ValueError, match="architecture"):
        llama2_convert.gguf_weights(file, json.dumps({**published, "model_type": "granite", "attention_multiplier": 0.0625}))
    # Granite 4.1's kind, read alone: what config.json would be refused for
    _, _, file, _ = granite_gguf(scales={"embedding_scale": 12.0})
    with pytest.raises(ValueError, match="embedding_multiplier is 12.0"):
        fed(file, "int8")


@pytest.mark.parametrize("wrong", [None, "attention_multiplier", "embedding_multiplier", "a layer's q"])
def test_gguf_check_holds_a_granite_gguf_to_its_original_and_its_multipliers(tmp_path, capsys, wrong):
    """tests/gguf_check.py tensors (gguf.yml's candidates), the separate reference a GGUF is held to before the list
    takes it: a Granite's passes against its original with q found turned, and one whose metadata says another
    multiplier than config.json, or whose q is other values, does not."""
    import gguf_check
    settings, published, file, same = granite_gguf()
    original = {name: tensor.copy() for name, tensor in same.items()}
    if wrong == "a layer's q":
        original["model.layers.1.self_attn.q_proj.weight"] = original["model.layers.1.self_attn.q_proj.weight"][::-1].copy()
    elif wrong:
        published = {**published, wrong: 0.5}
    (tmp_path / "model.gguf").write_bytes(file)
    (tmp_path / "model.safetensors").write_bytes(safetensors_file(original))
    (tmp_path / "config.json").write_text(json.dumps(published))
    assert gguf_check.check_tensors(tmp_path / "model.gguf", tmp_path) is (wrong is None)
    out = capsys.readouterr().out
    assert (json.loads(out.strip().splitlines()[-1])["mismatches"] == 0) is (wrong is None)
    assert "| granite.attention.scale | 0.0625 | attention_multiplier = " in out and "turned (llama2.c order)" in out
