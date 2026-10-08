"""A SmolLM3 (T255: Hugging Face's, transformers' model_type "smollm3") is a Llama some of whose layers RoPE leaves
alone: every fourth of the published 3B, the layers where config.json's no_rope_layers has a 0. The file is a Llama's
to the byte; the conversion's options name the layers (unturned), and the engine, forward.js and the GPU turn nothing
of q and k there. The real model against transformers: tests/reference_llama.py, in CI."""
import json
import struct

import numpy as np
import pytest
from conftest import naive_logits, pack_tokenizer, synthetic_weights, tiny_vocab
from test_convert import hugging_face, safetensors_file
from test_external import Outside
from test_gguf import fed, gguf_file, unigram, with_original

import llama2_convert
from llama2_convert import Conversion, check_config, normalize, unturned_layers
from llama2_numpy import Llama

# the published 3B's kind (grouped queries, the classifier shared), a list that is no interval, and a classifier of
# its own with the first layer left alone
MODELS = [(dict(n_layers=8, n_kv_heads=2), dict(no_rope_layer_interval=4, no_rope_layers=[1, 1, 1, 0, 1, 1, 1, 0]), [3, 7]),
          (dict(n_layers=5, n_kv_heads=2), dict(no_rope_layers=[1, 0, 0, 1, 0]), [1, 2, 4]),
          (dict(n_layers=4, shared=False), dict(no_rope_layer_interval=2), [1, 3]),
          (dict(n_layers=3, n_kv_heads=1, shared=False), dict(no_rope_layers=[0, 1, 1]), [0])]
TOKENS = [1, 5, 7, 9, 5, 11, 7, 3]


def smollm3(settings, weights, **said):
    """What Hugging Face would publish for a SmolLM3 of these weights (a Llama's tensors, by a Llama's names)."""
    tensors, published = hugging_face(settings, weights, settings["shared"])
    return tensors, {**published, "model_type": "smollm3", "use_sliding_window": False, **said}


def conversion(tensors, published, dtype="float32"):
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    made = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), unigram(published["vocab_size"]),
                      "tokenizer.json", dtype=dtype, max_seq_len=1 << 20, start=8 + size)
    made.feed(file[8 + size:])
    made.finish()
    return made


@pytest.mark.parametrize("config, said, unturned", MODELS)
def test_a_smollm3_converts_and_runs_like_the_reference(config, said, unturned):
    settings, weights = synthetic_weights(**config)
    tensors, published = smollm3(settings, weights, **said)
    made = conversion(tensors, published)
    assert made.options["unturned"] == unturned and made.options["arch"] == "llama"
    llama = Llama(bytes(made.checkpoint), pack_tokenizer(tiny_vocab(settings["vocab_size"])), unturned=made.options["unturned"])
    want = naive_logits({**settings, "unturned": unturned}, weights, TOKENS)
    as_a_llama = naive_logits(settings, weights, TOKENS)
    wrong = sorted({(layer + 1) % settings["n_layers"] for layer in unturned})  # the layers after them
    others = naive_logits({**settings, "unturned": wrong}, weights, TOKENS)
    for pos, token in enumerate(TOKENS):
        got = llama.forward(token, pos)
        assert np.allclose(got, want[pos], rtol=1e-4, atol=1e-4)
        # and the reference is not blind to which layers turn (past the first position, where every angle is 0)
        assert pos == 0 or not np.allclose(got, as_a_llama[pos], rtol=1e-3, atol=1e-3)
        assert pos == 0 or wrong == unturned or not np.allclose(got, others[pos], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8", "int6"])
def test_the_file_is_the_llamas_and_only_the_options_know(dtype):
    """Turning the rows of q and k into llama2.c's order changes no score of a layer that turns nothing, so the
    conversion does to a SmolLM3 what it does to a Llama."""
    config, said, unturned = MODELS[0]
    settings, weights = synthetic_weights(**config)
    tensors, published = smollm3(settings, weights, **said)
    made, llama = conversion(tensors, published, dtype), conversion(tensors, {**published, "model_type": "llama"}, dtype)
    assert bytes(made.checkpoint) == bytes(llama.checkpoint) and bytes(made.tokenizer) == bytes(llama.tokenizer)
    assert {**llama.options, "unturned": unturned} == made.options and "unturned" not in llama.options


def test_the_layers_are_the_lists_zeros_or_every_interval_th():
    layers = dict(num_hidden_layers=12)
    assert unturned_layers({"model_type": "smollm3", **layers}) == [3, 7, 11]  # transformers' default interval, 4
    assert unturned_layers({"model_type": "smollm3", "no_rope_layer_interval": 5, **layers}) == [4, 9]
    assert unturned_layers({"model_type": "smollm3", "no_rope_layer_interval": 4, "no_rope_layers": [0] + [1] * 11, **layers}) == [0]
    assert unturned_layers({"model_type": "smollm3", "no_rope_layers": [1] * 12, **layers}) == []
    # only a SmolLM3's: the same keys in a Llama's config.json name nothing
    assert unturned_layers({"model_type": "llama", "no_rope_layers": [0] * 12, **layers}) == []


def test_a_smollm3_that_turns_every_layer_has_a_llamas_options():
    settings, weights = synthetic_weights(n_layers=3)
    tensors, published = smollm3(settings, weights, no_rope_layers=[1, 1, 1])
    assert "unturned" not in conversion(tensors, published).options


@pytest.mark.parametrize("change, reason", [
    (dict(no_rope_layers=[1, 0, 1]), "does not name every layer"),
    (dict(no_rope_layers="1101"), "does not name every layer"),
    (dict(no_rope_layers=None, no_rope_layer_interval=0), "no usable no_rope_layer_interval"),
    (dict(no_rope_layers=None, no_rope_layer_interval="4"), "no usable no_rope_layer_interval"),
    (dict(use_sliding_window=True), "sliding window"),
    (dict(mlp_bias=True), "biases"),
])
def test_what_the_engine_cannot_read_of_a_smollm3_is_refused(change, reason):
    settings, weights = synthetic_weights(n_layers=4)
    _, published = smollm3(settings, weights, no_rope_layers=[1, 1, 1, 0])
    check_config(normalize(published))
    with pytest.raises(ValueError, match=reason):
        check_config(normalize({**published, **change}))


def test_the_engine_refuses_layers_it_has_not_and_other_architectures():
    settings, weights = synthetic_weights(n_layers=4)
    tensors, published = smollm3(settings, weights, no_rope_layers=[1, 1, 1, 0])
    checkpoint, tokenizer = bytes(conversion(tensors, published).checkpoint), pack_tokenizer(tiny_vocab(settings["vocab_size"]))
    assert Llama(checkpoint, tokenizer, unturned=[3, 3, 0]).unturned == (0, 3)
    assert Llama(checkpoint, tokenizer).unturned == ()
    for layers in ([4], [-1], [0, 9]):
        with pytest.raises(ValueError, match="layers RoPE leaves alone"):
            Llama(checkpoint, tokenizer, unturned=layers)
    # (T255 review: no other architecture takes the layers, which it would otherwise read without turning anything)
    for arch in ("gpt2", "neox", "qwen35", "lfm2"):
        with pytest.raises(ValueError, match="layers RoPE leaves alone"):
            Llama(checkpoint, tokenizer, arch=arch, unturned=[3])


def test_the_plan_forward_js_gets_names_the_layers_and_a_llamas_names_none():
    """T255 review: forward.js (and through it the GPU) learns the layers from the plan Python hands over; a plan without
    them is a Llama's, which turns every layer: no error, worse text. Only forward-check (CI's full set) saw it before."""
    settings, weights = synthetic_weights(n_layers=8)
    for said, want in ((dict(no_rope_layer_interval=4), [3, 7]), (dict(no_rope_layers=[1] * 8), [])):
        tensors, published = smollm3(settings, weights, **said)
        made = conversion(tensors, published)
        outside = Outside(bytes(made.checkpoint))
        Llama(None, made.tokenizer, external=outside, **{key: value for key, value in made.options.items() if key != "template"})
        assert outside.plan["unturned"] == want


# ---- a SmolLM3's GGUF: llama.cpp's converter is its Llama's (q and k turned), and its model leaves every fourth
# layer alone whatever the file says (it says nothing)
def smollm3_gguf(n_layers=8, said=None, **keys):
    settings, weights = synthetic_weights(n_layers=n_layers, n_kv_heads=2)
    tensors, published = smollm3(settings, weights, **(dict(no_rope_layer_interval=4) if said is None else said))
    published["rms_norm_eps"] = 1e-5
    file, same = gguf_file(tensors, published, settings["vocab_size"], "smollm3",
                           more=[("smollm3.attention.layer_norm_rms_epsilon", 6, 1e-5)], **keys)
    return settings, published, file, same


@pytest.mark.parametrize("dtype", ["int8", "float32", "int6"])
def test_a_smollm3_gguf_with_the_originals_files_is_the_safetensors_conversion(dtype):
    settings, published, file, same = smollm3_gguf()
    got = with_original(file, published, unigram(settings["vocab_size"]), "tokenizer.json", dtype)
    expected = conversion(same, published, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer) and got.options == expected.options
    assert got.options["unturned"] == [3, 7]


def test_a_smollm3_gguf_alone_leaves_every_fourth_layer_alone():
    settings, published, file, same = smollm3_gguf()
    got = fed(file, "int8")
    assert got.options["arch"] == "llama" and got.options["unturned"] == [3, 7]
    assert bytes(got.checkpoint) == bytes(conversion(same, published, "int8").checkpoint)


@pytest.mark.parametrize("said", [dict(no_rope_layers=[1, 1, 0, 1, 1, 1, 1, 0]), dict(no_rope_layer_interval=2),
                                  dict(no_rope_layers=[1] * 8)])
def test_a_smollm3_gguf_of_an_original_with_other_layers_is_refused(said):
    """llama.cpp would run such a GGUF with every fourth layer left alone, which is not the model config.json describes."""
    settings, published, file, same = smollm3_gguf(said=said)
    with pytest.raises(ValueError, match="layers without RoPE"):
        llama2_convert.gguf_weights(file, json.dumps(published))


def test_a_llama_gguf_is_no_smollm3s_weights():
    settings, published, file, same = smollm3_gguf()
    with pytest.raises(ValueError, match="architecture"):
        llama2_convert.gguf_weights(file, json.dumps({**published, "model_type": "llama"}))
