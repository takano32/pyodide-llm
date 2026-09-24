"""Qwen2 is a Llama with a bias on q, k and v (T64): the converter writes three vectors per layer, and the engine
adds them after those projections. The file keeps the 7 int header, so the caller passes bias=True."""
import struct

import numpy as np
import pytest
from conftest import naive_logits, pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab
from test_convert import converted, hugging_face, reader, safetensors_file, streamed

from llama2_convert import Safetensors, checkpoint_header, checkpoint_size, has_bias, layout
from llama2_numpy import Llama, checkpoint_dtype

CONFIGS = [dict(n_kv_heads=4), dict(n_kv_heads=2), dict(n_kv_heads=1, shared=False)]


def qwen2(config, weights, shared, seed=7):
    """What Hugging Face would publish for a Qwen2 with these weights: the biases too, permuted like wq and wk."""
    tensors, published = hugging_face(config, weights, shared)
    head_size = config["dim"] // config["n_heads"]
    rng = np.random.default_rng(seed)
    permute = lambda v, heads: v.reshape(heads, head_size // 2, 2).transpose(0, 2, 1).reshape(-1)
    for name, ours, heads in (("q_proj", "bq", config["n_heads"]), ("k_proj", "bk", config["n_kv_heads"]),
                              ("v_proj", "bv", None)):
        rows = config["dim"] if ours == "bq" else config["n_kv_heads"] * head_size
        weights[ours] = (rng.standard_normal((config["n_layers"], rows)) * 0.3).astype(np.float32)
        for layer in range(config["n_layers"]):
            vector = weights[ours][layer]
            tensors[f"model.layers.{layer}.self_attn.{name}.bias"] = \
                np.ascontiguousarray(permute(vector, heads) if heads else vector)
    return tensors, {**published, "model_type": "qwen2"}


@pytest.mark.parametrize("config", CONFIGS)
def test_a_qwen2_converts_and_runs_like_the_reference(config):
    shared = config.pop("shared", True)
    settings, weights = synthetic_weights(shared=shared, **config)
    tensors, published = qwen2(settings, weights, shared)
    file = safetensors_file(tensors)
    source = Safetensors(reader(file))
    assert has_bias(source)
    checkpoint = converted(source, published, "float32")
    # three vectors per layer more than the same model without them, and the header still says nothing
    plain = len(pack_checkpoint(settings, weights))
    kv_dim = settings["kv_dim"]
    assert len(checkpoint) == plain + 4 * settings["n_layers"] * (settings["dim"] + 2 * kv_dim)
    assert struct.unpack_from("<7i", checkpoint, 0) == struct.unpack_from("<7i", pack_checkpoint(settings, weights), 0)
    assert checkpoint_dtype(struct.unpack_from("<7i", checkpoint, 0), len(checkpoint), bias=True) == "float32"

    llama = Llama(checkpoint, pack_tokenizer(tiny_vocab(settings["vocab_size"])), bias=True)
    tokens = [1, 5, 7, 9]
    want = naive_logits(settings, weights, tokens)
    for pos, token in enumerate(tokens):
        got = llama.forward(token, pos)
        assert np.allclose(got, want[pos], rtol=1e-4, atol=1e-4)


def test_the_file_in_its_own_order_gives_the_same_checkpoint():
    """The biases are permuted like wq and wk, and a vector needs its whole self before it can be permuted."""
    settings, weights = synthetic_weights()
    tensors, published = qwen2(settings, weights, True)
    file = safetensors_file(tensors)
    expected = converted(Safetensors(reader(file)), published, "float32")
    got, progress = streamed(file, published, "float32", 4096)
    assert got == expected and progress[-1][0] == progress[-1][1]


def test_the_conversion_tells_the_engine_about_the_biases():
    """The file cannot say it has biases: the options must (the lesson of T72, applied here in T77)."""
    import json

    import llama2_convert
    settings, weights = synthetic_weights()
    tensors, published = qwen2(settings, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(settings["vocab_size"])]}}).encode()
    conversion = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary,
                                           "tokenizer.json", dtype="float32", max_seq_len=settings["seq_len"])
    assert conversion.options["bias"] is True and conversion.options["arch"] == "llama"


def test_without_the_flag_the_engine_reads_the_file_it_always_read():
    settings, weights = synthetic_weights()
    checkpoint = pack_checkpoint(settings, weights)
    assert [shape for shape, _ in layout(*struct.unpack_from("<7i", checkpoint, 0))] \
        == [shape for shape, _ in layout(*struct.unpack_from("<7i", checkpoint, 0), bias=False)]
    assert checkpoint_size(struct.unpack_from("<7i", checkpoint, 0), "float32") == len(checkpoint)
