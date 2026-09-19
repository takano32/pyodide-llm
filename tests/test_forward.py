# forward() on synthetic checkpoints, against a naive loop implementation of llama2.c.
import numpy as np
import pytest

from conftest import naive_logits, pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab
import llama2_numpy
from llama2_numpy import Llama

TOKENS = [1, 5, 7, 5, 2, 9, 9, 3]


def build(**overrides):
    config, weights = synthetic_weights(**overrides)
    tokenizer = pack_tokenizer(tiny_vocab(config["vocab_size"]))
    llama = Llama(pack_checkpoint(config, weights), tokenizer)
    return config, weights, llama


def relative_error(got, expected):
    return float(np.abs(np.asarray(got, dtype=np.float64) - expected).max() / np.abs(expected).max())


@pytest.mark.parametrize("name,overrides", [
    ("multi head", {}),
    ("grouped-query", {"n_kv_heads": 2}),
    ("multi query", {"n_kv_heads": 1}),
    ("unshared classifier", {"shared": False}),
    ("odd sizes", {"dim": 32, "hidden_dim": 48, "n_heads": 8, "n_kv_heads": 2}),
])
def test_forward_matches_a_naive_implementation(name, overrides):
    config, weights, llama = build(**overrides)
    expected = naive_logits(config, weights, TOKENS)
    for pos, token in enumerate(TOKENS):
        logits = llama.forward(token, pos)
        assert relative_error(logits, expected[pos]) < 1e-4, f"{name} at position {pos}"


def test_header_is_read_as_llama2_c_writes_it():
    config, _, llama = build(n_kv_heads=2, shared=False)
    assert (llama.dim, llama.hidden_dim, llama.n_layers) == (config["dim"], config["hidden_dim"], config["n_layers"])
    assert (llama.n_heads, llama.n_kv_heads, llama.seq_len) == (config["n_heads"], config["n_kv_heads"], config["seq_len"])
    assert llama.vocab_size == config["vocab_size"] and llama.head_size == config["head_size"]
    assert llama.wcls is not llama.token_embedding_table  # a negative vocab size means an unshared classifier


def test_shared_weights_are_not_copied():
    _, _, llama = build()
    assert llama.wcls is llama.token_embedding_table


def test_need_logits_false_still_fills_the_cache():
    config, weights, llama = build()
    for pos, token in enumerate(TOKENS[:-1]):
        assert llama.forward(token, pos, need_logits=False) is None
    logits = llama.forward(TOKENS[-1], len(TOKENS) - 1)
    expected = naive_logits(config, weights, TOKENS)[-1]
    assert relative_error(logits, expected) < 1e-4


def test_float16_checkpoint_is_close_to_float32():
    config, weights = synthetic_weights()
    tokenizer = pack_tokenizer(tiny_vocab(config["vocab_size"]))
    data = pack_checkpoint(config, weights)
    half = np.frombuffer(data[28:], dtype=np.float32).astype(np.float16).tobytes()
    reference = Llama(data, tokenizer)
    llama = Llama(data[:28] + half, tokenizer, dtype="float16")
    for pos, token in enumerate(TOKENS):
        assert relative_error(llama.forward(token, pos), reference.forward(token, pos)) < 1e-2


def test_the_cache_grows_without_changing_a_logit(monkeypatch):
    config, weights = synthetic_weights(n_kv_heads=2, seq_len=24)
    checkpoint, tokenizer = pack_checkpoint(config, weights), pack_tokenizer(tiny_vocab(config["vocab_size"]))
    roomy = Llama(checkpoint, tokenizer)
    monkeypatch.setattr(llama2_numpy, "KV_START", 3)
    tight = Llama(checkpoint, tokenizer)
    assert tight.key_cache.shape[2] == 3
    for pos in range(config["seq_len"]):
        assert np.array_equal(roomy.forward(5 + pos, pos), tight.forward(5 + pos, pos)), pos
    assert tight.key_cache.shape[2] == config["seq_len"]  # 3 -> 6 -> 12 -> 24, never more than the context
