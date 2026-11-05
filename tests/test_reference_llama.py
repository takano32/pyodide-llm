"""tests/reference_llama.py's helpers (the review of T253), which need no torch: the engine's keys and values in
transformers' order, the gap of the states of every layer, and the weak faults put into the engine's weights, each of which
the check must see, with the weights put back after. The comparison with transformers itself runs in CI (the file's head)."""
import math

import numpy as np
import pytest
from conftest import pack_tokenizer, synthetic_weights, tiny_vocab
from test_convert import converted, reader, safetensors_file
from test_granite import granite

import reference_llama as R
from llama2_convert import Safetensors, query_scale
from llama2_numpy import Llama

TOKENS = [1, 5, 7, 9, 5, 11, 3, 3, 8, 2, 14, 15, 1, 6, 22, 30]


def engine_of(**config):
    settings, weights = synthetic_weights(**config)
    tensors, published = granite(settings, weights, 0.3)
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), published, "float32")
    return Llama(checkpoint, pack_tokenizer(tiny_vocab(settings["vocab_size"]))), tensors, published


def rms(x, weight):
    return weight * x / math.sqrt((x * x).mean() + 1e-5)


@pytest.mark.parametrize("config", [dict(dim=64, n_kv_heads=2, shared=False), dict(dim=32, n_kv_heads=4)])
def test_the_engines_first_layer_comes_back_in_transformers_order(config):
    """Keys after RoPE as transformers holds them (rotate_half: a head's first half against its second) from the Hugging
    Face weights in float64, against engine_cache(): the engine turns adjacent pairs of the rows the converter interleaved.
    A layer's values are the same in both orders."""
    llama, tensors, _ = engine_of(**config)
    for pos, token in enumerate(TOKENS):
        llama.forward(token, pos)
    keys, values = R.engine_cache(llama, len(TOKENS))[0]
    head = llama.head_size
    inverse = 1.0 / 10000.0 ** (np.arange(0, head, 2) / head)
    embedded = np.asarray(tensors["model.embed_tokens.weight"], dtype=np.float64)
    norm = np.asarray(tensors["model.layers.0.input_layernorm.weight"], dtype=np.float64)
    wk = np.asarray(tensors["model.layers.0.self_attn.k_proj.weight"], dtype=np.float64)
    wv = np.asarray(tensors["model.layers.0.self_attn.v_proj.weight"], dtype=np.float64)
    expected_keys, expected_values = [], []
    for pos, token in enumerate(TOKENS):
        x = rms(embedded[token], norm)
        k, v = (wk @ x).reshape(llama.n_kv_heads, head), (wv @ x).reshape(llama.n_kv_heads, head)
        cos, sin = np.cos(pos * inverse), np.sin(pos * inverse)
        first, second = k[:, :head // 2], k[:, head // 2:]
        expected_keys.append(np.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1))
        expected_values.append(v)
    assert np.allclose(keys, np.stack(expected_keys, axis=1), atol=1e-4)
    assert np.allclose(values, np.stack(expected_values, axis=1), atol=1e-4)
    # and in the order the engine holds them they are not those
    assert not np.allclose(llama.key_cache[0, :, :len(TOKENS)], np.stack(expected_keys, axis=1), atol=1e-2)


def test_state_gap_is_relative_to_each_layers_largest_and_nan_is_as_far_as_can_be():
    same = [(np.ones((2, 3, 4), np.float32) * 5, np.ones((2, 3, 4), np.float32))] * 3
    off = [(k * 1.01 if layer == 1 else k, v) for layer, (k, v) in enumerate(same)]
    gaps, top = R.state_gap(off, same)
    assert gaps["keys"][1] == 1 and gaps["keys"][0] == pytest.approx(0.01, rel=1e-4) and gaps["values"][0] == 0
    assert top == {"keys": 5.0, "values": 1.0}
    broken = [(k, np.full_like(v, np.nan) if layer == 2 else v) for layer, (k, v) in enumerate(same)]
    gaps, _ = R.state_gap(broken, same)
    assert gaps["values"] == (float("inf"), 2)
    assert R.state_line({"keys": 1e-6, "values": 2e-6}) == 1e-4
    assert R.state_line({"keys": 1e-5, "values": 2e-5}) == pytest.approx(2e-4)


def test_every_weak_fault_the_check_must_see_is_seen_and_the_weights_are_put_back(capsys):
    llama, _, published = engine_of(dim=64, n_layers=4, n_kv_heads=2, shared=False)
    whole = np.array([llama.forward(token, pos).copy() for pos, token in enumerate(TOKENS)])
    states = R.engine_cache(llama, len(TOKENS))
    before = (llama.wq, llama.wk)
    missed = R.weak_errors("t", llama, TOKENS, whole, states, 1e-3, 1e-4, query_scale(published))
    assert missed == []
    assert llama.wq is before[0] and llama.wk is before[1], "the matrices themselves stand in again"
    assert np.array_equal(np.array([llama.forward(token, pos) for pos, token in enumerate(TOKENS)]), whole)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7 and all("seen by" in line for line in lines)
    # the right engine against itself is seen by neither, and the faults that are not weak are
    assert "q of every layer 0.1% off" in lines[0] and "seen by neither" not in lines[0]
    assert any("left as it was" in line and "seen by neither" not in line for line in lines)


def test_a_fault_the_check_cannot_see_is_named_where_it_must(capsys):
    """The line that lets everything by (a logits line of 1e3 and a states line of 1e3) sees none of the faults it must."""
    llama, _, published = engine_of(dim=64, n_layers=4, n_kv_heads=2, shared=False)
    whole = np.array([llama.forward(token, pos).copy() for pos, token in enumerate(TOKENS)])
    states = R.engine_cache(llama, len(TOKENS))
    missed = R.weak_errors("t", llama, TOKENS, whole, states, 1e3, 1e3, query_scale(published))
    assert len(missed) == 4 and any("FAILED" in line for line in capsys.readouterr().out.splitlines())
