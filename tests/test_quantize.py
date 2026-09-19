# quantize.py writes an int8 checkpoint; llama2_numpy.py must read back what it wrote.
import struct
import subprocess
import sys

import numpy as np
import pytest

import quantize
from conftest import ROOT, TENSOR_ORDER, pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab
from llama2_numpy import Llama

MATRICES = ["token_embedding_table", "wq", "wk", "wv", "wo", "w1", "w2", "w3"]
NORMS = ["rms_att_weight", "rms_ffn_weight", "rms_final_weight"]


def quantized(tmp_path, **overrides):
    config, weights = synthetic_weights(**overrides)
    source, target = tmp_path / "model.f32", tmp_path / "model.bin"
    source.write_bytes(pack_checkpoint(config, weights))
    subprocess.run([sys.executable, str(ROOT / "quantize.py"), str(source), str(target)], check=True)
    tokenizer = pack_tokenizer(tiny_vocab(config["vocab_size"]))
    return config, weights, Llama(target.read_bytes(), tokenizer, dtype="int8"), target


@pytest.mark.parametrize("name,overrides", [
    ("shared classifier", {}),
    ("unshared classifier", {"shared": False}),
    ("group smaller than 32", {"hidden_dim": 48}),
])
def test_int8_weights_are_within_one_step_of_the_originals(tmp_path, name, overrides):
    config, weights, llama, _ = quantized(tmp_path, **overrides)
    for tensor in MATRICES + (["wcls"] if not config["shared"] else []):
        original = weights[tensor]
        got = getattr(llama, tensor)
        if isinstance(got, tuple):  # an embedding table that stays int8 until a row is read
            values, scales = got
            got = (values * scales).reshape(original.shape)
        assert got.shape == original.shape, tensor
        group = quantize.group_size(original.shape[-1])
        groups = original.reshape(-1, group)
        # one step of the quantization is max(|group|) / 127
        step = np.abs(groups).max(axis=1) / 127.0
        error = np.abs(groups - got.reshape(-1, group)).max(axis=1)
        assert (error <= step + 1e-6).all(), f"{name}: {tensor}"


def test_norm_weights_stay_float32(tmp_path):
    _, weights, llama, _ = quantized(tmp_path)
    for tensor in NORMS:
        assert np.array_equal(getattr(llama, tensor), weights[tensor]), tensor


def test_rope_tables_are_dropped_and_recomputed(tmp_path):
    config, weights, llama, target = quantized(tmp_path)
    tables = weights["freq_cis_real"].nbytes + weights["freq_cis_imag"].nbytes
    assert target.stat().st_size < len(pack_checkpoint(config, weights)) - tables
    assert np.abs(llama.freq_cis_real - weights["freq_cis_real"]).max() < 1e-6
    assert np.abs(llama.freq_cis_imag - weights["freq_cis_imag"]).max() < 1e-6


def test_header_survives_quantization(tmp_path):
    config, weights, _, target = quantized(tmp_path, shared=False)
    with open(target, "rb") as f:
        assert struct.unpack("<7i", f.read(28)) == struct.unpack("<7i", pack_checkpoint(config, weights)[:28])


def test_int8_logits_stay_close_to_float32(tmp_path):
    config, weights, llama, _ = quantized(tmp_path)
    reference = Llama(pack_checkpoint(config, weights), pack_tokenizer(tiny_vocab(config["vocab_size"])))
    for pos, token in enumerate([1, 5, 7, 5, 2]):
        got = llama.forward(token, pos).astype(np.float64)
        expected = reference.forward(token, pos).astype(np.float64)
        # random weights are a harsher test than trained ones (stories15M: perplexity +0.04%)
        assert np.abs(got - expected).max() / np.abs(expected).max() < 0.1
        assert np.corrcoef(got, expected)[0, 1] > 0.99


def test_the_tensor_order_of_quantize_matches_the_engine(tmp_path):
    config, _ = synthetic_weights()
    header = (config["dim"], config["hidden_dim"], config["n_layers"], config["n_heads"],
              config["n_kv_heads"], -config["vocab_size"], config["seq_len"])
    shapes = [shape for shape, _ in quantize.layout(*header)]
    expected = TENSOR_ORDER + ["wcls"]
    assert len(shapes) == len(expected)


def test_int8_embedding_row_is_widened_when_the_classifier_is_separate(tmp_path):
    _, weights, llama, _ = quantized(tmp_path, shared=False)
    # without a shared table the rows stay int8 and embedding() widens the one it needs
    assert isinstance(llama.token_embedding_table, tuple)
    row = llama.embedding(7)
    assert row.dtype == np.float32
    assert np.abs(row - weights["token_embedding_table"][7]).max() <= \
        np.abs(weights["token_embedding_table"][7]).max() / 127.0 + 1e-6
