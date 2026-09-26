# tests/gguf_check.py tensors, held to what T136's stage 2 asks of it (the review of 2026-09-26): the embedding and
# the classifier row by row, Llama 3's rope_freqs.weight against the engine's table, the vocabulary of a
# sentencepiece original, the config after normalize(), and the vocabulary shown but not counted when the page takes
# it from the original. Each test fails when its check is taken out.
import json
import math

import numpy as np
import pytest
from conftest import synthetic_weights
from make_hf_fixture import field
from test_convert import hugging_face, safetensors_file
from test_gguf import gguf_file

import gguf_check
from llama2_numpy import rope_frequencies

EPS = ("llama.attention.layer_norm_rms_epsilon", 6, 1e-5)
LLAMA3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 64}


def model(tmp_path, vocab_size=320, dim=32, theta=10000.0, extra=None, change=None, original_change=None, **config):
    """A GGUF and the directory of its original (config.json and model.safetensors); change(tensors) alters what
    the GGUF is written from, as a GGUF of other weights would be, and original_change(tensors) the original."""
    shape, weights = synthetic_weights(dim=dim, hidden_dim=2 * dim, vocab_size=vocab_size, seq_len=128)
    tensors, published = hugging_face(shape, weights, True)
    published = {**published, "rms_norm_eps": 1e-5, "rope_theta": theta, **config}
    written = {name: tensor.copy() for name, tensor in tensors.items()}
    if change:
        change(written)
    file, same = gguf_file(written, published, vocab_size, theta=theta, more=[EPS], extra=extra)
    # the original holds the values the GGUF stands for (Q8_0 rounds), except where change() made them differ
    original = {name: (same[name] if change is None or np.array_equal(written[name], tensors[name]) else tensors[name])
                for name in tensors}
    if original_change:
        original_change(original)
    (tmp_path / "model.gguf").write_bytes(file)
    (tmp_path / "model.safetensors").write_bytes(safetensors_file(original))
    (tmp_path / "config.json").write_text(json.dumps(published))
    return tmp_path / "model.gguf", tmp_path


def summary(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_the_same_weights_pass(tmp_path, capsys):
    assert gguf_check.check_tensors(*model(tmp_path))
    assert summary(capsys)["mismatches"] == 0


@pytest.mark.parametrize("block", [gguf_check.BLOCK, 4096])  # the whole tensor at once, and a block of rows at a time
def test_eight_swapped_rows_of_the_embedding_are_caught(tmp_path, capsys, monkeypatch, block):
    """Qwen2.5 0.5B's embedding with 8 rows swapped was 0.0122 off as a whole, under the line of 0.02 (the review of
    T136). Here 100000 rows of 32: the whole is about as far off, and every one of the 8 rows is past 0.05."""
    monkeypatch.setattr(gguf_check, "BLOCK", block)

    def swap(tensors):
        embedding = tensors["model.embed_tokens.weight"]
        for a, b in ((10, 20), (300, 4000), (50000, 60000), (99990, 7)):
            embedding[[a, b]] = embedding[[b, a]]

    assert not gguf_check.check_tensors(*model(tmp_path, vocab_size=100000, change=swap))
    result = summary(capsys)
    assert result["worst"] < 0.02, "the error of the whole tensor alone would have let this through"
    assert result["rows"]["token_embd.weight"] == 8 and result["mismatches"] == 8


def test_a_row_of_nearly_nothing_is_not_an_error(tmp_path, capsys):
    """Qwen2.5 7B's unused rows are 1.2e-37 at most, and a Q8_0 writes them as 0: no difference that matters."""
    def tiny(tensors):
        tensors["model.embed_tokens.weight"][5] = 1e-37

    assert gguf_check.check_tensors(*model(tmp_path, change=tiny, original_change=tiny))
    assert summary(capsys)["rows"]["token_embd.weight"] == 0


def test_rope_freqs_is_compared_with_the_engines_table(tmp_path, capsys):
    head = 32 // 4
    factors = rope_frequencies(head, 500000.0) / rope_frequencies(head, 500000.0, LLAMA3)
    assert not np.allclose(factors, 1), "the scaling must change some pairs for the test to mean anything"
    assert gguf_check.check_tensors(*model(tmp_path, theta=500000.0, extra={"rope_freqs.weight": factors},
                                           rope_scaling=LLAMA3))
    result = summary(capsys)
    assert result["rope_freqs_diff"] < 1e-6 and result["mismatches"] == 0

    # a table of another scaling (none at all) is a mismatch, not a crash
    other = tmp_path / "other"
    other.mkdir()
    assert not gguf_check.check_tensors(*model(other, theta=500000.0, extra={"rope_freqs.weight": np.ones(head // 2)},
                                               rope_scaling=LLAMA3))
    result = summary(capsys)
    assert result["rope_freqs_diff"] > 1e-3 and result["mismatches"] == 1


def test_the_config_is_read_after_normalize(tmp_path, capsys):
    """transformers 5 writes rope_theta inside rope_parameters (CAT-Translate 1.4b): read as it is, the theta would
    be the default 10000 and differ from the GGUF's."""
    path, directory = model(tmp_path, theta=500000.0)
    config = json.loads((directory / "config.json").read_text())
    del config["rope_theta"]
    config["rope_parameters"] = {"rope_theta": 500000.0, "rope_type": "default"}
    (directory / "config.json").write_text(json.dumps(config))
    assert gguf_check.check_tensors(path, directory)
    assert summary(capsys)["mismatches"] == 0


def sentencepiece(pieces):
    return b"".join(field(1, field(1, text.encode()) + field(2, -1.0) + field(3, 1)) for text in pieces)


@pytest.mark.parametrize("original_vocabulary", [False, True])
def test_a_sentencepiece_vocabulary_is_compared(tmp_path, capsys, original_vocabulary):
    path, directory = model(tmp_path)
    pieces = [f"w{i}" for i in range(320)]
    (directory / "tokenizer.model").write_bytes(sentencepiece(pieces))
    assert gguf_check.check_tensors(path, directory, original_vocabulary)
    assert summary(capsys)["vocab_diffs"] == {"tokenizer.model": 0}

    pieces[3] = "▁w3"  # the id of one piece written differently (llm-jp-4's GGUF: 13 of them)
    (directory / "tokenizer.model").write_bytes(sentencepiece(pieces))
    assert gguf_check.check_tensors(path, directory, original_vocabulary) is original_vocabulary
    result = summary(capsys)
    assert result["vocab_diffs"] == {"tokenizer.model": 1}
    assert result["mismatches"] == (0 if original_vocabulary else 1)


def test_a_unigram_tokenizer_json_is_read_by_id(tmp_path, capsys):
    """llm-jp's tokenizer.json is a Unigram: its vocab is a list of [piece, score], in the order of the ids."""
    path, directory = model(tmp_path)
    vocab = [[f"w{i}", -1.0] for i in range(320)]
    (directory / "tokenizer.json").write_text(json.dumps({"model": {"type": "Unigram", "vocab": vocab}, "added_tokens": []}))
    assert gguf_check.check_tensors(path, directory)
    assert summary(capsys)["vocab_diffs"] == {"tokenizer.json": 0}


def test_names_without_a_hugging_face_counterpart_do_not_raise():
    assert gguf_check.hugging_face_name("rope_freqs.weight") is None
    assert gguf_check.hugging_face_name("blk.0.attn_q.weight") == "model.layers.0.self_attn.q_proj.weight"
    assert math.isfinite(gguf_check.ROW_LINE)


def test_q_and_k_compared_in_blocks_are_read_turned(tmp_path, capsys, monkeypatch):
    """Llama 3.2 3B's q is 3072 x 3072, past BLOCK: compared a block at a time, it was read in Hugging Face's order
    and a GGUF turned the way llama.cpp turns it was 1.4 off (the first run of stage 2)."""
    monkeypatch.setattr(gguf_check, "BLOCK", 256)  # q and k of 32 x 32 go by blocks of 8 rows, one head each
    assert gguf_check.check_tensors(*model(tmp_path))
    result = summary(capsys)
    assert result["orders"] == ["turned (llama2.c order)"] and result["worst"] < 0.02
