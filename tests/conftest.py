# Shared helpers for the engine tests: they run on native Python + NumPy, no Pyodide and no torch.
# Synthetic checkpoints and tokenizers are built here, so that no binary has to live in the repository.
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "public"))
sys.path.insert(0, str(ROOT))  # quantize.py

import llama2_numpy  # noqa: E402


def model_file(name):
    """A file that `make models` produces, or a skip when it is not there."""
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is missing: run `make models` first")
    return path


def checkpoint_vocab_size(name):
    """The vocabulary size in a checkpoint header, read without loading the weights."""
    with open(model_file(name), "rb") as f:
        return abs(struct.unpack("<7i", f.read(28))[5])


# ------------------------------------------------------------------------------------- synthetic checkpoints

def rope_tables(seq_len, head_size, rope_theta=10000.0):
    angles = np.arange(seq_len)[:, None] / rope_theta ** (np.arange(0, head_size, 2) / head_size)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def synthetic_weights(dim=32, hidden_dim=64, n_layers=2, n_heads=4, n_kv_heads=4,
                      vocab_size=320, seq_len=24, shared=True, seed=0):
    """Random weights for a tiny model, plus its configuration. Small enough for a naive reference."""
    rng = np.random.default_rng(seed)
    head_size = dim // n_heads
    kv_dim = n_kv_heads * head_size
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    cos, sin = rope_tables(seq_len, head_size)
    weights = {
        "token_embedding_table": normal(vocab_size, dim),
        "rms_att_weight": (1.0 + normal(n_layers, dim) * 0.1).astype(np.float32),
        "wq": normal(n_layers, dim, dim), "wk": normal(n_layers, kv_dim, dim),
        "wv": normal(n_layers, kv_dim, dim), "wo": normal(n_layers, dim, dim),
        "rms_ffn_weight": (1.0 + normal(n_layers, dim) * 0.1).astype(np.float32),
        "w1": normal(n_layers, hidden_dim, dim), "w2": normal(n_layers, dim, hidden_dim),
        "w3": normal(n_layers, hidden_dim, dim),
        "rms_final_weight": (1.0 + normal(dim) * 0.1).astype(np.float32),
        "freq_cis_real": cos, "freq_cis_imag": sin,
    }
    weights["wcls"] = weights["token_embedding_table"] if shared else normal(vocab_size, dim)
    config = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
                  n_kv_heads=n_kv_heads, vocab_size=vocab_size, seq_len=seq_len,
                  head_size=head_size, kv_dim=kv_dim, shared=shared)
    return config, weights


# the tensor order of the llama2.c "legacy" format, as quantize.py and llama2_numpy.py read it
TENSOR_ORDER = ["token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo",
                "rms_ffn_weight", "w1", "w2", "w3", "rms_final_weight", "freq_cis_real", "freq_cis_imag"]


def pack_checkpoint(config, weights):
    """The float32 checkpoint: a 7 int header (negative vocab size means an unshared classifier) then tensors."""
    vocab_size = config["vocab_size"] if config["shared"] else -config["vocab_size"]
    out = [struct.pack("<7i", config["dim"], config["hidden_dim"], config["n_layers"], config["n_heads"],
                       config["n_kv_heads"], vocab_size, config["seq_len"])]
    names = TENSOR_ORDER + ([] if config["shared"] else ["wcls"])
    out += [np.ascontiguousarray(weights[name], dtype=np.float32).tobytes() for name in names]
    return b"".join(out)


# ------------------------------------------------------------------------------------- synthetic tokenizer

def pack_tokenizer(pieces):
    """llama2.c's tokenizer.bin: max piece length, then (score, length, bytes) per piece."""
    out = [struct.pack("<i", max(len(text) for _, text in pieces))]
    out += [struct.pack("<fi", score, len(text)) + text for score, text in pieces]
    return b"".join(out)


def tiny_vocab(vocab_size=320):
    """A vocabulary of that many pieces: <unk>, byte fallbacks, then words and characters.

    Byte and control pieces get the unmatchable score convert_hf.py writes, so they never match text directly.
    """
    pieces = [(-1e9, b"<unk>"), (-1e9, b"<s>"), (-1e9, b"</s>")]
    pieces += [(-1e9, b"<0x%02X>" % byte) for byte in range(256)]
    words = [" ", " the", " cat", " s", "a", "t", "o", "n", "e", "h", "c", " a", " o",
             "流", "行", "、", " 流行り", "Ａ", "!", "?", " \n", "\n", " hello", " world",
             " he", "l", "w", "r", "d", " t", "i", "g", "猫", " 猫", "り"]
    for i, word in enumerate(words):
        pieces.append((-float(i) * 0.5, word.encode("utf-8")))
    assert len(pieces) <= vocab_size, "the vocabulary does not fit"
    while len(pieces) < vocab_size:  # padding rows, never matchable
        pieces.append((-1e9, b"<pad%d>" % len(pieces)))
    return pieces[:vocab_size]


def tiny_tokenizer(kind="bpe", nfkc=False):
    pieces = tiny_vocab(320)
    return llama2_numpy.Tokenizer(pack_tokenizer(pieces), len(pieces), kind=kind, nfkc=nfkc)


def detokenize(tokenizer, tokens, bos=llama2_numpy.BOS):
    """Join the pieces the way generate() does: the first one loses the dummy prefix."""
    out, previous = [], bos
    for token in tokens:
        out.append(tokenizer.decode(previous, token, bos))
        previous = token
    return b"".join(out).decode("utf-8", "replace")


# ------------------------------------------------------------------------------------- naive reference

def naive_logits(config, weights, tokens):
    """llama2.c written out with Python loops: the reference forward() is compared against."""
    dim, n_layers = config["dim"], config["n_layers"]
    n_heads, n_kv_heads, head_size = config["n_heads"], config["n_kv_heads"], config["head_size"]
    kv_mul = n_heads // n_kv_heads
    cos, sin = weights["freq_cis_real"], weights["freq_cis_imag"]

    def rmsnorm(vector, weight):
        return weight * vector / math.sqrt(sum(float(v) * float(v) for v in vector) / dim + 1e-5)

    def rope(vector, pos, heads):
        out = vector.copy()
        for h in range(heads):
            for i in range(head_size // 2):
                a, b = h * head_size + 2 * i, h * head_size + 2 * i + 1
                out[a] = vector[a] * cos[pos, i] - vector[b] * sin[pos, i]
                out[b] = vector[a] * sin[pos, i] + vector[b] * cos[pos, i]
        return out

    x = [weights["token_embedding_table"][token].astype(np.float64) for token in tokens]
    for l in range(n_layers):
        queries, keys, values = [], [], []
        for pos in range(len(tokens)):
            xb = rmsnorm(x[pos], weights["rms_att_weight"][l])
            queries.append(rope(weights["wq"][l] @ xb, pos, n_heads))
            keys.append(rope(weights["wk"][l] @ xb, pos, n_kv_heads))
            values.append(weights["wv"][l] @ xb)
        for pos in range(len(tokens)):
            attended = np.zeros(dim, dtype=np.float64)
            for h in range(n_heads):
                kv = h // kv_mul
                q = queries[pos][h * head_size:(h + 1) * head_size]
                scores = np.array([float(q @ keys[t][kv * head_size:(kv + 1) * head_size]) / math.sqrt(head_size)
                                   for t in range(pos + 1)])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                for t in range(pos + 1):
                    attended[h * head_size:(h + 1) * head_size] += \
                        scores[t] * values[t][kv * head_size:(kv + 1) * head_size]
            x[pos] = x[pos] + weights["wo"][l] @ attended
            xb = rmsnorm(x[pos], weights["rms_ffn_weight"][l])
            h1 = weights["w1"][l] @ xb
            h1 = h1 / (1.0 + np.exp(-h1)) * (weights["w3"][l] @ xb)
            x[pos] = x[pos] + weights["w2"][l] @ h1
    return [weights["wcls"] @ rmsnorm(vector, weights["rms_final_weight"]) for vector in x]
