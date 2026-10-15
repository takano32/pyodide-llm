# Shared helpers for the engine tests: they run on native Python + NumPy, no Pyodide and no torch.
# Synthetic checkpoints and tokenizers are built here, so that no binary has to live in the repository.
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
from tree import python_folder
sys.path.insert(0, python_folder(ROOT))
sys.path.insert(0, str(ROOT))  # quantize.py

# tmp_path lives under the system's temporary directory, which on the development machine is a tmpfs, that is memory
# (AGENTS.md), and was written there without a word (the owner, 2026-09-26): here it is the repository's .tmp, which
# .gitignore has. pytest reads this where it first makes a tmp_path, after the conftests; a caller may say otherwise.
(ROOT / ".tmp").mkdir(exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(ROOT / ".tmp"))

import llama2_numpy  # noqa: E402
# the made-up models are tests/made_up.py's (T357.2): the tests take them from here as they always did
from made_up import (FOLDED, basis, folded, gemma3_model, lfm2_model, pack_checkpoint, pack_tokenizer, qwen35_model, rope_tables,  # noqa: E402, F401
                     synthetic_weights, tiny_vocab)


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


# ------------------------------------------------------------------------------------------- texts to split
# The texts the pre-tokenizers are checked on (test_bytebpe.py against the real tokenizers, test_llama3.py against
# the patterns). Here and not in test_bytebpe.py, which skips as a whole without tokenizers and took test_llama3.py's
# own tests along when it imported them from there (T144).
CORPUS = (
    "The quick brown fox jumps over the lazy dog. 日本語の文章も混ぜる。"
    "Don't stop; it's theirs, they'll go. I'D LIKE 'IT'.\n"
    "Pyodide は WebAssembly 版の Python で、ブラウザの中で NumPy が動く。"
    "def forward(x, w):\n\treturn w @ x  # matmul\n\n\n"
    "価格は1,234,567円（税込）です。2026-09-21T00:00:00Z\r\n"
    "絵文字 \U0001f600\U0001f389 と外字 \U00029E3D、全角ＡＢＣ１２３、半角ｶﾅ。"
    "https://example.com/a/b?c=1&d=2#frag  'single' \"double\" `tick`\n"
    "   spaces\tand\ttabs\n\n\nnewlines   \nTHE END. the end. The End?!  "
)
# every kind of boundary the patterns care about
TEXTS = [CORPUS, " ", "  ", "\n", "\r\n", " \n ", "0123", " 42 ", "a", " a", "  a", "\ta", "(abc", "、あ",
         "a 1b", " 1,234", "1a2", "v1.2.3", "第1章 2節", "it's a dog's life", "IT'S", "end.  ", "x\n\n\ny",
         # a line of spaces between line breaks, as pasted code has (the review of T106: Qwen's \s*[\r\n]+ takes the
         # whole run up to its last line break, this took it up to the first)
         "a\n  \nb", "\n \n \n", "def f(x):\n    a = 1\n    \n    return a\n", " \t\n \r\n x"]


# ------------------------------------------------------------------------------------------- sentencepiece
def charsmap(mapping):
    """A sentencepiece precompiled_charsmap for {text: its normalized text} (T216): the length of a Darts-clone double
    array, the array, and the normalized texts, each ended by a NUL. Built here the simple way (each node takes the
    first base no other node has and whose children's places are free; a unit: bit 31 a value, bits 0-7 its label,
    bit 8 a leaf below, bits 10 up the offset to its base), which llama2_numpy.Charsmap walks as Darts-clone does."""
    trie, texts = {}, bytearray()
    for key, normal in mapping.items():
        node = trie
        for byte in key.encode("utf-8"):
            node = node.setdefault(byte, {})
        node[None] = len(texts)
        texts += normal.encode("utf-8") + b"\0"
    units, used, bases = {}, {0}, set()

    def place(node, at):
        labels = sorted(label for label in node if label is not None)
        places = labels + ([0] if None in node else [])
        base = 1
        while base in bases or any(base ^ label in used for label in places):
            base += 1
        bases.add(base)
        used.update(base ^ label for label in places)
        units[at] = units.get(at, 0) | (at ^ base) << 10 | (0x100 if None in node else 0)
        if None in node:
            units[base] = node[None] | 1 << 31
        for label in labels:
            units[base ^ label] = label
            place(node[label], base ^ label)

    place(trie, 0)
    array = struct.pack(f"<{max(units) + 1}I", *(units.get(i, 0) for i in range(max(units) + 1)))
    return struct.pack("<I", len(array)) + array + bytes(texts)


# ------------------------------------------------------------------------------------- synthetic checkpoints

class NoWeights:
    """A converter's sink that keeps nothing: for the options and the tokenizer.bin, which it has before the weights."""

    def open(self, *args):
        pass

    def write(self, *args):
        pass


def vocabulary_conversion(tokenizer, name, vocab_size, tokenizer_config=None, **config):
    """llama2_convert.Conversion of a small model with this tokenizer (bytes, as the file name says) and vocabulary,
    and these keys of config.json besides (bos_token_id ...): its options and tokenizer, with no weights fed (T143)."""
    import json
    import llama2_convert
    from test_convert import hugging_face, safetensors_file
    settings, weights = synthetic_weights(vocab_size=vocab_size)
    tensors, published = hugging_face(settings, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    return llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps({**published, **config}), tokenizer,
                                     name, dtype="float32", max_seq_len=settings["seq_len"], start=8 + size,
                                     tokenizer_config=tokenizer_config, sink=NoWeights())


# ------------------------------------------------------------------------------------- synthetic tokenizer

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

    eps = config.get("eps", 1e-5)  # config.json's rms_norm_eps (T124)
    # what the scores are multiplied by: a Granite's attention_multiplier (T253), else one over the root of a head
    score_scale = config.get("attention_multiplier", 1.0 / math.sqrt(head_size))

    def rmsnorm(vector, weight):
        return weight * vector / math.sqrt(sum(float(v) * float(v) for v in vector) / len(vector) + eps)

    def head_norm(vector, name, l):
        # Qwen3 (T124): every head of q and k normalized on its own, with one weight of a head's size
        if name not in weights:
            return vector
        return np.concatenate([rmsnorm(vector[h:h + head_size], weights[name][l]) for h in range(0, len(vector), head_size)])

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
            # Qwen2 adds a bias to q, k and v before the rotation
            bias = lambda name: weights[name][l] if name in weights else 0.0
            # (T255: a SmolLM3 turns the q and k of some layers only)
            turn = (lambda vector, pos, heads: vector) if l in config.get("unturned", ()) else rope
            queries.append(turn(head_norm(weights["wq"][l] @ xb + bias("bq"), "q_norm", l), pos, n_heads))
            keys.append(turn(head_norm(weights["wk"][l] @ xb + bias("bk"), "k_norm", l), pos, n_kv_heads))
            values.append(weights["wv"][l] @ xb + bias("bv"))
        for pos in range(len(tokens)):
            attended = np.zeros(n_heads * head_size, dtype=np.float64)
            for h in range(n_heads):
                kv = h // kv_mul
                q = queries[pos][h * head_size:(h + 1) * head_size]
                scores = np.array([float(q @ keys[t][kv * head_size:(kv + 1) * head_size]) * score_scale
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


# ------------------------------------------------------------------------------------- Qwen3.5 (T229)

def naive_qwen35_logits(tensors, config, tokens):
    """transformers' Qwen3_5 (modeling_qwen3_5.py at 7fb5bcd1: Qwen3_5DecoderLayer, Qwen3_5Attention,
    Qwen3_5GatedDeltaNet with torch_recurrent_gated_delta_rule, Qwen3_5RMSNorm and Qwen3_5RMSNormGated), written out
    from the Hugging Face tensors in float64, token by token: the reference the converter and the engine are held to
    together (tests/reference_qwen35.py holds it to transformers itself, in CI)."""
    text = config["text_config"]
    prefix = "model.language_model." if "model.language_model.embed_tokens.weight" in tensors else "model."
    wide = {name: np.asarray(tensor, dtype=np.float64) for name, tensor in tensors.items()}
    eps = text["rms_norm_eps"]
    heads, kv_heads, head_dim = text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"]
    key_heads, value_heads = text["linear_num_key_heads"], text["linear_num_value_heads"]
    key_dim, value_dim, taps = text["linear_key_head_dim"], text["linear_value_head_dim"], text["linear_conv_kernel_dim"]
    keys = key_heads * key_dim
    rot = int(head_dim * text["rope_parameters"]["partial_rotary_factor"])
    inverse = 1.0 / text["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2, dtype=np.float64) / rot)
    sigmoid = lambda v: 1.0 / (1.0 + np.exp(-v))
    silu = lambda v: v * sigmoid(v)
    norm = lambda v, weight: v / math.sqrt(float(v @ v) / len(v) + eps) * (1.0 + weight)  # Qwen3_5RMSNorm

    def rotate(head, pos):
        """apply_rotary_pos_emb on one head: its first rot values turn, as rotate_half pairs them."""
        cos, sin = np.cos(pos * inverse), np.sin(pos * inverse)
        cos, sin = np.concatenate([cos, cos]), np.concatenate([sin, sin])
        turned = head[:rot]
        half = np.concatenate([-turned[rot // 2:], turned[:rot // 2]])
        return np.concatenate([turned * cos + half * sin, head[rot:]])

    x = [wide[prefix + "embed_tokens.weight"][token] for token in tokens]
    for layer, kind in enumerate(text["layer_types"]):
        p = f"{prefix}layers.{layer}."
        normed = [norm(v, wide[p + "input_layernorm.weight"]) for v in x]
        mixed = []
        if kind == "full_attention":
            a = p + "self_attn."
            queries, gates, ks, vs = [], [], [], []
            for pos, v in enumerate(normed):
                both = (wide[a + "q_proj.weight"] @ v).reshape(heads, 2, head_dim)  # each head: q, then its gate
                queries.append([rotate(norm(both[h, 0], wide[a + "q_norm.weight"]), pos) for h in range(heads)])
                gates.append(both[:, 1].reshape(-1))
                k = (wide[a + "k_proj.weight"] @ v).reshape(kv_heads, head_dim)
                ks.append([rotate(norm(k[h], wide[a + "k_norm.weight"]), pos) for h in range(kv_heads)])
                vs.append((wide[a + "v_proj.weight"] @ v).reshape(kv_heads, head_dim))
            for pos in range(len(tokens)):
                attended = np.zeros((heads, head_dim))
                for h in range(heads):
                    kv = h // (heads // kv_heads)
                    scores = np.array([queries[pos][h] @ ks[t][kv] / math.sqrt(head_dim) for t in range(pos + 1)])
                    scores = np.exp(scores - scores.max())
                    scores /= scores.sum()
                    attended[h] = sum(scores[t] * vs[t][kv] for t in range(pos + 1))
                mixed.append(wide[a + "o_proj.weight"] @ (attended.reshape(-1) * sigmoid(gates[pos])))
        else:
            a = p + "linear_attn."
            state = np.zeros((value_heads, key_dim, value_dim))
            projected = [wide[a + "in_proj_qkv.weight"] @ v for v in normed]
            weight = wide[a + "conv1d.weight"][:, 0, :]  # (channels, taps): the last tap is this token's
            for pos, v in enumerate(normed):
                convolved = np.zeros(len(weight))
                for j in range(taps):
                    at = pos - (taps - 1) + j
                    if at >= 0:
                        convolved += weight[:, j] * projected[at]
                convolved = silu(convolved)
                q = convolved[:keys].reshape(key_heads, key_dim)
                k = convolved[keys:2 * keys].reshape(key_heads, key_dim)
                value = convolved[2 * keys:].reshape(value_heads, value_dim)
                z = (wide[a + "in_proj_z.weight"] @ v).reshape(value_heads, value_dim)
                beta = sigmoid(wide[a + "in_proj_b.weight"] @ v)
                softplus = np.log1p(np.exp(wide[a + "in_proj_a.weight"] @ v + wide[a + "dt_bias"]))
                g = -np.exp(wide[a + "A_log"]) * softplus
                out = np.zeros((value_heads, value_dim))
                for h in range(value_heads):
                    of = h // (value_heads // key_heads)  # repeat_interleave: the key head this value head reads
                    q_h = q[of] / math.sqrt(float(q[of] @ q[of]) + 1e-6) / math.sqrt(key_dim)
                    k_h = k[of] / math.sqrt(float(k[of] @ k[of]) + 1e-6)
                    state[h] *= math.exp(g[h])
                    delta = (value[h] - k_h @ state[h]) * beta[h]
                    state[h] += np.outer(k_h, delta)
                    read = q_h @ state[h]
                    # Qwen3_5RMSNormGated: the weight as it is (no 1 +), then the gate
                    out[h] = wide[a + "norm.weight"] * read / math.sqrt(float(read @ read) / value_dim + eps) * silu(z[h])
                mixed.append(wide[a + "out_proj.weight"] @ out.reshape(-1))
        for pos in range(len(tokens)):
            x[pos] = x[pos] + mixed[pos]
            v = norm(x[pos], wide[p + "post_attention_layernorm.weight"])
            x[pos] = x[pos] + wide[p + "mlp.down_proj.weight"] @ (silu(wide[p + "mlp.gate_proj.weight"] @ v)
                                                                   * (wide[p + "mlp.up_proj.weight"] @ v))
    classifier = wide["lm_head.weight"] if "lm_head.weight" in wide else wide[prefix + "embed_tokens.weight"]
    return [classifier @ norm(v, wide[prefix + "norm.weight"]) for v in x]


# ------------------------------------------------------------------------------------- LFM2 (T260)

def naive_lfm2_logits(tensors, config, tokens, states=None):
    """transformers' Lfm2 (modeling_lfm2.py at 7cd73d9d: Lfm2DecoderLayer, Lfm2ShortConv with causal_conv1d_fn,
    Lfm2Attention, Lfm2MLP, Lfm2RMSNorm), written out from the Hugging Face tensors in float64, the whole sequence at
    once as its forward of a prompt computes: the reference the converter and the engine are held to together
    (tests/reference_lfm2.py holds it to transformers itself, in CI). states: a list that gets, for every layer, what
    the layer added to x at every position (the output of its attention or of its convolution)."""
    wide = {name: np.asarray(tensor, dtype=np.float64) for name, tensor in tensors.items()}
    eps = config["norm_eps"]
    dim, heads, kv_heads = config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"]
    head_dim, taps = dim // heads, config["conv_L_cache"]
    inverse = 1.0 / config["rope_parameters"]["rope_theta"] ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    silu = lambda v: v / (1.0 + np.exp(-v))
    norm = lambda v, weight: v / math.sqrt(float(v @ v) / len(v) + eps) * weight  # Lfm2RMSNorm

    def rotate(head, pos):
        """apply_rotary_pos_emb on one head, as rotate_half pairs its values"""
        cos, sin = np.cos(pos * inverse), np.sin(pos * inverse)
        cos, sin = np.concatenate([cos, cos]), np.concatenate([sin, sin])
        return head * cos + np.concatenate([-head[head_dim // 2:], head[:head_dim // 2]]) * sin

    x = [wide["model.embed_tokens.weight"][token] for token in tokens]
    for layer, kind in enumerate(config["layer_types"]):
        p = f"model.layers.{layer}."
        normed = [norm(v, wide[p + "operator_norm.weight"]) for v in x]
        mixed = []
        if kind == "full_attention":
            a = p + "self_attn."
            queries, ks, vs = [], [], []
            for pos, v in enumerate(normed):
                q = (wide[a + "q_proj.weight"] @ v).reshape(heads, head_dim)
                k = (wide[a + "k_proj.weight"] @ v).reshape(kv_heads, head_dim)
                queries.append([rotate(norm(q[h], wide[a + "q_layernorm.weight"]), pos) for h in range(heads)])
                ks.append([rotate(norm(k[h], wide[a + "k_layernorm.weight"]), pos) for h in range(kv_heads)])
                vs.append((wide[a + "v_proj.weight"] @ v).reshape(kv_heads, head_dim))
            for pos in range(len(tokens)):
                attended = np.zeros((heads, head_dim))
                for h in range(heads):
                    kv = h // (heads // kv_heads)
                    scores = np.array([queries[pos][h] @ ks[t][kv] / math.sqrt(head_dim) for t in range(pos + 1)])
                    scores = np.exp(scores - scores.max())
                    scores /= scores.sum()
                    attended[h] = sum(scores[t] * vs[t][kv] for t in range(pos + 1))
                mixed.append(wide[a + "out_proj.weight"] @ attended.reshape(-1))
        else:
            a = p + "conv."
            weight = wide[a + "conv.weight"][:, 0, :]  # (channels, taps): the last tap is this token's
            gated, passed = [], []
            for v in normed:
                b, c, z = (wide[a + "in_proj.weight"] @ v).reshape(3, dim)  # BCx.chunk(3)
                gated.append(b * z)
                passed.append(c)
            for pos in range(len(tokens)):
                convolved = np.zeros(dim)
                for j in range(taps):
                    at = pos - (taps - 1) + j  # zeros in front of the sequence (padding = taps - 1)
                    if at >= 0:
                        convolved += weight[:, j] * gated[at]
                mixed.append(wide[a + "out_proj.weight"] @ (passed[pos] * convolved))
        if states is not None:
            states.append(mixed)
        for pos in range(len(tokens)):
            x[pos] = x[pos] + mixed[pos]
            v = norm(x[pos], wide[p + "ffn_norm.weight"])
            x[pos] = x[pos] + wide[p + "feed_forward.w2.weight"] @ (silu(wide[p + "feed_forward.w1.weight"] @ v)
                                                                   * (wide[p + "feed_forward.w3.weight"] @ v))
    classifier = wide["lm_head.weight"] if "lm_head.weight" in wide else wide["model.embed_tokens.weight"]
    return [classifier @ norm(v, wide["model.embedding_norm.weight"]) for v in x]


# ------------------------------------------------------------------------------------- Gemma 3 (T261)

def naive_gemma3_logits(tensors, config, tokens, states=None):
    """transformers' Gemma3 text model (modeling_gemma3.py at 28034545: Gemma3TextScaledWordEmbedding, Gemma3DecoderLayer,
    Gemma3Attention with the masks of Gemma3TextModel.forward, Gemma3MLP, Gemma3RMSNorm, Gemma3RotaryEmbedding), written
    out from the Hugging Face tensors in float64, the whole sequence at once as its forward of a prompt computes: the
    reference the converter and the engine are held to together (tests/reference_gemma3.py holds it to transformers
    itself, in CI). config: config.json as transformers' Gemma3TextConfig would have read it (the kinds of the layers
    listed, the two bases of RoPE by their old names). states: a list that gets, for every layer, what its attention
    gave at every position (before the norm behind it)."""
    wide = {name: np.asarray(tensor, dtype=np.float64) for name, tensor in tensors.items()}
    eps = config["rms_norm_eps"]
    dim, heads, kv_heads = config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"]
    head_dim, window = config["head_dim"], config["sliding_window"]
    inverse = {kind: 1.0 / config[key] ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
               for kind, key in (("full_attention", "rope_theta"), ("sliding_attention", "rope_local_base_freq"))}
    norm = lambda v, weight: v / math.sqrt(float(v @ v) / len(v) + eps) * (1.0 + weight)  # Gemma3RMSNorm
    # gelu_pytorch_tanh: torch's gelu(approximate="tanh")
    gelu = lambda v: 0.5 * v * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))

    def rotate(head, pos, kind):
        """apply_rotary_pos_emb on one head, as rotate_half pairs its values, by the base of the layer's kind"""
        cos, sin = np.cos(pos * inverse[kind]), np.sin(pos * inverse[kind])
        cos, sin = np.concatenate([cos, cos]), np.concatenate([sin, sin])
        return head * cos + np.concatenate([-head[head_dim // 2:], head[:head_dim // 2]]) * sin

    x = [wide["model.embed_tokens.weight"][token] * float(np.float32(dim ** 0.5)) for token in tokens]
    for layer, kind in enumerate(config["layer_types"]):
        p, a = f"model.layers.{layer}.", f"model.layers.{layer}.self_attn."
        normed = [norm(v, wide[p + "input_layernorm.weight"]) for v in x]
        queries, ks, vs = [], [], []
        for pos, v in enumerate(normed):
            q = (wide[a + "q_proj.weight"] @ v).reshape(heads, head_dim)
            k = (wide[a + "k_proj.weight"] @ v).reshape(kv_heads, head_dim)
            queries.append([rotate(norm(q[h], wide[a + "q_norm.weight"]), pos, kind) for h in range(heads)])
            ks.append([rotate(norm(k[h], wide[a + "k_norm.weight"]), pos, kind) for h in range(kv_heads)])
            vs.append((wide[a + "v_proj.weight"] @ v).reshape(kv_heads, head_dim))
        mixed = []
        for pos in range(len(tokens)):
            # the causal mask, and over it the window of a sliding-attention layer: position t where t > pos - window
            seen = [t for t in range(pos + 1) if kind == "full_attention" or t > pos - window]
            attended = np.zeros((heads, head_dim))
            for h in range(heads):
                kv = h // (heads // kv_heads)
                scores = np.array([queries[pos][h] @ ks[t][kv] * config["query_pre_attn_scalar"] ** -0.5 for t in seen])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                attended[h] = sum(score * vs[t][kv] for score, t in zip(scores, seen))
            mixed.append(wide[a + "o_proj.weight"] @ attended.reshape(-1))
        if states is not None:
            states.append(mixed)
        for pos in range(len(tokens)):
            x[pos] = x[pos] + norm(mixed[pos], wide[p + "post_attention_layernorm.weight"])
            v = norm(x[pos], wide[p + "pre_feedforward_layernorm.weight"])
            out = wide[p + "mlp.down_proj.weight"] @ (gelu(wide[p + "mlp.gate_proj.weight"] @ v) * (wide[p + "mlp.up_proj.weight"] @ v))
            x[pos] = x[pos] + norm(out, wide[p + "post_feedforward_layernorm.weight"])
    classifier = wide["lm_head.weight"] if "lm_head.weight" in wide else wide["model.embed_tokens.weight"]
    return [classifier @ norm(v, wide["model.norm.weight"]) for v in x]


# ------------------------------------------------------------------------------------- Gemma 4 (T331)

def naive_gemma4_logits(tensors, config, tokens, states=None):
    """transformers' Gemma 4 language model (modeling_gemma4.py at 2126f65b: Gemma4TextScaledWordEmbedding,
    Gemma4TextDecoderLayer, Gemma4TextAttention with the masks of Gemma4TextModel.forward, Gemma4TextMLP, Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding with modeling_rope_utils.py's proportional RoPE, Gemma4ForCausalLM's cap of the logits),
    written out from the Hugging Face tensors in float64, the whole sequence at once as its forward of a prompt
    computes, independently of the converter and the engine. config: the text config.json as it is published (its
    rope_parameters a dict for each kind of layer). T330: the last num_kv_shared_layers layers take the keys and values
    the last layer of their kind before them made (shared_kv_states), and make none. T329: every layer's input of its own
    (get_per_layer_inputs(), project_per_layer_inputs()), and what the layer makes of it. states: a list that gets, for
    every layer, what its attention gave at every position (before the norm behind it)."""
    wide = {name: np.asarray(tensor, dtype=np.float64) for name, tensor in tensors.items()}
    m = "model.language_model." if "model.language_model.embed_tokens.weight" in wide else "model."
    eps, dim = config["rms_norm_eps"], config["hidden_size"]
    heads, kv_heads, window = config["num_attention_heads"], config["num_key_value_heads"], config["sliding_window"]
    sizes = {"sliding_attention": config["head_dim"], "full_attention": config["global_head_dim"]}
    rope = config["rope_parameters"]

    def inverse(kind):
        """the angle per position of each pair (i, i + size / 2) of a head of the layer's kind"""
        size, said = sizes[kind], rope[kind]
        angles = 1.0 / said["rope_theta"] ** (np.arange(0, size, 2, dtype=np.float64) / size)
        if said["rope_type"] == "proportional":  # the first quarter's angles, of the whole head's width; 0 past them
            turned = int(said["partial_rotary_factor"] * size // 2)
            angles[turned:] = 0.0
        return angles

    inverses = {kind: inverse(kind) for kind in sizes}
    norm = lambda v, weight=None: v / math.sqrt(float(v @ v) / len(v) + eps) * (1.0 if weight is None else weight)
    gelu = lambda v: 0.5 * v * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))

    def rotate(head, pos, kind):
        size = sizes[kind]
        cos, sin = np.cos(pos * inverses[kind]), np.sin(pos * inverses[kind])
        cos, sin = np.concatenate([cos, cos]), np.concatenate([sin, sin])
        return head * cos + np.concatenate([-head[size // 2:], head[:size // 2]]) * sin

    x = [wide[m + "embed_tokens.weight"][token] * dim ** 0.5 for token in tokens]
    first_shared, kept = len(config["layer_types"]) - config.get("num_kv_shared_layers", 0), {}
    width, layers = config.get("hidden_size_per_layer_input") or 0, len(config["layer_types"])
    inputs = None
    if width:
        projected = [(wide[m + "per_layer_model_projection.weight"] @ v * dim ** -0.5).reshape(layers, width) for v in x]
        normed = [np.array([norm(row, wide[m + "per_layer_projection_norm.weight"]) for row in rows]) for rows in projected]
        own = [wide[m + "embed_tokens_per_layer.weight"][token].reshape(layers, width) * width ** 0.5 for token in tokens]
        inputs = [(a + b) * 2 ** -0.5 for a, b in zip(normed, own)]
    for layer, kind in enumerate(config["layer_types"]):
        p, a, size = f"{m}layers.{layer}.", f"{m}layers.{layer}.self_attn.", sizes[kind]
        normed = [norm(v, wide[p + "input_layernorm.weight"]) for v in x]
        queries, ks, vs = [], [], []
        for pos, v in enumerate(normed):
            q = (wide[a + "q_proj.weight"] @ v).reshape(heads, size)
            queries.append([rotate(norm(q[h], wide[a + "q_norm.weight"]), pos, kind) for h in range(heads)])
            if layer < first_shared:
                k = (wide[a + "k_proj.weight"] @ v).reshape(kv_heads, size)
                values = (wide[a + "v_proj.weight"] @ v).reshape(kv_heads, size)
                ks.append([rotate(norm(k[h], wide[a + "k_norm.weight"]), pos, kind) for h in range(kv_heads)])
                vs.append([norm(values[h]) for h in range(kv_heads)])
        if layer < first_shared:
            kept[kind] = (ks, vs)  # (the last layer of its kind before the shared ones leaves its own)
        else:
            ks, vs = kept[kind]
        mixed = []
        for pos in range(len(tokens)):
            seen = [t for t in range(pos + 1) if kind == "full_attention" or t > pos - window]
            attended = np.zeros((heads, size))
            for h in range(heads):
                kv = h // (heads // kv_heads)
                scores = np.array([queries[pos][h] @ ks[t][kv] for t in seen])  # (scaling 1.0)
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                attended[h] = sum(score * vs[t][kv] for score, t in zip(scores, seen))
            mixed.append(wide[a + "o_proj.weight"] @ attended.reshape(-1))
        if states is not None:
            states.append(mixed)
        for pos in range(len(tokens)):
            x[pos] = x[pos] + norm(mixed[pos], wide[p + "post_attention_layernorm.weight"])
            v = norm(x[pos], wide[p + "pre_feedforward_layernorm.weight"])
            out = wide[p + "mlp.down_proj.weight"] @ (gelu(wide[p + "mlp.gate_proj.weight"] @ v) * (wide[p + "mlp.up_proj.weight"] @ v))
            x[pos] = x[pos] + norm(out, wide[p + "post_feedforward_layernorm.weight"])
            if inputs is not None:
                gated = gelu(wide[p + "per_layer_input_gate.weight"] @ x[pos]) * inputs[pos][layer]
                x[pos] = x[pos] + norm(wide[p + "per_layer_projection.weight"] @ gated, wide[p + "post_per_layer_input_norm.weight"])
            x[pos] = x[pos] * wide[p + "layer_scalar"][0]
    classifier = wide["lm_head.weight"] if "lm_head.weight" in wide else wide[m + "embed_tokens.weight"]
    cap = config.get("final_logit_softcapping")
    logits = [classifier @ norm(v, wide[m + "norm.weight"]) for v in x]
    return [cap * np.tanh(row / cap) for row in logits] if cap else logits
