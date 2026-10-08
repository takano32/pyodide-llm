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
sys.path.insert(0, str(ROOT / "public"))
sys.path.insert(0, str(ROOT))  # quantize.py

# tmp_path lives under the system's temporary directory, which on the development machine is a tmpfs, that is memory
# (AGENTS.md), and was written there without a word (the owner, 2026-09-26): here it is the repository's .tmp, which
# .gitignore has. pytest reads this where it first makes a tmp_path, after the conftests; a caller may say otherwise.
(ROOT / ".tmp").mkdir(exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(ROOT / ".tmp"))

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


def rope_tables(seq_len, head_size, rope_theta=10000.0):
    angles = np.arange(seq_len)[:, None] / rope_theta ** (np.arange(0, head_size, 2) / head_size)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def synthetic_weights(dim=32, hidden_dim=64, n_layers=2, n_heads=4, n_kv_heads=4,
                      vocab_size=320, seq_len=24, shared=True, seed=0, head_size=0):
    """Random weights for a tiny model, plus its configuration. Small enough for a naive reference.
    head_size: where a head is not dim / n_heads (T124), q and the attention's output are n_heads * head_size wide."""
    rng = np.random.default_rng(seed)
    head_size = head_size or dim // n_heads
    q_dim, kv_dim = n_heads * head_size, n_kv_heads * head_size
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    cos, sin = rope_tables(seq_len, head_size)
    weights = {
        "token_embedding_table": normal(vocab_size, dim),
        "rms_att_weight": (1.0 + normal(n_layers, dim) * 0.1).astype(np.float32),
        "wq": normal(n_layers, q_dim, dim), "wk": normal(n_layers, kv_dim, dim),
        "wv": normal(n_layers, kv_dim, dim), "wo": normal(n_layers, dim, q_dim),
        "rms_ffn_weight": (1.0 + normal(n_layers, dim) * 0.1).astype(np.float32),
        "w1": normal(n_layers, hidden_dim, dim), "w2": normal(n_layers, dim, hidden_dim),
        "w3": normal(n_layers, hidden_dim, dim),
        "rms_final_weight": (1.0 + normal(dim) * 0.1).astype(np.float32),
        "freq_cis_real": cos, "freq_cis_imag": sin,
    }
    weights["wcls"] = weights["token_embedding_table"] if shared else normal(vocab_size, dim)
    config = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
                  n_kv_heads=n_kv_heads, vocab_size=vocab_size, seq_len=seq_len,
                  head_size=head_size, q_dim=q_dim, kv_dim=kv_dim, shared=shared)
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

def qwen35_model(dim=32, hidden_dim=64, n_layers=4, every=2, n_heads=4, n_kv_heads=2, head_dim=16, rotary=0.25,
                 key_heads=2, value_heads=4, key_dim=8, value_dim=6, conv=4, vocab_size=320, seq_len=24, shared=True,
                 prefix="model.language_model.", seed=0, eps=1e-6):
    """Random Hugging Face tensors of a tiny Qwen3.5 (hybrid attention), with the names and shapes of
    Qwen/Qwen3.5-0.8B's model.safetensors, and its config.json (the language model's under text_config, as there)."""
    rng = np.random.default_rng(seed)
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    keys, values = key_heads * key_dim, value_heads * value_dim
    kinds = ["linear_attention" if (layer + 1) % every else "full_attention" for layer in range(n_layers)]
    tensors = {prefix + "embed_tokens.weight": normal(vocab_size, dim), prefix + "norm.weight": normal(dim)}
    for layer, kind in enumerate(kinds):
        p = f"{prefix}layers.{layer}."
        tensors[p + "input_layernorm.weight"] = normal(dim)  # around zero: the model multiplies by 1 + weight
        tensors[p + "post_attention_layernorm.weight"] = normal(dim)
        if kind == "full_attention":
            tensors[p + "self_attn.q_proj.weight"] = normal(2 * n_heads * head_dim, dim)
            tensors[p + "self_attn.k_proj.weight"] = normal(n_kv_heads * head_dim, dim)
            tensors[p + "self_attn.v_proj.weight"] = normal(n_kv_heads * head_dim, dim)
            tensors[p + "self_attn.o_proj.weight"] = normal(dim, n_heads * head_dim)
            tensors[p + "self_attn.q_norm.weight"] = normal(head_dim)
            tensors[p + "self_attn.k_norm.weight"] = normal(head_dim)
        else:
            tensors[p + "linear_attn.in_proj_qkv.weight"] = normal(2 * keys + values, dim)
            tensors[p + "linear_attn.in_proj_z.weight"] = normal(values, dim)
            tensors[p + "linear_attn.in_proj_b.weight"] = normal(value_heads, dim)
            tensors[p + "linear_attn.in_proj_a.weight"] = normal(value_heads, dim)
            tensors[p + "linear_attn.conv1d.weight"] = normal(2 * keys + values, 1, conv)
            tensors[p + "linear_attn.dt_bias"] = normal(value_heads)
            tensors[p + "linear_attn.A_log"] = normal(value_heads)
            tensors[p + "linear_attn.norm.weight"] = (1.0 + normal(value_dim)).astype(np.float32)
            tensors[p + "linear_attn.out_proj.weight"] = normal(dim, values)
        tensors[p + "mlp.gate_proj.weight"] = normal(hidden_dim, dim)
        tensors[p + "mlp.up_proj.weight"] = normal(hidden_dim, dim)
        tensors[p + "mlp.down_proj.weight"] = normal(dim, hidden_dim)
    if not shared:
        tensors["lm_head.weight"] = normal(vocab_size, dim)
    # what transformers leaves unread: the vision model and the look-ahead head
    tensors["model.visual.patch_embed.proj.bias"] = normal(8)
    tensors["mtp.norm.weight"] = normal(dim)
    text = dict(model_type="qwen3_5_text", hidden_size=dim, intermediate_size=hidden_dim, num_hidden_layers=n_layers,
                num_attention_heads=n_heads, num_key_value_heads=n_kv_heads, head_dim=head_dim, hidden_act="silu",
                layer_types=kinds, full_attention_interval=every, linear_conv_kernel_dim=conv,
                linear_key_head_dim=key_dim, linear_value_head_dim=value_dim, linear_num_key_heads=key_heads,
                linear_num_value_heads=value_heads, max_position_embeddings=seq_len, rms_norm_eps=eps,
                vocab_size=vocab_size, tie_word_embeddings=shared, eos_token_id=7, attn_output_gate=True,
                mlp_only_layers=[],
                rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": rotary,
                                 "mrope_interleaved": True, "mrope_section": [11, 11, 10]})
    return tensors, dict(model_type="qwen3_5", text_config=text, tie_word_embeddings=shared)


# T237: a model folded into a rotated basis (tests/test_rotated.py, tests/make_qwen35.py)
# the matrices of a Hugging Face checkpoint that the forward pass multiplies an activation by, and the embedding
FOLDED = ("embed_tokens.weight", "lm_head.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
          "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
          "mlp.down_proj.weight", "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
          "linear_attn.out_proj.weight")


def basis(block, widths, seed=5):
    """A rotated basis with random signs for these widths, as the options say one, and the signs."""
    rng = np.random.default_rng(seed)
    signs = {width: rng.choice([-1.0, 1.0], width).astype(np.float32) for width in sorted(set(widths))}
    return {"block": block, "signs": {str(width): llama2_numpy.sign_bits(values) for width, values in signs.items()}}, signs


def folded(tensors, block, signs):
    """The tensors of a model as a file in the rotated basis holds them: every row of a matrix (and of the embedding)
    is R of the row, with the signs of the row's length."""
    return {name: llama2_numpy.rotate(tensor, signs[tensor.shape[-1]], block).astype(np.float32) if name.endswith(FOLDED) else tensor
            for name, tensor in tensors.items()}


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

def lfm2_model(dim=32, block_ff_dim=96, kinds="ccaccaca", n_heads=4, n_kv_heads=2, taps=3, vocab_size=320, seq_len=24,
               shared=True, seed=0, eps=1e-5, adjust=True):
    """Random Hugging Face tensors of a tiny LFM2 (convolution layers among attention layers), with the names and
    shapes of LiquidAI/LFM2.5-350M's model.safetensors, and its config.json. kinds: a letter a layer, c for a
    convolution layer and a for an attention layer. block_ff_dim is what the config says; the FFN's inside is what
    transformers' Lfm2MLP makes of it (two thirds, up to a multiple of block_multiple_of: 96 becomes 64)."""
    rng = np.random.default_rng(seed)
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    weight = lambda n: (1.0 + normal(n)).astype(np.float32)  # a norm's, as stored: the model multiplies by it
    multiple = 32
    hidden_dim = multiple * ((int(2 * block_ff_dim / 3) + multiple - 1) // multiple) if adjust else block_ff_dim
    head_dim = dim // n_heads
    tensors = {"model.embed_tokens.weight": normal(vocab_size, dim), "model.embedding_norm.weight": weight(dim)}
    for layer, kind in enumerate(kinds):
        p = f"model.layers.{layer}."
        tensors[p + "operator_norm.weight"] = weight(dim)
        tensors[p + "ffn_norm.weight"] = weight(dim)
        if kind == "a":
            tensors[p + "self_attn.q_proj.weight"] = normal(n_heads * head_dim, dim)
            tensors[p + "self_attn.k_proj.weight"] = normal(n_kv_heads * head_dim, dim)
            tensors[p + "self_attn.v_proj.weight"] = normal(n_kv_heads * head_dim, dim)
            tensors[p + "self_attn.out_proj.weight"] = normal(dim, n_heads * head_dim)
            tensors[p + "self_attn.q_layernorm.weight"] = weight(head_dim)
            tensors[p + "self_attn.k_layernorm.weight"] = weight(head_dim)
        else:
            tensors[p + "conv.in_proj.weight"] = normal(3 * dim, dim)
            tensors[p + "conv.conv.weight"] = normal(dim, 1, taps)
            tensors[p + "conv.out_proj.weight"] = normal(dim, dim)
        tensors[p + "feed_forward.w1.weight"] = normal(hidden_dim, dim)
        tensors[p + "feed_forward.w3.weight"] = normal(hidden_dim, dim)
        tensors[p + "feed_forward.w2.weight"] = normal(dim, hidden_dim)
    if not shared:
        tensors["lm_head.weight"] = normal(vocab_size, dim)
    config = dict(model_type="lfm2", architectures=["Lfm2ForCausalLM"], hidden_size=dim, block_dim=dim, conv_dim=dim,
                  block_ff_dim=block_ff_dim, intermediate_size=block_ff_dim, block_auto_adjust_ff_dim=adjust,
                  block_ffn_dim_multiplier=1.0, block_multiple_of=multiple, num_hidden_layers=len(kinds),
                  num_attention_heads=n_heads, num_heads=n_heads, num_key_value_heads=n_kv_heads,
                  layer_types=["conv" if kind == "c" else "full_attention" for kind in kinds], conv_L_cache=taps,
                  conv_bias=False, norm_eps=eps, block_norm_eps=eps, max_position_embeddings=seq_len, vocab_size=vocab_size,
                  bos_token_id=1, eos_token_id=7, pad_token_id=0, tie_embedding=shared, use_pos_enc=True,
                  rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"})
    return tensors, config


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
