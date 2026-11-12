"""Microsoft's Phi-3 and Phi-4 mini (T258, model_type phi3): a Llama whose q, k and v are one matrix and whose gate and
up of the FFN are another, whose RoPE may turn the first values of each head only (Phi-4 mini's partial_rotary_factor
0.75), with a sliding window of the last positions (Phi-3 mini 4k's 2047), to which the context is cut.

The reference (naive_phi3_logits below) is written from the Hugging Face tensors the way transformers 5.16.1's
models/phi3 computes them (Phi3Attention: qkv_proj cut as q, k, v; apply_rotary_pos_emb: rotate_half over the first
rotary values; Phi3MLP: gate_up_proj chunked, gate first), so it checks the converter (the two matrices cut, the first
rotary rows of q and k interleaved) and the engine together. tests/reference_phi3.py holds that reference, and the
engine, to transformers itself (CI: PyTorch is not on the development machine)."""
import json
import math
import struct

import numpy as np
import pytest
from made_up import phi3_model, safetensors_file
from test_gguf import unigram

import llama2_convert
from llama2_convert import check_config, checkpoint_form, normalize
from llama2_numpy import Llama, checkpoint_dtype

TOKENS = [1, 5, 7, 9, 11, 5, 5, 300, 2, 17, 4, 40, 41, 42, 7, 7]
MODELS = {
    "small": dict(),
    "three quarters of a head turned": dict(rotary=0.75),
    "a quarter of a head turned, keys for every head": dict(rotary=0.25, n_kv_heads=4),
    "its classifier the embedding": dict(shared=True, rotary=0.75),
    "one key head, three layers": dict(n_kv_heads=1, layers=3),
    "another epsilon and base": dict(eps=1e-6, theta=500.0, rotary=0.5),
    "a window past the text": dict(window=40),
}


def naive_phi3_logits(tensors, config, tokens):
    """Phi-3 as transformers' Phi3ForCausalLM computes it, written out from the Hugging Face tensors in float64, the whole
    sequence at once (every position attends every one before it: the texts here are within the window)."""
    wide = {name: np.asarray(tensor, dtype=np.float64) for name, tensor in tensors.items()}
    eps, heads, kv_heads = config["rms_norm_eps"], config["num_attention_heads"], config["num_key_value_heads"]
    size, ffn = config["hidden_size"] // heads, config["intermediate_size"]
    rot = int(size * config.get("partial_rotary_factor", 1.0))
    inverse, magnitude = frequencies_of(config, rot)
    norm = lambda v, weight: weight * (v / math.sqrt(float(v @ v) / len(v) + eps))
    silu = lambda v: v / (1.0 + np.exp(-v))

    def rotate(head, pos):
        """apply_rotary_pos_emb on one head: rotate_half over the first rot values, the rest passed through"""
        angles = np.concatenate([pos * inverse] * 2)
        cos, sin = np.cos(angles) * magnitude, np.sin(angles) * magnitude
        part = head[:rot]
        turned = part * cos + np.concatenate([-part[rot // 2:], part[:rot // 2]]) * sin
        return np.concatenate([turned, head[rot:]])

    x = [wide["model.embed_tokens.weight"][token] for token in tokens]
    q_dim, kv_dim = heads * size, kv_heads * size
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        queries, ks, vs = [], [], []
        for pos, v in enumerate(x):
            qkv = wide[p + "self_attn.qkv_proj.weight"] @ norm(v, wide[p + "input_layernorm.weight"])
            q, k = qkv[:q_dim].reshape(heads, size), qkv[q_dim:q_dim + kv_dim].reshape(kv_heads, size)
            queries.append([rotate(q[h], pos) for h in range(heads)])
            ks.append([rotate(k[h], pos) for h in range(kv_heads)])
            vs.append(qkv[q_dim + kv_dim:].reshape(kv_heads, size))
        for pos in range(len(tokens)):
            attended = np.zeros((heads, size))
            for h in range(heads):
                kv = h // (heads // kv_heads)
                scores = np.array([queries[pos][h] @ ks[t][kv] / math.sqrt(size) for t in range(pos + 1)])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                attended[h] = sum(score * vs[t][kv] for score, t in zip(scores, range(pos + 1)))
            x[pos] = x[pos] + wide[p + "self_attn.o_proj.weight"] @ attended.reshape(-1)
            gate_up = wide[p + "mlp.gate_up_proj.weight"] @ norm(x[pos], wide[p + "post_attention_layernorm.weight"])
            x[pos] = x[pos] + wide[p + "mlp.down_proj.weight"] @ (silu(gate_up[:ffn]) * gate_up[ffn:])
    classifier = wide["lm_head.weight"] if "lm_head.weight" in wide else wide["model.embed_tokens.weight"]
    return [classifier @ norm(v, wide["model.norm.weight"]) for v in x]


def frequencies_of(config, rot):
    """(inv_freq, what cos and sin are multiplied by) of transformers' Phi3RotaryEmbedding: the default's, and (T258.2)
    longrope's at a length within the original context: the short factors, and the attention factor of the ratio of
    the context to the original one."""
    inverse = 1.0 / config["rope_theta"] ** (np.arange(0, rot, 2, dtype=np.float64) / rot)
    scaling = config.get("rope_scaling") or {}
    if scaling.get("type", scaling.get("rope_type")) != "longrope":
        return inverse, 1.0
    original = config.get("original_max_position_embeddings") or scaling["original_max_position_embeddings"]
    factor = scaling.get("factor") or config["max_position_embeddings"] / original
    magnitude = scaling.get("attention_factor") or (1.0 if factor <= 1 else math.sqrt(1 + math.log(factor) / math.log(original)))
    return inverse / np.asarray(scaling["short_factor"], dtype=np.float64), magnitude


def conversion(tensors, config, dtype="float32", max_seq_len=4096):
    """The page's way: the file in its own order, with a tokenizer.json, and the options that come out."""
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    made = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), unigram(config["vocab_size"]),
                                     "tokenizer.json", dtype=dtype, max_seq_len=max_seq_len, start=8 + size)
    made.stream.feed(file[8 + size:])
    made.stream.finish()
    return made


def engine(tensors, config, dtype="float32"):
    made = conversion(tensors, config, dtype)
    options = {key: value for key, value in made.options.items() if key != "template"}
    return Llama(bytes(made.checkpoint), made.tokenizer, **options), made


def close(got, want):
    return np.allclose(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("model", MODELS)
def test_a_phi3_converts_and_runs_like_transformers(model):
    tensors, config = phi3_model(**MODELS[model])
    llama, made = engine(tensors, config)
    header = struct.unpack_from("<7i", made.checkpoint, 0)
    assert header[:5] == (64, 96, config["num_hidden_layers"], 4, config["num_key_value_heads"])
    assert header[5] == (320 if config["tie_word_embeddings"] else -320)
    assert checkpoint_dtype(header, len(made.checkpoint), made.options) == "float32"
    want = naive_phi3_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_the_quantized_files_make_their_own_tables_of_the_part_turned(dtype):
    """A quantized file has no RoPE tables: the engine makes them, of the part of a head RoPE turns, the same as the
    converter writes into a float32 file (and a float16 file's logits are a float32 file's, but for its rounding)."""
    tensors, config = phi3_model(rotary=0.75)
    llama, _ = engine(tensors, config, dtype)
    written, _ = engine(tensors, config)
    assert np.array_equal(llama.freq_cis_real, written.freq_cis_real) and np.array_equal(llama.freq_cis_imag, written.freq_cis_imag)
    if dtype == "float16":
        want = naive_phi3_logits(tensors, config, TOKENS)
        for pos, token in enumerate(TOKENS):
            assert np.abs(llama.forward(token, pos) - want[pos]).max() < 0.01 * np.abs(want[pos]).max(), pos


def test_the_conversion_tells_the_engine_what_the_file_cannot():
    """The layout and how much of a head turns (the options; a Phi-3 mini's all of it), and nothing of a window."""
    for rotary, turned in ((1.0, 16), (0.75, 12)):
        tensors, config = phi3_model(rotary=rotary)
        options = conversion(tensors, config).options
        assert options["arch"] == "phi3" and options["rotary"] == turned and "rope_scaling" not in options


@pytest.mark.parametrize("window, context, cut", [(None, 24, 24), (2047, 4096, 2047), (8, 24, 8), (40, 24, 24)])
def test_the_context_is_cut_to_the_window(window, context, cut):
    """transformers masks the last sliding_window positions (Phi-3 mini 4k: 2047 of 4096): a context no longer than the
    window attends the same ones, and a longer one would attend more than the model does."""
    tensors, config = phi3_model(window=window, seq_len=context)
    assert normalize(config)["max_position_embeddings"] == cut
    assert normalize(normalize(config)) == normalize(config)  # (twice is once)
    assert struct.unpack_from("<7i", conversion(tensors, config).checkpoint, 0)[6] == cut


@pytest.mark.parametrize("part, rows", [("q", (0, 64)), ("k", (64, 96)), ("v", (96, 128))])
def test_each_part_of_qkv_is_cut_where_transformers_cuts_it(part, rows):
    """A row of qkv_proj changed changes the part transformers reads it as (q, then k, then v), and no other."""
    tensors, config = phi3_model()
    plain = engine(tensors, config)[1].checkpoint
    changed = {**tensors, "model.layers.1.self_attn.qkv_proj.weight": tensors["model.layers.1.self_attn.qkv_proj.weight"].copy()}
    changed["model.layers.1.self_attn.qkv_proj.weight"][rows[0]] += 1.0
    llama, _ = engine(changed, config)
    want = naive_phi3_logits(changed, config, TOKENS[:6])
    for pos, token in enumerate(TOKENS[:6]):
        assert close(llama.forward(token, pos), want[pos]), (part, pos)
    assert bytes(plain) != bytes(engine(changed, config)[1].checkpoint)


def test_a_phi3_the_engine_cannot_run_is_refused():
    _, config = phi3_model()
    for change, reason in (({"hidden_act": "gelu"}, "activation"), ({"attention_bias": True}, "biases"),
                           ({"lm_head_bias": True}, "biases"), ({"mlp_bias": True}, "biases"),
                           ({"rope_scaling": {"type": "dynamic", "factor": 2.0}}, "RoPE scaling")):
        with pytest.raises(ValueError, match=reason):
            check_config(normalize({**config, **change}))


def test_the_form_is_a_phi3s():
    tensors, config = phi3_model(rotary=0.75)
    assert checkpoint_form(config, tensors)["arch"] == "phi3"
