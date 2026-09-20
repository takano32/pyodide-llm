"""GPT-NeoX (T72): a GPT-2 that rotates part of every head, and may run its two branches in parallel.

The reference is written from the Hugging Face tensors, the way transformers computes them, so it checks the
converter (query_key_value holds q, k and v interleaved per head) and the engine together.
"""
import json
import struct

import numpy as np
import pytest
from conftest import pack_tokenizer, tiny_vocab
from test_convert import converted, reader, safetensors_file, streamed

from llama2_convert import Safetensors, normalize, rotary_dim
from llama2_numpy import Llama

DIM, HEADS, LAYERS, HIDDEN, VOCAB, POSITIONS = 32, 4, 2, 64, 320, 24
HEAD = DIM // HEADS


def neox_model(rotary_pct=0.25, parallel=True, seed=0):
    """Random Hugging Face tensors of a tiny GPT-NeoX, and its config.json."""
    rng = np.random.default_rng(seed)
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    tensors = {"gpt_neox.embed_in.weight": normal(VOCAB, DIM), "embed_out.weight": normal(VOCAB, DIM),
               "gpt_neox.final_layer_norm.weight": (1.0 + normal(DIM) * 0.1).astype(np.float32),
               "gpt_neox.final_layer_norm.bias": normal(DIM)}
    for layer in range(LAYERS):
        p = f"gpt_neox.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors[p + norm + ".weight"] = (1.0 + normal(DIM) * 0.1).astype(np.float32)
            tensors[p + norm + ".bias"] = normal(DIM)
        tensors[p + "attention.query_key_value.weight"] = normal(3 * DIM, DIM)
        tensors[p + "attention.query_key_value.bias"] = normal(3 * DIM)
        tensors[p + "attention.dense.weight"] = normal(DIM, DIM)
        tensors[p + "attention.dense.bias"] = normal(DIM)
        tensors[p + "mlp.dense_h_to_4h.weight"] = normal(HIDDEN, DIM)
        tensors[p + "mlp.dense_h_to_4h.bias"] = normal(HIDDEN)
        tensors[p + "mlp.dense_4h_to_h.weight"] = normal(DIM, HIDDEN)
        tensors[p + "mlp.dense_4h_to_h.bias"] = normal(DIM)
        # transformers ignores these two, and so must the converter
        tensors[p + "attention.masked_bias"] = normal(1)
        tensors[p + "attention.rotary_emb.inv_freq"] = normal(HEAD // 2)
    config = dict(model_type="gpt_neox", hidden_size=DIM, num_attention_heads=HEADS, num_hidden_layers=LAYERS,
                  intermediate_size=HIDDEN, max_position_embeddings=POSITIONS, vocab_size=VOCAB,
                  rotary_pct=rotary_pct, rotary_emb_base=10000.0, use_parallel_residual=parallel,
                  hidden_act="gelu", tie_word_embeddings=False)
    return tensors, config


def reference(tensors, config, tokens):
    """transformers' GPTNeoXForCausalLM, written out in float64."""
    rot = rotary_dim(normalize(config))
    parallel = config["use_parallel_residual"]
    ln = lambda v, w, b: w * (v - v.mean()) / np.sqrt(((v - v.mean()) ** 2).mean() + 1e-5) + b
    gelu = lambda v: 0.5 * v * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (v + 0.044715 * v ** 3)))
    wide = {name: tensor.astype(np.float64) for name, tensor in tensors.items()}
    inverse = 1.0 / 10000.0 ** (np.arange(0, rot, 2, dtype=np.float64) / rot)

    def rotate(vector, pos):
        """transformers rotates the first rot values of a head with rotate_half."""
        angles = pos * inverse
        cos, sin = np.cos(angles), np.sin(angles)
        out = vector.copy()
        half = rot // 2
        first, second = vector[:half], vector[half:rot]
        out[:half] = first * cos - second * sin
        out[half:rot] = second * cos + first * sin
        return out

    x = [wide["gpt_neox.embed_in.weight"][token] for token in tokens]
    for layer in range(LAYERS):
        p = f"gpt_neox.layers.{layer}."
        fused = [ln(v, wide[p + "input_layernorm.weight"], wide[p + "input_layernorm.bias"])
                 @ wide[p + "attention.query_key_value.weight"].T + wide[p + "attention.query_key_value.bias"]
                 for v in x]
        # (heads, 3, head_size) per position
        parts = [row.reshape(HEADS, 3, HEAD) for row in fused]
        q = [np.concatenate([rotate(row[h, 0], pos) for h in range(HEADS)]) for pos, row in enumerate(parts)]
        k = [np.concatenate([rotate(row[h, 1], pos) for h in range(HEADS)]) for pos, row in enumerate(parts)]
        v = [row[:, 2].reshape(-1) for row in parts]
        for pos in range(len(tokens)):
            attended = np.zeros(DIM)
            for h in range(HEADS):
                piece = slice(h * HEAD, (h + 1) * HEAD)
                scores = np.array([q[pos][piece] @ k[t][piece] / np.sqrt(HEAD) for t in range(pos + 1)])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                attended[piece] = sum(scores[t] * v[t][piece] for t in range(pos + 1))
            attention = attended @ wide[p + "attention.dense.weight"].T + wide[p + "attention.dense.bias"]
            source = x[pos] if parallel else x[pos] + attention
            hidden = ln(source, wide[p + "post_attention_layernorm.weight"], wide[p + "post_attention_layernorm.bias"])
            hidden = gelu(hidden @ wide[p + "mlp.dense_h_to_4h.weight"].T + wide[p + "mlp.dense_h_to_4h.bias"])
            hidden = hidden @ wide[p + "mlp.dense_4h_to_h.weight"].T + wide[p + "mlp.dense_4h_to_h.bias"]
            x[pos] = x[pos] + attention + hidden
    return [wide["embed_out.weight"] @ ln(v, wide["gpt_neox.final_layer_norm.weight"],
                                          wide["gpt_neox.final_layer_norm.bias"]) for v in x]


@pytest.mark.parametrize("rotary_pct, parallel", [(0.25, True), (1.0, False), (0.5, True)])
def test_a_neox_converts_and_runs_like_transformers(rotary_pct, parallel):
    tensors, config = neox_model(rotary_pct, parallel)
    file = safetensors_file(tensors)
    checkpoint = converted(Safetensors(reader(file)), config, "float32")
    assert struct.unpack_from("<7i", checkpoint, 0) == (DIM, HIDDEN, LAYERS, HEADS, HEADS, -VOCAB, POSITIONS)
    llama = Llama(checkpoint, pack_tokenizer(tiny_vocab(VOCAB)), arch="neox",
                  rotary=rotary_dim(normalize(config)), parallel_residual=parallel)
    tokens = [1, 5, 7, 9, 11]
    want = reference(tensors, config, tokens)
    for pos, token in enumerate(tokens):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rotary_pct, parallel", [(0.25, True), (1.0, False)])
def test_the_conversion_tells_the_engine_what_the_file_cannot(rotary_pct, parallel):
    """The checkpoint says nothing about the rotated part or the parallel branches: the options must carry them,
    or the model runs and writes nonsense (which is how this was found, in the browser)."""
    import llama2_convert
    tensors, config = neox_model(rotary_pct, parallel)
    file = safetensors_file(tensors)
    header, base = file[8:8 + struct.unpack("<Q", file[:8])[0]].decode(), 8 + struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(VOCAB)]}}).encode()
    conversion = llama2_convert.Conversion(header, base, json.dumps(config), vocabulary,
                                           "tokenizer.json", dtype="float32", max_seq_len=POSITIONS)
    assert conversion.options["arch"] == "neox"
    assert conversion.options["rotary"] == rotary_dim(normalize(config))
    assert conversion.options["parallel_residual"] is parallel


def test_the_file_in_its_own_order_gives_the_same_checkpoint():
    tensors, config = neox_model()
    file = safetensors_file(tensors)
    expected = converted(Safetensors(reader(file)), config, "float32")
    got, progress = streamed(file, config, "float32", 4096)
    assert got == expected and progress[-1][0] == progress[-1][1]
