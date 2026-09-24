"""GPT-2 (T65): LayerNorm, GELU, learned positions, a bias on every projection, and an FFN of two matrices.

The reference here is written from the Hugging Face tensors themselves, the way transformers computes them, so
it checks the converter (Conv1D is transposed, and c_attn holds q, k and v in one matrix) and the engine at once.
"""
import struct

import numpy as np
import pytest
from conftest import pack_tokenizer, tiny_vocab
from test_convert import converted, reader, safetensors_file, streamed

from llama2_convert import Safetensors, checkpoint_header, normalize
from llama2_numpy import Llama

DIM, HEADS, LAYERS, HIDDEN, VOCAB, POSITIONS = 32, 4, 2, 64, 320, 24


def gpt2_model(seed=0, shared=True):
    """Random Hugging Face tensors of a tiny GPT-2, and its config.json."""
    rng = np.random.default_rng(seed)
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    tensors = {"transformer.wte.weight": normal(VOCAB, DIM), "transformer.wpe.weight": normal(POSITIONS, DIM),
               "transformer.ln_f.weight": (1.0 + normal(DIM) * 0.1).astype(np.float32),
               "transformer.ln_f.bias": normal(DIM)}
    for layer in range(LAYERS):
        prefix = f"transformer.h.{layer}."
        tensors[prefix + "ln_1.weight"] = (1.0 + normal(DIM) * 0.1).astype(np.float32)
        tensors[prefix + "ln_1.bias"] = normal(DIM)
        tensors[prefix + "ln_2.weight"] = (1.0 + normal(DIM) * 0.1).astype(np.float32)
        tensors[prefix + "ln_2.bias"] = normal(DIM)
        tensors[prefix + "attn.c_attn.weight"] = normal(DIM, 3 * DIM)   # Conv1D: x @ W
        tensors[prefix + "attn.c_attn.bias"] = normal(3 * DIM)
        tensors[prefix + "attn.c_proj.weight"] = normal(DIM, DIM)
        tensors[prefix + "attn.c_proj.bias"] = normal(DIM)
        tensors[prefix + "mlp.c_fc.weight"] = normal(DIM, HIDDEN)
        tensors[prefix + "mlp.c_fc.bias"] = normal(HIDDEN)
        tensors[prefix + "mlp.c_proj.weight"] = normal(HIDDEN, DIM)
        tensors[prefix + "mlp.c_proj.bias"] = normal(DIM)
    if not shared:
        tensors["lm_head.weight"] = normal(VOCAB, DIM)
    config = dict(model_type="gpt2", n_embd=DIM, n_head=HEADS, n_layer=LAYERS, n_inner=HIDDEN,
                  n_positions=POSITIONS, vocab_size=VOCAB, activation_function="gelu_new",
                  tie_word_embeddings=shared)
    return tensors, config


def reference(tensors, tokens, shared=True):
    """transformers' GPT2LMHeadModel, written out in float64."""
    head_size = DIM // HEADS
    ln = lambda v, w, b: w * (v - v.mean()) / np.sqrt(((v - v.mean()) ** 2).mean() + 1e-5) + b
    gelu = lambda v: 0.5 * v * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (v + 0.044715 * v ** 3)))
    wide = {name: tensor.astype(np.float64) for name, tensor in tensors.items()}
    x = [wide["transformer.wte.weight"][token] + wide["transformer.wpe.weight"][pos]
         for pos, token in enumerate(tokens)]
    for layer in range(LAYERS):
        p = f"transformer.h.{layer}."
        qkv = [ln(v, wide[p + "ln_1.weight"], wide[p + "ln_1.bias"]) @ wide[p + "attn.c_attn.weight"]
               + wide[p + "attn.c_attn.bias"] for v in x]
        q, k, v = ([row[i * DIM:(i + 1) * DIM] for row in qkv] for i in range(3))
        for pos in range(len(tokens)):
            attended = np.zeros(DIM)
            for h in range(HEADS):
                piece = slice(h * head_size, (h + 1) * head_size)
                scores = np.array([q[pos][piece] @ k[t][piece] / np.sqrt(head_size) for t in range(pos + 1)])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                attended[piece] = sum(scores[t] * v[t][piece] for t in range(pos + 1))
            x[pos] = x[pos] + attended @ wide[p + "attn.c_proj.weight"] + wide[p + "attn.c_proj.bias"]
            h2 = ln(x[pos], wide[p + "ln_2.weight"], wide[p + "ln_2.bias"])
            hidden = gelu(h2 @ wide[p + "mlp.c_fc.weight"] + wide[p + "mlp.c_fc.bias"])
            x[pos] = x[pos] + hidden @ wide[p + "mlp.c_proj.weight"] + wide[p + "mlp.c_proj.bias"]
    classifier = wide["transformer.wte.weight"] if shared else wide["lm_head.weight"]
    return [classifier @ ln(v, wide["transformer.ln_f.weight"], wide["transformer.ln_f.bias"]) for v in x]


@pytest.mark.parametrize("shared", [True, False])
def test_a_gpt2_converts_and_runs_like_transformers(shared):
    tensors, config = gpt2_model(shared=shared)
    file = safetensors_file(tensors)
    checkpoint = converted(Safetensors(reader(file)), config, "float32")
    assert struct.unpack_from("<7i", checkpoint, 0) == (DIM, HIDDEN, LAYERS, HEADS, HEADS,
                                                        VOCAB if shared else -VOCAB, POSITIONS)
    llama = Llama(checkpoint, pack_tokenizer(tiny_vocab(VOCAB)), arch="gpt2")
    tokens = [1, 5, 7, 9, 11]
    want = reference(tensors, tokens, shared)
    for pos, token in enumerate(tokens):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=1e-4, atol=1e-4)


def test_the_conversion_tells_the_engine_the_architecture():
    """The file cannot say it is a GPT-2: the options must (the lesson of T72, applied here in T77)."""
    import json

    import llama2_convert
    tensors, config = gpt2_model()
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(VOCAB)]}}).encode()
    conversion = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), vocabulary,
                                           "tokenizer.json", dtype="float32", max_seq_len=POSITIONS)
    assert conversion.options["arch"] == "gpt2" and conversion.options["bias"] is False


def test_the_file_in_its_own_order_gives_the_same_checkpoint():
    tensors, config = gpt2_model()
    file = safetensors_file(tensors)
    expected = converted(Safetensors(reader(file)), config, "float32")
    got, progress = streamed(file, config, "float32", 4096)
    assert got == expected
    assert progress[-1][0] == progress[-1][1]


def test_the_context_of_a_gpt2_is_its_table_of_positions():
    tensors, config = gpt2_model()
    source = Safetensors(reader(safetensors_file(tensors)))
    # a shorter context would need a shorter table: the learned positions cannot be cut the way RoPE can
    assert checkpoint_header(normalize(config), source, 8)[6] == POSITIONS
