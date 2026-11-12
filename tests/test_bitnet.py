"""Microsoft's BitNet b1.58 (T283): a Llama whose matrices are ternary, one scale a matrix, packed four values a byte in
its file; a norm in front of wo and one in front of w2; the square of a ReLU in the gate of its FFN.

The reference (naive_bitnet_logits below) is written from the Hugging Face tensors the way transformers 5.16.1 computes
them (models/bitnet/modular_bitnet.py, integrations/bitnet.py's AutoBitLinear of a file of quantization_mode "offline":
the packed values unpacked, the output times weight_scale), so it checks the converter (the matrices unpacked and
multiplied by their scales, q and k interleaved, the two norms) and the engine together. The model rounds the input of
every matrix to 8 bits by the largest of the token's values (ActQuant), which the engine does not: the reference does it
where asked (quantized=True), and the engine is held to it without. tests/reference_bitnet.py holds both to
transformers itself (CI: PyTorch is not on the development machine)."""
import json
import math
import struct

import numpy as np
import pytest
from made_up import BITNET_MATRICES, bitnet_model, bitnet_pack, safetensors_file
from test_convert import converted, reader
from test_gguf import unigram

import llama2_convert
from llama2_convert import Safetensors, check_config, checkpoint_size
from llama2_numpy import Llama, checkpoint_dtype
from convert.plan import planes_of, resolved, transformed

TOKENS = [0, 5, 7, 9, 11, 5, 5, 300, 2, 17, 4]
MODELS = {
    "small": dict(),
    "a classifier of its own": dict(shared=False),
    "a key for every head": dict(n_kv_heads=4),
    "one key head, three layers": dict(n_kv_heads=1, layers=3),
    "heads that fill the stream": dict(head_dim=32),
    "another epsilon and base": dict(eps=1e-6, theta=10000.0),
}


def naive_bitnet_logits(tensors, config, tokens, quantized=False):
    """BitNet b1.58 as transformers computes it (BitNetForCausalLM: LlamaModel's layers with BitNetAttention and
    BitNetMLP, AutoBitLinear of an offline file), written out from the Hugging Face tensors in float64, the whole
    sequence at once. quantized: every matrix's input rounded to 8 bits by the largest of its values (ActQuant, as
    the model does; round half to even, as torch's round)."""
    wide = {name: np.asarray(tensor, dtype=np.float64) for name, tensor in tensors.items() if tensor.dtype != np.uint8}
    eps, heads, kv_heads = config["rms_norm_eps"], config["num_attention_heads"], config["num_key_value_heads"]
    size = config["head_dim"]
    norm = lambda v, weight: weight * (v / math.sqrt(float(v @ v) / len(v) + eps))

    def unpacked(packed):
        quarter = packed.shape[0]
        out = np.zeros((quarter * 4, packed.shape[1]))
        for i in range(4):
            out[i * quarter:(i + 1) * quarter] = ((packed >> (2 * i)) & 3).astype(np.float64) - 1.0
        return out

    def linear(name, v):
        if quantized:
            scale = 127.0 / max(float(np.abs(v).max()), 1e-5)
            v = np.clip(np.round(v * scale), -128, 127) / scale
        return (unpacked(tensors[name + ".weight"]) @ v) * float(tensors[name + ".weight_scale"][0])

    def rotate(head, pos):
        inverse = 1.0 / config["rope_theta"] ** (np.arange(0, size, 2, dtype=np.float64) / size)
        cos, sin = np.cos(pos * inverse), np.sin(pos * inverse)
        cos, sin = np.concatenate([cos, cos]), np.concatenate([sin, sin])
        return head * cos + np.concatenate([-head[size // 2:], head[:size // 2]]) * sin

    x = [wide["model.embed_tokens.weight"][token] for token in tokens]
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        normed = [norm(v, wide[p + "input_layernorm.weight"]) for v in x]
        queries = [[rotate(q, pos) for q in linear(p + "self_attn.q_proj", v).reshape(heads, size)] for pos, v in enumerate(normed)]
        ks = [[rotate(k, pos) for k in linear(p + "self_attn.k_proj", v).reshape(kv_heads, size)] for pos, v in enumerate(normed)]
        vs = [linear(p + "self_attn.v_proj", v).reshape(kv_heads, size) for v in normed]
        for pos in range(len(tokens)):
            attended = np.zeros((heads, size))
            for h in range(heads):
                kv = h // (heads // kv_heads)
                scores = np.array([queries[pos][h] @ ks[t][kv] / math.sqrt(size) for t in range(pos + 1)])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                attended[h] = sum(score * vs[t][kv] for t, score in enumerate(scores))
            mixed = norm(attended.reshape(-1), wide[p + "self_attn.attn_sub_norm.weight"])
            x[pos] = x[pos] + linear(p + "self_attn.o_proj", mixed)
        for pos in range(len(tokens)):
            v = norm(x[pos], wide[p + "post_attention_layernorm.weight"])
            gated = np.maximum(linear(p + "mlp.gate_proj", v), 0.0) ** 2 * linear(p + "mlp.up_proj", v)
            x[pos] = x[pos] + linear(p + "mlp.down_proj", norm(gated, wide[p + "mlp.ffn_sub_norm.weight"]))
    classifier = wide["lm_head.weight"] if "lm_head.weight" in wide else wide["model.embed_tokens.weight"]
    return [classifier @ norm(v, wide["model.norm.weight"]) for v in x]


def conversion(tensors, config, dtype="float32", stored="F32"):
    """The page's way: the file in its own order, with a tokenizer.json, and the options that come out."""
    file = safetensors_file(tensors, stored)
    size = struct.unpack("<Q", file[:8])[0]
    made = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), unigram(config["vocab_size"]),
                                     "tokenizer.json", dtype=dtype, max_seq_len=config["max_position_embeddings"], start=8 + size)
    made.stream.feed(file[8 + size:])
    made.stream.finish()
    return made


def engine(made):
    return Llama(bytes(made.checkpoint), bytes(made.tokenizer), **made.options)


def close(got, want, tolerance=2e-4):
    return np.allclose(got, want, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("model", MODELS)
def test_a_bitnet_converts_and_runs_as_transformers_computes_it(model):
    tensors, _, config = bitnet_model(**MODELS[model])
    made = conversion(tensors, config)
    assert made.options["arch"] == "bitnet" and set(made.options["kinds"]) == {"wq", "wk", "wv", "wo", "w1", "w2", "w3"}
    assert set(made.options["kinds"].values()) == {"ternary"}
    assert checkpoint_size(made.stream.header, "float32", made.options) == len(made.stream.out)
    llama = engine(made)
    want = naive_bitnet_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos


def test_the_page_and_the_build_convert_alike():
    """The file in its own order (the page's Stream) and read where each tensor is (the build's convert_weights())."""
    tensors, _, config = bitnet_model(shared=False)
    made = conversion(tensors, config)
    built = converted(Safetensors(reader(safetensors_file(tensors))), llama2_convert.normalize(config), "float32",
                      config["max_position_embeddings"])
    assert bytes(made.checkpoint) == built


def test_the_matrices_are_kept_ternary_and_the_embedding_follows_the_file():
    """The matrices are ternary in a file of any dtype (their values are -1, 0 and 1 times one scale), the embedding,
    which is the classifier too, the file's: a file of int8 holds the model in a little more than the ternary
    matrices and an int8 table, and runs as the reference does but for the rounding of that table and of what its
    matrices read (by groups of 32: 0.9% of the largest logit here, where the model's own rounding by the token's
    largest value moves them by 2%)."""
    tensors, _, config = bitnet_model()
    floats, eights = conversion(tensors, config), conversion(tensors, config, "int8")
    table = config["vocab_size"] * config["hidden_size"]
    # (and an int8 file has no RoPE tables: the engine makes them)
    tables = config["max_position_embeddings"] * config["head_dim"] // 2 * 2 * 4
    assert len(floats.checkpoint) - len(eights.checkpoint) == table * 4 - (table + table // 32 * 4) + tables
    assert checkpoint_dtype(eights.stream.header, len(eights.checkpoint), eights.options) == "int8"
    want = naive_bitnet_logits(tensors, config, TOKENS)
    llama = engine(eights)
    for pos, token in enumerate(TOKENS):
        assert np.abs(llama.forward(token, pos) - want[pos]).max() <= 0.015 * np.abs(want[pos]).max(), pos


def test_the_file_may_hold_its_floats_as_bfloat16():
    """microsoft/bitnet-b1.58-2B-4T's are: the embedding, the norms and the scales."""
    tensors, _, config = bitnet_model()
    halves = {name: tensor if tensor.dtype == np.uint8 else
              (tensor.view(np.uint32) >> 16 << 16).astype(np.uint32).view(np.float32) for name, tensor in tensors.items()}
    made = conversion(tensors, config, stored="BF16")
    want = naive_bitnet_logits(halves, config, TOKENS)
    llama = engine(made)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos


# every tensor of a layer: changed in one layer, the engine must follow the reference to other numbers, or the tensor
# is not read where transformers reads it (a scale of another matrix, a norm of another layer)
KINDS = [*(name + ".weight_scale" for name in BITNET_MATRICES), "self_attn.attn_sub_norm.weight", "mlp.ffn_sub_norm.weight",
         "input_layernorm.weight", "post_attention_layernorm.weight"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which", [0, -1])
def test_every_tensor_is_read_where_transformers_reads_it(kind, which):
    tensors, _, config = bitnet_model(layers=3)
    name = f"model.layers.{range(3)[which]}.{kind}"
    # (a scale of what a norm reads next, v's, the gate's and the up's, shows only through the norm's epsilon: one
    # that makes what it multiplies about as small as that)
    times = np.float32(1 / 128) if kind.endswith("scale") else np.float32(1.0)
    changed = {**tensors, name: (tensors[name] * times + np.float32(0 if kind.endswith("scale") else 0.3) *
                                 np.sin(np.arange(tensors[name].size, dtype=np.float32))).astype(np.float32)}
    want, before = naive_bitnet_logits(changed, config, TOKENS), naive_bitnet_logits(tensors, config, TOKENS)
    assert not close(want[-1], before[-1])
    llama = engine(conversion(changed, config))
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), (name, pos)


@pytest.mark.parametrize("matrix", BITNET_MATRICES)
def test_every_quarter_of_a_matrix_is_unpacked_to_its_rows(matrix):
    """A quarter of a matrix's rows (one pair of bits of every byte) changed alone: an engine that took the quarters in
    another order, or the bits of a byte the other way round, would not follow the reference."""
    tensors, codes, config = bitnet_model()
    for quarter in range(4):
        values, scale = codes["model.layers.1." + matrix]
        values = values.copy()
        rows = values.shape[0] // 4
        values[quarter * rows:(quarter + 1) * rows] = -values[quarter * rows:(quarter + 1) * rows][::-1]
        changed = {**tensors, f"model.layers.1.{matrix}.weight": bitnet_pack(values)}
        want = naive_bitnet_logits(changed, config, TOKENS)
        assert not close(want[-1], naive_bitnet_logits(tensors, config, TOKENS)[-1])
        llama = engine(conversion(changed, config))
        for pos, token in enumerate(TOKENS):
            assert close(llama.forward(token, pos), want[pos]), (matrix, quarter, pos)


def test_the_planes_are_transformers_own():
    """unpack_weights()'s example in transformers' integrations/bitnet.py, its packed bytes and the values it says."""
    packed = np.array([[0b10100001, 0b00011000], [0b10010000, 0b00001010]], dtype=np.uint8)
    assert planes_of(packed, ("planes",), 0).tolist() == [[0, -1], [-1, 1], [-1, 1], [-1, 1], [1, 0], [0, -1], [1, -1], [1, -1]]
    # and bitnet_pack() above is its inverse, so the made-up files are what transformers would pack
    values = np.random.default_rng(3).integers(-1, 2, size=(16, 5)).astype(np.int8)
    assert planes_of(bitnet_pack(values), ("planes",), 0).tolist() == values.tolist()
    # ("by", name) is the number of another tensor, put in as it is read; unread, it is refused
    assert transformed(np.ones(3, dtype=np.float32), resolved(("by", "s"), {"s": 0.5}.get), 0).tolist() == [0.5] * 3
    with pytest.raises(ValueError, match="multiplies by s, which it has not read"):
        transformed(np.ones(3, dtype=np.float32), ("by", "s"), 0)


def test_the_quantized_activations_are_what_the_engine_leaves_out():
    """The model's own rounding of every matrix's input to 8 bits by the token's largest value moves the logits; the
    engine, which keeps them float (and its kernels round by groups of 32), is held to the reference without it. What
    the rounding moves here is far past the line the engine is held to: the two cannot be mistaken."""
    tensors, _, config = bitnet_model()
    rounded, floats = naive_bitnet_logits(tensors, config, TOKENS, quantized=True), naive_bitnet_logits(tensors, config, TOKENS)
    assert max(float(np.abs(a - b).max()) for a, b in zip(rounded, floats)) > 1e-2


# ---- what the converter refuses
def refused(config):
    with pytest.raises(ValueError) as caught:
        check_config(llama2_convert.normalize(config))
    return str(caught.value)


@pytest.mark.parametrize("change, said", [
    (dict(hidden_act="silu"), "its activation is silu, not relu2"),
    (dict(attention_bias=True), "its layers have biases"),
    (dict(quantization_config={"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "online"}), "not the ternary ones"),
    (dict(quantization_config={"quant_method": "bitnet", "linear_class": "bitlinear", "quantization_mode": "offline"}), "not the ternary ones"),
    (dict(quantization_config={"quant_method": "bitnet", "linear_class": "autobitlinear", "use_rms_norm": True}), "not the ternary ones"),
    (dict(quantization_config=None), "not the ternary ones"),
])
def test_what_the_engine_has_not_is_refused(change, said):
    _, _, config = bitnet_model()
    assert said in refused({**config, **change})


def test_a_scale_that_is_missing_or_no_number_or_late_is_refused():
    tensors, _, config = bitnet_model()
    name = "model.layers.1.mlp.up_proj.weight_scale"
    with pytest.raises(ValueError, match=f"{name} is missing, not one number"):
        conversion({key: value for key, value in tensors.items() if key != name}, config)
    with pytest.raises(ValueError, match=rf"{name} is \(2,\), not one number"):
        conversion({**tensors, name: np.array([0.5, 0.5], dtype=np.float32)}, config)
    # (the page reads the file in its order: a scale after its matrix is refused, not taken as 0 or as another's)
    late = {key: value for key, value in tensors.items() if key != name}
    late[name] = tensors[name]
    with pytest.raises(ValueError, match=f"{name} comes after the tensor it multiplies"):
        conversion(late, config)
    # (the build's reads it wherever it is)
    built = converted(Safetensors(reader(safetensors_file(late))), llama2_convert.normalize(config), "float32",
                      config["max_position_embeddings"])
    assert built == bytes(conversion(tensors, config).checkpoint)
