# GGUF as a source (T74): a GGUF written here the way llama.cpp writes one (Q8_0 matrices, F32 vectors, q and k of
# a Llama turned into llama2.c's order, a Qwen2's left alone) must give the very checkpoint that the safetensors of
# the same values gives. The real file is checked by tests/gguf_check.py.
import struct

import numpy as np
import pytest
from conftest import synthetic_weights
from test_bias import qwen2
from test_convert import converted, hugging_face, reader, safetensors_file

import llama2_convert
from llama2_convert import Conversion, Incomplete, Safetensors, gguf_read

NAMES = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
         "lm_head.weight": "output.weight"}
LAYER = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q",
         "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


def q8_0_blocks(values):
    """llama.cpp's Q8_0: per 32 values a float16 scale (largest / 127) and the int8 values."""
    groups = values.reshape(-1, 32).astype(np.float32)
    scales = (np.abs(groups).max(axis=1) / 127.0).astype(np.float16)
    inverse = np.divide(1.0, scales.astype(np.float32), out=np.zeros(len(scales), np.float32), where=scales > 0)
    ints = np.clip(np.rint(groups * inverse[:, None]), -127, 127).astype(np.int8)
    return np.concatenate([scales.view(np.uint8).reshape(-1, 2), ints.view(np.uint8)], axis=1).tobytes(), \
        (ints.astype(np.float32) * scales.astype(np.float32)[:, None]).reshape(values.shape)


def turn(w, heads):
    """What llama.cpp's convert does to q and k of a Llama."""
    rows = w.shape[0] // heads
    return w.reshape(heads, 2, rows // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def gguf_name(name):
    if name in NAMES:
        return NAMES[name]
    _, _, layer, *rest = name.split(".")
    return f"blk.{layer}.{LAYER['.'.join(rest[:-1])]}.{rest[-1]}"


def gguf_file(tensors, published, vocab_size, arch="llama", pre="gpt-2", extra=()):
    """A GGUF v3 of these Hugging Face tensors, and the tensors as the GGUF holds them (Q8_0 rounds). extra: more
    metadata, as (key, type, value) with the types 4 (uint32) and 6 (float32)."""
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    heads = {"q_proj": published["num_attention_heads"], "k_proj": published["num_key_value_heads"]}
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, published["num_hidden_layers"]),
                (f"{arch}.context_length", 4, published["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, published["hidden_size"]),
                (f"{arch}.feed_forward_length", 4, published["intermediate_size"]),
                (f"{arch}.attention.head_count", 4, published["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 4, published["num_key_value_heads"]),
                (f"{arch}.rope.freq_base", 6, 10000.0), ("tokenizer.ggml.model", 8, "gpt2"),
                ("tokenizer.ggml.pre", 8, pre), ("tokenizer.ggml.bos_token_id", 4, 1),
                ("tokenizer.ggml.eos_token_id", 4, 2), *extra]
    tokens = [f"w{i}" for i in range(vocab_size)]
    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(metadata) + 3)]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind))
        out.append(string(value) if kind == 8 else struct.pack({4: "<I", 6: "<f"}[kind], value))
    out.append(string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens)) + b"".join(map(string, tokens)))
    out.append(string("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, len(tokens)) + struct.pack(f"<{len(tokens)}i", *[1] * len(tokens)))
    out.append(string("tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, 0))
    held, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        stored = tensor
        kind = next((k for k in heads if f".{k}." in name), None)
        if arch == "llama" and kind:
            stored = turn(tensor, heads[kind])
        if tensor.ndim == 2:
            blob, rounded = q8_0_blocks(stored)
            held[name] = rounded  # as the GGUF holds it (turned, for q and k of a Llama)
            type_ = 8
        else:
            blob, type_ = stored.astype(np.float32).tobytes(), 0
            held[name] = stored.astype(np.float32)
        out.append(string(gguf_name(name)) + struct.pack("<I", tensor.ndim) + struct.pack(f"<{tensor.ndim}Q", *reversed(tensor.shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    # the values the GGUF stands for, back in Hugging Face's order: what a safetensors of the same model holds
    same = {}
    for name, values in held.items():
        kind = next((k for k in heads if f".{k}." in name), None)
        same[name] = llama2_convert.unturned(values, heads[kind]) if arch == "llama" and kind else values
    return head + b"".join(blobs), same


def fed(file, dtype, chunk=4096):
    for size in (100, 1000, len(file)):
        try:
            conversion = Conversion.from_gguf(file[:size], dtype=dtype, max_seq_len=1 << 20)
            break
        except Incomplete:
            continue
    for start in range(conversion.base, len(file), chunk):
        conversion.feed(file[start:start + chunk])
    conversion.finish()
    return conversion


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("model", ["llama", "llama-own-classifier", "qwen2"])
def test_a_gguf_converts_to_the_checkpoint_of_the_same_values(model, dtype):
    config, weights = synthetic_weights(n_kv_heads=2, shared=model != "llama-own-classifier")
    tensors, published = (qwen2 if model == "qwen2" else hugging_face)(config, weights, model != "llama-own-classifier")
    file, same = gguf_file(tensors, published, config["vocab_size"], "qwen2" if model == "qwen2" else "llama")
    conversion = fed(file, dtype)
    expected = converted(Safetensors(reader(safetensors_file(same))), published, dtype)
    assert bytes(conversion.checkpoint) == expected
    assert conversion.options["arch"] == "llama" and conversion.options["bias"] is (model == "qwen2")
    assert conversion.options["tokenizer_kind"] == "bytebpe" and conversion.options["pretokenizer"] == "gpt2"


@pytest.mark.parametrize("head_size", [0, 16])
def test_a_gguf_says_the_epsilon_and_the_size_of_a_head(head_size):
    """T144 (the review of T124): llama.cpp writes rms_norm_eps as attention.layer_norm_rms_epsilon and head_dim as
    attention.key_length. Unread, the GGUFs of the list (Qwen2.5, TinySwallow: 1e-6) would go back to 1e-5 without a
    word, and heads of another size than dim / heads would be laid out as dim / heads. A key_length of dim / heads
    (every GGUF of the list) is no head_dim in the options: they stay what they were."""
    config, weights = synthetic_weights(n_kv_heads=2, head_size=head_size)
    tensors, published = hugging_face(config, weights, True)
    extra = [("llama.attention.layer_norm_rms_epsilon", 6, 1e-6), ("llama.attention.key_length", 4, config["head_size"])]
    file, same = gguf_file(tensors, published, config["vocab_size"], extra=extra)
    conversion = fed(file, "float32")
    assert conversion.options["rms_norm_eps"] == 1e-6
    assert conversion.options.get("head_dim") == (head_size or None)
    assert bytes(conversion.checkpoint) == converted(Safetensors(reader(safetensors_file(same))), published, "float32")


def test_q8_0_comes_back_as_the_same_int8():
    values = (np.random.default_rng(3).standard_normal((4, 64)) * 0.2).astype(np.float32)
    blob, rounded = q8_0_blocks(values)
    back = llama2_convert.q8_0(blob).reshape(values.shape)
    assert np.array_equal(back, rounded)
    ints, scales = llama2_convert.quantize(back)
    assert np.array_equal(ints.astype(np.float32) * scales[:, None], rounded.reshape(-1, 32)), "lossless"


def test_a_short_head_asks_for_more_and_other_files_are_refused():
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    file, _ = gguf_file(tensors, published, config["vocab_size"])
    with pytest.raises(Incomplete):
        gguf_read(file[:200])
    with pytest.raises(ValueError, match="not a GGUF"):
        gguf_read(b"PK\x03\x04" + file[4:])
    k_quant = bytearray(file)
    metadata, found, base = gguf_read(file)
    # make the first matrix a Q4_K (type 12): its type is the u32 before its u64 offset in the tensor infos
    at = file.index(b"token_embd.weight") + len(b"token_embd.weight") + 4 + 16
    k_quant[at:at + 4] = struct.pack("<I", 12)
    with pytest.raises(ValueError, match="K-quants"):
        Conversion.from_gguf(bytes(k_quant))


def test_a_q8_0_tensor_whose_rows_are_not_groups_of_32_is_refused():
    """Fable's review of T74: the Q8_0 reader counts 34 bytes per 32 values, so a row that is not a multiple of 32
    would be read at the wrong offsets and write nonsense instead of failing."""
    config, weights = synthetic_weights(dim=48, hidden_dim=64, n_heads=2, n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    file, _ = gguf_file(tensors, published, config["vocab_size"])
    with pytest.raises(ValueError, match="multiple of 32"):
        Conversion.from_gguf(file)
