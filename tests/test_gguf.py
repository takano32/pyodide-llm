# GGUF as a source (T74): a GGUF written here the way llama.cpp writes one (Q8_0 matrices, F32 vectors, q and k of
# a Llama turned into llama2.c's order, a Qwen2's left alone) must give the very checkpoint that the safetensors of
# the same values gives. The real file is checked by tests/gguf_check.py.
import json
import struct

import numpy as np
import pytest
from conftest import synthetic_weights
from test_bias import qwen2
from test_convert import converted, hugging_face, reader, safetensors_file

import llama2_convert
from llama2_convert import Conversion, Incomplete, Safetensors, gguf_read
from llama2_numpy import rope_frequencies

NAMES = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
         "lm_head.weight": "output.weight"}
LAYER = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q",
         "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
         "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm"}  # a Qwen3's (T203)


def q8_0_blocks(values):
    """llama.cpp's Q8_0 (ggml-quants.c's quantize_row_q8_0_ref): per 32 values d = largest / 127 in float32, the
    values times 1 / d rounded half away from zero, and d kept as float16; a d under float16's smallest half step
    is kept as 0, and its values read back 0. T145: as llama.cpp rounds, to the bit (it rounded by the float16 d
    before), so that tests/gguf_check.py's nearest reference is 0 off for a GGUF of the original's values."""
    groups = values.reshape(-1, 32).astype(np.float32)
    d = np.abs(groups).max(axis=1) / np.float32(127)
    scales = d.astype(np.float16)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inverse = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d > 0)
        scaled = groups * inverse[:, None]
        ints = np.where(scales[:, None] > 0, np.sign(scaled) * np.floor(np.abs(scaled) + 0.5), 0).astype(np.int8)
    return np.concatenate([scales.view(np.uint8).reshape(-1, 2), ints.view(np.uint8)], axis=1).tobytes(), \
        (ints.astype(np.float32) * scales.astype(np.float32)[:, None]).reshape(values.shape)


def pq2_0_blocks(values):
    """Prism ML's PQ2_0 (T235) as its fork of llama.cpp writes it (ggml-quants.c's quantize_row_pq2_0_ref): per 128
    values d = the largest, the values over d rounded to -1, 0 or 1 and kept as that plus 1 in two bits, the first value
    in the lowest bits of a byte, after d as float16. And what the block then stands for: (code - 1) * d."""
    groups = values.reshape(-1, 128).astype(np.float32)
    d = np.abs(groups).max(axis=1).astype(np.float16)
    with np.errstate(divide="ignore", invalid="ignore"):
        codes = np.where(d[:, None] > 0, np.rint(groups / d.astype(np.float32)[:, None]), 0).astype(np.int64) + 1
    packed = (codes.reshape(-1, 32, 4) << (0, 2, 4, 6)).sum(axis=2).astype(np.uint8)
    return np.concatenate([d.view(np.uint8).reshape(-1, 2), packed], axis=1).tobytes(), \
        ((codes - 1).astype(np.float32) * d.astype(np.float32)[:, None]).reshape(values.shape)


def metadata_value(kind, value):
    """A GGUF metadata value of a ggml kind: 4 a uint32, 5 an int32, 6 a float32, 7 a bool, 8 a string, and 9 an array,
    given as (the kind of its items, the items)."""
    if kind == 8:
        return struct.pack("<Q", len(value.encode())) + value.encode()
    if kind == 9:
        item, values = value
        return struct.pack("<IQ", item, len(values)) + b"".join(metadata_value(item, one) for one in values)
    return struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?"}[kind], value)


def turn(w, heads):
    """What llama.cpp's convert does to q and k of a Llama."""
    rows = w.shape[0] // heads
    return w.reshape(heads, 2, rows // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def gguf_name(name):
    if name in NAMES:
        return NAMES[name]
    _, _, layer, *rest = name.split(".")
    return f"blk.{layer}.{LAYER['.'.join(rest[:-1])]}.{rest[-1]}"


def gguf_file(tensors, published, vocab_size, arch="llama", pre="gpt-2", theta=10000.0, more=(), extra=None, bos=1, eos=2,
              matrices=(q8_0_blocks, 8)):
    """A GGUF v3 of these Hugging Face tensors, and the tensors as the GGUF holds them (Q8_0 rounds).
    more: further metadata (key, GGUF type, value); extra: {GGUF name: float32 values} written as they are
    (rope_freqs.weight). bos, eos: None leaves the token out (unsloth's Qwen3 GGUFs name no BOS). matrices: what makes
    the blocks of a matrix and their ggml type (T235: pq2_0_blocks and 142)."""
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    heads = {"q_proj": published["num_attention_heads"], "k_proj": published["num_key_value_heads"]}
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, published["num_hidden_layers"]),
                (f"{arch}.context_length", 4, published["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, published["hidden_size"]),
                (f"{arch}.feed_forward_length", 4, published["intermediate_size"]),
                (f"{arch}.attention.head_count", 4, published["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 4, published["num_key_value_heads"]),
                (f"{arch}.rope.freq_base", 6, theta), ("tokenizer.ggml.model", 8, "gpt2"),
                ("tokenizer.ggml.pre", 8, pre),
                *[(f"tokenizer.ggml.{key}_token_id", 4, id) for key, id in (("bos", bos), ("eos", eos)) if id is not None],
                *more]
    tokens = [f"w{i}" for i in range(vocab_size)]
    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors) + len(extra or {}), len(metadata) + 3)]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind))
        out.append(metadata_value(kind, value))
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
            blob, rounded = matrices[0](stored)
            held[name] = rounded  # as the GGUF holds it (turned, for q and k of a Llama)
            type_ = matrices[1]
        else:
            blob, type_ = stored.astype(np.float32).tobytes(), 0
            held[name] = stored.astype(np.float32)
        out.append(string(gguf_name(name)) + struct.pack("<I", tensor.ndim) + struct.pack(f"<{tensor.ndim}Q", *reversed(tensor.shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    # after every other tensor, as Swallow 8B's GGUF has its rope_freqs
    for name, values in (extra or {}).items():
        blob = np.asarray(values, np.float32).tobytes()
        out.append(string(name) + struct.pack("<I", 1) + struct.pack("<Q", len(values)) + struct.pack("<IQ", 0, offset))
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
    file, same = gguf_file(tensors, published, config["vocab_size"], more=extra)
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


# ---- T136's second stage: the weights of a GGUF with the vocabulary and config.json of the original repository
def with_original(file, published, vocabulary, name, dtype, chunk=4096):
    """What worker.js does for such an item: the GGUF's header made a safetensors one, then the original's files."""
    header, base = llama2_convert.gguf_weights(file, json.dumps(published))
    conversion = Conversion(header, base, json.dumps(published), vocabulary, name, dtype=dtype, max_seq_len=1 << 20,
                            start=base)
    for start in range(base, len(file), chunk):
        conversion.feed(file[start:start + chunk])
    conversion.finish()
    return conversion


def sentencepiece(vocab_size):
    from make_hf_fixture import field
    NORMAL, UNKNOWN, CONTROL = 1, 2, 3
    pieces = [("<unk>", UNKNOWN), ("<s>", CONTROL), ("</s>", CONTROL)]
    pieces += [(f"▁w{i}", NORMAL) for i in range(vocab_size - len(pieces))]
    model = b"".join(field(1, field(1, text.encode()) + field(2, -float(i)) + field(3, kind)) for i, (text, kind) in enumerate(pieces))
    return model + field(2, field(3, 1)) + field(3, field(1, b"identity"))


EPS = ("llama.attention.layer_norm_rms_epsilon", 6, 1e-5)  # llama.cpp always writes it
LLAMA3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 64}


def llama3_table(published):
    width = llama2_convert.head_size(published)
    return rope_frequencies(width, 10000.0) / rope_frequencies(width, 10000.0, LLAMA3)


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("model", ["sentencepiece", "llama3"])
def test_a_gguf_with_the_originals_vocabulary_is_the_safetensors_conversion(model, dtype):
    """The same checkpoint, tokenizer.bin and options as the safetensors of the same values with the same files:
    the GGUF's own vocabulary and settings play no part, and a rope_freqs table is checked and dropped."""
    config, weights = synthetic_weights(n_kv_heads=2, shared=False)
    tensors, published = hugging_face(config, weights, False)
    published["rms_norm_eps"] = 1e-5
    table = None
    if model == "llama3":
        published["rope_scaling"] = LLAMA3
        table = llama3_table(published)
    file, same = gguf_file(tensors, published, config["vocab_size"], extra=None if table is None else {"rope_freqs.weight": table}, more=[EPS])
    vocabulary, name = (sentencepiece(config["vocab_size"]), "tokenizer.model") if model == "sentencepiece" else \
        (json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                     "vocab": [[f"w{i}", -float(i)] for i in range(config["vocab_size"])]}}).encode(), "tokenizer.json")
    got = with_original(file, published, vocabulary, name, dtype)
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, name,
                          dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert (got.options.get("rope_scaling") == LLAMA3) is (model == "llama3")


@pytest.mark.parametrize("change, what", [
    (dict(num_attention_heads=2, num_key_value_heads=2, head_dim=16), "number of heads"),
    (dict(num_key_value_heads=4), "number of key-value heads"),
    (dict(rope_theta=500000.0), "RoPE theta"),
    (dict(rms_norm_eps=1e-6), "RMSNorm epsilon"),
    (dict(model_type="qwen2"), "architecture"),
    (dict(tie_word_embeddings=False), "classifier of its own"),
    (dict(num_hidden_layers=1), "number of layers"),
])
def test_a_gguf_that_is_not_the_originals_is_refused(change, what):
    """What the sizes of the tensors do not show (T136's review, point 5): heads and key-value heads of the same
    product, fewer layers than the GGUF has (it went through, cut to them), a classifier that would silently be the
    embedding, and the numbers that are no tensor."""
    config, weights = synthetic_weights(n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    published["rms_norm_eps"] = 1e-5
    file, _ = gguf_file(tensors, published, config["vocab_size"], more=[EPS])
    llama2_convert.gguf_weights(file, json.dumps(published))  # its own config goes through
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(file, json.dumps({**published, **change}))


def test_a_rope_freqs_table_that_is_not_the_originals_scaling_is_refused():
    """A GGUF's rope_freqs is dropped for the engine's own tables (rope_frequencies of rope_scaling): only where the
    two agree. Here it comes after every other tensor, as in Swallow 8B's GGUF, and is refused before finish()."""
    config, weights = synthetic_weights(n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    published["rope_scaling"] = LLAMA3
    table = llama3_table(published)
    table[-1] *= 1.01
    file, _ = gguf_file(tensors, published, config["vocab_size"], extra={"rope_freqs.weight": table})
    vocabulary = sentencepiece(config["vocab_size"])
    with pytest.raises(ValueError, match="scales its RoPE otherwise"):
        with_original(file, published, vocabulary, "tokenizer.model", "int8")
    # without the original's config.json (?hf= of a GGUF) the table is refused as before
    with pytest.raises(ValueError, match="rope_freqs table"):
        Conversion.from_gguf(file)


# ---- T203 (T136's fourth stage): a Qwen3, whose GGUF holds q, k and the norms of their heads in Hugging Face's order
def qwen3_gguf(head_size, eps=1e-6, bos=1, eos=2):
    from test_qwen3 import qwen3
    config, weights = synthetic_weights(n_kv_heads=2, head_size=head_size)
    tensors, published = qwen3(config, weights, True)
    published["rms_norm_eps"] = eps
    # llama.cpp writes the size of a head as the length of a key, whatever it is, and always the epsilon
    more = [("qwen3.attention.key_length", 4, config["head_size"]), ("qwen3.attention.layer_norm_rms_epsilon", 6, eps)]
    file, same = gguf_file(tensors, published, config["vocab_size"], "qwen3", pre="qwen2", more=more, bos=bos, eos=eos)
    return config, published, file, same


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("head_size", [0, 16])
def test_a_qwen3_gguf_with_the_originals_files_is_the_safetensors_conversion(head_size, dtype):
    """The checkpoint, tokenizer.bin and options of the safetensors of the same values, the norms of q and k with them
    (qk_norm), and heads of another size than dim / heads (head_dim). Unread, the norms would leave a Qwen3 that runs
    as a Llama and writes nonsense without a word."""
    config, published, file, same = qwen3_gguf(head_size)
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, published, vocabulary, "tokenizer.json", dtype)
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary,
                          "tokenizer.json", dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert got.options["qk_norm"] is True and got.options.get("head_dim") == (head_size or None)


@pytest.mark.parametrize("head_size", [0, 16])
def test_a_qwen3_gguf_alone_converts_to_the_checkpoint_of_the_same_values(head_size):
    """?hf= of a Qwen3's GGUF alone (its own vocabulary and settings): the same checkpoint, and the options say the
    norms and the size of a head."""
    config, published, file, same = qwen3_gguf(head_size)
    conversion = fed(file, "int8")
    assert bytes(conversion.checkpoint) == converted(Safetensors(reader(safetensors_file(same))), published, "int8")
    assert conversion.options["qk_norm"] is True and conversion.options.get("head_dim") == (head_size or None)
    assert conversion.options["rms_norm_eps"] == pytest.approx(1e-6)


@pytest.mark.parametrize("token", ["bos", "eos"])
def test_a_gguf_alone_that_names_no_bos_or_eos_is_refused(token):
    """The review of T203: unsloth's Qwen3 GGUFs name no BOS (add_bos_token false, no bos_token_id). Alone, the
    conversion took token 1 for it and stopped at it: '"' of a byte-level BPE vocabulary, every answer cut at its first
    lone one. The list takes them with their original's config.json, whose BOS and EOS it has, as before."""
    config, published, file, same = qwen3_gguf(16, **{token: None})
    with pytest.raises(ValueError, match=f"names no {token.upper()} token"):
        fed(file, "int8")
    published = {**published, "bos_token_id": 5, "eos_token_id": 6}
    got = with_original(file, published, unigram(config["vocab_size"]), "tokenizer.json", "int8")
    assert got.options["bos"] == 5 and got.options["stop_tokens"] == [5, 6]


@pytest.mark.parametrize("change, what", [
    (dict(model_type="qwen2"), "architecture"),
    (dict(head_dim=8), "size of a head"),
    (dict(rms_norm_eps=1e-5), "RMSNorm epsilon"),
])
def test_a_qwen3_gguf_that_is_not_the_originals_is_refused(change, what):
    config, published, file, _ = qwen3_gguf(16)
    llama2_convert.gguf_weights(file, json.dumps(published))  # its own config goes through
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(file, json.dumps({**published, **change}))


# ---- T235: Prism ML's PQ2_0 (Ternary-Bonsai), ternary blocks of 128 that the engine's int8 holds as they are, and yarn
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192}


def test_pq2_0_reads_as_the_fork_does():
    """Every code (3, the +2 d no ternary file has, too) at every place of a byte, under several scales, against the
    fork's dequantize_row_pq2_0 written out value by value."""
    rng = np.random.default_rng(5)
    scales = np.array([0.0, 1.0, -0.5, 0.0123, 6e-8, 65504.0, 3.1e-5], dtype=np.float16)
    codes = rng.integers(0, 4, (len(scales), 128))
    codes[:, :16] = np.tile(np.arange(4), 4)  # each code at each of a byte's four places
    packed = np.zeros((len(scales), 32), dtype=np.uint8)
    for j in range(128):
        packed[:, j // 4] |= (codes[:, j] << (j % 4 * 2)).astype(np.uint8)
    raw = np.concatenate([scales.view(np.uint8).reshape(-1, 2), packed], axis=1).tobytes()
    expected = np.array([[(int(q) - 1) * np.float32(d) for q in row] for row, d in zip(codes, scales)], dtype=np.float32)
    assert np.array_equal(llama2_convert.pq2_0(raw).view(np.uint32), expected.reshape(-1).view(np.uint32))
    # and what the test's own writer makes of ternary values comes back as them
    values = rng.integers(-1, 2, (4, 256)).astype(np.float32) * np.float32(0.0625)
    blob, held = pq2_0_blocks(values)
    assert np.array_equal(held, values) and np.array_equal(llama2_convert.pq2_0(blob).reshape(values.shape), values)


def test_int8_holds_a_ternary_block_under_every_scale():
    """What "without loss" is, for all 63488 finite float16 scales (the subnormal ones and the negative ones too): the
    int8 values are exactly 127 times the ternary ones, in every group of 32, and the scale of a group is float32(d) /
    127 rounded to float32. So the engine multiplies 127 * float32(d / 127) where the file says d: not d to the bit,
    but within 6e-8 of it (float32's rounding of the quotient). A scale of 0 is a block of zeros."""
    d = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    d = d[np.isfinite(d)]
    assert d.size == 63488
    ternary = np.tile(np.array([-1, 0, 1, 1, 0, -1, 0, 1], dtype=np.int64), 16)  # all three in every group of 32
    packed = ((ternary + 1).reshape(32, 4) << (0, 2, 4, 6)).sum(axis=1).astype(np.uint8)
    raw = np.concatenate([d.view(np.uint8).reshape(-1, 2), np.tile(packed, (d.size, 1))], axis=1).tobytes()
    values = llama2_convert.pq2_0(raw).reshape(-1, 128)
    wide = d.astype(np.float32)
    assert np.array_equal(values, ternary.astype(np.float32) * wide[:, None])
    ints, scales = llama2_convert.quantize(values)
    sign = np.sign(wide).astype(np.int64)[:, None]
    assert np.array_equal(ints.reshape(-1, 128), 127 * ternary * sign)
    assert np.array_equal(scales.reshape(-1, 4), np.repeat((np.abs(wide) / np.float32(127.0))[:, None], 4, axis=1))
    some = wide != 0
    stands_for = 127.0 * scales.reshape(-1, 4)[some].astype(np.float64)
    worst = np.max(np.abs(stands_for - np.abs(wide[some]).astype(np.float64)[:, None]) / np.abs(wide[some])[:, None])
    assert 0 < worst < 6e-8, worst
    # six bits (T98, where int8 does not fit): the same, 31 times the ternary values (held as multiples of 4)
    sixes, _ = llama2_convert.quantize6(values[some])
    assert np.array_equal(sixes.reshape(-1, 128), 4 * 31 * ternary * sign[some])


def bonsai_gguf(yarn=YARN, head_size=0, bos=1, eos=2, also=(), shared=True, theta=1000000.0):
    """A small Qwen3 as Ternary-Bonsai's GGUF has one: PQ2_0 matrices (rows of 128 and 256), F32 norms, and yarn.
    also: further metadata. shared: the classifier is the embedding (the 1.7B and the 4B), or a PQ2_0 matrix of its own
    (the 8B)."""
    from test_qwen3 import qwen3
    config, weights = synthetic_weights(dim=128, hidden_dim=256, n_kv_heads=2, vocab_size=40, head_size=head_size, shared=shared)
    tensors, published = qwen3(config, weights, shared)
    published = {**published, "rms_norm_eps": 1e-6, "rope_theta": theta, "max_position_embeddings": 32768}
    more = [("qwen3.attention.key_length", 4, config["head_size"]), ("qwen3.attention.layer_norm_rms_epsilon", 6, 1e-6),
            *also]
    if yarn:
        published["rope_scaling"] = yarn
        more += [("qwen3.rope.scaling.type", 8, "yarn"), ("qwen3.rope.scaling.factor", 6, yarn["factor"]),
                 ("qwen3.rope.scaling.original_context_length", 4, yarn["original_max_position_embeddings"])]
    file, same = gguf_file(tensors, published, config["vocab_size"], "qwen3", pre="qwen2", theta=theta, more=more,
                           bos=bos, eos=eos, matrices=(pq2_0_blocks, 142))
    return config, published, file, same


# the three sizes of the list, by what differs between them (the review of T246: only the 1.7B's shape was tried; the 4B's
# heads are wider than dim / heads and its theta is 5e6, the 8B has a classifier of its own and a yarn from 16384)
BONSAI_SIZES = {"the 1.7B's": dict(),
                "the 4B's: heads wider than dim / heads, theta 5e6": dict(head_size=64, theta=5e6),
                "the 8B's: a classifier of its own, yarn from 16384":
                    dict(shared=False, yarn={**YARN, "original_max_position_embeddings": 16384})}


@pytest.mark.parametrize("dtype", ["int8", "float32", "int6"])
@pytest.mark.parametrize("shape", BONSAI_SIZES)
def test_a_pq2_0_gguf_with_the_originals_files_is_the_safetensors_conversion(dtype, shape):
    """The list's way in (the GGUF's weights, the original's config.json and vocabulary): the checkpoint, tokenizer.bin
    and options of a safetensors file of the values the blocks stand for, yarn in the options for the engine's tables.
    Fed 4096 bytes at a time: a chunk ends within a block and within a row."""
    config, published, file, same = bonsai_gguf(**BONSAI_SIZES[shape])
    metadata, found, _ = gguf_read(file)
    assert {info["type"] for name, info in found.items() if len(info["shape"]) == 2} == {142}
    assert ("output.weight" in found) == ("lm_head.weight" in same) == (not config["shared"])
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, published, vocabulary, "tokenizer.json", dtype)
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary,
                          "tokenizer.json", dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert got.options["qk_norm"] is True and got.options["rope_scaling"] == published["rope_scaling"]
    assert got.options["rope_theta"] == published["rope_theta"] and got.options.get("head_dim") == (64 if "heads wider" in shape else None)


def test_a_pq2_0_gguf_alone_converts_to_the_checkpoint_of_the_same_values():
    """?hf= of such a GGUF alone, were it to name a BOS (Ternary-Bonsai's names none and is refused, as T203 has it):
    yarn's numbers come from the GGUF's own metadata, by config.json's names."""
    config, published, file, same = bonsai_gguf(head_size=64)  # heads of another size than dim / heads: o's rows are 256
    conversion = fed(file, "int8")
    assert bytes(conversion.checkpoint) == converted(Safetensors(reader(safetensors_file(same))), published, "int8")
    assert conversion.options["rope_scaling"] == {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192}
    with pytest.raises(ValueError, match="names no BOS token"):
        fed(bonsai_gguf(bos=None)[2], "int8")


@pytest.mark.parametrize("theirs", [None, {**YARN, "factor": 2.0}, {**YARN, "original_max_position_embeddings": 4096},
                                    {"rope_type": "linear", "factor": 4.0}])
def test_a_gguf_whose_yarn_is_not_the_originals_is_refused(theirs):
    """yarn changes every angle and is no tensor: a GGUF and a config.json that differ in it are not one model. Either
    way round: a GGUF without yarn against a config.json with it too."""
    config, published, file, _ = bonsai_gguf()
    llama2_convert.gguf_weights(file, json.dumps(published))  # its own config goes through
    other = {key: value for key, value in published.items() if key != "rope_scaling"} | ({"rope_scaling": theirs} if theirs else {})
    with pytest.raises(ValueError, match="yarn RoPE scaling"):
        llama2_convert.gguf_weights(file, json.dumps(other))
    if theirs is None:
        plain = bonsai_gguf(yarn=None)[2]
        with pytest.raises(ValueError, match="yarn RoPE scaling"):
            llama2_convert.gguf_weights(plain, json.dumps(published))


@pytest.mark.parametrize("key, name", [("attn_factor", "attention_factor"), ("yarn_log_multiplier", "mscale_all_dim")])
def test_a_gguf_whose_yarn_says_more_than_the_tables_know_is_refused(key, name):
    """llama.cpp scales the turned values by a GGUF's attn_factor, or by its yarn_log_multiplier's: read under
    config.json's names, so that neither is dropped without a word, alone or with an original that does not say it.
    The keys are spelled as the fork's src/llama-arch.cpp spells them (the review of T235: the first version of this
    test and of the converter said yarn_log_mul, which no GGUF has, and so tested nothing real)."""
    config, published, file, _ = bonsai_gguf(also=[(f"qwen3.rope.scaling.{key}", 6, 0.5)])
    with pytest.raises(ValueError, match=f"yarn RoPE scaling sets {name}"):
        fed(file, "int8")
    with pytest.raises(ValueError, match="yarn RoPE scaling"):
        llama2_convert.gguf_weights(file, json.dumps(published))


def test_a_pq2_0_tensor_whose_rows_are_not_blocks_of_128_is_refused():
    """As a Q8_0's of 32: rows of 96 would be read at the wrong offsets and write nonsense instead of failing."""
    config, weights = synthetic_weights(dim=96, hidden_dim=128, n_heads=2, n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    file, _ = gguf_file(tensors, published, config["vocab_size"])
    wrong = bytearray(file)
    at = file.index(b"token_embd.weight") + len(b"token_embd.weight") + 4 + 16
    wrong[at:at + 4] = struct.pack("<I", 142)
    with pytest.raises(ValueError, match="PQ2_0 with rows of 96, which is not a multiple of 128"):
        Conversion.from_gguf(bytes(wrong))


# ---- T136's third stage: GPT-2 and GPT-NeoX, as llama.cpp's convert_hf_to_gguf.py writes them
GPT2_NAMES = {"wte.weight": "token_embd.weight", "wpe.weight": "position_embd.weight", "ln_f.weight": "output_norm.weight",
              "ln_f.bias": "output_norm.bias", "ln_1": "attn_norm", "attn.c_attn": "attn_qkv", "attn.c_proj": "attn_output",
              "ln_2": "ffn_norm", "mlp.c_fc": "ffn_up", "mlp.c_proj": "ffn_down"}
NEOX_NAMES = {"gpt_neox.embed_in.weight": "token_embd.weight", "embed_out.weight": "output.weight",
              "gpt_neox.final_layer_norm.weight": "output_norm.weight", "gpt_neox.final_layer_norm.bias": "output_norm.bias",
              "input_layernorm": "attn_norm", "attention.query_key_value": "attn_qkv", "attention.dense": "attn_output",
              "post_attention_layernorm": "ffn_norm", "mlp.dense_h_to_4h": "ffn_up", "mlp.dense_4h_to_h": "ffn_down"}
CONV1D = ("attn_qkv", "attn_output", "ffn_up", "ffn_down")  # GPT-2's matrices, which llama.cpp stores as (out, in)


def other_gguf(tensors, config):
    """A GGUF v3 of a GPT-2's or GPT-NeoX's Hugging Face tensors the way llama.cpp writes one (checked on the real
    files by Range, T136's third stage): GPT-2's Conv1D matrices turned to (out, in) and a copy of the embedding as
    output.weight, GPT-NeoX's query_key_value (and its bias) as all of q, then k, then v; Q8_0 matrices (GPT-2's
    positions F32), F32 vectors. Returns the file and, under the Hugging Face names, the values it stands for."""
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    neox = config["model_type"] == "gpt_neox"
    arch = "gptneox" if neox else "gpt2"
    heads = config["num_attention_heads"] if neox else config["n_head"]
    dim = config["hidden_size"] if neox else config["n_embd"]
    metadata = [("general.architecture", 8, arch), (f"{arch}.embedding_length", 4, dim),
                (f"{arch}.attention.head_count", 4, heads), (f"{arch}.attention.layer_norm_epsilon", 6, 1e-5)]
    if neox:
        metadata += [(f"{arch}.block_count", 4, config["num_hidden_layers"]),
                     (f"{arch}.context_length", 4, config["max_position_embeddings"]),
                     (f"{arch}.feed_forward_length", 4, config["intermediate_size"]),
                     (f"{arch}.rope.dimension_count", 4, llama2_convert.rotary_dim(config)),
                     (f"{arch}.use_parallel_residual", 7, config["use_parallel_residual"])]
    else:
        metadata += [(f"{arch}.block_count", 4, config["n_layer"]), (f"{arch}.context_length", 4, config["n_positions"]),
                     (f"{arch}.feed_forward_length", 4, config["n_inner"])]
    names = NEOX_NAMES if neox else GPT2_NAMES
    stored = {}  # GGUF name: (Hugging Face name, the values as the GGUF holds them)
    for name, tensor in tensors.items():
        if name.endswith(("masked_bias", "inv_freq")):
            continue  # nothing llama.cpp writes
        short = name.removeprefix("transformer.")
        if short in names:
            stored[names[short]] = (name, tensor)
            continue
        parts = short.split(".")
        at = 3 if neox else 2  # gpt_neox.layers.N. or h.N.
        layer, what, kind = parts[at - 1], ".".join(parts[at:-1]), parts[-1]
        gguf, value = names[what], tensor
        if not neox and kind == "weight" and gguf in CONV1D:
            value = tensor.T
        if neox and gguf == "attn_qkv":
            value = tensor.reshape(heads, 3, dim // heads, -1).swapaxes(0, 1).reshape(tensor.shape)
        stored[f"blk.{layer}.{gguf}.{kind}"] = (name, value)
    if not neox:
        stored["output.weight"] = (None, stored["token_embd.weight"][1])
    out = [b"GGUF", struct.pack("<IQQ", 3, len(stored), len(metadata))]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind))
        out.append(string(value) if kind == 8 else struct.pack({4: "<I", 6: "<f", 7: "<?"}[kind], value))
    blobs, offset, same = [], 0, {}
    for gguf, (name, value) in stored.items():
        if value.ndim == 2 and gguf != "position_embd.weight":
            blob, held = q8_0_blocks(np.ascontiguousarray(value))
            type_ = 8
        else:
            blob, held, type_ = np.ascontiguousarray(value, np.float32).tobytes(), value.astype(np.float32), 0
        if name is not None:  # back to the Hugging Face tensor these values stand for
            if not neox and gguf.startswith("blk.") and gguf.endswith(".weight") and gguf.split(".")[2] in CONV1D:
                held = held.T
            if neox and ".attn_qkv." in gguf:
                held = held.reshape(3, heads, dim // heads, -1).swapaxes(0, 1).reshape(held.shape)
            same[name] = np.ascontiguousarray(held)
        out.append(string(gguf) + struct.pack("<I", value.ndim) + struct.pack(f"<{value.ndim}Q", *reversed(value.shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    return head + b"".join(blobs), same


def unigram(vocab_size):
    return json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                       "vocab": [[f"w{i}", -float(i)] for i in range(vocab_size)]}}).encode()


def the_other(model):
    from test_gpt2 import gpt2_model
    from test_neox import neox_model
    if model == "gpt2":
        return gpt2_model()
    return neox_model(*{"neox": (0.25, True), "neox-serial": (1.0, False)}[model])


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("model", ["gpt2", "neox", "neox-serial"])
def test_a_gpt2_or_neox_gguf_with_the_originals_files_is_the_safetensors_conversion(model, dtype):
    """T136's third stage: GPT-2's matrices turned back to Conv1D and GPT-NeoX's q, k and v put back per head give
    the checkpoint, tokenizer.bin and options of the safetensors of the same values. Leaving either out of the reader
    (the transposed entries, llama2_convert.unsplit) makes the checkpoint differ."""
    tensors, config = the_other(model)
    file, same = other_gguf(tensors, config)
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(config), vocabulary, "tokenizer.json",
                          dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert got.options["arch"] == ("gpt2" if model == "gpt2" else "neox")


@pytest.mark.parametrize("model, change, what", [
    ("neox", dict(rotary_pct=0.5), "rotated values"),
    ("neox", dict(use_parallel_residual=False), "parallel residual"),
    ("neox", dict(layer_norm_eps=1e-6), "LayerNorm epsilon"),
    ("neox", dict(num_hidden_layers=1), "number of layers"),
    ("neox", dict(num_attention_heads=2), "number of heads"),
    ("neox", dict(model_type="llama"), "architecture"),
    ("gpt2", dict(n_head=2), "number of heads"),
    ("gpt2", dict(n_layer=1), "number of layers"),
])
def test_a_gpt2_or_neox_gguf_that_is_not_the_originals_is_refused(model, change, what):
    """What the tensors do not say: GPT-NeoX's rotated part and parallel branches (the options carry them, and the
    wrong ones write nonsense without a word, T72), heads of the same product, the layers."""
    tensors, config = the_other(model)
    file, _ = other_gguf(tensors, config)
    llama2_convert.gguf_weights(file, json.dumps(config))  # its own config goes through
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(file, json.dumps({**config, **change}))


def test_the_padding_of_a_gguf_vocabulary_is_empty_as_the_safetensors_path_pads():
    """T143 (T136's review): llama.cpp pads a vocabulary up to its size with unused pieces ([PAD151665] ..., type 5),
    which the safetensors path writes as empty text: the same tokenizer.bin now, which was spelling them out when
    chosen. Its control tokens (type 3) are the special ones, its user-defined ones (type 4) the added ones."""
    from llama2_numpy import Tokenizer
    # nine tenths of the vocabulary or more, or the converter takes it for another model's
    tokens = ["a", "b", "ab", *(f"w{i}" for i in range(15)), "<|endoftext|>", "<think>", "[PAD20]", "[PAD21]"]
    kinds = [1] * 18 + [3, 4, 5, 5]
    metadata = lambda count: {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "gpt-2",
                              "tokenizer.ggml.tokens": tokens[:count], "tokenizer.ggml.token_type": kinds[:count],
                              "tokenizer.ggml.merges": ["a b"]}
    data, _, _, controls, added = llama2_convert.gguf_tokenizer(metadata(22), 22)
    assert controls == ["<|endoftext|>"] and added == ["<think>"]
    assert data == llama2_convert.gguf_tokenizer(metadata(20), 22)[0]
    vocabulary = Tokenizer(data, 22, kind="bytebpe")
    assert vocabulary.vocab[20:] == [b"", b""] and vocabulary.vocab[19] == b"<think>"


# ---- T236: a Qwen3.5 (T229's hybrid attention), as llama.cpp's conversion/qwen.py writes one
QWEN35_NAMES = {"embed_tokens.weight": "token_embd.weight", "norm.weight": "output_norm.weight"}
QWEN35_LAYER = {**LAYER, "post_attention_layernorm": "post_attention_norm",
                "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
                "linear_attn.in_proj_a": "ssm_alpha", "linear_attn.in_proj_b": "ssm_beta",
                "linear_attn.conv1d": "ssm_conv1d", "linear_attn.norm": "ssm_norm", "linear_attn.out_proj": "ssm_out"}
QWEN35_WHOLE = {"linear_attn.A_log": "ssm_a", "linear_attn.dt_bias": "ssm_dt.bias"}
# rows of whole groups of 32 (Q8_0), a value head to each key head (the real 0.8B and 2B), heads that do not fill dim
QWEN35 = dict(dim=64, hidden_dim=128, n_heads=4, n_kv_heads=2, head_dim=32, key_heads=2, value_heads=2, key_dim=16,
              value_dim=16, vocab_size=40)


def qwen35_gguf(more=(), bos=1, eos=2, change=None, fold=None, matrices=None, **shape):
    """A GGUF v3 of a small Qwen3.5 the way llama.cpp writes one (as unsloth's Qwen3.5-0.8B Q8_0 is, T236): the
    language model's tensors alone, the norms with the 1 the model adds to them (not a linear-attention layer's own),
    A_log as -exp(A_log) and named ssm_a, dt_bias named ssm_dt.bias, the convolution without its axis of one, q with
    its gate and k as Hugging Face holds them; Q8_0 matrices (the two small ones of the gates too), F32 vectors and
    convolution. Returns the config.json, the file, and under the Hugging Face names the values it stands for: the
    matrices as Q8_0 rounds them, everything else as the original has it.
    more: further metadata; change(stored): alters what is written, {GGUF name: [bytes, ggml type, shape]}.
    fold(tensors): the Hugging Face tensors as the file is to hold them (T237: in a rotated basis); matrices: what
    makes a matrix's blocks and their ggml type, where not Q8_0 (as gguf_file's)."""
    from conftest import qwen35_model
    tensors, config = qwen35_model(**{**QWEN35, **shape})
    if fold:
        tensors = fold(tensors)
    blocks, block_type = matrices or (q8_0_blocks, 8)
    text = config["text_config"]
    prefix = "model.language_model." if "model.language_model.embed_tokens.weight" in tensors else "model."
    stored, same = {}, {}
    for name, tensor in tensors.items():
        if name == "lm_head.weight":
            gguf = "output.weight"
        elif name.startswith(prefix) and not name.startswith("model.visual."):
            short = name[len(prefix):]
            _, layer, *rest = short.split(".")
            what = ".".join(rest)
            gguf = QWEN35_NAMES.get(short) or (f"blk.{layer}.{QWEN35_WHOLE[what]}" if what in QWEN35_WHOLE else
                                               f"blk.{layer}.{QWEN35_LAYER['.'.join(rest[:-1])]}.{rest[-1]}")
        else:
            same[name] = tensor  # the vision model and the look-ahead head: in the original, not in this GGUF
            continue
        value = tensor
        if name.endswith(".A_log"):
            value = -np.exp(tensor.astype(np.float32))
        elif name.endswith(".conv1d.weight"):
            value = tensor.reshape(tensor.shape[0], tensor.shape[-1])
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            value = tensor.astype(np.float32) + np.float32(1)
        if value.ndim == 2 and not name.endswith(".conv1d.weight"):
            blob, held = blocks(np.ascontiguousarray(value))
            stored[gguf], same[name] = [blob, block_type, value.shape], held
        else:
            stored[gguf], same[name] = [np.ascontiguousarray(value, np.float32).tobytes(), 0, value.shape], tensor
    if change:
        change(stored)
    arch, rope = "qwen35", text["rope_parameters"]
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, text["num_hidden_layers"]),
                (f"{arch}.context_length", 4, text["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, text["hidden_size"]),
                (f"{arch}.feed_forward_length", 4, text["intermediate_size"]),
                (f"{arch}.attention.head_count", 4, text["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 4, text["num_key_value_heads"]),
                (f"{arch}.attention.key_length", 4, text["head_dim"]),
                (f"{arch}.attention.layer_norm_rms_epsilon", 6, text["rms_norm_eps"]),
                (f"{arch}.rope.freq_base", 6, float(rope["rope_theta"])),
                (f"{arch}.rope.dimension_count", 4, int(text["head_dim"] * rope["partial_rotary_factor"])),
                (f"{arch}.ssm.conv_kernel", 4, text["linear_conv_kernel_dim"]),
                (f"{arch}.ssm.state_size", 4, text["linear_key_head_dim"]),
                (f"{arch}.ssm.group_count", 4, text["linear_num_key_heads"]),
                (f"{arch}.ssm.time_step_rank", 4, text["linear_num_value_heads"]),
                (f"{arch}.ssm.inner_size", 4, text["linear_value_head_dim"] * text["linear_num_value_heads"]),
                (f"{arch}.full_attention_interval", 4, text["full_attention_interval"]),
                ("tokenizer.ggml.model", 8, "gpt2"), ("tokenizer.ggml.pre", 8, "qwen35"),
                *[(f"tokenizer.ggml.{key}_token_id", 4, id) for key, id in (("bos", bos), ("eos", eos)) if id is not None],
                *more]
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    tokens = [f"w{i}" for i in range(text["vocab_size"])]
    out = [b"GGUF", struct.pack("<IQQ", 3, len(stored), len(metadata) + 3)]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind))
        out.append(metadata_value(kind, value))
    out.append(string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens)) + b"".join(map(string, tokens)))
    out.append(string("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, len(tokens)) + struct.pack(f"<{len(tokens)}i", *[1] * len(tokens)))
    out.append(string("tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, 0))
    blobs, offset = [], 0
    for gguf, (blob, type_, shape) in stored.items():
        out.append(string(gguf) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *reversed(shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    return config, head + b"".join(blobs), same


def safetensors_conversion(same, config, vocabulary, dtype):
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(config), vocabulary, "tokenizer.json",
                          dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    return expected


QWEN35_SHAPES = {"every second": dict(), "every fourth, the language model alone": dict(n_layers=8, every=4, prefix="model."),
                 "a classifier of its own, whole heads turn": dict(shared=False, rotary=1.0, n_kv_heads=4, conv=2)}


@pytest.mark.parametrize("dtype", ["int8", "float32", "float16", "int6"])
@pytest.mark.parametrize("shape", QWEN35_SHAPES)
def test_a_qwen35_gguf_with_the_originals_files_is_the_safetensors_conversion(shape, dtype):
    """The list's way in: the checkpoint, tokenizer.bin and options of the safetensors of the same values, to the byte.
    The norms come with their 1 and A_log as -exp(A_log), which the conversion must not do to them again (and could not
    undo to the bit), the convolution comes without its axis of one, and two tensors under names of their own. Fed
    4096 bytes at a time."""
    config, file, same = qwen35_gguf(**QWEN35_SHAPES[shape])
    vocabulary = unigram(config["text_config"]["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
    expected = safetensors_conversion(same, config, vocabulary, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert got.options["arch"] == "qwen35" and got.options["linear"]["every"] == config["text_config"]["full_attention_interval"]


def test_a_qwen35_gguf_alone_converts_to_the_checkpoint_of_the_same_values():
    """Such a GGUF alone, were it to name a BOS (unsloth's names none and is refused, as T203 has it): the layers, the
    heads and the turned part of a head come from its own metadata, by config.json's names, and its pre-tokenizer is
    the one of a Qwen3.5's tokenizer.json, which normalizes to NFC."""
    config, file, same = qwen35_gguf()
    conversion = fed(file, "int8")
    expected = safetensors_conversion(same, config, unigram(config["text_config"]["vocab_size"]), "int8")
    assert bytes(conversion.checkpoint) == bytes(expected.checkpoint)
    for key in ("arch", "linear", "head_dim", "rotary", "rms_norm_eps", "rope_theta"):
        assert conversion.options[key] == expected.options[key], key
    assert conversion.options["pretokenizer"] == "qwen35" and conversion.options["nfc"] is True
    with pytest.raises(ValueError, match="names no BOS token"):
        fed(qwen35_gguf(bos=None)[1], "int8")


@pytest.mark.parametrize("change, what", [
    (dict(full_attention_interval=4, layer_types=None), "linear-attention layers"),
    (dict(linear_num_key_heads=1, linear_num_value_heads=1, linear_key_head_dim=32, linear_value_head_dim=32), "linear-attention layers"),
    (dict(linear_num_key_heads=1, linear_key_head_dim=32), "linear-attention layers"),
    (dict(rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.5}), "rotated values"),
    (dict(rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25}), "RoPE theta"),
    (dict(head_dim=16, num_attention_heads=8, num_key_value_heads=4), "number of heads"),
    (dict(rms_norm_eps=1e-5), "RMSNorm epsilon"),
    (dict(model_type="qwen3"), "architecture"),
])
def test_a_qwen35_gguf_that_is_not_the_originals_is_refused(change, what):
    """What the tensors do not say: which layers attend over all positions, heads of the same product, how much of a
    head turns."""
    config, file, _ = qwen35_gguf(n_layers=8)
    llama2_convert.gguf_weights(file, json.dumps(config))  # its own config goes through
    text = {**config["text_config"], **change}
    # another model's config.json has no text_config
    other = text if change.get("model_type") else {**config, "text_config": text}
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(file, json.dumps(other))


def test_a_qwen35_gguf_of_more_value_heads_than_key_heads_is_refused():
    """llama.cpp stores the value heads of such a model (Qwen3.5 4B and up) tiled, every key head's first and then
    every key head's second: read as they are, the heads would be other heads, without a word."""
    config, file, _ = qwen35_gguf(value_heads=4)
    with pytest.raises(ValueError, match="more value heads than key heads"):
        llama2_convert.gguf_weights(file, json.dumps(config))


def test_a_step_done_that_the_plan_has_not_is_refused():
    """A GGUF's tensor that comes with a step of the plan done (T236) is taken without it, and one said to come with a
    step its plan has not is refused: the 1 would be added to, or left off, a tensor the name table got wrong."""
    assert llama2_convert.left_to_do(("one",), None) == ("one",) and llama2_convert.left_to_do(None, None) is None
    assert llama2_convert.left_to_do(("one",), "one") is None
    assert llama2_convert.left_to_do((("heads", 1, 0, 1, 8), ("one",)), "one") == (("heads", 1, 0, 1, 8),)
    for transform in (None, ("decay",), (("heads", 1, 0, 1, 8),)):
        with pytest.raises(ValueError, match="comes with the step 'one' done"):
            llama2_convert.left_to_do(transform, "one")
