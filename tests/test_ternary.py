# ternary (T230): the weights of a ternary model as they are, two bits each, groups of 128 with a float32 scale each
# (llama2_numpy.pack_ternary), and the two GGUF types that hold such weights (PQ2_0, PTQ1_0). Nothing is rounded on the
# way: the NumPy engine widens a ternary checkpoint to exactly what a float32 checkpoint of the same values holds, so
# the logits of the two are the same to the bit. The kernels (forward.js) are held to NumPy by tests/forward-check.mjs
# and tests/smoke.mjs.
import itertools
import json
import struct

import numpy as np
import pytest
from conftest import pack_tokenizer, synthetic_weights, tiny_vocab
from test_convert import converted, hugging_face, reader, safetensors_file, streamed
from test_external import Outside
from test_gguf import bonsai_gguf, pq2_0_blocks, unigram, with_original

import llama2_convert
from llama2_convert import Arrays, Conversion, Safetensors, checkpoint_size, gguf_read
from llama2_numpy import (NOT_TERNARY, TERNARY_GROUP, Llama, checkpoint_dtype, external_tensors, pack_ternary, ternary,
                          unpack_ternary)

DTYPES = ("float32", "float16", "int8", "int6", "ternary")


def ptq1_0_blocks(values):
    """Prism ML's PTQ1_0 as its fork of llama.cpp writes it (docs/notes/t228-bonsai-2-2026-10-01.md, 3): per 128
    values d = the largest, the values over d rounded to -1, 0 or 1 and kept as that plus 1 in base 3, five to a byte
    (the first the most significant, the byte ceil(256 v / 243)); 16 bytes of values 16 n + m, 8 bytes of values 80 +
    8 n + m, 2 bytes of the four values 120 + 2 n + m (as the first four digits of five), then d as float16. Written
    value by value, apart from the reader's arithmetic. And what the block then stands for: (digit - 1) * d."""
    groups = values.reshape(-1, 128).astype(np.float32)
    d = np.abs(groups).max(axis=1).astype(np.float16)
    with np.errstate(divide="ignore", invalid="ignore"):
        digits = np.where(d[:, None] > 0, np.rint(groups / d.astype(np.float32)[:, None]), 0).astype(np.int64) + 1
    out = bytearray()
    for block, scale in zip(digits, d):
        byte = lambda five: -(-(sum(int(digit) * 3 ** (4 - place) for place, digit in enumerate(five)) * 256) // 243)
        out += bytes(byte([block[16 * n + m] for n in range(5)]) for m in range(16))
        out += bytes(byte([block[80 + 8 * n + m] for n in range(5)]) for m in range(8))
        out += bytes(byte([block[120 + 2 * n + m] for n in range(4)]) for m in range(2))
        out += scale.tobytes()
    return bytes(out), ((digits - 1).astype(np.float32) * d.astype(np.float32)[:, None]).reshape(values.shape)


def test_packing_is_lossless_and_32_bytes_a_group():
    values = np.random.default_rng(1).integers(-1, 2, size=(5, 256)).astype(np.int8)
    packed = pack_ternary(values)
    assert packed.shape == (5 * 64,) and packed.dtype == np.uint8
    assert np.array_equal(unpack_ternary(packed).reshape(values.shape), values)
    # the layout the kernels read, and PQ2_0's: weight j's code (the weight + 1) in byte j // 4 at bits 2 (j % 4)
    for j, value in itertools.product(range(8), (-1, 0, 1)):
        one = np.zeros(8, dtype=np.int8)
        one[j] = value
        assert pack_ternary(one).tolist() == [(0x55 & ~(3 << 2 * (j % 4)) | (value + 1) << 2 * (j % 4)) if k == j // 4 else 0x55
                                              for k in range(2)]
    assert unpack_ternary(np.array([0xFF], dtype=np.uint8)).tolist() == [2, 2, 2, 2]  # the code no ternary file has


def test_ternary_takes_the_values_as_they_are_and_refuses_any_other():
    rng = np.random.default_rng(2)
    signs = rng.integers(-1, 2, size=(6, 256)).astype(np.float32)
    scales = np.array([0.0078125, 3.0, 1e-30, 0.0, 65504.0, 1.5e-5] * 2, dtype=np.float32)
    values = (signs.reshape(-1, 128) * scales[:, None]).reshape(6, 256)
    packed, got = ternary(values)
    assert packed.shape == (12, 32) and packed.dtype == np.uint8 and got.dtype == np.float32
    some = np.abs(values.reshape(-1, 128)).max(axis=1) > 0
    assert np.array_equal(got[some], scales[some]) and not got[~some].any()
    assert np.array_equal(unpack_ternary(packed).reshape(-1, 128) * got[:, None], values.reshape(-1, 128))
    for spoil in (1.5, 2.9999998, np.nan, 3.0000002, 6.0, -1e-30):
        bad = values.copy()
        bad[0, 200] = spoil  # the second group, whose scale is 3
        with pytest.raises(ValueError, match="not ternary"):
            ternary(bad)
    # PQ2_0's fourth code, 2 d: the largest of its group, so every d of the group is half of it
    with pytest.raises(ValueError, match="not ternary"):
        ternary(np.array([2.0, 1.0] + [0.0] * 126, dtype=np.float32))
    assert NOT_TERNARY.startswith("These weights are not ternary")


def test_ptq1_0_reads_every_byte_as_the_fork_does():
    """Every five digits of a byte (243) at each of the 24 bytes of five, every four (81) at each of the 2 bytes of
    four, under several scales: the reader against the fork's dequantize_row_ptq1_0 written out value by value (the
    n-th digit of a byte is ((byte * 3 ** n) % 256 * 3) >> 8), and the order of the values, which is not the bytes'."""
    scales = np.array([1.0, 0.0078125, -2.5, 6.1e-5, 0.0, 65504.0], dtype=np.float16)
    blocks, expected = [], []
    for case in range(243):
        block = bytearray(28)
        for m in range(24):
            block[m] = -(-((case + 7 * m) % 243 * 256) // 243)
        for m in range(2):
            block[24 + m] = -(-((case + m) % 81 * 3 * 256) // 243)
        d = scales[case % len(scales)]
        block[26:28] = d.tobytes()
        digit = lambda byte, n: ((byte * 3 ** n) % 256 * 3) >> 8
        values = [digit(block[m], n) for n in range(5) for m in range(16)]
        values += [digit(block[16 + m], n) for n in range(5) for m in range(8)]
        values += [digit(block[24 + m], n) for n in range(4) for m in range(2)]
        assert len(values) == 128 and set(values) <= {0, 1, 2}
        blocks.append(bytes(block))
        expected.append([(value - 1) * np.float32(d) for value in values])
    got = llama2_convert.ptq1_0(b"".join(blocks))
    assert np.array_equal(got.view(np.uint32), np.array(expected, dtype=np.float32).reshape(-1).view(np.uint32))
    # and what the test's own writer makes of ternary values comes back as them, in their order
    values = np.random.default_rng(3).integers(-1, 2, (4, 256)).astype(np.float32) * np.float32(0.0625)
    blob, held = ptq1_0_blocks(values)
    assert len(blob) == 8 * 28 and np.array_equal(held, values)
    assert np.array_equal(llama2_convert.ptq1_0(blob).reshape(values.shape), values)


BLOCKS = {"PQ2_0": (pq2_0_blocks, 142), "PTQ1_0": (ptq1_0_blocks, 143)}


def bonsai(kind, **more):
    """bonsai_gguf() (a small Qwen3 with yarn, rows of 128 and 256) with its matrices as either type: the same ternary
    values either way, as the 27B's two files hold the same model."""
    return bonsai_gguf(matrices=BLOCKS[kind], **more)


@pytest.mark.parametrize("kind", ["PQ2_0", "PTQ1_0"])
@pytest.mark.parametrize("dtype", ["ternary", "int8", "float32"])
def test_a_ternary_gguf_is_the_safetensors_conversion(kind, dtype):
    """Either type, to ternary and to the other dtypes: the checkpoint of a safetensors file of the values the blocks
    stand for (ternary there is the same refusal of anything else: the values pass because they are ternary). Fed 4096
    bytes at a time: a chunk ends within a block and within a row."""
    config, published, file, same = bonsai(kind)
    _, found, _ = gguf_read(file)
    assert {info["type"] for info in found.values() if len(info["shape"]) == 2} == {BLOCKS[kind][1]}
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, published, vocabulary, "tokenizer.json", dtype)
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary,
                          "tokenizer.json", dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    assert bytes(got.checkpoint) == bytes(expected.checkpoint) and got.options == expected.options
    assert got.options["dtype"] == dtype
    if dtype == "ternary":
        header = struct.unpack_from("<7i", got.checkpoint, 0)
        assert checkpoint_dtype(header, len(got.checkpoint), got.options) == "ternary"
        assert len(got.checkpoint) == checkpoint_size(header, "ternary", got.options)
        assert len(got.checkpoint) < len(with_original(file, published, vocabulary, "tokenizer.json", "int8").checkpoint) / 3.5


def test_the_two_types_of_one_model_are_one_checkpoint():
    assert bonsai("PQ2_0")[3].keys() == bonsai("PTQ1_0")[3].keys()
    checkpoints = []
    for kind in BLOCKS:
        config, published, file, same = bonsai(kind)
        checkpoints.append(bytes(with_original(file, published, unigram(config["vocab_size"]), "tokenizer.json", "ternary").checkpoint))
    assert checkpoints[0] == checkpoints[1]


def test_the_engine_reads_ternary_as_the_float32_of_the_same_values():
    config, published, file, same = bonsai("PTQ1_0")
    vocabulary = unigram(config["vocab_size"])
    packed = with_original(file, published, vocabulary, "tokenizer.json", "ternary")
    wide = with_original(file, published, vocabulary, "tokenizer.json", "float32")
    # the float32 file holds the RoPE tables the converter made; the ternary one leaves them to the engine: the same
    # tables (T235), so the logits are the same to the bit
    a = Llama(bytes(packed.checkpoint), packed.tokenizer, **packed.options)
    b = Llama(bytes(wide.checkpoint), wide.tokenizer, **wide.options)
    for pos, token in enumerate([1, 5, 9, 3]):
        assert np.array_equal(a.forward(token, pos), b.forward(token, pos))
    # the embedding's rows are read from the packed table one at a time where the classifier is another table
    assert np.array_equal(a.embedding(7), b.embedding(7))


def test_the_tensors_of_a_ternary_file_are_placed_before_the_model_is_built():
    config, published, file, same = bonsai("PQ2_0")
    packed = with_original(file, published, unigram(config["vocab_size"]), "tokenizer.json", "ternary")
    outside = Outside(bytes(packed.checkpoint))
    Llama(None, packed.tokenizer, external=outside, **packed.options)
    header = np.frombuffer(packed.checkpoint, dtype=np.int32, count=7).tolist()
    tensors = outside.plan["tensors"]
    assert external_tensors(header, "ternary", packed.options) == tensors
    assert tensors["wq"]["kind"] == "ternary" and tensors["wq"]["group"] == TERNARY_GROUP
    count = int(np.prod(tensors["wq"]["shape"]))
    assert tensors["wq"]["scales"] == tensors["wq"]["offset"] + count // 4
    assert tensors["rms_att_weight"]["kind"] == "f32"


def test_weights_that_are_not_ternary_and_rows_that_are_not_groups_of_128_are_refused():
    config, weights = synthetic_weights(dim=128, hidden_dim=256, n_kv_heads=2, vocab_size=40)
    tensors, published = hugging_face(config, weights, True)
    with pytest.raises(ValueError, match="not ternary"):
        converted(Arrays(tensors), published, "ternary")
    config, weights = synthetic_weights(dim=96, hidden_dim=128, n_heads=2, n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    with pytest.raises(ValueError, match="groups of 128"):
        converted(Arrays(tensors), published, "ternary")


def test_the_file_in_its_own_order_gives_the_same_ternary():
    config, published, file, same = bonsai("PQ2_0")
    safetensors = safetensors_file(same)
    got, progress = streamed(safetensors, published, "ternary", 4096)
    assert got == converted(Safetensors(reader(safetensors)), published, "ternary") and progress[-1][0] == progress[-1][1]


# the headers and forms of the ternary models there are (Ternary Bonsai 1.7B, 4B and 8B: Qwen3s; Ternary Bonsai 2 27B:
# a Qwen3.5 whose value heads are three times its key heads), with the context the page cuts them to and their own
BONSAI = [([2048, 6144, 28, 16, 8, 151936, context], {"qk_norm": True, "head_dim": 128}) for context in (4096, 32768)] + \
    [([2560, 9728, 36, 32, 8, 151936, context], {"qk_norm": True, "head_dim": 128}) for context in (4096, 32768)] + \
    [([4096, 12288, 36, 32, 8, 151936, context], {"qk_norm": True, "head_dim": 128}) for context in (4096, 65536)] + \
    [([5120, 17408, 64, 24, 4, -248320, context],
      {"arch": "qwen35", "head_dim": 256,
       "linear": {"every": 4, "key_heads": 16, "value_heads": 48, "key_dim": 128, "value_dim": 128, "conv": 4}})
     for context in (4096, 262144)]


def sizes_of(header, form):
    return {dtype: checkpoint_size(header, dtype, form) for dtype in DTYPES}


@pytest.mark.parametrize("header, form", BONSAI, ids=lambda value: str(value[0]) + "x" + str(value[6]) if isinstance(value, list) else "")
def test_a_ternary_file_is_told_apart_by_its_size(header, form):
    sizes = sizes_of(header, form)
    assert len(set(sizes.values())) == len(DTYPES), sizes
    for dtype, size in sizes.items():
        assert checkpoint_dtype(header, size, form) == dtype
    # two bits and a quarter a weight: under a third of six bits
    assert sizes["ternary"] < sizes["int6"] / 3


def test_no_shape_gives_two_dtypes_one_size():
    """checkpoint_dtype() tells the dtypes of one header apart by the file's size alone: over many shapes whose rows are
    whole groups of 128 (so that ternary is possible), with and without the biases and the norms of the heads, small
    and large vocabularies and contexts, no two of the five have one size."""
    tried = 0
    for dim, hidden, layers, heads, vocab, context in itertools.product(
            (128, 256, 1024), (128, 384, 4096), (1, 2, 12), (1, 2, 8), (1, 300, 150000, -300, -150000), (1, 512, 1 << 18)):
        for form in ({}, {"bias": True}, {"qk_norm": True}, {"qk_norm": True, "head_dim": 128}, {"arch": "gpt2"}, {"arch": "neox"}):
            header = [dim, hidden, layers, heads, heads, vocab, context]
            sizes = sizes_of(header, form)
            assert len(set(sizes.values())) == len(DTYPES), (header, form, sizes)
            assert checkpoint_dtype(header, sizes["ternary"], form) == "ternary"
            tried += 1
    assert tried > 1500
