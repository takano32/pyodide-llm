# T505.1: llama.cpp's quantized types that are read and never kept: Q3_K, Q5_K and Q6_K (blocks of 256,
# src/python/convert/superblocks.py's WIDENED) and Q4_0, Q4_1, Q5_0, Q5_1 and IQ4_NL (blocks of 32,
# convert/readers.py), which a Q4_K_M, a Q5_K_M or a Q3_K_M file is made of besides the types T362 reads. Each reader
# against a reference written here one weight at a time from the form of its block, on blocks that go through every
# scale and every value of a weight; a small Qwen3.5 whose matrices are of these types, through the GGUF's reader and
# the stream to the checkpoint of a safetensors file of the same values; and that a conversion that keeps llama.cpp's
# blocks widens these all the same (the engine's BLOCKS does not have them: no kernel multiplies them). That the
# readers are llama.cpp's own C to the bit is tests/fork_readers.py's.
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import superblock_cases
from test_gguf import q8_0_blocks, qwen35_gguf, safetensors_conversion, unigram, with_original
from test_kept import MATRICES, Model
from test_superblocks import SALUKI, SPECIAL, blocks_of, half

from convert import grids
from convert.readers import read_types
from convert.stream import stays
from convert.superblocks import SUPERBLOCKS, WIDENED
from llama2_convert import GGUF_TENSORS, SOURCES, Conversion, gguf_weights
from llama2_numpy import BLOCKS, KINDS, Llama

FIXTURE = Path(__file__).parent / "fixtures" / "superblocks.json"
F = np.float32


def slow_q3_k(b, seen):
    d, s, out = half(b, 108), b[96:108], []
    for at in range(256):
        part, shift, byte = at // 128, at % 128 // 32, at % 32
        group = at // 16
        low = s[group] & 15 if group < 8 else s[group - 8] >> 4
        scale = (low | (s[8 + group % 4] >> 2 * (group // 4) & 3) << 4) - 32
        q = b[32 + 32 * part + byte] >> 2 * shift & 3
        value = q if b[byte] >> (4 * part + shift) & 1 else q - 4
        seen["scale"].add(scale), seen["value"].add(value)
        out.append(F(d * F(scale)) * F(value))
    return out


def six_bits(s, group):
    """(scale, minimum) of group of Q4_K's and Q5_K's 12 bytes."""
    if group < 4:
        return s[group] & 63, s[group + 4] & 63
    return s[group + 4] & 15 | s[group - 4] >> 6 << 4, s[group + 4] >> 4 | s[group] >> 6 << 4


def slow_q5_k(b, seen):
    d, least, out = half(b, 0), half(b, 2), []
    for at in range(256):
        group, l = at // 32, at % 32
        scale, minimum = six_bits(b[4:16], group)
        byte = b[48 + 32 * (group // 2) + l]
        q = (byte >> 4 if group % 2 else byte & 15) | (b[16 + l] >> group & 1) << 4
        seen["scale"].add(scale), seen["minimum"].add(minimum), seen["value"].add(q)
        out.append(F(d * F(scale)) * F(q) - F(least * F(minimum)))
    return out


def slow_q6_k(b, seen):
    d, out = half(b, 208), []
    for at in range(256):
        part, run, l = at // 128, at % 128 // 32, at % 32
        (scale,) = struct.unpack_from("<b", b, 192 + at // 16)
        byte = b[64 * part + 32 * (run % 2) + l]
        q = ((byte >> 4 if run >= 2 else byte & 15) | (b[128 + 32 * part + l] >> 2 * run & 3) << 4) - 32
        seen["scale"].add(scale), seen["value"].add(q)
        out.append(F(d * F(scale)) * F(q))
    return out


def four_bits(qs, j):
    return qs[j - 16] >> 4 if j >= 16 else qs[j] & 15


def slow_q4_0(b, seen):
    seen["value"].update(four_bits(b[2:], j) - 8 for j in range(32))
    return [half(b, 0) * F(four_bits(b[2:], j) - 8) for j in range(32)]


def slow_q4_1(b, seen):
    seen["value"].update(four_bits(b[4:], j) for j in range(32))
    return [half(b, 0) * F(four_bits(b[4:], j)) + half(b, 2) for j in range(32)]


def five_bits(qh, qs, j):
    return four_bits(qs, j) | (struct.unpack("<I", qh)[0] >> j & 1) << 4


def slow_q5_0(b, seen):
    seen["value"].update(five_bits(b[2:6], b[6:], j) - 16 for j in range(32))
    return [half(b, 0) * F(five_bits(b[2:6], b[6:], j) - 16) for j in range(32)]


def slow_q5_1(b, seen):
    seen["value"].update(five_bits(b[4:8], b[8:], j) for j in range(32))
    return [half(b, 0) * F(five_bits(b[4:8], b[8:], j)) + half(b, 2) for j in range(32)]


def slow_iq4_nl(b, seen):
    seen["value"].update(four_bits(b[2:], j) for j in range(32))
    return [half(b, 0) * F(int(grids.IQ4NL[four_bits(b[2:], j)])) for j in range(32)]


# the type -> (its reference, ggml's number, the weights and the bytes of a block, where its float16s are, what the
# blocks must have gone through: how many different scales, minima and values of a weight)
SLOW = {"Q3_K": (slow_q3_k, 11, 256, 110, (108,), dict(scale=64, value=8)),
        "Q5_K": (slow_q5_k, 13, 256, 176, (0, 2), dict(scale=64, minimum=64, value=32)),
        "Q6_K": (slow_q6_k, 14, 256, 210, (208,), dict(scale=256, value=64)),
        "Q4_0": (slow_q4_0, 2, 32, 18, (0,), dict(value=16)), "Q4_1": (slow_q4_1, 3, 32, 20, (0, 2), dict(value=16)),
        "Q5_0": (slow_q5_0, 6, 32, 22, (0,), dict(value=32)), "Q5_1": (slow_q5_1, 7, 32, 24, (0, 2), dict(value=32)),
        "IQ4_NL": (slow_iq4_nl, 20, 32, 18, (0,), dict(value=16))}


def made_up_blocks(name, count, seed=0, scale=None):
    """count blocks of the type: random bytes (any bytes are a block), every byte value in every place of the first
    256, and finite float16s, the first few of every kind (scale: all of them that one instead)."""
    _, ggml, _, size, places, _ = SLOW[name]
    blocks = np.random.default_rng([seed, ggml]).integers(0, 256, (count, size), dtype=np.uint8)
    for place in range(size):
        blocks[:256, place] = ((np.arange(256) + 7 * place) % 256)[:count]
    for place in places:
        blocks[:, place + 1] &= 0xBF  # not an infinity or a NaN
        if scale is None:
            blocks[:len(SPECIAL), place:place + 2] = np.array(SPECIAL, dtype="<u2").view(np.uint8).reshape(-1, 2)[:count]
        else:
            blocks[:, place:place + 2] = np.array([scale], dtype="<u2").view(np.uint8)
    return blocks


@pytest.mark.parametrize("name", SLOW)
def test_a_reader_reads_what_the_form_of_its_block_says_one_weight_at_a_time(name):
    """Every scale of a group, every value of a weight, float16s of every finite kind: the same float32, to the bit
    and to the sign of a zero."""
    slow, _, values, size, _, through = SLOW[name]
    blocks = made_up_blocks(name, 300)
    seen = {key: set() for key in ("scale", "minimum", "value")}
    expected = np.array([slow(bytes(block), seen) for block in blocks], dtype=np.float32).reshape(-1)
    assert {key: len(found) for key, found in seen.items() if found} == through
    got = SOURCES[name].read(blocks.tobytes())
    assert got.dtype == np.float32 and got.shape == (300 * values,) and np.isfinite(got).all()
    assert np.array_equal(got.view(np.uint32), expected.view(np.uint32))
    for held in (bytearray(blocks[:3].tobytes()), memoryview(blocks[:3].tobytes())):
        assert np.array_equal(SOURCES[name].read(held).view(np.uint32), expected[:3 * values].view(np.uint32))
    assert SOURCES[name].read(b"").shape == (0,)
    with pytest.raises(ValueError):
        SOURCES[name].read(blocks.tobytes()[:-1])


def test_the_types_are_the_tables_and_a_gguf_says_them_by_ggmls_numbers():
    """And none of them is a kind a row of a checkpoint is stored in, a kernel reads, or a refusal names."""
    assert set(WIDENED) == {"Q3_K", "Q5_K", "Q6_K"} and not set(WIDENED) & set(SUPERBLOCKS)
    for name, (_, ggml, values, size, _, _) in SLOW.items():
        source = SOURCES[name]
        assert (source.ggml, source.values, source.size, source.kernel, source.said) == (ggml, values, size, None, False)
        assert GGUF_TENSORS[ggml] == name and source.bytes(512) == 512 // values * size
        assert name not in BLOCKS and name not in KINDS
        assert not stays({"dtype": name}, None)
    assert set(BLOCKS) == set(SUPERBLOCKS) == {name for name, source in SOURCES.items() if source.values == 256} - set(WIDENED)
    # the words of a refusal are the owner's, and as they were before T362 (T501)
    assert read_types() == "F32, F16, BF16, Q8_0, PQ2_0 and PTQ1_0"


def test_what_is_read_of_blocks_made_the_same_everywhere_is_the_fixtures():
    """tests/fixtures/superblocks.json's "widened", which tests/smoke.mjs asks of the readers inside Pyodide (integers
    of 32 bits). Written by `python tests/test_widened.py`."""
    assert json.loads(FIXTURE.read_text())["widened"] == read_of_the_cases()


def read_of_the_cases():
    with np.errstate(invalid="ignore"):  # (some of the float16s are infinities and NaNs)
        return {name: superblock_cases.digest(SOURCES[name].read(superblock_cases.blocks(name, SOURCES[name].size)))
                for name in SLOW}


# ---- a GGUF of them
def blocks_of_these(name, scale=0x0C00):
    """What qwen35_gguf() makes a matrix's blocks with (test_superblocks.blocks_of(), for these types)."""
    def blocks(values):
        raw = made_up_blocks(name, values.size // SLOW[name][2], seed=values.shape[0], scale=scale).tobytes()
        return raw, SOURCES[name].read(raw).reshape(values.shape)
    return blocks, SOURCES[name].ggml


def as_a_mixed_file(seen, with_kept=()):
    """A type to a tensor, by its layer and its role: the eight of this file, and with_kept's of T362's among them (a
    Q4_K_M file is Q4_K with Q6_K; a Q3_K_M one Q3_K with Q4_K, Q5_K and Q6_K). The gates' two small matrices Q8_0."""
    order = [*SLOW, *with_kept]

    def matrices(gguf):
        parts = gguf.split(".")
        if parts[-2] in ("ssm_alpha", "ssm_beta"):
            return q8_0_blocks, 8
        layer = int(parts[1]) if parts[0] == "blk" else 0
        name = order[(3 * layer + sum(map(ord, parts[-2]))) % len(order)]
        seen.setdefault(parts[-2], set()).add(name)
        return blocks_of_these(name) if name in SLOW else blocks_of(name)
    return matrices


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_a_gguf_of_these_types_is_the_safetensors_conversion_of_what_its_blocks_stand_for(dtype, monkeypatch):
    """A small Qwen3.5 in llama.cpp's names and order whose matrices are of the eight types, differing from layer to
    layer: its checkpoint is that of a safetensors file of the values the blocks stand for, to the byte, the engine
    reads it and gives finite logits. Alone (its own vocabulary) it is the same checkpoint."""
    monkeypatch.setattr("convert.stream.PIECE", 70000)
    seen = {}
    config, file, same = qwen35_gguf(matrices=as_a_mixed_file(seen), **SALUKI)
    assert set().union(*seen.values()) == set(SLOW)
    header = json.loads(gguf_weights(file, json.dumps(config))[0])
    assert {info["dtype"] for name, info in header.items() if name != "__metadata__"} == {"F32", "Q8_0", *SLOW}
    vocabulary = unigram(config["text_config"]["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
    expected = safetensors_conversion(same, config, vocabulary, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint) and got.options == expected.options
    model = Llama(bytes(got.checkpoint), bytes(got.tokenizer), **got.options)
    logits = [model.forward(token, position) for position, token in enumerate((3, 9, 27, 5))]
    assert all(np.isfinite(row).all() and np.ptp(row) > 0 for row in logits)


def test_a_conversion_that_keeps_blocks_widens_these_and_keeps_the_others():
    """blocks=True of a file of Q4_K and IQ4_XS with these types among them: a stack one of whose layers is of these
    types is the file's dtype's whole, as a stack with a Q8_0 layer is; a stack all of whose layers are T362's types
    stays in its blocks. No row's kind is one of these, and the engine reads the checkpoint."""
    seen = {}
    model = Model(matrices=as_a_mixed_file(seen, with_kept=("Q4_K", "IQ4_XS")))
    kept, wide = model.converted(), model.converted(blocks=False)
    kinds = kept.options.get("kinds") or {}
    assert kinds, "nothing stayed in its blocks: the file has rows all of whose layers are Q4_K or IQ4_XS"
    named = {kind for row in kinds.values() for kind in ([row] if isinstance(row, str) else row)}
    assert named <= {"Q4_K", "IQ4_XS"} and not named & set(SLOW)
    assert "kinds" not in wide.options and bytes(kept.checkpoint) != bytes(wide.checkpoint)
    # (a row is kept exactly where every layer of it is of a type the engine holds)
    assert set(kinds) == {row for row in MATRICES if set(model.types(row)) <= set(BLOCKS)}
    assert 0 < len(kinds) < len(MATRICES)
    held = Llama(bytes(kept.checkpoint), bytes(kept.tokenizer), **kept.options)
    logits = np.stack([held.forward(token, position) for position, token in enumerate((3, 9, 27, 5))])
    assert np.isfinite(logits).all()
    # all of them widened: no row is kept, and the checkpoint is the one a conversion that was not asked makes
    alone = Model(matrices=as_a_mixed_file({}))
    asked, plain = alone.converted(), alone.converted(blocks=False)
    assert bytes(asked.checkpoint) == bytes(plain.checkpoint) and asked.options == plain.options


def blocks_of_any(name):
    def blocks(values):
        return bytes(-(-values.size // SLOW[name][2]) * SLOW[name][3]), np.zeros(values.shape, dtype=np.float32)
    return blocks, SOURCES[name].ggml


@pytest.mark.parametrize("name", SLOW)
def test_a_tensor_whose_rows_are_not_whole_blocks_is_refused(name):
    """Rows of 128 for a type of blocks of 256, rows of 48 for one of 32: ggml itself requires whole blocks, and such
    a file would be read at the wrong offsets and write nonsense instead of failing."""
    values = SLOW[name][2]
    shape = dict(dim=128, hidden_dim=256, head_dim=32, value_dim=64, key_dim=32) if values == 256 else \
        dict(dim=48, hidden_dim=96, head_dim=12, value_dim=24, key_dim=12, n_heads=4, n_kv_heads=2)
    file = qwen35_gguf(matrices=blocks_of_any(name), **shape)[1]
    with pytest.raises(ValueError, match=f"{name} with rows of {shape['dim']}, which is not a multiple of {values}"):
        Conversion.from_gguf(file)


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps({**json.loads(FIXTURE.read_text()), "widened": read_of_the_cases()}, indent=1) + "\n")
