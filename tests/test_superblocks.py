# T362: llama.cpp's K- and i-quantized types (src/python/convert/superblocks.py and its tables, convert/grids.py), the
# ten a GGUF of Underdog Saluki 27B is made of. Each reader against a reference written here one weight at a time, from
# the form of the block (docs/notes/t361-saluki-2026-10-13.md), on blocks that go through every entry of the type's
# codebook, every scale and every sign; the tables against what they are known to be; and a small Qwen3.5 whose
# matrices are of all ten types, a type to a tensor as the real file has them, through the GGUF's reader and the stream
# to a checkpoint the engine reads. That the readers are llama.cpp's own C to the bit is tests/fork_readers.py's (it
# fetches and compiles llama.cpp).
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import superblock_cases
from test_gguf import q8_0_blocks, qwen35_gguf, safetensors_conversion, unigram, with_original

from convert import grids, superblocks
from convert.superblocks import SUPERBLOCKS
from llama2_convert import GGUF_TENSORS, SOURCES, Conversion, gguf_weights
from llama2_numpy import Llama

FIXTURE = Path(__file__).parent / "fixtures" / "superblocks.json"
GRID = {name: grids.grid(name).tolist() for name in grids.DIGITS}
F = np.float32


def half(block, at):
    return F(np.frombuffer(block[at:at + 2], dtype="<f2")[0])


def signed(eight, bits):
    return [-v if bits >> j & 1 else v for j, v in enumerate(eight)]


def seven(bits):
    """The byte of signs of seven stored bits: the eighth makes the minus signs an even number."""
    return bits | (bin(bits).count("1") & 1) << 7


def slow_q2_k(b, seen):
    d, least, out = half(b, 80), half(b, 82), [None] * 256
    for part in range(2):
        for shift in range(4):
            for byte in range(32):
                at = 128 * part + 32 * shift + byte
                scales = b[at // 16]
                seen["scale"].add(scales & 15), seen["minimum"].add(scales >> 4)
                out[at] = F(d * F(scales & 15)) * F(b[16 + 32 * part + byte] >> 2 * shift & 3) - F(least * F(scales >> 4))
    return out


def slow_q4_k(b, seen):
    d, least, s, out = half(b, 0), half(b, 2), b[4:16], []
    for group in range(8):
        if group < 4:
            scale, minimum = s[group] & 63, s[group + 4] & 63
        else:
            scale, minimum = s[group + 4] & 15 | s[group - 4] >> 6 << 4, s[group + 4] >> 4 | s[group] >> 6 << 4
        seen["scale"].add(scale), seen["minimum"].add(minimum)
        q = b[16 + 32 * (group // 2):48 + 32 * (group // 2)]
        out += [F(d * F(scale)) * F(byte >> 4 if group % 2 else byte & 15) - F(least * F(minimum)) for byte in q]
    return out


def slow_iq2_xxs(b, seen):
    d, out = float(half(b, 0)), []
    for group in range(8):
        index, (aux,) = b[2 + 8 * group:6 + 8 * group], struct.unpack_from("<I", b, 6 + 8 * group)
        seen["scale"].add(aux >> 28)
        for l in range(4):
            seen["index"].add(index[l]), seen["signs"].add(aux >> 7 * l & 127)
            out += [d * (0.5 + (aux >> 28)) * 0.25 * v for v in signed(GRID["IQ2XXS"][index[l]], seven(aux >> 7 * l & 127))]
    return out


def slow_iq2_xs(b, seen):
    d, out = float(half(b, 0)), []
    for eight in range(32):
        (q,), scales = struct.unpack_from("<H", b, 2 + 2 * eight), b[66 + eight // 4]
        scale = scales >> 4 if eight % 4 >= 2 else scales & 15
        seen["scale"].add(scale), seen["index"].add(q & 511), seen["signs"].add(q >> 9)
        out += [d * (0.5 + scale) * 0.25 * v for v in signed(GRID["IQ2XS"][q & 511], seven(q >> 9))]
    return out


def slow_iq2_s(b, seen):
    d, out = float(half(b, 0)), []
    for eight in range(32):
        index, scales = b[2 + eight] | (b[66 + eight // 4] >> 2 * (eight % 4) & 3) << 8, b[74 + eight // 4]
        scale = scales >> 4 if eight % 4 >= 2 else scales & 15
        seen["scale"].add(scale), seen["index"].add(index), seen["signs"].add(b[34 + eight])
        out += [d * (0.5 + scale) * 0.25 * v for v in signed(GRID["IQ2S"][index], b[34 + eight])]
    return out


def slow_iq3_xxs(b, seen):
    d, out = float(half(b, 0)), []
    for group in range(8):
        (aux,) = struct.unpack_from("<I", b, 66 + 4 * group)
        seen["scale"].add(aux >> 28)
        for l in range(4):
            first, second = b[2 + 8 * group + 2 * l], b[3 + 8 * group + 2 * l]
            seen["index"].update((first, second)), seen["signs"].add(aux >> 7 * l & 127)
            eight = GRID["IQ3XXS"][first] + GRID["IQ3XXS"][second]
            out += [d * (0.5 + (aux >> 28)) * 0.5 * v for v in signed(eight, seven(aux >> 7 * l & 127))]
    return out


def slow_iq3_s(b, seen):
    d, out = float(half(b, 0)), []
    for eight in range(32):
        group, scales = eight // 4, b[106 + eight // 8]
        scale = scales >> 4 if group % 2 else scales & 15
        first = b[2 + 2 * eight] | (b[66 + group] >> 2 * (eight % 4) & 1) << 8
        second = b[3 + 2 * eight] | (b[66 + group] >> 2 * (eight % 4) + 1 & 1) << 8
        seen["scale"].add(scale), seen["index"].update((first, second)), seen["signs"].add(b[74 + eight])
        out += [d * (2 * scale + 1) * v for v in signed(GRID["IQ3S"][first] + GRID["IQ3S"][second], b[74 + eight])]
    return out


def slow_iq1_s(b, seen):
    d, out = float(half(b, 0)), []
    for eight in range(32):
        (high,) = struct.unpack_from("<H", b, 34 + 2 * (eight // 4))
        index, scale, shift = b[2 + eight] | (high >> 3 * (eight % 4) & 7) << 8, high >> 12 & 7, -0.125 if high >> 15 else 0.125
        seen["scale"].add(scale), seen["index"].add(index), seen["shift"].add(shift)
        out += [d * (2 * scale + 1) * (v + shift) for v in GRID["IQ1S"][index]]
    return out


def slow_iq1_m(b, seen):
    held, out = struct.unpack_from("<4H", b, 48), []
    d = float(np.array([held[0] >> 12 | held[1] >> 12 << 4 | held[2] >> 12 << 8 | held[3] >> 12 << 12], dtype="<u2").view("<f2")[0])
    for eight in range(32):
        four = b[32 + eight // 2] >> 4 if eight % 2 else b[32 + eight // 2] & 15
        sixteen = eight // 2
        scale = held[sixteen // 4] >> 3 * (sixteen % 4) & 7
        index, shift = b[eight] | (four & 7) << 8, -0.125 if four & 8 else 0.125
        seen["scale"].add(scale), seen["index"].add(index), seen["shift"].add(shift)
        out += [d * (2 * scale + 1) * (v + shift) for v in GRID["IQ1S"][index]]
    return out


def slow_iq4_xs(b, seen):
    d, (high,), out = float(half(b, 0)), struct.unpack_from("<H", b, 2), []
    for group in range(8):
        low = b[4 + group // 2] >> 4 if group % 2 else b[4 + group // 2] & 15
        scale, q = (low | (high >> 2 * group & 3) << 4) - 32, b[8 + 16 * group:24 + 16 * group]
        seen["scale"].add(scale)
        out += [d * scale * int(grids.IQ4NL[byte & 15]) for byte in q] + [d * scale * int(grids.IQ4NL[byte >> 4]) for byte in q]
    return out


# the type -> (its reference, how many blocks go through all of its codebook, what the blocks must have gone through:
# how many different indices, scales, minima, stored signs and shifts)
SLOW = {"Q2_K": (slow_q2_k, 300, dict(scale=16, minimum=16)), "Q4_K": (slow_q4_k, 300, dict(scale=64, minimum=64)),
        "IQ2_XXS": (slow_iq2_xxs, 300, dict(index=256, scale=16, signs=128)),
        "IQ2_XS": (slow_iq2_xs, 400, dict(index=512, scale=16, signs=128)),
        "IQ2_S": (slow_iq2_s, 600, dict(index=1024, scale=16, signs=256)),
        "IQ3_XXS": (slow_iq3_xxs, 300, dict(index=256, scale=16, signs=128)),
        "IQ3_S": (slow_iq3_s, 300, dict(index=512, scale=16, signs=256)),
        "IQ1_S": (slow_iq1_s, 1200, dict(index=2048, scale=8, shift=2)),
        "IQ1_M": (slow_iq1_m, 1200, dict(index=2048, scale=8, shift=2)),
        "IQ4_XS": (slow_iq4_xs, 300, dict(scale=64))}
# where the float16 scales of a block are (IQ1_M's is the top four bits of its last four words of 16 bits)
SCALES = {"Q2_K": (80, 82), "Q4_K": (0, 2), "IQ1_M": ()}
# float16s of every finite kind: both zeros, the smallest and the largest subnormal, the smallest normal, 1, -1, the
# largest, its negative, and two others
SPECIAL = (0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x2800, 0x5640)


def made_up_blocks(name, count, seed=0, scale=None):
    """count blocks of the type called name: random bytes (any bytes are a block), every byte value in every place of
    the first 256, and finite float16 scales, the first few of every kind (scale: all of them that one instead)."""
    size = SOURCES[name].size
    blocks = np.random.default_rng([seed, SOURCES[name].ggml]).integers(0, 256, (count, size), dtype=np.uint8)
    for place in range(size):
        blocks[:256, place] = ((np.arange(256) + 7 * place) % 256)[:count]
    if name == "IQ1_M":
        d = np.array(SPECIAL if scale is None else [scale], dtype=np.uint16)
        some = slice(0, len(SPECIAL)) if scale is None else slice(None)
        for word in range(4):
            blocks[some, 49 + 2 * word] = (blocks[some, 49 + 2 * word] & 15) | ((d >> 4 * word & 15) << 4).astype(np.uint8)
        blocks[:, 55] &= 0xBF  # not an infinity or a NaN
    for place in SCALES.get(name, (0,)):
        blocks[:, place + 1] &= 0xBF
        if scale is None:
            blocks[:len(SPECIAL), place:place + 2] = np.array(SPECIAL, dtype="<u2").view(np.uint8).reshape(-1, 2)[:count]
        else:
            blocks[:, place:place + 2] = np.array([scale], dtype="<u2").view(np.uint8)
    return blocks


@pytest.mark.parametrize("name", SLOW)
def test_a_reader_reads_what_the_form_of_its_block_says_one_weight_at_a_time(name):
    """Every entry of the type's codebook, every scale of a group, every stored sign, both shifts, float16 scales of
    every finite kind: the same float32, to the bit and to the sign of a zero."""
    slow, count, through = SLOW[name]
    blocks = made_up_blocks(name, count)
    seen = {key: set() for key in ("index", "scale", "minimum", "signs", "shift")}
    expected = np.array([slow(bytes(block), seen) for block in blocks], dtype=np.float32).reshape(-1)
    assert {key: len(values) for key, values in seen.items() if values} == through
    got = SOURCES[name].read(blocks.tobytes())
    assert got.dtype == np.float32 and got.shape == (count * 256,) and np.isfinite(got).all()
    assert np.array_equal(got.view(np.uint32), expected.view(np.uint32))
    assert np.unique(got).size > 1000
    # however many are read at once, and whatever holds the bytes
    few = SUPERBLOCKS[name][2]
    for held in (bytearray(blocks[:3].tobytes()), memoryview(blocks[:3].tobytes())):
        assert np.array_equal(few(held).view(np.uint32), expected[:768].view(np.uint32))
    assert SOURCES[name].read(b"").shape == (0,)
    with pytest.raises(ValueError):
        SOURCES[name].read(blocks.tobytes()[:-1])


def test_blocks_are_read_a_few_at_a_time_to_the_same_values(monkeypatch):
    blocks = made_up_blocks("IQ2_XS", 300)
    whole = SOURCES["IQ2_XS"].read(blocks.tobytes())
    monkeypatch.setattr(superblocks, "AT_ONCE", 7)
    assert np.array_equal(SOURCES["IQ2_XS"].read(blocks.tobytes()).view(np.uint32), whole.view(np.uint32))


def test_the_tables_are_the_ones_they_were_taken_as():
    """The grids as llama.cpp's header has them (an entry a little-endian word, a value a byte): their sizes, the few
    values each is made of, and the hash of their bytes, which tests/fork_readers.py held to ggml-common.h itself. The
    eighth sign: an even number of minus signs."""
    shapes = {"IQ2XXS": (256, 8), "IQ2XS": (512, 8), "IQ2S": (1024, 8), "IQ3XXS": (256, 4), "IQ3S": (512, 4), "IQ1S": (2048, 8)}
    hashes = {"IQ2XXS": "05826b5d3e472a3a", "IQ2XS": "06e47aaca60b4dc1", "IQ2S": "e1aa1473412b0552",
              "IQ3XXS": "46e35f5a997efdee", "IQ3S": "bd1af4945a1717c6", "IQ1S": "07540ffc1aeaf6ad"}
    assert set(grids.DIGITS) == set(shapes)
    for name, shape in shapes.items():
        grid = grids.grid(name)
        assert grid.shape == shape and grid.dtype == np.int8
        assert sorted(set(grid.reshape(-1).tolist())) == list(grids.LEVELS[name])
        assert len(set(map(tuple, grid.tolist()))) == shape[0], "two entries of a grid are the same"
        assert hashlib.sha256(grid.tobytes()).hexdigest()[:16] == hashes[name], name
    assert grids.IQ4NL.tolist() == [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
    assert superblocks.SEVEN.tolist() == [seven(bits) for bits in range(128)]
    assert all(bin(byte).count("1") % 2 == 0 for byte in superblocks.SEVEN.tolist())
    assert 33808 == sum(grids.grid(name).size for name in shapes) + 16



def test_the_ten_types_are_the_tables_and_a_gguf_says_them_by_ggmls_numbers():
    ids = {"Q2_K": 10, "Q4_K": 12, "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19, "IQ3_S": 21, "IQ2_S": 22,
           "IQ4_XS": 23, "IQ1_M": 29}
    sizes = {"Q2_K": 84, "Q4_K": 144, "IQ2_XXS": 66, "IQ2_XS": 74, "IQ3_XXS": 98, "IQ1_S": 50, "IQ3_S": 110, "IQ2_S": 82,
             "IQ4_XS": 136, "IQ1_M": 56}
    assert set(SUPERBLOCKS) == set(SLOW) == set(ids)
    for name in ids:
        source = SOURCES[name]
        assert (source.ggml, source.values, source.size, source.kernel) == (ids[name], 256, sizes[name], None)
        assert GGUF_TENSORS[ids[name]] == name and source.read is SUPERBLOCKS[name][2]
        assert source.bytes(512) == 2 * sizes[name]


def test_what_is_read_of_blocks_made_the_same_everywhere_is_the_fixtures():
    """tests/fixtures/superblocks.json, which tests/smoke.mjs asks of the readers inside Pyodide (integers of 32 bits):
    it is what the readers of this (64-bit) NumPy read. Written by `python tests/test_superblocks.py`."""
    assert json.loads(FIXTURE.read_text())["read"] == read_of_the_cases()


def read_of_the_cases():
    with np.errstate(invalid="ignore"):  # (some of the float16 scales are infinities and NaNs)
        return {name: superblock_cases.digest(SOURCES[name].read(superblock_cases.blocks(name, SOURCES[name].size)))
                for name in SUPERBLOCKS}


# ---- a GGUF of them
def blocks_of(name, scale=0x0C00):
    """What qwen35_gguf() makes a matrix's blocks with: blocks of the type called name whatever the values (any bytes
    are a block: no quantizer is needed), with a small float16 scale (0x0C00 is 2 ** -12) so that the model's numbers stay
    finite, and what they stand for: by the reader, which the test above holds to the form."""
    def blocks(values):
        raw = made_up_blocks(name, values.size // 256, seed=values.shape[0], scale=scale).tobytes()
        return raw, SOURCES[name].read(raw).reshape(values.shape)
    return blocks, SOURCES[name].ggml


# as the real file: rows of whole blocks of 256 whatever the matrix (dim, the heads' widths, the hidden size), three
# value heads to a key head, a full-attention layer every fourth, a classifier of its own
SALUKI = dict(dim=256, hidden_dim=512, n_layers=8, every=4, n_heads=4, n_kv_heads=2, head_dim=64, key_heads=2,
              value_heads=6, key_dim=64, value_dim=128, conv=4, vocab_size=40, shared=False, prefix="model.")


def as_saluki(seen):
    """A type to a tensor: the two small matrices of a linear-attention layer's gates Q8_0, every other matrix one of
    the ten by its layer and its role, so that one role's matrices are of several types (as the real file's are)."""
    order = list(SLOW)

    def matrices(gguf):
        if gguf.split(".")[-2] in ("ssm_alpha", "ssm_beta"):
            return q8_0_blocks, 8
        parts = gguf.split(".")
        layer = int(parts[1]) if parts[0] == "blk" else 0
        name = order[(3 * layer + sum(map(ord, parts[-2]))) % len(order)]
        seen.setdefault(parts[-2], set()).add(name)
        return blocks_of(name)
    return matrices


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_a_gguf_of_the_ten_types_is_the_safetensors_conversion_of_what_its_blocks_stand_for(dtype, monkeypatch):
    """A small Qwen3.5 as Underdog Saluki 27B's GGUF is: llama.cpp's names and order (the value heads tiled, three to
    a key head: the columns of ssm_out move by half a block), every matrix of one of the ten types and the types
    differing from layer to layer. Its checkpoint is that of a safetensors file of the values the blocks stand for, to
    the byte, and the engine reads it and gives finite logits."""
    monkeypatch.setattr("convert.stream.PIECE", 70000)
    seen = {}
    config, file, same = qwen35_gguf(matrices=as_saluki(seen), **SALUKI)
    assert set().union(*seen.values()) == set(SLOW) and all(len(kinds) > 1 for role, kinds in seen.items() if role not in ("token_embd", "output"))
    header = json.loads(gguf_weights(file, json.dumps(config))[0])
    assert {info["dtype"] for name, info in header.items() if name != "__metadata__"} == {"F32", "Q8_0", *SLOW}
    vocabulary = unigram(config["text_config"]["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
    expected = safetensors_conversion(same, config, vocabulary, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint) and got.options == expected.options
    assert got.options["arch"] == "qwen35" and got.options["linear"]["value_heads"] == 6
    model = Llama(bytes(got.checkpoint), bytes(got.tokenizer), **got.options)
    logits = [model.forward(token, position) for position, token in enumerate((3, 9, 27, 5))]
    assert all(np.isfinite(row).all() and np.ptp(row) > 0 for row in logits)


def blocks_of_any(name):
    """Blocks of the type for a matrix of any size (enough bytes for its values rounded up to whole blocks)."""
    def blocks(values):
        return bytes(-(-values.size // 256) * SOURCES[name].size), np.zeros(values.shape, dtype=np.float32)
    return blocks, SOURCES[name].ggml


def test_a_tensor_whose_rows_are_not_blocks_of_256_is_refused():
    """As a Q8_0's of 32 and a PQ2_0's of 128: ggml itself requires it, and rows of 128 would be read at the wrong
    offsets and write nonsense instead of failing."""
    for name in SLOW:
        file = qwen35_gguf(matrices=blocks_of_any(name), dim=128, hidden_dim=256, head_dim=32, value_dim=64, key_dim=32)[1]
        with pytest.raises(ValueError, match=f"{name} with rows of 128, which is not a multiple of 256"):
            Conversion.from_gguf(file)


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps({**json.loads(FIXTURE.read_text()), "read": read_of_the_cases()}, indent=1) + "\n")
