# llama.cpp's K- and i-quantized types read to float32 (T362): Q2_K, Q4_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
# IQ3_S and IQ4_XS, the ten a GGUF of Underdog Saluki 27B is made of. Every one of them holds 256 weights to a block (a
# "super-block"), with a float16 scale d of the block and small integer scales of its groups of 16 or 32: a weight is
# d times its group's integer times a small integer of its own, which the i-quantized types look up 8 or 4 at a time in
# a codebook (convert/grids.py) and give a sign.
#
# The forms are those of the block_* structs of ggml/src/ggml-common.h and of the dequantize_row_* functions of
# ggml/src/ggml-quants.c of llama.cpp (MIT, Copyright (c) 2023-2026 The ggml authors; no line of them is copied here:
# the readers are NumPy's, written from the forms) at the commit 4413a192410fad1f054e9216114087a3e01527a0 of
# https://github.com/ggml-org/llama.cpp. tests/fork_readers.py holds each reader to that C, to the bit;
# docs/notes/t361-saluki-2026-10-13.md has the forms in a table, and how a real file's rows were held to the original
# model's.
#
# Every product below is exact in float32 (a float16's 11 bits times integers of a few bits), so the order of the
# products does not matter to the value; it is llama.cpp's all the same, which settles the sign of a zero. The one
# rounding is the subtraction of the two K types.
import numpy as np

from convert.grids import IQ4NL, grid
from engine.dtypes import BLOCK, BLOCKS

VALUES = BLOCK  # the weights of a block, whatever the type
# how many blocks are read at once: what is in between (the indices, the looked-up values) stays a few megabytes
AT_ONCE = 2048
# the eighth sign of a group of 8 is not stored where the signs take 7 bits: the number of minus signs is even
# (llama.cpp's ksigns_iq2xs, which is this formula)
SEVEN = np.array([i | (bin(i).count("1") & 1) << 7 for i in range(128)], dtype=np.uint8)
BITS = (1 << np.arange(8)).astype(np.uint8)
EIGHTH = np.float32(0.125)  # IQ1's shift of a group's values
SHIFTS = {step: (step * np.arange(4)).astype(np.uint32) for step in (2, 3, 7)}


def halves(raw):
    """Little-endian float16 as float32."""
    return np.ascontiguousarray(raw).view("<f2").astype(np.float32)


def words(raw, size):
    """Little-endian unsigned integers of size bytes."""
    return np.ascontiguousarray(raw).view(f"<u{size}")


def signs(held):
    """A byte of signs -> its 8 factors: bit j set is a minus for the j-th value."""
    return np.where(held[..., None] & BITS, np.float32(-1), np.float32(1))


def nibbles(held):
    """(blocks, n) bytes -> (blocks, 2 n): the low four bits of a byte, then its high four."""
    return np.stack([held & 15, held >> 4], axis=2).reshape(len(held), -1)


def odd(scale):
    """2 s + 1 of a group's integer scale s, float32."""
    return (2 * scale.astype(np.int32) + 1).astype(np.float32)


def q2_k(b):
    """84 bytes: scales[16], qs[64], d, dmin. A group of 16 has a byte of scales: the low four bits its scale sc, the
    high four its minimum m, and a weight is d sc q - dmin m for its two bits q. The weights are not in the order of
    the bytes: a block is two halves of 128, and of a half's 32 bytes, bits 2 s and 2 s + 1 of byte b are weight
    32 s + b of the half."""
    n = len(b)
    scales, d, least = b[:, :16], halves(b[:, 80:82]), halves(b[:, 82:84])
    q = (b[:, 16:80].reshape(n, 2, 1, 32) >> SHIFTS[2].astype(np.uint8).reshape(1, 1, 4, 1)) & 3
    scale, minimum = d * (scales & 15).astype(np.float32), least * (scales >> 4).astype(np.float32)
    return scale[:, :, None] * q.reshape(n, 16, 16).astype(np.float32) - minimum[:, :, None]


def q4_k(b):
    """144 bytes: d, dmin, scales[12], qs[128]. Eight groups of 32, each a scale sc and a minimum m of six bits in the
    12 bytes: for the first four groups the low six bits of bytes j and j + 4; for the last four the low and the
    high four bits of byte j + 4, with the top two bits of bytes j - 4 and j above them. A weight is d sc q - dmin m
    for its four bits q; of every 32 bytes the low halves are 32 weights and the high halves the next 32."""
    n = len(b)
    d, least, s = halves(b[:, :2]), halves(b[:, 2:4]), b[:, 4:16]
    sc = np.concatenate([s[:, :4] & 63, (s[:, 8:] & 15) | ((s[:, :4] >> 6) << 4)], axis=1)
    m = np.concatenate([s[:, 4:8] & 63, (s[:, 8:] >> 4) | ((s[:, 4:8] >> 6) << 4)], axis=1)
    q = b[:, 16:].reshape(n, 4, 32)
    q = np.stack([q & 15, q >> 4], axis=2).reshape(n, 8, 32)
    scale, minimum = d * sc.astype(np.float32), least * m.astype(np.float32)
    return scale[:, :, None] * q.astype(np.float32) - minimum[:, :, None]


def seven_bit_signs(aux):
    """The signs of four groups of 8 held as four indices of 7 bits in the low 28 bits of a word: (.., 4, 8) factors."""
    return signs(SEVEN[(aux[..., None] >> SHIFTS[7]) & 127])


def iq2_xxs(b, grid=grid("IQ2XXS")):
    """66 bytes: d, then 8 bytes for each 32 weights: four indices of 8 bits into the grid, each 8 weights, and a word
    whose low 28 bits are their signs (seven_bit_signs) and whose top four the group's scale s. A weight is
    d (s + 1/2) / 4 times the grid's value, with its sign."""
    n = len(b)
    groups = b[:, 2:].reshape(n, 8, 8)
    aux = words(groups[:, :, 4:], 4).reshape(n, 8)
    scale = halves(b[:, :2]) * (np.float32(0.5) + (aux >> 28).astype(np.float32)) * np.float32(0.25)
    return scale[:, :, None, None] * grid[groups[:, :, :4]].astype(np.float32) * seven_bit_signs(aux)


def halved(d, scales):
    """IQ2_XS's and IQ2_S's scales: a byte for each 32 weights, the low four bits the scale s of the first 16 and the
    high four of the last 16, as d (s + 1/2) / 4 for each group of 8: (blocks, 32, 1)."""
    scale = d * (np.float32(0.5) + nibbles(scales).astype(np.float32)) * np.float32(0.25)
    return np.repeat(scale, 2, axis=1)[:, :, None]


def iq2_xs(b, grid=grid("IQ2XS")):
    """74 bytes: d, 32 words of 16 bits, scales[8]. A word is 8 weights: its low 9 bits the index into the grid, its
    high 7 their signs. The scales and the formula: halved() and iq2_xxs."""
    q = words(b[:, 2:66], 2)
    return halved(halves(b[:, :2]), b[:, 66:]) * grid[q & 511].astype(np.float32) * signs(SEVEN[q >> 9])


def iq2_s(b, grid=grid("IQ2S")):
    """82 bytes: d, qs[32], signs[32], qh[8], scales[8]. 8 weights are an index of 10 bits into the grid, its low 8 in
    qs and its high 2 in qh (a byte for each 32 weights: bits 2 l and 2 l + 1 for the l-th index), and a whole byte of
    signs. The scales and the formula: halved() and iq2_xxs."""
    high = (b[:, 66:74, None].astype(np.uint16) >> SHIFTS[2].astype(np.uint16)) & 3
    index = b[:, 2:34] | (high.reshape(len(b), 32) << 8)
    return halved(halves(b[:, :2]), b[:, 74:]) * grid[index].astype(np.float32) * signs(b[:, 34:66])


def iq3_xxs(b, grid=grid("IQ3XXS")):
    """98 bytes: d, qs[64], 8 words of 32 bits. A byte of qs is an index into the grid, 4 weights; a word is for 32
    weights, as iq2_xxs's: the signs of its four groups of 8 and the scale s. A weight is d (s + 1/2) / 2 times the
    grid's value, with its sign."""
    n = len(b)
    aux = words(b[:, 66:], 4)
    scale = halves(b[:, :2]) * (np.float32(0.5) + (aux >> 28).astype(np.float32)) * np.float32(0.5)
    return scale[:, :, None, None] * grid[b[:, 2:66]].reshape(n, 8, 4, 8).astype(np.float32) * seven_bit_signs(aux)


def iq3_s(b, grid=grid("IQ3S")):
    """110 bytes: d, qs[64], qh[8], signs[32], scales[4]. 4 weights are an index of 9 bits into the grid: a byte of
    qs, and above it bit k of the byte of qh of its 32 weights, for the k-th index of them. 8 weights have a byte of
    signs. A byte of scales is for 64 weights: its low four bits the scale s of the first 32, its high four of the
    last. A weight is d (2 s + 1) times the grid's value, with its sign."""
    n = len(b)
    high = (b[:, 66:74, None].astype(np.uint16) >> np.arange(8, dtype=np.uint16)) & 1
    index = b[:, 2:66] | (high.reshape(n, 64) << 8)
    scale = halves(b[:, :2]) * odd(nibbles(b[:, 106:]))
    return scale[:, :, None, None] * grid[index].reshape(n, 8, 4, 8).astype(np.float32) * signs(b[:, 74:106].reshape(n, 8, 4))


def iq1_s(b, grid=grid("IQ1S")):
    """50 bytes: d, qs[32], 8 words of 16 bits (qh). 8 weights are an index of 11 bits into the grid of -1, 0 and 1:
    a byte of qs and, above it, three bits of the word of its 32 weights (bits 3 l to 3 l + 2 for the l-th index). The
    word's bits 12 to 14 are the scale s of the 32 and its top bit the sign of their shift: a weight is d (2 s + 1)
    times (the grid's value + 1/8), or - 1/8 where the bit is set."""
    n = len(b)
    qh = words(b[:, 34:], 2)
    index = b[:, 2:34].reshape(n, 8, 4) | (((qh[:, :, None] >> SHIFTS[3].astype(np.uint16)) & 7) << 8)
    scale = halves(b[:, :2]) * odd((qh >> 12) & 7)
    shift = np.where(qh & 0x8000, -EIGHTH, EIGHTH)
    return scale[:, :, None, None] * (grid[index].astype(np.float32) + shift[:, :, None, None])


def iq1_m(b, grid=grid("IQ1S")):
    """56 bytes: qs[32], qh[16], 4 words of 16 bits (scales), and no d of its own: the float16 d is the top four bits
    of the four words, the first word's the lowest. 8 weights are an index of 11 bits into IQ1_S's grid: a byte of qs
    and the low three bits of their four bits of qh (a byte of qh is for 16 weights, its low half the first 8), whose
    fourth bit is the sign of the shift of those 8. A word's low 12 bits are the scales s of its four groups of 16.
    A weight is d (2 s + 1) times (the grid's value +- 1/8)."""
    n = len(b)
    held = words(b[:, 48:], 2)
    d = ((held[:, 0] >> 12) | ((held[:, 1] >> 8) & 0x00f0) | ((held[:, 2] >> 4) & 0x0f00) | (held[:, 3] & 0xf000))
    scale = halves(d.astype("<u2"))[:, None] * odd((held[:, :, None] >> SHIFTS[3].astype(np.uint16)) & 7).reshape(n, 16)
    four = nibbles(b[:, 32:48])
    index = b[:, :32] | ((four.astype(np.uint16) & 7) << 8)
    shift = np.where(four & 8, -EIGHTH, EIGHTH)
    return scale[:, :, None, None] * (grid[index].astype(np.float32) + shift[:, :, None]).reshape(n, 16, 2, 8)


def iq4_xs(b):
    """136 bytes: d, a word of 16 bits (scales_h), scales_l[4], qs[128]. Eight groups of 32, each a scale ls of six
    bits: four in scales_l (a byte for two groups, the low half first) and two above them in the word (bits 2 g and
    2 g + 1 for group g). A weight is d (ls - 32) times IQ4NL's value for its four bits; of a group's 16 bytes the low
    halves are its first 16 weights and the high halves its last."""
    n = len(b)
    high = (words(b[:, 2:4], 2).astype(np.int32) >> (2 * np.arange(8, dtype=np.int32))) & 3
    scale = halves(b[:, :2]) * ((nibbles(b[:, 4:8]).astype(np.int32) | (high << 4)) - 32).astype(np.float32)
    q = b[:, 8:].reshape(n, 8, 16)
    return scale[:, :, None] * IQ4NL[np.concatenate([q & 15, q >> 4], axis=2)].astype(np.float32)


def reader(size, read):
    """read (blocks of size bytes as (blocks, size) of uint8 -> their weights in any shape, float32) as a reader of
    the bytes of whole blocks: -> float32, 256 for each block. AT_ONCE blocks at a time."""
    def whole(raw):
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size)
        out = np.empty((len(blocks), VALUES), dtype=np.float32)
        for start in range(0, len(blocks), AT_ONCE):
            out[start:start + AT_ONCE] = read(blocks[start:start + AT_ONCE]).reshape(-1, VALUES)
        return out.reshape(-1)
    whole.__name__, whole.__doc__ = read.__name__, read.__doc__
    return whole


# the type's name in a GGUF -> (its id in ggml, the bytes of a block, the reader). The bytes of a block are the
# engine's to say (engine/dtypes.py's BLOCKS, which lays a checkpoint's rows of these types out, T363), and the
# engine reads such a row with the reader here, by the name.
SUPERBLOCKS = {name: (ggml, BLOCKS[name].block, reader(BLOCKS[name].block, read)) for name, ggml, read in (
    ("Q2_K", 10, q2_k), ("Q4_K", 12, q4_k), ("IQ2_XXS", 16, iq2_xxs), ("IQ2_XS", 17, iq2_xs),
    ("IQ3_XXS", 18, iq3_xxs), ("IQ1_S", 19, iq1_s), ("IQ3_S", 21, iq3_s), ("IQ2_S", 22, iq2_s),
    ("IQ4_XS", 23, iq4_xs), ("IQ1_M", 29, iq1_m))}
