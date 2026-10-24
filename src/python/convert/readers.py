# How the weights of a file are read: the types a file stores a tensor in (float32, float16, bfloat16, GGUF's Q8_0, the
# two ternary types, llama.cpp's K- and i-quantized types that convert/superblocks.py reads, T.362, and its older types of
# blocks of 32, T.505.1), one entry for each (SOURCES), to float32. A new one is a reader here and its entry (T.359): the GGUF's type id, the size of its
# blocks, the refusal that lists what is read, and its reader on the kernels come from the entry.
from typing import Callable, NamedTuple

import numpy as np

from convert.grids import IQ4NL
from convert.superblocks import SUPERBLOCKS, VALUES, WIDENED
from engine.kernels import kernel_wideners
from engine.packing import TERNARY_VALUES


def bfloat16(raw):
    # NumPy has no bfloat16, but a bfloat16 is exactly the upper half of a float32: widening is a shift
    wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    wide <<= 16
    return wide.view(np.float32)


def q8_0(raw):
    """GGUF's Q8_0 (T.74): blocks of 32 values, each a float16 scale and 32 int8. The same groups of 32 as this
    project's int8, so quantize() gets the very same int8 back: the scale of a block is its largest value / 127."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32)
    values = np.ascontiguousarray(blocks[:, 2:]).view(np.int8)
    return (values * scales).reshape(-1)


# the four values of each byte of PQ2_0, the lowest two bits first, as one little-endian word of four int8
PQ2_0_CODES = TERNARY_VALUES.view("<u4").reshape(256)


def pq2_0(raw):
    """Prism ML's PQ2_0 (T.235: Ternary-Bonsai's GGUFs, ggml type 142): blocks of 128 values, each a float16 scale d
    and 32 bytes of two bits a value, the first value in the lowest bits of the first byte. A value is (code - 1) * d:
    -d, 0 or +d in a ternary model, whose files leave the code 3 (+2 d) unused. The form is that of block_pq2_0 and
    dequantize_row_pq2_0() of the fork of llama.cpp that reads these files (MIT; no line of it is copied):
    https://github.com/PrismML-Eng/llama.cpp/blob/88c4bc60b9c9578f134385be9535e853f2db9b9f/ggml/src/ggml-common.h#L199-L207
    and ggml/src/ggml-quants.c#L494-L511 there.

    The engine's int8 holds a ternary block without loss of its values: quantize() makes every group of 32 of them
    -127, 0 and 127 and a scale of float32(d / 127), so what the forward pass multiplies is 127 * float32(d / 127)
    where the file says d, at most 6e-8 of d away (tests/test_gguf.py tries every float16 scale)."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32)
    values = PQ2_0_CODES[blocks[:, 2:]].view(np.int8)
    return (values * scales).reshape(-1)


def base3(packed, digits):
    """The digits (0, 1 or 2) of bytes that hold several in base 3, as PTQ1_0 packs them: (rows, bytes) -> (rows,
    digits, bytes), the first digit of every byte, then the second... A digit is the high byte of three times the
    byte, and what is left of the product goes on to the next one (the byte is ceil(256 v / 243) for the number v
    whose base 3 digits they are, the first the most significant)."""
    left, out = packed.astype(np.uint16), np.empty((packed.shape[0], digits, packed.shape[1]), dtype=np.uint8)
    for digit in range(digits):
        left *= 3
        out[:, digit] = left >> 8
        left &= 255
    return out


def ptq1_0(raw):
    """Prism ML's PTQ1_0 (T.230: Ternary Bonsai 2's smaller GGUF, ggml type 143): blocks of 128 ternary values in 28
    bytes, 1.75 bits a value. 24 bytes of five values each in base 3, 2 bytes of four, and a float16 scale d at the end;
    a value is (digit - 1) * d. The values are not in the order of the bytes: the first 16 bytes hold values 16 n + m
    (digit n of byte m), the next 8 bytes values 80 + 8 n + m, the 2 bytes values 120 + 2 n + m. The form is that of
    block_ptq1_0 and dequantize_row_ptq1_0() of the fork of llama.cpp that reads these files (MIT; no line of it is
    copied): ggml/src/ggml-common.h#L209-L220 and ggml/src/ggml-quants.c#L2196-L2285 of
    https://github.com/PrismML-Eng/llama.cpp/tree/88c4bc60b9c9578f134385be9535e853f2db9b9f (upstream's TQ1_0 in
    groups of 128). docs/notes/t228-bonsai-2-2026-10-01.md has how it was held to the F16 file of the same model."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28)
    scales = np.ascontiguousarray(blocks[:, 26:]).view(np.float16).astype(np.float32)
    digits = np.concatenate([base3(blocks[:, :16], 5).reshape(-1, 80), base3(blocks[:, 16:24], 5).reshape(-1, 40),
                             base3(blocks[:, 24:26], 4).reshape(-1, 8)], axis=1)
    return ((digits.view(np.int8) - np.int8(1)) * scales).reshape(-1)


# ---- T.505.1: llama.cpp's types of blocks of 32 other than Q8_0, the older ones (a file of Q4_0 ...) and the ones it
# stores a matrix in whose rows are no whole blocks of 256 (src/llama-quant.cpp's tensor_type_fallback(): Q5_0 for
# Q4_K, Q5_1 for Q5_K, Q4_0 for Q2_K and Q3_K, IQ4_NL for the i-quantized types). The forms are those of the block_q*
# structs of ggml/src/ggml-common.h and of the dequantize_row_* functions of ggml/src/ggml-quants.c of llama.cpp (MIT,
# Copyright (c) 2023-2026 The ggml authors; no line of them is copied: the readers are NumPy's, written from the
# forms) at the commit convert/superblocks.py names. tests/fork_readers.py holds each to that C, to the bit. Every
# product is exact in float32 (a float16 times an integer of five bits at most), so the one rounding is the sum.
def of_32(size, read):
    """read((blocks, size) of uint8) -> (blocks, 32) float32, as a reader of the bytes of whole blocks."""
    def whole(raw):
        return read(np.frombuffer(raw, dtype=np.uint8).reshape(-1, size)).astype(np.float32, copy=False).reshape(-1)
    whole.__name__, whole.__doc__ = read.__name__, read.__doc__
    return whole


def half_at(blocks, at):
    """The little-endian float16 at byte at of every block, float32: (blocks, 1)."""
    return np.ascontiguousarray(blocks[:, at:at + 2]).view("<f2").astype(np.float32)


def fours(held):
    """(blocks, 16) bytes -> (blocks, 32): their low four bits, then their high four."""
    return np.concatenate([held & 15, held >> 4], axis=1)


def fifth(held):
    """Q5's fifth bits: four bytes, a little-endian word whose bit j is weight j's -> (blocks, 32) of 0 and 16."""
    return np.unpackbits(held, axis=1, bitorder="little") << 4


def q4_0(b):
    """18 bytes: d, qs[16]. A weight is d (q - 8) for its four bits q: the low halves of the bytes are the first 16
    weights, the high halves the last."""
    return half_at(b, 0) * (fours(b[:, 2:]).astype(np.int8) - np.int8(8))


def q4_1(b):
    """20 bytes: d, m, qs[16]. A weight is d q + m for its four bits q, laid out as Q4_0's."""
    return half_at(b, 0) * fours(b[:, 4:]).astype(np.float32) + half_at(b, 2)


def q5_0(b):
    """22 bytes: d, qh[4], qs[16]. A weight is d (q - 16) for its five bits q: four as Q4_0's, the fifth bit j of the
    word qh for weight j."""
    return half_at(b, 0) * ((fours(b[:, 6:]) | fifth(b[:, 2:6])).astype(np.int8) - np.int8(16))


def q5_1(b):
    """24 bytes: d, m, qh[4], qs[16]. A weight is d q + m for its five bits q, laid out as Q5_0's."""
    return half_at(b, 0) * (fours(b[:, 8:]) | fifth(b[:, 4:8])).astype(np.float32) + half_at(b, 2)


def iq4_nl(b):
    """18 bytes: d, qs[16]. A weight is d times IQ4NL's value (convert/grids.py's 16, IQ4_XS's) for its four bits,
    laid out as Q4_0's."""
    return half_at(b, 0) * IQ4NL[fours(b[:, 2:])]


def float32(raw):
    return np.frombuffer(raw, dtype=np.float32)


def float16(raw):
    return np.frombuffer(raw, dtype=np.float16)


class Source(NamedTuple):
    """A type a file stores a tensor in. name: what a safetensors file calls it, and gguf_model() a GGUF's. ggml: the
    type's id in a GGUF. A tensor is whole blocks of values values in size bytes (a row is whole blocks too). read(the
    bytes of whole blocks) -> the values, float32 or float16: NumPy's. kernel: the export of the SIMD kernels that
    reads it to the same float32 (kernel(float32 out, blocks in, how many blocks); kernel_readers()), where there is
    one. said: whether a refusal names it among the types that are read (read_types())."""
    name: str
    ggml: int
    values: int
    size: int
    read: Callable
    kernel: str = None
    said: bool = True

    def bytes(self, count):
        """The bytes count values take (whole blocks of them)."""
        return count // self.values * self.size


# (30 is BF16: Ternary Bonsai 2's two small matrices of the gates. 142 and 143 are PQ2_0 and PTQ1_0 of Prism ML's fork
# of llama.cpp. T.362: the ten of convert/superblocks.py, blocks of 256, NumPy's readers alone. T.505.1: its other three K
# types, which are read and never kept (WIDENED), and the five of blocks of 32 above. A refusal names none of the
# eighteen yet (said): its words are the owner's, and they are as they were before any of them was read (T.501). ggml's
# other types are refused: docs/notes/t505-gguf-alone-2026-10-13.md says which and why)
SOURCES = {source.name: source for source in (
    Source("F32", 0, 1, 4, float32),
    Source("F16", 1, 1, 2, float16),
    Source("BF16", 30, 1, 2, bfloat16, "widen_bf16"),
    Source("Q8_0", 8, 32, 34, q8_0, "widen_q8_0"),
    Source("PQ2_0", 142, 128, 34, pq2_0, "widen_pq2_0"),
    Source("PTQ1_0", 143, 128, 28, ptq1_0, "widen_ptq1_0"),
    *(Source(name, ggml, VALUES, size, read, said=False) for name, (ggml, size, read) in SUPERBLOCKS.items()),
    *(Source(name, ggml, VALUES, size, read, said=False) for name, (ggml, size, read) in WIDENED.items()),
    Source("Q4_0", 2, 32, 18, of_32(18, q4_0), said=False),
    Source("Q4_1", 3, 32, 20, of_32(20, q4_1), said=False),
    Source("Q5_0", 6, 32, 22, of_32(22, q5_0), said=False),
    Source("Q5_1", 7, 32, 24, of_32(24, q5_1), said=False),
    Source("IQ4_NL", 20, 32, 18, of_32(18, iq4_nl), said=False),
)}
# a GGUF's type id -> the name
GGUF_TENSORS = {source.ggml: name for name, source in SOURCES.items()}


def read_types():
    """The types that are read, as a refusal lists them."""
    names = [name for name, source in SOURCES.items() if source.said]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def source_of(name, dtype):
    """The entry of the type a tensor called name is stored in, or the refusal that says which types are read."""
    if dtype not in SOURCES:
        raise ValueError(f"{name} is stored as {dtype}: only {read_types()} are read.")
    return SOURCES[dtype]


def kernel_readers(path):
    """The readers of SOURCES that have a kernel, on the SIMD kernels at path (simdkernel.so): {the type's name:
    read}, the same float32 several times faster, for Stream's readers. None where the kernels cannot be loaded."""
    return kernel_wideners(path, [(name, source.kernel, source.values, source.size) for name, source in SOURCES.items()
                                  if source.kernel])
