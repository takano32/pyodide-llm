# How the weights of a file are read: a stored type (float32, float16, bfloat16, GGUF's Q8_0 and the two ternary
# types) to float32, by its name.
import numpy as np

from llama2_numpy import TERNARY_VALUES


def bfloat16(raw):
    # NumPy has no bfloat16, but a bfloat16 is exactly the upper half of a float32: widening is a shift
    wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    wide <<= 16
    return wide.view(np.float32)


def q8_0(raw):
    """GGUF's Q8_0 (T74): blocks of 32 values, each a float16 scale and 32 int8. The same groups of 32 as this
    project's int8, so quantize() gets the very same int8 back: the scale of a block is its largest value / 127."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32)
    values = np.ascontiguousarray(blocks[:, 2:]).view(np.int8)
    return (values * scales).reshape(-1)


# the four values of each byte of PQ2_0, the lowest two bits first, as one little-endian word of four int8
PQ2_0_CODES = TERNARY_VALUES.view("<u4").reshape(256)


def pq2_0(raw):
    """Prism ML's PQ2_0 (T235: Ternary-Bonsai's GGUFs, ggml type 142): blocks of 128 values, each a float16 scale d
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
    """Prism ML's PTQ1_0 (T230: Ternary Bonsai 2's smaller GGUF, ggml type 143): blocks of 128 ternary values in 28
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


# bytes per value (Q8_0: 34 bytes for 32 of them, PQ2_0: 34 for 128, PTQ1_0: 28 for 128), and how to read them
READERS = {"F32": (4, lambda raw: np.frombuffer(raw, dtype=np.float32)),
           "F16": (2, lambda raw: np.frombuffer(raw, dtype=np.float16)), "BF16": (2, bfloat16),
           "Q8_0": (34 / 32, q8_0), "PQ2_0": (34 / 128, pq2_0), "PTQ1_0": (28 / 128, ptq1_0)}
# how many values a block of a GGUF's type holds: a row is whole blocks
BLOCKS = {"Q8_0": 32, "PQ2_0": 128, "PTQ1_0": 128}
