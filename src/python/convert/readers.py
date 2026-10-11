# How the weights of a file are read: the types a file stores a tensor in (float32, float16, bfloat16, GGUF's Q8_0 and
# the two ternary types), one entry for each (SOURCES), to float32. A new one is a reader here and its entry (T359):
# the GGUF's type id, the size of its blocks, the refusal that lists what is read, and its reader on the kernels come
# from the entry.
from typing import Callable, NamedTuple

import numpy as np

from engine.kernels import kernel_wideners
from engine.packing import TERNARY_VALUES


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


def float32(raw):
    return np.frombuffer(raw, dtype=np.float32)


def float16(raw):
    return np.frombuffer(raw, dtype=np.float16)


class Source(NamedTuple):
    """A type a file stores a tensor in. name: what a safetensors file calls it, and gguf_model() a GGUF's. ggml: the
    type's id in a GGUF. A tensor is whole blocks of values values in size bytes (a row is whole blocks too). read(the
    bytes of whole blocks) -> the values, float32 or float16: NumPy's. kernel: the export of the SIMD kernels that
    reads it to the same float32 (kernel(float32 out, blocks in, how many blocks); kernel_readers()), where there is
    one."""
    name: str
    ggml: int
    values: int
    size: int
    read: Callable
    kernel: str = None

    def bytes(self, count):
        """The bytes count values take (whole blocks of them)."""
        return count // self.values * self.size


# (30 is BF16: Ternary Bonsai 2's two small matrices of the gates. 142 and 143 are PQ2_0 and PTQ1_0 of Prism ML's fork
# of llama.cpp. ggml's other types, the K-quants among them, are refused)
SOURCES = {source.name: source for source in (
    Source("F32", 0, 1, 4, float32),
    Source("F16", 1, 1, 2, float16),
    Source("BF16", 30, 1, 2, bfloat16, "widen_bf16"),
    Source("Q8_0", 8, 32, 34, q8_0, "widen_q8_0"),
    Source("PQ2_0", 142, 128, 34, pq2_0, "widen_pq2_0"),
    Source("PTQ1_0", 143, 128, 28, ptq1_0, "widen_ptq1_0"),
)}
# a GGUF's type id -> the name
GGUF_TENSORS = {source.ggml: name for name, source in SOURCES.items()}


def read_types():
    """The types that are read, as a refusal lists them."""
    names = list(SOURCES)
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
