# The WASM SIMD kernels as Python calls them (ctypes), and the converter's readers and quantizer on them.
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is
# derived from two works under the MIT License, whose notice follows: tairov/llama2.py
# (https://github.com/tairov/llama2.py; its LICENSE names no copyright holder) and karpathy/llama2.c
# (https://github.com/karpathy/llama2.c), Copyright (c) 2023 Andrej.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np

from engine.packing import NOT_TERNARY, TERNARY_GROUP


def load_kernels(path, without_relaxed=False):
    """The WASM SIMD kernels of kernels/*.ts, as ctypes functions, or None when they cannot be used.

    They are Emscripten side modules: ctypes.CDLL links them into Pyodide's own memory, so they work in place
    on NumPy arrays. Anything may go wrong here (no file, not Pyodide, a future Emscripten that loads side
    modules differently), and then NumPy does the work as before.
    """
    try:
        import ctypes

        lib = ctypes.CDLL(path)
        i32, p = ctypes.c_int32, ctypes.c_void_p
        signatures = dict(matmul_f32=[p, p, p, i32, i32, i32], quantize_x=[p, p, p, i32, i32], quantize6_x=[p, p, p, i32],
                          ternary_x=[p, p, p, i32],
                          matmul_q8=[p, p, p, p, p, i32, i32, i32], rmsnorm=[p, p, p, i32, ctypes.c_float], rope=[p, p, p, i32, i32, i32],
                          attention=[p, p, p, p, p, i32, i32, i32, i32, i32, i32],
                          attention_f16=[p, p, p, p, p, i32, i32, i32, i32, i32, i32], to_f16=[p, p, i32], from_f16=[p, p, i32], finite_f16=[p, i32],
                          swiglu=[p, p, p, i32], add_inplace=[p, p, i32],
                          rotate=[p, p, p, i32, i32], unrotate=[p, p, p, i32, i32],
                          add_columns=[p, p, p, i32, i32],
                          layernorm=[p, p, p, p, i32], gelu=[p, p, p, i32],
                          penalize=[p, p, i32, ctypes.c_float, ctypes.c_float], widen_bf16=[p, p, i32], widen_q8_0=[p, p, i32],
                          widen_pq2_0=[p, p, i32], widen_ptq1_0=[p, p, i32],
                          sample=[p, i32, ctypes.c_float, ctypes.c_float, ctypes.c_double, p, p, i32, ctypes.c_float])
        kernels = {}
        for name, argtypes in signatures.items():
            kernels[name] = getattr(lib, name)
            kernels[name].argtypes, kernels[name].restype = argtypes, i32 if name in ("sample", "finite_f16", "ternary_x") else None
    except Exception:
        return None
    if without_relaxed:
        return kernels
    try:
        # a browser without relaxed SIMD (shipping Safari) refuses to compile this one: then int8 uses matmul_q8
        relaxed = ctypes.CDLL(path.replace(".so", "_relaxed.wasmlib")).matmul_q8r
        relaxed.argtypes, relaxed.restype = [p, p, p, p, p, p, i32, i32, i32], None
        kernels["matmul_q8r"] = relaxed
    except Exception:
        pass
    return kernels


def kernel_quantizer(path):
    """llama2_convert.quantize() on the SIMD kernels (T89): int8 values in groups of 32 and one float32 scale per
    group, the same bytes as NumPy's, six times faster (quantize_x with no bias: the activations' quantizer is the
    same computation). For the converter's quantize_rows; None where the kernels cannot be loaded.
    dtype "int6": quantize6() and pack6() in one pass on the kernel quantize6_x (T98), the same bytes: the packed
    groups (24 bytes each) and their scales. dtype "ternary" (T230): ternary() on the kernel ternary_x, the same bytes
    (32 a group of 128) and scales, and the same refusal of values that are not ternary."""
    kernels = load_kernels(path) if path else None
    if not kernels:
        return None
    quantize_x, quantize6_x, ternary_x = kernels["quantize_x"], kernels["quantize6_x"], kernels["ternary_x"]

    def quantize_rows(values, dtype="int8"):
        values = np.ascontiguousarray(values, dtype=np.float32)
        if dtype == "ternary":
            packed = np.empty(values.size // 4, dtype=np.uint8)
            scales = np.empty(values.size // TERNARY_GROUP, dtype=np.float32)
            if ternary_x(packed.ctypes.data, scales.ctypes.data, values.ctypes.data, values.size):
                raise ValueError(NOT_TERNARY)
            return packed.reshape(-1, TERNARY_GROUP // 4), scales
        if dtype == "int6":
            packed = np.empty(values.size // 32 * 24, dtype=np.uint8)
            scales = np.empty(values.size // 32, dtype=np.float32)
            quantize6_x(packed.ctypes.data, scales.ctypes.data, values.ctypes.data, values.size)
            return packed.reshape(-1, 24), scales
        quantized = np.empty(values.size, dtype=np.int8)
        scales = np.empty(values.size // 32, dtype=np.float32)
        quantize_x(quantized.ctypes.data, scales.ctypes.data, values.ctypes.data, values.size, 0)
        return quantized.reshape(-1, 32), scales

    return quantize_rows


def kernel_widener(path):
    """llama2_convert.bfloat16() on the SIMD kernels (T123): the same float32, a shift of every 16 bits, several
    times faster than NumPy's two passes. For the converter's bfloat16; None where the kernels cannot be loaded."""
    kernels = load_kernels(path) if path else None
    if not kernels:
        return None
    widen = kernels["widen_bf16"]

    def bfloat16(raw):
        halves = np.frombuffer(raw, dtype=np.uint16)
        out = np.empty(halves.size, dtype=np.float32)
        widen(out.ctypes.data, halves.ctypes.data, halves.size)
        return out

    return bfloat16


def kernel_q8_0(path):
    """llama2_convert.q8_0() on the SIMD kernels (T136): GGUF's Q8_0 blocks widened to the same float32, each int8
    times its block's float16 scale. For the converter's q8_0; None where the kernels cannot be loaded."""
    kernels = load_kernels(path) if path else None
    if not kernels:
        return None
    widen = kernels["widen_q8_0"]

    def q8_0(raw):
        blocks = np.frombuffer(raw, dtype=np.uint8)
        if blocks.size % 34:
            raise ValueError("Q8_0 data is not whole blocks of 34 bytes.")
        out = np.empty(blocks.size // 34 * 32, dtype=np.float32)
        widen(out.ctypes.data, blocks.ctypes.data, blocks.size // 34)
        return out

    return q8_0


def kernel_ternary_readers(path):
    """llama2_convert.pq2_0() and ptq1_0() on the SIMD kernels (T273): the blocks of Prism ML's two ternary types
    widened to the same float32, many times faster than NumPy's passes (most of the time of converting Ternary
    Bonsai 2 27B in the page). For the converter's readers, by the GGUF's type; None where the kernels cannot be
    loaded."""
    kernels = load_kernels(path) if path else None
    if not kernels:
        return None

    def reader(kind, widen, size):
        def read(raw):
            blocks = np.frombuffer(raw, dtype=np.uint8)
            if blocks.size % size:
                raise ValueError(f"{kind} data is not whole blocks of {size} bytes.")
            out = np.empty(blocks.size // size * 128, dtype=np.float32)
            widen(out.ctypes.data, blocks.ctypes.data, blocks.size // size)
            return out

        return read

    return {"PQ2_0": reader("PQ2_0", kernels["widen_pq2_0"], 34), "PTQ1_0": reader("PTQ1_0", kernels["widen_ptq1_0"], 28)}
