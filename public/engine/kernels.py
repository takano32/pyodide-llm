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

from engine.dtypes import DTYPES


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
                          penalize=[p, p, i32, ctypes.c_float, ctypes.c_float],
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
    """The packing of every quantized dtype (DTYPES' pack) on the SIMD kernels, for the converter's Writer:
    quantize_rows(float32 rows, the dtype's name) -> (the bytes of their values in rows of a group's, their float32
    scales), the same bytes as NumPy's and several times faster (T89: int8, six times; T98: int6, quantized and packed
    in one pass; T230: ternary, with the same refusal of values that are not ternary). Whole groups of 32 or more
    only: an int8 row of smaller groups is NumPy's (the Writer sees to it). None where the kernels cannot be loaded.
    Which kernel packs a dtype is its entry's packer."""
    kernels = load_kernels(path) if path else None
    if not kernels:
        return None

    # chosen once, not for every piece: of each dtype its group, the bytes of a group, the array's type, the kernel
    # and what it takes after the count (quantize_x is the activations' quantizer too, and takes their bias: none here)
    def packing(kind):
        packer = kernels[kind.packer]
        # (int8 values are the bytes of an int8 file as they are: the array says so, as NumPy's quantize() does)
        return (kind.group, kind.bits, np.int8 if kind.bits == 8 else np.uint8, packer,
                (0,) * (len(packer.argtypes) - 4), kind.refusal)

    packings = {name: packing(kind) for name, kind in DTYPES.items() if kind.packer in kernels}

    def quantize_rows(values, dtype="int8"):
        group_of, bits, stored_as, packer, more, refusal = packings[dtype]
        values = np.ascontiguousarray(values, dtype=np.float32)
        group = group_of(values.shape[-1])
        groups = values.size // group
        packed, scales = np.empty(groups * (group * bits // 8), dtype=stored_as), np.empty(groups, dtype=np.float32)
        if packer(packed.ctypes.data, scales.ctypes.data, values.ctypes.data, values.size, *more):
            raise ValueError(refusal)
        return packed.reshape(groups, -1), scales

    return quantize_rows


def kernel_wideners(path, types):
    """The converter's readers of the stored types that have a kernel (convert/readers.py's SOURCES, which calls this
    with them), on the SIMD kernels: {the type's name: read(bytes) -> float32}, the same float32 as NumPy's readers
    and several times faster (T123: bfloat16, a shift of every 16 bits; T136: GGUF's Q8_0; T273: Prism ML's two
    ternary types, most of the time of converting Ternary Bonsai 2 27B in the page).
    types: (the type's name, the kernel's export, the values of a block, the bytes of a block) of each. Every such
    kernel is export(float32 out, blocks in, how many blocks). A type whose export this build of the kernels has not
    is left out (NumPy reads it). None where the kernels cannot be loaded."""
    try:
        import ctypes

        lib = ctypes.CDLL(path) if path else None
    except Exception:
        lib = None
    if lib is None:
        return None

    def reader(name, widen, values, size):
        widen.argtypes, widen.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32], None

        def read(raw):
            blocks = np.frombuffer(raw, dtype=np.uint8)
            if blocks.size % size:
                raise ValueError(f"{name} data is not whole blocks of {size} bytes.")
            out = np.empty(blocks.size // size * values, dtype=np.float32)
            widen(out.ctypes.data, blocks.ctypes.data, blocks.size // size)
            return out

        return read

    return {name: reader(name, getattr(lib, export), values, size) for name, export, values, size in types
            if hasattr(lib, export)}
