# SIMD kernels for Pyodide (experiment)

The prototype behind TODO task T30. It is **not wired into the site**. Nothing here is built or deployed yet.

## What it shows

Python keeps sequencing the transformer, NumPy keeps owning the memory, and every numeric operation is a WASM
SIMD kernel loaded with `ctypes.CDLL`, which works in place on the NumPy buffers (no copies, no JavaScript glue).
Measured on 2026-09-19 with stories15M on a phone-class ARM CPU, single thread:

| setup | tokens/s |
|---|---:|
| NumPy only (what the site runs today) | 52 |
| NumPy + only `matmul_*` from the kernel | 119 float32 / 188 int8 / 209 int8 relaxed SIMD |
| every operation in the kernel (`example.py`) | 181 float32 / 282 int8 / 348 int8 relaxed SIMD |
| for comparison: JavaScript driving the same kernels | 170-334 |
| for comparison: native llama2.c, `gcc -Ofast` | 214 |

In headless Chromium 148, served like GitHub Pages: 166-196 float32, 270-359 int8. Firefox 150 worked too (on the
test machine all of its WebAssembly ran 5-7x slower, NumPy included). Safari was never tested.
Overheads inside Pyodide: one ctypes call 3-12 us, `array.ctypes.data` about 4 us (take addresses once), one small
NumPy call 1.5-2.5 us. The design makes roughly 90-110 kernel calls per token.

## Files

- `kernel.ts`: the kernels (AssemblyScript): float32 and int8 matmul, activation quantization, rmsnorm, RoPE,
  attention, SwiGLU, residual add, argmax.
- `kernel_relaxed.ts`: the int8 matmul on the relaxed-SIMD dot product. Separate, because a browser without
  relaxed SIMD (shipping Safari) refuses to compile a module that contains it.
- `build.py`: compiles both and turns them into Emscripten side modules by prepending a `dylink.0` section.
  No Emscripten needed. Verified to load in Pyodide 0.29.4 and 314.0.7.
- `example.py`: the float32 forward pass that drives `kernel.ts`.

## Build and try

```
cd experiments/simd-kernel
npm install --no-save assemblyscript
python3 build.py
```

Put `simdkernel.so` into Pyodide's file system (for example `pyodide.FS.writeFile("/home/pyodide/simdkernel.so", bytes)`),
then `ctypes.CDLL("/home/pyodide/simdkernel.so")` as in `example.py`. A correct build reproduces the NumPy
engine's greedy text for stories15M (`Once upon a time, there was a little girl named Lily. She loved to play
outside in the sunshine.`); int8 text differs after some tokens but stays coherent.

## int8

Weights as `quantize.py` writes them: int8 values in groups of 32 with one float32 scale per group. Per matmul:
`quantize_x(xq, xs, x, n, 0)` once per distinct input, then `matmul_q8(out, xq, xs, wq, ws, n, 0, d)`. The relaxed
variant quantizes with bias 64 (`quantize_x(..., 64)`) and needs one more array per matrix,
`wc = scale * sum of the int8 values of the group` (float32, same shape as the scales), passed to `matmul_q8r`.
Try to load `simdkernel_relaxed.wasmlib` in `try/except` and fall back to `matmul_q8`: the CompileError of a
browser without relaxed SIMD arrives as `OSError` (verified by switching the feature off in Firefox).

## Rules that are easy to break

- **No static data.** The side module has no relocations, so a data segment would be written over Pyodide's own
  memory. That rules out AssemblyScript's std math (`Mathf.exp` uses tables), strings and asserts; `build.py`
  refuses a module with a data section. `fexp` in `kernel.ts` is the table-free replacement.
- The relaxed module must not be named `*.so` if it ships inside a wheel: Pyodide pre-loads every `.so`.
- `attention` expects the KV cache as `[seq][heads * head_size]` per layer and no grouped-query attention;
  `llama2_numpy.py` stores `[kv_heads][seq][head_size]` and supports GQA (stories260K, stories3_5M need it).
- All row lengths must be multiples of 32 for int8 (true for tiny-lm, llm-jp-3-150m, stories15M and stories42M).
- Chromium compiles at most 8 MB synchronously on the main thread; these modules are a few KB, and the site
  runs Pyodide in a worker anyway.
- The ABI this relies on is Emscripten's dynamic linking, not the CPython C API, so one binary serves every
  Pyodide version so far. A future Emscripten could break it: the site must fall back to NumPy when loading fails.
