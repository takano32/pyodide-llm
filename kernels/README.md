# SIMD kernels

`kernel.ts` and `kernel_relaxed.ts` (AssemblyScript) are the numeric core of the site: float32 and int8 matmul,
activation quantization, rmsnorm, RoPE, attention, SwiGLU, the residual add, and sampling (repetition penalty, softmax,
top-p). `public/llama2_numpy.py` loads
them with `ctypes.CDLL` (`load_kernels`) and drives them from Python (`Llama.kernel_forward`): Python keeps
sequencing the layers, NumPy keeps owning the memory, the kernels get addresses and work in place. Nothing is
copied and there is no JavaScript glue. When they cannot be loaded, NumPy does the math as before.

`make kernels` compiles them into `public/simdkernel.so` and `public/simdkernel_relaxed.wasmlib` (a few KB, not
committed). `build.py` needs no Emscripten: it compiles with AssemblyScript and prepends the `dylink.0` section
that makes a module an Emscripten side module. Verified to load in Pyodide 0.29.4 and 314.0.7.

## Measured (2026-09-19, phone-class ARM CPU, one thread)

| model | NumPy | kernels |
|---|---:|---:|
| stories15M float32, Pyodide in Node | 53 tok/s | 200 tok/s, the same text |
| stories15M int8, Pyodide in Node | 55 | 351 |
| tiny-lm int8, greedy, Pyodide in Node | 56 | 422 |
| llm-jp-3-150m int8, Pyodide in Node | 9.3 tok/s, 897 MB of WASM heap | 81 tok/s, 283 MB |
| Chromium: stories15M float32 / int8 | 50 | 186 / 296 |
| Chromium: tiny-lm int8, sampled with a repetition penalty | 43 | 252-274 (171 while NumPy still did the sampling) |
| Chromium: llm-jp-3-150m int8, sampled | 8.5 | 61-66 (47 while NumPy still did the sampling) |
| Chromium: stories3_5M / stories260K float32 (grouped-query attention) | 141 / 268 | 402 / 951 |

For comparison: native llama2.c with `gcc -Ofast` runs stories15M at 214 tok/s on the same machine. Firefox 150
loads the kernels too (all of its WebAssembly was 5-7x slower on the test machine, NumPy included). Safari was
never tested. Overheads inside Pyodide: one ctypes call 3-12 us, `array.ctypes.data` about 4 us (addresses are
taken once), one small NumPy call 1.5-2.5 us; a token takes about 100 kernel calls.

## int8

Weights as `quantize.py` writes them: int8 values in groups of 32 with one float32 scale per group; they stay
views into the downloaded buffer and are never widened. Per matmul the input is quantized once
(`quantize_x`, shared by q/k/v and by w1/w3) and multiplied by `matmul_q8`. With relaxed SIMD, `matmul_q8r`
multiplies int8 by 7-bit unsigned values: activations get a bias of 64, and `corrections = scale * sum of the
group` takes it out again. The relaxed module is loaded in `try/except`: a browser without relaxed SIMD
(shipping Safari) refuses to compile it, which arrives in Python as `OSError` (verified by switching the
feature off in Firefox), and int8 then runs on `matmul_q8`.

## Rules that are easy to break

- **No static data.** The side module has no relocations, so a data segment would be written over Pyodide's own
  memory. That rules out AssemblyScript's std math (`Mathf.exp` uses tables), strings and asserts; `build.py`
  refuses a module with a data section. `fexp` in `kernel.ts` is the table-free replacement.
- **Keep every array alive whose address a kernel gets.** The kernels only know numbers: a scratch array that
  no Python object refers to any more is freed, and the kernel then writes into whatever lives there next. It
  shows up as a rare `memory access out of bounds` (`_kernel_buffers`, `_sampler_buffers` in the engine).
- Mutable globals are fine (they are no data segment): `sample` keeps the state of its partial sort in three.
- `sample` sorts only as far as the nucleus reaches and gets its random number from Python, so a seed gives the
  same text again. With the same random number it picks the token that `Llama.sample` (NumPy) picks.
- The relaxed module must not be named `*.so` if it ever ships inside a wheel: Pyodide pre-loads every `.so`.
- `attention` expects the KV cache as `[seq][kv_heads * head_size]` per layer (the NumPy forward uses another
  layout) and handles grouped-query attention and any head size. int8 models whose row lengths are not
  multiples of 32 run on NumPy.
- Chromium compiles at most 8 MB synchronously on the main thread; these modules are a few KB, and the site
  runs Pyodide in a worker anyway.
- The ABI this relies on is Emscripten's dynamic linking, not the CPython C API, so one binary has served every
  Pyodide version so far. A future Emscripten could break it, which is why NumPy stays as the fallback and why
  `tests/smoke.mjs` checks both paths against the latest Pyodide before every deployment.
- `?kernel=off` in the page URL forces NumPy, to compare.
