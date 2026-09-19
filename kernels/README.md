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
| Chromium: tiny-lm int8, sampled with a repetition penalty | 43 | 271-296 (171 while NumPy still did the sampling, 252-274 before T54) |
| Chromium: llm-jp-3-150m int8, sampled, 256 tokens | 8.5 | 75-79 (47 while NumPy still did the sampling, 61-66 before T54) |
| Chromium: stories3_5M / stories260K float32 (grouped-query attention) | 141 / 268 | 402 / 951 |

For comparison: native llama2.c with `gcc -Ofast` runs stories15M at 214 tok/s on the same machine. Firefox 150
loads the kernels too (all of its WebAssembly was 5-7x slower on the test machine, NumPy included). WebKit is
tested on a macOS runner of GitHub Actions (below). Overheads inside Pyodide: one ctypes call 3-12 us, `array.ctypes.data` about 4 us (addresses are
taken once), one small NumPy call 1.5-2.5 us; a token takes about 100 kernel calls.

## WebKit (2026-09-19, T44)

`.github/workflows/webkit.yml` runs `tests/e2e.mjs` in Playwright's WebKit 26.4 on a macOS runner (Apple M1,
virtual) against the deployed site, by hand or once a week. It is Safari's engine, not Safari itself. All four
models ran on the first try. WebKit has no relaxed SIMD, so the int8 models report `SIMD kernels, int8` and run
on `matmul_q8`: the fallback had never met a real browser before.

| model | backend | tok/s (another, much faster CPU than in the other tables) |
|---|---|---:|
| stories260K float32 | SIMD kernels, float32 | 4271 |
| stories15M int8 | SIMD kernels, int8 | 739 |
| tiny-lm int8 | SIMD kernels, int8 | 515 (a run of 20 tokens) |
| llm-jp-3-150m int8, 256 tokens | SIMD kernels, int8 | 146 |

## Where the time of one token goes (2026-09-19, T53)

`node tests/profile.mjs [node|chromium|firefox] [model ...]` measures this inside Pyodide. A browser's clock is too
coarse for single kernel calls, so every number is a difference of totals: the same model runs its forward pass
with the real kernels, with kernels that only count, and with kernels that do nothing (what Python and NumPy cost
around the calls); a kernel called on nothing (n = 0) costs exactly one ctypes call. Positions 16 to 79.

#### One token, Pyodide 314.0.7 in Node v24.19.0

| model | calls | per call | kernels: layers | kernels: classifier | ctypes calls | Python and NumPy | sampling | token | tok/s | read per token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm-jp-3-150m (SIMD kernels, int8, relaxed SIMD) | 231 | 2.9 us | 6.29 ms (52%) | 4.70 ms (39%) | 0.67 ms (6%) | 0.10 ms (1%) | 0.35 ms (3%) | 12.10 ms | 83 | 114 MB (classifier 57) |
| tiny-lm (SIMD kernels, int8, relaxed SIMD) | 79 | 2.9 us | 0.61 ms (26%) | 1.32 ms (55%) | 0.23 ms (9%) | 0.04 ms (2%) | 0.20 ms (8%) | 2.40 ms | 416 | 18 MB (classifier 15) |
| stories260K (SIMD kernels, float32) | 77 | 2.6 us | 0.15 ms (37%) | 0.03 ms (8%) | 0.20 ms (47%) | 0.03 ms (7%) | 0.01 ms (2%) | 0.42 ms | 2390 | 1 MB (classifier 0) |

#### Matmul throughput by matrix size (row length 768), Pyodide 314.0.7 in Node v24.19.0

| rows | int8 weights | matmul_q8 | matmul_q8r (relaxed SIMD) | float32 weights | matmul_f32 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.0 MB | 2.32 G MAC/s (2.6 GB/s) | 2.38 G MAC/s (2.7 GB/s) | 0.0 MB | 2.36 G MAC/s (9.4 GB/s) |
| 64 | 0.0 MB | 5.04 G MAC/s (5.7 GB/s) | 5.89 G MAC/s (6.6 GB/s) | 0.2 MB | 4.30 G MAC/s (17.2 GB/s) |
| 512 | 0.4 MB | 7.86 G MAC/s (8.8 GB/s) | 10.67 G MAC/s (12.0 GB/s) | 1.6 MB | 5.08 G MAC/s (20.3 GB/s) |
| 4096 | 3.1 MB | 8.00 G MAC/s (9.0 GB/s) | 10.68 G MAC/s (12.0 GB/s) | 12.6 MB | 5.45 G MAC/s (21.8 GB/s) |
| 32768 | 25.2 MB | 8.67 G MAC/s (9.8 GB/s) | 11.64 G MAC/s (13.1 GB/s) | 100.7 MB | 5.42 G MAC/s (21.7 GB/s) |
| 98304 | 75.5 MB | 8.74 G MAC/s (9.8 GB/s) | 11.97 G MAC/s (13.5 GB/s) | 302.0 MB | 4.34 G MAC/s (17.3 GB/s) |

#### One token, Pyodide 314.0.7 in chromium 148.0.7778.0

| model | calls | per call | kernels: layers | kernels: classifier | ctypes calls | Python and NumPy | sampling | token | tok/s | read per token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm-jp-3-150m (SIMD kernels, int8, relaxed SIMD) | 231 | 3.1 us | 6.39 ms (52%) | 4.58 ms (38%) | 0.72 ms (6%) | 0.10 ms (1%) | 0.40 ms (3%) | 12.19 ms | 82 | 114 MB (classifier 57) |
| tiny-lm (SIMD kernels, int8, relaxed SIMD) | 79 | 3.0 us | 0.62 ms (25%) | 1.35 ms (54%) | 0.24 ms (10%) | 0.05 ms (2%) | 0.22 ms (9%) | 2.48 ms | 403 | 18 MB (classifier 15) |
| stories260K (SIMD kernels, float32) | 77 | 2.8 us | 0.17 ms (36%) | 0.04 ms (8%) | 0.21 ms (47%) | 0.03 ms (6%) | 0.01 ms (2%) | 0.46 ms | 2181 | 1 MB (classifier 0) |

#### Matmul throughput by matrix size (row length 768), Pyodide 314.0.7 in chromium 148.0.7778.0

| rows | int8 weights | matmul_q8 | matmul_q8r (relaxed SIMD) | float32 weights | matmul_f32 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.0 MB | 2.31 G MAC/s (2.6 GB/s) | 2.36 G MAC/s (2.7 GB/s) | 0.0 MB | 2.40 G MAC/s (9.6 GB/s) |
| 64 | 0.0 MB | 5.05 G MAC/s (5.7 GB/s) | 5.89 G MAC/s (6.6 GB/s) | 0.2 MB | 4.21 G MAC/s (16.9 GB/s) |
| 512 | 0.4 MB | 7.85 G MAC/s (8.8 GB/s) | 10.48 G MAC/s (11.8 GB/s) | 1.6 MB | 4.91 G MAC/s (19.7 GB/s) |
| 4096 | 3.1 MB | 8.09 G MAC/s (9.1 GB/s) | 10.52 G MAC/s (11.8 GB/s) | 12.6 MB | 3.97 G MAC/s (15.9 GB/s) |
| 32768 | 25.2 MB | 8.70 G MAC/s (9.8 GB/s) | 11.14 G MAC/s (12.5 GB/s) | 100.7 MB | 5.14 G MAC/s (20.5 GB/s) |
| 98304 | 75.5 MB | 8.55 G MAC/s (9.6 GB/s) | 11.89 G MAC/s (13.4 GB/s) | 302.0 MB | 4.69 G MAC/s (18.8 GB/s) |

Inside a worker of Chromium, as the page runs it, the numbers are the same (llm-jp-3-150m 81 tok/s).

What this says:

- **The matrix products are nine tenths of a token of llm-jp-3-150m** (layers 52%, the classifier over 99584
  tokens 38%). ctypes calls are 6%, Python and NumPy around them 1%, sampling 3%. One call per layer instead of 19
  (T47) can win at most those 7%; for tiny-lm at most 12%; only a model as small as stories260K, which nobody
  waits for, would double.
- **The int8 product is not bound by memory bandwidth.** Its throughput does not fall when the matrix leaves the
  caches (10.7 G multiply-adds per second at 0.4 MB, 12.0 at 75 MB), and it moves 13 GB/s while the float32
  kernel moves 20 GB/s through the same memory. So it is the arithmetic of `matmul_q8r` that limits the default
  model, and a faster inner loop would show: up to about 1.5x before bandwidth becomes the limit.
- **Attention is slow for what it does.** A token of llm-jp-3-150m takes 12.6 ms at position 8 and 16.3 ms at
  position 248 (Node): 3.7 ms for some 3 million multiply-adds and 24000 exponentials, several times the cost
  per multiply-add of the matrix products. This is why a run of 256 tokens reached 70 tok/s in Node and 61-66 in
  the page, not the 82 of the table above. (T54 has since rewritten attention, see below.)

## What T54 made of that (2026-09-19)

The tables above are from before these changes. Old and new kernels were compared in one process, taking turns
(a run of its own is noisier than the differences): a token of llm-jp-3-150m went from 13.3 to 12.1 ms at position
8, from 15.2 to 12.4 at position 120 and from 17.9 to 12.8 at position 240 (1.09x, 1.23x, 1.39x); tiny-lm 1.08x to
1.20x. In the page, 256 tokens of llm-jp-3-150m: 61-66 -> 75-79 tok/s.

- **Attention walks the cache row by row, all heads of a position at once** (the scratch is now `nh * (pos + 1)`
  floats). A head at a time meant `nh` strided passes over a cache that does not fit the CPU's caches, 12 layers of
  it. With four accumulators for the scores, four exponentials at a time, and the values of four positions added
  per load and store of the output, one call at position 248 went from 109 to 67 us on a cache that fits, and the
  3.7 ms that position 248 added to a token became 0.7 ms.
- **`matmul_q8r`: four groups at a time** (their scales multiplied as one vector, two accumulators taking turns, the
  bias correction as a dot product of its own): 9.8-11.0 -> 11.2-12.5 G multiply-adds per second. That is where it
  ends: a loop of nothing but `i32x4.relaxed_dot_i8x16_i7x16_add_s` with four accumulators reaches 13.0 (9.7 with
  one), so the kernel is within a tenth of what the instruction gives under V8 on this CPU. (Whether V8 turns it
  into one SDOT was not looked up; the speed suggests it does not.)
- Tried and taken back: two rows at a time, sharing the loads of the activations, was slower (10.4 / 8.8 / 11.1
  against 11.2 / 12.1 / 12.5: too many live vectors). The same unrolling in `matmul_q8`, the path without relaxed
  SIMD, changed nothing (7.6-8.3 before and after: its widening multiplies are the cost).
- **The text of an int8 model changed, and that is no error.** Both versions of `matmul_q8r` agree with exact
  integer arithmetic to 1e-6 on the real weights, but a different order of float additions moves a last bit, the
  next layer rounds an activation to the neighbouring 7-bit step, and twelve layers later a logit of
  llm-jp-3-150m differs by up to 1.0. The kernels have always been that far from the NumPy path, which does not
  quantize activations (old kernels 1.7, new ones 1.6, largest difference over 99584 logits). tiny-lm did not
  flip anywhere in the same test (7.6e-6), and the float32 models still write NumPy's text letter for letter.
  What the 7-bit activations cost in perplexity was never measured: T55.

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
- `attention` needs a scratch of `heads * (pos + 1)` floats (the engine allocates `heads * seq_len`), since T54.
- `attention` expects the KV cache as `[seq][kv_heads * head_size]` per layer (the NumPy forward uses another
  layout) and handles grouped-query attention and any head size. int8 models whose row lengths are not
  multiples of 32 run on NumPy.
- Chromium compiles at most 8 MB synchronously on the main thread; these modules are a few KB, and the site
  runs Pyodide in a worker anyway.
- The ABI this relies on is Emscripten's dynamic linking, not the CPython C API, so one binary has served every
  Pyodide version so far. A future Emscripten could break it, which is why NumPy stays as the fallback and why
  `tests/smoke.mjs` checks both paths against the latest Pyodide before every deployment.
- `?kernel=off` in the page URL forces NumPy, to compare.
