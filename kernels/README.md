# SIMD kernels

`kernel.ts` and `kernel_relaxed.ts` (AssemblyScript) are the numeric core of the site: float32 and int8 matmul,
activation quantization, rmsnorm, RoPE, attention, SwiGLU, the residual add, LayerNorm and GELU (for GPT-2 models,
T65), and sampling (repetition penalty, softmax,
top-p). The page's forward pass is `src/runtime/forward.ts` (T93): JavaScript calls the kernels on a
`WebAssembly.Memory` of its own. Python (`src/python/engine/kernels.py`, `load_kernels`) loads them with
`ctypes.CDLL` for the sampling and the converter only: NumPy keeps owning the memory, the kernels get addresses
and work in place. When they cannot be loaded, NumPy does the math.

`kernel.ts` is a window (T356): the kernels are in `kernel/`, by what they are, and `kernel.ts` names every one of
them in the order of the module's exports. `kernel_relaxed.ts` is the window of the relaxed module in the same way (T644.6),
over `relaxed/`, by what the products multiply. `six.ts`, `ternary.ts` and `binary.ts` are what both
import (the widening of an int6 group, of a ternary one and, T634, of a binary one); `signs.ts` makes the last two
type parameters (T644.7: `Ternary`, `Binary`), so that the ternary and the binary products are one kernel a packing,
as the blocks' types are ("A type is a type parameter" below). `relaxed/` and `blocks/` take
the small things of `kernel/math.ts` and `kernel/halves.ts` too (T644.5: one copy of each, compiled into every module
that imports it, all `@inline`). `blocks.ts` and `blocks_relaxed.ts` are the windows of
two modules of their own, over `blocks/` (T363.4: "llama.cpp's blocks" below).

| file | what is in it |
|---|---|
| `kernel/matmul.ts` | the matrix products without relaxed SIMD (`matmul_f32`, `matmul_q8`, `matmul_q6`, `matmul_t2`, `matmul_b1`) and what they take beside the weights: `int8_sums` and `six_sums` (the corrections of the relaxed products), `interleave` (the activations as the ternary products read them), `add_columns` (the classifier's outlier columns); `affine_q8` (T648.3: Firefox Translations' int8 affine as fxtranslate multiplies it, the activations quantized with the matrix's fixed alpha and summed exactly in int32; Python's `engine/translation.py` calls it) |
| `kernel/quantize.ts` | numbers from one form to another: `widen_bf16`, `widen_q8_0`, `widen_pq2_0`, `widen_ptq1_0` (a checkpoint's stored forms to float32), `quantize_x` (the converter's int8, and the activations of every int8 product), `quantize6_x`, `ternary_x`, `binary_x` |
| `kernel/attention.ts` | `rope`, `attention`, `attention_f16` |
| `kernel/halves.ts` | float16: `to_f16`, `from_f16`, `finite_f16`, and the widening `attention_f16` reads the cache with (`halves4`, `half`, which the blocks read their scales with, and `halfOf` of a half's bits, for IQ1_M's d) |
| `kernel/activations.ts` | one vector between the matrix products: `rmsnorm`, `layernorm`, `gelu`, `swiglu`, `geglu` (a Gemma's gate), `add_inplace`, `rotate` and `unrotate` (the rotated basis) |
| `kernel/stateful.ts` | the layers with a state: `gate`, `convolve`, `delta_rule` (Qwen3.5), `short_conv` (LFM2) |
| `kernel/sample.ts` | `argmax`, `penalize`, `sample` |
| `kernel/math.ts` | what several kinds use, and no kernel: the group's size, a vector's sum (`hsum`, and `laneSum` of int32s), four groups' integer sums at once (`groupSums`), `fexp` and `vexp`, the largest of floats |
| `relaxed/int8.ts`, `six.ts`, `signs.ts` | the products with relaxed SIMD (`kernel_relaxed.ts`'s module): `matmul_q8r` and `matmul_q8r_tile`, `matmul_q6r`, `matmul_t2r` and `matmul_t2r_tile`, `matmul_b1r` and `matmul_b1r_tile` (T644.7: the ternary and the binary ones one kernel, `signsRows<P>` and `signsTile<P>`, the packing `P` of `../signs.ts`; `kernel/matmul.ts`'s `matmul_t2` and `matmul_b1` likewise) |

A new kernel goes into the file of its kind and into a line of `kernel.ts` (or, with relaxed SIMD, of `kernel_relaxed.ts`: `tests/forward-check.ts` fails when a
window and its files disagree, and when `src/runtime/jobs.ts` does not say which of its arguments are addresses). The Makefile's rule
reads its sources off the folder (`KERNEL_SOURCES`: every `.ts` of `kernels/` and of its folders), so a new file needs no line there. `node tests/kernels-same.ts [--before <commit>]` builds this tree's kernels and
another commit's and says for every built file whether it is the same bytes, the same functions in another order,
or different, and whether `src/runtime/built/`'s are this tree's. (The division itself changed the order of the functions in the
five builds of `kernel.ts`, and T644.6's division that of the four builds of `kernel_relaxed.ts`, and nothing else: Binaryen sorts the functions that are used equally often by their
names, and AssemblyScript's name of a function begins with its file.)

`make kernels` compiles them into `src/runtime/built/simdkernel.so` (the side module Python loads with ctypes, for the
sampling and the converter; T644.2: no relaxed one, which nothing in Python called) and the plain modules that
`src/runtime/forward.ts` compiles itself (`simdkernel_*.wasm`, `simdkernel_relaxed_*.wasm`), none of them
committed. `build.py` needs no Emscripten: it compiles with AssemblyScript and prepends the `dylink.0` section
that makes a module an Emscripten side module. Verified to load in Pyodide 0.29.4 and 314.0.7.

## Measured before (2026-09-19 and after)

The tables of the development machine before 2026-09-19 (a phone-class ARM CPU, one thread), of what each optimization is worth (T52), of every browser the runners offer (T44, T58), of where the time of one token goes (T53), of what T54 made of that and of the whole context (T56) are in [docs/notes/t644.9-kernels-measured-2026-09-19.md](../docs/notes/t644.9-kernels-measured-2026-09-19.md) (T644.9).

## int8

Weights as `quantize.py` writes them: int8 values in groups of 32 with one float32 scale per group; they stay
views into the downloaded buffer and are never widened. Per matmul the input is quantized once
(`quantize_x`, shared by q/k/v and by w1/w3) and multiplied by `matmul_q8`. With relaxed SIMD, `matmul_q8r`
multiplies int8 by 7-bit unsigned values: activations get a bias of 64, and `corrections = -64 * sum of the
group` (an int32 a group, T197; before it the float32 `scale * sum`) takes it out again, added to the group's
integer sum before that is scaled. The relaxed module is `src/runtime/forward.ts`'s: a browser without relaxed SIMD
(Safari 26 and before: Playwright's WebKit 26.4 refused it and 27.2 compiles it; a shipping Safari 27's is unchecked) refuses to compile it, and int8 then runs on `matmul_q8` (`?without=relaxed` does the same).
(Until T644.2 Python loaded a side module of it as well, in `try/except`, and called nothing of it.)

What quantizing the activations costs (2026-09-19, T55; `node tests/perplexity.ts [model]`, 1499 tokens from the
beginnings of three Japanese Wikipedia articles, contexts of 512 tokens, Pyodide in Node):

| computation | llm-jp-3-150m | tiny-lm |
|---|---:|---:|
| float16 original, no quantization at all | 29.976 | not measured |
| int8 weights, NumPy, activations not quantized | 29.907 (-0.23%) | 88.327 |
| int8 weights, kernels, 8-bit activations (`matmul_q8`) | 29.965 (-0.04%) | 88.339 (+0.01% against NumPy) |
| int8 weights, kernels, 7-bit activations (`matmul_q8r`) | 30.094 (+0.39%) | 88.628 (+0.34% against NumPy) |

The percentages of llm-jp-3-150m are against its original. So the seventh bit costs about half a percent of
perplexity and buys a third more speed (11.2-12.5 against 7.6-8.3 G multiply-adds per second); the largest
difference between two logits, 1.6 as noted above, sounds worse than it is. Nothing to fix.

Except for GPT-2 (2026-09-24, T92). openai-community/gpt2 lost 17% of perplexity to the seventh bit (41.4 against
36.1 with NumPy, 1500 tokens of English Wikipedia), and all of it in the classifier's input: its final LayerNorm
has a weight that blows a few channels up 12 to 17 times (316 against a median of 0.3). One such channel sets
the scale of its group of 32, and the other 31 round to nothing. The fix is `add_columns`: the 8 channels with the
largest norm weight are taken out of the vector before `quantize_x`, and their columns of the classifier, widened
to float32 once at load, are multiplied separately in one SIMD pass over the logits (vocab_size × 8 multiply-adds
next to vocab_size × dim). GPT-2 is back to 36.127 (+0.16% against NumPy); its speed did not measurably change
(old and new in one process, 4 rounds: 57-69 against 61-70 tok/s). Only a norm whose largest weight is 4 times
its median gets it (`OUTLIER_RATIO`; GPT-2 13.9, every other model of the list 1.1-1.9): tiny-lm gains nothing
from it (+0.31% either way) and would pay about 3% of speed. `simdkernel.so` is 6307 → 6572 bytes.

## Ternary weights (2026-10-01, T230 and T231)

A ternary model's weights (-1, 0, +1 times one scale a group of 128) are held two bits each, in the order of Prism
ML's PQ2_0 (weight j as its code, the weight + 1, in byte j >> 2 at bits 2 (j & 3)), with a scale a group (a float32 until T533, a float16 from it, below):
`ternary.ts`. Nothing is widened: `matmul_t2r` loads sixteen bytes (64 weights), and a shift by 0, 2, 4 and 6 bits
and a mask of 3 give four planes of codes, every fourth weight each. `interleave` lays the int8 activations of
`quantize_x` (no bias) out the same way, once a token, and leaves minus the sum of every group of 32 after their
scales. The codes are the 7-bit side of the relaxed dot product and the activations its signed side, in all 8 bits:
`dot(a, w + 1) = dot(a, w) + sum(a)`, so a row adds the token's sums to its integers and keeps no corrections of
its own. `matmul_t2` is the same without relaxed SIMD (the same numbers to the bit), `matmul_t2r_tile` a prompt's
tokens four to a row (the codes made once for the four), and `ternary_x` the converter's packing.

Chosen among the forms of `tests/ternary-forms.ts` by `tests/ternary-bench.ts` (matrices read from memory, the
widths of Ternary Bonsai 1.7B and 27B; billions of weights a second, and against `matmul_q8r` on the same weights
widened to int8):

| form | CI arm64 (Neoverse-N2), 1 / 4 threads | CI x86-64 (EPYC 7763) | CI x86-64 (Xeon 8573C) |
|---|---:|---:|---:|
| `matmul_t2r` | 20.2 (1.26) / 78.0 (1.30) | 17.8 (1.00) / 37.4 (1.05) | 15.8 (1.99) / 33.1 (1.52) |
| the same with its mask written as a constant | 17.8 / 68.2 | 13.0 / 27.9 | 13.8 / 28.5 |
| the pairs' products kept in int16 (x86's `pmaddubsw` alone) | 11.1 / 44.0 | 15.7 / 34.6 | 17.6 / 36.9 |
| `matmul_t2` (no relaxed SIMD) | 12.1 / 47.9 | 10.4 / 21.4 | 9.9 / 20.3 |
| widened to signed int8 for the int8 dot product | 12.5 / 49.3 | 9.3 / 19.4 | 9.9 / 20.4 |
| PTQ1_0's base 3 kept as it is | 6.9 / 27.6 | 5.4 / 12.1 | 5.7 / 11.5 |
| `matmul_q8r` on the weights widened to int8 | 16.1 / 59.9 | 17.8 / 35.6 | 8.0 / 21.7 |

The mask of 3 is an argument of the kernels (`jobs.ts` passes it): V8 makes a constant vector written in the loop
again at every use. The form that keeps int16 sums is as fast or faster on x86-64 and half as fast on arm64; the
phones this is for are arm64.

**The scales are float16s from T533 (2026-10-15)**, as PQ2_0 and PTQ1_0 hold them: 34 bytes a group of 128, not 36
(a ternary model 5.6% smaller: Ternary Bonsai 1.7B 457 MB, 2 27B 7.24 GB). The form is ssig33/pyodide-llm's
(`7d7b776c`, MPL-2.0: PQ2_0's blocks as they are, a half widened once a block), written anew in this layout (the
scales after all the codes, as before). A row's kernels widen four groups' scales at once (`halves4`) and broadcast a
lane to each group; one half widened a group (`half`) was 7 to 9% slower on arm64. The numbers are the float32
form's to the bit for the same scales (`tests/ternary-bench.ts` holds every form to it). Against the float32 form
(a copy of it, t2c, in the same process; a second copy of it is within 1%), G weights/s, 1 / 4 threads, the
geometric mean of the widths of 1.7B, 8B and 27B:

| form | CI arm64 (Neoverse-N2) | CI x86-64 (EPYC 9V74) |
|---|---:|---:|
| float32 scales (as before) | 20.3 / 73.0 | 21.4 / 42.3 |
| float16, four groups' at once (taken) | 20.8 (1.02) / 75.3 (1.03) | 21.0 (0.98) / 41.8 (0.99) |
| float16, one a group | 18.5 (0.91) / 67.6 (0.93) | 20.3 (0.95) / 40.0 (0.95) |
| PQ2_0's 34-byte blocks, one a group | 18.1 (0.89) / 65.9 (0.90) | 20.2 (0.95) / 40.0 (0.95) |

The prompt's tile widens one half a group for its four tokens: 0.98 of the float32 tile on arm64 (28.1 against 28.8
and 112.0 against 115.1 G weights/s), 0.99 on x86-64. The kernels as built against their float32 forms: `matmul_t2r`
1.02 / 1.02 on arm64 (Neoverse-N2), 1.02 / 1.00 on an EPYC 7763 and 0.96 / 0.96 on an EPYC 9V45; `matmul_t2` 1.00 /
1.00 on arm64 and 0.97 / 0.99 on the 7763 (TASKS.md's T533 has the runs).

**`matmul_t2` and `matmul_b1` add their int16 sums as a chain and in int16 (T275, 2026-11-12)**: the kernels Safari
runs (it has no relaxed SIMD). The four planes' products are added as a chain (`acc + product`) and the eight int16 lanes
of each group of 32 are added in int16 (64-bit lane shuffles, then 32-bit ones) and widened to int32 once a group of 128,
not four times: the same integers, so the same numbers to the bit. `tests/ternary-browser-bench.ts` times them, one
thread, in each browser's engine and in Node, against the kernels as built at a commit (`--before`) and against the forms
of `tests/ternary-plain-forms.ts`; G weights/s and the ratio's median over 15 turns, the geometric mean of the widths of
Ternary Bonsai 1.7B (2048x2048, 6144x2048, 2048x6144), runs 43257964376 and 43257971334:

| form, against the kernel before | CI arm64 (Neoverse-N2): WebKit 27.2 / Chromium 156 / Node 24 | CI x86-64 (EPYC 9V74): WebKit / Chromium / Node |
|---|---:|---:|
| `matmul_t2` before (12.7 / 12.4 / 12.1 G weights/s on arm64, 7.4 / 12.3 / 12.3 on x86-64) | 1.000 | 1.000 |
| a copy of it in another module (the noise) | 1.000 / 0.999 / 1.000 | 1.001 / 1.000 / 1.000 |
| the chain and the int16 sums (taken) | 1.082 / 1.163 / 1.172 | 1.023 / 1.029 / 1.029 |
| the same, the last plane a shift of bytes with no mask | 1.093 / 1.191 / 1.202 | 0.995 / 0.998 / 0.994 |
| the int16 sums with the tree of adds | 1.044 / 1.071 / 1.083 | 1.044 / 1.024 / 1.025 |
| `matmul_b1`, taken, against `matmul_b1` before | 1.089 / 1.154 / 1.151 | 1.076 / 1.033 / 1.037 |

And in run 43256814753 and 43256807057 (a first form of the bench, its WebKit clock read too finely): the chain alone
1.02 / 1.07 / 1.08 on arm64 and 0.97 / 1.00 / 1.00 on x86-64; the products as `i32x4.dot_i16x8_s` of the bytes widened
to int16 0.47 / 0.48 / 0.49 on arm64 (1.04 / 1.15 / 1.14 on x86-64); two rows at once, the activations loaded once,
1.07 / 1.14 / 1.16 on arm64 and 0.92 / 0.93 / 0.92 on x86-64. The shift of bytes is not taken: x86-64 has none, and it
is slower there than the kernel before (Safari on an Intel Mac); nor is it a form of `matmul_b1` (its codes are of bits).
The int16 sums with the chain are faster everywhere. Against relaxed SIMD (the `matmul_t2r` of Chromium) the kernel is
still the slower: T231's table, 0.6 of it on arm64.

**Other forms of `matmul_t2r`, timed and not taken (T288, 2026-11-12).** `tests/ternary-browser-bench.ts --relaxed` times
the forms of `tests/ternary-relaxed-forms.ts` against `matmul_t2r` as T275's bench times the plain ones (one thread, each
browser's engine and Node, every form held to `matmul_t2r` to the bit): `x16`, the planes' products as
`i16x8.relaxed_dot_i8x16_i7x16_s` added in int16 over a group of 128 and widened once (T231's review's `bxa`); `x16b`, the
same widened a block; `tl`, BitNet's TL1 (the weights of 16 rows a vector, a nibble a pair of weights, a table of 16
int16 sums a pair of activations made in every call and looked up with `relaxed_swizzle`: its own layout of the
weights). The geometric mean of the ratios over the widths of Ternary Bonsai 1.7B, runs 43278350428, 43279127526
(EPYC 9V74, Zen 4), 43278357272 (EPYC 7763, Zen 3) and 43278364945 (Neoverse-N2); a copy of `matmul_t2r` is 0.998 to
1.003:

| form, against `matmul_t2r` | Neoverse-N2: Node / Chromium 156 / Firefox 157 / WebKit 27.2 | EPYC 7763 | EPYC 9V74 |
|---|---:|---:|---:|
| `matmul_t2r` (G weights/s, 2048x2048) | 20.3 / 18.7 / 10.2 / 17.0 | 17.9 / 17.9 / 18.9 / 13.9 | 21.8 / 21.8 / 22.5 / 17.3 |
| `x16` | 0.626 / 0.654 / 1.046 / 0.742 | 1.332 / 1.327 / 1.118 / 1.294 | 1.357 / 1.362 / 1.105 / 1.350 |
| `x16b` | 0.601 / 0.633 / 1.026 / 0.730 | 1.305 / 1.304 / 1.106 / 1.218 | 1.327 / 1.330 / 1.105 / 1.271 |
| `tl` (its tables made in every call) | 0.940 / 1.006 / 1.809 / 1.112 | 1.272 / 1.301 / 1.275 / 1.093 | 1.063 / 1.078 / 1.043 / 0.980 |

(The second run on the 9V74 gave the same to 0.01.) Neither is taken. `x16` is 1.33 to 1.36 times on AMD's Zen 3 and 4
in V8, as T231's review found, and 0.63 to 0.65 on arm64's V8, where the owner's phone runs: it would need a choice made
on the device (a second kernel, its tile, and a timing at the load), which T231's review left to the owner and which
changes `WORK`. No Zen 5 or Intel CPU came up in three runs (T231 saw V8 make `matmul_t2r`'s dot one VNNI instruction on
a Zen 5, where `x16` would lose). `tl` is no faster on arm64's V8 (0.94 in Node, 1.01 in Chromium) and needs the file's
layout of the weights changed (`CONVERTER`: every ternary model fetched again). What would turn either: the owner taking
a choice made by timing on x86-64 (`x16`), or Firefox on arm64 (`matmul_t2r` 10 G weights/s there, half V8's: `tl` 1.81
times, `x16` 1.05) counting enough to change the layout.

Two things the review of 2026-10-02 found, which no test in Node can see:

- `interleave` is shifts, masks and narrows, with no shuffle. It was a transpose of three chained shuffles (a byte shuffle
  of each vector, then dwords, then qwords), and JavaScriptCore's optimizing tier on x86-64 (Playwright's WebKit on Linux,
  on AMD EPYCs and an Intel Xeon alike; Safari on an Intel Mac has the same compiler and was not tried) folded them wrongly once the function
  had been called about 2000 times: three quarters of a block
  came out wrong and the model wrote nonsense, with 1 thread and with 4. Each piece alone, a swizzle in place of the byte
  shuffle, shifts and narrows, and a scalar loop were right on that engine; the narrows cost 5 to 8 ns more a block of 64
  bytes in V8 (EPYC 9V74 3.6 to 10.5, EPYC 7763 4.8 to 9.9, arm64 4.5 to 12.4: 0.04 ms of a token of the 1.7B).
  `tests/kernels-in-browser.ts` runs the kernels in the engine of each browser, in batches up to 30000 calls, and says
  WRONG on the old `interleave` in WebKit on x86-64.
- On a 64-bit memory, the V8 of Node 24 (13.6) on arm64 reads `v128.load32_splat` and `v128.load32_lane` at the low 32 bits
  of an address above 4 GiB. Every kernel that takes a scale so reads a wrong one there: the ternary kernels' weights'
  scales and the int8 tile's. Chromium 148 on arm64, Firefox 150 and V8 on x86-64 read them right.
  `tests/ternary-check.ts` has a canary module for it and leaves its above-4 GiB config out where it fires, and
  `tests/kernels-in-browser.ts` runs every kernel above 4 GiB in each browser that has such a memory (the 27B model
  runs there).

## llama.cpp's blocks (2026-10-13, T363.4)

`blocks.ts` and `blocks_relaxed.ts` are the windows of two more modules, `blockkernel_*.wasm`: the matrix products on
rows kept as a GGUF has them, llama.cpp's blocks of 256 weights (Q2_K, Q4_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ4_XS: the ten types Underdog Saluki 27B is made of, 2.3 bits a weight on average). The weights are
never widened in memory: what a dot product takes is made of them in registers as they are read. `blocks/types.ts`
has, at its head, what each type is to the kernels, the one arithmetic every kernel follows, and the tables.

| kernel | module | what it does |
|---|---|---|
| `sb_tables(area, codebooks)` | `blockkernel` | the tables the others read (34992 bytes at `area`), of the checkpoint's row `codebooks`; returns the bytes written |
| `sums_x(xs, xq, n)` | `blockkernel` | after `quantize_x(xq, xs, x, n, 0)`: the int16 sums of each 16 and each 8 activations, after the scales at `xs` (`n / 2` bytes in all) |
| `matmul_sb(xout, xq, xs, w, tab, n, r0, r1, type)` | `blockkernel` | one token, rows `r0` to `r1`, without relaxed SIMD |
| `matmul_sbr(…)` | `blockkernel_relaxed` | the same numbers to the bit, with relaxed SIMD |
| `matmul_sbr_tile(xout, xq, xs, w, tab, n, r0, r1, count, os, frame, type)` | `blockkernel_relaxed` | a prompt's `count` tokens, four against one reading of a row: `matmul_sbr`'s numbers to the bit |
| `matmul_sbw_tile(…, type, scratch)` | `blockkernel_relaxed` | the same with a row spread once as signed weights into `scratch` (`n + n / 4 + n / 64` bytes of one thread's own): only where `signed_dots` says the engine multiplies signed bytes by signed bytes |
| `signed_dots(out, a, b)` | `blockkernel_relaxed` | the relaxed dot product of 16 bytes by 16, for the caller to ask that of every pair of bytes |
| `sb_row(out, w, tab, n, type)` | `blockkernel` | a row as float32, each weight the converter's reader's value to the bit (a token's row of the embedding) |

`type` is the type's number in a GGUF (ggml's: 10, 12, 16, 17, 18, 19, 21, 22, 23, 29). The form is T363.1's "rb3"
(`docs/notes/t363.1-saluki-kernel-forms-2026-10-13.md`): the magnitudes a grid holds are the 7-bit side of the relaxed
dot product and the activations its signed side in all 8 bits, each given its weight's sign as it is loaded; a
group's sum is an exact integer before it meets a float. So a row keeps no corrections, and the numbers do not depend
on relaxed SIMD, on how many tokens a kernel takes, or on which side the signs are put.

- **Modules of their own.** Built, they are 30 KB and 105 KB (5.7 and 16.1 KB gzipped) beside `kernel.ts`'s 23 KB and
  `kernel_relaxed.ts`'s 7 KB (7.4 and 2.4): in those two, every visit would fetch them. `src/runtime/built.ts` names
  the eight files; no page fetches them before a model with such rows does (T363.5).
- **A type is a type parameter** (`blockOf<XXS2, R>(…)`), and so is relaxed SIMD or not. AssemblyScript compiles every
  branch of an inlined function whatever constant it is called with (`if (kind == 16)` with `kind` the constant 16
  keeps all ten types' code at every use: the first version took three minutes to fail to compile, and a call of
  relaxed SIMD in a branch never taken fails a module built without it). A comparison of `idof<K>()` is settled as
  the function is compiled. The type's number is looked at once a block.
- **The tables are made at run time** (`sb_tables`): the grids as the checkpoint's row has them (IQ1's with one added),
  the masks of the 7-bit signs, IQ4_XS's values with their magnitudes and signs, and four vectors. A small constant a
  loop needs often (ones, sevens, fifteens of 32 bits) is a shift of a vector the kernel holds in a register: read
  from the tables wherever it was used, it cost IQ1_M's kernel a tenth.
- **The activations are int8 from -127 to 127** (`quantize_x` with no bias): -128 would keep its sign where its
  weight's is negative.
- **Above 4 GiB** (the 27B's weights are, on a 64-bit memory) the kernels load with `v128.load`, `load64_zero`,
  `load32_zero` and `load64_lane`, none of which V8's Liftoff on arm64 reads at the low 32 bits of the address (it
  does that to `load32_splat` and `load32_lane`): `tests/blocks-check.ts` runs every kernel three times, as V8's
  tiers come, with Liftoff alone and with TurboFan alone.
- **`matmul_sbw_tile` is the first kernel that leans on what relaxed SIMD leaves open**: the second operand of its dot
  product is a signed byte in all 8 bits, which arm64's SDOT multiplies exactly and x86's PMADDUBSW takes as unsigned.
  The caller asks the engine with `signed_dots` (every pair of bytes: `tests/blocks-model.ts`'s `signedDots()`), and
  where the answer is the exact product the kernel's integers are `matmul_sbr`'s, so its numbers are too. CI's
  WebKit, Firefox and Chromium on arm64 all answered yes, and all three on x86-64 no.

`tests/blocks-check.ts` holds the reading of every type to the converter's own (`convert/superblocks.py`, to the
bit) and every kernel to `tests/blocks-model.ts` (to the bit) on the eight builds; `tests/kernels-in-browser.ts`
calls them in each browser's engine up to 30000 times and above 4 GiB. Billions of weights a second, matrices of the
27B's shapes read from memory (`tests/saluki-kernel-bench.ts`, CI, 1 thread / 4 threads):

| kernel and type | CI arm64 (Neoverse-N2), 1 / 4 threads | CI x86-64 (EPYC 9V45), 1 / 4 threads |
|---|---:|---:|
| `matmul_sbr`, IQ2_XXS (36% of what a token of the 27B reads) | 7.43 / 29.65 | 15.59 / 33.96 |
| IQ2_XS | 5.76 / 23.00 | 12.78 / 27.91 |
| IQ2_S | 5.62 / 22.49 | 12.33 / 26.76 |
| IQ3_XXS | 5.14 / 20.57 | 10.31 / 20.82 |
| IQ3_S | 4.53 / 18.10 | 7.39 / 17.64 |
| IQ1_S | 8.19 / 32.72 | 14.95 / 31.90 |
| IQ1_M | 6.20 / 24.73 | 13.81 / 28.80 |
| IQ4_XS | 9.07 / 35.08 | 24.46 / 50.45 |
| Q2_K | 9.77 / 38.77 | 28.59 / 57.06 |
| Q4_K | 10.90 / 41.66 | 28.48 / 57.94 |
| `matmul_sb` (no relaxed SIMD), IQ2_XXS | 6.31 / 25.16 | 12.16 / 25.37 |
| `matmul_sb`, IQ3_S | 3.96 / 15.83 | 6.11 / 14.00 |
| T363.1's forms in the same runs: rb3 (IQ2_XXS), IQ3_S, Q2_K, IQ1_M | 7.40, 4.26, 10.37, 6.39 / 29.57, 17.03, 41.15, 25.48 | 15.83, 7.41, 28.62, 13.36 / 34.14, 17.46, 57.27, 27.31 |
| `matmul_t2r` (ternary: how Ternary Bonsai 2 27B runs) | 20.29 / 78.96 | 53.56 / 111.25 |
| a prompt of 16 tokens, `matmul_sbr_tile`: IQ2_XXS, IQ3_S, IQ2_S, IQ1_M, Q2_K (times the token's kernel on each token) | 12.12 (1.65), 9.73 (2.17), 9.65 (1.74), 8.84 (1.44), 11.38 (1.18) / 48.15, 38.73, 38.46, 35.21, 45.35 | 31.82 (2.05), 20.93 (2.86), 23.42 (1.92), 18.15 (1.33), 31.01 (1.11) / 68.74, 45.34, 50.04, 38.10, 63.74 |
| the same, `matmul_sbw_tile`: IQ2_XXS, IQ3_S, IQ2_S, IQ1_M | 16.17 (2.20), 15.31 (3.42), 13.30 (2.39), 13.57 (2.21) / 64.43, 60.69, 52.96, 53.89 | not called there |

From those, the matrix products of one token of the 27B (25.6 billion weights, every type at its own speed, an
estimate): 0.98 s on 4 threads of the Neoverse-N2 and 3.90 s on 1; 0.87 s and 1.91 s on the EPYC 9V45. Against
T363.1's forms the kernels are the same within 2% but for Q2_K on arm64 (0.94) and IQ1_M on arm64 (0.97): a type's
scales are read as a vector and the units' sums made one lane each, for every type alike.

## Rules that are easy to break

- **No static data.** The side module has no relocations, so a data segment would be written over Pyodide's own
  memory. That rules out AssemblyScript's std math (`Mathf.exp` uses tables), strings and asserts; `build.py`
  refuses a module with a data section. `fexp` in `kernel/math.ts` is the table-free replacement.
- **Keep every array alive whose address a kernel gets.** The kernels only know numbers: a scratch array that
  no Python object refers to any more is freed, and the kernel then writes into whatever lives there next. It
  shows up as a rare `memory access out of bounds` (`_kernel_buffers`, `KernelSampler.buffers` in the engine).
- Mutable globals are fine (they are no data segment): `sample` keeps the state of its partial sort in three.
- `sample` sorts only as far as the nucleus reaches and gets its random number from Python, so a seed gives the
  same text again. With the same random number it picks the token that `NumpySampler` (the reference, in NumPy) draws.
- The relaxed module must not be named `*.so` if it ever ships inside a wheel: Pyodide pre-loads every `.so`.
- `attention` needs a scratch of `heads * (pos + 1)` floats (the engine allocates `heads * seq_len`), since T54.
- `attention` expects the KV cache as `[seq][kv_heads * head_size]` per layer (the NumPy forward uses another
  layout) and handles grouped-query attention and any head size. int8 models whose row lengths are not
  multiples of 32 run on NumPy.
- Chromium compiles at most 8 MB synchronously on the main thread; these modules are a few KB, and the site
  runs Pyodide in a worker anyway.
- The ABI this relies on is Emscripten's dynamic linking, not the CPython C API, so one binary has served every
  Pyodide version so far. A future Emscripten could break it, which is why NumPy stays as the fallback and why
  `tests/smoke.ts` checks both paths against the latest Pyodide before every deployment.
- `?kernel=off` in the page URL forces NumPy, to compare.
