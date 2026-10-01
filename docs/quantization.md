# Quantization

The larger models are stored in int8, and in 6 bits where int8 does not fit; a ternary model keeps its weights as
they are, in 2 bits. This page says how, and what it costs
in quality. Perplexity is lower for better predictions; "+1%" means 1% worse than the original.

## int8

Weights are split into groups of 32, each with one float32 scale (the largest value of the group maps to 127).
The kernels multiply the int8 values as they are, without widening them to float32. For llm-jp-3-150m that took
the WebAssembly heap from 897 MB to 283 MB.

The weights alone are hard to tell from the original:

| model | original | int8 |
|---|---:|---:|
| stories15M | - | +0.04% |
| tiny-lm | 91.3 | 91.1 |
| llm-jp-3-150m | 22.76 | 22.69 |

The most likely token agrees about 98% of the time. Greedy output leaves the original's text part way through but
stays coherent. int4 cost +16.8% and was rejected.

## Activations: 8 or 7 bits

The kernels also quantize the input of each matrix product, per group of 32. Where the browser has relaxed SIMD
(Chrome, Edge, Firefox), they use 7 bits, which makes int8 about 30% faster. Measured on the first 1,499 tokens
of three Japanese Wikipedia articles (`node tests/perplexity.mjs`):

| llm-jp-3-150m | perplexity |
|---|---:|
| float16 original | 29.976 |
| int8 weights, NumPy | 29.907 |
| + 8-bit activations | 29.965 |
| + 7-bit activations | 30.094 (+0.39%) |

tiny-lm: +0.01% with 8 bits, +0.34% with 7.

Other architectures (1,500 tokens of English Wikipedia; rinna on Japanese):

| model | original float32 | int8, NumPy | + 8-bit activations | + 7-bit activations |
|---|---:|---:|---:|---:|
| rinna/japanese-gpt2-small | 33.253 | +0.05% | +0.11% | +0.20% |
| EleutherAI/pythia-160m | 47.795 | +1.17% | +1.50% | +2.23% |
| openai-community/gpt2 | 35.340 | +2.06% | +4.73% | +17.2% (before the fix below) |

**GPT-2 and 7 bits.** The loss came from one place: the input of the classifier. GPT-2's final LayerNorm
multiplies a few channels 12 to 17 times, and one such channel in a group of 32 rounds the other 31 away. The
engine now takes the 8 channels with the largest norm weights out before quantizing, and multiplies them in
float32 separately. It does this only where the largest norm weight is at least 4 times the median (GPT-2: 13.9;
the other models 1.1 to 1.9). GPT-2's 7-bit row went from 41.431 to 36.127 (+2.2% against the original). The
remaining difference comes from the int8 weights of the classifier.

## 6 bits

Where int8 does not fit (Safari, which has no 64-bit memory, and devices that report little memory), the page
stores the weights in 6 bits. Measured on five models (SmolLM2 135M and 360M, Qwen2.5 0.5B, Pythia 410M,
llm-jp-3 440M):

- 4 bits: +10 to +48% in every form tried (llama.cpp's Q4_0: +12 to +15%). Too much.
- 5 bits (groups of 32, float16 scale): up to +9.94%.
- **6 bits: up to +3.37%.** The owner accepted a loss of 3 to 5%, so 6 bits it is.
- Larger models lose less: llm-jp-3 980M +1.02%, Llama 3.2 1B +1.37%, Qwen2.5 1.5B +1.68%. Their greedy text is
  often the same as the original's, word for word.

A 6-bit value is stored as an int8 whose lowest 2 bits are 0, with a scale a quarter as large. So only the packing
(32 values in 24 bytes) is new, and everything else is the int8 path. A 6-bit file is 7/9 of the int8 file.

6 bits is slower: about 0.45 times int8 on one thread (stories15M, 243 against 490 tok/s), 0.37 to 0.53 times on 4
threads in CI. Since 2026-09-27 the kernel widens the 6-bit values in 8 instructions instead of 15, 1.64 times as
fast on CI's x86-64 runner and 1.26 to 1.30 times on its arm64 runner; the ratio to int8 after that has not been
measured. So the page uses 6 bits only where int8 does not fit. Where the browser has 64-bit memory, a model
too large for 32 bits stays int8: Llama 3.2 3B wrote 1.9 tok/s in 6 bits on 32-bit memory and 4.1 tok/s in int8
on 64-bit memory (Chromium in CI). `?bits=6` or `?bits=8` chooses by hand.

## Models from GGUF

58 of the 71 Hugging Face models of the list are fetched as a Q8_0 GGUF (llama.cpp's int8 with a float16 scale
per 32 values), with the vocabulary and the configuration of the original repository. Q8_0 turns back into int8
without loss. Each GGUF was compared with its original tensor by tensor before it went into the list
(`tests/gguf_check.py`): every row had to be within a relative error of 0.05 of the original, of llama.cpp's Q8_0 of the original, or
of the Q8_0 of the original rounded to float16 (some GGUFs were made through float16). This is checked for each
file, not for each publisher: Qwen's own GGUFs of Qwen3 0.6B and 1.7B differ from their originals by 0.9 to 2.2%
in every layer matrix, while Qwen's own 4B and 8B match, so the list takes those two from another publisher.

The GGUF's int8 is, byte for byte, llama.cpp's Q8_0 of the original (checked on the GPT-2 and Pythia GGUFs), not
the page's own int8 of it. The two differ like any two roundings of the same int8: on 1,000 tokens the most likely
token agreed 95.7 to 98.2% of the time and the perplexity moved by −0.13 to +0.31%, as much as a small change to
the rounding of the page's own int8 moves them.

In 6 bits, a GGUF is quantized twice, which adds about 3% to the error of the weights (Qwen2.5 0.5B: 0.02375
against 0.02308); on 1,500 tokens the perplexity could not tell the two apart.

One more model comes from a GGUF of another kind: Ternary Bonsai 1.7B (Prism ML), whose every weight is −1, 0 or 1
times a scale shared by 128 weights. Its GGUF holds two bits a weight (PQ2_0, a type of Prism ML's fork of
llama.cpp; 463 MB). The GGUF was compared with the float16 safetensors of the same weights: no tensor is further
than 8.7e-5 from it (a few blocks of 128 have two magnitudes there, 0.5% apart, and one in the GGUF). How the page
holds such weights is the next section.

## Ternary weights: 2 bits

A ternary model is not quantized by the page: its weights are kept as the file has them. The format is PQ2_0's
own, 32 bytes for 128 weights (2 bits each: the weight plus 1) and one float32 scale for the 128, 2.25 bits a
weight. Ternary Bonsai 1.7B takes 484 MB that way, a quarter of the 1.94 GB it took widened to int8 (which the
page did until 2026-10-01, and still does with `?bits=8`). Nothing is rounded: the converter refuses a value
that is neither 0 nor plus or minus its group's scale, so it cannot turn another model into this format by
mistake. The other ternary type of that fork, PTQ1_0 (five weights a byte in base 3, 28 bytes for 128), is read
into the same format; the two files of one model give the same bytes.

The kernel does not widen the weights either. A shift of sixteen bytes and a mask give the codes 0, 1, 2 of every
fourth weight of 64; the activations are laid out the same way once a token, so the codes meet them as they
are loaded. The codes are never negative, so the activations keep all their 8 bits (the int8 kernel gives them 7
with relaxed SIMD), and what the "+1" adds, the sum of the activations, is taken off once a token and not once a
row. Measured on Ternary Bonsai 1.7B in CI, on 1,500 tokens of Wikipedia:

| computation | English | Japanese |
|---|---:|---:|
| the file's values in float32, NumPy | 22.046 | 45.727 |
| widened to int8, kernels, 7-bit activations (the page before) | 22.087 (+0.19%) | 45.623 (−0.23%) |
| ternary, kernels, 8-bit activations | 22.066 (+0.09%) | 45.761 (+0.07%) |

and its speed against the same weights widened to int8, in tok/s (ternary / int8):

| threads | CI arm64 (Neoverse-N2) | CI x86-64 (AMD EPYC 7763) |
|---:|---:|---:|
| 1 | 10.9 / 9.3 | 10.1 / 9.2 |
| 2 | 19.4 / 16.0 | 18.6 / 15.1 |
| 4 | 33.0 / 26.5 | 19.5 / 15.8 |

It runs on the CPU only for now (no GPU path), and has not been measured on a phone.

Ternary Bonsai 4B and 8B come the same way (1.07 GB and 2.18 GB of PQ2_0, held as 1.1 GB and 2.3 GB of ternary
weights; widened to int8 they were 4.5 GB and 9.2 GB, on a 64-bit memory). Their speed has not been measured. Compared with their float16 safetensors, no tensor of the 4B is further than 8.5e-5 from them, and
the 8B's are the same values.

Qwen3.5 0.8B's Q8_0 GGUF holds some tensors otherwise than the original does: llama.cpp writes the norms with the 1
the model adds to them and `A_log` as −exp(A_log), and it quantizes the two small matrices of the gates of each
linear-attention layer (16 × 1024), which the page keeps in float32. The page takes the first two as they come (4 of
the 288 −exp(A_log) are one unit in the last place from NumPy's) and keeps the gates' matrices as Q8_0 rounded them
(0.57% from the original's). On 1,500 tokens of English Wikipedia the GGUF's weights measure 25.055 with NumPy
against 25.056 for the float32 original and 25.034 for the page's own int8 of it; on the kernels, 25.398 with 7-bit
activations and 25.086 with 8-bit ones (CI's x86-64 runner).

## Other small effects

- A BOS token at the start: the page always starts with one, while some models are used without it. The
  difference was within ±3%, in either direction depending on the text, so it was kept. One model broke without
  the right BOS (DeepSeek-R1 Distill Qwen 1.5B), and its entry now names it.
- RMSNorm's epsilon: 1e-5 for all models until Qwen3 0.6B showed +0.12% with it; the converter now passes the
  model's own value.
- The order of rounding inside the int8 kernels changed twice on 2026-09-27 (one scaling per group instead of
  four, and the relaxed SIMD correction added as an integer). The first moved perplexity by less than 0.1%
  (llm-jp-3-150m 29.992 → 29.997, tiny-lm 88.347 → 88.262); the second by −0.31 to +0.59% over eight comparisons,
  in no fixed direction, as much as between a float16 and a float32 KV cache.
- The KV cache: float16 (with threads) cannot be told from float32 (llm-jp-3-150m: 29.957 against 30.094). On the
  GPU the keys and values are float16 too.
