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
than 8.7e-5 from it (a few blocks of 128 have two magnitudes there, 0.78% apart, and one in the GGUF). How the page
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
row. Measured on Ternary Bonsai 1.7B in CI, on the first 1,500 tokens of an English and of a Japanese Wikipedia article:

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

On three English and three Japanese articles (the first 1,500 tokens of each), ternary with 8-bit activations differs from int8 with
7-bit activations by −0.31% to +0.30% (mean +0.02%); the sign changes from article to article.

It runs on the CPU only for now (no GPU path), and has not been measured on a phone.

Ternary Bonsai 4B and 8B come the same way (1.07 GB and 2.18 GB of PQ2_0, held as 1.1 GB and 2.3 GB of ternary
weights; widened to int8 they were 4.5 GB and 9.2 GB, on a 64-bit memory). On the site in Chromium on a CI x86-64 machine with four
threads the 4B wrote 6.4 tok/s and the 8B 4.2 tok/s, in readable Japanese (2026-10-02). The three computations of the table above
are within 0.09% of one another on the 4B too (English 16.863 ternary, 16.852 int8 with 7-bit activations; Japanese 35.248 and
35.218). Compared with their float16 safetensors, no tensor of the 4B is further than 8.5e-5 from them, and
the 8B's are the same values. The difference is in the originals: the 1.7B's and the 4B's float16 files hold a few blocks of 128
(about 5 in 100,000: 740 and 1,643) with two scales one bfloat16 step (0.78%) apart, of which the GGUF keeps the larger
for the whole block, and the 8B's hold none (64 million blocks, counted).

### Ternary Bonsai 2 27B

The largest model of the list is a ternary Qwen3.8 27B: 5.95 GB as its PTQ1_0 file, 7.66 GB as the page's ternary
weights (as int8 it would be 30 GB, past what a browser gives a page), on a 64-bit memory of 7.7 GiB with its 4,096
positions (Chrome and Firefox). Its matrices are stored in a rotated basis: every matrix reads its input after a
change of signs and a Walsh-Hadamard transform over blocks of 1,024 values, which the engine applies as it goes
(the weights turned back would not be ternary). Its attention is the hybrid one of Qwen3.5 (three layers of four
keep a state of a fixed size, the fourth keeps keys and values).

Measured in CI (2026-10-02), on the real model:

- The PTQ1_0 and the PQ2_0 file convert to the same checkpoint, byte for byte, natively and in Pyodide (Pyodide:
  216 s for PTQ1_0, 137 s for PQ2_0; its heap grows to 0.8 to 0.9 GB). The list takes PTQ1_0: 1.26 GB less to
  fetch, which wins on any line slower than 33 MB/s.
- The page's forward pass (`forward.js`, the ternary kernels, 4 threads) against Prism ML's fork of llama.cpp run
  with float32 activations, over 155 positions of four texts: the logits differ by 0.07 to 0.19, which is what
  rounding the activations to 8 bits moves them by (the engine's own NumPy forward pass without that rounding is
  within 0.008 of the fork); the most likely token is the same at 154 positions (at the other the fork's own first
  two are 0.004 apart), and the 16 tokens the fork writes greedily are the most likely ones for each text.
- Conversions and readings broken on purpose (the value heads of the output matrices in llama.cpp's order, a PTQ1_0
  block read in the order of its bytes, the embedding not turned back, the signs of one width reversed) move the
  logits by 17 to 26. One reversed sign of the 17,408 moves them by 0.11 to 0.45: the weakest ones are under what
  the rounding moves, and it is the NumPy comparison that sees them.

| threads | CI x86-64 (AMD EPYC 9V74) | CI arm64 (Neoverse-N2) |
|---:|---:|---:|
| 1 | 0.77 | 0.72 |
| 2 | 1.49 | 1.32 |
| 4 | 1.56 | 2.42 |

in tok/s, in Node on runners of 4 logical cores (the x86-64 ones are 2 cores). The fork's own CPU path wrote 0.70
and 1.13 tok/s on 4 threads of the same runners. A prompt goes through at 1.8 (x86-64) and 3.4 tok/s (arm64). At
the end of the context the forward pass holds 570 MB after the checkpoint. In Chromium on CI's runners (the deployed
page) it was ready in 291 s on x86-64 and 235 s on arm64 and wrote 1.2 and 2.2 tok/s; not measured on any device.

It has two entries: one that answers at once, and one that thinks first with the template's reasoning effort
"medium". The model's own default, "xhigh", plans for thousands of thinking tokens, hours at this speed.

Qwen3.5 0.8B's Q8_0 GGUF holds some tensors otherwise than the original does: llama.cpp writes the norms with the 1
the model adds to them and `A_log` as −exp(A_log), and it quantizes the two small matrices of the gates of each
linear-attention layer (16 × 1024), which the page keeps in float32. The page takes the first two as they come (4 of
the 288 −exp(A_log) are one unit in the last place from NumPy's) and keeps the gates' matrices as Q8_0 rounded them
(0.57% from the original's). On 1,500 tokens of English Wikipedia the GGUF's weights measure 25.055 with NumPy
against 25.056 for the float32 original and 25.034 for the page's own int8 of it; on the kernels, 25.398 with 7-bit
activations and 25.086 with 8-bit ones (CI's x86-64 runner).

The larger Qwen3.5 (4B and 9B) have two value heads to each key head in their linear-attention layers, and llama.cpp
writes those value heads into the GGUF in another order than Hugging Face keeps them: every key head's first value
head, then every key head's second. The page puts them back as it converts (eight tensors of each such layer; it
moves values and changes none). Held to the originals, the two GGUFs are llama.cpp's Q8_0 of them to 1e-7, in that
order. For the 4B, whose float32 is 17 GB, the engine's logits on the original's weights are within 1.4e-4 of
transformers' at 96 positions, and its perplexity on 192 tokens the same (7.873); on the GGUF's weights it is 7.574,
lower than the original's. On 1,500 tokens of English Wikipedia (the kernels, one CI runner each):

| model | float32 original | GGUF's weights, NumPy | 8-bit activations | 7-bit activations |
|---|---:|---:|---:|---:|
| Qwen3.5 2B | not measured | 16.589 | 16.608 | 16.873 |
| Qwen3.5 4B | 14.834 (transformers) | not measured | 14.492 | 14.710 |
| Qwen3.5 9B | not measured | not measured | 11.707 | 11.806 |

LFM2.5 (Liquid AI: convolution layers among attention layers) loses more to quantization than the other models, and
not in the order of the bits. On 1,500 tokens of English and of Japanese Wikipedia (CI's runners; the percentages are
against the float32 original):

| model | text | float32 original | the Q8_0 GGUF's values, NumPy | 8-bit activations (Safari) | 7-bit activations |
|---|---|---:|---:|---:|---:|
| LFM2.5 350M | English | 70.511 | 73.093 (+3.66%) | 76.562 (+8.58%) | 72.464 (+2.77%) |
| LFM2.5 350M | Japanese | 34.362 | 34.662 (+0.87%) | 35.693 (+3.87%) | 35.283 (+2.68%) |
| LFM2.5 1.2B JP | English | 18.255 | 18.229 (−0.14%) | 18.427 (+0.94%) | 18.962 (+3.87%) |
| LFM2.5 1.2B JP | Japanese | 16.829 | 16.785 (−0.26%) | 17.025 (+1.17%) | 17.491 (+3.94%) |

The second column is llama.cpp's own Q8_0 multiplied in float32, with none of this project's kernels: the 350M loses
3.66% to it on the English text. The page's int8 of a Q8_0 file is those same values. Of what 8-bit activations cost
the 350M, nine tenths is in the input of the query, key and value matrices of its attention layers (a NumPy copy of
the rounding, 300 tokens), and that part does not shrink with more bits; why is not settled. The same model moves its
logits by 17 when the norms' epsilon is 1e-6 instead of 1e-5. The answers it writes are not broken on either path.

## Other small effects

- A BOS token at the start: the page always starts with one, while some models are used without it. The
  difference was within ±3% on plain text, in either direction depending on the text, so it was kept. Two models
  were different. DeepSeek-R1 Distill Qwen 1.5B broke without the right BOS, and its entry now names it. Qwen3.5
  0.8B, whose layers mostly keep a state, is 20% to 65% worse on 512 tokens of plain text with the converter's
  `<|endoftext|>` in front (four Wikipedia texts, English and Japanese; the damage lasts: 5% to 30% still at tokens 256
  to 512), 1% to 4% worse with `<|im_start|>` and 0% to 1% with a newline. Its entries begin with the chat format's
  own first token, `<|im_start|>`, so the page sends the ids the real template writes. In chat form the
  first token does not change how likely an answer written by hand is (24 answers, `tests/chat_nll.py`: −0.8% in
  perplexity with `<|endoftext|>` in front, worse on 12 of 24), but it changes what the model writes: along the
  model's own answers the next-token distributions move by 0.12 nats a token (0.18 when it thinks), and the most
  likely token changes at 13% (9%) of the positions. The page's engine agrees with transformers on the most likely
  token at 98.9% (99.6%) of the positions with the entry's way of beginning, and at 86.7% (91.4%) with the old one.
  What `?hf=Qwen/Qwen3.5-0.8B` still loses is plain text, where there is no chat format and the converter's BOS
  comes first.
- RMSNorm's epsilon: 1e-5 for all models until Qwen3 0.6B showed +0.12% with it; the converter now passes the
  model's own value.
- The order of rounding inside the int8 kernels changed twice on 2026-09-27 (one scaling per group instead of
  four, and the relaxed SIMD correction added as an integer). The first moved perplexity by less than 0.1%
  (llm-jp-3-150m 29.992 → 29.997, tiny-lm 88.347 → 88.262); the second by −0.31 to +0.59% over eight comparisons,
  in no fixed direction, as much as between a float16 and a float32 KV cache.
- The KV cache: float16 (with threads) cannot be told from float32 (llm-jp-3-150m: 29.957 against 30.094). On the
  GPU the keys and values are float16 too.
