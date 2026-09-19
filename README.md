# Pyodide Llama

Run Llama 2 in your browser using Python and WebAssembly!

This project leverages [Pyodide](https://pyodide.org/) to run a Python implementation of Llama 2 (`llama2_numpy.py`, a NumPy port of `llama2.py`) directly in the web browser. It is an experiment in how far Python on WebAssembly can go, not a product. The default model is [llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m), which writes Japanese and English; the much smaller [tiny-lm](https://huggingface.co/sbintuitions/tiny-lm) is four times as fast but far less coherent; the TinyStories models from the [TinyLlamas](https://huggingface.co/karpathy/tinyllamas) project can be selected as well.

## Features

- **Pure Browser-based Inference:** No backend server required for inference.
- **Python in WebAssembly:** Python sequences the transformer layers, and small WASM SIMD kernels, loaded with `ctypes` and working in place on NumPy memory, do the math: 75 tokens/s for the 150M parameter model and 300 to 950 for the small ones, about a thousand times faster than the original pure Python loops (NumPy alone reaches 50). See [Measurements](#measurements).
- **Streaming Output:** Pyodide runs in a Web Worker and every token is shown as soon as it is generated, so the page never freezes. While a text is being written, the send button stops it.
- **Settings You Can See:** every answer says which temperature and seed wrote it and how fast; the button left of the prompt changes them, and the seed under an answer is a button that fixes it, so that two models can be compared on the same seed.
- **Your Own Model:** a llama2.c checkpoint from your disk runs without being uploaded ([how](#your-own-model)).
- **Several Models:** Japanese / English models (llm-jp-3 with 150M parameters by default, tiny-lm with 29M), and TinyStories models from 260K to 42M parameters. `?model=<id>` selects one directly.

## Live Demo

You can try the live demo on GitHub Pages (if configured):
[https://takano32.github.io/pyodide-llama-py/](https://takano32.github.io/pyodide-llama-py/)

## Getting Started

### Prerequisites

- Node.js 24 LTS (the page is built with [Astro](https://astro.build/))
- Python 3 with NumPy (`make` converts the tiny-lm checkpoint with `convert_hf.py` and quantizes with `quantize.py`)
- Docker (optional)

### Running Locally with Makefile

1. Clone the repository:
   ```bash
   git clone https://github.com/takano32/pyodide-llama-py.git
   cd pyodide-llama-py
   ```

2. Run the application:
   ```bash
   make run
   ```
   This will download the model files (about 1 GB) and convert tiny-lm, install dependencies, and start the Astro dev server at `http://localhost:8080/pyodide-llama-py/`.

### Running Locally with Docker

1. Build and run the Docker container:
   ```bash
   docker build -t pyodide-llama-py .
   docker run -p 8080:8080 pyodide-llama-py
   ```
2. Open `http://localhost:8080/pyodide-llama-py/` in your browser.

## Your Own Model

The folder button next to the model list (or dropping the files on the page) opens a model from your own disk.
The files are read where they are: nothing is uploaded, and nothing is requested from the network. Choose them
together:

- a checkpoint in llama2.c's legacy format, for example `stories110M.bin` of the
  [TinyLlamas](https://huggingface.co/karpathy/tinyllamas), or what `convert_hf.py` and `quantize.py` write.
  float32, float16 and int8 are told apart by the header and the file size;
- its `tokenizer.bin` (llama2.c's format; it must hold exactly the vocabulary of the checkpoint);
- optionally a `.json` with whatever differs from llama2.c's conventions, shaped like an entry of `src/models.js`:

```json
{ "name": "tiny-lm", "options": { "tokenizer_kind": "unigram", "nfkc": true, "stop_tokens": [1, 2] },
  "generation": { "steps": 256, "temperature": 0.7, "topp": 0.9, "repetition_penalty": 1.3 }, "prompt": "昔々、" }
```

WebAssembly addresses 32 bits and a phone gives a tab far less, so a checkpoint of more than 1 GB asks first.

The files Hugging Face publishes work as well: choose `model.safetensors`, `config.json` and the tokenizer
(`tokenizer.json` of the Unigram kind, or a sentencepiece `tokenizer.model`) together, and the page converts them
in the browser, with the same Python code that builds the models of this site (`public/llama2_convert.py`). It
reads the weights a few megabytes at a time and writes int8 directly, so llm-jp-3-150m (305 MB of bfloat16)
takes 7 seconds and no more memory than the converted model itself, and then writes, seed for seed, what the
site's own copy writes. A `.json` next to them may say `{"conversion": {"dtype": "float16", "max_seq_len": 1024}}`
(the defaults are int8 and a context of 512 tokens). Only plain Llama models are accepted.

## How it Works

1. **Pyodide Initialization:** The browser resolves the latest Pyodide release at page load and loads that runtime from the CDN, so there is no version to bump by hand. Append `?pyodide=<version>` to the URL to force a specific version.
2. **Environment Setup:** A Web Worker (`public/worker.js`) loads Pyodide, NumPy, `public/llama2_numpy.py` and the SIMD kernels (`kernels/`, compiled by `make kernels`; `?kernel=off` runs on NumPy alone). The chat-like page itself is `src/pages/index.astro`, and the model list is `src/models.js`.
3. **Model Loading:** The selected model checkpoint and its tokenizer are downloaded while Pyodide is still loading, in parts of 8 MiB over several connections at once (about 1.8x as fast as one stream), straight into one preallocated buffer while a progress bar shows the download. The larger models are distributed as int8 (3.5x smaller; measured perplexity cost on stories15M: +0.04%) and widened to float32 once, and their unquantized originals can be selected for comparison; float32 weights of the small models are NumPy views into the buffer, nothing is copied.
4. **Inference:** When you press the send button (or Ctrl / Cmd + Enter; Enter alone breaks the line), the prompt is sent to the worker, where a Python generator yields the text token by token; each piece is posted back and appended to the output.

No binary is committed to this repository: `make models` downloads the model files when the site is deployed (or for `make run`). llm-jp-3 and tiny-lm are published in Hugging Face format, so `convert_hf.py` converts them, with nothing but NumPy, into the llama2.c checkpoint and tokenizer formats that `llama2_numpy.py` reads, and `quantize.py` turns the larger checkpoints into int8. Their tokenizers are sentencepiece unigram models, which `llama2_numpy.py` encodes with a Viterbi search (the Llama 2 vocabulary of the TinyStories models uses llama2.c's pair merging).

## Measurements

All numbers below were measured on 2026-09-19 on one phone-class ARM CPU (Cortex-A78 x4 + A55 x4), a single
thread, no swap. The long version, including the survey of the other browser ports, is in this
[gist](https://gist.github.com/takano32/196c6f93979ad44f98cee5712fdd3901).

Tokens per second, stories15M, greedy:

| implementation | tok/s |
|---|---:|
| pure Python (`llama2.py`) | 0.26 |
| NumPy in Pyodide (Node), float32 | 53 |
| NumPy in Pyodide (Node), int8 | 55 |
| SIMD kernels in Pyodide (Node), float32 | 200 |
| SIMD kernels in Pyodide (Node), int8 | 351 |
| native llama2.c, `gcc -Ofast`, for reference | 214 |

Plain JavaScript reached 38 tok/s and a standalone WASM SIMD build 170-190 on the same machine. Threads did not
help at this model size, not even with native OpenMP, so the site does not use them.

In Chromium, per model (`?kernel=off` gives the NumPy column):

| model | NumPy | kernels |
|---|---:|---:|
| stories260K float32 | 268 | 951 |
| stories3_5M float32 | 141 | 402 |
| stories15M float32 / int8 | 50 | 186 / 296 |
| tiny-lm int8, sampled with a repetition penalty | 43 | 271-296 |
| llm-jp-3-150m int8, sampled, 256 tokens | 8.5 | 75-79 |

The tiny-lm and llm-jp rows are the current defaults. They were 171 and 47 tok/s while NumPy still did the
sampling, and 252-274 and 61-66 before attention learned to walk its cache row by row (T54). Firefox 150 loads the kernels as well, but all of its
WebAssembly was 5-7x slower on this machine, NumPy included. Safari's engine is tested on a macOS runner of GitHub Actions
(Playwright's WebKit 26.4 on an Apple M1, a much faster CPU): it has no relaxed SIMD, so int8 runs on the plain
SIMD kernel, and llm-jp-3-150m reaches 146 tok/s there. Safari itself, and a real iPhone, were not measured.

Where a token of the default model goes (`node tests/profile.mjs`, the same in Node and in Chromium): the matrix
products of the layers 52%, the classifier over 99584 tokens 38%, the ctypes calls 6%, sampling 3%, Python and
NumPy around the calls 1%. The interpreter is no longer what limits it. Nor is memory bandwidth: the int8 product
is as fast on a matrix of 75 MB as on one that fits the cache, so its arithmetic is the limit
(details in [kernels/README.md](kernels/README.md)).

Memory, llm-jp-3-150m int8: the kernels multiply the int8 weights as they are instead of widening them to
float32, which takes the WASM heap from 897 MB to 283 MB.

Quantization to int8 (groups of 32, one float32 scale per group) is not distinguishable from the original in
perplexity, while int4 is:

| model | original | int8 |
|---|---:|---:|
| stories15M | - | +0.04% |
| tiny-lm | 91.3 | 91.1 |
| llm-jp-3-150m | 22.76 | 22.69 |

The most likely token agrees about 98% of the time; greedy output diverges from the original part way through
but stays coherent. int4 cost +16.8% perplexity and was rejected.

Downloading the model in parts of 8 MiB over several connections at once is about 1.8x faster than one stream:
167 MB took 20.4 s in one stream and 11.2 s split, against the production CDN.

## Acknowledgments

- [Pyodide](https://pyodide.org/) for the Python WebAssembly runtime.
- [llama2.py](https://github.com/tairov/llama2.py) by tairov for the pure Python Llama 2 implementation.
- [llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m) by LLM-jp (Apache License 2.0) for the default Japanese / English model.
- [tiny-lm](https://huggingface.co/sbintuitions/tiny-lm) by SB Intuitions (MIT License) for the Japanese / English model; its license is deployed next to the converted file.
- [TinyLlamas](https://huggingface.co/karpathy/tinyllamas) by Andrej Karpathy and [ellishg/tinyllamas](https://huggingface.co/ellishg/tinyllamas) for the compact TinyStories checkpoints.
- [llama2.c](https://github.com/karpathy/llama2.c) for the inspiration and model format.
