# Pyodide Llama

Run Llama 2 in your browser using Python and WebAssembly!

This project leverages [Pyodide](https://pyodide.org/) to run a Python implementation of Llama 2 (`llama2_numpy.py`, a NumPy port of `llama2.py`) directly in the web browser. It is an experiment in how far Python on WebAssembly can go, not a product. The default model is [llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m), which writes Japanese and English; the much smaller [tiny-lm](https://huggingface.co/sbintuitions/tiny-lm) is three times as fast but far less coherent; the TinyStories models from the [TinyLlamas](https://huggingface.co/karpathy/tinyllamas) project can be selected as well.

## Features

- **Pure Browser-based Inference:** No backend server required for inference.
- **Python in WebAssembly:** Python sequences the transformer layers, and small WASM SIMD kernels, loaded with `ctypes` and working in place on NumPy memory, do the math: 50 tokens/s for the 150M parameter model and 300 to 900 for the small ones, about a thousand times faster than the original pure Python loops (NumPy alone reaches 50).
- **Streaming Output:** Pyodide runs in a Web Worker and every token is shown as soon as it is generated, so the page never freezes.
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

## How it Works

1. **Pyodide Initialization:** The browser resolves the latest Pyodide release at page load and loads that runtime from the CDN, so there is no version to bump by hand. Append `?pyodide=<version>` to the URL to force a specific version.
2. **Environment Setup:** A Web Worker (`public/worker.js`) loads Pyodide, NumPy, `public/llama2_numpy.py` and the SIMD kernels (`kernels/`, compiled by `make kernels`; `?kernel=off` runs on NumPy alone). The chat-like page itself is `src/pages/index.astro`, and the model list is `src/models.js`.
3. **Model Loading:** The selected model checkpoint and its tokenizer are downloaded while Pyodide is still loading, in parts of 8 MiB over several connections at once (about twice as fast as one stream), straight into one preallocated buffer while a progress bar shows the download. The larger models are distributed as int8 (3.5x smaller; measured perplexity cost on stories15M: +0.04%) and widened to float32 once, and their unquantized originals can be selected for comparison; float32 weights of the small models are NumPy views into the buffer, nothing is copied.
4. **Inference:** When you click "Run", the prompt is sent to the worker, where a Python generator yields the text token by token; each piece is posted back and appended to the output.

No binary is committed to this repository: `make models` downloads the model files when the site is deployed (or for `make run`). llm-jp-3 and tiny-lm are published in Hugging Face format, so `convert_hf.py` converts them, with nothing but NumPy, into the llama2.c checkpoint and tokenizer formats that `llama2_numpy.py` reads, and `quantize.py` turns the larger checkpoints into int8. Their tokenizers are sentencepiece unigram models, which `llama2_numpy.py` encodes with a Viterbi search (the Llama 2 vocabulary of the TinyStories models uses llama2.c's pair merging).

## Acknowledgments

- [Pyodide](https://pyodide.org/) for the Python WebAssembly runtime.
- [llama2.py](https://github.com/tairov/llama2.py) by tairov for the pure Python Llama 2 implementation.
- [llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m) by LLM-jp (Apache License 2.0) for the default Japanese / English model.
- [tiny-lm](https://huggingface.co/sbintuitions/tiny-lm) by SB Intuitions (MIT License) for the Japanese / English model; its license is deployed next to the converted file.
- [TinyLlamas](https://huggingface.co/karpathy/tinyllamas) by Andrej Karpathy and [ellishg/tinyllamas](https://huggingface.co/ellishg/tinyllamas) for the compact TinyStories checkpoints.
- [llama2.c](https://github.com/karpathy/llama2.c) for the inspiration and model format.
