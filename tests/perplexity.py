# perplexity.py
# The perplexity of a model on a text, for each way the engine can compute: NumPy (int8 weights widened to float32,
# activations left alone), the kernels with 8-bit activations (matmul_q8) and the kernels with 7-bit activations
# (matmul_q8r, relaxed SIMD). tests/perplexity.mjs runs this inside Pyodide.
#
# Input: the globals MODEL ({"checkpoint", "tokenizer", "options"}), TEXT, TOKENS and WINDOW. Output: a JSON string.
# MODEL["file"] (T229): the checkpoint as a file of this machine, read straight into forward.js's memory
# (tests/engine.mjs's kernel_llama_file), for a model whose weights widened to float32 pass Pyodide's 4 GiB: the
# kernels' rows alone, and NumPy's from tests/perplexity_native.py. MODEL["rows"] (T247): which of the rows, by number.
# tests/perplexity_native.py imports perplexity() from here, for the float32 originals that Pyodide cannot hold.
import gc
import json
import math
import time

import numpy as np

from llama2_numpy import Llama



def perplexity(llama, tokens, window):
    """exp of the mean negative log likelihood; the context starts anew every window tokens, from BOS."""
    total, count = 0.0, 0
    for start in range(0, len(tokens), window - 1):
        piece = [llama.bos] + tokens[start:start + window - 1]
        for pos in range(len(piece) - 1):
            logits = np.asarray(llama.forward(piece[pos], pos), dtype=np.float64)
            logits -= logits.max()
            total -= logits[piece[pos + 1]] - math.log(np.exp(logits).sum())
            count += 1
    return math.exp(total / count), count


def main():
    read = lambda name: open(name, "rb").read()
    file = MODEL.get("file")  # noqa: F821
    checkpoint, vocabulary = None if file else read(MODEL["checkpoint"]), read(MODEL["tokenizer"])  # noqa: F821
    results = []
    # the 8-bit row is the kernels without relaxed SIMD (T52's switch; it used to patch load_kernels)
    variants = [("kernels, 7-bit activations (matmul_q8r)", "simdkernel.so", ()),
                # T110: the same with the keys and values kept in float32 instead of float16
                ("kernels, 7-bit activations, keys and values in float32", "simdkernel.so", ("kv16",)),
                ("kernels, 8-bit activations (matmul_q8)", "simdkernel.so", ("relaxed",)),
                ("NumPy, activations not quantized", None, ())]
    if file:
        variants = [variant for variant in variants if variant[1]]
    if MODEL.get("rows") is not None:  # noqa: F821
        # T247: some of the rows only, a process each: two memories of a 10 GB model at once are more than a runner has,
        # and the first is not given back before the second is made
        variants = [variants[row] for row in MODEL["rows"]]  # noqa: F821
    tokens = None
    for label, kernels, disable in variants:
        # the kernels' rows run the forward pass of forward.js, as the page does (tests/engine.mjs, T93)
        options = dict(MODEL["options"], disable=disable)  # noqa: F821
        if file:
            llama = kernel_llama_file(file, vocabulary, **options)  # noqa: F821
        else:
            llama = kernel_llama(checkpoint, vocabulary, **options) if kernels else Llama(checkpoint, vocabulary, **options)  # noqa: F821
        if tokens is None:
            tokens = llama.tokenizer.encode(TEXT)[:TOKENS]  # noqa: F821
        started = time.perf_counter()
        value, count = perplexity(llama, tokens, min(WINDOW, llama.seq_len))  # noqa: F821
        results.append({"variant": label, "backend": llama.backend, "perplexity": value, "tokens": count,
                        "seconds": time.perf_counter() - started})
        llama.release()  # what forward.js holds for it (the memory of a model of gigabytes, with --file)
        del llama
        gc.collect()
    return json.dumps(results)


if "MODEL" in globals():
    RESULT = main()
