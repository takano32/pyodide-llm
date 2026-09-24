# perplexity.py
# The perplexity of a model on a text, for each way the engine can compute: NumPy (int8 weights widened to float32,
# activations left alone), the kernels with 8-bit activations (matmul_q8) and the kernels with 7-bit activations
# (matmul_q8r, relaxed SIMD). tests/perplexity.mjs runs this inside Pyodide.
#
# Input: the globals MODEL ({"checkpoint", "tokenizer", "options"}), TEXT, TOKENS and WINDOW. Output: a JSON string.
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
    checkpoint, vocabulary = read(MODEL["checkpoint"]), read(MODEL["tokenizer"])  # noqa: F821
    results = []
    # the 8-bit row is the kernels without relaxed SIMD (T52's switch; it used to patch load_kernels)
    variants = [("kernels, 7-bit activations (matmul_q8r)", "simdkernel.so", ()),
                ("kernels, 8-bit activations (matmul_q8)", "simdkernel.so", ("relaxed",)),
                ("NumPy, activations not quantized", None, ())]
    tokens = None
    for label, kernels, disable in variants:
        llama = Llama(checkpoint, vocabulary, kernels=kernels, disable=disable, **MODEL["options"])  # noqa: F821
        if tokens is None:
            tokens = llama.tokenizer.encode(TEXT)[:TOKENS]  # noqa: F821
        started = time.perf_counter()
        value, count = perplexity(llama, tokens, min(WINDOW, llama.seq_len))  # noqa: F821
        results.append({"variant": label, "backend": llama.backend, "perplexity": value, "tokens": count,
                        "seconds": time.perf_counter() - started})
        del llama
        gc.collect()
    return json.dumps(results)


if "MODEL" in globals():
    RESULT = main()
