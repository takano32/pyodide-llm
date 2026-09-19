# profile_token.py
# Where the time of one token goes, measured inside Pyodide (tests/profile.mjs runs this in Node or in a browser).
# A browser's clock is too coarse to time single kernel calls, so everything here is a difference of totals: the
# same model runs its forward pass with the real kernels, with kernels that only count, and with kernels that do
# nothing; a kernel called on nothing (n = 0) costs exactly one ctypes call.
#
# Input: the global MODELS, a list of {"name", "checkpoint", "tokenizer", "options", "generation"} (file names in the
# working directory). Output: a JSON string.
import gc
import json
import time

import numpy as np

import llama2_numpy
from llama2_numpy import Llama, load_kernels

WARMUP, POSITIONS = 16, 64  # tokens at positions 0..15 warm up, 16..79 are timed


def seconds_per_call(function, repeat):
    function()
    started = time.perf_counter()
    for _ in range(repeat):
        function()
    return (time.perf_counter() - started) / repeat


def call_costs(kernels):
    """The cost of one ctypes call of each kernel: called on nothing, it returns at once."""
    scratch = np.zeros(64, dtype=np.float32)
    p = scratch.ctypes.data
    nothing = {"matmul_f32": (p, p, p, 0, 0, 0), "quantize_x": (p, p, p, 0, 0), "matmul_q8": (p, p, p, p, p, 0, 0, 0),
               "matmul_q8r": (p, p, p, p, p, p, 0, 0, 0), "rmsnorm": (p, p, p, 0), "rope": (p, p, p, 0, 0),
               "attention": (p, p, p, p, p, 0, 0, 1, 2), "swiglu": (p, p, p, 0), "add_inplace": (p, p, 0)}
    return {name: seconds_per_call(lambda f=kernels[name], a=arguments: f(*a), 20000)
            for name, arguments in nothing.items() if name in kernels}


def forward_seconds(forward, need_logits=True):
    for pos in range(WARMUP):
        forward(100 + pos, pos, need_logits)
    started = time.perf_counter()
    for pos in range(WARMUP, WARMUP + POSITIONS):
        forward(100 + pos, pos, need_logits)
    return (time.perf_counter() - started) / POSITIONS


def profile(model):
    kernels = load_kernels("simdkernel.so")
    costs = call_costs(kernels)
    read = lambda name: open(name, "rb").read()
    llama = Llama(read(model["checkpoint"]), read(model["tokenizer"]), kernels="simdkernel.so", **model["options"])
    int8 = "int8" in llama.backend
    alive = [llama._kernel_buffers, getattr(llama, "_corrections", None)]  # the first forward() only knows addresses

    # 1. the real thing: all of a token, and a token without the classifier
    full = min(forward_seconds(llama.forward) for _ in range(3))
    layers_only = min(forward_seconds(llama.forward, need_logits=False) for _ in range(3))

    # 2. how often is each kernel called for one token?
    counts = {}

    def counting(name, function):
        def call(*arguments):
            counts[name] = counts.get(name, 0) + 1
            return function(*arguments)
        return call

    counted = llama.kernel_forward({name: counting(name, f) for name, f in kernels.items()}, int8)
    alive += [llama._kernel_buffers, getattr(llama, "_corrections", None)]
    counted(100, 0)
    calls = sum(counts.values())
    call_overhead = sum(costs[name] * count for name, count in counts.items())
    classifier_calls = {name: count for name, count in counts.items()}
    counts_layers = dict(counts)
    counts.clear()
    counted(100, 1, False)
    classifier_overhead = call_overhead - sum(costs[name] * count for name, count in counts.items())

    # 3. the same forward pass with kernels that do nothing: what Python and NumPy cost around the calls
    idle = llama.kernel_forward({name: (lambda *arguments: None) for name in kernels}, int8)
    alive += [llama._kernel_buffers, getattr(llama, "_corrections", None)]
    glue = min(forward_seconds(idle) for _ in range(3))

    # 4. sampling with the model's own settings, on real logits
    generation = model["generation"]
    sampling = 0.0
    if generation.get("temperature"):
        logits = llama.forward(100, 0).copy()
        rng, history = np.random.default_rng(1), list(range(100, 164))
        scratch = np.empty_like(logits)

        def draw():
            scratch[:] = logits
            if generation.get("repetition_penalty", 1.0) != 1.0:
                llama.penalize(scratch, history, generation["repetition_penalty"])
            llama.sample(scratch, generation["temperature"], generation.get("topp", 0.9), rng)

        copy = seconds_per_call(lambda: scratch.__setitem__(slice(None), logits), 200)
        sampling = seconds_per_call(draw, 200) - copy
    else:
        logits = llama.forward(100, 0)
        sampling = seconds_per_call(lambda: int(np.argmax(logits)), 200)

    weights = lambda tensor: sum(part.nbytes for part in tensor) if isinstance(tensor, tuple) else tensor.nbytes
    per_token = sum(weights(t) for t in (llama.wq, llama.wk, llama.wv, llama.wo, llama.w1, llama.w2, llama.w3))
    result = {
        "name": model["name"], "backend": llama.backend, "layers": llama.n_layers, "dim": llama.dim, "vocabulary": llama.vocab_size,
        "token_ms": (full + sampling) * 1e3, "tokens_per_second": 1.0 / (full + sampling),
        "calls": calls, "call_us": call_overhead / calls * 1e6,
        "kernels_layers_ms": (layers_only - glue - (call_overhead - classifier_overhead)) * 1e3,
        "kernels_classifier_ms": (full - layers_only - classifier_overhead) * 1e3,
        "ctypes_ms": call_overhead * 1e3, "python_ms": glue * 1e3, "sampling_ms": sampling * 1e3,
        "weights_read_mb": (per_token + weights(llama.wcls)) / 1e6, "classifier_read_mb": weights(llama.wcls) / 1e6,
    }
    del llama, counted, idle, alive
    gc.collect()
    return result


def bandwidth():
    """Multiply-adds per second of the matmul kernels on matrices that fit a cache and on ones that do not."""
    kernels = load_kernels("simdkernel.so")
    n, results = 768, []
    generator = np.random.default_rng(0)
    x = generator.standard_normal(n).astype(np.float32)
    xq, xs = np.zeros(n, dtype=np.int8), np.zeros(n // 32, dtype=np.float32)
    for rows in (16, 64, 512, 4096, 32768, 98304):
        repeat = max(2, 400000 // rows)
        values = generator.integers(-127, 128, size=(rows, n), dtype=np.int8)
        scales = np.full((rows, n // 32), 0.01, dtype=np.float32)
        corrections = (scales * values.reshape(rows, -1, 32).sum(axis=-1, dtype=np.int32)).astype(np.float32)
        out = np.zeros(rows, dtype=np.float32)
        # addresses are taken once: array.ctypes.data costs more than a small matmul
        out_p, xq_p, xs_p, x_p = (array.ctypes.data for array in (out, xq, xs, x))
        values_p, scales_p, corrections_p = values.ctypes.data, scales.ctypes.data, corrections.ctypes.data
        q8, q8r, f32, quantize_x = kernels["matmul_q8"], kernels.get("matmul_q8r"), kernels["matmul_f32"], kernels["quantize_x"]
        row = {"rows": rows, "int8_mb": values.nbytes / 1e6}
        variants = [("matmul_q8", 0, lambda: q8(out_p, xq_p, xs_p, values_p, scales_p, n, 0, rows))]
        if q8r:
            variants.append(("matmul_q8r", 64, lambda: q8r(out_p, xq_p, xs_p, values_p, scales_p, corrections_p, n, 0, rows)))
        for name, bias, function in variants:
            quantize_x(xq_p, xs_p, x_p, n, bias)
            row[name] = rows * n / min(seconds_per_call(function, repeat) for _ in range(3)) / 1e9
        del values, scales, corrections
        if rows * n * 4 <= 320e6:
            matrix = generator.standard_normal((rows, n)).astype(np.float32)
            matrix_p = matrix.ctypes.data
            row["float32_mb"] = matrix.nbytes / 1e6
            row["matmul_f32"] = rows * n / min(seconds_per_call(lambda: f32(out_p, x_p, matrix_p, n, 0, rows), repeat) for _ in range(3)) / 1e9
            del matrix
        results.append(row)
        gc.collect()
    return results


json.dumps({"models": [profile(model) for model in MODELS], "bandwidth": bandwidth()})  # noqa: F821
