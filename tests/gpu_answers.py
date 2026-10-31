# gpu_answers.py (T.465.1)
# What the engine's shaders compute, in NumPy, for the stand-in WebGPU device of the checks that run the model's GPU
# worker without a GPU (tests/fakes/webgpu.mjs's answer; tests/fakes/gpu-answers.ts hands each dispatch's buffers
# here, in Pyodide, and writes back what these return). The owner's (2026-10-31): Python where it can be, JavaScript the
# bridge alone. Each is the arithmetic its shader means, on the engine's own Python where there is one (engine.packing:
# the int8 quantization quantize_x reproduces, the ternary codes), in float64 sums stored as float32.
#
# The same as the shaders: what each writes, from the bindings as they lay out. Simpler: the whole output whatever the
# workgroups dispatched, and none of a form's tiles, f16 or subgroups.
import numpy as np

from engine.packing import quantize as quantized, unpack_ternary
from convert.grids import codebooks as grids
from convert.superblocks import SUPERBLOCKS

GROUP = 32


def bytes(raw, _bytes=bytes):
    """a buffer's bytes as JavaScript hands them (a Uint8Array: its to_py() is a memoryview), or bytes"""
    return raw.to_py().tobytes() if hasattr(raw, "to_py") else _bytes(raw)


def halves(raw):
    """float16 bits (bytes) -> float64 values"""
    return np.frombuffer(bytes(raw), dtype=np.float16).astype(np.float64)


def flash_tile(q, keys, values, heads, kv_heads, scale, window, tokens, pos, size):
    """The prompt's attention (shaders/attention.ts's flashTile): q float32 [tokens][heads][size], the keys and values
    float16 [positions][kv heads][size]; token t at pos + t attends the positions up to its own (the last window of them
    where window is not 0). -> the output, float32 [tokens][heads][size], as bytes"""
    q = np.frombuffer(bytes(q), dtype=np.float32).astype(np.float64)
    k, v = halves(keys), halves(values)
    kv_dim, q_dim = kv_heads * size, heads * size
    k, v = k[: len(k) // kv_dim * kv_dim].reshape(-1, kv_dim), v[: len(v) // kv_dim * kv_dim].reshape(-1, kv_dim)
    out = np.zeros(tokens * q_dim)
    for t in range(tokens):
        seen = pos + t + 1
        first = max(0, seen - window) if window else 0
        for h in range(heads):
            kv = h // (heads // kv_heads) * size
            row = q[t * q_dim + h * size: t * q_dim + (h + 1) * size]
            scores = np.full(seen, -np.inf)
            scores[first:] = k[first:seen, kv: kv + size] @ (row * scale)
            weights = np.exp(scores - scores.max())
            out[t * q_dim + h * size: t * q_dim + (h + 1) * size] = weights @ v[:seen, kv: kv + size] / weights.sum()
    return out.astype(np.float32).tobytes()


def quantize(x, n, stride, tokens):
    """QUANTIZE: x float32 [tokens][stride], the first n of each token -> (codes int8 [tokens][stride], scales float32
    [tokens][stride / 32]) as bytes, as quantize_x writes them (engine.packing's quantize, which the kernel reproduces)"""
    x = np.frombuffer(bytes(x), dtype=np.float32)
    codes, scales = np.zeros(tokens * stride, dtype=np.int8), np.zeros(tokens * stride // GROUP, dtype=np.float32)
    for t in range(tokens):
        c, s = quantized(x[t * stride: t * stride + n].reshape(1, n))
        codes[t * stride: t * stride + n] = c.reshape(-1)
        scales[t * stride // GROUP: t * stride // GROUP + n // GROUP] = s
    return codes.tobytes(), scales.tobytes()


def codebooks():
    """(T.465.2) the codebooks of llama.cpp's IQ types as a checkpoint carries them (convert/grids.py's), as bytes"""
    return grids()[0]


def tiled(packing, stored, scales, x, xs, y, rows, n, x_stride, y_stride, add, tokens):
    """A tiled form of a prompt's matrices (shaders/prompt.ts's promptForms): rows of n weights (int8 with a float32 scale
    a group of 32 of a row; ternary codes with one a group of 128; T.465.2: blocks of a type of llama.cpp's, read by the
    converter's own readers, convert/superblocks.py's, whose scales are in the blocks and whose codebooks are the ones
    codebooks() gives: the check's are the plan's, which a check writes from there), x float32 [tokens][x_stride] (xs None) or its int8
    codes with their scales (packed: xs a float32 a group of 32), y float32 [tokens][y_stride]. -> y = W x, or y + W x
    where add, as bytes"""
    if packing == "int8":
        w = np.frombuffer(bytes(stored), dtype=np.int8)[: rows * n].astype(np.float64).reshape(rows, n)
        per = GROUP
    elif packing == "ternary":
        w = unpack_ternary(np.frombuffer(bytes(stored), dtype=np.uint8)[: rows * n // 4]).astype(np.float64).reshape(rows, n)
        per = 4 * GROUP
    elif packing in SUPERBLOCKS:
        _, size, read = SUPERBLOCKS[packing]
        w = read(bytes(stored)[: rows * n // 256 * size]).astype(np.float64).reshape(rows, n)
        per = None
    else:
        raise ValueError(f"gpu_answers: no answer for {packing}")
    if per:
        s = np.frombuffer(bytes(scales), dtype=np.float32)[: rows * n // per].astype(np.float64).reshape(rows, n // per)
        w = w * np.repeat(s, per, axis=1)
    y = np.frombuffer(bytes(y), dtype=np.float32).astype(np.float64).copy()
    if xs is None:
        values = np.frombuffer(bytes(x), dtype=np.float32).astype(np.float64)
        inputs = [values[t * x_stride: t * x_stride + n] for t in range(tokens)]
    else:
        codes, groups = np.frombuffer(bytes(x), dtype=np.int8).astype(np.float64), np.frombuffer(bytes(xs), dtype=np.float32).astype(np.float64)
        inputs = [codes[t * x_stride: t * x_stride + n] * np.repeat(groups[t * x_stride // GROUP: t * x_stride // GROUP + n // GROUP], GROUP) for t in range(tokens)]
    for t, values in enumerate(inputs):
        product = w @ values
        y[t * y_stride: t * y_stride + rows] = (y[t * y_stride: t * y_stride + rows] if add else 0) + product
    return y.astype(np.float32).tobytes()
