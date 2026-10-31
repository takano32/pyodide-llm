# gpu_answers.py (T.465.1)
# What the engine's shaders compute, in NumPy, for the stand-in WebGPU device of the checks that run the model's GPU
# worker without a GPU (tests/fakes/webgpu.mjs's answer; tests/fakes/gpu-answers.ts hands each dispatch's buffers
# here, in Pyodide, and writes back what these return). The owner's (2026-10-31): Python where it can be, JavaScript the
# bridge alone. Each is the arithmetic its shader means, on the engine's own Python where there is one (engine.packing:
# the int8 quantization quantize_x reproduces, the ternary codes), in float64 sums stored as float32.
#
# The same as the shaders: what each writes, from the bindings as they lay out. Simpler: the whole output whatever the
# workgroups dispatched, and none of a form's tiles, f16 or subgroups.
import math

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


# ---- T.465.4: a generated token's step (gpu/tokens.ts's passes): its row of the embedding, the fused matrices of its
# layer, and the sampler. Each takes the bytes of its bindings and returns the bytes of what it writes.

def embed(table, scales, state, n, first, rows):
    """EMBED: the state's token's row of an int8 table (groups of 32 with a float32 scale each), a piece of rows from
    first (rows 0: the whole table) -> the row, float32, as bytes; None where the token is not in the piece"""
    token = int(np.frombuffer(bytes(state), dtype=np.uint32)[4])
    row = token - first
    # (a token before the piece: u32's row wraps past rows, and nothing is written; T.465.4's review)
    if rows and not 0 <= row < rows:
        return None
    values = np.frombuffer(bytes(table), dtype=np.int8)[row * n: (row + 1) * n].astype(np.float32)
    d = np.frombuffer(bytes(scales), dtype=np.float32)[row * n // GROUP: (row + 1) * n // GROUP]
    return (values * np.repeat(d, GROUP)).astype(np.float32).tobytes()


def fused(norm, output, weights, scales, x, params, norm_weight, dst):
    """fusedMatVec (shaders/fused.ts): the rows of an int8 matrix (float32 scales a group of 32 of a row) times x, x
    through the norm first where norm (its weight at params' normAt, the scale of x's mean square and eps), into dst:
    "write" (as they are), "add" (to what is there), "swiglu" and "geglu" (the rows params' second after them are up's).
    -> dst, as bytes. (The "rope" write is fused_rope's.)"""
    rows, words, per_row, second, eps, norm_at = row_params(params)
    n = per_row * GROUP
    return product(norm, output, weights, scales, np.frombuffer(bytes(x), dtype=np.float32)[:n].astype(np.float64), params, norm_weight, dst)


def product(norm, output, weights, scales, v, params, norm_weight, dst):
    """fused()'s, of the vector v (float64) as it is read"""
    rows, words, per_row, second, eps, norm_at = row_params(params)
    n = per_row * GROUP
    scale = 1.0
    if norm:
        scale = 1.0 / math.sqrt(float(np.mean(v * v)) + eps)
        v = v * np.frombuffer(bytes(norm_weight), dtype=np.float32)[norm_at: norm_at + n]
    sums = matrix(weights, scales, per_row, rows + (second if output in ("swiglu", "geglu") else 0)) @ v * scale
    out = np.frombuffer(bytes(dst), dtype=np.float32).astype(np.float64).copy()
    value = sums[:rows]
    if output == "write":
        out[:rows] = value
    elif output == "add":
        out[:rows] += value
    elif output in ("swiglu", "geglu"):
        up = sums[second: second + rows]
        out[:rows] = (value / (1.0 + np.exp(-value)) if output == "swiglu"
                      else 0.5 * value * (1.0 + np.tanh(np.clip(0.7978845608028654 * (value + 0.044715 * value ** 3), -9.010913, 9.010913)))) * up
    else:
        raise ValueError(f"gpu_answers: no answer for the fused output {output}")
    return out.astype(np.float32).tobytes()


def fused_rope(norm, weights, scales, x, params, norm_weight, keys, values, angles, step):
    """fusedMatVec's "rope" write: its rows (q's, then k's, then v's) as fused() makes them, q's and k's turned by RoPE
    at the step's position (the first `turned` values of a head, in pairs), q into q, the keys and values into the cache
    at the position as float16 pairs -> (q, the keys, the values), as bytes"""
    words = np.frombuffer(bytes(params), dtype=np.uint32)
    q_rows, kv_rows, size, turned = (int(w) for w in words[6:10])
    rows = int(words[0])
    made = np.frombuffer(fused(norm, "write", weights, scales, x, params, norm_weight, np.zeros(rows, dtype=np.float32).tobytes()), dtype=np.float32).astype(np.float64)
    return rope_into(made, q_rows, kv_rows, size, turned, keys, values, angles, step)


def rope_into(made, q_rows, kv_rows, size, turned, keys, values, angles, step):
    """write_pair's (shaders/fused.ts), of rows q, k and v one after the other: -> (q, the keys, the values) as bytes"""
    pos = int(np.frombuffer(bytes(step), dtype=np.uint32)[1])
    table = np.frombuffer(bytes(angles), dtype=np.float32).astype(np.float64)
    half = size // 2
    cos, sin = table[pos * size: pos * size + half], table[pos * size + half: pos * size + size]

    def turn(rows):
        out = rows.copy()
        for at in range(0, len(rows), size):
            for i in range(0, min(turned, size), 2):
                c, s = cos[i // 2], sin[i // 2]
                v0, v1 = rows[at + i], rows[at + i + 1]
                out[at + i], out[at + i + 1] = v0 * c - v1 * s, v0 * s + v1 * c
        return out

    q = turn(made[:q_rows]).astype(np.float32)
    k = turn(made[q_rows: q_rows + kv_rows])
    v = made[q_rows + kv_rows: q_rows + 2 * kv_rows]
    cache_k = np.frombuffer(bytes(keys), dtype=np.float16).copy()
    cache_v = np.frombuffer(bytes(values), dtype=np.float16).copy()
    cache_k[pos * kv_rows: (pos + 1) * kv_rows] = k.astype(np.float16)
    cache_v[pos * kv_rows: (pos + 1) * kv_rows] = v.astype(np.float16)
    return q.tobytes(), cache_k.tobytes(), cache_v.tobytes()


def row_params(params):
    """the fused matrices' Params: rows, words of a row, groups of a row, up's rows after gate's, eps, the norm's start"""
    words = np.frombuffer(bytes(params), dtype=np.uint32)
    return int(words[0]), int(words[1]), int(words[2]), int(words[3]), float(np.frombuffer(bytes(params), dtype=np.float32)[4]), int(words[5])


def matrix(weights, scales, per_row, rows):
    """rows of an int8 matrix with a float32 scale a group of 32, as float64 [rows][per_row * 32]"""
    n = per_row * GROUP
    w = np.frombuffer(bytes(weights), dtype=np.int8)[: rows * n].astype(np.float64).reshape(rows, n)
    s = np.frombuffer(bytes(scales), dtype=np.float32)[: rows * per_row].astype(np.float64).reshape(rows, per_row)
    return w * np.repeat(s, GROUP, axis=1)


WINDOW = 64  # the State's recent tokens (shaders/run.ts's REPETITION_WINDOW, the CPU's)


def sample(logits, state, chosen, randoms, settings):
    """SAMPLE (shaders/sample.ts): the repetition penalty on the distinct tokens of the state's window, the presence
    penalty on those of them sampled (both into the logits, in place), then the token: the largest logit at temperature
    0, else the CPU's draw (engine.sampler's NumpySampler, with the run's random number of this step); into chosen and
    the state, as finish() writes them (a stop token: stopped). A step whose logits hold a NaN or +inf, or none over
    -3.4e38: not_finite and stopped, nothing chosen. -> (the logits, the state, chosen), as bytes"""
    from engine.sampler import NumpySampler, Sampling
    words = np.frombuffer(bytes(state), dtype=np.uint32).copy()
    out = np.frombuffer(bytes(logits), dtype=np.float32).copy()
    picks = np.frombuffer(bytes(chosen), dtype=np.uint32).copy()
    unit = np.frombuffer(bytes(settings), dtype=np.uint32)
    reals = np.frombuffer(bytes(settings), dtype=np.float32)
    vocab, stops, top_k = int(unit[0]), int(unit[1]), int(unit[2])
    presence, temperature, topp, penalty, min_p = (float(reals[i]) for i in range(3, 8))
    stop = set(int(t) for t in unit[8: 8 + stops])
    if words[7]:
        return out.tobytes(), words.tobytes(), picks.tobytes()
    history, written = int(words[6]), int(words[2])
    recent = words[8: 8 + WINDOW]
    window, latest = min(history, WINDOW), min(written, WINDOW)
    for t in range(window):
        token = int(recent[t])
        if token in (int(r) for r in recent[:t]) or token >= vocab:
            continue
        value = out[token]
        if penalty != 1.0:
            value = value / np.float32(penalty) if value > 0 else value * np.float32(penalty)
        if any(int(recent[j]) == token and (history - 1 - j) % WINDOW < latest for j in range(window)):
            value = value - np.float32(presence)
        out[token] = value
    view = out[:vocab]
    if np.isnan(view).any() or np.isposinf(view).any() or not (view > -3.4e38).any():
        words[3], words[7] = 1, 1
        return out.tobytes(), words.tobytes(), picks.tobytes()
    if temperature == 0.0:
        token = int(np.argmax(view))
    else:
        r = float(np.frombuffer(bytes(randoms), dtype=np.float32)[int(words[5])])

        class Given:
            def random(self):
                return r
        draw = NumpySampler().drawing(Sampling(temperature=temperature, topp=topp, top_k=top_k, min_p=min_p), Given())
        token = draw(view.copy(), [], [])
    at = int(words[5])
    picks[at] = token
    words[5] = at + 1
    if token in stop:
        words[7] = 1
    else:
        words[4] = token
        words[1] += 1
        words[8 + history % WINDOW] = token
        words[6] = history + 1
        words[2] += 1
    return out.tobytes(), words.tobytes(), picks.tobytes()


# ---- T.465.4: the DP4A forms of a token's layer: the norm with the quantizer (NORM_QUANTIZE), or apart (RMSNORM, then
# QUANTIZE), and the matrix by the quantized vector (fusedDp4aMatVec: no norm on its read, T.150's writes)

def norm_params(norm):
    """Norm (device.ts's normShape): the size, the weight's start, eps"""
    words = np.frombuffer(bytes(norm), dtype=np.uint32)
    return int(words[0]), int(words[1]), float(np.frombuffer(bytes(norm), dtype=np.float32)[2])


def normed(x, weight, norm, tokens):
    """weight × (s × x) of each token's row, s the inverse root of its mean square and eps, in float32 as RMSNORM"""
    size, at, eps = norm_params(norm)
    rows = np.frombuffer(bytes(x), dtype=np.float32)[: tokens * size].reshape(tokens, size)
    w = np.frombuffer(bytes(weight), dtype=np.float32)[at: at + size]
    s = (1.0 / np.sqrt(np.mean(rows.astype(np.float64) ** 2, axis=1) + eps)).astype(np.float32)
    return (w * (s[:, None] * rows)).astype(np.float32)


def rmsnorm(x, weight, norm, step):
    """RMSNORM: x, the weight, the Norm, the step (tokens) -> dst, as bytes"""
    tokens = int(np.frombuffer(bytes(step), dtype=np.uint32)[0])
    return normed(x, weight, norm, tokens).tobytes()


def rms_rows(x, weight, norm, step, across, out, add):
    """(T.465.6) shaders/steps.ts's rmsNorm, each row of the dispatch (across a token: its x, from the Norm's fourth word
    first rows on) normed: weight × (s × x), s the inverse root of the row's mean square and eps, into out (None: x
    in place, HEAD_NORM's), added to what is there where add (RMSNORM_ADD's) -> what it wrote, as bytes"""
    size, at, eps = norm_params(norm)
    first = int(np.frombuffer(bytes(norm), dtype=np.uint32)[3])
    tokens = int(np.frombuffer(bytes(step), dtype=np.uint32)[0])
    source = np.frombuffer(bytes(x), dtype=np.float32)
    target = (source if out is None else np.frombuffer(bytes(out), dtype=np.float32)).copy()
    w = np.frombuffer(bytes(weight), dtype=np.float32)[at: at + size]
    for token in range(tokens):
        for row in range(across):
            start = (token * across + row + first) * size
            values = source[start: start + size]
            s = np.float32(1.0 / math.sqrt(float(np.mean(values.astype(np.float64) ** 2)) + eps))
            written = (w * (s * values)).astype(np.float32)
            target[start: start + size] = target[start: start + size] + written if add else written
    return target.tobytes()


def layer_norm(x, weight, bias, norm, step, across, out):
    """(T.465.6.2) LAYER_NORM: each row of the dispatch (across a token, its x) as the CPU's layernorm: weight × (s × (x
    - mean)) + bias, s the inverse root of the variance and eps, in float32 -> out, as bytes"""
    size, at, eps = norm_params(norm)
    tokens = int(np.frombuffer(bytes(step), dtype=np.uint32)[0])
    source = np.frombuffer(bytes(x), dtype=np.float32)
    target = np.frombuffer(bytes(out), dtype=np.float32).copy()
    w = np.frombuffer(bytes(weight), dtype=np.float32)[at: at + size]
    b = np.frombuffer(bytes(bias), dtype=np.float32)[at: at + size]
    for row in range(tokens * across):
        values = source[row * size: (row + 1) * size]
        mean = np.float32(np.mean(values.astype(np.float64)))
        s = np.float32(1.0 / math.sqrt(float(np.mean((values - mean).astype(np.float64) ** 2)) + eps))
        target[row * size: (row + 1) * size] = (w * (s * (values - mean)) + b).astype(np.float32)
    return target.tobytes()


def gelu(gate, size, step):
    """(T.465.6.2) GELU: each token's n values of gate, the tanh approximation (its argument clamped) -> gate, as bytes"""
    n = int(np.frombuffer(bytes(size), dtype=np.uint32)[0])
    tokens = int(np.frombuffer(bytes(step), dtype=np.uint32)[0])
    out = np.frombuffer(bytes(gate), dtype=np.float32).copy()
    v = out[: tokens * n].astype(np.float64)
    out[: tokens * n] = 0.5 * v * (1.0 + np.tanh(np.clip(0.7978845608028654 * (v + 0.044715 * v ** 3), -9.010913, 9.010913)))
    return out.tobytes()


def add(y, bias, shape, step):
    """ADD: each token's row of y (n of the Bias, the dispatch's y a token) plus the bias at its start, or (byPos) the
    bias's row of the token's position -> y, as bytes"""
    n, at, by_pos = (int(w) for w in np.frombuffer(bytes(shape), dtype=np.uint32)[:3])
    tokens, pos = (int(w) for w in np.frombuffer(bytes(step), dtype=np.uint32)[:2])
    out = np.frombuffer(bytes(y), dtype=np.float32).copy()
    b = np.frombuffer(bytes(bias), dtype=np.float32)
    for token in range(tokens):
        start = at + (pos + token) * by_pos
        out[token * n: (token + 1) * n] += b[start: start + n]
    return out.tobytes()


def token_rope(rows, params, keys, values, angles, step):
    """TOKEN_ROPE: the rows q, k and v as a matrix wrote them (rows of its Params), each pair as fused_rope's write ->
    (q, the keys, the values), as bytes"""
    words = np.frombuffer(bytes(params), dtype=np.uint32)
    count, q_rows, kv_rows, size, turned = int(words[0]), *(int(w) for w in words[6:10])
    made = np.frombuffer(bytes(rows), dtype=np.float32)[:count].astype(np.float64)
    return rope_into(made, q_rows, kv_rows, size, turned, keys, values, angles, step)


def norm_quantize(x, weight, norm, step):
    """NORM_QUANTIZE: RMSNORM's values of each token quantized as QUANTIZE does -> (codes, scales), as bytes"""
    tokens = int(np.frombuffer(bytes(step), dtype=np.uint32)[0])
    values = normed(x, weight, norm, tokens)
    codes, scales = quantized(values.reshape(1, -1))
    return codes.tobytes(), scales.tobytes()


def dp4a(output, weights, scales, xq, xs, params, dst):
    """fusedDp4aMatVec with an int8 matrix: its rows by the quantized vector (codes times their group's scale), written
    as fused() writes -> dst, as bytes"""
    rows, words, per_row, second, eps, norm_at = row_params(params)
    n = per_row * GROUP
    v = np.frombuffer(bytes(xq), dtype=np.int8)[:n].astype(np.float64) * np.repeat(np.frombuffer(bytes(xs), dtype=np.float32)[:per_row].astype(np.float64), GROUP)
    return product(False, output, weights, scales, v, params, None, dst)


def dp4a_rope(weights, scales, xq, xs, params, keys, values, angles, step):
    """fusedDp4aMatVec's "rope" write -> (q, the keys, the values), as bytes"""
    words = np.frombuffer(bytes(params), dtype=np.uint32)
    rows, q_rows, kv_rows, size, turned = int(words[0]), *(int(w) for w in words[6:10])
    made = np.frombuffer(dp4a("write", weights, scales, xq, xs, params, np.zeros(rows, dtype=np.float32).tobytes()), dtype=np.float32).astype(np.float64)
    return rope_into(made, q_rows, kv_rows, size, turned, keys, values, angles, step)
