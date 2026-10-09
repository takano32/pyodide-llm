# What a layer computes that is no matrix product: the norms and activations, RoPE's angles, and the forms of the
# layers that are no plain attention (Qwen3.5's linear attention, LFM2's convolutions, a rotated basis).
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is
# derived from two works under the MIT License, whose notice follows: tairov/llama2.py
# (https://github.com/tairov/llama2.py; its LICENSE names no copyright holder) and karpathy/llama2.c
# (https://github.com/karpathy/llama2.c), Copyright (c) 2023 Andrej.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math

import numpy as np


def partial_rope(heads, cos, sin, rotary):
    """GPT-NeoX rotates the first rotary values of every head and leaves the rest alone."""
    turned = rope(heads[:, :rotary], cos[:rotary // 2], sin[:rotary // 2])
    return np.concatenate([turned, heads[:, rotary:]], axis=1) if rotary < heads.shape[1] else turned


# the epsilon of RMSNorm where config.json does not say another (T124: Qwen3's 1e-6 moved perplexity by 0.12%)
RMS_EPS = 1e-5


def rmsnorm(x, weight, eps=RMS_EPS):
    return weight * (x / np.sqrt(x.dot(x) / x.size + np.float32(eps)))


def head_norm(x, weight, eps=RMS_EPS):
    """rmsnorm() of every head of x (heads one after another), all with the same weight of one head's size."""
    heads = x.reshape(-1, weight.size)
    return (weight * heads / np.sqrt((heads * heads).mean(axis=1, keepdims=True) + np.float32(eps))).reshape(-1)


def layernorm(x, weight, bias):
    """GPT-2 normalizes by the mean and the variance, and adds a bias."""
    centred = x - x.mean()
    return weight * (centred / np.sqrt(centred.dot(centred) / x.size + 1e-5)) + bias


def gelu(x):
    """GPT-2's gelu_new: the tanh approximation, written with exp so that the kernel can do the same."""
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return x * (1.0 / (1.0 + np.exp(-2.0 * inner)))


# Qwen3.5 and Qwen3.8 (config.json's model_type qwen3_5; arch="qwen35" here) mix two kinds of layers: every "every"-th
# layer (layer l where (l + 1) % every == 0; every is 4) attends over all positions as a Qwen3 does, and the others
# are Gated DeltaNet layers ("linear attention"), which keep a state of a fixed size instead of keys and values.
# The computation is taken from transformers' modeling code (Apache-2.0; the formulas, no line of it):
# https://github.com/huggingface/transformers/blob/7fb5bcd1d4b8a5c225a2c33429b2e9e023dd61ae/src/transformers/models/qwen3_5/modeling_qwen3_5.py
# (Qwen3_5GatedDeltaNet and torch_recurrent_gated_delta_rule, lines 437 to 662; Qwen3_5Attention, 748 to 820;
# Qwen3_5RMSNorm, 840 to 854; Qwen3_5RMSNormGated, 217 to 233), with the numbers of
# https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json
#
# Both kinds: x += mixer(norm(x)), then x += w2(silu(w1(norm(x))) * w3(norm(x))), a Llama's. RMSNorm multiplies by
# 1 + weight there; the converter adds the 1, so the file holds what rmsnorm() multiplies by.
#
# A Gated DeltaNet layer, one token (xb = norm(x); K key heads of key_dim, V value heads of value_dim, V a multiple of
# K; the state S of every value head is a matrix (key_dim, value_dim), zero before the first token):
#   mixed = wqkv xb                      2 K key_dim + V value_dim values: q, k and v, one after another
#   z = wz xb, b = wb xb, a = wa xb      V value_dim, V and V values
#   c = silu(sum over j of conv[j] * mixed of (conv - 1 - j) tokens ago)
#                                        a causal convolution of each channel with its own conv taps over this token
#                                        and the conv - 1 before it (zeros before the first token), then SiLU
#   q, k, v = c cut at 2 K key_dim       q and k in K heads, v in V heads
#   q = q / sqrt(sum(q * q) + 1e-6) / sqrt(key_dim), k = k / sqrt(sum(k * k) + 1e-6), head by head
#   value head h uses key head h // (V / K) (repeat_interleave)
#   beta = sigmoid(b), g = decay * softplus(a + dt_bias)      decay is -exp(A_log), which the converter computes
#   for every value head h:
#     S = S * exp(g[h])
#     delta = (v[h] - k[h] S) * beta[h]                       k[h] S: the sum over i of k[h][i] * S[i, :]
#     S = S + k[h] (outer) delta
#     o[h] = q[h] S
#   o[h] = delta_norm * o[h] / sqrt(mean(o[h] * o[h]) + eps) * silu(z[h])      delta_norm: value_dim weights, as stored
#   the layer's output is wout o
# transformers runs a prompt through a chunked form of the same rule (torch_chunk_gated_delta_rule); token by token it
# is this (tests/reference_qwen35.py compares the two on the real model).
#
# A full-attention layer is a Qwen3's (heads of head_dim, each head of q and k normalized, grouped keys and values)
# with two differences. RoPE turns only the first rotary values of every head (64 of 256, theta 1e7: text has the
# same position on all three axes of the model's 3D RoPE, which is then the ordinary one). And q's matrix has twice
# the rows: each head's q, then as many values of a gate; the attention's output is multiplied by sigmoid(gate)
# before wo. The converter cuts the matrix into wq and wg.
LINEAR = ("every", "key_heads", "value_heads", "key_dim", "value_dim", "conv")


def linear_form(linear):
    """The numbers of a hybrid model's linear-attention layers, FORM's "linear", as a dict of ints (LINEAR's keys:
    every "every"-th layer is a full-attention one, the heads and their sizes, the taps of the convolution), from a
    dict of Python or of JavaScript. None for a model without such layers."""
    if linear is None:
        return None
    linear = linear.to_py() if hasattr(linear, "to_py") else linear
    numbers = {key: int(linear[key]) for key in LINEAR}
    if min(numbers.values()) < 1 or numbers["every"] < 2 or numbers["value_heads"] % numbers["key_heads"]:
        raise ValueError(f"These are not the numbers of linear-attention layers: {numbers}.")
    return numbers


def linear_widths(linear):
    """(the values the convolution runs over: q, k and v; those of q or of k; those of v) of a linear-attention layer."""
    keys, values = linear["key_heads"] * linear["key_dim"], linear["value_heads"] * linear["value_dim"]
    return 2 * keys + values, keys, values


def layer_slots(n_layers, linear, convolution=None):
    """For every layer: (whether it keeps a state in place of keys and values: a Qwen3.5's linear-attention layer or an
    LFM2's convolution layer, its place among the layers of its kind), which is where its tensors are in the file's
    stacks: a model whose layers all attend has (False, l) for layer l."""
    if convolution is not None:
        kinds = [kind == "c" for kind in convolution["layers"]]
    else:
        kinds = [linear is not None and (l + 1) % linear["every"] != 0 for l in range(n_layers)]
    slots, counts = [], [0, 0]
    for kind in kinds:
        slots.append((kind, counts[kind]))
        counts[kind] += 1
    return slots


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    """log(1 + exp(x)), and x itself past 20: torch's softplus as Qwen3.5 calls it."""
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0)))).astype(np.float32)


def l2_heads(x, heads):
    """Each of heads rows of x over its length: x / sqrt(sum(x * x) + 1e-6), the l2norm of the gated delta rule."""
    rows = x.reshape(heads, -1)
    return rows / np.sqrt((rows * rows).sum(axis=1, keepdims=True) + np.float32(1e-6))


def delta_rule(state, q, k, v, beta, decay):
    """One token of the gated delta rule: state (value heads, key_dim, value_dim) is updated in place, and what q reads
    of it is returned (value heads, value_dim). q and k: (value heads, key_dim), v: (value heads, value_dim), beta and
    decay (exp(g)): one of each a head."""
    state *= decay[:, None, None]
    delta = (v - (k[:, None, :] @ state)[:, 0]) * beta[:, None]
    state += k[:, :, None] * delta[:, None, :]
    return (q[:, None, :] @ state)[:, 0]


# Liquid AI's LFM2 and LFM2.5 (config.json's model_type lfm2; arch="lfm2" here) mix two kinds of layers in an order
# the config.json lists (layer_types; LFM2.5-350M: conv, conv, full_attention, conv, conv, full_attention, ...):
# layers that attend over all positions, and convolution layers, which look at this token and the two before it.
# The computation is taken from transformers' modeling code (Apache-2.0; the formulas, no line of it):
# https://github.com/huggingface/transformers/blob/7cd73d9df0c14b151c684b708a9f27d8d0349dfe/src/transformers/models/lfm2/modeling_lfm2.py
# (Lfm2ShortConv with causal_conv1d_update and causal_conv1d_fn, lines 280 to 389; Lfm2Attention, 209 to 265; Lfm2MLP,
# 119 to 136; Lfm2DecoderLayer, 392 to 434; Lfm2Model.forward, 477 to 533), with the numbers of
# https://huggingface.co/LiquidAI/LFM2.5-350M/blob/9e6c6ccf47cd318696e137d381a7ded8fe4df09f/config.json
# and llama.cpp's graph of the same model, which computes the same (src/models/lfm2.cpp, build_shortconv_block, at
# f1cee9941b0e843ea260bf8dd9a090fbd9711b6a).
#
# One token (x: the token's row of the embedding as it is; RMSNorm multiplies by its weight as stored, eps 1e-5):
#   for every layer:
#     xb = operator_norm(x)
#     an attention layer:   x += wo attention(q, k, v)       a Qwen3's: q, k, v = wq xb, wk xb, wv xb, every head of q
#                                                            and of k normalized (weights of one head's size), RoPE
#                                                            over the whole head (theta 1e6), grouped keys and values,
#                                                            scores divided by sqrt(head), no bias anywhere
#     a convolution layer:  B, C, z = win xb cut in three    win has 3 dim rows: dim values each, one after another
#                           h = B * z                        value by value
#                           c = sum over j of conv[j] * h of (taps - 1 - j) tokens ago
#                                                            a causal convolution of each channel with its own taps
#                                                            over this token and the taps - 1 before it (zeros before
#                                                            the first token); taps is 3 (conv_L_cache). No activation
#                           x += wout (C * c)
#     x += w2(silu(w1 xn) * w3 xn), xn = ffn_norm(x)         a Llama's FFN
#   logits = embedding (the same table) times embedding_norm(x)       the last norm is called so; it is the last
# A convolution layer keeps no keys and values: its state is the h of the last taps - 1 tokens (2 dim numbers a layer).
# transformers pads a whole sequence with taps - 1 zeros in front (causal_conv1d_fn), and token by token keeps those
# values in its cache (causal_conv1d_update): the same numbers.
CONVOLUTION = ("layers", "taps")


def convolution_form(convolution, n_layers=None):
    """The convolution layers of an LFM2, FORM's "convolution", as {"layers": a letter for every layer, "c" for a
    convolution layer and "a" for one that attends, "taps": how many tokens the convolution reads}, from a dict of
    Python or of JavaScript. None for a model without such layers. n_layers: the header's, which the letters have to be
    as many as."""
    if convolution is None:
        return None
    convolution = convolution.to_py() if hasattr(convolution, "to_py") else convolution
    layers, taps = str(convolution["layers"]), int(convolution["taps"])
    if not layers or set(layers) - set("ac") or taps < 2 or (n_layers is not None and len(layers) != n_layers):
        raise ValueError(f"These are not the convolution layers of a model: {layers!r} with {taps} taps.")
    return {"layers": layers, "taps": taps}


# Ternary Bonsai 2 27B (prism-ml/Ternary-Bonsai-2-27B-gguf) stores its matrices in a rotated basis. With R = H S,
# where S flips the signs of the input's values (a vector of +1 and -1 for every width of an input) and H is the
# normalized Walsh-Hadamard transform of every block values in turn (Sylvester's order: entry (i, j) is
# (-1) ** popcount(i & j) / sqrt(block), so H is symmetric and its own inverse), the file holds W R^-1 = W S H in
# place of W, and the forward pass multiplies it by R x: W x as before, with a matrix that is ternary. The rows of the
# embedding are stored as R e, and the row looked up is turned back: e = S (H z). Every matrix the forward pass
# multiplies an activation by is stored so (q and its gate, k, v, o, a linear-attention layer's q-k-v, z and output,
# the three of the FFN, the classifier); the two small matrices of a linear-attention layer's gates, the norms and
# the convolution's taps are in the model's own basis. The matrices that read the same input share one R x.
# The rotation cannot be taken out of the file instead: (W S H) R is W again, which is not ternary (107 GB as float32).
# From Prism ML's fork of llama.cpp, the formulas and no line of it (MIT; 88c4bc60b9c9578f134385be9535e853f2db9b9f):
# https://github.com/PrismML-Eng/llama.cpp/blob/88c4bc60b9c9578f134385be9535e853f2db9b9f/src/llama-graph.cpp
# (build_lora_mm, lines 1546 to 1576: the signs, then the rotation, before the matrix, once for each input;
# build_embd_rows, 2398 to 2410: h = s * (H z)), src/llama-model.cpp (1196 to 1355: the metadata prism.hadamard.*, a
# sign vector for every width, a block that divides every width; 2054 to 2065: H's entries) and
# ggml/src/ggml-cpu/ops.cpp (12066 to 12140: times 1 / sqrt(block), then the butterflies of 1, 2, 4, ... apart).
# A layer whose value heads outnumber its key heads needs no more: the fork turns that layer's output into Hugging
# Face's order of heads before the signs (gdn_v_grouped), which is the order this engine computes in.
def hadamard(x, block):
    """The normalized Walsh-Hadamard transform of every block values of x in turn (its last axis, a multiple of block
    long; block a power of two): x / sqrt(block), then sums and differences of values 1, 2, 4, ... apart, the order
    the kernel and the fork compute in. Its own inverse."""
    y = (np.asarray(x, dtype=np.float32) * np.float32(1.0 / math.sqrt(block))).reshape(-1, block)
    half = 1
    while half < block:
        pairs = y.reshape(-1, block // (2 * half), 2, half)
        y = np.stack([pairs[:, :, 0] + pairs[:, :, 1], pairs[:, :, 0] - pairs[:, :, 1]], axis=2)
        half *= 2
    return y.reshape(np.shape(x))


def rotate(x, signs, block):
    """R x of the rotated basis: the signs, then the transform. What a matrix of the file is multiplied by, and how a
    row of a matrix (or of the embedding) is stored: W R^-1 has the rows rotate(w)."""
    return hadamard(x * signs, block)


def unrotate(z, signs, block):
    """R^-1 z: the transform, then the signs. A row of the embedding as the model reads it."""
    return hadamard(z, block) * signs


def sign_bits(signs):
    """A vector of +1 and -1 as the text the options carry: its bits (1 for -1, the first value the highest bit of the
    first byte) in hexadecimal."""
    return np.packbits(np.asarray(signs) < 0).tobytes().hex()


def rotated_form(rotated, widths=()):
    """The rotated basis of a model, FORM's "rotated", as {"block": the block of the transform, "signs": {width: a
    float32 vector of +1 and -1}}, from a dict of Python or of JavaScript whose signs are sign_bits() texts by the
    width (a text itself, as JSON's keys are). None for a model in its own basis. widths: the widths the model's
    matrices read, each of which has to be whole blocks and have its signs."""
    if rotated is None:
        return None
    rotated = rotated.to_py() if hasattr(rotated, "to_py") else rotated
    block = int(rotated["block"])
    if block < 1 or block & (block - 1):
        raise ValueError(f"The block of a rotated basis is a power of two, not {block}.")
    signs = {}
    for width, bits in dict(rotated["signs"]).items():
        width = int(width)
        raw = np.frombuffer(bytes.fromhex(str(bits)), dtype=np.uint8)
        if width < 1 or width % block or raw.size != (width + 7) // 8:
            raise ValueError(f"The signs of width {width} are not those of whole blocks of {block}.")
        signs[width] = (1.0 - 2.0 * np.unpackbits(raw)[:width]).astype(np.float32)
    missing = [width for width in widths if width not in signs]
    if missing:
        raise ValueError(f"The rotated basis has no signs for an input {missing[0]} wide.")
    return {"block": block, "signs": signs}


def rotated_widths(dim, q_dim, hidden_dim, linear=None):
    """The widths of what a model's matrices read: the residual stream, an attention's output (and a linear-attention
    layer's), the FFN's inside."""
    return sorted({dim, q_dim, hidden_dim} | ({linear_widths(linear)[2]} if linear else set()))


def rope(x, cos, sin):
    # Rotate each pair (x[2i], x[2i+1]) of every head by the angle for this position
    pairs = x.reshape(-1, cos.size, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    out = np.empty_like(pairs)
    out[..., 0] = x0 * cos - x1 * sin
    out[..., 1] = x0 * sin + x1 * cos
    return out.reshape(-1, 2 * cos.size)


def rope_frequencies(width, theta, scaling=None):
    """The angle per position of each pair of a head's first width values (float64), for the RoPE tables.

    scaling: config.json's rope_scaling, of three kinds. "linear" (T126: deepseek-coder): every pair turns factor times
    slower, as if the positions were divided by factor. "llama3": the pairs that turn slowly (a wavelength past
    original_max_position_embeddings / low_freq_factor) turn factor times slower, the fast ones (shorter than
    original / high_freq_factor) as before, and the ones between are a blend of the two (transformers'
    _compute_llama3_parameters). "yarn" (T235: Ternary-Bonsai): the pairs that turn 32 times or more within
    original_max_position_embeddings positions turn as before, the ones that turn once or less factor times slower,
    and the pairs between (counted by their number, each bound rounded outward) go from the one to the other in even
    steps: transformers' _compute_yarn_parameters and llama.cpp's rope_yarn(), which are the same table at every
    position, whatever the context. yarn also scales the turned values: rope_magnitude().
    """
    frequencies = 1.0 / theta ** (np.arange(0, width, 2, dtype=np.float64) / width)
    if not scaling:
        return frequencies
    kind = scaling.get("rope_type", scaling.get("type"))
    if kind == "linear":
        return frequencies / float(scaling["factor"])
    if kind == "yarn":
        original = float(scaling["original_max_position_embeddings"])
        # the pair whose angle makes this many turns in the original context (a real number: the pairs slow down evenly)
        pair = lambda turns: width * math.log(original / (turns * 2 * math.pi)) / (2 * math.log(theta))
        low, high = max(math.floor(pair(32)), 0), min(math.ceil(pair(1)), width - 1)
        slowed = np.clip((np.arange(width // 2) - low) / max(high - low, 0.001), 0, 1)
        return frequencies * (1 - slowed) + frequencies / float(scaling["factor"]) * slowed
    if kind != "llama3":
        raise ValueError(f"RoPE scaling of the {kind} kind is not supported.")
    factor, low, high = float(scaling["factor"]), float(scaling["low_freq_factor"]), float(scaling["high_freq_factor"])
    original = float(scaling["original_max_position_embeddings"])
    wavelength = 2 * math.pi / frequencies
    smooth = (original / wavelength - low) / (high - low)
    blended = (1 - smooth) * frequencies / factor + smooth * frequencies
    return np.where(wavelength < original / high, frequencies,
                    np.where(wavelength > original / low, frequencies / factor, blended))


def rope_magnitude(scaling=None):
    """What the cos and sin of the RoPE tables are multiplied by: 1, but 0.1 ln(factor) + 1 under yarn (T235), which so
    makes q and k that much longer and the attention's scores sharper by its square (yarn's temperature: transformers'
    attention_factor, which scales cos and sin, and llama.cpp's mscale in rope_yarn(), where it comes to the same).
    Left out, a yarn model's attention is 1.30 times too flat at factor 4 and nothing says so."""
    if not scaling or scaling.get("rope_type", scaling.get("type")) != "yarn":
        return 1.0
    return 0.1 * math.log(max(float(scaling["factor"]), 1.0)) + 1.0
