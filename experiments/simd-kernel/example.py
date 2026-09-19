# How Python drives the kernels inside Pyodide (the design measured at 181 / 282 / 348 tokens/s).
# NumPy owns every buffer; the kernels get raw addresses (array.ctypes.data) and work in place, so nothing is copied.
# This is the forward pass of the experiment, for float32 weights; see README.md for the int8 variant.
import ctypes

import numpy as np

lib = ctypes.CDLL("/home/pyodide/simdkernel.so")  # a path in Pyodide's file system
i32, p = ctypes.c_int32, ctypes.c_void_p
for name, args in dict(matmul_f32=[p, p, p, i32, i32, i32], rmsnorm=[p, p, p, i32], rope=[p, p, p, i32, i32],
                       attention=[p, p, p, p, p, i32, i32, i32], swiglu=[p, p, p, i32], add_inplace=[p, p, i32],
                       argmax=[p, i32]).items():
    getattr(lib, name).argtypes = args
    getattr(lib, name).restype = i32 if name == "argmax" else None


def make_forward(m):
    """m: a llama2_numpy.Llama with float32 weights and n_kv_heads == n_heads. Returns forward(token, pos) -> next token."""
    dim, hidden, layers, heads, hs, seq = m.dim, m.hidden_dim, m.n_layers, m.n_heads, m.head_size, m.seq_len
    f32 = np.float32
    x, xb, xb2, q = (np.zeros(dim, f32) for _ in range(4))
    h13, hb, att, logits = np.zeros(2 * hidden, f32), np.zeros(hidden, f32), np.zeros(seq, f32), np.zeros(m.vocab_size, f32)
    kc, vc = np.zeros((layers, seq, dim), f32), np.zeros((layers, seq, dim), f32)  # [seq][dim]: k and v are written in place
    w13 = np.ascontiguousarray(np.concatenate([m.w1, m.w3], axis=1))  # one matmul for w1 and w3
    keep = [x, xb, xb2, q, h13, hb, att, logits, kc, vc, w13]  # the addresses below stay valid while these live
    ptr = lambda a: a.ctypes.data  # ~4 us per call in Pyodide, so take every address once, outside the loop
    x_p, xb_p, xb2_p, q_p, h13_p, hb_p, att_p, logits_p, kc_p, vc_p = map(ptr, keep[:10])
    W = {("q", l): ptr(m.wq[l]) for l in range(layers)} | {("k", l): ptr(m.wk[l]) for l in range(layers)} \
        | {("v", l): ptr(m.wv[l]) for l in range(layers)} | {("o", l): ptr(m.wo[l]) for l in range(layers)} \
        | {("13", l): ptr(w13[l]) for l in range(layers)} | {("2", l): ptr(m.w2[l]) for l in range(layers)}
    att_w, ffn_w, final_w, cls = ptr(m.rms_att_weight), ptr(m.rms_ffn_weight), ptr(m.rms_final_weight), ptr(m.wcls)
    cos_p, sin_p = ptr(m.freq_cis_real), ptr(m.freq_cis_imag)
    mm = lib.matmul_f32

    def forward(token, pos):
        x[:] = m.token_embedding_table[token]
        cos, sin = cos_p + pos * (hs // 2) * 4, sin_p + pos * (hs // 2) * 4
        for l in range(layers):
            k_layer, v_layer = kc_p + l * seq * dim * 4, vc_p + l * seq * dim * 4
            k_p, v_p = k_layer + pos * dim * 4, v_layer + pos * dim * 4
            lib.rmsnorm(xb_p, x_p, att_w + l * dim * 4, dim)
            mm(q_p, xb_p, W["q", l], dim, 0, dim)
            mm(k_p, xb_p, W["k", l], dim, 0, dim)
            mm(v_p, xb_p, W["v", l], dim, 0, dim)
            lib.rope(q_p, cos, sin, heads, hs)
            lib.rope(k_p, cos, sin, heads, hs)
            lib.attention(xb_p, q_p, k_layer, v_layer, att_p, pos, heads, hs)
            mm(xb2_p, xb_p, W["o", l], dim, 0, dim)
            lib.add_inplace(x_p, xb2_p, dim)
            lib.rmsnorm(xb_p, x_p, ffn_w + l * dim * 4, dim)
            mm(h13_p, xb_p, W["13", l], dim, 0, 2 * hidden)
            lib.swiglu(hb_p, h13_p, h13_p + hidden * 4, hidden)
            mm(xb2_p, hb_p, W["2", l], hidden, 0, dim)
            lib.add_inplace(x_p, xb2_p, dim)
        lib.rmsnorm(xb_p, x_p, final_w, dim)
        mm(logits_p, xb_p, cls, dim, 0, m.vocab_size)
        return lib.argmax(logits_p, m.vocab_size)

    forward.keep = keep
    return forward
