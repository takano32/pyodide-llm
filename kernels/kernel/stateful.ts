// kernel/stateful.ts (T356): the layers that carry a state from token to token and not a cache of keys and values:
// Qwen3.5's (gate, convolve, delta_rule) and LFM2's (short_conv).

import { fexp, vexp } from "./math";

// ------------------------------------------------------------------------ Qwen3.5's hybrid attention (T229)
// The three things of llama2_numpy.py's linear_attention() and of its gated full attention that no kernel above
// does. Everything else of those layers is a matmul, rmsnorm (the l2 norm of the heads of q and k too: forward.js),
// swiglu (the norm's gate) or add_inplace.

// out[j] = x[j] * sigmoid(g[j]): what a full-attention layer read, through its gate
export function gate(out: usize, x: usize, g: usize, n: i32): void {
  const one = f32x4.splat(1.0);
  let j = 0;
  for (; j + 4 <= n; j += 4) {
    const o = <usize>j << 2;
    v128.store(out + o, f32x4.div(v128.load(x + o), f32x4.add(one, vexp(f32x4.neg(v128.load(g + o))))));
  }
  for (; j < n; j++) {
    const o = <usize>j << 2;
    store<f32>(out + o, load<f32>(x + o) / (<f32>1.0 + fexp(-load<f32>(g + o))));
  }
}

// out[c] = silu(taps[0][c] * rows[0][c] + ... + taps[count - 1][c] * rows[count - 1][c]): the causal convolution of a
// Gated DeltaNet layer, each of the n channels with its own taps over the last count tokens (rows: the oldest first,
// this token's last; both (count, n) row-major), added in that order as NumPy adds them, and SiLU after it
export function convolve(out: usize, taps: usize, rows: usize, n: i32, count: i32): void {
  const one = f32x4.splat(1.0);
  const stride = <usize>n << 2;
  let c = 0;
  for (; c + 4 <= n; c += 4) {
    const o = <usize>c << 2;
    let acc = f32x4.mul(v128.load(taps + o), v128.load(rows + o));
    for (let j = 1; j < count; j++) {
      const at = <usize>j * stride + o;
      acc = f32x4.add(acc, f32x4.mul(v128.load(taps + at), v128.load(rows + at)));
    }
    v128.store(out + o, f32x4.div(acc, f32x4.add(one, vexp(f32x4.neg(acc)))));
  }
  for (; c < n; c++) {
    const o = <usize>c << 2;
    let acc = load<f32>(taps + o) * load<f32>(rows + o);
    for (let j = 1; j < count; j++) {
      const at = <usize>j * stride + o;
      acc += load<f32>(taps + at) * load<f32>(rows + at);
    }
    store<f32>(out + o, acc / (<f32>1.0 + fexp(-acc)));
  }
}

// T260, an LFM2's convolution layer: what llama2_numpy.py's short_convolution() does between its two matmuls. b: what
// the matrix in gave, 3 n values (the gate B, the gate C, and z, one after another). rows: the layer's state, count rows
// of n values, the oldest token's first; the last is this token's and is written here, B * z. Then
// out[c] = C[c] * (taps[0][c] * rows[0][c] + ... + taps[count - 1][c] * rows[count - 1][c]), added in that order as
// NumPy adds them: the causal convolution of each channel with its own taps, and no activation (count is 2 or more)
export function short_conv(out: usize, taps: usize, rows: usize, b: usize, n: i32, count: i32): void {
  const stride = <usize>n << 2;
  const newest = rows + <usize>(count - 1) * stride, gate = b + stride, z = gate + stride;
  let c = 0;
  for (; c + 4 <= n; c += 4) {
    const o = <usize>c << 2;
    v128.store(newest + o, f32x4.mul(v128.load(b + o), v128.load(z + o)));
    let acc = f32x4.mul(v128.load(taps + o), v128.load(rows + o));
    for (let j = 1; j < count; j++) {
      const at = <usize>j * stride + o;
      acc = f32x4.add(acc, f32x4.mul(v128.load(taps + at), v128.load(rows + at)));
    }
    v128.store(out + o, f32x4.mul(v128.load(gate + o), acc));
  }
  for (; c < n; c++) {
    const o = <usize>c << 2;
    store<f32>(newest + o, load<f32>(b + o) * load<f32>(z + o));
    let acc = load<f32>(taps + o) * load<f32>(rows + o);
    for (let j = 1; j < count; j++) {
      const at = <usize>j * stride + o;
      acc += load<f32>(taps + at) * load<f32>(rows + at);
    }
    store<f32>(out + o, load<f32>(gate + o) * acc);
  }
}

// One token of the gated delta rule (llama2_numpy.delta_rule) for the value heads h0..h1 of vh: the software threads
// take heads as they take rows of a matmul, and every head is computed alone, the same whatever the range.
//   S' = S * decay, delta = (v - k S') * beta, next = S' + k (outer) delta, out = q next
// state, next: a matrix (kd, vd) a value head, row-major, the heads one after another. The new state is written at
// next and state is left as it was: a phase that is run again (T120: a software thread stopped in it) computes the
// same from the same. c: the token's q (kh heads of kd), k (the same) and v (vh heads of vd) one after another, q and
// k normalized already; value head h reads key head h / (vh / kh). work: beta of every value head, then decay of
// every value head, then room for vd floats a value head (its delta). out: vd floats a value head.
export function delta_rule(out: usize, state: usize, next: usize, c: usize, work: usize, kh: i32, kd: i32, vd: i32, vh: i32, h0: i32, h1: i32): void {
  const each = vh / kh, vd4 = vd & ~3;
  const row = <usize>vd << 2, head = <usize>kd * row;
  for (let h = h0; h < h1; h++) {
    const key = h / each;
    const q = c + ((<usize>key * <usize>kd) << 2), k = c + ((<usize>(kh + key) * <usize>kd) << 2);
    const v = c + ((<usize>(2 * kh) * <usize>kd) << 2) + <usize>h * row;
    const beta = load<f32>(work + (<usize>h << 2)), decay = load<f32>(work + (<usize>(vh + h) << 2));
    const delta = work + (<usize>(2 * vh) << 2) + <usize>h * row;
    const from = state + <usize>h * head, to = next + <usize>h * head, o = out + <usize>h * row;
    const decays = f32x4.splat(decay), betas = f32x4.splat(beta);
    // delta = (v - k S') * beta: k S' is the sum over the rows i of k[i] * S'[i]
    for (let j = 0; j < vd; j++) store<f32>(delta + (<usize>j << 2), 0);
    for (let i = 0; i < kd; i++) {
      const ki = load<f32>(k + (<usize>i << 2)), kis = f32x4.splat(ki), at = from + <usize>i * row;
      let j = 0;
      for (; j < vd4; j += 4) {
        const p = <usize>j << 2;
        v128.store(delta + p, f32x4.add(v128.load(delta + p), f32x4.mul(kis, f32x4.mul(v128.load(at + p), decays))));
      }
      for (; j < vd; j++) {
        const p = <usize>j << 2;
        store<f32>(delta + p, load<f32>(delta + p) + ki * (load<f32>(at + p) * decay));
      }
    }
    let j = 0;
    for (; j < vd4; j += 4) {
      const p = <usize>j << 2;
      v128.store(delta + p, f32x4.mul(f32x4.sub(v128.load(v + p), v128.load(delta + p)), betas));
      v128.store(o + p, f32x4.splat(0));
    }
    for (; j < vd; j++) {
      const p = <usize>j << 2;
      store<f32>(delta + p, (load<f32>(v + p) - load<f32>(delta + p)) * beta);
      store<f32>(o + p, 0);
    }
    // next = S' + k (outer) delta, and what q reads of it
    for (let i = 0; i < kd; i++) {
      const ki = load<f32>(k + (<usize>i << 2)), qi = load<f32>(q + (<usize>i << 2));
      const kis = f32x4.splat(ki), qis = f32x4.splat(qi), at = from + <usize>i * row, into = to + <usize>i * row;
      j = 0;
      for (; j < vd4; j += 4) {
        const p = <usize>j << 2;
        const s = f32x4.add(f32x4.mul(v128.load(at + p), decays), f32x4.mul(kis, v128.load(delta + p)));
        v128.store(into + p, s);
        v128.store(o + p, f32x4.add(v128.load(o + p), f32x4.mul(qis, s)));
      }
      for (; j < vd; j++) {
        const p = <usize>j << 2;
        const s = load<f32>(at + p) * decay + ki * load<f32>(delta + p);
        store<f32>(into + p, s);
        store<f32>(o + p, load<f32>(o + p) + qi * s);
      }
    }
  }
}
