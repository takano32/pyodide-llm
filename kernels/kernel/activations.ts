// kernel/activations.ts (T356): what is done to one vector of activations between the matrix products: the norms
// (rmsnorm, layernorm), the feed-forward's activations (gelu, swiglu), the residual's add, and the rotated basis of
// the matrices that are stored rotated (rotate, unrotate).

import { hsum, fexp, vexp } from "./math";

// eps: the model's (config.json's rms_norm_eps): Qwen3's 1e-6 against 1e-5 moved its perplexity by 0.12% (T124)
export function rmsnorm(out: usize, x: usize, w: usize, n: i32, eps: f32): void {
  let acc = f32x4.splat(0);
  let j = 0;
  const n4 = n & ~3;
  for (; j < n4; j += 4) { const v = v128.load(x + (<usize>j << 2)); acc = f32x4.add(acc, f32x4.mul(v, v)); }
  let ss: f32 = hsum(acc);
  for (; j < n; j++) { const v = load<f32>(x + (<usize>j << 2)); ss += v * v; }
  const s: f32 = <f32>1.0 / sqrt<f32>(ss / <f32>n + eps);
  for (j = 0; j < n; j++) {
    const o = <usize>j << 2;
    store<f32>(out + o, load<f32>(w + o) * (s * load<f32>(x + o)));
  }
}

export function layernorm(out: usize, x: usize, w: usize, b: usize, n: i32): void {
  // GPT-2 normalizes by the mean and the variance, and adds a bias after the scale
  let sum = f32x4.splat(0);
  let j = 0;
  const n4 = n & ~3;
  for (; j < n4; j += 4) { sum = f32x4.add(sum, v128.load(x + (<usize>j << 2))); }
  let total: f32 = hsum(sum);
  for (; j < n; j++) { total += load<f32>(x + (<usize>j << 2)); }
  const mean: f32 = total / <f32>n;
  let acc = f32x4.splat(0);
  const meanv = f32x4.splat(mean);
  for (j = 0; j < n4; j += 4) {
    const d = f32x4.sub(v128.load(x + (<usize>j << 2)), meanv);
    acc = f32x4.add(acc, f32x4.mul(d, d));
  }
  let ss: f32 = hsum(acc);
  for (; j < n; j++) { const d = load<f32>(x + (<usize>j << 2)) - mean; ss += d * d; }
  const s: f32 = <f32>1.0 / sqrt<f32>(ss / <f32>n + <f32>1e-5);
  for (j = 0; j < n; j++) {
    const o = <usize>j << 2;
    store<f32>(out + o, load<f32>(w + o) * (s * (load<f32>(x + o) - mean)) + load<f32>(b + o));
  }
}

// T162: gelu and swiglu four at a time by vexp. Their results are fexp's bit for bit: where vexp and fexp differ (below
// -87), 1 + exp(x) is 1 either way.
export function gelu(out: usize, x: usize, b: usize, n: i32): void {
  // GPT-2's gelu_new, with the bias of the projection added first. 0.5 * (1 + tanh(z)) is 1 / (1 + exp(-2z)).
  const one = f32x4.splat(1.0), c = f32x4.splat(0.7978845608028654), cube = f32x4.splat(0.044715), two = f32x4.splat(-2.0);
  let j = 0;
  for (; j + 4 <= n; j += 4) {
    const o = <usize>j << 2;
    const v = f32x4.add(v128.load(x + o), v128.load(b + o));
    const inner = f32x4.mul(c, f32x4.add(v, f32x4.mul(f32x4.mul(f32x4.mul(cube, v), v), v)));
    v128.store(out + o, f32x4.div(v, f32x4.add(one, vexp(f32x4.mul(two, inner)))));
  }
  for (; j < n; j++) {
    const o = <usize>j << 2;
    const v = load<f32>(x + o) + load<f32>(b + o);
    const inner = <f32>0.7978845608028654 * (v + <f32>0.044715 * v * v * v);
    store<f32>(out + o, v / (<f32>1.0 + fexp(<f32>-2.0 * inner)));
  }
}

export function swiglu(out: usize, h1: usize, h3: usize, n: i32): void {
  const one = f32x4.splat(1.0);
  let j = 0;
  for (; j + 4 <= n; j += 4) {
    const o = <usize>j << 2;
    const v = v128.load(h1 + o);
    v128.store(out + o, f32x4.mul(f32x4.div(v, f32x4.add(one, vexp(f32x4.neg(v)))), v128.load(h3 + o)));
  }
  for (; j < n; j++) {
    const o = <usize>j << 2;
    const v = load<f32>(h1 + o);
    store<f32>(out + o, v / (<f32>1.0 + fexp(-v)) * load<f32>(h3 + o));
  }
}

export function add_inplace(x: usize, y: usize, n: i32): void {
  for (let j = 0; j < n; j++) { const o = <usize>j << 2; store<f32>(x + o, load<f32>(x + o) + load<f32>(y + o)); }
}

// T237: the rotated basis (llama2_numpy.py has the definition, above hadamard()). The normalized Walsh-Hadamard
// transform of a block of values in place, after its values were multiplied by 1 / sqrt(block): sums and differences
// of values 1, 2, 4, ... apart, the order of llama2_numpy.hadamard() and of the fork's ggml_compute_forward_fwht
// (ggml/src/ggml-cpu/ops.cpp at 88c4bc60; the order, no line of it). Every number is one float32 sum or difference,
// so the result is NumPy's to the bit. block: a power of two. Values 1 and 2 apart are one pass over fours of values
// (the two sums and differences of a four), the rest four values a step.
// @ts-ignore: decorator
@inline function butterflies(p: usize, block: i32): void {
  if (block == 2) {
    const a = load<f32>(p), b = load<f32>(p + 4);
    store<f32>(p, a + b);
    store<f32>(p + 4, a - b);
    return;
  }
  for (let i = 0; i + 4 <= block; i += 4) {
    const o = p + (<usize>i << 2);
    const a = load<f32>(o), b = load<f32>(o + 4), c = load<f32>(o + 8), d = load<f32>(o + 12);
    const ab = a + b, a_b = a - b, cd = c + d, c_d = c - d;
    store<f32>(o, ab + cd);
    store<f32>(o + 4, a_b + c_d);
    store<f32>(o + 8, ab - cd);
    store<f32>(o + 12, a_b - c_d);
  }
  for (let half = 4; half < block; half <<= 1) {
    for (let i = 0; i < block; i += 2 * half) {
      for (let j = 0; j < half; j += 4) {
        const u = p + (<usize>(i + j) << 2), v = u + (<usize>half << 2);
        const a = v128.load(u), b = v128.load(v);
        v128.store(u, f32x4.add(a, b));
        v128.store(v, f32x4.sub(a, b));
      }
    }
  }
}
// out = R x: n values of x (whole blocks) times their signs, then the transform of every block. signs: +1 or -1 for
// every value, times 1 / sqrt(block) (x times that is (x times the sign) times 1 / sqrt(block) to the bit: the sign
// changes no digit). out may be x.
export function rotate(out: usize, x: usize, signs: usize, n: i32, block: i32): void {
  let j = 0;
  for (; j + 4 <= n; j += 4) {
    const o = <usize>j << 2;
    v128.store(out + o, f32x4.mul(v128.load(x + o), v128.load(signs + o)));
  }
  for (; j < n; j++) { const o = <usize>j << 2; store<f32>(out + o, load<f32>(x + o) * load<f32>(signs + o)); }
  if (block > 1) for (let b = 0; b < n; b += block) butterflies(out + (<usize>b << 2), block);
}
// out = R^-1 z: the transform of every block of n values of z, then their signs (a row of the embedding as the model
// reads it). The same signs as rotate(): their magnitude is the transform's 1 / sqrt(block), multiplied in first as
// NumPy does, and their sign bit flips the result's.
export function unrotate(out: usize, z: usize, signs: usize, n: i32, block: i32): void {
  let j = 0;
  for (; j + 4 <= n; j += 4) {
    const o = <usize>j << 2;
    v128.store(out + o, f32x4.mul(v128.load(z + o), f32x4.abs(v128.load(signs + o))));
  }
  for (; j < n; j++) { const o = <usize>j << 2; store<f32>(out + o, load<f32>(z + o) * abs<f32>(load<f32>(signs + o))); }
  if (block > 1) for (let b = 0; b < n; b += block) butterflies(out + (<usize>b << 2), block);
  for (j = 0; j < n; j++) {
    const o = <usize>j << 2;
    store<u32>(out + o, load<u32>(out + o) ^ (load<u32>(signs + o) & 0x80000000));
  }
}
