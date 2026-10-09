// kernel/attention.ts (T356): the rotation of q and k (rope) and the attention of one token over the layer's cache,
// of float32 keys and values or of float16 ones (kernel/halves.ts widens them).

import { hsum, fexp, vexp, largest } from "./math";
import { halves4, half } from "./halves";

export function rope(v: usize, fcr: usize, fci: usize, nh: i32, hs: i32, rot: i32): void {
  // rot: how many values of each head to turn (GPT-NeoX turns only the first ones, everyone else all of them)
  const turned = rot > 0 && rot < hs ? rot : hs;
  for (let h = 0; h < nh; h++) {
    const base = v + (<usize>(h * hs) << 2);
    for (let i = 0; i < turned; i += 2) {
      const c = load<f32>(fcr + (<usize>(i >> 1) << 2)), s = load<f32>(fci + (<usize>(i >> 1) << 2));
      const p0 = base + (<usize>i << 2);
      const v0 = load<f32>(p0), v1 = load<f32>(p0 + 4);
      store<f32>(p0, v0 * c - v1 * s);
      store<f32>(p0 + 4, v0 * s + v1 * c);
    }
  }
}

// kc / vc: this layer's cache laid out [seq][nkv * hs] (NOT the NumPy forward's [kv heads][seq][hs]);
// att: scratch of at least nh * (pos + 1) floats. Grouped-query attention: nh / nkv query heads share one kv head.
// The cache is walked position by position, all heads of a row at once: a head at a time meant nh strided passes
// over a cache that does not fit the caches of the CPU (12 layers of it), which cost more than the arithmetic.
// heads h0..h1 of nh (T109: the software threads take heads as they take rows of a matmul). Every head is computed
// alone, the same whatever the range, so the numbers do not depend on how the heads are shared out.
export function attention(out: usize, q: usize, kc: usize, vc: usize, att: usize, pos: i32, nh: i32, nkv: i32, hs: i32, h0: i32, h1: i32): void {
  attentionOf<f32>(out, q, kc, vc, att, pos, nh, nkv, hs, h0, h1);
}
// T110: the same over a cache of float16 keys and values (half the bytes to read, the longer the context the more
// that is). Each is widened to the float32 it stands for, exactly, and then everything is as above: the numbers are
// those of attention() on the widened cache.
export function attention_f16(out: usize, q: usize, kc: usize, vc: usize, att: usize, pos: i32, nh: i32, nkv: i32, hs: i32, h0: i32, h1: i32): void {
  attentionOf<u16>(out, q, kc, vc, att, pos, nh, nkv, hs, h0, h1);
}

// four keys or values from the cache as float32, and one
// @ts-ignore: decorator
@inline function kv4<T>(p: usize): v128 {
  return sizeof<T>() == 2 ? halves4(p) : v128.load(p);
}
// @ts-ignore: decorator
@inline function kv<T>(p: usize): f32 {
  return sizeof<T>() == 2 ? half(p) : load<f32>(p);
}

// T161: the sums of four accumulators, lane by lane: [sum a0, sum a1, sum a2, sum a3], each as (l0 + l2) + (l1 + l3),
// the order of sum4 below (a transpose by shuffles, then two adds: six instructions for four sums)
// @ts-ignore: decorator
@inline function sums4(a0: v128, a1: v128, a2: v128, a3: v128): v128 {
  const s01 = f32x4.add(v128.shuffle<f32>(a0, a1, 0, 4, 1, 5), v128.shuffle<f32>(a0, a1, 2, 6, 3, 7));  // a0 l0+l2, a1 l0+l2, a0 l1+l3, a1 l1+l3
  const s23 = f32x4.add(v128.shuffle<f32>(a2, a3, 0, 4, 1, 5), v128.shuffle<f32>(a2, a3, 2, 6, 3, 7));
  return f32x4.add(v128.shuffle<f32>(s01, s23, 0, 1, 4, 5), v128.shuffle<f32>(s01, s23, 2, 3, 6, 7));
}
// @ts-ignore: decorator
@inline function sum4(a: v128): f32 {
  return (f32x4.extract_lane(a, 0) + f32x4.extract_lane(a, 2)) + (f32x4.extract_lane(a, 1) + f32x4.extract_lane(a, 3));
}

// @ts-ignore: decorator
@inline function attentionOf<T>(out: usize, q: usize, kc: usize, vc: usize, att: usize, pos: i32, nh: i32, nkv: i32, hs: i32, h0: i32, h1: i32): void {
  const E: usize = sizeof<T>();  // bytes per key or value
  const kvDim = nkv * hs;
  const kvMul = nh / nkv;
  const hs4 = hs & ~3;
  const count = pos + 1, count4 = count & ~3;
  const isq: f32 = <f32>1.0 / sqrt<f32>(<f32>hs);
  const stride = <usize>kvDim * E;
  // 1. the scores of every head against every position. T161: four positions of a head at a time, each with its own
  // accumulator, so that the query's four floats are loaded once for four keys and one transposing reduction turns
  // the four accumulators into the four scores, stored at once (the scores of a head are side by side in att). The
  // form of several rows against one shared vector is llama.cpp's ggml_vec_dot_f16_unroll (ggml/src/ggml-cpu/vec.h,
  // MIT; no line of it is copied). A last position left over takes the same sums in the same order, one accumulator,
  // so a score does not depend on where its position falls.
  let t = 0;
  for (; t < count4; t += 4) {
    const row = kc + <usize>(t * kvDim) * E;
    for (let h = h0; h < h1; h++) {
      const qh = q + (<usize>(h * hs) << 2);
      const k0 = row + <usize>((h / kvMul) * hs) * E;
      const k1 = k0 + stride, k2 = k1 + stride, k3 = k2 + stride;
      let a0 = f32x4.splat(0), a1 = f32x4.splat(0), a2 = f32x4.splat(0), a3 = f32x4.splat(0);
      let j = 0;
      for (; j < hs4; j += 4) {
        const x = v128.load(qh + (<usize>j << 2)), e = <usize>j * E;
        a0 = f32x4.add(a0, f32x4.mul(x, kv4<T>(k0 + e)));
        a1 = f32x4.add(a1, f32x4.mul(x, kv4<T>(k1 + e)));
        a2 = f32x4.add(a2, f32x4.mul(x, kv4<T>(k2 + e)));
        a3 = f32x4.add(a3, f32x4.mul(x, kv4<T>(k3 + e)));
      }
      let sc = sums4(a0, a1, a2, a3);
      for (; j < hs; j++) {
        const x = load<f32>(qh + (<usize>j << 2)), e = <usize>j * E;
        sc = f32x4.add(sc, f32x4(x * kv<T>(k0 + e), x * kv<T>(k1 + e), x * kv<T>(k2 + e), x * kv<T>(k3 + e)));
      }
      v128.store(att + (<usize>(h * count + t) << 2), f32x4.mul(sc, f32x4.splat(isq)));
    }
  }
  for (; t < count; t++) {
    const row = kc + <usize>(t * kvDim) * E;
    for (let h = h0; h < h1; h++) {
      const qh = q + (<usize>(h * hs) << 2);
      const kt = row + <usize>((h / kvMul) * hs) * E;
      let a = f32x4.splat(0);
      let j = 0;
      for (; j < hs4; j += 4) a = f32x4.add(a, f32x4.mul(v128.load(qh + (<usize>j << 2)), kv4<T>(kt + <usize>j * E)));
      let sc: f32 = sum4(a);
      for (; j < hs; j++) sc += load<f32>(qh + (<usize>j << 2)) * kv<T>(kt + <usize>j * E);
      store<f32>(att + (<usize>(h * count + t) << 2), sc * isq);
    }
  }
  // 2. softmax, head by head, four exponentials at a time
  for (let h = h0; h < h1; h++) {
    const scores = att + (<usize>(h * count) << 2);
    const mx = largest(scores, count, -f32.MAX_VALUE);
    const mxs = f32x4.splat(mx);
    let sums = f32x4.splat(0);
    let t = 0;
    for (; t < count4; t += 4) {
      const e = vexp(f32x4.sub(v128.load(scores + (<usize>t << 2)), mxs));
      v128.store(scores + (<usize>t << 2), e);
      sums = f32x4.add(sums, e);
    }
    let sum: f32 = hsum(sums);
    for (; t < count; t++) {
      const e = fexp(load<f32>(scores + (<usize>t << 2)) - mx);
      store<f32>(scores + (<usize>t << 2), e);
      sum += e;
    }
    const inv: f32 = <f32>1.0 / sum;
    const invs = f32x4.splat(inv);
    for (t = 0; t < count4; t += 4) v128.store(scores + (<usize>t << 2), f32x4.mul(v128.load(scores + (<usize>t << 2)), invs));
    for (; t < count; t++) store<f32>(scores + (<usize>t << 2), load<f32>(scores + (<usize>t << 2)) * inv);
  }
  // 3. the weighted sum of the values, again row by row, four positions at a time: out is loaded and stored once
  // for the four of them
  for (let j = h0 * hs; j < h1 * hs; j++) store<f32>(out + (<usize>j << 2), 0);
  for (t = 0; t < count4; t += 4) {
    const row = vc + <usize>(t * kvDim) * E;
    for (let h = h0; h < h1; h++) {
      const weights = att + (<usize>(h * count + t) << 2);
      const w0 = f32x4.splat(load<f32>(weights)), w1 = f32x4.splat(load<f32>(weights + 4));
      const w2 = f32x4.splat(load<f32>(weights + 8)), w3 = f32x4.splat(load<f32>(weights + 12));
      const oh = out + (<usize>(h * hs) << 2);
      const v0 = row + <usize>((h / kvMul) * hs) * E;
      const v1 = v0 + stride, v2 = v1 + stride, v3 = v2 + stride;
      let j = 0;
      for (; j < hs4; j += 4) {
        const o = <usize>j << 2, e = <usize>j * E;
        const pair0 = f32x4.add(f32x4.mul(w0, kv4<T>(v0 + e)), f32x4.mul(w1, kv4<T>(v1 + e)));
        const pair1 = f32x4.add(f32x4.mul(w2, kv4<T>(v2 + e)), f32x4.mul(w3, kv4<T>(v3 + e)));
        v128.store(oh + o, f32x4.add(v128.load(oh + o), f32x4.add(pair0, pair1)));
      }
      for (; j < hs; j++) {
        const o = <usize>j << 2, e = <usize>j * E;
        store<f32>(oh + o, load<f32>(oh + o) + load<f32>(weights) * kv<T>(v0 + e) + load<f32>(weights + 4) * kv<T>(v1 + e)
          + load<f32>(weights + 8) * kv<T>(v2 + e) + load<f32>(weights + 12) * kv<T>(v3 + e));
      }
    }
  }
  for (; t < count; t++) {
    const row = vc + <usize>(t * kvDim) * E;
    for (let h = h0; h < h1; h++) {
      const weight: f32 = load<f32>(att + (<usize>(h * count + t) << 2));
      const a = f32x4.splat(weight);
      const oh = out + (<usize>(h * hs) << 2);
      const vt = row + <usize>((h / kvMul) * hs) * E;
      let j = 0;
      for (; j < hs4; j += 4) {
        const o = <usize>j << 2;
        v128.store(oh + o, f32x4.add(v128.load(oh + o), f32x4.mul(a, kv4<T>(vt + <usize>j * E))));
      }
      for (; j < hs; j++) {
        const o = <usize>j << 2;
        store<f32>(oh + o, load<f32>(oh + o) + weight * kv<T>(vt + <usize>j * E));
      }
    }
  }
}
