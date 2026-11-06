// relaxed/ternary.ts (T644.6): the ternary products with relaxed SIMD, a token's (matmul_t2r) and a prompt's
// (matmul_t2r_tile), on the codes of ../ternary.ts. planes and scaled are relaxed/binary.ts's too.

import { codes, fourSums } from "../ternary";
import { half, halves4 } from "../kernel/halves";
import { hsum } from "../kernel/math";

// T231: ternary weights (ternary.ts: 32 bytes a group of 128, a float16 scale a group from T533) against the activations
// interleave() left (kernel/matmul.ts: int8 in all 8 bits, the planes of a block of 64, and after their scales minus the sum
// of each group of 32). 64 weights a step: their sixteen bytes loaded once, four planes of codes, four dot products
// into one accumulator whose lanes 0 and 1 are of the first group of 32 activations and 2 and 3 of the second.
// A code times an activation is at most 2 x 128, two of them an int16 (x86's pmaddubsw adds pairs with saturation).
// @ts-ignore: decorator
@inline function dot64(w: usize, x: usize, three: v128): v128 {
  const v = v128.load(w);
  let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x), codes(v, 0, three), i32x4.splat(0));
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 16), codes(v, 1, three), acc);
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 32), codes(v, 2, three), acc);
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 48), codes(v, 3, three), acc);
}
// A group of 128 weights a turn: the exact integer sums of its four groups of 32 activations (minus each one's sum
// added: dot(a, w) of the weights -1, 0, 1), times the activations' four scales, times the weights' one (d: in all
// four lanes). Lane k of the accumulator holds the k-th group of 32 of every group of 128, in order; the lanes are
// added in order at the end.
// @ts-ignore: decorator
@inline function group128(facc: v128, w: usize, x: usize, sums: usize, xs: usize, at: usize, d: v128, mask: v128): v128 {
  const whole = i32x4.add(fourSums(dot64(w, x, mask), dot64(w + 16, x + 64, mask)), v128.load(sums + at));
  return f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(whole), v128.load(xs + at)), d));
}
// T533: the scales are float16s, four groups' widened at once (ternary.ts), the groups past the last four one at a time.
// three: 3 (ternary.ts says why it is an argument).
export function matmul_t2r(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  const groups = n >> 7, groups4 = groups & ~3, mask = i8x16.splat(<i8>three);
  const sums = xs + (<usize>(n >> 5) << 2);  // after the scales: minus the sum of each group's activations
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 2);
    const srow = ws + ((<usize>i * <usize>groups) << 1);
    let facc = f32x4.splat(0);
    let g = 0;
    for (; g < groups4; g += 4) {
      const d = halves4(srow + (<usize>g << 1)), w = row + (<usize>g << 5), x = xq + (<usize>g << 7), at = <usize>g << 4;
      facc = group128(facc, w, x, sums, xs, at, v128.shuffle<i32>(d, d, 0, 0, 0, 0), mask);
      facc = group128(facc, w + 32, x + 128, sums, xs, at + 16, v128.shuffle<i32>(d, d, 1, 1, 1, 1), mask);
      facc = group128(facc, w + 64, x + 256, sums, xs, at + 32, v128.shuffle<i32>(d, d, 2, 2, 2, 2), mask);
      facc = group128(facc, w + 96, x + 384, sums, xs, at + 48, v128.shuffle<i32>(d, d, 3, 3, 3, 3), mask);
    }
    for (; g < groups; g++) {
      facc = group128(facc, row + (<usize>g << 5), xq + (<usize>g << 7), sums, xs, <usize>g << 4, f32x4.splat(half(srow + (<usize>g << 1))), mask);
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// T231: matmul_t2r for count tokens of a prompt (T108) at once: token t's activations and their scales (and sums) at
// xq + t * frame and xs + t * frame, its outputs at xout + t * os. A row against four tokens a turn: the codes of its
// 64 weights are made once for the four (the load, the shifts and the masks are half of what a token's 64 weights
// cost) and meet each token's activations in turn. Every (row, token) is matmul_t2r's number to the bit: the same
// integer sums, scaled and added in the same order. The tokens past the last four go through matmul_t2r itself.
// A row's bytes are read in order, once for every four tokens, and stay in the first cache meanwhile (a row of 17408
// weights is 4 KB).
// @ts-ignore: decorator
@inline function planes(x: usize, c0: v128, c1: v128, c2: v128, c3: v128): v128 {
  let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x), c0, i32x4.splat(0));
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 16), c1, acc);
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 32), c2, acc);
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 48), c3, acc);
}
// @ts-ignore: decorator
@inline function scaled(lo: v128, hi: v128, xs: usize, sums: usize, at: usize, d: v128): v128 {
  const whole = i32x4.add(fourSums(lo, hi), v128.load(xs + sums + at));
  return f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(whole), v128.load(xs + at)), d);
}
export function matmul_t2r_tile(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, three: i32): void {
  const groups = n >> 7, mask = i8x16.splat(<i8>three), count4 = count & ~3;
  const sums = <usize>(n >> 5) << 2, outStride = <usize>os, f = <usize>frame;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 2);
    const srow = ws + ((<usize>i * <usize>groups) << 1);
    let t = 0;
    for (; t < count4; t += 4) {
      const x0 = xq + <usize>t * f, x1 = x0 + f, x2 = x1 + f, x3 = x2 + f;
      const s0 = xs + <usize>t * f, s1 = s0 + f, s2 = s1 + f, s3 = s2 + f;
      let f0 = f32x4.splat(0), f1 = f0, f2 = f0, f3 = f0;
      for (let g = 0; g < groups; g++) {
        const w = row + (<usize>g << 5), o = <usize>g << 7, at = <usize>g << 4;
        const v = v128.load(w);
        let c0 = codes(v, 0, mask), c1 = codes(v, 1, mask), c2 = codes(v, 2, mask), c3 = codes(v, 3, mask);
        const a0 = planes(x0 + o, c0, c1, c2, c3), a1 = planes(x1 + o, c0, c1, c2, c3);
        const a2 = planes(x2 + o, c0, c1, c2, c3), a3 = planes(x3 + o, c0, c1, c2, c3);
        const u = v128.load(w, 16);
        c0 = codes(u, 0, mask); c1 = codes(u, 1, mask); c2 = codes(u, 2, mask); c3 = codes(u, 3, mask);
        const d = f32x4.splat(half(srow + (<usize>g << 1)));  // T533: a float16, once for the four tokens
        f0 = f32x4.add(f0, scaled(a0, planes(x0 + o + 64, c0, c1, c2, c3), s0, sums, at, d));
        f1 = f32x4.add(f1, scaled(a1, planes(x1 + o + 64, c0, c1, c2, c3), s1, sums, at, d));
        f2 = f32x4.add(f2, scaled(a2, planes(x2 + o + 64, c0, c1, c2, c3), s2, sums, at, d));
        f3 = f32x4.add(f3, scaled(a3, planes(x3 + o + 64, c0, c1, c2, c3), s3, sums, at, d));
      }
      const out = xout + <usize>t * outStride + (<usize>i << 2);
      store<f32>(out, hsum(f0));
      store<f32>(out + outStride, hsum(f1));
      store<f32>(out + 2 * outStride, hsum(f2));
      store<f32>(out + 3 * outStride, hsum(f3));
    }
    for (; t < count; t++) matmul_t2r(xout + <usize>t * outStride, xq + <usize>t * f, xs + <usize>t * f, wq, ws, n, i, i + 1, three);
  }
}

export { planes, scaled };
