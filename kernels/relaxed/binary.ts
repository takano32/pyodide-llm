// relaxed/binary.ts (T644.6): the binary products with relaxed SIMD, a token's (matmul_b1r) and a prompt's
// (matmul_b1r_tile), on the codes of ../binary.ts: relaxed/ternary.ts's numbers to the bit for the same weights.

import { bitCodes } from "../binary";
import { fourSums } from "../ternary";
import { half, halves4 } from "../kernel/halves";
import { hsum } from "../kernel/math";
import { planes, scaled } from "./ternary";

// T634: binary weights (binary.ts: 16 bytes a group of 128, a float16 scale a group) against the activations
// interleave() left, as matmul_t2r takes them: the same integer sums of every group of 32 (the codes of a weight are
// ternary's), the same float32 operations in the same order, so matmul_t2r's numbers to the bit for the same weights.
// A group's sixteen bytes are loaded once for its 128 weights, eight planes of codes.
// @ts-ignore: decorator
@inline function bits64(v: v128, x: usize, h: i32, two: v128): v128 {
  let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x), bitCodes(v, 4 * h, two), i32x4.splat(0));
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 16), bitCodes(v, 4 * h + 1, two), acc);
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 32), bitCodes(v, 4 * h + 2, two), acc);
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 48), bitCodes(v, 4 * h + 3, two), acc);
}
// @ts-ignore: decorator
@inline function bits128(facc: v128, w: usize, x: usize, sums: usize, xs: usize, at: usize, d: v128, two: v128): v128 {
  const v = v128.load(w);
  const whole = i32x4.add(fourSums(bits64(v, x, 0, two), bits64(v, x + 64, 1, two)), v128.load(sums + at));
  return f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(whole), v128.load(xs + at)), d));
}
// two: 2 (binary.ts says why it is an argument)
export function matmul_b1r(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, two: i32): void {
  const groups = n >> 7, groups4 = groups & ~3, mask = i8x16.splat(<i8>two);
  const sums = xs + (<usize>(n >> 5) << 2);  // after the scales: minus the sum of each group's activations
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 3);
    const srow = ws + ((<usize>i * <usize>groups) << 1);
    let facc = f32x4.splat(0);
    let g = 0;
    for (; g < groups4; g += 4) {
      const d = halves4(srow + (<usize>g << 1)), w = row + (<usize>g << 4), x = xq + (<usize>g << 7), at = <usize>g << 4;
      facc = bits128(facc, w, x, sums, xs, at, v128.shuffle<i32>(d, d, 0, 0, 0, 0), mask);
      facc = bits128(facc, w + 16, x + 128, sums, xs, at + 16, v128.shuffle<i32>(d, d, 1, 1, 1, 1), mask);
      facc = bits128(facc, w + 32, x + 256, sums, xs, at + 32, v128.shuffle<i32>(d, d, 2, 2, 2, 2), mask);
      facc = bits128(facc, w + 48, x + 384, sums, xs, at + 48, v128.shuffle<i32>(d, d, 3, 3, 3, 3), mask);
    }
    for (; g < groups; g++) {
      facc = bits128(facc, row + (<usize>g << 4), xq + (<usize>g << 7), sums, xs, <usize>g << 4, f32x4.splat(half(srow + (<usize>g << 1))), mask);
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// T634: matmul_b1r for count tokens of a prompt at once, as matmul_t2r_tile is for ternary: a row against four tokens
// a turn, the eight planes of codes of a group made once for the four. Every (row, token) is matmul_b1r's number to the
// bit (and so matmul_t2r_tile's for the same weights). The tokens past the last four go through matmul_b1r itself.
export function matmul_b1r_tile(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, two: i32): void {
  const groups = n >> 7, mask = i8x16.splat(<i8>two), count4 = count & ~3;
  const sums = <usize>(n >> 5) << 2, outStride = <usize>os, f = <usize>frame;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 3);
    const srow = ws + ((<usize>i * <usize>groups) << 1);
    let t = 0;
    for (; t < count4; t += 4) {
      const x0 = xq + <usize>t * f, x1 = x0 + f, x2 = x1 + f, x3 = x2 + f;
      const s0 = xs + <usize>t * f, s1 = s0 + f, s2 = s1 + f, s3 = s2 + f;
      let f0 = f32x4.splat(0), f1 = f0, f2 = f0, f3 = f0;
      for (let g = 0; g < groups; g++) {
        const v = v128.load(row + (<usize>g << 4)), o = <usize>g << 7, at = <usize>g << 4;
        let c0 = bitCodes(v, 0, mask), c1 = bitCodes(v, 1, mask), c2 = bitCodes(v, 2, mask), c3 = bitCodes(v, 3, mask);
        const a0 = planes(x0 + o, c0, c1, c2, c3), a1 = planes(x1 + o, c0, c1, c2, c3);
        const a2 = planes(x2 + o, c0, c1, c2, c3), a3 = planes(x3 + o, c0, c1, c2, c3);
        c0 = bitCodes(v, 4, mask); c1 = bitCodes(v, 5, mask); c2 = bitCodes(v, 6, mask); c3 = bitCodes(v, 7, mask);
        const d = f32x4.splat(half(srow + (<usize>g << 1)));
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
    for (; t < count; t++) matmul_b1r(xout + <usize>t * outStride, xq + <usize>t * f, xs + <usize>t * f, wq, ws, n, i, i + 1, two);
  }
}
