// relaxed/signs.ts (T644.6, T644.7): the products of ternary and binary weights with relaxed SIMD, a token's
// (matmul_t2r, matmul_b1r) and a prompt's (matmul_t2r_tile, matmul_b1r_tile). One kernel for both packings, the
// packing a type parameter P (../signs.ts): binary's numbers are ternary's to the bit for the same weights.
// T686: and the form of a group's sums a second one S: Wide (the dot products added in int32, as ever) or Pairs (the
// planes' products added in int16 and widened once a group of 128: matmul_t2x and matmul_t2x_tile, ternary's alone).
// Both make the same exact integers, so the same numbers to the bit. Pairs is x86's form: there the int16 dot is
// pmaddubsw alone, where the int32 one is pmaddubsw and pmaddwd (T288: 1.33 to 1.36 times on AMD's Zen 3 and 4 in V8,
// 0.63 to 0.65 on arm64's Neoverse-N2), and the engine takes it where it times faster (src/runtime/forward/pairs.ts).

import { fourSums } from "../ternary";
import { Ternary, Binary, groupBytes, rowBytes, blockBytes, codesOf } from "../signs";
import { half, halves4 } from "../kernel/halves";
import { hsum } from "../kernel/math";

/** a group's sums as dot products into int32 (i32x4.relaxed_dot_i8x16_i7x16_add_s) */
export class Wide {}
/** T686: as the pairs' products into int16 (i16x8.relaxed_dot_i8x16_i7x16_s), widened once a group of 128 */
export class Pairs {}

// The four planes' dot products of 64 weights (their codes c0..c3, the tile's made once for four tokens; dot64() is the
// same for a token's, its codes made in it) by the activations at x. Wide: int32 lanes, 0 and 1
// of the first group of 32 activations, 2 and 3 of the second. Pairs (T686): int16 lanes, 0..3 of the first group,
// 4..7 of the second; a lane is at most 4 planes x a pair x 384 (|-128| x PQ2_0's fourth code, 3) = 3072, and no pair
// saturates (768 of x86's 32767)
// @ts-ignore: decorator
@inline function planes<S>(x: usize, c0: v128, c1: v128, c2: v128, c3: v128): v128 {
  if (idof<S>() == idof<Pairs>()) {
    let sum = i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x), c0);
    sum = i16x8.add(sum, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 16), c1));
    sum = i16x8.add(sum, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 32), c2));
    return i16x8.add(sum, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 48), c3));
  }
  let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x), c0, i32x4.splat(0));
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 16), c1, acc);
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 32), c2, acc);
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 48), c3, acc);
}
// lo and hi, planes<S>() of a group's two blocks of 64: the four groups' int32 sums. Pairs: the 32-bit lanes
// [lo0 lo1 | lo2 lo3] hold groups 0 and 1, hi's 2 and 3, two int16 a group in each lane of the sum (at most 6144)
// @ts-ignore: decorator
@inline function sumsOf<S>(lo: v128, hi: v128): v128 {
  if (idof<S>() == idof<Pairs>()) return i32x4.extadd_pairwise_i16x8_s(i16x8.add(v128.shuffle<i32>(lo, hi, 0, 2, 4, 6), v128.shuffle<i32>(lo, hi, 1, 3, 5, 7)));
  return fourSums(lo, hi);
}
// T231, T634: ternary weights (ternary.ts: 32 bytes a group of 128) or binary ones (binary.ts: 16 bytes), a float16
// scale a group (T533), against the activations interleave() left (kernel/matmul.ts: int8 in all 8 bits, the planes
// of a block of 64, and after their scales minus the sum of each group of 32). 64 weights a step: four planes of codes
// of their sixteen bytes v (block h's: ../signs.ts), four dot products into one accumulator whose lanes 0 and 1 are of
// the first group of 32 activations and 2 and 3 of the second. A code times an activation is at most 2 x 128, two of
// them an int16 (x86's pmaddubsw adds pairs with saturation).
// @ts-ignore: decorator
@inline function dot64<P, S>(v: v128, x: usize, h: i32, mask: v128): v128 {
  // (each plane's codes made where its product takes them, not the four first: Binaryen scheduled matmul_b1r's 64-bit
  // build otherwise, where the int32 form's code was to stay as it was)
  if (idof<S>() == idof<Pairs>()) {
    let sum = i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x), codesOf<P>(v, h, 0, mask));
    sum = i16x8.add(sum, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 16), codesOf<P>(v, h, 1, mask)));
    sum = i16x8.add(sum, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 32), codesOf<P>(v, h, 2, mask)));
    return i16x8.add(sum, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 48), codesOf<P>(v, h, 3, mask)));
  }
  let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x), codesOf<P>(v, h, 0, mask), i32x4.splat(0));
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 16), codesOf<P>(v, h, 1, mask), acc);
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 32), codesOf<P>(v, h, 2, mask), acc);
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 48), codesOf<P>(v, h, 3, mask), acc);
}
// A group of 128 weights a turn: the exact integer sums of its four groups of 32 activations (minus each one's sum
// added: dot(a, w) of the weights -1, 0, 1), times the activations' four scales, times the weights' one (d: in all
// four lanes). Lane k of the accumulator holds the k-th group of 32 of every group of 128, in order; the lanes are
// added in order at the end.
// @ts-ignore: decorator
@inline function group128<P, S>(facc: v128, w: usize, x: usize, sums: usize, xs: usize, at: usize, d: v128, mask: v128): v128 {
  const v = v128.load(w);
  const whole = i32x4.add(sumsOf<S>(dot64<P, S>(v, x, 0, mask), dot64<P, S>(blockBytes<P>(w, v, 1), x + 64, 1, mask)), v128.load(sums + at));
  return f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(whole), v128.load(xs + at)), d));
}
// T533: the scales are float16s, four groups' widened at once (ternary.ts), the groups past the last four one at a time.
// @ts-ignore: decorator
@inline function signsRows<P, S>(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, code: i32): void {
  const groups = n >> 7, groups4 = groups & ~3, mask = i8x16.splat(<i8>code), bytes = groupBytes<P>();
  const sums = xs + (<usize>(n >> 5) << 2);  // after the scales: minus the sum of each group's activations
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * rowBytes<P>(n);
    const srow = ws + ((<usize>i * <usize>groups) << 1);
    let facc = f32x4.splat(0);
    let g = 0;
    for (; g < groups4; g += 4) {
      const d = halves4(srow + (<usize>g << 1)), w = row + <usize>g * bytes, x = xq + (<usize>g << 7), at = <usize>g << 4;
      facc = group128<P, S>(facc, w, x, sums, xs, at, v128.shuffle<i32>(d, d, 0, 0, 0, 0), mask);
      facc = group128<P, S>(facc, w + bytes, x + 128, sums, xs, at + 16, v128.shuffle<i32>(d, d, 1, 1, 1, 1), mask);
      facc = group128<P, S>(facc, w + 2 * bytes, x + 256, sums, xs, at + 32, v128.shuffle<i32>(d, d, 2, 2, 2, 2), mask);
      facc = group128<P, S>(facc, w + 3 * bytes, x + 384, sums, xs, at + 48, v128.shuffle<i32>(d, d, 3, 3, 3, 3), mask);
    }
    for (; g < groups; g++) {
      facc = group128<P, S>(facc, row + <usize>g * bytes, xq + (<usize>g << 7), sums, xs, <usize>g << 4, f32x4.splat(half(srow + (<usize>g << 1))), mask);
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// T231: the token's kernel for count tokens of a prompt (T108) at once: token t's activations and their scales (and
// sums) at xq + t * frame and xs + t * frame, its outputs at xout + t * os. A row against four tokens a turn: the codes
// of a group are made once for the four (the loads, the shifts and the masks are half of what a token's 64 weights
// cost) and meet each token's activations in turn. Every (row, token) is the token's kernel's number to the bit: the
// same integer sums, scaled and added in the same order. The tokens past the last four go through the token's kernel
// itself. A row's bytes are read in order, once for every four tokens, and stay in the first cache meanwhile (a
// ternary row of 17408 weights is 4 KB).
// @ts-ignore: decorator
@inline function scaled<S>(lo: v128, hi: v128, xs: usize, sums: usize, at: usize, d: v128): v128 {
  const whole = i32x4.add(sumsOf<S>(lo, hi), v128.load(xs + sums + at));
  return f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(whole), v128.load(xs + at)), d);
}
// @ts-ignore: decorator
@inline function signsTile<P, S>(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, code: i32): void {
  const groups = n >> 7, mask = i8x16.splat(<i8>code), count4 = count & ~3, bytes = groupBytes<P>();
  const sums = <usize>(n >> 5) << 2, outStride = <usize>os, f = <usize>frame;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * rowBytes<P>(n);
    const srow = ws + ((<usize>i * <usize>groups) << 1);
    let t = 0;
    for (; t < count4; t += 4) {
      const x0 = xq + <usize>t * f, x1 = x0 + f, x2 = x1 + f, x3 = x2 + f;
      const s0 = xs + <usize>t * f, s1 = s0 + f, s2 = s1 + f, s3 = s2 + f;
      let f0 = f32x4.splat(0), f1 = f0, f2 = f0, f3 = f0;
      for (let g = 0; g < groups; g++) {
        const w = row + <usize>g * bytes, o = <usize>g << 7, at = <usize>g << 4;
        const v = v128.load(w);
        let c0 = codesOf<P>(v, 0, 0, mask), c1 = codesOf<P>(v, 0, 1, mask), c2 = codesOf<P>(v, 0, 2, mask), c3 = codesOf<P>(v, 0, 3, mask);
        const a0 = planes<S>(x0 + o, c0, c1, c2, c3), a1 = planes<S>(x1 + o, c0, c1, c2, c3);
        const a2 = planes<S>(x2 + o, c0, c1, c2, c3), a3 = planes<S>(x3 + o, c0, c1, c2, c3);
        const u = blockBytes<P>(w, v, 1);
        c0 = codesOf<P>(u, 1, 0, mask); c1 = codesOf<P>(u, 1, 1, mask); c2 = codesOf<P>(u, 1, 2, mask); c3 = codesOf<P>(u, 1, 3, mask);
        const d = f32x4.splat(half(srow + (<usize>g << 1)));  // T533: a float16, once for the four tokens
        f0 = f32x4.add(f0, scaled<S>(a0, planes<S>(x0 + o + 64, c0, c1, c2, c3), s0, sums, at, d));
        f1 = f32x4.add(f1, scaled<S>(a1, planes<S>(x1 + o + 64, c0, c1, c2, c3), s1, sums, at, d));
        f2 = f32x4.add(f2, scaled<S>(a2, planes<S>(x2 + o + 64, c0, c1, c2, c3), s2, sums, at, d));
        f3 = f32x4.add(f3, scaled<S>(a3, planes<S>(x3 + o + 64, c0, c1, c2, c3), s3, sums, at, d));
      }
      const out = xout + <usize>t * outStride + (<usize>i << 2);
      store<f32>(out, hsum(f0));
      store<f32>(out + outStride, hsum(f1));
      store<f32>(out + 2 * outStride, hsum(f2));
      store<f32>(out + 3 * outStride, hsum(f3));
    }
    for (; t < count; t++) {
      const to = xout + <usize>t * outStride, x = xq + <usize>t * f, s = xs + <usize>t * f;
      if (idof<P>() != idof<Ternary>()) matmul_b1r(to, x, s, wq, ws, n, i, i + 1, code);
      else if (idof<S>() == idof<Pairs>()) matmul_t2x(to, x, s, wq, ws, n, i, i + 1, code);
      else matmul_t2r(to, x, s, wq, ws, n, i, i + 1, code);
    }
  }
}

// three: 3, two: 2 (ternary.ts and binary.ts say why they are arguments)
export function matmul_t2r(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  signsRows<Ternary, Wide>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
export function matmul_t2r_tile(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, three: i32): void {
  signsTile<Ternary, Wide>(xout, xq, xs, wq, ws, n, r0, r1, count, os, frame, three);
}
export function matmul_b1r(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, two: i32): void {
  signsRows<Binary, Wide>(xout, xq, xs, wq, ws, n, r0, r1, two);
}
export function matmul_b1r_tile(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, two: i32): void {
  signsTile<Binary, Wide>(xout, xq, xs, wq, ws, n, r0, r1, count, os, frame, two);
}
// T686: matmul_t2r and matmul_t2r_tile with the sums of Pairs, each number theirs to the bit
export function matmul_t2x(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  signsRows<Ternary, Pairs>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
export function matmul_t2x_tile(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, three: i32): void {
  signsTile<Ternary, Pairs>(xout, xq, xs, wq, ws, n, r0, r1, count, os, frame, three);
}
