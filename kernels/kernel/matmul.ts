// kernel/matmul.ts (T356): the matrix products without relaxed SIMD (those with it are kernel_relaxed.ts), W (d, n)
// times x (n) for rows r0 to r1, and what they take beside the weights: the sums of the weights' groups (the
// corrections of the relaxed products), the activations laid out for the ternary products (interleave; the binary ones,
// T634, take them so too), and the
// outliers' columns of the classifier (add_columns).

import { sixFirst, sixSecond, sixTops } from "../six";
import { Ternary, Binary, groupBytes, rowBytes, blockBytes, codesOf } from "../signs";
import { half, halves4 } from "./halves";
import { GS, hsum, laneSum } from "./math";

// W (d,n) @ x (n,) -> xout, rows [r0, r1)
export function matmul_f32(xout: usize, x: usize, w: usize, n: i32, r0: i32, r1: i32): void {
  const n16 = n & ~15;
  for (let i = r0; i < r1; i++) {
    const row = w + ((<usize>i * <usize>n) << 2);
    let a0 = f32x4.splat(0), a1 = f32x4.splat(0), a2 = f32x4.splat(0), a3 = f32x4.splat(0);
    let j = 0;
    for (; j < n16; j += 16) {
      const o = <usize>j << 2;
      a0 = f32x4.add(a0, f32x4.mul(v128.load(row + o), v128.load(x + o)));
      a1 = f32x4.add(a1, f32x4.mul(v128.load(row + o + 16), v128.load(x + o + 16)));
      a2 = f32x4.add(a2, f32x4.mul(v128.load(row + o + 32), v128.load(x + o + 32)));
      a3 = f32x4.add(a3, f32x4.mul(v128.load(row + o + 48), v128.load(x + o + 48)));
    }
    let val: f32 = hsum(f32x4.add(f32x4.add(a0, a1), f32x4.add(a2, a3)));
    for (; j < n; j++) {
      const o = <usize>j << 2;
      val += load<f32>(row + o) * load<f32>(x + o);
    }
    store<f32>(xout + (<usize>i << 2), val);
  }
}

// T98: the corrections of matmul_q6r for int6 weights (w, groups of 24 bytes): out[g] = -64 times the sum of the
// group's int8 values, an int32 (T197: before it the float32 scale times the sum; the kernels of kernel_relaxed.ts
// add it to the group's integer sum now)
export function six_sums(out: usize, w: usize, groups: i32): void {
  for (let g = 0; g < groups; g++) {
    const p = w + <usize>g * 24, low = v128.load(p), t = sixTops(p);
    const pairs = i16x8.add(i16x8.extadd_pairwise_i8x16_s(sixFirst(low, t)), i16x8.extadd_pairwise_i8x16_s(sixSecond(low, t)));
    const quads = i32x4.extadd_pairwise_i16x8_s(pairs);
    const sum = laneSum(quads);
    store<i32>(out + (<usize>g << 2), -64 * sum);
  }
}

// T123: the same for int8 weights: the sums of groups of 32 int8 values (w), times -64 (T197), the corrections of
// matmul_q8r. forward.ts made them one value at a time in JavaScript (7B: 266 s of its construct)
export function int8_sums(out: usize, w: usize, groups: i32): void {
  for (let g = 0; g < groups; g++) {
    const p = w + (<usize>g << 5);
    const pairs = i16x8.add(i16x8.extadd_pairwise_i8x16_s(v128.load(p)), i16x8.extadd_pairwise_i8x16_s(v128.load(p + 16)));
    const quads = i32x4.extadd_pairwise_i16x8_s(pairs);
    const sum = laneSum(quads);
    store<i32>(out + (<usize>g << 2), -64 * sum);
  }
}

// T165: the 32 products of one group (a0 b0, a1 b1: 16 int8 each), summed into four int32 lanes. Two products are
// added in int16 before the widening: 9 instructions, not 11 (V8 on arm64 folds the add into smlal: 6, not 9).
// int8 times int8 is 16384 at most (-128 times -128), and two such sums are within int16 unless both factors are
// -128 twice over. The weights and the activations reach -128 only in a group whose scale is below 2.9e-39
// (quantize_x where 1/scale overflows, T136), so the scale of that product (the two scales multiplied) is 0 in
// float32 and the lane is multiplied by 0 either way (at most the sign of a zero changes). The lanes hold the same
// products as before (lane k: products 2k, 2k + 1, 2k + 8, 2k + 9 of each half), so the sums are the same integers.
// @ts-ignore: decorator
@inline function dot32(a0: v128, b0: v128, a1: v128, b1: v128): v128 {
  return i32x4.add(
    i32x4.extadd_pairwise_i16x8_s(i16x8.add(i16x8.extmul_low_i8x16_s(a0, b0), i16x8.extmul_high_i8x16_s(a0, b0))),
    i32x4.extadd_pairwise_i16x8_s(i16x8.add(i16x8.extmul_low_i8x16_s(a1, b1), i16x8.extmul_high_i8x16_s(a1, b1))));
}

// int8 weights (wq) with one float32 scale per group (ws), int8 activations from quantize_x(bias = 0)
// T165: four groups a turn share one load of their scales (ws times xs, four at a time: the same products as one by
// one) and one step of the loop, and add into facc in the same order, so the result is the same to the bit
export function matmul_q8(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>n;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    let facc = f32x4.splat(0);
    let g = 0;
    for (; g + 4 <= ng; g += 4) {
      const sv = f32x4.mul(v128.load(srow + (<usize>g << 2)), v128.load(xs + (<usize>g << 2)));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g)), f32x4.splat(f32x4.extract_lane(sv, 0))));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g + 1)), f32x4.splat(f32x4.extract_lane(sv, 1))));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g + 2)), f32x4.splat(f32x4.extract_lane(sv, 2))));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g + 3)), f32x4.splat(f32x4.extract_lane(sv, 3))));
    }
    for (; g < ng; g++) {
      const s = load<f32>(srow + (<usize>g << 2)) * load<f32>(xs + (<usize>g << 2));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g)), f32x4.splat(s)));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// the four int32 lanes of group g of an int8 row (row) against the int8 activations (xq)
// @ts-ignore: decorator
@inline function groupQ8(row: usize, xq: usize, g: i32): v128 {
  const o = <usize>(g * GS);
  return dot32(v128.load(row + o), v128.load(xq + o), v128.load(row + o + 16), v128.load(xq + o + 16));
}

// ---- T230, T231: ternary weights (ternary.ts)
// The int8 activations of quantize_x (no bias) as the ternary kernels read them, in place: every block of 64 in four
// planes, byte 16 p + c of the block activation 4 c + p (plane p meets the codes of weights 4 c + p), and after the
// n / 32 scales at xs an int32 a group of 32: minus the sum of its activations, which a row's sums take (the codes
// are the weights + 1). n is a multiple of 64 (a ternary row is whole groups of 128).
// @ts-ignore: decorator
@inline function sumOf32(a: v128, b: v128): i32 {
  const quads = i32x4.extadd_pairwise_i16x8_s(i16x8.add(i16x8.extadd_pairwise_i8x16_s(a), i16x8.extadd_pairwise_i8x16_s(b)));
  return laneSum(quads);
}
// Plane r of a block of 64: byte r of every dword of the four 16-byte vectors (activations 4 c + r for c = 0..15), a byte of
// every dword shifted down and masked to 0..255, then narrowed from 32 bits to 16 and to 8 (the values pass the narrowing
// unchanged): the four vectors' four dwords each are the sixteen bytes in order.
// With no shuffle in it, on purpose. This was a transpose of byte, dword and qword shuffles (a byte shuffle of each
// vector, then dwords of two, then qwords of two), and JavaScriptCore's optimizing tier on x86-64 (seen in WebKit on Linux, on
// AMD EPYCs and an Intel Xeon alike; Safari on an Intel Mac has the same compiler and was not tried), which a function reaches after about 2000
// calls, folded those three wrongly: three quarters of the bytes
// came out wrong, and a ternary model wrote nonsense (T230's review; tests/kernels-in-browser.ts finds it). Every piece of
// it alone and a swizzle in place of the first shuffle were right, and this costs 5 to 8 ns more a block of 64 bytes
// (V8 on CI's runners: 3.6 to 10.5 on an EPYC 9V74, 4.8 to 9.9 on a 7763, 4.5 to 12.4 on arm64): of the 5,400 blocks of a
// token of the 1.7B, 0.03 to 0.04 ms (the other threads wait for it), against a token of 30 ms (arm64, 4 threads) to 90 ms (1 thread).
// @ts-ignore: decorator
@inline function plane(v0: v128, v1: v128, v2: v128, v3: v128, r: i32): v128 {
  const low = i32x4.splat(255);
  return i8x16.narrow_i16x8_u(
    i16x8.narrow_i32x4_u(v128.and(i32x4.shr_u(v0, 8 * r), low), v128.and(i32x4.shr_u(v1, 8 * r), low)),
    i16x8.narrow_i32x4_u(v128.and(i32x4.shr_u(v2, 8 * r), low), v128.and(i32x4.shr_u(v3, 8 * r), low)));
}
export function interleave(xq: usize, xs: usize, n: i32): void {
  const sums = xs + (<usize>(n >> 5) << 2);
  for (let b = 0; b < n; b += 64) {
    const p = xq + <usize>b, at = sums + (<usize>(b >> 5) << 2);
    const v0 = v128.load(p), v1 = v128.load(p, 16), v2 = v128.load(p, 32), v3 = v128.load(p, 48);
    store<i32>(at, -sumOf32(v0, v1));
    store<i32>(at, -sumOf32(v2, v3), 4);
    v128.store(p, plane(v0, v1, v2, v3, 0));
    v128.store(p, plane(v0, v1, v2, v3, 1), 16);
    v128.store(p, plane(v0, v1, v2, v3, 2), 32);
    v128.store(p, plane(v0, v1, v2, v3, 3), 48);
  }
}

// matmul_t2r and matmul_b1r (relaxed/signs.ts) without relaxed SIMD: the same weights, activations and sums, the same
// exact integers a group and the same float32 operations in the same order, so the same numbers to the bit (and
// binary's ternary's for the same weights: signs.ts). A plane's products are widened to int16 (its low eight bytes are
// of the first group of 32 activations, its high eight of the second) and added over the four planes there as a chain
// (4 x 3 x 128 a lane at most, PQ2_0's fourth code too); then the eight lanes of each of a group's four groups of 32
// are added in int16 too (at most 8 x 1536), and widened once. T644.7: one kernel for both packings, P (signs.ts).
// T275 (2026-11-12): the chain and the sums in int16 in place of a tree of adds and four widenings to int32, timed by
// tests/ternary-browser-bench.ts against tests/ternary-plain-forms.ts's copy of the form before (kernels/README.md has
// the table): 1.16 to 1.17 times as fast in V8 on arm64, 1.03 on x86-64, 1.07 in WebKit's JavaScriptCore on arm64.
// the int16 sums of the four planes' products: their low halves (low16) and their high (high16)
// @ts-ignore: decorator
@inline function low16(x0: v128, x1: v128, x2: v128, x3: v128, c0: v128, c1: v128, c2: v128, c3: v128): v128 {
  let sum = i16x8.extmul_low_i8x16_s(x0, c0);
  sum = i16x8.add(sum, i16x8.extmul_low_i8x16_s(x1, c1));
  sum = i16x8.add(sum, i16x8.extmul_low_i8x16_s(x2, c2));
  return i16x8.add(sum, i16x8.extmul_low_i8x16_s(x3, c3));
}
// @ts-ignore: decorator
@inline function high16(x0: v128, x1: v128, x2: v128, x3: v128, c0: v128, c1: v128, c2: v128, c3: v128): v128 {
  let sum = i16x8.extmul_high_i8x16_s(x0, c0);
  sum = i16x8.add(sum, i16x8.extmul_high_i8x16_s(x1, c1));
  sum = i16x8.add(sum, i16x8.extmul_high_i8x16_s(x2, c2));
  return i16x8.add(sum, i16x8.extmul_high_i8x16_s(x3, c3));
}
// the int16 sums of a group's four groups of 32 (l0 h0 of its first block of 64, l1 h1 of its second) to their four
// int32 sums: the halves of 64 bits added, then the pairs of 32, then the int16 pairs widened
// @ts-ignore: decorator
@inline function sums16(l0: v128, h0: v128, l1: v128, h1: v128): v128 {
  const u = i16x8.add(v128.shuffle<i64>(l0, h0, 0, 2), v128.shuffle<i64>(l0, h0, 1, 3));  // lanes 0..3 group 0, 4..7 group 1
  const t = i16x8.add(v128.shuffle<i64>(l1, h1, 0, 2), v128.shuffle<i64>(l1, h1, 1, 3));  // groups 2 and 3
  return i32x4.extadd_pairwise_i16x8_s(i16x8.add(v128.shuffle<i32>(u, t, 0, 2, 4, 6), v128.shuffle<i32>(u, t, 1, 3, 5, 7)));
}
// @ts-ignore: decorator
@inline function plain128<P>(facc: v128, w: usize, x: usize, sums: usize, xs: usize, at: usize, d: v128, mask: v128): v128 {
  const v = v128.load(w), u = blockBytes<P>(w, v, 1);
  const x0 = v128.load(x), x1 = v128.load(x, 16), x2 = v128.load(x, 32), x3 = v128.load(x, 48);
  const c0 = codesOf<P>(v, 0, 0, mask), c1 = codesOf<P>(v, 0, 1, mask), c2 = codesOf<P>(v, 0, 2, mask), c3 = codesOf<P>(v, 0, 3, mask);
  const y0 = v128.load(x, 64), y1 = v128.load(x, 80), y2 = v128.load(x, 96), y3 = v128.load(x, 112);
  const e0 = codesOf<P>(u, 1, 0, mask), e1 = codesOf<P>(u, 1, 1, mask), e2 = codesOf<P>(u, 1, 2, mask), e3 = codesOf<P>(u, 1, 3, mask);
  const integers = sums16(low16(x0, x1, x2, x3, c0, c1, c2, c3), high16(x0, x1, x2, x3, c0, c1, c2, c3),
                          low16(y0, y1, y2, y3, e0, e1, e2, e3), high16(y0, y1, y2, y3, e0, e1, e2, e3));
  const whole = i32x4.add(integers, v128.load(sums + at));
  return f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(whole), v128.load(xs + at)), d));
}
// (T533: the float16 scales as matmul_t2r reads them)
// @ts-ignore: decorator
@inline function signsRows<P>(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, code: i32): void {
  const groups = n >> 7, groups4 = groups & ~3, mask = i8x16.splat(<i8>code), bytes = groupBytes<P>();
  const sums = xs + (<usize>(n >> 5) << 2);
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * rowBytes<P>(n);
    const srow = ws + ((<usize>i * <usize>groups) << 1);
    let facc = f32x4.splat(0);
    let g = 0;
    for (; g < groups4; g += 4) {
      const d = halves4(srow + (<usize>g << 1)), w = row + <usize>g * bytes, x = xq + (<usize>g << 7), at = <usize>g << 4;
      facc = plain128<P>(facc, w, x, sums, xs, at, v128.shuffle<i32>(d, d, 0, 0, 0, 0), mask);
      facc = plain128<P>(facc, w + bytes, x + 128, sums, xs, at + 16, v128.shuffle<i32>(d, d, 1, 1, 1, 1), mask);
      facc = plain128<P>(facc, w + 2 * bytes, x + 256, sums, xs, at + 32, v128.shuffle<i32>(d, d, 2, 2, 2, 2), mask);
      facc = plain128<P>(facc, w + 3 * bytes, x + 384, sums, xs, at + 48, v128.shuffle<i32>(d, d, 3, 3, 3, 3), mask);
    }
    for (; g < groups; g++) {
      facc = plain128<P>(facc, row + <usize>g * bytes, xq + (<usize>g << 7), sums, xs, <usize>g << 4, f32x4.splat(half(srow + (<usize>g << 1))), mask);
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}
// three: 3 (ternary.ts says why it is an argument)
export function matmul_t2(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  signsRows<Ternary>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
// T634: two: 2 (binary.ts)
export function matmul_b1(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, two: i32): void {
  signsRows<Binary>(xout, xq, xs, wq, ws, n, r0, r1, two);
}

// T98: int6 weights (24 bytes a group, six.ts) with one float32 scale per group: matmul_q8 on the widened groups
export function matmul_q6(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>ng * 24;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    let facc = f32x4.splat(0);
    for (let g = 0; g < ng; g++) {
      const p = row + <usize>g * 24, o = <usize>(g * GS);
      const low = v128.load(p), t = sixTops(p);
      const a0 = sixFirst(low, t), b0 = v128.load(xq + o);
      const a1 = sixSecond(low, t), b1 = v128.load(xq + o + 16);
      const acc = dot32(a0, b0, a1, b1);
      const s = load<f32>(srow + (<usize>g << 2)) * load<f32>(xs + (<usize>g << 2));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(acc), f32x4.splat(s)));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// y[j] += sum over c of a[c] * rows[c][j]: the outlier channels of the classifier's input, multiplied by their own
// columns of the classifier in float32 (OUTLIER_CHANNELS in engine/checkpoint.py). rows is (count, n) row-major.
export function add_columns(y: usize, rows: usize, a: usize, count: i32, n: i32): void {
  const n4 = n & ~3;
  let j = 0;
  for (; j < n4; j += 4) {
    const o = <usize>j << 2;
    let acc = v128.load(y + o);
    for (let c = 0; c < count; c++) {
      const row = rows + ((<usize>c * <usize>n) << 2);
      acc = f32x4.add(acc, f32x4.mul(f32x4.splat(load<f32>(a + (<usize>c << 2))), v128.load(row + o)));
    }
    v128.store(y + o, acc);
  }
  for (; j < n; j++) {
    const o = <usize>j << 2;
    let v = load<f32>(y + o);
    for (let c = 0; c < count; c++) v += load<f32>(a + (<usize>c << 2)) * load<f32>(rows + ((<usize>c * <usize>n) << 2) + o);
    store<f32>(y + o, v);
  }
}

// ---- T648.3: Firefox Translations' int8 affine, as fxtranslate multiplies it (intgemm's int8shiftAlphaAll; the form
// read from fxtranslate's scalar path, docs/notes/t648-firefox-translations-engines-2026-11-05.md; no line copied):
// out (m, n) = x (m, k) times the int8 rows w (n, k: Marian's order of an intgemm8 matrix) plus the bias, each
// activation quantized with the matrix's fixed alpha, not a scale of its own.
// fxtranslate's activation is clamp(round_even(x · alpha), −127, 127) + 127 (0..254). Here it is s = the same less its
// 127 (−127..127, into xq: m k bytes of the caller's), and the 127 comes back as an integer: shifts[j] = 127 Σ_k w[j, k]
// (the caller's), so Σ_k s w + shifts[j] is fxtranslate's Σ_k (s + 127) w, the same integer. It is exact in int32:
// |s w| ≤ 127 · 128, two of them added in int16 are within ±32,512, and the whole is within 254 · 128 · k, below 2**31
// for k up to 66,052 (the caller keeps to that). It is turned into a float32 once and multiplied by unquant
// (1 / (alpha · the weights' quantMult)), then the bias added (the one with the shift taken off: b − 127 · unquant · Σ w),
// each a float32 operation: engine/translation.py's Affine, and so fxtranslate's numbers, to the bit. (A NaN of x is
// an activation of 0 here and a NaN there: a NaN is broken on every path.)
// Not relaxed: relaxed_dot_i8x16_i7x16's second factor is 7 bits, which neither the activations nor the weights are.
// The rows of x are quantized once, then each row of w meets every row of x while it is in the cache.
export function affine_q8(out: usize, x: usize, xq: usize, m: i32, k: i32, w: usize, n: i32, alpha: f32, shifts: usize, unquant: f32, bias: usize): void {
  const a = f32x4.splat(alpha), floor = i8x16.splat(<i8>-127);
  const k16 = k & ~15, k32 = k & ~31;
  for (let i = 0; i < m; i++) {
    const from = x + ((<usize>i * <usize>k) << 2), to = xq + <usize>i * <usize>k;
    let c = 0;
    for (; c < k16; c += 16) {
      const p = from + (<usize>c << 2);
      const q0 = i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v128.load(p), a)));
      const q1 = i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v128.load(p, 16), a)));
      const q2 = i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v128.load(p, 32), a)));
      const q3 = i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v128.load(p, 48), a)));
      // (the narrowings saturate to −128..127: then −127 below)
      v128.store(to + <usize>c, i8x16.max_s(i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(q0, q1), i16x8.narrow_i32x4_s(q2, q3)), floor));
    }
    for (; c < k; c++) {
      const q = i32x4.extract_lane(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.splat(load<f32>(from + (<usize>c << 2)) * alpha))), 0);
      store<i8>(to + <usize>c, <i8>(q < -127 ? -127 : q > 127 ? 127 : q));
    }
  }
  for (let j = 0; j < n; j++) {
    const row = w + <usize>j * <usize>k;
    const shift = load<i32>(shifts + (<usize>j << 2)), b = load<f32>(bias + (<usize>j << 2));
    for (let i = 0; i < m; i++) {
      const xr = xq + <usize>i * <usize>k;
      let acc = i32x4.splat(0);
      let c = 0;
      for (; c < k32; c += 32) {
        acc = i32x4.add(acc, dot32(v128.load(row + <usize>c), v128.load(xr + <usize>c), v128.load(row + <usize>c, 16), v128.load(xr + <usize>c, 16)));
      }
      if (c < k16) {
        const r = v128.load(row + <usize>c), s = v128.load(xr + <usize>c);
        acc = i32x4.add(acc, i32x4.extadd_pairwise_i16x8_s(i16x8.add(i16x8.extmul_low_i8x16_s(r, s), i16x8.extmul_high_i8x16_s(r, s))));
        c += 16;
      }
      let sum = laneSum(acc) + shift;
      for (; c < k; c++) sum += <i32>load<i8>(row + <usize>c) * <i32>load<i8>(xr + <usize>c);
      store<f32>(out + ((<usize>i * <usize>n + <usize>j) << 2), unquant * <f32>sum + b);
    }
  }
}
