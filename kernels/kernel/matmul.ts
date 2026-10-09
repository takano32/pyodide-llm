// kernel/matmul.ts (T356): the matrix products without relaxed SIMD (those with it are kernel_relaxed.ts), W (d, n)
// times x (n) for rows r0 to r1, and what they take beside the weights: the sums of the weights' groups (the
// corrections of the relaxed products), the activations laid out for the ternary products (interleave), and the
// outliers' columns of the classifier (add_columns).

import { sixFirst, sixSecond, sixTops } from "../six";
import { codes, fourSums } from "../ternary";
import { GS, hsum } from "./math";

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
    const sum = i32x4.extract_lane(quads, 0) + i32x4.extract_lane(quads, 1) + i32x4.extract_lane(quads, 2) + i32x4.extract_lane(quads, 3);
    store<i32>(out + (<usize>g << 2), -64 * sum);
  }
}

// T123: the same for int8 weights: the sums of groups of 32 int8 values (w), times -64 (T197), the corrections of
// matmul_q8r. forward.js made them one value at a time in JavaScript (7B: 266 s of its construct)
export function int8_sums(out: usize, w: usize, groups: i32): void {
  for (let g = 0; g < groups; g++) {
    const p = w + (<usize>g << 5);
    const pairs = i16x8.add(i16x8.extadd_pairwise_i8x16_s(v128.load(p)), i16x8.extadd_pairwise_i8x16_s(v128.load(p + 16)));
    const quads = i32x4.extadd_pairwise_i16x8_s(pairs);
    const sum = i32x4.extract_lane(quads, 0) + i32x4.extract_lane(quads, 1) + i32x4.extract_lane(quads, 2) + i32x4.extract_lane(quads, 3);
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
  return i32x4.extract_lane(quads, 0) + i32x4.extract_lane(quads, 1) + i32x4.extract_lane(quads, 2) + i32x4.extract_lane(quads, 3);
}
// Plane r of a block of 64: byte r of every dword of the four 16-byte vectors (activations 4 c + r for c = 0..15), a byte of
// every dword shifted down and masked to 0..255, then narrowed from 32 bits to 16 and to 8 (the values pass the narrowing
// unchanged): the four vectors' four dwords each are the sixteen bytes in order.
// With no shuffle in it, on purpose. This was a transpose of byte, dword and qword shuffles (a byte shuffle of each
// vector, then dwords of two, then qwords of two), and JavaScriptCore's optimizing tier on x86-64 (seen in WebKit on Linux, on
// AMD EPYCs and an Intel Xeon alike; Safari on an Intel Mac has the same compiler and was not tried), which a function reaches after about 2000
// calls, folded those three wrongly: three quarters of the bytes
// came out wrong, and a ternary model wrote nonsense (T230's review; tests/kernels-in-browser.mjs finds it). Every piece of
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

// matmul_t2r (kernel_relaxed.ts) without relaxed SIMD: the same weights, activations and sums, the same exact
// integers a group and the same float32 operations in the same order, so the same numbers to the bit. A plane's
// products are widened to int16 (its low eight bytes are of the first group of 32 activations, its high eight of the
// second) and added over the four planes there (4 x 2 x 128 a lane at most), then widened once.
// @ts-ignore: decorator
@inline function plain64(w: usize, x: usize, three: v128): v128 {
  const v = v128.load(w);
  const x0 = v128.load(x), x1 = v128.load(x, 16), x2 = v128.load(x, 32), x3 = v128.load(x, 48);
  const c0 = codes(v, 0, three), c1 = codes(v, 1, three), c2 = codes(v, 2, three), c3 = codes(v, 3, three);
  const low = i16x8.add(i16x8.add(i16x8.extmul_low_i8x16_s(x0, c0), i16x8.extmul_low_i8x16_s(x1, c1)),
                        i16x8.add(i16x8.extmul_low_i8x16_s(x2, c2), i16x8.extmul_low_i8x16_s(x3, c3)));
  const high = i16x8.add(i16x8.add(i16x8.extmul_high_i8x16_s(x0, c0), i16x8.extmul_high_i8x16_s(x1, c1)),
                         i16x8.add(i16x8.extmul_high_i8x16_s(x2, c2), i16x8.extmul_high_i8x16_s(x3, c3)));
  // lanes 0 and 1 the first group's, 2 and 3 the second's, as the relaxed dot product leaves them
  const first = i32x4.extadd_pairwise_i16x8_s(low), second = i32x4.extadd_pairwise_i16x8_s(high);
  return i32x4.add(v128.shuffle<i32>(first, second, 0, 1, 4, 5), v128.shuffle<i32>(first, second, 2, 3, 6, 7));
}
export function matmul_t2(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  const groups = n >> 7, mask = i8x16.splat(<i8>three);
  const sums = xs + (<usize>(n >> 5) << 2);
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 2);
    const srow = ws + ((<usize>i * <usize>groups) << 2);
    let facc = f32x4.splat(0);
    for (let g = 0; g < groups; g++) {
      const w = row + (<usize>g << 5), x = xq + (<usize>g << 7), at = <usize>g << 4;
      const whole = i32x4.add(fourSums(plain64(w, x, mask), plain64(w + 16, x + 64, mask)), v128.load(sums + at));
      facc = f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(whole), v128.load(xs + at)), v128.load32_splat(srow + (<usize>g << 2))));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
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
// columns of the classifier in float32 (OUTLIER_CHANNELS in llama2_numpy.py). rows is (count, n) row-major.
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
