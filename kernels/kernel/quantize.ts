// kernel/quantize.ts (T356): numbers from one form to another. The converter's kernels: the stored forms of a
// checkpoint widened to float32 (bfloat16, GGUF's Q8_0, Prism ML's PQ2_0 and PTQ1_0) and float32 quantized to what
// the page keeps (int8, six bits, ternary), each the same bytes as the converter's NumPy. quantize_x is the forward's
// too: the activations of every int8 matrix product go through it.

import { GS } from "./math";

// the largest |value| of a group of 32, lane by lane as the scalar comparisons were (see quantize_x)
// @ts-ignore: decorator
@inline function groupMax(v0: v128, v1: v128, v2: v128, v3: v128, v4: v128, v5: v128, v6: v128, v7: v128): f32 {
  const m = f32x4.max(f32x4.max(f32x4.max(f32x4.abs(v0), f32x4.abs(v1)), f32x4.max(f32x4.abs(v2), f32x4.abs(v3))),
                      f32x4.max(f32x4.max(f32x4.abs(v4), f32x4.abs(v5)), f32x4.max(f32x4.abs(v6), f32x4.abs(v7))));
  let amax = f32x4.extract_lane(m, 0);
  const m1 = f32x4.extract_lane(m, 1), m2 = f32x4.extract_lane(m, 2), m3 = f32x4.extract_lane(m, 3);
  if (m1 > amax) amax = m1;
  if (m2 > amax) amax = m2;
  if (m3 > amax) amax = m3;
  return amax;
}

// T123: n bfloat16 values (raw) to float32 (out): a bfloat16 is the upper half of a float32, so each is its 16 bits
// shifted up, exactly as the converter's NumPy bfloat16() does. 8 at a time, the last few one by one.
export function widen_bf16(out: usize, raw: usize, n: i32): void {
  let i = 0;
  for (; i + 8 <= n; i += 8) {
    const h = v128.load(raw + (<usize>i << 1));
    v128.store(out + (<usize>i << 2), i32x4.shl(i32x4.extend_low_i16x8_u(h), 16));
    v128.store(out + (<usize>i << 2), i32x4.shl(i32x4.extend_high_i16x8_u(h), 16), 16);
  }
  for (; i < n; i++) store<u32>(out + (<usize>i << 2), <u32>load<u16>(raw + (<usize>i << 1)) << 16);
}

// a float16 (its 16 bits) as float32, exactly: every float16 is a float32. The bits as NumPy's software conversion
// makes them: infinities and NaNs keep their payload (shifted up), subnormals become normal float32.
// @ts-ignore: decorator
@inline function halfToFloat(h: u32): f32 {
  const sign = (h & 0x8000) << 16, exp = h & 0x7c00, sig = h & 0x03ff;
  if (exp == 0x7c00) return reinterpret<f32>(sign | 0x7f800000 | (sig << 13));
  if (exp == 0) {
    const tiny = <f32>sig * <f32>5.9604644775390625e-8; // sig * 2^-24: exact, a power of two times an integer < 1024
    return sign ? -tiny : tiny;
  }
  return reinterpret<f32>(sign | (((h & 0x7fff) << 13) + (112 << 23))); // the exponent bias 15 -> 127
}

// T136: n blocks of GGUF's Q8_0 (raw, 34 bytes each: a float16 scale and 32 int8) to 32 n float32 (out), each int8
// times its block's scale in float32: exactly the converter's NumPy q8_0() (one rounding, the product).
export function widen_q8_0(out: usize, raw: usize, n: i32): void {
  for (let b = 0; b < n; b++) {
    const block = raw + <usize>b * 34, to = out + (<usize>b << 7);
    const scale = f32x4.splat(halfToFloat(<u32>load<u16>(block)));
    const lo = v128.load(block, 2), hi = v128.load(block, 18);
    const l0 = i16x8.extend_low_i8x16_s(lo), l1 = i16x8.extend_high_i8x16_s(lo);
    const h0 = i16x8.extend_low_i8x16_s(hi), h1 = i16x8.extend_high_i8x16_s(hi);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(l0)), scale));
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(l0)), scale), 16);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(l1)), scale), 32);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(l1)), scale), 48);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(h0)), scale), 64);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(h0)), scale), 80);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(h1)), scale), 96);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(h1)), scale), 112);
  }
}

// T273: n blocks of Prism ML's PQ2_0 (raw, 34 bytes each: a float16 scale d and 32 bytes of two bits a value, the
// first value in the lowest bits of the first byte) to 128 n float32 (out), (code - 1) * d each: exactly the
// converter's NumPy pq2_0(). A byte's four codes are its products with 64, 16, 4 and 1, shifted down by six.
export function widen_pq2_0(out: usize, raw: usize, n: i32): void {
  const places = i32x4(64, 16, 4, 1), three = i32x4.splat(3), one = i32x4.splat(1);
  for (let b = 0; b < n; b++) {
    const block = raw + <usize>b * 34, to = out + (<usize>b << 9);
    const scale = f32x4.splat(halfToFloat(<u32>load<u16>(block)));
    for (let i = 0; i < 32; i++) {
      const codes = v128.and(i32x4.shr_u(i32x4.mul(i32x4.splat(<i32>load<u8>(block + 2 + <usize>i)), places), 6), three);
      v128.store(to + (<usize>i << 4), f32x4.mul(f32x4.convert_i32x4_s(i32x4.sub(codes, one)), scale));
    }
  }
}

// eight base 3 digits (0, 1 or 2, in 16-bit lanes) as eight float32, (digit - 1) * d each
// @ts-ignore: decorator
@inline function storeDigits(to: usize, digits: v128, scale: v128, one: v128): void {
  v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.sub(i32x4.extend_low_i16x8_u(digits), one)), scale));
  v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.sub(i32x4.extend_high_i16x8_u(digits), one)), scale), 16);
}

// T273: n blocks of Prism ML's PTQ1_0 (raw, 28 bytes each: 24 bytes of five base 3 digits, 2 bytes of four, and a
// float16 scale d) to 128 n float32 (out), (digit - 1) * d each: exactly the converter's NumPy ptq1_0() and base3().
// A digit is the high byte of three times the byte, and the low byte goes on to the next digit. The values are
// digit by digit, not byte by byte: 16 n + m for digit n of byte m of the first 16 bytes, 80 + 8 n + m of the next
// 8, 120 + 2 n + m of the 2. So a digit of 8 bytes at once is 8 values next to each other.
export function widen_ptq1_0(out: usize, raw: usize, n: i32): void {
  const low = i16x8.splat(255), thrice = i16x8.splat(3), one = i32x4.splat(1);
  for (let b = 0; b < n; b++) {
    const block = raw + <usize>b * 28, to = out + (<usize>b << 9);
    const d = halfToFloat(<u32>load<u16>(block, 26)), scale = f32x4.splat(d);
    const bytes = v128.load(block);
    let first = i16x8.extend_low_i8x16_u(bytes), second = i16x8.extend_high_i8x16_u(bytes);
    let third = i16x8.extend_low_i8x16_u(v128.load64_zero(block, 16));
    for (let digit = 0; digit < 5; digit++) {
      first = i16x8.mul(first, thrice);
      second = i16x8.mul(second, thrice);
      third = i16x8.mul(third, thrice);
      storeDigits(to + (<usize>digit << 6), i16x8.shr_u(first, 8), scale, one);
      storeDigits(to + (<usize>digit << 6) + 32, i16x8.shr_u(second, 8), scale, one);
      storeDigits(to + 320 + (<usize>digit << 5), i16x8.shr_u(third, 8), scale, one);
      first = v128.and(first, low);
      second = v128.and(second, low);
      third = v128.and(third, low);
    }
    for (let m = 0; m < 2; m++) {
      let left = <u32>load<u8>(block + 24 + <usize>m);
      for (let digit = 0; digit < 4; digit++) {
        left *= 3;
        store<f32>(to + (<usize>(120 + 2 * digit + m) << 2), <f32>(<i32>(left >> 8) - 1) * d);
        left &= 255;
      }
    }
  }
}

// bias = 0: signed int8 in [-127,127];  bias = 64: 7-bit unsigned, real value = (q - 64) * scale (for kernel_relaxed.ts)
export function quantize_x(xq: usize, xs: usize, x: usize, n: i32, bias: i32): void {
  // SIMD, 32 values (one group) at a time. Every step is the scalar one lane by lane (abs, max, the division, the
  // product, round half to even), so the int8 and the scales are the same to the bit as before, and as NumPy's
  // llama2_convert.quantize(), which the converter now hands to this (T89). Only a NaN in the input tells them
  // apart (SIMD max keeps it, the scalar compare skipped it), and a NaN in a checkpoint is broken on every path.
  const qmax: f32 = bias == 0 ? 127.0 : 63.0;
  const offset = i32x4.splat(bias);
  for (let g = 0; g < n; g += GS) {
    const p = x + (<usize>g << 2);
    const v0 = v128.load(p), v1 = v128.load(p, 16), v2 = v128.load(p, 32), v3 = v128.load(p, 48);
    const v4 = v128.load(p, 64), v5 = v128.load(p, 80), v6 = v128.load(p, 96), v7 = v128.load(p, 112);
    const amax = groupMax(v0, v1, v2, v3, v4, v5, v6, v7);
    const scale: f32 = amax / qmax;
    store<f32>(xs + (<usize>(g / GS) << 2), scale);
    const inv = f32x4.splat(scale > 0 ? <f32>1.0 / scale : 0);
    const q0 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v0, inv))), offset);
    const q1 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v1, inv))), offset);
    const q2 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v2, inv))), offset);
    const q3 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v3, inv))), offset);
    const q4 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v4, inv))), offset);
    const q5 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v5, inv))), offset);
    const q6 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v6, inv))), offset);
    const q7 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v7, inv))), offset);
    const out = xq + <usize>g;
    v128.store(out, i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(q0, q1), i16x8.narrow_i32x4_s(q2, q3)));
    v128.store(out, i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(q4, q5), i16x8.narrow_i32x4_s(q6, q7)), 16);
  }
}

// T98: the converter's six bits (llama2_numpy.quantize6 and pack6 in one pass, the same bytes): per group of 32,
// scale = the largest |value| / 31, v = round(value / scale) in -32..31, stored as the int8 4 v packed into 24 bytes
// (six.ts reads them back) and the scale as scale / 4 (xs).
// round(value / scale), clipped to six bits (-32..31), as NumPy's rint and clip
// @ts-ignore: decorator
@inline function six(v: v128, inv: v128): v128 {
  return i32x4.max_s(i32x4.min_s(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v, inv))), i32x4.splat(31)), i32x4.splat(-32));
}
export function quantize6_x(out: usize, xs: usize, x: usize, n: i32): void {
  const low6 = i8x16.splat(63);
  for (let g = 0; g < n; g += GS) {
    const p = x + (<usize>g << 2);
    const v0 = v128.load(p), v1 = v128.load(p, 16), v2 = v128.load(p, 32), v3 = v128.load(p, 48);
    const v4 = v128.load(p, 64), v5 = v128.load(p, 80), v6 = v128.load(p, 96), v7 = v128.load(p, 112);
    const scale: f32 = groupMax(v0, v1, v2, v3, v4, v5, v6, v7) / <f32>31.0;
    store<f32>(xs + (<usize>(g / GS) << 2), scale * <f32>0.25);
    const inv = f32x4.splat(scale > 0 ? <f32>1.0 / scale : 0);
    // the six bits of values 0..15 and 16..31
    const first = v128.and(i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(six(v0, inv), six(v1, inv)),
                                                i16x8.narrow_i32x4_s(six(v2, inv), six(v3, inv))), low6);
    const second = v128.and(i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(six(v4, inv), six(v5, inv)),
                                                 i16x8.narrow_i32x4_s(six(v6, inv), six(v7, inv))), low6);
    const at = out + <usize>(g / GS) * 24;
    v128.store(at, v128.or(v128.and(first, i8x16.splat(15)), i8x16.shl(second, 4)));
    // top two bits: lane k of a holds those of values k and k + 16; byte k takes lanes k and k + 8 of a
    const a = v128.or(i8x16.shr_u(first, 4), i8x16.shl(i8x16.shr_u(second, 4), 4));
    const b = i8x16.shuffle(a, a, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15);
    v128.store64_lane(at + 16, v128.or(a, i8x16.shl(b, 2)), 0);
  }
}

// T230: the converter's ternary() (llama2_numpy.py) in one pass, the same bytes: n float32 values (x), whole groups of
// 128, to two bits a value (out, 32 bytes a group: the code sign + 1 of value j in byte j >> 2 at bits 2 (j & 3)) and
// the largest |value| of every group (xs). Returns 1 where a value is neither 0 nor plus or minus its group's largest
// (the weights are not ternary, or not a number), else 0.
export function ternary_x(out: usize, xs: usize, x: usize, n: i32): i32 {
  let bad = i32x4.splat(0);
  const one = i32x4.splat(1), zero = f32x4.splat(0);
  for (let g = 0; g < n; g += 128) {
    const p = x + (<usize>g << 2);
    let largest = f32x4.splat(0);
    for (let j: usize = 0; j < 512; j += 16) largest = f32x4.max(largest, f32x4.abs(v128.load(p + j)));
    const scale = f32x4.splat(max<f32>(max<f32>(f32x4.extract_lane(largest, 0), f32x4.extract_lane(largest, 1)),
                                      max<f32>(f32x4.extract_lane(largest, 2), f32x4.extract_lane(largest, 3))));
    v128.store32_lane(xs + (<usize>(g >> 7) << 2), scale, 0);
    const to = out + <usize>(g >> 2);
    for (let j: usize = 0; j < 8; j++) {  // sixteen values, four bytes
      const at = p + (j << 6);
      const v0 = v128.load(at), v1 = v128.load(at, 16), v2 = v128.load(at, 32), v3 = v128.load(at, 48);
      // 1 + (v > 0) - (v < 0): a comparison is -1 where it holds
      const c0 = i32x4.add(i32x4.sub(one, f32x4.gt(v0, zero)), f32x4.lt(v0, zero));
      const c1 = i32x4.add(i32x4.sub(one, f32x4.gt(v1, zero)), f32x4.lt(v1, zero));
      const c2 = i32x4.add(i32x4.sub(one, f32x4.gt(v2, zero)), f32x4.lt(v2, zero));
      const c3 = i32x4.add(i32x4.sub(one, f32x4.gt(v3, zero)), f32x4.lt(v3, zero));
      // a lane of four codes, a byte each, to one byte: the codes at bits 0, 2, 4 and 6
      const four = i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(c0, c1), i16x8.narrow_i32x4_s(c2, c3));
      const packed = v128.or(v128.or(four, i32x4.shr_u(four, 6)), v128.or(i32x4.shr_u(four, 12), i32x4.shr_u(four, 18)));
      const bytes = v128.and(packed, i32x4.splat(255));
      v128.store32_lane(to + (j << 2), i8x16.narrow_i16x8_u(i16x8.narrow_i32x4_s(bytes, bytes), i16x8.splat(0)), 0);
      // every value 0 or of the scale's size
      const ok0 = v128.or(f32x4.eq(f32x4.abs(v0), scale), f32x4.eq(v0, zero)), ok1 = v128.or(f32x4.eq(f32x4.abs(v1), scale), f32x4.eq(v1, zero));
      const ok2 = v128.or(f32x4.eq(f32x4.abs(v2), scale), f32x4.eq(v2, zero)), ok3 = v128.or(f32x4.eq(f32x4.abs(v3), scale), f32x4.eq(v3, zero));
      bad = v128.or(bad, v128.not(v128.and(v128.and(ok0, ok1), v128.and(ok2, ok3))));
    }
  }
  return v128.any_true(bad) ? 1 : 0;
}
