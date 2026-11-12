// tests/ternary-plain-forms.ts (T275): forms of the ternary product without relaxed SIMD (what Safari runs: it has no
// relaxed SIMD), set against kernels/'s matmul_t2. tests/ternary-browser-bench.ts compiles this file and times each
// form beside matmul_t2 in a browser's engine (WebKit's JavaScriptCore above all) and in Node's V8, and holds every
// form's numbers to matmul_t2's to the bit. Not a kernel of the page: nothing here is shipped.
//
// The weights, the activations and the float16 scales are matmul_t2's (kernels/ternary.ts, kernel/matmul.ts's
// interleave): every form makes the same exact integers a group of 32 activations and the same float32 operations in
// the same order, so the same numbers to the bit. What they change is how the integers are added up.
//
//   t2c  a copy of matmul_t2 (kernel/matmul.ts's signsRows<Ternary>) in this module: the noise of two copies of one form
//   ch   the four planes' products added as a chain, acc = acc + product (arm64's smlal and sadalp fold such an add
//        into the multiply or the widening; the copy adds them as a tree)
//   r16  ch, and a group's four sums of 32 added up in int16 (at most 8192 a group of 32: 4 planes x 8 lanes x 256)
//        with 64-bit lane shuffles, one widening to int32 a group in place of four
//   rt   r16 with the copy's tree of adds (what the int16 sums alone are worth)
//   r16s r16 whose last plane of codes is a shift of bytes (i8x16.shr_u) with no mask
//   dt   the planes' products as i32x4.dot_i16x8_s of the bytes widened to int16 (no int16 sums to widen)
//   w2   r16 on two rows at once, the activations loaded once for both
import { codes } from "../kernels/ternary";
import { fourSums } from "../kernels/ternary";
import { half, halves4 } from "../kernels/kernel/halves";
import { hsum } from "../kernels/kernel/math";

// ---- t2c: kernel/matmul.ts's plain64 and plain128 for Ternary, as they are
// @ts-ignore: decorator
@inline function copy64(v: v128, x: usize, mask: v128): v128 {
  const x0 = v128.load(x), x1 = v128.load(x, 16), x2 = v128.load(x, 32), x3 = v128.load(x, 48);
  const c0 = codes(v, 0, mask), c1 = codes(v, 1, mask), c2 = codes(v, 2, mask), c3 = codes(v, 3, mask);
  const low = i16x8.add(i16x8.add(i16x8.extmul_low_i8x16_s(x0, c0), i16x8.extmul_low_i8x16_s(x1, c1)),
                        i16x8.add(i16x8.extmul_low_i8x16_s(x2, c2), i16x8.extmul_low_i8x16_s(x3, c3)));
  const high = i16x8.add(i16x8.add(i16x8.extmul_high_i8x16_s(x0, c0), i16x8.extmul_high_i8x16_s(x1, c1)),
                         i16x8.add(i16x8.extmul_high_i8x16_s(x2, c2), i16x8.extmul_high_i8x16_s(x3, c3)));
  const first = i32x4.extadd_pairwise_i16x8_s(low), second = i32x4.extadd_pairwise_i16x8_s(high);
  return i32x4.add(v128.shuffle<i32>(first, second, 0, 1, 4, 5), v128.shuffle<i32>(first, second, 2, 3, 6, 7));
}
// @ts-ignore: decorator
@inline function copyWhole(w: usize, x: usize, mask: v128): v128 {
  return fourSums(copy64(v128.load(w), x, mask), copy64(v128.load(w, 16), x + 64, mask));
}

// ---- ch: the products added as a chain
// @ts-ignore: decorator
@inline function chain64(v: v128, x: usize, mask: v128): v128 {
  const x0 = v128.load(x), x1 = v128.load(x, 16), x2 = v128.load(x, 32), x3 = v128.load(x, 48);
  const c0 = codes(v, 0, mask), c1 = codes(v, 1, mask), c2 = codes(v, 2, mask), c3 = codes(v, 3, mask);
  let low = i16x8.extmul_low_i8x16_s(x0, c0);
  low = i16x8.add(low, i16x8.extmul_low_i8x16_s(x1, c1));
  low = i16x8.add(low, i16x8.extmul_low_i8x16_s(x2, c2));
  low = i16x8.add(low, i16x8.extmul_low_i8x16_s(x3, c3));
  let high = i16x8.extmul_high_i8x16_s(x0, c0);
  high = i16x8.add(high, i16x8.extmul_high_i8x16_s(x1, c1));
  high = i16x8.add(high, i16x8.extmul_high_i8x16_s(x2, c2));
  high = i16x8.add(high, i16x8.extmul_high_i8x16_s(x3, c3));
  const first = i32x4.extadd_pairwise_i16x8_s(low), second = i32x4.extadd_pairwise_i16x8_s(high);
  return i32x4.add(v128.shuffle<i32>(first, second, 0, 1, 4, 5), v128.shuffle<i32>(first, second, 2, 3, 6, 7));
}
// @ts-ignore: decorator
@inline function chainWhole(w: usize, x: usize, mask: v128): v128 {
  return fourSums(chain64(v128.load(w), x, mask), chain64(v128.load(w, 16), x + 64, mask));
}

// ---- r16: the chain's int16 sums of the four groups of 32 added up in int16
// a plane's products added as a chain: the low halves (the block's first group of 32) and the high (its second)
// @ts-ignore: decorator
@inline function low4(x0: v128, x1: v128, x2: v128, x3: v128, c0: v128, c1: v128, c2: v128, c3: v128): v128 {
  let low = i16x8.extmul_low_i8x16_s(x0, c0);
  low = i16x8.add(low, i16x8.extmul_low_i8x16_s(x1, c1));
  low = i16x8.add(low, i16x8.extmul_low_i8x16_s(x2, c2));
  return i16x8.add(low, i16x8.extmul_low_i8x16_s(x3, c3));
}
// @ts-ignore: decorator
@inline function high4(x0: v128, x1: v128, x2: v128, x3: v128, c0: v128, c1: v128, c2: v128, c3: v128): v128 {
  let high = i16x8.extmul_high_i8x16_s(x0, c0);
  high = i16x8.add(high, i16x8.extmul_high_i8x16_s(x1, c1));
  high = i16x8.add(high, i16x8.extmul_high_i8x16_s(x2, c2));
  return i16x8.add(high, i16x8.extmul_high_i8x16_s(x3, c3));
}
// two blocks' int16 sums (four groups of 32: l0 h0 l1 h1) to the four groups' int32 sums
// @ts-ignore: decorator
@inline function sums16(l0: v128, h0: v128, l1: v128, h1: v128): v128 {
  const u = i16x8.add(v128.shuffle<i64>(l0, h0, 0, 2), v128.shuffle<i64>(l0, h0, 1, 3));  // lanes 0..3 group 0, 4..7 group 1
  const t = i16x8.add(v128.shuffle<i64>(l1, h1, 0, 2), v128.shuffle<i64>(l1, h1, 1, 3));  // groups 2 and 3
  return i32x4.extadd_pairwise_i16x8_s(i16x8.add(v128.shuffle<i32>(u, t, 0, 2, 4, 6), v128.shuffle<i32>(u, t, 1, 3, 5, 7)));
}
// @ts-ignore: decorator
@inline function r16Whole(w: usize, x: usize, mask: v128): v128 {
  const v0 = v128.load(w), v1 = v128.load(w, 16);
  const x0 = v128.load(x), x1 = v128.load(x, 16), x2 = v128.load(x, 32), x3 = v128.load(x, 48);
  const c0 = codes(v0, 0, mask), c1 = codes(v0, 1, mask), c2 = codes(v0, 2, mask), c3 = codes(v0, 3, mask);
  const y0 = v128.load(x, 64), y1 = v128.load(x, 80), y2 = v128.load(x, 96), y3 = v128.load(x, 112);
  const d0 = codes(v1, 0, mask), d1 = codes(v1, 1, mask), d2 = codes(v1, 2, mask), d3 = codes(v1, 3, mask);
  return sums16(low4(x0, x1, x2, x3, c0, c1, c2, c3), high4(x0, x1, x2, x3, c0, c1, c2, c3),
                low4(y0, y1, y2, y3, d0, d1, d2, d3), high4(y0, y1, y2, y3, d0, d1, d2, d3));
}

// ---- rt: r16's int16 sums with the copy's tree of adds in place of the chain
// @ts-ignore: decorator
@inline function rtWhole(w: usize, x: usize, mask: v128): v128 {
  const v0 = v128.load(w), v1 = v128.load(w, 16);
  const x0 = v128.load(x), x1 = v128.load(x, 16), x2 = v128.load(x, 32), x3 = v128.load(x, 48);
  const c0 = codes(v0, 0, mask), c1 = codes(v0, 1, mask), c2 = codes(v0, 2, mask), c3 = codes(v0, 3, mask);
  const y0 = v128.load(x, 64), y1 = v128.load(x, 80), y2 = v128.load(x, 96), y3 = v128.load(x, 112);
  const d0 = codes(v1, 0, mask), d1 = codes(v1, 1, mask), d2 = codes(v1, 2, mask), d3 = codes(v1, 3, mask);
  const l0 = i16x8.add(i16x8.add(i16x8.extmul_low_i8x16_s(x0, c0), i16x8.extmul_low_i8x16_s(x1, c1)),
                       i16x8.add(i16x8.extmul_low_i8x16_s(x2, c2), i16x8.extmul_low_i8x16_s(x3, c3)));
  const h0 = i16x8.add(i16x8.add(i16x8.extmul_high_i8x16_s(x0, c0), i16x8.extmul_high_i8x16_s(x1, c1)),
                       i16x8.add(i16x8.extmul_high_i8x16_s(x2, c2), i16x8.extmul_high_i8x16_s(x3, c3)));
  const l1 = i16x8.add(i16x8.add(i16x8.extmul_low_i8x16_s(y0, d0), i16x8.extmul_low_i8x16_s(y1, d1)),
                       i16x8.add(i16x8.extmul_low_i8x16_s(y2, d2), i16x8.extmul_low_i8x16_s(y3, d3)));
  const h1 = i16x8.add(i16x8.add(i16x8.extmul_high_i8x16_s(y0, d0), i16x8.extmul_high_i8x16_s(y1, d1)),
                       i16x8.add(i16x8.extmul_high_i8x16_s(y2, d2), i16x8.extmul_high_i8x16_s(y3, d3)));
  return sums16(l0, h0, l1, h1);
}

// ---- r16s: r16 whose last plane is a shift of bytes with no mask (arm64 has one; x86-64 makes it of 16-bit shifts)
// @ts-ignore: decorator
@inline function last(v: v128): v128 {
  return i8x16.shr_u(v, 6);
}
// @ts-ignore: decorator
@inline function shiftedWhole(w: usize, x: usize, mask: v128): v128 {
  const v0 = v128.load(w), v1 = v128.load(w, 16);
  const x0 = v128.load(x), x1 = v128.load(x, 16), x2 = v128.load(x, 32), x3 = v128.load(x, 48);
  const c0 = codes(v0, 0, mask), c1 = codes(v0, 1, mask), c2 = codes(v0, 2, mask), c3 = last(v0);
  const y0 = v128.load(x, 64), y1 = v128.load(x, 80), y2 = v128.load(x, 96), y3 = v128.load(x, 112);
  const d0 = codes(v1, 0, mask), d1 = codes(v1, 1, mask), d2 = codes(v1, 2, mask), d3 = last(v1);
  return sums16(low4(x0, x1, x2, x3, c0, c1, c2, c3), high4(x0, x1, x2, x3, c0, c1, c2, c3),
                low4(y0, y1, y2, y3, d0, d1, d2, d3), high4(y0, y1, y2, y3, d0, d1, d2, d3));
}

// ---- dt: dot products of the bytes widened to int16
// @ts-ignore: decorator
@inline function dotLow(a: v128, c: v128): v128 {
  return i32x4.dot_i16x8_s(i16x8.extend_low_i8x16_s(a), i16x8.extend_low_i8x16_u(c));
}
// @ts-ignore: decorator
@inline function dotHigh(a: v128, c: v128): v128 {
  return i32x4.dot_i16x8_s(i16x8.extend_high_i8x16_s(a), i16x8.extend_high_i8x16_u(c));
}
// @ts-ignore: decorator
@inline function dot64(v: v128, x: usize, mask: v128): v128 {
  const x0 = v128.load(x), x1 = v128.load(x, 16), x2 = v128.load(x, 32), x3 = v128.load(x, 48);
  const c0 = codes(v, 0, mask), c1 = codes(v, 1, mask), c2 = codes(v, 2, mask), c3 = codes(v, 3, mask);
  const first = i32x4.add(i32x4.add(dotLow(x0, c0), dotLow(x1, c1)), i32x4.add(dotLow(x2, c2), dotLow(x3, c3)));
  const second = i32x4.add(i32x4.add(dotHigh(x0, c0), dotHigh(x1, c1)), i32x4.add(dotHigh(x2, c2), dotHigh(x3, c3)));
  return i32x4.add(v128.shuffle<i32>(first, second, 0, 1, 4, 5), v128.shuffle<i32>(first, second, 2, 3, 6, 7));
}
// @ts-ignore: decorator
@inline function dotWhole(w: usize, x: usize, mask: v128): v128 {
  return fourSums(dot64(v128.load(w), x, mask), dot64(v128.load(w, 16), x + 64, mask));
}

// A form's row loop (signsRows<Ternary>'s): F picks how a group's four int32 sums are made
class Copy {}
class Chain {}
class R16 {}
class Dot {}
class Tree {}
class Shifted {}
// @ts-ignore: decorator
@inline function wholeOf<F>(w: usize, x: usize, mask: v128): v128 {
  if (idof<F>() == idof<Copy>()) return copyWhole(w, x, mask);
  if (idof<F>() == idof<Chain>()) return chainWhole(w, x, mask);
  if (idof<F>() == idof<R16>()) return r16Whole(w, x, mask);
  if (idof<F>() == idof<Tree>()) return rtWhole(w, x, mask);
  if (idof<F>() == idof<Shifted>()) return shiftedWhole(w, x, mask);
  return dotWhole(w, x, mask);
}
// @ts-ignore: decorator
@inline function group<F>(facc: v128, w: usize, x: usize, sums: usize, xs: usize, at: usize, d: v128, mask: v128): v128 {
  const whole = i32x4.add(wholeOf<F>(w, x, mask), v128.load(sums + at));
  return f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(whole), v128.load(xs + at)), d));
}
// @ts-ignore: decorator
@inline function rows<F>(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, code: i32): void {
  const groups = n >> 7, groups4 = groups & ~3, mask = i8x16.splat(<i8>code);
  const sums = xs + (<usize>(n >> 5) << 2);
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 2);
    const srow = ws + ((<usize>i * <usize>groups) << 1);
    let facc = f32x4.splat(0);
    let g = 0;
    for (; g < groups4; g += 4) {
      const d = halves4(srow + (<usize>g << 1)), w = row + <usize>g * 32, x = xq + (<usize>g << 7), at = <usize>g << 4;
      facc = group<F>(facc, w, x, sums, xs, at, v128.shuffle<i32>(d, d, 0, 0, 0, 0), mask);
      facc = group<F>(facc, w + 32, x + 128, sums, xs, at + 16, v128.shuffle<i32>(d, d, 1, 1, 1, 1), mask);
      facc = group<F>(facc, w + 64, x + 256, sums, xs, at + 32, v128.shuffle<i32>(d, d, 2, 2, 2, 2), mask);
      facc = group<F>(facc, w + 96, x + 384, sums, xs, at + 48, v128.shuffle<i32>(d, d, 3, 3, 3, 3), mask);
    }
    for (; g < groups; g++) {
      facc = group<F>(facc, row + <usize>g * 32, xq + (<usize>g << 7), sums, xs, <usize>g << 4, f32x4.splat(half(srow + (<usize>g << 1))), mask);
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}
export function matmul_t2c(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  rows<Copy>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
export function matmul_ch(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  rows<Chain>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
export function matmul_r16(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  rows<R16>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
export function matmul_rt(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  rows<Tree>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
export function matmul_r16s(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  rows<Shifted>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
export function matmul_dt(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  rows<Dot>(xout, xq, xs, wq, ws, n, r0, r1, three);
}

// ---- w2: r16 on two rows at once (a and b), the activations loaded once for both. (Written out: AssemblyScript's
// inlined functions return one value.)
export function matmul_w2(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  const groups = n >> 7, mask = i8x16.splat(<i8>three);
  const sums = xs + (<usize>(n >> 5) << 2), bytes = <usize>(n >> 2);
  let i = r0;
  for (; i + 1 < r1; i += 2) {
    const rowA = wq + <usize>i * bytes, rowB = rowA + bytes;
    const sA = ws + ((<usize>i * <usize>groups) << 1), sB = sA + (<usize>groups << 1);
    let fa = f32x4.splat(0), fb = f32x4.splat(0);
    for (let g = 0; g < groups; g++) {
      const x = xq + (<usize>g << 7), at = <usize>g << 4, wa = rowA + <usize>g * 32, wb = rowB + <usize>g * 32;
      const x0 = v128.load(x), x1 = v128.load(x, 16), x2 = v128.load(x, 32), x3 = v128.load(x, 48);
      const y0 = v128.load(x, 64), y1 = v128.load(x, 80), y2 = v128.load(x, 96), y3 = v128.load(x, 112);
      const a0 = v128.load(wa), a1 = v128.load(wa, 16), b0 = v128.load(wb), b1 = v128.load(wb, 16);
      const p0 = codes(a0, 0, mask), p1 = codes(a0, 1, mask), p2 = codes(a0, 2, mask), p3 = codes(a0, 3, mask);
      const q0 = codes(a1, 0, mask), q1 = codes(a1, 1, mask), q2 = codes(a1, 2, mask), q3 = codes(a1, 3, mask);
      const wholeA = sums16(low4(x0, x1, x2, x3, p0, p1, p2, p3), high4(x0, x1, x2, x3, p0, p1, p2, p3),
                            low4(y0, y1, y2, y3, q0, q1, q2, q3), high4(y0, y1, y2, y3, q0, q1, q2, q3));
      const e0 = codes(b0, 0, mask), e1 = codes(b0, 1, mask), e2 = codes(b0, 2, mask), e3 = codes(b0, 3, mask);
      const f0 = codes(b1, 0, mask), f1 = codes(b1, 1, mask), f2 = codes(b1, 2, mask), f3 = codes(b1, 3, mask);
      const wholeB = sums16(low4(x0, x1, x2, x3, e0, e1, e2, e3), high4(x0, x1, x2, x3, e0, e1, e2, e3),
                            low4(y0, y1, y2, y3, f0, f1, f2, f3), high4(y0, y1, y2, y3, f0, f1, f2, f3));
      const s = v128.load(sums + at), scale = v128.load(xs + at);
      fa = f32x4.add(fa, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(i32x4.add(wholeA, s)), scale), f32x4.splat(half(sA + (<usize>g << 1)))));
      fb = f32x4.add(fb, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(i32x4.add(wholeB, s)), scale), f32x4.splat(half(sB + (<usize>g << 1)))));
    }
    store<f32>(xout + (<usize>i << 2), hsum(fa));
    store<f32>(xout + (<usize>(i + 1) << 2), hsum(fb));
  }
  if (i < r1) rows<R16>(xout, xq, xs, wq, ws, n, i, r1, three);
}
