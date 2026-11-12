// tests/ternary-relaxed-forms.ts (T288): forms of the ternary product with relaxed SIMD, set against kernels/'s
// matmul_t2r (what Chromium, Firefox and WebKit 27 run). tests/ternary-browser-bench.ts --relaxed compiles this file and
// times each form beside matmul_t2r in a browser's engine and in Node's V8, and holds every form's numbers to
// matmul_t2r's to the bit. Not a kernel of the page: nothing here is shipped.
//
// The activations and the float16 scales are matmul_t2r's (kernel/matmul.ts's interleave), and so are the weights but
// for tl's: every form makes the same exact integers a group of 32 activations (dot(a, w + 1), then the activations'
// sums that interleave() left) and the same float32 operations in the same order, so the same numbers to the bit.
//
//   rc   a copy of matmul_t2r (relaxed/signs.ts's signsRows<Ternary>) in this module: the noise of two copies of one form
//   x16  the planes' products as i16x8.relaxed_dot_i8x16_i7x16_s (x86's pmaddubsw alone, no pmaddwd), added in int16
//        over the four planes of a block and the two blocks of a group of 128, one widening to int32 a group of 128.
//        T231's review timed such a form (bxa) at 1.31 to 1.36 times matmul_t2r on AMD's Zen 3 and 4, 0.60 on arm64.
//        At most 3072 a lane of a block (4 planes x a pair x 384, PQ2_0's fourth code x |-128|), 6144 once two lanes add
//   x16b x16 widened a block (as T231's bxa: one extadd a block of 64, then the four groups' sums as matmul_t2r makes them)
//   tl   BitNet's TL1 (microsoft/BitNet, MIT; none of its lines is here): the weights of 16 rows a vector, a pair of
//        weights a nibble (4 bits: two codes), and for every pair of activations a table of the 16 sums a0 c0 + a1 c1
//        (c0, c1 = 0..3) made once a token (in int16: its low bytes and its high bytes, two tables of 16 bytes), looked
//        up by the weights' nibbles (relaxed_swizzle) for 16 rows at once. Its own layout of the weights (layout_tl
//        below; the bench writes its copies so) and its tables at `scratch`, made in every call (the cost of a token's
//        tables counted against every matrix, as if no two matrices shared their input). At most 12288 a group of 32 in
//        int16 (16 tables x 768)
import { codes, fourSums } from "../kernels/ternary";
import { half, halves4 } from "../kernels/kernel/halves";
import { hsum } from "../kernels/kernel/math";

// ---- rc: relaxed/signs.ts's dot64 and group128 for Ternary, as they are
// @ts-ignore: decorator
@inline function copy64(v: v128, x: usize, mask: v128): v128 {
  let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x), codes(v, 0, mask), i32x4.splat(0));
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 16), codes(v, 1, mask), acc);
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 32), codes(v, 2, mask), acc);
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 48), codes(v, 3, mask), acc);
}
// @ts-ignore: decorator
@inline function copyWhole(w: usize, x: usize, mask: v128): v128 {
  return fourSums(copy64(v128.load(w), x, mask), copy64(v128.load(w, 16), x + 64, mask));
}

// ---- x16: the pairs' products in int16 (lanes 0..3 of a block the first group of 32, 4..7 the second)
// @ts-ignore: decorator
@inline function pairs64(v: v128, x: usize, mask: v128): v128 {
  let acc = i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x), codes(v, 0, mask));
  acc = i16x8.add(acc, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 16), codes(v, 1, mask)));
  acc = i16x8.add(acc, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 32), codes(v, 2, mask)));
  return i16x8.add(acc, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 48), codes(v, 3, mask)));
}
// @ts-ignore: decorator
@inline function x16Whole(w: usize, x: usize, mask: v128): v128 {
  const a = pairs64(v128.load(w), x, mask), b = pairs64(v128.load(w, 16), x + 64, mask);
  // 32-bit lanes [a0 a1 | a2 a3] are groups 0 and 1, b's 2 and 3: two int16 a group in each lane of the sum
  return i32x4.extadd_pairwise_i16x8_s(i16x8.add(v128.shuffle<i32>(a, b, 0, 2, 4, 6), v128.shuffle<i32>(a, b, 1, 3, 5, 7)));
}
// @ts-ignore: decorator
@inline function x16bWhole(w: usize, x: usize, mask: v128): v128 {
  return fourSums(i32x4.extadd_pairwise_i16x8_s(pairs64(v128.load(w), x, mask)),
                  i32x4.extadd_pairwise_i16x8_s(pairs64(v128.load(w, 16), x + 64, mask)));
}

// A form's row loop (signsRows<Ternary>'s): F picks how a group's four int32 sums are made
class Copy {}
class X16 {}
class X16b {}
// @ts-ignore: decorator
@inline function wholeOf<F>(w: usize, x: usize, mask: v128): v128 {
  if (idof<F>() == idof<Copy>()) return copyWhole(w, x, mask);
  if (idof<F>() == idof<X16>()) return x16Whole(w, x, mask);
  return x16bWhole(w, x, mask);
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
export function matmul_rc(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  rows<Copy>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
export function matmul_x16(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  rows<X16>(xout, xq, xs, wq, ws, n, r0, r1, three);
}
export function matmul_x16b(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32): void {
  rows<X16b>(xout, xq, xs, wq, ws, n, r0, r1, three);
}

// ---- tl. A group of 128 activations (two interleaved blocks of 64: byte 16 p + c of a block is activation 4 c + p)
// is 64 pairs, table t = 32 blk + 16 k + 4 p + j (k = 0, 1 the block's groups of 32, p = 0..3 the plane, j = 0..3) of
// the block's bytes 16 p + 2 (4 k + j) and the next: both in group of 32 number 2 blk + k (lane 2 blk + k of
// matmul_t2r's sums). Table t is 32 bytes at scratch + TABLES + 32 (64 g + t): the low bytes of the 16 int16 sums
// a0 (i & 3) + a1 (i >> 2), then their high bytes.
// The weights: rows in blocks of 16, a block's groups of 128 in order, a group's 32 vectors of 16 bytes: vector q's
// byte r is row r's nibbles of tables 2 q (its low 4 bits) and 2 q + 1 (its high), a nibble c0 | c1 << 2 (the codes
// of the pair's first and second weight). Then the scales: a block's groups in order, 16 float16 a group (its rows').
const TABLES: usize = 256;  // scratch: 16 rows x 4 lanes of float32 accumulators, then the tables
// @ts-ignore: decorator
@inline function pairAt(t: i32): i32 {  // the byte of the group of 128's activations that is table t's first
  return ((t >> 5) << 6) + (((t >> 2) & 3) << 4) + (((((t >> 4) & 1) << 2) + (t & 3)) << 1);
}
// a token's tables
// @ts-ignore: decorator
@inline function tables(xq: usize, n: i32, scratch: usize): void {
  const c0 = i16x8.replace_lane(i16x8.replace_lane(i16x8.replace_lane(i16x8.splat(0), 1, 1), 2, 2), 3, 3);
  const first = v128.shuffle<i16>(c0, c0, 0, 1, 2, 3, 0, 1, 2, 3);  // i & 3
  const lowOne = v128.shuffle<i16>(c0, c0, 0, 0, 0, 0, 1, 1, 1, 1), highOne = v128.shuffle<i16>(c0, c0, 2, 2, 2, 2, 3, 3, 3, 3);  // i >> 2
  const pairs = n >> 1;
  for (let t = 0; t < pairs; t++) {
    const at = xq + (<usize>(t >> 6) << 7) + <usize>pairAt(t & 63);
    const a0 = i16x8.splat(<i16>load<i8>(at)), a1 = i16x8.splat(<i16>load<i8>(at, 1));
    const lo = i16x8.add(i16x8.mul(a0, first), i16x8.mul(a1, lowOne)), hi = i16x8.add(i16x8.mul(a0, first), i16x8.mul(a1, highOne));
    const out = scratch + TABLES + (<usize>t << 5);
    v128.store(out, v128.shuffle<u8>(lo, hi, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30));
    v128.store(out, v128.shuffle<u8>(lo, hi, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31), 16);
  }
}
// two tables' sums of 16 rows (v: their nibbles) added to a (rows 0..7) and b (8..15): a vector of two
// @ts-ignore: decorator
@inline function look(table: usize, i: v128, a: v128): v128 {
  const lo = i8x16.relaxed_swizzle(v128.load(table), i), hi = i8x16.relaxed_swizzle(v128.load(table, 16), i);
  return i16x8.add(a, v128.shuffle<u8>(lo, hi, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23));
}
// @ts-ignore: decorator
@inline function lookHigh(table: usize, i: v128, b: v128): v128 {
  const lo = i8x16.relaxed_swizzle(v128.load(table), i), hi = i8x16.relaxed_swizzle(v128.load(table, 16), i);
  return i16x8.add(b, v128.shuffle<u8>(lo, hi, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31));
}
// four rows' float32 accumulator of lane k, plus the group of 32's numbers (as matmul_t2r's group128 scales them)
// @ts-ignore: decorator
@inline function scaledInto(place: usize, ints: v128, sum: v128, scale: v128, d: v128): void {
  v128.store(place, f32x4.add(v128.load(place), f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(i32x4.add(ints, sum)), scale), d)));
}
export function matmul_tl(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32, three: i32, scratch: usize): void {
  tables(xq, n, scratch);
  const groups = n >> 7, sums = xs + (<usize>(n >> 5) << 2);
  const nibble = i8x16.splat(<i8>(three * 5));  // 15
  for (let block = r0 >> 4; block < (r1 + 15) >> 4; block++) {
    for (let k = 0; k < 16; k++) v128.store(scratch + (<usize>k << 4), f32x4.splat(0));
    for (let g = 0; g < groups; g++) {
      const w = wq + ((<usize>block * <usize>groups + <usize>g) << 9);
      const sp = ws + ((<usize>block * <usize>groups + <usize>g) << 5);
      const d0 = halves4(sp), d1 = halves4(sp + 8), d2 = halves4(sp + 16), d3 = halves4(sp + 24);
      for (let lane = 0; lane < 4; lane++) {
        let a = i16x8.splat(0), b = a;
        const table = scratch + TABLES + ((<usize>(g << 6) + <usize>(lane << 4)) << 5);
        for (let q = 0; q < 8; q++) {
          const v = v128.load(w + (<usize>((lane << 3) + q) << 4));
          const i0 = v128.and(v, nibble), i1 = i8x16.shr_u(v, 4);
          const t0 = table + (<usize>q << 6);
          a = look(t0, i0, a); b = lookHigh(t0, i0, b);
          a = look(t0 + 32, i1, a); b = lookHigh(t0 + 32, i1, b);
        }
        const at = <usize>((g << 2) + lane) << 2;
        const sum = v128.load32_splat(sums + at), scale = v128.load32_splat(xs + at), place = scratch + (<usize>lane << 6);
        scaledInto(place, i32x4.extend_low_i16x8_s(a), sum, scale, d0);
        scaledInto(place + 16, i32x4.extend_high_i16x8_s(a), sum, scale, d1);
        scaledInto(place + 32, i32x4.extend_low_i16x8_s(b), sum, scale, d2);
        scaledInto(place + 48, i32x4.extend_high_i16x8_s(b), sum, scale, d3);
      }
    }
    // a row's lanes added in matmul_t2r's order (hsum: ((l0 + l1) + l2) + l3)
    for (let quad = 0; quad < 4; quad++) {
      const at = scratch + (<usize>quad << 4);
      const out = f32x4.add(f32x4.add(f32x4.add(v128.load(at), v128.load(at, 64)), v128.load(at, 128)), v128.load(at, 192));
      const row = (block << 4) + (quad << 2), to = xout + (<usize>row << 2);
      if (row >= r0 && row + 4 <= r1) {
        v128.store(to, out);
      } else {
        if (row >= r0 && row < r1) store<f32>(to, f32x4.extract_lane(out, 0));
        if (row + 1 >= r0 && row + 1 < r1) store<f32>(to + 4, f32x4.extract_lane(out, 1));
        if (row + 2 >= r0 && row + 2 < r1) store<f32>(to + 8, f32x4.extract_lane(out, 2));
        if (row + 3 >= r0 && row + 3 < r1) store<f32>(to + 12, f32x4.extract_lane(out, 3));
      }
    }
  }
}
// @ts-ignore: decorator
@inline function codeAt(src: usize, bytes: usize, row: i32, j: i32): i32 {
  return (<i32>load<u8>(src + <usize>row * bytes + <usize>(j >> 2)) >> ((j & 3) << 1)) & 3;
}
// matmul_t2r's weights (rows x n, the scales after the codes at `scales`) into tl's layout at dst (its scales after
// its codes). rows a multiple of 16.
export function layout_tl(dst: usize, src: usize, scales: usize, rows: i32, n: i32): void {
  const groups = n >> 7, bytes = <usize>(n >> 2);
  const dscales = dst + <usize>rows * bytes;
  for (let block = 0; block < rows >> 4; block++) {
    for (let g = 0; g < groups; g++) {
      const w = dst + ((<usize>block * <usize>groups + <usize>g) << 9);
      for (let t = 0; t < 64; t++) {
        // the weights of table t: the group's activations at pairAt(t) and the next, in interleaved order (a block's byte
        // 16 p + c is weight 4 c + p)
        const b0 = pairAt(t), b1 = b0 + 1;
        const j0 = (g << 7) + ((b0 >> 6) << 6) + ((b0 & 15) << 2) + ((b0 >> 4) & 3);
        const j1 = (g << 7) + ((b1 >> 6) << 6) + ((b1 & 15) << 2) + ((b1 >> 4) & 3);
        for (let r = 0; r < 16; r++) {
          const row = (block << 4) + r, nib = codeAt(src, bytes, row, j0) | (codeAt(src, bytes, row, j1) << 2);
          const at = w + (<usize>(t >> 1) << 4) + <usize>r;
          store<u8>(at, (<i32>load<u8>(at) & ((t & 1) ? 0x0f : 0xf0)) | (nib << ((t & 1) << 2)));
        }
      }
      for (let r = 0; r < 16; r++) {
        store<u16>(dscales + ((((<usize>block * <usize>groups + <usize>g) << 4) + <usize>r) << 1),
          load<u16>(scales + (((<usize>((block << 4) + r) * <usize>groups) + <usize>g) << 1)));
      }
    }
  }
}
