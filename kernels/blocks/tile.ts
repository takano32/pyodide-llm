// blocks/tile.ts (T363.4): a prompt's tokens on rows of llama.cpp's blocks, four against one reading of a row:
// matmul_sbr_tile (T363.1's "rb tile": on every CPU) and matmul_sbw_tile ("sw tile": where the engine's relaxed dot
// product multiplies signed bytes by signed bytes, which signed_dots asks). Each number is matmul_sbr's to the bit.
// blocks/types.ts has what a type is and the arithmetic.
import { V_BITS, V_FIRST, V_THREE, V_FIFTEEN, K2, K4, XXS2, XS2, S2, XXS3, S3, S1, M1, XS4, Relaxed, Q2_K, Q4_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, bytesOf, sizeOf, bySixteen, hasSigns, hasMinimums, vectorAt, groupSums, four, signedBy, dotAdd, magnitudes, minuses, weights, scaleOf, minimumOf, blockScale, blockMinimum } from "./types";
import { matmul_sbr } from "./token";

// ---- a prompt's tokens (T108), four at a time against one reading of a row: token t's activations and their scales
// and sums at xq + t * frame and xs + t * frame, its outputs at xout + t * os. A unit's 16 weights stay in registers
// for the four tokens. A lane a token; every (row, token) is the token's kernel's number to the bit (the arithmetic at
// the head of this file). The tokens past the last four go through the token's kernel itself.
// the four tokens' values of 32 bits at p, frame apart
// @ts-ignore: decorator
@inline function gathered(p: usize, frame: usize): v128 {
  let v = v128.load32_zero(p);
  v = i32x4.replace_lane(v, 1, load<i32>(p + frame));
  v = i32x4.replace_lane(v, 2, load<i32>(p + 2 * frame));
  return i32x4.replace_lane(v, 3, load<i32>(p + 3 * frame));
}
// the 16 weights h against the four tokens' activations (x: the first token's 256 of the block), added to d0 .. d3,
// each made one lane: the four tokens' sums of |W| a
// @ts-ignore: decorator
@inline function dots4<K, R>(p: usize, h: i32, more: bool, x: usize, frame: usize, tab: usize, bits: v128, first: v128, three: v128, fifteen: v128): v128 {
  const x0 = x + (<usize>h << 4), x1 = x0 + frame, x2 = x1 + frame, x3 = x2 + frame, zero = i32x4.splat(0);
  const held = magnitudes<K, R>(p, h, tab, three, fifteen);
  let d0: v128, d1: v128, d2: v128, d3: v128;
  if (hasSigns<K>()) {
    const m = minuses<K, R>(p, h, tab, bits, first, fifteen);
    d0 = dotAdd<R>(zero, signedBy(v128.load(x0), m), held);
    d1 = dotAdd<R>(zero, signedBy(v128.load(x1), m), held);
    d2 = dotAdd<R>(zero, signedBy(v128.load(x2), m), held);
    d3 = dotAdd<R>(zero, signedBy(v128.load(x3), m), held);
  } else {
    d0 = dotAdd<R>(zero, v128.load(x0), held);
    d1 = dotAdd<R>(zero, v128.load(x1), held);
    d2 = dotAdd<R>(zero, v128.load(x2), held);
    d3 = dotAdd<R>(zero, v128.load(x3), held);
  }
  if (more) {  // a unit of 32: the next 16 as well, before the lanes are added up
    const next = magnitudes<K, R>(p, h + 1, tab, three, fifteen);
    if (hasSigns<K>()) {
      const m = minuses<K, R>(p, h + 1, tab, bits, first, fifteen);
      d0 = dotAdd<R>(d0, signedBy(v128.load(x0, 16), m), next);
      d1 = dotAdd<R>(d1, signedBy(v128.load(x1, 16), m), next);
      d2 = dotAdd<R>(d2, signedBy(v128.load(x2, 16), m), next);
      d3 = dotAdd<R>(d3, signedBy(v128.load(x3, 16), m), next);
    } else {
      d0 = dotAdd<R>(d0, v128.load(x0, 16), next);
      d1 = dotAdd<R>(d1, v128.load(x1, 16), next);
      d2 = dotAdd<R>(d2, v128.load(x2, 16), next);
      d3 = dotAdd<R>(d3, v128.load(x3, 16), next);
    }
  }
  return groupSums(d0, d1, d2, d3);
}
// G of group k of 32 of the block for the four tokens, a lane each. xs: the first token's scales of the block, after
// which (at `sums`: n / 8 bytes on) its sums of 16s and (n / 8 more) of 8s
// @ts-ignore: decorator
@inline function group4<K, R>(p: usize, k: i32, x: usize, s16: usize, s8: usize, frame: usize, tab: usize, bits: v128, first: v128, three: v128, fifteen: v128): v128 {
  const h = k << 1;
  if (bySixteen<K>()) {
    let low = dots4<K, R>(p, h, false, x, frame, tab, bits, first, three, fifteen), high = dots4<K, R>(p, h + 1, false, x, frame, tab, bits, first, three, fifteen);
    if (idof<K>() == idof<M1>()) {  // minus 7 or 9 times each token's two sums of 8, by bits 3 and 7 of the unit's byte
      const held = <i32>load<u8>(p + 32 + <usize>h), after = <i32>load<u8>(p + 33 + <usize>h);
      const times = i32x4.splat(((-7 - ((held >> 2) & 2)) & 0xffff) | ((-7 - ((held >> 6) & 2)) << 16));
      const next = i32x4.splat(((-7 - ((after >> 2) & 2)) & 0xffff) | ((-7 - ((after >> 6) & 2)) << 16));
      low = i32x4.add(i32x4.shl(low, 3), i32x4.dot_i16x8_s(times, gathered(s8 + (<usize>h << 2), frame)));
      high = i32x4.add(i32x4.shl(high, 3), i32x4.dot_i16x8_s(next, gathered(s8 + (<usize>h << 2) + 4, frame)));
    }
    return i32x4.add(i32x4.mul(low, i32x4.splat(scaleOf<K>(p, h))), i32x4.mul(high, i32x4.splat(scaleOf<K>(p, h + 1))));
  }
  let sums = dots4<K, R>(p, h, true, x, frame, tab, bits, first, three, fifteen);
  if (idof<K>() == idof<S1>()) {
    const times = i32x4.splat(7 + <i32>((load<u16>(p + 34 + (<usize>k << 1)) >> 15) << 1));
    sums = i32x4.sub(i32x4.shl(sums, 3), i32x4.mul(times, i32x4.extadd_pairwise_i16x8_s(gathered(s16 + (<usize>k << 2), frame))));
  }
  return i32x4.mul(sums, i32x4.splat(scaleOf<K>(p, h)));
}
// the K types: M of group k for the four tokens
// @ts-ignore: decorator
@inline function minimum4<K>(p: usize, k: i32, s16: usize, frame: usize): v128 {
  const h = k << 1;
  return i32x4.dot_i16x8_s(i32x4.splat(minimumOf<K>(p, h) | (minimumOf<K>(p, h + 1) << 16)), gathered(s16 + (<usize>k << 2), frame));
}
// what groups k and k + 4 of the block add to lane k of the four tokens' accumulators (k: 0 to 3)
// @ts-ignore: decorator
@inline function lane4<K, R>(p: usize, k: i32, x: usize, xs: usize, s16: usize, s8: usize, frame: usize, tab: usize, bits: v128, first: v128, three: v128, fifteen: v128): v128 {
  const a = gathered(xs + (<usize>k << 2), frame), b = gathered(xs + (<usize>k << 2) + 16, frame);
  const part = f32x4.mul(f32x4.add(f32x4.mul(f32x4.convert_i32x4_s(group4<K, R>(p, k, x, s16, s8, frame, tab, bits, first, three, fifteen)), a),
    f32x4.mul(f32x4.convert_i32x4_s(group4<K, R>(p, k + 4, x, s16, s8, frame, tab, bits, first, three, fifteen)), b)), f32x4.splat(blockScale<K>(p)));
  if (!hasMinimums<K>()) return part;
  return f32x4.sub(part, f32x4.mul(f32x4.add(f32x4.mul(f32x4.convert_i32x4_s(minimum4<K>(p, k, s16, frame)), a),
    f32x4.mul(f32x4.convert_i32x4_s(minimum4<K>(p, k + 4, s16, frame)), b)), f32x4.splat(blockMinimum<K>(p))));
}
// the four tokens' outputs of a row
// @ts-ignore: decorator
@inline function put4(out: usize, os: usize, a0: v128, a1: v128, a2: v128, a3: v128): void {
  const sums = f32x4.add(f32x4.add(f32x4.add(a0, a1), a2), a3);
  store<f32>(out, f32x4.extract_lane(sums, 0));
  store<f32>(out + os, f32x4.extract_lane(sums, 1));
  store<f32>(out + 2 * os, f32x4.extract_lane(sums, 2));
  store<f32>(out + 3 * os, f32x4.extract_lane(sums, 3));
}
// @ts-ignore: decorator
@inline function tileOf<R>(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, kind: i32): void {
  const blocks = n >> 8, size = bytesOf(kind), f = <usize>frame, count4 = count & ~3;
  const sums16 = <usize>(n >> 5) << 2, sums8 = sums16 + (<usize>(n >> 4) << 1);
  const bits = vectorAt(tab, V_BITS), first = vectorAt(tab, V_FIRST), three = vectorAt(tab, V_THREE), fifteen = vectorAt(tab, V_FIFTEEN);
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * <usize>blocks * size;
    for (let t = 0; t < count4; t += 4) {
      const xt = xq + <usize>t * f, st = xs + <usize>t * f;
      let a0 = f32x4.splat(0), a1 = a0, a2 = a0, a3 = a0;
      for (let b = 0; b < blocks; b++) {
        const p = row + <usize>b * size, x = xt + (<usize>b << 8), s = st + (<usize>b << 5), s16 = st + sums16 + (<usize>b << 5), s8 = st + sums8 + (<usize>b << 6);
        if (kind == IQ2_XXS) {
          a0 = f32x4.add(a0, lane4<XXS2, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<XXS2, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<XXS2, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<XXS2, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        } else if (kind == IQ3_S) {
          a0 = f32x4.add(a0, lane4<S3, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<S3, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<S3, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<S3, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        } else if (kind == IQ2_XS) {
          a0 = f32x4.add(a0, lane4<XS2, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<XS2, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<XS2, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<XS2, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        } else if (kind == IQ2_S) {
          a0 = f32x4.add(a0, lane4<S2, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<S2, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<S2, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<S2, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        } else if (kind == IQ3_XXS) {
          a0 = f32x4.add(a0, lane4<XXS3, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<XXS3, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<XXS3, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<XXS3, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        } else if (kind == IQ1_M) {
          a0 = f32x4.add(a0, lane4<M1, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<M1, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<M1, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<M1, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        } else if (kind == IQ1_S) {
          a0 = f32x4.add(a0, lane4<S1, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<S1, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<S1, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<S1, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        } else if (kind == Q2_K) {
          a0 = f32x4.add(a0, lane4<K2, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<K2, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<K2, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<K2, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        } else if (kind == IQ4_XS) {
          a0 = f32x4.add(a0, lane4<XS4, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<XS4, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<XS4, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<XS4, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        } else if (kind == Q4_K) {
          a0 = f32x4.add(a0, lane4<K4, R>(p, 0, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a1 = f32x4.add(a1, lane4<K4, R>(p, 1, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a2 = f32x4.add(a2, lane4<K4, R>(p, 2, x, s, s16, s8, f, tab, bits, first, three, fifteen));
          a3 = f32x4.add(a3, lane4<K4, R>(p, 3, x, s, s16, s8, f, tab, bits, first, three, fifteen));
        }
      }
      put4(xout + <usize>t * <usize>os + (<usize>i << 2), <usize>os, a0, a1, a2, a3);
    }
  }
}
/** count tokens of a prompt: matmul_sbr for each, four at a time against one reading of a row. Token t's activations
 * and their xs frame bytes apart, its outputs os bytes apart. (No tile without relaxed SIMD: a browser that lacks it
 * takes a prompt's tokens one by one, as jobs.ts does for every kind without a tile.) */
export function matmul_sbr_tile(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, type: i32): void {
  tileOf<Relaxed>(xout, xq, xs, w, tab, n, r0, r1, count, os, frame, type);
  for (let t = count & ~3; t < count; t++) matmul_sbr(xout + <usize>t * <usize>os, xq + <usize>t * <usize>frame, xs + <usize>t * <usize>frame, w, tab, n, r0, r1, type);
}

// ---- the tile with the signs on the weights (T363.1's "sw"): a row is spread once into a scratch area for all the
// tokens of the prompt, as W itself (int8) with S of each 16 weights and F of each block, and the dot products are of
// the activations by W: signed bytes by signed bytes in all their 8 bits, which relaxed SIMD does not promise (its
// second operand is 7 bits; arm64's SDOT multiplies signed by signed all the same, x86's PMADDUBSW takes the second as
// unsigned). signed_dots() is how the caller asks which this engine does, and only where every answer is the exact
// product may it call this kernel: there G is the same integers, and every number is matmul_sbr's to the bit. The K
// types go through matmul_sbr_tile (their W is not negative, and their M needs the sums anyway).
// scratch: n + n / 4 + n / 64 bytes of the caller's own (of one thread at a time): W, an int32 S for each 16 weights,
// a float32 F for each block.
/** out[0..3] = the four lanes of the relaxed dot product of the 16 bytes at a by the 16 at b. Where they are the sums of
 * the products of signed bytes for every a and b (tests/blocks-check.mjs's signedDots() asks), matmul_sbw_tile is right. */
export function signed_dots(out: usize, a: usize, b: usize): void {
  v128.store(out, i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(a), v128.load(b), i32x4.splat(0)));
}
// the 16 weights h of the block at p, and S of their unit. (h is written out, 0 to 15, where this is used: as a
// loop's counter every shift by it is a shift by a register, and the row was spread a quarter slower)
// @ts-ignore: decorator
@inline function spread16<K>(p: usize, h: i32, to: usize, s: usize, tab: usize, bits: v128, first: v128, three: v128, fifteen: v128): void {
  v128.store(to + (<usize>h << 4), weights<K, Relaxed>(p, h, tab, bits, first, three, fifteen));
  store<i32>(s + (<usize>h << 2), scaleOf<K>(p, h));
}
// @ts-ignore: decorator
@inline function spread<K>(row: usize, tab: usize, n: i32, scratch: usize, bits: v128, first: v128, three: v128, fifteen: v128): void {
  const size = sizeOf<K>(), scales = scratch + <usize>n, floats = scales + (<usize>n >> 2);
  for (let b = 0; b < n >> 8; b++) {
    const p = row + <usize>b * size, to = scratch + (<usize>b << 8), s = scales + (<usize>b << 6);
    store<f32>(floats + (<usize>b << 2), blockScale<K>(p));
    spread16<K>(p, 0, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 1, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 2, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 3, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 4, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 5, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 6, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 7, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 8, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 9, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 10, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 11, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 12, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 13, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 14, to, s, tab, bits, first, three, fifteen);
    spread16<K>(p, 15, to, s, tab, bits, first, three, fifteen);
  }
}
// group k of 32 of a spread block (its W at a, its S at s) for the four tokens: G times the tokens' scales
// @ts-ignore: decorator
@inline function spread4(sixteens: bool, a: usize, s: usize, k: i32, x: usize, xs: usize, frame: usize): v128 {
  const o = <usize>k << 5, x0 = x + o, x1 = x0 + frame, x2 = x1 + frame, x3 = x2 + frame, zero = i32x4.splat(0);
  const low = v128.load(a + o), high = v128.load(a + o, 16);
  let sums: v128;
  if (sixteens) {
    const first = groupSums(i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x0), low, zero), i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x1), low, zero),
      i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x2), low, zero), i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x3), low, zero));
    const second = groupSums(i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x0, 16), high, zero), i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x1, 16), high, zero),
      i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x2, 16), high, zero), i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x3, 16), high, zero));
    sums = i32x4.add(i32x4.mul(first, i32x4.splat(load<i32>(s + (<usize>k << 3)))), i32x4.mul(second, i32x4.splat(load<i32>(s + (<usize>k << 3), 4))));
  } else {
    sums = i32x4.mul(groupSums(i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x0, 16), high, i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x0), low, zero)),
      i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x1, 16), high, i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x1), low, zero)),
      i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x2, 16), high, i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x2), low, zero)),
      i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x3, 16), high, i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x3), low, zero))), i32x4.splat(load<i32>(s + (<usize>k << 3))));
  }
  return f32x4.mul(f32x4.convert_i32x4_s(sums), gathered(xs + (<usize>k << 2), frame));
}
// @ts-ignore: decorator
@inline function spreadTokens(sixteens: bool, out: usize, xq: usize, xs: usize, n: i32, count4: i32, os: usize, f: usize, scratch: usize): void {
  const scales = scratch + <usize>n, floats = scales + (<usize>n >> 2);
  for (let t = 0; t < count4; t += 4) {
    const xt = xq + <usize>t * f, st = xs + <usize>t * f;
    let a0 = f32x4.splat(0), a1 = a0, a2 = a0, a3 = a0;
    for (let b = 0; b < n >> 8; b++) {
      const a = scratch + (<usize>b << 8), s = scales + (<usize>b << 6), x = xt + (<usize>b << 8), g = st + (<usize>b << 5);
      const d = f32x4.splat(load<f32>(floats + (<usize>b << 2)));
      a0 = f32x4.add(a0, f32x4.mul(f32x4.add(spread4(sixteens, a, s, 0, x, g, f), spread4(sixteens, a, s, 4, x, g, f)), d));
      a1 = f32x4.add(a1, f32x4.mul(f32x4.add(spread4(sixteens, a, s, 1, x, g, f), spread4(sixteens, a, s, 5, x, g, f)), d));
      a2 = f32x4.add(a2, f32x4.mul(f32x4.add(spread4(sixteens, a, s, 2, x, g, f), spread4(sixteens, a, s, 6, x, g, f)), d));
      a3 = f32x4.add(a3, f32x4.mul(f32x4.add(spread4(sixteens, a, s, 3, x, g, f), spread4(sixteens, a, s, 7, x, g, f)), d));
    }
    put4(out + <usize>t * os, os, a0, a1, a2, a3);
  }
}
export function matmul_sbw_tile(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, type: i32, scratch: usize): void {
  const count4 = count & ~3;
  if (type == Q2_K || type == Q4_K || count4 == 0) {
    matmul_sbr_tile(xout, xq, xs, w, tab, n, r0, r1, count, os, frame, type);
    return;
  }
  const rowBytes = <usize>(n >> 8) * bytesOf(type);
  const bits = vectorAt(tab, V_BITS), first = vectorAt(tab, V_FIRST), three = vectorAt(tab, V_THREE), fifteen = vectorAt(tab, V_FIFTEEN);
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * rowBytes, out = xout + (<usize>i << 2);
    if (type == IQ2_XXS) spread<XXS2>(row, tab, n, scratch, bits, first, three, fifteen);
    else if (type == IQ3_S) spread<S3>(row, tab, n, scratch, bits, first, three, fifteen);
    else if (type == IQ2_XS) spread<XS2>(row, tab, n, scratch, bits, first, three, fifteen);
    else if (type == IQ2_S) spread<S2>(row, tab, n, scratch, bits, first, three, fifteen);
    else if (type == IQ3_XXS) spread<XXS3>(row, tab, n, scratch, bits, first, three, fifteen);
    else if (type == IQ1_M) spread<M1>(row, tab, n, scratch, bits, first, three, fifteen);
    else if (type == IQ1_S) spread<S1>(row, tab, n, scratch, bits, first, three, fifteen);
    else if (type == IQ4_XS) spread<XS4>(row, tab, n, scratch, bits, first, three, fifteen);
    if (type == IQ2_XS || type == IQ2_S || type == IQ1_M) spreadTokens(true, out, xq, xs, n, count4, <usize>os, <usize>frame, scratch);
    else spreadTokens(false, out, xq, xs, n, count4, <usize>os, <usize>frame, scratch);
  }
  for (let t = count4; t < count; t++) matmul_sbr(xout + <usize>t * <usize>os, xq + <usize>t * <usize>frame, xs + <usize>t * <usize>frame, w, tab, n, r0, r1, type);
}
