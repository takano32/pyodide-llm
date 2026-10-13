// blocks/token.ts (T363.4): one token's rows of llama.cpp's blocks, matmul_sb and matmul_sbr (the same numbers, with
// relaxed SIMD and without), and sums_x, what its activations need. blocks/types.ts has what a type is and the
// arithmetic; T363.1's form "rb3" (docs/notes/t363.1-saluki-kernel-forms-2026-10-13.md).
import { V_BITS, V_FIRST, V_THREE, V_FIFTEEN, V_POWERS3, K2, K4, XXS2, XS2, S2, XXS3, S3, S1, M1, XS4, Relaxed, Plain, Q2_K, Q4_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, bytesOf, bySixteen, hasSigns, hasMinimums, hsum, vectorAt, groupSums, pairSums, four, signedBy, dotAdd, magnitudes, minuses, weights, scaleOf, minimumOf, blockScale, blockMinimum } from "./types";

// ---- one token. S of the four units from 16 weights h on, a lane each (h a multiple of 4 where a unit is 16 weights,
// of 8 where it is 32)
// (the small constants are made of `fifteen`, sixteen bytes of 15 that every kernel holds in a register: a vector read
// from the tables wherever it is used is a load each time, which cost IQ1_M a tenth)
// @ts-ignore: decorator
@inline function ones32(fifteen: v128): v128 {
  return i32x4.shr_u(fifteen, 27);
}
// @ts-ignore: decorator
@inline function sevens32(fifteen: v128): v128 {
  return i32x4.shr_u(fifteen, 25);
}
// the four halves of the two bytes of word, a lane each, the low half of the first byte first
// @ts-ignore: decorator
@inline function nibbles4(word: i32, fifteen: v128): v128 {
  const both = i32x4.splat(word), low = v128.and(both, fifteen), high = v128.and(i16x8.shr_u(both, 4), fifteen);
  return v128.shuffle<i8>(low, high, 0, 2, 2, 2, 16, 2, 2, 2, 1, 2, 2, 2, 17, 2, 2, 2);  // (bytes 2 and 3 of a lane are zero)
}
// @ts-ignore: decorator
@inline function scales4<K>(p: usize, h: i32, tab: usize, fifteen: v128): v128 {
  const at = <usize>h, one = ones32(fifteen);
  if (idof<K>() == idof<XXS2>()) {  // the words are every other int32 from the block's byte 2
    const q = p + 2 + (at << 2);
    return v128.or(i32x4.shr_u(v128.shuffle<i32>(v128.load(q), v128.load(q, 16), 1, 3, 5, 7), 27), one);
  }
  if (idof<K>() == idof<XXS3>()) return v128.or(i32x4.shr_u(v128.load(p + 66 + (at << 1)), 27), one);
  if (idof<K>() == idof<XS2>() || idof<K>() == idof<S2>() || idof<K>() == idof<S3>()) {  // the four halves of two bytes
    const word = <i32>load<u16>(idof<K>() == idof<XS2>() ? p + 66 + (at >> 1) : idof<K>() == idof<S2>() ? p + 74 + (at >> 1) : p + 106 + (at >> 2));
    return v128.or(i32x4.shl(nibbles4(word, fifteen), 1), one);
  }
  if (idof<K>() == idof<S1>()) {
    const words = i32x4.extend_low_i16x8_u(v128.load64_zero(p + 34 + (at >> 1) * 2));
    return v128.or(i32x4.shl(v128.and(i32x4.shr_u(words, 12), sevens32(fifteen)), 1), one);
  }
  if (idof<K>() == idof<M1>()) {  // four fields of three bits of a word
    const word = <i32>load<u16>(p + 48 + (at >> 2) * 2);
    const threes = v128.and(i32x4.shr_u(i32x4.mul(i32x4.splat(word), vectorAt(tab, V_POWERS3)), 9), sevens32(fifteen));
    return v128.or(i32x4.shl(threes, 1), one);
  }
  if (idof<K>() == idof<K2>()) {  // the low halves of four bytes
    return v128.and(i32x4.extend_low_i16x8_u(i16x8.extend_low_i8x16_u(v128.load32_zero(p + at))), i32x4.shr_u(fifteen, 24));
  }
  return four(scaleOf<K>(p, h), scaleOf<K>(p, h + 2), scaleOf<K>(p, h + 4), scaleOf<K>(p, h + 6));
}
// the 16 weights h against their activations (x: the block's 256): four lanes whose sum is the sum of |W| a (of the
// code or W itself where there is no sign apart), added to acc
// @ts-ignore: decorator
@inline function dot16<K, R>(acc: v128, p: usize, h: i32, x: usize, tab: usize, bits: v128, first: v128, three: v128, fifteen: v128): v128 {
  const a = v128.load(x + (<usize>h << 4)), held = magnitudes<K, R>(p, h, tab, three, fifteen);
  if (hasSigns<K>()) return dotAdd<R>(acc, signedBy(a, minuses<K, R>(p, h, tab, bits, first, fifteen)), held);
  return dotAdd<R>(acc, a, held);
}
// G of the four groups of 32 from 16 weights h on (0 or 8), a lane each. sums16, sums8: the block's activations' sums
// @ts-ignore: decorator
@inline function sums4<K, R>(p: usize, h: i32, x: usize, sums16: usize, sums8: usize, tab: usize, bits: v128, first: v128, three: v128, fifteen: v128): v128 {
  const zero = i32x4.splat(0);
  if (bySixteen<K>()) {  // eight units: each a lane, times its S, then two by two
    let low = groupSums(dot16<K, R>(zero, p, h, x, tab, bits, first, three, fifteen), dot16<K, R>(zero, p, h + 1, x, tab, bits, first, three, fifteen),
      dot16<K, R>(zero, p, h + 2, x, tab, bits, first, three, fifteen), dot16<K, R>(zero, p, h + 3, x, tab, bits, first, three, fifteen));
    let high = groupSums(dot16<K, R>(zero, p, h + 4, x, tab, bits, first, three, fifteen), dot16<K, R>(zero, p, h + 5, x, tab, bits, first, three, fifteen),
      dot16<K, R>(zero, p, h + 6, x, tab, bits, first, three, fifteen), dot16<K, R>(zero, p, h + 7, x, tab, bits, first, three, fifteen));
    if (idof<K>() == idof<M1>()) {
      low = i32x4.add(i32x4.shl(low, 3), sevensM(p, h, sums8, fifteen));
      high = i32x4.add(i32x4.shl(high, 3), sevensM(p, h + 4, sums8, fifteen));
    }
    return pairSums(i32x4.mul(low, scales4<K>(p, h, tab, fifteen)), i32x4.mul(high, scales4<K>(p, h + 4, tab, fifteen)));
  }
  let sums = groupSums(dot16<K, R>(dot16<K, R>(zero, p, h, x, tab, bits, first, three, fifteen), p, h + 1, x, tab, bits, first, three, fifteen),
    dot16<K, R>(dot16<K, R>(zero, p, h + 2, x, tab, bits, first, three, fifteen), p, h + 3, x, tab, bits, first, three, fifteen),
    dot16<K, R>(dot16<K, R>(zero, p, h + 4, x, tab, bits, first, three, fifteen), p, h + 5, x, tab, bits, first, three, fifteen),
    dot16<K, R>(dot16<K, R>(zero, p, h + 6, x, tab, bits, first, three, fifteen), p, h + 7, x, tab, bits, first, three, fifteen));
  if (idof<K>() == idof<S1>()) {  // 7 or 9 times each 32 activations' sum, by the top bit of the unit's word
    const words = i32x4.extend_low_i16x8_u(v128.load64_zero(p + 34 + <usize>h));
    const times = i32x4.add(sevens32(fifteen), i32x4.shl(i32x4.shr_u(words, 15), 1));
    sums = i32x4.sub(i32x4.shl(sums, 3), i32x4.mul(times, i32x4.extadd_pairwise_i16x8_s(v128.load(sums16 + (<usize>h << 1)))));
  }
  return i32x4.mul(sums, scales4<K>(p, h, tab, fifteen));
}
// IQ1_M: minus 7 or minus 9 times the sums of the eight groups of 8 from 16 weights h on (a multiple of 4), two by two:
// a lane each unit of 16. A unit's byte of qh says by its bits 3 and 7 which of its two groups of 8 are shifted down
// @ts-ignore: decorator
@inline function sevensM(p: usize, h: i32, sums8: usize, fifteen: v128): v128 {
  const held = i16x8.extend_low_i8x16_u(v128.load32_zero(p + 32 + <usize>h));  // four int16s, a unit each
  const first = v128.and(i16x8.shr_u(held, 3), i16x8.shr_u(fifteen, 11)), second = i16x8.shr_u(held, 7);
  const down = v128.shuffle<i16>(first, second, 0, 8, 1, 9, 2, 10, 3, 11);  // 1 where a group of 8 is down
  return i32x4.dot_i16x8_s(i16x8.sub(i16x8.neg(i16x8.shr_u(fifteen, 9)), i16x8.add(down, down)), v128.load(sums8 + (<usize>h << 2)));
}
// the K types: M of the four groups of 32 from 16 weights h on (0 or 8), a lane each
// @ts-ignore: decorator
@inline function minimums4<K>(p: usize, h: i32, sums16: usize, fifteen: v128): v128 {
  const sums = v128.load(sums16 + (<usize>h << 1));
  if (idof<K>() == idof<K2>()) {
    const least = v128.and(i16x8.shr_u(v128.load64_zero(p + <usize>h), 4), fifteen);
    return i32x4.dot_i16x8_s(i16x8.extend_low_i8x16_u(least), sums);
  }
  return i32x4.mul(four(minimumOf<K>(p, h), minimumOf<K>(p, h + 2), minimumOf<K>(p, h + 4), minimumOf<K>(p, h + 6)), i32x4.extadd_pairwise_i16x8_s(sums));
}
// eight groups' integers (low: the first four, a lane each) times their activations' scales: lane k is groups k and k + 4
// @ts-ignore: decorator
@inline function scaled(low: v128, high: v128, xs: usize): v128 {
  return f32x4.add(f32x4.mul(f32x4.convert_i32x4_s(low), v128.load(xs)), f32x4.mul(f32x4.convert_i32x4_s(high), v128.load(xs, 16)));
}
// a block's part of a row's sum: four lanes
// @ts-ignore: decorator
@inline function blockOf<K, R>(p: usize, x: usize, xs: usize, sums16: usize, sums8: usize, tab: usize, bits: v128, first: v128, three: v128, fifteen: v128): v128 {
  const part = f32x4.mul(scaled(sums4<K, R>(p, 0, x, sums16, sums8, tab, bits, first, three, fifteen),
    sums4<K, R>(p, 8, x, sums16, sums8, tab, bits, first, three, fifteen), xs), f32x4.splat(blockScale<K>(p)));
  if (!hasMinimums<K>()) return part;
  return f32x4.sub(part, f32x4.mul(scaled(minimums4<K>(p, 0, sums16, fifteen), minimums4<K>(p, 8, sums16, fifteen), xs), f32x4.splat(blockMinimum<K>(p))));
}
// @ts-ignore: decorator
@inline function rowsOf<R>(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, kind: i32): void {
  const blocks = n >> 8, size = bytesOf(kind);
  const sums16 = xs + (<usize>(n >> 5) << 2), sums8 = sums16 + (<usize>(n >> 4) << 1);
  const bits = vectorAt(tab, V_BITS), first = vectorAt(tab, V_FIRST), three = vectorAt(tab, V_THREE), fifteen = vectorAt(tab, V_FIFTEEN);
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * <usize>blocks * size;
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const p = row + <usize>b * size, x = xq + (<usize>b << 8), s = xs + (<usize>b << 5), s16 = sums16 + (<usize>b << 5), s8 = sums8 + (<usize>b << 6);
      let part = f32x4.splat(0);
      if (kind == IQ2_XXS) part = blockOf<XXS2, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      else if (kind == IQ3_S) part = blockOf<S3, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      else if (kind == IQ2_XS) part = blockOf<XS2, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      else if (kind == IQ2_S) part = blockOf<S2, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      else if (kind == IQ3_XXS) part = blockOf<XXS3, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      else if (kind == IQ1_M) part = blockOf<M1, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      else if (kind == IQ1_S) part = blockOf<S1, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      else if (kind == Q2_K) part = blockOf<K2, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      else if (kind == IQ4_XS) part = blockOf<XS4, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      else if (kind == Q4_K) part = blockOf<K4, R>(p, x, s, s16, s8, tab, bits, first, three, fifteen);
      facc = f32x4.add(facc, part);
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}
/** One token: xout[i] = the dot product of row i of w (n weights in blocks of `type`) and the activations at xq, xs,
 * for rows r0 to r1. Without relaxed SIMD; matmul_sbr's numbers to the bit. */
export function matmul_sb(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, type: i32): void {
  rowsOf<Plain>(xout, xq, xs, w, tab, n, r0, r1, type);
}
export function matmul_sbr(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, type: i32): void {
  rowsOf<Relaxed>(xout, xq, xs, w, tab, n, r0, r1, type);
}

// ---- what a token's activations need besides quantize_x(xq, xs, x, n, 0): after their n / 32 scales at xs, the int16
// sums of each 16 of the n int8 at xq (n / 16 of them), then of each 8 (n / 8). n: a multiple of 32.
export function sums_x(xs: usize, xq: usize, n: i32): void {
  const sums16 = xs + (<usize>(n >> 5) << 2), sums8 = sums16 + (<usize>(n >> 4) << 1);
  for (let g = 0; g < n >> 5; g++) {
    const p = xq + (<usize>g << 5);
    // the sums of each 4 bytes, then of each 8, then of each 16
    const low = i32x4.extadd_pairwise_i16x8_s(i16x8.extadd_pairwise_i8x16_s(v128.load(p)));
    const high = i32x4.extadd_pairwise_i16x8_s(i16x8.extadd_pairwise_i8x16_s(v128.load(p, 16)));
    const eights = pairSums(low, high);
    store<i64>(sums8 + (<usize>g << 3), i64x2.extract_lane(i16x8.narrow_i32x4_s(eights, eights), 0));
    const sixteens = pairSums(eights, eights);
    store<i32>(sums16 + (<usize>g << 2), i32x4.extract_lane(i16x8.narrow_i32x4_s(sixteens, sixteens), 0));
  }
}

export { scales4, dot16, sums4, sevensM, minimums4, scaled, blockOf, rowsOf };
