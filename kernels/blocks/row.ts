// blocks/row.ts (T363.4): a row of llama.cpp's blocks as float32, sb_row. blocks/types.ts has what a type is.
import { V_BITS, V_FIRST, V_THREE, V_FIFTEEN, V_ONES, K2, K4, XXS2, XS2, S2, XXS3, S3, S1, M1, XS4, Plain, Q2_K, Q4_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, sizeOf, hasMinimums, halfAt, vectorAt, magnitudes, weights, shiftsDown, scaleOf, minimumOf, wholeScale, blockMinimum } from "./types";

// ---- a row as float32: the n weights of the row at w, each as src/python/convert/superblocks.py reads it, to the
// bit (the scales in that reader's order of float32 products; W an integer). A token's row of the embedding.
// what the 16 weights h are multiplied by
// @ts-ignore: decorator
@inline function floatScale<K>(p: usize, h: i32): f32 {
  const s = <f32>(scaleOf<K>(p, h) >> 1);
  if (idof<K>() == idof<XXS2>() || idof<K>() == idof<XS2>() || idof<K>() == idof<S2>()) return halfAt(p) * (<f32>0.5 + s) * 0.25;
  if (idof<K>() == idof<XXS3>()) return halfAt(p) * (<f32>0.5 + s) * 0.5;
  if (idof<K>() == idof<M1>()) return wholeScale(p) * <f32>scaleOf<K>(p, h);
  if (idof<K>() == idof<K2>()) return halfAt(p + 80) * <f32>scaleOf<K>(p, h);
  return halfAt(p) * <f32>scaleOf<K>(p, h);
}
// @ts-ignore: decorator
@inline function rowOf<K>(out: usize, row: usize, tab: usize, n: i32, bits: v128, first: v128, three: v128, fifteen: v128): void {
  const size = sizeOf<K>();
  for (let b = 0; b < n >> 8; b++) {
    const p = row + <usize>b * size;
    for (let h = 0; h < 16; h++) {
      const to = out + (((<usize>b << 8) + (<usize>h << 4)) << 2), scale = f32x4.splat(floatScale<K>(p, h));
      let v0: v128, v1: v128, v2: v128, v3: v128;
      if (idof<K>() == idof<S1>() || idof<K>() == idof<M1>()) {  // the grid's -1, 0, 1 and an eighth more or less, as the reader adds them
        const g = i8x16.sub(magnitudes<K, Plain>(p, h, tab, three, fifteen), vectorAt(tab, V_ONES)), down = shiftsDown<K>(p, h);
        const low = i16x8.extend_low_i8x16_s(g), high = i16x8.extend_high_i8x16_s(g);
        const lowDown = i16x8.extend_low_i8x16_s(down), highDown = i16x8.extend_high_i8x16_s(down);
        const eighth = f32x4.splat(0.125), less = f32x4.splat(-0.125);
        v0 = f32x4.add(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(low)), v128.bitselect(less, eighth, i32x4.extend_low_i16x8_s(lowDown)));
        v1 = f32x4.add(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(low)), v128.bitselect(less, eighth, i32x4.extend_high_i16x8_s(lowDown)));
        v2 = f32x4.add(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(high)), v128.bitselect(less, eighth, i32x4.extend_low_i16x8_s(highDown)));
        v3 = f32x4.add(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(high)), v128.bitselect(less, eighth, i32x4.extend_high_i16x8_s(highDown)));
      } else {
        const held = weights<K, Plain>(p, h, tab, bits, first, three, fifteen);
        const low = i16x8.extend_low_i8x16_s(held), high = i16x8.extend_high_i8x16_s(held);
        v0 = f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(low));
        v1 = f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(low));
        v2 = f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(high));
        v3 = f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(high));
      }
      v0 = f32x4.mul(scale, v0);
      v1 = f32x4.mul(scale, v1);
      v2 = f32x4.mul(scale, v2);
      v3 = f32x4.mul(scale, v3);
      if (hasMinimums<K>()) {
        const least = f32x4.splat(blockMinimum<K>(p) * <f32>minimumOf<K>(p, h));
        v0 = f32x4.sub(v0, least);
        v1 = f32x4.sub(v1, least);
        v2 = f32x4.sub(v2, least);
        v3 = f32x4.sub(v3, least);
      }
      v128.store(to, v0);
      v128.store(to, v1, 16);
      v128.store(to, v2, 32);
      v128.store(to, v3, 48);
    }
  }
}
export function sb_row(out: usize, w: usize, tab: usize, n: i32, type: i32): void {
  const bits = vectorAt(tab, V_BITS), first = vectorAt(tab, V_FIRST), three = vectorAt(tab, V_THREE), fifteen = vectorAt(tab, V_FIFTEEN);
  if (type == IQ1_M) rowOf<M1>(out, w, tab, n, bits, first, three, fifteen);
  else if (type == IQ2_XXS) rowOf<XXS2>(out, w, tab, n, bits, first, three, fifteen);
  else if (type == IQ3_S) rowOf<S3>(out, w, tab, n, bits, first, three, fifteen);
  else if (type == IQ2_XS) rowOf<XS2>(out, w, tab, n, bits, first, three, fifteen);
  else if (type == IQ2_S) rowOf<S2>(out, w, tab, n, bits, first, three, fifteen);
  else if (type == IQ3_XXS) rowOf<XXS3>(out, w, tab, n, bits, first, three, fifteen);
  else if (type == IQ1_S) rowOf<S1>(out, w, tab, n, bits, first, three, fifteen);
  else if (type == Q2_K) rowOf<K2>(out, w, tab, n, bits, first, three, fifteen);
  else if (type == IQ4_XS) rowOf<XS4>(out, w, tab, n, bits, first, three, fifteen);
  else if (type == Q4_K) rowOf<K4>(out, w, tab, n, bits, first, three, fifteen);
}
