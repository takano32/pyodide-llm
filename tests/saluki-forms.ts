// tests/saluki-forms.ts (T363.1): the forms that were set against one another for the matrix product on the weights
// of Underdog Saluki 27B as its GGUF holds them: llama.cpp's blocks of 256 weights ("super-blocks": IQ2_XXS, IQ3_S,
// Q2_K, IQ1_M here; convert/superblocks.py has what each is, docs/notes/t361-saluki-2026-10-13.md the table).
// tests/saluki-kernel-bench.mjs compiles this file and times each form beside the kernels of kernels/ (matmul_t2r on
// ternary weights, matmul_q8r on int8); docs/notes/t363.1-saluki-kernel-forms-2026-10-13.md has the table and the
// choice. Not a kernel of the page: nothing here is shipped, and kernels/kernel.ts names none of it.
//
// No line of llama.cpp is here: the blocks are read as convert/superblocks.py reads them (T362), from the forms of
// ggml-common.h's structs. What is taken from llama.cpp's own kernels (ggml/src/ggml-cpu/arch/x86/quants.c at
// 4413a192410fad1f054e9216114087a3e01527a0, MIT, Copyright (c) 2023-2026 The ggml authors) is one idea: a weight's
// sign goes onto its activation, and the grid's magnitudes are the unsigned side of the dot product.
//
// A row is whole blocks of 256 weights. Every form takes one token's activations as int8 in groups of 32 with a
// float32 scale a group (xs), as the int8 kernels do.
//
// The forms of IQ2_XXS (66 bytes a block: d, then for each 32 weights four indices of 8 bits into a grid of 8
// magnitudes and a word of four 7-bit indices of signs and a 4-bit scale):
//
//   rb2, rb3  rb with fewer instructions: a group's eight bytes loaded once; an entry loaded into its lane, and the
//        scales multiplied in four groups at a time. The same numbers to the bit
//   rb   registers, the signs on the activations ("beta"): the grid's magnitudes (8, 25, 43) are the 7-bit side of the
//        relaxed dot product, the activations its signed side in all their 8 bits, each given its weight's sign
//        ((a ^ m) - m) as it is loaded. Nothing is written anywhere and a row needs no correction
//   rbp  rb without relaxed SIMD (extmul and extadd): the same numbers to the bit
//   ra   registers, the signs on the weights ("alpha"): the weights are signed int8, the activations as matmul_q8r
//        takes them (7 bits and a bias of 64), and minus 64 times the sum of a group's weights is made in the lanes
//   sb   rb in two passes: a row's magnitudes, masks and scales are written to a scratch area, then dotted
//   sa   ra in two passes: a row is widened to the int8, scales and corrections that matmul_q8r itself takes
//   any  rb's block inside the loop every type shares, the type looked at once a block (what one kernel for all the
//        types costs over one a type)
//
// and of a prompt's tokens (four at a time against one reading of a row):
//
//   rb tile   the 32 weights of a group stay in registers for four tokens; the numbers of rb to the bit
//   sa tile   four rows widened to int8 in the scratch area, then matmul_q8r_tile
//   sb tile   a row's magnitudes and masks written to the scratch area once for all the tokens, then rb tile's sums
//   sw tile   the same with the weights signed in the scratch area (as rw tile: arm64)
//   rw tile   rb tile with the signs on the weights, once a group: only where the engine's relaxed dot product
//             multiplies signed bytes by signed bytes (arm64), and then rb's numbers to the bit
//
// The other three types have rb's form alone (iq3s, q2k, iq1m): what it becomes with signs in whole bytes and a grid
// of four values (IQ3_S), with no grid and a term to take away (Q2_K), and with a grid of -1, 0, 1 shifted by an
// eighth (IQ1_M).
//
// tab: the tables, laid out by tests/saluki-kernel-bench.mjs (the grids are convert/grids.py's):
//   0      IQ2_XXS's grid: 256 entries of 8 magnitudes
//   2048   the signs of 8 weights by their 7-bit index: 128 entries of 8 bytes, 0xff a minus (the eighth sign makes
//          the number of minus signs even)
//   3072   IQ3_S's grid: 512 entries of 4 magnitudes
//   5120   IQ1_S's grid plus one: 2048 entries of 8 codes 0, 1, 2
//   21504  the vectors a loop needs, loaded once a call (written as constants V8 makes them again at every use: T163):
//          +0 bytes 1, 2, 4, .. 128 twice; +16 bytes 0 x 8, 1 x 8; +32 bytes 2 x 8, 3 x 8; +48 sixteen 3s; +64 sixteen
//          15s; +80 bytes 0, 0, 0, 0, 1, 1, 1, 1, ..; +96 the int16s 0x08, 0x80 four times; +112 eight int16s of -7;
//          +128 the int32s 512, 64, 8, 1; +144 four int32s of 7
//
// xs: n / 32 float32 scales, then (what the bench's "ready" leaves there once a token) n / 16 int16 sums of each 16
// activations and n / 8 int16 sums of each 8.

import { matmul_q8r, matmul_q8r_tile } from "../kernels/kernel_relaxed";

const T_SIGNS: usize = 2048, T_IQ3S: usize = 3072, T_IQ1S: usize = 5120, T_VECTORS: usize = 21504;
export const IQ2_XXS: i32 = 0, IQ3_S: i32 = 1, Q2_K: i32 = 2, IQ1_M: i32 = 3;

// @ts-ignore: decorator
@inline function hsum(a: v128): f32 {
  return f32x4.extract_lane(a, 0) + f32x4.extract_lane(a, 1) + f32x4.extract_lane(a, 2) + f32x4.extract_lane(a, 3);
}
// a float16's bits as float32 (finite ones: kernel/halves.ts's half)
// @ts-ignore: decorator
@inline function halfOf(bits: i32): f32 {
  return reinterpret<f32>(((bits << 16 >> 16) << 13) & <i32>0x8fffffff) * 5.192296858534828e33;
}
// one lane a vector: [sum d0, sum d1, sum d2, sum d3] (kernel_relaxed.ts's)
// @ts-ignore: decorator
@inline function groupSums(d0: v128, d1: v128, d2: v128, d3: v128): v128 {
  const s01 = i32x4.add(v128.shuffle<i32>(d0, d1, 0, 4, 1, 5), v128.shuffle<i32>(d0, d1, 2, 6, 3, 7));
  const s23 = i32x4.add(v128.shuffle<i32>(d2, d3, 0, 4, 1, 5), v128.shuffle<i32>(d2, d3, 2, 6, 3, 7));
  return i32x4.add(v128.shuffle<i32>(s01, s23, 0, 1, 4, 5), v128.shuffle<i32>(s01, s23, 2, 3, 6, 7));
}
// two entries of 8 bytes as one vector
// @ts-ignore: decorator
@inline function pair(a: usize, b: usize): v128 {
  return i64x2.replace_lane(v128.load64_zero(a), 1, load<u64>(b));
}
// the activations a with the signs of their weights: m is 0xff where the weight is negative
// @ts-ignore: decorator
@inline function signedBy(a: v128, m: v128): v128 {
  return i8x16.sub(v128.xor(a, m), m);
}
// acc plus the products of 16 signed bytes a and 16 magnitudes (0 to 127), four lanes. The same sums either way
// @ts-ignore: decorator
@inline function dotAdd(acc: v128, a: v128, magnitudes: v128, relaxed: bool): v128 {
  if (relaxed) return i32x4.relaxed_dot_i8x16_i7x16_add_s(a, magnitudes, acc);
  // two products of 127 x 127 at most fit an int16
  const products = i16x8.add(i16x8.extmul_low_i8x16_s(a, magnitudes), i16x8.extmul_high_i8x16_s(a, magnitudes));
  return i32x4.add(acc, i32x4.extadd_pairwise_i16x8_s(products));
}
// eight groups of 32's exact integer sums (four lanes each), each times its activations' scale (eight at xs): four lanes
// @ts-ignore: decorator
@inline function scaled8(g0: v128, g1: v128, g2: v128, g3: v128, g4: v128, g5: v128, g6: v128, g7: v128, xs: usize): v128 {
  return f32x4.add(f32x4.mul(f32x4.convert_i32x4_s(groupSums(g0, g1, g2, g3)), v128.load(xs)),
    f32x4.mul(f32x4.convert_i32x4_s(groupSums(g4, g5, g6, g7)), v128.load(xs, 16)));
}

// ---- IQ2_XXS
// the 32 weights at p (four indices, then the word of signs and scale): their magnitudes and masks
// @ts-ignore: decorator
@inline function xxsLow(p: usize, tab: usize): v128 {
  return pair(tab + (<usize>load<u8>(p) << 3), tab + (<usize>load<u8>(p, 1) << 3));
}
// @ts-ignore: decorator
@inline function xxsHigh(p: usize, tab: usize): v128 {
  return pair(tab + (<usize>load<u8>(p, 2) << 3), tab + (<usize>load<u8>(p, 3) << 3));
}
// @ts-ignore: decorator
@inline function xxsLowSigns(aux: u32, tab: usize): v128 {
  return pair(tab + T_SIGNS + (<usize>(aux & 127) << 3), tab + T_SIGNS + (<usize>((aux >> 7) & 127) << 3));
}
// @ts-ignore: decorator
@inline function xxsHighSigns(aux: u32, tab: usize): v128 {
  return pair(tab + T_SIGNS + (<usize>((aux >> 14) & 127) << 3), tab + T_SIGNS + (<usize>((aux >> 21) & 127) << 3));
}
// @ts-ignore: decorator
@inline function xxsScale(aux: u32): v128 {
  return i32x4.splat(<i32>((aux >> 27) | 1));  // 2 s + 1 of the top four bits s
}
// rb: the group's sum with its 32 activations at x, times its integer scale: four lanes
// @ts-ignore: decorator
@inline function xxs32(p: usize, x: usize, tab: usize, relaxed: bool): v128 {
  const aux = load<u32>(p, 4);
  let acc = dotAdd(i32x4.splat(0), signedBy(v128.load(x), xxsLowSigns(aux, tab)), xxsLow(p, tab), relaxed);
  acc = dotAdd(acc, signedBy(v128.load(x, 16), xxsHighSigns(aux, tab)), xxsHigh(p, tab), relaxed);
  return i32x4.mul(acc, xxsScale(aux));
}
// a block's part of a row's sum: four lanes
// @ts-ignore: decorator
@inline function xxsBlock(p: usize, x: usize, xs: usize, tab: usize, relaxed: bool): v128 {
  const sums = scaled8(xxs32(p + 2, x, tab, relaxed), xxs32(p + 10, x + 32, tab, relaxed), xxs32(p + 18, x + 64, tab, relaxed),
    xxs32(p + 26, x + 96, tab, relaxed), xxs32(p + 34, x + 128, tab, relaxed), xxs32(p + 42, x + 160, tab, relaxed),
    xxs32(p + 50, x + 192, tab, relaxed), xxs32(p + 58, x + 224, tab, relaxed), xs);
  return f32x4.mul(sums, f32x4.splat(halfOf(<i32>load<u16>(p)) * 0.125));
}

// ---- IQ3_S (110 bytes: d, qs[64], qh[8], signs[32], scales[4])
// four entries of 4 magnitudes: the indices are the bytes at q and bits k .. k + 3 of h above them
// @ts-ignore: decorator
@inline function fourOfFour(q: usize, h: u32, k: u32, grid: usize): v128 {
  let v = v128.load32_zero(grid + ((<usize>load<u8>(q) | <usize>((h << (8 - k)) & 256)) << 2));
  v = i32x4.replace_lane(v, 1, load<i32>(grid + ((<usize>load<u8>(q, 1) | <usize>((h << (7 - k)) & 256)) << 2)));
  v = i32x4.replace_lane(v, 2, load<i32>(grid + ((<usize>load<u8>(q, 2) | <usize>((h << (6 - k)) & 256)) << 2)));
  return i32x4.replace_lane(v, 3, load<i32>(grid + ((<usize>load<u8>(q, 3) | <usize>((h << (5 - k)) & 256)) << 2)));
}
// group g of the block at p with its 32 activations
// @ts-ignore: decorator
@inline function s32(p: usize, g: i32, x: usize, tab: usize, bits: v128, first: v128, second: v128): v128 {
  const q = p + 2 + (<usize>g << 3), h = <u32>load<u8>(p + 66 + <usize>g);
  const signs = i32x4.splat(load<i32>(p + 74 + (<usize>g << 2)));
  const m0 = i8x16.eq(v128.and(i8x16.relaxed_swizzle(signs, first), bits), bits);
  const m1 = i8x16.eq(v128.and(i8x16.relaxed_swizzle(signs, second), bits), bits);
  let acc = dotAdd(i32x4.splat(0), signedBy(v128.load(x), m0), fourOfFour(q, h, 0, tab + T_IQ3S), true);
  acc = dotAdd(acc, signedBy(v128.load(x, 16), m1), fourOfFour(q + 4, h, 4, tab + T_IQ3S), true);
  const scale = (<u32>load<u8>(p + 106 + (<usize>g >> 1)) >> ((<u32>g & 1) << 2)) & 15;
  return i32x4.mul(acc, i32x4.splat(<i32>(scale << 1 | 1)));
}
// @ts-ignore: decorator
@inline function sBlock(p: usize, x: usize, xs: usize, tab: usize, bits: v128, first: v128, second: v128): v128 {
  const sums = scaled8(s32(p, 0, x, tab, bits, first, second), s32(p, 1, x + 32, tab, bits, first, second),
    s32(p, 2, x + 64, tab, bits, first, second), s32(p, 3, x + 96, tab, bits, first, second),
    s32(p, 4, x + 128, tab, bits, first, second), s32(p, 5, x + 160, tab, bits, first, second),
    s32(p, 6, x + 192, tab, bits, first, second), s32(p, 7, x + 224, tab, bits, first, second), xs);
  return f32x4.mul(sums, f32x4.splat(halfOf(<i32>load<u16>(p))));
}

// ---- Q2_K (84 bytes: scales[16], qs[64], d, dmin). A weight is d sc q - dmin m: the first term is a dot product of
// the two bits q (not negative: the 7-bit side), the second the minimums against the sums of each 16 activations
// 32 weights: bits 2 s of the bytes of v0 are 16 of them and of v1 the next 16; sc: the two groups' scale bytes
// @ts-ignore: decorator
@inline function k32(v0: v128, v1: v128, s: i32, sc: usize, x: usize, three: v128): v128 {
  const c0 = v128.and(s == 0 ? v0 : i16x8.shr_u(v0, 2 * s), three), c1 = v128.and(s == 0 ? v1 : i16x8.shr_u(v1, 2 * s), three);
  const a = i32x4.mul(dotAdd(i32x4.splat(0), v128.load(x), c0, true), i32x4.splat(<i32>load<u8>(sc) & 15));
  const b = i32x4.mul(dotAdd(i32x4.splat(0), v128.load(x, 16), c1, true), i32x4.splat(<i32>load<u8>(sc, 1) & 15));
  return i32x4.add(a, b);
}
// sums: the int16 sums of the block's sixteen groups of 16 activations
// @ts-ignore: decorator
@inline function kBlock(p: usize, x: usize, xs: usize, sums: usize, three: v128, fifteen: v128): v128 {
  const v0 = v128.load(p, 16), v1 = v128.load(p, 32), v2 = v128.load(p, 48), v3 = v128.load(p, 64);
  const first = scaled8(k32(v0, v1, 0, p, x, three), k32(v0, v1, 1, p + 2, x + 32, three), k32(v0, v1, 2, p + 4, x + 64, three),
    k32(v0, v1, 3, p + 6, x + 96, three), k32(v2, v3, 0, p + 8, x + 128, three), k32(v2, v3, 1, p + 10, x + 160, three),
    k32(v2, v3, 2, p + 12, x + 192, three), k32(v2, v3, 3, p + 14, x + 224, three), xs);
  // the minimums (the high halves of the scales' bytes) against the sums: a lane a group of 32
  const least = v128.and(i16x8.shr_u(v128.load(p), 4), fifteen);
  const low = i32x4.dot_i16x8_s(i16x8.extend_low_i8x16_u(least), v128.load(sums));
  const high = i32x4.dot_i16x8_s(i16x8.extend_high_i8x16_u(least), v128.load(sums, 16));
  const second = f32x4.add(f32x4.mul(f32x4.convert_i32x4_s(low), v128.load(xs)), f32x4.mul(f32x4.convert_i32x4_s(high), v128.load(xs, 16)));
  return f32x4.sub(f32x4.mul(first, f32x4.splat(halfOf(<i32>load<u16>(p, 80)))), f32x4.mul(second, f32x4.splat(halfOf(<i32>load<u16>(p, 82)))));
}

// ---- IQ1_M (56 bytes: qs[32], qh[16], four words of scales and of d). A weight times 8 is 8 g + 1 or 8 g - 1 for
// the grid's g (-1, 0, 1), times d / 8 and the 2 s + 1 of its group of 16. With the codes c = g + 1 (0, 1, 2: the 7-bit
// side) a group of 8's sum is 8 dot(a, c) - 7 sum(a), or - 9 sum(a) where its shift is down
// 16 weights: the bytes at q and q + 1 and the byte h above them
// @ts-ignore: decorator
@inline function m16(q: usize, h: u32, x: usize, grid: usize): v128 {
  const codes = pair(grid + ((<usize>load<u8>(q) | <usize>((h & 7) << 8)) << 3), grid + ((<usize>load<u8>(q, 1) | <usize>((h & 0x70) << 4)) << 3));
  return dotAdd(i32x4.splat(0), v128.load(x), codes, true);
}
// 64 weights (four groups of 16) at q, their four bytes of qh at h, their word of scales: lanes 0 and 1 are the first
// 32's sum, lanes 2 and 3 the second's. sums8: the int16 sums of their eight groups of 8 activations
// @ts-ignore: decorator
@inline function m64(q: usize, h: usize, word: u32, x: usize, sums8: usize, tab: usize, spread: v128, shifts: v128, less: v128, powers: v128, seven: v128): v128 {
  const high = load<u32>(h);
  // 2 s + 1 of the four groups, a lane each
  const scales = v128.or(i32x4.shl(v128.and(i32x4.shr_u(i32x4.mul(i32x4.splat(<i32>word), powers), 9), seven), 1), i32x4.shr_u(seven, 2));
  const grid = tab + T_IQ1S;
  const d0 = i32x4.mul(m16(q, high & 255, x, grid), v128.shuffle<i32>(scales, scales, 0, 0, 0, 0));
  const d1 = i32x4.mul(m16(q + 2, (high >> 8) & 255, x + 16, grid), v128.shuffle<i32>(scales, scales, 1, 1, 1, 1));
  const d2 = i32x4.mul(m16(q + 4, (high >> 16) & 255, x + 32, grid), v128.shuffle<i32>(scales, scales, 2, 2, 2, 2));
  const d3 = i32x4.mul(m16(q + 6, high >> 24, x + 48, grid), v128.shuffle<i32>(scales, scales, 3, 3, 3, 3));
  // -7 or -9 for each group of 8, by bit 3 of its four bits of qh
  const down = i16x8.eq(v128.and(i8x16.relaxed_swizzle(i32x4.splat(<i32>high), spread), shifts), shifts);
  const away = i32x4.mul(i32x4.dot_i16x8_s(i16x8.add(less, i16x8.add(down, down)), v128.load(sums8)), scales);
  const d01 = i32x4.add(d0, d1), d23 = i32x4.add(d2, d3);
  // lanes 0 and 1: the first 32 (d01's four lanes, away's lanes 0 and 1); lanes 2 and 3: the second
  const dots = i32x4.add(v128.shuffle<i32>(d01, d23, 0, 1, 4, 5), v128.shuffle<i32>(d01, d23, 2, 3, 6, 7));
  return i32x4.add(i32x4.shl(dots, 3), away);
}
// @ts-ignore: decorator
@inline function mBlock(p: usize, x: usize, xs: usize, sums8: usize, tab: usize, spread: v128, shifts: v128, less: v128, powers: v128, seven: v128): v128 {
  const w0 = <u32>load<u16>(p, 48), w1 = <u32>load<u16>(p, 50), w2 = <u32>load<u16>(p, 52), w3 = <u32>load<u16>(p, 54);
  const d = halfOf(<i32>((w0 >> 12) | ((w1 >> 8) & 0xf0) | ((w2 >> 4) & 0xf00) | (w3 & 0xf000)));
  const a = m64(p, p + 32, w0, x, sums8, tab, spread, shifts, less, powers, seven);
  const b = m64(p + 8, p + 36, w1, x + 64, sums8 + 16, tab, spread, shifts, less, powers, seven);
  const c = m64(p + 16, p + 40, w2, x + 128, sums8 + 32, tab, spread, shifts, less, powers, seven);
  const e = m64(p + 24, p + 44, w3, x + 192, sums8 + 48, tab, spread, shifts, less, powers, seven);
  // the sums of 32 are two lanes each: [a01 a23 b01 b23], [c01 c23 e01 e23]
  const low = i32x4.add(v128.shuffle<i32>(a, b, 0, 2, 4, 6), v128.shuffle<i32>(a, b, 1, 3, 5, 7));
  const high = i32x4.add(v128.shuffle<i32>(c, e, 0, 2, 4, 6), v128.shuffle<i32>(c, e, 1, 3, 5, 7));
  const sums = f32x4.add(f32x4.mul(f32x4.convert_i32x4_s(low), v128.load(xs)), f32x4.mul(f32x4.convert_i32x4_s(high), v128.load(xs, 16)));
  return f32x4.mul(sums, f32x4.splat(d * 0.125));
}

// ---- the loop every type shares. kind is a constant where a kernel is of one type, and then the other types' blocks
// are not in it; in matmul_any it is an argument, looked at once a block
// @ts-ignore: decorator
@inline function sizeOf(kind: i32): usize {
  return kind == IQ2_XXS ? 66 : kind == IQ3_S ? 110 : kind == Q2_K ? 84 : 56;
}
// @ts-ignore: decorator
@inline function rowsOf(kind: i32, xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, relaxed: bool): void {
  const blocks = n >> 8, size = sizeOf(kind);
  const sums16 = xs + (<usize>(n >> 5) << 2), sums8 = sums16 + (<usize>(n >> 4) << 1);
  const at = tab + T_VECTORS;
  const bits = v128.load(at), first = v128.load(at, 16), second = v128.load(at, 32), three = v128.load(at, 48), fifteen = v128.load(at, 64);
  const spread = v128.load(at, 80), shifts = v128.load(at, 96), less = v128.load(at, 112), powers = v128.load(at, 128), seven = v128.load(at, 144);
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * <usize>blocks * size;
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const p = row + <usize>b * size, x = xq + (<usize>b << 8), s = xs + (<usize>b << 5);
      let part: v128;
      if (kind == IQ2_XXS) part = xxsBlock(p, x, s, tab, relaxed);
      else if (kind == IQ3_S) part = sBlock(p, x, s, tab, bits, first, second);
      else if (kind == Q2_K) part = kBlock(p, x, s, sums16 + (<usize>b << 5), three, fifteen);
      else part = mBlock(p, x, s, sums8 + (<usize>b << 6), tab, spread, shifts, less, powers, seven);
      facc = f32x4.add(facc, part);
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}
export function matmul_iq2xxs(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32): void {
  rowsOf(IQ2_XXS, xout, xq, xs, w, tab, n, r0, r1, true);
}
export function matmul_iq2xxs_plain(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32): void {
  rowsOf(IQ2_XXS, xout, xq, xs, w, tab, n, r0, r1, false);
}
export function matmul_iq3s(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32): void {
  rowsOf(IQ3_S, xout, xq, xs, w, tab, n, r0, r1, true);
}
export function matmul_q2k(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32): void {
  rowsOf(Q2_K, xout, xq, xs, w, tab, n, r0, r1, true);
}
export function matmul_iq1m(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32): void {
  rowsOf(IQ1_M, xout, xq, xs, w, tab, n, r0, r1, true);
}
export function matmul_any(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, kind: i32): void {
  rowsOf(kind, xout, xq, xs, w, tab, n, r0, r1, true);
}

// ---- rb2: rb with a group's eight bytes loaded once (the four indices and the word), not five times
// @ts-ignore: decorator
@inline function xxs32once(p: usize, x: usize, tab: usize): v128 {
  const v = load<u64>(p), aux = <u32>(v >> 32), signs = tab + T_SIGNS;
  const low = pair(tab + (<usize>(v & 255) << 3), tab + (<usize>((v >> 8) & 255) << 3));
  const high = pair(tab + (<usize>((v >> 16) & 255) << 3), tab + (<usize>((v >> 24) & 255) << 3));
  const m0 = pair(signs + (<usize>(aux & 127) << 3), signs + (<usize>((aux >> 7) & 127) << 3));
  const m1 = pair(signs + (<usize>((aux >> 14) & 127) << 3), signs + (<usize>((aux >> 21) & 127) << 3));
  let acc = dotAdd(i32x4.splat(0), signedBy(v128.load(x), m0), low, true);
  acc = dotAdd(acc, signedBy(v128.load(x, 16), m1), high, true);
  return i32x4.mul(acc, xxsScale(aux));
}
export function matmul_iq2xxs_once(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32): void {
  const blocks = n >> 8;
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * <usize>blocks * 66;
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const p = row + <usize>b * 66, x = xq + (<usize>b << 8);
      const sums = scaled8(xxs32once(p + 2, x, tab), xxs32once(p + 10, x + 32, tab), xxs32once(p + 18, x + 64, tab), xxs32once(p + 26, x + 96, tab),
        xxs32once(p + 34, x + 128, tab), xxs32once(p + 42, x + 160, tab), xxs32once(p + 50, x + 192, tab), xxs32once(p + 58, x + 224, tab), xs + (<usize>b << 5));
      facc = f32x4.add(facc, f32x4.mul(sums, f32x4.splat(halfOf(<i32>load<u16>(p)) * 0.125)));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// ---- rb3: rb with an entry loaded into its lane in one instruction, and the eight scales of a block taken from its
// words four at a time (the words are every other int32 of the block from its byte 2) and multiplied into the groups'
// sums once they are a lane each: 2 s sums + sums, the same integers as rb's
// @ts-ignore: decorator
@inline function pairLane(a: usize, b: usize): v128 {
  return v128.load64_lane(b, v128.load64_zero(a), 1);
}
// @ts-ignore: decorator
@inline function xxsDot(p: usize, x: usize, tab: usize): v128 {
  const aux = load<u32>(p, 4), signs = tab + T_SIGNS;
  const low = pairLane(tab + (<usize>load<u8>(p) << 3), tab + (<usize>load<u8>(p, 1) << 3));
  const high = pairLane(tab + (<usize>load<u8>(p, 2) << 3), tab + (<usize>load<u8>(p, 3) << 3));
  const m0 = pairLane(signs + (<usize>(aux & 127) << 3), signs + (<usize>((aux >> 7) & 127) << 3));
  const m1 = pairLane(signs + (<usize>((aux >> 14) & 127) << 3), signs + (<usize>((aux >> 21) & 127) << 3));
  return dotAdd(dotAdd(i32x4.splat(0), signedBy(v128.load(x), m0), low, true), signedBy(v128.load(x, 16), m1), high, true);
}
// four groups' sums (a lane each) times the 2 s + 1 of the four words in a and b
// @ts-ignore: decorator
@inline function timesScales(sums: v128, a: v128, b: v128): v128 {
  return i32x4.add(i32x4.mul(sums, i32x4.shl(i32x4.shr_u(v128.shuffle<i32>(a, b, 1, 3, 5, 7), 28), 1)), sums);
}
export function matmul_iq2xxs_lanes(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32): void {
  const blocks = n >> 8;
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * <usize>blocks * 66;
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const p = row + <usize>b * 66, x = xq + (<usize>b << 8), s = xs + (<usize>b << 5);
      const low = groupSums(xxsDot(p + 2, x, tab), xxsDot(p + 10, x + 32, tab), xxsDot(p + 18, x + 64, tab), xxsDot(p + 26, x + 96, tab));
      const high = groupSums(xxsDot(p + 34, x + 128, tab), xxsDot(p + 42, x + 160, tab), xxsDot(p + 50, x + 192, tab), xxsDot(p + 58, x + 224, tab));
      const sums = f32x4.add(f32x4.mul(f32x4.convert_i32x4_s(timesScales(low, v128.load(p, 2), v128.load(p, 18))), v128.load(s)),
        f32x4.mul(f32x4.convert_i32x4_s(timesScales(high, v128.load(p, 34), v128.load(p, 50))), v128.load(s, 16)));
      facc = f32x4.add(facc, f32x4.mul(sums, f32x4.splat(halfOf(<i32>load<u16>(p)) * 0.125)));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// ---- ra: the signs on the weights, the activations of matmul_q8r (7 bits, a bias of 64)
// @ts-ignore: decorator
@inline function xxs32a(p: usize, x: usize, tab: usize): v128 {
  const aux = load<u32>(p, 4);
  const w0 = signedBy(xxsLow(p, tab), xxsLowSigns(aux, tab)), w1 = signedBy(xxsHigh(p, tab), xxsHighSigns(aux, tab));
  let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(w0, v128.load(x), i32x4.splat(0));
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(w1, v128.load(x, 16), acc);
  // minus 64 times the weights' sum, in the lanes
  const sum = i32x4.extadd_pairwise_i16x8_s(i16x8.add(i16x8.extadd_pairwise_i8x16_s(w0), i16x8.extadd_pairwise_i8x16_s(w1)));
  return i32x4.mul(i32x4.sub(acc, i32x4.shl(sum, 6)), xxsScale(aux));
}
export function matmul_iq2xxs_ra(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32): void {
  const blocks = n >> 8;
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * <usize>blocks * 66;
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const p = row + <usize>b * 66, x = xq + (<usize>b << 8);
      const sums = scaled8(xxs32a(p + 2, x, tab), xxs32a(p + 10, x + 32, tab), xxs32a(p + 18, x + 64, tab), xxs32a(p + 26, x + 96, tab),
        xxs32a(p + 34, x + 128, tab), xxs32a(p + 42, x + 160, tab), xxs32a(p + 50, x + 192, tab), xxs32a(p + 58, x + 224, tab), xs + (<usize>b << 5));
      facc = f32x4.add(facc, f32x4.mul(sums, f32x4.splat(halfOf(<i32>load<u16>(p)) * 0.125)));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// ---- sa: a row widened to what matmul_q8r takes: its n signed int8 at wq, a float32 scale a group of 32 at ws and
// an int32 correction a group (-64 times the sum of its weights) at wc
// @ts-ignore: decorator
@inline function widen32(p: usize, tab: usize, wq: usize): v128 {  // the lanes' sum is the sum of the 32 weights
  const aux = load<u32>(p, 4);
  const w0 = signedBy(xxsLow(p, tab), xxsLowSigns(aux, tab)), w1 = signedBy(xxsHigh(p, tab), xxsHighSigns(aux, tab));
  v128.store(wq, w0);
  v128.store(wq, w1, 16);
  return i32x4.extadd_pairwise_i16x8_s(i16x8.add(i16x8.extadd_pairwise_i8x16_s(w0), i16x8.extadd_pairwise_i8x16_s(w1)));
}
function widenRow(row: usize, tab: usize, n: i32, wq: usize, ws: usize, wc: usize): void {
  const blocks = n >> 8;
  for (let b = 0; b < blocks; b++) {
    const p = row + <usize>b * 66, d = halfOf(<i32>load<u16>(p)) * 0.125;
    for (let half = 0; half < 2; half++) {
      const q = p + 2 + (<usize>half << 5), o = wq + (<usize>b << 8) + (<usize>half << 7), g = (<usize>b << 5) + (<usize>half << 4);
      const sums = groupSums(widen32(q, tab, o), widen32(q + 8, tab, o + 32), widen32(q + 16, tab, o + 64), widen32(q + 24, tab, o + 96));
      v128.store(wc + g, i32x4.neg(i32x4.shl(sums, 6)));
      store<f32>(ws + g, d * <f32>((load<u32>(q, 4) >> 27) | 1));
      store<f32>(ws + g, d * <f32>((load<u32>(q, 12) >> 27) | 1), 4);
      store<f32>(ws + g, d * <f32>((load<u32>(q, 20) >> 27) | 1), 8);
      store<f32>(ws + g, d * <f32>((load<u32>(q, 28) >> 27) | 1), 12);
    }
  }
}
// scratch: n + n / 4 bytes of the caller's (the thread's) own
export function matmul_iq2xxs_sa(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, scratch: usize): void {
  const rowBytes = <usize>(n >> 8) * 66, ws = scratch + <usize>n, wc = ws + (<usize>n >> 3);
  for (let i = r0; i < r1; i++) {
    widenRow(w + <usize>i * rowBytes, tab, n, scratch, ws, wc);
    matmul_q8r(xout + (<usize>i << 2), xq, xs, scratch, ws, wc, n, 0, 1);
  }
}
// sa tile: four rows widened, then int8's tile (T159) on them. scratch: 4 (n + n / 4) bytes
export function matmul_iq2xxs_sa_tile(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, scratch: usize): void {
  const rowBytes = <usize>(n >> 8) * 66, groups = <usize>n >> 3, ws = scratch + 4 * <usize>n, wc = ws + 4 * groups;
  for (let i = r0; i < r1; i += 4) {
    const rows = min(4, r1 - i);
    for (let r = 0; r < rows; r++) widenRow(w + <usize>(i + r) * rowBytes, tab, n, scratch + <usize>r * <usize>n, ws + <usize>r * groups, wc + <usize>r * groups);
    matmul_q8r_tile(xout + (<usize>i << 2), xq, xs, scratch, ws, wc, n, 0, rows, count, os, frame);
  }
}

// ---- sb: rb in two passes. scratch: a row's n magnitudes, its n masks, and n / 32 float32 scales
export function matmul_iq2xxs_sb(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, scratch: usize): void {
  const blocks = n >> 8, masks = scratch + <usize>n, ws = masks + <usize>n;
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * <usize>blocks * 66;
    for (let b = 0; b < blocks; b++) {
      const p = row + <usize>b * 66, d = halfOf(<i32>load<u16>(p)) * 0.125;
      for (let g = 0; g < 8; g++) {
        const q = p + 2 + (<usize>g << 3), o = (<usize>b << 8) + (<usize>g << 5), aux = load<u32>(q, 4);
        v128.store(scratch + o, xxsLow(q, tab));
        v128.store(scratch + o, xxsHigh(q, tab), 16);
        v128.store(masks + o, xxsLowSigns(aux, tab));
        v128.store(masks + o, xxsHighSigns(aux, tab), 16);
        store<f32>(ws + (o >> 3), d * <f32>((aux >> 27) | 1));
      }
    }
    let facc = f32x4.splat(0);
    for (let g = 0; g < n >> 5; g += 4) {
      const o = <usize>g << 5, x = xq + o, m = masks + o, a = scratch + o;
      const d0 = dotAdd(dotAdd(i32x4.splat(0), signedBy(v128.load(x), v128.load(m)), v128.load(a), true), signedBy(v128.load(x, 16), v128.load(m, 16)), v128.load(a, 16), true);
      const d1 = dotAdd(dotAdd(i32x4.splat(0), signedBy(v128.load(x, 32), v128.load(m, 32)), v128.load(a, 32), true), signedBy(v128.load(x, 48), v128.load(m, 48)), v128.load(a, 48), true);
      const d2 = dotAdd(dotAdd(i32x4.splat(0), signedBy(v128.load(x, 64), v128.load(m, 64)), v128.load(a, 64), true), signedBy(v128.load(x, 80), v128.load(m, 80)), v128.load(a, 80), true);
      const d3 = dotAdd(dotAdd(i32x4.splat(0), signedBy(v128.load(x, 96), v128.load(m, 96)), v128.load(a, 96), true), signedBy(v128.load(x, 112), v128.load(m, 112)), v128.load(a, 112), true);
      const scales = f32x4.mul(v128.load(ws + (<usize>g << 2)), v128.load(xs + (<usize>g << 2)));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupSums(d0, d1, d2, d3)), scales));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// ---- rb tile: a group's 32 weights in registers for four tokens (their activations frame apart, their outputs os
// apart). A lane a token; every number is matmul_iq2xxs's to the bit: a block's groups k and k + 4 are added, times d,
// into accumulator k, and the four accumulators are added in hsum's order
// @ts-ignore: decorator
@inline function scalesOf(xs: usize, frame: usize): v128 {  // the four tokens' scales of the group at xs
  let v = v128.load32_zero(xs);
  v = f32x4.replace_lane(v, 1, load<f32>(xs + frame));
  v = f32x4.replace_lane(v, 2, load<f32>(xs + 2 * frame));
  return f32x4.replace_lane(v, 3, load<f32>(xs + 3 * frame));
}
// @ts-ignore: decorator
@inline function xxs32x4(p: usize, x: usize, xs: usize, frame: usize, tab: usize): v128 {
  const aux = load<u32>(p, 4);
  const low = xxsLow(p, tab), high = xxsHigh(p, tab), m0 = xxsLowSigns(aux, tab), m1 = xxsHighSigns(aux, tab);
  const x1 = x + frame, x2 = x1 + frame, x3 = x2 + frame, zero = i32x4.splat(0);
  const d0 = dotAdd(dotAdd(zero, signedBy(v128.load(x), m0), low, true), signedBy(v128.load(x, 16), m1), high, true);
  const d1 = dotAdd(dotAdd(zero, signedBy(v128.load(x1), m0), low, true), signedBy(v128.load(x1, 16), m1), high, true);
  const d2 = dotAdd(dotAdd(zero, signedBy(v128.load(x2), m0), low, true), signedBy(v128.load(x2, 16), m1), high, true);
  const d3 = dotAdd(dotAdd(zero, signedBy(v128.load(x3), m0), low, true), signedBy(v128.load(x3, 16), m1), high, true);
  return f32x4.mul(f32x4.convert_i32x4_s(i32x4.mul(groupSums(d0, d1, d2, d3), xxsScale(aux))), scalesOf(xs, frame));
}
export function matmul_iq2xxs_tile(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32): void {
  const blocks = n >> 8, f = <usize>frame;
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * <usize>blocks * 66;
    let t = 0;
    for (; t + 4 <= count; t += 4) {
      const xt = xq + <usize>t * f, st = xs + <usize>t * f;
      let a0 = f32x4.splat(0), a1 = a0, a2 = a0, a3 = a0;
      for (let b = 0; b < blocks; b++) {
        const p = row + <usize>b * 66, x = xt + (<usize>b << 8), s = st + (<usize>b << 5);
        const d = f32x4.splat(halfOf(<i32>load<u16>(p)) * 0.125);
        const p0 = xxs32x4(p + 2, x, s, f, tab), p1 = xxs32x4(p + 10, x + 32, s + 4, f, tab);
        const p2 = xxs32x4(p + 18, x + 64, s + 8, f, tab), p3 = xxs32x4(p + 26, x + 96, s + 12, f, tab);
        a0 = f32x4.add(a0, f32x4.mul(f32x4.add(p0, xxs32x4(p + 34, x + 128, s + 16, f, tab)), d));
        a1 = f32x4.add(a1, f32x4.mul(f32x4.add(p1, xxs32x4(p + 42, x + 160, s + 20, f, tab)), d));
        a2 = f32x4.add(a2, f32x4.mul(f32x4.add(p2, xxs32x4(p + 50, x + 192, s + 24, f, tab)), d));
        a3 = f32x4.add(a3, f32x4.mul(f32x4.add(p3, xxs32x4(p + 58, x + 224, s + 28, f, tab)), d));
      }
      const sums = f32x4.add(f32x4.add(f32x4.add(a0, a1), a2), a3);
      const out = xout + <usize>t * <usize>os + (<usize>i << 2);
      store<f32>(out, f32x4.extract_lane(sums, 0));
      store<f32>(out + <usize>os, f32x4.extract_lane(sums, 1));
      store<f32>(out + 2 * <usize>os, f32x4.extract_lane(sums, 2));
      store<f32>(out + 3 * <usize>os, f32x4.extract_lane(sums, 3));
    }
    for (; t < count; t++) rowsOf(IQ2_XXS, xout + <usize>t * <usize>os, xq + <usize>t * f, xs + <usize>t * f, w, tab, n, i, i + 1, true);
  }
}

// ---- rw tile: rb tile with the signs put on the weights once a group, not on every token's activations: the dot
// product is then of signed bytes by signed bytes in all their 8 bits, which relaxed SIMD does not promise (its second
// operand is 7 bits; arm64's SDOT multiplies signed by signed all the same, x86's PMADDUBSW takes the second as
// unsigned). signedDot() says which this engine does: where it returns 1 the tile's numbers are rb's to the bit
export function signedDot(): i32 {
  const dots = i32x4.relaxed_dot_i8x16_i7x16_add_s(i8x16.splat(-5), i8x16.splat(-3), i32x4.splat(0));
  return <i32>(i32x4.extract_lane(dots, 0) == 60);
}
// @ts-ignore: decorator
@inline function xxs32x4w(p: usize, x: usize, xs: usize, frame: usize, tab: usize): v128 {
  const aux = load<u32>(p, 4);
  const low = signedBy(xxsLow(p, tab), xxsLowSigns(aux, tab)), high = signedBy(xxsHigh(p, tab), xxsHighSigns(aux, tab));
  const x1 = x + frame, x2 = x1 + frame, x3 = x2 + frame, zero = i32x4.splat(0);
  const d0 = dotAdd(dotAdd(zero, v128.load(x), low, true), v128.load(x, 16), high, true);
  const d1 = dotAdd(dotAdd(zero, v128.load(x1), low, true), v128.load(x1, 16), high, true);
  const d2 = dotAdd(dotAdd(zero, v128.load(x2), low, true), v128.load(x2, 16), high, true);
  const d3 = dotAdd(dotAdd(zero, v128.load(x3), low, true), v128.load(x3, 16), high, true);
  return f32x4.mul(f32x4.convert_i32x4_s(i32x4.mul(groupSums(d0, d1, d2, d3), xxsScale(aux))), scalesOf(xs, frame));
}
export function matmul_iq2xxs_tile_signed(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32): void {
  const blocks = n >> 8, f = <usize>frame;
  for (let i = r0; i < r1; i++) {
    const row = w + <usize>i * <usize>blocks * 66;
    for (let t = 0; t + 4 <= count; t += 4) {
      const xt = xq + <usize>t * f, st = xs + <usize>t * f;
      let a0 = f32x4.splat(0), a1 = a0, a2 = a0, a3 = a0;
      for (let b = 0; b < blocks; b++) {
        const p = row + <usize>b * 66, x = xt + (<usize>b << 8), s = st + (<usize>b << 5);
        const d = f32x4.splat(halfOf(<i32>load<u16>(p)) * 0.125);
        const p0 = xxs32x4w(p + 2, x, s, f, tab), p1 = xxs32x4w(p + 10, x + 32, s + 4, f, tab);
        const p2 = xxs32x4w(p + 18, x + 64, s + 8, f, tab), p3 = xxs32x4w(p + 26, x + 96, s + 12, f, tab);
        a0 = f32x4.add(a0, f32x4.mul(f32x4.add(p0, xxs32x4w(p + 34, x + 128, s + 16, f, tab)), d));
        a1 = f32x4.add(a1, f32x4.mul(f32x4.add(p1, xxs32x4w(p + 42, x + 160, s + 20, f, tab)), d));
        a2 = f32x4.add(a2, f32x4.mul(f32x4.add(p2, xxs32x4w(p + 50, x + 192, s + 24, f, tab)), d));
        a3 = f32x4.add(a3, f32x4.mul(f32x4.add(p3, xxs32x4w(p + 58, x + 224, s + 28, f, tab)), d));
      }
      const sums = f32x4.add(f32x4.add(f32x4.add(a0, a1), a2), a3);
      const out = xout + <usize>t * <usize>os + (<usize>i << 2);
      store<f32>(out, f32x4.extract_lane(sums, 0));
      store<f32>(out + <usize>os, f32x4.extract_lane(sums, 1));
      store<f32>(out + 2 * <usize>os, f32x4.extract_lane(sums, 2));
      store<f32>(out + 3 * <usize>os, f32x4.extract_lane(sums, 3));
    }
  }
}

// ---- sb tile and sw tile: a row spread once into the scratch area for all the tokens of a prompt (the tiles above
// look a group up again for every four tokens), then four tokens at a time as rb tile takes them. sb: the magnitudes
// and the masks, as rb; sw: the weights signed, as rw (arm64). The numbers are rb's to the bit.
// scratch: n magnitudes or weights, n masks, an int32 scale a group of 32, a float32 d / 8 a block
// @ts-ignore: decorator
@inline function spreadRow(row: usize, tab: usize, n: i32, scratch: usize, signed: bool): void {
  const masks = scratch + <usize>n, scales = masks + <usize>n, ds = scales + (<usize>n >> 3);
  for (let b = 0; b < n >> 8; b++) {
    const p = row + <usize>b * 66;
    store<f32>(ds + (<usize>b << 2), halfOf(<i32>load<u16>(p)) * 0.125);
    for (let g = 0; g < 8; g++) {
      const q = p + 2 + (<usize>g << 3), o = (<usize>b << 8) + (<usize>g << 5), aux = load<u32>(q, 4);
      const low = xxsLow(q, tab), high = xxsHigh(q, tab), m0 = xxsLowSigns(aux, tab), m1 = xxsHighSigns(aux, tab);
      if (signed) {
        v128.store(scratch + o, signedBy(low, m0));
        v128.store(scratch + o, signedBy(high, m1), 16);
      } else {
        v128.store(scratch + o, low);
        v128.store(scratch + o, high, 16);
        v128.store(masks + o, m0);
        v128.store(masks + o, m1, 16);
      }
      store<i32>(scales + (o >> 3), <i32>((aux >> 27) | 1));
    }
  }
}
// @ts-ignore: decorator
@inline function spread32x4(a: usize, m: usize, scale: i32, x: usize, xs: usize, frame: usize, signed: bool): v128 {
  const low = v128.load(a), high = v128.load(a, 16), zero = i32x4.splat(0);
  const x1 = x + frame, x2 = x1 + frame, x3 = x2 + frame;
  let d0: v128, d1: v128, d2: v128, d3: v128;
  if (signed) {
    d0 = dotAdd(dotAdd(zero, v128.load(x), low, true), v128.load(x, 16), high, true);
    d1 = dotAdd(dotAdd(zero, v128.load(x1), low, true), v128.load(x1, 16), high, true);
    d2 = dotAdd(dotAdd(zero, v128.load(x2), low, true), v128.load(x2, 16), high, true);
    d3 = dotAdd(dotAdd(zero, v128.load(x3), low, true), v128.load(x3, 16), high, true);
  } else {
    const m0 = v128.load(m), m1 = v128.load(m, 16);
    d0 = dotAdd(dotAdd(zero, signedBy(v128.load(x), m0), low, true), signedBy(v128.load(x, 16), m1), high, true);
    d1 = dotAdd(dotAdd(zero, signedBy(v128.load(x1), m0), low, true), signedBy(v128.load(x1, 16), m1), high, true);
    d2 = dotAdd(dotAdd(zero, signedBy(v128.load(x2), m0), low, true), signedBy(v128.load(x2, 16), m1), high, true);
    d3 = dotAdd(dotAdd(zero, signedBy(v128.load(x3), m0), low, true), signedBy(v128.load(x3, 16), m1), high, true);
  }
  return f32x4.mul(f32x4.convert_i32x4_s(i32x4.mul(groupSums(d0, d1, d2, d3), i32x4.splat(scale))), scalesOf(xs, frame));
}
// @ts-ignore: decorator
@inline function spreadTile(signed: bool, xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, scratch: usize): void {
  const blocks = n >> 8, f = <usize>frame;
  const masks = scratch + <usize>n, scales = masks + <usize>n, ds = scales + (<usize>n >> 3);
  for (let i = r0; i < r1; i++) {
    spreadRow(w + <usize>i * <usize>blocks * 66, tab, n, scratch, signed);
    for (let t = 0; t + 4 <= count; t += 4) {
      const xt = xq + <usize>t * f, st = xs + <usize>t * f;
      let a0 = f32x4.splat(0), a1 = a0, a2 = a0, a3 = a0;
      for (let b = 0; b < blocks; b++) {
        const o = <usize>b << 8, a = scratch + o, m = masks + o, c = scales + (<usize>b << 5), x = xt + o, s = st + (<usize>b << 5);
        const d = f32x4.splat(load<f32>(ds + (<usize>b << 2)));
        const p0 = spread32x4(a, m, load<i32>(c), x, s, f, signed), p1 = spread32x4(a + 32, m + 32, load<i32>(c, 4), x + 32, s + 4, f, signed);
        const p2 = spread32x4(a + 64, m + 64, load<i32>(c, 8), x + 64, s + 8, f, signed), p3 = spread32x4(a + 96, m + 96, load<i32>(c, 12), x + 96, s + 12, f, signed);
        a0 = f32x4.add(a0, f32x4.mul(f32x4.add(p0, spread32x4(a + 128, m + 128, load<i32>(c, 16), x + 128, s + 16, f, signed)), d));
        a1 = f32x4.add(a1, f32x4.mul(f32x4.add(p1, spread32x4(a + 160, m + 160, load<i32>(c, 20), x + 160, s + 20, f, signed)), d));
        a2 = f32x4.add(a2, f32x4.mul(f32x4.add(p2, spread32x4(a + 192, m + 192, load<i32>(c, 24), x + 192, s + 24, f, signed)), d));
        a3 = f32x4.add(a3, f32x4.mul(f32x4.add(p3, spread32x4(a + 224, m + 224, load<i32>(c, 28), x + 224, s + 28, f, signed)), d));
      }
      const sums = f32x4.add(f32x4.add(f32x4.add(a0, a1), a2), a3);
      const out = xout + <usize>t * <usize>os + (<usize>i << 2);
      store<f32>(out, f32x4.extract_lane(sums, 0));
      store<f32>(out + <usize>os, f32x4.extract_lane(sums, 1));
      store<f32>(out + 2 * <usize>os, f32x4.extract_lane(sums, 2));
      store<f32>(out + 3 * <usize>os, f32x4.extract_lane(sums, 3));
    }
  }
}
export function matmul_iq2xxs_sb_tile(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, scratch: usize): void {
  spreadTile(false, xout, xq, xs, w, tab, n, r0, r1, count, os, frame, scratch);
}
export function matmul_iq2xxs_sw_tile(xout: usize, xq: usize, xs: usize, w: usize, tab: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32, scratch: usize): void {
  spreadTile(true, xout, xq, xs, w, tab, n, r0, r1, count, os, frame, scratch);
}
