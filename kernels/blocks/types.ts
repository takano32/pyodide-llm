// blocks/types.ts (T363.4): what each of llama.cpp's blocks of 256 weights ("super-blocks") is to the kernels of
// blocks.ts and blocks_relaxed.ts: Q2_K, Q4_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS, the
// ten types Underdog Saluki 27B is made of. The weights stay packed as the GGUF has them (2.3 bits each, on average);
// what a dot product takes is made of them in registers as they are read. The kernels:
//
//   sb_tables(area, codebooks)                              the tables the others read, made once a model (here)
//   sums_x(xs, xq, n)                                       what a token's activations need after quantize_x(.., 0)
//   matmul_sb, matmul_sbr (xout, xq, xs, w, tab, n, r0, r1, type)                        one token, rows r0 to r1
//   matmul_sbr_tile (xout, xq, xs, w, tab, n, r0, r1, count, os, frame, type)            a prompt's tokens
//   matmul_sbw_tile (.., type, scratch)                     the same, where the engine multiplies signed by signed
//   signed_dots(out, a, b)                                  the question matmul_sbw_tile is chosen by
//   sb_row(out, w, tab, n, type)                            a row as float32 (a token's row of the embedding)
//
// type: the type's number in a GGUF (ggml's: Q2_K 10, Q4_K 12, IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ1_S 19, IQ3_S 21,
// IQ2_S 22, IQ4_XS 23, IQ1_M 29). A row is whole blocks: n is a multiple of 256. tab: sb_tables's area. Every address
// may be above 4 GiB on a 64-bit memory (the 27B's weights are): the loads are v128.load, load64_zero, load32_zero and
// load64_lane, which V8's Liftoff on arm64 reads right there (T233: it does not read load32_splat and load32_lane
// right; tests/blocks-check.mjs runs every kernel as Liftoff alone compiles it).
//
// No line of llama.cpp is here. The blocks are read as src/python/convert/superblocks.py reads them (T362), from the
// forms of ggml/src/ggml-common.h's block_* structs, and tests/blocks-check.mjs holds every type's reading here to
// that reader's (to the bit, through tests/blocks-model.mjs). What is taken from llama.cpp's own kernels (ggml/src/ggml-cpu/arch/x86/quants.c at the commit
// 4413a192410fad1f054e9216114087a3e01527a0 of https://github.com/ggml-org/llama.cpp, MIT, Copyright (c) 2023-2026
// The ggml authors) is one idea: a weight's sign goes onto its activation, and the grid's magnitudes are the unsigned
// side of the dot product. The codebooks are llama.cpp's tables (src/python/convert/grids.py has them and their
// notice); they are in the checkpoint, not in this module.
//
// ---- What every type is, to these kernels. A weight is F S W - Fm M: F a float32 of its block (the float16 d times
// a power of two), S a small integer of its unit (16 weights, or 32: the type's "unit"), W a small integer of its own,
// and for the two K types a minimum M of its unit times the block's float16 dmin.
//
//   type      bytes  unit  W                         S                F       how W is held
//   IQ2_XXS   66     32    +-(8, 25, 43)             2 s + 1 (4 bits) d / 8   8 a byte into a grid of 256; 7 bits of signs
//   IQ2_XS    74     16    the same                  2 s + 1 (4 bits) d / 8   8 a word: 9 bits into a grid of 512, 7 of signs
//   IQ2_S     82     16    the same                  2 s + 1 (4 bits) d / 8   8 by 10 bits into a grid of 1024; a byte of signs
//   IQ3_XXS   98     32    +-(4, 12, .. 62)          2 s + 1 (4 bits) d / 4   4 a byte into a grid of 256; 7 bits of signs for 8
//   IQ3_S     110    32    +-(1, 3, .. 15)           2 s + 1 (4 bits) d       4 by 9 bits into a grid of 512; a byte of signs for 8
//   IQ1_S     50     32    8 g + 1 or 8 g - 1        2 s + 1 (3 bits) d / 8   8 by 11 bits into a grid of -1, 0, 1; the shift's sign a unit
//   IQ1_M     56     16    the same                  2 s + 1 (3 bits) d / 8   the same; the shift's sign for each 8; d in four words' tops
//   IQ4_XS    136    32    one of 16 from -127 to 113  ls - 32 (6 bits)  d    4 bits
//   Q2_K      84     16    0 to 3                    4 bits           d       2 bits; M 4 bits
//   Q4_K      144    32    0 to 15                   6 bits           d       4 bits; M 6 bits
//
// ---- The arithmetic, which is one for the token's kernel and the tiles (so a prompt read in blocks leaves the same
// bits as its tokens read one by one). The activations are int8 in groups of 32 with a float32 scale each, as
// quantize_x(.., 0) leaves them (-127 to 127). For each block and each of its eight groups k of 32 activations:
//   G[k] = the sum over the group of S W a: an integer, exact in int32 and in float32 (IQ4_XS is the largest:
//          32 x 127 x 127 x 32 = 16.5 million, under 2^24)
//   lane k & 3 of the row's accumulator += (G[k] xs[k] + G[k + 4] xs[k + 4]) F     (float32, in that order)
//   and for the K types, with M[k] the sum of the group's M times the sums of its units' activations:
//   lane k & 3 += (G[k] xs[k] + G[k + 4] xs[k + 4]) F - (M[k] xs[k] + M[k + 4] xs[k + 4]) Fm
// and the four lanes are added in order at the end of the row. The integers are the same however they are summed
// (with relaxed SIMD or without, a token or four, the signs on the activations or on the weights), and the float32
// steps are written in this order everywhere: tests/blocks-check.mjs follows them with Math.fround.
//
// The token's kernel has T363.1's form "rb3" (docs/notes/t363.1-saluki-kernel-forms-2026-10-13.md): the magnitudes
// |W| are the 7-bit side of the relaxed dot product and the activations its signed side in all their 8 bits, each
// given its weight's sign as it is loaded ((a ^ m) - m, m a mask of 0xff where the weight is negative); the units'
// integer sums are one lane each before they are multiplied by S. IQ1's grid of -1, 0, 1 is held as codes 0, 1, 2
// (the 7-bit side) and the activations' sums take the rest: the sum over 8 weights of (8 g +- 1) a is
// 8 dot(code, a) - 7 sum(a), or - 9 sum(a) where the shift is down. The K types' M meet the same sums.
//
// xs (one token's): n / 32 float32 scales, then n / 16 int16 sums of each 16 activations, then n / 8 int16 sums of
// each 8 (sums_x): n / 2 bytes in all.
//
// ---- The tables (sb_tables; 34992 bytes, TABLES). Nothing static can be in a module of kernels (nothing relocates a
// data segment), and a vector written as a constant in a loop is made again at every use (T163):
//   0      IQ2_XXS's grid: 256 entries of 8 magnitudes        2048   IQ2_XS's: 512 of 8        6144  IQ2_S's: 1024 of 8
//   14336  IQ3_XXS's grid: 256 entries of 4                   15360  IQ3_S's: 512 of 4
//   17408  IQ1_S's grid (IQ1_M's too) plus one: 2048 entries of 8 codes 0, 1, 2
//   33792  the signs of 8 weights by their 7-bit index: 128 entries of 8 bytes, 0xff a minus (the eighth sign makes the
//          number of minus signs even)
//   34816  IQ4_XS's 16 values, then their magnitudes, then 0xff where one is negative
//   34864  the vectors (V_*)
// The first 33792 bytes are the checkpoint's row "codebooks" as it is (convert/grids.py's codebooks(): the row is the
// codebooks and nothing of these kernels', so that a kernel written otherwise needs no other file), but for IQ1_S's
// grid, which has one added. A type's entries are looked up twice to eight times for each 32 weights and stay in the
// first caches (IQ1's 16 KB cost its kernel 2 to 10% with indices drawn evenly, the worst case: T363.1).

const T_IQ2XS: usize = 2048, T_IQ2S: usize = 6144, T_IQ3XXS: usize = 14336, T_IQ3S: usize = 15360, T_IQ1S: usize = 17408;
const T_SIGNS: usize = 33792, T_IQ4: usize = 34816, T_VECTORS: usize = 34864;
const GRIDS: usize = 33792, TABLES: i32 = 34992;
// the vectors: bytes 1, 2, 4, .. 128 twice; bytes 0 x 8, 1 x 8; sixteen 3s; sixteen 15s; the int32s 512, 64, 8, 1;
// sixteen 7s; sixteen 1s; the int32s 4096, 256, 16, 1
const V_BITS: usize = 0, V_FIRST: usize = 16, V_THREE: usize = 32, V_FIFTEEN: usize = 48, V_POWERS3: usize = 64, V_SEVENS: usize = 80, V_ONES: usize = 96,
  V_POWERS4: usize = 112;

// a type as a type: what a kernel is of is known as it is compiled (an argument that is a constant is not: every branch
// would be compiled for every use), and R says the same of relaxed SIMD
class K2 {}
class K4 {}
class XXS2 {}
class XS2 {}
class S2 {}
class XXS3 {}
class S3 {}
class S1 {}
class M1 {}
class XS4 {}
class Relaxed {}
class Plain {}
const Q2_K: i32 = 10, Q4_K: i32 = 12, IQ2_XXS: i32 = 16, IQ2_XS: i32 = 17, IQ3_XXS: i32 = 18, IQ1_S: i32 = 19, IQ3_S: i32 = 21,
  IQ2_S: i32 = 22, IQ4_XS: i32 = 23, IQ1_M: i32 = 29;

// ---- what a type is
// the bytes of a block of the type numbered kind (0: no type of these)
function bytesOf(kind: i32): usize {
  return kind == IQ2_XXS ? 66 : kind == IQ3_S ? 110 : kind == IQ2_XS ? 74 : kind == IQ2_S ? 82 : kind == IQ3_XXS ? 98 :
    kind == IQ1_M ? 56 : kind == IQ1_S ? 50 : kind == Q2_K ? 84 : kind == IQ4_XS ? 136 : kind == Q4_K ? 144 : 0;
}
// the same of the type K, a constant
// @ts-ignore: decorator
@inline function sizeOf<K>(): usize {
  return idof<K>() == idof<XXS2>() ? 66 : idof<K>() == idof<S3>() ? 110 : idof<K>() == idof<XS2>() ? 74 : idof<K>() == idof<S2>() ? 82 : idof<K>() == idof<XXS3>() ? 98 :
    idof<K>() == idof<M1>() ? 56 : idof<K>() == idof<S1>() ? 50 : idof<K>() == idof<K2>() ? 84 : idof<K>() == idof<XS4>() ? 136 : 144;
}
// whether S is of 16 weights (of 32 otherwise)
// @ts-ignore: decorator
@inline function bySixteen<K>(): bool {
  return idof<K>() == idof<XS2>() || idof<K>() == idof<S2>() || idof<K>() == idof<M1>() || idof<K>() == idof<K2>();
}
// whether W has a sign held apart from its magnitude
// @ts-ignore: decorator
@inline function hasSigns<K>(): bool {
  return idof<K>() == idof<XXS2>() || idof<K>() == idof<XS2>() || idof<K>() == idof<S2>() || idof<K>() == idof<XXS3>() || idof<K>() == idof<S3>() || idof<K>() == idof<XS4>();
}
// @ts-ignore: decorator
@inline function hasMinimums<K>(): bool {
  return idof<K>() == idof<K2>() || idof<K>() == idof<K4>();
}

// ---- small things
// @ts-ignore: decorator
@inline function hsum(a: v128): f32 {
  return f32x4.extract_lane(a, 0) + f32x4.extract_lane(a, 1) + f32x4.extract_lane(a, 2) + f32x4.extract_lane(a, 3);
}
// the float16 at p as float32 (the finite ones, exactly, subnormals too: kernel/halves.ts's half; an infinity or a
// NaN, which no scale of a file is, reads as a finite number)
// @ts-ignore: decorator
@inline function halfAt(p: usize): f32 {
  return halfOf(<i32>load<u16>(p));
}
// @ts-ignore: decorator
@inline function halfOf(bits: i32): f32 {
  return reinterpret<f32>(((bits << 16 >> 16) << 13) & <i32>0x8fffffff) * 5.192296858534828e33;
}
// @ts-ignore: decorator
@inline function vectorAt(tab: usize, which: usize): v128 {
  return v128.load(tab + T_VECTORS + which);
}
// one lane a vector: [sum d0, sum d1, sum d2, sum d3] (kernel_relaxed.ts's)
// @ts-ignore: decorator
@inline function groupSums(d0: v128, d1: v128, d2: v128, d3: v128): v128 {
  const s01 = i32x4.add(v128.shuffle<i32>(d0, d1, 0, 4, 1, 5), v128.shuffle<i32>(d0, d1, 2, 6, 3, 7));
  const s23 = i32x4.add(v128.shuffle<i32>(d2, d3, 0, 4, 1, 5), v128.shuffle<i32>(d2, d3, 2, 6, 3, 7));
  return i32x4.add(v128.shuffle<i32>(s01, s23, 0, 1, 4, 5), v128.shuffle<i32>(s01, s23, 2, 3, 6, 7));
}
// lanes 0 + 1 and 2 + 3 of a, then of b
// @ts-ignore: decorator
@inline function pairSums(a: v128, b: v128): v128 {
  return i32x4.add(v128.shuffle<i32>(a, b, 0, 2, 4, 6), v128.shuffle<i32>(a, b, 1, 3, 5, 7));
}
// @ts-ignore: decorator
@inline function four(a: i32, b: i32, c: i32, d: i32): v128 {
  return i32x4.replace_lane(i32x4.replace_lane(i32x4.replace_lane(i32x4.splat(a), 1, b), 2, c), 3, d);
}
// two entries of 8 bytes as one vector
// @ts-ignore: decorator
@inline function pair(a: usize, b: usize): v128 {
  return v128.load64_lane(b, v128.load64_zero(a), 1);
}
// four entries of 4 bytes
// @ts-ignore: decorator
@inline function quad(a: usize, b: usize, c: usize, d: usize): v128 {
  return i32x4.replace_lane(i32x4.replace_lane(i32x4.replace_lane(v128.load32_zero(a), 1, load<i32>(b)), 2, load<i32>(c)), 3, load<i32>(d));
}
// the activations a with the signs of their weights: m is 0xff where the weight is negative
// @ts-ignore: decorator
@inline function signedBy(a: v128, m: v128): v128 {
  return i8x16.sub(v128.xor(a, m), m);
}
// acc plus the products of 16 signed bytes a and 16 magnitudes (0 to 127), four lanes. Which products a lane holds
// differs between the two; their sum does not, and a lane is never used alone
// @ts-ignore: decorator
@inline function dotAdd<R>(acc: v128, a: v128, magnitudes: v128): v128 {
  if (idof<R>() == idof<Relaxed>()) return i32x4.relaxed_dot_i8x16_i7x16_add_s(a, magnitudes, acc);
  // two products of 127 x 127 at most fit an int16
  const products = i16x8.add(i16x8.extmul_low_i8x16_s(a, magnitudes), i16x8.extmul_high_i8x16_s(a, magnitudes));
  return i32x4.add(acc, i32x4.extadd_pairwise_i16x8_s(products));
}
// the bytes of table by the indices i, all under 16
// @ts-ignore: decorator
@inline function lookup<R>(table: v128, i: v128): v128 {
  if (idof<R>() == idof<Relaxed>()) return i8x16.relaxed_swizzle(table, i);
  return i8x16.swizzle(table, i);
}
// 16 bits as 16 bytes: 0xff where a bit is set, the lowest bit first
// @ts-ignore: decorator
@inline function bytesOfBits<R>(word: u32, bits: v128, first: v128): v128 {
  return i8x16.eq(v128.and(lookup<R>(i16x8.splat(<i16>word), first), bits), bits);
}
// the low four bits of each byte of the group's 16 at v, or the high four
// @ts-ignore: decorator
@inline function nibblesOf(v: v128, high: bool, fifteen: v128): v128 {
  return v128.and(high ? i16x8.shr_u(v, 4) : v, fifteen);
}

// ---- the 16 weights h (0 to 15) of the block at p: their magnitudes (IQ1: their codes; the K types: W itself)
// @ts-ignore: decorator
@inline function magnitudes<K, R>(p: usize, h: i32, tab: usize, three: v128, fifteen: v128): v128 {
  const at = <usize>h;
  if (idof<K>() == idof<XXS2>()) {
    const q = p + 2 + ((at >> 1) << 3) + ((at & 1) << 1);
    return pair(tab + (<usize>load<u8>(q) << 3), tab + (<usize>load<u8>(q, 1) << 3));
  }
  if (idof<K>() == idof<XS2>()) {
    const q = p + 2 + (at << 2);
    return pair(tab + T_IQ2XS + ((<usize>load<u16>(q) & 511) << 3), tab + T_IQ2XS + ((<usize>load<u16>(q, 2) & 511) << 3));
  }
  if (idof<K>() == idof<S2>()) {
    const q = p + 2 + (at << 1), high = <usize>load<u8>(p + 66 + (at >> 1)) >> ((at & 1) << 2);
    return pair(tab + T_IQ2S + ((<usize>load<u8>(q) | ((high & 3) << 8)) << 3), tab + T_IQ2S + ((<usize>load<u8>(q, 1) | ((high & 12) << 6)) << 3));
  }
  if (idof<K>() == idof<XXS3>()) {
    const q = p + 2 + (at << 2), grid = tab + T_IQ3XXS;
    return quad(grid + (<usize>load<u8>(q) << 2), grid + (<usize>load<u8>(q, 1) << 2), grid + (<usize>load<u8>(q, 2) << 2), grid + (<usize>load<u8>(q, 3) << 2));
  }
  if (idof<K>() == idof<S3>()) {
    const q = p + 2 + (at << 2), grid = tab + T_IQ3S, high = <usize>load<u8>(p + 66 + (at >> 1)) >> ((at & 1) << 2);
    return quad(grid + ((<usize>load<u8>(q) | ((high & 1) << 8)) << 2), grid + ((<usize>load<u8>(q, 1) | ((high & 2) << 7)) << 2),
      grid + ((<usize>load<u8>(q, 2) | ((high & 4) << 6)) << 2), grid + ((<usize>load<u8>(q, 3) | ((high & 8) << 5)) << 2));
  }
  if (idof<K>() == idof<S1>()) {
    const q = p + 2 + (at << 1), grid = tab + T_IQ1S, high = <usize>load<u16>(p + 34 + ((at >> 1) << 1)) >> ((at & 1) * 6);
    return pair(grid + ((<usize>load<u8>(q) | ((high & 7) << 8)) << 3), grid + ((<usize>load<u8>(q, 1) | ((high & 56) << 5)) << 3));
  }
  if (idof<K>() == idof<M1>()) {
    const q = p + (at << 1), grid = tab + T_IQ1S, high = <usize>load<u8>(p + 32 + at);
    return pair(grid + ((<usize>load<u8>(q) | ((high & 7) << 8)) << 3), grid + ((<usize>load<u8>(q, 1) | ((high & 0x70) << 4)) << 3));
  }
  if (idof<K>() == idof<K2>()) {  // bits 2 s and 2 s + 1 of byte b of a half's 32 are weight 32 s + b of the half
    const v = v128.load(p + 16 + ((at >> 3) << 5) + ((at & 1) << 4)), s = (h & 7) >> 1;
    return v128.and(s == 0 ? v : i16x8.shr_u(v, 2 * s), three);
  }
  if (idof<K>() == idof<K4>()) {  // of every 32 bytes the low halves are 32 weights and the high halves the next 32
    return nibblesOf(v128.load(p + 16 + ((at >> 2) << 5) + ((at & 1) << 4)), ((h >> 1) & 1) != 0, fifteen);
  }
  // IQ4_XS: of a group's 16 bytes the low halves are its first 16 weights
  return lookup<R>(v128.load(tab + T_IQ4, 16), nibblesOf(v128.load(p + 8 + ((at >> 1) << 4)), (h & 1) != 0, fifteen));
}
// 0xff where one of those 16 is negative (the types of hasSigns())
// @ts-ignore: decorator
@inline function minuses<K, R>(p: usize, h: i32, tab: usize, bits: v128, first: v128, fifteen: v128): v128 {
  const at = <usize>h, signs = tab + T_SIGNS;
  if (idof<K>() == idof<XXS2>() || idof<K>() == idof<XXS3>()) {  // a word for 32 weights: four indices of 7 bits
    const aux = (idof<K>() == idof<XXS2>() ? load<u32>(p + 6 + ((at >> 1) << 3)) : load<u32>(p + 66 + ((at >> 1) << 2))) >> <u32>((h & 1) * 14);
    return pair(signs + (<usize>(aux & 127) << 3), signs + (<usize>((aux >> 7) & 127) << 3));
  }
  if (idof<K>() == idof<XS2>()) {
    const q = p + 2 + (at << 2);
    return pair(signs + ((<usize>load<u16>(q) >> 9) << 3), signs + ((<usize>load<u16>(q, 2) >> 9) << 3));
  }
  if (idof<K>() == idof<S2>()) return bytesOfBits<R>(<u32>load<u16>(p + 34 + (at << 1)), bits, first);
  if (idof<K>() == idof<S3>()) return bytesOfBits<R>(<u32>load<u16>(p + 74 + (at << 1)), bits, first);
  return lookup<R>(v128.load(tab + T_IQ4, 32), nibblesOf(v128.load(p + 8 + ((at >> 1) << 4)), (h & 1) != 0, fifteen));
}
// the 16 as W itself, signed int8 (the tile with the signs on the weights, and sb_row). IQ1: 8 g + 1 or 8 g - 1
// @ts-ignore: decorator
@inline function weights<K, R>(p: usize, h: i32, tab: usize, bits: v128, first: v128, three: v128, fifteen: v128): v128 {
  const at = <usize>h;
  if (idof<K>() == idof<XS4>()) return lookup<R>(v128.load(tab + T_IQ4), nibblesOf(v128.load(p + 8 + ((at >> 1) << 4)), (h & 1) != 0, fifteen));
  const held = magnitudes<K, R>(p, h, tab, three, fifteen);
  if (hasSigns<K>()) return signedBy(held, minuses<K, R>(p, h, tab, bits, first, fifteen));
  if (idof<K>() == idof<S1>() || idof<K>() == idof<M1>()) {  // 8 code - 7, and 2 less where the shift is down
    const eights = i8x16.sub(i16x8.shl(held, 3), vectorAt(tab, V_SEVENS)), down = shiftsDown<K>(p, h);
    return i8x16.add(eights, i8x16.add(down, down));
  }
  return held;
}
// IQ1: 0xff for those of the 16 weights whose shift is down (-1/8)
// @ts-ignore: decorator
@inline function shiftsDown<K>(p: usize, h: i32): v128 {
  if (idof<K>() == idof<S1>()) return i64x2.splat(-<i64>(load<u16>(p + 34 + ((<usize>h >> 1) << 1)) >> 15));
  const high = <i64>load<u8>(p + 32 + <usize>h);
  return i64x2.replace_lane(i64x2.splat(-((high >> 3) & 1)), 1, -(high >> 7));
}

// ---- the integers of the unit that the 16 weights h are in
// @ts-ignore: decorator
@inline function scaleOf<K>(p: usize, h: i32): i32 {
  const at = <usize>h, g = at >> 1;
  if (idof<K>() == idof<XXS2>()) return <i32>((load<u32>(p + 6 + (g << 3)) >> 27) | 1);
  if (idof<K>() == idof<XXS3>()) return <i32>((load<u32>(p + 66 + (g << 2)) >> 27) | 1);
  if (idof<K>() == idof<XS2>()) return <i32>((((<u32>load<u8>(p + 66 + g) >> <u32>((h & 1) << 2)) & 15) << 1) | 1);
  if (idof<K>() == idof<S2>()) return <i32>((((<u32>load<u8>(p + 74 + g) >> <u32>((h & 1) << 2)) & 15) << 1) | 1);
  if (idof<K>() == idof<S3>()) return <i32>((((<u32>load<u8>(p + 106 + (g >> 1)) >> <u32>(((h >> 1) & 1) << 2)) & 15) << 1) | 1);
  if (idof<K>() == idof<S1>()) return <i32>((((<u32>load<u16>(p + 34 + (g << 1)) >> 12) & 7) << 1) | 1);
  if (idof<K>() == idof<M1>()) return <i32>((((<u32>load<u16>(p + 48 + ((at >> 2) << 1)) >> <u32>((h & 3) * 3)) & 7) << 1) | 1);
  if (idof<K>() == idof<K2>()) return <i32>(load<u8>(p + at) & 15);
  if (idof<K>() == idof<K4>()) {  // six bits: the first four groups' in bytes 0 to 3, the last four's in the low halves of bytes 8 to 11 and the tops of bytes 0 to 3
    return h < 8 ? <i32>(load<u8>(p + 4 + g) & 63) : <i32>((<u32>load<u8>(p + 8 + g) & 15) | ((<u32>load<u8>(p + g) >> 6) << 4));
  }
  // IQ4_XS: four bits in scales_l, two above them in the word, less 32
  return <i32>(((<u32>load<u8>(p + 4 + (g >> 1)) >> <u32>(((h >> 1) & 1) << 2)) & 15) | (((<u32>load<u16>(p, 2) >> <u32>((h >> 1) << 1)) & 3) << 4)) - 32;
}
// @ts-ignore: decorator
@inline function minimumOf<K>(p: usize, h: i32): i32 {
  const at = <usize>h, g = at >> 1;
  if (idof<K>() == idof<K2>()) return <i32>(load<u8>(p + at) >> 4);
  // Q4_K: in bytes 4 to 7, and in the high halves of bytes 8 to 11 with the tops of bytes 4 to 7
  return h < 8 ? <i32>(load<u8>(p + 8 + g) & 63) : <i32>((<u32>load<u8>(p + 8 + g) >> 4) | ((<u32>load<u8>(p + 4 + g) >> 6) << 4));
}
// @ts-ignore: decorator
@inline function blockScale<K>(p: usize): f32 {
  if (idof<K>() == idof<XXS2>() || idof<K>() == idof<XS2>() || idof<K>() == idof<S2>() || idof<K>() == idof<S1>()) return halfAt(p) * 0.125;
  if (idof<K>() == idof<XXS3>()) return halfAt(p) * 0.25;
  if (idof<K>() == idof<M1>()) return wholeScale(p) * 0.125;
  if (idof<K>() == idof<K2>()) return halfAt(p + 80);
  return halfAt(p);
}
// IQ1_M's d: the top four bits of its four words, the first word's the lowest
// @ts-ignore: decorator
@inline function wholeScale(p: usize): f32 {
  const w0 = <u32>load<u16>(p, 48), w1 = <u32>load<u16>(p, 50), w2 = <u32>load<u16>(p, 52), w3 = <u32>load<u16>(p, 54);
  return halfOf(<i32>((w0 >> 12) | ((w1 >> 8) & 0xf0) | ((w2 >> 4) & 0xf00) | (w3 & 0xf000)));
}
// @ts-ignore: decorator
@inline function blockMinimum<K>(p: usize): f32 {
  return idof<K>() == idof<K2>() ? halfAt(p + 82) : halfAt(p + 2);
}

// ---- the tables, at area (TABLES bytes), of the checkpoint's row of codebooks (convert/grids.py's codebooks():
// the six grids one after the other, a value a byte, then IQ4_XS's 16 values). Returns the bytes written.
export function sb_tables(area: usize, codebooks: usize): i32 {
  for (let i: usize = 0; i < GRIDS; i++) store<u8>(area + i, <u8>(<u32>load<u8>(codebooks + i) + (i >= T_IQ1S ? 1 : 0)));
  for (let i: u32 = 0; i < 128; i++) {
    let ones: u32 = 0;
    for (let j: u32 = 0; j < 7; j++) ones += (i >> j) & 1;
    const held = i | ((ones & 1) << 7);
    for (let j: u32 = 0; j < 8; j++) store<u8>(area + T_SIGNS + (<usize>i << 3) + <usize>j, (held >> j) & 1 ? <u8>255 : <u8>0);
  }
  for (let i: usize = 0; i < 16; i++) {
    const value = load<i8>(codebooks + GRIDS + i);
    store<i8>(area + T_IQ4 + i, value);
    store<i8>(area + T_IQ4 + 16 + i, <i8>(value < 0 ? -<i32>value : <i32>value));
    store<u8>(area + T_IQ4 + 32 + i, value < 0 ? <u8>255 : <u8>0);
  }
  const v = area + T_VECTORS;
  for (let j: usize = 0; j < 16; j++) {
    store<u8>(v + V_BITS + j, <u8>(<u32>1 << (<u32>j & 7)));
    store<u8>(v + V_FIRST + j, <u8>(j >> 3));
    store<u8>(v + V_THREE + j, 3);
    store<u8>(v + V_FIFTEEN + j, 15);
    store<u8>(v + V_SEVENS + j, 7);
    store<u8>(v + V_ONES + j, 1);
  }
  for (let j: usize = 0; j < 4; j++) {
    store<i32>(v + V_POWERS3 + (j << 2), 512 >> (<i32>j * 3));
    store<i32>(v + V_POWERS4 + (j << 2), 4096 >> (<i32>j * 4));
  }
  return TABLES;
}

export { T_IQ2XS, T_IQ2S, T_IQ3XXS, T_IQ3S, T_IQ1S, T_SIGNS, T_IQ4, T_VECTORS, GRIDS, TABLES, V_BITS, V_FIRST, V_THREE, V_FIFTEEN, V_POWERS3, V_SEVENS, V_ONES, V_POWERS4, K2, K4, XXS2, XS2, S2, XXS3, S3, S1, M1, XS4, Relaxed, Plain, Q2_K, Q4_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, bytesOf, sizeOf, bySixteen, hasSigns, hasMinimums, hsum, halfAt, halfOf, vectorAt, groupSums, pairSums, four, pair, quad, signedBy, dotAdd, lookup, bytesOfBits, nibblesOf, magnitudes, minuses, weights, shiftsDown, scaleOf, minimumOf, blockScale, wholeScale, blockMinimum };
