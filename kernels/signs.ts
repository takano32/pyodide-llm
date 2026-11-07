// signs.ts (T644.7): the two packings of weights that are only signs and a scale, ternary (ternary.ts: 32 bytes a
// group of 128) and binary (binary.ts: 16 bytes), to the kernels that are one for both: kernel/matmul.ts's matmul_t2
// and matmul_b1, relaxed/signs.ts's matmul_t2r, matmul_t2r_tile, matmul_b1r and matmul_b1r_tile. A packing is a type
// parameter (Ternary or Binary), as a block type is in blocks/ (kernels/README.md: "A type is a type parameter"), so
// each kernel is compiled for its packing alone, with no branch of the other's.
//
// Both leave the codes of a weight w as w + 1 (0, 1 or 2; binary's 0 or 2) in planes that meet the planes interleave()
// laid out, and the activations' sums take the + 1 away (ternary.ts): every integer sum, and every number after it, is
// the same for the same weights however they are packed. What differs is where a group's codes are: ternary's are in
// two vectors of 16 bytes (h = 0 and 1, the two blocks of 64 activations), binary's all in one; codesOf() takes the
// vector of block h, which blockBytes() loads for ternary and is the first again for binary (one load a group, as before).

import { codes } from "./ternary";
import { bitCodes } from "./binary";

/** ternary.ts's packing: two bits a weight */
export class Ternary {}
/** binary.ts's packing: one bit a weight */
export class Binary {}

/** the bytes of a group of 128 weights */
// @ts-ignore: decorator
@inline export function groupBytes<P>(): usize {
  return idof<P>() == idof<Ternary>() ? 32 : 16;
}
/** the bytes of a row of n weights */
// @ts-ignore: decorator
@inline export function rowBytes<P>(n: i32): usize {
  return <usize>(idof<P>() == idof<Ternary>() ? n >> 2 : n >> 3);
}
/** the 16 bytes that hold the codes of block h (0 or 1) of the group at w, whose first 16 bytes are v */
// @ts-ignore: decorator
@inline export function blockBytes<P>(w: usize, v: v128, h: i32): v128 {
  return idof<P>() == idof<Ternary>() && h == 1 ? v128.load(w, 16) : v;
}
/** plane k (0..3) of the codes of block h, v being blockBytes<P>(w, .., h): the codes of weights 64 h + 4 c + k, in byte c */
// @ts-ignore: decorator
@inline export function codesOf<P>(v: v128, h: i32, k: i32, mask: v128): v128 {
  return idof<P>() == idof<Ternary>() ? codes(v, k, mask) : bitCodes(v, 4 * h + k, mask);
}
