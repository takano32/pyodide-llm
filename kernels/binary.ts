// binary.ts (T634): the weights of a binary model, one bit each, as the dot products take them. kernel/matmul.ts
// (matmul_b1) and relaxed/binary.ts (matmul_b1r) import it. The layout is llama2_numpy.pack_binary's: a group of 128
// weights in 16 bytes, weight 64 h + 4 c + p (h 0 or 1, c 0..15, p 0..3) at bit 4 h + p of byte c, set for +1, and
// one float16 scale a group after all the bits of the matrix (as ternary's, ternary.ts).
//
// The ternary kernels' form, with half the bytes: a weight w of -1 or +1 is the code w + 1 = 2 b of its bit b, which
// is what a ternary weight's code is (ternary.ts: dot(a, w + 1) = dot(a, w) + sum(a), and interleave() leaves minus
// each group's sum), so the activations interleave() laid out and its sums are taken as they are, every group's
// integer sum is ternary's to the bit for the same weights, and so is every number after it (the same float32
// operations in the same order). Sixteen bytes hold the 128 weights of a group: a shift of all sixteen and a mask of
// 2 leave twice the bit of weight 64 h + 4 c + p in byte c, the plane of codes that meets plane p of block h of the
// activations. A code is 0 or 2, the 7-bit side of a relaxed dot product, as ternary's are.
//
// two: sixteen bytes of 2, made by the caller's caller from an argument of the kernel (as ternary.ts's three, for
// the reason it gives). The shifts are of 16-bit lanes: the mask of bit 1 takes away what a byte gets from its
// neighbour (bits 8 - s and up, s at most 6), and bit 0 is doubled by an add of bytes, which drops each one's top bit.

/** the codes of bit k (0..7) of the sixteen bytes v: twice the bit of weight 64 (k >> 2) + 4 c + (k & 3), in byte c */
// @ts-ignore: decorator
@inline export function bitCodes(v: v128, k: i32, two: v128): v128 {
  return v128.and(k == 0 ? i8x16.add(v, v) : k == 1 ? v : i16x8.shr_u(v, k - 1), two);
}
