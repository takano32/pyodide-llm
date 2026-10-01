// ternary.ts (T231): the weights of a ternary model, two bits each, as the dot products take them. kernel.ts
// (matmul_t2) and kernel_relaxed.ts (matmul_t2r) import it. The layout is llama2_numpy.pack_ternary's, which is
// Prism ML's PQ2_0's: a group of 128 weights in 32 bytes, weight j as its code (the weight + 1: 0, 1 or 2) in byte
// j >> 2 at bits 2 (j & 3), and one float32 scale a group.
//
// Nothing is widened. Sixteen bytes hold 64 weights, and a shift of all sixteen by 2 p bits and a mask leave the codes
// of every fourth weight in the bytes: plane p is weights 4 c + p, c = 0..15. The activations are laid out the same
// way once a token (kernel.ts's interleave: byte 16 p + c of a block of 64 is activation 4 c + p), so that a plane of
// codes meets its sixteen activations as they are loaded. The codes are not negative: they are the 7-bit side of a
// relaxed dot product and the activations its signed side, in all their 8 bits, with no bias to take out of a row's
// sums: dot(a, w + 1) = dot(a, w) + sum(a), and interleave leaves minus the sum of each group of 32 activations after
// their scales, once a token. So a row needs nothing besides its weights and its scales.
//
// The form was chosen among those of tests/ternary-forms.ts, by tests/ternary-bench.mjs on CI's arm64 and x86-64
// (TODO.md's T231 has the table). The fork of llama.cpp that reads these files widens a group to int8 first (its NEON
// kernel for PQ2_0, ggml/src/ggml-cpu/arch/arm/quants.c, MIT); none of its lines is here.
//
// three: sixteen bytes of 3, made by the caller's caller from an argument of the kernel. Written here as a constant,
// V8 makes it again at every use (T163): the token's kernel ran 1.36 times as slow on CI's EPYC 7763 and 1.15 on its
// Neoverse-N2 that way. The shifts are of 16-bit lanes: the mask takes away what a byte gets from its neighbour, and
// x86 has no shift of bytes (T166).

/** the codes of plane p (0..3) of the sixteen bytes v: those of weights 4 c + p of 64, in byte c */
// @ts-ignore: decorator
@inline export function codes(v: v128, p: i32, three: v128): v128 {
  return v128.and(p == 0 ? v : i16x8.shr_u(v, 2 * p), three);
}

/** lo and hi: the int32 lanes of two blocks of 64, lanes 0 and 1 of each the sums of its first group of 32
 * activations, lanes 2 and 3 of its second (a plane's bytes 0..7 are of activations 0..31). The four groups' sums */
// @ts-ignore: decorator
@inline export function fourSums(lo: v128, hi: v128): v128 {
  return i32x4.add(v128.shuffle<i32>(lo, hi, 0, 2, 4, 6), v128.shuffle<i32>(lo, hi, 1, 3, 5, 7));
}
