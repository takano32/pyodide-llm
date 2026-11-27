// Optional kernel that needs the relaxed-SIMD proposal (Chrome 114+, Firefox 146+, not in shipping Safari).
// Kept in its own module: a browser without relaxed SIMD rejects the whole module at compile time, so the
// Python loader simply falls back to kernel.ts.
//
// T644.6: this file is the window, as kernel.ts is. The kernels are in relaxed/, by what they multiply, and every one
// of them is named here: what this file exports is what the module exports, in this order (the order they had as one
// file). A new relaxed kernel is written in the file of its kind and added to a line here.
//   relaxed/int8.ts     int8 weights: a token's (matmul_q8r) and a prompt's four by four (matmul_q8r_tile)
//   relaxed/six.ts      int6 weights (matmul_q6r), widened by ../six.ts
//   relaxed/signs.ts    ternary and binary weights (matmul_t2r, matmul_t2r_tile, matmul_b1r, matmul_b1r_tile): one
//                       kernel for both, the packing a type parameter (../signs.ts); binary's numbers are ternary's to
//                       the bit for the same weights; T686: matmul_t2x and matmul_t2x_tile, ternary's with the
//                       sums in int16 (x86's form, taken where it times faster)
//   relaxed/dots.ts     signed_dots: how this engine's relaxed dot product takes its second operand (T686: x86 for
//                       unsigned, arm64 for signed), which blocks_relaxed.ts's module exports too

export { matmul_q8r, matmul_q8r_tile } from "./relaxed/int8";
export { matmul_q6r } from "./relaxed/six";
export { matmul_t2r, matmul_t2r_tile, matmul_b1r, matmul_b1r_tile, matmul_t2x, matmul_t2x_tile } from "./relaxed/signs";
export { signed_dots } from "./relaxed/dots";
