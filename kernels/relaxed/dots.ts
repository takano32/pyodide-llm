// relaxed/dots.ts (T686: from blocks/tile.ts, which had it from T363.5): what the caller asks of this engine's relaxed
// dot product, which relaxed SIMD leaves to the engine where the second operand's top bit is set: arm64's SDOT
// multiplies signed bytes by signed bytes, x86's PMADDUBSW takes the second for unsigned. The kernel_relaxed module (the
// ternary products: forward/pairs.ts takes the int16 sums on x86 alone) and the blockkernel_relaxed one
// (matmul_sbw_tile: forward/blocks.ts) both export it.

/** out[0..3] = the four lanes of the relaxed dot product of the 16 bytes at a by the 16 at b. Where they are the sums of
 * the products of signed bytes for every a and b (tests/blocks-check.ts's signedDots() asks), matmul_sbw_tile is right. */
export function signed_dots(out: usize, a: usize, b: usize): void {
  v128.store(out, i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(a), v128.load(b), i32x4.splat(0)));
}
