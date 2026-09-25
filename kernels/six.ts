// six.ts (T98): the 32 values of one int6 group, widened to the int8 the dot products take. kernel.ts (matmul_q6)
// and kernel_relaxed.ts (matmul_q6r) both import it. The layout is llama2_numpy.pack6's: an int6 value is an int8
// with its two low bits zero, so a group widens straight into int8 with no offset. 24 bytes a group: the four low
// bits of the six of value j and of value j + 16 in byte j (0..15), the top two bits of values k, k + 8, k + 16,
// k + 24 in byte 16 + k (0..7) at bits 0, 2, 4, 6. Masks and shifts by constants only: no table.

/** the top bits of the group at p, read once for both halves: the eight bytes twice over, so that lanes 0..7 and
 * 8..15 both read byte k */
// @ts-ignore: decorator
@inline export function sixTops(p: usize): v128 {
  const h = v128.load64_zero(p + 16);
  return i8x16.shuffle(h, h, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
}

/** values 0..15 of a group, from its first 16 bytes and sixTops(): the low nibble to bits 2-5, and to bits 6-7 the
 * top bits of value k (bits 0-1 of byte k, lanes 0..7) or of value k + 8 (bits 2-3, lanes 8..15) */
// @ts-ignore: decorator
@inline export function sixFirst(low: v128, t: v128): v128 {
  const top = v128.and(v128.bitselect(i8x16.shl(t, 6), i8x16.shl(t, 4), i64x2(-1, 0)), i8x16.splat(-64));
  return v128.or(i8x16.shl(v128.and(low, i8x16.splat(15)), 2), top);
}

/** values 16..31: the high nibble to bits 2-5, and the top bits of value k + 16 (bits 4-5) or k + 24 (bits 6-7) */
// @ts-ignore: decorator
@inline export function sixSecond(low: v128, t: v128): v128 {
  const top = v128.and(v128.bitselect(i8x16.shl(t, 2), t, i64x2(-1, 0)), i8x16.splat(-64));
  return v128.or(v128.and(i8x16.shr_u(low, 2), i8x16.splat(60)), top);
}
