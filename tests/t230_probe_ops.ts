// T230's review (a probe): the pieces of kernels/kernel.ts's interleave() and four ways to the same 64 bytes, to be called
// over and over in a browser's engine (tests/t230_probe_ops.mjs): JavaScriptCore on x86-64 got interleave() wrong after
// about 2000 calls (the optimizing tier), and these say which instruction does it and which formulation survives.
//   a block of 64 int8 at src -> the same 64 bytes in four planes at dst (byte 16 p + c is source byte 4 c + p)

// @ts-ignore: decorator
@inline function fourths(v: v128): v128 {
  return i8x16.shuffle(v, v, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
}

// the pieces
export function op_fourths(src: usize, dst: usize): void {  // each 16 bytes: byte 4 r + c of the result is byte r + 4 c
  for (let b: usize = 0; b < 64; b += 16) v128.store(dst + b, fourths(v128.load(src + b)));
}
export function op_unpack32(src: usize, dst: usize): void {  // dwords: (s0 s1) 0 4 1 5 and 2 6 3 7, (s2 s3) the same
  const s0 = v128.load(src), s1 = v128.load(src, 16), s2 = v128.load(src, 32), s3 = v128.load(src, 48);
  v128.store(dst, v128.shuffle<i32>(s0, s1, 0, 4, 1, 5));
  v128.store(dst + 16, v128.shuffle<i32>(s0, s1, 2, 6, 3, 7));
  v128.store(dst + 32, v128.shuffle<i32>(s2, s3, 0, 4, 1, 5));
  v128.store(dst + 48, v128.shuffle<i32>(s2, s3, 2, 6, 3, 7));
}
export function op_unpack64(src: usize, dst: usize): void {  // qwords: (a c) 0 2 and 1 3, (b d) the same
  const a = v128.load(src), b = v128.load(src, 16), c = v128.load(src, 32), d = v128.load(src, 48);
  v128.store(dst, v128.shuffle<i64>(a, c, 0, 2));
  v128.store(dst + 16, v128.shuffle<i64>(a, c, 1, 3));
  v128.store(dst + 32, v128.shuffle<i64>(b, d, 0, 2));
  v128.store(dst + 48, v128.shuffle<i64>(b, d, 1, 3));
}
// @ts-ignore: decorator
@inline function sumOf32(a: v128, b: v128): i32 {
  const quads = i32x4.extadd_pairwise_i16x8_s(i16x8.add(i16x8.extadd_pairwise_i8x16_s(a), i16x8.extadd_pairwise_i8x16_s(b)));
  return i32x4.extract_lane(quads, 0) + i32x4.extract_lane(quads, 1) + i32x4.extract_lane(quads, 2) + i32x4.extract_lane(quads, 3);
}
export function op_sums(src: usize, dst: usize): void {  // minus the sum of the two groups of 32, at dst and dst + 4
  store<i32>(dst, -sumOf32(v128.load(src), v128.load(src, 16)));
  store<i32>(dst, -sumOf32(v128.load(src, 32), v128.load(src, 48)), 4);
}

// the whole block, as kernel.ts's interleave() does it (a)
export function blk_a(src: usize, dst: usize): void {
  const v0 = v128.load(src), v1 = v128.load(src, 16), v2 = v128.load(src, 32), v3 = v128.load(src, 48);
  const s0 = fourths(v0), s1 = fourths(v1), s2 = fourths(v2), s3 = fourths(v3);
  const a01 = v128.shuffle<i32>(s0, s1, 0, 4, 1, 5), a23 = v128.shuffle<i32>(s2, s3, 0, 4, 1, 5);
  const b01 = v128.shuffle<i32>(s0, s1, 2, 6, 3, 7), b23 = v128.shuffle<i32>(s2, s3, 2, 6, 3, 7);
  v128.store(dst, v128.shuffle<i64>(a01, a23, 0, 2));
  v128.store(dst, v128.shuffle<i64>(a01, a23, 1, 3), 16);
  v128.store(dst, v128.shuffle<i64>(b01, b23, 0, 2), 32);
  v128.store(dst, v128.shuffle<i64>(b01, b23, 1, 3), 48);
}
// (b) the same with a zero in the second place of the byte shuffle (not the same register twice)
// @ts-ignore: decorator
@inline function fourthsZ(v: v128, z: v128): v128 {
  return i8x16.shuffle(v, z, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
}
export function blk_b(src: usize, dst: usize): void {
  const z = v128.splat<i32>(0);
  const v0 = v128.load(src), v1 = v128.load(src, 16), v2 = v128.load(src, 32), v3 = v128.load(src, 48);
  const s0 = fourthsZ(v0, z), s1 = fourthsZ(v1, z), s2 = fourthsZ(v2, z), s3 = fourthsZ(v3, z);
  const a01 = v128.shuffle<i32>(s0, s1, 0, 4, 1, 5), a23 = v128.shuffle<i32>(s2, s3, 0, 4, 1, 5);
  const b01 = v128.shuffle<i32>(s0, s1, 2, 6, 3, 7), b23 = v128.shuffle<i32>(s2, s3, 2, 6, 3, 7);
  v128.store(dst, v128.shuffle<i64>(a01, a23, 0, 2));
  v128.store(dst, v128.shuffle<i64>(a01, a23, 1, 3), 16);
  v128.store(dst, v128.shuffle<i64>(b01, b23, 0, 2), 32);
  v128.store(dst, v128.shuffle<i64>(b01, b23, 1, 3), 48);
}
// (c) the byte shuffle as a swizzle by a mask loaded from memory-free constants (a vector of the indices)
export function blk_c(src: usize, dst: usize): void {
  const mask = i8x16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  const v0 = v128.load(src), v1 = v128.load(src, 16), v2 = v128.load(src, 32), v3 = v128.load(src, 48);
  const s0 = i8x16.swizzle(v0, mask), s1 = i8x16.swizzle(v1, mask), s2 = i8x16.swizzle(v2, mask), s3 = i8x16.swizzle(v3, mask);
  const a01 = v128.shuffle<i32>(s0, s1, 0, 4, 1, 5), a23 = v128.shuffle<i32>(s2, s3, 0, 4, 1, 5);
  const b01 = v128.shuffle<i32>(s0, s1, 2, 6, 3, 7), b23 = v128.shuffle<i32>(s2, s3, 2, 6, 3, 7);
  v128.store(dst, v128.shuffle<i64>(a01, a23, 0, 2));
  v128.store(dst, v128.shuffle<i64>(a01, a23, 1, 3), 16);
  v128.store(dst, v128.shuffle<i64>(b01, b23, 0, 2), 32);
  v128.store(dst, v128.shuffle<i64>(b01, b23, 1, 3), 48);
}
// (d) with no shuffle at all: shifts, masks and narrows. Plane r: byte r of every dword of each 16 bytes, narrowed to
// 32-bit values (0..255), then to 16 and to 8 bits: four vectors' four dwords make the sixteen bytes
// @ts-ignore: decorator
@inline function plane(v0: v128, v1: v128, v2: v128, v3: v128, r: i32): v128 {
  const m = i32x4.splat(255);
  const t0 = v128.and(i32x4.shr_u(v0, 8 * r), m), t1 = v128.and(i32x4.shr_u(v1, 8 * r), m);
  const t2 = v128.and(i32x4.shr_u(v2, 8 * r), m), t3 = v128.and(i32x4.shr_u(v3, 8 * r), m);
  return i8x16.narrow_i16x8_u(i16x8.narrow_i32x4_u(t0, t1), i16x8.narrow_i32x4_u(t2, t3));
}
export function blk_d(src: usize, dst: usize): void {
  const v0 = v128.load(src), v1 = v128.load(src, 16), v2 = v128.load(src, 32), v3 = v128.load(src, 48);
  v128.store(dst, plane(v0, v1, v2, v3, 0));
  v128.store(dst, plane(v0, v1, v2, v3, 1), 16);
  v128.store(dst, plane(v0, v1, v2, v3, 2), 32);
  v128.store(dst, plane(v0, v1, v2, v3, 3), 48);
}
// (e) scalar: the 64 bytes one at a time
export function blk_e(src: usize, dst: usize): void {
  for (let p: usize = 0; p < 4; p++) {
    for (let c: usize = 0; c < 16; c++) store<i8>(dst + 16 * p + c, load<i8>(src + 4 * c + p));
  }
}
