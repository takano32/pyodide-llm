// tests/ternary-forms.ts (T231): the inner loops that were set against the one kernels/ took for the matrix product
// on ternary weights (-1, 0, +1 times one scale a group of 128, as Prism ML's PQ2_0 and PTQ1_0 hold them).
// tests/ternary-bench.mjs compiles this file and times each form beside the kernels of kernels/ (matmul_t2r and
// matmul_t2, and the int8 kernels on the same weights widened to int8); TODO.md's T231 has the table and the choice.
// Not a kernel of the page: nothing here is shipped.
//
// In every form a row is whole groups of 128 weights, the activations are int8 in groups of 32 with a float32 scale a
// group (xs), and the group's scale of the weights is a float32 (ws, one a group of 128 of every row).
//
//   bc  the form taken (kernels/ternary.ts: two bits a weight in PQ2_0's own order, the activations interleaved and
//       signed, the codes 0, 1, 2 the 7-bit side of the relaxed dot product) with its mask of 3 written as a constant
//       in the loop, which V8 makes again at every use; the kernel takes it from an argument
//   bs  bc whose last plane is a shift of bytes with no mask (arm64 has one, x86-64 not)
//   bx  bc with the products of pairs kept in int16 over the four planes of a vector (x86's pmaddubsw alone, one
//       pmaddwd a vector instead of four)
//   c   two bits a weight widened to the signed int8 -1, 0, 1 and given to the int8 kernel's dot product as it is:
//       the activations as matmul_q8r takes them (7 bits and a bias of 64, in their own order) and so an int32
//       correction a group of 32 of every row (-64 times the sum of its weights: a bit a weight more memory). The
//       weights' order is chosen for it: byte c of a block of 16 holds weights c, c + 16, c + 32, c + 48.
//   a   PTQ1_0's own packing, five weights a byte in base 3 (1.75 bits a weight with the float16 scale): a digit is
//       the high byte of three times the byte, the low byte goes on to the next digit (as the fork's dequantize), here
//       as two unsigned compares (the digit) and two adds (the byte times 3, wrapping). The block as the file has it,
//       28 bytes: 16 bytes of 5 digits (weights 16 n + m), 8 bytes of 5 (80 + 8 n + m), 2 bytes of 4 (120 + 2 n + m),
//       2 bytes of the scale (not read here: ws). The activations of a block are laid out for it in 160 bytes: its
//       first 80 in order, then for n = 0..4 sixteen bytes of [80 + 8 n .. 88 + 8 n), 120 + 2 n, 121 + 2 n (zeros for
//       n = 4) and six zeros.

// @ts-ignore: decorator
@inline function hsum(a: v128): f32 {
  return f32x4.extract_lane(a, 0) + f32x4.extract_lane(a, 1) + f32x4.extract_lane(a, 2) + f32x4.extract_lane(a, 3);
}
// one lane a vector: [sum d0, sum d1, sum d2, sum d3] (kernel_relaxed.ts's)
// @ts-ignore: decorator
@inline function groupSums(d0: v128, d1: v128, d2: v128, d3: v128): v128 {
  const s01 = i32x4.add(v128.shuffle<i32>(d0, d1, 0, 4, 1, 5), v128.shuffle<i32>(d0, d1, 2, 6, 3, 7));
  const s23 = i32x4.add(v128.shuffle<i32>(d2, d3, 0, 4, 1, 5), v128.shuffle<i32>(d2, d3, 2, 6, 3, 7));
  return i32x4.add(v128.shuffle<i32>(s01, s23, 0, 1, 4, 5), v128.shuffle<i32>(s01, s23, 2, 3, 6, 7));
}

// ---- bc: 64 weights (16 bytes at w) against their 64 interleaved activations (x): lanes 0 and 1 hold the first
// group of 32, lanes 2 and 3 the second
// @ts-ignore: decorator
@inline function dot64(w: usize, x: usize, three: v128): v128 {
  const v = v128.load(w);
  let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x), v128.and(v, three), i32x4.splat(0));
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 16), v128.and(i16x8.shr_u(v, 2), three), acc);
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 32), v128.and(i16x8.shr_u(v, 4), three), acc);
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 48), v128.and(i16x8.shr_u(v, 6), three), acc);
}
export function matmul_bc(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32): void {
  const blocks = n >> 7, three = i8x16.splat(3);
  const xa = xs + (<usize>(n >> 5) << 2);
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 2);
    const srow = ws + ((<usize>i * <usize>blocks) << 2);
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const w = row + (<usize>b << 5), x = xq + (<usize>b << 7), at = <usize>b << 4;
      const lo = dot64(w, x, three), hi = dot64(w + 16, x + 64, three);
      const sums = i32x4.add(i32x4.add(v128.shuffle<i32>(lo, hi, 0, 2, 4, 6), v128.shuffle<i32>(lo, hi, 1, 3, 5, 7)), v128.load(xa + at));
      facc = f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(sums), v128.load(xs + at)), v128.load32_splat(srow + (<usize>b << 2))));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// ---- bs: bc whose last plane is a shift of bytes with no mask
// @ts-ignore: decorator
@inline function dot64s(w: usize, x: usize, three: v128): v128 {
  const v = v128.load(w);
  let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x), v128.and(v, three), i32x4.splat(0));
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 16), v128.and(i16x8.shr_u(v, 2), three), acc);
  acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 32), v128.and(i16x8.shr_u(v, 4), three), acc);
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 48), i8x16.shr_u(v, 6), acc);
}
export function matmul_bs(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32): void {
  const blocks = n >> 7, three = i8x16.splat(3);
  const xa = xs + (<usize>(n >> 5) << 2);
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 2);
    const srow = ws + ((<usize>i * <usize>blocks) << 2);
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const w = row + (<usize>b << 5), x = xq + (<usize>b << 7), at = <usize>b << 4;
      const lo = dot64s(w, x, three), hi = dot64s(w + 16, x + 64, three);
      const sums = i32x4.add(i32x4.add(v128.shuffle<i32>(lo, hi, 0, 2, 4, 6), v128.shuffle<i32>(lo, hi, 1, 3, 5, 7)), v128.load(xa + at));
      facc = f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(sums), v128.load(xs + at)), v128.load32_splat(srow + (<usize>b << 2))));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// ---- bx: the pairs' products in int16 over the four planes (2 x 2 x 127 a pair, four planes: 2032 at most)
// @ts-ignore: decorator
@inline function dot64x(w: usize, x: usize, three: v128): v128 {
  const v = v128.load(w);
  let acc = i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x), v128.and(v, three));
  acc = i16x8.add(acc, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 16), v128.and(i16x8.shr_u(v, 2), three)));
  acc = i16x8.add(acc, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 32), v128.and(i16x8.shr_u(v, 4), three)));
  acc = i16x8.add(acc, i16x8.relaxed_dot_i8x16_i7x16_s(v128.load(x, 48), v128.and(i16x8.shr_u(v, 6), three)));
  return i32x4.extadd_pairwise_i16x8_s(acc);
}
export function matmul_bx(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32): void {
  const blocks = n >> 7, three = i8x16.splat(3);
  const xa = xs + (<usize>(n >> 5) << 2);
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 2);
    const srow = ws + ((<usize>i * <usize>blocks) << 2);
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const w = row + (<usize>b << 5), x = xq + (<usize>b << 7), at = <usize>b << 4;
      const lo = dot64x(w, x, three), hi = dot64x(w + 16, x + 64, three);
      const sums = i32x4.add(i32x4.add(v128.shuffle<i32>(lo, hi, 0, 2, 4, 6), v128.shuffle<i32>(lo, hi, 1, 3, 5, 7)), v128.load(xa + at));
      facc = f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(sums), v128.load(xs + at)), v128.load32_splat(srow + (<usize>b << 2))));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// ---- c: widened to the signed int8 the int8 dot takes (matmul_q8r's activations and corrections)
// @ts-ignore: decorator
@inline function signed(v: v128, shift: i32, three: v128, one: v128): v128 {
  return i8x16.sub(v128.and(i16x8.shr_u(v, shift), three), one);
}
export function matmul_c(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, wc: usize, n: i32, r0: i32, r1: i32): void {
  const blocks = n >> 7, three = i8x16.splat(3), one = i8x16.splat(1);
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>(n >> 2);
    const srow = ws + ((<usize>i * <usize>blocks) << 2);
    const crow = wc + ((<usize>i * <usize>blocks) << 4);
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const w = row + (<usize>b << 5), x = xq + (<usize>b << 7), at = <usize>b << 4;
      const v0 = v128.load(w), v1 = v128.load(w, 16);
      let d0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(i8x16.sub(v128.and(v0, three), one), v128.load(x), i32x4.splat(0));
      d0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(signed(v0, 2, three, one), v128.load(x, 16), d0);
      let d1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(signed(v0, 4, three, one), v128.load(x, 32), i32x4.splat(0));
      d1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(signed(v0, 6, three, one), v128.load(x, 48), d1);
      let d2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(i8x16.sub(v128.and(v1, three), one), v128.load(x, 64), i32x4.splat(0));
      d2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(signed(v1, 2, three, one), v128.load(x, 80), d2);
      let d3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(signed(v1, 4, three, one), v128.load(x, 96), i32x4.splat(0));
      d3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(signed(v1, 6, three, one), v128.load(x, 112), d3);
      const sums = i32x4.add(groupSums(d0, d1, d2, d3), v128.load(crow + at));
      facc = f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(sums), v128.load(xs + at)), v128.load32_splat(srow + (<usize>b << 2))));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// ---- a: PTQ1_0's blocks as the file has them. A digit of every byte of t: 0, 1 or 2, the high byte of 3 t
// (3 t >= 256 from 86, >= 512 from 171)
// @ts-ignore: decorator
@inline function digit(t: v128): v128 {
  return i8x16.abs(i8x16.add(i8x16.ge_u(t, i8x16.splat(86)), i8x16.ge_u(t, i8x16.splat(<i8>171))));
}
// @ts-ignore: decorator
@inline function thrice(t: v128): v128 {  // the low byte of 3 t: what the next digit is taken from
  return i8x16.add(t, i8x16.add(t, t));
}
export function matmul_a(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32): void {
  const blocks = n >> 7;
  const xa = xs + (<usize>(n >> 5) << 2);
  const ten = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0), zero = i32x4.splat(0);
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>blocks * 28;
    const srow = ws + ((<usize>i * <usize>blocks) << 2);
    let facc = f32x4.splat(0);
    for (let b = 0; b < blocks; b++) {
      const w = row + <usize>b * 28, x = xq + <usize>b * 160, at = <usize>b << 4;
      // the sixteen bytes of five digits: weights 0..31 (the first group), 32..63, 64..79
      let t = v128.load(w);
      let g0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x), digit(t), zero);
      t = thrice(t);
      g0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 16), digit(t), g0);
      t = thrice(t);
      let g1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 32), digit(t), zero);
      t = thrice(t);
      g1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 48), digit(t), g1);
      t = thrice(t);
      let g2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 64), digit(t), zero);
      // the eight bytes of five digits and the two of four: lanes 0 and 1 weights 80 + 8 n .., lane 2 weights 120 + 2 n ..
      t = v128.and(v128.load(w, 16), ten);
      let d = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 80), digit(t), zero);
      t = thrice(t);
      d = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 96), digit(t), d);
      t = thrice(t);
      let g3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 112), digit(t), zero);
      t = thrice(t);
      g3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 128), digit(t), g3);
      t = thrice(t);
      g3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(x, 144), digit(t), g3);
      // d's lanes 0 and 1 are of the third group (weights 80..95), its lane 2 of the fourth (120..123)
      g2 = i32x4.add(g2, v128.shuffle<i32>(d, zero, 0, 1, 4, 4));
      g3 = i32x4.add(g3, v128.shuffle<i32>(d, zero, 2, 4, 4, 4));
      const sums = i32x4.add(groupSums(g0, g1, g2, g3), v128.load(xa + at));
      facc = f32x4.add(facc, f32x4.mul(f32x4.mul(f32x4.convert_i32x4_s(sums), v128.load(xs + at)), v128.load32_splat(srow + (<usize>b << 2))));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}
