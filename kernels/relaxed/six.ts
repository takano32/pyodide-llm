// relaxed/six.ts (T644.6): the int6 product with relaxed SIMD (matmul_q6r), on the groups ../six.ts widens.

import { sixFirst, sixSecond, sixTops } from "../six";
import { GS, hsum, groupSums, laneSum } from "../kernel/math";

// T98: the same on int6 weights (six.ts: 24 bytes a group, widened straight into int8), in the same order as
// matmul_q8r: four groups at a time, one sum a group (T167) with the corrections in it (T197), their scales as
// vectors. wc as for matmul_q8r (-64 times the sum of the group's int8 values, six_sums).
// @ts-ignore: decorator
@inline function dot6(p: usize, x: usize): v128 {
  const low = v128.load(p), t = sixTops(p);
  const d = i32x4.relaxed_dot_i8x16_i7x16_add_s(sixFirst(low, t), v128.load(x), i32x4.splat(0));
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(sixSecond(low, t), v128.load(x, 16), d);
}

export function matmul_q6r(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, wc: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  const ng4 = ng & ~3;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>ng * 24;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    const crow = wc + ((<usize>i * <usize>ng) << 2);
    let facc = f32x4.splat(0);
    let g = 0;
    for (; g < ng4; g += 4) {
      const p = row + <usize>g * 24, o = xq + <usize>(g * GS);
      const scales = f32x4.mul(v128.load(srow + (<usize>g << 2)), v128.load(xs + (<usize>g << 2)));
      const d0 = dot6(p, o), d1 = dot6(p + 24, o + 32), d2 = dot6(p + 48, o + 64), d3 = dot6(p + 72, o + 96);
      const sums = i32x4.add(groupSums(d0, d1, d2, d3), v128.load(crow + (<usize>g << 2)));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(sums), scales));
    }
    let sum = hsum(facc);
    for (; g < ng; g++) {
      const whole = laneSum(dot6(row + <usize>g * 24, xq + <usize>(g * GS))) + load<i32>(crow + (<usize>g << 2));
      sum += <f32>whole * (load<f32>(srow + (<usize>g << 2)) * load<f32>(xs + (<usize>g << 2)));
    }
    store<f32>(xout + (<usize>i << 2), sum);
  }
}
