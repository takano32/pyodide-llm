// Optional kernel that needs the relaxed-SIMD proposal (Chrome 114+, Firefox 146+, not in shipping Safari).
// Kept in its own module: a browser without relaxed SIMD rejects the whole module at compile time, so the
// Python loader simply falls back to kernel.ts.

import { sixFirst, sixSecond, sixTops } from "./six";

const GS: i32 = 32;

// activations from quantize_x(bias = 64). wc = scale * sum(group weights) removes that bias again:
// dot(w, q - 64) = dot(w, q) - 64 * sum(w)
export function matmul_q8r(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, wc: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  const ng4 = ng & ~3;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>n;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    const crow = wc + ((<usize>i * <usize>ng) << 2);
    // Four groups at a time: their scales are multiplied as one vector, and two accumulators take turns, so that
    // no addition waits for the one before it. The bias correction is a dot product of its own, also by fours.
    let f0 = f32x4.splat(0), f1 = f32x4.splat(0), corrs = f32x4.splat(0);
    let g = 0;
    for (; g < ng4; g += 4) {
      const o = <usize>(g * GS);
      const xsv = v128.load(xs + (<usize>g << 2));
      const scales = f32x4.mul(v128.load(srow + (<usize>g << 2)), xsv);
      corrs = f32x4.add(corrs, f32x4.mul(v128.load(crow + (<usize>g << 2)), xsv));
      let d0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o), v128.load(xq + o), i32x4.splat(0));
      let d1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 32), v128.load(xq + o + 32), i32x4.splat(0));
      let d2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 64), v128.load(xq + o + 64), i32x4.splat(0));
      let d3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 96), v128.load(xq + o + 96), i32x4.splat(0));
      d0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 16), v128.load(xq + o + 16), d0);
      d1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 48), v128.load(xq + o + 48), d1);
      d2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 80), v128.load(xq + o + 80), d2);
      d3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 112), v128.load(xq + o + 112), d3);
      f0 = f32x4.add(f0, f32x4.mul(f32x4.convert_i32x4_s(d0), f32x4.splat(f32x4.extract_lane(scales, 0))));
      f1 = f32x4.add(f1, f32x4.mul(f32x4.convert_i32x4_s(d1), f32x4.splat(f32x4.extract_lane(scales, 1))));
      f0 = f32x4.add(f0, f32x4.mul(f32x4.convert_i32x4_s(d2), f32x4.splat(f32x4.extract_lane(scales, 2))));
      f1 = f32x4.add(f1, f32x4.mul(f32x4.convert_i32x4_s(d3), f32x4.splat(f32x4.extract_lane(scales, 3))));
    }
    const facc = f32x4.add(f0, f1);
    let sum: f32 = f32x4.extract_lane(facc, 0) + f32x4.extract_lane(facc, 1) + f32x4.extract_lane(facc, 2) + f32x4.extract_lane(facc, 3);
    let corr: f32 = f32x4.extract_lane(corrs, 0) + f32x4.extract_lane(corrs, 1) + f32x4.extract_lane(corrs, 2) + f32x4.extract_lane(corrs, 3);
    for (; g < ng; g++) {
      const o = <usize>(g * GS);
      let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o), v128.load(xq + o), i32x4.splat(0));
      acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 16), v128.load(xq + o + 16), acc);
      const xsg = load<f32>(xs + (<usize>g << 2));
      const part = f32x4.mul(f32x4.convert_i32x4_s(acc), f32x4.splat(load<f32>(srow + (<usize>g << 2)) * xsg));
      sum += f32x4.extract_lane(part, 0) + f32x4.extract_lane(part, 1) + f32x4.extract_lane(part, 2) + f32x4.extract_lane(part, 3);
      corr += load<f32>(crow + (<usize>g << 2)) * xsg;
    }
    store<f32>(xout + (<usize>i << 2), sum - <f32>64.0 * corr);
  }
}

// T98: the same on int6 weights (six.ts: 24 bytes a group, widened straight into int8), in the same order as
// matmul_q8r: four groups at a time, their scales and corrections as vectors, two accumulators taking turns. wc as
// for matmul_q8r (scale times the sum of the group's int8 values).
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
    let f0 = f32x4.splat(0), f1 = f32x4.splat(0), corrs = f32x4.splat(0);
    let g = 0;
    for (; g < ng4; g += 4) {
      const p = row + <usize>g * 24, o = xq + <usize>(g * GS);
      const xsv = v128.load(xs + (<usize>g << 2));
      const scales = f32x4.mul(v128.load(srow + (<usize>g << 2)), xsv);
      corrs = f32x4.add(corrs, f32x4.mul(v128.load(crow + (<usize>g << 2)), xsv));
      const d0 = dot6(p, o), d1 = dot6(p + 24, o + 32), d2 = dot6(p + 48, o + 64), d3 = dot6(p + 72, o + 96);
      f0 = f32x4.add(f0, f32x4.mul(f32x4.convert_i32x4_s(d0), f32x4.splat(f32x4.extract_lane(scales, 0))));
      f1 = f32x4.add(f1, f32x4.mul(f32x4.convert_i32x4_s(d1), f32x4.splat(f32x4.extract_lane(scales, 1))));
      f0 = f32x4.add(f0, f32x4.mul(f32x4.convert_i32x4_s(d2), f32x4.splat(f32x4.extract_lane(scales, 2))));
      f1 = f32x4.add(f1, f32x4.mul(f32x4.convert_i32x4_s(d3), f32x4.splat(f32x4.extract_lane(scales, 3))));
    }
    const facc = f32x4.add(f0, f1);
    let sum: f32 = f32x4.extract_lane(facc, 0) + f32x4.extract_lane(facc, 1) + f32x4.extract_lane(facc, 2) + f32x4.extract_lane(facc, 3);
    let corr: f32 = f32x4.extract_lane(corrs, 0) + f32x4.extract_lane(corrs, 1) + f32x4.extract_lane(corrs, 2) + f32x4.extract_lane(corrs, 3);
    for (; g < ng; g++) {
      const xsg = load<f32>(xs + (<usize>g << 2));
      const part = f32x4.mul(f32x4.convert_i32x4_s(dot6(row + <usize>g * 24, xq + <usize>(g * GS))), f32x4.splat(load<f32>(srow + (<usize>g << 2)) * xsg));
      sum += f32x4.extract_lane(part, 0) + f32x4.extract_lane(part, 1) + f32x4.extract_lane(part, 2) + f32x4.extract_lane(part, 3);
      corr += load<f32>(crow + (<usize>g << 2)) * xsg;
    }
    store<f32>(xout + (<usize>i << 2), sum - <f32>64.0 * corr);
  }
}
