// Optional kernel that needs the relaxed-SIMD proposal (Chrome 114+, Firefox 146+, not in shipping Safari).
// Kept in its own module: a browser without relaxed SIMD rejects the whole module at compile time, so the
// Python loader simply falls back to kernel.ts.

const GS: i32 = 32;

// activations from quantize_x(bias = 64). wc = scale * sum(group weights) removes that bias again:
// dot(w, q - 64) = dot(w, q) - 64 * sum(w)
export function matmul_q8r(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, wc: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>n;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    const crow = wc + ((<usize>i * <usize>ng) << 2);
    let facc = f32x4.splat(0);
    let corr: f32 = 0;
    for (let g = 0; g < ng; g++) {
      const o = <usize>(g * GS);
      let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o), v128.load(xq + o), i32x4.splat(0));
      acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 16), v128.load(xq + o + 16), acc);
      const xsg = load<f32>(xs + (<usize>g << 2));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(acc), f32x4.splat(load<f32>(srow + (<usize>g << 2)) * xsg)));
      corr += load<f32>(crow + (<usize>g << 2)) * xsg;
    }
    const s = f32x4.extract_lane(facc, 0) + f32x4.extract_lane(facc, 1) + f32x4.extract_lane(facc, 2) + f32x4.extract_lane(facc, 3);
    store<f32>(xout + (<usize>i << 2), s - <f32>64.0 * corr);
  }
}
