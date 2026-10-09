// kernel/math.ts (T356): what the kernels of several kinds use. No kernel is here: kernel.ts exports none of this.
// No static data and no std math, as everywhere in kernels/: exp is table-free.

const GS: i32 = 32; // int8 quantization group size, as in quantize.py

// @ts-ignore: decorator
@inline function hsum(a: v128): f32 {
  return f32x4.extract_lane(a, 0) + f32x4.extract_lane(a, 1) + f32x4.extract_lane(a, 2) + f32x4.extract_lane(a, 3);
}

// @ts-ignore: decorator
@inline function fexp(x: f32): f32 { // table-free Cephes expf
  if (x < -87.0) return 0;
  if (x > 88.0) x = 88.0;
  const k = nearest<f32>(x * <f32>1.44269504088896341);
  const r = x - k * <f32>0.693359375 - k * <f32>-2.12194440e-4;
  let p: f32 = 1.9875691500e-4;
  p = p * r + <f32>1.3981999507e-3;
  p = p * r + <f32>8.3334519073e-3;
  p = p * r + <f32>4.1665795894e-2;
  p = p * r + <f32>1.6666665459e-1;
  p = p * r + <f32>5.0000001201e-1;
  const e = p * (r * r) + r + <f32>1.0;
  return e * reinterpret<f32>(<u32>(<i32>k + 127) << 23);
}

// fexp() on four numbers at a time, bit for bit, except that below -87 it gives exp(-87) (1.6e-38) where fexp gives 0
// @ts-ignore: decorator
@inline function vexp(x: v128): v128 {
  x = f32x4.min(f32x4.max(x, f32x4.splat(-87.0)), f32x4.splat(88.0));
  const k = f32x4.nearest(f32x4.mul(x, f32x4.splat(1.44269504088896341)));
  const r = f32x4.sub(f32x4.sub(x, f32x4.mul(k, f32x4.splat(0.693359375))), f32x4.mul(k, f32x4.splat(-2.12194440e-4)));
  let p = f32x4.splat(1.9875691500e-4);
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(1.3981999507e-3));
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(8.3334519073e-3));
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(4.1665795894e-2));
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(1.6666665459e-1));
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(5.0000001201e-1));
  const e = f32x4.add(f32x4.add(f32x4.mul(p, f32x4.mul(r, r)), r), f32x4.splat(1.0));
  const scale = i32x4.shl(i32x4.add(i32x4.trunc_sat_f32x4_s(k), i32x4.splat(127)), 23);
  return f32x4.mul(e, scale);
}

// The largest of from and the n floats at at. A NaN anywhere makes it NaN (f32x4.max and max keep a NaN, T195), and
// max does not depend on the order, so the result is the one of a scalar walk to the bit (and -0 is below +0 in both).
// T189 (sample): maxima side by side, since x86 has no instruction for f32x4.max (NaN, -0): V8 writes it as 8
// instructions, a chain of about 14 cycles, and one maximum waited on itself. T201: eight of them, 32 floats a turn
// (four still waited on the chain on x86), for sample's logits and attention's scores (which took one scalar maximum
// a position). No public implementation's lines: llama.cpp's ggml_vec_max_f32 is a scalar walk.
// @ts-ignore: decorator
@inline function largest(at: usize, n: i32, from: f32): f32 {
  let best = from;
  let i = 0;
  if (n >= 4) {
    let b0 = f32x4.max(f32x4.splat(from), v128.load(at)), b1 = b0, b2 = b0, b3 = b0, b4 = b0, b5 = b0, b6 = b0, b7 = b0;
    for (i = 4; i + 32 <= n; i += 32) {
      const a = at + (<usize>i << 2);
      b0 = f32x4.max(b0, v128.load(a));
      b1 = f32x4.max(b1, v128.load(a, 16));
      b2 = f32x4.max(b2, v128.load(a, 32));
      b3 = f32x4.max(b3, v128.load(a, 48));
      b4 = f32x4.max(b4, v128.load(a, 64));
      b5 = f32x4.max(b5, v128.load(a, 80));
      b6 = f32x4.max(b6, v128.load(a, 96));
      b7 = f32x4.max(b7, v128.load(a, 112));
    }
    for (; i + 4 <= n; i += 4) b0 = f32x4.max(b0, v128.load(at + (<usize>i << 2)));
    const bests = f32x4.max(f32x4.max(f32x4.max(b0, b1), f32x4.max(b2, b3)), f32x4.max(f32x4.max(b4, b5), f32x4.max(b6, b7)));
    best = max(max(f32x4.extract_lane(bests, 0), f32x4.extract_lane(bests, 1)), max(f32x4.extract_lane(bests, 2), f32x4.extract_lane(bests, 3)));
  }
  for (; i < n; i++) best = max(best, load<f32>(at + (<usize>i << 2)));
  return best;
}

export { GS, hsum, fexp, vexp, largest };
