// WASM SIMD128 kernels for a llama2.c-style transformer (AssemblyScript).
// Loaded into Pyodide with ctypes as an Emscripten side module, they work in place on NumPy-owned memory.
// No static data and no std math on purpose: nothing relocates a data segment in this hand-made side module.

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

// W (d,n) @ x (n,) -> xout, rows [r0, r1)
export function matmul_f32(xout: usize, x: usize, w: usize, n: i32, r0: i32, r1: i32): void {
  const n16 = n & ~15;
  for (let i = r0; i < r1; i++) {
    const row = w + ((<usize>i * <usize>n) << 2);
    let a0 = f32x4.splat(0), a1 = f32x4.splat(0), a2 = f32x4.splat(0), a3 = f32x4.splat(0);
    let j = 0;
    for (; j < n16; j += 16) {
      const o = <usize>j << 2;
      a0 = f32x4.add(a0, f32x4.mul(v128.load(row + o), v128.load(x + o)));
      a1 = f32x4.add(a1, f32x4.mul(v128.load(row + o + 16), v128.load(x + o + 16)));
      a2 = f32x4.add(a2, f32x4.mul(v128.load(row + o + 32), v128.load(x + o + 32)));
      a3 = f32x4.add(a3, f32x4.mul(v128.load(row + o + 48), v128.load(x + o + 48)));
    }
    let val: f32 = hsum(f32x4.add(f32x4.add(a0, a1), f32x4.add(a2, a3)));
    for (; j < n; j++) {
      const o = <usize>j << 2;
      val += load<f32>(row + o) * load<f32>(x + o);
    }
    store<f32>(xout + (<usize>i << 2), val);
  }
}

// bias = 0: signed int8 in [-127,127];  bias = 64: 7-bit unsigned, real value = (q - 64) * scale (for kernel_relaxed.ts)
export function quantize_x(xq: usize, xs: usize, x: usize, n: i32, bias: i32): void {
  // SIMD, 32 values (one group) at a time. Every step is the scalar one lane by lane (abs, max, the division, the
  // product, round half to even), so the int8 and the scales are the same to the bit as before, and as NumPy's
  // llama2_convert.quantize(), which the converter now hands to this (T89). Only a NaN in the input tells them
  // apart (SIMD max keeps it, the scalar compare skipped it), and a NaN in a checkpoint is broken on every path.
  const qmax: f32 = bias == 0 ? 127.0 : 63.0;
  const offset = i32x4.splat(bias);
  for (let g = 0; g < n; g += GS) {
    const p = x + (<usize>g << 2);
    const v0 = v128.load(p), v1 = v128.load(p, 16), v2 = v128.load(p, 32), v3 = v128.load(p, 48);
    const v4 = v128.load(p, 64), v5 = v128.load(p, 80), v6 = v128.load(p, 96), v7 = v128.load(p, 112);
    const m = f32x4.max(f32x4.max(f32x4.max(f32x4.abs(v0), f32x4.abs(v1)), f32x4.max(f32x4.abs(v2), f32x4.abs(v3))),
                        f32x4.max(f32x4.max(f32x4.abs(v4), f32x4.abs(v5)), f32x4.max(f32x4.abs(v6), f32x4.abs(v7))));
    let amax = f32x4.extract_lane(m, 0);
    const m1 = f32x4.extract_lane(m, 1), m2 = f32x4.extract_lane(m, 2), m3 = f32x4.extract_lane(m, 3);
    if (m1 > amax) amax = m1;
    if (m2 > amax) amax = m2;
    if (m3 > amax) amax = m3;
    const scale: f32 = amax / qmax;
    store<f32>(xs + (<usize>(g / GS) << 2), scale);
    const inv = f32x4.splat(scale > 0 ? <f32>1.0 / scale : 0);
    const q0 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v0, inv))), offset);
    const q1 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v1, inv))), offset);
    const q2 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v2, inv))), offset);
    const q3 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v3, inv))), offset);
    const q4 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v4, inv))), offset);
    const q5 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v5, inv))), offset);
    const q6 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v6, inv))), offset);
    const q7 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v7, inv))), offset);
    const out = xq + <usize>g;
    v128.store(out, i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(q0, q1), i16x8.narrow_i32x4_s(q2, q3)));
    v128.store(out, i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(q4, q5), i16x8.narrow_i32x4_s(q6, q7)), 16);
  }
}

// int8 weights (wq) with one float32 scale per group (ws), int8 activations from quantize_x(bias = 0)
export function matmul_q8(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>n;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    let facc = f32x4.splat(0);
    for (let g = 0; g < ng; g++) {
      const o = <usize>(g * GS);
      const a0 = v128.load(row + o), b0 = v128.load(xq + o);
      const a1 = v128.load(row + o + 16), b1 = v128.load(xq + o + 16);
      const acc = i32x4.add(
        i32x4.add(
          i32x4.extadd_pairwise_i16x8_s(i16x8.extmul_low_i8x16_s(a0, b0)),
          i32x4.extadd_pairwise_i16x8_s(i16x8.extmul_high_i8x16_s(a0, b0))),
        i32x4.add(
          i32x4.extadd_pairwise_i16x8_s(i16x8.extmul_low_i8x16_s(a1, b1)),
          i32x4.extadd_pairwise_i16x8_s(i16x8.extmul_high_i8x16_s(a1, b1))));
      const s = load<f32>(srow + (<usize>g << 2)) * load<f32>(xs + (<usize>g << 2));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(acc), f32x4.splat(s)));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

export function rmsnorm(out: usize, x: usize, w: usize, n: i32): void {
  let acc = f32x4.splat(0);
  let j = 0;
  const n4 = n & ~3;
  for (; j < n4; j += 4) { const v = v128.load(x + (<usize>j << 2)); acc = f32x4.add(acc, f32x4.mul(v, v)); }
  let ss: f32 = hsum(acc);
  for (; j < n; j++) { const v = load<f32>(x + (<usize>j << 2)); ss += v * v; }
  const s: f32 = <f32>1.0 / sqrt<f32>(ss / <f32>n + <f32>1e-5);
  for (j = 0; j < n; j++) {
    const o = <usize>j << 2;
    store<f32>(out + o, load<f32>(w + o) * (s * load<f32>(x + o)));
  }
}

export function rope(v: usize, fcr: usize, fci: usize, nh: i32, hs: i32, rot: i32): void {
  // rot: how many values of each head to turn (GPT-NeoX turns only the first ones, everyone else all of them)
  const turned = rot > 0 && rot < hs ? rot : hs;
  for (let h = 0; h < nh; h++) {
    const base = v + (<usize>(h * hs) << 2);
    for (let i = 0; i < turned; i += 2) {
      const c = load<f32>(fcr + (<usize>(i >> 1) << 2)), s = load<f32>(fci + (<usize>(i >> 1) << 2));
      const p0 = base + (<usize>i << 2);
      const v0 = load<f32>(p0), v1 = load<f32>(p0 + 4);
      store<f32>(p0, v0 * c - v1 * s);
      store<f32>(p0 + 4, v0 * s + v1 * c);
    }
  }
}

// kc / vc: this layer's cache laid out [seq][nkv * hs] (NOT the NumPy forward's [kv heads][seq][hs]);
// att: scratch of at least nh * (pos + 1) floats. Grouped-query attention: nh / nkv query heads share one kv head.
// The cache is walked position by position, all heads of a row at once: a head at a time meant nh strided passes
// over a cache that does not fit the caches of the CPU (12 layers of it), which cost more than the arithmetic.
export function attention(out: usize, q: usize, kc: usize, vc: usize, att: usize, pos: i32, nh: i32, nkv: i32, hs: i32): void {
  const kvDim = nkv * hs;
  const kvMul = nh / nkv;
  const hs16 = hs & ~15, hs4 = hs & ~3;
  const count = pos + 1;
  const isq: f32 = <f32>1.0 / sqrt<f32>(<f32>hs);
  // 1. the scores of every head against every position
  for (let t = 0; t < count; t++) {
    const row = kc + (<usize>(t * kvDim) << 2);
    for (let h = 0; h < nh; h++) {
      const qh = q + (<usize>(h * hs) << 2);
      const kt = row + (<usize>((h / kvMul) * hs) << 2);
      let a0 = f32x4.splat(0), a1 = f32x4.splat(0), a2 = f32x4.splat(0), a3 = f32x4.splat(0);
      let j = 0;
      for (; j < hs16; j += 16) {
        const o = <usize>j << 2;
        a0 = f32x4.add(a0, f32x4.mul(v128.load(qh + o), v128.load(kt + o)));
        a1 = f32x4.add(a1, f32x4.mul(v128.load(qh + o + 16), v128.load(kt + o + 16)));
        a2 = f32x4.add(a2, f32x4.mul(v128.load(qh + o + 32), v128.load(kt + o + 32)));
        a3 = f32x4.add(a3, f32x4.mul(v128.load(qh + o + 48), v128.load(kt + o + 48)));
      }
      for (; j < hs4; j += 4) a0 = f32x4.add(a0, f32x4.mul(v128.load(qh + (<usize>j << 2)), v128.load(kt + (<usize>j << 2))));
      let sc: f32 = hsum(f32x4.add(f32x4.add(a0, a1), f32x4.add(a2, a3)));
      for (; j < hs; j++) sc += load<f32>(qh + (<usize>j << 2)) * load<f32>(kt + (<usize>j << 2));
      store<f32>(att + (<usize>(h * count + t) << 2), sc * isq);
    }
  }
  // 2. softmax, head by head, four exponentials at a time
  const count4 = count & ~3;
  for (let h = 0; h < nh; h++) {
    const scores = att + (<usize>(h * count) << 2);
    let mx: f32 = -f32.MAX_VALUE;
    for (let t = 0; t < count; t++) mx = max<f32>(mx, load<f32>(scores + (<usize>t << 2)));
    const mxs = f32x4.splat(mx);
    let sums = f32x4.splat(0);
    let t = 0;
    for (; t < count4; t += 4) {
      const e = vexp(f32x4.sub(v128.load(scores + (<usize>t << 2)), mxs));
      v128.store(scores + (<usize>t << 2), e);
      sums = f32x4.add(sums, e);
    }
    let sum: f32 = hsum(sums);
    for (; t < count; t++) {
      const e = fexp(load<f32>(scores + (<usize>t << 2)) - mx);
      store<f32>(scores + (<usize>t << 2), e);
      sum += e;
    }
    const inv: f32 = <f32>1.0 / sum;
    const invs = f32x4.splat(inv);
    for (t = 0; t < count4; t += 4) v128.store(scores + (<usize>t << 2), f32x4.mul(v128.load(scores + (<usize>t << 2)), invs));
    for (; t < count; t++) store<f32>(scores + (<usize>t << 2), load<f32>(scores + (<usize>t << 2)) * inv);
  }
  // 3. the weighted sum of the values, again row by row, four positions at a time: out is loaded and stored once
  // for the four of them
  const size = nh * hs;
  for (let j = 0; j < size; j++) store<f32>(out + (<usize>j << 2), 0);
  const stride = <usize>kvDim << 2;
  let t = 0;
  for (; t < count4; t += 4) {
    const row = vc + (<usize>(t * kvDim) << 2);
    for (let h = 0; h < nh; h++) {
      const weights = att + (<usize>(h * count + t) << 2);
      const w0 = f32x4.splat(load<f32>(weights)), w1 = f32x4.splat(load<f32>(weights + 4));
      const w2 = f32x4.splat(load<f32>(weights + 8)), w3 = f32x4.splat(load<f32>(weights + 12));
      const oh = out + (<usize>(h * hs) << 2);
      const v0 = row + (<usize>((h / kvMul) * hs) << 2);
      const v1 = v0 + stride, v2 = v1 + stride, v3 = v2 + stride;
      let j = 0;
      for (; j < hs4; j += 4) {
        const o = <usize>j << 2;
        const pair0 = f32x4.add(f32x4.mul(w0, v128.load(v0 + o)), f32x4.mul(w1, v128.load(v1 + o)));
        const pair1 = f32x4.add(f32x4.mul(w2, v128.load(v2 + o)), f32x4.mul(w3, v128.load(v3 + o)));
        v128.store(oh + o, f32x4.add(v128.load(oh + o), f32x4.add(pair0, pair1)));
      }
      for (; j < hs; j++) {
        const o = <usize>j << 2;
        store<f32>(oh + o, load<f32>(oh + o) + load<f32>(weights) * load<f32>(v0 + o) + load<f32>(weights + 4) * load<f32>(v1 + o)
          + load<f32>(weights + 8) * load<f32>(v2 + o) + load<f32>(weights + 12) * load<f32>(v3 + o));
      }
    }
  }
  for (; t < count; t++) {
    const row = vc + (<usize>(t * kvDim) << 2);
    for (let h = 0; h < nh; h++) {
      const weight: f32 = load<f32>(att + (<usize>(h * count + t) << 2));
      const a = f32x4.splat(weight);
      const oh = out + (<usize>(h * hs) << 2);
      const vt = row + (<usize>((h / kvMul) * hs) << 2);
      let j = 0;
      for (; j < hs4; j += 4) {
        const o = <usize>j << 2;
        v128.store(oh + o, f32x4.add(v128.load(oh + o), f32x4.mul(a, v128.load(vt + o))));
      }
      for (; j < hs; j++) {
        const o = <usize>j << 2;
        store<f32>(oh + o, load<f32>(oh + o) + weight * load<f32>(vt + o));
      }
    }
  }
}

export function layernorm(out: usize, x: usize, w: usize, b: usize, n: i32): void {
  // GPT-2 normalizes by the mean and the variance, and adds a bias after the scale
  let sum = f32x4.splat(0);
  let j = 0;
  const n4 = n & ~3;
  for (; j < n4; j += 4) { sum = f32x4.add(sum, v128.load(x + (<usize>j << 2))); }
  let total: f32 = hsum(sum);
  for (; j < n; j++) { total += load<f32>(x + (<usize>j << 2)); }
  const mean: f32 = total / <f32>n;
  let acc = f32x4.splat(0);
  const meanv = f32x4.splat(mean);
  for (j = 0; j < n4; j += 4) {
    const d = f32x4.sub(v128.load(x + (<usize>j << 2)), meanv);
    acc = f32x4.add(acc, f32x4.mul(d, d));
  }
  let ss: f32 = hsum(acc);
  for (; j < n; j++) { const d = load<f32>(x + (<usize>j << 2)) - mean; ss += d * d; }
  const s: f32 = <f32>1.0 / sqrt<f32>(ss / <f32>n + <f32>1e-5);
  for (j = 0; j < n; j++) {
    const o = <usize>j << 2;
    store<f32>(out + o, load<f32>(w + o) * (s * (load<f32>(x + o) - mean)) + load<f32>(b + o));
  }
}

export function gelu(out: usize, x: usize, b: usize, n: i32): void {
  // GPT-2's gelu_new, with the bias of the projection added first. 0.5 * (1 + tanh(z)) is 1 / (1 + exp(-2z)),
  // so the same table-free exp as swiglu does it.
  for (let j = 0; j < n; j++) {
    const o = <usize>j << 2;
    const v = load<f32>(x + o) + load<f32>(b + o);
    const inner = <f32>0.7978845608028654 * (v + <f32>0.044715 * v * v * v);
    store<f32>(out + o, v / (<f32>1.0 + fexp(<f32>-2.0 * inner)));
  }
}

export function swiglu(out: usize, h1: usize, h3: usize, n: i32): void {
  for (let j = 0; j < n; j++) {
    const o = <usize>j << 2;
    const v = load<f32>(h1 + o);
    store<f32>(out + o, v / (<f32>1.0 + fexp(-v)) * load<f32>(h3 + o));
  }
}

// y[j] += sum over c of a[c] * rows[c][j]: the outlier channels of the classifier's input, multiplied by their own
// columns of the classifier in float32 (OUTLIER_CHANNELS in llama2_numpy.py). rows is (count, n) row-major.
export function add_columns(y: usize, rows: usize, a: usize, count: i32, n: i32): void {
  const n4 = n & ~3;
  let j = 0;
  for (; j < n4; j += 4) {
    const o = <usize>j << 2;
    let acc = v128.load(y + o);
    for (let c = 0; c < count; c++) {
      const row = rows + ((<usize>c * <usize>n) << 2);
      acc = f32x4.add(acc, f32x4.mul(f32x4.splat(load<f32>(a + (<usize>c << 2))), v128.load(row + o)));
    }
    v128.store(y + o, acc);
  }
  for (; j < n; j++) {
    const o = <usize>j << 2;
    let v = load<f32>(y + o);
    for (let c = 0; c < count; c++) v += load<f32>(a + (<usize>c << 2)) * load<f32>(rows + ((<usize>c * <usize>n) << 2) + o);
    store<f32>(y + o, v);
  }
}

export function add_inplace(x: usize, y: usize, n: i32): void {
  for (let j = 0; j < n; j++) { const o = <usize>j << 2; store<f32>(x + o, load<f32>(x + o) + load<f32>(y + o)); }
}

export function argmax(x: usize, n: i32): i32 {
  let mi = 0; let mv = load<f32>(x);
  for (let j = 1; j < n; j++) { const v = load<f32>(x + (<usize>j << 2)); if (v > mv) { mv = v; mi = j; } }
  return mi;
}

// ---------------------------------------------------------------------------------------------- sampling
// What Llama.sample() does in NumPy, without walking a vocabulary of 50000 or 100000 tokens several times.

// tokens: the recently generated ones; each is made less likely once, however often it occurs
export function penalize(logits: usize, tokens: usize, count: i32, penalty: f32): void {
  for (let i = 0; i < count; i++) {
    const token = load<i32>(tokens + (<usize>i << 2));
    let seen = false;
    for (let j = 0; j < i; j++) {
      if (load<i32>(tokens + (<usize>j << 2)) == token) { seen = true; break; }
    }
    if (seen) continue;
    const address = logits + (<usize>token << 2);
    const value = load<f32>(address);
    store<f32>(address, value > 0 ? value / penalty : value * penalty);
  }
}

// fexp() on four numbers at a time, for x <= 0
// @ts-ignore: decorator
@inline function vexp(x: v128): v128 {
  x = f32x4.max(x, f32x4.splat(-87.0));
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

// The state of sortNucleus(): globals are no static data
let nucleusMass: f64 = 0;
let nucleusLimit: f64 = 0;
let nucleusLast: i32 = -1;

// adds probs[lo..hi], already in their final order, to the nucleus: true when it is complete
// @ts-ignore: decorator
@inline function intoNucleus(probs: usize, lo: i32, hi: i32): bool {
  for (let k = lo; k <= hi; k++) {
    nucleusMass += load<f32>(probs + (<usize>k << 2));
    if (nucleusMass >= nucleusLimit) { nucleusLast = k; return true; }
  }
  return false;
}

// Sorts probs[lo..hi] in descending order, and index[] along with it, from the left and only as far as the nucleus
// reaches: usually a few dozen tokens out of thousands. Quicksort (median of three), insertion sort for short ranges.
function sortNucleus(probs: usize, index: usize, lo: i32, hi: i32): void {
  while (hi - lo > 12) {
    const mid = lo + ((hi - lo) >> 1);
    let a = load<f32>(probs + (<usize>lo << 2)), b = load<f32>(probs + (<usize>mid << 2));
    const c = load<f32>(probs + (<usize>hi << 2));
    if (a < b) { const t = a; a = b; b = t; }
    if (b < c) { b = c; if (a < b) b = a; }
    const pivot = b;
    let i = lo, j = hi;
    while (i <= j) {
      while (load<f32>(probs + (<usize>i << 2)) > pivot) i++;
      while (load<f32>(probs + (<usize>j << 2)) < pivot) j--;
      if (i <= j) {
        const pi = probs + (<usize>i << 2), pj = probs + (<usize>j << 2), xi = index + (<usize>i << 2), xj = index + (<usize>j << 2);
        const p = load<f32>(pi); store<f32>(pi, load<f32>(pj)); store<f32>(pj, p);
        const x = load<i32>(xi); store<i32>(xi, load<i32>(xj)); store<i32>(xj, x);
        i++; j--;
      }
    }
    sortNucleus(probs, index, lo, j);
    if (nucleusLast >= 0) return;
    if (intoNucleus(probs, j + 1, i - 1)) return; // equal to the pivot
    lo = i;
  }
  for (let i = lo + 1; i <= hi; i++) {
    const p = load<f32>(probs + (<usize>i << 2));
    const x = load<i32>(index + (<usize>i << 2));
    let j = i - 1;
    while (j >= lo && load<f32>(probs + (<usize>j << 2)) < p) {
      store<f32>(probs + (<usize>(j + 1) << 2), load<f32>(probs + (<usize>j << 2)));
      store<i32>(index + (<usize>(j + 1) << 2), load<i32>(index + (<usize>j << 2)));
      j--;
    }
    store<f32>(probs + (<usize>(j + 1) << 2), p);
    store<i32>(index + (<usize>(j + 1) << 2), x);
  }
  intoNucleus(probs, lo, hi);
}

// Draws a token from softmax(logits / temperature), restricted to the nucleus when 0 < topp < 1.
// random: one number in [0, 1) from Python's generator, so that a seed reproduces. probs and index: scratch of n each.
export function sample(logits: usize, n: i32, temperature: f32, topp: f32, random: f64, probs: usize, index: usize): i32 {
  let best = load<f32>(logits);
  let i = 0;
  if (n >= 4) {
    let bests = v128.load(logits);
    for (i = 4; i + 4 <= n; i += 4) bests = f32x4.max(bests, v128.load(logits + (<usize>i << 2)));
    best = max(max(f32x4.extract_lane(bests, 0), f32x4.extract_lane(bests, 1)), max(f32x4.extract_lane(bests, 2), f32x4.extract_lane(bests, 3)));
  }
  for (; i < n; i++) best = max(best, load<f32>(logits + (<usize>i << 2)));
  const nucleus = topp > 0 && topp < 1;
  // With a nucleus, tokens less than a ten millionth as probable as the best one cannot matter (ln 1e-7 = -16.118):
  // they are left out before exp(), which is the expensive part
  const floor: f32 = nucleus ? best - temperature * <f32>16.118095 : -f32.MAX_VALUE;
  let count = 0;
  for (i = 0; i < n; i++) {
    const v = load<f32>(logits + (<usize>i << 2));
    if (v >= floor) {
      store<f32>(probs + (<usize>count << 2), v - best);
      store<i32>(index + (<usize>count << 2), i);
      count++;
    }
  }
  const inverse = f32x4.splat(<f32>1.0 / temperature);
  for (i = 0; i + 4 <= count; i += 4) {
    const address = probs + (<usize>i << 2);
    v128.store(address, vexp(f32x4.mul(v128.load(address), inverse)));
  }
  for (; i < count; i++) {
    const address = probs + (<usize>i << 2);
    store<f32>(address, fexp(load<f32>(address) / temperature));
  }
  let total: f64 = 0;
  for (i = 0; i < count; i++) total += load<f32>(probs + (<usize>i << 2));
  let last = count - 1;
  let mass = total;
  if (nucleus) {
    // Tokens below (1 - topp) / (n - 1) cannot be part of the nucleus (llama2.c), so they need not be sorted
    const cutoff: f64 = (1.0 - <f64>topp) / <f64>(count > 1 ? count - 1 : 1) * total;
    let likely = 0;
    for (let k = 0; k < count; k++) {
      const p = load<f32>(probs + (<usize>k << 2));
      if (<f64>p >= cutoff) {
        store<f32>(probs + (<usize>likely << 2), p);
        store<i32>(index + (<usize>likely << 2), load<i32>(index + (<usize>k << 2)));
        likely++;
      }
    }
    // the most probable tokens whose probabilities add up to topp
    nucleusMass = 0;
    nucleusLimit = <f64>topp * total;
    nucleusLast = -1;
    sortNucleus(probs, index, 0, likely - 1);
    last = nucleusLast >= 0 ? nucleusLast : likely - 1;
    mass = nucleusMass;
  }
  const target: f64 = random * mass;
  let cumulative: f64 = 0;
  for (let k = 0; k <= last; k++) {
    cumulative += load<f32>(probs + (<usize>k << 2));
    if (cumulative > target) return load<i32>(index + (<usize>k << 2));
  }
  return load<i32>(index + (<usize>last << 2));
}
