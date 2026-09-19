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
  const qmax: f32 = bias == 0 ? 127.0 : 63.0;
  for (let g = 0; g < n; g += GS) {
    let amax: f32 = 0;
    for (let j = 0; j < GS; j++) {
      const v = abs<f32>(load<f32>(x + (<usize>(g + j) << 2)));
      if (v > amax) amax = v;
    }
    const scale: f32 = amax / qmax;
    store<f32>(xs + (<usize>(g / GS) << 2), scale);
    const inv: f32 = scale > 0 ? <f32>1.0 / scale : 0;
    for (let j = 0; j < GS; j++) {
      const q = <i32>nearest<f32>(load<f32>(x + (<usize>(g + j) << 2)) * inv) + bias;
      store<i8>(xq + <usize>(g + j), <i8>q);
    }
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

export function rope(v: usize, fcr: usize, fci: usize, nh: i32, hs: i32): void {
  for (let h = 0; h < nh; h++) {
    const base = v + (<usize>(h * hs) << 2);
    for (let i = 0; i < hs; i += 2) {
      const c = load<f32>(fcr + (<usize>(i >> 1) << 2)), s = load<f32>(fci + (<usize>(i >> 1) << 2));
      const p0 = base + (<usize>i << 2);
      const v0 = load<f32>(p0), v1 = load<f32>(p0 + 4);
      store<f32>(p0, v0 * c - v1 * s);
      store<f32>(p0 + 4, v0 * s + v1 * c);
    }
  }
}

// kc / vc: this layer's cache laid out [seq][nh*hs] (NOT llama2_numpy.py's [heads][seq][hs]); att: scratch of
// at least pos+1 floats; hs % 4 == 0; n_kv_heads == n_heads (no grouped-query attention yet)
export function attention(out: usize, q: usize, kc: usize, vc: usize, att: usize, pos: i32, nh: i32, hs: i32): void {
  const dim = nh * hs;
  const isq: f32 = <f32>1.0 / sqrt<f32>(<f32>hs);
  for (let h = 0; h < nh; h++) {
    const qh = q + (<usize>(h * hs) << 2);
    let mx: f32 = -f32.MAX_VALUE;
    for (let t = 0; t <= pos; t++) {
      const kt = kc + (<usize>(t * dim + h * hs) << 2);
      let acc = f32x4.splat(0);
      for (let j = 0; j < hs; j += 4) acc = f32x4.add(acc, f32x4.mul(v128.load(qh + (<usize>j << 2)), v128.load(kt + (<usize>j << 2))));
      const sc = hsum(acc) * isq;
      store<f32>(att + (<usize>t << 2), sc);
      if (sc > mx) mx = sc;
    }
    let sum: f32 = 0;
    for (let t = 0; t <= pos; t++) {
      const e = fexp(load<f32>(att + (<usize>t << 2)) - mx);
      store<f32>(att + (<usize>t << 2), e);
      sum += e;
    }
    const inv: f32 = <f32>1.0 / sum;
    const oh = out + (<usize>(h * hs) << 2);
    for (let j = 0; j < hs; j += 4) v128.store(oh + (<usize>j << 2), f32x4.splat(0));
    for (let t = 0; t <= pos; t++) {
      const a = f32x4.splat(load<f32>(att + (<usize>t << 2)) * inv);
      const vt = vc + (<usize>(t * dim + h * hs) << 2);
      for (let j = 0; j < hs; j += 4) {
        const o = <usize>j << 2;
        v128.store(oh + o, f32x4.add(v128.load(oh + o), f32x4.mul(a, v128.load(vt + o))));
      }
    }
  }
}

export function swiglu(out: usize, h1: usize, h3: usize, n: i32): void {
  for (let j = 0; j < n; j++) {
    const o = <usize>j << 2;
    const v = load<f32>(h1 + o);
    store<f32>(out + o, v / (<f32>1.0 + fexp(-v)) * load<f32>(h3 + o));
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
