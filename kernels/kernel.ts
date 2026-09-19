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

// kc / vc: this layer's cache laid out [seq][nkv * hs] (NOT the NumPy forward's [kv heads][seq][hs]);
// att: scratch of at least pos + 1 floats. Grouped-query attention: nh / nkv query heads share one kv head.
export function attention(out: usize, q: usize, kc: usize, vc: usize, att: usize, pos: i32, nh: i32, nkv: i32, hs: i32): void {
  const kvDim = nkv * hs;
  const kvMul = nh / nkv;
  const hs4 = hs & ~3;
  const isq: f32 = <f32>1.0 / sqrt<f32>(<f32>hs);
  for (let h = 0; h < nh; h++) {
    const qh = q + (<usize>(h * hs) << 2);
    const head = (h / kvMul) * hs;
    let mx: f32 = -f32.MAX_VALUE;
    for (let t = 0; t <= pos; t++) {
      const kt = kc + (<usize>(t * kvDim + head) << 2);
      let acc = f32x4.splat(0);
      let j = 0;
      for (; j < hs4; j += 4) acc = f32x4.add(acc, f32x4.mul(v128.load(qh + (<usize>j << 2)), v128.load(kt + (<usize>j << 2))));
      let sc: f32 = hsum(acc);
      for (; j < hs; j++) sc += load<f32>(qh + (<usize>j << 2)) * load<f32>(kt + (<usize>j << 2));
      sc *= isq;
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
    for (let j = 0; j < hs; j++) store<f32>(oh + (<usize>j << 2), 0);
    for (let t = 0; t <= pos; t++) {
      const weight: f32 = load<f32>(att + (<usize>t << 2)) * inv;
      const a = f32x4.splat(weight);
      const vt = vc + (<usize>(t * kvDim + head) << 2);
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
