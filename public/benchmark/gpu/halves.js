// The float16 of the cache: a float32 to a half and back in JavaScript, and what a device may round a key or a value to
// (WGSL leaves the direction to it, T225): pure, and tested in Node (tests/gpu-choice-check.mjs, tests/layer-check.mjs).
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)

// float32 to float16's bits, rounded to the nearest (ties to even), as a cache holds its keys and values
const f32 = new Float32Array(1), bits32 = new Uint32Array(f32.buffer);
function toHalf(value) {
  f32[0] = value;
  const b = bits32[0], sign = (b >>> 16) & 0x8000, exponent = ((b >>> 23) & 0xff) - 112;
  let mantissa = b & 0x7fffff;
  if (exponent >= 31) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - exponent, kept = mantissa >> shift, rest = mantissa & ((1 << shift) - 1), middle = 1 << (shift - 1);
    return sign | (kept + (rest > middle || (rest === middle && kept & 1) ? 1 : 0));
  }
  const kept = (exponent << 10) | (mantissa >> 13), rest = mantissa & 0x1fff;
  return sign | (kept + (rest > 0x1000 || (rest === 0x1000 && kept & 1) ? 1 : 0));
}
function fromHalf(h) {
  const exponent = (h >> 10) & 31, mantissa = h & 1023, sign = h & 0x8000 ? -1 : 1;
  return sign * (exponent ? 2 ** (exponent - 15) * (1 + mantissa / 1024) : 2 ** -14 * (mantissa / 1024));
}
// T225: the keys or values of a position as the GPU wrote them (got: float16 bits), against the reference's own (x,
// float64). WGSL leaves to the implementation which of the two float16 next to a float32 a conversion gives (§15.7.6
// Floating Point Conversion: "WGSL does not specify whether the higher or lower representable value is chosen, and
// different instances of such a conversion may choose differently"; §17.9.9: pack2x16float converts so), and Direct3D
// rounds toward zero (the D3D11.3 functional specification, 3.2.2 Floating Point Conversion: "Round-to-zero must be
// used during conversion to another float format"; 22.13.2: f32tof16 "Follows D3D rules for floating point
// conversion", which is what Dawn's HLSL writer makes of pack2x16float), where JavaScript's toHalf() and the software
// adapters here (lavapipe, SwiftShader) round to the nearest. A reference that rounds its own way is then off by up to
// a float16's ulp (2^-10 of the value) in every key and value of the position, which the attention carries into its
// output, the keys' shift of the scores the most (a score moves by up to 2^-10 × Σ|q·k| ÷ √head: 5e-2 to 1e-1 at the
// largest head of the layer check, 7e-3 to 3e-2 as it falls out): on the layer check's numbers, in JavaScript, keys
// and values rounded toward zero move the scales of the attention's quantized output by 5.7e-4 to 1.2e-2, in 97% of
// 300 draws past QUANTIZED_SCALE_LINE (1e-3), and the stream by 3.4e-4 to 2.5e-3 of what the layer added, in 45% past
// LAYER_LINE (1e-3), with the cache at 4.5e-4 to 9.7e-4 of its largest: what the owner's NVIDIA PC on Windows reported
// twice (T225; the T225 review's 300 draws, 2026-10-01). So the reference takes the GPU's bits wherever they are a
// float16 next to its own value: no farther from x than a float16's ulp at x and HALF_SLACK of the row's largest, for
// the GPU's float32 value
// is not x itself (a float32 sum of n products in another order is off by about sqrt(n) × 2^-24 of the terms' spread,
// 2.7e-6 of it at n = 2112, about 1e-6 of the row's largest: the slack is ten times that, 1 to 2% of an ulp at the
// largest). Any other value stays the reference's own nearest, and the check goes on as it did (a key or value read
// from the wrong place, turned by the wrong angle or written to the wrong row is off by far more than an ulp). Returns
// { bits: what the reference goes on with, nearest: its own, same / inward / outward: the GPU's that are the nearest,
// the neighbour toward zero, the neighbour away from it, far: neither }
const HALF_SLACK = 1e-5;
function heldHalves(x, got) {
  let largest = 0;
  for (const value of x) largest = Math.max(largest, Math.abs(value));
  const nearest = Uint16Array.from(x, toHalf), bits = nearest.slice(), counts = { same: 0, inward: 0, outward: 0, far: 0 };
  got.forEach((half, i) => {
    if (half === nearest[i]) return counts.same++;
    // a float16's ulp at x (its subnormals' below 2^-14)
    const value = fromHalf(half), ulp = 2 ** (Math.max(Math.floor(Math.log2(Math.abs(x[i]))), -14) - 10);
    if (!(Math.abs(value - x[i]) <= ulp + HALF_SLACK * largest)) return counts.far++;
    bits[i] = half;
    return Math.abs(value) < Math.abs(x[i]) ? counts.inward++ : counts.outward++;
  });
  return { bits, nearest, ...counts };
}
// T225: how the GPU rounded its keys and values (heldHalves' counts, summed), in a few words
const halvesSaid = (counts) => {
  const sum = (key) => counts.reduce((total, c) => total + c[key], 0), [same, inward, outward, far] = ["same", "inward", "outward", "far"].map(sum);
  return inward + outward + far === 0 ? `K and V ${same} to the nearest float16`
    : `K and V ${same} to the nearest float16, ${inward} toward zero, ${outward} away from it, ${far} farther`;
};
// T225: the largest difference of two vectors over the largest magnitude of the second
const farthest = (got, want) => {
  let off = 0, largest = 0;
  want.forEach((value, i) => {
    off = Math.max(off, Math.abs(got[i] - value));
    largest = Math.max(largest, Math.abs(value));
  });
  return off / largest;
};

export { toHalf, fromHalf, heldHalves, halvesSaid, farthest };
