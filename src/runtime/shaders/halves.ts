// shaders/halves.ts (T642.7): float16 in JavaScript, both ways: halvesOf (float32 values into float16 bits, as the GPU's
// keys and values are rounded) and floatOfHalf (a float16's bits as the number it is). The one copy of each in the GPU's
// worker (gpu/) and the shaders' parts (the checks' answers of superblocks.ts and attention.ts). Two others stay, each
// for its chunk: forward/memory.ts's halfToFloat (the forward chunk takes nothing of the shaders) and /benchmark/'s
// benchmark/gpu/halves.ts's fromHalf (its worker fetches the shaders when it uses them). tests/gpu-choice-check.ts
// holds the three the same on all 65,536 float16s.
// A part of src/runtime/shaders.ts, which is the window. It imports nothing.

/** T152's review (T160): float32 values into float16 bits, rounded to the nearest (ties to even), as the GPU's own keys
 * and values are (pack2x16float) where it rounds so: a float32 cache's keys and values going up to the GPU. out: a
 * Uint16Array as long as floats. Infinities and NaN stay so, what is past float16's largest becomes an infinity, and
 * what is below half its least subnormal 0. */
const floatWord = new Float32Array(1), floatBits = new Uint32Array(floatWord.buffer);
export function halvesOf(floats: ArrayLike<number>, out: Uint16Array) {
  for (let i = 0; i < floats.length; i++) {
    floatWord[0] = floats[i];
    const bits = floatBits[0], sign = (bits >>> 16) & 0x8000, exponent = (bits >>> 23) & 0xff, fraction = bits & 0x7fffff;
    let half;
    if (exponent === 0xff) half = 0x7c00 | (fraction ? 0x200 : 0);
    else {
      const e = exponent - 112;  // float16's biased exponent
      if (e >= 0x1f) half = 0x7c00;
      else if (e <= 0) {
        // a subnormal (or 0): the fraction with its leading 1, shifted to units of 2^-24
        const shift = 14 - e;
        if (shift > 24) half = 0;
        else {
          const whole = fraction | 0x800000, rest = whole & ((1 << shift) - 1), middle = 1 << (shift - 1);
          half = whole >>> shift;
          if (rest > middle || (rest === middle && (half & 1))) half += 1;
        }
      } else {
        const rest = fraction & 0x1fff;
        half = (e << 10) | (fraction >>> 13);
        // (a carry into the exponent is right: to the next power of two, or to the infinity at the top)
        if (rest > 0x1000 || (rest === 0x1000 && (half & 1))) half += 1;
      }
    }
    out[i] = sign | half;
  }
  return out;
}
/** T642.7: a float16's bits (the low 16 of bits) as the number it is, exactly (every float16 is a float32): ±0, the
 * subnormals (fraction × 2^-24), ±Infinity and NaN (exponent 31) too, as NumPy's astype(float32) reads them. */
export function floatOfHalf(bits: number) {
  const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >> 10) & 0x1f, fraction = bits & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}
