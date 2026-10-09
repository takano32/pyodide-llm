// kernel/halves.ts (T356): float16. The cache's keys and values are written as halves (to_f16), read back by
// attention_f16 four at a time (halves4, which kernel/attention.ts imports) and by from_f16, and looked at before
// they are trusted (finite_f16). The converter's float16 (the scale of a GGUF block, halfToFloat) is not this one:
// it keeps infinities and NaNs, and is in kernel/quantize.ts.

// float16 to float32 without the FP16 proposal (T104 is on hold): the bits of the magnitude, moved to where float32
// keeps them, are the number times 2^-112, which one multiplication by 2^112 puts right (exactly, subnormals and
// zero included). An infinity (what to_f16 writes past 65520) reads back as 65536, and NaN is not kept: neither is
// expected of a key or value.
// T160: in four instructions, not seven. The half is loaded sign-extended, so that the shift by 13 puts its sign in
// bit 31 as well as in bits 30..28 (the copies the extension made); one AND clears those three, and the sign rides
// through the multiplication (-a times 2^112 is -(a times 2^112) to the bit, -0 included). The same bits as the
// seven-instruction form (magnitude, multiply, then OR the sign back in) for every one of the 65536 halves.
const HALF_BITS: i32 = <i32>0x8fffffff;  // the sign, and the exponent and mantissa of a half shifted by 13
const HALF_SCALE: f32 = 5.192296858534828e33;  // 2^112 (0x77800000)
// @ts-ignore: decorator
@inline function halves4(p: usize): v128 {
  return f32x4.mul(v128.and(i32x4.shl(v128.load16x4_s(p), 13), i32x4.splat(HALF_BITS)), f32x4.splat(HALF_SCALE));
}
// @ts-ignore: decorator
@inline function half(p: usize): f32 {
  return reinterpret<f32>((<i32>load<i16>(p) << 13) & HALF_BITS) * HALF_SCALE;
}

// T110: float32 to float16, rounded to the nearest (ties to even), as NumPy's astype(float16) does: the keys and
// values of a token, into the cache. Too large a number becomes infinity (a key never is one).
export function to_f16(out: usize, x: usize, n: i32): void {
  for (let i = 0; i < n; i++) {
    const bits = reinterpret<u32>(load<f32>(x + (<usize>i << 2)));
    const sign = (bits >> 16) & 0x8000;
    const exponent = <i32>((bits >> 23) & 0xff) - 112;  // rebased for float16 (127 - 15)
    let mantissa = bits & 0x7fffff;
    let result: u32;
    if (exponent >= 31) {
      result = sign | 0x7c00;
    } else if (exponent <= 0) {
      // a subnormal float16 (or zero): the implicit bit comes in, and the rest is shifted out with rounding
      if (exponent < -10) {
        result = sign;
      } else {
        mantissa |= 0x800000;
        const shift = <u32>(14 - exponent);
        const rest = mantissa & ((1 << shift) - 1), middle = <u32>1 << (shift - 1);
        let value = mantissa >> shift;
        if (rest > middle || (rest == middle && (value & 1))) value++;
        result = sign | value;
      }
    } else {
      const rest = mantissa & 0x1fff;
      let value = (<u32>exponent << 10) | (mantissa >> 13);
      if (rest > 0x1000 || (rest == 0x1000 && (value & 1))) value++;  // a carry moves into the exponent, as it should
      result = sign | value;
    }
    store<u16>(out + (<usize>i << 1), <u16>result);
  }
}

// T160 (the review of (4)): float16 to float32, n values: the keys and values the GPU writes back (float16) into a
// float32 cache (a grouped-query model's), widened as attention_f16 widens them. In JavaScript one at a time this took
// 27 to 33 ns a value on CI's runners, 1.7 ms a token of Qwen3 0.6B's prompt.
export function from_f16(out: usize, x: usize, n: i32): void {
  let i = 0;
  for (; i + 4 <= n; i += 4) v128.store(out + (<usize>i << 2), halves4(x + (<usize>i << 1)));
  for (; i < n; i++) store<f32>(out + (<usize>i << 2), half(x + (<usize>i << 1)));
}

// T243: whether n float16 values are all finite numbers: 1, or 0 where one has every bit of its exponent set (a NaN or
// an infinity, which halves4 above reads as a finite number: 65536 and more). The keys and values a GPU wrote back,
// looked at once before they go into the cache (stagingFinite of public/forward/engine.js), eight a step: the attention's loops, which
// read the cache at every token, stay as they are
export function finite_f16(x: usize, n: i32): i32 {
  const exponent = i16x8.splat(0x7c00);
  let found = i16x8.splat(0);
  let i = 0;
  for (; i + 8 <= n; i += 8) {
    found = v128.or(found, i16x8.eq(v128.and(v128.load(x + (<usize>i << 1)), exponent), exponent));
  }
  let rest: i32 = 0;
  for (; i < n; i++) rest |= <i32>((<i32>load<u16>(x + (<usize>i << 1)) & 0x7c00) == 0x7c00);
  return <i32>(!v128.any_true(found) && rest == 0);
}

export { halves4, half };
