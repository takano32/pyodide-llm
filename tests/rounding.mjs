// T225's review: a WebGPU (Dawn's in Node: the npm package webgpu, as tests/bench-dawn.mjs and tests/gpu-check.mjs use it)
// whose device rounds a float32 to a float16 the way another implementation may, where CI's only implementations round
// to the nearest (Mesa's lavapipe: LLVM's fptrunc; SwiftShader). WGSL leaves it to the device which of the two float16 next
// to a float32 a conversion gives (§15.7.6 Floating Point Conversion: "WGSL does not specify whether the higher or lower
// representable value is chosen, and different instances of such a conversion may choose differently"), and Direct3D,
// where Chrome on Windows runs WebGPU, takes the one toward zero (D3D11.3 functional specification, 3.2.2: "Round-to-zero
// must be used during conversion to another float format"; Dawn's HLSL writer makes pack2x16float of f32tof16): the
// owner's NVIDIA PC did, and /benchmark/'s layer check, which rounded to the nearest in JavaScript, called it WRONG (T225).
//
//   import { roundedGpu } from "./rounding.mjs";   navigator.gpu = roundedGpu(create([]), how)
//   how: "toward-zero" | "away" (every pack2x16float, which writes the keys and values of the cache, rounds so)
//        "everything" (toward zero, and every scalar conversion into the tiles' workgroup memory, shmem_t(x), too: a device
//        that rounds all so; the conversions of vectors, vec4<shmem_t>(...), are left to the device's own)
//        "" | "nearest" (the device's own: nothing changed)
// A shader's text is rewritten as it is given to createShaderModule (the calls and the helper function appended, which
// WGSL's module scope allows in any order), so no shader of the repository has to say anything of it. The rounding is cut
// by hand in 32-bit integer arithmetic: toward zero is the bits' truncation (a float32's exponent and 10 bits of its
// fraction, the subnormals shifted), away from zero is that and one more where the value was not exactly a float16.
// (A "round to the nearest and one lower where it rounded up" wrapper was the first one tried: it moved a quarter of the
// keys and values, not half, for a reason not found, so the number that did round the other way is counted by the check.)

const TOWARD_ZERO = `
fn rounded16(x: f32) -> u32 {
  let b = bitcast<u32>(x);
  let sign = (b >> 16u) & 0x8000u;
  let e = i32((b >> 23u) & 0xffu) - 112;
  let m = b & 0x7fffffu;
  if (e >= 31) { return sign | 0x7bffu; }
  if (e <= 0) {
    if (e < -10) { return sign; }
    return sign | ((m | 0x800000u) >> u32(14 - e));
  }
  return sign | (u32(e) << 10u) | (m >> 13u);
}`;
const AWAY = `
fn toward16(x: f32) -> u32 {
  let b = bitcast<u32>(x);
  let sign = (b >> 16u) & 0x8000u;
  let e = i32((b >> 23u) & 0xffu) - 112;
  let m = b & 0x7fffffu;
  if (e >= 31) { return sign | 0x7bffu; }
  if (e <= 0) {
    if (e < -10) { return sign; }
    return sign | ((m | 0x800000u) >> u32(14 - e));
  }
  return sign | (u32(e) << 10u) | (m >> 13u);
}
fn rounded16(x: f32) -> u32 {
  let h = toward16(x);
  return select(h, h + 1u, unpack2x16float(h).x != x && h != 0x7bffu);
}`;
const PACK = `
fn pack2x16float_rounded(v: vec2<f32>) -> u32 { return rounded16(v.x) | (rounded16(v.y) << 16u); }`;

/** The text of a shader as a device of this kind is given it (how: see above). */
export function rounded(code, how) {
  if (!how || how === "nearest") return code;
  if (!["toward-zero", "away", "everything"].includes(how)) throw new Error(`unknown rounding ${how}`);
  const packs = /\bpack2x16float\(/.test(code), tiles = how === "everything" && /\balias shmem_t\b/.test(code);
  if (!packs && !tiles) return code;
  let out = code.replace(/\bpack2x16float\(/g, "pack2x16float_rounded(");
  if (tiles) out = out.replace(/\bshmem_t\(/g, "to_shmem(");
  // (shmem_t is f32 in the tiles of the float32 forms: their conversion is no conversion at all)
  const toTile = !tiles ? "" : /\balias shmem_t = f16;/.test(code)
    ? "\nfn to_shmem(x: f32) -> shmem_t { return shmem_t(unpack2x16float(rounded16(x)).x); }"
    : "\nfn to_shmem(x: f32) -> shmem_t { return x; }";
  return out + (how === "away" ? AWAY : TOWARD_ZERO) + (packs ? PACK : "") + toTile;
}

/** A GPU (navigator.gpu) whose devices are given the shaders rounded by how. */
export function roundedGpu(gpu, how) {
  if (!how || how === "nearest") return gpu;
  rounded("", how);  // (a name that is not one stops here)
  const own = (target, key) => {
    const value = Reflect.get(target, key);
    return typeof value === "function" ? value.bind(target) : value;
  };
  const device = (real) => {
    const make = real.createShaderModule.bind(real);
    real.createShaderModule = (descriptor) => make({ ...descriptor, code: rounded(descriptor.code, how) });
    return real;
  };
  const adapter = (real) => new Proxy(real, {
    get: (target, key) => (key === "requestDevice" ? async (...args) => device(await target.requestDevice(...args)) : own(target, key)),
  });
  return new Proxy(gpu, {
    get: (target, key) => (key === "requestAdapter" ? async (...args) => {
      const found = await target.requestAdapter(...args);
      return found && adapter(found);
    } : own(target, key)),
  });
}
