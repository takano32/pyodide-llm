// int8-check.ts (T.644.1): the int8 and int6 kernels the default model takes at every token (kernel_relaxed.ts's
// matmul_q8r, matmul_q8r_tile and matmul_q6r; kernel/matmul.ts's matmul_q8, matmul_q6, int8_sums and six_sums; and
// kernel/quantize.ts's quantize_x, whose bytes they read) against the same arithmetic written out here, in the same
// order, value by value: int8's numbers change with the order of the adds, so the model adds as the kernels do. Node and
// the compiled kernels alone, a second or so: the light suite runs it. On a 32-bit memory and on a 64-bit one, below and
// above 4 GiB (the addresses through jobs.ts's table; the canary of ternary-check.ts for V8's Liftoff on arm64).
//   To the bit, everywhere: relaxed_dot_i8x16_i7x16_add_s is not determined only where its second operand has the top bit
// set (and x86's pmaddubsw saturates a pair to int16: 2 x 128 x 127 = 32512 of 32767, which no int8 weight reaches against
// 0..127), and quantize_x(bias = 64) writes 1..127 only: every activation here is 0..127 for the relaxed kernels, and the
// integer sums they make are exact on every CPU. The plain ones (dot32, kernel/matmul.ts) add two products in int16:
// exact unless both factors are -128 twice over, which quantize_x(bias = 0) never writes (-127..127); the weights here
// take -128 and the activations do not.
//   int8_sums and six_sums: -64 times the sum of each group, an int32, for every n.
//   matmul_q8r: lane k of the accumulator adds the groups 4j + k in order (each group's exact integer sum with the
//     correction in it, dot(w, q - 64), times the float32 product of the two scales), the four lanes are added in order,
//     then the groups past the fours one at a time (kernel_relaxed.ts 26-36).
//   matmul_q8: dot32's lane k holds products 2k, 2k + 1, 2k + 8, 2k + 9 of each half of a group; each lane times the
//     product of the scales into its own float32 lane, group by group, and the four lanes added in order at the end.
//   matmul_q6, matmul_q6r: those two on the int6 weights widened (llama2_numpy.pack6's bytes, written here).
//   matmul_q8r_tile: every (row, token) is matmul_q8r's number to the bit (kernel_relaxed.ts 80-85), for 1 to 9 tokens,
//     ranges of rows that begin and end off the fours and fewer than four rows, rows of 1 to 9 groups (ng % 4 of 0 to 3)
//     and rows of 2560 and of 2592 bytes (the read ahead of a quad of rows up to 10 KB, and none past it).
//   quantize_x: the scales and bytes of the scalar steps (the largest |x| over 127 or 63, round half to even, plus the bias)
//   at both biases, a group of zeros, a group at the ends.
// Rows of 13, groups of 1 to 9, 80 and 81; the scales of every size (a zero and a subnormal among them); rows of one
// end of int8 against activations of one end (the largest sums) beside random ones.
import fs from "node:fs";
import v8 from "node:v8";
import { built, runtimeUrl } from "./tree.ts";
const { addressed } = await import(runtimeUrl("jobs.ts"));

const f = Math.fround;
let checked = 0;

// the canary of ternary-check.ts (which says why): V8's Liftoff on arm64 (Node 24) reads v128.load32_splat and
// v128.load32_lane above 4 GiB at the low 32 bits of the address, and matmul_q8r_tile takes the tokens' scales so
const HIGH = 4 * 2 ** 30 + 2 * 65536;
const CANARY = Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 126, 1, 125, 2, 15, 1, 3, 101, 110, 118, 6, 109, 101, 109, 111, 114, 121, 2, 4, 1,
  3, 2, 1, 0, 7, 9, 1, 5, 115, 112, 108, 97, 116, 0, 0, 10, 13, 1, 11, 0, 32, 0, 253, 9, 2, 0, 253, 31, 0, 11]);
function splatsRight(memory, again = false) {
  const bytes = again ? Uint8Array.from([...CANARY, 0, 2, 1, 97]) : CANARY;
  const splat = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } }).exports.splat;
  const F = new Float32Array(memory.buffer), at = HIGH + 4096;
  F[at / 4] = 1.5;
  F[(at - 2 ** 32) / 4] = 2.5;
  return splat(BigInt(at)) === 1.5;
}

// ---- the arithmetic, written out
// round half to even, as f32x4.nearest (the value is a float32 already)
const nearest = (v) => {
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
};
// int6: the 32 int8 values (multiples of 4) of a group in llama2_numpy.pack6's 24 bytes
function pack6(values, out, at) {
  for (let j = 0; j < 16; j++) out[at + j] = ((values[j] >> 2) & 15) | (((values[j + 16] >> 2) & 15) << 4);
  for (let k = 0; k < 8; k++) {
    let top = 0;
    for (let s = 0; s < 4; s++) top |= (((values[k + 8 * s] & 255) >> 6) & 3) << (2 * s);
    out[at + 16 + k] = top;
  }
}
// matmul_q8r for one row: w the row's int8 values, q the activations (0..127, q - 64 the value), ws and xs the scales
function q8r(w, q, ws, xs, ng) {
  const ng4 = ng & ~3, lanes = [0, 0, 0, 0];
  let g = 0;
  const exact = (g) => {
    let sum = 0;
    for (let j = g * 32; j < g * 32 + 32; j++) sum += w[j] * (q[j] - 64);
    return sum;
  };
  for (; g < ng4; g++) lanes[g & 3] = f(lanes[g & 3] + f(exact(g) * f(ws[g] * xs[g])));
  let sum = f(f(f(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
  for (; g < ng; g++) sum = f(sum + f(exact(g) * f(ws[g] * xs[g])));
  return sum;
}
// matmul_q8 for one row: q the activations (-127..127)
function q8(w, q, ws, xs, ng) {
  const lanes = [0, 0, 0, 0];
  for (let g = 0; g < ng; g++) {
    const s = f(ws[g] * xs[g]);
    for (let k = 0; k < 4; k++) {
      let sum = 0;
      for (const half of [0, 16]) for (const p of [2 * k, 2 * k + 1, 2 * k + 8, 2 * k + 9]) sum += w[g * 32 + half + p] * q[g * 32 + half + p];
      lanes[k] = f(lanes[k] + f(sum * s));
    }
  }
  return f(f(f(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
}

for (const [wide, base] of [[false, 0], [true, 0], [true, HIGH]]) {
  let memory, liftoffOff = false;
  try {
    memory = wide ? new WebAssembly.Memory({ initial: BigInt(Math.ceil((base + 4 * 2 ** 20) / 65536)), address: "i64" }) : new WebAssembly.Memory({ initial: 64 });
  } catch (error) {
    console.log(`int8-check: no 64-bit memory of ${(base / 2 ** 30).toFixed(1)} GiB and more in this engine: skipped (${error.message})`);
    continue;
  }
  if (base && !splatsRight(memory)) {
    v8.setFlagsFromString("--no-liftoff");
    liftoffOff = true;
    if (!splatsRight(memory, true)) {
      console.log("int8-check: this engine reads v128.load32_splat above 4 GiB at the low 32 bits of the address even with --no-liftoff: the kernels above 4 GiB cannot be checked here, skipped");
      continue;
    }
    console.log("int8-check: Liftoff reads v128.load32_splat above 4 GiB at the low 32 bits of the address (V8 13.6 on arm64, Node 24; fixed in V8 14.3): --no-liftoff is set, the kernels above 4 GiB are TurboFan's here");
  }
  const bytesOf = (name) => {
    const bytes = fs.readFileSync(built(`${name}${wide ? "64" : ""}.wasm`));
    return liftoffOff ? Uint8Array.from([...bytes, 0, 2, 1, 99]) : bytes;
  };
  const load = (name) => addressed(new WebAssembly.Instance(new WebAssembly.Module(bytesOf(name)), { env: { memory } }).exports, wide);
  const k = load("simdkernel_plain"), r = load("simdkernel_relaxed_plain");
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
  let seed = 644;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const where = wide ? (base ? " (64-bit memory, above 4 GiB)" : " (64-bit memory)") : "";
  const scale = (i) => (i % 11 === 7 ? 2 ** -130 * (1 + (next() % 1000)) : i % 13 === 5 ? 0 : f(1e-3 * (1 + (next() % 1000))));

  const rows = 13, most = 81, tokens = 9, frame = 8192, outFrame = 64;
  const w = base + 65536, ws = w + 40960, wc = ws + 8192, w6 = wc + 8192, c6 = w6 + 32768;
  const frames = c6 + 8192, xsAt = 4096, outs = frames + tokens * frame, alone = outs + 4096, xf = alone + 4096, end = xf + 16384;
  if (end > memory.buffer.byteLength) throw new Error("int8-check's memory is too small");
  const ranges = [[0, rows], [0, 1], [0, 3], [rows - 1, rows], [1, 6], [3, 8], [5, 5], [4, 12]];

  for (const ng of [1, 2, 3, 4, 5, 6, 7, 8, 9, 80, most]) {
    const n = ng * 32;
    // the weights: int8 (-128 too) and int6 (4 v, v in -32..31), rows 0 and 1 at the ends, row 2 alternating
    const w8 = Int8Array.from({ length: rows * n }, () => (next() & 255) - 128);
    const v6 = Int8Array.from({ length: rows * n }, () => 4 * ((next() & 63) - 32));
    w8.fill(127, 0, n); w8.fill(-128, n, 2 * n);
    v6.fill(124, 0, n); v6.fill(-128, n, 2 * n);
    for (let j = 0; j < n; j++) { w8[2 * n + j] = j & 1 ? 127 : -128; v6[2 * n + j] = j & 1 ? 124 : -128; }
    I.set(w8, w);
    for (let g = 0; g < rows * ng; g++) pack6(v6.subarray(g * 32, g * 32 + 32), U, w6 + g * 24);
    const wsv = Float32Array.from({ length: rows * ng }, (_, i) => scale(i));
    F.set(wsv, ws / 4);

    // ---- int8_sums and six_sums
    N.fill(-7, wc / 4, wc / 4 + rows * ng + 4);
    N.fill(-7, c6 / 4, c6 / 4 + rows * ng + 4);
    k.int8_sums(wc, w, rows * ng);
    k.six_sums(c6, w6, rows * ng);
    for (let g = 0; g < rows * ng; g++) {
      let s8 = 0, s6 = 0;
      for (let j = g * 32; j < g * 32 + 32; j++) { s8 += w8[j]; s6 += v6[j]; }
      if (N[wc / 4 + g] !== -64 * s8) throw new Error(`int8_sums${where}: group ${g} of ${ng} a row is ${N[wc / 4 + g]}, not ${-64 * s8}`);
      if (N[c6 / 4 + g] !== -64 * s6) throw new Error(`six_sums${where}: group ${g} of ${ng} a row is ${N[c6 / 4 + g]}, not ${-64 * s6}`);
    }
    if (N[wc / 4 + rows * ng] !== -7 || N[c6 / 4 + rows * ng] !== -7) throw new Error(`int8_sums or six_sums${where} wrote past ${rows * ng} groups`);
    checked += 2;

    // ---- the relaxed ones: activations of 0..127 (token 0 all 127, token 1 all 0, token 2 alternating, token 3 from
    // quantize_x(bias = 64) of random floats, the others random), their scales a frame apart
    const acts = [], scales = [];
    for (let t = 0; t < tokens; t++) {
      const at = frames + t * frame;
      if (t === 3) {
        for (let j = 0; j < n; j++) F[xf / 4 + j] = f(((next() % 2001) - 1000) * 1e-3 * (1 + (j % 5)));
        k.quantize_x(at, at + xsAt, xf, n, 64);
      } else {
        for (let j = 0; j < n; j++) U[at + j] = t === 0 ? 127 : t === 1 ? 0 : t === 2 ? (j & 1 ? 127 : 0) : next() & 127;
        for (let g = 0; g < ng; g++) F[(at + xsAt) / 4 + g] = scale(g + t);
      }
      acts.push(U.slice(at, at + n));
      scales.push(F.slice((at + xsAt) / 4, (at + xsAt) / 4 + ng));
    }
    const expected = (model, weights, t) => Array.from({ length: rows }, (_, i) => model(weights.subarray(i * n, i * n + n), acts[t], wsv.subarray(i * ng, i * ng + ng), scales[t], ng));
    for (const [name, kernel, weights, wq, corrections] of [["matmul_q8r", r.matmul_q8r, w8, w, wc], ["matmul_q6r", r.matmul_q6r, v6, w6, c6]]) {
      for (let t = 0; t < tokens; t++) {
        const want = expected(q8r, weights, t), at = frames + t * frame;
        for (const [r0, r1] of ranges) {
          F.fill(-7, alone / 4, alone / 4 + rows + 1);
          kernel(alone, at, at + xsAt, wq, ws, corrections, n, r0, r1);
          for (let i = 0; i <= rows; i++) {
            const value = i >= r0 && i < r1 ? want[i] : -7;
            if (!Object.is(F[alone / 4 + i], value)) throw new Error(`${name}${where} differs at row ${i} of rows ${r0}..${r1}, token ${t}, ${ng} groups: ${F[alone / 4 + i]} against ${value}`);
          }
          checked += 1;
        }
      }
    }
    // ---- matmul_q8r_tile: each (row, token) matmul_q8r's number to the bit, and nothing written outside the rows and tokens
    for (let count = 1; count <= tokens; count++) {
      for (const [r0, r1] of [[0, rows], [0, 3], [1, 6], [2, rows], [4, 12], [5, 5]]) {
        F.fill(-7, outs / 4, (outs + tokens * outFrame) / 4);
        r.matmul_q8r_tile(outs, frames, frames + xsAt, w, ws, wc, n, r0, r1, count, outFrame, frame);
        for (let t = 0; t < tokens; t++) {
          F.fill(-7, alone / 4, alone / 4 + outFrame / 4);
          if (t < count) r.matmul_q8r(alone, frames + t * frame, frames + t * frame + xsAt, w, ws, wc, n, r0, r1);
          for (let i = 0; i < outFrame / 4; i++) {
            if (!Object.is(F[(outs + t * outFrame) / 4 + i], F[alone / 4 + i])) {
              throw new Error(`matmul_q8r_tile${where} differs from matmul_q8r at row ${i}, token ${t} of ${count}, rows ${r0}..${r1}, ${ng} groups: ${F[(outs + t * outFrame) / 4 + i]} against ${F[alone / 4 + i]}`);
            }
          }
        }
        checked += 1;
      }
    }

    // ---- the plain ones: activations of -127..127 (token 0 all 127, token 1 all -127, token 2 alternating, token 3 from
    // quantize_x(bias = 0), the others random)
    for (let t = 0; t < tokens; t++) {
      const at = frames + t * frame;
      if (t === 3) {
        k.quantize_x(at, at + xsAt, xf, n, 0);
      } else {
        for (let j = 0; j < n; j++) I[at + j] = t === 0 ? 127 : t === 1 ? -127 : t === 2 ? (j & 1 ? 127 : -127) : (next() % 255) - 127;
      }
      acts[t] = I.slice(at, at + n);
      scales[t] = F.slice((at + xsAt) / 4, (at + xsAt) / 4 + ng);
    }
    for (const [name, kernel, weights, wq] of [["matmul_q8", k.matmul_q8, w8, w], ["matmul_q6", k.matmul_q6, v6, w6]]) {
      for (let t = 0; t < tokens; t++) {
        const want = expected(q8, weights, t), at = frames + t * frame;
        for (const [r0, r1] of ranges) {
          F.fill(-7, alone / 4, alone / 4 + rows + 1);
          kernel(alone, at, at + xsAt, wq, ws, n, r0, r1);
          for (let i = 0; i <= rows; i++) {
            const value = i >= r0 && i < r1 ? want[i] : -7;
            if (!Object.is(F[alone / 4 + i], value)) throw new Error(`${name}${where} differs at row ${i} of rows ${r0}..${r1}, token ${t}, ${ng} groups: ${F[alone / 4 + i]} against ${value}`);
          }
          checked += 1;
        }
      }
    }
  }

  // ---- quantize_x at both biases: groups of random values, of zeros, of -0, at the ends (the largest first and last,
  // a value of half a step), the scales of every size
  {
    const groups = 12, n = groups * 32, xq = alone, xs = xq + 1024;
    for (let g = 0; g < groups; g++) {
      const size = [1, 3.0000002, 1e-30, 65504, 1.5e-5, 2 ** -140][g % 6];  // (2^-140: 1 / scale overflows)
      for (let j = 0; j < 32; j++) F[xf / 4 + g * 32 + j] = g === 4 ? 0 : g === 5 ? -0 : f(((next() % 2001) - 1000) * 1e-3 * size);
    }
    F[xf / 4 + 6 * 32] = f(-3); F[xf / 4 + 6 * 32 + 31] = f(3);  // group 6: the ends at both places
    for (let j = 1; j < 31; j++) F[xf / 4 + 7 * 32 + j] = f((j - 15) * 0.5);  // group 7: values a half apart
    F[xf / 4 + 7 * 32] = f(63); F[xf / 4 + 7 * 32 + 31] = f(-63);
    for (const bias of [0, 64]) {
      const qmax = bias === 0 ? 127 : 63;
      U.fill(0xAA, xq, xq + n + 16);
      k.quantize_x(xq, xs, xf, n, bias);
      for (let g = 0; g < groups; g++) {
        let largest = 0;
        for (let j = 0; j < 32; j++) largest = Math.max(largest, Math.abs(F[xf / 4 + g * 32 + j]));
        const s = f(largest / qmax), inv = s > 0 ? f(1 / s) : 0;
        if (!Object.is(F[xs / 4 + g], s)) throw new Error(`quantize_x${where}, bias ${bias}: the scale of group ${g} is ${F[xs / 4 + g]}, not ${s}`);
        for (let j = 0; j < 32; j++) {
          // trunc_sat (a NaN, 0 times an infinite 1 / scale, is 0), the bias added in int32 (it wraps), narrowed with saturation
          const v = nearest(f(F[xf / 4 + g * 32 + j] * inv)), truncated = Number.isNaN(v) ? 0 : Math.max(-(2 ** 31), Math.min(2 ** 31 - 1, v));
          const want = Math.max(-128, Math.min(127, (truncated + bias) | 0)) & (bias ? 255 : -1);
          const got = bias ? U[xq + g * 32 + j] : I[xq + g * 32 + j];
          if (got !== want) throw new Error(`quantize_x${where}, bias ${bias}: value ${j} of group ${g} is ${got}, not ${want}`);
          // (where 1 / scale overflows, a group of subnormals, the values reach -128 and 0: kernel/matmul.ts's dot32 says why
          // its scale makes that product 0)
          if (Number.isFinite(inv) && (bias ? got < 1 || got > 127 : got < -127)) throw new Error(`quantize_x${where}, bias ${bias}: value ${j} of group ${g} is ${got}, past the ends the kernels take`);
        }
      }
      if (U[xq + n] !== 0xAA) throw new Error(`quantize_x${where} wrote past ${n} values`);
      checked += 1;
    }
  }
}
console.log(`int8-check: matmul_q8r, matmul_q6r, matmul_q8 and matmul_q6 are their arithmetic to the bit, matmul_q8r_tile matmul_q8r's numbers, int8_sums, six_sums and quantize_x their values (${checked} cases, on a 32-bit and a 64-bit memory)`);
