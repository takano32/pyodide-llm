// ternary-check.mjs (T230, T231): the kernels of the ternary weights (kernels/ternary.ts; kernel.ts's interleave,
// matmul_t2 and ternary_x; kernel_relaxed.ts's matmul_t2r) against the same arithmetic written out here, value by
// value. Node and the compiled kernels alone, under a second: the light suite runs it. On a 32-bit memory and on a
// 64-bit one (the addresses through jobs.js's table).
//   interleave: the bytes of every block of 64 in their planes (byte 16 p + c is activation 4 c + p), and minus the
//     sum of every group of 32 after the scales, for rows of 128 to 1152 and activations at int8's ends.
//   matmul_t2r and matmul_t2: to the bit, and so the same as each other. A group of 128's four integer sums are
//     exact (the weights -1, 0, 1 and PQ2_0's fourth code, 2; the activations every int8, -128 too), each is
//     multiplied by its activations' scale and then by the weights' in float32, lane k of the accumulator adds the
//     k-th of every group, and the four lanes are added in order: Math.fround follows that here. Rows of 1 to 9 groups
//     (the weights of a row are 32 bytes a group, so a kernel that read a row at the wrong stride or took a group's
//     scale from its neighbour fails at more than one group), ranges of rows that begin and end anywhere, and groups
//     of all +1 against activations of 127 and of -128 (the largest sums: 2 x 128 a product before the sums are taken).
//   the largest sums (T230's review): every row of one code against activations all at one end of int8 or alternating
//     between the two, for every code (0, 1, 2 and PQ2_0's fourth, 3), the most a lane or a group can reach (16 products of
//     128 x 3 a lane, 32 of them a group, and minus the group's sum): the relaxed dot product's pairs (x86's pmaddubsw adds
//     two products with saturation to int16: 2 x 128 x 3 = 768, nowhere near 32767), the sums in int32, and float32 hold
//     all of it exactly, and matmul_t2r, matmul_t2 and the tile are the same to the bit.
//   ternary_x: the bytes and scales of llama2_numpy.ternary() (the sign + 1 of value j at bits 2 (j & 3) of byte
//     j >> 2, the largest |value| of a group), and its refusal of a value that is neither 0 nor of that size.
import fs from "node:fs";
import v8 from "node:v8";
import { built, runtimeUrl } from "./tree.mjs";
const { addressed } = await import(runtimeUrl("jobs.js"));

const root = new URL("../", import.meta.url).pathname;
const f = Math.fround;
let checked = 0;

// where the weights of a model past 4 GiB lie: the kernels take a 64-bit address, and every product of a row, a scale and an
// activation of this check is read from above the first 4 GiB (the memory is made, not touched: a few pages of it are)
const HIGH = 4 * 2 ** 30 + 2 * 65536;
// A canary for the engine, not the kernels: the V8 of Node 24 (13.6) on arm64 reads v128.load32_splat (and v128.load32_lane) of
// an address above 4 GiB at its low 32 bits, so that every kernel that takes a scale so (the ternary ones' weights' scales, the
// int8 tile's) reads the wrong number there (Chromium 148's V8 on arm64 reads them right: T230's review). A module of its own
// (written out here: a 64-bit memory it imports, splat(address) = v128.load32_splat of it, lane 0) says whether this engine
// does. T233's review found that it is V8's Liftoff alone (arm64; fixed in V8 14.3, Chrome 143: ff9dbb26c2), whose code a
// function leaves for TurboFan's within milliseconds of its first call, and whose first call the canary is: so where it
// fails, Liftoff is turned off for what is compiled from here (the kernels below) and the canary asked again, and the
// kernels above 4 GiB are checked as TurboFan has them. Where it still fails, the config is left out, saying why.
// (The canary asked again, and the kernels loaded after it, are the same modules with a custom section the engine
// ignores: V8 keeps the compiled code of a module by its bytes, and the same bytes would hand back the functions Liftoff
// compiled for the first question and for the 64-bit kernels of the config before, which read the splats wrongly.)
const CANARY = Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 126, 1, 125, 2, 15, 1, 3, 101, 110, 118, 6, 109, 101, 109, 111, 114, 121, 2, 4, 1,
  3, 2, 1, 0, 7, 9, 1, 5, 115, 112, 108, 97, 116, 0, 0, 10, 13, 1, 11, 0, 32, 0, 253, 9, 2, 0, 253, 31, 0, 11]);
function splatsRight(memory, again = false) {
  const bytes = again ? Uint8Array.from([...CANARY, 0, 2, 1, 97]) : CANARY;
  const splat = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } }).exports.splat;
  const F = new Float32Array(memory.buffer), at = HIGH + 4096;
  F[at / 4] = 1.5;
  F[(at - 2 ** 32) / 4] = 2.5;  // where an address that lost its upper 32 bits would read
  return splat(BigInt(at)) === 1.5;
}
for (const [wide, base] of [[false, 0], [true, 0], [true, HIGH]]) {
  let memory, liftoffOff = false;
  try {
    memory = wide ? new WebAssembly.Memory({ initial: BigInt(Math.ceil((base + 4 * 2 ** 20) / 65536)), address: "i64" }) : new WebAssembly.Memory({ initial: 64 });
  } catch (error) {
    console.log(`ternary-check: no 64-bit memory of ${(base / 2 ** 30).toFixed(1)} GiB and more in this Node: skipped (${error.message})`);
    continue;
  }
  if (base && !splatsRight(memory)) {
    v8.setFlagsFromString("--no-liftoff");
    liftoffOff = true;
    if (!splatsRight(memory, true)) {
      console.log("ternary-check: this engine reads v128.load32_splat above 4 GiB at the low 32 bits of the address even with --no-liftoff: the kernels above 4 GiB cannot be checked here, skipped");
      continue;
    }
    console.log("ternary-check: Liftoff reads v128.load32_splat above 4 GiB at the low 32 bits of the address (V8 13.6 on arm64, Node 24; fixed in V8 14.3): --no-liftoff is set, the kernels above 4 GiB are TurboFan's here");
  }
  const bytesOf = (name) => {
    const bytes = fs.readFileSync(built(`${name}${wide ? "64" : ""}.wasm`));
    return liftoffOff ? Uint8Array.from([...bytes, 0, 2, 1, 98]) : bytes;
  };
  const load = (name) => addressed(new WebAssembly.Instance(new WebAssembly.Module(bytesOf(name)), { env: { memory } }).exports, wide);
  const k = load("simdkernel_plain"), r = load("simdkernel_relaxed_plain");
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
  let seed = 231;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const where = wide ? (base ? " (64-bit memory, above 4 GiB)" : " (64-bit memory)") : "";

  // ---- interleave
  for (const n of [128, 256, 384, 640, 1152]) {
    const xq = base + 4096, xs = xq + 2048, ng = n / 32;
    const a = Int8Array.from({ length: n }, () => (next() & 255) - 128);
    a.fill(127, 0, 32);
    a.fill(-128, 32, 64);
    I.set(a, xq);
    I.fill(55, xq + n, xq + n + 64);  // what follows the row is left alone
    for (let g = 0; g < ng; g++) F[xs / 4 + g] = g + 0.5;
    N.fill(-7, xs / 4 + ng, xs / 4 + 2 * ng + 2);
    k.interleave(xq, xs, n);
    for (let at = 0; at < n; at++) {
      const want = a[(at & ~63) + 4 * (at & 15) + ((at >> 4) & 3)];
      if (I[xq + at] !== want) throw new Error(`interleave${where}: byte ${at} of ${n} is ${I[xq + at]}, not ${want}`);
    }
    for (let g = 0; g < ng; g++) {
      const want = -a.subarray(g * 32, g * 32 + 32).reduce((sum, value) => sum + value, 0);
      if (N[xs / 4 + ng + g] !== want) throw new Error(`interleave${where}: the sum of group ${g} of ${n} is ${N[xs / 4 + ng + g]}, not ${want}`);
      if (F[xs / 4 + g] !== g + 0.5) throw new Error(`interleave${where} wrote over the scale of group ${g}`);
    }
    if (I[xq + n] !== 55 || N[xs / 4 + 2 * ng] !== -7) throw new Error(`interleave${where} wrote past a row of ${n}`);
    checked += 1;
  }

  // ---- matmul_t2r and matmul_t2
  const rows = 13, most = 9;  // groups of 128 a row, at most
  const w = base + 65536, ws = w + rows * most * 32, xq = ws + rows * most * 4, xs = xq + most * 128, out = xs + most * 4 * 8, end = out + rows * 4;
  if (end > memory.buffer.byteLength) throw new Error("ternary-check's memory is too small");
  for (const groups of [1, 2, 3, 4, 5, 7, most]) {
    const n = groups * 128, ng = n / 32;
    // the codes: every one of the four, PQ2_0's fourth too (a weight of 2), at every place of a byte
    const codes = Uint8Array.from({ length: rows * n }, () => next() & 3);
    codes.fill(2, 0, 128);  // row 0's first group: all +1
    if (groups > 1) codes.fill(0, n + 128, n + 256);  // row 1's second: all -1
    U.fill(0, w, w + rows * groups * 32);
    codes.forEach((code, j) => { U[w + (j >> 2)] |= code << (2 * (j & 3)); });
    for (let g = 0; g < rows * groups; g++) F[ws / 4 + g] = f(1e-3 * (1 + (next() % 1000)));
    const a = Int8Array.from({ length: n }, () => (next() & 255) - 128);
    a.fill(127, 0, 64);
    a.fill(-128, 64, 128);
    if (groups > 1) a.fill(-128, 128, 256);
    I.set(a, xq);
    for (let g = 0; g < ng; g++) F[xs / 4 + g] = f(1e-2 * (1 + (next() % 1000)));
    k.interleave(xq, xs, n);
    const expected = Array.from({ length: rows }, (_, i) => {
      const lanes = [0, 0, 0, 0];
      for (let g = 0; g < groups; g++) {
        for (let lane = 0; lane < 4; lane++) {
          let sum = 0;
          for (let j = g * 128 + lane * 32; j < g * 128 + lane * 32 + 32; j++) sum += (codes[i * n + j] - 1) * a[j];
          lanes[lane] = f(lanes[lane] + f(f(sum * F[xs / 4 + g * 4 + lane]) * F[ws / 4 + i * groups + g]));
        }
      }
      return f(f(f(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
    });
    for (const [name, kernel] of [["matmul_t2r", r.matmul_t2r], ["matmul_t2", k.matmul_t2]]) {
      for (const [r0, r1] of [[0, rows], [0, 1], [rows - 1, rows], [3, 8], [5, 5]]) {
        F.fill(-7, out / 4, out / 4 + rows);
        kernel(out, xq, xs, w, ws, n, r0, r1, 3);
        for (let i = 0; i < rows; i++) {
          const want = i >= r0 && i < r1 ? expected[i] : -7;
          if (!Object.is(F[out / 4 + i], want)) {
            throw new Error(`${name}${where} differs at row ${i} of rows ${r0}..${r1} of ${groups} groups: ${F[out / 4 + i]} against ${want}`);
          }
        }
        checked += 1;
      }
    }
    // ---- matmul_t2r_tile: a prompt's tokens, a frame apart, each (row, token) matmul_t2r's number to the bit, for 1 to
    // 9 tokens (the fours and what is left after them) and rows that begin and end anywhere
    {
      const frame = 2048, outFrame = 64, tokens = 9;
      const frames = end + 4096, outs = frames + tokens * frame, alone = outs + tokens * outFrame;
      if (alone + rows * 4 > memory.buffer.byteLength) throw new Error("ternary-check's memory is too small for the tile");
      for (let t = 0; t < tokens; t++) {
        const at = frames + t * frame;  // a token's activations, and after them (at most * 128) its scales and sums
        for (let j = 0; j < n; j++) I[at + j] = (next() & 255) - 128;
        for (let g = 0; g < ng; g++) F[(at + most * 128) / 4 + g] = f(1e-2 * (1 + (next() % 1000)));
        k.interleave(at, at + most * 128, n);
      }
      for (const count of [1, 2, 3, 4, 5, 7, 8, tokens]) {
        for (const [r0, r1] of [[0, rows], [2, 9], [rows - 1, rows]]) {
          F.fill(-7, outs / 4, (outs + tokens * outFrame) / 4);
          r.matmul_t2r_tile(outs, frames, frames + most * 128, w, ws, n, r0, r1, count, outFrame, frame, 3);
          for (let t = 0; t < tokens; t++) {
            F.fill(-7, alone / 4, alone / 4 + rows);
            if (t < count) r.matmul_t2r(alone, frames + t * frame, frames + t * frame + most * 128, w, ws, n, r0, r1, 3);
            for (let i = 0; i < rows; i++) {
              if (!Object.is(F[(outs + t * outFrame) / 4 + i], F[alone / 4 + i])) {
                throw new Error(`matmul_t2r_tile${where} differs from matmul_t2r at row ${i}, token ${t} of ${count}, rows ${r0}..${r1}, ${groups} groups`);
              }
            }
          }
          checked += 1;
        }
      }
    }
  }

  // ---- the largest sums (T230's review): the rows of one code each (0, 1, 2, and 3, which a PQ2_0 file does not have but
  // the mask lets through) against activations at one end of int8, at the other and alternating, so that every product of
  // every lane is the largest it can be (|a| x code: 128 x 3 = 384, 16 a lane, 32 a group of 32 activations, and minus
  // their sum, 4096): the relaxed dot product's pairs (pmaddubsw on x86 adds two products with saturation to int16: 768 of
  // 32767) and sums (int32) take all of it, and so does float32. The three kernels give the exact sums' number to the bit.
  {
    const groups = most, n = groups * 128, ng = n / 32, count = 5, frame = 2048, outFrame = 64;
    const patterns = [(j) => -128, (j) => 127, (j) => (j & 1 ? 127 : -128), (j) => (j & 32 ? -128 : 127), (j) => 0, (j) => -127];
    U.fill(0, w, w + rows * groups * 32);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < n; j++) U[w + (i * n + j >> 2)] |= (i & 3) << (2 * (j & 3));  // row i: every weight the code i & 3
    }
    for (let g = 0; g < rows * groups; g++) F[ws / 4 + g] = f(0.5 + 0.01 * g);
    const frames = end + 4096, outs = frames + count * frame, alone = outs + count * outFrame;
    if (alone + rows * 4 > memory.buffer.byteLength) throw new Error("ternary-check's memory is too small for the largest sums");
    for (const [index, pattern] of patterns.entries()) {
      for (let t = 0; t < count; t++) {
        const at = frames + t * frame;  // five tokens: pattern, its negative where it is not 0, and the pattern again shifted
        for (let j = 0; j < n; j++) I[at + j] = Math.max(-128, Math.min(127, t % 2 ? -pattern(j) : pattern(j + t)));
        for (let g = 0; g < ng; g++) F[(at + most * 128) / 4 + g] = f(1e-2 * (1 + ((g + t) % 7)));
        k.interleave(at, at + most * 128, n);
      }
      for (let t = 0; t < count; t++) {
        const at = frames + t * frame;
        // the activations as they were before interleave(), from the same rule
        const a = Int8Array.from({ length: n }, (_, j) => Math.max(-128, Math.min(127, t % 2 ? -pattern(j) : pattern(j + t))));
        const expected = Array.from({ length: rows }, (_, i) => {
          const lanes = [0, 0, 0, 0];
          for (let g = 0; g < groups; g++) {
            for (let lane = 0; lane < 4; lane++) {
              let sum = 0;
              for (let j = g * 128 + lane * 32; j < g * 128 + lane * 32 + 32; j++) sum += ((i & 3) - 1) * a[j];
              lanes[lane] = f(lanes[lane] + f(f(sum * F[(at + most * 128) / 4 + g * 4 + lane]) * F[ws / 4 + i * groups + g]));
            }
          }
          return f(f(f(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
        });
        for (const [name, kernel] of [["matmul_t2r", r.matmul_t2r], ["matmul_t2", k.matmul_t2]]) {
          F.fill(-7, out / 4, out / 4 + rows);
          kernel(out, at, at + most * 128, w, ws, n, 0, rows, 3);
          for (let i = 0; i < rows; i++) {
            if (!Object.is(F[out / 4 + i], expected[i])) {
              throw new Error(`${name}${where}: the largest sums, pattern ${index}, token ${t}, row ${i} (code ${i & 3}): ${F[out / 4 + i]} against ${expected[i]}`);
            }
          }
          checked += 1;
        }
        // the tile takes the five tokens whole (a four and the one left after it)
        if (t === count - 1) {
          F.fill(-7, outs / 4, (outs + count * outFrame) / 4);
          r.matmul_t2r_tile(outs, frames, frames + most * 128, w, ws, n, 0, rows, count, outFrame, frame, 3);
          for (let u = 0; u < count; u++) {
            F.fill(-7, alone / 4, alone / 4 + rows);
            r.matmul_t2r(alone, frames + u * frame, frames + u * frame + most * 128, w, ws, n, 0, rows, 3);
            for (let i = 0; i < rows; i++) {
              if (!Object.is(F[(outs + u * outFrame) / 4 + i], F[alone / 4 + i])) {
                throw new Error(`matmul_t2r_tile${where}: the largest sums, pattern ${index}, token ${u}, row ${i}: ${F[(outs + u * outFrame) / 4 + i]} against ${F[alone / 4 + i]}`);
              }
            }
          }
          checked += 1;
        }
      }
    }
  }

  // ---- ternary_x
  {
    const groups = 40, x = base + 65536, packed = x + groups * 512, scales = packed + groups * 32;
    const make = () => {
      for (let g = 0; g < groups; g++) {
        // a scale of every size, a group of zeros, one of -0 alone
        const d = g === 3 ? 0 : f([1, 0.0078125, 3.0000002, 1e-30, 65504, 1.5e-5][g % 6] * (1 + (next() % 7)));
        for (let j = 0; j < 128; j++) F[x / 4 + g * 128 + j] = g === 5 ? -0 : ((next() % 3) - 1) * d;
      }
    };
    make();
    U.fill(0xAA, packed, packed + groups * 32);
    if (k.ternary_x(packed, scales, x, groups * 128) !== 0) throw new Error(`ternary_x${where} refuses ternary values`);
    for (let g = 0; g < groups; g++) {
      let largest = 0;
      for (let j = 0; j < 128; j++) largest = Math.max(largest, Math.abs(F[x / 4 + g * 128 + j]));
      if (F[scales / 4 + g] !== largest) throw new Error(`ternary_x${where}: the scale of group ${g} is ${F[scales / 4 + g]}, not ${largest}`);
      for (let b = 0; b < 32; b++) {
        let want = 0;
        for (let p = 0; p < 4; p++) want |= (Math.sign(F[x / 4 + g * 128 + 4 * b + p]) + 1) << (2 * p);
        if (U[packed + g * 32 + b] !== want) throw new Error(`ternary_x${where}: byte ${b} of group ${g} is ${U[packed + g * 32 + b]}, not ${want}`);
      }
    }
    // one value of any one place that is not ternary: half the scale, a little more than it, twice it, not a number
    // (and half the scale in each of the four vectors of sixteen values, which a look at three of them would miss)
    for (const [at, spoil] of [[0, 0.5], [127, 1.0000001], [128 + 63, 2], [128 * 39 + 127, NaN], [128 * 7 + 5, -0.999999],
      [128 * 9 + 32 + 1, 0.5], [128 * 9 + 32 + 6, 0.5], [128 * 9 + 32 + 11, 0.5], [128 * 9 + 32 + 14, 0.5], [128 * 39 + 127, 0.5]]) {
      make();
      const g = at >> 7;
      for (let j = 0; j < 128; j++) F[x / 4 + g * 128 + j] = j % 2 ? 3 : -3;
      F[x / 4 + at] = spoil * 3;
      if (k.ternary_x(packed, scales, x, groups * 128) !== 1) throw new Error(`ternary_x${where} takes a value of ${spoil} times its group's scale at ${at}`);
    }
    checked += 6;
  }
}
console.log(`ternary-check: interleave, matmul_t2r, matmul_t2 and ternary_x are their arithmetic to the bit (${checked} cases, on a 32-bit and a 64-bit memory)`);
