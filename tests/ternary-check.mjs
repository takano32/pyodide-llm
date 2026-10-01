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
//   ternary_x: the bytes and scales of llama2_numpy.ternary() (the sign + 1 of value j at bits 2 (j & 3) of byte
//     j >> 2, the largest |value| of a group), and its refusal of a value that is neither 0 nor of that size.
import fs from "node:fs";
import { addressed } from "../public/jobs.js";

const root = new URL("../", import.meta.url).pathname;
const f = Math.fround;
let checked = 0;

for (const wide of [false, true]) {
  const memory = wide ? new WebAssembly.Memory({ initial: 64n, address: "i64" }) : new WebAssembly.Memory({ initial: 64 });
  const load = (name) => addressed(new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/${name}${wide ? "64" : ""}.wasm`)),
    { env: { memory } }).exports, wide);
  const k = load("simdkernel_plain"), r = load("simdkernel_relaxed_plain");
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
  let seed = 231;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const where = wide ? " (64-bit memory)" : "";

  // ---- interleave
  for (const n of [128, 256, 384, 640, 1152]) {
    const xq = 4096, xs = xq + 2048, ng = n / 32;
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
  const w = 65536, ws = w + rows * most * 32, xq = ws + rows * most * 4, xs = xq + most * 128, out = xs + most * 4 * 8, end = out + rows * 4;
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
  }

  // ---- ternary_x
  {
    const groups = 40, x = 65536, packed = x + groups * 512, scales = packed + groups * 32;
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
    for (const [at, spoil] of [[0, 0.5], [127, 1.0000001], [128 + 63, 2], [128 * 39 + 127, NaN], [128 * 7 + 5, -0.999999]]) {
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
