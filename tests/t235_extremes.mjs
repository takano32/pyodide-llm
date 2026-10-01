// t235_extremes.mjs (T235's review, a probe for CI, not for main): the int8 matrix products on what a ternary model gives
// them at the worst: weights that are ±127 everywhere (a ternary matrix's every nonzero value, quantize() of d, 0, -d),
// against activations that make every product 127 × 127 with the same sign, in long rows. matmul_q8 (8-bit activations,
// Safari's path) adds two products in int16 before it widens them, and matmul_q8r (relaxed SIMD's dot of int8 and 7-bit
// values) takes pairs on some CPUs with a saturating add: 2 × 127 × 127 = 32258 is the most either meets (and 2 × 124 × 127
// for the 6 bits' ±124, not tried here). Held to float64 from the very integers the kernels read, to float32's rounding.
//   node tests/t235_extremes.mjs        (after make kernels)
import fs from "node:fs";

const root = new URL("../", import.meta.url).pathname;
const memory = new WebAssembly.Memory({ initial: 256 });  // 16 MiB
const load = (file) => new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/${file}`)), { env: { memory } }).exports;
const plain = load("simdkernel_plain.wasm");
let relaxed;
try { relaxed = load("simdkernel_relaxed_plain.wasm"); } catch (error) { console.log(`no relaxed SIMD here: ${error.message}`); }
const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), W = new Int32Array(memory.buffer);

const GS = 32;
let seed = 7;
const random = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8) / 2 ** 24;
let top = 4096;
const take = (bytes) => { const at = top; top += Math.ceil(bytes / 64) * 64; return at; };

let cases = 0, failed = 0, worst = 0;
for (const n of [4096, 6144, 2048 + 32 * 3]) {  // long rows; the last has three groups past a multiple of four
  const ng = n / GS, rows = 16;
  top = 4096;
  const wq = take(rows * n), ws = take(rows * ng * 4), wc = take(rows * ng * 4), x = take(n * 4), xq = take(n), xs = take(ng * 4), out = take(rows * 4);
  const patterns = {
    "all +127 against +": (r, j) => 127, "all +127 against -": (r, j) => 127, "alternating blocks of 32": (r, j) => (((j >> 5) & 1) ? -127 : 127),
    "random ±127 and 0": () => [-127, 0, 127][Math.floor(random() * 3)],
  };
  for (const [name, weight] of Object.entries(patterns)) {
    for (let r = 0; r < rows; r++) for (let j = 0; j < n; j++) I[wq + r * n + j] = weight(r, j);
    for (let g = 0; g < rows * ng; g++) F[ws / 4 + g] = Math.fround(0.01 * (1 + (g % 5)));
    // the activations: every one the same sign as the weight it meets, as large as the group's largest (so the quantized
    // value is ±127 and every product 127 × 127), or random
    for (let j = 0; j < n; j++) {
      const sign = name === "all +127 against -" ? -1 : name === "alternating blocks of 32" ? (((j >> 5) & 1) ? -1 : 1) : 1;
      F[x / 4 + j] = name.startsWith("random") ? random() * 2 - 1 : sign * 1.0;
    }
    for (const [kernel, bias] of [["matmul_q8", 0], ["matmul_q8r", 64]]) {
      if (kernel === "matmul_q8r" && !relaxed) continue;
      plain.quantize_x(xq, xs, x, n, bias);
      if (bias) plain.int8_sums(wc, wq, rows * ng);
      F.fill(NaN, out / 4, out / 4 + rows);
      const k = bias ? relaxed : plain;
      if (bias) k.matmul_q8r(out, xq, xs, wq, ws, wc, n, 0, rows); else k.matmul_q8(out, xq, xs, wq, ws, n, 0, rows);
      for (let r = 0; r < rows; r++) {
        // float64 from the integers the kernel read: sum over groups of (w · (xq - bias)) × ws × xs
        let want = 0, mag = 0;
        for (let g = 0; g < ng; g++) {
          let dot = 0;
          for (let j = 0; j < GS; j++) dot += I[wq + r * n + g * GS + j] * ((bias ? U[xq + g * GS + j] : I[xq + g * GS + j]) - bias);
          want += dot * F[ws / 4 + r * ng + g] * F[xs / 4 + g];
          mag += Math.abs(dot) * F[ws / 4 + r * ng + g] * F[xs / 4 + g];
        }
        const got = F[out / 4 + r];
        const off = Math.abs(got - want) / Math.max(mag, 1e-30);  // against what the terms add up to, not what they cancel to
        worst = Number.isNaN(off) ? NaN : Math.max(worst, off);
        cases++;
        if (!(off < 2e-6)) {
          if (failed < 8) console.log(`${kernel} n=${n} ${name} row ${r}: ${got} where ${want} is right`);
          failed++;
        }
      }
    }
  }
}
console.log(`${cases} products, worst relative difference ${worst.toExponential(2)}, ${failed} past 2e-6`);
process.exit(failed ? 1 : 0);
