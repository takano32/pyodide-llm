// probe-kernel.mjs (T233 review): the real ternary kernel (matmul_t2r, matmul_t2r_tile) on a 64-bit memory, its scales and
// weights above 4 GiB, from its very first call: right, or wrong, and from which call (the tier-up of V8's Liftoff code
// to TurboFan's) on.
//   node [--liftoff-only | --no-liftoff] .tmp/probe-kernel.mjs [shared] [rows=3072] [calls=40] [kernel=matmul_t2r]
import fs from "node:fs";

const root = new URL("../", import.meta.url).pathname;
const GiB = 2 ** 30, args = process.argv.slice(2);
const shared = args.includes("shared");
const named = (key, value) => Number((args.find((a) => a.startsWith(`${key}=`)) ?? `${key}=${value}`).split("=")[1]);
const rows = named("rows", 3072), calls = named("calls", 40), n = 5120, groups = n >> 7;
const tile = args.includes("tile");
const pages = Math.ceil((4 * GiB + 256 * 2 ** 20) / 65536);
const memory = new WebAssembly.Memory(shared ? { initial: BigInt(pages), maximum: BigInt(pages + 64), shared: true, address: "i64" } : { initial: BigInt(pages), address: "i64" });
const file = shared ? "simdkernel_relaxed_shared64.wasm" : "simdkernel_relaxed_plain64.wasm";
const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/${file}`)), { env: { memory } }).exports;
const U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), I32 = new Int32Array(memory.buffer);
let seed = 233;
const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;

// activations (low): xq n bytes, xs: n/32 scales then n/32 int sums
const xq = 4096, xs = xq + n + 64, out = xs + (n >> 5) * 8 + 64;
for (let i = 0; i < n; i++) U[xq + i] = next() & 255;
for (let g = 0; g < n >> 5; g++) { F[xs / 4 + g] = 0.01 + (next() % 100) / 1000; I32[xs / 4 + (n >> 5) + g] = (next() % 2001) - 1000; }
// weights and scales, once low, once high (the same bytes)
const weights = Uint8Array.from({ length: rows * (n >> 2) }, () => next() & 255);
const scales = Float32Array.from({ length: rows * groups }, () => 0.5 + (next() % 1000) / 1000);
const placeAt = (at) => { const wq = at, ws = at + weights.length; U.set(weights, wq); F.set(scales, ws / 4); return [wq, ws]; };
const [lowW, lowS] = placeAt(1 * 2 ** 20 + 4096);
const highAt = 4 * GiB + 2 * 65536 + 4096;
const [highW, highS] = placeAt(highAt);
const outHigh = out + rows * 4 + 64;
const call = (wq, ws, to) => (tile
  ? k.matmul_t2r_tile(BigInt(to), BigInt(xq), BigInt(xs), BigInt(wq), BigInt(ws), n, 0, rows, 1, 0, 0, 3)
  : k.matmul_t2r(BigInt(to), BigInt(xq), BigInt(xs), BigInt(wq), BigInt(ws), n, 0, rows, 3));
void tile;
console.log(`${process.arch} V8 ${process.versions.v8} ${process.execArgv.join(" ") || "(default flags)"}; ${shared ? "shared" : "non-shared"} 64-bit memory; ${rows} rows of ${n}, ${calls} calls`);
// the reference: the very kernel at low addresses, cold; the high calls follow (cold high: the high call is the first)
const began = performance.now();
let want;
const coldHigh = args.includes("coldhigh");
let first = null;
if (coldHigh) {
  F.fill(NaN, outHigh / 4, outHigh / 4 + rows);
  call(highW, highS, outHigh);
  first = F.slice(outHigh / 4, outHigh / 4 + rows);
  call(lowW, lowS, out);  // (by now tiered up, or not)
  want = F.slice(out / 4, out / 4 + rows);
  let wrong = 0;
  for (let i = 0; i < rows; i++) if (!Object.is(first[i], want[i])) wrong++;
  console.log(`the first call of the kernel, at an address above 4 GiB: ${wrong ? `WRONG (${wrong} of ${rows} rows)` : "right"} (${(performance.now() - began).toFixed(1)} ms with the low one after it)`);
} else {
  call(lowW, lowS, out);
  want = F.slice(out / 4, out / 4 + rows);
  console.log(`low call (cold): ${(performance.now() - began).toFixed(1)} ms`);
}
const log = [];
let previous = null;
for (let c = 0; c < calls; c++) {
  F.fill(NaN, outHigh / 4, outHigh / 4 + rows);
  const t = performance.now();
  call(highW, highS, outHigh);
  const took = performance.now() - t;
  let wrong = 0;
  for (let i = 0; i < rows; i++) if (!Object.is(F[outHigh / 4 + i], want[i])) wrong++;
  const state = wrong ? `WRONG (${wrong} of ${rows} rows)` : "right";
  if (state !== previous || c === calls - 1) log.push(`call ${c} at ${(t - began).toFixed(0)} ms (took ${took.toFixed(1)} ms): ${state}`);
  previous = state;
}
console.log(log.join("\n"));
