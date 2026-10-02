// T230's review: which address arguments of the kernels a 64-bit memory reads right above 4 GiB, one argument at a time.
// A probe for CI (tests.yml's extra=, x86-64 and arm64): it prints one line for every kernel, naming the arguments that
// lose their upper 32 bits (a result that is not the low-address result).
import fs from "node:fs";
import os from "node:os";
import { addressed } from "../public/jobs.js";

const root = new URL("../", import.meta.url).pathname;
const HIGH = 4 * 2 ** 30 + 2 * 65536;
const memory = new WebAssembly.Memory({ initial: BigInt(Math.ceil((HIGH + 8 * 2 ** 20) / 65536)), address: "i64" });
const load = (name) => addressed(new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/${name}64.wasm`)), { env: { memory } }).exports, true);
const k = load("simdkernel_plain"), r = load("simdkernel_relaxed_plain");
const U = new Uint8Array(memory.buffer), I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
console.log(`high-probe: ${os.cpus()[0].model}, ${process.arch}, node ${process.version}, v8 ${process.versions.v8}`);

const LOW = 65536, UP = HIGH + 65536, SPREAD = [0, 8192, 16384, 24576, 40960, 49152, 57344];
// each case: its pointer arguments' names, a setup(at) that writes the inputs, a call(at), and what the first output should be
const cases = {
  q8r: { args: ["out", "xq", "xs", "w", "ws", "wc"], rows: 1, tokens: 1,
    setup(p) { for (let i = 0; i < 128; i++) I[p.w + i] = 2; for (let g = 0; g < 4; g++) { F[p.ws / 4 + g] = 1; N[p.wc / 4 + g] = -64 * 2 * 32; F[p.xs / 4 + g] = 1; } for (let j = 0; j < 128; j++) I[p.xq + j] = 67; },
    call: (p) => r.matmul_q8r(p.out, p.xq, p.xs, p.w, p.ws, p.wc, 128, 0, 1), want: 768 },
  q8r_tile: { args: ["out", "xq", "xs", "w", "ws", "wc"], rows: 4, tokens: 4,
    setup(p) { for (let i = 0; i < 512; i++) I[p.w + i] = 2; for (let g = 0; g < 16; g++) { F[p.ws / 4 + g] = 1; N[p.wc / 4 + g] = -64 * 2 * 32; }
      for (let t = 0; t < 4; t++) { for (let j = 0; j < 128; j++) I[p.xq + t * 1024 + j] = 67; for (let g = 0; g < 4; g++) F[(p.xs + t * 1024) / 4 + g] = 1; } },
    call: (p) => r.matmul_q8r_tile(p.out, p.xq, p.xs, p.w, p.ws, p.wc, 128, 0, 4, 4, 64, 1024), want: 768 },
  q8: { args: ["out", "xq", "xs", "w", "ws"], rows: 1, tokens: 1,
    setup(p) { for (let i = 0; i < 128; i++) I[p.w + i] = 2; for (let g = 0; g < 4; g++) { F[p.ws / 4 + g] = 1; F[p.xs / 4 + g] = 1; } for (let j = 0; j < 128; j++) I[p.xq + j] = 3; },
    call: (p) => k.matmul_q8(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1), want: 768 },
  t2r: { args: ["out", "xq", "xs", "w", "ws"], rows: 1, tokens: 1,
    setup(p) { U.fill(0, p.w, p.w + 32); for (let j = 0; j < 128; j++) U[p.w + (j >> 2)] |= 2 << (2 * (j & 3)); F[p.ws / 4] = 1;
      for (let j = 0; j < 128; j++) I[p.xq + j] = 3; for (let g = 0; g < 4; g++) F[p.xs / 4 + g] = 1; k.interleave(p.xq, p.xs, 128); },
    call: (p) => r.matmul_t2r(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1, 3), want: 384 },
  t2: { args: ["out", "xq", "xs", "w", "ws"], rows: 1, tokens: 1,
    setup(p) { cases.t2r.setup(p); },
    call: (p) => k.matmul_t2(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1, 3), want: 384 },
  t2r_tile: { args: ["out", "xq", "xs", "w", "ws"], rows: 1, tokens: 4,
    setup(p) { U.fill(0, p.w, p.w + 32); for (let j = 0; j < 128; j++) U[p.w + (j >> 2)] |= 2 << (2 * (j & 3)); F[p.ws / 4] = 1;
      for (let t = 0; t < 4; t++) { for (let j = 0; j < 128; j++) I[p.xq + t * 1024 + j] = 3; for (let g = 0; g < 4; g++) F[(p.xs + t * 1024) / 4 + g] = 1; k.interleave(p.xq + t * 1024, p.xs + t * 1024, 128); } },
    call: (p) => r.matmul_t2r_tile(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1, 4, 64, 1024, 3), want: 384 },
  q6: { args: ["out", "xq", "xs", "w", "ws"], rows: 1, tokens: 1,
    // int6: a group of 32 values of 2 (the int8 4 v with v = ... a multiple of 4 as 8): 24 bytes, packed by hand
    setup(p) { U.fill(0, p.w, p.w + 24 * 4); for (let g = 0; g < 4; g++) { const b = p.w + g * 24; for (let j = 0; j < 16; j++) U[b + j] = 0x22; /* low nibbles of values 2 (v = 2: 8 as int8) */ }
      for (let g = 0; g < 4; g++) { F[p.ws / 4 + g] = 1; F[p.xs / 4 + g] = 1; } for (let j = 0; j < 128; j++) I[p.xq + j] = 3; },
    call: (p) => k.matmul_q6(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1), want: null },
};

for (const [name, c] of Object.entries(cases)) {
  const reference = {};
  const bad = [];
  for (let mask = 0; mask < 2 ** c.args.length; mask++) {
    const p = Object.fromEntries(c.args.map((arg, i) => [arg, (mask >> i & 1 ? UP : LOW) + SPREAD[i]]));
    c.setup(p);
    F.fill(-7, p.out / 4, p.out / 4 + 40);
    c.call(p);
    const got = F[p.out / 4];
    if (c.want === null) { if (mask === 0) reference.value = got; if (got !== reference.value) bad.push(`${c.args.filter((_, i) => mask >> i & 1).join("+")}: ${got}`); continue; }
    if (got !== c.want) bad.push(`${c.args.filter((_, i) => mask >> i & 1).join("+")}: ${got}`);
  }
  // which single arguments are enough to break it
  const single = c.args.filter((arg, i) => bad.some((entry) => entry.split(":")[0] === arg));
  console.log(`high-probe: ${name}: ${bad.length === 0 ? "every argument right above 4 GiB" : `WRONG (${bad.length} of ${2 ** c.args.length} placements); alone: ${single.join(", ") || "only in combination"}`}`);
}
