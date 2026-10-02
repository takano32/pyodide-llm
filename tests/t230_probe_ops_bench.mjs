// T230's review (a probe, for CI): what the five formulations of a block of interleave() cost in V8 (ns a block of 64 bytes),
// taking turns in one process. node tests/t230_probe_ops_bench.mjs
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";

const root = new URL("../", import.meta.url).pathname;
fs.mkdirSync(`${root}.tmp`, { recursive: true });
execFileSync("npx", ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1", "--enable", "simd",
  `${root}tests/t230_probe_ops.ts`, "-o", `${root}.tmp/probe-ops.wasm`], { cwd: root, stdio: "inherit" });
const memory = new WebAssembly.Memory({ initial: 1 });
const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}.tmp/probe-ops.wasm`)), { env: { memory } }).exports;
const U = new Uint8Array(memory.buffer);
for (let i = 0; i < 64; i++) U[1024 + i] = (i * 37 + 11) & 255;
const names = ["a", "b", "c", "d", "e"];
const calls = 2e7, results = Object.fromEntries(names.map((name) => [name, []]));
for (const name of names) k["loop_" + name](1024, 2048, 4e6);  // tier up
for (let round = 0; round < 7; round++) {
  for (const name of names) {
    const t0 = performance.now();
    k["loop_" + name](1024, 2048, calls);
    results[name].push(((performance.now() - t0) * 1e6) / calls);
  }
}
const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
console.log(`ops-bench: ${os.cpus()[0].model} (${process.arch}), ns a block of 64 bytes (median of 7): ` +
  names.map((name) => `${name} ${median(results[name]).toFixed(2)}`).join(" | "));
