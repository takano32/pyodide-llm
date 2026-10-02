// T230's review (a probe): the pieces of interleave() in a browser's engine, called over and over (JavaScriptCore on x86-64
// miscompiles interleave() after about 2000 calls). Compiles tests/t230_probe_ops.ts and says, for every piece and every
// formulation of the block, the first count of calls after which its result is wrong. node tests/t230_probe_ops.mjs <webkit|chromium|firefox>
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import * as playwright from "playwright-core";

const root = new URL("../", import.meta.url).pathname;
const engine = process.argv[2] ?? "webkit";
fs.mkdirSync(`${root}.tmp`, { recursive: true });
execFileSync("npx", ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1", "--enable", "simd",
  `${root}tests/t230_probe_ops.ts`, "-o", `${root}.tmp/probe-ops.wasm`], { cwd: root, stdio: "inherit" });
const wasm = fs.readFileSync(`${root}.tmp/probe-ops.wasm`).toString("base64");

async function inPage({ wasm }) {
  const bytes = Uint8Array.from(atob(wasm), (c) => c.charCodeAt(0));
  const memory = new WebAssembly.Memory({ initial: 1 });
  const k = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } }).exports;
  const U = new Uint8Array(memory.buffer), I = new Int8Array(memory.buffer), N = new Int32Array(memory.buffer);
  const SRC = 1024, DST = 2048;
  let seed = 99;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const input = Uint8Array.from({ length: 64 }, () => next() & 255);
  const dword = (arr, base, d) => [arr[base + 4 * d], arr[base + 4 * d + 1], arr[base + 4 * d + 2], arr[base + 4 * d + 3]];
  const expectations = {
    op_fourths: () => { const out = new Uint8Array(64); for (let b = 0; b < 64; b += 16) for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[b + 4 * r + c] = input[b + r + 4 * c]; return out; },
    op_unpack32: () => {
      const out = [];
      for (const [x, y] of [[0, 16], [32, 48]]) {
        out.push(...dword(input, x, 0), ...dword(input, y, 0), ...dword(input, x, 1), ...dword(input, y, 1));
        out.push(...dword(input, x, 2), ...dword(input, y, 2), ...dword(input, x, 3), ...dword(input, y, 3));
      }
      return Uint8Array.from(out);
    },
    op_unpack64: () => {
      const q = (base, i) => Array.from(input.subarray(base + 8 * i, base + 8 * i + 8));
      return Uint8Array.from([...q(0, 0), ...q(32, 0), ...q(0, 1), ...q(32, 1), ...q(16, 0), ...q(48, 0), ...q(16, 1), ...q(48, 1)]);
    },
    op_sums: () => {
      const out = new Uint8Array(64), view = new Int32Array(out.buffer);
      const s = (a, b) => -Array.from(new Int8Array(input.buffer, a, b - a)).reduce((x, y) => x + y, 0);
      view[0] = s(0, 32); view[1] = s(32, 64);
      return out;
    },
  };
  for (const name of ["blk_a", "blk_b", "blk_c", "blk_d", "blk_e"]) {
    expectations[name] = () => { const out = new Uint8Array(64); for (let p = 0; p < 4; p++) for (let c = 0; c < 16; c++) out[16 * p + c] = input[4 * c + p]; return out; };
  }
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const sizes = [1, 30, 400, 2000, 5000, 20000, 80000];
  const report = {};
  for (const [name, expect] of Object.entries(expectations)) {
    const want = expect();
    const words = name === "op_sums" ? 8 : 64;
    let calls = 0, bad = null;
    for (const size of sizes) {
      U.set(input, SRC);
      for (; calls < size; calls++) k[name](SRC, DST);
      await pause(250);
      U.fill(0, DST, DST + 64);
      k[name](SRC, DST);
      const got = U.subarray(DST, DST + words), expected = want.subarray(0, words);
      let wrong = 0;
      for (let i = 0; i < words; i++) if (got[i] !== expected[i]) wrong++;
      if (wrong && !bad) bad = `after ${calls} calls ${wrong} of ${words} bytes`;
    }
    report[name] = bad ?? `right through ${calls} calls`;
  }
  return report;
}

const browser = await playwright[engine].launch();
try {
  const page = await browser.newPage();
  await page.goto("about:blank");
  console.log(`ops-probe: ${engine} on ${os.cpus()[0].model} (${process.arch})`);
  console.log(`ops-probe: ${JSON.stringify(await page.evaluate(inPage, { wasm }))}`);
} finally {
  await browser.close();
}
process.exit(0);
