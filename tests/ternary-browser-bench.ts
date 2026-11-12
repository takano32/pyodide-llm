// tests/ternary-browser-bench.ts (T275): the ternary product without relaxed SIMD (matmul_t2, what Safari runs) against
// the forms of tests/ternary-plain-forms.ts, in a browser's own wasm engine (WebKit's JavaScriptCore above all: Safari
// has no relaxed SIMD) and in the engine of the runtime that runs this file (Node's V8, or bun's JavaScriptCore).
// tests/ternary-bench.ts times in Node alone, on threads; this is one thread, in the page as the browser runs it.
//
// One copy of the weights a call, taken in turn from copies past every cache (--megabytes of them, default 128): a
// model's matrices are read once a token. Every form is called on the same copies, the forms in an order turned back
// every turn, and each turn's G weights/s of a form is set against its base's of the same turn (matmul_t2's): the
// ratio's median over the turns is what is reported (t2c, a copy of the form matmul_t2 had before T275 in the forms'
// module, is the noise of two copies). Each form's outputs are held to its base's to the bit, after it has been called
// often enough to be compiled again. A timing reads --weights of them (default 2e9: some 0.1 to 0.3 s), as WebKit's
// clock in a page that is not cross-origin isolated ticks by the millisecond.
//
// --before <commit>: and the kernels as built at that commit (kernels/ of its tree): t2@before and the binary kernel
// b1@before, the bases of this tree's t2 and b1 (b1 reads the first half of the ternary codes as its bits): what a
// change to kernels/ is worth, each held to its before to the bit and timed against it.
//
//   node tests/ternary-browser-bench.ts [here] [chromium webkit firefox] [--turns 15] [--megabytes 128] [--weights 2e9]
//        [--shapes 2048x2048,6144x2048,2048x6144] [--forms t2c,ch,r16,r16s,rt,dt,w2] [--before origin/main]
// "here" is the runtime that runs the file (node or bun). Exit 1 where a form's numbers are not its base's.
// Compiles with AssemblyScript into .tmp/ternary-browser-bench/ (needs `npm ci`); the browsers are playwright-core's.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { closed, leave } from "./leave.ts";

const root = fileURLToPath(new URL("../", import.meta.url));
const work = root + ".tmp/ternary-browser-bench/";
const args = process.argv.slice(2);
const valued = new Set(["--turns", "--megabytes", "--weights", "--shapes", "--forms", "--before"]);
const text = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
const engines = args.filter((arg, i) => !arg.startsWith("--") && !valued.has(args[i - 1]));
const before = text("--before", "");
const options = {
  turns: Number(text("--turns", 15)),
  megabytes: Number(text("--megabytes", 128)),
  weights: Number(text("--weights", 2e9)),
  shapes: text("--shapes", "2048x2048,6144x2048,2048x6144").split(",").map((shape) => shape.split("x").map(Number)),
  forms: text("--forms", "t2c,ch,r16,r16s,rt,dt,w2").split(",").filter(Boolean),
};

fs.mkdirSync(work, { recursive: true });
const asc = ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1", "--enable", "simd"];
const sources = [[root + "kernels/kernel.ts", "plain"], [root + "tests/ternary-plain-forms.ts", "forms"]];
if (before) {
  // kernels/ as it was at that commit, in a folder of its own
  fs.rmSync(work + "before", { recursive: true, force: true });
  fs.mkdirSync(work + "before");
  const archive = execFileSync("git", ["archive", before, "kernels"], { cwd: root, maxBuffer: 1 << 28 });
  execFileSync("tar", ["-x", "-C", work + "before"], { input: archive });
  sources.push([work + "before/kernels/kernel.ts", "before"]);
  console.log(`before: kernels/ of ${execFileSync("git", ["rev-parse", "--short", before], { cwd: root }).toString().trim()}`);
}
for (const [source, name] of sources) execFileSync("npx", [...asc, source, "-o", `${work}${name}.wasm`], { cwd: root, stdio: "inherit" });
const modules = Object.fromEntries(sources.map(([, name]) => [name, fs.readFileSync(`${work}${name}.wasm`).toString("base64")]));

const cpu = () => {
  const model = os.cpus()[0]?.model;
  if (model && model !== "unknown") return model;
  const part = (fs.existsSync("/proc/cpuinfo") ? fs.readFileSync("/proc/cpuinfo", "utf8") : "").match(/CPU part\s*:\s*(0x[0-9a-f]+)/)?.[1];
  return part ? `arm64 ${{ "0xd0c": "Neoverse-N1", "0xd49": "Neoverse-N2", "0xd40": "Neoverse-V1", "0xd4f": "Neoverse-V2" }[part] ?? part}` : "unknown";
};
console.log(`ternary-browser-bench: ${cpu()}, ${os.arch()}, ${os.platform()}, ${os.cpus().length} logical cores`);

// Runs in the page (and here): no name from outside it. Returns { engine, rows: [{ shape, form, base, gws, ratio, spread }], wrong: [] }
async function timeForms(modules, options) {
  const bytesOf = (base64) => Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const PAGE = 65536, align = (bytes, to = 4096) => Math.ceil(bytes / to) * to;
  const largest = Math.max(...options.shapes.map(([rows, n]) => rows * n));
  const region = Math.max(options.megabytes * 2 ** 20, 2 * align(largest / 4 + (largest / 128) * 2 + 64));
  const small = align(64 * 1024 + Math.max(...options.shapes.map(([rows, n]) => 2 * n + rows * 4)) * 2, PAGE);
  const memory = new WebAssembly.Memory({ initial: Math.ceil((small + region) / PAGE) });
  const [plain, forms, old] = await Promise.all([modules.plain, modules.forms, modules.before].map(async (m) =>
    m && (await WebAssembly.instantiate(bytesOf(m), { env: { memory } })).instance.exports));
  // every form: its call, and the form it is timed against and held to (its base)
  const ALL = {}, base = {};
  const add = (name, of, call) => { ALL[name] = call; base[name] = of; };
  const first = old ? "t2@before" : "t2";
  if (old) add("t2@before", first, (o, xq, xs, w, s, n, r0, r1) => old.matmul_t2(o, xq, xs, w, s, n, r0, r1, 3));
  add("t2", first, (o, xq, xs, w, s, n, r0, r1) => plain.matmul_t2(o, xq, xs, w, s, n, r0, r1, 3));
  for (const name of options.forms) add(name, first, (o, xq, xs, w, s, n, r0, r1) => forms[`matmul_${name}`](o, xq, xs, w, s, n, r0, r1, 3));
  if (old) {
    add("b1@before", "b1@before", (o, xq, xs, w, s, n, r0, r1) => old.matmul_b1(o, xq, xs, w, s, n, r0, r1, 2));
    add("b1", "b1@before", (o, xq, xs, w, s, n, r0, r1) => plain.matmul_b1(o, xq, xs, w, s, n, r0, r1, 2));
  }
  const names = Object.keys(ALL), bases = [...new Set(Object.values(base))];
  const U = new Uint8Array(memory.buffer), I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const U16 = new Uint16Array(memory.buffer), U32 = new Uint32Array(memory.buffer);
  const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
  const rows = [], wrong = [];
  let seed = 7;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  for (const [count, n] of options.shapes) {
    const groups = (count * n) / 128, matrix = align(count * n / 4 + groups * 2);
    const copies = Math.max(2, Math.floor(region / matrix));
    const XQ = 1024, XS = XQ + align(n), OUT = XS + align((n / 32) * 8), at = small;
    // the activations: int8 in -127..127 and a positive scale a group of 32, laid out by interleave()
    for (let j = 0; j < n; j++) I[XQ + j] = (next() % 255) - 127;
    for (let g = 0; g < n / 32; g++) F[(XS >> 2) + g] = 0.001 + (next() % 1000) / 100000;
    plain.interleave(XQ, XS, n);
    // a matrix: codes 0..2 (two bits a weight) and float16 scales of either sign, then its copies
    for (let b = 0; b < count * n / 4; b++) {
      let byte = 0;
      for (let k = 0; k < 4; k++) byte |= (next() % 3) << (2 * k);
      U[at + b] = byte;
    }
    for (let g = 0; g < groups; g++) U16[((at + count * n / 4) >> 1) + g] = (next() & 0x8000) | (0x2000 + (next() % 0x1800));
    for (let c = 1; c < copies; c++) U.copyWithin(at + c * matrix, at, at + matrix);
    const call = (name, copy, r0, r1) => {
      const w = at + (copy % copies) * matrix;
      ALL[name](OUT, XQ, XS, w, w + count * n / 4, n, r0, r1);
    };
    // tiered up (thousands of calls of a few rows), then held to its base to the bit on every row
    for (const name of names) for (let k = 0; k < 3000; k++) call(name, k, k % count, Math.min(count, (k % count) + 2));
    const held = {};
    for (const name of bases) { call(name, 0, 0, count); held[name] = U32.slice(OUT >> 2, (OUT >> 2) + count); }
    for (const name of names) {
      U.fill(0, OUT, OUT + count * 4);
      call(name, 1, 0, count);
      const ref = held[base[name]];
      for (let r = 0; r < count; r++) {
        if (U32[(OUT >> 2) + r] !== ref[r]) { wrong.push(`${name} ${count}x${n} row ${r}: ${F[(OUT >> 2) + r]} against ${base[name]}'s ${new Float32Array(ref.buffer)[r]}`); break; }
      }
    }
    // the turns: every form through the same calls, the order turned back each turn
    const calls = Math.max(8, Math.ceil(options.weights / (count * n)));
    const speeds = Object.fromEntries(names.map((name) => [name, []]));
    let start = 0;
    for (let turn = 0; turn < options.turns; turn++) {
      const order = turn % 2 ? names.slice().reverse() : names;
      for (const name of order) {
        const t0 = performance.now();
        for (let k = 0; k < calls; k++) call(name, start + k, 0, count);
        const seconds = (performance.now() - t0) / 1000;
        speeds[name].push((count * n * calls) / seconds / 1e9);
      }
      start += calls;
    }
    for (const name of names) {
      const ratios = speeds[name].map((speed, turn) => speed / speeds[base[name]][turn]);
      rows.push({ shape: `${count}x${n}`, form: name, base: base[name], gws: median(speeds[name]), ratio: median(ratios),
        spread: [Math.min(...ratios), Math.max(...ratios)] });
    }
  }
  return { engine: typeof navigator === "undefined" ? "" : navigator.userAgent, rows, wrong };
}

function report(engine, { rows, wrong }) {
  console.log(`\n== ${engine}`);
  console.log("G weights/s (the median over the turns of the ratio to its base in the same turn; the least..the most)");
  const shapes = [...new Set(rows.map((row) => row.shape))], forms = [...new Set(rows.map((row) => row.form))];
  console.log(`| form vs its base | ${shapes.join(" | ")} | geometric mean of the ratios |`);
  console.log(`|---|${shapes.map(() => "---:|").join("")}---:|`);
  for (const form of forms) {
    const mine = shapes.map((shape) => rows.find((row) => row.shape === shape && row.form === form));
    const mean = Math.exp(mine.reduce((sum, row) => sum + Math.log(row.ratio), 0) / mine.length);
    console.log(`| ${form} vs ${mine[0].base} | ${mine.map((row) => `${row.gws.toFixed(2)} (${row.ratio.toFixed(3)}; ${row.spread[0].toFixed(2)}..${row.spread[1].toFixed(2)})`).join(" | ")} | ${mean.toFixed(3)} |`);
  }
  for (const line of wrong) console.log(`WRONG: ${line}`);
  return wrong.length > 0;
}

let failed = false;
for (const engine of engines.length ? engines : ["here"]) {
  if (engine === "here") {
    const runtime = typeof Bun !== "undefined" ? `bun ${Bun.version} (JavaScriptCore)` : `node ${process.version} (V8)`;
    failed = report(runtime, await timeForms(modules, options)) || failed;
    continue;
  }
  const playwright = await import("playwright-core");
  const browser = await playwright[engine].launch();
  try {
    const page = await browser.newPage();
    await page.goto("about:blank");
    const result = await page.evaluate(`(${timeForms.toString()})(${JSON.stringify(modules)}, ${JSON.stringify(options)})`);
    failed = report(`${engine} ${browser.version()}: ${result.engine}`, result) || failed;
  } finally {
    await closed(() => browser.close());
  }
}
await leave(failed ? 1 : 0);
