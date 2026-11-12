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
// T288: --relaxed times the product with relaxed SIMD instead: matmul_t2r (kernel_relaxed.ts) against the forms of
// tests/ternary-relaxed-forms.ts (default rc,x16,x16b,tl), and with --before t2r@before and b1r@before. A form whose
// module exports layout_<form> reads its own copies of the weights, written by that function from the base's; every
// form's call is given a scratch place (tl's tables) after its usual arguments.
//
//   node tests/ternary-browser-bench.ts [here] [chromium webkit firefox] [--turns 15] [--megabytes 128] [--weights 2e9]
//        [--shapes 2048x2048,6144x2048,2048x6144] [--forms t2c,ch,r16,r16s,rt,dt,w2] [--before origin/main] [--relaxed]
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
const relaxed = args.includes("--relaxed");
const options = {
  turns: Number(text("--turns", 15)),
  megabytes: Number(text("--megabytes", 128)),
  weights: Number(text("--weights", 2e9)),
  shapes: text("--shapes", "2048x2048,6144x2048,2048x6144").split(",").map((shape) => shape.split("x").map(Number)),
  forms: text("--forms", relaxed ? "rc,x16,x16b,tl" : "t2c,ch,r16,r16s,rt,dt,w2").split(",").filter(Boolean),
  relaxed,
};

fs.mkdirSync(work, { recursive: true });
const asc = ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1", "--enable",
  relaxed ? "simd,relaxed-simd" : "simd"];
// (the kernels' own module: kernel.ts, or kernel_relaxed.ts with --relaxed; interleave() is kernel.ts's in either)
const sources = [[root + "kernels/kernel.ts", "plain"], [root + (relaxed ? "tests/ternary-relaxed-forms.ts" : "tests/ternary-plain-forms.ts"), "forms"]];
if (relaxed) sources.push([root + "kernels/kernel_relaxed.ts", "relaxed"]);
if (before) {
  // kernels/ as it was at that commit, in a folder of its own
  fs.rmSync(work + "before", { recursive: true, force: true });
  fs.mkdirSync(work + "before");
  // (CI's checkout is one commit deep: a commit it lacks is fetched by its name)
  try {
    execFileSync("git", ["cat-file", "-e", `${before}^{commit}`], { cwd: root, stdio: "ignore" });
  } catch {
    execFileSync("git", ["fetch", "-q", "--depth", "1", "origin", before], { cwd: root, stdio: "inherit" });
  }
  const archive = execFileSync("git", ["archive", before, "kernels"], { cwd: root, maxBuffer: 1 << 28 });
  execFileSync("tar", ["-x", "-C", work + "before"], { input: archive });
  sources.push([work + (relaxed ? "before/kernels/kernel_relaxed.ts" : "before/kernels/kernel.ts"), "before"]);
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
  const widest = Math.max(...options.shapes.map(([, n]) => n));
  const small = align(64 * 1024 + Math.max(...options.shapes.map(([rows, n]) => 2 * n + rows * 4)) * 2 + 1024 + 16 * widest, PAGE);
  const SCRATCH = small - align(1024 + 16 * widest);  // T288: a form's own place (tl's accumulators and tables)
  const [plain, forms, old, own] = await Promise.all([modules.plain, modules.forms, modules.before, modules.relaxed].map(async (m) =>
    m && (await WebAssembly.compile(bytesOf(m)))));
  // the forms that read weights of their own layout: a second region of copies
  const laid = options.forms.filter((name) => WebAssembly.Module.exports(forms).some((e) => e.name === `layout_${name}`));
  const memory = new WebAssembly.Memory({ initial: Math.ceil((small + region * (laid.length ? 2 : 1)) / PAGE) });
  const instance = async (m) => m && (await WebAssembly.instantiate(m, { env: { memory } })).exports;
  const [plainK, formsK, oldK, ownK] = await Promise.all([plain, forms, old, own].map(instance));
  // every form: its call, and the form it is timed against and held to (its base)
  const ALL = {}, base = {};
  const add = (name, of, call) => { ALL[name] = call; base[name] = of; };
  // (the kernels this tree builds: kernel.ts's, or kernel_relaxed.ts's with --relaxed)
  const mine = options.relaxed ? ownK : plainK, t2 = options.relaxed ? "matmul_t2r" : "matmul_t2", b1 = options.relaxed ? "matmul_b1r" : "matmul_b1";
  const tag = options.relaxed ? "r" : "";
  const first = oldK ? `t2${tag}@before` : `t2${tag}`;
  if (oldK) add(first, first, (o, xq, xs, w, s, n, r0, r1) => oldK[t2](o, xq, xs, w, s, n, r0, r1, 3));
  add(`t2${tag}`, first, (o, xq, xs, w, s, n, r0, r1) => mine[t2](o, xq, xs, w, s, n, r0, r1, 3));
  for (const name of options.forms) add(name, first, (o, xq, xs, w, s, n, r0, r1) => formsK[`matmul_${name}`](o, xq, xs, w, s, n, r0, r1, 3, SCRATCH));
  if (oldK) {
    add(`b1${tag}@before`, `b1${tag}@before`, (o, xq, xs, w, s, n, r0, r1) => oldK[b1](o, xq, xs, w, s, n, r0, r1, 2));
    add(`b1${tag}`, `b1${tag}@before`, (o, xq, xs, w, s, n, r0, r1) => mine[b1](o, xq, xs, w, s, n, r0, r1, 2));
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
    plainK.interleave(XQ, XS, n);
    // a matrix: codes 0..2 (two bits a weight) and float16 scales of either sign, then its copies
    for (let b = 0; b < count * n / 4; b++) {
      let byte = 0;
      for (let k = 0; k < 4; k++) byte |= (next() % 3) << (2 * k);
      U[at + b] = byte;
    }
    for (let g = 0; g < groups; g++) U16[((at + count * n / 4) >> 1) + g] = (next() & 0x8000) | (0x2000 + (next() % 0x1800));
    // T288: the forms of their own layout, written from the first copy into the second region
    const second = small + region;
    if (laid.length > 1) throw new Error(`one form of its own layout at a time: ${laid}`);
    for (const name of laid) {
      if (count % 16) throw new Error(`${name}: rows of ${count} are not blocks of 16`);
      formsK[`layout_${name}`](second, at, at + count * n / 4, count, n);
    }
    for (let c = 1; c < copies; c++) U.copyWithin(at + c * matrix, at, at + matrix);
    if (laid.length) for (let c = 1; c < copies; c++) U.copyWithin(second + c * matrix, second, second + matrix);
    const call = (name, copy, r0, r1) => {
      const w = (laid.includes(name) ? second : at) + (copy % copies) * matrix;
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
