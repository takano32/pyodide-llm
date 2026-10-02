// T230's review: the ternary kernels in a real browser's engine, where the same checks as tests/ternary-check.mjs run inside
// the page: a probe for CI (Playwright's WebKit on Linux x86-64 wrote nonsense with a ternary model). Prints, for each
// kernel, how many values differ from the same arithmetic written out in JavaScript, and where, for the engine named by argv[2]
// (webkit | chromium | firefox).
import fs from "node:fs";
import os from "node:os";
import * as playwright from "playwright-core";

const root = new URL("../", import.meta.url).pathname;
const engine = process.argv[2] ?? "webkit";
const b64 = (name) => fs.readFileSync(`${root}public/${name}.wasm`).toString("base64");

async function inPage({ plain, relaxed }) {
  const bytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const memory = new WebAssembly.Memory({ initial: 64 });
  const make = (data) => new WebAssembly.Instance(new WebAssembly.Module(data), { env: { memory } }).exports;
  const k = make(bytes(plain));
  let r = null;
  try { r = make(bytes(relaxed)); } catch (error) { r = null; }
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
  const f = Math.fround;
  let seed = 231;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const report = { userAgent: navigator.userAgent, relaxed: Boolean(r) };
  const ranges = (list) => {
    const out = [];
    for (const x of list) { const last = out[out.length - 1]; if (last && last[1] === x - 1) last[1] = x; else out.push([x, x]); }
    return out.map(([a, b]) => (a === b ? `${a}` : `${a}-${b}`)).join(",");
  };

  // ---- interleave: the bytes (byte 16 p + c of a block of 64 is activation 4 c + p) and the sums after the scales
  {
    const bad = [], badSums = [];
    for (const n of [128, 256, 640, 1152]) {
      const xq = 4096, xs = xq + 2048, ng = n / 32;
      const a = Int8Array.from({ length: n }, () => (next() & 255) - 128);
      I.set(a, xq);
      for (let g = 0; g < ng; g++) F[xs / 4 + g] = g + 0.5;
      N.fill(-7, xs / 4 + ng, xs / 4 + 2 * ng);
      k.interleave(xq, xs, n);
      for (let at = 0; at < n; at++) if (I[xq + at] !== a[(at & ~63) + 4 * (at & 15) + ((at >> 4) & 3)]) bad.push(`${n}:${at}`);
      for (let g = 0; g < ng; g++) {
        const want = -a.subarray(g * 32, g * 32 + 32).reduce((s, v) => s + v, 0);
        if (N[xs / 4 + ng + g] !== want) badSums.push(`${n}:${g}(${N[xs / 4 + ng + g]} not ${want})`);
      }
    }
    report.interleave = { badBytes: bad.length, first: bad.slice(0, 6), badSums: badSums.length, firstSums: badSums.slice(0, 4) };
  }

  // ---- matmul_t2 (and t2r where there is relaxed SIMD) against JavaScript, by groups
  const rows = 13, most = 9;
  const w = 65536, ws = w + rows * most * 32, xq = ws + rows * most * 4, xs = xq + most * 128, out = xs + most * 4 * 8;
  const expectedOf = (codes, a, groups, scalesX, scalesW) => Array.from({ length: rows }, (_, i) => {
    const n = groups * 128, lanes = [0, 0, 0, 0];
    for (let g = 0; g < groups; g++) {
      for (let lane = 0; lane < 4; lane++) {
        let sum = 0;
        for (let j = g * 128 + lane * 32; j < g * 128 + lane * 32 + 32; j++) sum += (codes[i * n + j] - 1) * a[j];
        lanes[lane] = f(lanes[lane] + f(f(sum * scalesX[g * 4 + lane]) * scalesW[i * groups + g]));
      }
    }
    return f(f(f(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
  });
  report.matmul = {};
  for (const [name, kernel] of [["t2", k.matmul_t2], ...(r ? [["t2r", r.matmul_t2r]] : [])]) {
    const bad = [];
    for (const groups of [1, 2, 4, most]) {
      const n = groups * 128, ng = n / 32;
      const codes = Uint8Array.from({ length: rows * n }, () => next() % 3);  // -1, 0, 1 only
      U.fill(0, w, w + rows * groups * 32);
      codes.forEach((code, j) => { U[w + (j >> 2)] |= code << (2 * (j & 3)); });
      const sw = Float32Array.from({ length: rows * groups }, () => f(1e-3 * (1 + (next() % 1000))));
      F.set(sw, ws / 4);
      const a = Int8Array.from({ length: n }, () => (next() & 255) - 128);
      I.set(a, xq);
      const sx = Float32Array.from({ length: ng }, () => f(1e-2 * (1 + (next() % 1000))));
      F.set(sx, xs / 4);
      k.interleave(xq, xs, n);
      const want = expectedOf(codes, a, groups, sx, sw);
      F.fill(-7, out / 4, out / 4 + rows);
      kernel(out, xq, xs, w, ws, n, 0, rows, 3);
      for (let i = 0; i < rows; i++) if (!Object.is(F[out / 4 + i], want[i])) bad.push(`${groups}g:row${i}(${F[out / 4 + i]} not ${want[i]})`);
    }
    report.matmul[name] = { bad: bad.length, first: bad.slice(0, 5) };
  }

  // ---- the isolating probes (one group of 128, interleaved by interleave(), scales 1): a single weight of +1 against
  // activations of 1; then a single activation of 5 against weights of +1. The row's number is 1 or 5 where it is right
  for (const [name, kernel] of [["t2", k.matmul_t2], ...(r ? [["t2r", r.matmul_t2r]] : [])]) {
    const weightSpots = [], activationSpots = [];
    for (let j = 0; j < 128; j++) {
      // a single weight of +1 at j (the rest 0, code 1)
      U.fill(0, w, w + 32);
      for (let q = 0; q < 128; q++) U[w + (q >> 2)] |= (q === j ? 2 : 1) << (2 * (q & 3));
      F[ws / 4] = 1;
      for (let q = 0; q < 128; q++) I[xq + q] = 1;
      for (let g = 0; g < 4; g++) F[xs / 4 + g] = 1;
      k.interleave(xq, xs, 128);
      F[out / 4] = -7;
      kernel(out, xq, xs, w, ws, 128, 0, 1, 3);
      if (F[out / 4] !== 1) weightSpots.push(j);
      // a single activation of 5 at j against weights all +1
      U.fill(0, w, w + 32);
      for (let q = 0; q < 128; q++) U[w + (q >> 2)] |= 2 << (2 * (q & 3));
      for (let q = 0; q < 128; q++) I[xq + q] = q === j ? 5 : 0;
      for (let g = 0; g < 4; g++) F[xs / 4 + g] = 1;
      k.interleave(xq, xs, 128);
      F[out / 4] = -7;
      kernel(out, xq, xs, w, ws, 128, 0, 1, 3);
      if (F[out / 4] !== 5) activationSpots.push(j);
    }
    report[`single_${name}`] = { weightWrongAt: ranges(weightSpots), activationWrongAt: ranges(activationSpots) };
  }

  // ---- ternary_x against its definition
  {
    const groups = 40, x = 65536, packed = x + groups * 512, scales = packed + groups * 32;
    for (let g = 0; g < groups; g++) {
      const d = f([1, 0.0078125, 3.0000002, 1e-30, 65504, 1.5e-5][g % 6] * (1 + (next() % 7)));
      for (let j = 0; j < 128; j++) F[x / 4 + g * 128 + j] = ((next() % 3) - 1) * d;
    }
    U.fill(0xAA, packed, packed + groups * 32);
    const refused = k.ternary_x(packed, scales, x, groups * 128);
    const bad = [];
    for (let g = 0; g < groups; g++) {
      let largest = 0;
      for (let j = 0; j < 128; j++) largest = Math.max(largest, Math.abs(F[x / 4 + g * 128 + j]));
      if (F[scales / 4 + g] !== largest) bad.push(`scale ${g}`);
      for (let b = 0; b < 32; b++) {
        let want = 0;
        for (let p = 0; p < 4; p++) want |= (Math.sign(F[x / 4 + g * 128 + 4 * b + p]) + 1) << (2 * p);
        if (U[packed + g * 32 + b] !== want) bad.push(`byte ${g}:${b}(${U[packed + g * 32 + b]} not ${want})`);
      }
    }
    report.ternary_x = { refused, bad: bad.length, first: bad.slice(0, 6) };
  }

  // ---- quantize_x with no bias: the int8 and the scales of a group of 32
  {
    const n = 128, x = 65536, q = x + 1024, s = q + 256;
    const v = Float32Array.from({ length: n }, () => (((next() % 2001) - 1000) / 250));
    F.set(v, x / 4);
    k.quantize_x(q, s, x, n, 0);
    const bad = [];
    for (let g = 0; g < n / 32; g++) {
      let top = 0;
      for (let j = 0; j < 32; j++) top = Math.max(top, Math.abs(v[g * 32 + j]));
      const scale = f(top / 127);
      if (F[s / 4 + g] !== scale) bad.push(`scale ${g}`);
      for (let j = 0; j < 32; j++) {
        const t = f(v[g * 32 + j] * f(1 / scale));
        let rounded = Math.round(t); if (Math.abs(t % 1) === 0.5) rounded = 2 * Math.round(t / 2);  // half to even
        if (I[q + g * 32 + j] !== rounded) bad.push(`q ${g * 32 + j}(${I[q + g * 32 + j]} not ${rounded})`);
      }
    }
    report.quantize_x = { bad: bad.length, first: bad.slice(0, 4) };
  }
  return report;
}

// the same kernels on a 64-bit memory, each address argument above 4 GiB one at a time (the arm64 V8 of Node 24 reads the
// weights' scales and the activations' scales of the tile at the low 32 bits of such an address)
async function inPageHigh({ plain, relaxed }) {
  const bytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const HIGH = 4 * 2 ** 30 + 2 * 65536;
  let memory;
  try {
    memory = new WebAssembly.Memory({ initial: BigInt(Math.ceil((HIGH + 8 * 2 ** 20) / 65536)), address: "i64" });
  } catch (error) {
    try {
      memory = new WebAssembly.Memory({ initial: BigInt(Math.ceil((HIGH + 8 * 2 ** 20) / 65536)), index: "i64" });
    } catch (second) {
      return { skipped: `no 64-bit memory: ${error.message}` };
    }
  }
  const wrap = (exports, names) => {
    const out = { ...exports };
    for (const [name, at] of Object.entries(names)) {
      const kernel = exports[name];
      out[name] = (...args) => { for (const i of at) args[i] = BigInt(args[i]); return kernel(...args); };
    }
    return out;
  };
  const addresses = { interleave: [0, 1], matmul_t2: [0, 1, 2, 3, 4], matmul_t2r: [0, 1, 2, 3, 4], matmul_t2r_tile: [0, 1, 2, 3, 4],
    matmul_q8r: [0, 1, 2, 3, 4, 5], matmul_q8r_tile: [0, 1, 2, 3, 4, 5], matmul_q8: [0, 1, 2, 3, 4], matmul_q6: [0, 1, 2, 3, 4] };
  let k, r;
  try {
    k = wrap(new WebAssembly.Instance(new WebAssembly.Module(bytes(plain)), { env: { memory } }).exports, addresses);
    r = wrap(new WebAssembly.Instance(new WebAssembly.Module(bytes(relaxed)), { env: { memory } }).exports, addresses);
  } catch (error) {
    return { skipped: `the 64-bit kernels do not instantiate: ${error.message}` };
  }
  const U = new Uint8Array(memory.buffer), I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
  const LOW = 65536, UP = HIGH + 65536, SPREAD = [0, 8192, 16384, 24576, 40960, 49152];
  const cases = {
    q8r: { args: ["out", "xq", "xs", "w", "ws", "wc"], want: 768,
      setup(p) { for (let i = 0; i < 128; i++) I[p.w + i] = 2; for (let g = 0; g < 4; g++) { F[p.ws / 4 + g] = 1; N[p.wc / 4 + g] = -64 * 2 * 32; F[p.xs / 4 + g] = 1; } for (let j = 0; j < 128; j++) I[p.xq + j] = 67; },
      call: (p) => r.matmul_q8r(p.out, p.xq, p.xs, p.w, p.ws, p.wc, 128, 0, 1) },
    q8r_tile: { args: ["out", "xq", "xs", "w", "ws", "wc"], want: 768,
      setup(p) { for (let i = 0; i < 512; i++) I[p.w + i] = 2; for (let g = 0; g < 16; g++) { F[p.ws / 4 + g] = 1; N[p.wc / 4 + g] = -64 * 2 * 32; }
        for (let t = 0; t < 4; t++) { for (let j = 0; j < 128; j++) I[p.xq + t * 1024 + j] = 67; for (let g = 0; g < 4; g++) F[(p.xs + t * 1024) / 4 + g] = 1; } },
      call: (p) => r.matmul_q8r_tile(p.out, p.xq, p.xs, p.w, p.ws, p.wc, 128, 0, 4, 4, 64, 1024) },
    t2r: { args: ["out", "xq", "xs", "w", "ws"], want: 384,
      setup(p) { U.fill(0, p.w, p.w + 32); for (let j = 0; j < 128; j++) U[p.w + (j >> 2)] |= 2 << (2 * (j & 3)); F[p.ws / 4] = 1;
        for (let j = 0; j < 128; j++) I[p.xq + j] = 3; for (let g = 0; g < 4; g++) F[p.xs / 4 + g] = 1; k.interleave(p.xq, p.xs, 128); },
      call: (p) => r.matmul_t2r(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1, 3) },
    t2: { args: ["out", "xq", "xs", "w", "ws"], want: 384, setup(p) { cases.t2r.setup(p); }, call: (p) => k.matmul_t2(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1, 3) },
  };
  const result = {};
  for (const [name, c] of Object.entries(cases)) {
    const bad = new Set();
    for (let mask = 0; mask < 2 ** c.args.length; mask++) {
      const p = Object.fromEntries(c.args.map((arg, i) => [arg, (mask >> i & 1 ? UP : LOW) + SPREAD[i]]));
      c.setup(p);
      F.fill(-7, p.out / 4, p.out / 4 + 40);
      c.call(p);
      if (F[p.out / 4] !== c.want) for (let i = 0; i < c.args.length; i++) if (mask === (1 << i)) bad.add(c.args[i]);
    }
    result[name] = bad.size ? `WRONG alone: ${[...bad].join(", ")}` : "right";
  }
  return result;
}

// the same kernels called over and over, so that an engine that compiles a hot function again (JavaScriptCore's BBQ then OMG,
// V8's Liftoff then TurboFan) runs the optimized code: after each batch of calls (and a pause for the compile to land) the
// result is checked against the JavaScript arithmetic. Reports the first batch whose result is wrong, for every kernel.
async function inPageWarm({ plain, relaxed }) {
  const bytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const memory = new WebAssembly.Memory({ initial: 64 });
  const make = (data) => new WebAssembly.Instance(new WebAssembly.Module(data), { env: { memory } }).exports;
  const k = make(bytes(plain));
  let r = null;
  try { r = make(bytes(relaxed)); } catch (error) { r = null; }
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
  const f = Math.fround;
  let seed = 7;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const rows = 64, groups = 16, n = groups * 128, ng = n / 32;
  const w = 65536, ws = w + rows * n / 4, xq = ws + rows * groups * 4 + 64, xs = xq + n + 64, out = xs + ng * 8 + 64, copy = out + rows * 4 + 64;
  const codes = Uint8Array.from({ length: rows * n }, () => next() % 3);
  U.fill(0, w, w + rows * n / 4);
  codes.forEach((code, j) => { U[w + (j >> 2)] |= code << (2 * (j & 3)); });
  const sw = Float32Array.from({ length: rows * groups }, () => f(1e-3 * (1 + (next() % 1000))));
  F.set(sw, ws / 4);
  const a = Int8Array.from({ length: n }, () => (next() & 255) - 128);
  const sx = Float32Array.from({ length: ng }, () => f(1e-2 * (1 + (next() % 1000))));
  const wantRows = Array.from({ length: rows }, (_, i) => {
    const lanes = [0, 0, 0, 0];
    for (let g = 0; g < groups; g++) {
      for (let lane = 0; lane < 4; lane++) {
        let sum = 0;
        for (let j = g * 128 + lane * 32; j < g * 128 + lane * 32 + 32; j++) sum += (codes[i * n + j] - 1) * a[j];
        lanes[lane] = f(lanes[lane] + f(f(sum * sx[g * 4 + lane]) * sw[i * groups + g]));
      }
    }
    return f(f(f(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
  });
  const prepare = () => { I.set(a, xq); F.set(sx, xs / 4); k.interleave(xq, xs, n); };
  const kernels = { t2: () => k.matmul_t2(out, xq, xs, w, ws, n, 0, rows, 3), ...(r ? { t2r: () => r.matmul_t2r(out, xq, xs, w, ws, n, 0, rows, 3),
    t2r_tile: () => r.matmul_t2r_tile(out, xq, xs, w, ws, n, 0, rows, 1, 0, 0, 3) } : {}) };
  const sizes = [1, 20, 300, 2000, 8000, 30000, 60000];
  const report = {};
  const verify = (call) => {
    prepare();
    F.fill(-7, out / 4, out / 4 + rows);
    call();
    for (let i = 0; i < rows; i++) if (!Object.is(F[out / 4 + i], wantRows[i])) return `row ${i}: ${F[out / 4 + i]} not ${wantRows[i]}`;
    return null;
  };
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  for (const [name, call] of Object.entries(kernels)) {
    let calls = 0, firstBad = null;
    for (const size of sizes) {
      prepare();
      for (; calls < size; calls++) call();
      await pause(300);
      const wrong = verify(call);
      if (wrong && !firstBad) firstBad = `after ${calls} calls: ${wrong}`;
    }
    report[name] = firstBad ?? `right through ${calls} calls`;
  }
  // the interleave itself, many times, and quantize_x and ternary_x
  {
    let firstBad = null, calls = 0;
    for (const size of sizes) {
      for (; calls < size; calls++) { I.set(a, xq); F.set(sx, xs / 4); k.interleave(xq, xs, n); }
      await pause(300);
      I.set(a, xq); F.set(sx, xs / 4); k.interleave(xq, xs, n);
      let bad = 0;
      for (let at = 0; at < n; at++) if (I[xq + at] !== a[(at & ~63) + 4 * (at & 15) + ((at >> 4) & 3)]) bad++;
      for (let g = 0; g < ng; g++) if (N[xs / 4 + ng + g] !== -a.subarray(g * 32, g * 32 + 32).reduce((s, v) => s + v, 0)) bad++;
      if (bad && !firstBad) firstBad = `after ${calls} calls: ${bad} wrong`;
    }
    report.interleave = firstBad ?? `right through ${calls} calls`;
  }
  {
    // ternary_x on a matrix of ternary values
    const groupsX = 40, x = 262144, packed = x + groupsX * 512, scales = packed + groupsX * 32;
    for (let g = 0; g < groupsX; g++) {
      const d = f(0.01 * (1 + g % 7));
      for (let j = 0; j < 128; j++) F[x / 4 + g * 128 + j] = ((next() % 3) - 1) * d;
    }
    const check = () => {
      U.fill(0xAA, packed, packed + groupsX * 32);
      const refused = k.ternary_x(packed, scales, x, groupsX * 128);
      let bad = refused ? 1 : 0;
      for (let g = 0; g < groupsX; g++) for (let b = 0; b < 32; b++) {
        let want = 0;
        for (let p = 0; p < 4; p++) want |= (Math.sign(F[x / 4 + g * 128 + 4 * b + p]) + 1) << (2 * p);
        if (U[packed + g * 32 + b] !== want) bad++;
      }
      return bad;
    };
    let firstBad = null, calls = 0;
    for (const size of [1, 20, 300, 2000, 8000]) {
      for (; calls < size; calls++) k.ternary_x(packed, scales, x, groupsX * 128);
      await pause(300);
      const bad = check();
      if (bad && !firstBad) firstBad = `after ${calls} calls: ${bad} wrong`;
    }
    report.ternary_x = firstBad ?? `right through ${calls} calls`;
  }
  return report;
}

const browser = await playwright[engine].launch();
try {
  const page = await browser.newPage();
  await page.goto("about:blank");
  const report = await page.evaluate(inPage, { plain: b64("simdkernel_plain"), relaxed: b64("simdkernel_relaxed_plain") });
  console.log(`jsc-probe: ${engine} on ${os.cpus()[0].model} (${process.arch})`);
  console.log(`jsc-probe: ${JSON.stringify(report)}`);
  const high = await page.evaluate(inPageHigh, { plain: b64("simdkernel_plain64"), relaxed: b64("simdkernel_relaxed_plain64") });
  console.log(`jsc-probe: above 4 GiB: ${JSON.stringify(high)}`);
  const warm = await page.evaluate(inPageWarm, { plain: b64("simdkernel_plain"), relaxed: b64("simdkernel_relaxed_plain") });
  console.log(`jsc-probe: called over and over: ${JSON.stringify(warm)}`);
} finally {
  await browser.close();
}
process.exit(0);
