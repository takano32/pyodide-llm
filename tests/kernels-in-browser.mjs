// kernels-in-browser.mjs (T230's review): the kernels' arithmetic in the engine of each browser, called often enough that
// the engine compiles them again. tests/ternary-check.mjs and tests/smoke.mjs run the kernels in V8 (Node) alone, and a
// browser's wasm engine is another compiler: JavaScriptCore on x86-64 (Playwright's WebKit on Linux, on AMD EPYCs and an Intel Xeon
// alike; Safari on an Intel Mac has the same compiler and was not tried) wrote nonsense with a ternary model because its
// optimizing tier, which a function reaches after about 2000 calls and a test of a few calls never does, folded the
// three shuffles of interleave() wrongly. Nothing else saw it: every
// end-to-end run of a model asks for an answer to appear, not for it to be right.
//
// Each engine gets the compiled kernels (src/runtime/built/*.wasm of this checkout after `make kernels`, or those of a deployed site
// with --site) in a blank page, and each kernel is called in batches (1, 400, 4000 and 30000 calls, a pause after each for
// the background compile to land) with its result held to the same arithmetic written out in JavaScript after every batch.
// The relaxed kernels are checked where the engine has relaxed SIMD (the others are not run there, as in the page).
// T233's review: and rotate and unrotate (the rotated basis of Ternary Bonsai 2 27B, T237: a token turns 2,122 blocks of
// 1024 with them, thousands of calls an answer), which only V8 had seen, a few calls at a time (tests/rotate-check.mjs):
// Chrome and Firefox, the engines this model runs in, tier them up too.
// Then the 64-bit builds on a memory of more than 4 GiB (what a 27B model runs on, in Chrome and Firefox): every address
// argument low or high in every combination, the result held to the all-low one. The V8 of Node 24 on arm64 fails this
// (it reads v128.load32_splat above 4 GiB at the low 32 bits); WebKit has no such memory and is skipped.
// T363.4: and the kernels of llama.cpp's blocks (kernels/blocks.ts, blocks_relaxed.ts: Underdog Saluki's weights), every
// type of the ten through every kernel, each held after every batch to tests/blocks-model.mjs, to the bit (the model's
// text goes to the page; the codebooks are made up there, which the kernels do not mind: blocks-check.mjs is where the
// reading is held to the converter's). matmul_sbw_tile where the engine's relaxed dot product multiplies signed bytes by
// signed bytes, which is said. Then the 64-bit builds with every address low or above 4 GiB.
//
//   node tests/kernels-in-browser.mjs [--site https://takano32.github.io/pyodide-llm/] [chromium firefox webkit chrome msedge]
// Exit 1 where a result is wrong. Needs playwright-core's browsers installed (the workflows' own step).
import fs from "node:fs";
import os from "node:os";
import * as playwright from "playwright-core";
import { blocksModel } from "./blocks-model.mjs";
import { fetched } from "./fetching.mjs";
import { closed, leave } from "./leave.mjs";
import { built, runtimeUrl } from "./tree.mjs";
const { ADDRESSES } = await import(runtimeUrl("jobs.ts"));

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const siteAt = args.indexOf("--site");
const site = siteAt >= 0 ? args[siteAt + 1].replace(/\/?$/, "/") : null;
const engines = args.filter((arg, i) => !arg.startsWith("--") && args[i - 1] !== "--site");
const channels = { chrome: "chrome", msedge: "msedge" };

// T367.3.2: a deployed site names a built file by a hash of its bytes (_astro/simdkernel_plain.<hash>.wasm), and the
// name is written nowhere but in the worker's text: the page names its script, the script the worker, the worker the
// kernels. So the site is read as a browser reads it. (A site of before T367.3 served <name>.wasm at its root.)
let workerText;
const textOf = async (url) => (await fetched(url)).toString();
async function siteWorker() {
  const page = await textOf(site);
  for (const script of new Set(page.match(/_astro\/[\w.-]+\.js/g) ?? [])) {
    const worker = /\bworker\.[\w-]{6,}\.js/.exec(await textOf(new URL(script, site)))?.[0];
    if (worker) return textOf(new URL(`_astro/${worker}`, site));
  }
  return null;  // (no bundled worker: the built files are at the site's root)
}
async function wasm(name) {
  if (!site) return fs.readFileSync(built(`${name}.wasm`)).toString("base64");
  workerText ??= siteWorker();
  const hashed = new RegExp(`\\b${name}\\.[\\w-]{6,}\\.wasm`).exec((await workerText) ?? "")?.[0];
  return (await fetched(new URL(hashed ? `_astro/${hashed}` : `${name}.wasm`, site))).toString("base64");
}
const modules = { plain: await wasm("simdkernel_plain"), relaxed: await wasm("simdkernel_relaxed_plain") };
const wideModules = { plain: await wasm("simdkernel_plain64"), relaxed: await wasm("simdkernel_relaxed_plain64"), addresses: ADDRESSES };
// (T363.4. A site of before T363.5 serves no kernels of the blocks: they are then not checked there, and that is said)
const blocks = await (async () => {
  try {
    return { plain: await wasm("blockkernel_plain"), relaxed: await wasm("blockkernel_relaxed_plain"), plain64: await wasm("blockkernel_plain64"),
      relaxed64: await wasm("blockkernel_relaxed_plain64"), model: String(blocksModel), addresses: ADDRESSES };
  } catch (error) {
    if (!site) throw error;
    return null;
  }
})();

// runs in the page (T363.4): the kernels of llama.cpp's blocks. { relaxed, signed, results, high }
async function inPageBlocks({ plain, relaxed, plain64, relaxed64, model, addresses }) {
  const bytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const m = (0, eval)(`(${model})`)(), NAMES = Object.keys(m.TYPES);
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let seed = 3634;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  // codebooks of this page's own: magnitudes a grid may hold (the 7-bit side of a dot product), IQ1's -1, 0, 1, and 16
  // signed values
  const books = new Int8Array(m.BOOKS.bytes);
  for (let i = 0; i < m.BOOKS.IQ1S; i++) books[i] = 1 + (next() % 62);
  for (let i = m.BOOKS.IQ1S; i < m.BOOKS.IQ4NL; i++) books[i] = (next() % 3) - 1;
  for (let j = 0; j < 16; j++) books[m.BOOKS.IQ4NL + j] = j === 0 ? -127 : j === 15 ? 127 : (next() % 255) - 127;
  // rows of random bytes whose float16 scales are finite (blocks-check.mjs's)
  const rowsOf = (name, count) => {
    const size = m.TYPES[name][1], raw = Uint8Array.from({ length: count * size }, () => next() & 255);
    const fix = (at) => { if ((raw[at + 1] & 0x7c) === 0x7c) raw[at + 1] &= ~0x04; };
    for (let b = 0; b < count; b++) {
      const p = b * size;
      if (name === "IQ1_M") raw[p + 55] &= ~0x40;
      else if (name === "Q2_K") { fix(p + 80); fix(p + 82); }
      else if (name === "Q4_K") { fix(p); fix(p + 2); }
      else fix(p);
    }
    return raw;
  };
  const activations = (n) => Int8Array.from({ length: n }, (_, j) => (j % 61 === 7 ? 127 : j % 67 === 11 ? -127 : (next() % 255) - 127));
  const scales = (n) => Float32Array.from({ length: n / 32 }, () => Math.fround(1e-3 * (1 + (next() % 4000))));
  const same = (got, want) => {
    for (let i = 0; i < want.length; i++) if (!Object.is(got[i], want[i])) return `value ${i}: ${got[i]} not ${want[i]}`;
    return null;
  };
  const results = [], batches = [1, 400, 4000, 30000];
  async function check(name, call, verify) {
    let calls = 0, wrong = null;
    for (const size of batches) {
      for (; calls < size; calls++) call();
      await pause(100);
      wrong = wrong ?? verify();
    }
    results.push({ name, wrong: wrong && `after up to ${calls} calls: ${wrong}`, calls });
  }
  const memory = new WebAssembly.Memory({ initial: 64 });
  const make = (data) => new WebAssembly.Instance(new WebAssembly.Module(data), { env: { memory } }).exports;
  const k = make(bytes(plain));
  let r = null;
  try { r = make(bytes(relaxed)); } catch (error) { r = null; }  // no relaxed SIMD in this engine
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
  // the places: rows of two blocks, a prompt of five tokens (a tile of four and one more)
  const ROWS = 6, n = 512, TOKENS = 5, frame = n + n / 2 + 96, os = ROWS * 4 + 24;
  const tab = 65536, bookAt = tab + 65536, w = bookAt + 65536, xq = w + 65536, xs = xq + n, outs = xq + 65536, floatsAt = outs + 65536, scratch = floatsAt + 65536;
  I.set(books, bookAt);
  k.sb_tables(tab, bookAt);
  const signed = r ? m.signedDots(r.signed_dots, I, N, scratch) : false;
  for (const name of NAMES) {
    const [type, size] = m.TYPES[name], raw = rowsOf(name, ROWS * 2);
    U.set(raw, w);
    const prompt = Array.from({ length: TOKENS }, (_, t) => {
      const a = activations(n), sx = scales(n);
      I.set(a, xq + t * frame);
      F.set(sx, (xs + t * frame) / 4);
      k.sums_x(xs + t * frame, xq + t * frame, n);
      return m.expected(name, raw, 0, n, ROWS, books, a, sx);
    });
    const all = Float32Array.from({ length: TOKENS * (os / 4) }, (_, i) => (i % (os / 4) < ROWS ? prompt[Math.floor(i / (os / 4))][i % (os / 4)] : -7));
    const token = (kernel) => [() => kernel(outs, xq, xs, w, tab, n, 0, ROWS, type), () => {
      F.fill(-7, outs / 4, outs / 4 + ROWS);
      kernel(outs, xq, xs, w, tab, n, 0, ROWS, type);
      return same(F.subarray(outs / 4, outs / 4 + ROWS), prompt[0]);
    }];
    const tile = (kernel, ...more) => [() => kernel(outs, xq, xs, w, tab, n, 0, ROWS, TOKENS, os, frame, type, ...more), () => {
      F.fill(-7, outs / 4, (outs + TOKENS * os) / 4);
      kernel(outs, xq, xs, w, tab, n, 0, ROWS, TOKENS, os, frame, type, ...more);
      return same(F.subarray(outs / 4, (outs + TOKENS * os) / 4), all);
    }];
    await check(`matmul_sb, ${name}`, ...token(k.matmul_sb));
    const values = m.values(name, raw, 0, n, 1, books);
    await check(`sb_row, ${name}`, () => k.sb_row(floatsAt, w, tab, n, type), () => {
      F.fill(-7, floatsAt / 4, floatsAt / 4 + n);
      k.sb_row(floatsAt, w, tab, n, type);
      return same(F.subarray(floatsAt / 4, floatsAt / 4 + n), values);
    });
    if (r) {
      await check(`matmul_sbr, ${name}`, ...token(r.matmul_sbr));
      await check(`matmul_sbr_tile, ${name}`, ...tile(r.matmul_sbr_tile));
      if (signed) await check(`matmul_sbw_tile, ${name}`, ...tile(r.matmul_sbw_tile, scratch));
    }
  }
  {
    const a = activations(n), want = m.sums(a), H = new Int16Array(memory.buffer);
    I.set(a, xq);
    await check("sums_x", () => k.sums_x(xs, xq, n), () => {
      H.fill(-7, (xs + n / 8) / 2, (xs + n / 2) / 2);
      k.sums_x(xs, xq, n);
      return same(H.subarray((xs + n / 8) / 2, (xs + n / 2) / 2), want);
    });
  }
  // ---- the 64-bit builds: every address argument low or above 4 GiB, in every combination, the first output the
  // all-low one (which is the model's)
  const high = await (async () => {
    const HIGH = 4 * 2 ** 30 + 2 * 65536, pages = BigInt(Math.ceil((HIGH + 16 * 2 ** 20) / 65536));
    let wide;
    try {
      wide = new WebAssembly.Memory({ initial: pages, address: "i64" });
    } catch (error) {
      try { wide = new WebAssembly.Memory({ initial: pages, index: "i64" }); } catch (second) { return { skipped: `no 64-bit memory (${error.message})` }; }
    }
    const wrap = (exports) => {
      const out = { ...exports };
      for (const [name, at] of Object.entries(addresses)) {
        const kernel = exports[name];
        if (kernel) out[name] = (...args) => { for (const i of at) args[i] = BigInt(args[i]); return kernel(...args); };
      }
      return out;
    };
    let k64, r64 = null;
    try {
      k64 = wrap(new WebAssembly.Instance(new WebAssembly.Module(bytes(plain64)), { env: { memory: wide } }).exports);
    } catch (error) {
      return { skipped: `the 64-bit kernels do not instantiate (${error.message})` };
    }
    try { r64 = wrap(new WebAssembly.Instance(new WebAssembly.Module(bytes(relaxed64)), { env: { memory: wide } }).exports); } catch (error) { r64 = null; }
    const U = new Uint8Array(wide.buffer), I = new Int8Array(wide.buffer), F = new Float32Array(wide.buffer);
    const LOW = 2 ** 20, UP = HIGH + 2 ** 20, n = 256, frame = 1024;
    const found = [];
    for (const name of NAMES) {
      const [type] = m.TYPES[name], raw = rowsOf(name, 1), tokens = Array.from({ length: 4 }, () => [activations(n), scales(n)]);
      const want = m.expected(name, raw, 0, n, 1, books, ...tokens[0])[0], value = m.values(name, raw, 0, n, 1, books)[0];
      const cases = [
        { name: "matmul_sb", args: ["out", "xq", "xs", "w", "tab"], want, call: (p) => k64.matmul_sb(p.out, p.xq, p.xs, p.w, p.tab, n, 0, 1, type) },
        { name: "sb_row", args: ["out", "w", "tab"], want: value, call: (p) => k64.sb_row(p.out, p.w, p.tab, n, type) },
        r64 && { name: "matmul_sbr", args: ["out", "xq", "xs", "w", "tab"], want, call: (p) => r64.matmul_sbr(p.out, p.xq, p.xs, p.w, p.tab, n, 0, 1, type) },
        r64 && { name: "matmul_sbr_tile", args: ["out", "xq", "xs", "w", "tab"], want, call: (p) => r64.matmul_sbr_tile(p.out, p.xq, p.xs, p.w, p.tab, n, 0, 1, 4, 64, frame, type) },
        r64 && signed && { name: "matmul_sbw_tile", args: ["out", "xq", "xs", "w", "tab", "scratch"], want,
          call: (p) => r64.matmul_sbw_tile(p.out, p.xq, p.xs, p.w, p.tab, n, 0, 1, 4, 64, frame, type, p.scratch) },
      ].filter(Boolean);
      for (const c of cases) {
        const bad = [];
        for (let mask = 0; mask < 2 ** c.args.length; mask++) {
          const p = Object.fromEntries(["out", "xq", "xs", "w", "tab", "scratch", "books"].map((arg, i) => [arg, ((mask >> c.args.indexOf(arg)) & 1 && c.args.includes(arg) ? UP : LOW) + i * 65536]));
          I.set(books, p.books);
          k64.sb_tables(p.tab, p.books);
          U.set(raw, p.w);
          tokens.forEach(([a, sx], t) => { I.set(a, p.xq + t * frame); F.set(sx, (p.xs + t * frame) / 4); k64.sums_x(p.xs + t * frame, p.xq + t * frame, n); });
          F.fill(-7, p.out / 4, p.out / 4 + 40);
          c.call(p);
          if (!Object.is(F[p.out / 4], c.want)) bad.push(mask);
        }
        const alone = c.args.filter((_, i) => bad.includes(1 << i));
        found.push({ name: `${c.name}, ${name}`, calls: 2 ** c.args.length,
          wrong: bad.length ? `${bad.length} of ${2 ** c.args.length} placements wrong${bad.includes(0) ? " (all low too)" : ""}; above 4 GiB alone: ${alone.join(", ") || "only together"}` : null });
      }
    }
    return { results: found };
  })().catch((error) => ({ skipped: `the page could not run it (${String(error.message).split("\n")[0]})` }));
  return { relaxed: Boolean(r), signed, results, high };
}

// runs in the page: [{ name, wrong }]: wrong is null or the first batch's mismatch
async function inPage({ plain, relaxed }) {
  const bytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const memory = new WebAssembly.Memory({ initial: 128 });
  const make = (data) => new WebAssembly.Instance(new WebAssembly.Module(data), { env: { memory } }).exports;
  const k = make(bytes(plain));
  let r = null;
  try { r = make(bytes(relaxed)); } catch (error) { r = null; }  // no relaxed SIMD in this engine
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
  const f = Math.fround;
  let seed = 7;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const batches = [1, 400, 4000, 30000];
  const results = [];
  // one check: call() until the batch is reached, then verify() says null or what is wrong
  async function check(name, call, verify) {
    let calls = 0, wrong = null;
    for (const size of batches) {
      for (; calls < size; calls++) call();
      await pause(250);
      wrong = wrong ?? verify();
    }
    results.push({ name, wrong: wrong && `after up to ${calls} calls: ${wrong}`, calls });
  }
  // the memory: a ternary matrix of 64 rows of 16 groups of 128 (a row of 2048), its scales, the activations of a token
  // and of a prompt's tokens, the outputs
  const rows = 64, groups = 16, n = groups * 128, ng = n / 32, tokens = 5, frame = 4096;
  const w = 65536, ws = w + rows * n / 4, xq = ws + rows * groups * 4 + 64, xs = xq + n, out = xs + ng * 8 + 64;
  const frames = out + rows * 4 + 4096, outs = frames + tokens * frame, extra = outs + tokens * 1024;
  const codes = Uint8Array.from({ length: rows * n }, () => next() % 3);  // -1, 0, 1
  U.fill(0, w, w + rows * n / 4);
  codes.forEach((code, j) => { U[w + (j >> 2)] |= code << (2 * (j & 3)); });
  const sw = Float32Array.from({ length: rows * groups }, () => f(1e-3 * (1 + (next() % 1000))));
  F.set(sw, ws / 4);
  const activations = (t) => Int8Array.from({ length: n }, (_, j) => ((next() ^ (j * 31 + t * 17)) & 255) - 128);
  const a = activations(0), sx = Float32Array.from({ length: ng }, () => f(1e-2 * (1 + (next() % 1000))));
  const want = (a, sx) => Array.from({ length: rows }, (_, i) => {
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
  const prepare = (at, a, sx) => { I.set(a, at); F.set(sx, (at + n) / 4); k.interleave(at, at + n, n); };
  const against = (got, expected, rowsOf = rows) => {
    for (let i = 0; i < rowsOf; i++) if (!Object.is(got(i), expected[i])) return `row ${i}: ${got(i)} not ${expected[i]}`;
    return null;
  };

  // ---- interleave(): the bytes of a block in four planes and the sums after the scales
  await check("interleave", () => { I.set(a, xq); F.set(sx, xs / 4); k.interleave(xq, xs, n); }, () => {
    I.set(a, xq); F.set(sx, xs / 4); k.interleave(xq, xs, n);
    let bad = 0;
    for (let at = 0; at < n; at++) if (I[xq + at] !== a[(at & ~63) + 4 * (at & 15) + ((at >> 4) & 3)]) bad++;
    for (let g = 0; g < ng; g++) if (N[xs / 4 + ng + g] !== -a.subarray(g * 32, g * 32 + 32).reduce((s, v) => s + v, 0)) bad++;
    return bad ? `${bad} of ${n} bytes and ${ng} sums wrong` : null;
  });

  // ---- T273: the converter's readers of the two ternary types of a GGUF (24 blocks each, high in the memory)
  const blocksOf = 24, raw = 7 << 20, wide = raw + 4096;
  const scaleBits = Array.from({ length: blocksOf }, () => ((1 + next() % 29) << 10) | (next() % 1024));  // normal, positive
  const scaleOf = (bits) => f(2 ** ((bits >> 10) - 15) * (1 + (bits & 1023) / 1024));
  for (const [name, size, scaleAt, valueOf] of [
    // PQ2_0: value j of a block is the j-th two bits of its 32 bytes (after the scale), less one
    ["widen_pq2_0", 34, 0, (block, j) => ((block[2 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1],
    // PTQ1_0: digit n of byte m in base 3, the first the most significant (the byte is ceil(256 v / 243))
    ["widen_ptq1_0", 28, 26, (block, j) => {
      const [byte, digit, digits] = j < 80 ? [j % 16, j >> 4, 5] : j < 120 ? [16 + (j - 80) % 8, (j - 80) >> 3, 5] : [24 + (j - 120) % 2, (j - 120) >> 1, 4];
      let left = block[byte], found = 0;
      for (let d = 0; d <= digit; d++) { left *= 3; found = left >> 8; left &= 255; }
      return found - 1;
    }],
  ]) {
    const data = Uint8Array.from({ length: blocksOf * size }, (_, at) => at < 256 ? at : next() & 255);
    scaleBits.forEach((bits, b) => { data[b * size + scaleAt] = bits & 255; data[b * size + scaleAt + 1] = bits >> 8; });
    const call = () => { U.set(data, raw); k[name](wide, raw, blocksOf); };
    await check(name, call, () => {
      F.fill(-7, wide / 4, wide / 4 + blocksOf * 128);
      call();
      for (let b = 0; b < blocksOf; b++) {
        const block = data.subarray(b * size, b * size + size), d = scaleOf(scaleBits[b]);
        for (let j = 0; j < 128; j++) {
          const expected = f(valueOf(block, j) * d), got = F[wide / 4 + b * 128 + j];
          if (!Object.is(got, expected)) return `block ${b}, value ${j}: ${got} not ${expected}`;
        }
      }
      return null;
    });
  }

  // ---- the ternary matrix products, to the bit
  const expected = want(a, sx);
  const product = (name, call) => check(name, call, () => {
    prepare(xq, a, sx);
    F.fill(-7, out / 4, out / 4 + rows);
    call();
    return against((i) => F[out / 4 + i], expected);
  });
  prepare(xq, a, sx);
  await product("matmul_t2", () => k.matmul_t2(out, xq, xs, w, ws, n, 0, rows, 3));
  if (r) {
    await product("matmul_t2r", () => r.matmul_t2r(out, xq, xs, w, ws, n, 0, rows, 3));
    // the tile: five tokens a frame apart (a four and one), each its own activations
    const perToken = Array.from({ length: tokens }, (_, t) => { const at = activations(t + 1); const sc = Float32Array.from({ length: ng }, () => f(1e-2 * (1 + (next() % 1000)))); return { at, sc, expected: want(at, sc) }; });
    const setUp = () => perToken.forEach(({ at, sc }, t) => prepare(frames + t * frame, at, sc));
    await check("matmul_t2r_tile", () => r.matmul_t2r_tile(outs, frames, frames + n, w, ws, n, 0, rows, tokens, 1024, frame, 3), () => {
      setUp();
      F.fill(-7, outs / 4, outs / 4 + tokens * 256);
      r.matmul_t2r_tile(outs, frames, frames + n, w, ws, n, 0, rows, tokens, 1024, frame, 3);
      for (let t = 0; t < tokens; t++) {
        const wrong = against((i) => F[(outs + t * 1024) / 4 + i], perToken[t].expected);
        if (wrong) return `token ${t} ${wrong}`;
      }
      return null;
    });
  }

  // ---- ternary_x (the converter's): the bytes and scales of a matrix of ternary values
  {
    const count = 40, x = extra + 65536, packed = x + count * 512, scales = packed + count * 32;
    for (let g = 0; g < count; g++) {
      const d = f(0.01 * (1 + (g % 7)));
      for (let j = 0; j < 128; j++) F[x / 4 + g * 128 + j] = ((next() % 3) - 1) * d;
    }
    const run = () => { U.fill(0xAA, packed, packed + count * 32); return k.ternary_x(packed, scales, x, count * 128); };
    await check("ternary_x", () => k.ternary_x(packed, scales, x, count * 128), () => {
      const refused = run();
      let bad = refused ? 1 : 0;
      for (let g = 0; g < count; g++) for (let b = 0; b < 32; b++) {
        let wanted = 0;
        for (let p = 0; p < 4; p++) wanted |= (Math.sign(F[x / 4 + g * 128 + 4 * b + p]) + 1) << (2 * p);
        if (U[packed + g * 32 + b] !== wanted) bad++;
      }
      return bad ? `${bad} bytes wrong` : null;
    });
  }

  // ---- rotate and unrotate (T237; the review of T233): the same float32 arithmetic written out here, to the bit, with the signs
  // as forward.js hands them over (a sign times 1 / sqrt(block): rotate multiplies by them, unrotate by their size and
  // flips the result's sign bit by theirs)
  {
    const at = extra + 2097152, x = at, signs = at + 16384, out = at + 32768;
    const butterflies = (v, from, block) => {
      for (let half = 1; half < block; half *= 2) {
        for (let i = 0; i < block; i += 2 * half) {
          for (let j = 0; j < half; j++) {
            const p = v[from + i + j], q = v[from + i + half + j];
            v[from + i + j] = f(p + q);
            v[from + i + half + j] = f(p - q);
          }
        }
      }
    };
    for (const [block, count] of [[1024, 2], [128, 3]]) {
      const len = block * count, scale = f(1 / Math.sqrt(block));
      const input = Float32Array.from({ length: len }, () => f((next() % 20001) / 10000 - 1));
      const given = Float32Array.from({ length: len }, () => f((next() & 1 ? 1 : -1) * scale));
      const rotatedOf = () => {
        const o = Float32Array.from(input, (v, i) => f(v * given[i]));
        for (let b = 0; b < len; b += block) butterflies(o, b, block);
        return o;
      };
      const unrotatedOf = () => {
        const o = Float32Array.from(input, (v, i) => f(v * Math.abs(given[i])));
        for (let b = 0; b < len; b += block) butterflies(o, b, block);
        return o.map((v, i) => (given[i] < 0 ? -v : v));
      };
      for (const [name, kernel, wanted] of [["rotate", k.rotate, rotatedOf], ["unrotate", k.unrotate, unrotatedOf]]) {
        const call = () => { F.set(input, x / 4); F.set(given, signs / 4); kernel(out, x, signs, len, block); };
        await check(`${name} (${count} blocks of ${block})`, call, () => {
          F.fill(-7, out / 4, out / 4 + len + 4);
          call();
          const want = new Uint32Array(wanted().buffer), got = new Uint32Array(memory.buffer, out, len);
          let bad = 0;
          for (let i = 0; i < len; i++) if (got[i] !== want[i]) bad++;
          for (let i = len; i < len + 4; i++) if (F[out / 4 + i] !== -7) bad++;
          return bad ? `${bad} of ${len} values wrong` : null;
        });
      }
    }
  }

  // ---- geglu (T261: the gate of a Gemma 3's FFN, gelu(h1) * h3). Its exp is the kernels' own (vexp, four at a time),
  // which is not written out here: it is held to the engine's gelu of the same values (a bias of -0, which changes no
  // value and no zero's sign) times h3 in float32, to the bit, and to the formula in float64 within a millionth (where
  // the kernels' exp is not clamped). 259
  // values: 64 fours and a rest of three, small and large, both zeros, past exp's clamps on either side
  {
    const at = extra + 3145728, count = 259, h1 = at, h3 = at + 2048, zeros = at + 4096, plain = at + 6144, out = at + 8192;
    const edges = [0, -0, 1e-30, -1e-30, 0.5, -0.5, 3, -3, 6, -6, 9.5, -9.5, 12, -12, 44, -44, 88, -88, 200, -200, 3e38, -3e38];
    const first = Float32Array.from({ length: count }, (_, j) => (j < edges.length ? edges[j] : f(((next() % 24001) - 12000) / 1000)));
    const other = Float32Array.from({ length: count }, () => f(((next() % 8001) - 4000) / 1000));
    const call = () => { F.set(first, h1 / 4); F.set(other, h3 / 4); k.geglu(out, h1, h3, count); };
    await check("geglu", call, () => {
      F.fill(-7, out / 4, out / 4 + count + 4);
      call();
      F.fill(-0, zeros / 4, zeros / 4 + count);
      k.gelu(plain, h1, zeros, count);
      const got = new Uint32Array(memory.buffer, out, count);
      let bad = 0;
      for (let j = 0; j < count; j++) {
        const want = f(F[plain / 4 + j] * other[j]);
        if (got[j] !== new Uint32Array(Float32Array.of(want).buffer)[0]) bad++;
        const x = first[j], exact = (x / (1 + Math.exp(-2 * 0.7978845608028654 * (x + 0.044715 * x * x * x)))) * other[j];
        // (but past where the kernels' exp is clamped: -3e38 over 1 + exp(88) is a number, and over infinity it is 0)
        if (Math.abs(x) < 1e6 && !(Math.abs(F[out / 4 + j] - exact) <= 1e-6 * Math.max(1, Math.abs(exact)))) bad++;
      }
      for (let j = count; j < count + 4; j++) if (F[out / 4 + j] !== -7) bad++;
      return bad ? `${bad} of ${count} values wrong` : null;
    });
  }

  // ---- geglu_erf (T342: the gate of a Spark-X2.5's FFN, the exact GELU of h1 times h3). Four at a time is held to the
  // kernel one value at a time (its scalar tail) to the bit, and to the same formula (erf by Abramowitz and Stegun's
  // 7.1.26, llama2_numpy's gelu_erf()) in float64 within a millionth. The same 259 values as geglu's
  {
    const at = extra + 3145728 + 16384, count = 259, h1 = at, h3 = at + 2048, out = at + 4096, one = at + 6144;
    const edges = [0, -0, 1e-30, -1e-30, 0.5, -0.5, 3, -3, 6, -6, 9.5, -9.5, 12, -12, 13, -13, 44, -44, 88, -88, 200, -200, 3e38, -3e38];
    const first = Float32Array.from({ length: count }, (_, j) => (j < edges.length ? edges[j] : f(((next() % 24001) - 12000) / 1000)));
    const other = Float32Array.from({ length: count }, () => f(((next() % 8001) - 4000) / 1000));
    const call = () => { F.set(first, h1 / 4); F.set(other, h3 / 4); k.geglu_erf(out, h1, h3, count); };
    const formula = (x) => {
      const z = Math.min(Math.abs(x) * 0.7071067811865476, 9), t = 1 / (1 + 0.3275911 * z);
      const half = 0.5 * t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z);
      return x * (x >= 0 ? 1 - half : half);
    };
    await check("geglu_erf", call, () => {
      F.fill(-7, out / 4, out / 4 + count + 4);
      call();
      const got = new Uint32Array(memory.buffer, out, count), alone = new Uint32Array(memory.buffer, one, 1);
      let bad = 0;
      for (let j = 0; j < count; j++) {
        k.geglu_erf(one, h1 + 4 * j, h3 + 4 * j, 1);
        if (got[j] !== alone[0]) bad++;
        const x = first[j], exact = formula(x) * other[j];
        if (Math.abs(x) < 1e6 && !(Math.abs(F[out / 4 + j] - exact) <= 1e-6 * Math.max(1, Math.abs(exact)))) bad++;
      }
      for (let j = count; j < count + 4; j++) if (F[out / 4 + j] !== -7) bad++;
      return bad ? `${bad} of ${count} values wrong` : null;
    });
  }

  // ---- what every model runs, as a control: quantize_x (8 bits and 7 with a bias), matmul_q8 and matmul_q8r against float64 sums
  {
    const x = extra + 262144, q = x + n * 4 + 64, s = q + n + 64;
    const v = Float32Array.from({ length: n }, () => ((next() % 2001) - 1000) / 250);
    F.set(v, x / 4);
    for (const [name, bias] of [["quantize_x", 0], ["quantize_x (bias 64)", 64]]) {
      await check(name, () => k.quantize_x(q, s, x, n, bias), () => {
        k.quantize_x(q, s, x, n, bias);
        let bad = 0;
        for (let g = 0; g < ng; g++) {
          let top = 0;
          for (let j = 0; j < 32; j++) top = Math.max(top, Math.abs(v[g * 32 + j]));
          const scale = f(top / (bias ? 63 : 127));
          if (F[s / 4 + g] !== scale) bad++;
          for (let j = 0; j < 32; j++) {
            const t = f(v[g * 32 + j] * f(1 / scale));
            let rounded = Math.round(t);
            if (Math.abs(t % 1) === 0.5) rounded = 2 * Math.round(t / 2);  // half to even
            if (I[q + g * 32 + j] !== rounded + bias) bad++;
          }
        }
        return bad ? `${bad} values wrong` : null;
      });
    }
    // an int8 matrix of 64 rows: values -127..127, scales; the activations quantized without a bias (and with one for q8r)
    const m = extra + 524288, ms = m + rows * n + 64, mc = ms + rows * ng * 4 + 64, o = mc + rows * ng * 4 + 64;
    for (let i = 0; i < rows * n; i++) I[m + i] = (next() % 255) - 127;
    for (let g = 0; g < rows * ng; g++) F[ms / 4 + g] = f(1e-3 * (1 + (next() % 100)));
    const exact = (bias) => {  // float64 sums of what the kernels' integers and scales say
      const outRows = [];
      for (let i = 0; i < rows; i++) {
        let sum = 0;
        for (let g = 0; g < ng; g++) {
          let dot = 0;
          for (let j = g * 32; j < g * 32 + 32; j++) dot += I[m + i * n + j] * (I[q + j] - bias);
          sum += dot * F[s / 4 + g] * F[ms / 4 + i * ng + g];
        }
        outRows.push(sum);
      }
      return outRows;
    };
    const close = (got, wanted) => {
      const most = wanted.reduce((a, b) => Math.max(a, Math.abs(b)), 0);
      for (let i = 0; i < rows; i++) if (!(Math.abs(got(i) - wanted[i]) <= 3e-6 * most)) return `row ${i}: ${got(i)} not ${wanted[i]}`;
      return null;
    };
    k.quantize_x(q, s, x, n, 0);
    await check("matmul_q8", () => k.matmul_q8(o, q, s, m, ms, n, 0, rows), () => {
      F.fill(-7, o / 4, o / 4 + rows);
      k.matmul_q8(o, q, s, m, ms, n, 0, rows);
      return close((i) => F[o / 4 + i], exact(0));
    });
    if (r) {
      k.quantize_x(q, s, x, n, 64);
      for (let g = 0; g < rows * ng; g++) {  // -64 times the sum of each group's weights
        let sum = 0;
        for (let j = 0; j < 32; j++) sum += I[m + g * 32 + j];
        N[mc / 4 + g] = -64 * sum;
      }
      await check("matmul_q8r", () => r.matmul_q8r(o, q, s, m, ms, mc, n, 0, rows), () => {
        F.fill(-7, o / 4, o / 4 + rows);
        r.matmul_q8r(o, q, s, m, ms, mc, n, 0, rows);
        return close((i) => F[o / 4 + i], exact(64));
      });
    }

    // ---- int6 (what a Safari holds a large model in): quantize6_x's bytes and scales, and matmul_q6 on weights packed as pack6
    // does (24 bytes a group: the low nibbles of values j and j + 16 in byte j, the top bits of k, k + 8, k + 16, k + 24 in byte 16 + k)
    const sixOf = (value, inverse) => {
      const t = f(value * inverse);
      let rounded = Math.round(t);
      if (Math.abs(t % 1) === 0.5) rounded = 2 * Math.round(t / 2);
      return Math.max(-32, Math.min(31, rounded));
    };
    const packSix = (six) => {  // 32 values of -32..31 -> 24 bytes
      const b = six.map((value) => value & 63), out6 = new Uint8Array(24);
      for (let j = 0; j < 16; j++) out6[j] = (b[j] & 15) | ((b[j + 16] & 15) << 4);
      for (let kk = 0; kk < 8; kk++) out6[16 + kk] = (b[kk] >> 4) | ((b[kk + 8] >> 4) << 2) | ((b[kk + 16] >> 4) << 4) | ((b[kk + 24] >> 4) << 6);
      return out6;
    };
    const out6 = s + 8192;  // 24 bytes a group, then the scales after them
    await check("quantize6_x", () => k.quantize6_x(out6, out6 + ng * 24, x, n), () => {
      k.quantize6_x(out6, out6 + ng * 24, x, n);
      let bad = 0;
      for (let g = 0; g < ng; g++) {
        let top = 0;
        for (let j = 0; j < 32; j++) top = Math.max(top, Math.abs(v[g * 32 + j]));
        const scale = f(top / 31), inverse = scale > 0 ? f(1 / scale) : 0;
        if (F[(out6 + ng * 24) / 4 + g] !== f(scale * 0.25)) bad++;
        const packed = packSix(Array.from({ length: 32 }, (_, j) => sixOf(v[g * 32 + j], inverse)));
        for (let b = 0; b < 24; b++) if (U[out6 + g * 24 + b] !== packed[b]) bad++;
      }
      return bad ? `${bad} bytes and scales wrong` : null;
    });
    const m6 = extra + 786432, m6s = m6 + rows * ng * 24 + 64, m6c = m6s + rows * ng * 4 + 64, o6 = m6c + rows * ng * 4 + 64;
    const six = Array.from({ length: rows * ng }, () => Array.from({ length: 32 }, () => (next() % 64) - 32));
    six.forEach((group, g) => U.set(packSix(group), m6 + g * 24));
    for (let g = 0; g < rows * ng; g++) F[m6s / 4 + g] = f(1e-3 * (1 + (next() % 100)));
    const exactSix = (bias) => Array.from({ length: rows }, (_, i) => {
      let sum = 0;
      for (let g = 0; g < ng; g++) {
        let dot = 0;
        for (let j = 0; j < 32; j++) dot += 4 * six[i * ng + g][j] * (I[q + g * 32 + j] - bias);
        sum += dot * F[s / 4 + g] * F[m6s / 4 + i * ng + g];
      }
      return sum;
    });
    k.quantize_x(q, s, x, n, 0);
    await check("matmul_q6", () => k.matmul_q6(o6, q, s, m6, m6s, n, 0, rows), () => {
      F.fill(-7, o6 / 4, o6 / 4 + rows);
      k.matmul_q6(o6, q, s, m6, m6s, n, 0, rows);
      return close((i) => F[o6 / 4 + i], exactSix(0));
    });
    if (r) {
      k.quantize_x(q, s, x, n, 64);
      k.six_sums(m6c, m6, rows * ng);
      await check("matmul_q6r", () => r.matmul_q6r(o6, q, s, m6, m6s, m6c, n, 0, rows), () => {
        F.fill(-7, o6 / 4, o6 / 4 + rows);
        r.matmul_q6r(o6, q, s, m6, m6s, m6c, n, 0, rows);
        return close((i) => F[o6 / 4 + i], exactSix(64));
      });
    }
  }
  return { relaxed: Boolean(r), results };
}

// runs in the page, on a 64-bit memory (what a model past 4 GiB runs on: the 64-bit builds, their addresses BigInt, as
// jobs.js's addressed() passes them): each kernel's address arguments sit low or above 4 GiB in every combination, and
// the result must be the all-low one. The V8 of Node 24 (13.6) on arm64 read v128.load32_splat and load32_lane at the low
// 32 bits of such an address, which loses the weights' scales of the ternary kernels and the int8 tile's; Chromium 148
// does not. An engine that cannot make a memory this size (WebKit) is skipped.
async function inPageHigh({ plain, relaxed, addresses }) {
  const bytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const HIGH = 4 * 2 ** 30 + 2 * 65536;
  const pages = BigInt(Math.ceil((HIGH + 8 * 2 ** 20) / 65536));
  let memory;
  try {
    memory = new WebAssembly.Memory({ initial: pages, address: "i64" });
  } catch (error) {
    try { memory = new WebAssembly.Memory({ initial: pages, index: "i64" }); } catch (second) { return { skipped: `no 64-bit memory (${error.message})` }; }
  }
  const wrap = (exports) => {
    const out = { ...exports };
    for (const [name, at] of Object.entries(addresses)) {
      const kernel = exports[name];
      if (kernel) out[name] = (...args) => { for (const i of at) args[i] = BigInt(args[i]); return kernel(...args); };
    }
    return out;
  };
  let k, r = null;
  try {
    k = wrap(new WebAssembly.Instance(new WebAssembly.Module(bytes(plain)), { env: { memory } }).exports);
  } catch (error) {
    return { skipped: `the 64-bit kernels do not instantiate (${error.message})` };
  }
  try { r = wrap(new WebAssembly.Instance(new WebAssembly.Module(bytes(relaxed)), { env: { memory } }).exports); } catch (error) { r = null; }
  const U = new Uint8Array(memory.buffer), I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer);
  const LOW = 65536, UP = HIGH + 65536, SPREAD = [0, 8192, 16384, 24576, 40960, 49152, 57344];
  // every case: the names of its address arguments, setup(p) writing the inputs at p, call(p), and the result it must give
  // (null: whatever it gives with every argument low)
  const ternary = (p, tokens) => {
    U.fill(0, p.w, p.w + 32);
    for (let j = 0; j < 128; j++) U[p.w + (j >> 2)] |= 2 << (2 * (j & 3));  // 128 weights of +1
    F[p.ws / 4] = 1;
    for (let t = 0; t < tokens; t++) {
      for (let j = 0; j < 128; j++) I[p.xq + t * 1024 + j] = 3;
      for (let g = 0; g < 4; g++) F[(p.xs + t * 1024) / 4 + g] = 1;
      k.interleave(p.xq + t * 1024, p.xs + t * 1024, 128);
    }
  };
  const int8 = (p, tokens, bias) => {
    for (let i = 0; i < 128 * tokens; i++) I[p.w + i] = 2;
    for (let g = 0; g < 4 * tokens; g++) { F[p.ws / 4 + g] = 1; if (p.wc !== undefined) N[p.wc / 4 + g] = -64 * 2 * 32; }
    for (let t = 0; t < 4; t++) {
      for (let j = 0; j < 128; j++) I[p.xq + t * 1024 + j] = 3 + bias;
      for (let g = 0; g < 4; g++) F[(p.xs + t * 1024) / 4 + g] = 1;
    }
  };
  // rotate and unrotate (T237; the review of T233): 128 values of a block of 128, x and signs up or down, the result of the
  // first value (a sum of all of them) the all-low one
  const turn = (p) => {
    for (let j = 0; j < 128; j++) { F[p.x / 4 + j] = (j % 7) - 3.25; F[p.signs / 4 + j] = (j % 3 ? 1 : -1) * 0.08838835; }
  };
  const cases = [
    { name: "rotate", args: ["out", "x", "signs"], want: null, setup: turn, call: (p) => k.rotate(p.out, p.x, p.signs, 128, 128) },
    { name: "unrotate", args: ["out", "x", "signs"], want: null, setup: turn, call: (p) => k.unrotate(p.out, p.x, p.signs, 128, 128) },
    { name: "matmul_q8", args: ["out", "xq", "xs", "w", "ws"], want: 768, setup: (p) => int8(p, 1, 0), call: (p) => k.matmul_q8(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1) },
    { name: "matmul_q6", args: ["out", "xq", "xs", "w", "ws"], want: null,
      setup: (p) => {  // 4 groups of 32 values of 2, packed as pack6 does (the low nibbles in bytes 0..15, the top bits 0 in 16..23)
        U.fill(0, p.w, p.w + 96);
        for (let g = 0; g < 4; g++) U.fill(0x22, p.w + g * 24, p.w + g * 24 + 16);
        for (let g = 0; g < 4; g++) { F[p.ws / 4 + g] = 1; F[p.xs / 4 + g] = 1; }
        for (let j = 0; j < 128; j++) I[p.xq + j] = 3;
      },
      call: (p) => k.matmul_q6(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1) },
    { name: "matmul_t2", args: ["out", "xq", "xs", "w", "ws"], want: 384, setup: (p) => ternary(p, 1), call: (p) => k.matmul_t2(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1, 3) },
    r && { name: "matmul_q8r", args: ["out", "xq", "xs", "w", "ws", "wc"], want: 768, setup: (p) => int8(p, 1, 64),
      call: (p) => r.matmul_q8r(p.out, p.xq, p.xs, p.w, p.ws, p.wc, 128, 0, 1) },
    r && { name: "matmul_q8r_tile", args: ["out", "xq", "xs", "w", "ws", "wc"], want: 768, setup: (p) => int8(p, 4, 64),
      call: (p) => r.matmul_q8r_tile(p.out, p.xq, p.xs, p.w, p.ws, p.wc, 128, 0, 4, 4, 64, 1024) },
    r && { name: "matmul_t2r", args: ["out", "xq", "xs", "w", "ws"], want: 384, setup: (p) => ternary(p, 1), call: (p) => r.matmul_t2r(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1, 3) },
    r && { name: "matmul_t2r_tile", args: ["out", "xq", "xs", "w", "ws"], want: 384, setup: (p) => ternary(p, 4),
      call: (p) => r.matmul_t2r_tile(p.out, p.xq, p.xs, p.w, p.ws, 128, 0, 1, 4, 64, 1024, 3) },
  ].filter(Boolean);
  const results = [];
  for (const c of cases) {
    let reference = c.want, wrong = null;
    const bad = [];
    for (let mask = 0; mask < 2 ** c.args.length; mask++) {
      const p = Object.fromEntries(c.args.map((arg, i) => [arg, (mask >> i & 1 ? UP : LOW) + SPREAD[i]]));
      c.setup(p);
      F.fill(-7, p.out / 4, p.out / 4 + 40);
      c.call(p);
      const got = F[p.out / 4];
      if (mask === 0 && reference === null) reference = got;
      if (got !== reference) bad.push(mask);
    }
    // the arguments that break it alone: only that one above 4 GiB
    const alone = c.args.filter((_, i) => bad.includes(1 << i));
    if (bad.length) wrong = `${bad.length} of ${2 ** c.args.length} placements wrong; above 4 GiB alone: ${alone.join(", ") || "only together"}`;
    results.push({ name: c.name, wrong, calls: 2 ** c.args.length });
  }
  return { relaxed: Boolean(r), results };
}

let failed = false;
for (const engine of engines.length ? engines : ["chromium"]) {
  const browser = await (channels[engine] ? playwright.chromium.launch({ channel: channels[engine] }) : playwright[engine].launch());
  try {
    const page = await browser.newPage();
    await page.goto("about:blank");
    const { relaxed, results } = await page.evaluate(inPage, modules);
    console.log(`kernels-in-browser: ${engine} ${browser.version()} on ${os.cpus()[0].model} (${process.arch}), ${relaxed ? "with" : "without"} relaxed SIMD`);
    for (const { name, wrong, calls } of results) {
      console.log(`kernels-in-browser:   ${wrong ? "WRONG" : "ok"}: ${name}${wrong ? ` ${wrong}` : ` (${calls} calls)`}`);
      if (wrong) failed = true;
    }
    // a runner that cannot give a page 4 GiB of address space (the commit limit of a small Windows machine) makes the page
    // crash or throw: that is the runner, and is said, not failed
    const high = await page.evaluate(inPageHigh, wideModules).catch((error) => ({ skipped: `the page could not run it (${String(error.message).split("\n")[0]})` }));
    if (high.skipped) console.log(`kernels-in-browser:   skipped: above 4 GiB: ${high.skipped}`);
    for (const { name, wrong, calls } of high.results ?? []) {
      console.log(`kernels-in-browser:   ${wrong ? "WRONG" : "ok"}: above 4 GiB, ${name}${wrong ? ` ${wrong}` : ` (${calls} placements)`}`);
      if (wrong) failed = true;
    }
    // T363.4: the kernels of llama.cpp's blocks. A line a kernel: its ten types are named where one is wrong
    if (!blocks) {
      console.log("kernels-in-browser:   skipped: the kernels of llama.cpp's blocks: this site serves none");
    } else {
      const of = await page.evaluate(inPageBlocks, blocks);
      console.log(`kernels-in-browser:   the blocks' kernels: this engine's relaxed dot product ${!of.relaxed ? "is not there" : of.signed ? "multiplies signed bytes by signed bytes (matmul_sbw_tile is checked)" : "does not multiply signed bytes by signed bytes (matmul_sbw_tile is not called)"}`);
      for (const [where, list] of [["", of.results], ["above 4 GiB, ", of.high.results ?? []]]) {
        const kernels = [...new Set(list.map(({ name }) => name.split(", ")[0]))];
        for (const kernel of kernels) {
          const mine = list.filter(({ name }) => name.split(", ")[0] === kernel), bad = mine.filter(({ wrong }) => wrong);
          for (const { name, wrong } of bad) console.log(`kernels-in-browser:   WRONG: ${where}${name} ${wrong}`);
          if (bad.length) failed = true;
          else console.log(`kernels-in-browser:   ok: ${where}${kernel} (${mine.length > 1 ? `${mine.length} types, ` : ""}${mine[0].calls} ${where ? "placements" : "calls"}${mine.length > 1 ? " each" : ""})`);
        }
      }
      if (of.high.skipped) console.log(`kernels-in-browser:   skipped: above 4 GiB, the blocks' kernels: ${of.high.skipped}`);
    }
  } finally {
    await closed(() => browser.close());  // (T141, T391.6: Windows' WebKit may never return from it)
  }
}
await leave(failed ? 1 : 0);
