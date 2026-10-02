// kernels-in-browser.mjs (T230's review): the kernels' arithmetic in the engine of each browser, called often enough that
// the engine compiles them again. tests/ternary-check.mjs and tests/smoke.mjs run the kernels in V8 (Node) alone, and a
// browser's wasm engine is another compiler: JavaScriptCore on x86-64 (Playwright's WebKit on Linux, Safari on an Intel
// Mac) wrote nonsense with a ternary model because its optimizing tier, which a function reaches after about 2000 calls
// and a test of a few calls never does, folded the three shuffles of interleave() wrongly. Nothing else saw it: every
// end-to-end run of a model asks for an answer to appear, not for it to be right.
//
// Each engine gets the compiled kernels (public/*.wasm of this checkout after `make kernels`, or those of a deployed site
// with --site) in a blank page, and each kernel is called in batches (1, 400, 4000 and 30000 calls, a pause after each for
// the background compile to land) with its result held to the same arithmetic written out in JavaScript after every batch.
// The relaxed kernels are checked where the engine has relaxed SIMD (the others are not run there, as in the page).
//
//   node tests/kernels-in-browser.mjs [--site https://takano32.github.io/pyodide-llm/] [chromium firefox webkit chrome msedge]
// Exit 1 where a result is wrong. Needs playwright-core's browsers installed (the workflows' own step).
import fs from "node:fs";
import os from "node:os";
import * as playwright from "playwright-core";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const siteAt = args.indexOf("--site");
const site = siteAt >= 0 ? args[siteAt + 1].replace(/\/?$/, "/") : null;
const engines = args.filter((arg, i) => !arg.startsWith("--") && args[i - 1] !== "--site");
const channels = { chrome: "chrome", msedge: "msedge" };

async function wasm(name) {
  if (!site) return fs.readFileSync(`${root}public/${name}.wasm`).toString("base64");
  const response = await fetch(new URL(`${name}.wasm`, site));
  if (!response.ok) throw new Error(`${name}.wasm: ${response.status}`);
  return Buffer.from(await response.arrayBuffer()).toString("base64");
}
const modules = { plain: await wasm("simdkernel_plain"), relaxed: await wasm("simdkernel_relaxed_plain") };

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
  } finally {
    await browser.close().catch(() => undefined);
  }
}
process.exit(failed ? 1 : 0);
