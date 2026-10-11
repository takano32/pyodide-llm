// delta-check.mjs (T229): the three kernels of Qwen3.5's hybrid attention (kernels/kernel.ts: gate, convolve,
// delta_rule) against the same arithmetic written out here, as llama2_numpy.py does it (linear_attention(),
// delta_rule()). Node and the compiled kernels alone, under a second: the light suite runs it.
//   delta_rule: to the bit. Every number is a float32 product or sum in an order the kernel fixes (the rows of the
//     state in turn), which Math.fround follows here. Heads of 5, 6, 8 and 13 values (the fours and what is left after
//     them), one, two and three value heads to a key head, and ranges of heads that begin and end anywhere: the heads
//     outside a range, the state read and the token's q, k and v are left as they were.
//   convolve: the sum to the bit (the taps in turn, the oldest token first), then SiLU by the kernels' own exp
//     (Cephes's expf, a few ulp from Math.exp): within 4e-7 of JavaScript's, relative.
//   gate: x * sigmoid(g), the same way.
// And (the review of T229) against llama2_numpy.py's own functions on random numbers, from a native Python (PYTHON, else
// python3; NumPy is there where pytest is): delta_rule(), silu() and the sigmoid, which the NumPy engine runs and which
// the engine, transformers and the made-up models' float32 logits agree on, within 2e-5. The arithmetic above is written
// from the kernels, and a kernel and its copy could be wrong the same way; NumPy's is the definition.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { built, python as sources } from "./tree.mjs";

const root = new URL("../", import.meta.url).pathname;
const memory = new WebAssembly.Memory({ initial: 16 });
const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(built("simdkernel_plain.wasm"))), { env: { memory } }).exports;
const F = new Float32Array(memory.buffer), f = Math.fround;
let seed = 229;
const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
const random = () => f((next() % 20001) / 10000 - 1);  // -1 .. 1
const fill = (at, n, make = random) => { for (let i = 0; i < n; i++) F[at / 4 + i] = make(); };
const close = (got, want) => Math.abs(got - want) <= 4e-7 * Math.max(1, Math.abs(want));
let checked = 0;

// ---- delta_rule
for (const [kh, vh, kd, vd] of [[1, 1, 4, 4], [2, 2, 8, 6], [2, 4, 8, 5], [2, 6, 5, 13], [3, 3, 16, 8], [1, 3, 7, 12]]) {
  const state = 4096, heads = vh * kd * vd, next_ = state + heads * 4, c = next_ + heads * 4;
  const work = c + (2 * kh * kd + vh * vd) * 4, out = work + (2 * vh + vh * vd) * 4, end = out + vh * vd * 4;
  if (end > memory.buffer.byteLength) throw new Error("delta-check's memory is too small");
  for (const [h0, h1] of [[0, vh], [0, 1], [vh - 1, vh], [1, vh - 1], [1, 1]].filter(([a, b]) => a <= b && a >= 0)) {
    fill(state, heads);
    fill(next_, heads, () => -7);
    fill(c, 2 * kh * kd + vh * vd);
    fill(work, 2 * vh, () => f((next() % 10000) / 10000));  // beta and decay, 0 .. 1
    fill(work + 2 * vh * 4, vh * vd, () => -7);
    fill(out, vh * vd, () => -7);
    const before = F.slice(state / 4, end / 4);
    const at = (name) => ({ state: 0, next: heads, c: 2 * heads, work: 2 * heads + 2 * kh * kd + vh * vd,
      out: 2 * heads + 2 * kh * kd + vh * vd + 2 * vh + vh * vd })[name];
    const want = before.slice();
    for (let h = h0; h < h1; h++) {
      const key = Math.floor(h / (vh / kh));
      const q = (i) => before[at("c") + key * kd + i], key_ = (i) => before[at("c") + (kh + key) * kd + i];
      const v = (j) => before[at("c") + 2 * kh * kd + h * vd + j];
      const beta = before[at("work") + h], decay = before[at("work") + vh + h];
      const S = (i, j) => before[at("state") + (h * kd + i) * vd + j];
      for (let j = 0; j < vd; j++) {
        let kv = 0;
        for (let i = 0; i < kd; i++) kv = f(kv + f(key_(i) * f(S(i, j) * decay)));
        const delta = f(f(v(j) - kv) * beta);
        let read = 0;
        for (let i = 0; i < kd; i++) {
          const s = f(f(S(i, j) * decay) + f(key_(i) * delta));
          want[at("next") + (h * kd + i) * vd + j] = s;
          read = f(read + f(q(i) * s));
        }
        want[at("work") + 2 * vh + h * vd + j] = delta;
        want[at("out") + h * vd + j] = read;
      }
    }
    k.delta_rule(out, state, next_, c, work, kh, kd, vd, vh, h0, h1);
    const got = F.subarray(state / 4, end / 4);
    for (let i = 0; i < want.length; i++) {
      if (!Object.is(got[i], want[i])) {
        throw new Error(`delta_rule differs at float ${i} (${kh} key heads, ${vh} value heads of ${kd} by ${vd}, heads ${h0}..${h1}): ${got[i]} against ${want[i]}`);
      }
    }
    checked += 1;
  }
}

// ---- convolve and gate
const silu = (x) => x / (1 + Math.exp(-x));
for (const n of [1, 3, 4, 5, 8, 13, 64]) {
  for (const count of [1, 2, 4, 5]) {
    const taps = 4096, rows = taps + count * n * 4, out = rows + count * n * 4;
    fill(taps, count * n);
    fill(rows, count * n, () => f(4 * random()));
    // the ends of the kernels' exp: far below and above, where sigmoid is 0 and 1
    if (n >= 4 && count === 1) { F[taps / 4] = 1; F[rows / 4] = -200; F[taps / 4 + 1] = 1; F[rows / 4 + 1] = 200; }
    fill(out, n + 1, () => -7);
    k.convolve(out, taps, rows, n, count);
    for (let c = 0; c < n; c++) {
      let sum = f(F[taps / 4 + c] * F[rows / 4 + c]);
      for (let j = 1; j < count; j++) sum = f(sum + f(F[(taps + j * n * 4) / 4 + c] * F[(rows + j * n * 4) / 4 + c]));
      if (!close(F[out / 4 + c], silu(sum))) throw new Error(`convolve differs at channel ${c} of ${n}, ${count} taps: ${F[out / 4 + c]} against ${silu(sum)}`);
    }
    if (F[out / 4 + n] !== -7) throw new Error(`convolve wrote past its ${n} channels`);
    checked += 1;
  }
  const x = 4096, g = x + n * 4, out = g + n * 4;
  fill(x, n);
  fill(g, n, () => f(6 * random()));
  if (n >= 4) { F[g / 4] = -200; F[g / 4 + 1] = 200; }
  fill(out, n + 1, () => -7);
  k.gate(out, x, g, n);
  for (let j = 0; j < n; j++) {
    const want = F[x / 4 + j] / (1 + Math.exp(-F[g / 4 + j]));
    if (!close(F[out / 4 + j], want)) throw new Error(`gate differs at ${j} of ${n}: ${F[out / 4 + j]} against ${want}`);
  }
  if (F[out / 4 + n] !== -7) throw new Error(`gate wrote past its ${n} values`);
  k.gate(x, x, g, n);  // in place, as forward.js calls it
  for (let j = 0; j < n; j++) if (!Object.is(F[x / 4 + j], F[out / 4 + j])) throw new Error(`gate in place differs at ${j} of ${n}`);
  checked += 1;
}

// ---- against NumPy's own functions
const python = `
import json, sys
import numpy as np
sys.path.insert(0, ${JSON.stringify(sources())})
import llama2_numpy as L

rng = np.random.default_rng(229)
f32 = lambda *shape: rng.uniform(-1, 1, shape).astype(np.float32)
unit = lambda *shape: rng.uniform(0, 1, shape).astype(np.float32)
cases = {"delta": [], "convolve": [], "gate": []}
for kh, vh, kd, vd in [(2, 6, 8, 12), (1, 3, 7, 12), (4, 4, 16, 8), (2, 4, 128, 128)]:
    state, q, k, v = f32(vh, kd, vd), f32(kh, kd), f32(kh, kd), f32(vh, vd)
    beta, decay = unit(vh), unit(vh)
    after, ratio = state.copy(), vh // kh
    read = L.delta_rule(after, np.repeat(q, ratio, axis=0), np.repeat(k, ratio, axis=0), v, beta, decay)
    cases["delta"].append(dict(kh=kh, vh=vh, kd=kd, vd=vd, state=state.ravel().tolist(), q=q.ravel().tolist(), k=k.ravel().tolist(),
                               v=v.ravel().tolist(), beta=beta.tolist(), decay=decay.tolist(), read=read.ravel().tolist(),
                               after=after.ravel().tolist()))
for n, count in [(7, 4), (64, 4), (1024, 4), (6144, 4), (13, 3), (40, 2)]:
    taps, before, mixed = f32(count, n), f32(count - 1, n), f32(n)
    # llama2_numpy.linear_attention(): the taps over the history and this token, then SiLU
    got = L.silu((taps[:-1] * before).sum(axis=0) + taps[-1] * mixed)
    cases["convolve"].append(dict(n=n, count=count, taps=taps.ravel().tolist(), rows=np.concatenate([before, mixed[None]]).ravel().tolist(),
                                  want=got.tolist()))
for n in (5, 64, 2048):
    x, g = f32(n), (6 * f32(n)).astype(np.float32)
    cases["gate"].append(dict(n=n, x=x.tolist(), g=g.tolist(), want=(x / (1.0 + np.exp(-g))).astype(np.float32).tolist()))
print(json.dumps(cases))
`;
const numpy = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c", python], { cwd: root, maxBuffer: 1 << 28 }).toString());
const near = (got, want) => Math.abs(got - want) <= 2e-5 * Math.max(1, Math.abs(want));
const wider = new WebAssembly.Memory({ initial: 64 });
const kw = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(built("simdkernel_plain.wasm"))), { env: { memory: wider } }).exports;
const G = new Float32Array(wider.buffer);
const putW = (at, values) => { for (let i = 0; i < values.length; i++) G[at / 4 + i] = values[i]; };
const getW = (at, n) => Array.from(G.subarray(at / 4, at / 4 + n));
let againstNumpy = 0;
for (const c of numpy.delta) {
  const heads = c.vh * c.kd * c.vd, qkv = (2 * c.kh * c.kd + c.vh * c.vd) * 4, state = 4096, next_ = state + heads * 4, cc = next_ + heads * 4;
  const work = cc + qkv, out = work + (2 * c.vh + c.vh * c.vd) * 4;
  putW(state, c.state);
  putW(cc, [...c.q, ...c.k, ...c.v]);
  putW(work, [...c.beta, ...c.decay]);
  kw.delta_rule(out, state, next_, cc, work, c.kh, c.kd, c.vd, c.vh, 0, c.vh);
  const read = getW(out, c.vh * c.vd), after = getW(next_, heads);
  read.forEach((got, i) => { if (!near(got, c.read[i])) throw new Error(`delta_rule's output differs from NumPy's at ${i} (${c.kh} key heads, ${c.vh} value heads of ${c.kd} by ${c.vd}): ${got} against ${c.read[i]}`); });
  after.forEach((got, i) => { if (!near(got, c.after[i])) throw new Error(`delta_rule's new state differs from NumPy's at ${i} (${c.kh} key heads, ${c.vh} value heads of ${c.kd} by ${c.vd}): ${got} against ${c.after[i]}`); });
  againstNumpy++;
}
for (const c of numpy.convolve) {
  const taps = 4096, rows = taps + c.count * c.n * 4, out = rows + c.count * c.n * 4;
  putW(taps, c.taps);
  putW(rows, c.rows);
  kw.convolve(out, taps, rows, c.n, c.count);
  getW(out, c.n).forEach((got, i) => { if (!near(got, c.want[i])) throw new Error(`convolve differs from NumPy's silu(taps * rows) at ${i} of ${c.n} (${c.count} taps): ${got} against ${c.want[i]}`); });
  againstNumpy++;
}
for (const c of numpy.gate) {
  const x = 4096, g = x + c.n * 4, out = g + c.n * 4;
  putW(x, c.x);
  putW(g, c.g);
  kw.gate(out, x, g, c.n);
  getW(out, c.n).forEach((got, i) => { if (!near(got, c.want[i])) throw new Error(`gate differs from NumPy's x / (1 + exp(-g)) at ${i} of ${c.n}: ${got} against ${c.want[i]}`); });
  againstNumpy++;
}
console.log(`ok: delta_rule to the bit, convolve and gate within 4e-7 of JavaScript's (${checked} cases), and all three within 2e-5 of NumPy's own (${againstNumpy} cases)`);
