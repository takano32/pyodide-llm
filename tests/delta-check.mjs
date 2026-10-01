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
import fs from "node:fs";

const root = new URL("../", import.meta.url).pathname;
const memory = new WebAssembly.Memory({ initial: 16 });
const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/simdkernel_plain.wasm`)), { env: { memory } }).exports;
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
console.log(`ok: delta_rule to the bit, convolve and gate within 4e-7 of JavaScript's (${checked} cases)`);
