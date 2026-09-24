// One thread of the prototype (T93): the coordinator (share 0) runs the forward pass and hands out rows of the
// int8 matmuls; a helper only computes its share of the rows. All threads use the same WebAssembly kernels on
// the same shared memory, so the weights are there once. Synchronization is Atomics.wait / notify, never a spin.
import fs from "node:fs";
import { parentPort, workerData } from "node:worker_threads";

const { memory, share, threads, kernelsDir, layout, mode } = workerData;
const imports = { env: { memory } };
const k = (await WebAssembly.instantiate(fs.readFileSync(`${kernelsDir}/simdkernel_shared.wasm`), imports)).instance.exports;
const r = (await WebAssembly.instantiate(fs.readFileSync(`${kernelsDir}/simdkernel_relaxed_shared.wasm`), imports)).instance.exports;
const ctl = new Int32Array(memory.buffer, 0, 1024);   // [0] generation, [1] quit, [64 + share] done, [256..] jobs
const GEN = 0, QUIT = 1, DONE = 64, JOBS = 256, JOB = 8;

// every job is one matmul_q8r(out, xq, xs, wq, ws, wc, n, rows); this share takes its slice of the rows
function runShare(s) {
  const count = ctl[JOBS];
  for (let j = 0; j < count; j++) {
    const at = JOBS + 1 + j * JOB;
    const rows = ctl[at + 7], r0 = Math.floor((rows * s) / threads), r1 = Math.floor((rows * (s + 1)) / threads);
    if (r1 > r0) r.matmul_q8r(ctl[at], ctl[at + 1], ctl[at + 2], ctl[at + 3], ctl[at + 4], ctl[at + 5], ctl[at + 6], r0, r1);
  }
}

if (share > 0) {
  let gen = 0;
  for (;;) {
    Atomics.wait(ctl, GEN, gen);
    gen = Atomics.load(ctl, GEN);
    if (Atomics.load(ctl, QUIT)) break;
    runShare(share);
    Atomics.store(ctl, DONE + share, gen);
    Atomics.notify(ctl, DONE + share);
  }
  parentPort.postMessage("bye");
} else {
  const L = layout, { dim, hidden, layers, heads, kvHeads, headSize, vocab } = L.config;
  const kvDim = kvHeads * headSize;
  let gen = 0;
  // one round: publish the jobs, compute share 0, wait for the others
  function phase(jobs) {
    if (threads === 1) {
      for (const j of jobs) r.matmul_q8r(j[0], j[1], j[2], j[3], j[4], j[5], j[6], 0, j[7]);
      return;
    }
    ctl[JOBS] = jobs.length;
    jobs.forEach((job, i) => job.forEach((v, n) => { ctl[JOBS + 1 + i * JOB + n] = v; }));
    gen += 1;
    Atomics.store(ctl, GEN, gen);
    Atomics.notify(ctl, GEN);
    runShare(0);
    for (let s = 1; s < threads; s++) {
      for (let seen = Atomics.load(ctl, DONE + s); seen !== gen; seen = Atomics.load(ctl, DONE + s)) Atomics.wait(ctl, DONE + s, seen);
    }
  }
  const job = (out, w, l, n, rows) => [out, L.xq, L.xs, w.values + l * rows * n, w.scales + l * rows * (n / 32) * 4,
    w.corrections + l * rows * (n / 32) * 4, n, rows];
  const matmul = (jobs) => (mode === "all" ? phase(jobs) : jobs.forEach((j) => r.matmul_q8r(...j.slice(0, 7), 0, j[7])));
  const f32 = new Float32Array(memory.buffer);
  const i8 = new Int8Array(memory.buffer);

  function forward(token, pos) {
    // the embedding row, int8 times the scale of its group: as Llama.embedding() does it
    const e = L.embedding, row = token * dim;
    for (let i = 0; i < dim; i++) f32[L.x / 4 + i] = i8[e.values + row + i] * f32[e.scales / 4 + (row + i) / 32 | 0];
    const cos = L.cos + pos * (headSize / 2) * 4, sin = L.sin + pos * (headSize / 2) * 4;
    for (let l = 0; l < layers; l++) {
      const keys = L.keys + l * L.capacity * kvDim * 4, values = L.values + l * L.capacity * kvDim * 4;
      const kp = keys + pos * kvDim * 4, vp = values + pos * kvDim * 4;
      k.rmsnorm(L.xb, L.x, L.attNorm + l * dim * 4, dim);
      k.quantize_x(L.xq, L.xs, L.xb, dim, 64);
      matmul([job(L.q, L.wq, l, dim, dim), job(kp, L.wk, l, dim, kvDim), job(vp, L.wv, l, dim, kvDim)]);
      k.rope(L.q, cos, sin, heads, headSize, headSize);
      k.rope(kp, cos, sin, kvHeads, headSize, headSize);
      k.attention(L.xb, L.q, keys, values, L.att, pos, heads, kvHeads, headSize);
      k.quantize_x(L.xq, L.xs, L.xb, dim, 64);
      matmul([job(L.xb2, L.wo, l, dim, dim)]);
      k.add_inplace(L.x, L.xb2, dim);
      k.rmsnorm(L.xb, L.x, L.ffnNorm + l * dim * 4, dim);
      k.quantize_x(L.xq, L.xs, L.xb, dim, 64);
      matmul([job(L.hb, L.w1, l, dim, hidden), job(L.hb2, L.w3, l, dim, hidden)]);
      k.swiglu(L.hb, L.hb, L.hb2, hidden);
      k.quantize_x(L.xq, L.xs, L.hb, hidden, 64);
      matmul([job(L.xb2, L.w2, l, hidden, dim)]);
      k.add_inplace(L.x, L.xb2, dim);
    }
    k.rmsnorm(L.xb, L.x, L.finalNorm, dim);
    k.quantize_x(L.xq, L.xs, L.xb, dim, 64);
    phase([job(L.logits, L.wcls, 0, dim, vocab)]);  // both modes split the classifier
    return k.argmax(L.logits, vocab);
  }

  parentPort.on("message", ({ prompt, positions }) => {
    const tokens = [];
    let token = L.bos;
    const began = performance.now();
    for (let pos = 0; pos < positions; pos++) {
      const next = forward(token, pos);
      token = pos < prompt.length ? prompt[pos] : next;
      if (pos >= prompt.length) tokens.push(next);
    }
    const seconds = (performance.now() - began) / 1000;
    parentPort.postMessage({ tokens, seconds });
  });
  parentPort.postMessage("ready");
}
