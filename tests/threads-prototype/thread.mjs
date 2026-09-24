// One thread of the prototype (T93): the coordinator (share 0) runs the forward pass and hands out rows of the
// int8 matmuls; a helper only computes rows. All threads use the same WebAssembly kernels on the same shared
// memory, so the weights are there once. Synchronization is Atomics.wait / notify, never a spin.
//
// Two ways of handing out rows (schedule):
//   equal  every thread takes rows / threads of every matmul, and everyone waits for the slowest (a little core)
//   steal  the rows are cut into chunks, and whoever is free takes the next one from an atomic counter, so a slow
//          core simply takes fewer chunks. The chunk size is derived: rows / (threads × chunksPerThread)
import fs from "node:fs";
import { Worker, parentPort, workerData } from "node:worker_threads";

const { memory, share, kernelsDir, layout, mode, schedule = "equal", chunksPerThread = 4, auto = false } = workerData;
// auto: the coordinator finds the number of threads itself (below); otherwise it is fixed
let threads = auto ? 1 : workerData.threads;
const imports = { env: { memory } };
const k = (await WebAssembly.instantiate(fs.readFileSync(`${kernelsDir}/simdkernel_shared.wasm`), imports)).instance.exports;
const r = (await WebAssembly.instantiate(fs.readFileSync(`${kernelsDir}/simdkernel_relaxed_shared.wasm`), imports)).instance.exports;
const ctl = new Int32Array(memory.buffer, 0, 1024);
// GEN: even when a phase is published, odd while the coordinator rewrites the jobs. COUNTER: the next chunk.
// FINISHED: chunks done. ACTIVE: helpers awake inside a phase. TOTAL: chunks of this phase. DONE + s: equal's flags.
const GEN = 0, QUIT = 1, COUNTER = 2, FINISHED = 3, ACTIVE = 4, TOTAL = 5, DONE = 64, JOBS = 256, JOB = 10;

const matmulRows = (at, r0, r1) => r.matmul_q8r(ctl[at], ctl[at + 1], ctl[at + 2], ctl[at + 3], ctl[at + 4], ctl[at + 5], ctl[at + 6], r0, r1);

// equal: this share's slice of every job
function runShare(s) {
  const count = ctl[JOBS];
  for (let j = 0; j < count; j++) {
    const at = JOBS + 1 + j * JOB;
    const rows = ctl[at + 7], r0 = Math.floor((rows * s) / threads), r1 = Math.floor((rows * (s + 1)) / threads);
    if (r1 > r0) matmulRows(at, r0, r1);
  }
}

// steal: take chunks until there are none; job j owns chunks first..first + ceil(rows / size) - 1
function steal() {
  const total = ctl[TOTAL], count = ctl[JOBS];
  for (let c = Atomics.add(ctl, COUNTER, 1); c < total; c = Atomics.add(ctl, COUNTER, 1)) {
    let j = count - 1;
    while (ctl[JOBS + 1 + j * JOB + 9] > c) j--;
    const at = JOBS + 1 + j * JOB, size = ctl[at + 8], r0 = (c - ctl[at + 9]) * size;
    matmulRows(at, r0, Math.min(r0 + size, ctl[at + 7]));
    if (Atomics.add(ctl, FINISHED, 1) + 1 === total) Atomics.notify(ctl, FINISHED);
  }
}

if (share > 0) {
  let gen = 0;
  for (;;) {
    Atomics.wait(ctl, GEN, gen);
    gen = Atomics.load(ctl, GEN);
    if (Atomics.load(ctl, QUIT)) break;
    if (schedule === "equal") {
      runShare(share);
      Atomics.store(ctl, DONE + share, gen);
      Atomics.notify(ctl, DONE + share);
      continue;
    }
    if (gen & 1) continue;  // the coordinator is rewriting the jobs: wait for the next publication
    Atomics.add(ctl, ACTIVE, 1);
    if (Atomics.load(ctl, GEN) === gen) steal();  // the jobs are this generation's
    if (Atomics.sub(ctl, ACTIVE, 1) === 1) Atomics.notify(ctl, ACTIVE);
  }
  parentPort.postMessage("bye");
} else {
  const L = layout, { dim, hidden, layers, heads, kvHeads, headSize, vocab } = L.config;
  const kvDim = kvHeads * headSize;
  let gen = 0;
  const waitUntil = (index, done) => {
    for (let seen = Atomics.load(ctl, index); !done(seen); seen = Atomics.load(ctl, index)) Atomics.wait(ctl, index, seen);
  };
  // one round: publish the jobs, work too, wait for the others
  function phase(jobs) {
    if (threads === 1) {
      for (const j of jobs) r.matmul_q8r(j[0], j[1], j[2], j[3], j[4], j[5], j[6], 0, j[7]);
      return;
    }
    if (schedule === "equal") {
      ctl[JOBS] = jobs.length;
      jobs.forEach((job, i) => job.forEach((v, n) => { ctl[JOBS + 1 + i * JOB + n] = v; }));
      gen += 1;
      Atomics.store(ctl, GEN, gen);
      Atomics.notify(ctl, GEN);
      runShare(0);
      for (let s = 1; s < threads; s++) waitUntil(DONE + s, (seen) => seen === gen);
      return;
    }
    // steal: close the previous phase (odd), let every helper that is still awake leave, then rewrite
    Atomics.store(ctl, GEN, gen + 1);
    waitUntil(ACTIVE, (seen) => seen === 0);
    let total = 0;
    ctl[JOBS] = jobs.length;
    jobs.forEach((job, i) => {
      const at = JOBS + 1 + i * JOB, rows = job[7];
      const size = Math.max(1, Math.ceil(rows / (threads * chunksPerThread)));
      job.forEach((v, n) => { ctl[at + n] = v; });
      ctl[at + 8] = size;
      ctl[at + 9] = total;
      total += Math.ceil(rows / size);
    });
    ctl[TOTAL] = total;
    Atomics.store(ctl, COUNTER, 0);
    Atomics.store(ctl, FINISHED, 0);
    gen += 2;
    Atomics.store(ctl, GEN, gen);
    // wake no more helpers than there are chunks left for them
    Atomics.notify(ctl, GEN, Math.min(threads - 1, total - 1));
    steal();
    waitUntil(FINISHED, (seen) => seen === total);
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

  // auto: helpers are started when the search first needs them, and never more than it asks for. There is no
  // upper bound written down anywhere: the search doubles until a doubling is slower, then stops.
  const helpers = [];
  async function haveThreads(n) {
    while (helpers.length < n - 1) {
      const helper = new Worker(new URL(import.meta.url), { workerData: { ...workerData, share: helpers.length + 1, auto: false, threads: 0 } });
      await new Promise((resolve) => helper.once("online", resolve));
      helpers.push(helper);
    }
  }
  // One comparison runs the current best and the doubled count in blocks, best-candidate-candidate-best, so that
  // the growing cost of later positions falls on both sides alike. Switching every token instead biased it against
  // more threads: while one thread works, the helpers' cores idle and clock down. The first token of every block
  // is not timed (the switch, and a new helper's unoptimized kernels). A doubling must be faster by more than the
  // noise of a run here (5-10%) to be taken.
  const BLOCK = 4, BETTER = 0.95;
  const needed = 4 * (BLOCK + 1);
  const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
  let remembered = 0;

  parentPort.on("message", async ({ prompt, positions, remember }) => {
    const tokens = [], log = [];
    let token = L.bos, pos = 0;
    const step = () => {
      const next = forward(token, pos);
      token = pos < prompt.length ? prompt[pos] : next;
      if (pos >= prompt.length) tokens.push(next);
      pos += 1;
    };
    const began = performance.now();
    if (auto && remember && remembered) {
      threads = remembered;
      await haveThreads(threads);
    } else if (auto) {
      let best = 1;
      for (let candidate = 2; pos + needed <= positions; candidate *= 2) {
        await haveThreads(candidate);
        const times = { [best]: [], [candidate]: [] };
        for (const count of [best, candidate, candidate, best]) {
          threads = count;
          step();
          for (let i = 0; i < BLOCK; i++) {
            const t = performance.now();
            step();
            times[count].push(performance.now() - t);
          }
        }
        const [a, b] = [median(times[best]), median(times[candidate])];
        log.push(`${best}: ${a.toFixed(2)} ms against ${candidate}: ${b.toFixed(2)} ms`);
        if (b >= a * BETTER) break;
        best = candidate;
      }
      threads = remembered = best;
    }
    while (pos < positions) step();
    const seconds = (performance.now() - began) / 1000;
    parentPort.postMessage({ tokens, seconds, threads, log });
  });
  parentPort.postMessage("ready");
}
