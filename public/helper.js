// helper.js (T93, stage 2): a thread that takes chunks of rows of the matrix multiplications that forward.js hands
// out, on the same shared WebAssembly memory. It holds no weights of its own: they are in the shared memory. Runs as
// a module Web Worker in the page and as a worker_threads worker in Node (the tests).
//
// The coordinator (forward.js, in the model's worker) publishes a phase in the control area at the start of the
// memory; see CONTROL in forward.js. A helper sleeps on GEN with Atomics.wait, takes chunks from COUNTER until there
// are none, counts them in FINISHED, and sleeps again. It never spins.
const node = typeof process !== "undefined" && process.versions?.node;
const port = node ? (await import("node:worker_threads")).parentPort : self;
const listen = (handler) => (node ? port.on("message", handler) : (self.onmessage = (event) => handler(event.data)));

// the layout of the control area: the same numbers as in forward.js. WAKE + share is this helper's own word: the
// coordinator writes the generation there and wakes exactly the helpers it wants, always the same ones. (With one
// word for all, Atomics.notify wakes whoever has slept longest, so a different, cold helper took every phase.)
const GEN = 0, QUIT = 1, COUNTER = 2, FINISHED = 3, ACTIVE = 4, TOTAL = 5, WAKE = 256, JOBS = 512, JOB = 12;

listen(({ memory, plain, relaxed, share }) => {
  const imports = { env: { memory } };
  const k = new WebAssembly.Instance(plain, imports).exports;
  const q8r = relaxed ? new WebAssembly.Instance(relaxed, imports).exports.matmul_q8r : null;
  const ctl = new Int32Array(memory.buffer, 0, 1024);
  // one matmul over rows r0..r1 of job j; the kinds are forward.js's: 0 matmul_q8r, 1 matmul_q8, 2 matmul_f32
  const run = (at, r0, r1) => {
    const kind = ctl[at], n = ctl[at + 7];
    if (kind === 0) q8r(ctl[at + 1], ctl[at + 2], ctl[at + 3], ctl[at + 4], ctl[at + 5], ctl[at + 6], n, r0, r1);
    else if (kind === 1) k.matmul_q8(ctl[at + 1], ctl[at + 2], ctl[at + 3], ctl[at + 4], ctl[at + 5], n, r0, r1);
    else k.matmul_f32(ctl[at + 1], ctl[at + 2], ctl[at + 4], n, r0, r1);
  };
  const steal = () => {
    const total = ctl[TOTAL], count = ctl[JOBS];
    for (let c = Atomics.add(ctl, COUNTER, 1); c < total; c = Atomics.add(ctl, COUNTER, 1)) {
      let j = count - 1;
      while (ctl[JOBS + 1 + j * JOB + 10] > c) j--;
      const at = JOBS + 1 + j * JOB, size = ctl[at + 9], r0 = (c - ctl[at + 10]) * size;
      run(at, r0, Math.min(r0 + size, ctl[at + 8]));
      if (Atomics.add(ctl, FINISHED, 1) + 1 === total) Atomics.notify(ctl, FINISHED);
    }
  };
  // Warm the kernels up before anyone waits for this thread: a new thread runs them unoptimized at first, and the
  // search for the number of threads would take that for the speed of the count (T93). Tiny matmuls on scratch
  // space at the end of the control area, many times; what they compute is thrown away.
  const scratch = 3072, xq = scratch, xs = scratch + 64, w = scratch + 128, s = scratch + 192, c = scratch + 256, out = scratch + 320;
  for (let i = 0; i < 4000; i++) {
    k.matmul_q8(out, xq, xs, w, s, 32, 0, 1);
    k.matmul_f32(out, out + 64, w, 8, 0, 1);
    if (q8r) q8r(out, xq, xs, w, s, c, 32, 0, 1);
  }
  port.postMessage("ready");
  let gen = Atomics.load(ctl, WAKE + share);
  for (;;) {
    Atomics.wait(ctl, WAKE + share, gen);
    gen = Atomics.load(ctl, WAKE + share);
    if (Atomics.load(ctl, QUIT)) break;
    if (gen & 1) continue;  // the coordinator is rewriting the jobs
    Atomics.add(ctl, ACTIVE, 1);
    if (Atomics.load(ctl, GEN) === gen) steal();
    if (Atomics.sub(ctl, ACTIVE, 1) === 1) Atomics.notify(ctl, ACTIVE);
  }
  if (node) process.exit(0);
  else self.close();
});
