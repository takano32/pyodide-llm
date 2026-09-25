// helper.js (T93, stage 2): a thread that takes chunks of rows of the matrix multiplications that forward.js hands
// out, on the same shared WebAssembly memory. It holds no weights of its own: they are in the shared memory. Runs as
// a module Web Worker in the page and as a worker_threads worker in Node (the tests).
//
// The coordinator (forward.js, in the model's worker) publishes a phase in the control area at the start of the
// memory (jobs.js has its layout). A helper sleeps on its own word with Atomics.wait, takes chunks from COUNTER until
// there are none, counts them in FINISHED, and sleeps again. It never spins.
const node = typeof process !== "undefined" && process.versions?.node;
const port = node ? (await import("node:worker_threads")).parentPort : self;
const listen = (handler) => (node ? port.on("message", handler) : (self.onmessage = (event) => handler(event.data)));

// what a job is and how its rows are run: jobs.js, the same file forward.js reads, from the same deployment
const { GEN, QUIT, COUNTER, FINISHED, ACTIVE, TOTAL, WAKE, JOBS, JOB, ROWS, SIZE, FIRST, CONTROL_BYTES, runner, warmUp } =
  await import(new URL(`jobs.js${new URL(import.meta.url).search}`, import.meta.url));

listen(({ memory, plain, relaxed, share }) => {
  const imports = { env: { memory } };
  const k = new WebAssembly.Instance(plain, imports).exports;
  const q8r = relaxed ? new WebAssembly.Instance(relaxed, imports).exports.matmul_q8r : null;
  const ctl = new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4);
  const runRows = runner(k, q8r);
  const steal = () => {
    const total = ctl[TOTAL], count = ctl[JOBS];
    for (let c = Atomics.add(ctl, COUNTER, 1); c < total; c = Atomics.add(ctl, COUNTER, 1)) {
      let j = count - 1;
      while (ctl[JOBS + 1 + j * JOB + FIRST] > c) j--;
      const at = JOBS + 1 + j * JOB, size = ctl[at + SIZE], r0 = (c - ctl[at + FIRST]) * size;
      runRows(ctl.subarray(at, at + SIZE), r0, Math.min(r0 + size, ctl[at + ROWS]));
      if (Atomics.add(ctl, FINISHED, 1) + 1 === total) Atomics.notify(ctl, FINISHED);
    }
  };
  warmUp(k, q8r);  // before anyone waits for this thread
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
