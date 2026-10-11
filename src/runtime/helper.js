// helper.js (T93, stage 2): a thread that takes chunks of rows of the matrix multiplications that forward.js hands
// out, on the same shared WebAssembly memory. It holds no weights of its own: they are in the shared memory. Runs as
// a module Web Worker in the page and as a worker_threads worker in Node (the tests).
//
// The coordinator (forward.js, in the model's worker) publishes a phase in the control area at the start of the
// memory (jobs.js has its layout). A helper sleeps on its own word with Atomics.wait, takes chunks from COUNTER until
// there are none, counts them in FINISHED, and sleeps again. It never spins.
const node = typeof process !== "undefined" && process.versions?.node;
const port = node ? (await import("node:worker_threads")).parentPort : self;
// The one message, { memory, plain, relaxed, share }, is claimed before anything else is awaited: a module worker's
// port opens at the module's first await, and a message that arrives before onmessage is set is lost (Node queues
// it, the browser does not: the page hung on it once, T109).
const started = new Promise((resolve) => (node ? port.once("message", resolve) : (self.onmessage = (event) => resolve(event.data))));

// what a job is and how its rows are run: jobs.js, the same file forward.js reads, from the same deployment
const { GEN, QUIT, COUNTER, FINISHED, ACTIVE, TOTAL, WAKE, JOBS, JOB, JOB_TABLE, BATCH, ROWS, SIZE, FIRST, CONTROL_BYTES, addressed, runner, warmUp } =
  await import(new URL(`jobs.js${new URL(import.meta.url).search}`, import.meta.url));

const { memory, plain, relaxed, share, wide = false } = await started;
const imports = { env: { memory } };
const k = addressed(new WebAssembly.Instance(plain, imports).exports, wide);
const r = relaxed ? addressed(new WebAssembly.Instance(relaxed, imports).exports, wide) : null;
const ctl = new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4);
const table = new Float64Array(memory.buffer, JOB_TABLE, BATCH * JOB);  // the jobs, in float64 (jobs.js)
const runRows = runner(k, r);
const steal = () => {
  const total = ctl[TOTAL], count = ctl[JOBS];
  for (let c = Atomics.add(ctl, COUNTER, 1); c < total; c = Atomics.add(ctl, COUNTER, 1)) {
    let j = count - 1;
    while (table[j * JOB + FIRST] > c) j--;
    const at = j * JOB, size = table[at + SIZE], r0 = (c - table[at + FIRST]) * size;
    runRows(table.subarray(at, at + SIZE), r0, Math.min(r0 + size, table[at + ROWS]));
    if (Atomics.add(ctl, FINISHED, 1) + 1 === total) Atomics.notify(ctl, FINISHED);
  }
};
warmUp(k, r);  // before anyone waits for this thread
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
