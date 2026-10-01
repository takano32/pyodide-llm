// threads-late-helper.mjs (T229's review): a software thread of tests/threads-check.mjs that stops late. It is
// public/helper.js, with one difference: of the chunk of a delta rule's phase (job kind 7, a hybrid model's Gated
// DeltaNet layer) it takes, it writes the first value head only, and ends without counting the chunk: what a thread
// that a browser stops half way through its work leaves behind (T120). The coordinator must give it up and run the
// phase again, to the numbers a single thread computes (the new state is written beside the old one: forward.js's flips).
import { parentPort } from "node:worker_threads";

const started = new Promise((resolve) => parentPort.once("message", resolve));
const J = await import(new URL("../public/jobs.js", import.meta.url));
const { memory, plain, relaxed, share, wide = false } = await started;
const imports = { env: { memory } };
const k = J.addressed(new WebAssembly.Instance(plain, imports).exports, wide);
const r = relaxed ? J.addressed(new WebAssembly.Instance(relaxed, imports).exports, wide) : null;
const ctl = new Int32Array(memory.buffer, 0, J.CONTROL_BYTES / 4);
const table = new Float64Array(memory.buffer, J.JOB_TABLE, J.BATCH * J.JOB);
const runRows = J.runner(k, r);
J.warmUp(k, r);
parentPort.postMessage("ready");
let gen = Atomics.load(ctl, J.WAKE + share);
for (;;) {
  Atomics.wait(ctl, J.WAKE + share, gen);
  gen = Atomics.load(ctl, J.WAKE + share);
  if (Atomics.load(ctl, J.QUIT)) break;
  if (gen & 1) continue;
  Atomics.add(ctl, J.ACTIVE, 1);
  if (Atomics.load(ctl, J.GEN) === gen) {
    const total = ctl[J.TOTAL], count = ctl[J.JOBS];
    for (let c = Atomics.add(ctl, J.COUNTER, 1); c < total; c = Atomics.add(ctl, J.COUNTER, 1)) {
      let j = count - 1;
      while (table[j * J.JOB + J.FIRST] > c) j--;
      const at = j * J.JOB, size = table[at + J.SIZE], r0 = (c - table[at + J.FIRST]) * size;
      if (table[at] === 7) {  // the delta rule: the first value head of the chunk is written, and this thread is gone
        runRows(table.subarray(at, at + J.SIZE), r0, r0 + 1);
        process.exit(0);
      }
      runRows(table.subarray(at, at + J.SIZE), r0, Math.min(r0 + size, table[at + J.ROWS]));
      if (Atomics.add(ctl, J.FINISHED, 1) + 1 === total) Atomics.notify(ctl, J.FINISHED);
    }
  }
  if (Atomics.sub(ctl, J.ACTIVE, 1) === 1) Atomics.notify(ctl, J.ACTIVE);
}
process.exit(0);
