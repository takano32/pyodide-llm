// T.163: a reading thread of the CPU section's ceilings (sections.ts): each time the section says go, it takes chunks
// of the made-up model's weights in turn with the section (NEXT, until TOTAL) and reads each with the loop of
// kernels/ceilings.ts, on the section's shared memory, then counts itself done. It sleeps between (Atomics.wait) and
// never spins. The control words are sections.ts's; the first message is claimed before anything is awaited (T.109).
const GO = 0, DONE = 1, COUNT = 2, TOTAL = 3, STOP = 4, NEXT = 5;

// T.368.7: what the section starts a thread with (sections.ts's started()): the ceilings' module and the section's
// memory, the control words, where the weights are and how they are cut, and which thread this is
type Started = { module: WebAssembly.Module; memory: WebAssembly.Memory; control: SharedArrayBuffer; base: number; chunk: number; chunks: number; share: number };
self.onmessage = ({ data: { module, memory, control, base, chunk, chunks, share } }: MessageEvent<Started>) => {
  self.onmessage = null;
  const read = new WebAssembly.Instance(module, { env: { memory } }).exports.read as (at: number, bytes: number, passes: number) => number;
  const ctl = new Int32Array(control);
  let seen = Atomics.load(ctl, GO);
  postMessage("ready");
  for (;;) {
    Atomics.wait(ctl, GO, seen);
    seen = Atomics.load(ctl, GO);
    if (Atomics.load(ctl, STOP)) break;
    const count = Atomics.load(ctl, COUNT);
    if (share >= count) continue;  // not one of this run's threads
    const total = Atomics.load(ctl, TOTAL);
    for (let k; (k = Atomics.add(ctl, NEXT, 1)) < total;) read(base + (k % chunks) * chunk, chunk, 1);
    if (Atomics.add(ctl, DONE, 1) + 1 === count - 1) Atomics.notify(ctl, DONE);
  }
};
