// forward/control.js (T378.2): the control area of one engine: the words at the start of a shared memory (jobs.js has
// their layout) through which this thread and the others meet: the software threads (threads.js, helper.js) and the
// GPU's worker (gpuside.js, gpu.js). It belongs to neither of them: createForward() makes it and hands it to both.
// A module of src/runtime/forward.js; it reads jobs.js by a static import.

import { CONTROL_BYTES } from "../jobs.ts";

/** T378.2: the control area of memory, and the wait on a word of it. ctl is null on a memory that is not shared (no
 * other thread is there to meet). stalledMs: how long a wait may see no progress before it gives up (T120). */
export function controlArea({ memory, sharedMemory, stalledMs }) {
  // the control area of a shared memory: the helpers' words, and the GPU's (T135: with or without helpers)
  const ctl = sharedMemory ? new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4) : null;
  // the memory is kept from model to model (T96), and the control area with it: what the last engine's phases left
  // there (a helper counted in ACTIVE when it was ended, a generation) would hold this one's first phase for ever
  ctl?.fill(0);
  // stopThreads() runs on this thread too, never in the middle of a phase: a wait here ends when the helpers have done
  // their part. A helper the browser itself stopped (iOS may end a worker for its memory) never counts the chunk it
  // took: T120, the wait gives up when its count has not moved for stalledMs, and says false. (A check of a flag that
  // stopThreads() raised and lowered again stood here and could never be seen, the review of T96 found.)
  // A wait that took far longer than it asked for means this thread did not run either (a frozen tab, a phone that
  // suspended the page): the helpers were stopped with it, and the time from before does not count (the review of
  // T120: a stop of 10 s gave a thread up now and then as the page came back)
  // alive (T135): the word whose change counts as progress, where it is not the one waited on (the GPU's worker counts
  // one up while it works, and answers only at the end)
  const waitUntil = (index, done, alive = index) => {
    const tick = Math.min(1000, stalledMs);
    let moved = performance.now(), last = moved, beat = Atomics.load(ctl, alive);
    for (let seen = Atomics.load(ctl, index); !done(seen);) {
      Atomics.wait(ctl, index, seen, tick);
      const now = Atomics.load(ctl, index), beatNow = Atomics.load(ctl, alive), at = performance.now();
      if (at - last > 2 * tick) moved = at;
      last = at;
      if (now !== seen || beatNow !== beat) [seen, beat, moved] = [now, beatNow, at];
      else if (at - moved > stalledMs) return false;
    }
    return true;
  };
  return { ctl, waitUntil };
}
