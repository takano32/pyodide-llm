// forward/threads.js (T349): the software threads of one engine: a phase's rows shared out among them on the control
// area of a shared memory (jobs.js has its layout; helper.js is a thread), a thread that stopped given up (T120), and
// the search for how many of them to use (T93's stage 2b, T199, T223, T239).
// A module of public/forward.js, asked for with its ?v=<build>; it reads jobs.js and its neighbours the same way.

const { CONTROL_BYTES, GEN, QUIT, COUNTER, FINISHED, ACTIVE, TOTAL, WAKE, JOBS, JOB, JOB_TABLE, BATCH, ROWS, COUNT, SIZE,
  FIRST } = await import(new URL(`../jobs.js${new URL(import.meta.url).search}`, import.meta.url));
const { lowerMedian, BETTER } = await import(new URL(`choice.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- the helper threads (stage 2): the control area at the start of a shared memory (jobs.js has its layout)
// how many chunks per thread the rows of a matmul are cut into: whoever is free takes the next one, so a slow core
// (a little core of a big.LITTLE phone) simply takes fewer. 2 to 16 measured the same (T93); fewer does not steal.
const CHUNKS_PER_THREAD = 4;

/** T349: the phases and the software threads of one engine (createForward(), which hands in what they read of it):
 * phase(jobs) shares a phase's rows out among the threads in use and waits for them (T93, T120), and the search finds
 * how many to use (T199, T223, T239, T240). Returns what the rest of the engine calls of them, and its variables that
 * the rest reads or sets, each through a getter and a setter (the variables themselves stay this function's own, as
 * they were createForward()'s: phase() reads `threads` as it did). gpuGettingReady(): whether the GPU's side is still
 * getting ready, which is made after this. */
export function softwareThreads({ memory, kernels, plan, spawn, stalledMs, wide, sharedMemory, runRows, gpuGettingReady }) {
  // ---- the matrix multiplications, in phases: the matmuls that read the same input (q, k and v; w1 and w3) go
  // out together. The input is quantized here, once; the rows are computed here and, with helper threads, by
  // whoever takes them. Every row is computed whole by one thread with the same kernel, so the numbers are the same
  // with any number of threads.
  //
  // A job is what jobs.js says: [kind, eight arguments, rows, count, out stride, a stride, b stride].
  const shared = sharedMemory && spawn;
  // the control area of a shared memory: the helpers' words, and the GPU's (T135: with or without helpers)
  const ctl = sharedMemory ? new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4) : null;
  const table = shared ? new Float64Array(memory.buffer, JOB_TABLE, BATCH * JOB) : null;  // the jobs (jobs.js)
  // the memory is kept from model to model (T96), and the control area with it: what the last engine's phases left
  // there (a helper counted in ACTIVE when it was ended, a generation) would hold this one's first phase for ever
  ctl?.fill(0);
  const helpers = [];
  let threads = 1, gen = 0;
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
  // T120: every helper goes, and this engine keeps to one thread from here on. The phase is then run again here:
  // each chunk writes only its own rows, from inputs that no phase changes while it runs, so what the helpers did
  // before they stopped is written over with the same numbers
  let lost = false;
  function giveUp() {
    lost = true;
    console.warn("forward.js: a software thread stopped in the middle of its work; this model goes on with one thread");
    stopHelpers();
    search = null;
    chosen = 1;
  }
  function phase(jobs) {
    const alone = () => {
      for (const job of jobs) runRows(job, 0, job[ROWS]);
    };
    if (threads <= 1) return alone();
    // close the previous phase (odd), let every helper still awake leave it, then rewrite the jobs
    Atomics.store(ctl, GEN, gen + 1);
    if (!waitUntil(ACTIVE, (seen) => seen === 0)) {
      giveUp();
      return alone();
    }
    let total = 0;
    ctl[JOBS] = jobs.length;
    jobs.forEach((job, i) => {
      const at = i * JOB, rows = job[ROWS];
      const quad = job[COUNT] > 1 ? 4 : 1;  // T159: a prompt's chunks in fours of rows, the tiles of matmul_q8r_tile
      const size = quad * Math.ceil(rows / (threads * CHUNKS_PER_THREAD * quad));
      table.set(job, at);
      table[at + SIZE] = size;
      table[at + FIRST] = total;
      total += Math.ceil(rows / size);
    });
    ctl[TOTAL] = total;
    Atomics.store(ctl, COUNTER, 0);
    Atomics.store(ctl, FINISHED, 0);
    gen += 2;
    Atomics.store(ctl, GEN, gen);
    // wake helpers 1.. by name, no more than there are chunks for them and no more than this many threads
    for (let h = 1, wake = Math.min(threads - 1, total - 1); h <= wake; h++) {
      Atomics.store(ctl, WAKE + h, gen);
      Atomics.notify(ctl, WAKE + h, 1);
    }
    for (let c = Atomics.add(ctl, COUNTER, 1); c < total; c = Atomics.add(ctl, COUNTER, 1)) {
      let j = jobs.length - 1;
      while (table[j * JOB + FIRST] > c) j--;
      const at = j * JOB, size = table[at + SIZE], r0 = (c - table[at + FIRST]) * size;
      runRows(jobs[j], r0, Math.min(r0 + size, jobs[j][ROWS]));
      Atomics.add(ctl, FINISHED, 1);
    }
    if (!waitUntil(FINISHED, (seen) => seen === total)) {
      giveUp();
      alone();
    }
  }

  // ---- the number of threads (stage 2b): found by measuring, never written down. The search starts from a hint
  // (navigator.hardwareConcurrency, which counts the little cores of a big.LITTLE phone too) and compares the best
  // count so far with half of it and, if half is not faster, with twice as many; it goes on in that direction while
  // the other is faster by more than the noise of a run, and stops at the first that is not (T239: on the way down, at
  // the second in a row that is not: a quarter is compared where half was not faster). Only the tokens that
  // make logits are timed (a prompt's tokens skip the classifier). One comparison runs the two counts in blocks,
  // best-candidate-candidate-best, so that the growing cost of later positions falls on both alike, and drops the
  // first token of every block (the switch). Helpers that a count needs are started in the background; until they
  // are ready the tokens run on the best count and are not timed.
  // T199: a count's time is the lower median of its 8 (two blocks of 4). What else runs (a collection of the garbage,
  // another tab, the page on its way to the background) slows a block, never speeds one up, so one block slowed whole
  // leaves the other block's 4 below it and the verdict stands. The upper median it was took that block's time: twice
  // in CI 2 threads 1.24 to 1.34 times as fast as 1 lost to it (T190's review).
  // T223: a count is remembered only where a search timed it with nothing of the GPU's getting ready beside it (its
  // upload, its shaders checked against JavaScript and timed use the CPU and the memory: T148 does not time the CPU's
  // prompts then either). A search on the first texts goes on while the GPU gets ready (the first visit is not left on
  // the logical cores meanwhile), but its verdict is used only until the GPU is ready: the first generation after that
  // searches again from it, and that search's verdict is the one remembered. A count remembered from an earlier visit
  // is searched again the same way once a visit (the first generation with no GPU getting ready), not only every
  // recheck generations of one load: the owner's Android kept 4 threads for llm-jp-3 150M where 2 wrote 3.2 times as
  // fast (T223), and a visit seldom writes 8 answers. unchecked: the count in use is not such a search's verdict yet.
  // T239: half may be a dip with a faster count below it. The owner's PC (16 logical cores) stopped at 8 threads ("16 or
  // 8: 8, 8 or 4: 8") where 2 wrote faster, and the search of every later visit began from that 8 and ended on it. So a
  // count that half did not beat is compared with a quarter of it too (far), and the way down goes on by halves from a
  // quarter that is faster. A comparison more (20 tokens) where the best count is 4 or more and nothing below it is
  // faster; none more where it is 1 or 2. Not on the way up: a visit that remembers 2 would time 8 threads every time
  // (the owner's Android: 0.23 s), and no device's report has a dip above its count (TODO.md's T239 has the table).
  // T240: the search the count in use is owed (unchecked) does not wait for the next generation where the GPU is ready
  // inside one: it begins at the first token after that (forward() below), so that a long first answer is not written
  // to its end on a count timed beside the GPU's getting ready. Only where the page began a generation: /benchmark/
  // begins none and takes the count the model page remembers as it is (T190).
  const BLOCK = 4;
  let search = null, chosen = 0, generations = 0, recheckEvery = 0, onChosen = null, onCompared = null, unchecked = false;
  // whether a search from the count in use may begin now (none under way, nothing of the GPU's getting ready beside it)
  const mayRecheck = () => !search && chosen && recheckEvery && !gpuGettingReady();
  const searchLog = [];  // every comparison: the counts, their times in ms per token, and the verdict
  function beginSearch(from) {
    search = { best: Math.max(1, from), direction: from > 1 ? "down" : "up", moved: false, far: false, candidate: 0, times: null, step: 0, waiting: false, whileGpu: false };
    nextCandidate();
  }
  function nextCandidate() {
    if (lost) return finish();
    const { best, direction, far } = search;
    const candidate = direction === "down" ? Math.floor(best / (far ? 4 : 2)) : best * 2;
    if (candidate < 1) return passed();
    search.candidate = candidate;
    search.times = { [best]: [], [candidate]: [] };
    search.step = 0;
    if (helpers.length < candidate - 1) {
      search.waiting = true;
      ensureHelpers(candidate).then((complete) => {
        if (!search) return;
        if (complete) search.waiting = false;
        else finish();
      }, () => finish());
    }
  }
  // the candidate was not faster than the best, or there is no count there: the next one, or the end (T239: a quarter
  // after half; then twice as many, where the best is still the count the search began from)
  function passed() {
    if (search.direction === "down" && !search.far) search.far = true;
    else if (search.direction === "down" && !search.moved) search.direction = "up";
    else return finish();
    nextCandidate();
  }
  function finish() {
    chosen = lost ? 1 : search ? search.best : threads;
    threads = chosen;
    unchecked = !lost && Boolean(search?.whileGpu);  // T223: searched again once the GPU is ready, and remembered then
    search = null;
    // one thread after a give-up says nothing about the device: the page would start with it next time (the review of T120)
    if (!lost && !unchecked) onChosen?.(chosen);
  }
  // the count for the next token, and whether it is timed
  function countForToken() {
    if (lost) return [1, false];
    if (!search || search.waiting) return [search ? search.best : threads, false];
    const order = [search.best, search.candidate, search.candidate, search.best];
    const block = Math.floor(search.step / (BLOCK + 1)), inBlock = search.step % (BLOCK + 1);
    return [order[block], inBlock > 0];
  }
  function recordToken(count, milliseconds, timed) {
    if (!search || search.waiting) return;
    if (timed) {
      search.times[count].push(milliseconds);
      if (gpuGettingReady()) search.whileGpu = true;
    }
    search.step += 1;
    if (search.step < 4 * (BLOCK + 1)) return;
    const { best, candidate } = search;
    const bestMs = lowerMedian(search.times[best]), candidateMs = lowerMedian(search.times[candidate]);
    const faster = candidateMs < bestMs * BETTER;
    const { whileGpu } = search;
    searchLog.push({ best, candidate, times: search.times, faster, whileGpu });
    // T114: every verdict, so that a device's choice can be followed afterwards (the page writes it to the console)
    onCompared?.({ best, candidate, bestMs, candidateMs, faster, whileGpu, tokens: search.times[best].length + search.times[candidate].length });
    if (!faster) return passed();
    Object.assign(search, { best: candidate, moved: true, far: false });
    nextCandidate();
  }
  // the helpers in QUIT's hands: set, every one woken to see it, and each ended
  let stops = 0;  // stopHelpers() counts them: a helper whose start began before one is not kept
  function stopHelpers() {
    stops += 1;
    Atomics.store(ctl, QUIT, 1);
    for (let h = 1; h <= helpers.length; h++) {
      Atomics.add(ctl, WAKE + h, 2);
      Atomics.notify(ctl, WAKE + h);
    }
    // QUIT stays set until the next ensureHelpers(): a helper that wakes late must still see it
    helpers.splice(0).forEach((helper) => helper.terminate?.());
    threads = 1;
  }
  // Resolves to whether the n threads are there. A helper that becomes ready after this engine let its helpers go
  // (release(), stopThreads(), giveUp()) is ended at once: the review of T120 found such ones left alive when the
  // visitor chose another model while the search started more, and the next model's engine on the same memory
  // (T96) woke them, with the old engine's kernels (one threw holding a chunk: 10 s still, then one thread)
  async function ensureHelpers(n) {
    if (lost) return false;  // T120: none again after a helper stopped under this engine
    const since = stops;
    if (helpers.length < n - 1 && helpers.length === 0) Atomics.store(ctl, QUIT, 0);  // after stopThreads(): a fresh start
    while (helpers.length < n - 1) {
      const helper = await spawn({ memory, wide, plain: kernels.plain, relaxed: plan.int8 && plan.relaxed ? kernels.relaxed : null,
        share: helpers.length + 1 });
      if (lost || stops !== since) {
        helper.terminate?.();
        return false;
      }
      helpers.push(helper);
    }
    return true;
  }
  // T349: what the rest of the engine calls of the phases and the software threads, and reads and sets of their search
  return {
    shared, ctl, waitUntil, phase, mayRecheck, searchLog, beginSearch, countForToken, recordToken, stopHelpers,
    ensureHelpers,
    get threads() { return threads; }, set threads(to) { threads = to; },
    get lost() { return lost; },
    get search() { return search; }, set search(to) { search = to; },
    get chosen() { return chosen; }, set chosen(to) { chosen = to; },
    get generations() { return generations; }, set generations(to) { generations = to; },
    get recheckEvery() { return recheckEvery; }, set recheckEvery(to) { recheckEvery = to; },
    get onChosen() { return onChosen; }, set onChosen(to) { onChosen = to; },
    get onCompared() { return onCompared; }, set onCompared(to) { onCompared = to; },
    get unchecked() { return unchecked; }, set unchecked(to) { unchecked = to; },
  };
}
