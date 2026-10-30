// forward/threads.ts (T.349): the software threads of one engine: a phase's rows shared out among them on the control
// area of a shared memory (jobs.ts has its layout; helper.ts is a thread), a thread that stopped given up (T.120), and
// the search for how many of them to use (T.93's stage 2b, T.199, T.223, T.239).
// A module of src/runtime/forward.ts; it reads jobs.ts and its neighbours by static imports.

import { GEN, QUIT, COUNTER, FINISHED, ACTIVE, TOTAL, WAKE, JOBS, JOB, JOB_TABLE, BATCH, ROWS, COUNT, SIZE,
  FIRST, TILE_ROWS } from "../jobs.ts";
import { lowerMedian, BETTER } from "./choice.ts";
import type { Job } from "../jobs.ts";
import type { Control } from "./control.ts";
import type { ThreadsCompared } from "../contracts/messages.ts";

// (T.368.4, the types) a software thread, started and warm, and what starts one (the worker's spawn(): share is the
// thread's own word of the control area)
export type Thread = { terminate?(): void };
// (blocks, T.363.5: the modules of the kernels of llama.cpp's blocks, for a model with such rows; else null)
export type Spawn = (start: { memory: WebAssembly.Memory; wide: boolean; plain: WebAssembly.Module; relaxed: WebAssembly.Module | null; blocks: Blocks | null; share: number }) => Promise<Thread>;
type Blocks = { plain: WebAssembly.Module; relaxed: WebAssembly.Module | null };
// a comparison of two counts as the page is told it, and as the log keeps it
export type Compared = Omit<ThreadsCompared, "type" | "load" | "model">;
type Logged = { best: number; candidate: number; times: Record<number, number[]>; faster: boolean; whileGpu: boolean };
// the search under way: the best count so far and the one it is held against, each one's ms a token (null until a
// candidate is chosen), the step of the comparison, whether the candidate's helpers are still starting
type Search = { best: number; direction: "down" | "up"; moved: boolean; far: boolean; candidate: number; times: Record<number, number[]> | null; step: number;
  waiting: boolean; whileGpu: boolean };
// (the control area's words and the table of the jobs are read where threads run and nowhere else: taken once there, ctl! and table!)

// ---- the helper threads (stage 2): the control area at the start of a shared memory (jobs.ts has its layout)
// how many chunks per thread the rows of a matmul are cut into: whoever is free takes the next one, so a slow core
// (a little core of a big.LITTLE phone) simply takes fewer. 2 to 16 measured the same (T.93); fewer does not steal.
const CHUNKS_PER_THREAD = 4;

/** T.349: the phases and the software threads of one engine (createForward(), which hands in what they read of it):
 * phase(jobs) shares a phase's rows out among the threads in use and waits for them (T.93, T.120), and the search finds
 * how many to use (T.199, T.223, T.239, T.240).
 * T.378.1: the state here (the count in use, the search, what a visit remembers) has this one owner. The rest of the
 * engine tells it what happened, by functions named for that (use, find, generationBegan, tokenBegan and tokenEnded,
 * stop), and reads three things of it (count, lost, searching); nothing outside assigns a variable of it. The engine
 * takes the functions out by name once (phase() and tokenBegan() are a token's path: plain calls of closures).
 * spawn: left out where no software thread is ever to run (a memory that is not shared has none either; the engine
 * leaves it out where the model is on the GPU alone, T.156). gpuGettingReady(): whether the GPU's side is still getting
 * ready, which is made after this.
 * T.378.2: kernels: the two a thread is started with ({ plain, relaxed }, relaxed null where the model does not run on
 * it): the part knows nothing of the plan. control: the control area (control.js), which is the engine's and the GPU's
 * side's as well. */
export function softwareThreads({ memory, wide, kernels, spawn, control, runRows, gpuGettingReady }: {
  memory: WebAssembly.Memory; wide: boolean; kernels: { plain: WebAssembly.Module; relaxed: WebAssembly.Module | null; blocks?: Blocks | null }; spawn: Spawn | null | undefined;
  control: Control; runRows: (job: Job, r0: number, r1: number) => unknown; gpuGettingReady: () => boolean }) {
  const { ctl, waitUntil } = control;
  // ---- the matrix multiplications, in phases: the matmuls that read the same input (q, k and v; w1 and w3) go
  // out together. The input is quantized here, once; the rows are computed here and, with helper threads, by
  // whoever takes them. Every row is computed whole by one thread with the same kernel, so the numbers are the same
  // with any number of threads.
  //
  // A job is what jobs.ts says: [kind, eight arguments, rows, count, out stride, a stride, b stride].
  // (ctl: null on a memory that is not shared, where no thread can be started)
  const shared = ctl && spawn;
  const table = shared ? new Float64Array(memory.buffer, JOB_TABLE, BATCH * JOB) : null;  // the jobs (jobs.ts)
  const helpers: Thread[] = [];
  let threads = 1, gen = 0;
  // T.120: every helper goes, and this engine keeps to one thread from here on. The phase is then run again here:
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
  function phase(jobs: number[][]) {
    const alone = () => {
      for (const job of jobs) runRows(job, 0, job[ROWS]);
    };
    if (threads <= 1) return alone();
    // (more threads than one: a shared memory, whose words and table of the jobs there are)
    const words = ctl!, list = table!;
    // close the previous phase (odd), let every helper still awake leave it, then rewrite the jobs
    Atomics.store(words, GEN, gen + 1);
    if (!waitUntil(ACTIVE, (seen) => seen === 0)) {
      giveUp();
      return alone();
    }
    let total = 0;
    words[JOBS] = jobs.length;
    jobs.forEach((job, i) => {
      const at = i * JOB, rows = job[ROWS];
      // T.159: a prompt's chunks in fours of rows, the tiles of matmul_q8r_tile; T.620: in the rows of its kind's tile
      // (a row at a time for every other kind)
      const quad = job[COUNT] > 1 ? TILE_ROWS[job[0]] : 1;
      const size = quad * Math.ceil(rows / (threads * CHUNKS_PER_THREAD * quad));
      list.set(job, at);
      list[at + SIZE] = size;
      list[at + FIRST] = total;
      total += Math.ceil(rows / size);
    });
    words[TOTAL] = total;
    Atomics.store(words, COUNTER, 0);
    Atomics.store(words, FINISHED, 0);
    gen += 2;
    Atomics.store(words, GEN, gen);
    // wake helpers 1.. by name, no more than there are chunks for them and no more than this many threads
    for (let h = 1, wake = Math.min(threads - 1, total - 1); h <= wake; h++) {
      Atomics.store(words, WAKE + h, gen);
      Atomics.notify(words, WAKE + h, 1);
    }
    for (let c = Atomics.add(words, COUNTER, 1); c < total; c = Atomics.add(words, COUNTER, 1)) {
      let j = jobs.length - 1;
      while (list[j * JOB + FIRST] > c) j--;
      const at = j * JOB, size = list[at + SIZE], r0 = (c - list[at + FIRST]) * size;
      runRows(jobs[j], r0, Math.min(r0 + size, jobs[j][ROWS]));
      Atomics.add(words, FINISHED, 1);
    }
    if (!waitUntil(FINISHED, (seen) => seen === total)) {
      giveUp();
      alone();
    }
  }

  // ---- the number of threads (stage 2b): found by measuring, never written down. The search starts from a hint
  // (navigator.hardwareConcurrency, which counts the little cores of a big.LITTLE phone too) and compares the best
  // count so far with half of it and, if half is not faster, with twice as many; it goes on in that direction while
  // the other is faster by more than the noise of a run, and stops at the first that is not (T.239: on the way down, at
  // the second in a row that is not: a quarter is compared where half was not faster). Only the tokens that
  // make logits are timed (a prompt's tokens skip the classifier). One comparison runs the two counts in blocks,
  // best-candidate-candidate-best, so that the growing cost of later positions falls on both alike, and drops the
  // first token of every block (the switch). Helpers that a count needs are started in the background; until they
  // are ready the tokens run on the best count and are not timed.
  // T.199: a count's time is the lower median of its 8 (two blocks of 4). What else runs (a collection of the garbage,
  // another tab, the page on its way to the background) slows a block, never speeds one up, so one block slowed whole
  // leaves the other block's 4 below it and the verdict stands. The upper median it was took that block's time: twice
  // in CI 2 threads 1.24 to 1.34 times as fast as 1 lost to it (T.190's review).
  // T.223: a count is remembered only where a search timed it with nothing of the GPU's getting ready beside it (its
  // upload, its shaders checked against JavaScript and timed use the CPU and the memory: T.148 does not time the CPU's
  // prompts then either). A search on the first texts goes on while the GPU gets ready (the first visit is not left on
  // the logical cores meanwhile), but its verdict is used only until the GPU is ready: the first generation after that
  // searches again from it, and that search's verdict is the one remembered. A count remembered from an earlier visit
  // is searched again the same way once a visit (the first generation with no GPU getting ready), not only every
  // recheck generations of one load: the owner's Android kept 4 threads for llm-jp-3 150M where 2 wrote 3.2 times as
  // fast (T.223), and a visit seldom writes 8 answers. unchecked: the count in use is not such a search's verdict yet.
  // T.239: half may be a dip with a faster count below it. The owner's PC (16 logical cores) stopped at 8 threads ("16 or
  // 8: 8, 8 or 4: 8") where 2 wrote faster, and the search of every later visit began from that 8 and ended on it. So a
  // count that half did not beat is compared with a quarter of it too (far), and the way down goes on by halves from a
  // quarter that is faster. A comparison more (20 tokens) where the best count is 4 or more and nothing below it is
  // faster; none more where it is 1 or 2. Not on the way up: a visit that remembers 2 would time 8 threads every time
  // (the owner's Android: 0.23 s), and no device's report has a dip above its count (TASKS.md's T.239 has the table).
  // T.240: the search the count in use is owed (unchecked) does not wait for the next generation where the GPU is ready
  // inside one: it begins at the first token after that (engine.ts's forward()), so that a long first answer is not written
  // to its end on a count timed beside the GPU's getting ready. Only where the page began a generation: /benchmark/
  // begins none and takes the count the model page remembers as it is (T.190).
  const BLOCK = 4;
  let search: Search | null = null, chosen = 0, generations = 0, recheckEvery = 0, onChosen: ((count: number) => void) | null | undefined = null,
    onCompared: ((verdict: Compared) => void) | null | undefined = null, unchecked = false;
  // whether a search from the count in use may begin now (none under way, nothing of the GPU's getting ready beside it)
  const mayRecheck = () => !search && chosen && recheckEvery && !gpuGettingReady();
  const searchLog: Logged[] = [];  // every comparison: the counts, their times in ms per token, and the verdict
  function beginSearch(from: number) {
    search = { best: Math.max(1, from), direction: from > 1 ? "down" : "up", moved: false, far: false, candidate: 0, times: null, step: 0, waiting: false, whileGpu: false };
    nextCandidate();
  }
  function nextCandidate() {
    if (lost) return finish();
    const { best, direction, far } = search!;
    const candidate = direction === "down" ? Math.floor(best / (far ? 4 : 2)) : best * 2;
    if (candidate < 1) return passed();
    search!.candidate = candidate;
    search!.times = { [best]: [], [candidate]: [] };
    search!.step = 0;
    if (helpers.length < candidate - 1) {
      search!.waiting = true;
      ensureHelpers(candidate).then((complete) => {
        if (!search) return;
        if (complete) search.waiting = false;
        else finish();
      }, (error) => {
        // (T.380.2: said. forward.ts's parts do not read failure.ts, so the caught value in its own words)
        console.warn(`threads: a software thread did not start (${String(error)}): the search ends at ${best}`);
        finish();
      });
    }
  }
  // the candidate was not faster than the best, or there is no count there: the next one, or the end (T.239: a quarter
  // after half; then twice as many, where the best is still the count the search began from)
  function passed() {
    if (search!.direction === "down" && !search!.far) search!.far = true;
    else if (search!.direction === "down" && !search!.moved) search!.direction = "up";
    else return finish();
    nextCandidate();
  }
  function finish() {
    chosen = lost ? 1 : search ? search.best : threads;
    threads = chosen;
    unchecked = !lost && Boolean(search?.whileGpu);  // T.223: searched again once the GPU is ready, and remembered then
    search = null;
    // one thread after a give-up says nothing about the device: the page would start with it next time (the review of T.120)
    if (!lost && !unchecked) onChosen?.(chosen);
  }
  // the count for the next token, and whether it is timed
  function countForToken(): [number, boolean] {
    if (lost) return [1, false];
    if (!search || search.waiting) return [search ? search.best : threads, false];
    const order = [search.best, search.candidate, search.candidate, search.best];
    const block = Math.floor(search.step / (BLOCK + 1)), inBlock = search.step % (BLOCK + 1);
    return [order[block], inBlock > 0];
  }
  function recordToken(count: number, milliseconds: number, timed: boolean) {
    if (!search || search.waiting) return;
    if (timed) {
      search.times![count].push(milliseconds);
      if (gpuGettingReady()) search.whileGpu = true;
    }
    search.step += 1;
    if (search.step < 4 * (BLOCK + 1)) return;
    const { best, candidate } = search;
    const bestMs = lowerMedian(search.times![best]), candidateMs = lowerMedian(search.times![candidate]);
    const faster = candidateMs < bestMs * BETTER;
    const { whileGpu } = search;
    searchLog.push({ best, candidate, times: search.times!, faster, whileGpu });
    // T.114: every verdict, so that a device's choice can be followed afterwards (the page writes it to the console)
    onCompared?.({ best, candidate, bestMs, candidateMs, faster, whileGpu, tokens: search.times![best].length + search.times![candidate].length });
    if (!faster) return passed();
    Object.assign(search, { best: candidate, moved: true, far: false });
    nextCandidate();
  }
  // the helpers in QUIT's hands: set, every one woken to see it, and each ended
  let stops = 0;  // stopHelpers() counts them: a helper whose start began before one is not kept
  function stopHelpers() {
    // (a shared memory's: stop() asks for it there alone, and giveUp() where a helper ran)
    const words = ctl!;
    stops += 1;
    Atomics.store(words, QUIT, 1);
    for (let h = 1; h <= helpers.length; h++) {
      Atomics.add(words, WAKE + h, 2);
      Atomics.notify(words, WAKE + h);
    }
    // QUIT stays set until the next ensureHelpers(): a helper that wakes late must still see it
    helpers.splice(0).forEach((helper) => helper.terminate?.());
    threads = 1;
  }
  // Resolves to whether the n threads are there. A helper that becomes ready after this engine let its helpers go
  // (release(), stopThreads(), giveUp()) is ended at once: the review of T.120 found such ones left alive when the
  // visitor chose another model while the search started more, and the next model's engine on the same memory
  // (T.96) woke them, with the old engine's kernels (one threw holding a chunk: 10 s still, then one thread)
  async function ensureHelpers(n: number) {
    if (lost) return false;  // T.120: none again after a helper stopped under this engine
    const since = stops;
    if (helpers.length < n - 1 && helpers.length === 0) Atomics.store(ctl!, QUIT, 0);  // after stopThreads(): a fresh start
    while (helpers.length < n - 1) {
      const helper = await spawn!({ memory, wide, plain: kernels.plain, relaxed: kernels.relaxed, blocks: kernels.blocks ?? null, share: helpers.length + 1 });
      if (lost || stops !== since) {
        helper.terminate?.();
        return false;
      }
      helpers.push(helper);
    }
    return true;
  }

  // ---- T.378.1: what happens to the threads, as the rest of the engine tells it
  /** n threads from the next token on, with no search: the helpers that are missing are started. Resolves to the count
   * in use: 1 where there are no software threads here, or one stopped under this engine (T.120) */
  async function use(n: number) {
    if (!shared || lost) return (threads = 1);
    search = null;
    if (!(await ensureHelpers(n))) return (threads = 1);
    threads = Math.max(1, n);
    return threads;
  }
  /** The count is to be found while tokens are written: from a hint, or from a count remembered from an earlier visit,
   * which is searched again from on this visit (T.223) and then every recheck generations. chose(count) is told the
   * answer, compared(verdict) every comparison on the way. The helpers of the starting count are started before this
   * resolves (to the count in use), so the first tokens do not wait for them */
  async function find({ from, remembered = 0, recheck = 8, chose, compared }:
    { from: number; remembered?: number; recheck?: number; chose?: (count: number) => void; compared?: (verdict: Compared) => void }) {
    if (!shared || lost) return 1;
    onChosen = chose;
    onCompared = compared;
    recheckEvery = recheck;
    const start = Math.max(1, remembered || from);
    if (!(await ensureHelpers(start))) return (threads = 1);
    threads = start;
    if (remembered) {
      chosen = remembered;
      unchecked = true;  // T.223: searched again from it on this visit
    } else {
      beginSearch(start);
    }
    return threads;
  }
  /** the page began a generation: now and then the count in use is checked against its neighbours again (T.223: and
   * first on the first generation with no GPU getting ready, where the count is remembered from an earlier visit or
   * was found while the GPU got ready) */
  function generationBegan() {
    generations += 1;
    if (mayRecheck() && (unchecked || generations % recheckEvery === 0)) beginSearch(chosen);
  }
  // A token's turn in the search, between tokenBegan() and tokenEnded(): [the count it runs on, whether it is timed]
  let turn: [number, boolean] | null = null;
  /** A token is about to run. Whether it is the search's: it then runs on the count the search wants next, and the
   * caller times it and says tokenEnded(its milliseconds). T.240: the search the count in use is owed begins here. Only
   * the tokens that make logits are the search's */
  function tokenBegan(needLogits: boolean) {
    if (unchecked && generations && mayRecheck()) beginSearch(chosen);
    if (!search || !needLogits) return false;
    turn = countForToken();
    threads = turn[0];
    return true;
  }
  /** the token tokenBegan() said was the search's took this long: counted, and the count in use is the best so far
   * again (or what the search ended on). Nothing where no such token began */
  function tokenEnded(milliseconds: number) {
    if (!turn) return;
    const [count, timed] = turn;
    turn = null;
    recordToken(count, milliseconds, timed);
    if (search) threads = search.best;
  }
  /** the helper threads end; the engine runs on its own again */
  function stop() {
    if (shared) stopHelpers();
  }
  return {
    phase, searchLog, use, find, generationBegan, tokenBegan, tokenEnded, stop,
    get count() { return threads; },
    get lost() { return lost; },
    get searching() { return search !== null; },
  };
}
// (T.368.4) what the rest of an engine has of its software threads
export type Threads = ReturnType<typeof softwareThreads>;
