// forward/paths.js (T349): what /benchmark/ times of the model page's own path through an engine: prompts as the
// page chooses, on the CPU only and on the GPU only (timePrompts, T184), and the threads' search run to its end (endSearch, T190).
// A module of public/forward.js, which asks for it with its own ?v=<build> and exports its names as before; it reads
// its neighbour the same way.

const { BETTER } = await import(new URL(`choice.js${new URL(import.meta.url).search}`, import.meta.url));

// T184: the model page's own path on /benchmark/ (worker/timing.js's timedPaths, src/bench.js's pathTable): prompts of counts
// tokens through forwardMany() at position 0 as the page chooses (T148), on the CPU only and on the GPU only
// (engine.gpuSide). The sides take turns, a warm-up round and then PATH_ROUNDS, so that a device that heats up or is
// busy for a while slows all of them alike; each cell is the median with the slowest and the fastest, unsteady where
// they are further apart than the choice's own margin (BETTER). Where the GPU is not on, the page's choice is the CPU:
// one side, timed once ("same" in the other). A GPU side a run of which the GPU did not take whole (it failed or was
// lost on the way: its time is the CPU's) is no GPU's time, and is said as such with why (as T157's lost device).
export const PATH_ROUNDS = 4;
export function timePrompts(engine, { words, counts, rounds = PATH_ROUNDS }) {
  const gpu = Boolean(engine.gpuReady);
  const sides = gpu ? { chosen: null, cpu: "cpu", gpu: "gpu" } : { cpu: "cpu" };
  const rows = [];
  try {
    for (const count of counts) {
      const tokens = Array.from({ length: count }, (_, i) => words[i % words.length]);
      const runs = Object.fromEntries(Object.keys(sides).map((name) => [name, []]));
      for (let round = 0; round <= rounds; round++) {
        for (const [name, side] of Object.entries(sides)) {
          engine.gpuSide = side;
          const before = engine.gpuTokens, began = performance.now();
          engine.forwardMany(tokens, 0);
          if (round) runs[name].push({ ms: performance.now() - began, gpuTokens: engine.gpuTokens - before });
        }
      }
      const row = { what: "prompt", tokens: count, chosen: { same: "cpu" } };
      for (const [name, list] of Object.entries(runs)) row[name] = timedCell(list, count);
      if (!gpu) row.gpu = { skip: engine.gpuWhyNot ?? "not ready" };
      else if (runs.gpu.some((run) => run.gpuTokens < count)) row.gpu = { skip: engine.gpuWhyNot ?? "the GPU did not take every block" };
      rows.push(row);
    }
  } finally {
    engine.gpuSide = null;
  }
  return rows;
}
/** T184: runs of count tokens ({ ms, gpuTokens }) as a cell: tok/s of the median run, of the slowest and the fastest,
 * the median run's tokens on the GPU, and whether they spread more than BETTER's margin */
export function timedCell(runs, count) {
  const sorted = [...runs].sort((a, b) => a.ms - b.ms), middle = sorted[(sorted.length - 1) >> 1];
  const speed = (run) => count / (run.ms / 1000);
  return { speed: speed(middle), low: speed(sorted.at(-1)), high: speed(sorted[0]), gpuTokens: middle.gpuTokens,
           unsteady: sorted.at(-1).ms * BETTER > sorted[0].ms };
}

// T190: the number of threads the page path is timed on is the model page's own. findThreads() started from the count
// the model page remembers for this device and model (then there is no search), or from the logical cores; a search is
// run here to its end on generations (write(): one of them, synchronous), as the model page ends it on its first texts.
// T184 stopped it after 8 generations wherever it had come to: the owner's Android timed the path on 1 thread, where the
// model page runs 4. A turn of the event loop between generations: the helpers a larger count needs start meanwhile.
// SEARCH_SECONDS at most; ended says whether the search came to its end.
export const SEARCH_SECONDS = 120;
export async function endSearch(engine, write, { seconds = SEARCH_SECONDS } = {}) {
  const until = performance.now() + seconds * 1000;
  let generations = 0;
  while (engine.searching && performance.now() < until) {
    write();
    generations += 1;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const ended = !engine.searching, found = engine.threads;
  return { threads: await engine.setThreads(found), found, ended, generations };
}
