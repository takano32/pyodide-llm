// worker/timing.js (T350): what /benchmark/ has the worker time: a generation, and the model page's own path (T184).
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const { state } = await import(new URL(`state.js${new URL(import.meta.url).search}`, import.meta.url));

// T45: a warm-up, then the measured run: the prompt, greedy, steps tokens (fewer where the model ends it first). The
// warm-up runs its 8 steps (T184's review: return() on a generator that has not started runs none of it)
export function timedGeneration(prompt, steps) {
  warmUp(prompt);
  const begin = performance.now();
  const pieces = state.llama.generate.callKwargs(prompt, { steps, temperature: 0, echo: false });
  let tokens = 0;
  try {
    while (!pieces.next().done) tokens += 1;
  } finally {
    pieces.destroy();
  }
  return { tokens, speed: tokens / ((performance.now() - begin) / 1000) };
}
function warmUp(prompt) {
  const pieces = state.llama.generate.callKwargs(prompt, { steps: 8, temperature: 0, echo: false });
  try {
    while (!pieces.next().done);
  } finally {
    pieces.destroy();
  }
}

// T184: the model page's own path on the model loaded now (src/bench.js's pathTable() writes it). The GPU is waited for
// first, GPU_WAIT_S at most (the page does not wait: its first prompts go on the CPU meanwhile; a GPU not ready by then
// is said as such), then the number of threads is the model page's (T190: the count the model page remembers, which
// /benchmark/ passes as the model page does, or else the search run to its end: forward.js's endSearch()), and every
// side is timed on it (the CPU's times, and so the choice, are per number of threads). The prompts:
// forward.js's timePrompts(). The writing: sampled tokens after the prompt at the tok/s the page's status line says
// (the engine's stats); T152: where the GPU takes a generation's steps, as the page chooses, on the CPU only and on the
// GPU only (engine.gpuSide), the sides in turn, a run each after the warm-up.
const GPU_WAIT_S = 240, WRITING_RUNS = 3;
export async function timedPaths({ prompt, counts, sampled }) {
  const engine = state.outsideNow?.engine;
  if (!engine) return { error: `${state.llama.backend} runs this model here: the page's path is the NumPy engine's` };
  if (engine.gpu && !engine.gpuReady) {
    postMessage({ type: "status", text: "the GPU gets ready" });
    let timer;
    await Promise.race([engine.gpu, new Promise((resolve) => (timer = setTimeout(resolve, GPU_WAIT_S * 1000)))]);
    clearTimeout(timer);
  }
  const chosen = engine.gpuReady;
  const gpu = chosen ? { seconds: chosen.seconds, matrices: chosen.matrices, attention: chosen.attention }
    : { why: engine.gpuWhyNot ?? `not ready after ${GPU_WAIT_S} s` };
  postMessage({ type: "status", text: "the software threads" });
  const { threads, found, ended } = await state.forwardModule.endSearch(engine, () => timedGeneration(prompt, 64));
  // how the count came about, for the table's head (T190): src/bench.js's pathTable() says it. A software thread that
  // stopped (T120: the engine gave its helpers up and runs on one) is said first: found is 1 then, and the search's
  // verdicts or the remembered count would name another count (T190's review)
  const remembered = state.threadsRequest?.remembered || 0;
  const how = !state.weightsPool?.shared ? { alone: "no shared memory here" }
    : engine.lostThreads ? { alone: "a software thread stopped" }
    : threads < found ? { alone: `not the ${found} asked for: its software threads did not start` }
    : !ended ? { unfinished: state.forwardModule.SEARCH_SECONDS }
    : remembered && threads !== remembered ? { alone: `not the ${remembered} the model page remembers: its software threads did not start` }
    : remembered ? { remembered: true }
    : { searched: engine.searchLog.map(({ best, candidate, faster }) => [best, candidate, faster ? candidate : best]) };
  const encoded = state.llama.tokenizer.encode(prompt);
  const words = encoded.toJs();
  encoded.destroy();
  postMessage({ type: "status", text: `prompts of ${counts.join(" and ")} tokens` });
  const rows = state.forwardModule.timePrompts(engine, { words: words.length ? words : [state.llama.bos], counts: counts.filter((n) => n <= state.llama.seq_len) });
  const writes = Math.min(sampled, state.llama.seq_len - words.length);  // steps counts the prompt's positions too
  postMessage({ type: "status", text: `writing ${writes} tokens` });
  warmUp(prompt);
  let fewest = writes;  // a stop token may end a run first: its tok/s stands, and the row says the fewest
  // a run of the writing on side ("cpu", "gpu", or null: as the page chooses, T152); whether the GPU took every step
  const written = (side = null) => {
    engine.gpuSide = side;
    const before = engine.gpuSampled;
    try {
      const pieces = state.llama.generate.callKwargs(prompt, { steps: words.length + writes, temperature: 0, echo: false });
      try {
        while (!pieces.next().done);
      } finally {
        pieces.destroy();
      }
      const stats = state.llama.stats.toJs({ dict_converter: Object.fromEntries });
      fewest = Math.min(fewest, stats.sampled);
      const whole = engine.gpuSampled - before >= stats.sampled;
      // as if each had written them all, at its tok/s
      return { ms: (1000 * writes) / stats.tokens_per_second, gpuTokens: whole ? writes : engine.gpuSampled - before };
    } finally {
      engine.gpuSide = null;
    }
  };
  // T152: the sides, where the GPU takes a generation's steps (else the CPU's alone), in turn
  const tokens = Boolean(chosen?.tokens);
  const sides = tokens ? { chosen: null, cpu: "cpu", gpu: "gpu" } : { cpu: "cpu" };
  const runs = Object.fromEntries(Object.keys(sides).map((name) => [name, []]));
  for (let run = 0; run < WRITING_RUNS; run++) {
    for (const [name, side] of Object.entries(sides)) runs[name].push(written(side));
  }
  const row = { what: "generation", tokens: fewest, chosen: tokens ? state.forwardModule.timedCell(runs.chosen, writes) : { same: "cpu" },
    cpu: state.forwardModule.timedCell(runs.cpu, writes), gpu: tokens ? state.forwardModule.timedCell(runs.gpu, writes) : { skip: chosen ? engine.gpuTokensWhyNot ?? "not on the GPU" : gpu.why } };
  // a GPU side the GPU did not take whole (it failed, or was lost, on the way: its time is the CPU's)
  if (tokens && runs.gpu.some((run) => run.gpuTokens < writes)) row.gpu = { skip: engine.gpuTokensWhyNot ?? "the GPU did not take every step" };
  rows.push(row);
  // T190's review: the writing on each number of threads the search goes through (1, 2, 4, ... up to the logical cores,
  // and the page's), in turn: the page's count against the others on this very model. A count the model page remembers
  // is not searched here, and the CPU section's made-up model (2 layers: 11 waits between phases a token) says little
  // of a model like llm-jp-3 150M (12 layers: 61 waits, most on a phase of about 1 MB)
  const hint = Math.max(1, state.threadsRequest?.hint || 1);
  const tried = [...new Set([1, ...Array.from({ length: Math.floor(Math.log2(hint)) }, (_, i) => 2 ** (i + 1)), hint, threads])].sort((a, b) => a - b);
  const byCount = new Map(tried.map((n) => [n, []]));
  if (state.weightsPool?.shared && !engine.lostThreads && tried.length > 1) {
    for (let run = 0; run < WRITING_RUNS; run++) {
      for (const n of tried.filter((c) => byCount.has(c))) {
        postMessage({ type: "status", text: `writing on ${n} software thread${n === 1 ? "" : "s"}` });
        // a count whose software threads did not start runs on fewer: no times of it
        if ((await engine.setThreads(n)) !== n) {
          byCount.delete(n);
          continue;
        }
        warmUp(prompt);  // the helpers a switch wakes, out of the time
        byCount.get(n).push(written("cpu"));  // (T152: the CPU's steps, whichever side the page chooses)
      }
    }
    await engine.setThreads(threads);
  }
  const perCount = engine.lostThreads ? [] : [...byCount].filter(([, list]) => list.length).map(([n, list]) => ({ threads: n, ...state.forwardModule.timedCell(list, writes) }));
  // a software thread that stopped while the sides were timed: the times after it are one thread's
  if (engine.lostThreads && !how.alone) how.stopped = true;
  // the rounds (T45) that follow load the model again: on this count too, not searching again while they are timed
  if (state.weightsPool?.shared && !engine.lostThreads) state.threadsRequest = { ...state.threadsRequest, remembered: threads };
  // whatever stopped the GPU while the sides were timed (a failure, a lost device): its cells are empty (timePrompts)
  if (chosen && engine.gpuWhyNot) gpu.lost = engine.gpuWhyNot;
  return { threads, how, perCount, gpu, status: engine.gpuStatus, rows };
}
