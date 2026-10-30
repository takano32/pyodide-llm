// T.477: forward/paths.ts, what /benchmark/ times of the model page's own path (T.184, T.190), alone: no suite ran it
// before (worker/timing.ts's timedPaths() calls it only in a browser's /benchmark/). An engine that stands in for
// forward.ts's (the members paths.ts reads: gpuReady, gpuWhyNot, gpuSide, gpuTokens, forwardMany, searching, threads,
// setThreads) and a made-up clock (performance.now, which paths.ts reads, set here):
//   - timedCell(): the lower middle run's speed and its GPU's tokens, the slowest's and the fastest's, unsteady past
//     choice.ts's margin;
//   - timePrompts() with no GPU: the CPU's side alone, a warm-up round and then rounds, the words over and over from
//     position 0, the GPU's cell why not (the engine's reason, else "not ready");
//   - with a GPU: the three sides in turn every round (the page's choice, the CPU, the GPU), the warm-up round left out
//     of the cells, and a GPU's side a run of which the GPU did not take whole said as no GPU's time;
//   - the side set back to the page's own (null) also where a run fails;
//   - endSearch(): writes until the search ends, or until its seconds are past, and sets the threads it found.
//   node tests/paths-check.mjs
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";

const { timePrompts, timedCell, endSearch, PATH_ROUNDS } = await import(runtimeUrl("forward/paths.ts"));
const { REASONS } = await import(runtimeUrl("forward/reasons.ts"));

// the clock paths.ts reads, which only the stand-in's runs move
let now = 0;
performance.now = () => now;

// ---- timedCell()
{
  const cell = timedCell([{ ms: 10, gpuTokens: 4 }, { ms: 30, gpuTokens: 2 }, { ms: 20, gpuTokens: 3 }, { ms: 40, gpuTokens: 1 }], 8);
  assert.deepEqual(cell, { speed: 400, low: 200, high: 800, gpuTokens: 3, unsteady: true },
    "the lower middle of four runs (20 ms: 8 tokens at 400 tok/s) and its GPU's tokens, the slowest's and the fastest's speed, unsteady");
  assert.equal(timedCell([{ ms: 10, gpuTokens: 0 }, { ms: 10.4, gpuTokens: 0 }, { ms: 10.2, gpuTokens: 0 }], 8).unsteady, false,
    "runs within choice.ts's margin (0.95) are steady");
  assert.equal(timedCell([{ ms: 10, gpuTokens: 0 }, { ms: 10.6, gpuTokens: 0 }], 8).unsteady, true, "and past it unsteady");
}
console.log("paths-check: timedCell() is a cell of the lower middle run, the slowest and the fastest, unsteady past the choice's margin");

// an engine as forward.ts's, of what paths.ts reads: each run moves the clock by ms(side, round) and the GPU's tokens by
// what took(side, count, round) says; every call is kept
function standIn({ gpu, why, ms, took = () => 0, failAt = -1 }) {
  const calls = [];
  let side = null, gpuTokens = 0;
  const rounds = new Map();
  return {
    calls,
    get gpuReady() { return gpu; },
    get gpuWhyNot() { return why; },
    get gpuSide() { return side; },
    set gpuSide(value) { side = value; },
    get gpuTokens() { return gpuTokens; },
    forwardMany(tokens, pos) {
      const key = `${tokens.length}:${side}`, round = rounds.get(key) ?? 0;
      rounds.set(key, round + 1);
      calls.push({ tokens: [...tokens], pos, side, round });
      if (calls.length - 1 === failAt) throw new Error("a run failed");
      now += ms(side, round);
      gpuTokens += took(side, tokens.length, round);
    },
  };
}

// ---- timePrompts() with no GPU
{
  const engine = standIn({ gpu: false, why: "no WebGPU here", ms: (side, round) => (round ? 5 : 1000) });
  const rows = timePrompts(engine, { words: [7, 8, 9], counts: [2, 5], rounds: 2 });
  assert.equal(engine.calls.length, 2 * 3, "a warm-up round and 2 rounds of each count, the CPU's side alone");
  assert.ok(engine.calls.every(({ side, pos }) => side === "cpu" && pos === 0), "on the CPU, from position 0");
  assert.deepEqual(engine.calls.at(-1).tokens, [7, 8, 9, 7, 8], "the words over and over, as many as the count");
  assert.deepEqual(rows.map(({ what, tokens, chosen, gpu }) => ({ what, tokens, chosen, gpu })),
    [2, 5].map((tokens) => ({ what: "prompt", tokens, chosen: { same: "cpu" }, gpu: { skip: "no WebGPU here" } })),
    "a row a count: the page's choice is the CPU's, and the GPU's cell says why not");
  assert.equal(rows[1].cpu.speed, 1000, "the warm-up round (1000 ms) is not in the cell: 5 tokens in 5 ms");
  assert.equal(engine.gpuSide, null, "the page's own choice again after");
  const without = timePrompts(standIn({ gpu: false, ms: () => 5 }), { words: [1], counts: [1], rounds: 1 });
  assert.deepEqual(without[0].gpu, { skip: REASONS.notReady }, "the GPU's cell where the engine says no reason");
}
console.log("paths-check: timePrompts() with no GPU times the CPU's side alone, from position 0, past a warm-up round, and says why the GPU has no cell");

// ---- timePrompts() with a GPU
{
  const ms = (side, round) => (!round ? 1000 : side === "cpu" ? 10 : 4);
  const engine = standIn({ gpu: true, ms, took: (side, count) => (side === "cpu" ? 0 : count) });
  const rows = timePrompts(engine, { words: [3], counts: [16], rounds: PATH_ROUNDS });
  assert.equal(engine.calls.length, 3 * (PATH_ROUNDS + 1), "three sides, a warm-up round and PATH_ROUNDS");
  assert.deepEqual(engine.calls.slice(0, 6).map(({ side }) => side), [null, "cpu", "gpu", null, "cpu", "gpu"],
    "the sides in turn every round: the page's choice, the CPU, the GPU");
  const [row] = rows;
  assert.deepEqual([row.chosen.speed, row.cpu.speed, row.gpu.speed], [4000, 1600, 4000], "each side's cell of its own runs, the warm-up left out");
  assert.deepEqual([row.chosen.gpuTokens, row.cpu.gpuTokens, row.gpu.gpuTokens], [16, 0, 16], "and the tokens of its median run the GPU took");
  assert.equal(engine.gpuSide, null);

  // a GPU's run that the GPU did not take whole (it failed on the way: the CPU's time)
  const partly = standIn({ gpu: true, why: undefined, ms, took: (side, count, round) => (side === "gpu" && round === 2 ? count - 1 : side === "cpu" ? 0 : count) });
  assert.deepEqual(timePrompts(partly, { words: [3], counts: [16], rounds: PATH_ROUNDS })[0].gpu, { skip: REASONS.notEveryBlock },
    "a GPU's side one of whose runs the GPU did not take whole is no GPU's time");
  const said = standIn({ gpu: true, why: "the GPU was lost", ms, took: (side, count, round) => (side === "gpu" && round === 3 ? 0 : side === "cpu" ? 0 : count) });
  assert.deepEqual(timePrompts(said, { words: [3], counts: [16], rounds: PATH_ROUNDS })[0].gpu, { skip: "the GPU was lost" }, "with the engine's reason where it has one");
  const warm = standIn({ gpu: true, ms, took: (side, count, round) => (side === "gpu" && round === 0 ? 0 : side === "cpu" ? 0 : count) });
  assert.equal(timePrompts(warm, { words: [3], counts: [16], rounds: PATH_ROUNDS })[0].gpu.skip, undefined, "the warm-up round's run is not held against it");
}
console.log("paths-check: timePrompts() with a GPU times the three sides in turn, past a warm-up round, and a GPU's side not taken whole is no GPU's time");

// ---- a run that fails
{
  const engine = standIn({ gpu: true, ms: () => 4, failAt: 4 });
  assert.throws(() => timePrompts(engine, { words: [1], counts: [4], rounds: 2 }), /a run failed/);
  assert.equal(engine.gpuSide, null, "the side the page chose again where a run fails");
}
console.log("paths-check: a run that fails leaves the side as the page chose it");

// ---- endSearch()
{
  let writes = 0, set;
  const engine = { get searching() { return writes < 3; }, threads: 4, async setThreads(count) { set = count; return count; } };
  assert.deepEqual(await endSearch(engine, () => { writes += 1; }), { threads: 4, found: 4, ended: true, generations: 3 },
    "writes until the search ends, and sets the threads it found");
  assert.equal(set, 4);
  const endless = { searching: true, threads: 2, async setThreads(count) { return count; } };
  assert.deepEqual(await endSearch(endless, () => { now += 1000; }, { seconds: 2 }), { threads: 2, found: 2, ended: false, generations: 2 },
    "and no longer than its seconds: then the search did not end");
}
console.log("paths-check: endSearch() writes until the search ends or its seconds are past, and sets the threads found");
