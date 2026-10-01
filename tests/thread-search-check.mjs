// T223 (the review of 2026-10-01): forward.js's real search for the number of software threads, on a made-up clock with
// noise, over visits that remember what the last one found as the page does (localStorage's threads:<model>:...). The
// engine is createForward() on a shared memory with kernels that do nothing and no helper started (spawn answers at once),
// so that Node alone runs it in a few seconds; what the search reads is the clock, and the clock here says how long a
// token of n threads takes. gpu-default-check.mjs holds the search to Pyodide's engine with the deterministic times and a
// made-up GPU; this holds it to a device whose times are noisy:
//   - the owner's Android's llm-jp-3 150M on the CPU alone (T223: 1 thread 110 tok/s, 2 123, 4 38.6, 8 32.3): from the
//     logical cores, from a remembered 4 (what the page kept before T223), 2 and 1, the count remembered is 2, in the
//     same comparisons as gpu-default-check.mjs's, and each visit that remembers a count costs two comparisons (40 tokens)
//     where the count is right;
//   - the same with the noise of a phone (a token's time spread by 10%, and a block of 5 tokens now and then 3 times as
//     long): the count settles on 2, a visit never leaves 4 or 8 remembered, and a remembered count changes between
//     visits seldom (the lower median of 8 times a count, T199, and the order best, candidate, candidate, best);
//   - CI's runner (1 thread 13 ms, 2 10, 4 11): 2 from every start.
//   node tests/thread-search-check.mjs [--forward <another forward.js, to see a broken one fail>]
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = new URL("..", import.meta.url);
const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : fileURLToPath(new URL("public/forward.js", root));
const { createForward, footprint } = await import(forwardFile);
const { CONTROL_BYTES } = await import(new URL("public/jobs.js", root));
const { threadsKey } = await import(new URL("src/bench.js", root));

// a float32 Llama of 2 layers: its tensors one after the other (the engine reads none of them with kernels that do nothing)
const [dim, hidden, layers, heads, vocab, seqLen] = [32, 64, 2, 2, 64, 64], headSize = dim / heads;
const tensors = {};
let end = 28;
for (const [name, shape] of [["token_embedding_table", [vocab, dim]], ["rms_att_weight", [layers, dim]], ["wq", [layers, dim, dim]],
  ["wk", [layers, dim, dim]], ["wv", [layers, dim, dim]], ["wo", [layers, dim, dim]], ["rms_ffn_weight", [layers, dim]],
  ["w1", [layers, hidden, dim]], ["w2", [layers, dim, hidden]], ["w3", [layers, hidden, dim]], ["rms_final_weight", [dim]],
  ["freq_cis_real", [seqLen, headSize / 2]], ["freq_cis_imag", [seqLen, headSize / 2]]]) {
  tensors[name] = { kind: "f32", offset: end, shape, group: 0, scales: 0 };
  end += shape.reduce((a, b) => a * b, 1) * 4;
}
const size = end;
const plan = { arch: "llama", dim, hidden_dim: hidden, n_layers: layers, n_heads: heads, n_kv_heads: heads, head_size: headSize, vocab_size: vocab,
  seq_len: seqLen, rotary: headSize, parallel_residual: false, kv_start: seqLen, rms_norm_eps: 1e-5, shared_classifier: true, int8: false,
  relaxed: false, tensors, derived: {}, outliers: [], half_kv: false };
const PAGE = 65536, after = footprint([dim, hidden, layers, heads, heads, vocab, seqLen], size, { dtype: "float32", shared: true });
const memory = new WebAssembly.Memory({ initial: Math.ceil((CONTROL_BYTES + size + after) / PAGE) + 1, maximum: Math.ceil((CONTROL_BYTES + size + after) / PAGE) + 2, shared: true });
const empty = new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
const spawn = async () => ({ terminate() {} });

// a seeded generator (mulberry32) and a normal one (Box-Muller): the same run every time
let seed = 20261001;
const random = () => {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const normal = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

const ANDROID = { 1: 1000 / 110, 2: 1000 / 123, 4: 1000 / 38.6, 8: 1000 / 32.3 };
const RUNNER = { 1: 13, 2: 10, 4: 11 };

/** One visit: a new engine on the shared memory, told the count the page remembers (0: none) and the logical cores (hint);
 * generations of 20 tokens (as many as a comparison's tokens, 4 blocks of 5) until the verdicts end. The clock: a token of
 * n threads takes ms[n] (30 where ms has none: a search gone the wrong way), times exp(sigma × a normal), a whole block of
 * 5 tokens times 3 with the probability slow. Returns what the engine told the page to remember, and its comparisons. */
async function visit(ms, { remembered = 0, hint = 8, sigma = 0, slow = 0 } = {}) {
  let engine = null, starting = false, time = 0, comparison = -1, token = 0, blockSlowed = 1;
  const clock = () => {
    starting = !starting;  // the search reads the clock as a token starts and as it ends
    if (starting) return time;
    if (engine.searchLog.length !== comparison) [comparison, token] = [engine.searchLog.length, 0];
    if (token++ % 5 === 0) blockSlowed = random() < slow ? 3 : 1;
    time += (ms[engine.threads] ?? 30) * Math.exp(sigma * normal()) * blockSlowed;
    return time;
  };
  engine = createForward({ memory, base: CONTROL_BYTES, size, kernels: { plain: empty, relaxed: null, wide: false }, plan, spawn, clock,
    wrap: () => new Proxy({}, { get: (_, name) => (name === "then" ? undefined : () => 0) }) });
  const told = [];
  await engine.findThreads({ from: hint, remembered, chose: (count) => told.push(count) });
  const atLoad = engine.searching;
  let generations = 0, startedAt = atLoad ? 0 : null;  // the generation the search began on
  for (; generations < 40 && !(told.length && !engine.searching); generations++) {
    engine.newGeneration();
    if (startedAt === null && engine.searching) startedAt = generations;
    for (let pos = 0; pos < 20; pos++) engine.forward(1 + (pos % 20), pos, true);
    await Promise.resolve();  // the helpers a larger count needs "start"
    await Promise.resolve();
  }
  const log = engine.searchLog.map(({ best, candidate, faster }) => `${best} or ${candidate}: ${faster ? candidate : best}`);
  engine.release();
  return { told, log, atLoad, startedAt, generations };
}

// ---- the deterministic times: the comparisons gpu-default-check.mjs's made-up clock has, and what a visit costs
{
  const fromCores = await visit(ANDROID, { remembered: 0 });
  assert.deepEqual(fromCores.log, ["8 or 4: 4", "4 or 2: 2", "2 or 1: 2"], "the owner's Android from its 8 logical cores");
  assert.deepEqual(fromCores.told, [2], "...remembers 2");
  // what the page kept before T223 (4) is searched again on the first generation of the visit, not at its load
  const stale = await visit(ANDROID, { remembered: 4 });
  assert.equal(stale.atLoad, false, "a remembered count is not searched while the model loads");
  assert.equal(stale.startedAt, 0, "a remembered count is checked on the first generation of the visit, not the eighth (T223: a visit seldom writes 8 answers)");
  assert.deepEqual(stale.log, ["4 or 2: 2", "2 or 1: 2"], "a remembered 4 on the owner's Android");
  assert.deepEqual(stale.told, [2], "...is told 2 to remember");
  // a right count is checked against both its neighbours: two comparisons, 20 tokens each
  const right = await visit(ANDROID, { remembered: 2 });
  assert.deepEqual(right.log, ["2 or 1: 2", "2 or 4: 2"], "a remembered 2: its neighbours");
  assert.equal(right.startedAt, 0, "...on the first generation too");
  assert.deepEqual(right.told, [2], "...is told 2 again");
  const low = await visit(ANDROID, { remembered: 1 });
  assert.deepEqual(low.log, ["1 or 2: 2", "2 or 4: 2"], "a remembered 1 goes up");
  assert.deepEqual(low.told, [2]);
  // CI's runner
  for (const remembered of [0, 1, 2, 4]) {
    const run = await visit(RUNNER, { remembered, hint: 4 });
    assert.deepEqual(run.told, [2], `CI's runner, remembered ${remembered}: 2 threads`);
  }
  console.log(`ok: the search on the owner's Android's times: from 8 ${JSON.stringify(fromCores.log)}, from a remembered 4 ${JSON.stringify(stale.log)}, ` +
    `from 2 ${JSON.stringify(right.log)}; 2 remembered each time`);
}

// ---- the noise of a phone
{
  const noises = { "a token's time spread by 10%": { sigma: 0.1 }, "a token's time spread by 10%, a block of 5 slowed 3 times 1 in 10": { sigma: 0.1, slow: 0.1 } };
  const lines = [];
  for (const [name, noise] of Object.entries(noises)) {
    // a chain of visits as the page makes them: each starts from the count the last one remembered (the first from the cores)
    let remembered = 0;
    const kept = {};
    let changes = 0, visits = 120;
    for (let v = 0; v < visits; v++) {
      const run = await visit(ANDROID, { remembered, ...noise });
      const next = run.told.at(-1);
      assert.ok(next !== undefined, `${name}: a visit ended with no count to remember`);
      assert.ok(next === 1 || next === 2 || (next === 8 && v === 0) || (next === 4 && v === 0), `${name}: visit ${v} remembered ${next}, past a neighbour of 2 where its times are 3 times 2's`);
      if (v > 0 && next !== remembered) changes++;
      kept[next] = (kept[next] ?? 0) + 1;
      remembered = next;
    }
    assert.ok((kept[2] ?? 0) >= 0.9 * visits, `${name}: 2 remembered by ${kept[2] ?? 0} of ${visits} visits`);
    assert.ok(changes <= 0.1 * visits, `${name}: the remembered count changed between ${changes} of ${visits} visits`);
    lines.push(`${name}: ${Object.entries(kept).map(([count, n]) => `${count} by ${n}`).join(", ")} of ${visits} visits, changed ${changes} times`);
  }
  // CI's runner, 4 and 2 close (11 against 10 ms): the count is one of the two, never 1 or 8
  let kept = {};
  for (let v = 0, remembered = 0; v < 60; v++) {
    const run = await visit(RUNNER, { remembered, hint: 4, sigma: 0.1 });
    remembered = run.told.at(-1);
    kept[remembered] = (kept[remembered] ?? 0) + 1;
  }
  assert.ok(!kept[1] && !kept[8] && (kept[2] ?? 0) >= 40, `CI's runner with noise: ${JSON.stringify(kept)}`);
  lines.push(`CI's runner, a token's time spread by 10%: ${JSON.stringify(kept)}`);
  console.log(`ok: the search with noise, ${lines.join("; ")}`);
}

// ---- the page's key of the count: one a model, a device and a browser (the page and /benchmark/ read the same one)
{
  const device = { hardwareConcurrency: 8, deviceMemory: 8, userAgent: "made up" };
  assert.notEqual(threadsKey("llm-jp-3-150m", device), threadsKey("tiny-lm", device), "the count is remembered one a model");
  assert.notEqual(threadsKey("llm-jp-3-150m", device), threadsKey("llm-jp-3-150m", { ...device, hardwareConcurrency: 4 }), "...and one a number of logical cores");
  console.log("ok: the page's key of the count is one a model and a device");
}
