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
//   - CI's runner (1 thread 13 ms, 2 10, 4 11): 2 from every start;
//   - T239: the owner's PC (16 logical cores: 2 threads 171 tok/s, 8 threads 158, and 4 no faster than 8) ends on 2, from
//     its logical cores and from the 8 the page remembered, by the quarter the search compares where half was not faster;
//     the Android's visits cost what they did, and a device whose logical cores are its best count one comparison more.
//   - T240: the search a count is owed (one found beside the GPU's getting ready, or remembered from an earlier visit)
//     begins at the token after the GPU is ready, inside the generation, on an int8 model with a made-up GPU's worker;
//     never while the GPU gets ready, and never where the page began no generation (/benchmark/).
//   node tests/thread-search-check.mjs [--forward <another forward.js, to see a broken one fail>] [--table: T239's table]
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runtime, runtimeUrl } from "./tree.mjs";

const root = new URL("..", import.meta.url);
const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : runtime("forward.js");
const { createForward, footprint } = await import(forwardFile);
const { CONTROL_BYTES } = await import(runtimeUrl("jobs.js"));
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
// T239: the owner's PC's llm-jp-3 150M (16 logical cores, T225's report): 2 threads write 171 tok/s, the 8 the page chose
// 158, and the page's verdicts were "16 or 8: 8, 8 or 4: 8". The times of 1, 4 and 16 threads are not in the report: made
// up here, to agree with those verdicts
const PC = { 1: 1000 / 100, 2: 1000 / 171, 4: 1000 / 150, 8: 1000 / 158, 16: 1000 / 120 };
// made-up shapes: a dip between a count that is good and the best, far below it (the PC's shape with a clear best), on
// 16, 8 and 4 logical cores; every doubling 4/3 times as fast; a dip above a remembered count
const DIP = { 1: 10, 2: 5, 4: 8, 8: 6.5, 16: 9 }, DIP4 = { 1: 5, 2: 8, 4: 6.5 };
const MORE = { 1: 16, 2: 12, 4: 9, 8: 6.75, 16: 5.0625 };
const DIP_ABOVE = { 1: 10, 2: 6, 4: 8, 8: 5 };
const upTo = (ms, cores) => Object.fromEntries(Object.entries(ms).filter(([count]) => count <= cores));
const fastest = (ms) => Number(Object.keys(ms).reduce((a, b) => (ms[b] < ms[a] ? b : a)));

/** One visit: a new engine on the shared memory, told the count the page remembers (0: none) and the logical cores (hint);
 * generations of 20 tokens (as many as a comparison's tokens, 4 blocks of 5) until the verdicts end. The clock: a token of
 * n threads takes ms[n] (30 where ms has none: a search gone the wrong way), times exp(sigma × a normal), a whole block of
 * 5 tokens times by (a number, or [from, to]: one between them) with the probability slow. Returns what the engine told
 * the page to remember, its comparisons, and extra: the ms they took over as many tokens on the fastest count of ms. */
async function visit(ms, { remembered = 0, hint = 8, sigma = 0, slow = 0, by = 3 } = {}) {
  let engine = null, starting = false, time = 0, comparison = -1, token = 0, blockSlowed = 1;
  const clock = () => {
    starting = !starting;  // the search reads the clock as a token starts and as it ends
    if (starting) return time;
    if (engine.searchLog.length !== comparison) [comparison, token] = [engine.searchLog.length, 0];
    if (token++ % 5 === 0) blockSlowed = random() >= slow ? 1 : Array.isArray(by) ? by[0] + (by[1] - by[0]) * random() : by;
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
  const extra = engine.searchLog.reduce((sum, { best, candidate }) => sum + 10 * ((ms[best] ?? 30) + (ms[candidate] ?? 30) - 2 * ms[fastest(ms)]), 0);
  engine.release();
  return { told, log, atLoad, startedAt, generations, extra };
}

// ---- T239's table (--table): what this forward.js's search spends and where it ends, a shape of times and a number of
// logical cores a row. No check: it is how the form of the search was chosen (TODO.md's T239), run on each form's copy of
// forward.js through --forward. first: the comparisons of a first visit (from the logical cores), the count it ends on,
// and the ms they take over as many tokens on the fastest count; later: the same of a visit that remembers the fastest
// count; then, under each noise, 200 first visits and a chain of 200 visits (each from what the last one remembered): the
// share that ends on the fastest count, and how many times as long a token of the count remembered takes as one of the
// fastest, on average
const NOISES = { "10%": { sigma: 0.1 }, "20%": { sigma: 0.2 }, "10%, 2 blocks in 10 slowed 2 to 3 times": { sigma: 0.1, slow: 0.2, by: [2, 3] } };
if (args.includes("--table")) {
  const shapes = [["the owner's PC (1, 4, 16 made up)", PC, 16], ["a dip, 16 cores", DIP, 16], ["a dip, 8 cores", upTo(DIP, 8), 8], ["a dip, 4 cores", DIP4, 4],
    ["the owner's Android", ANDROID, 8], ["the Android's 1, 2, 4 on 4 cores", upTo(ANDROID, 4), 4], ["the Android's, 16 cores (16 made up)", { ...ANDROID, 16: 40 }, 16],
    ["more is faster, 4 cores", upTo(MORE, 4), 4], ["more is faster, 8 cores", upTo(MORE, 8), 8], ["more is faster, 16 cores", MORE, 16],
    ["a dip above a remembered 2, 8 cores", DIP_ABOVE, 8, 2]];
  console.log(`| times | first visit | later visit | ${Object.keys(NOISES).map((n) => `${n}: first visits, chain`).join(" | ")} |`);
  console.log(`|---|---|---|${Object.keys(NOISES).map(() => "---|").join("")}`);
  for (const [name, ms, hint, from = 0] of shapes) {
    const best = fastest(ms), cell = (run) => `${run.log.length} (${run.told.at(-1)}, ${run.extra.toFixed(0)} ms)`;
    const cells = [cell(await visit(ms, { hint, remembered: from })), cell(await visit(ms, { hint, remembered: best }))];
    for (const noise of Object.values(NOISES)) {
      const share = async (chained) => {
        let right = 0, slower = 0, remembered = from;
        for (let v = 0; v < 200; v++) {
          const next = (await visit(ms, { hint, remembered, ...noise })).told.at(-1);
          right += next === best;
          slower += (ms[next] ?? 30) / ms[best];
          remembered = chained ? next : from;
        }
        return `${(right / 2).toFixed(0)}% (${(slower / 200).toFixed(2)}×)`;
      };
      cells.push(`${await share(false)}, ${await share(true)}`);
    }
    console.log(`| ${name}: ${Object.entries(ms).map(([count, t]) => `${count}: ${t.toFixed(1)}`).join(", ")} ms, fastest ${best} | ${cells.join(" | ")} |`);
  }
  process.exit(0);
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

// ---- T239: a count that half does not beat may have a faster one below the dip: a quarter is compared too
{
  const pc = await visit(PC, { hint: 16 });
  assert.deepEqual(pc.log, ["16 or 8: 8", "8 or 4: 8", "8 or 2: 2", "2 or 1: 2"], "the owner's PC from its 16 logical cores: 8 or 2 after 8 or 4");
  assert.deepEqual(pc.told, [2], "...remembers 2");
  // the 8 the page remembers there today: a later visit's search (T223) leaves it
  const kept = await visit(PC, { hint: 16, remembered: 8 });
  assert.deepEqual([kept.log, kept.told, kept.startedAt], [["8 or 4: 8", "8 or 2: 2", "2 or 1: 2"], [2], 0], "a remembered 8 on the owner's PC: 2, on the first generation");
  const right = await visit(PC, { hint: 16, remembered: 2 });
  assert.deepEqual([right.log, right.told], [["2 or 1: 2", "2 or 4: 2"], [2]], "a remembered 2 there: its neighbours, two comparisons as before");
  // the same shape with a clear best, on 16, 8 and 4 logical cores
  assert.deepEqual((await visit(DIP, { hint: 16 })).log, ["16 or 8: 8", "8 or 4: 8", "8 or 2: 2", "2 or 1: 2"], "a dip, 16 logical cores");
  assert.deepEqual((await visit(upTo(DIP, 8), { hint: 8 })).log, ["8 or 4: 8", "8 or 2: 2", "2 or 1: 2"], "a dip, 8 logical cores");
  assert.deepEqual((await visit(DIP4, { hint: 4 })).log, ["4 or 2: 4", "4 or 1: 1"], "a dip, 4 logical cores");
  // two dips: the way down goes on by halves from a quarter that was faster, and a half that loses there has its quarter
  // too; a count that went down does not go up afterwards
  const twice = await visit({ 1: 4, 2: 9, 4: 5, 8: 9, 16: 6 }, { hint: 16 });
  assert.deepEqual([twice.log, twice.told], [["16 or 8: 16", "16 or 4: 4", "4 or 2: 4", "4 or 1: 1"], [1]], "two dips");
  const stays = await visit({ 1: 9, 2: 9, 4: 5, 8: 9, 16: 6 }, { hint: 16 });
  assert.deepEqual([stays.log, stays.told], [["16 or 8: 16", "16 or 4: 4", "4 or 2: 4", "4 or 1: 4"], [4]], "a count gone down to does not go up");
  // what it costs where there is no dip: one comparison (20 tokens) more where the best count is 4 or more, of a first
  // visit and of a visit that remembers it; none more where it is 1 or 2 (the Android's are above)
  for (const cores of [4, 8, 16]) {
    const ms = upTo(MORE, cores), want = [`${cores} or ${cores / 2}: ${cores}`, `${cores} or ${cores / 4}: ${cores}`, `${cores} or ${cores * 2}: ${cores}`];
    for (const remembered of [0, cores]) {
      const run = await visit(ms, { hint: cores, remembered });
      assert.deepEqual([run.log, run.told], [want, [cores]], `more threads faster, ${cores} logical cores${remembered ? ", remembered" : ""}: half, a quarter, twice`);
    }
  }
  assert.deepEqual((await visit(MORE, { hint: 16, remembered: 2 })).log, ["2 or 1: 2", "2 or 4: 4", "4 or 8: 8", "8 or 16: 16", "16 or 32: 16"], "the way up: by doubles, as before");
  // the PC's chain of visits with noise: 2 threads are faster than 8 by 8% there, 3% past the search's margin of 5%, so
  // a visit may keep 8; a later one leaves it, and none comes back from 2 (4 is slower than 2, and 8 is not tried from 2)
  seed = 239;  // (this chain's own run of random numbers, whatever ran before it)
  const lines = [];
  for (const [name, noise] of Object.entries({ "a token's time spread by 10%": { sigma: 0.1 }, "spread by 10%, a block of 5 slowed 3 times 1 in 10": { sigma: 0.1, slow: 0.1 } })) {
    const kept = {}, visits = 120;
    for (let v = 0, remembered = 0; v < visits; v++) {
      const next = (await visit(PC, { hint: 16, remembered, ...noise })).told.at(-1);
      kept[next] = (kept[next] ?? 0) + 1;
      remembered = next;
    }
    assert.ok((kept[2] ?? 0) >= 0.8 * visits, `the owner's PC, ${name}: 2 remembered by ${kept[2] ?? 0} of ${visits} visits`);
    lines.push(`${name}: ${Object.entries(kept).map(([count, n]) => `${count} by ${n}`).join(", ")} of ${visits} visits`);
  }
  console.log(`ok: T239, the owner's PC: from 16 ${JSON.stringify(pc.log)}, from a remembered 8 ${JSON.stringify(kept.log)}; ${lines.join("; ")}`);
}

// ---- T240: the GPU ready inside a generation. An int8 model (the GPU takes no other) of the same size, and a GPU's worker
// that is this test: ready when ready() is called, "ended" at once when stopped. A token's times are 4 threads' best
// while the GPU gets ready (as gpu-default-check.mjs's) and the owner's Android's after it
{
  const quantized = {};
  let at = 28;
  for (const [name, shape] of [["token_embedding_table", [vocab, dim]], ["rms_att_weight", [layers, dim]], ["wq", [layers, dim, dim]],
    ["wk", [layers, dim, dim]], ["wv", [layers, dim, dim]], ["wo", [layers, dim, dim]], ["rms_ffn_weight", [layers, dim]],
    ["w1", [layers, hidden, dim]], ["w2", [layers, dim, hidden]], ["w3", [layers, hidden, dim]], ["rms_final_weight", [dim]]]) {
    const count = shape.reduce((a, b) => a * b, 1);
    if (name.startsWith("rms")) {
      quantized[name] = { kind: "f32", offset: at, shape, group: 0, scales: 0 };
      at += count * 4;
    } else {
      quantized[name] = { kind: "int8", offset: at, shape, group: 32, scales: at + count };
      at += count + (count / 32) * 4;
    }
  }
  const bytes = at, cos = new Float32Array(seqLen * headSize / 2).fill(1), sin = new Float32Array(seqLen * headSize / 2);
  const int8Plan = { ...plan, int8: true, half_kv: true, tensors: quantized, derived: { freq_cis_real: cos, freq_cis_imag: sin } };
  const room = footprint([dim, hidden, layers, heads, heads, vocab, seqLen], bytes, { dtype: "int8", relaxed: false, halfKV: true, shared: true, gpu: true });
  const pages = Math.ceil((CONTROL_BYTES + bytes + room) / PAGE) + 1;
  const int8Memory = new WebAssembly.Memory({ initial: pages, maximum: pages + 1, shared: true });
  const BESIDE = { 1: 16, 2: 12, 4: 8, 8: 10 };
  const info = console.info;
  console.info = () => {};  // forward.js's lines about the GPU
  /** an engine with a GPU getting ready; ready(): the GPU's worker says it is; write(n): n tokens with logits */
  async function withGpu({ remembered = 0 } = {}) {
    let engine = null, starting = false, time = 0, said = null, isReady = false, pos = 0;
    const clock = () => {
      starting = !starting;
      if (!starting) time += (isReady ? ANDROID : BESIDE)[engine.threads] ?? 30;
      return time;
    };
    const gpu = () => ({ postMessage: (data) => { if (data.type === "stop") said({ data: { type: "ended" } }); },
      set onmessage(f) { said = f; }, set onerror(f) {}, terminate() {} });
    engine = createForward({ memory: int8Memory, base: CONTROL_BYTES, size: bytes, kernels: { plain: empty, relaxed: null, wide: false }, plan: int8Plan, spawn, clock, gpu,
      wrap: () => new Proxy({}, { get: (_, name) => (name === "then" ? undefined : () => 0) }) });
    assert.equal(engine.gpuWhyNot, null, "the made-up model is one the GPU takes");
    const told = [];
    await engine.findThreads({ from: 8, remembered, chose: (count) => told.push(count) });
    const ready = () => {
      isReady = true;
      said({ data: { type: "ready", adapter: "made up", key: "k", bytes: 1, seconds: 0, form: "made up", attention: "made up", forms: [], remembered: false, blocks: [] } });
    };
    const write = async (tokens) => {
      for (let t = 0; t < tokens; t++, pos = (pos + 1) % seqLen) {
        engine.forward(1, pos, true);
        if (t % 20 === 19) await Promise.resolve();
      }
    };
    const log = () => engine.searchLog.map(({ best, candidate, faster, whileGpu }) => `${best} or ${candidate}: ${faster ? candidate : best}${whileGpu ? " (GPU getting ready)" : ""}`);
    return { engine, told, ready, write, log };
  }
  // (1) the first search ended beside the GPU's getting ready: 4 threads, not remembered. The GPU is ready 30 tokens into
  // the next generation: the search begins at the next token, in that generation, and 2 is remembered
  {
    const { engine, told, ready, write, log } = await withGpu();
    for (let g = 0; g < 10 && engine.searching; g++) {
      engine.newGeneration();
      await write(20);
    }
    const first = log();
    assert.deepEqual([engine.searching, engine.threads, told, first], [false, 4, [], ["8 or 4: 4 (GPU getting ready)", "4 or 2: 4 (GPU getting ready)", "4 or 1: 4 (GPU getting ready)"]],
      "the first search beside the GPU's getting ready: 4 threads in use, none remembered");
    engine.newGeneration();
    await write(30);
    assert.deepEqual([engine.searching, log().length, engine.threads], [false, first.length, 4], "no search begins while the GPU gets ready, inside a generation either");
    ready();
    await write(1);
    assert.equal(engine.searching, true, "T240: the GPU ready inside a generation: the search begins at the next token, not at the next generation");
    await write(60);
    assert.deepEqual([engine.searching, engine.threads, told, log().slice(first.length)], [false, 2, [2], ["4 or 2: 2", "2 or 1: 2"]],
      "...and ends in that generation: 2 threads, remembered");
    await engine.release();
  }
  // (2) the GPU ready while the first search is under way: that search ends as it began (marked, not remembered), and the
  // next token begins the one that is remembered, with no generation begun between
  {
    const { engine, told, ready, write, log } = await withGpu();
    engine.newGeneration();
    await write(20);
    assert.deepEqual(log(), ["8 or 4: 4 (GPU getting ready)"]);
    ready();
    await write(200);
    assert.deepEqual([engine.searching, engine.threads, told, log()], [false, 2, [2],
      ["8 or 4: 4 (GPU getting ready)", "4 or 2: 2 (GPU getting ready)", "2 or 1: 2 (GPU getting ready)", "2 or 1: 2", "2 or 4: 2"]],
      "T240: a search that ends after the GPU is ready is run again from the next token on");
    await engine.release();
  }
  // (3) a count remembered from an earlier visit, its first generation begun while the GPU gets ready
  {
    const { engine, told, ready, write, log } = await withGpu({ remembered: 4 });
    engine.newGeneration();
    await write(30);
    assert.deepEqual([engine.searching, log()], [false, []], "a remembered count is not searched while the GPU gets ready");
    ready();
    await write(60);
    assert.deepEqual([engine.searching, engine.threads, told, log()], [false, 2, [2], ["4 or 2: 2", "2 or 1: 2"]], "T240: a remembered 4, the GPU ready inside its first generation: 2");
    await engine.release();
  }
  // (4) /benchmark/ begins no generation: the count the model page remembers is taken as it is (T190), GPU ready or not
  {
    const { engine, told, ready, write, log } = await withGpu({ remembered: 4 });
    await write(30);
    ready();
    await write(60);
    assert.deepEqual([engine.searching, engine.threads, told, log()], [false, 4, [], []], "no generation begun by the page: no search of a remembered count");
    await engine.release();
  }
  console.info = info;
  console.log("ok: T240, the GPU ready inside a generation: the search owed begins at the next token (after a first search beside the GPU, under one, of a remembered count), none where no generation began");
}

// ---- the page's key of the count: one a model, a device and a browser (the page and /benchmark/ read the same one)
{
  const device = { hardwareConcurrency: 8, deviceMemory: 8, userAgent: "made up" };
  assert.notEqual(threadsKey("llm-jp-3-150m", device), threadsKey("tiny-lm", device), "the count is remembered one a model");
  assert.notEqual(threadsKey("llm-jp-3-150m", device), threadsKey("llm-jp-3-150m", { ...device, hardwareConcurrency: 4 }), "...and one a number of logical cores");
  console.log("ok: the page's key of the count is one a model and a device");
}
