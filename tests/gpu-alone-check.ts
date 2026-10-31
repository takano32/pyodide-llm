// tests/gpu-alone-check.ts (T.450, with T.378.2): an engine of a model on the GPU alone (T.156's direct), made in Node:
// forward.js's own createForward() with kernels that do nothing, a made-up model's plan (the native Python places its
// tensors, as memory-check.ts has it) and a stand-in for the GPU's worker on this very thread, which answers a request
// in the control area as it is posted. What it holds is what such an engine does and no test outside CI's GPU jobs saw
// by what happens (T.378.1 held it by the syntax tree alone):
//   - no software thread is ever started for it, whatever count is asked or searched for, and its stopThreads() writes
//     nothing to the control area (the same model not on the GPU alone, beside it, starts them: the check sees threads);
//   - a prompt's blocks and a generation's steps go to the GPU (the tokens themselves: the GPU embeds them), and nothing
//     of them runs on the CPU's kernels;
//   - a token that would run on the CPU stops the GPU's side, tells the worker once (onLost) and throws.
//
//   node tests/gpu-alone-check.ts [--forward <another forward.js, to see a broken one fail>]     (PYTHON=.venv/bin/python)
import assert from "node:assert/strict";
import path from "node:path";
import { workers, endedThen, READY } from "./fakes/worker.ts";
import { plans } from "./plans.ts";
import { runtime, runtimeUrl } from "./tree.ts";

const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : runtime("forward.js");
const { createForward, GPU_BLOCK, GPU_TOKENS } = await import(forwardFile);
const { CONTROL_BYTES, QUIT, GPU_DONE, GPU_FAILED } = await import(runtimeUrl("jobs.ts"));
const { plansOf, planOf, FORM, empty } = plans(new URL("..", import.meta.url));

const PAGE = 65536;
const [p] = plansOf([{ header: [64, 128, 2, 4, 4, 100, 128], form: FORM, dtype: "int8" }]);
const plan = planOf(p);
console.info = () => {};
console.warn = () => {};

/** An engine of the made-up model and what stands beside it. direct: on the GPU alone */
function made(direct, plan = planOf(p), size = p.size, gpuForce = {}) {
  const memory = new WebAssembly.Memory({ initial: Math.ceil((CONTROL_BYTES + size) / PAGE) + 64, maximum: Math.ceil((CONTROL_BYTES + size) / PAGE) + 64, shared: true });
  const ctl = new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4);
  const beside = { ctl, spawned: 0, ended: 0, kernels: 0, lost: [] };
  const spawn = (data) => {
    beside.spawned += 1;
    beside.started = data;
    return Promise.resolve({ terminate() { beside.ended += 1; } });
  };
  // the GPU's worker: ready as soon as it is started (with a generation's steps), and every request answered at once
  const { factory: gpu, posted } = workers({ answer: endedThen((data, worker) => {
    if (data.type === "start") {
      queueMicrotask(() => worker.say(READY({ blocks: [{ count: 16, ms: 1 }, { count: 64, ms: 2 }], tokens: { form: "fused", attention: "vec", ms: 1, forms: [] } })));
    } else {
      if (data.type === "tokens") new Int32Array(memory.buffer, worker.posted[0].plan.tokens?.ids, 2 + GPU_TOKENS).set([data.count, ...Array.from({ length: GPU_TOKENS }, (_, i) => 10 + i), 0]);
      Atomics.store(ctl, GPU_FAILED, 0);
      Atomics.store(ctl, GPU_DONE, data.serial);
    }
  }) });
  beside.posted = posted;
  // kernels that do nothing, counted: none is called for a model on the GPU alone once it is made
  // (finite_f16 says yes: what the GPU wrote back is looked at by that kernel, and a stand-in that writes nothing wrote zeros)
  const wrap = () => new Proxy({}, { get: (_, name) => () => { beside.kernels += 1; return name === "finite_f16" ? 1 : 0; } });
  const engine = createForward({ memory, base: CONTROL_BYTES, size, kernels: { plain: empty, relaxed: empty, wide: false }, plan, spawn, gpu, wrap, gpuForce,
    direct: direct ? { onLost: (why) => beside.lost.push(why) } : undefined });
  return { engine, beside };
}
const tokens = (count) => Array.from({ length: count }, (_, i) => i % 100);

// ---- the model on the GPU alone
{
  const { engine, beside } = made(true);
  assert.equal(await engine.gpu, "prompts and answers on WebGPU");
  assert.equal(beside.posted[0].plan.direct, true, "the GPU's worker is started for a model it alone holds");
  beside.kernels = 0;
  const chosen = [];
  assert.equal(await engine.setThreads(4), 1, "a count asked for: one");
  assert.equal(await engine.findThreads({ from: 8, remembered: 4, chose: (count) => chosen.push(count) }), 1, "a count searched for: one");
  engine.newGeneration();
  assert.deepEqual([engine.threads, engine.searching, engine.lostThreads, beside.spawned, chosen], [1, false, false, 0, []], "no software thread is started for a model on the GPU alone");
  assert.deepEqual([engine.promptBlock, engine.tokenBlock], [GPU_BLOCK, GPU_TOKENS]);
  engine.forwardMany(tokens(100), 0);
  const blocks = beside.posted.filter((data) => data.type === "prompt");
  assert.deepEqual(blocks.map(({ count, pos, tokens: handed }) => [count, pos, handed.length]), [[64, 0, 64], [36, 64, 36]], "a prompt's blocks go to the GPU, the tokens themselves");
  assert.deepEqual(engine.generateMany(5, 100, [1, 2], 2, 4, 0, 0.9, 1, [], []), [10, 11, 12, 13], "and a generation's steps");
  assert.deepEqual([engine.gpuTokens, engine.gpuSampled, beside.kernels], [100, 4, 0], "nothing of them runs on the CPU's kernels");
  engine.stopThreads();
  assert.deepEqual([Atomics.load(beside.ctl, QUIT), beside.ended], [0, 0], "stopThreads() has no thread to stop and writes nothing");
  // a token on the CPU: there is nothing to run it with
  assert.throws(() => engine.forward(5, 104), /this model's weights were on it alone/);
  assert.throws(() => engine.forwardMany(tokens(16), 104), /this model's weights were on it alone/);
  assert.deepEqual([beside.lost, beside.kernels, engine.gpuWhyNot], [["the GPU did not take a block or a step"], 0, "the GPU did not take a block or a step"], "the worker is told once that the GPU is lost");
  assert.equal(await engine.release(), true);
  assert.deepEqual([beside.spawned, Atomics.load(beside.ctl, QUIT)], [0, 0]);
}
console.log("ok: T.450, an engine on the GPU alone starts no software thread whatever is asked, sends its blocks and steps to the GPU, and is lost where a token would run on the CPU");

// ---- T.558.3 (T.261.4.1's review): fromStaging() into a ring, by what the GPU's worker writes into the staging place. A
// made-up Gemma 3 of three layers (plans.ts: the first two attend a window of 8, the last all positions) in a context
// of 64, so that the two local layers keep their keys and values in a ring of 24 (T.510.1: each position at p mod 24 and
// p mod 24 + 24); its blocks go to a GPU that writes known float16s for every position, layer and column: 48 positions,
// then 10. What the cache holds after (read through keysAndValues(), and where a step's request says the rings are) is
// those, both copies of a ring alike over the ring's last 24 positions, and the ring knows the last position written
{
  // (heads of 32: the int8 kernels' groups; a key head for every head, so that the cache keeps float16s, which go in as
  // they are: a float32 cache widens them by a kernel, which does nothing here)
  const [g] = plansOf([{ header: [64, 128, 3, 4, 4, 100, 64], form: { ...FORM, arch: "gemma3", head_dim: 32 }, dtype: "int8" }]);
  const plan = planOf(g), kvDim = 128, layers = 3, ring = 24;
  const memory = new WebAssembly.Memory({ initial: Math.ceil((CONTROL_BYTES + g.size) / PAGE) + 64, maximum: Math.ceil((CONTROL_BYTES + g.size) / PAGE) + 64, shared: true });
  const ctl = new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4);
  // (a value of position, layer and column: integers under 2048, which a float16 holds; a value's the negative)
  const key = (pos, l, i) => pos + 64 * l + 192 * (i % 10);
  const half = (value) => { const view = new DataView(new ArrayBuffer(2)); view.setFloat16(0, value, true); return view.getUint16(0, true); };
  const { factory: gpu, posted } = workers({ answer: endedThen((data, worker) => {
    if (data.type === "start") {
      queueMicrotask(() => worker.say(READY({ blocks: [{ count: 16, ms: 1 }, { count: 64, ms: 2 }], tokens: { form: "fused", attention: "vec", ms: 1, forms: [] } })));
      return;
    }
    // (a block's, and a step's: they come back through the same place)
    if (data.type === "prompt" || data.type === "tokens") {
      const staging = new Uint16Array(memory.buffer, worker.posted[0].plan.staging, 2 * layers * GPU_BLOCK * kvDim);
      for (let l = 0; l < layers; l++) {
        for (let t = 0; t < data.count; t++) {
          for (let i = 0; i < kvDim; i++) {
            staging[(l * GPU_BLOCK + t) * kvDim + i] = half(key(data.pos + t, l, i));
            staging[((layers + l) * GPU_BLOCK + t) * kvDim + i] = half(-key(data.pos + t, l, i));
          }
        }
      }
    }
    if (data.type === "tokens") new Int32Array(memory.buffer, worker.posted[0].plan.tokens?.ids, 2 + GPU_TOKENS).set([data.count, ...Array.from({ length: GPU_TOKENS }, (_, i) => 10 + i), 0]);
    Atomics.store(ctl, GPU_FAILED, 0);
    Atomics.store(ctl, GPU_DONE, data.serial);
  }) });
  const wrap = () => new Proxy({}, { get: (_, name) => () => (name === "finite_f16" ? 1 : 0) });
  const engine = createForward({ memory, base: CONTROL_BYTES, size: g.size, kernels: { plain: empty, relaxed: empty, wide: false }, plan, spawn: () => Promise.resolve({ terminate() {} }),
    gpu, wrap, gpuForce: { always: true } });
  assert.match(await engine.gpu, /^prompts (and answers )?on WebGPU/);
  engine.newGeneration();
  engine.forwardMany(tokens(48), 0);
  engine.forwardMany(tokens(10), 48);
  assert.deepEqual(posted.filter((data) => data.type === "prompt").map(({ pos, count }) => [pos, count]), [[0, 48], [48, 10]], "the two blocks went to the GPU");
  // through the cache's reader: the last 24 positions of the local layers, every position of the last
  const { keys, values } = engine.keysAndValues(0, 58);
  for (let l = 0; l < layers; l++) {
    for (let pos = l < 2 ? 58 - ring : 0; pos < 58; pos++) {
      for (let i = 0; i < kvDim; i++) {
        const at = (l * 58 + pos) * kvDim + i;
        assert.ok(keys[at] === key(pos, l, i) && values[at] === -key(pos, l, i), `layer ${l}, position ${pos}, column ${i}: ${keys[at]} and ${values[at]}`);
      }
    }
  }
  // and where a step's request says the rings are: both copies of every position the ring holds
  engine.generateMany(5, 58, [1, 2], 2, 4, 0, 0.9, 1, [], []);
  const asked = posted.filter((data) => data.type === "tokens").at(-1).cache;
  assert.deepEqual([asked.places.map((place) => place.ring), asked.half], [[ring, ring, 0], true], "the local layers' rings, of 24, of float16s");
  const read = (at) => (asked.half ? new Uint16Array(memory.buffer, at, kvDim) : new Float32Array(memory.buffer, at, kvDim));
  // (after the four steps, positions 58 to 61: the ring's last 24 positions are 38 to 61)
  for (const l of [0, 1]) {
    const place = asked.places[l];
    for (let pos = 62 - ring; pos < 62; pos++) {
      const slot = pos % ring;
      for (const [base, sign] of [[place.keys, 1], [place.values, -1]]) {
        const first = [...read(base + slot * asked.row)], second = [...read(base + (slot + ring) * asked.row)];
        assert.deepEqual(second, first, `layer ${l}, position ${pos}: the ring's two copies`);
        assert.deepEqual(first, Array.from({ length: kvDim }, (_, i) => (asked.half ? half(sign * key(pos, l, i)) : sign * key(pos, l, i))), `layer ${l}, position ${pos}: what the GPU wrote`);
      }
    }
  }
  // (the ring's last position written, which the CPU's attention reads: a position the ring no longer holds is refused)
  assert.throws(() => engine.forward(5, 20), /too far back after 61/);
  assert.equal(await engine.release(), true);
}
console.log("ok: T.558.3, fromStaging() writes a block's keys and values into a window's ring at both of its places, and the ring's last position written");

// ---- the same model with its weights in the memory: the threads are started (what the check above would see)
{
  const { engine, beside } = made(false);
  await engine.gpu;
  assert.equal(await engine.setThreads(4), 4);
  assert.deepEqual([engine.threads, beside.spawned], [4, 3], "the same model not on the GPU alone starts its threads");
  engine.stopThreads();
  assert.deepEqual([Atomics.load(beside.ctl, QUIT), beside.ended, engine.threads], [1, 3, 1], "and stops them through the control area");
  assert.deepEqual([beside.started.plain, beside.started.relaxed, typeof beside.started.share], [empty, empty, "number"], "a thread is started with the two kernels this model runs on");
  assert.equal(await engine.release(), true);
  // T.378.2: the kernels a thread starts with are chosen by the engine: no relaxed one where the model does not run on it
  const plain = made(false, planOf(p, { relaxed: false }));
  assert.equal(await plain.engine.setThreads(2), 2);
  assert.deepEqual([plain.beside.started.plain, plain.beside.started.relaxed], [empty, null], "a model that does not run on relaxed SIMD starts its threads without that kernel");
  assert.equal(await plain.engine.release(), true);
}
console.log("ok: the same model in the memory starts the threads it is asked for, with the kernels it runs on");

// ---- T.375.2: what the GPU's worker is told of a model's layers is the plan's facts, and no architecture's name: how
// many values of a head RoPE turns (none where the plan's layers say it turns none), the kind of the norms, and the
// vectors that go with them
{
  const header = [64, 128, 2, 4, 4, 100, 128];
  const models = plansOf([{}, { arch: "gpt2" }, { arch: "neox" }].map((form) => ({ header, form: { ...FORM, ...form }, dtype: "int8" })));
  const told = [];
  for (const placed of models) {
    const { engine, beside } = made(false, planOf(placed), placed.size);
    await engine.gpu;
    const [{ plan: started }] = beside.posted;
    told.push([placed.form.arch, started.turned, started.layerNorm, Object.keys(started.vectors).join(" ")]);
    assert.equal(await engine.release(), true);
  }
  assert.deepEqual(told, [["llama", 16, false, "attention ffn"],
    ["gpt2", 0, true, "attention ffn attentionBias ffnBias bq bk bv bo b1 b2"],
    ["neox", 16, true, "attention ffn attentionBias ffnBias bq bk bv bo b1 b2"]]);
  // (T.375.5, T.484: a plan whose norms and FFN are no pair the GPU's layers are of, RMSNorms with GPT-2's FFN or
  // LayerNorms with a gated one: the CPU's engine runs it by its facts, and the GPU's worker is not started to find out)
  const [llama, gpt2] = models;
  for (const [plan, size] of [[{ ...planOf(gpt2), layer_norm: false }, gpt2.size], [{ ...planOf(llama), layer_norm: true }, llama.size]]) {
    const odd = made(false, plan, size);
    await odd.engine.gpu;
    assert.deepEqual([odd.beside.posted.length, odd.engine.gpuWhyNot], [0, "this pair of norms and FFN is not on the GPU yet"]);
    assert.equal(await odd.engine.release(), true);
  }
  const still = made(false, { ...planOf(llama), layers: planOf(llama).layers.map((layer) => ({ ...layer, rope: false })) }, llama.size);
  await still.engine.gpu;
  assert.equal(still.beside.posted[0].plan.turned, 0, "a model RoPE turns no layer of: no value of a head is turned on the GPU");
  assert.equal(await still.engine.release(), true);
}
console.log("ok: T.375.2, the GPU's worker is told how much of a head RoPE turns and the kind of the norms from the plan's facts");

// ---- T.375.3 (the review): the cache of the keys and values is the last thing in the memory, below it the places the
// GPU answers into, and the GPU is told where its keys and values are as they are now, after the cache grew. A cache
// placed before the staging place, or addresses told as they were when the engine was made, leaves the GPU reading and
// writing over what is not its own (it passed the whole light suite, and only gpu-default-check saw it)
{
  const { engine, beside } = made(false, planOf(p, { kvStart: 16 }), p.size, { always: true });
  await engine.gpu;
  const [{ plan: started }] = beside.posted;
  engine.forwardMany(tokens(20), 0);
  assert.ok(engine.gpuTokens >= 20, "the prompt went through the GPU");
  engine.generateMany(5, 20, [1, 2], 2, 4, 0, 0.9, 1, [], []);
  engine.forwardMany(tokens(80), 24);
  engine.generateMany(5, 104, [1, 2], 2, 4, 0, 0.9, 1, [], []);
  const [first, last] = beside.posted.filter((data) => data.type === "tokens").map(({ cache }) => cache);
  assert.ok(first && last, "the steps were asked of the GPU twice, with the cache's places");
  // (T.261.4.2: each layer's place, its block a capacity of positions after the one before it)
  const layers = 2, capacityOf = ({ places: [one, two] }) => (two.keys - one.keys) / last.row;
  assert.equal(capacityOf(first), 32, "the cache had grown to twice its start");
  assert.equal(capacityOf(last), 128, "and to the context after the larger prompt");
  assert.deepEqual(last.places.map((place) => place.ring), [0, 0], "no layer's is a ring");
  assert.equal(last.places[0].values, last.places[0].keys + layers * capacityOf(last) * last.row, "the values begin where the keys end, at the capacity the GPU is told");
  assert.notEqual(last.places[0].values, first.places[0].values, "the values moved as the cache grew, and the GPU is told where they are now");
  assert.ok(last.places[0].keys > Math.max(started.staging, started.rows, started.tokens.ids), "the cache is placed after what the GPU answers into, where it can grow");
  assert.equal(await engine.release(), true);
}
console.log("ok: T.375.3, the cache of the keys and values grows in place after the GPU's places, and the GPU is told where it is as it is now");

// ---- T.375.4: a model on the GPU alone is told where each layer's values and scales are in the checkpoint: a layer is
// its rows of the packing's bytes (jobs.ts's PACKED: int8 a byte a weight, ternary a quarter; T.533: its scales float16s) after the one before
for (const dtype of ["int8", "ternary"]) {
  const [q] = plansOf([{ header: [128, 256, 2, 4, 4, 100, 128], form: FORM, dtype }]);
  const plan = planOf(q), { engine, beside } = made(true, plan, q.size), ternary = dtype === "ternary";
  await engine.gpu;
  const said = beside.posted[0].plan.matrices;
  assert.deepEqual(Object.keys(said), ["wq", "wk", "wv", "wo", "w1", "w2", "w3"], dtype);
  for (const [name, m] of Object.entries(said)) {
    const t = plan.tensors[name], [rows, n] = t.shape.slice(-2);
    assert.equal(t.kind, dtype);
    assert.deepEqual(m, { rows, n, packing: dtype, layers: [0, 1].map((l) => [t.offset + l * rows * (ternary ? n / 4 : n), t.scales + l * rows * (n / t.group) * (ternary ? 2 : 4)]),
      ...(name.startsWith("w") && "qkvo".includes(name[1]) ? { of: "attention" } : {}) }, `${dtype}: ${name}`);  // (T.545: the attending layers' matrices say so)
  }
  assert.equal(await engine.release(), true);
}
console.log("ok: T.375.4, a model on the GPU alone, of int8 or of ternary weights: the GPU is told where every layer's values and scales are, a layer's rows of the packing's bytes apart");
// T.486: a model of float32 weights is kept off the GPU alone before it is loaded (gpuOnlyUnfit()); an engine made of one
// on it anyway says which matrix, and is not read as a packing it has not ("Cannot read properties of null")
{
  const [f] = plansOf([{ header: [64, 128, 2, 4, 4, 100, 128], form: FORM, dtype: "float32" }]);
  assert.throws(() => made(true, planOf(f), f.size), /The matrix wq is of f32, which is not on the GPU alone\./);
}
console.log("ok: T.486, an engine of float32 weights on the GPU alone says which matrix the GPU does not take");
process.exit(0);
