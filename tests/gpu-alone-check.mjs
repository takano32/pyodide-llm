// tests/gpu-alone-check.mjs (T450, with T378.2): an engine of a model on the GPU alone (T156's direct), made in Node:
// forward.js's own createForward() with kernels that do nothing, a made-up model's plan (the native Python places its
// tensors, as memory-check.mjs has it) and a stand-in for the GPU's worker on this very thread, which answers a request
// in the control area as it is posted. What it holds is what such an engine does and no test outside CI's GPU jobs saw
// by what happens (T378.1 held it by the syntax tree alone):
//   - no software thread is ever started for it, whatever count is asked or searched for, and its stopThreads() writes
//     nothing to the control area (the same model not on the GPU alone, beside it, starts them: the check sees threads);
//   - a prompt's blocks and a generation's steps go to the GPU (the tokens themselves: the GPU embeds them), and nothing
//     of them runs on the CPU's kernels;
//   - a token that would run on the CPU stops the GPU's side, tells the worker once (onLost) and throws.
//
//   node tests/gpu-alone-check.mjs [--forward <another forward.js, to see a broken one fail>]     (PYTHON=.venv/bin/python)
import assert from "node:assert/strict";
import path from "node:path";
import { plans } from "./plans.mjs";
import { runtime, runtimeUrl } from "./tree.mjs";

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
function made(direct, plan = planOf(p), size = p.size) {
  const memory = new WebAssembly.Memory({ initial: Math.ceil((CONTROL_BYTES + size) / PAGE) + 64, maximum: Math.ceil((CONTROL_BYTES + size) / PAGE) + 64, shared: true });
  const ctl = new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4);
  const beside = { ctl, spawned: 0, ended: 0, kernels: 0, posted: [], lost: [] };
  const spawn = (data) => {
    beside.spawned += 1;
    beside.started = data;
    return Promise.resolve({ terminate() { beside.ended += 1; } });
  };
  // the GPU's worker: ready as soon as it is started (with a generation's steps), and every request answered at once
  const gpu = () => {
    let listen, ids;
    return { set onmessage(handler) { listen = handler; }, set onerror(handler) {}, terminate() {},
      postMessage(data) {
        beside.posted.push(data);
        if (data.type === "start") {
          ids = data.plan.tokens?.ids;
          queueMicrotask(() => listen({ data: { type: "ready", adapter: "a stand-in", bytes: 1e6, seconds: 0.1, form: "tiles", attention: "flash", key: "k", forms: [],
            blocks: [{ count: 16, ms: 1 }, { count: 64, ms: 2 }], tokens: { form: "fused", attention: "vec", ms: 1, forms: [] } } }));
        } else if (data.type === "stop") {
          queueMicrotask(() => listen({ data: { type: "ended" } }));
        } else {
          if (data.type === "tokens") new Int32Array(memory.buffer, ids, 2 + GPU_TOKENS).set([data.count, ...Array.from({ length: GPU_TOKENS }, (_, i) => 10 + i), 0]);
          Atomics.store(ctl, GPU_FAILED, 0);
          Atomics.store(ctl, GPU_DONE, data.serial);
        }
      } };
  };
  // kernels that do nothing, counted: none is called for a model on the GPU alone once it is made
  const wrap = () => new Proxy({}, { get: () => () => { beside.kernels += 1; return 0; } });
  const engine = createForward({ memory, base: CONTROL_BYTES, size, kernels: { plain: empty, relaxed: empty, wide: false }, plan, spawn, gpu, wrap,
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
  assert.throws(() => engine.forward(5, 104), (error) => error.directLost === true && /this model's weights were on it alone/.test(error.message));
  assert.throws(() => engine.forwardMany(tokens(16), 104), /this model's weights were on it alone/);
  assert.deepEqual([beside.lost, beside.kernels, engine.gpuWhyNot], [["the GPU did not take a block or a step"], 0, "the GPU did not take a block or a step"], "the worker is told once that the GPU is lost");
  assert.equal(await engine.release(), true);
  assert.deepEqual([beside.spawned, Atomics.load(beside.ctl, QUIT)], [0, 0]);
}
console.log("ok: T450, an engine on the GPU alone starts no software thread whatever is asked, sends its blocks and steps to the GPU, and is lost where a token would run on the CPU");

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
  // T378.2: the kernels a thread starts with are chosen by the engine: no relaxed one where the model does not run on it
  const plain = made(false, planOf(p, { relaxed: false }));
  assert.equal(await plain.engine.setThreads(2), 2);
  assert.deepEqual([plain.beside.started.plain, plain.beside.started.relaxed], [empty, null], "a model that does not run on relaxed SIMD starts its threads without that kernel");
  assert.equal(await plain.engine.release(), true);
}
console.log("ok: the same model in the memory starts the threads it is asked for, with the kernels it runs on");

// ---- T375.2: what the GPU's worker is told of a model's layers is the plan's facts, and no architecture's name: how
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
  // (a plan whose facts are no architecture's pair: the GPU's worker is told each as it is)
  const [llama, gpt2] = models;
  const odd = made(false, { ...planOf(gpt2), layer_norm: false }, gpt2.size);
  await odd.engine.gpu;
  assert.deepEqual([odd.beside.posted[0].plan.layerNorm, odd.beside.posted[0].plan.turned, Object.keys(odd.beside.posted[0].plan.vectors).join(" ")], [false, 0, "attention ffn bq bk bv bo b1 b2"]);
  assert.equal(await odd.engine.release(), true);
  const still = made(false, { ...planOf(llama), layers: planOf(llama).layers.map((layer) => ({ ...layer, rope: false })) }, llama.size);
  await still.engine.gpu;
  assert.equal(still.beside.posted[0].plan.turned, 0, "a model RoPE turns no layer of: no value of a head is turned on the GPU");
  assert.equal(await still.engine.release(), true);
}
console.log("ok: T375.2, the GPU's worker is told how much of a head RoPE turns and the kind of the norms from the plan's facts");
process.exit(0);
