// tests/gpu-part-check.mjs (T378.2): the GPU's side of the engine (src/runtime/forward/gpuside.js's gpuSide() and
// gpuFit()) by itself, with stand-ins for everything beside it: two pages of shared memory of its own, a "GPU's worker"
// on this very thread that answers a request in the control area as it is posted (so that the wait finds its answer
// there), a clock the test moves (a block on the stand-in GPU and a block on the stand-in CPU each take what the test
// says), a CPU's side that writes down what it was asked for, and the software threads as two numbers. No engine, no
// kernels, no model, no Pyodide, no GPU. What the part is still bound to: jobs.js (the words of the control area),
// control.js (the wait on one, the real one here), choice.js (the times of either side and what the status line says)
// and alone.js (a model on the GPU alone's verdict).
//
// The state has one owner (T378.2): the rest of the engine tells it what happened (generationBegan, prompt, cpuWrote,
// cpuStepBegan and cpuStepEnded, generateMany, put, notHere, release), asks it (gettingReady, promptBlock, tokenBlock,
// keysAndValues) and reads what the page is shown. Held here:
//   - each of those does what it says, seen in what the GPU's worker is posted, what the CPU's side is asked and what is
//     read of the part afterwards;
//   - which blocks of a prompt and which steps go to the GPU: none until the CPU is timed, then by the times, the side
//     not chosen again every GPU_RECHECK generations, never while the threads are searched;
//   - every way a request is refused (no answer, a failure, keys and values or logits that are no finite numbers, an id
//     outside the vocabulary) stops the GPU, says why, and leaves the block or step to the CPU;
//   - a model on the GPU alone: nothing of it runs on the CPU, and what the GPU does not take is lost, said once;
//   - what gpuFit() says of a model: why the GPU would not take it, and the vectors it reads.
//
//   node tests/gpu-part-check.mjs        (Node only, about a second)
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";

const { gpuSide, gpuFit } = await import(runtimeUrl("forward/gpuside.js"));
const { controlArea } = await import(runtimeUrl("forward/control.js"));
const { BATCH, CONTROL_BYTES, GPU_DONE, GPU_FAILED, GPU_BEAT, GPU_WANTED } = await import(runtimeUrl("jobs.ts"));
const { GPU_BLOCK, GPU_TOKENS, GPU_RECHECK, TOKEN_RECHECK, STOPS_MOST, PROMPTS_UNTIMED, PROMPTS_GPU, PROMPTS_CPU } = await import(runtimeUrl("forward/choice.js"));
const { OUTSIDE_VOCABULARY } = await import(runtimeUrl("forward/alone.js"));

const PAGE = 65536;
// a made-up model: small, and no two of its sizes the same where one could stand for another
const SHAPE = { dim: 8, hidden: 24, layers: 2, heads: 2, kvHeads: 1, headSize: 4, kvDim: 4, qDim: 8, vocab: 100, seqLen: 512, D: 32, KV: 8, halfKV: true,
  eps: 1e-5, rotary: 0, kvStart: 256, turning: true, layerNorm: false, parallel: false, unturned: [], linear: null, convolution: null, rotated: 0, sharedClassifier: true };
const matrixOf = (rows, n, at, more = {}) => ({ rows, n, int8: true, six: false, group: 32, layer: (l) => [at + l * 1000, at + l * 1000 + 500, at + l * 1000 + 700], ...more });
const WEIGHTS = { base: 8192, embedding: { kind: "int8", group: 32, offset: 100, scales: 200 },
  wq: matrixOf(8, 8, 30000), wk: matrixOf(4, 8, 32000), wv: matrixOf(4, 8, 34000), wo: matrixOf(8, 8, 36000),
  w1: matrixOf(24, 8, 38000), w2: matrixOf(8, 24, 40000), w3: matrixOf(24, 8, 42000), wcls: matrixOf(100, 8, 44000),
  attW: 50000, ffnW: 50100, attB: 0, ffnB: 0, finalW: 50200, finalB: 0, bo: 0, b1: 0, b2: 0, bq: 0, bk: 0, bv: 0, qNorm: 0, kNorm: 0,
  cosTable: 51000, sinTable: 52000, positions: 0, channels: [] };
const STAGING = CONTROL_BYTES, ROWS = STAGING + 2 * SHAPE.layers * GPU_BLOCK * SHAPE.kvDim * 2, IDS = ROWS + GPU_BLOCK * SHAPE.D;
assert.ok(IDS + (2 + GPU_TOKENS) * 4 <= 2 * PAGE, "the places the GPU answers into fit this test's memory");

const quiet = { info: [], warn: [] };
console.info = (...words) => quiet.info.push(words.join(" "));
console.warn = (...words) => quiet.warn.push(words.join(" "));
const said = () => [...quiet.info.splice(0), ...quiet.warn.splice(0)];
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

/** The part and its stand-ins. gpu: false for a worker without WebGPU. What the stand-in GPU does with a request is
 * beside.answer (see below); beside.cpuMs and beside.gpuMs are what a token of a block takes on either side. */
function made({ gpu = true, direct, whyNot = null, tokensWhyNot = null, steps = true, stalledMs = 60, shape = {}, force } = {}) {
  const memory = new WebAssembly.Memory({ initial: 2, maximum: 2, shared: true });
  const control = controlArea({ memory, sharedMemory: true, stalledMs });
  const words = new Int32Array(memory.buffer, IDS, 2 + GPU_TOKENS), halves = new Uint16Array(memory.buffer);
  const beside = { memory, ctl: control.ctl, now: 1000, made: 0, posted: [], terminated: 0, worker: null, ran: [], grown: [], embedded: [], staged: [], views: 0,
    finite: true, capacity: 256, cpuMs: 1, gpuMs: (count) => 4 + count / 16, stepMs: 1, threads: { count: 1, searching: false },
    // what the stand-in GPU answers a request with: silent (no answer), failed, ended (says "ended" after a stop), and
    // for the steps: sampled (how many, by default all asked), ids (by default 500 + position), notFinite
    answer: { silent: false, failed: false, ended: true } };
  const factory = () => {
    beside.made += 1;
    const worker = { handlers: {},
      set onmessage(handler) { this.handlers.message = handler; },
      set onerror(handler) { this.handlers.error = handler; },
      say(data) { this.handlers.message({ data }); },
      terminate() { beside.terminated += 1; },
      postMessage(data) {
        beside.posted.push(data);
        const { answer } = beside;
        if (data.type === "stop") {
          if (answer.ended) queueMicrotask(() => this.say({ type: "ended" }));
          return;
        }
        if (data.type === "start" || answer.silent) return;
        assert.equal(Atomics.load(beside.ctl, GPU_WANTED), data.serial, "a request is posted with the serial the control area says is wanted");
        if (data.type === "prompt") beside.now += beside.gpuMs(data.count);
        if (data.type === "tokens") {
          const sampled = answer.sampled ?? data.count;
          words.fill(0);
          words[0] = sampled;
          for (let i = 0; i < Math.max(0, Math.min(sampled, GPU_TOKENS)); i++) words[1 + i] = answer.ids?.[i] ?? (500 + data.pos + i) % SHAPE.vocab;
          words[1 + GPU_TOKENS] = answer.notFinite ? 1 : 0;
          beside.now += beside.stepMs * data.count;
        }
        if (data.type === "keys") {
          // every half says where it is: 0x3c00 (1.0) + side, layer and position, read back below
          for (let side = 0; side < 2; side++) for (let l = 0; l < SHAPE.layers; l++) for (let t = 0; t < data.count; t++) for (let i = 0; i < SHAPE.kvDim; i++) {
            halves[(STAGING + ((side * SHAPE.layers + l) * GPU_BLOCK + t) * SHAPE.kvDim * 2) / 2 + i] = 0x4000 + side * 0x400;  // 2.0 and 4.0
          }
        }
        Atomics.store(beside.ctl, GPU_FAILED, answer.failed ? 1 : 0);
        Atomics.add(beside.ctl, GPU_BEAT, 1);
        Atomics.store(beside.ctl, GPU_DONE, data.serial);
      } };
    beside.worker = worker;
    return worker;
  };
  const cpuSide = { memory,
    views() { beside.views += 1; },
    grow(pos) { beside.grown.push(pos); beside.capacity = Math.max(2 * beside.capacity, pos + 1); },
    embed(tokens, pos0, rows, stride) { beside.embedded.push([tokens.length, pos0, rows, stride]); },
    fromStaging(pos, count) { beside.staged.push([pos, count]); },
    stagingFinite(count) { return beside.finite; },
    // (as the engine's run(): it says the CPU wrote from there, or that a model on the GPU alone cannot run here)
    run(tokens, pos0, needLogits) {
      if (direct) throw part.notHere();
      part.cpuWrote(pos0);
      beside.ran.push([tokens.length, pos0, needLogits]);
      beside.now += beside.cpuMs * tokens.length;
    },
    get H() { return halves; },
    get capacity() { return beside.capacity; },
    get keys() { return 60000; },
    get values() { return 61000; } };
  const model = { ...SHAPE, ...shape };
  const fit = gpuFit({ shape: model, weights: WEIGHTS, device: { gpuRoom: undefined, memoryUnsaid: false, sharedMemory: true }, direct });
  const taken = { matrices: fit.matrices, vectors: fit.vectors, whyNot, tokensWhyNot, staging: whyNot ? 0 : STAGING, rows: whyNot || direct ? 0 : ROWS,
    ids: whyNot || tokensWhyNot ? 0 : IDS };
  const part = gpuSide({ shape: model, weights: WEIGHTS, taken, cpuSide, control, threads: beside.threads,
    options: { gpu: gpu ? factory : undefined, remembered: { key: "kept" }, force, direct, stalledMs, now: () => beside.now } });
  // the GPU says it is ready: with its timed blocks (16 and 64 tokens), and the steps where it made them
  beside.ready = (more = {}) => beside.worker.say({ type: "ready", adapter: "a stand-in", bytes: 2e6, seconds: 1.5, form: "tiles", attention: "flash", key: "the key",
    remembered: false, forms: [{ name: "tiles", ms: 1 }], blocks: [{ count: 16, ms: beside.gpuMs(16) }, { count: 64, ms: beside.gpuMs(64) }],
    ...(steps ? { tokens: { form: "fused", attention: "vec", pieces: 1, ms: beside.stepMs, forms: [{ name: "fused", ms: 1 }], attentions: [] } } : {}), ...more });
  return { part, beside };
}
const requests = (beside, type) => beside.posted.filter((data) => data.type === type);
const tokensOf = (count, from = 0) => Array.from({ length: count }, (_, i) => (from + i) % SHAPE.vocab);
// a generation's prompt as Python hands it over: a block of promptBlock() at a time
const feed = (part, count) => {
  part.generationBegan();
  const list = tokensOf(count);
  for (let at = 0; at < count;) {
    const block = list.slice(at, at + part.promptBlock());
    part.prompt(block, at);
    at += block.length;
  }
};
// what a generation's prompt went to: [tokens on the CPU, tokens on the GPU]
const sides = (beside) => {
  const cpu = beside.ran.splice(0).reduce((sum, [count]) => sum + count, 0), gpu = requests(beside, "prompt").reduce((sum, data) => sum + data.count, 0);
  beside.posted = beside.posted.filter((data) => data.type !== "prompt");
  return [cpu, gpu];
};
// a step on the CPU as the engine's forward() runs one
const cpuStep = (part, beside, ms) => {
  if (!part.cpuStepBegan(true)) return false;
  part.cpuStepEnded(ms ?? beside.cpuMs);
  return true;
};
const step = (part, pos, { count = GPU_TOKENS, stops = [] } = {}) => part.generateMany(7, pos, [1, 2, 3], 3, count, 0.7, 0.9, 1.1, [0.1, 0.2, 0.3, 0.4].slice(0, count), stops);

// ---- no WebGPU in the worker: everything on the CPU, and the page is told so
{
  const { part, beside } = made({ gpu: false });
  assert.deepEqual([part.note, part.status, part.whyNot, part.stepsWhyNot, part.ready, part.chosen], [undefined, undefined, "no WebGPU in a worker here", "no WebGPU in a worker here", undefined, null]);
  assert.deepEqual([part.gettingReady(), part.promptBlock(), part.tokenBlock(), part.cpuStepBegan(true), step(part, 10)], [false, BATCH, 0, false, undefined]);
  feed(part, 40);
  assert.deepEqual(beside.ran, [[BATCH, 0, false], [BATCH, BATCH, false], [40 - 2 * BATCH, 2 * BATCH, false]], "a prompt goes through the CPU, BATCH at a time, with no logits");
  assert.deepEqual([beside.made, part.promptTokens, part.sampled], [0, 0, 0]);
  assert.equal(await part.release(), true, "nothing to wait for");
  assert.deepEqual(Object.keys(part).sort(), ["chosen", "cpuStepBegan", "cpuStepEnded", "cpuWrote", "generateMany", "generationBegan", "gettingReady", "keysAndValues", "notHere",
    "note", "prompt", "promptBlock", "promptTokens", "put", "ready", "release", "sampled", "status", "stepsWhyNot", "tokenBlock", "whyNot"], "what the part hands out");
  for (const name of Object.keys(part)) assert.equal(Object.getOwnPropertyDescriptor(part, name).set, undefined, `${name} cannot be assigned from outside`);
  assert.deepEqual(said(), []);
}
console.log("ok: with no WebGPU the part keeps everything on the CPU and says why; nothing of its state can be assigned from outside");

// ---- a model the GPU does not take: its worker is never made, and the status line says why
{
  const { part, beside } = made({ whyNot: "linear-attention layers are not on the GPU yet" });
  assert.equal(await part.note, "prompts on the CPU (linear-attention layers are not on the GPU yet)");
  assert.deepEqual([beside.made, part.status, part.whyNot, part.gettingReady(), part.promptBlock()],
    [0, "prompts on the CPU (linear-attention layers are not on the GPU yet)", "linear-attention layers are not on the GPU yet", false, BATCH]);
  feed(part, 20);
  assert.deepEqual(sides(beside), [20, 0]);
}
console.log("ok: a model the GPU does not take never starts the GPU's worker");

// ---- the GPU's worker is started with what it needs, and until it is ready the prompts are the CPU's, untimed
{
  const { part, beside } = made();
  assert.deepEqual([beside.made, beside.posted.length, part.gettingReady(), part.status], [1, 1, true, "prompts on the CPU while the GPU gets ready"]);
  const [{ type, memory, plan }] = beside.posted;
  assert.deepEqual([type, memory], ["start", beside.memory]);
  assert.deepEqual(Object.keys(plan).sort(), ["batch", "cos", "dim", "direct", "eps", "force", "headSize", "heads", "hidden", "kvHeads", "kvStart", "layerNorm", "layers", "matrices",
    "parallel", "remembered", "rows", "seqLen", "sin", "staging", "tokens", "turned", "unturned", "vectors", "words"], "the plan the GPU's worker is started with");
  assert.deepEqual([plan.dim, plan.hidden, plan.layers, plan.heads, plan.kvHeads, plan.headSize, plan.turned, plan.seqLen, plan.kvStart, plan.eps, plan.batch, plan.direct, plan.rows, plan.staging],
    [8, 24, 2, 2, 1, 4, 4, 512, 256, 1e-5, GPU_BLOCK, false, ROWS, STAGING]);
  assert.deepEqual(plan.words, { done: GPU_DONE, failed: GPU_FAILED, beat: GPU_BEAT, wanted: GPU_WANTED });
  assert.deepEqual(Object.keys(plan.matrices), ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]);
  assert.deepEqual(plan.matrices.w2, { rows: 8, n: 24, six: false, ternary: false, layers: [[40000, 40500], [41000, 41500]] }, "a matrix: its values and scales a layer");
  assert.deepEqual(plan.vectors, { attention: { at: 50000, size: 8 }, ffn: { at: 50100, size: 8 } });
  assert.deepEqual(plan.tokens, { classifier: { rows: 100, n: 8, six: false, ternary: false, at: [44000, 44500] }, embedding: null, final: 50200, finalBias: 0, positions: 0,
    outliers: false, channels: [], ids: IDS, most: GPU_TOKENS });
  assert.deepEqual([plan.cos, plan.sin, plan.remembered, plan.force], [51000, 52000, { key: "kept" }, {}]);
  feed(part, 40);
  assert.deepEqual([sides(beside), part.promptBlock(), part.tokenBlock(), part.cpuStepBegan(true), part.ready], [[40, 0], BATCH, 0, false, undefined], "getting ready: the CPU's, and no step timed");
  beside.worker.say({ type: "progress" });
  assert.equal(part.gettingReady(), true, "a step of its getting ready is not its being ready");
  beside.ready();
  assert.equal(await part.note, "WebGPU where it is faster than the CPU");
  assert.deepEqual([part.gettingReady(), part.status, part.whyNot, part.stepsWhyNot], [false, "WebGPU where it is faster than the CPU", null, null]);
  assert.deepEqual(part.ready, { matrices: "tiles", attention: "flash", key: "the key", remembered: false, seconds: 1.5, tokens: "fused", tokenAttention: "vec", tablePieces: 1 });
  assert.equal(part.ready, part.chosen);
  assert.equal(said().length, 3, "the console says the adapter, the prompt's shaders and the token's");
  // the first prompt after it: the CPU is timed first, on the whole of it (a prompt goes to the GPU, then to the CPU, never back)
  feed(part, 200);
  assert.deepEqual(beside.ran.slice(0, 2), [[BATCH, 0, false], [BATCH, BATCH, false]], "the CPU first, to be timed");
  assert.deepEqual([sides(beside), part.promptTokens, part.promptBlock()], [[200, 0], 0, GPU_BLOCK], "the GPU holds nothing of a prompt the CPU began");
  // the next prompt: the verdict as it begins (the status line and the console), every block on the GPU
  feed(part, 200);
  const posted = requests(beside, "prompt");
  assert.deepEqual(posted.map(({ count, pos, tokens }) => [count, pos, tokens]), [[64, 0, undefined], [64, 64, undefined], [64, 128, undefined], [8, 192, undefined]], "the GPU's blocks, GPU_BLOCK at a time");
  assert.deepEqual(posted.map(({ serial }) => serial - posted[0].serial), [0, 1, 2, 3], "each request a serial of its own");
  assert.deepEqual([beside.embedded, beside.staged], [[[64, 0, ROWS, 32], [64, 64, ROWS, 32], [64, 128, ROWS, 32], [8, 192, ROWS, 32]], [[0, 64], [64, 64], [128, 64], [192, 8]]],
    "its rows are embedded for it, and its keys and values go into the cache");
  assert.deepEqual([sides(beside), part.promptTokens, part.status], [[0, 200], 200, `${PROMPTS_GPU}, answers on WebGPU where it is faster than the CPU`]);
  assert.match(said().join("\n"), /a block of 64 tokens: 8\.0 ms on the GPU, 64\.0 ms on the CPU \(1 thread\), the GPU from 5 tokens: prompts on WebGPU/);
  part.generationBegan();
  assert.equal(part.promptTokens, 0, "a generation's count begins again");
  // a cache too small for the block is grown first
  beside.capacity = 100;
  beside.grown.length = 0;
  feed(part, 128);
  assert.deepEqual(beside.grown, [127], "the cache grows to hold a block before the GPU writes it back");
  sides(beside);
  // T135: where the CPU wrote, the GPU's keys and values end: a block that begins after it is the CPU's
  part.generationBegan();
  part.prompt(tokensOf(64), 0);
  part.cpuWrote(10);
  part.prompt(tokensOf(64), 64);
  assert.deepEqual(sides(beside), [64, 64], "after the CPU wrote position 10, a block at 64 is the CPU's");
  part.prompt(tokensOf(64), 10);
  assert.deepEqual(sides(beside), [0, 64], "and one that begins where the GPU's end is its own");
  // T205: let go: the worker is told to stop and the engine waits for its "ended"
  assert.equal(await part.release(), true);
  assert.deepEqual([requests(beside, "stop").length, part.status, part.whyNot, part.ready, part.promptBlock(), Atomics.load(beside.ctl, GPU_WANTED)],
    [1, "prompts on the CPU (the model was let go)", "the model was let go", undefined, BATCH, 0]);
  assert.equal(await part.release(), true, "let go twice: nothing more");
  assert.equal(requests(beside, "stop").length, 1);
  feed(part, 20);
  assert.deepEqual(sides(beside), [20, 0], "a prompt after it is the CPU's");
  beside.ready();
  assert.equal(part.ready, undefined, "what a worker that was stopped says is not heard");
}
console.log("ok: the GPU's worker is started with the model's plan; the CPU is timed first, then the GPU takes the blocks it is faster for, up to where the CPU wrote");

// ---- which blocks: a GPU slower than the CPU takes none, a GPU with a wait takes the long ones; the side not chosen
// is timed again every GPU_RECHECK generations, not while the threads are searched; put() overrides
{
  const { part, beside } = made();
  beside.gpuMs = (count) => 2 * count;  // twice the CPU's time for any count
  beside.ready();
  said();
  feed(part, 128);
  said();
  feed(part, 128);
  assert.deepEqual([sides(beside), part.status, part.promptBlock()], [[256, 0], `${PROMPTS_CPU}, answers on WebGPU where it is faster than the CPU`, BATCH], "a slower GPU takes no block");
  assert.match(said().join("\n"), /the GPU from no count: prompts on the CPU \(faster here than WebGPU\)/);
  // GPU_RECHECK generations after the first verdict the GPU is given the beginning of a long prompt, to be timed again
  for (let g = 2; g < GPU_RECHECK; g++) feed(part, 128);
  assert.deepEqual([sides(beside), said()], [[128 * (GPU_RECHECK - 2), 0], []]);
  beside.threads.searching = true;
  feed(part, 128);
  assert.deepEqual([sides(beside), said()], [[128, 0], []], "not while the threads are searched");
  beside.threads.searching = false;
  part.generationBegan();
  assert.deepEqual(said(), ["gpu: the GPU on the beginning of the next long prompt, to time it again"]);
  assert.equal(part.promptBlock(), GPU_BLOCK, "the first block of that prompt is asked for whole");
  // a prompt too short for a whole block leaves the timing owed, and what follows in it is not asked for whole
  part.prompt(tokensOf(16), 0);
  assert.deepEqual([part.promptBlock(), sides(beside)], [BATCH, [16, 0]], "past a prompt's beginning its blocks are the CPU's size");
  part.generationBegan();
  assert.deepEqual([said(), part.promptBlock()], [[], GPU_BLOCK], "the next prompt's first block is");
  part.prompt(tokensOf(64), 0);
  assert.equal(part.promptBlock(), BATCH);
  part.prompt(tokensOf(16), 64);
  assert.deepEqual(sides(beside), [16, 64], "the GPU on the first whole block of the prompt, once");
  feed(part, 128);
  assert.deepEqual(sides(beside), [128, 0]);
  // T184: the benchmark puts every block on a side
  part.put("gpu");
  assert.equal(part.promptBlock(), GPU_BLOCK);
  feed(part, 100);
  assert.deepEqual(sides(beside), [0, 100], 'put("gpu"): every block the GPU holds what is before');
  part.put("cpu");
  assert.deepEqual([part.promptBlock(), part.tokenBlock()], [BATCH, 0]);
  feed(part, 100);
  assert.deepEqual(sides(beside), [100, 0], 'put("cpu")');
  part.put("somewhere");
  feed(part, 128);
  assert.deepEqual(sides(beside), [128, 0], "anything else: each where it is faster again");
  // another count of threads has times of its own: the CPU is timed first again
  beside.threads.count = 4;
  beside.cpuMs = 10;
  feed(part, 128);
  assert.deepEqual(sides(beside), [128, 0], "on 4 threads the CPU is timed anew");
  feed(part, 128);
  assert.deepEqual(sides(beside), [0, 128], "and here it is the slower");
  said();
}
{
  // a GPU with a wait of 20 ms a block and nothing a token: faster from 22 tokens on (20 < 0.95 × count)
  const { part, beside } = made();
  beside.gpuMs = () => 20;
  beside.ready();
  said();
  feed(part, 96);
  said();
  feed(part, 200);
  assert.equal(part.status, "prompts of 32 tokens and more on WebGPU, answers on WebGPU where it is faster than the CPU", "the fewest tokens, rounded up to the CPU's blocks");
  assert.deepEqual(sides(beside), [96 + 8, 192], "a short last block stays on the CPU");
  // GPU_RECHECK generations on, the CPU is timed again on the end of a long prompt: the last 2 × BATCH tokens of its short block
  for (let g = 2; g < GPU_RECHECK; g++) feed(part, 64);
  sides(beside);
  said();
  feed(part, 64 + 40);
  assert.deepEqual(said(), ["gpu: the CPU on the end of the next long prompt, to time it again"]);
  assert.deepEqual(beside.ran.slice(-2), [[BATCH, 72, false], [BATCH, 88, false]], "the last 32 tokens of the short block on the CPU, in whole blocks to time");
  assert.deepEqual(sides(beside), [40, 64], "(and the 8 before them, too few for this GPU)");
  feed(part, 64 + 40);
  assert.deepEqual(sides(beside), [0, 104], "once");
  // T184: where the GPU is the faster, put("cpu") keeps every block from it, and Python is handed the CPU's size
  part.put("cpu");
  assert.equal(part.promptBlock(), BATCH);
  feed(part, 100);
  assert.deepEqual(sides(beside), [100, 0]);
  part.put(null);
  assert.equal(part.promptBlock(), GPU_BLOCK);
}
console.log("ok: a block goes where it is faster; the other side is timed again every GPU_RECHECK generations, not during a search; put() puts every block on a side");

// ---- the steps of a generation
{
  const { part, beside } = made();
  beside.ready();
  said();
  assert.equal(part.tokenBlock(), 0, "no step on the GPU before the CPU's is timed");
  assert.equal(part.cpuStepBegan(false), false, "a token without logits is no step");
  beside.cpuMs = 5;
  assert.deepEqual([cpuStep(part, beside), part.tokenBlock(), cpuStep(part, beside), part.tokenBlock()], [true, 0, true, GPU_TOKENS], "two steps of the CPU timed, and the GPU's are faster");
  part.generationBegan();
  assert.deepEqual([said(), part.status], [["gpu: a step of a generation: 1.00 ms on the GPU, 5.00 ms on the CPU (1 thread): answers on WebGPU"], `${PROMPTS_UNTIMED}, answers on WebGPU`]);
  part.prompt(tokensOf(BATCH), 0);
  beside.posted.length = 0;
  beside.ran.length = 0;
  beside.capacity = 16;
  assert.deepEqual(step(part, 16), [16, 17, 18, 19], "the ids the GPU sampled");
  const [asked] = beside.posted;
  assert.deepEqual([asked.type, asked.count, asked.pos, asked.from, asked.token, asked.history, asked.length, asked.randoms], ["tokens", 4, 16, 0, 7, [1, 2, 3], 3, [0.1, 0.2, 0.3, 0.4]]);
  assert.deepEqual([asked.cache, asked.settings], [{ keys: 60000, values: 61000, capacity: 32, row: 8, half: true }, { temperature: 0.7, topp: 0.9, penalty: 1.1, stops: [] }],
    "the cache as it is now: after it grew for the steps");
  assert.deepEqual([beside.grown, beside.staged, part.sampled], [[19], [[16, 4]], 4]);
  assert.deepEqual(step(part, 20, { count: 2 }), [20, 21]);
  assert.equal(beside.posted[1].from, 20, "the GPU holds the positions it sampled");
  part.cpuWrote(18);
  step(part, 22);
  assert.equal(beside.posted[2].from, 18, "and after the CPU wrote one, the cache goes up from there");
  assert.deepEqual([step(part, 30, { count: 0 }), step(part, 30, { count: GPU_TOKENS + 1 })], [undefined, undefined], "no steps, or more than a submission holds: the CPU's");
  // a stop token sampled: the ids up to it
  beside.answer.sampled = 2;
  assert.deepEqual([step(part, 30, { stops: [31] }), part.sampled], [[30, 31], 12]);
  delete beside.answer.sampled;
  // not while the threads are searched (that times the CPU's tokens)
  beside.threads.searching = true;
  assert.deepEqual([part.tokenBlock(), step(part, 40)], [0, undefined]);
  beside.threads.searching = false;
  // GPU_RECHECK generations after the first verdict the CPU takes the first TOKEN_RECHECK steps, to be timed again
  part.generationBegan();
  assert.equal(part.sampled, 0, "a generation's count begins again");
  for (let g = 3; g < GPU_RECHECK; g++) part.generationBegan();
  assert.deepEqual([said(), part.tokenBlock()], [[], GPU_TOKENS]);
  part.generationBegan();
  assert.deepEqual([said(), part.tokenBlock()], [[`gpu: the CPU on the first ${TOKEN_RECHECK} steps of the next generation, to time it again`], 0]);
  for (let i = 1; i < TOKEN_RECHECK; i++) assert.deepEqual([cpuStep(part, beside), part.tokenBlock()], [true, 0]);
  assert.deepEqual([cpuStep(part, beside), part.tokenBlock()], [true, GPU_TOKENS], "after them the GPU's again");
  // a CPU that got faster than the GPU: the verdict of the next generation
  for (let i = 0; i < 5; i++) cpuStep(part, beside, 0.5);
  part.generationBegan();
  assert.deepEqual([said(), part.status, part.tokenBlock()], [["gpu: a step of a generation: 1.00 ms on the GPU, 0.50 ms on the CPU (1 thread): answers on the CPU"], `${PROMPTS_UNTIMED}, answers on the CPU (faster here)`, 0]);
  part.put("gpu");
  assert.equal(part.tokenBlock(), GPU_TOKENS, 'put("gpu"): the steps as well');
  part.put(null);
  // more stop tokens than the GPU's sampling holds: the steps are the CPU's from then on, said in the console
  for (let i = 0; i < 5; i++) cpuStep(part, beside, 5);
  assert.equal(part.tokenBlock(), GPU_TOKENS);
  assert.equal(step(part, 50, { stops: Array.from({ length: STOPS_MOST + 1 }, (_, i) => i) }), undefined);
  assert.deepEqual([said(), part.tokenBlock(), part.stepsWhyNot, part.cpuStepBegan(true)], [[`gpu: answers on the CPU (more than ${STOPS_MOST} stop tokens)`], 0, `more than ${STOPS_MOST} stop tokens`, false]);
  assert.equal(part.promptBlock() > 0 && part.ready !== undefined, true, "the prompts stay where they were");
}
{
  // a model whose steps are not asked of the GPU, and a GPU that did not make them
  const none = made({ tokensWhyNot: "a classifier of float weights is not on the GPU's tokens", steps: false });
  assert.equal(none.beside.posted[0].plan.tokens, null);
  none.beside.ready();
  assert.deepEqual([await none.part.note, none.part.stepsWhyNot, none.part.tokenBlock(), none.part.cpuStepBegan(true)],
    [`${PROMPTS_UNTIMED}, answers on the CPU`, "a classifier of float weights is not on the GPU's tokens", 0, false]);
  const refused = made({ steps: false });
  refused.beside.ready({ tokensWhy: "no form of a token was right" });
  assert.deepEqual([await refused.part.note, refused.part.stepsWhyNot], [`${PROMPTS_UNTIMED}, answers on the CPU`, "no form of a token was right"]);
  assert.deepEqual(said().filter((line) => /answers/.test(line)), ["gpu: answers on the CPU (a classifier of float weights is not on the GPU's tokens)", "gpu: answers on the CPU (no form of a token was right)"]);
}
console.log("ok: the steps go to the GPU where they are faster, the CPU's are timed again every GPU_RECHECK generations, and a model or a GPU without them says why");

// ---- every way a request is refused: the GPU stops, the status line says why, the CPU takes the block or the step
{
  const refusals = [
    ["a block the GPU fails on", (b) => { b.answer.failed = true; }, "prompt", /^prompts on the CPU \(the GPU failed on a block of the prompt\)$/],
    ["a block the GPU never answers", (b) => { b.answer.silent = true; }, "prompt", /^prompts on the CPU \(the GPU's worker stopped answering for 0\.06 s\)$/],
    ["a block whose keys or values are not finite", (b) => { b.finite = false; }, "prompt", /^prompts on the CPU \(the GPU computed keys or values that are not finite numbers \(NaN or infinity\) in a block of the prompt at position 0\)$/],
    ["a step the GPU fails on", (b) => { b.answer.failed = true; }, "tokens", /^prompts on the CPU \(the GPU failed on a token\)$/],
    ["a step the GPU never answers", (b) => { b.answer.silent = true; }, "tokens", /^prompts on the CPU \(the GPU's worker stopped answering for 0\.06 s\)$/],
    ["a step whose logits are not finite", (b) => { b.answer.notFinite = true; b.answer.sampled = 1; }, "tokens", /^prompts on the CPU \(the GPU computed logits that are not finite numbers \(NaN or infinity\) at position 65\)$/],
    ["no step sampled", (b) => { b.answer.sampled = 0; }, "tokens", /^prompts on the CPU \(the GPU sampled 0 of 4 tokens\)$/],
    ["more steps sampled than asked", (b) => { b.answer.sampled = 5; }, "tokens", /^prompts on the CPU \(the GPU sampled 5 of 4 tokens\)$/],
    ["an id below the vocabulary", (b) => { b.answer.ids = [3, -1]; }, "tokens", /^prompts on the CPU \(the GPU sampled -1, outside the vocabulary of 100\)$/],
    ["an id past the vocabulary", (b) => { b.answer.ids = [3, 4, 5, 100]; }, "tokens", /^prompts on the CPU \(the GPU sampled 100, outside the vocabulary of 100\)$/],
    ["a step whose keys or values are not finite", (b) => { b.finite = false; }, "tokens", /^prompts on the CPU \(the GPU computed keys or values that are not finite numbers \(NaN or infinity\) in a step at position 64\)$/],
  ];
  for (const [title, breakIt, type, status] of refusals) {
    const { part, beside } = made();
    beside.ready();
    part.put("gpu");
    part.generationBegan();
    if (type === "tokens") part.prompt(tokensOf(64), 0);
    said();
    beside.posted.length = 0;
    beside.ran.length = 0;
    beside.staged.length = 0;
    breakIt(beside);
    if (type === "prompt") {
      part.prompt(tokensOf(64), 0);
      assert.deepEqual(beside.ran, [[16, 0, false], [16, 16, false], [16, 32, false], [16, 48, false]], `${title}: the CPU takes the block`);
      assert.equal(part.promptTokens, 0, `${title}: not counted as the GPU's`);
    } else {
      assert.equal(step(part, 64), undefined, `${title}: the CPU takes the step`);
      assert.equal(part.sampled, 0, `${title}: none of its ids taken`);
    }
    assert.deepEqual(beside.staged, [], `${title}: nothing of it goes into the cache`);
    assert.match(part.status, status, title);
    assert.deepEqual([part.ready, part.promptBlock(), part.tokenBlock(), part.cpuStepBegan(true), requests(beside, "stop").length, Atomics.load(beside.ctl, GPU_WANTED)], [undefined, BATCH, 0, false, 1, 0], `${title}: the GPU is stopped`);
    assert.equal(said().filter((line) => /the prompts and the tokens go on on the CPU$/.test(line)).length, 1, `${title}: said once in the console`);
    sides(beside);
    beside.answer = { ended: true };
    beside.finite = true;
    feed(part, 64);
    assert.deepEqual([sides(beside), step(part, 64)], [[64, 0], undefined], `${title}: and stays stopped`);
    await part.release();
  }
}
{
  // the worker's own words: it cannot take the model, it did not start, it failed (said, and nothing more)
  const unusable = made();
  unusable.beside.worker.say({ type: "unusable", reason: "a fallback adapter" });
  assert.deepEqual([await unusable.part.note, unusable.part.whyNot, unusable.part.gettingReady()], ["prompts on the CPU (a fallback adapter)", "a fallback adapter", false]);
  const unstarted = made();
  unstarted.beside.worker.handlers.error({ message: "no such file" });
  assert.equal(await unstarted.part.note, "prompts on the CPU (the GPU's worker did not start (no such file))");
  assert.equal(await unstarted.part.release(), true, "a worker that did not start holds nothing to wait for");
  const failed = made();
  failed.beside.worker.say({ type: "failed", reason: "a shader was refused" });
  assert.deepEqual([said(), failed.part.gettingReady()], [["gpu: a shader was refused"], true]);
}
console.log("ok: a request with no answer, a failure, an id outside the vocabulary or numbers that are not finite stops the GPU with its reason, and the CPU goes on");

// ---- T156, T210: a model on the GPU alone
{
  const lost = [];
  const { part, beside } = made({ direct: { onLost: (why) => lost.push(why) } });
  assert.deepEqual([beside.posted[0].plan.direct, beside.posted[0].plan.rows], [true, 0]);
  beside.ready();
  said();
  assert.equal(await part.note, "prompts and answers on WebGPU");
  assert.deepEqual([part.promptBlock(), part.tokenBlock(), part.cpuStepBegan(true)], [GPU_BLOCK, GPU_TOKENS, true]);
  feed(part, 100);
  assert.deepEqual(requests(beside, "prompt").map(({ count, pos, tokens }) => [count, pos, tokens.length]), [[64, 0, 64], [36, 64, 36]], "the GPU is handed the tokens: it embeds them itself");
  assert.deepEqual([beside.embedded, beside.staged, beside.grown, beside.ran], [[], [], [], []], "no rows, no cache and nothing run on the CPU");
  assert.deepEqual(step(part, 100), [0, 1, 2, 3]);
  assert.deepEqual([beside.posted.at(-1).cache, beside.posted.at(-1).from, beside.staged], [null, 100, []]);
  // T210: its keys and values are read back from the GPU, GPU_BLOCK positions a request
  const { keys, values } = part.keysAndValues(3, 70);
  assert.deepEqual(requests(beside, "keys").map(({ count, pos }) => [count, pos]), [[64, 3], [6, 67]]);
  assert.deepEqual([keys.length, values.length, new Set(keys), new Set(values)], [2 * 70 * 4, 2 * 70 * 4, new Set([2]), new Set([4])], "[layers][count][kvDim] in float32, the keys and the values each from their side");
  // T219: an id outside the vocabulary cannot be taken again on the CPU: it stops with words, and the model is not loaded again
  beside.answer.ids = [-1];
  assert.throws(() => step(part, 104), { message: OUTSIDE_VOCABULARY });
  delete beside.answer.ids;
  assert.deepEqual([lost, step(part, 104)], [[], [4, 5, 6, 7]], "the GPU still takes the steps");
  // something that would run on the CPU: lost, said once, and the error the engine throws
  const first = part.notHere(), second = part.notHere();
  assert.deepEqual([first.message, first.directLost, second.message], ["The GPU stopped (the GPU did not take a block or a step), and this model's weights were on it alone", true, first.message]);
  assert.deepEqual([lost, part.whyNot, requests(beside, "stop").length], [["the GPU did not take a block or a step"], "the GPU did not take a block or a step", 1]);
  assert.throws(() => part.prompt(tokensOf(16), 0), /this model's weights were on it alone/, "a block after it has nowhere to run");
  assert.throws(() => part.keysAndValues(0, 4), /The GPU stopped \(the GPU did not take a block or a step\), and this model's keys and values were on it alone/);
  assert.equal(lost.length, 1, "lost is said once");
  await part.release();
  assert.equal(lost.length, 1, "and not when the model is let go");
  said();
}
{
  // a GPU that does not take the steps cannot hold a model alone; nor one the CPU beats as /benchmark/ measured it
  const lost = [];
  const none = made({ direct: { onLost: (why) => lost.push(why) }, steps: false });
  none.beside.ready({ tokensWhy: "no form of a token was right" });
  assert.deepEqual([await none.part.note, lost], ["prompts on the CPU (no form of a token was right)", ["no form of a token was right"]]);
  const direct = { onLost: (why) => lost.push(why), size: 1e9, layerWeights: 1e9, cpu: { GBps: 1e6, promptGMACs: 1e6 }, usage: { prompt: 100, written: 100 } };
  const beaten = made({ direct });
  beaten.beside.ready();
  assert.match(await beaten.part.note, /^prompts on the CPU \(the CPU as \/benchmark\/ measured it \(1000000\.0 GB\/s\): /);
  assert.deepEqual([direct.verdict, lost.length], [{ key: "the key", cpu: direct.cpu }, 2], "the page is handed the verdict to keep");
  const letGo = made({ direct: { onLost: (why) => lost.push(why) } });
  await letGo.part.release();
  assert.equal(lost.length, 2, "a model let go before its GPU was ready is not lost");
  said();
}
console.log("ok: a model on the GPU alone runs nothing on the CPU, reads its keys and values back from the GPU, and what the GPU does not take is lost, said once");

// ---- T205: a worker that does not say it ended is not waited for for ever (the wait itself, GPU_END_MS, is
// gpu-default-check's: five seconds); and one that says so late is waited for
{
  const { part, beside } = made();
  beside.ready();
  said();
  beside.answer.ended = false;
  let settled = null;
  const released = part.release().then((ended) => { settled = ended; });
  await tick();
  assert.equal(settled, null, "the release waits for the worker's end");
  beside.worker.say({ type: "ended" });
  await released;
  assert.deepEqual([settled, beside.terminated], [true, 0]);
}
console.log("ok: letting go waits for the GPU's worker to say it ended");

// ---- the control area (control.js), which the part waits on: cleared for a new engine, none on a memory that is not
// shared, and a wait that sees the GPU's worker count while it works does not give it up (T135: the beat)
{
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
  new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4).fill(7);
  const { ctl, waitUntil } = controlArea({ memory, sharedMemory: true, stalledMs: 60 });
  assert.deepEqual([ctl.length, new Set(ctl)], [CONTROL_BYTES / 4, new Set([0])], "what the engine before left in the control area is gone");
  assert.equal(controlArea({ memory: new WebAssembly.Memory({ initial: 1 }), sharedMemory: false, stalledMs: 60 }).ctl, null);
  const { Worker } = await import("node:worker_threads");
  // another thread counts the beat every 10 ms for 240 ms, four times the wait's patience, and then answers
  const beats = new Worker(`const { workerData: [buffer, beat, done] } = require("node:worker_threads");
    const ctl = new Int32Array(buffer, 0, ${CONTROL_BYTES / 4});
    let n = 0;
    const timer = setInterval(() => {
      Atomics.add(ctl, beat, 1);
      if (++n < 24) return;
      clearInterval(timer);
      Atomics.store(ctl, done, 5);
      Atomics.notify(ctl, done);
    }, 10);`, { eval: true, workerData: [memory.buffer, GPU_BEAT, GPU_DONE] });
  await new Promise((resolve) => beats.once("online", resolve));
  const began = performance.now();
  assert.equal(waitUntil(GPU_DONE, (seen) => seen === 5, GPU_BEAT), true, "a worker that counts while it works is waited for past the time a silent one is given");
  assert.ok(performance.now() - began > 120, "(and it was waited for)");
  assert.equal(waitUntil(GPU_DONE, (seen) => seen === 6, GPU_BEAT), false, "one that neither answers nor counts is given up");
  await beats.terminate();
}
console.log("ok: the control area is cleared for a new engine, and a wait is kept up by the beat of a worker at work");

// ---- gpuFit(): why the GPU would not take a model, and the vectors it reads
{
  const fitOf = (shape = {}, weights = {}, device = {}, direct) =>
    gpuFit({ shape: { ...SHAPE, ...shape }, weights: { ...WEIGHTS, ...weights }, device: { gpuRoom: undefined, memoryUnsaid: false, sharedMemory: true, ...device }, direct });
  assert.deepEqual([fitOf().unfit(), fitOf().tokensUnfit()], [null, null]);
  assert.deepEqual(Object.keys(fitOf()).sort(), ["matrices", "tokensUnfit", "unfit", "vectors"]);
  const reasons = [
    [{ linear: { every: 4 } }, {}, {}, /^linear-attention layers are not on the GPU yet$/],
    [{ convolution: { taps: 3 } }, {}, {}, /^convolution layers are not on the GPU yet$/],
    [{ rotated: 1024 }, {}, {}, /^a rotated basis is not on the GPU yet$/],
    [{}, {}, { sharedMemory: false }, /^the page is not cross-origin isolated$/],
    [{ headSize: 6 }, {}, {}, /^heads of a size that is no multiple of 4 are not on the GPU$/],
    [{}, { wq: { rows: 8, n: 8, int8: false, layer: () => [0] } }, {}, /^float32 weights are not on the GPU yet$/],
    [{}, { w1: matrixOf(24, 8, 0, { group: 128 }) }, {}, /^float32 weights are not on the GPU yet$/],
    [{}, {}, { gpuRoom: 1000 }, /^the layers on the GPU as well \(0 MB\) would not leave this device enough memory$/],
  ];
  for (const [shape, weights, device, reason] of reasons) assert.match(fitOf(shape, weights, device).unfit(), reason);
  assert.equal(fitOf({}, {}, { gpuRoom: 1000 }, { size: 1 }).unfit(), null, "a model on the GPU alone is not held twice");
  assert.equal(fitOf({}, { w3: null }).unfit(), null, "an FFN with no gate (GPT-2) has no w3");
  assert.deepEqual(Object.keys(fitOf({}, { w3: null }).matrices()), ["wq", "wk", "wv", "wo", "w1", "w2"]);
  assert.equal(fitOf({}, { w1: matrixOf(24, 128, 0, { ternary: true, group: 128 }), w2: matrixOf(8, 128, 0, { ternary: true, group: 128 }) }).unfit(), null, "ternary weights in groups of 128");
  const steps = [
    [{}, { wcls: { rows: 100, n: 8, int8: false, layer: () => [0] } }, {}, /^a classifier of float weights is not on the GPU's tokens$/],
    [{}, { embedding: { kind: "f16", group: 32 } }, {}, /^a classifier of float weights is not on the GPU's tokens$/],
    [{}, {}, { memoryUnsaid: true }, /^this browser does not say how much memory the device has$/],
    [{}, {}, { gpuRoom: 20000 }, /^the classifier on the GPU as well \(0 MB with the layers\) would not leave this device enough memory$/],
  ];
  for (const [shape, weights, device, reason] of steps) assert.match(fitOf(shape, weights, device).tokensUnfit(), reason);
  // the bytes it counts: the layers' weights and vectors and its own keys and values, then the tables of the steps
  const weightsOf = (m) => SHAPE.layers * m.rows * m.n * 1.125, { wq, wk, wv, wo, w1, w2, w3 } = WEIGHTS;
  const layers = [wq, wk, wv, wo, w1, w2, w3].reduce((sum, m) => sum + weightsOf(m), 0) + 2 * SHAPE.layers * 8 * 4 + 2 * SHAPE.layers * SHAPE.seqLen * SHAPE.kvDim * 2;
  assert.deepEqual([fitOf({}, {}, { gpuRoom: layers }).unfit(), fitOf({}, {}, { gpuRoom: layers - 1 }).unfit() === null], [null, false], "the layers on the GPU, to the byte");
  const table = 100 * 8 * 1.125, tokens = layers + table + SHAPE.seqLen * SHAPE.headSize * 4 + 3 * 100 * 4;
  assert.deepEqual([fitOf({}, {}, { gpuRoom: tokens }).tokensUnfit(), fitOf({}, {}, { gpuRoom: tokens - 1 }).tokensUnfit() === null], [null, false], "and the steps' tables with them");
  assert.deepEqual([fitOf({ sharedClassifier: false }, {}, { gpuRoom: tokens + table }).tokensUnfit(), fitOf({ sharedClassifier: false }, {}, { gpuRoom: tokens + table - 1 }).tokensUnfit() === null], [null, false],
    "an embedding that is another table counts twice");
  // the vectors: the norms' weights always, the rest where the model has them
  assert.deepEqual(Object.keys(fitOf({ layerNorm: true }, { attB: 1, ffnB: 2, bq: 3, bk: 4, bv: 5, bo: 6, b1: 7, b2: 8, qNorm: 9, kNorm: 10 }).vectors()),
    ["attention", "ffn", "attentionBias", "ffnBias", "bq", "bk", "bv", "qNorm", "kNorm", "bo", "b1", "b2"]);
  assert.deepEqual(fitOf({}, { bq: 3, bk: 4, bv: 5, qNorm: 9, kNorm: 10 }).vectors(), { attention: { at: 50000, size: 8 }, ffn: { at: 50100, size: 8 },
    bq: { at: 3, size: 8 }, bk: { at: 4, size: 4 }, bv: { at: 5, size: 4 }, qNorm: { at: 9, size: 4 }, kNorm: { at: 10, size: 4 } });
}
console.log("ok: gpuFit() says why the GPU would not take a model or its steps, counts what it would hold to the byte, and names the vectors it reads");
process.exit(0);
