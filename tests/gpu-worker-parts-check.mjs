// tests/gpu-worker-parts-check.mjs (T378.4): the parts of the model's GPU worker (src/runtime/gpu/) that hold its state,
// each by itself on a stand-in for WebGPU that writes down what it is asked and keeps every buffer's bytes. It computes
// nothing: no shader runs here, and nothing here says that a shader is right (CI's gpu-check does, on a device).
// What is held is who holds what, and what each owner does when it is told that something happened:
//   - device.js: what openDevice() puts in the model's record and no more; a fallback adapter refused before anything
//     is made; handed() (the memory and the plan of start()); held() and end() (the model's own buffers, then the ones
//     held last, then the device, then "ended"); stopAsked() and a device lost, read by the other modules as they are;
//   - weights.js: the record it makes (m.weights), every matrix's and vector's bytes on the GPU as the shared memory
//     holds them; a matrix past a buffer in pieces, and why a token's layer is not joined then; a model on the GPU
//     alone: its routes, and routed() writing a checkpoint that comes in stretches of any size as whole words (the
//     bytes of a word a stretch cuts wait for the rest: unwritten() says so);
//   - block.js: the record it makes (m.block) from the weights' record and a stand-in for the forms'; the cache made,
//     grown (the old one copied and let go, the new one's buffers held for end()), roomFor(); a block's writes,
//     dispatches and its keys and values into the staging, or nowhere where the request was given up; keysIn().
// Not here, because their work is checked against JavaScript by a device that computes: forms.js, tokens.js,
// tokenattention.js, tokenforms.js (m.forms and m.tokens are made after checks pass), start.js and requests.js.
// tests/gpu-modules-check.mjs holds their names. Node only, under a second:
//
//   node tests/gpu-worker-parts-check.mjs
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";

// ---- the stand-in: a device that writes down what it is asked (calls) and keeps the bytes of every buffer
const posted = [];
let closed = 0;
globalThis.self = globalThis;
globalThis.postMessage = (message) => posted.push(message);
globalThis.close = () => { closed += 1; };
function standIn({ fallback = false, language = [] } = {}) {
  const calls = [];
  let losing;
  const limits = { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535,
    maxComputeWorkgroupStorageSize: 16384, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, minStorageBufferOffsetAlignment: 256 };
  const bytesOf = (data) => (data instanceof ArrayBuffer || data instanceof SharedArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  const device = {
    features: new Set(), limits, lost: new Promise((resolve) => { losing = resolve; }),
    createBuffer({ size, usage }) {
      const made = { size, usage, data: new Uint8Array(size), destroyed: false, destroy() { made.destroyed = true; calls.push(["destroy", made]); },
        async mapAsync() {}, getMappedRange: (from = 0, length = size - from) => made.data.slice(from, from + length).buffer, unmap() {} };
      calls.push(["buffer", made]);
      return made;
    },
    queue: {
      // (as WebGPU has it: the offset and the size of a typed array's data count its elements)
      writeBuffer(target, offset, data, from = 0, size) {
        const each = ArrayBuffer.isView(data) ? data.BYTES_PER_ELEMENT : 1, bytes = bytesOf(data);
        const length = size === undefined ? bytes.length - from * each : size * each;
        assert.ok(offset % 4 === 0 && length % 4 === 0, `a write of ${length} bytes at ${offset}: WebGPU takes whole words`);
        assert.ok(offset + length <= target.size, "a write past its buffer");
        target.data.set(bytes.subarray(from * each, from * each + length), offset);
        calls.push(["write", target, offset, length]);
      },
      submit() { calls.push(["submit"]); },
      onSubmittedWorkDone: async () => {},
    },
    createShaderModule: ({ code }) => ({ code }),
    async createComputePipelineAsync({ compute }) {
      const pipeline = { code: compute.module.code, getBindGroupLayout: () => ({}) };
      calls.push(["pipeline", pipeline]);
      return pipeline;
    },
    createBindGroup: ({ entries }) => ({ entries }),
    createCommandEncoder() {
      return {
        beginComputePass() {
          const pass = { setPipeline(pipeline) { pass.pipeline = pipeline; }, setBindGroup(index, group) { pass.group = group; },
            dispatchWorkgroups(x, y, z) { calls.push(["dispatch", pass.pipeline, pass.group, x, y, z]); }, end() {} };
          return pass;
        },
        // (done as it is encoded: nothing here runs between the encoding and the submission)
        copyBufferToBuffer(source, from, target, to, length) {
          target.data.set(source.data.subarray(from, from + length), to);
          calls.push(["copy", source, from, target, to, length]);
        },
        finish: () => ({}),
      };
    },
    pushErrorScope() {}, popErrorScope: async () => null, destroy() { calls.push(["device gone"]); },
  };
  const adapter = { info: { vendor: "stand-in", architecture: "", device: "", description: "writes down what it is asked", isFallbackAdapter: fallback,
    subgroupMinSize: 4, subgroupMaxSize: 16 }, limits, features: new Set(), requestDevice: async () => device };
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { userAgent: "gpu-worker-parts-check",
    gpu: { requestAdapter: async () => adapter, wgslLanguageFeatures: new Set(language) } } });
  return { calls, device, adapter, lose: (info) => losing(info) };
}

// ---- a made-up model in a memory of its own: no two of its matrices the same size where one could stand for another
const DIM = 32, HIDDEN = 64, HEADS = 2, KV_HEADS = 1, HEAD = 16, LAYERS = 2, BATCH = 8, CONTEXT = 64, GROUP = 32;
const Q_DIM = HEADS * HEAD, KV_DIM = KV_HEADS * HEAD, KV_ROW = KV_DIM * 2;
const SIZES = { wq: [Q_DIM, DIM], wk: [KV_DIM, DIM], wv: [KV_DIM, DIM], wo: [DIM, Q_DIM], w1: [HIDDEN, DIM], w3: [HIDDEN, DIM], w2: [DIM, HIDDEN] };
/** A plan as forward.js hands one, with the addresses of what it names in bytes (a memory, or a checkpoint): the
 * matrices' values and scales a layer, the vectors, the block's rows and staging, RoPE's tables. */
function planned(more = {}, sizes = SIZES) {
  let at = 4096;
  const take = (bytes) => { const here = at; at += Math.ceil(bytes / 256) * 256; return here; };
  const matrices = Object.fromEntries(Object.entries(sizes).map(([name, [rows, n]]) =>
    [name, { rows, n, layers: Array.from({ length: LAYERS }, () => [take(rows * n), take((rows * n / GROUP) * 4)]) }]));
  const vectors = { attention: { at: take(LAYERS * DIM * 4), size: DIM }, ffn: { at: take(LAYERS * DIM * 4), size: DIM } };
  const plan = { layers: LAYERS, dim: DIM, hidden: HIDDEN, heads: HEADS, kvHeads: KV_HEADS, headSize: HEAD, batch: BATCH, seqLen: CONTEXT, kvStart: 8, turned: HEAD,
    eps: 1e-5, layerNorm: false, parallel: false, matrices, vectors, tokens: null, force: {},
    rows: take(BATCH * DIM * 4), staging: take(2 * LAYERS * BATCH * KV_ROW), cos: take(CONTEXT * (HEAD / 2) * 4), sin: take(CONTEXT * (HEAD / 2) * 4), ...more };
  return { plan, bytes: at };
}
const random = (bytes) => Uint8Array.from({ length: bytes }, (_, i) => (i * 131 + 7 * (i >> 8) + 89) & 255);
const same = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);

const device_ = await import(runtimeUrl("gpu/device.js"));
const { openDevice, handed, held, end, STORAGE, COPY_DST, COPY_SRC } = device_;
const { upload, uploadLayers, tablesOn, routed, unwritten } = await import(runtimeUrl("gpu/weights.js"));
const { prepare, bindLayers, grow, roomFor, keysIn, block } = await import(runtimeUrl("gpu/block.js"));
const { RMSNORM, HEAD_NORM, ADD, ROPE, SWIGLU, QUANTIZE } = await import(runtimeUrl("shaders/steps.js"));
const { FLASH_Q_TILE } = await import(runtimeUrl("shaders/attention.js"));

// ---- device.js: the record it makes, and what it is told
{
  const refused = standIn({ fallback: true }), said = [];
  assert.equal(await openDevice({ force: {}, matrices: {} }, (why) => { said.push(why); }), undefined);
  assert.deepEqual(said, ["a fallback adapter: the CPU in the GPU's place"]);
  assert.equal(refused.calls.length, 0, "nothing is made on an adapter that is refused");
  assert.equal(device_.model, null, "and there is no record of a model");
}
const gpu = standIn();
const { plan, bytes: memoryBytes } = planned();
const memory = { buffer: new SharedArrayBuffer(memoryBytes) };
new Uint8Array(memory.buffer).set(random(memoryBytes));
const m = await openDevice(plan);
assert.equal(m, device_.model, "openDevice() returns the record the other modules read as model");
assert.deepEqual(Object.keys(m).sort(), ["adapter", "device", "fallback", "held", "info", "key", "limit", "memory", "owned", "plan", "remembered", "ternary"],
  "what device.js puts in the model's record: the device and what it said of itself, the plan and the memory, the buffers owned and held; the records are their owners'");
assert.equal(m.device, gpu.device);
assert.equal(m.memory, null, "no memory until start() hands one");
assert.equal(m.ternary, false);
assert.equal(m.remembered, null, "nothing remembered where the page kept nothing");
assert.equal(m.limit, 1 << 27, "the largest buffer the device binds");
// (start()'s plan is another object than the one a model on the GPU alone was opened with)
const opened = m.plan;
handed(memory, { ...plan });
assert.equal(m.memory, memory, "handed(): the memory start() was given");
assert.notEqual(m.plan, opened, "and its plan");
assert.deepEqual(m.plan, plan);
handed(memory, plan);
{
  // (what the page kept counts for the same adapter and shaders alone)
  const again = standIn();
  const kept = await openDevice({ ...plan, remembered: { key: m.key, matrices: "a form" } });
  assert.deepEqual(kept.remembered, { key: m.key, matrices: "a form" });
  const other = await openDevice({ ...plan, remembered: { key: "another device's", matrices: "a form" } });
  assert.equal(other.remembered, null, "what the page kept of another device is not remembered");
  assert.equal(again.calls.length, 0);
  assert.equal(device_.model, other, "the record is the one opened last");
}
console.log("gpu-worker-parts-check: device.js makes the model's record of 12 fields, refuses a fallback adapter before anything is made, and takes the memory and the plan as handed()");

// ---- weights.js: m.weights, from the shared memory
const up = await upload(m);
{
  const W = m.weights;
  assert.deepEqual(Object.keys(W).sort(), ["angleTable", "finalBias", "finalNorm", "joined", "matrices", "partial", "positions", "routes", "tables", "together", "unjoined", "vectors", "widen"]);
  let expected = 0;
  for (const [name, [rows, n]] of Object.entries(SIZES)) {
    assert.deepEqual(W.matrices[name].pieces.map((piece) => [piece.first, piece.rows]), [[0, rows]], `${name} is one piece where it fits a buffer`);
    for (let l = 0; l < LAYERS; l++) {
      const [values, scales] = W.matrices[name].pieces[0].layers[l], [valuesAt, scalesAt] = plan.matrices[name].layers[l];
      assert.ok(same(values.data.subarray(0, rows * n), new Uint8Array(memory.buffer, valuesAt, rows * n)), `${name} of layer ${l}: its values as the memory holds them`);
      assert.ok(same(scales.data.subarray(0, (rows * n / GROUP) * 4), new Uint8Array(memory.buffer, scalesAt, (rows * n / GROUP) * 4)), `${name} of layer ${l}: its scales`);
      assert.equal(values.usage, STORAGE | COPY_DST, "no COPY_SRC where nothing is read back");
      expected += rows * n + (rows * n / GROUP) * 4;
    }
  }
  for (const [name, { at, size }] of Object.entries(plan.vectors)) {
    assert.ok(same(W.vectors[name].data.subarray(0, LAYERS * size * 4), new Uint8Array(memory.buffer, at, LAYERS * size * 4)), `the vector ${name}`);
    expected += LAYERS * size * 4;
  }
  assert.equal(up, expected, "upload() says the bytes it put up");
  assert.equal(W.routes, null, "no routes where the weights come from the shared memory");
  assert.equal(W.together, null, "no layer joined where no token goes on the GPU");
  assert.equal(W.unjoined, null);
  assert.equal(W.tables, null);
  assert.ok(m.owned.length >= 2 * LAYERS * 7 + 2 && m.owned.every((buffer) => !buffer.destroyed), "the buffers are the model's own, for end()");
}
// a matrix past a buffer: in pieces of whole rows, each from its rows of the memory
{
  standIn();
  const { plan: wide, bytes } = planned({ force: { pieceBytes: 2048 } }, { ...SIZES, w1: [160, DIM] });
  const wideMemory = { buffer: new SharedArrayBuffer(bytes) };
  new Uint8Array(wideMemory.buffer).set(random(bytes));
  const pieces = await openDevice(wide);
  handed(wideMemory, wide);
  await uploadLayers(pieces);
  // (a piece's first row is where the device binds its part of the output: a multiple of 256 / 4 rows)
  assert.deepEqual(pieces.weights.matrices.w1.pieces.map((piece) => [piece.first, piece.rows]), [[0, 64], [64, 64], [128, 32]], "w1's 160 rows of 32 bytes in pieces of 2048 bytes");
  assert.deepEqual(pieces.weights.matrices.wq.pieces.map((piece) => [piece.first, piece.rows]), [[0, Q_DIM]], "and a matrix that fits is one");
  for (const { first, rows, layers } of pieces.weights.matrices.w1.pieces) {
    const [values, scales] = layers[1], [valuesAt, scalesAt] = wide.matrices.w1.layers[1];
    assert.ok(same(values.data.subarray(0, rows * DIM), new Uint8Array(wideMemory.buffer, valuesAt + first * DIM, rows * DIM)), `the piece from row ${first}: its rows' values`);
    assert.ok(same(scales.data.subarray(0, rows * 4), new Uint8Array(wideMemory.buffer, scalesAt + first * 4, rows * 4)), "and their scales");
  }
}
{
  standIn();
  // (wk starts after wq's 1024 bytes, which is no multiple of the 256 a buffer binds at for the scales (128): not joined)
  const unbound = await openDevice({ ...plan, tokens: { classifier: { rows: 96, n: DIM, at: [0, 0] } } });
  handed(memory, unbound.plan);
  await uploadLayers(unbound);
  assert.equal(unbound.weights.unjoined, "wk would not start where this GPU binds a buffer", "why a token's layer could not be joined is the weights' to say");
  assert.equal(unbound.weights.together, null);
  assert.deepEqual(unbound.weights.joined, []);
}
console.log("gpu-worker-parts-check: weights.js makes m.weights: every matrix and vector on the GPU as the shared memory holds it, and says why a token's layer is not joined");

// ---- weights.js: a model on the GPU alone, its checkpoint coming in stretches
{
  const alone = standIn();
  const { plan: direct, bytes: size } = planned({ tables: { classifier: { rows: 96, n: DIM, at: [0, 0] } } });
  direct.tables.classifier.at = [size, size + 96 * DIM];
  const checkpoint = random(size + 96 * DIM + (96 * DIM / GROUP) * 4 + 64);
  const a = await openDevice({ ...direct, tokens: true });
  const layersUp = await uploadLayers(a, true), tablesUp = tablesOn(a);
  assert.equal(tablesUp, 96 * DIM + (96 * DIM / GROUP) * 4);
  assert.equal(layersUp + tablesUp, a.weights.routes.reduce((sum, [start, stop]) => sum + stop - start, 0), "every byte put up has a route");
  assert.ok(a.weights.routes.every(([start], i, routes) => !i || routes[i - 1][0] <= start), "the routes in the checkpoint's order");
  assert.ok(a.weights.routes.every(([, , target]) => target.usage & COPY_SRC), "COPY_SRC: the first layer and the tables' rows are read back for the check");
  assert.equal(alone.calls.filter(([what]) => what === "write").length, 0, "nothing is written before a byte comes");
  assert.equal(unwritten(a), false);
  // stretches of any size: one that ends in the middle of a word leaves its bytes waiting
  const [first] = a.weights.routes;
  routed(a, first[0], checkpoint.subarray(first[0], first[0] + 6));
  assert.equal(unwritten(a), true, "two bytes of a word wait for the other two");
  assert.ok(same(first[2].data.subarray(0, 4), checkpoint.subarray(first[0], first[0] + 4)), "the whole word of the stretch is written");
  assert.ok(first[2].data.subarray(4, 8).every((byte) => byte === 0), "the half word is not");
  routed(a, first[0] + 6, checkpoint.subarray(first[0] + 6, first[0] + 9));
  assert.ok(same(first[2].data.subarray(4, 8), checkpoint.subarray(first[0] + 4, first[0] + 8)), "and is once its other bytes came");
  const steps = [1, 2, 3, 5, 7, 250, 1001, 4, 6, 4099];
  for (let offset = first[0] + 9, k = 0; offset < checkpoint.length; k++) {
    const length = Math.min(steps[k % steps.length], checkpoint.length - offset);
    routed(a, offset, checkpoint.subarray(offset, offset + length));
    offset += length;
  }
  routed(a, 0, checkpoint.subarray(0, first[0]));  // (what comes before the first route, last: nothing of it has a route)
  assert.equal(unwritten(a), false, "every word is whole once the checkpoint has come");
  for (const [start, stop, target, at] of a.weights.routes) {
    assert.ok(same(target.data.subarray(at, at + stop - start), checkpoint.subarray(start, stop)), `the stretch at ${start} of the checkpoint is on the GPU as it came`);
  }
  // (the same stretch twice writes the same bytes; a stretch outside every route writes nothing)
  const before = alone.calls.length;
  routed(a, checkpoint.length - 64, checkpoint.subarray(checkpoint.length - 64));
  assert.equal(alone.calls.length, before, "bytes of the checkpoint that no buffer holds are written nowhere");
}
console.log("gpu-worker-parts-check: a model on the GPU alone: routed() writes a checkpoint that comes in stretches of any size as whole words, and unwritten() says a word that waits");

// ---- block.js: m.block, from the weights' record and a stand-in for the forms'
gpu.calls.length = 0;
// (forms.js makes m.forms once its checks have passed on a device that computes: here the test stands for it)
const attention = { name: "a stand-in attention", pipeline: { code: "attention", getBindGroupLayout: () => ({}) } };
const form = { name: "a stand-in form", pipeline: { code: "tiles", getBindGroupLayout: () => ({}) }, packed: true, tile: { rows: 4, tokens: 4 } };
m.forms = { attention, form, tried: [] };
await prepare(m);
{
  const b = m.block;
  assert.deepEqual(gpu.calls.filter(([what]) => what === "pipeline").map(([, pipeline]) => pipeline.code), [RMSNORM, HEAD_NORM, ADD, ROPE, SWIGLU, QUANTIZE],
    "the small steps' shaders of a model with RMSNorm and a gate, each compiled once");
  assert.deepEqual([b.norm.code, b.headNorm.code, b.add.code, b.rope.code, b.activation.code, b.quantize.code], [RMSNORM, HEAD_NORM, ADD, ROPE, SWIGLU, QUANTIZE]);
  assert.deepEqual([b.x.size, b.xb.size, b.q.size, b.k.size, b.v.size, b.gate.size, b.up.size, b.angles.size, b.step.size, b.readback.size],
    [BATCH * DIM * 4, BATCH * DIM * 4, BATCH * Q_DIM * 4, BATCH * KV_DIM * 4, BATCH * KV_DIM * 4, BATCH * HIDDEN * 4, BATCH * HIDDEN * 4, BATCH * HEAD * 4, 16, 2 * LAYERS * BATCH * KV_ROW]);
  assert.equal(b.ffnInput, b.xb, "no buffer of its own for the FFN's input without a parallel residual");
  assert.equal(b.quantizeFfn, b.quantizeXb);
  assert.equal(b.ropeAlone, null, "no layer RoPE leaves alone");
  assert.deepEqual([b.embedRows, b.ids, b.embedGroups], [null, null, null], "the embedding is the CPU's but for a model on the GPU alone");
  assert.deepEqual([b.layers, b.cache], [null, null], "the layers' bind groups and the cache are not made yet");
}
bindLayers(m);
assert.equal(m.block.layers.length, LAYERS);
for (const layer of m.block.layers) {
  for (const name of ["q", "k", "v", "o", "gate", "up", "down"]) assert.deepEqual(layer[name].map((piece) => piece.rows), [SIZES[{ q: "wq", k: "wk", v: "wv", o: "wo", gate: "w1", up: "w3", down: "w2" }[name]][0]], `a layer's ${name}: one piece of the matrix's rows`);
  assert.equal(layer.q[0].group.entries[2].resource.buffer, m.block.xq, "a packed form reads the quantized input");
  assert.equal(layer.q[0].group.entries.length, 7, "and its scales");
  assert.deepEqual([layer.biases, layer.headNorms, layer.oBias, layer.gateBias, layer.downBias], [[], [], undefined, undefined, undefined], "no bias and no norm of a head where the model has no such vector");
}
assert.equal(m.held, null, "nothing is held for end() before the cache is made");
grow(m, 8);
const firstCache = m.block.cache;
assert.equal(firstCache.capacity, 8);
assert.equal(firstCache.keys.length, LAYERS);
assert.equal(firstCache.keys[0].size, 8 * KV_ROW);
assert.equal(m.held, firstCache.owned, "grow() tells device.js which buffers the cache has now");
firstCache.keys[1].data.set(random(8 * KV_ROW));
firstCache.values[0].data.set(random(8 * KV_ROW).reverse());
roomFor(m, 8);
assert.equal(m.block.cache, firstCache, "roomFor(): nothing grows where the positions are held");
roomFor(m, 20);
const cache = m.block.cache;
assert.notEqual(cache, firstCache);
assert.equal(cache.capacity, 20, "twice the old capacity, or what is needed where that is more");
assert.ok(same(cache.keys[1].data.subarray(0, 8 * KV_ROW), firstCache.keys[1].data) && same(cache.values[0].data.subarray(0, 8 * KV_ROW), firstCache.values[0].data), "the old keys and values are copied to the new");
assert.ok(firstCache.owned.every((buffer) => buffer.destroyed), "the old cache's buffers are let go");
assert.equal(m.held, cache.owned, "and the new one's are the ones held");
assert.ok(cache.owned.every((buffer) => !buffer.destroyed));
console.log("gpu-worker-parts-check: block.js makes m.block from the weights' record and the forms', and the cache grows: copied, the old one let go, the new one's buffers held for end()");

// ---- block.js: a block of a prompt
{
  // a block past the cache grows it first, and its dispatches read the new one
  gpu.calls.length = 0;
  await block(m, 3, 19, () => false);
  assert.equal(m.block.cache.capacity, 40, "a block at positions past the cache grows it");
  assert.notEqual(m.block.cache, cache);
  assert.equal(gpu.calls.filter(([what]) => what === "dispatch").at(-1)[2], m.block.cache.rope[LAYERS - 1], "and writes its keys and values into the cache as it is now");
  assert.equal(m.held, m.block.cache.owned);
  roomFor(m, 1000);
  assert.equal(m.block.cache.capacity, CONTEXT, "never more than the context");
}
{
  const { cache: now, x, step, angles } = m.block, count = 5, pos = 3;
  const rows = new Float32Array(memory.buffer, plan.rows, count * DIM).map((_, i) => i / 7 - 3);
  new Float32Array(memory.buffer, plan.rows, count * DIM).set(rows);
  const cos = new Float32Array(memory.buffer, plan.cos, CONTEXT * HEAD / 2), sin = new Float32Array(memory.buffer, plan.sin, CONTEXT * HEAD / 2);
  cos.forEach((_, i) => { cos[i] = Math.cos(i); sin[i] = Math.sin(i); });
  // (what the layers "wrote" of these positions: the stand-in computes nothing, so the test puts them there)
  for (let l = 0; l < LAYERS; l++) {
    now.keys[l].data.set(random(now.capacity * KV_ROW).map((byte) => byte ^ (l + 1)));
    now.values[l].data.set(random(now.capacity * KV_ROW).map((byte) => byte ^ (0x80 + l)));
  }
  const staging = new Uint8Array(memory.buffer, plan.staging, 2 * LAYERS * BATCH * KV_ROW);
  staging.fill(0xee);
  gpu.calls.length = 0;
  await block(m, count, pos, () => false);
  assert.ok(staging.every((byte) => byte === 0xee), "a request given up writes nothing into the shared memory");
  const given = gpu.calls.filter(([what]) => what === "dispatch").length;
  gpu.calls.length = 0;
  await block(m, count, pos, () => true);
  assert.ok(same(new Float32Array(x.data.buffer, 0, count * DIM), rows), "the rows forward.js embedded go into x");
  assert.deepEqual([...new Uint32Array(step.data.buffer, 0, 4)], [count, pos, 0, 0], "the Step: how many tokens, from which position");
  const turned = new Float32Array(angles.data.buffer, 0, count * HEAD);
  assert.ok([...Array(count).keys()].every((t) => turned[t * HEAD] === cos[(pos + t) * HEAD / 2] && turned[t * HEAD + HEAD / 2] === sin[(pos + t) * HEAD / 2]),
    "the angles of the block's positions: a token's cos, then its sin");
  const dispatches = gpu.calls.filter(([what]) => what === "dispatch");
  // a layer: the norm, the quantizing, q, k, v, RoPE; and but for the last: the attention, its quantizing, o, the FFN's
  // norm, its quantizing, gate, up, SwiGLU, its quantizing, down
  assert.equal(dispatches.length, (LAYERS - 1) * 16 + 6, "the dispatches of a block of a packed form, a pass for all the layers");
  assert.equal(given, dispatches.length, "a request given up was dispatched all the same: only its answer is not written");
  assert.equal(dispatches.filter(([, pipeline]) => pipeline === attention.pipeline).length, LAYERS - 1, "the attention of every layer but the last");
  assert.deepEqual(dispatches.find(([, pipeline]) => pipeline === attention.pipeline).slice(3), [HEADS * Math.ceil(count / FLASH_Q_TILE), 1, 1]);
  assert.equal(dispatches.filter(([, pipeline]) => pipeline === form.pipeline).length, (LAYERS - 1) * 7 + 3, "the tiled shader: seven matrices a layer, three of the last");
  assert.deepEqual(dispatches[0].slice(1, 3), [m.block.norm, m.block.layers[0].attentionNorm], "the first dispatch: the first layer's norm");
  assert.equal(dispatches.at(-1)[2], now.rope[LAYERS - 1], "the last: the last layer's RoPE, into the cache");
  assert.equal(gpu.calls.filter(([what]) => what === "submit").length, 1, "one submission");
  for (let l = 0; l < LAYERS; l++) {
    assert.ok(same(staging.subarray(l * BATCH * KV_ROW, l * BATCH * KV_ROW + count * KV_ROW), now.keys[l].data.subarray(pos * KV_ROW, (pos + count) * KV_ROW)), `layer ${l}'s keys of the block's positions into the staging`);
    assert.ok(same(staging.subarray((LAYERS + l) * BATCH * KV_ROW, (LAYERS + l) * BATCH * KV_ROW + count * KV_ROW), now.values[l].data.subarray(pos * KV_ROW, (pos + count) * KV_ROW)), `layer ${l}'s values after all the keys`);
  }
}
// ---- block.js: keysIn(), forward.js's cache going up
{
  const { cache: now } = m.block, row = KV_ROW, capacity = 16, from = 2, pos = 7;
  const address = plan.rows;  // (any place of the memory: [keys, values] of LAYERS × capacity rows)
  const halves = { keys: address, values: address + LAYERS * capacity * row, capacity, row, half: true };
  new Uint8Array(memory.buffer, address, 2 * LAYERS * capacity * row).set(random(2 * LAYERS * capacity * row));
  for (let l = 0; l < LAYERS; l++) { now.keys[l].data.fill(0); now.values[l].data.fill(0); }
  keysIn(m, halves, from, pos);
  for (let l = 0; l < LAYERS; l++) {
    assert.ok(same(now.keys[l].data.subarray(from * KV_ROW, pos * KV_ROW), new Uint8Array(memory.buffer, halves.keys + (l * capacity + from) * row, (pos - from) * KV_ROW)), `layer ${l}'s keys of positions ${from} to ${pos - 1} go up as they are (float16)`);
    assert.ok(same(now.values[l].data.subarray(from * KV_ROW, pos * KV_ROW), new Uint8Array(memory.buffer, halves.values + (l * capacity + from) * row, (pos - from) * KV_ROW)), `and its values`);
    assert.ok(now.keys[l].data.subarray(0, from * KV_ROW).every((byte) => byte === 0) && now.keys[l].data.subarray(pos * KV_ROW).every((byte) => byte === 0), "no other position is written");
  }
  // a float32 cache (a grouped-query model's, T160): narrowed to float16 on the way up
  const floats = { keys: address, values: address + LAYERS * capacity * KV_DIM * 4, capacity, row: KV_DIM * 4, half: false };
  new Float32Array(memory.buffer, address, 2 * LAYERS * capacity * KV_DIM).forEach((_, i, all) => { all[i] = (i % 17) - 8; });
  keysIn(m, floats, from, pos);
  const half = (value) => { const view = new DataView(new ArrayBuffer(2)); view.setFloat16(0, value, true); return view.getUint16(0, true); };
  if (DataView.prototype.setFloat16) {
    const got = new Uint16Array(now.values[1].data.buffer, from * KV_ROW, (pos - from) * KV_DIM);
    const want = new Float32Array(memory.buffer, floats.values + (capacity + from) * KV_DIM * 4, (pos - from) * KV_DIM);
    assert.ok(got.every((bits, i) => bits === half(want[i])), "a float32 cache's values go up as float16");
  }
}
console.log("gpu-worker-parts-check: a block writes its rows, Step and angles, dispatches the layers in one submission, and answers into the staging only where it is still wanted; keysIn() puts forward.js's cache up");

// ---- device.js: the end, a stop, a device lost
{
  const owned = [...m.owned], kept = [...m.held];
  assert.ok(owned.length && kept.length);
  // (the record read as model is the one opened last: open this model's device again, for end() to let go of its buffers)
  const last = standIn();
  const again = await openDevice(plan);
  const buffers = [last.device.createBuffer({ size: 16, usage: STORAGE }), last.device.createBuffer({ size: 16, usage: STORAGE }), last.device.createBuffer({ size: 16, usage: STORAGE })];
  again.owned.push(buffers[0], buffers[1]);
  held(again, [buffers[2]]);
  last.calls.length = 0;
  posted.length = 0;
  end();
  assert.deepEqual(last.calls, [["destroy", buffers[0]], ["destroy", buffers[1]], ["destroy", buffers[2]], ["device gone"]],
    "end(): the model's own buffers, then the ones held last, then the device");
  assert.deepEqual(posted, [{ type: "ended" }]);
  assert.equal(closed, 1, "and the worker ends");
  assert.equal(device_.model, null, "no record after the end");
  end();
  assert.deepEqual(posted, [{ type: "ended" }, { type: "ended" }], "an end with nothing on the GPU says so too");
}
{
  const lost = standIn();
  await openDevice(plan);
  assert.equal(device_.lost, null);
  lost.lose({ reason: "destroyed", message: "the driver went away" });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(device_.lost, "the GPU was lost (destroyed: the driver went away)", "why the device was lost, for the modules that read it");
}
{
  const stopped = standIn();
  assert.equal(device_.stopping, false);
  device_.stopAsked();
  assert.equal(device_.stopping, true, "stopAsked(): the others read it as it is");
  posted.length = 0;
  assert.equal(await openDevice(plan), undefined, "a device opened after a stop was asked is none");
  assert.deepEqual(posted.map(({ type }) => type), ["ended"], "the worker ends where it was stopped");
  assert.equal(device_.model, null, "with nothing left on the GPU");
  assert.equal(stopped.calls.length, 0, "and nothing made on the device it was not given");
  // the weights going up end at their next step: one layer's, where the stop was asked before
  const one = await uploadLayers(m);
  assert.equal(one, Object.values(SIZES).reduce((sum, [rows, n]) => sum + rows * n + (rows * n / GROUP) * 4, 0), "uploadLayers() reads the stop as it is now: it ends after the layer it was at");
}
console.log("gpu-worker-parts-check: end() lets go of the model's own buffers, the ones held and the device, in that order; a stop asked and a device lost are read by all as device.js holds them");
console.log("gpu-worker-parts-check: ok");
process.exit(0);
