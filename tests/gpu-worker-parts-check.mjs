// tests/gpu-worker-parts-check.mjs (T.378.4): the parts of the model's GPU worker (src/runtime/gpu/) that hold its state,
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
//     dispatches and its keys and values into the staging, or nowhere where the request was given up; keysIn();
//   - requests.js: a prompt's block answered in the control area (done, failed, the beat), a request given up
//     answered nowhere, and a device lost before or during the work read as it is then: failed, and said.
//   - tokens.js (T.379.1), of what makes no pipeline: the WGSL of every form of a token's layer is the catalogue's
//     (shaders/engine.ts), and the matrices are those of the model's own packing of the weights.
// Not here, because their work is checked against JavaScript by a device that computes: forms.js, the rest of tokens.js,
// tokenattention.js, tokenforms.js (m.forms and m.tokens are made after checks pass) and start.js.
// The type check holds their names (tests/types.mjs), tests/gpu-modules-check.mjs their owners. Node only, under a second:
//
//   node tests/gpu-worker-parts-check.mjs
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { standIn } from "./fakes/webgpu.mjs";

// ---- the stand-in (tests/fakes/webgpu.mjs): a device that writes down what it is asked (calls) and keeps the bytes of every buffer
const posted = [];
let closed = 0;
globalThis.self = globalThis;
globalThis.postMessage = (message) => posted.push(message);
globalThis.close = () => { closed += 1; };

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
    eps: 1e-5, layerNorm: false, parallel: false, activation: "silu", windows: Array(LAYERS).fill(0),
    kinds: Array(LAYERS).fill("attention"), places: Array.from({ length: LAYERS }, (_, l) => l), local: [], localCos: 0, localSin: 0, unturned: [], matrices, vectors,
    tokens: null, force: {},
    rows: take(BATCH * DIM * 4), staging: take(2 * LAYERS * BATCH * KV_ROW), cos: take(CONTEXT * (HEAD / 2) * 4), sin: take(CONTEXT * (HEAD / 2) * 4), ...more };
  return { plan, bytes: at };
}
const random = (bytes) => Uint8Array.from({ length: bytes }, (_, i) => (i * 131 + 7 * (i >> 8) + 89) & 255);
const same = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);

const device_ = await import(runtimeUrl("gpu/device.js"));
const { openDevice, handed, held, end, STORAGE, COPY_DST, COPY_SRC } = device_;
const { upload, uploadLayers, tablesOn, routed, unwritten } = await import(runtimeUrl("gpu/weights.js"));
const { prepare, bindLayers, grow, roomFor, keysIn, block } = await import(runtimeUrl("gpu/block.js"));
const { prompt } = await import(runtimeUrl("gpu/requests.js"));
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
// ---- device.js: a step past its time is told as late (T.380.1), which the checks of the forms throw on (it ends the whole
// attempt) where they note any other failure and go on to the next form: nothing here runs those 180 s, so the timer is a
// stand-in that is fired by hand
{
  const { within, isLate } = device_;
  const real = globalThis.setTimeout, timers = [];
  globalThis.setTimeout = (callback, ms) => { timers.push([callback, ms]); return timers.length; };
  try {
    const late = within(new Promise(() => {}), "compiling a form");
    assert.deepEqual([timers.length, timers[0][1]], [1, 180000], "a step is given 180 s");
    timers[0][0]();
    const error = await late.then(() => undefined, (one) => one);
    assert.deepEqual([isLate(error), error.name, error.message], [true, "Error", "compiling a form took more than 180 s"], "what is thrown is late, and says it by its sentence alone");
    for (const other of [new Error("compiling a form took more than 180 s"), new TypeError("a shader is wrong"), Object.assign(new Error("x"), { late: true }), undefined, null, "late"]) {
      assert.equal(isLate(other), false, `${String(other)} is none that took too long`);
    }
    const done = await within(Promise.resolve(7), "something quick");
    assert.equal(done, 7, "a step in time is its own value");
  } finally {
    globalThis.setTimeout = real;
  }
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
  assert.deepEqual(Object.keys(W).sort(), ["angleTable", "finalBias", "finalNorm", "joined", "localAngleTable", "matrices", "partial", "positions", "routes", "signs", "tables", "together", "unjoined", "vectors", "widen"]);
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

// ---- weights.js, ternary (T.533): the checkpoint holds a group's scale as a float16, the GPU as a float32 (the shaders
// read float32s: their text and the device's key are what they were). Widened as they go up from the shared memory,
// and as a model on the GPU alone's checkpoint comes, in stretches of any size: a half that a stretch cuts waits
// for its other byte, as a word does
{
  // (a half's value written here, not got from the code under test; NaN and infinity are in the random bytes, too)
  const widenedHalf = (h) => {
    const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, fraction = h & 1023;
    return exponent === 0 ? sign * fraction * 2 ** -24 : exponent === 31 ? (fraction ? NaN : sign * Infinity) : sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
  };
  const TERNARY = { wq: [Q_DIM, 256], wk: [KV_DIM, 256], wv: [KV_DIM, 256], wo: [DIM, 256], w1: [HIDDEN, 256], w3: [HIDDEN, 256], w2: [DIM, 256] };
  const ternary = (made) => ({ ...made, matrices: Object.fromEntries(Object.entries(made.matrices).map(([name, matrix]) => [name, { ...matrix, ternary: true }])) });
  const floatsOf = (buffer, at, count) => new Float32Array(buffer.data.buffer, buffer.data.byteOffset + at, count);
  // from the shared memory (the CPU's side beside the GPU's, and a prompt on the GPU): the scales of a piece, widened
  {
    standIn({ language: ["packed_4x8_integer_dot_product"] });
    const { plan: made, bytes: size } = planned({}, TERNARY);
    const shared = { buffer: new SharedArrayBuffer(size) };
    new Uint8Array(shared.buffer).set(random(size));
    const model = await openDevice(ternary(made));
    handed(shared, model.plan);
    await uploadLayers(model);
    for (const [name, [rows, n]] of Object.entries(TERNARY)) {
      for (let l = 0; l < LAYERS; l++) {
        const [values, scales] = model.weights.matrices[name].pieces[0].layers[l], [valuesAt, scalesAt] = model.plan.matrices[name].layers[l];
        const groups = rows * n / 128, halves = new Uint16Array(shared.buffer, scalesAt, groups);
        assert.ok(same(values.data.subarray(0, rows * n / 4), new Uint8Array(shared.buffer, valuesAt, rows * n / 4)), `${name} of layer ${l}: its codes as the memory holds them`);
        assert.equal(scales.size ?? scales.data.length, groups * 4, `${name}: a float32 for each of the ${groups} halves`);
        assert.ok(floatsOf(scales, 0, groups).every((value, g) => Object.is(value, Math.fround(widenedHalf(halves[g])))), `${name} of layer ${l}: its scales, each half widened to a float32`);
      }
    }
  }
  // a model on the GPU alone: the checkpoint's stretches of any size (the steps are odd and even, so that a stretch
  // cuts a half in the middle and begins in one)
  {
    const alone = standIn({ language: ["packed_4x8_integer_dot_product"] });
    const { plan: made, bytes: size } = planned({ tables: { classifier: { rows: 96, n: 256, ternary: true, at: [0, 0] } } }, TERNARY);
    const direct = ternary(made);
    direct.tables.classifier.at = [size, size + 96 * 64];
    const checkpoint = random(size + 96 * 64 + 96 * 2 * 2 + 64);
    const a = await openDevice({ ...direct, tokens: true });
    const layersUp = await uploadLayers(a, true), tablesUp = tablesOn(a);
    const halved = a.weights.routes.filter(([, , , , halves]) => halves);
    assert.equal(halved.length, Object.keys(TERNARY).length * LAYERS + 1, "a route of halves for each ternary matrix of each layer, and the table's");
    assert.equal(tablesUp, 96 * 64 + 96 * 2 * 4, "the table: its codes, and its scales as the GPU holds them (float32)");
    assert.equal(layersUp + tablesUp, a.weights.routes.reduce((sum, [start, stop, , , halves]) => sum + (stop - start) * (halves ? 2 : 1), 0), "every byte put up has a route, a half's twice");
    const steps = [1, 2, 3, 5, 7, 250, 1001, 4, 6, 4099, 1, 1];
    for (const step of [steps, [1], [2], [3]]) {
      for (const [, , target] of a.weights.routes) target.data.fill(0);
      a.weights.partial.clear();
      for (let offset = 0, k = 0; offset < checkpoint.length; k++) {
        const length = Math.min(step[k % step.length], checkpoint.length - offset);
        routed(a, offset, checkpoint.subarray(offset, offset + length));
        offset += length;
      }
      assert.equal(unwritten(a), false, `every half is whole once the checkpoint has come (stretches of ${step.slice(0, 3)}...)`);
      for (const [start, stop, target, at, halves] of a.weights.routes) {
        if (!halves) {
          assert.ok(same(target.data.subarray(at, at + stop - start), checkpoint.subarray(start, stop)), `the codes at ${start} are on the GPU as they came`);
          continue;
        }
        const count = (stop - start) / 2, got = floatsOf(target, at, count);
        for (let j = 0; j < count; j++) {
          const want = Math.fround(widenedHalf(checkpoint[start + 2 * j] | (checkpoint[start + 2 * j + 1] << 8)));
          assert.ok(Object.is(got[j], want), `the scale ${j} of the stretch at ${start} (stretches of ${step.slice(0, 3)}...): ${got[j]} for ${want}`);
        }
      }
    }
    // a half that waits is told, and a stretch that begins in the middle of a half is held to its low byte
    for (const [, , target] of a.weights.routes) target.data.fill(0);
    const [start] = halved[0];
    routed(a, start + 1, checkpoint.subarray(start + 1, start + 2));
    assert.equal(unwritten(a), true, "the high byte of a half waits for the low one");
    routed(a, start, checkpoint.subarray(start, start + 1));
    assert.equal(unwritten(a), false, "and is widened and written once it came");
    assert.ok(Object.is(floatsOf(halved[0][2], halved[0][3], 1)[0], Math.fround(widenedHalf(checkpoint[start] | (checkpoint[start + 1] << 8)))), "the half's float32");
    assert.ok(alone.calls.some(([what]) => what === "write"));
  }
}
console.log("gpu-worker-parts-check: a ternary model's float16 scales go up as float32s, from the shared memory and as the checkpoint of a model on the GPU alone comes in stretches of any size");

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
  for (const name of ["gate", "up", "down"]) assert.deepEqual(layer[name].map((piece) => piece.rows), [SIZES[{ gate: "w1", up: "w3", down: "w2" }[name]][0]], `a layer's ${name}: one piece of the matrix's rows`);
  assert.equal(layer.gate[0].group.entries[2].resource.buffer, m.block.xq, "a packed form reads the quantized input");
  assert.equal(layer.gate[0].group.entries.length, 7, "and its scales");
  assert.deepEqual([layer.gateBias, layer.downBias], [undefined, undefined], "no bias where the model has no such vector");
  assert.equal(typeof layer.step, "function", "the layer's kind's step (T.568.2: gpu/kinds.js)");
}
assert.equal(m.held, null, "nothing is held for end() before the cache is made");
grow(m, 8);
const firstCache = m.block.cache;
assert.equal(firstCache.capacity, 8);
assert.equal(firstCache.keys.length, LAYERS);
assert.equal(firstCache.keys[0].size, 8 * KV_ROW);
assert.equal(m.held, firstCache.owned, "grow() tells device.js which buffers the cache has now");
// (T.568.2: an attending layer's step, what it asks block.js to dispatch for it: its products in order, q's reading the
// quantized input; no bias and no norm of a head where the model has no such vector)
{
  const products = [], biases = [], dispatched = [], quantized = [];
  const pass = { setPipeline(pipeline) { pass.pipeline = pipeline; }, setBindGroup(index, group) { pass.group = group; }, dispatchWorkgroups(x) { dispatched.push([pass.pipeline, pass.group, x]); } };
  const ended = m.block.layers[0].step({ pass, count: 5, last: false, multiplied: (pieces) => products.push(pieces), quantized: (input) => quantized.push(input),
    biased: (bias) => biases.push(bias), ffnNorm: () => assert.fail("no parallel residual") });
  assert.equal(ended, false, "a layer that is not the last does not end the block");
  assert.deepEqual(products.map((pieces) => pieces.map((piece) => piece.rows)), [[Q_DIM], [KV_DIM], [KV_DIM], [DIM]], "q, k, v, then o: one piece of each matrix's rows");
  assert.equal(products[0][0].group.entries[2].resource.buffer, m.block.xq, "q reads the quantized input");
  assert.deepEqual(quantized, [m.block.quantizeAttention], "o's input is the attention's output, quantized");
  assert.deepEqual(biases, [undefined], "no bias after o (and none of q, k and v asked for)");
  assert.deepEqual(dispatched.map(([pipeline, group]) => [pipeline, group]), [[m.block.rope, firstCache.rope[0]], [m.forms.attention.pipeline, firstCache.attention[0]]],
    "RoPE into the layer's cache, then the attention over it: no norm of a head");
  assert.equal(m.block.layers[1].step({ pass, count: 5, last: true, multiplied() {}, quantized() {}, biased() {}, ffnNorm() {} }), true, "the last layer ends the block after RoPE");
}
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
  // (T.261.4.2: where each layer's are: here the blocks of a cache of LAYERS × capacity rows each)
  const placesOf = (bytes) => Array.from({ length: LAYERS }, (_, l) => ({ keys: address + l * capacity * bytes, values: address + (LAYERS + l) * capacity * bytes, ring: 0 }));
  const halves = { places: placesOf(row), row, half: true };
  new Uint8Array(memory.buffer, address, 2 * LAYERS * capacity * row).set(random(2 * LAYERS * capacity * row));
  for (let l = 0; l < LAYERS; l++) { now.keys[l].data.fill(0); now.values[l].data.fill(0); }
  keysIn(m, halves, from, pos);
  for (let l = 0; l < LAYERS; l++) {
    assert.ok(same(now.keys[l].data.subarray(from * KV_ROW, pos * KV_ROW), new Uint8Array(memory.buffer, address + (l * capacity + from) * row, (pos - from) * KV_ROW)), `layer ${l}'s keys of positions ${from} to ${pos - 1} go up as they are (float16)`);
    assert.ok(same(now.values[l].data.subarray(from * KV_ROW, pos * KV_ROW), new Uint8Array(memory.buffer, address + ((LAYERS + l) * capacity + from) * row, (pos - from) * KV_ROW)), `and its values`);
    assert.ok(now.keys[l].data.subarray(0, from * KV_ROW).every((byte) => byte === 0) && now.keys[l].data.subarray(pos * KV_ROW).every((byte) => byte === 0), "no other position is written");
  }
  // a float32 cache (a grouped-query model's, T.160): narrowed to float16 on the way up
  const floats = { places: placesOf(KV_DIM * 4), row: KV_DIM * 4, half: false };
  new Float32Array(memory.buffer, address, 2 * LAYERS * capacity * KV_DIM).forEach((_, i, all) => { all[i] = (i % 17) - 8; });
  keysIn(m, floats, from, pos);
  const half = (value) => { const view = new DataView(new ArrayBuffer(2)); view.setFloat16(0, value, true); return view.getUint16(0, true); };
  if (DataView.prototype.setFloat16) {
    const got = new Uint16Array(now.values[1].data.buffer, from * KV_ROW, (pos - from) * KV_DIM);
    const want = new Float32Array(memory.buffer, floats.places[1].values + from * KV_DIM * 4, (pos - from) * KV_DIM);
    assert.ok(got.every((bits, i) => bits === half(want[i])), "a float32 cache's values go up as float16");
  }
  // T.261.4.2: a layer whose window a ring of 3 positions holds (T.510.1: position p at p mod 3 and at p mod 3 + 3): the
  // last three positions before pos and no more, from where the oldest of them is, wrapped or not
  for (const late of [7, 8]) {
    const ring = 3, rows = new Uint8Array(memory.buffer, address, 4 * ring * row);
    rows.set(random(4 * ring * row));
    const ringed = { places: [{ keys: address, values: address + 2 * ring * row, ring }, ...placesOf(row).slice(1)], row, half: true };
    for (let l = 0; l < LAYERS; l++) { now.keys[l].data.fill(0); now.values[l].data.fill(0); }
    keysIn(m, ringed, from, late);
    const slot = (late - ring) % ring;
    assert.ok(same(now.keys[0].data.subarray((late - ring) * KV_ROW, late * KV_ROW), rows.subarray(slot * row, (slot + ring) * row)), `positions ${late - ring} to ${late - 1} of a ring, from its slot ${slot}`);
    assert.ok(same(now.values[0].data.subarray((late - ring) * KV_ROW, late * KV_ROW), rows.subarray((2 * ring + slot) * row, (3 * ring + slot) * row)), "and their values");
    assert.ok(now.keys[0].data.subarray(0, (late - ring) * KV_ROW).every((byte) => byte === 0), "none older: the ring no longer holds them");
  }
}
console.log("gpu-worker-parts-check: a block writes its rows, Step and angles, dispatches the layers in one submission, and answers into the staging only where it is still wanted; keysIn() puts forward.js's cache up");

// ---- requests.js: a prompt's block answered in the control area, by the model device.js holds now
{
  const now = standIn(), WORDS = { beat: 0, wanted: 1, failed: 2, done: 3 };
  const record = await openDevice({ ...plan, words: WORDS });
  handed(memory, record.plan);
  await upload(record);
  record.forms = { attention, form, tried: [] };
  await prepare(record);
  bindLayers(record);
  grow(record, 16);
  const words = new Int32Array(memory.buffer, 0, 4), staging = new Uint8Array(memory.buffer, plan.staging, 2 * LAYERS * BATCH * KV_ROW);
  const asked = async (serial, { wanted = serial } = {}) => {
    Atomics.store(words, WORDS.wanted, wanted);
    Atomics.store(words, WORDS.failed, 7);
    prompt({ serial, count: 4, pos: 0 });
    for (let i = 0; i < 50 && Atomics.load(words, WORDS.done) !== serial; i++) await new Promise((resolve) => setTimeout(resolve, 0));
    return { done: Atomics.load(words, WORDS.done), failed: Atomics.load(words, WORDS.failed) };
  };
  record.block.cache.keys[0].data.fill(0x11);
  staging.fill(0xee);
  posted.length = 0;
  assert.deepEqual(await asked(1), { done: 1, failed: 0 }, "a block is answered in the control area: done says its serial, failed 0");
  assert.equal(staging[0], 0x11, "with its keys and values in the staging");
  assert.deepEqual(posted, []);
  staging.fill(0xee);
  assert.deepEqual(await asked(2, { wanted: 3 }), { done: 1, failed: 7 }, "a request forward.js gave up is answered nowhere: the words are another request's by then");
  assert.ok(staging.every((byte) => byte === 0xee), "nor in the staging");
  // the device lost while the block is at work: read after the work as it is then
  now.device.onSubmit = () => { now.device.onSubmit = null; now.lose({ reason: "unknown", message: "" }); };
  assert.equal(device_.lost, null);
  assert.deepEqual(await asked(4), { done: 4, failed: 1 }, "a device lost during the work fails the request");
  assert.deepEqual(posted, [{ type: "failed", reason: "the GPU was lost (unknown)" }], "and says why");
  // and before it: nothing is asked of the device
  now.calls.length = 0;
  assert.deepEqual(await asked(5), { done: 5, failed: 1 }, "a request after the loss fails at once");
  assert.equal(now.calls.length, 0, "with nothing asked of the device that is gone");
}
console.log("gpu-worker-parts-check: requests.js answers a block in the control area, a request given up nowhere, and reads a device lost before or during the work as it is then");

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
  lost.lose({ reason: "destroyed", message: "the driver went away" });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(device_.lost, "the GPU was lost (destroyed: the driver went away)", "why the device was lost last, with what the device said, for the modules that read it");
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

// ---- T.379.1: tokens.js's forms of a token's layer compile the catalogue's texts, and the ones of the model's weights.
// (tokenCandidates() and tokenCodes() make no pipeline: which forms a device and a model may take, and each form's WGSL.)
// Every text is a row's of shaders/engine.ts (what the device's key hashes), a ternary model's matrices are the rows of
// the ternary packing and no other model's are: the worker gives the maker the packing, and a packing given the wrong
// way round is a shader of one kind of weights run on the other (a form refused on every device, or wrong steps).
{
  const { tokenCandidates, tokenCodes } = await import(runtimeUrl("gpu/tokens.js"));
  const { ENGINE_SHADERS, TERNARY_SHADERS } = await import(runtimeUrl("shaders.js"));
  const rowOf = new Map([...ENGINE_SHADERS, ...TERNARY_SHADERS].map((row) => [row.text(), row]));
  const was = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { wgslLanguageFeatures: new Set(["packed_4x8_integer_dot_product", "subgroup_id"]) } } });
  let forms = 0, codes = 0;
  for (const ternary of [false, true]) for (const layerNorm of [false, true]) {
    const candidates = tokenCandidates({ ternary, device: { features: new Set(["subgroups"]) }, plan: { layerNorm } });
    assert.ok(candidates.length > 0, "a device with subgroups and the packed dot has forms of a token's layer");
    for (const form of candidates) for (const apart of [false, true]) for (const gated of [false, true]) for (const floatHead of [false, true]) {
      forms += 1;
      for (const [key, code] of tokenCodes(form, { apart, gated, layerNorm, floatHead })) {
        const row = rowOf.get(code), said = `${form.name}${ternary ? " (a ternary model)" : ""}, ${key}`;
        assert.ok(row, `${said}: a text of no row of the catalogue, which the device's key does not hold`);
        codes += 1;
        if (row.name !== "fusedDp4aMatVec") continue;
        assert.equal(row.given.weights, ternary ? "ternary" : "int8", `${said}: the matrix of the other packing of the weights`);
        assert.equal(TERNARY_SHADERS.includes(row), ternary, `${said}: a ternary model's shaders are its own rows, and no other model takes them`);
      }
    }
  }
  if (was) Object.defineProperty(globalThis, "navigator", was); else delete globalThis.navigator;
  console.log(`gpu-worker-parts-check: tokens.js's ${forms} forms of a token's layer (by the model's packing, its norm and its shape) compile ${codes} texts, each a row of the catalogue and of the model's own packing`);
}
console.log("gpu-worker-parts-check: ok");
process.exit(0);
