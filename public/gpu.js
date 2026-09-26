// gpu.js (T135): the model page's GPU, in a worker of its own. forward.js (in the model's worker) makes it where the
// page asked for it (?gpu=on) and the model is one it takes, and hands it the blocks of a prompt (up to BATCH tokens)
// one at a time, waiting in Atomics.wait for the answer: the model's worker cannot wait for a promise while Python
// calls forwardMany(), and this one can (T94's design; T134's bridge measured 12 to 95 µs a round trip).
//
//   { type: "start", memory, plan }  the layers of the model onto the GPU, read from the shared memory of forward.js
//                                    (plan: what forward.js says of them, addresses in that memory). Answers
//                                    { type: "progress" } after every layer, then { type: "ready", adapter, bytes,
//                                    seconds } or { type: "unusable", reason }
//   { type: "prompt", serial, count, pos }
//                                    the rows of count tokens at positions pos, pos + 1, ... (embedded by forward.js,
//                                    in its frames) through every layer, and their keys and values of every layer into
//                                    plan.staging. The answer is in the control area: words.failed, then
//                                    words.done = serial, while words.beat counts up meanwhile; a failure also says
//                                    { type: "failed", reason }
//   { type: "stop" }                 every buffer and the device let go, and the worker ends
//
// A block goes as one submission: per layer the RMSNorm, the matrices of q, k and v (each weight read once for the
// block's tokens: shaders.js's BATCHED), RoPE with the keys and values into the GPU's own cache, the attention (every
// token sees the positions up to its own), the output matrix added to the residual, the RMSNorm, the gate and the up
// matrices, SwiGLU, the down matrix added. The last layer stops at its keys and values: nothing of a prompt's token
// after them is used. Then the block's keys and values of every layer are copied out and read back. In float32
// throughout, the weights widened in the shader (the CPU quantizes the activations as well: tests/gpu-check.mjs holds
// the two together).
//
// The first message is claimed before anything is awaited (a module worker's port opens at its first await, and a
// message that comes before onmessage is set is lost: T109).

const shaders = import(new URL(`shaders.js${new URL(import.meta.url).search}`, import.meta.url));

// GPUBufferUsage's values (the name itself is missing where there is no WebGPU)
const STORAGE = 0x80, COPY_DST = 0x8, COPY_SRC = 0x4, MAP_READ = 0x1, UNIFORM = 0x40;
// the bytes that go to the GPU through a copy of their own at a time (writeBuffer takes no view of a shared memory
// everywhere, and copies what it is given at once)
const CHUNK = 8 << 20;
// the attention's workgroup holds a head of q
const MOST_HEAD = 256;

let model = null;  // what is on the GPU for the model: the device, the plan, the buffers, the pipelines, the cache
let starting = false, stopping = false, lost = null;

onmessage = ({ data }) => {
  if (data.type === "start") start(data.memory, data.plan);
  else if (data.type === "prompt") prompt(data);
  else if (data.type === "stop") stop();
};

async function start(memory, plan) {
  const began = performance.now();
  starting = true;
  try {
    const wgsl = await shaders;
    if (!self.navigator?.gpu) return unusable("no WebGPU in a worker here");
    if (plan.headSize > MOST_HEAD) return unusable(`heads of more than ${MOST_HEAD} values are not on the GPU yet`);
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (stopping) return end();
    if (!adapter) return unusable("no GPU adapter here");
    // the largest buffer of the layers: one of a layer's matrices
    const largest = Math.max(...Object.values(plan.matrices).map(({ rows, n }) => rows * n));
    const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
    if (largest > limit) return unusable(`a matrix of ${megabytes(largest)} is more than a buffer of this GPU (${megabytes(limit)})`);
    const device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    device.lost.then((info) => { lost = `the GPU was lost (${info.reason}${info.message ? `: ${info.message}` : ""})`; });
    model = { device, memory, plan, wgsl, owned: [], limit };
    if (stopping) return end();
    // a buffer the device cannot give fails quietly, as an error of these scopes
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    const bytes = await upload(model);
    if (stopping) return end();
    await prepare(model);
    grow(model, Math.min(plan.kvStart, plan.seqLen));
    await device.queue.onSubmittedWorkDone();
    const invalid = await device.popErrorScope(), full = await device.popErrorScope();
    if (stopping) return end();
    if (invalid || full) return unusable(`the GPU did not take the layers (${(invalid ?? full).message})`);
    if (lost) return unusable(lost);
    postMessage({ type: "ready", adapter: describe(adapter), bytes, seconds: (performance.now() - began) / 1000 });
  } catch (error) {
    unusable(String(error?.message ?? error));
  } finally {
    starting = false;
  }
}

const megabytes = (bytes) => `${Math.round(bytes / 2 ** 20)} MiB`;
function describe(adapter) {
  const info = adapter.info ?? {};
  const name = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "a GPU";
  return (info.isFallbackAdapter ?? adapter.isFallbackAdapter) ? `${name} (a fallback adapter: the CPU in the GPU's place)` : name;
}

function unusable(reason) {
  postMessage({ type: "unusable", reason });
  end();
}
function stop() {
  stopping = true;
  if (!starting) end();  // else start() ends at its next step
}
// every buffer and the device let go (T94: a model changed for another leaves nothing on the GPU), and this worker ends
function end() {
  if (model) {
    model.owned.forEach((buffer) => buffer.destroy());
    model.cache?.owned.forEach((buffer) => buffer.destroy());
    model.device.destroy();
    model = null;
  }
  self.close();
}

// ---- buffers
function buffer(m, bytes, usage = STORAGE, owned = m.owned) {
  const made = m.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
  owned.push(made);
  return made;
}
function uniform(m, data) {
  const made = buffer(m, data.byteLength, UNIFORM | COPY_DST);
  m.device.queue.writeBuffer(made, 0, data);
  return made;
}
// bytes of the shared memory at address onto the GPU, a CHUNK at a time through a copy
let scratch;
function copyIn(m, target, address, bytes) {
  scratch ??= new Uint8Array(CHUNK);
  for (let done = 0; done < bytes; done += CHUNK) {
    const n = Math.min(CHUNK, bytes - done);
    scratch.set(new Uint8Array(m.memory.buffer, address + done, n));
    m.device.queue.writeBuffer(target, done, scratch, 0, n);
  }
}

// every layer's matrices (values and scales, as the checkpoint holds them) and the weights of its two norms
async function upload(m) {
  const { plan } = m, group = m.wgsl.GROUP;
  let bytes = 0;
  m.matrices = Object.fromEntries(Object.entries(plan.matrices).map(([name, { rows, n }]) => [name, { rows, n, layers: [] }]));
  for (let l = 0; l < plan.layers; l++) {
    for (const [name, matrix] of Object.entries(plan.matrices)) {
      const [valuesAt, scalesAt] = matrix.layers[l], valueBytes = matrix.rows * matrix.n, scaleBytes = (valueBytes / group) * 4;
      const values = buffer(m, valueBytes, STORAGE | COPY_DST), scales = buffer(m, scaleBytes, STORAGE | COPY_DST);
      copyIn(m, values, valuesAt, valueBytes);
      copyIn(m, scales, scalesAt, scaleBytes);
      m.matrices[name].layers.push([values, scales]);
      bytes += valueBytes + scaleBytes;
    }
    // what was written waits in memory until the GPU takes it: let it, before more comes, and say that this moves
    await m.device.queue.onSubmittedWorkDone();
    if (stopping) return bytes;
    postMessage({ type: "progress", layer: l + 1 });
  }
  const normBytes = plan.layers * plan.dim * 4;
  m.norms = {};
  for (const [name, address] of Object.entries(plan.norms)) {
    m.norms[name] = buffer(m, normBytes, STORAGE | COPY_DST);
    copyIn(m, m.norms[name], address, normBytes);
    bytes += normBytes;
  }
  return bytes;
}

// the pipelines, the buffers of a block's activations, and every layer's bind groups but those of the cache
async function prepare(m) {
  const { device, plan, wgsl } = m, B = plan.batch;
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize;
  const make = (code) => device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  [m.matmul, m.norm, m.rope, m.attention, m.swiglu] = await Promise.all([wgsl.BATCHED, wgsl.RMSNORM, wgsl.ROPE, wgsl.ATTENTION, wgsl.SWIGLU].map(make));
  // a block's tokens, each array dense: token t's row at t times its width
  m.x = buffer(m, B * plan.dim * 4, STORAGE | COPY_DST);
  m.xb = buffer(m, B * Math.max(plan.dim, qDim) * 4);
  m.q = buffer(m, B * qDim * 4);
  m.k = buffer(m, B * kvDim * 4);
  m.v = buffer(m, B * kvDim * 4);
  m.gate = buffer(m, B * plan.hidden * 4);
  m.up = buffer(m, B * plan.hidden * 4);
  m.angles = buffer(m, B * plan.headSize * 4, STORAGE | COPY_DST);
  m.step = buffer(m, 16, UNIFORM | COPY_DST);
  m.readback = buffer(m, 2 * plan.layers * B * kvDim * 4, MAP_READ | COPY_DST);
  // what a request writes into them, from the shared memory
  m.rows = new Float32Array(B * plan.dim);
  m.turns = new Float32Array(B * plan.headSize);
  m.ropeShape = uniform(m, new Uint32Array([plan.heads, plan.kvHeads, plan.headSize, plan.turned]));
  m.swigluGroup = bind(m, m.swiglu, [m.gate, m.up, uniform(m, new Uint32Array([plan.hidden, 0, 0, 0])), m.step]);
  m.layers = [];
  for (let l = 0; l < plan.layers; l++) {
    const norm = new ArrayBuffer(16);
    new Uint32Array(norm, 0, 2).set([plan.dim, l * plan.dim]);
    new Float32Array(norm, 8, 1)[0] = plan.eps;
    const normShape = uniform(m, norm);
    m.layers.push({
      attentionNorm: bind(m, m.norm, [m.x, m.norms.attention, m.xb, normShape, m.step]),
      q: product(m, "wq", l, m.xb, m.q), k: product(m, "wk", l, m.xb, m.k), v: product(m, "wv", l, m.xb, m.v),
      o: product(m, "wo", l, m.xb, m.x, true),
      ffnNorm: bind(m, m.norm, [m.x, m.norms.ffn, m.xb, normShape, m.step]),
      gate: product(m, "w1", l, m.xb, m.gate), up: product(m, "w3", l, m.xb, m.up),
      down: product(m, "w2", l, m.gate, m.x, true),
    });
  }
}
const bind = (m, pipeline, buffers) => m.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
// a layer's matrix by the tokens in from, into to (added to what to holds where add): its bind group and workgroups
function product(m, name, l, from, to, add = false) {
  const { rows, n, layers } = m.matrices[name];
  const shape = uniform(m, new Uint32Array([rows, n / 4, n / m.wgsl.GROUP, 0, n, rows, add ? 1 : 0, 0]));
  const across = Math.min(rows, m.device.limits.maxComputeWorkgroupsPerDimension);
  return { group: bind(m, m.matmul, [...layers[l], from, to, shape, m.step]), x: across, y: Math.ceil(rows / across) };
}

// The GPU's own keys and values, per layer [positions][kvDim], grown as the CPU's cache grows (doubling from
// plan.kvStart: a prompt seldom needs the whole context) and kept from block to block; with them the attention's
// scores and the bind groups that read them
function grow(m, needed) {
  const { device, plan } = m, old = m.cache, B = plan.batch;
  const kvDim = plan.kvHeads * plan.headSize;
  const capacity = Math.min(Math.max(2 * (old?.capacity ?? 0), needed), plan.seqLen);
  if (capacity * kvDim * 4 > m.limit || B * plan.heads * capacity * 4 > m.limit) {
    throw new Error(`the keys of ${capacity} positions are more than a buffer of this GPU`);
  }
  const cache = { capacity, owned: [], keys: [], values: [], rope: [], attention: [] };
  const usage = STORAGE | COPY_SRC | COPY_DST;
  const encoder = device.createCommandEncoder();
  for (let l = 0; l < plan.layers; l++) {
    const keys = buffer(m, capacity * kvDim * 4, usage, cache.owned), values = buffer(m, capacity * kvDim * 4, usage, cache.owned);
    if (old) {
      encoder.copyBufferToBuffer(old.keys[l], 0, keys, 0, old.capacity * kvDim * 4);
      encoder.copyBufferToBuffer(old.values[l], 0, values, 0, old.capacity * kvDim * 4);
    }
    cache.keys.push(keys);
    cache.values.push(values);
  }
  device.queue.submit([encoder.finish()]);
  // the old ones go once the copies are done (a buffer destroyed after its submission lives until the GPU is through)
  old?.owned.forEach((buffer) => buffer.destroy());
  const scores = buffer(m, B * plan.heads * capacity * 4, STORAGE, cache.owned);
  const shape = new ArrayBuffer(32);
  new Uint32Array(shape, 0, 4).set([plan.heads, plan.kvHeads, plan.headSize, capacity]);
  new Float32Array(shape, 16, 1)[0] = 1 / Math.sqrt(plan.headSize);
  const attentionShape = buffer(m, 32, UNIFORM | COPY_DST, cache.owned);
  device.queue.writeBuffer(attentionShape, 0, shape);
  for (let l = 0; l < plan.layers; l++) {
    cache.rope.push(bind(m, m.rope, [m.q, m.k, m.v, cache.keys[l], cache.values[l], m.angles, m.ropeShape, m.step]));
    cache.attention.push(bind(m, m.attention, [m.q, cache.keys[l], cache.values[l], scores, m.xb, attentionShape, m.step]));
  }
  m.cache = cache;
}

// ---- a block of a prompt
async function prompt({ serial, count, pos }) {
  const { memory, plan } = model;
  const words = new Int32Array(memory.buffer, 0, plan.words.beat + 1);
  // the model's worker waits: this says that the work goes on, however long the GPU takes (a software adapter)
  const beat = setInterval(() => Atomics.add(words, plan.words.beat, 1), 250);
  let failed = 1;
  try {
    if (lost) throw new Error(lost);
    await block(model, count, pos);
    if (lost) throw new Error(lost);
    failed = 0;
  } catch (error) {
    postMessage({ type: "failed", reason: String(error?.message ?? error) });
  } finally {
    clearInterval(beat);
    Atomics.store(words, plan.words.failed, failed);
    Atomics.store(words, plan.words.done, serial);
    Atomics.notify(words, plan.words.done);
  }
}

async function block(m, count, pos) {
  const { device, plan } = m, B = plan.batch, half = plan.headSize / 2, kvDim = plan.kvHeads * plan.headSize;
  // the rows forward.js embedded (a frame apart), and the angles of their positions (the tables forward.js has)
  const F = new Float32Array(m.memory.buffer);
  for (let t = 0; t < count; t++) {
    const row = (plan.x + t * plan.frame) / 4, cos = plan.cos / 4 + (pos + t) * half, sin = plan.sin / 4 + (pos + t) * half;
    m.rows.set(F.subarray(row, row + plan.dim), t * plan.dim);
    m.turns.set(F.subarray(cos, cos + half), t * plan.headSize);
    m.turns.set(F.subarray(sin, sin + half), t * plan.headSize + half);
  }
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  device.queue.writeBuffer(m.x, 0, m.rows, 0, count * plan.dim);
  device.queue.writeBuffer(m.angles, 0, m.turns, 0, count * plan.headSize);
  device.queue.writeBuffer(m.step, 0, new Uint32Array([count, pos, 0, 0]));
  if (pos + count > m.cache.capacity) grow(m, pos + count);
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  const dispatch = (pipeline, group, x, y = 1, z = 1) => {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, y, z);
  };
  const tiles = Math.ceil(count / m.wgsl.TILE);
  const multiply = (product) => dispatch(m.matmul, product.group, product.x, product.y, tiles);
  for (let l = 0; l < plan.layers; l++) {
    const layer = m.layers[l];
    dispatch(m.norm, layer.attentionNorm, count);
    multiply(layer.q);
    multiply(layer.k);
    multiply(layer.v);
    dispatch(m.rope, m.cache.rope[l], count);
    if (l === plan.layers - 1) break;  // the keys and values are all a prompt's token leaves
    dispatch(m.attention, m.cache.attention[l], plan.heads, count);
    multiply(layer.o);
    dispatch(m.norm, layer.ffnNorm, count);
    multiply(layer.gate);
    multiply(layer.up);
    dispatch(m.swiglu, m.swigluGroup, Math.ceil(plan.hidden / 64), count);
    multiply(layer.down);
  }
  pass.end();
  // the block's keys of every layer, then its values, [layers][B][kvDim] each, as plan.staging lays them out
  for (let l = 0; l < plan.layers; l++) {
    encoder.copyBufferToBuffer(m.cache.keys[l], pos * kvDim * 4, m.readback, l * B * kvDim * 4, count * kvDim * 4);
    encoder.copyBufferToBuffer(m.cache.values[l], pos * kvDim * 4, m.readback, (plan.layers + l) * B * kvDim * 4, count * kvDim * 4);
  }
  device.queue.submit([encoder.finish()]);
  const invalid = await device.popErrorScope(), full = await device.popErrorScope();
  if (invalid || full) throw new Error(`the GPU refused a block (${(invalid ?? full).message})`);
  await m.readback.mapAsync(MAP_READ);
  try {
    const floats = 2 * plan.layers * B * kvDim;
    new Float32Array(m.memory.buffer, plan.staging, floats).set(new Float32Array(m.readback.getMappedRange(), 0, floats));
  } finally {
    m.readback.unmap();
  }
}
