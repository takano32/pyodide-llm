// The GPU section of /benchmark/ (T134; T94's stage 0 until then, at /gpu-test/): what WebGPU gives this device,
// measured in a worker (where a GPU forward pass would run). The page (src/pages/benchmark.astro) asks for one step at
// a time and shows what comes back, and ends the worker after the section; nothing here touches the model page.
//
//   { step: "info" }                      the adapter, its limits and features, WGSL's language features
//   { step: "check" }                     the two int8 shaders against JavaScript on a small matrix
//   { step: "bandwidth", shape }          GB/s of one int8 matrix times a vector, both shaders, and the CPU's
//   { step: "token", model }              a whole token's work of a model's shapes (every layer's matrices, a few small
//                                         dispatches, the classifier, the logits read back), ms per token
//   { step: "bridge", memory, rounds }    the round trip of a worker that waits with Atomics.wait and this one, which
//                                         answers with Atomics.waitAsync (stage 1's design), in microseconds
//
// The weights are random: only their size and layout matter. int8 in groups of 32 with a float32 scale each, as the
// checkpoints of this project (llama2_numpy's layout), 4 values to a u32.

const GROUP = 32;

// ---- the shaders. Every matrix times vector: one workgroup of 64 per row, each thread a word (4 weights) at a time
// with the stride of the workgroup, so that neighbours read neighbouring words; the partial sums add up in the
// workgroup's memory. rows past 65535 go to a second dimension of the dispatch.
const WIDEN = /* wgsl */ `
struct Shape { rows: u32, words: u32, perRow: u32, first: u32 }
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let row = id.x + id.y * count.x;
  if (row >= shape.rows) { return; }
  var sum = 0.0;
  for (var i = t; i < shape.words; i += 64u) {
    let word = bitcast<i32>(w[row * shape.words + i]);
    let at = i * 4u;
    let dot = f32(extractBits(word, 0u, 8u)) * x[at] + f32(extractBits(word, 8u, 8u)) * x[at + 1u]
            + f32(extractBits(word, 16u, 8u)) * x[at + 2u] + f32(extractBits(word, 24u, 8u)) * x[at + 3u];
    sum += dot * scales[row * shape.perRow + i / 8u];
  }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  if (t == 0u) { y[shape.first + row] = partial[0]; }
}`;
// the same with the activations quantized to int8 as the CPU's matmul_q8 takes them (a float32 scale per group of
// 32), and WGSL's packed dot product: where the language feature is there
const PACKED = /* wgsl */ `
requires packed_4x8_integer_dot_product;
struct Shape { rows: u32, words: u32, perRow: u32, first: u32 }
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<storage, read> xs: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let row = id.x + id.y * count.x;
  if (row >= shape.rows) { return; }
  var sum = 0.0;
  for (var i = t; i < shape.words; i += 64u) {
    sum += f32(dot4I8Packed(w[row * shape.words + i], xq[i])) * scales[row * shape.perRow + i / 8u] * xs[i / 8u];
  }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  if (t == 0u) { y[shape.first + row] = partial[0]; }
}`;
// the small steps of a layer (norms, RoPE, the attention of a short context, SwiGLU, the residual adds): what they
// cost is mostly that they are dispatches of their own, so one that adds a vector of dim stands for each
const SMALL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = arrayLength(&x);
  for (var i = id.x; i < n; i += 64u) { y[i] = y[i] + x[i]; }
}`;

// the shapes of the models in the list that the measurement stands for (legacy header: dim, hidden, layers, heads,
// kv heads, vocab), and the int8 matrices of a layer, [rows, n]
const MODELS = {
  "llm-jp-3 150M": { dim: 512, hidden: 2048, layers: 12, heads: 8, kvHeads: 8, vocab: 99584 },
  "Llama 3.2 1B": { dim: 2048, hidden: 8192, layers: 16, heads: 32, kvHeads: 8, vocab: 128256 },
  "Llama 3.2 3B": { dim: 3072, hidden: 8192, layers: 28, heads: 24, kvHeads: 8, vocab: 128256 },
};
const layerMatrices = ({ dim, hidden, heads, kvHeads }) => {
  const kvDim = (dim / heads) * kvHeads;
  return [[dim, dim], [kvDim, dim], [kvDim, dim], [dim, dim], [hidden, dim], [hidden, dim], [dim, hidden]];
};
const SMALL_PER_LAYER = 7;
const matrixBytes = ([rows, n]) => rows * n + (rows * n / GROUP) * 4;

let device, adapter, packed = false;
async function gpu() {
  if (device) return device;
  if (!self.navigator?.gpu) throw new Error("no navigator.gpu in a worker here");
  adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("navigator.gpu gave no adapter");
  packed = navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product") ?? false;
  // as much of a buffer and of a binding as the adapter allows: the weights are the point
  device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.lost.then((info) => postMessage({ lost: `${info.reason}: ${info.message}` }));
  return device;
}

async function info() {
  const found = { worker: Boolean(self.navigator?.gpu) };
  if (!found.worker) return found;
  await gpu();
  const about = adapter.info ?? {};
  Object.assign(found, {
    adapter: [about.vendor, about.architecture, about.device, about.description].filter(Boolean).join(" · ") || "(not told)",
    fallback: adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    features: [...adapter.features].sort(),
    wgsl: [...(navigator.gpu.wgslLanguageFeatures ?? [])].sort(),
    packed,
  });
  return found;
}

// ---- buffers
// GPUBufferUsage's values (the name itself is missing where there is no WebGPU)
const STORAGE = 0x80, COPY_DST = 0x8, COPY_SRC = 0x4, MAP_READ = 0x1, UNIFORM = 0x40;
function buffer(bytes, usage = STORAGE | COPY_DST) {
  return device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
}
// random bytes, written a few megabytes at a time (writeBuffer copies them, so one block serves them all)
const noise = new Uint8Array(4 << 20).map(() => (Math.random() * 256) | 0);
function fill(target, bytes) {
  for (let at = 0; at < bytes; at += noise.length) device.queue.writeBuffer(target, at, noise, 0, Math.min(noise.length, bytes - at) & ~3);
}
function floats(count, scale = 0.01) {
  const values = new Float32Array(count);
  for (let i = 0; i < count; i++) values[i] = (Math.random() - 0.5) * scale;
  return values;
}

let pipelines;
function pipelinesFor() {
  if (pipelines) return pipelines;
  const make = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  pipelines = { widen: make(WIDEN), packed: packed ? make(PACKED) : null, small: make(SMALL) };
  return pipelines;
}

// A matrix of [rows, n] on the GPU, cut into chunks of rows that each fit a binding. Returns the dispatches that
// multiply it by x into y: [pipeline, bind group, workgroups x, workgroups y] each.
function matrix([rows, n], io, kind = "widen", data) {
  const { widen, packed: packedPipeline } = pipelinesFor();
  const pipeline = kind === "packed" ? packedPipeline : widen;
  const words = n / 4, perRow = n / GROUP, rowBytes = n;
  const most = Math.max(1, Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) / rowBytes));
  const dispatches = [], owned = [];
  for (let first = 0; first < rows; first += most) {
    const count = Math.min(most, rows - first);
    const w = buffer(count * rowBytes), s = buffer(count * perRow * 4), shape = buffer(16, UNIFORM | COPY_DST);
    owned.push(w, s, shape);
    if (data) {
      device.queue.writeBuffer(w, 0, data.w, first * rowBytes, count * rowBytes);
      device.queue.writeBuffer(s, 0, data.s, first * perRow, count * perRow);
    } else {
      fill(w, count * rowBytes);
      device.queue.writeBuffer(s, 0, floats(count * perRow, 0.002));
    }
    device.queue.writeBuffer(shape, 0, new Uint32Array([count, words, perRow, first]));
    const entries = [{ binding: 0, resource: { buffer: w } }, { binding: 1, resource: { buffer: s } },
      { binding: 2, resource: { buffer: kind === "packed" ? io.xq : io.x } }, { binding: 3, resource: { buffer: io.y } },
      { binding: 4, resource: { buffer: shape } }];
    if (kind === "packed") entries.push({ binding: 5, resource: { buffer: io.xs } });
    const across = Math.min(count, device.limits.maxComputeWorkgroupsPerDimension);
    dispatches.push([pipeline, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }), across, Math.ceil(count / across)]);
  }
  return { dispatches, owned, bytes: matrixBytes([rows, n]) };
}
// the vectors every matrix reads and writes: x of the longest row, y of the most rows
function vectors(longest, most) {
  const io = { x: buffer(longest * 4), xq: buffer(longest), xs: buffer((longest / GROUP) * 4), y: buffer(most * 4 + 16, STORAGE | COPY_DST | COPY_SRC) };
  device.queue.writeBuffer(io.x, 0, floats(longest, 2));
  fill(io.xq, longest);
  device.queue.writeBuffer(io.xs, 0, floats(longest / GROUP, 0.1));
  return io;
}
function run(pass, [pipeline, group, x, y]) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(x, y);
}

// ---- check: both shaders against JavaScript, on 300 rows of 512 (rows that are not a power of two)
async function check() {
  await gpu();
  const rows = 300, n = 512, words = n / 4;
  const w = new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s = floats(rows * n / GROUP, 0.01);
  const io = vectors(n, rows), x = floats(n, 2);
  device.queue.writeBuffer(io.x, 0, x);
  // x quantized as the CPU's quantize_x does it (largest |value| / 127, round half to even)
  const xq = new Int8Array(n), xs = new Float32Array(n / GROUP);
  for (let g = 0; g < n / GROUP; g++) {
    let largest = 0;
    for (let i = 0; i < GROUP; i++) largest = Math.max(largest, Math.abs(x[g * GROUP + i]));
    xs[g] = Math.fround(largest / 127);
    for (let i = 0; i < GROUP; i++) {
      const v = x[g * GROUP + i] / xs[g], r = Math.round(v);
      xq[g * GROUP + i] = Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 ? r - 1 : r;
    }
  }
  device.queue.writeBuffer(io.xq, 0, new Uint8Array(xq.buffer));
  device.queue.writeBuffer(io.xs, 0, xs);
  const signed = new Int8Array(w.buffer);
  const verdicts = {};
  for (const kind of packed ? ["widen", "packed"] : ["widen"]) {
    const m = matrix([rows, n], io, kind, { w, s });
    const readback = buffer(rows * 4, MAP_READ | COPY_DST);
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    m.dispatches.forEach((d) => run(pass, d));
    pass.end();
    encoder.copyBufferToBuffer(io.y, 0, readback, 0, rows * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(MAP_READ);
    const got = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    let worst = 0;
    for (let r = 0; r < rows; r++) {
      let want = 0;
      for (let i = 0; i < words; i++) {
        let dot = 0;
        for (let k = 0; k < 4; k++) dot += signed[r * n + i * 4 + k] * (kind === "packed" ? xq[i * 4 + k] : x[i * 4 + k]);
        want += dot * s[r * (n / GROUP) + (i >> 3)] * (kind === "packed" ? xs[i >> 3] : 1);
      }
      worst = Math.max(worst, Math.abs(got[r] - want) / (Math.abs(want) + 1e-3));
    }
    // float32 sums in another order: a relative 1e-4 is the rounding, a wrong index is off by a multiple
    verdicts[kind] = { worstRelative: worst, ok: worst < 1e-3 };
    m.owned.forEach((b) => b.destroy());
  }
  return verdicts;
}

// ---- bandwidth: one matrix, 20 times in one submission after a warm-up; and the CPU's kernel on the same bytes
async function bandwidth(shape) {
  await gpu();
  const found = {};
  const io = vectors(shape[1], shape[0]);
  for (const kind of packed ? ["widen", "packed"] : ["widen"]) {
    const m = matrix(shape, io, kind);
    const time = async (times) => {
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let i = 0; i < times; i++) m.dispatches.forEach((d) => run(pass, d));
      pass.end();
      const began = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    await time(2);
    const ms = await time(20);
    found[kind] = { GBps: (20 * m.bytes) / (ms / 1000) / 1e9, msEach: ms / 20 };
    m.owned.forEach((b) => b.destroy());
  }
  found.cpu = await cpuBandwidth(shape);
  return found;
}

// the CPU on the kernel the model page uses for int8 on one thread (matmul_q8, without relaxed SIMD: every
// browser has it), on four matrices of this shape in turn so that no cache holds them. The page's software threads
// reach about twice this at most (T93: the memory's bandwidth).
let cpuKernel;
async function cpuBandwidth([rows, n]) {
  const weights = rows * n, scales = (rows * n / GROUP) * 4, copies = Math.max(1, Math.min(4, Math.floor(128e6 / weights)));
  const bytes = 4096 + n * 8 + rows * 4 + copies * (weights + scales);
  const memory = new WebAssembly.Memory({ initial: Math.ceil(bytes / 65536) + 1 });
  // the kernels of the same deployment as this file (?v=, GitHub Pages keeps a file for ten minutes)
  cpuKernel ??= await WebAssembly.compile(await (await fetch(new URL(`../simdkernel_plain.wasm${new URL(import.meta.url).search}`, import.meta.url))).arrayBuffer());
  const k = (await WebAssembly.instantiate(cpuKernel, { env: { memory } })).exports;
  const U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const x = 4096, xq = x + n * 4, xs = xq + n, out = xs + (n / GROUP) * 4 + 64, first = Math.ceil((out + rows * 4) / 64) * 64;
  F.set(floats(n, 2), x / 4);
  for (let c = 0; c < copies; c++) {
    const at = first + c * (weights + scales);
    for (let i = 0; i < weights; i += noise.length) U.set(noise.subarray(0, Math.min(noise.length, weights - i)), at + i);
    F.set(floats(rows * n / GROUP, 0.002), (at + weights) / 4);
  }
  k.quantize_x(xq, xs, x, n, 0);
  const pass = (c) => k.matmul_q8(out, xq, xs, first + c * (weights + scales), first + c * (weights + scales) + weights, n, 0, rows);
  for (let c = 0; c < copies; c++) pass(c);
  const rounds = Math.max(1, Math.round(200e6 / weights));
  const began = performance.now();
  for (let r = 0; r < rounds; r++) pass(r % copies);
  const ms = performance.now() - began;
  return { GBps: (rounds * (weights + scales)) / (ms / 1000) / 1e9, msEach: ms / rounds, threads: 1 };
}

// ---- a token's work: every layer's seven matrices and seven small dispatches, the classifier, the logits back
async function token(name, kind = "widen") {
  await gpu();
  const model = MODELS[name];
  const shapes = [...Array(model.layers)].flatMap(() => layerMatrices(model));
  const classifier = [model.vocab, model.dim];
  const longest = Math.max(model.dim, model.hidden), most = Math.max(model.hidden, model.vocab);
  const io = vectors(longest, most);
  const made = [];
  let bytes = 0;
  // a buffer the device cannot give fails later and quietly, as an error of these scopes
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  try {
    for (const shape of [...shapes, classifier]) {
      const m = matrix(shape, io, kind);
      made.push(m);
      bytes += m.bytes;
    }
  } catch (error) {
    await device.popErrorScope();
    await device.popErrorScope();
    made.forEach((m) => m.owned.forEach((b) => b.destroy()));
    return { model: name, error: `could not hold the weights (${(bytes / 1e9).toFixed(2)} GB made): ${error.message}` };
  }
  const { small } = pipelinesFor();
  const a = buffer(model.dim * 4), b = buffer(model.dim * 4);
  const smallGroup = device.createBindGroup({ layout: small.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } }] });
  const logits = buffer(model.vocab * 4, MAP_READ | COPY_DST);
  // the uploads may still be going: wait for them, and for a device that could not take them
  await device.queue.onSubmittedWorkDone();
  const invalid = await device.popErrorScope(), outOfMemory = await device.popErrorScope();
  const refused = invalid ?? outOfMemory;
  if (refused) {
    made.forEach((m) => m.owned.forEach((x) => x.destroy()));
    return { model: name, error: `the GPU did not take ${(bytes / 1e9).toFixed(2)} GB of weights: ${refused.message}` };
  }
  const once = async () => {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    made.forEach((m, i) => {
      m.dispatches.forEach((d) => run(pass, d));
      // after every layer's last matrix, its small steps (and none after the classifier)
      if (i < shapes.length && i % 7 === 6) {
        for (let j = 0; j < SMALL_PER_LAYER; j++) { pass.setPipeline(small); pass.setBindGroup(0, smallGroup); pass.dispatchWorkgroups(1); }
      }
    });
    pass.end();
    encoder.copyBufferToBuffer(io.y, 0, logits, 0, model.vocab * 4);
    device.queue.submit([encoder.finish()]);
    await logits.mapAsync(MAP_READ);
    const first = new Float32Array(logits.getMappedRange(0, 16))[0];
    logits.unmap();
    return first;
  };
  for (let i = 0; i < 3; i++) await once();
  const times = [];
  for (let i = 0; i < 10; i++) {
    const began = performance.now();
    await once();
    times.push(performance.now() - began);
  }
  times.sort((p, q) => p - q);
  const ms = times[5];
  made.forEach((m) => m.owned.forEach((x) => x.destroy()));
  [a, b, logits, io.x, io.xq, io.xs, io.y].forEach((x) => x.destroy());
  return { model: name, kind, GB: bytes / 1e9, msPerToken: ms, tokPerSecond: 1000 / ms, GBps: bytes / (ms / 1000) / 1e9,
    dispatches: made.reduce((n, m) => n + m.dispatches.length, 0) + model.layers * SMALL_PER_LAYER };
}

// ---- the bridge: a worker that waits (Atomics.wait, as the model's worker would while Python calls forward())
// and this one, which must not block (it awaits the GPU): Atomics.waitAsync where there is one
async function bridge(memory, rounds) {
  const words = new Int32Array(memory);
  const hasWaitAsync = typeof Atomics.waitAsync === "function";
  if (!hasWaitAsync) return { waitAsync: false };
  // words[0]: the request counter (the waiter adds 1), words[1]: the answer counter (this one sets it)
  let answered = 0;
  while (answered < rounds) {
    const seen = Atomics.load(words, 0);
    if (seen === answered) {
      const result = Atomics.waitAsync(words, 0, seen);
      if (result.async) await result.value;
      continue;
    }
    answered = seen;
    Atomics.store(words, 1, answered);
    Atomics.notify(words, 1);
  }
  return { waitAsync: true };
}

onmessage = async ({ data }) => {
  try {
    let result;
    if (data.step === "info") result = await info();
    else if (data.step === "check") result = await check();
    else if (data.step === "bandwidth") result = await bandwidth(data.shape);
    else if (data.step === "token") result = await token(data.model, data.kind);
    else if (data.step === "bridge") result = await bridge(data.memory, data.rounds);
    postMessage({ step: data.step, result });
  } catch (error) {
    postMessage({ step: data.step, error: String(error?.message ?? error) });
  }
};
