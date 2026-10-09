// What every step of /benchmark/'s GPU worker stands on: the shaders' module, the device and its adapter, the models' shapes,
// buffers and random weights, the int8 shaders of a matrix and how one is bound and run, a read back, and how a time is
// taken (pairs of n and 2n, forms in turn, an error scope around a measured path).
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)

// the WGSL, shared with the model's GPU worker (public/shaders.js, T135), from the same deployment as this file. Not
// awaited here: a module worker's port opens at its first await, and a message that comes before onmessage is set is
// lost (T109); every step awaits it instead
const shaders = import(new URL(`../../shaders.js${new URL(import.meta.url).search}`, import.meta.url));
// T353: what the steps share and one of them sets (a module's `let` cannot be assigned from another module, and the
// steps are the modules of gpu/): the fields were this file's `let`s of the same names
const shared = {
  // shaders.js's module, and its GROUP and TILE (load())
  WGSL: undefined, GROUP: undefined, TILE: undefined,
  // the device, its adapter, and whether WGSL has packed_4x8_integer_dot_product here (gpu())
  device: undefined, adapter: undefined, packed: false,
  // a fallback adapter (SwiftShader: the CPU pretending to be a GPU, as in CI) says nothing of a GPU's speed, and takes
  // 16 s for one token of Llama 3.2 1B: every measurement is taken once there, and only its answers are worth anything
  fallback: false,
  // the layer a token runs on the GPU (generate(), T151): T175's fused DP4A where the check found it right (checkLayer
  // ran in this worker and its verdict is ok: T175's review), else T150's fused one (generate() builds the fused forms
  // only; which is fastest on the device is the layer table's, and the engine's choice is T152's)
  layerVerdicts: undefined,
  // T208: the layer table's rows (layer()), for the steps' table to break down the layer a token would run here
  layerTimes: undefined,
};
// (every step awaits this first; so do the tests that import the pure helpers below)
async function load() {
  shared.WGSL ??= await shaders;
  ({ GROUP: shared.GROUP, TILE: shared.TILE } = shared.WGSL);
}

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
// the small steps a layer dispatches on its own, as the CPU's forward pass has them (a stand-in each, SMALL; T150's
// layer step runs the real ones)
const SMALL_PER_LAYER = 7;
// the CPU section's made-up model (public/benchmark/sections.js): Llama 3.2 1B's width, two layers
const PROMPT_MODEL = { dim: 2048, hidden: 8192, layers: 2, heads: 32, kvHeads: 8 };
const matrixBytes = ([rows, n]) => rows * n + (rows * n / shared.GROUP) * 4;

async function gpu() {
  if (shared.device) return shared.device;
  if (!self.navigator?.gpu) throw new Error("no navigator.gpu in a worker here");
  shared.adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!shared.adapter) throw new Error("navigator.gpu gave no adapter");
  shared.fallback = Boolean(shared.adapter.info?.isFallbackAdapter ?? shared.adapter.isFallbackAdapter);
  shared.packed = navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product") ?? false;
  // as much of a buffer and of a binding as the adapter allows: the weights are the point. T146's tiled shaders use
  // shader-f16 and subgroups where the adapter has them (asked for only then: a device refuses a feature it lacks)
  shared.device = await shared.adapter.requestDevice({
    requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"].filter((name) => shared.adapter.features.has(name)),
    requiredLimits: {
      maxStorageBufferBindingSize: shared.adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: shared.adapter.limits.maxBufferSize,
    },
  });
  shared.device.lost.then((info) => postMessage({ lost: `${info.reason}: ${info.message}` }));
  return shared.device;
}

async function info() {
  const found = { worker: Boolean(self.navigator?.gpu) };
  if (!found.worker) return found;
  await gpu();
  const about = shared.adapter.info ?? {};
  Object.assign(found, {
    adapter: [about.vendor, about.architecture, about.device, about.description].filter(Boolean).join(" · ") || "(not told)",
    fallback: shared.adapter.info?.isFallbackAdapter ?? shared.adapter.isFallbackAdapter ?? null,
    maxStorageBufferBindingSize: shared.adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: shared.adapter.limits.maxBufferSize,
    maxComputeWorkgroupsPerDimension: shared.adapter.limits.maxComputeWorkgroupsPerDimension,
    // what a tiled shader (T146) may take: the device's, though the tiles keep to the defaults (16 KiB, 256)
    maxComputeWorkgroupStorageSize: shared.adapter.limits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: shared.adapter.limits.maxComputeInvocationsPerWorkgroup,
    // the DP4A shader's subgroup path runs only where a subgroup is 16 wide (SwiftShader's is not: CI never runs it)
    subgroupSizes: shared.adapter.info?.subgroupMinSize ? [shared.adapter.info.subgroupMinSize, shared.adapter.info.subgroupMaxSize] : null,
    features: [...shared.adapter.features].sort(),
    wgsl: [...(navigator.gpu.wgslLanguageFeatures ?? [])].sort(),
    packed: shared.packed,
  });
  return found;
}

// ---- buffers
// GPUBufferUsage's values (the name itself is missing where there is no WebGPU)
const STORAGE = 0x80, COPY_DST = 0x8, COPY_SRC = 0x4, MAP_READ = 0x1, UNIFORM = 0x40;
function buffer(bytes, usage = STORAGE | COPY_DST) {
  return shared.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
}
// random bytes, written a few megabytes at a time (writeBuffer copies them, so one block serves them all)
const noise = new Uint8Array(4 << 20).map(() => (Math.random() * 256) | 0);
function fill(target, bytes) {
  for (let at = 0; at < bytes; at += noise.length) shared.device.queue.writeBuffer(target, at, noise, 0, Math.min(noise.length, bytes - at) & ~3);
}
function floats(count, scale = 0.01) {
  const values = new Float32Array(count);
  for (let i = 0; i < count; i++) values[i] = (Math.random() - 0.5) * scale;
  return values;
}

let pipelines;
function pipelinesFor() {
  if (pipelines) return pipelines;
  const make = (code) => shared.device.createComputePipeline({ layout: "auto", compute: { module: shared.device.createShaderModule({ code }), entryPoint: "main" } });
  pipelines = { widen: make(shared.WGSL.WIDEN), packed: shared.packed ? make(shared.WGSL.PACKED) : null, small: make(shared.WGSL.SMALL),
                batched: make(shared.WGSL.BATCHED), argmax: make(shared.WGSL.ARGMAX), empty: make(shared.WGSL.EMPTY) };
  return pipelines;
}

// T146: the shaders of a prompt: T135's batched one, then the tiled ones of shaders.js: llama.cpp's register tiles
// (f16 in the workgroup's memory where shader-f16 is, else f32) in both shapes of REG_TILES, and ONNX Runtime's DP4A
// (where the packed int8 dot is), with its subgroup path where subgroups are. A shape past the device's workgroup
// memory or threads is not made (none says why). A tiled shader's pipeline is made when first asked for, asynchronously
// and in an error scope: one this device refuses rejects there, and only its own rows say so
const tiledPipelines = new Map();
function promptShaders() {
  const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = shared.device.limits;
  // T147: the tiled ones are the model's GPU worker's candidates (shaders.js's promptForms), the same list
  return [{ name: "batched (T135)", kind: "batched" }, ...shared.WGSL.promptForms({ half: shared.device.features.has("shader-f16"),
    subgroups: shared.device.features.has("subgroups"), packed: shared.packed, memory, threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) })];
}
// T149: the shaders of a matrix times one vector (a generated token's): T134's two, then llama.cpp's mul_mat_vec (its
// float form and its MMVQ one, each with the workgroup's reduction and, where subgroups are, with subgroupAdd) and
// ONNX Runtime's MatMulNBits and DP4A for small M (shaders.js). rows: the rows a workgroup takes
function matVecShaders() {
  const noPacked = shared.packed ? undefined : "no packed int8 dot here";
  const subgroups = shared.device.features.has("subgroups");
  const noSubgroupId = navigator.gpu.wgslLanguageFeatures?.has("subgroup_id") ? undefined : "no subgroup_id in this WGSL";
  // check: the name of the shader's verdict in check() (T134's two are checked on their own there)
  const shaders = [{ name: "widened (T134)", kind: "widen", check: "widen" },
    { name: "packed int8 (T134)", kind: "packed", check: "packed", none: noPacked }];
  for (const [form, isPacked] of [["mul_mat_vec", false], ["MMVQ", true]]) {
    for (const withSubgroups of subgroups ? [false, true] : [false]) {
      shaders.push({ name: `llama.cpp ${form}, ${shared.WGSL.MUL_MAT_VEC_ROWS} rows${withSubgroups ? ", subgroups" : ""}`,
        code: shared.WGSL.mulMatVec({ packed: isPacked, subgroups: withSubgroups }), rows: shared.WGSL.MUL_MAT_VEC_ROWS, packed: isPacked,
        none: (isPacked && noPacked) || (withSubgroups && noSubgroupId) || undefined });
    }
  }
  shaders.push({ name: `ORT MatMulNBits, ${shared.WGSL.ORT_MATVEC_ROWS} rows`, code: shared.WGSL.ortMatVec, rows: shared.WGSL.ORT_MATVEC_ROWS, packed: false });
  shaders.push({ name: `ORT DP4A small M, ${shared.WGSL.ORT_DP4A_MATVEC_ROWS} rows`, code: shared.WGSL.ortDp4aMatVec,
    rows: shared.WGSL.ORT_DP4A_MATVEC_ROWS, packed: true, none: noPacked });
  return shaders;
}
// what matrix() takes for a shader: "widen", "packed", "batched", or { pipeline, tile or rows, packed } of one with code
// of its own (a tiled one, T146, or a matrix × vector's of T149)
async function kindOf(shader) {
  if (!shader.code) return shader.kind;
  if (!tiledPipelines.has(shader.name)) {
    tiledPipelines.set(shader.name, validated(() => shared.device.createComputePipelineAsync({ layout: "auto",
      compute: { module: shared.device.createShaderModule({ code: shader.code }), entryPoint: "main", constants: shader.constants } })));
  }
  return { pipeline: await tiledPipelines.get(shader.name), tile: shader.tile, rows: shader.rows, packed: shader.packed };
}
// the activations of the packed tiled shaders, quantized on the GPU (shaders.js's QUANTIZE): the first n values of
// each of io's tokens, from io.x into io.xq and io.xs. One thread a group of 32, the tokens along y
let quantizePipeline;
function quantizer(io, n) {
  quantizePipeline ??= shared.device.createComputePipeline({ layout: "auto", compute: { module: shared.device.createShaderModule({ code: shared.WGSL.QUANTIZE }), entryPoint: "main" } });
  const shape = buffer(16, UNIFORM | COPY_DST);
  shared.device.queue.writeBuffer(shape, 0, new Uint32Array([n, io.xStride, 0, 0]));
  const group = shared.device.createBindGroup({ layout: quantizePipeline.getBindGroupLayout(0),
    entries: [io.x, io.xq, io.xs, shape, io.step].map((b, binding) => ({ binding, resource: { buffer: b } })) });
  return { dispatch: [quantizePipeline, group, Math.ceil(n / shared.GROUP / 64), io.tokens, 1], owned: [shape] };
}

// A matrix of [rows, n] on the GPU, cut into chunks of rows that each fit a binding (chunk: rows a chunk, for the
// checks; else as many as fit). Returns the dispatches that multiply it by x into y: [pipeline, bind group, workgroups
// x, workgroups y, workgroups z] each. kind "batched" or a tiled one (kindOf): by io.tokens vectors at once (x and y
// hold them io.xStride and io.yStride floats apart); add: the products added to what y holds (the residual stream of
// the model's layers), for the checks
function matrix([rows, n], io, kind = "widen", data, { add = false, chunk } = {}) {
  const held = placed([rows, n], io, data, { add, chunk });
  return { ...bound(held, io, kind), owned: held.owned, bytes: held.bytes };
}
// the weights of a matrix on the GPU, in chunks of rows that each fit a binding, with each chunk's Shape (T149: placed
// once and bound by every shader of a matrix × vector)
function placed([rows, n], io, data, { add = false, chunk } = {}) {
  const words = n / 4, perRow = n / shared.GROUP, rowBytes = n;
  const most = chunk ?? Math.max(1, Math.floor(Math.min(shared.device.limits.maxStorageBufferBindingSize, shared.device.limits.maxBufferSize) / rowBytes));
  const chunks = [], owned = [];
  for (let first = 0; first < rows; first += most) {
    const count = Math.min(most, rows - first);
    const w = buffer(count * rowBytes), s = buffer(count * perRow * 4), shape = buffer(32, UNIFORM | COPY_DST);
    owned.push(w, s, shape);
    if (data) {
      shared.device.queue.writeBuffer(w, 0, data.w, first * rowBytes, count * rowBytes);
      shared.device.queue.writeBuffer(s, 0, data.s, first * perRow, count * perRow);
    } else {
      fill(w, count * rowBytes);
      shared.device.queue.writeBuffer(s, 0, floats(count * perRow, 0.002));
    }
    // the batched and tiled shaders' shape goes on with the strides of the tokens and "add", the others read four
    shared.device.queue.writeBuffer(shape, 0, new Uint32Array([count, words, perRow, first, io.xStride ?? 0, io.yStride ?? 0, add ? 1 : 0, 0]));
    chunks.push({ w, s, shape, count });
  }
  return { chunks, owned, bytes: matrixBytes([rows, n]) };
}
// the dispatches that multiply placed weights by x into y for a shader: [pipeline, bind group, workgroups x, y, z] each
function bound({ chunks }, io, kind = "widen") {
  const own = typeof kind === "object", tiled = own && Boolean(kind.tile), matVec = own && Boolean(kind.rows);
  const { widen, packed: packedPipeline, batched } = pipelinesFor();
  const pipeline = own ? kind.pipeline : { widen, packed: packedPipeline, batched }[kind];
  const quantized = own ? kind.packed : kind === "packed";
  const dispatches = [];
  for (const { w, s, shape, count } of chunks) {
    const entries = [{ binding: 0, resource: { buffer: w } }, { binding: 1, resource: { buffer: s } },
      { binding: 2, resource: { buffer: quantized ? io.xq : io.x } }, { binding: 3, resource: { buffer: io.y } },
      { binding: 4, resource: { buffer: shape } }];
    if (kind === "packed" || (matVec && quantized)) entries.push({ binding: 5, resource: { buffer: io.xs } });
    if (kind === "batched" || tiled) entries.push({ binding: 5, resource: { buffer: io.step } });
    if (tiled && quantized) entries.push({ binding: 6, resource: { buffer: io.xs } });
    const group = shared.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    if (tiled) {
      // the tiles numbered over x, then y (as both sources number them: the rows' tiles first, then the tokens')
      const tiles = Math.ceil(count / kind.tile.rows) * Math.ceil(io.tokens / kind.tile.tokens);
      const across = Math.min(tiles, shared.device.limits.maxComputeWorkgroupsPerDimension);
      dispatches.push([pipeline, group, across, Math.ceil(tiles / across)]);
    } else if (matVec) {
      // T149: kind.rows rows a workgroup, numbered over x and then y
      const groups = Math.ceil(count / kind.rows), across = Math.min(groups, shared.device.limits.maxComputeWorkgroupsPerDimension);
      dispatches.push([pipeline, group, across, Math.ceil(groups / across)]);
    } else {
      const across = Math.min(count, shared.device.limits.maxComputeWorkgroupsPerDimension);
      dispatches.push([pipeline, group, across, Math.ceil(count / across), kind === "batched" ? Math.ceil(io.tokens / shared.TILE) : 1]);
    }
  }
  return { dispatches };
}
// the vectors every matrix reads and writes: x of the longest row, y of the most rows, tokens of each (the batched and
// tiled shaders take several); xq and xs: x quantized, 8 bits a value and a float32 scale a group of 32, of every token
function vectors(longest, most, tokens = 1) {
  const io = { x: buffer(tokens * longest * 4), xq: buffer(tokens * longest, STORAGE | COPY_DST | COPY_SRC),
               xs: buffer(tokens * (longest / shared.GROUP) * 4, STORAGE | COPY_DST | COPY_SRC),
               y: buffer(tokens * most * 4 + 16, STORAGE | COPY_DST | COPY_SRC), tokens, xStride: longest, yStride: most,
               step: buffer(16, UNIFORM | COPY_DST) };
  shared.device.queue.writeBuffer(io.step, 0, new Uint32Array([tokens, 0, 0, 0]));  // the batched and tiled shaders' tokens
  shared.device.queue.writeBuffer(io.x, 0, floats(tokens * longest, 2));
  fill(io.xq, tokens * longest);
  shared.device.queue.writeBuffer(io.xs, 0, floats(tokens * (longest / shared.GROUP), 0.1));
  return io;
}
const destroyVectors = (io) => [io.x, io.xq, io.xs, io.y, io.step].forEach((b) => b.destroy());
function run(pass, [pipeline, group, x, y, z = 1]) {
  pass.setPipeline(pipeline);
  if (group) pass.setBindGroup(0, group);  // none for the empty dispatch
  pass.dispatchWorkgroups(x, y, z);
}
// the most likely token of the logits in y (the classifier's output), into a buffer of its own
function argmaxOf(io, vocab) {
  const { argmax } = pipelinesFor();
  const chosen = buffer(16, STORAGE | COPY_SRC), count = buffer(16, UNIFORM | COPY_DST);
  shared.device.queue.writeBuffer(count, 0, new Uint32Array([vocab, 0, 0, 0]));
  const group = shared.device.createBindGroup({ layout: argmax.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: io.y } },
    { binding: 1, resource: { buffer: chosen } }, { binding: 2, resource: { buffer: count } }] });
  return { dispatch: [argmax, group, 1, 1], chosen, owned: [chosen, count] };
}
// read size bytes of source back: the copy, the submission and the mapping, as a token's logits come back. target: a
// buffer to read into again and again (a measurement's loop), else one made and destroyed here
async function readBack(encoder, source, size, target) {
  const bytes = Math.ceil(size / 4) * 4, into = target ?? buffer(bytes, MAP_READ | COPY_DST);
  encoder.copyBufferToBuffer(source, 0, into, 0, bytes);
  shared.device.queue.submit([encoder.finish()]);
  await into.mapAsync(MAP_READ);
  const got = into.getMappedRange(0, bytes).slice(0);
  into.unmap();
  if (!target) into.destroy();
  return got;
}
// the median of runs of a measurement, in ms, after warm-up runs (one run and no warm-up on a fallback adapter)
async function median(measure, times = 10, warm = 3) {
  if (shared.fallback) [times, warm] = [1, 0];
  for (let i = 0; i < warm; i++) await measure();
  const ms = [];
  for (let i = 0; i < times; i++) {
    const began = performance.now();
    await measure();
    ms.push(performance.now() - began);
  }
  return ms.sort((a, b) => a - b)[ms.length >> 1];
}
// what fn does on the GPU, a validation error of it thrown (a pipeline or a bind group the device refused)
async function validated(fn) {
  shared.device.pushErrorScope("validation");
  try {
    return await fn();
  } finally {
    const invalid = await shared.device.popErrorScope();
    if (invalid) throw new Error(invalid.message);
  }
}

// ---- the ceilings (T168): each loop of shaders.js alone. First its loop count doubles until one dispatch takes
// DISPATCH_MS (a dispatch's own cost is then small beside its work), then the dispatches of a submission until it
// takes SUBMISSION_MS. The time of n dispatches is that of a submission of 2n less one of n: what a submission costs
// besides its work (3.7 ms waited for on the owner's Android, T134) is in both and drops out. The two are measured
// in turn, PAIRS times, and the median of the pairs' differences taken: measured one after the other, a device's
// load moved the numbers by several times (T168's review, lavapipe under load: 4 of 55 readings 30% or more off,
// one ten times). A pair's 2n should take about twice its n: past STEADY the pairs are taken again once, and if
// they are still past it the ceiling says "unsteady" and the prompt is not held against it. Everything runs in
// error scopes, validation and out of memory: a pipeline, bind group or buffer the device refused makes an error,
// not a number (a bind group that does not match took no time, and read as 13,915 GB/s). So does a loop that never
// takes DISPATCH_MS in MOST_LOOPS or SUBMISSION_MS in MOST_DISPATCHES (a loop a compiler removed: lavapipe then read
// 60 million GFLOPS). A fallback adapter runs the same (its speed is no GPU's, but a few seconds
// a ceiling)
const DISPATCH_MS = 2, SUBMISSION_MS = 40, PAIRS = 5, STEADY = [1.6, 2.2];
// MOST_LOOPS: a dispatch of 65536 threads that many loops is 4 TFLOP of multiply-adds, 2 s at 2 TFLOPS
const CEILING_GROUPS = 256, FIRST_LOOPS = 4, MOST_LOOPS = 1 << 14, MOST_DISPATCHES = 4096;
// the storage buffer the global read streams through: 128 MiB, WebGPU's default largest binding (or the device's
// largest, if smaller), meant to be larger than the GPU's caches (their sizes on the owner's devices are not measured)
const GLOBAL_BYTES = 128 << 20;
const middle = (values) => [...values].sort((x, y) => x - y)[values.length >> 1];
// the time of n dispatches (what submission(n) submits and waits for, in ms): n doubles, up to most, until a submission
// takes SUBMISSION_MS (short: it never did), then PAIRS submissions of n and of 2n in turn, and the median of their
// differences; past STEADY the pairs once more, then unsteady, or an error where 2n took no longer than n (lavapipe
// under load once read a negative time, T149). T168's, and T149's matrix × vector. strict: none of the pairs when
// most never took SUBMISSION_MS (a loop that did no work), only { short: true }
async function paired(submission, most, strict = false) {
  let n = 1, ms;
  while ((ms = await submission(n)) < SUBMISSION_MS && n < most) n *= 2;
  if (strict && ms < SUBMISSION_MS) return { short: true };
  for (let tries = 0; ; tries++) {
    const differences = [], ratios = [];
    for (let i = 0; i < PAIRS; i++) {
      const once = await submission(n), twice = await submission(2 * n);
      differences.push(twice - once);
      ratios.push(twice / once);
      postMessage({ alive: true });
    }
    // a median difference of 0 or less (2n no longer than n: the load moved) is no time at all, never steady
    const ratio = middle(ratios), took = middle(differences), steady = took > 0 && ratio >= STEADY[0] && ratio <= STEADY[1];
    if (steady) return { ms: took, dispatches: n, ratio };
    if (tries) {
      if (!(took > 0)) throw new Error(`a submission of ${2 * n} took no longer than one of ${n}: the device's load moved`);
      return { ms: took, dispatches: n, ratio, unsteady: true };
    }
  }
}
// T150's review: paired() for several things at once, in turn (T147's timeForms): n for each as paired() finds it, then
// PAIRS rounds of a submission of n and one of 2n of each, so that a device's warming and load fall on all of them
// alike. Each: { ms, dispatches (n), ratio } as paired() says it, unsteady past STEADY after the rounds are taken
// once more, or { error } where its 2n took no longer than its n. On a fallback adapter one submission of 1 each.
async function interleaved(submissions, most) {
  if (shared.fallback) {
    const results = [];
    for (const submission of submissions) results.push({ ms: await submission(1), dispatches: 1 });
    return results;
  }
  const counts = [];
  for (const submission of submissions) {
    await submission(2);
    let n = 1;
    while ((await submission(n)) < SUBMISSION_MS && n < most) n *= 2;
    counts.push(n);
    postMessage({ alive: true });
  }
  for (let tries = 0; ; tries++) {
    const differences = submissions.map(() => []), ratios = submissions.map(() => []);
    for (let round = 0; round < PAIRS; round++) {
      for (let i = 0; i < submissions.length; i++) {
        const once = await submissions[i](counts[i]), twice = await submissions[i](2 * counts[i]);
        differences[i].push(twice - once);
        ratios[i].push(twice / once);
      }
      postMessage({ alive: true });
    }
    const results = submissions.map((_, i) => {
      const ratio = middle(ratios[i]), took = middle(differences[i]);
      return { ms: took, dispatches: counts[i], ratio, steady: took > 0 && ratio >= STEADY[0] && ratio <= STEADY[1] };
    });
    if (tries || results.every((r) => r.steady)) {
      return results.map(({ steady, ...r }) => (r.ms > 0 ? { ...r, ...(steady ? {} : { unsteady: true }) }
        : { error: `a submission of ${2 * r.dispatches} took no longer than one of ${r.dispatches}: the device's load moved` }));
    }
  }
}
// what fn does on the GPU (given an array for the buffers it makes, destroyed after), with an error of the device
// thrown: validation (a pipeline or bind group refused) or out of memory (a buffer it could not give)
async function scoped(fn) {
  const owned = [];
  shared.device.pushErrorScope("out-of-memory");
  shared.device.pushErrorScope("validation");
  let result, failure;
  try {
    result = await fn(owned);
  } catch (error) {
    failure = error;
  }
  const invalid = await shared.device.popErrorScope(), outOfMemory = await shared.device.popErrorScope();
  owned.forEach((b) => b.destroy());
  if (invalid ?? outOfMemory) throw new Error((invalid ?? outOfMemory).message);
  if (failure) throw failure;
  return result;
}

export { shared, load, MODELS, layerMatrices, SMALL_PER_LAYER, PROMPT_MODEL, matrixBytes, gpu, info, STORAGE,
  COPY_DST, COPY_SRC, UNIFORM, MAP_READ, buffer, noise, fill, floats, pipelinesFor, promptShaders, matVecShaders,
  kindOf, quantizer, matrix, placed, bound, vectors, destroyVectors, run, argmaxOf, readBack, median, validated,
  PAIRS, DISPATCH_MS, SUBMISSION_MS, CEILING_GROUPS, FIRST_LOOPS, MOST_LOOPS, MOST_DISPATCHES, GLOBAL_BYTES, middle,
  paired, interleaved, scoped };
