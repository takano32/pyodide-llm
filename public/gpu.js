// gpu.js (T135, T147, T148): the model page's GPU, in a worker of its own. forward.js (in the model's worker) makes it
// wherever the worker has WebGPU and the model is one it takes (T148: by default, no option), and hands it the blocks
// of a prompt (up to plan.batch tokens) one at a time that it finds faster here than on the CPU, waiting in
// Atomics.wait for the answer: the model's worker cannot wait for a promise while Python calls forwardMany(), and this
// one can (T94's design; T134's bridge measured 12 to 95 µs a round trip).
//
//   { type: "start", memory, plan }  the layers of the model onto the GPU, read from the shared memory of forward.js
//                                    (plan: what forward.js says of them, addresses in that memory). Says
//                                    { type: "progress" } after every step (a layer's weights, a shader compiled,
//                                    checked, timed: STEP_MS each at most, T147), then { type: "ready",
//                                    adapter, key, bytes, seconds, form, attention, forms, remembered, blocks } or
//                                    { type: "unusable", reason }. T148: a fallback adapter (the CPU in the GPU's
//                                    place) is refused before anything is compiled, unless plan.force.fallback
//                                    (tests); plan.remembered ({ key, matrices, attention }: what the page kept of an
//                                    earlier visit) spares the timing of the matrices' shaders where key is this
//                                    adapter's and the shader is still right here; blocks: what a whole block of 16
//                                    and of 64 tokens takes here (forward.js weighs the GPU against the CPU by it)
//   { type: "prompt", serial, count, pos }
//                                    the rows of count tokens at positions pos, pos + 1, ... (embedded by forward.js,
//                                    at plan.rows) through every layer, and their keys and values of every layer into
//                                    plan.staging as float16. The answer is in the control area: words.failed, then
//                                    words.done = serial, while words.beat counts up meanwhile; a failure also says
//                                    { type: "failed", reason }. Nothing is written where words.wanted is no longer
//                                    serial: forward.js gave the request up (T147: a worker that went on late must not
//                                    write into a memory that may hold the next model by then)
//   { type: "tokens", serial, count, pos, from, token, history, length, cache, settings, randoms }
//                                    T152: count steps of a generation from token at pos (the forward pass and the
//                                    sampling of each, SAMPLE's), after the keys and values of positions from to pos - 1
//                                    went up from forward.js's cache; the ids into plan.tokens.ids and the keys and
//                                    values of their positions into plan.staging (see generate() below). The answer is
//                                    in the control area as a prompt's. Where plan.tokens asks for it, "ready" says
//                                    tokens ({ form, ms, forms, remembered, attention, attentions }: the layer of a
//                                    token chosen here and its ms a step, T224: the attention of a token chosen here
//                                    and what each came to) or tokensWhy (why they stay on the CPU)
//   { type: "stop" }                 every buffer and the device let go, and the worker ends; T205: it says
//                                    { type: "ended" } as it does (so does a start that ends as "unusable")
//
// A block goes as one submission: per layer the RMSNorm, the matrices of q, k and v (each weight read once for the
// block's tokens, by the tiled shader chosen on this device: see chooseMatrices), T153: their biases (Qwen2) and the
// norms of every head of q and k (Qwen3) where the model has them, RoPE with the keys and values into the
// GPU's own cache (float16), the attention (llama.cpp's flash attention with tiles: every token sees the positions up to
// its own), the output matrix added to the residual, the RMSNorm, the gate and the up matrices, SwiGLU, the down matrix
// added. T154: GPT-2 and GPT-NeoX (plan.layerNorm) have LayerNorm with a bias for either norm, a bias after every
// matrix, no gate (the up matrix is w1, then GELU), GPT-2 no RoPE at all (plan.turned 0: its learned positions are in
// the rows forward.js embeds) and GPT-NeoX RoPE on a part of every head; GPT-NeoX's parallel residual (plan.parallel)
// has the FFN's norm read the layer's input before the output matrix adds to it, into a buffer of its own (xn). The
// last layer stops at its keys and values: nothing of a prompt's token after them is used. Then the block's
// keys and values of every layer are copied out and read back. The activations are float32 (quantized to 8 bits first
// for the packed shaders, as the CPU's matmul_q8 takes them), the weights int8 widened or multiplied as int8. T155:
// int6 weights (T98) are widened to int8 once, as they go onto the GPU; a model in a 64-bit memory (T101) goes as
// one in a 32-bit memory (its addresses are Numbers); a matrix larger than a buffer the device binds, in pieces of rows.
// T232: ternary weights (T230) go up as the checkpoint holds them, two bits a weight and a scale a group of 128, and
// are multiplied by the packed shaders alone (shaders.js's TERNARY_PACKED: the codes unpacked to int8 where the
// shader loads them); a device without the packed int8 dot keeps such a model on the CPU.
//
// The first message is claimed before anything is awaited (a module worker's port opens at its first await, and a
// message that comes before onmessage is set is lost: T109).

const shaders = import(new URL(`shaders.js${new URL(import.meta.url).search}`, import.meta.url));

// GPUBufferUsage's values (the name itself is missing where there is no WebGPU)
const STORAGE = 0x80, COPY_DST = 0x8, COPY_SRC = 0x4, MAP_READ = 0x1, UNIFORM = 0x40;
// the bytes that go to the GPU through a copy of their own at a time (writeBuffer takes no view of a shared memory
// everywhere, and copies what it is given at once)
const CHUNK = 8 << 20;
// The right tiled shaders are timed together, in turn, on the model's first layer (its seven matrices by a whole block,
// with the quantizations of a packed one): a submission of n passes of the layer and one of 2n, their difference the
// time of n passes (what a submission costs besides its work is in both and drops out: 2.6 to 8.4 ms waited for on
// the owner's Android, T134, where llm-jp-3 150M's gate by 64 tokens is 0.6 ms of work), n doubled until a submission
// of n takes TIMED_MS (at most MOST_PASSES), PAIRS pairs a shader, the shaders in turn within each round (a device that
// warms up or is loaded meanwhile falls on all of them alike: T168's review), the median of each shader's pairs. On a
// fallback adapter (the CPU in the GPU's place: its times are no GPU's) one pair of one pass
const TIMED_MS = 20, MOST_PASSES = 256, PAIRS = 5;

// T232: the bytes of a row of a matrix or a table ({ n, ternary }) on the GPU: a byte a weight (int8, and int6 widened),
// or two bits (ternary: the codes as the checkpoint holds them, 16 weights a u32). A float32 scale goes with every
// wgsl.GROUP (32) bytes of a row either way: an int8 group of 32 weights, a ternary group of 128
const rowBytes = ({ n, ternary }) => (ternary ? n / 4 : n);
// (whether the model's weights are ternary: its layers' matrices are all of one kind, llama2_numpy's dtype)
const ternaryPlan = (plan) => Object.values(plan.matrices).some((matrix) => matrix.ternary);

let model = null;  // what is on the GPU for the model: the device, the plan, the buffers, the pipelines, the cache
let starting = false, stopping = false, lost = null;

onmessage = ({ data }) => {
  if (data.type === "open") open(data.plan, data.flow);
  else if (data.type === "weights") take(data);
  else if (data.type === "start") start(data.memory, data.plan);
  else if (data.type === "prompt") prompt(data);
  else if (data.type === "tokens") generate(data);
  else if (data.type === "keys") keysOf(data);
  else if (data.type === "stop") stop();
};

// The most a step of putting the model on the GPU may take: a shader's compilation, a check, the timing, a layer's
// weights. SwiftShader took up to 90 s to compile llama.cpp's 64×64 tiles on the development machine (T147), a GPU
// compiles in far less; twice that is a GPU that hangs. The worker then gives the GPU up (the prompt stays on the
// CPU), and says { type: "progress" } after every step meanwhile (forward.js gives up a worker that says nothing for
// longer than this: one the browser ended)
const STEP_MS = 180000;
function within(promise, what) {
  let timer;
  const late = new Promise((_, reject) => {
    timer = setTimeout(() => reject(Object.assign(new Error(`${what} took more than ${STEP_MS / 1000} s`), { late: true })), STEP_MS);
  });
  return Promise.race([promise, late]).finally(() => {
    clearTimeout(timer);
    postMessage({ type: "progress" });
  });
}

// The device, the model's record (model), and the error scopes the layers' buffers go up in; undefined where there is
// none (say(why): unusable() by default) or the worker is stopping
async function openDevice(plan, say = unusable) {
    const wgsl = await shaders;
    if (!self.navigator?.gpu) return say("no WebGPU in a worker here");
    // the browser's own adapter (T148, the review: "high-performance" would keep a laptop's second GPU awake for the
    // whole visit, for prompts that are mostly short; the device measures whichever it gets against its CPU anyway)
    const adapter = await navigator.gpu.requestAdapter();
    if (stopping) return end();
    if (!adapter) return say("no GPU adapter here");
    // T148: a fallback adapter is the CPU doing the GPU's work (SwiftShader, lavapipe): never faster than the CPU's
    // own kernels, and its compilation of the shaders alone took 2 to 4 minutes (T147). Refused before anything is
    // made on it, but where a test asks for it (it is the only WebGPU of CI and of the development machine)
    const info = adapter.info ?? {};
    const fallback = Boolean(info.isFallbackAdapter ?? adapter.isFallbackAdapter);
    if (fallback && !plan.force.fallback) return say("a fallback adapter: the CPU in the GPU's place");
    // T232: ternary weights are multiplied as packed int8 alone (shaders.js's TERNARY_PACKED): said before anything goes up
    const ternary = ternaryPlan(plan);
    if (ternary && !navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product")) {
      return say("ternary weights need the packed int8 dot product of WGSL, which this browser lacks");
    }
    // the largest buffer the device binds (T155: a matrix larger than it goes in pieces, piecesOf)
    const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
    // shader-f16 and subgroups where the adapter has them (a device refuses a feature it lacks), and the adapter's
    // workgroup memory and threads (the tiles and the attention size themselves by them)
    const { maxStorageBufferBindingSize, maxBufferSize, maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX } = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16", "subgroups"].filter((name) => adapter.features.has(name)),
      requiredLimits: { maxStorageBufferBindingSize, maxBufferSize, maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX } });
    device.lost.then((info) => { lost = `the GPU was lost (${info.reason}${info.message ? `: ${info.message}` : ""})`; });
    // T148: what the page keeps of an earlier visit counts only for the same adapter and browser
    // (and the shaders of this deployment: a site whose shaders changed chooses anew, the review of T148)
    // (T232: a ternary model's key holds its own shaders too)
    const key = wgsl.deviceKey(adapter, device, ternary);
    const remembered = plan.remembered?.key === key ? plan.remembered : null;
    model = { device, plan, wgsl, owned: [], limit, info, fallback, remembered, adapter, key, ternary };
    if (stopping) return end();
    // a buffer the device cannot give fails quietly, as an error of these scopes
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    return model;
}

// ---- T156: a model on the GPU alone. open() makes the device and every buffer of the layers' matrices before a byte
// of them comes (plan: forward.js's gpuOnlyPlan(): each matrix's rows and length, and where a layer's values and scales
// start in the checkpoint; the layers are joined for a token, as a model on the GPU alone always runs its steps here),
// and take() writes each stretch of them to its buffer as the worker posts it (routes: [start, end) in the checkpoint,
// the buffer and the offset in it). flow[0] counts the bytes on the GPU (the worker waits on it: a disk read faster
// than the GPU takes it would pile up in the messages). What comes before the buffers are made waits (backlog). A
// failure (no adapter, a buffer the device refused) takes the rest as if written, and start() says it
let opening = null, opened = false, failure = null, flow = null;
const backlog = [];
function open(plan, shared) {
  starting = true;
  flow = new BigInt64Array(shared);
  opening = (async () => {
    try {
      if (!(await openDevice({ ...plan, tokens: true }, (reason) => { failure = reason; }))) {
        failure ??= "the GPU's worker was stopped";
        return;
      }
      model.direct = { routes: [], partial: new Map(), bytes: 0 };
      model.direct.bytes = await uploadLayers(model);
      // T210: the tables too, which nothing holds in the shared memory either
      model.direct.bytes += tablesOn(model);
      model.direct.routes.sort((a, b) => a[0] - b[0]);
      if (stopping) failure ??= "the GPU's worker was stopped";
    } catch (error) {
      failure = String(error?.message ?? error);
    } finally {
      opened = true;
      starting = startAsked;  // (a start() waiting on this is still starting)
      for (const data of backlog.splice(0)) take(data);
      if (stopping) end();
    }
  })();
}
// the bytes at offset of the checkpoint (bytes: a Uint8Array of their own) onto the buffers whose routes they fall in.
// writeBuffer takes whole words: the bytes of a word a stretch begins or ends in the middle of wait for the rest of it
// (partial), which another message brings
function take(data) {
  if (!opened) return void backlog.push(data);
  const { offset, bytes } = data;
  if (!failure) {
    try {
      const { routes } = model.direct, end = offset + bytes.length;
      let lo = 0, hi = routes.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (routes[mid][1] <= offset) lo = mid + 1;
        else hi = mid;
      }
      for (let i = lo; i < routes.length && routes[i][0] < end; i++) {
        const [start, stop, target, at] = routes[i];
        let a = Math.max(start, offset) - offset, here = at + (a + offset - start);
        const b = Math.min(stop, end) - offset;
        for (; a < b && here % 4; a++, here++) partialByte(target, here, bytes[a]);
        const whole = (b - a) & ~3;
        if (whole) model.device.queue.writeBuffer(target, here, bytes, a, whole);
        for (let k = a + whole; k < b; k++) partialByte(target, here + (k - a), bytes[k]);
      }
    } catch (error) {
      failure = String(error?.message ?? error);
    }
  }
  const count = BigInt(bytes.length), done = () => {
    Atomics.add(flow, 0, count);
    Atomics.notify(flow, 0);
  };
  if (failure) done();
  else model.device.queue.onSubmittedWorkDone().then(done, done);
}
// a byte of target at the byte offset at, the word it is in written once all four of its bytes are there
function partialByte(target, at, value) {
  const { partial } = model.direct, word = at - (at % 4);
  if (!partial.has(target)) partial.set(target, new Map());
  const words = partial.get(target);
  if (!words.has(word)) words.set(word, { bytes: new Uint8Array(4), count: 0 });
  const held = words.get(word);
  held.bytes[at % 4] = value;
  if (++held.count < 4) return;
  model.device.queue.writeBuffer(target, word, held.bytes);
  words.delete(word);
  if (!words.size) partial.delete(target);
}

let startAsked = false;
async function start(memory, plan) {
  const began = performance.now();
  starting = startAsked = true;
  try {
    let bytes;
    if (plan.direct) {
      // T156: opened before the checkpoint came, its layers written as they came (the worker waited for the last)
      await opening;
      if (failure && !stopping) return unusable(`the layers did not go up to the GPU (${failure})`);
      if (stopping || !model) return end();
      if (model.direct.partial.size) return unusable("the layers' bytes did not all come to the GPU");
      model.memory = memory;
      model.plan = plan;
      bytes = model.direct.bytes + (await uploadRest(model));
    } else {
      if (!(await openDevice(plan))) return;
      model.memory = memory;
      bytes = await upload(model);
    }
    const { device, adapter, key } = model;
    if (stopping) return end();
    await prepare(model);
    await chooseAttention(model);
    if (stopping) return end();
    await chooseMatrices(model);
    if (stopping) return end();
    bindLayers(model);
    grow(model, Math.max(Math.min(plan.kvStart, plan.seqLen), Math.min(plan.batch, plan.seqLen)));
    await within(device.queue.onSubmittedWorkDone(), "the GPU's work");
    const invalid = await device.popErrorScope(), full = await device.popErrorScope();
    if (stopping) return end();
    if (invalid || full) return unusable(`the GPU did not take the layers (${(invalid ?? full).message})`);
    // (not for the page's tests: SwiftShader took more than STEP_MS to time Llama 3.2 1B's shaders, T147 in CI)
    const blocks = plan.force.quick ? [] : await within(timeBlocks(model), "timing a block");
    if (stopping) return end();
    // T152: a token and the ones after it, where the model's layers were put up for them (tokensLayout). What fails
    // here leaves the tokens on the CPU and the prompts on the GPU
    if (plan.tokens && !model.tokensWhy) {
      try {
        await chooseTokens(model);
      } catch (error) {
        model.tokensWhy = String(error?.message ?? error);
        model.gen = null;
      } finally {
        // (the review of T156: the first layer read back for the checks of a model on the GPU alone goes with them: it
        // stayed in this worker for the whole visit, 113 MB of Llama 3.2 3B, 245 MB of Llama 3.1 Swallow 8B)
        model.firstLayer = model.tableRows = undefined;
      }
      if (stopping) return end();
    }
    if (lost) return unusable(lost);
    const g = model.gen;
    postMessage({ type: "ready", adapter: describe(adapter), key, bytes, seconds: (performance.now() - began) / 1000,
      form: model.form.name, attention: model.attention.name, forms: model.forms, remembered: Boolean(model.form.remembered), blocks,
      tokens: g?.form ? { form: g.form.name, ms: g.ms, forms: g.forms, remembered: g.forms.some((f) => f.remembered), pieces: model.tables.classifier.length,
        attention: g.attention.name, attentions: g.attentions } : null,
      tokensWhy: plan.tokens ? model.tokensWhy ?? null : undefined });
  } catch (error) {
    unusable(String(error?.message ?? error));
  } finally {
    starting = false;
  }
}

function describe(adapter) {
  const info = adapter.info ?? {};
  const name = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "a GPU";
  return (info.isFallbackAdapter ?? adapter.isFallbackAdapter) ? `${name} (a fallback adapter: the CPU in the GPU's place)` : name;
}
// T148: the adapter and the browser whose shaders the page remembers: another GPU, driver architecture or browser
// version chooses anew (the user agent carries the browser's version)
// the tiled shaders of T146 this device can make (shaders.js's promptForms)
const candidates = ({ device, wgsl, ternary }) => wgsl.devicePromptForms(device, ternary);

function unusable(reason) {
  postMessage({ type: "unusable", reason });
  end();
}
function stop() {
  stopping = true;
  if (!starting) end();  // else start() ends at its next step
}
// every buffer and the device let go (T94: a model changed for another leaves nothing on the GPU), and this worker ends.
// T205: it says { type: "ended" } first, for forward.js's release() to read the next model after it. Every way out of
// start() comes here (a stop in the middle of it at its next step), and a stop before or after start() at once
function end() {
  if (model) {
    model.owned.forEach((buffer) => buffer.destroy());
    model.cache?.owned.forEach((buffer) => buffer.destroy());
    model.device.destroy();
    model = null;
  }
  postMessage({ type: "ended" });
  self.close();
}

// ---- buffers
function buffer(m, bytes, usage = STORAGE, owned = m.owned) {
  const made = m.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
  owned.push(made);
  return made;
}
function uniform(m, data, owned = m.owned) {
  const made = buffer(m, data.byteLength, UNIFORM | COPY_DST, owned);
  m.device.queue.writeBuffer(made, 0, data);
  return made;
}
// bytes of the shared memory at address onto the GPU (at offset of target), a CHUNK at a time through a copy
let scratch;
function copyIn(m, target, address, bytes, offset = 0) {
  scratch ??= new Uint8Array(CHUNK);
  for (let done = 0; done < bytes; done += CHUNK) {
    const n = Math.min(CHUNK, bytes - done);
    scratch.set(new Uint8Array(m.memory.buffer, address + done, n));
    m.device.queue.writeBuffer(target, offset + done, scratch, 0, n);
  }
}
// T152's review (T160): values float32 of the shared memory at address narrowed to float16 onto the GPU (at offset
// of target, in bytes of float16), a CHUNK of them at a time: a float32 cache's keys and values going up
let narrowed;
function narrowIn(m, target, address, values, offset = 0) {
  narrowed ??= new Uint16Array(CHUNK / 2);
  for (let done = 0; done < values; done += narrowed.length) {
    const n = Math.min(narrowed.length, values - done);
    m.wgsl.halvesOf(new Float32Array(m.memory.buffer, address + done * 4, n), narrowed);
    m.device.queue.writeBuffer(target, offset + done * 2, narrowed, 0, n);
  }
}
async function readBack(m, source, bytes, from = 0) {
  const target = m.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: MAP_READ | COPY_DST });
  try {
    const encoder = m.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, from, target, 0, Math.ceil(bytes / 4) * 4);
    m.device.queue.submit([encoder.finish()]);
    await target.mapAsync(MAP_READ);
    return target.getMappedRange().slice(0, bytes);
  } finally {
    target.destroy();
  }
}
// what fn does on the GPU, a validation error of it thrown (a pipeline or a bind group the device refused)
async function validated(m, fn) {
  m.device.pushErrorScope("validation");
  let result, failure;
  try {
    result = await fn();
  } catch (error) {
    failure = error;
  }
  const invalid = await m.device.popErrorScope();
  if (failure) throw failure;
  if (invalid) throw new Error(invalid.message);
  return result;
}
const pipelineOf = (m, code, constants) => m.device.createComputePipelineAsync({ layout: "auto",
  compute: { module: m.device.createShaderModule({ code }), entryPoint: "main", constants } });
// (a buffer, or { buffer, offset, size }: a part of one, T155's pieces of a matrix writing into their rows of the output)
const bind = (m, pipeline, buffers) => m.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: buffers.map((buffer, binding) => ({ binding, resource: buffer.buffer ? buffer : { buffer } })) });
function dispatch(pass, pipeline, group, x, y = 1, z = 1) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(x, y, z);
}

// every layer's matrices (values and scales, as the checkpoint holds them; T154: no w3 where there is no gate) and its
// vectors: the weights of its two norms, and (T153) Qwen2's biases of q, k and v, Qwen3's norms of a head of q and of
// k, (T154) GPT-2's and GPT-NeoX's biases of the two LayerNorms and of every matrix (plan.vectors: each one's address in
// the shared memory and its floats a layer). T155: a matrix in pieces of whole rows (piecesOf), a buffer of values and
// one of scales each; int6 (plan.matrices' six) widened to int8 on the way (widener), its scales as they are (the
// quarter of an int6 group's is already the int8 values' scale, T98). The addresses are Numbers in a 64-bit memory too
// (exact to 2^53; the views and copies take them as they are). T232: ternary (plan.matrices' ternary) as it is, a row
// of n weights n / 4 bytes (rowBytes)
// T152: where a token goes on the GPU too (plan.tokens), q, k and v are one matrix a layer, and gate and up one (the
// layer of a token reads them so: shaders.js's fusedMatVec and fusedDp4aMatVec, T150 and T175), each a range of it for
// the prompt's tiled shaders (tokensLayout), and the classifier, the embedding, the final norm and RoPE's table go up
// as well (uploadTokens)
// T156, a model on the GPU alone (m.direct): the layers' buffers are made as the worker opens, before a byte comes, and
// each stretch of the checkpoint they hold is a route ([start, end) in the checkpoint, the buffer and its offset) that
// take() writes the bytes to as they come; the rest (the tables, the norms) goes up from the shared memory as start()
// is asked (uploadRest)
async function upload(m) {
  try {
    const bytes = await uploadLayers(m);
    if (stopping) return bytes;
    return bytes + (await uploadRest(m));
  } finally {
    m.widen?.done();
  }
}
async function uploadLayers(m) {
  const { plan } = m, group = m.wgsl.GROUP;
  let bytes = 0;
  m.matrices = Object.fromEntries(Object.entries(plan.matrices).map(([name, { rows, n, ternary }]) =>
    [name, { rows, n, ternary, pieces: piecesOf(m, rows, rowBytes({ n, ternary })).map(([first, count]) => ({ first, rows: count, layers: [] })) }]));
  const together = plan.tokens ? tokensLayout(m) : null;
  m.together = together;
  // (a model on the GPU alone opens before its tables are placed, and is int8: T156)
  const tables = together && !m.direct ? Object.values(tablesOf(plan)) : [];
  const six = [...Object.values(plan.matrices), ...tables].some((matrix) => matrix.six);
  // (kept for the tables, uploadRest; upload() lets it go)
  const widen = m.widen = six ? await widener(m, tables) : null;
  // (T156: a model on the GPU alone reads its first layer back to check a token's layer against: firstLayer)
  const usage = STORAGE | COPY_DST | (m.direct ? COPY_SRC : 0);
  for (let l = 0; l < plan.layers; l++) {
    // a layer's matrices that are one (T152): a buffer of values and one of scales for all of them
    const joined = together && Object.fromEntries(Object.entries(together.sizes).map(([name, [valueBytes, scaleBytes]]) =>
      [name, [buffer(m, valueBytes, usage), buffer(m, scaleBytes, usage)]]));
    if (joined) (m.joined ??= []).push(joined);
    for (const [name, matrix] of Object.entries(plan.matrices)) {
      const [valuesAt, scalesAt] = matrix.layers[l];
      const home = together?.homes[name], row = rowBytes(matrix);
      for (const piece of m.matrices[name].pieces) {
        const valueBytes = piece.rows * row, scaleBytes = (valueBytes / group) * 4;
        // its own buffers, or its range of the layer's joined ones
        const [values, scales] = home
          ? joined[home.joined].map((b, i) => ({ buffer: b, offset: home.at[i], size: i ? scaleBytes : valueBytes }))
          : [buffer(m, valueBytes, usage), buffer(m, scaleBytes, usage)];
        const from = [valuesAt + piece.first * row, scalesAt + (piece.first * row / group) * 4];
        if (m.direct) {
          // (T156: int8 alone, and T232 ternary, whose bytes are the buffer's as they come)
          m.direct.routes.push([from[0], from[0] + valueBytes, values.buffer ?? values, values.offset ?? 0],
            [from[1], from[1] + scaleBytes, scales.buffer ?? scales, scales.offset ?? 0]);
        } else {
          // an int6 row is 3/4 of an int8 one (24 bytes a group of 32)
          if (matrix.six) widen.into(values, valuesAt + (piece.first * matrix.n * 3) / 4, valueBytes / group);
          else copyIn(m, values.buffer ?? values, from[0], valueBytes, values.offset ?? 0);
          copyIn(m, scales.buffer ?? scales, from[1], scaleBytes, scales.offset ?? 0);
        }
        piece.layers.push([values, scales]);
        bytes += valueBytes + scaleBytes;
      }
    }
    if (m.direct) continue;
    // what was written waits in memory until the GPU takes it: let it, before more comes
    await within(m.device.queue.onSubmittedWorkDone(), `layer ${l + 1}'s weights`);
    if (stopping) return bytes;
  }
  return bytes;
}
// the tables of a token (where the layers are joined for them, T152), and the norms, from the shared memory
async function uploadRest(m) {
  const { plan } = m;
  let bytes = 0;
  if (m.together) bytes += await uploadTokens(m, m.widen);
  m.vectors = {};
  for (const [name, { at, size }] of Object.entries(plan.vectors)) {
    const vectorBytes = plan.layers * size * 4;
    m.vectors[name] = buffer(m, vectorBytes, STORAGE | COPY_DST);
    copyIn(m, m.vectors[name], at, vectorBytes);
    bytes += vectorBytes;
  }
  return bytes;
}

// T155: the pieces of a matrix of rows of n bytes (rowBytes), [first row, rows] each: whole rows, no more bytes than a buffer the device
// binds (m.limit, or plan.force.pieceBytes in the tests), each piece's first row where its part of the output may be
// bound (a multiple of minStorageBufferOffsetAlignment over the 4 bytes of a float32). One piece where the matrix
// fits, as every matrix of the models of the list does at WebGPU's least limit of 128 MiB (the largest, Qwen2.5 7B's
// w1, is 68 MB): llama.cpp's WebGPU keeps each tensor in one buffer within maxStorageBufferBindingSize and refuses a
// larger one (ggml-webgpu.cpp, ggml_backend_webgpu_buffer_type_get_max_size, commit 2145525a, MIT; no line copied),
// where this cuts it by rows, as T94's design had it for a classifier past a buffer
function piecesOf(m, rows, n, most = m.plan.force.pieceBytes) {
  const bytes = Math.min(m.limit, most ?? Infinity), align = m.device.limits.minStorageBufferOffsetAlignment / 4;
  if (rows * n <= bytes) return [[0, rows]];
  const step = Math.floor(bytes / n / align) * align;
  if (!step) throw new Error(`${align} rows of ${n} weights are more than a buffer of this GPU (${bytes} bytes)`);
  return Array.from({ length: Math.ceil(rows / step) }, (_, i) => [i * step, Math.min(step, rows - i * step)]);
}

// T152: the tables a token needs besides the layers (plan.tokens; T210, a model on the GPU alone as it opens:
// plan.tables): the classifier, and the embedding where it is another table (else the classifier is the embedding
// too), each made by make(spec) where it is given, else the specs themselves
function tablesOf(plan, make = (spec) => spec) {
  const { classifier, embedding } = plan.tables ?? plan.tokens, made = make(classifier);
  return { classifier: made, embedding: embedding ? make(embedding) : made };
}
// T209: a table ({ rows, n }) in pieces of rows where it is past what the device binds (piecesOf), a buffer of values
// and one of scales each, [{ first, rows, values, scales }]; fill(piece, valueBytes, scaleBytes) puts its bytes there
function tablePieces(m, spec, fill, usage = STORAGE | COPY_DST) {
  return piecesOf(m, spec.rows, rowBytes(spec), m.plan.force.tablePieceBytes ?? m.plan.force.pieceBytes).map(([first, count]) => {
    const valueBytes = count * rowBytes(spec), scaleBytes = (valueBytes / m.wgsl.GROUP) * 4;
    const piece = { first, rows: count, values: buffer(m, valueBytes, usage), scales: buffer(m, scaleBytes, usage) };
    fill(piece, valueBytes, scaleBytes);
    return piece;
  });
}
// T210: the tables of a model on the GPU alone, made as it opens (plan.tables: where each starts in the checkpoint),
// their stretches routes that take() writes the bytes to as they come; COPY_SRC for the check's rows (checkTokens).
// The bytes on the GPU
function tablesOn(m) {
  const group = m.wgsl.GROUP;
  let bytes = 0;
  m.tables = tablesOf(m.plan, (spec) => tablePieces(m, spec, ({ first, values, scales }, valueBytes, scaleBytes) => {
    const [valuesAt, scalesAt] = spec.at, from = [valuesAt + first * rowBytes(spec), scalesAt + (first * rowBytes(spec) / group) * 4];
    m.direct.routes.push([from[0], from[0] + valueBytes, values, 0], [from[1], from[1] + scaleBytes, scales, 0]);
    bytes += valueBytes + scaleBytes;
  }, STORAGE | COPY_DST | COPY_SRC));
  return bytes;
}
// T152: how a token's layer holds its matrices (shaders.js's fusedMatVec and fusedDp4aMatVec read q, k and v as one
// matrix, gate and up as one, up's rows after gate's): a buffer of values and one of scales a layer for each (sizes),
// and each matrix's range of them (homes: its offsets in bytes), which the prompt's tiled shaders bind as a matrix of
// its own. A range must start where the device binds a buffer (minStorageBufferOffsetAlignment: a matrix of a
// multiple of 2048 weights, the values and the scales alike at 256; stories15M's k starts at 82944 weights, T150's
// (b); T232: of 8192 ternary weights, whose scales are a byte to 32 weights), and a matrix a token reads must be one piece (T155). The tables may be in pieces (T209: uploadTokens). null
// where it cannot, and why in m.tokensWhy: the prompts go on the GPU as before, the tokens stay on the CPU
function tokensLayout(m) {
  const { plan } = m, align = m.device.limits.minStorageBufferOffsetAlignment, group = m.wgsl.GROUP;
  const why = (reason) => {
    m.tokensWhy = reason;
    return null;
  };
  // (T226: GPT-2's and GPT-NeoX's FFN has no gate: w1 is a matrix of its own, as wo and w2 are)
  const gated = Boolean(m.matrices.w3);
  for (const name of ["wo", "w2", ...(gated ? [] : ["w1"])]) if (m.matrices[name].pieces.length > 1) return why(`${name} is past a buffer of this GPU`);
  const homes = {}, sizes = {};
  for (const [joined, names] of Object.entries({ qkv: ["wq", "wk", "wv"], ...(gated ? { gateUp: ["w1", "w3"] } : {}) })) {
    let values = 0, scales = 0;
    for (const name of names) {
      const matrix = m.matrices[name];
      if (matrix.pieces.length > 1) return why(`${name} is past a buffer of this GPU`);
      if (values % align || scales % align) {
        return why(`${name} would not start where this GPU binds a buffer`);
      }
      homes[name] = { joined, at: [values, scales] };
      values += matrix.rows * rowBytes(matrix);
      scales += (matrix.rows * rowBytes(matrix) / group) * 4;
    }
    if (values > m.limit) return why(`${names.join(" and ")} together are past a buffer of this GPU`);
    sizes[joined] = [values, scales];
  }
  return { homes, sizes };
}
// T152: the tables onto the GPU (int6 widened as a layer's matrices are), the final norm's weights, and RoPE's table
// of every position, a row a position: the cos of its headSize / 2 angles, then their sin (shaders.js's fusedMatVec
// reads it so, T151; from the CPU's two tables, Llama 3's scaling in them)
// T209: a table in pieces of rows where it is past what the device binds (piecesOf: Llama 3.2 3B's classifier is
// 394 MB, the owner's Android binds 256 MiB; T152's review (e)), [{ first, rows, values, scales }] each. The
// classifier is then a dispatch a piece into its range of the logits, and EMBED one a piece (tokenPass)
// T210: a model on the GPU alone makes them as it opens, their bytes routes of the checkpoint as the layers' are
// (tablesOn), and only the final norm and RoPE's table come from the shared memory here
async function uploadTokens(m, widen) {
  const { plan } = m, group = m.wgsl.GROUP, half = plan.headSize / 2;
  let bytes = 0;
  m.tables ??= tablesOf(plan, (spec) => tablePieces(m, spec, (piece, valueBytes, scaleBytes) => {
    const { first, values, scales } = piece, { n, six, at: [valuesAt, scalesAt] } = spec;
    if (six) widen.into(values, valuesAt + (first * n * 3) / 4, valueBytes / group);
    else copyIn(m, values, valuesAt + first * rowBytes(spec), valueBytes);
    copyIn(m, scales, scalesAt + (first * rowBytes(spec) / group) * 4, scaleBytes);
    bytes += valueBytes + scaleBytes;
  }));
  // (T226: the final LayerNorm's bias, and GPT-2's learned positions, a row a position as the CPU widened them: a
  // step adds its position's row to the embedding's, as the CPU's embed() does)
  const vector = (address, floats) => {
    const made = buffer(m, floats * 4, STORAGE | COPY_DST);
    copyIn(m, made, address, floats * 4);
    bytes += floats * 4;
    return made;
  };
  m.finalNorm = vector(plan.tokens.final, plan.dim);
  m.finalBias = plan.tokens.finalBias ? vector(plan.tokens.finalBias, plan.dim) : null;
  m.positions = plan.tokens.positions ? vector(plan.tokens.positions, plan.seqLen * plan.dim) : null;
  m.angleTable = buffer(m, plan.seqLen * plan.headSize * 4, STORAGE | COPY_DST);
  const rows = Math.max(1, Math.floor(CHUNK / (plan.headSize * 4))), chunk = new Float32Array(rows * plan.headSize);
  // (GPT-2 turns nothing and has no tables: the buffer stays, bound and never read)
  for (let p = 0; plan.turned && p < plan.seqLen; p += rows) {
    const count = Math.min(rows, plan.seqLen - p);
    for (let r = 0; r < count; r++) {
      chunk.set(new Float32Array(m.memory.buffer, plan.cos + (p + r) * half * 4, half), r * plan.headSize);
      chunk.set(new Float32Array(m.memory.buffer, plan.sin + (p + r) * half * 4, half), r * plan.headSize + half);
    }
    m.device.queue.writeBuffer(m.angleTable, p * plan.headSize * 4, chunk, 0, count * plan.headSize);
  }
  bytes += plan.seqLen * plan.headSize * 4;
  await within(m.device.queue.onSubmittedWorkDone(), "the classifier's weights");
  return bytes;
}

// T155: shaders.js's WIDEN_SIX compiled and checked against JavaScript (sixValues), and into(values, address, groups):
// groups of int6 at address in the shared memory widened into values (a buffer of int8, or T152 a range of one) on
// the GPU. The packed bytes go
// through one buffer of their own, written again for every piece (the queue runs a write after the submissions before
// it); done() lets it go.
async function widener(m, tables = []) {
  const { device, wgsl } = m;
  const pipeline = await within(validated(m, () => pipelineOf(m, wgsl.WIDEN_SIX)), "compiling the widening of int6");
  const owned = [];
  // (T152: and the classifier's and the embedding's, one piece each)
  const largest = Math.max(CHECK_GROUPS * 24, ...Object.entries(m.plan.matrices).filter(([, matrix]) => matrix.six)
    .flatMap(([name, { n }]) => m.matrices[name].pieces.map((piece) => (piece.rows * n * 3) / 4)),
  ...tables.filter((table) => table.six).map(({ rows, n }) => (rows * n * 3) / 4));
  const packed = buffer(m, largest, STORAGE | COPY_DST, owned);
  const run = (values, groups, most = device.limits.maxComputeWorkgroupsPerDimension) => {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, pipeline, bind(m, pipeline, [packed, values, uniform(m, new Uint32Array([groups, 0, 0, 0]), owned)]),
      ...wgsl.sixDispatch(groups, most));
    pass.end();
    device.queue.submit([encoder.finish()]);
  };
  const done = () => owned.splice(0).forEach((b) => b.destroy());
  try {
    const wrong = await within(checkWiden(m, packed, run), "checking the widening of int6");
    if (wrong) throw new Error(`the widening of int6 is wrong on this GPU: ${wrong}`);
  } catch (error) {
    done();
    throw error;
  }
  return {
    into(values, address, groups) {
      copyIn(m, packed, address, groups * 24);
      run(values, groups);
    },
    done,
  };
}
// The check: CHECK_GROUPS groups of random bytes (any 24 bytes are a group), dispatched over rows of 7 workgroups (a
// second dimension, and threads past the last group), into a buffer with 64 groups more of a known byte after them
// (which no thread may write), against JavaScript's to the bit. The reason it is wrong, or null
const CHECK_GROUPS = 1000;
async function checkWiden(m, packed, run) {
  const bytes = new Uint8Array(CHECK_GROUPS * 24).map(() => (Math.random() * 256) | 0), after = 64 * 32, owned = [];
  try {
    const values = buffer(m, CHECK_GROUPS * 32 + after, STORAGE | COPY_DST | COPY_SRC, owned);
    m.device.queue.writeBuffer(values, 0, new Uint8Array(CHECK_GROUPS * 32 + after).fill(0x5a));
    m.device.queue.writeBuffer(packed, 0, bytes);
    run(values, CHECK_GROUPS, 7);
    const got = new Int8Array(await readBack(m, values, CHECK_GROUPS * 32 + after)), want = m.wgsl.sixValues(bytes);
    const at = want.findIndex((v, i) => got[i] !== v);
    if (at >= 0) return `value ${at % 32} of group ${(at / 32) | 0} is ${got[at]}, JavaScript's ${want[at]}`;
    const past = got.subarray(want.length).findIndex((v) => v !== 0x5a);
    return past >= 0 ? `byte ${past} past the last group was written` : null;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

// the pipelines of a layer's small steps and the buffers of a block's activations
async function prepare(m) {
  const { plan, wgsl } = m, B = plan.batch, gated = Boolean(plan.matrices.w3);
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, widest = Math.max(plan.dim, qDim, plan.hidden);
  // T154: LayerNorm and GELU where the model has them (GPT-2, GPT-NeoX), else RMSNorm and SwiGLU
  [m.norm, m.headNorm, m.add, m.rope, m.activation, m.quantize] = await within(Promise.all([plan.layerNorm ? wgsl.LAYER_NORM : wgsl.RMSNORM,
    wgsl.HEAD_NORM, wgsl.ADD, wgsl.ROPE, gated ? wgsl.SWIGLU : wgsl.GELU, wgsl.QUANTIZE].map((code) => pipelineOf(m, code))), "the small steps' shaders");
  // a block's tokens, each array dense: token t's row at t times its width
  m.x = buffer(m, B * plan.dim * 4, STORAGE | COPY_DST);
  m.xb = buffer(m, B * Math.max(plan.dim, qDim) * 4, STORAGE | COPY_DST);
  m.q = buffer(m, B * qDim * 4);
  m.k = buffer(m, B * kvDim * 4);
  m.v = buffer(m, B * kvDim * 4);
  m.gate = buffer(m, B * plan.hidden * 4, STORAGE | COPY_DST);
  m.up = gated ? buffer(m, B * plan.hidden * 4) : null;
  // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, which xb cannot hold (the attention
  // writes there first)
  m.ffnInput = plan.parallel ? buffer(m, B * plan.dim * 4) : m.xb;
  // the packed shaders' input: the activations of a matrix quantized, 8 bits a value and a float32 scale a group
  m.xq = buffer(m, B * widest);
  m.xs = buffer(m, B * (widest / wgsl.GROUP) * 4);
  m.angles = buffer(m, B * plan.headSize * 4, STORAGE | COPY_DST);
  m.step = buffer(m, 16, UNIFORM | COPY_DST);
  m.readback = buffer(m, 2 * plan.layers * B * kvDim * 2, MAP_READ | COPY_DST);
  // what a request writes into them, from the shared memory
  m.rows = new Float32Array(B * plan.dim);
  m.turns = new Float32Array(B * plan.headSize);
  m.ropeShape = uniform(m, new Uint32Array([plan.heads, plan.kvHeads, plan.headSize, plan.turned]));
  const hidden = uniform(m, new Uint32Array([plan.hidden, 0, 0, 0]));
  m.activationGroup = bind(m, m.activation, gated ? [m.gate, m.up, hidden, m.step] : [m.gate, hidden, m.step]);
  // QUANTIZE of the inputs the matrices read: xb (the norm's, dim wide; the attention's, heads × headSize wide, which
  // is not dim where the heads are of another size, T153) and gate (SwiGLU's)
  const quantizing = (from, n) => ({ group: bind(m, m.quantize, [from, m.xq, m.xs, uniform(m, new Uint32Array([n, n, 0, 0])), m.step]),
    x: Math.ceil(n / wgsl.GROUP / 64) });
  m.quantizeXb = quantizing(m.xb, plan.dim);
  m.quantizeAttention = quantizing(m.xb, qDim);
  m.quantizeFfn = plan.parallel ? quantizing(m.ffnInput, plan.dim) : m.quantizeXb;
  m.quantizeGate = quantizing(m.gate, plan.hidden);
  // T210: a model on the GPU alone embeds a block's tokens here (the CPU holds no embedding): EMBED_ROWS a piece of
  // the table (T209), from the block's ids into x
  if (plan.direct) {
    // (T232: from a table of ternary weights, EMBED_ROWS_TERNARY)
    const rowsCode = tablesOf(plan).embedding.ternary ? wgsl.EMBED_ROWS_TERNARY : wgsl.EMBED_ROWS;
    m.embedRows = await within(validated(m, () => pipelineOf(m, rowsCode)), "compiling the embedding's rows");
    m.ids = buffer(m, B * 4, STORAGE | COPY_DST);
    m.embedGroups = m.tables.embedding.map((piece) => bind(m, m.embedRows,
      [piece.values, piece.scales, m.ids, m.x, uniform(m, new Uint32Array([plan.dim, piece.first, piece.rows, 0]))]));
  }
}

// ---- the attention: llama.cpp's flash attention with tiles, with f16 in the workgroup's memory where there is
// shader-f16 and its subgroups where there are (and subgroup_id), else the same without them; each checked first
// against JavaScript (checkAttention), the next one tried where it is wrong
async function chooseAttention(m) {
  const { device, plan, wgsl } = m;
  const half = device.features.has("shader-f16");
  const subgroups = device.features.has("subgroups") && Boolean(navigator.gpu.wgslLanguageFeatures?.has("subgroup_id"));
  const limits = { memory: device.limits.maxComputeWorkgroupStorageSize, threads: threadsOf(device),
    subgroupMin: m.info.subgroupMinSize, subgroupMax: m.info.subgroupMaxSize };
  const tried = [];
  const nameOf = (option) => `llama.cpp flash attention tiles${option.half ? ", f16" : ""}${option.subgroups ? ", subgroups" : ""}`;
  // T148: the one the page remembers for this adapter first (it was the first right one then: the other was wrong)
  const options = [{ half, subgroups }, { half: false, subgroups: false }].filter((o, i) => i === 0 || o.half !== half || o.subgroups !== subgroups)
    .sort((a, b) => (nameOf(b) === m.remembered?.attention) - (nameOf(a) === m.remembered?.attention));
  for (const option of options) {
    const shape = wgsl.flashShape({ headSize: plan.headSize, ...option, ...limits });
    const name = nameOf(option);
    if (plan.force.attention && plan.force.attention !== name) continue;
    if (shape.none) {
      tried.push(`${name}: ${shape.none}`);
      continue;
    }
    try {
      const pipeline = await within(validated(m, () => pipelineOf(m, wgsl.flashTile(shape))), `compiling ${name}`);
      const wrong = await within(checkAttention(m, pipeline), `checking ${name}`);
      if (!wrong) {
        m.attention = { name, pipeline, shape };
        return;
      }
      tried.push(`${name}: ${wrong}`);
    } catch (error) {
      if (error?.late) throw error;
      tried.push(`${name}: ${error?.message ?? error}`);
    }
  }
  throw new Error(`no attention is right on this GPU (${tried.join("; ") || `none named ${plan.force.attention}`})`);
}

// The attention on made-up numbers against JavaScript's: 9 tokens at positions 61 to 69 (two tiles of 4 and one
// token of a third), 4 heads of q on 2 of keys and values, 70 positions (a KV_TILE of 64 and a part of one) of keys and
// values as float16 of random bits between 2^-3 and 4 in size. Each token's output against its softmax over the
// positions up to its own, no more than LINE of the largest |value| of its head: f16 weights in the workgroup's
// memory are within 2^-10 of theirs (each rounded either way, WGSL leaves the direction to the device) and so the
// output within about 1e-3 of the largest value (the sum of the weights is float32), while a wrong mask, head or
// tile is off by a tenth and more.
const LINE = 4e-3;
async function checkAttention(m, pipeline) {
  const { plan } = m, size = plan.headSize, heads = 4, kvHeads = 2, tokens = 9, pos = 61, positions = pos + tokens;
  const kvDim = kvHeads * size, qDim = heads * size, scale = 1 / Math.sqrt(size);
  const q = new Float32Array(tokens * qDim).map(() => Math.random() * 2 - 1);
  const halfBits = () => ((Math.random() < 0.5 ? 0x8000 : 0) | ((12 + ((Math.random() * 5) | 0)) << 10) | ((Math.random() * 1024) | 0));
  const keys = new Uint16Array(positions * kvDim).map(halfBits), values = new Uint16Array(positions * kvDim).map(halfBits);
  const owned = [];
  const make = (data, usage = STORAGE | COPY_DST) => {
    const made = buffer(m, data.byteLength, usage, owned);
    m.device.queue.writeBuffer(made, 0, data);
    return made;
  };
  try {
    const out = buffer(m, tokens * qDim * 4, STORAGE | COPY_SRC, owned);
    const params = new ArrayBuffer(16);
    new Uint32Array(params, 0, 2).set([heads, kvHeads]);
    new Float32Array(params, 8, 1)[0] = scale;
    const group = bind(m, pipeline, [make(q), make(keys), make(values), out, uniform(m, params, owned),
      uniform(m, new Uint32Array([tokens, pos, 0, 0]), owned)]);
    const encoder = m.device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, pipeline, group, heads * Math.ceil(tokens / m.wgsl.FLASH_Q_TILE));
    pass.end();
    m.device.queue.submit([encoder.finish()]);
    const got = new Float32Array(await readBack(m, out, tokens * qDim * 4));
    const k = Float32Array.from(keys, halfToFloat), v = Float32Array.from(values, halfToFloat);
    let worst = 0;
    for (let t = 0; t < tokens; t++) {
      for (let h = 0; h < heads; h++) {
        const kv = Math.floor(h / (heads / kvHeads)) * size, row = t * qDim + h * size, seen = pos + t + 1;
        const scores = new Float64Array(seen);
        for (let p = 0; p < seen; p++) {
          for (let d = 0; d < size; d++) scores[p] += q[row + d] * scale * k[p * kvDim + kv + d];
        }
        const most = Math.max(...scores);
        let sum = 0, largest = 0;
        for (let p = 0; p < seen; p++) sum += (scores[p] = Math.exp(scores[p] - most));
        for (let p = 0; p < positions; p++) for (let d = 0; d < size; d++) largest = Math.max(largest, Math.abs(v[p * kvDim + kv + d]));
        for (let d = 0; d < size; d++) {
          let want = 0;
          for (let p = 0; p < seen; p++) want += scores[p] * v[p * kvDim + kv + d];
          worst = Math.max(worst, Math.abs(got[row + d] - want / sum) / largest);
        }
      }
    }
    return worst <= LINE ? null : `its output is ${worst.toExponential(2)} of the largest value from JavaScript's (line ${LINE})`;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}
function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, fraction = h & 1023;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// ---- the matrices: the tiled shaders of T146 that this device can make (shaders.js's promptForms), each checked on a
// small matrix against JavaScript (checkForm: subgroups, f16 rounding and drivers differ from device to device, and
// only the device can say), the right ones timed on the model's own gate matrix by a whole block, and the fastest
// taken. forms: what each came to (ms, or why not), for the console. plan.force.matrices (tests only): that form alone, untimed.
async function chooseMatrices(m) {
  const { plan } = m;
  let forms = candidates(m);
  if (plan.force.matrices) forms = forms.filter((form) => form.name === plan.force.matrices);
  m.forms = [];
  // T148: the shader the page remembers for this adapter, alone, where it is one this device still makes and it is
  // still right here (a driver may have changed under the same names): no other is compiled or timed. Else all of them
  const kept = !plan.force.matrices && forms.find((form) => !form.none && form.name === m.remembered?.matrices);
  if (kept) {
    try {
      const tiled = { ...kept, remembered: true, pipeline: await within(validated(m, () => pipelineOf(m, kept.code, kept.constants)), `compiling ${kept.name}`) };
      const wrong = await within(checkForm(m, tiled), `checking ${kept.name}`);
      if (!wrong) {
        m.forms.push({ name: kept.name, remembered: true });
        m.form = tiled;
        return;
      }
      m.forms.push({ name: kept.name, none: `remembered, but wrong now: ${wrong}` });
    } catch (error) {
      if (error?.late) throw error;
      m.forms.push({ name: kept.name, none: `remembered, but ${error?.message ?? error}` });
    }
    if (stopping) return;
  }
  const right = [];
  for (const form of forms) {
    if (plan.force.quick && right.length) break;  // T148: the page's tests, the first right one untimed
    if (form === kept) continue;  // wrong just now
    if (form.none) {
      m.forms.push({ name: form.name, none: form.none });
      continue;
    }
    try {
      const pipeline = await within(validated(m, () => pipelineOf(m, form.code, form.constants)), `compiling ${form.name}`);
      const tiled = { ...form, pipeline };
      const wrong = await within(checkForm(m, tiled), `checking ${form.name}`);
      if (wrong) m.forms.push({ name: form.name, none: `wrong: ${wrong}` });
      else right.push(tiled);
    } catch (error) {
      if (error?.late) throw error;
      m.forms.push({ name: form.name, none: String(error?.message ?? error) });
    }
    if (stopping) return;
  }
  if (!right.length) throw new Error(`no tiled shader is right on this GPU (${m.forms.map((f) => `${f.name}: ${f.none}`).join("; ") || `none named ${plan.force.matrices}`})`);
  const ms = right.length > 1 ? await within(timeForms(m, right), "timing the tiled shaders") : [0];
  right.forEach((form, i) => m.forms.push({ name: form.name, ms: ms[i] }));
  m.form = right[ms.indexOf(Math.min(...ms))];
}
// the threads of a workgroup of one dimension this device takes
const threadsOf = (device) => Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX);

// layer l's matrix name by the tokens of from into to (added where add): a bind group of form and its rows for each
// piece (T155), which writes its rows of to (bound from its first row; a token's row of to is the whole matrix's)
const productGroups = (m, form, name, l, from, to, add, owned = m.owned) => {
  const { rows, n, pieces } = m.matrices[name];
  return pieces.map(({ first, rows: count, layers }) => {
    const out = first ? { buffer: to, offset: first * 4, size: to.size - first * 4 } : to;
    return { rows: count, group: bind(m, form.pipeline, [...layers[l], form.packed ? m.xq : from, out,
      uniform(m, new Uint32Array([count, n / 4, n / m.wgsl.GROUP, 0, n, rows, add ? 1 : 0, 0]), owned), m.step, ...(form.packed ? [m.xs] : [])]) };
  });
};
// the workgroups of form for rows by count tokens: the tiles numbered over x, then y (as T146's shaders number them)
function multiply(m, pass, form, group, rows, count) {
  const tiles = Math.ceil(rows / form.tile.rows) * Math.ceil(count / form.tile.tokens);
  const across = Math.min(tiles, m.device.limits.maxComputeWorkgroupsPerDimension);
  dispatch(pass, form.pipeline, group, across, Math.ceil(tiles / across));
}

// T146's check (public/benchmark/gpu.js's checkTiled) of a form: 300 rows of 544 (17 groups of 32: a part of a tile
// of rows everywhere), 11 and 70 tokens (a part of a tile of tokens; two or one and a part), and 11 tokens whose x and
// y are wider than the product (xStride 608, yStride 320), each product twice into the same y (the second added:
// shape.add) against JavaScript's (shaders.js's tiledOff). The reason it is wrong, or null
// T232, a ternary form: rows of 640 weights (5 groups of 128 with a scale each, 20 of the activations' groups of 32:
// a weight's scale holds for four steps of the width, and the next is another), the weights random bytes of codes (all
// four codes, the one no file has too: every bit of a word matters), against JavaScript's on the codes' values
// (shaders.js's ternaryValues)
async function checkForm(m, form) {
  const { device, wgsl } = m, rows = 300, ternary = Boolean(form.ternary), n = ternary ? 640 : 544, perRow = n / wgsl.GROUP;
  const group = ternary ? 4 * wgsl.GROUP : wgsl.GROUP;
  const stored = ternary ? new Uint8Array((rows * n) / 4).map(() => (Math.random() * 256) | 0) : new Int8Array(rows * n).map(() => (Math.random() * 256) | 0);
  const w = ternary ? wgsl.ternaryValues(stored) : stored, s = new Float32Array((rows * n) / group).map(() => Math.random() * 0.01);
  for (const { tokens, wider } of [{ tokens: 11, wider: 0 }, { tokens: 70, wider: 0 }, { tokens: 11, wider: 64 }]) {
    const xStride = n + wider, yStride = rows + (wider ? 20 : 0), owned = [];
    const x = new Float32Array(tokens * xStride).map(() => (Math.random() - 0.5) * 2);
    const make = (data, usage = STORAGE | COPY_DST | COPY_SRC) => {
      const made = buffer(m, data.byteLength, usage, owned);
      device.queue.writeBuffer(made, 0, data);
      return made;
    };
    try {
      const io = { xq: buffer(m, tokens * xStride, STORAGE | COPY_SRC, owned), xs: buffer(m, tokens * (xStride / wgsl.GROUP) * 4, STORAGE | COPY_SRC, owned),
        step: uniform(m, new Uint32Array([tokens, 0, 0, 0]), owned) };
      const wb = make(stored), sb = make(s), xb = make(x), y = make(new Float32Array(tokens * yStride));
      const shape = (add) => uniform(m, new Uint32Array([rows, n / 4, perRow, 0, xStride, yStride, add ? 1 : 0, 0]), owned);
      const group = (add) => bind(m, form.pipeline, [wb, sb, form.packed ? io.xq : xb, y, shape(add), io.step, ...(form.packed ? [io.xs] : [])]);
      const groups = await validated(m, () => [group(false), group(true)]);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      if (form.packed) {
        dispatch(pass, m.quantize, bind(m, m.quantize, [xb, io.xq, io.xs, uniform(m, new Uint32Array([n, xStride, 0, 0]), owned), io.step]),
          Math.ceil(n / wgsl.GROUP / 64), tokens);
      }
      groups.forEach((g) => multiply(m, pass, form, g, rows, tokens));
      pass.end();
      device.queue.submit([encoder.finish()]);
      const got = new Float32Array(await readBack(m, y, tokens * yStride * 4));
      const xq = form.packed ? new Int8Array(await readBack(m, io.xq, tokens * xStride)) : null;
      const xs = form.packed ? new Float32Array(await readBack(m, io.xs, tokens * (xStride / wgsl.GROUP) * 4)) : null;
      const { wrong } = wgsl.tiledOff({ w, s, x, got, xq, xs, rows, n, tokens, xStride, yStride, half: form.half, group });
      if (wrong) return `${wrong} (${tokens} tokens${wider ? ", wider x and y" : ""})`;
    } finally {
      owned.forEach((b) => b.destroy());
    }
  }
  return null;
}

// ms of a pass of the model's first layer with each of forms (see TIMED_MS): the seven matrices by plan.batch tokens,
// each packed one's inputs quantized first
async function timeForms(m, forms) {
  const { device, plan } = m, owned = [], layer = [];
  try {
    device.queue.writeBuffer(m.step, 0, new Uint32Array([plan.batch, 0, 0, 0]));
    const products = [["wq", m.xb, m.q, false, m.quantizeXb], ["wk", m.xb, m.k], ["wv", m.xb, m.v], ["wo", m.xb, m.x, true, m.quantizeAttention],
      ["w1", m.ffnInput, m.gate, false, m.quantizeFfn], ["w3", m.ffnInput, m.up], ["w2", m.gate, m.x, true, m.quantizeGate]]
      .filter(([name]) => m.matrices[name]);
    // (a matrix's input quantized once, before its first piece)
    for (const form of forms) {
      layer.push(await validated(m, () => products.flatMap(([name, from, to, add, quantize]) =>
        productGroups(m, form, name, 0, from, to, add, owned).map((piece, i) => ({ ...piece, quantize: i ? null : quantize })))));
    }
    const submission = async (i, passes) => {
      const form = forms[i], encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let p = 0; p < passes; p++) {
        for (const { group, rows, quantize } of layer[i]) {
          if (form.packed && quantize) dispatch(pass, m.quantize, quantize.group, quantize.x, plan.batch);
          multiply(m, pass, form, group, rows, plan.batch);
        }
      }
      pass.end();
      const began = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    const passes = [], differences = forms.map(() => []);
    for (let i = 0; i < forms.length; i++) {
      await submission(i, 1);  // warm
      let n = 1;
      while (!m.fallback && n < MOST_PASSES && (await submission(i, n)) < TIMED_MS) n *= 2;
      passes.push(n);
    }
    for (let round = 0; round < (m.fallback ? 1 : PAIRS); round++) {
      for (let i = 0; i < forms.length; i++) {
        const once = await submission(i, passes[i]), twice = await submission(i, 2 * passes[i]);
        differences[i].push((twice - once) / passes[i]);
      }
    }
    return differences.map((d) => d.sort((a, b) => a - b)[d.length >> 1]);
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

// every layer's bind groups but those of the cache, with the form chosen. T153: a bias (Qwen2's) added to q, k and
// v, or (Qwen3's) the norm of every head of q and of k, where the model has them. T154: LayerNorm with its bias, and a
// bias after o, w1 and w2 (GPT-2's, GPT-NeoX's)
function bindLayers(m) {
  const { plan } = m, form = m.form, V = m.vectors;
  // (a piece each, T155)
  const product = (name, l, from, to, add = false) => productGroups(m, form, name, l, from, to, add);
  m.layers = [];
  for (let l = 0; l < plan.layers; l++) {
    const norm = new ArrayBuffer(16);
    new Uint32Array(norm, 0, 2).set([plan.dim, l * plan.dim]);
    new Float32Array(norm, 8, 1)[0] = plan.eps;
    const normShape = uniform(m, norm);
    const added = (to, name) => V[name] && { group: bind(m, m.add, [to, V[name], uniform(m, new Uint32Array([plan.vectors[name].size, l * plan.vectors[name].size, 0, 0])), m.step]),
      x: Math.ceil(plan.vectors[name].size / 64) };
    // T154: LayerNorm reads its bias from the same place in its own buffer
    const normed = (weight, bias, out) => bind(m, m.norm, plan.layerNorm ? [m.x, V[weight], V[bias], out, normShape, m.step]
      : [m.x, V[weight], out, normShape, m.step]);
    const headNorm = (of, name, heads) => {
      if (!V[name]) return null;
      const shape = new ArrayBuffer(16);
      new Uint32Array(shape, 0, 2).set([plan.headSize, l * plan.headSize]);
      new Float32Array(shape, 8, 1)[0] = plan.eps;
      return { group: bind(m, m.headNorm, [of, V[name], uniform(m, shape), m.step]), heads };
    };
    m.layers.push({
      attentionNorm: normed("attention", "attentionBias", m.xb),
      q: product("wq", l, m.xb, m.q), k: product("wk", l, m.xb, m.k), v: product("wv", l, m.xb, m.v),
      biases: [added(m.q, "bq"), added(m.k, "bk"), added(m.v, "bv")].filter(Boolean),
      headNorms: [headNorm(m.q, "qNorm", plan.heads), headNorm(m.k, "kNorm", plan.kvHeads)].filter(Boolean),
      o: product("wo", l, m.xb, m.x, true), oBias: added(m.x, "bo"),
      ffnNorm: normed("ffn", "ffnBias", m.ffnInput),
      gate: product("w1", l, m.ffnInput, m.gate), gateBias: added(m.gate, "b1"),
      up: m.matrices.w3 ? product("w3", l, m.ffnInput, m.up) : null,
      down: product("w2", l, m.gate, m.x, true), downBias: added(m.x, "b2"),
    });
  }
}

// The GPU's own keys and values (float16, a pair to a u32), per layer [positions][kvDim], grown as the CPU's cache
// grows (doubling from plan.kvStart: a prompt seldom needs the whole context) and kept from block to block; with them
// the bind groups that read them
function grow(m, needed) {
  const { device, plan } = m, old = m.cache;
  const kvDim = plan.kvHeads * plan.headSize, row = kvDim * 2;
  const capacity = Math.min(Math.max(2 * (old?.capacity ?? 0), needed), plan.seqLen);
  if (capacity * row > m.limit) throw new Error(`the keys of ${capacity} positions are more than a buffer of this GPU`);
  const cache = { capacity, owned: [], keys: [], values: [], rope: [], attention: [] };
  const usage = STORAGE | COPY_SRC | COPY_DST;
  const encoder = device.createCommandEncoder();
  for (let l = 0; l < plan.layers; l++) {
    const keys = buffer(m, capacity * row, usage, cache.owned), values = buffer(m, capacity * row, usage, cache.owned);
    if (old) {
      encoder.copyBufferToBuffer(old.keys[l], 0, keys, 0, old.capacity * row);
      encoder.copyBufferToBuffer(old.values[l], 0, values, 0, old.capacity * row);
    }
    cache.keys.push(keys);
    cache.values.push(values);
  }
  device.queue.submit([encoder.finish()]);
  // the old ones go once the copies are done (a buffer destroyed after its submission lives until the GPU is through)
  old?.owned.forEach((buffer) => buffer.destroy());
  const params = new ArrayBuffer(16);
  new Uint32Array(params, 0, 2).set([plan.heads, plan.kvHeads]);
  new Float32Array(params, 8, 1)[0] = 1 / Math.sqrt(plan.headSize);
  const attentionParams = uniform(m, params, cache.owned);
  for (let l = 0; l < plan.layers; l++) {
    cache.rope.push(bind(m, m.rope, [m.q, m.k, m.v, cache.keys[l], cache.values[l], m.angles, m.ropeShape, m.step]));
    cache.attention.push(bind(m, m.attention.pipeline, [m.q, cache.keys[l], cache.values[l], m.xb, attentionParams, m.step]));
  }
  m.cache = cache;
}

// T148: what a whole block takes here, from the submission to its keys and values read back, for 16 and for 64 tokens
// (plan.batch) at position 0 of the GPU's own cache (which holds nothing of forward.js's yet): forward.js weighs a
// block of a prompt on the GPU against the same tokens on the CPU by it (a line through the two, scaled by the blocks
// it then times itself). The two in turn, BLOCK_ROUNDS rounds after one of each to warm up (the pipelines' first
// dispatches, the weights' first reads), the median of each (a device that warms up or is loaded meanwhile falls on
// both alike); one round on a fallback adapter (tests: its times are no GPU's)
const BLOCK_ROUNDS = 3;
async function timeBlocks(m) {
  const { batch, seqLen } = m.plan, counts = [...new Set([Math.min(16, batch, seqLen), Math.min(batch, seqLen)])], times = counts.map(() => []);
  const timed = async (count) => {
    const began = performance.now();
    await block(m, count, 0, () => false, true);
    return performance.now() - began;
  };
  for (const count of counts) await timed(count);
  for (let round = 0; round < (m.fallback ? 1 : BLOCK_ROUNDS); round++) {
    for (let i = 0; i < counts.length; i++) {
      times[i].push(await timed(counts[i]));
      if (stopping) return [];
    }
  }
  return counts.map((count, i) => ({ count, ms: times[i].sort((a, b) => a - b)[times[i].length >> 1] }));
}

// ---- T152: tokens on the GPU. A step of the engine's generate() (the forward pass of the token fed, the sampling of
// the next) as one compute pass of shaders.js's T151 run: EMBED of the state's token, every layer in a fused form, the
// final norm and the classifier, SAMPLE (the penalty, softmax, top-p and the draw, with the random number the CPU drew
// for the step), after which the state's first four words go to the Step uniform for the next pass (a uniform is not a
// shader's to write). plan.tokens.most steps a submission, their ids read back once (T151: the wait of a submission,
// 3 to 8.6 ms on the owner's Android, T134, is paid once for them), with the keys and values of their positions, which
// go into forward.js's cache as the CPU would have written them: the CPU may take the next step, or the next
// generation, at any position. The keys and values the GPU does not hold (positions the CPU computed since) go up from
// forward.js's cache first: float16 there as here (T110, T147).
//
// The form of a layer, from those this device can make: T150's fusedMatVec (llama.cpp's mul_mat_vec, with the norm on
// its read and RoPE, the residual's add or SwiGLU on its write) with the workgroup's reduction, the same with
// subgroupAdd where there are subgroups, and T175's fusedDp4aMatVec (ORT's DP4A for small M, the vector quantized
// before each matrix, NORM_QUANTIZE where a norm is) where there is the packed int8 dot. Each is checked against
// JavaScript on the model's own weights (checkTokens), and the right ones are timed, a run of plan.tokens.most steps
// each in turn: the fastest is taken (T150's and T175's tables left which is fastest to the device: the owner's
// Android read a layer at 18.5% of its buffer's reads with mul_mat_vec and a matrix at 96.8% with DP4A). The attention
// (T224, chooseTokenAttention) is llama.cpp's decode form, flash_attn_vec, split over the positions and reduced, or the
// prompt's tiles, whichever is right and faster here, on this pass's q. Where it came from: the run of T151 (public/benchmark/gpu.js's generate()), whose
// form is WebLLM's decode loop without its sync of every token (web-llm, src/llm_chat.ts; no line taken) and llama.cpp's
// WebGPU graph of a token, one command encoder for all of it (ggml-webgpu.cpp, commit 2145525a, MIT; no line taken).
// T226: Qwen2's and Qwen3's steps too. What they have between the matrix of q, k and v and RoPE (T153: a bias of each,
// a norm of every head of q and of k) is the prompt's dispatches, ADD and HEAD_NORM, as the CPU orders them, on the
// three as their matrix wrote them, and then shaders.js's TOKEN_ROPE (the fused write's own lines, as a dispatch):
// two dispatches a layer more with the biases, three with the norms of the heads, none for a model without either.
// (T175's fused DP4A with the norms apart as well, RMSNORM and QUANTIZE where NORM_QUANTIZE is one: the owner's
// Android ran a layer so in 3.36 ms against 3.67 fused, the fastest of its table, 2026-09-27; T175's condition to
// reverse NORM_QUANTIZE, here chosen on the device)
const TOKEN_FORMS = [{ name: "llama.cpp, fused (T150)", dp4a: false, subgroups: false },
  { name: "llama.cpp, fused (T150), subgroups", dp4a: false, subgroups: true },
  { name: "DP4A, fused (T175)", dp4a: true, subgroups: false },
  { name: "DP4A, fused (T175), the norms apart", dp4a: true, subgroups: false, normApart: true },
  // T232: the same two on ternary weights (shaders.js's ternaryMatVec), which a model of ternary weights alone takes,
  // and takes no other
  { name: "DP4A, fused (T175), ternary", dp4a: true, subgroups: false, ternary: true },
  { name: "DP4A, fused (T175), ternary, the norms apart", dp4a: true, subgroups: false, ternary: true, normApart: true }];
function tokenCandidates(m) {
  const features = navigator.gpu.wgslLanguageFeatures;
  const subgroups = m.device.features.has("subgroups") && Boolean(features?.has("subgroup_id"));
  const packed = Boolean(features?.has("packed_4x8_integer_dot_product"));
  // (T226: NORM_QUANTIZE is RMSNorm's: a model with LayerNorm has its norms apart, and the form with them fused is
  // the same as the one with them apart)
  return TOKEN_FORMS.filter((form) => Boolean(form.ternary) === m.ternary && (!form.subgroups || subgroups) && (!form.dp4a || packed) &&
    (!m.plan.layerNorm || !form.dp4a || form.normApart));
}
// T226: what of the model's form shapes a token's dispatches: apart, q, k and v are written as they are for what comes
// before RoPE (the biases, the norms of the heads); gated, the FFN has a gate (SwiGLU on the write of gate and up; else
// one matrix written as it is, then its bias and GELU); floatHead, the classifier multiplies floats on a DP4A form too
// (T92's outlier channels: a few of the final norm's weights are 12 to 17 times the others (GPT-2), and a group of 32
// quantized to 8 bits with one of them loses the other 31; the CPU multiplies their columns apart, and the GPU has
// T150's matrix of floats, which needs no such thing)
// (T232: a ternary classifier is no matrix of floats: its outlier channels are taken apart as the CPU takes them,
// shaders.js's TAKE_OUTLIERS and TERNARY_COLUMNS)
const floatHead = (m) => Boolean(m.plan.tokens.outliers) && !m.ternary;
const tokenShape = (m) => ({ apart: Boolean(m.gen.qkv), gated: Boolean(m.matrices.w3), layerNorm: m.plan.layerNorm, floatHead: floatHead(m) });
// a form's WGSL, [key, code] each (the pipelines every form shares are compiled apart: tokenBuffers); the same code
// under two keys is compiled once (chooseTokens). The norm is on the read of T150's matrices where it is RMSNorm
// (LayerNorm takes the mean out first, which no sum of the matrix's rows gives: a dispatch of its own before them)
const tokenCodes = (wgsl, { dp4a, subgroups, normApart, ternary }, { apart, gated, layerNorm, floatHead }) => {
  const floats = (output) => wgsl.fusedMatVec({ input: layerNorm ? "plain" : "norm", output, subgroups });
  const matrix = dp4a ? (output) => (ternary ? wgsl.ternaryMatVec : wgsl.fusedDp4aMatVec)({ output }) : floats;
  return [...(dp4a && !normApart ? [["normQuantize", wgsl.NORM_QUANTIZE]] : []), ["qkv", matrix(apart ? "write" : "rope")],
    ["add", dp4a ? matrix("add") : wgsl.fusedMatVec({ input: "plain", output: "add", subgroups })], ["glu", matrix(gated ? "swiglu" : "write")],
    ["classifier", (floatHead ? floats : matrix)("write")]];
};

// a bind group of the bindings given ([binding, a buffer or a range of one] each: the fused shaders skip binding 4
// where the norm is not on their read)
const bindAt = (m, pipeline, entries) => m.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: entries.map(([binding, buffer]) => ({ binding, resource: buffer.buffer ? buffer : { buffer } })) });

// The buffers and uniforms of a token, and the pipelines every form shares (EMBED, SAMPLE; QUANTIZE and the attention
// are the prompt's). A token's vectors: the stream h, q, the attention's output, SwiGLU's; the logits and SAMPLE's two
// scratch arrays of the vocabulary; the state, the ids, the random numbers, the Step and the settings; the quantized
// vector (DP4A: one serves every quantization of a token, each read before the next is made: T175's (l)); what a
// submission reads back (the ids, the state, and the keys and values of its positions, [keys, values][layer][step])
async function tokenBuffers(m) {
  const { plan, wgsl } = m, most = plan.tokens.most, vocab = plan.tokens.classifier.rows;
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, widest = Math.max(plan.dim, qDim, plan.hidden);
  // (one at a time: each in an error scope of its own)
  // (T232: from a table of ternary weights, EMBED_TERNARY)
  const embedCode = tablesOf(plan).embedding.ternary ? wgsl.EMBED_TERNARY : wgsl.EMBED;
  const embed = await within(validated(m, () => pipelineOf(m, embedCode)), "compiling the embedding's row");
  const sample = await within(validated(m, () => pipelineOf(m, wgsl.SAMPLE)), "compiling the sampling");
  const out = STORAGE | COPY_SRC;
  const g = { embed, sample, most, vocab, h: buffer(m, plan.dim * 4, out | COPY_DST), q: buffer(m, qDim * 4), att: buffer(m, qDim * 4),
    gate: buffer(m, plan.hidden * 4), logits: buffer(m, vocab * 4, out), probs: buffer(m, vocab * 4), order: buffer(m, vocab * 4),
    state: buffer(m, wgsl.STATE_BYTES, out | COPY_DST), chosen: buffer(m, most * 4, out), randoms: buffer(m, most * 4, STORAGE | COPY_DST),
    step: buffer(m, 16, UNIFORM | COPY_DST), settings: buffer(m, wgsl.SAMPLING_BYTES, UNIFORM | COPY_DST),
    xq: buffer(m, widest), xs: buffer(m, (widest / wgsl.GROUP) * 4), xb: buffer(m, plan.dim * 4),
    readback: buffer(m, most * 4 + wgsl.STATE_BYTES + 2 * plan.layers * most * kvDim * 2, MAP_READ | COPY_DST) };
  // fusedMatVec's Params: rows, words, perRow, second, eps, normAt, qRows, kvRows, headSize, turned
  const params = (rows, n, second = 0, normAt = 0) => {
    const bytes = new ArrayBuffer(48);
    new Uint32Array(bytes).set([rows, n / 4, n / wgsl.GROUP, second, 0, normAt, qDim, kvDim, plan.headSize, plan.turned, 0, 0]);
    new Float32Array(bytes, 16, 1)[0] = plan.eps;
    return uniform(m, bytes);
  };
  // RMSNORM's, HEAD_NORM's and NORM_QUANTIZE's Norm: size, at, eps, first (HEAD_NORM's: the rows before its first)
  const norm = (at, size = plan.dim, first = 0) => {
    const bytes = new ArrayBuffer(16);
    new Uint32Array(bytes).set([size, at, 0, first]);
    new Float32Array(bytes, 8, 1)[0] = plan.eps;
    return uniform(m, bytes);
  };
  const flash = new ArrayBuffer(16);
  new Uint32Array(flash, 0, 2).set([plan.heads, plan.kvHeads]);
  new Float32Array(flash, 8, 1)[0] = 1 / Math.sqrt(plan.headSize);
  const layers = [...Array(plan.layers)].map((_, l) => l * plan.dim), qkvRows = qDim + 2 * kvDim;
  g.u = { embed: m.tables.embedding.map((piece) => uniform(m, new Uint32Array([plan.dim, piece.first, piece.rows, 0]))), flash: uniform(m, flash), o: params(plan.dim, qDim), down: params(plan.dim, plan.hidden),
    qkv: layers.map((at) => params(qkvRows, plan.dim, 0, at)), gateUp: layers.map((at) => params(plan.hidden, plan.dim, plan.hidden, at)),
    norm: layers.map((at) => norm(at)), final: norm(0), classifier: m.tables.classifier.map((piece) => params(piece.rows, plan.dim)),
    // QUANTIZE's (n, xStride): the attention's output, SwiGLU's
    quantizeAttention: uniform(m, new Uint32Array([qDim, qDim, 0, 0])), quantizeGate: uniform(m, new Uint32Array([plan.hidden, plan.hidden, 0, 0])),
    quantizeNormed: uniform(m, new Uint32Array([plan.dim, plan.dim, 0, 0])) };
  // T226: a model whose q, k and v are not ready for RoPE as their matrix leaves them (T153: Qwen2's biases, Qwen3's
  // norms of the heads) has the matrix write the three as they are into qkv (q's rows, then k's, then v's), the
  // prompt's ADD and HEAD_NORM change them there, and TOKEN_ROPE turns them into q and the cache (tokenPass). The
  // biases of a layer are one vector then, in the rows' order (one ADD for the three); k's heads are the rows after q's
  const { bq, bk, bv, qNorm } = plan.vectors;
  if (bq || qNorm) {
    g.rope = await within(validated(m, () => pipelineOf(m, wgsl.TOKEN_ROPE)), "compiling a token's RoPE");
    g.qkv = buffer(m, qkvRows * 4);
  }
  if (bq) {
    g.qkvBias = buffer(m, plan.layers * qkvRows * 4, STORAGE | COPY_DST);
    for (let l = 0; l < plan.layers; l++) {
      let at = l * qkvRows * 4;
      for (const bias of [bq, bk, bv]) {
        copyIn(m, g.qkvBias, bias.at + l * bias.size * 4, bias.size * 4, at);
        at += bias.size * 4;
      }
    }
    g.u.qkvBias = layers.map((_, l) => uniform(m, new Uint32Array([qkvRows, l * qkvRows, 0, 0])));
  }
  if (qNorm) g.u.headNorms = layers.map((_, l) => [0, plan.heads].map((first) => norm(l * plan.headSize, plan.headSize, first)));
  // T226, GPT-2's and GPT-NeoX's (T154's dispatches of a prompt's block, on a token's vectors): ADD's shapes of the
  // biases after o, w1 and w2 (a layer's from its place in the vector) and of GPT-2's positions (a row a position), and
  // GELU's size
  const { bo, b1, b2 } = plan.vectors;
  if (bo) g.u.biases = layers.map((_, l) => Object.fromEntries([["bo", bo], ["b1", b1], ["b2", b2]].map(([name, { size }]) =>
    [name, uniform(m, new Uint32Array([size, l * size, 0, 0]))])));
  if (m.positions) g.u.positions = uniform(m, new Uint32Array([plan.dim, 0, plan.dim, 0]));
  // T232: a ternary classifier's outlier channels (plan.tokens.channels, T92): the two dispatches around its matrix
  // (shaders.js's TAKE_OUTLIERS and TERNARY_COLUMNS), the values taken, and the Outliers of every piece of the table
  const channels = m.ternary ? plan.tokens.channels ?? [] : [];
  if (channels.length) {
    g.take = await within(validated(m, () => pipelineOf(m, wgsl.TAKE_OUTLIERS)), "compiling the outlier channels' taking");
    g.columns = await within(validated(m, () => pipelineOf(m, wgsl.TERNARY_COLUMNS)), "compiling the outlier channels' columns");
    g.picked = buffer(m, wgsl.OUTLIERS_MOST * 4);
    g.u.take = uniform(m, wgsl.outliersOf(channels));
    g.u.columns = m.tables.classifier.map((piece) => uniform(m, wgsl.outliersOf(channels, piece.rows, plan.dim)));
  }
  g.u.hidden = uniform(m, new Uint32Array([plan.hidden, 0, 0, 0]));
  return g;
}

// the workgroups of rows by the fused shaders' rows a workgroup, over x and then y (they number them so)
function spread(m, rows, perGroup) {
  const groups = Math.ceil(rows / perGroup), across = Math.min(groups, m.device.limits.maxComputeWorkgroupsPerDimension);
  return [across, Math.ceil(groups / across)];
}
// The dispatches of one step in a form: EMBED, the layers (from, to: those of a check), the head (the final norm and
// the classifier) and SAMPLE, [pipeline, bind group, x, y] each. positions: how many the attention reads at the most
// (T224: flash_attn_vec's parts a head go by it)
function tokenPass(m, form, { from = 0, to = m.plan.layers, head = true, embed = true, positions = m.cache.capacity } = {}) {
  const { plan, wgsl, gen: g } = m, V = m.vectors, P = form.pipes, cache = m.cache, qDim = plan.heads * plan.headSize;
  const rows = form.dp4a ? wgsl.ORT_DP4A_MATVEC_ROWS : wgsl.MUL_MAT_VEC_ROWS;
  const matrix = (pipeline, [w, s], input, params, count, output) =>
    [pipeline, bindAt(m, pipeline, [[0, w], [1, s], ...input, [3, params], ...output]), ...spread(m, count, rows)];
  const quantize = (x, params, n) => [m.quantize, bindAt(m, m.quantize, [[0, x], [1, g.xq], [2, g.xs], [3, params], [4, g.step]]), Math.ceil(n / wgsl.GROUP / 64), 1];
  // What a matrix reads: { norm, quantize, input }, the dispatches before it (the stream's norm where it is one of its
  // own, the quantizing of DP4A's vector: T175) and its bindings of the vector.
  // The stream normed, for a matrix of floats (T150's; T226: the classifier of a model with outlier channels on DP4A
  // too) or of DP4A: RMSNorm on the read of T150's matrices, or with DP4A's quantizer (NORM_QUANTIZE); else a dispatch
  // of its own into xb (the prompt's RMSNORM, T175's form with the norms apart; T226: the prompt's LAYER_NORM with its
  // bias, T154), which the matrix reads as it is or quantized
  // (T232, taken: what changes the normed stream before it is quantized, a ternary classifier's outlier channels
  // taken out of it: the norm is then a dispatch of its own whatever the form)
  const normed = (weights, bias, params, floats = !form.dp4a, taken = null) => {
    const apart = () => [[m.norm, bind(m, m.norm, plan.layerNorm ? [g.h, weights, bias, g.xb, params, g.step] : [g.h, weights, g.xb, params, g.step]), 1, 1], ...(taken ?? [])];
    if (floats) return plan.layerNorm ? { norm: apart(), quantize: [], input: [[2, g.xb]] } : { norm: [], quantize: [], input: [[2, g.h], [4, weights]] };
    if (!form.normApart && !plan.layerNorm && !taken) {
      return { norm: [[P.normQuantize, bindAt(m, P.normQuantize, [[0, g.h], [1, weights], [2, g.xq], [3, g.xs], [4, params], [5, g.step]]), 1, 1]],
        quantize: [], input: [[2, g.xq], [4, g.xs]] };
    }
    return { norm: apart(), quantize: [quantize(g.xb, g.u.quantizeNormed, plan.dim)], input: [[2, g.xq], [4, g.xs]] };
  };
  // a vector as it is (the attention's output, the activation's)
  const plain = (x, params, n) => (form.dp4a ? { quantize: [quantize(x, params, n)], input: [[2, g.xq], [4, g.xs]] } : { quantize: [], input: [[2, x]] });
  // T226: a bias added after a matrix (ADD: Qwen2's of q, k and v as one vector; T154's after o, w1 and w2)
  const added = (to, bias, shape, n) => [m.add, bind(m, m.add, [to, bias, shape, g.step]), Math.ceil(n / 64), 1];
  const biased = (to, name, l) => (V[name] ? [added(to, V[name], g.u.biases[l][name], plan.vectors[name].size)] : []);
  // (T209: a dispatch a piece of the table, each writing the row where the token is in its rows)
  const list = !embed ? [] : m.tables.embedding.map((piece, i) =>
    [g.embed, bindAt(m, g.embed, [[0, piece.values], [1, piece.scales], [2, g.state], [3, g.h], [4, g.u.embed[i]]]), 1, 1]);
  // T226: GPT-2's learned positions: the row of the Step's position added to the embedding's (the CPU's embed())
  if (embed && m.positions) list.push(added(g.h, m.positions, g.u.positions, plan.dim));
  const qkvRows = qDim + 2 * plan.kvHeads * plan.headSize, gated = Boolean(m.matrices.w3);
  for (let l = from; l < to; l++) {
    const [o] = m.matrices.wo.pieces, [down] = m.matrices.w2.pieces;
    const attention = normed(V.attention, V.attentionBias, g.u.norm[l]), ffn = normed(V.ffn, V.ffnBias, g.u.norm[l]);
    const attended = plain(g.att, g.u.quantizeAttention, qDim), activated = plain(g.gate, g.u.quantizeGate, plan.hidden);
    // where q, k and v go turned: q, and the cache at the Step's position
    const turned = [[5, g.q], [6, cache.keys[l]], [7, cache.values[l]], [8, m.angleTable], [9, g.step]];
    list.push(...attention.norm, ...attention.quantize,
      // T226: turned on the matrix's write (Llama: one dispatch), or written as they are where something comes before
      // RoPE: as the CPU has it and a prompt's block (T153), the biases, the norms of the heads of q and of k, RoPE
      matrix(P.qkv, m.joined[l].qkv, attention.input, g.u.qkv[l], qkvRows, g.qkv ? [[5, g.qkv]] : turned),
      ...(g.qkvBias ? [added(g.qkv, g.qkvBias, g.u.qkvBias[l], qkvRows)] : []),
      ...(g.u.headNorms ? [[V.qNorm, plan.heads], [V.kNorm, plan.kvHeads]].map(([weights, heads], i) =>
        [m.headNorm, bind(m, m.headNorm, [g.qkv, weights, g.u.headNorms[l][i], g.step]), heads, 1]) : []),
      ...(g.qkv ? [[g.rope, bindAt(m, g.rope, [[2, g.qkv], [3, g.u.qkv[l]], ...turned]), 1, 1]] : []),
      // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, before o adds to it (into xb, which q,
      // k and v have read by now; a norm on the matrix's read would come too late: chooseTokens refuses the pair)
      ...(plan.parallel ? ffn.norm : []),
      ...attentionPasses(m, g.attention, { q: g.q, keys: cache.keys[l], values: cache.values[l], out: g.att, parts: g.parts, params: g.params,
        flash: g.u.flash, step: g.step }, plan.heads, positions),
      ...attended.quantize, matrix(P.add, o.layers[l], attended.input, g.u.o, plan.dim, [[5, g.h]]), ...biased(g.h, "bo", l),
      ...(plan.parallel ? [] : ffn.norm), ...ffn.quantize,
      // gate and up as one matrix with SwiGLU on its write; or (T154: no gate) w1, its bias and GELU
      ...(gated ? [matrix(P.glu, m.joined[l].gateUp, ffn.input, g.u.gateUp[l], plan.hidden, [[5, g.gate]])]
        : [matrix(P.glu, m.matrices.w1.pieces[0].layers[l], ffn.input, g.u.gateUp[l], plan.hidden, [[5, g.gate]]), ...biased(g.gate, "b1", l),
          [m.activation, bind(m, m.activation, [g.gate, g.u.hidden, g.step]), Math.ceil(plan.hidden / 64), 1]]),
      ...activated.quantize, matrix(P.add, down.layers[l], activated.input, g.u.down, plan.dim, [[5, g.h]]), ...biased(g.h, "b2", l));
  }
  if (head) {
    // T232: a ternary classifier's outlier channels go out of the normed stream before it is quantized (TAKE_OUTLIERS),
    // and their columns are added to the piece's logits after its matrix (TERNARY_COLUMNS), as the CPU has them
    const take = g.take ? [[g.take, bind(m, g.take, [g.xb, g.picked, g.u.take]), 1, 1]] : null;
    const final = normed(m.finalNorm, m.finalBias, g.u.final, !form.dp4a || floatHead(m), take);
    const logitsOf = (piece) => ({ buffer: g.logits, offset: piece.first * 4, size: piece.rows * 4 });
    list.push(...final.norm, ...final.quantize,
      // (T209: a piece at a time into its range of the logits: its first row is where the device binds, piecesOf)
      ...m.tables.classifier.flatMap((piece, i) => [matrix(P.classifier, [piece.values, piece.scales], final.input, g.u.classifier[i], piece.rows, [[5, logitsOf(piece)]]),
        ...(take ? [[g.columns, bind(m, g.columns, [piece.values, piece.scales, g.picked, logitsOf(piece), g.u.columns[i]]), ...spread(m, piece.rows, 256)]] : [])]),
      [g.sample, bindAt(m, g.sample, [[0, g.logits], [1, g.probs], [2, g.order], [3, g.state], [4, g.chosen], [5, g.randoms], [6, g.settings]]), 1, 1]);
  }
  return list;
}
// a form's whole step for a run that reads up to positions, made again when the cache has grown (its buffers are others
// then); T224: one a count of the attention's parts a head (flash_attn_vec's nwg goes by the positions)
function tokenStep(m, form, positions) {
  const a = m.gen.attention, parts = a.tiles ? 0 : m.wgsl.flashVecSplits(a.shape, positions);
  if (form.steps?.cache !== m.cache) form.steps = { cache: m.cache, lists: new Map() };
  if (!form.steps.lists.has(parts)) form.steps.lists.set(parts, tokenPass(m, form, { positions }));
  return form.steps.lists.get(parts);
}
// count steps of the dispatches from the state given (with the settings and a random number a step), in one
// submission, and read back: the ids and the state's words, and (keep) the keys and values of the positions pos to
// pos + count - 1. extra(encoder): what a check copies out besides. Returns { ids, sampled, stopped, notFinite (T219:
// the sampler refused the step after the sampled ones: its logits were not finite), kv: [keys, values][layer], the
// positions' rows of float16 as bytes }
async function runTokens(m, dispatches, { count, pos, state, settings, randoms, keep = false, extra }) {
  const { device, plan, wgsl, gen: g } = m, kvRow = plan.kvHeads * plan.headSize * 2, idsBytes = g.most * 4;
  if (pos + count > m.cache.capacity) throw new Error(`positions to ${pos + count} are past the GPU's cache`);
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  device.queue.writeBuffer(g.state, 0, state);
  device.queue.writeBuffer(g.step, 0, state, 0, 4);
  if (settings) device.queue.writeBuffer(g.settings, 0, settings);
  device.queue.writeBuffer(g.randoms, 0, randoms ?? new Float32Array(count));
  const encoder = device.createCommandEncoder();
  for (let i = 0; i < count; i++) {
    const pass = encoder.beginComputePass();
    for (const [pipeline, group, x, y] of dispatches) dispatch(pass, pipeline, group, x, y);
    pass.end();
    // the Step of the next pass: the state's first four words
    encoder.copyBufferToBuffer(g.state, 0, g.step, 0, 16);
  }
  encoder.copyBufferToBuffer(g.chosen, 0, g.readback, 0, count * 4);
  encoder.copyBufferToBuffer(g.state, 0, g.readback, idsBytes, wgsl.STATE_BYTES);
  const kvAt = idsBytes + wgsl.STATE_BYTES;
  if (keep) {
    for (let l = 0; l < plan.layers; l++) {
      encoder.copyBufferToBuffer(m.cache.keys[l], pos * kvRow, g.readback, kvAt + l * g.most * kvRow, count * kvRow);
      encoder.copyBufferToBuffer(m.cache.values[l], pos * kvRow, g.readback, kvAt + (plan.layers + l) * g.most * kvRow, count * kvRow);
    }
  }
  extra?.(encoder);
  device.queue.submit([encoder.finish()]);
  const invalid = await device.popErrorScope(), full = await device.popErrorScope();
  if (invalid || full) throw new Error(`the GPU refused a token (${(invalid ?? full).message})`);
  await g.readback.mapAsync(MAP_READ);
  try {
    const words = new Uint32Array(g.readback.getMappedRange(0, idsBytes + wgsl.STATE_BYTES).slice(0));
    const after = words.subarray(g.most);
    const out = { ids: words.slice(0, count), sampled: after[5], stopped: after[7], notFinite: after[wgsl.STATE_NOT_FINITE] };
    if (keep) {
      const bytes = new Uint8Array(g.readback.getMappedRange(kvAt, 2 * plan.layers * g.most * kvRow).slice(0));
      out.kv = [0, 1].map((side) => [...Array(plan.layers)].map((_, l) => {
        const at = (side * plan.layers + l) * g.most * kvRow;
        return bytes.subarray(at, at + count * kvRow);
      }));
    }
    return out;
  } finally {
    g.readback.unmap();
  }
}

// The forms of a token this device can make, compiled, checked and (where more than one is right) timed; the fastest
// taken (m.gen.form), with what each came to (m.gen.forms) and its ms a step (m.gen.ms). plan.remembered.tokens: the
// one the page kept for this adapter, alone where it is still right here. plan.force.tokens (tests): that form alone;
// plan.force.quick: the first right one, untimed. Throws where none is right (the tokens then stay on the CPU)
async function chooseTokens(m) {
  const { plan, wgsl } = m;
  // (tokenPass puts a parallel residual's second norm before o as a dispatch: LayerNorm's, GPT-NeoX's)
  if (plan.parallel && !plan.layerNorm) throw new Error("a parallel residual without LayerNorm is not on the GPU's tokens");
  m.gen = await tokenBuffers(m);
  await chooseTokenAttention(m);
  if (stopping) return;
  // T156: the first layer's matrices of a model on the GPU alone are not in the shared memory: checkTokens reads them
  // back from the GPU (T210: and the rows of the tables it reads)
  if (m.direct) {
    m.firstLayer ??= await within(firstLayer(m), "reading the first layer back");
    m.tableRows ??= await within(tableRows(m), "reading the tables' rows back");
  }
  let forms = tokenCandidates(m);
  if (plan.force.tokens) forms = forms.filter((form) => form.name === plan.force.tokens);
  const kept = !plan.force.tokens && forms.find((form) => form.name === m.remembered?.tokens);
  if (kept) forms = [kept, ...forms.filter((form) => form !== kept)];
  const right = [], tried = [];
  for (const candidate of forms) {
    // the remembered one right: no other is compiled or timed (as the matrices, T148)
    if ((plan.force.quick || (kept && right[0]?.name === kept.name)) && right.length) break;
    const form = { ...candidate, pipes: {} }, compiled = new Map();
    try {
      for (const [key, code] of tokenCodes(wgsl, form, tokenShape(m))) {
        if (!compiled.has(code)) compiled.set(code, await within(validated(m, () => pipelineOf(m, code)), `compiling ${form.name}`));
        form.pipes[key] = compiled.get(code);
      }
      const wrong = await within(checkTokens(m, form), `checking ${form.name}`);
      if (wrong) tried.push({ name: form.name, none: `wrong: ${wrong}` });
      else right.push(form);
    } catch (error) {
      if (error?.late) throw error;
      tried.push({ name: form.name, none: String(error?.message ?? error) });
    }
    if (stopping) return;
  }
  if (!right.length) {
    throw new Error(`no layer of a token is right on this GPU (${tried.map((f) => `${f.name}: ${f.none}`).join("; ") || `none named ${plan.force.tokens}`})`);
  }
  const ms = plan.force.quick ? right.map(() => undefined) : await within(timeTokens(m, right), "timing a token");
  const best = ms.reduce((b, t, i) => (t !== undefined && (ms[b] === undefined || t < ms[b]) ? i : b), 0);
  m.gen.form = right[best];
  m.gen.ms = ms[best];
  m.gen.forms = [...tried, ...right.map((form, i) => ({ name: form.name, ms: ms[i], ...(kept && form.name === kept.name ? { remembered: true } : {}) }))];
}

// ms a step of each form: runs of plan.tokens.most steps a submission from position 0 (a made-up state, the settings
// of the list's sampled models, T151's), the forms in turn after one run each to warm up, BLOCK_ROUNDS rounds (one on
// a fallback adapter: its times are no GPU's), the median of each over its steps. The runs write the GPU's own keys
// and values of those positions, which hold nothing of forward.js's yet (as timeBlocks)
async function timeTokens(m, forms) {
  const { wgsl, gen: g } = m, most = g.most;
  const state = wgsl.samplingState({ token: 1, pos: 0, history: [1] });
  const settings = wgsl.samplingSettings({ vocab: g.vocab, temperature: 0.7, topp: 0.9, penalty: 1.1 });
  const randoms = new Float32Array(most).map(() => Math.fround(Math.random()) % 1);
  if (most > m.cache.capacity) grow(m, most);
  const timed = async (form) => {
    const began = performance.now();
    await runTokens(m, tokenStep(m, form, most), { count: most, pos: 0, state, settings, randoms });
    return performance.now() - began;
  };
  for (const form of forms) await timed(form);
  const times = forms.map(() => []);
  for (let round = 0; round < (m.fallback ? 1 : BLOCK_ROUNDS); round++) {
    for (let i = 0; i < forms.length; i++) times[i].push(await timed(forms[i]));
    if (stopping) break;
  }
  return times.map((list) => list.sort((a, b) => a - b)[list.length >> 1] / most);
}

// ---- T224: the attention of a generated token. The candidates: llama.cpp's decode form (shaders.js's flashVec and
// flashVecReduce: the positions split over nwg workgroups a head, then reduced), with subgroups where there are (and
// subgroup_id), and with the lanes of the workgroup standing for a subgroup; and the prompt's tiles (m.attention: one
// row of four used, a workgroup a head, T150's (a)). Each checked against JavaScript (checkTokenAttention), the right
// ones timed (timeTokenAttention), the fastest taken: m.gen.attention, with what each came to (m.gen.attentions).
// plan.force.tokenAttention (tests): that one alone; plan.force.quick: the first right one, untimed. None is
// remembered (T148 remembers the matrices, the prompt's attention and the token's layer): the two small shaders are
// compiled, checked and timed at every start.
const TOKEN_ATTENTION_VEC = "llama.cpp flash_attn_vec", TOKEN_ATTENTION_TILES = "the prompt's attention tiles";
async function chooseTokenAttention(m) {
  const { device, plan, wgsl, gen: g } = m;
  const subgroups = device.features.has("subgroups") && Boolean(navigator.gpu.wgslLanguageFeatures?.has("subgroup_id"));
  const vec = (withSubgroups) => ({ name: `${TOKEN_ATTENTION_VEC}${withSubgroups ? ", subgroups" : ""}`,
    shape: wgsl.flashVecShape({ headSize: plan.headSize, subgroups: withSubgroups, threads: threadsOf(device),
      subgroupMin: m.info.subgroupMinSize, subgroupMax: m.info.subgroupMaxSize }) });
  let candidates = [...(subgroups ? [vec(true)] : []), vec(false), { name: TOKEN_ATTENTION_TILES, tiles: true, pipeline: m.attention.pipeline }];
  if (plan.force.tokenAttention) candidates = candidates.filter((a) => a.name === plan.force.tokenAttention);
  const right = [];
  g.attentions = [];
  for (const a of candidates) {
    if (plan.force.quick && right.length) break;
    if (a.shape?.none) {
      g.attentions.push({ name: a.name, none: a.shape.none });
      continue;
    }
    try {
      if (!a.tiles) {
        a.pipeline = await within(validated(m, () => pipelineOf(m, wgsl.flashVec(a.shape))), `compiling ${a.name}`);
        a.reduce = await within(validated(m, () => pipelineOf(m, wgsl.flashVecReduce(a.shape))), `compiling ${a.name}'s reduce`);
      }
      const wrong = await within(checkTokenAttention(m, a), `checking ${a.name}`);
      if (wrong) g.attentions.push({ name: a.name, none: `wrong: ${wrong}` });
      else right.push(a);
    } catch (error) {
      if (error?.late) throw error;
      g.attentions.push({ name: a.name, none: String(error?.message ?? error) });
    }
    if (stopping) return;
  }
  if (!right.length) {
    throw new Error(`no attention of a token is right on this GPU (${g.attentions.map((a) => `${a.name}: ${a.none}`).join("; ") || `none named ${plan.force.tokenAttention}`})`);
  }
  const ms = plan.force.quick || right.length === 1 ? right.map(() => undefined) : await within(timeTokenAttention(m, right), "timing the attention of a token");
  const best = ms.reduce((b, t, i) => (t !== undefined && (ms[b] === undefined || t < ms[b]) ? i : b), 0);
  right.forEach((a, i) => g.attentions.push({ name: a.name, ms: ms[i] }));
  g.attention = right[best];
  // the vec form's parts (every head's, nwg of the most) and its Params a count of parts, made as a run needs them
  if (!g.attention.tiles) {
    g.parts = buffer(m, wgsl.flashVecPartsBytes(g.attention.shape, plan.heads));
    g.params = paramsOf(m, g.attention.shape, plan.heads, plan.kvHeads, m.owned);
  }
}
// flashVec's Params a count of parts, each made once (owned: where they go)
function paramsOf(m, shape, heads, kvHeads, owned) {
  const made = new Map();
  return (nwg) => {
    if (!made.has(nwg)) made.set(nwg, uniform(m, m.wgsl.flashVecParams(shape, heads, kvHeads, nwg), owned));
    return made.get(nwg);
  };
}
// The dispatches of attention a for a token of heads that reads positions: io's q, keys, values into out (flash: the
// tiles' Params; parts and params(nwg): the vec form's parts and Params; step: the token's Step), [pipeline, bind
// group, x, y] each: the tiles a workgroup a head; the vec form nwg a head, then (nwg more than 1) the reduce
function attentionPasses(m, a, io, heads, positions) {
  if (a.tiles) return [[a.pipeline, bind(m, a.pipeline, [io.q, io.keys, io.values, io.out, io.flash, io.step]), heads, 1]];
  const nwg = m.wgsl.flashVecSplits(a.shape, positions), params = io.params(nwg);
  return [[a.pipeline, bind(m, a.pipeline, [io.q, io.keys, io.values, io.parts, io.out, params, io.step]), heads * nwg, 1],
    ...(nwg > 1 ? [[a.reduce, bind(m, a.reduce, [io.parts, io.out, params]), heads, 1]] : [])];
}
// Made-up buffers of an attention of a token over positions (heads of q on kvHeads of keys and values, the model's
// headSize; shaders.js's tokenAttentionData: steep, the heads whose q is steep), the output, the Step of the token at
// positions - 1, the tiles' Params, and the vec form's parts and Params
function attentionIo(m, a, { heads, kvHeads, positions, steep = [] }, owned) {
  const size = m.plan.headSize, { q, keys, values } = m.wgsl.tokenAttentionData({ heads, kvHeads, size, positions, steep });
  const make = (data) => {
    const made = buffer(m, data.byteLength, STORAGE | COPY_DST, owned);
    m.device.queue.writeBuffer(made, 0, data);
    return made;
  };
  const flash = new ArrayBuffer(16);
  new Uint32Array(flash, 0, 2).set([heads, kvHeads]);
  new Float32Array(flash, 8, 1)[0] = 1 / Math.sqrt(size);
  return { data: { q, keys, values }, q: make(q), keys: make(keys), values: make(values), out: buffer(m, heads * size * 4, STORAGE | COPY_SRC, owned),
    step: uniform(m, new Uint32Array([1, positions - 1, 0, 0]), owned), flash: uniform(m, flash, owned),
    ...(a.tiles ? {} : { parts: buffer(m, m.wgsl.flashVecPartsBytes(a.shape, heads), STORAGE, owned), params: paramsOf(m, a.shape, heads, kvHeads, owned) }) };
}
// The check of an attention of a token, on made-up numbers against JavaScript's (shaders.js's tokenAttentionData and
// tokenAttentionOff): 4 heads of q on 2 of keys and values (each head of q to its own), a token that reads 40 positions
// (a KV_TILE of 32 and a part of the next: one part), 70 (two parts), 300 and 1100 (as many parts as the vec form takes,
// each of more than one tile: the reduce over them), with positions past the token's that it must not read, and head
// 3's q steep (a largest taken wrong shows only in a steep softmax). Each head's output against its softmax over the
// positions up to the token's, no farther than LINE of the largest |value| of its head (checkAttention's: the tiles
// hold the weights in float16)
const TOKEN_ATTENTION_LENGTHS = [40, 70, 300, 1100];
async function checkTokenAttention(m, a) {
  const size = m.plan.headSize, heads = 4, kvHeads = 2;
  for (const positions of TOKEN_ATTENTION_LENGTHS) {
    const owned = [];
    try {
      const io = attentionIo(m, a, { heads, kvHeads, positions, steep: [3] }, owned);
      const passes = await validated(m, () => attentionPasses(m, a, io, heads, positions));
      const encoder = m.device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (const [pipeline, group, x, y] of passes) dispatch(pass, pipeline, group, x, y);
      pass.end();
      m.device.queue.submit([encoder.finish()]);
      const got = new Float32Array(await readBack(m, io.out, heads * size * 4));
      const worst = m.wgsl.tokenAttentionOff(got, io.data, { heads, kvHeads, size, positions });
      if (!(worst <= LINE)) return `its output is ${worst.toExponential(2)} of the largest value from JavaScript's at ${positions} positions (line ${LINE})`;
    } finally {
      owned.forEach((b) => b.destroy());
    }
  }
  return null;
}
// ms of the attention of a token (every layer's) with each of right, on made-up numbers: at 128 positions and at 2048
// (the model's context where shorter), all in turn, as the tiled shaders are timed (timeForms: a submission of n
// attentions and one of 2n, n doubled from 1 until n takes TIMED_MS, up to MOST_PASSES layers of them, PAIRS pairs,
// the median), times the layers; the two lengths' ms added. (T224's review: n counts attentions, not layers of them.
// A submission of a whole token's layers at the least made the slowest one long: the prompt's tiles at 2048
// positions, if they take the time of T202's 0.90 ms at 127 positions in proportion (an estimate: the f32 tiles, on
// the owner's Android), 16 × 14 ms a submission and some 3.9 s of timing at every start, 0.5 s so)
async function timeTokenAttention(m, right) {
  const { device, plan } = m, owned = [];
  try {
    const lengths = [...new Set([128, 2048].map((n) => Math.min(n, plan.seqLen)))];
    const items = right.flatMap((a) => lengths.map((positions) => {
      const io = attentionIo(m, a, { heads: plan.heads, kvHeads: plan.kvHeads, positions }, owned);
      return { a, passes: attentionPasses(m, a, io, plan.heads, positions) };
    }));
    const submission = async ({ passes }, n) => {
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let i = 0; i < n; i++) for (const [pipeline, group, x, y] of passes) dispatch(pass, pipeline, group, x, y);
      pass.end();
      const began = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    const counts = [], differences = items.map(() => []);
    for (const item of items) {
      await submission(item, 1);  // warm
      let n = 1;
      while (!m.fallback && n < MOST_PASSES * plan.layers && (await submission(item, n)) < TIMED_MS) n *= 2;
      counts.push(n);
    }
    for (let round = 0; round < (m.fallback ? 1 : PAIRS); round++) {
      for (const [i, item] of items.entries()) {
        const once = await submission(item, counts[i]), twice = await submission(item, 2 * counts[i]);
        differences[i].push((twice - once) / counts[i]);
      }
      if (stopping) break;
    }
    const ms = differences.map((d) => d.sort((x, y) => x - y)[d.length >> 1] * plan.layers);
    return right.map((a) => items.reduce((sum, item, i) => (item.a === a ? sum + ms[i] : sum), 0));
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

// ---- T152's check of a form on the model's own weights, against JavaScript (the drivers, the subgroups and the
// packed dot differ from device to device, and only the device can say):
//   1. EMBED of a token of the second half of the vocabulary and the first layer at position 1 (the cache's row 0 of random
//      float16), the stream h and the keys and values of position 1 read back: against JavaScript's layer in float64
//      (the weights read from the shared memory as forward.js holds them: int6 widened by shaders.js's sixValues; RoPE
//      from the CPU's tables; the keys and values rounded to float16 where the attention reads them (T225's review: the
//      device's own float16 where it is a neighbour of the float64 value, heldFloats; else the nearest); on DP4A each
//      matrix's input quantized as quantize_x quantizes it, in float32). The stream: no farther than LAYER_LINE of the
//      largest change the layer made; the keys and values no farther than it of their largest. A wrong row, group,
//      head, angle or scale is a tenth and more off; on DP4A a value quantized to the other side of a rounding moves
//      the layer by about 1e-3 (T175), and so its lines are DP4A_LINE.
//   2. the head on a made-up stream (±2), greedy: the logits of every 256th row (and the last) against JavaScript's,
//      no farther than LOGITS_LINE of their largest (DP4A: DP4A_LINE), and the id SAMPLE chose the first largest of the
//      GPU's own logits; then sampled (temperature 0.8, top-p 0.9, penalty 1.3 on a history that holds that id, a
//      random number of 0.7): the logits SAMPLE penalized in place as penalizeLikeCpu penalizes the greedy ones
//      (within 1e-6), and the id one that the CPU's walk (walkLikeCpu) over those logits reaches within 1e-4 of the
//      random number's share of the mass (T151's line: the GPU's exp and its float32 sums).
// The reason it is wrong, or null.
const LAYER_LINE = 2e-3, LOGITS_LINE = 1e-3, DP4A_LINE = 2e-2;
// T225's review: WGSL leaves it to the device which of its two float16 neighbours a float32 becomes (§15.7.6 Floating
// Point Conversion: "WGSL does not specify whether the higher or lower representable value is chosen, and different
// instances of such a conversion may choose differently"; pack2x16float is such a conversion), and Direct3D, where
// Chrome on Windows runs WebGPU, converts toward zero (D3D11.3 functional specification 3.2.2: "Round-to-zero must be
// used during conversion to another float format"; Dawn's HLSL writer makes pack2x16float of f32tof16). The position's
// keys and values the GPU writes are then 1 float16 spacing from the nearest's on about half of their numbers, and the
// stream computed from the nearest's was up to 3.2 times the line off the GPU's (CI, Dawn on lavapipe with every
// conversion cut toward zero: a llama.cpp form on llm-jp-3 150M 6.5e-3 at the line 2e-3; with the GPU's own keys and
// values 2.3e-6, as public/benchmark/gpu.js's heldHalves takes them: T225). So the reference takes the device's own
// float16 (got, as floats) of a value where it is within a float16 spacing of the value and the float32 sums' noise
// (HALF_SLACK of the largest: about 10 times a sum of 2112 products's), and the nearest where it is not: a wrong key
// or value is as far from the line as before, and the keys' and values' lines below are unchanged.
const HALF_SLACK = 1e-5;
function heldFloats(x, got) {
  const round16 = Math.f16round ?? ((value) => value), slack = HALF_SLACK * x.reduce((top, value) => Math.max(top, Math.abs(value)), 0);
  return x.map((value, i) => (Math.abs(got[i] - value) <= 2 ** (Math.max(Math.floor(Math.log2(Math.abs(value))), -14) - 10) + slack ? got[i] : round16(value)));
}
export { heldFloats };  // (tests/gpu-choice-check.mjs)
// T156: the first layer's matrices, { name: [values, scales] } as the bytes of their rows (rowBytes: int8, or T232
// ternary codes) and a Float32Array each, read back from their buffers (a piece after another: the rows in order)
async function firstLayer(m) {
  const group = m.wgsl.GROUP, out = {};
  for (const [name, matrix] of Object.entries(m.matrices)) {
    const row = rowBytes(matrix), rows = matrix.pieces.reduce((sum, piece) => sum + piece.rows, 0);
    const values = new Uint8Array(rows * row), scales = new Float32Array((rows * row) / group);
    for (const piece of matrix.pieces) {
      const [v, s] = piece.layers[0], valueBytes = piece.rows * row, scaleBytes = (valueBytes / group) * 4;
      values.set(new Uint8Array(await readBack(m, v.buffer ?? v, valueBytes, v.offset ?? 0)), piece.first * row);
      scales.set(new Float32Array(await readBack(m, s.buffer ?? s, scaleBytes, s.offset ?? 0)), (piece.first * row) / group);
    }
    out[name] = [values, scales];
  }
  return out;
}
// the tokens whose rows of the embedding the first layer's check may take (of the second half of the vocabulary: see
// below), and the rows of the classifier it holds to JavaScript's (every 256th, and the last)
const checkCandidates = (vocab) => [...Array(64).keys()].map((i) => vocab - 1 - Math.floor((i * vocab) / 128));
const checkRows = (vocab) => [...new Set([...[...Array(Math.ceil(vocab / 256)).keys()].map((i) => i * 256), vocab - 1])];
// T210: those rows of the tables of a model on the GPU alone, which are there only: { embedding, classifier }, Maps of
// a row to [the bytes of its values (rowBytes: int8, or T232 ternary codes), its scales], read back in one copy each
// (a row of a piece at a time)
async function tableRows(m) {
  const vocab = m.plan.tokens.classifier.rows, specs = tablesOf(m.plan);
  const read = async (pieces, spec, rows) => {
    const n = rowBytes(spec), perRow = (n / m.wgsl.GROUP) * 4;
    const target = m.device.createBuffer({ size: rows.length * (n + perRow), usage: MAP_READ | COPY_DST });
    try {
      const encoder = m.device.createCommandEncoder();
      rows.forEach((r, j) => {
        const piece = pieces.find((p) => r >= p.first && r < p.first + p.rows);
        encoder.copyBufferToBuffer(piece.values, (r - piece.first) * n, target, j * n, n);
        encoder.copyBufferToBuffer(piece.scales, (r - piece.first) * perRow, target, rows.length * n + j * perRow, perRow);
      });
      m.device.queue.submit([encoder.finish()]);
      await target.mapAsync(MAP_READ);
      const bytes = target.getMappedRange().slice(0);
      return new Map(rows.map((r, j) => [r, [new Uint8Array(bytes, j * n, n), new Float32Array(bytes, rows.length * n + j * perRow, perRow / 4)]]));
    } finally {
      target.destroy();
    }
  };
  return { embedding: await read(m.tables.embedding, specs.embedding, checkCandidates(vocab)),
    classifier: await read(m.tables.classifier, specs.classifier, checkRows(vocab)) };
}
async function checkTokens(m, form) {
  const { plan, wgsl, gen: g, device } = m, vocab = g.vocab, dim = plan.dim, headSize = plan.headSize, half = headSize / 2;
  const qDim = plan.heads * headSize, kvDim = plan.kvHeads * headSize, line = form.dp4a ? DP4A_LINE : LAYER_LINE;
  const floats = (address, n) => Float64Array.from(new Float32Array(m.memory.buffer, address, n));
  // row r of a matrix or a table ({ n, six, ternary }, its values and scales at [valuesAt, scalesAt] in the shared
  // memory): [its int8 values, its scales (one a group of 32 weights; T232, ternary: of 128)]; (T156) of a matrix read
  // back from the GPU (firstLayer: [values, scales]); (T210) of a table's rows read back (tableRows)
  const stored = (spec) => (spec.six ? (spec.n * 3) / 4 : rowBytes(spec)), scalesOf = (spec) => rowBytes(spec) / wgsl.GROUP;
  // the bytes of a row as its int8 values: int6 widened, ternary codes less one, int8 as they are
  const valuesOf = (spec, bytes) => (spec.six ? wgsl.sixValues(bytes) : spec.ternary ? wgsl.ternaryValues(bytes)
    : new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length));
  const inMemory = (spec, [valuesAt, scalesAt]) => (r) => [
    valuesOf(spec, new Uint8Array(m.memory.buffer, valuesAt + r * stored(spec), stored(spec))),
    new Float32Array(m.memory.buffer, scalesAt + r * scalesOf(spec) * 4, scalesOf(spec))];
  const readBackRows = (spec, [values, scales]) => (r) => [valuesOf(spec, values.subarray(r * stored(spec), (r + 1) * stored(spec))),
    scales.subarray(r * scalesOf(spec), (r + 1) * scalesOf(spec))];
  const tableSpec = (name) => (name === "embedding" ? plan.tokens.embedding ?? plan.tokens.classifier : plan.tokens.classifier);
  const tableRow = (name) => (m.tableRows ? (r) => {
    const [bytes, scales] = m.tableRows[name].get(r);
    return [valuesOf(tableSpec(name), bytes), scales];
  } : inMemory(tableSpec(name), tableSpec(name).at));
  // a matrix's rows (n weights each, rowOf(r): [values, scales]) times x, the rows given (packed: x quantized first).
  // The vector's groups are of 32; a row's scales cover as many of them each as they are fewer (one; T232, ternary: four)
  const product = (n, rowOf, x, rows, packed = form.dp4a) => {
    const q = packed ? wgsl.quantizedLikeCpu(Float32Array.from(x)) : null, perRow = n / wgsl.GROUP;
    return Float64Array.from(rows, (r) => {
      const [w, s] = rowOf(r), each = perRow / s.length;
      let sum = 0;
      for (let b = 0; b < perRow; b++) {
        let part = 0;
        for (let i = b * wgsl.GROUP; i < (b + 1) * wgsl.GROUP; i++) part += w[i] * (q ? q.xq[i] : x[i]);
        sum += part * s[Math.floor(b / each)] * (q ? q.xs[b] : 1);
      }
      return sum;
    });
  };
  const all = (n) => [...Array(n).keys()];
  const matmul = (name, x) => {
    const matrix = plan.matrices[name];
    return product(matrix.n, m.firstLayer ? readBackRows(matrix, m.firstLayer[name]) : inMemory(matrix, matrix.layers[0]), x, all(matrix.rows));
  };
  const rms = (x, weights) => {
    const s = 1 / Math.sqrt(x.reduce((sum, v) => sum + v * v, 0) / x.length + plan.eps);
    return x.map((v, i) => weights[i] * (s * v));
  };
  // T226: the stream's norm, RMSNorm or (T154) LayerNorm with its bias, as the CPU's kernels have them
  const norm = (x, weights, bias) => {
    if (!plan.layerNorm) return rms(x, weights);
    const mean = x.reduce((sum, v) => sum + v, 0) / x.length;
    const s = 1 / Math.sqrt(x.reduce((sum, v) => sum + (v - mean) ** 2, 0) / x.length + plan.eps);
    return x.map((v, i) => weights[i] * (s * (v - mean)) + bias[i]);
  };
  // the first layer's vector of a name (plan.vectors), where the model has it; and a vector with it added
  const vectorOf = (name) => plan.vectors[name] && floats(plan.vectors[name].at, plan.vectors[name].size);
  const plus = (x, name) => {
    vectorOf(name)?.forEach((b, i) => { x[i] += b; });
    return x;
  };
  const largest =(xs) => xs.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
  const off = (got, want) => largest(want.map((v, i) => got[i] - v));
  const owned = [];
  try {
    // 1. the first layer at position 1
    // a token of the second half of the vocabulary whose row is large: the rows of tokens no text has (llm-jp-3's
    // last) are nearly 0, and their keys and values then float16's subnormals (a check of them was 15% off, CI)
    const embeddingRow = tableRow("embedding");
    const sizeOf = (t) => embeddingRow(t)[1].reduce((sum, v) => sum + Math.abs(v), 0);
    const token = checkCandidates(vocab).reduce((best, t) => (sizeOf(t) > sizeOf(best) ? t : best));
    const pos = 1;
    const halfBits = () => (Math.random() < 0.5 ? 0x8000 : 0) | ((13 + ((Math.random() * 3) | 0)) << 10) | ((Math.random() * 1024) | 0);
    const row0 = [0, 1].map(() => new Uint16Array(kvDim).map(halfBits));
    if (m.cache.capacity < 2) grow(m, 2);
    device.queue.writeBuffer(m.cache.keys[0], 0, row0[0]);
    device.queue.writeBuffer(m.cache.values[0], 0, row0[1]);
    const readH = buffer(m, dim * 4, MAP_READ | COPY_DST, owned);
    const { kv } = await runTokens(m, tokenPass(m, form, { to: 1, head: false, positions: pos + 1 }), { count: 1, pos, keep: true,
      state: wgsl.samplingState({ token, pos, history: [token] }), extra: (encoder) => encoder.copyBufferToBuffer(g.h, 0, readH, 0, dim * 4) });
    await readH.mapAsync(MAP_READ);
    const h = new Float32Array(readH.getMappedRange().slice(0));
    readH.unmap();
    const [gotK, gotV] = kv.map((layers) => Float64Array.from(new Uint16Array(layers[0].buffer, layers[0].byteOffset, kvDim), halfToFloat));
    // JavaScript's layer
    const [eRow, eScales] = embeddingRow(token);
    // (T226: with GPT-2's learned position of pos, as the CPU's embed() adds it)
    const x0 = Float64Array.from(eRow, (v, i) => v * eScales[Math.floor((i * eScales.length) / eRow.length)]);
    if (plan.tokens.positions) floats(plan.tokens.positions + pos * dim * 4, dim).forEach((p, i) => { x0[i] += p; });
    const xn = norm(x0, vectorOf("attention"), vectorOf("attentionBias"));
    // T226: as the CPU has it (forward.js), the biases (Qwen2's; T154's), then the norms of the heads of q and k
    // (Qwen3's), then RoPE (all of a head, T154: a part of it, or none); the first layer's of each, where the model
    // has them
    const [q, k, v] = [["wq", "bq"], ["wk", "bk"], ["wv", "bv"]].map(([name, bias]) => plus(matmul(name, xn), bias));
    for (const [name, x] of [["qNorm", q], ["kNorm", k]]) {
      const weights = vectorOf(name);
      for (let at = 0; weights && at < x.length; at += headSize) x.set(rms(x.subarray(at, at + headSize), weights), at);
    }
    const cos = floats(plan.cos + pos * half * 4, half), sin = floats(plan.sin + pos * half * 4, half);
    const turn = (vector) => {
      for (let at = 0; at < vector.length; at += 2) {
        const i = (at % headSize) / 2;
        if (at % headSize >= plan.turned) continue;
        const [a, b] = [vector[at], vector[at + 1]];
        vector[at] = a * cos[i] - b * sin[i];
        vector[at + 1] = a * sin[i] + b * cos[i];
      }
    };
    turn(q);
    turn(k);
    const keys = [Float64Array.from(row0[0], halfToFloat), heldFloats(k, gotK)], values = [Float64Array.from(row0[1], halfToFloat), heldFloats(v, gotV)];
    const att = new Float64Array(qDim), group = plan.heads / plan.kvHeads;
    for (let head = 0; head < plan.heads; head++) {
      const kvAt = Math.floor(head / group) * headSize, at = head * headSize;
      const scores = keys.map((key) => {
        let sum = 0;
        for (let d = 0; d < headSize; d++) sum += q[at + d] * key[kvAt + d];
        return sum / Math.sqrt(headSize);
      });
      const top = Math.max(...scores), weights = scores.map((score) => Math.exp(score - top)), total = weights[0] + weights[1];
      for (let d = 0; d < headSize; d++) att[at + d] = (weights[0] * values[0][kvAt + d] + weights[1] * values[1][kvAt + d]) / total;
    }
    const o = plus(matmul("wo", att), "bo"), h1 = x0.map((value, i) => value + o[i]);
    // (T154: GPT-NeoX's parallel residual, the FFN's norm of the layer's input; no gate, w1's bias and GELU as
    // shaders.js's has it)
    const xn2 = norm(plan.parallel ? x0 : h1, vectorOf("ffn"), vectorOf("ffnBias"));
    const gate = plus(matmul("w1", xn2), "b1"), up = plan.matrices.w3 && matmul("w3", xn2);
    const gelu = (value) => 0.5 * value * (1 + Math.tanh(Math.min(9.010913, Math.max(-9.010913, 0.7978845608028654 * (value + 0.044715 * value ** 3)))));
    const activated = gate.map(up ? (value, i) => (value / (1 + Math.exp(-value))) * up[i] : gelu);
    const down = plus(matmul("w2", activated), "b2"), h2 = h1.map((value, i) => value + down[i]);
    const stream = off(h, h2) / largest(h2.map((value, i) => value - x0[i]));
    const keyOff = off(gotK, k) / largest(k), valueOff = off(gotV, v) / largest(v);
    if (!(stream <= line)) return `the first layer's stream is ${stream.toExponential(2)} of its change from JavaScript's (line ${line})`;
    if (!(keyOff <= line && valueOff <= line)) {
      return `the first layer's keys are ${keyOff.toExponential(2)} and its values ${valueOff.toExponential(2)} from JavaScript's (line ${line})`;
    }
    // 2. the head: greedy, then sampled
    const stream2 = new Float32Array(dim).map(() => (Math.random() - 0.5) * 4);
    const rows = checkRows(vocab);
    const readLogits = buffer(m, vocab * 4, MAP_READ | COPY_DST, owned);
    const head = tokenPass(m, form, { to: 0, embed: false });
    const run = async (settings, history, random) => {
      device.queue.writeBuffer(g.h, 0, stream2);
      const { ids } = await runTokens(m, head, { count: 1, pos, state: wgsl.samplingState({ token, pos, history }), settings,
        randoms: new Float32Array([random]), extra: (encoder) => encoder.copyBufferToBuffer(g.logits, 0, readLogits, 0, vocab * 4) });
      await readLogits.mapAsync(MAP_READ);
      // (T226: the vocabulary's logits and no more. A buffer is made in whole 16 bytes, and GPT-2's 50257 logits left
      // 3 zeros after them: where every logit of the made-up stream was negative, the check took one of those zeros
      // for the largest, and refused a form that was right: CI's run 36869126011, 2 runs of 16)
      const logits = new Float32Array(readLogits.getMappedRange().slice(0, vocab * 4));
      readLogits.unmap();
      return { id: ids[0], logits };
    };
    const greedy = await run(wgsl.samplingSettings({ vocab, temperature: 0, topp: 0.9 }), [token], 0);
    // (T226: a classifier of floats on DP4A too where the model has outlier channels: tokenShape's floatHead)
    const packedHead = form.dp4a && !floatHead(m);
    const normedStream = norm(Float64Array.from(stream2), floats(plan.tokens.final, dim), plan.tokens.finalBias && floats(plan.tokens.finalBias, dim));
    // T232: a ternary classifier's outlier channels as the CPU has them (forward.js's picked): taken out of the normed
    // stream before it is quantized, and their columns of the table multiplied apart, in floats
    const taken = g.take ? plan.tokens.channels.map((c) => {
      const value = normedStream[c];
      normedStream[c] = 0;
      return [c, value];
    }) : [];
    const want = product(dim, tableRow("classifier"), normedStream, rows, packedHead);
    rows.forEach((r, j) => {
      const [w, s] = tableRow("classifier")(r);
      for (const [c, value] of taken) want[j] += value * w[c] * s[Math.floor((c * s.length) / w.length)];
    });
    const logitsOff = off(rows.map((r) => greedy.logits[r]), want) / largest(want), logitsLine = packedHead ? DP4A_LINE : LOGITS_LINE;
    if (!(logitsOff <= logitsLine)) return `the logits are ${logitsOff.toExponential(2)} of their largest from JavaScript's (line ${logitsLine})`;
    if (greedy.id !== wgsl.argmaxLikeCpu(greedy.logits)) return `the greedy token is ${greedy.id}, the largest logit's ${wgsl.argmaxLikeCpu(greedy.logits)}`;
    const sampled = { temperature: 0.8, topp: 0.9, penalty: 1.3 }, history = [token, greedy.id, 0, 1];
    const drawn = await run(wgsl.samplingSettings({ vocab, ...sampled }), history, 0.7);
    const penalized = greedy.logits.slice();
    wgsl.penalizeLikeCpu(penalized, history, sampled.penalty);
    const penaltyOff = off(history.map((t) => drawn.logits[t]), history.map((t) => penalized[t])) / Math.max(largest(history.map((t) => penalized[t])), 1e-30);
    if (!(penaltyOff <= 1e-6)) return `the penalized logits are ${penaltyOff.toExponential(2)} from JavaScript's`;
    const walk = wgsl.walkLikeCpu(drawn.logits, sampled.temperature, sampled.topp), goal = 0.7 * walk.mass, band = 1e-4 * walk.mass;
    const reached = walk.tokens.filter((_, j) => walk.cumulative[j] > goal - band && (j ? walk.cumulative[j - 1] : 0) <= goal + band);
    if (!reached.includes(drawn.id)) return `the sampled token is ${drawn.id}, the CPU's walk reaches ${reached.join(" or ")}`;
    return null;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

// A request for count steps from token at pos (forward.js's generateMany): the keys and values of positions from to
// pos - 1 up from forward.js's cache first (cache: its addresses of the keys and the values, its capacity and the
// bytes of a position; half: float16 as here, else float32, narrowed on the way up: T160, a grouped-query model's),
// the state (token, pos, the end of the history and its length), the settings and a random number a step; the ids into
// plan.tokens.ids ([sampled, id, id, ...] and, after plan.tokens.most ids, the State's not_finite word: T219, the step
// after the sampled ones was refused, its logits not finite; a stop token last where one came), and the keys and values of the positions
// sampled in float16 into plan.staging as a prompt's block's ([keys, values][layer][plan.batch positions]), which
// forward.js puts into its cache (widened where it is float32), where it still wants the answer. T210: a model on the
// GPU alone (cache null) has no cache there: nothing goes up, and nothing but the ids comes back
function generate({ serial, count, pos, from, token, history, length, cache, settings, randoms }) {
  serve(serial, async (wanted) => {
    const m = model, { plan, wgsl, gen: g } = m, kvDim = plan.kvHeads * plan.headSize, kvRow = kvDim * 2;
    if (!g?.form) throw new Error("no tokens on this GPU");
    if (pos + count > m.cache.capacity) grow(m, pos + count);
    const at = (block, l, p) => block + l * cache.capacity * cache.row + p * cache.row;
    for (let l = 0; cache && l < plan.layers; l++) {
      for (const [block, target] of [[cache.keys, m.cache.keys[l]], [cache.values, m.cache.values[l]]]) {
        if (cache.half) copyIn(m, target, at(block, l, from), (pos - from) * kvRow, from * kvRow);
        else narrowIn(m, target, at(block, l, from), (pos - from) * kvDim, from * kvRow);
      }
    }
    const out = await runTokens(m, tokenStep(m, g.form, pos + count), { count, pos, keep: Boolean(cache),
      state: wgsl.samplingState({ token, pos, history, length }), settings: wgsl.samplingSettings({ vocab: g.vocab, ...settings }),
      randoms: Float32Array.from({ length: count }, (_, i) => randoms[i] ?? 0) });
    if (!wanted()) return;
    for (let side = 0; cache && side < 2; side++) {
      for (let l = 0; l < plan.layers; l++) {
        new Uint8Array(m.memory.buffer, plan.staging + (side * plan.layers + l) * plan.batch * kvRow, out.sampled * kvRow)
          .set(out.kv[side][l].subarray(0, out.sampled * kvRow));
      }
    }
    const ids = new Int32Array(m.memory.buffer, plan.tokens.ids, 2 + g.most);
    ids[0] = out.sampled;
    ids.set(out.ids.subarray(0, out.sampled), 1);
    ids[1 + g.most] = out.notFinite;
  });
}

// ---- a block of a prompt (T210: tokens, the ids of a model on the GPU alone, which embeds them here)
function prompt({ serial, count, pos, tokens }) {
  serve(serial, (wanted) => block(model, count, pos, wanted, false, tokens));
}
// The answer to a request (a block of a prompt, T152: the steps of a generation), in the control area: work(wanted)
// runs while words.beat counts up, then words.failed and words.done = serial, where forward.js still waits for this
// request (wanted: T147, it gave up, and the memory may soon be another model's)
async function serve(serial, work) {
  const { memory, plan } = model;
  const words = new Int32Array(memory.buffer, 0, Math.max(...Object.values(plan.words)) + 1);
  // the model's worker waits: this says that the work goes on, however long the GPU takes (a software adapter)
  const beat = setInterval(() => Atomics.add(words, plan.words.beat, 1), 250);
  const wanted = () => Atomics.load(words, plan.words.wanted) === serial;
  let failed = 1;
  try {
    if (lost) throw new Error(lost);
    await work(wanted);
    if (lost) throw new Error(lost);
    failed = 0;
  } catch (error) {
    postMessage({ type: "failed", reason: String(error?.message ?? error) });
  } finally {
    clearInterval(beat);
    if (wanted()) {
      Atomics.store(words, plan.words.failed, failed);
      Atomics.store(words, plan.words.done, serial);
      Atomics.notify(words, plan.words.done);
    }
  }
}

// A block of count tokens at pos through the layers, its keys and values into plan.staging where wanted() still says
// so. timing (T148, timeBlocks): rows and angles of zeros instead of forward.js's, and nothing written back
async function block(m, count, pos, wanted, timing = false, tokens) {
  const { device, plan } = m, half = plan.headSize / 2;
  // the rows forward.js embedded (dense), and the angles of their positions (the tables forward.js has); views of just
  // those floats (T155: not a view of the whole of a 64-bit memory of some GB for every block)
  const floats = (address, n) => new Float32Array(m.memory.buffer, address, n);
  // (T210: on the GPU alone, the tokens themselves, embedded here; token 0 where it is timed)
  if (plan.direct) m.device.queue.writeBuffer(m.ids, 0, Uint32Array.from({ length: count }, (_, t) => (timing ? 0 : tokens[t])));
  else if (timing) m.rows.fill(0);
  else m.rows.set(floats(plan.rows, count * plan.dim));
  // (T154: GPT-2 turns nothing and has no tables)
  if (timing || !plan.turned) {
    m.turns.fill(0);
  } else {
    for (let t = 0; t < count; t++) {
      m.turns.set(floats(plan.cos + (pos + t) * half * 4, half), t * plan.headSize);
      m.turns.set(floats(plan.sin + (pos + t) * half * 4, half), t * plan.headSize + half);
    }
  }
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  if (!plan.direct) device.queue.writeBuffer(m.x, 0, m.rows, 0, count * plan.dim);
  device.queue.writeBuffer(m.angles, 0, m.turns, 0, count * plan.headSize);
  device.queue.writeBuffer(m.step, 0, new Uint32Array([count, pos, 0, 0]));
  if (pos + count > m.cache.capacity) grow(m, pos + count);
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(), form = m.form;
  const multiplied = (pieces) => pieces.forEach((piece) => multiply(m, pass, form, piece.group, piece.rows, count));
  // a packed form reads its input quantized: once for the matrices that read the same one
  const quantize = (q) => form.packed && dispatch(pass, m.quantize, q.group, q.x, count);
  const flash = m.attention;
  // a bias added to a matrix's output (T153: Qwen2's q, k and v; T154: GPT-2's and GPT-NeoX's every matrix's)
  const biased = (bias) => bias && dispatch(pass, m.add, bias.group, bias.x, count);
  m.embedGroups?.forEach((group) => dispatch(pass, m.embedRows, group, 1, count));
  for (let l = 0; l < plan.layers; l++) {
    const layer = m.layers[l];
    dispatch(pass, m.norm, layer.attentionNorm, 1, count);
    quantize(m.quantizeXb);
    multiplied(layer.q);
    multiplied(layer.k);
    multiplied(layer.v);
    // T153: as the CPU has it (forward.js), the biases, then the norms of the heads, then RoPE
    layer.biases.forEach(biased);
    for (const norm of layer.headNorms) dispatch(pass, m.headNorm, norm.group, norm.heads, count);
    dispatch(pass, m.rope, m.cache.rope[l], count);
    if (l === plan.layers - 1) break;  // the keys and values are all a prompt's token leaves
    // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, before o adds to it
    if (plan.parallel) dispatch(pass, m.norm, layer.ffnNorm, 1, count);
    dispatch(pass, flash.pipeline, m.cache.attention[l], plan.heads * Math.ceil(count / m.wgsl.FLASH_Q_TILE));
    quantize(m.quantizeAttention);
    multiplied(layer.o);
    biased(layer.oBias);
    if (!plan.parallel) dispatch(pass, m.norm, layer.ffnNorm, 1, count);
    quantize(m.quantizeFfn);
    multiplied(layer.gate);
    if (layer.up) multiplied(layer.up);
    biased(layer.gateBias);
    dispatch(pass, m.activation, m.activationGroup, Math.ceil(plan.hidden / 64), count);
    quantize(m.quantizeGate);
    multiplied(layer.down);
    biased(layer.downBias);
  }
  pass.end();
  // (T210: a model on the GPU alone keeps them there: nothing comes back)
  if (!plan.direct) keysOut(m, encoder, pos, count);
  device.queue.submit([encoder.finish()]);
  const invalid = await device.popErrorScope(), full = await device.popErrorScope();
  if (invalid || full) throw new Error(`the GPU refused a block (${(invalid ?? full).message})`);
  if (plan.direct) await device.queue.onSubmittedWorkDone();
  else await keysBack(m, () => !timing && wanted());
}
// the keys of count positions from pos of every layer, then their values, [layers][plan.batch][kvDim] each in float16,
// into m.readback as plan.staging lays them out
function keysOut(m, encoder, pos, count) {
  const { plan } = m, B = plan.batch, row = plan.kvHeads * plan.headSize * 2;
  for (let l = 0; l < plan.layers; l++) {
    encoder.copyBufferToBuffer(m.cache.keys[l], pos * row, m.readback, l * B * row, count * row);
    encoder.copyBufferToBuffer(m.cache.values[l], pos * row, m.readback, (plan.layers + l) * B * row, count * row);
  }
}
// and from m.readback into plan.staging, where wanted() still says so
async function keysBack(m, wanted) {
  const { plan } = m;
  await m.readback.mapAsync(MAP_READ);
  try {
    const bytes = 2 * plan.layers * plan.batch * plan.kvHeads * plan.headSize * 2;
    if (wanted()) new Uint8Array(m.memory.buffer, plan.staging, bytes).set(new Uint8Array(m.readback.getMappedRange(), 0, bytes));
  } finally {
    m.readback.unmap();
  }
}
// T210: the GPU's own keys and values of count positions (plan.batch at most) from pos into plan.staging, as a block
// writes them back: a model on the GPU alone keeps none in the shared memory, and the tests read them so
// (forward.js's keysAndValues)
function keysOf({ serial, count, pos }) {
  serve(serial, async (wanted) => {
    const encoder = model.device.createCommandEncoder();
    keysOut(model, encoder, pos, count);
    model.device.queue.submit([encoder.finish()]);
    await keysBack(model, wanted);
  });
}
