// What every part of the model's GPU worker stands on: the shaders' module, what the parts share (the model's record,
// whether a stop was asked, a device lost), the time a step may take, the device and why there is none, the worker's
// end, and buffers: made, written from the shared memory, read back, bound and dispatched.
// (T352: a module of the model's GPU worker, public/gpu.js, which asks for it with its own ?v=<build>)

const shaders = import(new URL(`../shaders.js${new URL(import.meta.url).search}`, import.meta.url));

// GPUBufferUsage's values (the name itself is missing where there is no WebGPU)
const STORAGE = 0x80, COPY_DST = 0x8, COPY_SRC = 0x4, MAP_READ = 0x1, UNIFORM = 0x40;
// the bytes that go to the GPU through a copy of their own at a time (writeBuffer takes no view of a shared memory
// everywhere, and copies what it is given at once)
const CHUNK = 8 << 20;


// T232: the bytes of a row of a matrix or a table ({ n, ternary }) on the GPU: a byte a weight (int8, and int6 widened),
// or two bits (ternary: the codes as the checkpoint holds them, 16 weights a u32). A float32 scale goes with every
// wgsl.GROUP (32) bytes of a row either way: an int8 group of 32 weights, a ternary group of 128
const rowBytes = ({ n, ternary }) => (ternary ? n / 4 : n);
// (whether the model's weights are ternary: its layers' matrices are all of one kind, llama2_numpy's dtype)
const ternaryPlan = (plan) => Object.values(plan.matrices).some((matrix) => matrix.ternary);
// T255: how many values of a head RoPE turns in a layer: plan.turned, but none where the model leaves the layer's q
// and k alone (a SmolLM3's every fourth; the shaders take the number as GPT-2's 0)
const turnedAt = (plan, l) => (plan.unturned?.includes(l) ? 0 : plan.turned);

// T352: what the parts of this worker share and one of them sets (a module's `let` cannot be assigned from another
// module): the fields were gpu.js's `let`s of the same names
const common = {
  model: null,  // what is on the GPU for the model: the device, the plan, the buffers, the pipelines, the cache
  stopping: false,  // a stop was asked: start() ends at its next step
  lost: null,  // why the device was lost, once it is
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
    if (common.stopping) return end();
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
    device.lost.then((info) => { common.lost = `the GPU was lost (${info.reason}${info.message ? `: ${info.message}` : ""})`; });
    // T148: what the page keeps of an earlier visit counts only for the same adapter and browser
    // (and the shaders of this deployment: a site whose shaders changed chooses anew, the review of T148)
    // (T232: a ternary model's key holds its own shaders too)
    const key = wgsl.deviceKey(adapter, device, ternary);
    const remembered = plan.remembered?.key === key ? plan.remembered : null;
    common.model = { device, plan, wgsl, owned: [], limit, info, fallback, remembered, adapter, key, ternary };
    if (common.stopping) return end();
    // a buffer the device cannot give fails quietly, as an error of these scopes
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    return common.model;
}

function describe(adapter) {
  const info = adapter.info ?? {};
  const name = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "a GPU";
  return (info.isFallbackAdapter ?? adapter.isFallbackAdapter) ? `${name} (a fallback adapter: the CPU in the GPU's place)` : name;
}

function unusable(reason) {
  postMessage({ type: "unusable", reason });
  end();
}

// every buffer and the device let go (T94: a model changed for another leaves nothing on the GPU), and this worker ends.
// T205: it says { type: "ended" } first, for public/forward/engine.js's release() to read the next model after it. Every way out of
// start() comes here (a stop in the middle of it at its next step), and a stop before or after start() at once
function end() {
  if (common.model) {
    common.model.owned.forEach((buffer) => buffer.destroy());
    common.model.cache?.owned.forEach((buffer) => buffer.destroy());
    common.model.device.destroy();
    common.model = null;
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

export { STORAGE, COPY_DST, COPY_SRC, MAP_READ, UNIFORM, CHUNK, rowBytes, turnedAt, common, within, openDevice,
  describe, unusable, end, buffer, uniform, copyIn, narrowIn, readBack, validated, pipelineOf, bind, dispatch };
