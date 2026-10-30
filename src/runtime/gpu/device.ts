// What every part of the model's GPU worker stands on: what the parts share (the model's record, whether a stop was
// asked, a device lost), the time a step may take, the device and why there is none, the worker's end, and buffers:
// made, written from the shared memory, read back, bound and dispatched.
// (T.352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T.367.3)
// (T.367.5: the WGSL is no module handed round any more. Every module of this worker takes the names it uses from the
// shaders' parts, src/runtime/shaders/<part>.ts, as it takes its neighbours': linked into the worker's file, with
// nothing to fetch after it and nothing of /benchmark/'s own shaders in it. The device's key is key.ts's, the same
// text the model's worker reads its key from: tests/dist-check.mjs holds the two built files to one key.)
// (T.379.1: the shaders themselves, the texts and their makers, are taken from one catalogue, shaders/engine.ts's
// ENGINE, which is what the key hashes: a module compiles ENGINE.<name> and imports no shader of a part.)
import { halvesOf } from "../shaders/run.ts";
import { deviceKey } from "../shaders/key.ts";
import { Failure, kindOf } from "../failure.ts";
import { REASONS } from "../forward/reasons.ts";
import { PACKINGS, packingsOf, isBlock } from "./packings.ts";
import type { FromGpu, GpuPacking, Matrices, Opened, Remembered, Started, Tables, Tokens as TokensPlan } from "../contracts/gpu/protocol.ts";
import type { Own } from "../shaders/prompt.ts";
import type { Weights } from "./weights.ts";
import type { Block } from "./block.ts";
import type { Forms } from "./forms.ts";
import type { TokenBuffers } from "./tokens.ts";
import type { Chosen as TokenAttentionChosen } from "./tokenattention.ts";
import type { TokenChoice } from "./tokenforms.ts";

// T.368.6: the types of what the parts of this worker hand one another. They say the shapes as they are (the record's
// own notes are below, at model).
//
// What the adapter said of itself, where it says anything (adapter.info is missing in an older browser, which said
// whether it is a fallback on the adapter itself)
type Info = Partial<Pick<GPUAdapterInfo, "vendor" | "architecture" | "device" | "description" | "isFallbackAdapter" | "subgroupMinSize" | "subgroupMaxSize">>;
type Adapter = GPUAdapter & { isFallbackAdapter?: boolean };
// the plan a model on the GPU alone is opened with (start.ts's open(): its layers are joined for a token always)
type Opening = Opened & { tokens: true };
// The record of the model on the GPU, as its readers hold it: what openDevice() puts in it, and the six records of
// the other modules. plan is start()'s: a model on the GPU alone is opened with another (Opening) before its
// checkpoint comes, of which only what the two plans share is read (matrices, layers, force, remembered, whether
// there are tokens; the tables through weights.ts's tablesOf()) until handed() puts start()'s in, with the memory
// (null till then). weights, block and forms are set by their owners before another module reads them, in start()'s
// order; tokens, tokenAttention and tokenForm are set as a token's are chosen (T.466), with the form null (or the
// records unset, T.467) where no form of them is right here (the two that may meet it ask for the form: requests.ts's generate(), tokenforms.ts's
// tokensSaid()). T.479: the type says those three may be missing, and their readers take them once at their heads
type Base = {
  device: GPUDevice;
  plan: Started;
  memory: WebAssembly.Memory;
  owned: GPUBuffer[];
  held: GPUBuffer[] | null;
  limit: number;
  info: Info;
  fallback: boolean;
  remembered: Remembered | null;
  adapter: Adapter;
  key: string;
  ternary: boolean;
  // T.547.2.3: the shaders the model has of its own (ternary's, or Q4_K's beside int8), which its key holds
  own: Own;
};
type Model = Base & { weights: Weights; block: Block; forms: Forms; tokens?: TokenBuffers; tokenAttention?: TokenAttentionChosen; tokenForm?: TokenChoice };
// (the record as openDevice() makes it: its literal is held to this, then taken as the Model it becomes)
type Made = Omit<Base, "plan" | "memory"> & { plan: Started | Opening; memory: null };
// a buffer, or a part of one ({ buffer, offset, size }), as bind() takes either (the whole one has no .buffer to ask)
type Whole = GPUBuffer & { buffer?: undefined; offset?: undefined };
type Bound = Whole | GPUBufferBinding;
// a dispatch: [pipeline, bind group, x, y]
type Pass = [GPUComputePipeline, GPUBindGroup, number, number];
// what is caught, as it is read: an Error's message, and within()'s late
type Thrown = { message?: string } | null | undefined;
// (T.479: what a caught value says, as the worker tells it: its message, else the value itself)
const whyOf = (error: unknown) => String((error as Thrown)?.message ?? error);

// GPUBufferUsage's values (the name itself is missing where there is no WebGPU)
const STORAGE = 0x80, COPY_DST = 0x8, COPY_SRC = 0x4, MAP_READ = 0x1, UNIFORM = 0x40;
// the bytes that go to the GPU through a copy of their own at a time (writeBuffer takes no view of a shared memory
// everywhere, and copies what it is given at once)
const CHUNK = 8 << 20;


// T.232: the bytes of a row of a matrix or a table ({ n, packing }) on the GPU (T.547.2.1: its packing's, packings.ts)
const rowBytes = ({ n, packing }: { n: number; packing: GpuPacking }) => PACKINGS[packing].row(n);
// (whether the model's weights are ternary: a ternary checkpoint's matrices are all ternary, llama2_numpy's dtype, and no
// other model has a ternary one)
// (T.547.3.1's review: a layer's of a stack whose layers are each of a packing of their own too)
const ternaryPlan = (plan: { matrices: Matrices }) => Object.values(plan.matrices).some((matrix) => packingsOf(matrix).includes("ternary"));
// T.547.2.3: the shaders a model has of its own: ternary's, or (a matrix of llama.cpp's blocks among int8 ones) Q4_K's
// (T.547.3.1: a layer's of a stack whose layers are each of a packing of their own too)
// (T.547.3.6's review: and a table's, the classifier's or the embedding's, of a model whose matrices are no blocks)
const ownOf = (plan: { matrices: Matrices; tokens?: TokensPlan | null | true; tables?: Tables }): Own => {
  const tables = plan.tables ?? (plan.tokens === true ? null : plan.tokens);
  return ternaryPlan(plan) ? "ternary"
  : [...Object.values(plan.matrices), ...[tables?.classifier, tables?.embedding].filter((table) => table !== null && table !== undefined)]
    .some((matrix) => packingsOf(matrix).some(isBlock)) ? "blocks" : null;
};
// T.255: how many values of a head RoPE turns in a layer: plan.turned, but none where the model leaves the layer's q
// and k alone (a SmolLM3's every fourth; the shaders take the number as GPT-2's 0)
const turnedAt = (plan: Pick<Started, "unturned" | "turned">, l: number) => (plan.unturned?.includes(l) ? 0 : plan.turned);

// T.378.4: what the parts of this worker read and this one alone sets (a static import is the live binding: the others
// take the names and read them, and say what happened where they have something to say: stopAsked(), handed(),
// held()). They were the fields of one object, common, that any part could set (T.352)
//
// model is the record of what is on the GPU for the model. What openDevice() puts in it is this module's and is read
// by all: the device, the plan and the shared memory, the buffers that live as long as the model (owned), and what
// the adapter said of itself. The rest is six records, each made and set by one module and read by the others:
//   model.weights  weights.ts   the layers' matrices and vectors and a token's tables, as buffers
//   model.block    block.ts     the small steps' pipelines, a block's buffers and bind groups, the keys and values
//   model.forms    forms.ts     the prompt's attention and tiled shader chosen here, and what each one tried came to
//   model.tokens   tokenforms.ts  a token's buffers and pipelines (tokens.ts's tokenBuffers() declares them, T.466)
//   model.tokenAttention  tokenattention.ts  the attention of a token chosen here, and what each one tried came to
//   model.tokenForm  tokenforms.ts  the form of a token's layer chosen here, its ms a step and what each form came to
// (tests/gpu-modules-check.mjs holds every field read to one its owner declares, and every write to its owner)
let model: Model | null = null;
let stopping = false;  // a stop was asked: start() ends at its next step
let lost: string | null = null;  // why the device was lost, once it is
// a stop was asked (start.ts's stop(): the worker ends at once, or start() does at its next step)
function stopAsked() {
  stopping = true;
}
// start() was asked: the shared memory the model's weights and requests are read from, and the plan that says where
// (a model on the GPU alone was opened with another, before its checkpoint came: T.156)
function handed(memory: WebAssembly.Memory, plan: Started) {
  model!.memory = memory;
  model!.plan = plan;
}
// the buffers of what is made again as it grows (block.ts's keys and values): end() lets go of the ones held last,
// after the model's own (the ones before them went as they were replaced)
function held(m: Model, buffers: GPUBuffer[]) {
  m.held = buffers;
}


// The most a step of putting the model on the GPU may take: a shader's compilation, a check, the timing, a layer's
// weights. SwiftShader took up to 90 s to compile llama.cpp's 64×64 tiles on the development machine (T.147), a GPU
// compiles in far less; twice that is a GPU that hangs. The worker then gives the GPU up (the prompt stays on the
// CPU), and says { type: "progress" } after every step meanwhile (forward.ts gives up a worker that says nothing for
// longer than this: one the browser ended)
const STEP_MS = 180000;
// (T.380.1: a step past its time is a Failure of the kind "late": whoever tries one shader after another asks isLate()
// of what it caught, and throws that on: it is not a shader that is wrong, it ends the whole attempt)
const isLate = (error: unknown) => kindOf(error) === "late";
function within<T>(promise: Promise<T>, what: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const late = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Failure("late", `${what} took more than ${STEP_MS / 1000} s`)), STEP_MS);
  });
  return Promise.race([promise, late]).finally(() => {
    clearTimeout(timer);
    postMessage({ type: "progress" } satisfies FromGpu);
  });
}

// The device, the model's record (model), and the error scopes the layers' buffers go up in; undefined where there is
// none (say(why): unusable() by default) or the worker is stopping
async function openDevice(plan: Started | Opening, say: (reason: string) => void = unusable): Promise<Model | void> {
    if (!self.navigator?.gpu) return say(REASONS.noWebGpu);
    // the browser's own adapter (T.148, the review: "high-performance" would keep a laptop's second GPU awake for the
    // whole visit, for prompts that are mostly short; the device measures whichever it gets against its CPU anyway)
    const adapter: Adapter | null = await navigator.gpu.requestAdapter();
    if (stopping) return end();
    if (!adapter) return say(REASONS.noAdapter);
    // T.148: a fallback adapter is the CPU doing the GPU's work (SwiftShader, lavapipe): never faster than the CPU's
    // own kernels, and its compilation of the shaders alone took 2 to 4 minutes (T.147). Refused before anything is
    // made on it, but where a test asks for it (it is the only WebGPU of CI and of the development machine)
    const info: Info = adapter.info ?? {};
    const fallback = Boolean(info.isFallbackAdapter ?? adapter.isFallbackAdapter);
    if (fallback && !plan.force.fallback) return say(REASONS.fallback);
    // T.232: ternary weights are multiplied as packed int8 alone (shaders/prompt.ts's TERNARY_PACKED; T.547.2.3: and Q4_K's
    // blocks, packings.ts's dot): said before anything goes up
    const ternary = ternaryPlan(plan), own = ownOf(plan);
    const dotted = Object.values(plan.matrices).flatMap(packingsOf).find((packing) => PACKINGS[packing].dot);
    if (dotted && !navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product")) {
      return say(REASONS.packed(dotted));
    }
    // the largest buffer the device binds (T.155: a matrix larger than it goes in pieces, piecesOf)
    const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
    // shader-f16 and subgroups where the adapter has them (a device refuses a feature it lacks), and the adapter's
    // workgroup memory and threads (the tiles and the attention size themselves by them)
    const { maxStorageBufferBindingSize, maxBufferSize, maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX } = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: (["shader-f16", "subgroups"] satisfies GPUFeatureName[]).filter((name) => adapter.features.has(name)),
      requiredLimits: { maxStorageBufferBindingSize, maxBufferSize, maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX } });
    device.lost.then((info) => { lost = `the GPU was lost (${info.reason}${info.message ? `: ${info.message}` : ""})`; });
    // T.148: what the page keeps of an earlier visit counts only for the same adapter and browser
    // (and the shaders of this deployment: a site whose shaders changed chooses anew, the review of T.148)
    // (T.232: a ternary model's key holds its own shaders too; T.547.2.3: and a model of Q4_K blocks its own)
    const key = deviceKey(adapter, device, own);
    const remembered = plan.remembered?.key === key ? plan.remembered : null;
    model = { device, plan, memory: null, owned: [], held: null, limit, info, fallback, remembered, adapter, key, ternary, own } satisfies Made as unknown as Model;
    if (stopping) return end();
    // a buffer the device cannot give fails quietly, as an error of these scopes
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    return model;
}

function describe(adapter: Adapter) {
  const info: Info = adapter.info ?? {};
  const name = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "a GPU";
  return (info.isFallbackAdapter ?? adapter.isFallbackAdapter) ? `${name} (a fallback adapter: the CPU in the GPU's place)` : name;
}

function unusable(reason: string) {
  postMessage({ type: "unusable", reason } satisfies FromGpu);
  end();
}

// every buffer and the device let go (T.94: a model changed for another leaves nothing on the GPU), and this worker ends.
// T.205: it says { type: "ended" } first, for src/runtime/forward/engine.ts's release() to read the next model after it. Every way out of
// start() comes here (a stop in the middle of it at its next step), and a stop before or after start() at once
function end() {
  if (model) {
    model.owned.forEach((buffer) => buffer.destroy());
    model.held?.forEach((buffer) => buffer.destroy());
    model.device.destroy();
    model = null;
  }
  postMessage({ type: "ended" } satisfies FromGpu);
  self.close();
}

// ---- buffers
function buffer(m: Model, bytes: number, usage: number = STORAGE, owned: GPUBuffer[] = m.owned) {
  const made = m.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
  owned.push(made);
  return made;
}
function uniform(m: Model, data: GPUAllowSharedBufferSource, owned: GPUBuffer[] = m.owned) {
  const made = buffer(m, data.byteLength, UNIFORM | COPY_DST, owned);
  m.device.queue.writeBuffer(made, 0, data);
  return made;
}
// bytes of the shared memory at address onto the GPU (at offset of target), a CHUNK at a time through a copy
let scratch: Uint8Array<ArrayBuffer> | undefined;
function copyIn(m: Model, target: GPUBuffer, address: number, bytes: number, offset = 0) {
  scratch ??= new Uint8Array(CHUNK);
  for (let done = 0; done < bytes; done += CHUNK) {
    const n = Math.min(CHUNK, bytes - done);
    scratch.set(new Uint8Array(m.memory.buffer, address + done, n));
    m.device.queue.writeBuffer(target, offset + done, scratch, 0, n);
  }
}
// T.152's review (T.160): values float32 of the shared memory at address narrowed to float16 onto the GPU (at offset
// of target, in bytes of float16), a CHUNK of them at a time: a float32 cache's keys and values going up
let narrowed: Uint16Array<ArrayBuffer> | undefined;
function narrowIn(m: Model, target: GPUBuffer, address: number, values: number, offset = 0) {
  narrowed ??= new Uint16Array(CHUNK / 2);
  for (let done = 0; done < values; done += narrowed.length) {
    const n = Math.min(narrowed.length, values - done);
    halvesOf(new Float32Array(m.memory.buffer, address + done * 4, n), narrowed);
    m.device.queue.writeBuffer(target, offset + done * 2, narrowed, 0, n);
  }
}
// T.533: float16 scales of the shared memory at address (halves of them) widened to float32 onto the GPU (at offset of
// target, in bytes), a CHUNK at a time: ternary's scales, which the checkpoint holds as float16s and the shaders read
// as float32s (exactly: every finite half is a float32's). widenedHalves (the same of bytes as they come, routed())
let widened: Float32Array<ArrayBuffer> | undefined;
function widenIn(m: Model, target: GPUBuffer, address: number, halves: number, offset = 0) {
  widened ??= new Float32Array(CHUNK / 4);
  for (let done = 0; done < halves; done += widened.length) {
    const n = Math.min(widened.length, halves - done);
    widenedHalves(new Uint16Array(m.memory.buffer, address + done * 2, n), widened);
    m.device.queue.writeBuffer(target, offset + done * 4, widened, 0, n);
  }
}
/** float16s (their bits) as float32s into out */
function widenedHalves(halves: Uint16Array, out: Float32Array) {
  for (let i = 0; i < halves.length; i++) {
    const h = halves[i], sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
    out[i] = exponent === 0 ? sign * fraction * 2 ** -24 : exponent === 31 ? (fraction ? NaN : sign * Infinity) : sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
  }
  return out;
}
async function readBack(m: Model, source: GPUBuffer, bytes: number, from = 0) {
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
async function validated<T>(m: Model, fn: () => T): Promise<Awaited<T>> {
  m.device.pushErrorScope("validation");
  let result!: Awaited<T>, failure: unknown;
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
const pipelineOf = (m: Model, code: string, constants?: Record<string, number>) => m.device.createComputePipelineAsync({ layout: "auto",
  compute: { module: m.device.createShaderModule({ code }), entryPoint: "main", constants } });
// RMSNORM's, HEAD_NORM's and NORM_QUANTIZE's Norm: the size of what it norms, at (of its weights), eps, and first (HEAD_NORM's:
// the rows before its first)
function normShape(m: Model, size: number, at: number, eps: number, first = 0) {
  const shape = new ArrayBuffer(16);
  new Uint32Array(shape).set([size, at, 0, first]);
  new Float32Array(shape, 8, 1)[0] = eps;
  return uniform(m, shape);
}
// (a buffer, or { buffer, offset, size }: a part of one, T.155's pieces of a matrix writing into their rows of the output)
// (a buffer's binding is its place among them; T.547.2.3: null, none there: the scales of a matrix whose blocks hold them)
const bind = (m: Model, pipeline: GPUComputePipeline, buffers: (Bound | null)[]) => m.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: buffers.flatMap((buffer, binding) => (buffer ? [{ binding, resource: buffer.buffer ? buffer : { buffer } }] : [])) });
function dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup, x: number, y = 1, z = 1) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(x, y, z);
}

export { STORAGE, COPY_DST, COPY_SRC, MAP_READ, UNIFORM, CHUNK, rowBytes, turnedAt, model, stopping, lost, stopAsked, handed, held, within, isLate, openDevice,
  describe, unusable, end, whyOf, buffer, uniform, normShape, copyIn, narrowIn, widenIn, widenedHalves, readBack, validated, pipelineOf, bind, dispatch };
export type { Model, Bound, Whole, Pass, Opening };
