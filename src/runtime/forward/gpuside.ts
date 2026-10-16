// forward/gpuside.ts (T349): the GPU's side of one engine: what the GPU would hold of the model and why it would not
// take it, the GPU's worker (src/runtime/gpu.ts) started and stopped, a prompt's blocks and a generation's steps through
// it, and which of them go there (T135, T148, T152, T156, T210).
// A module of src/runtime/forward.ts; it reads jobs.ts and its neighbours by static imports.

import { BATCH, GPU_DONE, GPU_FAILED, GPU_BEAT, GPU_WANTED, PACKED, packingOf } from "../jobs.ts";
import { GPU_END_MS, GPU_BLOCK, GPU_TOKENS, GPU_RECHECK, TOKEN_RECHECK, promptTimes, tokenTimes, PROMPTS_UNTIMED, PROMPTS_GPU, PROMPTS_CPU, gpuLine,
  STOPS_MOST } from "./choice.ts";
import { gpuWeightBytes, cpuReadBytes, aloneVerdict, OUTSIDE_VOCABULARY } from "./alone.ts";
import { REASONS } from "./reasons.ts";
import { halfToFloat } from "./memory.ts";
import type { Packed, Ready } from "../jobs.ts";
import type { Answers } from "./choice.ts";
import type { Control } from "./control.ts";
import type { Direct } from "./alone.ts";
import type { Convolution, Linear } from "../contracts/form.ts";
import type { CachePlace, Force, FromGpu, Matrix, Remembered, ToGpu, Tokens, Vectors } from "../contracts/gpu/protocol.ts";
import type { GpuChose } from "../contracts/messages.ts";
import type { Activation, LayerKind, Tensor } from "../contracts/plan.ts";

// ---- (T368.4, the types) what createForward() (engine.ts) makes once and hands the two functions of this file
// the model's facts
export type Shape = {
  dim: number; hidden: number; layers: number; heads: number; kvHeads: number; headSize: number; kvDim: number; qDim: number; vocab: number; seqLen: number;
  // the bytes of a token's row, and of a position's keys in the cache; whether the cache is float16
  D: number; KV: number; halfKV: boolean;
  eps: number; rotary: number; kvStart: number; turning: boolean; layerNorm: boolean; gatedFfn: boolean; parallel: boolean; unturned: number[];
  // T261: the FFN's activation, and whether some layer attends a window or is turned by tables of its own; T261.4: and
  // every layer's window (0: every position) and the layers turned by those tables; extras: what a Gemma 4 has besides
  // (a norm of v's heads, a cap of the logits, a scale of a layer, layers of other widths, keys and values shared)
  activation: Activation; local: boolean; windows: number[]; locals: number[]; extras: boolean;
  linear: Linear | null; convolution: Convolution | null;
  // T375.2: some layer keeps a state in place of keys and values, of whatever kind (the plan's layers say)
  stateful: boolean;
  // the block of a rotated basis (0: the model's own basis)
  rotated: number;
  sharedClassifier: boolean;
  // T545: every layer's kind and its place among the layers of its kind; how many layers keep keys and values in the
  // cache (every one, but an LFM2's convolution layers and a Qwen3.5's linear-attention layers); and the bytes of the
  // state the GPU's worker goes on from, as it goes up and comes back (an LFM2's convolution layers' rows, T271.2 a
  // Qwen3.5's linear-attention layers' states and rows; 0: none)
  kinds: LayerKind[]; places: number[]; attending: number; stateBytes: number;
};
// A matrix as createForward()'s matrix() made it: kind, how it is multiplied: packed (the name of its line of jobs.ts's
// PACKED, in groups of group) or float32; and for a layer the addresses of its values and, where it is packed, of its
// scales and its corrections (0: none). onGpu (T156): on the GPU alone, its addresses the checkpoint's offsets.
// T375.4: job and ready are the CPU's (the kind of the job that multiplies by it and what makes its input ready:
// engine.ts's matmuls()); this side reads neither
// (T363.5: layers, of a stack whose layers are each of a kind of their own: each layer's matrix, and kind is "layers")
export type Held = { rows: number; n: number; kind: "f32" | Packed | "layers"; group?: number; onGpu?: boolean; layer(l: number): number[]; job?: number; ready?: Ready | null; layers?: Held[] };
// T375.4, T375.5: whether the GPU takes a matrix as it is packed: its packing's line of jobs.ts's PACKED says how the GPU
// holds it (gpu.ts widens int6 to int8, T155, and takes ternary as it is, T232), in the groups that line says. A
// matrix that is not packed, or whose packing the GPU has not, stays on the CPU
const onGpuAs = (m: Held) => {
  const packing = packingOf(m.kind);
  return Boolean(packing?.gpu) && m.group === packing!.group;
};
// T545: the kind of the layers a matrix or a vector of the GPU's is of, where it is not every layer's (its places index
// those layers alone: an LFM2's attending layers are some of its layers). T271.2: a Qwen3.5's linear-attention layers'
// and the gate of its full-attention layers' output; wout and conv are the kind's that keeps a state, whichever the
// model has (an LFM2's convolution layers, a Qwen3.5's linear-attention layers: no model has both)
const KIND_OF: Partial<Record<string, LayerKind>> = { wq: "attention", wk: "attention", wv: "attention", wo: "attention", wg: "attention", bq: "attention",
  bk: "attention", bv: "attention", qNorm: "attention", kNorm: "attention", win: "convolution", wqkv: "linear", wz: "linear", wb: "linear", wa: "linear",
  dtBias: "linear", decay: "linear", deltaNorm: "linear" };
const kindOf = (name: string, { convolution, linear }: { convolution: Convolution | null; linear: Linear | null }): LayerKind | undefined =>
  KIND_OF[name] ?? (name === "wout" || name === "conv" ? (convolution ? "convolution" : linear ? "linear" : undefined) : undefined);
// T271.2: the widest key dim and value dim of a Qwen3.5's linear-attention layers the GPU's delta rule takes (a thread a
// value of a head, its state's column in the thread's own memory): shaders/steps.ts's DELTA_THREADS, a second copy
// here (no shader's text comes into this module's chunk), which tests/gpu-part-check.mjs holds to it
export const DELTA_MOST = 128;
// (T271.2: what a linear-attention layer's convolution runs over, q and k of the key heads and v of the value heads)
const mixedOf = (linear: Linear) => 2 * linear.key_heads * linear.key_dim + linear.value_heads * linear.value_dim;
// where the model's tensors are in the memory: the matrices (null: the model has none of that name), and the addresses
// of the vectors and the tables (0: none)
export type Placed = {
  base: number; embedding: Tensor;
  wq: Held | null; wk: Held | null; wv: Held | null; wo: Held | null; w1: Held | null; w2: Held | null; w3: Held | null; wcls: Held | null;
  attW: number; ffnW: number; attB: number; ffnB: number; finalW: number; finalB: number; bo: number; b1: number; b2: number; bq: number; bk: number; bv: number;
  qNorm: number; kNorm: number; cosTable: number; sinTable: number; positions: number; channels: number[];
  // T261: the norms behind the attention and behind the FFN (0: the model has none); T261.4: RoPE's tables of the
  // local layers (0: none)
  attPost: number; ffnPost: number; localCos: number; localSin: number;
  // T545: an LFM2's matrices in and out of its convolution layers, and their taps (null, 0: none). T271.2: a Qwen3.5's
  // linear-attention layers' matrices of q, k and v and of z, its matrix out and taps (wout, conv), their gates' small
  // matrices and vectors (float32), and the gate of its full-attention layers' output
  win: Held | null; wout: Held | null; conv: number; wqkv: Held | null; wz: Held | null; wg: Held | null;
  wb: number; wa: number; dtBias: number; decay: number; deltaNorm: number;
};
// what the GPU is to take of the model (gpuFit()'s matrices and vectors), why it would take none or no steps, and the
// places in the memory it answers into (0: none)
export type Taken = { matrices(): Record<string, Held>; vectors(): Vectors; whyNot: string | null; tokensWhyNot: string | null; staging: number; rows: number; ids: number };
// the CPU's side: the memory and its views made anew, the cache of the keys and values as it is now and grown, and what
// the CPU does for a block
export type CpuSide = {
  memory: WebAssembly.Memory;
  views(): void;
  grow(pos: number): void;
  embed(tokens: number[], pos0: number, rows?: number, stride?: number): void;
  fromStaging(pos: number, count: number): void;
  stagingFinite(count: number): boolean;
  run(tokens: number[], pos0: number, needLogits: boolean): void;
  // T545: whether a block at pos goes on from the state of the layers that keep one, and that state taken from what the
  // GPU's worker left of a block (next: the position that comes after it)
  follows(pos: number): boolean;
  stateToStaging(): void;
  stateFromStaging(next: number | null, taken?: boolean): void;
  // T261.4.2: where each attending layer's keys and values are now (a block, or a ring: a step's go up from there)
  places(): CachePlace[];
  readonly H: Uint16Array; readonly capacity: number;
};
// The GPU's worker as this side has it: a Worker (gpu.ts's), or what posts and listens as one
export type GpuWorker = { postMessage(message: ToGpu): void; onmessage: ((event: MessageEvent<FromGpu>) => void) | null; onerror: ((event: ErrorEvent) => void) | null; terminate?(): void };
// what the GPU chose as it started, as the page is shown it and keeps it
export type Chosen = Omit<GpuChose, "type" | "load" | "note">;
// ids as Python hands them (a PyProxy of a list, which toJs() copies) or as JavaScript does; none: an empty list
type Listed = (Iterable<number> & { toJs?(): number[] }) | null | undefined;
// (the control area's words and the GPU's worker are read where the GPU is on and nowhere else: ctl! and gpuWorker! below)

// T135: the GPU's worker says something at least this often while it puts a model on the GPU: after every step, each
// of which it gives up itself after 180 s (gpu.ts's STEP_MS, T147: SwiftShader compiles a shader in up to 90 s). A
// worker quiet for longer than that is one the browser ended
const GPU_QUIET_MS = 200000;
// T147: the number of the GPU's requests, for this worker and every model it loads (the memory and its control area
// are kept from model to model, T96): a request of an engine let go is never one of the next engine's
let gpuRequests = 0;

/** T349: what the GPU would hold of one engine's model, and why it would not take it (createForward() asks before it
 * allocates the places the GPU writes to).
 * T378.2: shape, the model's facts, and weights, where its tensors are in the memory (createForward() makes both once,
 * for this and for gpuSide()); device: { gpuRoom, memoryUnsaid, sharedMemory }, what the device and the page give;
 * direct: the model is on the GPU alone (T156). */
export function gpuFit({ shape, weights, device, direct }:
  { shape: Shape; weights: Placed; device: { gpuRoom: number | undefined; memoryUnsaid: boolean | undefined; sharedMemory: boolean }; direct: Direct | undefined }) {
  const { dim, layers, headSize, vocab, seqLen, hidden, kvDim, qDim, D, layerNorm, gatedFfn, activation, local, extras, linear, convolution, stateful, rotated,
    sharedClassifier, kinds, attending, stateBytes } = shape;
  const { embedding, wq, wk, wv, wo, w1, w2, w3, wcls, attW, ffnW, attB, ffnB, bo, b1, b2, bq, bk, bv, qNorm, kNorm, positions, attPost, ffnPost, win, wout,
    conv, wqkv, wz, wg, wb, wa, dtBias, decay, deltaNorm } = weights;
  // T545: how many layers a matrix or a vector of that name is of (every layer, or those of its kind: KIND_OF)
  const countOf = (name: string) => (kindOf(name, shape) ? kinds.filter((kind) => kind === kindOf(name, shape)).length : layers);
  const { gpuRoom, memoryUnsaid, sharedMemory } = device;
  // why this model's prompt stays on the CPU, or null: the first stage (T135) takes Llama's layers of int8 weights
  // (T153: with Qwen2's biases, Qwen3's norms of the heads and heads of another size than dim / heads as well; T154:
  // GPT-2's and GPT-NeoX's as well, whose FFN has no gate: w3 is null; T155: int6 weights and 64-bit memories too)
  // (a function, not a const: gpuUnfit runs before this line, AGENTS.md)
  function gpuMatrices() {
    return Object.fromEntries(Object.entries({ wq, wk, wv, wo, w1, w2, w3, win, wout, wqkv, wz, wg }).filter(([, m]) => m)) as Record<string, Held>;
  }
  // (T155: int6 weights too, widened to int8 on the GPU, and a model in a 64-bit memory: gpu.ts; T232: ternary weights
  // as they are, in their groups of 128. The 27B, T233, is ternary in a rotated basis with linear-attention layers:
  // either of the first two lines keeps it on the CPU)
  function gpuUnfit() {
    // (T229: a Qwen3.5's linear-attention layers were kept off here until T271.2, which gave the GPU's worker a block's
    // step for them; a key dim or a value dim wider than its delta rule takes, or value heads that are no whole multiple
    // of the key heads, stays on the CPU)
    if (linear && (linear.key_dim > DELTA_MOST || linear.value_dim > DELTA_MOST || linear.value_heads % linear.key_heads)) return REASONS.linear;
    // (T260: an LFM2's convolution layers were kept off here until T545, which gave the GPU's worker a step for them and
    // the state they go on from. T375.2: and a kind of layer that keeps a state which the engine comes to have, and the
    // GPU's worker has no step for: on the CPU until the GPU has it too)
    if (stateful) return REASONS.stateful;
    if (rotated) return REASONS.rotated;  // T237
    // T375.5 (T484): the GPU's layers are of two pairs, RMSNorms with a gated FFN (a Llama's) and LayerNorms with
    // GPT-2's FFN (T154); gpu.ts picks between them by the norms and by whether there is a third matrix. An
    // architecture of another pair, which the CPU's engine runs by its facts (T375.2), stays there
    if (layerNorm === gatedFfn) return REASONS.pair;
    // T261.4: and what a Gemma 3 has besides, which the GPU's layers have now: a GELU in the gate of its FFN (GEGLU), a
    // norm behind the attention and behind the FFN (RMSNORM_ADD), layers that attend a window (the flash tiles' window)
    // or are turned by tables of their own (a second table of angles). Not an activation no shader has (a Spark-X2.5's
    // exact GELU), nor norms behind LayerNorms, nor what a Gemma 4 has besides (T261: until T261.4 a Gemma 3 was kept
    // off by this line too). An FFN without a gate is GELU's on the GPU, as it was (T154)
    if ((gatedFfn && activation !== "silu" && activation !== "gelu") || extras) return REASONS.pair;
    if ((attPost || ffnPost) && layerNorm) return REASONS.pair;
    if (!sharedMemory) return REASONS.notIsolated;
    if (headSize % 4) return REASONS.heads;
    if (!Object.values(gpuMatrices()).every(onGpuAs)) return REASONS.floats;
    // T148: the layers twice, in this memory and on the GPU (T156 will keep one): a device with too little memory
    // for both keeps the CPU's alone (a phone or an Apple shares its memory between the two)
    // (T153, the review: with the GPU's own keys and values, float16, as the prompt may fill the whole context: Qwen3
    // 0.6B's are 0.47 GB at 4096 positions, 95% of its layers')
    const onGpu = layersOnGpu();
    if (gpuRoom !== undefined && onGpu > gpuRoom && !direct) return REASONS.layersRoom(onGpu);
    return null;
  }
  // the bytes of the layers on the GPU, and its own keys and values
  function layersOnGpu() {
    // (T545: each of the layers it is of, and the keys and values of the attending ones; and the state the convolution
    // layers go on from, with its copy where a block's comes back; T271.2: the linear-attention layers' too)
    return Object.entries(gpuMatrices()).reduce((bytes, [name, m]) => bytes + countOf(name) * m.rows * m.n * gpuWeightBytes(m.kind), 0) +
      Object.entries(gpuVectors()).reduce((bytes, [name, { size }]) => bytes + countOf(name) * size * 4, 0) + 2 * attending * seqLen * kvDim * 2 +
      2 * stateBytes;
  }
  // T152: why a generation's steps stay on the CPU, or null. A step on the GPU is T150's and T175's fused layer (gpu.ts):
  // Llama's (RMSNorm, RoPE on whole heads, SwiGLU), and T226: with what the prompt's blocks take besides (T153: Qwen2's
  // biases of q, k and v, Qwen3's norms of the heads and heads of another size than dim / heads; T154: GPT-2's and
  // GPT-NeoX's LayerNorm, biases, GELU, learned positions, RoPE on a part of a head, parallel residual), and T92's
  // outlier channels (the GPU's classifier multiplies floats for such a model, and needs no columns apart). The keys
  // and values are float16 as the GPU's, or float32 where the CPU keeps them so (T160, widened on the way back and
  // narrowed on the way up). What is left: a classifier
  // and an embedding of int8 or int6 in groups of 32 (T232: or ternary in groups of 128; of a ternary classifier the
  // GPU takes the outlier channels apart as this does, shaders.ts's TAKE_OUTLIERS); and the memory for the classifier, the embedding where it is
  // another table, RoPE's table and the vocabulary's three arrays of the sampling, besides the layers
  function tokensUnfit() {
    // (T545: an LFM2's steps stayed on the CPU here until T545.2, which gave a token's fused layer its convolution layers)
    // (T271.2: a Qwen3.5's steps stayed on the CPU here until T271.3, which gave a token's layer its linear-attention layers)
    // (the embedding as the GPU holds the classifier: in the same packing there, and in its groups)
    const held = wcls && onGpuAs(wcls) ? packingOf(wcls.kind)!.gpu!.as : null;
    if (!wcls || !held || packingOf(embedding.kind)?.gpu?.as !== held || embedding.group !== PACKED[held].group) {
      return REASONS.floatClassifier;
    }
    // T205: the classifier and the embedding on the GPU as well (llm-jp-3 150M's 73 MB of layers came to 189 MB) where
    // the browser does not say what the device has: an iPhone's tab went down in /benchmark/'s model section. The
    // prompts' blocks still go (their layers alone)
    if (memoryUnsaid) return REASONS.memoryUnsaid;
    const table = vocab * dim * gpuWeightBytes(wcls.kind);
    // (GPT-2's positions on the GPU as well, a row a position)
    // (T261.4.2: a Gemma 3's local layers' table of RoPE as well)
    // (T545.2: and the state as the last step of a request leaves it, read back with its ids)
    const onGpu = layersOnGpu() + table * (sharedClassifier ? 1 : 2) + (local ? 2 : 1) * seqLen * headSize * 4 + 3 * vocab * 4 + (positions ? seqLen * D : 0) +
      stateBytes;
    if (gpuRoom !== undefined && onGpu > gpuRoom) return REASONS.classifierRoom(onGpu);
    return null;
  }
  // the vectors of every layer the GPU reads (gpu.ts's plan.vectors): the norms' weights; T153: Qwen2's biases of q,
  // k and v, Qwen3's norms of a head of q and of k; T154: GPT-2's and GPT-NeoX's biases of the two LayerNorms, of q,
  // k and v and of o, w1 and w2. Each one's address here and its floats a layer
  function gpuVectors(): Vectors {
    return { attention: { at: attW, size: dim }, ffn: { at: ffnW, size: dim },
      ...(layerNorm ? { attentionBias: { at: attB, size: dim }, ffnBias: { at: ffnB, size: dim } } : {}),
      ...(bq ? { bq: { at: bq, size: qDim }, bk: { at: bk, size: kvDim }, bv: { at: bv, size: kvDim } } : {}),
      ...(qNorm ? { qNorm: { at: qNorm, size: headSize }, kNorm: { at: kNorm, size: headSize } } : {}),
      ...(bo ? { bo: { at: bo, size: dim }, b1: { at: b1, size: hidden }, b2: { at: b2, size: dim } } : {}),
      ...(attPost ? { attentionPost: { at: attPost, size: dim } } : {}), ...(ffnPost ? { ffnPost: { at: ffnPost, size: dim } } : {}),
      // T545: an LFM2's taps, taps rows of dim a convolution layer (T271.2: a Qwen3.5's, taps rows of the mixed q, k and v
      // a linear-attention layer), and T271.2 a Qwen3.5's gates' two small matrices (a row of dim a value head), their
      // biases, decays and the norm of a value head
      ...(conv ? { conv: { at: conv, size: convolution ? convolution.taps * dim : linear!.conv * mixedOf(linear!) } } : {}),
      ...(wb ? { wb: { at: wb, size: linear!.value_heads * dim }, wa: { at: wa, size: linear!.value_heads * dim }, dtBias: { at: dtBias, size: linear!.value_heads },
        decay: { at: decay, size: linear!.value_heads }, deltaNorm: { at: deltaNorm, size: linear!.value_dim } } : {}) };
  }
  // T545: the same, each with the kind of the layers it is of where it is not every layer's (KIND_OF), for the plan
  function gpuVectorsOf(): Vectors {
    return Object.fromEntries(Object.entries(gpuVectors()).map(([name, vector]) => [name, kindOf(name, shape) ? { ...vector, of: kindOf(name, shape) } : vector])) as Vectors;
  }
  return { matrices: gpuMatrices, vectors: gpuVectorsOf, unfit: gpuUnfit, tokensUnfit };

}

/** T349: the GPU's side of one engine (createForward(), which hands in what it reads of it): the GPU's worker started
 * and stopped, a prompt's blocks and a generation's steps through it, and which of them it takes (T135, T148, T152,
 * T156). Returns what the rest of the engine tells it and asks it (the end of this function has the list).
 * T378.2: what it is handed is seven things, each taken apart here once (the names in the functions below are locals):
 *   shape    the model's facts: its sizes, its kind of norm and of residual, what RoPE turns
 *   weights  where the model's tensors are in the memory: addresses, and the matrices as createForward() made them
 *   taken    what the GPU is to take of this model: gpuFit()'s matrices and vectors, why it would take none (whyNot)
 *            or no steps (tokensWhyNot), and the places in the memory it answers into (staging, rows, ids: 0 for none)
 *   cpuSide  the CPU's side: the memory, its views made anew (views), the cache of the keys and values as it is now
 *            (cpuSide.H, .capacity, .keys, .values: read each time, they move as the cache grows) and grown (grow),
 *            and what the CPU does for a block (embed, fromStaging, stagingFinite, run)
 *   control  the control area (control.js): the words the GPU's worker answers in, and the wait on one
 *   threads  the software threads' (threads.js): threads.count and threads.searching are read
 *   options  { gpu: makes the GPU's worker (none: no WebGPU here), remembered, force (gpuForce), direct (T156),
 *            stalledMs, now (tests only: the clock the blocks and steps are timed by) } */
export function gpuSide({ shape, weights, taken, cpuSide, control, threads, options }: {
  shape: Shape; weights: Placed; taken: Taken; cpuSide: CpuSide; control: Control; threads: { readonly count: number; readonly searching: boolean };
  options: { gpu: (() => GpuWorker) | undefined; remembered: Remembered | undefined; force?: Force; direct: Direct | undefined; stalledMs: number; now?: () => number } }) {
  const { dim, hidden, layers, heads, kvHeads, headSize, kvDim, vocab, seqLen, D, KV, halfKV, eps, rotary, kvStart, turning, layerNorm, parallel,
    unturned, sharedClassifier, activation, windows, locals, convolution, linear, kinds, places, attending, stateBytes } = shape;
  const { base, embedding, wo, wcls, finalW, finalB, cosTable, sinTable, positions, channels, localCos, localSin } = weights;
  const { matrices: gpuMatrices, vectors: gpuVectors, whyNot: gpuWhyNot, tokensWhyNot, staging, rows: gpuRows, ids: gpuIds } = taken;
  const { memory, views, grow, embed, fromStaging, stagingFinite, run, follows, stateToStaging, stateFromStaging } = cpuSide;
  const { ctl, waitUntil } = control;
  const { gpu, remembered: gpuRemembered, force: gpuForce = {}, direct, stalledMs, now = () => performance.now() } = options;
  // every block and step the GPU can take goes there, whatever the CPU's time: the tests' fallback adapter, and a model
  // on the GPU alone (T156)
  const always = Boolean(gpuForce.always || direct);
  // T243: what the status line says of a GPU stopped for keys or values that are no finite numbers (where: the request)
  // ---- T135: a prompt on the GPU. gpu() makes the GPU's worker (gpu.ts), which puts this model's layers on the GPU
  // once (a second copy of them: the CPU keeps its own, for the tokens after the prompt), and then runs each block of a
  // prompt that forwardMany() hands it through all the layers in one submission, while this thread waits for its
  // answer in the control area (Python calls forwardMany() and cannot wait for a promise; the GPU's answer is one). The
  // GPU multiplies a block's tokens by each weight it reads once (a matrix by a matrix: T134's "A prompt"). It writes the
  // keys and values of the block back (staging), and they go into the cache here as if the CPU had computed them: the
  // prompt's last token, with its logits, and every token after it stay the CPU's (T94: one token at a time the GPU was
  // slower on the owner's three devices). A model it does not take, or any failure, leaves the prompt on the CPU, said
  // once. gpuEnd: the positions up to which the GPU's own keys and values are the cache's (a block that begins after it
  // goes to the CPU; run() ends it where the CPU writes).
  // T148: by default (AGENTS.md's policy 9), and without waiting for it: the model is ready on the CPU at once, and the
  // GPU takes the blocks of a prompt from the first one after it is ready (a block past gpuEnd stays on the CPU, so a
  // GPU that is ready in the middle of a prompt changes nothing of it). Which blocks it takes: promptTimes (choice.ts).
  // T205: gpuEnded, settled once the last GPU's worker made has let go of its buffers and its device (or never began)
  let gpuEnded: Promise<void> = Promise.resolve(), gpuLast: GpuWorker | null = null, quietTimer: ReturnType<typeof setTimeout> | undefined;
  let gpuWorker: GpuWorker | null = null, gpuOn = false, gpuEnd = 0, gpuSerial = 0, gpuTokens = 0, settleGpu: ((note: string) => void) | null = null,
    gpuChosen: Chosen | null = null;
  const gpuGettingReady = () => settleGpu !== null;
  // T148: what the status line says of the GPU now, and whether this generation checks the side not chosen again
  // recheck: the side not chosen that a part of the next long prompt goes to, to time it again ("cpu": the last
  // 2 × BATCH tokens of the prompt, "gpu": its first GPU_BLOCK; a GPU, then the CPU, is the order a prompt may take),
  // every GPU_RECHECK generations after the first verdict (sinceCheck), until it is timed (the review of T148: every
  // block on the other side put a GPU faster from 17 to 64 tokens on blocks of 16, 1.47 times as long, and never
  // timed the CPU). written: the positions this generation has filled
  let gpuStatus: string | null | undefined = gpu ? null : undefined, recheck: "cpu" | "gpu" | null = null, sinceCheck = 0, written = 0;
  // T184: why the prompts stay on the CPU (the model, the device, a failure), once it is known; and the side the
  // benchmark puts every block of a prompt on ("cpu", "gpu"), or null: the choice above
  let gpuReason: string | null = null, gpuSide: "cpu" | "gpu" | null = null;
  // T156: why the GPU stopped under a model on the GPU alone, once it has
  let directLost: string | null = null;
  // T271.2: the position the GPU's own state of the layers that keep one is at, where it is the one the CPU's goes on
  // from (-1: it is not: the GPU's worker takes the CPU's up first, staged). A request that goes on from it puts nothing
  // up (fresh); the CPU writing anything, a request refused or given up, or a GPU's state that is no position's (steps
  // that stopped before their last) makes it -1. The GPU's worker clears its own at position 0 whatever this says
  let gpuStateAt = -1;
  // whether a request at pos goes on from the GPU's own state (else the CPU's is put where the GPU's worker takes it up)
  function stateUp(pos: number) {
    const fresh = direct ? true : !stateBytes || pos === 0 || gpuStateAt === pos;
    gpuStateAt = -1;
    if (!fresh) stateToStaging();
    return fresh;
  }
  const times = promptTimes();
  // T152: a generation's steps. tokensOn: the GPU takes them where it is faster (steps: their times); tokenStatus: what
  // the status line says of them (gpuLine's answers); tokenRecheck, cpuRecheck, sinceTokens: the side not chosen timed again now and then
  // (the CPU's first TOKEN_RECHECK steps of a generation, or the GPU's first run), every GPU_RECHECK generations after
  // the first verdict, as a prompt's; gpuSampled: the steps the GPU took since the generation began
  let tokensOn = false, tokenStatus: Answers | null = null, tokensReason: string | null | undefined = tokensWhyNot, tokenRecheck: "cpu" | "gpu" | null = null,
    cpuRecheck = 0, sinceTokens = 0, gpuSampled = 0;
  const steps = tokenTimes();
  // what the status line says of the GPU: of the prompts, and of the answers where there is something to say (gpuLine)
  const statusNow = () => gpuLine(gpuStatus, tokenStatus);
  const gpuNote = !gpu ? undefined : new Promise<string>((resolve) => {
    settleGpu = (note) => {
      settleGpu = null;
      gpuStatus = note;
      resolve(statusNow()!);
    };
    // (T156: a model on the GPU alone that the GPU cannot take, or whose steps it cannot, is lost at once)
    const why = gpuWhyNot ?? (direct ? tokensWhyNot : null);
    if (why) stopGpu(why);
    else startGpu();
  });
  // T152: what gpu.ts takes for a generation's steps: the classifier and (where it is another table) the embedding,
  // { rows, n, six, ternary (T232), at: [values, scales] }, the final norm's weights, where the ids go, and the steps a submission.
  // T226: the final LayerNorm's bias and GPT-2's positions (float32, a row a position), 0 where the model has none; and
  // whether its classifier has outlier channels (T92), and (T232) which they are
  function gpuTokensPlan(): Tokens {
    return { classifier: { rows: wcls!.rows, n: wcls!.n, six: wcls!.kind === "int6", ternary: wcls!.kind === "ternary", at: wcls!.layer(0).slice(0, 2) as [number, number] },
      embedding: sharedClassifier ? null
        : { rows: vocab, n: dim, six: embedding.kind === "int6", ternary: embedding.kind === "ternary", at: [base + embedding.offset, base + embedding.scales] },
      final: finalW, finalBias: finalB, positions, outliers: channels.length > 0, channels, ids: gpuIds, most: GPU_TOKENS };
  }
  function startGpu() {
    // the values of a head RoPE turns: all of them, GPT-NeoX's first rotary (T154), none of GPT-2's
    const turned = !turning ? 0 : rotary > 0 && rotary < headSize ? rotary : headSize;
    // (six, T155: the values at a layer's address are int6, packed as llama2_numpy.pack6 packs them; ternary, T232:
    // two bits a weight, as llama2_numpy.pack_ternary packs them, which the GPU takes as they are)
    // (T545: a matrix of some of the layers, an LFM2's attending or convolution layers', is theirs alone: of says whose)
    const matrices = Object.fromEntries(Object.entries(gpuMatrices()).map(([name, m]): [string, Matrix] => {
      const of = kindOf(name, shape), count = of ? kinds.filter((kind) => kind === of).length : layers;
      return [name, { rows: m.rows, n: m.n, six: m.kind === "int6", ternary: m.kind === "ternary",
        layers: Array.from({ length: count }, (_, l) => m.layer(l).slice(0, 2) as [number, number]), ...(of ? { of } : {}) }];
    }));
    const listen = () => {
      clearTimeout(quietTimer);
      quietTimer = setTimeout(() => stopGpu(REASONS.quiet(GPU_QUIET_MS)), GPU_QUIET_MS);
    };
    const worker = gpuLast = gpuWorker = gpu!();
    let ended: () => void;
    gpuEnded = new Promise<void>((resolve) => { ended = resolve; });
    gpuStatus = "prompts on the CPU while the GPU gets ready";
    worker.onmessage = ({ data }) => {
      // T205: its buffers and its device let go (gpu.ts's end()), and nothing more from it after a stop
      if (data.type === "ended") return ended();
      if (worker !== gpuWorker) return;
      if (data.type === "progress") return listen();
      clearTimeout(quietTimer);
      if (data.type === "ready") {
        gpuOn = true;
        if (data.blocks.length) times.started(data.blocks);
        gpuChosen = { matrices: data.form, attention: data.attention, key: data.key, remembered: data.remembered, seconds: data.seconds };
        console.info(`gpu: ${data.adapter}: ${Math.round(data.bytes / 1e6)} MB of layers on it in ${data.seconds.toFixed(1)} s`);
        // T147: the matrices' shader this device runs fastest of those that are right here, and what the others came to
        const forms = data.forms.map((f) => `${f.name} ${f.none ?? (f.remembered ? "remembered" : f.ms ? `${f.ms.toFixed(1)} ms` : "untimed")}`).join("; ");
        const blocks = data.blocks.map(({ count, ms }) => `${count} tokens ${ms.toFixed(1)} ms`).join(", ");
        console.info(`gpu: the matrices by ${data.form}, the attention by ${data.attention} (a pass of the first layer by ${GPU_BLOCK} tokens: ${forms}; a whole block: ${blocks})`);
        // T152: a generation's steps, where gpu.ts made them (else why not)
        if (data.tokens) {
          tokensOn = true;
          gpuChosen.tokens = data.tokens.form;
          gpuChosen.tokenAttention = data.tokens.attention;  // T224: the attention of a token chosen here (tests)
          gpuChosen.tablePieces = data.tokens.pieces;  // T209: the classifier in pieces past what the device binds
          if (data.tokens.ms !== undefined) steps.gpu(data.tokens.ms);
          tokenStatus = always ? "gpu" : "untimed";
          const kinds = data.tokens.forms.map((f) => `${f.name} ${f.none ?? (f.remembered ? "remembered" : f.ms ? `${f.ms.toFixed(2)} ms` : "untimed")}`).join("; ");
          const attentions = (data.tokens.attentions ?? []).map((a) => `${a.name} ${a.none ?? (a.ms ? `${a.ms.toFixed(2)} ms` : "untimed")}`).join("; ");
          console.info(`gpu: a token by ${data.tokens.form} (a step of a run of ${GPU_TOKENS}: ${kinds}), its attention by ${data.tokens.attention} (${attentions})`);
        } else if (data.tokensWhy) {
          tokensReason = data.tokensWhy;
          tokenStatus = "why";
          console.info(`gpu: answers on the CPU (${tokensReason})`);
        } else if (tokensReason) {
          // (T152's review: a model whose steps were not asked of the GPU says why too, in the console alone: the
          // owner's words have the status line say no reason for the answers)
          tokenStatus = "why";
          console.info(`gpu: answers on the CPU (${tokensReason})`);
        }
        // T156: a model on the GPU alone needs its steps there too; and where /benchmark/ measured the CPU here, its
        // step (the checkpoint read once at the CPU section's fastest, T157's estimate) must not be faster by more
        // than BETTER's margin (the owner's B: the CPU's side estimated, not measured, for a model that does not fit twice)
        if (direct && !data.tokens) return stopGpu(data.tokensWhy ?? tokensReason ?? REASONS.noSteps);
        // (the owner, 2026-09-27: the prompts as well, which the GPU runs about five times as fast, weighed by use)
        if (direct) {
          const block = data.blocks.find((b) => b.count === GPU_BLOCK);
          const verdict = aloneVerdict({ size: cpuReadBytes(direct.size, wo!.kind), layerWeights: direct.layerWeights, cpu: direct.cpu, usage: direct.usage,
            gpu: { stepMs: data.tokens!.ms, promptMs: block ? block.ms / block.count : undefined } });
          if (verdict.cpuFaster) {
            direct.verdict = { key: data.key, cpu: direct.cpu! };  // the page keeps it: the next load goes on the CPU at once
            return stopGpu(REASONS.cpuFaster(direct.cpu!.GBps, verdict.cpu, verdict.gpu, direct.usage.prompt, direct.usage.written));
          }
        }
        settleGpu?.(always ? PROMPTS_GPU : PROMPTS_UNTIMED);
      } else if (data.type === "unusable") {
        stopGpu(data.reason);
      } else if (data.type === "failed") {
        console.warn(`gpu: ${data.reason}`);
      }
    };
    worker.onerror = (event) => {
      ended();  // a worker that did not start holds nothing
      if (worker === gpuWorker) stopGpu(REASONS.notStarted(event.message));
    };
    listen();
    // T154: LayerNorm's epsilon is the CPU's layernorm kernel's and NumPy's, 1e-5 (GPT-2's and GPT-NeoX's
    // layer_norm_epsilon); parallel: GPT-NeoX's parallel residual
    worker.postMessage({ type: "start", memory, plan: { dim, hidden, layers, heads, kvHeads, headSize, turned, unturned, seqLen,
      kvStart, eps: layerNorm ? 1e-5 : eps, layerNorm, parallel: Boolean(parallel), activation, windows, local: locals, localCos, localSin,
      kinds, places, convolution: convolution ? { taps: convolution.taps } : null,
      linear: linear ? { keyHeads: linear.key_heads, valueHeads: linear.value_heads, keyDim: linear.key_dim, valueDim: linear.value_dim, taps: linear.conv } : null,
      batch: GPU_BLOCK,
      matrices,
      vectors: gpuVectors(), rows: gpuRows, tokens: gpuIds ? gpuTokensPlan() : null,
      cos: cosTable, sin: sinTable, staging, force: gpuForce, remembered: gpuRemembered, direct: Boolean(direct),
      words: { done: GPU_DONE, failed: GPU_FAILED, beat: GPU_BEAT, wanted: GPU_WANTED } } });
  }
  // the prompt stays on the CPU from here on (why: what the console says, where it was on the GPU); the GPU's worker
  // lets go of the device and ends
  function stopGpu(why?: string) {
    clearTimeout(quietTimer);
    if (gpuOn && why) console.warn(`gpu: ${why}: the prompts and the tokens go on on the CPU`);
    // T156: the layers were on the GPU alone: the worker loads the model again on the CPU (not when it is let go)
    if (direct && why && !directLost) {
      directLost = why;
      direct.onLost?.(why);
    }
    gpuOn = false;
    tokensOn = false;
    tokenStatus = null;
    if (ctl) Atomics.store(ctl, GPU_WANTED, 0);  // T147: a request still under way writes nothing now
    gpuWorker?.postMessage({ type: "stop" });
    gpuWorker = null;
    gpuReason = why ?? REASONS.letGo;
    const note = `prompts on the CPU (${gpuReason})`;
    if (settleGpu) settleGpu(note);
    else if (gpu) gpuStatus = note;
  }
  // T148: whether a block of count tokens at pos0 goes to the GPU: where the GPU holds the keys and values before it,
  // and is faster for count tokens than the CPU on the threads in use now (promptTimes); and the first whole block of
  // a prompt where the GPU is to be timed again (recheck)
  function gpuTakes(count: number, pos0: number) {
    // (T545: and where the state of the layers that keep one is that of the position before it, as the CPU's would be:
    // else the CPU says it is not)
    if (!gpuOn || pos0 > gpuEnd || gpuSide === "cpu" || !follows(pos0)) return false;
    if (always || gpuSide === "gpu") return true;
    const known = times.of(count, threads.count);
    if (!known) return false;  // the CPU is timed first, on this prompt
    return known.faster || (recheck === "gpu" && pos0 === 0 && count === GPU_BLOCK);
  }
  // T148: the verdict for a whole block, as a prompt begins: the status line has it (the fewest tokens the GPU takes,
  // rounded up to a block of the CPU's, and none said up to one: the line does not move with every prompt), and the
  // console says it where it changes. Until the CPU is timed, the line stays as the GPU left it
  function verdict() {
    const most = times.of(GPU_BLOCK, threads.count), from = times.threshold(GPU_BLOCK, threads.count);
    if (!most || always) return;
    const status = from! > GPU_BLOCK ? PROMPTS_CPU
      : from! > BATCH ? `prompts of ${Math.ceil(from! / BATCH) * BATCH} tokens and more on WebGPU` : PROMPTS_GPU;
    if (status !== gpuStatus) {
      console.info(`gpu: a block of ${GPU_BLOCK} tokens: ${most.gpu.toFixed(1)} ms on the GPU, ${most.cpu.toFixed(1)} ms on the ` +
        `CPU (${threads.count} thread${threads.count > 1 ? "s" : ""}), the GPU from ${from! > GPU_BLOCK ? "no count" : `${from} tokens`}: ${status}`);
    }
    gpuStatus = status;
  }
  // A block of a prompt (up to GPU_BLOCK tokens at pos0, pos0 + 1, ...) through the layers on the GPU: false where it
  // must go to the CPU instead (no GPU, keys and values the GPU does not have, a failure)
  function promptOnGpu(tokens: number[], pos0: number) {
    const count = tokens.length, began = now();
    if (!direct && pos0 + count - 1 >= cpuSide.capacity) grow(pos0 + count - 1);
    views();
    // the GPU reads the rows from there (T210: on the GPU alone, it embeds the tokens itself)
    if (!direct) embed(tokens, pos0, gpuRows, D);
    const fresh = stateUp(pos0);
    gpuSerial = ++gpuRequests;
    Atomics.store(ctl!, GPU_WANTED, gpuSerial);
    gpuWorker!.postMessage({ type: "prompt", serial: gpuSerial, count, pos: pos0, fresh, ...(direct ? { tokens } : {}) });
    if (!waitUntil(GPU_DONE, (seen) => seen === gpuSerial, GPU_BEAT)) {
      stopGpu(REASONS.stalled(stalledMs));
      return false;
    }
    if (Atomics.load(ctl!, GPU_FAILED)) {
      stopGpu(REASONS.blockFailed);  // the GPU's worker said why in the console
      return false;
    }
    views();
    // T243: a key or a value that is no finite number: the block is refused as a step with an id outside the vocabulary
    // is (T219): nothing of it written, the GPU's positions not counted as the cache's, the GPU stopped, and the CPU
    // takes the block from its own keys and values (forwardMany). T210: a model on the GPU alone reads none back (its
    // keys and values stay on the GPU, where T219's flag on the logits of the steps is what guards them)
    if (!direct && !stagingFinite(count)) {
      stopGpu(REASONS.keysInBlock(pos0));
      return false;
    }
    if (!direct) fromStaging(pos0, count);
    // (T545: the state the convolution layers left, which the next block or token goes on from; T271.2: the GPU's own is
    // that position's too)
    stateFromStaging(pos0 + count, !direct);
    gpuStateAt = pos0 + count;
    gpuEnd = pos0 + count;
    gpuTokens += count;
    times.gpu(count, now() - began);
    if (recheck === "gpu") recheck = null;
    return true;
  }
  // T210: the GPU's own keys and values of count positions from pos (a model on the GPU alone keeps none here), read
  // back GPU_BLOCK positions a request through staging: in float32, [layers][count][kvDim] each, as keysAndValues
  function gpuKeysAndValues(pos: number, count: number) {
    // (T545: the attending layers', by their places)
    const out = [0, 1].map(() => new Float32Array(attending * count * kvDim));
    for (let at = 0; at < count; at += GPU_BLOCK) {
      const part = Math.min(GPU_BLOCK, count - at);
      if (!gpuOn) throw new Error(`The GPU stopped (${directLost ?? gpuReason}), and this model's keys and values were on it alone`);
      gpuSerial = ++gpuRequests;
      Atomics.store(ctl!, GPU_WANTED, gpuSerial);
      gpuWorker!.postMessage({ type: "keys", serial: gpuSerial, count: part, pos: pos + at });
      if (!waitUntil(GPU_DONE, (seen) => seen === gpuSerial, GPU_BEAT) || Atomics.load(ctl!, GPU_FAILED)) {
        throw new Error("The GPU did not read its keys and values back");
      }
      views();
      for (let side = 0; side < 2; side++) {
        for (let l = 0; l < attending; l++) {
          for (let t = 0; t < part; t++) {
            const from = (staging + ((side * attending + l) * GPU_BLOCK + t) * kvDim * 2) / 2, to = (l * count + at + t) * kvDim;

            for (let i = 0; i < kvDim; i++) out[side][to + i] = halfToFloat(cpuSide.H[from + i]);
          }
        }
      }
    }
    return { keys: out[0], values: out[1] };
  }
  // T148: a block of a prompt on the CPU, BATCH at a time (T108), each whole one timed where the GPU is there to
  // weigh against (not while it starts: its upload and compilation share the CPU and the memory)
  function promptOnCpu(tokens: number[], pos0: number) {
    for (let i = 0; i < tokens.length; i += BATCH) {
      const piece = tokens.slice(i, i + BATCH), began = now();
      run(piece, pos0 + i, false);
      if (gpuOn && piece.length === BATCH) {
        times.cpu(threads.count, (now() - began) / BATCH);
        if (recheck === "cpu") recheck = null;
      }
    }
  }

  // T152: the verdict for a generation's steps, as a generation begins: the status line has it (it changes only there),
  // and the console says it where it changes
  function tokenVerdict() {
    const known = tokensOn && !always ? steps.of(threads.count) : null;
    if (!known) return;
    const status = known.faster ? "gpu" : "cpu";
    if (status !== tokenStatus) {
      console.info(`gpu: a step of a generation: ${known.gpu.toFixed(2)} ms on the GPU, ${known.cpu.toFixed(2)} ms on the CPU ` +
        `(${threads.count} thread${threads.count > 1 ? "s" : ""}): answers on ${known.faster ? "WebGPU" : "the CPU"}`);
    }
    tokenStatus = status;
  }
  // T152: whether the steps of a generation go to the GPU now (see tokenBlock)
  function gpuSteps() {
    if (!tokensOn || !gpuOn || gpuSide === "cpu") return false;
    if (always || gpuSide === "gpu") return true;
    if (threads.searching) return false;
    if (tokenRecheck === "gpu") return true;
    if (tokenRecheck === "cpu") return false;
    return Boolean(steps.of(threads.count)?.faster);
  }
  /** T152: count steps of generate() on the GPU from token at pos: the forward pass of each token and the sampling of
   * the next (the penalty over the last of history, whose length is length; temperature, topp; randoms: a number for
   * each step, the CPU's generator's, none where greedy; stops: the stop tokens; T546: topK, minP, presence, and
   * written: how many of the history's latest were sampled, of which the presence penalty is). Returns the ids sampled (a stop
   * token last where one came), whose keys and values are in the cache then as the CPU would have written them; or
   * undefined where the GPU did not take them (the CPU takes the step instead: nothing of it was written). Not null:
   * Pyodide makes JavaScript's null jsnull, which is not None (T152's review), and undefined None */
  function generateMany(token: number, pos: number, history: Listed, length: number, count: number, temperature: number, topp: number, penalty: number,
    randoms: Listed, stops: Listed, topK = 0, minP = 0, presence = 0, written = 0) {
    const list = (x: Listed) => (x?.toJs ? x.toJs() : [...(x ?? [])]);
    // (T545.2: and steps that go on from the state of the layers that keep one, where the model has those: else the
    // CPU's step says why it cannot, follow()'s words)
    if (!gpuSteps() || count < 1 || count > GPU_TOKENS || (!direct && !follows(pos))) return undefined;
    const stopList = list(stops);
    if (stopList.length > STOPS_MOST) {
      tokensOn = false;
      tokensReason = REASONS.stops(STOPS_MOST);
      tokenStatus = "why";
      console.info(`gpu: answers on the CPU (${tokensReason})`);
      return undefined;
    }
    const began = now();
    if (!direct && pos + count - 1 >= cpuSide.capacity) grow(pos + count - 1);
    views();
    const fresh = stateUp(pos);
    gpuSerial = ++gpuRequests;
    Atomics.store(ctl!, GPU_WANTED, gpuSerial);
    // (T210: on the GPU alone, no cache here to go up from, nor any position the GPU does not hold)
    gpuWorker!.postMessage({ type: "tokens", serial: gpuSerial, count, pos, fresh, from: direct ? pos : Math.min(gpuEnd, pos), token, history: list(history), length, written,
      cache: direct ? null : { places: cpuSide.places(), row: KV, half: halfKV }, settings: { temperature, topp, penalty, topK, minP, presence, stops: stopList }, randoms: list(randoms) });
    if (!waitUntil(GPU_DONE, (seen) => seen === gpuSerial, GPU_BEAT)) {
      stopGpu(REASONS.stalled(stalledMs));
      return undefined;
    }
    if (Atomics.load(ctl!, GPU_FAILED)) {
      stopGpu(REASONS.tokenFailed);  // the GPU's worker said why in the console
      return undefined;
    }
    views();
    const words = new Int32Array(memory.buffer, gpuIds, 2 + GPU_TOKENS), sampled = words[0], notFinite = words[1 + GPU_TOKENS];
    if (notFinite) {
      // T219: the sampler refused the step after the `sampled` ones: its logits held a NaN or +inf, or none over
      // -3.4e38 (the State's not_finite word, set by the bits: WGSL lets a GPU take NaN and infinities as absent). The
      // whole request is refused as one with an id outside the vocabulary below is (nothing of it written, the ids
      // before it not taken), the GPU stops, and the CPU takes the step again and stops where its own logits are not
      // finite either (T195's NOT_FINITE). A model on the GPU alone cannot: it stops, said in words
      if (direct) throw new Error(OUTSIDE_VOCABULARY);
      stopGpu(REASONS.logits(pos + sampled));
      return undefined;
    }
    if (!(sampled >= 1 && sampled <= count)) {
      stopGpu(REASONS.sampledFew(sampled, count));
      return undefined;
    }
    // T219: an id outside the vocabulary is what SAMPLE gives for logits that are not finite numbers (NONE, -1: T195,
    // WGSL lets a GPU take NaN and infinities as absent). The step is refused, nothing of it written: the CPU takes it
    // again and stops where its own logits are not finite either (T195's NOT_FINITE), and the GPU, whose numbers are
    // no longer trusted, stops here. A model on the GPU alone cannot take the step on the CPU: it stops, said in words
    const ids = Array.from(words.subarray(1, 1 + sampled));
    const outside = ids.find((id) => !(id >= 0 && id < vocab));
    if (outside !== undefined) {
      if (direct) throw new Error(OUTSIDE_VOCABULARY);
      stopGpu(REASONS.sampledOutside(outside, vocab));
      return undefined;
    }
    // the keys and values of the positions sampled, float16 in the staging place as a prompt's block's (T147), into the
    // cache (T160's review of T152: a float32 cache, a grouped-query model's, widens them as a prompt's). T210: none
    // on the GPU alone. T243: where one of them is no finite number, the whole request is refused as above (nothing
    // written, none of its ids taken): the CPU takes the step, from keys and values that are its own
    if (!direct && !stagingFinite(sampled)) {
      stopGpu(REASONS.keysInStep(pos));
      return undefined;
    }
    // (T545.2: the state of the layers that keep one as the last step left it; T271.2: where a stop token came before the
    // last, the steps after it went on past the position the CPU is at, and no state of that position came back: only
    // position 0 goes on from there, as a generation that stopped does)
    const whole = sampled === count;
    if (!direct) fromStaging(pos, sampled);
    stateFromStaging(whole ? pos + count : null, !direct);
    gpuStateAt = whole ? pos + count : -1;
    gpuEnd = pos + sampled;
    gpuSampled += sampled;
    if (sampled === count) steps.gpu((now() - began) / count);
    if (tokenRecheck === "gpu") tokenRecheck = null;
    return ids;
  }
  // ---- T378.2: what happens to the GPU's side, as the rest of the engine tells it, and what it is asked
  /** the page began a generation (after the threads were told so: a search of theirs that begins with it is seen
   * here): the counts of the generation begin again, and now and then the side not chosen is to be timed again */
  function generationBegan() {
    gpuTokens = 0;
    // T148: halfway between the threads' checks, a prompt goes to the side not chosen, so that its time stays
    // today's (a device that heats up, a GPU timed while the CPU was busy)
    written = 0;
    const whole = gpuOn && !always ? times.of(GPU_BLOCK, threads.count) : null;
    if (whole && ++sinceCheck >= GPU_RECHECK && !recheck && !threads.searching) {
      recheck = whole.faster ? "cpu" : "gpu";
      sinceCheck = 0;
      console.info(`gpu: the ${recheck === "cpu" ? "CPU on the end" : "GPU on the beginning"} of the next long prompt, to time it again`);
    }
    // T152: the same for the steps of a generation, the side not chosen taking its first ones
    gpuSampled = 0;
    const stepsNow = tokensOn && !always ? steps.of(threads.count) : null;
    if (stepsNow && ++sinceTokens >= GPU_RECHECK && !tokenRecheck && !threads.searching) {
      tokenRecheck = stepsNow.faster ? "cpu" : "gpu";
      cpuRecheck = TOKEN_RECHECK;
      sinceTokens = 0;
      console.info(`gpu: the ${tokenRecheck === "cpu" ? `CPU on the first ${TOKEN_RECHECK} steps` : "GPU on the first steps"} of the next generation, to time it again`);
    }
    tokenVerdict();
  }
  /** T147: how many tokens of a prompt Python hands prompt() at once: GPU_BLOCK where the GPU takes a whole block
   * (T148: where it is faster), else BATCH (a longer call keeps the worker from answering for longer, and the CPU
   * gains nothing from it) */
  function promptBlock() {
    if (!gpuOn || gpuSide === "cpu") return BATCH;
    if (always || gpuSide === "gpu") return GPU_BLOCK;
    const whole = times.of(GPU_BLOCK, threads.count);
    return whole && (whole.faster || (recheck === "gpu" && written === 0)) ? GPU_BLOCK : BATCH;
  }
  /** T108, T135: the tokens of a prompt (list) at positions pos, pos + 1, ..., none of which makes logits: each block
   * on the GPU where it is on and takes it (GPU_BLOCK at a time), else on the CPU, BATCH at a time through the layers */
  function prompt(list: number[], pos: number) {
    for (let at = 0; at < list.length;) {
      const block = list.slice(at, at + (gpuOn ? GPU_BLOCK : BATCH)), from = pos + at;
      if (from === 0) verdict();
      // T148: the CPU timed again on the last 2 × BATCH tokens of a prompt (its last block is the short one), the
      // rest of that block where it goes
      const tail = recheck === "cpu" && block.length < GPU_BLOCK ? Math.min(block.length, 2 * BATCH) : 0;
      const head = block.slice(0, block.length - tail);
      if (head.length && (!gpuTakes(head.length, from) || !promptOnGpu(head, from))) promptOnCpu(head, from);
      if (tail) promptOnCpu(block.slice(head.length), from + head.length);
      at += block.length;
    }
    written = pos + list.length;
  }
  /** T135: the CPU is about to write the positions from pos on (run()): from there on the cache holds what the GPU's
   * own keys and values do not */
  function cpuWrote(pos: number) {
    gpuEnd = Math.min(gpuEnd, pos);
    gpuStateAt = -1;  // (T271.2: the CPU's state moves on: the GPU's is not the one to go on from)
  }
  /** T156: something was to run on the CPU of a model whose layers are on the GPU alone (a block or a step the GPU
   * did not take): the GPU's side is lost (the worker loads the model again on the CPU), and this is the error to throw */
  function notHere() {
    if (!directLost) stopGpu(REASONS.notTaken);
    return new Error(`The GPU stopped (${directLost}), and this model's weights were on it alone`);
  }
  /** T152: a token is about to run on the CPU. Whether it is a step to time against the GPU's (one that makes logits,
   * where the GPU takes steps): the caller times it then and says cpuStepEnded(its milliseconds) */
  const cpuStepBegan = (needLogits: boolean) => needLogits && tokensOn && gpuOn;
  /** the step cpuStepBegan() said to time took this long */
  function cpuStepEnded(milliseconds: number) {
    steps.cpu(threads.count, milliseconds);
    if (tokenRecheck === "cpu" && --cpuRecheck <= 0) tokenRecheck = null;
  }
  /** T152: how many steps of a generation Python hands generateMany() at once now: GPU_TOKENS where the GPU takes
   * them (it is faster, or it is timed again), 0 where the CPU takes the step (the forward pass, and the sampling in
   * Python). Not while the threads are searched: that times the CPU's tokens */
  const tokenBlock = () => (gpuSteps() ? GPU_TOKENS : 0);
  /** T184 (the benchmark's page path): every block of a prompt on "cpu" or on "gpu" (where the GPU is on and holds
   * the keys and values before it, as gpuForce.always), or anything else: each where it is faster (promptTimes) */
  function put(side: string | null) {
    gpuSide = side === "cpu" || side === "gpu" ? side : null;
  }
  /** The engine is let go: the GPU's worker stops. T205: resolves once it has let go of its buffers and its device, or
   * after GPU_END_MS (the worker is terminated then), true where it said so: the next model is read after it (an
   * iPhone's tab went down where the GPU of the one before still held up to 189 MB as the next came from the cache) */
  function release() {
    if (gpuWorker) stopGpu();
    const worker = gpuLast;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const late = new Promise<boolean>((resolve) => { timer = setTimeout(() => resolve(false), GPU_END_MS); });
    return Promise.race([gpuEnded.then(() => true), late]).then((ended) => {
      clearTimeout(timer);
      if (!ended) {
        console.warn(`gpu: the GPU's worker did not end within ${GPU_END_MS / 1000} s: terminated`);
        worker?.terminate?.();
      }
      return ended;
    });
  }
  // T378.2: the state above has this one owner: the engine tells it what happened (generationBegan, prompt, cpuWrote,
  // cpuStepBegan and cpuStepEnded, generateMany, put, notHere, release), asks it (gettingReady, promptBlock, tokenBlock,
  // keysAndValues), and reads what the page is shown of it; nothing outside assigns a variable of it
  return {
    generationBegan, prompt, cpuWrote, cpuStepBegan, cpuStepEnded, generateMany, put, notHere, release,
    gettingReady: gpuGettingReady, promptBlock, tokenBlock, keysAndValues: gpuKeysAndValues,
    /** T135: where the worker has WebGPU, a promise of what the status line says of the GPU, settled once the layers
     * are on the GPU or it is known that they will not be (T148: nothing waits for it); undefined where it has none */
    note: gpuNote,
    /** T148: what the status line says of the GPU now (it changes with the times of the prompts) */
    // (T368.4: a line or, where the engine has no GPU, undefined: null, nothing to say yet, is only while this function
    // runs, which starts the GPU's worker or refuses it before it returns)
    get status() { return statusNow() as string | undefined; },
    /** what the GPU chose as it started ({ matrices, attention, key, remembered, seconds, ... }: the page shows it and
     * remembers it): once it did, and while it is on */
    get chosen() { return gpuChosen; },
    get ready() { return gpuOn ? gpuChosen : undefined; },
    /** T184: why the prompts stay on the CPU (no WebGPU, a fallback adapter, a model the GPU does not take yet, a
     * failure), or null while none is known; T152: and why a generation's steps do */
    get whyNot() { return gpu ? gpuReason : REASONS.noWebGpu; },
    get stepsWhyNot() { return gpu ? gpuReason ?? tokensReason : REASONS.noWebGpu; },
    /** T135, T152: the tokens of a prompt and the steps that went through the GPU since the generation began */
    get promptTokens() { return gpuTokens; },
    get sampled() { return gpuSampled; },
  };
}
