// forward/alone.ts (T.349): a model on the GPU alone (T.156, T.210): what the GPU holds of it, whether it goes there
// (weightsPlace, gpuOnlyUnfit) and stays there (aloneVerdict), and its weights as they arrive, the layers' matrices and
// the tables to the GPU's worker and the rest to this memory without the holes they leave (gpuOnlyWeights, gpuOnlyPlan).
// A module of src/runtime/forward.ts, which exports its names as before.

import { PACKED, packingOf, weightBytes } from "../jobs.ts";
import { BETTER } from "./choice.ts";
import { architectureOf } from "./architectures.ts";
import type { Architecture } from "./architectures.ts";
import { LAYER_KINDS } from "./kinds.ts";
import { ATTENTION, sizesOfForm } from "./sizes.ts";
import { REASONS } from "./reasons.ts";
import type { Header } from "./memory.ts";
import type { Formed } from "./sizes.ts";
import type { Dtype, Rotated } from "../contracts/form.ts";
import type { Force, Matrix, Opened, Remembered, Table, ToGpu } from "../contracts/gpu/protocol.ts";
import type { Alone, CpuSpeed, Usage } from "../contracts/messages.ts";
import type { LayerKind, Tensor, TensorName, Tensors } from "../contracts/plan.ts";

// (T.368.4, the types) what of a form says whether a model can go on the GPU, and how many bytes it takes there
// (T.375.5: the architecture's name, the size of a head, and the numbers of the layers that keep a state: sizes.ts's Formed)
export type GpuForm = Formed & { bias?: boolean; qk_norm?: boolean };
// the adapter as the worker keeps it (worker/runtime.ts)
export type Adapter = { fallback: boolean; limits: { maxStorageBufferBindingSize: number; maxBufferSize: number }; packed: boolean; key?: string; ternaryKey?: string };
// T.156: where a model goes (weightsPlace()); gpuRoom: the bytes left for the GPU's copy where it is not on the GPU alone
export type Place = { mode: "both" | "gpu" | "cpu"; gpuRoom?: number };
// a stretch of a checkpoint that the GPU alone holds: [start, end)
export type Hole = [number, number];
// T.156: a model on the GPU alone, as the worker makes it and the engine reads it: the bytes kept in the memory (stored)
// of size, where each is (place), the multiply-adds of its layers, /benchmark/'s CPU reading and the page's use (what
// the GPU is held against), and onLost(why), told where the GPU stops under it. The GPU's side writes verdict: the CPU
// is faster here (the page keeps it)
export type Direct = {
  stored: number;
  size: number;
  place(offset: number): number;
  layerWeights: number;
  cpu: CpuSpeed | undefined;
  usage: Usage;
  onLost?(why: string): void;
  verdict?: Alone;
};
// T.156: the weights of a model on the GPU alone as they come (what gpuOnlyWeights() returns)
export type GpuOnlyWeights = ReturnType<typeof gpuOnlyWeights>;

// ---- T.156: a model on the GPU alone. Where the layers could not be held twice (in this memory and on the GPU), the
// worker decides before a byte comes that the GPU alone takes them (the owner, 2026-09-27: "大きいモデルは最初から GPU
// だけ", T.156's B): the layers' matrices (T.210: and the tables) go to the GPU's worker as they arrive and are never in
// this memory, the rest (the norms, the header) comes here, packed without the holes they leave (place()). Nothing
// runs on the CPU then, and it keeps none of the keys and values (T.210); a GPU that fails means the model is loaded
// again on the CPU (the worker).

// T.232, T.375.5: the bytes of a weight on the GPU, of a tensor stored as the checkpoint stores it: those of the packing
// the GPU holds it in (jobs.ts's PACKED: an int8 value and a float32 scale a group of 32, 1.125, which int6 is widened
// to, T.155; ternary, T.230, as the checkpoint holds it, two bits and a scale a group of 128, 0.28125). The GPU holds
// every scale as a float32 (T.533: ternary's float16 ones are widened as they go up, gpu/weights.ts). A model whose
// weights the GPU does not take is counted as int8's, as it always was (it does not go there: gpuFit())
export const gpuWeightBytes = (stored: string | undefined) => weightBytes({ ...PACKED[packingOf(stored)?.gpu?.as ?? "int8"], scale: 4 });
/** the weights of the layers' matrices, all of them: an attending layer's q, k, v and output (and T.271.4 the gate of
 * its output, a Qwen3.5's), a linear-attention layer's (T.271.4: q, k and v as one, z, and its output), and every
 * layer's FFN (three where it is gated, else GPT-2's two); how many layers attend; and the bytes of what a Qwen3.5's
 * linear-attention layers keep and read besides on the GPU (their states and rows, the gates' small matrices of
 * float32 and their vectors) */
const layersOf = (header: Header, form: Formed) => {
  const [dim, hidden, layers] = header, sizes = sizesOfForm(header, form, LAYER_KINDS), { q: qDim, kv: kvDim, linear } = sizes.widths;
  const attending = sizes.kinds.filter((kind) => kind === ATTENTION).length, deltas = sizes.kinds.filter((kind) => kind === "linear").length;
  const ffn = (architectureOf(form.arch ?? "llama").gatedFfn ? 3 : 2) * hidden * dim;
  const attention = (2 * qDim + 2 * kvDim + (sizes.gate ? qDim : 0)) * dim, delta = linear ? (linear.mixed + 2 * linear.read) * dim : 0;
  const numbers = form.linear, held = numbers && linear ? deltas * 4 * (numbers.value_heads * numbers.key_dim * numbers.value_dim + numbers.conv * linear.mixed +
    2 * numbers.value_heads * dim + 2 * numbers.value_heads + numbers.value_dim) : 0;
  return { weights: attending * attention + deltas * delta + layers * ffn, attending, held };
};
/** T.156: the bytes the GPU holds of a model with this header and form (FORM): its layers' matrices (as the GPU holds
 * their packing: gpuWeightBytes()), its two norms a layer, its own keys
 * and values (float16, the whole context), and for a generation's steps (T.152) the classifier (and the embedding where
 * it is another table), RoPE's table of every position and the sampling's three arrays of the vocabulary. forward.js
 * counts the same of a model it runs (layersOnGpu, tokensUnfit). */
// (T.271.4: the keys and values of the attending layers alone, and a Qwen3.5's linear-attention layers' own: layersOf())
export function gpuBytes(header: Header, { dtype = "int8", ...form }: GpuForm & { dtype?: Dtype } = {}) {
  const [dim, , layers, heads, kvHeads, signedVocab, seqLen] = header;
  const vocab = Math.abs(signedVocab), headSize = form.head_dim || dim / heads, kvDim = kvHeads * headSize;
  const each = gpuWeightBytes(dtype), { weights, attending, held } = layersOf(header, form);
  const layerBytes = weights * each + layers * 2 * dim * 4 + 2 * attending * seqLen * kvDim * 2 + held;
  const table = vocab * dim * each;
  return layerBytes + table * (signedVocab > 0 ? 1 : 2) + seqLen * headSize * 4 + 3 * vocab * 4;
}

// the layers' matrices of a Llama, which the GPU alone holds (T.156); T.271.4: and a Qwen3.5's (its linear-attention
// layers' and the gate of its attending layers' output), each of the layers of its kind where some layers are of
// another kind than attention (T.568.4: the kind of the model's whose tensors it is, kinds.ts's gpu.names)
const LAYER_MATRICES: TensorName[] = ["wq", "wk", "wv", "wo", "w1", "w2", "w3", "wqkv", "wz", "wout", "wg"];
// T.210: and its tables, the embedding and a classifier of its own (wcls is the embedding's tensor where it is shared):
// the GPU embeds a prompt's rows too (shaders.ts's EMBED_ROWS), and nothing of them is here
const GPU_ALONE: TensorName[] = [...LAYER_MATRICES, "token_embedding_table", "wcls"];
/** T.156: the stretches of the checkpoint the GPU alone holds, the layers' matrices and (T.210) the tables ([start, end)
 * of each, in file order: every one is its values and then its scales, llama2_numpy's Tensor), from the tensors
 * llama2_numpy.external_tensors() places */
export function gpuHoles(tensors: Tensors) {
  const holes = new Map(GPU_ALONE.filter((name) => tensors[name]).map((name): [number, Hole] => {
    const t = tensors[name]!, values = t.shape.reduce((a, b) => a * b, 1);
    return [t.offset, [t.offset, t.scales + (values / t.group) * packingOf(t.kind)!.scale]];
  }));
  return [...holes.values()].sort((a, b) => a[0] - b[0]);
}
/** T.156: where an offset of the checkpoint (outside the holes) is in the memory that holds the rest: less the holes
 * before it */
export const placer = (holes: Hole[]) => (offset: number) => holes.reduce((at, [start, end]) => (end <= offset ? at - (end - start) : at), offset);

/** T.156: whether a model goes on the GPU alone, from what is known before its bytes come (the owner's B, 2026-09-27;
 * TASKS.md's T.156 has the numbers): both, as before (T.148: the layers in this memory and on the GPU, and each block and
 * step where it is faster), where both fit half of what the device says it has; else the GPU alone where the model can
 * be (eligible: a Llama the GPU's steps take, T.152) and it fits; else the CPU (and the prompts on the GPU where the
 * layers alone fit, gpuRoom, as before). deviceMemory: navigator.deviceMemory, 4 where the browser does not say
 * (Safari, Firefox); Chromium says 8 for 8 GB or more: both are then held to BOTH_ON_8 (6.5 GiB, the owner's choice,
 * 2026-09-27: the models of the list up to 2B both as before, llm-jp-3.1 1.8B's 6.45 GB = 6.01 GiB with the keys and
 * values of 4096 positions on either side too (at 6 GiB it was 11 MB past: the second review of T.156); 3B and larger
 * (Qwen2.5 3B's 7.2 GB = 6.70 GiB, T.153's risk on an 8 GB phone) not), and the GPU alone to nothing, as the CPU alone is (a 7B's 9.2 GB runs on the CPU
 * there, T.132: the GPU alone takes about as much as the CPU alone, the matrices once either way).
 * cpu: the checkpoint and what the forward pass puts after it (footprint); gpuOnly: the checkpoint without the
 * matrices and what the forward pass puts after that (footprint with direct); gpu: gpuBytes. Returns { mode: "both" |
 * "gpu" | "cpu", gpuRoom } */
export const BOTH_ON_8 = 6.5 * 2 ** 30;
export function weightsPlace({ cpu, gpuOnly, gpu, deviceMemory = 4, eligible = false, forced = false }:
  { cpu: number; gpuOnly?: number; gpu: number; deviceMemory?: number; eligible?: boolean; forced?: boolean }): Place {
  const room = deviceMemory >= 8 ? BOTH_ON_8 : (deviceMemory * 2 ** 30) / 2;
  if (forced && eligible) return { mode: "gpu" };
  if (cpu + gpu <= room) return { mode: "both", gpuRoom: room - cpu };
  if (eligible && (deviceMemory >= 8 || gpuOnly! + gpu <= room)) return { mode: "gpu" };
  return { mode: "cpu", gpuRoom: room - cpu };
}

/** T.156: the multiply-adds of a model's layers a token of a prompt makes (every weight of its matrices once) */
export const layerWeightsOf = (header: Header, form: Formed = {}) => layersOf(header, form).weights;
// T.156 (the owner, 2026-09-27: the prompt counts too): the tokens of a prompt against the tokens written, where the
// page has not kept how they are used here yet: as many (a chat's question with its template and an answer of about
// the same length; no measurement of the site's visitors says otherwise)
export const USAGE_UNKNOWN = { prompt: 1, written: 1 };
/** T.156: whether the CPU would be faster than the GPU for a model on the GPU alone (then it is loaded again on the
 * CPU), by the time of what the page does: usage.prompt tokens of prompts and usage.written tokens written (the page's
 * recent use, decayed, or USAGE_UNKNOWN). The CPU's side is /benchmark/'s CPU section (cpu: { GBps, promptGMACs }): a
 * token written reads the checkpoint once (size bytes at GBps, T.157), a prompt's token makes layerWeights multiply-adds
 * at promptGMACs (its model is two layers as wide as Llama 3.2 1B; a narrower model runs slower than that says,
 * T.157's review: 1.09 to 1.21 on llm-jp-3 150M's shape). The GPU's side is its own (gpu: { stepMs, promptMs }: a step
 * of a run of GPU_TOKENS, a token of a block of GPU_BLOCK, as gpu.js timed them as it started). A side not known on
 * either leaves its part out; nothing known of the CPU: the GPU stays. Returns { cpuFaster, cpu, gpu } (ms of that use).
 * T.232: size is what an int8 model's token reads, 1.125 bytes a weight. A ternary model's checkpoint is a quarter of
 * that, and its CPU kernel is bound by its arithmetic, not by the reading (T.231: 1.0 to 2.0 times the int8 kernel's
 * weights a second on the CPUs measured): the caller hands the bytes of the same weights as int8 (cpuReadBytes), so
 * that the CPU is not taken for four times as fast as it is. (T.375.5: of any packing, by its line of PACKED; stored:
 * a dtype's name or a tensor's kind. Weights that are not packed are read as they are.) */
export const cpuReadBytes = (size: number, stored: string) => {
  const packing = packingOf(stored);
  return packing ? size * (weightBytes(PACKED.int8) / weightBytes(packing)) : size;
};
export function aloneVerdict({ size, layerWeights, cpu = {}, gpu = {}, usage = USAGE_UNKNOWN }:
  { size: number; layerWeights: number; cpu?: CpuSpeed; gpu?: { stepMs?: number; promptMs?: number }; usage?: Usage }) {
  const parts: [number, number, number][] = [];
  if (cpu.GBps! > 0 && gpu.stepMs! > 0) parts.push([usage.written, size / (cpu.GBps! * 1e6), gpu.stepMs!]);
  if (cpu.promptGMACs! > 0 && gpu.promptMs! > 0) parts.push([usage.prompt, layerWeights / (cpu.promptGMACs! * 1e6), gpu.promptMs!]);
  const onCpu = parts.reduce((sum, [n, ms]) => sum + n * ms, 0), onGpu = parts.reduce((sum, [n, , ms]) => sum + n * ms, 0);
  return { cpuFaster: parts.length > 0 && onCpu < BETTER * onGpu, cpu: onCpu, gpu: onGpu };
}
/** T.156: whether a verdict the page kept (alone: { key, cpu }, the device's key and /benchmark/'s CPU then) still holds:
 * the same device, browser and shaders (the key, T.148's), and the same CPU reading of /benchmark/ (a new run of its
 * CPU section, or none now, asks again) */
export const aloneHolds = (alone: Alone | null | undefined, key: string | undefined, cpu: CpuSpeed | undefined) => Boolean(alone && key && alone.key === key && cpu &&
  alone.cpu?.GBps === cpu.GBps && alone.cpu?.promptGMACs === cpu.promptGMACs);

// T.220: where the GPU's device binds a buffer (minStorageBufferOffsetAlignment). gpu.ts asks its device for the
// adapter's limits of size but not for this one, so the device has WebGPU's default, 256, whatever smaller value the
// adapter says it could give: gpuOnlyUnfit judges by the device's value, the one gpu.ts's tokensLayout and piecesOf
// read (asking the device for the adapter's value instead would move where every model's pieces start on the GPU, the
// prompt's too, for no model of the list: they all start on 256, T.213)
const BINDS_AT = 256;
/** T.156: why a model cannot go on the GPU alone, from its header, dtype and form before its bytes come, and the
 * adapter the worker asked for ({ fallback, limits }); null where it can. A model whose steps the GPU takes (T.152): by
 * its facts (T.375.5), one of RMSNorms and a gated FFN with no table of positions, no gate of the attention's output
 * and no layer that keeps a state: of the architectures there are, a Llama (T.226: Qwen2 and Qwen3 with it, whose
 * biases of q, k and v and norms of the heads are vectors that stay in this memory as the norms' weights do; not GPT-2
 * or GPT-NeoX, and no such model of the list is too large to hold twice). Of a packing that may be there alone (its
 * line of jobs.ts's PACKED: int8; not six bits, which are for a device short of memory, where the widened int8 on the
 * GPU, 1.125 bytes a weight against six bits' 0.875, would not fit either: T.155's review; T.232 ternary, where the
 * browser's WGSL has the packed int8 dot: adapter.packed),
 * heads of a multiple of 4, and q, k and v and gate and up each one range of a buffer the device binds (gpu.js's
 * tokensLayout says the last word: a GPU that refuses then means the model is loaded again on the CPU). */
// T.219: what a model on the GPU alone says where the GPU sampled an id outside the vocabulary (its logits were not
// finite numbers, most likely: the words of T.195's NOT_FINITE, llama2_numpy.py), since no CPU can take the step again
export const OUTSIDE_VOCABULARY = "The model computed logits on the GPU that are not finite numbers (NaN or infinity), so no token can be drawn: its weights are broken or its numbers overflowed.";
/** T.382: why a model of these facts cannot go on the GPU alone, or null: the first fact that keeps it off, each by its
 * own reason (stateful: it has a layer that keeps a state the GPU alone does not hold, T.271.4: an LFM2's convolution
 * layers; the rest are its architecture's line). Apart from gpuOnlyUnfit() so that a set of facts no architecture of
 * the table has can be asked too (the tests). (T.271.4: a gate of the attending layers' output is a token's layer's now,
 * a Qwen3.5's; a Spark-X2.5's gate of its heads goes with its exact GELU, below) */
export function aloneFactsUnfit({ stateful, positions, gatedFfn, layerNorm, activation, local }:
  Pick<Architecture, "positions" | "gatedFfn" | "layerNorm" | "activation" | "local"> & { stateful: boolean }) {
  if (stateful) return REASONS.stateful;
  if (positions) return REASONS.alonePositions;
  if (!gatedFfn) return REASONS.aloneFfn;
  if (layerNorm) return REASONS.aloneNorms;
  // T.261: a Gemma 3's facts (its norms behind the attention and the FFN are rows of its file, which no line says: the
  // other two keep it off). An activation that is not a SiLU's is none the GPU's gate has, whatever it is. (T.261.4: a
  // Gemma 3's prompts go to the GPU beside the CPU now; on the GPU alone its steps would have to as well, which a
  // token's fused layer does not take yet: and the 270M and the 1B, 0.3 and 1.1 GB, are held twice where a GPU is)
  if (activation !== "silu") return REASONS.aloneActivation;
  if (local) return REASONS.aloneLocal;
  return null;
}
// T.271.1: the largest block of a rotated basis the GPU's ROTATE_QUANTIZE and UNROTATE hold in a workgroup's memory
// (shaders/rotate.ts's ROTATE_MOST, a second copy here: no shader's text comes into this chunk; tests/gpu-part-check.mjs
// holds it to that one), and whether a rotated basis's block is one they take: a power of two (the butterflies) of 32 (a
// group of the 8 bits a thread) to that most (0: the model's own basis, which they are not asked of)
export const ROTATE_MOST = 1024;
export const rotatedBlockUnfit = (block: number) => Boolean(block) && ((block & (block - 1)) !== 0 || block < 32 || block > ROTATE_MOST);

export function gpuOnlyUnfit(header: Header, dtype: Dtype, form: GpuForm & { rotated?: Rotated | null; kinds?: Record<string, unknown> | null } = {},
  adapter: Adapter | null | undefined, force: Force = {}) {
  const [dim, hidden] = header;
  if (!adapter) return REASONS.noAdapter;
  if (adapter.fallback && !force.fallback) return REASONS.aloneFallback;
  // T.375.5: by the model's facts, not by its architecture's name: RMSNorms, a gated FFN, no table of positions, and
  // every layer attending (which of the five architectures a Llama's alone are). T.271.4: or a Qwen3.5's linear-attention
  // layers, whose state the GPU keeps itself, and the gate of its attending layers' output, which a token's layer has
  const sizes = sizesOfForm(header, form, LAYER_KINDS), { q: qDim, kv: kvDim, linear } = sizes.widths;
  // (T.382: the architecture's line as it is, so that no fact of it is handed on by hand, and the one the sizes say)
  const facts = aloneFactsUnfit({ ...architectureOf(form.arch ?? "llama"), stateful: sizes.kinds.some((kind) => kind !== ATTENTION && !LAYER_KINDS[kind as LayerKind].gpu?.alone) });
  if (facts) return facts;
  // T.271.1: nor a rotated basis whose blocks the GPU's shader does not take
  if (form.rotated && rotatedBlockUnfit(Number(form.rotated.block))) return REASONS.rotatedBlock;
  // T.363.5: nor a model whose rows have kinds of their own (llama.cpp's blocks are the CPU's kernels' alone: gpu.ts
  // takes one packing for a file)
  if (form.kinds && Object.keys(form.kinds).length) return REASONS.aloneDtype("rows of kinds of their own: such");
  // (the packing's line of jobs.ts's PACKED: whether a model of it may be on the GPU alone, and what its shaders need)
  const packing = packingOf(dtype), held = packing?.gpu;
  if (!packing || !held?.alone) return REASONS.aloneDtype(dtype);
  if (held.dot && !adapter.packed) return REASONS.packed(dtype);
  if ((qDim / sizes.heads) % 4) return REASONS.aloneHeads;
  const { maxStorageBufferBindingSize, maxBufferSize } = adapter.limits;
  const binds = Math.min(maxStorageBufferBindingSize, maxBufferSize);
  // (a joined matrix's parts start where the device binds: the values a row's bytes on, the scales four bytes a group)
  const starts = [qDim * dim, (qDim + kvDim) * dim, hidden * dim];
  if (starts.some((weights) => packing.rowBytes(weights) % BINDS_AT || ((weights / packing.group) * 4) % BINDS_AT)) return REASONS.aloneStarts;
  // (T.271.4: and a Qwen3.5's linear-attention layers' matrices, each one piece: a token's layer reads them whole)
  if ([(qDim + 2 * kvDim) * dim, 2 * hidden * dim, dim * Math.max(qDim, hidden), ...(linear ? [linear.mixed * dim, linear.read * dim] : [])]
    .some((weights) => packing.rowBytes(weights) > binds)) return REASONS.alonePast;
  return null;
}
/** T.375.5: whether a model of this dtype may be on the GPU alone at all (its packing's line of PACKED): what the worker
 * asks before it opens a file to keep a conversion in as it comes */
export const aloneTakes = (dtype: string) => Boolean(packingOf(dtype)?.gpu?.alone);

// T.156: the GPU's worker takes the layers' bytes as they come, and says in flow[0] how many it has put on the GPU: the
// loops that write wait (room()) where they are more than FLOW_BYTES ahead (a kept model read from the disk comes
// faster than a GPU takes it, and what waits in the messages is memory), and give up where nothing was taken for
// FLOW_STALL_MS. They wait without blocking: a worker made from a worker may not even start while the one that made
// it blocks (Chromium, gpu-check's first run: the GPU's worker took nothing for 60 s)
// T.377: a worker that is gone (it did not start, or threw: whoever made it hears its "error") takes nothing more and
// counts nothing more: gone() says so, and nothing waits for it from then on (else every load waited those 60 s and
// then failed, where the model can be loaded again on the CPU at once)
const FLOW_BYTES = 64 * 2 ** 20, FLOW_STALL_MS = 60000;
/** T.156: the weights of a model on the GPU alone. worker: the GPU's worker (gpu.js, told to open with gpuOnlyPlan()),
 * tensors: llama2_numpy.external_tensors(), size: the checkpoint's bytes. Returns { holes, place, stored, flow,
 * write(offset, bytes), room(), drained() }: write() puts the bytes outside the holes into memory at base +
 * place(offset), and posts those in them to the worker (a copy each); room() resolves once the worker is no more than
 * FLOW_BYTES behind, drained() once it has all of them; stored is what memory holds (the checkpoint less the holes).
 * gone() (T.377): the worker takes no more (the bytes for it are dropped from then on, and room() and drained() resolve). */
export function gpuOnlyWeights({ memory, base, size, tensors, worker }:
  { memory: WebAssembly.Memory; base: number; size: number; tensors: Tensors; worker: { postMessage(message: ToGpu, transfer: Transferable[]): void } }) {
  const holes = gpuHoles(tensors), place = placer(holes);
  const stored = size - holes.reduce((sum, [start, end]) => sum + (end - start), 0);
  const flow = new SharedArrayBuffer(8), taken = new BigInt64Array(flow);
  let sent = 0n, gone = false;
  const wait = async (ahead: bigint) => {
    let seen = Atomics.load(taken, 0), moved = performance.now();
    while (!gone && sent - seen > ahead) {
      if (Atomics.waitAsync) await Atomics.waitAsync(taken, 0, seen, 1000).value;
      else await new Promise((resolve) => setTimeout(resolve, 10));
      const now = Atomics.load(taken, 0);
      if (now !== seen) [seen, moved] = [now, performance.now()];
      else if (performance.now() - moved > FLOW_STALL_MS) throw new Error(`the GPU took no weights for ${FLOW_STALL_MS / 1000} s`);
    }
  };
  const post = (offset: number, chunk: Uint8Array, from: number, to: number) => {
    if (gone) return;
    const bytes = chunk.slice(from, to);
    worker.postMessage({ type: "weights", offset: offset + from, bytes }, [bytes.buffer]);
    sent += BigInt(to - from);
  };
  return {
    holes, place, stored, flow,
    write(offset: number, chunk: Uint8Array) {
      let at = 0;
      for (const [start, end] of holes) {
        if (end <= offset + at || start >= offset + chunk.length) continue;
        const from = Math.max(start - offset, at), to = Math.min(end - offset, chunk.length);
        if (from > at) new Uint8Array(memory.buffer, base + place(offset + at), from - at).set(chunk.subarray(at, from));
        post(offset, chunk, from, to);
        at = to;
      }
      if (at < chunk.length) new Uint8Array(memory.buffer, base + place(offset + at), chunk.length - at).set(chunk.subarray(at));
    },
    room: () => wait(BigInt(FLOW_BYTES)),
    /** every byte posted is on the GPU (or was dropped by a GPU that failed) */
    drained: () => wait(0n),
    /** T.377: the GPU's worker takes no more of them: what waits for it stops waiting, now */
    gone() {
      gone = true;
      Atomics.notify(taken, 0);
    },
  };
}
/** T.156: what the GPU's worker opens with (gpu.js's open(): the device, and a buffer for every piece of every layer's
 * matrices, before a byte comes): each matrix's rows and length and, a layer each, where its values and scales start in
 * the checkpoint (the offsets the bytes come with). T.210: and the tables' ({ classifier, embedding }, the embedding
 * null where the classifier is it: { rows, n, at: [values, scales] } each, in the checkpoint too) */
// T.271.4: form, the model's (FORM): a Qwen3.5's linear-attention layers, whose matrices (and the attending layers') are
// of the layers of their kind alone: the plan says every layer's kind and its place among the layers of its kind then,
// and those matrices' kind (of), as the plan of { type: "start" } does
export function gpuOnlyPlan(header: Header, tensors: Tensors, force: Force = {}, remembered?: Remembered, form: Formed = {}): Opened {
  const layers = header[2], kinds = sizesOfForm(header, form, LAYER_KINDS).kinds as LayerKind[], mixed = kinds.some((kind) => kind !== ATTENTION);
  const places = kinds.map((kind, l) => kinds.slice(0, l).filter((other) => other === kind).length);
  // (T.232: the bytes of a layer's values are its packing's, jobs.ts's PACKED; ternary is what the GPU's worker is told)
  const matrices = Object.fromEntries(LAYER_MATRICES.filter((name) => tensors[name]).map((name): [string, Matrix] => {
    const t = tensors[name]!, [, rows, n] = t.shape, perLayer = rows * n, layerBytes = packingOf(t.kind)!.rowBytes(perLayer);
    const of = mixed ? kinds.find((kind) => LAYER_KINDS[kind].gpu?.names.includes(name)) : undefined, count = of ? kinds.filter((kind) => kind === of).length : layers;
    return [name, { rows, n, ternary: t.kind === "ternary", layers: Array.from({ length: count }, (_, l) =>
      [t.offset + l * layerBytes, t.scales + (l * perLayer / t.group) * packingOf(t.kind)!.scale]), ...(of ? { of } : {}) }];  // (T.533: ternary's scales float16s)
  }));
  const table = (t: Tensor): Table => ({ rows: t.shape[0], n: t.shape[1], ternary: t.kind === "ternary", at: [t.offset, t.scales] });
  const embedding = tensors.token_embedding_table!, classifier = tensors.wcls ?? embedding;
  const tables = { classifier: table(classifier), embedding: classifier.offset === embedding.offset ? null : table(embedding) };
  return { layers, matrices, tables, force, remembered, ...(mixed ? { kinds, places } : {}) };
}

// (T.349) what forward.ts and the other modules read besides, which was not exported where it was one file
export { GPU_ALONE };
