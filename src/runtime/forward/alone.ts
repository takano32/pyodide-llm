// forward/alone.ts (T349): a model on the GPU alone (T156, T210): what the GPU holds of it, whether it goes there
// (weightsPlace, gpuOnlyUnfit) and stays there (aloneVerdict), and its weights as they arrive, the layers' matrices and
// the tables to the GPU's worker and the rest to this memory without the holes they leave (gpuOnlyWeights, gpuOnlyPlan).
// A module of src/runtime/forward.ts, which exports its names as before.

import { BLOCK, PACKED, packingOf, weightBytes } from "../jobs.ts";
import { BETTER } from "./choice.ts";
import { architectureOf } from "./architectures.ts";
import type { Architecture } from "./architectures.ts";
import { LAYER_KINDS } from "./kinds.ts";
import { ATTENTION, kindsHad, sizesOfForm } from "./sizes.ts";
import { REASONS } from "./reasons.ts";
import type { Header } from "./memory.ts";
import type { Formed, Sizes } from "./sizes.ts";
import type { CacheForm } from "./attention.ts";
import type { Rows } from "./perlayer.ts";
import type { Dtype, Rotated } from "../contracts/form.ts";
import type { Force, GpuPacking, Matrix, Opened, Remembered, Table, ToGpu } from "../contracts/gpu/protocol.ts";
import type { Alone, CpuSpeed, Usage } from "../contracts/messages.ts";
import type { Activation, LayerKind, Tensor, TensorName, Tensors } from "../contracts/plan.ts";

// (T368.4, the types) what of a form says whether a model can go on the GPU, and how many bytes it takes there
// (T375.5: the architecture's name, the size of a head, and the numbers of the layers that keep a state: sizes.ts's Formed)
// (T334.4: and what sizes each layer of its own, attention.ts's CacheForm, and the width of every layer's input of a table of its own)
type LayersForm = Formed & CacheForm & { per_layer?: number };
// (T334.3's review: and the cap of the attention's scores, a Gemma 2's, which the converter writes into the options:
// no FORM key, gemma2.py's gemma2_options())
export type GpuForm = LayersForm & { bias?: boolean; qk_norm?: boolean; score_softcap?: number };
// the adapter as the worker keeps it (worker/runtime.ts). keys: the device's key of a model by the shaders of its own
// (shaders/key.ts's deviceKey(): plain, T232 ternary weights', T547.3.7 llama.cpp's blocks'), none where it was not made
export type Adapter = { fallback: boolean; limits: { maxStorageBufferBindingSize: number; maxBufferSize: number }; packed: boolean;
  keys?: { plain: string; ternary: string; blocks: string } };
// T156: where a model goes (weightsPlace()); gpuRoom: the bytes left for the GPU's copy where it is not on the GPU alone
export type Place = { mode: "both" | "gpu" | "cpu"; gpuRoom?: number };
// a stretch of a checkpoint that the GPU alone holds: [start, end)
export type Hole = [number, number];
// T156: a model on the GPU alone, as the worker makes it and the engine reads it: the bytes kept in the memory (stored)
// of size, where each is (place), the multiply-adds of its layers, /benchmark/'s CPU reading and the page's use (what
// the GPU is held against), and onLost(why), told where the GPU stops under it. The GPU's side writes verdict: the CPU
// is faster here (the page keeps it). T334.3: rows, what reads the rows of a table of every layer's inputs (a Gemma
// 4's) from where the model is kept, where the table is neither in the memory nor on the GPU (place() leaves its
// offsets as they are in the checkpoint then: its stretch is one of the holes, gpuOnlyWeights()'s apart)
export type Direct = {
  stored: number;
  size: number;
  place(offset: number): number;
  rows?: Rows;
  layerWeights: number;
  cpu: CpuSpeed | undefined;
  usage: Usage;
  onLost?(why: string): void;
  verdict?: Alone;
};
// T156: the weights of a model on the GPU alone as they come (what gpuOnlyWeights() returns)
export type GpuOnlyWeights = ReturnType<typeof gpuOnlyWeights>;

// ---- T156: a model on the GPU alone. Where the layers could not be held twice (in this memory and on the GPU), the
// worker decides before a byte comes that the GPU alone takes them (the owner, 2026-09-27: "大きいモデルは最初から GPU
// だけ", T156's B): the layers' matrices (T210: and the tables) go to the GPU's worker as they arrive and are never in
// this memory, the rest (the norms, the header) comes here, packed without the holes they leave (place()). Nothing
// runs on the CPU then, and it keeps none of the keys and values (T210); a GPU that fails means the model is loaded
// again on the CPU (the worker).

// T232, T375.5: the bytes of a weight on the GPU, of a tensor stored as the checkpoint stores it: those of the packing
// the GPU holds it in (jobs.ts's PACKED: an int8 value and a float32 scale a group of 32, 1.125, which int6 is widened
// to, T155; ternary, T230, as the checkpoint holds it, two bits and a scale a group of 128, 0.28125). The GPU holds
// every scale as a float32 (T533: ternary's float16 ones are widened as they go up, gpu/weights.ts). A model whose
// weights the GPU does not take is counted as int8's, as it always was (it does not go there: gpuFit())
// (T547.2.3: a scale a group as a float32 there, but none apart from the blocks that hold their own: Q4_K's)
export const gpuWeightBytes = (stored: string | undefined) => {
  const held = PACKED[packingOf(stored)?.gpu?.as ?? "int8"];
  return weightBytes({ ...held, scale: held.scale ? 4 : 0 });
};
// T641.3: the kinds of a model's layers that the GPU alone may hold (kinds.ts's gpu.alone, in the table's order), each
// with the number of its layers: none of a kind that it may not (an LFM2's convolution layers: gpuOnlyUnfit() keeps
// such a model off it, and gpuBytes() counts no matrix of theirs, as before)
const aloneKinds = (sizes: Sizes) => kindsHad(sizes, LAYER_KINDS).flatMap(({ kind, count }) => {
  const alone = LAYER_KINDS[kind as LayerKind].gpu?.alone;
  return alone ? [{ alone, count }] : [];
});
/** the weights of the layers' matrices, all of them: those of each kind's layers (kinds.ts's gpu.alone: an attending
 * layer's q, k, v and output, and T271.4 the gate of its output, a Qwen3.5's; a linear-attention layer's q, k and v as
 * one, z, and its output) and every layer's FFN (three where it is gated, else GPT-2's two); how many layers keep keys
 * and values (the cache's kinds: those that attend); the bytes the kinds' layers keep and read besides on the GPU (a
 * Qwen3.5's linear-attention layers' states and rows, the gates' small matrices of float32 and their vectors); and
 * (T547.3.7) each matrix's weights in each of the layers it is of, for a form whose rows have kinds of their own.
 * T334.4: every layer's of its own sizes, a Gemma 4's (its wide heads, no k and v where it reads another's keys and
 * values, an FFN twice as wide in the layers of the form's double), and the matrices of its inputs of every layer (the
 * projection of the embedded rows, the model's one, and every layer's gate and projection back: gpufit.ts's sized);
 * and the widths of every attending layer's keys and values (the cache's kinds' cached(), as footprint() counts them:
 * 0 where a layer reads another's) */
const layersOf = (header: Header, form: LayersForm) => {
  const [dim, hidden, layers] = header, sizes = sizesOfForm(header, form, LAYER_KINDS), kinds = aloneKinds(sizes);
  const roles: Record<string, number[]> = {};
  const add = (name: string, weights: number) => (roles[name] ??= []).push(weights);
  // (a kind the GPU alone may not hold has no matrices here, as before: aloneKinds())
  sizes.kinds.forEach((kind, l) => {
    for (const [name, weights] of Object.entries(LAYER_KINDS[kind as LayerKind].gpu?.alone?.matrices(sizes, form, l) ?? {})) add(name, weights!);
  });
  const gated = architectureOf(form.arch ?? "llama").gatedFfn;
  for (let l = 0; l < layers; l++) {
    const inside = (form.double?.includes(l) ? 2 : 1) * hidden * dim;
    for (const name of gated ? ["w1", "w2", "w3"] : ["w1", "w2"]) add(name, inside);
  }
  const perLayer = form.per_layer || 0;
  if (perLayer) add("per_layer_projection", layers * perLayer * dim);
  for (let l = 0; perLayer && l < layers; l++) {
    add("ple_gate", perLayer * dim);
    add("ple_proj", dim * perLayer);
  }
  const keptWidths = kindsHad(sizes, LAYER_KINDS).flatMap(({ kind }) =>
    LAYER_KINDS[kind as LayerKind].cached?.(sizes, form, sizes.kinds.flatMap((had, l) => (had === kind ? [l] : []))).widths ?? []);
  const held = kinds.reduce((sum, { alone, count }) => sum + count * (alone.held?.(sizes) ?? 0), 0);
  return { weights: Object.values(roles).flat().reduce((sum, weights) => sum + weights, 0), keptWidths, held, roles };
};
/** T156: the bytes the GPU holds of a model with this header and form (FORM): its layers' matrices (as the GPU holds
 * their packing: gpuWeightBytes()), its two norms a layer, its own keys
 * and values (float16, the whole context), and for a generation's steps (T152) the classifier (and the embedding where
 * it is another table), RoPE's table of every position and the sampling's three arrays of the vocabulary. forward.ts
 * counts the same of a model it runs (gpufit.ts's layersOnGpu, tokensUnfit). */
// (T271.4: the keys and values of the attending layers alone, and a Qwen3.5's linear-attention layers' own: layersOf())
// (T547.3.7: of a form whose rows have kinds of their own, each row's and layer's as its kind: Saluki's types of blocks)
export function gpuBytes(header: Header, { dtype = "int8", ...form }: GpuForm & { dtype?: Dtype; kinds?: Record<string, unknown> | null } = {}) {
  const [dim, , layers, heads, , signedVocab, seqLen] = header;
  const vocab = Math.abs(signedVocab), { keptWidths, held, roles } = layersOf(header, form), kinds = form.kinds ?? {};
  // (each layer's as the form's kind of the row, or of that layer where it gives a list, else as the file's dtype)
  const ofKind = (name: string, weights: number[]) => {
    const kind = kinds[name];
    return weights.reduce((sum, one, i) => sum + one * gpuWeightBytes(Array.isArray(kind) ? (kind as string[])[i] : typeof kind === "string" ? kind : dtype), 0);
  };
  const matrices = Object.entries(roles).reduce((sum, [name, weights]) => sum + ofKind(name, weights), 0);
  const layerBytes = matrices + layers * 2 * dim * 4 + 2 * keptWidths.reduce((sum, width) => sum + width, 0) * seqLen * 2 + held;
  const tables = ofKind("token_embedding_table", [vocab * dim]) + (signedVocab > 0 ? 0 : ofKind("wcls", [vocab * dim]));
  // (RoPE's table of the widest heads: T331, a Gemma 4's are its wide ones'; one table alone, T334.4's review: a local
  // layer's second one (a Gemma 3's or 4's, about 4 MiB of an E2B's) is not counted)
  return layerBytes + tables + seqLen * Math.max(form.head_dim || dim / heads, form.wide?.head_dim ?? 0) * 4 + 3 * vocab * 4;
}

// the layers' matrices of a Llama, which the GPU alone holds (T156); T271.4: and a Qwen3.5's (its linear-attention
// layers' and the gate of its attending layers' output), each of the layers of its kind where some layers are of
// another kind than attention (T568.4: the kind of the model's whose tensors it is, kinds.ts's gpu.names). T641.3:
// every model's (gpufit.ts's too), then those of each kind the GPU alone may hold of its own (kinds.ts's gpu.matrices)
// T334.3: and a Gemma 4's own (gpufit.ts's sized: its wide heads' stacks, the projection of its embedded rows into the
// inputs of every layer and every layer's gate and projection back of its input, its wider FFN's stacks), last
const LAYER_MATRICES: TensorName[] = ["wq", "wk", "wv", "wo", "w1", "w2", "w3",
  ...Object.values(LAYER_KINDS).flatMap(({ gpu }) => (gpu?.alone ? gpu.matrices ?? [] : [])),
  "wq_wide", "wk_wide", "wv_wide", "wo_wide", "per_layer_projection", "ple_gate", "ple_proj", "w1_double", "w2_double", "w3_double"];
// T210: and its tables, the embedding and a classifier of its own (wcls is the embedding's tensor where it is shared):
// the GPU embeds a prompt's rows too (shaders.ts's EMBED_ROWS), and nothing of them is here
const GPU_ALONE: TensorName[] = [...LAYER_MATRICES, "token_embedding_table", "wcls"];
/** T156: the stretches of the checkpoint the GPU alone holds, the layers' matrices and (T210) the tables ([start, end)
 * of each, in file order: every one is its values and then its scales, llama2_numpy's Tensor), from the tensors
 * llama2_numpy.external_tensors() places */
// (T547.3.7: a stack whose layers are each of a kind of their own from its first layer's values to its last's end)
export function gpuHoles(tensors: Tensors) {
  const holes = new Map(GPU_ALONE.filter((name) => tensors[name]).map((name): [number, Hole] => {
    const t = tensors[name]!, values = t.shape.reduce((a, b) => a * b, 1);
    if (t.layers) {
      const last = t.layers[t.layers.length - 1];
      return [t.offset, [t.offset, last.scales + (values / t.layers.length / last.group) * packingOf(last.kind)!.scale]];
    }
    return [t.offset, [t.offset, t.scales + (values / t.group) * packingOf(t.kind)!.scale]];
  }));
  return [...holes.values()].sort((a, b) => a[0] - b[0]);
}
/** T156: where an offset of the checkpoint (outside the holes) is in the memory that holds the rest: less the holes
 * before it */
export const placer = (holes: Hole[]) => (offset: number) => holes.reduce((at, [start, end]) => (end <= offset ? at - (end - start) : at), offset);

/** T156: whether a model goes on the GPU alone, from what is known before its bytes come (the owner's B, 2026-09-27;
 * TASKS.md's T156 has the numbers): both, as before (T148: the layers in this memory and on the GPU, and each block and
 * step where it is faster), where both fit half of what the device says it has; else the GPU alone where the model can
 * be (eligible: a Llama the GPU's steps take, T152) and it fits; else the CPU (and the prompts on the GPU where the
 * layers alone fit, gpuRoom, as before). deviceMemory: navigator.deviceMemory, 4 where the browser does not say
 * (Safari, Firefox); Chromium says 8 for 8 GB or more: both are then held to BOTH_ON_8 (6.5 GiB, the owner's choice,
 * 2026-09-27: the models of the list up to 2B both as before, llm-jp-3.1 1.8B's 6.45 GB = 6.01 GiB with the keys and
 * values of 4096 positions on either side too (at 6 GiB it was 11 MB past: the second review of T156); 3B and larger
 * (Qwen2.5 3B's 7.2 GB = 6.70 GiB, T153's risk on an 8 GB phone) not), and the GPU alone to nothing, as the CPU alone is (a 7B's 9.2 GB runs on the CPU
 * there, T132: the GPU alone takes about as much as the CPU alone, the matrices once either way).
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

/** T156: the multiply-adds of a model's layers a token of a prompt makes (every weight of its matrices once) */
export const layerWeightsOf = (header: Header, form: LayersForm = {}) => layersOf(header, form).weights;
// T156 (the owner, 2026-09-27: the prompt counts too): the tokens of a prompt against the tokens written, where the
// page has not kept how they are used here yet: as many (a chat's question with its template and an answer of about
// the same length; no measurement of the site's visitors says otherwise)
export const USAGE_UNKNOWN = { prompt: 1, written: 1 };
/** T156: whether the CPU would be faster than the GPU for a model on the GPU alone (then it is loaded again on the
 * CPU), by the time of what the page does: usage.prompt tokens of prompts and usage.written tokens written (the page's
 * recent use, decayed, or USAGE_UNKNOWN). The CPU's side is /benchmark/'s CPU section (cpu: { GBps, promptGMACs }): a
 * token written reads the checkpoint once (size bytes at GBps, T157), a prompt's token makes layerWeights multiply-adds
 * at promptGMACs (its model is two layers as wide as Llama 3.2 1B; a narrower model runs slower than that says,
 * T157's review: 1.09 to 1.21 on llm-jp-3 150M's shape). The GPU's side is its own (gpu: { stepMs, promptMs }: a step
 * of a run of GPU_TOKENS, a token of a block of GPU_BLOCK, as gpu/start.ts timed them as it started). A side not known on
 * either leaves its part out; nothing known of the CPU: the GPU stays. Returns { cpuFaster, cpu, gpu } (ms of that use).
 * T232: size is what an int8 model's token reads, 1.125 bytes a weight. A ternary model's checkpoint is a quarter of
 * that, and its CPU kernel is bound by its arithmetic, not by the reading (T231: 1.0 to 2.0 times the int8 kernel's
 * weights a second on the CPUs measured): the caller hands the bytes of the same weights as int8 (cpuReadBytes), so
 * that the CPU is not taken for four times as fast as it is. (T375.5: of any packing, by its line of PACKED; stored:
 * a dtype's name or a tensor's kind. Weights that are not packed are read as they are.) */
export const cpuReadBytes = (size: number, stored: string) => {
  const packing = packingOf(stored);
  return packing ? size * (weightBytes(PACKED.int8) / weightBytes(packing)) : size;
};
/** T313 (C): /benchmark/'s CPU (cpu) as it would run a model whose weights are stored so: of ternary codes (PACKED's
 * gpu.as ternary: ternary, and binary, whose CPU kernel is ternary's with a type argument), its int8 reading and G MAC/s
 * times what the CPU section measured of the ternary kernels against int8's (CpuSpeed's ternary: a token's and a prompt's
 * tile's, one thread, the matrix alone), each where it was measured; else, and for every other packing, cpu as it is
 * (with cpuReadBytes, T232's estimate: as fast as the same weights of int8). */
export const cpuOf = (cpu: CpuSpeed | undefined, stored: string): CpuSpeed | undefined => {
  const times = packingOf(stored)?.gpu?.as === "ternary" ? cpu?.ternary : undefined;
  if (!times) return cpu;
  return { ...cpu, ...(times.token! > 0 && cpu!.GBps ? { GBps: cpu!.GBps * times.token! } : {}),
    ...(times.prompt! > 0 && cpu!.promptGMACs ? { promptGMACs: cpu!.promptGMACs * times.prompt! } : {}) };
};
export function aloneVerdict({ size, layerWeights, cpu = {}, gpu = {}, usage = USAGE_UNKNOWN }:
  { size: number; layerWeights: number; cpu?: CpuSpeed; gpu?: { stepMs?: number; promptMs?: number }; usage?: Usage }) {
  const parts: [number, number, number][] = [];
  if (cpu.GBps! > 0 && gpu.stepMs! > 0) parts.push([usage.written, size / (cpu.GBps! * 1e6), gpu.stepMs!]);
  if (cpu.promptGMACs! > 0 && gpu.promptMs! > 0) parts.push([usage.prompt, layerWeights / (cpu.promptGMACs! * 1e6), gpu.promptMs!]);
  const onCpu = parts.reduce((sum, [n, ms]) => sum + n * ms, 0), onGpu = parts.reduce((sum, [n, , ms]) => sum + n * ms, 0);
  return { cpuFaster: parts.length > 0 && onCpu < BETTER * onGpu, cpu: onCpu, gpu: onGpu };
}
/** T641.9: which of the device's keys (the worker's gpuAdapter.keys) a verdict of a model on the GPU alone is kept by:
 * the key of the shaders the GPU's worker multiplies the model's weights with, its ownOf(). A dtype whose packing goes
 * up as ternary's codes (PACKED's gpu.as: ternary, T232, and binary, T634.3) by the ternary key; a model of llama.cpp's
 * blocks (blocksForm) by the blocks' (T547.3.7); any other by the plain one */
export const keyKindOf = (dtype: Dtype, blocks = false): "plain" | "ternary" | "blocks" =>
  packingOf(dtype)?.gpu?.as === "ternary" ? "ternary" : blocks ? "blocks" : "plain";
/** T156: whether a verdict the page kept (alone: { key, cpu }, the device's key and /benchmark/'s CPU then) still holds:
 * the same device, browser and shaders (the key, T148's), and the same CPU reading of /benchmark/ (a new run of its
 * CPU section, or none now, asks again) */
export const aloneHolds = (alone: Alone | null | undefined, key: string | undefined, cpu: CpuSpeed | undefined) => Boolean(alone && key && alone.key === key && cpu &&
  alone.cpu?.GBps === cpu.GBps && alone.cpu?.promptGMACs === cpu.promptGMACs);

// T220: where the GPU's device binds a buffer (minStorageBufferOffsetAlignment). gpu.ts asks its device for the
// adapter's limits of size but not for this one, so the device has WebGPU's default, 256, whatever smaller value the
// adapter says it could give: gpuOnlyUnfit judges by the device's value, the one gpu.ts's tokensLayout and piecesOf
// read (asking the device for the adapter's value instead would move where every model's pieces start on the GPU, the
// prompt's too, for no model of the list: they all start on 256, T213)
const BINDS_AT = 256;
/** T156: why a model cannot go on the GPU alone, from its header, dtype and form before its bytes come, and the
 * adapter the worker asked for ({ fallback, limits }); null where it can. A model whose steps the GPU takes (T152): by
 * its facts (T375.5), one of RMSNorms and a gated FFN with no table of positions, no gate of the attention's output
 * and no layer that keeps a state: of the architectures there are, a Llama (T226: Qwen2 and Qwen3 with it, whose
 * biases of q, k and v and norms of the heads are vectors that stay in this memory as the norms' weights do; not GPT-2
 * or GPT-NeoX, and no such model of the list is too large to hold twice). Of a packing that may be there alone (its
 * line of jobs.ts's PACKED: int8; not six bits, which are for a device short of memory, where the widened int8 on the
 * GPU, 1.125 bytes a weight against six bits' 0.875, would not fit either: T155's review; T232 ternary, where the
 * browser's WGSL has the packed int8 dot: adapter.packed),
 * heads of a multiple of 4, and q, k and v and gate and up each one range of a buffer the device binds (gpu/weights.ts's
 * tokensLayout says the last word: a GPU that refuses then means the model is loaded again on the CPU). */
// T219: what a model on the GPU alone says where the GPU sampled an id outside the vocabulary (its logits were not
// finite numbers, most likely: the words of T195's NOT_FINITE, llama2_numpy.py), since no CPU can take the step again
export const OUTSIDE_VOCABULARY = "The model computed logits on the GPU that are not finite numbers (NaN or infinity), so no token can be drawn: its weights are broken or its numbers overflowed.";
// the activations of a gated FFN the GPU's shaders have: SwiGLU's, GEGLU (T261.4) and (T344) GEGLU of the exact GELU;
// gpufit.ts reads it too. (T344 review: the one list of them: the GPU's tables of a gate by its activation
// (gpu/block.ts's GATE_SHADERS, gpu/tokens.ts's GLUS, gpu/reference.ts's GATES) are each a Record of Gated, so tsc says
// where one lacks a row, and gpu/activations.ts's gatedBy() refuses an activation outside it rather than take it as
// SwiGLU's)
export const GATED = ["silu", "gelu", "gelu_erf"] as const satisfies readonly Activation[];
export type Gated = (typeof GATED)[number];
export const isGated = (activation: Activation): activation is Gated => (GATED as readonly Activation[]).includes(activation);
/** T382: why a model of these facts cannot go on the GPU alone, or null: the first fact that keeps it off, each by its
 * own reason (stateful: it has a layer that keeps a state the GPU alone does not hold, T271.4: an LFM2's convolution
 * layers; the rest are its architecture's line). Apart from gpuOnlyUnfit() so that a set of facts no architecture of
 * the table has can be asked too (the tests). (T271.4: a gate of the attending layers' output is a token's layer's now,
 * a Qwen3.5's; T344: and a Spark-X2.5's gate of its heads) */
export function aloneFactsUnfit({ stateful, positions, gatedFfn, layerNorm, activation }:
  Pick<Architecture, "positions" | "gatedFfn" | "layerNorm" | "activation"> & { stateful: boolean }) {
  if (stateful) return REASONS.stateful;
  if (positions) return REASONS.alonePositions;
  if (!gatedFfn) return REASONS.aloneFfn;
  if (layerNorm) return REASONS.aloneNorms;
  // T261: an activation none of the GPU's gates has (GATED above), whatever it is (a BitNet's squared ReLU).
  // T334.3: a Gemma 3's and a Gemma 4's GELU gate and their layers that attend a window are the GPU alone's too (its
  // layers are the same shaders as the prompts' and the steps' beside the CPU, T261.4: what differs is where the weights
  // and the keys and values are), and so are a PLaMo 3's windows; T344: and a Spark-X2.5's exact GELU and gate of its heads
  if (!isGated(activation)) return REASONS.aloneActivation;
  return null;
}
// T271.1: the largest block of a rotated basis the GPU's ROTATE_QUANTIZE and UNROTATE hold in a workgroup's memory
// (shaders/rotate.ts's ROTATE_MOST, a second copy here: no shader's text comes into this chunk; tests/gpu-part-check.ts
// holds it to that one), and whether a rotated basis's block is one they take: a power of two (the butterflies) of 32 (a
// group of the 8 bits a thread) to that most (0: the model's own basis, which they are not asked of)
export const ROTATE_MOST = 1024;
export const rotatedBlockUnfit = (block: number) => Boolean(block) && ((block & (block - 1)) !== 0 || block < 32 || block > ROTATE_MOST);

/** T547.3.7: the kinds a form's rows have of their own (Saluki's types of llama.cpp's blocks, by row and by layer), and
 * whether the model is one of llama.cpp's blocks (its GPU's shaders and key are a model of blocks': the GPU's worker's
 * ownOf() says so of its plan, the worker of its form) */
export const kindsOfForm = (form: { kinds?: Record<string, unknown> | null }) => Object.values(form.kinds ?? {}).flat() as string[];
export const blocksForm = (form: { kinds?: Record<string, unknown> | null }) => kindsOfForm(form).some((kind) => packingOf(kind)?.tabled !== undefined);
export function gpuOnlyUnfit(header: Header, dtype: Dtype, form: GpuForm & { rotated?: Rotated | null; kinds?: Record<string, unknown> | null } = {},
  adapter: Adapter | null | undefined, force: Force = {}) {
  const [dim, hidden] = header;
  if (!adapter) return REASONS.noAdapter;
  if (adapter.fallback && !force.fallback) return REASONS.aloneFallback;
  // T375.5: by the model's facts, not by its architecture's name: RMSNorms, a gated FFN, no table of positions, and
  // every layer attending (which of the five architectures a Llama's alone are). T271.4: or a Qwen3.5's linear-attention
  // layers, whose state the GPU keeps itself, and the gate of its attending layers' output, which a token's layer has
  const sizes = sizesOfForm(header, form, LAYER_KINDS), { q: qDim, kv: kvDim } = sizes.widths;
  // (T382: the architecture's line as it is, so that no fact of it is handed on by hand, and the one the sizes say.
  // T641.3: a layer of a kind the GPU alone may not hold, kinds.ts's gpu.alone)
  const facts = aloneFactsUnfit({ ...architectureOf(form.arch ?? "llama"), stateful: sizes.kinds.some((kind) => !LAYER_KINDS[kind as LayerKind].gpu?.alone) });
  if (facts) return facts;
  // T334.3's review: nor attention whose scores are capped (a Gemma 2's: its GELU gate and windows pass the facts above
  // since T334.3), which no shader has: refused here, before its bytes go to the GPU's worker, and not by the engine
  // once built (gpufit.ts's scoresCapped), which would load it again on the CPU
  if (form.score_softcap) return REASONS.scoresCapped;
  // T334.3: a model with a table of every layer's inputs (a Gemma 4's) where a test asks for the GPU alone (?gpuTest=only)
  // and nowhere else yet: on a device that says 8 GB its E2B is as much too large to hold twice as its E4B, and where the
  // E2B goes is the owner's choice (TASKS.md's T334.3: this line goes then)
  if (form.per_layer && !force.only) return REASONS.alonePerLayer;
  // T271.1: nor a rotated basis whose blocks the GPU's shader does not take
  if (form.rotated && rotatedBlockUnfit(Number(form.rotated.block))) return REASONS.rotatedBlock;
  // T363.5: nor a model whose rows have kinds of their own (llama.cpp's blocks are the CPU's kernels' alone: gpu.ts
  // takes one packing for a file). T547.3.7: but of kinds the GPU takes, every one (Saluki's ten types of blocks, by row
  // and by layer): its shaders need the packed int8 dot, and its matrices are joined or apart as the GPU's worker finds
  // them (tokensLayout()), so the starts below are not asked of it
  const own = kindsOfForm(form);
  if (own.some((kind) => !packingOf(kind)?.gpu)) return REASONS.aloneDtype("rows of kinds of their own: such");
  if (own.length && !adapter.packed) return REASONS.packed(String(own.find((kind) => packingOf(kind)!.gpu!.dot) ?? dtype));
  // (the packing's line of jobs.ts's PACKED: whether a model of it may be on the GPU alone, and what its shaders need)
  const packing = packingOf(dtype), held = packing?.gpu;
  if (!packing || !held?.alone) return REASONS.aloneDtype(dtype);
  if (held.dot && !adapter.packed) return REASONS.packed(dtype);
  if ((qDim / sizes.heads) % 4) return REASONS.aloneHeads;
  const { maxStorageBufferBindingSize, maxBufferSize } = adapter.limits;
  const binds = Math.min(maxStorageBufferBindingSize, maxBufferSize);
  // (a joined matrix's parts start where the device binds: the values a row's bytes on, the scales four bytes a group.
  // T634.3: the bytes on the GPU, the packing it is held as there's: binary's are ternary's codes, twice its own)
  const heldOnGpu = PACKED[held.as];
  const starts = [qDim * dim, (qDim + kvDim) * dim, hidden * dim];
  if (!own.length && starts.some((weights) => heldOnGpu.rowBytes(weights) % BINDS_AT || ((weights / packing.group) * 4) % BINDS_AT)) return REASONS.aloneStarts;
  // (T271.4: and a Qwen3.5's linear-attention layers' matrices, each one piece: a token's layer reads them whole.
  // T641.3: every matrix of the kinds' own, kinds.ts's gpu.alone: an attending layer's are within the joined ones above)
  // (T547.3.7's review: of rows of kinds of their own, by the widest of them: Saluki's are no wider than Q4_K's, 0.56 of a
  // byte a weight, and int8's count would keep it off a device that binds 128 MiB)
  const widest = own.length ? Math.max(...own.map((kind) => packingOf(kind)!.rowBytes(BLOCK) / BLOCK)) : 0;
  const rowBytes = (weights: number) => (own.length ? Math.ceil(weights * widest) : heldOnGpu.rowBytes(weights));
  const kinds = aloneKinds(sizes);
  // (T334.4: each layer's of its own sizes)
  const layerMatrices = sizes.kinds.flatMap((kind, l) => Object.values(LAYER_KINDS[kind as LayerKind].gpu?.alone?.matrices(sizes, form, l) ?? {}) as number[]);
  if ([(qDim + 2 * kvDim) * dim, 2 * hidden * dim, dim * Math.max(qDim, hidden), ...layerMatrices]
    .some((weights) => rowBytes(weights) > binds)) return REASONS.alonePast;
  // (T573: nor a linear-attention layer whose state alone is past a buffer: the GPU binds a place's state whole)
  if (kinds.some(({ alone }) => (alone.state?.(sizes) ?? 0) > binds)) return REASONS.aloneState;
  return null;
}
/** T375.5: whether a model of this dtype may be on the GPU alone at all (its packing's line of PACKED): what the worker
 * asks before it opens a file to keep a conversion in as it comes */
export const aloneTakes = (dtype: string) => Boolean(packingOf(dtype)?.gpu?.alone);
/** T547.1: the packing of a matrix or a table of the GPU's plan: a stored kind the GPU takes, its line of PACKED having
 * a gpu (what gpuFit() and gpuOnlyUnfit() have let through: one it does not take is a mistake of theirs). GpuPacking is
 * those names, which kernel-registry-check holds to PACKED */
export function gpuPacking(kind: string): GpuPacking {
  if (packingOf(kind)?.gpu) return kind as GpuPacking;
  throw new Error(`the GPU takes no matrix of ${kind}`);
}

// T156: the GPU's worker takes the layers' bytes as they come, and says in flow[0] how many it has put on the GPU: the
// loops that write wait (room()) where they are more than FLOW_BYTES ahead (a kept model read from the disk comes
// faster than a GPU takes it, and what waits in the messages is memory), and give up where nothing was taken for
// FLOW_STALL_MS. They wait without blocking: a worker made from a worker may not even start while the one that made
// it blocks (Chromium, gpu-check's first run: the GPU's worker took nothing for 60 s)
// T377: a worker that is gone (it did not start, or threw: whoever made it hears its "error") takes nothing more and
// counts nothing more: gone() says so, and nothing waits for it from then on (else every load waited those 60 s and
// then failed, where the model can be loaded again on the CPU at once)
const FLOW_BYTES = 64 * 2 ** 20, FLOW_STALL_MS = 60000;
/** T156: the weights of a model on the GPU alone. worker: the GPU's worker (gpu.ts, told to open with gpuOnlyPlan()),
 * tensors: llama2_numpy.external_tensors(), size: the checkpoint's bytes. Returns { holes, place, stored, flow,
 * write(offset, bytes), room(), drained() }: write() puts the bytes outside the holes into memory at base +
 * place(offset), and posts those in them to the worker (a copy each); room() resolves once the worker is no more than
 * FLOW_BYTES behind, drained() once it has all of them; stored is what memory holds (the checkpoint less the holes).
 * gone() (T377): the worker takes no more (the bytes for it are dropped from then on, and room() and drained() resolve). */
// T334.3: apart, a stretch of the checkpoint neither in the memory nor on the GPU (a Gemma 4's table of every layer's
// inputs, which stays where the model is kept as it comes: perlayer.ts's tableHole()), or null: its bytes go nowhere
// here, and place() leaves it out as it leaves the GPU's holes out
export function gpuOnlyWeights({ memory, base, size, tensors, worker, apart = null }:
  { memory: WebAssembly.Memory; base: number; size: number; tensors: Tensors; worker: { postMessage(message: ToGpu, transfer: Transferable[]): void }; apart?: Hole | null }) {
  const holes = gpuHoles(tensors), all = apart ? [...holes, apart].sort((a, b) => a[0] - b[0]) : holes, place = placer(all);
  const stored = size - all.reduce((sum, [start, end]) => sum + (end - start), 0);
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
    holes: all, place, stored, flow,
    write(offset: number, chunk: Uint8Array) {
      let at = 0;
      for (const hole of all) {
        const [start, end] = hole;
        if (end <= offset + at || start >= offset + chunk.length) continue;
        const from = Math.max(start - offset, at), to = Math.min(end - offset, chunk.length);
        if (from > at) new Uint8Array(memory.buffer, base + place(offset + at), from - at).set(chunk.subarray(at, from));
        if (hole !== apart) post(offset, chunk, from, to);
        at = to;
      }
      if (at < chunk.length) new Uint8Array(memory.buffer, base + place(offset + at), chunk.length - at).set(chunk.subarray(at));
    },
    room: () => wait(BigInt(FLOW_BYTES)),
    /** every byte posted is on the GPU (or was dropped by a GPU that failed) */
    drained: () => wait(0n),
    /** T377: the GPU's worker takes no more of them: what waits for it stops waiting, now */
    gone() {
      gone = true;
      Atomics.notify(taken, 0);
    },
  };
}
// T334.1: what a layer's places in a Gemma 4's stacks are read from (Python's layer_facts(): the plan's layers, or T334.3
// stackFactsOf() below, of the form): its kind, whether its heads are the wide ones, its place in its attention's stacks (q, o) and
// in those of k and v (-1: it reads another's), whether its FFN is the wider one, and its place in its FFN's stacks
export type StackFact = { kind: LayerKind; wide: boolean; stack: number; kept: number; double: boolean; ffn: number };
/** T334.1: every layer's place in a Gemma 4's stacks of its narrow and wide heads and of its narrower and wider FFN, by
 * the tensor's name (-1: the layer has none of it), and of the projection of its embedded rows into the inputs of every
 * layer (perLayer), a matrix of the model's, not a layer's: the first's. None where every stack is every layer's (no wide
 * heads, no wider FFN, no layer that reads another's keys and values, no such inputs). T334.3: of the plan's layers
 * (engine.ts, the GPU's plan as it starts) and of the form's (gpuOnlyPlan() below, the plan a model on the GPU alone
 * opens with, before there is a plan): the two are the same places */
export function stackPlacesOf(facts: StackFact[], perLayer: boolean): Partial<Record<TensorName, number[]>> {
  if (!facts.some(({ wide, double, kept }) => wide || double || kept < 0) && !perLayer) return {};
  return Object.fromEntries(([
    [["per_layer_projection"], (_: StackFact, l: number) => (l === 0 ? 0 : -1)],
    [["wq", "wo"], ({ kind, wide, stack }: StackFact) => (kind === "attention" && !wide ? stack : -1)],
    [["wq_wide", "wo_wide"], ({ kind, wide, stack }: StackFact) => (kind === "attention" && wide ? stack : -1)],
    [["wk", "wv"], ({ kind, wide, kept }: StackFact) => (kind === "attention" && !wide ? kept : -1)],
    [["wk_wide", "wv_wide"], ({ kind, wide, kept }: StackFact) => (kind === "attention" && wide ? kept : -1)],
    [["w1", "w2", "w3"], ({ double, ffn }: StackFact) => (double ? -1 : ffn)],
    // (T334.2: and of its wider FFN, which a token's steps run: no block runs a layer of it)
    [["w1_double", "w2_double", "w3_double"], ({ double, ffn }: StackFact) => (double ? ffn : -1)]] satisfies [TensorName[], (fact: StackFact, l: number) => number][])
    .flatMap(([names, place]) => names.map((name) => [name, facts.map(place)] as const)));
}

// T334.3: every layer's facts of its places in the stacks (StackFact above), of the form, as Python's layer_facts()
// says them of the plan (engine/plan.py, its Dims's layers()): a layer's place among the attending layers of its heads'
// size (the form's wide), among those of them that keep keys and values of their own (none of the form's kv_shared:
// -1), and among the layers of its FFN's width (the form's double). A layer of another kind's places are its own kind's
// (no Gemma 4 has one: the stacks of every layer's, where they are of some layers, are a Gemma 4's alone)
function stackFactsOf(kinds: LayerKind[], { wide, kv_shared, double: doubled }: GpuForm): StackFact[] {
  const double = doubled ?? [], attending = kinds.flatMap((kind, l) => (kind === ATTENTION ? [l] : [])), wideLayers = wide?.layers ?? [], shared = kv_shared?.layers ?? [];
  return kinds.map((kind, l) => {
    const isWide = wideLayers.includes(l), alike = attending.filter((one) => wideLayers.includes(one) === isWide), isDouble = double.includes(l);
    const place = kinds.slice(0, l).filter((other) => other === kind).length;
    return { kind, wide: isWide, stack: kind === ATTENTION ? alike.indexOf(l) : place, kept: kind === ATTENTION ? alike.filter((one) => !shared.includes(one)).indexOf(l) : place,
      double: isDouble, ffn: kinds.map((_, one) => one).filter((one) => double.includes(one) === isDouble).indexOf(l) };
  });
}
/** T156: what the GPU's worker opens with (gpu/start.ts's open(): the device, and a buffer for every piece of every layer's
 * matrices, before a byte comes): each matrix's rows and length and, a layer each, where its values and scales start in
 * the checkpoint (the offsets the bytes come with). T210: and the tables' ({ classifier, embedding }, the embedding
 * null where the classifier is it: { rows, n, at: [values, scales] } each, in the checkpoint too) */
// T271.4: form, the model's (FORM): a Qwen3.5's linear-attention layers, whose matrices (and the attending layers') are
// of the layers of their kind alone: the plan says every layer's kind and its place among the layers of its kind then,
// and those matrices' kind (of), as the plan of { type: "start" } does
export function gpuOnlyPlan(header: Header, tensors: Tensors, force: Force = {}, remembered?: Remembered, form: GpuForm = {}): Opened {
  const layers = header[2], kinds = sizesOfForm(header, form, LAYER_KINDS).kinds as LayerKind[], mixed = kinds.some((kind) => kind !== ATTENTION);
  const places = kinds.map((kind, l) => kinds.slice(0, l).filter((other) => other === kind).length);
  // (T334.3: every layer's place in a Gemma 4's stacks of some of its layers, as the GPU's plan has them when it starts:
  // stackPlacesOf(), of the form's layers as Python's layer_facts() makes them of its plan)
  const stacks = stackPlacesOf(stackFactsOf(kinds, form), Boolean(form.per_layer));
  // (T232: the bytes of a layer's values are its packing's, jobs.ts's PACKED; T547.1: its name is what the GPU's worker is told)
  const matrices = Object.fromEntries(LAYER_MATRICES.filter((name) => tensors[name]).map((name): [string, Matrix] => {
    // (T334.3: a matrix of the model's, not a layer's, is one layer of a stack: a Gemma 4's projection of its embedded rows)
    const t = tensors[name]!, [rows, n] = t.shape.slice(-2), perLayer = rows * n, layerBytes = t.layers ? 0 : packingOf(t.kind)!.rowBytes(perLayer);
    const at = stacks[name], of = mixed ? kinds.find((kind) => LAYER_KINDS[kind].gpu?.names.includes(name)) : undefined;
    const count = at ? (t.shape.length === 3 ? t.shape[0] : 1) : of ? kinds.filter((kind) => kind === of).length : layers;
    // (T547.3.7: a stack whose layers are each of a kind of their own, each layer where its own part is)
    if (t.layers) {
      return [name, { rows, n, packing: "layers", packings: t.layers.map((part) => gpuPacking(part.kind)), layers: t.layers.map((part): [number, number] => [part.offset, part.scales]),
        ...(of ? { of } : {}) }];
    }
    return [name, { rows, n, packing: gpuPacking(t.kind), layers: Array.from({ length: count }, (_, l) =>
      [t.offset + l * layerBytes, t.scales + (l * perLayer / t.group) * packingOf(t.kind)!.scale]), ...(of ? { of } : {}), ...(at ? { at } : {}) }];  // (T533: ternary's scales float16s)
  }));
  const table = (t: Tensor): Table => ({ rows: t.shape[0], n: t.shape[1], packing: gpuPacking(t.kind), at: [t.offset, t.scales] });
  const embedding = tensors.token_embedding_table!, classifier = tensors.wcls ?? embedding;
  const tables = { classifier: table(classifier), embedding: classifier.offset === embedding.offset ? null : table(embedding) };
  return { layers, matrices, tables, force, remembered, ...(mixed ? { kinds, places } : {}) };
}

// (T349) what forward.ts and the other modules read besides, which was not exported where it was one file
export { GPU_ALONE };
