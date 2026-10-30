// forward/gpufit.ts (T.457.1, from forward/gpuside.ts): what the GPU would hold of one engine's model and why it would not
// take it (T.349), and the types of what createForward() (engine.ts) hands it and gpuside.ts.
// A module of src/runtime/forward.ts; it reads jobs.ts and its neighbours by static imports.

import { PACKED, packingOf } from "../jobs.ts";
import { gpuWeightBytes } from "./alone.ts";
import { REASONS } from "./reasons.ts";
import { rotatedBlockUnfit } from "./alone.ts";
import type { Packed } from "../jobs.ts";
import type { Held } from "./kinds.ts";
import type { Direct } from "./alone.ts";
import type { KindNumbers, Vector, Vectors } from "../contracts/gpu/protocol.ts";
import type { Activation, LayerKind, Tensor } from "../contracts/plan.ts";

// ---- (T.368.4, the types) what createForward() (engine.ts) makes once and hands gpuFit() and gpuSide() (gpuside.ts)
// the model's facts
export type Shape = {
  dim: number; hidden: number; layers: number; heads: number; kvHeads: number; headSize: number; kvDim: number; qDim: number; vocab: number; seqLen: number;
  // the bytes of a token's row, and of a position's keys in the cache; whether the cache is float16
  D: number; KV: number; halfKV: boolean;
  eps: number; rotary: number; kvStart: number; turning: boolean; layerNorm: boolean; gatedFfn: boolean; parallel: boolean; unturned: number[];
  // T.261: the FFN's activation, and whether some layer attends a window or is turned by tables of its own; T.261.4: and
  // every layer's window (0: every position) and the layers turned by those tables; extras: what a Gemma 4 has besides
  // (a norm of v's heads, a cap of the logits, a scale of a layer, layers of other widths, keys and values shared)
  activation: Activation; local: boolean; windows: number[]; locals: number[]; extras: boolean;
  // T.568.4: what the GPU's side reads of each kind of the model's layers that the GPU's worker has a step for, in the table's
  // order (forward/kinds.ts's gpu): the names of the tensors of its layers alone, why the GPU would not take them (null:
  // it would), its numbers in the GPU's plan
  gpuKinds: { kind: LayerKind; names: string[]; unfit: string | null; plan: Partial<KindNumbers> }[];
  // T.375.2: some layer keeps a state in place of keys and values, of whatever kind (the plan's layers say)
  stateful: boolean;
  // the block of a rotated basis (0: the model's own basis)
  rotated: number;
  sharedClassifier: boolean;
  // T.545: every layer's kind and its place among the layers of its kind; how many layers keep keys and values in the
  // cache (every one, but an LFM2's convolution layers and a Qwen3.5's linear-attention layers); and the bytes of the
  // state the GPU's worker goes on from, as it goes up and comes back (an LFM2's convolution layers' rows, T.271.2 a
  // Qwen3.5's linear-attention layers' states and rows; 0: none)
  kinds: LayerKind[]; places: number[]; attending: number; stateBytes: number;
};
// T.375.4, T.375.5: whether the GPU takes a matrix as it is packed: its packing's line of jobs.ts's PACKED says how the GPU
// holds it (gpu.ts widens int6 to int8, T.155, and takes ternary as it is, T.232), in the groups that line says. A
// matrix that is not packed, or whose packing the GPU has not, stays on the CPU. (T.547.3.1: a stack whose layers are each
// of a kind of their own, where the GPU takes every layer's)
const onGpuAs = (m: Held): boolean => {
  if (m.kind === "layers") return m.layers!.every(onGpuAs);
  const packing = packingOf(m.kind);
  return Boolean(packing?.gpu) && m.group === packing!.group;
};
// T.545: the kind of the layers a matrix or a vector of the GPU's is of, where it is not every layer's (its places index
// those layers alone: an LFM2's attending layers are some of its layers). T.568.4: the kind of the model's that names it
// (forward/kinds.ts's gpu: an attending layer's q, k, v and o; T.271.2 a Qwen3.5's linear-attention layers'; wout and conv
// are those of the kind that keeps a state, whichever the model has: no model has both, and one that did would be the
// first of the table's)
export const kindOf = (name: string, { gpuKinds }: { gpuKinds: Shape["gpuKinds"] }) => gpuKinds.find(({ names }) => names.includes(name))?.kind;
// where the model's tensors are in the memory: the matrices (null: the model has none of that name), and the addresses
// of the vectors and the tables (0: none)
export type Placed = {
  base: number; embedding: Tensor;
  wq: Held | null; wk: Held | null; wv: Held | null; wo: Held | null; w1: Held | null; w2: Held | null; w3: Held | null; wcls: Held | null;
  attW: number; ffnW: number; attB: number; ffnB: number; finalW: number; finalB: number; bo: number; b1: number; b2: number; bq: number; bk: number; bv: number;
  qNorm: number; kNorm: number; cosTable: number; sinTable: number; positions: number; channels: number[];
  // T.261: the norms behind the attention and behind the FFN (0: the model has none); T.261.4: RoPE's tables of the
  // local layers (0: none)
  attPost: number; ffnPost: number; localCos: number; localSin: number;
  // T.271.1: the signs of a rotated basis by the width a matrix reads (their addresses: null in the model's own basis)
  signs: Record<number, number> | null;
  // T.547.3.4: the codebooks of llama.cpp's blocks, the checkpoint's row (its address: 0, a file with no such rows)
  codebooks: number;
  // T.568.4: the matrices and vectors the kinds of its layers have of their own on the GPU, by the GPU's names (forward/kinds.ts's
  // gpu: T.545 an LFM2's matrices in and out of its convolution layers and their taps; T.271.2 a Qwen3.5's linear-attention
  // layers' matrices and their gates' vectors, and the gate of its full-attention layers' output), where the model has them
  own: { matrices: Record<string, Held>; vectors: Record<string, Vector> };
};
// what the GPU is to take of the model (gpuFit()'s matrices and vectors), why it would take none or no steps, and the
// places in the memory it answers into (0: none)
export type Taken = { matrices(): Record<string, Held>; vectors(): Vectors; whyNot: string | null; tokensWhyNot: string | null; staging: number; rows: number; ids: number };

/** T.349: what the GPU would hold of one engine's model, and why it would not take it (createForward() asks before it
 * allocates the places the GPU writes to).
 * T.378.2: shape, the model's facts, and weights, where its tensors are in the memory (createForward() makes both once,
 * for this and for gpuSide()); device: { gpuRoom, sharedMemory, oneThread }, what the device and the page give;
 * direct: the model is on the GPU alone (T.156). */
export function gpuFit({ shape, weights, device, direct }:
  { shape: Shape; weights: Placed; device: { gpuRoom: number | undefined; sharedMemory: boolean; oneThread?: boolean }; direct: Direct | undefined }) {
  const { dim, layers, headSize, vocab, seqLen, hidden, kvDim, qDim, D, layerNorm, gatedFfn, activation, local, extras, gpuKinds, stateful, rotated,
    sharedClassifier, kinds, attending, stateBytes } = shape;
  const { embedding, wq, wk, wv, wo, w1, w2, w3, wcls, attW, ffnW, attB, ffnB, bo, b1, b2, bq, bk, bv, qNorm, kNorm, positions, attPost, ffnPost, own, channels } = weights;
  // T.545: how many layers a matrix or a vector of that name is of (every layer, or those of its kind: kindOf())
  const countOf = (name: string) => (kindOf(name, shape) ? kinds.filter((kind) => kind === kindOf(name, shape)).length : layers);
  const { gpuRoom, sharedMemory, oneThread } = device;
  // why this model's prompt stays on the CPU, or null: the first stage (T.135) takes Llama's layers of int8 weights
  // (T.153: with Qwen2's biases, Qwen3's norms of the heads and heads of another size than dim / heads as well; T.154:
  // GPT-2's and GPT-NeoX's as well, whose FFN has no gate: w3 is null; T.155: int6 weights and 64-bit memories too)
  // (a function, not a const: gpuUnfit runs before this line, AGENTS.md)
  function gpuMatrices() {
    // (T.568.4: and the kinds' own, after every model's)
    return Object.fromEntries(Object.entries({ wq, wk, wv, wo, w1, w2, w3, ...own.matrices }).filter(([, m]) => m)) as Record<string, Held>;
  }
  // (T.155: int6 weights too, widened to int8 on the GPU, and a model in a 64-bit memory: gpu.ts; T.232: ternary weights
  // as they are, in their groups of 128. The 27B, T.233, is ternary in a rotated basis with linear-attention layers: the
  // GPU's worker has both since T.271.2 and T.271.1, a rotated basis the packed forms' alone, which it chooses among)
  function gpuUnfit() {
    // (T.568.4: why the GPU would not take the layers of a kind, the kind's to say: T.229, a Qwen3.5's linear-attention
    // layers were kept off here until T.271.2, and those of heads wider than the delta rule takes still are)
    const kindUnfit = gpuKinds.find(({ unfit }) => unfit)?.unfit;
    if (kindUnfit) return kindUnfit;
    // (T.260: an LFM2's convolution layers were kept off here until T.545, which gave the GPU's worker a step for them and
    // the state they go on from. T.375.2: and a kind of layer that keeps a state which the engine comes to have, and the
    // GPU's worker has no step for: on the CPU until the GPU has it too)
    if (stateful) return REASONS.stateful;
    // (T.271.1: a rotated basis whose blocks the GPU's shader takes)
    if (rotatedBlockUnfit(rotated)) return REASONS.rotatedBlock;
    // T.375.5 (T.484): the GPU's layers are of two pairs, RMSNorms with a gated FFN (a Llama's) and LayerNorms with
    // GPT-2's FFN (T.154); gpu.ts picks between them by the norms and by whether there is a third matrix. An
    // architecture of another pair, which the CPU's engine runs by its facts (T.375.2), stays there
    if (layerNorm === gatedFfn) return REASONS.pair;
    // T.261.4: and what a Gemma 3 has besides, which the GPU's layers have now: a GELU in the gate of its FFN (GEGLU), a
    // norm behind the attention and behind the FFN (RMSNORM_ADD), layers that attend a window (the flash tiles' window)
    // or are turned by tables of their own (a second table of angles). Not an activation no shader has (a Spark-X2.5's
    // exact GELU), nor norms behind LayerNorms, nor what a Gemma 4 has besides (T.261: until T.261.4 a Gemma 3 was kept
    // off by this line too). An FFN without a gate is GELU's on the GPU, as it was (T.154)
    if ((gatedFfn && activation !== "silu" && activation !== "gelu") || extras) return REASONS.pair;
    if ((attPost || ffnPost) && layerNorm) return REASONS.pair;
    // (T.371, the owner's choice C: a memory not shared on an isolated page is ?threads=1's, or one the browser would
    // not share, or a 64-bit one with no shared kernels for it)
    if (!sharedMemory) return !globalThis.crossOriginIsolated ? REASONS.notIsolated : oneThread ? REASONS.oneThread : REASONS.notShared;
    if (headSize % 4) return REASONS.heads;
    if (!Object.values(gpuMatrices()).every(onGpuAs)) return REASONS.floats;
    // T.148: the layers twice, in this memory and on the GPU (T.156 will keep one): a device with too little memory
    // for both keeps the CPU's alone (a phone or an Apple shares its memory between the two)
    // (T.153, the review: with the GPU's own keys and values, float16, as the prompt may fill the whole context: Qwen3
    // 0.6B's are 0.47 GB at 4096 positions, 95% of its layers')
    const onGpu = layersOnGpu();
    if (gpuRoom !== undefined && onGpu > gpuRoom && !direct) return REASONS.layersRoom(onGpu);
    return null;
  }
  // the bytes of the layers on the GPU, and its own keys and values
  function layersOnGpu() {
    // (T.545: each of the layers it is of, and the keys and values of the attending ones; and the state the convolution
    // layers go on from, with its copy where a block's comes back; T.271.2: the linear-attention layers' too)
    // (T.547.3.1: a stack whose layers are each of a kind of their own, each layer's as its kind)
    return Object.entries(gpuMatrices()).reduce((bytes, [name, m]) => bytes + m.rows * m.n *
      (m.layers ? m.layers.reduce((sum, layer) => sum + gpuWeightBytes(layer.kind), 0) : countOf(name) * gpuWeightBytes(m.kind)), 0) +
      Object.entries(gpuVectors()).reduce((bytes, [name, { size }]) => bytes + countOf(name) * size * 4, 0) + 2 * attending * seqLen * kvDim * 2 +
      2 * stateBytes;
  }
  // T.152: why a generation's steps stay on the CPU, or null. A step on the GPU is T.150's and T.175's fused layer (gpu.ts):
  // Llama's (RMSNorm, RoPE on whole heads, SwiGLU), and T.226: with what the prompt's blocks take besides (T.153: Qwen2's
  // biases of q, k and v, Qwen3's norms of the heads and heads of another size than dim / heads; T.154: GPT-2's and
  // GPT-NeoX's LayerNorm, biases, GELU, learned positions, RoPE on a part of a head, parallel residual), and T.92's
  // outlier channels (the GPU's classifier multiplies floats for such a model, and needs no columns apart); T.261.4.2: a
  // Gemma 3's windows, GELU gate and norms behind; T.545.2: an LFM2's convolution layers; T.271.3: a Qwen3.5's
  // linear-attention layers; T.547: matrices of llama.cpp's blocks. The keys and values are float16 as the GPU's, or
  // float32 where the CPU keeps them so (T.160, widened on the way back and narrowed on the way up). What is left: a
  // classifier and an embedding of int8 or int6 in groups of 32 (T.232: or ternary in groups of 128; of a ternary
  // classifier the GPU takes the outlier channels apart as this does, shaders.ts's TAKE_OUTLIERS; T.547.3.6: or each of
  // its own packing where one is of llama.cpp's blocks); and the memory for the classifier, the embedding where it is
  // another table, RoPE's table and the vocabulary's three arrays of the sampling, besides the layers
  function tokensUnfit() {
    // (T.545: an LFM2's steps stayed on the CPU here until T.545.2, which gave a token's fused layer its convolution layers)
    // (T.271.2: a Qwen3.5's steps stayed on the CPU here until T.271.3, which gave a token's layer its linear-attention layers)
    // (the embedding as the GPU holds the classifier: in the same packing there, and in its groups)
    // (T.547.2.3: and a packing of a scale a group: no EMBED reads llama.cpp's blocks, Q4_K's. T.547.3.6: each table in a
    // packing of its own where one of them is of llama.cpp's blocks, which the GPU embeds and multiplies by its type's
    // shaders (Saluki's: an embedding of IQ1_M, a classifier of IQ3_S); not with outlier channels, whose classifier is
    // multiplied as floats of int8)
    const held = wcls && onGpuAs(wcls) ? packingOf(wcls.kind)!.gpu!.as : null;
    const embedded = packingOf(embedding.kind)?.gpu && embedding.group === packingOf(embedding.kind)!.group ? packingOf(embedding.kind)!.gpu!.as : null;
    const blocks = Boolean(held && embedded) && (!PACKED[held!].scale || !PACKED[embedded!].scale);
    if (!wcls || !held || !embedded || (blocks ? channels.length > 0 : !PACKED[held].scale || embedded !== held)) {
      return REASONS.floatClassifier;
    }
    // (T.548, the owner's choice: a browser that does not say the device's memory (Safari, Firefox) takes them too; T.205
    // kept them off it after an iPhone's tab went down in /benchmark/'s model section with the classifier and the
    // embedding on the GPU as well. Undone for Safari if the owner's iPhone goes down)
    // (T.547.3.6's review: each table as it is packed: Saluki's embedding of IQ1_M is half the bytes of its classifier)
    const table = vocab * dim * gpuWeightBytes(wcls.kind), embeddingBytes = vocab * dim * gpuWeightBytes(embedding.kind);
    // (GPT-2's positions on the GPU as well, a row a position)
    // (T.261.4.2: a Gemma 3's local layers' table of RoPE as well)
    // (T.545.2: and the state as the last step of a request leaves it, read back with its ids)
    const onGpu = layersOnGpu() + table + (sharedClassifier ? 0 : embeddingBytes) + (local ? 2 : 1) * seqLen * headSize * 4 + 3 * vocab * 4 + (positions ? seqLen * D : 0) +
      stateBytes;
    if (gpuRoom !== undefined && onGpu > gpuRoom) return REASONS.classifierRoom(onGpu);
    return null;
  }
  // the vectors of every layer the GPU reads (gpu.ts's plan.vectors): the norms' weights; T.153: Qwen2's biases of q,
  // k and v, Qwen3's norms of a head of q and of k; T.154: GPT-2's and GPT-NeoX's biases of the two LayerNorms, of q,
  // k and v and of o, w1 and w2. Each one's address here and its floats a layer
  function gpuVectors(): Vectors {
    return { attention: { at: attW, size: dim }, ffn: { at: ffnW, size: dim },
      ...(layerNorm ? { attentionBias: { at: attB, size: dim }, ffnBias: { at: ffnB, size: dim } } : {}),
      ...(bq ? { bq: { at: bq, size: qDim }, bk: { at: bk, size: kvDim }, bv: { at: bv, size: kvDim } } : {}),
      ...(qNorm ? { qNorm: { at: qNorm, size: headSize }, kNorm: { at: kNorm, size: headSize } } : {}),
      ...(bo ? { bo: { at: bo, size: dim }, b1: { at: b1, size: hidden }, b2: { at: b2, size: dim } } : {}),
      ...(attPost ? { attentionPost: { at: attPost, size: dim } } : {}), ...(ffnPost ? { ffnPost: { at: ffnPost, size: dim } } : {}),
      // T.568.4: the kinds' own (forward/kinds.ts's gpu: T.545 an LFM2's taps; T.271.2 a Qwen3.5's taps, its gates' two
      // small matrices, their biases, decays and the norm of a value head)
      ...own.vectors };
  }
  // T.545: the same, each with the kind of the layers it is of where it is not every layer's (kindOf()), for the plan
  function gpuVectorsOf(): Vectors {
    return Object.fromEntries(Object.entries(gpuVectors()).map(([name, vector]) => [name, kindOf(name, shape) ? { ...vector, of: kindOf(name, shape) } : vector])) as Vectors;
  }
  return { matrices: gpuMatrices, vectors: gpuVectorsOf, unfit: gpuUnfit, tokensUnfit };

}
