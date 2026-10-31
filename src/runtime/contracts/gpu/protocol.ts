// contracts/gpu/protocol.ts (T.368.2): what forward.ts and the model's GPU worker say to one another. Types alone:
// nothing of this file is in what a browser gets (it is read with `import type`).
//
// The worker is src/runtime/gpu.ts (its head says what each message does), which forward.ts makes
// (src/runtime/forward/gpuside.ts) and hands the model in a plan: Started, of { type: "start" }, a model whose layers are
// in the shared memory too; Opened, of { type: "open" } (src/runtime/forward/alone.ts's gpuOnlyPlan(), posted by
// src/runtime/worker/weights.ts), a model on the GPU alone (T.156), whose bytes follow as { type: "weights" }.
// A prompt's block, a generation's steps and the keys and values are asked for by a message and answered in the control
// area: the words of that answer are src/runtime/jobs.ts's constants (GPU_DONE, GPU_FAILED, GPU_BEAT, GPU_WANTED), which
// the plan carries as words. They are numbers there, and stay that.
//
// Both ends are JavaScript still, which no compiler reads: tests/plan-keys-check.ts holds the plans' keys written
// (forward/) to the keys read (gpu/) and to these types, and the messages' kinds and keys to ToGpu and FromGpu.

import type { Activation, LayerKind, TensorName } from "../plan.ts";

// T.547.1: how a matrix's or a table's values are packed at their addresses, the checkpoint's own (jobs.ts's PACKED
// names): int8, int6 (T.155: widened to int8 as it goes up), ternary (T.232: two bits a weight, as they are) or Q4_K
// (T.547.2.3: llama.cpp's blocks of 256, as they are, their scales in them)
export type GpuPacking = "int8" | "int6" | "ternary" | "Q4_K" | "Q2_K" | "IQ2_XXS" | "IQ2_XS" | "IQ2_S" | "IQ3_XXS" | "IQ3_S" | "IQ1_S" | "IQ1_M" | "IQ4_XS";
// a matrix of every layer: rows of n values, packed so, and for every layer [where its values are, where its scales
// are (T.547.2.3: unread where they are in its blocks: Q4_K's)]. T.545: of: the kind of the layers it is a matrix of (an LFM2's q is its attending layers', its matrix in its
// convolution layers'), whose places index layers; none: every layer's. T.547.3.1: packing "layers", a stack whose
// layers are each of a packing of its own (Saluki's, llama.cpp's types a tensor), and packings, each layer's by its place
export type Matrix = { rows: number; n: number; packing: GpuPacking | "layers"; packings?: GpuPacking[]; layers: [number, number][]; of?: LayerKind };
// a table (the classifier, the embedding): as a matrix, at [values, scales]
export type Table = { rows: number; n: number; packing: GpuPacking; at: [number, number] };
// a vector of every layer: where it is, and its size a layer (T.545: of, as a matrix's)
export type Vector = { at: number; size: number; of?: LayerKind };

// The vectors a model has (gpuside.js's gpuVectors()): the norms' weights always; LayerNorm's biases (T.154), Qwen2's
// biases of q, k and v and Qwen3's norms of a head (T.153), a GPT-2's biases after its matrices, where the model has them
export type Vectors = {
  attention: Vector;
  ffn: Vector;
  attentionBias?: Vector;
  ffnBias?: Vector;
  bq?: Vector;
  bk?: Vector;
  bv?: Vector;
  qNorm?: Vector;
  kNorm?: Vector;
  bo?: Vector;
  b1?: Vector;
  b2?: Vector;
  // T.261.4: a Gemma 3's norms behind the attention and behind the FFN
  attentionPost?: Vector;
  ffnPost?: Vector;
  // T.545: the taps of an LFM2's convolution layers (taps rows of dim a layer; T.271.2: a Qwen3.5's linear-attention
  // layers', taps rows of their mixed q, k and v)
  conv?: Vector;
  // T.271.2: a Qwen3.5's linear-attention layers' gates: their two small matrices (float32, a row of dim a value head),
  // dt_bias and the decay (one a value head), and the norm of a value head
  wb?: Vector;
  wa?: Vector;
  dtBias?: Vector;
  decay?: Vector;
  deltaNorm?: Vector;
};

// T.625: a plan's matrices, by the names of the plan's tensors, and only those the model has
export type Matrices = Partial<Record<TensorName, Matrix>>;

// T.152: what the worker takes for a generation's steps (gpuside.js's gpuTokensPlan()): the classifier and, where it is
// another table, the embedding; the final norm (its bias and GPT-2's positions: 0 where the model has none); whether
// the classifier has outlier channels (T.92) and which; where the ids go; the most steps of a submission
export type Tokens = {
  classifier: Table;
  embedding: Table | null;
  final: number;
  finalBias: number;
  positions: number;
  outliers: boolean;
  channels: number[];
  ids: number;
  most: number;
};

// T.210: the tables of a model on the GPU alone, which come with the checkpoint's bytes
export type Tables = { classifier: Table; embedding: Table | null };

// the words of the control area a request is answered in (jobs.ts's GPU_DONE, GPU_FAILED, GPU_BEAT, GPU_WANTED)
export type Words = { done: number; failed: number; beat: number; wanted: number };

// What the tests ask for (the page's ?gpuTest=: fallback, always, quick, only; tests/gpu-check.ts: the rest). Nothing
// of it is set for a visitor.
export type Force = {
  fallback?: boolean;
  always?: boolean;
  quick?: boolean;
  only?: boolean;
  matrices?: string;
  attention?: string;
  tokens?: string;
  tokenAttention?: string;
  pieceBytes?: number;
  tablePieceBytes?: number;
  // T.565: the most bytes of a bundle of a linear-attention layer's states (gpu/linear.ts), so that a small model's go
  // in bundles as the 27B's past a buffer of the device do
  stateBytes?: number;
};

// what the page kept of an earlier visit on this device (T.148, T.152): the shaders chosen then, under the device's key
export type Remembered = { key?: string; matrices?: string; attention?: string; tokens?: string; alone?: unknown };

// T.568.4: the numbers of the kinds of layers that keep a state, each under its kind's name (forward/kinds.ts's gpu.plan()
// says its own: forward/convolution.ts, linear.ts), null where the model has none of that kind. A new kind's numbers are
// a key here
export type KindNumbers = {
  convolution: { taps: number } | null;
  linear: { keyHeads: number; valueHeads: number; keyDim: number; valueDim: number; taps: number } | null;
};

// The plan of { type: "start" }: the model's sizes, what RoPE turns, its norm and residual, and addresses in the
// shared memory (the matrices, the vectors, rows: the rows forward.ts embeds, cos and sin: RoPE's tables, staging:
// where a block's keys and values come back, and T.545 the state of the layers that keep one after them: where it goes
// up from too, T.271.2)
export type Started = KindNumbers & {
  dim: number;
  hidden: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headSize: number;
  // the values of a head RoPE turns (0: a GPT-2), and the layers it leaves alone (T.255)
  turned: number;
  unturned: number[];
  seqLen: number;
  kvStart: number;
  eps: number;
  layerNorm: boolean;
  parallel: boolean;
  // T.261.4: the activation of the FFN (its gate's, where it has one: a Gemma 3's GELU); every layer's window, the last
  // positions it attends (0: every one); the layers RoPE turns by the local tables, and where those are (0: none)
  activation: Activation;
  windows: number[];
  local: number[];
  localCos: number;
  localSin: number;
  // T.545: every layer's kind ("attention", or "convolution": an LFM2's; T.271.2 "linear": a Qwen3.5's) and its place
  // among the layers of its kind; of the convolution layers, the taps; T.271.2: of the linear-attention layers, their
  // heads and their sizes and the taps of their convolution; null where the model has none. The state of those layers
  // is staged after the keys and values (staging), as the CPU's kinds put it there (forward/convolution.ts, linear.ts:
  // the convolution layers' rows; the linear-attention layers' states, then their rows). The GPU's own keys and values,
  // and the places a block's come back to, are the attending layers' alone
  kinds: LayerKind[];
  places: number[];
  // T.271.1: a rotated basis (T.237): its block, and where the signs of each width a matrix reads are (float32, Python's,
  // each times 1 / sqrt(block)); null: the model's own basis
  rotated: { block: number; signs: { width: number; at: number }[] } | null;
  // the most tokens of a block
  batch: number;
  matrices: Matrices;
  // T.547.3.4: where the codebooks of llama.cpp's blocks are (the checkpoint's row "codebooks", engine/dtypes.py's
  // CODEBOOKS bytes: convert/grids.py's codebooks()); 0: the model has none
  codebooks: number;
  vectors: Vectors;
  rows: number;
  tokens: Tokens | null;
  cos: number;
  sin: number;
  staging: number;
  force: Force;
  remembered: Remembered | undefined;
  // the model is on the GPU alone (it was opened first)
  direct: boolean;
  words: Words;
};

// The plan of { type: "open" } (gpuOnlyPlan()): the layers and their matrices by where they are in the checkpoint
// (T.271.4: kinds and places, as Started's, where some layers are of another kind than attention: a Qwen3.5's)
export type Opened = { layers: number; matrices: Matrices; tables: Tables; force: Force; remembered: Remembered | undefined; kinds?: LayerKind[];
  places?: number[] };

// forward.ts's cache of the keys and values a generation's steps go up from: where each attending layer's are now (T.261.4.2,
// places: the address of position 0's keys and values, position p's p rows after; or, where ring is not 0, a ring of so
// many positions that holds position p at p mod ring, T.510.1), the bytes of a position, and whether they are float16
export type CachePlace = { keys: number; values: number; ring: number };
export type Cache = { places: CachePlace[]; row: number; half: boolean };
// the sampling of the steps (the engine's settings; T.546: topK, minP and presence too, 0 where none; stops: the stop
// tokens)
export type Sampling = { temperature: number; topp: number; penalty: number; topK: number; minP: number; presence: number; stops: number[] };

export type ToGpu =
  | { type: "open"; plan: Opened; flow: SharedArrayBuffer }
  | { type: "weights"; offset: number; bytes: Uint8Array }
  | { type: "start"; memory: WebAssembly.Memory; plan: Started }
  // T.210: tokens, the ids of a model on the GPU alone, which embeds them there. T.271.2: fresh, the request goes on from
  // the GPU's own state of the layers that keep one (none, or false: from the CPU's, staged; position 0: none)
  | { type: "prompt"; serial: number; count: number; pos: number; fresh?: boolean; tokens?: number[] }
  | { type: "tokens"; serial: number; count: number; pos: number; fresh?: boolean; from: number; token: number; history: number[]; length: number;
      written: number; cache: Cache | null; settings: Sampling; randoms: number[] }
  | { type: "keys"; serial: number; count: number; pos: number }
  | { type: "stop" };

// what a form (a shader's way with the matrices, the attention, a token's layer) came to where it was tried: its ms,
// or that it was remembered, or why it is none
export type Tried = { name: string; ms?: number; remembered?: boolean; none?: string };
// T.152, T.224: the layer of a token chosen and its ms a step, the attention of a token, the pieces of the classifier
export type TokensChosen = { form: string; ms: number | undefined; forms: Tried[]; remembered: boolean; pieces: number; attention: string; attentions: Tried[] };

export type FromGpu =
  | { type: "progress" }
  // form, attention, forms, remembered: gpu/forms.js's formsSaid(); blocks: what a whole block of count tokens takes
  | { type: "ready"; adapter: string; key: string; bytes: number; seconds: number; form: string; attention: string; forms: Tried[];
      remembered: boolean; blocks: { count: number; ms: number }[]; tokens: TokensChosen | null; tokensWhy: string | null | undefined }
  | { type: "unusable"; reason: string }
  | { type: "failed"; reason: string }
  | { type: "ended" };
