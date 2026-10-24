// A generated token's steps (T.152): the forms of a layer this device can make, a token's buffers and uniforms, the
// dispatches of a step, and a run of steps as one submission with its ids, state, keys and values read back.
// (T.352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T.367.3)

import { STORAGE, COPY_DST, COPY_SRC, MAP_READ, UNIFORM, within, buffer, uniform, normShape, validated,
  pipelineOf, bind, dispatch } from "./device.ts";
import { tablesOf } from "./weights.ts";
import { GROUP } from "../shaders/common.ts";
import { flashVecSplits } from "../shaders/attention.ts";
import { MUL_MAT_VEC_ROWS, ORT_DP4A_MATVEC_ROWS } from "../shaders/fused.ts";
import { STATE_NOT_FINITE, STATE_BYTES, SAMPLING_BYTES, OUTLIERS_MOST, outliersOf } from "../shaders/run.ts";
import { ENGINE } from "../shaders/engine.ts";
import type { Model, Bound, Pass } from "./device.ts";
import type { TablePiece } from "./weights.ts";
import { GPU_KINDS, kindsOf, attendingOf } from "./kinds.ts";
import type { Kept } from "./block.ts";
import type { Steps } from "./kinds.ts";
import type { LayerKind } from "../contracts/plan.ts";
import type { FusedOutput } from "../shaders/fused.ts";

// T.368.6: a form of a token's layer: one of TOKEN_FORMS (Candidate), and, once tokenforms.ts's choose() tries it, its
// pipelines by their keys (tokenCodes()'s: normQuantize, qkv, add, glu, classifier) and its steps' dispatches, made
// for the cache they bind and kept by the attention's parts a head (tokenStep())
type Candidate = { name: string; dp4a: boolean; subgroups: boolean; normApart?: boolean; ternary?: boolean };
// (T.547.2.4: blocks, a pipeline's counterpart for a matrix of Q4_K blocks, where the model has them: tokenCodes' keys
// with ":Q4_K" after them, chooseTokens')
type TokenForm = Candidate & { pipes: { [key: string]: GPUComputePipeline }; blocks?: Map<GPUComputePipeline, GPUComputePipeline>;
  steps?: { cache: Kept | null; lists: Map<number, Pass[]> } };
// a binding and what is bound there (bindAt()), and what a matrix reads: the dispatches before it and its bindings
type Entry = [number, Bound];
type Feed = { norm: Pass[]; quantize: Pass[]; input: Entry[] };
// A token's uniforms: every model's, and what only some have (Qwen2's biases of q, k and v as one, Qwen3's norms of
// the heads, GPT-2's and GPT-NeoX's biases and positions, a ternary classifier's outlier channels; hidden is GELU's
// size, set for every model and read where the FFN has no gate)
// (T.261.4.2: flash a layer, the tiles' Params with the layer's window in them)
// (T.568.2: what a kind of layers has of its own, an attending layer's q, k and v and its attention's among them, is
// the kind's: kinds)
type Uniforms = {
  embed: GPUBuffer[];
  down: GPUBuffer;
  gateUp: GPUBuffer[];
  norm: GPUBuffer[];
  final: GPUBuffer;
  classifier: GPUBuffer[];
  quantizeGate: GPUBuffer;
  quantizeNormed: GPUBuffer;
  biases?: { [name: string]: GPUBuffer }[];
  positions?: GPUBuffer;
  take?: GPUBuffer;
  columns?: GPUBuffer[];
  hidden?: GPUBuffer;
  // T.271.1: a rotated basis's Shape of each width (ROTATE_QUANTIZE's and UNROTATE's: n, xStride, block)
  rotate?: Record<number, GPUBuffer>;
};
// What tokenBuffers() makes, the first part of the model's record of a token (tokenforms.ts's m.tokens): the
// pipelines every form shares, the buffers, the uniforms, and what only some models have (null without)
type TokenBuffers = {
  embed: GPUComputePipeline;
  sample: GPUComputePipeline;
  most: number;
  vocab: number;
  h: GPUBuffer;
  q: GPUBuffer;
  att: GPUBuffer;
  gate: GPUBuffer;
  logits: GPUBuffer;
  probs: GPUBuffer;
  order: GPUBuffer;
  state: GPUBuffer;
  chosen: GPUBuffer;
  randoms: GPUBuffer;
  step: GPUBuffer;
  settings: GPUBuffer;
  xq: GPUBuffer;
  xs: GPUBuffer;
  xb: GPUBuffer;
  readback: GPUBuffer;
  u: Uniforms;
  take: GPUComputePipeline | null;
  columns: GPUComputePipeline | null;
  picked: GPUBuffer | null;
  // T.568.2: what each kind the model's layers are of made of a token (gpu/kinds.ts: its buffers and uniforms, its
  // layers' steps), by its name
  kinds: Partial<Record<LayerKind, Steps>>;
};
// a run of steps: what runTokens() is given, and what it read back (kv where it was asked to keep them)
type Run = { count: number; pos: number; state: GPUAllowSharedBufferSource; settings?: GPUAllowSharedBufferSource; randoms?: Float32Array<ArrayBuffer>; keep?: boolean;
  extra?: (encoder: GPUCommandEncoder) => void };
// (T.545.2: state, the state of the layers that keep one as the last step left it, where kv is: T.271.2, as it goes up)
type Ran = { ids: Uint32Array<ArrayBuffer>; sampled: number; stopped: number; notFinite: number; kv?: Uint8Array<ArrayBuffer>[][]; state?: Uint8Array<ArrayBuffer> };

// ---- T.152: tokens on the GPU. A step of the engine's generate() (the forward pass of the token fed, the sampling of
// the next) as one compute pass of shaders.ts's T.151 run: EMBED of the state's token, every layer in a fused form, the
// final norm and the classifier, SAMPLE (the penalty, softmax, top-p and the draw, with the random number the CPU drew
// for the step), after which the state's first four words go to the Step uniform for the next pass (a uniform is not a
// shader's to write). plan.tokens.most steps a submission, their ids read back once (T.151: the wait of a submission,
// 3 to 8.6 ms on the owner's Android, T.134, is paid once for them), with the keys and values of their positions, which
// go into forward.ts's cache as the CPU would have written them: the CPU may take the next step, or the next
// generation, at any position. The keys and values the GPU does not hold (positions the CPU computed since) go up from
// forward.ts's cache first: float16 there as here (T.110, T.147).
//
// The form of a layer, from those this device can make: T.150's fusedMatVec (llama.cpp's mul_mat_vec, with the norm on
// its read and RoPE, the residual's add or SwiGLU on its write) with the workgroup's reduction, the same with
// subgroupAdd where there are subgroups, and T.175's fusedDp4aMatVec (ORT's DP4A for small M, the vector quantized
// before each matrix, NORM_QUANTIZE where a norm is) where there is the packed int8 dot. Each is checked against
// JavaScript on the model's own weights (checkTokens), and the right ones are timed, a run of plan.tokens.most steps
// each in turn: the fastest is taken (T.150's and T.175's tables left which is fastest to the device: the owner's
// Android read a layer at 18.5% of its buffer's reads with mul_mat_vec and a matrix at 96.8% with DP4A). The attention
// (T.224, chooseTokenAttention) is llama.cpp's decode form, flash_attn_vec, split over the positions and reduced, or the
// prompt's tiles, whichever is right and faster here, on this pass's q. Where it came from: the run of T.151 (src/runtime/benchmark/gpu/generate.js's generate()), whose
// form is WebLLM's decode loop without its sync of every token (web-llm, src/llm_chat.ts; no line taken) and llama.cpp's
// WebGPU graph of a token, one command encoder for all of it (ggml-webgpu.cpp, commit 2145525a, MIT; no line taken).
// T.226: Qwen2's and Qwen3's steps too. What they have between the matrix of q, k and v and RoPE (T.153: a bias of each,
// a norm of every head of q and of k) is the prompt's dispatches, ADD and HEAD_NORM, as the CPU orders them, on the
// three as their matrix wrote them, and then shaders.ts's TOKEN_ROPE (the fused write's own lines, as a dispatch):
// two dispatches a layer more with the biases, three with the norms of the heads, none for a model without either.
// (T.175's fused DP4A with the norms apart as well, RMSNORM and QUANTIZE where NORM_QUANTIZE is one: the owner's
// Android ran a layer so in 3.36 ms against 3.67 fused, the fastest of its table, 2026-09-27; T.175's condition to
// reverse NORM_QUANTIZE, here chosen on the device)
const TOKEN_FORMS: Candidate[] = [{ name: "llama.cpp, fused (T150)", dp4a: false, subgroups: false },
  { name: "llama.cpp, fused (T150), subgroups", dp4a: false, subgroups: true },
  { name: "DP4A, fused (T175)", dp4a: true, subgroups: false },
  { name: "DP4A, fused (T175), the norms apart", dp4a: true, subgroups: false, normApart: true },
  // T.232: the same two on ternary weights (fusedDp4aMatVec's weights: "ternary"), which a model of ternary weights alone takes,
  // and takes no other
  { name: "DP4A, fused (T175), ternary", dp4a: true, subgroups: false, ternary: true },
  { name: "DP4A, fused (T175), ternary, the norms apart", dp4a: true, subgroups: false, ternary: true, normApart: true }];
function tokenCandidates(m: Model) {
  const features = navigator.gpu.wgslLanguageFeatures;
  const subgroups = m.device.features.has("subgroups") && Boolean(features?.has("subgroup_id"));
  const packed = Boolean(features?.has("packed_4x8_integer_dot_product"));
  // (T.226: NORM_QUANTIZE is RMSNorm's: a model with LayerNorm has its norms apart, and the form with them fused is
  // the same as the one with them apart)
  // (T.271.1: a rotated basis turns every matrix's input as it quantizes it, ROTATE_QUANTIZE: the forms of DP4A with the
  // norms apart alone, where no norm is on a matrix's read)
  // (T.547.2.4: a model of Q4_K blocks besides int8, the forms of DP4A alone: its blocks are read as packed int8 there)
  return TOKEN_FORMS.filter((form) => Boolean(form.ternary) === m.ternary && (!form.subgroups || subgroups) && (!form.dp4a || packed) && (form.dp4a || m.own !== "Q4_K") &&
    (!m.plan.layerNorm || !form.dp4a || form.normApart) && (!m.plan.rotated || (form.dp4a && form.normApart)));
}
// T.226: what of the model's form shapes a token's dispatches: apart, q, k and v are written as they are for what comes
// before RoPE (the biases, the norms of the heads); gated, the FFN has a gate (SwiGLU on the write of gate and up; else
// one matrix written as it is, then its bias and GELU); floatHead, the classifier multiplies floats on a DP4A form too
// (T.92's outlier channels: a few of the final norm's weights are 12 to 17 times the others (GPT-2), and a group of 32
// quantized to 8 bits with one of them loses the other 31; the CPU multiplies their columns apart, and the GPU has
// T.150's matrix of floats, which needs no such thing)
// (T.232: a ternary classifier is no matrix of floats: its outlier channels are taken apart as the CPU takes them,
// shaders.ts's TAKE_OUTLIERS and TERNARY_COLUMNS)
// (T.261.4.2: geglu, the gate's activation is GELU (a Gemma 3's); posts, the norms behind the attention and the FFN, so
// that o and down write their rows as they are, which RMSNORM_ADD norms into the stream)
const floatHead = (m: Model) => Boolean(m.plan.tokens!.outliers) && !m.ternary;
// (T.568.2: what the kinds of the model's layers write besides (gpu/kinds.ts's writes()): mix, a matrix of the normed
// stream written as it is (T.545.2: an LFM2's convolution layers' matrix in; T.271.3: a Qwen3.5's gate of its attending
// layers' output); write, a matrix of a vector of the kind's own written as it is (a Qwen3.5's linear-attention layers')
// T.226: apart, the biases or the norms of the heads that come between q, k and v and RoPE)
const tokenShape = (m: Model) => {
  const writes = new Set(kindsOf(m.plan).flatMap((kind) => GPU_KINDS[kind].writes?.(m.plan) ?? []));
  // (T.547.2.4: split, v apart from q and k (of two packings, weights.ts's tokensLayout), written as they are, as apart's;
  // blocks, matrices of Q4_K's among the int8 ones, whose shaders are the same forms' of Q4_K)
  const split = Boolean(m.weights.together?.vAt);
  return { apart: Boolean(m.plan.vectors.bq || m.plan.vectors.qNorm) || split, gated: Boolean(m.weights.matrices.w3), geglu: m.plan.activation === "gelu", layerNorm: m.plan.layerNorm,
    floatHead: floatHead(m), posts: Boolean(m.plan.vectors.attentionPost || m.plan.vectors.ffnPost), mix: writes.has("mix"), write: writes.has("write"),
    blocks: m.own === "Q4_K" };
};
// a form's WGSL, [key, code] each (the pipelines every form shares are compiled apart: tokenBuffers); the same code
// under two keys is compiled once (chooseTokens). The norm is on the read of T.150's matrices where it is RMSNorm
// (LayerNorm takes the mean out first, which no sum of the matrix's rows gives: a dispatch of its own before them)
// (T.547.2.4: blocks, and each of the matrices' keys again with ":Q4_K" after it, fusedDp4aMatVec's text for Q4_K's blocks:
// a matrix of them dispatches its key's counterpart; the classifier is no such matrix, tokensUnfit())
const tokenCodes = ({ dp4a, subgroups, normApart, ternary }: Candidate, { apart, gated, geglu, layerNorm, floatHead, posts, mix, write, blocks }: ReturnType<typeof tokenShape>): [string, string][] => {
  const floats = (output: FusedOutput) => ENGINE.fusedMatVec({ input: layerNorm ? "plain" : "norm", output, subgroups });
  const matrix = dp4a ? (output: FusedOutput) => ENGINE.fusedDp4aMatVec({ output, weights: ternary ? "ternary" : "int8" }) : floats;
  // (a matrix whose input is no norm of the stream: the attention's output, the activation's)
  const plain = (output: FusedOutput) => (dp4a ? matrix(output) : ENGINE.fusedMatVec({ input: "plain", output, subgroups }));
  const outputs: [string, FusedOutput][] = [["qkv", apart ? "write" : "rope"], ["add", "add"], ["glu", !gated ? "write" : geglu ? "geglu" : "swiglu"],
    ...(posts || write ? [["write", "write"]] satisfies [string, FusedOutput][] : []), ...(mix ? [["mix", "write"]] satisfies [string, FusedOutput][] : [])];
  return [...(dp4a && !normApart ? [["normQuantize", ENGINE.NORM_QUANTIZE]] satisfies [string, string][] : []), ["qkv", matrix(apart ? "write" : "rope")],
    ["add", plain("add")], ["glu", matrix(!gated ? "write" : geglu ? "geglu" : "swiglu")],
    ["classifier", (floatHead ? floats : matrix)("write")], ...(posts || write ? [["write", plain("write")]] satisfies [string, string][] : []),
    ...(mix ? [["mix", matrix("write")]] satisfies [string, string][] : []),
    ...(blocks && dp4a ? outputs.map(([key, output]): [string, string] => [`${key}:Q4_K`, ENGINE.fusedDp4aMatVec({ output, weights: "Q4_K" })]) : [])];
};

// a bind group of the bindings given ([binding, a buffer or a range of one] each: the fused shaders skip binding 4
// where the norm is not on their read)
const bindAt = (m: Model, pipeline: GPUComputePipeline, entries: Entry[]) => m.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: entries.map(([binding, buffer]) => ({ binding, resource: buffer.buffer ? buffer : { buffer } })) });

// The buffers and uniforms of a token, and the pipelines every form shares (EMBED, SAMPLE; QUANTIZE and the attention
// are the prompt's). A token's vectors: the stream h, q, the attention's output, SwiGLU's; the logits and SAMPLE's two
// scratch arrays of the vocabulary; the state, the ids, the random numbers, the Step and the settings; the quantized
// vector (DP4A: one serves every quantization of a token, each read before the next is made: T.175's (l)); what a
// submission reads back (the ids, the state, and the keys and values of its positions, [keys, values][layer][step])
// T.378.4: what this makes is the first part of the model's record of a token, m.tokens, which tokenforms.ts's
// chooseTokens() sets (with the attention and the form chosen); the functions below read it
async function tokenBuffers(m: Model): Promise<TokenBuffers> {
  // (T.271.4: the state of the layers that keep one comes back where the CPU holds it too: not on the GPU alone)
  const { plan } = m, most = plan.tokens!.most, vocab = plan.tokens!.classifier.rows, attending = attendingOf(plan).length, stateBytes = plan.direct ? 0 : m.block.stateBytes;
  // (T.568.2: and what a kind's layers quantize besides: T.271.3, the read of a Qwen3.5's linear-attention layer, which z's
  // gate leaves for its matrix out)
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, present = kindsOf(plan);
  const widest = Math.max(plan.dim, qDim, plan.hidden, ...present.map((kind) => GPU_KINDS[kind].widest?.(plan) ?? 0));
  // (one at a time: each in an error scope of its own)
  // (T.232: from a table of ternary weights, EMBED_TERNARY)
  const embedCode = tablesOf(plan).embedding.packing === "ternary" ? ENGINE.EMBED_TERNARY : ENGINE.EMBED;
  const embed = await within(validated(m, () => pipelineOf(m, embedCode)), "compiling the embedding's row");
  const sample = await within(validated(m, () => pipelineOf(m, ENGINE.SAMPLE)), "compiling the sampling");
  const out = STORAGE | COPY_SRC;
  // (u is null until the uniforms are made, below: what this returns has them)
  const g: Omit<TokenBuffers, "u"> & { u: Uniforms | null } = { embed, sample, most, vocab, h: buffer(m, plan.dim * 4, out | COPY_DST), q: buffer(m, qDim * 4), att: buffer(m, qDim * 4),
    gate: buffer(m, plan.hidden * 4), logits: buffer(m, vocab * 4, out), probs: buffer(m, vocab * 4), order: buffer(m, vocab * 4),
    state: buffer(m, STATE_BYTES, out | COPY_DST), chosen: buffer(m, most * 4, out), randoms: buffer(m, most * 4, STORAGE | COPY_DST),
    step: buffer(m, 16, UNIFORM | COPY_DST), settings: buffer(m, SAMPLING_BYTES, UNIFORM | COPY_DST),
    xq: buffer(m, widest), xs: buffer(m, (widest / GROUP) * 4), xb: buffer(m, plan.dim * 4),
    // (T.545.2: the attending layers' keys and values, and the state of the layers that keep one after the last step)
    readback: buffer(m, most * 4 + STATE_BYTES + 2 * attending * most * kvDim * 2 + stateBytes, MAP_READ | COPY_DST),
    // (below: the uniforms, and what only some models have)
    u: null, take: null, columns: null, picked: null, kinds: {} };
  // fusedMatVec's Params: rows, words, perRow, second, eps, normAt, qRows, kvRows, headSize, turned
  const params = (rows: number, n: number, second = 0, normAt = 0, turned = plan.turned) => {
    const bytes = new ArrayBuffer(48);
    new Uint32Array(bytes).set([rows, n / 4, n / GROUP, second, 0, normAt, qDim, kvDim, plan.headSize, turned, 0, 0]);
    new Float32Array(bytes, 16, 1)[0] = plan.eps;
    return uniform(m, bytes);
  };
  // RMSNORM's, HEAD_NORM's and NORM_QUANTIZE's Norm with the model's epsilon (device.ts's normShape)
  const norm = (at: number, size = plan.dim, first = 0) => normShape(m, size, at, plan.eps, first);
  const layers = [...Array(plan.layers)].map((_, l) => l * plan.dim);
  g.u = { embed: m.weights.tables!.embedding.map((piece) => uniform(m, new Uint32Array([plan.dim, piece.first, piece.rows, 0]))),
    down: params(plan.dim, plan.hidden), gateUp: layers.map((at) => params(plan.hidden, plan.dim, plan.hidden, at)),
    norm: layers.map((at) => norm(at)), final: norm(0), classifier: m.weights.tables!.classifier.map((piece) => params(piece.rows, plan.dim)),
    // QUANTIZE's (n, xStride): SwiGLU's, the normed stream's
    quantizeGate: uniform(m, new Uint32Array([plan.hidden, plan.hidden, 0, 0])), quantizeNormed: uniform(m, new Uint32Array([plan.dim, plan.dim, 0, 0])) };
  // T.568.2: what each kind of the model's layers makes of a token (gpu/kinds.ts: T.226 an attending layer's q, k and v
  // and what comes before RoPE; T.545.2 an LFM2's convolution layers'; T.271.3 a Qwen3.5's linear-attention layers')
  for (const kind of present) g.kinds[kind] = await m.block.kinds[kind]!.tokens({ params, norm, layers });
  // T.226, GPT-2's and GPT-NeoX's (T.154's dispatches of a prompt's block, on a token's vectors): ADD's shapes of the
  // biases after o, w1 and w2 (a layer's from its place in the vector) and of GPT-2's positions (a row a position), and
  // GELU's size
  const { bo, b1, b2 } = plan.vectors;
  if (bo) g.u.biases = layers.map((_, l) => Object.fromEntries(([["bo", bo], ["b1", b1!], ["b2", b2!]] satisfies [string, { size: number }][]).map(([name, { size }]): [string, GPUBuffer] =>
    [name, uniform(m, new Uint32Array([size, l * size, 0, 0]))])));
  if (m.weights.positions) g.u.positions = uniform(m, new Uint32Array([plan.dim, 0, plan.dim, 0]));
  // T.232: a ternary classifier's outlier channels (plan.tokens.channels, T.92): the two dispatches around its matrix
  // (shaders.ts's TAKE_OUTLIERS and TERNARY_COLUMNS), the values taken, and the Outliers of every piece of the table
  const channels = m.ternary ? plan.tokens!.channels ?? [] : [];
  if (channels.length) {
    g.take = await within(validated(m, () => pipelineOf(m, ENGINE.TAKE_OUTLIERS)), "compiling the outlier channels' taking");
    g.columns = await within(validated(m, () => pipelineOf(m, ENGINE.TERNARY_COLUMNS)), "compiling the outlier channels' columns");
    g.picked = buffer(m, OUTLIERS_MOST * 4);
    g.u.take = uniform(m, outliersOf(channels));
    g.u.columns = m.weights.tables!.classifier.map((piece) => uniform(m, outliersOf(channels, piece.rows, plan.dim)));
  }
  g.u.hidden = uniform(m, new Uint32Array([plan.hidden, 0, 0, 0]));
  if (plan.rotated) g.u.rotate = Object.fromEntries(plan.rotated.signs.map(({ width }) => [width, uniform(m, new Uint32Array([width, width, plan.rotated!.block, 0]))]));
  return g as TokenBuffers;
}

// the workgroups of rows by the fused shaders' rows a workgroup, over x and then y (they number them so)
function spread(m: Model, rows: number, perGroup: number): [number, number] {
  const groups = Math.ceil(rows / perGroup), across = Math.min(groups, m.device.limits.maxComputeWorkgroupsPerDimension);
  return [across, Math.ceil(groups / across)];
}
// The dispatches of one step in a form: EMBED, the layers (from, to: those of a check), the head (the final norm and
// the classifier) and SAMPLE, [pipeline, bind group, x, y] each. positions: how many the attention reads at the most
// (T.224: flash_attn_vec's parts a head go by it)
function tokenPass(m: Model, form: TokenForm, { from = 0, to = m.plan.layers, head = true, embed = true, positions = m.block.cache!.capacity }: { from?: number; to?: number; head?: boolean;
  embed?: boolean; positions?: number } = {}) {
  const { plan, tokens: g } = m, W = m.weights, b = m.block, V = W.vectors!, P = form.pipes;
  const rows = form.dp4a ? ORT_DP4A_MATVEC_ROWS : MUL_MAT_VEC_ROWS;
  // (T.547.2.3: no scales where a matrix's blocks hold them, Q4_K's)
  // (T.547.2.4: a matrix of no scales apart, Q4_K's blocks, by the pipeline's counterpart for them)
  const matrix = (given: GPUComputePipeline, [w, s]: [Bound, Bound | null], input: Entry[], params: GPUBuffer, count: number, output: Entry[]): Pass => {
    const pipeline = s ? given : form.blocks!.get(given)!;
    return [pipeline, bindAt(m, pipeline, [[0, w], ...(s ? [[1, s] satisfies Entry] : []), ...input, [3, params], ...output]), ...spread(m, count, rows)];
  };
  // (T.271.1: in a rotated basis R of the vector, quantized: ROTATE_QUANTIZE with the signs of its width)
  const quantize = (x: GPUBuffer, params: GPUBuffer, n: number): Pass => (b.rotate
    ? [b.rotate.quantize, bindAt(m, b.rotate.quantize, [[0, x], [1, g.xq], [2, g.xs], [3, W.signs![n]], [4, g.u.rotate![n]]]), n / plan.rotated!.block, 1]
    : [b.quantize, bindAt(m, b.quantize, [[0, x], [1, g.xq], [2, g.xs], [3, params], [4, g.step]]), Math.ceil(n / GROUP / 64), 1]);
  // What a matrix reads: { norm, quantize, input }, the dispatches before it (the stream's norm where it is one of its
  // own, the quantizing of DP4A's vector: T.175) and its bindings of the vector.
  // The stream normed, for a matrix of floats (T.150's; T.226: the classifier of a model with outlier channels on DP4A
  // too) or of DP4A: RMSNorm on the read of T.150's matrices, or with DP4A's quantizer (NORM_QUANTIZE); else a dispatch
  // of its own into xb (the prompt's RMSNORM, T.175's form with the norms apart; T.226: the prompt's LAYER_NORM with its
  // bias, T.154), which the matrix reads as it is or quantized
  // (T.232, taken: what changes the normed stream before it is quantized, a ternary classifier's outlier channels
  // taken out of it: the norm is then a dispatch of its own whatever the form)
  const normed = (weights: GPUBuffer, bias: GPUBuffer | null | undefined, params: GPUBuffer, floats = !form.dp4a, taken: Pass[] | null = null): Feed => {
    const apart = (): Pass[] => [[b.norm, bind(m, b.norm, plan.layerNorm ? [g.h, weights, bias!, g.xb, params, g.step] : [g.h, weights, g.xb, params, g.step]), 1, 1], ...(taken ?? [])];
    if (floats) return plan.layerNorm ? { norm: apart(), quantize: [], input: [[2, g.xb]] } : { norm: [], quantize: [], input: [[2, g.h], [4, weights]] };
    if (!form.normApart && !plan.layerNorm && !taken) {
      return { norm: [[P.normQuantize, bindAt(m, P.normQuantize, [[0, g.h], [1, weights], [2, g.xq], [3, g.xs], [4, params], [5, g.step]]), 1, 1]],
        quantize: [], input: [[2, g.xq], [4, g.xs]] };
    }
    return { norm: apart(), quantize: [quantize(g.xb, g.u.quantizeNormed, plan.dim)], input: [[2, g.xq], [4, g.xs]] };
  };
  // a vector as it is (the attention's output, the activation's)
  const plain = (x: GPUBuffer, params: GPUBuffer, n: number): Omit<Feed, "norm"> => (form.dp4a ? { quantize: [quantize(x, params, n)], input: [[2, g.xq], [4, g.xs]] } : { quantize: [], input: [[2, x]] });
  // T.226: a bias added after a matrix (ADD: Qwen2's of q, k and v as one vector; T.154's after o, w1 and w2)
  const added = (to: GPUBuffer, bias: GPUBuffer, shape: GPUBuffer, n: number): Pass => [b.add, bind(m, b.add, [to, bias, shape, g.step]), Math.ceil(n / 64), 1];
  const biased = (to: GPUBuffer, name: "bo" | "b1" | "b2", l: number) => (V[name] ? [added(to, V[name], g.u.biases![l][name], plan.vectors[name]!.size)] : []);
  // (T.209: a dispatch a piece of the table, each writing the row where the token is in its rows)
  const list: Pass[] = !embed ? [] : W.tables!.embedding.map((piece, i): Pass =>
    [g.embed, bindAt(m, g.embed, [[0, piece.values], [1, piece.scales], [2, g.state], [3, g.h], [4, g.u.embed[i]]]), 1, 1]);
  // T.226: GPT-2's learned positions: the row of the Step's position added to the embedding's (the CPU's embed())
  if (embed && W.positions) list.push(added(g.h, W.positions, g.u.positions!, plan.dim));
  // T.271.1: the embedding of a rotated basis holds R^-1 of its rows: turned back (UNROTATE)
  if (embed && b.rotate) list.push([b.rotate.unrotate, bindAt(m, b.rotate.unrotate, [[0, g.h], [1, W.signs![plan.dim]], [2, g.u.rotate![plan.dim]]]), plan.dim / plan.rotated!.block, 1]);
  const gated = Boolean(W.matrices.w3);
  for (let l = from; l < to; l++) {
    // (T.545.2: a matrix of the layers of a kind, an LFM2's, at the layer's place among them: an attending layer's keys
    // and values too; every layer's place where there is one kind)
    const place = plan.places[l], [down] = W.matrices.w2.pieces;
    const attention = normed(V.attention, V.attentionBias, g.u.norm[l]), ffn = normed(V.ffn, V.ffnBias, g.u.norm[l]);
    const activated = plain(g.gate, g.u.quantizeGate, plan.hidden);
    // T.261.4.2: what joins the stream after a matrix: added on its write, or (a Gemma 3's norm behind it) written as it
    // is into xb, which nothing reads by then, and normed into the stream there (RMSNORM_ADD, as a prompt's block)
    const joining = (post: GPUBuffer | undefined, [w, s]: [Bound, Bound | null], input: Entry[], params: GPUBuffer): Pass[] => (post
      ? [matrix(P.write, [w, s], input, params, plan.dim, [[5, g.xb]]), [b.normAdd!, bind(m, b.normAdd!, [g.xb, post, g.h, g.u.norm[l], g.step]), 1, 1]]
      : [matrix(P.add, [w, s], input, params, plan.dim, [[5, g.h]])]);
    // T.568.2: the layer's kind's dispatches (gpu/kinds.ts), from the stream as it is (the kind norms it: attention.norm and
    // quantize, or a dispatch of its own) to what joins it
    list.push(...g.kinds[plan.kinds[l]]!.step({ l, place, form, matrix, plain, joining, normed: attention, ffnNorm: ffn.norm, biased, positions,
      bindAt: (pipeline, entries) => bindAt(m, pipeline, entries) }));
    list.push(
      ...(plan.parallel ? [] : ffn.norm), ...ffn.quantize,
      // gate and up as one matrix with SwiGLU on its write; or (T.154: no gate) w1, its bias and GELU
      ...(gated ? [matrix(P.glu, W.joined[l].gateUp, ffn.input, g.u.gateUp[l], plan.hidden, [[5, g.gate]])]
        : [matrix(P.glu, W.matrices.w1.pieces[0].layers[l], ffn.input, g.u.gateUp[l], plan.hidden, [[5, g.gate]]), ...biased(g.gate, "b1", l),
          [b.activation, bind(m, b.activation, [g.gate, g.u.hidden!, g.step]), Math.ceil(plan.hidden / 64), 1]] satisfies Pass[]),
      ...activated.quantize, ...joining(V.ffnPost, down.layers[l], activated.input, g.u.down), ...biased(g.h, "b2", l));
  }
  if (head) {
    // T.232: a ternary classifier's outlier channels go out of the normed stream before it is quantized (TAKE_OUTLIERS),
    // and their columns are added to the piece's logits after its matrix (TERNARY_COLUMNS), as the CPU has them
    const take: Pass[] | null = g.take ? [[g.take, bind(m, g.take, [g.xb, g.picked!, g.u.take!]), 1, 1]] : null;
    const final = normed(W.finalNorm!, W.finalBias, g.u.final, !form.dp4a || floatHead(m), take);
    const logitsOf = (piece: TablePiece) => ({ buffer: g.logits, offset: piece.first * 4, size: piece.rows * 4 });
    list.push(...final.norm, ...final.quantize,
      // (T.209: a piece at a time into its range of the logits: its first row is where the device binds, piecesOf)
      ...W.tables!.classifier.flatMap((piece, i): Pass[] => [matrix(P.classifier, [piece.values, piece.scales], final.input, g.u.classifier[i], piece.rows, [[5, logitsOf(piece)]]),
        ...(take ? [[g.columns!, bind(m, g.columns!, [piece.values, piece.scales, g.picked!, logitsOf(piece), g.u.columns![i]]), ...spread(m, piece.rows, 256)]] satisfies Pass[] : [])]),
      [g.sample, bindAt(m, g.sample, [[0, g.logits], [1, g.probs], [2, g.order], [3, g.state], [4, g.chosen], [5, g.randoms], [6, g.settings]]), 1, 1]);
  }
  return list;
}
// a form's whole step for a run that reads up to positions, made again when the cache has grown (its buffers are others
// then); T.224: one a count of the attention's parts a head (flash_attn_vec's nwg goes by the positions)
function tokenStep(m: Model, form: TokenForm, positions: number) {
  const a = m.tokens.attention, { cache } = m.block, parts = a.tiles ? 0 : flashVecSplits(a.shape, positions);
  if (form.steps?.cache !== cache) form.steps = { cache, lists: new Map() };
  if (!form.steps.lists.has(parts)) form.steps.lists.set(parts, tokenPass(m, form, { positions }));
  return form.steps.lists.get(parts)!;
}
// count steps of the dispatches from the state given (with the settings and a random number a step), in one
// submission, and read back: the ids and the state's words, and (keep) the keys and values of the positions pos to
// pos + count - 1. extra(encoder): what a check copies out besides. Returns { ids, sampled, stopped, notFinite (T.219:
// the sampler refused the step after the sampled ones: its logits were not finite), kv: [keys, values][layer], the
// positions' rows of float16 as bytes }
// T.545.2: kv by the attending layers' places; state (keep), the state of the layers that keep one after the last step
// (T.271.2: of the last step alone: steps that stopped before it leave no state of their position, and the CPU goes on
// from position 0 then, forward/gpuside.ts)
async function runTokens(m: Model, dispatches: Pass[], { count, pos, state, settings, randoms, keep = false, extra }: Run) {
  const { device, tokens: g } = m, { cache, states } = m.block, kvRow = m.plan.kvHeads * m.plan.headSize * 2, idsBytes = g.most * 4;
  const layers = cache!.keys.length, stateBytes = keep && !m.plan.direct ? m.block.stateBytes : 0, stateAt = idsBytes + STATE_BYTES + 2 * layers * g.most * kvRow;
  if (pos + count > cache!.capacity) throw new Error(`positions to ${pos + count} are past the GPU's cache`);
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
  if (stateBytes) {
    let at = stateAt;
    for (const { buffer: from, bytes } of states) {
      encoder.copyBufferToBuffer(from, 0, g.readback, at, bytes);
      at += bytes;
    }
  }
  encoder.copyBufferToBuffer(g.chosen, 0, g.readback, 0, count * 4);
  encoder.copyBufferToBuffer(g.state, 0, g.readback, idsBytes, STATE_BYTES);
  const kvAt = idsBytes + STATE_BYTES;
  if (keep) {
    for (let l = 0; l < layers; l++) {
      encoder.copyBufferToBuffer(cache!.keys[l], pos * kvRow, g.readback, kvAt + l * g.most * kvRow, count * kvRow);
      encoder.copyBufferToBuffer(cache!.values[l], pos * kvRow, g.readback, kvAt + (layers + l) * g.most * kvRow, count * kvRow);
    }
  }
  extra?.(encoder);
  device.queue.submit([encoder.finish()]);
  const invalid = await device.popErrorScope(), full = await device.popErrorScope();
  if (invalid || full) throw new Error(`the GPU refused a token (${(invalid ?? full)!.message})`);
  await g.readback.mapAsync(MAP_READ);
  try {
    const words = new Uint32Array(g.readback.getMappedRange(0, idsBytes + STATE_BYTES).slice(0));
    const after = words.subarray(g.most);
    const out: Ran = { ids: words.slice(0, count), sampled: after[5], stopped: after[7], notFinite: after[STATE_NOT_FINITE] };
    if (keep) {
      const bytes = new Uint8Array(g.readback.getMappedRange(kvAt, 2 * layers * g.most * kvRow + stateBytes).slice(0));
      out.kv = [0, 1].map((side) => [...Array(layers)].map((_, l) => {
        const at = (side * layers + l) * g.most * kvRow;
        return bytes.subarray(at, at + count * kvRow);
      }));
      if (stateBytes) out.state = bytes.subarray(stateAt - kvAt, stateAt - kvAt + stateBytes);
    }
    return out;
  } finally {
    g.readback.unmap();
  }
}

export { tokenCandidates, floatHead, tokenShape, tokenCodes, tokenBuffers, tokenPass, tokenStep, runTokens };
export type { TokenBuffers, TokenForm, Entry, Feed };
