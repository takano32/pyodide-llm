// A block of a prompt: the small steps' pipelines and a block's buffers, every layer's bind groups, the GPU's own keys
// and values, the block itself as one submission with its keys and values read back, and what a whole block takes (T148).
// (T352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T367.3)

import { STORAGE, COPY_DST, COPY_SRC, MAP_READ, UNIFORM, stopping, held, within, buffer, uniform, copyIn, narrowIn, validated, pipelineOf, bind,
  dispatch } from "./device.ts";
import { tablesOf } from "./weights.ts";
import { productGroups, multiply } from "./forms.ts";
import { GROUP } from "../shaders/common.ts";
import { FLASH_Q_TILE } from "../shaders/attention.ts";
import { ENGINE } from "../shaders/engine.ts";
import type { Model } from "./device.ts";
import type { Tiled } from "./forms.ts";
import type { Cache as Sent, Vectors } from "../contracts/gpu/protocol.ts";

// T368.6: this module's record, m.block, as prepare() makes it (the layers' bind groups and the keys and values come
// later, bindLayers() and grow(): null until then, and read with a ! after).
// an input's quantization (QUANTIZE's bind group and workgroups), a bias added (ADD's), a matrix's product (a bind
// group and its rows for each piece, forms.ts's productGroups())
type Quantizing = { group: GPUBindGroup; x: number };
type Added = { group: GPUBindGroup; x: number };
type Product = { rows: number; group: GPUBindGroup }[];
type HeadNorm = { group: GPUBindGroup; heads: number };
// T545: an LFM2's convolution layer's: the matrix in (from xb into the block's mixed), SHORT_CONV (into xb) and
// CONV_STATE, the matrix out
type Convolving = { into: Product; conv: GPUBindGroup; state: GPUBindGroup; out: Product };
// T271.2: a Qwen3.5's linear-attention layer's: its matrices of q, k and v (into mixed) and of z, its gates, its
// convolution (DELTA_CONV into conv) and its rows moved on (DELTA_CONV_STATE), the l2 norms of q and of k (HEAD_NORM),
// the delta rule (into out), the norm of a value head, z's gate (SWIGLU: into z), its matrix out
type Delta = { into: Product; z: Product; gates: GPUBindGroup; conv: GPUBindGroup; state: GPUBindGroup; qNorm: GPUBindGroup; kNorm: GPUBindGroup;
  rule: GPUBindGroup; norm: GPUBindGroup; gate: GPUBindGroup; out: Product };
// a layer's bind groups (a bias only where the model has it; up where the FFN has a gate). T545: a convolution layer's
// convolution; its attention's (q, k, v, its biases and norms of the heads, o) are empty
type Layer = {
  convolution: Convolving | null;
  // T271.2: a linear-attention layer's (its attention's are empty then), and a full-attention layer's gate of its
  // output (a matrix of xb into the block's gated, OUTPUT_GATE on the attention's output; null: none)
  delta: Delta | null;
  outputGate: { into: Product; gate: GPUBindGroup } | null;
  attentionNorm: GPUBindGroup;
  q: Product;
  k: Product;
  v: Product;
  biases: Added[];
  headNorms: HeadNorm[];
  o: Product;
  oBias: Added | undefined;
  // T261.4: a Gemma 3's norms behind the attention and the FFN, each added to the stream (null: none)
  attentionPost: GPUBindGroup | null;
  ffnNorm: GPUBindGroup;
  gate: Product;
  gateBias: Added | undefined;
  up: Product | null;
  down: Product;
  downBias: Added | undefined;
  ffnPost: GPUBindGroup | null;
};
// the GPU's own keys and values, a buffer a layer each (T545: an attending layer each, by its place), and the bind
// groups that read them
type Kept = { capacity: number; owned: GPUBuffer[]; keys: GPUBuffer[]; values: GPUBuffer[]; rope: GPUBindGroup[]; attention: GPUBindGroup[] };
type Block = {
  norm: GPUComputePipeline;
  headNorm: GPUComputePipeline;
  add: GPUComputePipeline;
  rope: GPUComputePipeline;
  activation: GPUComputePipeline;
  quantize: GPUComputePipeline;
  normAdd: GPUComputePipeline | null;
  x: GPUBuffer;
  xb: GPUBuffer;
  q: GPUBuffer;
  k: GPUBuffer;
  v: GPUBuffer;
  gate: GPUBuffer;
  up: GPUBuffer | null;
  ffnInput: GPUBuffer;
  xq: GPUBuffer;
  xs: GPUBuffer;
  angles: GPUBuffer;
  anglesLocal: GPUBuffer | null;
  step: GPUBuffer;
  readback: GPUBuffer;
  rows: Float32Array<ArrayBuffer>;
  turns: Float32Array<ArrayBuffer>;
  turnsLocal: Float32Array<ArrayBuffer> | null;
  ropeShape: GPUBuffer;
  ropeAlone: GPUBuffer | null;
  activationGroup: GPUBindGroup;
  quantizeXb: Quantizing;
  quantizeAttention: Quantizing;
  quantizeFfn: Quantizing;
  quantizeGate: Quantizing;
  embedRows: GPUComputePipeline | null;
  ids: GPUBuffer | null;
  embedGroups: GPUBindGroup[] | null;
  layers: Layer[] | null;
  cache: Kept | null;
  // T545: an LFM2's convolution layers' shaders, what the matrix in gives a block (3 dim a token), and the rows they go
  // on from (taps rows of dim a layer, place after place, as the CPU keeps them: bytes of them); null where there are none
  conv: { short: GPUComputePipeline; state: GPUComputePipeline; mixed: GPUBuffer; rows: GPUBuffer; bytes: number } | null;
  // T271.2: a Qwen3.5's linear-attention layers' shaders and buffers (what wqkv gives a block, what the convolution
  // gives, z, the gates, the delta rule's output; the states and the rows they go on from, place after place, as the CPU
  // keeps them; the l2 norms' weights), null where there are none; and the gate of its full-attention layers' output
  linear: { gates: GPUComputePipeline; conv: GPUComputePipeline; state: GPUComputePipeline; rule: GPUComputePipeline; swiglu: GPUComputePipeline;
    mixed: GPUBuffer; convolved: GPUBuffer; z: GPUBuffer; work: GPUBuffer; out: GPUBuffer; states: GPUBuffer; matricesBytes: number; rows: GPUBuffer; rowsBytes: number;
    units: GPUBuffer; quantizeZ: Quantizing } | null;
  outputGate: { pipeline: GPUComputePipeline; gated: GPUBuffer } | null;
  // the state of the layers that keep one, as the CPU stages it (T545: an LFM2's rows; T271.2: a Qwen3.5's states,
  // then its rows), and its bytes
  states: { buffer: GPUBuffer; bytes: number }[];
  stateBytes: number;
};
// T545: how many layers keep keys and values here (every one, but an LFM2's convolution layers), and which they are
const attendingOf = (plan: { kinds?: string[]; layers: number }) =>
  plan.kinds ? plan.kinds.flatMap((kind, l) => (kind === "attention" ? [l] : [])) : Array.from({ length: plan.layers }, (_, l) => l);

// the pipelines of a layer's small steps and the buffers of a block's activations
// T378.4: they are this module's record, m.block, made here once and set by this module alone (bindLayers() puts the
// layers' bind groups in it once the forms are chosen, grow() the keys and values); forms.ts and tokens.ts read its
// pipelines and buffers for their own bind groups
async function prepare(m: Model) {
  const { plan } = m, B = plan.batch, gated = Boolean(plan.matrices.w3);
  // (T271.2: and a Qwen3.5's linear-attention layers' read, which z's gate leaves for their matrix out)
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, read = plan.linear ? plan.linear.valueHeads * plan.linear.valueDim : 0;
  const widest = Math.max(plan.dim, qDim, plan.hidden, read);
  // T154: LayerNorm and GELU where the model has them (GPT-2, GPT-NeoX), else RMSNorm and SwiGLU (T261.4: or GEGLU, a
  // Gemma 3's gate); and the norm added to the stream behind the attention and the FFN where the model has those
  const posts = Boolean(plan.vectors.attentionPost || plan.vectors.ffnPost);
  const [norm, headNorm, add, rope, activation, quantize, normAdd] = await within(Promise.all([plan.layerNorm ? ENGINE.LAYER_NORM : ENGINE.RMSNORM,
    ENGINE.HEAD_NORM, ENGINE.ADD, ENGINE.ROPE, !gated ? ENGINE.GELU : plan.activation === "gelu" ? ENGINE.GEGLU : ENGINE.SWIGLU, ENGINE.QUANTIZE,
    ...(posts ? [ENGINE.RMSNORM_ADD] : [])].map((code) => pipelineOf(m, code))), "the small steps' shaders");
  // a block's tokens, each array dense: token t's row at t times its width
  const x = buffer(m, B * plan.dim * 4, STORAGE | COPY_DST);
  const xb = buffer(m, B * Math.max(plan.dim, qDim) * 4, STORAGE | COPY_DST);
  const q = buffer(m, B * qDim * 4);
  const k = buffer(m, B * kvDim * 4);
  const v = buffer(m, B * kvDim * 4);
  const gate = buffer(m, B * plan.hidden * 4, STORAGE | COPY_DST);
  const up = gated ? buffer(m, B * plan.hidden * 4) : null;
  // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, which xb cannot hold (the attention
  // writes there first)
  const ffnInput = plan.parallel ? buffer(m, B * plan.dim * 4) : xb;
  // the packed shaders' input: the activations of a matrix quantized, 8 bits a value and a float32 scale a group
  const xq = buffer(m, B * widest);
  const xs = buffer(m, B * (widest / GROUP) * 4);
  const angles = buffer(m, B * plan.headSize * 4, STORAGE | COPY_DST);
  // (T261.4: and those of the local layers' tables, where the model has them)
  const anglesLocal = plan.localCos ? buffer(m, B * plan.headSize * 4, STORAGE | COPY_DST) : null;
  const step = buffer(m, 16, UNIFORM | COPY_DST);
  // T545: of an LFM2's convolution layers
  let conv: Block["conv"] = null;
  if (plan.convolution) {
    const [short, state] = await within(Promise.all([ENGINE.SHORT_CONV, ENGINE.CONV_STATE].map((code) => pipelineOf(m, code))), "the convolution's shaders");
    const bytes = plan.kinds.filter((kind) => kind === "convolution").length * plan.convolution.taps * plan.dim * 4;
    conv = { short, state, mixed: buffer(m, B * 3 * plan.dim * 4), rows: buffer(m, bytes, STORAGE | COPY_DST | COPY_SRC), bytes };
  }
  // T271.2: of a Qwen3.5's linear-attention layers
  let linear: Block["linear"] = null;
  if (plan.linear) {
    const { keyHeads, valueHeads, keyDim, valueDim, taps } = plan.linear, places = plan.kinds.filter((kind) => kind === "linear").length;
    const keys = keyHeads * keyDim, mixed = 2 * keys + read;
    const [gates, convolving, moving, rule, swiglu] = await within(Promise.all([ENGINE.DELTA_GATES, ENGINE.DELTA_CONV, ENGINE.DELTA_CONV_STATE, ENGINE.DELTA_NET,
      ENGINE.SWIGLU].map((code) => pipelineOf(m, code))), "the linear attention's shaders");
    const z = buffer(m, B * read * 4);
    const units = buffer(m, 2 * keyDim * 4, STORAGE | COPY_DST);
    m.device.queue.writeBuffer(units, 0, Float32Array.from({ length: 2 * keyDim }, (_, i) => (i < keyDim ? 1 / keyDim : 1 / Math.sqrt(keyDim))));
    const matricesBytes = places * valueHeads * keyDim * valueDim * 4, rowsBytes = places * taps * mixed * 4;
    linear = { gates, conv: convolving, state: moving, rule, swiglu, mixed: buffer(m, B * mixed * 4), convolved: buffer(m, B * mixed * 4), z,
      work: buffer(m, B * 2 * valueHeads * 4), out: buffer(m, B * read * 4), states: buffer(m, matricesBytes, STORAGE | COPY_DST | COPY_SRC), matricesBytes,
      rows: buffer(m, rowsBytes, STORAGE | COPY_DST | COPY_SRC), rowsBytes, units,
      quantizeZ: { group: bind(m, quantize, [z, xq, xs, uniform(m, new Uint32Array([read, read, 0, 0])), step]), x: Math.ceil(read / GROUP / 64) } };
  }
  // T271.2: the gate of a Qwen3.5's full-attention layers' output
  const outputGate = plan.matrices.wg ? { pipeline: await within(pipelineOf(m, ENGINE.OUTPUT_GATE), "the output's gate"), gated: buffer(m, B * qDim * 4) } : null;
  const states = [...(conv ? [{ buffer: conv.rows, bytes: conv.bytes }] : []),
    ...(linear ? [{ buffer: linear.states, bytes: linear.matricesBytes }, { buffer: linear.rows, bytes: linear.rowsBytes }] : [])];
  const stateBytes = states.reduce((sum, { bytes }) => sum + bytes, 0);
  // (the attending layers' keys and values, then the state of the layers that keep one: T545; T271.4: none of it on the
  // GPU alone, whose state stays here, and whose keys and values come back for the tests alone)
  const readback = buffer(m, 2 * attendingOf(plan).length * B * kvDim * 2 + (plan.direct ? 0 : stateBytes), MAP_READ | COPY_DST);
  // what a request writes into them, from the shared memory
  const rows = new Float32Array(B * plan.dim);
  const turns = new Float32Array(B * plan.headSize);
  const turnsLocal = plan.localCos ? new Float32Array(B * plan.headSize) : null;
  // (T255: a layer RoPE leaves alone turns none of a head, as every layer of a GPT-2)
  const shapeOf = (turned: number) => uniform(m, new Uint32Array([plan.heads, plan.kvHeads, plan.headSize, turned]));
  const ropeShape = shapeOf(plan.turned);
  const ropeAlone = plan.unturned?.length ? shapeOf(0) : null;
  const hidden = uniform(m, new Uint32Array([plan.hidden, 0, 0, 0]));
  const activationGroup = bind(m, activation, gated ? [gate, up!, hidden, step] : [gate, hidden, step]);
  // QUANTIZE of the inputs the matrices read: xb (the norm's, dim wide; the attention's, heads × headSize wide, which
  // is not dim where the heads are of another size, T153) and gate (SwiGLU's)
  const quantizing = (from: GPUBuffer, n: number): Quantizing => ({ group: bind(m, quantize, [from, xq, xs, uniform(m, new Uint32Array([n, n, 0, 0])), step]),
    x: Math.ceil(n / GROUP / 64) });
  const quantizeXb = quantizing(xb, plan.dim);
  const quantizeAttention = quantizing(xb, qDim);
  const quantizeFfn = plan.parallel ? quantizing(ffnInput, plan.dim) : quantizeXb;
  const quantizeGate = quantizing(gate, plan.hidden);
  // T210: a model on the GPU alone embeds a block's tokens here (the CPU holds no embedding): EMBED_ROWS a piece of
  // the table (T209), from the block's ids into x
  let embedRows: GPUComputePipeline | null = null, ids: GPUBuffer | null = null, embedGroups: GPUBindGroup[] | null = null;
  if (plan.direct) {
    // (T232: from a table of ternary weights, EMBED_ROWS_TERNARY)
    const rowsCode = tablesOf(plan).embedding.ternary ? ENGINE.EMBED_ROWS_TERNARY : ENGINE.EMBED_ROWS;
    embedRows = await within(validated(m, () => pipelineOf(m, rowsCode)), "compiling the embedding's rows");
    ids = buffer(m, B * 4, STORAGE | COPY_DST);
    embedGroups = m.weights.tables!.embedding.map((piece) => bind(m, embedRows!,
      [piece.values, piece.scales, ids!, x, uniform(m, new Uint32Array([plan.dim, piece.first, piece.rows, 0]))]));
  }
  m.block = { norm, headNorm, add, rope, activation, quantize, normAdd: normAdd ?? null, x, xb, q, k, v, gate, up, ffnInput, xq, xs, angles, anglesLocal, step,
    readback, rows, turns, turnsLocal, ropeShape, ropeAlone, activationGroup, quantizeXb, quantizeAttention, quantizeFfn, quantizeGate, embedRows, ids, embedGroups,
    // (bindLayers() and grow())
    layers: null, cache: null, conv, linear, outputGate, states, stateBytes };
}

// every layer's bind groups but those of the cache, with the form chosen. T153: a bias (Qwen2's) added to q, k and
// v, or (Qwen3's) the norm of every head of q and of k, where the model has them. T154: LayerNorm with its bias, and a
// bias after o, w1 and w2 (GPT-2's, GPT-NeoX's)
function bindLayers(m: Model) {
  const { plan } = m, form = m.forms.form!, V = m.weights.vectors!, b = m.block;
  const layers: Layer[] = [];
  for (let l = 0; l < plan.layers; l++) {
    // T545: a matrix or a vector of the layers of a kind (an LFM2's attending or convolution layers': of) is read at the
    // layer's place among them, every layer's at the layer
    const place = plan.places?.[l] ?? l, convolving = plan.kinds?.[l] === "convolution" || plan.kinds?.[l] === "linear";
    const index = (of: string | undefined) => (of ? place : l);
    // (a piece each, T155)
    const product = (name: string, from: GPUBuffer, to: GPUBuffer, add = false) => productGroups(m, form, name, index(plan.matrices[name]?.of), from, to, add);
    const norm = new ArrayBuffer(16);
    new Uint32Array(norm, 0, 2).set([plan.dim, l * plan.dim]);
    new Float32Array(norm, 8, 1)[0] = plan.eps;
    const normShape = uniform(m, norm);
    const added = (to: GPUBuffer, name: keyof Vectors) => V[name] && { group: bind(m, b.add, [to, V[name], uniform(m, new Uint32Array([plan.vectors[name]!.size, index(plan.vectors[name]!.of) * plan.vectors[name]!.size, 0, 0])), b.step]),
      x: Math.ceil(plan.vectors[name]!.size / 64) };
    // T154: LayerNorm reads its bias from the same place in its own buffer
    const normed = (weight: string, bias: string, out: GPUBuffer) => bind(m, b.norm, plan.layerNorm ? [b.x, V[weight], V[bias], out, normShape, b.step]
      : [b.x, V[weight], out, normShape, b.step]);
    // T261.4: the norm of what joins the stream (from), added to it (a Gemma 3's: the output of o, and of w2, written
    // apart into gate and xb, which nothing reads by then)
    const post = (name: keyof Vectors, from: GPUBuffer) => (V[name] ? bind(m, b.normAdd!, [from, V[name], b.x, normShape, b.step]) : null);
    const headNorm = (of: GPUBuffer, name: string, heads: number) => {
      if (!V[name]) return null;
      const shape = new ArrayBuffer(16);
      new Uint32Array(shape, 0, 2).set([plan.headSize, index(plan.vectors[name as keyof Vectors]!.of) * plan.headSize]);
      new Float32Array(shape, 8, 1)[0] = plan.eps;
      return { group: bind(m, b.headNorm, [of, V[name], uniform(m, shape), b.step]), heads };
    };
    // T545: a convolution layer's matrix in, its two shaders on the layer's taps and rows (at its place), its matrix out
    let convolution: Convolving | null = null;
    if (plan.kinds?.[l] === "convolution") {
      const c = b.conv!, shape = uniform(m, new Uint32Array([plan.dim, plan.convolution!.taps, place * plan.convolution!.taps * plan.dim, 0]));
      convolution = { into: product("win", b.xb, c.mixed), conv: bind(m, c.short, [c.mixed, V.conv, c.rows, b.xb, shape, b.step]),
        state: bind(m, c.state, [c.mixed, c.rows, shape, b.step]), out: V.attentionPost ? product("wout", b.xb, b.gate) : product("wout", b.xb, b.x, true) };
    }
    // T271.2: a linear-attention layer's matrices, its gates, convolution, l2 norms, delta rule, norm, z's gate (at its
    // place among those layers: its rows, its state, its vectors) and its matrix out; and an attending layer's gate of
    // its output where the model has one (a Qwen3.5's)
    let delta: Delta | null = null;
    if (plan.kinds?.[l] === "linear") {
      delta = linearLayer(m, form, place);
    }
    const outputGate = !convolving && b.outputGate ? { into: product("wg", b.xb, b.outputGate.gated),
      gate: bind(m, b.outputGate.pipeline, [b.xb, b.outputGate.gated, uniform(m, new Uint32Array([plan.heads * plan.headSize, 0, 0, 0])), b.step]) } : null;
    layers.push({
      convolution, delta, outputGate,
      attentionNorm: normed("attention", "attentionBias", b.xb),
      q: convolving ? [] : product("wq", b.xb, b.q), k: convolving ? [] : product("wk", b.xb, b.k), v: convolving ? [] : product("wv", b.xb, b.v),
      biases: convolving ? [] : [added(b.q, "bq"), added(b.k, "bk"), added(b.v, "bv")].filter(Boolean) as Added[],
      headNorms: convolving ? [] : [headNorm(b.q, "qNorm", plan.heads), headNorm(b.k, "kNorm", plan.kvHeads)].filter(Boolean) as HeadNorm[],
      o: convolving ? [] : V.attentionPost ? product("wo", b.xb, b.gate) : product("wo", b.xb, b.x, true), oBias: added(b.x, "bo"), attentionPost: post("attentionPost", b.gate),
      ffnNorm: normed("ffn", "ffnBias", b.ffnInput),
      gate: product("w1", b.ffnInput, b.gate), gateBias: added(b.gate, "b1"),
      up: m.weights.matrices.w3 ? product("w3", b.ffnInput, b.up!) : null,
      down: V.ffnPost ? product("w2", b.gate, b.xb) : product("w2", b.gate, b.x, true), downBias: added(b.x, "b2"), ffnPost: post("ffnPost", b.xb),
    });
  }
  m.block.layers = layers;
}
// T271.2: the bind groups of the linear-attention layer at place among them (a block's: B tokens' room in its buffers)
function linearLayer(m: Model, form: Tiled, place: number): Delta {
  const { plan } = m, V = m.weights.vectors!, b = m.block, L = b.linear!, B = plan.batch;
  const { keyHeads, valueHeads, keyDim, valueDim, taps } = plan.linear!, keys = keyHeads * keyDim, read = valueHeads * valueDim, mixed = 2 * keys + read;
  const product = (name: string, from: GPUBuffer, to: GPUBuffer, add = false) => productGroups(m, form, name, place, from, to, add);
  // HEAD_NORM's Norm: size, at (of the weights), eps, first (the rows before its first)
  const headNorm = (size: number, at: number, eps: number, first: number) => {
    const shape = new ArrayBuffer(16);
    new Uint32Array(shape).set([size, at, 0, first]);
    new Float32Array(shape, 8, 1)[0] = eps;
    return uniform(m, shape);
  };
  const conv = uniform(m, new Uint32Array([mixed, taps, place * taps * mixed, keys, B, 0, 0, 0]));
  return { into: product("wqkv", b.xb, L.mixed), z: product("wz", b.xb, L.z),
    gates: bind(m, L.gates, [b.xb, V.wb, V.wa, V.dtBias, V.decay, L.work, uniform(m, new Uint32Array([plan.dim, valueHeads, place * valueHeads, 0])), b.step]),
    conv: bind(m, L.conv, [L.mixed, V.conv, L.rows, L.convolved, conv, b.step]), state: bind(m, L.state, [L.mixed, L.rows, conv, b.step]),
    qNorm: bind(m, b.headNorm, [L.convolved, L.units, headNorm(keyDim, 0, 1e-6 / keyDim, 0), b.step]),
    kNorm: bind(m, b.headNorm, [L.convolved, L.units, headNorm(keyDim, keyDim, 1e-6 / keyDim, B * keyHeads), b.step]),
    rule: bind(m, L.rule, [L.convolved, L.work, L.states, L.out, uniform(m, new Uint32Array([keyHeads, valueHeads, keyDim, valueDim, B, place * valueHeads * keyDim * valueDim, 0, 0])), b.step]),
    norm: bind(m, b.headNorm, [L.out, V.deltaNorm, headNorm(valueDim, place * valueDim, plan.eps, 0), b.step]),
    gate: bind(m, L.swiglu, [L.z, L.out, uniform(m, new Uint32Array([read, 0, 0, 0])), b.step]),
    out: V.attentionPost ? product("wout", L.z, b.gate) : product("wout", L.z, b.x, true) };
}

// The GPU's own keys and values (float16, a pair to a u32), per layer [positions][kvDim], grown as the CPU's cache
// grows (doubling from plan.kvStart: a prompt seldom needs the whole context) and kept from block to block; with them
// the bind groups that read them
function grow(m: Model, needed: number) {
  const { device, plan } = m, b = m.block, old = b.cache, attending = attendingOf(plan);
  const kvDim = plan.kvHeads * plan.headSize, row = kvDim * 2;
  const capacity = Math.min(Math.max(2 * (old?.capacity ?? 0), needed), plan.seqLen);
  if (capacity * row > m.limit) throw new Error(`the keys of ${capacity} positions are more than a buffer of this GPU`);
  const cache: Kept = { capacity, owned: [], keys: [], values: [], rope: [], attention: [] };
  const usage = STORAGE | COPY_SRC | COPY_DST;
  const encoder = device.createCommandEncoder();
  for (let l = 0; l < attending.length; l++) {
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
  // (T261.4: a uniform for each window the layers attend, 0 for every position: a Gemma 3's local layers and the others)
  const windows = new Map<number, GPUBuffer>();
  const attentionParams = (window: number) => {
    if (!windows.has(window)) {
      const params = new ArrayBuffer(16);
      new Uint32Array(params, 0, 2).set([plan.heads, plan.kvHeads]);
      new Float32Array(params, 8, 1)[0] = 1 / Math.sqrt(plan.headSize);
      new Uint32Array(params, 12, 1)[0] = window;
      windows.set(window, uniform(m, params, cache.owned));
    }
    return windows.get(window)!;
  };
  // (T545: an attending layer's by its place p, its layer l's window, tables and RoPE)
  for (const [p, l] of attending.entries()) {
    // (T261.4: a local layer turned by the angles of its own tables)
    const angles = plan.local.includes(l) ? b.anglesLocal! : b.angles;
    cache.rope.push(bind(m, b.rope, [b.q, b.k, b.v, cache.keys[p], cache.values[p], angles, plan.unturned?.includes(l) ? b.ropeAlone! : b.ropeShape, b.step]));
    cache.attention.push(bind(m, m.forms.attention.pipeline, [b.q, cache.keys[p], cache.values[p], b.xb, attentionParams(plan.windows[l] ?? 0), b.step]));
  }
  m.block.cache = cache;
  // (device.ts's end() lets go of these, the ones held last)
  held(m, cache.owned);
}

// the cache grown where it does not hold positions yet (a block's, a step's, a check's)
function roomFor(m: Model, positions: number) {
  if (positions > m.block.cache!.capacity) grow(m, positions);
}
// T152: the keys and values of positions from to pos - 1 up from forward.ts's cache (cache: where each attending layer's
// are in the shared memory and the bytes of a position; half: float16 as here, else float32, narrowed on the way up:
// T160, a grouped-query model's), before the steps of a generation read them. T261.4.2: a layer whose window a ring
// holds there (T510.1) has the last ring positions of them and no more, and needs no more (its window is shorter); a
// position at p mod ring and again ring after it, so that they are one run however they wrap
function keysIn(m: Model, cache: Sent, from: number, pos: number) {
  const { plan } = m, { keys, values } = m.block.cache!, kvDim = plan.kvHeads * plan.headSize, kvRow = kvDim * 2;
  cache.places.forEach((place, l) => {
    const first = place.ring ? Math.max(from, pos - place.ring) : from, slot = place.ring ? first % place.ring : first;
    if (first >= pos) return;
    for (const [base, target] of [[place.keys, keys[l]], [place.values, values[l]]] satisfies [number, GPUBuffer][]) {
      if (cache.half) copyIn(m, target, base + slot * cache.row, (pos - first) * kvRow, first * kvRow);
      else narrowIn(m, target, base + slot * cache.row, (pos - first) * kvDim, first * kvRow);
    }
  });
}

// T148: what a whole block takes here, from the submission to its keys and values read back, for 16 and for 64 tokens
// (plan.batch) at position 0 of the GPU's own cache (which holds nothing of forward.ts's yet): forward.ts weighs a
// block of a prompt on the GPU against the same tokens on the CPU by it (a line through the two, scaled by the blocks
// it then times itself). The two in turn, BLOCK_ROUNDS rounds after one of each to warm up (the pipelines' first
// dispatches, the weights' first reads), the median of each (a device that warms up or is loaded meanwhile falls on
// both alike); one round on a fallback adapter (tests: its times are no GPU's)
const BLOCK_ROUNDS = 3;
async function timeBlocks(m: Model) {
  const { batch, seqLen } = m.plan, counts = [...new Set([Math.min(16, batch, seqLen), Math.min(batch, seqLen)])], times: number[][] = counts.map(() => []);
  const timed = async (count: number) => {
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

// T545, T271.2: the state of the layers that keep one, as a request goes on from it: none at position 0 (the CPU clears
// its own there), its own where the request says so (fresh: the CPU's is the GPU's then), else the CPU's, which it put
// after the keys and values of plan.staging and holds while it waits (the kinds' order: forward/engine.ts's
// stateToStaging)
function stateIn(m: Model, encoder: GPUCommandEncoder, pos: number, fresh: boolean) {
  const { plan } = m, { states } = m.block;
  let at = plan.staging + 2 * attendingOf(plan).length * plan.batch * plan.kvHeads * plan.headSize * 2;
  for (const { buffer: target, bytes } of states) {
    if (pos === 0) encoder.clearBuffer(target);
    else if (!fresh) copyIn(m, target, at, bytes);
    at += bytes;
  }
}
// A block of count tokens at pos through the layers, its keys and values into plan.staging where wanted() still says
// so. timing (T148, timeBlocks): rows and angles of zeros instead of forward.ts's, and nothing written back. fresh
// (T271.2): the block goes on from the GPU's own state of the layers that keep one (stateIn)
async function block(m: Model, count: number, pos: number, wanted: () => boolean, timing = false, tokens?: number[], fresh = false) {
  const { device, plan } = m, half = plan.headSize / 2;
  // (T378.4: this module's record taken apart once: the layers' loop below reads locals. The cache after grow())
  const { norm, headNorm, add, rope, activation, quantize, normAdd, x, angles, anglesLocal, step, rows, turns, turnsLocal, ids, activationGroup, quantizeXb,
    quantizeAttention, quantizeFfn, quantizeGate, embedRows, embedGroups, layers, conv, linear, outputGate } = m.block;
  // the rows forward.ts embedded (dense), and the angles of their positions (the tables forward.ts has); views of just
  // those floats (T155: not a view of the whole of a 64-bit memory of some GB for every block)
  const floats = (address: number, n: number) => new Float32Array(m.memory.buffer, address, n);
  // (T210: on the GPU alone, the tokens themselves, embedded here; token 0 where it is timed)
  if (plan.direct) device.queue.writeBuffer(ids!, 0, Uint32Array.from({ length: count }, (_, t) => (timing ? 0 : tokens![t])));
  else if (timing) rows.fill(0);
  else rows.set(floats(plan.rows, count * plan.dim));
  // (T154: GPT-2 turns nothing and has no tables)
  if (timing || !plan.turned) {
    turns.fill(0);
    turnsLocal?.fill(0);
  } else {
    for (let t = 0; t < count; t++) {
      turns.set(floats(plan.cos + (pos + t) * half * 4, half), t * plan.headSize);
      turns.set(floats(plan.sin + (pos + t) * half * 4, half), t * plan.headSize + half);
      // (T261.4: the local layers' tables, laid out as the others)
      turnsLocal?.set(floats(plan.localCos + (pos + t) * half * 4, half), t * plan.headSize);
      turnsLocal?.set(floats(plan.localSin + (pos + t) * half * 4, half), t * plan.headSize + half);
    }
  }
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  if (!plan.direct) device.queue.writeBuffer(x, 0, rows, 0, count * plan.dim);
  device.queue.writeBuffer(angles, 0, turns, 0, count * plan.headSize);
  if (anglesLocal) device.queue.writeBuffer(anglesLocal, 0, turnsLocal!, 0, count * plan.headSize);
  device.queue.writeBuffer(step, 0, new Uint32Array([count, pos, 0, 0]));
  roomFor(m, pos + count);
  const { cache } = m.block;
  const encoder = device.createCommandEncoder(), form = m.forms.form!;
  // T545: the rows the convolution layers go on from (T271.2: and the linear-attention layers' states and rows)
  stateIn(m, encoder, pos, fresh);
  const pass = encoder.beginComputePass();
  const multiplied = (pieces: Product) => pieces.forEach((piece) => multiply(m, pass, form, piece.group, piece.rows, count));
  // a packed form reads its input quantized: once for the matrices that read the same one
  const quantized = (input: Quantizing) => form.packed && dispatch(pass, quantize, input.group, input.x, count);
  const flash = m.forms.attention;
  // a bias added to a matrix's output (T153: Qwen2's q, k and v; T154: GPT-2's and GPT-NeoX's every matrix's)
  const biased = (bias: Added | undefined) => bias && dispatch(pass, add, bias.group, bias.x, count);
  embedGroups?.forEach((group) => dispatch(pass, embedRows!, group, 1, count));
  for (let l = 0; l < plan.layers; l++) {
    const layer = layers![l];
    dispatch(pass, norm, layer.attentionNorm, 1, count);
    quantized(quantizeXb);
    // T545: a convolution layer: its matrix in, the convolution into xb and the rows moved on by the block's tokens (the
    // last layer leaves those alone: the rows are all a prompt's token leaves there), its matrix out
    if (layer.convolution) {
      const { into, conv: convolved, state, out } = layer.convolution, groups = Math.ceil(plan.dim / 64);
      multiplied(into);
      if (l < plan.layers - 1) dispatch(pass, conv!.short, convolved, groups, count);
      dispatch(pass, conv!.state, state, groups);
      if (l === plan.layers - 1) break;
      quantized(quantizeXb);
      multiplied(out);
      if (layer.attentionPost) dispatch(pass, normAdd!, layer.attentionPost, 1, count);
    } else if (layer.delta) {
      // T271.2: a linear-attention layer: its two matrices of the normed stream, its gates, the convolution and its rows
      // moved on, the l2 norms of q and k, the delta rule over its state, the norm of every value head, z's gate, its
      // matrix out (the last layer's too: its state is what the next token goes on from)
      const { into, z, gates, conv: convolved, state, qNorm, kNorm, rule, norm: valueNorm, gate, out } = layer.delta, L = linear!, { keyHeads, valueHeads, valueDim } = plan.linear!;
      const keys = keyHeads * plan.linear!.keyDim, mixed = 2 * keys + valueHeads * valueDim;
      multiplied(into);
      multiplied(z);
      dispatch(pass, L.gates, gates, valueHeads, count);
      dispatch(pass, L.conv, convolved, Math.ceil(mixed / 64), count);
      dispatch(pass, L.state, state, Math.ceil(mixed / 64));
      dispatch(pass, headNorm, qNorm, keyHeads, count);
      dispatch(pass, headNorm, kNorm, keyHeads, count);
      dispatch(pass, L.rule, rule, valueHeads);
      dispatch(pass, headNorm, valueNorm, valueHeads, count);
      dispatch(pass, L.swiglu, gate, Math.ceil((valueHeads * valueDim) / 64), count);
      if (l === plan.layers - 1) break;
      quantized(L.quantizeZ);
      multiplied(out);
      if (layer.attentionPost) dispatch(pass, normAdd!, layer.attentionPost, 1, count);
    } else {
      const at = plan.places?.[l] ?? l;
      multiplied(layer.q);
      multiplied(layer.k);
      multiplied(layer.v);
      if (layer.outputGate) multiplied(layer.outputGate.into);  // (T271.2: a Qwen3.5's gate of the output)
      // T153: as the CPU has it (forward.ts), the biases, then the norms of the heads, then RoPE
      layer.biases.forEach(biased);
      for (const head of layer.headNorms) dispatch(pass, headNorm, head.group, head.heads, count);
      dispatch(pass, rope, cache!.rope[at], count);
      if (l === plan.layers - 1) break;  // the keys and values are all a prompt's token leaves
      // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, before o adds to it
      if (plan.parallel) dispatch(pass, norm, layer.ffnNorm, 1, count);
      dispatch(pass, flash.pipeline, cache!.attention[at], plan.heads * Math.ceil(count / FLASH_Q_TILE));
      if (layer.outputGate) dispatch(pass, outputGate!.pipeline, layer.outputGate.gate, Math.ceil((plan.heads * plan.headSize) / 64), count);
      quantized(quantizeAttention);
      multiplied(layer.o);
      biased(layer.oBias);
      if (layer.attentionPost) dispatch(pass, normAdd!, layer.attentionPost, 1, count);
    }
    if (!plan.parallel) dispatch(pass, norm, layer.ffnNorm, 1, count);
    quantized(quantizeFfn);
    multiplied(layer.gate);
    if (layer.up) multiplied(layer.up);
    biased(layer.gateBias);
    dispatch(pass, activation, activationGroup, Math.ceil(plan.hidden / 64), count);
    quantized(quantizeGate);
    multiplied(layer.down);
    biased(layer.downBias);
    if (layer.ffnPost) dispatch(pass, normAdd!, layer.ffnPost, 1, count);
  }
  pass.end();
  // (T210: a model on the GPU alone keeps them there: nothing comes back)
  if (!plan.direct) keysOut(m, encoder, pos, count);
  device.queue.submit([encoder.finish()]);
  const invalid = await device.popErrorScope(), full = await device.popErrorScope();
  if (invalid || full) throw new Error(`the GPU refused a block (${(invalid ?? full)!.message})`);
  if (plan.direct) await device.queue.onSubmittedWorkDone();
  else await keysBack(m, () => !timing && wanted());
}
// the keys of count positions from pos of every layer, then their values, [layers][plan.batch][kvDim] each in float16,
// into the block's readback as plan.staging lays them out. T545: of every attending layer, by its place; and after
// them the state of the layers that keep one, as the block left it (T271.2: as it goes up: stateIn)
function keysOut(m: Model, encoder: GPUCommandEncoder, pos: number, count: number) {
  const { plan } = m, { cache, readback, states } = m.block, B = plan.batch, row = plan.kvHeads * plan.headSize * 2, layers = attendingOf(plan).length;
  for (let l = 0; l < layers; l++) {
    encoder.copyBufferToBuffer(cache!.keys[l], pos * row, readback, l * B * row, count * row);
    encoder.copyBufferToBuffer(cache!.values[l], pos * row, readback, (layers + l) * B * row, count * row);
  }
  let at = 2 * layers * B * row;
  for (const { buffer: from, bytes } of plan.direct ? [] : states) {
    encoder.copyBufferToBuffer(from, 0, readback, at, bytes);
    at += bytes;
  }
}
// and from the block's readback into plan.staging, where wanted() still says so
async function keysBack(m: Model, wanted: () => boolean) {
  const { plan } = m, { readback } = m.block;
  await readback.mapAsync(MAP_READ);
  try {
    const bytes = 2 * attendingOf(plan).length * plan.batch * plan.kvHeads * plan.headSize * 2 + (plan.direct ? 0 : m.block.stateBytes);
    if (wanted()) new Uint8Array(m.memory.buffer, plan.staging, bytes).set(new Uint8Array(readback.getMappedRange(), 0, bytes));
  } finally {
    readback.unmap();
  }
}

export { prepare, bindLayers, grow, roomFor, keysIn, BLOCK_ROUNDS, timeBlocks, block, keysOut, keysBack, attendingOf, stateIn };

export type { Block, Quantizing, Kept };
