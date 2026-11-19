// A block of a prompt: the small steps' pipelines and a block's buffers, every layer's bind groups, the GPU's own keys
// and values, the block itself as one submission with its keys and values read back, and what a whole block takes (T148).
// (T352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T367.3)

import { STORAGE, COPY_DST, COPY_SRC, MAP_READ, UNIFORM, stopping, held, within, buffer, uniform, copyIn, narrowIn, validated, pipelineOf, bind,
  dispatch, normShape, angleHeadsOf, widestHead } from "./device.ts";
import { tablesOf } from "./weights.ts";
import { productGroups, multiply } from "./forms.ts";
import { GPU_KINDS, kindsOf, attendingOf } from "./kinds.ts";
import { GROUP } from "../shaders/common.ts";
import { ENGINE } from "../shaders/engine.ts";
import { flashParams } from "../shaders/attention.ts";
import { perLayerParams } from "../shaders/steps.ts";
import { isBlock, asTernary } from "./packings.ts";
import { roundsMs } from "./timing.ts";
import type { Model } from "./device.ts";
import type { Added, BlockStep, KindBlock, Product, Quantizing, State } from "./kinds.ts";
import type { LayerKind, TensorName } from "../contracts/plan.ts";
import type { Cache as Sent, Started, Vectors } from "../contracts/gpu/protocol.ts";

// T368.6: this module's record, m.block, as prepare() makes it (the layers' bind groups and the keys and values come
// later, bindLayers() and grow(): null until then, and read with a ! after).
// (a matrix's product, an input's quantization and a bias added are gpu/kinds.ts's types, which the kinds' layers bind too)
// a layer's bind groups: its kind's step (T568.2: gpu/kinds.ts's table; T545: an LFM2's convolution layers, T271.2 a
// Qwen3.5's linear-attention layers, besides the attending ones), and what every layer has around it: the norm before,
// T261.4 a Gemma 3's norms behind the step and the FFN, each added to the stream (null: none), the FFN (a bias only where
// the model has it; up where the FFN has a gate)
type Layer = {
  step: BlockStep;
  attentionNorm: GPUBindGroup;
  attentionPost: GPUBindGroup | null;
  ffnNorm: GPUBindGroup;
  gate: Product;
  gateBias: Added | undefined;
  up: Product | null;
  down: Product;
  downBias: Added | undefined;
  ffnPost: GPUBindGroup | null;
  // T334.1: what the layer makes of its input of a table of its own (a Gemma 4's: null where the model has none), and the
  // number it ends by multiplying the stream by (null: none)
  perLayer: PerLayerStep | null;
  scale: GPUBindGroup | null;
};
// T334.1: the stream quantized for the gate, the gate, gelu(gate) times the layer's input (GEGLU of the layer's slice of
// the inputs), the gate quantized, the projection back into xb, and its norm added to the stream (RMSNORM_ADD)
type PerLayerStep = { gate: Product; activation: GPUBindGroup; projection: Product; post: GPUBindGroup };
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
  // T334.1: the heads of the layers RoPE turns by each table (a Gemma 4's wide ones by the first, its narrow by the local),
  // which are the strides of a token's angles in angles and anglesLocal; and the RoPE's uniform of each layer
  angleHeads: [number, number];
  ropeOf(l: number): GPUBuffer;
  activationGroup: GPUBindGroup;
  quantizeXb: Quantizing;
  // T334.1: of the attention's output, the layers' of each size of heads (a Gemma 4's wide ones too)
  quantizeHeads: Record<number, Quantizing>;
  quantizeFfn: Quantizing;
  quantizeGate: Quantizing;
  // T271.1: what quantizes n values of a block's buffer for the packed matrices (the kinds' inputs too: QUANTIZE, or in a
  // rotated basis ROTATE_QUANTIZE), and the rotated basis's two shaders (null: the model's own basis), and the turning back
  // of the rows a model on the GPU alone embeds here (null: none)
  quantizing(from: GPUBuffer, n: number): Quantizing;
  rotate: { quantize: GPUComputePipeline; unrotate: GPUComputePipeline } | null;
  // T334.1: a Gemma 4's: the weights of ones the heads of v are normed by (null: no such norm); its inputs of every layer
  // of a block's tokens (null: none): the rows of their table as the CPU reads them ([token][layer][width]), the
  // projection of the embedded rows (the same), the inputs PER_LAYER_INPUTS makes of the two ([layer][batch][width]: a
  // layer's tokens together, so that GEGLU reads a layer's at an offset), the gate of a layer's, the stream quantized
  // for the projection and the gate, the gate quantized for the projection back, PER_LAYER_INPUTS's pipeline and bind
  // group, and the projection (bindLayers()); and SCALE (null: none)
  ones: GPUBuffer | null;
  perLayer: { rows: GPUBuffer; projected: GPUBuffer; inputs: GPUBuffer; gate: GPUBuffer; quantizeStream: Quantizing; quantizeGate: Quantizing;
    pipeline: GPUComputePipeline; group: GPUBindGroup; project: Product | null } | null;
  scale: GPUComputePipeline | null;
  unrotateRows: Quantizing | null;
  embedRows: GPUComputePipeline | null;
  ids: GPUBuffer | null;
  embedGroups: GPUBindGroup[] | null;
  layers: Layer[] | null;
  cache: Kept | null;
  // T568.2: what each kind the model's layers are of made of a block (gpu/kinds.ts: its shaders, its buffers, its
  // layers' steps, its own of a token), by its name
  kinds: Partial<Record<LayerKind, KindBlock>>;
  // the state of the layers that keep one, as the CPU stages it (the kinds' in their table's order: T545 an LFM2's rows;
  // T271.2 a Qwen3.5's states, then its rows), and its bytes
  states: State[];
  stateBytes: number;
  // T466: the matrices of a layer the model has, each with what it reads and writes (up is null where the FFN has no
  // gate, and then no w3 is there to multiply), whether it adds, and the quantization of its input where it is the
  // first to read it: forms.ts's timeForms() multiplies them so on the first layer, knowing none of the buffers
  products: Timed[];
};
// a matrix of the layer as products holds it
type Timed = [TensorName, GPUBuffer, GPUBuffer | null, boolean?, Quantizing?];
// the pipelines of a layer's small steps and the buffers of a block's activations
// T378.4: they are this module's record, m.block, made here once and set by this module alone (bindLayers() puts the
// layers' bind groups in it once the forms are chosen, grow() the keys and values); forms.ts and tokens.ts read its
// pipelines and buffers for their own bind groups
async function prepare(m: Model) {
  const { plan, device } = m, B = plan.batch, gated = Boolean(plan.matrices.w3);
  // (T568.2: and what a kind's layers quantize besides: T271.2 a Qwen3.5's linear-attention layers' read, which z's gate
  // leaves for their matrix out)
  // (T334.1: of the widest heads, where the layers have heads of their own sizes: a Gemma 4's)
  const widest = widestHead(plan), qDim = plan.heads * widest, kvDim = plan.kvHeads * widest, present = kindsOf(plan);
  const quantized = Math.max(plan.dim, qDim, plan.hidden, plan.perLayer?.width ?? 0, ...present.map((kind) => GPU_KINDS[kind].widest?.(plan) ?? 0));
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
  const xq = buffer(m, B * quantized);
  const xs = buffer(m, B * (quantized / GROUP) * 4);
  // (T334.1: a token's angles of each table are as many as the heads of its layers have: a Gemma 4's wide ones' and its
  // narrow ones')
  const angleHeads = angleHeadsOf(plan);
  const angles = buffer(m, B * angleHeads[0] * 4, STORAGE | COPY_DST);
  // (T261.4: and those of the local layers' tables, where the model has them)
  const anglesLocal = plan.localCos ? buffer(m, B * angleHeads[1] * 4, STORAGE | COPY_DST) : null;
  const step = buffer(m, 16, UNIFORM | COPY_DST);
  // T271.1: a rotated basis's shaders
  const rotate: Block["rotate"] = plan.rotated ? await within(Promise.all([ENGINE.ROTATE_QUANTIZE, ENGINE.UNROTATE].map((code) => pipelineOf(m, code))), "the rotated basis's shaders")
    .then(([turn, back]) => ({ quantize: turn, unrotate: back })) : null;
  // T568.2: what each kind of the model's layers makes of a block (T545: an LFM2's convolution layers'; T271.2: a
  // Qwen3.5's linear-attention layers', and the gate of its attending layers' output)
  const kinds: Block["kinds"] = {};
  for (const kind of present) kinds[kind] = await GPU_KINDS[kind].block(m);
  const states = present.flatMap((kind) => kinds[kind]!.states);
  const stateBytes = states.reduce((sum, { bytes }) => sum + bytes, 0);
  // (the attending layers' keys and values, then the state of the layers that keep one: T545; T271.4: none of it on the
  // GPU alone, whose state stays here, and whose keys and values come back for the tests alone)
  const readback = buffer(m, 2 * attendingOf(plan).length * B * kvDim * 2 + (plan.direct ? 0 : stateBytes), MAP_READ | COPY_DST);
  // what a request writes into them, from the shared memory
  const rows = new Float32Array(B * plan.dim);
  const turns = new Float32Array(B * angleHeads[0]);
  const turnsLocal = plan.localCos ? new Float32Array(B * angleHeads[1]) : null;
  // (T255: a layer RoPE leaves alone turns none of a head, as every layer of a GPT-2; T334.1: each layer of its own heads
  // and turned values, one uniform for each such pair, made in the order of the layers)
  const ropes = new Map<string, GPUBuffer>();
  for (const { head, turned } of plan.shapes) {
    if (!ropes.has(`${head} ${turned}`)) ropes.set(`${head} ${turned}`, uniform(m, new Uint32Array([plan.heads, plan.kvHeads, head, turned])));
  }
  const ropeOf = (l: number) => ropes.get(`${plan.shapes[l].head} ${plan.shapes[l].turned}`)!;
  const hidden = uniform(m, new Uint32Array([plan.hidden, 0, 0, 0]));
  const activationGroup = bind(m, activation, gated ? [gate, up!, hidden, step] : [gate, hidden, step]);
  // QUANTIZE of the inputs the matrices read: xb (the norm's, dim wide; the attention's, heads × headSize wide, which
  // is not dim where the heads are of another size, T153) and gate (SwiGLU's)
  // (T271.1: in a rotated basis R of them, a workgroup a block of a token: the signs of their width)
  const quantizing = (from: GPUBuffer, n: number): Quantizing => (rotate
    ? { pipeline: rotate.quantize, group: bind(m, rotate.quantize, [from, xq, xs, m.weights.signs![n], uniform(m, new Uint32Array([n, n, plan.rotated!.block, 0]))]),
      x: n / plan.rotated!.block }
    : { pipeline: quantize, group: bind(m, quantize, [from, xq, xs, uniform(m, new Uint32Array([n, n, 0, 0])), step]), x: Math.ceil(n / GROUP / 64) });
  const quantizeXb = quantizing(xb, plan.dim);
  // (T334.1: of each size of heads the layers have)
  const quantizeHeads = Object.fromEntries([...new Set([plan.headSize, ...plan.shapes.map(({ head }) => head)])].map((head) => [head, quantizing(xb, plan.heads * head)] as const));
  const quantizeAttention = quantizeHeads[plan.headSize];
  const quantizeFfn = plan.parallel ? quantizing(ffnInput, plan.dim) : quantizeXb;
  const quantizeGate = quantizing(gate, plan.hidden);
  // T210: a model on the GPU alone embeds a block's tokens here (the CPU holds no embedding): EMBED_ROWS a piece of
  // the table (T209), from the block's ids into x
  let embedRows: GPUComputePipeline | null = null, ids: GPUBuffer | null = null, embedGroups: GPUBindGroup[] | null = null;
  if (plan.direct) {
    // (T232: from a table of ternary weights, EMBED_ROWS_TERNARY)
    // (T547.3.7: from a table of llama.cpp's blocks, its type's rows of embedBlocks, which bind no scales, or an IQ
    // type's codebooks there)
    const embedded = tablesOf(plan).embedding.packing;
    const rowsCode = isBlock(embedded) ? ENGINE.embedBlocks({ weights: embedded, rows: true }) : asTernary(embedded) ? ENGINE.EMBED_ROWS_TERNARY : ENGINE.EMBED_ROWS;
    embedRows = await within(validated(m, () => pipelineOf(m, rowsCode)), "compiling the embedding's rows");
    ids = buffer(m, B * 4, STORAGE | COPY_DST);
    embedGroups = m.weights.tables!.embedding.map((piece) => bind(m, embedRows!,
      [piece.values, piece.scales, ids!, x, uniform(m, new Uint32Array([plan.dim, piece.first, piece.rows, 0]))]));
  }
  // (T271.1: the embedding of a rotated basis holds R^-1 of its rows: turned back here, where the GPU embeds them)
  const unrotateRows = plan.direct && rotate ? { pipeline: rotate.unrotate, group: bind(m, rotate.unrotate, [x, m.weights.signs![plan.dim],
    uniform(m, new Uint32Array([plan.dim, plan.dim, plan.rotated!.block, 0]))]), x: plan.dim / plan.rotated!.block } : null;
  // T334.1: a Gemma 4's heads of v normed by weights of ones, its inputs of every layer and its number of a layer
  let ones: GPUBuffer | null = null;
  if (plan.valueNorm) {
    ones = buffer(m, widest * 4, STORAGE | COPY_DST);
    device.queue.writeBuffer(ones, 0, new Float32Array(widest).fill(1));
  }
  const width = plan.perLayer?.width ?? 0;
  let perLayer: Block["perLayer"] = null;
  if (width) {
    const rows = buffer(m, B * plan.layers * width * 4, STORAGE | COPY_DST), projected = buffer(m, B * plan.layers * width * 4);
    const inputs = buffer(m, plan.layers * B * width * 4), gate = buffer(m, B * width * 4);
    const pipeline = await within(pipelineOf(m, ENGINE.PER_LAYER_INPUTS), "the inputs of every layer");
    const shape = perLayerParams(width, B, plan.eps);
    perLayer = { rows, projected, inputs, gate, quantizeStream: quantizing(x, plan.dim), quantizeGate: quantizing(gate, width), pipeline,
      group: bind(m, pipeline, [projected, m.weights.vectors!.perLayerNorm!, rows, inputs, uniform(m, shape), step]), project: null };
  }
  const scale = plan.vectors.layerScale ? await within(pipelineOf(m, ENGINE.SCALE), "the layers' number") : null;
  m.block = { norm, headNorm, add, rope, activation, quantize, normAdd: normAdd ?? null, x, xb, q, k, v, gate, up, ffnInput, xq, xs, angles, anglesLocal, step,
    readback, rows, turns, turnsLocal, angleHeads, ropeOf, activationGroup, quantizeXb, quantizeHeads, quantizeFfn, quantizeGate, quantizing, rotate, unrotateRows,
    ones, perLayer, scale,
    embedRows, ids, embedGroups,
    // (bindLayers() and grow())
    layers: null, cache: null, kinds, states, stateBytes,
    products: ([["wq", xb, q, false, quantizeXb], ["wk", xb, k], ["wv", xb, v], ["wo", xb, x, true, quantizeAttention],
      ["w1", ffnInput, gate, false, quantizeFfn], ["w3", ffnInput, up], ["w2", gate, x, true, quantizeGate]] satisfies Timed[])
      .filter(([name]) => m.weights.matrices[name]) };
}

// every layer's bind groups but those of the cache, with the form chosen. T153: a bias (Qwen2's) added to q, k and
// v, or (Qwen3's) the norm of every head of q and of k, where the model has them. T154: LayerNorm with its bias, and a
// bias after o, w1 and w2 (GPT-2's, GPT-NeoX's)
function bindLayers(m: Model) {
  const { plan } = m, form = m.forms.form!, V = m.weights.vectors!, b = m.block;
  const layers: Layer[] = [];
  // (T334.1: a Gemma 4's projection of a block's embedded rows into the inputs of every layer, the model's matrix)
  if (b.perLayer) b.perLayer.project = productGroups(m, form, "per_layer_projection", 0, b.x, b.perLayer.projected);
  // (T334.1: those a block runs: the layers after the last that keeps keys and values of its own or a state, a Gemma 4's
  // that read another's, have none of their matrices here, or not in the stacks this binds)
  for (let l = 0; l < plan.blockLayers; l++) {
    // T545: a matrix or a vector of the layers of a kind (an LFM2's attending or convolution layers': of) is read at the
    // layer's place among them, every layer's at the layer
    const place = plan.places?.[l] ?? l;
    const index = (of: string | undefined) => (of ? place : l);
    // (a piece each, T155)
    // (T334.1: or at its place in the matrix's stack, where the caller says it)
    const product = (name: TensorName, from: GPUBuffer, to: GPUBuffer, add = false, at = plan.matrices[name]?.at?.[l] ?? index(plan.matrices[name]?.of)) => productGroups(m, form, name, at, from, to, add);
    const normBound = normShape(m, plan.dim, l * plan.dim, plan.eps);
    const added = (to: GPUBuffer, name: keyof Vectors) => V[name] && { group: bind(m, b.add, [to, V[name], uniform(m, new Uint32Array([plan.vectors[name]!.size, index(plan.vectors[name]!.of) * plan.vectors[name]!.size, 0, 0])), b.step]),
      x: Math.ceil(plan.vectors[name]!.size / 64) };
    // T154: LayerNorm reads its bias from the same place in its own buffer
    const normed = (weight: "attention" | "ffn", bias: "attentionBias" | "ffnBias", out: GPUBuffer) => bind(m, b.norm, plan.layerNorm ? [b.x, V[weight], V[bias]!, out, normBound, b.step]
      : [b.x, V[weight], out, normBound, b.step]);
    // T261.4: the norm of what joins the stream (from), added to it (a Gemma 3's: the output of the step's matrix out,
    // and of w2, written apart into gate and xb, which nothing reads by then)
    const post = (name: keyof Vectors, from: GPUBuffer) => (V[name] ? bind(m, b.normAdd!, [from, V[name], b.x, normBound, b.step]) : null);
    // (the step's matrix out: added to the stream, or written into gate for the norm behind it)
    const out = (name: TensorName, from: GPUBuffer, at?: number) => (V.attentionPost ? product(name, from, b.gate, false, at) : product(name, from, b.x, true, at));
    layers.push({
      // T568.2: the layer's kind's step (gpu/kinds.ts)
      step: b.kinds[plan.kinds[l]]!.layer({ l, place, product, out, added }),
      attentionNorm: normed("attention", "attentionBias", b.xb), attentionPost: post("attentionPost", b.gate),
      ffnNorm: normed("ffn", "ffnBias", b.ffnInput),
      gate: product("w1", b.ffnInput, b.gate), gateBias: added(b.gate, "b1"),
      up: m.weights.matrices.w3 ? product("w3", b.ffnInput, b.up!) : null,
      down: V.ffnPost ? product("w2", b.gate, b.xb) : product("w2", b.gate, b.x, true), downBias: added(b.x, "b2"), ffnPost: post("ffnPost", b.xb),
      // T334.1: a Gemma 4's input of the layer: gelu(gate · the stream) times its slice of the inputs, projected back into
      // xb and its norm added to the stream; and its number of the layer
      perLayer: b.perLayer && {
        gate: product("ple_gate", b.x, b.perLayer.gate),
        activation: bind(m, b.activation, [b.perLayer.gate, { buffer: b.perLayer.inputs, offset: l * plan.batch * plan.perLayer!.width * 4, size: plan.batch * plan.perLayer!.width * 4 },
          uniform(m, new Uint32Array([plan.perLayer!.width, 0, 0, 0])), b.step]),
        projection: product("ple_proj", b.perLayer.gate, b.xb), post: post("plePost", b.xb)! },
      scale: b.scale && bind(m, b.scale, [b.x, V.layerScale!, uniform(m, new Uint32Array([plan.dim, l, 0, 0])), b.step]),
    });
  }
  m.block.layers = layers;
}
// The GPU's own keys and values (float16, a pair to a u32), per layer [positions][kvDim], grown as the CPU's cache
// grows (doubling from plan.kvStart: a prompt seldom needs the whole context) and kept from block to block; with them
// the bind groups that read them
function grow(m: Model, needed: number) {
  const { device, plan } = m, b = m.block, old = b.cache, attending = attendingOf(plan);
  const capacity = Math.min(Math.max(2 * (old?.capacity ?? 0), needed), plan.seqLen);
  // (T334.1: a layer's position of its own heads' size: a Gemma 4's wide ones' are the widest)
  const rowOf = (l: number) => plan.kvHeads * plan.shapes[l].head * 2;
  if (capacity * Math.max(...attending.map(rowOf)) > m.limit) throw new Error(`the keys of ${capacity} positions are more than a buffer of this GPU`);
  const cache: Kept = { capacity, owned: [], keys: [], values: [], rope: [], attention: [] };
  const usage = STORAGE | COPY_SRC | COPY_DST;
  const encoder = device.createCommandEncoder();
  for (const [p, l] of attending.entries()) {
    // (T334.1: a layer that reads another's keys and values, a Gemma 4's, has that layer's buffers, which come before it)
    const { kept, reads } = plan.shapes[l];
    if (kept < 0) {
      cache.keys.push(cache.keys[reads]);
      cache.values.push(cache.values[reads]);
      continue;
    }
    const row = rowOf(l), keys = buffer(m, capacity * row, usage, cache.owned), values = buffer(m, capacity * row, usage, cache.owned);
    if (old) {
      encoder.copyBufferToBuffer(old.keys[p], 0, keys, 0, old.capacity * row);
      encoder.copyBufferToBuffer(old.values[p], 0, values, 0, old.capacity * row);
    }
    cache.keys.push(keys);
    cache.values.push(values);
  }
  device.queue.submit([encoder.finish()]);
  // the old ones go once the copies are done (a buffer destroyed after its submission lives until the GPU is through)
  old?.owned.forEach((buffer) => buffer.destroy());
  // (T261.4: a uniform for each window the layers attend, 0 for every position: a Gemma 3's local layers and the others;
  // T334.1: and for each size of heads, a Gemma 4's two)
  const windows = new Map<string, GPUBuffer>();
  const attentionParams = (window: number, head: number) => {
    if (!windows.has(`${window} ${head}`)) {
      windows.set(`${window} ${head}`, uniform(m, flashParams(plan.heads, plan.kvHeads, head, window), cache.owned));
    }
    return windows.get(`${window} ${head}`)!;
  };
  // (T545: an attending layer's by its place p, its layer l's window, tables and RoPE; T334.1: and its heads', whose
  // attention is the one of their size)
  for (const [p, l] of attending.entries()) {
    // (T261.4: a local layer turned by the angles of its own tables)
    const angles = plan.local.includes(l) ? b.anglesLocal! : b.angles, { head } = plan.shapes[l];
    cache.rope.push(bind(m, b.rope, [b.q, b.k, b.v, cache.keys[p], cache.values[p], angles, b.ropeOf(l), b.step]));
    cache.attention.push(bind(m, m.forms.attention.pipelines[head], [b.q, cache.keys[p], cache.values[p], b.xb, attentionParams(plan.windows[l] ?? 0, head), b.step]));
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
// are in the shared memory and the bytes of a position there; half: float16 as here, else float32, narrowed on the way
// up: T160, a grouped-query model's), before the steps of a generation read them. T261.4.2: a layer whose window a ring
// holds there (T510.1) has the last ring positions of them and no more, and needs no more (its window is shorter); a
// position at p mod ring and again ring after it, so that they are one run however they wrap. T334.2: each layer's of
// its own heads (a place's own bytes of a position there), and none of a layer that reads another's keys and values (a
// Gemma 4's: the CPU keeps none of its own, and the GPU's are those of the layer it reads, which went up with it)
function keysIn(m: Model, cache: Sent, from: number, pos: number) {
  const { plan } = m, { keys, values } = m.block.cache!, attending = attendingOf(plan);
  cache.places.forEach((place, p) => {
    const { head, kept } = plan.shapes[attending[p]], kvDim = plan.kvHeads * head, kvRow = kvDim * 2;
    const first = place.ring ? Math.max(from, pos - place.ring) : from, slot = place.ring ? first % place.ring : first;
    if (kept < 0 || first >= pos) return;
    for (const [base, target] of [[place.keys, keys[p]], [place.values, values[p]]] satisfies [number, GPUBuffer][]) {
      if (cache.half) copyIn(m, target, base + slot * place.row, (pos - first) * kvRow, first * kvRow);
      else narrowIn(m, target, base + slot * place.row, (pos - first) * kvDim, first * kvRow);
    }
  });
}

// T148: what a whole block takes here, from the submission to its keys and values read back, for 16 and for 64 tokens
// (plan.batch) at position 0 of the GPU's own cache (which holds nothing of forward.ts's yet): forward.ts weighs a
// block of a prompt on the GPU against the same tokens on the CPU by it (a line through the two, scaled by the blocks
// it then times itself). The two in turn: timing.ts's roundsMs (a device that warms up or is loaded meanwhile
// falls on both alike); none if a stop came
async function timeBlocks(m: Model) {
  const { batch, seqLen } = m.plan, counts = [...new Set([Math.min(16, batch, seqLen), Math.min(batch, seqLen)])];
  const ms = await roundsMs(m, counts, (count) => block(m, count, 0, () => false, true));
  return stopping ? [] : counts.map((count, i) => ({ count, ms: ms[i] }));
}

// T545, T271.2: the state of the layers that keep one, as a request goes on from it: none at position 0 (the CPU clears
// its own there), its own where the request says so (fresh: the CPU's is the GPU's then), else the CPU's, which it put
// after the keys and values of plan.staging and holds while it waits (the kinds' order: forward/engine.ts's
// stateToStaging)
function stateIn(m: Model, encoder: GPUCommandEncoder, pos: number, fresh: boolean) {
  const { plan } = m, { states } = m.block;
  let at = plan.staging + 2 * attendingOf(plan).length * plan.batch * plan.kvHeads * widestHead(plan) * 2;
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
  const { device, plan } = m;
  // (T378.4: this module's record taken apart once: the layers' loop below reads locals. The cache after grow())
  const { norm, add, activation, normAdd, x, angles, anglesLocal, step, rows, turns, turnsLocal, ids, activationGroup, quantizeXb,
    quantizeFfn, quantizeGate, unrotateRows, embedRows, embedGroups, layers, angleHeads, perLayer, scale } = m.block;
  // (T334.1: a token's angles of a table are as many as its layers' heads have)
  const [head, headLocal] = angleHeads, half = head / 2, halfLocal = headLocal / 2;
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
      turns.set(floats(plan.cos + (pos + t) * half * 4, half), t * head);
      turns.set(floats(plan.sin + (pos + t) * half * 4, half), t * head + half);
      // (T261.4: the local layers' tables, laid out as the others)
      turnsLocal?.set(floats(plan.localCos + (pos + t) * halfLocal * 4, halfLocal), t * headLocal);
      turnsLocal?.set(floats(plan.localSin + (pos + t) * halfLocal * 4, halfLocal), t * headLocal + halfLocal);
    }
  }
  // (T334.1: a Gemma 4's rows of the table of every layer's inputs, as the CPU read them for the block's tokens; zeros
  // where it is timed)
  const tableRows = perLayer && (timing ? new Float32Array(count * plan.layers * plan.perLayer!.width) : floats(plan.perLayer!.inputs, count * plan.layers * plan.perLayer!.width));
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  if (!plan.direct) device.queue.writeBuffer(x, 0, rows, 0, count * plan.dim);
  device.queue.writeBuffer(angles, 0, turns, 0, count * head);
  if (anglesLocal) device.queue.writeBuffer(anglesLocal, 0, turnsLocal!, 0, count * headLocal);
  if (perLayer) device.queue.writeBuffer(perLayer.rows, 0, tableRows!);
  device.queue.writeBuffer(step, 0, new Uint32Array([count, pos, 0, 0]));
  roomFor(m, pos + count);
  const encoder = device.createCommandEncoder(), form = m.forms.form!;
  // T545: the rows the convolution layers go on from (T271.2: and the linear-attention layers' states and rows)
  stateIn(m, encoder, pos, fresh);
  const pass = encoder.beginComputePass();
  const multiplied = (pieces: Product) => pieces.forEach((piece) => multiply(m, pass, form, piece.pipeline, piece.group, piece.rows, count));
  // a packed form reads its input quantized: once for the matrices that read the same one
  const quantized = (input: Quantizing) => form.packed && dispatch(pass, input.pipeline, input.group, input.x, count);
  // a bias added to a matrix's output (T153: Qwen2's q, k and v; T154: GPT-2's and GPT-NeoX's every matrix's)
  const biased = (bias: Added | undefined) => bias && dispatch(pass, add, bias.group, bias.x, count);
  embedGroups?.forEach((group) => dispatch(pass, embedRows!, group, 1, count));
  if (unrotateRows) dispatch(pass, unrotateRows.pipeline, unrotateRows.group, unrotateRows.x, count);
  // (T334.1: a Gemma 4's inputs of every layer, of the embedded rows and the table's)
  if (perLayer) {
    quantized(perLayer.quantizeStream);
    multiplied(perLayer.project!);
    dispatch(pass, perLayer.pipeline, perLayer.group, plan.layers, count);
  }
  for (let l = 0; l < plan.blockLayers; l++) {
    const layer = layers![l];
    dispatch(pass, norm, layer.attentionNorm, 1, count);
    quantized(quantizeXb);
    // T568.2: the layer's kind's step (gpu/kinds.ts), which the last layer ends the block in
    const running = { pass, count, last: l === plan.blockLayers - 1, multiplied, quantized, biased, ffnNorm: () => dispatch(pass, norm, layer.ffnNorm, 1, count) };
    if (layer.step(running)) break;
    if (layer.attentionPost) dispatch(pass, normAdd!, layer.attentionPost, 1, count);
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
    // T334.1: a Gemma 4's input of the layer (after the FFN, as the CPU's perLayer.ts), then its number
    if (layer.perLayer) {
      quantized(perLayer!.quantizeStream);
      multiplied(layer.perLayer.gate);
      dispatch(pass, activation, layer.perLayer.activation, Math.ceil(plan.perLayer!.width / 64), count);
      quantized(perLayer!.quantizeGate);
      multiplied(layer.perLayer.projection);
      dispatch(pass, normAdd!, layer.perLayer.post, 1, count);
    }
    if (layer.scale) dispatch(pass, scale!, layer.scale, Math.ceil(plan.dim / 64), count);
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
  const { plan } = m, { cache, readback, states } = m.block, B = plan.batch, row = plan.kvHeads * widestHead(plan) * 2, attending = attendingOf(plan), layers = attending.length;
  // (T334.1: a layer's positions of its own heads' size, dense, in a place of the widest's; none of a layer that reads
  // another's keys and values, a Gemma 4's: the CPU's cache has none of its own either)
  for (const [p, l] of attending.entries()) {
    const own = plan.kvHeads * plan.shapes[l].head * 2;
    if (plan.shapes[l].kept < 0) continue;
    encoder.copyBufferToBuffer(cache!.keys[p], pos * own, readback, p * B * row, count * own);
    encoder.copyBufferToBuffer(cache!.values[p], pos * own, readback, (layers + p) * B * row, count * own);
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
    const bytes = 2 * attendingOf(plan).length * plan.batch * plan.kvHeads * widestHead(plan) * 2 + (plan.direct ? 0 : m.block.stateBytes);
    if (wanted()) new Uint8Array(m.memory.buffer, plan.staging, bytes).set(new Uint8Array(readback.getMappedRange(), 0, bytes));
  } finally {
    readback.unmap();
  }
}

export { prepare, bindLayers, grow, roomFor, keysIn, timeBlocks, block, keysOut, keysBack, stateIn };

export type { Block, Quantizing, Kept };
