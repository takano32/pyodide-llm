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
import type { Cache as Sent, Vectors } from "../contracts/gpu/protocol.ts";

// T368.6: this module's record, m.block, as prepare() makes it (the layers' bind groups and the keys and values come
// later, bindLayers() and grow(): null until then, and read with a ! after).
// an input's quantization (QUANTIZE's bind group and workgroups), a bias added (ADD's), a matrix's product (a bind
// group and its rows for each piece, forms.ts's productGroups())
type Quantizing = { group: GPUBindGroup; x: number };
type Added = { group: GPUBindGroup; x: number };
type Product = { rows: number; group: GPUBindGroup }[];
type HeadNorm = { group: GPUBindGroup; heads: number };
// a layer's bind groups (a bias only where the model has it; up where the FFN has a gate)
type Layer = {
  attentionNorm: GPUBindGroup;
  q: Product;
  k: Product;
  v: Product;
  biases: Added[];
  headNorms: HeadNorm[];
  o: Product;
  oBias: Added | undefined;
  ffnNorm: GPUBindGroup;
  gate: Product;
  gateBias: Added | undefined;
  up: Product | null;
  down: Product;
  downBias: Added | undefined;
};
// the GPU's own keys and values, a buffer a layer each, and the bind groups that read them
type Kept = { capacity: number; owned: GPUBuffer[]; keys: GPUBuffer[]; values: GPUBuffer[]; rope: GPUBindGroup[]; attention: GPUBindGroup[] };
type Block = {
  norm: GPUComputePipeline;
  headNorm: GPUComputePipeline;
  add: GPUComputePipeline;
  rope: GPUComputePipeline;
  activation: GPUComputePipeline;
  quantize: GPUComputePipeline;
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
  step: GPUBuffer;
  readback: GPUBuffer;
  rows: Float32Array<ArrayBuffer>;
  turns: Float32Array<ArrayBuffer>;
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
};

// the pipelines of a layer's small steps and the buffers of a block's activations
// T378.4: they are this module's record, m.block, made here once and set by this module alone (bindLayers() puts the
// layers' bind groups in it once the forms are chosen, grow() the keys and values); forms.ts and tokens.ts read its
// pipelines and buffers for their own bind groups
async function prepare(m: Model) {
  const { plan } = m, B = plan.batch, gated = Boolean(plan.matrices.w3);
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, widest = Math.max(plan.dim, qDim, plan.hidden);
  // T154: LayerNorm and GELU where the model has them (GPT-2, GPT-NeoX), else RMSNorm and SwiGLU
  const [norm, headNorm, add, rope, activation, quantize] = await within(Promise.all([plan.layerNorm ? ENGINE.LAYER_NORM : ENGINE.RMSNORM,
    ENGINE.HEAD_NORM, ENGINE.ADD, ENGINE.ROPE, gated ? ENGINE.SWIGLU : ENGINE.GELU, ENGINE.QUANTIZE].map((code) => pipelineOf(m, code))), "the small steps' shaders");
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
  const step = buffer(m, 16, UNIFORM | COPY_DST);
  const readback = buffer(m, 2 * plan.layers * B * kvDim * 2, MAP_READ | COPY_DST);
  // what a request writes into them, from the shared memory
  const rows = new Float32Array(B * plan.dim);
  const turns = new Float32Array(B * plan.headSize);
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
  m.block = { norm, headNorm, add, rope, activation, quantize, x, xb, q, k, v, gate, up, ffnInput, xq, xs, angles, step, readback, rows, turns,
    ropeShape, ropeAlone, activationGroup, quantizeXb, quantizeAttention, quantizeFfn, quantizeGate, embedRows, ids, embedGroups,
    // (bindLayers() and grow())
    layers: null, cache: null };
}

// every layer's bind groups but those of the cache, with the form chosen. T153: a bias (Qwen2's) added to q, k and
// v, or (Qwen3's) the norm of every head of q and of k, where the model has them. T154: LayerNorm with its bias, and a
// bias after o, w1 and w2 (GPT-2's, GPT-NeoX's)
function bindLayers(m: Model) {
  const { plan } = m, form = m.forms.form!, V = m.weights.vectors!, b = m.block;
  // (a piece each, T155)
  const product = (name: string, l: number, from: GPUBuffer, to: GPUBuffer, add = false) => productGroups(m, form, name, l, from, to, add);
  const layers: Layer[] = [];
  for (let l = 0; l < plan.layers; l++) {
    const norm = new ArrayBuffer(16);
    new Uint32Array(norm, 0, 2).set([plan.dim, l * plan.dim]);
    new Float32Array(norm, 8, 1)[0] = plan.eps;
    const normShape = uniform(m, norm);
    const added = (to: GPUBuffer, name: keyof Vectors) => V[name] && { group: bind(m, b.add, [to, V[name], uniform(m, new Uint32Array([plan.vectors[name]!.size, l * plan.vectors[name]!.size, 0, 0])), b.step]),
      x: Math.ceil(plan.vectors[name]!.size / 64) };
    // T154: LayerNorm reads its bias from the same place in its own buffer
    const normed = (weight: string, bias: string, out: GPUBuffer) => bind(m, b.norm, plan.layerNorm ? [b.x, V[weight], V[bias], out, normShape, b.step]
      : [b.x, V[weight], out, normShape, b.step]);
    const headNorm = (of: GPUBuffer, name: string, heads: number) => {
      if (!V[name]) return null;
      const shape = new ArrayBuffer(16);
      new Uint32Array(shape, 0, 2).set([plan.headSize, l * plan.headSize]);
      new Float32Array(shape, 8, 1)[0] = plan.eps;
      return { group: bind(m, b.headNorm, [of, V[name], uniform(m, shape), b.step]), heads };
    };
    layers.push({
      attentionNorm: normed("attention", "attentionBias", b.xb),
      q: product("wq", l, b.xb, b.q), k: product("wk", l, b.xb, b.k), v: product("wv", l, b.xb, b.v),
      biases: [added(b.q, "bq"), added(b.k, "bk"), added(b.v, "bv")].filter(Boolean) as Added[],
      headNorms: [headNorm(b.q, "qNorm", plan.heads), headNorm(b.k, "kNorm", plan.kvHeads)].filter(Boolean) as HeadNorm[],
      o: product("wo", l, b.xb, b.x, true), oBias: added(b.x, "bo"),
      ffnNorm: normed("ffn", "ffnBias", b.ffnInput),
      gate: product("w1", l, b.ffnInput, b.gate), gateBias: added(b.gate, "b1"),
      up: m.weights.matrices.w3 ? product("w3", l, b.ffnInput, b.up!) : null,
      down: product("w2", l, b.gate, b.x, true), downBias: added(b.x, "b2"),
    });
  }
  m.block.layers = layers;
}

// The GPU's own keys and values (float16, a pair to a u32), per layer [positions][kvDim], grown as the CPU's cache
// grows (doubling from plan.kvStart: a prompt seldom needs the whole context) and kept from block to block; with them
// the bind groups that read them
function grow(m: Model, needed: number) {
  const { device, plan } = m, b = m.block, old = b.cache;
  const kvDim = plan.kvHeads * plan.headSize, row = kvDim * 2;
  const capacity = Math.min(Math.max(2 * (old?.capacity ?? 0), needed), plan.seqLen);
  if (capacity * row > m.limit) throw new Error(`the keys of ${capacity} positions are more than a buffer of this GPU`);
  const cache: Kept = { capacity, owned: [], keys: [], values: [], rope: [], attention: [] };
  const usage = STORAGE | COPY_SRC | COPY_DST;
  const encoder = device.createCommandEncoder();
  for (let l = 0; l < plan.layers; l++) {
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
  const params = new ArrayBuffer(16);
  new Uint32Array(params, 0, 2).set([plan.heads, plan.kvHeads]);
  new Float32Array(params, 8, 1)[0] = 1 / Math.sqrt(plan.headSize);
  const attentionParams = uniform(m, params, cache.owned);
  for (let l = 0; l < plan.layers; l++) {
    cache.rope.push(bind(m, b.rope, [b.q, b.k, b.v, cache.keys[l], cache.values[l], b.angles, plan.unturned?.includes(l) ? b.ropeAlone! : b.ropeShape, b.step]));
    cache.attention.push(bind(m, m.forms.attention.pipeline, [b.q, cache.keys[l], cache.values[l], b.xb, attentionParams, b.step]));
  }
  m.block.cache = cache;
  // (device.ts's end() lets go of these, the ones held last)
  held(m, cache.owned);
}

// the cache grown where it does not hold positions yet (a block's, a step's, a check's)
function roomFor(m: Model, positions: number) {
  if (positions > m.block.cache!.capacity) grow(m, positions);
}
// T152: the keys and values of positions from to pos - 1 up from forward.ts's cache (cache: its addresses of the keys
// and the values in the shared memory, its capacity and the bytes of a position; half: float16 as here, else float32,
// narrowed on the way up: T160, a grouped-query model's), before the steps of a generation read them
function keysIn(m: Model, cache: Sent, from: number, pos: number) {
  const { plan } = m, { keys, values } = m.block.cache!, kvDim = plan.kvHeads * plan.headSize, kvRow = kvDim * 2;
  const at = (block: number, l: number, p: number) => block + l * cache.capacity * cache.row + p * cache.row;
  for (let l = 0; l < plan.layers; l++) {
    for (const [block, target] of [[cache.keys, keys[l]], [cache.values, values[l]]] satisfies [number, GPUBuffer][]) {
      if (cache.half) copyIn(m, target, at(block, l, from), (pos - from) * kvRow, from * kvRow);
      else narrowIn(m, target, at(block, l, from), (pos - from) * kvDim, from * kvRow);
    }
  }
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

// A block of count tokens at pos through the layers, its keys and values into plan.staging where wanted() still says
// so. timing (T148, timeBlocks): rows and angles of zeros instead of forward.ts's, and nothing written back
async function block(m: Model, count: number, pos: number, wanted: () => boolean, timing = false, tokens?: number[]) {
  const { device, plan } = m, half = plan.headSize / 2;
  // (T378.4: this module's record taken apart once: the layers' loop below reads locals. The cache after grow())
  const { norm, headNorm, add, rope, activation, quantize, x, angles, step, rows, turns, ids, activationGroup, quantizeXb, quantizeAttention,
    quantizeFfn, quantizeGate, embedRows, embedGroups, layers } = m.block;
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
  } else {
    for (let t = 0; t < count; t++) {
      turns.set(floats(plan.cos + (pos + t) * half * 4, half), t * plan.headSize);
      turns.set(floats(plan.sin + (pos + t) * half * 4, half), t * plan.headSize + half);
    }
  }
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  if (!plan.direct) device.queue.writeBuffer(x, 0, rows, 0, count * plan.dim);
  device.queue.writeBuffer(angles, 0, turns, 0, count * plan.headSize);
  device.queue.writeBuffer(step, 0, new Uint32Array([count, pos, 0, 0]));
  roomFor(m, pos + count);
  const { cache } = m.block;
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(), form = m.forms.form!;
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
    multiplied(layer.q);
    multiplied(layer.k);
    multiplied(layer.v);
    // T153: as the CPU has it (forward.ts), the biases, then the norms of the heads, then RoPE
    layer.biases.forEach(biased);
    for (const head of layer.headNorms) dispatch(pass, headNorm, head.group, head.heads, count);
    dispatch(pass, rope, cache!.rope[l], count);
    if (l === plan.layers - 1) break;  // the keys and values are all a prompt's token leaves
    // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, before o adds to it
    if (plan.parallel) dispatch(pass, norm, layer.ffnNorm, 1, count);
    dispatch(pass, flash.pipeline, cache!.attention[l], plan.heads * Math.ceil(count / FLASH_Q_TILE));
    quantized(quantizeAttention);
    multiplied(layer.o);
    biased(layer.oBias);
    if (!plan.parallel) dispatch(pass, norm, layer.ffnNorm, 1, count);
    quantized(quantizeFfn);
    multiplied(layer.gate);
    if (layer.up) multiplied(layer.up);
    biased(layer.gateBias);
    dispatch(pass, activation, activationGroup, Math.ceil(plan.hidden / 64), count);
    quantized(quantizeGate);
    multiplied(layer.down);
    biased(layer.downBias);
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
// into the block's readback as plan.staging lays them out
function keysOut(m: Model, encoder: GPUCommandEncoder, pos: number, count: number) {
  const { plan } = m, { cache, readback } = m.block, B = plan.batch, row = plan.kvHeads * plan.headSize * 2;
  for (let l = 0; l < plan.layers; l++) {
    encoder.copyBufferToBuffer(cache!.keys[l], pos * row, readback, l * B * row, count * row);
    encoder.copyBufferToBuffer(cache!.values[l], pos * row, readback, (plan.layers + l) * B * row, count * row);
  }
}
// and from the block's readback into plan.staging, where wanted() still says so
async function keysBack(m: Model, wanted: () => boolean) {
  const { plan } = m, { readback } = m.block;
  await readback.mapAsync(MAP_READ);
  try {
    const bytes = 2 * plan.layers * plan.batch * plan.kvHeads * plan.headSize * 2;
    if (wanted()) new Uint8Array(m.memory.buffer, plan.staging, bytes).set(new Uint8Array(readback.getMappedRange(), 0, bytes));
  } finally {
    readback.unmap();
  }
}

export { prepare, bindLayers, grow, roomFor, keysIn, BLOCK_ROUNDS, timeBlocks, block, keysOut, keysBack };
export type { Block, Quantizing, Kept };
