// A block of a prompt: the small steps' pipelines and a block's buffers, every layer's bind groups, the GPU's own keys
// and values, the block itself as one submission with its keys and values read back, and what a whole block takes (T148).
// (T352: a module of the model's GPU worker, public/gpu.js, which asks for it with its own ?v=<build>)

const { STORAGE, COPY_DST, COPY_SRC, MAP_READ, UNIFORM, common, within, buffer, uniform, validated, pipelineOf, bind,
  dispatch } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { tablesOf } = await import(new URL(`weights.js${new URL(import.meta.url).search}`, import.meta.url));
const { productGroups, multiply } =
  await import(new URL(`forms.js${new URL(import.meta.url).search}`, import.meta.url));

// the pipelines of a layer's small steps and the buffers of a block's activations
async function prepare(m) {
  const { plan, wgsl } = m, B = plan.batch, gated = Boolean(plan.matrices.w3);
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, widest = Math.max(plan.dim, qDim, plan.hidden);
  // T154: LayerNorm and GELU where the model has them (GPT-2, GPT-NeoX), else RMSNorm and SwiGLU
  [m.norm, m.headNorm, m.add, m.rope, m.activation, m.quantize] = await within(Promise.all([plan.layerNorm ? wgsl.LAYER_NORM : wgsl.RMSNORM,
    wgsl.HEAD_NORM, wgsl.ADD, wgsl.ROPE, gated ? wgsl.SWIGLU : wgsl.GELU, wgsl.QUANTIZE].map((code) => pipelineOf(m, code))), "the small steps' shaders");
  // a block's tokens, each array dense: token t's row at t times its width
  m.x = buffer(m, B * plan.dim * 4, STORAGE | COPY_DST);
  m.xb = buffer(m, B * Math.max(plan.dim, qDim) * 4, STORAGE | COPY_DST);
  m.q = buffer(m, B * qDim * 4);
  m.k = buffer(m, B * kvDim * 4);
  m.v = buffer(m, B * kvDim * 4);
  m.gate = buffer(m, B * plan.hidden * 4, STORAGE | COPY_DST);
  m.up = gated ? buffer(m, B * plan.hidden * 4) : null;
  // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, which xb cannot hold (the attention
  // writes there first)
  m.ffnInput = plan.parallel ? buffer(m, B * plan.dim * 4) : m.xb;
  // the packed shaders' input: the activations of a matrix quantized, 8 bits a value and a float32 scale a group
  m.xq = buffer(m, B * widest);
  m.xs = buffer(m, B * (widest / wgsl.GROUP) * 4);
  m.angles = buffer(m, B * plan.headSize * 4, STORAGE | COPY_DST);
  m.step = buffer(m, 16, UNIFORM | COPY_DST);
  m.readback = buffer(m, 2 * plan.layers * B * kvDim * 2, MAP_READ | COPY_DST);
  // what a request writes into them, from the shared memory
  m.rows = new Float32Array(B * plan.dim);
  m.turns = new Float32Array(B * plan.headSize);
  // (T255: a layer RoPE leaves alone turns none of a head, as every layer of a GPT-2)
  const ropeShape = (turned) => uniform(m, new Uint32Array([plan.heads, plan.kvHeads, plan.headSize, turned]));
  m.ropeShape = ropeShape(plan.turned);
  m.ropeAlone = plan.unturned?.length ? ropeShape(0) : null;
  const hidden = uniform(m, new Uint32Array([plan.hidden, 0, 0, 0]));
  m.activationGroup = bind(m, m.activation, gated ? [m.gate, m.up, hidden, m.step] : [m.gate, hidden, m.step]);
  // QUANTIZE of the inputs the matrices read: xb (the norm's, dim wide; the attention's, heads × headSize wide, which
  // is not dim where the heads are of another size, T153) and gate (SwiGLU's)
  const quantizing = (from, n) => ({ group: bind(m, m.quantize, [from, m.xq, m.xs, uniform(m, new Uint32Array([n, n, 0, 0])), m.step]),
    x: Math.ceil(n / wgsl.GROUP / 64) });
  m.quantizeXb = quantizing(m.xb, plan.dim);
  m.quantizeAttention = quantizing(m.xb, qDim);
  m.quantizeFfn = plan.parallel ? quantizing(m.ffnInput, plan.dim) : m.quantizeXb;
  m.quantizeGate = quantizing(m.gate, plan.hidden);
  // T210: a model on the GPU alone embeds a block's tokens here (the CPU holds no embedding): EMBED_ROWS a piece of
  // the table (T209), from the block's ids into x
  if (plan.direct) {
    // (T232: from a table of ternary weights, EMBED_ROWS_TERNARY)
    const rowsCode = tablesOf(plan).embedding.ternary ? wgsl.EMBED_ROWS_TERNARY : wgsl.EMBED_ROWS;
    m.embedRows = await within(validated(m, () => pipelineOf(m, rowsCode)), "compiling the embedding's rows");
    m.ids = buffer(m, B * 4, STORAGE | COPY_DST);
    m.embedGroups = m.tables.embedding.map((piece) => bind(m, m.embedRows,
      [piece.values, piece.scales, m.ids, m.x, uniform(m, new Uint32Array([plan.dim, piece.first, piece.rows, 0]))]));
  }
}

// every layer's bind groups but those of the cache, with the form chosen. T153: a bias (Qwen2's) added to q, k and
// v, or (Qwen3's) the norm of every head of q and of k, where the model has them. T154: LayerNorm with its bias, and a
// bias after o, w1 and w2 (GPT-2's, GPT-NeoX's)
function bindLayers(m) {
  const { plan } = m, form = m.form, V = m.vectors;
  // (a piece each, T155)
  const product = (name, l, from, to, add = false) => productGroups(m, form, name, l, from, to, add);
  m.layers = [];
  for (let l = 0; l < plan.layers; l++) {
    const norm = new ArrayBuffer(16);
    new Uint32Array(norm, 0, 2).set([plan.dim, l * plan.dim]);
    new Float32Array(norm, 8, 1)[0] = plan.eps;
    const normShape = uniform(m, norm);
    const added = (to, name) => V[name] && { group: bind(m, m.add, [to, V[name], uniform(m, new Uint32Array([plan.vectors[name].size, l * plan.vectors[name].size, 0, 0])), m.step]),
      x: Math.ceil(plan.vectors[name].size / 64) };
    // T154: LayerNorm reads its bias from the same place in its own buffer
    const normed = (weight, bias, out) => bind(m, m.norm, plan.layerNorm ? [m.x, V[weight], V[bias], out, normShape, m.step]
      : [m.x, V[weight], out, normShape, m.step]);
    const headNorm = (of, name, heads) => {
      if (!V[name]) return null;
      const shape = new ArrayBuffer(16);
      new Uint32Array(shape, 0, 2).set([plan.headSize, l * plan.headSize]);
      new Float32Array(shape, 8, 1)[0] = plan.eps;
      return { group: bind(m, m.headNorm, [of, V[name], uniform(m, shape), m.step]), heads };
    };
    m.layers.push({
      attentionNorm: normed("attention", "attentionBias", m.xb),
      q: product("wq", l, m.xb, m.q), k: product("wk", l, m.xb, m.k), v: product("wv", l, m.xb, m.v),
      biases: [added(m.q, "bq"), added(m.k, "bk"), added(m.v, "bv")].filter(Boolean),
      headNorms: [headNorm(m.q, "qNorm", plan.heads), headNorm(m.k, "kNorm", plan.kvHeads)].filter(Boolean),
      o: product("wo", l, m.xb, m.x, true), oBias: added(m.x, "bo"),
      ffnNorm: normed("ffn", "ffnBias", m.ffnInput),
      gate: product("w1", l, m.ffnInput, m.gate), gateBias: added(m.gate, "b1"),
      up: m.matrices.w3 ? product("w3", l, m.ffnInput, m.up) : null,
      down: product("w2", l, m.gate, m.x, true), downBias: added(m.x, "b2"),
    });
  }
}

// The GPU's own keys and values (float16, a pair to a u32), per layer [positions][kvDim], grown as the CPU's cache
// grows (doubling from plan.kvStart: a prompt seldom needs the whole context) and kept from block to block; with them
// the bind groups that read them
function grow(m, needed) {
  const { device, plan } = m, old = m.cache;
  const kvDim = plan.kvHeads * plan.headSize, row = kvDim * 2;
  const capacity = Math.min(Math.max(2 * (old?.capacity ?? 0), needed), plan.seqLen);
  if (capacity * row > m.limit) throw new Error(`the keys of ${capacity} positions are more than a buffer of this GPU`);
  const cache = { capacity, owned: [], keys: [], values: [], rope: [], attention: [] };
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
    cache.rope.push(bind(m, m.rope, [m.q, m.k, m.v, cache.keys[l], cache.values[l], m.angles, plan.unturned?.includes(l) ? m.ropeAlone : m.ropeShape, m.step]));
    cache.attention.push(bind(m, m.attention.pipeline, [m.q, cache.keys[l], cache.values[l], m.xb, attentionParams, m.step]));
  }
  m.cache = cache;
}

// T148: what a whole block takes here, from the submission to its keys and values read back, for 16 and for 64 tokens
// (plan.batch) at position 0 of the GPU's own cache (which holds nothing of forward.js's yet): forward.js weighs a
// block of a prompt on the GPU against the same tokens on the CPU by it (a line through the two, scaled by the blocks
// it then times itself). The two in turn, BLOCK_ROUNDS rounds after one of each to warm up (the pipelines' first
// dispatches, the weights' first reads), the median of each (a device that warms up or is loaded meanwhile falls on
// both alike); one round on a fallback adapter (tests: its times are no GPU's)
const BLOCK_ROUNDS = 3;
async function timeBlocks(m) {
  const { batch, seqLen } = m.plan, counts = [...new Set([Math.min(16, batch, seqLen), Math.min(batch, seqLen)])], times = counts.map(() => []);
  const timed = async (count) => {
    const began = performance.now();
    await block(m, count, 0, () => false, true);
    return performance.now() - began;
  };
  for (const count of counts) await timed(count);
  for (let round = 0; round < (m.fallback ? 1 : BLOCK_ROUNDS); round++) {
    for (let i = 0; i < counts.length; i++) {
      times[i].push(await timed(counts[i]));
      if (common.stopping) return [];
    }
  }
  return counts.map((count, i) => ({ count, ms: times[i].sort((a, b) => a - b)[times[i].length >> 1] }));
}

// A block of count tokens at pos through the layers, its keys and values into plan.staging where wanted() still says
// so. timing (T148, timeBlocks): rows and angles of zeros instead of forward.js's, and nothing written back
async function block(m, count, pos, wanted, timing = false, tokens) {
  const { device, plan } = m, half = plan.headSize / 2;
  // the rows forward.js embedded (dense), and the angles of their positions (the tables forward.js has); views of just
  // those floats (T155: not a view of the whole of a 64-bit memory of some GB for every block)
  const floats = (address, n) => new Float32Array(m.memory.buffer, address, n);
  // (T210: on the GPU alone, the tokens themselves, embedded here; token 0 where it is timed)
  if (plan.direct) m.device.queue.writeBuffer(m.ids, 0, Uint32Array.from({ length: count }, (_, t) => (timing ? 0 : tokens[t])));
  else if (timing) m.rows.fill(0);
  else m.rows.set(floats(plan.rows, count * plan.dim));
  // (T154: GPT-2 turns nothing and has no tables)
  if (timing || !plan.turned) {
    m.turns.fill(0);
  } else {
    for (let t = 0; t < count; t++) {
      m.turns.set(floats(plan.cos + (pos + t) * half * 4, half), t * plan.headSize);
      m.turns.set(floats(plan.sin + (pos + t) * half * 4, half), t * plan.headSize + half);
    }
  }
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  if (!plan.direct) device.queue.writeBuffer(m.x, 0, m.rows, 0, count * plan.dim);
  device.queue.writeBuffer(m.angles, 0, m.turns, 0, count * plan.headSize);
  device.queue.writeBuffer(m.step, 0, new Uint32Array([count, pos, 0, 0]));
  if (pos + count > m.cache.capacity) grow(m, pos + count);
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(), form = m.form;
  const multiplied = (pieces) => pieces.forEach((piece) => multiply(m, pass, form, piece.group, piece.rows, count));
  // a packed form reads its input quantized: once for the matrices that read the same one
  const quantize = (q) => form.packed && dispatch(pass, m.quantize, q.group, q.x, count);
  const flash = m.attention;
  // a bias added to a matrix's output (T153: Qwen2's q, k and v; T154: GPT-2's and GPT-NeoX's every matrix's)
  const biased = (bias) => bias && dispatch(pass, m.add, bias.group, bias.x, count);
  m.embedGroups?.forEach((group) => dispatch(pass, m.embedRows, group, 1, count));
  for (let l = 0; l < plan.layers; l++) {
    const layer = m.layers[l];
    dispatch(pass, m.norm, layer.attentionNorm, 1, count);
    quantize(m.quantizeXb);
    multiplied(layer.q);
    multiplied(layer.k);
    multiplied(layer.v);
    // T153: as the CPU has it (forward.js), the biases, then the norms of the heads, then RoPE
    layer.biases.forEach(biased);
    for (const norm of layer.headNorms) dispatch(pass, m.headNorm, norm.group, norm.heads, count);
    dispatch(pass, m.rope, m.cache.rope[l], count);
    if (l === plan.layers - 1) break;  // the keys and values are all a prompt's token leaves
    // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, before o adds to it
    if (plan.parallel) dispatch(pass, m.norm, layer.ffnNorm, 1, count);
    dispatch(pass, flash.pipeline, m.cache.attention[l], plan.heads * Math.ceil(count / m.wgsl.FLASH_Q_TILE));
    quantize(m.quantizeAttention);
    multiplied(layer.o);
    biased(layer.oBias);
    if (!plan.parallel) dispatch(pass, m.norm, layer.ffnNorm, 1, count);
    quantize(m.quantizeFfn);
    multiplied(layer.gate);
    if (layer.up) multiplied(layer.up);
    biased(layer.gateBias);
    dispatch(pass, m.activation, m.activationGroup, Math.ceil(plan.hidden / 64), count);
    quantize(m.quantizeGate);
    multiplied(layer.down);
    biased(layer.downBias);
  }
  pass.end();
  // (T210: a model on the GPU alone keeps them there: nothing comes back)
  if (!plan.direct) keysOut(m, encoder, pos, count);
  device.queue.submit([encoder.finish()]);
  const invalid = await device.popErrorScope(), full = await device.popErrorScope();
  if (invalid || full) throw new Error(`the GPU refused a block (${(invalid ?? full).message})`);
  if (plan.direct) await device.queue.onSubmittedWorkDone();
  else await keysBack(m, () => !timing && wanted());
}
// the keys of count positions from pos of every layer, then their values, [layers][plan.batch][kvDim] each in float16,
// into m.readback as plan.staging lays them out
function keysOut(m, encoder, pos, count) {
  const { plan } = m, B = plan.batch, row = plan.kvHeads * plan.headSize * 2;
  for (let l = 0; l < plan.layers; l++) {
    encoder.copyBufferToBuffer(m.cache.keys[l], pos * row, m.readback, l * B * row, count * row);
    encoder.copyBufferToBuffer(m.cache.values[l], pos * row, m.readback, (plan.layers + l) * B * row, count * row);
  }
}
// and from m.readback into plan.staging, where wanted() still says so
async function keysBack(m, wanted) {
  const { plan } = m;
  await m.readback.mapAsync(MAP_READ);
  try {
    const bytes = 2 * plan.layers * plan.batch * plan.kvHeads * plan.headSize * 2;
    if (wanted()) new Uint8Array(m.memory.buffer, plan.staging, bytes).set(new Uint8Array(m.readback.getMappedRange(), 0, bytes));
  } finally {
    m.readback.unmap();
  }
}

export { prepare, bindLayers, grow, BLOCK_ROUNDS, timeBlocks, block, keysOut, keysBack };
