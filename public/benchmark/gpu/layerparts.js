// One layer of a token on the GPU (T150, T175, T224): its shape, the forms it runs in (separate steps, fused, DP4A), their
// shaders and pipelines, and the buffers and bind groups of one (layerParts()). The layer's check, its times and the
// generated tokens all build on these.
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)
const { shared, matrixBytes, STORAGE, COPY_DST, COPY_SRC, UNIFORM, buffer, fill, floats, pipelinesFor, validated } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { toHalf } = await import(new URL(`halves.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T150: one layer of a generated token (Llama 3.2 1B's width, at position LAYER_POS), as its fourteen separate
// steps (RMSNorm, q, k, v, RoPE and the cache, the attention, o, the residual add, RMSNorm, gate, up, SwiGLU, down, the
// residual add: the prompt's shaders and T149's matrix × vector), fused into five (shaders.js's fusedMatVec: q, k
// and v with the norm and RoPE, the attention, o with the add, gate and up with the norm and SwiGLU, down with the
// add), each with the workgroup's reduction and, where subgroups are, with subgroupAdd; and T175's forms on DP4A
// below (LAYER_KINDS). All read the same weights: a layer's four matrices (q, k and v one after the other; o; gate and up; down), the separate
// steps a range of rows of them each. Timed as the matrix × vector is (T149), n layers a submission, each on the next
// copy of the weights (copies that make MATVEC_BYTES, so that a layer is not read from the GPU's caches), a submission
// of 2n less one of n, but the forms in turn (interleaved: T150's review, as T147's timeForms), so that a device that
// warms up and slows down does so for all of them alike. The attention is the prompt's (flashTile, f32 in the
// workgroup's memory and no subgroups, the same in every form: its cost is in every row alike).
const LAYER_POS = 127, LAYER_MOST = 4096, EPS = 1e-5, THETA = 500000;
const layerShape = ({ dim, hidden, heads, kvHeads }) => {
  const headSize = dim / heads, kvDim = headSize * kvHeads;
  return { dim, hidden, heads, kvHeads, headSize, kvDim,
    matrices: { qkv: [dim + 2 * kvDim, dim], o: [dim, dim], gateUp: [2 * hidden, dim], down: [dim, hidden] } };
};
// T175: the same on ONNX Runtime's DP4A for small M (the matrix × vector that read Llama 3.2 1B's w1 at 96.8% of the
// buffer's reads on the owner's Android, T149, where llama.cpp's read a layer at 18.5%, T150): its vector quantized
// before each matrix (QUANTIZE, four a layer), as separate steps (eighteen dispatches), fused but for the norms (eleven:
// the matrices with T150's writes, RMSNORM and QUANTIZE apart) and fused (nine: the norm with the quantizer,
// NORM_QUANTIZE). Where a device has no packed int8 dot, those rows say so. llama.cpp's rows stay beside them.
// base: whose matrix × vector; fused: the matrices with T150's writes (fusedMatVec, fusedDp4aMatVec); normApart: the
// norms dispatches of their own. T150's llama.cpp form with the norms apart runs only where DP4A cannot (withoutDp4a:
// T175's review): there it is the layer a device without the packed int8 dot chooses from (the owner's Android on
// llama.cpp's matrix: separate 9.27 ms, the norms apart 9.92, fused 10.61, T150); elsewhere the DP4A one stands for what
// folding the norm saves.
const LAYER_KINDS = [{ name: "llama.cpp, separate steps", base: "llama.cpp", fused: false },
  { name: "llama.cpp, fused (T150), the norms apart", base: "llama.cpp", fused: true, normApart: true, withoutDp4a: true },
  { name: "llama.cpp, fused (T150)", base: "llama.cpp", fused: true },
  { name: "DP4A, separate steps", base: "DP4A", dp4a: true, fused: false },
  { name: "DP4A, fused (T175), the norms apart", base: "DP4A", dp4a: true, fused: true, normApart: true },
  { name: "DP4A, fused (T175)", base: "DP4A", dp4a: true, fused: true }];
// T224: each fused form the engine has (not withoutDp4a) twice, with the prompt's tiles for its attention and with
// llama.cpp's decode form, flash_attn_vec (vecAttention()): the engine chooses the attention of a token on the device
const layerForms = () => {
  const subgroups = hasSubgroupId();
  const llama = LAYER_KINDS.filter((kind) => !kind.dp4a && !(kind.withoutDp4a && shared.packed)), dp4a = LAYER_KINDS.filter((kind) => kind.dp4a);
  const vec = vecAttention();
  const withVec = (form) => (form.fused && !form.withoutDp4a ? [form, { ...form, name: `${form.name}, ${vec.name}`, attention: "vec", vecSubgroups: vec.subgroups }] : [form]);
  return [...(subgroups ? [false, true] : [false]).flatMap((withSubgroups) => llama.map((kind) =>
    ({ ...kind, name: `${kind.name}${withSubgroups ? ", subgroups" : ""}`, subgroups: withSubgroups }))),
    ...dp4a.map((kind) => ({ ...kind, subgroups: false, ...(shared.packed ? {} : { none: "no packed int8 dot here" }) }))].flatMap(withVec);
};
// whether this device's shaders may use subgroups and subgroup_id
const hasSubgroupId = () => shared.device.features.has("subgroups") && (navigator.gpu.wgslLanguageFeatures?.has("subgroup_id") ?? false);
// T224's review: the prompt's tiles for a head of headSize as the engine makes them on this device (public/gpu.js's
// chooseAttention: f16 in the workgroup's memory where there is shader-f16, subgroups where there are and subgroup_id,
// the first it tries), which it chooses a token's attention against; the layer rows' tiles are f32 without subgroups
// (T150), which the engine makes only where the first is not here. { name, shape }, or null where the two are one
function engineTiles(headSize) {
  const half = shared.device.features.has("shader-f16"), subgroups = hasSubgroupId();
  if (!half && !subgroups) return null;
  const shape = shared.WGSL.flashShape({ headSize, half, subgroups, memory: shared.device.limits.maxComputeWorkgroupStorageSize,
    threads: Math.min(shared.device.limits.maxComputeInvocationsPerWorkgroup, shared.device.limits.maxComputeWorkgroupSizeX),
    subgroupMin: shared.adapter.info?.subgroupMinSize, subgroupMax: shared.adapter.info?.subgroupMaxSize });
  return shape.none ? null : { shape, name: `the prompt's tiles${half ? ", f16" : ""}${subgroups ? ", subgroups" : ""} (the engine's here)` };
}
// T224: the attention of a token by llama.cpp's flash_attn_vec (shaders.js's flashVec and flashVecReduce), with
// subgroups where there are, else with the lanes of the workgroup standing for a subgroup; shape(headSize): its shape
const vecAttention = (subgroups = hasSubgroupId()) => ({ subgroups, name: `flash_attn_vec${subgroups ? " (subgroups)" : ""}`,
  shape: (headSize) => shared.WGSL.flashVecShape({ headSize, subgroups, threads: Math.min(shared.device.limits.maxComputeInvocationsPerWorkgroup, shared.device.limits.maxComputeWorkgroupSizeX),
    subgroupMin: shared.adapter.info?.subgroupMinSize, subgroupMax: shared.adapter.info?.subgroupMaxSize }) });
const DP4A_TOKEN = "DP4A, fused (T175)";
const tokenForm = () => {
  const dp4a = shared.packed && shared.layerVerdicts?.[`a layer, ${DP4A_TOKEN}`]?.ok === true;
  return { ...LAYER_KINDS.find((kind) => (dp4a ? kind.name === DP4A_TOKEN : kind.name === "llama.cpp, fused (T150)")), subgroups: false };
};
// what a form of the layer dispatches besides the attention, [key, WGSL] each (layer() and generate())
const layerCodes = ({ dp4a, fused, normApart, subgroups }) => {
  if (!dp4a) {
    const input = normApart ? "plain" : "norm";
    return fused ? [...(normApart ? [["norm", shared.WGSL.RMSNORM]] : []), ["qkv", shared.WGSL.fusedMatVec({ input, output: "rope", subgroups })],
      ["add", shared.WGSL.fusedMatVec({ input: "plain", output: "add", subgroups })], ["glu", shared.WGSL.fusedMatVec({ input, output: "swiglu", subgroups })]]
      : [["norm", shared.WGSL.RMSNORM], ["rope", shared.WGSL.ROPE], ["swiglu", shared.WGSL.SWIGLU], ["product", shared.WGSL.mulMatVec({ packed: false, subgroups })]];
  }
  if (!fused) return [["quantize", shared.WGSL.QUANTIZE], ["norm", shared.WGSL.RMSNORM], ["rope", shared.WGSL.ROPE], ["swiglu", shared.WGSL.SWIGLU], ["product", shared.WGSL.ortDp4aMatVec]];
  return [["quantize", shared.WGSL.QUANTIZE], normApart ? ["norm", shared.WGSL.RMSNORM] : ["normQuantize", shared.WGSL.NORM_QUANTIZE],
    ["qkv", shared.WGSL.fusedDp4aMatVec({ output: "rope" })], ["add", shared.WGSL.fusedDp4aMatVec({ output: "add" })], ["glu", shared.WGSL.fusedDp4aMatVec({ output: "swiglu" })]];
};
// T175: a vector of n values quantized (QUANTIZE) into { xq, xs }: a thread a group of 32
const quantizing = (pipes, group, x, into, uniform, n, step) => [pipes.quantize, group(pipes.quantize, [[0, x], [1, into.xq], [2, into.xs], [3, uniform], [4, step]]), Math.ceil(n / shared.GROUP / 64), 1];
// The dispatches of one layer of a token in a fused form (layer() and generate()): T150's five (fusedMatVec), or on
// DP4A (T175) the norm and its quantizing (NORM_QUANTIZE, or with the norms apart RMSNORM and QUANTIZE), q, k and v
// with RoPE and the cache, the attention, its output quantized, o with the add, the norm and its quantizing, gate and up
// with SwiGLU, its output quantized, down with the add. m: the layer's matrices ({w, s, rows} each); cache: its keys and
// values; v: the vectors (h, xb, q, att, g, the norms' weights, the angles, and quantized: the { xq, xs } of each
// quantization, q, k and v's, o's, gate and up's, down's); u: the uniforms (step, flash, the matrices' Params, the norms'
// attentionNorm and ffnNorm, and quantize: QUANTIZE's of dim and of hidden); group(pipeline, entries): a bind group
function fusedLayer(form, pipes, { dim, hidden, heads }, m, cache, v, u, group) {
  const attention = attentionSteps(form, pipes, heads, cache, v, u, group);
  const rope = [[5, v.q], [6, cache.keys], [7, cache.values], [8, v.angles], [9, u.step]];
  // RMSNORM of the stream into xb (the forms with the norms apart)
  const norm = (params) => named("the norm (RMSNORM)", "small", [pipes.norm, group(pipes.norm, [[0, v.h], [1, v.norms], [2, v.xb], [3, params], [4, u.step]]), 1, 1]);
  if (!form.dp4a) {
    const groups = (rows) => Math.ceil(rows / shared.WGSL.MUL_MAT_VEC_ROWS);
    // the matrices read the stream and norm it, or (the norms apart) read xb
    const input = form.normApart ? [[2, v.xb]] : [[2, v.h], [4, v.norms]];
    const withNorm = form.normApart ? "" : " the norm,";
    return [...(form.normApart ? [norm(u.attentionNorm)] : []),
      named(`q, k and v with${withNorm} RoPE and the cache`, "matrix", [pipes.qkv, group(pipes.qkv, [[0, m.qkv.w], [1, m.qkv.s], ...input, [3, u.qkv], ...rope]), groups(m.qkv.rows), 1], "qkv"),
      ...attention,
      named("o with the residual's add", "matrix", [pipes.add, group(pipes.add, [[0, m.o.w], [1, m.o.s], [2, v.att], [3, u.o], [5, v.h]]), groups(dim), 1], "o"),
      ...(form.normApart ? [norm(u.ffnNorm)] : []),
      named(`gate and up with${withNorm} SwiGLU`, "matrix", [pipes.glu, group(pipes.glu, [[0, m.gateUp.w], [1, m.gateUp.s], ...input, [3, u.gateUp], [5, v.g]]), groups(hidden), 1], "gateUp"),
      named("down with the residual's add", "matrix", [pipes.add, group(pipes.add, [[0, m.down.w], [1, m.down.s], [2, v.g], [3, u.down], [5, v.h]]), groups(dim), 1], "down")];
  }
  const [qkvIn, oIn, gluIn, downIn] = v.quantized;
  const quantized = (x, into, params, n) => named(`a vector of ${n} quantized (QUANTIZE)`, "small", quantizing(pipes, group, x, into, params, n, u.step));
  const normed = (params, into) => (form.normApart
    ? [norm(params), quantized(v.xb, into, u.quantize[0], dim)]
    : [named("the norm with its quantizing (NORM_QUANTIZE)", "small", [pipes.normQuantize, group(pipes.normQuantize, [[0, v.h], [1, v.norms], [2, into.xq], [3, into.xs], [4, params], [5, u.step]]), 1, 1])]);
  // a matrix by its quantized input: rows its workgroups take (gate's, where up's go beside them)
  const product = (step, matrix, pipeline, { w, s }, into, params, rows, output) => named(step, "matrix", [pipeline, group(pipeline, [[0, w], [1, s], [2, into.xq], [3, params], [4, into.xs], ...output]),
    Math.ceil(rows / shared.WGSL.ORT_DP4A_MATVEC_ROWS), 1], matrix);
  return [...normed(u.attentionNorm, qkvIn), product("q, k and v with RoPE and the cache", "qkv", pipes.qkv, m.qkv, qkvIn, u.qkv, m.qkv.rows, rope),
    ...attention, quantized(v.att, oIn, u.quantize[0], dim), product("o with the residual's add", "o", pipes.add, m.o, oIn, u.o, dim, [[5, v.h]]),
    ...normed(u.ffnNorm, gluIn), product("gate and up with SwiGLU", "gateUp", pipes.glu, m.gateUp, gluIn, u.gateUp, hidden, [[5, v.g]]),
    quantized(v.g, downIn, u.quantize[1], hidden), product("down with the residual's add", "down", pipes.add, m.down, downIn, u.down, dim, [[5, v.h]])];
}
// the attention of a fused layer on cache ({ keys, values }: buffers, or T208's ranges of one) for a token that reads
// u.positions: flash attention's tile, or (T224, form.attention "vec") flash_attn_vec's parts of the positions, nwg a
// head, and where there is more than one their reduce (pipes.vecShape: its shape; v.parts, u.vecParams(nwg))
function attentionSteps(form, pipes, heads, cache, v, u, group) {
  if (form.attention !== "vec") {
    return [named("the attention (flash attention's tile)", "attention", [pipes.flash, group(pipes.flash, [[0, v.q], [1, cache.keys], [2, cache.values], [3, v.att], [4, u.flash], [5, u.step]]), heads, 1])];
  }
  const nwg = shared.WGSL.flashVecSplits(pipes.vecShape, u.positions), params = u.vecParams(nwg), which = form.vecSubgroups ? ", subgroups" : "";
  return [named(`the attention (flash_attn_vec${which}: ${nwg} parts of the positions a head)`, "attention",
    [pipes.vec, group(pipes.vec, [[0, v.q], [1, cache.keys], [2, cache.values], [3, v.parts], [4, v.att], [5, params], [6, u.step]]), heads * nwg, 1]),
  ...(nwg > 1 ? [named(`the attention's parts reduced (flash_attn_vec${which})`, "attention", [pipes.vecReduce, group(pipes.vecReduce, [[0, v.parts], [1, v.att], [2, params]]), heads, 1])] : [])];
}
// T202: a dispatch of a fused layer with what it is (run() reads only its first five): step, its name in the steps'
// table (the same name where two dispatches cost the same: the two norms, the quantizing of a vector of dim); kind,
// "matrix" (with what is folded into its write), "attention" or "small"; matrix, the layer's matrix it reads
function named(step, kind, dispatch, matrix) {
  return Object.assign(dispatch, { step, kind, ...(matrix ? { matrix } : {}) });
}
// the check's verdict of a form, by name
const layerCheck = (form) => `a layer, ${form.name}`;
// a pipeline of its WGSL, made once (in a validation scope: a shader this device refuses rejects there)
const layerPipelines = new Map();
async function compiled(code) {
  if (!layerPipelines.has(code)) {
    layerPipelines.set(code, validated(() => shared.device.createComputePipelineAsync({ layout: "auto",
      compute: { module: shared.device.createShaderModule({ code }), entryPoint: "main" } })));
  }
  return layerPipelines.get(code);
}
// the pipelines a form of the layer runs (what layerCodes() says, the attention, and SMALL for the separate steps' adds)
async function layerPipes(shape, form) {
  const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = shared.device.limits;
  const flash = shared.WGSL.flashShape({ headSize: shape.headSize, half: false, subgroups: false, memory,
    threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) });
  if (flash.none) throw new Error(flash.none);
  // one at a time: each in an error scope of its own
  const pipes = { small: pipelinesFor().small };
  for (const [key, code] of [["flash", shared.WGSL.flashTile(flash)], ...layerCodes(form)]) pipes[key] = await compiled(code);
  // T224: flash_attn_vec and its reduce, where the form's attention is it
  if (form.attention === "vec") {
    pipes.vecShape = vecAttention(form.vecSubgroups).shape(shape.headSize);
    if (pipes.vecShape.none) throw new Error(pipes.vecShape.none);
    pipes.vec = await compiled(shared.WGSL.flashVec(pipes.vecShape));
    pipes.vecReduce = await compiled(shared.WGSL.flashVecReduce(pipes.vecShape));
  }
  return pipes;
}
// A layer's buffers: copies of its four matrices (data: the check's weights, else random), the vectors, the norms'
// weights, the angles of the position, the cache (positions up to pos), and the uniforms. owned: where they go to be
// destroyed. dispatches(form, pipes, copy): one layer's, [pipeline, bind group, x, y] each. reach (T208, the steps'
// table): the bytes every matrix's ranges (ranges()) make at the least, spare buffers of random weights beside the
// copies where they fall short, and the attention's caches (caches()) as many.
function layerParts(shape, pos, copies, owned, data, reach = 0) {
  const { dim, hidden, heads, kvHeads, headSize, kvDim } = shape;
  const make = (bytes, usage = STORAGE | COPY_DST | COPY_SRC) => {
    const b = buffer(bytes, usage);
    owned.push(b);
    return b;
  };
  const uniform = (bytes) => {
    const b = make(bytes.byteLength, UNIFORM | COPY_DST);
    shared.device.queue.writeBuffer(b, 0, bytes);
    return b;
  };
  // a matrix's weights and scales, the check's (given) or random
  const matrixOf = (key, [rows, n], given) => {
    if (rows * n > shared.device.limits.maxStorageBufferBindingSize) throw new Error(`${key} is past a binding of this device`);
    const w = make(rows * n), s = make((rows * n / shared.GROUP) * 4);
    if (given) {
      shared.device.queue.writeBuffer(w, 0, given.w);
      shared.device.queue.writeBuffer(s, 0, given.s);
    } else {
      fill(w, rows * n);
      shared.device.queue.writeBuffer(s, 0, floats(rows * n / shared.GROUP, 0.002));
    }
    return { w, s, rows, n };
  };
  const copiesOf = [];
  for (let c = 0; c < copies; c++) {
    copiesOf.push(Object.fromEntries(Object.entries(shape.matrices).map(([key, size]) => [key, matrixOf(key, size, data?.[key])])));
  }
  const cacheBytes = (pos + 1) * kvDim * 2;
  const v = { h: make(dim * 4), xb: make(dim * 4), q: make(dim * 4), k: make(kvDim * 4), v: make(kvDim * 4), att: make(dim * 4),
    t: make(dim * 4), g: make(hidden * 4), u: make(hidden * 4),
    // T202: where a matrix alone writes (gate and up's rows, the most) and a one-workgroup dispatch adds, read by nothing
    scratch: make(2 * hidden * 4), norms: make(2 * dim * 4), angles: make((pos + 1) * headSize * 4),
    keys: make(cacheBytes), values: make(cacheBytes),
    // T175: the DP4A forms' four quantized vectors, each its own (the check reads every one back)
    quantized: [dim, dim, dim, hidden].map((n) => ({ xq: make(n), xs: make((n / shared.GROUP) * 4) })),
    // T224: flash_attn_vec's parts, as many as its shape with or without subgroups takes a head at the most
    parts: make(Math.max(...[true, false].map((sub) => shared.WGSL.flashVecPartsBytes(vecAttention(sub).shape(headSize), heads)))) };
  const eps = data?.eps ?? EPS;
  // RoPE's table on the GPU, a row a position up to pos (T151: fusedMatVec reads the Step's row; ROPE, the prompt's,
  // takes the rows of its block's positions, here the row at pos bound on its own: a row of headSize 64 is 256 bytes,
  // the alignment of a binding's offset)
  shared.device.queue.writeBuffer(v.angles, 0, ropeTable(headSize, pos + 1));
  const angleRow = { buffer: v.angles, offset: pos * headSize * 4, size: headSize * 4 };
  // the state a layer starts from: the residual stream, the norms' weights, the cache of the positions before
  const reset = (state) => {
    shared.device.queue.writeBuffer(v.h, 0, state.h);
    shared.device.queue.writeBuffer(v.norms, 0, state.norms);
    shared.device.queue.writeBuffer(v.keys, 0, state.keys);
    shared.device.queue.writeBuffer(v.values, 0, state.values);
  };
  const start = data ?? layerState(shape, pos);
  reset(start);
  const step = uniform(new Uint32Array([1, pos, 0, 0]));
  const normParams = (at) => {
    const bytes = new ArrayBuffer(16);
    new Uint32Array(bytes, 0, 2).set([dim, at]);
    new Float32Array(bytes, 8, 1)[0] = eps;
    return uniform(new Uint8Array(bytes));
  };
  const flashParams = new ArrayBuffer(16);
  new Uint32Array(flashParams, 0, 2).set([heads, kvHeads]);
  new Float32Array(flashParams, 8, 1)[0] = 1 / Math.sqrt(headSize);
  // fusedMatVec's Params: rows, words, perRow, second, eps, normAt, qRows, kvRows, headSize, turned (T151: the
  // position is the Step's)
  const fusedParams = (rows, n, second = 0, normAt = 0) => {
    const bytes = new ArrayBuffer(48);
    new Uint32Array(bytes).set([rows, n / 4, n / shared.GROUP, second, 0, normAt, dim, kvDim, headSize, headSize, 0, 0]);
    new Float32Array(bytes, 16, 1)[0] = eps;
    return uniform(new Uint8Array(bytes));
  };
  // T224: flash_attn_vec's Params a count of parts, made as the forms ask for them
  const vecParams = new Map();
  const u = { step, positions: pos + 1, attentionNorm: normParams(0), ffnNorm: normParams(dim), rope: uniform(new Uint32Array([heads, kvHeads, headSize, headSize])),
    vecParams: (nwg) => vecParams.get(nwg) ?? vecParams.set(nwg, uniform(shared.WGSL.flashVecParams({ headSize }, heads, kvHeads, nwg))).get(nwg),
    flash: uniform(new Uint8Array(flashParams)), swiglu: uniform(new Uint32Array([hidden, 0, 0, 0])),
    qkv: fusedParams(dim + 2 * kvDim, dim), o: fusedParams(dim, dim), gateUp: fusedParams(hidden, dim, hidden, dim), down: fusedParams(dim, hidden),
    // QUANTIZE's (n, xStride) of dim and of hidden
    quantize: [dim, hidden].map((n) => uniform(new Uint32Array([n, n, 0, 0]))) };
  // the matrix × vector's Shape (rows, words, perRow, first) of each range the separate steps read, made once
  const shapes = new Map();
  const shapeOf = (rows, n) => {
    const key = `${rows},${n}`;
    if (!shapes.has(key)) shapes.set(key, uniform(new Uint32Array([rows, n / 4, n / shared.GROUP, 0])));
    return shapes.get(key);
  };
  const group = (pipeline, entries) => shared.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: entries.map(([binding, resource]) => ({ binding, resource: "offset" in resource ? resource : { buffer: resource } })) });
  // over (T202's review): a fused form's matrices bound elsewhere, { key: a range of ranges(key) }
  const dispatches = (form, pipes, copy, over = {}) => {
    const m = copiesOf[copy];
    if (form.fused) return fusedLayer(form, pipes, shape, { ...m, ...over }, v, v, u, group);
    const attention = [pipes.flash, group(pipes.flash, [[0, v.q], [1, v.keys], [2, v.values], [3, v.att], [4, u.flash], [5, u.step]]), heads, 1];
    const norm = (params) => [pipes.norm, group(pipes.norm, [[0, v.h], [1, v.norms], [2, v.xb], [3, params], [4, u.step]]), 1, 1];
    // the separate steps: rows first to first + rows of a matrix, x into y (on DP4A, x's quantizing into)
    const [qkvIn, oIn, gluIn, downIn] = v.quantized;
    const product = ({ w, s, n }, first, rows, x, y, into) => [pipes.product, group(pipes.product, [
      [0, { buffer: w, offset: first * n, size: rows * n }], [1, { buffer: s, offset: first * n / 8, size: rows * n / 8 }],
      [2, form.dp4a ? into.xq : x], [3, y], [4, shapeOf(rows, n)], ...(form.dp4a ? [[5, into.xs]] : [])]),
      Math.ceil(rows / (form.dp4a ? shared.WGSL.ORT_DP4A_MATVEC_ROWS : shared.WGSL.MUL_MAT_VEC_ROWS)), 1];
    const quantize = (x, into, n) => (form.dp4a ? [quantizing(pipes, group, x, into, u.quantize[n === dim ? 0 : 1], n, u.step)] : []);
    const add = [pipes.small, group(pipes.small, [[0, v.t], [1, v.h]]), 1, 1];
    return [norm(u.attentionNorm), ...quantize(v.xb, qkvIn, dim), product(m.qkv, 0, dim, v.xb, v.q, qkvIn),
      product(m.qkv, dim, kvDim, v.xb, v.k, qkvIn), product(m.qkv, dim + kvDim, kvDim, v.xb, v.v, qkvIn),
      [pipes.rope, group(pipes.rope, [[0, v.q], [1, v.k], [2, v.v], [3, v.keys], [4, v.values], [5, angleRow], [6, u.rope], [7, u.step]]), 1, 1],
      attention, ...quantize(v.att, oIn, dim), product(m.o, 0, dim, v.att, v.t, oIn), add, norm(u.ffnNorm), ...quantize(v.xb, gluIn, dim),
      product(m.gateUp, 0, hidden, v.xb, v.g, gluIn), product(m.gateUp, hidden, hidden, v.xb, v.u, gluIn),
      [pipes.swiglu, group(pipes.swiglu, [[0, v.g], [1, v.u], [2, u.swiglu], [3, u.step]]), Math.ceil(hidden / 64), 1],
      ...quantize(v.g, downIn, hidden), product(m.down, 0, dim, v.g, v.t, downIn), add];
  };
  // T202's review: every range of the copies' matrix buffers that the layer's matrix key can be bound to, the copies
  // one after another. The weights are random bytes, so a matrix read from any range of its size costs the same; a
  // step timed alone on one matrix goes through all of them before it reads one again, as a layer reads each of its
  // matrices once in a round of its copies (T149's review: a matrix read again soon is read from the GPU's caches, and
  // o's two copies alone would be 9.4 MB a round, q, k and v's 14.2, down's 37.7). A range starts on a binding's
  // alignment, 256 bytes, its scales an eighth of the way in (so 2048 bytes of the weights)
  // T208: spare matrices (the largest's size, random) where the copies' ranges make fewer than reach bytes of a matrix:
  // T202 left gate and up two ranges of Llama 3.2 1B's, 75.5 MB a round (the copies' gate and up buffers alone), short
  // of T149's 128 MiB; two spares make it four, 151 MB, and the other matrices' rounds longer
  const spares = [];
  const ranges = (key) => {
    const [rows, n] = shape.matrices[key], bytes = rows * n;
    return [...copiesOf.flatMap((one) => Object.values(one)), ...spares].flatMap((matrix) =>
      [...Array(Math.floor((matrix.rows * matrix.n) / bytes))].map((_, k) => k * bytes).filter((at) => at % 2048 === 0)
        .map((at) => ({ w: { buffer: matrix.w, offset: at, size: bytes }, s: { buffer: matrix.s, offset: at / 8, size: bytes / 8 }, rows, n })));
  };
  const largest = Object.entries(shape.matrices).sort(([, a], [, b]) => b[0] * b[1] - a[0] * a[1])[0];
  while (Object.entries(shape.matrices).some(([key, size]) => ranges(key).length * matrixBytes(size) < reach)) spares.push(matrixOf(...largest));
  // T208: the attention's caches for the step timed alone, each a copy of the cache the layer starts from, in one
  // buffer (a cache's keys, then its values, each on a binding's alignment), as many as make reach bytes: T202 read the
  // one cache of 256 KB again and again, from the GPU's caches, where a token's 16 layers each read their own after
  // the other layers' weights. Made when first asked for (the steps' table only)
  let cacheRanges;
  const caches = () => {
    if (cacheRanges) return cacheRanges;
    const stride = Math.ceil(cacheBytes / 256) * 256, count = Math.max(1, Math.ceil(reach / (2 * stride)));
    const all = make(2 * stride * count, STORAGE | COPY_DST);
    const encoder = shared.device.createCommandEncoder();
    cacheRanges = [...Array(count)].map((_, i) => {
      encoder.copyBufferToBuffer(v.keys, 0, all, 2 * i * stride, cacheBytes);
      encoder.copyBufferToBuffer(v.values, 0, all, (2 * i + 1) * stride, cacheBytes);
      return { keys: { buffer: all, offset: 2 * i * stride, size: cacheBytes }, values: { buffer: all, offset: (2 * i + 1) * stride, size: cacheBytes } };
    });
    shared.device.queue.submit([encoder.finish()]);
    return cacheRanges;
  };
  // T208: the attention alone on a cache of caches() (T224: a form's, its dispatches)
  const attention = (form, pipes, cache) => attentionSteps(form, pipes, heads, cache, v, u, group);
  // T202: a matrix of the layer alone on a range of ranges(key), into the scratch buffer (no RoPE, cache, add or
  // SwiGLU after it): its plain matrix × vector (pipeline: mulMatVec, or ortDp4aMatVec reading the quantized vector
  // the fused form reads), one dispatch over all its rows
  const alone = (form, pipeline, key, { w, s, rows, n }) => {
    const input = v.quantized[MATRIX_KEYS.indexOf(key)];
    return named(`${MATRIX_NAMES[key]} alone`, "alone", [pipeline, group(pipeline, [
      [0, w], [1, s], [2, form.dp4a ? input.xq : key === "down" ? v.g : v.xb], [3, v.scratch], [4, shapeOf(rows, n)], ...(form.dp4a ? [[5, input.xs]] : [])]),
      Math.ceil(rows / (form.dp4a ? shared.WGSL.ORT_DP4A_MATVEC_ROWS : shared.WGSL.MUL_MAT_VEC_ROWS)), 1], key);
  };
  // T202: a dispatch of one workgroup that adds a vector of dim into the scratch buffer (SMALL): what a step costs
  // in a chain of dispatches when it does next to nothing (its launch and the barrier before the next)
  const floor = (pipeline) => named("a dispatch of one workgroup (SMALL: a vector of dim added)", "floor", [pipeline, group(pipeline, [[0, v.t], [1, v.scratch]]), 1, 1]);
  // T202: the residual stream written back as it started, before a submission (the adds write over it), so that every
  // submission starts alike. What these steps do does not depend on the values (unlike SAMPLE's work, T191's review),
  // and the stream grows only by a bounded add a layer, so this keeps the submissions alike rather than their times
  const restart = () => shared.device.queue.writeBuffer(v.h, 0, start.h);
  return { vectors: v, reset, dispatches, ranges, caches, attention, alone, floor, restart, spares: spares.length };
}
// T202: the matrices of a layer (layerShape's keys, in the order of the DP4A forms' quantized inputs, v.quantized), and
// as the steps' table names them
const MATRIX_KEYS = ["qkv", "o", "gateUp", "down"];
const MATRIX_NAMES = { qkv: "q, k and v", o: "o", gateUp: "gate and up", down: "down" };
// the cos of a position's headSize / 2 angles, then their sin (Llama 3's theta, unscaled: any angles would do)
function layerAngles(headSize, pos) {
  const angles = new Float32Array(headSize), half = headSize / 2;
  for (let i = 0; i < half; i++) {
    const angle = pos * THETA ** (-2 * i / headSize);
    angles[i] = Math.cos(angle);
    angles[half + i] = Math.sin(angle);
  }
  return angles;
}
// T151: RoPE's table of positions 0 to count - 1, a row of layerAngles() each (as the model's GPU worker would hold the
// CPU's, T152)
function ropeTable(headSize, count) {
  const table = new Float32Array(count * headSize);
  for (let p = 0; p < count; p++) table.set(layerAngles(headSize, p), p * headSize);
  return table;
}
// a made-up state: the residual stream, the norms' weights about 1, and the cache before pos of float16 pairs
function layerState({ dim, kvDim }, pos) {
  const h = floats(dim, 4), norms = new Float32Array(2 * dim).map(() => 0.5 + Math.random());
  const cache = () => new Uint16Array((pos + 1) * kvDim).map(() => toHalf((Math.random() - 0.5) * 4));
  return { h, norms, keys: cache(), values: cache() };
}

export { LAYER_POS, LAYER_MOST, EPS, layerShape, LAYER_KINDS, layerForms, hasSubgroupId, engineTiles, vecAttention,
  tokenForm, layerCodes, fusedLayer, attentionSteps, layerCheck, compiled, layerPipes, layerParts, MATRIX_KEYS,
  layerAngles, ropeTable, layerState };
