// A generated token's steps (T152): the forms of a layer this device can make, a token's buffers and uniforms, the
// dispatches of a step, and a run of steps as one submission with its ids, state, keys and values read back.
// (T352: a module of the model's GPU worker, public/gpu.js, which asks for it with its own ?v=<build>)

const { STORAGE, COPY_DST, COPY_SRC, MAP_READ, UNIFORM, turnedAt, within, buffer, uniform, copyIn, validated,
  pipelineOf, bind, dispatch } =
  await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { tablesOf } = await import(new URL(`weights.js${new URL(import.meta.url).search}`, import.meta.url));
const { attentionPasses } =
  await import(new URL(`tokenattention.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T152: tokens on the GPU. A step of the engine's generate() (the forward pass of the token fed, the sampling of
// the next) as one compute pass of shaders.js's T151 run: EMBED of the state's token, every layer in a fused form, the
// final norm and the classifier, SAMPLE (the penalty, softmax, top-p and the draw, with the random number the CPU drew
// for the step), after which the state's first four words go to the Step uniform for the next pass (a uniform is not a
// shader's to write). plan.tokens.most steps a submission, their ids read back once (T151: the wait of a submission,
// 3 to 8.6 ms on the owner's Android, T134, is paid once for them), with the keys and values of their positions, which
// go into forward.js's cache as the CPU would have written them: the CPU may take the next step, or the next
// generation, at any position. The keys and values the GPU does not hold (positions the CPU computed since) go up from
// forward.js's cache first: float16 there as here (T110, T147).
//
// The form of a layer, from those this device can make: T150's fusedMatVec (llama.cpp's mul_mat_vec, with the norm on
// its read and RoPE, the residual's add or SwiGLU on its write) with the workgroup's reduction, the same with
// subgroupAdd where there are subgroups, and T175's fusedDp4aMatVec (ORT's DP4A for small M, the vector quantized
// before each matrix, NORM_QUANTIZE where a norm is) where there is the packed int8 dot. Each is checked against
// JavaScript on the model's own weights (checkTokens), and the right ones are timed, a run of plan.tokens.most steps
// each in turn: the fastest is taken (T150's and T175's tables left which is fastest to the device: the owner's
// Android read a layer at 18.5% of its buffer's reads with mul_mat_vec and a matrix at 96.8% with DP4A). The attention
// (T224, chooseTokenAttention) is llama.cpp's decode form, flash_attn_vec, split over the positions and reduced, or the
// prompt's tiles, whichever is right and faster here, on this pass's q. Where it came from: the run of T151 (public/benchmark/gpu/generate.js's generate()), whose
// form is WebLLM's decode loop without its sync of every token (web-llm, src/llm_chat.ts; no line taken) and llama.cpp's
// WebGPU graph of a token, one command encoder for all of it (ggml-webgpu.cpp, commit 2145525a, MIT; no line taken).
// T226: Qwen2's and Qwen3's steps too. What they have between the matrix of q, k and v and RoPE (T153: a bias of each,
// a norm of every head of q and of k) is the prompt's dispatches, ADD and HEAD_NORM, as the CPU orders them, on the
// three as their matrix wrote them, and then shaders.js's TOKEN_ROPE (the fused write's own lines, as a dispatch):
// two dispatches a layer more with the biases, three with the norms of the heads, none for a model without either.
// (T175's fused DP4A with the norms apart as well, RMSNORM and QUANTIZE where NORM_QUANTIZE is one: the owner's
// Android ran a layer so in 3.36 ms against 3.67 fused, the fastest of its table, 2026-09-27; T175's condition to
// reverse NORM_QUANTIZE, here chosen on the device)
const TOKEN_FORMS = [{ name: "llama.cpp, fused (T150)", dp4a: false, subgroups: false },
  { name: "llama.cpp, fused (T150), subgroups", dp4a: false, subgroups: true },
  { name: "DP4A, fused (T175)", dp4a: true, subgroups: false },
  { name: "DP4A, fused (T175), the norms apart", dp4a: true, subgroups: false, normApart: true },
  // T232: the same two on ternary weights (shaders.js's ternaryMatVec), which a model of ternary weights alone takes,
  // and takes no other
  { name: "DP4A, fused (T175), ternary", dp4a: true, subgroups: false, ternary: true },
  { name: "DP4A, fused (T175), ternary, the norms apart", dp4a: true, subgroups: false, ternary: true, normApart: true }];
function tokenCandidates(m) {
  const features = navigator.gpu.wgslLanguageFeatures;
  const subgroups = m.device.features.has("subgroups") && Boolean(features?.has("subgroup_id"));
  const packed = Boolean(features?.has("packed_4x8_integer_dot_product"));
  // (T226: NORM_QUANTIZE is RMSNorm's: a model with LayerNorm has its norms apart, and the form with them fused is
  // the same as the one with them apart)
  return TOKEN_FORMS.filter((form) => Boolean(form.ternary) === m.ternary && (!form.subgroups || subgroups) && (!form.dp4a || packed) &&
    (!m.plan.layerNorm || !form.dp4a || form.normApart));
}
// T226: what of the model's form shapes a token's dispatches: apart, q, k and v are written as they are for what comes
// before RoPE (the biases, the norms of the heads); gated, the FFN has a gate (SwiGLU on the write of gate and up; else
// one matrix written as it is, then its bias and GELU); floatHead, the classifier multiplies floats on a DP4A form too
// (T92's outlier channels: a few of the final norm's weights are 12 to 17 times the others (GPT-2), and a group of 32
// quantized to 8 bits with one of them loses the other 31; the CPU multiplies their columns apart, and the GPU has
// T150's matrix of floats, which needs no such thing)
// (T232: a ternary classifier is no matrix of floats: its outlier channels are taken apart as the CPU takes them,
// shaders.js's TAKE_OUTLIERS and TERNARY_COLUMNS)
const floatHead = (m) => Boolean(m.plan.tokens.outliers) && !m.ternary;
const tokenShape = (m) => ({ apart: Boolean(m.gen.qkv), gated: Boolean(m.matrices.w3), layerNorm: m.plan.layerNorm, floatHead: floatHead(m) });
// a form's WGSL, [key, code] each (the pipelines every form shares are compiled apart: tokenBuffers); the same code
// under two keys is compiled once (chooseTokens). The norm is on the read of T150's matrices where it is RMSNorm
// (LayerNorm takes the mean out first, which no sum of the matrix's rows gives: a dispatch of its own before them)
const tokenCodes = (wgsl, { dp4a, subgroups, normApart, ternary }, { apart, gated, layerNorm, floatHead }) => {
  const floats = (output) => wgsl.fusedMatVec({ input: layerNorm ? "plain" : "norm", output, subgroups });
  const matrix = dp4a ? (output) => (ternary ? wgsl.ternaryMatVec : wgsl.fusedDp4aMatVec)({ output }) : floats;
  return [...(dp4a && !normApart ? [["normQuantize", wgsl.NORM_QUANTIZE]] : []), ["qkv", matrix(apart ? "write" : "rope")],
    ["add", dp4a ? matrix("add") : wgsl.fusedMatVec({ input: "plain", output: "add", subgroups })], ["glu", matrix(gated ? "swiglu" : "write")],
    ["classifier", (floatHead ? floats : matrix)("write")]];
};

// a bind group of the bindings given ([binding, a buffer or a range of one] each: the fused shaders skip binding 4
// where the norm is not on their read)
const bindAt = (m, pipeline, entries) => m.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: entries.map(([binding, buffer]) => ({ binding, resource: buffer.buffer ? buffer : { buffer } })) });

// The buffers and uniforms of a token, and the pipelines every form shares (EMBED, SAMPLE; QUANTIZE and the attention
// are the prompt's). A token's vectors: the stream h, q, the attention's output, SwiGLU's; the logits and SAMPLE's two
// scratch arrays of the vocabulary; the state, the ids, the random numbers, the Step and the settings; the quantized
// vector (DP4A: one serves every quantization of a token, each read before the next is made: T175's (l)); what a
// submission reads back (the ids, the state, and the keys and values of its positions, [keys, values][layer][step])
async function tokenBuffers(m) {
  const { plan, wgsl } = m, most = plan.tokens.most, vocab = plan.tokens.classifier.rows;
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, widest = Math.max(plan.dim, qDim, plan.hidden);
  // (one at a time: each in an error scope of its own)
  // (T232: from a table of ternary weights, EMBED_TERNARY)
  const embedCode = tablesOf(plan).embedding.ternary ? wgsl.EMBED_TERNARY : wgsl.EMBED;
  const embed = await within(validated(m, () => pipelineOf(m, embedCode)), "compiling the embedding's row");
  const sample = await within(validated(m, () => pipelineOf(m, wgsl.SAMPLE)), "compiling the sampling");
  const out = STORAGE | COPY_SRC;
  const g = { embed, sample, most, vocab, h: buffer(m, plan.dim * 4, out | COPY_DST), q: buffer(m, qDim * 4), att: buffer(m, qDim * 4),
    gate: buffer(m, plan.hidden * 4), logits: buffer(m, vocab * 4, out), probs: buffer(m, vocab * 4), order: buffer(m, vocab * 4),
    state: buffer(m, wgsl.STATE_BYTES, out | COPY_DST), chosen: buffer(m, most * 4, out), randoms: buffer(m, most * 4, STORAGE | COPY_DST),
    step: buffer(m, 16, UNIFORM | COPY_DST), settings: buffer(m, wgsl.SAMPLING_BYTES, UNIFORM | COPY_DST),
    xq: buffer(m, widest), xs: buffer(m, (widest / wgsl.GROUP) * 4), xb: buffer(m, plan.dim * 4),
    readback: buffer(m, most * 4 + wgsl.STATE_BYTES + 2 * plan.layers * most * kvDim * 2, MAP_READ | COPY_DST) };
  // fusedMatVec's Params: rows, words, perRow, second, eps, normAt, qRows, kvRows, headSize, turned
  const params = (rows, n, second = 0, normAt = 0, turned = plan.turned) => {
    const bytes = new ArrayBuffer(48);
    new Uint32Array(bytes).set([rows, n / 4, n / wgsl.GROUP, second, 0, normAt, qDim, kvDim, plan.headSize, turned, 0, 0]);
    new Float32Array(bytes, 16, 1)[0] = plan.eps;
    return uniform(m, bytes);
  };
  // RMSNORM's, HEAD_NORM's and NORM_QUANTIZE's Norm: size, at, eps, first (HEAD_NORM's: the rows before its first)
  const norm = (at, size = plan.dim, first = 0) => {
    const bytes = new ArrayBuffer(16);
    new Uint32Array(bytes).set([size, at, 0, first]);
    new Float32Array(bytes, 8, 1)[0] = plan.eps;
    return uniform(m, bytes);
  };
  const flash = new ArrayBuffer(16);
  new Uint32Array(flash, 0, 2).set([plan.heads, plan.kvHeads]);
  new Float32Array(flash, 8, 1)[0] = 1 / Math.sqrt(plan.headSize);
  const layers = [...Array(plan.layers)].map((_, l) => l * plan.dim), qkvRows = qDim + 2 * kvDim;
  g.u = { embed: m.tables.embedding.map((piece) => uniform(m, new Uint32Array([plan.dim, piece.first, piece.rows, 0]))), flash: uniform(m, flash), o: params(plan.dim, qDim), down: params(plan.dim, plan.hidden),
    qkv: layers.map((at, l) => params(qkvRows, plan.dim, 0, at, turnedAt(plan, l))), gateUp: layers.map((at) => params(plan.hidden, plan.dim, plan.hidden, at)),
    norm: layers.map((at) => norm(at)), final: norm(0), classifier: m.tables.classifier.map((piece) => params(piece.rows, plan.dim)),
    // QUANTIZE's (n, xStride): the attention's output, SwiGLU's
    quantizeAttention: uniform(m, new Uint32Array([qDim, qDim, 0, 0])), quantizeGate: uniform(m, new Uint32Array([plan.hidden, plan.hidden, 0, 0])),
    quantizeNormed: uniform(m, new Uint32Array([plan.dim, plan.dim, 0, 0])) };
  // T226: a model whose q, k and v are not ready for RoPE as their matrix leaves them (T153: Qwen2's biases, Qwen3's
  // norms of the heads) has the matrix write the three as they are into qkv (q's rows, then k's, then v's), the
  // prompt's ADD and HEAD_NORM change them there, and TOKEN_ROPE turns them into q and the cache (tokenPass). The
  // biases of a layer are one vector then, in the rows' order (one ADD for the three); k's heads are the rows after q's
  const { bq, bk, bv, qNorm } = plan.vectors;
  if (bq || qNorm) {
    g.rope = await within(validated(m, () => pipelineOf(m, wgsl.TOKEN_ROPE)), "compiling a token's RoPE");
    g.qkv = buffer(m, qkvRows * 4);
  }
  if (bq) {
    g.qkvBias = buffer(m, plan.layers * qkvRows * 4, STORAGE | COPY_DST);
    for (let l = 0; l < plan.layers; l++) {
      let at = l * qkvRows * 4;
      for (const bias of [bq, bk, bv]) {
        copyIn(m, g.qkvBias, bias.at + l * bias.size * 4, bias.size * 4, at);
        at += bias.size * 4;
      }
    }
    g.u.qkvBias = layers.map((_, l) => uniform(m, new Uint32Array([qkvRows, l * qkvRows, 0, 0])));
  }
  if (qNorm) g.u.headNorms = layers.map((_, l) => [0, plan.heads].map((first) => norm(l * plan.headSize, plan.headSize, first)));
  // T226, GPT-2's and GPT-NeoX's (T154's dispatches of a prompt's block, on a token's vectors): ADD's shapes of the
  // biases after o, w1 and w2 (a layer's from its place in the vector) and of GPT-2's positions (a row a position), and
  // GELU's size
  const { bo, b1, b2 } = plan.vectors;
  if (bo) g.u.biases = layers.map((_, l) => Object.fromEntries([["bo", bo], ["b1", b1], ["b2", b2]].map(([name, { size }]) =>
    [name, uniform(m, new Uint32Array([size, l * size, 0, 0]))])));
  if (m.positions) g.u.positions = uniform(m, new Uint32Array([plan.dim, 0, plan.dim, 0]));
  // T232: a ternary classifier's outlier channels (plan.tokens.channels, T92): the two dispatches around its matrix
  // (shaders.js's TAKE_OUTLIERS and TERNARY_COLUMNS), the values taken, and the Outliers of every piece of the table
  const channels = m.ternary ? plan.tokens.channels ?? [] : [];
  if (channels.length) {
    g.take = await within(validated(m, () => pipelineOf(m, wgsl.TAKE_OUTLIERS)), "compiling the outlier channels' taking");
    g.columns = await within(validated(m, () => pipelineOf(m, wgsl.TERNARY_COLUMNS)), "compiling the outlier channels' columns");
    g.picked = buffer(m, wgsl.OUTLIERS_MOST * 4);
    g.u.take = uniform(m, wgsl.outliersOf(channels));
    g.u.columns = m.tables.classifier.map((piece) => uniform(m, wgsl.outliersOf(channels, piece.rows, plan.dim)));
  }
  g.u.hidden = uniform(m, new Uint32Array([plan.hidden, 0, 0, 0]));
  return g;
}

// the workgroups of rows by the fused shaders' rows a workgroup, over x and then y (they number them so)
function spread(m, rows, perGroup) {
  const groups = Math.ceil(rows / perGroup), across = Math.min(groups, m.device.limits.maxComputeWorkgroupsPerDimension);
  return [across, Math.ceil(groups / across)];
}
// The dispatches of one step in a form: EMBED, the layers (from, to: those of a check), the head (the final norm and
// the classifier) and SAMPLE, [pipeline, bind group, x, y] each. positions: how many the attention reads at the most
// (T224: flash_attn_vec's parts a head go by it)
function tokenPass(m, form, { from = 0, to = m.plan.layers, head = true, embed = true, positions = m.cache.capacity } = {}) {
  const { plan, wgsl, gen: g } = m, V = m.vectors, P = form.pipes, cache = m.cache, qDim = plan.heads * plan.headSize;
  const rows = form.dp4a ? wgsl.ORT_DP4A_MATVEC_ROWS : wgsl.MUL_MAT_VEC_ROWS;
  const matrix = (pipeline, [w, s], input, params, count, output) =>
    [pipeline, bindAt(m, pipeline, [[0, w], [1, s], ...input, [3, params], ...output]), ...spread(m, count, rows)];
  const quantize = (x, params, n) => [m.quantize, bindAt(m, m.quantize, [[0, x], [1, g.xq], [2, g.xs], [3, params], [4, g.step]]), Math.ceil(n / wgsl.GROUP / 64), 1];
  // What a matrix reads: { norm, quantize, input }, the dispatches before it (the stream's norm where it is one of its
  // own, the quantizing of DP4A's vector: T175) and its bindings of the vector.
  // The stream normed, for a matrix of floats (T150's; T226: the classifier of a model with outlier channels on DP4A
  // too) or of DP4A: RMSNorm on the read of T150's matrices, or with DP4A's quantizer (NORM_QUANTIZE); else a dispatch
  // of its own into xb (the prompt's RMSNORM, T175's form with the norms apart; T226: the prompt's LAYER_NORM with its
  // bias, T154), which the matrix reads as it is or quantized
  // (T232, taken: what changes the normed stream before it is quantized, a ternary classifier's outlier channels
  // taken out of it: the norm is then a dispatch of its own whatever the form)
  const normed = (weights, bias, params, floats = !form.dp4a, taken = null) => {
    const apart = () => [[m.norm, bind(m, m.norm, plan.layerNorm ? [g.h, weights, bias, g.xb, params, g.step] : [g.h, weights, g.xb, params, g.step]), 1, 1], ...(taken ?? [])];
    if (floats) return plan.layerNorm ? { norm: apart(), quantize: [], input: [[2, g.xb]] } : { norm: [], quantize: [], input: [[2, g.h], [4, weights]] };
    if (!form.normApart && !plan.layerNorm && !taken) {
      return { norm: [[P.normQuantize, bindAt(m, P.normQuantize, [[0, g.h], [1, weights], [2, g.xq], [3, g.xs], [4, params], [5, g.step]]), 1, 1]],
        quantize: [], input: [[2, g.xq], [4, g.xs]] };
    }
    return { norm: apart(), quantize: [quantize(g.xb, g.u.quantizeNormed, plan.dim)], input: [[2, g.xq], [4, g.xs]] };
  };
  // a vector as it is (the attention's output, the activation's)
  const plain = (x, params, n) => (form.dp4a ? { quantize: [quantize(x, params, n)], input: [[2, g.xq], [4, g.xs]] } : { quantize: [], input: [[2, x]] });
  // T226: a bias added after a matrix (ADD: Qwen2's of q, k and v as one vector; T154's after o, w1 and w2)
  const added = (to, bias, shape, n) => [m.add, bind(m, m.add, [to, bias, shape, g.step]), Math.ceil(n / 64), 1];
  const biased = (to, name, l) => (V[name] ? [added(to, V[name], g.u.biases[l][name], plan.vectors[name].size)] : []);
  // (T209: a dispatch a piece of the table, each writing the row where the token is in its rows)
  const list = !embed ? [] : m.tables.embedding.map((piece, i) =>
    [g.embed, bindAt(m, g.embed, [[0, piece.values], [1, piece.scales], [2, g.state], [3, g.h], [4, g.u.embed[i]]]), 1, 1]);
  // T226: GPT-2's learned positions: the row of the Step's position added to the embedding's (the CPU's embed())
  if (embed && m.positions) list.push(added(g.h, m.positions, g.u.positions, plan.dim));
  const qkvRows = qDim + 2 * plan.kvHeads * plan.headSize, gated = Boolean(m.matrices.w3);
  for (let l = from; l < to; l++) {
    const [o] = m.matrices.wo.pieces, [down] = m.matrices.w2.pieces;
    const attention = normed(V.attention, V.attentionBias, g.u.norm[l]), ffn = normed(V.ffn, V.ffnBias, g.u.norm[l]);
    const attended = plain(g.att, g.u.quantizeAttention, qDim), activated = plain(g.gate, g.u.quantizeGate, plan.hidden);
    // where q, k and v go turned: q, and the cache at the Step's position
    const turned = [[5, g.q], [6, cache.keys[l]], [7, cache.values[l]], [8, m.angleTable], [9, g.step]];
    list.push(...attention.norm, ...attention.quantize,
      // T226: turned on the matrix's write (Llama: one dispatch), or written as they are where something comes before
      // RoPE: as the CPU has it and a prompt's block (T153), the biases, the norms of the heads of q and of k, RoPE
      matrix(P.qkv, m.joined[l].qkv, attention.input, g.u.qkv[l], qkvRows, g.qkv ? [[5, g.qkv]] : turned),
      ...(g.qkvBias ? [added(g.qkv, g.qkvBias, g.u.qkvBias[l], qkvRows)] : []),
      ...(g.u.headNorms ? [[V.qNorm, plan.heads], [V.kNorm, plan.kvHeads]].map(([weights, heads], i) =>
        [m.headNorm, bind(m, m.headNorm, [g.qkv, weights, g.u.headNorms[l][i], g.step]), heads, 1]) : []),
      ...(g.qkv ? [[g.rope, bindAt(m, g.rope, [[2, g.qkv], [3, g.u.qkv[l]], ...turned]), 1, 1]] : []),
      // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, before o adds to it (into xb, which q,
      // k and v have read by now; a norm on the matrix's read would come too late: chooseTokens refuses the pair)
      ...(plan.parallel ? ffn.norm : []),
      ...attentionPasses(m, g.attention, { q: g.q, keys: cache.keys[l], values: cache.values[l], out: g.att, parts: g.parts, params: g.params,
        flash: g.u.flash, step: g.step }, plan.heads, positions),
      ...attended.quantize, matrix(P.add, o.layers[l], attended.input, g.u.o, plan.dim, [[5, g.h]]), ...biased(g.h, "bo", l),
      ...(plan.parallel ? [] : ffn.norm), ...ffn.quantize,
      // gate and up as one matrix with SwiGLU on its write; or (T154: no gate) w1, its bias and GELU
      ...(gated ? [matrix(P.glu, m.joined[l].gateUp, ffn.input, g.u.gateUp[l], plan.hidden, [[5, g.gate]])]
        : [matrix(P.glu, m.matrices.w1.pieces[0].layers[l], ffn.input, g.u.gateUp[l], plan.hidden, [[5, g.gate]]), ...biased(g.gate, "b1", l),
          [m.activation, bind(m, m.activation, [g.gate, g.u.hidden, g.step]), Math.ceil(plan.hidden / 64), 1]]),
      ...activated.quantize, matrix(P.add, down.layers[l], activated.input, g.u.down, plan.dim, [[5, g.h]]), ...biased(g.h, "b2", l));
  }
  if (head) {
    // T232: a ternary classifier's outlier channels go out of the normed stream before it is quantized (TAKE_OUTLIERS),
    // and their columns are added to the piece's logits after its matrix (TERNARY_COLUMNS), as the CPU has them
    const take = g.take ? [[g.take, bind(m, g.take, [g.xb, g.picked, g.u.take]), 1, 1]] : null;
    const final = normed(m.finalNorm, m.finalBias, g.u.final, !form.dp4a || floatHead(m), take);
    const logitsOf = (piece) => ({ buffer: g.logits, offset: piece.first * 4, size: piece.rows * 4 });
    list.push(...final.norm, ...final.quantize,
      // (T209: a piece at a time into its range of the logits: its first row is where the device binds, piecesOf)
      ...m.tables.classifier.flatMap((piece, i) => [matrix(P.classifier, [piece.values, piece.scales], final.input, g.u.classifier[i], piece.rows, [[5, logitsOf(piece)]]),
        ...(take ? [[g.columns, bind(m, g.columns, [piece.values, piece.scales, g.picked, logitsOf(piece), g.u.columns[i]]), ...spread(m, piece.rows, 256)]] : [])]),
      [g.sample, bindAt(m, g.sample, [[0, g.logits], [1, g.probs], [2, g.order], [3, g.state], [4, g.chosen], [5, g.randoms], [6, g.settings]]), 1, 1]);
  }
  return list;
}
// a form's whole step for a run that reads up to positions, made again when the cache has grown (its buffers are others
// then); T224: one a count of the attention's parts a head (flash_attn_vec's nwg goes by the positions)
function tokenStep(m, form, positions) {
  const a = m.gen.attention, parts = a.tiles ? 0 : m.wgsl.flashVecSplits(a.shape, positions);
  if (form.steps?.cache !== m.cache) form.steps = { cache: m.cache, lists: new Map() };
  if (!form.steps.lists.has(parts)) form.steps.lists.set(parts, tokenPass(m, form, { positions }));
  return form.steps.lists.get(parts);
}
// count steps of the dispatches from the state given (with the settings and a random number a step), in one
// submission, and read back: the ids and the state's words, and (keep) the keys and values of the positions pos to
// pos + count - 1. extra(encoder): what a check copies out besides. Returns { ids, sampled, stopped, notFinite (T219:
// the sampler refused the step after the sampled ones: its logits were not finite), kv: [keys, values][layer], the
// positions' rows of float16 as bytes }
async function runTokens(m, dispatches, { count, pos, state, settings, randoms, keep = false, extra }) {
  const { device, plan, wgsl, gen: g } = m, kvRow = plan.kvHeads * plan.headSize * 2, idsBytes = g.most * 4;
  if (pos + count > m.cache.capacity) throw new Error(`positions to ${pos + count} are past the GPU's cache`);
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
  encoder.copyBufferToBuffer(g.chosen, 0, g.readback, 0, count * 4);
  encoder.copyBufferToBuffer(g.state, 0, g.readback, idsBytes, wgsl.STATE_BYTES);
  const kvAt = idsBytes + wgsl.STATE_BYTES;
  if (keep) {
    for (let l = 0; l < plan.layers; l++) {
      encoder.copyBufferToBuffer(m.cache.keys[l], pos * kvRow, g.readback, kvAt + l * g.most * kvRow, count * kvRow);
      encoder.copyBufferToBuffer(m.cache.values[l], pos * kvRow, g.readback, kvAt + (plan.layers + l) * g.most * kvRow, count * kvRow);
    }
  }
  extra?.(encoder);
  device.queue.submit([encoder.finish()]);
  const invalid = await device.popErrorScope(), full = await device.popErrorScope();
  if (invalid || full) throw new Error(`the GPU refused a token (${(invalid ?? full).message})`);
  await g.readback.mapAsync(MAP_READ);
  try {
    const words = new Uint32Array(g.readback.getMappedRange(0, idsBytes + wgsl.STATE_BYTES).slice(0));
    const after = words.subarray(g.most);
    const out = { ids: words.slice(0, count), sampled: after[5], stopped: after[7], notFinite: after[wgsl.STATE_NOT_FINITE] };
    if (keep) {
      const bytes = new Uint8Array(g.readback.getMappedRange(kvAt, 2 * plan.layers * g.most * kvRow).slice(0));
      out.kv = [0, 1].map((side) => [...Array(plan.layers)].map((_, l) => {
        const at = (side * plan.layers + l) * g.most * kvRow;
        return bytes.subarray(at, at + count * kvRow);
      }));
    }
    return out;
  } finally {
    g.readback.unmap();
  }
}

export { tokenCandidates, floatHead, tokenShape, tokenCodes, tokenBuffers, tokenPass, tokenStep, runTokens };
