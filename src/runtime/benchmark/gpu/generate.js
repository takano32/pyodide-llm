// The step "generate" (T151, T191): tokens generated on the GPU, the sampling too, read back each or once a submission, the
// sampling alone, and the logits the sampling is measured and checked on.
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)
const { shared, MODELS, PROMPT_MODEL, matrixBytes, gpu, STORAGE, COPY_DST, COPY_SRC, UNIFORM, MAP_READ, buffer, fill,
  floats, run, middle, interleaved, scoped } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { EPS, layerShape, tokenForm, layerCodes, fusedLayer, compiled, ropeTable } = await import(new URL(`layerparts.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T151: generated tokens on the GPU (shaders.js's EMBED, fusedMatVec and SAMPLE): a token a compute pass, the
// sampler's state carried from one to the next on the GPU, and the ids read back once for the run: against reading
// each one back, whose wait (3 to 8.6 ms for 4 bytes on the owner's Android, T134) a token pays alone.
// The layers are tokenForm()'s (T175): the fused layer on DP4A where the packed int8 dot is, else T150's with the
// workgroup's reduction (which layer is faster is the layer table's; every row here runs the same layers, so their
// difference is the reading back alone). On DP4A the classifier takes its normed input quantized too
// (NORM_QUANTIZE, then fusedDp4aMatVec's "write")
const GENERATE_MODEL = { ...PROMPT_MODEL, vocab: 32000 };  // the CPU section's model and its vocabulary
// the settings a token is timed with: the list's sampled models' (src/models.js: temperature 0.7, top-p 0.9, and
// tiny-lm's penalty 1.3)
const GENERATE_SETTINGS = { temperature: 0.7, topp: 0.9, penalty: 1.3 };
// the forms, in turn: tokens a submission (1: each read back as it comes), GENERATE_TOKENS tokens each round
const GENERATE_COUNTS = [1, 4, 8, 16], GENERATE_TOKENS = 16, GENERATE_ROUNDS = 5, GENERATE_MOST = 64, GENERATE_POS = 127;
// a fallback adapter runs the check's small model, 2 tokens a form once: its times are no GPU's
const GENERATE_CHECK = { dim: 256, hidden: 512, heads: 4, kvHeads: 2, layers: 2, vocab: 1003 };
async function generationPipes(headSize, form) {
  const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = shared.device.limits;
  const flash = shared.WGSL.flashShape({ headSize, half: false, subgroups: false, memory, threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) });
  if (flash.none) throw new Error(flash.none);
  const pipes = {};
  const classifier = form.dp4a ? shared.WGSL.fusedDp4aMatVec({ output: "write" }) : shared.WGSL.fusedMatVec({ input: "norm", output: "write", subgroups: false });
  for (const [key, code] of [["embed", shared.WGSL.EMBED], ["flash", shared.WGSL.flashTile(flash)], ...layerCodes(form), ["classifier", classifier]]) {
    pipes[key] = await compiled(code);
    postMessage({ alive: true });
  }
  pipes.sampler = await samplerPipes();
  return pipes;
}
// T191: the two ways of sampling a token on the GPU, measured side by side: SAMPLE's one workgroup (T151) and the
// sampling in chunks of the vocabulary (shaders.js's SAMPLER_STAGES, a workgroup a chunk)
const SAMPLERS = ["one", "chunks"];
async function samplerPipes() {
  const pipes = { one: await compiled(shared.WGSL.SAMPLE), chunks: [] };
  postMessage({ alive: true });
  for (const stage of shared.WGSL.SAMPLER_STAGES) {
    pipes.chunks.push({ ...stage, pipeline: await compiled(stage.code) });
    postMessage({ alive: true });
  }
  return pipes;
}
// a sampler's dispatches, [pipeline, bind group, x, y] each: b holds the buffers by SAMPLE's binding numbers (0 the
// logits, 1 probs, 2 order, 3 the state, 4 chosen, 5 the random numbers, 6 the settings) and 7 the chunks' partial
// results (WGSL.samplePartsBytes of the vocabulary)
function samplerDispatches(kind, pipes, b, vocab) {
  const group = (pipeline, bindings) => shared.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((binding) => ({ binding, resource: { buffer: b[binding] } })) });
  if (kind === "one") return [[pipes.one, group(pipes.one, [0, 1, 2, 3, 4, 5, 6]), 1, 1]];
  const chunks = shared.WGSL.sampleChunks(vocab);
  return pipes.chunks.map((stage) => [stage.pipeline, group(stage.pipeline, stage.bindings), stage.chunks ? chunks : 1, 1]);
}
// A model on the GPU for a run of tokens: its layers (T150's four matrices each, the cache of its own), the norms
// (two a layer, then the final one), the classifier (the embedding too: tied, as Llama 3.2 1B's), RoPE's table of
// `positions`, and the sampler's buffers. form: the layer's (tokenForm()). data: the check's ({layers: [{qkv, o,
// gateUp, down: {w, s}}], classifier, norms, keys, values}), else random. encode(encoder): one token's pass and the
// Step copied from the state after it. T175: on DP4A with the check's data, every quantized vector of a pass is its
// own (four a layer, then the classifier's), and encode copies them into `recorded` after the pass, token by token
// from reset() on (recording: their sizes; recorded(bytes) cuts what was read back into [token][vector] {xq, xs}).
// T225: with the check's data every form records the token's logits after them (as the sampling left them: the
// penalty divides in place), recorded(bytes).logits[token], and caches holds the layers' keys and values
function generationParts(model, form, pipes, positions, owned, data) {
  const shape = layerShape(model), { dim, hidden, heads, kvDim, headSize } = shape, { vocab, layers } = model;
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
  const weights = ([rows, n], given) => {
    if (rows * n > shared.device.limits.maxStorageBufferBindingSize) throw new Error(`a matrix of ${rows} × ${n} is past a binding of this device`);
    const w = make(rows * n), s = make((rows * n / shared.GROUP) * 4);
    if (given) {
      shared.device.queue.writeBuffer(w, 0, given.w);
      shared.device.queue.writeBuffer(s, 0, given.s);
    } else {
      fill(w, rows * n);
      shared.device.queue.writeBuffer(s, 0, floats(rows * n / shared.GROUP, 0.002));
    }
    return { w, s, rows };
  };
  const cacheBytes = positions * kvDim * 2;
  const stack = [...Array(layers)].map((_, l) => {
    const m = Object.fromEntries(Object.entries(shape.matrices).map(([key, matrix]) => [key, weights(matrix, data?.layers[l][key])]));
    const keys = make(cacheBytes), values = make(cacheBytes);
    if (data) {
      shared.device.queue.writeBuffer(keys, 0, data.keys[l]);
      shared.device.queue.writeBuffer(values, 0, data.values[l]);
    }
    return { ...m, keys, values };
  });
  const classifier = weights([vocab, dim], data?.classifier);
  // T175: one quantized vector serves all of a token's quantizations (each is read before the next is made), but for
  // the check, which reads each one back
  const pair = (n) => ({ n, xq: make(n), xs: make((n / shared.GROUP) * 4) });
  const kept = form.dp4a && data;
  const sizes = [...[...Array(layers)].flatMap(() => [dim, dim, dim, hidden]), dim];
  const quantizedAll = kept ? sizes.map(pair) : Array(sizes.length).fill(pair(Math.max(dim, hidden)));
  const v = { h: make(dim * 4), q: make(dim * 4), att: make(dim * 4), g: make(hidden * 4), logits: make(vocab * 4),
    probs: make(vocab * 4), order: make(vocab * 4), parts: make(shared.WGSL.samplePartsBytes(vocab)), norms: make((2 * layers + 1) * dim * 4), angles: make(positions * headSize * 4),
    state: make(shared.WGSL.STATE_BYTES), chosen: make(Math.max(positions, 4) * 4), randoms: make(positions * 4),
    step: make(16, UNIFORM | COPY_DST), settings: make(shared.WGSL.SAMPLING_BYTES, UNIFORM | COPY_DST) };
  shared.device.queue.writeBuffer(v.norms, 0, data?.norms ?? new Float32Array((2 * layers + 1) * dim).map(() => 0.5 + Math.random()));
  shared.device.queue.writeBuffer(v.angles, 0, ropeTable(headSize, positions));
  const eps = data?.eps ?? EPS;
  // fusedMatVec's Params: rows, words, perRow, second, eps, normAt, qRows, kvRows, headSize, turned
  const params = (rows, n, second = 0, normAt = 0) => {
    const bytes = new ArrayBuffer(48);
    new Uint32Array(bytes).set([rows, n / 4, n / shared.GROUP, second, 0, normAt, dim, kvDim, headSize, headSize, 0, 0]);
    new Float32Array(bytes, 16, 1)[0] = eps;
    return uniform(new Uint8Array(bytes));
  };
  const flashParams = new ArrayBuffer(16);
  new Uint32Array(flashParams, 0, 2).set([heads, shape.kvHeads]);
  new Float32Array(flashParams, 8, 1)[0] = 1 / Math.sqrt(headSize);
  const step = v.step;
  // RMSNORM's and NORM_QUANTIZE's Norm (size, at, eps): a layer's two norms and the final one, from `at` in the norms
  const normParams = (at) => {
    const bytes = new ArrayBuffer(16);
    new Uint32Array(bytes, 0, 2).set([dim, at]);
    new Float32Array(bytes, 8, 1)[0] = eps;
    return uniform(new Uint8Array(bytes));
  };
  const common = { step, flash: uniform(new Uint8Array(flashParams)), o: params(dim, dim), down: params(dim, hidden),
    quantize: [dim, hidden].map((n) => uniform(new Uint32Array([n, n, 0, 0]))) };
  const u = { embed: uniform(new Uint32Array([dim, 0, 0, 0])), classifier: params(vocab, dim, 0, 2 * layers * dim), final: normParams(2 * layers * dim),
    layers: stack.map((_, l) => ({ ...common, qkv: params(dim + 2 * kvDim, dim, 0, 2 * l * dim), gateUp: params(hidden, dim, hidden, (2 * l + 1) * dim),
      attentionNorm: normParams(2 * l * dim), ffnNorm: normParams((2 * l + 1) * dim) })) };
  const group = (pipeline, entries) => shared.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: entries.map(([binding, resource]) => ({ binding, resource: { buffer: resource } })) });
  const dispatches = [[pipes.embed, group(pipes.embed, [[0, classifier.w], [1, classifier.s], [2, v.state], [3, v.h], [4, u.embed]]), 1, 1]];
  stack.forEach((m, l) => dispatches.push(...fusedLayer(form, pipes, shape, m, m, { ...v, quantized: quantizedAll.slice(4 * l, 4 * l + 4) }, u.layers[l], group)));
  const quantized = quantizedAll[4 * layers];
  const rows = Math.ceil(vocab / (form.dp4a ? shared.WGSL.ORT_DP4A_MATVEC_ROWS : shared.WGSL.MUL_MAT_VEC_ROWS));
  const across = Math.min(rows, shared.device.limits.maxComputeWorkgroupsPerDimension);
  if (form.dp4a) {
    dispatches.push([pipes.normQuantize, group(pipes.normQuantize, [[0, v.h], [1, v.norms], [2, quantized.xq], [3, quantized.xs], [4, u.final], [5, step]]), 1, 1],
      [pipes.classifier, group(pipes.classifier, [[0, classifier.w], [1, classifier.s], [2, quantized.xq], [3, u.classifier], [4, quantized.xs], [5, v.logits]]), across, Math.ceil(rows / across)]);
  } else {
    dispatches.push([pipes.classifier, group(pipes.classifier, [[0, classifier.w], [1, classifier.s], [2, v.h], [3, u.classifier], [4, v.norms], [5, v.logits]]), across, Math.ceil(rows / across)]);
  }
  // the token's sampling: SAMPLE's one workgroup, or in chunks (T191), as use() says
  const samplerBuffers = { 0: v.logits, 1: v.probs, 2: v.order, 3: v.state, 4: v.chosen, 5: v.randoms, 6: v.settings, 7: v.parts };
  const samplers = Object.fromEntries(SAMPLERS.map((kind) => [kind, samplerDispatches(kind, pipes.sampler, samplerBuffers, vocab)]));
  let sampling = samplers.one;
  const use = (kind) => (sampling = samplers[kind]);
  // the check's record of the quantized vectors: a token's are perToken bytes, each xq then its xs
  const vectorBytes = kept ? quantizedAll.reduce((sum, { n }) => sum + n + (n / shared.GROUP) * 4, 0) : 0;
  const perToken = vectorBytes + (data ? vocab * 4 : 0);
  const record = data ? make(positions * perToken) : undefined;
  let passes = 0;
  const encode = (encoder) => {
    const pass = encoder.beginComputePass();
    dispatches.forEach((d) => run(pass, d));
    sampling.forEach((d) => run(pass, d));
    pass.end();
    // the Step of the next token: the state's first four words (a uniform is not a shader's to write)
    encoder.copyBufferToBuffer(v.state, 0, v.step, 0, 16);
    if (record) {
      let at = passes * perToken;
      for (const { n, xq, xs } of kept ? quantizedAll : []) {
        encoder.copyBufferToBuffer(xq, 0, record, at, n);
        encoder.copyBufferToBuffer(xs, 0, record, at + n, (n / shared.GROUP) * 4);
        at += n + (n / shared.GROUP) * 4;
      }
      encoder.copyBufferToBuffer(v.logits, 0, record, at, vocab * 4);
    }
    passes++;
  };
  // what was recorded read back (bytes: count tokens' of it): { vectors: on DP4A [token][vector] { xq, xs }, logits:
  // [token] the logits }
  const recorded = (bytes, count) => ({ vectors: kept ? [...Array(count)].map((_, k) => {
    let at = k * perToken;
    return quantizedAll.map(({ n }) => {
      const one = { xq: new Int8Array(bytes, at, n), xs: new Float32Array(bytes, at + n, n / shared.GROUP) };
      at += n + (n / shared.GROUP) * 4;
      return one;
    });
  }) : undefined, logits: [...Array(count)].map((_, k) => new Float32Array(bytes, k * perToken + vectorBytes, vocab)) });
  // a run from the start: the state (samplingState), the random numbers and the settings (samplingSettings)
  const reset = (state, randoms, settings) => {
    passes = 0;
    shared.device.queue.writeBuffer(v.state, 0, state);
    shared.device.queue.writeBuffer(v.step, 0, state.subarray(0, 4));
    if (randoms) shared.device.queue.writeBuffer(v.randoms, 0, randoms);
    if (settings) shared.device.queue.writeBuffer(v.settings, 0, settings);
  };
  const bytes = layers * Object.values(shape.matrices).reduce((sum, matrix) => sum + matrixBytes(matrix), 0) + matrixBytes([vocab, dim]);
  return { vectors: v, encode, reset, use, dispatches: dispatches.length + samplers.one.length, chunkDispatches: dispatches.length + samplers.chunks.length,
    bytes, caches: stack.map(({ keys, values }) => ({ keys, values })), ...(record ? { record, recording: positions * perToken, recorded } : {}) };
}
// what submitting the tokens of a run and reading back their ids and the state costs: count tokens, per of them a
// submission (each read back before the next is submitted, as a token's text is shown), from the state given. Returns
// the ms and what came back: the ids and the state's words
async function generationRun(parts, count, per, target) {
  const v = parts.vectors, bytes = per * 4 + shared.WGSL.STATE_BYTES;
  const into = target ?? buffer(bytes, MAP_READ | COPY_DST);
  const ids = new Uint32Array(count);
  let state;
  const began = performance.now();
  for (let done = 0; done < count; done += per) {
    const encoder = shared.device.createCommandEncoder();
    for (let i = 0; i < per; i++) parts.encode(encoder);
    encoder.copyBufferToBuffer(v.chosen, done * 4, into, 0, per * 4);
    encoder.copyBufferToBuffer(v.state, 0, into, per * 4, shared.WGSL.STATE_BYTES);
    shared.device.queue.submit([encoder.finish()]);
    await into.mapAsync(MAP_READ);
    const words = new Uint32Array(into.getMappedRange(0, bytes).slice(0));
    into.unmap();
    ids.set(words.subarray(0, per), done);
    state = words.subarray(per);
  }
  const ms = performance.now() - began;
  if (!target) into.destroy();
  // T219's review: a step whose logits were not finite is refused by the sampler (the run is stopped there, the tokens
  // after it are not sampled), which would make the time of a run, and of a token, a shorter one than it is: said, not timed
  if (state[shared.WGSL.STATE_NOT_FINITE]) throw new Error("the sampler refused a step: its logits were not finite numbers, so the time is not a token's");
  return { ms, ids, state };
}
// made-up random numbers in [0, 1) (the engine's are NumPy's generator's, drawn by the CPU in the same order)
const randomsOf = (count) => new Float32Array(count).map(() => Math.fround(Math.random()) % 1);
async function generate() {
  await gpu();
  const model = shared.fallback ? GENERATE_CHECK : GENERATE_MODEL, headSize = model.dim / model.heads;
  const counts = shared.fallback ? [1, 2] : GENERATE_COUNTS, tokens = shared.fallback ? 2 : GENERATE_TOKENS;
  const positions = GENERATE_POS + 2 * GENERATE_MOST + GENERATE_TOKENS + 1;
  const history = [...Array(GENERATE_POS + 1)].map(() => (Math.random() * model.vocab) | 0);
  const start = shared.WGSL.samplingState({ token: history[history.length - 1], pos: GENERATE_POS, history });
  const randoms = randomsOf(positions);
  return scoped(async (owned) => {
    const form = tokenForm(), pipes = await generationPipes(headSize, form);
    const parts = generationParts(model, form, pipes, positions, owned);
    parts.reset(start, randoms, shared.WGSL.samplingSettings({ vocab: model.vocab, ...GENERATE_SETTINGS }));
    await shared.device.queue.onSubmittedWorkDone();
    const targets = new Map(counts.map((per) => [per, buffer(per * 4 + shared.WGSL.STATE_BYTES, MAP_READ | COPY_DST)]));
    owned.push(...targets.values());
    postMessage({ alive: true });
    // the ms of a run of `tokens` from the start, per of them a submission, sampled the sampler's way (T191)
    const timed = async (per, kind) => {
      parts.use(kind);
      parts.reset(start);
      return (await generationRun(parts, tokens, per, targets.get(per))).ms;
    };
    // the work of a token without the reading back: n tokens in one submission against 2n (T168's), each sampler's,
    // the samplers in turn (interleaved(): T150's review)
    const works = {};
    if (!shared.fallback) {
      const found = await interleaved(SAMPLERS.map((kind) => async (n) => {
        parts.use(kind);
        parts.reset(start);
        return (await generationRun(parts, n, n)).ms;
      }), GENERATE_MOST);
      SAMPLERS.forEach((kind, k) => {
        const r = found[k];
        if (!r.error) works[kind] = { ms: r.ms / r.dispatches, tokens: r.dispatches, ratio: r.ratio, ...(r.unsteady ? { unsteady: true } : {}) };
      });
    }
    postMessage({ alive: true });
    // the forms and the samplers in turn (T150's review), each a run of `tokens`, after one run each to warm up
    const rounds = shared.fallback ? 1 : GENERATE_ROUNDS, times = SAMPLERS.map(() => counts.map(() => []));
    if (!shared.fallback) for (const per of counts) for (const kind of SAMPLERS) await timed(per, kind);
    for (let round = 0; round < rounds; round++) {
      for (let i = 0; i < counts.length; i++) {
        for (let k = 0; k < SAMPLERS.length; k++) times[k][i].push(await timed(counts[i], SAMPLERS[k]));
      }
      postMessage({ alive: true });
    }
    const work = works.one;
    const rows = counts.map((per, i) => {
      const msPerToken = middle(times[0][i]) / tokens;
      // what a submission costs besides its tokens' work: the submission, the wait and the ids read back
      const fixed = work && !work.unsteady ? msPerToken * per - work.ms * per : undefined;
      return { perSubmission: per, msPerToken, ...(fixed === undefined ? {} : { fixedMs: fixed }), chunks: middle(times[1][i]) / tokens };
    });
    // the sampling alone, on Llama 3's vocabulary: logits as a model's, and flat ones (every token over the floor)
    const sampling = shared.fallback ? undefined : await samplingAlone();
    return { model: shared.fallback ? "the check's small model" : "Llama 3.2 1B's width", layer: form.name, layers: model.layers, vocab: model.vocab,
      GB: parts.bytes / 1e9, dispatches: parts.dispatches, chunkDispatches: parts.chunkDispatches, tokens, settings: GENERATE_SETTINGS, work,
      ...(works.chunks ? { chunkWork: works.chunks } : {}), rows, sampling };
  });
}
// the sampling alone (paired(): T168's n and 2n) on Llama 3's vocabulary, twice: on logits as a model's (madeUpLogits:
// a spread of 2 and 20 tokens far above it, a few percent of the vocabulary over the nucleus's floor) and on flat ones
// (a spread of 1, none above: every token over the floor, so that SAMPLE gathers and reads all of them each round
// of its searches: its worst case). `over` is how many tokens are over the floor.
// Without the repetition penalty (T191's review): the penalty changes the logits in place, sampling after sampling,
// and they are written once for all the submissions, so with it each sampled peak was divided down and the logits as a
// model's went about flat while they were timed: the count over the floor the table named was not what was timed. The
// penalty itself is a thread a token of the window in either sampler.
// T191: SAMPLE's one workgroup, the sampling in chunks, and of that its last stage alone (one workgroup: the nucleus
// and the draw, the same search as SAMPLE's): each submission of the last runs one sampling in chunks, then n of its
// last stage on what that gathered, so that 2n less n is n of the last stage alone
async function samplingAlone() {
  const vocab = MODELS["Llama 3.2 1B"].vocab, positions = 2 * GENERATE_MOST + 2;
  return scoped(async (owned) => {
    const pipes = await samplerPipes();
    const make = (bytes, usage = STORAGE | COPY_DST) => {
      const b = buffer(bytes, usage);
      owned.push(b);
      return b;
    };
    const logits = make(vocab * 4), probs = make(vocab * 4), order = make(vocab * 4), state = make(shared.WGSL.STATE_BYTES),
      chosen = make(positions * 4), randoms = make(positions * 4), settings = make(shared.WGSL.SAMPLING_BYTES, UNIFORM | COPY_DST),
      parts = make(shared.WGSL.samplePartsBytes(vocab));
    shared.device.queue.writeBuffer(randoms, 0, randomsOf(positions));
    shared.device.queue.writeBuffer(settings, 0, shared.WGSL.samplingSettings({ vocab, ...GENERATE_SETTINGS, penalty: 1 }));
    const history = [...Array(64)].map(() => (Math.random() * vocab) | 0), start = shared.WGSL.samplingState({ token: history[63], pos: 0, history });
    const b = { 0: logits, 1: probs, 2: order, 3: state, 4: chosen, 5: randoms, 6: settings, 7: parts };
    // T191: SAMPLE's one workgroup, the sampling in chunks and its last stage alone, in turn (interleaved(): T150's
    // review), each [the dispatches once before the n, the dispatches n times]
    const [one, chunks] = SAMPLERS.map((kind) => samplerDispatches(kind, pipes, b, vocab));
    const last = chunks[shared.WGSL.SAMPLER_STAGES.findIndex((stage) => stage.name === "pick")];
    const lists = [[[], one], [[], chunks], [chunks, [last]]];
    const timed = async (values) => {
      shared.device.queue.writeBuffer(logits, 0, values);
      const found = await interleaved(lists.map(([before, list]) => async (n) => {
        shared.device.queue.writeBuffer(state, 0, start);
        const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
        before.forEach((d) => run(pass, d));
        for (let i = 0; i < n; i++) list.forEach((d) => run(pass, d));
        pass.end();
        const began = performance.now();
        shared.device.queue.submit([encoder.finish()]);
        await shared.device.queue.onSubmittedWorkDone();
        return performance.now() - began;
      }), GENERATE_MOST);
      const [once, inChunks, pick] = found.map((r) => (r.error ? { error: r.error } : { msEach: r.ms / r.dispatches, ...(r.unsteady ? { unsteady: true } : {}) }));
      return { ...once, over: overTheFloor(values, GENERATE_SETTINGS.temperature), chunks: { ...inChunks, pick } };
    };
    const peaked = await timed(madeUpLogits(vocab, 2)), flat = await timed(madeUpLogits(vocab, 1, 0));
    return { vocab, ...peaked, flat };
  });
}
// how many of the logits SAMPLE keeps over the nucleus's floor (kernel.ts's: within temperature × ln 1e7 of the largest)
function overTheFloor(logits, temperature) {
  let best = -Infinity;
  for (const value of logits) best = Math.max(best, value);
  const floor = best - temperature * 16.118095;
  let over = 0;
  for (const value of logits) if (value >= floor) over++;
  return over;
}
// logits with equal ones where the draw's order among them shows (the check's ties cases): two at the top (10: the
// first index, 100, is the most likely), a run of 20 at TIED_RUN (a fifth of the top's probability each at temperature
// 0.7, their indices descending as written so the order of the index is not the order written), and a tail of 200
// distinct values below (a fiftieth each, rising by the index) that a nucleus of top-p 0.9 reaches into: no equal
// logits at its border, where SAMPLE takes every equal token and the CPU's sort takes some (a different mass).
// T191: past one chunk of the sampling in chunks (WGSL.SAMPLE_CHUNK), the same spread over the chunks: the second of the
// top at 64000, the run at multiples of 3000 (from 60000 down), the tail at odd indices from 1001 (so that the order
// of equal ones and of the nucleus crosses the chunks)
const TIED_RUN = Math.fround(8.87);
function tiedLogits(vocab) {
  const logits = new Float32Array(vocab), wide = vocab > 64 * shared.WGSL.SAMPLE_CHUNK;
  logits[wide ? 64000 : 700] = 10;
  logits[100] = 10;
  for (let i = 0; i < 20; i++) logits[wide ? 60000 - 3000 * i : 900 - 3 * i] = TIED_RUN;
  for (let i = 0; i < 200; i++) logits[wide ? 1001 + 618 * i : 200 + i] = 7.26 + i * 1e-3;
  return logits;
}
// T191 (Fable's check): a few tokens far over the rest, on the borders of the chunks of the sampling in chunks (the
// first and the last token of a chunk, the very last token of the vocabulary: a thread's fourth), two of them equal
// across a border (1023 and 1024), the rest a narrow normal spread: at temperature 0.4 the floor (best − 6.45) is
// above every other token, so nearly every chunk gathers nothing, the count over the floor is far below a workgroup,
// and the draw without a nucleus walks through chunks whose sums are next to nothing
const SPARSE_PEAKS = [[0, 10], [2047, 10.25], [1023, 10.5], [1024, 10.5], [50000, 10.75], [64511, 11], [128255, 11.25], [64512, 11.5], [128000, 11.75], [1, 12]];
function sparseLogits(vocab) {
  const logits = madeUpLogits(vocab, 0.5, 0);
  for (const [at, value] of SPARSE_PEAKS) logits[at] = value;
  return logits;
}
// T219: logits that are not finite, in place, at `at` (the kinds of two or three logits take the places after it). What
// the sampler must refuse (T195's rule: the largest logit is no finite number: a NaN or +inf anywhere, or every logit
// -inf), and what it must not (UNREFUSED). The bits of a NaN are a device's to give (the review of T219): the NaN
// JavaScript writes is the quiet one, and a device may give one with the sign set (x86's default NaN), a signaling one,
// or one of all ones, which a test of a quiet NaN alone, or of the bits as a signed number, does not see. The kinds
// "... in the window" are set where the history is known (the penalty multiplies or divides the logit of its tokens).
// A few -inf are seven, every 131st token from 5 (where the most likely is among them both sides take the next: the CPU
// leaves them out, and draws as ever); one finite logit is the only token that can be drawn. (The largest finite float
// is not here: that its bits are under the infinity's is proved for every float32 in the review's notes, and a draw
// from logits of 3.4e38 asks the device's arithmetic overflowing, which a verdict of WRONG on the owner's device, with
// its ratios withheld, should not hang on.)
const UNREFUSED = ["-inf some", "denormals and -0", "one finite"];
function unfiniteLogits(logits, kind, at) {
  const bits = new Uint32Array(logits.buffer, logits.byteOffset, logits.length);
  if (kind === "nan") logits[at] = NaN;
  else if (kind === "nan, the sign set") bits[at] = 0xffc00000;
  else if (kind === "nan, signaling") bits[at] = 0x7f800001;
  else if (kind === "nan, all ones") bits[at] = 0xffffffff;
  else if (kind === "nan all") logits.fill(NaN);
  else if (kind === "+inf") logits[at] = Infinity;
  else if (kind === "+inf and -inf") (logits[at] = Infinity), (logits[at + 1] = -Infinity);
  else if (kind === "-inf all") logits.fill(-Infinity);
  else if (kind === "-inf some") for (let i = 5; i < logits.length; i += 131) logits[i] = -Infinity;
  else if (kind === "denormals and -0") (logits[at] = 1e-45), (logits[at + 1] = -0), (logits[at + 2] = -1e-45);
  else if (kind === "one finite") (logits.fill(-Infinity), (logits[at] = 1.5));
}
// logits as a model's look (a few tokens far above the rest), made up: a normal spread and `peaks` tokens 8 to 14 over it
function madeUpLogits(vocab, spread, peaks = 20) {
  const logits = new Float32Array(vocab);
  for (let i = 0; i < vocab; i++) logits[i] = spread * Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  for (let i = 0; i < peaks; i++) logits[(Math.random() * vocab) | 0] += 8 + 6 * Math.random();
  return logits;
}

export { GENERATE_SETTINGS, GENERATE_CHECK, generationPipes, samplerPipes, samplerDispatches, generationParts,
  generationRun, randomsOf, generate, TIED_RUN, tiedLogits, sparseLogits, UNREFUSED, unfiniteLogits, madeUpLogits };
