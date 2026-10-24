const search = "?v=gpu-check";
const { compileKernels, createForward, weightsMemory, footprint, external, gpuOnlyWeights, gpuOnlyPlan, layerWeightsOf } = await import("/runtime/forward.ts" + search);
const { GPU_DONE, GPU_FAILED, GPU_BEAT, GPU_WANTED } = await import("/runtime/jobs.ts" + search);
const fetched = async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer());
const b64 = (floats) => {
  const bytes = new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
  let text = "";
  for (let i = 0; i < bytes.length; i += 32768) text += String.fromCharCode(...bytes.subarray(i, i + 32768));
  return btoa(text);
};
const openGpu = () => new Worker("/runtime/gpu.ts" + search, { type: "module" });
// T241: the numbers of the requests the harness itself sends the GPU's worker (forward.js counts its own from 1)
let ownSerial = 0x40000000;
// T243: the blocks forward.js is to refuse, a run each of those that end with it (refusedBlock): the value in the row
// of a token (the block's first and its last), at a column
let refusals = 0;
// T147: every tiled shader of the matrices this adapter can make (each forced in a run of its own), and the one the
// GPU's worker chooses by timing them (the first run)
const wgsl = await import("/runtime/shaders.ts" + search);
const adapter = await navigator.gpu?.requestAdapter();
// (T232, ternary: those of a model of ternary weights, the packed ones alone; --forms does not narrow them)
const formsOf = (ternary) => (!adapter ? [] : wgsl.promptForms({ half: adapter.features.has("shader-f16"), subgroups: adapter.features.has("subgroups"),
  packed: navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product"),
  memory: adapter.limits.maxComputeWorkgroupStorageSize, threads: adapter.limits.maxComputeInvocationsPerWorkgroup, ternary })
  .filter((form) => !form.none).map((form) => form.name).filter((name) => ternary || !ONLY.length || ONLY.some((part) => name.includes(part))));
// T152: the forms of a token's layer this adapter can make (gpu.js's TOKEN_FORMS), each forced in a run of its own
// (T271.1: a model in a rotated basis takes those with the norms apart alone, gpu/tokens.ts's tokenCandidates())
const features = navigator.gpu?.wgslLanguageFeatures;
const tokenFormsOf = (ternary, rotated) => tokenFormsAll(ternary).filter((name) => !rotated || name.endsWith("the norms apart"));
const tokenFormsAll = (ternary) => (!adapter ? [] : ternary
  ? (features?.has("packed_4x8_integer_dot_product") ? ["DP4A, fused (T175), ternary", "DP4A, fused (T175), ternary, the norms apart"] : [])
  : ["llama.cpp, fused (T150)",
    ...(adapter.features.has("subgroups") && features?.has("subgroup_id") ? ["llama.cpp, fused (T150), subgroups"] : []),
    ...(features?.has("packed_4x8_integer_dot_product") ? ["DP4A, fused (T175)", "DP4A, fused (T175), the norms apart"] : [])]);
// T224: the attentions of a token this adapter can make (gpu.js's chooseTokenAttention), each forced in a run of its own
// with the first form of a token's layer
const tokenAttentions = !adapter ? [] : [...(adapter.features.has("subgroups") && features?.has("subgroup_id") ? ["llama.cpp flash_attn_vec, subgroups"] : []),
  "llama.cpp flash_attn_vec", "the prompt's attention tiles"];
// T241's review: the two quantizers alone (QUANTIZE and NORM_QUANTIZE of shaders.js) on groups that hold a value that is no
// finite number, and the word each stores for a group's scale (xs, a u32). The rounds of the layers below cannot tell a scale
// of infinity from a NaN one: it makes a row of the next matrix infinite, and the next norm's 0 x inf a NaN, on lavapipe and
// SwiftShader alike (the implementer's mutants "no-or" and "no-select" passed all of them). A finite group's word is to be the
// scale of JavaScript's quantize_x (the largest |value| / 127, within 1e-4: a device's division and square root are not
// rounded exactly); a group with a NaN or an infinity (a NaN's own, an infinity of a value, a NaN or an infinity of the norm's
// weight, what the norm makes of a row that holds an infinity) a NaN word: the exponent all ones and a fraction. The inputs
// are words, so that a signalling NaN and a negative one arrive as they are written. Returns { rows: [{ shader, what, word,
// want, ok }] } (a row a group), or { skipped } or { error }
const quantizerProbe = async () => {
  // (an adapter of its own: an adapter makes one device, and this harness reads its features and key beside the probe)
  const own = await navigator.gpu?.requestAdapter();
  if (!own) return { skipped: "no adapter" };
  const device = await own.requestDevice();
  const STORAGE = 0x80, UNIFORM = 0x40, COPY_SRC = 0x04, COPY_DST = 0x08, MAP_READ = 0x01, GROUP = wgsl.GROUP;
  const room = (bytes) => Math.max(16, Math.ceil(bytes / 16) * 16);
  const make = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const upload = (data, usage) => {
    const buffer = device.createBuffer({ size: room(data.byteLength), usage: usage | COPY_DST });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const output = (bytes) => device.createBuffer({ size: room(bytes), usage: STORAGE | COPY_SRC });
  // one dispatch of pipeline on the buffers (binding 0, 1, ...), then the count words of the one at scaleAt
  const scales = async (pipeline, buffers, scaleAt, workgroups, count) => {
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const back = device.createBuffer({ size: room(count * 4), usage: MAP_READ | COPY_DST });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(workgroups, 1, 1);
    pass.end();
    encoder.copyBufferToBuffer(buffers[scaleAt], 0, back, 0, room(count * 4));
    device.queue.submit([encoder.finish()]);
    await back.mapAsync(MAP_READ);
    const got = Array.from(new Uint32Array(back.getMappedRange().slice(0)).subarray(0, count));
    back.unmap();
    return got;
  };
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  const bits = (value) => ((f32[0] = value), u32[0]);
  const float = (word) => ((u32[0] = word), f32[0]);
  const isNaNWord = (word) => ((word >>> 23) & 0xff) === 0xff && (word & 0x7fffff) !== 0;
  let seed = 241;
  const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const spread = (width, count = GROUP) => Array.from({ length: count }, () => bits(Math.fround((random() * 2 - 1) * width)));
  const at = (word, index, rest = 1.5) => Array.from({ length: GROUP }, (_, i) => (i === index ? word : bits(rest)));
  const INF = 0x7f800000, NEGATIVE_INF = 0xff800000, NAN = 0x7fc00000, SIGNALLING = 0x7f800001, NEGATIVE_NAN = 0xffc00000, LARGEST = 0x7f7fffff;
  const rows = [];
  const say = (shader, what, word, want, tolerance) => {
    const ok = want === "NaN" ? isNaNWord(word) : !isNaNWord(word) && Number.isFinite(float(word)) && Math.abs(float(word) - want) <= tolerance * want;
    rows.push({ shader, what, word: word.toString(16), want: want === "NaN" ? "a NaN" : want, ok });
  };
  const largest = (words) => words.reduce((most, word) => Math.max(most, Math.abs(float(word))), 0);
  // QUANTIZE: a group each: finite ones (random; zeros and negative zeros; a denormal among small values; the largest
  // finite value; 1e30 among 1e-30), then +inf, -inf, a NaN, a signalling NaN, a negative NaN, only -inf, an infinity and a NaN
  const cases = [["a finite group", spread(3)], ["zeros and negative zeros", Array.from({ length: GROUP }, (_, i) => (i % 2 ? 0x80000000 : 0))],
    ["a denormal among small values", at(5, 3, 0.25)], ["the largest finite value", at(LARGEST, 31, 0.5)], ["1e30 among 1e-30", at(bits(1e30), 12, 1e-30)],
    ["+inf", at(INF, 5)], ["-inf", at(NEGATIVE_INF, 31)], ["a NaN", at(NAN, 0)], ["a signalling NaN", at(SIGNALLING, 17)], ["a negative NaN", at(NEGATIVE_NAN, 9)],
    ["only -inf", new Array(GROUP).fill(NEGATIVE_INF)], ["an infinity and a NaN", Array.from({ length: GROUP }, (_, i) => (i === 2 ? INF : i === 20 ? NAN : bits(1)))]];
  const words = cases.flatMap(([, group]) => group), tokens = (count) => upload(new Uint32Array([count, 0, 0, 0]), UNIFORM);
  const quantizedScales = await scales(make(wgsl.QUANTIZE), [upload(new Uint32Array(words), STORAGE), output(words.length), output(cases.length * 4),
    upload(new Uint32Array([words.length, words.length, 0, 0]), UNIFORM), tokens(1)], 2, Math.ceil(cases.length / 64), cases.length);
  cases.forEach(([what, group], g) => say("QUANTIZE", what, quantizedScales[g], group.some((w) => ((w >>> 23) & 0xff) === 0xff) ? "NaN" : Math.fround(largest(group) / 127), 1e-5));
  // NORM_QUANTIZE: rows of 4 groups, the weight 1 and eps 1 where it is not the case's own. Three tokens: finite; an infinity
  // at 40 (group 1: its norm's scale is 0, and 0 x inf a NaN); a NaN at 70 (group 2, and the norm's scale a NaN). Then one
  // token whose weights hold an infinity (40, over x = 1.5) and a NaN (70)
  const size = 4 * GROUP, row = spread(2, size), ones = new Array(size).fill(bits(1));
  const norm = new ArrayBuffer(16);
  new Uint32Array(norm).set([size, 0, 0, 0]);
  new Float32Array(norm)[2] = 1;
  const withWord = (list, index, word) => list.map((w, i) => (i === index ? word : w));
  const expected = (x, weight) => {
    const values = x.map(float);
    const s = 1 / Math.sqrt(values.reduce((sum, v) => sum + v * v, 0) / size + 1);
    return Array.from({ length: size / GROUP }, (_, g) => Math.fround(Array.from({ length: GROUP }, (_, i) => Math.abs(float(weight[g * GROUP + i]) * s * values[g * GROUP + i])).reduce((m, v) => Math.max(m, v), 0) / 127));
  };
  const normed = make(wgsl.NORM_QUANTIZE);
  const normalize = async (what, xRows, weight, groupsOf) => {
    const got = await scales(normed, [upload(new Uint32Array(xRows.flat()), STORAGE), upload(new Uint32Array(weight), STORAGE), output(xRows.length * size),
      output(xRows.length * 4 * 4), upload(norm, UNIFORM), tokens(xRows.length)], 3, xRows.length, xRows.length * 4);
    xRows.forEach((x, t) => groupsOf(t, expected(x.map((w) => (isNaNWord(w) || (w & 0x7fffffff) === INF ? bits(0) : w)), weight)).forEach(([g, want, tolerance]) => say("NORM_QUANTIZE", what[t] + ", group " + g, got[t * 4 + g], want, tolerance)));
  };
  await normalize(["finite", "an infinity at 40", "a NaN at 70"], [row, withWord(row, 40, INF), withWord(row, 70, NAN)], ones,
    (t, finite) => (t === 0 ? finite.map((want, g) => [g, want, 1e-4]) : t === 1 ? [[1, "NaN"]] : [[2, "NaN"]]));
  await normalize(["weights with an infinity at 40 and a NaN at 70"], [withWord(row, 40, bits(1.5))], withWord(withWord(ones, 40, INF), 70, NAN),
    (t, finite) => [[0, finite[0], 1e-4], [1, "NaN"], [2, "NaN"], [3, finite[3], 1e-4]]);
  device.destroy();
  return { rows };
};
try {
  const narrow = compileKernels(await fetched("/built/simdkernel_shared.wasm"), await fetched("/built/simdkernel_relaxed_shared.wasm"));
  const results = [];
  const quantizers = await quantizerProbe().catch((error) => ({ error: String(error?.stack ?? error) }));
  for (const c of await (await fetch("/cases.json")).json()) {
    const plan = c.plan, ternary = c.dtype === "ternary";
    const forms = formsOf(ternary), tokenForms = tokenFormsOf(ternary, Boolean(plan.rotated));
    // T209: what a third of a table is in bytes on the GPU (T232: a ternary weight is a quarter of a byte)
    const tablePieceBytes = Math.ceil((plan.vocab_size * plan.dim) / (ternary ? 4 : 1) / 3);
    for (const name of Object.keys(plan.derived)) plan.derived[name] = Uint8Array.from(atob(plan.derived[name]), (ch) => ch.charCodeAt(0));
    const checkpoint = await fetched(c.checkpoint), size = checkpoint.length, tokens = c.reference.tokens, n = tokens.length - 1;
    // T155: a 64-bit memory (c.wide) with its kernels, the checkpoint 4 GiB up as threads-check's --high puts it
    const high = c.wide ? 2 ** 32 : 0;
    const kernels = c.wide ? compileKernels(await fetched("/built/simdkernel_shared64.wasm"), await fetched("/built/simdkernel_relaxed_shared64.wasm"), true) : narrow;
    const after = footprint(c.reference.header, size, { dtype: c.dtype, halfKV: true, shared: true, gpu: true, head_dim: c.headDim, arch: c.arch });
    const { memory, base: low } = weightsMemory(size + high, { shared: true, wide: Boolean(c.wide), after });
    const base = low + high;
    new Uint8Array(memory.buffer, base, size).set(checkpoint);
    // T148: SwiftShader and lavapipe are fallback adapters, which the page refuses: the tests take them (fallback),
    // and give the GPU every block it can take (always: a fallback adapter is far slower than the CPU). T155: the
    // case's own (the pieces of its matrices)
    const TESTS = { fallback: true, always: true, ...c.force };
    // T241's review: whether this browser puts a NaN in this model's rows and values (NAN_ROUNDS; models.mjs says why)
    const nan = NAN_ROUNDS.models === "all" || (NAN_ROUNDS.models === "made-up" ? c.id.startsWith("synthetic") : NAN_ROUNDS.models.includes(c.id));
    // the seconds the NaN rounds take in this model's runs (the log's "seconds of the NaN rounds of <model>"): rows and
    // values (T241), the NaN-logits round (T219), the block through forward.js (T243), the weights (T241)
    const nanSeconds = { rows: 0, values: 0, logits: 0, block: 0, weights: 0 };
    const timed = (round, fn) => {
      const began = performance.now();
      try {
        return fn();
      } finally {
        nanSeconds[round] += (performance.now() - began) / 1000;
      }
    };
    // T152: the steps of a generation (see the head of steps.mjs), where the GPU took them
    const generation = (engine) => {
      // (T152's review: and the most this adapter binds, which a classifier may pass, Llama 3.2 1B's 262.7 MB on lavapipe's 128 MiB)
      if (!engine.tokenBlock) return { why: engine.gpuTokensWhyNot ?? "the GPU took no steps", planned: engine.gpuTokensPlanned,
        binds: Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize) };
      const greedy = c.reference.greedy, history = (ids) => [...tokens, ...ids];
      // T545.2: a model whose layers keep a state from token to token (an LFM2's) goes on only from the position after
      // the last it computed (forward.js's follow()), where keys and values can be written again at any position: a
      // step asked at position n again has the prompt through first (by the side that took it in the run, or the CPU's
      // where rewound says so: the CPU's keys and values going up, below), its state then the prompt's
      const keepsState = plan.layers.some((layer) => layer.kind !== "attention");
      let rewound = null;
      const again = (pos) => {
        if (!keepsState || pos !== n) return;
        const side = engine.gpuSide;
        engine.gpuSide = rewound;
        engine.forwardMany(tokens.slice(0, n), 0);
        engine.gpuSide = side;
      };
      // (T546: narrowing, a top-k, min-p and the presence penalty, of which the ids before are the sampled ones)
      const ask = (token, pos, before, count, settings = [0, 0.9, 1], randoms = [], stops = [], narrowing = [0, 0, 0]) => {
        const h = history(before);
        again(pos);
        return engine.generateMany(token, pos, h.slice(-64), h.length, count, ...settings, randoms, stops, ...narrowing, before.length);
      };
      const out = { form: engine.gpuReady?.tokens, attention: engine.gpuReady?.tokenAttention, pieces: engine.gpuReady?.tablePieces, first: ask(tokens[n], n, [], 4) };
      if (out.first && out.first.every((id, i) => id === greedy[i])) {
        // the CPU's step, on the keys and values the GPU wrote back; then the GPU's, from the CPU's position up
        engine.forward(greedy[3], n + 4);
        out.cpuStep = Array.from(engine.logits()).reduce((best, x, i, xs) => (x > xs[best] ? i : best), 0);
        out.second = ask(greedy[4], n + 5, greedy.slice(0, 5), 3);
      }
      const rows = engine.keysAndValues(n, 8);
      Object.assign(out, { keys: b64(rows.keys), values: b64(rows.values) });
      out.stopped = ask(tokens[n], n, [], 4, [0, 0.9, 1], [], [greedy[1]]);
      // the penalty: NumPy's first id in the history, 100 greedy
      out.penaltyHistory = [...tokens.slice(0, -1), greedy[0], tokens[n]];
      again(n);
      out.penalized = engine.generateMany(tokens[n], n, out.penaltyHistory.slice(-64), out.penaltyHistory.length, 1, 0, 0.9, 100, [], []);
      out.sampled = [0.02, 0.98].map((random) => ask(tokens[n], n, [], 1, [2, 0.999, 1], [random]));
      // T546: a top-k of 3 and min-p 0.5 at temperature 2 with random numbers of 0.02 and 0.98; the presence penalty
      // (100, greedy) on the history's latest 1 and 2 (NumPy's first id the second last: penalized as a sampled
      // token only in the second); and 4 steps with all three in one submission against the same 4 one at a time
      out.narrowed = {
        topK: [0.02, 0.98].map((random) => ask(tokens[n], n, [], 1, [2, 1, 1], [random], [], [3, 0, 0])),
        minP: [0.02, 0.98].map((random) => ask(tokens[n], n, [], 1, [2, 1, 1], [random], [], [0, 0.5, 0])),
        presence: [1, 2].map((written) => again(n) ?? engine.generateMany(tokens[n], n, out.penaltyHistory.slice(-64), out.penaltyHistory.length, 1, 0, 0.9, 1, [], [], 0, 0, 100, written)),
      };
      const narrowing = [20, 0.05, 1.5];
      out.narrowed.together = ask(tokens[n], n, [], 4, [2, 0.95, 1.1], [0.3, 0.7, 0.1, 0.9], [], narrowing);
      out.narrowed.apart = [];
      for (let i = 0; i < 4; i++) {
        const before = out.narrowed.apart;
        out.narrowed.apart.push(...(ask(i ? before[i - 1] : tokens[n], n + i, before.slice(0, i), 1, [2, 0.95, 1.1], [[0.3, 0.7, 0.1, 0.9][i]], [], narrowing) ?? [NaN]));
      }
      // (T152's review) 4 sampled steps with a penalty in one submission against the same 4 asked one at a time: the
      // GPU's arithmetic is the same, so the ids are to the bit (the random numbers taken in their order, and the
      // penalty's window handed over from the history at every request as the GPU carries it within a submission)
      const drawn = [2, 0.999, 1.3], numbers = [0.3, 0.7, 0.1, 0.9];
      out.together = ask(tokens[n], n, [], 4, drawn, numbers);
      out.apart = [];
      for (let i = 0; i < 4; i++) out.apart.push(...(ask(i ? out.apart[i - 1] : tokens[n], n + i, out.apart.slice(0, i), 1, drawn, [numbers[i]]) ?? [NaN]));
      // the CPU's keys and values going up: a prompt of other tokens through the GPU (its own cache then holds theirs),
      // the prompt through the CPU, and the GPU's step from its last token, which must see the CPU's of every position.
      // Twice, over two others (the review: the ids alone let a position left out pass, the GPU's own keys and values of
      // the same token being near the CPU's): the keys and values the step wrote back are then to the bit the same
      const upload = (other) => {
        engine.forwardMany(other, 0);
        engine.gpuSide = "cpu";
        engine.forwardMany(tokens.slice(0, -1), 0);
        engine.gpuSide = null;
        // (the CPU's logits of the prompt's last token on the CPU's keys and values, the answer: NumPy's is not, the
        // CPU's keys and values being up to 46% from NumPy's on the made-up models)
        engine.forward(tokens[n], n);
        rewound = "cpu";
        const logits = b64(engine.logits().slice()), id = ask(tokens[n], n, [], 1), rows = engine.keysAndValues(n, 1);
        rewound = null;
        return { logits, id, kv: b64(rows.keys) + b64(rows.values) };
      };
      // (the two others differ at every position, the last as well: the first try's two ended on the same token, whose
      // first layer's keys and values are the token's alone, and an upload that left the last position out passed on a
      // model of two layers)
      const reversed = tokens.slice(0, -1).reverse();
      const [one, two] = [reversed, [...reversed.slice(1), reversed[0]]].map(upload);
      Object.assign(out, { uploadedLogits: one.logits, uploaded: one.id, uploadedTwice: two.id, uploadedSame: one.kv === two.kv });
      return out;
    };
    // T219 (2): the sampler's refusal of logits that are not finite numbers, end to end from the GPU's own forward pass
    // (shaders.js's SAMPLE sets the State's not_finite word, gpu.js's generate() writes it after the ids, forward.js's
    // generateMany() reads it). The CPU's norm weights of layer 0 are made NaN (the GPU holds its own copy, uploaded as it
    // got ready), the prompt goes through the CPU (its keys and values are NaN then), and the GPU's step from there reads
    // those up: its logits are NaN. Asked after everything else a run does, for the GPU stops. The weights go back
    const refusal = (engine) => {
      const t = plan.tensors.rms_att_weight;
      if (t?.kind !== "f32") return { skipped: "the norm weights are not float32 here" };
      const norm = new Float32Array(memory.buffer, base + t.offset, plan.dim), kept = norm.slice();
      engine.newGeneration();
      engine.gpuSide = "cpu";
      norm.fill(NaN);
      engine.forwardMany(tokens.slice(0, -1), 0);
      norm.set(kept);
      engine.gpuSide = null;
      const taken = engine.generateMany(tokens[n], n, tokens.slice(-64), tokens.length, 4, 0, 0.9, 1, [], []);
      return { taken: taken ?? null, status: engine.gpuStatus };
    };
    // T241: the GPU's worker as forward.js is given it, with what forward.js posts seen on the way (seen.plan: the
    // start's plan, where the rows of a prompt's block and the ids of a step are; seen.tokens: the last request for
    // steps, where the CPU's cache is; seen.worker: the worker itself) and a block of a prompt changed before it goes
    // (seen.block), for the rounds below. gpu.js has no message that puts a number into what it holds: these rounds
    // change what forward.js hands it, in the shared memory
    const watched = (open, seen) => () => {
      const inner = open();
      seen.worker = inner;
      return { postMessage: (data, ...rest) => {
        if (data.type === "start") seen.plan = data.plan;
        if (data.type === "tokens") seen.tokens = data;
        if (data.type === "prompt") seen.block?.(data);
        inner.postMessage(data, ...rest);
      }, terminate: () => inner.terminate(), set onmessage(f) { inner.onmessage = f; }, set onerror(f) { inner.onerror = f; } };
    };
    // T241: a request of this harness's own to the GPU's worker, past forward.js (which stops the GPU at the first
    // refusal), waited for where gpu.js answers: [answered, failed]
    const own = (worker, message) => {
      const words = new Int32Array(memory.buffer, 0, GPU_WANTED + 1), serial = ownSerial++;
      Atomics.store(words, GPU_WANTED, serial);
      worker.postMessage({ ...message, serial });
      const until = performance.now() + 120000;
      for (let done = Atomics.load(words, GPU_DONE); done !== serial && performance.now() < until; done = Atomics.load(words, GPU_DONE)) {
        Atomics.wait(words, GPU_DONE, done, 1000);
      }
      return [Atomics.load(words, GPU_DONE) === serial, Atomics.load(words, GPU_FAILED)];
    };
    // T241: a block of a prompt with a row (the embedding of one token, which forward.js hands the GPU) that holds a
    // NaN, or an infinity: the stream of that token is no finite number from the first norm on, and its keys and
    // values of every layer are to come back so. A packed form quantizes the normed stream (QUANTIZE), whose float max
    // and i32() made zeros and finite numbers of it before T241, on lavapipe; with an infinity the norm's scale is 0
    // and the row holds one NaN (0 x inf) among zeros. They are read as the GPU wrote them back, float16 in the staging
    // place, by their bits. T243: forward.js refuses such a block and stops the GPU (refusedBlock below), so the
    // requests are this harness's own: a block through forward.js first (its rows are then where the GPU reads them),
    // the value put into a row, the same block asked again past forward.js, the value put back
    const brokenRows = (engine, seen) => {
      const P = seen.plan;
      if (!P?.rows) return { skipped: "no rows of a prompt's block in the shared memory" };
      engine.newGeneration();
      engine.forwardMany(tokens.slice(0, 16), 0);
      if (engine.gpuTokens !== 16) return { skipped: "the GPU took " + engine.gpuTokens + " tokens of a block of 16" };
      // (T545: the staging place holds the attending layers' keys and values alone)
      const cases = [], row = P.kvHeads * P.headSize, attending = P.kinds.filter((kind) => kind === "attention").length;
      for (const [value, t, j] of [[NaN, 3, 5], [Infinity, 15, plan.dim - 1]]) {
        const cell = new Float32Array(memory.buffer, P.rows + (t * plan.dim + j) * 4, 1), kept = cell[0];
        cell[0] = value;
        const [answered, failed] = own(seen.worker, { type: "prompt", count: 16, pos: 0 });
        cell[0] = kept;
        let finite = 0;
        for (let part = 0; part < 2 * attending; part++) {
          for (const half of new Uint16Array(memory.buffer, P.staging + (part * P.batch + t) * row * 2, row)) finite += (half & 0x7c00) !== 0x7c00;
        }
        cases.push({ value: String(value), token: t, at: j, answered, failed, finite, of: 2 * attending * row });
      }
      return { cases };
    };
    // T243: the same through forward.js, which is to refuse the block: the CPU's readers of a float16 make a finite
    // number of a NaN and of an infinity (kernels/kernel.ts's halves4: 65536 and more), so that forward.js looks at the
    // keys and values the GPU wrote back by their bits before any goes into the cache. Nothing of the block is taken
    // (no token counted on the GPU), the GPU stops and the status line says why, the CPU takes the block and the next:
    // the cache of both is the CPU's own, to the bit (held to the CPU's run below). The last a run asks of its GPU
    const refusedBlock = (engine, seen, [value, t, j]) => {
      const P = seen.plan;
      if (!P?.rows || !engine.gpuReady) return { skipped: "no GPU that takes a prompt's block" };
      engine.newGeneration();
      seen.block = () => { new Float32Array(memory.buffer, P.rows + (t * plan.dim + j) * 4, 1)[0] = value; };
      engine.forwardMany(tokens.slice(0, 16), 0);
      seen.block = null;
      const gpuTokens = engine.gpuTokens, status = engine.gpuStatus;
      engine.forwardMany(tokens.slice(16, 32), 16);
      const { keys, values } = engine.keysAndValues(0, 32);
      return { value: String(value), token: t, at: j, gpuTokens, after: engine.gpuTokens, status, keys: b64(keys), values: b64(values) };
    };
    // T241: a step over keys and values of which one value is a NaN, or an infinity: of the first layer and of the
    // last, the first column of the first position and the last column of the last (the attention's output then has
    // it in its first group and in its last: what the DP4A forms quantize before o). The prompt goes through the CPU
    // and one step through forward.js (which shows where the CPU's cache is: the request it posted); then the requests
    // are this harness's own, past forward.js (which stops the GPU at the first refusal): the value changed in the
    // CPU's cache, the positions sent up again from 0, one step, the answer read where gpu.js writes it (the ids and
    // after them T219's word), the value put back. Every form is to refuse every one (sampled 0, the word 1)
    const brokenValues = (engine, seen) => {
      engine.newGeneration();
      engine.gpuSide = "cpu";
      engine.forwardMany(tokens.slice(0, -1), 0);
      engine.gpuSide = null;
      const clean = engine.generateMany(tokens[n], n, tokens.slice(-64), tokens.length, 1, 0, 0.9, 1, [], []);
      const request = seen.tokens, cache = request?.cache, most = seen.plan?.tokens?.most;
      if (!clean || !cache) return { skipped: "no step on the GPU over the CPU's keys and values (" + JSON.stringify(clean ?? null) + ")" };
      const ids = new Int32Array(memory.buffer, seen.plan.tokens.ids, 2 + most);
      const width = cache.half ? 2 : 4, columns = cache.row / width, cases = [];
      for (const [value, half] of [[NaN, 0x7e00], [Infinity, 0x7c00]]) {
        // (T261.4.2: an attending layer's place in the CPU's cache, a block or a ring: the first and the last of them)
        for (const layer of [0, cache.places.length - 1]) {
          // (and of a layer that attends a window, a Gemma 3's local ones, the first position the step's window reads,
          // not 0: the step reads nothing before it)
          const reach = plan.layers.filter((one) => one.kind === "attention")[layer]?.window ?? 0;
          for (const [position, column] of [[reach ? Math.max(0, n + 1 - reach) : 0, 0], [n - 1, columns - 1]]) {
            const place = cache.places[layer], at = place.values + (place.ring ? position % place.ring : position) * cache.row + column * width;
            const cell = cache.half ? new Uint16Array(memory.buffer, at, 1) : new Float32Array(memory.buffer, at, 1), kept = cell[0];
            cell[0] = cache.half ? half : value;
            ids.fill(-2);
            const [answered, failed] = own(seen.worker, { ...request, count: 1, pos: n, from: 0 });
            cell[0] = kept;
            cases.push({ value: String(value), layer, position, column, answered, failed, sampled: ids[0], id: ids[1], notFinite: ids[1 + most] });
          }
        }
      }
      return { cases };
    };
    // T241: a step on a GPU that holds a weight that is no finite number, in the last layer (the GPU's worker checks
    // the first layer and the head against JavaScript on the same weights as it gets ready, and would refuse the form:
    // a weight of those cannot be changed so): a weight of the attention's norm (the normed stream then has it, which
    // the DP4A forms quantize before q, k and v), of the FFN's norm (before gate and up), a scale of w1 (a row of the
    // gate, and so the activation's output, quantized before down). The weight is changed in the shared memory before
    // the engine starts (the GPU copies the weights as it gets ready) and put back once the GPU is ready; the prompt
    // goes through the CPU, the step through forward.js, which is to refuse it (the status line says the logits were
    // not finite). An engine a case: the GPU stops at a refusal
    const brokenPlaces = () => {
      const T = plan.tensors, last = plan.n_layers - 1, w1 = T.w1, [rows, wide] = w1?.shape.slice(-2) ?? [], groups = wide / w1?.group;
      const float = (t, i) => (t?.kind === "f32" ? base + t.offset + (last * plan.dim + i) * 4 : 0);
      return [["a weight of the attention's norm (before q, k and v)", float(T.rms_att_weight, 7)],
        ["a weight of the FFN's norm (before gate and up)", float(T.rms_ffn_weight, plan.dim - 1)],
        // (T533: a ternary matrix's scales are float16s, 2 bytes each)
        ["a scale of w1 (before down)", w1?.scales ? base + w1.scales + ((last * rows + 3) * groups + 1) * (w1.kind === "ternary" ? 2 : 4) : 0, w1?.kind === "ternary"]]
        .filter(([, address]) => last > 0 && address);
    };
    // (a float16's bits of the two numbers written: NaN and Infinity)
    const halfBits = (value) => (Number.isNaN(value) ? 0x7e00 : value > 0 ? 0x7c00 : 0xfc00);
    const brokenWeight = async (form, [name, address, half], value, force) => {
      const cell = half ? new Uint16Array(memory.buffer, address, 1) : new Float32Array(memory.buffer, address, 1), kept = cell[0];
      cell[0] = half ? halfBits(value) : value;
      const engine = createForward({ memory, base, size, kernels, plan, gpu: openGpu, gpuForce: { ...TESTS, ...force, tokens: form } });
      const note = await engine.gpu;
      cell[0] = kept;
      engine.gpuSide = "cpu";
      engine.forwardMany(tokens.slice(0, -1), 0);
      engine.gpuSide = null;
      const asked = Boolean(engine.tokenBlock);
      const taken = asked ? engine.generateMany(tokens[n], n, tokens.slice(-64), tokens.length, 1, 0, 0.9, 1, [], []) : undefined;
      const out = { form, name, value: String(value), note, asked, why: engine.gpuTokensWhyNot, taken: taken ?? null, status: engine.gpuStatus };
      await engine.release();
      return out;
    };
    const run = async (gpu, gpuForce, gpuRemembered, steps = false) => {
      const started = performance.now(), seen = {};
      const engine = createForward({ memory, base, size, kernels, plan, gpu: gpu && watched(gpu, seen), gpuForce: { ...TESTS, ...gpuForce }, gpuRemembered });
      const note = gpu ? await engine.gpu : undefined;
      const readySeconds = (performance.now() - started) / 1000;
      const ready = engine.gpuReady;
      const began = performance.now();
      // blocks of 16 at first (T108's, as Python handed them before T147): each block sees the keys and values the
      // GPU keeps of the ones before it, and the caches grow between them
      for (let at = 0; at < n; at += 16) engine.forwardMany(tokens.slice(at, Math.min(at + 16, n)), at);
      const promptMs = performance.now() - began, gpuTokens = engine.gpuTokens;
      engine.forward(tokens[n], n);
      const logits = engine.logits().slice(), { keys, values } = engine.keysAndValues(0, n);
      const out = { note, ready, form: engine.gpuForm, attention: engine.gpuAttention, promptMs, gpuTokens, logits: b64(logits), keys: b64(keys), values: b64(values) };
      // T152: a generation's steps on the GPU, where this run takes them (steps)
      if (gpu && steps) out.steps = { ...generation(engine), forced: gpuForce.tokens, cut: Boolean(gpuForce.tablePieceBytes) && plan.vocab_size >= 192 };
      if (gpu) {
        // the same prompt again from position 0, all of it at once (one block of the GPU's): the GPU's keys and values
        // of the first run are written over
        engine.newGeneration();
        engine.forwardMany(tokens.slice(0, -1), 0);
        engine.forward(tokens[n], n);
        const again = engine.keysAndValues(0, n);
        out.again = { gpuTokens: engine.gpuTokens, logits: b64(engine.logits().slice()), keys: b64(again.keys), values: b64(again.values) };
        // a block that begins past what the GPU holds (the CPU wrote position n) goes to the CPU
        engine.newGeneration();
        engine.forwardMany(tokens.slice(0, 2), n + 1);
        out.past = { gpuTokens: engine.gpuTokens };
        if (nan) out.brokenRows = timed("rows", () => brokenRows(engine, seen));
      }
      if (gpu && steps && out.steps && !out.steps.why) {
        if (nan) out.steps.brokenValues = timed("values", () => brokenValues(engine, seen));
        out.steps.nanLogits = timed("logits", () => refusal(engine));
      } else if (gpu && nan) {
        // T243 (the runs whose last request is not refusal()'s: either stops the GPU; T241's review: where this browser
        // puts a NaN in this model's rows, NAN_ROUNDS)
        out.refusedBlock = timed("block", () => refusedBlock(engine, seen, [[NaN, 0, 5], [Infinity, 15, plan.dim - 1], [NaN, 15, 0], [Infinity, 0, 3]][refusals++ % 4]));
      }
      // T205: the GPU's worker says it let go of its device before the next model is read (false: not within 5 s)
      out.ended = await engine.release();
      // T183: the seconds of the run, and of those until the GPU's worker said it was ready (its shaders compiled)
      return { ...out, seconds: (performance.now() - started) / 1000, readySeconds };
    };
    // T147: a GPU that answers late. forward.js gives its request up (stalledMs 1: the GPU's worker counts every 250
    // ms) and runs the block itself; the stop it posts never reaches the worker (a worker that went on before it read
    // the stop). When the worker is done (its count stands for a second) it must have answered nothing: forward.js
    // wanted the request no more, and the memory could be the next model's by then
    const late = async () => {
      let inner;
      const stopLost = () => {
        inner = openGpu();
        return { postMessage: (data) => data.type !== "stop" && inner.postMessage(data), set onmessage(f) { inner.onmessage = f; },
          set onerror(f) { inner.onerror = f; } };
      };
      const engine = createForward({ memory, base, size, kernels, plan, gpu: stopLost, gpuForce: TESTS, stalledMs: 1 });
      const note = await engine.gpu;
      const words = new Int32Array(memory.buffer, 0, GPU_WANTED + 1);
      engine.forwardMany(tokens.slice(0, -1), 0);
      // T148: the CPU did the blocks the GPU gave up (the same numbers as the CPU's own run: its blocks of 16)
      const cpuKeys = b64(engine.keysAndValues(0, n).keys);
      const gpuTokens = engine.gpuTokens;
      for (let beat = -1, still = 0, waited = 0; still < 1000 && waited < 120000; waited += 100) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        still = Atomics.load(words, GPU_BEAT) === beat ? still + 100 : 0;
        beat = Atomics.load(words, GPU_BEAT);
      }
      const out = { note, gpuTokens, keys: cpuKeys, done: Atomics.load(words, GPU_DONE), failed: Atomics.load(words, GPU_FAILED) };
      engine.release();
      inner.terminate();
      return out;
    };
    // T156: the model on the GPU alone, as the worker puts one it cannot hold twice (forward.js's gpuOnlyWeights): the
    // checkpoint streamed in stretches of odd sizes (a stretch ends in the middle of a word, a matrix, a layer), the
    // layers' matrices to the GPU's worker and the rest into memory packed without them. The prompt and 4 greedy steps
    // (none on the CPU); then a step on the CPU, which must throw (the weights are not here). And a GPU that fails as
    // it opens (pieces of 4096 bytes, fewer than the device's alignment of rows: gpu.js's piecesOf refuses): the bytes
    // still counted as taken, the GPU lost with the reason as the engine starts, a step on the CPU throws. cpu (the made-up
    // model only): /benchmark/'s CPU reading made up, far slower than the GPU (it stays) or far faster (the GPU is given
    // up as it is ready, and the verdict to keep has the device's key)
    const direct = async (force, cpu) => {
      const worker = openGpu();
      const weights = gpuOnlyWeights({ memory, base, size, tensors: c.places, worker });
      // (T271.4: by the model's form, a Qwen3.5's layers of two kinds)
      worker.postMessage({ type: "open", plan: gpuOnlyPlan(c.reference.header, c.places, force, undefined, c.options), flow: weights.flow });
      const began = performance.now(), stretches = [1234567, 777, 3, 65536, 1 << 20];
      for (let at = 0, k = 0; at < size; at += stretches[k++ % stretches.length]) {
        weights.write(at, checkpoint.subarray(at, Math.min(size, at + stretches[k % stretches.length])));
        await weights.room();
      }
      await weights.drained();
      let lostWhy = null;
      const told = { place: weights.place, stored: weights.stored, onLost: (why) => { lostWhy = why; }, size, cpu,
        layerWeights: layerWeightsOf(c.reference.header, { head_dim: c.headDim }), usage: { prompt: 1, written: 1 } };
      const outside = external({ memory, base, size, kernels, gpu: () => worker, gpuForce: force, halfKeys: true, direct: told });
      const engine = outside.start(structuredClone(plan));
      const note = await engine.gpu;
      const out = { note, stored: weights.stored };
      if (!lostWhy) {
        for (let at = 0; at < n; at += 16) engine.forwardMany(tokens.slice(at, Math.min(at + 16, n)), at);
        const { keys, values } = engine.keysAndValues(0, n);
        Object.assign(out, { keys: b64(keys), values: b64(values), gpuTokens: engine.gpuTokens, status: engine.gpuStatus,
          first: engine.generateMany(tokens[n], n, tokens.slice(-64), tokens.length, 4, 0, 0.9, 1, [], []) });
      }
      out.lost = lostWhy;  // (before the step on the CPU below, which loses the GPU: the worker would load it again)
      out.verdict = told.verdict;
      try {
        engine.forward(tokens[n], n);
      } catch (error) {
        out.threw = String(error?.message ?? error);
      }
      out.seconds = (performance.now() - began) / 1000;
      engine.release();
      return out;
    };
    const gpu = [];
    // (the forced ones untimed, T153: a block of 64 tokens of Qwen3 0.6B took more than the 180 s of a step on lavapipe)
    for (const form of [undefined, ...forms]) gpu.push(await run(openGpu, form ? { matrices: form, quick: true } : {}, undefined, !form));
    // T152: every form of a token's layer, forced, where the model's steps go to the GPU (T226: every made-up model's;
    // a model with LayerNorm has no DP4A form with the norm in the quantizer, NORM_QUANTIZE being RMSNorm's: gpu.js's
    // tokenCandidates); in one piece each (a token's layer reads a matrix whole: the first run's pieces, T155, left the steps on
    // the CPU)
    // T209: the classifier and the embedding in pieces of about a third of the table (as a table past what the device
    // binds: Llama 3.2 3B's on the owner's Android), so that EMBED's and the classifier's pieces are what the steps read
    if (gpu[0].steps?.planned !== false) {
      for (const form of tokenForms.filter((name) => c.arch === "llama" || name !== "DP4A, fused (T175)")) gpu.push(await run(openGpu, { matrices: forms[0], quick: true, tokens: form, pieceBytes: Infinity, tablePieceBytes }, undefined, true));
      for (const attention of tokenAttentions) {
        gpu.push(await run(openGpu, { matrices: forms[0], quick: true, tokens: tokenForms[0], tokenAttention: attention, pieceBytes: Infinity, tablePieceBytes }, undefined, true));
      }
    }
    // T241: the DP4A forms on a GPU with a weight of the last layer that is no finite number (brokenWeight), on three
    // of the made-up models (Llama's form, Qwen3's norms of the heads, GPT-2's LayerNorm and GELU: an engine a case)
    const broken = [];
    // (T232: and the ternary one, whose matrices multiply the same quantized vectors: T241's NaN scales carry on there too)
    if (NAN_ROUNDS.weights && gpu[0].steps?.planned !== false && ["synthetic", "synthetic-qwen3", "synthetic-gpt2", "synthetic-ternary"].includes(c.id)) {
      const force = { matrices: forms[0], quick: true, pieceBytes: Infinity }, weightsBegan = performance.now();
      for (const form of tokenForms.filter((name) => /DP4A/.test(name) && (c.arch === "llama" || name !== "DP4A, fused (T175)"))) {
        for (const place of brokenPlaces()) for (const value of [NaN, Infinity]) broken.push(await brokenWeight(form, place, value, force));
      }
      nanSeconds.weights = (performance.now() - weightsBegan) / 1000;
    }
    // the attention without subgroups or f16 (the lanes of the workgroup stand for a subgroup), where the adapter
    // has them and so chose the other
    if (forms.length) gpu.push(await run(openGpu, { matrices: forms[0], attention: "llama.cpp flash attention tiles", quick: true }));
    // T148 (the made-up model only): a fallback adapter refused as the page refuses it, before anything is compiled;
    // the shaders of the first run remembered for this adapter (its key), which are then the only ones compiled, and
    // not for another key
    let refused, remembered;
    if (c.id === "synthetic") {
      const began = performance.now(), engine = createForward({ memory, base, size, kernels, plan, gpu: openGpu });
      refused = { note: await engine.gpu, seconds: (performance.now() - began) / 1000, ended: await engine.release() };
      // T205: a model let go while its GPU is still getting ready (gpu.js's start() stops at its next step and says so)
      const starting = createForward({ memory, base, size, kernels, plan, gpu: openGpu, gpuForce: TESTS });
      const stopped = performance.now();
      refused.whileStarting = { ended: await starting.release(), seconds: (performance.now() - stopped) / 1000 };
      const first = gpu[0].ready ?? {}, kept = { key: first.key, matrices: first.matrices, attention: first.attention };
      remembered = { same: (await run(openGpu, {}, kept)).ready, other: (await run(openGpu, {}, { ...kept, key: kept.key + "|another" })).ready };
    }
    const cpu = await run(undefined), lateRun = c.id === "synthetic" ? await late() : undefined;
    // T156: the model on the GPU alone, last (it packs the memory the others read the checkpoint whole from)
    let alone;
    const tokenRun = gpu.find((r) => r.steps?.forced === tokenForms[0] && r.steps?.first);
    if (c.places && tokenRun && !c.wide && !c.force) {
      const force = { ...TESTS, matrices: forms[0], quick: true, tokens: tokenForms[0], pieceBytes: Infinity, tablePieceBytes };
      const synthetic = c.id === "synthetic";
      alone = { right: await direct(force, synthetic ? { GBps: 1e-6, promptGMACs: 1e-6 } : undefined), pieces: await direct({ ...force, pieceBytes: 4096 }),
        // (timed: the GPU's step and block are what the verdict weighs, which the tests' quick leaves untimed)
        ...(synthetic ? { cpuFaster: await direct({ ...force, quick: false }, { GBps: 1e9, promptGMACs: 1e9 }), key: adapter && wgsl.deviceKey(adapter) } : {}),
        keys: tokenRun.keys, values: tokenRun.values, first: tokenRun.steps.first };
    }
    results.push({ id: c.id, cpu, gpu, late: lateRun, refused, remembered, alone, broken, nanSeconds });
  }
  postMessage({ results, forms: formsOf(false), quantizers });
} catch (error) {
  postMessage({ error: String(error?.stack ?? error) });
}
