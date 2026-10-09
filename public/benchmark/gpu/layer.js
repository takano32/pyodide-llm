// The steps "layer" and "layer steps": ms a layer by every form (T150, T175), where the time of a layer goes (T202, T208),
// its attention by the cache's length, and the whole layer by timestamps.
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)
const { shared, MODELS, matrixBytes, gpu, STORAGE, COPY_DST, UNIFORM, COPY_SRC, buffer, floats, pipelinesFor, run,
  readBack, validated, PAIRS, middle, interleaved, scoped } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { toHalf } = await import(new URL(`halves.js${new URL(import.meta.url).search}`, import.meta.url));
const { LAYER_POS, LAYER_MOST, layerShape, LAYER_KINDS, layerForms, hasSubgroupId, engineTiles, vecAttention,
  attentionSteps, layerCheck, compiled, layerPipes, layerParts, MATRIX_KEYS } = await import(new URL(`layerparts.js${new URL(import.meta.url).search}`, import.meta.url));
const { MATVEC_BYTES, MATVEC_MOST } = await import(new URL(`matvec.js${new URL(import.meta.url).search}`, import.meta.url));

async function layer() {
  await gpu();
  const shape = layerShape(MODELS["Llama 3.2 1B"]);
  const bytes = Object.values(shape.matrices).reduce((sum, matrix) => sum + matrixBytes(matrix), 0);
  const copies = shared.fallback ? 1 : Math.ceil(MATVEC_BYTES / bytes), rows = [];
  await scoped(async (owned) => {
    const parts = layerParts(shape, LAYER_POS, copies, owned);
    await shared.device.queue.onSubmittedWorkDone();
    // the forms that can run here, each with what submits n layers of it (each on the next copy) and waits
    const timed = [];
    for (const form of layerForms()) {
      const row = { form: form.name, check: layerCheck(form), base: form.base, fused: form.fused, subgroups: form.subgroups, ...(form.normApart ? { normApart: true } : {}) };
      if (form.none) {
        timed.push({ row: { ...row, none: form.none } });
        continue;
      }
      try {
        const pipes = await layerPipes(shape, form);
        const each = await validated(async () => [...Array(copies)].map((_, copy) => parts.dispatches(form, pipes, copy)));
        let next = 0;
        const submission = async (n) => {
          const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
          for (let i = 0; i < n; i++) each[next++ % copies].forEach((d) => run(pass, d));
          pass.end();
          const began = performance.now();
          shared.device.queue.submit([encoder.finish()]);
          await shared.device.queue.onSubmittedWorkDone();
          return performance.now() - began;
        };
        row.dispatches = each[0].length;
        timed.push({ row, submission });
      } catch (error) {
        timed.push({ row: { ...row, error: String(error?.message ?? error) } });
      } finally {
        postMessage({ alive: true });
      }
    }
    const running = timed.filter((t) => t.submission);
    try {
      const results = await validated(() => interleaved(running.map((t) => t.submission), LAYER_MOST));
      running.forEach((t, i) => {
        const r = results[i];
        Object.assign(t.row, r.error ? { error: r.error } : { msPerLayer: r.ms / r.dispatches, layers: r.dispatches, GBps: bytes / (r.ms / r.dispatches) / 1e6,
          ...(r.ratio ? { ratio: r.ratio } : {}), ...(r.unsteady ? { unsteady: true } : {}) });
      });
    } catch (error) {
      running.forEach((t) => (t.row.error = String(error?.message ?? error)));
    }
    rows.push(...timed.map((t) => t.row));
  });
  shared.layerTimes = rows;
  return { model: "Llama 3.2 1B", pos: LAYER_POS, layers: MODELS["Llama 3.2 1B"].layers, copies, GB: bytes / 1e9, rows };
}

// ---- T202: where the time of a layer goes (T152's review: on the owner's Android the fastest layer took 3.36 ms, its
// matrices read at the matrix × vector's 38.6 GB/s would take 1.77, and what the other 1.59 ms is was not measured).
// The fused forms a token would run here (stepForms(), T208: the layer table's fastest fused form the check found
// right, as the engine chooses, T152, and beside it the same with the norms folded or apart), and for each:
//   - every step of it alone (one of its dispatches, the same name counted where two cost the same),
//   - its four matrices alone (the plain matrix × vector over all of a matrix's rows, nothing folded into the write:
//     the "1.77 ms"),
//   - a dispatch of one workgroup that does next to nothing (what a step costs in a chain for being a dispatch),
//   - the whole layer,
// all timed in turn (interleaved(): n of each a submission, 2n less n, the rounds taken of every item alike). The
// layer on the next copy of the weights each time, as layer() times it; a step or a matrix alone on the next range of
// its size of all the copies' weights and the spares (layerParts' ranges(): T202's review, T149's, T208), the
// attention alone on the next of its caches (caches(), T208), so that none is read again before MATVEC_BYTES of others
// (not from the GPU's caches).
// Why each step alone and not the layer less one step: a step of 20 to 60 µs is 1 or 2% of a layer, about what a
// layer's time moves from one pair to the next, so a difference of two layers could not tell it; alone it is repeated
// until a submission takes SUBMISSION_MS. What alone leaves out, the layer less the sum of its steps says: what the
// chain of different dependent dispatches costs beyond each on its own (a small step on a GPU full of the matrix's
// workgroups waits for their tail). Each dispatch still waits for the one before it (the same buffers written), as
// in the layer. The residual stream is written back before every submission (the adds write over it), so that every
// submission starts alike; the whole layer's attention reads the one cache of positions up to LAYER_POS (256 KB of
// keys and values), as in layer(): between two reads of it, a copy's weights (68 MB).
// T208: then the whole layer once more by the GPU's own clock where the device gives timestamp-query (timestamps()).
async function layerSteps() {
  await gpu();
  const shape = layerShape(MODELS["Llama 3.2 1B"]);
  const bytes = Object.values(shape.matrices).reduce((sum, matrix) => sum + matrixBytes(matrix), 0);
  const copies = shared.fallback ? 1 : Math.ceil(MATVEC_BYTES / bytes);
  const { forms, chosen } = stepForms();
  const result = { model: "Llama 3.2 1B", pos: LAYER_POS, layers: MODELS["Llama 3.2 1B"].layers, copies, GB: bytes / 1e9, dp4a: Boolean(forms[0]?.dp4a), chosen };
  if (!forms.length) return { ...result, forms: [], steps: [], lengths: await attentionLengths(shape) };
  const out = await scoped(async (owned) => {
    const parts = layerParts(shape, LAYER_POS, copies, owned, undefined, shared.fallback ? 0 : MATVEC_BYTES);
    const caches = parts.caches();
    // the fewest MB of weights read before the same ones again: the layer's copies or a matrix's ranges (the
    // attention's caches are their own line, result.caches)
    result.cycleMB = Math.min(copies * bytes, ...MATRIX_KEYS.map((key) => parts.ranges(key).length * matrixBytes(shape.matrices[key]))) / 1e6;
    result.spares = parts.spares;
    result.caches = { count: caches.length, MB: (caches.length * 2 * caches[0].keys.size) / 1e6 };
    await shared.device.queue.onSubmittedWorkDone();
    // units: the dispatches of one unit each, on a copy or a range of the weights, taken in turn; a submission of n
    // units, the stream written back first
    const timing = (units) => {
      let next = 0;
      return async (n) => {
        parts.restart();
        const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
        for (let i = 0; i < n; i++) units[next++ % units.length].forEach((d) => run(pass, d));
        pass.end();
        const began = performance.now();
        shared.device.queue.submit([encoder.finish()]);
        await shared.device.queue.onSubmittedWorkDone();
        return performance.now() - began;
      };
    };
    // what is timed: a step { step, kind, matrix } or a form's layer { form }, each with its submission
    const items = [], rows = [], seen = new Set(), layers = [];
    const add = (item, units) => items.push({ ...item, submission: timing(units) });
    for (const form of forms) {
      const row = { form: form.name, check: layerCheck(form), normApart: Boolean(form.normApart) };
      rows.push(row);
      try {
        const pipes = await layerPipes(shape, form);
        const each = await validated(async () => [...Array(copies)].map((_, copy) => parts.dispatches(form, pipes, copy)));
        row.dispatches = each[0].length;
        // how many of each step a layer dispatches, in the layer's order
        row.steps = [];
        each[0].forEach(({ step }) => {
          const known = row.steps.find((one) => one.step === step);
          if (known) known.count++;
          else row.steps.push({ step, count: 1 });
        });
        // each step not timed yet (the first of its name): a matrix's on every range of its size in turn (its other
        // bindings as the layer's), the attention's on every cache in turn, the others' on the copies as the layer
        const fresh = each[0].map((d, k) => ({ d, k })).filter(({ d, k }) => !seen.has(d.step) && each[0].findIndex((e) => e.step === d.step) === k);
        const stepUnits = await validated(async () => fresh.map(({ d: { matrix, kind }, k }) => (matrix
          ? parts.ranges(matrix).map((range) => [parts.dispatches(form, pipes, 0, { [matrix]: range })[k]])
          : kind === "attention" ? caches.map((cache) => parts.attention(form, pipes, cache).filter((one) => one.step === each[0][k].step))
          : each.map((layer) => [layer[k]]))));
        add({ form: form.name }, each);
        layers.push({ form: form.name, each });
        fresh.forEach(({ d: { step, kind, matrix } }, i) => {
          seen.add(step);
          add({ step, kind, ...(matrix ? { matrix } : {}) }, stepUnits[i]);
        });
      } catch (error) {
        row.error = String(error?.message ?? error);
      } finally {
        postMessage({ alive: true });
      }
    }
    if (!items.length) return { ...result, forms: rows, steps: [] };
    // the matrices alone (on the plain matrix × vector of the forms' base, with their subgroups or not) and the floor
    // of a dispatch
    const product = await compiled(forms[0].dp4a ? shared.WGSL.ortDp4aMatVec : shared.WGSL.mulMatVec({ packed: false, subgroups: forms[0].subgroups }));
    const alone = await validated(async () => MATRIX_KEYS.map((key) => parts.ranges(key).map((range) => [parts.alone(forms[0], product, key, range)])));
    alone.forEach((units, i) => add({ step: units[0][0].step, kind: "alone", matrix: MATRIX_KEYS[i] }, units));
    const floor = parts.floor(pipelinesFor().small);
    add({ step: floor.step, kind: floor.kind }, [[floor]]);
    postMessage({ alive: true });
    const results = await validated(() => interleaved(items.map((item) => item.submission), MATVEC_MOST));
    const timed = (r) => (r.error ? { error: r.error }
      : { ms: r.ms / r.dispatches, n: r.dispatches, ratio: r.ratio, ...(r.unsteady ? { unsteady: true } : {}) });
    const steps = [];
    items.forEach((item, i) => {
      if (item.form) Object.assign(rows.find((row) => row.form === item.form), timed(results[i]));
      else steps.push({ step: item.step, kind: item.kind, ...(item.matrix ? { matrix: item.matrix } : {}), ...timed(results[i]) });
    });
    postMessage({ alive: true });
    return { ...result, forms: rows, steps, timestamps: await timestamps(parts, layers) };
  });
  return { ...out, lengths: await attentionLengths(shape) };
}
// T224: a token's attention alone at ATTENTION_LENGTHS positions (the owner asked for long contexts too: the prompt's
// tiles run a workgroup a head at any length, flash_attn_vec splits the positions over more of them as they grow), each
// attention the steps' table could meet (the prompt's tiles, flash_attn_vec with subgroups where there are, and with
// the lanes of the workgroup standing for a subgroup), all in turn at a length (interleaved()), each on the next of
// copies of a cache of that length (the same random float16 keys and values in each, MATVEC_BYTES of them: not from the GPU's
// caches, as T208's attention alone), one length at a time (4096 positions of Llama 3.2 1B's keys and values are 8.4 MB
// a copy). T224's review: the prompt's tiles twice where the engine makes them otherwise here (engineTiles(): f16 and
// subgroups), the f32 tiles without subgroups of the layer rows and the engine's, which a token's attention is chosen
// against and the vec rows are read against (base). { positions, rows: [{ attention, tiles, times: [{ ms, n, ratio,
// unsteady }, { error } or { none } a length] }], base, MB } or { error }
const ATTENTION_LENGTHS = [128, 1024, 4096];
async function attentionLengths(shape) {
  const { heads, kvHeads, headSize, kvDim } = shape;
  const engine = engineTiles(headSize);
  const forms = [{ name: "the prompt's tiles (flash attention's tile)", tiles: true }, ...(engine ? [{ name: engine.name, tiles: true, engine: true }] : []),
    ...(hasSubgroupId() ? [true] : []).map(() => ({ name: vecAttention(true).name, attention: "vec", vecSubgroups: true })),
    { name: vecAttention(false).name, attention: "vec", vecSubgroups: false }];
  const rows = forms.map((form) => ({ attention: form.name, ...(form.tiles ? { tiles: true } : {}), times: [] }));
  const base = engine ? 1 : 0;
  let MB = 0;
  try {
    const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = shared.device.limits;
    const flash = shared.WGSL.flashShape({ headSize, half: false, subgroups: false, memory, threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) });
    if (flash.none) throw new Error(flash.none);
    // (one at a time: each in an error scope of its own)
    const pipes = [];
    for (const form of forms) {
      const made = { flash: await compiled(shared.WGSL.flashTile(form.engine ? engine.shape : flash)) };
      if (form.attention === "vec") {
        made.vecShape = vecAttention(form.vecSubgroups).shape(headSize);
        if (made.vecShape.none) {
          pipes.push({ none: made.vecShape.none });
          continue;
        }
        made.vec = await compiled(shared.WGSL.flashVec(made.vecShape));
        made.vecReduce = await compiled(shared.WGSL.flashVecReduce(made.vecShape));
      }
      pipes.push(made);
    }
    for (const positions of ATTENTION_LENGTHS) {
      await scoped(async (owned) => {
        const make = (bytes, usage = STORAGE | COPY_DST) => {
          const b = buffer(bytes, usage);
          owned.push(b);
          return b;
        };
        const uniform = (bytes) => {
          const b = make(bytes.byteLength, UNIFORM | COPY_DST);
          shared.device.queue.writeBuffer(b, 0, bytes);
          return b;
        };
        const cacheBytes = positions * kvDim * 2, stride = Math.ceil(cacheBytes / 256) * 256;
        const count = shared.fallback ? 1 : Math.max(1, Math.ceil(MATVEC_BYTES / (2 * stride)));
        const all = make(2 * stride * count);
        const pattern = () => new Uint16Array(positions * kvDim).map(() => toHalf((Math.random() - 0.5) * 4));
        const [keys, values] = [pattern(), pattern()];
        for (let i = 0; i < count; i++) {
          shared.device.queue.writeBuffer(all, 2 * i * stride, keys);
          shared.device.queue.writeBuffer(all, (2 * i + 1) * stride, values);
        }
        MB = Math.max(MB, (2 * stride * count) / 1e6);
        const caches = [...Array(count)].map((_, i) => ({ keys: { buffer: all, offset: 2 * i * stride, size: cacheBytes },
          values: { buffer: all, offset: (2 * i + 1) * stride, size: cacheBytes } }));
        const v = { q: make(heads * headSize * 4), att: make(heads * headSize * 4),
          parts: make(Math.max(...[true, false].map((sub) => shared.WGSL.flashVecPartsBytes(vecAttention(sub).shape(headSize), heads)))) };
        shared.device.queue.writeBuffer(v.q, 0, floats(heads * headSize, 2));
        const flashParams = new ArrayBuffer(16);
        new Uint32Array(flashParams, 0, 2).set([heads, kvHeads]);
        new Float32Array(flashParams, 8, 1)[0] = 1 / Math.sqrt(headSize);
        const vecParams = new Map();
        const u = { step: uniform(new Uint32Array([1, positions - 1, 0, 0])), positions, flash: uniform(new Uint8Array(flashParams)),
          vecParams: (nwg) => vecParams.get(nwg) ?? vecParams.set(nwg, uniform(shared.WGSL.flashVecParams({ headSize }, heads, kvHeads, nwg))).get(nwg) };
        const group = (pipeline, entries) => shared.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: entries.map(([binding, resource]) => ({ binding, resource: "offset" in resource ? resource : { buffer: resource } })) });
        const timing = (units) => {
          let next = 0;
          return async (n) => {
            const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
            for (let i = 0; i < n; i++) units[next++ % units.length].forEach((d) => run(pass, d));
            pass.end();
            const began = performance.now();
            shared.device.queue.submit([encoder.finish()]);
            await shared.device.queue.onSubmittedWorkDone();
            return performance.now() - began;
          };
        };
        const running = forms.map((form, i) => (pipes[i].none ? null
          : timing(caches.map((cache) => attentionSteps(form, pipes[i], heads, cache, v, u, group)))));
        const results = await validated(() => interleaved(running.filter(Boolean), MATVEC_MOST));
        let k = 0;
        running.forEach((submission, i) => {
          if (!submission) return rows[i].times.push({ none: pipes[i].none });
          const r = results[k++];
          rows[i].times.push(r.error ? { error: r.error } : { ms: r.ms / r.dispatches, n: r.dispatches, ratio: r.ratio, ...(r.unsteady ? { unsteady: true } : {}) });
        });
        postMessage({ alive: true });
      });
    }
    // (bytes: the keys and values of a length, read once: float16, two of them; src/bench.js turns the time into GB/s)
    return { positions: ATTENTION_LENGTHS, rows, base, MB, bytes: ATTENTION_LENGTHS.map((positions) => positions * kvDim * 2 * 2) };
  } catch (error) {
    return { error: String(error?.message ?? error) };
  }
}
// T208: the fused forms the steps' table breaks down. The engine runs the fastest fused layer the check found right
// (T152: T150's fusedMatVec, with subgroupAdd where subgroups are, and T175's fusedDp4aMatVec where the packed int8
// dot is), so: the layer table's fastest fused row (layer(), which the page runs just before in this worker) that the
// check found right, and beside it the same form with the norms folded or apart where there is one (the two differ
// only in their norms, and how far their layers less those come apart is how far the run's times move). T202 took
// DP4A's two where the packed int8 dot is and llama.cpp's elsewhere, which on a device whose fastest layer is another
// (Apple, where ONNX Runtime does not use DP4A, or llama.cpp's with subgroups) broke down a layer no token runs. Where
// the layer table gave no time (it failed, or ran in no worker before), as T202.
// The fastest only of the forms the engine has (public/gpu.js's TOKEN_FORMS; T208's review): not llama.cpp's with the
// norms apart (withoutDp4a), which the layer table times where there is no packed int8 dot but no token runs; it can
// still be the partner.
function stepForms() {
  const fused = layerForms().filter((form) => form.fused && !form.none), engine = fused.filter((form) => !form.withoutDp4a);
  const fastest = (shared.layerTimes ?? []).filter((row) => row.fused && row.msPerLayer > 0 && shared.layerVerdicts?.[row.check]?.ok === true)
    .sort((a, b) => a.msPerLayer - b.msPerLayer).map((row) => engine.find((form) => form.name === row.form)).find(Boolean);
  if (!fastest) {
    const forms = LAYER_KINDS.filter((kind) => kind.fused && Boolean(kind.dp4a) === shared.packed).map((kind) => ({ ...kind, subgroups: false }));
    return { forms, chosen: { by: "packed", why: shared.layerTimes ? "the layer table has no fused layer timed and found right" : "no layer table was timed before" } };
  }
  const partner = fused.find((form) => form !== fastest && form.base === fastest.base && form.subgroups === fastest.subgroups && form.attention === fastest.attention &&
    Boolean(form.normApart) !== Boolean(fastest.normApart));
  // in the layer table's order (the norms apart first)
  return { forms: fused.filter((form) => form === fastest || form === partner), chosen: { by: "layer", fastest: fastest.name } };
}
// T208 (T202's review's (d)): the whole layer by the GPU's own clock, each form's: a submission of TIMESTAMP_LAYERS
// layers (on the copies in turn, as layer() times them), each layer a compute pass of its own that writes a timestamp
// as it begins and as it ends (llama.cpp's GGML_WEBGPU_GPU_PROFILE writes them a dispatch a pass). The forms in turn,
// PAIRS rounds, a form's ms a layer the median of its rounds' means. A check of the whole layer only, one line of the
// table: Chrome's Dawn cuts every timestamp down to a multiple of 65.5 µs (2^16 ns; Chrome's own words say 100 µs;
// unless its developer features are on), too coarse for a step of 20 to 60 µs, and each layer's time is off by up to
// that much. T208's review: the mean of the passes is off by 65.5 µs / (2 √32) = 6 µs or less (its spread) where the
// passes start anywhere on that clock's steps (their times are not a multiple of it), and span (the first pass's
// beginning to the last one's end, over the layers) by 65.5 µs / 32 = 2 µs or less whatever they do, the cuts of the
// passes between cancelling. span also holds what comes between two passes; a mean above it says the passes overlapped
// (a pass stamped as begun before the one before it ended), and then the mean is not the layers' work. Against the
// submissions' times: those hold the wait of a submission, taken out as the difference of 2n and n, while these hold
// only what runs on the GPU. { none } where the device gives no timestamp-query, { error } where it failed.
const TIMESTAMP_LAYERS = 32;
async function timestamps(parts, layers) {
  if (!shared.device.features.has("timestamp-query")) return { none: "this device gives no timestamp-query (the GPU's own clock) to this page" };
  if (!layers.length) return { none: "no form's layer ran" };
  const count = shared.fallback ? 1 : TIMESTAMP_LAYERS, rounds = shared.fallback ? 1 : PAIRS;
  try {
    return await scoped(async (owned) => {
      const set = shared.device.createQuerySet({ type: "timestamp", count: 2 * count });
      const resolved = buffer(16 * count, 0x200 | COPY_SRC);  // GPUBufferUsage.QUERY_RESOLVE
      owned.push(resolved, { destroy: () => set.destroy() });
      const means = layers.map(() => []), spans = layers.map(() => []);
      for (let round = 0; round < rounds; round++) {
        for (const [i, { each }] of layers.entries()) {
          parts.restart();
          const encoder = shared.device.createCommandEncoder();
          for (let l = 0; l < count; l++) {
            const pass = encoder.beginComputePass({ timestampWrites: { querySet: set, beginningOfPassWriteIndex: 2 * l, endOfPassWriteIndex: 2 * l + 1 } });
            each[(round * count + l) % each.length].forEach((d) => run(pass, d));
            pass.end();
          }
          encoder.resolveQuerySet(set, 0, 2 * count, resolved, 0);
          const stamps = new BigInt64Array(await readBack(encoder, resolved, 16 * count));
          let ns = 0;
          for (let l = 0; l < count; l++) ns += Number(stamps[2 * l + 1] - stamps[2 * l]);
          means[i].push(ns / count / 1e6);
          spans[i].push(Number(stamps[2 * count - 1] - stamps[0]) / count / 1e6);
        }
        postMessage({ alive: true });
      }
      return { layers: count, rounds, forms: layers.map(({ form }, i) => ({ form, ms: middle(means[i]), span: middle(spans[i]) })) };
    });
  } catch (error) {
    return { error: String(error?.message ?? error) };
  }
}

export { layer, layerSteps };
