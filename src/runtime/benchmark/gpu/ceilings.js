// The steps "overhead" and "ceilings": what a token costs besides the weights, and the device's ceilings (T168): f32 and f16
// multiply-adds, dot4I8Packed, reading the workgroup's memory and a storage buffer.
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)
const { shared, MODELS, gpu, STORAGE, COPY_SRC, COPY_DST, MAP_READ, UNIFORM, buffer, fill, floats, pipelinesFor, run,
  readBack, median, DISPATCH_MS, SUBMISSION_MS, CEILING_GROUPS, FIRST_LOOPS, MOST_LOOPS, MOST_DISPATCHES,
  GLOBAL_BYTES, paired, scoped } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- what a token costs besides its weights: the dispatches of Llama 3.2 1B's token (seven matrices and seven small
// steps a layer, 16 layers, and the classifier: about 240) doing nothing, a submission with and without waiting for
// the GPU, and reading back the id of a token against Llama 3's 128256 logits
async function overhead() {
  await gpu();
  const { empty } = pipelinesFor();
  const dispatches = 240, vocab = MODELS["Llama 3.2 1B"].vocab;
  const submit = (count, wait) => async () => {
    const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
    for (let i = 0; i < count; i++) run(pass, [empty, null, 1, 1]);
    pass.end();
    shared.device.queue.submit([encoder.finish()]);
    if (wait) await shared.device.queue.onSubmittedWorkDone();
  };
  const logits = buffer(vocab * 4, STORAGE | COPY_SRC | COPY_DST);
  shared.device.queue.writeBuffer(logits, 0, floats(vocab));
  const targets = { 4: buffer(4, MAP_READ | COPY_DST), [vocab * 4]: buffer(vocab * 4, MAP_READ | COPY_DST) };
  const read = (bytes) => () => readBack(shared.device.createCommandEncoder(), logits, bytes, targets[bytes]);
  const found = {
    dispatches, emptyDispatches: await median(submit(dispatches, true)),
    submitOnly: await median(submit(1, false)), submitAndWait: await median(submit(1, true)),
    readToken: await median(read(4)), readLogits: await median(read(vocab * 4)), vocab,
  };
  [logits, ...Object.values(targets)].forEach((x) => x.destroy());
  return found;
}
async function ceilings() {
  await gpu();
  const threads = CEILING_GROUPS * shared.WGSL.CEILING_WORKGROUP;
  const globalBytes = Math.min(GLOBAL_BYTES, shared.device.limits.maxStorageBufferBindingSize, shared.device.limits.maxBufferSize);
  const globalThreads = Math.floor(globalBytes / shared.WGSL.GLOBAL_PER_THREAD / shared.WGSL.CEILING_WORKGROUP) * shared.WGSL.CEILING_WORKGROUP;
  // what a loop counts (FLOPs, ops or bytes) a second: perLoop a thread and a loop, or perDispatch (the global read)
  const rate = (code, { perLoop, perDispatch, groups = CEILING_GROUPS, bytes = 0 }) => scoped(async (owned) => {
    const out = buffer(Math.max(threads, groups * shared.WGSL.CEILING_WORKGROUP) * 4, STORAGE), plan = buffer(16, UNIFORM | COPY_DST);
    owned.push(out, plan);
    const more = bytes ? [buffer(bytes)] : [];
    owned.push(...more);
    if (bytes) fill(more[0], bytes);
    let loops = FIRST_LOOPS;
    const setLoops = () => shared.device.queue.writeBuffer(plan, 0, new Uint32Array([loops, (Math.random() * 2 ** 32) >>> 0,
      new Uint32Array(new Float32Array([0.999]).buffer)[0], new Uint32Array(new Float32Array([0.001]).buffer)[0]]));
    setLoops();
    const pipeline = await shared.device.createComputePipelineAsync({ layout: "auto",
      compute: { module: shared.device.createShaderModule({ code }), entryPoint: "main" } });
    const group = shared.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [out, plan, ...more].map((b, binding) => ({ binding, resource: { buffer: b } })) });
    const submission = async (n) => {
      const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let i = 0; i < n; i++) run(pass, [pipeline, group, groups, 1]);
      pass.end();
      const began = performance.now();
      shared.device.queue.submit([encoder.finish()]);
      await shared.device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    await submission(1);
    // a dispatch of DISPATCH_MS (8 of them, so that the submission's own cost is an eighth in each)
    if (perLoop) {
      while ((await submission(8)) / 8 < DISPATCH_MS) {
        if ((loops *= 2) > MOST_LOOPS) throw new Error(`${MOST_LOOPS} loops took less than ${DISPATCH_MS} ms a dispatch: the loop did no work`);
        setLoops();
      }
    }
    const r = await paired(submission, MOST_DISPATCHES, true);
    if (r.short) throw new Error(`${MOST_DISPATCHES} dispatches took less than ${SUBMISSION_MS} ms: the loop did no work`);
    const work = r.dispatches * (perLoop ? threads * loops * perLoop : perDispatch);
    return { rate: work / (r.ms / 1000), loops, dispatches: r.dispatches, ratio: r.ratio, ...(r.unsteady ? { unsteady: true } : {}) };
  });
  const found = { fallback: shared.fallback };
  const measure = async (name, why, how) => {
    if (why) return (found[name] = { none: why });
    try {
      found[name] = await how();
    } catch (error) {
      found[name] = { error: String(error?.message ?? error) };
    } finally {
      postMessage({ alive: true });
    }
  };
  // the multiply-adds in both shapes; the faster is the ceiling (an unsteady one only when both are)
  const fma = async (half) => {
    const shapes = [];
    for (const shape of shared.WGSL.FMA_SHAPES) shapes.push({ shape, ...await rate(shared.WGSL.fmaCeiling(half, shape), { perLoop: shared.WGSL.FMA_PER_LOOP }) });
    const best = [...shapes].sort((x, y) => Boolean(x.unsteady) - Boolean(y.unsteady) || y.rate - x.rate)[0];
    return { GFLOPS: best.rate / 1e9, shape: best.shape, unsteady: best.unsteady, shapes: shapes.map(({ shape, rate: r }) => ({ shape, GFLOPS: r / 1e9 })) };
  };
  const scaled = (key, r) => ({ [key]: r.rate / 1e9, unsteady: r.unsteady, loops: r.loops, dispatches: r.dispatches, ratio: r.ratio });
  await measure("f32", null, () => fma(false));
  await measure("f16", shared.device.features.has("shader-f16") ? null : "no shader-f16 here", () => fma(true));
  await measure("dot4", shared.packed ? null : "no packed int8 dot here",
    async () => scaled("GOPS", await rate(shared.WGSL.DOT4_CEILING, { perLoop: shared.WGSL.DOT4_PER_LOOP })));
  await measure("shared", null, async () => scaled("GBps", await rate(shared.WGSL.SHARED_CEILING, { perLoop: shared.WGSL.SHARED_PER_LOOP })));
  await measure("global", null, async () => ({ MiB: globalThreads * shared.WGSL.GLOBAL_PER_THREAD / 2 ** 20,
    ...scaled("GBps", await rate(shared.WGSL.GLOBAL_CEILING, { perDispatch: globalThreads * shared.WGSL.GLOBAL_PER_THREAD,
      groups: globalThreads / shared.WGSL.CEILING_WORKGROUP, bytes: globalThreads * shared.WGSL.GLOBAL_PER_THREAD })) }));
  return found;
}

export { overhead, ceilings };
