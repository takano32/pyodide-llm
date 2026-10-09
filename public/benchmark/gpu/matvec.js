// The steps "bandwidth" and "token": GB/s of one int8 matrix times a vector by every shader (T134, T149) against the CPU's
// kernel, and a whole token's work of a model's shapes.
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)
const { shared, MODELS, layerMatrices, SMALL_PER_LAYER, matrixBytes, gpu, MAP_READ, COPY_DST, buffer, noise, floats,
  pipelinesFor, matVecShaders, kindOf, quantizer, matrix, placed, bound, vectors, destroyVectors, run, argmaxOf,
  readBack, median, validated, paired, scoped } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- bandwidth: one matrix by every shader of matVecShaders() (T149), timed as the ceilings are (paired: a submission
// of 2n of it less one of n, so that what a submission costs besides its work drops out, T168). The weights are
// placed once for every shader (only the bind groups are the shader's), in as many copies as make MATVEC_BYTES, read
// in turn: the same small matrix read again and again stays in the GPU's caches (a phone's system cache of 19 MB and
// more holds Llama 3.2 1B's w1, T149's review), and a token reads each matrix once. The first shader is measured again
// at the end (a device that slows down as it warms shows it there, as the prompt's batched row does). A packed shader
// takes its vector quantized: what QUANTIZE costs for this width, once, goes beside them. On a fallback adapter each
// once, one copy, and no matrix past FALLBACK_BYTES (the check holds the shaders to JavaScript; SwiftShader took 13
// minutes of bench-check's 20 for the GPU section with the classifier's 295 MB 8 times). widen and packed: T134's two;
// cpu: the CPU's kernel on the same bytes
const MATVEC_MOST = 1 << 14, MATVEC_BYTES = 128 << 20, FALLBACK_BYTES = 32 << 20;
async function bandwidth(shape) {
  await gpu();
  const bytes = matrixBytes(shape), shaders = matVecShaders(), rows = [];
  const label = (shader) => ({ shader: shader.name, check: shader.check ?? shader.name });
  if (shared.fallback && bytes > FALLBACK_BYTES) {
    return { rows: shaders.map((shader) => ({ ...label(shader), none: "not on a fallback adapter" })), cpu: await cpuBandwidth(shape) };
  }
  const io = vectors(shape[1], shape[0]), held = [];
  const copies = shared.fallback ? 1 : Math.ceil(MATVEC_BYTES / bytes);
  // what the dispatches do, timed: n of them a submission, each on the next copy (never the one just read)
  let next = 0;
  const timer = (each) => async (n) => {
    const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
    for (let i = 0; i < n; i++) each[next++ % each.length].forEach((d) => run(pass, d));
    pass.end();
    const began = performance.now();
    shared.device.queue.submit([encoder.finish()]);
    await shared.device.queue.onSubmittedWorkDone();
    return performance.now() - began;
  };
  const time = async (each) => {
    const submission = timer(each);
    if (shared.fallback) return { ms: await submission(1), dispatches: 1 };
    await submission(2);
    return paired(submission, MATVEC_MOST);
  };
  try {
    await scoped(async () => {
      for (let c = 0; c < copies; c++) held.push(placed(shape, io));
      await shared.device.queue.onSubmittedWorkDone();
    });
    const measure = async (shader, again = false) => {
      const row = { ...label(shader), ...(again ? { shader: `${shader.name}, again at the end`, again } : {}) };
      if (shader.none) return { ...row, none: shader.none };
      try {
        const r = await validated(async () => {
          const kind = await kindOf(shader);
          return time(held.map((h) => bound(h, io, kind).dispatches));
        });
        return { ...row, GBps: (r.dispatches * bytes) / (r.ms / 1000) / 1e9, msEach: r.ms / r.dispatches, dispatches: r.dispatches,
          ...(r.ratio ? { ratio: r.ratio } : {}), ...(r.unsteady ? { unsteady: true } : {}) };
      } catch (error) {
        return { ...row, error: String(error?.message ?? error) };
      } finally {
        postMessage({ alive: true });
      }
    };
    for (const shader of shaders) rows.push(await measure(shader));
    rows.push(await measure(shaders[0], true));
  } finally {
    held.forEach((h) => h.owned.forEach((b) => b.destroy()));
  }
  // the vector of a packed shader quantized (QUANTIZE, one dispatch a vector of this width)
  let quantize;
  if (shared.packed) {
    const q = quantizer(io, shape[1]);
    try {
      const r = await validated(() => time([[q.dispatch]]));
      quantize = { msEach: r.ms / r.dispatches, ...(r.unsteady ? { unsteady: true } : {}) };
    } catch (error) {
      quantize = { error: String(error?.message ?? error) };
    } finally {
      q.owned.forEach((b) => b.destroy());
    }
  }
  destroyVectors(io);
  const measured = (kind) => rows.find((row) => row.check === kind && !row.again && row.GBps);
  return { rows, copies, quantize, widen: measured("widen"), packed: measured("packed"), cpu: await cpuBandwidth(shape) };
}

// the CPU on the kernel the model page uses for int8 on one thread (matmul_q8, without relaxed SIMD: every
// browser has it), on four matrices of this shape in turn so that no cache holds them. The page's forward pass does
// more than this with relaxed SIMD and its software threads (2.9 times on the owner's Android, T157), which is why the
// token's table holds the GPU against the CPU section instead.
let cpuKernel;
async function cpuBandwidth([rows, n]) {
  const weights = rows * n, scales = (rows * n / shared.GROUP) * 4, copies = Math.max(1, Math.min(4, Math.floor(128e6 / weights)));
  const bytes = 4096 + n * 8 + rows * 4 + copies * (weights + scales);
  const memory = new WebAssembly.Memory({ initial: Math.ceil(bytes / 65536) + 1 });
  // the kernels of the same deployment as this file (?v=, GitHub Pages keeps a file for ten minutes)
  cpuKernel ??= await WebAssembly.compile(await (await fetch(new URL(`../../simdkernel_plain.wasm${new URL(import.meta.url).search}`, import.meta.url))).arrayBuffer());
  const k = (await WebAssembly.instantiate(cpuKernel, { env: { memory } })).exports;
  const U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const x = 4096, xq = x + n * 4, xs = xq + n, out = xs + (n / shared.GROUP) * 4 + 64, first = Math.ceil((out + rows * 4) / 64) * 64;
  F.set(floats(n, 2), x / 4);
  for (let c = 0; c < copies; c++) {
    const at = first + c * (weights + scales);
    for (let i = 0; i < weights; i += noise.length) U.set(noise.subarray(0, Math.min(noise.length, weights - i)), at + i);
    F.set(floats(rows * n / shared.GROUP, 0.002), (at + weights) / 4);
  }
  k.quantize_x(xq, xs, x, n, 0);
  const pass = (c) => k.matmul_q8(out, xq, xs, first + c * (weights + scales), first + c * (weights + scales) + weights, n, 0, rows);
  for (let c = 0; c < copies; c++) pass(c);
  const rounds = Math.max(1, Math.round(200e6 / weights));
  const began = performance.now();
  for (let r = 0; r < rounds; r++) pass(r % copies);
  const ms = performance.now() - began;
  return { GBps: (rounds * (weights + scales)) / (ms / 1000) / 1e9, msEach: ms / rounds, threads: 1 };
}

// ---- a token's work: every layer's seven matrices and seven small dispatches, the classifier, the logits back.
// sample: the argmax on the GPU and 4 bytes back instead of the logits
async function token(name, kind = "widen", { sample = false } = {}) {
  await gpu();
  const model = MODELS[name];
  const perLayer = layerMatrices(model), small = SMALL_PER_LAYER;
  const shapes = [...Array(model.layers)].flatMap(() => perLayer);
  const classifier = [model.vocab, model.dim];
  const longest = Math.max(model.dim, model.hidden), most = Math.max(2 * model.hidden, model.vocab);
  const io = vectors(longest, most);
  const made = [];
  let bytes = 0;
  // a buffer the device cannot give fails later and quietly, as an error of these scopes
  shared.device.pushErrorScope("out-of-memory");
  shared.device.pushErrorScope("validation");
  try {
    for (const shape of [...shapes, classifier]) {
      const m = matrix(shape, io, kind);
      made.push(m);
      bytes += m.bytes;
    }
  } catch (error) {
    await shared.device.popErrorScope();
    await shared.device.popErrorScope();
    made.forEach((m) => m.owned.forEach((b) => b.destroy()));
    return { model: name, error: `could not hold the weights (${(bytes / 1e9).toFixed(2)} GB made): ${error.message}` };
  }
  const smallPipeline = pipelinesFor().small;
  const a = buffer(model.dim * 4), b = buffer(model.dim * 4);
  const smallGroup = shared.device.createBindGroup({ layout: smallPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } }] });
  const picked = sample ? argmaxOf(io, model.vocab) : null;
  const back = buffer(picked ? 4 : model.vocab * 4, MAP_READ | COPY_DST);
  // the uploads may still be going: wait for them, and for a device that could not take them
  await shared.device.queue.onSubmittedWorkDone();
  const invalid = await shared.device.popErrorScope(), outOfMemory = await shared.device.popErrorScope();
  const refused = invalid ?? outOfMemory;
  if (refused) {
    made.forEach((m) => m.owned.forEach((x) => x.destroy()));
    return { model: name, error: `the GPU did not take ${(bytes / 1e9).toFixed(2)} GB of weights: ${refused.message}` };
  }
  const once = async () => {
    const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
    made.forEach((m, i) => {
      m.dispatches.forEach((d) => run(pass, d));
      // after every layer's last matrix, its small steps (and none after the classifier)
      if (i < shapes.length && i % perLayer.length === perLayer.length - 1) {
        for (let j = 0; j < small; j++) run(pass, [smallPipeline, smallGroup, 1, 1]);
      }
    });
    if (picked) run(pass, picked.dispatch);
    pass.end();
    // the token's id, or every logit for the CPU to sample from
    return picked ? readBack(encoder, picked.chosen, 4, back) : readBack(encoder, io.y, model.vocab * 4, back);
  };
  const ms = await median(once);
  made.forEach((m) => m.owned.forEach((x) => x.destroy()));
  picked?.owned.forEach((x) => x.destroy());
  [a, b, back].forEach((x) => x.destroy());
  destroyVectors(io);
  return { model: name, kind, sample, GB: bytes / 1e9, msPerToken: ms, tokPerSecond: 1000 / ms,
    GBps: bytes / (ms / 1000) / 1e9,
    dispatches: made.reduce((n, m) => n + m.dispatches.length, 0) + model.layers * small + (picked ? 1 : 0) };
}

export { MATVEC_BYTES, MATVEC_MOST, bandwidth, token };
