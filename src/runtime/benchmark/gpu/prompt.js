// The step "prompt": the tokens of a prompt through the matrices all at once (T135's batched shader and T146's tiled ones).
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)
const { shared, layerMatrices, SMALL_PER_LAYER, PROMPT_MODEL, matrixBytes, gpu, buffer, pipelinesFor, promptShaders,
  kindOf, quantizer, matrix, vectors, destroyVectors, run, median, validated } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- a prompt: its tokens through the matrices of the CPU section's made-up model (two layers of Llama 3.2 1B's
// width, no classifier: a prompt's tokens make no logits) all at once, count tokens at a time, with the small steps
// once a layer as for one token. ms per token, for every shader of promptShaders() (T146: the tiled ones, whose rows
// say none or error on their own where they cannot run), and the GFLOPS of it: a multiply and an add for each weight
// and token. A packed shader's input is quantized first where a matrix reads an input of its own (q: the norm's; o:
// the attention's; gate: the norm's; down: SwiGLU's), as the model's layers would
const NEW_INPUT = new Set([0, 3, 4, 6]);
async function prompt(counts = [1, 16, 64]) {
  await gpu();
  // a fallback adapter measures one count, the block of 16 the CPU also takes: each shader and count takes 10 to 100 s
  // there (SwiftShader on the development machine, T146: 580 s for all three), and its times are no GPU's anyway
  if (shared.fallback) counts = counts.filter((tokens) => tokens === 16).slice(0, 1);
  const model = PROMPT_MODEL, perLayer = layerMatrices(model), shapes = [...Array(model.layers)].flatMap(() => perLayer);
  const weights = shapes.reduce((sum, [rows, n]) => sum + rows * n, 0);
  const longest = Math.max(model.dim, model.hidden), most = model.hidden;
  const smallPipeline = pipelinesFor().small, a = buffer(model.dim * 4), b = buffer(model.dim * 4);
  const smallGroup = shared.device.createBindGroup({ layout: smallPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } }] });
  const measure = async (kind, tokens) => {
    const io = vectors(longest, most, tokens), owned = [];
    try {
      return await validated(async () => {
        const made = shapes.map((shape) => matrix(shape, io, kind));
        // one quantizer a width (q, k, v and gate, up read the same width: one each, not one a matrix left unowned)
        const quantize = kind.packed ? new Map([...new Set(perLayer.map(([, n]) => n))].map((n) => [n, quantizer(io, n)])) : null;
        owned.push(...made.flatMap((m) => m.owned), ...[...(quantize?.values() ?? [])].flatMap((q) => q.owned));
        await shared.device.queue.onSubmittedWorkDone();
        const once = async () => {
          const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
          made.forEach((m, i) => {
            if (quantize && NEW_INPUT.has(i % perLayer.length)) run(pass, quantize.get(shapes[i][1]).dispatch);
            m.dispatches.forEach((d) => run(pass, d));
            if (i % perLayer.length === perLayer.length - 1) for (let j = 0; j < SMALL_PER_LAYER; j++) run(pass, [smallPipeline, smallGroup, 1, 1]);
          });
          pass.end();
          shared.device.queue.submit([encoder.finish()]);
          await shared.device.queue.onSubmittedWorkDone();
        };
        const ms = await median(once, 5, 2);
        return { tokens, ms, msPerToken: ms / tokens, GFLOPS: (2 * weights * tokens) / (ms / 1000) / 1e9 };
      });
    } finally {
      owned.forEach((x) => x.destroy());
      destroyVectors(io);
      // the page stops a section that says nothing for 5 minutes, and a fallback adapter takes minutes for all of these
      postMessage({ alive: true });
    }
  };
  const rows = [];
  for (const shader of promptShaders()) {
    if (shader.none) {
      rows.push({ shader: shader.name, none: shader.none });
      continue;
    }
    try {
      const kind = await kindOf(shader);
      // what the row's GFLOPS are held against (T168): the dot4I8Packed ceiling, the f16 or the f32 one
      for (const tokens of counts) rows.push({ shader: shader.name, packed: shader.packed, half: shader.half, ...await measure(kind, tokens) });
    } catch (error) {
      rows.push({ shader: shader.name, error: String(error?.message ?? error) });
    }
  }
  // the batched shader once more at the most tokens, last: a device that has warmed up and slowed down since shows it
  // here, beside the same shader's row at the start (T146's review)
  const last = counts[counts.length - 1];
  if (last) rows.push({ shader: "batched (T135), again at the end", again: true, ...await measure("batched", last) });
  [a, b].forEach((x) => x.destroy());
  return { rows, layers: model.layers, weights, GB: shapes.reduce((sum, shape) => sum + matrixBytes(shape), 0) / 1e9 };
}

export { prompt };
