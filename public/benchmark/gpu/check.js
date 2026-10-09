// The step "check": the int8 shaders against JavaScript on small matrices (T134; T146: the tiled ones with their edges),
// then the layer's check and the sampling's and the generation's (layercheck.js, generatecheck.js).
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)
const { shared, gpu, MAP_READ, COPY_DST, buffer, floats, promptShaders, matVecShaders, kindOf, quantizer, matrix,
  vectors, destroyVectors, run, argmaxOf, readBack, validated } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { checkLayer, checkTokenAttentions } = await import(new URL(`layercheck.js${new URL(import.meta.url).search}`, import.meta.url));
const { checkSampling, checkGeneration } = await import(new URL(`generatecheck.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- check: every shader against JavaScript: the two of a matrix × vector on 300 rows of 512 (rows that are not a
// power of two), the batched one on the same with 11 tokens (a tile and a part of one), the tiled ones (checkTiled),
// the argmax on logits of Llama 3's vocabulary and on a tie (the first of the largest, as JavaScript finds it)
async function check() {
  await gpu();
  const rows = 300, n = 512, words = n / 4;
  const w = new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s = floats(rows * n / shared.GROUP, 0.01);
  const io = vectors(n, rows), x = floats(n, 2);
  shared.device.queue.writeBuffer(io.x, 0, x);
  const { xq, xs } = shared.WGSL.quantizedLikeCpu(x);
  shared.device.queue.writeBuffer(io.xq, 0, new Uint8Array(xq.buffer));
  shared.device.queue.writeBuffer(io.xs, 0, xs);
  const signed = new Int8Array(w.buffer);
  const verdicts = {};
  for (const kind of shared.packed ? ["widen", "packed"] : ["widen"]) {
    const m = matrix([rows, n], io, kind, { w, s });
    const readback = buffer(rows * 4, MAP_READ | COPY_DST);
    const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
    m.dispatches.forEach((d) => run(pass, d));
    pass.end();
    encoder.copyBufferToBuffer(io.y, 0, readback, 0, rows * 4);
    shared.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(MAP_READ);
    const got = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    let worst = 0;
    for (let r = 0; r < rows; r++) {
      let want = 0;
      for (let i = 0; i < words; i++) {
        let dot = 0;
        for (let k = 0; k < 4; k++) dot += signed[r * n + i * 4 + k] * (kind === "packed" ? xq[i * 4 + k] : x[i * 4 + k]);
        want += dot * s[r * (n / shared.GROUP) + (i >> 3)] * (kind === "packed" ? xs[i >> 3] : 1);
      }
      worst = Math.max(worst, Math.abs(got[r] - want) / (Math.abs(want) + 1e-3));
    }
    // float32 sums in another order: a relative 1e-4 is the rounding, a wrong index is off by a multiple
    verdicts[kind] = { worstRelative: worst, ok: worst < 1e-3 };
    m.owned.forEach((b) => b.destroy());
  }
  destroyVectors(io);
  verdicts.batched = await checkBatched(w, s, rows, n);
  Object.assign(verdicts, await checkMatVec());
  Object.assign(verdicts, await checkTiled());
  verdicts.argmax = await checkArgmax();
  Object.assign(verdicts, await checkLayer());
  Object.assign(verdicts, await checkTokenAttentions());
  verdicts.sampling = await checkSampling();
  verdicts["sampling in chunks"] = await checkSampling("chunks");
  Object.assign(verdicts, await checkGeneration());
  return verdicts;
}
// T149: every matrix × vector shader of its own (llama.cpp's and ONNX Runtime's) on 300 rows cut into chunks of 101 (a
// workgroup's 4 or 8 rows and a part of them in each, and shape.first past 0), of widths 544 and 2080: 17 and 65 groups
// of 32, a part of what llama.cpp's 64 groups a pass, ORT's 512 values a step and its DP4A's 32 groups a step take, and
// 2080 more than one of them. x is 64 values longer than the width (a shader that takes the width from the buffer
// reads them), and y holds a sentinel past the rows that must stay. The sums as checkTiled holds them: WGSL.TILED_LINE of
// the sum of the |products| of the row (a wrong index or scale is off by about 1/sqrt(n) of it); the packed ones on the
// vector quantized in JavaScript (as PACKED is checked), whose products are exact integers times the two scales
const SENTINEL = 7.25;
async function checkMatVec() {
  const rows = 300, past = 16, verdicts = {};
  for (const shader of matVecShaders().filter((one) => one.code && !one.none)) {
    try {
      const kind = await kindOf(shader);
      let worst = 0, over = false, touched = false;
      for (const n of [544, 2080]) {
        const perRow = n / shared.GROUP, longest = n + 64;
        const w = new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s = floats(rows * perRow, 0.01);
        const signed = new Int8Array(w.buffer);
        const io = vectors(longest, rows + past), x = floats(longest, 2), { xq, xs } = shared.WGSL.quantizedLikeCpu(x);
        shared.device.queue.writeBuffer(io.x, 0, x);
        shared.device.queue.writeBuffer(io.xq, 0, new Uint8Array(xq.buffer));
        shared.device.queue.writeBuffer(io.xs, 0, xs);
        shared.device.queue.writeBuffer(io.y, 0, new Float32Array(rows + past).fill(SENTINEL));
        const owned = [];
        const got = await validated(async () => {
          const m = matrix([rows, n], io, kind, { w, s }, { chunk: 101 });
          owned.push(...m.owned);
          const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
          m.dispatches.forEach((d) => run(pass, d));
          pass.end();
          return new Float32Array(await readBack(encoder, io.y, (rows + past) * 4));
        }).finally(() => {
          owned.forEach((b) => b.destroy());
          destroyVectors(io);
        });
        for (let r = 0; r < rows; r++) {
          let want = 0, size = 0;
          for (let i = 0; i < n; i++) {
            const g = r * perRow + Math.floor(i / shared.GROUP);
            const product = shader.packed ? signed[r * n + i] * xq[i] * s[g] * xs[Math.floor(i / shared.GROUP)] : signed[r * n + i] * s[g] * x[i];
            want += product;
            size += Math.abs(product);
          }
          const off = Math.abs(got[r] - want);
          worst = Math.max(worst, off / size);
          over ||= !(off < shared.WGSL.TILED_LINE * size);
        }
        for (let r = rows; r < rows + past; r++) touched ||= got[r] !== SENTINEL;
      }
      verdicts[shader.name] = { worstRelative: worst, ok: !over && !touched, ...(touched ? { wrotePastTheRows: true } : {}) };
    } catch (error) {
      verdicts[shader.name] = { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
    } finally {
      // the page stops a section that says nothing for 5 minutes: SwiftShader compiles a shader for up to 90 s, and
      // the checks before the layer's took 380 s together under load (T151)
      postMessage({ alive: true });
    }
  }
  return verdicts;
}
// T146: every tiled shader on 300 rows of 544 (17 groups of 32), cut into chunks of 100 rows (tiles of 32 or 64 rows
// and a part of one each, a subtile of 16 and a part, and shape.first past 0), with 11 and 70 tokens (a part of a tile
// of 32 or 64; two or one and a part), and 11 tokens whose x and y are wider than the product (xStride 608 for 544 of
// the width, yStride 320 for 300 rows: the prompt's model reads 2048 of 8192), twice into the same y (the second added
// to the first: shape.add) against JavaScript's product: half of y. The packed ones on what the GPU quantized, and that
// against JavaScript's quantize_x: a scale may differ in its last bits (WGSL's division is not rounded exactly) and a
// value then by 1, a wrong index by far more. The products are held (shaders.js's tiledOff) to 1e-4 of the sum of the |products| of the
// row and token: what a float32 sum in another order may differ by is 544 × 2^-24 = 3.2e-5 of it at most, and a wrong
// index, scale or group is off by about |value| / |sum of |products|| = 1 / sqrt(544) = 4e-2. The f16 tiles hold a
// weight times its scale and an activation as halves, and WGSL leaves the direction of that rounding to the device
// (round to nearest or toward zero, T146's review): each is then within 1 ulp, 2^-10 of it, or 2^-24 where it is
// subnormal, whatever the direction, so a product is within 2^-9 + 2^-20 of it and 2^-24 × (|weight| + |activation|),
// and the float32 sum adds (n + 1) × 2^-24 of the sum of |products|: the f16 tiles are held to that bound, row by row
// (about 2e-3 of the sum; still 1/20 of a wrong index's)
const CASES = [{ tokens: 11, wider: 0 }, { tokens: 70, wider: 0 }, { tokens: 11, wider: 64 }];
async function checkTiled() {
  const rows = 300, n = 544, perRow = n / shared.GROUP;
  const w = new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s = floats(rows * perRow, 0.01);
  const signed = new Int8Array(w.buffer);
  const verdicts = {};
  for (const shader of promptShaders().filter((one) => one.tile && !one.none)) {
    try {
      const kind = await kindOf(shader);
      let worst = 0, over = false, far = false, apart = 0, values = 0;
      for (const { tokens, wider } of CASES) {
        const xStride = n + wider, yStride = rows + (wider ? 20 : 0);
        const io = vectors(xStride, yStride, tokens), x = floats(tokens * xStride, 2);
        shared.device.queue.writeBuffer(io.x, 0, x);
        const owned = [];
        const [got, xq, xs] = await validated(async () => {
          const made = [matrix([rows, n], io, kind, { w, s }, { chunk: 100 }), matrix([rows, n], io, kind, { w, s }, { chunk: 100, add: true })];
          const quantize = shader.packed ? quantizer(io, n) : null;
          owned.push(...made.flatMap((m) => m.owned), ...(quantize?.owned ?? []));
          const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
          if (quantize) run(pass, quantize.dispatch);
          made.forEach((m) => m.dispatches.forEach((d) => run(pass, d)));
          pass.end();
          const y = new Float32Array(await readBack(encoder, io.y, tokens * yStride * 4));
          if (!quantize) return [y];
          return [y, new Int8Array(await readBack(shared.device.createCommandEncoder(), io.xq, tokens * xStride)),
            new Float32Array(await readBack(shared.device.createCommandEncoder(), io.xs, tokens * (xStride / shared.GROUP) * 4))];
        }).finally(() => {
          owned.forEach((b) => b.destroy());
          destroyVectors(io);
        });
        // T147: the comparison is shaders.js's tiledOff, the model's GPU worker's too
        const off = shared.WGSL.tiledOff({ w: signed, s, x, got, xq, xs, rows, n, tokens, xStride, yStride, half: shader.half });
        worst = Math.max(worst, off.worst);
        over ||= Boolean(off.wrong);
        far ||= off.far;
        apart += off.apart;
        values += off.values;
      }
      // the quantized values no more than 1 apart, and apart in no more than 1 of 100
      const quantizing = values ? { apart: apart / values, far } : {};
      verdicts[shader.name] = { worstRelative: worst, ok: !over && !far && apart <= 0.01 * values, ...quantizing };
    } catch (error) {
      verdicts[shader.name] = { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
    } finally {
      // the page stops a section that says nothing for 5 minutes: SwiftShader compiles a shader for up to 90 s, and
      // the checks before the layer's took 380 s together under load (T151)
      postMessage({ alive: true });
    }
  }
  return verdicts;
}
async function checkBatched(w, s, rows, n) {
  const tokens = 11, io = vectors(n, rows, tokens), x = floats(tokens * n, 2);
  shared.device.queue.writeBuffer(io.x, 0, x);
  const m = matrix([rows, n], io, "batched", { w, s });
  const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
  m.dispatches.forEach((d) => run(pass, d));
  pass.end();
  const got = new Float32Array(await readBack(encoder, io.y, tokens * rows * 4));
  const signed = new Int8Array(w.buffer);
  let worst = 0;
  for (let t = 0; t < tokens; t++) {
    for (let r = 0; r < rows; r++) {
      let want = 0;
      for (let i = 0; i < n; i++) want += signed[r * n + i] * s[r * (n / shared.GROUP) + Math.floor(i / shared.GROUP)] * x[t * n + i];
      worst = Math.max(worst, Math.abs(got[t * rows + r] - want) / (Math.abs(want) + 1e-3));
    }
  }
  m.owned.forEach((b) => b.destroy());
  destroyVectors(io);
  return { worstRelative: worst, ok: worst < 1e-3 };
}
async function checkArgmax() {
  let right = true;
  for (const [vocab, tie] of [[128256, false], [1000, true]]) {
    const io = vectors(4, vocab), logits = floats(vocab, 20);
    if (tie) logits[700] = logits[300] = 50;  // the first of the two
    shared.device.queue.writeBuffer(io.y, 0, logits);
    const picked = argmaxOf(io, vocab);
    const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
    run(pass, picked.dispatch);
    pass.end();
    const got = new Uint32Array(await readBack(encoder, picked.chosen, 4))[0];
    let want = 0;
    for (let i = 1; i < vocab; i++) if (logits[i] > logits[want]) want = i;
    right &&= got === want;
    picked.owned.forEach((b) => b.destroy());
    destroyVectors(io);
  }
  return { worstRelative: 0, ok: right };
}

export { check };
