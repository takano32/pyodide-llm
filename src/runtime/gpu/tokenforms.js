// The forms of a token's layer checked on the model's own weights against JavaScript (T152), timed and chosen.
// (T352: a module of the model's GPU worker, public/gpu.js, which asks for it with its own ?v=<build>)

const { COPY_DST, MAP_READ, rowBytes, turnedAt, common, within, buffer, readBack, validated, pipelineOf } =
  await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { tablesOf } = await import(new URL(`weights.js${new URL(import.meta.url).search}`, import.meta.url));
const { halfToFloat } = await import(new URL(`forms.js${new URL(import.meta.url).search}`, import.meta.url));
const { grow, BLOCK_ROUNDS } = await import(new URL(`block.js${new URL(import.meta.url).search}`, import.meta.url));
const { chooseTokenAttention } =
  await import(new URL(`tokenattention.js${new URL(import.meta.url).search}`, import.meta.url));
const { tokenCandidates, floatHead, tokenShape, tokenCodes, tokenBuffers, tokenPass, tokenStep, runTokens } =
  await import(new URL(`tokens.js${new URL(import.meta.url).search}`, import.meta.url));

// The forms of a token this device can make, compiled, checked and (where more than one is right) timed; the fastest
// taken (m.gen.form), with what each came to (m.gen.forms) and its ms a step (m.gen.ms). plan.remembered.tokens: the
// one the page kept for this adapter, alone where it is still right here. plan.force.tokens (tests): that form alone;
// plan.force.quick: the first right one, untimed. Throws where none is right (the tokens then stay on the CPU)
async function chooseTokens(m) {
  const { plan, wgsl } = m;
  // (tokenPass puts a parallel residual's second norm before o as a dispatch: LayerNorm's, GPT-NeoX's)
  if (plan.parallel && !plan.layerNorm) throw new Error("a parallel residual without LayerNorm is not on the GPU's tokens");
  m.gen = await tokenBuffers(m);
  await chooseTokenAttention(m);
  if (common.stopping) return;
  // T156: the first layer's matrices of a model on the GPU alone are not in the shared memory: checkTokens reads them
  // back from the GPU (T210: and the rows of the tables it reads)
  if (m.direct) {
    m.firstLayer ??= await within(firstLayer(m), "reading the first layer back");
    m.tableRows ??= await within(tableRows(m), "reading the tables' rows back");
  }
  let forms = tokenCandidates(m);
  if (plan.force.tokens) forms = forms.filter((form) => form.name === plan.force.tokens);
  const kept = !plan.force.tokens && forms.find((form) => form.name === m.remembered?.tokens);
  if (kept) forms = [kept, ...forms.filter((form) => form !== kept)];
  const right = [], tried = [];
  for (const candidate of forms) {
    // the remembered one right: no other is compiled or timed (as the matrices, T148)
    if ((plan.force.quick || (kept && right[0]?.name === kept.name)) && right.length) break;
    const form = { ...candidate, pipes: {} }, compiled = new Map();
    try {
      for (const [key, code] of tokenCodes(wgsl, form, tokenShape(m))) {
        if (!compiled.has(code)) compiled.set(code, await within(validated(m, () => pipelineOf(m, code)), `compiling ${form.name}`));
        form.pipes[key] = compiled.get(code);
      }
      const wrong = await within(checkTokens(m, form), `checking ${form.name}`);
      if (wrong) tried.push({ name: form.name, none: `wrong: ${wrong}` });
      else right.push(form);
    } catch (error) {
      if (error?.late) throw error;
      tried.push({ name: form.name, none: String(error?.message ?? error) });
    }
    if (common.stopping) return;
  }
  if (!right.length) {
    throw new Error(`no layer of a token is right on this GPU (${tried.map((f) => `${f.name}: ${f.none}`).join("; ") || `none named ${plan.force.tokens}`})`);
  }
  const ms = plan.force.quick ? right.map(() => undefined) : await within(timeTokens(m, right), "timing a token");
  const best = ms.reduce((b, t, i) => (t !== undefined && (ms[b] === undefined || t < ms[b]) ? i : b), 0);
  m.gen.form = right[best];
  m.gen.ms = ms[best];
  m.gen.forms = [...tried, ...right.map((form, i) => ({ name: form.name, ms: ms[i], ...(kept && form.name === kept.name ? { remembered: true } : {}) }))];
}

// ms a step of each form: runs of plan.tokens.most steps a submission from position 0 (a made-up state, the settings
// of the list's sampled models, T151's), the forms in turn after one run each to warm up, BLOCK_ROUNDS rounds (one on
// a fallback adapter: its times are no GPU's), the median of each over its steps. The runs write the GPU's own keys
// and values of those positions, which hold nothing of forward.js's yet (as timeBlocks)
async function timeTokens(m, forms) {
  const { wgsl, gen: g } = m, most = g.most;
  const state = wgsl.samplingState({ token: 1, pos: 0, history: [1] });
  const settings = wgsl.samplingSettings({ vocab: g.vocab, temperature: 0.7, topp: 0.9, penalty: 1.1 });
  const randoms = new Float32Array(most).map(() => Math.fround(Math.random()) % 1);
  if (most > m.cache.capacity) grow(m, most);
  const timed = async (form) => {
    const began = performance.now();
    await runTokens(m, tokenStep(m, form, most), { count: most, pos: 0, state, settings, randoms });
    return performance.now() - began;
  };
  for (const form of forms) await timed(form);
  const times = forms.map(() => []);
  for (let round = 0; round < (m.fallback ? 1 : BLOCK_ROUNDS); round++) {
    for (let i = 0; i < forms.length; i++) times[i].push(await timed(forms[i]));
    if (common.stopping) break;
  }
  return times.map((list) => list.sort((a, b) => a - b)[list.length >> 1] / most);
}

// ---- T152's check of a form on the model's own weights, against JavaScript (the drivers, the subgroups and the
// packed dot differ from device to device, and only the device can say):
//   1. EMBED of a token of the second half of the vocabulary and the first layer at position 1 (the cache's row 0 of random
//      float16), the stream h and the keys and values of position 1 read back: against JavaScript's layer in float64
//      (the weights read from the shared memory as forward.js holds them: int6 widened by shaders.js's sixValues; RoPE
//      from the CPU's tables; the keys and values rounded to float16 where the attention reads them (T225's review: the
//      device's own float16 where it is a neighbour of the float64 value, heldFloats; else the nearest); on DP4A each
//      matrix's input quantized as quantize_x quantizes it, in float32). The stream: no farther than LAYER_LINE of the
//      largest change the layer made; the keys and values no farther than it of their largest. A wrong row, group,
//      head, angle or scale is a tenth and more off; on DP4A a value quantized to the other side of a rounding moves
//      the layer by about 1e-3 (T175), and so its lines are DP4A_LINE.
//   2. the head on a made-up stream (±2), greedy: the logits of every 256th row (and the last) against JavaScript's,
//      no farther than LOGITS_LINE of their largest (DP4A: DP4A_LINE), and the id SAMPLE chose the first largest of the
//      GPU's own logits; then sampled (temperature 0.8, top-p 0.9, penalty 1.3 on a history that holds that id, a
//      random number of 0.7): the logits SAMPLE penalized in place as penalizeLikeCpu penalizes the greedy ones
//      (within 1e-6), and the id one that the CPU's walk (walkLikeCpu) over those logits reaches within 1e-4 of the
//      random number's share of the mass (T151's line: the GPU's exp and its float32 sums).
// The reason it is wrong, or null.
const LAYER_LINE = 2e-3, LOGITS_LINE = 1e-3, DP4A_LINE = 2e-2;
// T225's review: WGSL leaves it to the device which of its two float16 neighbours a float32 becomes (§15.7.6 Floating
// Point Conversion: "WGSL does not specify whether the higher or lower representable value is chosen, and different
// instances of such a conversion may choose differently"; pack2x16float is such a conversion), and Direct3D, where
// Chrome on Windows runs WebGPU, converts toward zero (D3D11.3 functional specification 3.2.2: "Round-to-zero must be
// used during conversion to another float format"; Dawn's HLSL writer makes pack2x16float of f32tof16). The position's
// keys and values the GPU writes are then 1 float16 spacing from the nearest's on about half of their numbers, and the
// stream computed from the nearest's was up to 3.2 times the line off the GPU's (CI, Dawn on lavapipe with every
// conversion cut toward zero: a llama.cpp form on llm-jp-3 150M 6.5e-3 at the line 2e-3; with the GPU's own keys and
// values 2.3e-6, as public/benchmark/gpu/halves.js's heldHalves takes them: T225). So the reference takes the device's own
// float16 (got, as floats) of a value where it is within a float16 spacing of the value and the float32 sums' noise
// (HALF_SLACK of the largest: about 10 times a sum of 2112 products's), and the nearest where it is not: a wrong key
// or value is as far from the line as before, and the keys' and values' lines below are unchanged.
const HALF_SLACK = 1e-5;
function heldFloats(x, got) {
  const round16 = Math.f16round ?? ((value) => value), slack = HALF_SLACK * x.reduce((top, value) => Math.max(top, Math.abs(value)), 0);
  return x.map((value, i) => (Math.abs(got[i] - value) <= 2 ** (Math.max(Math.floor(Math.log2(Math.abs(value))), -14) - 10) + slack ? got[i] : round16(value)));
}
export { heldFloats };  // (tests/gpu-choice-check.mjs)
// T156: the first layer's matrices, { name: [values, scales] } as the bytes of their rows (rowBytes: int8, or T232
// ternary codes) and a Float32Array each, read back from their buffers (a piece after another: the rows in order)
async function firstLayer(m) {
  const group = m.wgsl.GROUP, out = {};
  for (const [name, matrix] of Object.entries(m.matrices)) {
    const row = rowBytes(matrix), rows = matrix.pieces.reduce((sum, piece) => sum + piece.rows, 0);
    const values = new Uint8Array(rows * row), scales = new Float32Array((rows * row) / group);
    for (const piece of matrix.pieces) {
      const [v, s] = piece.layers[0], valueBytes = piece.rows * row, scaleBytes = (valueBytes / group) * 4;
      values.set(new Uint8Array(await readBack(m, v.buffer ?? v, valueBytes, v.offset ?? 0)), piece.first * row);
      scales.set(new Float32Array(await readBack(m, s.buffer ?? s, scaleBytes, s.offset ?? 0)), (piece.first * row) / group);
    }
    out[name] = [values, scales];
  }
  return out;
}
// the tokens whose rows of the embedding the first layer's check may take (of the second half of the vocabulary: see
// below), and the rows of the classifier it holds to JavaScript's (every 256th, and the last)
const checkCandidates = (vocab) => [...Array(64).keys()].map((i) => vocab - 1 - Math.floor((i * vocab) / 128));
const checkRows = (vocab) => [...new Set([...[...Array(Math.ceil(vocab / 256)).keys()].map((i) => i * 256), vocab - 1])];
// T210: those rows of the tables of a model on the GPU alone, which are there only: { embedding, classifier }, Maps of
// a row to [the bytes of its values (rowBytes: int8, or T232 ternary codes), its scales], read back in one copy each
// (a row of a piece at a time)
async function tableRows(m) {
  const vocab = m.plan.tokens.classifier.rows, specs = tablesOf(m.plan);
  const read = async (pieces, spec, rows) => {
    const n = rowBytes(spec), perRow = (n / m.wgsl.GROUP) * 4;
    const target = m.device.createBuffer({ size: rows.length * (n + perRow), usage: MAP_READ | COPY_DST });
    try {
      const encoder = m.device.createCommandEncoder();
      rows.forEach((r, j) => {
        const piece = pieces.find((p) => r >= p.first && r < p.first + p.rows);
        encoder.copyBufferToBuffer(piece.values, (r - piece.first) * n, target, j * n, n);
        encoder.copyBufferToBuffer(piece.scales, (r - piece.first) * perRow, target, rows.length * n + j * perRow, perRow);
      });
      m.device.queue.submit([encoder.finish()]);
      await target.mapAsync(MAP_READ);
      const bytes = target.getMappedRange().slice(0);
      return new Map(rows.map((r, j) => [r, [new Uint8Array(bytes, j * n, n), new Float32Array(bytes, rows.length * n + j * perRow, perRow / 4)]]));
    } finally {
      target.destroy();
    }
  };
  return { embedding: await read(m.tables.embedding, specs.embedding, checkCandidates(vocab)),
    classifier: await read(m.tables.classifier, specs.classifier, checkRows(vocab)) };
}
async function checkTokens(m, form) {
  const { plan, wgsl, gen: g, device } = m, vocab = g.vocab, dim = plan.dim, headSize = plan.headSize, half = headSize / 2;
  const qDim = plan.heads * headSize, kvDim = plan.kvHeads * headSize, line = form.dp4a ? DP4A_LINE : LAYER_LINE;
  const floats = (address, n) => Float64Array.from(new Float32Array(m.memory.buffer, address, n));
  // row r of a matrix or a table ({ n, six, ternary }, its values and scales at [valuesAt, scalesAt] in the shared
  // memory): [its int8 values, its scales (one a group of 32 weights; T232, ternary: of 128)]; (T156) of a matrix read
  // back from the GPU (firstLayer: [values, scales]); (T210) of a table's rows read back (tableRows)
  const stored = (spec) => (spec.six ? (spec.n * 3) / 4 : rowBytes(spec)), scalesOf = (spec) => rowBytes(spec) / wgsl.GROUP;
  // the bytes of a row as its int8 values: int6 widened, ternary codes less one, int8 as they are
  const valuesOf = (spec, bytes) => (spec.six ? wgsl.sixValues(bytes) : spec.ternary ? wgsl.ternaryValues(bytes)
    : new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length));
  const inMemory = (spec, [valuesAt, scalesAt]) => (r) => [
    valuesOf(spec, new Uint8Array(m.memory.buffer, valuesAt + r * stored(spec), stored(spec))),
    new Float32Array(m.memory.buffer, scalesAt + r * scalesOf(spec) * 4, scalesOf(spec))];
  const readBackRows = (spec, [values, scales]) => (r) => [valuesOf(spec, values.subarray(r * stored(spec), (r + 1) * stored(spec))),
    scales.subarray(r * scalesOf(spec), (r + 1) * scalesOf(spec))];
  const tableSpec = (name) => (name === "embedding" ? plan.tokens.embedding ?? plan.tokens.classifier : plan.tokens.classifier);
  const tableRow = (name) => (m.tableRows ? (r) => {
    const [bytes, scales] = m.tableRows[name].get(r);
    return [valuesOf(tableSpec(name), bytes), scales];
  } : inMemory(tableSpec(name), tableSpec(name).at));
  // a matrix's rows (n weights each, rowOf(r): [values, scales]) times x, the rows given (packed: x quantized first).
  // The vector's groups are of 32; a row's scales cover as many of them each as they are fewer (one; T232, ternary: four)
  const product = (n, rowOf, x, rows, packed = form.dp4a) => {
    const q = packed ? wgsl.quantizedLikeCpu(Float32Array.from(x)) : null, perRow = n / wgsl.GROUP;
    return Float64Array.from(rows, (r) => {
      const [w, s] = rowOf(r), each = perRow / s.length;
      let sum = 0;
      for (let b = 0; b < perRow; b++) {
        let part = 0;
        for (let i = b * wgsl.GROUP; i < (b + 1) * wgsl.GROUP; i++) part += w[i] * (q ? q.xq[i] : x[i]);
        sum += part * s[Math.floor(b / each)] * (q ? q.xs[b] : 1);
      }
      return sum;
    });
  };
  const all = (n) => [...Array(n).keys()];
  const matmul = (name, x) => {
    const matrix = plan.matrices[name];
    return product(matrix.n, m.firstLayer ? readBackRows(matrix, m.firstLayer[name]) : inMemory(matrix, matrix.layers[0]), x, all(matrix.rows));
  };
  const rms = (x, weights) => {
    const s = 1 / Math.sqrt(x.reduce((sum, v) => sum + v * v, 0) / x.length + plan.eps);
    return x.map((v, i) => weights[i] * (s * v));
  };
  // T226: the stream's norm, RMSNorm or (T154) LayerNorm with its bias, as the CPU's kernels have them
  const norm = (x, weights, bias) => {
    if (!plan.layerNorm) return rms(x, weights);
    const mean = x.reduce((sum, v) => sum + v, 0) / x.length;
    const s = 1 / Math.sqrt(x.reduce((sum, v) => sum + (v - mean) ** 2, 0) / x.length + plan.eps);
    return x.map((v, i) => weights[i] * (s * (v - mean)) + bias[i]);
  };
  // the first layer's vector of a name (plan.vectors), where the model has it; and a vector with it added
  const vectorOf = (name) => plan.vectors[name] && floats(plan.vectors[name].at, plan.vectors[name].size);
  const plus = (x, name) => {
    vectorOf(name)?.forEach((b, i) => { x[i] += b; });
    return x;
  };
  const largest =(xs) => xs.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
  const off = (got, want) => largest(want.map((v, i) => got[i] - v));
  const owned = [];
  try {
    // 1. the first layer at position 1
    // a token of the second half of the vocabulary whose row is large: the rows of tokens no text has (llm-jp-3's
    // last) are nearly 0, and their keys and values then float16's subnormals (a check of them was 15% off, CI)
    const embeddingRow = tableRow("embedding");
    const sizeOf = (t) => embeddingRow(t)[1].reduce((sum, v) => sum + Math.abs(v), 0);
    const token = checkCandidates(vocab).reduce((best, t) => (sizeOf(t) > sizeOf(best) ? t : best));
    const pos = 1;
    const halfBits = () => (Math.random() < 0.5 ? 0x8000 : 0) | ((13 + ((Math.random() * 3) | 0)) << 10) | ((Math.random() * 1024) | 0);
    const row0 = [0, 1].map(() => new Uint16Array(kvDim).map(halfBits));
    if (m.cache.capacity < 2) grow(m, 2);
    device.queue.writeBuffer(m.cache.keys[0], 0, row0[0]);
    device.queue.writeBuffer(m.cache.values[0], 0, row0[1]);
    const readH = buffer(m, dim * 4, MAP_READ | COPY_DST, owned);
    const { kv } = await runTokens(m, tokenPass(m, form, { to: 1, head: false, positions: pos + 1 }), { count: 1, pos, keep: true,
      state: wgsl.samplingState({ token, pos, history: [token] }), extra: (encoder) => encoder.copyBufferToBuffer(g.h, 0, readH, 0, dim * 4) });
    await readH.mapAsync(MAP_READ);
    const h = new Float32Array(readH.getMappedRange().slice(0));
    readH.unmap();
    const [gotK, gotV] = kv.map((layers) => Float64Array.from(new Uint16Array(layers[0].buffer, layers[0].byteOffset, kvDim), halfToFloat));
    // JavaScript's layer
    const [eRow, eScales] = embeddingRow(token);
    // (T226: with GPT-2's learned position of pos, as the CPU's embed() adds it)
    const x0 = Float64Array.from(eRow, (v, i) => v * eScales[Math.floor((i * eScales.length) / eRow.length)]);
    if (plan.tokens.positions) floats(plan.tokens.positions + pos * dim * 4, dim).forEach((p, i) => { x0[i] += p; });
    const xn = norm(x0, vectorOf("attention"), vectorOf("attentionBias"));
    // T226: as the CPU has it (forward.js), the biases (Qwen2's; T154's), then the norms of the heads of q and k
    // (Qwen3's), then RoPE (all of a head, T154: a part of it, or none); the first layer's of each, where the model
    // has them
    const [q, k, v] = [["wq", "bq"], ["wk", "bk"], ["wv", "bv"]].map(([name, bias]) => plus(matmul(name, xn), bias));
    for (const [name, x] of [["qNorm", q], ["kNorm", k]]) {
      const weights = vectorOf(name);
      for (let at = 0; weights && at < x.length; at += headSize) x.set(rms(x.subarray(at, at + headSize), weights), at);
    }
    const cos = floats(plan.cos + pos * half * 4, half), sin = floats(plan.sin + pos * half * 4, half);
    const turn = (vector) => {
      for (let at = 0; at < vector.length; at += 2) {
        const i = (at % headSize) / 2;
        if (at % headSize >= turnedAt(plan, 0)) continue;
        const [a, b] = [vector[at], vector[at + 1]];
        vector[at] = a * cos[i] - b * sin[i];
        vector[at + 1] = a * sin[i] + b * cos[i];
      }
    };
    turn(q);
    turn(k);
    const keys = [Float64Array.from(row0[0], halfToFloat), heldFloats(k, gotK)], values = [Float64Array.from(row0[1], halfToFloat), heldFloats(v, gotV)];
    const att = new Float64Array(qDim), group = plan.heads / plan.kvHeads;
    for (let head = 0; head < plan.heads; head++) {
      const kvAt = Math.floor(head / group) * headSize, at = head * headSize;
      const scores = keys.map((key) => {
        let sum = 0;
        for (let d = 0; d < headSize; d++) sum += q[at + d] * key[kvAt + d];
        return sum / Math.sqrt(headSize);
      });
      const top = Math.max(...scores), weights = scores.map((score) => Math.exp(score - top)), total = weights[0] + weights[1];
      for (let d = 0; d < headSize; d++) att[at + d] = (weights[0] * values[0][kvAt + d] + weights[1] * values[1][kvAt + d]) / total;
    }
    const o = plus(matmul("wo", att), "bo"), h1 = x0.map((value, i) => value + o[i]);
    // (T154: GPT-NeoX's parallel residual, the FFN's norm of the layer's input; no gate, w1's bias and GELU as
    // shaders.js's has it)
    const xn2 = norm(plan.parallel ? x0 : h1, vectorOf("ffn"), vectorOf("ffnBias"));
    const gate = plus(matmul("w1", xn2), "b1"), up = plan.matrices.w3 && matmul("w3", xn2);
    const gelu = (value) => 0.5 * value * (1 + Math.tanh(Math.min(9.010913, Math.max(-9.010913, 0.7978845608028654 * (value + 0.044715 * value ** 3)))));
    const activated = gate.map(up ? (value, i) => (value / (1 + Math.exp(-value))) * up[i] : gelu);
    const down = plus(matmul("w2", activated), "b2"), h2 = h1.map((value, i) => value + down[i]);
    const stream = off(h, h2) / largest(h2.map((value, i) => value - x0[i]));
    const keyOff = off(gotK, k) / largest(k), valueOff = off(gotV, v) / largest(v);
    if (!(stream <= line)) return `the first layer's stream is ${stream.toExponential(2)} of its change from JavaScript's (line ${line})`;
    if (!(keyOff <= line && valueOff <= line)) {
      return `the first layer's keys are ${keyOff.toExponential(2)} and its values ${valueOff.toExponential(2)} from JavaScript's (line ${line})`;
    }
    // 2. the head: greedy, then sampled
    const stream2 = new Float32Array(dim).map(() => (Math.random() - 0.5) * 4);
    const rows = checkRows(vocab);
    const readLogits = buffer(m, vocab * 4, MAP_READ | COPY_DST, owned);
    const head = tokenPass(m, form, { to: 0, embed: false });
    const run = async (settings, history, random) => {
      device.queue.writeBuffer(g.h, 0, stream2);
      const { ids } = await runTokens(m, head, { count: 1, pos, state: wgsl.samplingState({ token, pos, history }), settings,
        randoms: new Float32Array([random]), extra: (encoder) => encoder.copyBufferToBuffer(g.logits, 0, readLogits, 0, vocab * 4) });
      await readLogits.mapAsync(MAP_READ);
      // (T226: the vocabulary's logits and no more. A buffer is made in whole 16 bytes, and GPT-2's 50257 logits left
      // 3 zeros after them: where every logit of the made-up stream was negative, the check took one of those zeros
      // for the largest, and refused a form that was right: CI's run 36869126011, 2 runs of 16)
      const logits = new Float32Array(readLogits.getMappedRange().slice(0, vocab * 4));
      readLogits.unmap();
      return { id: ids[0], logits };
    };
    const greedy = await run(wgsl.samplingSettings({ vocab, temperature: 0, topp: 0.9 }), [token], 0);
    // (T226: a classifier of floats on DP4A too where the model has outlier channels: tokenShape's floatHead)
    const packedHead = form.dp4a && !floatHead(m);
    const normedStream = norm(Float64Array.from(stream2), floats(plan.tokens.final, dim), plan.tokens.finalBias && floats(plan.tokens.finalBias, dim));
    // T232: a ternary classifier's outlier channels as the CPU has them (public/forward/engine.js's picked): taken out of the normed
    // stream before it is quantized, and their columns of the table multiplied apart, in floats
    const taken = g.take ? plan.tokens.channels.map((c) => {
      const value = normedStream[c];
      normedStream[c] = 0;
      return [c, value];
    }) : [];
    const want = product(dim, tableRow("classifier"), normedStream, rows, packedHead);
    rows.forEach((r, j) => {
      const [w, s] = tableRow("classifier")(r);
      for (const [c, value] of taken) want[j] += value * w[c] * s[Math.floor((c * s.length) / w.length)];
    });
    const logitsOff = off(rows.map((r) => greedy.logits[r]), want) / largest(want), logitsLine = packedHead ? DP4A_LINE : LOGITS_LINE;
    if (!(logitsOff <= logitsLine)) return `the logits are ${logitsOff.toExponential(2)} of their largest from JavaScript's (line ${logitsLine})`;
    if (greedy.id !== wgsl.argmaxLikeCpu(greedy.logits)) return `the greedy token is ${greedy.id}, the largest logit's ${wgsl.argmaxLikeCpu(greedy.logits)}`;
    const sampled = { temperature: 0.8, topp: 0.9, penalty: 1.3 }, history = [token, greedy.id, 0, 1];
    const drawn = await run(wgsl.samplingSettings({ vocab, ...sampled }), history, 0.7);
    const penalized = greedy.logits.slice();
    wgsl.penalizeLikeCpu(penalized, history, sampled.penalty);
    const penaltyOff = off(history.map((t) => drawn.logits[t]), history.map((t) => penalized[t])) / Math.max(largest(history.map((t) => penalized[t])), 1e-30);
    if (!(penaltyOff <= 1e-6)) return `the penalized logits are ${penaltyOff.toExponential(2)} from JavaScript's`;
    const walk = wgsl.walkLikeCpu(drawn.logits, sampled.temperature, sampled.topp), goal = 0.7 * walk.mass, band = 1e-4 * walk.mass;
    const reached = walk.tokens.filter((_, j) => walk.cumulative[j] > goal - band && (j ? walk.cumulative[j - 1] : 0) <= goal + band);
    if (!reached.includes(drawn.id)) return `the sampled token is ${drawn.id}, the CPU's walk reaches ${reached.join(" or ")}`;
    return null;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

export { chooseTokens };
