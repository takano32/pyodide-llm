// The forms of a token's layer checked on the model's own weights against JavaScript (T.152), timed and chosen.
// (T.352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T.367.3)

import { COPY_DST, MAP_READ, rowBytes, stopping, within, isLate, buffer, readBack, validated, pipelineOf, widenedHalves } from "./device.ts";
import { tablesOf } from "./weights.ts";
import { PACKINGS } from "./packings.ts";
import { halfToFloat } from "./forms.ts";
import { roomFor, BLOCK_ROUNDS } from "./block.ts";
import { chooseTokenAttention } from "./tokenattention.ts";
import { tokenCandidates, floatHead, tokenShape, tokenCodes, tokenBuffers, tokenPass, tokenStep, runTokens } from "./tokens.ts";
import { GROUP } from "../shaders/common.ts";
import { quantizedLikeCpu } from "../shaders/prompt.ts";
import { samplingState, samplingSettings } from "../shaders/run.ts";
import { penalizeLikeCpu, argmaxLikeCpu, walkLikeCpu } from "../shaders/likecpu.ts";
import { rotateLikeCpu, unrotateLikeCpu } from "../shaders/rotate.ts";
import type { Model, Thrown } from "./device.ts";
import type { TablePiece } from "./weights.ts";
import type { Chosen } from "./tokenattention.ts";
import type { TokenBuffers, TokenForm } from "./tokens.ts";
import type { GpuPacking, Table, TokensChosen, Tried, Vectors } from "../contracts/gpu/protocol.ts";

// T.368.6: this module's record, m.tokens: what tokens.ts's tokenBuffers() made, what tokenattention.ts's
// chooseTokenAttention() chose, and the form of a token's layer chosen here with its ms a step and what each form
// came to (null until choose() has them; the record itself is null where no form is right)
type Tokens = TokenBuffers & Chosen & { form: TokenForm | null; ms: number | undefined; forms: Tried[] | null };
// T.156, T.210: what a model on the GPU alone reads back for the check (firstLayer(), tableRows()); nothing of another
type Back = { firstLayer?: Awaited<ReturnType<typeof firstLayer>>; tableRows?: Awaited<ReturnType<typeof tableRows>> };
// the check's own: what says how a row is packed (a plan's matrix or table), a row as [its int8 values, its scales],
// and a list of numbers as its sums read one (an array or a typed array)
type Spec = { n: number; packing: GpuPacking };
type Row = [ArrayLike<number>, Float32Array];
type Numbers = ArrayLike<number> & { reduce(step: (sum: number, value: number) => number, from: number): number; map(each: (value: number, i: number) => number): Numbers };
// (Math.f16round is newer than the lib this is checked against, and than some browsers: heldFloats() asks)
type Rounding = Math & { f16round?: (value: number) => number };

// The forms of a token this device can make, compiled, checked and (where more than one is right) timed; the fastest
// taken (m.tokens.form), with what each came to (m.tokens.forms) and its ms a step (m.tokens.ms). plan.remembered.tokens: the
// one the page kept for this adapter, alone where it is still right here. plan.force.tokens (tests): that form alone;
// plan.force.quick: the first right one, untimed. Throws where none is right (the tokens then stay on the CPU)
// T.378.4: m.tokens is this module's to set, and no other's: a token's buffers and pipelines (tokens.ts's
// tokenBuffers() makes them), its attention (tokenattention.ts's chooseTokenAttention() chooses it), and the form of
// its layer chosen here. null where no form is right
async function chooseTokens(m: Model) {
  try {
    await choose(m);
  } catch (error) {
    // (the record is null from here: Model says what its readers hold, device.ts)
    m.tokens = null as unknown as Tokens;
    throw error;
  }
}
async function choose(m: Model) {
  const { plan } = m;
  // (tokenPass puts a parallel residual's second norm before o as a dispatch: LayerNorm's, GPT-NeoX's)
  if (plan.parallel && !plan.layerNorm) throw new Error("a parallel residual without LayerNorm is not on the GPU's tokens");
  const made = await tokenBuffers(m), chosen = await chooseTokenAttention(m);
  if (stopping) return;
  m.tokens = { ...made, ...chosen!, form: null, ms: undefined, forms: null };
  // T.156: the first layer's matrices of a model on the GPU alone are not in the shared memory: checkTokens reads them
  // back from the GPU (T.210: and the rows of the tables it reads). (The review of T.156: they go as this returns: the
  // first layer stayed in this worker for the whole visit, 113 MB of Llama 3.2 3B, 245 MB of Llama 3.1 Swallow 8B)
  const back: Back = plan.direct ? { firstLayer: await within(firstLayer(m), "reading the first layer back"),
    tableRows: await within(tableRows(m), "reading the tables' rows back") } : {};
  let forms = tokenCandidates(m);
  if (plan.force.tokens) forms = forms.filter((form) => form.name === plan.force.tokens);
  const kept = !plan.force.tokens && forms.find((form) => form.name === m.remembered?.tokens);
  if (kept) forms = [kept, ...forms.filter((form) => form !== kept)];
  const right: TokenForm[] = [], tried: Tried[] = [];
  for (const candidate of forms) {
    // the remembered one right: no other is compiled or timed (as the matrices, T.148)
    if ((plan.force.quick || (kept && right[0]?.name === kept.name)) && right.length) break;
    const form: TokenForm = { ...candidate, pipes: {} }, compiled = new Map<string, GPUComputePipeline>();
    try {
      for (const [key, code] of tokenCodes(form, tokenShape(m))) {
        if (!compiled.has(code)) compiled.set(code, await within(validated(m, () => pipelineOf(m, code)), `compiling ${form.name}`));
        form.pipes[key] = compiled.get(code)!;
      }
      const wrong = await within(checkTokens(m, form, back), `checking ${form.name}`);
      if (wrong) tried.push({ name: form.name, none: `wrong: ${wrong}` });
      else right.push(form);
    } catch (error) {
      if (isLate(error)) throw error;
      tried.push({ name: form.name, none: String((error as Thrown)?.message ?? error) });
    }
    if (stopping) return;
  }
  if (!right.length) {
    throw new Error(`no layer of a token is right on this GPU (${tried.map((f) => `${f.name}: ${f.none}`).join("; ") || `none named ${plan.force.tokens}`})`);
  }
  const ms: (number | undefined)[] = plan.force.quick ? right.map(() => undefined) : await within(timeTokens(m, right), "timing a token");
  const best = ms.reduce((b: number, t, i) => (t !== undefined && (ms[b] === undefined || t < ms[b]) ? i : b), 0);
  m.tokens.form = right[best];
  m.tokens.ms = ms[best];
  m.tokens.forms = [...tried, ...right.map((form, i) => ({ name: form.name, ms: ms[i], ...(kept && form.name === kept.name ? { remembered: true } : {}) }))];
}

// what "ready" says of a token here: the form of its layer chosen and its ms a step, what each form came to, the pieces
// of the classifier, the attention and what each of those came to; null where the tokens stay on the CPU
function tokensSaid(m: Model): TokensChosen | null {
  const g = m.tokens;
  return g?.form ? { form: g.form.name, ms: g.ms, forms: g.forms!, remembered: g.forms!.some((f) => f.remembered), pieces: m.weights.tables!.classifier.length,
    attention: g.attention.name, attentions: g.attentions } : null;
}

// ms a step of each form: runs of plan.tokens.most steps a submission from position 0 (a made-up state, the settings
// of the list's sampled models, T.151's), the forms in turn after one run each to warm up, BLOCK_ROUNDS rounds (one on
// a fallback adapter: its times are no GPU's), the median of each over its steps. The runs write the GPU's own keys
// and values of those positions, which hold nothing of forward.ts's yet (as timeBlocks)
async function timeTokens(m: Model, forms: TokenForm[]) {
  const { tokens: g } = m, most = g.most;
  const state = samplingState({ token: 1, pos: 0, history: [1] });
  const settings = samplingSettings({ vocab: g.vocab, temperature: 0.7, topp: 0.9, penalty: 1.1 });
  const randoms = new Float32Array(most).map(() => Math.fround(Math.random()) % 1);
  roomFor(m, most);
  const timed = async (form: TokenForm) => {
    const began = performance.now();
    await runTokens(m, tokenStep(m, form, most), { count: most, pos: 0, state, settings, randoms });
    return performance.now() - began;
  };
  for (const form of forms) await timed(form);
  const times: number[][] = forms.map(() => []);
  for (let round = 0; round < (m.fallback ? 1 : BLOCK_ROUNDS); round++) {
    for (let i = 0; i < forms.length; i++) times[i].push(await timed(forms[i]));
    if (stopping) break;
  }
  return times.map((list) => list.sort((a, b) => a - b)[list.length >> 1] / most);
}

// ---- T.152's check of a form on the model's own weights, against JavaScript (the drivers, the subgroups and the
// packed dot differ from device to device, and only the device can say):
//   1. EMBED of a token of the second half of the vocabulary and the first layer at position 1 (the cache's row 0 of random
//      float16), the stream h and the keys and values of position 1 read back: against JavaScript's layer in float64
//      (the weights read from the shared memory as forward.ts holds them: int6 widened by shaders.ts's sixValues; RoPE
//      from the CPU's tables; the keys and values rounded to float16 where the attention reads them (T.225's review: the
//      device's own float16 where it is a neighbour of the float64 value, heldFloats; else the nearest); on DP4A each
//      matrix's input quantized as quantize_x quantizes it, in float32). The stream: no farther than LAYER_LINE of the
//      largest change the layer made; the keys and values no farther than it of their largest. A wrong row, group,
//      head, angle or scale is a tenth and more off; on DP4A a value quantized to the other side of a rounding moves
//      the layer by about 1e-3 (T.175), and so its lines are DP4A_LINE.
//   2. the head on a made-up stream (±2), greedy: the logits of every 256th row (and the last) against JavaScript's,
//      no farther than LOGITS_LINE of their largest (DP4A: DP4A_LINE), and the id SAMPLE chose the first largest of the
//      GPU's own logits; then sampled (temperature 0.8, top-p 0.9, penalty 1.3 on a history that holds that id, a
//      random number of 0.7): the logits SAMPLE penalized in place as penalizeLikeCpu penalizes the greedy ones
//      (within 1e-6), and the id one that the CPU's walk (walkLikeCpu) over those logits reaches within 1e-4 of the
//      random number's share of the mass (T.151's line: the GPU's exp and its float32 sums).
// The reason it is wrong, or null.
const LAYER_LINE = 2e-3, LOGITS_LINE = 1e-3, DP4A_LINE = 2e-2;
// T.225's review: WGSL leaves it to the device which of its two float16 neighbours a float32 becomes (§15.7.6 Floating
// Point Conversion: "WGSL does not specify whether the higher or lower representable value is chosen, and different
// instances of such a conversion may choose differently"; pack2x16float is such a conversion), and Direct3D, where
// Chrome on Windows runs WebGPU, converts toward zero (D3D11.3 functional specification 3.2.2: "Round-to-zero must be
// used during conversion to another float format"; Dawn's HLSL writer makes pack2x16float of f32tof16). The position's
// keys and values the GPU writes are then 1 float16 spacing from the nearest's on about half of their numbers, and the
// stream computed from the nearest's was up to 3.2 times the line off the GPU's (CI, Dawn on lavapipe with every
// conversion cut toward zero: a llama.cpp form on llm-jp-3 150M 6.5e-3 at the line 2e-3; with the GPU's own keys and
// values 2.3e-6, as src/runtime/benchmark/gpu/halves.ts's heldHalves takes them: T.225). So the reference takes the device's own
// float16 (got, as floats) of a value where it is within a float16 spacing of the value and the float32 sums' noise
// (HALF_SLACK of the largest: about 10 times a sum of 2112 products's), and the nearest where it is not: a wrong key
// or value is as far from the line as before, and the keys' and values' lines below are unchanged.
const HALF_SLACK = 1e-5;
function heldFloats(x: Float64Array, got: ArrayLike<number>) {
  const round16 = (Math as Rounding).f16round ?? ((value: number) => value), slack = HALF_SLACK * x.reduce((top, value) => Math.max(top, Math.abs(value)), 0);
  return x.map((value, i) => (Math.abs(got[i] - value) <= 2 ** (Math.max(Math.floor(Math.log2(Math.abs(value))), -14) - 10) + slack ? got[i] : round16(value)));
}
export { heldFloats };  // (tests/gpu-choice-check.mjs)
// T.156: the first layer's matrices, { name: [values, scales] } as the bytes of their rows (rowBytes: int8, or T.232
// ternary codes) and a Float32Array each, read back from their buffers (a piece after another: the rows in order)
async function firstLayer(m: Model) {
  const group = GROUP, out: { [name: string]: [Uint8Array<ArrayBuffer>, Float32Array<ArrayBuffer>] } = {};
  for (const [name, matrix] of Object.entries(m.weights.matrices)) {
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
const checkCandidates = (vocab: number) => [...Array(64).keys()].map((i) => vocab - 1 - Math.floor((i * vocab) / 128));
const checkRows = (vocab: number) => [...new Set([...[...Array(Math.ceil(vocab / 256)).keys()].map((i) => i * 256), vocab - 1])];
// T.210: those rows of the tables of a model on the GPU alone, which are there only: { embedding, classifier }, Maps of
// a row to [the bytes of its values (rowBytes: int8, or T.232 ternary codes), its scales], read back in one copy each
// (a row of a piece at a time)
async function tableRows(m: Model) {
  const vocab = m.plan.tokens!.classifier.rows, specs = tablesOf(m.plan);
  const read = async (pieces: TablePiece[], spec: Table, rows: number[]) => {
    const n = rowBytes(spec), perRow = (n / GROUP) * 4;
    const target = m.device.createBuffer({ size: rows.length * (n + perRow), usage: MAP_READ | COPY_DST });
    try {
      const encoder = m.device.createCommandEncoder();
      rows.forEach((r, j) => {
        const piece = pieces.find((p) => r >= p.first && r < p.first + p.rows)!;
        encoder.copyBufferToBuffer(piece.values, (r - piece.first) * n, target, j * n, n);
        encoder.copyBufferToBuffer(piece.scales, (r - piece.first) * perRow, target, rows.length * n + j * perRow, perRow);
      });
      m.device.queue.submit([encoder.finish()]);
      await target.mapAsync(MAP_READ);
      const bytes = target.getMappedRange().slice(0);
      return new Map(rows.map((r, j): [number, [Uint8Array<ArrayBuffer>, Float32Array<ArrayBuffer>]] => [r, [new Uint8Array(bytes, j * n, n), new Float32Array(bytes, rows.length * n + j * perRow, perRow / 4)]]));
    } finally {
      target.destroy();
    }
  };
  return { embedding: await read(m.weights.tables!.embedding, specs.embedding, checkCandidates(vocab)),
    classifier: await read(m.weights.tables!.classifier, specs.classifier, checkRows(vocab)) };
}
async function checkTokens(m: Model, form: TokenForm, back: Back = {}) {
  const { plan, tokens: g, device } = m, vocab = g.vocab, dim = plan.dim, headSize = plan.headSize;
  const kvDim = plan.kvHeads * headSize, line = form.dp4a ? DP4A_LINE : LAYER_LINE;
  const floats = (address: number, n: number) => Float64Array.from(new Float32Array(m.memory.buffer, address, n));
  // row r of a matrix or a table ({ n, packing }, its values and scales at [valuesAt, scalesAt] in the shared
  // memory): [its int8 values, its scales (one a group of 32 weights; T.232, ternary: of 128)]; (T.156) of a matrix read
  // back from the GPU (firstLayer: [values, scales]); (T.210) of a table's rows read back (tableRows)
  const stored = (spec: Spec) => PACKINGS[spec.packing].stored(spec.n), scalesOf = (spec: Spec) => rowBytes(spec) / GROUP;
  // the bytes of a row as its int8 values: int6 widened, ternary codes less one, int8 as they are
  const valuesOf = (spec: Spec, bytes: Uint8Array) => PACKINGS[spec.packing].values(bytes);
  const inMemory = (spec: Spec, [valuesAt, scalesAt]: [number, number]) => (r: number): Row => [
    valuesOf(spec, new Uint8Array(m.memory.buffer, valuesAt + r * stored(spec), stored(spec))),
    // (T.533: ternary's scales are float16s there, which went up widened)
    PACKINGS[spec.packing].scale === 2 ? widenedHalves(new Uint16Array(m.memory.buffer, scalesAt + r * scalesOf(spec) * 2, scalesOf(spec)), new Float32Array(scalesOf(spec)))
      : new Float32Array(m.memory.buffer, scalesAt + r * scalesOf(spec) * 4, scalesOf(spec))];
  const readBackRows = (spec: Spec, [values, scales]: [Uint8Array, Float32Array]) => (r: number): Row => [valuesOf(spec, values.subarray(r * stored(spec), (r + 1) * stored(spec))),
    scales.subarray(r * scalesOf(spec), (r + 1) * scalesOf(spec))];
  const tableSpec = (name: "embedding" | "classifier") => (name === "embedding" ? plan.tokens!.embedding ?? plan.tokens!.classifier : plan.tokens!.classifier);
  const tableRow = (name: "embedding" | "classifier") => (back.tableRows ? (r: number): Row => {
    const [bytes, scales] = back.tableRows![name].get(r)!;
    return [valuesOf(tableSpec(name), bytes), scales];
  } : inMemory(tableSpec(name), tableSpec(name).at));
  // T.271.1: a rotated basis's signs of a width (float32, Python's, each times 1 / sqrt(block)), and R x, as the CPU's
  // rotate (the matrices read it)
  const signsOf = (n: number) => new Float32Array(m.memory.buffer, plan.rotated!.signs.find(({ width }) => width === n)!.at, n);
  const turned = (x: ArrayLike<number>, n: number) => (plan.rotated ? rotateLikeCpu(x, signsOf(n), plan.rotated.block) : x);
  // a matrix's rows (n weights each, rowOf(r): [values, scales]) times x, the rows given (packed: x quantized first; T.271.1,
  // in a rotated basis R x). The vector's groups are of 32; a row's scales cover as many of them each as they are fewer
  // (one; T.232, ternary: four)
  const product = (n: number, rowOf: (r: number) => Row, given: ArrayLike<number>, rows: number[], packed = form.dp4a) => {
    const x = turned(given, n), q = packed ? quantizedLikeCpu(Float32Array.from(x)) : null, perRow = n / GROUP;
    return Float64Array.from(rows, (r) => {
      const [w, s] = rowOf(r), each = perRow / s.length;
      let sum = 0;
      for (let b = 0; b < perRow; b++) {
        let part = 0;
        for (let i = b * GROUP; i < (b + 1) * GROUP; i++) part += w[i] * (q ? q.xq[i] : x[i]);
        sum += part * s[Math.floor(b / each)] * (q ? q.xs[b] : 1);
      }
      return sum;
    });
  };
  const all = (n: number) => [...Array(n).keys()];
  const matmul = (name: string, x: ArrayLike<number>) => {
    const matrix = plan.matrices[name];
    return product(matrix.n, back.firstLayer ? readBackRows(matrix, back.firstLayer[name]) : inMemory(matrix, matrix.layers[0]), x, all(matrix.rows));
  };
  const rms = (x: Float64Array, weights: Float64Array) => {
    const s = 1 / Math.sqrt(x.reduce((sum, v) => sum + v * v, 0) / x.length + plan.eps);
    return x.map((v, i) => weights[i] * (s * v));
  };
  // T.226: the stream's norm, RMSNorm or (T.154) LayerNorm with its bias, as the CPU's kernels have them
  const norm = (x: Float64Array, weights: Float64Array, bias: Float64Array | 0 | undefined) => {
    if (!plan.layerNorm) return rms(x, weights);
    const mean = x.reduce((sum, v) => sum + v, 0) / x.length;
    const s = 1 / Math.sqrt(x.reduce((sum, v) => sum + (v - mean) ** 2, 0) / x.length + plan.eps);
    return x.map((v, i) => weights[i] * (s * (v - mean)) + (bias as Float64Array)[i]);
  };
  // the first layer's vector of a name (plan.vectors), where the model has it; and a vector with it added
  const vectorOf = (name: keyof Vectors) => plan.vectors[name] && floats(plan.vectors[name].at, plan.vectors[name].size);
  const plus = (x: Float64Array, name: keyof Vectors) => {
    vectorOf(name)?.forEach((b, i) => { x[i] += b; });
    return x;
  };
  const largest =(xs: Numbers) => xs.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
  const off = (got: ArrayLike<number>, want: Numbers) => largest(want.map((v, i) => got[i] - v));
  const owned: GPUBuffer[] = [];
  try {
    // 1. the first layer at position 1
    // a token of the second half of the vocabulary whose row is large: the rows of tokens no text has (llm-jp-3's
    // last) are nearly 0, and their keys and values then float16's subnormals (a check of them was 15% off, CI)
    const embeddingRow = tableRow("embedding");
    const sizeOf = (t: number) => embeddingRow(t)[1].reduce((sum, v) => sum + Math.abs(v), 0);
    const token = checkCandidates(vocab).reduce((best, t) => (sizeOf(t) > sizeOf(best) ? t : best));
    const pos = 1;
    const halfBits = () => (Math.random() < 0.5 ? 0x8000 : 0) | ((13 + ((Math.random() * 3) | 0)) << 10) | ((Math.random() * 1024) | 0);
    const row0 = [0, 1].map(() => new Uint16Array(kvDim).map(halfBits));
    roomFor(m, 2);
    const { keys: cacheKeys, values: cacheValues } = m.block.cache!;
    device.queue.writeBuffer(cacheKeys[0], 0, row0[0]);
    device.queue.writeBuffer(cacheValues[0], 0, row0[1]);
    // (T.568.2: the first layer's kind writes what it goes on from, and copies out what it left: gpu/kinds.ts's first().
    // T.545.2: an LFM2's first layer is a convolution layer, T.271.3 a Qwen3.5's a linear-attention layer, which keep no
    // keys and values)
    const first = m.block.kinds[plan.kinds[0]]!.first(owned);
    const readH = buffer(m, dim * 4, MAP_READ | COPY_DST, owned);
    const { kv } = await runTokens(m, tokenPass(m, form, { to: 1, head: false, positions: pos + 1 }), { count: 1, pos, keep: true,
      state: samplingState({ token, pos, history: [token] }), extra: (encoder) => {
        encoder.copyBufferToBuffer(g.h, 0, readH, 0, dim * 4);
        first.copies(encoder);
      } });
    await readH.mapAsync(MAP_READ);
    const h = new Float32Array(readH.getMappedRange().slice(0));
    readH.unmap();
    const [gotK, gotV] = kv!.map((layers) => Float64Array.from(new Uint16Array(layers[0].buffer, layers[0].byteOffset, kvDim), halfToFloat));
    // JavaScript's layer
    const [eRow, eScales] = embeddingRow(token);
    // (T.226: with GPT-2's learned position of pos, as the CPU's embed() adds it)
    const x0 = Float64Array.from(eRow, (v, i) => v * eScales[Math.floor((i * eScales.length) / eRow.length)]);
    if (plan.tokens!.positions) floats(plan.tokens!.positions + pos * dim * 4, dim).forEach((p, i) => { x0[i] += p; });
    // (T.271.1: the embedding of a rotated basis holds R^-1 of its rows, turned back as the CPU's embed() does)
    if (plan.rotated) x0.set(unrotateLikeCpu(x0, signsOf(dim), plan.rotated.block));
    const xn = norm(x0, vectorOf("attention")!, vectorOf("attentionBias"));
    // (T.261.4.2: a Gemma 3's norm behind the attention and behind the FFN, of what joins the stream)
    const behind = (x: Float64Array, name: keyof Vectors) => (vectorOf(name) ? rms(x, vectorOf(name)!) : x);
    const { o, k, v, wrong } = await first.then({ x0, xn, pos, line, row0: [Float64Array.from(row0[0], halfToFloat), Float64Array.from(row0[1], halfToFloat)], got: [gotK, gotV],
      held: heldFloats, matmul, vectorOf, plus, rms, behind, largest, off });
    if (wrong) return wrong;
    const h1 = x0.map((value, i) => value + o[i]);
    // (T.154: GPT-NeoX's parallel residual, the FFN's norm of the layer's input; no gate, w1's bias and GELU as
    // shaders.ts's has it)
    const xn2 = norm(plan.parallel ? x0 : h1, vectorOf("ffn")!, vectorOf("ffnBias"));
    const gate = plus(matmul("w1", xn2), "b1"), up = plan.matrices.w3 && matmul("w3", xn2);
    const gelu = (value: number) => 0.5 * value * (1 + Math.tanh(Math.min(9.010913, Math.max(-9.010913, 0.7978845608028654 * (value + 0.044715 * value ** 3)))));
    // (T.261.4.2: GELU in the gate, a Gemma 3's GEGLU)
    const activated = gate.map(!up ? gelu : plan.activation === "gelu" ? (value, i) => gelu(value) * up[i] : (value, i) => (value / (1 + Math.exp(-value))) * up[i]);
    const down = behind(plus(matmul("w2", activated), "b2"), "ffnPost"), h2 = h1.map((value, i) => value + down[i]);
    const stream = off(h, h2) / largest(h2.map((value, i) => value - x0[i]));
    if (!(stream <= line)) return `the first layer's stream is ${stream.toExponential(2)} of its change from JavaScript's (line ${line})`;
    const keyOff = k ? off(gotK, k) / largest(k) : 0, valueOff = v ? off(gotV, v!) / largest(v!) : 0;
    if (!(keyOff <= line && valueOff <= line)) {
      return `the first layer's keys are ${keyOff.toExponential(2)} and its values ${valueOff.toExponential(2)} from JavaScript's (line ${line})`;
    }
    // 2. the head: greedy, then sampled
    const stream2 = new Float32Array(dim).map(() => (Math.random() - 0.5) * 4);
    const rows = checkRows(vocab);
    const readLogits = buffer(m, vocab * 4, MAP_READ | COPY_DST, owned);
    const head = tokenPass(m, form, { to: 0, embed: false });
    const run = async (settings: ReturnType<typeof samplingSettings>, history: number[], random: number) => {
      device.queue.writeBuffer(g.h, 0, stream2);
      const { ids } = await runTokens(m, head, { count: 1, pos, state: samplingState({ token, pos, history }), settings,
        randoms: new Float32Array([random]), extra: (encoder) => encoder.copyBufferToBuffer(g.logits, 0, readLogits, 0, vocab * 4) });
      await readLogits.mapAsync(MAP_READ);
      // (T.226: the vocabulary's logits and no more. A buffer is made in whole 16 bytes, and GPT-2's 50257 logits left
      // 3 zeros after them: where every logit of the made-up stream was negative, the check took one of those zeros
      // for the largest, and refused a form that was right: CI's run 36869126011, 2 runs of 16)
      const logits = new Float32Array(readLogits.getMappedRange().slice(0, vocab * 4));
      readLogits.unmap();
      return { id: ids[0], logits };
    };
    const greedy = await run(samplingSettings({ vocab, temperature: 0, topp: 0.9 }), [token], 0);
    // (T.226: a classifier of floats on DP4A too where the model has outlier channels: tokenShape's floatHead)
    const packedHead = form.dp4a && !floatHead(m);
    const normedStream = norm(Float64Array.from(stream2), floats(plan.tokens!.final, dim), plan.tokens!.finalBias && floats(plan.tokens!.finalBias, dim));
    // T.232: a ternary classifier's outlier channels as the CPU has them (src/runtime/forward/engine.ts's picked): taken out of the normed
    // stream before it is quantized, and their columns of the table multiplied apart, in floats
    const taken = g.take ? plan.tokens!.channels.map((c): [number, number] => {
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
    if (greedy.id !== argmaxLikeCpu(greedy.logits)) return `the greedy token is ${greedy.id}, the largest logit's ${argmaxLikeCpu(greedy.logits)}`;
    const sampled = { temperature: 0.8, topp: 0.9, penalty: 1.3 }, history = [token, greedy.id, 0, 1];
    const drawn = await run(samplingSettings({ vocab, ...sampled }), history, 0.7);
    const penalized = greedy.logits.slice();
    penalizeLikeCpu(penalized, history, sampled.penalty);
    const penaltyOff = off(history.map((t) => drawn.logits[t]), history.map((t) => penalized[t])) / Math.max(largest(history.map((t) => penalized[t])), 1e-30);
    if (!(penaltyOff <= 1e-6)) return `the penalized logits are ${penaltyOff.toExponential(2)} from JavaScript's`;
    const walk = walkLikeCpu(drawn.logits, sampled.temperature, sampled.topp), goal = 0.7 * walk.mass, band = 1e-4 * walk.mass;
    const reached = walk.tokens.filter((_, j) => walk.cumulative[j] > goal - band && (j ? walk.cumulative[j - 1] : 0) <= goal + band);
    if (!reached.includes(drawn.id)) return `the sampled token is ${drawn.id}, the CPU's walk reaches ${reached.join(" or ")}`;
    return null;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

export { chooseTokens, tokensSaid };
export type { Tokens };
