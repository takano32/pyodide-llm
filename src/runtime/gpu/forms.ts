// The forms of a prompt's shaders on this device, checked against JavaScript, timed and chosen: the attention's tiles
// and the matrices' tiled shaders (T146, T147), with how a matrix's pieces are bound and dispatched.
// (T352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T367.3)

import { STORAGE, COPY_DST, COPY_SRC, stopping, within, buffer, uniform, readBack, validated, pipelineOf, bind, dispatch } from "./device.ts";
import { GROUP } from "../shaders/common.ts";
import { ternaryValues, tiledOff } from "../shaders/prompt.ts";
import { FLASH_Q_TILE, flashShape, flashTile } from "../shaders/attention.ts";
import { devicePromptForms } from "../shaders/key.ts";
import type { Model, Thrown } from "./device.ts";
import type { Quantizing } from "./block.ts";
import type { PromptForm } from "../shaders/prompt.ts";
import type { FlashShape } from "../shaders/attention.ts";
import type { Tried } from "../contracts/gpu/protocol.ts";

// T368.6: this module's record, m.forms: the prompt's attention chosen here (chooseAttention() makes the record with
// it), the tiled shader chosen (chooseMatrices(): null until then, read with a ! after it), and what each tried came to.
// A tiled shader is one of shaders/prompt.ts's forms with its pipeline, and whether it is the one the page remembered
type Attention = { name: string; pipeline: GPUComputePipeline; shape: FlashShape };
type Tiled = PromptForm & { pipeline: GPUComputePipeline; remembered?: boolean };
type Forms = { attention: Attention; form: Tiled | null; tried: Tried[] };
// timeForms(): a matrix of the layer, what it reads and writes (up is null where the FFN has no gate, and then no w3
// is there to multiply), whether it adds, and the quantization of its input where it is the first to read it
type Timed = [string, GPUBuffer, GPUBuffer | null, boolean?, Quantizing?];

// The right tiled shaders are timed together, in turn, on the model's first layer (its seven matrices by a whole block,
// with the quantizations of a packed one): a submission of n passes of the layer and one of 2n, their difference the
// time of n passes (what a submission costs besides its work is in both and drops out: 2.6 to 8.4 ms waited for on
// the owner's Android, T134, where llm-jp-3 150M's gate by 64 tokens is 0.6 ms of work), n doubled until a submission
// of n takes TIMED_MS (at most MOST_PASSES), PAIRS pairs a shader, the shaders in turn within each round (a device that
// warms up or is loaded meanwhile falls on all of them alike: T168's review), the median of each shader's pairs. On a
// fallback adapter (the CPU in the GPU's place: its times are no GPU's) one pair of one pass
const TIMED_MS = 20, MOST_PASSES = 256, PAIRS = 5;

// the tiled shaders of T146 this device can make (shaders.ts's promptForms)
const candidates = ({ device, ternary }: Model) => devicePromptForms(device, ternary);

// ---- the attention: llama.cpp's flash attention with tiles, with f16 in the workgroup's memory where there is
// shader-f16 and its subgroups where there are (and subgroup_id), else the same without them; each checked first
// against JavaScript (checkAttention), the next one tried where it is wrong
async function chooseAttention(m: Model) {
  const { device, plan } = m;
  const half = device.features.has("shader-f16");
  const subgroups = device.features.has("subgroups") && Boolean(navigator.gpu.wgslLanguageFeatures?.has("subgroup_id"));
  const limits = { memory: device.limits.maxComputeWorkgroupStorageSize, threads: threadsOf(device),
    subgroupMin: m.info.subgroupMinSize, subgroupMax: m.info.subgroupMaxSize };
  const tried: string[] = [];
  const nameOf = (option: { half: boolean; subgroups: boolean }) => `llama.cpp flash attention tiles${option.half ? ", f16" : ""}${option.subgroups ? ", subgroups" : ""}`;
  // T148: the one the page remembers for this adapter first (it was the first right one then: the other was wrong)
  const options = [{ half, subgroups }, { half: false, subgroups: false }].filter((o, i) => i === 0 || o.half !== half || o.subgroups !== subgroups)
    .sort((a, b) => ((nameOf(b) === m.remembered?.attention) as unknown as number) - ((nameOf(a) === m.remembered?.attention) as unknown as number));
  for (const option of options) {
    const shape = flashShape({ headSize: plan.headSize, ...option, ...limits });
    const name = nameOf(option);
    if (plan.force.attention && plan.force.attention !== name) continue;
    if (shape.none) {
      tried.push(`${name}: ${shape.none}`);
      continue;
    }
    try {
      const pipeline = await within(validated(m, () => pipelineOf(m, flashTile(shape))), `compiling ${name}`);
      const wrong = await within(checkAttention(m, pipeline), `checking ${name}`);
      if (!wrong) {
        // (T378.4: this module's record, which it alone sets: the attention, then chooseMatrices()'s form and what
        // each one tried came to)
        m.forms = { attention: { name, pipeline, shape }, form: null, tried: [] };
        return;
      }
      tried.push(`${name}: ${wrong}`);
    } catch (error) {
      if ((error as Thrown)?.late) throw error;
      tried.push(`${name}: ${(error as Thrown)?.message ?? error}`);
    }
  }
  throw new Error(`no attention is right on this GPU (${tried.join("; ") || `none named ${plan.force.attention}`})`);
}

// The attention on made-up numbers against JavaScript's: 9 tokens at positions 61 to 69 (two tiles of 4 and one
// token of a third), 4 heads of q on 2 of keys and values, 70 positions (a KV_TILE of 64 and a part of one) of keys and
// values as float16 of random bits between 2^-3 and 4 in size. Each token's output against its softmax over the
// positions up to its own, no more than LINE of the largest |value| of its head: f16 weights in the workgroup's
// memory are within 2^-10 of theirs (each rounded either way, WGSL leaves the direction to the device) and so the
// output within about 1e-3 of the largest value (the sum of the weights is float32), while a wrong mask, head or
// tile is off by a tenth and more.
const LINE = 4e-3;
async function checkAttention(m: Model, pipeline: GPUComputePipeline) {
  const { plan } = m, size = plan.headSize, heads = 4, kvHeads = 2, tokens = 9, pos = 61, positions = pos + tokens;
  const kvDim = kvHeads * size, qDim = heads * size, scale = 1 / Math.sqrt(size);
  const q = new Float32Array(tokens * qDim).map(() => Math.random() * 2 - 1);
  const halfBits = () => ((Math.random() < 0.5 ? 0x8000 : 0) | ((12 + ((Math.random() * 5) | 0)) << 10) | ((Math.random() * 1024) | 0));
  const keys = new Uint16Array(positions * kvDim).map(halfBits), values = new Uint16Array(positions * kvDim).map(halfBits);
  const owned: GPUBuffer[] = [];
  const make = (data: GPUAllowSharedBufferSource, usage = STORAGE | COPY_DST) => {
    const made = buffer(m, data.byteLength, usage, owned);
    m.device.queue.writeBuffer(made, 0, data);
    return made;
  };
  try {
    const out = buffer(m, tokens * qDim * 4, STORAGE | COPY_SRC, owned);
    const params = new ArrayBuffer(16);
    new Uint32Array(params, 0, 2).set([heads, kvHeads]);
    new Float32Array(params, 8, 1)[0] = scale;
    const group = bind(m, pipeline, [make(q), make(keys), make(values), out, uniform(m, params, owned),
      uniform(m, new Uint32Array([tokens, pos, 0, 0]), owned)]);
    const encoder = m.device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, pipeline, group, heads * Math.ceil(tokens / FLASH_Q_TILE));
    pass.end();
    m.device.queue.submit([encoder.finish()]);
    const got = new Float32Array(await readBack(m, out, tokens * qDim * 4));
    const k = Float32Array.from(keys, halfToFloat), v = Float32Array.from(values, halfToFloat);
    let worst = 0;
    for (let t = 0; t < tokens; t++) {
      for (let h = 0; h < heads; h++) {
        const kv = Math.floor(h / (heads / kvHeads)) * size, row = t * qDim + h * size, seen = pos + t + 1;
        const scores = new Float64Array(seen);
        for (let p = 0; p < seen; p++) {
          for (let d = 0; d < size; d++) scores[p] += q[row + d] * scale * k[p * kvDim + kv + d];
        }
        const most = Math.max(...scores);
        let sum = 0, largest = 0;
        for (let p = 0; p < seen; p++) sum += (scores[p] = Math.exp(scores[p] - most));
        for (let p = 0; p < positions; p++) for (let d = 0; d < size; d++) largest = Math.max(largest, Math.abs(v[p * kvDim + kv + d]));
        for (let d = 0; d < size; d++) {
          let want = 0;
          for (let p = 0; p < seen; p++) want += scores[p] * v[p * kvDim + kv + d];
          worst = Math.max(worst, Math.abs(got[row + d] - want / sum) / largest);
        }
      }
    }
    return worst <= LINE ? null : `its output is ${worst.toExponential(2)} of the largest value from JavaScript's (line ${LINE})`;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}
function halfToFloat(h: number) {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, fraction = h & 1023;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// ---- the matrices: the tiled shaders of T146 that this device can make (shaders.ts's promptForms), each checked on a
// small matrix against JavaScript (checkForm: subgroups, f16 rounding and drivers differ from device to device, and
// only the device can say), the right ones timed on the model's own gate matrix by a whole block, and the fastest
// taken. forms: what each came to (ms, or why not), for the console. plan.force.matrices (tests only): that form alone, untimed.
async function chooseMatrices(m: Model) {
  const { plan } = m;
  let forms = candidates(m);
  if (plan.force.matrices) forms = forms.filter((form) => form.name === plan.force.matrices);
  // T148: the shader the page remembers for this adapter, alone, where it is one this device still makes and it is
  // still right here (a driver may have changed under the same names): no other is compiled or timed. Else all of them
  const kept = !plan.force.matrices && forms.find((form) => !form.none && form.name === m.remembered?.matrices);
  if (kept) {
    try {
      const tiled = { ...kept, remembered: true, pipeline: await within(validated(m, () => pipelineOf(m, kept.code, kept.constants)), `compiling ${kept.name}`) };
      const wrong = await within(checkForm(m, tiled), `checking ${kept.name}`);
      if (!wrong) {
        m.forms.tried.push({ name: kept.name, remembered: true });
        m.forms.form = tiled;
        return;
      }
      m.forms.tried.push({ name: kept.name, none: `remembered, but wrong now: ${wrong}` });
    } catch (error) {
      if ((error as Thrown)?.late) throw error;
      m.forms.tried.push({ name: kept.name, none: `remembered, but ${(error as Thrown)?.message ?? error}` });
    }
    if (stopping) return;
  }
  const right: Tiled[] = [];
  for (const form of forms) {
    if (plan.force.quick && right.length) break;  // T148: the page's tests, the first right one untimed
    if (form === kept) continue;  // wrong just now
    if (form.none) {
      m.forms.tried.push({ name: form.name, none: form.none });
      continue;
    }
    try {
      const pipeline = await within(validated(m, () => pipelineOf(m, form.code, form.constants)), `compiling ${form.name}`);
      const tiled = { ...form, pipeline };
      const wrong = await within(checkForm(m, tiled), `checking ${form.name}`);
      if (wrong) m.forms.tried.push({ name: form.name, none: `wrong: ${wrong}` });
      else right.push(tiled);
    } catch (error) {
      if ((error as Thrown)?.late) throw error;
      m.forms.tried.push({ name: form.name, none: String((error as Thrown)?.message ?? error) });
    }
    if (stopping) return;
  }
  if (!right.length) throw new Error(`no tiled shader is right on this GPU (${m.forms.tried.map((f) => `${f.name}: ${f.none}`).join("; ") || `none named ${plan.force.matrices}`})`);
  const ms = right.length > 1 ? await within(timeForms(m, right), "timing the tiled shaders") : [0];
  right.forEach((form, i) => m.forms.tried.push({ name: form.name, ms: ms[i] }));
  m.forms.form = right[ms.indexOf(Math.min(...ms))];
}
// what "ready" says of the forms chosen: the tiled shader and the attention by their names, what each tiled shader
// tried came to, and whether the one taken is the one the page remembered
const formsSaid = (m: Model) => ({ form: m.forms.form!.name, attention: m.forms.attention.name, forms: m.forms.tried, remembered: Boolean(m.forms.form!.remembered) });
// the threads of a workgroup of one dimension this device takes
const threadsOf = (device: GPUDevice) => Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX);

// layer l's matrix name by the tokens of from into to (added where add): a bind group of form and its rows for each
// piece (T155), which writes its rows of to (bound from its first row; a token's row of to is the whole matrix's)
const productGroups = (m: Model, form: Tiled, name: string, l: number, from: GPUBuffer, to: GPUBuffer, add?: boolean, owned = m.owned) => {
  const { rows, n, pieces } = m.weights.matrices[name], { xq, xs, step } = m.block;
  return pieces.map(({ first, rows: count, layers }) => {
    const out = first ? { buffer: to, offset: first * 4, size: to.size - first * 4 } : to;
    return { rows: count, group: bind(m, form.pipeline, [...layers[l], form.packed ? xq : from, out,
      uniform(m, new Uint32Array([count, n / 4, n / GROUP, 0, n, rows, add ? 1 : 0, 0]), owned), step, ...(form.packed ? [xs] : [])]) };
  });
};
// the workgroups of form for rows by count tokens: the tiles numbered over x, then y (as T146's shaders number them)
function multiply(m: Model, pass: GPUComputePassEncoder, form: Tiled, group: GPUBindGroup, rows: number, count: number) {
  const tiles = Math.ceil(rows / form.tile.rows) * Math.ceil(count / form.tile.tokens);
  const across = Math.min(tiles, m.device.limits.maxComputeWorkgroupsPerDimension);
  dispatch(pass, form.pipeline, group, across, Math.ceil(tiles / across));
}

// T146's check (src/runtime/benchmark/gpu/check.js's checkTiled) of a form: 300 rows of 544 (17 groups of 32: a part of a tile
// of rows everywhere), 11 and 70 tokens (a part of a tile of tokens; two or one and a part), and 11 tokens whose x and
// y are wider than the product (xStride 608, yStride 320), each product twice into the same y (the second added:
// shape.add) against JavaScript's (shaders.ts's tiledOff). The reason it is wrong, or null
// T232, a ternary form: rows of 640 weights (5 groups of 128 with a scale each, 20 of the activations' groups of 32:
// a weight's scale holds for four steps of the width, and the next is another), the weights random bytes of codes (all
// four codes, the one no file has too: every bit of a word matters), against JavaScript's on the codes' values
// (shaders.ts's ternaryValues)
async function checkForm(m: Model, form: Tiled) {
  const { device } = m, { quantize } = m.block, rows = 300, ternary = Boolean(form.ternary), n = ternary ? 640 : 544, perRow = n / GROUP;
  const scaled = ternary ? 4 * GROUP : GROUP;  // the weights a scale
  const stored = ternary ? new Uint8Array((rows * n) / 4).map(() => (Math.random() * 256) | 0) : new Int8Array(rows * n).map(() => (Math.random() * 256) | 0);
  const w = ternary ? ternaryValues(stored) : stored, s = new Float32Array((rows * n) / scaled).map(() => Math.random() * 0.01);
  for (const { tokens, wider } of [{ tokens: 11, wider: 0 }, { tokens: 70, wider: 0 }, { tokens: 11, wider: 64 }]) {
    const xStride = n + wider, yStride = rows + (wider ? 20 : 0), owned: GPUBuffer[] = [];
    const x = new Float32Array(tokens * xStride).map(() => (Math.random() - 0.5) * 2);
    const make = (data: GPUAllowSharedBufferSource, usage = STORAGE | COPY_DST | COPY_SRC) => {
      const made = buffer(m, data.byteLength, usage, owned);
      device.queue.writeBuffer(made, 0, data);
      return made;
    };
    try {
      const io = { xq: buffer(m, tokens * xStride, STORAGE | COPY_SRC, owned), xs: buffer(m, tokens * (xStride / GROUP) * 4, STORAGE | COPY_SRC, owned),
        step: uniform(m, new Uint32Array([tokens, 0, 0, 0]), owned) };
      const wb = make(stored), sb = make(s), xb = make(x), y = make(new Float32Array(tokens * yStride));
      const shape = (add: boolean) => uniform(m, new Uint32Array([rows, n / 4, perRow, 0, xStride, yStride, add ? 1 : 0, 0]), owned);
      const group = (add: boolean) => bind(m, form.pipeline, [wb, sb, form.packed ? io.xq : xb, y, shape(add), io.step, ...(form.packed ? [io.xs] : [])]);
      const groups = await validated(m, () => [group(false), group(true)]);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      if (form.packed) {
        dispatch(pass, quantize, bind(m, quantize, [xb, io.xq, io.xs, uniform(m, new Uint32Array([n, xStride, 0, 0]), owned), io.step]),
          Math.ceil(n / GROUP / 64), tokens);
      }
      groups.forEach((g) => multiply(m, pass, form, g, rows, tokens));
      pass.end();
      device.queue.submit([encoder.finish()]);
      const got = new Float32Array(await readBack(m, y, tokens * yStride * 4));
      const xq = form.packed ? new Int8Array(await readBack(m, io.xq, tokens * xStride)) : null;
      const xs = form.packed ? new Float32Array(await readBack(m, io.xs, tokens * (xStride / GROUP) * 4)) : null;
      const { wrong } = tiledOff({ w, s, x, got, xq, xs, rows, n, tokens, xStride, yStride, half: form.half, group: scaled });
      if (wrong) return `${wrong} (${tokens} tokens${wider ? ", wider x and y" : ""})`;
    } finally {
      owned.forEach((b) => b.destroy());
    }
  }
  return null;
}

// ms of a pass of the model's first layer with each of forms (see TIMED_MS): the seven matrices by plan.batch tokens,
// each packed one's inputs quantized first
async function timeForms(m: Model, forms: Tiled[]) {
  const { device, plan } = m, owned: GPUBuffer[] = [], layer: { rows: number; group: GPUBindGroup; quantize: Quantizing | null | undefined }[][] = [];
  // (the block's own buffers and quantizations, block.ts's: read here, written by a pass)
  const { step, x, xb, q, k, v, gate, up, ffnInput, quantize: quantizer, quantizeXb, quantizeAttention, quantizeFfn, quantizeGate } = m.block;
  try {
    device.queue.writeBuffer(step, 0, new Uint32Array([plan.batch, 0, 0, 0]));
    const products = ([["wq", xb, q, false, quantizeXb], ["wk", xb, k], ["wv", xb, v], ["wo", xb, x, true, quantizeAttention],
      ["w1", ffnInput, gate, false, quantizeFfn], ["w3", ffnInput, up], ["w2", gate, x, true, quantizeGate]] satisfies Timed[])
      .filter(([name]) => m.weights.matrices[name]);
    // (a matrix's input quantized once, before its first piece)
    for (const form of forms) {
      layer.push(await validated(m, () => products.flatMap(([name, from, to, add, quantize]) =>
        productGroups(m, form, name, 0, from, to!, add, owned).map((piece, i) => ({ ...piece, quantize: i ? null : quantize })))));
    }
    const submission = async (i: number, passes: number) => {
      const form = forms[i], encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let p = 0; p < passes; p++) {
        for (const { group, rows, quantize } of layer[i]) {
          if (form.packed && quantize) dispatch(pass, quantizer, quantize.group, quantize.x, plan.batch);
          multiply(m, pass, form, group, rows, plan.batch);
        }
      }
      pass.end();
      const began = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    const passes: number[] = [], differences: number[][] = forms.map(() => []);
    for (let i = 0; i < forms.length; i++) {
      await submission(i, 1);  // warm
      let n = 1;
      while (!m.fallback && n < MOST_PASSES && (await submission(i, n)) < TIMED_MS) n *= 2;
      passes.push(n);
    }
    for (let round = 0; round < (m.fallback ? 1 : PAIRS); round++) {
      for (let i = 0; i < forms.length; i++) {
        const once = await submission(i, passes[i]), twice = await submission(i, 2 * passes[i]);
        differences[i].push((twice - once) / passes[i]);
      }
    }
    return differences.map((d) => d.sort((a, b) => a - b)[d.length >> 1]);
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

export { TIMED_MS, MOST_PASSES, PAIRS, chooseAttention, LINE, halfToFloat, chooseMatrices, formsSaid, threadsOf, productGroups,
  multiply };
export type { Forms, Tiled, Attention };
