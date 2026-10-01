// tests/gpu-token-plan-check.mjs (T226's review): the dispatches of a generated token's layers (public/gpu.js's tokenPass)
// on a fake device, for a model of every family the GPU takes (Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX) in every form of a
// token's layer this device could make. What a shader computes is gpu-check's (on a GPU in CI); what is checked here is
// what gpu-check reads through lines of rounding noise and Q8's: the plan.
//   - how many dispatches a layer has (5, 9 and 11 for a Llama: the form with the norm on the matrix's read, with
//     NORM_QUANTIZE, with the norms apart; 7, 11 and 13 for Qwen2, 8, 12 and 14 for Qwen3, 13 and 17 for GPT-2 and
//     GPT-NeoX: TODO.md's T226);
//   - that every vector a layer reads of its own (a norm's weights, a bias, a head's norm weights) is its layer's: each
//     is made of numbers that say their layer (vector * 1e5 + layer * 1e3 + index), and what an ADD, a norm, a head's
//     norm, NORM_QUANTIZE and a fused matrix with the norm on its read start reading at is looked up in them (the
//     biases of q, k and v where tokenBuffers joins them: the first of each).
// The review of T226 found that gpu-check's packed rows (8-bit activations, a line of 3 times Q8's distance) took a
// vector of another layer on the made-up models for noise, and added two models whose lines are tight; this is the other
// way to the same ground: it cannot be fooled by noise, takes under a second and sees every form. A bias of o, w1 and w2,
// of q, k and v, the norms of the heads and the FFN's norm each read from layer 0 in every layer (the review's four
// throwaway branches) fail it.
//
// gpu.js is a worker's module and exports nothing: this copies it (and shaders.js) under .tmp/gpu-token-plan/ with an
// export of what is tested added, and imports the copy. It stands in for what start() makes of a model (uploadLayers,
// prepare, grow) with the same names gpu.js gives them; a change of those names breaks it, and the message says where.
//   node tests/gpu-token-plan-check.mjs      (exit 1 if a layer's dispatches are not as expected)
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = new URL("../", import.meta.url).pathname, work = path.join(root, ".tmp", "gpu-token-plan");
fs.mkdirSync(work, { recursive: true });
const EXPORTED = ["tokenBuffers", "tokenPass", "tokenCodes", "tokenShape", "tokenCandidates"];
fs.writeFileSync(path.join(work, "gpu.js"), `${fs.readFileSync(path.join(root, "public", "gpu.js"), "utf8")}\nexport const __test = { ${EXPORTED.join(", ")} };\n`);
fs.copyFileSync(path.join(root, "public", "shaders.js"), path.join(work, "shaders.js"));
// what gpu.js reads of a worker's (it sets onmessage at once; the rest it reads where it runs)
globalThis.onmessage = null;
globalThis.postMessage = () => {};
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { userAgent: "node", gpu: { wgslLanguageFeatures: new Set(["packed_4x8_integer_dot_product", "subgroup_id"]) } } });
let T;
try {
  const url = pathToFileURL(path.join(work, "gpu.js")).href + "?v=plan";
  T = { ...(await import(url)).__test, wgsl: await import(pathToFileURL(path.join(work, "shaders.js")).href + "?v=plan") };
} catch (error) {
  console.error(`gpu.js no longer has what this check needs (${EXPORTED.join(", ")}): ${error.message}`);
  process.exit(1);
}

// a device that records: buffers hold the bytes written to them, a bind group its entries and its shader's code
function fakeDevice() {
  let id = 0;
  return {
    features: new Set(["shader-f16", "subgroups"]),
    limits: { maxComputeWorkgroupsPerDimension: 65535, minStorageBufferOffsetAlignment: 256, maxStorageBufferBindingSize: 2 ** 30, maxBufferSize: 2 ** 30,
      maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 },
    createBuffer: ({ size, usage }) => ({ id: ++id, size, usage, content: new Uint8Array(size), destroy() {} }),
    createShaderModule: ({ code }) => ({ code }),
    createComputePipelineAsync: async ({ compute }) => ({ code: compute.module.code, getBindGroupLayout: () => ({ code: compute.module.code }) }),
    createBindGroup: ({ layout, entries }) => ({ code: layout.code, entries }),
    pushErrorScope() {},
    popErrorScope: async () => null,
    queue: {
      writeBuffer(target, offset, data, dataOffset = 0, size) {
        const typed = ArrayBuffer.isView(data), view = typed ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
        const per = typed ? data.BYTES_PER_ELEMENT ?? 1 : 1, start = dataOffset * per;
        target.content.set(view.subarray(start, size === undefined ? view.length : start + size * per), offset);
      },
      onSubmittedWorkDone: async () => {},
    },
  };
}

// the families: dim 64 and three layers each (a vector is read at layer * its size: two layers would not tell layer 1's
// from 0 + size)
const DIM = 64, LAYERS = 3, VOCAB = 320, SEQ = 64;
const FAMILIES = {
  llama: { heads: 4, kvHeads: 2, headSize: 16, hidden: 128, layerNorm: false, parallel: false, bias: false, qkNorm: false, turned: 16, eps: 1e-5 },
  qwen2: { heads: 4, kvHeads: 2, headSize: 16, hidden: 128, layerNorm: false, parallel: false, bias: true, qkNorm: false, turned: 16, eps: 1e-6 },
  qwen3: { heads: 4, kvHeads: 2, headSize: 32, hidden: 128, layerNorm: false, parallel: false, bias: false, qkNorm: true, turned: 32, eps: 0.5 },
  gpt2: { heads: 4, kvHeads: 4, headSize: 16, hidden: 256, layerNorm: true, parallel: false, bias: true, qkNorm: false, turned: 0, eps: 1e-5, positions: true, outliers: true },
  neox: { heads: 4, kvHeads: 4, headSize: 16, hidden: 256, layerNorm: true, parallel: true, bias: true, qkNorm: false, turned: 4, eps: 1e-5 },
};
// the dispatches of a layer in the forms a device with subgroups and the packed dot product can make, in the order
// of gpu.js's TOKEN_FORMS (the forms with subgroups have the dispatches of those without): the norm on the matrix's
// read, with subgroups, DP4A with NORM_QUANTIZE, DP4A with the norms apart (a LayerNorm has no NORM_QUANTIZE form)
const EXPECTED = { llama: [5, 5, 9, 11], qwen2: [7, 7, 11, 13], qwen3: [8, 8, 12, 14], gpt2: [13, 13, 17], neox: [13, 13, 17] };

function modelOf(family) {
  const f = FAMILIES[family], { wgsl } = T, device = fakeDevice();
  const qDim = f.heads * f.headSize, kvDim = f.kvHeads * f.headSize, gated = !f.layerNorm;
  const memory = { buffer: new ArrayBuffer(1 << 22) }, floats = new Float32Array(memory.buffer);
  let top = 0;
  const vectors = {}, order = [];
  const vector = (name, size) => {
    const index = order.length, at = top;
    order.push(name);
    for (let l = 0; l < LAYERS; l++) for (let i = 0; i < size; i++) floats[at / 4 + l * size + i] = index * 1e5 + l * 1e3 + i;
    top += LAYERS * size * 4;
    vectors[name] = { at, size };
  };
  vector("attention", DIM);
  vector("ffn", DIM);
  if (f.layerNorm) { vector("attentionBias", DIM); vector("ffnBias", DIM); }
  if (f.bias) { vector("bq", qDim); vector("bk", kvDim); vector("bv", kvDim); }
  if (f.qkNorm) { vector("qNorm", f.headSize); vector("kNorm", f.headSize); }
  if (f.layerNorm) { vector("bo", DIM); vector("b1", f.hidden); vector("b2", DIM); }
  const room = (count) => { const at = top; top += count * 4; return at; };
  const plan = { dim: DIM, hidden: f.hidden, layers: LAYERS, heads: f.heads, kvHeads: f.kvHeads, headSize: f.headSize, turned: f.turned, seqLen: SEQ,
    kvStart: 8, eps: f.eps, layerNorm: f.layerNorm, parallel: f.parallel, batch: 64, matrices: {}, vectors, rows: 0, force: {},
    tokens: { classifier: { rows: VOCAB, n: DIM, six: false, at: [0, 0] }, embedding: null, final: room(DIM), finalBias: f.layerNorm ? room(DIM) : 0,
      positions: f.positions ? room(SEQ * DIM) : 0, outliers: Boolean(f.outliers), ids: 0, most: 4 },
    cos: room(SEQ * f.headSize / 2), sin: room(SEQ * f.headSize / 2) };
  const m = { device, plan, wgsl, owned: [], memory, limit: 2 ** 30, fallback: false, info: {} };
  const buffer = (size, usage = 0x80) => device.createBuffer({ size: Math.max(16, size), usage });
  const piece = (rows, n) => ({ first: 0, rows, values: buffer(rows * n), scales: buffer(rows * n / 8) });
  m.tables = { classifier: [piece(VOCAB, DIM)] };
  m.tables.embedding = m.tables.classifier;
  m.vectors = Object.fromEntries(Object.entries(vectors).map(([name, { size }]) => [name, buffer(LAYERS * size * 4)]));
  m.finalNorm = buffer(DIM * 4);
  m.finalBias = f.layerNorm ? buffer(DIM * 4) : null;
  m.positions = f.positions ? buffer(SEQ * DIM * 4) : null;
  m.angleTable = buffer(SEQ * f.headSize * 4);
  const rows = { wq: qDim, wk: kvDim, wv: kvDim, wo: DIM, w1: f.hidden, w2: DIM, ...(gated ? { w3: f.hidden } : {}) };
  const columns = { wq: DIM, wk: DIM, wv: DIM, wo: qDim, w1: DIM, w2: f.hidden, w3: DIM };
  m.matrices = Object.fromEntries(Object.entries(rows).map(([name, count]) => [name, { rows: count, n: columns[name], pieces: [{ first: 0, rows: count,
    layers: Array.from({ length: LAYERS }, () => [buffer(count * columns[name]), buffer(count * columns[name] / 8)]) }] }]));
  m.joined = Array.from({ length: LAYERS }, () => ({ qkv: [buffer((qDim + 2 * kvDim) * DIM), buffer((qDim + 2 * kvDim) * DIM / 8)],
    ...(gated ? { gateUp: [buffer(2 * f.hidden * DIM), buffer(2 * f.hidden * DIM / 8)] } : {}) }));
  m.cache = { capacity: 8, owned: [], keys: Array.from({ length: LAYERS }, () => buffer(8 * kvDim * 2)), values: Array.from({ length: LAYERS }, () => buffer(8 * kvDim * 2)) };
  const pipeline = (code) => ({ code, getBindGroupLayout: () => ({ code }) });
  m.norm = pipeline(f.layerNorm ? wgsl.LAYER_NORM : wgsl.RMSNORM);
  m.headNorm = pipeline(wgsl.HEAD_NORM);
  m.add = pipeline(wgsl.ADD);
  m.activation = pipeline(gated ? wgsl.SWIGLU : wgsl.GELU);
  m.quantize = pipeline(wgsl.QUANTIZE);
  m.attention = { pipeline: pipeline("the prompt's attention tiles") };
  return { m, plan, order, floats, vectors, pipeline };
}

const words = (buffer) => Array.from(new Uint32Array(buffer.content.buffer, 0, Math.min(12, buffer.size >> 2)));

// the vectors a dispatch of layer `layer` reads of its own, and the layer each is the one of: [{ name, layer, ok }]
function readsOf(M, labelOf, [, group]) {
  const { m } = M, out = [], entry = (binding) => { const e = group.entries.find((x) => x.binding === binding); return e && (e.resource.buffer ?? e.resource); };
  const nameOf = (buffer) => Object.entries(m.vectors).find(([, v]) => v === buffer)?.[0];
  const of = (name, at, label = name) => {
    const index = M.order.indexOf(name), value = M.floats[M.vectors[name].at / 4 + at], layer = Math.round((value - index * 1e5) / 1e3);
    out.push({ name: label, layer, ok: Math.abs(value - (index * 1e5 + layer * 1e3 + (at - layer * M.vectors[name].size))) < 1e-3 });
  };
  const label = labelOf(group.code);
  if (label === "RMSNORM" || label === "LAYER_NORM") {
    const at = words(entry(label === "LAYER_NORM" ? 4 : 3))[1];
    of(nameOf(entry(1)), at);
    if (label === "LAYER_NORM") of(nameOf(entry(2)), at);
  } else if (label === "HEAD_NORM") {
    of(nameOf(entry(1)), words(entry(2))[1]);
  } else if (label === "NORM_QUANTIZE") {
    of(nameOf(entry(1)), words(entry(4))[1]);
  } else if (label === "ADD") {
    const holder = entry(1), name = nameOf(holder), at = words(entry(2))[1];
    if (name) of(name, at);
    else if (holder === m.gen.qkvBias) {
      // the biases of q, k and v joined a layer after another by tokenBuffers: the first of each where this ADD starts
      const { heads, kvHeads, headSize } = M.plan, qDim = heads * headSize, kvDim = kvHeads * headSize;
      const joined = new Float32Array(holder.content.buffer, 0, holder.content.length / 4);
      for (const [vector, offset] of [["bq", 0], ["bk", qDim], ["bv", qDim + kvDim]]) {
        const index = M.order.indexOf(vector), value = joined[at + offset], layer = Math.round((value - index * 1e5) / 1e3);
        out.push({ name: `${vector} (joined)`, layer, ok: Math.abs(value - (index * 1e5 + layer * 1e3)) < 1e-3 });
      }
    }
  } else if (label.startsWith("matrix:") && entry(4) && nameOf(entry(4))) {
    // a fused matrix with the norm on its read: the norm's weights at binding 4, where they start in Params' word 5
    of(nameOf(entry(4)), words(entry(3))[5]);
  }
  return out;
}

let failures = 0, reads = 0, forms = 0;
const fail = (text) => { failures++; console.error(`FAILED ${text}`); };
for (const family of Object.keys(FAMILIES)) {
  const counts = [];
  const candidates = T.tokenCandidates(modelOf(family).m);
  for (const candidate of candidates) {
    const M = modelOf(family), { m } = M;
    m.gen = await T.tokenBuffers(m);
    m.gen.attention = { tiles: true, pipeline: m.attention.pipeline };
    const form = { ...candidate, pipes: {} };
    for (const [key, code] of T.tokenCodes(T.wgsl, form, T.tokenShape(m))) form.pipes[key] = M.pipeline(code);
    const labels = new Map([[T.wgsl.RMSNORM, "RMSNORM"], [T.wgsl.LAYER_NORM, "LAYER_NORM"], [T.wgsl.HEAD_NORM, "HEAD_NORM"], [T.wgsl.ADD, "ADD"],
      [T.wgsl.GELU, "GELU"], [T.wgsl.SWIGLU, "SWIGLU"], [T.wgsl.QUANTIZE, "QUANTIZE"], [T.wgsl.NORM_QUANTIZE, "NORM_QUANTIZE"], [T.wgsl.TOKEN_ROPE, "TOKEN_ROPE"],
      ["the prompt's attention tiles", "attention"]]);
    for (const pipe of Object.values(form.pipes)) if (!labels.has(pipe.code)) labels.set(pipe.code, `matrix:${Object.keys(form.pipes).filter((k) => form.pipes[k].code === pipe.code).join("=")}`);
    const labelOf = (code) => labels.get(code) ?? "?";
    let size;
    for (let l = 0; l < LAYERS; l++) {
      const list = T.tokenPass(m, form, { from: l, to: l + 1, head: false, embed: false, positions: 8 });
      size ??= list.length;
      if (list.length !== size) fail(`${family}, ${candidate.name}: layer ${l} has ${list.length} dispatches, layer 0 ${size}`);
      for (const dispatch of list) {
        for (const read of readsOf(M, labelOf, dispatch)) {
          reads++;
          if (!read.ok || read.layer !== l) fail(`${family}, ${candidate.name}: layer ${l}'s ${labelOf(dispatch[1].code)} reads ${read.name} of layer ${read.layer}${read.ok ? "" : " (not at the start of a layer's part)"}`);
        }
      }
    }
    counts.push(size);
    forms++;
  }
  if (JSON.stringify(counts) !== JSON.stringify(EXPECTED[family])) fail(`${family}: a layer has ${counts.join(", ")} dispatches in its forms, expected ${EXPECTED[family].join(", ")} (TODO.md's T226)`);
}
if (reads < 250) fail(`only ${reads} reads of a vector of a layer were checked (312 when this was written): the check no longer sees the plan's vectors`);
console.log(`${failures ? "FAILED" : "ok"}: ${Object.keys(FAMILIES).length} families, ${forms} forms of a token's layer, ${reads} reads of a vector of a layer`);
process.exit(failures ? 1 : 0);
