// unchanged-gpusteps.mjs (T568.1)
// What the model's GPU worker (src/runtime/gpu/) asks of a device for a few made-up models, as one JSON object of
// hashes: every buffer it makes (their sizes and usages, counted), the shaders it compiles (counted), a prompt's blocks
// (the writes, the clears, the copies and the dispatches of two blocks, the second going on from the state the CPU
// staged), and a token's steps in every form a device with subgroups and the packed dot may take (the dispatches of a
// run, of the first layer's check and of the head's, and what a run copies back). Each dispatch is its shader, the
// buffers it binds and the workgroups; a buffer is named by where it first comes in what is written down (the order
// the worker makes its buffers in is not held, the order it binds and dispatches them in is), a uniform by its bytes.
// No shader runs (tests/fakes/webgpu.mjs): what is held is that a moved branch asks for the same work, not that the
// work is right (CI's gpu-check is).
//
//   node tests/unchanged-gpusteps.mjs <the root of a tree>
//   GPUSTEPS_TRACE=<a file> ...: the whole record as text besides, to read a difference by (a diff of two such files)
//
// The models: a Llama; a Qwen3 (biases of q, k and v, norms of the heads); a Gemma 3 (GEGLU, the norms behind, a
// window and a local layer); an LFM2 (convolution layers around an attending one); a Qwen3.5 (two linear-attention
// layers and an attending one with the gate of its output). Each with a packed tiled form and a float one.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { treeOf } from "./tree.mjs";
import { standIn } from "./fakes/webgpu.mjs";

const root = path.resolve(process.argv[2] ?? ".");
// (the worker's global, as a worker has it)
globalThis.self = globalThis;
globalThis.postMessage = () => {};
const at = (name) => treeOf(root).runtimeUrl(name);
const device_ = await import(at("gpu/device.js"));
const { upload } = await import(at("gpu/weights.js"));
const block_ = await import(at("gpu/block.js"));
const tokens_ = await import(at("gpu/tokens.js"));

const DIM = 128, HIDDEN = 256, HEADS = 2, KV_HEADS = 1, HEAD = 16, BATCH = 8, CONTEXT = 64, VOCAB = 96, GROUP = 32, MOST = 4;
const Q_DIM = HEADS * HEAD, KV_DIM = KV_HEADS * HEAD, KV_ROW = KV_DIM * 2;
const LINEAR = { keyHeads: 1, valueHeads: 2, keyDim: 16, valueDim: 16, taps: 4 }, KEYS = LINEAR.keyHeads * LINEAR.keyDim, READ = LINEAR.valueHeads * LINEAR.valueDim;
const MIXED = 2 * KEYS + READ, CONV_TAPS = 3;

/** A plan as forward/gpuside.ts hands one (a Started), over a memory it lays out: kinds, every layer's kind; more, the
 * model's own; matrices and vectors, [rows, n, of] and [size, of] by name. */
function planned({ kinds, matrices, vectors, more = {} }) {
  let next = 4096;
  const take = (bytes) => { const here = next; next += Math.ceil(bytes / 256) * 256; return here; };
  const count = (of) => (of ? kinds.filter((kind) => kind === of).length : kinds.length);
  const places = kinds.map((kind, l) => kinds.slice(0, l).filter((other) => other === kind).length);
  const plan = {
    dim: DIM, hidden: HIDDEN, layers: kinds.length, heads: HEADS, kvHeads: KV_HEADS, headSize: HEAD, turned: HEAD, unturned: [], seqLen: CONTEXT, kvStart: 8,
    eps: 1e-5, layerNorm: false, parallel: false, activation: "silu", windows: kinds.map(() => 0), local: [], localCos: 0, localSin: 0, kinds, places,
    convolution: null, linear: null, batch: BATCH,
    matrices: Object.fromEntries(Object.entries(matrices).map(([name, [rows, n, of]]) =>
      [name, { rows, n, six: false, ternary: false, layers: Array.from({ length: count(of) }, () => [take(rows * n), take((rows * n / GROUP) * 4)]), ...(of ? { of } : {}) }])),
    vectors: Object.fromEntries(Object.entries(vectors).map(([name, [size, of]]) => [name, { at: take(count(of) * size * 4), size, ...(of ? { of } : {}) }])),
    rows: take(BATCH * DIM * 4), cos: take(CONTEXT * (HEAD / 2) * 4), sin: take(CONTEXT * (HEAD / 2) * 4),
    tokens: { classifier: { rows: VOCAB, n: DIM, six: false, ternary: false, at: [take(VOCAB * DIM), take((VOCAB * DIM / GROUP) * 4)] }, embedding: null, final: take(DIM * 4),
      finalBias: 0, positions: 0, outliers: false, channels: [], ids: take(4096), most: MOST },
    staging: take(1 << 20), force: {}, remembered: undefined, direct: false, words: { done: 3, failed: 2, beat: 0, wanted: 1 }, ...more,
  };
  if (plan.localCos === true) Object.assign(plan, { localCos: take(CONTEXT * (HEAD / 2) * 4), localSin: take(CONTEXT * (HEAD / 2) * 4) });
  return { plan, bytes: next };
}
const ffn = { w1: [HIDDEN, DIM], w3: [HIDDEN, DIM], w2: [DIM, HIDDEN] };
const attending = (of) => ({ wq: [Q_DIM, DIM, of], wk: [KV_DIM, DIM, of], wv: [KV_DIM, DIM, of], wo: [DIM, Q_DIM, of] });
const norms = { attention: [DIM], ffn: [DIM] };
const MODELS = {
  llama: { kinds: ["attention", "attention"], matrices: { ...attending(), ...ffn }, vectors: norms },
  qwen3: { kinds: ["attention", "attention"], matrices: { ...attending(), ...ffn },
    vectors: { ...norms, bq: [Q_DIM], bk: [KV_DIM], bv: [KV_DIM], qNorm: [HEAD], kNorm: [HEAD] } },
  gemma3: { kinds: ["attention", "attention"], matrices: { ...attending(), ...ffn }, vectors: { ...norms, attentionPost: [DIM], ffnPost: [DIM] },
    more: { activation: "gelu", windows: [4, 0], local: [0], localCos: true } },
  lfm2: { kinds: ["convolution", "attention", "convolution"], matrices: { ...attending("attention"), win: [3 * DIM, DIM, "convolution"], wout: [DIM, DIM, "convolution"], ...ffn },
    vectors: { ...norms, conv: [CONV_TAPS * DIM, "convolution"], qNorm: [HEAD, "attention"], kNorm: [HEAD, "attention"] }, more: { convolution: { taps: CONV_TAPS } } },
  qwen35: { kinds: ["linear", "linear", "attention"],
    matrices: { wqkv: [MIXED, DIM, "linear"], wz: [READ, DIM, "linear"], wout: [DIM, READ, "linear"], ...attending("attention"), wg: [Q_DIM, DIM, "attention"], ...ffn },
    vectors: { ...norms, conv: [LINEAR.taps * MIXED, "linear"], wb: [LINEAR.valueHeads * DIM, "linear"], wa: [LINEAR.valueHeads * DIM, "linear"],
      dtBias: [LINEAR.valueHeads, "linear"], decay: [LINEAR.valueHeads, "linear"], deltaNorm: [LINEAR.valueDim, "linear"], qNorm: [HEAD, "attention"], kNorm: [HEAD, "attention"] },
    more: { linear: LINEAR } },
};

const hash = (text) => crypto.createHash("sha256").update(text).digest("hex").slice(0, 16);
const random = (bytes) => Uint8Array.from({ length: bytes }, (_, i) => (i * 131 + 7 * (i >> 8) + 89) & 255);
const found = {}, trace = [];
const record = (name, lines) => {
  found[name] = `${lines.length} ${hash(lines.join("\n"))}`;
  trace.push(`== ${name}`, ...lines);
};

for (const [model, made] of Object.entries(MODELS)) {
  for (const packed of [true, false]) {
    const name = `${model}${packed ? "" : " (floats)"}`;
    const { calls, device } = standIn({ language: ["packed_4x8_integer_dot_product", "subgroup_id"] });
    device.features.add("subgroups");
    const { plan, bytes } = planned(made);
    // (a memory of random bytes whose floats are small and finite: the scales and vectors are never read as NaN here, but
    // the record is of the bytes all the same)
    const memory = { buffer: new SharedArrayBuffer(bytes) };
    new Uint8Array(memory.buffer).set(random(bytes));
    const m = await device_.openDevice(plan);
    device_.handed(memory, plan);
    await upload(m);
    // (the forms forms.ts would choose on a device that computes: stand-ins of the tiles and the prompt's attention)
    const attention = { name: "a stand-in attention", pipeline: { code: "the prompt's attention", getBindGroupLayout: () => ({}) } };
    const form = { name: "a stand-in form", pipeline: { code: "the tiles", getBindGroupLayout: () => ({}) }, packed, tile: { rows: 4, tokens: 4 } };
    m.forms = { attention, form, tried: [] };
    // (a buffer by where it first comes in the record, a uniform by its bytes)
    const ids = new Map();
    const named = (buffer) => {
      if (buffer.usage & 0x40) return `u${Buffer.from(buffer.data).toString("hex")}`;
      if (!ids.has(buffer)) ids.set(buffer, `b${ids.size}:${buffer.size}:${buffer.usage}`);
      return ids.get(buffer);
    };
    const bound = (resource) => `${named(resource.buffer ?? resource)}${resource.offset !== undefined ? `+${resource.offset}` : ""}${resource.size !== undefined ? `/${resource.size}` : ""}`;
    const group = (g) => g.entries.map(({ binding, resource }) => `${binding}=${bound(resource)}`).join(" ");
    const said = (call) => {
      const [what, ...rest] = call;
      if (what === "dispatch") return `dispatch ${hash(rest[0].code)} [${group(rest[1])}] ${rest.slice(2).join(",")}`;
      if (what === "write") return `write ${named(rest[0])} ${rest[1]} ${rest[2]}`;
      if (what === "clear") return `clear ${named(rest[0])} ${rest[1]} ${rest[2]}`;
      if (what === "copy") return `copy ${named(rest[0])} ${rest[1]} ${named(rest[2])} ${rest[3]} ${rest[4]}`;
      return what;
    };
    const since = (from) => calls.slice(from).filter(([what]) => ["dispatch", "write", "clear", "copy", "submit"].includes(what)).map(said);
    const passes = (list) => list.map(([pipeline, g, x, y]) => `dispatch ${hash(pipeline.code)} [${group(g)}] ${x},${y}`);
    await block_.prepare(m);
    block_.bindLayers(m);
    block_.grow(m, 16);
    let from = calls.length;
    await block_.block(m, 5, 0, () => true);
    record(`${name}: a block of 5 at 0`, since(from));
    from = calls.length;
    await block_.block(m, 3, 5, () => true);
    record(`${name}: a block of 3 at 5, from the CPU's state`, since(from));
    from = calls.length;
    await block_.block(m, 2, 8, () => true, false, undefined, true);
    record(`${name}: a block of 2 at 8, from the GPU's own state`, since(from));
    if (packed) {
      m.tokens = { ...(await tokens_.tokenBuffers(m)), attention: { name: "the prompt's attention tiles", tiles: true, pipeline: attention.pipeline }, attentions: [],
        parts: undefined, params: undefined, form: null, ms: undefined, forms: null };
      for (const candidate of tokens_.tokenCandidates(m)) {
        const token = { ...candidate, pipes: {} };
        for (const [key, code] of tokens_.tokenCodes(token, tokens_.tokenShape(m))) token.pipes[key] = await device_.pipelineOf(m, code);
        record(`${name}: a token's step by ${candidate.name}`, passes(tokens_.tokenPass(m, token, { positions: 16 })));
        record(`${name}: the first layer's check by ${candidate.name}`, passes(tokens_.tokenPass(m, token, { to: 1, head: false, positions: 2 })));
        record(`${name}: the head's check by ${candidate.name}`, passes(tokens_.tokenPass(m, token, { to: 0, embed: false })));
      }
      const [first] = tokens_.tokenCandidates(m), token = { ...first, pipes: {} };
      for (const [key, code] of tokens_.tokenCodes(token, tokens_.tokenShape(m))) token.pipes[key] = await device_.pipelineOf(m, code);
      from = calls.length;
      await tokens_.runTokens(m, tokens_.tokenStep(m, token, 12), { count: 2, pos: 10, state: new Uint32Array(64), keep: true });
      record(`${name}: a run of 2 steps at 10, kept`, since(from));
    }
    const counted = (list) => [...list.reduce((map, key) => map.set(key, (map.get(key) ?? 0) + 1), new Map())].map(([key, n]) => `${key} ×${n}`).sort();
    record(`${name}: the buffers made`, counted(calls.filter(([what]) => what === "buffer").map(([, b]) => `${b.size}:${b.usage}`)));
    record(`${name}: the shaders compiled`, counted(calls.filter(([what]) => what === "pipeline").map(([, p]) => hash(p.code))));
  }
}
if (process.env.GPUSTEPS_TRACE) fs.writeFileSync(process.env.GPUSTEPS_TRACE, `${trace.join("\n")}\n`);
console.log(JSON.stringify(found));
