// unchanged-gpusteps.mjs (T.568.1)
// What the model's GPU worker (src/runtime/gpu/) asks of a device for a few made-up models, as one JSON object of
// hashes: every buffer it makes (their sizes and usages, counted), the shaders it compiles (counted), a prompt's blocks
// (the writes with their bytes, the clears, the copies and the dispatches of three blocks, the second going on from the state the CPU
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
// layers and an attending one with the gate of its output); a GPT-2 and a GPT-NeoX (LayerNorms, biases, no gate); a Llama in a rotated basis (T.271.1); the Qwen3.5 with its states in bundles (T.565, packed alone). Each
// with a packed tiled form and a float one; a Llama and a Qwen3.5 on the GPU alone too (T.156, T.271.4). A step's keys and
// values going up from the CPU's cache (keysIn) besides, float16 and float32, a ring where a layer has a window.
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
const { upload, uploadLayers, uploadRest, tablesOn, routed } = await import(at("gpu/weights.js"));
const block_ = await import(at("gpu/block.js"));
const tokens_ = await import(at("gpu/tokens.js"));

import { DIM, HIDDEN, HEAD, VOCAB, KV_DIM, LINEAR, MODELS, ALONE, planned } from "./fakes/gpu-plans.ts";

const hash = (text) => crypto.createHash("sha256").update(text).digest("hex").slice(0, 16);
const random = (bytes) => Uint8Array.from({ length: bytes }, (_, i) => (i * 131 + 7 * (i >> 8) + 89) & 255);
const found = {}, trace = [];
const record = (name, lines) => {
  found[name] = `${lines.length} ${hash(lines.join("\n"))}`;
  trace.push(`== ${name}`, ...lines);
};

// (a rotated basis takes the packed forms alone: gpu/forms.ts's candidates)
const runs = Object.entries(MODELS).flatMap(([model, made]) => [[model, made, true, false], ...(made.more?.rotated || made.more?.force ? [] : [[model, made, false, false]]),
  ...(ALONE.includes(model) ? [[model, made, true, true]] : [])]);
for (const [model, made, packed, direct] of runs) {
  {
    const name = `${model}${packed ? "" : " (floats)"}${direct ? " (on the GPU alone)" : ""}`;
    const { calls, device } = standIn({ language: ["packed_4x8_integer_dot_product", "subgroup_id"] });
    device.features.add("subgroups");
    const { plan, bytes, cache } = planned(made, direct);
    // (the memory's bytes, which nothing here computes with: the record is of what goes where)
    const memory = { buffer: new SharedArrayBuffer(bytes) };
    new Uint8Array(memory.buffer).set(random(bytes));
    let m;
    if (direct) {
      // (start.ts's open() and start(): the layers' buffers and the tables' made first, the checkpoint routed into them,
      // then the rest from the shared memory)
      m = await device_.openDevice({ layers: plan.layers, matrices: plan.matrices, tables: plan.tables, force: {}, remembered: undefined, kinds: plan.kinds,
        places: plan.places, tokens: true });
      // (T.626: a tree after it opens by the record's memory, null here, and takes the tables from open()'s plan; one
      // before it, by its second argument and from the record's plan, which is open()'s here: either reads its own)
      await uploadLayers(m, true);
      tablesOn(m, m.plan.tables);
      routed(m, 0, new Uint8Array(memory.buffer));
      device_.handed(memory, plan);
      await uploadRest(m);
    } else {
      m = await device_.openDevice(plan);
      device_.handed(memory, plan);
      await upload(m);
    }
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
      if (what === "write") return `write ${named(rest[0])} ${rest[1]} ${rest[2]} ${hash(Buffer.from(rest[3]).toString("hex"))}`;
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
    const tokens = (count) => Array.from({ length: count }, (_, t) => (t * 37 + 5) % VOCAB);
    await block_.block(m, 5, 0, () => true, false, tokens(5));
    record(`${name}: a block of 5 at 0`, since(from));
    from = calls.length;
    await block_.block(m, 3, 5, () => true, false, tokens(3));
    record(`${name}: a block of 3 at 5, from the CPU's state`, since(from));
    from = calls.length;
    await block_.block(m, 2, 8, () => true, false, tokens(2), true);
    record(`${name}: a block of 2 at 8, from the GPU's own state`, since(from));
    // a step's keys and values up from the CPU's cache (requests.ts's generate()): float16 rows as they are, and float32 ones
    // narrowed; a ring (T.510.1) for the windowed layers, the others' whole
    if (!direct) {
      for (const half of [true, false]) {
        const places = cache.map(([keys, values], p) => ({ keys, values, ring: plan.windows[plan.kinds.flatMap((kind, l) => (kind === "attention" ? [l] : []))[p]] ? 6 : 0 }));
        from = calls.length;
        block_.keysIn(m, { places, row: KV_DIM * (half ? 2 : 4), half }, 3, 13);
        record(`${name}: the keys and values of 3 to 12 up, ${half ? "float16" : "float32"}`, since(from));
      }
    }
    if (packed) {
      // (T.466: a token's attention and form are records of their own, m.tokenAttention and m.tokenForm; a tree before it
      // reads them of m.tokens: both are set, and each tree reads its own)
      const chosen = { attention: { name: "the prompt's attention tiles", tiles: true, pipeline: attention.pipeline }, attentions: [], parts: undefined, params: undefined };
      m.tokens = { ...(await tokens_.tokenBuffers(m)), ...chosen, form: null, ms: undefined, forms: null };
      m.tokenAttention = chosen;
      m.tokenForm = { form: null, ms: undefined, forms: null };
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
      await tokens_.runTokens(m, tokens_.tokenStep(m, token, 12), { count: 2, pos: 10, state: new Uint32Array(64), keep: !direct });
      record(`${name}: a run of 2 steps at 10${direct ? "" : ", kept"}`, since(from));
    }
    const counted = (list) => [...list.reduce((map, key) => map.set(key, (map.get(key) ?? 0) + 1), new Map())].map(([key, n]) => `${key} ×${n}`).sort();
    record(`${name}: the buffers made`, counted(calls.filter(([what]) => what === "buffer").map(([, b]) => `${b.size}:${b.usage}`)));
    record(`${name}: the shaders compiled`, counted(calls.filter(([what]) => what === "pipeline").map(([, p]) => hash(p.code))));
  }
}
if (process.env.GPUSTEPS_TRACE) fs.writeFileSync(process.env.GPUSTEPS_TRACE, `${trace.join("\n")}\n`);
console.log(JSON.stringify(found));
