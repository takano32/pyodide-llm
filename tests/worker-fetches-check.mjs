// tests/worker-fetches-check.mjs (T357)
// What a conversion asks of the network, in order, for every kind of source the page converts: the requests (the
// file and its range) public/worker/convert.js makes for a made-up repository of huggingface.co, and what it hands the
// converter on the way, held to tests/fixtures/conversion-fetches.json. The conduct of a conversion is the worker's
// today (which file, how much of its head, which candidate next, the parts of the weights); if it moves (to Python, as
// the design of T376 proposes), this is what shows that the same things are asked for in the same order, or says what
// changed.
//
//   node tests/worker-fetches-check.mjs            compares; exit 1 where a case differs (the first lines that do are said)
//   node tests/worker-fetches-check.mjs --write    writes the fixture (read the difference before it is committed)
//
// The worker runs in tests/worker-harness.mjs's context, on a clock that moves only where a made-up line is slow: the
// same requests every run. The converter is a stand-in (no Pyodide): it reads the made-up files as far as the worker
// depends on it (a GGUF's head that is not all there yet, the bytes of each shard, a tokenizer it refuses), opens the
// place of the weights as Writer does and counts what it is fed. The kept models are a stand-in that keeps nothing.
// What this does not see: the bytes themselves (the unit tests and the net's "python" hold what a conversion writes),
// the Service Worker, a real line, and the files of this site (worker-check.mjs's download()).
import assert from "node:assert/strict";
import fs from "node:fs";
import { workerHarness } from "./worker-harness.mjs";

const FIXTURE = new URL("fixtures/conversion-fetches.json", import.meta.url);
const { MiB, sleep, requests, messages, navigatorStandIn, context, run, failure, fresh } = workerHarness({ told: true });
setTimeout(() => {
  console.error("worker-fetches-check: still waiting after 120 s");
  process.exit(1);
}, 120000).unref();

// ---- the made-up files. A safetensors file: 8 bytes that say how long its header is, the header (a JSON text that
// says how many bytes of tensors follow), the tensors. A GGUF: "GGUF", 8 bytes that say where its tensors begin.
const text = (value) => new TextEncoder().encode(value);
function safetensors(data, { header = 300 } = {}) {
  const json = text(JSON.stringify({ __metadata__: { data } }).padEnd(header, " "));
  const head = new Uint8Array(8 + json.length);
  new DataView(head.buffer).setBigUint64(0, BigInt(json.length), true);
  head.set(json, 8);
  return { head, size: head.length + data };
}
function gguf(base, data) {
  const head = new Uint8Array(12);
  head.set(text("GGUF"));
  new DataView(head.buffer).setBigUint64(4, BigInt(base), true);
  return { head, size: base + data };
}
// a body of a file's bytes [from, to): its head, then zeros, a MiB at a time, each after `delay` ms of the worker's clock;
// held: its first MiB comes only after 300 turns of the event loop (everything else that can come has come by then)
function stream(file, from, to, delay, held) {
  let at = from;
  return new ReadableStream({
    async pull(controller) {
      if (held && at === from) for (let turn = 0; turn < 300; turn++) await new Promise(setImmediate);
      if (delay) await sleep(delay);
      if (at >= to) return controller.close();
      const end = Math.min(to, at + MiB), bytes = new Uint8Array(end - at);
      if (at < file.head.length) bytes.set(file.head.subarray(at, Math.min(end, file.head.length)));
      controller.enqueue(bytes);
      at = end;
    },
  }, { highWaterMark: 0 });
}
const missing = () => new Response("", { status: 404, headers: { "X-Error-Code": "EntryNotFound" } });
// the hub: { "<repository>/<file>": a text, or a file ({ head, size }) }; line: { delay (ms a MiB of each connection),
// unsaid (no Content-Range), held (the part of the weights that begins at this byte comes after all the others that can) }
const REVISION = "0123456789abcdef0123456789abcdef01234567";
const hub = (files, line = {}) => (url, init) => {
  const [, repository, file] = /^https:\/\/huggingface\.co\/(.+?)\/resolve\/[0-9a-f]{40}\/(.+)$/.exec(url) ?? [];
  const found = files[`${repository}/${file}`];
  if (found === undefined) return missing();
  if (typeof found === "string") return new Response(found, { status: 200 });
  if (init.method === "HEAD") return new Response(null, { status: 200, headers: { "Content-Length": String(found.size) } });
  const range = /bytes=(\d+)-(\d+)/.exec(init.headers?.Range ?? "");
  if (!range) return new Response(stream(found, 0, found.size, line.delay), { status: 200, headers: { "Content-Length": String(found.size) } });
  const from = Number(range[1]), end = Math.min(found.size, Number(range[2]) + 1);
  return new Response(stream(found, from, end, line.delay, line.held === from), { status: 206, headers: line.unsaid ? {} : { "Content-Range": `bytes ${from}-${end - 1}/${found.size}` } });
};

// ---- the converter's stand-in (llama2_convert, as convert.js calls it through Pyodide), which writes down what it is handed
const HEADER = [64, 128, 2, 4, 4, 256, 128], FORM = { bias: false, arch: "llama", qk_norm: false, head_dim: 0, linear: null, rotated: null, convolution: null };
const proxy = (value) => ({ toJs: () => value, destroy() {} });
let handed = [];
const incomplete = () => Object.assign(new Error("the head of the GGUF is not all there"), { type: "Incomplete" });
// where a GGUF's tensors begin, once its head is all there
const ggufBase = (first) => {
  const base = Number(new DataView(first.buffer, first.byteOffset).getBigUint64(4, true));
  if (first.length < base) throw incomplete();
  return base;
};
function conversion(what, expected, { sink, dtype }) {
  let fed = 0, feeds = 0;
  sink.open(1000, proxy(HEADER), "int8", proxy(FORM));
  handed.push(`${what}, dtype ${typeof dtype === "function" ? "the worker's choice" : dtype}`);
  return {
    feed(bytes) {
      fed += bytes.length;
      feeds++;
      return expected ? fed / expected : 1;
    },
    finish() {
      handed.push(`fed ${fed} bytes in ${feeds} pieces${expected === undefined || fed === expected ? "" : `, NOT the ${expected} of its tensors`}; finish()`);
    },
    options: proxy({ dtype: "int8", bos: 1 }),
    tokenizer: { getBuffer: () => ({ data: new Uint8Array(4), release() {} }), destroy() {} },
    destroy() {},
  };
}
const dataOf = (header) => JSON.parse(header).__metadata__.data;
const converter = {
  Conversion: Object.assign({
    callKwargs(header, base, config, tokenizer, name, kwargs) {
      const start = `Conversion(a header of ${header.length} characters, base ${base}, start ${kwargs.start}, the config ${JSON.stringify(config)}, ${name} of ${tokenizer.length} bytes, ` +
        `tokenizer_config ${JSON.stringify(kwargs.tokenizer_config)}, chat_template ${JSON.stringify(kwargs.chat_template)})`;
      if (new TextDecoder().decode(tokenizer).startsWith("unreadable")) {
        handed.push(`${start}: refused`);
        throw new Error(`This model cannot be converted: ${name} is of a kind the engine does not read.`);
      }
      const joined = header.startsWith("[");
      return conversion(start, joined ? JSON.parse(header).reduce((sum, data) => sum + data, 0) : header.startsWith("gguf") ? undefined : dataOf(header), kwargs);
    },
  }, {
    from_gguf: {
      callKwargs(first, kwargs) {
        let base;
        try {
          base = ggufBase(first);
        } catch (error) {
          handed.push(`Conversion.from_gguf(the first ${first.length} bytes): not all of the head yet`);
          throw error;
        }
        return Object.assign(conversion(`Conversion.from_gguf(the first ${first.length} bytes)`, undefined, kwargs), { base });
      },
    },
  }),
  gguf_weights(first, config) {
    try {
      const base = ggufBase(first);
      handed.push(`gguf_weights(the first ${first.length} bytes, the config ${JSON.stringify(config)})`);
      return proxy(["gguf header", base]);
    } catch (error) {
      handed.push(`gguf_weights(the first ${first.length} bytes): not all of the head yet`);
      throw error;
    }
  },
  joined_shards(headers) {
    handed.push(`joined_shards(${headers.length} headers)`);
    const lengths = headers.map(dataOf);
    return proxy([JSON.stringify(lengths), lengths]);
  },
};
let kept = [];
context.stand = {
  converter,
  kept: {
    openKept: async () => null, replaced: async () => [], forget: async () => {}, keeper: async () => undefined,
    keep: async (model, manifest) => { kept.push(`kept as ${model.conversion.dtype}: ${manifest.repo}@${manifest.revision.slice(0, 7)}, ${manifest.bytes} bytes, options ${JSON.stringify(manifest.options)}`); },
  },
  // (no kernels: the checkpoint's place is a Python bytearray, as with ?without=kernels)
  pyodide: { globals: { get: () => () => ({ destroy() {}, getBuffer: () => ({ data: new Uint8Array(1000), release() {} }) }) } },
  numpy: { Llama: { callKwargs: () => ({}) }, OUTLIER_CHANNELS: 8, KV_START: 256 },
};
run("state.llama2_convert = stand.converter; state.keptModule = stand.kept; state.pyodide = stand.pyodide; state.llama2_numpy = stand.numpy; " +
  "state.jsKernels = undefined; state.kernels = undefined; state.disabled = [];");

// ---- the cases: [name, the model's hf, the hub's files, the line]
const one = safetensors(100 * MiB), small = safetensors(3 * MiB + 100);
const CONFIG = "{\"model_type\":\"llama\"}", TEMPLATED = "{\"chat_template\":\"{{ messages }}\"}", PLAIN = "{\"bos_token\":\"<s>\"}";
const repo = (name, files) => Object.fromEntries(Object.entries(files).map(([file, value]) => [`${name}/${file}`, value]));
const whole = (weights, more = {}) => ({ "config.json": CONFIG, "tokenizer_config.json": TEMPLATED, "tokenizer.json": "a tokenizer", "model.safetensors": weights, ...more });
const hf = (more = {}) => ({ repo: "owner/model", revision: REVISION, weights: "model.safetensors", tokenizer: "tokenizer.json", ...more });
const left = (files, ...names) => Object.fromEntries(Object.entries(files).filter(([file]) => !names.includes(file)));
const shards = { "model-00001-of-00002.safetensors": safetensors(9 * MiB), "model-00002-of-00002.safetensors": safetensors(2 * MiB + 7, { header: 500 }) };
const index = (names) => JSON.stringify({ weight_map: Object.fromEntries(names.map((name, i) => [`tensor.${i}`, name])) });
const ANY = ["tokenizer.json", "tokenizer.model", "spiece.model"];  // (what ?hf= asks for: src/page/address.ts)
const CASES = [
  // (six connections take a MiB each in turn: the first part's 8 MiB take 48 delays of the clock. 20 ms: 8.7 MB/s, past
  // the 4 MB/s that make the later parts 16 MiB; 1000 ms: 0.17 MB/s)
  ["one safetensors file, a fast line (16 MiB parts after the first)", hf(), repo("owner/model", whole(one)), { delay: 20 }],
  ["one safetensors file, a slow line (8 MiB parts)", hf(), repo("owner/model", whole(one)), { delay: 1000 }],
  // (no more than two parts a connection wait for an earlier one: twelve are asked for, and the rest once the first is in)
  ["one safetensors file, its first part the last to come (the others wait for it)", hf(), repo("owner/model", whole(safetensors(200 * MiB))), { delay: 10, held: 308 }],
  ["one safetensors file, a device of 4 GB (8 MiB parts)", hf(), repo("owner/model", whole(one)), { deviceMemory: 4 }],
  ["one safetensors file, a small one", hf(), repo("owner/model", whole(small))],
  ["one safetensors file, a header past the first 512 KiB", hf(), repo("owner/model", whole(safetensors(2 * MiB, { header: 700000 })))],
  ["one safetensors file, a server that does not show Content-Range", hf(), repo("owner/model", whole(small)), { unsaid: true }],
  ["shards with an index", hf(), repo("owner/model", { ...left(whole(small), "model.safetensors"), ...shards,
    "model.safetensors.index.json": index(["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors", "model-00001-of-00002.safetensors"]) })],
  ["one shard named by the index", hf(), repo("owner/model", { ...left(whole(small), "model.safetensors"), "model-00001-of-00001.safetensors": small,
    "model.safetensors.index.json": index(["model-00001-of-00001.safetensors"]) })],
  ["no weights and no index", hf(), repo("owner/model", left(whole(small), "model.safetensors"))],
  ["a GGUF alone (a head of 3 MiB)", hf({ weights: "model.Q8_0.gguf", tokenizer: undefined }), repo("owner/model", { "model.Q8_0.gguf": gguf(3 * MiB, 20 * MiB) })],
  ["a GGUF alone, a small head", hf({ weights: "model.Q8_0.gguf", tokenizer: undefined }), repo("owner/model", { "model.Q8_0.gguf": gguf(1000, 1 * MiB) })],
  ["a GGUF with the vocabulary of another repository", hf({ repo: "maker/model-GGUF", weights: "model.Q8_0.gguf", tokenizer: undefined,
    vocabulary: { repo: "owner/model", revision: REVISION.split("").reverse().join(""), tokenizer: "tokenizer.model" } }),
  { ...repo("maker/model-GGUF", { "model.Q8_0.gguf": gguf(9 * MiB, 20 * MiB), "config.json": "{\"the\":\"maker's, never asked for\"}" }),
    ...repo("owner/model", { "config.json": CONFIG, "tokenizer_config.json": PLAIN, "chat_template.jinja": "a template", "tokenizer.model": "a sentencepiece model" }) }],
  ["a sentencepiece model (no tokenizer.json)", hf({ tokenizer: ANY }), repo("owner/model", { ...left(whole(small), "tokenizer.json"), "tokenizer.model": "a sentencepiece model" })],
  ["a tokenizer.json the converter refuses, then spiece.model", hf({ tokenizer: ANY }), repo("owner/model", whole(small, { "tokenizer.json": "unreadable", "spiece.model": "a sentencepiece model" }))],
  ["no tokenizer at all", hf({ tokenizer: ANY }), repo("owner/model", left(whole(small), "tokenizer.json"))],
  ["no tokenizer_config.json (an optional file) and no chat_template.jinja", hf(), repo("owner/model", left(whole(small), "tokenizer_config.json"))],
  ["a chat_template.jinja beside a tokenizer_config.json without a template", hf(), repo("owner/model", whole(small, { "tokenizer_config.json": PLAIN, "chat_template.jinja": "a template" }))],
  ["a config.json under another name, int6 asked for", { ...hf({ config: "configs/text.json" }), dtype: "int6" }, repo("owner/model", { ...left(whole(small), "config.json"), "configs/text.json": CONFIG })],
];

const shortUrl = (url) => url.replace(/^https:\/\/huggingface\.co\/(.+?)\/resolve\/([0-9a-f]{7})[0-9a-f]{33}\//, "$1@$2 ");
const found = {};
for (const [name, { dtype, ...source }, files, line = {}] of CASES) {
  handed = [];
  kept = [];
  navigatorStandIn.deviceMemory = line.deviceMemory ?? 8;
  fresh(hub(files, line));
  const model = { id: "made-up", name: "Made up", hf: source, ...(dtype ? { conversion: { dtype } } : {}) };
  const failed = await failure(context.convert(model, new AbortController().signal, 1));
  await sleep(0);
  found[name] = {
    requests: requests.map((r) => `${r.method} ${shortUrl(r.url)}${r.range ? ` ${r.range}` : ""}`),
    converter: [...handed, ...kept],
    ended: failed ? `failed: ${failed.error.message}` : `converted; ${messages.filter((m) => m.type === "progress").length ? "progress was told" : "NO progress was told"}`,
  };
  run("state.llama = undefined");
}

const written = `${JSON.stringify(found, null, 1)}\n`;
if (process.argv.includes("--write")) {
  fs.writeFileSync(FIXTURE, written);
  console.log(`worker-fetches-check: wrote ${Object.keys(found).length} cases, ${Object.values(found).reduce((sum, c) => sum + c.requests.length, 0)} requests, to ${FIXTURE.pathname}`);
  process.exit(0);
}
const expected = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));
let differ = 0;
for (const name of new Set([...Object.keys(expected), ...Object.keys(found)])) {
  const was = expected[name], now = found[name];
  if (JSON.stringify(was) === JSON.stringify(now)) {
    console.log(`ok: ${name}: ${now.requests.length} requests`);
    continue;
  }
  differ++;
  console.log(`DIFFERENT: ${name}`);
  if (!was || !now) { console.log(`    ${was ? "the fixture has it and the check makes it no more" : "not in the fixture"}`); continue; }
  for (const part of ["requests", "converter", "ended"]) {
    const a = [].concat(was[part]), b = [].concat(now[part]);
    const at = a.findIndex((line, i) => line !== b[i]);
    const first = at === -1 && b.length > a.length ? a.length : at;
    if (first !== -1) console.log(`    ${part}, line ${first + 1} of ${a.length} (now ${b.length}):\n      was ${a[first] ?? "(nothing more)"}\n      now ${b[first] ?? "(nothing more)"}`);
  }
}
console.log(differ ? `worker-fetches-check: FAILED: ${differ} cases ask for something else than tests/fixtures/conversion-fetches.json says (--write after reading why)`
  : `worker-fetches-check: ${Object.keys(found).length} cases ask for what the fixture says`);
// (a ReadableStream left unread keeps nothing alive here; the harness's timers are let go of with the process)
// T384: not at once. An exit right after the last case stood still for good about once in 70 runs on the development
// machine (and twice in CI, to the job's limit): the main thread and one of V8's own threads both asleep on a futex,
// inside Node's leaving (24.20; nodejs/node 54918 tells of one like it). With this pause, 0 of 400.
setTimeout(() => process.stdout.write("", () => process.exit(differ ? 1 : 0)), 200);
