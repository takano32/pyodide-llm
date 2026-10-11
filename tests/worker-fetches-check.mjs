// tests/worker-fetches-check.mjs (T357)
// What a conversion asks of the network, in order, for every kind of source the page converts: the requests (the
// file and its range) made for a made-up repository of huggingface.co, and what the converter is handed on the way,
// held to tests/fixtures/conversion-fetches.json. The conduct of a conversion (which file, how much of its head, which
// candidate next) is Python's since T374.2.1 (src/python/convert/conduct.py), the answers (the fetches, the parts of the
// weights) the worker's (public/worker/conduct.js): this ran the worker's own steps before, and the fixture is what
// showed that the two together ask for the same things in the same order.
//
//   node tests/worker-fetches-check.mjs            compares; exit 1 where a case differs (the first lines that do are said)
//   node tests/worker-fetches-check.mjs --write    writes the fixture (read the difference before it is committed)
//
// The worker runs in tests/worker-harness.mjs's context, on a clock that moves only where a made-up line is slow: the
// same requests every run. The conduct is the real one, in Pyodide (Node's, with NumPy: about four seconds). The
// converter is a stand-in (tests/conduct_hub.py's StandIn): it reads the made-up files as far as the conduct depends on
// it (a GGUF's head that is not all there yet, the bytes of each shard, a tokenizer it refuses), opens the place of the
// weights as Writer does and counts what it is fed. The kept models are a stand-in that keeps nothing.
// What this does not see: the bytes themselves (the unit tests and the net's "python" hold what a conversion writes),
// the Service Worker, a real line, and the files of this site (worker-check.mjs's download()). The loop that answers
// the conduct is seen alone, without Pyodide, by tests/worker-conduct-check.mjs.
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadPyodide } from "pyodide";
import { placePython, runtimeUrl, treeOf } from "./tree.mjs";
import { workerHarness } from "./worker-harness.mjs";
import { openHuggingFace, TOKENIZERS } from "../src/page/folder.ts";
// (the page's module reads its address where it is loaded: there is none here)
globalThis.location ??= { search: "" };
const { hfEntry } = await import("../src/page/address.ts");

const FIXTURE = new URL("fixtures/conversion-fetches.json", import.meta.url);
// T374.1: the cases themselves (the model as it is listed, what the made-up hub has, the line), written down beside
// what they ask for: tests/test_conduct.py reads them, so that the Python that conducts a conversion is held to the
// requests of the very repositories this check makes. This file is where a case is written; that one is its record
const CASES_FIXTURE = new URL("fixtures/conversion-cases.json", import.meta.url);
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
  return { head, size: head.length + data, made: { safetensors: { data, header } } };
}
function gguf(base, data) {
  const head = new Uint8Array(12);
  head.set(text("GGUF"));
  new DataView(head.buffer).setBigUint64(4, BigInt(base), true);
  return { head, size: base + data, made: { gguf: { base, data } } };
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

// ---- a folder of the visitor's disk: Files that write down what is read of them, as a request is written down
// ("text config.json", "range model.safetensors bytes=0-524287", "stream model.safetensors bytes=308-3145835"). The
// disk gives a stream a MiB at a time
const reads = [];
function diskFile(name, value) {
  const file = typeof value === "string" ? { head: text(value), size: text(value).length } : value;
  const bytes = (from, to) => {
    const made = new Uint8Array(to - from);
    if (from < file.head.length) made.set(file.head.subarray(from, Math.min(to, file.head.length)), 0);
    return made;
  };
  const between = (from, to) => (from < to ? ` bytes=${from}-${to - 1}` : ` bytes=${from}-`);
  return {
    name, size: file.size,
    text: async () => { reads.push(`text ${name}`); return new TextDecoder().decode(bytes(0, file.size)); },
    arrayBuffer: async () => { reads.push(`bytes ${name}`); return bytes(0, file.size).buffer; },
    slice(begin = 0, end = file.size) {
      const from = Math.min(begin, file.size), to = Math.max(from, Math.min(end, file.size));
      return {
        size: to - from,
        arrayBuffer: async () => { reads.push(`range ${name}${between(from, to)}`); return bytes(from, to).buffer; },
        stream() { reads.push(`stream ${name}${between(from, to)}`); return stream(file, from, to); },
      };
    },
  };
}
// the model the page makes of the files chosen: by the page's own function (src/page/folder.ts), so that what the
// page hands the worker of a folder and what the worker reads of it are held together
const chosen = async (files) => (await openHuggingFace(Object.entries(files).map(([name, value]) => diskFile(name, value)))).hf;

// ---- the conduct is the real one (src/python/convert/conduct.py, in Pyodide, as the worker has it) and the worker's loop
// answers it (public/worker/conduct.js): the requests below are what the two make together. The converter is a
// stand-in in the conduct's place of it (tests/conduct_hub.py's StandIn, T374.1), which writes down what it is handed;
// here it also opens the place of the weights as Writer does, and has the options and the tokenizer a conversion ends with
const pyodide = await loadPyodide();
await pyodide.loadPackage("numpy", { messageCallback: () => {} });
placePython(pyodide, treeOf());
pyodide.FS.writeFile("conduct_hub.py", fs.readFileSync(new URL("conduct_hub.py", import.meta.url)));
pyodide.runPython(`
import convert.conduct
from conduct_hub import StandIn

HEADER, FORM = [64, 128, 2, 4, 4, 256, 128], {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0, "linear": None, "rotated": None, "convolution": None}
stand = StandIn()

class Opened(stand.Conversion):
    def __init__(self, *files, sink, **more):
        # (a refused tokenizer opens nothing, as the converter's own)
        super().__init__(*files, **more)
        sink.open(1000, HEADER, "int8", FORM)
        self.options, self.tokenizer = {"dtype": "int8", "bos": 1}, bytes(4)

for name, value in dict(Conversion=Opened, gguf_weights=stand.gguf_weights, joined_shards=stand.joined_shards).items():
    setattr(convert.conduct, name, value)
`);
// what the stand-in was handed since the last time this was asked
const handedSoFar = () => {
  const taken = pyodide.runPython("taken = list(stand.handed); stand.handed.clear(); taken"), handed = taken.toJs();
  taken.destroy();
  return handed;
};
let kept = [];
context.stand = {
  // (nothing of the converter's window is called from JavaScript but the kernels' readers, and there are no kernels here)
  converter: {},
  kept: {
    openKept: async () => null, replaced: async () => [], forget: async () => {}, keeper: async () => undefined,
    keep: async (model, manifest) => { kept.push(`kept as ${model.conversion.dtype}: ${manifest.repo}@${manifest.revision.slice(0, 7)}, ${manifest.bytes} bytes, options ${JSON.stringify(manifest.options)}`); },
  },
  // (no kernels: the checkpoint's place is a Python bytearray, as with ?without=kernels)
  pyodide,
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
// a repository nobody has looked at (?hf=, the sheet): the model as the page's own function lists it (src/page/address.ts),
// so that what the page hands the worker of such a repository and what is then asked of it are held together
const unlisted = () => hfEntry("owner/model", REVISION).hf;
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
  // (T374.3: the one above comes whole with the third piece of its head, which ends past the file; this one goes on after it)
  ["a GGUF whose head takes three pieces, and a stream after them", hf({ weights: "model.Q8_0.gguf", tokenizer: undefined }), repo("owner/model", { "model.Q8_0.gguf": gguf(9 * MiB, 40 * MiB) })],
  ["a sentencepiece model (no tokenizer.json)", unlisted(), repo("owner/model", { ...left(whole(small), "tokenizer.json"), "tokenizer.model": "a sentencepiece model" })],
  ["a tokenizer.json the converter refuses, then spiece.model", unlisted(), repo("owner/model", whole(small, { "tokenizer.json": "unreadable", "spiece.model": "a sentencepiece model" }))],
  ["no tokenizer at all", unlisted(), repo("owner/model", left(whole(small), "tokenizer.json"))],
  ["no tokenizer_config.json (an optional file) and no chat_template.jinja", hf(), repo("owner/model", left(whole(small), "tokenizer_config.json"))],
  ["a chat_template.jinja beside a tokenizer_config.json without a template", hf(), repo("owner/model", whole(small, { "tokenizer_config.json": PLAIN, "chat_template.jinja": "a template" }))],
  ["a config.json under another name, int6 asked for", { ...hf({ config: "configs/text.json" }), dtype: "int6" }, repo("owner/model", { ...left(whole(small), "config.json"), "configs/text.json": CONFIG })],
  // ---- a folder of the visitor's disk (no repository: the files are the folder's, by their names)
  ["a folder: one safetensors file, as the disk gives it", { weights: "model.safetensors" }, whole(safetensors(20 * MiB + 5))],
  ["a folder: a header past the first 512 KiB", { weights: "model.safetensors" }, whole(safetensors(2 * MiB, { header: 700000 }))],
  ["a folder: a tokenizer.json the converter refuses, then spiece.model", { weights: "model.safetensors" }, whole(small, { "tokenizer.json": "unreadable", "spiece.model": "a sentencepiece model" })],
  ["a folder: every tokenizer of it refused", { weights: "model.safetensors" }, whole(small, { "tokenizer.json": "unreadable", "tokenizer.model": "unreadable too" })],
  ["a folder: no tokenizer_config.json, a chat_template.jinja", { weights: "model.safetensors" }, { ...left(whole(small), "tokenizer_config.json"), "chat_template.jinja": "a template" }],
  ["a folder: a chat_template.jinja beside a tokenizer_config.json without a template", { weights: "model.safetensors" }, whole(small, { "tokenizer_config.json": PLAIN, "chat_template.jinja": "a template" })],
  ["a folder: neither a tokenizer_config.json nor a chat_template.jinja, a sentencepiece model", { weights: "weights.safetensors" },
    { "config.json": CONFIG, "tokenizer.model": "a sentencepiece model", "weights.safetensors": small, "README.md": "never read" }],
  ["a folder: names in capital letters", { weights: "Model.SafeTensors" }, { "Config.JSON": CONFIG, "Tokenizer_Config.json": TEMPLATED, "Tokenizer.JSON": "a tokenizer", "Model.SafeTensors": small }],
  ["a folder: a file that is no safetensors file", { weights: "model.safetensors" }, whole("not a model")],
  ["a folder: a file that is no safetensors file, and an index beside it that names shards", { weights: "model.safetensors" },
    whole("not a model", { "model.safetensors.index.json": index(["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]) })],
];

// ---- T374.2.3: the candidates for a tokenizer are in one place, Python's (TOKENIZERS of src/python/convert/conduct.py). The
// page names none for a repository nobody has looked at, and keeps one list of its own, for the folder it must look
// into before there is a Pyodide to ask: that list is held to Python's here
{
  const proxy = pyodide.runPython("list(convert.conduct.TOKENIZERS)"), candidates = proxy.toJs();
  proxy.destroy();
  assert.deepEqual(TOKENIZERS, candidates, `the tokenizers the page looks for in a folder (TOKENIZERS of src/page/folder.ts: ${TOKENIZERS.join(", ")}) are not ` +
    `the candidates of a conversion (TOKENIZERS of ${treeOf().folders.python}/convert/conduct.py: ${candidates.join(", ")}). The list is Python's: edit src/page/folder.ts to say ` +
    "the same names in the same order (its sentence for a folder without a tokenizer is made of them, and a visitor reads it)");
  const entry = hfEntry("owner/model", REVISION, { template: "{prompt}" });
  assert.deepEqual(entry.hf, { repo: "owner/model", revision: REVISION, weights: "model.safetensors", config: "config.json" },
    "the page names more of a repository nobody has looked at than its weights and its config.json: the candidates for its tokenizer are the conduct's to name");
  console.log(`ok: the candidates for a tokenizer are Python's (${candidates.join(", ")}): the page names none for ?hf=, and looks for the same in a folder`);
}

// ---- the page's side of a folder (src/page/folder.ts): what it hands the worker, and what it asks for before it does
{
  const picked = (...names) => names.map((name) => diskFile(name, name.endsWith(".safetensors") ? small : "{}"));
  const model = await openHuggingFace(picked("README.md", "Model.safetensors", "config.json", "spiece.model", "tokenizer_config.json"));
  assert.equal(model.hf.weights, "Model.safetensors", "the page names the weights of a folder by the file's name");
  assert.deepEqual(model.hf.files.map(({ name }) => name), ["README.md", "Model.safetensors", "config.json", "spiece.model", "tokenizer_config.json"], "the page hands the worker the folder as it was chosen");
  assert.deepEqual(Object.keys(model.hf), ["files", "weights"], "the page names more of a folder than its weights: the conduct asks for the rest by the names it knows");
  assert.deepEqual(reads, [], "the page read a file of a folder that has no settings");
  for (const names of [["config.json", "tokenizer.json"], ["a.safetensors", "b.safetensors", "config.json", "tokenizer.json"], ["model.safetensors", "tokenizer.json"], ["model.safetensors", "config.json"]]) {
    await assert.rejects(openHuggingFace(picked(...names)), /^Error: A Hugging Face model needs three files together: /, `the page took a folder of ${names.join(", ")}`);
  }
  // the sentence a visitor reads, whole: it names the candidates, and a change of it is the owner's to see first
  await assert.rejects(openHuggingFace(picked("model.safetensors", "config.json")), { message: "A Hugging Face model needs three files together: one .safetensors file " +
    "(a model in several shards is not supported), config.json, and tokenizer.json, tokenizer.model or spiece.model." }, "the page's sentence for a folder without a tokenizer changed");
  // any one of the candidates will do, and a file of another name will not
  for (const name of TOKENIZERS) assert.equal((await openHuggingFace(picked("model.safetensors", "config.json", name))).hf.weights, "model.safetensors", `the page refused a folder whose tokenizer is ${name}`);
  await assert.rejects(openHuggingFace(picked("model.safetensors", "config.json", "vocab.txt")), /needs three files together/, "the page took a folder whose only tokenizer is vocab.txt");
  for (const [name, , files] of CASES.filter(([, source]) => !source.repo)) {
    assert.equal((await chosen(files)).weights.toLowerCase(), CASES.find(([title]) => title === name)[1].weights.toLowerCase(), `${name}: the page takes another file for the weights than the case says`);
  }
  console.log("ok: the page hands the worker a folder as it was chosen and the name of its weights, and asks for one .safetensors file, config.json and a tokenizer first");
}

const shortUrl = (url) => url.replace(/^https:\/\/huggingface\.co\/(.+?)\/resolve\/([0-9a-f]{7})[0-9a-f]{33}\//, "$1@$2 ");
const found = {};
for (const [name, { dtype, ...source }, files, line = {}] of CASES) {
  handedSoFar();
  kept = [];
  navigatorStandIn.deviceMemory = line.deviceMemory ?? 8;
  fresh(hub(files, line));
  const folder = !source.repo;
  reads.length = 0;
  const model = { id: "made-up", name: "Made up", hf: folder ? await chosen(files) : source, ...(dtype ? { conversion: { dtype } } : {}) };
  const failed = await failure(context.convert(model, new AbortController().signal, 1));
  await sleep(0);
  // (a failure of Python's is told by its last line, as the worker tells a ValueError)
  const words = (error) => (error.type === "ValueError" ? error.message.trim().split("\n").pop().replace(/^ValueError: /, "") : error.message);
  const progress = messages.filter((m) => m.type === "progress");
  if (folder) assert.deepEqual(requests.map((r) => r.url), [], `${name}: a folder asked the network for something`);
  else assert.deepEqual(reads, [], `${name}: a file of the disk was read for a model of huggingface.co`);
  found[name] = {
    requests: folder ? [...reads] : requests.map((r) => `${r.method} ${shortUrl(r.url)}${r.range ? ` ${r.range}` : ""}`),
    converter: [...handedSoFar(), ...kept],
    ended: failed ? `failed: ${words(failed.error)}` : `converted; ${progress.length ? "progress was told" : "NO progress was told"}`,
    // (what a folder's progress says has arrived, and of how much: the page shows the share converted)
    ...(folder && { progress: [...new Set(progress.map((m) => `${m.received} of ${m.total}`))] }),
  };
  run("state.llama = undefined");
}

const written = `${JSON.stringify(found, null, 1)}\n`;
// (a made-up file as what it was made of: the hub's files are made again from that in Python)
const cases = `${JSON.stringify(CASES.map(([name, hf, files, line = {}]) => ({ name, hf, line,
  files: Object.fromEntries(Object.entries(files).map(([file, value]) => [file, typeof value === "string" ? value : value.made])) })), null, 1)}\n`;
if (process.argv.includes("--write")) {
  fs.writeFileSync(FIXTURE, written);
  fs.writeFileSync(CASES_FIXTURE, cases);
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
  for (const part of ["requests", "converter", "ended", "progress"]) {
    const a = [].concat(was[part]), b = [].concat(now[part]);
    const at = a.findIndex((line, i) => line !== b[i]);
    const first = at === -1 && b.length > a.length ? a.length : at;
    if (first !== -1) console.log(`    ${part}, line ${first + 1} of ${a.length} (now ${b.length}):\n      was ${a[first] ?? "(nothing more)"}\n      now ${b[first] ?? "(nothing more)"}`);
  }
}
if (cases !== fs.readFileSync(CASES_FIXTURE, "utf8")) {
  differ++;
  console.log(`DIFFERENT: the cases are not the ones ${CASES_FIXTURE.pathname} records (--write, and read what tests/test_conduct.py then says)`);
}
console.log(differ ? `worker-fetches-check: FAILED: ${differ} cases ask for something else than tests/fixtures/conversion-fetches.json says (--write after reading why)`
  : `worker-fetches-check: ${Object.keys(found).length} cases ask for what the fixture says`);
// (a ReadableStream left unread keeps nothing alive here; the harness's timers are let go of with the process)
// T384: not at once. An exit right after the last case stood still for good about once in 70 runs on the development
// machine (and twice in CI, to the job's limit): the main thread and one of V8's own threads both asleep on a futex,
// inside Node's leaving (24.20; nodejs/node 54918 tells of one like it). With this pause, 0 of 400.
setTimeout(() => process.stdout.write("", () => process.exit(differ ? 1 : 0)), 200);
