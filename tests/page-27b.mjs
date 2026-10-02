// page-27b.mjs (T233): Ternary Bonsai 2 27B on the page's forward pass (public/forward.js, the ternary kernels, the
// rotated basis, a 64-bit shared memory, software threads), in Node, for CI: the model is 7.66 GB as the ternary
// checkpoint, so the development machine runs none of this (tests/page_27b.sh has the stages and fetches the files).
//
//   node tests/page-27b.mjs <out> compare <references> [--texts 0,1,2,3] [--broken <name>,...] [--weak <name>,...]
//                                                        [--expect-failure]
//                                                        [--threads 4] [--lines none | <logits>,<kl>,<kv>]
//   node tests/page-27b.mjs <out> speed [--threads 1,2,4] [--positions 12] [--rounds 2]
//   node tests/page-27b.mjs <out> memory [--threads 4]
//   node tests/page-27b.mjs <out> write <prompt> [--tokens 1500] [--seed 1] [--threads 4]
//   node tests/page-27b.mjs <out> long <references> [--context 8192] [--threads 4] [--lines none | <logits>,<kl>,<growth>] [--broken <name>,...]
//   node tests/page-27b.mjs <out> convert <folder with config.json, the tokenizer and the GGUF> [--context 4096]
//
// <out>: what tests/page_27b.py convert (or tests/perplexity_prepare.py) wrote, <out>.bin, <out>.tokenizer.bin and
// <out>.json (--wide: on a 64-bit memory whatever its size, as the 27B is). The engine gets the options the page's worker merges: the conversion's, then those of the list's entry
// (--entry <id>, hf-ternary-bonsai-2-27b where the list has it; --entry none: the conversion's alone).
//
// compare: what the review of T238 listed (TODO.md, "T233 が比べること"). <references> is the work directory of
//   tests/reference_27b.sh run with SAVE_LOGITS=<references>/saved: fork-<i>.ids (the fork's own ids of each text, and
//   the 16 tokens it wrote), f32/fork-<i>.single (the logits of the fork with float32 activations, a token at a time),
//   saved/engine-<i>-as-8-bits-round.{logits,keys,values} (the engine's NumPy forward pass with the activations rounded
//   as the ternary kernels round them: the run the page's path corresponds to). The ids go through forward.js one at a
//   time, on --threads threads. Then, for every text:
//     the keys and values of the layers that attend (keysAndValues) against the reference's, rounded to the type the
//       page keeps them in (float16 on a shared memory for this model: keysInHalf);
//     the logits against the float32 fork's: the largest difference, the KL of every position, the most likely token;
//     the 16 tokens the fork wrote: the most likely token after the prompt and after each of them.
//   The lines are LINES below. --broken: the engine broken on purpose in ways that need no other checkpoint (a sign of
//   the rotated basis the other way round, the embedding's rows not turned back), each on the first text: it must
//   fail. --weak: the same for errors smaller than what the rounding of the activations moves this comparison by (one
//   sign of the 17408 or of the 6144 values): what is seen of them is said, and it fails nothing.
//   --expect-failure: the checkpoint itself is a broken conversion (tests/page_27b.py --broken): it must fail.
// speed: tokens a second with each count of threads (the counts in turn, --rounds times, the median), the logits the
//   same to the bit with every count, a prompt in blocks (forwardMany) of 4 and 16 tokens and its last logits the same
//   to the bit, what is placed after the checkpoint against footprint(), and that no GPU is started (with why).
// memory: the whole context in blocks of 16, as a long prompt goes: what forward.js has placed after the checkpoint
//   at its end against footprint(), which the worker sizes the memory by, and the tokens a second on the way.
// write: the page's generate() (Python's loop, the sampling kernels, the entry's sampling and format) for a prompt:
//   how many tokens until it stops, how long, and the text (--entry hf-ternary-bonsai-2-27b-thinking: the entry that
//   thinks, its format and its sampling).
// long (the review of T233): a prompt past the 4096 positions the list's context is, through the page's way of a prompt
//   (forwardMany in blocks, a token with its logits where the reference has a row), against what the fork of llama.cpp
//   computes for the same ids (tests/reference_27b.sh's `long` stage: <references>/fork-long.{ids,rows,logits} and
//   long.txt). --context 8192: the checkpoint's header says that many positions (the 4 bytes of it are all that differs
//   between the contexts of a ternary checkpoint, so the file made for 4096 serves: the memory is sized by that header
//   and the engine reads it from there, as it would from a file converted with max_seq_len 8192). What it says: the
//   difference of the logits row by row (the largest, the KL, the most likely token) in bands of positions, and
//   whether it grows past the floor of the first rows (under 128 positions: the rounding of the activations, the
//   fork's own batch path and the keys in float16, which no context length touches), the 16 tokens the fork wrote after
//   the prompt, the tokens a second by 500 positions, and what is placed after the checkpoint against footprint()
//   at that context. The lines: the largest difference of a row past 127 positions at most LOGITS or GROWTH times the
//   floor's largest, whichever is more; the KL likewise; a most likely token that differs only where the fork's own first
//   two are no more than twice the row's largest difference apart. --lines none: the numbers alone.
// convert: the page's conversion (Pyodide, NumPy with 32-bit integers, the kernels' quantizer) of the GGUF into a
//   64-bit shared memory through a sink, as the worker's checkpointSink has it, fed 16 MiB at a time; the time, the
//   megabytes a second, Pyodide's heap at the end, and the sha256 of the checkpoint (page_27b.sh holds it to the
//   native conversion's file).
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import v8 from "node:v8";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { compileKernels, createForward, footprint, keysInHalf, needsWide, weightsMemory } from "../public/forward.js";

const root = new URL("../", import.meta.url).pathname;
const GiB = 2 ** 30, MiB = 2 ** 20;
// The lines of compare. Logits against the float32 fork: twice the step the rounding of the activations moves the
// engine's own NumPy forward pass by (the review of T238 saw 0.075 to 0.171 and a KL of 1.0e-3 at most, on three kinds
// of x86-64 and on arm64): 0.35 and 2e-3; the most likely token may be another only where the fork's own first two
// are closer than twice the largest difference. This line does not see an error as small as one sign of the 17408
// values (0.11), which is what the keys and values are for: against the engine's own run with the same rounding the
// noise is float32's order of sums and the rounding steps it tips, not the step itself. kv: the largest difference of
// a key or a value, as a part of the largest key or value of its layer (TODO.md's T233 has what was measured).
const LINES = { logits: 0.35, kl: 2e-3, kv: 0.05 };
// The lines of long (the review of T233): not the fork with float32 activations (it widens every row it multiplies by,
// 9 seconds a token: 15 hours for 6,000) but the fork as it is, whose own batch and one-token paths are 0.1 apart: so the
// floor is measured in the same run, on the first rows, and what is looked for is growth with the position. A row
// past 127 positions may be as far as LOGITS, or GROWTH times the largest of the floor, whichever is more (KL likewise).
const LONG_LINES = { logits: 0.5, kl: 4e-3, growth: 2 };
const FLOOR_BELOW = 128;
// the engine broken on purpose, by what the plan hands forward.js (a sign of the rotated basis: sign-<the width>-<the
// place, or all>) or by a kernel that does nothing (embedding: the rows of the embedding are not turned back)
const breakOf = (name) => {
  if (name === "embedding") return { what: "the embedding's rows not turned back", kernel: "unrotate" };
  const sign = /^sign-(\d+)-(\d+|all)$/.exec(name);
  if (!sign) throw new Error(`no break called ${name}: embedding, or sign-<width>-<place | all>`);
  return { sign: [Number(sign[1]), sign[2] === "all" ? -1 : Number(sign[2])],
    what: sign[2] === "all" ? `every sign of the ${sign[1]} values the other way round` : `the sign at ${sign[2]} of the ${sign[1]} values the other way round` };
};

const spawn = (data) => new Promise((resolve) => {
  const worker = new Worker(new URL("../public/helper.js", import.meta.url));
  worker.once("message", () => resolve({ terminate: () => worker.terminate() }));
  worker.postMessage(data);
});
const base4GiB = (bytes) => bytes > 4 * GiB;
/** whether this engine reads v128.load32_splat of an address above 4 GiB where it is (tests/ternary-check.mjs's canary:
 * a module of a 64-bit memory it imports, splat(address) = lane 0 of v128.load32_splat) */
function splatsRight() {
  const canary = Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 126, 1, 125, 2, 15, 1, 3, 101, 110, 118, 6, 109, 101, 109, 111, 114, 121, 2, 4, 1,
    3, 2, 1, 0, 7, 9, 1, 5, 115, 112, 108, 97, 116, 0, 0, 10, 13, 1, 11, 0, 32, 0, 253, 9, 2, 0, 253, 31, 0, 11]);
  const high = 4 * GiB + 2 * 65536, at = high + 4096;
  const memory = new WebAssembly.Memory({ initial: BigInt(Math.ceil((high + 4 * MiB) / 65536)), address: "i64" });
  const splat = new WebAssembly.Instance(new WebAssembly.Module(canary), { env: { memory } }).exports.splat;
  const F = new Float32Array(memory.buffer);
  F[at / 4] = 1.5;
  F[(at - 2 ** 32) / 4] = 2.5;  // where an address that lost its upper 32 bits would read
  return splat(BigInt(at)) === 1.5;
}
const argmax = (row) => { let at = 0; for (let i = 1; i < row.length; i++) if (row[i] > row[at]) at = i; return at; };
const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];

if (isMainThread) {
  const args = process.argv.slice(2);
  const option = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
  const VALUED = ["--texts", "--broken", "--weak", "--threads", "--lines", "--positions", "--rounds", "--tokens", "--seed", "--entry", "--context"];
  const positional = args.filter((a, i) => !a.startsWith("--") && !VALUED.includes(args[i - 1]));
  const [out, mode, third] = positional;
  if (!out || !["compare", "speed", "memory", "write", "long", "convert"].includes(mode)) {
    console.error("usage: node tests/page-27b.mjs <out> compare <references> | speed | memory | write <prompt> | long <references> | convert <folder>");
    process.exit(2);
  }
  const { pyodideWithEngine } = await import("./engine.mjs");
  const { MODELS, filled } = await import("../src/models.js");
  const asked = option("--entry", "hf-ternary-bonsai-2-27b");
  const entry = MODELS.find((model) => model.id === asked);
  if (mode === "convert") {
    await convert(out, third, Number(option("--context", entry?.conversion?.max_seq_len ?? 4096)), pyodideWithEngine);
    process.exit(0);
  }
  const converted = JSON.parse(fs.readFileSync(`${out}.json`, "utf8"));
  const options = { ...converted, ...(entry?.options ?? {}) };  // as the worker merges them: the entry's win
  console.log(`page: the options of the conversion: ${JSON.stringify({ ...converted, rotated: converted.rotated && { block: converted.rotated.block } })}`);
  console.log(entry ? `page: the options of ${entry.id} over them: ${JSON.stringify(entry.options)}` : `page: no entry ${asked} in the list: the conversion's options alone`);
  const file = `${out}.bin`, size = fs.statSync(file).size;
  const fd = fs.openSync(file, "r");
  const head = Buffer.alloc(28);
  fs.readSync(fd, head, 0, 28, 0);
  const header = Array.from({ length: 7 }, (_, i) => head.readInt32LE(4 * i));
  // --context: the header says another number of positions (the review of T233: a ternary checkpoint of another context
  // is this file with those 4 bytes changed; the memory is made for it and the engine reads it from there)
  const asksContext = args.includes("--context") && Number(option("--context", 0)) !== header[6];
  if (asksContext) header[6] = Number(option("--context", 0));
  // what the worker asks footprint() with (worker.js's forwardOptions): on a shared memory, with relaxed SIMD (Node has it)
  const forwardOptions = { ...options, int8: true, relaxed: true, halfKV: ["int8", "int6", "ternary"].includes(options.dtype), shared: true, outliers: 8, gpu: false };
  const after = footprint(header, size, forwardOptions), wide = needsWide(size, after) || args.includes("--wide"), halfKeys = keysInHalf(header, size, forwardOptions);
  console.log(`page: ${path.basename(file)} is ${size} bytes (${(size / GiB).toFixed(3)} GiB), header ${JSON.stringify(header)}; footprint() counts ` +
    `${(after / GiB).toFixed(3)} GiB after it (${((size + after) / GiB).toFixed(2)} GiB in all), a ${wide ? "64" : "32"}-bit memory, keys and values in ${halfKeys ? "float16" : "float32"}`);
  // The review of T230 and T231: the V8 of Node 24 (13.6) on arm64 reads v128.load32_splat of an address above 4 GiB at
  // its low 32 bits, and the ternary kernels take every scale so. The review of T233 found where: in Liftoff, V8's baseline
  // compiler (a function runs as Liftoff's code until its budget runs out and TurboFan's replaces it, within a few
  // milliseconds of a kernel's first call: so a kernel whose first call is above 4 GiB is computed wrongly, and one that
  // begins below, as every kernel of this model does, is not: the 27B's numbers on arm64 agreed with x86-64's); TurboFan's
  // code reads it right; V8 fixed Liftoff in 14.3 (Chrome 143: commit ff9dbb26c2, "[wasm][arm64] Fix splat on memory64").
  // So where the canary (tests/ternary-check.mjs's) fails, Liftoff is turned off for everything compiled from here on
  // (the kernels in the worker too: V8's flags are the process's) and the canary asked again: the numbers are then the
  // model's, as on an engine that reads it right. --anyway: run all the same where it still reads wrongly
  if (base4GiB(size + after) && !splatsRight()) {
    v8.setFlagsFromString("--no-liftoff");
    const right = splatsRight();
    console.log(`page: Node ${process.version} (V8 ${process.versions.v8}, ${process.arch}) reads v128.load32_splat wrongly above 4 GiB in Liftoff's code: ` +
      (right ? "--no-liftoff is set, and TurboFan's code reads it where it is (what is computed here is this model)"
        : `and with --no-liftoff too: what it would compute here is not this model${args.includes("--anyway") ? " (run anyway, as asked)" : " — FAILED (run it on x86-64)"}`));
    if (!right && !args.includes("--anyway")) process.exit(1);
  } else if (base4GiB(size + after)) console.log(`page: Node ${process.version} (V8 ${process.versions.v8}, ${process.arch}) reads v128.load32_splat above 4 GiB where it is`);
  const { memory, base } = weightsMemory(size, { shared: true, wide, after });
  let began = performance.now();
  for (let offset = 0; offset < size;) {
    const length = Math.min(64 * MiB, size - offset);
    offset += fs.readSync(fd, new Uint8Array(memory.buffer, base + offset, length), 0, length, offset);
  }
  fs.closeSync(fd);
  if (asksContext) new Int32Array(memory.buffer, base + 24, 1)[0] = header[6];
  console.log(`page: read into the memory in ${((performance.now() - began) / 1000).toFixed(0)} s${asksContext ? `; its header now says ${header[6]} positions` : ""}`);
  const { pyodide: py } = await pyodideWithEngine({ shared: true, wide });
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${out}.tokenizer.bin`));
  let plan;
  // Python says where every tensor is and makes the tokenizer; the forward pass itself is made in the worker
  const outside = { size, read: (o, l) => new Uint8Array(memory.buffer, base + o, l).slice(),
    start: (p) => { plan = p.toJs({ dict_converter: Object.fromEntries }); return { backend: "", bind() {}, forward() {}, release() {} }; } };
  py.globals.set("OUTSIDE", outside);
  py.globals.set("OPTIONS", py.toPy(options));
  began = performance.now();
  py.runPython(`from llama2_numpy import Llama\nllama = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
  console.log(`page: the engine's plan and its tokenizer in ${((performance.now() - began) / 1000).toFixed(1)} s; BOS ${options.bos}, ` +
    `the answer stops at ${JSON.stringify(options.stop_tokens)}`);
  const job = { mode, bos: Number(options.bos ?? 1), threads: option("--threads", mode === "speed" ? "1,2,4" : "4").split(",").map(Number) };
  if (mode === "compare") {
    const references = path.resolve(third);
    job.references = references;
    const linesAsked = option("--lines", "");  // none, or logits,kl,kv (a made-up model's own)
    job.lines = linesAsked === "none" ? null : linesAsked ? Object.fromEntries(linesAsked.split(",").map((value, i) => [["logits", "kl", "kv"][i], Number(value)])) : LINES;
    job.broken = option("--broken", "").split(",").filter(Boolean);
    job.weak = option("--weak", "").split(",").filter(Boolean);
    job.weak.forEach(breakOf);
    job.expectFailure = args.includes("--expect-failure");
    job.broken.forEach(breakOf);  // (a name that is none is said before anything runs)
    job.texts = [];
    const only = option("--texts", "");
    for (let index = 0; fs.existsSync(`${references}/fork-${index}.ids`); index++) {
      if (only && !only.split(",").map(Number).includes(index)) continue;
      const [prompt, wrote] = fs.readFileSync(`${references}/fork-${index}.ids`, "utf8").split("\n").slice(0, 2).map((line) => line.trim().split(/\s+/).map(Number));
      job.texts.push({ index, prompt, wrote });
      // the engine's tokenizer (the converter's tokenizer.bin, the merged options' specials) against the fork's ids:
      // a text as it is, and for a text that is the entry's format of a question, the ids the page sends for that
      // question: its BOS, then the format filled (the review of T250: what stands in front matters to this family)
      const textFile = `${references}/prompt-${index}.txt`;
      if (!fs.existsSync(textFile)) continue;
      const text = fs.readFileSync(textFile, "utf8");
      py.globals.set("TEXT", text);
      const ids = py.runPython("llama.tokenizer.encode(TEXT, llama.specials)").toJs();
      const same = (a, b) => a.length === b.length && a.every((id, i) => id === b[i]);
      console.log(`page: text ${index}: the tokenizer gives ${same(ids, prompt) ? "the fork's ids" : `${JSON.stringify(Array.from(ids))}, the fork ${JSON.stringify(prompt)} — FAILED`}`);
      job.tokenizerFailed ||= !same(ids, prompt);
      const question = "What is 17 times 24?";
      if (entry?.template && text.includes(question) && text === `<|im_start|>${filled(entry.template, question)}`) {
        py.globals.set("TEXT", filled(entry.template, question));
        const sent = [Number(options.bos), ...py.runPython("llama.tokenizer.encode(TEXT, llama.specials)").toJs()];
        console.log(`page: text ${index} is ${entry.id}'s format of a question: the page sends ${same(sent, prompt) ? "the fork's ids exactly (its BOS, then the format)" : `${JSON.stringify(sent)}, the fork ${JSON.stringify(prompt)} — FAILED`}`);
        job.tokenizerFailed ||= !same(sent, prompt);
      }
    }
    if (!job.texts.length) throw new Error(`no fork-<i>.ids in ${references}`);
  }
  if (mode === "long") {
    const references = path.resolve(third);
    const linesAsked = option("--lines", "");  // none, or logits,kl,growth
    job.lines = linesAsked === "none" ? null : linesAsked ? Object.fromEntries(linesAsked.split(",").map((value, i) => [["logits", "kl", "growth"][i], Number(value)])) : LONG_LINES;
    job.references = references;
    job.broken = option("--broken", "").split(",").filter(Boolean);
    job.broken.forEach(breakOf);
    const [prompt, wrote = []] = fs.readFileSync(`${references}/fork-long.ids`, "utf8").split("\n").slice(0, 2).map((line) => line.trim().split(/\s+/).filter(Boolean).map(Number));
    const rows = fs.readFileSync(`${references}/fork-long.rows`, "utf8").trim().split(/\s+/).map(Number);
    job.long = { prompt, wrote, rows };
    console.log(`page: the fork's prompt is ${prompt.length} tokens, it wrote ${wrote.length} after it, and kept the logits of ${rows.length} positions of the prompt (${rows[0]} to ${rows.at(-1)})`);
    // the engine's tokenizer against the fork's ids for the text, which the fork took whole
    const textFile = `${references}/long.txt`;
    if (fs.existsSync(textFile)) {
      py.globals.set("TEXT", fs.readFileSync(textFile, "utf8"));
      const ids = Array.from(py.runPython("llama.tokenizer.encode(TEXT, llama.specials)").toJs());
      const first = ids.findIndex((id, i) => id !== prompt[i]);
      const same = ids.length === prompt.length && first < 0;
      console.log(`page: the tokenizer gives ${same ? `the fork's ${ids.length} ids` : `${ids.length} ids, the fork ${prompt.length}; the first that differs is at ${first} — FAILED`}`);
      job.tokenizerFailed ||= !same;
    }
  }
  if (mode === "speed") Object.assign(job, { positions: Number(option("--positions", 12)), rounds: Number(option("--rounds", 2)) });
  if (mode === "write") {
    // generate() is Python's loop: Pyodide and the forward pass in one worker, as on the page
    if (!entry?.template) throw new Error("write needs an entry of the list with a format");
    Object.assign(job, { prompt: third, thinking: entry.template.endsWith("<think>\n"), entry: entry.id, tokens: Number(option("--tokens", 1500)),
      seed: Number(option("--seed", 1)), sampling: entry.generation, options, template: entry.template, tokenizer: fs.readFileSync(`${out}.tokenizer.bin`) });
    job.text = filled(job.template, third);
  }
  const worker = new Worker(new URL(import.meta.url), { workerData: { memory, base, size, plan, wide, halfKeys, header, forwardOptions, job } });
  const failed = await new Promise((resolve, reject) => {
    worker.on("message", (message) => (message.line !== undefined ? console.log(message.line) : resolve(message.failed)));
    worker.once("error", reject);
  });
  await worker.terminate();
  process.exit(failed || job.tokenizerFailed ? 1 : 0);
} else {
  const { memory, base, size, plan, wide, halfKeys, header, forwardOptions, job } = workerData;
  const say = (line) => parentPort.postMessage({ line: `page: ${line}` });
  const suffix = wide ? "64" : "";
  const kernels = compileKernels(fs.readFileSync(`${root}public/simdkernel_shared${suffix}.wasm`), fs.readFileSync(`${root}public/simdkernel_relaxed_shared${suffix}.wasm`), wide);
  const cpu = (await import("node:os")).cpus()[0].model;
  /** an engine on the memory, as the page's worker makes it; broken: what breakOf() made of a name */
  const engineOf = (broken) => {
    let made = plan, wrap;
    if (broken?.sign) {
      const [width, at] = broken.sign, signs = new Float32Array(plan.derived[`signs.${width}`].slice().buffer);
      for (let i = at < 0 ? 0 : at; i < (at < 0 ? width : at + 1); i++) signs[i] = -signs[i];
      made = { ...plan, derived: { ...plan.derived, [`signs.${width}`]: new Uint8Array(signs.buffer) } };
    }
    if (broken?.kernel) wrap = (exports) => ({ ...exports, [broken.kernel]: () => {} });
    return createForward({ memory, base, size, kernels, plan: made, spawn, halfKeys, ...(wrap ? { wrap } : {}) });
  };
  const floats = (file) => {
    const data = fs.readFileSync(file);
    return new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  };
  const vocab = plan.vocab_size;
  const logSoftmax = (row) => {
    let top = -Infinity;
    for (let i = 0; i < row.length; i++) if (row[i] > top) top = row[i];
    let sum = 0;
    for (let i = 0; i < row.length; i++) sum += Math.exp(row[i] - top);
    const shift = top + Math.log(sum), out = new Float64Array(row.length);
    for (let i = 0; i < row.length; i++) out[i] = row[i] - shift;
    return out;
  };
  /** our rows against theirs (flat, [positions][vocab]): the largest difference, the KL(theirs || ours) of every
   * position, where the most likely token differs and how far apart their own first two are there */
  const distance = (ours, theirs) => {
    const count = Math.min(ours.length, theirs.length / vocab);
    const found = { count, worst: 0, where: 0, klWorst: 0, klSum: 0, same: 0, gaps: [] };
    for (let position = 0; position < count; position++) {
      const a = ours[position], b = theirs.subarray(position * vocab, (position + 1) * vocab);
      let worst = 0;
      for (let i = 0; i < vocab; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
      if (worst > found.worst) [found.worst, found.where] = [worst, position];
      const la = logSoftmax(a), lb = logSoftmax(b);
      let kl = 0;
      for (let i = 0; i < vocab; i++) kl += Math.exp(lb[i]) * (lb[i] - la[i]);
      found.klSum += kl;
      found.klWorst = Math.max(found.klWorst, kl);
      const top = argmax(b);
      if (argmax(a) === top) found.same += 1;
      else {
        let second = -Infinity;
        for (let i = 0; i < vocab; i++) if (i !== top && b[i] > second) second = b[i];
        found.gaps.push({ position, gap: b[top] - second });
      }
    }
    return found;
  };
  let failed = false;
  if (job.mode === "compare") {
    const { references, lines } = job;
    const told = (d) => `largest difference ${d.worst.toFixed(4)} (at position ${d.where}), KL ${(d.klSum / d.count).toExponential(2)} on average and ` +
      `${d.klWorst.toExponential(2)} at most, the same most likely token at ${d.same} of ${d.count} positions` +
      (d.gaps.length ? ` (${d.gaps.slice(0, 4).map(({ position, gap }) => `position ${position}: the other's first two are ${gap.toFixed(4)} apart`).join("; ")})` : "");
    /** the keys or values of forward.js against the reference's ([layers][positions][kvDim], float32), the reference
     * rounded to the type the page keeps them in: the largest difference of every layer as a part of that layer's
     * largest value, and the worst of those */
    const cached = (ours, theirs, count) => {
      const layers = ours.length / count / (plan.n_kv_heads * plan.head_size), row = count * plan.n_kv_heads * plan.head_size;
      const each = [];
      for (let layer = 0; layer < layers; layer++) {
        let worst = 0, largest = 0, apart = 0, size = 0;
        for (let i = layer * row; i < (layer + 1) * row; i++) {
          const want = halfKeys ? Math.f16round(theirs[i]) : theirs[i];
          worst = Math.max(worst, Math.abs(ours[i] - want));
          largest = Math.max(largest, Math.abs(want));
          apart += (ours[i] - want) ** 2;
          size += want ** 2;
        }
        each.push({ worst, largest, part: worst / largest, rms: Math.sqrt(apart / size) });
      }
      return { each, part: Math.max(...each.map((layer) => layer.part)), worst: Math.max(...each.map((layer) => layer.worst)),
        rms: Math.max(...each.map((layer) => layer.rms)) };
    };
    const runs = [null, ...job.broken, ...job.weak];
    for (const name of runs) {
      const engine = engineOf(name ? breakOf(name) : null);
      await engine.setThreads(job.threads[0]);
      for (const text of name ? job.texts.slice(0, 1) : job.texts) {
        const ids = [...text.prompt, ...text.wrote.slice(0, -1)], rows = [];
        const began = performance.now();
        for (let position = 0; position < ids.length; position++) {
          engine.forward(ids[position], position, true);
          rows.push(engine.logits().slice());
        }
        const seconds = (performance.now() - began) / 1000;
        const kv = engine.keysAndValues(0, ids.length);
        const label = `text ${text.index}${name ? `, broken (${breakOf(name).what})` : ""}`;
        if (!name) say(`${label}: ${ids.length} positions a token at a time on ${engine.threads} threads in ${seconds.toFixed(0)} s (${(ids.length / seconds).toFixed(2)} tok/s, ${cpu}), ${engine.backend}`);
        const reasons = [];
        // the keys and values, against the engine's NumPy run with the same rounding of the activations
        const stem = `${references}/saved/engine-${text.index}-as-8-bits-round`;
        const parts = {};
        for (const kind of ["keys", "values"]) {
          const found = cached(kv[kind], floats(`${stem}.${kind}`), ids.length);
          parts[kind] = found;
          say(`${label}: the ${kind} against "as 8 bits round" in ${halfKeys ? "float16" : "float32"}: at most ${found.part.toExponential(2)} of a layer's largest ` +
            `(${found.worst.toExponential(2)}); by layer ${found.each.map((layer) => layer.part.toExponential(1)).join(" ")}; ` +
            `the root of the mean square of a layer's differences over that of its ${kind}, at most ${found.rms.toExponential(2)}: ${found.each.map((layer) => layer.rms.toExponential(1)).join(" ")}`);
          if (lines && found.part > lines.kv) reasons.push(`the ${kind} are ${found.part.toExponential(2)} of a layer's largest off, past ${lines.kv}`);
        }
        const rounded = distance(rows, floats(`${stem}.logits`));
        say(`${label}: the logits against "as 8 bits round": ${told(rounded)}`);
        // the logits, against the fork with float32 activations
        const fork = distance(rows, floats(`${references}/f32/fork-${text.index}.single`));
        say(`${label}: the logits against the float32 fork: ${told(fork)}`);
        if (fork.count < ids.length) say(`${label}: the float32 fork wrote ${fork.count} of the ${ids.length} positions`);
        if (lines && fork.worst > lines.logits) reasons.push(`the logits are ${fork.worst.toFixed(4)} from the float32 fork's, past ${lines.logits}`);
        if (lines && fork.klWorst > lines.kl) reasons.push(`a position's KL is ${fork.klWorst.toExponential(2)}, past ${lines.kl}`);
        const far = fork.gaps.filter(({ gap }) => gap > 2 * fork.worst);
        if (lines && far.length) reasons.push(`the most likely token is another at ${far.length} position(s) where the fork's first two are more than ${(2 * fork.worst).toFixed(4)} apart`);
        // the 16 tokens the fork wrote (with its rounded activations): the most likely token after the prompt and after
        // each of them, which is what a greedy generation of the page would write as long as it agrees
        const first = text.prompt.length - 1;
        const greedy = text.wrote.map((token, i) => argmax(rows[first + i]) === token);
        const other = greedy.map((same, i) => (same ? null : fork.gaps.find(({ position }) => position === first + i) ?? { position: first + i, gap: 0 })).filter(Boolean);
        say(`${label}: greedy: ${greedy.filter(Boolean).length} of ${text.wrote.length} tokens the fork's` +
          (other.length ? ` (${other.map(({ position, gap }) => `position ${position}: the float32 fork's first two are ${gap.toFixed(4)} apart`).join("; ")})` : ""));
        if (lines && other.some(({ gap }) => gap > 2 * fork.worst)) reasons.push("a greedy token is another where the margin does not allow it");
        const mustFail = job.broken.includes(name) || job.expectFailure;
        if (!lines) say(`${label}: no lines asked for`);
        else if (job.weak.includes(name)) say(`${label}: ${reasons.length ? `seen: ${reasons.join("; ")}` : "not seen: under what the rounding of the activations moves this comparison by"}`);
        else if (mustFail) say(`${label}: ${reasons.length ? `caught: ${reasons.join("; ")}` : "NOT CAUGHT — FAILED"}`);
        else say(`${label}: ${reasons.length ? `FAILED: ${reasons.join("; ")}` : `ok: the keys and values within ${lines.kv} of a layer's largest, the logits within ${lines.logits} and KL ${lines.kl} of the float32 fork's`}`);
        failed ||= Boolean(lines) && !job.weak.includes(name) && (mustFail ? !reasons.length : reasons.length > 0);
      }
      engine.stopThreads();
      await engine.release();
    }
  }
  if (job.mode === "speed") {
    const engine = engineOf(null), { positions, rounds } = job, counts = job.threads;
    const greedy = () => {
      const seen = [];
      let token = job.bos;
      const began = performance.now();
      for (let position = 0; position < positions; position++) {
        engine.forward(token, position, true);
        seen.push(engine.logits().slice());
        token = argmax(seen[position]);
      }
      return { seen, tokensPerSecond: positions * 1000 / (performance.now() - began) };
    };
    let reference;
    const times = Object.fromEntries(counts.map((n) => [n, []])), differ = [];
    for (let round = 0; round < rounds; round++) {
      for (const n of counts) {
        await engine.setThreads(n);
        const { seen, tokensPerSecond } = greedy();
        times[n].push(tokensPerSecond);
        reference ??= seen;
        if (!seen.every((row, i) => row.every((v, j) => Object.is(v, reference[i][j]))) && !differ.includes(n)) differ.push(n);
      }
    }
    const one = median(times[counts[0]]);
    say(`${cpu}, ${engine.backend}: ${differ.length ? `logits DIFFER with ${differ.join(", ")} threads — FAILED` : `logits the same to the bit with ${counts.join(", ")} threads`} ` +
      `(${positions} positions from the BOS, ${rounds} rounds); ` + counts.map((n) => `${n}: ${median(times[n]).toFixed(3)} tok/s (${(median(times[n]) / one).toFixed(2)}×)`).join(", "));
    failed ||= differ.length > 0;
    // the same text as a prompt in blocks, then the last token with its logits
    const text = [job.bos, ...reference.slice(0, -1).map(argmax)];
    const blocks = [];
    for (const n of counts) {
      await engine.setThreads(n);
      for (const k of [4, 16]) {
        const began = performance.now();
        for (let at = 0; at < positions - 1; at += k) engine.forwardMany(text.slice(at, Math.min(at + k, positions - 1)), at);
        const tokensPerSecond = (positions - 1) * 1000 / (performance.now() - began);
        engine.forward(text[positions - 1], positions - 1, true);
        const same = engine.logits().every((v, j) => Object.is(v, reference[positions - 1][j]));
        blocks.push(`${n} thread(s), blocks of ${k}: ${tokensPerSecond.toFixed(3)} tok/s${same ? "" : " and the last logits DIFFER — FAILED"}`);
        failed ||= !same;
      }
    }
    say(`a prompt of ${positions - 1} tokens in blocks, the last logits the same to the bit as a token at a time: ${blocks.join("; ")}`);
    const placed = engine.memoryBytes() - base - size, bound = footprint(header, size, forwardOptions);
    say(`after the checkpoint at position ${positions}: ${(placed / MiB).toFixed(1)} MiB placed, footprint() counts ${(bound / MiB).toFixed(1)} MiB for the whole context of ${plan.seq_len}`);
    engine.stopThreads();
    await engine.release();
    // a browser with WebGPU: the page's worker hands forward.js the GPU's worker, and this model must not start one
    let spawned = 0;
    const withGpu = createForward({ memory, base, size, kernels, plan, spawn, halfKeys,
      gpu: () => { spawned++; return { postMessage() {}, set onmessage(handler) {}, set onerror(handler) {}, terminate() {} }; } });
    const status = await Promise.race([withGpu.gpu, new Promise((resolve) => setTimeout(() => resolve("no answer in 3 s"), 3000))]);
    say(`with a GPU offered: ${spawned ? `its worker was started ${spawned} time(s) — FAILED` : "no worker of the GPU is started"}; the status line: ${JSON.stringify(withGpu.gpuStatus ?? status)}`);
    failed ||= spawned > 0;
    await withGpu.release();
  }
  if (job.mode === "memory") {
    const engine = engineOf(null), seqLen = plan.seq_len, bound = footprint(header, size, forwardOptions);
    await engine.setThreads(job.threads[0]);
    let began = performance.now(), last = 0;
    const every = Math.max(256, seqLen >> 4);
    for (let at = 0; at < seqLen; at += 16) {
      // (some text: the tokens 1000 to 1999 in turn; forwardMany makes no logits)
      engine.forwardMany(Array.from({ length: Math.min(16, seqLen - at) }, (_, i) => 1000 + (at + i) % 1000), at);
      const done = Math.min(at + 16, seqLen);
      if (done - last >= every || done === seqLen) {
        const now = performance.now();
        say(`positions ${last} to ${done}: ${((done - last) * 1000 / (now - began)).toFixed(2)} tok/s in blocks of 16 on ${engine.threads} threads, ` +
          `${((engine.memoryBytes() - base - size) / MiB).toFixed(1)} MiB after the checkpoint`);
        [began, last] = [now, done];
      }
    }
    const placed = engine.memoryBytes() - base - size;
    // above what was placed, and by little (tests/forward-check.mjs's line: a few percent and the megabytes of alignment)
    const close = placed <= bound && bound - placed <= 0.05 * bound + 6 * MiB;
    say(`${cpu}: at the end of the context of ${seqLen}: ${(placed / MiB).toFixed(1)} MiB placed after the checkpoint, footprint() counts ${(bound / MiB).toFixed(1)} MiB` +
      `${close ? "" : " — FAILED"}; in all ${((base + size + placed) / GiB).toFixed(3)} GiB`);
    failed ||= !close;
    engine.stopThreads();
    await engine.release();
  }
  if (job.mode === "long") {
    const { references, lines } = job, { prompt, wrote, rows } = job.long, seqLen = plan.seq_len;
    // the fork's rows: those of the prompt's positions listed in fork-long.rows, then one for each token it wrote but the last
    // (decoded one at a time): a fork that ran out of time wrote fewer than the list says
    const kept = floats(`${references}/fork-long.logits`);
    const positions = [...rows, ...Array.from({ length: Math.max(0, wrote.length - 1) }, (_, i) => prompt.length + i)];
    const have = Math.min(positions.length, Math.floor(kept.length / vocab));
    const wanted = new Set(positions.slice(0, have)), stop = Math.max(...wanted) + 1;
    const ids = [...prompt, ...wrote.slice(0, -1)];
    if (stop > seqLen) throw new Error(`the fork's rows go to position ${stop - 1}, the engine's context is ${seqLen} positions: --context ${stop} or more`);
    let limits = null;  // (the right engine's lines, which a broken one is held to)
    for (const name of [null, ...job.broken]) {
      const engine = engineOf(name ? breakOf(name) : null);
      await engine.setThreads(job.threads[0]);
      const label = name ? `broken (${breakOf(name).what})` : "the page's forward pass";
      const got = new Map(), bands = [];
      const began = performance.now();
      let bandBegan = began, bandFrom = 0;
      for (let at = 0; at < stop;) {
        if (wanted.has(at)) {
          engine.forward(ids[at], at, true);
          got.set(at, engine.logits().slice());
          at += 1;
        } else {
          // (a run of positions the reference has no row for: a prompt in blocks, as the page's generate() hands one over)
          let next = at;
          while (next < stop && !wanted.has(next)) next++;
          engine.forwardMany(ids.slice(at, next), at);
          at = next;
        }
        if (at - bandFrom >= 500 || at === stop) {
          const now = performance.now();
          bands.push({ from: bandFrom, to: at, rate: (at - bandFrom) * 1000 / (now - bandBegan) });
          if (!name) say(`positions ${bandFrom} to ${at}: ${bands.at(-1).rate.toFixed(2)} tok/s on ${engine.threads} threads, ${((engine.memoryBytes() - base - size) / MiB).toFixed(0)} MiB after the checkpoint`);
          [bandBegan, bandFrom] = [now, at];
        }
      }
      const seconds = (performance.now() - began) / 1000;
      // the rows, one by one, against the fork's
      const found = [];
      for (let k = 0; k < have; k++) {
        const position = positions[k], d = distance([got.get(position)], kept.subarray(k * vocab, (k + 1) * vocab));
        found.push({ position, worst: d.worst, kl: d.klWorst, same: d.same === 1, gap: d.gaps[0]?.gap });
      }
      const band = (from, to) => found.filter(({ position }) => position >= from && position < to);
      const tell = (list) => list.length ? `${list.length} rows, largest difference ${Math.max(...list.map((r) => r.worst)).toFixed(4)}, KL ${Math.max(...list.map((r) => r.kl)).toExponential(2)} at most, ` +
        `the same most likely token at ${list.filter((r) => r.same).length}` : "no rows";
      const edges = [FLOOR_BELOW, 1024, 2048, 4096, Infinity];
      say(`${label}: ${stop} positions in ${seconds.toFixed(0)} s (${(stop / seconds).toFixed(2)} tok/s, ${cpu}), ${engine.backend}`);
      if (!name) say(`${label}: by band of positions (the first, under ${FLOOR_BELOW}, is the floor): ` + edges.map((to, i) => {
        const from = i ? edges[i - 1] : 0;
        return `${from} to ${to === Infinity ? "the end" : to}: ${tell(band(from, to))}`;
      }).join("; "));
      say(`${label}: the rows: ` + found.map((r) => `${r.position}: ${r.worst.toFixed(3)}/${r.kl.toExponential(1)}${r.same ? "" : " (another most likely token)"}`).join(", "));
      // what is looked for: a difference that grows with the position, past the floor the first rows measure. A broken engine is
      // held to the lines of the right one (its own first rows are as wrong as the rest): every row of it is looked at
      const floor = band(0, FLOOR_BELOW), past = found.filter(({ position }) => position >= FLOOR_BELOW);
      const reasons = [];
      if (lines) {
        if (!name) {
          const floorWorst = Math.max(0, ...floor.map((r) => r.worst)), floorKl = Math.max(0, ...floor.map((r) => r.kl));
          limits = { floorWorst, floorKl, worst: Math.max(lines.logits, lines.growth * floorWorst), kl: Math.max(lines.kl, lines.growth * floorKl) };
          say(`${label}: the floor (under ${FLOOR_BELOW} positions) is ${floorWorst.toFixed(4)} and KL ${floorKl.toExponential(2)}; the lines past it: ${limits.worst.toFixed(4)} and ${limits.kl.toExponential(2)}`);
        }
        for (const r of name ? found : past) {
          if (r.worst > limits.worst) reasons.push(`position ${r.position}: ${r.worst.toFixed(4)} from the fork's, past ${limits.worst.toFixed(4)}`);
          if (r.kl > limits.kl) reasons.push(`position ${r.position}: KL ${r.kl.toExponential(2)}, past ${limits.kl.toExponential(2)}`);
          if (!r.same && r.gap > 2 * r.worst) reasons.push(`position ${r.position}: another most likely token where the fork's first two are ${r.gap.toFixed(4)} apart (more than twice ${r.worst.toFixed(4)})`);
        }
      }
      const greedy = wrote.map((token, i) => (got.has(prompt.length - 1 + i) ? argmax(got.get(prompt.length - 1 + i)) === token : null));
      say(`${label}: greedy: ${greedy.filter((same) => same).length} of the ${greedy.filter((same) => same !== null).length} tokens the fork wrote after the prompt are the most likely ones after it`);
      const mustFail = Boolean(name);
      if (!lines) say(`${label}: no lines asked for`);
      else if (mustFail) say(`${label}: ${reasons.length ? `caught (${reasons.length} rows): ${reasons[0]}` : "NOT CAUGHT — FAILED"}`);
      else say(`${label}: ${reasons.length ? `FAILED: ${reasons.slice(0, 6).join("; ")}` : "ok: no row past the floor's lines"}`);
      failed ||= Boolean(lines) && (mustFail ? !reasons.length : reasons.length > 0);
      if (!name) {
        const placed = engine.memoryBytes() - base - size, bound = footprint(header, size, forwardOptions);
        const close = placed <= bound && bound - placed <= 0.05 * bound + 6 * MiB;
        say(`${label}: after ${stop} positions of a context of ${seqLen}: ${(placed / MiB).toFixed(1)} MiB placed after the checkpoint, footprint() counts ${(bound / MiB).toFixed(1)} MiB${close ? "" : " — FAILED"}; ` +
          `in all ${((base + size + placed) / GiB).toFixed(3)} GiB`);
        failed ||= !close;
      }
      engine.stopThreads();
      await engine.release();
    }
  }
  if (job.mode === "write") {
    const { pyodideWithEngine } = await import("./engine.mjs");
    const { external } = await import("../public/forward.js");
    const { pyodide: py } = await pyodideWithEngine({ shared: true, wide });
    py.FS.writeFile("tokenizer.bin", job.tokenizer);
    const live = external({ memory, base, size, kernels, spawn, halfKeys });
    py.globals.set("OUTSIDE", live);
    py.globals.set("OPTIONS", py.toPy(job.options));
    py.runPython(`from llama2_numpy import Llama\nllama = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
    await live.engine.setThreads(job.threads[0]);
    const { sampling } = job;
    py.globals.set("PROMPT", job.text);
    py.globals.set("SETTINGS", py.toPy({ steps: job.tokens, temperature: sampling.temperature, topp: sampling.topp, repetition_penalty: sampling.repetition_penalty, seed: job.seed }));
    say(`${job.entry}: writing with ${JSON.stringify(sampling)} and the seed ${job.seed}, ${live.engine.threads} threads, at most ${job.tokens} positions, the format ${JSON.stringify(job.template)}`);
    const began = performance.now();
    const text = py.runPython(`"".join(llama.generate(PROMPT, echo=False, **SETTINGS))`);
    const seconds = (performance.now() - began) / 1000;
    const stats = py.runPython("llama.stats").toJs({ dict_converter: Object.fromEntries });
    const closed = text.indexOf("</think>");
    py.globals.set("THOUGHT", text.slice(0, Math.max(closed, 0)));
    const thought = py.runPython("len(llama.tokenizer.encode(THOUGHT, llama.specials))");
    say(`${JSON.stringify(job.prompt)}${job.thinking ? " (thinking)" : ""}: ${seconds.toFixed(0)} s, ${JSON.stringify(stats)}, ${cpu}; ` +
      (job.thinking ? (closed < 0 ? "the thought did not end; " : `the thought ended after ${thought} tokens (${closed} characters); `) : "") + `it wrote ${JSON.stringify(text)}`);
    await live.engine.release();
  }
  parentPort.postMessage({ failed });
}

/** The page's conversion in Pyodide, into a memory of forward.js through a sink (worker.js's checkpointSink) */
async function convert(out, folder, context, pyodideWithEngine) {
  const { pyodide: py } = await pyodideWithEngine({ shared: true, wide: true });
  const convert = py.pyimport("llama2_convert"), numpy = py.pyimport("llama2_numpy");
  const quantizeRows = numpy.kernel_quantizer("simdkernel.so"), bfloat16 = numpy.kernel_widener("simdkernel.so"), q8_0 = numpy.kernel_q8_0("simdkernel.so");
  const gguf = fs.readdirSync(folder).filter((name) => name.endsWith(".gguf")).sort()[0];
  const tokenizer = ["tokenizer.json", "spiece.model", "tokenizer.model"].find((name) => fs.existsSync(`${folder}/${name}`));
  const fd = fs.openSync(`${folder}/${gguf}`, "r"), size = fs.fstatSync(fd).size;
  const range = (begin, end) => { const bytes = new Uint8Array(end - begin); fs.readSync(fd, bytes, 0, end - begin, begin); return bytes; };
  const config = fs.readFileSync(`${folder}/config.json`, "utf8");
  const tokenizerConfig = fs.existsSync(`${folder}/tokenizer_config.json`) ? fs.readFileSync(`${folder}/tokenizer_config.json`, "utf8") : "";
  // the GGUF's header in growing pieces, as the worker fetches it (4 x HF_HEADER_BYTES, then four times as much)
  let header, first;
  for (let bytes = 4 * 512 * 1024; ; bytes *= 4) {
    try {
      const made = convert.gguf_weights(range(0, Math.min(bytes, size)), config);
      [header, first] = made.toJs();
      made.destroy();
      break;
    } catch (error) {
      if (error.type !== "Incomplete" || bytes >= size) throw error;
    }
  }
  const into = {};
  const sink = {
    open(bytes, head, dtype, form) {
      const ints = head.toJs(), options = { dtype, ...form.toJs({ dict_converter: Object.fromEntries }) };
      head.destroy();
      form.destroy();
      const forwardOptions = { ...options, int8: true, relaxed: true, halfKV: true, shared: true, outliers: 8, gpu: false };
      const after = footprint(ints, bytes, forwardOptions), wide = needsWide(bytes, after);
      Object.assign(into, weightsMemory(bytes, { shared: true, wide, after }), { bytes, wide, after, ints });
    },
    write(offset, array) {
      const view = array.getBuffer("u8");
      new Uint8Array(into.memory.buffer, into.base + offset, view.data.length).set(view.data);
      view.release();
    },
  };
  const conversion = convert.Conversion.callKwargs(header, first, config, new Uint8Array(fs.readFileSync(`${folder}/${tokenizer}`)), tokenizer,
    { start: first, tokenizer_config: tokenizerConfig, dtype: "ternary", max_seq_len: context, sink, quantize_rows: quantizeRows, bfloat16, q8_0 });
  console.log(`pyodide: ${gguf} (${size} bytes) to ${into.bytes} bytes of ternary checkpoint, header ${JSON.stringify(into.ints)}, on a ${into.wide ? "64" : "32"}-bit shared memory ` +
    `(footprint() counts ${(into.after / GiB).toFixed(3)} GiB after it); Pyodide's heap is ${(py._module.HEAPU8.length / MiB).toFixed(0)} MiB before the weights`);
  const began = performance.now();
  let reading = 0;
  for (let at = first; at < size; at += 16 * MiB) {
    const t = performance.now(), chunk = range(at, Math.min(at + 16 * MiB, size));
    reading += performance.now() - t;
    conversion.feed(chunk);
  }
  conversion.finish();
  const seconds = (performance.now() - began) / 1000;
  console.log(`pyodide: converted in ${seconds.toFixed(0)} s (${(size / 1e6 / seconds).toFixed(1)} MB/s of the file, ${(reading / 1000).toFixed(0)} s of them reading it), ` +
    `Pyodide's heap is ${(py._module.HEAPU8.length / MiB).toFixed(0)} MiB at the end`);
  const hash = crypto.createHash("sha256");
  for (let offset = 0; offset < into.bytes; offset += 64 * MiB) {
    hash.update(new Uint8Array(into.memory.buffer, into.base + offset, Math.min(64 * MiB, into.bytes - offset)).slice());
  }
  const view = conversion.tokenizer.getBuffer("u8");
  const vocabulary = crypto.createHash("sha256").update(view.data.slice()).digest("hex");
  view.release();
  const options = conversion.options.toJs({ dict_converter: Object.fromEntries });
  console.log(`pyodide: sha256 of the checkpoint ${hash.digest("hex")}`);
  console.log(`pyodide: sha256 of tokenizer.bin ${vocabulary}`);
  if (out !== "-") fs.writeFileSync(`${out}.pyodide.json`, JSON.stringify({ ...options, template: undefined }));
}
