// t246_page.mjs (T246, a probe for CI, not for main): a Ternary Bonsai on the page's own forward pass (forward.js on the
// int8 kernels, 7-bit activations, software threads on a shared memory, a 64-bit one where the model needs it), the
// checkpoint read from its file:
//   1. tokens/s by the number of threads, a prompt in blocks and the tokens written after it; the logits of the last
//      token the same to the bit with every count
//   2. what it writes greedily from several prompts in the model's template (the converter's, which
//      tests/format_check.py held to transformers' apply_chat_template)
//   3. the perplexity of texts under config.json's yarn and under a plain RoPE (rope_scaling: {}, what an entry's
//      options would say), the same ids ([BOS] + the tokenizer's) as tests/t246_reference.py feeds transformers
// The forward pass runs in a worker, as on the page; Python in the main thread makes the plans and the ids.
//
//   node tests/t246_page.mjs <out of tests/perplexity_prepare.py> <model id> [--threads 1,2,4] [--tokens 64]
//        [--rounds 3] [--ppl <text file>:<tokens> ...]
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { compileKernels, createForward, footprint, keysInHalf, needsWide, weightsMemory } from "../public/forward.js";

const root = new URL("../", import.meta.url).pathname;
const TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const PROMPTS = ["これからの流行りを3つ挙げてください。", "日本で一番高い山はどこですか。理由も説明してください。",
  "What is the capital of Japan? Answer in two sentences.", "富士山について、三つの文で説明してください。",
  "次の文を英語に訳してください。「今日は天気がいいので、散歩に行きます。」"];
const spawn = (data) => new Promise((resolve) => {
  const worker = new Worker(new URL("../public/helper.js", import.meta.url));
  worker.once("message", () => resolve({ terminate: () => worker.terminate() }));
  worker.postMessage(data);
});

if (isMainThread) {
  const { pyodideWithEngine } = await import("./engine.mjs");
  const { MODELS } = await import("../src/models.js");
  const args = process.argv.slice(2);
  const option = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
  const [out, id] = args;
  const counts = option("--threads", "1,2,4").split(",").map(Number), tokens = Number(option("--tokens", 64)), rounds = Number(option("--rounds", 3));
  const texts = args.includes("--ppl") ? args.slice(args.indexOf("--ppl") + 1).filter((a) => a.includes(":") && !a.startsWith("--")) : [];
  const entry = MODELS.find((model) => model.id === id);
  // what the worker merges: the conversion's options, then the entry's (bos, stop_tokens)
  const options = { ...JSON.parse(fs.readFileSync(`${out}.json`, "utf8")), ...entry.options };
  const file = path.resolve(`${out}.bin`), size = fs.statSync(file).size;
  const head = new Int32Array(7), handle = fs.openSync(file, "r");
  fs.readSync(handle, new Uint8Array(head.buffer), 0, 28, 0);
  fs.closeSync(handle);
  const header = [...head];
  const form = { arch: options.arch, head_dim: options.head_dim ?? 0 };
  const forward = { ...form, dtype: "int8", int8: true, relaxed: true, halfKV: true, shared: true, outliers: 8 };
  const after = footprint(header, size, forward), wide = needsWide(size, after), halfKeys = keysInHalf(header, size, forward);
  console.log(`T246 ${id}: ${os.cpus()[0].model}, ${os.cpus().length} logical cores, ${(os.totalmem() / 2 ** 30).toFixed(1)} GiB; checkpoint ${size} bytes, header ${JSON.stringify(header)}, ` +
    `after it ${(after / 2 ** 30).toFixed(3)} GiB at ${header[6]} positions, whole ${((size + after) / 2 ** 30).toFixed(3)} GiB, a 64-bit memory: ${wide}, keys and values in float16: ${halfKeys}`);
  console.log(`T246 ${id}: options ${JSON.stringify({ ...options, specials: `${options.specials?.length} of them` })}`);
  const { memory, base } = weightsMemory(size, { shared: true, wide, after });
  const fd = fs.openSync(file, "r");
  for (let offset = 0; offset < size;) {
    const length = Math.min(64 << 20, size - offset);
    offset += fs.readSync(fd, new Uint8Array(memory.buffer, base + offset, length), 0, length, offset);
  }
  fs.closeSync(fd);
  const { pyodide: py } = await pyodideWithEngine();
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${out}.tokenizer.bin`));
  const plans = {};
  for (const [name, change] of [["yarn (config.json)", {}], ["plain RoPE (rope_scaling: {})", { rope_scaling: {} }]]) {
    let plan;
    py.globals.set("OUTSIDE", { size, read: (o, l) => new Uint8Array(memory.buffer, base + o, l).slice(),
      start: (p) => { plan = p.toJs({ dict_converter: Object.fromEntries }); return { backend: "", bind() {}, forward() {}, release() {} }; } });
    py.globals.set("OPTIONS", py.toPy({ ...options, ...change }));
    py.runPython(`from llama2_numpy import Llama\nllama = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
    plans[name] = plan;
  }
  py.globals.set("TEMPLATE", TEMPLATE);
  py.globals.set("PROMPTS", py.toPy(PROMPTS));
  const prompts = py.runPython(`[[llama.bos] + [int(t) for t in llama.tokenizer.encode(TEMPLATE.replace("{prompt}", p), llama.specials)] for p in PROMPTS]`).toJs();
  const stops = py.runPython("sorted(int(t) for t in llama.stop_tokens)").toJs();
  const bos = py.runPython("int(llama.bos)");
  console.log(`T246 ${id}: bos ${bos}, stops ${JSON.stringify(stops)}, the first prompt's ids ${JSON.stringify(prompts[0])}`);
  const passages = texts.map((text) => {
    const [name, count] = [text.slice(0, text.lastIndexOf(":")), Number(text.slice(text.lastIndexOf(":") + 1))];
    const content = fs.readFileSync(name, "utf8");
    py.globals.set("TEXT", content);
    const ids = [bos, ...py.runPython(`[int(t) for t in llama.tokenizer.encode(TEXT)]`).toJs().slice(0, count)];
    console.log(`T246 ${id}: ${path.basename(name)} (sha256 ${crypto.createHash("sha256").update(content).digest("hex").slice(0, 12)}): ` +
      `${ids.length - 1} tokens after the BOS, the first ${JSON.stringify(ids.slice(1, 9))}, their sum ${ids.reduce((a, b) => a + b, 0)}`);
    return { name: path.basename(name), ids };
  });
  const worker = new Worker(new URL(import.meta.url), { workerData: { memory, base, size, plans, counts, rounds, tokens, prompts, stops, passages, wide, halfKeys, id } });
  const decode = (ids) => {
    py.globals.set("IDS", py.toPy(ids));
    return py.runPython(`
prev, out = llama.bos, b""
for token in IDS:
    out += llama.tokenizer.decode(prev, token, llama.bos)
    prev = token
out.decode("utf-8", errors="replace")`);
  };
  await new Promise((resolve, reject) => {
    worker.on("error", reject);
    worker.on("message", (message) => {
      if (message.line) console.log(`T246 ${id}: ${message.line}`);
      if (message.answer) console.log(`T246 ${id}: wrote for ${JSON.stringify(PROMPTS[message.prompt])} (${message.note}): ${JSON.stringify(decode(message.answer))}`);
      if (message.done) resolve();
    });
  });
  await worker.terminate();
  process.exit(0);
} else {
  const { memory, base, size, plans, counts, rounds, tokens, prompts, stops, passages, wide, halfKeys, id } = workerData;
  const say = (line) => parentPort.postMessage({ line });
  const suffix = wide ? "64" : "";
  const kernels = compileKernels(fs.readFileSync(`${root}public/simdkernel_shared${suffix}.wasm`), fs.readFileSync(`${root}public/simdkernel_relaxed_shared${suffix}.wasm`), wide);
  const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
  const argmax = (logits) => { let best = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i; return best; };
  // a prompt as the page feeds it (blocks of 16, the last token alone for its logits), then greedy tokens
  const feed = (engine, ids) => {
    for (let at = 0; at < ids.length - 1; at += 16) engine.forwardMany(ids.slice(at, Math.min(at + 16, ids.length - 1)), at);
    engine.forward(ids[ids.length - 1], ids.length - 1, true);
  };
  const names = Object.keys(plans);
  let engine = createForward({ memory, base, size, kernels, plan: plans[names[0]], spawn, halfKeys });
  say(`backend ${engine.backend}`);
  // 1. the speeds, the counts in turn
  const short = Math.min(16, tokens), ids = prompts[0];
  const prompt = Object.fromEntries(counts.map((n) => [n, []])), written = Object.fromEntries(counts.map((n) => [n, []]));
  let reference, differ = [];
  for (let round = 0; round < rounds; round++) {
    for (const n of counts) {
      await engine.setThreads(n);
      let began = performance.now();
      feed(engine, ids);
      prompt[n].push((ids.length * 1000) / (performance.now() - began));
      let token = argmax(engine.logits());
      began = performance.now();
      for (let pos = ids.length; pos < ids.length + short; pos++) {
        engine.forward(token, pos, true);
        token = argmax(engine.logits());
      }
      written[n].push((short * 1000) / (performance.now() - began));
      const last = engine.logits().slice();
      if (!reference) reference = last;
      else if (!last.every((v, j) => Object.is(v, reference[j]))) differ.push(n);
    }
  }
  say(`tokens/s by threads (median of ${rounds}, the first prompt's ${ids.length} tokens in blocks of 16, then ${short} tokens written at positions ${ids.length} to ${ids.length + short - 1}): ` +
    counts.map((n) => `${n} thread(s): prompt ${median(prompt[n]).toFixed(2)}, written ${median(written[n]).toFixed(2)} (all ${written[n].map((v) => v.toFixed(2)).join(" ")})`).join("; ") +
    `; the last logits with every count ${differ.length ? `DIFFER (${differ})` : "the same to the bit"}`);
  const fastest = counts.reduce((a, b) => (median(written[b]) > median(written[a]) ? b : a));
  await engine.setThreads(fastest);
  // 2. greedy answers
  prompts.forEach((ids, index) => {
    feed(engine, ids);
    const answer = [];
    const began = performance.now();
    for (let pos = ids.length; answer.length < tokens; pos++) {
      const token = argmax(engine.logits());
      if (stops.includes(token)) break;
      answer.push(token);
      engine.forward(token, pos, true);
    }
    const seconds = (performance.now() - began) / 1000;
    parentPort.postMessage({ answer, prompt: index, note: `${answer.length} tokens${answer.length < tokens ? ", then a stop token" : ""}, ${fastest} threads, ${(answer.length / seconds).toFixed(2)} tok/s` });
  });
  // 3. the perplexity under each reading of the RoPE
  for (const name of names) {
    if (name !== names[0]) {
      engine.stopThreads();
      engine = createForward({ memory, base, size, kernels, plan: plans[name], spawn, halfKeys });
      await engine.setThreads(fastest);
    }
    for (const passage of passages) {
      const began = performance.now();
      let total = 0;
      for (let pos = 0; pos < passage.ids.length - 1; pos++) {
        engine.forward(passage.ids[pos], pos, true);
        const logits = engine.logits();
        let most = -Infinity;
        for (let i = 0; i < logits.length; i++) if (logits[i] > most) most = logits[i];
        let sum = 0;
        for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - most);
        total -= logits[passage.ids[pos + 1]] - most - Math.log(sum);
      }
      const count = passage.ids.length - 1;
      say(`PPLENGINE ${passage.name} ${name}: ${Math.exp(total / count).toFixed(4)} over ${count} targets (${fastest} threads, ${((performance.now() - began) / 1000).toFixed(0)} s)`);
    }
  }
  engine.stopThreads();
  parentPort.postMessage({ done: true });
}
