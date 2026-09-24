// T93 stage 2: the forward pass of forward.js with helper threads on a shared memory. The forward pass runs in a
// worker, as on the page (the model's worker coordinates); run from the main thread next to Pyodide it waited on
// its helpers several times longer. Python in the main thread only makes the plan.
//   1. Every row is computed whole by one thread with the same kernel, so the logits of a greedy run must be the
//      same to the bit with any number of threads.
//   2. The speeds, the counts in turn, several rounds, the median.
//   3. The search (stage 2b): which count it chooses from navigator.hardwareConcurrency, within a few generations,
//      against the fastest of 2.
//   4. T108: the same text as a prompt in blocks (forwardMany): the last logits the same to the bit with every
//      count, and what a token of a prompt costs in blocks of 1, 2, 4, 8 and 16, with each count.
//   5. T100: the same text in drafts of k tokens with the logits of each (forwardEach): every token's logits the same
//      to the bit, and what one such pass costs against one token's, for k = 1, 2, 4, 8 and 16.
//
//   node tests/threads-check.mjs [model id ...] [--threads 1,2,4,8] [--rounds 5] [--positions 64] [--kv-start 16]
//        [--from 0]
//
// --from: the speeds of step 2 are measured from this position on, the KV cache filled up to it first (T109: the
// attention of a long context).
//
// --kv-start: the KV cache starts this small (the page's KV_START is 256), so that it has to grow, and move, under
// the helper threads within the positions of a run (Fable's review of T93).
import fs from "node:fs";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { compileKernels, createForward, weightsMemory } from "../public/forward.js";

const root = new URL("../", import.meta.url).pathname;
// a helper thread: resolves once it has its kernels on the shared memory
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
  const counts = option("--threads", "1,2,4,8").split(",").map(Number);
  const rounds = Number(option("--rounds", 5)), positions = Number(option("--positions", 64)), kvStart = Number(option("--kv-start", 16));
  const from = Number(option("--from", 0));
  const ids = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
  const { pyodide: py } = await pyodideWithEngine();
  let failed = false;
  for (const id of ids.length ? ids : ["tiny-lm", "llm-jp-3-150m"]) {
    const entry = MODELS.find((m) => m.id === id);
    const checkpoint = fs.readFileSync(root + entry.checkpoint);
    const { memory, base } = weightsMemory(checkpoint.length, { shared: true });
    new Uint8Array(memory.buffer).set(checkpoint, base);
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
    let plan;
    // Python says where every tensor is; the forward pass itself is made in the worker
    const outside = { size: checkpoint.length, read: (o, l) => new Uint8Array(memory.buffer, base + o, l).slice(),
      start: (p) => { plan = p.toJs({ dict_converter: Object.fromEntries }); return { backend: "", bind() {}, forward() {}, release() {} }; } };
    py.globals.set("OUTSIDE", outside);
    py.globals.set("OPTIONS", py.toPy(entry.options));
    py.runPython(`import llama2_numpy\nfrom llama2_numpy import Llama\nllama2_numpy.KV_START = ${kvStart}\nLlama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
    const worker = new Worker(new URL(import.meta.url), { workerData: { memory, base, size: checkpoint.length, plan, counts, rounds, positions, from } });
    const result = await new Promise((resolve, reject) => { worker.once("message", resolve); worker.once("error", reject); });
    console.log(`${entry.name}: ${result}`);
    failed ||= result.includes("DIFFER");
    await worker.terminate();
  }
  process.exit(failed ? 1 : 0);
} else {
  const { memory, base, size, plan, counts, rounds, positions, from } = workerData;
  const kernels = compileKernels(fs.readFileSync(`${root}public/simdkernel_shared.wasm`), fs.readFileSync(`${root}public/simdkernel_relaxed_shared.wasm`));
  const engine = createForward({ memory, base, size, kernels, plan, spawn });
  const greedy = () => {
    const seen = [];
    let token = plan.bos ?? 1;
    for (let pos = 0; pos < positions; pos++) {
      engine.forward(token, pos, true);
      const logits = engine.logits();
      seen.push(logits.slice());
      token = logits.indexOf(Math.max(...logits));
    }
    return seen;
  };
  let reference, differ = [];
  for (const n of counts) {
    await engine.setThreads(n);
    const seen = greedy();
    if (!reference) reference = seen;
    else if (!seen.every((row, i) => row.every((v, j) => Object.is(v, reference[i][j])))) differ.push(n);
  }
  // T108: the greedy text fed back as a prompt, in blocks
  const text = [plan.bos ?? 1, ...reference.slice(0, -1).map((logits) => logits.indexOf(Math.max(...logits)))];
  const blocksDiffer = [];
  for (const n of counts) {
    await engine.setThreads(n);
    for (let at = 0; at < positions - 1; at += 5) engine.forwardMany(text.slice(at, Math.min(at + 5, positions - 1)), at);
    engine.forward(text[positions - 1], positions - 1, true);
    if (!engine.logits().every((v, j) => Object.is(v, reference[positions - 1][j]))) blocksDiffer.push(n);
  }
  // T100: drafts of 3 tokens (a size that does not divide the text), every token's logits against the reference
  const draftsDiffer = [];
  for (const n of counts) {
    await engine.setThreads(n);
    for (let at = 0; at < positions; at += 3) {
      const draft = text.slice(at, Math.min(at + 3, positions));
      engine.forwardEach(draft, at);
      if (!draft.every((_, i) => engine.logits(i).every((v, j) => Object.is(v, reference[at + i][j])))) {
        draftsDiffer.push(n);
        break;
      }
    }
  }
  const blockSizes = [1, 2, 4, 8, 16], prompt = Object.fromEntries(counts.map((n) => [n, Object.fromEntries(blockSizes.map((k) => [k, []]))]));
  for (let r = 0; r < rounds; r++) {
    for (const n of counts) {
      await engine.setThreads(n);
      for (const k of blockSizes) {
        const began = performance.now();
        for (let at = 0; at < positions; at += k) engine.forwardMany(text.slice(at, Math.min(at + k, positions)).map((t) => t ?? 1), at);
        prompt[n][k].push((positions * 1000) / (performance.now() - began));
      }
    }
  }
  // what one pass over a draft of k tokens costs, with the logits of each, in milliseconds
  const pass = Object.fromEntries(counts.map((n) => [n, Object.fromEntries(blockSizes.map((k) => [k, []]))]));
  for (let r = 0; r < rounds; r++) {
    for (const n of counts) {
      await engine.setThreads(n);
      for (const k of blockSizes) {
        const passes = Math.floor(positions / k), began = performance.now();
        for (let p = 0; p < passes; p++) engine.forwardEach(text.slice(p * k, (p + 1) * k).map((t) => t ?? 1), p * k);
        pass[n][k].push((performance.now() - began) / passes);
      }
    }
  }
  const times = Object.fromEntries(counts.map((n) => [n, []]));
  for (let r = 0; r < rounds; r++) {
    for (const n of counts) {
      await engine.setThreads(n);
      if (from) engine.forwardMany(new Array(from).fill(1), 0);  // the KV cache up to where the timing starts
      for (let pos = from; pos < from + 8; pos++) engine.forward(1, pos, true);  // the helpers wake up
      const began = performance.now();
      for (let pos = from; pos < from + positions; pos++) engine.forward(1, pos, true);
      times[n].push((positions * 1000) / (performance.now() - began));
    }
  }
  const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
  const one = median(times[counts[0]]);
  // the search, as the page runs it: from the hint, over generations of `positions` tokens
  const hint = globalThis.navigator?.hardwareConcurrency ?? 4;
  let found = 0, generationsUsed = 0;
  await engine.findThreads({ from: hint, chose: (n) => { found = n; } });
  for (; !found && generationsUsed < 8; generationsUsed++) {
    engine.newGeneration();
    for (let pos = 0; pos < positions; pos++) engine.forward(1, pos, true);
    await new Promise((resolve) => setTimeout(resolve, 0));  // let a helper that is being started come up
  }
  const fastest = counts.reduce((a, b) => (median(times[b]) > median(times[a]) ? b : a));
  const comparisons = engine.searchLog.map(({ best, candidate, times, faster }) =>
    `${best} [${times[best].map((t) => t.toFixed(2)).join(" ")}] vs ${candidate} [${times[candidate].map((t) => t.toFixed(2)).join(" ")}] ${faster ? "->" : "stay"}`).join("; ");
  const searchLine = `the search from ${hint} chose ${found || "nothing"} in ${generationsUsed} generation(s) (${comparisons}); the fastest measured was ${fastest}` +
    (found && times[found] ? ` (${found} runs at ${(median(times[found]) / median(times[fastest]) * 100).toFixed(0)}% of it)` : "");
  const promptLine = "a prompt in blocks of " + blockSizes.join(", ") + ": " + counts.map((n) =>
    `${n} thread(s) ${blockSizes.map((k) => median(prompt[n][k]).toFixed(0)).join(" / ")} tok/s (${(median(prompt[n][16]) / median(prompt[n][1])).toFixed(2)}×)`).join(", ");
  const draftLine = "a draft of " + blockSizes.join(", ") + " tokens with every token's logits: " + counts.map((n) =>
    `${n} thread(s) ${blockSizes.map((k) => median(pass[n][k]).toFixed(1)).join(" / ")} ms per pass (` +
    blockSizes.map((k) => (median(pass[n][k]) / median(pass[n][1])).toFixed(2)).join(" / ") + " times one token's)").join(", ");
  parentPort.postMessage(`${engine.backend}: ${differ.length ? `logits DIFFER with ${differ.join(", ")} threads` : `logits the same to the bit with ${counts.join(", ")} threads`}; ` +
    `${blocksDiffer.length ? `the prompt in blocks DIFFERS with ${blocksDiffer.join(", ")} threads` : "the prompt in blocks the same to the bit"}; ` +
    `${draftsDiffer.length ? `drafts DIFFER with ${draftsDiffer.join(", ")} threads` : "drafts the same to the bit"}; ` +
    counts.map((n) => `${n}: ${median(times[n]).toFixed(1)} tok/s (${(median(times[n]) / one).toFixed(2)}×)`).join(", ") + `; ${promptLine}; ${draftLine}; ${searchLine}`);
  engine.stopThreads();
}
