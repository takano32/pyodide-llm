// tests/threads-compare.ts (T.378.1, T.447): two builds' forward pass WITH software threads, in one process, turn by turn.
// tests/compare-engines.ts measures the page's path with no software thread; a change of the threads' part
// (src/runtime/forward/threads.js), of helper.js or of jobs.js is not seen there. Here each side is a build's
// _astro/forward.<hash>.js and its own _astro/helper.<hash>.js (what a browser runs: tests/bundle.ts), each on a shared
// memory of its own with the same checkpoint, both made once and kept: the forward pass runs in a worker, as on the
// page, and its threads sleep while the other side's turn is on. This tree's Python makes the plan for both.
//
// Before timing, the two must compute the same logits to the bit with every count (exit 1 otherwise). Then, a count of
// threads at a time: a round is old new new old, the next new old old new; a run is `positions` tokens with logits
// from position 0 after 8 that wake the threads. Each side is measured twice a round: the two of one side say how far
// a measurement moves by itself, and the control (the same code built twice, on the two sides) what the tool says of
// no difference: read a ratio against both.
// T.620: --prompt times a prompt instead: `positions` tokens from position 0 in blocks of the engine's promptBlock
// (forwardMany, no logits), as Python feeds one; the logits held to the bit are then those of the token after a prompt
// as well. A model is an id of the list, or the files of one (<out>.bin, .tokenizer.bin, .json: tests/make_ternary.py,
// tests/perplexity_prepare.py, quantize.py with that .json beside it): the ternary and six-bit dtypes have no model of
// the site's own.
//
//   node tests/threads-compare.ts [models = tiny-lm llm-jp-3-150m] --old <a dist> [--new <a dist> = dist]
//                                  [--threads 2,4] [--rounds 21] [--positions 128] [--prompt]
//
// tests/compare-builds.sh runs it after compare-engines where THREADS is set (THREADS=2,4), the control beside it.
// The speed of a development machine decides nothing: CI's runners, x86-64 and arm64, the CPU's name with the numbers.
// On a runner of 4 logical cores, 4 threads share the cores with whatever else runs there: its spread is wider.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { builtModule } from "./bundle.ts";
import { softwareThread } from "./helper-node.ts";
import { built } from "./tree.ts";

const root = new URL("../", import.meta.url).pathname;
const median = (values) => { const s = [...values].sort((a, b) => a - b); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const range = (values, digits = 1) => `${Math.min(...values).toFixed(digits)}–${Math.max(...values).toFixed(digits)}`;

if (isMainThread) {
  const { pyodideWithEngine } = await import("./engine.ts");
  const { MODELS } = await import("../src/models.ts");
  const { leave } = await import("./leave.ts");
  const args = process.argv.slice(2);
  const option = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
  const dists = { old: option("--old"), new: option("--new", path.join(root, "dist")) };
  if (!dists.old) throw new Error("threads-compare: --old <the dist of the build to compare with>");
  const counts = option("--threads", "2,4").split(",").map(Number);
  const rounds = Number(option("--rounds", 21)), positions = Number(option("--positions", 128)), prompt = args.includes("--prompt");
  const flags = ["--prompt"];
  const ids = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1]?.startsWith("--") && !flags.includes(args[i - 1])));
  const files = Object.fromEntries(Object.entries(dists).map(([side, dist]) => [side, { forward: String(builtModule("forward", dist)), helper: String(builtModule("helper", dist)) }]));
  const here = (url) => new URL(url).pathname.replace(root, "");
  console.log(`threads-compare: ${here(files.old.forward)} with ${here(files.old.helper)} (old) against ${here(files.new.forward)} with ${here(files.new.helper)} (new); ` +
    `${os.cpus()[0]?.model ?? "?"} (${process.arch}, ${os.cpus().length} logical cores), Node ${process.version}, load ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}`);
  console.log(`threads-compare: ${prompt ? `a prompt of ${positions} tokens` : `${positions} tokens with logits`} a run, ${rounds} rounds after a warm-up (old new new old, then new old old new), a memory and its threads a side`);
  const { pyodide: py } = await pyodideWithEngine();
  const rows = [];
  let failed = false;
  for (const id of ids.length ? ids : ["tiny-lm", "llm-jp-3-150m"]) {
    // (a model of the list, or the files of one, as tests/threads-check.ts takes them)
    const entry = MODELS.find((m) => m.id === id) ?? (fs.existsSync(`${id}.bin`) ? { name: path.basename(id), checkpoint: path.resolve(`${id}.bin`),
      tokenizer: path.resolve(`${id}.tokenizer.bin`), options: JSON.parse(fs.readFileSync(`${id}.json`, "utf8")) } : null);
    if (!entry || entry.hf) throw new Error(`${id}: a model of the site's own files (make models), or <out> of <out>.bin, .tokenizer.bin and .json`);
    const file = (name) => (path.isAbsolute(name) ? name : root + name);
    const checkpoint = fs.readFileSync(file(entry.checkpoint));
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(file(entry.tokenizer)));
    let plan;
    // Python says where every tensor is (threads-check.ts's way); the forward passes themselves are made in the worker
    py.globals.set("OUTSIDE", { size: checkpoint.length, read: (o, l) => new Uint8Array(checkpoint.buffer, checkpoint.byteOffset + o, l).slice(),
      start: (p) => { plan = p.toJs({ dict_converter: Object.fromEntries }); return { backend: "", bind() {}, forward() {}, release() {} }; } });
    py.globals.set("OPTIONS", py.toPy(entry.options ?? {}));
    py.runPython(`from llama2_numpy import Llama\nLlama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
    const worker = new Worker(new URL(import.meta.url), { workerData: { files, checkpoint, plan, counts, rounds, positions, prompt } });
    const result = await new Promise((resolve, reject) => { worker.once("message", resolve); worker.once("error", reject); });
    await worker.terminate();
    if (result.differ) {
      failed = true;
      console.log(`${id}: FAILED: ${result.differ}`);
      continue;
    }
    console.log(`${id}: ${result.backend}; the same logits to the bit from both with ${counts.join(", ")} threads`);
    for (const n of counts) {
      const { old, new: fresh, twice } = result.speeds[n];
      const paired = [];
      for (let i = 0; i < fresh.length; i += 2) paired.push((fresh[i] + fresh[i + 1]) / (old[i] + old[i + 1]));
      console.log(`${id}, ${n} threads: old ${old.map((s) => s.toFixed(1)).join(" ")}; new ${fresh.map((s) => s.toFixed(1)).join(" ")}`);
      rows.push(`| ${entry.name} | ${n} | ${median(old).toFixed(1)} | ${range(old)} | ${median(fresh).toFixed(1)} | ${range(fresh)} | ${(median(fresh) / median(old)).toFixed(3)} | ` +
        `${median(paired).toFixed(3)} | ${range(paired, 3)} | ${median(twice).toFixed(3)} | ${Math.max(...twice).toFixed(3)} |`);
    }
  }
  console.log("tok/s: the medians of the rounds' runs and their range; new ÷ old of the medians, and round by round (median, range);");
  console.log("one side twice in a round, the faster ÷ the slower (median, largest): what a measurement moves by itself");
  console.log("| model | threads | old | range | new | range | new ÷ old | by round | range | one side twice | largest |");
  console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const row of rows) console.log(row);
  console.log(`threads-compare: load after ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}${failed ? "; FAILED" : ""}`);
  await leave(failed ? 1 : 0);
} else if (workerData?.files) {
  const { files, checkpoint, plan, counts, rounds, positions, prompt } = workerData;
  const engines = {};
  for (const [side, { forward, helper }] of Object.entries(files)) {
    const { compileKernels, createForward, weightsMemory } = await import(forward);
    const kernels = compileKernels(fs.readFileSync(built("simdkernel_shared.wasm")), fs.readFileSync(built("simdkernel_relaxed_shared.wasm")));
    const { memory, base } = weightsMemory(checkpoint.length, { shared: true });
    new Uint8Array(memory.buffer, base, checkpoint.length).set(checkpoint);
    const spawn = (data) => new Promise((resolve) => {
      const thread = softwareThread(helper);
      thread.once("message", () => resolve({ terminate: () => thread.terminate() }));
      thread.postMessage(data);
    });
    engines[side] = createForward({ memory, base, size: checkpoint.length, kernels, plan, spawn });
  }
  // (a prompt's tokens: made up, the same every time)
  const tokens = Array.from({ length: positions }, (_, i) => (i ? 100 + ((i * 37) % 400) : plan.bos ?? 1));
  // a prompt of count tokens from position 0, in blocks as Python hands them over (no logits)
  const fed = (engine, count) => {
    for (let at = 0; at < count;) {
      const block = tokens.slice(at, Math.min(count, at + engine.promptBlock));
      engine.forwardMany(block, at);
      at += block.length;
    }
  };
  const greedy = (engine) => {
    const seen = [];
    let token = plan.bos ?? 1;
    for (let pos = 0; pos < 32; pos++) {
      engine.forward(token, pos, true);
      const logits = engine.logits();
      seen.push(logits.slice());
      token = logits.indexOf(Math.max(...logits));
    }
    // (T.620: and the logits of the token after a prompt, where the prompt's blocks are what is timed)
    if (prompt) {
      fed(engine, positions - 1);
      engine.forward(tokens[positions - 1], positions - 1, true);
      seen.push(engine.logits().slice());
    }
    return seen;
  };
  const run = (engine) => {
    for (let pos = 0; pos < 8; pos++) engine.forward(1, pos, true);  // the threads wake up
    const began = performance.now();
    if (prompt) fed(engine, positions);
    else for (let pos = 0; pos < positions; pos++) engine.forward(1, pos, true);
    return (positions * 1000) / (performance.now() - began);
  };
  const speeds = {};
  let differ = "";
  for (const n of counts) {
    for (const [side, engine] of Object.entries(engines)) {
      const got = await engine.setThreads(n);
      if (got !== n) differ ||= `${side} runs on ${got} threads where ${n} were asked for`;
    }
    const [a, b] = [greedy(engines.old), greedy(engines.new)];
    if (!a.every((row, i) => row.every((v, j) => Object.is(v, b[i][j])))) differ ||= `the two compute other logits with ${n} threads`;
    if (differ) break;
    const all = { old: [], new: [], twice: [] };
    for (let round = 0; round <= rounds; round++) {  // (the first is a warm-up)
      const measured = { old: [], new: [] };
      for (const side of round % 2 === 0 ? ["old", "new", "new", "old"] : ["new", "old", "old", "new"]) measured[side].push(run(engines[side]));
      if (!round) continue;
      for (const side of ["old", "new"]) {
        all[side].push(...measured[side]);
        all.twice.push(Math.max(...measured[side]) / Math.min(...measured[side]));
      }
    }
    speeds[n] = all;
  }
  const backend = engines.new.backend;
  for (const engine of Object.values(engines)) engine.stopThreads();
  parentPort.postMessage(differ ? { differ } : { speeds, backend });
}
