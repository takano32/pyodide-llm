// gpu-hybrid-check.mjs (T229's review): a Qwen3.5 (hybrid attention) is not put on the GPU, whatever adapter the browser has.
// forward/gpuside.js's gpuUnfit() says so in its first line, and nothing ran that line: CI has no real adapter (a fallback one is
// refused before it), and the made-up GPU of gpu-default-check.mjs is given llm-jp-3's Llama layers. Without that line a
// visitor with a GPU would have a linear-attention layer's matrices read as a full-attention layer's (a Qwen3.5 has no wq,
// wk, wv in a Gated DeltaNet layer), and the owner's PC and Android are such visitors once T236 lists the model.
// A model that goes to the GPU alone (T156) is refused by gpuOnlyUnfit()'s arch, which gpu-choice-check.mjs holds.
// T260: an LFM2 the same (its convolution layers have no wq, wk, wv either, and no shader), by the next line of gpuUnfit().
// T545: an LFM2's prompts are the GPU's now: its worker is started once, with a plan that says every layer's kind and
// its place among the layers of its kind, the taps and where the CPU keeps the convolution layers' rows, the matrices
// in and out and the attending layers' matrices as theirs alone (of); T545.2: and its steps (the plan's tokens).
// T261: and a Gemma 3, whose tensors are a Qwen3's to the GPU's worker (it would run them as one: no norm behind the
// attention or the FFN, a SiLU in the gate, every position attended), by gpuUnfit()'s line for it. T261.4: a Gemma 3's
// prompts are the GPU's now: its worker is started, and the plan it is started with says what the GPU's worker runs
// differently for it (GEGLU, the norms behind, every layer's window, the local layers and their tables); T261.4.2: and
// its steps (the plan's tokens).
// T342: and a Spark-X2.5, whose gate of the heads and exact GELU no shader has, by the same line (its FFN's activation).
//
//   node tests/gpu-hybrid-check.mjs <a made-up Qwen3.5 of tests/make_qwen35.py, its int8 file without the extension>
//   node tests/gpu-hybrid-check.mjs <a made-up LFM2 of tests/make_lfm2.py, the same>
//   node tests/gpu-hybrid-check.mjs <a made-up Gemma 3 of tests/make_gemma3.py, the same>
//   node tests/gpu-hybrid-check.mjs <a made-up Spark-X2.5 of tests/make_spark.py, the same>
//     [--forward <another forward.js, to see a broken one fail>]
// What it sees: the factory of the GPU's worker is never called, and the engine says why. A made-up GPU that never answers
// stands in for the real one: with the line gone, the factory is called and the check fails there.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { loadPyodide } from "pyodide";
import { workers, ended } from "./fakes/worker.mjs";
import { built, placeKernels, placePython, runtime, runtimeUrl, treeOf } from "./tree.mjs";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const prefix = args.find((arg, i) => !arg.startsWith("--") && args[i - 1] !== "--forward");
if (!prefix) {
  console.error("usage: node tests/gpu-hybrid-check.mjs <a made-up Qwen3.5, without .bin> [--forward <forward.js>]");
  process.exit(2);
}
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : runtime("forward.js");
const { compileKernels, external, weightsMemory } = await import(forwardFile);

const options = JSON.parse(fs.readFileSync(`${prefix}.json`, "utf8"));
// the kind of layer that keeps it off the GPU, as forward/gpuside.js's gpuUnfit() says it
// (T261: a Gemma 3 by its local layers' option, though any of its facts keeps it off: the GELU in its gate and the norms
// behind as well, which are no option of a model's)
const REASONS = { qwen35: ["a Qwen3.5", "linear", /linear-attention layers are not on the GPU/],
  lfm2: ["an LFM2", "convolution", null],
  gemma3: ["a Gemma 3", "local", null],
  spark: ["a Spark-X2.5", "local", /this pair of norms and FFN is not on the GPU/] };
assert.ok(options.arch in REASONS, `${prefix} is no Qwen3.5, no LFM2, no Gemma 3 and no Spark-X2.5`);
const [model, layers, reason] = REASONS[options.arch];
assert.ok(options[layers], `${prefix} has no ${layers} layers`);
const checkpoint = fs.readFileSync(`${prefix}.bin`);

const py = await loadPyodide();
await py.loadPackage("numpy", { messageCallback: () => {} });
placePython(py, treeOf());
placeKernels(py, treeOf());
py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${prefix}.tokenizer.bin`));
const kernels = compileKernels(fs.readFileSync(built("simdkernel_shared.wasm")), fs.readFileSync(built("simdkernel_relaxed_shared.wasm")));
const { memory, base } = weightsMemory(checkpoint.length, { shared: true });
new Uint8Array(memory.buffer, base, checkpoint.length).set(checkpoint);

// a GPU's worker that is asked for and never answers (but that it ended, once stopped)
const { factory: gpu, made: gpus } = workers({ answer: ended });  // (T261.4: one that is started ends when it is stopped)
const outside = external({ memory, base, size: checkpoint.length, kernels, gpu });
py.globals.set("OUTSIDE", outside);
py.globals.set("OPTIONS", py.toPy(options));
py.runPython(`from llama2_numpy import Llama
llama = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
// (a GPU that was started would never say it is ready: do not wait for it for ever)
const status = await Promise.race([outside.engine.gpu, new Promise((resolve) => setTimeout(() => resolve("no answer in 3 s"), 3000))]);
const failures = [];
if (!reason && options.arch === "lfm2") {
  // (T545: started once, with the plan of an LFM2)
  const { plan } = gpus[0]?.posted.find((data) => data.type === "start") ?? {};
  const letters = [...options.convolution.layers], places = letters.map((letter, l) => letters.slice(0, l).filter((x) => x === letter).length);
  const count = (letter) => letters.filter((x) => x === letter).length;
  if (gpus.length !== 1 || !plan) failures.push(`the GPU's worker was started ${gpus.length} time(s) for ${model}, with ${plan ? "a plan" : "no plan"}`);
  else {
    const wanted = { kinds: letters.map((letter) => (letter === "c" ? "convolution" : "attention")), places, taps: options.convolution.taps, rows: true,
      matrices: { wq: ["attention", count("a")], wk: ["attention", count("a")], wv: ["attention", count("a")], wo: ["attention", count("a")],
        w1: [null, letters.length], w2: [null, letters.length], w3: [null, letters.length], win: ["convolution", count("c")], wout: ["convolution", count("c")] },
      conv: ["convolution", options.convolution.taps * plan.dim], tokens: true };
    const got = { kinds: plan.kinds, places: plan.places, taps: plan.convolution?.taps, rows: plan.convolution?.rows > 0,
      matrices: Object.fromEntries(Object.entries(plan.matrices).map(([name, m]) => [name, [m.of ?? null, m.layers.length]])),
      conv: [plan.vectors.conv?.of, plan.vectors.conv?.size], tokens: Boolean(plan.tokens) };
    if (JSON.stringify(got) !== JSON.stringify(wanted)) failures.push(`the plan says ${JSON.stringify(got)}, where ${JSON.stringify(wanted)} is to be`);
  }
} else if (!reason) {
  // (T261.4: started once, with the plan of a Gemma 3)
  const { plan } = gpus[0]?.posted.find((data) => data.type === "start") ?? {};
  const { layers: local, window } = options.local, count = plan?.layers;
  if (gpus.length !== 1 || !plan) failures.push(`the GPU's worker was started ${gpus.length} time(s) for ${model}, with ${plan ? "a plan" : "no plan"}`);
  else {
    const wanted = { activation: "gelu", windows: Array.from({ length: count }, (_, l) => (local.includes(l) ? window : 0)), local: [...local].sort((a, b) => a - b),
      posts: ["attentionPost", "ffnPost"], tables: true, tokens: true };
    const got = { activation: plan.activation, windows: plan.windows, local: plan.local, posts: Object.keys(plan.vectors).filter((name) => name.endsWith("Post")),
      tables: plan.localCos > 0 && plan.localSin > 0 && plan.localCos !== plan.cos, tokens: Boolean(plan.tokens) };
    if (JSON.stringify(got) !== JSON.stringify(wanted)) failures.push(`the plan says ${JSON.stringify(got)}, where ${JSON.stringify(wanted)} is to be`);
  }
} else {
  if (gpus.length !== 0) failures.push(`the GPU's worker was started ${gpus.length} time(s) for ${model}`);
  if (!reason.test(status ?? "")) failures.push(`the status line says "${status}", not why the GPU is not used`);
  if (!reason.test(outside.engine.gpuStatus ?? "")) failures.push(`gpuStatus is "${outside.engine.gpuStatus}"`);
}
await outside.engine.release();
if (failures.length) {
  console.error(`FAILED\n- ${failures.join("\n- ")}`);
  process.exit(1);
}
console.log(reason ? `ok: ${model} with a GPU adapter present stays on the CPU, its GPU's worker never started (${status})`
  : options.arch === "lfm2" ? `ok: ${model} with a GPU adapter present starts the GPU's worker once, with a plan of its layers' kinds and places, its taps and rows, and its steps`

  : `ok: ${model} with a GPU adapter present starts the GPU's worker once, with a plan of its window, its local layers' tables, its norms behind and GEGLU, and its steps`);
process.exit(0);
