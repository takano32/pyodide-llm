// gpu-hybrid-check.mjs (T229's review): a Qwen3.5 (hybrid attention) is not put on the GPU, whatever adapter the browser has.
// forward/gpuside.js's gpuUnfit() says so in its first line, and nothing ran that line: CI has no real adapter (a fallback one is
// refused before it), and the made-up GPU of gpu-default-check.mjs is given llm-jp-3's Llama layers. Without that line a
// visitor with a GPU would have a linear-attention layer's matrices read as a full-attention layer's (a Qwen3.5 has no wq,
// wk, wv in a Gated DeltaNet layer), and the owner's PC and Android are such visitors once T236 lists the model.
// A model that goes to the GPU alone (T156) is refused by gpuOnlyUnfit()'s arch, which gpu-choice-check.mjs holds.
// T260: an LFM2 the same (its convolution layers have no wq, wk, wv either, and no shader), by the next line of gpuUnfit().
//
//   node tests/gpu-hybrid-check.mjs <a made-up Qwen3.5 of tests/make_qwen35.py, its int8 file without the extension>
//   node tests/gpu-hybrid-check.mjs <a made-up LFM2 of tests/make_lfm2.py, the same>
//     [--forward <another forward.js, to see a broken one fail>]
// What it sees: the factory of the GPU's worker is never called, and the engine says why. A made-up GPU that never answers
// stands in for the real one: with the line gone, the factory is called and the check fails there.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { loadPyodide } from "pyodide";
import { PYTHON, placeFile } from "../public/python.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const prefix = args.find((arg, i) => !arg.startsWith("--") && args[i - 1] !== "--forward");
if (!prefix) {
  console.error("usage: node tests/gpu-hybrid-check.mjs <a made-up Qwen3.5, without .bin> [--forward <forward.js>]");
  process.exit(2);
}
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : path.join(root, "public", "forward.js");
const { compileKernels, external, weightsMemory } = await import(forwardFile);

const options = JSON.parse(fs.readFileSync(`${prefix}.json`, "utf8"));
// the kind of layer that keeps it off the GPU, as forward/gpuside.js's gpuUnfit() says it
const REASONS = { qwen35: ["a Qwen3.5", "linear", /linear-attention layers are not on the GPU/],
  lfm2: ["an LFM2", "convolution", /convolution layers are not on the GPU/] };
assert.ok(options.arch in REASONS, `${prefix} is neither a Qwen3.5 nor an LFM2`);
const [model, layers, reason] = REASONS[options.arch];
assert.ok(options[layers], `${prefix} has no ${layers} layers`);
const checkpoint = fs.readFileSync(`${prefix}.bin`);

const py = await loadPyodide();
await py.loadPackage("numpy", { messageCallback: () => {} });
for (const name of [...PYTHON.llama2_numpy, ...PYTHON.llama2_convert, "simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
  placeFile(py, name, fs.readFileSync(path.join(root, "public", name)));
}
py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${prefix}.tokenizer.bin`));
const kernels = compileKernels(fs.readFileSync(path.join(root, "public/simdkernel_shared.wasm")), fs.readFileSync(path.join(root, "public/simdkernel_relaxed_shared.wasm")));
const { memory, base } = weightsMemory(checkpoint.length, { shared: true });
new Uint8Array(memory.buffer, base, checkpoint.length).set(checkpoint);

// a GPU's worker that is asked for and never answers
let spawned = 0;
const gpu = () => {
  spawned++;
  return { postMessage() {}, set onmessage(handler) {}, set onerror(handler) {}, terminate() {} };
};
const outside = external({ memory, base, size: checkpoint.length, kernels, gpu });
py.globals.set("OUTSIDE", outside);
py.globals.set("OPTIONS", py.toPy(options));
py.runPython(`from llama2_numpy import Llama
llama = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
// (a GPU that was started would never say it is ready: do not wait for it for ever)
const status = await Promise.race([outside.engine.gpu, new Promise((resolve) => setTimeout(() => resolve("no answer in 3 s"), 3000))]);
const failures = [];
if (spawned !== 0) failures.push(`the GPU's worker was started ${spawned} time(s) for ${model}`);
if (!reason.test(status ?? "")) failures.push(`the status line says "${status}", not why the GPU is not used`);
if (!reason.test(outside.engine.gpuStatus ?? "")) failures.push(`gpuStatus is "${outside.engine.gpuStatus}"`);
await outside.engine.release();
if (failures.length) {
  console.error(`FAILED\n- ${failures.join("\n- ")}`);
  process.exit(1);
}
console.log(`ok: ${model} with a GPU adapter present stays on the CPU, its GPU's worker never started (${status})`);
process.exit(0);
