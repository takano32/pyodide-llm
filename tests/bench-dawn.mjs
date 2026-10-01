// A step of /benchmark/'s GPU worker (public/benchmark/gpu.js) in Node on Dawn (the npm package webgpu, not a
// dependency of this project: gpu-prompt.yml installs it under .tmp/dawn) with Mesa's lavapipe (shader-f16 and
// subgroups, whose width LP_NATIVE_VECTOR_WIDTH sets: 128 → 4, 512 → 16), with no browser: the worker's answer as one
// line "RESULT <seconds> <json>" (its result, or { error }). T219 (from T191's review's throwaway tool): the CI's
// tests.yml runs it with extra= on a branch, e.g. the step "check" for the sampling's verdicts.
//   VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1) \
//   node tests/bench-dawn.mjs <the webgpu package's directory> [<a public directory>] [<step>] [<counts>] [<keys>]
// step: as the page asks the worker ("info", "check", "prompt", "generate", ...); counts: the prompt's, as 1,16,64;
// keys: only these of the result, as "sampling,sampling in chunks" (the check's verdicts are long).
import path from "node:path";
import { pathToFileURL } from "node:url";

const [webgpu, dir = "public", step = "check", counts = "16", keys = ""] = process.argv.slice(2);
if (!webgpu) {
  console.error("node tests/bench-dawn.mjs <the webgpu package's directory> [<a public directory>] [<step>] [<counts>] [<keys>]");
  process.exit(2);
}
const { create, globals } = await import(pathToFileURL(path.resolve(webgpu, "index.js")).href);
Object.assign(globalThis, globals);
globalThis.self = globalThis;
globalThis.onmessage = null;
// (T225's review: GPU_ROUNDING=toward-zero|away|everything gives the device's shaders another rounding of a float32 to a
// float16 than lavapipe's own, the nearest: tests/rounding.mjs, tests/rounding-check.mjs)
const { roundedGpu } = await import("./rounding.mjs");
Object.defineProperty(globalThis, "navigator", { value: { gpu: roundedGpu(create([]), process.env.GPU_ROUNDING) }, configurable: true });
let answer;
globalThis.postMessage = (message) => {
  if (message.lost) console.log("LOST", message.lost);
  // (the worker's progress: alive, lost and stage are no answer)
  if (!message.alive && !message.lost && !message.stage) answer?.(message);
};
await import(pathToFileURL(path.resolve(dir, "benchmark/gpu.js")).href);
const began = performance.now();
const reply = await new Promise((resolve) => {
  answer = resolve;
  globalThis.onmessage({ data: { step, counts: counts.split(",").map(Number) } });
});
const result = reply.result ?? { error: reply.error };
const shown = keys && !result.error ? Object.fromEntries(keys.split(",").map((key) => [key, result[key]])) : result;
console.log("RESULT", ((performance.now() - began) / 1000).toFixed(1), JSON.stringify(shown));
process.exit(0);
