// tests/bench.ts's part (T643.13): the GPU section's steps, asked and kept by their keys (src/bench/gpusection.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { gpuPlan, gpuKept } from "../../src/bench.ts";

// T643.12: the steps the page asks after the adapter's, counted before the first (T192: by hand, the last two read "N / N"):
// the check alone on a fallback adapter (T182), and otherwise every step, a token of each model twice where the packed
// int8 dot is; and each kept by its key, the bandwidths and the tokens in the order asked
const MODELS2 = ["llm-jp-3 150M", "Llama 3.2 1B"];
assert.deepEqual(gpuPlan({ packed: true, fallback: true }, MODELS2), [{ name: "the shaders against JavaScript", message: { step: "check" } }]);
const planned = gpuPlan({ packed: true, fallback: false }, MODELS2);
assert.deepEqual(planned.map(({ name }) => name), ["the shaders against JavaScript", "bandwidth: llm-jp-3 150M w1", "bandwidth: Llama 3.2 1B w1", "bandwidth: Llama 3.2 1B classifier",
  "a token of llm-jp-3 150M", "a token of llm-jp-3 150M, packed int8", "a token of Llama 3.2 1B", "a token of Llama 3.2 1B, packed int8", "a token of Llama 3.2 1B, chosen on the GPU",
  "a layer of a token", "the steps of a layer", "tokens generated on the GPU", "what a token costs besides the weights", "the device's ceilings", "a prompt all at once"]);
assert.deepEqual(planned.map(({ message }) => message.step), ["check", "bandwidth", "bandwidth", "bandwidth", "token", "token", "token", "token", "token",
  "layer", "layer steps", "generate", "overhead", "ceilings", "prompt"]);
assert.deepEqual(planned.filter(({ message }) => message.step === "token").map(({ message }) => [message.model, message.kind, message.sample]),
  [["llm-jp-3 150M", "widen", undefined], ["llm-jp-3 150M", "packed", undefined], ["Llama 3.2 1B", "widen", undefined], ["Llama 3.2 1B", "packed", undefined], ["Llama 3.2 1B", "widen", true]]);
assert.deepEqual(planned[1].message, { step: "bandwidth", shape: [2048, 512] });
assert.deepEqual(planned.at(-1).message, { step: "prompt", counts: [1, 16, 64] });
assert.equal(gpuPlan({ packed: false, fallback: false }, [...MODELS2, "Llama 3.2 3B"]).length, 15 - 2 + 1, "without the packed dot a token of each model once, and the 3B where it is asked for");
// a step kept by its key in its place is kept once: no other step of the plan has its key (gpuKept() would write over it)
const onceKeys = planned.map(({ message }) => message.step).filter((key) => key !== "bandwidth" && key !== "token");
assert.equal(new Set(onceKeys).size, onceKeys.length, `a step asked twice is kept once: ${onceKeys}`);
const keptFound = { bandwidth: [], token: [] };
for (const [i, { name, message }] of planned.entries()) gpuKept(keptFound, message.step, { name, result: i });
assert.deepEqual(keptFound.bandwidth.map(({ result }) => result), [1, 2, 3]);
assert.deepEqual(keptFound.token.map(({ name }) => name), planned.slice(4, 9).map(({ name }) => name));
assert.deepEqual(Object.keys(keptFound), ["bandwidth", "token", "check", "layer", "layer steps", "generate", "overhead", "ceilings", "prompt"]);
assert.deepEqual([keptFound.check, keptFound.prompt], [{ name: "the shaders against JavaScript", result: 0 }, { name: "a prompt all at once", result: 14 }]);
