// T.135: the prompt on the GPU (src/runtime/gpu.js) against the CPU's (src/runtime/forward.js) and NumPy's (llama2_numpy.py), in
// a real browser, as WebGPU is nowhere else: Playwright's Chromium, whose WebGPU without a GPU is SwiftShader (the CPU
// in the GPU's place: its speed means nothing, its numbers are right or not).
//
//   node tests/gpu-check.mjs [model id | synthetic | made-up ...] [--engine chromium|chrome|msedge] [--forms <part,part>]
//   node tests/gpu-check.mjs ... --engine dawn --webgpu <the npm package webgpu's directory>
//   ... [--nan all|made-up|small|none]   (T.241's review: where the rounds with a NaN or an infinity run; see gpu-check/models.mjs's NAN_CHOICES)
//
// T.147: --engine dawn runs the same in Node on Dawn (the npm package webgpu, not a dependency of this project: install
// it under .tmp/) with the Vulkan of the machine, Mesa's lavapipe on the development machine: shader-f16 and subgroups,
// which SwiftShader lacks (AGENTS.md). The harness and the GPU's worker are then worker threads.
//
// Node reads each model with Pyodide as the page does and records two things: the plan that forward.js gets from
// Python (where every tensor is), and NumPy's answer for a prompt of 150 tokens: the keys and values of every layer at
// every position the prompt's blocks fill, and the logits of its last token. NumPy multiplies the int8 weights
// widened to float32 by float32 activations, which is what the GPU does; forward.js on the CPU quantizes the
// activations as well (7 bits with relaxed SIMD), so it is farther from both by design. The browser then runs
// forward.js in a worker (it waits in Atomics.wait, which a page may not) on the same memory: the prompt through
// forwardMany() and its last token through forward(), once on the CPU and with the GPU's worker once for the shaders
// it chooses and once for every tiled shader of the matrices (T.147), the prompt in blocks of 16 and then all at once.
// The cache starts at 8 positions, so that both grow within the prompt (8 to 256). Checked:
//   - the GPU took every token of the prompt (gpuTokens), the same again from position 0, and a block that begins
//     past the keys and values it holds went to the CPU;
//   - the keys and values it wrote back into the cache, against NumPy's: the worst row (a layer's keys or values of
//     one position) no more than the line of the shader's arithmetic (GPU_LINE and the next, see gpu-check/lines.mjs);
//   - T.147: a request forward.js gave up on (a GPU that answers late) writes nothing (the made-up model only);
//   - T.187: the scale of the first layer's keys and values against NumPy's (SCALE_LINE, see gpu-check/lines.mjs);
//   - the logits of the prompt's last token (the CPU's in both runs, on the GPU's keys and values in one): NumPy's
//     most likely token or a near tie (their KL divergence from NumPy's is printed, not held to a line: see lines.mjs).
//
// T.357.3: this file is the window: the arguments, and the run from the models to the exit. Its parts are in
// tests/gpu-check/: models.mjs (the made-up models, what each is for, and which an engine runs), lines.mjs (the numbers
// a run is held to, and why), reference.mjs (NumPy's answers and the plan, a case a model), engines.mjs and harness.js
// (the browser or Dawn, and the worker that runs forward.js there), measures.mjs, verdicts.mjs, steps.mjs and
// tables.mjs (the comparisons and the lines of the log: functions of what the harness posted, which
// tests/gpu-check-parts-check.mjs runs on made-up numbers).
import fs from "node:fs";
import path from "node:path";
import { leave } from "./leave.mjs";
import { NAN_CHOICES, madeUpOn } from "./gpu-check/models.mjs";
import { referencePython, referenceCase } from "./gpu-check/reference.mjs";
import { served, inBrowser, inDawn } from "./gpu-check/engines.mjs";
import { quantizersWrong, modelWrong } from "./gpu-check/verdicts.mjs";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args.splice(args.indexOf(name), 2)[1] : value);
const engine = option("--engine", "chromium");
// T.147: --forms <part,part>: only the matrices' shaders whose names hold one of these (all of them by default)
const only = option("--forms", "");
const webgpu = option("--webgpu", "");
// T.241's review: where the rounds with a NaN or an infinity run (models.mjs's NAN_CHOICES, and why)
const nanRounds = NAN_CHOICES[option("--nan", { dawn: "made-up", chromium: "small" }[engine] ?? "none")] ?? NAN_CHOICES.none;
// the models to check: by default every made-up one and the site's three; "made-up" stands for every made-up one (T.193:
// gpu-prompt.yml's suites name them so, and a made-up model added to models.mjs's SYNTHETIC joins them)
const madeUp = madeUpOn(engine);
const ids = (args.length ? args : ["made-up", "stories15M", "tiny-lm", "llm-jp-3-150m"])
  .flatMap((id) => (id === "made-up" ? madeUp : [id]));
if ((args.length ? args : ["made-up"]).includes("made-up")) console.log(`suite: "made-up" on ${engine} is ${madeUp.join(" ")}`);

// ---- Node: the plans and NumPy's answers
const py = await referencePython();
const directory = path.join(root, ".tmp", "gpu-check");
fs.mkdirSync(directory, { recursive: true });
const cases = ids.map((id) => referenceCase(py, directory, id));

// ---- the browser or Dawn: what the harness posted
const server = served(cases, only, nanRounds);
const lines = [];
const outcome = engine === "dawn" ? await inDawn({ webgpu, directory, server, only, nanRounds, lines }) : await inBrowser({ engine, server, lines });
server.close();
if (lines.length) console.log(lines.join("\n"));
// T.147: the list of forms comes from an adapter of its own, which SwiftShader now and then does not give (AGENTS.md)
if (!outcome.error && !outcome.forms?.length) outcome.error = "no tiled shader to force: the harness got no GPU adapter (run it again)";
// (T.226: an exit once what was printed has gone out. An exit at once cut the log of CI's Dawn job in the middle of a
// line of the console, before the error: run 36867483459 failed, and nothing said why)
// (tests/leave.mjs, T.357: the one place for it)
if (outcome.error) {
  console.error(`FAILED\n- ${outcome.error}`);
  await leave(1);
}

// ---- the comparisons
let failed = quantizersWrong(outcome);
for (const result of outcome.results) failed = modelWrong(cases, result) || failed;
await leave(failed ? 1 : 0);
