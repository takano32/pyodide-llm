// T135: the prompt on the GPU (src/runtime/gpu.js) against the CPU's (src/runtime/forward.js) and NumPy's (llama2_numpy.py), in
// a real browser, as WebGPU is nowhere else: Playwright's Chromium, whose WebGPU without a GPU is SwiftShader (the CPU
// in the GPU's place: its speed means nothing, its numbers are right or not).
//
//   node tests/gpu-check.ts [model id | synthetic | made-up ...] [--engine chromium|chrome|msedge] [--forms <part,part>]
//   node tests/gpu-check.ts ... --engine dawn --webgpu <the npm package webgpu's directory>
//   ... [--minutes 90]   (T645: the tool's limit from its start; progress lines go to stderr as the harness goes)
//   ... [--nan all|made-up|small|none]   (T241's review: where the rounds with a NaN or an infinity run; see gpu-check/models.ts's NAN_CHOICES)
//
// T147: --engine dawn runs the same in Node on Dawn (the npm package webgpu, not a dependency of this project: install
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
// it chooses and once for every tiled shader of the matrices (T147), the prompt in blocks of 16 and then all at once.
// The cache starts at 8 positions, so that both grow within the prompt (8 to 256). Checked:
//   - the GPU took every token of the prompt (gpuTokens), the same again from position 0, and a block that begins
//     past the keys and values it holds went to the CPU;
//   - the keys and values it wrote back into the cache, against NumPy's: the worst row (a layer's keys or values of
//     one position) no more than the line of the shader's arithmetic (GPU_LINE and the next, see gpu-check/lines.ts);
//   - T147: a request forward.js gave up on (a GPU that answers late) writes nothing (the made-up model only);
//   - T187: the scale of the first layer's keys and values against NumPy's (SCALE_LINE, see gpu-check/lines.ts);
//   - the logits of the prompt's last token (the CPU's in both runs, on the GPU's keys and values in one): NumPy's
//     most likely token or a near tie (their KL divergence from NumPy's is printed, not held to a line: see lines.ts).
//
// T357.3: this file is the window: the arguments, and the run from the models to the exit. Its parts are in
// tests/gpu-check/: models.ts (the made-up models, what each is for, and which an engine runs), lines.ts (the numbers
// a run is held to, and why), reference.ts (NumPy's answers and the plan, a case a model), engines.ts and harness.js
// (the browser or Dawn, and the worker that runs forward.js there), measures.ts, verdicts.ts, steps.ts and
// tables.ts (the comparisons and the lines of the log: functions of what the harness posted, which
// tests/gpu-check-parts-check.ts runs on made-up numbers).
import fs from "node:fs";
import path from "node:path";
import { leave } from "./leave.ts";
import { NAN_CHOICES, madeUpOn } from "./gpu-check/models.ts";
import { referencePython, referenceCase } from "./gpu-check/reference.ts";
import { served, inBrowser, inDawn } from "./gpu-check/engines.ts";
import { quantizersWrong, modelWrong } from "./gpu-check/verdicts.ts";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args.splice(args.indexOf(name), 2)[1] : value);
const engine = option("--engine", "chromium");
// T147: --forms <part,part>: only the matrices' shaders whose names hold one of these (all of them by default)
const only = option("--forms", "");
const webgpu = option("--webgpu", "");
// T645: the minutes the tool may take from its start, under its job's (gpu-prompt.yml: 120, of which the setup takes up to
// 20); past them it gives the verdicts of the models through, and fails naming the model and the run it was at
const minutes = Number(option("--minutes", "90"));
// T241's review: where the rounds with a NaN or an infinity run (models.ts's NAN_CHOICES, and why)
const nanRounds = NAN_CHOICES[option("--nan", { dawn: "made-up", chromium: "small" }[engine] ?? "none")] ?? NAN_CHOICES.none;
// the models to check: by default every made-up one and the site's three; "made-up" stands for every made-up one (T193:
// gpu-prompt.yml's suites name them so, and a made-up model added to models.ts's SYNTHETIC joins them)
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
const outcome = engine === "dawn" ? await inDawn({ webgpu, directory, server, only, nanRounds, lines, minutes }) : await inBrowser({ engine, server, lines, minutes });
server.close();
if (lines.length) console.log(lines.join("\n"));
// T147: the list of forms comes from an adapter of its own, which SwiftShader now and then does not give (AGENTS.md)
if (!outcome.error && !outcome.forms?.length) outcome.error = "no tiled shader to force: the harness got no GPU adapter (run it again)";
// (T226: an exit once what was printed has gone out. An exit at once cut the log of CI's Dawn job in the middle of a
// line of the console, before the error: run 36867483459 failed, and nothing said why)
// (tests/leave.ts, T357: the one place for it)
// (T645: an error after some models were through, the tool's limit among them, gives their verdicts first)
if (outcome.error && !outcome.results?.length) {
  console.error(`FAILED\n- ${outcome.error}`);
  await leave(1);
}

// ---- the comparisons
let failed = quantizersWrong(outcome);
for (const result of outcome.results) failed = modelWrong(cases, result) || failed;
if (outcome.error) {
  console.error(`FAILED after ${outcome.results.length} of ${cases.length} models\n- ${outcome.error}`);
  await leave(1);
}
await leave(failed ? 1 : 0);
