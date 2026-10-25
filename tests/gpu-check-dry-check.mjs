// T.357.3: tests/gpu-check.mjs itself, run from its arguments to its exit where no WebGPU runs. The tool's GPU is CI's
// (gpu-prompt.yml), and until this check a slip in what is not the GPU's (the arguments, NumPy's answers and the plan,
// the server, Dawn's worker threads, the harness's runs on the CPU, the verdicts and the tables) showed there alone,
// twenty minutes into a run. Here the tool runs as CI's Dawn job runs it, with a stand-in for the npm package webgpu
// whose adapter gives no device (tests/gpu-check/no-device): every run of the harness goes to the CPU, and the tool is
// to say so of each and exit 1. Checked, of three made-up models (Llama's int8, GPT-2's, ternary weights):
//   - the tool ends by its verdicts (exit 1, no error of its own), and the harness's error is nowhere;
//   - a model's first line: the CPU's keys and values near NumPy's, as forward.js's 7-bit activations are (far under
//     the 1 and more of a wrong model), E16 small;
//   - a run whose GPU did not open is the CPU's own run: the same keys and values block by block and all at once, to
//     the digits printed, and the line says FAILED and why;
//   - the steps are said to be on the CPU, the tables have a column a run and a row a layer, the seconds are said.
// It says nothing of a shader: no shader runs.
//
//   node tests/gpu-check-dry-check.mjs      (about ten seconds: Pyodide's start; make kernels first)
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = (name) => fileURLToPath(new URL(name, import.meta.url));
const MODELS = { synthetic: 2, "synthetic-gpt2": 3, "synthetic-ternary": 3 };   // their layers
const ran = spawnSync(process.execPath, [here("./gpu-check.mjs"), ...Object.keys(MODELS), "--nan", "small", "--engine", "dawn", "--webgpu", here("./gpu-check/no-device")],
  { encoding: "utf8", maxBuffer: 1 << 28, timeout: 240000 });
const out = ran.stdout, lines = out.split("\n");
const shown = `\n${out.slice(0, 4000)}\n${ran.stderr.slice(0, 4000)}`;
assert.equal(ran.status, 1, `gpu-check ended with ${ran.status ?? ran.signal}, where its verdicts are to end it with 1${shown}`);
assert.doesNotMatch(ran.stderr, /Error|FAILED/, "gpu-check wrote an error of its own");
assert.ok(!lines.includes("FAILED"), `the harness did not run to its end${shown}`);
assert.match(lines[0], /^the quantizers alone: FAILED$/);
assert.match(lines[1], /^ {4}- Error: the stand-in gives no device$/);

let runs = 0;
for (const [id, layers] of Object.entries(MODELS)) {
  const first = lines.findIndex((line) => line.startsWith(`${id} (${layers} layers, `));
  assert.ok(first > 0, `no first line of ${id}${shown}`);
  const head = lines[first].match(/149 tokens\): the CPU's keys and values (\S+) from NumPy's .*, most likely (\d+) .*; E16 \(NumPy's cache in float16\) (\S+) /);
  assert.ok(head, lines[first]);
  const [, cpu, , e16] = head;
  assert.ok(Number(cpu) > 1e-3 && Number(cpu) < 0.6, `${id}: the CPU's keys and values are ${cpu} from NumPy's`);
  assert.ok(Number(e16) > 1e-5 && Number(e16) < 2e-2, `${id}: E16 is ${e16}`);
  // the model's lines, up to the next model's first
  const next = lines.findIndex((line, i) => i > first && /^\S+ \(\d+ layers, /.test(line));
  const own = lines.slice(first, next < 0 ? lines.length : next);
  const of = own.filter((line) => line.startsWith("  no form, no attention: "));
  assert.ok(of.length >= 3, `${id}: ${of.length} runs`);
  for (const line of of) {
    assert.ok(line.startsWith(`  no form, no attention: keys and values ${cpu} (all at once ${cpu}, `), `${id}: a run whose GPU gave no device is not the CPU's own run: ${line}`);
    assert.match(line, /\(prompts on the CPU \(the stand-in gives no device\)\).* — FAILED$/);
  }
  runs += of.length;
  assert.ok(own.includes("    - the GPU did not take it: prompts on the CPU (the stand-in gives no device)"));
  assert.ok(own.includes(`    - the GPU took 0 and 0 of 149 tokens`));
  assert.ok(own.some((line) => line === "  a token: on the CPU (the stand-in gives no device) — FAILED"), `${id}: the steps' line`);
  // the tables: E16, Q8, the CPU and a letter a run; a row a layer and one of all
  const letters = of.map((_, i) => String.fromCharCode(65 + i)).join(" | ");
  const table = own.indexOf(`| layer | E16 | Q8 | CPU | ${letters} |`);
  assert.ok(table > 0, `${id}: the first table's columns`);
  assert.deepEqual(own.slice(table + 2, table + 3 + layers).map((row) => row.split(" | ")[0]), [...Array.from({ length: layers }, (_, l) => `| ${l}`), "| all"]);
  assert.equal(own.filter((line) => line === `| layer | CPU | ${letters} |`).length, 2, `${id}: the other two tables`);
  assert.ok(own.some((line) => line.startsWith("- CPU: forward.js on the CPU: ")) && own.some((line) => line.startsWith(`- ${String.fromCharCode(64 + of.length)}: no form, no attention: `)));
  assert.ok(own.some((line) => line.startsWith(`seconds of ${id}: NumPy (with its cache in float16, Q8 and H16) `)));
}
// what the made-up model alone is asked
for (const start of ["  a GPU that answers late: not tried (", "  a fallback adapter without the tests' leave: prompts on the CPU (a fallback adapter", "  a GPU let go while it got ready: its worker ended in ",
  "  the shaders remembered: for this adapter not taken"]) {
  assert.equal(lines.filter((line) => line.startsWith(start)).length, 1, start);
}
console.log(`gpu-check-dry-check: ok (gpu-check ran to its verdicts with no device: ${Object.keys(MODELS).length} made-up models, ${runs} runs, each the CPU's own and said to have failed)`);
