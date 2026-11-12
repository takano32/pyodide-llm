// tests/ladder.ts: the table of the Pythia ladder (T84). Node alone.
//
//   node tests/ladder-check.ts
import assert from "node:assert/strict";
import { ladderMarkdown, threadsOf, tokensOf } from "./ladder.ts";

const run = (model, tokPerSecond, extra = {}) => ({ engine: "chromium", browserVersion: "148.0", model, os: "linux x64",
  cpu: "AMD EPYC 7763 64-Core Processor", ok: true, readySeconds: 20, tokPerSecond,
  meta: `${model} · 256 tokens · ${tokPerSecond} tok/s · 4 threads · int8`, backend: "int8 kernels, 4 threads",
  load: { pyodide: 3, download: 15, convert: 6, construct: 1 }, heapMB: 300, ...extra });
const table = ladderMarkdown([run("hf-pythia-160m", 60), run("tiny-lm", 400), run("hf-pythia-70m", 1, { ok: false }),
  run("hf-pythia-70m", 120), run("hf-pythia-1.4b", 5, { load: null, heapMB: null, meta: "x · 9 tokens · 5 tok/s · int8" })]);
const rows = table.split("\n").filter((line) => line.startsWith("| EleutherAI Pythia"));
assert.deepEqual(rows.map((row) => row.split(" | ")[0]), ["| EleutherAI Pythia 70M", "| EleutherAI Pythia 160M", "| EleutherAI Pythia 1.4B"], "in the order of size, only the ladder");
assert.ok(rows[0].includes("| 96 | 20.0 | 15.0 | 6.0 | 30 | 256 | 120.0 | 4 | 11.52 | 300 |  |"), rows[0]);  // the last run of a model wins
assert.ok(rows[2].includes("| 1600 | 20.0 |  |  |  | 9 | 5.0 | 1 | 8.00 |  |  |"), rows[2]);  // missing numbers stay empty; one thread unsaid
assert.ok(!table.includes("undefined") && !table.includes("NaN") && !table.includes("null"), table);
assert.ok(table.includes("chromium 148.0; linux x64, AMD EPYC 7763 64-Core Processor; on the CPU (no WebGPU)."), table);
assert.equal(ladderMarkdown([run("tiny-lm", 400)]), "No Pythia run was recorded.\n");
// two systems: two tables, each with its own CPU, and a rung of one does not take the place of the other's
const two = ladderMarkdown([run("hf-pythia-70m", 120), run("hf-pythia-70m", 90, { os: "linux arm64", cpu: "unknown" })]);
assert.equal(two.split("\n").filter((line) => line.startsWith("| EleutherAI Pythia 70M")).length, 2, two);
assert.ok(two.includes("linux arm64, unknown;") && two.includes("linux x64, AMD EPYC"), two);
assert.ok(ladderMarkdown([run("hf-pythia-70m", 120, { meta: "x · 3 tok/s · on WebGPU" })]).includes("the GPU took a part"));
assert.equal(threadsOf(undefined), null);
assert.equal(tokensOf(undefined), null);
assert.equal(threadsOf("a failed run"), null);
console.log("ok");
