// tests/ladder.ts: the table of the Pythia ladder (T84). Node alone.
//
//   node tests/ladder-check.ts
import assert from "node:assert/strict";
import { ladderMarkdown, threadsAsked, threadsOf, tokensOf } from "./ladder.ts";

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
// the same system with two CPUs (GitHub's runners differ from job to job) is two tables too
const cpus = ladderMarkdown([run("hf-pythia-70m", 120), run("hf-pythia-70m", 90, { cpu: "AMD EPYC 9V74 80-Core Processor" })]);
assert.equal(cpus.split("\n").filter((line) => line.startsWith("| EleutherAI Pythia 70M")).length, 2, cpus);
assert.ok(ladderMarkdown([run("hf-pythia-70m", 120, { meta: "x · 3 tok/s · on WebGPU" })]).includes("the GPU took a part"));
// a thread count the URL asked for and the page did not run on is said in the cell (the 70M of run 43291611379)
const asked = ladderMarkdown([run("hf-pythia-70m", 120, { query: "threads=4&seed=1", meta: "x · 256 tokens · 120 tok/s · temp 0.7" }),
  run("hf-pythia-160m", 60, { query: "seed=1&threads=4" })]).split("\n").filter((line) => line.startsWith("| EleutherAI Pythia"));
assert.ok(asked[0].includes("| 120.0 | 1 (asked 4) |") && asked[1].includes("| 60.0 | 4 |"), asked.join("\n"));
assert.equal(threadsAsked("seed=1&threads=4"), 4);
assert.equal(threadsAsked("mythreads=4"), null);
assert.equal(threadsAsked(undefined), null);
assert.equal(threadsOf(undefined), null);
assert.equal(tokensOf(undefined), null);
assert.equal(threadsOf("a failed run"), null);
console.log("ok");
