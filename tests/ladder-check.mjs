// tests/ladder.mjs: the table of the Pythia ladder (T84). Node alone.
//
//   node tests/ladder-check.mjs
import assert from "node:assert/strict";
import { ladderMarkdown } from "./ladder.mjs";

const run = (model, tokPerSecond, extra = {}) => ({ engine: "chromium", browserVersion: "148.0", model, os: "linux x64",
  ok: true, readySeconds: 20, tokPerSecond, load: { pyodide: 3, download: 15, convert: 6, construct: 1 }, heapMB: 300, ...extra });
const table = ladderMarkdown([run("hf-pythia-160m", 60), run("tiny-lm", 400), run("hf-pythia-70m", 1, { ok: false }),
  run("hf-pythia-70m", 120), run("hf-pythia-1.4b", 5, { load: null, heapMB: null })]);
const rows = table.split("\n").filter((line) => line.startsWith("| Pythia"));
assert.deepEqual(rows.map((row) => row.split(" | ")[0]), ["| Pythia 70M", "| Pythia 160M", "| Pythia 1.4B"], "in the order of size, only the ladder");
assert.ok(rows[0].includes("| 96 | 20.0 | 15.0 | 6.0 | 120.0 | 11.52 | 300 |  |"), rows[0]);  // the last run of a model wins
assert.ok(rows[2].includes("| 1500 | 20.0 |  |  | 5.0 | 7.50 |  |  |"), rows[2]);  // missing numbers stay empty
assert.ok(!table.includes("undefined") && !table.includes("NaN") && !table.includes("null"), table);
assert.ok(table.includes("chromium 148.0; linux x64."), table);
assert.equal(ladderMarkdown([run("tiny-lm", 400)]), "No Pythia run was recorded.\n");
console.log("ok");
