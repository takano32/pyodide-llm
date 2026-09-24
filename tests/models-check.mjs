// src/models.js: every model says where it comes from and under which license (T87). Node alone.
//
//   node tests/models-check.mjs
import assert from "node:assert/strict";
import { LICENSES, MODELS, PAGE_MEMORY, memoryFailure, memoryWarning, modelBytes, sourceOf, sources } from "../src/models.js";

for (const entry of MODELS) {
  const repo = sourceOf(entry);
  assert.ok(repo, `${entry.id} has neither hf.repo nor source`);
  assert.ok(LICENSES[repo], `${entry.id}: no license for ${repo} in LICENSES`);
}
const used = new Set(MODELS.map(sourceOf));
for (const repo of Object.keys(LICENSES)) assert.ok(used.has(repo), `LICENSES names ${repo}, which no model uses`);
const listed = sources();
assert.equal(listed.length, used.size, "one line per source");
assert.equal(listed.reduce((n, { names }) => n + names.length, 0), MODELS.length, "every model on some line");

// T90: every model knows how much memory it takes, so the warning never silently skips one
for (const entry of MODELS) assert.ok(modelBytes(entry) > 0, `${entry.id}: no size (bytes, or "int8 N MB" in the note)`);
const byId = (id) => MODELS.find((entry) => entry.id === id);
const big = byId("hf-qwen2.5-1.5b-instruct");
assert.equal(modelBytes(big), 1.6e9);
assert.equal(memoryWarning(big, undefined), "", "a browser that does not tell gets no guess");
assert.equal(memoryWarning(big, 8), "", "8 GB holds it");
assert.match(memoryWarning(big, 2), /needs about 1,900 MB of memory, and this device has 2 GB/);
assert.equal(memoryWarning(byId("tiny-lm"), 1), "", "the default fits a 1 GB device");
assert.equal(PAGE_MEMORY, 300e6);
assert.match(memoryFailure(big, 1234567890, "MemoryError"), /ran out of memory for Qwen2.5 1.5B Instruct \(it needs about 1,900 MB\); the page was using 1,235 MB when it happened\. .* \(MemoryError\)$/);
assert.ok(!memoryFailure({ name: "mine.bin" }, undefined, "").includes("undefined"));
console.log("ok");
