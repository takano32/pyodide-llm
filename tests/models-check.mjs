// src/models.js: every model says where it comes from and under which license (T87). Node alone.
//
//   node tests/models-check.mjs
import assert from "node:assert/strict";
import { GROUPS, LICENSES, MODELS, PAGE_MEMORY, SIX_OF_EIGHT, filled, memoryFailure, writesJapanese, memoryWarning, modelBytes, sourceOf, sources, weightsFor } from "../src/models.js";

for (const entry of MODELS) {
  const repo = sourceOf(entry);
  assert.ok(repo, `${entry.id} has neither hf.repo nor source`);
  assert.ok(LICENSES[repo], `${entry.id}: no license for ${repo} in LICENSES`);
}
for (const entry of MODELS.filter((entry) => entry.original)) {
  assert.ok(LICENSES[entry.original], `${entry.id}: no license for its original ${entry.original}`);
}
const used = new Set(MODELS.flatMap((entry) => [sourceOf(entry), entry.original].filter(Boolean)));
for (const repo of Object.keys(LICENSES)) assert.ok(used.has(repo), `LICENSES names ${repo}, which no model uses`);
const listed = sources();
assert.equal(listed.length, used.size, "one line per source");
assert.equal(listed.reduce((n, { names }) => n + names.length, 0),
  MODELS.length + MODELS.filter((entry) => entry.original).length, "every model on some line, a GGUF on two");

// T90: every model knows how much memory it takes, so the warning never silently skips one
for (const entry of MODELS) assert.ok(modelBytes(entry) > 0, `${entry.id}: no size (bytes, or "int8 N MB" in the note)`);
const byId = (id) => MODELS.find((entry) => entry.id === id);
const big = byId("hf-qwen2.5-1.5b-instruct");
assert.equal(modelBytes(big), 1.7e9);
assert.equal(memoryWarning(big, undefined), "", "a browser that does not tell gets no guess");
assert.equal(memoryWarning(big, 8), "", "8 GB holds it");
assert.match(memoryWarning(big, 2), /needs about 2,000 MB of memory, and this device has 2 GB/);
assert.equal(memoryWarning(byId("tiny-lm"), 1), "", "the default fits a 1 GB device");
assert.equal(PAGE_MEMORY, 300e6);
// a float16 original is widened when loaded (Fable's review): three times its file, so a 1 GB device is warned
assert.equal(modelBytes(byId("llm-jp-3-150m-f16")), 305161244 * 3);
assert.match(memoryWarning(byId("llm-jp-3-150m-f16"), 2), /needs about 1,215 MB/);
assert.equal(modelBytes(byId("stories15M-f32")), 60816028, "float32 is used as it is");
assert.match(memoryFailure(big, 1234567890, "MemoryError"), /ran out of memory for Qwen2.5 1.5B Instruct \(it needs about 2,000 MB\); the page was using 1,235 MB when it happened\. .* \(MemoryError\)$/);
assert.ok(!memoryFailure({ name: "mine.bin" }, undefined, "").includes("undefined"));

// T98: six bits where int8 does not fit, and where asked
const small = byId("hf-smollm2-135m-instruct"), qwen = byId("hf-qwen2.5-1.5b-instruct");
assert.equal(weightsFor(byId("tiny-lm"), "6", 2), undefined, "a model of the site is not converted");
assert.equal(weightsFor(small, undefined, undefined), undefined, "nothing told: the worker chooses (T115)");
assert.equal(weightsFor(small, "6", 8), "int6", "?bits=6");
assert.equal(weightsFor(qwen, "8", 2), "int8", "?bits=8 even where it does not fit");
assert.equal(weightsFor({ hf: {}, conversion: { dtype: "float16" }, note: "int8 4.2 GB" }, "6", 2), "float16", "the settings of a visitor's files win (T119)");
assert.equal(weightsFor(qwen, undefined, 8), undefined, "1.7 GB fits in half of 8 GB: the worker chooses");
assert.equal(weightsFor(qwen, undefined, 2), "int6", "not in half of 2 GB");
assert.equal(weightsFor({ hf: {}, note: "int8 4.2 GB" }, undefined, undefined), undefined, "past a 32-bit memory: the worker, which knows the header, chooses");
assert.equal(modelBytes({ ...qwen, conversion: { dtype: "int6" } }), modelBytes(qwen) * SIX_OF_EIGHT);
// T133: Chromium says at most 8 GB: a device at the cap may have any more, so six bits are not asked for there, and
// only a model past 8 GB is warned of
const seven = { name: "7B", hf: {}, note: "int8 9.2 GB" }, three = { name: "3B", hf: {}, note: "int8 3.6 GB" };
assert.equal(weightsFor(three, undefined, 8), undefined, "at the cap: the worker chooses (int8 on a 64-bit memory)");
assert.equal(weightsFor(three, undefined, 4), "int6", "4 GB is told as it is");
assert.equal(memoryWarning(three, 8), "", "3.9 GB on a device of 8 GB or more");
assert.match(memoryWarning(seven, 8), /7B needs about 9,500 MB of memory, and this device has 8 GB or more: it may run out of memory\./);
assert.match(memoryWarning(three, 4), /this device has 4 GB:/);
// T132: a template's {date} is the visitor's day, and {prompt} what was typed (a {date} typed stays as it is)
assert.equal(filled("Current date: {date}\n{prompt}", "a {date} b", new Date(2026, 8, 6)), "Current date: 2026-09-06\na {date} b");
assert.equal(filled(byId("hf-llm-jp-4-8b-instruct").template, "x").includes("{"), false);
for (const typed of ["cost $$5", "a $& b", "x $` y", "y $' z"]) {
  assert.equal(filled("<u>{prompt}</u>", typed), `<u>${typed}</u>`, "what was typed goes in as it is, $ and all");
}
// the line of a redistribution says what it is: a GGUF, or a copy of the same files (the review of T132)
const lineOf = (repo) => listed.find((line) => line.repo === repo).names;
assert.deepEqual(lineOf("bartowski/SmolLM2-135M-Instruct-GGUF"), ["SmolLM2 135M Instruct (GGUF)"]);
assert.deepEqual(lineOf("unsloth/Llama-3.2-1B-Instruct"), ["Llama 3.2 1B Instruct (copy)"]);
assert.ok(lineOf("meta-llama/Llama-3.2-1B-Instruct").includes("Llama 3.2 1B Instruct"), "the original's line has the plain name");
// T128: the list's order (the owner's): the groups as GROUPS has them, and within each the ones that write Japanese
// from light to heavy, then the English-only ones from light to heavy. The default is the first
assert.equal(MODELS[0].id, "tiny-lm", "the default comes first");
assert.equal(new Set(MODELS.map(({ id }) => id)).size, MODELS.length, "no model twice");
const groupOf = (entry) => Object.keys(GROUPS).indexOf(entry.group ?? "site");
MODELS.slice(1).forEach((entry, i) => {
  const before = MODELS[i];
  const key = (m) => [groupOf(m), writesJapanese(m) ? 0 : 1, modelBytes(m)];
  const [a, b] = [key(before), key(entry)];
  assert.ok(a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] <= b[2]))),
    `${before.id} comes before ${entry.id}, out of the order of T128`);
});
assert.ok(writesJapanese({ note: "translates 日本語 ⇄ English" }) && !writesJapanese({ note: "English" }));
console.log("ok");
