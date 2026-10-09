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
for (const entry of MODELS.filter((entry) => entry.hf?.vocabulary)) {
  // T136: the repository its vocabulary and config.json come from is a source too
  assert.ok(LICENSES[entry.hf.vocabulary.repo], `${entry.id}: no license for ${entry.hf.vocabulary.repo} in LICENSES`);
}
// every Hugging Face entry pins what it fetches by a full commit hash (AGENTS.md: so that nothing changes under the
// page), the repository its vocabulary comes from as well. The review of T235 found all 54 entries so, with no check
// that said it must be: a branch name left in an entry would pass every test and let its files change
for (const entry of MODELS.filter((entry) => entry.hf)) {
  for (const { revision } of [entry.hf, entry.hf.vocabulary].filter(Boolean)) {
    assert.match(revision, /^[0-9a-f]{40}$/, `${entry.id}: a revision is a full commit hash, not ${revision}`);
  }
}
// T274's review: an entry's generation reaches Llama.generate() as keyword arguments, so a misspelt key (topk for top_k)
// is a TypeError at the first answer, in the browser only, and a value outside what generate() accepts is a ValueError
// there. Only the keys the page and the engine know, in the ranges they take
{
  const ranges = { steps: [0, Infinity], temperature: [0, 1.5], topp: [0.05, 1], repetition_penalty: [1, 2], top_k: [0, Infinity],
    min_p: [0, 1], presence_penalty: [0, 2] };
  for (const entry of MODELS) {
    for (const [key, value] of Object.entries(entry.generation ?? {})) {
      assert.ok(ranges[key], `${entry.id}: generation.${key} is no setting of the page`);
      assert.ok(Number.isFinite(value) && value >= ranges[key][0] && value <= ranges[key][1], `${entry.id}: generation.${key} = ${value} is out of ${ranges[key]}`);
    }
  }
}
const used = new Set(MODELS.flatMap((entry) => [sourceOf(entry), entry.original, entry.hf?.vocabulary?.repo].filter(Boolean)));
for (const repo of Object.keys(LICENSES)) assert.ok(used.has(repo), `LICENSES names ${repo}, which no model uses`);
const listed = sources();
assert.equal(listed.length, used.size, "one line per source");
assert.equal(listed.reduce((n, { names }) => n + names.length, 0),
  MODELS.reduce((n, entry) => n + new Set([entry.original, entry.hf?.vocabulary?.repo, sourceOf(entry)].filter(Boolean)).size, 0),
  "every model on the line of each repository it comes from: a GGUF on two, or three where its vocabulary is a copy's");

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
// T230: a ternary model is converted to the ternary dtype, its weights as they are, on any device and whatever fits;
// ?bits=8 and ?bits=6 widen them (to compare), and the memory is then four times as much (and 7/9 of that)
const bonsai = byId("hf-ternary-bonsai-1.7b");
for (const deviceMemory of [undefined, 1, 2, 8]) assert.equal(weightsFor(bonsai, undefined, deviceMemory), "ternary");
assert.equal(weightsFor(bonsai, "8", 8), "int8", "?bits=8 of a ternary model");
assert.equal(weightsFor(bonsai, "6", 8), "int6", "?bits=6 of a ternary model");
assert.equal(modelBytes(bonsai), 484e6);
assert.equal(modelBytes({ ...bonsai, conversion: { dtype: "ternary" } }), 484e6);
assert.equal(modelBytes({ ...bonsai, conversion: { dtype: "int8" } }), 484e6 * 4);
assert.equal(modelBytes({ ...bonsai, conversion: { dtype: "int6" } }), 484e6 * 4 * SIX_OF_EIGHT);
// T233: Ternary Bonsai 2 27B, ternary on any device (as int8 it is past every browser's memory). A device that says
// less than 8 GB is warned; one at the cap (8 GB or more) is not, by the page's rule (T133: the note's 7.7 GB and the
// page's 0.3 are under 8 GiB), although the checkpoint with its forward pass and the page is about 8.6 GB, which a
// device of exactly 8 GB does not hold: what to tell such a device is the owner's to word (TODO.md's T233).
// Its original is the model it is built from, whose weights are other ones (rebuilt: gguf.yml skips it)
const largest = byId("hf-ternary-bonsai-2-27b");
for (const deviceMemory of [undefined, 4, 8]) assert.equal(weightsFor(largest, undefined, deviceMemory), "ternary");
assert.equal(modelBytes(largest), 7.7e9);
assert.match(memoryWarning(largest, 4), /needs about 8,000 MB of memory, and this device has 4 GB:/);
assert.equal(memoryWarning(largest, 8), "");
assert.ok(largest.rebuilt && largest.original === largest.hf.vocabulary.repo && largest.download === 5946648928);
assert.deepEqual(MODELS.filter((entry) => entry.rebuilt).map(({ id }) => id), ["hf-ternary-bonsai-2-27b-thinking", "hf-ternary-bonsai-2-27b"],
  "rebuilt is for a GGUF no check can hold to its original");
// its two forms share one conversion (kept.js's replaced() takes both ids), and both are kept ternary
assert.deepEqual(byId("hf-ternary-bonsai-2-27b-thinking").shares, ["hf-ternary-bonsai-2-27b-thinking", "hf-ternary-bonsai-2-27b"]);
assert.equal(weightsFor(byId("hf-ternary-bonsai-2-27b-thinking"), undefined, 8), "ternary");
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
// T127's review: a template's strftime_now() is written {date:format}, and is the day the prompt is sent
assert.equal(filled("Today Date: {date:%d %b %Y}\n{prompt}", "{date}", new Date(2026, 8, 6)), "Today Date: 06 Sep 2026\n{date}");
// T138: {prompt:trim} is what was typed without the white space at its ends (a template that trims it); {prompt} keeps it
assert.equal(filled("USER: {prompt:trim} ASSISTANT:", "  x \n"), "USER: x ASSISTANT:");
assert.equal(filled("[{prompt}]", "  x \n"), "[  x \n]");
assert.equal(filled("[{prompt:trim}]", " $' "), "[$']");
// T144: and the white space it takes off is Python's str.strip()'s (Jinja's trim), not JavaScript's trim()'s. These
// are the code points of str.isspace() (Python 3.14; none past the BMP): every other one stays, U+FEFF too
const PYTHON_SPACE = new Set([0x9, 0xa, 0xb, 0xc, 0xd, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001,
  0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000]);
for (let code = 0; code < 0x10000; code++) {
  if (code >= 0xd800 && code < 0xe000) continue;  // no characters of their own
  const c = String.fromCharCode(code), got = filled("[{prompt:trim}]", `${c}${c}x${c}y${c}${c}`);
  assert.equal(got, PYTHON_SPACE.has(code) ? `[x${c}y]` : `[${c}${c}x${c}y${c}${c}]`, `U+${code.toString(16).padStart(4, "0")}`);
}
assert.equal(filled("{date:%B %d, %Y (%A) %m/%y %%}", "", new Date(2026, 8, 26)), "September 26, 2026 (Saturday) 09/26 %");
for (const typed of ["cost $$5", "a $& b", "x $` y", "y $' z"]) {
  assert.equal(filled("<u>{prompt}</u>", typed), `<u>${typed}</u>`, "what was typed goes in as it is, $ and all");
}
// the line of a redistribution says what it is: a GGUF, or a copy of the same files (the review of T132)
const lineOf = (repo) => listed.find((line) => line.repo === repo).names;
assert.deepEqual(lineOf("bartowski/SmolLM2-135M-Instruct-GGUF"), ["SmolLM2 135M Instruct (GGUF)"]);
assert.deepEqual(lineOf("unsloth/Llama-3.2-1B-Instruct"), ["Llama 3.2 1B Instruct (copy)"]);
assert.ok(lineOf("meta-llama/Llama-3.2-1B-Instruct").includes("Llama 3.2 1B Instruct"), "the original's line has the plain name");
// every GGUF of the list names its original, is under the original's license (its card's, the same: T136) and is
// "(GGUF)" on its line (the review of T136: a copy's word, another license or no original went through before)
for (const entry of MODELS.filter((entry) => entry.hf?.weights?.endsWith(".gguf"))) {
  assert.ok(entry.original, `${entry.id}: a GGUF without its original`);
  assert.deepEqual(LICENSES[entry.hf.repo], LICENSES[entry.original], `${entry.id}: the GGUF's license is not its original's`);
  assert.ok(lineOf(entry.hf.repo).includes(`${entry.name} (GGUF)`), `${entry.id}: the GGUF's line does not say (GGUF)`);
}
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
