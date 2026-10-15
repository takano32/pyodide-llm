// src/models.js: every model says where it comes from and under which license (T87). Node alone.
//
//   node tests/models-check.mjs
import assert from "node:assert/strict";
import { DEFAULT_MODEL, GROUPS, LICENSES, MODELS, PAGE_MEMORY, SIX_OF_EIGHT, filled, memoryFailure, writesJapanese, memoryWarning, modelBytes, sourceOf, sources, weightsFor } from "../src/models.js";
import { PLAIN_ROPE, QWEN3_FROM_IM_START, QWEN3_OWN_BOS, qwen35 } from "../src/models/formats.js";
import { BUILT } from "../src/models/built.js";
import { HF_JAPANESE } from "../src/models/hf-japanese.js";
import { HF_CLASSICS } from "../src/models/hf-classics.js";
import { HF_FAMILIES } from "../src/models/hf-families.js";

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
assert.match(memoryFailure(big, 1234567890, "MemoryError"), /ran out of memory for Alibaba Qwen2.5 1.5B Instruct \(it needs about 2,000 MB\); the page was using 1,235 MB when it happened\. .* \(MemoryError\)$/);
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
assert.deepEqual(MODELS.filter((entry) => entry.rebuilt).map(({ id }) => id),
  ["hf-shisa-v2.1c-lfm2-350m-q4", "hf-ternary-bonsai-2-27b-thinking", "hf-ternary-bonsai-2-27b", "hf-underdog-saluki-27b-thinking", "hf-underdog-saluki-27b"],
  "rebuilt is for a GGUF no check can hold to its original");
// its two forms share one conversion (kept.js's replaced() takes both ids), and both are kept ternary
assert.deepEqual(byId("hf-ternary-bonsai-2-27b-thinking").shares, ["hf-ternary-bonsai-2-27b-thinking", "hf-ternary-bonsai-2-27b"]);
assert.equal(weightsFor(byId("hf-ternary-bonsai-2-27b-thinking"), undefined, 8), "ternary");
// T364.3: Underdog Saluki 27B, its GGUF's blocks kept as they are whatever is asked and whatever the device says
// (conversion.blocks beside the dtype of the rows that are no blocks: ?bits= changes nothing of it, where a ternary
// model is widened; as int8 it is past every browser's memory). What it takes is the entry's own word (loaded: the
// checkpoint's bytes), not its note's, which the owner worded for a visitor. Warned as Ternary Bonsai 2 27B is (a device
// under the cap; at the cap not, by the page's rule, though the whole is past 8 GB: the owner's to word, T503's shelf).
// The names, the note and the sampling are the owner's choices of 2026-10-13: the card's, greedy where it answers at
// once and never a Qwen3.5's presence penalty
{
  const [thinks, answers] = ["hf-underdog-saluki-27b-thinking", "hf-underdog-saluki-27b"].map(byId);
  const after = "2.3-bit weights · fetches 7.9 GB (GGUF), kept as it is: 8.0 GB · about a word a second or less · desktop only · Chrome and Firefox";
  assert.deepEqual([thinks.name, answers.name], ["Underdog Saluki 27B (thinking)", "Underdog Saluki 27B (no thinking)"]);
  // T537: the languages its card says (its metadata's en alone), and so among the English ones
  assert.deepEqual([thinks.note, answers.note], [`thinks before it answers · English · ${after}`, `answers at once · English · ${after}`]);
  assert.deepEqual(thinks.generation, { steps: 0, temperature: 0.6, topp: 0.95, repetition_penalty: 1.0, top_k: 20 });
  assert.deepEqual(answers.generation, { steps: 0, temperature: 0.0 });
  for (const entry of [thinks, answers]) {
    assert.deepEqual(entry.conversion, { dtype: "int8", blocks: true }, entry.id);
    assert.deepEqual(entry.options, qwen35, entry.id);
    assert.deepEqual(entry.shares, [thinks.id, answers.id], "one conversion for both forms");
    assert.ok(entry.rebuilt && entry.original === entry.hf.vocabulary.repo && entry.download === 7898369152 && entry.weights === undefined);
    for (const asked of [undefined, "6", "8"]) for (const deviceMemory of [undefined, 1, 4, 8]) assert.equal(weightsFor(entry, asked, deviceMemory), "int8", `${entry.id}: ?bits=${asked} on ${deviceMemory} GB`);
    assert.equal(modelBytes(entry), 7956712492);
    assert.match(memoryWarning(entry, 4), /needs about 8,257 MB of memory, and this device has 4 GB:/);
    assert.equal(memoryWarning(entry, 8), "");
    assert.equal(memoryWarning(entry, undefined), "");
  }
  assert.ok(thinks.template.endsWith("<think>\n") && answers.template === `${thinks.template}\n</think>\n\n`, "a Qwen3.5's two formats, no system turn");
  assert.deepEqual(MODELS.filter((entry) => entry.conversion?.blocks).map(({ id }) => id), ["hf-shisa-v2.1c-lfm2-350m-q4", thinks.id, answers.id], "the entries whose blocks are kept");
  // loaded: read before the note, of an entry that says it, and nothing of any other entry changed by it: each still
  // says its size by its file or by its note's words
  // (T333: and Gemma 4's two, whose table of every layer's inputs stays in the kept file: below)
  assert.deepEqual(MODELS.filter((entry) => entry.loaded).map(({ id }) => id).sort(), ["hf-gemma-4-e2b", "hf-gemma-4-e4b", "hf-shisa-v2.1c-lfm2-350m-q4", thinks.id, answers.id].sort());
  for (const entry of MODELS.filter((entry) => !entry.loaded && !entry.bytes)) assert.match(entry.note, /(int8|ternary) [\d.]+ (MB|GB)/, `${entry.id}: its size is its note's`);
}
// T333: Gemma 4's E2B and E4B, Google's originals (not gated, Apache 2.0) converted in the page. What they take is the
// int8 checkpoint without the table of every layer's inputs, which stays in the kept file (loaded, a second time here:
// the converter's sizes of the real headers less forward.ts's tableHole()), not the 5.2 and 8.4 GB of their notes. So
// a 4 GB phone is warned of both, and only the E4B needs a 64-bit memory (the E2B ran in a 32-bit one in WebKit, T329.3).
// The stops are generation_config.json's (<eos>, <turn|>, <|tool_response>: the converter says <bos> and <eos>, T524);
// the BOS and the format are the converter's (no `bos` nor `template`); the sampling is generation_config.json's
{
  // (T529: the weights from ggml-org's Q8_0 GGUF, half the download; the vocabulary, config.json and template the original's)
  for (const [size, revision, gguf, download, loaded, devices] of [
    ["E2B", "3e22461f65e89153144f8adb70e3b8c2cc9845a7", "b4243c156154b6dca9324415f8c7ccc098b4aed1", 4967497152, 5207962792 - 2642411520, "desktop only"],
    ["E4B", "ee0ef6023621cff504d758262d4e04895a5af4a2", "b8093469224f83f5c38f691eb906c380e9e63114", 8031242688, 8397500612 - 3170893824, "desktop only · Chrome and Firefox"]]) {
    const entry = byId(`hf-gemma-4-${size.toLowerCase()}`);
    assert.equal(entry.name, `Google Gemma 4 ${size}`);
    assert.deepEqual(entry.hf, { repo: `ggml-org/gemma-4-${size}-it-GGUF`, revision: gguf, weights: `gemma-4-${size}-it-Q8_0.gguf`,
      vocabulary: { repo: `google/gemma-4-${size}-it`, revision, tokenizer: "tokenizer.json" } });
    assert.equal(entry.original, `google/gemma-4-${size}-it`, entry.id);
    assert.equal(LICENSES[entry.hf.repo], "Apache License 2.0", entry.id);
    assert.equal(LICENSES[entry.original], "Apache License 2.0", entry.id);
    assert.ok(entry.download === download && entry.loaded === loaded && modelBytes(entry) === loaded, entry.id);
    assert.ok(entry.note.endsWith(` · ${devices}`), `${entry.id}: ${entry.note}`);
    assert.deepEqual(entry.options, { stop_tokens: [1, 106, 50] }, entry.id);
    assert.ok(!("template" in entry) && !("bos" in entry.options), `${entry.id}: the converter's BOS and format`);
    assert.deepEqual(entry.generation, { steps: 0, temperature: 1.0, topp: 0.95, top_k: 64, repetition_penalty: 1.0 });
    assert.deepEqual(entry.conversion, {}, `${entry.id}: the worker chooses the bits`);
  }
}
// T528: the default (Shisa V2.1c LFM2 350M, int8) and the same model lighter beside it: the same repository's Q4_K_M,
// its blocks kept where the page has kernels for them (as Underdog Saluki 27B's: whatever ?bits= or the device says),
// what it takes said by loaded (tests/quant_sizes.py's checkpoint), rebuilt (below). All else is the default's: the original's
// vocabulary, config.json and format, the stops, the sampling, the license. It comes before the default (lighter), and
// the default stays the int8
{
  const [light, heavy] = ["hf-shisa-v2.1c-lfm2-350m-q4", "hf-shisa-v2.1c-lfm2-350m"].map(byId);
  assert.equal(light.name, "Shisa V2.1c LFM2 350M (4-bit)");
  assert.equal(light.note, "answers instructions · 日本語 / English · 4-bit weights · fetches 229 MB (GGUF), kept mostly as it is: 282 MB");
  assert.deepEqual(light.hf, { ...heavy.hf, weights: "shisa-v2.1c-lfm2-350m.Q4_K_M.gguf" });
  assert.deepEqual(light.conversion, { dtype: "int8", blocks: true });
  assert.ok(light.download === 229311104 && light.loaded === 281641004 && modelBytes(light) === 281641004 && light.weights === undefined);
  for (const asked of [undefined, "6", "8"]) for (const deviceMemory of [undefined, 1, 4, 8]) assert.equal(weightsFor(light, asked, deviceMemory), "int8");
  for (const key of ["original", "options", "generation", "prompt", "placeholder", "template", "bos", "shares"]) assert.deepEqual(light[key], heavy[key], key);
  assert.equal(LICENSES[light.hf.repo], LICENSES[heavy.hf.repo]);
  // rebuilt: gguf.yml skips it (tests/gguf_check.py tensors holds a Q8_0 to its original to 1e-3, which 4 bits are not)
  assert.ok(light.rebuilt && !heavy.rebuilt && light.original === light.hf.vocabulary.repo);
  assert.ok(memoryWarning(light, 1).includes("needs about 582 MB") && memoryWarning(light, 2) === "", "a 1 GB device is warned, as of the default");
  assert.ok(MODELS.indexOf(light) < MODELS.indexOf(heavy), "the lighter before the default (the list's order: what it takes once loaded)");
}
// T364.3: loaded, an entry's own word of what it takes once loaded: read before the note, and after a site's file
assert.equal(modelBytes({ note: "int8 1.7 GB", loaded: 5 }), 5);
assert.equal(modelBytes({ bytes: 7, loaded: 5 }), 7, "a file of this site is its bytes");
// T235 (the owner, 2026-10-11): the three Ternary Bonsai of the list turn by a plain RoPE: rope_scaling: {} beside the
// tokens each had (the 8B's own BOS kept), which an empty object says to the engine and null would not (a jsnull in
// Pyodide). No other entry names a RoPE, and Ternary Bonsai 2 27B, which the choice did not cover, has the options of
// a Qwen3.5 and nothing more
assert.deepEqual(PLAIN_ROPE, { rope_scaling: {} });
for (const [size, tokens] of [["1.7b", QWEN3_OWN_BOS], ["4b", QWEN3_OWN_BOS], ["8b", QWEN3_FROM_IM_START]]) {
  assert.deepEqual(byId(`hf-ternary-bonsai-${size}`).options, { ...tokens, rope_scaling: {} }, `Ternary Bonsai ${size}`);
}
assert.deepEqual(MODELS.filter((entry) => entry.options && "rope_scaling" in entry.options).map(({ id }) => id).sort(),
  ["hf-ternary-bonsai-1.7b", "hf-ternary-bonsai-4b", "hf-ternary-bonsai-8b"], "an entry names a RoPE of its own that the owner did not choose");
for (const id of ["hf-ternary-bonsai-2-27b-thinking", "hf-ternary-bonsai-2-27b"]) assert.deepEqual(byId(id).options, qwen35, id);
assert.ok(!("rope_scaling" in qwen35) && !("rope_scaling" in QWEN3_OWN_BOS) && !("rope_scaling" in QWEN3_FROM_IM_START));
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
assert.deepEqual(lineOf("bartowski/SmolLM2-135M-Instruct-GGUF"), ["Hugging Face SmolLM2 135M Instruct (GGUF)"]);
assert.deepEqual(lineOf("unsloth/Llama-3.2-1B-Instruct"), ["Meta Llama 3.2 1B Instruct (copy)"]);
assert.ok(lineOf("meta-llama/Llama-3.2-1B-Instruct").includes("Meta Llama 3.2 1B Instruct"), "the original's line has the plain name");
// every GGUF of the list names its original, is under the original's license (its card's, the same: T136) and is
// "(GGUF)" on its line (the review of T136: a copy's word, another license or no original went through before)
for (const entry of MODELS.filter((entry) => entry.hf?.weights?.endsWith(".gguf"))) {
  assert.ok(entry.original, `${entry.id}: a GGUF without its original`);
  assert.deepEqual(LICENSES[entry.hf.repo], LICENSES[entry.original], `${entry.id}: the GGUF's license is not its original's`);
  assert.ok(lineOf(entry.hf.repo).includes(`${entry.name} (GGUF)`), `${entry.id}: the GGUF's line does not say (GGUF)`);
}
// T128: the list's order (the owner's): the groups as GROUPS has them, and within each the ones that write Japanese
// from light to heavy, then the English-only ones from light to heavy. The first is tiny-lm, the lightest
assert.equal(MODELS[0].id, "tiny-lm", "the lightest comes first");
// T511.6: the default is one of the list, named in one place, and small enough to start unasked (remembered.ts and
// list.ts start nothing past 500 MB that is not kept; index.astro falls back on the default)
assert.equal(DEFAULT_MODEL?.id, "hf-shisa-v2.1c-lfm2-350m", "the default is Shisa V2.1c LFM2 350M (the owner, 2026-10-15)");
assert.ok((DEFAULT_MODEL.download ?? DEFAULT_MODEL.bytes) <= 500e6, "the default is started unasked: no more than 500 MB");
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
// T537 (the owner's choice of 2026-10-15): a note's languages are its card's (docs/notes/t537-note-languages-2026-10-15.md).
// Spark-X2.5's GGUF cards say English and Chinese: among the English ones, with the thinkingAndNot() note's form
for (const id of ["hf-spark-x2.5-1.7b-thinking", "hf-spark-x2.5-1.7b", "hf-spark-x2.5-4b-thinking", "hf-spark-x2.5-4b"]) {
  assert.match(byId(id).note, /^(thinks before it answers|answers at once) · English \/ 中文 · fetches /, id);
  assert.ok(!writesJapanese(byId(id)), id);
}
// T511.4 (the owner's choice of 2026-10-15): "desktop only" and "Chrome and Firefox" are the page's rules, not a hand's.
// Every note ends with them where the rule says, and with nothing of them where it does not: "desktop only" where a
// device that says 4 GB is warned (memoryWarning()), "Chrome and Firefox" where what is loaded passes 4 GiB
{
  const DEVICES = /( · desktop only)?( · Chrome and Firefox)?$/;
  for (const entry of MODELS) {
    const [, desktop, browsers] = DEVICES.exec(entry.note);
    assert.equal(Boolean(desktop), memoryWarning(entry, 4) !== "", `${entry.id}: "desktop only" where a 4 GB device is warned, and only there`);
    assert.equal(Boolean(browsers), modelBytes(entry) > 2 ** 32, `${entry.id}: "Chrome and Firefox" where it loads more than 4 GiB, and only there`);
    assert.equal(entry.note.split(/ · desktop only| · Chrome and Firefox/).length, 1 + Boolean(desktop) + Boolean(browsers), `${entry.id}: the words once each, at the note's end`);
  }
  // no part of the list writes them itself (the rule adds them; written by hand, a note would say them twice)
  for (const entry of [...BUILT, ...HF_JAPANESE, ...HF_CLASSICS, ...HF_FAMILIES]) {
    assert.ok(!/desktop only|Chrome and Firefox/.test(entry.note), `${entry.id}'s note says "desktop only" or "Chrome and Firefox" by hand: devicesOf() adds them`);
  }
  // the lines a second time, by the entries on either side of each (TODO.md's T511.4): the largest without and the
  // smallest with, at the sizes the notes say
  const said = (id) => DEVICES.exec(byId(id).note).slice(1).map(Boolean);
  assert.deepEqual(said("hf-gpt2-xl"), [false, false], "1.8 GB: a 4 GB phone is not warned");
  assert.deepEqual(said("hf-japanese-large-lm-1.7b"), [true, false], "1.9 GB: a 4 GB phone is warned");
  assert.deepEqual(said("stories42M"), [false, false], "47 MB (it said desktop only from the days of float32 on NumPy)");
  assert.deepEqual(said("hf-granite-4.2-3b"), [true, false], "4.1 GB: Safari takes it in six bits");
  assert.deepEqual(said("hf-llm-jp-3-3.7b-instruct3"), [true, true], "4.3 GB: past 4 GiB");
}
console.log("ok");
