// tests/repository-check.mjs (T505.3)
// What the page makes of a repository of huggingface.co nobody listed (src/page/repository.ts, and hfEntry() of
// src/page/address.ts): which file of it is the model. Node imports the two as the modules they are; the list of a
// repository's files is a made-up answer of the API, or the names of a real one (tests/fixtures/repository-files.json).
//
//   node tests/repository-check.mjs
//
// Held here: the order of the quantizations (the second copy of the table: ORDER), which words of a name say one, the
// files that are never taken (the parts of a split model, a projector, an importance matrix, what is under a folder),
// safetensors before any GGUF, a named file before both, the sentence of a repository with nothing to open (whole),
// what is asked of huggingface.co and what becomes of an answer that is none, the address a file may be named in,
// and the name a conversion is kept under (src/runtime/kept.ts).
// Not seen here: a browser (the sheet, the URL, the Service Worker: tests/e2e.mjs with hf:<owner>/<repository>), and
// whether huggingface.co lets a page read its list (CORS).
import assert from "node:assert/strict";
import fs from "node:fs";
import { QUANTIZATIONS, SAFETENSORS, bestGguf, filesOf, looked, noteOf, quantization, settled, unnamed } from "../src/page/repository.ts";
import { keptName, keptNames } from "../src/runtime/kept.ts";
// (the page's module reads its address where it is loaded: there is none here)
globalThis.location ??= { search: "" };
const { FILE, hfEntry, unlisted } = await import("../src/page/address.ts");

// ---- the order, written a second time: the owner's four (2026-10-13: Q8_0, else Q6_K, Q5_K_M, Q4_K_M) keep their
// order among themselves; the floats and a ternary model's own types stand right after Q8_0; what is below four bits
// comes last
const ORDER = ["Q8_0", "PQ2_0", "PTQ1_0", "F16", "BF16", "F32", "Q6_K", "Q6_K_L", "Q6_K_XL", "Q5_K_M", "Q5_K_L", "Q5_K_XL", "Q4_K_M",
  "Q5_K_S", "Q5_K", "Q5_1", "Q5_0", "Q4_K_L", "Q4_K_XL", "Q4_K_S", "Q4_K", "Q4_1", "Q4_0", "IQ4_NL", "IQ4_XS",
  "Q3_K_L", "Q3_K_XL", "Q3_K_M", "Q3_K", "Q3_K_S", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS",
  "Q2_K_L", "Q2_K_XL", "Q2_K", "Q2_K_S", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S"];
assert.deepEqual(QUANTIZATIONS.map(([first]) => first), ORDER, "the order of the quantizations (QUANTIZATIONS of src/page/repository.ts) is not the one written here: " +
  "which file a visitor's repository is opened by changed. Edit both, and tell the owner (the first of them are his order)");
const owners = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"].map((word) => ORDER.indexOf(word));
assert.deepEqual(owners, [...owners].sort((a, b) => a - b), "the owner's order (Q8_0, Q6_K, Q5_K_M, Q4_K_M) is not kept");
assert.equal(owners[0], 0, "Q8_0 is not the first taken");
assert.equal(new Set(QUANTIZATIONS.flat()).size, QUANTIZATIONS.flat().length, "a quantization is spelled in two rows");
console.log(`ok: ${ORDER.length} quantizations in the order written here, the owner's four in his`);

// ---- the word of a name
const place = (word) => ORDER.indexOf(word);
for (const [name, word] of [
  ["Qwen3-0.6B-Q8_0.gguf", "Q8_0"], ["qwen2.5-0.5b-instruct-q4_k_m.gguf", "Q4_K_M"], ["model.Q4_K_M.gguf", "Q4_K_M"], ["ggml-model-f16.gguf", "F16"],
  ["model-fp16.gguf", "F16"], ["model_Q6_K.gguf", "Q6_K"], ["Model.i1-IQ4_XS.gguf", "IQ4_XS"], ["locate-anything-q4_k.gguf", "Q4_K"],
  ["gemma-4-E2B_q4_0-it.gguf", "Q4_0"], ["main.bf16.gguf", "BF16"], ["Model-UD-Q4_K_XL.gguf", "Q4_K_XL"], ["Model-Q8_0-imat.gguf", "Q8_0"],
  // the last word of a name that says two: a model's own name may hold one
  ["Model-Q4K-Q8_0-F32.gguf", "F32"], ["Ternary-Bonsai-8B-PQ2_0.gguf", "PQ2_0"],
]) assert.equal(quantization(name), place(word), `${name} is not read as ${word}`);
// no word: none of the table as a word of its own (Q4_K is none of Q4_K_XS, Q8 none of Q8_0, F16 none of BF16 or F160)
for (const name of ["model.gguf", "Model-Q2.gguf", "Model-Q6_K_M.gguf", "Model-Q8_K_XL.gguf", "Bonsai-8B-Q1_0.gguf", "ModelQ8_0.gguf", "Model-F160.gguf", "Model-3.7bpw.gguf"]) {
  assert.equal(quantization(name), ORDER.length, `${name} is read as a quantization of the table`);
}
assert.equal(quantization("Model-BF16.gguf"), place("BF16"));
console.log("ok: a name says its quantization as a word of its own, in either case; a name that says none comes last");

// ---- which file of a list: made-up repositories
const listOf = (...names) => names.map((path, at) => ({ path, size: 1000 + at }));
const taken = (...names) => bestGguf(listOf(...names))?.path;
assert.equal(taken("m-Q4_K_M.gguf", "m-Q8_0.gguf", "m-Q6_K.gguf", "m-F16.gguf"), "m-Q8_0.gguf", "Q8_0 is not taken where there is one");
assert.equal(taken("m-Q4_K_M.gguf", "m-Q5_K_M.gguf", "m-Q6_K.gguf", "m-Q2_K.gguf"), "m-Q6_K.gguf", "without Q8_0, Q6_K");
assert.equal(taken("m-Q4_K_M.gguf", "m-Q5_K_M.gguf", "m-Q5_K_S.gguf"), "m-Q5_K_M.gguf", "without Q6_K, Q5_K_M");
assert.equal(taken("m-Q4_K_S.gguf", "m-Q4_K_M.gguf", "m-Q3_K_L.gguf", "m-IQ4_XS.gguf"), "m-Q4_K_M.gguf", "without Q5_K_M, Q4_K_M");
assert.equal(taken("m-Q3_K_M.gguf", "m-IQ3_M.gguf", "m-Q2_K.gguf", "m-IQ4_XS.gguf"), "m-IQ4_XS.gguf", "K-quants and i-quants alone: the most bits");
assert.equal(taken("m-IQ2_XXS.gguf", "m-IQ1_S.gguf"), "m-IQ2_XXS.gguf", "the lowest are taken where nothing else is there");
assert.equal(taken("m-F16.gguf", "m-Q6_K.gguf"), "m-F16.gguf", "a float comes after Q8_0 alone");
assert.equal(taken("m-F32.gguf", "m-F16.gguf", "m-BF16.gguf"), "m-F16.gguf", "of the floats, the smallest fetch");
// the parts of a split model, a projector and an importance matrix are no model to open
assert.equal(taken("m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf", "m-Q4_K_M.gguf"), "m-Q4_K_M.gguf", "a part of a split model was taken");
assert.equal(taken("m.Q8_0.gguf-00001-of-00009.gguf", "m.Q2_K.gguf"), "m.Q2_K.gguf", "a part of a split model (the other spelling) was taken");
assert.equal(taken("mmproj-m-F16.gguf", "m-mmproj.gguf", "m-Q4_0.gguf"), "m-Q4_0.gguf", "a projector was taken");
assert.equal(taken("imatrix.gguf", "imatrix_unsloth.gguf", "m-Q2_K-imatrix.gguf"), "m-Q2_K-imatrix.gguf", "an importance matrix was taken (or a model made with one was not)");
assert.equal(taken("m-Q8_0-00001-of-00002.gguf", "mmproj-F16.gguf", "imatrix.gguf", "m.gguf.part1of2", "README.md", "m.GGUF.txt"), undefined, "something that is no model was taken");
// a name without a word comes after every one with one; of two of one place, the first by its name
assert.equal(taken("model.gguf", "m-IQ1_S.gguf"), "m-IQ1_S.gguf");
assert.equal(taken("model.gguf"), "model.gguf", "the one GGUF of a repository is not taken because its name says no quantization");
assert.equal(taken("b-Q8_0.gguf", "a-Q8_0.gguf", "B-Q8_0.gguf"), "B-Q8_0.gguf");
assert.equal(taken(), undefined);
console.log("ok: of a made-up repository's files, the best GGUF by the order; never a part, a projector or an importance matrix");

// ---- the model the worker is handed
const REVISION = "0123456789abcdef0123456789abcdef01234567";
const entry = hfEntry("owner/model", "main"), pinned = hfEntry("owner/model", REVISION);
assert.deepEqual(entry.hf, { repo: "owner/model", revision: "main", weights: SAFETENSORS, config: "config.json" });
assert.ok(unnamed(entry) && entry.note === "Hugging Face" && pinned.note === "Hugging Face · 0123456789ab" && entry.download === undefined);
// safetensors where the repository has them, as it always was: the very entry (nothing of a GGUF beside them is looked at)
for (const has of [SAFETENSORS, `${SAFETENSORS}.index.json`]) {
  assert.equal(settled(entry, listOf("config.json", has, "m-Q8_0.gguf", "m-Q4_K_M.gguf")), entry, `a repository with ${has} and GGUFs is not opened by its safetensors`);
}
// (a file of that name under a folder is not the repository's)
assert.equal(settled(entry, listOf(`onnx/${SAFETENSORS}`, "m-Q4_K_M.gguf")).hf.weights, "m-Q4_K_M.gguf");
// a GGUF alone: the best there, with its size, and the note says which
{
  const found = settled(pinned, [{ path: "README.md", size: 10 }, { path: "m-Q4_K_M.gguf", size: 400e6 }, { path: "m-Q8_0.gguf", size: 700e6 }, { path: "Q8_0/m-Q8_0.gguf", size: 1 }]);
  assert.deepEqual(found.hf, { repo: "owner/model", revision: REVISION, weights: "m-Q8_0.gguf", config: "config.json" });
  assert.equal(found.download, 700e6, "the page does not know how much of a GGUF will be fetched");
  assert.equal(found.note, "Hugging Face · 0123456789ab · m-Q8_0.gguf");
  assert.ok(!unnamed(found) && found.name === "owner/model" && found.id === "local");
  assert.deepEqual({ ...found, hf: 0, note: 0, download: 0 }, { ...pinned, hf: 0, note: 0, download: 0 }, "more of the entry changed than its file, its note and what is fetched");
  // (under a folder alone: nothing at the top)
  assert.throws(() => settled(entry, [{ path: "Q8_0/m-Q8_0.gguf", size: 1 }]), { message: "owner/model has no model.safetensors and no GGUF file this page can read." });
}
// a file named (&file=, the sheet's third field): that file, whatever else is there
{
  const named = hfEntry("owner/model", "main", { file: "m-Q4_K_M.gguf", template: "{prompt}", prompt: "Hello" });
  assert.deepEqual(named.hf, { repo: "owner/model", revision: "main", weights: "m-Q4_K_M.gguf", config: "config.json" });
  assert.ok(!unnamed(named) && named.note === "Hugging Face · m-Q4_K_M.gguf" && named.template === "{prompt}" && named.prompt === "Hello");
  const found = settled(named, listOf(SAFETENSORS, "m-Q8_0.gguf", "m-Q4_K_M.gguf"));
  assert.equal(found.hf.weights, "m-Q4_K_M.gguf", "a file that was named is not the one opened (safetensors or a better GGUF was)");
  assert.equal(found.download, 1002);
  // one the repository does not have: the entry as it is, for the worker to say (the words of any file that is not there)
  assert.equal(settled(named, listOf(SAFETENSORS, "m-Q8_0.gguf")), named, "a named file that is not there is not left to the worker");
  assert.equal(settled(named, []), named);
  // a named file that is no GGUF is opened as the safetensors file it is taken for; nothing is said of its fetch
  const other = hfEntry("owner/model", "main", { file: "model.fp16.safetensors" });
  assert.equal(settled(other, listOf("model.fp16.safetensors")).download, undefined);
}
// nothing to open: the owner's sentence, whole (2026-10-13)
for (const files of [[], listOf("README.md", "config.json", "pytorch_model.bin"), listOf("model.fp16.safetensors"), listOf("mmproj-F16.gguf", "m-Q8_0-00001-of-00002.gguf")]) {
  assert.throws(() => settled(hfEntry("some-owner/some.model-GGUF", "main"), files), { name: "Error", message: "some-owner/some.model-GGUF has no model.safetensors and no GGUF file this page can read." },
    "the sentence of a repository with nothing this page opens changed (it is the owner's: show him first)");
}
assert.equal(noteOf("main", SAFETENSORS), "Hugging Face");
console.log("ok: safetensors first where a repository has them, else its best GGUF with its size, a named file before both, and the owner's sentence where there is nothing");

// ---- real repositories: the names huggingface.co listed, and the file written beside them by hand
{
  const real = JSON.parse(fs.readFileSync(new URL("fixtures/repository-files.json", import.meta.url), "utf8"));
  let count = 0;
  for (const [repository, { takes, files }] of Object.entries(real).filter(([name]) => name !== "_")) {
    const listed = files.map((path) => ({ path, size: 1 }));
    if (takes === null) {
      assert.throws(() => settled(hfEntry(repository, "main"), listed), { message: `${repository} has no model.safetensors and no GGUF file this page can read.` }, `${repository}: something of it is opened`);
    } else {
      assert.equal(settled(hfEntry(repository, "main"), listed).hf.weights, takes, `${repository} is not opened by ${takes}`);
    }
    count++;
  }
  assert.ok(count >= 10, "fewer than ten real repositories");
  console.log(`ok: ${count} real repositories are opened by the file written beside their names`);
}

// ---- what is asked of huggingface.co: one request, at the entry's revision; an answer that is none leaves the entry as it is
{
  const asked = [];
  const answering = (answer) => async (url, init) => {
    asked.push([url, init]);
    return answer();
  };
  const json = (value, status = 200) => () => new Response(JSON.stringify(value), { status });
  const tree = [{ type: "directory", path: "Q8_0", size: 0 }, { type: "file", path: "m-Q8_0.gguf", size: 7, lfs: { size: 7 } }, { type: "file", path: "README.md", size: 3 }, { type: "file" }, null];
  assert.deepEqual(await filesOf("owner/model", REVISION, answering(json(tree))), [{ path: "m-Q8_0.gguf", size: 7 }, { path: "README.md", size: 3 }]);
  assert.deepEqual(asked.map(([url, init]) => [url, init?.signal instanceof AbortSignal]), [[`https://huggingface.co/api/models/owner/model/tree/${REVISION}`, true]], "the list of a repository is asked for otherwise than by one plain request of its tree at the revision");
  const found = await looked(pinned, answering(json(tree)));
  assert.equal(found.hf.weights, "m-Q8_0.gguf");
  assert.equal(asked.length, 2, "a repository is looked at with more than one request");
  // no such repository (or a gated one), an answer that is no list, no line: the entry as it was, and the worker says
  // what is wrong with it. A line that fails is said on the console, once
  const warned = [], warn = console.warn;
  console.warn = (...words) => warned.push(words.join(" "));
  try {
    for (const answer of [json({ error: "Repository not found" }, 404), json({ error: "gated" }, 401), json({ not: "a list" }), () => new Response("<html>", { status: 200 })]) {
      assert.equal(await looked(pinned, answering(answer)), pinned, "an answer that is no list changed the entry");
    }
    assert.deepEqual(warned.filter((line) => !/could not be listed \(SyntaxError/.test(line)), [], "a refusal of huggingface.co is said on the console (the worker says it to the visitor)");
    warned.length = 0;
    assert.equal(await looked(pinned, answering(() => { throw new TypeError("Failed to fetch"); })), pinned);
    assert.deepEqual(warned, ["repository: the files of owner/model could not be listed (TypeError: Failed to fetch): opened as named"]);
  } finally {
    console.warn = warn;
  }
  // a line that does not answer is let go: the signal of the request is one that ends, and the entry goes on as named
  {
    let signal;
    const hanging = (url, init) => new Promise((resolve, reject) => { signal = init?.signal; signal?.addEventListener("abort", () => reject(signal.reason)); });
    const quiet = console.warn;
    console.warn = () => {};
    try {
      const waiting = looked(pinned, hanging);
      assert.ok(signal instanceof AbortSignal, "the list of a repository is asked for with no signal to end the wait");
      assert.equal(signal.aborted, false);
      signal.dispatchEvent(new Event("abort"));
      assert.equal(await waiting, pinned);
    } finally {
      console.warn = quiet;
    }
  }
  // the sentence of a repository with nothing comes through looked() as it is
  await assert.rejects(looked(entry, answering(json([{ type: "file", path: "README.md", size: 1 }]))), { message: "owner/model has no model.safetensors and no GGUF file this page can read." });
  console.log("ok: one request lists a repository; without a list the model is opened as named, and a line that failed is said once on the console");
}

// ---- the address a file may be named in (&file=, and the sheet's field by the same pattern)
for (const name of ["m-Q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf", "Q8_0/m.gguf", "a.b/c_d/e-f.gguf", "model.fp16.safetensors", "..gguf"]) assert.ok(FILE.test(name), `${name} is no file of an address`);
for (const name of ["", "../m.gguf", "a/../m.gguf", "a/..", ".", "./m.gguf", "/m.gguf", "m.gguf/", "a//b.gguf", "m gguf", "m.gguf?x=1", "m.gguf#x", "https://example.com/m.gguf", "a\\b.gguf", "m%2Fgguf"]) {
  assert.ok(!FILE.test(name), `${JSON.stringify(name)} is taken for a file of an address`);
}
// (the sheet hands the same source to the field's pattern, which a browser compiles with the flag v)
new RegExp(`^(?:${FILE.source})$`, "v");
assert.equal(unlisted(entry), entry);
assert.equal(unlisted({ id: "tiny-lm", name: "x", hf: { repo: "a/b" } }), undefined, "a model of the list is taken for one nobody listed");
assert.equal(unlisted({ id: "local", name: "x", hf: { files: [] } }), undefined, "a folder of the disk is taken for a repository");
assert.equal(unlisted(undefined), undefined);
console.log("ok: a file is named in an address by names with / between them, none of them dots alone");

// ---- what a conversion is kept under (src/runtime/kept.ts): a repository nobody listed, by the file it was opened by,
// and by nothing more where that is the model.safetensors it always was; a model of the list as it always was
{
  const names = (model) => keptNames(model).map(decodeURIComponent);
  assert.deepEqual(names(pinned), [`owner/model@${REVISION}:int8:4096`, `owner/model@${REVISION}:int6:4096`], "the name ?hf= kept a repository of safetensors under changed: what visitors kept is fetched again");
  const q8 = settled(pinned, listOf("m-Q8_0.gguf")), q4 = hfEntry("owner/model", REVISION, { file: "m-Q4_K_M.gguf" });
  assert.deepEqual(names(q8), [`owner/model@${REVISION}/m-Q8_0.gguf:int8:4096`, `owner/model@${REVISION}/m-Q8_0.gguf:int6:4096`]);
  assert.equal(new Set([pinned, q8, q4].map((model) => keptName(model))).size, 3, "two files of one repository share what is kept of one of them");
  // a model of the list names its GGUF too, and is kept as before (T136, T203: its name is its repository's)
  const listed = { id: "hf-some-model", hf: { repo: "owner/model", revision: REVISION, weights: "m-Q8_0.gguf" } };
  assert.deepEqual(names(listed), [`owner/model@${REVISION}:int8:4096`, `owner/model@${REVISION}:int6:4096`], "a model of the list is kept under another name than it was: every visitor's conversion of it is fetched again");
  console.log("ok: a conversion of ?hf= is kept by its repository, revision and file; of model.safetensors and of the list, as it always was");
}
console.log("repository-check: all checks passed");
