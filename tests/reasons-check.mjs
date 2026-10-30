// tests/reasons-check.mjs (T.380.1): the table of why a model is not on the GPU (src/runtime/forward/reasons.ts), by
// itself: no engine, no worker, no GPU. Node only, under a second:
//
//   node tests/reasons-check.mjs
//
// What is held:
//   - every sentence of the table, whole (SAID below, the second copy: a visitor reads these after "prompts on the CPU
//     (", so a word changed here is changed on purpose, and the owner has seen it);
//   - one cause, one sentence: no two entries say the same, every entry is read by a source, and no source writes an
//     entry's words itself;
//   - who may read the table: forward.ts's parts, the GPU's worker and the pages; not the model's worker, whose entry
//     would take the module with it and leave forward.ts's chunk importing the worker's own file (and forward.ts's
//     parts read neither failure.ts nor updated.ts, for the same reason);
//   - /benchmark/'s few words for a cell the GPU did not run (src/bench/path.ts's gpuSkipped()) are chosen by which
//     reason it is, not by words found in it.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { untyped } from "./untyped.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const { REASONS } = await import("../src/runtime/forward/reasons.ts");
const { gpuSkipped } = await import("../src/bench.ts");

// ---- the sentences, each whole: [the entry, what it is handed (a function's), the sentence]
const SAID = [
  ["linear", "linear-attention layers are not on the GPU yet"],
  ["rotatedBlock", "a rotated basis of blocks of this size is not on the GPU"],
  ["stateful", "layers that keep a state are not on the GPU yet"],
  ["pair", "this pair of norms and FFN is not on the GPU yet"],
  ["notIsolated", "the page is not cross-origin isolated"],
  ["oneThread", "one thread was asked for"],
  ["notShared", "the model's memory is not shared"],
  ["heads", "heads of a size that is no multiple of 4 are not on the GPU"],
  ["floats", "float32 weights are not on the GPU yet"],
  ["layersRoom", [72_600_000], "the layers on the GPU as well (73 MB) would not leave this device enough memory"],
  ["floatClassifier", "a classifier of float weights is not on the GPU's tokens"],
  ["classifierRoom", [189_400_000], "the classifier on the GPU as well (189 MB with the layers) would not leave this device enough memory"],
  ["stops", [8], "more than 8 stop tokens"],
  ["noWebGpu", "no WebGPU in a worker here"],
  ["noAdapter", "no GPU adapter here"],
  ["fallback", "a fallback adapter: the CPU in the GPU's place"],
  ["packed", ["ternary"], "ternary weights need the packed int8 dot product of WGSL, which this browser lacks"],
  ["aloneFallback", "a fallback adapter"],
  ["alonePositions", "a table of positions is not on the GPU alone yet"],
  ["aloneFfn", "an FFN without a gate is not on the GPU alone yet"],
  ["aloneNorms", "LayerNorms are not on the GPU alone yet"],
  ["aloneActivation", "a gate of the FFN that is no SiLU's is not on the GPU alone yet"],
  ["aloneLocal", "layers that attend a window are not on the GPU alone yet"],
  ["aloneDtype", ["int6"], "int6 weights stay on the CPU"],
  ["aloneHeads", "heads of a size that is no multiple of 4"],
  ["aloneStarts", "q, k and v or gate and up would not start where this GPU binds a buffer"],
  ["alonePast", "a layer's matrices are past a buffer of this GPU"],
  ["aloneState", "a layer's state is past a buffer of this GPU"],
  ["quiet", [180000], "the GPU said nothing for 180 s"],
  ["stalled", [10000], "the GPU's worker stopped answering for 10 s"],
  ["notStarted", ["Unexpected token"], "the GPU's worker did not start (Unexpected token)"],
  ["notStarted", [undefined], "the GPU's worker did not start (an error)"],
  ["letGo", "the model was let go"],
  ["noSteps", "the GPU did not take the steps"],
  ["notTaken", "the GPU did not take a block or a step"],
  ["blockFailed", "the GPU failed on a block of the prompt"],
  ["tokenFailed", "the GPU failed on a token"],
  ["keysInBlock", [64], "the GPU computed keys or values that are not finite numbers (NaN or infinity) in a block of the prompt at position 64"],
  ["keysInStep", [7], "the GPU computed keys or values that are not finite numbers (NaN or infinity) in a step at position 7"],
  ["logits", [9], "the GPU computed logits that are not finite numbers (NaN or infinity) at position 9"],
  ["sampledFew", [3, 4], "the GPU sampled 3 of 4 tokens"],
  ["sampledOutside", [70000, 32000], "the GPU sampled 70000, outside the vocabulary of 32000"],
  ["cpuFaster", [14.26, 812.4, 1203.7, 40, "200.4"], "the CPU as /benchmark/ measured it (14.3 GB/s): 812 ms against the GPU's 1204 ms for 40 tokens of prompts and 200 written"],
  ["cpuFaster", [undefined, 1, 2, 3, 4], "the CPU as /benchmark/ measured it (undefined GB/s): 1 ms against the GPU's 2 ms for 3 tokens of prompts and 4 written"],
  ["notReady", "not ready"],
  ["notEveryBlock", "the GPU did not take every block"],
];
const said = (entry, handed) => (typeof REASONS[entry] === "function" ? REASONS[entry](...handed) : REASONS[entry]);
for (const [entry, ...rest] of SAID) {
  const sentence = rest.at(-1), handed = rest.length === 2 ? rest[0] : [];
  assert.ok(Object.hasOwn(REASONS, entry), `the table has no entry ${entry}`);
  assert.equal(typeof REASONS[entry] === "function", rest.length === 2, `${entry} is ${rest.length === 2 ? "held as a sentence with numbers, and is none" : "held as a sentence, and takes numbers"}`);
  assert.equal(said(entry, handed), sentence, `the sentence of ${entry}: a visitor reads it, so it changes here as well, on purpose`);
}
assert.deepEqual(Object.keys(REASONS), [...new Set(SAID.map(([entry]) => entry))], "every entry of the table is held here, in the table's order (a new one: its sentence above)");
// one cause, one sentence: no two entries say the same of the same numbers
{
  const first = new Map();
  for (const [entry, ...rest] of SAID) {
    const sentence = rest.at(-1);
    assert.ok(!first.has(sentence) || first.get(sentence) === entry, `${entry} and ${first.get(sentence)} say the same sentence: one cause has one entry`);
    first.set(sentence, entry);
  }
}
console.log(`reasons-check: the table's ${Object.keys(REASONS).length} reasons, each sentence whole (${SAID.length} held), and no two the same`);

// ---- the sources: who reads the table, and that nobody writes its words
{
  const folders = ["src/runtime", "src/page", "src/benchmark", "src/bench"];
  const sources = folders.flatMap((folder) => fs.readdirSync(path.join(root, folder), { recursive: true })
    .filter((name) => /\.(?:ts|js)$/.test(name) && !/\.d\.ts$/.test(name)).map((name) => `${folder}/${String(name).split(path.sep).join("/")}`)).sort();
  const TABLE = "src/runtime/forward/reasons.ts";
  assert.ok(sources.includes(TABLE) && sources.length > 100);
  const code = (name) => untyped(fs.readFileSync(path.join(root, name), "utf8"), name).replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "").replace(/\s\/\/ .*$/gm, "");
  const texts = new Map(sources.map((name) => [name, code(name)]));
  const reads = (name, what) => new RegExp(`from "[^"]*${what.replace(/[./]/g, "\\$&")}"`).test(texts.get(name));
  const readers = sources.filter((name) => reads(name, "/reasons.ts"));
  // (the worker's entry and its modules: a module they import statically is put into the entry, and forward.ts's
  // chunk, fetched later, would import the worker's file for it; /benchmark/'s sections' worker the same)
  const entries = (name) => /^src\/runtime\/(?:worker\.ts|worker\/|benchmark\/sections\.ts|updated\.ts|built\.ts|failure\.ts|kept\.ts)/.test(name);
  for (const name of readers) assert.ok(!entries(name), `${name} reads the table of the reasons: it is of forward.ts's chunk, and the worker asks the engine (engine.gpuWhyNot)`);
  assert.deepEqual(readers, ["src/bench/path.ts", "src/benchmark/gpu.ts", "src/runtime/forward/alone.ts", "src/runtime/forward/gpufit.ts", "src/runtime/forward/gpuside.ts", "src/runtime/forward/linear.ts", "src/runtime/forward/paths.ts", "src/runtime/gpu/device.ts"].sort((a, b) => sources.indexOf(a) - sources.indexOf(b)),
    "who reads the table (a new reader: named here, and never one of the model's worker)");
  for (const name of sources.filter((one) => one.startsWith("src/runtime/forward"))) {
    assert.ok(!reads(name, "/failure.ts") && !reads(name, "/updated.ts"), `${name} reads failure.ts or updated.ts: a module of forward.ts's chunk reads nothing the worker's entry reads too`);
  }
  // every entry is read somewhere, by its name
  const used = new Set(readers.flatMap((name) => [...texts.get(name).matchAll(/\bREASONS\.(\w+)/g)].map(([, entry]) => entry)));
  for (const entry of Object.keys(REASONS)) assert.ok(used.has(entry), `no source reads REASONS.${entry}: a reason nobody gives is not one`);
  for (const entry of used) assert.ok(Object.hasOwn(REASONS, entry), `a source reads REASONS.${entry}, which the table has not`);
  // nobody writes an entry's words: the sentence, or of one with numbers the longest stretch of words between them
  const words = fs.readFileSync(path.join(root, TABLE), "utf8");
  const whole = [], stretches = [];
  for (const [entry, ...rest] of SAID) {
    if (rest.length === 1) whole.push(rest[0]);
    else {
      const source = new RegExp(`\\b${entry}: \\([^)]*\\) =>\\s*(?:notFiniteKV\\()?\`([^\`]*)\``).exec(words);
      const stretch = (source?.[1] ?? "").split(/\$\{[^}]*\}/).sort((a, b) => b.length - a.length)[0];
      assert.ok(stretch.length >= 12, `the words of ${entry} were not found in the table`);
      stretches.push(stretch);
    }
  }
  // (the one sentence that is written twice: the shaders' samplingSettings() throws the same words where it is handed
  // more stop tokens than it holds, an error of its own that the GPU's side never lets it come to; the shaders' parts
  // read nothing of forward.ts's)
  const TWICE = { "src/runtime/shaders/run.ts": [" stop tokens"] };
  for (const [name, text] of texts) {
    if (name === TABLE) continue;
    for (const sentence of new Set(whole)) {
      // (as a whole string: "not ready" is also the beginning of the worker's "not ready after N s", another cause)
      const at = [`"${sentence}"`, `'${sentence}'`, `\`${sentence}\``].find((one) => text.includes(one));
      assert.ok(!at, `${name} writes ${at}: that reason's words are in ${TABLE} alone`);
    }
    for (const stretch of new Set(stretches)) assert.ok(TWICE[name]?.includes(stretch) || !text.includes(stretch), `${name} writes "${stretch}": that reason's words are in ${TABLE} alone`);
  }
  const written = new Set([...whole, ...stretches]);
  console.log(`reasons-check: read by ${readers.length} sources (none of the model's worker), every entry by its name; ${written.size} sentences written in the table alone`);
}

// ---- /benchmark/'s words for a cell the GPU did not run: by which reason it is
assert.deepEqual([REASONS.noWebGpu, REASONS.noAdapter, REASONS.fallback, REASONS.heads, "", undefined].map((why) => gpuSkipped(why)),
  ["not here", "not here", "not on a fallback adapter", REASONS.heads, "not measured", "not measured"]);
// (a reason that only has such words in it is said as it is: it is another cause)
for (const why of [`the layers did not go up to the GPU (${REASONS.noAdapter})`, "a fallback adapter", `${REASONS.fallback}, as it said`]) assert.equal(gpuSkipped(why), why);
console.log("reasons-check: a cell the GPU did not run is said by which reason it is");
console.log("reasons-check: ok");
