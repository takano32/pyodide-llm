// T.357.3: the parts of tests/gpu-check.mjs that need no GPU, no browser and no Pyodide, each by itself on made-up
// numbers. gpu-check itself runs in CI alone (gpu-prompt.yml: WebGPU is not run on the development machine), so until
// this check nothing of it ran anywhere else: a verdict that passed a wrong GPU, or a line of the log that lost a
// column, showed in a run of twenty minutes or not at all.
//   - measures.mjs: the worst row, the scale, the KL divergence, on numbers worked by hand;
//   - models.mjs: which made-up models an engine runs, and that every name it sorts by is a model;
//   - verdicts.mjs: a model's lines and whether they failed, for runs that are as they should be and for runs that are
//     wrong in one thing each (a layer past its line, a NaN, a scale, another token, tokens the GPU did not take), by
//     the arithmetic of the run's form (float32, f16, the packed shaders against Q8); the quantizers' line; the lines of
//     a late GPU, a refused adapter, the shaders remembered, the GPU alone, a weight that is no finite number;
//   - steps.mjs: the steps left on the CPU for a reason that is right or not, and a generation's steps as NumPy's and
//     wrong in one thing each;
//   - tables.mjs: the tables' columns and rows, a line a run, the seconds;
//   - harness.js: it is a module Node can read, the runtime's files it imports are in this tree, and the one rewriting
//     Dawn's worker thread needs (engines.mjs) reaches each of them;
//   - reference.mjs's Python, in the native Python with NumPy alone: the cache that rounds to float16 and the matrix
//     that takes its input in 8 bits.
// What it does not run: reference.mjs's cases (Pyodide and the engine) and engines.mjs (a browser, or Dawn); and no
// number here says that a shader is right.
//
//   node tests/gpu-check-parts-check.mjs      (PYTHON=.venv/bin/python; about a second)
import assert from "node:assert/strict";
import fs from "node:fs";
import { execFileSync, spawnSync } from "node:child_process";
import { runtime } from "./tree.mjs";
import { floats, worstRow, worstAt, argmax, logSoftmax, divergence, scaleOff } from "./gpu-check/measures.mjs";
import { NAN_MODELS, NAN_CHOICES, SYNTHETIC, DAWN_ONLY, madeUpOn } from "./gpu-check/models.mjs";
import { GEN, GPU_LINE, HALF_LINE, K, K_PACKED, SCALE_LINE, TIE } from "./gpu-check/lines.mjs";
import { promptsOnGpu, quantizersWrong, modelWrong } from "./gpu-check/verdicts.mjs";
import { unbound, stepsRight } from "./gpu-check/steps.mjs";
import { layerTables } from "./gpu-check/tables.mjs";
import { PYTHON } from "./gpu-check/reference.mjs";

let checks = 0;
const check = (name, body) => {
  body();
  checks += 1;
};
// what a part printed, and what it returned
const said = (body) => {
  const log = console.log, lines = [];
  console.log = (...parts) => lines.push(parts.join(" "));
  try {
    const value = body();
    return { value, text: lines.join("\n"), lines };
  } finally {
    console.log = log;
  }
};
const b64 = (values) => Buffer.from(Float32Array.from(values).buffer).toString("base64");
// made-up numbers that are the same every time (a small linear congruential generator), in [-1, 1)
const numbers = (count, seed) => {
  let state = seed >>> 0;
  return Float32Array.from({ length: count }, () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 31) - 1);
};

// ---- measures.mjs
check("the measures", () => {
  assert.deepEqual([...floats(b64([1.5, -2, 0.25]))], [1.5, -2, 0.25]);
  // two rows of two: the first off by 0.5 of a largest 2, the second by 0.25 of a largest 1
  assert.equal(worstRow([2, 1.5, 1, 0.75], [2, 1, 1, 1], 2), 0.25);
  assert.equal(worstRow([2, 1.5, 1, 0.75], [2, 1, 1, 0.5], 2), 0.25);
  assert.equal(worstRow([4, 0, 1, 1], [2, 0, 1, 1], 2), 1);
  // a row of zeros is held by its difference itself, and a NaN is the worst of all only where the caller asks so:
  // Math.max keeps it
  assert.equal(worstRow([0.5, 0], [0, 0], 2), 0.5);
  assert.ok(Number.isNaN(worstRow([NaN, 0], [1, 0], 2)));
  assert.equal(argmax([1, 3, 3, 2]), 1);
  // the worst layer: the largest ratio (the first of equals), and a NaN wherever it is (the first NaN; a comparison with
  // it is false, which let the layer after a NaN take its place: T.357.3's review)
  assert.equal(worstAt([0.5, 0.9, 0.3]), 1);
  assert.equal(worstAt([0.5, 0.5, 0.3]), 0);
  assert.equal(worstAt([0.5, NaN, 0.3]), 1);
  assert.equal(worstAt([NaN, 0.1, 9]), 0);
  assert.equal(worstAt([0.1, NaN, NaN, 9]), 1);
  assert.equal(worstAt([0.1, 9, NaN]), 2);
  assert.equal(worstAt([0.4]), 0);
  const p = logSoftmax([0, Math.log(3)]);
  assert.ok(Math.abs(Math.exp(p[0]) - 0.25) < 1e-12 && Math.abs(Math.exp(p[1]) - 0.75) < 1e-12);
  assert.equal(divergence(p, [5, 5 + Math.log(3)]), 0);
  // KL(p || q) of (1/4, 3/4) from (1/2, 1/2)
  assert.ok(Math.abs(divergence(p, [0, 0]) - (0.25 * Math.log(0.5) + 0.75 * Math.log(1.5))) < 1e-12);
  assert.ok(Math.abs(scaleOff([2.02, -1.01, 3.03], [2, -1, 3]) - 0.01) < 1e-12);
  assert.equal(scaleOff([2, -1, 3], [2, -1, 3]), 0);
});

// ---- models.mjs
check("the made-up models an engine runs", () => {
  const all = Object.keys(SYNTHETIC);
  assert.deepEqual(madeUpOn("dawn"), all);
  assert.deepEqual(madeUpOn("chromium"), all.filter((name) => !DAWN_ONLY.includes(name)));
  for (const browser of ["chrome", "msedge"]) assert.deepEqual(madeUpOn(browser), NAN_MODELS);
  for (const name of [...DAWN_ONLY, ...NAN_MODELS]) assert.ok(all.includes(name), `${name} is sorted by and is no model`);
  assert.ok(NAN_MODELS.every((name) => !DAWN_ONLY.includes(name)), "a model of the browsers' three is Dawn's alone");
  assert.deepEqual(Object.keys(NAN_CHOICES), ["all", "made-up", "small", "none"]);
  assert.deepEqual(NAN_CHOICES.small.models, NAN_MODELS);
  // a model is [the maker's form, the engine's options, and what its run is forced to, if anything]
  for (const [name, model] of Object.entries(SYNTHETIC)) {
    assert.ok(model.length === 2 || model.length === 3, name);
    assert.ok(model.every((part) => part && typeof part === "object"), name);
    assert.equal(model[0].dtype ?? "int8", model[1].dtype ?? "int8", `${name}: the file's dtype and the engine's`);
    assert.equal(model[0].arch ?? "llama", model[1].arch ?? "llama", `${name}: the file's architecture and the engine's`);
  }
});

// ---- lines.mjs
check("the numbers a run is held to are the ones decided", () => {
  // each is a decision with its derivation in lines.mjs (T.187: the lines are the arithmetic's own distances): loosening one
  // lets a wrong GPU through, so changing one is changing this too, and saying why in TASKS.md
  assert.deepEqual({ GPU_LINE, HALF_LINE, K, K_PACKED, SCALE_LINE, TIE }, { GPU_LINE: 8e-3, HALF_LINE: 1.5e-2, K: { float32: 4, f16: 8 }, K_PACKED: 3, SCALE_LINE: 1e-4, TIE: Math.LN2 });
});

// ---- a made-up case: 2 layers, 3 positions of the prompt, keys and values 8 wide, a vocabulary of 6
const LAYERS = 2, N = 3, KV = 8, VOCAB = 6, HEADER = [8, 16, LAYERS, 2, 2, VOCAB, 16];
const SIZE = LAYERS * N * KV;
// (every row begins with 1 and -1: its largest value is 1)
const rows = (count, seed) => numbers(count, seed).map((x, i) => (i % KV === 0 ? 1 : i % KV === 1 ? -1 : x));
const KEYS = rows(SIZE, 1), VALUES = rows(SIZE, 2);
// `off` added to the first two values of every row of the layers given (every layer's rows are `positions` × KV values):
// the worst row is `off` away, and the scale is the same (what the 1 gains, the -1 loses)
const moved = (from, off, layers = [0, 1], positions = N) => from.map((x, i) => (layers.includes(Math.floor(i / (positions * KV))) && i % KV < 2 ? x + off : x));
const E16 = 1e-3, Q8 = 2e-2;
const LOGITS = [0, 1, 4, 2, 3.9, -1];   // most likely 2; 4 is a near tie (0.1 below), 3 is not (2 below)
// the greedy steps: every step's most likely token is its own (step i gives i % VOCAB + ... below)
const GREEDY = [2, 4, 1, 3, 0, 5, 2, 4];
const greedyRows = () => {
  // row 0 is the prompt's last token's logits (it gives GREEDY[0]), row i gives GREEDY[i]
  const rows = new Float32Array(GEN * VOCAB).fill(-2);
  GREEDY.forEach((id, step) => { rows[step * VOCAB + id] = 6; });
  // (row 0: the token a penalty of 100 on its first id leaves the largest is 3, and its nucleus at temperature 2 is wide)
  rows.set([1, 2, 6, 5, 4.5, 0], 0);
  return rows;
};
const GREEDY_KEYS = rows(LAYERS * GEN * KV, 7), GREEDY_VALUES = rows(LAYERS * GEN * KV, 8);
const REFERENCE = { tokens: [1, 2, 3, 4], header: HEADER, keys: b64(KEYS), values: b64(VALUES),
  keys16: b64(moved(KEYS, E16)), values16: b64(moved(VALUES, E16)), keys8: b64(moved(KEYS, Q8)), values8: b64(moved(VALUES, Q8)),
  logits: b64(LOGITS), logits16: b64(LOGITS.map((x) => x + 1e-3)), logits8: b64(LOGITS.map((x, i) => x + 0.01 * i)),
  greedy: GREEDY, greedyLogits: b64(greedyRows()), greedyKeys: b64(GREEDY_KEYS), greedyValues: b64(GREEDY_VALUES) };
// (T.545: plan.layers[0].kind, the kind of the first layer, which verdicts.mjs reads: a Llama's attends)
const CASE = { id: "made-up", reference: REFERENCE, plan: { layers: [{ kind: "attention" }, { kind: "attention" }] }, headDim: 0, arch: "llama", dtype: "int8", numpySeconds: 1.25, planSeconds: 0.5 };
const CPU = { keys: b64(moved(KEYS, 0.1)), values: b64(moved(VALUES, 0.1)), logits: b64(LOGITS.map((x, i) => x + 0.05 * i)), promptMs: 6, seconds: 2 };
// a run of the GPU: its keys and values `off` from `from` (NumPy's by default), block by block and all at once
const runOf = ({ form = "TF.js's tiles", off = 1e-3, keys = KEYS, values = VALUES, layers, logits = LOGITS, ...more } = {}) => {
  const answer = () => ({ gpuTokens: N, keys: b64(moved(keys, off, layers)), values: b64(moved(values, off, layers)), logits: b64(logits) });
  return { ...answer(), again: answer(), past: { gpuTokens: 0 }, note: "prompts on WebGPU", form, attention: "llama.cpp flash attention tiles",
    promptMs: 3, seconds: 4, readySeconds: 1.5, ...more };
};
const model = (runs, more = {}, c = CASE) => said(() => modelWrong([c], { id: "made-up", cpu: CPU, gpu: runs, nanSeconds: { rows: 0, values: 0 }, ...more }));
// the line of the one run, without the tables after it
const runLine = (text) => text.split("\n\n")[0].split("\n").slice(1).join("\n");

// ---- verdicts.mjs: the prompt's keys and values, scale and logits
check("a model's first line says the CPU's, E16 and Q8", () => {
  const { value, lines } = model([runOf()]);
  assert.equal(value, false, lines.join("\n"));
  assert.match(lines[0], /^made-up \(2 layers, 2 heads, 2 of keys and values, 3 tokens\): the CPU's keys and values 1\.00e-1 from NumPy's/);
  assert.match(lines[0], /most likely 2 as NumPy's, the prompt 2\.00 ms a token; E16 \(NumPy's cache in float16\) 1\.00e-3 /);
  assert.match(lines[0], /Q8 \(and its matrices' inputs in 8 bits\) 2\.00e-2, the first layer 2\.00e-2 /);
});
check("a run as it should be passes, by its form's arithmetic", () => {
  // float32: the shader's line (E16 is small here: K × E16 is under it)
  assert.ok(K.float32 * E16 < GPU_LINE && GPU_LINE < HALF_LINE);
  const float = model([runOf({ off: GPU_LINE * 0.9 })]);
  assert.equal(float.value, false, float.text);
  assert.match(runLine(float.text), /^ {2}TF\.js's tiles, llama\.cpp flash attention tiles: keys and values 7\.20e-3 \(all at once 7\.20e-3, the first layer 7\.20e-3 from NumPy's/);
  assert.match(runLine(float.text), /; 7\.2 E16, 0\.36 Q8; layer [01] 0\.90 of its line\), the first layer's scale /);
  assert.doesNotMatch(float.text, /FAILED/);
  // f16: its line is HALF_LINE, and what passes there fails a float32 form
  const off = (GPU_LINE + HALF_LINE) / 2;
  assert.equal(model([runOf({ form: "llama.cpp's f16", off })]).value, false);
  assert.equal(model([runOf({ form: "TF.js's tiles", off })]).value, true);
  // the packed shaders: all layers by K_PACKED × Q8's distance from NumPy's, the first layer against Q8's own
  const packed = { form: "ORT's DP4A", keys: floats(REFERENCE.keys8), values: floats(REFERENCE.values8) };
  const eight = model([runOf({ ...packed, off: 1e-3 })]);
  assert.equal(eight.value, false, eight.text);
  assert.match(runLine(eight.text), /the first layer 9\.80e-4 from Q8,/);
  // (as far from NumPy's as Q8 is, and more: under K_PACKED × Q8, which no float form's line allows)
  assert.ok(Q8 > HALF_LINE);
  assert.equal(model([runOf({ form: "TF.js's tiles", keys: packed.keys, values: packed.values, off: 1e-3 })]).value, true);
});
check("a packed run of a model whose first layer is no attending layer's is held by every layer's line, not by the first keys' against Q8 and their scale (T.545)", () => {
  // an LFM2's first keys and values are of a layer after convolution layers (the real 230M's and 350M's were 1.2e-2 and 2.0e-2
  // from Q8 where Q8 itself was 2.6e-2 to 3.6e-2 from NumPy's, run 38859332497): one layer's arithmetic no longer
  const convolved = { ...CASE, plan: { layers: [{ kind: "convolution" }, { kind: "attention" }] } };
  const packed = { form: "ORT's DP4A", keys: floats(REFERENCE.keys8), values: floats(REFERENCE.values8) };
  const apart = runOf({ ...packed, off: GPU_LINE * 1.5 }), scaledUp = runOf({ ...packed, off: 0, keys: packed.keys.map((x) => x * (1 + 3e-4)), values: packed.values.map((x) => x * (1 + 3e-4)) });
  // past the shader's line from Q8, within K_PACKED × Q8's of NumPy's: the attending first layer fails, the convolution's does not
  assert.ok(Q8 + GPU_LINE * 1.5 < K_PACKED * Q8);
  for (const run of [apart, scaledUp]) {
    assert.equal(model([run]).value, true, "the first layer attends: held to Q8 as it was");
    const held = model([run], {}, convolved);
    assert.equal(held.value, false, held.text);
  }
  // the lines of every layer hold still (K_PACKED × Q8 from NumPy's, in the second layer), and so does every form that is not packed
  const failed = (run, pattern) => {
    const { value, text } = model([run], {}, convolved);
    assert.equal(value, true, text);
    assert.match(runLine(text), pattern);
  };
  failed(runOf({ ...packed, off: K_PACKED * Q8, layers: [1] }), /- the keys and values of the GPU's layer 1 are 8\.00e-2 and 8\.00e-2 from NumPy's \(its line 6\.00e-2\)$/);
  failed(runOf({ off: GPU_LINE * 1.5, layers: [0] }), /- the first layer's keys and values are 1\.20e-2 from NumPy's \(line 8\.00e-3\)/);
});
check("a run wrong in one thing fails, and says which", () => {
  const failed = (run, pattern) => {
    const { value, text } = model([run]);
    assert.equal(value, true, text);
    assert.match(runLine(text), / — FAILED\n {4}- /);
    assert.match(runLine(text), pattern);
  };
  // the second layer alone past its line: the first layer's line and scale hold
  failed(runOf({ off: GPU_LINE * 1.5, layers: [1] }), /- the keys and values of the GPU's layer 1 are 1\.20e-2 and 1\.20e-2 from NumPy's \(its line 8\.00e-3\)$/);
  // the first layer past the shader's line: said twice, by its layer and as the first layer
  failed(runOf({ off: GPU_LINE * 1.5, layers: [0] }), /- the first layer's keys and values are 1\.20e-2 from NumPy's \(line 8\.00e-3\)/);
  // a NaN is past every line (a comparison with it is false: the lines are written as "not within")
  const nan = runOf();
  const broken = floats(nan.keys).slice();
  broken[SIZE - 1] = NaN;
  failed({ ...nan, keys: b64(broken) }, /- the keys and values of the GPU's layer 1 are NaN and /);
  // (and in the first of the layers as well: the worst layer is the NaN's, not the layer after it)
  const first = floats(nan.keys).slice();
  first[0] = NaN;
  failed({ ...nan, keys: b64(first) }, /- the keys and values of the GPU's layer 0 are NaN and /);
  // the run all at once is held as the run block by block is: its keys and values, its scale, its tokens, its logits
  const past = runOf({ off: GPU_LINE * 1.5 }), scaledUp = runOf({ off: 0, keys: KEYS.map((x) => x * (1 + 3e-4)), values: VALUES.map((x) => x * (1 + 3e-4)) });
  failed({ ...runOf(), again: { ...runOf().again, keys: past.again.keys } }, /- the keys and values of the GPU's layer 0 are 1\.00e-3 and 1\.20e-2 from NumPy's \(its line 8\.00e-3\)$/m);
  failed({ ...runOf(), again: { ...runOf().again, keys: scaledUp.again.keys, values: scaledUp.again.values } }, /- the first layer's keys and values are scaled by 1 \+ -?\d\.\d\de-\d+ and 1 \+ 3\.00e-4 of NumPy's/);
  failed({ ...runOf(), again: { ...runOf().again, gpuTokens: N - 1 } }, /- the GPU took 3 and 2 of 3 tokens/);
  failed({ ...runOf(), again: { ...runOf().again, logits: b64(LOGITS.map((x, i) => (i === 3 ? 5 : x))) } }, /- another most likely token \(2, 3\) than NumPy's 2, and not a near tie$/);
  // every value 3e-4 too large: under the rows' lines, past the scale's
  const scaled = (factor) => runOf({ off: 0, keys: KEYS.map((x) => x * factor), values: VALUES.map((x) => x * factor) });
  assert.ok(3e-4 > SCALE_LINE && 3e-4 < GPU_LINE);
  failed(scaled(1 + 3e-4), /- the first layer's keys and values are scaled by 1 \+ 3\.00e-4 and 1 \+ 3\.00e-4 of NumPy's \(line 0\.0001\)$/);
  failed(scaled(1 - 3e-4), /- the first layer's keys and values are scaled by 1 \+ -3\.00e-4 and 1 \+ -3\.00e-4 of NumPy's/);
  assert.equal(model([scaled(1 + 5e-5)]).value, false);
  // a packed shader's second layer K_PACKED × Q8 and more from NumPy's (Q8's own distance, and as much again three times)
  failed(runOf({ form: "ORT's DP4A", keys: floats(REFERENCE.keys8), values: floats(REFERENCE.values8), off: K_PACKED * Q8, layers: [1] }),
    /- the keys and values of the GPU's layer 1 are 8\.00e-2 and 8\.00e-2 from NumPy's \(its line 6\.00e-2\)$/);
  // (a layer's line is its own Q8's: where the second layer's Q8 is three times the first's, the same run passes)
  const deeper = { ...CASE, reference: { ...REFERENCE, keys8: b64(moved(floats(REFERENCE.keys8), 2 * Q8, [1])), values8: b64(moved(floats(REFERENCE.values8), 2 * Q8, [1])) } };
  const within = model([runOf({ form: "ORT's DP4A", keys: floats(REFERENCE.keys8), values: floats(REFERENCE.values8), off: K_PACKED * Q8, layers: [1] })], {}, deeper);
  assert.equal(within.value, false, within.text);
  assert.match(runLine(within.text), /layer 1 0\.44 of its line\)/);
  // another most likely token: a near tie passes (half of NumPy's probability at least), another does not
  const logits = (token) => LOGITS.map((x, i) => (i === token ? 5 : x));
  assert.ok(logSoftmax(LOGITS)[4] >= logSoftmax(LOGITS)[2] - TIE && logSoftmax(LOGITS)[3] < logSoftmax(LOGITS)[2] - TIE);
  assert.equal(model([runOf({ logits: logits(4) })]).value, false);
  failed(runOf({ logits: logits(3) }), /- another most likely token \(3, 3\) than NumPy's 2, and not a near tie$/);
  // the GPU did not take the prompt, or not all of it, or a block past its keys and values, or did not end
  failed(runOf({ note: "prompts on the CPU (no GPU adapter here)" }), /- the GPU did not take it: prompts on the CPU \(no GPU adapter here\)/);
  failed(runOf({ gpuTokens: N - 1 }), /- the GPU took 2 and 3 of 3 tokens/);
  failed(runOf({ past: { gpuTokens: 2 } }), /- a block past the GPU's keys and values went to the GPU \(2 tokens\)/);
  failed(runOf({ ended: false }), /- the GPU's worker did not say it ended within 5 s of the release \(T.205\)/);
  // the status line of a run whose steps were asked for: with the steps, or without them
  failed(runOf({ steps: { why: "wk would not start where this GPU binds a buffer" }, note: "prompts and answers on WebGPU" }),
    /- the status line is "prompts and answers on WebGPU", not "prompts on WebGPU, answers on the CPU"/);
});
check("of a layer whose keys and values the CPU keeps in a ring, the positions the ring holds alone (T.558.1)", () => {
  // the first two positions of layer 0 wrong in the CPU's and in the GPU's run (a ring of the last position alone has
  // written over them): past the lines where every position is compared, within them where layer 0 holds position 2 on
  const wrong = (from) => ({ keys: b64(moved(floats(from.keys), 0.5, [0], 2)), values: b64(moved(floats(from.values), 0.5, [0], 2)) });
  const run = runOf();
  const ringed = { ...run, ...wrong(run), again: { ...run.again, ...wrong(run.again) } };
  const cpu = { ...CPU, ...wrong(CPU) };
  assert.equal(model([ringed], { cpu }).value, true, "every position compared: the rows a ring wrote over fail");
  const within = model([ringed], { cpu, heldFrom: [2, 0] });
  assert.equal(within.value, false, within.text);
});
check("the status of a GPU that takes the prompts", () => {
  for (const note of ["prompts on WebGPU", "prompts and answers on WebGPU", "prompts on WebGPU, answers on the CPU"]) assert.ok(promptsOnGpu(note), note);
  for (const note of ["prompts on the CPU (a fallback adapter)", "prompts on WebGPU's", undefined, ""]) assert.ok(!promptsOnGpu(note), String(note));
});
check("the rounds with a NaN in a block's row", () => {
  const rows = (cases) => ({ brokenRows: { cases } });
  const right = { value: "NaN", token: 5, at: 3, answered: true, failed: false, finite: 0, of: 32 };
  const good = model([runOf(rows([right, { ...right, value: "Infinity" }]))]);
  assert.equal(good.value, false, good.text);
  assert.match(runLine(good.text), /, the keys and values of a token whose row held NaN 0 of 32 finite, Infinity 0 of 32 finite$/);
  for (const wrong of [{ finite: 4 }, { answered: false }, { failed: true }]) {
    const bad = model([runOf(rows([{ ...right, ...wrong }]))]);
    assert.equal(bad.value, true);
    assert.match(runLine(bad.text), /- a block with NaN in the row of token 5 \(at 3\): the request /);
  }
  assert.match(runLine(model([runOf({ brokenRows: { skipped: "not this model" } })]).text), /whose row held not tried \(not this model\)$/);
  // the same block through forward.js: refused whole, the status line saying why, the cache the CPU's own to the bit
  const status = "prompts on the CPU (the GPU computed keys or values that are not finite numbers (NaN or infinity) in a block of the prompt at position 0)";
  const block = { value: "NaN", token: 1, at: 0, gpuTokens: 0, after: 0, status };
  const through = model([runOf({ refusedBlock: { skipped: "not this model" } })]);
  assert.match(runLine(through.text), /through forward\.js not tried \(not this model\)$/);
  const taken = model([runOf({ refusedBlock: { ...block, gpuTokens: 16, keys: CPU.keys, values: CPU.values } })]);
  assert.equal(taken.value, true);
  assert.match(runLine(taken.text), /through forward\.js: 16 of its tokens and 0 with the next block's taken on the GPU/);
});

// ---- verdicts.mjs: the lines of what the made-up model alone is asked
check("the quantizers' line", () => {
  const row = { shader: "QUANTIZE", what: "a group with a NaN", word: "0x7fc00000", want: "a NaN", ok: true };
  const right = said(() => quantizersWrong({ quantizers: { rows: [row, row] } }));
  assert.equal(right.value, false);
  assert.match(right.text, /^the quantizers alone \(QUANTIZE, NORM_QUANTIZE\): 2 of 2 groups' scale words as they are to be /);
  const wrong = said(() => quantizersWrong({ quantizers: { rows: [row, { ...row, ok: false, word: "0x3f800000" }] } }));
  assert.equal(wrong.value, true);
  assert.match(wrong.text, /: 1 of 2 groups' scale words .* — FAILED\n {4}- QUANTIZE, a group with a NaN: the word is 0x3f800000, where a NaN is to be$/s);
  const error = said(() => quantizersWrong({ quantizers: { error: "no device" } }));
  assert.deepEqual([error.value, error.text], [true, "the quantizers alone: FAILED\n    - no device"]);
  const skipped = said(() => quantizersWrong({ quantizers: { skipped: "no adapter" } }));
  assert.deepEqual([skipped.value, skipped.text], [false, "the quantizers alone: not tried (no adapter)"]);
  assert.deepEqual(said(() => quantizersWrong({})).text, "the quantizers alone: not tried (no answer)");
});
check("a late GPU, a refused adapter, the shaders remembered, a weight that is no finite number", () => {
  const line = (more, start) => {
    const { value, lines } = model([runOf()], more);
    return { value, line: lines.find((text) => text.startsWith(start)) ?? assert.fail(`no line starts with "${start}"`) };
  };
  const late = { note: "prompts on WebGPU", gpuTokens: 0, done: 0, failed: 0, keys: CPU.keys };
  assert.deepEqual(line({ late }, "  a GPU that answers late"), { value: false, line: "  a GPU that answers late: given up, it wrote nothing, and the CPU did the blocks" });
  for (const wrong of [{ done: 1 }, { failed: 1 }, { keys: REFERENCE.keys }]) {
    const got = line({ late: { ...late, ...wrong } }, "  a GPU that answers late");
    assert.equal(got.value, true);
    assert.match(got.line, / — FAILED$/);
  }
  assert.deepEqual(line({ late: { ...late, note: "prompts on the CPU (x)" } }, "  a GPU that answers late"),
    { value: false, line: "  a GPU that answers late: not tried (prompts on the CPU (x), 0 tokens on the GPU)" });

  const refused = { note: "prompts on the CPU (a fallback adapter: the CPU in the GPU's place)", seconds: 0.2, ended: true, whileStarting: { ended: true, seconds: 0.3 } };
  assert.deepEqual(line({ refused }, "  a fallback adapter"), { value: false,
    line: "  a fallback adapter without the tests' leave: prompts on the CPU (a fallback adapter: the CPU in the GPU's place) in 0.2 s, its worker ended" });
  for (const wrong of [{ note: "prompts on WebGPU" }, { seconds: 31 }, { ended: false }]) {
    const got = line({ refused: { ...refused, ...wrong } }, "  a fallback adapter");
    assert.equal(got.value, true);
    assert.match(got.line, / — FAILED$/);
  }
  const starting = line({ refused: { ...refused, whileStarting: { ended: false, seconds: 5 } } }, "  a GPU let go while it got ready");
  assert.deepEqual(starting, { value: true, line: "  a GPU let go while it got ready: its worker did not say it ended in 5.0 s — FAILED" });

  const remembered = { same: { remembered: true, matrices: "TF.js's tiles" }, other: { remembered: false } };
  assert.deepEqual(line({ remembered }, "  the shaders remembered"), { value: false,
    line: "  the shaders remembered: for this adapter taken (TF.js's tiles), for another not taken" });
  assert.equal(line({ remembered: { ...remembered, other: { remembered: true } } }, "  the shaders remembered").value, true);
  assert.equal(line({ remembered: { same: {}, other: {} } }, "  the shaders remembered").value, true);

  const status = "prompts on the CPU (the GPU computed logits that are not finite numbers (NaN or infinity)";
  const weight = { form: "ORT's DP4A", name: "w2", value: "NaN", asked: true, taken: null, status };
  assert.deepEqual(line({ broken: [weight, { ...weight, value: "Infinity" }] }, "  a token by ORT's DP4A, the GPU holding"), { value: false,
    line: "  a token by ORT's DP4A, the GPU holding in its last layer: w2 NaN refused, w2 Infinity refused" });
  for (const wrong of [{ taken: [3] }, { asked: false, why: "no steps", note: "x" }, { status: "prompts and answers on WebGPU" }]) {
    const got = line({ broken: [{ ...weight, ...wrong }] }, "  a token by ORT's DP4A, the GPU holding");
    assert.equal(got.value, true);
    assert.match(got.line, /w2 NaN NOT refused — FAILED\n {4}- w2 NaN: /);
  }
  // (a line a form)
  assert.equal(model([runOf()], { broken: [weight, { ...weight, form: "TF.js's tiles" }] }).lines.filter((text) => /the GPU holding in its last layer/.test(text)).length, 2);
});
check("the model on the GPU alone", () => {
  const first = [2, 4, 1, 3], on = "a step on the CPU threw: this model's weights are on the GPU, on it alone";
  const alone = { keys: REFERENCE.keys, values: REFERENCE.values, first, key: "device|key",
    right: { stored: 2.5e6, gpuTokens: N, keys: REFERENCE.keys, values: REFERENCE.values, first, status: "prompts and answers on WebGPU", threw: on },
    pieces: { lost: "the GPU could not hold the weights", threw: on } };
  const line = (given) => {
    const { value, lines } = model([runOf()], { alone: given });
    return { value, line: lines.find((text) => text.startsWith("  on the GPU alone")) };
  };
  const right = line(alone);
  assert.equal(right.value, false, right.line);
  assert.match(right.line, /^ {2}on the GPU alone \(2\.50 MB here\): 3 tokens of the prompt, keys and values the same to the bit as with the weights here, steps \[2,4,1,3\] \(\[2,4,1,3\]\)/);
  const wrong = (change, pattern) => {
    const got = line(change(structuredClone(alone)));
    assert.equal(got.value, true, got.line);
    assert.match(got.line, / — FAILED$/);
    if (pattern) assert.match(got.line, pattern);
  };
  wrong((a) => { a.right.keys = REFERENCE.values; return a; }, /keys and values NOT the same as with the weights here/);
  wrong((a) => { a.right.first = [2, 4, 1, 0]; return a; });
  wrong((a) => { a.right.gpuTokens = 2; return a; });
  wrong((a) => { a.right.threw = undefined; return a; }, /a step on the CPU ran;/);
  wrong((a) => { a.right.lost = "the device was lost"; return a; }, /: lost \(the device was lost\);/);
  wrong((a) => { a.pieces.lost = undefined; return a; }, /a GPU that fails as it opens: not lost/);
  // the made-up model's: a CPU far faster than the GPU is taken, and the verdict has the device's key
  const faster = { lost: "the CPU as /benchmark/ measured it is faster", verdict: { key: "device|key" }, threw: on };
  assert.equal(line({ ...alone, cpuFaster: faster }).value, false);
  wrong((a) => ({ ...a, cpuFaster: { ...faster, verdict: { key: "another" } } }), /the verdict WITHOUT the device's key/);
  wrong((a) => ({ ...a, cpuFaster: { ...faster, lost: undefined } }), /a CPU far faster: not lost/);
});

// ---- steps.mjs
check("the first matrix a token's layer could not bind", () => {
  const c = (header, more = {}) => ({ reference: { header }, headDim: 0, arch: "llama", dtype: "int8", ...more });
  // stories15M: k's scales (288 × 288 weights, 4 bytes a group of 32) are no multiple of 256
  assert.equal(unbound(c([288, 768, 6, 6, 6, 32000, 256])), "wk");
  assert.equal(unbound(c([64, 256, 2, 4, 2, 512, 256])), null);
  // (a ternary weight is a quarter of a byte: the same rows start elsewhere)
  assert.equal(unbound(c([128, 384, 3, 8, 4, 320, 256], { headDim: 32, dtype: "ternary" })), null);
  // GPT-2's FFN has no gate: no w3 to start anywhere
  assert.equal(unbound(c([64, 200, 3, 4, 4, 512, 256])), "w3");
  assert.equal(unbound(c([64, 200, 3, 4, 4, 512, 256], { arch: "gpt2" })), null);
});
const E16S = [E16, E16], Q8S = [Q8, Q8], PROMPT = [1e-3, 1e-3];
const steps = (given, c = CASE) => said(() => stepsRight(c, given, { e16s: E16S, q8s: Q8S, kvDim: KV, prompt: PROMPT }));
check("the steps left on the CPU, for a reason that is right or not", () => {
  const stories = { ...CASE, reference: { ...REFERENCE, header: [288, 768, 6, 6, 6, VOCAB, 256] } };
  assert.deepEqual(steps({ why: "wk would not start where this GPU binds a buffer" }, stories),
    { value: true, text: "  a token: on the CPU (wk would not start where this GPU binds a buffer)", lines: ["  a token: on the CPU (wk would not start where this GPU binds a buffer)"] });
  // the same words of a model whose matrices all start where a buffer binds, or another matrix's name: not right
  const aligned = { ...CASE, reference: { ...REFERENCE, header: [64, 256, 2, 4, 2, VOCAB, 256] } };
  assert.equal(steps({ why: "wk would not start where this GPU binds a buffer" }, aligned).value, false);
  assert.match(steps({ why: "wv would not start where this GPU binds a buffer" }, stories).text, / — FAILED$/);
  // (T.547.2.4) a model with matrices of Q4_K blocks: its steps on the CPU for T.547.2.3's reason are a failure now
  const blocks = { ...CASE, options: { kinds: { wq: "Q4_K", w1: "Q4_K" } } };
  assert.equal(steps({ why: "wq is of llama.cpp's blocks (Q4_K), which a token's step does not read yet" }, blocks).value, false);
  // a classifier past what the adapter binds: right where the table (vocabulary × dim) is larger than that
  const table = VOCAB * HEADER[0];
  assert.equal(steps({ why: "the classifier is past a buffer of this GPU", binds: table - 1 }).value, true);
  assert.equal(steps({ why: "the classifier is past a buffer of this GPU", binds: table }).value, false);
  // the run in pieces, whose steps were not forced
  const pieces = { ...CASE, force: { pieceBytes: 4096 } };
  assert.equal(steps({ why: "w1 is past a buffer of this GPU" }, pieces).value, true);
  assert.equal(steps({ why: "w1 is past a buffer of this GPU", forced: "a form" }, pieces).value, false);
  assert.equal(steps({ why: "the GPU took no steps" }).value, false);
});
// a generation's steps as NumPy's: the ids, the keys and values of the steps' positions, the rest as it is to be
const stepsOf = (more = {}) => {
  const cpuLogits = [0, 0, 1, 5, 0, 0];
  return { form: "fused f32", attention: "llama.cpp flash_attn_vec", pieces: 1, first: GREEDY.slice(0, 4), cpuStep: GREEDY[4], second: GREEDY.slice(5, 8),
    keys: b64(GREEDY_KEYS), values: b64(GREEDY_VALUES),
    uploaded: [3], uploadedTwice: [3], uploadedSame: true, uploadedLogits: b64(cpuLogits),
    together: [1, 2, 3, 4], apart: [1, 2, 3, 4], stopped: GREEDY.slice(0, 2), penaltyHistory: [GREEDY[0], 5], penalized: [3], sampled: [[2], [0]],
    narrowed: { topK: [[2], [4]], minP: [[2], [3]], presence: [[2], [3]], together: [1, 2, 3, 4], apart: [1, 2, 3, 4] }, ...more };
};
check("a generation's steps as NumPy's", () => {
  const { value, text } = steps(stepsOf());
  assert.equal(value, true, text);
  assert.equal(text, "  a token by fused f32, its attention by llama.cpp flash_attn_vec: 4 and 3 greedy ids as NumPy's, after a prompt on the CPU 3 as the CPU's, " +
    "the CPU's step between them NumPy's, keys and values of 4 positions 0.00e+0, layer 0 0.00 of its line 8.00e-3, 4 sampled steps [1,2,3,4] one at a time too, " +
    "a stop token after 2 steps, penalized 3 as NumPy's, sampled 2 and 0, a top-k of 3 2 and 4 of NumPy's 3, min-p 0.5 2 and 3 of NumPy's 2, " +
    "presence on the latest 1 and 2 2 and 3, 4 of them [1,2,3,4] one at a time too");
});
check("a generation's steps wrong in one thing", () => {
  const wrong = (more, pattern) => {
    const { value, text } = steps(stepsOf(more));
    assert.equal(value, false, text);
    assert.match(text, / — FAILED\n {4}- /);
    assert.match(text, pattern);
  };
  // an id that is not NumPy's and no near tie of NumPy's logits of that step
  wrong({ first: [2, 4, 0, 3] }, /- step 2 is 0, NumPy's 1, and not a near tie/);
  wrong({ first: [2, 4, 1] }, /- 4 steps from 0 came back as \[2,4,1\]/);
  wrong({ second: [5, 2, 0] }, /- step 7 is 0, NumPy's 4, and not a near tie/);
  // the keys and values of the first steps' positions: past the float forms' line, and the packed forms' own
  const off = moved(GREEDY_KEYS, GPU_LINE * 1.5, [1], GEN);
  wrong({ keys: b64(off) }, /- the keys and values of the steps' layer 1 are 1\.20e-2 from NumPy's \(its line 8\.00e-3\)/);
  assert.ok(GPU_LINE * 1.5 < K_PACKED * Q8);
  assert.equal(steps(stepsOf({ form: "ORT's DP4A", keys: b64(off) })).value, true);
  const nan = floats(b64(GREEDY_KEYS)).slice();
  // (a NaN in the last layer; in the first, below. Before the review of T.357.3 a NaN in a layer before the last one was
  // passed over, here and in the prompt's verdict: ratios.reduce() left a NaN for whatever came after it)
  nan[(LAYERS - 1) * GEN * KV] = NaN;
  wrong({ keys: b64(nan) }, /- the keys and values of the steps' layer 1 are NaN from NumPy's/);
  // (and in the first layer: it is the worst layer as well, not the one after it: T.357.3's review)
  const first = floats(b64(GREEDY_KEYS)).slice();
  first[0] = NaN;
  wrong({ keys: b64(first) }, /- the keys and values of the steps' layer 0 are NaN from NumPy's/);
  // (the positions after the first id that is not NumPy's are not compared: their inputs differ)
  const late = moved(GREEDY_KEYS, 1, [0, 1], GEN).map((x, i) => (i % (GEN * KV) < 2 * KV ? GREEDY_KEYS[i] : x));
  assert.match(steps(stepsOf({ first: [2, 0, 1, 3], keys: b64(late) })).text, /keys and values of 2 positions 0\.00e\+0,/);
  wrong({ together: [1, 2, 3, 5] }, /- 4 sampled steps together \[1,2,3,5\], one at a time \[1,2,3,4\]/);
  wrong({ uploadedSame: false }, /- after a prompt on the CPU over two other caches of the GPU, the step wrote other keys and values/);
  wrong({ uploaded: [0], uploadedTwice: [0] }, /- after a prompt on the CPU the step is 0, the CPU's 3, and not a near tie \(5\.00 below it in log-probability\)/);
  wrong({ stopped: GREEDY.slice(0, 3) }, /- with 4 a stop token the steps were \[2,4,1\], not \[2,4\]/);
  wrong({ penalized: [2] }, /- with a penalty of 100 the step is 2, NumPy's 3, and not a near tie/);
  wrong({ sampled: [[2], [2]] }, /- sampled at 0\.02 and 0\.98: 2 and 2, of NumPy's nucleus of \d+, which are to differ/);
  wrong({ cut: true }, /- the tables in 1 piece, not cut/);
  // T.546: a top-k that kept a 4th, the same token at both ends, the presence penalty on the prompt's tokens too (the
  // latest 1 as the latest 2), the steps of one submission not those one at a time, and none asked
  const narrowed = stepsOf().narrowed;
  wrong({ narrowed: { ...narrowed, topK: [[2], [5]] } }, /- a top-k of 3 sampled at 0\.02 and 0\.98: 2 and 5, of NumPy's \[2,3,4\]/);
  wrong({ narrowed: { ...narrowed, minP: [[2], [2]] } }, /- min-p 0\.5 sampled at 0\.02 and 0\.98: 2 and 2, of NumPy's \[2,3\], which are to differ/);
  wrong({ narrowed: { ...narrowed, presence: [[3], [3]] } }, /- with a presence penalty of 100 on the latest 1 the step is 3, NumPy's 2, and not a near tie/);
  wrong({ narrowed: { ...narrowed, together: [1, 2, 3, 5] } }, /- 4 steps with a top-k, min-p and presence together \[1,2,3,5\], one at a time \[1,2,3,4\]/);
  wrong({ narrowed: undefined }, /- no steps with a top-k, min-p or the presence penalty/);
  assert.equal(steps(stepsOf({ cut: true, pieces: 3 })).value, true);
  // the rounds with a NaN: a value among the keys and values, the logits
  const value = { value: "NaN", layer: 0, position: 1, column: 0, answered: true, failed: false, sampled: 0, notFinite: 1 };
  assert.match(steps(stepsOf({ brokenValues: { cases: [value, value] } })).text, /a NaN or an infinity among the values refused in 2 of 2$/);
  wrong({ brokenValues: { cases: [value, { ...value, sampled: 1, id: 7, notFinite: 0 }] } }, /- NaN in the values of layer 0 \(position 1, column 0\): the step sampled 1 \(id 7\)/);
  wrong({ brokenValues: { cases: [{ ...value, notFinite: 0 }] } }, /- NaN in the values of layer 0 \(position 1, column 0\): the step sampled 0 \(id undefined\) and its word of logits that are not finite is 0, where it is to be refused/);
  const status = "prompts on the CPU (the GPU computed logits that are not finite numbers (NaN or infinity)";
  assert.match(steps(stepsOf({ nanLogits: { taken: null, status } })).text, /, NaN logits refused$/);
  wrong({ nanLogits: { taken: [1, 2, 3, 4], status } }, /- logits made NaN: the request gave \[1,2,3,4\]/);
  wrong({ nanLogits: { taken: null, status: "prompts and answers on WebGPU" } }, /- logits made NaN: the request gave null/);
});
check("a model's verdict holds its steps'", () => {
  const right = model([runOf({ steps: stepsOf(), note: "prompts and answers on WebGPU" })]);
  assert.equal(right.value, false, right.text);
  assert.match(right.text, /\n {2}a token by fused f32, /);
  assert.equal(model([runOf({ steps: stepsOf({ penalized: [2] }), note: "prompts and answers on WebGPU" })]).value, true);
});

// ---- tables.mjs
check("the tables of a model, a line a run, the seconds", () => {
  const { lines } = model([runOf({ off: 2e-3 }), runOf({ form: "ORT's DP4A", off: 4e-3 })]);
  const at = (start) => lines.findIndex((text) => text.startsWith(start));
  const first = at("\nkeys and values of made-up by layer, against NumPy's (E16: NumPy's with its cache in float16; Q8: and the matrices' inputs in 8 bits): the worst row");
  assert.ok(first > 0, lines.join("\n"));
  assert.deepEqual(lines.slice(first + 1, first + 6), ["| layer | E16 | Q8 | CPU | A | B |", "|---:|---:|---:|---:|---:|---:|",
    "| 0 | 1.0e-3 | 2.0e-2 | 1.0e-1 | 2.0e-3 | 4.0e-3 |", "| 1 | 1.0e-3 | 2.0e-2 | 1.0e-1 | 2.0e-3 | 4.0e-3 |", "| all | 1.0e-3 | 2.0e-2 | 1.0e-1 | 2.0e-3 | 4.0e-3 |"]);
  for (const title of ["against NumPy's with its cache in float16", "against Q8 (T.187: what the packed shaders are held to)"]) {
    const table = at(`\nkeys and values of made-up by layer, ${title}: `);
    assert.equal(lines[table + 1], "| layer | CPU | A | B |");
    assert.equal(lines.slice(table + 3, table + 6).length, 3);
    assert.match(lines[table + 5], /^\| all \| /);
  }
  assert.match(lines[at("- CPU: ")], /^- CPU: forward\.js on the CPU: 1\.0e-1, 100\.0 E16, 5\.00 Q8, the first layer's scale .*, logits KL .*, the ten most likely .*, most likely 2$/);
  assert.match(lines[at("- A: ")], /^- A: TF\.js's tiles, llama\.cpp flash attention tiles: 2\.0e-3 \(all at once 2\.0e-3\), the worst layer [01] at 0\.25 of its line 8\.0e-3, 2\.0 E16, 0\.10 Q8, /);
  assert.match(lines[at("- B: ")], /^- B: ORT's DP4A, .* the worst layer [01] at 0\.07 of its line 6\.0e-2, .* most likely 2 as the CPU's$/);
  assert.equal(lines.at(-1), "\nseconds of made-up: NumPy (with its cache in float16, and Q8) 1.3, the plan 0.5, CPU 2.0, A (its GPU ready in 1.5) 4.0, B (its GPU ready in 1.5) 4.0\n");
  // the seconds of the NaN rounds, where any were run
  const nan = model([runOf()], { nanSeconds: { rows: 1.25, values: 0, logits: 2 } });
  assert.equal(nan.lines.at(-1), "seconds of the NaN rounds of made-up: rows 1.3, values 0.0, logits 2.0");
  // (by itself, with the measures a verdict gives it)
  const alone = said(() => layerTables(CASE, CPU, [], { e16: E16, q8: Q8, cpuLogits: { kl: 0, ten: 0, token: 2 }, logits: () => ({ kl: 0, ten: 0, token: 2 }), scale: () => 0 }));
  assert.equal(alone.lines.filter((text) => text.startsWith("| layer |")).length, 3);
});

// ---- harness.js and the rewriting Dawn's worker thread needs
check("the harness is a module, and the runtime's files it names are in this tree", () => {
  const file = new URL("./gpu-check/harness.js", import.meta.url), text = fs.readFileSync(file, "utf8");
  const parsed = spawnSync(process.execPath, ["--check", file.pathname], { encoding: "utf8" });
  assert.equal(parsed.status, 0, parsed.stderr);
  // the two constants its window puts before it (engines.mjs), and nothing else of Node's side
  assert.match(text, /\bONLY\b/);
  assert.match(text, /\bNAN_ROUNDS\b/);
  // what it takes from the runtime: by the server's paths, each a file of this tree; engines.mjs rewrites the imports
  // for Dawn's worker thread by their beginning, and the GPU's worker is a Worker of the same path
  const named = [...text.matchAll(/"\/runtime\/([^"]+)"/g)].map((match) => match[1]);
  assert.deepEqual([...new Set(named)].sort(), ["forward.ts", "gpu.ts", "jobs.ts", "shaders.ts"]);
  for (const name of named) assert.ok(fs.statSync(runtime(name)).isFile(), name);
  const imports = [...text.matchAll(/import\("\/runtime\/([^"]+)"/g)].map((match) => match[1]);
  assert.deepEqual(imports.sort(), ["forward.ts", "jobs.ts", "shaders.ts"]);
  assert.equal(text.split('import("/runtime/').length - 1, imports.length);
  assert.equal(text.split("/runtime/").length - 1, named.length, "a file of the runtime named in another way than by a string of its path");
  const engines = fs.readFileSync(new URL("./gpu-check/engines.mjs", import.meta.url), "utf8");
  assert.ok(engines.includes(`HARNESS.replaceAll('import("/runtime/', `), "engines.mjs no longer rewrites the harness's imports so");
  // (T.547.2.3: and the kernels of llama.cpp's blocks, for a model with rows of them)
  for (const built of text.matchAll(/"\/built\/([^"]+)"/g)) assert.match(built[1], /^(simd|block)kernel_(relaxed_)?shared(64)?\.wasm$/);
});

// ---- reference.mjs's Python: the two classes that make E16 and Q8, in the native Python (NumPy alone)
check("the cache in float16 and the matrix that takes its input in 8 bits", () => {
  // (the text imports the engine too: its classes are what this reads, cut out of it)
  const classes = PYTHON.slice(PYTHON.indexOf("class Half"), PYTHON.indexOf("def answer("));
  assert.ok(classes.includes("class Half(np.ndarray)") && classes.includes("class Q8(np.ndarray)"));
  const out = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c", `import json, numpy as np
${classes}
cache = np.zeros((2, 4), dtype=np.float32).view(Half)
third = np.float32(1) / np.float32(3)
cache[0, 1] = third
cache[1] = [third, 3.0, 1e-30, -third]
product = cache * np.float32(2)
x = np.concatenate([[127.0, 0.5, 1.5, 2.5, -0.5, -1.5, 126.4, 126.6], np.zeros(24), np.zeros(32), [2.54, 1.0, 0.011, -2.54], np.zeros(28)]).astype(np.float32)
w = np.eye(96, dtype=np.float32).view(Q8)
print(json.dumps({"written": float(cache[0, 1]), "half": float(np.float32(third).astype(np.float16)), "row": [float(v) for v in cache[1]],
    "plain": type(product).__name__, "kept": type(cache).__name__, "doubled": float(product[0, 1]),
    "q8": [float(v) for v in (w @ x)], "q8type": type(w @ x).__name__}))`], { env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).toString());
  // what is written into the cache is the nearest float16 (under its range 0); what is computed of it is plain
  assert.equal(out.written, out.half);
  assert.notEqual(out.written, Math.fround(1 / 3));
  assert.deepEqual(out.row, [out.half, 3, 0, -out.half]);
  assert.deepEqual([out.kept, out.plain, out.doubled], ["Half", "ndarray", 2 * out.half]);
  // a group of 32 by its largest / 127, each value rounded half to even; a group of zeros stays zeros (no NaN)
  assert.deepEqual(out.q8.slice(0, 8).map((value) => value + 0), [127, 0, 2, 2, 0, -2, 126, 127]);
  assert.ok(out.q8.slice(8, 64).every((value) => value === 0));
  const scale = Math.fround(Math.fround(2.54) / 127);
  assert.ok(Math.abs(out.q8[64] - 127 * scale) < 1e-6 && Math.abs(out.q8[65] - 50 * scale) < 1e-6 && Math.abs(out.q8[66] - scale) < 1e-6 && Math.abs(out.q8[67] + 127 * scale) < 1e-6, String(out.q8.slice(64, 68)));
  assert.equal(out.q8type, "ndarray");
});

console.log(`gpu-check-parts-check: ok (${checks} checks; ${Object.keys(SYNTHETIC).length} made-up models, a made-up case of ${LAYERS} layers and ${N} positions)`);
