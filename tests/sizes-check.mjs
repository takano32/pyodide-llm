// tests/sizes-check.mjs (T375.5): what sizes a model before it is loaded, each part by itself with stand-ins for its
// neighbours: the table of the architectures (src/runtime/forward/architectures.ts), a model's sizes of its header and
// form (sizes.ts's sizesOfForm(), kindsHad()), footprint() (memory.ts) with a table of kinds written here, and what
// alone.ts answers of the GPU (gpuBytes(), layerWeightsOf(), gpuOnlyUnfit()) by the facts of the two tables. No engine,
// no model, no kernels, no Pyodide. Node only, under a second, in the light suite.
//
//   node tests/sizes-check.mjs
//
// What it sees: that footprint() counts what the kinds it is handed say (a token's frame, what is apart from it, what
// the layers keep, keys and values for the kinds that cache them) and names none of them; that the packing's line and
// the architecture's line are what it reads (a line changed changes the count as the line says); that a model goes on
// the GPU alone by its facts.
// What it does not see: whether the lines are right (tests/plan-keys-check.mjs holds the architectures and the kinds'
// layers() and widths() to Python's plans, tests/kernel-registry-check.mjs the packings to Python's dtypes), and
// whether footprint() holds what an engine places (tests/memory-check.mjs, tests/plan-facts-check.mjs).
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";

const { ARCHITECTURES, architectureOf } = await import(runtimeUrl("forward/architectures.js"));
const { sizesOfForm, sizesOfPlan, kindsHad } = await import(runtimeUrl("forward/sizes.js"));
const { footprint, keysInHalf } = await import(runtimeUrl("forward/memory.js"));
const { gpuBytes, layerWeightsOf, gpuOnlyUnfit, aloneFactsUnfit, gpuWeightBytes } = await import(runtimeUrl("forward/alone.js"));
const { PACKED, BATCH, weightBytes } = await import(runtimeUrl("jobs.js"));
const { LAYER_KINDS } = await import(runtimeUrl("forward/kinds.js"));

const align = (n) => Math.ceil(n / 64) * 64;
// [dim, hidden, layers, heads, kvHeads, vocab, seqLen]
const HEADER = [64, 192, 6, 4, 2, 1000, 128];

// ---- the architectures
{
  assert.ok(["llama", "gpt2", "neox", "qwen35", "lfm2"].every((name) => Object.hasOwn(ARCHITECTURES, name)));
  for (const [name, line] of Object.entries(ARCHITECTURES)) {
    assert.deepEqual(Object.keys(line).sort(), ["gate", "gatedFfn", "layerNorm", "positions", "rope"], name);
    assert.equal(architectureOf(name), line);
  }
  for (const name of ["no such architecture", "toString", "constructor", ""]) assert.equal(architectureOf(name), ARCHITECTURES.llama, "a name the table has not is sized as a Llama (Python refuses it by its own words)");
  console.log(`ok: the ${Object.keys(ARCHITECTURES).length} architectures' lines, and a Llama's for a name that is none of them`);
}

// ---- stand-ins: two kinds of layers written here. "plain" attends and caches (as the product's attention: the kind of
// a layer that no kind says is its own); "echo" keeps a state, says which layers of a form are its own by a key of its
// own, has a width of its own, an array apart from the frame and two things it keeps
const asked = [];
const plain = { frame: (sizes) => { asked.push(["plain", sizes]); return { arrays: sizes.gate ? [["gate", 40]] : [], apart: [["scores", 1000]] }; }, cached: true, make() {} };
const echo = {
  layers: ({ echo: numbers }, count) => (numbers ? Array.from({ length: count }, (_, l) => l % numbers.every === 0) : null),
  widths: ({ echo: numbers }) => (numbers ? { echo: { wide: numbers.wide } } : {}),
  frame: (sizes) => { asked.push(["echo", sizes]); return { arrays: [["heard", sizes.widths.echo.wide * 4]], apart: [["far", 300]], widest: sizes.widths.echo.wide }; },
  state: (sizes, count) => [count * 1000, 20],
  make() {},
};
const KINDS = { attention: plain, echo };

// ---- a model's sizes of its header and its form
{
  const none = sizesOfForm(HEADER, {}, KINDS);
  assert.deepEqual(none, { dim: 64, hidden: 192, heads: 4, seqLen: 128, kinds: Array(6).fill("attention"), widths: { q: 64, kv: 32, linear: null, rotated: [64, 192] }, gate: false, numbers: {} });
  const form = { head_dim: 32, echo: { every: 3, wide: 256 } }, some = sizesOfForm(HEADER, form, KINDS);
  assert.deepEqual(some.kinds, ["echo", "attention", "attention", "echo", "attention", "attention"], "a layer is of the kind that says so of the form, and attends where none does");
  assert.deepEqual(some.widths, { q: 128, kv: 64, linear: null, echo: { wide: 256 }, rotated: [64, 128, 192, 256] },
    "the heads' widths by the head's size, a kind's own widths, and every width a matrix reads (a kind's widest among them), in order");
  assert.equal(some.numbers, form, "the kinds' numbers are the form's, under their names");
  assert.deepEqual(kindsHad(some, KINDS), [{ kind: "attention", count: 4 }, { kind: "echo", count: 2 }], "the kinds a model has, in the table's order");
  assert.deepEqual(kindsHad(none, KINDS), [{ kind: "attention", count: 6 }], "and not a kind it has no layer of");
  assert.deepEqual(kindsHad(some, { echo, attention: plain }).map(({ kind }) => kind), ["echo", "attention"], "(the table's order, not the layers')");
  // two kinds that both say a layer is theirs: the first of the table has it
  const twice = sizesOfForm(HEADER, form, { attention: plain, first: { ...echo, layers: (_, count) => Array(count).fill(true) }, echo });
  assert.deepEqual(twice.kinds, Array(6).fill("first"));
  // the gate is the architecture's line, nothing of the form's
  assert.deepEqual(["llama", "gpt2", "neox", "qwen35", "lfm2", undefined, "no such"].map((arch) => sizesOfForm(HEADER, { arch }, KINDS).gate), [false, false, false, true, false, false, false]);
  // and of a plan: what the plan says, said again
  const plan = { dim: 1, hidden_dim: 2, n_heads: 3, seq_len: 4, layers: [{ kind: "a" }, { kind: "b" }], widths: { q: 5 }, tensors: { wg: {} } };
  assert.deepEqual(sizesOfPlan(plan), { dim: 1, hidden: 2, heads: 3, seqLen: 4, kinds: ["a", "b"], widths: plan.widths, gate: true, numbers: plan });
  assert.equal(sizesOfPlan({ ...plan, tensors: {} }).gate, false);
  console.log("ok: a model's sizes of its header and form: every layer's kind by the kinds' own word, the widths, the kinds had in the table's order; and of a plan as the plan says");
}

// ---- footprint() counts what the kinds say
{
  const options = { dtype: "float32" };
  const count = (form, kinds = KINDS) => footprint(HEADER, 1e6, { ...options, ...form }, kinds);
  const [dim, hidden, layers, heads, kvHeads, vocab, seqLen] = HEADER, kv = kvHeads * (dim / heads);
  // every layer attends: a megabyte, BATCH frames and what is apart, the logits, and the keys and values of every layer
  const frame = (q, read, extra) => [dim, Math.max(dim, q, read), dim, q, kv, kv, dim, hidden, hidden].reduce((sum, n) => sum + align(n * 4), 0) +
    align(Math.max(dim, hidden, q, read)) + align(Math.ceil(Math.max(dim, hidden, q, read) / 32) * 4) + extra;
  asked.length = 0;
  assert.equal(count({}), 2 ** 20 + BATCH * (frame(dim, 0, 0) + align(1000)) + vocab * 4 + seqLen * layers * 2 * kv * 4);
  assert.ok(asked.length > 0 && asked.every(([kind]) => kind === "plain"), "a kind the model has no layer of is not asked");
  // two layers of six echo: their array in the frame (and xb and xq as wide as it reads), what they keep apart from it,
  // their two things kept (each as the engine takes it of the memory), and no keys and values for them
  asked.length = 0;
  const form = { echo: { every: 3, wide: 256 } };
  assert.equal(count(form), 2 ** 20 + BATCH * (frame(dim, 256, align(256 * 4)) + align(1000) + align(300)) + vocab * 4 + align(2 * 1000) + align(20) + seqLen * 4 * 2 * kv * 4);
  assert.ok(asked.some(([kind]) => kind === "echo") && asked.every(([, sizes]) => sizes.widths.echo.wide === 256), "each kind is asked with the model's sizes");
  // a rotated basis: a place in the frame for a rotated input, and the signs of every width (the kind's among them)
  assert.equal(count({ ...form, rotated: { block: 64 } }) - count(form), BATCH * align(256 * 4) + [64, 192, 256].reduce((sum, width) => sum + align(width * 4), 0));
  // a kind that says nothing of what it keeps, and caches nothing, adds its frame alone
  const quiet = { ...echo, state: undefined };
  assert.equal(count(form, { attention: plain, echo: quiet }), count(form) - align(2000) - align(20));
  assert.equal(count(form, { attention: plain, echo: { ...echo, cached: true } }), count(form) + seqLen * 2 * 2 * kv * 4, "keys and values for the layers of a kind that caches them");
  // the gate, by the architecture's line alone
  assert.equal(count({ arch: "qwen35" }) - count({ arch: "llama" }), BATCH * align(40));
  // (the product's table is the default)
  assert.equal(footprint(HEADER, 1e6, options), footprint(HEADER, 1e6, options, LAYER_KINDS));
  console.log("ok: footprint() with a table of kinds written here: a token's frame, what is apart from it, what the layers keep and the keys and values of those that cache, each as the kinds say");
}

// ---- footprint() reads the packing's line and the architecture's
{
  const size = 36e6, wide = [1024, 4096, 4, 16, 16, 32000, 256], [dim, , , , , vocab, seqLen] = wide, headSize = 64;
  for (const [dtype, packing] of Object.entries(PACKED)) {
    const of = (more) => footprint(wide, size, { dtype, outliers: 0, ...more });
    // the weights of a file are its bytes by the bytes of a weight; relaxed SIMD's corrections are four bytes a group
    // where the packing's relaxed kernel takes them, and nothing where it does not
    const weights = size / weightBytes(packing);
    assert.ok(Math.abs(of({ relaxed: true }) - of({ relaxed: false }) - (packing.sums ? (weights / packing.group) * 4 : 0)) <= 1, `${dtype}: the corrections`);
    // off the int8 kernels every weight is widened to float32
    assert.ok(Math.abs(of({ int8: false, relaxed: false }) - of({ relaxed: false }) - weights * 4) <= 1, `${dtype}: widened`);
    // the scales of a token's activations as its kernels take them: xs bytes a group of 32
    assert.equal(of({ relaxed: false }) - footprint(wide, size, { dtype: "float32" }) - seqLen * headSize * 4, BATCH * (align((4096 / 32) * packing.xs) - align((4096 / 32) * 4)), `${dtype}: the bytes of xs, and the RoPE tables a packed file leaves out`);
    // float16 keys and values only of packed weights
    assert.ok(of({ halfKV: true, shared: true }) < of({ shared: true }) && keysInHalf(wide, size, { dtype, halfKV: true, shared: true }), `${dtype}: float16 keys and values`);
  }
  for (const dtype of ["float32", "float16"]) {
    assert.equal(footprint(wide, size, { dtype, halfKV: true, shared: true }), footprint(wide, size, { dtype, shared: true }), `${dtype}: float32 keys and values, whatever the caller allows`);
    assert.equal(keysInHalf(wide, size, { dtype, halfKV: true, shared: true }), false);
  }
  // a table of positions: no matrix (not among the weights that have corrections), widened of a packed file; and the
  // RoPE tables Python makes of any file of a model without RoPE, and of a packed file of one with it
  const lines = Object.entries(ARCHITECTURES).filter(([, { gate }]) => !gate);
  for (const [arch, { positions, rope }] of lines) {
    const float = footprint(wide, size, { dtype: "float32", arch }) - footprint(wide, size, { dtype: "float32" });
    assert.equal(float, rope ? 0 : seqLen * headSize * 4, `${arch}: a float32 file`);
    const packed = footprint(wide, size, { dtype: "int8", arch, relaxed: false, outliers: 0 }) - footprint(wide, size, { dtype: "int8", relaxed: false, outliers: 0 });
    assert.equal(packed, positions ? seqLen * dim * 4 : 0, `${arch}: an int8 file`);
    const corrected = footprint(wide, size, { dtype: "int8", arch, outliers: 0 }) - footprint(wide, size, { dtype: "int8", outliers: 0 }) - packed;
    assert.ok(Math.abs(corrected + (positions ? (seqLen * dim) / 8 : 0)) <= 1, `${arch}: no corrections for a table of positions`);
  }
  assert.ok(lines.some(([, { positions }]) => positions) && lines.some(([, { rope }]) => !rope), "(the table has a line with a table of positions and one without RoPE)");
  // an embedding apart from the classifier (a negative vocabulary) is no matrix either
  assert.ok(Math.abs(footprint(wide, size, { dtype: "int8", outliers: 0 }) - footprint([...wide.slice(0, 5), -vocab, seqLen], size, { dtype: "int8", outliers: 0 }) - (vocab * dim) / 8) <= 1);
  console.log(`ok: footprint() by the lines of the two tables: the weights, the corrections and the activations' scales of ${Object.keys(PACKED).join(", ")}; float16 keys and values of packed weights only; a table of positions and RoPE's tables by the architecture`);
}

// ---- the GPU: what it would hold, and whether a model goes on it alone, by the same facts
{
  const header = [2048, 8192, 16, 32, 8, 128256, 4096], [dim, hidden, layers] = header;
  for (const [arch, { gatedFfn }] of Object.entries(ARCHITECTURES)) {
    const matrices = 2 * dim * dim + 2 * (dim / 4) * dim + (gatedFfn ? 3 : 2) * hidden * dim;
    assert.equal(layerWeightsOf(header, { arch }), layers * matrices, `${arch}: the weights of a layer's matrices, ${gatedFfn ? "three" : "two"} of the FFN`);
    for (const dtype of ["int8", "int6", "ternary"]) {
      assert.equal(gpuBytes(header, { arch, dtype }) - gpuBytes(header, { arch: "llama", dtype }), layers * (matrices - layerWeightsOf(header, {}) / layers) * gpuWeightBytes(dtype), `${arch}, ${dtype}: on the GPU`);
    }
  }
  assert.equal(layerWeightsOf(header, { head_dim: 128 }), layers * (2 * 4096 * dim + 2 * 1024 * dim + 3 * hidden * dim), "(by the head's size where the form says one)");
  const adapter = { fallback: false, packed: true, limits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 32 } };
  // by the architecture's facts: RMSNorms, a gated FFN, no table of positions, no gate of the attention's output. T382:
  // each fact by a reason of its own, the first of them in aloneFactsUnfit()'s order (said to nobody but the tests:
  // the worker reads yes or no)
  const GATE = "a gate of the attention's output is not on the GPU alone yet", POSITIONS = "a table of positions is not on the GPU alone yet",
    FFN = "an FFN without a gate is not on the GPU alone yet", NORMS = "LayerNorms are not on the GPU alone yet", STATE = "layers that keep a state are not on the GPU yet";
  for (const [arch, { gatedFfn, layerNorm, positions, gate }] of Object.entries(ARCHITECTURES)) {
    const said = gpuOnlyUnfit(header, "int8", { arch }, adapter);
    assert.equal(said, gate ? GATE : positions ? POSITIONS : !gatedFfn ? FFN : layerNorm ? NORMS : null, `${arch}: ${said}`);
  }
  assert.deepEqual(["llama", "gpt2", "neox", "qwen35", "lfm2"].map((arch) => gpuOnlyUnfit(header, "int8", { arch }, adapter)), [null, POSITIONS, FFN, GATE, null]);
  // (T487: the table's five lines never have one of these facts without another, so each is asked by itself here, of
  // facts no architecture has: RMSNorms with GPT-2's FFN, LayerNorms with a gated one, a table of positions alone)
  const LLAMAS = { stateful: false, gate: false, positions: false, gatedFfn: true, layerNorm: false };
  assert.equal(aloneFactsUnfit(LLAMAS), null);
  for (const [fact, value, reason] of [["stateful", true, STATE], ["gate", true, GATE], ["positions", true, POSITIONS], ["gatedFfn", false, FFN], ["layerNorm", true, NORMS]]) {
    assert.equal(aloneFactsUnfit({ ...LLAMAS, [fact]: value }), reason, `a model that is a Llama but for ${fact} = ${value}`);
  }
  // (the order: the first of them that holds)
  assert.deepEqual([aloneFactsUnfit({ stateful: true, gate: true, positions: true, gatedFfn: false, layerNorm: true }),
    aloneFactsUnfit({ ...LLAMAS, gate: true, positions: true, gatedFfn: false, layerNorm: true }), aloneFactsUnfit({ ...LLAMAS, positions: true, gatedFfn: false, layerNorm: true }),
    aloneFactsUnfit({ ...LLAMAS, gatedFfn: false, layerNorm: true })], [STATE, GATE, POSITIONS, FFN]);
  // and every layer attending: the form's layers that keep a state keep the model off it, whatever its name
  const linear = { every: 4, key_heads: 16, value_heads: 16, key_dim: 128, value_dim: 128, conv: 4 };
  assert.equal(gpuOnlyUnfit(header, "int8", { arch: "llama", linear }, adapter), STATE);
  assert.equal(gpuOnlyUnfit(header, "int8", { arch: "qwen35", linear }, adapter), STATE);
  assert.equal(gpuOnlyUnfit(header, "int8", { arch: "lfm2", convolution: { layers: "ca".repeat(8), taps: 3 } }, adapter), STATE);
  assert.equal(gpuOnlyUnfit(header, "int8", { arch: "lfm2", convolution: { layers: "a".repeat(16), taps: 3 } }, adapter), null, "(an LFM2 without a convolution layer has a Llama's facts)");
  assert.equal(gpuOnlyUnfit(header, "int8", { rotated: { block: 1024 } }, adapter), "a rotated basis is not on the GPU yet");
  // (T363.5's review: a model of llama.cpp's blocks says the dtype int8, so this line of gpuOnlyUnfit() alone keeps it
  // off the GPU alone; a form that names no kinds is as before)
  assert.notEqual(gpuOnlyUnfit(header, "int8", { arch: "llama", kinds: { w1: ["IQ2_XXS", "int8"] } }, adapter), null, "rows as blocks are not for the GPU alone");
  assert.notEqual(gpuOnlyUnfit(header, "int8", { arch: "llama", kinds: { wcls: "Q4_K" } }, adapter), null, "nor one row of a kind of its own");
  for (const kinds of [{}, null, undefined]) assert.equal(gpuOnlyUnfit(header, "int8", { arch: "llama", kinds }, adapter), null, "a form that names no kinds is a Llama's");
  // by the packing's line: whether a model of it may be there alone, and what its shaders need of the browser
  const lacking = { ...adapter, packed: false };
  // (a packing the GPU does not hold at all, gpu: null, stays on the CPU as well)
  for (const [dtype, { gpu }] of Object.entries(PACKED)) {
    assert.equal(gpuOnlyUnfit(header, dtype, {}, adapter), gpu?.alone ? null : `${dtype} weights stay on the CPU`);
    assert.equal(gpuOnlyUnfit(header, dtype, {}, lacking), !gpu?.alone ? `${dtype} weights stay on the CPU`
      : gpu.dot ? `${dtype} weights need the packed int8 dot product of WGSL, which this browser lacks` : null);
  }
  assert.ok(Object.values(PACKED).some(({ gpu }) => !gpu?.alone) && Object.values(PACKED).some(({ gpu }) => gpu?.dot), "(the table has a packing that stays on the CPU and one that needs the dot)");
  for (const dtype of ["float32", "float16", "made-up"]) assert.equal(gpuOnlyUnfit(header, dtype, {}, adapter), `${dtype} weights stay on the CPU`);
  assert.deepEqual([gpuOnlyUnfit(header, "int8", {}, null), gpuOnlyUnfit(header, "int8", {}, { ...adapter, fallback: true }), gpuOnlyUnfit(header, "int8", {}, { ...adapter, fallback: true }, { fallback: true })],
    ["no GPU adapter here", "a fallback adapter", null]);
  assert.equal(gpuOnlyUnfit(header, "int8", { head_dim: 30 }, adapter), "heads of a size that is no multiple of 4");
  // where a joined matrix's parts start and how long they are, in the packing's bytes of a row and its scales a group
  const small = { ...adapter, limits: { maxStorageBufferBindingSize: 2 ** 24, maxBufferSize: 2 ** 28 } };
  assert.equal(gpuOnlyUnfit(header, "int8", {}, small), "a layer's matrices are past a buffer of this GPU");
  assert.equal(gpuOnlyUnfit(header, "ternary", {}, small), null, "(a quarter of the bytes)");
  assert.equal(gpuOnlyUnfit([96, 200, 6, 4, 2, 500, 77], "int8", {}, adapter), "q, k and v or gate and up would not start where this GPU binds a buffer");
  console.log("ok: the GPU by the same lines: the matrices of a layer by the FFN's kind, a weight's bytes by how the GPU holds its packing, and a model on the GPU alone by its facts and its packing");
}
console.log("sizes-check: ok");
