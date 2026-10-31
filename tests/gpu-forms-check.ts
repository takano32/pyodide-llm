// tests/gpu-forms-check.ts (T.465.1): the model's GPU worker choosing a prompt's forms (src/runtime/gpu/forms.ts), on a
// device that runs no shader (tests/fakes/webgpu.mjs) and answers the checks' dispatches as the shaders would
// (tests/fakes/gpu-answers.ts, whose arithmetic is tests/gpu_answers.py's in NumPy). forms.ts takes a candidate only once
// its output on made-up numbers is JavaScript's; nothing ran it without a GPU (T.465: five breaks of a review passed the
// light suite), as nothing wrote the answers. A made-up Llama (tests/fakes/gpu-plans.ts), packed int8 dot on every
// device, each a fallback adapter (a pair of one pass times a form there: the timing is not what is held):
//   - on two devices (none of shader-f16 and subgroups, and both): chooseAttention() checks and takes the device's first
//     option, and chooseMatrices() checks every form the device makes right, times them and takes one;
//   - each check bites: an attention off by a little in one head, a form off in one row of y or writing nothing, the
//     quantized activations off by one in one of fifty codes (a packed form: the float ones read no codes); and takes
//     what is off inside its line; a device's first attention wrong, the next one is taken;
//   - T.465.2: a model's own forms: ternary weights (its forms checked right, one off in a row refused), and each of the
//     ten types of llama.cpp's blocks (two a model, on w2's layers, the plan's codebooks the converter's): every form's
//     text of each type checked right, the model's types alone compiled, a type's text off in a row refused by it.
// Not held here: that the fastest form is taken (the timing's clock is the real one), a window's attention, the forms a
// page remembers, a token's forms (T.465.3, T.465.4).
//   node tests/gpu-forms-check.ts
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { standIn } from "./fakes/webgpu.mjs";
import { MODELS, planned } from "./fakes/gpu-plans.ts";
import { gpuAnswers } from "./fakes/gpu-answers.ts";

// (the worker's global, as a worker has it)
globalThis.self = globalThis;
globalThis.postMessage = () => {};
const device_ = await import(runtimeUrl("gpu/device.ts"));
const { upload } = await import(runtimeUrl("gpu/weights.ts"));
const block_ = await import(runtimeUrl("gpu/block.ts"));
const forms_ = await import(runtimeUrl("gpu/forms.ts"));
const { ENGINE } = await import(runtimeUrl("shaders/engine.ts"));
const { flashShape } = await import(runtimeUrl("shaders/attention.ts"));
const { devicePromptForms } = await import(runtimeUrl("shaders/key.ts"));

// a made-up Llama on the stand-in, its forms answered; spoil: a model to put in place of one ("attention", "form",
// "quantize") or null. T.465.2: weights, its matrices' own: "ternary" (all of them), or two types of llama.cpp's blocks,
// those of w2's two layers (the one matrix of rows of 256: dim is 128), the plan's codebooks the converter's after the
// checkpoint; spoil.block: a maker of a block type's model, given the right one and the type
async function opened({ features = [], spoil = {}, weights = null } = {}) {
  const answers = await gpuAnswers();
  // (a fallback adapter, which the plan allows as the tests' gpuForce does: forms.ts times each right form by one pair of
  // one pass there, and every pass of a timing would be answered too)
  const { device } = standIn({ fallback: true, language: ["packed_4x8_integer_dot_product", "subgroup_id"], answer: answers.answer });
  for (const feature of features) device.features.add(feature);
  const { plan, bytes } = planned(MODELS.llama);
  plan.force = { fallback: true };
  // (a ternary matrix in the place of an int8 one, which has room for it: a quarter of the bytes and its scales)
  if (weights === "ternary") for (const matrix of Object.values(plan.matrices)) Object.assign(matrix, { packing: "ternary", ternary: true });
  // (no codebooks where no type is an IQ one: a Q4_K_M's plan has none, T.465.2's review)
  if (Array.isArray(weights)) Object.assign(plan.matrices.w2, { packing: "layers", packings: weights });
  const codebooks = Array.isArray(weights) && weights.some((type) => type.startsWith("IQ")) ? answers.codebooks() : null;
  const memory = { buffer: new SharedArrayBuffer(bytes + (codebooks?.length ?? 0)) };
  new Uint8Array(memory.buffer).forEach((_, i, all) => { all[i] = (i * 131 + 7 * (i >> 8) + 89) & 255; });
  if (codebooks) {
    new Uint8Array(memory.buffer).set(codebooks, bytes);
    plan.codebooks = bytes;
  }
  const m = await device_.openDevice(plan);
  device_.handed(memory, plan);
  await upload(m);
  await block_.prepare(m);
  // the attention's texts the device may make (forms.ts's options), the quantization, and every form's text
  const limits = { memory: device.limits.maxComputeWorkgroupStorageSize, threads: Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX),
    subgroupMin: m.info.subgroupMinSize, subgroupMax: m.info.subgroupMaxSize };
  for (const half of [false, true]) {
    for (const subgroups of [false, true]) {
      const shape = flashShape({ headSize: plan.headSize, half, subgroups, ...limits });
      // (spoil.halves: the f16 or subgroups texts alone, the device's first option)
      if (!shape.none) answers.add(ENGINE.flashTile(shape), ((half || subgroups) && spoil.halves) || spoil.attention || flashTile(plan.headSize));
    }
  }
  answers.addPipeline(m.block.quantize, spoil.quantize ?? quantize);
  // (spoil.form: a maker of each form's model, given the right one)
  for (const form of devicePromptForms(device, m.own)) {
    const right = tiled({ packed: form.packed, packing: form.ternary ? "ternary" : "int8", tile: form.tile });
    answers.add(form.code, spoil.form ? spoil.form(right) : right);
    // (T.465.2: and each text of a type of llama.cpp's blocks beside it)
    for (const [packing, code] of Object.entries(form.blocks ?? {})) {
      const block = tiled({ packed: true, packing, tile: form.tile });
      answers.add(code, spoil.block ? spoil.block(block, packing) : block);
    }
  }
  return { m, answers };
}
// (the models, for the cases that spoil one: the same Pyodide as every answerer's)
const { flashTile, quantize, tiled } = await gpuAnswers();
// a model off by a little where spoiled() says, after the right one wrote
const off = (model, spoiled) => (entries, x, y) => {
  model(entries, x, y);
  spoiled(entries);
};
const output = (entries, binding) => {
  const { resource } = entries.find((entry) => entry.binding === binding);
  const buffer = resource.buffer ?? resource;
  return new Float32Array(buffer.data.buffer, buffer.data.byteOffset + (resource.offset ?? 0), (resource.size ?? buffer.size) / 4);
};

for (const features of [[], ["shader-f16", "subgroups"]]) {
  const where = features.length ? "with shader-f16 and subgroups" : "with neither shader-f16 nor subgroups";
  const { m, answers } = await opened({ features });
  await forms_.chooseAttention(m);
  const halves = features.includes("shader-f16");
  assert.equal(m.forms.attention.name, `llama.cpp flash attention tiles${halves ? ", f16, subgroups" : ""}`, `${where}: the device's first option`);
  assert.deepEqual([m.forms.form, m.forms.tried], [null, []], `${where}: the record, nothing of the matrices yet`);
  await forms_.chooseMatrices(m);
  const made = devicePromptForms(m.device, m.own), wrong = m.forms.tried.filter(({ none }) => none?.startsWith("wrong"));
  assert.deepEqual(wrong, [], `${where}: no form wrong`);
  const timed = m.forms.tried.filter(({ ms }) => ms !== undefined).map(({ name }) => name);
  assert.deepEqual(timed, made.filter(({ none }) => !none).map(({ name }) => name), `${where}: every form the device makes right, and timed`);
  assert.ok(timed.length >= 2 && timed.includes(m.forms.form.name), `${where}: one taken of ${timed.length}`);
  assert.ok(answers.seen.length > 0);
  console.log(`ok: ${where}: ${m.forms.attention.name} checked and taken; ${timed.length} forms right and timed (${timed.join(", ")}), ${m.forms.form.name} taken`);
}

// each check bites
{
  // the attention off in its second head of the first token, by 1% of a value
  const { m } = await opened({ spoil: { attention: off(flashTile(16), (entries) => { output(entries, 3)[16] += 0.05; }) } });
  await assert.rejects(forms_.chooseAttention(m), /no attention is right on this GPU \(llama\.cpp flash attention tiles: its output is .* of the largest value from JavaScript's/,
    "an attention a little off in one head");
}
{
  // a form off in one row of y
  const { m: spoilt } = await opened({ spoil: { form: (right) => off(right, (entries) => { output(entries, 3)[7] *= 1.01; }) } });
  await forms_.chooseAttention(spoilt);
  await assert.rejects(forms_.chooseMatrices(spoilt), /no tiled shader is right on this GPU \(.*: wrong: products .* from JavaScript's/, "a form off in one row");
  // y left as it was: the product never written
  const { m: idle } = await opened({ spoil: { form: () => () => {} } });
  await forms_.chooseAttention(idle);
  await assert.rejects(forms_.chooseMatrices(idle), /wrong: products/, "a form that writes nothing");
}
{
  // the quantized activations off by one in a code of fifty (more than tiledOff's 1% of codes that may differ)
  const { m } = await opened({ spoil: { quantize: off(quantize, (entries) => {
    const { resource } = entries.find((entry) => entry.binding === 1), buffer = resource.buffer ?? resource;
    // (one step towards zero: a code of 127 or -128 moved past its edge would be far, another wrong)
    for (let i = 0; i < buffer.size; i += 50) buffer.data[i] = (buffer.data[i] + (buffer.data[i] < 128 ? -1 : 1)) & 255;
  }) } });
  await forms_.chooseAttention(m);
  await forms_.chooseMatrices(m);
  // (the packed forms read the codes; the float ones do not, and are right: one of them is taken)
  const packed = devicePromptForms(m.device, m.own).filter(({ none, packed }) => !none && packed).map(({ name }) => name);
  assert.ok(packed.length > 0);
  for (const name of packed) assert.match(m.forms.tried.find((tried) => tried.name === name)?.none ?? "", /^wrong: .* are not quantize_x's/, `${name}: quantized activations off by one in a code of fifty`);
  assert.ok(!m.forms.form.packed, "a float form taken");
}
console.log("ok: each check bites: an attention, a form and a quantization a little off are refused, and a form that writes nothing");

// and takes what is a little off inside its line (T.465.1's review: a line drawn too tight passed): the attention off by
// 0.1% of the largest value in one head (its line 0.4%), a form's one row off by 3e-5 of itself (the float forms' line
// 1e-4 of a row's size), the quantized codes off by one in one of 400 (1% of them may differ)
{
  const { m } = await opened({ spoil: { attention: off(flashTile(16), (entries) => { output(entries, 3)[16] += 0.004; }),
    quantize: off(quantize, (entries) => {
      const { resource } = entries.find((entry) => entry.binding === 1), buffer = resource.buffer ?? resource;
      for (let i = 0; i < buffer.size; i += 400) buffer.data[i] = (buffer.data[i] + (buffer.data[i] < 128 ? -1 : 1)) & 255;
    }) } });
  await forms_.chooseAttention(m);
  assert.equal(m.forms.attention.name, "llama.cpp flash attention tiles", "an attention inside its line");
  await forms_.chooseMatrices(m);
  assert.deepEqual(m.forms.tried.filter(({ none }) => none), [], "codes a little off, inside the line: every form right");
  const { m: near } = await opened({ spoil: { form: (right) => off(right, (entries) => { output(entries, 3)[7] *= 1 + 3e-5; }) } });
  await forms_.chooseAttention(near);
  await forms_.chooseMatrices(near);
  assert.deepEqual(near.forms.tried.filter(({ none }) => none), [], "a row a little off, inside the line: right");
}
// a device's first attention wrong: the next one taken (the f16 and subgroups texts off, the plain tiles right)
{
  const { m } = await opened({ features: ["shader-f16", "subgroups"], spoil: { halves: off(flashTile(16), (entries) => { output(entries, 3)[16] += 0.05; }) } });
  await forms_.chooseAttention(m);
  assert.equal(m.forms.attention.name, "llama.cpp flash attention tiles", "the plain tiles, after the f16 ones were wrong");
}
console.log("ok: what is a little off inside its line is taken, and a device's first attention wrong gives the next one");

// ---- T.465.2: a model's own forms: ternary weights, and llama.cpp's blocks (the ten types, two a model on w2's layers)
{
  const { m } = await opened({ weights: "ternary" });
  await forms_.chooseAttention(m);
  await forms_.chooseMatrices(m);
  const made = devicePromptForms(m.device, m.own).filter(({ none }) => !none).map(({ name }) => name);
  assert.ok(made.length > 0 && made.every((name) => name.endsWith(", ternary")), `a ternary model's own forms: ${made.join(", ")}`);
  assert.deepEqual(m.forms.tried.filter(({ none }) => none), [], "ternary: every form right");
  assert.ok(m.forms.form.ternary);
  // (a ternary form off in one row: wrong)
  const { m: spoilt } = await opened({ weights: "ternary", spoil: { form: (right) => off(right, (entries) => { output(entries, 3)[7] *= 1.01; }) } });
  await forms_.chooseAttention(spoilt);
  await assert.rejects(forms_.chooseMatrices(spoilt), /no tiled shader is right on this GPU \(.*, ternary: wrong: products/, "a ternary form off in one row");
  console.log(`ok: ternary: ${made.join(", ")} checked right, ${m.forms.form.name} taken; one off in a row refused`);
}
const PAIRS_OF_TYPES = [["Q2_K", "Q4_K"], ["IQ2_XXS", "IQ2_XS"], ["IQ2_S", "IQ3_XXS"], ["IQ3_S", "IQ1_S"], ["IQ1_M", "IQ4_XS"]];
for (const types of PAIRS_OF_TYPES) {
  const { m } = await opened({ weights: types });
  await forms_.chooseAttention(m);
  await forms_.chooseMatrices(m);
  assert.deepEqual(m.forms.tried.filter(({ none }) => none), [], `${types.join(" and ")}: every form right`);
  assert.deepEqual(Object.keys(m.forms.form.blockPipelines).sort(), [...types].sort(), `${types.join(" and ")}: the texts of the model's types alone compiled`);
  // (the second type's text off in one row: the form is wrong, said by that text)
  const { m: spoilt } = await opened({ weights: types, spoil: { block: (right, packing) => (packing === types[1] ? off(right, (entries) => { output(entries, 3)[7] *= 1.01; }) : right) } });
  await forms_.chooseAttention(spoilt);
  await assert.rejects(forms_.chooseMatrices(spoilt), new RegExp(`wrong: its ${types[1]} text: products`), `${types[1]}'s text off in one row`);
}
console.log(`ok: llama.cpp's blocks: each of the ten types' texts checked right on the forms the device makes (${PAIRS_OF_TYPES.flat().join(", ")}), and each second of a pair off in a row refused by its text`);
console.log("gpu-forms-check: ok");
