// tests/gpu-forms-check.mjs (T.465.1): the model's GPU worker choosing a prompt's forms (src/runtime/gpu/forms.ts), on a
// device that runs no shader (tests/fakes/webgpu.mjs) and answers the checks' dispatches as the shaders would
// (tests/fakes/gpu-answers.mjs, whose arithmetic is tests/gpu_answers.py's in NumPy). forms.ts takes a candidate only once its output on made-up numbers is JavaScript's;
// nothing ran it without a GPU (T.465: five breaks of a review passed the light suite), as nothing wrote the answers.
// A made-up Llama (tests/fakes/gpu-plans.mjs) on two devices (none of shader-f16 and subgroups, and both), packed
// int8 dot both times, each a fallback adapter (a pair of one pass times a form there: the timing is not what is held):
//   - chooseAttention(): the first option of the device is checked and taken (its record m.forms, with nothing tried yet);
//   - chooseMatrices(): every form the device makes is checked right (none "wrong"), the right ones timed, one taken;
//   - each check bites: an attention answered off by a little in one head is no attention, and with no other option the
//     choice throws why; a form answered off in one row of y is "wrong", and the quantized activations off by one in
//     one of fifty codes "not quantize_x's" (a packed form: the float ones read no codes); an answer that leaves y as it was is wrong too.
// Not held here: the tiles' workgroups (an answer writes the whole output whatever was dispatched: CI's gpu-check holds
// those), ternary and llama.cpp's blocks (T.465.2), a token's forms (T.465.3, T.465.4).
//   node tests/gpu-forms-check.mjs
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { standIn } from "./fakes/webgpu.mjs";
import { MODELS, planned } from "./fakes/gpu-plans.mjs";
import { gpuAnswers } from "./fakes/gpu-answers.mjs";

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
// "quantize") or null
async function opened({ features = [], spoil = {} } = {}) {
  const answers = await gpuAnswers();
  // (a fallback adapter, which the plan allows as the tests' gpuForce does: forms.ts times each right form by one pair of
  // one pass there, and every pass of a timing would be answered too)
  const { device } = standIn({ fallback: true, language: ["packed_4x8_integer_dot_product", "subgroup_id"], answer: answers.answer });
  for (const feature of features) device.features.add(feature);
  const { plan, bytes } = planned(MODELS.llama);
  plan.force = { fallback: true };
  const memory = { buffer: new SharedArrayBuffer(bytes) };
  new Uint8Array(memory.buffer).forEach((_, i, all) => { all[i] = (i * 131 + 7 * (i >> 8) + 89) & 255; });
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
      if (!shape.none) answers.add(ENGINE.flashTile(shape), spoil.attention ?? flashTile(plan.headSize));
    }
  }
  answers.addPipeline(m.block.quantize, spoil.quantize ?? quantize);
  for (const form of devicePromptForms(device, m.own)) answers.add(form.code, spoil.form ?? tiled({ packed: form.packed, packing: form.ternary ? "ternary" : "int8" }));
  return { m, answers };
}
// (the models, for the cases that spoil one: the same Pyodide as every answerer's)
const { flashTile, quantize, tiled } = await gpuAnswers();
// a model off by a little where spoiled() says, after the right one wrote
const off = (model, spoiled) => (entries) => {
  model(entries);
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
  const { m } = await opened();
  await forms_.chooseAttention(m);
  const [form] = devicePromptForms(m.device, m.own).filter(({ none }) => !none);
  const { m: spoilt } = await opened({ spoil: { form: off(tiled({ packed: form.packed, packing: "int8" }), (entries) => { output(entries, 3)[7] *= 1.01; }) } });
  await forms_.chooseAttention(spoilt);
  await assert.rejects(forms_.chooseMatrices(spoilt), /no tiled shader is right on this GPU \(.*: wrong: products .* from JavaScript's/, "a form off in one row");
  // y left as it was: the product never written
  const { m: idle } = await opened({ spoil: { form: () => {} } });
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
console.log("gpu-forms-check: ok");
