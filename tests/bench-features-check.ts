// tests/bench-features-check.ts (T642.17's review): which shaders /benchmark/'s GPU worker offers a device by the features
// it has, on a WebGPU that writes down what it is asked and computes nothing (tests/fakes/webgpu.ts), in Node:
//
//   node tests/bench-features-check.ts
//
// The worker reads four things of a device once (shared.features, shaders.ts's deviceFeatures()): shader-f16 (half), the
// device's subgroups (shuffles: the prompt's tiles and the matrix × vector's subgroup forms), subgroups and WGSL's
// subgroup_id together (subgroups: the attention and the layer's forms), and WGSL's packed int8 dot (packed). A device
// with subgroups and no subgroup_id is the one where shuffles and subgroups differ, and the one CI's devices (SwiftShader,
// lavapipe: all four there) never are: a worker that reads the one for the other offers the subgroup forms of the
// matrix × vector to no device that has no subgroup_id, or marks them "no subgroup_id" on a device that has it, and
// nothing else in the tests sees it. Each case is a process of its own: the steps share a module's state, made once.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { runtimeUrl } from "./tree.ts";
import { standIn } from "./fakes/webgpu.ts";

const CASES = [];
for (const subgroups of [false, true]) for (const id of [false, true]) for (const packed of [false, true]) CASES.push({ subgroups, id, packed, half: subgroups !== packed });

if (process.argv[2] !== "case") {
  for (const one of CASES) {
    const ran = spawnSync(process.execPath, [import.meta.filename, "case", JSON.stringify(one)], { encoding: "utf8", timeout: 120000 });
    assert.equal(ran.status, 0, `${JSON.stringify(one)}:\n${ran.stdout}${ran.stderr}`);
  }
  console.log(`bench-features-check: the matrix × vector's forms by the device's features, ${CASES.length} cases of subgroups, subgroup_id and the packed dot: ok`);
  process.exit(0);
}

const { subgroups, id, packed, half } = JSON.parse(process.argv[3]);
const language = [...(id ? ["subgroup_id"] : []), ...(packed ? ["packed_4x8_integer_dot_product"] : [])];
// (a fallback adapter: one submission each, and the matrix small enough to be measured)
const { device, adapter } = standIn({ fallback: true, language });
for (const feature of [...(subgroups ? ["subgroups"] : []), ...(half ? ["shader-f16"] : [])]) { device.features.add(feature); adapter.features.add(feature); }
// (the fake compiles asynchronously only; the worker's fixed pipelines are made at once)
device.createComputePipeline = ({ compute }) => ({ code: compute.module.code, getBindGroupLayout: () => ({}) });
// (the CPU's kernel the step times beside the GPU, from the built folder)
globalThis.fetch = async (url) => new Response(fs.readFileSync(fileURLToPath(url)));
globalThis.self = globalThis;
const posted = [];
globalThis.postMessage = (message) => posted.push(message);
let receiver;
Object.defineProperty(globalThis, "onmessage", { configurable: true, get: () => receiver, set: (handler) => { receiver = handler; } });
await import(runtimeUrl("benchmark/gpu.js"));
const asked = async (data) => {
  posted.length = 0;
  await receiver({ data });
  const answer = posted.find((message) => message.step === data.step);
  assert.ok(answer && !answer.error, `${data.step}: ${answer?.error}`);
  return answer.result;
};
const label = JSON.stringify({ subgroups, id, packed, half });

const about = await asked({ step: "info" });
assert.equal(about.packed, packed, `${label}: info says the packed dot is ${packed}`);
const rows = (await asked({ step: "bandwidth", shape: [256, 256] })).rows;
const named = (text) => rows.filter((row) => row.shader.includes(text) && !row.again);
// subgroup forms are offered to a device that has the subgroups, and say "no subgroup_id" only where WGSL has none; the
// packed forms say "no packed int8 dot" only where WGSL has none
for (const row of rows) {
  if (row.again) continue;
  const subgroupForm = row.shader.endsWith(", subgroups");
  assert.ok(!subgroupForm || subgroups, `${label}: "${row.shader}" is offered to a device without subgroups`);
  const wants = [...(subgroupForm && !id ? ["no subgroup_id in this WGSL"] : []), ...(/MMVQ|DP4A|packed int8/.test(row.shader) && !packed ? ["no packed int8 dot here"] : [])];
  if (wants.length) assert.ok(wants.includes(row.none), `${label}: "${row.shader}" says ${JSON.stringify(row.none)}, not one of ${JSON.stringify(wants)}`);
  else assert.equal(row.none, undefined, `${label}: "${row.shader}" says ${JSON.stringify(row.none)} where nothing is missing`);
}
for (const base of ["llama.cpp MMVQ", "llama.cpp mul_mat_vec (engine)"]) {
  const forms = named(base).map((row) => row.shader.endsWith(", subgroups"));
  assert.deepEqual(forms, subgroups ? [false, true] : [false], `${label}: ${base} is offered with${subgroups ? "" : "out"} its subgroup form`);
}

// the prompt's tiled forms are shaders.ts's promptForms for the device's shader-f16, its subgroups alone (shuffles: the
// tiles use subgroupShuffle and no subgroup_id) and the packed dot, which this case says itself
const { promptForms } = await import(runtimeUrl("shaders.js"));
const expected = promptForms({ half, subgroups, packed, memory: device.limits.maxComputeWorkgroupStorageSize,
  threads: Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX) }).map((form) => form.name);
assert.ok(expected.length > 0, `${label}: the case has tiled forms`);
const promptRows = (await asked({ step: "prompt", counts: [16] })).rows.filter((row) => row.shader !== "batched (T135)" && !row.again);
assert.deepEqual([...new Set(promptRows.map((row) => row.shader))], expected, `${label}: the prompt's tiled forms`);

// the layer's forms with subgroups are offered where the device has subgroups and WGSL subgroup_id both
const layerRows = (await asked({ step: "layer" })).rows;
assert.equal(layerRows.some((row) => row.subgroups), subgroups && id, `${label}: the layer's forms with subgroups (${layerRows.filter((row) => row.subgroups).length})`);
assert.equal(layerRows.some((row) => /flash_attn_vec \(subgroups\)/.test(row.form)), subgroups && id, `${label}: flash_attn_vec with subgroups`);
assert.equal(layerRows.some((row) => /DP4A/.test(row.form) && !row.none), packed, `${label}: the DP4A forms run where the packed dot is`);

// the attention of a token is tried with the prompt's tile the engine makes here (f16 where shader-f16, subgroups where
// the device's subgroups and subgroup_id both are) and flash_attn_vec with subgroups where those are
const flash = subgroups && id;
const attentions = (await asked({ step: "layer steps" })).lengths.rows.map((row) => row.attention);
assert.deepEqual(attentions, ["the prompt's tiles (flash attention's tile)",
  ...(half || flash ? [`the prompt's tiles${half ? ", f16" : ""}${flash ? ", subgroups" : ""} (the engine's here)`] : []),
  ...(flash ? ["flash_attn_vec (subgroups)"] : []), "flash_attn_vec"], `${label}: the attention forms of a token`);
console.log(`bench-features-check: ${label}: ok`);
