// T344: the activations and gates of the GPU's small steps, each shader alone on a real device (Dawn + lavapipe in Node),
// against JavaScript in float64: GEGLU_ERF (a Spark-X2.5's exact GELU in its gate), GEGLU (the tanh form, a Gemma 3's)
// and HEAD_GATE (a Spark-X2.5's gate of each head of the attention's output). Why a check of its own: erf's GELU and the
// tanh form are 4.7e-4 apart at the most (near |x| = 2), and a model's keys and values carry float16's rounding and
// the 8-bit activations (gpu-check's lines are a few E16 and Q8), so a GPU that took one for the other passed every line
// of gpu-check on the made-up Spark (T344's break, run 44729139997); a gate read from another head moves a model's
// numbers by more, but the shader is checked here by itself too. The fused matrices' "geglu_erf" write takes the same
// function (steps.ts's GELU_ERF_FN): this checks that its text holds it.
//   node tests/activations-check.ts <the webgpu package's directory>
// (a second or two; gpu-prompt.yml's Dawn job runs it)
import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { runtimeUrl } from "./tree.ts";

const [webgpu] = process.argv.slice(2);
if (!webgpu) {
  console.error("node tests/activations-check.ts <the webgpu package's directory>");
  process.exit(2);
}
const { create, globals } = await import(pathToFileURL(path.resolve(webgpu, "index.js")).href);
Object.assign(globalThis, globals);
// (the instance kept, as tests/sample-gpu-bench.ts says why)
Object.defineProperty(globalThis, "navigator", { value: { gpu: create([]) }, configurable: true });
const { ENGINE } = await import(runtimeUrl("shaders.js"));
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();
const { STORAGE, COPY_DST, COPY_SRC, UNIFORM, MAP_READ } = GPUBufferUsage;

const made = (data, usage = STORAGE | COPY_DST | COPY_SRC) => {
  const buffer = device.createBuffer({ size: Math.ceil(data.byteLength / 16) * 16, usage });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};
// run a shader of four bindings (the first written in place) over tokens of n values each, and read the first back
async function ran(code, first, second, size, tokens, n) {
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const target = made(first), buffers = [target, made(second), made(size, UNIFORM | COPY_DST), made(Uint32Array.of(tokens, 0, 0, 0), UNIFORM | COPY_DST)];
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const back = device.createBuffer({ size: target.size, usage: MAP_READ | COPY_DST });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(n / 64), tokens);
  pass.end();
  encoder.copyBufferToBuffer(target, 0, back, 0, target.size);
  device.queue.submit([encoder.finish()]);
  await back.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(back.getMappedRange().slice(0, first.byteLength));
  back.unmap();
  return out;
}

// the values: a ramp over [-12, 12] (both tails, 0, the region where the two GELUs differ most) and its ups, two tokens
const N = 1000, TOKENS = 2;
const x = Float32Array.from({ length: N * TOKENS }, (_, i) => -12 + (24 * (i % N)) / (N - 1));
const up = Float32Array.from({ length: N * TOKENS }, (_, i) => 1 + ((i * 7919) % 13) / 13);
// erf by Abramowitz and Stegun's 7.1.26, as the shaders and the CPU's kernel have it (within 1.5e-7 of erf), in float64
const erfGelu = (v) => {
  const z = Math.abs(v) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z));
  return 0.5 * v * (1 + Math.sign(v) * y);
};
const tanhGelu = (v) => 0.5 * v * (1 + Math.tanh(Math.min(9.010913, Math.max(-9.010913, 0.7978845608028654 * (v + 0.044715 * v ** 3)))));
// how far a shader's values are from JavaScript's, each over the larger of 1 and the value (float32's own rounding of
// a few operations: some 1e-7; the other GELU's is 4.7e-4 off)
const apart = (got, want) => got.reduce((most, value, i) => Math.max(most, Math.abs(value - want[i]) / Math.max(1, Math.abs(want[i]))), 0);
const LINE = 1e-5;
const failures = [];
const size = Uint32Array.of(N, 0, 0, 0);
for (const [name, gelu, other] of [["GEGLU_ERF", erfGelu, tanhGelu], ["GEGLU", tanhGelu, erfGelu]]) {
  const got = await ran(ENGINE[name], x, up, size, TOKENS, N);
  const want = Array.from(x, (v, i) => gelu(v) * up[i]), wrong = Array.from(x, (v, i) => other(v) * up[i]);
  const off = apart(got, want), offOther = apart(got, wrong);
  console.log(`${name}: ${off.toExponential(2)} from its GELU (line ${LINE}), ${offOther.toExponential(2)} from the other`);
  if (!(off <= LINE)) failures.push(`${name} is ${off.toExponential(2)} from its GELU`);
  if (!(offOther > 10 * LINE)) failures.push(`${name} is as near the other GELU (${offOther.toExponential(2)}): the check does not tell them apart`);
}
// HEAD_GATE: 8 heads of 16 values a token, each divided by 1 + exp(-its head's gate)
const HEADS = 8, HEAD = 16, n = HEADS * HEAD;
const values = Float32Array.from({ length: n * TOKENS }, (_, i) => Math.sin(i * 0.37) * 3);
const gates = Float32Array.from({ length: HEADS * TOKENS }, (_, i) => -4 + i * 0.6);
const gated = await ran(ENGINE.HEAD_GATE, values, gates, Uint32Array.of(n, HEAD, 0, 0), TOKENS, n);
const wantGated = Array.from(values, (v, i) => {
  const t = Math.floor(i / n), h = Math.floor((i % n) / HEAD);
  return v / (1 + Math.exp(-gates[t * HEADS + h]));
});
const gateOff = apart(gated, wantGated);
console.log(`HEAD_GATE: ${gateOff.toExponential(2)} from JavaScript's (line ${LINE})`);
if (!(gateOff <= LINE)) failures.push(`HEAD_GATE is ${gateOff.toExponential(2)} from JavaScript's`);
// the fused matrices' write of "geglu_erf" calls the same function, whose text it holds
for (const subgroups of [false, true]) {
  assert.match(ENGINE.fusedMatVec({ input: "norm", output: "geglu_erf", subgroups }), /fn gelu_erf\(a: f32\) -> f32[\s\S]*dst\[row\] = gelu_erf\(value\) \* up;/);
}
assert.match(ENGINE.fusedDp4aMatVec({ output: "geglu_erf", weights: "int8" }), /fn gelu_erf\(a: f32\) -> f32[\s\S]*dst\[row\] = gelu_erf\(value\) \* up;/);
device.destroy();
if (failures.length) {
  console.log(`activations-check: FAILED\n- ${failures.join("\n- ")}`);
  process.exit(1);
}
console.log("activations-check: ok");
process.exit(0);
