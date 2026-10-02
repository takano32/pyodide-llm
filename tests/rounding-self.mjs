// T232's review: tests/rounding.mjs's rounding of a float32 to a float16 (toward zero, away from zero), checked alone, on a
// device of Dawn + lavapipe in Node, against a JavaScript reference: every value goes through pack2x16float in a shader of
// its own (the wrapper rewrites the call as it does the repository's shaders) and comes back as float16 bits. Nobody
// had looked at what the emulation does to a NaN and an infinity: both went to the largest finite float16 (65504), and a
// negative overflow went to -infinity under "away"; gpu-check's NaN round of the logits then failed under "everything" on
// models whose attention reads its keys and values through the tiles' conversions, for no fault of the engine's.
//   node tests/rounding-self.mjs <the webgpu package's directory> [toward-zero] [away]
// (a second or two; tests/rounding-check.mjs runs it first)
import path from "node:path";
import { pathToFileURL } from "node:url";

const [webgpu, ...asked] = process.argv.slice(2);
if (!webgpu) {
  console.error("node tests/rounding-self.mjs <the webgpu package's directory> [toward-zero] [away]");
  process.exit(2);
}
const { create, globals } = await import(pathToFileURL(path.resolve(webgpu, "index.js")).href);
Object.assign(globalThis, globals);
const { roundedGpu } = await import("./rounding.mjs");

// the values: the edges of the float16 (the largest finite, an overflow, the smallest normal, the subnormals, zeros), a
// NaN and an infinity of each sign (and NaNs whose fraction is in the low bits alone, which a truncation to ten bits would
// make an infinity), and about 4000 of any bits
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const values = [];
const push = (x) => { f32[0] = x; values.push(u32[0]); };
for (const x of [0, -0, 1, -1, 65504, -65504, 65519, 65520, 65535, 65536, 70000, -70000, 1e30, -1e30, 6.1035e-5, 6.0e-5, 5.96e-8, 3e-8, -3e-8, 1e-10, 1.0009765625, 1.0004, -1.0004, 0.1, -0.1, Infinity, -Infinity, NaN]) push(x);
values.push(0x7f800001, 0xff800001, 0x7fc00000, 0xffc00001, 0x7f81ffff, 0x7fffffff);
// (not the float32 subnormals, which a device may flush to zero on its way in: no keys or values are of 1e-38)
let seed = 12345;
while (values.length < 4096) {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  if (((seed >>> 23) & 0xff) !== 0) values.push(seed);
}

const half = (code) => {
  const sign = code & 0x8000 ? -1 : 1, e = (code >> 10) & 31, m = code & 1023;
  if (e === 31) return m ? NaN : sign * Infinity;
  return e === 0 ? sign * m * 2 ** -24 : sign * (1 + m / 1024) * 2 ** (e - 15);
};
// what a device of this rounding gives: the float16 code (NaN: any with the exponent all ones and a fraction)
const wanted = (bits, how) => {
  u32[0] = bits;
  const x = f32[0], sign = (bits >>> 16) & 0x8000;
  if (Number.isNaN(x)) return "NaN";
  if (!Number.isFinite(x)) return sign | 0x7c00;
  const magnitude = Math.abs(x);
  if (magnitude >= 65504) return sign | 0x7bff;   // an overflow keeps the largest finite, toward zero and (here) away too
  let low = 0, high = 0x7bff;                      // the largest code whose value is not over the magnitude
  while (low < high) { const mid = (low + high + 1) >> 1; if (half(mid) <= magnitude) low = mid; else high = mid - 1; }
  const exact = half(low) === magnitude;
  return sign | (how === "away" && !exact ? low + 1 : low);
};

const hows = asked.length ? asked : ["toward-zero", "away"];
const gpu = create([]);
let failed = 0;
for (const how of hows) {
  const adapter = await roundedGpu(gpu, how).requestAdapter();
  const device = await adapter.requestDevice();
  const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&xs)) { out[id.x] = pack2x16float(vec2<f32>(xs[id.x], 0.0)) & 0xffffu; }
}`;
  const input = device.createBuffer({ size: values.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: values.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: values.length * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(input, 0, new Uint32Array(values));
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } }] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(values.length / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, read, 0, values.length * 4);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const got = new Uint32Array(read.getMappedRange().slice(0));
  read.unmap();
  const wrong = [];
  values.forEach((bits, i) => {
    const want = wanted(bits, how), have = got[i];
    const ok = want === "NaN" ? (have & 0x7c00) === 0x7c00 && (have & 0x3ff) !== 0 : have === want;
    if (!ok) wrong.push(`0x${bits.toString(16)}: 0x${have.toString(16)}, where ${want === "NaN" ? "a NaN" : "0x" + want.toString(16)} is to be`);
  });
  console.log(`rounding emulation ${how}: ${wrong.length ? "FAILED" : "ok"} (${values.length} values)`);
  for (const line of wrong.slice(0, 8)) console.log(`  ${line}`);
  if (wrong.length > 8) console.log(`  and ${wrong.length - 8} more`);
  failed += wrong.length > 0;
  device.destroy();
}
process.exit(failed ? 1 : 0);
