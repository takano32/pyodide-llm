// T232's review (throwaway): the subgroup size Dawn's device on lavapipe reports under LP_NATIVE_VECTOR_WIDTH (the
// tiles' `sg_size == 16u` branch of ORT's DP4A runs at 16 only).
//   node tests/t232-subgroup.mjs <the directory of the npm package webgpu>
import path from "node:path";
import { pathToFileURL } from "node:url";

const { create, globals } = await import(pathToFileURL(path.resolve(process.argv[2], "index.js")).href);
Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await gpu.requestAdapter();
const has = adapter.features.has("subgroups");
console.log(`LP_NATIVE_VECTOR_WIDTH=${process.env.LP_NATIVE_VECTOR_WIDTH ?? "(unset)"}: subgroups ${has}, packed ${gpu.wgslLanguageFeatures.has("packed_4x8_integer_dot_product")}, subgroup_id ${gpu.wgslLanguageFeatures.has("subgroup_id")}`);
if (has) {
  const device = await adapter.requestDevice({ requiredFeatures: ["subgroups"] });
  const module = device.createShaderModule({ code: `enable subgroups;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) i: u32, @builtin(subgroup_size) s: u32) { if (i == 255u) { out[0] = s; } }` });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(buffer, 0, read, 0, 16);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  console.log(`  a workgroup of 256 sees subgroup_size ${new Uint32Array(read.getMappedRange())[0]}; limits: min ${adapter.info?.subgroupMinSize ?? adapter.limits.minSubgroupSize}, max ${adapter.info?.subgroupMaxSize ?? adapter.limits.maxSubgroupSize}`);
}
process.exit(0);
