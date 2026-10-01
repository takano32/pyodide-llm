// T224's review (throwaway): which WGSL subgroup operation lavapipe gets wrong at a subgroup of 32 (LP_NATIVE_VECTOR_WIDTH
// 1024), where flash_attn_vec's split shader fails and everything else of the probe's passes (4, 8 and 16).
//   LP_NATIVE_VECTOR_WIDTH=1024 VK_ICD_FILENAMES=... node t224-subgroup-ops.mjs <the npm package webgpu's dir>
import path from "node:path";
import { pathToFileURL } from "node:url";

const [webgpuDir] = process.argv.slice(2);
const { create, globals } = await import(pathToFileURL(path.resolve(webgpuDir, "index.js")).href);
Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredFeatures: ["subgroups"].filter((f) => adapter.features.has(f)),
  requiredLimits: { maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX } });
const info = adapter.info ?? {};
console.log(`adapter: ${info.vendor} ${info.architecture}, subgroups ${info.subgroupMinSize}..${info.subgroupMaxSize}, LP_NATIVE_VECTOR_WIDTH ${process.env.LP_NATIVE_VECTOR_WIDTH ?? "(default)"}`);

const OPS = ["size", "lane", "subgroup_id", "num_subgroups", "add", "max", "down1", "down2", "down4", "down8", "down16", "shuffle3", "add4", "down4v", "down16v", "bcast4", "max-and-add-loop"];
const code = (size) => /* wgsl */ `
diagnostic(off, subgroup_uniformity);
enable subgroups;
requires subgroup_id;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
const N: u32 = ${OPS.length}u;
@compute @workgroup_size(${size})
fn main(@builtin(local_invocation_index) i: u32,
        @builtin(subgroup_id) sid: u32,
        @builtin(num_subgroups) nsg: u32,
        @builtin(subgroup_size) size: u32,
        @builtin(subgroup_invocation_id) lane: u32) {
  let v = f32(lane);
  out[0u * ${size}u + i] = f32(size);
  out[1u * ${size}u + i] = v;
  out[2u * ${size}u + i] = f32(sid);
  out[3u * ${size}u + i] = f32(nsg);
  out[4u * ${size}u + i] = subgroupAdd(v);
  out[5u * ${size}u + i] = subgroupMax(v);
  out[6u * ${size}u + i] = subgroupShuffleDown(v, 1u);
  out[7u * ${size}u + i] = subgroupShuffleDown(v, 2u);
  out[8u * ${size}u + i] = subgroupShuffleDown(v, 4u);
  out[9u * ${size}u + i] = subgroupShuffleDown(v, 8u);
  out[10u * ${size}u + i] = subgroupShuffleDown(v, 16u);
  out[11u * ${size}u + i] = subgroupShuffle(v, (lane * 3u) % size);
  let w = vec4<f32>(v, v + 100.0, v + 200.0, v + 300.0);
  out[12u * ${size}u + i] = subgroupAdd(w).y;
  out[13u * ${size}u + i] = subgroupShuffleDown(w, 4u).y;
  out[14u * ${size}u + i] = subgroupShuffleDown(w, 16u).y;
  out[15u * ${size}u + i] = subgroupShuffle(v, 4u * (lane / 4u));
  // flash_attn_vec's softmax: the largest and the sum of a row of 32 values over the lanes, as its two passes do
  var final_max = -1.0e9;
  var total = 0.0;
  for (var off = 0u; off < 32u; off += size) {
    let idx = off + lane;
    final_max = subgroupMax(max(final_max, select(-1.0e9, f32(idx), idx < 32u)));
  }
  for (var off = 0u; off < 32u; off += size) {
    let idx = off + lane;
    total += subgroupAdd(select(0.0, f32(idx) - final_max + 1.0, idx < 32u));
  }
  out[16u * ${size}u + i] = final_max * 1000.0 + total;
}`;

const STORAGE = GPUBufferUsage.STORAGE, COPY_SRC = GPUBufferUsage.COPY_SRC, MAP_READ = GPUBufferUsage.MAP_READ, COPY_DST = GPUBufferUsage.COPY_DST;
let bad = 0;
for (const size of [32, 64, 128]) {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: code(size) });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  const error = await device.popErrorScope();
  if (error) {
    const messages = (await module.getCompilationInfo()).messages.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ");
    console.log(`workgroup ${size}: ${error.message} ${messages}`);
    bad++;
    continue;
  }
  const out = device.createBuffer({ size: OPS.length * size * 4, usage: STORAGE | COPY_SRC });
  const read = device.createBuffer({ size: OPS.length * size * 4, usage: MAP_READ | COPY_DST });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(out, 0, read, 0, OPS.length * size * 4);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const got = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  const at = (op, i) => got[OPS.indexOf(op) * size + i];
  const sg = at("size", 0), problems = [];
  const lanes = [...Array(size).keys()].map((i) => at("lane", i));
  for (let i = 0; i < size; i++) {
    const lane = lanes[i], base = i - lane;  // the lane's subgroup starts at i - lane (subgroups in order)
    const expected = {
      add: (sg * (sg - 1)) / 2, max: sg - 1, shuffle3: (lane * 3) % sg, add4: (sg * (sg - 1)) / 2 + 100 * sg, bcast4: 4 * Math.floor(lane / 4),
    };
    for (const d of [1, 2, 4, 8, 16]) if (lane + d < sg) expected[`down${d}`] = lane + d;
    if (lane + 4 < sg) expected.down4v = lane + 4 + 100;
    if (lane + 16 < sg) expected.down16v = lane + 16 + 100;
    let total = 0;
    for (let k = 0; k < 32; k++) total += k - 31 + 1;
    expected["max-and-add-loop"] = 31 * 1000 + total;
    for (const [op, want] of Object.entries(expected)) {
      const x = at(op, i);
      if (Math.abs(x - want) > 1e-3) problems.push(`${op} at lane ${lane} (invocation ${i}): ${x}, expected ${want}`);
    }
  }
  const byOp = {};
  for (const p of problems) byOp[p.split(" ")[0]] = (byOp[p.split(" ")[0]] ?? 0) + 1;
  console.log(`workgroup ${size}: subgroup_size ${sg}, num_subgroups ${at("num_subgroups", 0)}, subgroup_id of the last invocation ${at("subgroup_id", size - 1)}, ` +
    `${problems.length ? `WRONG in ${Object.entries(byOp).map(([op, n]) => `${op} (${n} lanes)`).join(", ")}; first: ${problems.slice(0, 3).join(" | ")}` : "every operation right"}`);
  bad += problems.length ? 1 : 0;
}
process.exit(bad ? 1 : 0);
