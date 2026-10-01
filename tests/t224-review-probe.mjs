// T224's review (throwaway, not for main): flash_attn_vec (shaders.js's flashVec and flashVecReduce) on Node's Dawn
// with Mesa's lavapipe, where lavapipe's subgroup width is chosen by LP_NATIVE_VECTOR_WIDTH (128: 4, 256: 8, 512: 16,
// 1024: 32). What the hints of the interrupted review (3360 edge cases at widths 4 and 16) did not do:
//   1. a workgroup of more than one subgroup (the subgroups the device reports run from min to max: a real GPU's
//      workgroup is the largest, its subgroup may be smaller, and then only the first works); lavapipe has min = max,
//      so the range is faked: shaders.js's flashVecShape takes subgroupMin and subgroupMax as the device says them;
//   2. a steep head whose largest sits in every part (one head a part, MHA): a part dropped or misplaced then moves
//      that head's whole output, where random keys move an average of tiles by 1/parts of it;
//   3. how far the shipped check's data (4 heads on 2, head 3 steep with its peak in tile 1, 40/70/300/1100 positions)
//      sees a reduce that leaves out one part, and a NaN that stays in one head (tokenAttentionOff's aggregation).
//   LP_NATIVE_VECTOR_WIDTH=512 VK_ICD_FILENAMES=... node t224-review-probe.mjs <the npm package webgpu's dir>
import path from "node:path";
import { pathToFileURL } from "node:url";

const [webgpuDir] = process.argv.slice(2);
const { create, globals } = await import(pathToFileURL(path.resolve(webgpuDir, "index.js")).href);
Object.assign(globalThis, globals);
const W = await import(pathToFileURL(path.resolve("public/shaders.js")).href);

const gpu = create([]);
const adapter = await gpu.requestAdapter();
const features = ["shader-f16", "subgroups"].filter((f) => adapter.features.has(f));
const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: {
  maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
  maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
  maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
const info = adapter.info ?? {};
const threads = Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX);
const hasSubgroups = device.features.has("subgroups") && Boolean(gpu.wgslLanguageFeatures?.has("subgroup_id"));
const actual = info.subgroupMinSize;
console.log(`adapter: ${[info.vendor, info.architecture, info.description].filter(Boolean).join(" ")}, subgroups ${info.subgroupMinSize}..${info.subgroupMaxSize}, ` +
  `threads ${threads}, LP_NATIVE_VECTOR_WIDTH ${process.env.LP_NATIVE_VECTOR_WIDTH ?? "(default)"}, subgroup_id ${hasSubgroups}`);

const STORAGE = GPUBufferUsage.STORAGE, COPY_DST = GPUBufferUsage.COPY_DST, COPY_SRC = GPUBufferUsage.COPY_SRC;
const pipelines = new Map();
async function pipelineOf(code) {
  if (!pipelines.has(code)) {
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const error = await device.popErrorScope();
    if (error) {
      const messages = (await module.getCompilationInfo()).messages.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ");
      throw new Error(`${error.message} ${messages}`);
    }
    pipelines.set(code, pipeline);
  }
  return pipelines.get(code);
}
const put = (data, usage = STORAGE | COPY_DST) => {
  const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage });
  device.queue.writeBuffer(b, 0, data);
  return b;
};
const bind = (pipeline, buffers) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });

// the output of one attention of a token that reads positions, in nwg parts a head (undefined: flashVecSplits'), from
// the code of the split and of the reduce given
async function attend(shape, splitCode, reduceCode, data, { heads, kvHeads, positions, nwg }) {
  const size = shape.headSize, parts = nwg ?? W.flashVecSplits(shape, positions);
  const split = await pipelineOf(splitCode), reduce = await pipelineOf(reduceCode);
  const owned = [];
  const own = (b) => (owned.push(b), b);
  try {
    const q = own(put(data.q)), keys = own(put(data.keys)), values = own(put(data.values));
    const tmp = own(device.createBuffer({ size: W.flashVecPartsBytes(shape, heads) + 16, usage: STORAGE }));
    const out = own(device.createBuffer({ size: heads * size * 4, usage: STORAGE | COPY_SRC }));
    const params = own(put(W.flashVecParams(shape, heads, kvHeads, parts), GPUBufferUsage.UNIFORM | COPY_DST));
    const step = own(put(new Uint32Array([1, positions - 1, 0, 0]), GPUBufferUsage.UNIFORM | COPY_DST));
    const read = own(device.createBuffer({ size: heads * size * 4, usage: GPUBufferUsage.MAP_READ | COPY_DST }));
    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(split);
    pass.setBindGroup(0, bind(split, [q, keys, values, tmp, out, params, step]));
    pass.dispatchWorkgroups(heads * parts);
    if (parts > 1) {
      pass.setPipeline(reduce);
      pass.setBindGroup(0, bind(reduce, [tmp, out, params]));
      pass.dispatchWorkgroups(heads);
    }
    pass.end();
    encoder.copyBufferToBuffer(out, 0, read, 0, heads * size * 4);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
    await read.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    return { got, parts };
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

// shaders.js's tokenAttentionOff with its aggregation made to keep a NaN (Math.max), and the shipped one's result
const fromHalf = (h) => (h & 0x8000 ? -1 : 1) * ((h >> 10) & 31 ? 2 ** (((h >> 10) & 31) - 15) * (1 + (h & 1023) / 1024) : 2 ** -14 * ((h & 1023) / 1024));
function stickyOff(got, { q, keys, values }, { heads, kvHeads, size, positions }) {
  const kvDim = kvHeads * size, scale = 1 / Math.sqrt(size);
  let worst = 0;
  for (let h = 0; h < heads; h++) {
    const kv = Math.floor(h / (heads / kvHeads)) * size, row = h * size, weights = new Float64Array(positions);
    for (let p = 0; p < positions; p++) for (let d = 0; d < size; d++) weights[p] += q[row + d] * fromHalf(keys[p * kvDim + kv + d]) * scale;
    const most = weights.reduce((a, b) => Math.max(a, b), -Infinity);
    let sum = 0, largest = 0;
    for (let p = 0; p < positions; p++) sum += (weights[p] = Math.exp(weights[p] - most));
    for (let p = 0; p < positions; p++) for (let d = 0; d < size; d++) largest = Math.max(largest, Math.abs(fromHalf(values[p * kvDim + kv + d])));
    for (let d = 0; d < size; d++) {
      let want = 0;
      for (let p = 0; p < positions; p++) want += weights[p] * fromHalf(values[p * kvDim + kv + d]);
      worst = Math.max(worst, Math.abs(got[row + d] - want / sum) / largest);
    }
  }
  return worst;
}
const shippedOff = (got, data, dims) => W.tokenAttentionOff(got, data, dims);
const fmt = (x) => (Number.isFinite(x) ? x.toExponential(2) : String(x));

let failures = 0;
const fail = (what) => {
  failures++;
  console.log(`  FAILED: ${what}`);
};

// ---- 1. a workgroup of more than one subgroup (the range of subgroup sizes faked around the actual one)
const LINE = 1e-4;
const ranges = hasSubgroups ? (actual === 4 ? [[4, 4], [4, 16], [4, 32], [2, 8]] : actual === 8 ? [[8, 8], [8, 32], [4, 16], [8, 64]]
  : actual === 16 ? [[16, 16], [16, 64], [8, 32], [4, 128]] : [[actual, actual], [16, 64], [actual, actual * 4]]) : [];
console.log(`1. flash_attn_vec with subgroups, the range of subgroup sizes reported faked (actual ${actual}): ${ranges.map((r) => r.join("..")).join(", ")}`);
const POSITIONS = [1, 31, 33, 64, 65, 129, 300, 1100];
const HEADS = [[4, 2], [14, 2], [32, 8]];
let cases = 0, worstAll = 0;
for (const [min, max] of ranges) {
  for (const size of [64, 128]) {
    const shape = W.flashVecShape({ headSize: size, subgroups: true, threads, subgroupMin: min, subgroupMax: max });
    if (shape.none) {
      console.log(`  ${min}..${max} head ${size}: none (${shape.none})`);
      continue;
    }
    let worst = 0, where = "";
    for (const [heads, kvHeads] of HEADS) {
      for (const positions of POSITIONS) {
        const data = W.tokenAttentionData({ heads, kvHeads, size, positions, steep: [heads - 1] });
        const { got, parts } = await attend(shape, W.flashVec(shape), W.flashVecReduce(shape), data, { heads, kvHeads, positions });
        const off = stickyOff(got, data, { heads, kvHeads, size, positions });
        cases++;
        if (!(off <= worst)) [worst, where] = [off, `${heads}/${kvHeads} heads, ${positions} positions, ${parts} parts`];
        if (!(off <= LINE)) fail(`${min}..${max} head ${size} ${heads}/${kvHeads} heads ${positions} positions ${parts} parts: ${off}`);
      }
    }
    worstAll = Math.max(worstAll, worst);
    console.log(`  ${min}..${max} head ${size} (workgroup ${shape.wgSize}, D_SPLIT ${shape.dSplit}, parts up to ${shape.splits}, reduce ${shape.reduceSize}): worst ${fmt(worst)} at ${where}`);
  }
}
console.log(`  ${cases} cases, worst ${fmt(worstAll)} (line ${LINE})`);

// ---- 2. a steep head a part: MHA, every head's largest score (a key at ±4 in its q's sign, q 40 times) in another tile
function peaks({ heads, size, positions }) {
  const tiles = Math.ceil(positions / 32);
  let stride = 3;
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  while (gcd(stride, tiles) !== 1) stride += 2;
  const q = new Float32Array(heads * size).map(() => (Math.random() * 2 - 1) * W.TOKEN_ATTENTION_STEEP);
  const halfBits = () => ((Math.random() < 0.5 ? 0x8000 : 0) | ((12 + ((Math.random() * 5) | 0)) << 10) | ((Math.random() * 1024) | 0));
  const rows = (positions + W.TOKEN_ATTENTION_PAST) * heads * size;
  const keys = new Uint16Array(rows).map(halfBits), values = new Uint16Array(rows).map(halfBits);
  const at = [];
  for (let h = 0; h < heads; h++) {
    const p = Math.min(32 * ((h * stride + 1) % tiles) + ((h * 11) % 32), positions - 1);
    at.push(p);
    for (let d = 0; d < size; d++) keys[p * heads * size + h * size + d] = q[h * size + d] < 0 ? 0xc400 : 0x4400;
  }
  return { data: { q, keys, values }, at };
}
console.log("2. a steep head a part (MHA, 32 heads, each one's peak in another tile): the default shapes, subgroups and lanes");
const kinds = [...(hasSubgroups ? [["subgroups", true]] : []), ["lanes", false]];
for (const [name, subgroups] of kinds) {
  for (const size of [64, 128]) {
    const shape = W.flashVecShape({ headSize: size, subgroups, threads, subgroupMin: info.subgroupMinSize, subgroupMax: info.subgroupMaxSize });
    if (shape.none) continue;
    let worst = 0, where = "";
    for (const positions of [33, 70, 129, 257, 300, 1100, 2100]) {
      const heads = 32, { data, at } = peaks({ heads, size, positions });
      const { got, parts } = await attend(shape, W.flashVec(shape), W.flashVecReduce(shape), data, { heads, kvHeads: heads, positions });
      const off = stickyOff(got, data, { heads, kvHeads: heads, size, positions });
      if (!(off <= worst)) [worst, where] = [off, `${positions} positions, ${parts} parts, peaks in tiles ${[...new Set(at.map((p) => p >> 5))].length} of ${Math.ceil(positions / 32)}`];
      if (!(off <= LINE)) fail(`${name} head ${size} ${positions} positions ${parts} parts: ${off}`);
    }
    console.log(`  ${name} head ${size}: worst ${fmt(worst)} at ${where}`);
  }
}

// ---- 3. what the shipped check's data sees: a reduce leaving out a part, and a NaN in one head
console.log("3. the shipped check's data (4 heads on 2, head 3 steep, peak at 33) against reduces broken one way each");
const mutations = [
  ["no mutation", (code) => code],
  ["the last part left out", (code) => code.replace("let active_thread = thread < params.nwg;", "let active_thread = thread + 1u < params.nwg;")],
  ["part 5 left out", (code) => code.replace("let active_thread = thread < params.nwg;", "let active_thread = thread < params.nwg && thread != 5u;")],
  ["part 9 left out", (code) => code.replace("let active_thread = thread < params.nwg;", "let active_thread = thread < params.nwg && thread != 9u;")],
  ["head 0's output NaN", (code) => code.replace("dst[(row_base + elem_base) >> 2u] = sum * inv_s;",
    "dst[(row_base + elem_base) >> 2u] = select(sum * inv_s, vec4<f32>(f32(params.unused0) / f32(params.unused0)), rid == 0u);")],
  ["head 2's first four values NaN", (code) => code.replace("dst[(row_base + elem_base) >> 2u] = sum * inv_s;",
    "dst[(row_base + elem_base) >> 2u] = select(sum * inv_s, vec4<f32>(f32(params.unused0) / f32(params.unused0)), rid == 2u && elem_base == 0u);")],
];
const LENGTHS = [40, 70, 300, 1100];
if (hasSubgroups) {
  for (const size of [64, 128]) {
    const shape = W.flashVecShape({ headSize: size, subgroups: true, threads, subgroupMin: info.subgroupMinSize, subgroupMax: info.subgroupMaxSize });
    if (shape.none) continue;
    const heads = 4, kvHeads = 2, splitCode = W.flashVec(shape), reduceCode = W.flashVecReduce(shape);
    for (const [name, edit] of mutations) {
      const mutated = edit(reduceCode);
      if (name !== "no mutation" && mutated === reduceCode) throw new Error(`the mutation "${name}" changed nothing`);
      const cells = [];
      for (const positions of LENGTHS) {
        const data = W.tokenAttentionData({ heads, kvHeads, size, positions, steep: [3] });
        const { got, parts } = await attend(shape, splitCode, mutated, data, { heads, kvHeads, positions });
        const dims = { heads, kvHeads, size, positions };
        const shipped = shippedOff(got, data, dims), sticky = stickyOff(got, data, dims);
        cells.push(`${positions}p/${parts}w: shipped ${fmt(shipped)} ${shipped <= 4e-3 ? "PASSES" : "caught"}, kept-NaN ${fmt(sticky)}${sticky <= 5e-4 ? "" : " caught at 5e-4"}`);
      }
      console.log(`  head ${size} (parts up to ${shape.splits}), ${name}:\n    ${cells.join("\n    ")}`);
    }
  }
} else {
  console.log("  (no subgroups here: part 3 needs them)");
}
console.log(`probe: ${failures} failures`);
process.exit(failures ? 1 : 0);
