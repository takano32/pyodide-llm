// T.357.3: a stand-in for the npm package webgpu (what tests/gpu-check.ts's --webgpu names) with no GPU in it: an
// adapter that says what it has and gives no device. Nothing of WebGPU runs: the GPU's worker fails as it opens, every
// run of the harness goes to the CPU, and gpu-check says FAILED of each, as it is to. tests/gpu-check-dry-check.ts
// runs the tool with it, on a machine that runs no WebGPU.
export const globals = {};
const limits = { maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456, maxComputeWorkgroupsPerDimension: 65535,
  maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 1024, maxComputeWorkgroupSizeX: 1024, minStorageBufferOffsetAlignment: 256 };
const adapter = { features: new Set(["shader-f16", "subgroups"]), limits,
  info: { vendor: "none", architecture: "none", device: "", description: "no device", isFallbackAdapter: true, subgroupMinSize: 4, subgroupMaxSize: 16 },
  requestDevice: async () => { throw new Error("the stand-in gives no device"); } };
export const create = () => ({ requestAdapter: async () => adapter, wgslLanguageFeatures: new Set(["packed_4x8_integer_dot_product", "subgroup_id"]) });
