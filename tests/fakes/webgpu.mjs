// tests/fakes/webgpu.mjs (T568.1)
// A stand-in for WebGPU as the model's GPU worker (src/runtime/gpu/) sees it: an adapter, and a device that writes
// down what it is asked (calls: every buffer made, write (with a copy of its bytes), clear, copy, pipeline compiled, dispatch, submission, destroy, the
// device gone) and keeps the bytes of every buffer. For the checks that run that worker's parts without a GPU:
// gpu-worker-parts-check.mjs, unchanged-gpusteps.mjs.
//
//   const { calls, device, adapter, lose } = standIn({ fallback, language })
//
//   It puts navigator.gpu on the global (an adapter that is a fallback where fallback is true, and the WGSL language
//   features given), whose device every module of the worker then takes. lose(info) loses the device.
//
// The same as WebGPU: a write's offset and size of a typed array count its elements, and only whole words go (it
// throws on another); a write past its buffer throws; a copy moves the bytes; a mapped range is a copy of the bytes.
// Simpler: no shader runs (a dispatch is written down, and the buffers stay as they were), a pipeline is its WGSL
// ({ code }), a bind group its entries, a copy is done as it is encoded (nothing runs between the encoding and the
// submission), and no error scope ever holds an error.
import assert from "node:assert/strict";

export function standIn({ fallback = false, language = [] } = {}) {
  const calls = [];
  let losing;
  const limits = { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535,
    maxComputeWorkgroupStorageSize: 16384, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, minStorageBufferOffsetAlignment: 256 };
  const bytesOf = (data) => (data instanceof ArrayBuffer || data instanceof SharedArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  const device = {
    features: new Set(), limits, lost: new Promise((resolve) => { losing = resolve; }),
    onSubmit: null,
    createBuffer({ size, usage }) {
      const made = { size, usage, data: new Uint8Array(size), destroyed: false, destroy() { made.destroyed = true; calls.push(["destroy", made]); },
        async mapAsync() {}, getMappedRange: (from = 0, length = size - from) => made.data.slice(from, from + length).buffer, unmap() {} };
      calls.push(["buffer", made]);
      return made;
    },
    queue: {
      // (as WebGPU has it: the offset and the size of a typed array's data count its elements)
      writeBuffer(target, offset, data, from = 0, size) {
        const each = ArrayBuffer.isView(data) ? data.BYTES_PER_ELEMENT : 1, bytes = bytesOf(data);
        const length = size === undefined ? bytes.length - from * each : size * each;
        assert.ok(offset % 4 === 0 && length % 4 === 0, `a write of ${length} bytes at ${offset}: WebGPU takes whole words`);
        assert.ok(offset + length <= target.size, "a write past its buffer");
        target.data.set(bytes.subarray(from * each, from * each + length), offset);
        calls.push(["write", target, offset, length, bytes.slice(from * each, from * each + length)]);
      },
      submit() { calls.push(["submit"]); device.onSubmit?.(); },
      onSubmittedWorkDone: async () => {},
    },
    createShaderModule: ({ code }) => ({ code }),
    async createComputePipelineAsync({ compute }) {
      const pipeline = { code: compute.module.code, getBindGroupLayout: () => ({}) };
      calls.push(["pipeline", pipeline]);
      return pipeline;
    },
    createBindGroup: ({ entries }) => ({ entries }),
    createCommandEncoder() {
      return {
        beginComputePass() {
          const pass = { setPipeline(pipeline) { pass.pipeline = pipeline; }, setBindGroup(index, group) { pass.group = group; },
            dispatchWorkgroups(x, y, z) { calls.push(["dispatch", pass.pipeline, pass.group, x, y, z]); }, end() {} };
          return pass;
        },
        // (done as it is encoded: nothing here runs between the encoding and the submission)
        clearBuffer(target, offset = 0, size = target.size - offset) {
          target.data.fill(0, offset, offset + size);
          calls.push(["clear", target, offset, size]);
        },
        copyBufferToBuffer(source, from, target, to, length) {
          target.data.set(source.data.subarray(from, from + length), to);
          calls.push(["copy", source, from, target, to, length]);
        },
        finish: () => ({}),
      };
    },
    pushErrorScope() {}, popErrorScope: async () => null, destroy() { calls.push(["device gone"]); },
  };
  const adapter = { info: { vendor: "stand-in", architecture: "", device: "", description: "writes down what it is asked", isFallbackAdapter: fallback,
    subgroupMinSize: 4, subgroupMaxSize: 16 }, limits, features: new Set(), requestDevice: async () => device };
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { userAgent: "gpu-worker-parts-check",
    gpu: { requestAdapter: async () => adapter, wgslLanguageFeatures: new Set(language) } } });
  return { calls, device, adapter, lose: (info) => losing(info) };
}
