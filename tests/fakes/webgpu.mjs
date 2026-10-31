// tests/fakes/webgpu.mjs (T.568.1)
// A stand-in for WebGPU as the model's GPU worker (src/runtime/gpu/) sees it: an adapter, and a device that writes
// down what it is asked (calls: every buffer made, write (with a copy of its bytes), clear, copy, pipeline compiled, dispatch, submission, destroy, the
// device gone) and keeps the bytes of every buffer. For the checks that run that worker's parts without a GPU:
// gpu-worker-parts-check.mjs, unchanged-gpusteps.mjs.
//
//   const { calls, device, adapter, lose } = standIn({ fallback, language, answer })
//
//   It puts navigator.gpu on the global (an adapter that is a fallback where fallback is true, and the WGSL language
//   features given), whose device every module of the worker then takes. lose(info) loses the device.
//   T.465.1: answer(pipeline, entries, x, y, z), where given, is called for every dispatch of a command buffer as it is
//   submitted, with the bind group's entries ({ binding, resource: a buffer or { buffer, offset, size } }): what a
//   shader would write, a check of the worker's may write into the buffers' bytes (tests/fakes/gpu-answers.ts).
//
// The same as WebGPU: a write's offset and size of a typed array count its elements, and only whole words go (it
// throws on another); a write past its buffer throws; a copy moves the bytes; a mapped range is a copy of the bytes.
// The same too (T.465.1's review): what a command buffer does (its copies, clears and the answers of its dispatches) is
// done when it is submitted, in the order it was encoded (a command buffer never submitted does nothing; a write of the
// queue between the encoding and the submission comes first), while calls writes each down as it is encoded.
// Simpler: no shader runs (a dispatch is written down, and the buffers stay as they were, but for what answer writes),
// a pipeline is its WGSL ({ code }), a bind group its entries, and no error scope ever holds an error.
import assert from "node:assert/strict";

export function standIn({ fallback = false, language = [], answer = null } = {}) {
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
      submit(buffers = []) {
        calls.push(["submit"]);
        for (const buffer of buffers) for (const done of buffer.work ?? []) done();
        device.onSubmit?.();
      },
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
      // (what the command buffer does when it is submitted, in order)
      const work = [];
      return {
        beginComputePass() {
          const pass = { setPipeline(pipeline) { pass.pipeline = pipeline; }, setBindGroup(index, group) { pass.group = group; },
            dispatchWorkgroups(x, y = 1, z = 1) {
              const { pipeline, group } = pass;
              calls.push(["dispatch", pipeline, group, x, y, z]);
              if (answer) work.push(() => answer(pipeline, group.entries, x, y, z));
            }, end() {} };
          return pass;
        },
        clearBuffer(target, offset = 0, size = target.size - offset) {
          calls.push(["clear", target, offset, size]);
          work.push(() => target.data.fill(0, offset, offset + size));
        },
        copyBufferToBuffer(source, from, target, to, length) {
          calls.push(["copy", source, from, target, to, length]);
          work.push(() => target.data.set(source.data.subarray(from, from + length), to));
        },
        finish: () => ({ work }),
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
