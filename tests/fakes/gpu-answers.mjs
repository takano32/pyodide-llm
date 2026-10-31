// tests/fakes/gpu-answers.mjs (T.465.1)
// The engine's shaders answered for tests/fakes/webgpu.mjs's answer: a dispatch of a pipeline whose WGSL is named here
// has its output written as that shader would write it, so that the model's GPU worker can choose its forms (gpu/forms.ts
// checks each candidate's output before it takes it) on a device that runs no shader. For the checks that run those
// parts: gpu-forms-check.mjs. What a shader computes is tests/gpu_answers.py's, in NumPy (the owner's, 2026-10-31:
// Python where it can be): this is the bridge alone, which reads a dispatch's bindings, hands their bytes to Python in
// Pyodide and writes back what it returns.
//
//   const answers = await gpuAnswers();
//   answers.add(code, model)        a pipeline of this WGSL is answered by model (one of answers.flashTile(size),
//   answers.addPipeline(p, model)   answers.quantize, answers.tiled({ packed, packing })), or a pipeline object itself
//   standIn({ answer: answers.answer })
//
// The same as the shaders: what each writes, where its bindings say. Simpler: the whole output whatever the workgroups
// dispatched (a shader that dispatched too few tiles is not seen here: CI's gpu-check is where the tiles are). A
// pipeline of no WGSL named here is not answered (its output stays as it was written).
import fs from "node:fs";
import { loadPyodide } from "pyodide";
import { placePython, treeOf } from "../tree.mjs";

// a binding's bytes: its buffer's, from its offset (a part of one, T.155's pieces of a matrix)
const bound = (entries, binding) => {
  const { resource } = entries.find((entry) => entry.binding === binding);
  const buffer = resource.buffer ?? resource, offset = resource.offset ?? 0;
  return buffer.data.subarray(offset, offset + (resource.size ?? buffer.size - offset));
};
const has = (entries, binding) => entries.some((entry) => entry.binding === binding);
const words = (entries, binding) => { const b = bound(entries, binding); return new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const float = (entries, binding, at) => { const b = bound(entries, binding); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4)[at]; };

// (one Pyodide a process, whatever the answerers made)
let loaded;
const pythonOf = () => (loaded ??= (async () => {
  const py = await loadPyodide();
  await py.loadPackage("numpy", { messageCallback: () => {} });
  placePython(py, treeOf());
  py.FS.writeFile("gpu_answers.py", fs.readFileSync(new URL("../gpu_answers.py", import.meta.url)));
  return py.pyimport("gpu_answers");
})());

export async function gpuAnswers() {
  const python = await pythonOf();
  // what Python returns (bytes, or a tuple of them) as Uint8Arrays, and the proxies let go
  const called = (name, ...args) => {
    const result = python[name](...args), out = result.toJs();
    result.destroy?.();
    return Array.isArray(out) ? out : [out];
  };
  const byCode = new Map(), byPipeline = new Map(), seen = [];
  return {
    add(code, model) { byCode.set(code, model); },
    addPipeline(pipeline, model) { byPipeline.set(pipeline, model); },
    // (the WGSL texts dispatched that were answered, in order)
    seen,
    answer(pipeline, entries) {
      const model = byPipeline.get(pipeline) ?? byCode.get(pipeline.code);
      if (!model) return;
      seen.push(pipeline.code);
      model(entries);
    },
    // the prompt's attention (flashTile): q, the keys, the values, out, the params (heads, kv heads, scale, window), the block (tokens, pos)
    flashTile: (size) => (entries) => {
      const [heads, kvHeads, , window] = words(entries, 4), [tokens, pos] = words(entries, 5);
      const [out] = called("flash_tile", bound(entries, 0), bound(entries, 1), bound(entries, 2), heads, kvHeads, float(entries, 4, 2), window, tokens, pos, size);
      bound(entries, 3).set(out);
    },
    // QUANTIZE: x, its codes, their scales, the params (n, x's stride), the block (tokens)
    quantize: (entries) => {
      const [n, stride] = words(entries, 3), [tokens] = words(entries, 4);
      const [codes, scales] = called("quantize", bound(entries, 0), n, stride, tokens);
      bound(entries, 1).set(codes.subarray(0, bound(entries, 1).length));
      bound(entries, 2).set(scales.subarray(0, bound(entries, 2).length));
    },
    // ("none" to Python is undefined: a null is jsnull there)
    // a tiled form: the weights, their scales, x (packed: its codes, their scales at binding 6), y, the shape (rows, n / 4,
    // -, -, x's stride, y's stride, add), the block (tokens)
    tiled: ({ packed, packing }) => (entries) => {
      const [rows, quarter, , , xStride, yStride, add] = words(entries, 4), [tokens] = words(entries, 5), y = bound(entries, 3);
      const [out] = called("tiled", packing, bound(entries, 0), has(entries, 1) ? bound(entries, 1) : undefined, bound(entries, 2), packed ? bound(entries, 6) : undefined,
        y, rows, quarter * 4, xStride, yStride, add, tokens);
      y.set(out.subarray(0, y.length));
    },
  };
}
