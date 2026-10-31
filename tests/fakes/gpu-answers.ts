// tests/fakes/gpu-answers.ts (T.465.1)
// The engine's shaders answered for tests/fakes/webgpu.mjs's answer: a dispatch of a pipeline whose WGSL is named here
// has its output written as that shader would write it, so that the model's GPU worker can choose its forms (gpu/forms.ts
// checks each candidate's output before it takes it) on a device that runs no shader. For the checks that run those
// parts: gpu-forms-check.ts. What a shader computes is tests/gpu_answers.py's, in NumPy (the owner's, 2026-10-31:
// Python where it can be): this is the bridge alone, which reads a dispatch's bindings, hands their bytes to Python in
// Pyodide and writes back what it returns.
//
//   const answers = await gpuAnswers();
//   answers.add(code, model)        a pipeline of this WGSL is answered by model (one of answers.flashTile(size),
//   answers.addPipeline(p, model)   answers.quantize, answers.tiled({ packed, packing })), or a pipeline object itself
//   standIn({ answer: answers.answer })
//
// The same as the shaders: what each writes, where its bindings say, and nothing where the workgroups dispatched are
// fewer than its work's tiles (T.465.1's review: a dispatch of too few, the worker's own count, is seen here). Simpler:
// the whole output or none (which tiles a shader numbers how is not held: CI's gpu-check is where the tiles are). A
// pipeline of no WGSL named here is not answered (its output stays as it was written).
import fs from "node:fs";
import { loadPyodide } from "pyodide";
import { placePython, treeOf, runtimeUrl } from "../tree.mjs";
const { FLASH_Q_TILE } = await import(runtimeUrl("shaders/attention.ts"));
const { CODEBOOK_BYTES } = await import(runtimeUrl("shaders/prompt.ts"));

// what the stand-in device (tests/fakes/webgpu.mjs) hands an answer: a binding's buffer (its bytes in data), or a part of one
type Buffer = { size: number; data: Uint8Array };
type Entry = { binding: number; resource: Buffer | { buffer: Buffer; offset?: number; size?: number } };
type Model = (entries: Entry[], x: number, y: number) => void;
type Pipeline = { code: string };

// a binding's bytes: its buffer's, from its offset (a part of one, T.155's pieces of a matrix)
const bound = (entries: Entry[], binding: number): Uint8Array => {
  const { resource } = entries.find((entry) => entry.binding === binding)!;
  const whole = "buffer" in resource ? resource.buffer : resource, offset = "buffer" in resource ? resource.offset ?? 0 : 0;
  const size = "buffer" in resource ? resource.size : undefined;
  return whole.data.subarray(offset, offset + (size ?? whole.size - offset));
};
const has = (entries: Entry[], binding: number) => entries.some((entry) => entry.binding === binding);
const words = (entries: Entry[], binding: number) => { const b = bound(entries, binding); return new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const float = (entries: Entry[], binding: number, at: number) => { const b = bound(entries, binding); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4)[at]; };

// (one Pyodide a process, whatever the answerers made)
// (gpu_answers.py as Pyodide hands it: each function returns a PyProxy of bytes or of a tuple of them)
type Python = Record<string, (...args: unknown[]) => { toJs(): Uint8Array | Uint8Array[]; destroy?(): void }>;
let loaded: Promise<Python> | undefined;
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
  const called = (name: string, ...args: unknown[]): Uint8Array[] => {
    const result = python[name](...args), out = result.toJs();
    result.destroy?.();
    return Array.isArray(out) ? out : [out];
  };
  const byCode = new Map<string, Model>(), byPipeline = new Map<Pipeline, Model>(), seen: string[] = [];
  return {
    add(code: string, model: Model) { byCode.set(code, model); },
    addPipeline(pipeline: Pipeline, model: Model) { byPipeline.set(pipeline, model); },
    // (the WGSL texts dispatched that were answered, in order)
    seen,
    answer(pipeline: Pipeline, entries: Entry[], x: number, y: number) {
      const model = byPipeline.get(pipeline) ?? byCode.get(pipeline.code);
      if (!model) return;
      seen.push(pipeline.code);
      model(entries, x, y);
    },
    // the prompt's attention (flashTile): q, the keys, the values, out, the params (heads, kv heads, scale, window), the block (tokens, pos)
    // (a workgroup for every head and tile of FLASH_Q_TILE tokens)
    flashTile: (size: number): Model => (entries, x) => {
      const [heads, kvHeads, , window] = words(entries, 4), [tokens, pos] = words(entries, 5);
      if (x < heads * Math.ceil(tokens / FLASH_Q_TILE)) return;
      const [out] = called("flash_tile", bound(entries, 0), bound(entries, 1), bound(entries, 2), heads, kvHeads, float(entries, 4, 2), window, tokens, pos, size);
      bound(entries, 3).set(out);
    },
    // QUANTIZE: x, its codes, their scales, the params (n, x's stride), the block (tokens)
    // (64 groups of 32 a workgroup across, a token a workgroup down)
    quantize: ((entries, x, y) => {
      const [n, stride] = words(entries, 3), [tokens] = words(entries, 4);
      if (x < Math.ceil(n / 32 / 64) || y < tokens) return;
      const [codes, scales] = called("quantize", bound(entries, 0), n, stride, tokens);
      bound(entries, 1).set(codes.subarray(0, bound(entries, 1).length));
      bound(entries, 2).set(scales.subarray(0, bound(entries, 2).length));
    }) as Model,
    // (T.465.2: the codebooks of llama.cpp's IQ types, as a checkpoint carries them: what a plan's codebooks are to hold)
    codebooks: () => called("codebooks")[0],
    // ("none" to Python is undefined: a null is jsnull there)
    // a tiled form: the weights, their scales, x (packed: its codes, their scales at binding 6), y, the shape (rows, n / 4,
    // -, -, x's stride, y's stride, add), the block (tokens)
    // (a workgroup for every tile of the form's rows by tokens, numbered over x and then y)
    tiled: ({ packed, packing, tile }: { packed: boolean; packing: string; tile: { rows: number; tokens: number } }): Model => (entries, x, down) => {
      const [rows, quarter, , , xStride, yStride, add] = words(entries, 4), [tokens] = words(entries, 5), y = bound(entries, 3);
      if (x * down < Math.ceil(rows / tile.rows) * Math.ceil(tokens / tile.tokens)) return;
      // (T.465.2's review: an IQ type's text binds the codebooks where an int8 one binds its scales, and a K type's binds
      // nothing there: the answer reads its own tables, so a binding missing or wrong would pass unseen otherwise)
      if (packing.startsWith("IQ") && !(has(entries, 1) && bound(entries, 1).length >= CODEBOOK_BYTES)) throw new Error(`gpu-answers: ${packing}'s text without the codebooks bound`);
      if (packing.endsWith("_K") && has(entries, 1)) throw new Error(`gpu-answers: ${packing}'s text binds something where none is`);
      const [out] = called("tiled", packing, bound(entries, 0), has(entries, 1) ? bound(entries, 1) : undefined, bound(entries, 2), packed ? bound(entries, 6) : undefined,
        y, rows, quarter * 4, xStride, yStride, add, tokens);
      y.set(out.subarray(0, y.length));
    },
  };
}
