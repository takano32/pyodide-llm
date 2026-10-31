// tests/fakes/gpu-answers.ts (T.465.1)
// The engine's shaders answered for tests/fakes/webgpu.mjs's answer: a dispatch of a pipeline whose WGSL is named here
// has its output written as that shader would write it, so that the model's GPU worker can choose its forms (gpu/forms.ts
// checks each candidate's output before it takes it) on a device that runs no shader. For the checks that run those
// parts: gpu-forms-check.ts, gpu-tokens-check.ts. What a shader computes is tests/gpu_answers.py's, in NumPy (the owner's, 2026-10-31:
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

// (T.465.3: a token's attention by parts, kept by its parts' buffer for the reduce that comes after: the process's, as
// the one Pyodide is, so that a model of one answerer and the reduce of another meet)
const reducing = new Map<ArrayBufferLike, { out: Uint8Array; nwg: number; stats: number; size: number }>();

export async function gpuAnswers() {
  const python = await pythonOf();
  // what Python returns (bytes, or a tuple of them) as Uint8Arrays, and the proxies let go
  const called = (name: string, ...args: unknown[]): Uint8Array[] => {
    const result = python[name](...args), out = result.toJs();
    result.destroy?.();
    return Array.isArray(out) ? out : [out];
  };
  const byCode = new Map<string, Model>(), byPipeline = new Map<Pipeline, Model>(), seen: string[] = [], missed: string[] = [];
  return {
    add(code: string, model: Model) { byCode.set(code, model); },
    addPipeline(pipeline: Pipeline, model: Model) { byPipeline.set(pipeline, model); },
    // (the WGSL texts dispatched that were answered, in order; T.465.6: and those that were not, which a check may hold empty)
    seen, missed,
    answer(pipeline: Pipeline, entries: Entry[], x: number, y: number) {
      const model = byPipeline.get(pipeline) ?? byCode.get(pipeline.code);
      if (!model) return void missed.push(pipeline.code);
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
    // T.465.3: a token's attention by llama.cpp's decode form (flashVec): q, the keys, the values, the parts, out, the params
    // (heads, kv heads, scale, nwg, where the sums and largests start, window), the step (1, the token's position), a
    // workgroup for each head and part, no more (T.465.3's review: those past the heads write into the parts' sums). The
    // token's output is the prompt's attention of one token (gpu_answers.py's flash_tile); with one part it goes to out,
    // with more it is kept for the reduce (flashVecReduce: the parts, out, the params; a workgroup a head), which writes
    // it where the parts are that buffer, nwg and the sums' start as the decode had them, and the parts as large as nwg
    // takes (T.465.3's review); a reduce of one part writes 0 (its parts were never written: no sum). ignoring: the window
    // left out (a check's break: a form that reads every position)
    flashVec: (size: number, { ignoring = false } = {}): Model => (entries, x) => {
      const [heads, kvHeads, , nwg, stats, window] = words(entries, 5), [, pos] = words(entries, 6);
      if (x !== heads * nwg) return;
      const [out] = called("flash_tile", bound(entries, 0), bound(entries, 1), bound(entries, 2), heads, kvHeads, float(entries, 5, 2), ignoring ? 0 : window, 1, pos, size);
      if (nwg === 1) bound(entries, 4).set(out.subarray(0, bound(entries, 4).length));
      else reducing.set(bound(entries, 3).buffer, { out, nwg, stats, size });
    },
    flashVecReduce: ((entries, x) => {
      const [heads, , , nwg, stats] = words(entries, 2), parts = bound(entries, 0), kept = reducing.get(parts.buffer);
      if (x !== heads) return;
      if (!kept) {
        bound(entries, 1).fill(0);
        return;
      }
      reducing.delete(parts.buffer);
      if (nwg !== kept.nwg || stats !== kept.stats || stats !== heads * nwg * kept.size || parts.length < heads * nwg * (kept.size + 2) * 4) return;
      bound(entries, 1).set(kept.out.subarray(0, bound(entries, 1).length));
    }) as Model,
    // ---- T.465.4: a generated token's step (gpu_answers.py's embed, fused, fused_rope, sample)
    // EMBED: the table, its scales, the state (its token), x, the shape (the row's width, the piece's first row and rows)
    embed: ((entries) => {
      const [n, first, rows] = words(entries, 4);
      const [row] = called("embed", bound(entries, 0), bound(entries, 1), bound(entries, 2), n, first, rows);
      if (row) bound(entries, 3).set(row.subarray(0, bound(entries, 3).length));
    }) as Model,
    // fusedMatVec ({ norm, output }): the weights, their scales, x, the Params, the norm's weight (norm), then dst, or (rope)
    // q, the keys, the values, the angles and the step; a workgroup for every 4 rows (MUL_MAT_VEC_ROWS)
    fusedMatVec: ({ norm, output }: { norm: boolean; output: string }): Model => (entries, x, y) => {
      const [rows] = words(entries, 3);
      if (x * y < Math.ceil(rows / 4)) return;
      const weight = norm ? bound(entries, 4) : undefined;
      if (output === "rope") {
        const [q, keys, values] = called("fused_rope", norm, bound(entries, 0), bound(entries, 1), bound(entries, 2), bound(entries, 3), weight,
          bound(entries, 6), bound(entries, 7), bound(entries, 8), bound(entries, 9));
        for (const [binding, made] of [[5, q], [6, keys], [7, values]] as const) bound(entries, binding).set(made.subarray(0, bound(entries, binding).length));
        return;
      }
      const [out] = called("fused", norm, output, bound(entries, 0), bound(entries, 1), bound(entries, 2), bound(entries, 3), weight, bound(entries, 5));
      bound(entries, 5).set(out.subarray(0, bound(entries, 5).length));
    },
    // the DP4A forms: RMSNORM (x, the weight, dst, the Norm, the step), NORM_QUANTIZE (x, the weight, the codes, their
    // scales, the Norm, the step; a workgroup a token), and fusedDp4aMatVec ({ output }: the weights, their scales, the
    // codes, the Params, the codes' scales, then dst or (rope) q, the keys, the values, the angles and the step)
    // (T.465.6: the RMSNorm family, each row of the dispatch: RMSNORM into out, HEAD_NORM in place (x, the weight, the
    // Norm, the step), RMSNORM_ADD added to out; a workgroup a row across, a token down)
    rmsnorm: ({ inPlace = false, add = false } = {}): Model => (entries, x, y) => {
      const [norm, step] = inPlace ? [2, 3] : [3, 4];
      if (y < words(entries, step)[0]) return;
      const [out] = called("rms_rows", bound(entries, 0), bound(entries, 1), bound(entries, norm), bound(entries, step), x, inPlace ? undefined : bound(entries, 2), add);
      bound(entries, inPlace ? 0 : 2).set(out.subarray(0, bound(entries, inPlace ? 0 : 2).length));
    },
    // ADD (bias: add() is the answerer's own): y, the bias, the Bias (n, its start, by position), the step; 64 values a
    // workgroup across, a token down
    bias: ((entries, x, y) => {
      const [n] = words(entries, 2);
      if (x * 64 < n || y < words(entries, 3)[0]) return;
      const [out] = called("add", bound(entries, 0), bound(entries, 1), bound(entries, 2), bound(entries, 3));
      bound(entries, 0).set(out.subarray(0, bound(entries, 0).length));
    }) as Model,
    // TOKEN_ROPE: the rows (2), the matrix's Params (3), then q, the keys, the values, the angles and the step (5 to 9)
    tokenRope: ((entries, x) => {
      if (x < 1) return;
      const [q, keys, values] = called("token_rope", bound(entries, 2), bound(entries, 3), bound(entries, 6), bound(entries, 7), bound(entries, 8), bound(entries, 9));
      for (const [binding, made] of [[5, q], [6, keys], [7, values]] as const) bound(entries, binding).set(made.subarray(0, bound(entries, binding).length));
    }) as Model,
    normQuantize: ((entries, x) => {
      if (x < words(entries, 5)[0]) return;
      const [codes, scales] = called("norm_quantize", bound(entries, 0), bound(entries, 1), bound(entries, 4), bound(entries, 5));
      bound(entries, 2).set(codes.subarray(0, bound(entries, 2).length));
      bound(entries, 3).set(scales.subarray(0, bound(entries, 3).length));
    }) as Model,
    dp4a: ({ output }: { output: string }): Model => (entries, x, y) => {
      const [rows] = words(entries, 3);
      if (x * y < Math.ceil(rows / 4)) return;
      if (output === "rope") {
        const [q, keys, values] = called("dp4a_rope", bound(entries, 0), bound(entries, 1), bound(entries, 2), bound(entries, 4), bound(entries, 3),
          bound(entries, 6), bound(entries, 7), bound(entries, 8), bound(entries, 9));
        for (const [binding, made] of [[5, q], [6, keys], [7, values]] as const) bound(entries, binding).set(made.subarray(0, bound(entries, binding).length));
        return;
      }
      const [out] = called("dp4a", output, bound(entries, 0), bound(entries, 1), bound(entries, 2), bound(entries, 4), bound(entries, 3), bound(entries, 5));
      bound(entries, 5).set(out.subarray(0, bound(entries, 5).length));
    },
    // SAMPLE: the logits, two scratch buffers, the state, chosen, the random numbers, the settings; one workgroup
    sample: ((entries, x) => {
      if (x < 1) return;
      const [logits, state, chosen] = called("sample", bound(entries, 0), bound(entries, 3), bound(entries, 4), bound(entries, 5), bound(entries, 6));
      for (const [binding, made] of [[0, logits], [3, state], [4, chosen]] as const) bound(entries, binding).set(made.subarray(0, bound(entries, binding).length));
    }) as Model,
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
