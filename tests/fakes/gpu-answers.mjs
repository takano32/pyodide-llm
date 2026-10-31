// tests/fakes/gpu-answers.mjs (T.465.1)
// What the engine's shaders compute, in JavaScript, for tests/fakes/webgpu.mjs's answer: a dispatch of a pipeline whose
// WGSL is one of those named here gets its output written as that shader would write it, so that the model's GPU worker
// can choose its forms (gpu/forms.ts: it checks each candidate's output against JavaScript before it takes it) on a
// device that runs no shader. For the checks that run those parts: gpu-forms-check.mjs.
//
//   const answers = gpuAnswers();
//   answers.add(code, model)        a pipeline of this WGSL is answered by model (one of the makers below)
//   answers.addPipeline(p, model)   a pipeline object itself (one compiled already: m.block.quantize)
//   standIn({ answer: answers.answer })
//
// The same as the shaders: what each writes, where its bindings say, the numbers of the arithmetic they mean (float64
// sums stored as float32: inside every check's line). Simpler: a dispatch writes its whole output whatever its
// workgroups (a shader that dispatched too few tiles is not seen here: CI's gpu-check is where the tiles are), and a
// form's tiles, f16 or subgroups make no difference. A pipeline of no WGSL named here is not answered (its output stays
// as it was written), and so is a broken model: a check's own breaks give one (wrong()).
import { quantizedLikeCpu, ternaryValues, BLOCK_WEIGHTS } from "../../src/runtime/shaders/prompt.ts";
import { GROUP } from "../../src/runtime/shaders/common.ts";

// a binding's bytes: its buffer's, from its offset (a part of one, T.155's pieces of a matrix)
const bound = (entries, binding) => {
  const { resource } = entries.find((entry) => entry.binding === binding);
  const buffer = resource.buffer ?? resource, offset = resource.offset ?? 0;
  return { data: buffer.data, offset, size: resource.size ?? buffer.size - offset };
};
const floats = (entries, binding) => { const { data, offset, size } = bound(entries, binding); return new Float32Array(data.buffer, data.byteOffset + offset, size / 4); };
const words = (entries, binding) => { const { data, offset, size } = bound(entries, binding); return new Uint32Array(data.buffer, data.byteOffset + offset, size / 4); };
const bytes = (entries, binding) => { const { data, offset, size } = bound(entries, binding); return new Uint8Array(data.buffer, data.byteOffset + offset, size); };
function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, fraction = h & 1023;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  return exponent === 31 ? (fraction ? NaN : sign * Infinity) : sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// ---- the prompt's attention (shaders/attention.ts's flashTile): q (float32, [tokens][heads][size]), the keys and values
// (float16, [positions][kv heads][size]), out, the params (heads, kv heads, scale, window) and the block (tokens, pos).
// Token t is at pos + t and attends the positions up to its own (the last window of them where window is not 0)
export const flashTile = (size) => (entries) => {
  const q = floats(entries, 0), keys = bytes(entries, 1), values = bytes(entries, 2), out = floats(entries, 3);
  const [heads, kvHeads] = words(entries, 4), scale = floats(entries, 4)[2], window = words(entries, 4)[3];
  const [tokens, pos] = words(entries, 5);
  const halves = (data) => new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2);
  const k = halves(keys), v = halves(values), kvDim = kvHeads * size, qDim = heads * size;
  for (let t = 0; t < tokens; t++) {
    for (let h = 0; h < heads; h++) {
      const kv = Math.floor(h / (heads / kvHeads)) * size, row = t * qDim + h * size, seen = pos + t + 1, from = window ? Math.max(0, seen - window) : 0;
      const scores = new Float64Array(seen).fill(-Infinity);
      for (let p = from; p < seen; p++) {
        scores[p] = 0;
        for (let d = 0; d < size; d++) scores[p] += q[row + d] * scale * halfToFloat(k[p * kvDim + kv + d]);
      }
      const most = Math.max(...scores);
      let sum = 0;
      for (let p = 0; p < seen; p++) sum += (scores[p] = Math.exp(scores[p] - most));
      for (let d = 0; d < size; d++) {
        let value = 0;
        for (let p = 0; p < seen; p++) value += scores[p] * halfToFloat(v[p * kvDim + kv + d]);
        out[row + d] = value / sum;
      }
    }
  }
};

// ---- QUANTIZE (shaders/prompt.ts's quantize_x as the CPU's kernel does it: quantizedLikeCpu): x (float32), xq (int8)
// and xs (a float32 a group of 32), the params (n, x's stride), the block (tokens)
export const quantize = (entries) => {
  const x = floats(entries, 0), xq = new Int8Array(bytes(entries, 1).buffer), xqAt = bytes(entries, 1).byteOffset, xs = floats(entries, 2);
  const [n, stride] = words(entries, 3), [tokens] = words(entries, 4);
  for (let t = 0; t < tokens; t++) {
    const { xq: codes, xs: scales } = quantizedLikeCpu(x.subarray(t * stride, t * stride + n));
    xq.set(codes, xqAt + t * stride);
    xs.set(scales, t * (stride / GROUP));
  }
};

// ---- a tiled form of a prompt's matrices (shaders/prompt.ts's promptForms): the weights, their scales (int8: a float32 a
// group of 32 of a row; ternary: one a group of 128) or, of an IQ type of llama.cpp's blocks, the codebooks (blocks:
// the scales are in the blocks, BLOCK_WEIGHTS reads them), x (packed: its codes, with their scales at binding 6), y,
// the shape (rows, n / 4, groups of a row, -, x's stride, y's stride, add) and the block (tokens). y = W x, or y + W x
export const tiled = ({ packed, packing }) => (entries) => {
  const [rows, quarter, , , xStride, yStride, add] = words(entries, 4), n = quarter * 4, [tokens] = words(entries, 5);
  const stored = bytes(entries, 0), y = floats(entries, 3);
  let w, scaleOf;
  if (packing === "int8") {
    w = new Int8Array(stored.buffer, stored.byteOffset, rows * n);
    const s = floats(entries, 1);
    scaleOf = (r, i) => s[r * (n / GROUP) + Math.floor(i / GROUP)];
  } else if (packing === "ternary") {
    w = ternaryValues(stored.slice(0, (rows * n) / 4));
    const s = floats(entries, 1);
    scaleOf = (r, i) => s[r * (n / (4 * GROUP)) + Math.floor(i / (4 * GROUP))];
  } else {
    const grids = entries.some((entry) => entry.binding === 1) ? bytes(entries, 1).slice() : undefined;
    w = BLOCK_WEIGHTS[packing].weights(stored.slice(0, (rows * n) / 256 * blockBytes(packing)), grids);
    scaleOf = () => 1;
  }
  const x = packed ? null : floats(entries, 2);
  const xq = packed ? new Int8Array(bytes(entries, 2).buffer, bytes(entries, 2).byteOffset) : null, xs = packed ? floats(entries, 6) : null;
  for (let t = 0; t < tokens; t++) {
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const weight = w[r * n + i] * scaleOf(r, i);
        sum += packed ? weight * xq[t * xStride + i] * xs[t * (xStride / GROUP) + Math.floor(i / GROUP)] : weight * x[t * xStride + i];
      }
      y[t * yStride + r] = (add ? y[t * yStride + r] : 0) + sum;
    }
  }
};
// (a block's bytes of a type: shaders/prompt.ts's BLOCK_READS has them, which a check imports beside; set by gpuAnswers)
let blockBytes = () => { throw new Error("gpu-answers: the bytes of a block of llama.cpp's are not known here"); };

// the answerer: the WGSL texts and pipelines named, and the model of each
export function gpuAnswers({ blockReads } = {}) {
  if (blockReads) blockBytes = (packing) => blockReads[packing].bytes;
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
  };
}
