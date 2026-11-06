// The first layer and the head of a token in JavaScript, in float64: the reference a token's check holds the GPU's
// answer to (tokenforms.ts's checkTokens(), T152). Nothing of a device: the shared memory's bytes, the plan, and what
// the check read back from the GPU are handed in, and what it computes is a number of them alone (T642.11: so that
// tests/gpu-reference-check.ts holds it to tests/gpu_answers.py's NumPy by itself).
// (T642.11: a module of the model's GPU worker, gpu.ts, which a bundler links it into)

import { PACKINGS, packingAt } from "./packings.ts";
import { floatOfHalf } from "../shaders/halves.ts";
import { GROUP } from "../shaders/common.ts";
import { quantizedLikeCpu } from "../shaders/prompt.ts";
import { CODEBOOK_BYTES } from "../shaders/superblocks.ts";
import { rotateLikeCpu, unrotateLikeCpu } from "../shaders/rotate.ts";
import type { GpuPacking, Started, Vectors } from "../contracts/gpu/protocol.ts";
import type { TensorName } from "../contracts/plan.ts";

// what says how a row is packed (a plan's matrix or table), a row as [its int8 values, its scales], a row's bytes and
// scales as the check read them back from the GPU, and a list of numbers as its sums read one (an array or a typed array)
type Spec = { n: number; packing: GpuPacking };
type Row = [ArrayLike<number>, Float32Array];
type Held = [Uint8Array, Float32Array];
type Numbers = ArrayLike<number> & { reduce(step: (sum: number, value: number) => number, from: number): number; map(each: (value: number, i: number) => number): Numbers };
// T156, T210: what a model on the GPU alone read back for the check (tokenforms.ts's firstLayer(), tableRows()): the first
// layer's matrices by name, and the rows of the tables it holds by row
type Back = { firstLayer?: { [name: string]: Held }; tableRows?: { embedding: Map<number, Held>; classifier: Map<number, Held> } };
// (Math.f16round is newer than the lib this is checked against, and than some browsers: heldFloats() asks)
type Rounding = Math & { f16round?: (value: number) => number };

// T225's review: WGSL leaves it to the device which of its two float16 neighbours a float32 becomes (§15.7.6 Floating
// Point Conversion: "WGSL does not specify whether the higher or lower representable value is chosen, and different
// instances of such a conversion may choose differently"; pack2x16float is such a conversion), and Direct3D, where
// Chrome on Windows runs WebGPU, converts toward zero (D3D11.3 functional specification 3.2.2: "Round-to-zero must be
// used during conversion to another float format"; Dawn's HLSL writer makes pack2x16float of f32tof16). The position's
// keys and values the GPU writes are then 1 float16 spacing from the nearest's on about half of their numbers, and the
// stream computed from the nearest's was up to 3.2 times the line off the GPU's (CI, Dawn on lavapipe with every
// conversion cut toward zero: a llama.cpp form on llm-jp-3 150M 6.5e-3 at the line 2e-3; with the GPU's own keys and
// values 2.3e-6, as src/runtime/benchmark/gpu/halves.ts's heldHalves takes them: T225). So the reference takes the device's own
// float16 (got, as floats) of a value where it is within a float16 spacing of the value and the float32 sums' noise
// (HALF_SLACK of the largest: about 10 times a sum of 2112 products's), and the nearest where it is not: a wrong key
// or value is as far from the line as before, and the keys' and values' lines below are unchanged.
const HALF_SLACK = 1e-5;
function heldFloats(x: Float64Array, got: ArrayLike<number>) {
  const round16 = (Math as Rounding).f16round ?? ((value: number) => value), slack = HALF_SLACK * x.reduce((top, value) => Math.max(top, Math.abs(value)), 0);
  return x.map((value, i) => (Math.abs(got[i] - value) <= 2 ** (Math.max(Math.floor(Math.log2(Math.abs(value))), -14) - 10) + slack ? got[i] : round16(value)));
}

// the tokens whose rows of the embedding the first layer's check may take (of the second half of the vocabulary: see
// largestToken()), and the rows of the classifier it holds to JavaScript's (every 256th, and the last)
const checkCandidates = (vocab: number) => [...Array(64).keys()].map((i) => vocab - 1 - Math.floor((i * vocab) / 128));
const checkRows = (vocab: number) => [...new Set([...[...Array(Math.ceil(vocab / 256)).keys()].map((i) => i * 256), vocab - 1])];

// the largest of a list (by size), and how far one list is from another at the most
const largest = (xs: Numbers) => xs.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
const off = (got: ArrayLike<number>, want: Numbers) => largest(want.map((v, i) => got[i] - v));
// GELU, the tanh approximation with its argument clamped, as the shaders have it
const gelu = (value: number) => 0.5 * value * (1 + Math.tanh(Math.min(9.010913, Math.max(-9.010913, 0.7978845608028654 * (value + 0.044715 * value ** 3)))));
// RMSNorm with an epsilon; and (T226) the stream's norm, RMSNorm or (T154) LayerNorm with its bias, as the CPU's kernels
// have them
const rms = (x: Float64Array, weights: ArrayLike<number>, eps: number) => {
  const s = 1 / Math.sqrt(x.reduce((sum, v) => sum + v * v, 0) / x.length + eps);
  return x.map((v, i) => weights[i] * (s * v));
};
const norm = (x: Float64Array, weights: ArrayLike<number>, bias: ArrayLike<number> | 0 | undefined, eps: number, layerNorm: boolean) => {
  if (!layerNorm) return rms(x, weights, eps);
  const mean = x.reduce((sum, v) => sum + v, 0) / x.length;
  const s = 1 / Math.sqrt(x.reduce((sum, v) => sum + (v - mean) ** 2, 0) / x.length + eps);
  return x.map((v, i) => weights[i] * (s * (v - mean)) + (bias as ArrayLike<number>)[i]);
};
// a matrix's rows (n weights each, rowOf(r): [values, scales]) times x, the rows given (packed: x quantized first, as
// the CPU quantizes it). The vector's groups are of 32; a row's scales cover as many of them each as they are fewer
// (one; T232, ternary: four)
const product = (n: number, rowOf: (r: number) => Row, x: ArrayLike<number>, rows: number[], packed: boolean) => {
  const q = packed ? quantizedLikeCpu(Float32Array.from(x)) : null, perRow = n / GROUP;
  return Float64Array.from(rows, (r) => {
    const [w, s] = rowOf(r), each = perRow / s.length;
    let sum = 0;
    for (let b = 0; b < perRow; b++) {
      let part = 0;
      for (let i = b * GROUP; i < (b + 1) * GROUP; i++) part += w[i] * (q ? q.xq[i] : x[i]);
      sum += part * s[Math.floor(b / each)] * (q ? q.xs[b] : 1);
    }
    return sum;
  });
};

// the bytes of a row as its int8 values: int6 widened, ternary codes less one, int8 as they are (T547.2.4: Q4_K's
// blocks as their weights, whose one scale is 1: ONE; T547.3.4: an IQ type's of the model's codebooks)
const ONE = Float32Array.of(1);
const stored = (spec: Spec) => PACKINGS[spec.packing].stored(spec.n), scalesOf = (spec: Spec) => PACKINGS[spec.packing].scales(spec.n);
const valuesOf = (spec: Spec, bytes: Uint8Array, codebooks?: Uint8Array) => PACKINGS[spec.packing].values(bytes, codebooks);
// row r of a matrix or a table in the shared memory (its buffer as it is now, memory(); its values and scales at
// [valuesAt, scalesAt])
const rowsInMemory = (memory: () => ArrayBufferLike, codebooks?: Uint8Array) => (spec: Spec, [valuesAt, scalesAt]: [number, number]) => (r: number): Row => [
  valuesOf(spec, new Uint8Array(memory(), valuesAt + r * stored(spec), stored(spec)), codebooks),
  // (T533: ternary's scales are float16s there, which went up widened)
  !scalesOf(spec) ? ONE : PACKINGS[spec.packing].scale === 2 ? Float32Array.from(new Uint16Array(memory(), scalesAt + r * scalesOf(spec) * 2, scalesOf(spec)), floatOfHalf)
    : new Float32Array(memory(), scalesAt + r * scalesOf(spec) * 4, scalesOf(spec))];
// (T634.3: what is read back is the packing on the GPU's: binary's ternary codes)
const onGpu = (spec: Spec): Spec => ({ ...spec, packing: PACKINGS[spec.packing].on });
// row r of a matrix read back from the GPU (T156, firstLayer(): [values, scales] of every row)
const rowsReadBack = (codebooks?: Uint8Array) => (given: Spec, [values, scales]: Held) => (r: number): Row => {
  const spec = onGpu(given);
  return [valuesOf(spec, values.subarray(r * stored(spec), (r + 1) * stored(spec)), codebooks), scalesOf(spec) ? scales.subarray(r * scalesOf(spec), (r + 1) * scalesOf(spec)) : ONE];
};

// The model's reference of the check of a form: of the shared memory (memory(): its buffer as it is now) as forward.ts
// holds the weights, the plan, whether the form quantizes a matrix's input (packed: DP4A) and what was read back from
// the GPU (back). The first layer: the embedding of a token at a position (embedded), a matrix of the first layer
// times a vector (matmul), the first layer's vector of a name (vectorOf; plus(): added), RMSNorm with the model's
// epsilon (rms) and the stream's norm (norm), a norm behind (behind), the FFN after the layer's output o (ffn: the
// stream after the layer); the head: the logits of rows of a stream (logits). And the token whose row the check takes
// (largestToken)
function referenceOf({ plan, memory, packed, back = {} }: { plan: Started; memory: () => ArrayBufferLike; packed: boolean; back?: Back }) {
  const planned = plan.tokens!, dim = plan.dim;
  const floats = (address: number, n: number) => Float64Array.from(new Float32Array(memory(), address, n));
  const codebooks = plan.codebooks ? new Uint8Array(memory(), plan.codebooks, CODEBOOK_BYTES).slice() : undefined;
  const inMemory = rowsInMemory(memory, codebooks), readBackRows = rowsReadBack(codebooks);
  const tableSpec = (name: "embedding" | "classifier") => (name === "embedding" ? planned.embedding ?? planned.classifier : planned.classifier);
  // (T210: of a table's rows read back, tableRows(); T547.3.7: a table of blocks' one scale of 1, as inMemory's)
  const tableRow = (name: "embedding" | "classifier") => (back.tableRows ? (r: number): Row => {
    const [bytes, scales] = back.tableRows![name].get(r)!;
    return [valuesOf(onGpu(tableSpec(name)), bytes, codebooks), scalesOf(tableSpec(name)) ? scales : ONE];
  } : inMemory(tableSpec(name), tableSpec(name).at));
  // T271.1: a rotated basis's signs of a width (float32, Python's, each times 1 / sqrt(block)), and R x, as the CPU's
  // rotate (the matrices read it)
  const signsOf = (n: number) => new Float32Array(memory(), plan.rotated!.signs.find(({ width }) => width === n)!.at, n);
  const turned = (x: ArrayLike<number>, n: number) => (plan.rotated ? rotateLikeCpu(x, signsOf(n), plan.rotated.block) : x);
  const times = (n: number, rowOf: (r: number) => Row, x: ArrayLike<number>, rows: number[], quantized = packed) => product(n, rowOf, turned(x, n), rows, quantized);
  const all = (n: number) => [...Array(n).keys()];
  const matmul = (name: TensorName, x: ArrayLike<number>) => {
    const matrix = plan.matrices[name]!, first = { ...matrix, packing: packingAt(matrix, 0) };
    return times(matrix.n, back.firstLayer ? readBackRows(first, back.firstLayer[name]) : inMemory(first, matrix.layers[0]), x, all(matrix.rows));
  };
  const vectorOf = (name: keyof Vectors) => plan.vectors[name] && floats(plan.vectors[name].at, plan.vectors[name].size);
  const plus = (x: Float64Array, name: keyof Vectors) => {
    vectorOf(name)?.forEach((b, i) => { x[i] += b; });
    return x;
  };
  const rmsOf = (x: Float64Array, weights: Float64Array) => rms(x, weights, plan.eps);
  const normOf = (x: Float64Array, weights: Float64Array, bias: Float64Array | 0 | undefined) => norm(x, weights, bias, plan.eps, !!plan.layerNorm);
  // (T261.4.2: a Gemma 3's norm behind the attention and behind the FFN, of what joins the stream)
  const behind = (x: Float64Array, name: keyof Vectors) => (vectorOf(name) ? rmsOf(x, vectorOf(name)!) : x);
  const embeddingRow = tableRow("embedding");
  return {
    matmul, vectorOf, plus, behind, rms: rmsOf, norm: normOf,
    // a token of the second half of the vocabulary whose row is large: the rows of tokens no text has (llm-jp-3's
    // last) are nearly 0, and their keys and values then float16's subnormals (a check of them was 15% off, CI).
    // (T547.3.6's review: of a table of llama.cpp's blocks, whose one scale is 1, by its weights themselves)
    largestToken(vocab: number) {
      const sizeOf = (t: number) => {
        const [values, scales] = embeddingRow(t), of = scales.length === 1 ? values : scales;
        let sum = 0;
        for (let i = 0; i < of.length; i++) sum += Math.abs(of[i]);
        return sum;
      };
      return checkCandidates(vocab).reduce((best, t) => (sizeOf(t) > sizeOf(best) ? t : best));
    },
    // the stream a token comes in as at pos
    embedded(token: number, pos: number) {
      const [eRow, eScales] = embeddingRow(token);
      // (T226: with GPT-2's learned position of pos, as the CPU's embed() adds it)
      const x0 = Float64Array.from(eRow, (v, i) => v * eScales[Math.floor((i * eScales.length) / eRow.length)]);
      if (planned.positions) floats(planned.positions + pos * dim * 4, dim).forEach((p, i) => { x0[i] += p; });
      // (T271.1: the embedding of a rotated basis holds R^-1 of its rows, turned back as the CPU's embed() does)
      if (plan.rotated) x0.set(unrotateLikeCpu(x0, signsOf(dim), plan.rotated.block));
      return x0;
    },
    // the stream after the first layer, of its input x0 and its attention's (or kind's) output o
    ffn(x0: Float64Array, o: Float64Array) {
      const h1 = x0.map((value, i) => value + o[i]);
      // (T154: GPT-NeoX's parallel residual, the FFN's norm of the layer's input; no gate, w1's bias and GELU as
      // the shaders have it)
      const xn2 = normOf(plan.parallel ? x0 : h1, vectorOf("ffn")!, vectorOf("ffnBias"));
      const gate = plus(matmul("w1", xn2), "b1"), up = plan.matrices.w3 && matmul("w3", xn2);
      // (T261.4.2: GELU in the gate, a Gemma 3's GEGLU)
      const activated = gate.map(!up ? gelu : plan.activation === "gelu" ? (value, i) => gelu(value) * up[i] : (value, i) => (value / (1 + Math.exp(-value))) * up[i]);
      const down = behind(plus(matmul("w2", activated), "b2"), "ffnPost");
      return h1.map((value, i) => value + down[i]);
    },
    // the logits of rows of a stream: the final norm, the classifier (quantized: its input quantized first). T232: a
    // ternary classifier's outlier channels as the CPU has them (src/runtime/forward/engine.ts's picked) where take:
    // taken out of the normed stream before it is quantized, and their columns of the table multiplied apart, in floats
    logits(stream: ArrayLike<number>, rows: number[], quantized: boolean, take: boolean) {
      const normed = normOf(Float64Array.from(stream), floats(planned.final, dim), planned.finalBias && floats(planned.finalBias, dim));
      const taken = take ? planned.channels.map((c): [number, number] => {
        const value = normed[c];
        normed[c] = 0;
        return [c, value];
      }) : [];
      const want = times(dim, tableRow("classifier"), normed, rows, quantized);
      rows.forEach((r, j) => {
        const [w, s] = tableRow("classifier")(r);
        for (const [c, value] of taken) want[j] += value * w[c] * s[Math.floor((c * s.length) / w.length)];
      });
      return want;
    },
  };
}

export { HALF_SLACK, heldFloats, checkCandidates, checkRows, largest, off, gelu, rms, norm, product, rowsInMemory, rowsReadBack, referenceOf };
export type { Spec, Row, Held, Back, Numbers };
