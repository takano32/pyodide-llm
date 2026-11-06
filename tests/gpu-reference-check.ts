// tests/gpu-reference-check.ts (T642.11): src/runtime/gpu/reference.ts, the JavaScript a token's check holds the GPU's
// answer to (tokenforms.ts's checkTokens()), by itself: no device, no worker. Each part against tests/gpu_answers.py's
// NumPy of the shader that computes the same (in Pyodide, through tests/fakes/gpu-answers.ts's called()), on made-up
// numbers: a matrix's rows times a vector, as they are (fusedMatVec's "write") and quantized first (fusedDp4aMatVec's),
// of int8 rows, ternary rows (in the shared memory with float16 scales, and read back with float32 ones) and Q4_K's
// blocks; RMSNorm, LayerNorm and GELU. The reference is float64 and NumPy's answer float32: within 1e-6 of the largest.
// And the parts of no shader: the check's rows of the vocabulary, the largest and how far, heldFloats().
// T642.12: and the answer a prompt's attention is checked against (gpu/forms.ts's checkAttention()), by itself too:
// shaders/attention.ts's promptAttentionOff() of promptAttentionData()'s numbers, against gpu_answers.py's flash_tile
// (with a window and without), and that one head of another's, a token of another's and a NaN are past the check's line.
//
//   node tests/gpu-reference-check.ts        (Node and Pyodide, about 10 seconds)
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.ts";
import { gpuAnswers } from "./fakes/gpu-answers.ts";
const { product, rms, norm, gelu, largest, off, checkCandidates, checkRows, heldFloats, rowsInMemory, rowsReadBack } = await import(runtimeUrl("gpu/reference.ts"));
const { quantizedLikeCpu } = await import(runtimeUrl("shaders/prompt.ts"));
const { SUPERBLOCKS } = await import(runtimeUrl("shaders/superblocks.ts"));
const { halvesOf } = await import(runtimeUrl("shaders/halves.ts"));
const { promptAttentionData, promptAttentionOff } = await import(runtimeUrl("shaders/attention.ts"));

const { called } = await gpuAnswers();
let passed = 0;
const ok = (name) => { passed++; console.log(`ok: ${name}`); };

// made-up numbers, the same every run
let seed = 642011;
const random = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const floats = (n, scale = 1) => Float32Array.from({ length: n }, () => (random() * 2 - 1) * scale);
const bytesOf = (typed) => new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
const u32 = (...words) => bytesOf(Uint32Array.from(words));
const asFloats = (bytes) => new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
// the fused matrices' Params (gpu_answers.py's row_params): rows, words of a row, groups of a row, up's rows, eps, the norm's start
const params = (rows, n) => {
  const words = new Uint32Array(12);
  words.set([rows, n / 4, n / 32, 0]);
  return bytesOf(words);
};
// within 1e-6 of the largest of want (NaN past it)
const near = (got, want, name) => {
  const far = off(got, want) / largest(want);
  assert.ok(far <= 1e-6, `${name}: ${far} of the largest from NumPy's`);
};

const rows = 5, n = 512, all = [...Array(rows).keys()];
const x = floats(n);

// ---- int8 rows, a float32 scale a group of 32, in the shared memory: as they are, and quantized first
{
  const values = Int8Array.from({ length: rows * n }, () => Math.floor(random() * 255) - 127), scales = floats((rows * n) / 32, 0.01).map(Math.abs);
  const memory = new ArrayBuffer(values.byteLength + scales.byteLength);
  new Int8Array(memory).set(values);
  new Float32Array(memory, values.byteLength).set(scales);
  const rowOf = rowsInMemory(() => memory)({ n, packing: "int8" }, [0, values.byteLength]);
  const dst = new Uint8Array(rows * 4);
  const [plain] = called("fused", false, "write", bytesOf(values), bytesOf(scales), bytesOf(x), params(rows, n), undefined, dst);
  near(product(n, rowOf, x, all, false), asFloats(plain), "int8 rows times x");
  const [xq, xs] = called("quantize", bytesOf(x), n, n, 1);
  const q = quantizedLikeCpu(x);
  assert.deepEqual([...q.xq], [...new Int8Array(xq.buffer, xq.byteOffset, n)], "x quantized as quantize_x");
  const [packed] = called("dp4a", "write", bytesOf(values), bytesOf(scales), xq, xs, params(rows, n), dst);
  near(product(n, rowOf, x, all, true), asFloats(packed), "int8 rows times x quantized");
  // a row read back from the GPU is read as one in the memory
  const back = rowsReadBack()({ n, packing: "int8" }, [new Uint8Array(values.buffer), scales]);
  assert.deepEqual(product(n, back, x, all, true), product(n, rowOf, x, all, true));
  ok("int8 rows: as they are and quantized, in the memory and read back");
}

// ---- ternary rows (the GPU's codes, four a byte), quantized first: their scales float16 in the memory, float32 read back
{
  const codes = Uint8Array.from({ length: (rows * n) / 4 }, () => {
    let byte = 0;
    for (let k = 0; k < 4; k++) byte |= Math.floor(random() * 3) << (2 * k);
    return byte;
  });
  const wide = floats((rows * n) / 128, 0.05).map(Math.abs), halves = halvesOf(wide, new Uint16Array(wide.length)), narrowed = Float32Array.from(wide, (v) => Math.fround(Math.f16round?.(v) ?? v));
  const memory = new ArrayBuffer(codes.byteLength + halves.byteLength + 2);
  new Uint8Array(memory).set(codes);
  new Uint16Array(memory, codes.byteLength).set(halves);
  const inMemory = rowsInMemory(() => memory)({ n, packing: "ternary" }, [0, codes.byteLength]);
  const widened = Float32Array.from({ length: wide.length }, (_, i) => inMemory(Math.floor(i / (n / 128)))[1][i % (n / 128)]);
  const back = rowsReadBack()({ n, packing: "ternary" }, [codes, widened]);
  const [xq, xs] = called("quantize", bytesOf(x), n, n, 1);
  const [want] = called("dp4a", "write", codes, bytesOf(widened), xq, xs, params(rows, n), new Uint8Array(rows * 4), "ternary");
  near(product(n, inMemory, x, all, true), asFloats(want), "ternary rows in the memory times x quantized");
  near(product(n, back, x, all, true), asFloats(want), "ternary rows read back times x quantized");
  if (Math.f16round) assert.deepEqual([...widened], [...narrowed], "the float16 scales widened as they are");
  ok("ternary rows: in the memory (float16 scales) and read back (float32), quantized");
}

// ---- Q4_K's blocks (their weights, whose one scale is 1), quantized first
{
  const { bytes, scale } = SUPERBLOCKS.Q4_K;
  const blocks = Uint8Array.from({ length: ((rows * n) / 256) * bytes }, () => Math.floor(random() * 256));
  for (let at = 0; at < blocks.length; at += bytes) scale(blocks, at, Math.floor(random() * 1024) | 0x0c00 | (random() < 0.5 ? 0x8000 : 0));
  const rowOf = rowsReadBack()({ n, packing: "Q4_K" }, [blocks, new Float32Array(0)]);
  const [xq, xs] = called("quantize", bytesOf(x), n, n, 1);
  const [want] = called("dp4a", "write", blocks, new Uint8Array(4), xq, xs, params(rows, n), new Uint8Array(rows * 4), "Q4_K");
  near(product(n, rowOf, x, all, true), asFloats(want), "Q4_K rows times x quantized");
  ok("Q4_K's blocks, quantized");
}

// ---- the norms and GELU
{
  const eps = 1e-5, weights = floats(n, 2), bias = floats(n);
  const normParams = bytesOf(new Uint32Array([n, 0, new Uint32Array(Float32Array.of(eps).buffer)[0], 0]));
  const [rmsWant] = called("rmsnorm", bytesOf(x), bytesOf(weights), normParams, u32(1));
  near(rms(Float64Array.from(x), weights, eps), asFloats(rmsWant), "RMSNorm");
  near(norm(Float64Array.from(x), weights, bias, eps, false), asFloats(rmsWant), "the stream's norm, RMSNorm");
  const [layerWant] = called("layer_norm", bytesOf(x), bytesOf(weights), bytesOf(bias), normParams, u32(1), 1, new Uint8Array(n * 4));
  near(norm(Float64Array.from(x), weights, bias, eps, true), asFloats(layerWant), "the stream's norm, LayerNorm");
  const gate = floats(n, 6);
  const [geluWant] = called("gelu", bytesOf(gate), u32(n), u32(1));
  near(Float64Array.from(gate, gelu), asFloats(geluWant), "GELU");
  ok("RMSNorm, LayerNorm and GELU");
}

// ---- the parts of no shader
{
  assert.deepEqual(checkRows(1000), [0, 256, 512, 768, 999], "every 256th row of the classifier, and the last");
  assert.deepEqual(checkRows(512), [0, 256, 511]);
  const candidates = checkCandidates(1000);
  assert.equal(candidates.length, 64);
  assert.ok(candidates.every((t) => t >= 500 && t < 1000), "the tokens of the second half of the vocabulary");
  assert.equal(candidates[0], 999);
  assert.equal(largest([1, -3, 2]), 3);
  assert.equal(off([1, 2, 3], Float64Array.of(1, 2.5, 2)), 1);
  assert.ok(!(off([NaN], Float64Array.of(1)) <= 1), "a NaN is past every line");
  // heldFloats: the device's own float16 where it is a neighbour of the value, the nearest where it is not
  if (Math.f16round) {
    const values = Float64Array.of(1.00031, -2.00049, 0.5);
    const below = Float64Array.of(1, -2, 0.5), far = Float64Array.of(1.5, -3, 0.25);
    assert.deepEqual([...heldFloats(values, below)], [...below]);
    assert.deepEqual([...heldFloats(values, far)], [...values].map(Math.f16round));
  }
  ok("the check's rows, the largest and how far, heldFloats");
}

// ---- T642.12: a prompt's attention (the check's sizes: 9 tokens at 61, 4 heads of q on 2, a window of 13 or none)
{
  for (const window of [0, 13]) {
    const sizes = { heads: 4, kvHeads: 2, size: 64, tokens: 9, pos: 61, window }, { heads, kvHeads, size, tokens, pos } = sizes;
    const data = promptAttentionData(sizes), qDim = heads * size;
    assert.equal(data.q.length, tokens * qDim);
    assert.equal(data.keys.length, (pos + tokens) * kvHeads * size);
    assert.equal(data.values.length, data.keys.length);
    const [out] = called("flash_tile", bytesOf(data.q), bytesOf(data.keys), bytesOf(data.values), heads, kvHeads, Math.fround(1 / Math.sqrt(size)), window, tokens, pos, size);
    const got = asFloats(out);
    const far = promptAttentionOff(got, data, sizes);
    assert.ok(far <= 1e-6, `the prompt's attention${window ? ", a window" : ""}: ${far} of the largest from NumPy's flash_tile`);
    // (the check's line is 4e-3: a head that reads another's, a token another's, a NaN, are past it)
    const head = got.slice(), at = 5 * qDim;
    head.copyWithin(at, at + size, at + 2 * size);
    assert.ok(promptAttentionOff(head, data, sizes) > 4e-3, "a head of another's");
    const token = got.slice();
    token.copyWithin(3 * qDim, 4 * qDim, 5 * qDim);
    assert.ok(promptAttentionOff(token, data, sizes) > 4e-3, "a token of another's");
    const nan = got.slice();
    nan[tokens * qDim - 1] = NaN;
    assert.ok(!(promptAttentionOff(nan, data, sizes) <= 4e-3), "a NaN");
  }
  ok("a prompt's attention: against flash_tile, with a window and without");
}

console.log(`${passed} passed`);
