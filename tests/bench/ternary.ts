// tests/bench.ts's part (T270): the tables of ternary weights, the GPU's (src/bench/ternary.ts) and the CPU's kernels
// (src/bench/cpu.ts's kernelsTable()), on made-up answers of their steps.
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { ternaryTable, ternaryForms, kernelsTable, gpuSummary, cellsOf } from "../../src/bench.ts";

// ---- the GPU: a token's matrix × vector on two shapes (the codes made signed, and less Σa, against int8), a tile, a layer
const matrix = (shape, form, ternary, ms, extra = {}) => ({ what: "matrix", form, ternary, shape, tokens: 1, weights: 8192 * 2048,
  bytes: ternary ? 8192 * 2048 / 4 + 8192 * 2048 / 128 * 4 : 8192 * 2048 * 1.125, ms, ...extra });
const SIGNED = "ternary, the codes made signed (engine)", SUMS = "ternary, the codes as they are less Σa";
const found = { pos: 127, rows: [
  matrix("Llama 3.2 1B w1", "int8 (engine)", false, 0.4), matrix("Llama 3.2 1B w1", SIGNED, true, 0.2), matrix("Llama 3.2 1B w1", SUMS, true, 0.16),
  matrix("Ternary Bonsai 8B w1", "int8 (engine)", false, 1.2), matrix("Ternary Bonsai 8B w1", SIGNED, true, 0.5),
  matrix("Ternary Bonsai 8B w1", SUMS, true, undefined, { wrong: "3.10e-1 of the products off JavaScript's" }),
  { what: "tile", form: "ORT DP4A 64×64, int8", ternary: false, shape: "Llama 3.2 1B w1", tokens: 64, weights: 8192 * 2048, bytes: 1, ms: 2 },
  { what: "tile", form: "ORT DP4A 64×64, subgroups, int8", ternary: false, shape: "Llama 3.2 1B w1", tokens: 64, weights: 8192 * 2048, bytes: 1, ms: 1 },
  { what: "tile", form: "ORT DP4A 64×64, ternary", ternary: true, shape: "Llama 3.2 1B w1", tokens: 64, weights: 8192 * 2048, bytes: 1, ms: 4 },
  { what: "tile", form: "ORT DP4A 64×64, subgroups, ternary", ternary: true, shape: "Llama 3.2 1B w1", tokens: 64, weights: 8192 * 2048, bytes: 1, ms: 4 },
  { what: "layer", form: "DP4A, fused (T175), int8 (engine)", ternary: false, shape: "Llama 3.2 1B, a layer", tokens: 1, weights: 60817408, bytes: 68e6, ms: 3 },
  { what: "layer", form: "DP4A, fused (T175), ternary (engine)", ternary: true, shape: "Llama 3.2 1B, a layer", tokens: 1, weights: 60817408, bytes: 17e6, ms: 1.5, unsteady: true }] };
const ceilings = { fallback: false, global: { GBps: 40 } };
const lines = ternaryTable({ name: "ternary weights", result: found }, ceilings);
const rowOf = (shape, form) => cellsOf(lines.find((line) => line.includes(`| ${shape} | ${form} |`)));
// against the int8 row of its own table and shape: 0.4 / 0.2, 0.4 / 0.16, 1.2 / 0.5; a tile's against its own tile
assert.equal(rowOf("Llama 3.2 1B w1", SIGNED).at(-1), "2.0×");
assert.equal(rowOf("Llama 3.2 1B w1", SUMS).at(-1), "2.5×");
assert.equal(rowOf("Ternary Bonsai 8B w1", SIGNED).at(-1), "2.4×");
assert.equal(rowOf("Llama 3.2 1B w1", "ORT DP4A 64×64, ternary").at(-1), "0.50×");
assert.equal(rowOf("Llama 3.2 1B w1", "ORT DP4A 64×64, subgroups, ternary").at(-1), "0.25×", "a tile with subgroups against the int8 tile with subgroups");
assert.equal(rowOf("Llama 3.2 1B w1", "int8 (engine)").at(-1), "", "an int8 row is not held against itself");
// G weights/s (a weight and token a multiply-add): 16.8 M weights in 0.2 ms, 64 tokens of them in 4 ms; GB/s with its
// share of reading a buffer: 4.72 MB in 0.2 ms is 23.6 GB/s, 59% of 40
assert.equal(rowOf("Llama 3.2 1B w1", SIGNED)[4], "83.9");
assert.equal(rowOf("Llama 3.2 1B w1", "ORT DP4A 64×64, ternary")[4], "268.4");
assert.equal(rowOf("Llama 3.2 1B w1", SIGNED)[5], "23.6 (59%)");
// a form the check found wrong is not timed, and says the word warnings() picks up; an unsteady one says so
assert.match(rowOf("Ternary Bonsai 8B w1", SUMS)[3], /^WRONG in the check: /);
assert.match(rowOf("Llama 3.2 1B, a layer", "DP4A, fused \\(T175\\), ternary \\(engine\\)".replace(/\\/g, ""))[3], /^unsteady: /);
// the two forms against each other, a shape each where both ran: 0.2 / 0.16; none on the 8B, whose second was wrong
assert.deepEqual(ternaryForms(found), ["1.3× on Llama 3.2 1B w1"]);
assert.ok(lines.at(-1).endsWith("1.3× on Llama 3.2 1B w1"), lines.at(-1));
// no share where the device was lost; why there is no table
assert.equal(cellsOf(ternaryTable({ name: "t", result: found }, ceilings, { lost: "unknown: gone" }).find((line) => line.includes(SIGNED)))[5], "23.6");
assert.equal(ternaryTable({ name: "t", error: "the worker stopped" }, ceilings).at(-1), "failed: the worker stopped");
assert.equal(ternaryTable(undefined, ceilings).at(-1), "not measured");
assert.equal(ternaryTable({ name: "t", result: { rows: [], none: "not on a fallback adapter" } }, ceilings).at(-1), "not on a fallback adapter");
// the summary's line, and none where the ratios stand for nothing
const summaryOf = (gpu) => gpuSummary({ bandwidth: [], token: [], info: { result: { adapter: "a GPU", fallback: false } }, ternary: { name: "t", result: found } }, {}, gpu);
assert.ok(summaryOf({}).includes("GPU, ternary weights: the codes as they are less Σa against the engine's form 1.3× on Llama 3.2 1B w1"));
assert.ok(!summaryOf({ lost: "unknown: gone" }).some((line) => line.startsWith("GPU, ternary weights")));

// ---- the CPU's kernels: a pass's ms (ms of passes), against the int8 kernel of the same SIMD and tokens
const kernel = (name, weights, relaxed, tokens, ms, extra = {}) => ({ kernel: name, weights, relaxed, tokens, values: 8192 * 2048,
  bytes: weights === "ternary" ? 4456448 : 17825792, copies: 4, ms, passes: 10, ratio: 2, ...extra });
const kernels = { shape: [8192, 2048], rows: [kernel("matmul_q8r", "int8", true, 1, 20), kernel("matmul_t2r", "ternary", true, 1, 10),
  kernel("matmul_q8", "int8", false, 1, 30), kernel("matmul_t2", "ternary", false, 1, 40),
  kernel("matmul_q8r_tile", "int8", true, 16, 80), kernel("matmul_t2r_tile", "ternary", true, 16, 100, { unsteady: true }),
  { kernel: "rotate, 5120 values in blocks of 1024", weights: "rotate", relaxed: false, tokens: 1, values: 5120, bytes: 40960, copies: 1, ms: 0.11, passes: 20, ratio: 2 },
  { kernel: "rotate, 17408 values in blocks of 1024", weights: "rotate", relaxed: false, tokens: 1, values: 17408, bytes: 139264, copies: 1, error: "RangeError: out of bounds" }] };
const cpuLines = kernelsTable(kernels, 15);
const kernelRow = (name) => cellsOf(cpuLines.find((line) => line.startsWith(`| ${name} |`)));
assert.deepEqual(kernelRow("matmul_t2r"), ["matmul_t2r", "ternary", "yes", "1", "1.000", "16.8", "4.5 (30%)", "2.0×"]);
assert.equal(kernelRow("matmul_t2").at(-1), "0.75×", "the plain kernel against the plain int8 one");
assert.equal(kernelRow("matmul_t2r_tile")[4], "unsteady: 10.000");
assert.equal(kernelRow("matmul_t2r_tile")[5], "26.8", "16 tokens of 16.8 M weights in 10 ms");
assert.equal(kernelRow("matmul_t2r_tile").at(-1), "0.80×");
assert.equal(kernelRow("matmul_q8r").at(-1), "");
assert.equal(kernelRow("rotate, 5120 values in blocks of 1024")[4], "5.50 µs a call");
assert.equal(kernelRow("rotate, 17408 values in blocks of 1024")[4], "failed: RangeError: out of bounds");
assert.equal(kernelsTable({ rows: [], error: "TypeError: no" }).at(-1), "failed: TypeError: no");
assert.equal(kernelRow("matmul_q8").length, 8);
console.log("bench (ternary): ok");
