// tests/bench.ts's part (T643.13): where a layer's time goes, and a token's attention alone (src/bench/layersteps.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { layerStepsTable, layerTable } from "../../src/bench.ts";
import { cellsOf, layerStep, layerRight, layerCeilings, layerWrong } from "./gpu.ts";

// T202: where a layer's time goes. The times are multiples of 1/64 ms, so that every sum below is exact: the fused
// form's steps sum to 3.25 ms (matrices 2.75, attention 0.25, norms and quantizing 0.25), its layer 3.5 (the chain
// 0.25), the matrices alone 2.0 (fusing adds 0.75); the form with the norms apart sums to 3.375 against a layer of
// 3.25 (the chain less than nothing, as it can come out)
const NQ = "the norm with its quantizing (NORM_QUANTIZE)", NORM = "the norm (RMSNORM)", QD = "a vector of 2048 quantized (QUANTIZE)",
  QH = "a vector of 8192 quantized (QUANTIZE)", QKV = "q, k and v with RoPE and the cache", ATT = "the attention (flash attention's tile)",
  O = "o with the residual's add", GLU = "gate and up with SwiGLU", DOWN = "down with the residual's add";
const stepsStep = { name: "the steps of a layer", result: { model: "Llama 3.2 1B", pos: 127, layers: 16, copies: 2, GB: 0.0684, dp4a: true,
  forms: [
    { form: "DP4A, fused (T175), the norms apart", check: "a layer, DP4A, fused (T175), the norms apart", normApart: true, dispatches: 11, ms: 3.25, n: 8,
      steps: [[NORM, 2], [QD, 3], [QKV, 1], [ATT, 1], [O, 1], [GLU, 1], [QH, 1], [DOWN, 1]].map(([step, count]) => ({ step, count })) },
    { form: "DP4A, fused (T175)", check: "a layer, DP4A, fused (T175)", normApart: false, dispatches: 9, ms: 3.5, n: 8,
      steps: [[NQ, 2], [QKV, 1], [ATT, 1], [QD, 1], [O, 1], [GLU, 1], [QH, 1], [DOWN, 1]].map(([step, count]) => ({ step, count })) }],
  steps: [
    { step: NORM, kind: "small", ms: 0.0625 }, { step: QD, kind: "small", ms: 0.0625 }, { step: QKV, kind: "matrix", matrix: "qkv", ms: 0.25 },
    { step: ATT, kind: "attention", ms: 0.25 }, { step: O, kind: "matrix", matrix: "o", ms: 0.25 }, { step: GLU, kind: "matrix", matrix: "gateUp", ms: 1.5 },
    { step: QH, kind: "small", ms: 0.0625 }, { step: DOWN, kind: "matrix", matrix: "down", ms: 0.75 }, { step: NQ, kind: "small", ms: 0.0625 },
    { step: "q, k and v alone", kind: "alone", matrix: "qkv", ms: 0.125 }, { step: "o alone", kind: "alone", matrix: "o", ms: 0.125 },
    { step: "gate and up alone", kind: "alone", matrix: "gateUp", ms: 1.25 }, { step: "down alone", kind: "alone", matrix: "down", ms: 0.5 },
    { step: "a dispatch of one workgroup (SMALL: a vector of dim added)", kind: "floor", ms: 0.015625 }] } };
const stepsLines = layerStepsTable(stepsStep, layerRight, layerCeilings);
const stepsText = stepsLines.join("\n");
assert.ok(stepsLines.includes(`| ${NQ} | 62.5 |  | 2 × = 125.0 |`), stepsText);
assert.ok(stepsLines.includes(`| ${QD} | 62.5 | 3 × = 187.5 | 1 × = 62.5 |`), stepsText);
assert.ok(stepsLines.includes("| gate and up alone | 1250.0 |  |  |"), stepsText);
assert.ok(stepsText.includes('- "DP4A, fused (T175)", 9 dispatches: the layer 3.50 ms. Its steps one by one 3.25 ms (the matrices with what they write 2.75, ' +
  "the attention 0.25, the norms and quantizing 0.25); the layer less them, what the chain costs beyond each step alone: 0.25 ms. " +
  "The matrices alone 2.00 ms (34.2 GB/s, 85.5% of the buffer's reads (40.0 GB/s)); the layer less them, 1.50 ms, is what fusing adds to the matrices 0.75 " +
  "+ the attention 0.25 + the norms and quantizing 0.25 + the chain 0.25. A dispatch of one workgroup takes 15.6 µs: 9 of them 0.14 ms."), stepsText);
assert.ok(stepsText.includes('- "DP4A, fused (T175), the norms apart", 11 dispatches: the layer 3.25 ms. Its steps one by one 3.38 ms'), stepsText);
assert.ok(stepsText.includes("what the chain costs beyond each step alone: −0.13 ms"), stepsText);
// the fewest MB read before the same weights again (T202's review: a matrix alone on ranges of all the copies), and
// without it a word, never "undefined"
assert.ok(layerStepsTable({ ...stepsStep, result: { ...stepsStep.result, cycleMB: 75.5 } }, layerRight, layerCeilings)[0].includes("(none read again before 76 MB of other weights: not from the GPU's caches)"));
assert.ok(stepsLines[0].includes("(none read again soon: not from the GPU's caches)"), stepsLines[0]);
// the two DP4A forms share their matrices and attention: their layers less their norms and quantizing (3.25 − 0.375
// and 3.5 − 0.25) are how far the run's times move; a form that failed, or forms of other matrices, have no such line
assert.ok(stepsLines.includes("The two forms run the same matrices and attention and differ only in their norms and quantizing, so their layers less those should be the same: " +
  "2.88 and 3.25 ms, 0.38 apart. That is about how far this run's times move: read the chain, and any part, only where it is larger."), stepsText);
// every row as wide as the head, nothing undefined, NaN or empty: right, a lost device, a fallback adapter, no check,
// a step that failed, one unsteady, and a form that failed
const stepsFailed = { ...stepsStep, result: { ...stepsStep.result, steps: stepsStep.result.steps.map((one) => (one.step === ATT ? { step: ATT, kind: "attention", error: "a | b\nc" } : one)) } };
const stepsUnsteady = { ...stepsStep, result: { ...stepsStep.result, steps: stepsStep.result.steps.map((one) => (one.step === GLU ? { ...one, unsteady: true } : one)) } };
const stepsFailedForm = { ...stepsStep, result: { ...stepsStep.result, forms: [stepsStep.result.forms[0], { form: "DP4A, fused (T175)", check: "a layer, DP4A, fused (T175)", error: "refused" }] } };
assert.ok(!layerStepsTable(stepsFailedForm, layerRight, layerCeilings).some((line) => line.startsWith("The two forms")));
const otherMatrices = { ...stepsStep, result: { ...stepsStep.result, forms: stepsStep.result.forms.map((form, i) => (i ? form : { ...form, steps: form.steps.map((s) => (s.step === QKV ? { ...s, step: O } : s)) })) } };
assert.ok(!layerStepsTable(otherMatrices, layerRight, layerCeilings).some((line) => line.startsWith("The two forms")));
for (const [label, lines] of [["right", stepsLines], ["lost", layerStepsTable(stepsStep, layerRight, layerCeilings, { lost: "lost" })],
  ["fallback", layerStepsTable(stepsStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true })], ["no check", layerStepsTable(stepsStep)],
  ["a step failed", layerStepsTable(stepsFailed, layerRight, layerCeilings)], ["unsteady", layerStepsTable(stepsUnsteady, layerRight, layerCeilings)],
  ["a form failed", layerStepsTable(stepsFailedForm, layerRight, layerCeilings)]]) {
  const rowsOf = lines.filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 2 + stepsStep.result.steps.length, label);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!lines.join("\n").includes("undefined") && !lines.join("\n").includes("NaN"), label);
}
// a step that failed says so in its row, and the sums it is in are "?" rather than a number that leaves it out
const stepsFailedLines = layerStepsTable(stepsFailed, layerRight, layerCeilings);
assert.ok(stepsFailedLines.includes(`| ${ATT} | failed: a \\| b c | 1 × = ? | 1 × = ? |`), stepsFailedLines.join("\n"));
assert.ok(stepsFailedLines.join("\n").includes("Its steps one by one ? ms"), stepsFailedLines.join("\n"));
assert.ok(layerStepsTable(stepsUnsteady, layerRight, layerCeilings).includes(`| ${GLU} | unsteady: 1500.0 | 1 × = 1500.0 | 1 × = 1500.0 |`));
assert.ok(layerStepsTable(stepsUnsteady, layerRight, layerCeilings).some((line) => line.startsWith("Unsteady:")));
assert.ok(layerStepsTable(stepsFailedForm, layerRight, layerCeilings).includes('- "DP4A, fused (T175)": failed: refused'));
// no share of the buffer's reads after a lost device; a form the check found WRONG says so
assert.ok(!layerStepsTable(stepsStep, layerRight, layerCeilings, { lost: "lost" }).join("\n").includes("of the buffer's reads"));
assert.ok(layerStepsTable(stepsStep, layerRight, layerCeilings, { lost: "lost" }).some((line) => line.includes("the device was lost")));
assert.ok(layerStepsTable(stepsStep, layerWrong, layerCeilings).join("\n").includes('in "DP4A, fused (T175)" (WRONG in the check), µs a layer'));
assert.deepEqual(layerStepsTable({ name: "the steps of a layer", error: "no | here" }), ["**Where a layer's time goes**: failed: no \\| here"]);
// T208: the forms chosen by the layer table's fastest (named in the head, with the partner's norms), or by the packed
// int8 dot where it gave none (and why); the attention alone on its caches; the spares; the whole layer by timestamps
// in one line, or why not
const chosenStep = { ...stepsStep, result: { ...stepsStep.result, chosen: { by: "layer", fastest: "DP4A, fused (T175)" }, spares: 2, cycleMB: 137.2,
  caches: { count: 512, MB: 134.2 }, timestamps: { layers: 32, rounds: 5, forms: [{ form: "DP4A, fused (T175), the norms apart", ms: 3.125, span: 3.1875 }, { form: "DP4A, fused (T175)", ms: 3.375, span: 3.4375 }] } } };
const chosenLines = layerStepsTable(chosenStep, layerRight, layerCeilings), chosenText = chosenLines.join("\n");
assert.ok(chosenLines[0].includes('the fastest fused layer of the table above that the check found right, "DP4A, fused (T175)", as the engine chooses it (T152), ' +
  'and beside it the same with its norms apart: "DP4A, fused (T175), the norms apart" and "DP4A, fused (T175)"'), chosenLines[0]);
assert.ok(chosenLines[0].includes("and of 2 spare matrices of random weights (none read again before 137 MB of other weights"), chosenLines[0]);
assert.ok(chosenLines[0].includes("The attention alone reads 512 copies of the cache"), chosenLines[0]);
assert.ok(chosenText.includes('The whole layer by the GPU\'s own clock (timestamp-query), a check of the times above: "DP4A, fused (T175), the norms apart" 3.13 ms (first to last 3.19; above: 3.25), ' +
  '"DP4A, fused (T175)" 3.38 ms (first to last 3.44; above: 3.50) a layer, each the median of 5 submissions of 32 layers'), chosenText);
// T208's review: first to last cuts the timestamps by 65.5 µs / the layers or less
assert.ok(chosenText.includes("first to last by 2 µs or less whatever they do"), chosenText);
const oneForm = { ...chosenStep, result: { ...chosenStep.result, chosen: { by: "layer", fastest: "llama.cpp, fused (T150), subgroups" },
  forms: [{ ...stepsStep.result.forms[1], form: "llama.cpp, fused (T150), subgroups", check: "a layer, llama.cpp, fused (T150), subgroups" }], timestamps: { none: "no | clock" } } };
const oneFormLines = layerStepsTable(oneForm, layerRight, layerCeilings);
assert.ok(oneFormLines[0].includes('as the engine chooses it (T152): "llama.cpp, fused (T150), subgroups")'), oneFormLines[0]);
assert.ok(oneFormLines.includes("The whole layer by the GPU's own clock (timestamp-query), a check of the times above: not here: no \\| clock."), oneFormLines.join("\n"));
const byPacked = layerStepsTable({ ...stepsStep, result: { ...stepsStep.result, chosen: { by: "packed", why: "no layer table was timed before" } } }, layerRight, layerCeilings);
assert.ok(byPacked[0].includes("by the packed int8 dot alone (no layer table was timed before: ONNX Runtime's DP4A, the packed int8 dot is here)"), byPacked[0]);
assert.ok(layerStepsTable({ ...chosenStep, result: { ...chosenStep.result, timestamps: { error: "refused" } } }, layerRight, layerCeilings)
  .some((line) => line.endsWith("a check of the times above: failed: refused.")));
assert.ok(!stepsText.includes("GPU's own clock"), "no line where no timestamps were taken (a result before T208)");
for (const [label, lines] of [["chosen", chosenLines], ["one form", oneFormLines], ["by packed", byPacked],
  ["timestamps lost", layerStepsTable({ ...chosenStep, result: { ...chosenStep.result, timestamps: { layers: 32, rounds: 5, forms: [{ form: "DP4A, fused (T175)", ms: NaN, span: undefined }] } } }, layerRight, layerCeilings)]]) {
  const rowsOf = lines.filter((line) => line.startsWith("|"));
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!lines.join("\n").includes("undefined") && !lines.join("\n").includes("NaN"), `${label}: ${lines.join("\n")}`);
}
// T224: a token's attention alone by the positions it reads, under the steps' table: µs a length, the vec forms how
// many times faster than the tiles (not where either is unsteady, nor on a fallback adapter), a form not here, a
// length that failed; nothing where the result has no lengths (before T224), one line where they failed
const lengthsStep = { ...chosenStep, result: { ...chosenStep.result, lengths: { positions: [128, 1024, 4096], MB: 134.2, rows: [
  { attention: "the prompt's tiles (flash attention's tile)", times: [{ ms: 0.05 }, { ms: 0.4 }, { ms: 1.6 }] },
  { attention: "flash_attn_vec (subgroups)", times: [{ ms: 0.025 }, { ms: 0.1, unsteady: true }, { ms: 0.2 }] },
  { attention: "flash_attn_vec", times: [{ none: "a | b" }, { error: "refused" }, { ms: 0.4 }] }] } } };
const lengthsLines = layerStepsTable(lengthsStep, layerRight, layerCeilings), lengthsText = lengthsLines.join("\n");
assert.ok(lengthsLines.includes("| attention | 128 positions, µs | 1024 positions, µs | 4096 positions, µs |"), lengthsText);
assert.ok(lengthsLines.includes("| the prompt's tiles (flash attention's tile) | 50.0 | 400.0 | 1600.0 |"), lengthsText);
assert.ok(lengthsLines.includes("| flash_attn_vec (subgroups) | 25.0 (2.0× the tiles) | unsteady: 100.0 | 200.0 (8.0× the tiles) |"), lengthsText);
assert.ok(lengthsLines.includes("| flash_attn_vec | not here: a \\| b | failed: refused | 400.0 (4.0× the tiles) |"), lengthsText);
assert.ok(!layerStepsTable(lengthsStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true }).join("\n").includes("the tiles)"));
// T224's review: the tiles twice where the engine makes them otherwise (f16, subgroups): the vec rows against the
// engine's (base), the f32 tiles with no ratio
{
  const twoStep = { ...chosenStep, result: { ...chosenStep.result, lengths: { positions: [128, 1024, 4096], MB: 134.2, base: 1, rows: [
    { attention: "the prompt's tiles (flash attention's tile)", tiles: true, times: [{ ms: 0.9 }, { ms: 7.2 }, { ms: 28.8 }] },
    { attention: "the prompt's tiles, f16, subgroups (the engine's here)", tiles: true, times: [{ ms: 0.3 }, { ms: 2.4 }, { ms: 9.6 }] },
    { attention: "flash_attn_vec (subgroups)", times: [{ ms: 0.05 }, { ms: 0.12 }, { ms: 0.48 }] }] } } };
  const twoLines = layerStepsTable(twoStep, layerRight, layerCeilings), twoText = twoLines.join("\n");
  assert.ok(twoLines.includes("| the prompt's tiles (flash attention's tile) | 900.0 | 7200.0 | 28800.0 |"), twoText);
  assert.ok(twoLines.includes("| the prompt's tiles, f16, subgroups (the engine's here) | 300.0 | 2400.0 | 9600.0 |"), twoText);
  assert.ok(twoLines.includes("| flash_attn_vec (subgroups) | 50.0 (6.0× the engine's tiles) | 120.0 (20.0× the engine's tiles) | 480.0 (20.0× the engine's tiles) |"), twoText);
  assert.ok(twoText.includes("The tiles are here twice"), twoText);
  assert.ok(!twoText.includes("undefined") && !twoText.includes("NaN"), twoText);
  assert.ok(!twoText.includes("GB/s: the keys and values"), "no GB/s of the cache before the review's bytes");
  // T224's review: bytes, the keys and values of each length read once (Llama 3.2 1B's: positions × 512 × 2 B × 2): the
  // GB/s of every cell, and against the buffer's reads (40 GB/s here) where there is that ceiling
  const bytes = [128, 1024, 4096].map((positions) => positions * 512 * 4);
  const rateStep = { ...twoStep, result: { ...twoStep.result, lengths: { ...twoStep.result.lengths, bytes } } };
  const rateLines = layerStepsTable(rateStep, layerRight, layerCeilings), rateText = rateLines.join("\n");
  assert.ok(rateLines.includes("| the prompt's tiles (flash attention's tile) | 900.0 (0.3 GB/s, 1% of the buffer's reads) | 7200.0 (0.3 GB/s, 1% of the buffer's reads) | 28800.0 (0.3 GB/s, 1% of the buffer's reads) |"), rateText);
  assert.ok(rateLines.includes("| flash_attn_vec (subgroups) | 50.0 (6.0× the engine's tiles; 5.2 GB/s, 13% of the buffer's reads) | 120.0 (20.0× the engine's tiles; 17.5 GB/s, 44% of the buffer's reads) | 480.0 (20.0× the engine's tiles; 17.5 GB/s, 44% of the buffer's reads) |"), rateText);
  assert.ok(rateText.includes("GB/s: the keys and values of the length read once over the time, and its share of what a loop that only reads a buffer reads (40.0 GB/s)"), rateText);
  const noCeiling = layerStepsTable(rateStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true }).join("\n");
  assert.ok(noCeiling.includes("| flash_attn_vec (subgroups) | 50.0 (5.2 GB/s) | 120.0 (17.5 GB/s) | 480.0 (17.5 GB/s) |"), noCeiling);
  assert.ok(!rateText.includes("undefined") && !rateText.includes("NaN"), rateText);
  // an unsteady time has no rate
  const unsteadyRates = layerStepsTable({ ...rateStep, result: { ...rateStep.result, lengths: { ...rateStep.result.lengths, base: 0, rows: [{ attention: "a", times: [{ ms: 1, unsteady: true }, { ms: 1 }, { ms: 1 }] }] } } },
    layerRight, layerCeilings).join("\n");
  assert.ok(unsteadyRates.includes("| a | unsteady: 1000.0 | 1000.0 (2.1 GB/s, 5% of the buffer's reads) | 1000.0 (8.4 GB/s, 21% of the buffer's reads) |"), unsteadyRates);
}
assert.ok(!chosenText.includes("A token's attention alone"), "no lengths where none were taken (a result before T224)");
assert.ok(layerStepsTable({ ...chosenStep, result: { ...chosenStep.result, lengths: { error: "no | memory" } } }, layerRight, layerCeilings)
  .includes("**A token's attention alone, by the positions it reads** (T224): failed: no \\| memory"));
{
  const rowsOf = lengthsLines.slice(lengthsLines.indexOf("| attention | 128 positions, µs | 1024 positions, µs | 4096 positions, µs |")).filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 5, lengthsText);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, 4, line);
  assert.ok(!lengthsText.includes("undefined") && !lengthsText.includes("NaN"), lengthsText);
}
const unsteadyStep = { ...layerStep, result: { ...layerStep.result, rows: layerStep.result.rows.map((row) => (row.form === "llama.cpp, fused (T150)" ? { ...row, unsteady: true } : row)) } };
assert.ok(layerTable(unsteadyStep, layerRight).includes("| llama.cpp, fused (T150) | 5 | unsteady: 2.10 | 32.6 | 33.6 |  |"));
// a fallback adapter's few hundredths of a GB/s still show
const slowStep = { ...layerStep, result: { ...layerStep.result, rows: [{ ...layerStep.result.rows[0], msPerLayer: 2900, GBps: 0.0236 }] } };
assert.ok(layerTable(slowStep, layerRight, undefined, { fallback: true }).includes("| llama.cpp, separate steps | 14 | 2900.00 | 0.024 | 46400.0 |  |"));
assert.equal(layerTable({ name: "a layer of a token", error: "x | y" })[0], "**A layer of a token**: failed: x \\| y");
export { stepsStep, stepsFailedForm, unsteadyStep };
