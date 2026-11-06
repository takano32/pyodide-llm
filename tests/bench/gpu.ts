// tests/bench.ts's part (T643.13): the GPU section's tables of a token, a matrix × vector and a layer (src/bench/gpu.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { cpuBaseline, parseReport, reportBody, layerTable, matVecTable, tableCell, tokenTable } from "../../src/bench.ts";
import { markdown } from "./rounds.ts";
import { cpuSection, baseline } from "./cpu.ts";

const tokenSteps = [
  { name: "a token of Llama 3.2 1B", result: { kind: "widen", GB: 1.39, dispatches: 241, msPerToken: 81.6, tokPerSecond: 12.25 } },
  { name: "a token of Llama 3.2 1B, packed int8", result: { kind: "packed", GB: 1.39, dispatches: 241, msPerToken: 48.4, tokPerSecond: 20.66 } },
  { name: "a token of Llama 3.2 1B, chosen on the GPU", result: { kind: "widen", sample: true, GB: 1.39, dispatches: 242, msPerToken: 59.6, tokPerSecond: 16.78 } },
  { name: "a token of Llama 3.2 3B", error: "could not hold | the weights\nat all" },
  { name: "a token of llm-jp-3 150M", result: { model: "llm-jp-3 150M", error: "the GPU did not take 0.2 GB of weights" } }];
const right = { widen: { ok: true }, packed: { ok: true }, argmax: { ok: true } };
const variants = [["with the CPU", tokenTable(tokenSteps, baseline, { check: right })], ["without it", tokenTable(tokenSteps, cpuBaseline(undefined))],
  ["a lost device", tokenTable(tokenSteps, baseline, { lost: "lost" })], ["a fallback adapter", tokenTable(tokenSteps, baseline, { fallback: true })],
  ["a WRONG argmax", tokenTable(tokenSteps, baseline, { check: { ...right, argmax: { ok: false } } })]];
// the page's reading of a row (src/pages/benchmark.astro's rendered(), and GitHub's): a \| is a | of a cell
const cellsOf = (line) => line.split(/(?<!\\)\|/).slice(1, -1);
for (const [label, table] of variants) {
  // every row of the table as many cells as its header: a report that renders
  const rowsOf = table.filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 2 + tokenSteps.length, label);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!table.join("\n").includes("undefined") && !table.join("\n").includes("NaN"), label);
}
const ratios = (table) => table.filter((line) => /^\| Llama 3\.2 1B/.test(line)).map((line) => cellsOf(line).at(-1).trim());
const [withCpu, withoutCpu, lostTable, fallbackTable, wrongTable] = variants.map(([, table]) => table);
// 1.39 GB at 28.7 GB/s is 20.6 tok/s: the packed token the old column called 1.44× is 1.0×, GPU tok/s over CPU tok/s
assert.ok(withCpu.includes("| Llama 3.2 1B, packed int8 | 1.39 | 241 | 48.4 | 20.7 | 20.6 | 1.0× |"), withCpu.join("\n"));
assert.deepEqual(ratios(withCpu), ["0.59×", "1.0×", "0.81×"]);
assert.ok(withCpu[0].startsWith("The CPU (an estimate): each model's weights at the CPU section's fastest, 28.7 GB/s with 4 software threads."), withCpu[0]);
assert.ok(withoutCpu[0].includes("run the CPU section for it"), withoutCpu[0]);
assert.ok(withoutCpu.includes("| Llama 3.2 1B | 1.39 | 241 | 81.6 | 12.3 |  |  |"), withoutCpu.join("\n"));
// no ratio from a lost device (its times too fast), a fallback adapter, or a shader the check found wrong
assert.deepEqual(ratios(lostTable), ["", "", ""]);
assert.ok(lostTable.some((line) => line.includes("the device was lost")));
assert.deepEqual(ratios(fallbackTable), ["", "", ""]);
assert.deepEqual(ratios(wrongTable), ["0.59×", "1.0×", ""]);
assert.ok(wrongTable.some((line) => line.startsWith("| Llama 3.2 1B, chosen on the GPU (WRONG in the check) |")));
assert.deepEqual(ratios(tokenTable(tokenSteps, baseline, { check: { ...right, packed: { ok: false, error: "no dot4I8Packed" } } })), ["0.59×", "", "0.81×"]);
// a page that was not cross-origin isolated has one thread, and says so
const alone = cpuBaseline({ status: "ok", data: { shared: false, rows: [cpuSection.data.rows[0]] } });
assert.ok(tokenTable(tokenSteps, alone)[0].includes("18.9 GB/s with 1 software thread (not cross-origin isolated)."));
// an error's | and line breaks stay in their cell
assert.equal(tableCell("a | b\n c"), "a \\| b c");
assert.ok(withCpu.some((line) => line.includes("could not hold \\| the weights at all")));
// and the report with it still reads as the model's table alone
const withGpu = parseReport(reportBody([markdown, "#### GPU", ...withCpu].join("\n\n")));
assert.deepEqual(withGpu.rows.map((row) => row.name), ["everything", "without the kernels"]);
// T149: the matrix × vector table holds each GB/s against the buffer's reads, but not after a lost device (its times
// are no GPU's, and too fast: past 100%) nor on a fallback adapter; a failure's | stays in its cell
const matVecSteps = [{ name: "bandwidth: Llama 3.2 1B w1", result: { rows: [{ shader: "widened (T134)", check: "widen", GBps: 20 },
  { shader: "llama.cpp MMVQ, 4 rows", check: "llama.cpp MMVQ, 4 rows", error: "a | b" }], cpu: { GBps: 9 } } }];
const matVecCeilings = { global: { GBps: 40 } };
const matVecRow = (lines) => lines.find((line) => line.startsWith("| widened (T134) |"));
assert.equal(matVecRow(matVecTable(matVecSteps, {}, matVecCeilings)), "| widened (T134) | 20.0 GB/s (50.0%) |");
assert.equal(matVecRow(matVecTable(matVecSteps, {}, matVecCeilings, { lost: "lost" })), "| widened (T134) | 20.0 GB/s |");
assert.ok(matVecTable(matVecSteps, {}, matVecCeilings, { lost: "lost" }).some((line) => line.includes("the device was lost")));
assert.equal(matVecRow(matVecTable(matVecSteps, {}, { ...matVecCeilings, fallback: true })), "| widened (T134) | 20.0 GB/s |");
assert.ok(matVecTable(matVecSteps, {}, matVecCeilings).includes("| llama.cpp MMVQ, 4 rows | failed: a \\| b |"));
// T150: the layer's table: "faster than the separate steps" beside a fused row, against the separate steps of the
// same reduction, but none after a lost device, on a fallback adapter, for a row the check found WRONG or an unsteady
// one; the share of the buffer's reads beside GB/s, but not after a lost device or on a fallback adapter; every row
// as many cells as the header, a failure's | in its cell
const layerStep = { name: "a layer of a token", result: { model: "Llama 3.2 1B", pos: 127, layers: 16, GB: 0.0684, rows: [
  { form: "llama.cpp, separate steps", check: "a layer, llama.cpp, separate steps", base: "llama.cpp", fused: false, subgroups: false, dispatches: 14, msPerLayer: 4.2, GBps: 16.3 },
  { form: "llama.cpp, fused (T150)", check: "a layer, llama.cpp, fused (T150)", base: "llama.cpp", fused: true, subgroups: false, dispatches: 5, msPerLayer: 2.1, GBps: 32.6 },
  { form: "llama.cpp, separate steps, subgroups", check: "a layer, llama.cpp, separate steps, subgroups", base: "llama.cpp", fused: false, subgroups: true, error: "a | b\nc" },
  { form: "llama.cpp, fused (T150), subgroups", check: "a layer, llama.cpp, fused (T150), subgroups", base: "llama.cpp", fused: true, subgroups: true, dispatches: 5, msPerLayer: 1.9, GBps: 36 },
  { form: "DP4A, separate steps", check: "a layer, DP4A, separate steps", base: "DP4A", fused: false, subgroups: false, dispatches: 18, msPerLayer: 3.0, GBps: 22.8 },
  { form: "DP4A, fused (T175), the norms apart", check: "a layer, DP4A, fused (T175), the norms apart", base: "DP4A", fused: true, normApart: true, subgroups: false, dispatches: 11, msPerLayer: 2.0, GBps: 34.2 },
  { form: "DP4A, fused (T175)", check: "a layer, DP4A, fused (T175)", base: "DP4A", fused: true, subgroups: false, dispatches: 9, msPerLayer: 1.5, GBps: 45.6 }] } };
const layerRight = { "a layer, llama.cpp, separate steps": { ok: true }, "a layer, llama.cpp, fused (T150)": { ok: true }, "a layer, DP4A, separate steps": { ok: true },
  "a layer, DP4A, fused (T175)": { ok: true } };
const layerCeilings = { global: { GBps: 40 } };
const fasterOf = (lines, form) => cellsOf(lines.find((line) => line.startsWith(`| ${form} |`) || line.startsWith(`| ${form} (WRONG`))).at(-1).trim();
const layerLines = layerTable(layerStep, layerRight, layerCeilings);
assert.ok(layerLines.includes("| llama.cpp, fused (T150) | 5 | 2.10 | 32.6 (81.5%) | 33.6 | 2.0× |"), layerLines.join("\n"));
// T175: the DP4A forms against DP4A's separate steps (not llama.cpp's: 4.2 / 1.5 would say 2.8×)
assert.ok(layerLines.includes("| DP4A, fused (T175) | 9 | 1.50 | 45.6 (114.0%) | 24.0 | 2.0× |"), layerLines.join("\n"));
assert.ok(layerLines.includes("| DP4A, fused (T175), the norms apart | 11 | 2.00 | 34.2 (85.5%) | 32.0 | 1.5× |"), layerLines.join("\n"));
assert.equal(fasterOf(layerLines, "llama.cpp, separate steps"), "");
assert.equal(fasterOf(layerLines, "DP4A, separate steps"), "");
// no separate steps measured with subgroups: nothing to hold the fused one against
assert.equal(fasterOf(layerLines, "llama.cpp, fused (T150), subgroups"), "");
assert.ok(layerLines.includes("| llama.cpp, separate steps, subgroups | failed: a \\| b c | | | | |"), layerLines.join("\n"));
// a device without the packed int8 dot: the DP4A rows say so, and nothing is held against them
const noPackedStep = { ...layerStep, result: { ...layerStep.result, rows: layerStep.result.rows.map((row) => (row.base === "DP4A" ? { form: row.form, check: row.check, base: row.base, fused: row.fused, none: "no packed int8 dot here" } : row)) } };
const noPackedLines = layerTable(noPackedStep, layerRight, layerCeilings);
assert.ok(noPackedLines.includes("| DP4A, fused (T175) | not here: no packed int8 dot here | | | | |"), noPackedLines.join("\n"));
for (const [label, lines] of [["right", layerLines], ["lost", layerTable(layerStep, layerRight, layerCeilings, { lost: "lost" })],
  ["fallback", layerTable(layerStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true })], ["no check", layerTable(layerStep)], ["no packed", noPackedLines]]) {
  const rowsOf = lines.filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 2 + layerStep.result.rows.length, label);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!lines.join("\n").includes("undefined") && !lines.join("\n").includes("NaN"), label);
}
const lostLayer = layerTable(layerStep, layerRight, layerCeilings, { lost: "lost" });
assert.equal(fasterOf(lostLayer, "llama.cpp, fused (T150)"), "");
assert.ok(lostLayer.includes("| llama.cpp, fused (T150) | 5 | 2.10 | 32.6 | 33.6 |  |"), lostLayer.join("\n"));
assert.ok(lostLayer.some((line) => line.includes("the device was lost")));
const fallbackLayer = layerTable(layerStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true });
assert.equal(fasterOf(fallbackLayer, "DP4A, fused (T175)"), "");
assert.ok(fallbackLayer.includes("| llama.cpp, fused (T150) | 5 | 2.10 | 32.6 | 33.6 |  |"), fallbackLayer.join("\n"));
const layerWrong = { ...layerRight, "a layer, DP4A, fused (T175)": { ok: false } };
assert.equal(fasterOf(layerTable(layerStep, layerWrong), "DP4A, fused (T175)"), "");
assert.ok(layerTable(layerStep, layerWrong).some((line) => line.startsWith("| DP4A, fused (T175) (WRONG in the check) |")));
// the separate steps WRONG: nothing right to hold the fused one against
assert.equal(fasterOf(layerTable(layerStep, { ...layerRight, "a layer, DP4A, separate steps": { ok: false } }), "DP4A, fused (T175)"), "");
export { cellsOf, layerStep, layerRight, layerCeilings, layerWrong };
