// tests/bench.ts's part (T643.13): the tokens generated on the GPU (src/bench/generate.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { generateTable } from "../../src/bench.ts";
import { cellsOf } from "./gpu.ts";

// T151: the table of tokens generated on the GPU: what a submission costs besides its tokens and how many times faster
// several a submission are than one, neither after a lost device, on a fallback adapter or with the sampling WRONG
const generateStep = { name: "tokens generated on the GPU", result: { model: "Llama 3.2 1B's width", layers: 2, vocab: 32000, GB: 0.21,
  dispatches: 13, chunkDispatches: 16, tokens: 16, settings: { temperature: 0.7, topp: 0.9, penalty: 1.3 }, work: { ms: 9.1 }, chunkWork: { ms: 4.25 },
  rows: [{ perSubmission: 1, msPerToken: 14.2, fixedMs: 5.1, chunks: 9.3 }, { perSubmission: 4, msPerToken: 10.4, fixedMs: 5.2, chunks: 5.5 },
    { perSubmission: 16, msPerToken: 9.4, fixedMs: 4.8, chunks: 4.51 }],
  sampling: { vocab: 128256, msEach: 0.31, over: 5114, flat: { msEach: 1.2, over: 128256, unsteady: true } } } };
const generateRight = { sampling: { ok: true }, "sampling in chunks": { ok: true }, "tokens on the GPU": { ok: true } };
const generateLines = generateTable(generateStep, generateRight);
// T175: the layer the tokens ran, by its name in the layer table
assert.ok(generateTable({ ...generateStep, result: { ...generateStep.result, layer: "DP4A, fused (T175)" } }, generateRight)[0].includes(`the layer table's "DP4A, fused (T175)"`));
assert.ok(generateLines.includes("| 1, each read back as it comes | 14.20 | 9.30 | 5.10 | 5.10 |  |"), generateLines.join("\n"));
assert.ok(generateLines.includes("| 16, read back once | 9.40 | 4.51 | 4.80 | 0.30 | 1.5× |"), generateLines.join("\n"));
assert.ok(generateLines[0].includes("(16 dispatches a token)"), generateLines[0]);
assert.ok(generateLines.at(-1).includes("9.10 ms (with the sampling in chunks 4.25 ms)"), generateLines.at(-1));
// T191: the sampling alone in one workgroup and in chunks, and the chunks' column WRONG where their check is
const bothSamplers = generateTable({ ...generateStep, result: { ...generateStep.result, sampling: { vocab: 128256, msEach: 5.893, over: 3695,
  chunks: { msEach: 0.812 }, flat: { msEach: 6.104, over: 128256, chunks: { error: "a | b" } } } } }, generateRight).at(-1);
assert.ok(bothSamplers.includes("one workgroup 5.893 ms (3695 tokens over the floor), in chunks 0.812 ms; on flat logits, one workgroup 6.104 ms " +
  "(128256 tokens over the floor), in chunks failed: a \\| b."), bothSamplers);
// T191's review: of the sampling in chunks, its last stage alone (SAMPLE's search), and a timing that failed on either
const withLast = generateTable({ ...generateStep, result: { ...generateStep.result, sampling: { vocab: 128256, error: "c | d", over: 3695,
  chunks: { msEach: 0.812, pick: { msEach: 0.25, unsteady: true } } } } }, generateRight).at(-1);
assert.ok(withLast.includes("one workgroup failed: c \\| d (3695 tokens over the floor), in chunks 0.812 ms (of it the last stage, one workgroup's " +
  "nucleus and draw: unsteady: 0.250 ms)."), withLast);
assert.ok(withLast.includes("the same settings but no penalty"), withLast);
const chunksWrong = generateTable(generateStep, { ...generateRight, "sampling in chunks": { ok: false } });
assert.ok(chunksWrong.includes("| 4, read back once | 10.40 | 5.50 (WRONG in the check) | 5.20 | 1.30 | 1.4× |"), chunksWrong.join("\n"));
assert.ok(chunksWrong[0].includes("The check found the sampling in chunks WRONG."), chunksWrong[0]);
assert.ok(generateLines.at(-1).includes("9.10 ms") && generateLines.at(-1).includes("0.310 ms (5114 tokens over the floor)"), generateLines.at(-1));
assert.ok(generateLines.at(-1).includes("on flat logits, unsteady: 1.200 ms (128256 tokens over the floor)."), generateLines.at(-1));
assert.ok(generateTable({ ...generateStep, result: { ...generateStep.result, sampling: { vocab: 128256, msEach: 0.31 } } }, generateRight).at(-1).endsWith("0.310 ms."));
for (const [label, lines] of [["right", generateLines], ["lost", generateTable(generateStep, generateRight, { lost: "lost" })],
  ["fallback", generateTable(generateStep, generateRight, { fallback: true })], ["wrong", generateTable(generateStep, { ...generateRight, sampling: { ok: false } })],
  ["no check", generateTable(generateStep)]]) {
  const rowsOf = lines.filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 2 + generateStep.result.rows.length, label);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!lines.join("\n").includes("undefined") && !lines.join("\n").includes("NaN"), label);
  if (label !== "right" && label !== "no check") {
    assert.ok(rowsOf.slice(2).every((line) => cellsOf(line).slice(3).every((cell) => cell.trim() === "")), `${label}: no derived numbers\n${lines.join("\n")}`);
  }
}
const noisy = { ...generateStep, result: { ...generateStep.result, rows: [generateStep.result.rows[0], { perSubmission: 8, msPerToken: 9.0, fixedMs: -0.8 }] } };
assert.ok(generateTable(noisy, generateRight).includes("| 8, read back once | 9.00 |  | under the noise | under the noise | 1.6× |"), generateTable(noisy, generateRight).join("\n"));
assert.ok(generateTable(generateStep, { ...generateRight, "tokens on the GPU": { ok: false } }).some((line) => line.startsWith("| 4, read back once (WRONG in the check) |")));
assert.ok(generateTable(generateStep, generateRight, { fallback: true }).some((line) => line.includes("none on a fallback adapter")));
assert.equal(generateTable({ name: "x", error: "a | b" })[0], "**Tokens generated on the GPU**: failed: a \\| b");
const unmeasured = generateTable({ ...generateStep, result: { ...generateStep.result, work: undefined, sampling: undefined, rows: [{ perSubmission: 1, msPerToken: 900 }, { perSubmission: 2, msPerToken: 800 }] } }, generateRight, { fallback: true });
assert.ok(unmeasured.at(-1).includes("not measured here"), unmeasured.at(-1));
