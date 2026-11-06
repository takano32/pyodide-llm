// tests/bench.ts's part (T643.13): the summary in the link of a report too long for it (src/bench/summary.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { PASTE, REPORT_LIMIT, TOO_LONG, benchMarkdown, cpuBaseline, cpuSummary, cpuTable, deviceSummary, environmentOf, gpuSummary, lineSummary,
         loginUrl, parseReport, reportBody, shortReport, storageSummary, generateTable, layerTable, matVecTable, pathTable, reportTooLong, reportUrl,
         reportsTable, tokenTable } from "../../src/bench.ts";
import { rows, environment, everything, long } from "./rounds.ts";
import { stepsStep } from "./layersteps.ts";
import { real } from "./path.ts";

// T185: a report of a real GPU is about four times the link's limit (the owner's Android's shape: packed int8 dot,
// shader-f16 and subgroups, 8 logical cores, the page's path; the numbers made up). Where the whole is too long the link
// holds the head and a line a section, and asks for the whole from the clipboard; parseReport() reads it as before.
const android = environmentOf({ hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Mobile Safari/537.36" },
  { model: "tiny-lm 29M", pyodide: "314.0.7", build: "abc1234", site: "https://takano32.github.io/pyodide-llm/benchmark/" });
const aDevice = { cores: 8, memoryGB: 8, simd: true, relaxedSimd: true, memory64: true, crossOriginIsolated: true, sharedMemory: true, webgpu: true, opfs: true, syncHandle: true };
const aCpu = { backend: "SIMD kernels, int8, relaxed SIMD", shared: true, megabytes: 211, tokenMegabytes: 234, layerWeights: 121.6e6,
  rows: [1, 2, 4, 8].map((threads, i) => ({ threads, msPerToken: [11.2, 8.1, 7.6, 7.9][i], GBps: [18.9, 26.0, 27.8, 26.7][i], promptMsPerToken: [7.9, 4.6, 3.9, 4.4][i] })),
  ceilings: { read: [1, 2, 4, 8].map((threads, i) => ({ threads, GBps: [24.3, 31.2, 33.0, 32.1][i] })), dot: { GMACs: 44.2 }, dotRegisters: { GMACs: 61.5 }, fma: { GMACs: 17.3 } } };
const MATVEC = ["widened (T134)", "packed int8 (T134)", "llama.cpp mul_mat_vec (engine), 4 rows", "llama.cpp mul_mat_vec (engine), 4 rows, subgroups", "llama.cpp MMVQ, 4 rows",
  "llama.cpp MMVQ, 4 rows, subgroups", "ORT MatMulNBits, 8 rows", "ORT DP4A small M (engine), 4 rows"];
const matVecChecks = ["widen", "packed", ...MATVEC.slice(2)];
const aCheck = Object.fromEntries([...matVecChecks, "llama.cpp tiles 32×32, f16", "llama.cpp tiles 64×64, f16", "TF.js tiles 32×32, vec4", "ORT DP4A 64×64",
  "ORT DP4A 64×64, subgroups", "argmax", "a layer, llama.cpp, separate steps", "a layer, llama.cpp, fused (T150)", "a layer, DP4A, separate steps",
  "a layer, DP4A, fused (T175)", "sampling", "tokens on the GPU"].map((k) => [k, { ok: true, worstRelative: 3.2e-8 }]));
const aBandwidths = ["llm-jp-3 150M w1", "Llama 3.2 1B w1", "Llama 3.2 1B classifier"].map((name, j) => ({ name: `bandwidth: ${name}`,
  result: { rows: MATVEC.map((shader, i) => ({ shader, check: matVecChecks[i], GBps: 3.1 + i * 2.37 + j * 4.11 })), cpu: { GBps: 9.8 + j }, quantize: { msEach: 0.041 + j * 0.013 } } }));
const aCeilings = { global: { GBps: 48.3, MiB: 128 }, f32: { GFLOPS: 812 }, f16: { GFLOPS: 1530 }, dot4: { GOPS: 1210 }, shared: { GBps: 402 } };
const aTokens = ["llm-jp-3 150M", "llm-jp-3 150M, packed int8", "Llama 3.2 1B", "Llama 3.2 1B, packed int8", "Llama 3.2 1B, chosen on the GPU"].map((n, i) => ({
  name: `a token of ${n}`, result: { kind: n.includes("packed") ? "packed" : "widen", sample: n.includes("chosen"), GB: [0.17, 0.17, 1.4, 1.4, 1.4][i],
    dispatches: [120, 132, 240, 256, 242][i], msPerToken: 12.34 + i * 17.1, tokPerSecond: 81.0 - i * 13.3 } }));
const layerForms = ["llama.cpp, separate steps", "llama.cpp, fused (T150), the norms apart", "llama.cpp, fused (T150)"];
const aLayer = { name: "a layer of a token", result: { model: "Llama 3.2 1B", pos: 127, layers: 16, GB: 0.0612, rows: [
  ...[false, true].flatMap((subgroups) => layerForms.map((f) => ({ form: `${f}${subgroups ? ", subgroups" : ""}`, check: `a layer, ${f}`, base: "llama.cpp", fused: f !== layerForms[0], subgroups }))),
  ...["DP4A, separate steps", "DP4A, fused (T175), the norms apart", "DP4A, fused (T175)"].map((f, i) => ({ form: f, check: `a layer, ${f}`, base: "DP4A", fused: i > 0, subgroups: false }))]
  .map((row, i) => ({ ...row, dispatches: [14, 7, 5, 14, 7, 5, 18, 11, 9][i], msPerLayer: 3.1 - i * 0.211, GBps: 21.3 + i * 1.7 })) } };
const aGenerate = { name: "tokens generated on the GPU", result: { model: "Llama 3.2 1B", layers: 16, vocab: 128256, GB: 1.36, dispatches: 83, tokens: 16, layer: "DP4A, fused (T175)",
  settings: { penalty: 1.1, temperature: 0.7, topp: 0.9 }, work: { ms: 41.23 }, rows: [1, 4, 8, 16].map((perSubmission, i) => ({ perSubmission, msPerToken: 58.12 - i * 5.3, fixedMs: 16.87 + i * 3.1 })),
  sampling: { vocab: 128256, msEach: 1.234, over: 312, flat: { msEach: 4.567, over: 12045 } } } };
const PROMPT_SHADERS = ["batched (T135)", "llama.cpp tiles 32×32, f16", "llama.cpp tiles 64×64, f16", "TF.js tiles 32×32, vec4", "ORT DP4A 64×64", "ORT DP4A 64×64, subgroups"];
const aPrompt = { name: "a prompt all at once", result: { GB: 0.14, weights: 121.6e6, rows: [...PROMPT_SHADERS.flatMap((shader, i) => [1, 16, 64].map((tokens, j) =>
  ({ shader, packed: shader.startsWith("ORT"), tokens, ms: 12.3 + i * 7 + j * 11, msPerToken: 3.21 - j * 0.9 - i * 0.1, GFLOPS: 34 + i * 20 + j * 40 }))),
  { shader: "batched (T135), again at the end", again: true, tokens: 64, ms: 101.2, msPerToken: 0.5, GFLOPS: 154 }] } };
// T643.12: the GPU section's steps as the page keeps them, by the step's key (src/bench/gpusection.ts's GpuFound)
const aFound = { info: { name: "the adapter", result: { adapter: "arm · valhall", fallback: false, packed: true, features: ["shader-f16", "subgroups"], subgroupSizes: [16, 16] } },
  check: { name: "the shaders against JavaScript", result: aCheck }, bandwidth: aBandwidths, token: aTokens, layer: aLayer, generate: aGenerate,
  ceilings: { name: "the device's ceilings", result: aCeilings }, prompt: aPrompt };
// the same steps with another check's verdicts
const checkedBy = (found, result) => ({ ...found, check: { ...found.check, result } });
const aStorage = { mib: 256, pieces: 32, sequential: { seconds: 0.95 }, scattered: { seconds: 1.15 }, scatteredFlushEach: { seconds: 1.64 }, read: { seconds: 0.12 } };
const aLine = { site: { bytes: 8.4e6, firstByteMs: 162, MBps: 12.7 }, hf: { bytes: 33.6e6, firstByteMs: 863, MBps: 8.4 },
  paced: [1, 4, 8].map((rate, i) => ({ rate, MBps: [1, 4, 7.6][i] })) };
const aBaseline = cpuBaseline({ status: "ok", data: aCpu });
const aHead = [benchMarkdown(rows, android), pathTable(real, "tiny-lm 29M")].join("\n\n");
const aWhole = [aHead, "#### CPU", ...cpuTable(aCpu), "#### GPU", ...matVecTable(aBandwidths, aCheck, aCeilings), ...tokenTable(aTokens, aBaseline),
  ...layerTable(aLayer, aCheck, aCeilings), ...generateTable(aGenerate, aCheck)].join("\n");
const aLines = [...deviceSummary(aDevice), ...cpuSummary(aCpu), ...gpuSummary(aFound, aBaseline), ...storageSummary(aStorage), ...lineSummary(aLine)];
const aSummary = shortReport(aHead, aLines);
assert.ok(reportTooLong(aWhole, android), "a GPU's whole report is too long for the link");
assert.ok(loginUrl(aWhole, android, aSummary).length <= REPORT_LIMIT, `${loginUrl(aWhole, android, aSummary).length}`);
const aBody = new URL(reportUrl(aWhole, android, aSummary)).searchParams.get("body");
assert.ok(aBody.endsWith(`#### Summary\n\n${aLines.map((line) => `- ${line}`).join("\n")}\n\n${PASTE}`), aBody);
assert.ok(aBody.includes("**Device**: (") && aBody.includes(aHead), "the questions and the head as they are");
assert.deepEqual(parseReport(aBody), parseReport(reportBody(aWhole)), "parseReport() reads the summary as the whole");
assert.deepEqual(parseReport(aBody).rows.map((row) => row.name), ["everything", "without the kernels"]);
assert.equal(reportsTable([{ number: 1, url: "u", body: aBody }]).split("\n").length, 3, "a row of reportsTable()");
// every section's line, with the fastest of each and the CPU beside the GPU
assert.deepEqual(aLines, [
  "Browser: SIMD yes, relaxed SIMD yes, 64-bit memory yes, cross-origin isolated yes, WebGPU in a worker yes, private file system yes",
  "CPU: 7.6 ms a token with 4 software threads, 27.8 GB/s (93% of reading alone); a prompt 31.2 G MAC/s with 4; ceilings: reading alone 33.0 GB/s, relaxed_dot 44.2 G MAC/s, f32 17.3 G MAC/s",
  "GPU: arm · valhall; packed int8 dot yes, shader-f16 yes, subgroups yes (16 wide); the check 20 of 20 ok",
  "GPU, a token: llm-jp-3 150M 81.0 tok/s (0.50× the CPU); Llama 3.2 1B 54.4 tok/s (2.7× the CPU); the fastest matrix × vector of Llama 3.2 1B classifier ORT DP4A small M (engine), 4 rows, 27.9 GB/s (58% of reading a buffer)",
  "GPU, the fastest layer DP4A, fused (T175), 9 dispatches, 1.41 ms; generated 58.12 ms a token one a submission, 42.22 with 16 (1.4×)",
  "GPU, a prompt of 64 tokens: ORT DP4A 64×64, subgroups, 214 GFLOPS (4.3× the CPU)",
  "Storage: writes 283 MB/s in order, 233 far apart, 164 with a flush each piece; read back 2237 MB/s",
  "Line: this site 12.7 MB/s (first byte 162 ms), huggingface.co 8.4 MB/s (first byte 863 ms); huggingface.co read no faster than 1, 4, 8 MB/s: 1.00, 4.00, 7.60"]);
// no ratio on a fallback adapter or after a lost device, and a WRONG row is never the fastest nor held against the CPU
const onFallbackAdapter = gpuSummary({ ...aFound, info: { ...aFound.info, result: { ...aFound.info.result, fallback: true } } }, aBaseline);
assert.ok(onFallbackAdapter[0].includes("a fallback adapter: nothing timed") && !onFallbackAdapter.slice(1).join().includes("the CPU)") && !onFallbackAdapter.join().includes("% of"), onFallbackAdapter.join("\n"));
const afterLost = gpuSummary(aFound, aBaseline, { lost: "gone" });
assert.ok(afterLost[0].endsWith("; the device was lost (gone)") && !/\(\d[\d.]*×/.test(afterLost.join()) && !afterLost.join().includes("% of"), afterLost.join("\n"));
// T202: the summary says where the fastest broken-down layer's time goes (the norms apart: 3.25 ms, its small steps
// 0.375, its chain −0.125), not after a lost device
const withSteps = gpuSummary({ ...aFound, "layer steps": stepsStep }, aBaseline);
assert.ok(withSteps.includes("GPU, where the time of a layer DP4A, fused (T175), the norms apart goes: 3.25 ms, the matrices alone 2.00, fusing adds 0.75, " +
  "the attention 0.25, the norms and quantizing 0.38, the chain -0.13"), withSteps.join("\n"));
assert.ok(!gpuSummary({ ...aFound, "layer steps": stepsStep }, aBaseline, { lost: "gone" }).join().includes("where the time"));
const wrongDp4a = gpuSummary(checkedBy(aFound, { ...aCheck, "ORT DP4A small M (engine), 4 rows": { ok: false, worstRelative: 1 }, "ORT DP4A 64×64, subgroups": { ok: false, worstRelative: 1 } }), aBaseline);
assert.ok(wrongDp4a[0].includes("the check 18 of 20 ok (ORT DP4A small M (engine), 4 rows WRONG, ORT DP4A 64×64, subgroups WRONG)"), wrongDp4a[0]);
assert.ok(!wrongDp4a[1].includes("ORT DP4A small M") && !wrongDp4a[3].includes("ORT DP4A 64×64, subgroups"), wrongDp4a.join("\n"));
// a GPU section with only its adapter and check (the fallback's), or with no check: no undefined, no NaN
for (const lines of [gpuSummary({ info: aFound.info, check: aFound.check, bandwidth: [], token: [] }, {}), gpuSummary({ info: { name: "the adapter", result: {} }, check: { name: "the shaders against JavaScript", error: "lost" }, bandwidth: [], token: [] }, {}),
  cpuSummary({ ...aCpu, ceilings: { error: "no" } }), cpuSummary({ ...aCpu, shared: false, rows: aCpu.rows.slice(0, 1), ceilings: {} }),
  lineSummary({ site: { error: "x" }, hf: { error: "y" }, paced: [] }), lineSummary({ ...aLine, paced: [{ rate: 1, slower: true }] })]) {
  assert.ok(!lines.join().includes("undefined") && !lines.join().includes("NaN"), lines.join("\n"));
}
// a report that fits is in the link whole, summary or not; a summary too long as well leaves the request alone
assert.equal(new URL(reportUrl(everything, environment, aSummary)).searchParams.get("body"), reportBody(everything));
assert.ok(new URL(reportUrl(long, environment, long)).searchParams.get("body").endsWith(TOO_LONG));
export { android, aDevice, aCpu, aCheck, aBandwidths, aCeilings, aTokens, aLayer, aGenerate, aPrompt, aFound, checkedBy, aStorage, aLine, aBaseline,
         aHead, aWhole, aLines };
