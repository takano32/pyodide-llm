// E2: the summary's room for warnings, for runs of 8 and 12 WRONG rows (real stage strings), quiet and noisy, with the
// GPU line as it is and shortened to a count (what the duplication costs), and with the warnings put by severity
import { gpuSummary, reportBody, shortReport, warnings, warningsBlock, checkVerdict, REPORT_LIMIT, pathTable, benchMarkdown, cpuSummary, deviceSummary,
  storageSummary, lineSummary, cpuTable, matVecTable, tokenTable, layerTable, layerStepsTable, generateTable } from "../../src/bench.js";
import * as F from "./fixtures.mjs";

const { android, aCheck, aSteps, aBaseline, aCpu, aDevice, aStorage, aLine, aLayer, aBandwidths, aCeilings, aTokens, aGenerate, stepsStep } = F;
const login = (text) => `https://github.com/login?return_to=${encodeURIComponent(`https://github.com/takano32/pyodide-llm/issues/new?${new URLSearchParams({ template: "benchmark.md", title: "Benchmark: tiny-lm 29M", body: reportBody(text) })}`)}`.length;

const layerNames = [
  "a layer, llama.cpp, separate steps", "a layer, llama.cpp, fused (T150)", "a layer, DP4A, separate steps", "a layer, DP4A, fused (T175)",
  "a layer, llama.cpp, fused (T150), flash_attn_vec (subgroups)", "a layer, llama.cpp, separate steps, subgroups", "a layer, llama.cpp, fused (T150), subgroups",
  "a layer, llama.cpp, fused (T150), subgroups, flash_attn_vec (subgroups)", "a layer, DP4A, fused (T175), the norms apart",
  "a layer, DP4A, fused (T175), the norms apart, flash_attn_vec (subgroups)", "a layer, DP4A, fused (T175), flash_attn_vec (subgroups)"];
const stagesFloat = "stages: q 1.3e-7, K and V 0 to the nearest float16, 152 toward zero, 0 away from it, 232 farther, attention 3.8e-3, silu(gate) × up 2.9e-3, stream 2.6e-3; cache 1.7e-3; first to depart: K and V";
const stagesDp4a = "stages: qkv quantized: scales 1.0e-7, 1 of 2112 off by 1, q 1.6e-7, K and V 1 to the nearest float16, 146 toward zero, 0 away from it, 237 farther, attention 3.2e-3, " +
  "o quantized: scales 8.8e-3, 32 of 2112 off by 1, ffn quantized: scales 2.3e-7, 0 of 2112 off by 1, silu(gate) × up 2.0e-7, down quantized: scales 5.1e-7, 0 of 2080 off by 1, stream 2.0e-7; cache 1.7e-3; first to depart: K and V";
const quantized = [{ point: "qkv", wrong: null, scale: 1.0e-7, apart: 1, of: 2112 }, { point: "o", wrong: "far from quantize_x's", scale: 8.8e-3, apart: 32, of: 2112 },
  { point: "ffn", wrong: null, scale: 2.3e-7, apart: 0, of: 2112 }, { point: "down", wrong: null, scale: 5.1e-7, apart: 0, of: 2080 }];
const tokensSteps = "steps: T 0: logits within 1.3e-2 of the largest, the most likely token the same at 6 of 6 steps, K and V 144 to the nearest float16, 739 toward zero, 58 away from it, 2131 farther; " +
  "T 0.7: logits 1.2e-3 1.1e-3 1.3e-3 1.6e-3 3.4e-3 2.4e-3 of the largest by step, the most likely token the same at 6 of 6 steps, K and V 181 to the nearest float16, 782 toward zero, 58 away from it, 2051 farther";

function scenario(wrongLayers) {
  const check = { ...aCheck };
  for (const name of layerNames.slice(0, wrongLayers)) {
    const dp4a = name.includes("DP4A");
    check[name] = { worstRelative: dp4a ? 1.7e-3 : 2.6e-3, ok: false, stages: dp4a ? stagesDp4a : stagesFloat, ...(dp4a ? { quantized } : {}) };
  }
  check["tokens on the GPU"] = { worstRelative: 0, ok: false, tokens: 8, edge: 0, problems: ["T 0.7, token 1: 331, the CPU 260"], steps: tokensSteps };
  return check;
}
const steady = { ...F.real, rows: F.real.rows.map((row) => ({ ...row, cpu: { ...row.cpu, unsteady: false } })) };
const heads = {
  quiet: [benchMarkdown(F.rows, android), pathTable(steady, "tiny-lm 29M")].join("\n\n"),
  noisy: [benchMarkdown(F.rows, android), pathTable({ ...F.real, perCount: [{ threads: 1, speed: 107.2, low: 105, high: 108 }, { threads: 2, speed: 151, low: 149, high: 160, unsteady: true }, { threads: 4, speed: 163.4, low: 160, high: 165 }] }, "tiny-lm 29M")].join("\n\n") };
const cpuSection = { title: "CPU", status: "ok", markdown: cpuTable(aCpu).join("\n") };

for (const wrongLayers of [7, 11]) {
  const check = scenario(wrongLayers), verdicts = Object.entries(check), said = verdicts.filter(([, v]) => v.error || !v.ok).map(checkVerdict);
  const steps = aSteps.map((s) => (s.name === "the shaders against JavaScript" ? { ...s, result: check } : s));
  const gpu = { title: "GPU", status: "wrong", said, markdown: [`**Shaders against JavaScript**: ${verdicts.map(checkVerdict).join(", ")}`, "", ...matVecTable(aBandwidths, check, aCeilings), "", ...tokenTable(aTokens, aBaseline, { check }),
    "", ...layerTable(aLayer, check, aCeilings), "", ...layerStepsTable(stepsStep, check, aCeilings), "", ...generateTable(aGenerate, check)].join("\n") };
  for (const [kind, head] of Object.entries(heads)) {
    const warned = warnings([{ title: "Model", markdown: head }, cpuSection, gpu]);
    // severity first: WRONG, FAILED and failed (and what a section says itself), then skipped, then unsteady
    const rank = (w) => (/\b(?:WRONG|FAILED|failed)\b/.test(w) || /lost|not checked/.test(w) ? 0 : /\bskipped\b/.test(w) ? 1 : 2);
    const bySeverity = [...warned].sort((a, b) => rank(a) - rank(b));
    const gpuLine = gpuSummary(steps, aBaseline);
    const shortGpu = [gpuLine[0].replace(/\(a layer.*$/, `(${said.length} WRONG, listed under Warnings)`), ...gpuLine.slice(1)];
    const lines = (g) => [...deviceSummary(aDevice), ...cpuSummary(aCpu), ...g, ...storageSummary(aStorage), ...lineSummary(aLine)];
    const count = (w, g) => {
      const summary = shortReport(heads[kind], lines(g), w, android);
      return `${summary.split("\n").filter((l) => w.some((x) => l === `- ${x}`)).length}/${w.length} (login ${login(summary)})`;
    };
    const wrongKept = (w, g) => shortReport(heads[kind], lines(g), w, android).split("\n").filter((l) => /WRONG/.test(l) && l.startsWith("- GPU:")).length;
    console.log(`${wrongLayers} layers + tokens WRONG, ${kind} head: section order ${count(warned, gpuLine)} [WRONG rows kept ${wrongKept(warned, gpuLine)}]; ` +
      `by severity ${count(bySeverity, gpuLine)} [${wrongKept(bySeverity, gpuLine)}]; GPU line shortened: section order ${count(warned, shortGpu)} [${wrongKept(warned, shortGpu)}], by severity ${count(bySeverity, shortGpu)} [${wrongKept(bySeverity, shortGpu)}]; ` +
      `GPU line ${gpuLine[0].length} -> ${shortGpu[0].length} chars`);
  }
}
