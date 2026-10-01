// E1: how many of a real run's warnings the summary keeps within the login URL's 7,000, with the real WRONG rows of
// T225's probe (CI run 36867111944, "two-down": K and V rounded one float16 further than the nearest)
import { environmentOf, gpuSummary, reportBody, reportTooLong, reportUrl, loginUrl, shortReport, warnings, warningsBlock, checkVerdict,
  parseReport, REPORT_LIMIT, pathTable, benchMarkdown, cpuSummary, deviceSummary, storageSummary, lineSummary, cpuTable, matVecTable, tokenTable,
  layerTable, layerStepsTable, generateTable, cpuBaseline } from "../../src/bench.js";
import * as F from "./fixtures.mjs";

const { android, aCheck, aSteps, aBaseline, aHead, aCpu, aDevice, aStorage, aLine, aLayer, aBandwidths, aCeilings, aTokens, aGenerate, stepsStep } = F;

const layerNames = [
  "a layer, llama.cpp, separate steps", "a layer, llama.cpp, fused (T150)", "a layer, llama.cpp, fused (T150), flash_attn_vec (subgroups)",
  "a layer, llama.cpp, separate steps, subgroups", "a layer, llama.cpp, fused (T150), subgroups", "a layer, llama.cpp, fused (T150), subgroups, flash_attn_vec (subgroups)",
  "a layer, DP4A, separate steps", "a layer, DP4A, fused (T175), the norms apart", "a layer, DP4A, fused (T175), the norms apart, flash_attn_vec (subgroups)",
  "a layer, DP4A, fused (T175)", "a layer, DP4A, fused (T175), flash_attn_vec (subgroups)"];
const stagesFloat = "stages: q 1.3e-7, K and V 0 to the nearest float16, 152 toward zero, 0 away from it, 232 farther, attention 3.8e-3, silu(gate) × up 2.9e-3, stream 2.6e-3; cache 1.7e-3; first to depart: K and V";
const stagesDp4a = "stages: qkv quantized: scales 1.0e-7, 1 of 2112 off by 1, q 1.6e-7, K and V 1 to the nearest float16, 146 toward zero, 0 away from it, 237 farther, attention 3.2e-3, " +
  "o quantized: scales 8.8e-3, 32 of 2112 off by 1, ffn quantized: scales 2.3e-7, 0 of 2112 off by 1, silu(gate) × up 2.0e-7, down quantized: scales 5.1e-7, 0 of 2080 off by 1, stream 2.0e-7; cache 1.7e-3; first to depart: K and V";
const quantized = [{ point: "qkv", wrong: null, scale: 1.0e-7, apart: 1, of: 2112 }, { point: "o", wrong: "far from quantize_x's", scale: 8.8e-3, apart: 32, of: 2112 },
  { point: "ffn", wrong: null, scale: 2.3e-7, apart: 0, of: 2112 }, { point: "down", wrong: null, scale: 5.1e-7, apart: 0, of: 2080 }];
const tokensSteps = "steps: T 0: logits within 1.3e-2 of the largest, the most likely token the same at 6 of 6 steps, K and V 144 to the nearest float16, 739 toward zero, 58 away from it, 2131 farther; " +
  "T 0.7: logits 1.2e-3 1.1e-3 1.3e-3 1.6e-3 3.4e-3 2.4e-3 of the largest by step, the most likely token the same at 6 of 6 steps, K and V 181 to the nearest float16, 782 toward zero, 58 away from it, 2051 farther";
const wrongCheck = { ...aCheck };
for (const name of layerNames) {
  const dp4a = name.includes("DP4A");
  wrongCheck[name] = { worstRelative: dp4a ? 1.7e-3 : 2.6e-3, ok: false, stages: dp4a ? stagesDp4a : stagesFloat, stream: 2e-7, cache: 1.7e-3,
    ...(dp4a ? { quantized, ...(name.endsWith("(T175)") ? { sameAsNormsApart: { bitForBit: true, ulps: 0, apart: 0, stream: 0 } } : {}) } : {}) };
}
wrongCheck["tokens on the GPU"] = { worstRelative: 0, ok: false, tokens: 8, edge: 0, problems: ["T 0.7, token 1: 331, the CPU 260"], steps: tokensSteps };

const steps = aSteps.map((s) => (s.name === "the shaders against JavaScript" ? { ...s, result: wrongCheck } : s));
const gpu = { title: "GPU", status: "wrong", ...(() => {
  const verdicts = Object.entries(wrongCheck), said = verdicts.filter(([, v]) => v.error || !v.ok).map(checkVerdict);
  return { said, markdown: [`**Shaders against JavaScript**: ${verdicts.map(checkVerdict).join(", ")}`, "", ...matVecTable(aBandwidths, wrongCheck, aCeilings), "", ...tokenTable(aTokens, aBaseline, { check: wrongCheck }),
    "", ...layerTable(aLayer, wrongCheck, aCeilings), "", ...layerStepsTable(stepsStep, wrongCheck, aCeilings), "", ...generateTable(aGenerate, wrongCheck)].join("\n") };
})() };

// a quiet device: nothing unsteady besides the WRONG
const steady = { ...F.real, rows: F.real.rows.map((row) => ({ ...row, cpu: { ...row.cpu, unsteady: false } })) };
const cleanHead = [benchMarkdown(F.rows, android), pathTable(steady, "tiny-lm 29M")].join("\n\n");
const noisyHead = [benchMarkdown(F.rows, android), pathTable({ ...F.real, perCount: [{ threads: 1, speed: 107.2, low: 105, high: 108 }, { threads: 2, speed: 151, low: 149, high: 160, unsteady: true },
  { threads: 4, speed: 163.4, low: 160, high: 165 }, { threads: 8, speed: 120, low: 118, high: 121 }] }, "tiny-lm 29M")].join("\n\n");

const cpuSection = { title: "CPU", status: "ok", markdown: cpuTable(aCpu).join("\n") };
function report(head, label, sections) {
  const warned = warnings([{ title: "Model", markdown: head }, ...sections]);
  const lines = [...deviceSummary(aDevice), ...cpuSummary(aCpu), ...gpuSummary(steps, aBaseline), ...storageSummary(aStorage), ...lineSummary(aLine)];
  const whole = [head, warningsBlock(warned), ...sections.map((s) => `#### ${s.title}\n\n${s.markdown}`)].filter(Boolean).join("\n\n");
  const summary = shortReport(head, lines, warned, android);
  const kept = summary.split("\n").filter((l) => warned.some((w) => l === `- ${w}`)).length;
  const login = (text) => `https://github.com/login?return_to=${encodeURIComponent(`https://github.com/takano32/pyodide-llm/issues/new?${new URLSearchParams({ template: "benchmark.md", title: "Benchmark: tiny-lm 29M", body: reportBody(text) })}`)}`.length;
  console.log(`${label}: warnings ${warned.length}, chars ${warned.reduce((a, w) => a + w.length + 3, 0)}; whole ${whole.length} chars (login ${login(whole)}); ` +
    `summary kept ${kept} of ${warned.length}, login ${login(summary)} (limit ${REPORT_LIMIT}); the summary without any warning ${login(shortReport(head, lines, [], android))}; ` +
    `the GPU summary line: ${gpuSummary(steps, aBaseline)[0].length} chars`);
  console.log(`  summary tail: ${summary.split("\n").filter((l) => l.startsWith("- … ") || l.startsWith("- ") && l.includes("of them")).join(" | ")}`);
  return { warned, summary, lines };
}
const r1 = report(cleanHead, "a quiet device, 11 layers + tokens WRONG", [cpuSection, gpu]);
const r2 = report(noisyHead, "a noisy device, the same WRONG", [cpuSection, gpu]);
console.log("\nfirst warnings of the quiet device, with the length of each:");
r1.warned.forEach((w, i) => console.log(`${String(i).padStart(2)} ${String(w.length).padStart(4)} ${w.slice(0, 130)}`));
console.log("\nthe GPU line of the summary:\n", r1.lines.find((l) => l.startsWith("GPU: ")));
console.log("\nnoisy device: the warnings in order (length, start) and which of them the summary keeps:");
r2.warned.forEach((w, i) => console.log(`${String(i).padStart(2)} ${String(w.length).padStart(4)} ${r2.summary.includes(`- ${w}\n`) ? "KEPT" : "    "} ${w.slice(0, 120)}`));
