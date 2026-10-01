// E4: error messages with line breaks, pipes, backticks and brackets (what Dawn's validation errors look like) through the
// real checkVerdict(), the table functions and warnings()
import { warnings, checkVerdict, layerTable, tokenTable, generateTable, matVecTable, tableCell } from "../../src/bench.js";
import * as F from "./fixtures.mjs";

const { aCheck, aBandwidths, aCeilings, aTokens, aLayer, aBaseline } = F;
const dawn = 'Invalid ComputePipeline "tile 32x32".\n - While validating compute stage ([ShaderModule "main"], entryPoint: "main").\n - While calling [Device].CreateComputePipeline([ComputePipelineDescriptor "x|y"]).';

// 1. a shader the device refused: the verdict's error as the GPU section writes it (src/pages/benchmark.astro gpuMarkdown)
const check = { ...aCheck, "llama.cpp tiles 32×32, f16": { worstRelative: NaN, ok: false, error: dawn }, "TF.js tiles 32×32, vec4": { worstRelative: NaN, ok: false, error: "refused" } };
const verdicts = Object.entries(check), said = verdicts.filter(([, v]) => v.error || !v.ok).map(checkVerdict);
const line = `**Shaders against JavaScript**: ${verdicts.map(checkVerdict).join(", ")}`;
const markdown = [line, "", ...matVecTable(aBandwidths, check, aCeilings)].join("\n");
console.log("said:");
said.forEach((s) => console.log("   ", JSON.stringify(s)));
console.log("warnings:");
warnings([{ title: "GPU", status: "error", said, markdown }]).forEach((w) => console.log("   ", JSON.stringify(w).slice(0, 260), `(${w.length} chars)`));

// 2. the same with the error's line breaks and pipe collapsed (tableCell, as problems[0] and steps already are)
const check2 = { ...check, "llama.cpp tiles 32×32, f16": { worstRelative: NaN, ok: false, error: tableCell(dawn) } };
const verdicts2 = Object.entries(check2), said2 = verdicts2.filter(([, v]) => v.error || !v.ok).map(checkVerdict);
const markdown2 = [`**Shaders against JavaScript**: ${verdicts2.map(checkVerdict).join(", ")}`, ""].join("\n");
console.log("with the error collapsed first:");
warnings([{ title: "GPU", status: "error", said: said2, markdown: markdown2 }]).forEach((w) => console.log("   ", JSON.stringify(w).slice(0, 260), `(${w.length} chars)`));

// 3. other characters in a failed step
console.log("a step failing with backticks, brackets, <angle brackets>, @mentions and #12:");
warnings([{ title: "GPU", status: "ok", markdown: layerTable({ name: "a layer of a token", error: "`x` [y](z) <anonymous> @someone #12 *a* _b_" }).join("\n") }]).forEach((w) => console.log("   ", w));
