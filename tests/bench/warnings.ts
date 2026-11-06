// tests/bench.ts's part (T643.13): the warnings at the top of the report (src/bench/warnings.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { MEMORY_UNSAID, PASTE, REPORT_LIMIT, benchMarkdown, cpuSummary, cpuTable, deviceSummary, gpuSummary, lineSummary, loginUrl, parseReport,
         reportBody, shortReport, storageSummary, generateTable, layerStepsTable, layerTable, matVecTable, pathTable, reportTooLong, reportUrl,
         reportsTable, tokenTable, checkVerdict, pathWarnings, warnings, warningsBlock } from "../../src/bench.ts";
import { rows, iphone } from "./rounds.ts";
import { layerRight, layerCeilings } from "./gpu.ts";
import { stepsStep, stepsFailedForm, unsteadyStep } from "./layersteps.ts";
import { real } from "./path.ts";
import { android, aDevice, aCpu, aCheck, aBandwidths, aCeilings, aTokens, aLayer, aGenerate, aFound, checkedBy, aStorage, aLine, aBaseline, aHead,
         aWhole, aLines } from "./summary.ts";

// T227: all that came out WRONG, failed, unsteady or skipped, in one place at the top of the report and in the summary,
// in the page's words. The GPU section as the page's gpuMarkdown() writes it: the line of every verdict, then the tables
// (which mark the rows of a WRONG shader: those marks are not listed again); said, its verdicts that are not ok.
const gpuSection = (check) => {
  const verdicts = Object.entries(check), said = verdicts.filter(([, v]) => v.error || !v.ok).map(checkVerdict);
  return { title: "GPU", status: said.length ? "wrong" : "ok", said, markdown: [`**Shaders against JavaScript**: ${verdicts.map(checkVerdict).join(", ")}`, "",
    ...matVecTable(aBandwidths, check, aCeilings), "", ...tokenTable(aTokens, aBaseline, { check }), "", ...layerTable(aLayer, check, aCeilings), "",
    ...layerStepsTable(stepsStep, check, aCeilings), "", ...generateTable(aGenerate, check)].join("\n") };
};
const sectionsOf = (head, check) => [{ title: "Model", markdown: head }, { title: "CPU", status: "ok", markdown: cpuTable(aCpu).join("\n") }, gpuSection(check)];
const wholeOf = (head, sections) => [head, warningsBlock(warnings(sections)), ...sections.slice(1).map((s) => `#### ${s.title}\n\n${s.markdown}`)].filter(Boolean).join("\n\n");
// a run where nothing did: no list, the report and the summary as they were (the path's times all steady here)
const steady = { ...real, rows: real.rows.map((row) => ({ ...row, cpu: { ...row.cpu, unsteady: false } })) };
const cleanHead = [benchMarkdown(rows, android), pathTable(steady, "tiny-lm 29M")].join("\n\n");
const cleanSections = sectionsOf(cleanHead, aCheck);
assert.deepEqual(warnings(cleanSections), []);
assert.equal(warningsBlock([]), "");
assert.ok(!wholeOf(cleanHead, cleanSections).includes("Warnings"));
assert.equal(shortReport(cleanHead, aLines, [], android), shortReport(cleanHead, aLines));
// T225's run (the owner's PC, an NVIDIA GPU): the seven layers of the check WRONG, the DP4A ones with their quantized o,
// and the tokens generated on the GPU WRONG with the first token that differed; here also a time of the page's path
// unsteady (the head's table: a row under its headers)
const offLine = { ok: false, worstRelative: 1.8e-3 };
const farO = { ok: false, worstRelative: 9.1e-4, quantized: [{ point: "q, k, v", wrong: null, scale: 1.2e-7, apart: 3, of: 2112 }, { point: "o", wrong: "far from quantize_x's", scale: 0.3, apart: 2112, of: 2112 }] };
const t225Check = { ...aCheck, "a layer, llama.cpp, separate steps": offLine, "a layer, llama.cpp, fused (T150), the norms apart": offLine, "a layer, llama.cpp, fused (T150)": offLine,
  "a layer, llama.cpp, fused (T150), vec": offLine, "a layer, DP4A, separate steps": farO, "a layer, DP4A, fused (T175), the norms apart": farO, "a layer, DP4A, fused (T175)": farO,
  "tokens on the GPU": { ok: false, worstRelative: 0, tokens: 9, edge: 0, problems: ["T 0.7, token 2: 48, the CPU 483", "T 0.7, token 3: 9, the CPU 12"] } };
const t225Sections = sectionsOf(aHead, t225Check), t225Warned = warnings(t225Sections);
// (the WRONG ones first, in the order the check wrote them, then the rough time of the page's path: severity())
assert.deepEqual(t225Warned, [
  "GPU: a layer, llama.cpp, separate steps WRONG (worst 1.8e-3)",
  "GPU: a layer, llama.cpp, fused (T150) WRONG (worst 1.8e-3)",
  "GPU: a layer, DP4A, separate steps WRONG (worst 9.1e-4; quantized: o far from quantize_x's)",
  "GPU: a layer, DP4A, fused (T175) WRONG (worst 9.1e-4; quantized: o far from quantize_x's)",
  "GPU: tokens on the GPU WRONG (9 tokens, 0 next to a border: T 0.7, token 2: 48, the CPU 483)",
  "GPU: a layer, llama.cpp, fused (T150), the norms apart WRONG (worst 1.8e-3)",
  "GPU: a layer, llama.cpp, fused (T150), vec WRONG (worst 1.8e-3)",
  "GPU: a layer, DP4A, fused (T175), the norms apart WRONG (worst 9.1e-4; quantized: o far from quantize_x's)",
  "Model: the page: a prompt of 64 tokens; as chosen: 820 tok/s (800–830), GPU; CPU only: 612 tok/s (600–640, unsteady); GPU only: 830 tok/s (810–840); GPU ÷ CPU: 1.4×"]);
// the whole report: the list right under the top, before the first section, each line as it is; parseReport() reads
// the rounds above it as before
const t225Whole = wholeOf(aHead, t225Sections);
assert.ok(t225Whole.startsWith(`${aHead}\n\n#### Warnings\n\n${t225Warned.map((line) => `- ${line}`).join("\n")}\n\n#### CPU\n\n`), t225Whole.slice(0, 3000));
assert.deepEqual(parseReport(reportBody(t225Whole)), parseReport(reportBody(aWhole)));
assert.equal(reportsTable([{ number: 1, url: "u", body: reportBody(t225Whole) }]).split("\n").length, 3);
// the summary: the same list between the top and the sections' lines, within the link's limit; here all of it fits
const t225Lines = [...deviceSummary(aDevice), ...cpuSummary(aCpu), ...gpuSummary(checkedBy(aFound, t225Check), aBaseline),
  ...storageSummary(aStorage), ...lineSummary(aLine)];
const t225Summary = shortReport(aHead, t225Lines, t225Warned, android);
assert.ok(reportTooLong(t225Whole, android));
assert.ok(loginUrl(t225Whole, android, t225Summary).length <= REPORT_LIMIT, `${loginUrl(t225Whole, android, t225Summary).length}`);
assert.ok(t225Summary.startsWith(`${aHead}\n\n${warningsBlock(t225Warned)}\n\n#### Summary\n\n`) && t225Summary.endsWith(PASTE), t225Summary);
assert.equal(new URL(reportUrl(t225Whole, android, t225Summary)).searchParams.get("body"), reportBody(t225Summary), "the link holds the summary");
assert.deepEqual(parseReport(reportBody(t225Summary)), parseReport(reportBody(t225Whole)), "parseReport() reads the summary as the whole");
// more warnings than the link has room for: the first of them, and a line that says how many are left to the whole
// report (never dropped without a word); with room for none, their count alone
const many = [...t225Warned, ...Array.from({ length: 40 }, (_, i) => `GPU: a layer: form ${i}; GPU ms: unsteady: 0.${i}`)];
const cut = shortReport(aHead, t225Lines, many, android);
const listed = cut.split("\n").filter((line) => many.some((one) => line === `- ${one}`)).length;
assert.ok(listed > 0 && listed < many.length, `${listed}`);
assert.ok(cut.includes(`\n- ${many[listed - 1]}\n- … and ${many.length - listed} more, in the whole report below\n\n#### Summary`), cut);
assert.ok(loginUrl(t225Whole, android, cut).length <= REPORT_LIMIT && reportTooLong(shortReport(aHead, t225Lines, many), android));
assert.equal(new URL(reportUrl(t225Whole, android, cut)).searchParams.get("body"), reportBody(cut));
assert.equal(warningsBlock(many, 0), `#### Warnings\n\n- ${many.length} of them, in the whole report below`);
// what else is listed: a skipped round with why (the row under its headers), a section that failed as a whole (its
// Markdown is why), a device lost (the section's own line), a row with a failure or an unsteady time, a sentence
// outside the tables, and a section the page calls WRONG that names nothing; a line twice is there once
const others = warnings([{ title: "Model", markdown: iphone }, { title: "CPU", status: "error", markdown: "the worker failed: a | b\nc" },
  { title: "GPU", status: "error", said: ["**The device was lost** (gone): the times measured after it are not the GPU's."],
    markdown: ["**The device was lost** (gone): the times measured after it are not the GPU's.", "", ...layerTable(unsteadyStep, layerRight, layerCeilings), "",
      ...layerStepsTable(stepsFailedForm, layerRight, layerCeilings), "", ...layerStepsTable(stepsFailedForm, layerRight, layerCeilings)].join("\n") },
  { title: "Storage", status: "wrong", markdown: "| writes | s |\n|---|---:|\n| in order | 0.95 |" }]);
// (the worst first: what failed or is WRONG, then a skipped round, then a rough time; within each, the sections' order)
assert.deepEqual(others, [
  "CPU: failed: the worker failed: a | b c",
  "GPU: **The device was lost** (gone): the times measured after it are not the GPU's.",
  "GPU: a layer: llama.cpp, separate steps, subgroups; dispatches: failed: a | b c",
  'GPU: "DP4A, fused (T175)": failed: refused',
  "Storage: computed something wrong",
  `Model: what ran: without the kernels; tok/s: skipped; backend: ${MEMORY_UNSAID}`,
  "GPU: a layer: llama.cpp, fused (T150); dispatches: 5; GPU ms: unsteady: 2.10; GB/s: 32.6 (81.5%); its 16 layers, ms: 33.6"], others.join("\n"));
// a step of the GPU section that failed as a whole says so in that word (a token's row, a table's one line), so it is
// listed; one that was not run is no warning
assert.deepEqual(warnings([{ title: "GPU", status: "ok", markdown: [...tokenTable([{ name: "a token of Llama 3.2 1B", error: "out of memory" }], aBaseline), "",
  ...layerTable({ name: "a layer of a token", error: "x | y" }), "", ...generateTable({ name: "tokens generated on the GPU" })].join("\n") }]),
  ["GPU: a token (weights, dispatches, logits back): Llama 3.2 1B; GPU ms: failed: out of memory", "GPU: **A layer of a token**: failed: x \\| y"]);
// T227's review: a device's error message has line breaks (a validation error of Dawn's: "Invalid ComputePipeline …\n - While
// validating …\n - While calling …"). A verdict that FAILED with one is a line of the check's line and one warning: the
// message's own lines are no lines to read again, however it ran over them (the line of the check was cut at the first break
// into a warning of every verdict before it, 451 characters here)
const dawn = 'Invalid ComputePipeline "tile 32x32".\n - While validating compute stage ([ShaderModule "main"]).\n - While calling [Device].CreateComputePipeline().';
assert.equal(checkVerdict(["llama.cpp tiles 32×32, f16", { worstRelative: NaN, ok: false, error: dawn }]),
  'llama.cpp tiles 32×32, f16 FAILED (Invalid ComputePipeline "tile 32x32". - While validating compute stage ([ShaderModule "main"]). - While calling [Device].CreateComputePipeline().)');
const refused = gpuSection({ ...aCheck, "llama.cpp tiles 32×32, f16": { worstRelative: NaN, ok: false, error: dawn }, "TF.js tiles 32×32, vec4": { worstRelative: NaN, ok: false, error: "refused | x" } });
assert.ok(!refused.said.some((line) => line.includes("\n")) && !refused.markdown.split("\n")[0].includes("\n"));
assert.ok(refused.markdown.split("\n")[0].includes("FAILED (Invalid ComputePipeline") && refused.markdown.split("\n")[0].includes("CreateComputePipeline()."), "the check's line holds the whole message");
assert.deepEqual(warnings([refused]), [
  'GPU: llama.cpp tiles 32×32, f16 FAILED (Invalid ComputePipeline "tile 32x32". - While validating compute stage ([ShaderModule "main"]). - While calling [Device].CreateComputePipeline().)',
  "GPU: TF.js tiles 32×32, vec4 FAILED (refused \\| x)"]);
// what the section says itself runs over lines as well (the page's lost device and unchecked shaders): once, whole, and
// not again by a piece of it with a word in it
const lost = "**The device was lost** (destroyed: Device was destroyed.\n - While calling [Queue].Submit() failed): the times measured after it are not the GPU's.";
assert.deepEqual(warnings([{ title: "GPU", status: "error", said: [lost], markdown: `${lost}\n\n${layerTable({ name: "a layer of a token", error: "a\nb" }).join("\n")}` }]),
  ["GPU: **The device was lost** (destroyed: Device was destroyed. - While calling [Queue].Submit() failed): the times measured after it are not the GPU's.",
    "GPU: **A layer of a token**: failed: a b"]);
// T227's review: the worst first. T225 made the WRONG rows 250 to 530 characters each (the worker's line of the stages and
// where the layer departed first; the words are of CI run 36867111944's probe, K and V rounded a float16 too far), and a
// device with every one of the 11 forms of the layer table and the tokens WRONG has 12 of them. At the link's limit a
// summary has room for one or two. On a device whose page's path is rough all through (as CI's are) the section order put
// three rough times and the counts of threads before the first WRONG row: the summary kept one of them and no WRONG row.
const stagesFloat = "stages: q 1.3e-7, K and V 0 to the nearest float16, 152 toward zero, 0 away from it, 232 farther, attention 3.8e-3, silu(gate) × up 2.9e-3, stream 2.6e-3; cache 1.7e-3; first to depart: K and V";
const stagesDp4a = "stages: qkv quantized: scales 1.0e-7, 1 of 2112 off by 1, q 1.6e-7, K and V 1 to the nearest float16, 146 toward zero, 0 away from it, 237 farther, attention 3.2e-3, " +
  "o quantized: scales 8.8e-3, 32 of 2112 off by 1, ffn quantized: scales 2.3e-7, 0 of 2112 off by 1, silu(gate) × up 2.0e-7, down quantized: scales 5.1e-7, 0 of 2080 off by 1, stream 2.0e-7; cache 1.7e-3; first to depart: K and V";
const elevenForms = ["llama.cpp, separate steps", "llama.cpp, fused (T150)", "llama.cpp, fused (T150), flash_attn_vec (subgroups)", "llama.cpp, separate steps, subgroups", "llama.cpp, fused (T150), subgroups",
  "llama.cpp, fused (T150), subgroups, flash_attn_vec (subgroups)", "DP4A, separate steps", "DP4A, fused (T175), the norms apart", "DP4A, fused (T175), the norms apart, flash_attn_vec (subgroups)",
  "DP4A, fused (T175)", "DP4A, fused (T175), flash_attn_vec (subgroups)"];
const probeCheck = { ...aCheck, "tokens on the GPU": { ok: false, worstRelative: 0, tokens: 8, edge: 0, problems: ["T 0.7, token 1: 331, the CPU 260"],
  steps: "steps: T 0: logits within 1.3e-2 of the largest, the most likely token the same at 6 of 6 steps, K and V 144 to the nearest float16, 739 toward zero, 58 away from it, 2131 farther" } };
for (const form of elevenForms) {
  probeCheck[`a layer, ${form}`] = form.startsWith("DP4A")
    ? { ok: false, worstRelative: 1.7e-3, stages: stagesDp4a, quantized: [{ point: "qkv", wrong: null, scale: 1e-7, apart: 1, of: 2112 }, { point: "o", wrong: "far from quantize_x's", scale: 8.8e-3, apart: 32, of: 2112 }] }
    : { ok: false, worstRelative: 2.6e-3, stages: stagesFloat };
}
const roughHead = [benchMarkdown(rows, android), pathTable({ ...real, rows: real.rows.map((row) => ({ ...row, cpu: { ...row.cpu, unsteady: true } })),
  perCount: [{ threads: 1, speed: 107, low: 100, high: 108, unsteady: true }, { threads: 4, speed: 163, low: 120, high: 165, unsteady: true }] }, "tiny-lm 29M")].join("\n\n");
const probeWarned = warnings(sectionsOf(roughHead, probeCheck));
const probeWrong = probeWarned.filter((line) => line.startsWith("GPU: ") && line.includes(" WRONG ("));
assert.equal(probeWrong.length, 12, probeWarned.join("\n"));
assert.deepEqual(probeWarned.slice(0, 12), probeWrong, "the WRONG ones come first");
assert.ok(probeWarned.length > 12 && probeWarned.slice(12).every((line) => line.startsWith("Model: ") && line.includes("unsteady")), "then the rough times");
const probeLines = [...deviceSummary(aDevice), ...cpuSummary(aCpu), ...gpuSummary(checkedBy(aFound, probeCheck), aBaseline),
  ...storageSummary(aStorage), ...lineSummary(aLine)];
const probeSummary = shortReport(roughHead, probeLines, probeWarned, android);
const probeKept = probeSummary.split("\n").filter((line) => probeWarned.some((one) => line === `- ${one}`));
assert.ok(loginUrl(aWhole, android, probeSummary).length <= REPORT_LIMIT, `${loginUrl(aWhole, android, probeSummary).length}`);
assert.ok(probeKept.length >= 1 && probeKept.length < 12 && probeKept.every((line) => line.startsWith("- GPU: ") && line.includes(" WRONG (")), `the summary keeps WRONG rows first: ${probeKept.join("\n")}`);
assert.ok(probeSummary.includes(`- … and ${probeWarned.length - probeKept.length} more, in the whole report below`), "and says how many are left to the whole report");
// T227's review: what the model page's path says went wrong in words none of warnings()'s: the GPU stopped while the sides
// were timed (forward.js's reasons: only "the GPU failed on …" has "failed" in it), software threads that stopped or did not
// start, a search for their count that did not end. pathTable() writes them in its first line, pathWarnings() hands the same
// words to warnings() as the section's own; a page that is not isolated has one thread, which is no failure
const steadyPath = { ...real, rows: real.rows.map((row) => ({ ...row, cpu: { ...row.cpu, unsteady: false } })) };
const stoppedGpu = (lost) => ({ ...steadyPath, gpu: { ...steadyPath.gpu, lost } });
const notFinite = "the GPU computed logits that are not finite numbers (NaN or infinity) at position 3";
const pathCases = [
  [stoppedGpu("the GPU said nothing for 10 s"), ["WebGPU stopped while timed: the GPU said nothing for 10 s"]],
  [stoppedGpu(notFinite), [`WebGPU stopped while timed: ${notFinite}`]],
  [{ ...steadyPath, threads: 1, how: { alone: "a software thread stopped" } }, ["a software thread stopped"]],
  [{ ...steadyPath, threads: 1, how: { alone: "not the 4 asked for: its software threads did not start" } }, ["not the 4 asked for: its software threads did not start"]],
  [{ ...steadyPath, how: { unfinished: 120 } }, ["the search had not ended after 120 s"]],
  [{ ...steadyPath, how: { remembered: true, stopped: true } }, ["a software thread stopped while timed, and one thread went on"]],
  [{ ...stoppedGpu("the GPU's worker stopped answering for 10 s"), how: { unfinished: 120, stopped: true } },
    ["WebGPU stopped while timed: the GPU's worker stopped answering for 10 s", "the search had not ended after 120 s", "a software thread stopped while timed, and one thread went on"]],
];
for (const [paths, said] of pathCases) {
  const head = [benchMarkdown(rows, android), pathTable(paths, "tiny-lm 29M")].join("\n\n");
  assert.deepEqual(pathWarnings(paths), said);
  for (const line of said) assert.ok(head.split("\n").find((one) => one.startsWith("**The model page's path**")).includes(line), `${line} is the table's own words`);
  assert.deepEqual(warnings([{ title: "Model", markdown: head, said: pathWarnings(paths) }]), said.map((line) => `Model: ${line}`));
}
// none of them where nothing stopped: a path that went well, one with no shared memory, one that went wrong whole (its own word)
for (const paths of [steadyPath, { ...steadyPath, threads: 1, how: { alone: "no shared memory here" } }, { ...steadyPath, how: { searched: [[8, 4, 4]], remembered: false } }, undefined, { error: "x" }]) {
  assert.deepEqual(pathWarnings(paths), []);
}
assert.deepEqual(warnings([{ title: "Model", markdown: pathTable({ error: "x" }, "m"), said: pathWarnings({ error: "x" }) }]), ["Model: **The model page's path** (m): failed: x"]);
// a line of several sentences lists the sentence with the word, not the line: a form of the layer's steps that was unsteady
// is one in a bullet of five sentences (549 characters)
const unsteadyForm = { ...stepsStep, result: { ...stepsStep.result, forms: stepsStep.result.forms.map((form, i) => (i ? { ...form, unsteady: true } : form)) } };
export { unsteadyForm };
