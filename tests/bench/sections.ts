// tests/bench.ts's part (T643.13): the sections' results and the whole report (src/bench/device.ts, storage.ts, sections.ts, whole.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { benchMarkdown, cpuBaseline, cpuSummary, cpuTable, deviceSummary, gpuSummary, lineSummary, shortReport, generateTable, layerStepsTable,
         layerTable, matVecTable, pathTable, reportTooLong, reportUrl, tableCell, tokenTable, checkVerdict, pathWarnings, warnings, warningsBlock,
         SECTIONS, STATE, TITLES, deviceResult, cpuResult as cpuOf, DOWNLOAD_MBPS, storageNone, storageResult, lineResult, overheadMarkdown,
         ceilingsMarkdown, promptMarkdown, BRIDGE_ROUNDS, gpuNone, gpuResult, gpuMarkdown, reportOf } from "../../src/bench.ts";
import fs from "node:fs";
import { rows, pageText } from "./rounds.ts";
import { layerRight, layerCeilings } from "./gpu.ts";
import { real } from "./path.ts";
import { android, aDevice, aCpu, aCheck, aBandwidths, aCeilings, aTokens, aLayer, aGenerate, aPrompt, aFound, checkedBy, aStorage, aLine, aBaseline } from "./summary.ts";
import { unsteadyForm } from "./warnings.ts";

assert.deepEqual(warnings([{ title: "GPU", status: "ok", markdown: layerStepsTable(unsteadyForm, layerRight, layerCeilings).join("\n") }]),
  ['GPU: "DP4A, fused (T175)", 9 dispatches: the layer (unsteady) 3.50 ms.']);
// WRONG in a row of its own (no verdict of the check says it): the storage section's read back (T643.11: storageResult(),
// which the page wrote itself before)
const aRun = { ...aStorage, mib: 256, pieces: 32, openSeconds: 0.0123, secondHandle: "refused (NoModificationAllowedError)", reopen: "yes", truncateSeconds: 0.1,
  sequential: { seconds: 0.33, flushSeconds: 0.31 }, scattered: { seconds: 0.4, flushSeconds: 0.2 }, scatteredFlushEach: { seconds: 0.5, flushSeconds: 0.25 }, read: { seconds: 0.04, wrong: 0 } };
const storageWrong = storageResult({ result: { ...aRun, read: { seconds: 0.04, wrong: 3 } } });
assert.equal(storageWrong.status, "wrong");
assert.deepEqual(storageWrong.markdown.split("\n"), ["256 MiB in 32 pieces of 8 MiB; opening a file 12.3 ms; a second handle on it: refused (NoModificationAllowedError); opened again after close: yes", "",
  "| writes | s | of it flushing, s | MB/s | × a download of 8.3 MB/s |", "|---|---:|---:|---:|---:|",
  "| in order, one flush | 0.33 | 0.31 | 813 | 98× |", "| far apart, one flush | 0.40 | 0.20 | 671 | 81× |",
  "| far apart, a flush and a record each piece | 0.50 | 0.25 | 537 | 65× |", "| read back in order (3 pieces WRONG) | 0.04 |  | 6711 | 809× |"]);
assert.deepEqual(warnings([{ title: "Storage", ...storageWrong }]),
  ["Storage: writes: read back in order (3 pieces WRONG); s: 0.04; MB/s: 6711; × a download of 8.3 MB/s: 809×"]);
const storageRight = storageResult({ result: aRun });
assert.equal(storageRight.status, "ok");
assert.ok(storageRight.markdown.endsWith("| read back in order, every piece right | 0.04 |  | 6711 | 809× |") && storageRight.data.mib === 256, storageRight.markdown);
assert.deepEqual(warnings([{ title: "Storage", ...storageRight }]), []);
// T446 (2): a private file system that has the call but refuses to open a file so is none, not a failure; other failures fail
const refusal = "NotSupportedError: Backend does not support this operation (while opening the file (createSyncAccessHandle))";
assert.deepEqual(storageResult({ error: refusal }), { status: "none", markdown: `no private file system that a worker can write in place here: it refuses to open a file so (${refusal})` });
assert.deepEqual(storageResult({ error: "QuotaExceededError: full" }), { status: "error", markdown: "QuotaExceededError: full" });
assert.deepEqual(storageResult({ error: "NotSupportedError: something else" }), { status: "error", markdown: "NotSupportedError: something else" });
assert.deepEqual(storageNone({ result: { opfs: false, syncHandle: false } }), { status: "none", markdown: "no private file system that a worker can write in place here" });
assert.deepEqual(storageNone({ error: "x" }), { status: "error", markdown: "x" });
assert.equal(storageNone({ result: { opfs: true, syncHandle: true } }), undefined);
assert.equal(DOWNLOAD_MBPS, 8.3);
// the line's table (lineResult()): both fetched, the paced reads, a failure in a cell's words; failed only where both fetches failed
const lineOk = lineResult({ result: { ...aLine, paced: [{ rate: 1, MBps: 1.02 }, { rate: 4, slower: true }, { rate: 8, error: "a | b" }, { rate: 2, MBps: 2.5 }] } });
assert.equal(lineOk.status, "ok");
assert.deepEqual(lineOk.markdown.split("\n"), ["| fetched | size | first byte | speed |", "|---|---:|---:|---:|",
  "| this site: a part of the model | 8.4 MB | 162 ms | 12.7 MB/s |", "| huggingface.co: a range of a model | 33.6 MB | 863 ms | 8.4 MB/s |", "",
  "| huggingface.co read no faster than | got | within 10% |", "|---:|---:|---|",
  "| 1 MB/s | 1.02 MB/s | yes |", "| 4 MB/s | the line is slower | |", "| 8 MB/s | failed: a \\| b | |", "| 2 MB/s | 2.50 MB/s | no |"]);
const lineDown = lineResult({ result: { site: { error: "offline" }, hf: { error: "offline" }, paced: [] } });
assert.equal(lineDown.status, "error");
assert.ok(lineDown.markdown.includes("| this site: a part of the model | failed: offline | | |") && !lineDown.markdown.includes("read no faster"), lineDown.markdown);
assert.equal(lineResult({ result: { site: aLine.site, hf: { error: "offline" }, paced: [] } }).status, "ok", "one fetch is enough");
assert.deepEqual(lineResult({ error: "gone" }), { status: "error", markdown: "gone" });
// this browser's table (deviceResult())
const deviceFound = deviceResult({ result: { ...aDevice, memoryGB: null, relaxedSimd: false, usage: 1.5e9, quota: 9.9e10 } });
assert.equal(deviceFound.status, "ok");
assert.deepEqual(deviceFound.markdown.split("\n"), ["| feature | here |", "|---|---|", "| logical cores | 8 |", "| memory the browser says | not told |",
  "| WebAssembly SIMD | yes |", "| relaxed SIMD | none |", "| 64-bit memory (models past 4 GB) | yes |", "| cross-origin isolated (software threads) | yes |",
  "| WebGPU in a worker | yes |", "| private file system, writable in place | yes |", "| storage the browser grants | 1.5 GB used of 99.0 GB |"]);
assert.ok(deviceResult({ result: { ...aDevice, cores: null } }).markdown.includes("| logical cores | ? |\n| memory the browser says | 8 GB or more |"));
assert.ok(deviceResult({ result: aDevice }).markdown.endsWith("| storage the browser grants | ? used of ? |"));
assert.ok(deviceResult({ result: { ...aDevice, sharedMemory: false } }).markdown.includes("| cross-origin isolated (software threads) | none |"), "isolated, but no shared memory: no threads");
assert.deepEqual(deviceResult({ error: "x" }), { status: "error", markdown: "x" });
// the sections' own words of what went wrong reach warnings() as said (T643.11: src/bench/'s cpuOf(), gpuResult() and
// reportOf(), which were the page's before and were read here as its text): the CPU section's, the GPU section's (written
// again with the CPU section's result when that runs after it), the model section's path, and the summary cut to the link
const cpuNotFinite = cpuOf({ result: { ...aCpu, rows: [{ ...aCpu.rows[0], finite: false }, ...aCpu.rows.slice(1)] } });
const cutShort = { ...aCpu, ceilings: { ...aCpu.ceilings, dot: { error: "x", cutShort: true } } };
assert.equal(cpuNotFinite.status, "wrong");
assert.deepEqual(cpuNotFinite.said, ["**The logits were not finite numbers: the forward pass computed something wrong.**"]);
assert.ok(cpuNotFinite.markdown.endsWith("\n\n**The logits were not finite numbers: the forward pass computed something wrong.**"), cpuNotFinite.markdown);
assert.deepEqual(cpuOf({ result: cutShort }).said, ["**A ceiling's loop did not do its work (its checksum or its time): the compiler cut it short, and its number is no ceiling.**"]);
assert.deepEqual(warnings([{ title: "CPU", ...cpuNotFinite }]), ["CPU: **The logits were not finite numbers: the forward pass computed something wrong.**"]);
const cpuRight = cpuOf({ result: aCpu });
assert.deepEqual([cpuRight.status, cpuRight.said, cpuRight.markdown, cpuRight.data], ["ok", [], cpuTable(aCpu).join("\n"), aCpu]);
assert.deepEqual(cpuOf({ result: { none: "no kernels" } }), { status: "none", markdown: "no kernels" });
assert.deepEqual(cpuOf({ error: "x" }), { status: "error", markdown: "x" });
// the GPU section: its Markdown is the tables' (the GPU ÷ CPU from the CPU section's result), and what it says went wrong
const gpuAdapter = { name: "the adapter", result: { ...aFound.info.result, worker: true, maxStorageBufferBindingSize: 2 ** 30, maxBufferSize: 2 ** 31, maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256 } };
const gpuFound = { ...aFound, info: gpuAdapter };
const noBridge = { error: "not measured: the page is not cross-origin isolated" };
const gpuRight = gpuResult(gpuFound, noBridge, undefined, cpuRight);
assert.equal(gpuRight.status, "ok");
assert.deepEqual(gpuRight.data, { found: gpuFound, bridge: noBridge, lost: undefined });
assert.deepEqual(gpuRight.said, []);
const gpuLines = gpuRight.markdown.split("\n");
assert.equal(gpuLines[0], "**Adapter**: arm · valhall; max binding 1024 MiB, max buffer 2048 MiB, workgroup memory 32 KiB, 256 threads a workgroup, packed int8 dot yes, shader-f16 yes, subgroups yes (16 wide)");
for (const table of [matVecTable(aBandwidths, aCheck, aCeilings, {}), tokenTable(aTokens, aBaseline, { fallback: false, check: aCheck }),
  layerTable(aLayer, aCheck, aCeilings, { fallback: false, check: aCheck }), generateTable(aGenerate, aCheck, { fallback: false, check: aCheck })]) {
  assert.ok(gpuRight.markdown.includes(table.join("\n")), table[0]);
}
assert.equal(gpuLines.at(-1), `**Bridge** (Atomics.wait ↔ Atomics.waitAsync, ${BRIDGE_ROUNDS} round trips): not measured: the page is not cross-origin isolated`);
assert.ok(gpuResult(gpuFound, { microseconds: 12.345 }, undefined, cpuRight).markdown.endsWith("2000 round trips): 12.3 µs each"));
assert.ok(gpuRight.markdown.includes("the CPU, 16 tokens at once: 7.90 ms with 1 (31 GOPS), 4.60 ms with 2 (53 GOPS), 3.90 ms with 4 (62 GOPS), 4.40 ms with 8 (55 GOPS)"), "the prompt's CPU is the CPU section's rows");
assert.ok(gpuResult(gpuFound, noBridge, undefined, undefined).markdown.includes("the CPU, 16 tokens at once: run the CPU section for it. "), "no CPU section: none");
assert.ok(gpuResult(gpuFound, noBridge, undefined, undefined).markdown.includes("GPU ÷ CPU: not measured (run the CPU section for it)"), "no CPU section: no ratio");
// a verdict not ok is the section's WRONG and its own warning (the verdicts that are not ok, said); a check that failed, or a
// verdict that could not be checked, or a lost device, fail it
const wrongCheck = { ...aCheck, "ORT DP4A 64×64": { ok: false, worstRelative: 1 } };
const gpuWrong = gpuResult(checkedBy(gpuFound, wrongCheck), noBridge, undefined, cpuRight);
assert.equal(gpuWrong.status, "wrong");
assert.deepEqual(gpuWrong.said, [checkVerdict(["ORT DP4A 64×64", wrongCheck["ORT DP4A 64×64"]])]);
assert.deepEqual(warnings([{ title: "GPU", ...gpuWrong }]).slice(0, 1), [`GPU: ${checkVerdict(["ORT DP4A 64×64", wrongCheck["ORT DP4A 64×64"]])}`]);
const unchecked = gpuResult({ info: gpuAdapter, check: { name: "the shaders against JavaScript", error: "lost it" }, bandwidth: [], token: [] }, noBridge, undefined, cpuRight);
assert.equal(unchecked.status, "error");
assert.deepEqual(unchecked.said, ["**Shaders against JavaScript**: not checked (lost it)"]);
assert.equal(gpuResult(checkedBy(gpuFound, { ...aCheck, x: { ok: false, worstRelative: 0, error: "no tile" } }), noBridge, undefined, cpuRight).status, "error");
const gpuLost = gpuResult(gpuFound, noBridge, "the device went away: x", cpuRight);
assert.equal(gpuLost.status, "error");
assert.deepEqual(gpuLost.said, ["**The device was lost** (the device went away: x): the times measured after it are not the GPU's."]);
// written again after the CPU section: the same Markdown and said as the section's own, with that CPU
assert.deepEqual(gpuMarkdown(gpuFound, noBridge, undefined, cpuRight), { markdown: gpuRight.markdown, said: gpuRight.said });
assert.notEqual(gpuMarkdown(gpuFound, noBridge, undefined, undefined).markdown, gpuRight.markdown);
// no GPU here: why, in the words the page says
assert.deepEqual(gpuNone({ name: "the adapter", result: { worker: false } }), { status: "none", markdown: "no WebGPU in a worker here" });
assert.deepEqual(gpuNone({ name: "the adapter", result: { worker: true } }), { status: "none", markdown: "navigator.gpu gave no adapter" });
assert.deepEqual(gpuNone({ name: "the adapter", error: "boom" }), { status: "none", markdown: "boom" });
assert.equal(gpuNone(gpuAdapter), undefined);
// the tables of the GPU's besides, ceilings and prompt, each not measured where its step failed or did not run
assert.deepEqual(overheadMarkdown({ name: "x", error: "a | b" }), ["**Besides the weights**: failed: a | b"]);
assert.deepEqual(ceilingsMarkdown(undefined), ["**The device's ceilings**: not measured"]);
assert.deepEqual(promptMarkdown({ name: "x", error: "e" }), ["**A prompt all at once**: failed: e"]);
assert.deepEqual(overheadMarkdown({ name: "o", result: { dispatches: 120, emptyDispatches: 1.234, submitOnly: 0.0123, submitAndWait: 2.5, readToken: 0.4, readLogits: 3.3, vocab: 99584 } }),
  ["| besides the weights | GPU ms |", "|---|---:|", "| 120 dispatches that do nothing, one submission | 1.23 |", "| one submission, not waited for | 0.012 |",
   "| one submission, waited for | 2.50 |", "| the token's id read back (4 bytes) | 0.40 |", "| every logit read back (99584 floats) | 3.30 |"]);
assert.deepEqual(ceilingsMarkdown({ name: "c", result: { ...aCeilings, fallback: false, f16: { unsteady: true, GFLOPS: 1530, shape: "4×4" }, shared: { GBps: 0.512 }, dot4: { error: "a | b" } } }).slice(4),
  ["| f32 multiply-adds | 812 GFLOPS |", "| f16 multiply-adds | unsteady: 1530 GFLOPS (4×4) |", "| int8 dots (dot4I8Packed) | failed: a \\| b |",
   "| reading the workgroup's memory (16-byte reads) | 0.512 GB/s |", "| reading a buffer (128 MiB) | 48.3 GB/s |"]);
const promptLines = promptMarkdown(aPrompt, { ...aCheck, "ORT DP4A 64×64": { ok: false, worstRelative: 1 } }, aCeilings, aBaseline, {}, cpuRight);
assert.ok(promptLines.some((line) => line.startsWith("| ORT DP4A 64×64 (WRONG in the check) | 16 |") && line.endsWith("|  |")), "a WRONG shader has no ratio");
assert.ok(!/ORT DP4A 64×64, \d/.test(promptLines.at(-1)) && promptLines.at(-1).includes("ORT DP4A 64×64, subgroups, ") && promptLines.at(-1).startsWith("**Fastest on the GPU**"), promptLines.at(-1));
assert.ok(promptLines.some((line) => line.startsWith("| batched (T135) | 16 |") && / of (f32|reading) \| [\d.]+× \|$/.test(line)), "the ceiling's share and the ratio");
assert.ok(promptMarkdown(aPrompt, aCheck, aCeilings, aBaseline, { lost: "x" }, cpuRight)[0].includes("GPU ÷ CPU: none: the device was lost"));
// the whole report (reportOf()): the model's table on top, the warnings of the head and the sections, a section with its
// state where it did not measure, and the summary in the link where the whole is too long (T185, T192)
const aResults = { model: { status: "ok", markdown: "THE MODEL SECTION'S OWN TABLE" }, device: deviceResult({ result: aDevice }), cpu: cpuNotFinite, gpu: gpuWrong, storage: storageResult({ error: refusal }), line: lineResult({ result: aLine }) };
const aMeasured = { name: "tiny-lm 29M", rows, pyodide: "314.0.7", paths: pathTable(real, "tiny-lm 29M"), said: pathWarnings(real) };
const theReport = reportOf({ results: aResults, measured: aMeasured, environment: android, hidden: "" });
assert.ok(theReport.markdown.startsWith(`${benchMarkdown(rows, android)}\n\n${aMeasured.paths}\n\n`), theReport.markdown.slice(0, 400));
const wholeWarned = warnings([{ title: "Model", markdown: [benchMarkdown(rows, android), aMeasured.paths].join("\n\n"), said: aMeasured.said },
  ...["device", "cpu", "gpu", "storage", "line"].map((name) => ({ title: TITLES[name], ...aResults[name] }))]);
assert.ok(wholeWarned.length >= 2 && theReport.markdown.includes(warningsBlock(wholeWarned)), "the warnings of the head and the sections");
assert.ok(theReport.markdown.includes(`#### This browser\n\n${aResults.device.markdown}`) && theReport.markdown.includes(`#### CPU (WRONG)\n\n${cpuNotFinite.markdown}`)
  && theReport.markdown.includes(`#### Storage (not in this browser)\n\n`) && !theReport.markdown.includes("#### Model") && !theReport.markdown.includes("THE MODEL SECTION"),
  "a section's title and state; the model section that measured is the table on top, not a section");
assert.ok(theReport.markdown.indexOf("#### This browser") < theReport.markdown.indexOf("#### CPU") && theReport.markdown.indexOf("#### CPU") < theReport.markdown.indexOf("#### GPU")
  && theReport.markdown.indexOf("#### GPU") < theReport.markdown.indexOf("#### Storage") && theReport.markdown.indexOf("#### Storage") < theReport.markdown.indexOf("#### Line"), "the sections in the page's order");
assert.equal(theReport.summary, shortReport([benchMarkdown(rows, android), aMeasured.paths].join("\n\n"), [...deviceSummary(aDevice), "CPU: WRONG", ...cpuSummary(cpuNotFinite.data),
  "GPU: WRONG", ...gpuSummary(gpuWrong.data.found, cpuBaseline(cpuNotFinite)), `Storage: not in this browser: ${tableCell(aResults.storage.markdown).slice(0, 200)}`, ...lineSummary(aLine)], wholeWarned, android));
assert.equal(theReport.href, reportUrl(theReport.markdown, android, theReport.summary));
assert.deepEqual([theReport.long, theReport.longer], [reportTooLong(theReport.markdown, android), reportTooLong(theReport.markdown, android) && reportTooLong(theReport.summary, android)]);
assert.ok(theReport.long && !theReport.longer, "a GPU's report is too long, its summary is not");
// the seconds the page was hidden, right under the machine; a model section that did not measure is a section of its own
const hiddenWhole = reportOf({ results: { model: { status: "error", markdown: "stopped: nothing | came\nsecond" } }, environment: android, hidden: "**hidden for 3 s** x" });
assert.ok(hiddenWhole.markdown.startsWith(`${benchMarkdown([], android)}\n\n**hidden for 3 s** x\n\n`), hiddenWhole.markdown);
assert.ok(hiddenWhole.markdown.endsWith("#### Model (failed)\n\nstopped: nothing | came\nsecond"), hiddenWhole.markdown);
assert.ok(hiddenWhole.summary.includes("- Model: failed: stopped: nothing \\| came"), hiddenWhole.summary);
assert.deepEqual([hiddenWhole.long, hiddenWhole.longer], [false, false]);
assert.deepEqual(reportOf({ results: {}, environment: android, hidden: "" }).markdown, benchMarkdown([], android));
// longer is "the summary is too long as well" and only where the whole is too long (the `long &&` of reportOf()): a short
// report just under the limit has a summary just over it (the summary is the head, a line a section and the request to
// paste), and the page must not say its summary is too long while the link holds the whole report. The head is padded
// by the model page's path until the whole fits and its summary does not.
const padded = (pad: number) => reportOf({ results: { gpu: { status: "none", markdown: "no WebGPU in a worker here" } }, environment: android, hidden: "",
  measured: { name: "tiny-lm 29M", rows, pyodide: "314.0.7", paths: "x".repeat(pad) } });
const edge = Array.from({ length: 4000 }, (_, i) => 4000 + i).filter((pad) => { const w = padded(pad); return !reportTooLong(w.markdown, android) && reportTooLong(w.summary, android); });
assert.ok(edge.length > 10, "there is a report that fits the link with a summary that does not");
for (const pad of edge) assert.deepEqual([padded(pad).long, padded(pad).longer], [false, false], `a report that fits (padded by ${pad})`);
assert.deepEqual(SECTIONS, ["device", "cpu", "model", "gpu", "storage", "line", "memory"]);
// T643.16: the sections' elements are written by hand in benchmark.astro, and nothing else says they match SECTIONS and
// TITLES: run.ts makes a section's parts from each <section data-section> (a name of SECTIONS with no element is a
// section whose parts[name] is missing when it runs), the page's progress reads the element's <h2>, and the report reads
// TITLES. So the markup's sections are SECTIONS in their order, each with its <h2> written as TITLES says.
{
  const markup = fs.readFileSync(new URL("../../src/pages/benchmark.astro", import.meta.url), "utf8");
  const opened = [...markup.matchAll(/<section\b[^>]*>/g)].map(([tag]) => tag);
  const sections = [...markup.matchAll(/<section data-section="([^"]*)">\s*<h2>([^<]*)<\/h2>/g)].map(([, name, title]) => [name, title]);
  assert.equal(sections.length, opened.length, `benchmark.astro: every <section> is <section data-section="…"> with its <h2> first (${opened.join(" ")})`);
  assert.deepEqual(sections.map(([name]) => name), [...SECTIONS], "benchmark.astro's sections are SECTIONS, in their order");
  assert.deepEqual(sections.map(([, title]) => title), SECTIONS.map((name) => TITLES[name]), "benchmark.astro's <h2> of each section is its TITLES");
}
assert.deepEqual(STATE, { ok: "done", none: "not in this browser", wrong: "WRONG", error: "failed" });
// the page's wiring, which only a browser runs (and no runner makes anything WRONG: the --wrong of tests/bench-check.ts, a
// step of preview.yml, does): each section hands its worker's answer to its table, the GPU section is written again
// after the CPU section, and the report is reportOf()'s
const benchmarkPage = pageText("benchmark.astro");
for (const [what, pattern] of [
  ["the sections hand their answers to src/bench/'s tables", /return deviceResult\(await ask\(/],
  ["", /return cpuResult\(await ask\(/],
  ["", /return storageResult\(await ask\(/],
  ["", /return lineResult\(await ask\(/],
  ["the GPU section's steps go to gpuResult() with the CPU section's result", /return gpuResult\(found, bridge, lost, results\.cpu\);/],
  ["and are written again after the CPU section", /Object\.assign\(g, gpuMarkdown\(g\.data\.found, g\.data\.bridge, g\.data\.lost, results\.cpu\)\);/],
  ["the model section hands over what its path says", /said: pathWarnings\(paths\)/],
  ["the report is reportOf()'s", /reportOf\(\{ results, measured, environment, hidden: hiddenLine\(awake\.hidden\(\)\) \}\)/],
  // T242's review: the worker makes its shared memory for the loads it is told follow, without a gigabyte for a next model
  // (the Windows WebKit of bench.yml went down in that gigabyte, one run in four), and only this page's init says them
  ["the model section tells its worker the rounds that follow", /type: "init"[^}]*\}[^;]*ahead: rounds\.map\(\(round\) => round\.without\)/],
]) assert.ok(pattern.test(benchmarkPage), `benchmark.astro: ${what || "and the others"}`);
// ... and the model page, whose visitor may choose a next model, tells none: its memory keeps that gigabyte (T96)
assert.ok(/type: "init"/.test(pageText("index.astro")) && !/\bahead\b/.test(pageText("index.astro")), "index.astro tells its worker no loads ahead (T242)");
// T367.3's review: the worker builds the URLs of models/ from the root the page hands it (state.root, set by the first init
// or load that carries one): an init without it is a worker whose every model's parts are a TypeError. Every message that
// starts a worker of the model (both pages' inits, and the loads) names it.
for (const [page, text] of [["index.astro", pageText("index.astro")], ["benchmark.astro", benchmarkPage]]) {
  const starts = [...text.matchAll(/type: "(init|load)"[^}]*?\}/g)];
  assert.ok(starts.length > 0, `${page} starts a worker with an init or a load`);
  for (const [message, kind] of starts) assert.ok(/\broot\b/.test(message), `${page}: its ${kind} message hands the worker the site's root (models/ is under it)`);
}
// the CPU section's ceilings that could not be measured at all say so in that word, as the GPU section's steps do (T227's own
// change left this one at "Not measured:")
const noCeilings = cpuTable({ ...aCpu, ceilings: { error: "out of memory\nsecond | line" } });
assert.equal(noCeilings.at(-1), "failed: out of memory second \\| line");
assert.deepEqual(warnings([{ title: "CPU", status: "ok", markdown: noCeilings.join("\n") }]), ["CPU: failed: out of memory second \\| line"]);
