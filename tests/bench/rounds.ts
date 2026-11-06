// tests/bench.ts's part (T643.13): the rounds' table, the report a visitor pastes and its link (src/bench/rounds.ts, report.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { FULL_ROUNDS, MEMORY_UNSAID, QUESTIONS, REPORT_LIMIT, ROUNDS, TOO_LONG, benchMarkdown, environmentOf, loginUrl, parseReport, reportBody,
         reportTooLong, reportUrl, reportsTable, roundsHere, roundsTable, warnings, drift, DRIFT_LINE } from "../../src/bench.ts";
import fs from "node:fs";
import { runtimeUrl } from "../tree.ts";

const rows = [
  { name: "everything", without: [], tokens: 64, speed: 334.62, seconds: 8.4, backend: "SIMD kernels, int8, relaxed SIMD" },
  { name: "without the kernels", without: ["kernels"], tokens: 64, speed: 44.81, seconds: 8.1, backend: "NumPy (without kernels)" },
];
const environment = environmentOf({ hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (X11)" },
                                  { model: "tiny-lm 29M", pyodide: "314.0.7", build: "abc1234", site: "https://example.invalid/" });
const markdown = benchMarkdown(rows, environment);

// a table GitHub renders: a header, the separator, and one row per round
const lines = markdown.split("\n");
assert.equal(lines.filter((line) => line.startsWith("|")).length, rows.length + 2, "one row per round, and the header");
assert.ok(lines.some((line) => line.includes("|---|---|---|---|")), "the separator GitHub needs");
assert.ok(markdown.includes("| everything | 334.6 | 8.4 s | SIMD kernels, int8, relaxed SIMD |"), markdown);
assert.ok(markdown.includes("8 logical cores") && markdown.includes("8 GB or more"), "what the browser told us");
assert.ok(markdown.includes("Mozilla/5.0 (X11)"), "the user agent, for a report that means something");
assert.ok(markdown.includes("Pyodide 314.0.7 · site abc1234"), "the site's version (T176), for which shaders and kernels ran");

// a browser that says nothing about itself must not make anything up
const bare = benchMarkdown(rows, environmentOf({}, {}));
assert.ok(!bare.includes("site "), "no version, no word of it");
assert.ok(!bare.includes("undefined") && !bare.includes("NaN"), bare);
assert.ok(!bare.includes("cores"), "no invented core count");

// a round that failed to measure leaves a question mark, never a wrong number
const broken = benchMarkdown([{ name: "everything", speed: undefined, backend: "" }], environmentOf({}, {}));
assert.ok(broken.includes("| everything | ? |"), broken);

assert.deepEqual(ROUNDS.map((round) => [round.name, round.without.join()]), [["everything", ""], ["everything, before", ""], ["without the kernels", "kernels"], ["everything, after", ""]],
  "?bench=1 runs with and without the kernels, and the first twice more (T555)");
// T555: the last round against the first, a warning below DRIFT_LINE; none where the two are not the same switches,
// or a speed is missing
const rowsOf = (before, after) => [{ name: "everything", without: [], speed: 1 }, { name: "everything, before", without: [], speed: before },
  { name: "without the kernels", without: ["kernels"], speed: 9 }, { name: "everything, after", without: [], speed: after }];
assert.equal(drift(rowsOf(100, 97)), "The last round ran the first one's switches again at 97% of the speed of the second.");
assert.equal(drift(rowsOf(100, 80)), "The last round ran the first one's switches again at 80% of the speed of the second: unsteady, the device was slower after the rounds between them (heat, or other work).");
assert.equal(DRIFT_LINE, 0.9);
assert.equal(drift(rowsOf(100, undefined)), "");
assert.equal(drift(rowsOf(100, 97).slice(0, 3)), "", "the last is not the first's switches");
assert.equal(drift([{ name: "a", without: [], speed: 5 }]), "");
assert.deepEqual(warnings([{ title: "Model", markdown: drift(rowsOf(100, 80)) }]), ["Model: The last round ran the first one's switches again at 80% of the speed of the second: unsteady, the device was slower after the rounds between them (heat, or other work)."]);
assert.deepEqual(warnings([{ title: "Model", markdown: drift(rowsOf(100, 97)) }]), []);
assert.ok(benchMarkdown(rowsOf(100, 80), environmentOf({}, {})).includes("\n\nThe last round ran the first one's switches again at 80%"), "?bench='s report says it under its table");
assert.equal(FULL_ROUNDS.length, 6, "?bench=full walks the steps of T52, and T110's");
assert.deepEqual(FULL_ROUNDS.at(-1).without, [], "the last step is everything switched on");
// T214: a browser that does not say its memory (Safari, Firefox: no navigator.deviceMemory) skips the round without the
// kernels (NumPy's float32 weights took an iPhone's tab down, T205's review), and its row says why; one that says runs all
const quiet = roundsHere(ROUNDS, undefined);
assert.deepEqual(quiet.map((round) => round.name), ROUNDS.map((round) => round.name), "every round keeps its row");
assert.deepEqual(quiet.map((round) => round.skip), [undefined, undefined, MEMORY_UNSAID, undefined], "only the round without the kernels is skipped");
assert.equal(ROUNDS[1].skip, undefined, "ROUNDS itself is left as it is");
for (const memory of [8, 4, 0.5]) assert.deepEqual(roundsHere(ROUNDS, memory), ROUNDS, `a browser that says ${memory} GB runs every round`);
assert.deepEqual(roundsHere(FULL_ROUNDS, undefined).filter((round) => round.skip).map((round) => round.name), ["NumPy only", "the kernels, int8 widened"], "?bench=full: the two rounds that widen the weights");
assert.ok(/does not say how much memory/.test(MEMORY_UNSAID) && /float32/.test(MEMORY_UNSAID) && !/NumPy/.test(MEMORY_UNSAID), MEMORY_UNSAID);
// the worker's row for it (src/runtime/worker.js: nothing loaded, the reason carried), in the report and back out of it
const iphoneRows = [rows[0], { name: "without the kernels", without: ["kernels"], skip: MEMORY_UNSAID }];
const iphone = benchMarkdown(iphoneRows, environmentOf({ hardwareConcurrency: 6, userAgent: "Mozilla/5.0 (iPhone)" }, { model: "llm-jp-3 150M" }));
const skippedLine = iphone.split("\n").find((line) => line.startsWith("| without the kernels |"));
assert.equal(skippedLine, `| without the kernels | skipped |  | ${MEMORY_UNSAID} |`, iphone);
assert.ok(!iphone.includes("NaN") && !iphone.includes("undefined") && !skippedLine.includes("?"), iphone);
assert.deepEqual(roundsTable(iphoneRows), iphone.split("\n").filter((line) => line.startsWith("|")), "the model section writes the same table");
const iphoneReport = parseReport(reportBody(iphone));
assert.deepEqual(iphoneReport.rows.map((row) => row.name), ["everything", "without the kernels"]);
assert.equal(iphoneReport.rows[1].skip, MEMORY_UNSAID, "parseReport() reads why");
assert.ok(Number.isNaN(iphoneReport.rows[1].speed) && iphoneReport.rows[0].skip === undefined);
assert.equal(reportsTable([{ number: 11, url: "u", body: reportBody(iphone) }]).split("\n")[2],
  "| ? | ? | ? | llm-jp-3 150M | 6 | 334.6 | skipped | SIMD kernels, int8, relaxed SIMD | [#11](u) |");
// a reason with a | in it stays one cell
assert.equal(parseReport(benchMarkdown([{ name: "x", skip: "a | b" }], environmentOf({}, {}))).rows[0].skip, "a | b");
// both pages ask for the rounds through roundsHere() with the browser's own deviceMemory, and the worker writes the row
// (T355: the model page's script is index.astro's and the modules of src/page/: a page's text is all of them; T353:
// and /benchmark/'s is benchmark.astro's and the modules of src/benchmark/)
const PAGE_MODULES = { "index.astro": "page", "benchmark.astro": "benchmark" };
const pageText = (page) => [new URL(`../../src/pages/${page}`, import.meta.url),
  ...fs.readdirSync(new URL(`../../src/${PAGE_MODULES[page]}/`, import.meta.url)).sort().map((file) => new URL(`../../src/${PAGE_MODULES[page]}/${file}`, import.meta.url))]
  .map((url) => fs.readFileSync(url, "utf8")).join("\n");
assert.ok(pageText("index.astro").length > fs.readFileSync(new URL("../../src/pages/index.astro", import.meta.url), "utf8").length + 40000, "src/page/ has the model page's script");
assert.ok(pageText("benchmark.astro").length > fs.readFileSync(new URL("../../src/pages/benchmark.astro", import.meta.url), "utf8").length + 40000, "src/benchmark/ has /benchmark/'s script");
for (const page of ["index.astro", "benchmark.astro"]) {
  assert.ok(/roundsHere\(.*\(navigator as [^)]*\)\.deviceMemory\)/.test(pageText(page)),
    `${page} asks for the rounds through roundsHere()`);
}
assert.ok(/if \(round\.skip !== undefined\) \{\s*rows\.push\(\{ name: round\.name, without: round\.without, skip: round\.skip \}\);\s*continue;/
  .test(fs.readFileSync(runtimeUrl("worker.js"), "utf8")), "the worker writes a skipped round's row and loads nothing");
// T91: the issue the page opens, and the table the issues make
const url = new URL(reportUrl(markdown, environment));
assert.equal(url.searchParams.get("template"), "benchmark.md");
assert.equal(url.searchParams.get("title"), "Benchmark: tiny-lm 29M");
assert.ok(url.searchParams.get("body").endsWith(markdown), "the page's Markdown, unchanged, at the end");
// the questions of the page are the template's, word for word (the parser reads what either of them wrote)
const template = fs.readFileSync(new URL("../../.github/ISSUE_TEMPLATE/benchmark.md", import.meta.url), "utf8");
for (const [name, example] of QUESTIONS) assert.ok(template.includes(`**${name}**: (${example})`), `${name} in the template`);
// a visitor who answered two of the three, and one who changed nothing
const answered = reportBody(markdown).replace(/\*\*Device\*\*: \(.*\)/, "**Device**: Pixel 8")
  .replace(/\*\*Browser\*\*: \(.*\)/, "**Browser**: Chrome 148");
const report = parseReport(answered);
assert.deepEqual([report.device, report.os, report.browser, report.model, report.cores], ["Pixel 8", "", "Chrome 148", "tiny-lm 29M", "8"]);
assert.deepEqual(report.rows.map((row) => row.speed), [334.6, 44.8]);
assert.equal(parseReport("no table here"), undefined);
// a report from before T91 (the Markdown alone, "8 threads") still reads
const old = markdown.replace("8 logical cores", "8 threads");
const table = reportsTable([{ number: 7, url: "https://github.com/x/y/issues/7", body: answered },
                            { number: 8, url: "https://github.com/x/y/issues/8", body: reportBody(old) },
                            { number: 9, url: "https://github.com/x/y/issues/9", body: "Something else entirely" }]);
const tableLines = table.split("\n");
assert.equal(tableLines.length, 4, "a header, the separator, and the two issues that are reports");
assert.equal(tableLines[2], "| Pixel 8 | ? | Chrome 148 | tiny-lm 29M | 8 | 334.6 | 44.8 | SIMD kernels, int8, relaxed SIMD | [#7](https://github.com/x/y/issues/7) |");
assert.ok(tableLines[3].startsWith("| ? | ? | ? | tiny-lm 29M | 8 | 334.6 |"), tableLines[3]);
// T134: /benchmark/ writes the model's table first and its other sections after it, with tables and bold lines of
// their own: the report reads as the model's table alone
const everything = [markdown, "#### This browser\n\n| feature | here |\n|---|---|\n| WebAssembly SIMD | yes |",
  "#### GPU\n\n**Adapter**: apple · metal-3; max binding 2048 MiB\n\n| int8 matrix × vector | GPU |\n|---|---:|\n| Llama 3.2 1B w1 | 51.0 GB/s |",
  // T168: the device's ceilings, and the share of them in the prompt's table (with a % in its cells)
  "**The device's ceilings** (each a loop of that alone):\n\n| ceiling | GPU |\n|---|---:|\n| f32 multiply-adds | 1520.3 GFLOPS |\n" +
  "| f16 multiply-adds | no shader-f16 here |\n| int8 dots (dot4I8Packed) | 2710.0 GOPS |\n| reading the workgroup's memory (16-byte reads) | 310 GB/s |\n| reading a buffer (128 MiB) | 38.2 GB/s |",
  // T146: the prompt's table names its shaders with × and parentheses
  "| shader | tokens at once on the GPU | GPU ms | GPU ms a token | GFLOPS | of the ceiling |\n|---|---:|---:|---:|---:|---:|\n| batched (T135) | 64 | 352.8 | 5.51 | 44 | 2.9% of f32 |\n" +
  "| ORT DP4A 64×64 | 64 | 60.0 | 0.94 | 260 | 9.6% of int8 dots |\n\n**Fastest on the GPU**: at 64 tokens ORT DP4A 64×64, 0.94 ms a token (260 GFLOPS)"].join("\n\n");
const whole = parseReport(reportBody(everything).replace(/\*\*Device\*\*: \(.*\)/, "**Device**: iPhone 15"));
assert.deepEqual([whole.device, whole.model, whole.cores], ["iPhone 15", "tiny-lm 29M", "8"]);
assert.deepEqual(whole.rows.map((row) => row.name), ["everything", "without the kernels"]);
// T134: a report without the model's section has no table and no model, so that reportsTable() leaves it out rather
// than writing a row of question marks under a model that did not run
const deviceOnly = [benchMarkdown([], environmentOf({ hardwareConcurrency: 4, userAgent: "UA" }, {})),
  "#### GPU\n\n| int8 matrix × vector | GPU |\n|---|---:|\n| Llama 3.2 1B w1 | 51.0 GB/s |"].join("\n\n");
assert.ok(!deviceOnly.includes("| what ran |"), deviceOnly);
assert.equal(parseReport(reportBody(deviceOnly)), undefined);
assert.equal(reportsTable([{ number: 10, url: "u", body: reportBody(deviceOnly) }]).split("\n").length, 2, "no row for it");
assert.equal(new URL(reportUrl(deviceOnly, environmentOf({}, {}))).searchParams.get("title"), "Benchmark: this device");
// T134: a report too long for the address goes by the clipboard; the address says so and stays short
assert.ok(!reportTooLong(everything, environment));
const long = [everything, `#### Line\n\n${"| x | y |\n".repeat(600)}`].join("\n\n");
assert.ok(reportTooLong(long, environment));
const longUrl = reportUrl(long, environment);
assert.ok(loginUrl(long, environment).length <= REPORT_LIMIT, `${loginUrl(long, environment).length}`);
// what GitHub's login drops is the address once more encoded: a report of pipes and × (a table's) is too long long
// before its own address is (the second review of T134: 4,935 characters of it were dropped by the login)
for (let rows = 1; rows < 120; rows++) {
  const markdown = `#### GPU\n\n${"| Llama 3.2 1B · 16 tokens | 12.3 ms | 45.6 GB/s | ok × |\n".repeat(rows)}`;
  if (!reportTooLong(markdown, environment)) assert.ok(reportUrl(markdown, environment).length < 4935, `${rows} rows`);
  assert.ok(loginUrl(markdown, environment).length <= REPORT_LIMIT, `${rows} rows`);
}
assert.ok(new URL(longUrl).searchParams.get("body").endsWith(TOO_LONG));
assert.ok(new URL(longUrl).searchParams.get("body").includes("**Device**: ("), "the three questions stay");
export { rows, environment, markdown, iphone, pageText, everything, long };
