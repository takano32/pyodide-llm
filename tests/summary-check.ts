// tests/summary.ts: the table a CI job writes into its summary (T82). Node alone.
//
//   node tests/summary-check.ts
import assert from "node:assert/strict";
import fs from "node:fs";
import { readResults, summaryMarkdown } from "./summary.ts";
import { UNISOLATED, unisolated } from "./unisolated.ts";

const lines = [
  { engine: "chromium", browserVersion: "148.0", model: "tiny-lm", os: "linux x64", ok: true, timedOut: false,
    readySeconds: 8.42, tokPerSecond: 307.2, backend: "tiny-lm 29M · SIMD kernels, int8, relaxed SIMD", failures: [],
    threadReports: 1, then: [{ model: "llm-jp-3-150m", threadReports: 0 }], offline: { threadReports: 2 } },
  { engine: "webkit", browserVersion: "26.4", model: "tiny-lm", os: "linux x64", ok: false, timedOut: true,
    failures: ["timed out after 900s"] },
  { engine: "firefox", browserVersion: "150", model: "llm-jp-3-150m", os: "linux x64", ok: false, timedOut: false,
    readySeconds: 12, tokPerSecond: null, backend: "a | pipe", failures: ["no speed line under the answer"] },
].map((entry) => JSON.stringify(entry));
// a run killed while writing leaves half a line: it is skipped, not fatal
const results = readResults(lines.join("\n") + "\n{\"engine\": \"chro");
assert.equal(results.length, 3);

const table = summaryMarkdown(results);
const rows = table.split("\n").filter((line) => line.startsWith("|"));
assert.equal(rows.length, 2 + 3, "a header, the separator GitHub needs, and a row per run");
assert.ok(table.includes("| tiny-lm | chromium 148.0 | 8.4 | 307.2 | 1, 0, 2 | tiny-lm 29M · SIMD kernels, int8, relaxed SIMD | ok |  |"), table);
// T172: a record from before the count (or a run that never answered) leaves the cell empty, not "undefined"
assert.ok(table.includes("| webkit 26.4 |  |  |  |  | **timed out** |"), table);
assert.ok(table.includes("**timed out**") && table.includes("timed out after 900s"), "a timeout says so, and why");
assert.ok(table.includes("a \\| pipe"), "a | in a cell must not break the table");
assert.ok(!table.includes("undefined") && !table.includes("NaN") && !table.includes("null"), table);
assert.ok(table.includes("1 of 3 ran on linux x64. 2 failed or timed out"), table);
assert.ok(table.includes("stock-firefox.ts writes no line here"), "the table says what it does not count");
assert.equal(summaryMarkdown([]), "No results were recorded.\n");

// T662: a run whose page was not cross-origin isolated fails (tests/unisolated.ts), but where the visit asked for no
// isolation (?coi=off) or the pair of system and browser is known not to isolate
const site = "https://takano32.github.io/pyodide-llm/";
const run = (more) => ({ isolated: false, os: "linux x64", engine: "webkit", visited: `${site}?model=tiny-lm`, ...more });
assert.match(unisolated(run({})), /not cross-origin isolated \(linux x64, webkit\)/, "a page not isolated fails");
assert.equal(unisolated(run({ isolated: true })), null, "an isolated page passes");
assert.equal(unisolated(run({ visited: `${site}?model=tiny-lm&coi=off` })), null, "?coi=off asks for no isolation");
assert.equal(unisolated(run({ visited: `${site}?coi=off&model=tiny-lm` })), null, "?coi=off first in the query");
assert.ok(unisolated(run({ visited: `${site}?model=tiny-lm&coi=on` })), "?coi= anything but off is no excuse");
assert.ok(unisolated(run({ visited: `${site}?model=tiny-lm&nocoi=off` })), "nor a name that ends in coi");
const known = { "linux x64 webkit": "a made-up reason" };
assert.equal(unisolated(run({}), known), null, "a pair the table knows passes");
assert.ok(unisolated(run({ os: "linux arm64" }), known), "the same browser on another system does not");
assert.ok(unisolated(run({ engine: "chromium" }), known), "nor another browser on the same system");
for (const [pair, why] of Object.entries(UNISOLATED)) {
  assert.match(pair, /^(linux|win32|darwin) (x64|arm64) (chromium|firefox|webkit|chrome|msedge|stock-firefox)$/, `UNISOLATED's ${pair}: "<platform> <arch> <browser>"`);
  assert.ok(typeof why === "string" && why.length > 20, `UNISOLATED's ${pair} says why`);
}
// and e2e.ts (which runs a browser as it is imported) asks it of every run, with the URL it opened, into its failures
const e2e = fs.readFileSync(new URL("./e2e.ts", import.meta.url), "utf8");
assert.match(e2e, /const notIsolated = unisolated\(\{ isolated: result\.isolated, os: `\$\{os\.platform\(\)\} \$\{os\.arch\(\)\}`, engine, visited \}\);\nif \(notIsolated\) failures\.push\(notIsolated\);/,
  "tests/e2e.ts asks tests/unisolated.ts of every run and records what it says as a failure");
assert.match(e2e, /await page\.goto\(visited\);/, "tests/e2e.ts hands it the URL it opened");
// and tests/stock-firefox.ts (Selenium: the weekly browsers.yml's Firefox, no line in results.jsonl) asks it as well
const stock = fs.readFileSync(new URL("./stock-firefox.ts", import.meta.url), "utf8");
assert.match(stock, /const notIsolated = unisolated\(\{ isolated: result\.isolated, os: `\$\{os\.platform\(\)\} \$\{os\.arch\(\)\}`, engine: "stock-firefox", visited \}\);\n  if \(notIsolated\) failures\.push\(notIsolated\);/,
  "tests/stock-firefox.ts asks tests/unisolated.ts and records what it says as a failure");
assert.match(stock, /isolated: self\.crossOriginIsolated/, "tests/stock-firefox.ts reads the page's crossOriginIsolated");
console.log("ok");
