// ?bench=1 in a real browser (T.45): the rows must be there, and the Markdown must be a table. On Playwright's Chromium,
// in CI and on the development machine (T.469: stories260K in about a minute there).
//
//   node tests/bench-browser.ts [model id] [url of a deployed site]
import assert from "node:assert/strict";
import * as playwright from "playwright-core";
import { MODELS } from "../src/models.ts";
import { closed, leave } from "./leave.ts";
import { serving } from "./serving.ts";

const [model = "stories260K", deployed] = process.argv.slice(2);
// an id the list does not have would open the default model and pass on it (T.144)
if (!MODELS.some((entry) => entry.id === model)) {
  console.error(`FAILED\n- no model ${model} in src/models.ts`);
  process.exit(1);
}
let server, site = deployed;
if (!site) {
  server = await serving();  // dist/, as tests/e2e.ts serves it (T.391.6)
  site = server.url;
}

const browser = await playwright.chromium.launch();
const page = await browser.newPage();
page.on("pageerror", (error) => console.log("page error:", error.message));
// every line of the console: a worker error comes with its stack (T.96), and the rounds say where they are
page.on("console", (message) => console.log(`[${message.type()}] ${message.text()}`));
// and the status line as it changes, to see where a run that never ends got to
let lastStatus = "";
const watch = setInterval(async () => {
  const status = await page.evaluate(() => document.getElementById("status-text")?.textContent ?? "").catch(() => "");
  if (status !== lastStatus) console.log(`[status] ${status}`);
  lastStatus = status;
}, 2000);
await page.goto(`${site}?model=${model}&bench=1`, { waitUntil: "commit" });
// T.93: the first visit reloads once under the service worker (coi.js); a wait the reload interrupts starts again
for (;;) {
  try {
    await page.waitForFunction(() => window.__bench || document.querySelector(".error"), null, { timeout: 1800000 });
    break;
  } catch (error) {
    if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
  }
}
const failure = await page.evaluate(() => document.querySelector(".error")?.textContent);
assert.equal(failure, undefined, `the page reported: ${failure}`);
const bench = await page.evaluate(() => window.__bench);
console.log(bench.markdown);
assert.deepEqual(bench.rows.map((row) => row.name), ["everything", "everything, before", "without the kernels", "everything, after"], "with and without the kernels, and the first twice more (T.555)");
assert.match(bench.markdown, /\n\nThe last round ran the first one's switches again at \d+% of the speed of the second[.:]/, "the drift under the table (T.555)");
for (const row of bench.rows) {
  assert.ok(row.speed > 0, `${row.name} measured nothing`);
  assert.ok(row.backend, `${row.name} did not say what ran`);
}
const [, fast, slow] = bench.rows;
assert.ok(fast.speed > slow.speed, `the kernels (${fast.speed.toFixed(1)}) must beat NumPy (${slow.speed.toFixed(1)})`);
assert.ok(bench.markdown.includes("|---|---|---|---|"), "a table GitHub renders");
// T.91: the link under the table opens the issue template, with the page's Markdown as the end of the body
const report = new URL(await page.evaluate(() => [...document.querySelectorAll(".bench-actions a")].at(-1)?.href ?? "about:blank"));
assert.equal(report.searchParams.get("template"), "benchmark.md", `the report link: ${report}`);
assert.ok(report.searchParams.get("body")?.endsWith(bench.markdown), "the Markdown in the issue's body");
clearInterval(watch);
// T.141: end here, as tests/e2e.ts does: what the browser leaves behind must not keep the job waiting
await closed(() => browser.close());
server?.close();
console.log("ok");
await leave(0);
