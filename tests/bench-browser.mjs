// ?bench=1 in a real browser (T45): the rows must be there, and the Markdown must be a table. Meant for CI:
// the development machine has not the memory to run browsers any more.
//
//   node tests/bench-browser.mjs [model id] [url of a deployed site]
import assert from "node:assert/strict";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import * as playwright from "playwright-core";

const [model = "stories260K", deployed] = process.argv.slice(2);
const base = "/pyodide-llm/";
const root = new URL("../dist/", import.meta.url).pathname;
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".py": "text/plain" };
let server, site = deployed;
if (!site) {
  server = http.createServer((req, res) => {
    let file = path.join(root, decodeURIComponent(req.url.split("?")[0]).replace(base, ""));
    if (file.endsWith("/")) file += "index.html";
    fs.readFile(file, (error, data) => {
      if (error) {
        res.writeHead(404);
        res.end();
        return;
      }
      res.writeHead(200, { "content-type": types[path.extname(file)] ?? "application/octet-stream" });
      res.end(data);
    });
  }).listen(0);
  site = `http://127.0.0.1:${server.address().port}${base}`;
}

const browser = await playwright.chromium.launch();
const page = await browser.newPage();
page.on("pageerror", (error) => console.log("page error:", error.message));
// every line of the console: a worker error comes with its stack (T96), and the rounds say where they are
page.on("console", (message) => console.log(`[${message.type()}] ${message.text()}`));
// and the status line as it changes, to see where a run that never ends got to
let lastStatus = "";
const watch = setInterval(async () => {
  const status = await page.evaluate(() => document.getElementById("status-text")?.textContent ?? "").catch(() => "");
  if (status !== lastStatus) console.log(`[status] ${status}`);
  lastStatus = status;
}, 2000);
await page.goto(`${site}?model=${model}&bench=1`, { waitUntil: "commit" });
// T93: the first visit reloads once under the service worker (coi.js); a wait the reload interrupts starts again
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
assert.equal(bench.rows.length, 2, "with and without the kernels");
for (const row of bench.rows) {
  assert.ok(row.speed > 0, `${row.name} measured nothing`);
  assert.ok(row.backend, `${row.name} did not say what ran`);
}
const [fast, slow] = bench.rows;
assert.ok(fast.speed > slow.speed, `the kernels (${fast.speed.toFixed(1)}) must beat NumPy (${slow.speed.toFixed(1)})`);
assert.ok(bench.markdown.includes("|---|---|---|---|"), "a table GitHub renders");
// T91: the link under the table opens the issue template, with the page's Markdown as the end of the body
const report = new URL(await page.evaluate(() => [...document.querySelectorAll(".bench-actions a")].at(-1)?.href ?? "about:blank"));
assert.equal(report.searchParams.get("template"), "benchmark.md", `the report link: ${report}`);
assert.ok(report.searchParams.get("body")?.endsWith(bench.markdown), "the Markdown in the issue's body");
clearInterval(watch);
// T141: end here, as tests/e2e.mjs does: what the browser leaves behind must not keep the job waiting
await Promise.race([browser.close(), new Promise((resolve) => setTimeout(resolve, 15000))]);
server?.close();
process.stdout.write("ok\n", () => process.exit(0));
