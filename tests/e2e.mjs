// End-to-end check in a real browser: serves dist/ under the GitHub Pages base path, waits until the model is
// ready, runs the default prompt, and checks that text streams in and that the page itself never scrolls.
//
//   npm run build && node tests/e2e.mjs [model id] [chromium|firefox] [url of a deployed site]
//
// Needs a browser for playwright-core (a devDependency): `npx playwright-core install chromium` once.
// Mind the memory: a browser with Pyodide and a model takes 400 MB and more; keep `free -m` above 1 GB available.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import * as playwright from "playwright-core";

const [model = "stories260K", engine = "chromium", deployed] = process.argv.slice(2);
const base = "/pyodide-llama-py/";
const root = new URL("../dist/", import.meta.url).pathname;
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".py": "text/plain" };
// expected beginning of the greedy text, where the model is deterministic
const expected = {
  stories260K: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park.",
  "stories15M-f32": "Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine.",
};

let server;
let url = deployed;
if (!url) {
  // plain static files and no special headers, like GitHub Pages
  server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    const file = path.join(root, pathname.slice(base.length) || "index.html");
    if (!pathname.startsWith(base) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404);
      return res.end();
    }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  }).listen(0);
  url = `http://localhost:${server.address().port}${base}`;
}

const browser = await playwright[engine].launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
const errors = [];
page.on("pageerror", (error) => errors.push(String(error)));
page.on("console", (message) => message.type() === "error" && errors.push(message.text()));

const started = Date.now();
await page.goto(`${url}?model=${model}`);
await page.waitForFunction(() => !document.getElementById("run").disabled || document.querySelector(".error"), null, { timeout: 600000 });
const readySeconds = (Date.now() - started) / 1000;
await page.press("#prompt", "Enter");
await page.waitForFunction(() => document.querySelector(".model .meta") || document.querySelector(".error"), null, { timeout: 600000 });
const result = await page.evaluate(() => ({
  text: document.querySelector(".model .bubble")?.textContent ?? "",
  // the closed line; the breakdown below it is in the same element
  meta: document.querySelector(".model .meta summary")?.textContent ?? "",
  error: document.querySelector(".error .bubble")?.textContent ?? "",
  pageScrolls: document.documentElement.scrollHeight > innerHeight,
}));
await browser.close();
server?.close();

const failures = [];
if (result.error) failures.push(`the page reported: ${result.error}`);
if (errors.length) failures.push(`console errors: ${errors.join(" | ")}`);
if (!/tok\/s/.test(result.meta)) failures.push("no speed line under the answer");
if (result.pageScrolls) failures.push("the page itself scrolls");
if (expected[model] && !result.text.startsWith(expected[model])) failures.push(`unexpected text: ${result.text.slice(0, 120)}`);
console.log(`${engine}, ${model}: ready in ${readySeconds.toFixed(1)}s, ${result.meta}`);
console.log(result.text.slice(0, 160).replace(/\n/g, " / "));
if (failures.length) {
  console.error("FAILED\n- " + failures.join("\n- "));
  process.exit(1);
}
console.log("ok");
