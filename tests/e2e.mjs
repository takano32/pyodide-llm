// End-to-end check in a real browser: serves dist/ under the GitHub Pages base path, waits until the model is
// ready, runs the default prompt, and checks that text streams in and that the page itself never scrolls.
//
//   npm run build && node tests/e2e.mjs [model id] [chromium|firefox|webkit|msedge|chrome] [url of a deployed site]
//
// msedge and chrome are the browsers installed on the machine (Playwright calls them channels of chromium).
//
// The model id "local" opens stories260K.bin and tok512.bin of this directory through the folder button instead,
// as a visitor would open a model of their own disk. "hf" does the same with the files Hugging Face would publish
// for that model (tests/make_hf_fixture.py writes them), which the page converts in the browser. "url" reads the same
// model from huggingface.co by ?checkpoint=&tokenizer=, and an id that begins with hf- is fetched from there and
// converted (hundreds of megabytes, or gigabytes).
//
// Needs a browser for playwright-core (a devDependency): `npx playwright-core install chromium` once.
// Mind the memory: a browser with Pyodide and a model takes 400 MB and more; keep `free -m` above 1 GB available.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execFileSync } from "node:child_process";
import * as playwright from "playwright-core";

const [model = "stories260K", engine = "chromium", deployed] = process.argv.slice(2);
const base = "/pyodide-llm/";
const root = new URL("../dist/", import.meta.url).pathname;
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".py": "text/plain" };
// expected beginning of the greedy text, where the model is deterministic
const expected = {
  stories260K: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park.",
  local: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park.",
  hf: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park.",
  url: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park.",
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

const channel = ["msedge", "chrome"].includes(engine) ? engine : undefined;
const browser = await playwright[channel ? "chromium" : engine].launch({ headless: true, channel });
const browserVersion = browser.version();
const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
const errors = [];
page.on("pageerror", (error) => errors.push(String(error)));
page.on("console", (message) => message.type() === "error" && errors.push(message.text()));

const started = Date.now();
const opens = model === "local" || model === "hf";
// the page asks before it fetches more than 500 MB
page.on("dialog", (dialog) => dialog.accept());
const tinyllamas = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K";
const query = model === "url" ? `checkpoint=${encodeURIComponent(`${tinyllamas}/stories260K.bin`)}&tokenizer=${encodeURIComponent(`${tinyllamas}/tok512.bin`)}`
  : `model=${opens ? "stories3_5M" : model}`;
await page.goto(`${url}?${query}`);
const idle = () => page.waitForFunction(() => !document.getElementById("run").disabled || document.querySelector(".error"), null, { timeout: 1800000 });
await idle();
if (opens) {
  const repository = new URL("../", import.meta.url).pathname;
  let chosen = [`${repository}stories260K.bin`, `${repository}tok512.bin`];
  if (model === "hf") {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "hf-fixture-"));
    execFileSync("python3", [`${repository}tests/make_hf_fixture.py`, directory]);
    chosen = fs.readdirSync(directory).map((name) => path.join(directory, name));
  }
  await page.setInputFiles("#files", chosen);
  await page.waitForFunction(() => /^stories260K(\.bin| from Hugging Face files) ·|^Could not/.test(document.getElementById("status-text").textContent), null, { timeout: 600000 });
  await idle();
}
const readySeconds = (Date.now() - started) / 1000;
// By default a model writes until its context is full (4096 tokens for llm-jp-3-150m). The test, and every tok/s in
// the documents, is about 256 tokens: set that in the settings, as a visitor would.
await page.evaluate(() => {
  const steps = document.getElementById("steps");
  steps.value = "256";
  steps.dispatchEvent(new Event("input"));
});
// Enter alone breaks the line
await page.press("#prompt", "Control+Enter");
await page.waitForFunction(() => document.querySelector(".model .meta") || document.querySelector(".error"), null, { timeout: 600000 });
const result = await page.evaluate(() => ({
  text: document.querySelector(".model .bubble")?.textContent ?? "",
  // the closed line; the breakdown below it is in the same element
  meta: document.querySelector(".model .meta summary")?.textContent ?? "",
  error: document.querySelector(".error .bubble")?.textContent ?? "",
  // which kernels this browser got: "SIMD kernels, int8, relaxed SIMD", or less
  status: document.getElementById("status-text")?.textContent ?? "",
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
console.log(`${engine} ${browserVersion}, ${model}: ready in ${readySeconds.toFixed(1)}s, ${result.meta}`);
console.log(`status: ${result.status}`);
console.log(result.text.slice(0, 160).replace(/\n/g, " / "));
if (failures.length) {
  console.error("FAILED\n- " + failures.join("\n- "));
  process.exit(1);
}
console.log("ok");
