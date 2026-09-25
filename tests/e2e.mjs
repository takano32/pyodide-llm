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
// converted (hundreds of megabytes, or gigabytes). hf:<owner>/<repository>[@<revision>] opens a repository that is in
// no list, by ?hf= (T105: to try a model before it is added).
//
// Needs a browser for playwright-core (a devDependency): `npx playwright-core install chromium` once.
// Mind the memory: a browser with Pyodide and a model takes 400 MB and more; keep `free -m` above 1 GB available.
//
// For CI (T82), three environment variables, all optional:
//   E2E_TIMEOUT    seconds this one run may take (default 900). Past it the run is recorded as timed out and the
//                  process ends, so that one stuck model never takes the rest of the job with it.
//   E2E_RESULTS    a file to append one JSON line to: the browser and its version, the model, the seconds to ready,
//                  tok/s, the backend line, and what failed. tests/summary.mjs turns those lines into one table.
//   E2E_ARTIFACTS  a directory: when the run fails or times out, a screenshot, the DOM and the last lines of the
//                  console go there, named after the browser and the model.
//   E2E_TWICE      when set: after the answer, the page is loaded again in the same browser, and the seconds to
//                  ready then go into the JSON line too (T99: a model converted from Hugging Face must come from what
//                  this browser kept, the origin private file system or the Cache API).
//   E2E_QUERY      more of the page's URL, such as hfParts=16&hfConnections=8 (T107), added to what the model needs
//                  and kept in the JSON line.
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
const started = Date.now();
// The time limit is set before the browser starts: a run that hangs in launch() or newPage() (the Windows WebKit of
// T70 looked like one) must end by it too, not by the job's own limit. Until the page exists there is nothing to
// keep but the reason. (Fable's review of T82.)
let browser, browserVersion = "", page;
const limit = Number(process.env.E2E_TIMEOUT ?? 900) * 1000;
const watchdog = setTimeout(async () => {
  const seconds = limit / 1000;
  record({ ok: false, timedOut: true, failures: [`timed out after ${seconds}s`] });
  if (page) await keepArtifacts(`timed out after ${seconds}s`);
  console.error(`FAILED\n- timed out after ${seconds}s`);
  // stdout may be a pipe (| tee): let it drain before the process ends
  process.stdout.write(`${engine} ${browserVersion}, ${model}: timed out after ${seconds}s\n`, () => process.exit(2));
}, limit);
const kind = playwright[channel ? "chromium" : engine], viewport = { width: 390, height: 844 };
if (process.env.E2E_TWICE) {
  // A profile on disk, as a visitor's browser has: Playwright's usual context is like private browsing, and WebKit
  // kept nothing across a reload there (T99)
  browser = await kind.launchPersistentContext(fs.mkdtempSync(path.join(os.tmpdir(), "e2e-profile-")), { headless: true, channel, viewport });
  browserVersion = browser.browser()?.version() ?? "";
  page = browser.pages()[0] ?? await browser.newPage();
} else {
  browser = await kind.launch({ headless: true, channel });
  browserVersion = browser.version();
  page = await browser.newPage({ viewport });
}
const errors = [];
// every line of the console, for the artifacts of a failed run: the last ones say where it stopped
const consoleLines = [];
const remember = (line) => {
  consoleLines.push(`${((Date.now() - started) / 1000).toFixed(1)}s ${line}`);
  if (consoleLines.length > 200) consoleLines.shift();
};
page.on("pageerror", (error) => {
  errors.push(String(error));
  remember(`[pageerror] ${error}`);
});
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
  remember(`[${message.type()}] ${message.text()}`);
});

// What is left behind when a run fails: in CI nothing else of it remains (T82)
async function keepArtifacts(reason) {
  const directory = process.env.E2E_ARTIFACTS;
  if (!directory) return;
  fs.mkdirSync(directory, { recursive: true });
  const name = path.join(directory, `${engine}-${model}`.replace(/[^\w.-]+/g, "_"));
  // a page that hangs may not answer these either: none of them may keep the run from ending
  const within = (promise) => Promise.race([promise, new Promise((resolve) => setTimeout(resolve, 15000))]).catch(() => {});
  const status = await within(page.evaluate(() => document.getElementById("status-text")?.textContent ?? ""));
  await within(page.screenshot({ path: `${name}.png`, fullPage: true }));
  const html = await within(page.content());
  if (html) fs.writeFileSync(`${name}.html`, html);
  fs.writeFileSync(`${name}.log`, [`${engine} ${browserVersion}, ${model}: ${reason}`, `status: ${status ?? "(no answer)"}`, "",
    ...consoleLines].join("\n") + "\n");
}

// One JSON line per run, for tests/summary.mjs
function record(entry) {
  const file = process.env.E2E_RESULTS;
  if (!file) return;
  const extra = process.env.E2E_QUERY ? { query: process.env.E2E_QUERY } : {};
  fs.appendFileSync(file, JSON.stringify({ engine, browserVersion, model, os: `${os.platform()} ${os.arch()}`, ...extra, ...entry }) + "\n");
}

// Anything that throws (a navigation that fails, a closed page) is a failed run too, and must be recorded as one
process.on("uncaughtException", async (error) => {
  console.error(`FAILED\n- ${error.message ?? error}`);
  record({ ok: false, timedOut: false, failures: [String(error.message ?? error)] });
  if (page) await keepArtifacts(String(error.message ?? error));
  process.exit(1);
});

const opens = model === "local" || model === "hf";
// the page asks before it fetches more than 500 MB
page.on("dialog", (dialog) => dialog.accept());
const tinyllamas = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K";
const [repository, revision] = model.startsWith("hf:") ? model.slice(3).split("@") : [];
const query = model === "url" ? `checkpoint=${encodeURIComponent(`${tinyllamas}/stories260K.bin`)}&tokenizer=${encodeURIComponent(`${tinyllamas}/tok512.bin`)}`
  : repository ? `hf=${encodeURIComponent(repository)}${revision ? `&revision=${encodeURIComponent(revision)}` : ""}`
  : `model=${opens ? "stories3_5M" : model}`;
await page.goto(`${url}?${query}${process.env.E2E_QUERY ? `&${process.env.E2E_QUERY}` : ""}`);
// T93: the first visit reloads once, under the service worker that makes the page cross-origin isolated (coi.js):
// a wait that the reload interrupts starts again on the new page
const acrossReload = async (wait) => {
  for (;;) {
    try {
      return await wait();
    } catch (error) {
      if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
    }
  }
};
// (during the reload the page may have no button yet: that is "not yet", not an error)
const idle = () => acrossReload(() => page.waitForFunction(() => document.getElementById("run")?.disabled === false || document.querySelector(".error"), null, { timeout: 0 }));
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
  await page.waitForFunction(() => /^stories260K(\.bin| from Hugging Face files) ·|^Could not/.test(document.getElementById("status-text").textContent), null, { timeout: 0 });
  await idle();
}
const readySeconds = (Date.now() - started) / 1000;
// T84: the worker's own breakdown (Pyodide, download, the conversion's share of it, Llama()) and its memory
const reported = await page.evaluate(() => window.__ready ?? null).catch(() => null);
// By default a model writes until its context is full (4096 tokens for llm-jp-3-150m). The test, and every tok/s in
// the documents, is about 256 tokens: set that in the settings, as a visitor would.
await page.evaluate(() => {
  const steps = document.getElementById("steps");
  steps.value = "256";
  steps.dispatchEvent(new Event("input"));
});
// Enter alone breaks the line
await page.press("#prompt", "Control+Enter");
await page.waitForFunction(() => document.querySelector(".model .meta") || document.querySelector(".error"), null, { timeout: 0 });
const result = await page.evaluate(() => ({
  text: document.querySelector(".model .bubble")?.textContent ?? "",
  // the closed line; the breakdown below it is in the same element
  meta: document.querySelector(".model .meta summary")?.textContent ?? "",
  error: document.querySelector(".error .bubble")?.textContent ?? "",
  // which kernels this browser got: "SIMD kernels, int8, relaxed SIMD", or less
  status: document.getElementById("status-text")?.textContent ?? "",
  pageScrolls: document.documentElement.scrollHeight > innerHeight,
  // T93: whether the service worker made the page cross-origin isolated (the software threads need it)
  isolated: self.crossOriginIsolated,
}));
const failures = [];
if (result.error) failures.push(`the page reported: ${result.error}`);
if (errors.length) failures.push(`console errors: ${errors.join(" | ")}`);
if (!/tok\/s/.test(result.meta)) failures.push("no speed line under the answer");
if (result.pageScrolls) failures.push("the page itself scrolls");
if (expected[model] && !result.text.startsWith(expected[model])) failures.push(`unexpected text: ${result.text.slice(0, 120)}`);
console.log(`${engine} ${browserVersion}, ${model}: ready in ${readySeconds.toFixed(1)}s, ${result.meta}`);
console.log(`status: ${result.status}${result.isolated ? "" : " (not cross-origin isolated)"}`);
console.log(result.text.slice(0, 160).replace(/\n/g, " / "));
let again = null;
if (process.env.E2E_TWICE && !failures.length) {
  // what the browser holds, as the page lists it (public/kept.js)
  const listKept = () => page.evaluate(async () => {
    const module = await import(new URL("kept.js", location.href).href);
    return (await module.keptModels()).map(({ where, manifest }) => `${manifest.name} in ${where}`);
  }).catch((error) => [`(could not list: ${error.message})`]);
  const kept = await listKept();
  const reloaded = Date.now();
  await page.reload({ waitUntil: "load" });
  // the new page's own report of ready: the old page's must not count
  await acrossReload(() => page.waitForFunction(() => window.__ready || document.querySelector(".error"), null, { timeout: 0 }));
  const ready = await page.evaluate(() => window.__ready ?? null).catch(() => null);
  again = { readySeconds: (Date.now() - reloaded) / 1000, fromCache: Boolean(ready?.fromCache), keptIn: ready?.keptIn ?? null,
    miss: ready?.keptMiss ?? null, kept, keptAfter: await listKept(),
    // T111: the worker's own breakdown of the second load (Pyodide, the model), to see what the browser's cache saved
    load: ready?.seconds ?? null };
  const parts = Object.entries(again.load ?? {}).map(([name, value]) => `${name} ${Number(value).toFixed(2)}s`).join(", ");
  console.log(`again: ready in ${again.readySeconds.toFixed(1)}s (${parts}), ${again.fromCache ? `kept in ${again.keptIn}` : `not kept: ${again.miss}`} (kept before: ${kept.join(", ") || "nothing"}; after: ${again.keptAfter.join(", ") || "nothing"})`);
  if (/^hf[-:]/.test(model) && !again.fromCache) failures.push(`not kept for the second visit: ${reported?.notKept ?? "no reason given"}`);
}
const speed = Number(result.meta.match(/([\d.]+) tok\/s/)?.[1]);
record({ ok: !failures.length, timedOut: false, readySeconds, tokPerSecond: Number.isFinite(speed) ? speed : null,
         backend: result.status, meta: result.meta, failures, isolated: result.isolated, load: reported?.seconds ?? null,
         heapMB: reported?.heap ? Math.round(reported.heap / 1e6) : null, notKept: reported?.notKept ?? null, again });
if (failures.length) await keepArtifacts(failures.join("; "));
clearTimeout(watchdog);
await browser.close();
server?.close();
if (failures.length) {
  console.error("FAILED\n- " + failures.join("\n- "));
  process.exit(1);
}
console.log("ok");
