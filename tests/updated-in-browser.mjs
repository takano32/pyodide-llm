// T367.4: a page left open across a deployment, in a real browser. The build in dist/ is served as GitHub Pages serves
// it, and the server is told, while a page is open, which files of _astro/ are gone (404, as after a deployment),
// which fail (500) and which drop the line; then the page is made to ask for a file it had not fetched yet.
//
//   npm run build && node tests/updated-in-browser.mjs [chromium|firefox|webkit|msedge|chrome] [a word of a case's name …]
//
// It sees, on the model page and on /benchmark/:
// - a file of the page's own build that is gone ends what asked for it with the one sentence, alone
//   ("The site was updated while this page was open. Reload the page to go on."): the converter's archive and the
//   module of what is kept at a first conversion, the model's worker itself, a section's worker, a module and a kernel
//   a section's worker asks for;
// - the same file failing in any other way keeps the words it had: a 500, a dropped line, a browser that is offline;
// - a model that goes on without a file (the kernels, a software thread: NumPy, fewer threads) goes on and says nothing.
// It does not see: a GPU (its worker's fallback is tests/updated-check.mjs's reading of the sources), a real deployment.
//
// The model is stories260K (1 MB), and a conversion is the folder of tests/make_hf_fixture.py, as tests/e2e.mjs's hf.
// Every case has a browser context of its own (its own service worker and copies). Pyodide comes from its CDN each time.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execFileSync } from "node:child_process";
import * as playwright from "playwright-core";
import { UPDATED } from "../src/runtime/updated.js";
import { leave } from "./leave.mjs";

const [engine = "chromium", ...only] = process.argv.slice(2);
const base = "/pyodide-llm/";
const root = new URL("../dist/", import.meta.url).pathname;
const repository = new URL("../", import.meta.url).pathname;
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".wasm": "application/wasm" };

// ---- the server: what becomes of a request is the case's rule(name under the site, method): undefined (served),
// a status, or "drop" (the connection ends without an answer)
let rule = () => undefined;
const requests = [];
const server = http.createServer((req, res) => {
  const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  const name = pathname.slice(base.length), file = path.join(root, name || "index.html");
  const directory = fs.existsSync(file) && fs.statSync(file).isDirectory();
  const served = directory ? path.join(file, "index.html") : file;
  const verdict = pathname.startsWith(base) ? rule(name, req.method) : 404;
  requests.push(`${req.method} ${name}${verdict === undefined ? "" : ` -> ${verdict}`}`);
  if (verdict === "drop") return req.socket.destroy();
  if (verdict !== undefined || !pathname.startsWith(base) || !fs.existsSync(served)) {
    res.writeHead(verdict ?? 404);
    return res.end();
  }
  res.writeHead(200, { "Content-Type": types[path.extname(served)] ?? "application/octet-stream", "Content-Length": fs.statSync(served).size });
  if (req.method === "HEAD") return res.end();
  fs.createReadStream(served).pipe(res);
}).listen(0);
const site = `http://localhost:${server.address().port}${base}`;
const built = (name) => name.startsWith("_astro/");
const family = (...names) => (name) => built(name) && names.some((one) => name.startsWith(`_astro/${one}.`));

const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "hf-fixture-"));
execFileSync("python3", [`${repository}tests/make_hf_fixture.py`, fixture]);
const folder = fs.readdirSync(fixture).map((name) => path.join(fixture, name));

const channel = ["msedge", "chrome"].includes(engine) ? engine : undefined;
const kind = playwright[channel ? "chromium" : engine];
const browser = await kind.launch({ headless: true, channel });
console.log(`${engine} ${browser.version()}`);

// T93: the first visit reloads once under the service worker; a wait the reload interrupts starts again
const acrossReload = async (wait) => {
  for (;;) {
    try {
      return await wait();
    } catch (error) {
      if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
    }
  }
};
const lastError = () => [...document.querySelectorAll(".error .bubble")].at(-1)?.textContent;

// ---- the model page
async function modelPage(context) {
  const page = await context.newPage();
  await page.goto(`${site}?model=stories260K`);
  return page;
}
const ready = (page) => acrossReload(() => page.waitForFunction(() => document.getElementById("run")?.disabled === false || document.querySelector(".error"), null, { timeout: 180000 }));
// the folder of a Hugging Face model, which the page converts: what the page said at its end (an error, or the status)
async function converted(page) {
  await page.setInputFiles("#files", folder);
  await page.waitForFunction(() => /from Hugging Face files ·|^Could not/.test(document.getElementById("status-text").textContent), null, { timeout: 180000 });
  return { status: await page.evaluate(() => document.getElementById("status-text").textContent), error: await page.evaluate(lastError) };
}
async function answered(page) {
  const before = await page.evaluate(() => document.querySelectorAll(".model .meta").length);
  await page.fill("#prompt", "Once upon a time");
  await page.click("#run");
  await page.waitForFunction((count) => document.querySelectorAll(".model .meta").length > count || document.querySelector(".error"), before, { timeout: 180000 });
  return { ...(await page.evaluate(() => ({ text: [...document.querySelectorAll(".model .bubble")].at(-1)?.textContent ?? "", status: document.getElementById("status-text").textContent }))),
    error: await page.evaluate(lastError) };
}

// ---- /benchmark/
async function benchmarkPage(context) {
  const page = await context.newPage();
  await page.goto(`${site}benchmark/`);
  // (its first visit reloads under the service worker too; the sections' buttons are there once its script ran)
  await acrossReload(async () => {
    await page.waitForFunction(() => document.querySelector('section[data-section="device"] button') && (self.crossOriginIsolated || !navigator.serviceWorker), null, { timeout: 60000 });
  });
  return page;
}
async function section(page, name) {
  await page.evaluate(() => { window.__benchmark.done = false; });
  await page.click(`section[data-section="${name}"] .head button`);
  await page.waitForFunction(() => window.__benchmark.done, null, { timeout: 180000 });
  return page.evaluate((name) => ({ ...window.__benchmark.results[name], data: undefined, shown: document.querySelector(`section[data-section="${name}"] .out`).textContent,
    state: document.querySelector(`section[data-section="${name}"] .state`).textContent }), name);
}

const failures = [];
const expect = (name, ok, what) => {
  if (!ok) failures.push(`${name}: ${what}`);
};
const isSentence = (name, said) => expect(name, said === UPDATED, `said ${JSON.stringify(said)}, and not the sentence alone`);
const isNot = (name, said, pattern) => {
  expect(name, typeof said === "string" && said !== "" && !said.includes("updated"), `said ${JSON.stringify(said)} of something that is no update`);
  if (pattern) expect(name, pattern.test(said ?? ""), `said ${JSON.stringify(said)}, not ${pattern}`);
};

// Offline: the browser is told so, and the server ends every connection without an answer as well (Playwright's Firefox
// goes on reaching localhost when it is told it is offline: its conversion "offline" simply succeeded, CI's run
// 38123992979), which is what offline is to a page either way: no answer from anywhere
async function offlineNow(context) {
  rule = () => "drop";
  await context.setOffline(true);
}

// ---- the cases. after: the rule once the page is ready (the deployment, or the failure); from the start: `first`
const converting = (after, check, { offline = false } = {}) => async (context, name) => {
  const page = await modelPage(context);
  await ready(page);
  rule = after;
  if (offline) await offlineNow(context);
  const said = await converted(page);
  console.log(`  the page: ${JSON.stringify(said)}`);
  check(name, said);
};
const measuring = (which, after, check, { first, offline = false } = {}) => async (context, name) => {
  if (first) rule = first;
  const page = await benchmarkPage(context);
  if (after) rule = after;
  if (offline) await offlineNow(context);
  const said = await section(page, which);
  console.log(`  the section: ${JSON.stringify({ status: said.status, state: said.state, shown: said.shown.slice(0, 160) })}`);
  check(name, said);
};
const sentenceOfConversion = (name, said) => {
  isSentence(name, said.error);
  expect(name, /^Could not load /.test(said.status), `the status line is ${JSON.stringify(said.status)}`);
};
const sentenceOfSection = (name, said) => {
  isSentence(name, said.markdown);
  isSentence(name, said.shown.trim());
  expect(name, said.status === "error", `the section's status is ${said.status}`);
};
const CASES = {
  // ---- the model page: a first conversion after the site was deployed again
  "model page, a conversion after everything of the build was replaced": converting((name) => (built(name) ? 404 : undefined), sentenceOfConversion),
  "model page, a conversion where the converter's archive alone is gone": converting((name) => (family("converter")(name) ? 404 : undefined), sentenceOfConversion),
  "model page, a conversion where the module of what is kept is gone, and the worker that names it": converting(
    (name) => (family("kept", "worker")(name) ? 404 : undefined), sentenceOfConversion),
  // ---- the same files failing otherwise: no update
  "model page, a conversion whose archive is answered 500": converting((name) => (family("converter")(name) ? 500 : undefined),
    (name, said) => isNot(name, said.error, /^Could not fetch converter\.zip: 500$/)),
  "model page, a conversion whose archive drops the line": converting((name) => (family("converter")(name) ? "drop" : undefined),
    (name, said) => isNot(name, said.error, /TypeError/)),
  "model page, a conversion whose module of what is kept is answered 500 (the worker is there)": converting((name) => (family("kept")(name) ? 500 : undefined),
    (name, said) => isNot(name, said.error, /TypeError/)),
  "model page, a conversion whose module of what is kept is gone alone (the worker that names it is there: no deployment does that)": converting(
    (name) => (family("kept")(name) ? 404 : undefined), (name, said) => isNot(name, said.error, /TypeError/)),
  "model page, a conversion whose module of what is kept drops the line, as does the worker": converting((name) => (family("kept", "worker")(name) ? "drop" : undefined),
    (name, said) => isNot(name, said.error, /TypeError/)),
  "model page, a conversion offline": converting(() => undefined, (name, said) => isNot(name, said.error), { offline: true }),
  // ---- the worker itself (a page that came from a cache, after the deployment: its script is held, the worker is not)
  "model page, the worker gone and the page's script with it": async (context, name) => {
    rule = (file, method) => (family("worker")(file) || (family("index.astro_astro_type_script_index_0_lang")(file) && method === "HEAD") ? 404 : undefined);
    const page = await modelPage(context);
    await acrossReload(() => page.waitForFunction(() => document.querySelector(".error .bubble"), null, { timeout: 60000 }));
    const said = await page.evaluate(lastError);
    console.log(`  the page: ${JSON.stringify(said)}`);
    isSentence(name, said);
  },
  "model page, the worker answered 500": async (context, name) => {
    rule = (file) => (family("worker")(file) ? 500 : undefined);
    const page = await modelPage(context);
    await acrossReload(() => page.waitForFunction(() => document.querySelector(".error .bubble"), null, { timeout: 60000 }));
    const said = await page.evaluate(lastError);
    console.log(`  the page: ${JSON.stringify(said)}`);
    isNot(name, said, /^Failed to start the worker: /);
  },
  // ---- what the engine goes on without: nothing is said, and the model answers
  "model page, the kernels and the software threads gone from the start, and every HEAD answered 404": async (context, name) => {
    rule = (file, method) => (family("simdkernel", "simdkernel_relaxed", "simdkernel_plain", "simdkernel_relaxed_plain", "simdkernel_shared", "simdkernel_relaxed_shared",
      "simdkernel_plain64", "simdkernel_relaxed_plain64", "simdkernel_shared64", "simdkernel_relaxed_shared64", "helper")(file) || (built(file) && method === "HEAD") ? 404 : undefined);
    const page = await modelPage(context);
    await ready(page);
    const said = await answered(page);
    console.log(`  the page: ${JSON.stringify({ ...said, text: said.text.slice(0, 60) })}`);
    expect(name, said.error === undefined, `an error was shown: ${said.error}`);
    expect(name, said.text.startsWith("Once upon a time"), `the model wrote ${JSON.stringify(said.text.slice(0, 60))}`);
    expect(name, /NumPy/.test(said.status), `the status line does not say NumPy: ${said.status}`);
    expect(name, requests.some((line) => /^GET _astro\/simdkernel_plain\..* -> 404$/.test(line)), "no kernel was asked for: the case saw nothing");
  },
  "model page, a software thread gone from the start, and every HEAD answered 404": async (context, name) => {
    rule = (file, method) => (family("helper")(file) || (built(file) && method === "HEAD") ? 404 : undefined);
    const page = await modelPage(context);
    await ready(page);
    const said = await answered(page);
    console.log(`  the page: ${JSON.stringify({ ...said, text: said.text.slice(0, 60) })}`);
    expect(name, said.error === undefined, `an error was shown: ${said.error}`);
    expect(name, said.text.startsWith("Once upon a time"), `the model wrote ${JSON.stringify(said.text.slice(0, 60))}`);
    expect(name, !/NumPy/.test(said.status), `the status line says NumPy: ${said.status}`);
    expect(name, requests.some((line) => /^GET _astro\/helper\..* -> 404$/.test(line)), "no software thread was asked for: the case saw nothing");
  },
  // ---- /benchmark/: a section's worker, and what the worker asks for
  "benchmark, a section after everything of the build was replaced": measuring("device", (name) => (built(name) ? 404 : undefined), sentenceOfSection),
  "benchmark, the model section after everything of the build was replaced": measuring("model", (name) => (built(name) ? 404 : undefined), sentenceOfSection),
  "benchmark, the storage section after everything of the build was replaced": measuring("storage", (name) => (built(name) ? 404 : undefined), sentenceOfSection),
  "benchmark, a section whose worker is answered 500": measuring("device", (name) => (family("benchmark-sections")(name) ? 500 : undefined),
    (name, said) => isNot(name, said.markdown, /^the worker failed: /)),
  "benchmark, a section whose worker drops the line": measuring("device", (name) => (family("benchmark-sections")(name) ? "drop" : undefined),
    (name, said) => isNot(name, said.markdown, /^the worker failed: /)),
  "benchmark, a section whose worker is gone alone (the page's script that names it is there: no deployment does that)": measuring("device",
    (name) => (family("benchmark-sections")(name) ? 404 : undefined), (name, said) => isNot(name, said.markdown, /^the worker failed: /)),
  "benchmark, a section offline": measuring("device", undefined, (name, said) => isNot(name, said.markdown, /^the worker failed: /), { offline: true }),
  "benchmark, a section whose worker came and whose engine's module is gone, with the worker's own file": measuring("device",
    (name, method) => (family("forward")(name) || (family("benchmark-sections")(name) && method === "HEAD") ? 404 : undefined), sentenceOfSection),
  "benchmark, a section whose engine's module is answered 500": measuring("device", (name) => (family("forward")(name) ? 500 : undefined),
    (name, said) => isNot(name, said.markdown, /TypeError/)),
  "benchmark, a section whose kernels are gone": measuring("device", (name) => (family("simdkernel_plain", "simdkernel_relaxed_plain")(name) ? 404 : undefined), sentenceOfSection),
  "benchmark, the CPU section whose kernels are gone": measuring("cpu", (name) => (/^_astro\/simdkernel/.test(name) ? 404 : undefined), sentenceOfSection),
  "benchmark, the CPU section whose ceilings' loops are gone": measuring("cpu", (name) => (/^_astro\/ceilings/.test(name) ? 404 : undefined), sentenceOfSection),
  "benchmark, the CPU section whose software threads are gone, with the worker's own file": measuring("cpu",
    (name, method) => (family("helper")(name) || (family("benchmark-sections")(name) && method === "HEAD") ? 404 : undefined), sentenceOfSection),
  "benchmark, the CPU section whose reading threads are gone, with the worker's own file": measuring("cpu",
    (name, method) => (family("benchmark-reader")(name) || (family("benchmark-sections")(name) && method === "HEAD") ? 404 : undefined), sentenceOfSection),
  "benchmark, the CPU section whose kernels are answered 500 (as before: it says there is no SIMD)": measuring("cpu", (name) => (/^_astro\/simdkernel/.test(name) ? 500 : undefined),
    (name, said) => {
      expect(name, said.status === "none" && /no WebAssembly SIMD/.test(said.markdown), `the section said ${JSON.stringify(said.markdown.slice(0, 120))}`);
    }),
};

for (const [name, run] of Object.entries(CASES)) {
  if (only.length && !only.some((word) => name.includes(word))) continue;
  console.log(`- ${name}`);
  const before = failures.length;
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const lines = [];
  context.on("console", (message) => lines.push(`[${message.type()}] ${message.text().slice(0, 300)}`));
  rule = () => undefined;
  requests.length = 0;
  try {
    await Promise.race([run(context, name), new Promise((_, reject) => setTimeout(() => reject(new Error("the case took more than 5 minutes")), 300000))]);
  } catch (error) {
    failures.push(`${name}: ${String(error.message ?? error).split("\n")[0]}`);
  }
  if (failures.length > before) {
    console.log(failures.slice(before).map((line) => `  FAILED ${line}`).join("\n"));
    console.log(`  what the server was asked for last:\n${requests.slice(-25).map((line) => `    ${line}`).join("\n")}`);
    console.log(`  the console's last lines:\n${lines.slice(-15).map((line) => `    ${line}`).join("\n")}`);
  } else {
    console.log("  ok");
  }
  rule = () => undefined;
  await Promise.race([context.close(), new Promise((resolve) => setTimeout(resolve, 15000))]);
}
fs.rmSync(fixture, { recursive: true, force: true });
console.log(failures.length ? `FAILED\n${failures.map((line) => `- ${line}`).join("\n")}` : "updated-in-browser: ok");
// (a browser's close() may never return, T141: the process ends by itself)
await Promise.race([browser.close(), new Promise((resolve) => setTimeout(resolve, 15000))]);
server.close();
await leave(failures.length ? 1 : 0);
