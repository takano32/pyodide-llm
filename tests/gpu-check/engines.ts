// T357.3: the two engines of tests/gpu-check.ts, a part of that tool: the server that gives a browser its page, the
// harness (harness.js, the text beside this file: the worker that runs forward.js on the CPU and with the GPU's
// worker), the cases and the runtime's files; a browser of Playwright's on that page; and Node with Dawn, the same
// harness in a worker thread. Either gives what the harness posted, and puts the console's lines into `lines`.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { built, runtime, runtimeUrl } from "../tree.ts";
import { typed, untyped } from "../untyped.ts";

const root = new URL("../../", import.meta.url).pathname;

// ---- the browser: a page that is cross-origin isolated (its own headers), a worker that runs forward.js
// (harness.js is that worker's text whole, as the template literal that held it here had it: a new line before its
// first line. The browser gets it after two lines of constants, Dawn's worker thread after its prelude, with the
// runtime's modules named by their files)
const HARNESS = `\n${fs.readFileSync(new URL("./harness.js", import.meta.url), "utf8")}`;
// an icon of its own: a browser without one asks for /favicon.ico (a 404 in the console of the real Chrome)
const PAGE = `<!doctype html><meta charset="utf-8"><title>gpu-check</title><link rel="icon" href="data:,"><script type="module">
const got = window.__gpuCheck = { results: [] };
worker.onmessage = ({ data }) => (data.line !== undefined ? console.log(data.line) : data.result ? got.results.push(data.result) : Object.assign(got, data));
worker.onerror = (event) => Object.assign(got, { error: event.message ?? "the harness did not start", ended: true });
</script>`;
const types = { ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".html": "text/html; charset=utf-8" };
const served = (cases, only, nanRounds) => http.createServer((req, res) => {
  const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  const headers = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
  const send = (type, body) => {
    res.writeHead(200, { ...headers, "Content-Type": type });
    res.end(body);
  };
  const found = cases.find((c) => c.checkpoint === pathname);
  if (pathname === "/") return send(types[".html"], PAGE);
  if (pathname === "/harness.js") return send(types[".js"], `const ONLY = ${JSON.stringify(only ? only.split(",") : [])};\nconst NAN_ROUNDS = ${JSON.stringify(nanRounds)};\n${HARNESS}`);
  // (T152: and NumPy's greedy ids after the prompt, which the harness feeds the GPU's steps)
  if (pathname === "/cases.json") return send(types[".json"], JSON.stringify(cases.map(({ file, reference: { tokens, header, greedy }, ...c }) => ({ ...c, reference: { tokens, header, greedy } }))));
  if (found) return send("application/octet-stream", fs.readFileSync(found.file));
  // (the runtime's files and the built kernels, each from where this tree keeps them: tests/tree.ts)
  const kinds = { "/runtime/": runtime, "/built/": built };
  const kind = Object.keys(kinds).find((prefix) => pathname.startsWith(prefix));
  // (T368.1: a file of the runtime that is TypeScript goes to the browser as the JavaScript it is with its types
  // blanked, tests/untyped.ts: a browser takes no .ts, and the sources' imports name it so)
  let file = "";
  try { file = kind ? kinds[kind](pathname.slice(kind.length)) : ""; } catch { /* not in this tree: 404 */ }
  if (!kind || !file || !fs.statSync(file).isFile()) {
    res.writeHead(404, headers);
    return res.end();
  }
  if (typed(file)) return send(types[".js"], untyped(fs.readFileSync(file, "utf8"), file));
  send(types[path.extname(file)] ?? "application/octet-stream", fs.readFileSync(file));
}).listen(0);

// T645: what the harness posts, as it goes: its forms and the quantizers' answers first, a model's result as each is
// through ({ result }), and { ended } at its end or with its error. Its progress lines (harness.js's PROGRESS) go to
// stderr at once with the minutes since the tool began, and the last is kept: a run stopped at the tool's limit
// (--minutes, under the job's: gpu-prompt.yml) has the verdicts of the models before and names where it stopped
// (T645: gpu-prompt.yml's Chromium job ran into its limit and said nothing of where)
const PROGRESS = "gpu-check progress: ";
const began = performance.now();
const watcher = (lines) => {
  let last = "", model = "";
  return {
    line(text, shown) {
      const at = text.indexOf(PROGRESS);
      if (at < 0) return lines.push(shown);
      last = text.slice(at + PROGRESS.length);
      if (last.endsWith(" begun")) model = last.slice(0, -" begun".length);
      process.stderr.write(`[${((performance.now() - began) / 60000).toFixed(1)} min] ${last}\n`);
    },
    stopped: (outcome, minutes) => ({ results: [], ...outcome, ended: true,
      error: `gave up at the limit of ${minutes} minutes (--minutes): the harness was ${model ? `in ${model}, the last it said "${last}"` : "not yet at a model"}` }),
  };
};
// the milliseconds left of the tool's limit (from its start: NumPy's answers take some of it)
const left = (minutes) => Math.max(1000, minutes * 60000 - (performance.now() - began));

async function inBrowser({ engine, server, lines, minutes }) {
  const playwright = await import("playwright-core");
  // Chromium's WebGPU without a GPU: SwiftShader (as tests/bench-check.ts has it)
  const WEBGPU = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-webgpu-adapter=swiftshader"];
  const browser = await playwright.chromium.launch({ ...(engine === "chromium" ? {} : { channel: engine }), args: WEBGPU });
  const page = await browser.newPage();
  const watch = watcher(lines);
  page.on("console", (message) => watch.line(message.text(), `[${message.type()}] ${message.text()}`));
  page.on("pageerror", (error) => lines.push(`[pageerror] ${error.message}`));
  await page.goto(`http://localhost:${server.address().port}/`);
  // SwiftShader compiles a tiled shader in 10 to 90 s (T147): every shader of the made-up model takes some minutes
  let result;
  try {
    await page.waitForFunction(() => window.__gpuCheck?.ended, null, { timeout: left(minutes) });
    result = await page.evaluate(() => window.__gpuCheck);
  } catch (error) {
    if (error?.name !== "TimeoutError") throw error;
    result = watch.stopped(await page.evaluate(() => window.__gpuCheck).catch(() => ({})), minutes);
  }
  await Promise.race([browser.close(), new Promise((resolve) => setTimeout(resolve, 15000))]);
  return result;
}

// T147: the harness in a worker thread of Node, WebGPU from Dawn (the npm package webgpu at --webgpu), the GPU's
// worker (src/runtime/gpu.js) in a worker thread of its own, the modules from their files and the rest from the server
async function inDawn({ webgpu, directory, server, only, nanRounds, lines, minutes }) {
  const { Worker } = await import("node:worker_threads");
  if (!webgpu) throw new Error("--engine dawn wants --webgpu <the directory of the npm package webgpu>");
  const dir = path.join(directory, "dawn");
  fs.mkdirSync(dir, { recursive: true });
  // (T225's review: GPU_ROUNDING=toward-zero|away|everything has the device round a float32 to a float16 that way, as
  // Direct3D's and Mesa's AMD do and lavapipe does not: tests/rounding.ts. The lines of this check that hold a key or a
  // value of the cache to NumPy's (the first layer's scale, T187) assume the nearest and fail where it is not: read the
  // engine's own checks of its forms by the "gpu:" lines of the log)
  const prelude = `import { parentPort, Worker as NodeWorker } from "node:worker_threads";
import { create, globals } from ${JSON.stringify(pathToFileURL(path.resolve(webgpu, "index.js")).href)};
import { roundedGpu } from ${JSON.stringify(pathToFileURL(path.join(root, "tests", "rounding.ts")).href)};
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, "navigator", { value: { gpu: roundedGpu(create([]), process.env.GPU_ROUNDING) }, configurable: true });
globalThis.self = globalThis;
globalThis.postMessage = (data) => parentPort.postMessage(data);
const origin = "http://localhost:${server.address().port}", nativeFetch = fetch;
globalThis.fetch = (url, init) => nativeFetch(new URL(url, origin), init);
console.log = console.info = console.warn = (...parts) => parentPort.postMessage({ line: parts.join(" ") });
`;
  const gpuFile = path.join(dir, "gpu-worker.mjs"), harnessFile = path.join(dir, "harness.mjs");
  // the GPU's worker: its messages wait until gpu.js has set onmessage (a module worker's port opens at its first await)
  // (T352: until gpu.js is through, not until it first sets onmessage: the window sets one of its own while its modules
  // load and plays what that one kept when they have; a message taken here before it and played after would overtake)
  fs.writeFileSync(gpuFile, `${prelude}
const waiting = [];
let through = false;
globalThis.onmessage = null;  // gpu.js sets it, a module's plain assignment
parentPort.on("message", (data) => (through ? globalThis.onmessage({ data }) : waiting.push(data)));
globalThis.close = () => process.exit(0);
await import(${JSON.stringify(pathToFileURL(runtime("gpu.js")).href + "?v=gpu-check")});
through = true;
waiting.splice(0).forEach((data) => globalThis.onmessage({ data }));
`);
  fs.writeFileSync(harnessFile, `${prelude}
globalThis.Worker = class {
  constructor() {
    this.worker = new NodeWorker(${JSON.stringify(gpuFile)});
    this.worker.on("message", (data) => (data.line !== undefined ? parentPort.postMessage(data) : this.onmessage?.({ data })));
    this.worker.on("error", (error) => this.onerror?.({ message: String(error?.stack ?? error) }));
  }
  postMessage(data) { this.worker.postMessage(data); }
  terminate() { this.worker.terminate(); }
};
const ONLY = ${JSON.stringify(only ? only.split(",") : [])};
const NAN_ROUNDS = ${JSON.stringify(nanRounds)};
${HARNESS.replaceAll('import("/runtime/', `import(${JSON.stringify(runtimeUrl("").href.replace(/\/?$/, "/"))} + "`)}
`);
  const watch = watcher(lines), got = { results: [] };
  return new Promise((resolve) => {
    const worker = new Worker(harnessFile);
    const end = (outcome) => {
      clearTimeout(limit);
      resolve(outcome);
      worker.terminate();
    };
    const limit = setTimeout(() => end(watch.stopped(got, minutes)), left(minutes));
    worker.on("message", (data) => {
      if (data.line !== undefined) return watch.line(data.line, `[log] ${data.line}`);
      if (data.result) return got.results.push(data.result);
      Object.assign(got, data);
      if (got.ended) end(got);
    });
    worker.on("error", (error) => end({ ...got, error: String(error?.stack ?? error), ended: true }));
  });
}

export { served, inBrowser, inDawn };
