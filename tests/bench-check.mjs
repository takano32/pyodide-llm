// /benchmark/ (T134) in real browsers: every section run once (?run=all), the Markdown printed, and a failure where a
// result cannot be right. A feature a browser lacks (no WebGPU, no private file system in a worker) is no failure: the
// page says so and goes on. The runners are not the owner's devices, so the numbers only show that the page works;
// Playwright's Firefox runs under a debugger and its numbers are not Firefox's (AGENTS.md). Meant for CI.
//
//   node tests/bench-check.mjs [url of a site | --dist] [engine ...] [--model id] [--size MiB] [--run sections]
//     engines: chromium firefox webkit chrome msedge (default: the first three)
//     --dist serves dist/ (npm run build) itself, as tests/screenshots.mjs does: a branch before it goes out
//     --run the page's ?run= (default all; "gpu" alone: the GPU section takes minutes on SwiftShader, T146)
//     --unsaid hides navigator.deviceMemory from the page (T214: the model section skips its NumPy round)
//     --wrong (with --dist) serves the GPU section's worker with its layer check's line cut to 1e-12, so that every layer
//       verdict is WRONG, as a device whose rounding differs from the check's would make them (T227's review: no CI
//       runner makes a WRONG verdict, and nothing else tries the page's wiring of the report's warnings): the report must
//       hold the WRONG rows under "#### Warnings", before the sections, and the link what fits of them. A change to a
//       served file, in this test alone: the page has no flag for it
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as playwright from "playwright-core";
import { REPORT_LIMIT, TOO_LONG, pathTable } from "../src/bench.js";

// T180: the browsers' profiles, in the repository's .tmp/ (never $HOME or /tmp)
// fileURLToPath, not URL.pathname: on Windows the pathname is "/D:/a/…" and joined it made "D:\\D:\\a\\…" (T181)
const PROFILES = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", ".tmp", "bench-check-profiles");

const args = process.argv.slice(2);
const option = (name, value) => {
  const at = args.indexOf(name);
  return at >= 0 ? args.splice(at, 2)[1] : value;
};
const model = option("--model", "tiny-lm"), size = option("--size", "256"), sections = option("--run", "all");
// T184: more of the page's query, e.g. --query gpuTest=on (CI's fallback adapter taken as a GPU in the model section)
const query = option("--query", "");
// T214: --unsaid hides navigator.deviceMemory from the page (as Safari and Firefox do not say it): the model section's
// round without the kernels must be a row that says why, and never load
const unsaid = args.includes("--unsaid") ? args.splice(args.indexOf("--unsaid"), 1) : null;
const dist = args.includes("--dist") ? args.splice(args.indexOf("--dist"), 1) : null;
const wrong = args.includes("--wrong") ? args.splice(args.indexOf("--wrong"), 1) : null;
if (wrong && !dist) {
  console.log("--wrong changes a file the page is served, so it needs --dist");
  process.exit(1);
}
// the layer check's line in public/benchmark/gpu.js: a stream off by more than it is WRONG
const LAYER_LINE = "LAYER_LINE = 1e-3", NO_LINE = "LAYER_LINE = 1e-12";
let [site = "https://takano32.github.io/pyodide-llm/", ...engines] = dist ? [undefined, ...args] : args;
let server;
if (dist) {
  const base = "/pyodide-llm/", root = new URL("../dist/", import.meta.url).pathname;
  const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
    ".py": "text/plain", ".wasm": "application/wasm" };
  server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    let file = path.join(root, pathname.slice(base.length));
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
    if (!pathname.startsWith(base) || !fs.existsSync(file)) {
      res.writeHead(404);
      return res.end();
    }
    // --wrong: the GPU section's worker with no line left for the layer check to hold (a rename of it fails here, loudly)
    if (wrong && pathname.endsWith("/benchmark/gpu.js")) {
      const text = fs.readFileSync(file, "utf8");
      if (!text.includes(LAYER_LINE)) {
        console.log(`--wrong: ${LAYER_LINE} is not in benchmark/gpu.js any more: change this test with it`);
        process.exit(1);
      }
      res.writeHead(200, { "Content-Type": types[".js"] });
      return res.end(text.replace(LAYER_LINE, NO_LINE));
    }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  }).listen(0);
  site = `http://localhost:${server.address().port}${base}`;
}
// Chromium's WebGPU without a GPU: SwiftShader's Vulkan, where the build has it (as tests/gpu-check.mjs had it)
const WEBGPU = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-webgpu-adapter=swiftshader"];
// the sections whose failure is this page's: the others depend on the runner (its GPU, its disk, its line), and what
// they could not do is printed
const OURS = ["device", "cpu", "model"];
// T177: every stage a section showed while it ran, as the page drew it (a watcher in the page from its start, so that a
// stage shown for a moment is seen too). A section of more than one stage that measured and never changed its stage
// on the page would look frozen to a visitor: that fails (the page's own "starting" is not counted: the stages come from
// the section). The browser's features are one quick stage, and a section
// that found no such thing here stops at its first.
const STAGED = ["cpu", "model", "gpu", "storage", "line"];
function watchStages() {
  const seen = (window.__stages = {});
  const look = () => {
    for (const box of document.querySelectorAll("section[data-section] .progress")) {
      const text = box.querySelector(".stage")?.textContent;
      const name = box.closest("section").dataset.section;
      if (box.hidden || !text) continue;
      const list = (seen[name] ??= []);
      if (list.at(-1) !== text) list.push(text);
    }
  };
  document.addEventListener("DOMContentLoaded", () => {
    new MutationObserver(look).observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true });
  });
}

let failed = false;
// T181: a browser that does not start is passed over (a runner may lack Chrome or Edge), but none starting is no pass
let started = 0;
for (const engine of engines.length ? engines : ["chromium", "firefox", "webkit"]) {
  const branded = ["chrome", "msedge"].includes(engine);
  const type = branded ? playwright.chromium : playwright[engine];
  console.log(`### ${engine}`);
  // T180: a profile on the disk, not the throwaway (off-the-record) one of browser.newContext(): Chromium keeps an
  // off-the-record profile's storage in memory, and on macOS' runner (7 GB) the model section's llm-jp-3 150M in the
  // cache and the storage section's 256 MiB no longer fit ("No space" at about 200 MB, 2026-09-27). A visitor's
  // browser keeps its storage on the disk, as this does. The profile is removed when the browser closes.
  const profile = path.join(PROFILES, `${engine}-${process.pid}`);
  let context;
  try {
    fs.rmSync(profile, { recursive: true, force: true });
    context = await type.launchPersistentContext(profile, { ...(branded ? { channel: engine } : {}), args: branded || engine === "chromium" ? WEBGPU : [] });
  } catch (error) {
    console.log(`could not start: ${String(error.message).split("\n")[0]}\n`);
    continue;
  }
  started++;
  const page = context.pages()[0] ?? await context.newPage();
  await page.addInitScript(watchStages);
  if (unsaid) await page.addInitScript(() => Object.defineProperty(Navigator.prototype, "deviceMemory", { get: () => undefined, configurable: true }));
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error.message)));
  // T242: what the page and its workers said and which requests failed, printed where a section failed or the page
  // went down ("[object Object]" and "Load failed" alone did not say which fetch, of which worker, nor why)
  const heard = [];
  const hear = (line) => heard.push(`${((Date.now() - opened) / 1000).toFixed(1)} s ${line}`.slice(0, 600));
  const opened = Date.now();
  page.on("console", (message) => {
    if (["error", "warning", "info"].includes(message.type())) hear(`console ${message.type()}: ${message.text()}`);
  });
  page.on("requestfailed", (request) => hear(`request failed: ${request.method()} ${request.url()} (${request.failure()?.errorText ?? "no reason given"})`));
  page.on("crash", () => hear("the page crashed"));
  page.on("close", () => hear("the page closed"));
  let wentWrong = false;
  try {
    await page.goto(`${site}benchmark/?run=${sections}&model=${model}&size=${size}${query ? `&${query}` : ""}`);
    // the first visit reloads once, when the service worker takes the page over (T93): the wait starts again
    for (;;) {
      try {
        await page.waitForFunction(() => window.__benchmark?.done, null, { timeout: 1200000 });
        break;
      } catch (error) {
        if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
      }
    }
    console.log(`${await page.evaluate(() => navigator.userAgent)}\n`);
    const { results, markdown } = await page.evaluate(() => window.__benchmark);
    console.log(markdown, "\n");
    const statuses = Object.entries(results).map(([name, result]) => `${name} ${result.status}`);
    console.log(`sections: ${statuses.join(", ")}`);
    for (const [name, result] of Object.entries(results)) {
      // (--wrong: the GPU section is meant to be WRONG, and is checked below that it is, and what its report says)
      if ((result.status === "wrong" && !(wrong && name === "gpu")) || (result.status === "error" && OURS.includes(name))) failed = wentWrong = true;
    }
    if (wrong) {
      // T227's review: the verdicts every layer form gave, WRONG, are in the report's warnings, the worst first, before the
      // sections (what parseReport() reads is above them), and the link holds what fits of them: the whole, or the summary
      // with the first of them and a line of how many are left, or (nothing fits) the request to paste, never nothing
      const at = markdown.indexOf("\n#### Warnings\n"), end = markdown.indexOf("\n#### ", at + 1);
      const firstSection = markdown.search(/\n#### (This browser|CPU|GPU|Storage|Line)/);
      const warned = at < 0 ? [] : markdown.slice(at, end < 0 ? undefined : end).split("\n").filter((line) => line.startsWith("- "));
      const layers = warned.filter((line) => line.startsWith("- GPU: a layer, ") && line.includes(" WRONG ("));
      const href = await page.evaluate(() => document.getElementById("issue").href);
      const body = new URL(href).searchParams.get("body") ?? "";
      const login = `https://github.com/login?return_to=${encodeURIComponent(href)}`.length;
      const kept = body.split("\n").filter((line) => layers.includes(line)).length;
      console.log(`wrong: the GPU section is ${results.gpu?.status}; ${warned.length} warnings, ${layers.length} of them a layer WRONG; ` +
        `the link holds ${body.includes("#### Summary") ? `the summary with ${kept} of them` : body.endsWith(TOO_LONG) ? "only the request to paste" : "the whole report"} (login ${login} of ${REPORT_LIMIT})`);
      const problems = [];
      if (results.gpu?.status !== "wrong") problems.push(`the GPU section is ${results.gpu?.status}, not WRONG`);
      if (!layers.length) problems.push("no layer WRONG among the report's warnings");
      if (!(at >= 0 && at < firstSection)) problems.push("the warnings are not before the sections");
      const other = warned.findIndex((line) => !line.includes(" WRONG ("));
      if (other >= 0 && warned.slice(other).some((line) => line.includes(" WRONG ("))) problems.push("a warning that is not WRONG comes before one that is");
      if (login > REPORT_LIMIT) problems.push("the link is past its limit");
      if (!body.endsWith(TOO_LONG) && !body.includes("- GPU: a layer, ")) problems.push("the link holds no layer WRONG");
      for (const problem of problems) console.log(`wrong: FAILED, ${problem}`);
      if (problems.length) failed = true;
    }
    // T184: the model page's own path, a line a row (CI's fallback adapter times the CPU sides only: the GPU's cells
    // say why), and none at all, or no row of it, is a failure of the model section
    if (results.model?.status === "ok") {
      const paths = results.model.data?.paths;
      for (const line of pathTable(paths).split("\n").filter((text) => text && !text.startsWith("|---"))) console.log(`model path: ${line}`);
      if (!paths?.rows?.length) {
        console.log("model path: FAILED, nothing timed");
        failed = true;
      }
    }
    const stages = await page.evaluate(() => window.__stages);
    // T214: where the page is not told the memory, the round without the kernels is skipped in words and never loads
    if (unsaid && results.model?.status === "ok") {
      const rows = results.model.data?.rows ?? [];
      const plain = rows.find((row) => row.name === "without the kernels");
      const loadedIt = (stages.model ?? []).some((text) => text.includes("round: without the kernels"));
      console.log(`model without deviceMemory: ${plain?.skip !== undefined ? `skipped (${plain.skip})` : "ran"}` +
        `${loadedIt ? ", and its load was staged" : ""}; the rounds' rows: ${rows.map((row) => row.name).join(", ")}`);
      if (!/does not say how much memory/.test(plain?.skip ?? "") || plain.speed !== undefined || loadedIt ||
          !results.model.markdown.includes("| without the kernels | skipped |") || !(rows.find((row) => row.name === "everything")?.speed > 0)) {
        console.log("model without deviceMemory: FAILED, the round without the kernels was not skipped in words");
        failed = true;
      }
    } else if (unsaid) {
      console.log(`model without deviceMemory: FAILED, the section is ${results.model?.status ?? "missing"}`);
      failed = true;
    }
    for (const [name, result] of Object.entries(results)) {
      const shown = stages[name] ?? [];
      console.log(`${name} stages on the page: ${shown.join(" → ") || "none"}`);
      if (STAGED.includes(name) && ["ok", "wrong"].includes(result.status) && new Set(shown.filter((text) => text !== "starting")).size < 2) {
        console.log(`${name}: its stage on the page never changed while it ran`);
        failed = true;
      }
    }
    // the shaders against JavaScript are the GPU section's check of its own results: with an adapter they must run
    // (and every tiled shader of T146 among them: one the runner's adapter could not make fails here too)
    const check = results.gpu?.data?.steps?.find((s) => s.name === "the shaders against JavaScript");
    if (check?.error || Object.values(check?.result ?? {}).some((v) => v.error)) failed = true;
    // T175: the numbers the layer's verdicts rest on, a line each (CI's logs are where they are read: the owner, 2026-09-27):
    // the stream's and the cache's differences, each quantized vector's scales and values off by 1, and the DP4A fused
    // form against the one with the norms apart (the ulp of the scales, the values off by 1, bit for bit or not)
    for (const [name, v] of Object.entries(check?.result ?? {})) {
      if (!name.startsWith("a layer, ") && !name.startsWith("a token's attention, ") && name !== "tokens on the GPU") continue;
      const apart = v.sameAsNormsApart;
      console.log(`check ${name}: ${v.ok ? "ok" : "WRONG"}` + (v.error ? `, ${v.error}` : "") +
        (Number.isFinite(v.stream) ? `, stream ${v.stream.toExponential(1)}, cache ${v.cache.toExponential(1)}` : "") +
        (v.quantized ? `; ${v.quantized.map((q) => `${q.point}: ${q.wrong ?? `scales ${q.scale.toExponential(1)}, ${q.apart} of ${q.of} values off by 1`}`).join("; ")}` : "") +
        (apart ? `; against the norms apart: scales ${apart.ulps} ulp, ${apart.apart} values off by 1, stream ${apart.stream.toExponential(1)}, ${apart.bitForBit ? "bit for bit" : "not bit for bit"}` : "") +
        (name === "tokens on the GPU" ? `; ${v.tokens} tokens, layer ${v.layer}${v.problems ? `, ${v.problems.join(" / ")}` : ""}` : "") +
        // T225: how each stage of the layer held, and each step of the tokens (the worker's own lines)
        (v.stages ? `; ${v.stages}` : "") + (v.steps ? `; ${v.steps}` : "") +
        // T224: a token's attention, its worst head over the largest value, and where (the head's size, the positions)
        (v.at ? `, worst ${v.worstRelative.toExponential(1)} at a head of ${v.at.headSize}, ${v.at.positions} positions` : ""));
    }
    // T157: after the CPU section, the GPU section's token table holds the GPU against it (and not the old estimate)
    if (results.cpu?.status === "ok" && results.gpu?.data && !results.gpu.markdown.includes("The CPU (an estimate): each model's weights at the CPU section's fastest")) {
      console.log("the GPU section's token table does not hold the GPU against the CPU section");
      failed = true;
    }
  } catch (error) {
    console.log(`failed: ${String(error.message).split("\n")[0]}`);
    failed = wentWrong = true;
  }
  if (errors.length) console.log(`page errors: ${errors.join(" / ")}`);
  if (wentWrong || errors.length) for (const line of heard.slice(-60)) console.log(`heard: ${line}`);
  console.log("");
  // T141: Windows' WebKit sometimes never returns from close(), and each browser waits for the one before it (bench.yml's
  // Windows job sat 55 minutes after WebKit's report, 2026-09-27): give it 15 seconds and go on
  const closed = await Promise.race([context.close().then(() => true), new Promise((resolve) => setTimeout(() => resolve(false), 15000))]);
  if (!closed) console.log(`${engine}: the browser did not close within 15 s; going on`);
  try {
    fs.rmSync(profile, { recursive: true, force: true });
  } catch {
    // a browser that did not close may hold its files: they go with the runner (or .tmp/ on this machine)
  }
}
server?.close();
if (!started) console.log("no browser started: nothing was measured");
process.exit(failed || !started ? 1 : 0);
