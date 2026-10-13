// tests/blocks-in-browser.mjs (T364.3, T502): a small model of llama.cpp's blocks opened by the built page in a real
// browser, as an entry of the list that keeps its GGUF's blocks is opened (Underdog Saluki 27B's, which is 7.9 GB and
// runs in CI alone): the bundled worker converts the GGUF with its blocks kept, fetches the kernels of the blocks from
// _astro/, answers, says "llama.cpp's blocks" in the status line, and (a second visit) loads what it kept.
//
//   npm run build && node tests/blocks-in-browser.mjs [chromium|chrome|msedge|firefox|webkit] [a visit's name ...]
//
// The model is the unit tests' (tests/test_kept.py's Model: ten types of blocks by the layer, three value heads to a
// key head, a vocabulary of w0..w39), written by tests/reference_saluki_numpy.py made-up; PYTHON names the Python that
// has NumPy and pytest (default "python").
//
// Nothing of the product is for this: the page has no way to open a GGUF by ?hf= or from a folder, and no flag was
// added. The tool does two things outside the page's code:
//   - it answers https://huggingface.co/made-up/saluki/resolve/main/<file> itself (the browser's requests are routed
//     here: ranges with their Content-Range, as huggingface.co shows it), and
//   - before the page's scripts run it wraps Worker.postMessage, so that the entry ?hf=made-up/saluki makes (a
//     safetensors repository) reaches the worker as the list's entry of such a model would: the GGUF by its name, the
//     vocabulary of "the original" (the same made-up repository), and conversion { dtype: "int8", blocks: true }.
// From there on everything is the product's: the conduct of the conversion, the worker's answers, the memory, the
// kernels' fetch, the engine, the status line.
//
// The visits (all by default):
//   isolated     the page as a visitor gets it: the service worker's isolation, a shared memory, software threads
//   again        the same page loaded once more in the same browser: the conversion comes from what was kept
//   wide         ?wide=on: a 64-bit memory, where the browser has one (Underdog Saluki 27B's is one)
//   plain        ?coi=off: no isolation, a memory that is not shared
//   no-relaxed   ?without=relaxed: the kernels without relaxed SIMD (what Safari runs)
//   widened      ?without=int8: the load does not run on the int8 kernels, so the blocks are widened, no kernel of
//                theirs is fetched, and the line says float32
// It fails where a visit does not get ready, does not answer, logs an error, or says another line than its own.
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
// (Chromium: the requests a service worker makes are routed too, which coi.js's fetches of huggingface.co are)
process.env.PW_EXPERIMENTAL_SERVICE_WORKER_NETWORK_EVENTS ??= "1";
const playwright = await import("playwright-core");
const { leave } = await import("./leave.mjs");
const { serving } = await import("./serving.mjs");

const VISITS = {
  isolated: { query: "", line: /^SIMD kernels, llama\.cpp's blocks, relaxed SIMD$/, kernels: true },
  again: { query: "", line: /^SIMD kernels, llama\.cpp's blocks, relaxed SIMD$/, kernels: true, again: true },
  wide: { query: "&wide=on", line: /^SIMD kernels, llama\.cpp's blocks, relaxed SIMD, 64-bit memory$/, kernels: true },
  plain: { query: "&coi=off", line: /^SIMD kernels, llama\.cpp's blocks, relaxed SIMD$/, kernels: true, isolated: false },
  "no-relaxed": { query: "&without=relaxed", line: /^SIMD kernels, llama\.cpp's blocks \(without relaxed\)$/, kernels: true },
  widened: { query: "&without=int8", line: /^SIMD kernels, float32 \(without int8\)$/, kernels: false },
};
const [engine = "chromium", ...asked] = process.argv.slice(2);
const names = asked.length ? asked : Object.keys(VISITS);
for (const name of names) if (!VISITS[name]) { console.error(`no visit called ${name}: ${Object.keys(VISITS).join(", ")}`); process.exit(2); }

const root = fileURLToPath(new URL("../", import.meta.url));
const work = path.join(root, ".tmp", "blocks-in-browser");
fs.rmSync(work, { recursive: true, force: true });
fs.mkdirSync(work, { recursive: true });
execFileSync(process.env.PYTHON ?? "python", [path.join(root, "tests", "reference_saluki_numpy.py"), "made-up", work], { stdio: ["ignore", "ignore", "inherit"] });
const REPOSITORY = "made-up/saluki", GGUF = "made-up-saluki.gguf";
const files = Object.fromEntries(fs.readdirSync(path.join(work, "gguf")).map((name) => [name, fs.readFileSync(path.join(work, "gguf", name))]));
console.log(`the made-up model: ${Object.entries(files).map(([name, bytes]) => `${name} ${bytes.length.toLocaleString("en")} bytes`).join(", ")}`);

// what the built site was asked for, by name under its base
const served = [];
const site = await serving(undefined, { instead: ({ name }) => { served.push(name); } });
const channel = ["msedge", "chrome"].includes(engine) ? engine : undefined;
const kind = playwright[channel ? "chromium" : engine];
const limit = setTimeout(() => { console.error("FAILED\n- timed out after 600 s"); process.exit(2); }, 600e3);
const browser = await kind.launch({ headless: true, channel });
console.log(`${engine} ${browser.version()}`);

// huggingface.co's answers for the made-up repository
const CORS = { "access-control-allow-origin": "*", "access-control-expose-headers": "Content-Range, Content-Length, Accept-Ranges, ETag",
  "access-control-allow-headers": "Range", "accept-ranges": "bytes" };
const asHub = (requests) => async (route) => {
  const request = route.request(), found = /^\/made-up\/saluki\/resolve\/main\/(.+)$/.exec(new URL(request.url()).pathname);
  const file = found && files[found[1]], range = /^bytes=(\d+)-(\d*)$/.exec(request.headers().range ?? "");
  requests.push(`${request.method()} ${found?.[1] ?? request.url()}${range ? ` ${range[0]}` : ""}${file ? "" : " (404)"}`);
  if (request.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
  if (!file) return route.fulfill({ status: 404, headers: CORS, body: "" });
  if (!range) return route.fulfill({ status: 200, headers: { ...CORS, "content-type": "application/octet-stream" }, body: request.method() === "HEAD" ? Buffer.alloc(0) : file });
  const from = Number(range[1]), to = Math.min(range[2] === "" ? file.length - 1 : Number(range[2]), file.length - 1);
  if (from >= file.length) return route.fulfill({ status: 416, headers: { ...CORS, "content-range": `bytes */${file.length}` }, body: "" });
  return route.fulfill({ status: 206, headers: { ...CORS, "content-type": "application/octet-stream", "content-range": `bytes ${from}-${to}/${file.length}` }, body: file.subarray(from, to + 1) });
};
// the entry ?hf= makes, turned into what an entry of the list that keeps a GGUF's blocks hands the worker
const asListed = ({ repo, weights }) => {
  const post = Worker.prototype.postMessage;
  Worker.prototype.postMessage = function (data, ...more) {
    if (data?.model?.hf?.repo === repo) {
      const { revision } = data.model.hf;
      data = { ...data, model: { ...data.model, hf: { repo, revision, weights, vocabulary: { repo, revision, tokenizer: "tokenizer.json" } },
        conversion: { dtype: "int8", blocks: true } } };
    }
    return post.call(this, data, ...more);
  };
};

const failures = [];
const visit = async (name, context) => {
  const { query, line, kernels, isolated = true } = VISITS[name], failed = (what) => failures.push(`${name}: ${what}`);
  const page = await context.newPage(), errors = [], warnings = [];
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); else if (message.type() === "warning") warnings.push(message.text()); });
  page.on("pageerror", (error) => errors.push(String(error)));
  served.length = 0;
  const began = Date.now();
  await page.goto(`${site.url}?hf=${encodeURIComponent(REPOSITORY)}&prompt=${encodeURIComponent("w3 w9 w27")}&temperature=0&steps=24${query}`);
  // (the first visit reloads once under the service worker: a wait the reload interrupts starts again)
  const across = async (wait) => {
    for (;;) {
      try { return await wait(); } catch (error) { if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error; }
    }
  };
  await across(() => page.waitForFunction(() => (window.__ready && document.getElementById("run")?.disabled === false) || document.querySelector(".error"), null, { timeout: 300e3 }));
  const ready = await page.evaluate(() => ({ told: window.__ready ?? null, error: document.querySelector(".error .bubble")?.textContent ?? "",
    status: document.getElementById("status-text")?.textContent ?? "", isolated: self.crossOriginIsolated }));
  if (ready.error || !ready.told) {
    failed(`not ready: ${ready.error || ready.status}`);
  } else {
    await page.press("#prompt", "Control+Enter");
    await page.waitForFunction(() => document.querySelector(".model .meta") || document.querySelector(".error"), null, { timeout: 300e3 });
  }
  const after = await page.evaluate(() => ({ text: document.querySelector(".model .bubble")?.textContent ?? "", meta: document.querySelector(".model .meta summary")?.textContent ?? "",
    error: document.querySelector(".error .bubble")?.textContent ?? "", status: document.getElementById("status-text")?.textContent ?? "" }));
  // "<the model> · <the engine's line>[, N threads][, the GPU's word] · Pyodide <version>"
  const [, said = ""] = /^made-up\/saluki · (.*) · Pyodide [\w.]+$/.exec(after.status) ?? [];
  const backend = ready.told?.backend ?? "";
  const fetched = served.filter((file) => /^_astro\/blockkernel_/.test(file)).map((file) => file.replace(/^_astro\/|\.[\w-]+\.wasm$/g, "")).sort();
  console.log(`${name}: ${after.status}`);
  console.log(`${name}: ready in ${((Date.now() - began) / 1000).toFixed(1)} s${ready.told?.fromCache ? ` from what this browser kept (${ready.told.keptIn})` : ready.told?.notKept ? ` (not kept: ${ready.told.notKept})` : ""}, ` +
    `${ready.isolated ? "isolated" : "not isolated"}, ${ready.told?.threads ?? "?"} thread(s); ${after.meta}; wrote ${JSON.stringify(after.text.slice(0, 60))}; ` +
    `the blocks' kernels fetched: ${fetched.join(", ") || "none"}${warnings.length ? `; warnings: ${warnings.join(" | ")}` : ""}`);
  if (after.error) failed(`the page reported: ${after.error}`);
  if (!line.test(backend)) failed(`the engine's line is ${JSON.stringify(backend)}, not ${line}`);
  if (!said.startsWith(backend) || !backend) failed(`the status line ${JSON.stringify(after.status)} does not begin with the engine's line`);
  if (!/tok\/s/.test(after.meta) || !after.text.trim()) failed("no answer with its speed");
  if (ready.isolated !== isolated) failed(`the page is ${ready.isolated ? "" : "not "}isolated`);
  // (a second visit has them from the service worker's copies: the server is not asked)
  if (!VISITS[name].again && kernels !== fetched.length > 0) failed(`the kernels of the blocks ${fetched.length ? "were" : "were not"} fetched`);
  if (errors.length) failed(`console errors: ${errors.join(" | ")}`);
  return { page, ready };
};

for (const name of names.filter((name) => !VISITS[name].again)) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } }), requests = [];
  await context.route("https://huggingface.co/**", asHub(requests));
  await context.addInitScript(asListed, { repo: REPOSITORY, weights: GGUF });
  const { page, ready } = await visit(name, context);
  console.log(`${name}: asked of huggingface.co: ${requests.join("; ")}`);
  if (!requests.some((request) => request.startsWith(`GET ${GGUF}`))) failures.push(`${name}: the GGUF was never asked for`);
  if (ready.told?.fromCache) failures.push(`${name}: a first visit came from a kept conversion`);
  if (name === "isolated" && names.includes("again")) {
    // the second visit: the same browser, the page loaded again. What was kept is the conversion with its blocks
    const asked = requests.length;
    await page.close();
    const second = await visit("again", context);
    const more = requests.slice(asked);
    console.log(`again: asked of huggingface.co: ${more.join("; ") || "nothing"}`);
    if (ready.told?.notKept) console.log(`again: the first visit kept nothing (${ready.told.notKept}): nothing to load again`);
    else if (!second.ready.told?.fromCache) failures.push("again: the second visit did not load what the first kept");
    else if (more.some((request) => request.startsWith(`GET ${GGUF}`))) failures.push("again: the GGUF was fetched again");
  }
  await context.close();
}
clearTimeout(limit);
await Promise.race([browser.close(), new Promise((resolve) => setTimeout(resolve, 15e3))]);
await site.close();
fs.rmSync(work, { recursive: true, force: true });
console.log(failures.length ? `FAILED\n- ${failures.join("\n- ")}` : `ok: a model of llama.cpp's blocks in ${engine}, ${names.length} visit(s)`);
await leave(failures.length ? 1 : 0);
