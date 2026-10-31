// tests/gemma4-in-browser.ts (T.329.2.2): a small made-up Gemma 4 with an embedding of every layer, opened by the built
// page in a real browser by ?hf= (E2B's is 10 GB and runs in CI alone): the worker converts it, keeps it in the origin
// private file system as it comes and reads the table of every layer's inputs from there, not from its memory; a second
// visit loads what was kept and reads the table from the kept file; a second tab of the same model, whose kept file the
// first tab holds, reads the table into its memory.
//
//   npm run build && node tests/gemma4-in-browser.ts [chromium|chrome|msedge|firefox|webkit]
//
// The model is tests/make_gemma4.py's "embedded" (14 layers, the last five reading others' keys and values with an FFN
// twice as wide, inputs of 32 values a layer) as a Hugging Face repository; PYTHON names the Python with NumPy. The tool
// answers https://huggingface.co/made-up/gemma4/resolve/main/<file> itself (Playwright's route, as
// tests/blocks-in-browser.ts does); everything else is the product's. Nothing of the product is for this.
//
// What says the table is read from the file: while the model is loaded, the kept file's model.bin has a sync access
// handle open (the worker's), so that the page cannot open it for writing (createWritable() is refused); in the second
// tab, the console says the table is read into the memory. A last visit is not isolated (?coi=off). It fails where a visit does not get ready, does not answer,
// logs an error, or the file is not held.
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
process.env.PW_EXPERIMENTAL_SERVICE_WORKER_NETWORK_EVENTS ??= "1";
const playwright = await import("playwright-core");
const { leave } = await import("./leave.ts");
const { serving } = await import("./serving.ts");

const [engine = "chromium"] = process.argv.slice(2);
const root = fileURLToPath(new URL("../", import.meta.url));
const work = path.join(root, ".tmp", "gemma4-in-browser");
fs.rmSync(work, { recursive: true, force: true });
execFileSync(process.env.PYTHON ?? "python", [path.join(root, "tests", "make_gemma4.py"), work, "hub", "embedded"], { stdio: ["ignore", "ignore", "inherit"] });
const REPOSITORY = "made-up/gemma4";
const files = Object.fromEntries(fs.readdirSync(work).map((name) => [name, fs.readFileSync(path.join(work, name))]));

const site = await serving();
const channel = ["msedge", "chrome"].includes(engine) ? engine : undefined;
const kind = playwright[channel ? "chromium" : engine];
const limit = setTimeout(() => { console.error("FAILED\n- timed out after 600 s"); process.exit(2); }, 600e3);
const browser = await kind.launch({ headless: true, channel });
console.log(`${engine} ${browser.version()}`);

const CORS = { "access-control-allow-origin": "*", "access-control-expose-headers": "Content-Range, Content-Length, Accept-Ranges, ETag",
  "access-control-allow-headers": "Range", "accept-ranges": "bytes" };
const requests = [];
const asHub = async (route) => {
  const request = route.request(), found = /^\/made-up\/gemma4\/resolve\/main\/(.+)$/.exec(new URL(request.url()).pathname);
  const file = found && files[found[1]], range = /^bytes=(\d+)-(\d*)$/.exec(request.headers().range ?? "");
  requests.push(`${request.method()} ${found?.[1] ?? request.url()}${file ? "" : " (404)"}`);
  if (request.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
  if (!file) return route.fulfill({ status: 404, headers: CORS, body: "" });
  if (!range) return route.fulfill({ status: 200, headers: { ...CORS, "content-type": "application/octet-stream" }, body: request.method() === "HEAD" ? Buffer.alloc(0) : file });
  const from = Number(range[1]), to = Math.min(range[2] === "" ? file.length - 1 : Number(range[2]), file.length - 1);
  return route.fulfill({ status: 206, headers: { ...CORS, "content-type": "application/octet-stream", "content-range": `bytes ${from}-${to}/${file.length}` }, body: file.subarray(from, to + 1) });
};

const failures = [];
// whether the kept model's file is held open (a sync access handle: the worker's), as the page finds it
const held = (page) => page.evaluate(async () => {
  const folder = await (await navigator.storage.getDirectory()).getDirectoryHandle("converted-v1");
  const names = [];
  for await (const [name] of folder.entries()) names.push(name);
  const found = names.find((name) => decodeURIComponent(name).startsWith("made-up/gemma4@"));
  if (!found) return `no kept folder (${names.join(", ")})`;
  try {
    const writable = await (await (await folder.getDirectoryHandle(found)).getFileHandle("model.bin")).createWritable();
    await writable.abort();
    return false;
  } catch (error) {
    return error.name === "NoModificationAllowedError" ? true : `${error.name}: ${error.message}`;
  }
});
const visit = async (name, context, { holds = true, warned, query = "" } = {}) => {
  const failed = (what) => failures.push(`${name}: ${what}`);
  const page = await context.newPage(), errors = [], warnings = [];
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); else if (message.type() === "warning") warnings.push(message.text()); });
  page.on("pageerror", (error) => errors.push(String(error)));
  const began = Date.now();
  await page.goto(`${site.url}?hf=${encodeURIComponent(REPOSITORY)}&prompt=${encodeURIComponent("w3 w9 w27")}&temperature=0&steps=24${query}`);
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
    return page;
  }
  await page.press("#prompt", "Control+Enter");
  await page.waitForFunction(() => document.querySelector(".model .meta") || document.querySelector(".error"), null, { timeout: 300e3 });
  const after = await page.evaluate(() => ({ text: document.querySelector(".model .bubble")?.textContent ?? "", meta: document.querySelector(".model .meta summary")?.textContent ?? "",
    error: document.querySelector(".error .bubble")?.textContent ?? "", status: document.getElementById("status-text")?.textContent ?? "" }));
  const holding = await held(page);
  console.log(`${name}: ${after.status}`);
  console.log(`${name}: ready in ${((Date.now() - began) / 1000).toFixed(1)} s${ready.told.fromCache ? ` from what this browser kept (${ready.told.keptIn})` : ready.told.notKept ? ` (not kept: ${ready.told.notKept})` : ""}, ` +
    `${ready.isolated ? "isolated" : "not isolated"}, ${ready.told.threads ?? "?"} thread(s); ${after.meta}; wrote ${JSON.stringify(after.text.slice(0, 60))}; ` +
    `the kept file held open: ${holding}${warnings.length ? `; warnings: ${warnings.join(" | ")}` : ""}`);
  if (after.error) failed(`the page reported: ${after.error}`);
  if (!/tok\/s/.test(after.meta) || !after.text.trim()) failed("no answer with its speed");
  if (holds && holding !== true) failed(`the kept file is not held open by the worker (${holding}): the table is not read from it`);
  if (warned && !warnings.some((text) => warned.test(text))) failed(`no warning ${warned}`);
  if (!warned && warnings.some((text) => /^kept:/.test(text))) failed("a warning of what is kept");
  // (the browser's line of each file the made-up repository does not have, which this tool answers 404, as huggingface.co would)
  const others = errors.filter((text) => !/^Failed to load resource: the server responded with a status of 404/.test(text));
  if (others.length) failed(`console errors: ${others.join(" | ")}`);
  return page;
};

const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
await context.route("https://huggingface.co/**", asHub);
const first = await visit("first", context);
const fetched = requests.filter((request) => request.startsWith("GET model.safetensors")).length;
if (!fetched) failures.push("first: the weights were never asked for");
// a second tab beside the first, whose worker holds the kept file: this one reads the table into its memory
const asked = requests.length;
const beside = await visit("beside", context, { warned: /^kept: the kept model's file cannot be read from at once here/ });
console.log(`beside: asked of huggingface.co: ${requests.slice(asked).join("; ") || "nothing"}`);
await beside.close();
await first.close();
// the next visit, both tabs closed (their workers' handles let go of): the kept model, its table read from its file
const before = requests.length;
await (await visit("again", context)).close();
if (requests.slice(before).some((request) => request.startsWith("GET model.safetensors"))) failures.push("again: the weights were fetched again");
await context.close();
// and a first visit on a memory that is not shared, in a browser of its own (?coi=off: one thread, the rows read into a
// plain memory)
const plain = await browser.newContext({ viewport: { width: 390, height: 844 } });
await plain.route("https://huggingface.co/**", asHub);
await (await visit("plain", plain, { query: "&coi=off" })).close();
await plain.close();
clearTimeout(limit);
await Promise.race([browser.close(), new Promise((resolve) => setTimeout(resolve, 15e3))]);
await site.close();
fs.rmSync(work, { recursive: true, force: true });
console.log(failures.length ? `FAILED\n- ${failures.join("\n- ")}` : `ok: a Gemma 4's table of every layer's inputs read from where it is kept, in ${engine}`);
await leave(failures.length ? 1 : 0);
