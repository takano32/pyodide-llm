// tests/wllama-in-browser.ts (T519): this page and wllama (llama.cpp compiled to WebAssembly, https://github.com/ngxson/wllama,
// MIT; read from its npm package as it is, nothing of it copied here) on the same GGUF, in one Chromium, by turns. The
// owner (2026-10-15): the comparison may be made, but not shown on the product's pages: it is this tool's and CI's, and
// its numbers go to docs/notes and the ledger. tests/wllama_compare.py is the driver: it fetches the package, writes the
// spec this tool reads and turns the lines it writes into tables.
//
//   npm run build && node tests/wllama-in-browser.ts <spec.json>
//
// The spec (a JSON file): { model: an id of the list, url: the GGUF wllama reads, wllama: the folder of the npm
// package (its esm/), version, prompt, chat: true (wllama renders the GGUF's chat template) or raw: a text with {prompt}
// in it (wllama's raw completion, the same text the page's format writes), kwargs: chat_template_kwargs, query: more
// of the page's address (such as without=relaxed), threads (0: half the logical cores, wllama's own default), tokens (the answer's length asked for), rounds, results: a file to
// append the JSON lines to, timeout: seconds }.
//
// What it does: one browser profile on disk (so that both keep what they fetched across a reload, the page in its
// origin private file system and wllama in its own). This page from dist/ as GitHub Pages serves it (tests/serving.ts),
// wllama's page from the browser's own routing with COOP and COEP (wllama needs the headers for its threads). Each loads the
// model once with the fetch (the page converts as it fetches) and once more after a reload, from what it kept. Then
// one answer each to warm up and to learn the prompt's length (the page's steps count the prompt too), and `rounds`
// answers each, by turns, the order turned round every round (AGENTS.md: the two sides in one process, by turns). Both
// greedy (the page: temperature 0, no penalty; wllama: temperature 0), the CPU alone (wllama: n_gpu_layers 0; the
// browser has no WebGPU here), the same number of threads (the page: ?threads=, which counts the coordinator, as
// wllama's n_threads counts its own), wllama's prompt cache off (cache_prompt false: every round reads the whole prompt).
//
// The speeds are each side's own: the page's line under an answer ("prompt N tokens · x tok/s", "generated N tokens ·
// y tok/s": src/python/engine/generation.py, whose prompt count leaves out the last token of the prompt, whose forward
// gives the first token written), and llama-server's timings (prompt_per_second, predicted_per_second).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import * as playwright from "playwright-core";
import { MODELS } from "../src/models.ts";
import { closed, leave, scratch } from "./leave.ts";
import { serving, TYPES } from "./serving.ts";

const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
if (!MODELS.some((entry) => entry.id === spec.model)) throw new Error(`no model ${spec.model} in src/models.ts`);
const started = Date.now();
const elapsed = () => ((Date.now() - started) / 1000).toFixed(0);
const say = (line) => console.log(`wllama-in-browser: ${spec.model}: ${elapsed()}s ${line}`);
setTimeout(() => {
  console.error(`FAILED\n- timed out after ${spec.timeout}s`);
  leave(2);
}, (spec.timeout ?? 3600) * 1000).unref();

// wllama's page: the package's files, and this harness (window.load, window.ask) at its root
const HARNESS = `<!doctype html><meta charset="utf-8"><title>wllama</title><script type="module">
import { Wllama } from "./esm/index.js";
let wllama;
const quiet = { debug() {}, log() {}, info() {}, warn: (...a) => console.warn(...a), error: (...a) => console.error(...a) };
window.load = async ({ url, threads, context }) => {
  if (wllama) await wllama.exit();
  wllama = new Wllama({ default: "./esm/wasm/wllama.wasm" }, { logger: quiet, suppressNativeLog: true });
  const begun = performance.now();
  let fetched = null;
  await wllama.loadModelFromUrl(url, { n_ctx: context, n_gpu_layers: 0, ...(threads ? { n_threads: threads } : {}),
    progressCallback: ({ loaded, total }) => { if (total && loaded >= total && fetched === null) fetched = performance.now() - begun; } });
  const info = wllama.getLoadedContextInfo();
  return { seconds: (performance.now() - begun) / 1000, fetchSeconds: fetched === null ? null : fetched / 1000,
    threads: wllama.getNumThreads(), multithread: wllama.isMultithread(), isolated: self.crossOriginIsolated,
    jspi: Boolean(WebAssembly.Suspending), mem64: (() => { try { new WebAssembly.Memory({ address: "i64", initial: 1n }); return true; } catch { return false; } })(),
    libllama: Wllama.getLibllamaVersion(), n_batch: info.n_batch, n_ubatch: info.n_ubatch, n_ctx: info.n_ctx };
};
window.ask = async ({ prompt, raw, chat, kwargs, tokens }) => {
  const common = { max_tokens: tokens, temperature: 0, cache_prompt: false };
  const answer = chat
    ? await wllama.createChatCompletion({ messages: [{ role: "user", content: prompt }], ...common, ...(kwargs ? { chat_template_kwargs: kwargs } : {}) })
    : await wllama.createCompletion({ prompt: raw.replace("{prompt}", prompt), ...common });
  const choice = answer.choices[0];
  return { text: choice.message?.content ?? choice.text ?? "", finish: choice.finish_reason, timings: answer.timings ?? null, usage: answer.usage ?? null };
};
window.harnessReady = true;
</script>`;
const root = path.resolve(spec.wllama);
// (no server of its own, tests/serving-check.ts: the browser's own routing answers for an origin nothing listens on,
// 127.0.0.1, which is not the site's localhost: an origin of its own, and a secure context as localhost is)
const wllamaUrl = "http://127.0.0.1:9/";
const HEADERS = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin" };
const answerWllama = (route) => {
  const name = decodeURIComponent(new URL(route.request().url()).pathname).replace(/^\/+/, "");
  if (!name) return route.fulfill({ status: 200, headers: { ...HEADERS, "Content-Type": TYPES[".html"] }, body: HARNESS });
  const file = path.resolve(root, name);
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) return route.fulfill({ status: 404 });
  return route.fulfill({ status: 200, headers: { ...HEADERS, "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream" }, path: file });
};
const site = await serving();

const context = await playwright.chromium.launchPersistentContext(scratch("wllama-profile-"), { headless: true, viewport: { width: 1280, height: 900 }, args: (process.env.WLLAMA_CHROMIUM_ARGS ?? "").split(" ").filter(Boolean) });
await context.route(`${wllamaUrl}**`, answerWllama);
// (which wasm wllama fetched: its own from this routing, or the compat build's from a CDN)
const wasmFetched = new Set();
context.on("request", (request) => { if (/\.wasm(\?|$)|jsdelivr|unpkg/.test(request.url())) wasmFetched.add(request.url().replace(/\?.*/, "")); });
const browserVersion = context.browser()?.version() ?? "";
const cpu = os.cpus()[0]?.model ?? null;
const hint = await (await context.newPage()).evaluate(() => navigator.hardwareConcurrency);
const threads = spec.threads || Math.max(1, Math.floor(hint / 2));
say(`Chromium ${browserVersion}, ${cpu}, ${hint} logical cores, ${threads} threads each, wllama ${spec.version}`);
const record = (entry) => {
  fs.mkdirSync(path.dirname(spec.results), { recursive: true });
  fs.appendFileSync(spec.results, JSON.stringify({ model: spec.model, os: `${os.platform()} ${os.arch()}`, cpu, cores: hint, threads,
    browserVersion, wllama: spec.version, tokensAsked: spec.tokens, query: spec.query ?? "", ...entry }) + "\n");
};
const failures = [];
const watched = (page, side) => {
  page.on("pageerror", (error) => failures.push(`${side}: ${error}`));
  page.on("console", (message) => { if (message.type() === "error") failures.push(`${side}: ${message.text()}`); });
  page.on("dialog", (dialog) => dialog.accept());  // (the page asks before it fetches more than 500 MB)
};

// --- this page
const ours = await context.newPage();
watched(ours, "page");
const query = new URLSearchParams({ model: spec.model, threads: String(threads), temperature: "0", repetition_penalty: "1", presence_penalty: "0", topp: "1",
  ...Object.fromEntries(new URLSearchParams(spec.query ?? "")) });
// (the first visit reloads once under coi.js: a wait the reload breaks starts again)
const acrossReload = async (wait) => {
  for (;;) {
    try {
      return await wait();
    } catch (error) {
      if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
    }
  }
};
const idle = () => acrossReload(() => ours.waitForFunction(() => document.getElementById("run")?.disabled === false || document.querySelector(".error"), null, { timeout: 0 }));
const ready = async () => (await acrossReload(() => ours.evaluate(() => {
  const error = document.querySelector(".error .bubble")?.textContent;
  return { error, ready: window.__ready ?? null, status: document.getElementById("status-text")?.textContent ?? "" };
})));
let begun = Date.now();
await ours.goto(`${site.url}?${query}`);
await idle();
let said = await ready();
if (said.error) throw new Error(`the page: ${said.error}`);
record({ side: "page", what: "load", fetched: true, seconds: (Date.now() - begun) / 1000, parts: said.ready?.seconds ?? null, status: said.status });
say(`page ready with the fetch in ${((Date.now() - begun) / 1000).toFixed(1)}s: ${said.status}`);
begun = Date.now();
await ours.reload();
await idle();
said = await ready();
record({ side: "page", what: "load", fetched: false, seconds: (Date.now() - begun) / 1000, parts: said.ready?.seconds ?? null,
  fromCache: Boolean(said.ready?.fromCache), status: said.status });
say(`page ready again in ${((Date.now() - begun) / 1000).toFixed(1)}s (kept: ${Boolean(said.ready?.fromCache)})`);

/** One answer of the page with `steps` positions (the prompt's and the answer's): its counts, speeds and text. */
async function askPage(steps) {
  await ours.evaluate((steps) => {
    const field = document.getElementById("steps");
    field.value = String(steps);
    field.dispatchEvent(new Event("input"));
  }, steps);
  await ours.fill("#prompt", spec.prompt);
  const before = await ours.evaluate(() => document.querySelectorAll(".model .meta").length);
  await ours.press("#prompt", "Control+Enter");
  await ours.waitForFunction((before) => document.querySelectorAll(".model .meta").length > before || document.querySelector(".error"), before, { timeout: 0 });
  const got = await ours.evaluate(() => {
    const lines = [...([...document.querySelectorAll(".model .meta")].at(-1)?.querySelectorAll("*") ?? [])].map((e) => e.textContent ?? "");
    return { error: document.querySelector(".error .bubble")?.textContent ?? "", text: [...document.querySelectorAll(".model .bubble")].at(-1)?.textContent ?? "",
      summary: [...document.querySelectorAll(".model .meta summary")].at(-1)?.textContent ?? "",
      prompt: lines.find((t) => /^prompt \d/.test(t)) ?? "", generated: lines.find((t) => /^generated \d/.test(t)) ?? "" };
  });
  if (got.error) throw new Error(`the page: ${got.error}`);
  const number = (line, re) => Number(re.exec(line)?.[1]);
  return { promptTokens: number(got.prompt, /^prompt (\d+) tokens/), promptPerSecond: number(got.prompt, /([\d.]+) tok\/s/),
    answerTokens: number(got.generated, /^generated (\d+) tokens/), answerPerSecond: number(got.generated, /([\d.]+) tok\/s/),
    ranThreads: Number(/· (\d+) threads/.exec(got.summary)?.[1] ?? 1), text: got.text, summary: got.summary };
}

// --- wllama
const theirs = await context.newPage();
watched(theirs, "wllama");
const loadWllama = async (fetched) => {
  await theirs.goto(wllamaUrl);
  await theirs.waitForFunction(() => window.harnessReady === true);
  const loaded = await theirs.evaluate((asked) => window.load(asked), { url: spec.url, threads, context: spec.context ?? 4096 });
  record({ side: "wllama", what: "load", fetched, ...loaded });
  say(`wllama ready ${fetched ? "with the fetch" : "again"} in ${loaded.seconds.toFixed(1)}s (${loaded.threads} threads, multithread ${loaded.multithread}, isolated ${loaded.isolated}, jspi ${loaded.jspi}, mem64 ${loaded.mem64}, wasm ${[...wasmFetched].join(" ")}, n_batch ${loaded.n_batch}, llama.cpp ${loaded.libllama})`);
  return loaded;
};
const firstLoad = await loadWllama(true);
// (wllama fetched the file into its cache; a reload reads it from there: a fetch of it again would say so)
const second = await loadWllama(false);
if (second.fetchSeconds !== null && second.fetchSeconds > 5) failures.push(`wllama fetched the model again (${second.fetchSeconds.toFixed(1)}s)`);
if (!firstLoad.multithread && threads > 1) failures.push("wllama ran on one thread");
const askWllama = async (tokens) => {
  const got = await theirs.evaluate((asked) => window.ask(asked), { prompt: spec.prompt, raw: spec.raw ?? null, chat: Boolean(spec.chat), kwargs: spec.kwargs ?? null, tokens });
  const t = got.timings ?? {};
  if (t.cache_n) failures.push(`wllama took ${t.cache_n} tokens of the prompt from its cache`);
  return { promptTokens: t.prompt_n, promptPerSecond: t.prompt_per_second, answerTokens: t.predicted_n, answerPerSecond: t.predicted_per_second,
    cached: t.cache_n ?? null, finish: got.finish, text: got.text };
};

// --- warming up, and the length of the prompt
// (the page's Max tokens is a slider of 16 to the context by 8, and counts the prompt's positions with the answer's: the
// page writes what it is set to less the prompt its line shows, and wllama is asked for as many, a few more than
// spec.tokens. Either may stop before)
const onSlider = (wanted) => Math.max(16, 16 + 8 * Math.ceil((wanted - 16) / 8));
const warmTheirs = await askWllama(8);
const warmOurs = await askPage(onSlider(warmTheirs.promptTokens + 8));
say(`the prompt: ${warmOurs.promptTokens} + 1 tokens on the page, ${warmTheirs.promptTokens} in wllama`);
record({ side: "page", what: "warm", ...warmOurs });
record({ side: "wllama", what: "warm", ...warmTheirs });
const steps = onSlider(warmOurs.promptTokens + spec.tokens);
const tokens = steps - warmOurs.promptTokens;
for (let round = 0; round < spec.rounds; round++) {
  const order = round % 2 ? ["wllama", "page"] : ["page", "wllama"];
  for (const side of order) {
    const got = side === "page" ? await askPage(steps) : await askWllama(tokens);
    record({ side, what: "answer", round, ...got });
    say(`round ${round + 1}, ${side}: prompt ${got.promptTokens} at ${got.promptPerSecond?.toFixed(1)} tok/s, answer ${got.answerTokens} at ${got.answerPerSecond?.toFixed(1)} tok/s`);
  }
}
await closed(() => context.close());
await site.close();
if (failures.length) {
  console.error(`FAILED\n${[...new Set(failures)].map((f) => `- ${f}`).join("\n")}`);
  record({ side: "tool", what: "failures", failures: [...new Set(failures)] });
  await leave(1);
}
say("done");
await leave(0);
