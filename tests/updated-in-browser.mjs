// T.367.4: a page left open across a deployment, in a real browser. The build in dist/ is served as GitHub Pages serves
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
// - a model that goes on without a file (the kernels, forward.js, a software thread, the GPU's worker: NumPy, fewer
//   threads, the CPU) goes on and answers, and (T.439) where the file is gone its status line ends with "the site was
//   updated: reload for full speed", after what it said before, and still does with the next model of the visit;
//   where the file failed otherwise (a 500, a dropped line, gone alone with the module that names it there) the line
//   is what it was; a visit where nothing fails says nothing and asks no HEAD.
// It does not see: a GPU at work (its worker is asked for wherever the browser's workers have WebGPU, and that it
// does not start is all this needs; where they have none the case says that it saw nothing), a real deployment.
//
// UPDATED_SHOTS=<folder>: a picture of the page for each case whose status line ends with the sentence (390 px wide).
//
// The model is stories260K (1 MB), and a conversion is the folder of tests/make_hf_fixture.py, as tests/e2e.mjs's hf.
// Every case has a browser context of its own (its own service worker and copies). Pyodide comes from its CDN each time.
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import * as playwright from "playwright-core";
import { UPDATED, SLOWER } from "../src/runtime/updated.ts";
import { closed, leave, scratch } from "./leave.mjs";
import { serving } from "./serving.mjs";

const [engine = "chromium", ...only] = process.argv.slice(2);
const repository = fileURLToPath(new URL("../", import.meta.url));  // (T.498: a path on Windows and with spaces too)

// ---- the server: what becomes of a request is the case's rule(name under the site, method): undefined (served),
// a status, or "drop" (the connection ends without an answer)
let rule = () => undefined;
const requests = [];
// (T.391.6: the serving is tests/serving.mjs's, and the rule is this tool's say in it)
const server = await serving(undefined, { instead: ({ name, method }) => {
  const verdict = rule(name, method);
  requests.push(`${method} ${name}${verdict === undefined ? "" : ` -> ${verdict}`}`);
  return verdict;
} });
const site = server.url;
const built = (name) => name.startsWith("_astro/");
const family = (...names) => (name) => built(name) && names.some((one) => name.startsWith(`_astro/${one}.`));

const fixture = scratch("hf-fixture-");
execFileSync("python3", [`${repository}tests/make_hf_fixture.py`, fixture]);
const folder = fs.readdirSync(fixture).map((name) => path.join(fixture, name));

const channel = ["msedge", "chrome"].includes(engine) ? engine : undefined;
const kind = playwright[channel ? "chromium" : engine];
const browser = await kind.launch({ headless: true, channel });
console.log(`${engine} ${browser.version()}`);

// T.93: the first visit reloads once under the service worker; a wait the reload interrupts starts again
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
async function modelPage(context, model = "stories260K") {
  const page = await context.newPage();
  await page.goto(`${site}?model=${model}`);
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
// T.439: the status line of a model that goes on without a file: with the sentence as its last part, after the line it
// would have been (the model's name, how it runs, Pyodide's version), or without a word of it
const shots = process.env.UPDATED_SHOTS;
async function saysSlower(name, page, status) {
  expect(name, status.endsWith(` · ${SLOWER}`) && / · Pyodide [\d.]+ · /.test(status) && status.split(SLOWER).length === 2, `the status line does not end with the sentence, once, after what it said: ${JSON.stringify(status)}`);
  // (a phone's width: the line wraps, and the page does not scroll sideways for it)
  const [wide, inner] = await page.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
  expect(name, wide <= inner, `the page is ${wide} px wide in a window of ${inner}`);
  if (shots) {
    fs.mkdirSync(shots, { recursive: true });
    await page.screenshot({ path: path.join(shots, `${engine}-${name.replace(/[^a-z0-9]+/gi, "-").slice(0, 80)}.png`) });
  }
}
const saysNothing = (name, status) => expect(name, / · Pyodide [\d.]+$/.test(status) && !/updated|reload/i.test(status), `the status line says more than it did: ${JSON.stringify(status)}`);
const wrote = (name, said) => {
  expect(name, said.error === undefined, `an error was shown: ${said.error}`);
  expect(name, said.text.startsWith("Once upon a time"), `the model wrote ${JSON.stringify(said.text.slice(0, 60))}`);
};
const KERNELS = ["simdkernel", "simdkernel_relaxed", "simdkernel_plain", "simdkernel_relaxed_plain", "simdkernel_shared", "simdkernel_relaxed_shared",
  "simdkernel_plain64", "simdkernel_relaxed_plain64", "simdkernel_shared64", "simdkernel_relaxed_shared64"];
// a model page whose files `fails` (names of _astro/) are answered `how` from the start; `head`: what a HEAD for a
// file of the build is answered (404: the build is gone, as after a deployment; undefined: it is there)
const goingOn = (fails, how, head, check, model) => async (context, name) => {
  rule = (file, method) => (method === "HEAD" && built(file) ? head : family(...fails)(file) ? how : undefined);
  const page = await modelPage(context, model);
  await ready(page);
  const said = await answered(page);
  console.log(`  the page: ${JSON.stringify({ ...said, text: said.text.slice(0, 60) })}`);
  wrote(name, said);
  expect(name, requests.some((line) => fails.some((one) => line.startsWith(`GET _astro/${one}.`) && line.endsWith(` -> ${how}`))), `none of ${fails} was asked for: the case saw nothing`);
  await check(name, page, said);
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
  // T.475: a model past 4 GB (here any, ?wide=on) on a page whose 64-bit kernels are gone is refused with the sentence,
  // not with "no 64-bit WebAssembly memory" (the page is old, not the browser)
  "model page, the 64-bit kernels gone, and a model on a 64-bit memory": async (context, name) => {
    rule = (file, method) => (method === "HEAD" && built(file) ? 404 : family("simdkernel_plain64", "simdkernel_relaxed_plain64", "simdkernel_shared64", "simdkernel_relaxed_shared64")(file) ? 404 : undefined);
    const page = await modelPage(context, "stories260K&wide=on");
    await ready(page);
    const said = await page.evaluate(lastError);
    console.log(`  the page: ${JSON.stringify(said)}`);
    expect(name, requests.some((line) => /^GET _astro\/simdkernel_plain64\..* -> 404$/.test(line)), "the 64-bit kernels were not asked for: the case saw nothing");
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
  // ---- what the engine goes on without: the model answers, and where the file is gone the status line says why it is slow
  "model page, the kernels gone from the start": goingOn(KERNELS, 404, 404, async (name, page, said) => {
    expect(name, /NumPy/.test(said.status), `the status line does not say NumPy: ${said.status}`);
    await saysSlower(name, page, said.status);
    // (a built file's own answer says it: nothing is asked beside)
    expect(name, !requests.some((line) => line.startsWith("HEAD ")), `a HEAD was asked for: ${requests.filter((line) => line.startsWith("HEAD "))}`);
    // and it is the page that is old, not the model: the next model of the visit says it too, though nothing fails now
    rule = () => undefined;
    await page.selectOption("#model", "tiny-lm");
    // (T.583: the name has its maker before it since T.523, "SB Intuitions tiny-lm 29M": not at the line's start)
    await page.waitForFunction(() => /\btiny-lm .* · Pyodide /.test(document.getElementById("status-text").textContent) && document.getElementById("run")?.disabled === false, null, { timeout: 180000 });
    await saysSlower(`${name} (the next model)`, page, await page.evaluate(() => document.getElementById("status-text").textContent));
  }),
  "model page, the kernels answered 500": goingOn(KERNELS, 500, 404, (name, page, said) => {
    expect(name, /NumPy/.test(said.status), `the status line does not say NumPy: ${said.status}`);
    saysNothing(name, said.status);
  }),
  "model page, the kernels drop the line": goingOn(KERNELS, "drop", 404, (name, page, said) => {
    expect(name, /NumPy/.test(said.status), `the status line does not say NumPy: ${said.status}`);
    saysNothing(name, said.status);
  }),
  "model page, the engine's module gone from the start, with the worker that names it": goingOn(["forward"], 404, 404, async (name, page, said) => {
    expect(name, /NumPy/.test(said.status), `the status line does not say NumPy: ${said.status}`);
    await saysSlower(name, page, said.status);
  }),
  "model page, the engine's module answered 500 (the worker is there)": goingOn(["forward"], 500, undefined, (name, page, said) => {
    expect(name, /NumPy/.test(said.status), `the status line does not say NumPy: ${said.status}`);
    saysNothing(name, said.status);
  }),
  "model page, the engine's module gone alone (the worker that names it is there: no deployment does that)": goingOn(["forward"], 404, undefined, (name, page, said) => {
    expect(name, /NumPy/.test(said.status), `the status line does not say NumPy: ${said.status}`);
    saysNothing(name, said.status);
  }),
  "model page, a software thread gone from the start, with the worker that names it": goingOn(["helper"], 404, 404, async (name, page, said) => {
    expect(name, !/NumPy/.test(said.status) && !/threads/.test(said.status), `the status line says NumPy, or threads: ${said.status}`);
    await saysSlower(name, page, said.status);
  }),
  "model page, a software thread answered 500 (the worker is there)": goingOn(["helper"], 500, undefined, (name, page, said) => {
    expect(name, !/NumPy/.test(said.status) && !/threads/.test(said.status), `the status line says NumPy, or threads: ${said.status}`);
    saysNothing(name, said.status);
  }),
  "model page, a software thread drops the line, as does every HEAD": goingOn(["helper"], "drop", "drop", (name, page, said) => saysNothing(name, said.status)),
  // (tiny-lm: int8, which forward.js puts on a GPU; where the browser's workers have no WebGPU nothing asks for the worker)
  ...Object.fromEntries([["gone from the start, with the worker that names it", 404, 404], ["answered 500 (the worker is there)", 500, undefined]].map(([what, how, head]) =>
    [`model page, the GPU's worker ${what}`, async (context, name) => {
      rule = (file, method) => (method === "HEAD" && built(file) ? head : family("gpu")(file) ? how : undefined);
      const page = await modelPage(context, "tiny-lm");
      await ready(page);
      if (!requests.some((line) => line.startsWith("GET _astro/gpu."))) {
        console.log(`  SAW NOTHING: this browser's workers asked for no GPU's worker (${await page.evaluate(() => document.getElementById("status-text").textContent)})`);
        return;
      }
      // (forward.js hears the worker's error and says so; the word of the update comes after one HEAD)
      await page.waitForFunction(() => /did not start/.test(document.getElementById("status-text").textContent), null, { timeout: 60000 });
      if (how === 404) await page.waitForFunction((slower) => document.getElementById("status-text").textContent.endsWith(slower), SLOWER, { timeout: 30000 }).catch(() => {});
      const said = await answered(page);
      console.log(`  the page: ${JSON.stringify({ ...said, text: said.text.slice(0, 60) })}`);
      expect(name, said.error === undefined && said.text.length > 0, `an error was shown, or nothing written: ${said.error}`);
      expect(name, /prompts on the CPU \(the GPU's worker did not start/.test(said.status), `the status line does not say the GPU's worker did not start: ${said.status}`);
      if (how === 404) await saysSlower(name, page, said.status);
      else saysNothing(name, said.status);
    }])),
  "model page, nothing gone": async (context, name) => {
    const page = await modelPage(context);
    await ready(page);
    const said = await answered(page);
    console.log(`  the page: ${JSON.stringify({ ...said, text: said.text.slice(0, 60) })}`);
    wrote(name, said);
    saysNothing(name, said.status);
    expect(name, !requests.some((line) => line.startsWith("HEAD ")), `a HEAD was asked for on a visit where nothing failed: ${requests.filter((line) => line.startsWith("HEAD "))}`);
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
  // T.475: the model section, whose worker goes on without its kernels (slower): its rounds would time that
  "benchmark, the model section whose kernels are gone": measuring("model", (name) => (family("simdkernel", "simdkernel_relaxed", "simdkernel_plain", "simdkernel_relaxed_plain", "simdkernel_shared", "simdkernel_relaxed_shared")(name) ? 404 : undefined), sentenceOfSection),
  "benchmark, the CPU section whose kernels are gone": measuring("cpu", (name) => (/^_astro\/simdkernel/.test(name) ? 404 : undefined), sentenceOfSection),
  "benchmark, the CPU section whose ceilings' loops are gone": measuring("cpu", (name) => (/^_astro\/ceilings/.test(name) ? 404 : undefined), sentenceOfSection),
  "benchmark, the CPU section whose software threads are gone, with the worker's own file": measuring("cpu",
    (name, method) => (family("helper")(name) || (family("benchmark-sections")(name) && method === "HEAD") ? 404 : undefined), sentenceOfSection),
  "benchmark, the CPU section whose reading threads are gone, with the worker's own file": measuring("cpu",
    (name, method) => (family("benchmark-reader")(name) || (family("benchmark-sections")(name) && method === "HEAD") ? 404 : undefined), sentenceOfSection),
  // (T.440: a file that did not come is said so, in built()'s words: no "no WebAssembly SIMD in this browser" any more)
  "benchmark, the CPU section whose kernels are answered 500": measuring("cpu", (name) => (/^_astro\/simdkernel/.test(name) ? 500 : undefined),
    (name, said) => {
      expect(name, said.status !== "none" && /Could not fetch simdkernel\S*: 500/.test(said.markdown), `the section said ${JSON.stringify(said.markdown.slice(0, 120))}`);
    }),
  "benchmark, the CPU section whose ceilings' loops are answered 500": measuring("cpu", (name) => (/^_astro\/ceilings/.test(name) ? 500 : undefined),
    (name, said) => {
      expect(name, /Could not fetch ceilings\S*: 500/.test(said.markdown), `the section said ${JSON.stringify(said.markdown.slice(0, 160))}`);
    }),
};

for (const [name, run] of Object.entries(CASES)) {
  if (only.length && !only.some((word) => name.includes(word))) continue;
  console.log(`- ${name}`);
  const before = failures.length;
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  // (T.380.2, UPDATED_CONSOLE: the console's warnings beside each case: what a fallback says of the failure it took)
  if (process.env.UPDATED_CONSOLE) context.on("console", (message) => { if (message.type() === "warning") console.log(`  console: ${message.text().slice(0, 300)}`); });
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
  await closed(() => context.close());
}
fs.rmSync(fixture, { recursive: true, force: true });
console.log(failures.length ? `FAILED\n${failures.map((line) => `- ${line}`).join("\n")}` : "updated-in-browser: ok");
// (a browser's close() may never return, T.141: the process ends by itself)
await closed(() => browser.close());
server.close();
await leave(failures.length ? 1 : 0);
