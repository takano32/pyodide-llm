// tests/wide-in-browser.mjs (T506): what a browser does when a worker of the site asks it for a large WebAssembly memory,
// as the model's worker does for a model past 4 GiB (worker/weights.ts: one memory with the whole checkpoint as its
// first size, 64-bit, shared where the page is isolated, then written from end to end as the weights come).
//
//   node tests/wide-in-browser.mjs [--site <url of a deployed site>] [--only <a word of the probes' names>] [browser ...]
//
// Each probe runs in a page of its own (a probe may take the page's process down: that is its answer) and in a worker
// (a blob's, under the site's isolation: the first visit reloads once for coi.js). A line a probe: what the memory was,
// the milliseconds of each step, and how it ended: "made", "filled", an exception's words, "the page crashed", or
// "no answer in N s" (the worker hangs). On a CI runner (WebKit does not run on the development machine):
//
//   node tests/ci.mjs run tests.yml only_extra=true extra="npm ci && bash tests/apt-timeouts.sh && npx playwright-core install --with-deps webkit chromium && node tests/wide-in-browser.mjs webkit chromium" --grep "^wide" --ref <branch>
//
// The default site is the deployed one (the probes need only its origin and its service worker). Mind the memory: the
// probes that fill write up to 8 GiB.
import os from "node:os";
import * as playwright from "playwright-core";
import { closed, leave } from "./leave.mjs";

const args = process.argv.slice(2);
const option = (name) => (args.includes(name) ? args.splice(args.indexOf(name), 2)[1] : undefined);
const site = option("--site") ?? "https://takano32.github.io/pyodide-llm/";
const only = option("--only");
const channels = { chrome: "chrome", msedge: "msedge" };
const SECONDS = 150;

// name, then what the worker is asked: gib (the first size), wide (64-bit), shared, fill (write every byte, 8 MiB at a
// time, as the weights arrive), grow (from one page, a GiB at a time)
const PROBES = [
  ["32-bit, 2 GiB", { gib: 2 }],
  ["64-bit, 1 GiB", { gib: 1, wide: true }],
  ["64-bit, 1 GiB, filled", { gib: 1, wide: true, fill: true }],
  ["64-bit, 4.5 GiB", { gib: 4.5, wide: true }],
  ["64-bit, 4.5 GiB, filled", { gib: 4.5, wide: true, fill: true }],
  ["64-bit, 4.5 GiB, shared, filled", { gib: 4.5, wide: true, shared: true, fill: true }],
  ["64-bit, 8 GiB", { gib: 8, wide: true }],
  ["64-bit, 8 GiB, shared", { gib: 8, wide: true, shared: true }],
  ["64-bit, 8 GiB, grown from a page", { gib: 8, wide: true, grow: true }],
  ["64-bit, 8 GiB, filled", { gib: 8, wide: true, fill: true }],
  ["64-bit, 8 GiB, shared, filled", { gib: 8, wide: true, shared: true, fill: true }],
  ["64-bit, 12 GiB", { gib: 12, wide: true }],
].filter(([name]) => !only || name.includes(only));

// in the page: a worker of a blob runs the probe and says each step; the page gathers what it said
function inPage({ asked, seconds }) {
  const probe = (asked) => {
    const began = performance.now();
    const say = (step) => postMessage({ ms: Math.round(performance.now() - began), ...step });
    try {
      const PAGE = 65536, pages = Math.ceil(asked.gib * 2 ** 30 / PAGE);
      const size = (n) => (asked.wide ? BigInt(n) : n);
      const memory = new WebAssembly.Memory({ initial: size(asked.grow ? 1 : pages), ...(asked.shared ? { maximum: size(pages + 16384), shared: true } : {}),
        ...(asked.wide ? { address: "i64" } : {}) });
      say({ made: memory.buffer.byteLength, buffer: memory.buffer.constructor.name });
      if (asked.grow) {
        for (let now = 1; now < pages;) {
          const more = Math.min(16384, pages - now);
          memory.grow(size(more));
          now += more;
          if (now % 32768 < 16384) say({ grown: memory.buffer.byteLength });
        }
        say({ grown: memory.buffer.byteLength });
      }
      if (asked.fill) {
        const chunk = new Uint8Array(8 << 20).fill(7), length = memory.buffer.byteLength;
        for (let at = 0; at + chunk.length <= length; at += chunk.length) {
          new Uint8Array(memory.buffer, at, chunk.length).set(chunk);
          if (at % 2 ** 30 === 0) say({ filled: at });
        }
        const read = new Uint8Array(memory.buffer, length - (8 << 20), 16);
        say({ filled: length, last: read[0] });
      }
      say({ done: true });
    } catch (error) {
      say({ error: String(error) });
    }
  };
  return new Promise((resolve) => {
    const said = [];
    const end = (how) => resolve({ isolated: self.crossOriginIsolated, said, ...how });
    let worker;
    try {
      worker = new Worker(URL.createObjectURL(new Blob([`(${probe})(${JSON.stringify(asked)})`], { type: "text/javascript" })));
    } catch (error) {
      return end({ error: `no worker (${error})` });
    }
    worker.onmessage = ({ data }) => {
      said.push(data);
      if (data.done || data.error) end({});
    };
    worker.onerror = (event) => end({ error: `the worker failed (${event.message})` });
    setTimeout(() => end({ silent: seconds }), seconds * 1000);
  });
}

const line = (text) => console.log(`wide: ${text}`);
const within = (promise, seconds, otherwise) => Promise.race([promise, new Promise((resolve) => setTimeout(() => resolve(otherwise), seconds * 1000))]);
let failed = false;
for (const engine of args.length ? args : ["chromium"]) {
  const browser = await (channels[engine] ? playwright.chromium.launch({ channel: channels[engine] }) : playwright[engine].launch());
  line(`${engine} ${browser.version()} on ${os.cpus()[0].model} (${os.platform()} ${process.arch}), ${Math.round(os.totalmem() / 2 ** 30)} GiB of memory, from ${site}`);
  try {
    for (const [name, asked] of PROBES) {
      const page = await browser.newPage();
      let crashed = false;
      const crash = new Promise((resolve) => page.on("crash", () => {
        crashed = true;
        resolve({ crashed: true, said: [] });
      }));
      const began = Date.now();
      let found;
      try {
        // (T511.6: the bare page opens the default, 379 MB from Hugging Face: the probes take the smallest model of the site)
        await page.goto(`${site}?model=stories260K`);
        // the first visit reloads once, under the service worker that isolates the page: asked again until it has
        for (let attempt = 0; attempt < 30 && !crashed; attempt++) {
          if (await page.evaluate(() => self.crossOriginIsolated).catch(() => false)) break;
          await page.waitForTimeout(1000);
        }
        const setUp = (Date.now() - began) / 1000;
        const answer = page.evaluate(inPage, { asked, seconds: SECONDS }).catch((error) => ({ thrown: String(error.message ?? error).split("\n")[0], said: [] }));
        found = await within(Promise.race([answer, crash]), SECONDS + 20, { mute: SECONDS + 20, said: [] });
        found.setUp = setUp;
      } catch (error) {
        found = { thrown: String(error.message ?? error).split("\n")[0], said: [] };
      }
      const steps = found.said.map((step) => `${step.ms} ms ${Object.entries(step).filter(([key]) => key !== "ms").map(([key, value]) =>
        `${key} ${typeof value === "number" && value > 2 ** 20 ? `${(value / 2 ** 30).toFixed(2)} GiB` : value}`).join(" ")}`);
      const last = found.said.at(-1) ?? {};
      const end = found.crashed ? "the page crashed" : found.mute ? `the page gave no answer in ${found.mute} s` : found.thrown ? `the page threw: ${found.thrown}`
        : found.silent ? `no answer in ${found.silent} s (the worker hangs)` : found.error ? found.error : last.error ? `refused: ${last.error}` : "done";
      line(`  ${name}${found.isolated === false ? " (the page is not isolated)" : ""}: ${end} [${steps.join("; ")}] (${((Date.now() - began) / 1000).toFixed(1)} s)`);
      if (found.crashed || found.mute || found.silent) failed = true;
      await within(page.close().catch(() => {}), 10);
    }
  } finally {
    await closed(() => browser.close());
  }
}
// (what a browser answers is the finding: a crash or a hang is said by the exit, a refusal is an answer)
await leave(failed ? 1 : 0);
