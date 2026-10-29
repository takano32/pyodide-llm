// The page as the owner sees it before a change to it goes out (AGENTS.md: a drawing task shows a phone-width and a
// desktop-width screenshot first). Serves dist/ like tests/e2e.mjs, or takes a deployed site, and saves one PNG per
// scene and width into a directory. Meant for CI (preview.yml): the development machine runs no browser.
//
//   node tests/screenshots.mjs <directory> [url of a deployed site]
//
// The scenes: the page with the smallest model ready; an answer's line under it, closed and open (T.372); the settings sheet open at its engine switches (T.75); the
// benchmark's bubble (T.76); the sheet that opens any repository, and a repository the converter refuses (T.88);
// /benchmark/ before a run, while "Run all" runs the CPU section (T.177: its progress) and after (T.134, whole pages: it is
// a page that scrolls).
import fs from "node:fs";
import path from "node:path";
import * as playwright from "playwright-core";
import { closed, leave } from "./leave.mjs";
import { serving } from "./serving.mjs";

const [directory = "screenshots", deployed] = process.argv.slice(2);
let server, site = deployed;
if (!site) {
  server = await serving();  // (T.391.6: tests/serving.mjs)
  site = server.url;
}
fs.mkdirSync(directory, { recursive: true });

const browser = await playwright.chromium.launch({ headless: true });
const widths = { phone: { width: 390, height: 844 }, desktop: { width: 1280, height: 800 } };
// a wait that the first visit's reload (coi.js, T.93) interrupts starts again
const acrossReload = async (wait) => {
  for (;;) {
    try {
      return await wait();
    } catch (error) {
      if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
    }
  }
};
const shot = (page, name, width) => page.screenshot({ path: path.join(directory, `${name}-${width}.png`) });

// Every scene in a context of its own: a second document in the same tab could not allocate its WebAssembly memory
// (T.96's failure, seen here first with the benchmark as the second document).
const fresh = async (viewport) => {
  const context = await browser.newContext({ viewport });
  const page = await context.newPage();
  page.on("dialog", (dialog) => dialog.accept());
  return { context, page };
};
for (const [width, viewport] of Object.entries(widths)) {
  let { context, page } = await fresh(viewport);
  const ready = () => acrossReload(() => page.waitForFunction(() => document.getElementById("run")?.disabled === false || document.querySelector(".error"), null, { timeout: 600000 }));

  // 1. the page, the smallest model ready
  await page.goto(`${site}?model=stories260K`);
  await ready();
  await shot(page, "home", width);
  // T.128: the order of the list. A browser draws an open <select> outside the page, where a screenshot does not
  // reach, so the select is shown as a list box of every entry (the order is the same; its look is not the phone's)
  await page.evaluate(() => {
    const select = document.getElementById("model");
    select.size = select.querySelectorAll("option, optgroup").length;
  });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(directory, `list-${width}.png`), fullPage: true });
  await page.evaluate(() => { document.getElementById("model").size = 0; });

  // T.372: an answer's line, with the seed's button beside its summary (not in it): closed, after the button is
  // pressed (the line stays closed, and the next run is to use the seed), and open. (stories260K, the one model
  // preview.yml builds, is greedy, and a greedy answer has no seed: the URL sets a temperature)
  {
    const answering = await fresh(viewport);
    await answering.page.goto(`${site}?model=stories260K&temperature=0.7`);
    await acrossReload(() => answering.page.waitForFunction(() => document.getElementById("run")?.disabled === false || document.querySelector(".error"), null, { timeout: 600000 }));
    await answering.page.fill("#prompt", "Once upon a time");
    await answering.page.click("#run");
    await answering.page.waitForFunction(() => document.querySelector(".model .meta") || document.querySelector(".error"), null, { timeout: 600000 });
    await answering.page.waitForTimeout(300);
    await shot(answering.page, "answer", width);
    const before = await answering.page.evaluate(() => ({ seed: Boolean(document.querySelector(".model .meta-line .seed")), inSummary: document.querySelectorAll("summary button, summary a, summary input").length }));
    if (!before.seed) { console.log(`${width}: NO seed's button beside the answer's line`); process.exitCode = 1; }
    else {
      await answering.page.click(".model .meta-line .seed");
      const open = await answering.page.evaluate(() => document.querySelector(".model .meta-line details").open);
      console.log(`${width}: after the seed's button, the line is ${open ? "OPEN" : "closed"}; interactive elements in a summary: ${before.inSummary}`);
      if (open || before.inSummary) process.exitCode = 1;
      await shot(answering.page, "answer-seed", width);
      await answering.page.click(".model .meta-line summary");
      await answering.page.waitForTimeout(300);
      await shot(answering.page, "answer-open", width);
    }
    await closed(() => answering.context.close());
  }

  // 2. the settings sheet, the engine's switches open (T.75)
  await page.click("#settings-open");
  await page.evaluate(() => { document.getElementById("engine").open = true; });
  await page.waitForTimeout(300);
  await shot(page, "settings-engine", width);
  // T.119 (5): Engine and More both open: the sheet stays under the top of the screen and scrolls
  await page.evaluate(() => { for (const part of document.querySelectorAll("#settings details")) part.open = true; });
  await page.waitForTimeout(300);
  await shot(page, "settings-both", width);
  await page.keyboard.press("Escape");

  // 3. the sheet that opens any repository, with a name typed (T.88)
  await page.selectOption("#model", "hf-other");
  await page.waitForTimeout(300);
  await page.fill("#repository-name", "Qwen/Qwen2.5-0.5B-Instruct");
  await shot(page, "repository", width);
  // T.119 (1): a wrong revision is marked on the revision's field, not the name's
  await page.fill("#repository-revision", "not a revision!");
  await page.click("#repository-form button[type=submit]");
  await page.waitForTimeout(300);
  await shot(page, "repository-revision", width);
  const marks = await page.evaluate(() => ["repository-name", "repository-revision"].map((id) => document.getElementById(id).validationMessage));
  console.log(`${width}: the name's field says "${marks[0]}", the revision's "${marks[1]}"`);
  await page.keyboard.press("Escape");

  // 4. the benchmark's bubble (T.76)
  await closed(() => context.close());
  ({ context, page } = await fresh(viewport));
  // T.214: at the phone's width the browser does not say its memory, as an iPhone's Safari does not: the round without
  // the kernels must be a row that says why (and nothing measured), where the desktop's runs both
  const unsaid = width === "phone";
  if (unsaid) await page.addInitScript(() => Object.defineProperty(Navigator.prototype, "deviceMemory", { get: () => undefined, configurable: true }));
  await page.goto(`${site}?model=stories260K&bench=1`);
  await acrossReload(() => page.waitForFunction(() => window.__bench || document.querySelector(".error"), null, { timeout: 600000 }));
  await page.waitForTimeout(300);
  await shot(page, "bench", width);
  const benched = await page.evaluate(() => window.__bench?.rows ?? []);
  console.log(`${width}: ?bench=1${unsaid ? " without deviceMemory" : ""}: ${benched.map((row) => `${row.name} ${row.skip !== undefined ? `skipped (${row.skip})` : `${row.speed?.toFixed(1)} tok/s`}`).join(", ")}`);
  // (T.590: by name: since T.555 the first round runs twice at the head and once at the end, so "everything" is the
  // first two rows and the last)
  const all = benched.find((row) => row.name === "everything"), plain = benched.find((row) => row.name === "without the kernels");
  if (!(all?.speed > 0) || all.skip !== undefined || !plain || (plain.skip !== undefined) !== unsaid ||
      (unsaid ? !/does not say how much memory/.test(plain.skip) || plain.speed !== undefined : !(plain.speed > 0))) {
    throw new Error(`${width}: ?bench=1 did not ${unsaid ? "skip the round without the kernels, in words" : "run both rounds"}`);
  }

  // 5. a repository the converter refuses, in words (T.88)
  await closed(() => context.close());
  ({ context, page } = await fresh(viewport));
  await page.goto(`${site}?hf=microsoft/phi-1_5`);
  await acrossReload(() => page.waitForFunction(() => document.querySelector(".error"), null, { timeout: 600000 }));
  await page.waitForTimeout(300);
  await shot(page, "refused", width);

  // T.119 (2): a gated repository, in words
  await closed(() => context.close());
  ({ context, page } = await fresh(viewport));
  await page.goto(`${site}?hf=meta-llama/Llama-3.2-1B`);
  await acrossReload(() => page.waitForFunction(() => document.querySelector(".error"), null, { timeout: 600000 }));
  await page.waitForTimeout(300);
  await shot(page, "gated", width);

  // T.134: /benchmark/, before a run and after every section ran (the smallest model, a small file for the storage)
  await closed(() => context.close());
  ({ context, page } = await fresh(viewport));
  await page.goto(`${site}benchmark/`);
  await acrossReload(() => page.waitForFunction(() => document.querySelector("section .head button"), null, { timeout: 120000 }));
  await page.screenshot({ path: path.join(directory, `benchmark-${width}.png`), fullPage: true });
  await page.goto(`${site}benchmark/?run=all&model=stories260K&size=64`);
  // T.177: the CPU section under way, its stage counted and its seconds going (the window, not the whole page)
  await acrossReload(() => page.waitForFunction(() => {
    const box = document.querySelector('section[data-section="cpu"] .progress');
    return box && !box.hidden && /\/ \d+ ·/.test(box.querySelector(".stage").textContent)
      && box.querySelector(".elapsed").textContent !== "0 s";
  }, null, { timeout: 300000, polling: 100 }));
  await page.locator('section[data-section="cpu"]').scrollIntoViewIfNeeded();
  await shot(page, "benchmark-running", width);
  await acrossReload(() => page.waitForFunction(() => window.__benchmark?.done, null, { timeout: 900000 }));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(directory, `benchmark-done-${width}.png`), fullPage: true });
  console.log(`${width}: /benchmark/ ${await page.evaluate(() => Object.entries(window.__benchmark.results).map(([n, r]) => `${n} ${r.status}`).join(", "))}`);
  // the page itself never scrolls sideways at a phone's width: a wide table scrolls inside its section
  console.log(`${width}: /benchmark/ is ${await page.evaluate(() => document.documentElement.scrollWidth)} px wide in a window of ${viewport.width}`);
  // T.186: "This browser" at 360 px (a narrow Android): its table's values wrap instead of running out on the right
  if (width === "phone") {
    await page.setViewportSize({ width: 360, height: viewport.height });
    await page.locator('section[data-section="device"]').scrollIntoViewIfNeeded();
    await shot(page, "benchmark-device", 360);
    console.log(`360: "This browser"'s table is ${await page.evaluate(() => {
      const out = document.querySelector('section[data-section="device"] table');
      return out ? `${out.scrollWidth} px wide in a box of ${out.parentElement.clientWidth}` : "not there";
    })}, the page ${await page.evaluate(() => document.documentElement.scrollWidth)} px`);
    await page.setViewportSize(viewport);
  }

  // T.119 (6): a conversion under way says what has arrived, how fast, and what is converted
  await closed(() => context.close());
  ({ context, page } = await fresh(viewport));
  await page.goto(`${site}?hf=HuggingFaceTB/SmolLM2-360M-Instruct`);
  await acrossReload(() => page.waitForFunction(() => /MB\/s/.test(document.getElementById("status-text")?.textContent ?? ""), null, { timeout: 600000 }));
  await shot(page, "converting", width);
  console.log(`${width}: ${await page.evaluate(() => document.getElementById("status-text").textContent)}`);
  await closed(() => context.close());
}
await closed(() => browser.close());
server?.close();
console.log(fs.readdirSync(directory).join("\n"));
// (T.141, T.391.6: the tool ends here, whatever a browser that did not close left behind)
await leave(0);
