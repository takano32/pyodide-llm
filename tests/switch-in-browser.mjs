// T.405: another model chosen while one is being fetched from Hugging Face and converted. The conversion stops (the
// page asks huggingface.co for nothing more), the model chosen comes to ready and answers, no error is drawn and none is
// in the console; and the model left half converted, chosen again, is converted whole and answers. Two moments: as the
// first request of the conversion goes out, and once a part of it is converted (T.374.2.1 looked at both with a tool
// thrown away). On this tree's dist/ (npm run build first), with the site's smallest model and a small GGUF of the list
// (SmolLM2 135M Instruct, 145 MB from huggingface.co each time: a new profile keeps nothing).
//
//   node tests/switch-in-browser.mjs [chromium|firefox|webkit]
import * as playwright from "playwright-core";
import { serving } from "./serving.mjs";
import { closed, leave } from "./leave.mjs";

const [engine = "chromium"] = process.argv.slice(2);
const CONVERTED = "hf-smollm2-135m-instruct", SMALL = "stories260K";
const server = await serving();
const browser = await playwright[engine].launch({ headless: true });
const failures = [];
const failed = (what) => { failures.push(what); console.log(`FAILED: ${what}`); };

async function visit(name, moment) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  page.on("dialog", (dialog) => dialog.accept());
  const errors = [], asked = [];
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
  page.on("pageerror", (error) => errors.push(String(error)));
  // (a worker's requests too: the context sees them)
  context.on("request", (request) => { if (new URL(request.url()).hostname.endsWith("huggingface.co")) asked.push(Date.now()); });
  const status = () => page.evaluate(() => document.getElementById("status-text")?.textContent ?? "").catch(() => "");
  // ready: Run is there to press, or an error; across the reload of the first visit (coi.js, T.93)
  const ready = async () => {
    for (;;) {
      try {
        return await page.waitForFunction(() => document.getElementById("run")?.disabled === false || document.querySelector(".error"), null, { timeout: 300e3 });
      } catch (error) {
        if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
      }
    }
  };
  try {
    await page.goto(`${server.url}?model=${SMALL}`);
    await ready();
    await page.selectOption("#model", CONVERTED);
    const began = Date.now();
    if (moment === "first request") {
      while (!asked.length && Date.now() - began < 60e3) await page.waitForTimeout(20);
      if (!asked.length) return failed(`${name}: huggingface.co was not asked for anything in 60 s`);
    } else {
      await page.waitForFunction(() => /(\d+) % converted/.test(document.getElementById("status-text")?.textContent ?? "") &&
        Number(/(\d+) % converted/.exec(document.getElementById("status-text").textContent)[1]) >= 5, null, { timeout: 300e3 });
    }
    const at = await status();
    await page.selectOption("#model", SMALL);
    const switched = Date.now();
    await ready();
    const readySeconds = (Date.now() - switched) / 1000;
    if (await page.$(".error")) failed(`${name}: an error was drawn: ${await page.textContent(".error")}`);
    const said = await status();
    if (!said.startsWith("Andrej Karpathy TinyStories 260K")) failed(`${name}: the status line is not ${SMALL}'s: ${said}`);
    // what was asked of huggingface.co from 3 s after the switch on, over 5 s: nothing (the conversion stopped)
    await page.waitForTimeout(Math.max(0, switched + 8000 - Date.now()));
    const after = asked.filter((time) => time > switched + 3000).length;
    if (after) failed(`${name}: ${after} requests to huggingface.co from 3 s after the switch on`);
    await page.click("#run");
    await page.waitForFunction(() => document.querySelector(".model .meta") || document.querySelector(".error"), null, { timeout: 120e3 });
    const answer = await page.evaluate(() => document.querySelector(".model .meta summary")?.textContent ?? "");
    if (!/tok\/s/.test(answer)) failed(`${name}: no answer of ${SMALL} (${answer || "no speed line"})`);
    console.log(`${name}: switched at "${at}", ${SMALL} ready ${readySeconds.toFixed(1)} s after, ${after} requests to huggingface.co after 3 s; ${answer}`);
    // the model left half converted, chosen again: converted whole, and it answers
    await page.selectOption("#model", CONVERTED);
    await page.waitForFunction((name) => document.getElementById("run")?.disabled === false && (document.getElementById("status-text")?.textContent ?? "").startsWith(name) || document.querySelector(".error"),
      "Hugging Face SmolLM2", { timeout: 600e3 });
    if (await page.$(".error")) return failed(`${name}: chosen again, ${CONVERTED} failed: ${await page.textContent(".error")}`);
    await page.click("#run");
    await page.waitForFunction(() => document.querySelectorAll(".model .meta").length >= 2 || document.querySelector(".error"), null, { timeout: 300e3 });
    const again = await page.evaluate(() => [...document.querySelectorAll(".model .meta summary")].at(-1)?.textContent ?? "");
    if (!/tok\/s/.test(again)) failed(`${name}: chosen again, ${CONVERTED} did not answer (${again || "no speed line"})`);
    console.log(`${name}: chosen again, ${CONVERTED} converted whole and answered: ${again}`);
    if (errors.length) failed(`${name}: console errors: ${errors.join(" | ")}`);
  } finally {
    await closed(() => context.close());
  }
}

try {
  await visit("at the first request", "first request");
  await visit("once a part is converted", "converted");
} catch (error) {
  failed(String(error.stack ?? error));
}
await closed(() => browser.close());
await server.close?.();
console.log(failures.length ? `switch-in-browser: ${failures.length} FAILED` : `switch-in-browser: ok (${engine}: two switches in the middle of a conversion)`);
leave(failures.length ? 1 : 0);
