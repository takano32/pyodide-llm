// tests/markdown-in-browser.mjs (T554, T556): how the built model page draws an answer, in a real browser, with a
// made-up model worker in place of the page's (no Pyodide, no model): the worker says it is ready with a format that
// leaves the answer in <think>, and answers with tokens of Markdown that holds what must not be drawn.
//
//   npm run build && node tests/markdown-in-browser.mjs [chromium|firefox|webkit|chrome]
//
// Visits: the page as a visitor opens it (under the service worker: the libraries come from jsDelivr through it), the
// same with ?coi=off (no service worker), and one where jsDelivr's packages do not answer (with ?coi=off). It sees: the thinking stays
// plain text (its ** too); headings, lists, bold, italic, code and a table are drawn; the model's HTML, links, images
// and an onerror are not (no such element, no attribute but align and start, nothing ran) and are said as the text it
// wrote; a block drawn earlier is the same element at the end (only the blocks that changed are made again: T556); the
// answer is drawn far fewer times than it has tokens (T554), and whole at its end; without the libraries the answer is
// plain text, whole, with one line in the console that says so; no console error on any visit.
// It does not see: a real model, offline (the service worker's copies: e2e's E2E_OFFLINE with a model that answers in a
// format), how it looks (preview.yml).
import assert from "node:assert/strict";
import * as playwright from "playwright-core";
import { serving } from "./serving.mjs";
import { closed, leave, scratch } from "./leave.mjs";

const engine = process.argv[2] ?? "chromium";
const channel = engine === "chrome" ? "chrome" : undefined;
const kind = playwright[channel ? "chromium" : engine];
const THOUGHT = "Let me think about **this** <b>first</b>.\n";
const ANSWER = "# Heading\n\nSome **bold**, *italic* and `code`.\nA second line.\n\n- one\n- two\n\n" +
  "| a | b |\n|:-|-:|\n| 1 | 2 |\n\n```js\nlet x = '<b>'; // a line far wider than a phone's screen, which scrolls in its block and never the page\n```\n\n" +
  "<img src=x onerror=\"window.__ran=1\"> [x](javascript:window.__ran=2) ![i](https://example.com/x.png) " +
  "<a href=\"https://example.com\">a</a> <script>window.__ran=3</script>\n\n> quoted\n\nThe end.";
const TEXT = `${THOUGHT}</think>\n\n${ANSWER}`;

/** Runs in the page before its scripts: the model's worker is a made-up one (other workers are as they were). */
function madeUpWorker(text) {
  const Real = window.Worker;
  window.__draws = 0;
  window.Worker = class extends EventTarget {
    constructor(url, options) {
      if (!/\/_astro\/worker\.[\w-]+\.js/.test(String(url))) return new Real(url, options);
      super();
      this.onmessage = null;
    }
    tell(data) {
      this.onmessage?.({ data });
    }
    postMessage(message) {
      if (message.type === "init" || message.type === "load") {
        setTimeout(() => this.tell({ type: "ready", load: message.load, pyodide: "0.0.0", backend: "made up", seq_len: 4096, heap: undefined, threads: 1, gpu: "",
          seconds: { pyodide: 0, download: 0, construct: 0 }, template: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n" }), 50);
      } else if (message.type === "generate") {
        const pieces = text.match(/[\s\S]{1,3}/g);
        let i = 0;
        const next = () => {
          if (i === Math.floor(pieces.length / 2)) {
            // halfway: mark the first block drawn, which must be the same element at the end
            const first = document.querySelector(".model .bubble .markdown > div");
            if (first) first.dataset.mark = "halfway";
          }
          if (i < pieces.length) {
            this.tell({ type: "token", text: pieces[i++] });
            setTimeout(next, 1);
          } else {
            this.tell({ type: "done", threads: 1, gpuTokens: 0, gpuSampled: 0, gpu: undefined, tokens: pieces.length + 5, prompt_tokens: 5, sampled: pieces.length,
              tokens_per_second: 100, prompt_tokens_per_second: 100, first_token_seconds: 0.1, seconds: 1 });
          }
        };
        next();
      }
    }
    terminate() {}
  };
  // every change of the answer's bubble, to count the draws
  new MutationObserver((records) => {
    if (records.some((record) => record.target.closest?.(".model .bubble") || record.target.parentElement?.closest(".model .bubble"))) window.__draws += 1;
  }).observe(document, { subtree: true, childList: true, characterData: true });
}

const site = await serving();
const failures = [];
async function visit(query, { blocked = false } = {}) {
  const browser = await kind.launchPersistentContext(scratch("markdown-"), { headless: true, channel, viewport: { width: 360, height: 760 } });
  const page = browser.pages()[0] ?? await browser.newPage();
  const said = [];
  page.on("console", (message) => said.push({ type: message.type(), text: message.text() }));
  page.on("pageerror", (error) => said.push({ type: "error", text: String(error) }));
  if (blocked) await browser.route(/cdn\.jsdelivr\.net\/npm\/|data\.jsdelivr\.com\/v1\/packages\/npm\//, (route) => route.abort());
  await page.addInitScript(madeUpWorker, TEXT);
  await page.goto(`${site.url}${query}`);
  for (;;) {
    try {
      await page.waitForFunction(() => !document.getElementById("run").disabled, null, { timeout: 60000 });
      break;
    } catch (error) {
      if (/context was destroyed|navigation/i.test(String(error))) continue;  // (the first visit reloads once)
      const status = await page.evaluate(() => document.getElementById("status-text")?.textContent).catch(() => "?");
      throw new Error(`${query}: the page did not come to ready (its status line: ${status}; its console: ${said.map((line) => line.text).join(" | ")})`);
    }
  }
  // the libraries are asked for at ready: give them their time, then answer
  if (!blocked) await page.waitForFunction(() => performance.getEntriesByType("resource").filter((e) => /\/\+esm$/.test(e.name)).length >= 2, null, { timeout: 60000 });
  else await page.waitForTimeout(500);
  await page.fill("#prompt", "Say something");
  await page.click("#run");
  await page.waitForFunction(() => document.querySelector(".model .meta summary"), null, { timeout: 60000 });
  const seen = await page.evaluate(() => {
    const bubble = [...document.querySelectorAll(".model .bubble")].pop();
    const all = [...bubble.querySelectorAll("*")];
    return {
      text: bubble.textContent,
      thought: bubble.firstChild?.nodeType === 3 ? bubble.firstChild.data : null,
      markdown: Boolean(bubble.querySelector(".markdown")),
      tags: [...new Set(all.map((element) => element.localName))].sort(),
      attributes: [...new Set(all.flatMap((element) => [...element.attributes].map((attribute) => attribute.name)))].sort(),
      inChat: [...document.querySelectorAll(".message.model :is(a, img, script, iframe, input)")].length,
      ran: window.__ran ?? null,
      halfway: Boolean(bubble.querySelector("[data-mark=halfway]")),
      draws: window.__draws,
      wide: document.documentElement.scrollWidth > innerWidth,
    };
  });
  await closed(() => browser.close());
  return { seen, said };
}
const tokens = TEXT.match(/[\s\S]{1,3}/g).length;
const check = (what, fact, why) => { if (!fact) failures.push(`${what}: ${why}`); };

for (const [query, name] of [["", "the page (its service worker)"], ["?coi=off", "?coi=off"]]) {
  const { seen, said } = await visit(query);
  check(name, seen.markdown, "the answer is not drawn as Markdown");
  check(name, seen.thought === `${THOUGHT}</think>`, `the thinking is not plain text as it was written: ${JSON.stringify(seen.thought)}`);
  for (const tag of ["h1", "p", "strong", "em", "code", "pre", "ul", "li", "table", "th", "td", "br"]) check(name, seen.tags.includes(tag), `no ${tag} drawn (${seen.tags})`);
  for (const tag of seen.tags) check(name, ["div", "h1", "p", "strong", "em", "code", "pre", "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td", "br"].includes(tag), `${tag} drawn`);
  check(name, seen.attributes.every((attribute) => ["class", "align", "start", "data-mark"].includes(attribute)), `attributes ${seen.attributes}`);
  check(name, seen.inChat === 0 && seen.ran === null, `a link, image, script or box in the chat (${seen.inChat}), or the model's script ran (${seen.ran})`);
  for (const text of ["<img src=x onerror=\"window.__ran=1\">", "[x](javascript:window.__ran=2)", "![i](https://example.com/x.png)", "<script>window.__ran=3</script>", "> quoted", "**this** <b>first</b>"]) {
    check(name, seen.text.includes(text), `not said as written: ${text}`);
  }
  check(name, !seen.wide, "the page scrolls sideways at 360 px");
  check(name, seen.halfway, "the first block drawn halfway is not the same element at the end");
  check(name, seen.draws < tokens / 2, `drawn ${seen.draws} times for ${tokens} tokens`);
  const errors = said.filter((line) => line.type === "error");
  check(name, errors.length === 0, `console errors: ${errors.map((line) => line.text).join(" | ")}`);
  check(name, !said.some((line) => /^markdown:/.test(line.text)), "it said the libraries did not come");
  console.log(`markdown-in-browser: ${engine}, ${name}: ${seen.tags.length} kinds of elements drawn, ${seen.draws} draws for ${tokens} tokens`);
}
{
  // (without the service worker: Playwright's routes do not reach the requests a service worker makes in Firefox)
  const { seen, said } = await visit("?coi=off", { blocked: true });
  const name = "jsDelivr's packages blocked";
  check(name, !seen.markdown && seen.text === TEXT, `not the plain text, whole: ${JSON.stringify(seen.text.slice(0, 80))}`);
  check(name, seen.tags.length === 0, `elements in the bubble: ${seen.tags}`);
  const told = said.filter((line) => /^markdown:/.test(line.text));
  check(name, told.length === 1 && told[0].type === "warning", `the console says ${told.length} times that the libraries did not come`);
  // (the browser's own line for each request it was told to drop is no error of the page's: Chromium's, Firefox's)
  const dropped = /^Failed to load resource: net::ERR_FAILED$|Cross-Origin Request Blocked: .* https:\/\/(data|cdn)\.jsdelivr\.(com|net)\//;
  const errors = said.filter((line) => line.type === "error" && !dropped.test(line.text));
  check(name, errors.length === 0, `console errors: ${errors.map((line) => line.text).join(" | ")}`);
  check(name, seen.draws < tokens / 2, `drawn ${seen.draws} times for ${tokens} tokens`);
  console.log(`markdown-in-browser: ${engine}, ${name}: plain text, ${seen.draws} draws for ${tokens} tokens; the console: ${told[0]?.text}`);
}
await site.close();
if (failures.length) {
  console.log(`FAILED\n${failures.map((failure) => `- ${failure}`).join("\n")}`);
  await leave(1);
}
console.log(`markdown-in-browser: ok (${engine})`);
await leave(0);
