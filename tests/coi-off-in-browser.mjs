// T.507, T.522: ?coi=off in a browser that has the service worker (public/coi.js) already. A first visit installs it
// and is isolated; the same profile then opens ?coi=off, which unregisters it and must come to ready without it (it
// stayed at "Initializing..." before: the page came with the worker's headers, under which the model's worker did not
// start). On this tree's dist/ (npm run build first), with the site's smallest model.
//
//   node tests/coi-off-in-browser.mjs [chromium|chrome]
import assert from "node:assert/strict";
import * as playwright from "playwright-core";
import { serving } from "./serving.mjs";
import { scratch, closed, leave } from "./leave.mjs";

const [engine = "chromium"] = process.argv.slice(2);
const server = await serving();
const context = await playwright.chromium.launchPersistentContext(scratch("coi-off-profile-"), { headless: true, channel: engine === "chrome" ? "chrome" : undefined });
const page = context.pages()[0] ?? await context.newPage();
// ready: the status line says what ran (" · Pyodide "), or a failure
const settled = async (what) => {
  for (let i = 0; i < 240; i++) {
    const said = await page.evaluate(() => document.querySelector("#status-text")?.textContent ?? "").catch(() => "");
    if (/ · Pyodide |Failed|Could not/.test(said)) return said;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`${what}: not ready in 120 s (the status line says "${await page.evaluate(() => document.querySelector("#status-text")?.textContent).catch(() => "?")}")`);
};
let code = 0;
try {
  await page.goto(`${server.url}?model=stories260K`);
  const first = await settled("the first visit");
  assert.ok(/ · Pyodide /.test(first) && await page.evaluate(() => self.crossOriginIsolated), `the first visit: isolated and ready (${first})`);
  await page.goto(`${server.url}?model=stories260K&coi=off`);
  const off = await settled("?coi=off");
  assert.match(off, / · Pyodide /, "?coi=off comes to ready");
  assert.equal(await page.evaluate(() => self.crossOriginIsolated), false, "?coi=off is not isolated");
  assert.equal(await page.evaluate(async () => (await navigator.serviceWorker.getRegistrations()).length), 0, "and the worker is unregistered");
  console.log(`ok: ?coi=off after an isolated visit in the same profile comes to ready without the service worker (${off})`);
} catch (error) {
  console.log(`FAILED\n- ${error.message}`);
  code = 1;
}
await closed(() => context.close());
await server.close();
await leave(code);
