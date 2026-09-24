// T93, part 2: does public/coi-test/ become cross-origin isolated in each browser, with each COEP, and do Pyodide
// from the CDN and a range of a Hugging Face file still load under it? Prints one Markdown table.
//
//   node tests/coi-check.mjs <site URL> [engine ...]      engines: chromium firefox webkit chrome msedge
import * as playwright from "playwright-core";

const [site = "https://takano32.github.io/pyodide-llm/", ...engines] = process.argv.slice(2);
const rows = [];
for (const engine of engines.length ? engines : ["chromium", "firefox", "webkit"]) {
  const type = ["chrome", "msedge"].includes(engine) ? playwright.chromium : playwright[engine];
  let browser;
  try {
    browser = await type.launch(["chrome", "msedge"].includes(engine) ? { channel: engine } : {});
  } catch (error) {
    rows.push(`| ${engine} | | could not start: ${String(error.message).split("\n")[0]} | | | | |`);
    continue;
  }
  for (const coep of ["credentialless", "require-corp"]) {
    const context = await browser.newContext();
    const page = await context.newPage();
    let result;
    try {
      await page.goto(`${site}coi-test/?coep=${coep}`);
      await page.waitForFunction(() => window.__coi, null, { timeout: 180000 });
      result = await page.evaluate(() => window.__coi);
    } catch (error) {
      result = { error: String(error.message).split("\n")[0] };
    }
    const cell = (v) => String(v ?? "").replace(/\|/g, "\\|").slice(0, 80);
    rows.push(`| ${engine} ${browser.version()} | ${coep} | ${cell(result.error ?? result.crossOriginIsolated)} | ${cell(result.atomics === true && result.sharedValue === 42 ? "yes" : result.atomics)} | ${cell(result.pyodide)} | ${cell(result.huggingface)} | ${result.reloads ?? ""} |`);
    await context.close();
  }
  await browser.close();
}
console.log("| browser | COEP | crossOriginIsolated | shared memory with a worker | Pyodide + NumPy from the CDN | Hugging Face range | reloads |");
console.log("|---|---|---|---|---|---|---|");
for (const row of rows) console.log(row);
