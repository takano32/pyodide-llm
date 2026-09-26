// T137, stage 0: opens public/opfs-test/ (?run=1&size=256) in a browser and prints what it measured, as Markdown.
// The runners' disks are not the owner's devices: this checks that the page works in every engine (a sync handle,
// the pieces read back right, a second handle refused) and gives a first number; the verdict waits for the owner's
// devices (TODO.md, T137). Playwright's Firefox runs under a debugger (AGENTS.md): its numbers are not Firefox's.
//
//   node tests/opfs-check.mjs <site URL> [engine ...] [--size MiB]      engines: chromium firefox webkit chrome msedge
import * as playwright from "playwright-core";

const args = process.argv.slice(2);
const at = args.indexOf("--size");
const size = at >= 0 ? Number(args.splice(at, 2)[1]) : 256;
const [site = "https://takano32.github.io/pyodide-llm/", ...engines] = args;
let failed = false;
for (const engine of engines.length ? engines : ["chromium", "firefox", "webkit"]) {
  const branded = ["chrome", "msedge"].includes(engine);
  const type = branded ? playwright.chromium : playwright[engine];
  console.log(`### ${engine}`);
  let browser;
  try {
    browser = await type.launch(branded ? { channel: engine } : {});
  } catch (error) {
    console.log(`could not start: ${String(error.message).split("\n")[0]}\n`);
    continue;
  }
  const page = await (await browser.newContext()).newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error.message)));
  try {
    await page.goto(`${site}opfs-test/?run=1&size=${size}`);
    await page.waitForFunction(() => window.__opfs, null, { timeout: 900000 });
    console.log(`${browser.version()}\n`);
    console.log(await page.evaluate(() => document.getElementById("markdown").textContent));
    // where there is a sync handle, the run must end with every piece read back right
    const steps = await page.evaluate(() => window.__opfs.steps);
    const run = steps.find((s) => s.name === "the writes");
    if (run && (run.error || run.result.read.wrong)) failed = true;
  } catch (error) {
    console.log(`failed: ${String(error.message).split("\n")[0]}`);
    failed = true;
  }
  if (errors.length) console.log(`page errors: ${errors.join(" / ")}`);
  console.log("");
  await browser.close();
}
process.exit(failed ? 1 : 0);
