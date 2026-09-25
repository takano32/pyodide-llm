// T94, stage 0: opens public/gpu-test/ (?run=small) in a browser and prints what it measured, as Markdown. CI runners
// have no GPU, so this is about whether the page and its shaders work (a software adapter, where the browser has
// one), not about speed: the speed is measured on the owner's devices (TODO.md, T94).
// The model page is opened first: its service worker makes the site cross-origin isolated, which the bridge needs.
//
//   node tests/gpu-check.mjs <site URL> [engine ...]      engines: chromium firefox webkit chrome msedge
import * as playwright from "playwright-core";

const [site = "https://takano32.github.io/pyodide-llm/", ...engines] = process.argv.slice(2);
// Chromium's WebGPU without a GPU: SwiftShader's Vulkan, where the build has it
const FLAGS = { chromium: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-webgpu-adapter=swiftshader"] };
let failed = false;
for (const engine of engines.length ? engines : ["chromium", "firefox", "webkit"]) {
  const type = ["chrome", "msedge"].includes(engine) ? playwright.chromium : playwright[engine];
  const flags = FLAGS[engine] ?? (["chrome", "msedge"].includes(engine) ? FLAGS.chromium : []);
  console.log(`### ${engine}`);
  let browser;
  try {
    browser = await type.launch({ ...(["chrome", "msedge"].includes(engine) ? { channel: engine } : {}), args: flags });
  } catch (error) {
    console.log(`could not start: ${String(error.message).split("\n")[0]}\n`);
    continue;
  }
  const page = await (await browser.newContext()).newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error.message)));
  try {
    await page.goto(site);
    // the service worker takes the page over after one reload of its own
    await page.waitForFunction(() => navigator.serviceWorker?.controller, null, { timeout: 120000 }).catch(() => {});
    await page.goto(`${site}gpu-test/?run=small`);
    await page.waitForFunction(() => window.__gpu, null, { timeout: 600000 });
    console.log(`${browser.version()}\n`);
    console.log(await page.evaluate(() => document.getElementById("markdown").textContent));
    // the shaders must agree with JavaScript wherever there is an adapter to run them
    const check = await page.evaluate(() => window.__gpu.steps.find((s) => s.name === "the shaders against JavaScript")?.result);
    if (check && Object.values(check).some((v) => !v.ok)) failed = true;
  } catch (error) {
    console.log(`failed: ${String(error.message).split("\n")[0]}`);
    failed = true;
  }
  if (errors.length) console.log(`page errors: ${errors.join(" / ")}`);
  console.log("");
  await browser.close();
}
process.exit(failed ? 1 : 0);
