// The page as the owner sees it before a change to it goes out (AGENTS.md: a drawing task shows a phone-width and a
// desktop-width screenshot first). Serves dist/ like tests/e2e.mjs, or takes a deployed site, and saves one PNG per
// scene and width into a directory. Meant for CI (preview.yml): the development machine runs no browser.
//
//   node tests/screenshots.mjs <directory> [url of a deployed site]
//
// The scenes: the page with the smallest model ready; the settings sheet open at its engine switches (T75); the
// benchmark's bubble (T76); the sheet that opens any repository, and a repository the converter refuses (T88).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import * as playwright from "playwright-core";

const [directory = "screenshots", deployed] = process.argv.slice(2);
const base = "/pyodide-llm/";
const root = new URL("../dist/", import.meta.url).pathname;
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".py": "text/plain" };
let server, site = deployed;
if (!site) {
  server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    const file = path.join(root, pathname.slice(base.length) || "index.html");
    if (!pathname.startsWith(base) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404);
      return res.end();
    }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  }).listen(0);
  site = `http://localhost:${server.address().port}${base}`;
}
fs.mkdirSync(directory, { recursive: true });

const browser = await playwright.chromium.launch({ headless: true });
const widths = { phone: { width: 390, height: 844 }, desktop: { width: 1280, height: 800 } };
// a wait that the first visit's reload (coi.js, T93) interrupts starts again
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

for (const [width, viewport] of Object.entries(widths)) {
  const page = await browser.newPage({ viewport });
  page.on("dialog", (dialog) => dialog.accept());
  const ready = () => acrossReload(() => page.waitForFunction(() => document.getElementById("run")?.disabled === false || document.querySelector(".error"), null, { timeout: 600000 }));

  // 1. the page, the smallest model ready
  await page.goto(`${site}?model=stories260K`);
  await ready();
  await shot(page, "home", width);

  // 2. the settings sheet, the engine's switches open (T75)
  await page.click("#settings-open");
  await page.evaluate(() => { document.getElementById("engine").open = true; });
  await page.waitForTimeout(300);
  await shot(page, "settings-engine", width);
  await page.keyboard.press("Escape");

  // 3. the sheet that opens any repository, with a name typed (T88)
  await page.selectOption("#model", "hf-other");
  await page.waitForTimeout(300);
  await page.fill("#repository-name", "Qwen/Qwen2.5-0.5B-Instruct");
  await shot(page, "repository", width);
  await page.keyboard.press("Escape");

  // 4. the benchmark's bubble (T76)
  await page.goto(`${site}?model=stories260K&bench=1`);
  await acrossReload(() => page.waitForFunction(() => window.__bench || document.querySelector(".error"), null, { timeout: 600000 }));
  await page.waitForTimeout(300);
  await shot(page, "bench", width);

  // 5. a repository the converter refuses, in words (T88)
  await page.goto(`${site}?hf=microsoft/phi-1_5`);
  await acrossReload(() => page.waitForFunction(() => document.querySelector(".error"), null, { timeout: 600000 }));
  await page.waitForTimeout(300);
  await shot(page, "refused", width);
  await page.close();
}
await browser.close();
server?.close();
console.log(fs.readdirSync(directory).join("\n"));
