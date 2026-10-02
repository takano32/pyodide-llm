// T233 review probe (a throwaway branch, never merged): where does Playwright's Chromium put the origin private file system,
// and how much does it take? Writes 8 MiB parts into one file of the site's origin through a sync access handle (as the
// page's keep() does) until a write comes back short or throws, or the limit is reached, while this process reads
// /proc/meminfo every 2 seconds.
//   node tests/t233_opfs_probe.mjs private|persistent [limit in GiB = 12]
// private: browser.launch() + newPage(), what tests/e2e.mjs does without E2E_TWICE (like private browsing);
// persistent: launchPersistentContext() with a profile on disk, what e2e.mjs does with E2E_TWICE and a visitor's browser has.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright-core";

const [mode = "private", limitArg = "12"] = process.argv.slice(2);
const limit = Number(limitArg);
const url = "https://takano32.github.io/pyodide-llm/probe-opfs.html";
const meminfo = () => {
  const text = fs.readFileSync("/proc/meminfo", "utf8");
  const gib = (key) => Number(text.match(new RegExp(`${key}:\\s+(\\d+) kB`))[1]) / 1048576;
  return `MemAvailable ${gib("MemAvailable").toFixed(2)} GiB, MemFree ${gib("MemFree").toFixed(2)}, Cached ${gib("Cached").toFixed(2)}, Shmem ${gib("Shmem").toFixed(2)}`;
};
const disk = () => {
  const stat = fs.statfsSync(os.tmpdir());
  return `${((stat.bavail * stat.bsize) / 2 ** 30).toFixed(1)} GiB free in ${os.tmpdir()}`;
};
console.log(`probe: ${mode}, limit ${limit} GiB; ${os.cpus()[0].model}, ${(os.totalmem() / 2 ** 30).toFixed(1)} GiB of memory; ${meminfo()}; ${disk()}`);
let browser, context, page;
if (mode === "persistent") {
  context = await chromium.launchPersistentContext(fs.mkdtempSync(path.join(os.tmpdir(), "probe-profile-")), { headless: true });
  page = context.pages()[0] ?? (await context.newPage());
} else {
  browser = await chromium.launch({ headless: true });
  page = await browser.newPage();
  console.log(`probe: chromium ${browser.version()}`);
}
await page.route(url, (route) => route.fulfill({ contentType: "text/html", body: "<!doctype html><title>probe</title>" }));
await page.goto(url);
console.log(`probe: storage estimate before: ${JSON.stringify(await page.evaluate(() => navigator.storage.estimate()))}`);
const timer = setInterval(() => console.log(`probe: ${((Date.now() - began) / 1000).toFixed(0)} s: ${meminfo()}; ${disk()}`), 2000);
const began = Date.now();
const result = await page.evaluate(async (limitGiB) => {
  const code = `
    self.onmessage = async ({ data: limitGiB }) => {
      const root = await navigator.storage.getDirectory();
      const file = await root.getFileHandle("probe.bin", { create: true });
      const handle = await file.createSyncAccessHandle();
      handle.truncate(0);
      const part = new Uint8Array(8 * 1024 * 1024);
      for (let i = 0; i < part.length; i += 4096) part[i] = (i >> 12) & 255;
      const limit = limitGiB * 1024 ** 3;
      let at = 0;
      while (at < limit) {
        let wrote;
        try {
          wrote = handle.write(part, { at });
        } catch (error) {
          postMessage({ done: true, at, error: String(error) });
          handle.close();
          return;
        }
        if (wrote !== part.length) {
          postMessage({ done: true, at: at + wrote, short: wrote });
          handle.close();
          return;
        }
        at += wrote;
        if (at % 1024 ** 3 === 0) postMessage({ progress: at / 1024 ** 3 });
      }
      handle.flush();
      handle.close();
      postMessage({ done: true, at });
    };`;
  const worker = new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
  return new Promise((resolve) => {
    const progress = [];
    worker.onmessage = ({ data }) => (data.done ? resolve({ ...data, progress }) : progress.push(data.progress));
    worker.postMessage(limitGiB);
  });
}, limit);
clearInterval(timer);
console.log(`probe: ${((Date.now() - began) / 1000).toFixed(0)} s: wrote ${(result.at / 2 ** 30).toFixed(3)} GiB${result.short !== undefined ? `, the last write came back short (${result.short} of ${8 * 2 ** 20})` : ""}${result.error ? `, threw: ${result.error}` : ""}${result.at >= limit * 2 ** 30 ? ", reached the limit" : ""}`);
console.log(`probe: ${meminfo()}; ${disk()}`);
console.log(`probe: storage estimate after: ${JSON.stringify(await page.evaluate(() => navigator.storage.estimate()))}`);
await page.evaluate(async () => (await navigator.storage.getDirectory()).removeEntry("probe.bin").catch(() => {}));
await (context ?? browser).close().catch(() => {});
process.exit(0);
