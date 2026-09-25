// T118: the page on a slow line and on one that stops, in Chromium, through a proxy of this script's own that
// passes every connection on at a shared rate (a token bucket: rate bytes a second for all of them together, the
// way one line is shared). For CI (slow.yml): the development machine runs no browser.
//
//   node tests/slow-check.mjs <slow|stall> [site URL] [--rate <bytes per second>] [--model <id>]
//
//   slow   the whole line at --rate (default 50000: 0.4 Mbps). Pyodide's 9 MB take minutes; the load must end
//          ready, without falling back to no service worker (before T118 a step was given up after 60 to 90
//          seconds however much was still arriving, and the page fell back, then failed again)
//   stall  cdn.jsdelivr.net stops sending after 200 kB. The load must end in an error that says so, after one
//          fallback, and not hang
import http from "node:http";
import net from "node:net";
import * as playwright from "playwright-core";

const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
const [scenario = "slow", site = "https://takano32.github.io/pyodide-llm/"] = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
const rate = Number(option("--rate", 50000)), model = option("--model", "stories260K");
const STALL_AFTER = 200_000;

// ---- the proxy: CONNECT tunnels; what comes back from a server is paced, what goes to it is not
const lanes = new Set();
const sentByHost = new Map();
let budget = 0;
setInterval(() => {
  budget = Math.min(rate / 10, budget + rate / 20);
  for (const lane of lanes) {
    while (lane.queue.length && budget > 0) {
      if (scenario === "stall" && lane.host === "cdn.jsdelivr.net" && (sentByHost.get(lane.host) ?? 0) >= STALL_AFTER) break;
      const chunk = lane.queue[0];
      const take = Math.min(chunk.length, Math.ceil(budget));
      lane.client.write(chunk.subarray(0, take));
      budget -= take;
      lane.size -= take;
      sentByHost.set(lane.host, (sentByHost.get(lane.host) ?? 0) + take);
      if (take === chunk.length) lane.queue.shift();
      else lane.queue[0] = chunk.subarray(take);
    }
    if (lane.size < 64 * 1024) lane.upstream.resume();
  }
}, 50).unref();
const proxy = http.createServer();
proxy.on("connect", (req, client, head) => {
  const [host, port] = req.url.split(":");
  const upstream = net.connect(Number(port) || 443, host, () => {
    client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
    if (head.length) upstream.write(head);
    client.pipe(upstream);
  });
  const lane = { host, client, upstream, queue: [], size: 0 };
  lanes.add(lane);
  upstream.on("data", (chunk) => {
    lane.queue.push(chunk);
    lane.size += chunk.length;
    if (lane.size > 256 * 1024) upstream.pause();
  });
  const end = () => {
    lanes.delete(lane);
    client.destroy();
    upstream.destroy();
  };
  upstream.on("error", end).on("close", end);
  client.on("error", end).on("close", end);
});
await new Promise((resolve) => proxy.listen(0, "127.0.0.1", resolve));

// ---- the page
const browser = await playwright.chromium.launch({ proxy: { server: `http://127.0.0.1:${proxy.address().port}` } });
const page = await (await browser.newContext()).newPage();
const console_ = [];
page.on("console", (message) => console_.push(message.text()));
const began = Date.now();
await page.goto(`${site}?model=${model}`, { timeout: 600000 });
// ready (the run button on) or an error, across the reloads (the service worker's first one, a fallback's)
let outcome;
for (;;) {
  try {
    outcome = await page.waitForFunction(() => {
      if (document.querySelector(".error")) return { error: document.querySelector(".error .bubble")?.textContent ?? "" };
      if (document.getElementById("run")?.disabled === false) return { ready: true };
      return null;
    }, null, { timeout: 900000, polling: 1000 }).then((handle) => handle.jsonValue());
    break;
  } catch (error) {
    if (!/destroyed|navigat|detached/i.test(String(error.message))) {
      outcome = { error: `the test: ${String(error.message).split("\n")[0]}` };
      break;
    }
  }
}
const state = await page.evaluate(() => ({ fellBack: sessionStorage.getItem("coi-fallback") === "1", isolated: self.crossOriginIsolated,
  status: document.getElementById("status-text")?.textContent ?? "" })).catch(() => ({}));
const seconds = ((Date.now() - began) / 1000).toFixed(0);
const megabytes = [...sentByHost].map(([host, bytes]) => `${host} ${(bytes / 1e6).toFixed(1)} MB`).join(", ");
console.log(`${scenario} at ${(rate * 8 / 1e6).toFixed(2)} Mbps: after ${seconds} s ${outcome.ready ? "ready" : `error "${outcome.error}"`}; ` +
  `fell back to no service worker: ${state.fellBack}; isolated: ${state.isolated}; status: ${state.status}`);
console.log(`through the proxy: ${megabytes}`);
const pyodideLines = console_.filter((line) => /pyodide|quiet|nothing/i.test(line)).slice(-5);
if (pyodideLines.length) console.log(`console: ${pyodideLines.join(" / ")}`);
await browser.close();
proxy.close();
const ok = scenario === "slow" ? outcome.ready && !state.fellBack : Boolean(outcome.error) && state.fellBack && !/the test:/.test(outcome.error);
console.log(ok ? "ok" : "FAILED");
process.exit(ok ? 0 : 1);
