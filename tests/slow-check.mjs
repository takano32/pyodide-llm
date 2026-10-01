// T118: the page on a slow line and on one that stops, in Chromium, through a proxy of this script's own that
// passes every connection on at a shared rate (a token bucket: rate bytes a second for all of them together, the
// way one line is shared). For CI (slow.yml): the development machine runs no browser.
//
//   node tests/slow-check.mjs <slow|stall> [site URL] [--rate <bytes per second>] [--model <id>]
//                             [--queue <lane|fair|fifo>] [--buffer <bytes>]
//
//   slow   the whole line at --rate (default 50000: 0.4 Mbps). Pyodide's 9 MB take minutes; the load must end
//          ready, without falling back to no service worker (before T118 a step was given up after 60 to 90
//          seconds however much was still arriving, and the page fell back, then failed again)
//   stall  cdn.jsdelivr.net stops sending after 200 kB. The load must end in an error that says so, after one
//          fallback, and not hang
//
// How the one line is shared by the connections (--queue; T129's review, 2026-10-01: the version of Pyodide is asked
// for while the model's parts fill the line, and how long its answer waits depends on this):
//   lane   the oldest connection is served first, a newer one gets what is left (T118's. A newer connection, such as
//          the one of data.jsdelivr.com, waits until every older one has nothing queued: the model of this site comes
//          over the connection the page itself came by, which is the oldest)
//   fair   every connection a share of the second, a packet each in turn
//   fifo   one queue of --buffer bytes (default 262144) in the order the servers sent, as a phone's line with a deep
//          buffer: what a connection that opens meanwhile, a TLS handshake or a small answer, waits behind is the
//          buffer's worth (the buffer over the rate, once for each time it must wait: about three or four times)
// The lines of data.jsdelivr.com (the version of Pyodide: a hundred bytes after the handshake) are told at the end,
// from the start of the page's worker: when it opened, and when its first and its last byte went through.
import http from "node:http";
import net from "node:net";
import * as playwright from "playwright-core";

const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
const [scenario = "slow", site = "https://takano32.github.io/pyodide-llm/"] = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
const rate = Number(option("--rate", 50000)), model = option("--model", "stories260K");
const queueMode = option("--queue", "lane"), bufferBytes = Number(option("--buffer", 262144));
if (!["lane", "fair", "fifo"].includes(queueMode)) throw new Error(`--queue is lane, fair or fifo, not ${queueMode}`);
const STALL_AFTER = 200_000;

// ---- the proxy: CONNECT tunnels; what comes back from a server is paced, what goes to it is not
const lanes = new Set(), everyLane = [];
const fifo = [];  // fifo: { lane, chunk } in the order the servers sent them
let fifoBytes = 0, tick = 0;
const sentByHost = new Map();
let budget = 0;
const deliver = (lane, chunk, take) => {
  lane.client.write(chunk.subarray(0, take));
  const now = Date.now();
  lane.first ??= now;
  lane.last = now;
  lane.bytes += take;
  sentByHost.set(lane.host, (sentByHost.get(lane.host) ?? 0) + take);
};
const stalled = (lane) => scenario === "stall" && lane.host === "cdn.jsdelivr.net" && (sentByHost.get(lane.host) ?? 0) >= STALL_AFTER;
setInterval(() => {
  budget = Math.min(rate / 10, budget + rate / 20);
  if (queueMode === "fifo") {
    while (fifo.length && budget > 0) {
      const { lane, chunk } = fifo[0];
      const take = Math.min(chunk.length, Math.ceil(budget));
      deliver(lane, chunk, take);
      budget -= take;
      fifoBytes -= take;
      if (take === chunk.length) fifo.shift();
      else fifo[0].chunk = chunk.subarray(take);
    }
    if (fifoBytes < bufferBytes / 2) for (const lane of lanes) lane.upstream.resume();
    return;
  }
  if (queueMode === "fair") {
    const list = [...lanes], start = tick++ % Math.max(list.length, 1);
    for (let moved = true; moved && budget > 0;) {
      moved = false;
      for (let i = 0; i < list.length && budget > 0; i++) {
        const lane = list[(start + i) % list.length];
        if (!lane.queue.length || stalled(lane)) continue;
        const chunk = lane.queue[0], take = Math.min(chunk.length, 1460, Math.ceil(budget));
        deliver(lane, chunk, take);
        budget -= take;
        lane.size -= take;
        moved = true;
        if (take === chunk.length) lane.queue.shift();
        else lane.queue[0] = chunk.subarray(take);
      }
    }
  } else {
    for (const lane of lanes) {
      while (lane.queue.length && budget > 0 && !stalled(lane)) {
        const chunk = lane.queue[0];
        const take = Math.min(chunk.length, Math.ceil(budget));
        deliver(lane, chunk, take);
        budget -= take;
        lane.size -= take;
        if (take === chunk.length) lane.queue.shift();
        else lane.queue[0] = chunk.subarray(take);
      }
    }
  }
  for (const lane of lanes) if (lane.size < 64 * 1024) lane.upstream.resume();
}, 50).unref();
const proxy = http.createServer();
proxy.on("connect", (req, client, head) => {
  const [host, port] = req.url.split(":");
  const upstream = net.connect(Number(port) || 443, host, () => {
    client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
    if (head.length) upstream.write(head);
    client.pipe(upstream);
  });
  const lane = { host, client, upstream, queue: [], size: 0, opened: Date.now(), first: undefined, last: undefined, bytes: 0 };
  lanes.add(lane);
  everyLane.push(lane);
  upstream.on("data", (chunk) => {
    if (queueMode === "fifo") {
      fifo.push({ lane, chunk });
      fifoBytes += chunk.length;
      if (fifoBytes > bufferBytes) for (const other of lanes) other.upstream.pause();
      return;
    }
    lane.queue.push(chunk);
    lane.size += chunk.length;
    if (lane.size > 256 * 1024) upstream.pause();
  });
  const end = () => {
    lanes.delete(lane);
    for (let i = fifo.length - 1; i >= 0; i--) {
      if (fifo[i].lane === lane) fifoBytes -= fifo.splice(i, 1)[0].chunk.length;
    }
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
const workers = [];  // when the page's workers (the model's first, then the software threads) began
page.on("worker", () => workers.push(Date.now()));
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
const [workerAt] = workers, since = (ms) => (ms === undefined ? "never" : workerAt ? `${((ms - workerAt) / 1000).toFixed(1)} s` : "?");
console.log(`the line was shared as ${queueMode}${queueMode === "fifo" ? ` (a queue of ${bufferBytes} bytes)` : ""}; model ${model}; ` +
  `the worker began ${workerAt ? ((workerAt - began) / 1000).toFixed(1) : "never"} s after the page was asked for`);
for (const lane of everyLane.filter((one) => one.host === "data.jsdelivr.com")) {
  console.log(`the version's line, from the worker's start: opened ${since(lane.opened)}, first byte ${since(lane.first)}, ` +
    `last byte ${since(lane.last)}, ${lane.bytes} bytes (T129: the worker gives the answer up after 30 s)`);
}
const pyodideLines = console_.filter((line) => /pyodide|quiet|nothing/i.test(line)).slice(-5);
if (pyodideLines.length) console.log(`console: ${pyodideLines.join(" / ")}`);
await browser.close();
proxy.close();
const ok = scenario === "slow" ? outcome.ready && !state.fellBack : Boolean(outcome.error) && state.fellBack && !/the test:/.test(outcome.error);
console.log(ok ? "ok" : "FAILED");
process.exit(ok ? 0 : 1);
