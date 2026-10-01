// T129: where fetching and loading meet in public/worker.js, run as it is in a vm context on made-up fetches, streams
// and a clock that goes a hundred times as fast: the parts of this site's models (download()), the ranges of
// huggingface.co (fetchRange(), inOrder(), refused()), the count of what arrives while Pyodide loads (watchArrivals()),
// the version of Pyodide and its steps (resolvePyodideVersion(), pyodideSteps()), and a model past even a 64-bit memory
// (weightsBuffer()), and load() as far as the place of the weights (what it stops where it ends before it). The review
// of T97, T118 and T119 (2026-09-26) had a bench like this and did not keep it; what it found is T129's (1) to (7).
// Node only, no Pyodide, a few seconds:
//
//   node tests/worker-check.mjs
//
// A slow or stopped line is played by the made-up fetches (a body that comes a chunk at a time, one that stops, one
// that breaks), on the fast clock: 30 seconds of the worker's are 0.3 s here. The Cache API is a made-up one that does
// what the specification says of put() (the review of T129, 2026-10-01). What it cannot see: the Service Worker
// (public/coi.js) between the worker and the network, a connection that is really stopped by an abort (the made-up
// fetch only says its signal aborted), a browser's own limits (a 64-bit memory's size: tests/mem64-limit.mjs of the
// probe branch asked Chromium and Firefox), and the queue of a real line (a version asked for while the model's parts
// fill it: tests/slow-check.mjs, slow.yml, sees a line with a queue, by its own proxy).
import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import * as forward from "../public/forward.js";

const SCALE = 100;  // the worker's milliseconds per real millisecond
// a check that waits for ever (a fix undone: the version asked for ever) fails rather than hangs
setTimeout(() => {
  console.error("worker-check: still waiting after 180 s");
  process.exit(1);
}, 180000).unref();
const MiB = 1024 * 1024, PART = 8 * MiB;
const realNow = () => performance.now();
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms / SCALE));  // ms of the worker's clock

// the worker's clock and timers: a hundred times as fast
const clock = {
  now: () => realNow() * SCALE,
  setTimeout: (f, ms = 0, ...args) => setTimeout(f, ms / SCALE, ...args),
  setInterval: (f, ms = 0, ...args) => setInterval(f, ms / SCALE, ...args),
};

// the bytes of a made-up file at [from, to): the same at every offset, whichever request brings them
function bytesOf(from, to) {
  const bytes = new Uint8Array(to - from);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Math.imul(from + i, 2654435761) >>> 24;
  return bytes;
}
const abortError = (signal) => signal.reason ?? new DOMException("aborted", "AbortError");

// a body of the file's bytes [from, to), chunk by chunk, each after delay ms of the worker's; breakAt: it breaks there;
// head: the bytes that begin it instead of the made-up ones (a header); cancelled(): told when its reader lets go of it
function body(from, to, { chunk = MiB, delay = 0, breakAt, signal, stall, head, cancelled } = {}) {
  let at = from;
  return new ReadableStream({
    async pull(controller) {
      if (delay) await sleep(delay);
      if (stall) await new Promise((resolve) => signal?.addEventListener("abort", resolve, { once: true }));
      if (signal?.aborted) return controller.error(abortError(signal));
      if (breakAt !== undefined && at >= breakAt) return controller.error(new TypeError("Error in input stream"));
      if (at >= to) return controller.close();
      const end = Math.min(to, at + chunk, breakAt ?? Infinity);
      const bytes = bytesOf(at, end);
      if (head && at === from) bytes.set(head.subarray(0, bytes.length));
      controller.enqueue(bytes);
      at = end;
    },
    cancel() {
      cancelled?.();
    },
  }, { highWaterMark: 0 });
}

// the made-up network: route(url, init, n) answers the n-th request of that URL (0 first) with a Response, a promise of
// one, or "hang" (no answer until the request is aborted). Every request is written down.
const requests = [];
let route = () => new Response("", { status: 404 });
function fetchStandIn(input, init = {}) {
  const url = String(input?.url ?? input);
  const n = requests.filter((r) => r.url === url).length;
  const request = { url, at: clock.now(), signal: init.signal, range: init.headers?.Range, method: init.method ?? "GET" };
  requests.push(request);
  return new Promise((resolve, reject) => {
    const signal = init.signal;
    if (signal?.aborted) return reject(abortError(signal));
    signal?.addEventListener("abort", () => reject(abortError(signal)), { once: true });
    Promise.resolve(route(url, init, n)).then((answer) => (answer === "hang" ? undefined : resolve(answer)), reject);
  });
}

// worker.js in a vm context whose global is its self, as in a worker
const at = new URL("../public/worker.js", import.meta.url);
const source = fs.readFileSync(at, "utf8").replaceAll("import.meta.url", JSON.stringify(at.href));
const messages = [];
const navigatorStandIn = { deviceMemory: 8 };
const context = vm.createContext({
  console, URL, URLSearchParams, TextDecoder, TextEncoder, AbortController, DOMException, Response, Headers,
  ReadableStream, WritableStream, TransformStream, WebAssembly, Atomics, SharedArrayBuffer,
  performance: { now: clock.now }, setTimeout: clock.setTimeout, clearTimeout, setInterval: clock.setInterval, clearInterval,
  // (breathe(): a turn of the event loop; Node's MessageChannel would keep the process alive)
  MessageChannel: class {
    constructor() {
      this.port1 = {};
      this.port2 = { postMessage: () => setImmediate(() => this.port1.onmessage?.()) };
    }
  },
  navigator: navigatorStandIn, location: { search: "" }, crossOriginIsolated: false,
  fetch: fetchStandIn, postMessage: (message) => messages.push(message),
});
context.self = context;
vm.runInContext(source, context, { filename: fileURLToPath(at) });
const run = (code) => vm.runInContext(code, context);
const quiet = run("QUIET_SECONDS");

// the error a promise rejects with, and when (on the worker's clock); undefined when it resolved
async function failure(promise) {
  try {
    await promise;
    return undefined;
  } catch (error) {
    return { error, at: clock.now() };
  }
}
const fresh = (routing) => {
  requests.length = 0;
  messages.length = 0;
  route = routing;
};
const partOf = (url) => Number(/\.(\d{3})$/.exec(url)?.[1]);
let passed = 0;
const ok = (line) => {
  passed++;
  console.log(`ok: ${line}`);
};

// ---- download(): the parts of this site's models (T97, T129 (3) and (4))
{
  const model = { checkpoint: "m", bytes: 2 * PART + 1000 };
  const whole = (part) => [part * PART, Math.min(model.bytes, (part + 1) * PART)];
  const written = () => {
    const into = new Uint8Array(model.bytes);
    return { into, write: (offset, chunk) => into.set(chunk, offset) };
  };
  const plain = (url, init) => new Response(body(...whole(partOf(url)), { signal: init.signal }), { status: 200 });
  const download = (signal = new AbortController().signal) => context.download(model, signal, 1);

  // all parts come: the bytes are the file's, and the header is its first 28
  fresh(plain);
  {
    const source = download(), { into, write } = written();
    await source.into(write);
    assert.deepEqual(into, bytesOf(0, model.bytes));
    assert.deepEqual([...await source.header], [...bytesOf(0, 28)]);
    ok("the parts of a model come whole");
  }

  // (4) a part the server failed (5xx) is fetched again, and one it refused (404) is not
  for (const [status, times, final] of [[503, 1, false], [500, 2, false], [408, 1, false], [404, 1, true], [403, 1, true]]) {
    fresh((url, init, n) => (partOf(url) === 1 && n < times ? new Response("", { status }) : plain(url, init)));
    const source = download(), { into, write } = written();
    const failed = await failure(source.into(write));
    const tries = requests.filter((r) => partOf(r.url) === 1).length;
    if (final) {
      assert.ok(failed, `a part answered ${status} was taken as a success`);
      assert.match(failed.error.message, new RegExp(`part 1 of m: ${status}`));
      assert.equal(tries, 1, `a part answered ${status} was fetched ${tries} times: the answer is the same the next time`);
    } else {
      assert.equal(failed, undefined, `a part answered ${status} ${times} time(s) failed the download: ${failed?.error.message}`);
      assert.deepEqual(into, bytesOf(0, model.bytes));
      assert.equal(tries, times + 1);
    }
  }
  ok("a part the server failed (5xx, 408) is fetched again, one it refused (4xx) once");

  // (4) three failures: the error says which part
  fresh((url, init) => (partOf(url) === 2 ? new Response("", { status: 502 }) : plain(url, init)));
  {
    const failed = await failure(download().into(written().write));
    assert.match(failed?.error.message ?? "", /Part 2 of m failed three times: .*502/);
    assert.equal(requests.filter((r) => partOf(r.url) === 2).length, 3);
  }
  // a body that breaks three times: the browser's words, and which part
  fresh((url, init) => new Response(body(...whole(partOf(url)), { signal: init.signal, breakAt: partOf(url) === 0 ? 3 * MiB : undefined }), { status: 200 }));
  {
    const failed = await failure(download().into(written().write));
    assert.match(failed?.error.message ?? "", /Part 0 of m failed three times: Error in input stream/);
  }
  ok("the third failure of a part says which part");

  // (4) a body that breaks once: fetched again, the bytes are right, and the progress never goes back
  fresh((url, init, n) => new Response(body(...whole(partOf(url)), {
    signal: init.signal, delay: partOf(url) === 1 ? 0 : 50, breakAt: partOf(url) === 1 && n === 0 ? PART + 5 * MiB : undefined,
  }), { status: 200 }));
  {
    const source = download(), { into, write } = written();
    await source.into(write);
    assert.deepEqual(into, bytesOf(0, model.bytes));
    const told = messages.filter((m) => m.type === "progress").map((m) => m.received);
    assert.ok(told.length > 3);
    told.forEach((received, i) => assert.ok(i === 0 || received >= told[i - 1], `the progress went back: ${told.slice(0, i + 1).join(", ")}`));
    assert.equal(told.at(-1), model.bytes);
  }
  ok("a part that broke is fetched again, and the progress does not go back meanwhile");

  // (3) a part that failed for good stops the other connections: none fetches another part, and the ones in flight are
  // aborted (before, they fetched the rest of the model until the next load)
  {
    const many = { checkpoint: "m", bytes: 12 * PART };
    fresh((url, init) => {
      const part = partOf(url);
      return part === 3 ? new Response("", { status: 404 })
        : new Response(body(part * PART, (part + 1) * PART, { signal: init.signal, delay: 100 }), { status: 200 });
    });
    const source = context.download(many, new AbortController().signal, 1);
    const failed = await failure(source.into(() => {}));
    assert.ok(failed);
    await sleep(3000);  // what would have come meanwhile
    const later = requests.filter((r) => r.at > failed.at);
    assert.deepEqual(later.map((r) => partOf(r.url)), [], "parts were fetched after one had failed for good");
    const flying = requests.filter((r) => partOf(r.url) !== 3);
    assert.ok(flying.length >= 7 && flying.every((r) => r.signal.aborted), "the parts in flight were not aborted");
    assert.equal(Math.max(...requests.map((r) => partOf(r.url))), 7, "a part past the first eight was asked for");
  }
  ok("a part that failed for good stops the other connections");

  // (3) a write the memory refused is not fetched again, and stops the rest; the body that was being read is let go of
  {
    const letGo = [];
    fresh((url, init) => new Response(body(...whole(partOf(url)), { signal: init.signal, delay: 20, cancelled: () => letGo.push(partOf(url)) }), { status: 200 }));
    const source = download();
    const failed = await failure(source.into((offset) => {
      if (offset >= PART) throw new RangeError("Invalid typed array length: 4188160");
    }));
    assert.equal(failed?.error.name, "RangeError");
    assert.deepEqual(requests.map((r) => partOf(r.url)).sort(), [0, 1, 2], "a write the memory refused was fetched again");
    assert.ok(requests.every((r) => r.signal.aborted || partOf(r.url) === 0));
    assert.ok(letGo.length >= 1, "the body of the part that failed was left open");
  }
  ok("a write the memory refused is not fetched again");

  // (3, the review) chunks that came before there was a memory (the download starts long before Pyodide is there, so
  // this is the usual order) and that the memory then refuses: the rest of the model is not fetched either
  {
    const many = { checkpoint: "m", bytes: 12 * PART };
    fresh((url, init) => new Response(body(partOf(url) * PART, (partOf(url) + 1) * PART, { signal: init.signal, delay: 100 }), { status: 200 }));
    const source = context.download(many, new AbortController().signal, 1);
    await sleep(500);  // chunks are queued meanwhile
    let writes = 0;  // (only the first write fails: what comes after it would stop the download by itself)
    const failed = await failure(source.into(() => {
      if (writes++ === 0) throw new RangeError("Invalid typed array length: 4188160");
    }));
    assert.equal(failed?.error.name, "RangeError");
    await sleep(3000);
    assert.deepEqual(requests.filter((r) => r.at > failed.at).map((r) => partOf(r.url)), [], "parts were fetched after the queued chunks were refused");
    assert.ok(requests.every((r) => r.signal.aborted), "the parts in flight were not aborted");
  }
  ok("queued chunks that the memory refuses stop the download");

  // (3, the review) a GPU's worker that takes no more of the weights (forward.js's room(), T156) is no more cured by
  // fetching the part again than a memory that refuses: the part is not asked for three times
  run("gpuOnlyNow = { room: () => Promise.reject(new Error('the GPU took no weights for 60 s')) }");
  try {
    fresh(plain);
    const failed = await failure(download().into(written().write));
    assert.match(failed?.error.message ?? "", /^the GPU took no weights/, "the part was fetched again where the GPU took no weights");
    assert.equal(requests.length, 3, `${requests.length} requests for 3 parts`);
  } finally {
    run("gpuOnlyNow = undefined");
  }
  ok("a GPU that takes no more of the weights is not fetched again");

  // the load's signal is let go of once the download is over, whichever way it ends
  fresh(plain);
  {
    const load = new AbortController(), source = download(load.signal);
    await source.into(written().write);
    assert.equal(getEventListeners(load.signal, "abort").length, 0, "a download that ended kept a listener on the load's signal");
  }
  fresh((url, init) => (partOf(url) === 1 ? new Response("", { status: 404 }) : plain(url, init)));
  {
    const load = new AbortController();
    assert.ok(await failure(download(load.signal).into(written().write)));
    assert.equal(getEventListeners(load.signal, "abort").length, 0, "a download that failed kept a listener on the load's signal");
  }
  ok("a download lets go of the load's signal");

  // (3) the load stops the download where it failed without it (weightsBuffer() said no)
  fresh((url, init) => new Response(body(...whole(partOf(url)), { signal: init.signal, delay: 200 }), { status: 200 }));
  {
    const source = download();
    await sleep(100);
    source.stop(new Error("no memory for this model"));
    assert.ok(requests.length === 3 && requests.every((r) => r.signal.aborted));
    const failed = await failure(source.into(() => {}));
    assert.ok(failed);
  }
  // and a cancelled load stops it too, without a word more
  fresh((url, init) => new Response(body(...whole(partOf(url)), { signal: init.signal, delay: 200 }), { status: 200 }));
  {
    const load = new AbortController(), source = download(load.signal);
    await sleep(100);
    load.abort();
    const failed = await failure(source.into(() => {}));
    assert.equal(failed?.error.name, "AbortError");
    assert.equal(requests.length, 3);
    assert.ok(!messages.some((m) => m.type === "progress" && m.received === model.bytes));
  }
  ok("the load stops a download it does not want any more");

  // ---- the Cache API (T97's finding, and T129's inner abort and reader.cancel() beside the copy that goes in as the part
  // streams): a Cache that does what the specification says of put(), which reads the body to its end and, where it
  // breaks, stores nothing and rejects. A part that came whole is kept for the next visit; none that did not
  {
    const kept = new Map();
    context.caches = {
      open: async () => ({
        match: async (key) => (kept.has(key) ? new Response(kept.get(key).slice(), { status: 200 }) : undefined),
        put: async (key, response) => { kept.set(key, new Uint8Array(await response.arrayBuffer())); },
        delete: async (key) => kept.delete(key),
        keys: async () => [],
      }),
    };
    const partsKept = async () => {
      await sleep(2000);  // the copies go in as the parts stream, and a little after
      return [...kept].map(([key, bytes]) => [partOf(key.split("?")[0]), bytes.length]).sort();
    };
    const keyOf = (part) => `${new URL(`models/m.${String(part).padStart(3, "0")}`, at).href}?bytes=${model.bytes}`;
    const expectKept = async (parts, what) => {
      assert.deepEqual(await partsKept(), parts.map((part) => [part, whole(part)[1] - whole(part)[0]]), what);
      for (const part of parts) assert.deepEqual(kept.get(keyOf(part)), bytesOf(...whole(part)), `${what}: the copy of part ${part} is not the file's`);
    };
    try {
      // a part whose body broke once comes again, and one whole copy is kept
      kept.clear();
      fresh((url, init, n) => new Response(body(...whole(partOf(url)), { signal: init.signal, breakAt: partOf(url) === 1 && n === 0 ? PART + 5 * MiB : undefined }), { status: 200 }));
      {
        const { into, write } = written();
        await download().into(write);
        assert.deepEqual(into, bytesOf(0, model.bytes));
        await expectKept([0, 1, 2], "a part that broke once");
      }
      // a part whose body ends short without an error is no whole: not kept, fetched again
      kept.clear();
      fresh((url, init, n) => {
        const [from, to] = whole(partOf(url));
        return new Response(body(from, partOf(url) === 1 && n === 0 ? from + 3 * MiB : to, { signal: init.signal }), { status: 200 });
      });
      {
        const { into, write } = written();
        await download().into(write);
        assert.deepEqual(into, bytesOf(0, model.bytes));
        await expectKept([0, 1, 2], "a part that ended short");
        assert.equal(requests.filter((r) => partOf(r.url) === 1).length, 2);
      }
      // a copy that is short (kept before T97's check) is not read again after it failed
      kept.clear();
      kept.set(keyOf(1), bytesOf(PART, PART + 3 * MiB));
      fresh(plain);
      {
        const { into, write } = written();
        await download().into(write);
        assert.deepEqual(into, bytesOf(0, model.bytes));
        await expectKept([0, 1, 2], "a short copy");
        assert.equal(requests.filter((r) => partOf(r.url) === 1).length, 1, "the short copy was not left for the network's answer");
      }
      // a load cancelled, or a write refused, while parts stream: no part that did not come whole is kept
      kept.clear();
      fresh((url, init) => new Response(body(...whole(partOf(url)), { signal: init.signal, delay: 200 }), { status: 200 }));
      {
        const load = new AbortController(), source = download(load.signal);
        await sleep(300);
        load.abort();
        await failure(source.into(() => {}));
        assert.deepEqual(await partsKept(), [], "a cancelled load kept parts that did not come whole");
      }
      kept.clear();
      fresh((url, init) => new Response(body(...whole(partOf(url)), { signal: init.signal, delay: 20 }), { status: 200 }));
      {
        const failed = await failure(download().into((offset) => {
          if (offset >= PART) throw new RangeError("Invalid typed array length: 4188160");
        }));
        assert.equal(failed?.error.name, "RangeError");
        assert.deepEqual(await partsKept(), [], "a refused write kept parts that did not come whole");
      }
    } finally {
      context.caches = undefined;
    }
  }
  ok("the Cache API keeps the parts that came whole, and none that did not");
}

// ---- huggingface.co: fetchRange(), inOrder(), refused() (T112, T119, T129 (3) and (5))
{
  const HF = "https://huggingface.co/owner/repo/resolve/0123456789abcdef0123456789abcdef01234567/model.safetensors";
  let size = 3 * PART + 12345;
  const ranged = (init, { delay = 0 } = {}) => {
    const [, from, to] = /bytes=(\d+)-(\d+)/.exec(init.headers.Range);
    const end = Math.min(size, Number(to) + 1);
    return new Response(body(Number(from), end, { signal: init.signal, delay }),
      { status: 206, headers: { "Content-Range": `bytes ${from}-${end - 1}/${size}` } });
  };
  const fed = () => {
    const into = new Uint8Array(size);
    let at = 0;
    return { into, feed: (bytes) => { into.set(bytes, at); at += bytes.length; }, at: () => at };
  };

  fresh((url, init) => ranged(init));
  {
    const { into, feed, at } = fed(), load = new AbortController();
    await context.inOrder(HF, 0, size, feed, load.signal);
    assert.equal(at(), size);
    assert.deepEqual(into, bytesOf(0, size));
    assert.equal(getEventListeners(load.signal, "abort").length, 0, "a file that came kept a listener on the load's signal");
  }
  ok("a file of huggingface.co comes whole and in order");

  // (3) a range that failed for good (or a feed the converter refused) stops the rest, and the connections that wait for
  // room are woken: nothing is fetched afterwards (a file of about twenty parts, whose bodies are made as they are read)
  size = 40 * PART;
  for (const [name, routing, feeding] of [
    ["a range answered 404", (url, init) => (/bytes=(\d+)/.exec(init.headers.Range)[1] === String(PART) ? new Response("", { status: 404 }) : ranged(init, { delay: 100 })), undefined],
    ["a feed the converter refused", (url, init) => ranged(init, { delay: 20 }), (bytes, n) => { if (n === 2) throw new Error("not a safetensors file"); }],
  ]) {
    fresh(routing);
    let n = 0;
    const load = new AbortController();
    const failed = await failure(context.inOrder(HF, 0, size, (bytes) => feeding?.(bytes, n++), load.signal));
    assert.ok(failed, `${name}: inOrder() did not fail`);
    assert.equal(getEventListeners(load.signal, "abort").length, 0, `${name}: a call that failed kept a listener on the load's signal`);
    await sleep(3000);
    assert.deepEqual(requests.filter((r) => r.at > failed.at).map((r) => r.range), [], `${name}: ranges were fetched afterwards`);
    assert.ok(requests.every((r) => r.signal.aborted || r.range.startsWith(`bytes=${PART}-`)), `${name}: ranges in flight were not aborted`);
  }
  ok("a range that failed for good, or a feed refused, stops the other connections");

  // (5) 429 is refused at once in its own words, 408 and 5xx are asked again
  for (const [status, times, words] of [[429, 1, /fewer requests for a while.*owner\/repo/], [401, 1, /no public repository/],
    [408, 3, /answered 408/], [503, 3, /answered 503/]]) {
    fresh(() => new Response("", { status }));
    const failed = await failure(context.fetchRange(HF, 0, 100, new AbortController().signal));
    assert.match(failed?.error.message ?? "", words, `${status}: ${failed?.error.message}`);
    assert.equal(requests.length, times, `${status} was asked ${requests.length} times, not ${times}`);
  }
  fresh((url, init, n) => (n === 0 ? new Response("", { status: 408 }) : ranged(init)));
  assert.deepEqual(new Uint8Array((await context.fetchRange(HF, 0, 100, new AbortController().signal)).bytes), bytesOf(0, 100));
  ok("429 is told as a limit to wait for, 408 is asked again");
}

// ---- watchArrivals() (T118, T129 (6))
{
  fresh((url, init) => new Response(body(0, 3 * MiB, { signal: init.signal, delay: 1000 }), { status: 200 }));
  const plain = context.fetch;
  const watch = run("watchArrivals()");
  assert.notEqual(context.fetch, plain);
  const res = await context.fetch("https://cdn.jsdelivr.net/pyodide/v1.0.0/full/pyodide.asm.wasm");
  assert.equal(res.status, 200);
  const bytes = new Uint8Array(await res.arrayBuffer());
  assert.deepEqual(bytes, bytesOf(0, 3 * MiB));
  assert.equal(watch.arrived, 1 + 3 * MiB, "the body's bytes were not counted");
  const quietSpell = watch.quiet(quiet), began = clock.now();
  await quietSpell.promise;
  assert.ok(clock.now() - began >= quiet * 1000);
  watch.stop();
  assert.equal(context.fetch, plain);
  ok("what arrives while Pyodide loads is counted, and nothing for 30 s is a stop");

  // (the review) a worker that was frozen (a phone switched to another app) or busy for longer than the 30 seconds has
  // run no tick, and tells nothing of the line: the silence is counted in the ticks that ran, not in the clock's
  // seconds. Here the event loop is held for 45 seconds of the worker's clock in the middle of the spell
  {
    const watch = run("watchArrivals()"), spell = watch.quiet(quiet);
    let stopped = false;
    spell.promise.then(() => { stopped = true; });
    await sleep(3000);
    for (const until = realNow() + 45000 / SCALE; realNow() < until;);
    await sleep(2000);
    assert.equal(stopped, false, "a worker that was frozen was told its line had stopped");
    await Promise.race([spell.promise, sleep(120000)]);  // (the ticks left to run: a line that stopped is found all the same)
    assert.equal(stopped, true, "a line that stopped after the freeze was not found");
    watch.stop();
  }
  ok("a freeze of the worker is no stop of the line");

  // (6) a browser that would not make the counting stream or Response: the responses come as they were, counted once
  // (before, the wrapper threw, and every fetch of the load failed)
  for (const [name, stand] of [["TransformStream", class {
    constructor() {
      throw new TypeError("no TransformStream here");
    }
  }], ["Response", class extends Response {
    constructor(content, init) {
      if (content instanceof ReadableStream) throw new TypeError("no Response of a stream here");
      super(content, init);
    }
  }]]) {
    const made = context[name];
    context[name] = stand;
    try {
      fresh((url, init) => new Response(body(0, MiB, { signal: init.signal }), { status: 200 }));
      const watch = run("watchArrivals()");
      const res = await context.fetch("https://cdn.jsdelivr.net/pyodide/v1.0.0/full/python_stdlib.zip");
      assert.deepEqual(new Uint8Array(await res.arrayBuffer()), bytesOf(0, MiB), name);
      assert.equal(watch.arrived, 1, name);
      watch.stop();
    } finally {
      context[name] = made;
    }
  }
  ok("a browser that cannot count a body still gets it");
}

// ---- the version of Pyodide and its steps (T118, T129 (1) and (2))
{
  const JSDELIVR = "https://data.jsdelivr.com/v1/packages/npm/pyodide/resolved?specifier=latest";
  // (1) data.jsdelivr.com that never answers: given up after QUIET_SECONDS, as a step of Pyodide's
  fresh(() => "hang");
  {
    const began = clock.now();
    const failed = await failure(Promise.race([context.resolvePyodideVersion(""),
      sleep(120000).then(() => { throw new Error("still waiting for the version after 120 s"); })]));
    assert.equal(failed.error.pyodide, true, "the page would not try again without the service worker");
    assert.match(failed.error.message, /did not answer in 30 seconds/);
    assert.ok(failed.at - began >= quiet * 1000 && failed.at - began < (quiet + 10) * 1000);
  }
  // (1) a network that fails at once is told as it is, not as a wait of 30 seconds
  fresh(() => Promise.reject(new TypeError("Failed to fetch")));
  {
    const began = clock.now();
    const failed = await failure(context.resolvePyodideVersion(""));
    assert.equal(failed?.error.message, "Failed to fetch");
    assert.ok(!failed.error.pyodide && failed.at - began < 1000);
  }
  fresh(() => new Response(JSON.stringify({ version: "314.0.7" }), { status: 200 }));
  assert.equal(await context.resolvePyodideVersion(""), "314.0.7");
  fresh(() => new Response(JSON.stringify({ version: "../evil" }), { status: 200 }));
  assert.match((await failure(context.resolvePyodideVersion("")))?.error.message ?? "", /Unexpected Pyodide version/);
  fresh(() => "hang");
  assert.equal(await context.resolvePyodideVersion("?pyodide=0.29.4"), "0.29.4");
  assert.equal(requests.length, 0);
  ok("the version of Pyodide is given up after 30 s without an answer");

  // (2) pyodideSteps(): loadPackage("numpy") is asked to fetch without integrity (the owner's choice, 2026-09-28), so
  // the wheel's bytes are counted as they come even where the prefetch did not read it. The stand-in loadPackage acts
  // as Pyodide's: with integrity its fetch settles only at the end of the body (counted as one arrival then), without
  // it the body is read through the worker's (counting) fetch as it comes.
  const base = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
  const wheel = "numpy-2.2.5-cp313-cp313-pyodide_2025_0_wasm32.whl";
  const lock = new Response(JSON.stringify({ packages: { numpy: { file_name: wheel } } }), { status: 200 });
  const asked = [];
  const steps = ({ lockAnswer = () => lock.clone(), wheelDelay = 500, stall = false } = {}) => {
    fresh((url, init) => (url === `${base}pyodide-lock.json` ? lockAnswer()
      : url === base + wheel ? new Response(body(0, 2 * MiB, { signal: init.signal, delay: wheelDelay, stall }), { status: 200 })
      : new Response("", { status: 404 })));
    const loadPackage = async (names, options = {}) => {
      asked.push(options);
      if (options.checkIntegrity !== false) {
        // what an integrity fetch shows the page while the wheel comes: nothing, then one arrival at its end
        await sleep(2 * wheelDelay);
        return;
      }
      await (await context.fetch(base + wheel)).arrayBuffer();
    };
    return run("pyodideSteps")("314.0.7", async () => ({ loadPyodide: async () => ({ loadPackage, version: "314.0.7" }) }));
  };
  for (const [name, answers] of [["read ahead", {}], ["no lock", { lockAnswer: () => new Response("", { status: 404 }) }],
    ["a lock of another shape", { lockAnswer: () => new Response("{}", { status: 200 }) }]]) {
    // a slow line: the wheel's two chunks come 20 s apart (40 s in all, past QUIET_SECONDS); NumPy is waited for
    asked.length = 0;
    const slow = await failure(steps({ ...answers, wheelDelay: 20000 }));
    assert.equal(slow, undefined, `${name}: NumPy on a slow line was given up: ${slow?.error.message}`);
    assert.deepEqual(asked.map((options) => options.checkIntegrity), [false], `${name}: NumPy was fetched with integrity`);
    // a wheel that stops: given up after QUIET_SECONDS of nothing, as a step of Pyodide's
    const began = clock.now();
    const failed = await failure(steps({ ...answers, stall: true }));
    assert.match(failed?.error.message ?? "", /"NumPy" got nothing from the network for 30 seconds/, name);
    assert.equal(failed.error.pyodide, true);
    assert.ok(failed.at - began < (2 * quiet + 20) * 1000, `${name}: a stopped NumPy was given up after ${(failed.at - began) / 1000} s`);
  }
  assert.equal(context.fetch, fetchStandIn, "the steps left the counting fetch behind");
  ok("NumPy is fetched without integrity: its bytes count on a slow line, and a stop is given up after 30 s");
}

// ---- (7) weightsBuffer(): a model past even a 64-bit memory is refused before a byte of its weights comes
{
  const made = [];
  context.stand = {
    forward: {
      ...forward,
      weightsMemory: (size, options) => {
        made.push(size);
        return { memory: new WebAssembly.Memory({ initial: 1 }), base: 64 };
      },
      growMemory() {},
    },
  };
  run("forwardModule = stand.forward; jsKernels = { relaxed: true }; wideKernels = { plain: {}, shared: null }; " +
    "llama2_numpy = { KV_START: 256, OUTLIER_CHANNELS: 8 }; disabled = [];");
  const QWEN32B = [5120, 27648, 64, 40, 8, 152064, 32768];
  const refused = await failure(Promise.resolve().then(() => context.weightsBuffer(34.8e9, QWEN32B, { dtype: "int8", bias: true })));
  assert.match(refused?.error.message ?? "", /too large for a web page: it needs about \d+ GB of memory, and a browser gives a page 16 GB at most\./);
  assert.deepEqual(made, [], "a memory was made for a model no memory holds");
  // a 7B (about 9.2 GiB with its forward pass) is still placed on a 64-bit memory
  const QWEN7B = [3584, 18944, 28, 28, 4, 152064, 4096];
  context.weightsBuffer(8.1e9, QWEN7B, { dtype: "int8", bias: true });
  assert.equal(made.length, 1);
  ok("a model past a 64-bit memory is refused before its weights come");

  // ---- load() as far as the weights' place: the parts that are on their way stop where the load ends before the
  // weights have one (T129 (3): the memory said no; the review: the runtime never came). A model of this site's kind
  // with a Qwen2.5 32B's header, whose parts are made as they are read
  const HEADER = new Uint8Array(new Int32Array(QWEN32B).buffer);
  const big = { id: "big", name: "Big", checkpoint: "big", tokenizer: "big.tokenizer.bin", bytes: 34.8e9, options: { dtype: "int8", bias: true } };
  const slowly = (url, init) => (url.endsWith("big.tokenizer.bin") ? new Response(new Uint8Array(64), { status: 200 })
    : new Response(body(partOf(url) * PART, (partOf(url) + 1) * PART, { signal: init.signal, delay: 100, head: partOf(url) === 0 ? HEADER : undefined }), { status: 200 }));
  const partRequests = () => requests.filter((r) => /\.\d{3}$/.test(r.url));
  for (const [name, why, init] of [
    ["the memory said no", /too large for a web page/, "initialized = undefined"],
    ["the runtime never came", /Pyodide did not come/, "initialized = Promise.reject(new Error('Pyodide did not come')); initialized.catch(() => {})"],
  ]) {
    fresh(slowly);
    run(init);
    const failed = await failure(context.load(big, new AbortController().signal, 1));
    assert.match(failed?.error.message ?? "", why, name);
    await sleep(3000);
    assert.deepEqual(partRequests().filter((r) => r.at > failed.at).map((r) => partOf(r.url)), [], `${name}: parts were fetched after the load ended`);
    assert.ok(partRequests().length >= 1 && partRequests().every((r) => r.signal.aborted), `${name}: the parts in flight were not aborted`);
  }
  run("initialized = undefined");
  ok("a load that ends before its weights have a place stops what is fetched for it");
}

// ---- T242: what is thrown and is no Error is told in words, not as "[object Object]" (which /benchmark/ showed)
{
  const told = context.told;
  assert.equal(told(new TypeError("Load failed")), "TypeError: Load failed");
  assert.equal(told(run("new TypeError('Load failed')")), "TypeError: Load failed");  // the worker's own realm's
  assert.equal(told(new DOMException("The operation was aborted.", "AbortError")), "AbortError: The operation was aborted.");
  class ExitStatus {  // Emscripten's: no Error, a name and a message
    name = "ExitStatus";
    constructor(status) {
      this.message = `Program terminated with exit(${status})`;
      this.status = status;
    }
  }
  assert.equal(told(new ExitStatus(1)), "ExitStatus: Program terminated with exit(1)");
  assert.equal(told({ message: "no name" }), "no name");
  assert.equal(told({ code: 7, why: "x" }), 'something that is not an error was thrown: {"code":7,"why":"x"}');
  assert.equal(told({}), "something that is not an error was thrown");
  const loop = {};
  loop.self = loop;
  assert.equal(told(loop), "something that is not an error was thrown");
  assert.equal(told(new (class Odd {})()), "Odd was thrown");
  assert.equal(told(undefined), "undefined");
  assert.equal(told("a string"), "a string");
  // and through the worker's own handler: a load whose runtime ended with such a value says it to the page
  for (const [thrown, said] of [["{ name: 'ExitStatus', message: 'Program terminated with exit(1)', status: 1 }", "ExitStatus: Program terminated with exit(1)"],
    ["{ code: 7 }", 'something that is not an error was thrown: {"code":7}'], ["undefined", "undefined"]]) {
    fresh(() => new Response(new Uint8Array(64), { status: 200 }));
    run(`initialized = Promise.reject(${thrown}); initialized.catch(() => {})`);
    await context.onmessage({ data: { type: "load", load: 9, model: { id: "small", name: "Small", checkpoint: "small", tokenizer: "small.tokenizer.bin", bytes: 64, options: {} } } });
    const error = messages.find((m) => m.type === "error");
    assert.equal(error?.message, said);
    assert.ok(!/\[object /.test(`${error.message} ${error.stack}`), error.stack);
  }
  run("initialized = undefined");
  ok("a thrown value that is no Error is told by its name and message, or its fields");

  // Pyodide's runtime that ends as it starts (its standard library did not arrive: loadPyodide() goes on without it, and
  // Python exits) rejects with Emscripten's ExitStatus: told as a step of Pyodide's that stopped; an Error stays itself
  fresh(() => new Response("", { status: 404 }));
  const ended = await failure(run("pyodideSteps")("314.0.7", async () => ({ loadPyodide: () => Promise.reject(new ExitStatus(1)) })));
  assert.equal(ended?.error.message,
    'Pyodide 314.0.7: "the runtime" ended as it started (ExitStatus: Program terminated with exit(1)): one of its files may not have arrived');
  assert.equal(ended.error.pyodide, true);
  const broke = await failure(run("pyodideSteps")("314.0.7", async () => { throw new TypeError("Importing a module script failed."); }));
  assert.equal(String(broke?.error), "TypeError: Importing a module script failed.");
  assert.equal(context.fetch, fetchStandIn, "the steps left the counting fetch behind");
  ok("a runtime of Pyodide's that ended as it started is told as a step that stopped");
}

console.log(`worker-check: ${passed} checks passed`);
