// tests/worker-harness.mjs (T357, out of tests/worker-check.mjs where T129 wrote it)
// public/worker.js and its modules in a vm context whose global is its self, as in a worker, on a made-up network and
// a clock of the test's: what the checks that run the worker in Node share (worker-check.mjs: where fetching and loading
// meet; worker-fetches-check.mjs: what a conversion asks for).
//
//   const { context, run, requests, messages, fresh, body, bytesOf, sleep, clock, ... } = workerHarness({ told })
//
//   context, run(code)     the worker's global (its functions and `state` are its properties) and a line run in it
//   fresh(route)           forgets the requests and the messages; route(url, init, n) answers the n-th request of that
//                          URL (0 first) with a Response, a promise of one, or "hang" (no answer until it is aborted)
//   requests, messages     every request ({ url, at, signal, range, method }) and every postMessage(), in order
//   body(from, to, how)    a stream of the made-up file's bytes [from, to), bytesOf(from, to) those bytes
//   clock, sleep(ms)       the worker's clock: a hundred times as fast as the real one (30 seconds of the worker's are
//                          0.3 s here), or with { told: true } one that stands still but for sleep(ms), which moves it
//                          (a body's delay is then the only time there is: the same every run)
//
// The Cache API is the check's own stand-in where it needs one. What this cannot see is said at the head of
// tests/worker-check.mjs (the Service Worker, a real line, a browser's limits) and of tests/worker-source.mjs (a missing
// import, the wait for the modules).
import vm from "node:vm";
import { runWorker } from "./worker-source.mjs";

export function workerHarness({ told = false } = {}) {
  const SCALE = 100;  // the worker's milliseconds per real millisecond
  const MiB = 1024 * 1024, PART = 8 * MiB;
  const realNow = () => performance.now();
  // A timer that never fires before its time by performance.now(), as a browser's never does. Node's may: of 400
  // setTimeout(30) one or two fired after 28.2 to 29.9 ms on the development machine. A hundred times as fast, that
  // millisecond is a tenth of the worker's second, and "given up after 30 seconds" was measured as 29.9 (T386: a
  // run of CI fell on it). A timer that comes early waits for the rest.
  function timer(f, ms, args = [], again = false) {
    const handle = { real: null, due: realNow() + ms };
    const fire = () => {
      const left = handle.due - realNow();
      if (left > 0) { handle.real = setTimeout(fire, Math.ceil(left)); return; }
      if (again) { handle.due = Math.max(handle.due + ms, realNow()); handle.real = setTimeout(fire, handle.due - realNow()); }
      f(...args);
    };
    handle.real = setTimeout(fire, ms);
    return handle;
  }
  const clear = (handle) => clearTimeout(handle?.real ?? handle);
  // ms of the worker's clock (told: the clock is moved on by that much and a turn of the event loop goes by)
  let toldNow = 0;
  const sleep = told ? (ms) => new Promise((resolve) => { toldNow += ms; setImmediate(resolve); })
    : (ms) => new Promise((resolve) => timer(resolve, ms / SCALE));

  // the worker's clock and timers: a hundred times as fast, or (told) a clock that moves only when a sleep() moves it
  const clock = {
    now: told ? () => toldNow : () => realNow() * SCALE,
    setTimeout: (f, ms = 0, ...args) => timer(f, ms / SCALE, args),
    setInterval: (f, ms = 0, ...args) => timer(f, Math.max(ms / SCALE, 1), args, true),
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

  // worker.js and its modules (T350: tests/worker-source.mjs makes scripts of them) in a vm context whose global is its
  // self, as in a worker
  const messages = [];
  const navigatorStandIn = { deviceMemory: 8 };
  const context = vm.createContext({
    console, URL, URLSearchParams, TextDecoder, TextEncoder, AbortController, DOMException, Response, Headers,
    ReadableStream, WritableStream, TransformStream, WebAssembly, Atomics, SharedArrayBuffer,
    performance: { now: clock.now }, setTimeout: clock.setTimeout, clearTimeout: clear, setInterval: clock.setInterval, clearInterval: clear,
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
  runWorker(context);
  const run = (code) => vm.runInContext(code, context);

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

  return { SCALE, MiB, PART, realNow, sleep, clock, bytesOf, abortError, body, requests, messages, fetchStandIn, navigatorStandIn, context, run, failure, fresh, partOf };
}
