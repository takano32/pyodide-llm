// tests/fakes-check.mjs (T.357.5): the stand-ins of tests/fakes/, each by itself. A check that runs the product on a
// stand-in sees what the stand-in lets it see: a Cache that stored half a part, a fetch that answered a request that
// was cancelled, a clock that moved between tokens, a worker that heard nothing would each let a broken product pass,
// and the checks that use them would not say so. What each does as the thing it stands for is held here; where each
// is simpler than that thing is at the head of its file.
//
//   node tests/fakes-check.mjs        (Node only, under a second)
import assert from "node:assert/strict";
import { cacheStorage } from "./fakes/cache.mjs";
import { madeUpNetwork, status, failing, hang, HANG, abortError } from "./fakes/fetch.mjs";
import { pyProxies } from "./fakes/pyodide.mjs";
import { tokenClock } from "./fakes/token-clock.mjs";
import { workers, ended, endedThen, READY } from "./fakes/worker.mjs";
import { standIn } from "./fakes/webgpu.mjs";
import { gpuAnswers } from "./fakes/gpu-answers.ts";
import { MODELS, planned } from "./fakes/gpu-plans.ts";

let passed = 0;
const ok = (name) => { passed++; console.log(`ok: ${name}`); };
const turn = () => new Promise((resolve) => setImmediate(resolve));
const failure = (promise) => promise.then(() => undefined, (error) => error);
const bytes = (...values) => Uint8Array.from(values);

// ---- the Cache API
{
  const { caches, cache, map, said } = cacheStorage();
  assert.equal(await caches.open("any name"), cache);
  assert.equal(await caches.open("another"), cache, "one cache under every name");
  assert.equal(await cache.match("https://site.test/a"), undefined, "nothing kept: no answer");
  await cache.put("https://site.test/a", new Response(bytes(1, 2, 3)));
  await cache.put({ url: "https://site.test/b?v=1" }, new Response("text"));
  assert.deepEqual([...map.keys()], ["https://site.test/a", "https://site.test/b?v=1"], "a key is an address or a request's url");
  assert.deepEqual(map.get("https://site.test/a"), bytes(1, 2, 3));
  // an answer is a copy: read twice, and changed by its reader, the same
  const first = new Uint8Array(await (await cache.match("https://site.test/a")).arrayBuffer());
  first[0] = 99;
  const again = await cache.match({ url: "https://site.test/a" });
  assert.deepEqual([again.status, new Uint8Array(await again.arrayBuffer())], [200, bytes(1, 2, 3)], "what a reader does with an answer changes nothing kept");
  // the query is part of the address unless it is said not to be
  assert.equal(await cache.match("https://site.test/b"), undefined);
  assert.equal(await cache.match("https://site.test/b?v=2"), undefined);
  assert.equal(await (await cache.match("https://site.test/b?v=2", { ignoreSearch: true })).text(), "text");
  assert.equal(await (await cache.match("https://site.test/b", { ignoreSearch: true })).text(), "text");
  // what a check seeds as an ArrayBuffer answers too
  map.set("https://site.test/seeded", new TextEncoder().encode("seeded").buffer);
  assert.equal(await (await cache.match("https://site.test/seeded")).text(), "seeded");
  assert.deepEqual((await cache.keys()).map(({ url }) => url), ["https://site.test/a", "https://site.test/b?v=1", "https://site.test/seeded"]);
  assert.equal(await cache.delete({ url: "https://site.test/a" }), true);
  assert.equal(await cache.delete("https://site.test/a"), false, "nothing to delete: false");
  assert.equal(await cache.delete("https://site.test/seeded"), true, "an address deletes as a request does");
  assert.deepEqual([...map.keys()], ["https://site.test/b?v=1"]);
  assert.deepEqual(said, ["open", "open", "keys", "delete https://site.test/a", "delete https://site.test/a", "delete https://site.test/seeded"], "what was asked of it, in order");
  assert.equal(await caches.delete("any name"), true);
  assert.deepEqual([map.size, said.at(-1)], [0, "delete the cache any name"]);
}
{
  // put() reads the body to its end, and where the body breaks nothing is stored and the put fails with the body's error
  const { cache, map, stored } = cacheStorage();
  const streamed = (chunks, breaks) => new Response(new ReadableStream({
    pull(controller) {
      if (chunks.length) return controller.enqueue(chunks.shift());
      return breaks ? controller.error(breaks) : controller.close();
    },
  }, { highWaterMark: 0 }));
  await cache.put("whole", streamed([bytes(1, 2), bytes(3), bytes(4, 5)]));
  assert.deepEqual(map.get("whole"), bytes(1, 2, 3, 4, 5), "every chunk of a body is in the copy");
  const broke = new TypeError("Error in input stream");
  assert.equal(await failure(cache.put("broken", streamed([bytes(1, 2), bytes(3)], broke))), broke, "a body that breaks fails the put with its error");
  // (T.495: stored, every copy put() stored, in order: none of a body that broke, and one written over is still there)
  assert.deepEqual(stored, [["whole", 5]], "a copy stored is written down; one of a body that broke is not");
  await cache.put("again", streamed([bytes(6, 7)]));
  await cache.put("again", streamed([bytes(8)]));
  assert.deepEqual([stored, map.get("again")], [[["whole", 5], ["again", 2], ["again", 1]], bytes(8)], "a copy written over is gone from the map, and still in stored");
  map.delete("again");
  assert.deepEqual([...map.keys()], ["whole"], "and leaves no copy, not the part that came");
  // a copy that is there stays as it was where the put of a new one breaks
  assert.equal(await failure(cache.put("whole", streamed([bytes(9)], broke))), broke);
  assert.deepEqual(map.get("whole"), bytes(1, 2, 3, 4, 5));
  // a put is not there before its body ended
  let more;
  const slow = new Response(new ReadableStream({ start(controller) { more = controller; } }));
  const putting = cache.put("slow", slow);
  more.enqueue(bytes(7));
  await turn();
  assert.equal(await cache.match("slow"), undefined, "a copy whose body has not ended answers nothing yet");
  more.close();
  await putting;
  assert.deepEqual(map.get("slow"), bytes(7));
}
{
  // what a browser may refuse
  const broken = cacheStorage({ broken: true });
  assert.match((await failure(broken.caches.open("x"))).message, /SecurityError/, "a Cache API that does not open");
  assert.deepEqual(broken.said, ["open"]);
  const { cache, map, said } = cacheStorage({ stuck: ["https://site.test/stuck"], unlisted: true });
  map.set("https://site.test/stuck", bytes(1));
  map.set("https://site.test/free", bytes(2));
  assert.match((await failure(cache.keys())).message, /UnknownError/, "a list that cannot be read");
  assert.match((await failure(cache.delete({ url: "https://site.test/stuck" }))).message, /Quota/, "a copy that cannot be deleted");
  assert.equal(await cache.delete({ url: "https://site.test/free" }), true);
  assert.deepEqual([[...map.keys()], said], [["https://site.test/stuck"], ["keys", "delete https://site.test/stuck", "delete https://site.test/free"]], "the stuck copy stays, the other goes, and both were asked");
}
ok("the Cache API: put() stores a body that came whole and nothing of one that broke, match() answers with a copy (by the address, or without its query), and what was asked of it is written down");

// ---- fetch
{
  let now = 5;
  const { fetch, requests, answers } = madeUpNetwork({ now: () => now });
  assert.equal((await fetch("https://site.test/nothing")).status, 404, "before any route: a 404");
  requests.length = 0;
  const routed = [];
  answers((url, init, n) => { routed.push([url, n]); return new Response(`${url} ${n}`, { status: 200 }); });
  const signal = new AbortController().signal, init = { method: "HEAD", cache: "no-store", headers: { Range: "bytes=0-7" }, signal };
  assert.equal(await (await fetch("https://site.test/a", init)).text(), "https://site.test/a 0");
  now = 9;
  assert.equal(await (await fetch(new URL("https://site.test/a"))).text(), "https://site.test/a 1", "the second request of an address");
  assert.equal(await (await fetch({ url: "https://site.test/a?x=1" })).text(), "https://site.test/a?x=1 0", "a query is another address, and a request is asked by its url");
  assert.deepEqual(routed, [["https://site.test/a", 0], ["https://site.test/a", 1], ["https://site.test/a?x=1", 0]]);
  assert.deepEqual(requests.map(({ url, at, method, range }) => [url, at, method, range]),
    [["https://site.test/a", 5, "HEAD", "bytes=0-7"], ["https://site.test/a", 9, "GET", undefined], ["https://site.test/a?x=1", 9, "GET", undefined]]);
  assert.deepEqual([requests[0].init, requests[0].signal, requests[1].init], [init, signal, {}], "the init as it was given, and its signal");
  // the counts begin again where the check forgets the requests
  requests.length = 0;
  assert.equal(await (await fetch("https://site.test/a")).text(), "https://site.test/a 0");
  // a route's promise is waited for; what a route throws or rejects with is what the fetch fails with
  answers(async () => new Response("later"));
  assert.equal(await (await fetch("x")).text(), "later");
  const dropped = new TypeError("Failed to fetch");
  answers(failing(dropped));
  assert.equal(await failure(fetch("x")), dropped, "a route that throws: the fetch fails with that");
  answers(() => Promise.reject(dropped));
  assert.equal(await failure(fetch("x")), dropped);
  answers(status(503));
  assert.deepEqual([(await fetch("x")).status, await (await fetch("x")).text()], [503, ""]);
}
{
  // cancelled requests, as a browser's fetch fails them
  const { fetch, requests, answers } = madeUpNetwork();
  let asked = 0;
  answers(() => { asked++; return new Response("answered"); });
  const before = new AbortController();
  before.abort();
  const early = await failure(fetch("x", { signal: before.signal }));
  assert.deepEqual([early?.name, asked, requests.length], ["AbortError", 0, 1], "a request cancelled already fails, is written down, and no route is asked");
  const why = new Error("the load was dropped"), reasoned = new AbortController();
  reasoned.abort(why);
  assert.equal(await failure(fetch("x", { signal: reasoned.signal })), why, "with the reason it was cancelled for");
  assert.equal(abortError(reasoned.signal), why);
  // no answer until it is cancelled
  assert.equal(hang(), HANG);
  answers(hang);
  const waiting = new AbortController();
  let settled = "waiting";
  const hung = fetch("x", { signal: waiting.signal }).then(() => { settled = "answered"; }, (error) => { settled = error.name; });
  await turn();
  await turn();
  assert.equal(settled, "waiting", "a server that never answers leaves the fetch waiting");
  waiting.abort();
  await hung;
  assert.equal(settled, "AbortError", "until it is cancelled");
  // cancelled while a slow route is on its way: fails then, and the late answer changes nothing
  let answer;
  answers(() => new Promise((resolve) => { answer = resolve; }));
  const late = new AbortController();
  const slow = failure(fetch("x", { signal: late.signal }));
  await turn();
  late.abort();
  assert.equal((await slow)?.name, "AbortError");
  answer(new Response("too late"));
  await turn();
}
ok("fetch: every request is written down and answered by the route (the n-th of its address), a route's failure is the fetch's, and a cancelled request fails with its reason, before the route where it was cancelled already");

// ---- the clock of a token
{
  const took = [3, 5, 0.5];
  let asked = 0;
  const clock = tokenClock(() => { asked++; return took.shift(); });
  assert.equal(clock(), 0, "the first token starts at 0");
  assert.equal(asked, 0, "a start asks nothing");
  assert.equal(clock(), 3, "and ends after what it took");
  assert.deepEqual([clock(), clock()], [3, 8], "no time between a token's end and the next one's start");
  assert.deepEqual([clock(), clock(), asked], [8, 8.5, 3], "what a token took is asked once, as it ends");
  // two clocks do not share their time
  const other = tokenClock(() => 1);
  assert.deepEqual([other(), other(), clock()], [0, 1, 8.5]);
}
ok("the clock of a token: read as a token starts and as it ends, it moves by what the check says the token took, and by nothing else");

// ---- a Worker's face
{
  const answered = [];
  const { Worker, factory, made, posted } = workers({ answer: (data, worker) => answered.push([data, worker]) });
  const byFactory = factory(), byNew = new Worker(new URL("https://site.test/_astro/gpu.x.js"), { type: "module" });
  assert.deepEqual(made, [byFactory, byNew]);
  assert.deepEqual([byFactory.url, byFactory.options, byNew.url, byNew.options], [undefined, undefined, "https://site.test/_astro/gpu.x.js", { type: "module" }]);
  assert.deepEqual([byNew.onmessage, byNew.onerror, byNew.terminated], [null, null, false], "as a Worker that was just made");
  // nobody listening: nothing happens
  byNew.say({ type: "early" });
  byNew.fail({ message: "early" });
  // what is posted is written down, on the worker and for all of them, and answered at once
  const first = { type: "start", n: 1 };
  byFactory.postMessage(first);
  assert.equal(answered.length, 1, "a message is answered inside postMessage()");
  byNew.postMessage({ type: "open" });
  byFactory.postMessage({ type: "stop" });
  assert.deepEqual([byFactory.posted, byNew.posted, posted], [[first, { type: "stop" }], [{ type: "open" }], [first, { type: "open" }, { type: "stop" }]]);
  assert.equal(posted[0], first, "a message is handed over as it is (not cloned)");
  assert.deepEqual(answered.map(([data, worker]) => [data.type, worker === byFactory]), [["start", true], ["open", false], ["stop", true]]);
  // what the worker says goes to the handler and the listeners, in the order they were first set
  const heard = [];
  byNew.addEventListener("message", (event) => heard.push(["first listener", event.data]));
  byNew.onmessage = (event) => heard.push(["a handler that is replaced", event.data]);
  byNew.addEventListener("message", (event) => heard.push(["second listener", event.data]));
  byNew.addEventListener("error", (event) => heard.push(["error listener", event.message]));
  byNew.onmessage = (event) => heard.push(["handler", event.data]);
  byNew.onerror = (event) => heard.push(["error handler", event.message]);
  assert.equal(typeof byNew.onmessage, "function");
  byNew.say({ type: "ready" });
  assert.deepEqual(heard, [["first listener", { type: "ready" }], ["handler", { type: "ready" }], ["second listener", { type: "ready" }]], "a message: to the listeners and the handler set last, in the turn of its first setting");
  heard.length = 0;
  byNew.fail({ message: "it threw" });
  byNew.fail();
  assert.deepEqual(heard, [["error listener", "it threw"], ["error handler", "it threw"], ["error listener", undefined], ["error handler", undefined]], "an error: to its listeners and handler, and none of a message's");
  // a handler taken away hears no more
  heard.length = 0;
  byNew.onmessage = null;
  byNew.say("after");
  assert.deepEqual(heard.map(([who]) => who), ["first listener", "second listener"]);
  byNew.terminate();
  assert.deepEqual([byNew.terminated, byFactory.terminated], [true, false]);
  assert.deepEqual(byFactory.listeners, [], "one worker's listeners are not another's");
}
{
  // a worker that never answers, by default
  const silent = workers();
  const worker = silent.factory();
  let heard = 0;
  worker.onmessage = () => heard++;
  worker.postMessage({ type: "start" });
  worker.postMessage({ type: "stop" });
  await turn();
  assert.deepEqual([heard, silent.posted.length], [0, 2]);
  // a GPU's worker that says it ended a turn after it is stopped, and nothing else
  const stopping = workers({ answer: ended }).factory();
  const said = [];
  stopping.onmessage = (event) => said.push(event.data);
  stopping.postMessage({ type: "start" });
  stopping.postMessage({ type: "stop" });
  assert.deepEqual(said, [], "not before the maker has gone on");
  await turn();
  assert.deepEqual(said, [{ type: "ended" }]);
  // and that answers the rest as the check says
  const others = [];
  const both = workers({ answer: endedThen((data) => others.push(data.type)) }).factory();
  both.onmessage = (event) => said.push(event.data);
  both.postMessage({ type: "prompt" });
  both.postMessage({ type: "stop" });
  await turn();
  assert.deepEqual([others, said], [["prompt"], [{ type: "ended" }, { type: "ended" }]], "a stop is not handed to the check's answer");
  const ready = READY({ blocks: [{ count: 16, ms: 1 }], key: "another" });
  assert.deepEqual([ready.type, ready.blocks, ready.key, ready.remembered, ready.forms], ["ready", [{ count: 16, ms: 1 }], "another", false, []]);
}
ok("a Worker's face: what it is posted is written down and answered on this thread, what it says and its errors go to the handler and the listeners in their turn, and by default it never answers");

// ---- Pyodide's proxies and a generator's face
{
  const { proxies, proxy, readable, generator, allDestroyed } = pyProxies();
  let read = 0;
  const made = proxy("a thing", { size: 3, get lazy() { read++; return read; } });
  assert.deepEqual([made.name, made.size, made.destroyed, read], ["a thing", 3, 0, 0], "a getter is not run as the proxy is made");
  assert.deepEqual([made.lazy, made.lazy], [1, 2], "and runs every time it is read");
  made.destroy();
  assert.equal(made.destroyed, 1);
  const options = readable("the options", { dtype: "int8" }), asked = readable("asked", (how) => how);
  assert.deepEqual([options.toJs(), asked.toJs({ depth: 1 })], [{ dtype: "int8" }, { depth: 1 }]);
  options.destroy();
  assert.throws(() => options.toJs(), /the options was read after it was destroyed/);
  assert.throws(() => allDestroyed(), /asked was destroyed 0 times/, "a proxy that was never let go is found");
  asked.destroy();
  assert.equal(allDestroyed(), 3);
  asked.destroy();
  assert.throws(() => allDestroyed(), /asked was destroyed 2 times/, "and one let go twice");
  proxies.length = 0;

  // a generator: next(answer) is send(answer), each request a proxy of its own
  const sent = [];
  let closed = 0;
  const conduct = (function* () {
    try {
      sent.push(yield ["text", "weights", "config.json"]);
      sent.push(yield ["range", "weights", "model.gguf", 0, 8]);
      return "the conversion";
    } finally {
      closed++;
    }
  })();
  const steps = generator(conduct);
  const one = steps.next(undefined);
  assert.deepEqual([one.done, one.value.name, sent], [false, "the request text weights config.json", []]);
  assert.throws(() => one.value.toJs(), /a request is read one level deep/);
  assert.throws(() => one.value.toJs({ depth: 2 }), /a request is read one level deep/);
  assert.deepEqual(one.value.toJs({ depth: 1 }), ["text", "weights", "config.json"]);
  one.value.destroy();
  assert.throws(() => one.value.toJs({ depth: 1 }), /was read after it was destroyed/);
  const two = steps.next("{}");
  assert.notEqual(two.value, one.value, "a step's value is a proxy of its own");
  assert.deepEqual([sent, two.value.toJs({ depth: 1 })], [["{}"], ["range", "weights", "model.gguf", 0, 8]]);
  two.value.destroy();
  assert.deepEqual(steps.next("bytes"), { done: true, value: "the conversion" }, "the generator's end is its return value, and no proxy");
  assert.deepEqual([steps.answers, steps.returned, steps.afterReturn, closed], [[undefined, "{}", "bytes"], 0, 0, 1]);
  assert.deepEqual(steps.return(), { done: true, value: undefined });
  steps.next("late");
  assert.deepEqual([steps.returned, steps.afterReturn], [1, 1], "a next() after return() is counted");
  steps.destroy();
  assert.throws(() => steps.next(), /next\(\) of a generator that was destroyed/);
  assert.throws(() => steps.return(), /return\(\) of a generator that was destroyed/);
  assert.equal(allDestroyed(), 3);

  // return() in the middle runs the conduct's finally; closing() is what a finally of Python's that throws would do
  let finallyRan = 0;
  const open = generator((function* () { try { yield ["size", "weights", "x"]; } finally { finallyRan++; } })(), { closing() { throw new Error("would not be closed"); } });
  open.next().value.destroy();
  assert.throws(() => open.return(), /would not be closed/);
  assert.deepEqual([finallyRan, open.returned], [1, 1], "the conduct was closed, and the failure of its closing thrown");
}
ok("Pyodide's proxies: each counts how it was let go and refuses to be read afterwards; a generator's face sends answers with next(), hands each request as a proxy to read one level deep, and closes with return()");

// ---- WebGPU (T.568.1)
{
  const { calls, device, adapter } = standIn({ fallback: true, language: ["subgroup_id"] });
  assert.equal(await navigator.gpu.requestAdapter(), adapter, "navigator.gpu is the stand-in's");
  assert.equal(adapter.info.isFallbackAdapter, true);
  assert.deepEqual([...navigator.gpu.wgslLanguageFeatures], ["subgroup_id"]);
  assert.equal(await adapter.requestDevice(), device);
  const a = device.createBuffer({ size: 32, usage: 0x80 }), b = device.createBuffer({ size: 16, usage: 0x1 });
  device.queue.writeBuffer(a, 8, new Uint32Array([1, 2, 3, 4]), 1, 2);
  assert.deepEqual([...new Uint32Array(a.data.buffer, 8, 2)], [2, 3], "a typed array's offset and size count its elements");
  assert.throws(() => device.queue.writeBuffer(a, 2, new Uint8Array(4)), /whole words/);
  assert.throws(() => device.queue.writeBuffer(a, 0, new Uint8Array(6)), /whole words/);
  assert.throws(() => device.queue.writeBuffer(a, 24, new Uint32Array(4)), /past its buffer/);
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(a, 8, b, 4, 8);
  encoder.clearBuffer(a, 8, 4);
  // (T.465.1's review: a command buffer does its work when it is submitted, in order: nothing yet)
  assert.deepEqual([...new Uint32Array(b.data.buffer, 4, 2), ...new Uint32Array(a.data.buffer, 8, 2)], [0, 0, 2, 3], "nothing done as it is encoded");
  const pipeline = await device.createComputePipelineAsync({ compute: { module: device.createShaderModule({ code: "WGSL" }) } }), group = device.createBindGroup({ entries: [{ binding: 0, resource: { buffer: a } }] });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(3, 2, 1);
  device.queue.submit([encoder.finish()]);
  assert.deepEqual([...new Uint32Array(b.data.buffer, 4, 2)], [2, 3], "a copy moves the bytes, before the clear after it");
  assert.deepEqual([...new Uint32Array(a.data.buffer, 8, 2)], [0, 3], "a clear zeroes its range");
  const range = b.getMappedRange();
  new Uint8Array(range)[4] = 99;
  assert.equal(b.data[4], 2, "a mapped range is a copy");
  b.destroy();
  assert.deepEqual(calls.map(([what]) => what), ["buffer", "buffer", "write", "copy", "clear", "pipeline", "dispatch", "submit", "destroy"], "every call written down, in order");
  const dispatched = calls.find(([what]) => what === "dispatch");
  assert.deepEqual([dispatched[1].code, dispatched[2], dispatched.slice(3)], ["WGSL", group, [3, 2, 1]], "a dispatch with its pipeline's WGSL and its bind group");
}
ok("WebGPU: navigator.gpu's adapter and device; a write of whole words within its buffer, counted in a typed array's elements; a copy, a clear, a pipeline, a dispatch and a submission written down, and a command buffer's work done as it is submitted");

// ---- T.465.1: the shaders' answers (tests/gpu_answers.py through the bridge), each on numbers whose answer is known; a
// dispatch answered where its WGSL is named
{
  const { flashTile, quantize, tiled } = await gpuAnswers();
  const made = (size, value) => ({ size, data: new Uint8Array(value ?? size) });
  const filled = (array) => ({ size: array.byteLength, data: new Uint8Array(array.buffer) });
  const entries = (...buffers) => buffers.map((buffer, binding) => ({ binding, resource: buffer }));
  const f = (buffer) => new Float32Array(buffer.data.buffer);
  // the attention: one position of one head attended is its value; two of the same key are their values' mean
  const half = (x) => ({ 1: 0x3c00, 2: 0x4000, 3: 0x4200, 0: 0 })[x];
  const keys = filled(new Uint16Array([half(1), half(1)])), values = filled(new Uint16Array([half(2), half(3)])), out = made(8);
  const params = new ArrayBuffer(16);
  new Uint32Array(params).set([1, 1, 0, 0]);
  new Float32Array(params, 8, 1)[0] = 1;
  flashTile(1)(entries(filled(new Float32Array([1, 1])), keys, values, out, filled(new Uint8Array(params)), filled(new Uint32Array([2, 0, 0, 0]))), 1, 1);
  assert.deepEqual([...f(out)], [2, 2.5], "a token at 0 attends its own value; at 1 the mean of two of the same key");
  // the quantization: quantize_x's codes and scale (the largest of a group of 32 is 127)
  const x = filled(new Float32Array(32).map((_, i) => i - 16)), xq = made(32), xs = made(4);
  quantize(entries(x, xq, xs, filled(new Uint32Array([32, 32, 0, 0])), filled(new Uint32Array([1, 0, 0, 0]))), 1, 1);
  assert.deepEqual([new Int8Array(xq.data.buffer)[0], f(xs)[0]], [-127, Math.fround(16 / 127)]);
  // a tiled form: y = W x, then y + W x where add (one row of 32 int8 weights of 2 by a scale of 0.5, x all ones)
  const w = filled(new Int8Array(32).fill(2)), s = filled(new Float32Array([0.5])), ones = filled(new Float32Array(32).fill(1)), y = made(4);
  const shape = (add) => filled(new Uint32Array([1, 8, 1, 0, 32, 1, add, 0]));
  tiled({ packed: false, packing: "int8", tile: { rows: 4, tokens: 4 } })(entries(w, s, ones, y, shape(0), filled(new Uint32Array([1, 0, 0, 0]))), 1, 1);
  tiled({ packed: false, packing: "int8", tile: { rows: 4, tokens: 4 } })(entries(w, s, ones, y, shape(1), filled(new Uint32Array([1, 0, 0, 0]))), 1, 1);
  assert.equal(f(y)[0], 64, "32 and then 32 more");
  const none = made(4);
  tiled({ packed: false, packing: "int8", tile: { rows: 4, tokens: 4 } })(entries(w, s, ones, none, shape(0), filled(new Uint32Array([1, 0, 0, 0]))), 0, 1);
  assert.equal(f(none)[0], 0, "no workgroup dispatched: nothing written");
  // the answerer: a pipeline by its WGSL or itself; another is left alone
  const answers = await gpuAnswers(), said = [], pipeline = { code: "B" };
  answers.add("A", () => said.push("A"));
  answers.addPipeline(pipeline, () => said.push("B itself"));
  for (const p of [{ code: "A" }, pipeline, { code: "C" }]) answers.answer(p, []);
  assert.deepEqual([said, answers.seen], [["A", "B itself"], ["A", "B"]]);
  // the plans: every attending layer's matrices with their int8 bytes and scales in the memory laid out
  const { plan, bytes } = planned(MODELS.llama);
  assert.ok(plan.matrices.wq.layers.every(([w8, scales]) => w8 < bytes && scales < bytes) && plan.layers === 2);
}
ok("the GPU's answers: the attention, the quantization and a tiled form on numbers whose answer is known, a dispatch answered by its WGSL or its pipeline; the made-up plans");

console.log(`fakes-check: ${passed} passed`);
