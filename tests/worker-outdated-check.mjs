// T.439: what the model's worker tells the page where it goes on, slower, without a file of its build that is gone
// (src/runtime/worker/outdated.js), by itself and where weights.js hands it a worker's error.
//
//   node tests/worker-outdated-check.mjs        (Node only, under a second)
//
// It sees:
// - outdated.js as the module it is, on a made-up server: the page is told { type: "updated" } where a built file's
//   own answer was the 404 of this build (built()'s updated()), where a module failed and the module that names it is
//   answered 404, where a worker did not start without a word and the module that made it is answered 404; and is told
//   nothing for a 200, a 500, a 503, a line that dropped, a page that is offline, a server that never answers, a file
//   that is not of a build (a development server, Node), a worker's script that threw; that it is told once a
//   worker, that failures at the same time share one HEAD, that nothing is asked where nothing needs asking, and that
//   a "no" is not kept (a deployment may come later in the visit);
// - weights.js in the worker's vm (tests/worker-harness.mjs): a software thread that does not start fails as it did
//   (the engine goes on with fewer), a GPU's worker is handed to forward.js as it was (its onerror is forward.js's to
//   set), and each hands its error event to outdated.js with weights.js's own address.
// It does not see: runtime.js's init() (Pyodide comes from its CDN: the kernels' and forward.js's catches are read as
// text by tests/updated-check.mjs, and run in tests/updated-in-browser.mjs), the page (tests/page-parts-check.mjs),
// a browser.
import assert from "node:assert/strict";
import { madeUpNetwork, status, failing, hang } from "./fakes/fetch.mjs";
import { workers } from "./fakes/worker.mjs";
import { workerHarness } from "./worker-harness.mjs";
import { runtimeUrl } from "./tree.mjs";
import { updated } from "../src/runtime/updated.ts";

const here = { origin: "https://takano32.github.io" };
const WORKER = "https://takano32.github.io/pyodide-llm/_astro/worker.BwFFQ8AO.js";
const posted = [];
globalThis.postMessage = (message) => posted.push(message);
let made = 0;
// outdated.js anew: what it holds (whether the page was told) is a visit's
const fresh = () => import(`${runtimeUrl("worker/outdated.js")}?case=${made++}`);
// a server (tests/fakes/fetch.mjs): what it was asked, and its answer to each (route(url, init, n): the n-th of that address)
function server(route) {
  const { fetch, requests: asked, answers } = madeUpNetwork();
  answers(route);
  return { asked, probe: { fetch, here } };
}
const NOT_GONE = [
  ["the namer is there", status(200)], ["the server fails", status(500)], ["the server is busy", status(503)], ["forbidden", status(403)],
  ["the line dropped", failing(new TypeError("Failed to fetch"))], ["offline", failing(new TypeError("NetworkError when attempting to fetch resource."))],
  ["blocked", failing(new DOMException("blocked", "SecurityError"))],
];
const importFailed = new TypeError("Failed to fetch dynamically imported module: https://takano32.github.io/pyodide-llm/_astro/forward.CSgc7CPR.js");
const UPDATED = [{ type: "updated" }];
let cases = 0;

// ---- a built file: its own answer says it
{
  const { wentWithout } = await fresh();
  const { asked, probe } = server(status(404));
  for (const failure of [new Error("Could not fetch simdkernel_plain.wasm: 500"), new Error("Could not fetch simdkernel.so: 404"), new TypeError("Failed to fetch"),
    new DOMException("aborted", "AbortError"), undefined, null, "updated", { updated: "yes" }]) {
    posted.length = 0;
    assert.equal(await wentWithout(failure, undefined, probe), false, `${failure?.message ?? failure}: no update`);
    assert.deepEqual(posted, [], `${failure?.message ?? failure}: the page is told nothing`);
    cases += 1;
  }
  assert.equal(await wentWithout(updated(), undefined, probe), true);
  assert.deepEqual(posted, UPDATED, "a built file whose own answer was this build's 404: the page is told, that and nothing else");
  assert.equal(asked.length, 0, "and no server is asked anything: the answer was the answer");
  // told once: the next kernel that is gone, the module, a worker
  assert.equal(await wentWithout(updated(), undefined, probe), true);
  assert.equal(await wentWithout(importFailed, WORKER, probe), true);
  assert.deepEqual(posted, UPDATED, "the page is told once a worker");
  assert.equal(asked.length, 0, "and once it was told nothing is asked again");
  cases += 2;
}
console.log("worker-outdated-check: a built file that is gone by its own answer is told, once, with nothing asked; any other failure of one is not");

// ---- a module: the module that names it is asked for
{
  for (const [what, answer] of NOT_GONE) {
    const { wentWithout } = await fresh();
    const { asked, probe } = server(answer);
    posted.length = 0;
    assert.equal(await wentWithout(importFailed, WORKER, probe), false, what);
    assert.deepEqual(posted, [], `${what}: the page is told nothing`);
    assert.equal(asked.length, 1, `${what}: asked once`);
    cases += 1;
  }
  const { wentWithout } = await fresh();
  const { asked, probe } = server(status(404));
  posted.length = 0;
  assert.equal(await wentWithout(importFailed, WORKER, probe), true);
  assert.deepEqual(posted, UPDATED, "a module that failed, and the module that names it is gone: the page is told");
  assert.deepEqual(asked.map(({ url, method, init }) => ({ address: url, method, cache: init.cache })), [{ address: WORKER, method: "HEAD", cache: "no-store" }],
    "one HEAD for the namer, past the caches (updated.js's gone())");
  // a server that never answers: given up, and nothing told
  {
    const { wentWithout } = await fresh();
    const { asked, probe } = server(hang);
    posted.length = 0;
    assert.equal(await wentWithout(importFailed, WORKER, { ...probe, seconds: 0.05 }), false);
    assert.ok(asked.length === 1 && asked[0].signal.aborted && posted.length === 0, "a HEAD that is not answered is given up, and the page is told nothing");
  }
  // what is not of a build: a development server's source, Node's file
  for (const namer of ["http://localhost:8080/src/runtime/worker/runtime.js", "file:///home/someone/pyodide-llm/src/runtime/worker/runtime.js", import.meta.url]) {
    const { wentWithout } = await fresh();
    const { asked, probe } = server(status(404));
    posted.length = 0;
    assert.equal(await wentWithout(importFailed, namer, { ...probe, here: { origin: new URL(namer).origin } }), false, namer);
    assert.deepEqual([posted.length, asked.length], [0, 0], `${namer} is no build's file: nothing asked, nothing told`);
    cases += 1;
  }
  // and by default (no probe) it is the global fetch and location: in Node there is no location, and nothing is told
  {
    const { wentWithout, workerWentWithout } = await fresh();
    posted.length = 0;
    assert.equal(await wentWithout(importFailed, WORKER), false);
    assert.equal(await workerWentWithout({}, WORKER), false);
    assert.deepEqual(posted, []);
  }
  cases += 3;
}
console.log("worker-outdated-check: a module that failed is told where the module that names it is answered 404, and for nothing else");

// ---- a worker that did not start
{
  for (const [what, answer] of NOT_GONE) {
    const { workerWentWithout } = await fresh();
    const { asked, probe } = server(answer);
    posted.length = 0;
    assert.equal(await workerWentWithout({}, WORKER, probe), false, what);
    assert.deepEqual(posted, [], `${what}: the page is told nothing`);
    assert.equal(asked.length, 1);
    cases += 1;
  }
  for (const event of [{}, { message: undefined }, { message: "" }, undefined]) {
    const { workerWentWithout } = await fresh();
    const { asked, probe } = server(status(404));
    posted.length = 0;
    assert.equal(await workerWentWithout(event, WORKER, probe), true);
    assert.deepEqual(posted, UPDATED, "a worker that did not start and says nothing, and the module that made it is gone: the page is told");
    assert.equal(asked[0].url, WORKER);
    cases += 1;
  }
  // a script that threw says what: no missing file, whatever the server would say
  {
    const { workerWentWithout } = await fresh();
    const { asked, probe } = server(status(404));
    posted.length = 0;
    assert.equal(await workerWentWithout({ message: "Uncaught ReferenceError: x is not defined" }, WORKER, probe), false);
    assert.deepEqual([posted.length, asked.length], [0, 0], "a worker whose script threw: nothing asked, nothing told");
  }
  // three threads that do not start at once (a search that starts them) share one HEAD, and the page is told once
  {
    const { workerWentWithout, wentWithout } = await fresh();
    let answer;
    const waiting = new Promise((resolve) => { answer = resolve; });
    const { asked, probe } = server(() => waiting);
    posted.length = 0;
    const three = [workerWentWithout({}, WORKER, probe), workerWentWithout({}, WORKER, probe), wentWithout(importFailed, WORKER, probe)];
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(asked.length, 1, "failures at the same time ask once");
    answer(new Response(null, { status: 404 }));
    assert.deepEqual(await Promise.all(three), [true, true, true]);
    assert.deepEqual(posted, UPDATED, "and the page is told once");
  }
  // a "no" is not kept: the site may be deployed again later in the visit, and the next failure asks again
  {
    const { workerWentWithout } = await fresh();
    const { asked, probe } = server((url, init, n) => new Response(null, { status: n === 0 ? 200 : 404 }));
    posted.length = 0;
    assert.equal(await workerWentWithout({}, WORKER, probe), false);
    assert.deepEqual(posted, []);
    assert.equal(await workerWentWithout({}, WORKER, probe), true);
    assert.deepEqual(posted, UPDATED);
    assert.equal(asked.length, 2);
    assert.equal(await workerWentWithout({}, WORKER, probe), true, "told already: it stays so");
    assert.equal(asked.length, 2, "and nothing more is asked");
    assert.deepEqual(posted, UPDATED);
  }
  // nothing here throws, whatever it is handed and whatever the page's side does with the word
  {
    const { workerWentWithout, wentWithout } = await fresh();
    const { probe } = server(status(404));
    for (const handed of [[undefined, undefined], [null, null], [{}, "not an address"], [{}, undefined], [{}, 7]]) {
      assert.equal(await workerWentWithout(handed[0], handed[1], probe), false);
      assert.equal(await wentWithout(handed[0], handed[1], probe), false);
    }
  }
  cases += 4;
}
console.log("worker-outdated-check: a worker that did not start is told where it says nothing and the module that made it is answered 404; one HEAD for failures at once; a no is not kept");

// ---- weights.js in the worker's vm: what it does with a worker that does not start is what it did, and beside that
// the error event goes to outdated.js with weights.js's own address. Whether that address is gone is the stand-in's
// to say here (in a vm the modules are files, of no build: updated.js's gone() itself is the part above)
// (the Worker is tests/fakes/worker.mjs's: worker.fail(event) as a browser tells a worker's error)
const { Worker: WorkerStandIn, made: workersMade } = workers();
const turn = () => new Promise((resolve) => setImmediate(resolve));
const weightsUrl = runtimeUrl("worker/weights.js").href ?? String(runtimeUrl("worker/weights.js"));
function harness(isGone) {
  const made = workerHarness();
  workersMade.length = 0;
  made.context.Worker = WorkerStandIn;
  const asked = [];
  made.context.goneStandIn = async (address) => { asked.push(String(address)); return isGone; };
  made.run("gone = goneStandIn");
  made.messages.length = 0;
  return { ...made, asked };
}
for (const [isGone, event, told, askedFor] of [
  [true, {}, UPDATED, [weightsUrl]],                                   // it did not load, and this build is gone
  [false, {}, [], [weightsUrl]],                                       // it did not load: a 500, a dropped line, offline
  [true, { message: "Uncaught SyntaxError: x" }, [], []],              // its script threw: no missing file
]) {
  const what = `a worker's error ${JSON.stringify(event)}, the build ${isGone ? "gone" : "there"}`;
  // a software thread
  {
    const { run, messages, asked } = harness(isGone);
    const data = { memory: 1, share: 1 };
    const starting = run("spawnThread")(data);
    const [worker] = workersMade;
    assert.match(worker.url, /helper\.ts$/, "the software thread's worker");
    assert.deepEqual(worker.posted, [data], "and it is started as it was");
    worker.fail(event);
    const failure = await starting.then(() => undefined, (error) => error);
    assert.equal(failure?.message, event.message ?? "a software thread did not start", `${what}: the thread's start fails as it did (the engine goes on with fewer)`);
    await turn();
    assert.deepEqual(JSON.parse(JSON.stringify(messages)), told, `${what}: what the page is told of a software thread`);
    assert.deepEqual(asked, askedFor, `${what}: weights.js's own address is what is asked for`);
    // ... and one that starts says nothing at all
    const { run: runAgain, messages: none, asked: nothing } = harness(true);
    const started = runAgain("spawnThread")(data);
    workersMade[0].onmessage({ data: "ready" });
    assert.equal(typeof (await started).terminate, "function");
    await turn();
    assert.deepEqual([none.length, nothing.length], [0, 0], "a thread that started: nothing asked, nothing told");
  }
  // the GPU's worker
  {
    const { run, messages, asked } = harness(isGone);
    const worker = run("openGpu")();
    assert.match(worker.url, /gpu\.ts$/, "the GPU's worker");
    assert.equal(worker.options?.type, "module");
    assert.equal(worker.onerror, null, "its onerror is forward.js's to set (forward/gpuside.js: the model stays on the CPU)");
    assert.deepEqual(worker.posted, [], "and nothing is sent to it here");
    let forwards;
    worker.onerror = (failed) => { forwards = failed; };
    worker.fail(event);
    assert.equal(forwards, event, "forward.js hears the error as it did");
    await turn();
    assert.deepEqual(JSON.parse(JSON.stringify(messages)), told, `${what}: what the page is told of the GPU's worker`);
    assert.deepEqual(asked, askedFor);
  }
  cases += 2;
}
// a model on the GPU alone opens the same worker (gpuOnlyBuffer): there is one place that makes it
{
  const { run } = harness(true);
  assert.equal((String(run("gpuOnlyBuffer")).match(/\bopenGpu\(\)/g) ?? []).length, 1, "a model on the GPU alone opens its GPU's worker by openGpu()");
  assert.ok(!/new Worker\(/.test(String(run("gpuOnlyBuffer"))));
  cases += 1;
}
console.log("worker-outdated-check: weights.js fails a thread's start and hands on the GPU's worker as it did, and hands each one's error to outdated.js with its own address");
console.log(`worker-outdated-check: ok (${cases} cases)`);
