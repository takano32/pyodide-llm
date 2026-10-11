// tests/worker-parts-check.mjs (T378.3): the parts of the worker that hold something, each by itself.
//
// What the worker's modules keep has owners (worker/request.js: what the page asked for with a load; worker/loaded.js:
// the model loaded now; worker/runtime.js: what the visit finds once; each other module its own `let`s). The others say
// what happened, to a function of the owner, and read. Held here, with no Pyodide, no model and no network:
//
//   request.js   imported as the module it is (it imports nothing): each of its functions changes what it says and
//                nothing else; the page's switches are read from the first message alone; a round's switches are a
//                load's while the round runs and the page's again after it; nothing of it can be assigned from outside
//   loaded.js    in the worker's one context (tests/worker-harness.mjs), on a stand-in of Pyodide and of a model: the
//                Llama and forward.js's engine come together and go together, in the order a load needs (release,
//                destroy, the cycle collector, the wait for the GPU's worker to end, and only then a cancelled load ends)
//
// What these parts are still bound to: loaded.js to runtime.js's pyodide (the collector is Python's). weights.js's own
// (the memory kept from model to model, the model on the GPU alone, the models back on the CPU) is held where its
// stand-ins are: tests/worker-sink-check.mjs. worker.js's own (the run going on, the stop): tests/worker-check.mjs.
//
//   node tests/worker-parts-check.mjs        (Node only, under a second)
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { workerHarness } from "./worker-harness.mjs";

// ---- request.js, a fresh one for every case (a module is made once for a URL)
let made = 0;
const request = () => import(`${runtimeUrl("worker/request.js").href}?case=${made++}`);
const read = (r) => ({ threads: r.threadsRequest, gpu: r.gpuRequest, root: r.root, ahead: r.loadsAhead, benching: r.benching, switches: r.switchesNow() });
{
  const r = await request();
  assert.deepEqual({ ...read(r), switches: undefined }, { threads: undefined, gpu: undefined, root: undefined, ahead: undefined, benching: false, switches: undefined },
    "before any load was asked for, nothing was asked for");
  const first = { type: "init", search: "?model=tiny-lm&without=relaxed,%20sampler,,&kernel=off", threads: { fixed: 0, remembered: 2, hint: 8 },
    gpu: { remembered: { matrices: "DP4A" } }, root: "https://site.test/app/", ahead: [["kernels"], []], model: { id: "tiny-lm" }, load: 1 };
  r.loadAsked(first);
  assert.deepEqual(read(r), { threads: first.threads, gpu: first.gpu, root: "https://site.test/app/", ahead: [["kernels"], []], benching: false,
    switches: ["relaxed", "sampler", "kernels"] }, "a load asks for its threads, what the page kept of the GPU, the root, the loads that follow, and the page's switches");
  // the next load: what it does not say is not asked for any more, but for the root (the page's place) and the page's
  // switches (its URL as the first message said it)
  r.loadAsked({ type: "load", search: "?without=int8", model: { id: "other" }, load: 2 });
  assert.deepEqual(read(r), { threads: undefined, gpu: undefined, root: "https://site.test/app/", ahead: undefined, benching: false,
    switches: ["relaxed", "sampler", "kernels"] }, "a load that says no threads, GPU or loads ahead kept the last one's; or lost the root or the page's switches");
  r.loadAsked({ type: "load", root: "https://other.test/", ahead: "all of them" });
  assert.deepEqual([r.root, r.loadsAhead], ["https://other.test/", undefined], "a root that is said is the root; loads ahead that are no list are none");
}
{
  // ?kernel=off alone, and with the kernels already named; no switches at all; a message with no search
  for (const [search, switches] of [["?kernel=off", ["kernels"]], ["?without=kernels&kernel=off", ["kernels"]], ["?kernel=on&without=", []], ["", []], [undefined, []]]) {
    const r = await request();
    r.loadAsked({ search });
    assert.deepEqual(r.switchesNow(), switches, `the switches of ${JSON.stringify(search)}`);
  }
}
{
  // the benchmark's rounds: each round's switches are the load's while it runs; the page's again once they end, however
  const r = await request();
  r.loadAsked({ search: "?without=sampler" });
  const page = r.switchesNow();
  r.roundsBegan();
  assert.deepEqual([r.benching, r.switchesNow()], [true, ["sampler"]], "the rounds began: no round's switches yet");
  r.roundBegan(["kernels", "int8"]);
  assert.deepEqual([r.benching, r.switchesNow()], [true, ["kernels", "int8"]]);
  r.roundBegan([]);
  assert.deepEqual(r.switchesNow(), [], "a round with everything on is a round's, not the page's");
  // (a change of model during the rounds is a load: it does not end the round's switches; the rounds' own end does)
  r.loadAsked({ search: "?without=relaxed", threads: { fixed: 1 } });
  assert.deepEqual([r.benching, r.switchesNow()], [true, []]);
  r.roundsEnded();
  assert.deepEqual([r.benching, r.switchesNow()], [false, ["sampler"]], "after the rounds the page's switches are not back");
  assert.equal(r.switchesNow(), page, "the page's switches are one list, the same one after the rounds");
  r.roundsEnded();
  assert.deepEqual([r.benching, r.switchesNow()], [false, ["sampler"]], "rounds ended twice");
}
{
  // the page path found a count: the next loads (the rounds) are asked for on it; the rest of what was asked stays
  const r = await request();
  r.threadsFound(4);
  assert.deepEqual(r.threadsRequest, { remembered: 4 }, "a count found where nothing was asked about the threads");
  r.loadAsked({ threads: { fixed: 0, remembered: 0, hint: 8 }, gpu: { cpu: { GBps: 12 } } });
  const before = r.threadsRequest;
  r.threadsFound(2);
  assert.deepEqual([r.threadsRequest, before, r.gpuRequest], [{ fixed: 0, remembered: 2, hint: 8 }, { fixed: 0, remembered: 0, hint: 8 }, { cpu: { GBps: 12 } }],
    "a count found is remembered beside what was asked, and the page's own object is not written to");
}
{
  // nothing of it is set from outside: what it exports is read, and the functions above are the only way in
  const r = await request();
  for (const name of ["threadsRequest", "gpuRequest", "root", "loadsAhead", "benching"]) {
    assert.ok(name in r, `request.js exports no ${name}`);
    assert.throws(() => { "use strict"; r[name] = 1; }, TypeError, `${name} of request.js was assigned from outside`);
  }
  assert.deepEqual(Object.keys(r).sort(), ["benching", "gpuRequest", "loadAsked", "loadsAhead", "root", "roundBegan", "roundsBegan", "roundsEnded", "switchesNow", "threadsFound", "threadsRequest"],
    "what request.js exports: five things to read, five things that happened, one question");
}
console.log("ok: request.js by itself: a load's asking, the page's switches read once, the rounds' switches, the count the page path found; nothing assigned from outside");

// ---- loaded.js, in the worker's context: a stand-in of Pyodide (the collector) and of a model
{
  const { context, run } = workerHarness();
  const told = [];
  let released;
  const model = (name, { release = true } = {}) => ({
    name, destroy: () => told.push(`${name}: destroy`),
    ...(release && { release: () => { told.push(`${name}: release`); return new Promise((resolve) => { released = () => { told.push(`${name}: the GPU's worker ended`); resolve(); }; }); } }),
  });
  context.stand = { pyodide: { runPython: (code) => told.push(`python: ${code}`) } };
  run("pyodide = stand.pyodide");
  const loaded = () => [run("llama")?.name, run("engineNow()")];
  assert.deepEqual(loaded(), [undefined, undefined], "a model before any was built");
  // nothing loaded: nothing to let go of, and the collector is not run
  await run("letGo")(new AbortController().signal);
  assert.deepEqual(told, []);

  // forward.js runs it: the Llama and the engine come together
  context.stand.made = { llama: model("first"), outside: { engine: { threads: 4 } } };
  run("adopted(stand.made)");
  assert.deepEqual(loaded(), ["first", { threads: 4 }]);
  assert.equal(run("engineNow()"), context.stand.made.outside.engine, "the engine is the one forward.js made, not a copy");
  // let go: Python's release first (it says what forward.js holds), the proxy, the two forgotten, the collector, and
  // only then the wait for what release() returned (T205: the GPU's worker)
  let done = false;
  const going = run("letGo")(new AbortController().signal).then(() => { done = true; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(told, ["first: release", "first: destroy", "python: import gc; gc.collect()"]);
  assert.deepEqual([...loaded(), done], [undefined, undefined, false], "the model is forgotten at once, and the load waits for the GPU's worker");
  released();
  await going;
  assert.equal(done, true);

  // NumPy runs it: no engine; and it replaces what was there
  context.stand.made = { llama: model("numpy", { release: false }) };
  run("adopted(stand.made)");
  assert.deepEqual(loaded(), ["numpy", undefined], "a model NumPy runs kept an engine");
  told.length = 0;
  await run("letGo")(new AbortController().signal);
  assert.deepEqual(told, ["numpy: destroy", "python: import gc; gc.collect()"], "a model without release()");
  assert.deepEqual(loaded(), [undefined, undefined]);

  // a load cancelled while the model before it lets go: the model is let go all the same, and the load ends there
  context.stand.made = { llama: model("cancelled"), outside: { engine: {} } };
  run("adopted(stand.made)");
  told.length = 0;
  const cancelled = new AbortController();
  const ending = run("letGo")(cancelled.signal);
  cancelled.abort();
  released();
  await assert.rejects(ending, { name: "AbortError" });
  assert.deepEqual(told, ["cancelled: release", "cancelled: destroy", "python: import gc; gc.collect()", "cancelled: the GPU's worker ended"]);
  assert.deepEqual(loaded(), [undefined, undefined], "a cancelled load left the model before it");
  // and a model that could not be built leaves what was there as it was: nothing half adopted
  context.stand.made = { llama: model("whole"), outside: { engine: { threads: 2 } } };
  run("adopted(stand.made)");
  assert.throws(() => run("adopted((() => { throw new Error('the Llama was not built'); })())"), /was not built/);
  assert.deepEqual(loaded(), ["whole", { threads: 2 }]);
  // (the harness's run() is strict: a line for a variable no module has is an error, not a new global)
  assert.throws(() => run("outsideNow = undefined"), { name: "ReferenceError" }, "a name no module declares was assigned in the worker's context");
  assert.throws(() => run("state.llama = undefined"), { name: "ReferenceError" }, "the shared object is back");
}
console.log("ok: loaded.js by itself: the Llama and its engine are adopted together and let go together, in a load's order; a cancelled load lets go all the same");
console.log("worker-parts-check: ok");
