// tests/worker-parts-check.mjs (T.378.3): the parts of the worker that hold something, each by itself.
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
//   load.js      a whole load of a model of the site to the ready it reports, on stand-ins of Pyodide, llama2_numpy and
//                the memory and engine forward.js makes: the model built is adopted, the one before it let go of first,
//                a model whose GPU failed on it alone is loaded again; and timing.js's page path tells the count it found
//
// What these parts are still bound to: loaded.js to runtime.js's pyodide (the collector is Python's). weights.js's own
// (the memory kept from model to model, the model on the GPU alone, the models back on the CPU) is held where its
// stand-ins are: tests/worker-sink-check.mjs. worker.js's own (the run going on, the stop): tests/worker-check.mjs.
//
//   node tests/worker-parts-check.mjs        (Node only, under a second)
import { fresh } from "./fresh.mjs";
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { workerHarness } from "./worker-harness.mjs";

// ---- request.js, a fresh one for every case (a module is made once for a URL)
// (T.584: tests/fresh.mjs, on Node and on Bun)
const request = () => fresh(runtimeUrl("worker/request.js"));
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

// ---- T.460: clock.js, the seconds of a load's parts: a load begins with none of the one before but Pyodide's
{
  const clock = await fresh(runtimeUrl("worker/clock.js"));
  clock.took("pyodide", 3);
  clock.took("download", 2);
  clock.took("convert", 5);
  clock.took("construct", 1);
  assert.deepEqual(clock.secondsSoFar(), { pyodide: 3, download: 2, convert: 5, construct: 1 }, "a conversion's load");
  clock.loadStarted();
  clock.took("download", 0.5);
  clock.took("construct", 0.25);
  assert.deepEqual(clock.secondsSoFar(), { pyodide: 3, download: 0.5, construct: 0.25 }, "T.460: a site's model after it, without the conversion's seconds");
}
console.log("ok: clock.js by itself: a load says its own parts' seconds and Pyodide's, none of the load before");

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
  // T.329.2.2 (review): a load that took the file of a table and was cancelled before the model was built (nothing adopted)
  // leaves the file open, which only one handle can have: the next load lets go of it all the same
  context.stand.reader = { close: () => told.push("the file of a table never built on let go of") };
  run("rowsNow = stand.reader");
  await run("letGo")(new AbortController().signal);
  assert.deepEqual(told, ["the file of a table never built on let go of"]);
  assert.equal(run("rowsNow"), undefined);
  told.length = 0;

  // forward.js runs it: the Llama and the engine come together
  context.stand.made = { llama: model("first"), outside: { engine: { threads: 4 } } };
  run("adopted(stand.made)");
  assert.deepEqual(loaded(), ["first", { threads: 4 }]);
  assert.equal(run("engineNow()"), context.stand.made.outside.engine, "the engine is the one forward.js made, not a copy");
  // let go: Python's release first (it says what forward.js holds), the proxy, the two forgotten, the collector, and
  // only then the wait for what release() returned (T.205: the GPU's worker)
  // (T.329.2.2: and the file its table of every layer's inputs is read from, weights.js's, once the rest is let go of)
  context.stand.reader = { close: () => told.push("first: the file of its table let go of") };
  run("rowsNow = stand.reader");
  let done = false;
  const going = run("letGo")(new AbortController().signal).then(() => { done = true; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(told, ["first: release", "first: destroy", "python: import gc; gc.collect()"]);
  assert.deepEqual([...loaded(), done], [undefined, undefined, false], "the model is forgotten at once, and the load waits for the GPU's worker");
  released();
  await going;
  assert.equal(done, true);
  assert.deepEqual(told.slice(3), ["first: the GPU's worker ended", "first: the file of its table let go of"]);
  assert.equal(run("rowsNow"), undefined);

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
  context.stand.reader = { close: () => told.push("cancelled: the file of its table let go of") };
  run("rowsNow = stand.reader");
  const ending = run("letGo")(cancelled.signal);
  cancelled.abort();
  released();
  await assert.rejects(ending, { name: "AbortError" });
  assert.deepEqual(told, ["cancelled: release", "cancelled: destroy", "python: import gc; gc.collect()", "cancelled: the GPU's worker ended", "cancelled: the file of its table let go of"]);
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

// ---- load.js with its neighbours' stand-ins: a whole load of a model of the site, to the ready it reports. Pyodide and
// llama2_numpy are stand-ins, forward.js is the real one but for the memory it makes and the engine it builds (as
// tests/worker-sink-check.mjs has it), the line is the harness's. No model's weights are computed with.
{
  const forward = await import(runtimeUrl("forward.js"));
  const { context, run, fresh, messages, requests, sleep } = workerHarness({ root: null });
  const HEADER = [64, 128, 2, 4, 4, 512, 256], BYTES = 4096;
  const file = new Uint8Array(BYTES);
  file.set(new Uint8Array(new Int32Array(HEADER).buffer));
  const model = (id) => ({ id, name: `Small ${id}`, checkpoint: id, tokenizer: `${id}.tokenizer.bin`, bytes: BYTES, options: { dtype: "float32" } });
  const told = [], built = [];
  let engines = 0;
  const said = [];
  context.console = { ...console, warn: (line) => said.push(line), info() {} };
  context.stand = {
    forward: { ...forward, weightsMemory: () => ({ memory: new WebAssembly.Memory({ initial: 1 }), base: 0 }), growMemory() {},
      external: () => ({ engine: context.stand.engineOf?.(++engines) ?? { n: ++engines, gpu: context.stand.gpuOf?.(engines) ?? Promise.resolve("prompts on the CPU (a stand-in)"), gpuStatus: "the GPU gets ready (a stand-in)", threads: 1 } }) },
    pyodide: { version: "0.0.0-stand-in", _module: { HEAPU8: { length: 1000 } }, runPython: (code) => told.push(`python: ${code}`),
      globals: { get: () => (size) => ({ destroy() {}, getBuffer: () => ({ data: new Uint8Array(size), release() {} }) }) } },
    numpy: { OUTLIER_CHANNELS: 8, Llama: { callKwargs: (checkpoint, tokenizer, options) => {
      const made = { n: built.length + 1, backend: "a stand-in's backend", seq_len: HEADER[6], options, destroy: () => told.push(`llama ${made.n}: destroy`) };
      built.push(made);
      context.stand.asBuilt?.(made);
      return made;
    } } },
  };
  run("pyodide = stand.pyodide; llama2_numpy = stand.numpy; forwardModule = stand.forward; jsKernels = { relaxed: false }; initialized = Promise.resolve();");
  const of = (type) => JSON.parse(JSON.stringify(messages.filter((message) => message.type === type)));
  const load = async (data) => {
    fresh((url) => new Response(url.endsWith(".bin") ? new Uint8Array(8) : file.slice(), { status: 200 }));
    await context.onmessage({ data: { type: "load", root: "https://site.test/app/", ...data } });
    await sleep(50);
  };
  await load({ load: 1, model: model("first"), search: "?without=relaxed" });
  assert.deepEqual(of("error"), []);
  const [ready] = of("ready");
  assert.deepEqual({ ...ready, seconds: Object.keys(ready.seconds).sort() }, { type: "ready", load: 1, pyodide: "0.0.0-stand-in", backend: "a stand-in's backend", seq_len: 256,
    seconds: ["construct", "download"], heap: 1000 + 65536, threads: 1, gpu: "the GPU gets ready (a stand-in)", overlapped: false },
    "the ready of a load: Pyodide's version, the engine's backend and context, the seconds of its parts, the heap with forward.js's memory, the threads, the GPU's line");
  assert.ok(Object.values(ready.seconds).every((seconds) => typeof seconds === "number" && seconds >= 0), "the seconds of a load's parts are numbers");
  assert.deepEqual([built.length, run("llama") === built[0], run("engineNow()").n, [...built[0].options.disable]], [1, true, 1, ["relaxed"]],
    "the model built is the model loaded now, with forward.js's engine, without what the page's URL switched off");
  assert.deepEqual(requests.map((r) => r.url).sort(), ["https://site.test/app/models/first.000", "https://site.test/app/models/first.tokenizer.bin"], "the model is asked for under the root the page said");
  // (the GPU's line once it is ready: told for the engine that is still the one loaded)
  assert.deepEqual(of("gpu"), [{ type: "gpu", load: 1, note: "prompts on the CPU (a stand-in)" }]);

  // the next model: the first is let go of before the second is built (two never have to fit), and the memory stays
  told.length = 0;
  context.stand.asBuilt = () => told.push("the next is built");
  run("took('convert', 9)");  // (T.460: as a conversion before it would have left)
  await load({ load: 2, model: model("second") });
  assert.deepEqual(told, ["llama 1: destroy", "python: import gc; gc.collect()", "the next is built"]);
  assert.ok(!("convert" in of("ready")[0].seconds), "T.460: the next load's ready says none of the conversion's seconds before it");
  assert.deepEqual([of("ready").length, of("ready")[0].load, run("llama") === built[1], run("engineNow()").n, of("ready")[0].seconds.download >= 0], [1, 2, true, 2, true]);

  // T.156: a model on the GPU alone whose GPU fails as it gets ready is loaded again, on the CPU. (The place on the GPU
  // alone is tests/worker-sink-check.mjs's to make: here the model is said to be on it as its engine is built, and its
  // GPU to have failed.) load() asks weights.js with the engine it just adopted, and loads again where the answer is no
  let ended = 0;
  context.stand.direct = { lost: "the GPU could not be opened (a test)", end: async () => { ended += 1; } };
  context.stand.asBuilt = (made) => { if (made.n === 3) run("gpuOnlyNow = stand.direct"); };
  await load({ load: 3, model: model("alone") });
  assert.deepEqual(of("error"), []);
  assert.deepEqual([built.length, ended, run("cpuOnly.has('alone')"), run("gpuOnlyLost()"), of("ready").map((r) => r.load), run("llama") === built[3]], [4, 1, true, undefined, [3], true],
    "a model on the GPU alone whose GPU failed was not loaded again on the CPU: built twice, its GPU's worker ended, one ready, on the CPU from now on");
  assert.ok(said.some((line) => /the model was on the GPU alone, and is loaded again on the CPU/.test(line)), "the console did not say why the model is loaded again");
  context.stand.asBuilt = undefined;

  // T.148: a GPU that gets ready after its model was let go of says nothing to the page (its line is of another engine)
  let late;
  const before = engines;
  context.stand.gpuOf = (n) => (n === before + 1 ? new Promise((resolve) => { late = resolve; }) : undefined);
  await load({ load: 5, model: model("late") });
  await load({ load: 6, model: model("after") });
  late("prompts on WebGPU (too late)");
  await sleep(50);
  assert.deepEqual(of("gpu").map(({ load, note }) => [load, note]), [[6, "prompts on the CPU (a stand-in)"]], "the GPU of a model let go of told the page its line");
  context.stand.gpuOf = undefined;

  // T.380.1: an engine with no GPU's side (no WebGPU in a worker here): the status line says the engine's own reason,
  // which is written once, where the engine can read it (forward/reasons.ts: the worker's entry cannot)
  context.stand.engineOf = (n) => ({ n, gpu: undefined, gpuWhyNot: "the engine's reason (a stand-in)", threads: 1 });
  await load({ load: 60, model: model("plain") });
  assert.deepEqual([of("error"), of("ready").map((r) => r.gpu), of("gpu")], [[], ["prompts on the CPU (the engine's reason (a stand-in))"], []],
    "the line of an engine without a GPU's side is the engine's reason, and no GPU is waited for");
  context.stand.engineOf = undefined;

  // T.380.2: software threads that do not start leave the model ready on one thread, as before, and the console is told
  // why; threads that start say nothing
  {
    let stopped = 0;
    const before = said.length;
    context.stand.engineOf = (n) => ({ n, gpu: undefined, gpuWhyNot: "a stand-in", threads: 1, findThreads: async () => { throw new Error("a software thread did not start"); }, stopThreads: () => { stopped += 1; } });
    await load({ load: 61, model: model("alone") });
    assert.deepEqual([of("error"), of("ready").at(-1).load, of("ready").at(-1).threads, stopped], [[], 61, 1, 1], "the model is ready on one thread, and what had started is stopped");
    assert.deepEqual(said.slice(before), ["threads: the software threads did not start (Error: a software thread did not start): one thread"]);
    context.stand.engineOf = (n) => ({ n, gpu: undefined, gpuWhyNot: "a stand-in", threads: 2, findThreads: async () => 2, stopThreads() {} });
    await load({ load: 62, model: model("several") });
    assert.deepEqual([of("error"), of("ready").at(-1).threads, said.length], [[], 2, before + 1], "threads that start say nothing");
    context.stand.engineOf = undefined;
  }

  // T.380.2: two files of the visitor's, the second of which is no tokenizer.bin of the first: said with the last line
  // of what the engine threw, Python's own with the exception's name (failure.ts's lastLine(); failureOf() would say
  // a ValueError without its name and another exception by its whole traceback: another sentence)
  {
    const python = (type, text) => Object.assign(new Error(text), { name: "PythonError", type });
    const traceback = (last) => `Traceback (most recent call last):\n  File "<exec>", line 3, in load\n${last}\n`;
    context.stand.pyodide.toPy = () => ({ destroy() {} });
    Object.assign(context.stand.numpy, { checkpoint_dtype: () => "float32", check_tokenizer() {} });
    const two = { id: "local", name: "Two files", bytes: BYTES, options: {}, file: new File([file], "model.bin"), tokenizerFile: new File([new Uint8Array(8)], "tok.bin") };
    for (const [thrown, reason] of [
      [python("ValueError", traceback("ValueError: The tokenizer has 512 pieces, and the checkpoint 256.")), "ValueError: The tokenizer has 512 pieces, and the checkpoint 256."],
      [python("KeyError", traceback("KeyError: 'dim'")), "KeyError: 'dim'"],
      [new Error("one line"), "one line"],
      ["a string\nof two lines ", "of two lines"],
    ]) {
      context.stand.asBuilt = () => { throw thrown; };
      await load({ load: 63, model: two });
      assert.equal(of("error").at(-1)?.message, `tok.bin does not work as the tokenizer.bin of model.bin (${reason})`);
    }
    context.stand.asBuilt = undefined;
    messages.splice(0, messages.length, ...messages.filter((message) => message.type !== "error"));
  }

  // the same for a model of Hugging Face: convert() adopts what the place of its weights built, and load() asks weights.js
  // with the engine that is the model's (not undefined, not the one before): the GPU that failed loads it again
  {
    let converts = 0, ended2 = 0;
    context.stand.convert = async (model) => {
      converts += 1;
      const made = { n: 100 + converts, backend: "a stand-in's backend", seq_len: HEADER[6], destroy() {} };
      context.stand.made = { llama: made, outside: { engine: { n: 100 + converts, gpu: Promise.resolve("a stand-in"), gpuStatus: "stand-in", threads: 1 } } };
      run("adopted(stand.made)");
      if (converts === 1) run("gpuOnlyNow = stand.direct2");
      return {};
    };
    context.stand.direct2 = { lost: "the GPU could not be opened (a test)", end: async () => { ended2 += 1; } };
    run("convert = stand.convert");
    await load({ load: 7, model: { ...model("remote"), hf: { repo: "someone/remote", revision: "0".repeat(40) } } });
    assert.deepEqual(of("error"), []);
    assert.deepEqual([converts, ended2, run("cpuOnly.has('hf:someone/remote@' + '0'.repeat(40))"), of("ready").map((r) => r.load), run("engineNow().n")], [2, 1, true, [7], 102],
      "a model of Hugging Face on the GPU alone whose GPU failed was not loaded again: converted twice, its GPU's worker ended, one ready, the engine of the second");
  }

  // ---- timing.js with stand-ins: the page's path (T.184) on the model loaded now. What it tells request.js: the count
  // the search ended on, which the rounds that follow load the model on (T.190)
  const generator = () => ({ next: () => ({ done: true }), destroy() {} });
  const engine = { gpuSampled: 0, lostThreads: false, searchLog: [], gpuStatus: "prompts on the CPU (a stand-in)", setThreads: async (count) => count };
  context.stand.loaded = { outside: { engine }, llama: { backend: "a stand-in's", bos: 1, seq_len: 256, destroy() {},
    generate: { callKwargs: generator }, stats: { toJs: () => ({ sampled: 8, tokens_per_second: 100 }) },
    tokenizer: { encode: () => ({ toJs: () => [1, 2, 3], destroy() {} }) } } };
  context.stand.forward = { ...context.stand.forward, SEARCH_SECONDS: 20, endSearch: async () => ({ threads: 2, found: 2, ended: true }),
    timePrompts: () => [], timedCell: (runs) => ({ runs: runs.length }) };
  run("forwardModule = stand.forward; adopted(stand.loaded); loadAsked({ threads: { fixed: 0, remembered: 0, hint: 4 } });");
  for (const shared of [true, false]) {
    run("loadAsked({ threads: { fixed: 0, remembered: 0, hint: 4 } })");
    context.stand.pool = { shared };
    run("weightsPool = stand.pool");
    const paths = JSON.parse(JSON.stringify(await run("timedPaths")({ prompt: "Once", counts: [64], sampled: 8 })));
    assert.equal(paths.threads, 2);
    assert.deepEqual(paths.how, shared ? { searched: [] } : { alone: "no shared memory here" });
    assert.deepEqual(JSON.parse(JSON.stringify(run("threadsRequest"))), { fixed: 0, remembered: shared ? 2 : 0, hint: 4 },
      shared ? "the count the page path ended on is not what the rounds after it are asked for" : "a count was remembered where there is no shared memory");
    assert.deepEqual(paths.perCount.map((row) => row.threads), shared ? [1, 2, 4] : []);
  }
  run("weightsPool = undefined; adopted({}); initialized = undefined;");
}
console.log("ok: load.js on stand-ins: a load to its ready, the next model after the first is let go of, a model whose GPU failed on it alone loaded again; timing.js: the page path's count told");

// ---- T.380.2: runtime.js asking for the adapter as the worker begins. What fails there was silent: an adapter that
// cannot be asked for, and a device with no key (in this vm no module can be imported later, which is such a failure)
{
  const begun = async (gpu) => {
    const warn = console.warn, said = [];
    console.warn = (...words) => said.push(words.join(" "));
    try {
      const { run } = workerHarness({ root: null, gpu });
      await run("adapterAsked");
      return { said, adapter: JSON.parse(JSON.stringify(run("gpuAdapter"))) };
    } finally {
      console.warn = warn;
    }
  };
  // no adapter here: nothing failed, and nothing is said
  assert.deepEqual(await begun({ requestAdapter: async () => null }), { said: [], adapter: null });
  // the asking itself fails
  assert.deepEqual(await begun({ requestAdapter: () => Promise.reject(new Error("the GPU process is gone")) }),
    { said: ["gpu: the adapter could not be asked for (Error: the GPU process is gone): no GPU on this visit"], adapter: null });
  // an adapter that is not what one is
  const odd = await begun({ requestAdapter: async () => ({}) });
  assert.equal(odd.adapter, null);
  assert.match(odd.said.join("|"), /^gpu: the adapter could not be asked for \(TypeError: [^|]+\): no GPU on this visit$/);
  // an adapter, and no key of it: the adapter stands, without one
  const keyless = await begun({ requestAdapter: async () => ({ limits: { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28 }, info: { isFallbackAdapter: false } }) });
  assert.deepEqual(keyless.adapter, { fallback: false, limits: { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28 }, packed: false });
  assert.match(keyless.said.join("|"), /^gpu: no key of this device \([^|]+\): what the page kept of it is not used$/);
}
console.log("ok: T.380.2, an adapter that cannot be asked for and a device without a key are said on the console; no adapter at all is no failure and says nothing");
console.log("worker-parts-check: ok");
