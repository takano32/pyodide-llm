// T.464 (out of tests/worker-check.ts, which T.129 wrote): src/runtime/worker.js run as it is in a vm context on made-up
// fetches and a clock that goes a hundred times as fast, past where fetching and Pyodide's steps meet (those are
// worker-check.ts's): a model past even a 64-bit memory (weightsBuffer()), and load() as far as the place of the
// weights (what it stops where it ends before it); the shared memory of a worker told which loads follow on its one
// model (T.242: /benchmark/'s model section); what is thrown and is no Error, told in words; and (T.357) a stop that
// arrives while a made-up model writes (generate() and the stop message), (T.378.3) the GPU that stops under a run of a
// model on the GPU alone. Node only, no Pyodide, a few seconds:
//
//   node tests/worker-memory-check.ts
//
// What it cannot see is said at the head of tests/worker-check.ts.
import assert from "node:assert/strict";
import fs from "node:fs";
import { pathToFileURL } from "node:url";
import { runtimeUrl, served } from "./tree.ts";
const forward = await import(runtimeUrl("forward.js"));
import { workerHarness } from "./worker-harness.ts";

// the made-up network, clock, messages and context (tests/worker-harness.ts, T.357: the conversion's fetch list uses them too)
const { PART, sleep, clock, body, requests, messages, fetchStandIn, context, run, failure, fresh, partOf } = workerHarness();
const at = runtimeUrl("worker.js");
// a check that waits for ever (a fix undone: the version asked for ever) fails rather than hangs
setTimeout(() => {
  console.error("worker-memory-check: still waiting after 180 s");
  process.exit(1);
}, 180000).unref();
let passed = 0;
const ok = (line) => {
  passed++;
  console.log(`ok: ${line}`);
};

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
    "llama2_numpy = { KV_START: 256, OUTLIER_CHANNELS: 8 }; roundBegan([]);");
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
  // weights have one (T.129 (3): the memory said no; the review: the runtime never came). A model of this site's kind
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

// ---- T.242: the shared memory of a worker that is told which loads follow on its one model (/benchmark/'s model
// section) is made for the largest of them, without the gigabyte for a next model; each of those loads then fits it
{
  const PAGE = 65536, pagesOf = (bytes) => Math.ceil(bytes / PAGE);
  context.stand.real = forward;
  const again = () => run("forwardModule = stand.real; weightsPool = weightsNow = undefined; loadsAhead = undefined");
  again();
  const size = 32891932, after = 12537888, widest = 150e6;
  const usual = context.pooledWeights(size, after, true, false);
  assert.ok(usual.shared && usual.maximum * PAGE >= usual.base + size + after + 2 ** 30, "the model page's memory keeps a gigabyte for the next model");
  again();
  const lone = context.pooledWeights(size, after, true, false, widest);
  assert.equal(lone.maximum, pagesOf(lone.base + size + widest) + 1);
  assert.equal(context.pooledWeights(size, widest, true, false, widest).memory, lone.memory, "the widest load that follows got another memory");
  assert.equal(context.pooledWeights(size, after, true, false, widest).memory, lone.memory, "a load that follows got another memory");
  // (what follows may all be smaller than the load going on)
  again();
  const least = context.pooledWeights(size, after, true, false, 0);
  assert.equal(least.maximum, pagesOf(least.base + size + after) + 1);
  // and from the page's init to the memory: the largest of the rounds that take one (NumPy's takes none)
  again();
  const HEADER = [288, 768, 6, 6, 6, 32000, 256], OPTIONS = { dtype: "int8" };
  const rounds = [[], ["kernels"], ["int8", "relaxed", "sampler", "kv16"], ["relaxed", "sampler", "kv16"]];
  run("sharedKernels = {}; jsKernels = { relaxed: true }; wideKernels = undefined; roundBegan([]); threadsRequest = undefined; " +
    "llama2_numpy = { KV_START: 256, OUTLIER_CHANNELS: 8 }");
  context.crossOriginIsolated = true;
  try {
    const footprintOf = (without) => forward.footprint(HEADER, size, { ...OPTIONS, int8: !without.includes("int8"), relaxed: !without.includes("relaxed"),
      halfKV: !without.includes("int8") && !without.includes("kv16"), shared: true, outliers: 8, gpu: false });
    const widened = footprintOf(rounds[2]);
    assert.ok(widened > footprintOf([]) && widened > footprintOf(rounds[3]), "the round with int8 widened is not the largest");
    context.rounds = rounds;
    run("loadsAhead = rounds");
    context.weightsBuffer(size, HEADER, OPTIONS);
    const pool = run("weightsPool");
    assert.equal(pool.maximum, pagesOf(pool.base + size + widened) + 1);
    // the model page's init says none: the gigabyte is there again (a larger model makes its own memory)
    again();
    context.weightsBuffer(size, HEADER, OPTIONS);
    assert.ok(run("weightsPool").maximum * PAGE >= size + 2 ** 30);
  } finally {
    context.crossOriginIsolated = false;
    run("sharedKernels = undefined; forwardModule = stand.forward; weightsPool = weightsNow = undefined; loadsAhead = undefined");
  }
  ok("a worker told the loads that follow on its one model makes its shared memory for the largest of them");
}

// ---- T.242's review: the page's message to the worker's memory. The handler takes the loads that follow from the page's message
// (/benchmark/'s init says them; the model page's loads say none). A handler that forgot the last one's would make the model
// page's memory without its gigabyte for the next model, or the benchmark's with one, and nothing else here sees the wiring:
// the memory's own checks above set loadsAhead by hand
{
  const small = { id: "small", name: "Small", checkpoint: "small", tokenizer: "small.tokenizer.bin", bytes: 64, options: {} };
  const said = async (data) => {
    fresh(() => new Response(new Uint8Array(64), { status: 200 }));
    run("initialized = Promise.reject(new Error('no runtime in this check')); initialized.catch(() => {})");
    await context.onmessage({ data: { type: "load", load: 9, model: small, ...data } });
    run("initialized = undefined");
    return run("loadsAhead");
  };
  assert.deepEqual(await said({ ahead: [[], ["kernels"], ["int8", "relaxed"]] }), [[], ["kernels"], ["int8", "relaxed"]]);
  assert.equal(await said({}), undefined, "a load that says none left the last one's loads ahead");
  assert.equal(await said({ ahead: [[]] }).then(() => said({ ahead: "int8" })), undefined, "a thing that is no list was taken for one");
  ok("the loads that follow come from the page's message, and the next load that says none forgets them");
}

// ---- T.242's review: a browser that refuses the benchmark's memory (the model's alone, a small one). It is asked for less after
// and never for more (a Windows WebKit that refused 45 MB was then asked for a gigabyte, the size its page went down in), and a
// shared memory it gave at a lowered maximum must hold the loads the page said follow, as the one going on (T.130's guard)
{
  const PAGE = 65536, RealMemory = WebAssembly.Memory, info = console.info;
  const refusing = (limit) => {
    const asked = [];
    WebAssembly.Memory = class extends RealMemory {
      constructor(descriptor) {
        asked.push(Number(descriptor.maximum ?? 0));
        if (descriptor.shared && Number(descriptor.maximum) > limit) throw new RangeError("too much address space");
        super(descriptor);
      }
    };
    return asked;
  };
  console.info = () => {};  // (the guard says what it did)
  try {
    const size = 32891932, after = 12537888;
    let asked = refusing(0);
    const refused = await failure(Promise.resolve().then(() => forward.weightsMemory(size, { shared: true, after, spare: PAGE })));
    assert.match(refused?.error.message ?? "", /no shared WebAssembly memory/);
    assert.equal(asked.length, 1, `a memory of the model alone was refused, and then the browser was asked for ${asked.slice(1).join(", ")} pages`);
    asked = refusing(0);
    await failure(Promise.resolve().then(() => forward.weightsMemory(size, { shared: true, after })));
    assert.equal(asked.length, 3, "the model page's memory has its two lesser tries still");
    assert.ok(asked[0] > asked[1] && asked[1] > asked[2], `the tries are not each less than the one before: ${asked.join(", ")}`);

    // a shared memory at a lowered maximum (weightsMemory's second try) that holds the load going on and not the largest that
    // follows is not kept: a plain one is made, as for a load that is alone (the model page's, with no loads ahead, keeps it)
    context.stand.real = forward;
    const widest = 600e6;  // past what the lesser tries hold: a quarter of a gigabyte over the checkpoint
    for (const [limit, ahead, shared] of [[Infinity, widest, true], [5000, widest, false], [5000, 100e6, true], [5000, undefined, true]]) {
      run("forwardModule = stand.real; weightsPool = weightsNow = undefined; loadsAhead = undefined");
      refusing(limit);
      const pool = context.pooledWeights(size, after, true, false, ahead);
      assert.equal(pool.shared, shared, `a browser that gives ${limit} pages, the loads ahead ${ahead}: ${pool.shared ? `a shared memory of ${pool.maximum} pages` : "a plain one"}`);
      if (pool.shared && ahead !== undefined) assert.ok(pool.maximum * PAGE >= pool.base + size + ahead, "the shared memory does not hold the loads ahead");
    }
  } finally {
    WebAssembly.Memory = RealMemory;
    console.info = info;
    run("forwardModule = stand.forward; weightsPool = weightsNow = undefined; loadsAhead = undefined");
  }
  ok("a browser that refuses the memory is asked for less and never for more, and a lowered one must hold the loads ahead too");
}

// ---- T.242's review: every load /benchmark/'s model section makes (the page's path, then each round that loads) fits the one memory
// the worker made for the largest of them, for every model that page can take (the site's own) and each set of rounds (?full or
// not; a browser that says its memory or one that does not, which skips the rounds that widen the weights, T.214). Another memory
// would be a page's third (T.96), and the memory has no gigabyte over to hide a load that does not fit. The headers are of the
// built models (make models; where there are none this says so and checks nothing)
{
  const { MODELS } = await import("../src/models.ts");
  const { ROUNDS, FULL_ROUNDS, roundsHere } = await import("../src/bench.ts");
  const PAGE = 65536, folder = pathToFileURL(served("models/", { maybe: true }));
  const headerOf = (entry) => {
    const part = new URL(`${entry.checkpoint}.000`, folder);
    if (!fs.existsSync(part)) return undefined;
    const bytes = Buffer.alloc(28), file = fs.openSync(part, "r");
    fs.readSync(file, bytes, 0, 28, 0);
    fs.closeSync(file);
    return Array.from(new Int32Array(bytes.buffer, bytes.byteOffset, 7));
  };
  const sited = MODELS.filter((one) => !one.hf && one.checkpoint), headers = sited.map(headerOf);
  if (headers.every((header) => !header)) {
    console.log(`skipped: every load of the benchmark's model section fits its memory (no built models in ${served("models", { maybe: true })})`);
  } else {
    context.stand.real = forward;
    context.crossOriginIsolated = true;
    let loads = 0;
    try {
      run("sharedKernels = {}; jsKernels = { relaxed: true }; wideKernels = undefined; threadsRequest = undefined; " +
        "llama2_numpy = { KV_START: 256, OUTLIER_CHANNELS: 8 }");
      for (const [at, entry] of sited.entries()) {
        const header = headers[at], options = { dtype: "float32", ...entry.options };
        if (!header) continue;
        for (const asked of [ROUNDS, FULL_ROUNDS]) {
          for (const deviceMemory of [8, undefined]) {
            const rounds = roundsHere(asked, deviceMemory).filter((round) => round.skip === undefined);
            run("forwardModule = stand.real; weightsPool = weightsNow = undefined");
            context.rounds = rounds.map((round) => round.without);
            run("loadsAhead = rounds");
            const memories = new Set(), what = `${entry.id}, ${asked === ROUNDS ? "the two rounds" : "every step"}, memory ${deviceMemory ?? "not said"}`;
            for (const without of [[], ...rounds.map((round) => round.without)]) {
              if (without.includes("kernels")) continue;  // (NumPy's weights are Python's: no memory of the worker's)
              context.without = without;
              run("roundBegan(without)");
              context.weightsBuffer(entry.bytes, header, options);
              const pool = run("weightsPool"), int8 = !without.includes("int8"), quantized = ["int8", "int6", "ternary"].includes(options.dtype);
              const needs = pool.base + entry.bytes + forward.footprint(header, entry.bytes, { ...options, int8, relaxed: !without.includes("relaxed"),
                halfKV: quantized && int8 && !without.includes("kv16"), shared: true, outliers: 8, gpu: false });
              assert.ok(pool.maximum * PAGE >= needs, `${what}, without ${without.join("+") || "nothing"}: a memory of ${pool.maximum} pages for a load that needs ${needs} bytes`);
              memories.add(pool.memory);
              loads++;
            }
            assert.equal(memories.size, 1, `${what}: a load of the model section made a memory of its own`);
          }
        }
      }
    } finally {
      context.crossOriginIsolated = false;
      run("sharedKernels = undefined; forwardModule = stand.forward; weightsPool = weightsNow = undefined; loadsAhead = undefined; roundBegan([])");
    }
    ok(`every load of the benchmark's model section fits the one memory made for it (${sited.length} models, ${loads} loads)`);
  }
}

// ---- T.242: what is thrown and is no Error is told in words, not as "[object Object]" (which /benchmark/ showed)
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
  // T.380.1: what the handler tells the page of a failure is what failure.ts reads of it (tests/failure-check.ts has
  // every shape): its sentence, and its kind as the page's flags (pyodide: the runtime did not load; memory, with the
  // heap: it ran out), whatever was thrown
  for (const [thrown, said, pyodide, memory] of [
    ["new Error('The model is not ready.')", "The model is not ready.", false, false],
    ["Object.assign(new Error('Traceback (most recent call last):\\n  File x\\nValueError: This is not a safetensors file.\\n'), { name: 'PythonError', type: 'ValueError' })", "This is not a safetensors file.", false, false],
    ["new RangeError('Invalid array buffer length')", "Invalid array buffer length", false, true],
    ["Object.assign(new Error('Traceback (most recent call last):\\n  File x\\npyodide.ffi.JsException: RangeError: WebAssembly.Memory.grow(): Maximum memory size exceeded\\n'), { name: 'PythonError', type: 'JsException' })",
      "WebAssembly.Memory.grow(): Maximum memory size exceeded", false, true],
    ["new RangeError('Maximum call stack size exceeded')", "RangeError: Maximum call stack size exceeded", false, false],
    ["new Failure('runtime', 'Pyodide 0.0.0: \"the runtime\" got nothing from the network for 30 seconds')", 'Pyodide 0.0.0: "the runtime" got nothing from the network for 30 seconds', true, false],
    ["new Failure('refused', 'huggingface.co answered 403 for a of b.', { status: 403, again: false })", "huggingface.co answered 403 for a of b.", false, false],
  ]) {
    fresh(() => new Response(new Uint8Array(64), { status: 200 }));
    run(`initialized = Promise.reject(${thrown}); initialized.catch(() => {})`);
    await context.onmessage({ data: { type: "load", load: 10, model: { id: "small", name: "Small", checkpoint: "small", tokenizer: "small.tokenizer.bin", bytes: 64, options: {} } } });
    const errors = messages.filter((m) => m.type === "error");
    assert.equal(errors.length, 1, `one error told of ${thrown}`);
    const [error] = errors;
    assert.deepEqual([error.message, error.load, error.pyodide, Boolean(error.memory), "heap" in error, typeof error.stack, typeof error.weights], [said, 10, pyodide, memory, memory, "string", "number"], `what the page is told of ${thrown}`);
  }
  run("initialized = undefined");
  ok("a thrown value that is no Error is told by its name and message, or its fields; a failure's kind is the page's flags");

  // T.367.3.2: the site's root comes with a load, from the page, and models/ is asked for under it (the worker's own
  // URL is under _astro/); a load that says none keeps the root it has. And the page says it with every init and load
  // (a load may reach the worker before the init does: src/page/choose.ts).
  {
    const small = { id: "small", name: "Small", checkpoint: "small", tokenizer: "small.tokenizer.bin", bytes: 64, options: {} };
    const asked = async (data) => {
      fresh(() => new Response(new Uint8Array(64), { status: 200 }));
      run("initialized = Promise.reject(new Error('no runtime here')); initialized.catch(() => {})");
      await context.onmessage({ data: { type: "load", load: 9, model: small, ...data } });
      return requests.map((r) => r.url).sort();
    };
    const before = run("root");
    assert.deepEqual(await asked({ root: "https://other.test/site/" }), ["https://other.test/site/models/small.000", "https://other.test/site/models/small.tokenizer.bin"]);
    assert.deepEqual(await asked({}), ["https://other.test/site/models/small.000", "https://other.test/site/models/small.tokenizer.bin"], "a load without a root keeps the one it was told");
    context.rootBefore = before;
    run("loadAsked({ root: rootBefore }); initialized = undefined");
    const posts = ["src/page/first.ts", "src/page/choose.ts", "src/benchmark/model.ts"].map((file) => [file, fs.readFileSync(new URL(`../${file}`, import.meta.url), "utf8").match(/\{ type: "(?:init|load)",[^}]*/g) ?? []]);
    for (const [file, found] of posts) assert.ok(found.length === 1 && /\broot\b/.test(found[0]), `${file}: the load it sends the worker says no root: ${found}`);
    ok("the site's root comes with a load and models/ is asked for under it; the page's three loads say it");
  }

  // Pyodide's runtime that ends as it starts (its standard library did not arrive: loadPyodide() goes on without it, and
  // Python exits) rejects with Emscripten's ExitStatus: told as a step of Pyodide's that stopped; an Error stays itself
  fresh(() => new Response("", { status: 404 }));
  const ended = await failure(run("pyodideSteps")("314.0.7", async () => ({ loadPyodide: () => Promise.reject(new ExitStatus(1)) })));
  assert.equal(ended?.error.message,
    'Pyodide 314.0.7: "the runtime" ended as it started (ExitStatus: Program terminated with exit(1)): one of its files may not have arrived');
  assert.equal(ended.error.kind, "runtime");
  const broke = await failure(run("pyodideSteps")("314.0.7", async () => { throw new TypeError("Importing a module script failed."); }));
  assert.equal(String(broke?.error), "TypeError: Importing a module script failed.");
  assert.equal(context.fetch, fetchStandIn, "the steps left the counting fetch behind");
  ok("a runtime of Pyodide's that ended as it started is told as a step that stopped");
}

// ---- T.357: a stop that arrives during a run stops it. Nothing ran generate() outside a browser: `stopped` spelled
// otherwise in its loop is undefined, the loop writes to the end of the context, and both checks of the worker passed (T.350)
{
  // a model that writes for ever (or `limit` pieces): to the worker, generate() is a Python generator (next, return, destroy)
  let written = 0, limit = Infinity, closed = 0, destroyed = 0;
  context.stand = { llama: {
    generate: { callKwargs: () => ({ next: () => (written < limit ? { done: false, value: `piece ${written++}` } : { done: true }), return: () => { closed++; }, destroy: () => { destroyed++; } }) },
    stats: { toJs: () => ({ tokens: written }) },
  } };
  run("adopted({ llama: stand.llama }); gpuOnlyNow = undefined; stopped = false; generating = undefined;");
  const of = (type) => messages.filter((message) => message.type === type);
  fresh(() => new Response("", { status: 404 }));
  const running = context.onmessage({ data: { type: "generate", prompt: "a prompt" } });
  await sleep(2000);
  assert.ok(of("token").length > 0 && of("done").length === 0, "the run wrote nothing, or ended by itself");
  context.onmessage({ data: { type: "stop" } });
  const stopped = await Promise.race([running.then(() => true), sleep(30000).then(() => false)]);
  if (!stopped) {
    // (let the run end, so that this fails in words and not at the 180 s)
    limit = 0;
    await running;
  }
  assert.ok(stopped, "a stop that arrived during a run did not stop it: it wrote on for 30 s of the worker's clock");
  const tokens = of("token").length;
  assert.deepEqual([of("done").length, of("error").length, closed, destroyed], [1, 0, 1, 1], "a stopped run did not end as a run does (done once, no error, the generator closed and let go of)");
  assert.equal(of("done")[0].tokens, written, "the page was not told the stopped run's statistics");
  await sleep(1000);
  assert.equal(of("token").length, tokens, "a stopped run wrote on");
  assert.equal(run("generating"), undefined);
  ok("a stop that arrives during a run stops it, and the page is told it is done");

  // the next run is not cut short: neither by the stop that ended the last one, nor by one that came while nothing ran
  for (const [name, before] of [["after a run that was stopped", () => {}], ["after a stop that came while nothing ran", () => context.onmessage({ data: { type: "stop" } })]]) {
    await before();
    fresh(() => new Response("", { status: 404 }));
    limit = written + 5;
    await context.onmessage({ data: { type: "generate", prompt: "another" } });
    assert.deepEqual([of("token").length, of("done").length, of("error").length], [5, 1, 0], `a run ${name} did not write its five pieces`);
  }
  ok("a run after a stop writes to its own end");
  run("adopted({})");
}

// ---- T.378.3 (T.156's path, which nothing ran outside a browser): the GPU stops under a run of a model that is on the GPU
// alone. The page is told so in the owner's words, the model is on the CPU from now on (weights.js's own list, which
// worker.js tells by onCpuFromNow), and the load the page last asked for is asked for again, by its own message. A
// run that fails with the GPU going on is an error like any other, and loads nothing.
{
  const small = { id: "small-on-the-gpu", name: "Small", checkpoint: "small", tokenizer: "small.tokenizer.bin", bytes: 64, options: {} };
  const of = (type) => messages.filter((message) => message.type === type);
  const noRuntime = () => run("initialized = Promise.reject(new Error('no runtime in this check')); initialized.catch(() => {})");
  // the load the page asked for (it fails here, where no runtime comes: the worker knows what was asked for all the same)
  noRuntime();
  fresh(() => new Response(new Uint8Array(64), { status: 200 }));
  await context.onmessage({ data: { type: "load", load: 21, model: small } });
  const told = [];
  context.stand = {
    pyodide: { runPython: (code) => told.push(code) },
    llama: { generate: { callKwargs: () => { throw new Error("the device was lost under the run"); } }, destroy: () => told.push("destroy") },
    direct: { lost: null },
  };
  for (const lost of [null, "the GPU's device was lost (a test)"]) {
    context.stand.direct.lost = lost;
    run("pyodide = stand.pyodide; adopted({ llama: stand.llama }); gpuOnlyNow = stand.direct;");
    noRuntime();
    fresh(() => new Response(new Uint8Array(64), { status: 200 }));
    told.length = 0;
    await context.onmessage({ data: { type: "generate", prompt: "a prompt" } });
    await sleep(2000);  // (the load that follows is the worker's own message to itself)
    assert.equal(run("generating"), undefined, "a run that failed is still the run going on");
    if (!lost) {
      assert.deepEqual(of("error").map(({ message, reloading }) => [message, reloading]), [["the device was lost under the run", undefined]]);
      assert.deepEqual([requests.length, run("cpuOnly.has('small-on-the-gpu')"), run("llama") === context.stand.llama, told], [0, false, true, []],
        "a run that failed with its GPU going on loaded something, or took the model away");
      continue;
    }
    assert.deepEqual(JSON.parse(JSON.stringify(of("error")[0])), { type: "error", load: 21, reloading: true,
      message: "The GPU stopped (the GPU's device was lost (a test)). This model was on the GPU alone, and is loaded again on the CPU." });
    assert.equal(run("cpuOnly.has('small-on-the-gpu')"), true, "the model whose GPU stopped under a run is not on the CPU from now on");
    assert.deepEqual(told, ["destroy", "import gc; gc.collect()"], "the model was not let go of for the load that follows");
    assert.deepEqual([run("llama"), run("gpuOnlyLost()")], [undefined, undefined], "the load that follows began with the lost model");
    assert.deepEqual(requests.map((r) => r.url.split("/").pop()).sort(), ["small.000", "small.tokenizer.bin"], "the load the page asked for last was not asked for again");
    // (that load ends here as the first one did: no runtime)
    assert.deepEqual(of("error").slice(1).map(({ load, message }) => [load, message]), [[21, "no runtime in this check"]]);
  }
  run("initialized = undefined; adopted({}); gpuOnlyNow = undefined; pyodide = undefined;");
  ok("the GPU stopped under a run of a model on the GPU alone: said, the model on the CPU from now on, and the page's last load asked for again");
}

console.log(`worker-memory-check: ${passed} checks passed`);
