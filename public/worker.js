// Pyodide lives in this worker, so the page stays responsive while the model is loading and generating.
// model is an entry of src/models.js, or one with {file, tokenizerFile}: two files of the visitor's own disk,
// which are read where they are and go nowhere. Or one with {hf: {weights, config, tokenizer}}: a Hugging Face
// model by the names of its files, of huggingface.co ({repo, revision}) or of that disk ({files}: the Files of a
// folder, T374.2.2), which public/llama2_convert.py converts in here as it arrives.
// The page sends   {type: "init", search, model, load},  {type: "load", search, model, load},
//                  {type: "generate", prompt, ...options}  and  {type: "stop"}; /benchmark/'s model section also
//                  ahead in its init (T242: the switches of each load that follows on the same model, see loadsAhead),
//                  {type: "bench", model, load, rounds, prompt, steps} and (T184) {type: "paths", load, prompt, counts, sampled}
// and receives     {type: "status" | "progress" | "ready" | "token" | "done" | "bench" | "paths" | "error"
//                         | "threads" | "threads-compared" | "gpu", ...}
// Of these, only ready (a load, and each round of the benchmark), bench (the benchmark), paths (the page's path timed,
// T184), done (a text, also one stopped) and error (any of them) end a piece of this worker's work: the page reads them, and nothing else, as the
// worker being idle (T172). status and progress come during a load, token during a text, threads and
// threads-compared from the search for the number of threads, which runs inside a text, and gpu once the GPU is
// ready or refused, whatever else is going on (T148). A new message is one of the ends or not, and the page's list
// of the ends says so.
// load is a number the page counts up: a newer load cancels the one that is going on, and whatever this worker
// reports about a load carries its number, so that the page can tell a late report of a cancelled one.

// T350: this file is the window of the worker: its URL, init(), generate() and the messages. The rest is in the modules
// of worker/: state.js (the variables they share, in one object), told.js, clock.js, pyodide.js (the runtime's
// loading), weights.js (where a model's weights go), ranges.js and sources.js (the fetching), conduct.js (T374.2.1: what
// answers the conduct of a conversion, which is Python's, from huggingface.co or from a folder), convert.js (a Hugging Face model), load.js and timing.js (what /benchmark/ times). Each is asked for with this worker's ?v=<build>, so
// that all come from one deployment (as forward.js reads jobs.js), and all at once: one after another's end would add
// a round trip for each to the first load. The page's first message may come while they are fetched, and a module
// worker's port opens at the module's first await, where a message that finds no onmessage is lost (helper.js, T109):
// so the messages are kept until the handler at the end of this file is set, which then takes them in the order
// they came.
const early = [];
self.onmessage = (event) => early.push(event);
const modules = Object.fromEntries(["state", "told", "clock", "pyodide", "weights", "ranges", "sources", "conduct", "convert", "load", "timing"].map((name) =>
  [name, import(new URL(`worker/${name}.js${self.location.search}`, import.meta.url))]));
// (the list of the engine's Python files, with the worker's modules and not after them: init() reads the files by it)
const python = import(new URL(`python.js${self.location.search}`, import.meta.url));
python.catch(() => {});  // (init() awaits it: until then a failure is nobody's)
const { state, cpuOnly, modelKey, loadSeconds } = await modules.state;
const { told } = await modules.told;
const { resolvePyodideVersion, pyodideSteps } = await modules.pyodide;
const { since, breathe } = await modules.clock;
const { load, threadsNow, heapBytes } = await modules.load;
const { timedGeneration, timedPaths } = await modules.timing;

// Only what the engine has a fallback for. A name it does not know is refused there, and the page says so.
function switchesOf(search) {
  const parameters = new URLSearchParams(search);
  const names = (parameters.get("without") ?? "").split(",").map((name) => name.trim()).filter(Boolean);
  if (parameters.get("kernel") === "off" && !names.includes("kernels")) {
    names.push("kernels");
  }
  return names;
}

async function init(search) {
  const started = performance.now();
  const asked = new URLSearchParams(search);
  const parts = Number(asked.get("hfParts")), connections = Number(asked.get("hfConnections"));
  if (parts >= 1 && parts <= 64) state.hfPartBytes = Math.round(parts * 1024 * 1024);
  if (connections >= 1 && connections <= 32) state.hfConnections = Math.floor(connections);
  state.forceWide = asked.get("wide") === "on";
  // (T156: ?gpuTest=only, the tests' too: a model the GPU can take on the GPU alone, whatever its size)
  state.gpuForce = ["on", "only"].includes(asked.get("gpuTest")) ? { fallback: true, always: true, quick: true, only: asked.get("gpuTest") === "only" } : {};
  state.benchPage = asked.has("bench");
  // T348: the engine is a window and its parts (python.js's list), each with the ?v=<build> of this worker, so that all
  // come from the same deployment. They are small and of this site, so they are asked for now, beside Pyodide, and
  // are there when it is ready (one file was fetched after it before; eleven, one after another's list, would add to
  // the time to ready).
  const engine = python.then(({ readPython }) => readPython("llama2_numpy", async (name) => {
    const res = await fetch(new URL(`${name}${self.location.search}`, import.meta.url));
    if (!res.ok) {
      throw new Error(`Could not fetch ${name}: ${res.status}`);
    }
    return res.text();
  }));
  engine.catch(() => {});  // (it is awaited below: a load that ends before that leaves no unhandled rejection)
  const version = await resolvePyodideVersion(search);
  state.pyodide = await pyodideSteps(version, (url) => import(url));

  await (await python).placePython(state.pyodide, "llama2_numpy", null, engine);
  state.llama2_numpy = state.pyodide.pyimport("llama2_numpy");

  // The WASM SIMD kernels (kernels/*.ts), which llama2_numpy.py loads with ctypes. They are optional: without
  // them, or with ?kernel=off, NumPy does the math, several times slower. They are read even with ?kernel=off: the
  // switches say what is used (disabled), and the benchmark's rounds with the kernels need them there (T119: its
  // "everything" round sampled on NumPy after the page's switch had turned them off, and said nothing).
  state.disabled = state.pageSwitches = switchesOf(search);
  for (const name of ["simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
    const kernel = await fetch(new URL(`${name}${self.location.search}`, import.meta.url)).catch(() => undefined);
    if (kernel?.ok) {
      state.pyodide.FS.writeFile(`/home/pyodide/${name}`, new Uint8Array(await kernel.arrayBuffer()));
      state.kernels = "/home/pyodide/simdkernel.so";
    }
  }
  // T93: the forward pass runs in forward.js, on the plain build of the same kernels (simdkernel.so stays for the
  // sampling, which works on Python's logits). Without them (no WebAssembly SIMD) the engine runs NumPy.
  try {
    state.forwardModule = await import(new URL(`forward.js${self.location.search}`, import.meta.url));
    // one build of the kernels: simdkernel_<kind>.wasm and simdkernel_relaxed_<kind>.wasm, or null
    const build = async (kind, wide = false) => {
      const [plain, relaxed] = await Promise.all([`simdkernel_${kind}.wasm`, `simdkernel_relaxed_${kind}.wasm`].map((name) =>
        fetch(new URL(`${name}${self.location.search}`, import.meta.url)).then((res) => (res.ok ? res.arrayBuffer() : null)).catch(() => null)));
      return plain ? state.forwardModule.compileKernels(plain, relaxed, wide) : null;
    };
    state.jsKernels = await build("plain");
    if (state.jsKernels && self.crossOriginIsolated) state.sharedKernels = await build("shared");
    // T101: on their own, so that a browser that says it makes 64-bit memories and then cannot compile their kernels
    // (Playwright's WebKit did, 2026-09-25) keeps the 32-bit ones: in the same try it lost every kernel and ran NumPy
    if (state.jsKernels && state.forwardModule.memory64()) {
      try {
        state.wideKernels = { plain: await build("plain64", true), shared: self.crossOriginIsolated ? await build("shared64", true) : null };
      } catch {
        state.wideKernels = undefined;
      }
    }
  } catch {
    state.jsKernels = null;
  }
  loadSeconds.pyodide = since(started);
  state.pyodideAt = performance.now();
}

async function generate({ type, prompt, ...options }) {
  state.outsideNow?.engine?.newGeneration?.();  // now and then the remembered number of threads is checked again
  // a Python generator: every step of the iteration runs one forward pass and hands over one piece of text
  const pieces = state.llama.generate.callKwargs(prompt, options);
  try {
    let breathed = performance.now();
    while (!state.stopped) {
      const { done, value } = pieces.next();
      if (done) {
        break;
      }
      postMessage({ type: "token", text: value });
      // by the clock and not by the token: often enough for the button to feel immediate with a model that
      // writes 50 tokens a second, rarely enough not to cost one that writes 900 a measurable tok/s
      if (performance.now() - breathed > 50) {
        await breathe();
        breathed = performance.now();
      }
    }
    // close the generator here, so that the engine has written its stats before the "done" below
    pieces.return();
  } finally {
    pieces.destroy();
  }
  postMessage({ type: "done", threads: threadsNow(), gpuTokens: state.outsideNow?.engine?.gpuTokens ?? 0, gpuSampled: state.outsideNow?.engine?.gpuSampled ?? 0,
                gpu: state.outsideNow?.engine?.gpuStatus,
                ...state.llama.stats.toJs({ dict_converter: Object.fromEntries }) });
}

self.onmessage = async ({ data }) => {
  let signal;
  try {
    if (data.type === "init" || data.type === "load") {
      state.lastLoad = data;  // (T156: loaded again on the CPU where its GPU fails while the model is on it alone)
      state.threadsRequest = data.threads;
      state.gpuRequest = data.gpu;
      state.loadsAhead = Array.isArray(data.ahead) ? data.ahead : undefined;  // (T242; a load of the model page says none)
      // The latest choice wins: the download that is going on stops, and its parts that are complete stay in
      // the cache. Pyodide is loaded once, whatever happens to the model that was asked for first.
      state.loading?.abort();
      state.loading = new AbortController();
      signal = state.loading.signal;
      // the model downloads while Pyodide loads
      state.initialized ??= init(data.search);
      // never take the model away from a run that is going on
      state.stopped = state.generating !== undefined;
      // The cancelled load frees its buffer a few turns of the event loop after the abort. Without waiting for
      // that the next buffer is allocated first, and the WebAssembly memory, which never shrinks, grows by a
      // whole model with every change of mind (587 MB after four of them).
      const previous = state.unloaded;
      const current = previous.then(() => state.generating?.catch(() => {})).then(() => load(data.model, signal, data.load));
      state.unloaded = current.catch(() => {});
      await current;
    } else if (data.type === "bench") {
      // T45: the same model, measured again for every combination of switches the page asked for. The model is
      // built once per round from the checkpoint that is already in the Cache API, so only the engine changes.
      // One at a time: a second request while the rounds run would end a round's threads under its coordinator
      if (state.benching) {
        return;
      }
      state.benching = true;
      const rows = [];
      try {
        for (const round of data.rounds) {
          // T214: a round the page skips (src/bench.js's roundsHere(): NumPy's float32 weights where the browser does
          // not say its memory) is a row that says why, with nothing loaded
          if (round.skip !== undefined) {
            rows.push({ name: round.name, without: round.without, skip: round.skip });
            continue;
          }
          // every round is a load of its own, and it cancels whatever went before, exactly like a change of model
          state.loading?.abort();
          state.loading = new AbortController();
          signal = state.loading.signal;
          state.disabled = round.without;
          const started = performance.now();
          const previous = state.unloaded;
          const current = previous.then(() => state.generating?.catch(() => {})).then(() => load(data.model, signal, data.load));
          state.unloaded = current.catch(() => {});
          await current;
          const ready = since(started);
          const { tokens, speed } = timedGeneration(data.prompt, data.steps);
          rows.push({ name: round.name, without: round.without, tokens, speed, backend: state.llama.backend, seconds: ready });
        }
      } finally {
        // also when a round failed or a change of model cancelled it (the review of T76): the next model must not
        // load with a round's switches off while the panel shows them on
        state.disabled = state.pageSwitches;  // not self.location.search: that is the worker's own URL (?v=hash)
        state.benching = false;
      }
      postMessage({ type: "bench", load: data.load, rows, pyodide: state.pyodide.version });
    } else if (data.type === "paths") {
      // T184: the model page's own path, on the model as the first load left it (the GPU too), before the rounds:
      // its prompts and its writing as the page chooses, on the CPU only and on the GPU only
      if (state.benching || state.pathing || !state.llama) {
        return;
      }
      state.pathing = true;
      try {
        postMessage({ type: "paths", load: data.load, ...(await timedPaths(data)) });
      } finally {
        state.pathing = false;
      }
    } else if (data.type === "generate") {
      if (!state.llama) {
        throw new Error("The model is not ready.");
      }
      // messages keep arriving while this runs, hence the promise the other branches look at
      state.stopped = false;
      state.generating = generate(data);
      try {
        await state.generating;
      } catch (err) {
        // T156: the GPU stopped under a model on it alone: said, and the model loaded again on the CPU
        const lost = state.gpuOnlyNow?.lost;
        if (!lost) throw err;
        cpuOnly.add(modelKey(state.lastLoad.model));
        postMessage({ type: "error", load: state.lastLoad.load, reloading: true,
          message: `The GPU stopped (${lost}). This model was on the GPU alone, and is loaded again on the CPU.` });
        state.generating = undefined;
        self.onmessage({ data: state.lastLoad });
      } finally {
        state.generating = undefined;
      }
    } else if (data.type === "stop") {
      // a stop that arrives before a run starts, or after it ended, must not cut the next one short
      state.stopped = state.generating !== undefined;
    }
  } catch (err) {
    // a cancelled load has nothing to report: the one that replaced it speaks for itself
    if (!signal?.aborted) {
      // A JavaScript error thrown inside a call from Python comes back as a PythonError of type JsException, its own
      // name and message on the last line: the memory of the weights, which the converter opens through the sink
      // (weightsBuffer: Safari's refusal of a model past 4 GB, a memory the browser will not make or grow), and
      // forward.js growing the keys and values. It is told as itself (the review of T133: it was a traceback)
      const [, name = err?.name, text = err?.message] = err?.type === "JsException"
        ? /^pyodide\.ffi\.JsException: (\w+): (.*)$/.exec(err.message.trim().split("\n").pop()) ?? [] : [];
      // a ValueError of the engine is a message for the reader (wrong file, prompt too long): no traceback
      const message = err?.type === "ValueError" ? err.message.trim().split("\n").pop().replace(/^ValueError: /, "")
        : name === "Error" ? text : told(err);
      // T90: the memory ran out, in Python (MemoryError: malloc could not grow the WebAssembly memory) or in
      // JavaScript (RangeError: an ArrayBuffer or WebAssembly.Memory.grow was refused). The page says so in words
      // a visitor understands, with how much memory the page had when it happened.
      // (V8 also raises RangeError for a stack overflow, which is not this; Firefox says InternalError: out of memory)
      const memory = err?.type === "MemoryError" || name === "InternalError" ||
        (name === "RangeError" && !/call stack/i.test(text ?? ""));
      // where it happened goes to the page's console (T96): tests/e2e.mjs keeps the console of a failed run
      postMessage({ type: "error", load: data.load, message: memory ? String(text ?? err).trim().split("\n").pop() : message,
                    stack: String(err?.stack ?? told(err)), weights: state.weightsNow?.buffer.byteLength ?? 0, pyodide: Boolean(err?.pyodide),
                    ...(memory && { memory: true, heap: heapBytes() }) });
    }
  }
};
// (T350) what came while the modules were fetched
early.splice(0).forEach((event) => self.onmessage(event));
