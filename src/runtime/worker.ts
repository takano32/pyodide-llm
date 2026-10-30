// Pyodide lives in this worker, so the page stays responsive while the model is loading and generating.
// model is an entry of src/models.ts, or one with {file, tokenizerFile}: two files of the visitor's own disk,
// which are read where they are and go nowhere. Or one with {hf: {weights, config, tokenizer}}: a Hugging Face
// model by the names of its files, of huggingface.co ({repo, revision}) or of that disk ({files}: the Files of a
// folder, T.374.2.2), which the converter (src/python/llama2_convert.py) converts in here as it arrives.
// The page sends   {type: "init", search, model, load},  {type: "load", search, model, load},
//                  {type: "generate", prompt, ...options}  and  {type: "stop"}; /benchmark/'s model section also
//                  ahead in its init (T.242: the switches of each load that follows on the same model, see loadsAhead),
//                  {type: "bench", model, load, rounds, prompt, steps} and (T.184) {type: "paths", load, prompt, counts, sampled}
// and receives     {type: "status" | "progress" | "ready" | "token" | "done" | "bench" | "paths" | "error"
//                         | "threads" | "threads-compared" | "gpu" | "gpu-alone" | "updated", ...}
// Of these, only ready (a load, and each round of the benchmark), bench (the benchmark), paths (the page's path timed,
// T.184), done (a text, also one stopped) and error (any of them) end a piece of this worker's work: the page reads them, and nothing else, as the
// worker being idle (T.172). status and progress come during a load, token during a text, threads and
// threads-compared from the search for the number of threads, which runs inside a text, and gpu once the GPU is
// ready or refused, whatever else is going on (T.148). A new message is one of the ends or not, and the page's list
// of the ends says so.
// load is a number the page counts up: a newer load cancels the one that is going on, and whatever this worker
// reports about a load carries its number, so that the page can tell a late report of a cancelled one.

// T.350: this file is the window of the worker: its URL, generate() and the messages. The rest is in the modules
// of worker/: runtime.ts (what the visit finds once), request.ts (what the page asked for with a load), loaded.ts (the
// model loaded now) (T.378.3: each sets its own, and the others say what happened and read), clock.ts, pyodide.ts (the runtime's
// loading), weights.ts (where a model's weights go), ranges.ts and sources.ts (the fetching), conduct.ts (T.374.2.1: what
// answers the conduct of a conversion, which is Python's, from huggingface.co or from a folder), convert.ts (a Hugging Face model), load.ts and timing.ts (what /benchmark/ times).
// T.367.3: a bundler links them (static imports, src/runtime/): they are part of this worker's own file, whose name
// carries a hash of its text, so that all are of one deployment, and nothing is awaited before the handler at the end
// of this file is set (the messages that came while the modules were fetched one by one were kept in a queue until
// then). What the worker asks for later is asked for by a literal name: forward.ts, kept.ts and the device's key as
// modules (import("./forward.ts")), the kernels and the Python's archives as files (built.ts). The site's root, where
// models/ is, the page hands over with a load (request.ts's root): this file's own URL is under _astro/.
// T.380.1: what fails is carried and read in one way (failure.ts): the catch at the end of this file tells the page what
// failureOf() reads of whatever was thrown, and no module puts a mark on an Error or reads an exception's words.
// T.367.4: a page left open across a deployment asks for such a file after it is gone. Where that ends a load (the
// Python's archives, kept.ts at a first conversion) the visitor is told that the site was updated (updated.ts); where
// the engine goes on without the file (the kernels and forward.ts: NumPy; a software thread; the GPU's worker; the
// device's key) it goes on as it always did: a model that answers is not interrupted. T.439: where that is slower
// (all but the key) the page is told { type: "updated" } once, and its status line says so (worker/outdated.ts).
import { pyodide, runtimeAsked } from "./worker/runtime.ts";
import { loadAsked, benching, roundsBegan, roundBegan, roundsEnded } from "./worker/request.ts";
import { llama, engineNow } from "./worker/loaded.ts";
import { weightsBytes, gpuOnlyLost, onCpuFromNow } from "./worker/weights.ts";
import { failureOf } from "./failure.ts";
import { since, breathe } from "./worker/clock.ts";
import { load, threadsNow, heapBytes } from "./worker/load.ts";
import { timedGeneration, timedPaths } from "./worker/timing.ts";
import type { Bench, BenchRow, Done, Failed, Paths, Token, ToWorker } from "./contracts/messages.ts";

// What this file alone holds (T.378.3: they were fields of the object all the modules shared):
// the AbortController of the load that is going on, and a promise that settles once it has cleaned up
let loading: AbortController | undefined, unloaded = Promise.resolve();
// the run that is going on, and whether the page asked it to stop
let generating: Promise<void> | undefined, stopped = false;
let lastLoad: Extract<ToWorker, { type: "init" | "load" }> | undefined;  // the load message of the model now (T.156)
let pathing = false;  // T.184: the benchmark's page path is being timed; see the paths message

async function generate({ type, prompt, ...options }: Extract<ToWorker, { type: "generate" }>) {
  engineNow()?.newGeneration?.();  // now and then the remembered number of threads is checked again
  // a Python generator: every step of the iteration runs one forward pass and hands over one piece of text
  const pieces = llama!.generate.callKwargs(prompt, options);
  try {
    let breathed = performance.now();
    while (!stopped) {
      const { done, value } = pieces.next();
      if (done) {
        break;
      }
      postMessage({ type: "token", text: value } satisfies Token);
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
  postMessage({ type: "done", threads: threadsNow(), gpuTokens: engineNow()?.gpuTokens ?? 0, gpuSampled: engineNow()?.gpuSampled ?? 0,
                gpu: engineNow()?.gpuStatus,
                ...llama!.stats.toJs({ dict_converter: Object.fromEntries }) } satisfies Done);
}

self.onmessage = async ({ data }: MessageEvent<ToWorker>) => {
  let signal: AbortSignal | undefined;
  try {
    if (data.type === "init" || data.type === "load") {
      lastLoad = data;  // (T.156: loaded again on the CPU where its GPU fails while the model is on it alone)
      loadAsked(data);  // (the threads, the GPU's shaders, the site's root, the loads that follow: request.ts)
      // The latest choice wins: the download that is going on stops, and its parts that are complete stay in
      // the cache. Pyodide is loaded once, whatever happens to the model that was asked for first.
      loading?.abort();
      loading = new AbortController();
      signal = loading.signal;
      // the model downloads while Pyodide loads
      runtimeAsked(data.search);
      // never take the model away from a run that is going on
      stopped = generating !== undefined;
      // The cancelled load frees its buffer a few turns of the event loop after the abort. Without waiting for
      // that the next buffer is allocated first, and the WebAssembly memory, which never shrinks, grows by a
      // whole model with every change of mind (587 MB after four of them).
      const previous = unloaded;
      const current = previous.then(() => generating?.catch(() => {})).then(() => load(data.model, signal!, data.load));
      unloaded = current.catch(() => {});
      await current;
    } else if (data.type === "bench") {
      // T.45: the same model, measured again for every combination of switches the page asked for. The model is
      // built once per round from the checkpoint that is already in the Cache API, so only the engine changes.
      // One at a time: a second request while the rounds run would end a round's threads under its coordinator
      if (benching) {
        return;
      }
      roundsBegan();
      const rows: BenchRow[] = [];
      try {
        for (const round of data.rounds) {
          // T.214: a round the page skips (src/bench.ts's roundsHere(): NumPy's float32 weights where the browser does
          // not say its memory) is a row that says why, with nothing loaded
          if (round.skip !== undefined) {
            rows.push({ name: round.name, without: round.without, skip: round.skip });
            continue;
          }
          // every round is a load of its own, and it cancels whatever went before, exactly like a change of model
          loading?.abort();
          loading = new AbortController();
          signal = loading.signal;
          roundBegan(round.without);
          const started = performance.now();
          const previous = unloaded;
          const current = previous.then(() => generating?.catch(() => {})).then(() => load(data.model, signal!, data.load));
          unloaded = current.catch(() => {});
          await current;
          const ready = since(started);
          const { tokens, speed } = timedGeneration(llama!, data.prompt, data.steps);
          rows.push({ name: round.name, without: round.without, tokens, speed, backend: llama!.backend, seconds: ready });
        }
      } finally {
        // also when a round failed or a change of model cancelled it (the review of T.76): the next model must not
        // load with a round's switches off while the panel shows them on (the page's again: request.ts)
        roundsEnded();
      }
      postMessage({ type: "bench", load: data.load, rows, pyodide: pyodide.version } satisfies Bench);
    } else if (data.type === "paths") {
      // T.184: the model page's own path, on the model as the first load left it (the GPU too), before the rounds:
      // its prompts and its writing as the page chooses, on the CPU only and on the GPU only
      if (benching || pathing || !llama) {
        return;
      }
      pathing = true;
      try {
        postMessage({ type: "paths", load: data.load, ...(await timedPaths(data)) } satisfies Paths);
      } finally {
        pathing = false;
      }
    } else if (data.type === "generate") {
      if (!llama) {
        throw new Error("The model is not ready.");
      }
      // messages keep arriving while this runs, hence the promise the other branches look at
      stopped = false;
      generating = generate(data);
      try {
        await generating;
      } catch (err) {
        // T.156: the GPU stopped under a model on it alone: said, and the model loaded again on the CPU
        const lost = gpuOnlyLost();
        if (!lost) throw err;
        onCpuFromNow(lastLoad!.model);
        postMessage({ type: "error", load: lastLoad!.load, reloading: true,
          message: `The GPU stopped (${lost}). This model was on the GPU alone, and is loaded again on the CPU.` } satisfies Failed);
        generating = undefined;
        self.onmessage!({ data: lastLoad } as MessageEvent<ToWorker>);
      } finally {
        generating = undefined;
      }
    } else if (data.type === "stop") {
      // a stop that arrives before a run starts, or after it ended, must not cut the next one short
      stopped = generating !== undefined;
    }
  } catch (err) {
    // a cancelled load has nothing to report: the one that replaced it speaks for itself
    if (!signal?.aborted) {
      // what it is and what is said of it, whatever was thrown (failure.ts reads them all: a sentence for the reader,
      // the memory that ran out, T.90, an error of this site's own that crossed Python, a runtime that did not load)
      const failure = failureOf(err);
      // where it happened goes to the page's console (T.96): tests/e2e.mjs keeps the console of a failed run
      postMessage({ type: "error", load: data.load, message: failure.message,
                    stack: String(failure.stack), weights: weightsBytes(), pyodide: failure.kind === "runtime",
                    ...(failure.kind === "memory" && { memory: true, heap: heapBytes() }) } satisfies Failed);
    }
  }
};
// (T.350) what came while the modules were fetched
