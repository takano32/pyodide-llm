// src/page/receiver.ts (T355): the worker, once the page is isolated or will not be, and what the page does with each
// of its reports. This module waits at its top level: whatever imports it runs after that wait, and the parts that
// must not start before the worker (choose.ts, composer.ts and the script in index.astro) all import it.
import { memoryFailure } from "../models.js";
import { FULL_ROUNDS, ROUNDS, USAGE_KEY, benchMarkdown, environmentOf, roundsHere, usedAfter } from "../bench.js";
import { $, chat, files, progress, run, select, statusText } from "./dom.ts";
import { benchmark, weighed } from "./address.ts";
import { gpuKey, remember, threadsKey } from "./remembered.ts";
import { page } from "./state.ts";
import { showKept } from "./list.ts";
import { showSettings } from "./settings.ts";
import { message, meta, onGpu, onGpuSampled, rows, seconds, setGenerating, showBench, showStatus } from "./draw.ts";
import { isolating } from "./isolation.ts";

declare const __BUILD__: string;

const loadedHow = (data: any) => ((page.model as any).hf ? (data.fromCache ? "from the cache of this browser" : "fetch and conversion") : (page.model as any).file ? "read" : "download");
let benchAsked = false;  // the benchmark is asked for once a page, whatever becomes of it
// T156: the worker's verdict that the CPU is faster here than a model on the GPU alone, until its load on the CPU is ready
let aloneOnReady: { load: number; alone: unknown } | undefined;
if (await isolating()) {
  await new Promise(() => {});  // the page reloads: nothing else may start
}
export const worker = new Worker(`${import.meta.env.BASE_URL}worker.js?v=${__BUILD__}`, { type: "module" });
worker.onmessage = ({ data }) => {
  // a late report about a load that another choice of model has cancelled
  if (data.load !== undefined && data.load !== page.loads) {
    return;
  }
  if (data.type === "status") {
    statusText.textContent = data.text;
  } else if (data.type === "progress") {
    progress.hidden = false;
    progress.max = data.total;
    const megabytes = (bytes: number) => (bytes / 1e6).toFixed(1);
    // T119: a conversion says what has arrived and how fast apart from what is converted: the share converted
    // moves only as fast as the bytes come, and alone it made a slow line look like a slow conversion. A file of
    // the visitor's own disk has nothing to fetch: the share alone
    const converted = data.converted === undefined ? "" : `${Math.floor(data.converted * 100)} % converted`;
    progress.value = data.received || (data.converted ?? 0) * data.total;
    statusText.textContent = !data.received && converted ? `${page.model.name}: ${converted}`
      : `${page.model.name}: ${megabytes(data.received)} / ${megabytes(data.total)} MB` +
        (data.perSecond ? ` · ${megabytes(data.perSecond)} MB/s` : "") + (converted ? ` · ${converted}` : "");
  } else if (data.type === "bench") {
    // T45 measured, T76 shows it: one bubble with the table, and under it the Markdown to copy or to report
    const environment = environmentOf(navigator, { model: page.model.name, pyodide: data.pyodide, build: __BUILD__, site: location.origin + location.pathname });
    const markdown = benchMarkdown(data.rows, environment);
    (window as any).__bench = { rows: data.rows, environment, markdown };
    page.benchmarking = false;
    console.log(markdown);
    showBench(data.rows, environment, markdown);
    statusText.textContent = `${page.model.name} · benchmark done`;
    run.disabled = false;
  } else if (data.type === "ready") {
    page.ready = true;
    if (aloneOnReady?.load === data.load) {
      try {
        const kept = JSON.parse(localStorage.getItem(gpuKey(page.model)) ?? "null") ?? {};
        localStorage.setItem(gpuKey(page.model), JSON.stringify({ ...kept, alone: aloneOnReady.alone }));
      } catch {
        // no storage here: the next visit weighs the two again
      }
    }
    aloneOnReady = undefined;
    // for tests/e2e.mjs: how long each part of the load took, and the WebAssembly memory then (T84). Not shown
    (window as any).__ready = data;
    page.fromTemplate = typeof data.template === "string" ? data.template : undefined;
    // once: every round of the benchmark loads the model again and reports ready too (T96: the second request
    // ended the first round's threads under its coordinator, which then waited for ever)
    if (benchmark && !benchAsked) {
      statusText.textContent = `${page.model.name} · benchmarking...`;
      benchAsked = page.benchmarking = true;
      worker.postMessage({ type: "bench", model: weighed(page.model), load: page.loads,
        // T214: where the browser does not say its memory, the round without the kernels is a row that says why
        rounds: roundsHere(benchmark === "full" ? FULL_ROUNDS : ROUNDS, (navigator as any).deviceMemory), prompt: page.model.prompt || "Once upon a time", steps: 64 });
    }
    page.longest = data.seq_len;
    showSettings();
    // the model is in the Cache API now: ask the browser not to evict it (only a page may ask, not a worker)
    navigator.storage?.persist?.();
    // T135, T148: whether the prompts go through the GPU, or why not (and the same line in the console); it changes
    // as the GPU gets ready and as the prompts are timed (the "gpu" and "done" messages)
    page.readyLine = { backend: data.backend, threads: data.threads, pyodide: data.pyodide };
    showStatus(data.gpu);
    const load = $("load-stats");
    load.textContent = "";
    rows(load, [
      `Pyodide ${seconds(data.seconds.pyodide)} · ${loadedHow(data)} ${seconds(data.seconds.download)}${data.overlapped ? " (at the same time)" : ""}`,
      ...(data.notKept ? [`not kept for the next visit: ${data.notKept}`] : []),
      `Llama() ${seconds(data.seconds.construct)}`,
    ]);
    $("load").classList.add("measured");
    showKept();
    remember(page.model, (page.model as any).hf && !data.notKept);
  } else if (data.type === "gpu") {
    // T148: the GPU is ready (or will not be): the status line says so, the load's breakdown says what it chose and
    // how long it took, and what it chose is kept for the next visit on this device
    showStatus(data.note);
    if (data.matrices) {
      $("load-stats").append(Object.assign(document.createElement("div"), {
        textContent: `WebGPU ${seconds(data.seconds)} · ${data.matrices}${data.remembered ? " (remembered)" : ""} · ${data.attention}` +
          (data.tokens ? ` · a token by ${data.tokens}` : ""),
      }));
      try {
        // (T152: and the layer of a token)
        // (T156: with what it kept of a model on the GPU alone: its own key says when it no longer holds)
        const alone = JSON.parse(localStorage.getItem(gpuKey(page.model)) ?? "null")?.alone;
        localStorage.setItem(gpuKey(page.model), JSON.stringify({ key: data.key, matrices: data.matrices, attention: data.attention, tokens: data.tokens, alone }));
      } catch {
        // no storage here: the next visit times them again
      }
    }
  } else if (data.type === "gpu-alone") {
    // T156: the CPU is faster here than the GPU for this model on the GPU alone: kept once the load on the CPU is
    // ready (the second review of T156: a CPU that cannot hold the model must not keep the next visits from the GPU),
    // and the next load goes on the CPU at once (forward.js's aloneHolds says when it no longer holds)
    aloneOnReady = { load: data.load, alone: data.alone };
  } else if (data.type === "threads-compared") {
    // T114: how the worker chose the number of threads, one line per comparison (medians in ms per token)
    const kept = data.faster ? data.candidate : data.best;
    console.info(`threads: ${data.best} at ${data.bestMs.toFixed(2)} ms against ${data.candidate} at ` +
      `${data.candidateMs.toFixed(2)} ms per token (${data.tokens} tokens timed): ${kept} ` +
      `${data.faster ? "is faster by more than 5%" : "stays, the other is not faster by more than 5%"}` +
      // T223: a verdict timed while the GPU got ready is searched again once it is, and not remembered before
      `${data.whileGpu ? " (timed while the GPU got ready: searched again once it is)" : ""}`);
  } else if (data.type === "threads") {
    console.info(`threads: ${data.count} for ${data.model}, searched from ${data.from === "remembered" ?
      "the count remembered from an earlier visit" : `${data.hint} (the logical cores)`}`);
    // what the worker found for this device and model: the next visit starts with it
    try {
      localStorage.setItem(threadsKey(data.model), String(data.count));
    } catch {
      // no storage here: the next visit searches again
    }
  } else if (data.type === "token") {
    // stick to the bottom only while the reader is there
    const pinned = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 40;
    page.answer!.firstElementChild!.textContent += data.text;
    if (pinned) {
      chat.scrollTop = chat.scrollHeight;
    }
  } else if (data.type === "done") {
    // (T223's review: the search checks the count remembered from an earlier visit on the first generation and may change
    // it, 4 to 2 on the owner's Android: the line says the count in use as the answer's line below does)
    if (page.readyLine) page.readyLine.threads = data.threads;
    showStatus(data.gpu);  // T148: the prompts timed on either side may have moved the GPU's verdict
    // T156: how the page is used, for a model on the GPU alone
    try {
      const kept = JSON.parse(localStorage.getItem(USAGE_KEY) ?? "null");
      localStorage.setItem(USAGE_KEY, JSON.stringify(usedAfter(kept, data.prompt_tokens, data.tokens - data.prompt_tokens)));
    } catch {
      // no storage here: the tokens are weighed as many
    }
    // the settings this very answer was written with: with the seed the same text can be asked for again
    const how = page.used.temperature ? `temp ${page.used.temperature}` : "greedy";
    meta(page.answer!, `${page.model.name} · ${data.tokens} tokens · ${data.tokens_per_second.toFixed(1)} tok/s${data.threads > 1 ? ` · ${data.threads} threads` : ""} · ${how}`, [
      `first token ${seconds(data.first_token_seconds)}`,
      `prompt ${data.prompt_tokens} tokens${onGpu(data)} · ${data.prompt_tokens_per_second.toFixed(1)} tok/s`,
      `generated ${data.tokens - data.prompt_tokens} tokens${onGpuSampled(data)} · ${data.tokens_per_second.toFixed(1)} tok/s · ${seconds(data.seconds)} in all`,
    ], page.used.temperature ? page.used.seed : undefined);
  } else if (data.type === "error") {
    aloneOnReady = undefined;  // (T156: the load on the CPU did not come to ready)
    // the console keeps where it happened, for a failed run's record (T82's artifacts, T96)
    // (T156's review: the worker's word that it loads a model on the GPU alone again on the CPU is no error of its own,
    // and has no memory or stack to say: tests/e2e.mjs counted it as a console error)
    if (data.reloading) console.warn(data.message);
    else console.error(`worker error (weights memory ${data.weights} bytes):`, data.stack);
    page.benchmarking = false;  // a round that failed ends the benchmark: the run button must not stay off (the review of T76)
    // Pyodide that never finishes loading under the service worker (iOS Safari, 2026-09-25: its NumPy step): once,
    // the page goes on without the worker, which is the one-core version the site runs on anyway (policy 6)
    let fallback = false;
    try {
      fallback = data.pyodide && self.crossOriginIsolated && !sessionStorage.getItem("coi-fallback");
      if (fallback) {
        sessionStorage.setItem("coi-fallback", "1");
      }
    } catch {
      // no session storage here: no second try either
    }
    if (fallback) {
      statusText.textContent = "Pyodide did not load under the service worker: trying again without it...";
      navigator.serviceWorker?.getRegistrations().then(async (registrations) => {
        for (const registration of registrations) {
          await registration.unregister();
        }
        location.reload();
      });
      return;
    }
    message("model error", data.memory ? memoryFailure(page.model, data.heap, data.message) : data.message);
    // T156: the worker loads the model again on the CPU (its GPU stopped while it was on the GPU alone)
    if (data.reloading) page.ready = false;
    statusText.textContent = data.reloading ? `${page.model.name}: loading again on the CPU...` : page.ready ? `${page.model.name}` : `Could not load ${page.model.name}`;
  }
  // T172: only the messages that end a piece of the worker's work mean it is idle again (a load: ready, the
  // benchmark: bench, a text: done, any of them: error). The others come in the middle of one: status, progress
  // and token, and the search for the number of threads (threads, threads-compared), which runs inside a text, and gpu (T148); read
  // as idle, they gave the run button back and a press started a second text on the same engine
  if (["ready", "bench", "done", "error"].includes(data.type)) {
    progress.hidden = true;
    setGenerating(false);
    run.disabled = !page.ready || page.benchmarking;  // a prompt sent during the benchmark would find no model ready
    select.disabled = files.disabled = false;
  }
};
worker.onerror = (event) => {
  message("model error", `Failed to start the worker: ${event.message}`);
};
