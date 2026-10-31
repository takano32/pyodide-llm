// src/page/receiver.ts (T.355): the worker, once the page is isolated or will not be, and what the page does with each
// of its reports. This module waits at its top level: whatever imports it runs after that wait, and the parts that
// must not start before the worker (choose.ts, composer.ts and the script in index.astro) all import it.
import { memoryFailure } from "../models.ts";
import { FULL_ROUNDS, ROUNDS, USAGE_KEY, benchMarkdown, environmentOf, roundsHere, usedAfter } from "../bench.ts";
import { $, chat, files, progress, run, select, statusText } from "./dom.ts";
import { benchmark, weighed } from "./address.ts";
import { gpuKey, keptAs, remember, threadsKey } from "./remembered.ts";
import { arrived, arriving, becameReady, fromTemplate, benchBegan, benchEnded, failed, loads, model, ready, siteUpdated, takesPrompts, threadsNow } from "./model.ts";
import { STILL_MS, advanced, leftSaid, secondsLeft } from "./remaining.ts";
import { answer, drew, idle, shape, undrawn, used, written, wrote } from "./answer.ts";
import { markdownCame, markdownWanted } from "./markdown.ts";
import { showKept } from "./list.ts";
import { showSettings } from "./settings.ts";
import { drawAnswer, message, meta, thoughtSaid, onGpu, onGpuSampled, rows, seconds, showBench, showRun, showStatus } from "./draw.ts";
import { isolating } from "./isolation.ts";
import { endsWork, ofThisLoad } from "./heard.ts";
import { progressLine } from "./lines.ts";
import { unstarted } from "../runtime/updated.ts";
import type { Alone, ModelWorker, Ready } from "../runtime/contracts/messages.ts";

declare const __BUILD__: string;

const loadedHow = (data: Ready) => ((model as any).hf ? (data.fromCache ? "from the cache of this browser" : "fetch and conversion") : (model as any).file ? "read" : "download");
let benchAsked = false;  // the benchmark is asked for once a page, whatever becomes of it
// T.156: the worker's verdict that the CPU is faster here than a model on the GPU alone, until its load on the CPU is ready
let aloneOnReady: { load: number; alone: Alone } | undefined;
// T.553: a line that says the time left, which says it no more once nothing has arrived for 10 seconds (no report comes then)
let stillTimer: ReturnType<typeof setTimeout> | undefined;
// T.554: the answer is drawn at once, then every 8 tokens or 32 ms (answer.ts's wrote()); what comes between is drawn by
// this timer, or at the answer's end
let drawTimer: ReturnType<typeof setTimeout> | undefined;
function drawNow() {
  clearTimeout(drawTimer);
  drawTimer = undefined;
  if (!answer) return;
  drawAnswer(answer.firstElementChild!, written, shape, markdownCame);
  drew(performance.now());
}
if (await isolating()) {
  await new Promise(() => {});  // the page reloads: nothing else may start
}
export const worker: ModelWorker = new Worker(new URL("../runtime/worker.ts", import.meta.url), { type: "module" });
/** the site's root, which the worker is told with a load: models/ is under it, and the worker's own URL is not (T.367.3) */
export const root = new URL(import.meta.env.BASE_URL, location.href).href;
worker.onmessage = ({ data }) => {
  // a late report about a load that another choice of model has cancelled (heard.ts)
  if (!ofThisLoad(data, loads)) {
    return;
  }
  if (data.type === "status") {
    statusText.textContent = data.text;
  } else if (data.type === "progress") {
    progress.hidden = false;
    progress.max = data.total;
    progress.value = data.received || (data.converted ?? 0) * data.total;
    // T.553: and, last, about how long the fetch has left (the owner's words: nothing for its first 3 seconds, nor while
    // nothing arrived for 10)
    const now = performance.now();
    if (data.received) arrived(advanced(arriving, data.received, now));
    // (lines.ts: the words, T.119's)
    const line = progressLine({ name: model.name, ...data });
    const left = data.received ? leftSaid(secondsLeft(arriving, data.total, now)) : "";
    statusText.textContent = line + left;
    clearTimeout(stillTimer);
    if (left) stillTimer = setTimeout(() => statusText.textContent === line + left && (statusText.textContent = line), STILL_MS);
  } else if (data.type === "bench") {
    // T.45 measured, T.76 shows it: one bubble with the table, and under it the Markdown to copy or to report
    const environment = environmentOf(navigator, { model: model.name, pyodide: data.pyodide, build: __BUILD__, site: location.origin + location.pathname });
    const markdown = benchMarkdown(data.rows, environment);
    (window as any).__bench = { rows: data.rows, environment, markdown };
    benchEnded();
    console.log(markdown);
    showBench(data.rows, environment, markdown);
    statusText.textContent = `${model.name} · benchmark done`;
    run.disabled = false;
  } else if (data.type === "ready") {
    becameReady(data);
    if (aloneOnReady?.load === data.load) {
      try {
        const kept = JSON.parse(localStorage.getItem(gpuKey(model)) ?? "null") ?? {};
        localStorage.setItem(gpuKey(model), JSON.stringify({ ...kept, alone: aloneOnReady!.alone }));
      } catch {
        // no storage here: the next visit weighs the two again
      }
    }
    aloneOnReady = undefined;
    // for tests/e2e.ts: how long each part of the load took, and the WebAssembly memory then (T.84). Not shown
    (window as any).__ready = data;
    // once: every round of the benchmark loads the model again and reports ready too (T.96: the second request
    // ended the first round's threads under its coordinator, which then waited for ever)
    if (benchmark && !benchAsked) {
      statusText.textContent = `${model.name} · benchmarking...`;
      benchAsked = true;
      benchBegan();
      worker.postMessage({ type: "bench", model: weighed(model), load: loads,
        // T.214: where the browser does not say its memory, the round without the kernels is a row that says why
        rounds: roundsHere(benchmark === "full" ? FULL_ROUNDS : ROUNDS, (navigator as any).deviceMemory), prompt: model.prompt || "Once upon a time", steps: 64 });
    }
    showSettings();
    // T.556: a model that answers in a format has its answers drawn as Markdown: its libraries are taken now (once a visit),
    // and an answer that was drawn as plain text meanwhile is drawn again once they come
    if ((model as { template?: string }).template ?? fromTemplate) void markdownWanted().then((came) => came && answer && shape.markdown && drawNow());
    // the model is in the Cache API now: ask the browser not to evict it (only a page may ask, not a worker)
    navigator.storage?.persist?.();
    // T.135, T.148: whether the prompts go through the GPU, or why not (and the same line in the console); it changes
    // as the GPU gets ready and as the prompts are timed (the "gpu" and "done" messages)
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
    remember(model, (model as any).hf && !data.notKept);
  } else if (data.type === "gpu") {
    // T.148: the GPU is ready (or will not be): the status line says so, the load's breakdown says what it chose and
    // how long it took, and what it chose is kept for the next visit on this device
    showStatus(data.note);
    if (data.matrices) {
      $("load-stats").append(Object.assign(document.createElement("div"), {
        textContent: `WebGPU ${seconds(data.seconds)} · ${data.matrices}${data.remembered ? " (remembered)" : ""} · ${data.attention}` +
          (data.tokens ? ` · a token by ${data.tokens}` : ""),
      }));
      try {
        // (T.152: and the layer of a token)
        // (T.156: with what it kept of a model on the GPU alone: its own key says when it no longer holds)
        const alone = JSON.parse(localStorage.getItem(gpuKey(model)) ?? "null")?.alone;
        localStorage.setItem(gpuKey(model), JSON.stringify({ key: data.key, matrices: data.matrices, attention: data.attention, tokens: data.tokens, alone }));
      } catch {
        // no storage here: the next visit times them again
      }
    }
  } else if (data.type === "updated") {
    // T.439: the worker went on without a file of this page's build, which the site no longer serves: the status line of
    // a ready model says so from now on (the model answers meanwhile, and nothing else changes)
    siteUpdated();
    showStatus();
  } else if (data.type === "gpu-alone") {
    // T.156: the CPU is faster here than the GPU for this model on the GPU alone: kept once the load on the CPU is
    // ready (the second review of T.156: a CPU that cannot hold the model must not keep the next visits from the GPU),
    // and the next load goes on the CPU at once (forward.ts's aloneHolds says when it no longer holds)
    aloneOnReady = { load: data.load, alone: data.alone };
  } else if (data.type === "threads-compared") {
    // T.114: how the worker chose the number of threads, one line per comparison (medians in ms per token)
    const kept = data.faster ? data.candidate : data.best;
    console.info(`threads: ${data.best} at ${data.bestMs.toFixed(2)} ms against ${data.candidate} at ` +
      `${data.candidateMs.toFixed(2)} ms per token (${data.tokens} tokens timed): ${kept} ` +
      `${data.faster ? "is faster by more than 5%" : "stays, the other is not faster by more than 5%"}` +
      // T.223: a verdict timed while the GPU got ready is searched again once it is, and not remembered before
      `${data.whileGpu ? " (timed while the GPU got ready: searched again once it is)" : ""}`);
  } else if (data.type === "threads") {
    console.info(`threads: ${data.count} for ${data.model}, searched from ${data.from === "remembered" ?
      "the count remembered from an earlier visit" : `${data.hint} (the logical cores)`}`);
    // what the worker found for this device and model: the next visit starts with it
    try {
      // (T.508.3: the model loaded now by the name its next visit asks with: a model of ?hf= is "local" to the worker)
      localStorage.setItem(threadsKey(data.model === model.id ? keptAs(model) : data.model), String(data.count));
    } catch {
      // no storage here: the next visit searches again
    }
  } else if (data.type === "token") {
    if (wrote(data.text, performance.now())) drawNow();
    else drawTimer ??= setTimeout(drawNow, 32);
  } else if (data.type === "done") {
    if (undrawn) drawNow();  // (T.554: the last tokens, before the line under the answer)
    thoughtSaid(answer!.firstElementChild!, data);  // (T.520: how long its thought was, where it had one)
    // (T.223's review: the search checks the count remembered from an earlier visit on the first generation and may change
    // it, 4 to 2 on the owner's Android: the line says the count in use as the answer's line below does)
    threadsNow(data.threads);
    showStatus(data.gpu);  // T.148: the prompts timed on either side may have moved the GPU's verdict
    // T.156: how the page is used, for a model on the GPU alone
    try {
      const kept = JSON.parse(localStorage.getItem(USAGE_KEY) ?? "null");
      localStorage.setItem(USAGE_KEY, JSON.stringify(usedAfter(kept, data.prompt_tokens, data.tokens - data.prompt_tokens)));
    } catch {
      // no storage here: the tokens are weighed as many
    }
    // the settings this very answer was written with: with the seed the same text can be asked for again
    const how = used.temperature ? `temp ${used.temperature}` : "greedy";
    meta(answer!, `${model.name} · ${data.tokens} tokens · ${data.tokens_per_second.toFixed(1)} tok/s${data.threads > 1 ? ` · ${data.threads} threads` : ""} · ${how}`, [
      `first token ${seconds(data.first_token_seconds)}`,
      `prompt ${data.prompt_tokens} tokens${onGpu(data)} · ${data.prompt_tokens_per_second.toFixed(1)} tok/s`,
      `generated ${data.tokens - data.prompt_tokens} tokens${onGpuSampled(data)} · ${data.tokens_per_second.toFixed(1)} tok/s · ${seconds(data.seconds)} in all`,
    ], used.temperature ? used.seed : undefined);
  } else if (data.type === "error") {
    if (undrawn) drawNow();  // (T.554: what the answer wrote before it failed)
    aloneOnReady = undefined;  // (T.156: the load on the CPU did not come to ready)
    // the console keeps where it happened, for a failed run's record (T.82's artifacts, T.96)
    // (T.156's review: the worker's word that it loads a model on the GPU alone again on the CPU is no error of its own,
    // and has no memory or stack to say: tests/e2e.ts counted it as a console error)
    if (data.reloading) console.warn(data.message);
    else console.error(`worker error (weights memory ${data.weights} bytes):`, data.stack);
    // (a round that failed ends the benchmark; T.156: a model the worker loads again on the CPU is not ready)
    failed(Boolean(data.reloading));
    // Pyodide that never finishes loading under the service worker (iOS Safari, 2026-09-25: its NumPy step): once,
    // the page goes on without the worker, which is the one-core version the site runs on anyway (policy 6)
    let fallback: boolean | undefined = false;
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
    message("model error", data.memory ? memoryFailure(model, data.heap, data.message) : data.message);
    statusText.textContent = data.reloading ? `${model.name}: loading again on the CPU...` : ready ? `${model.name}` : `Could not load ${model.name}`;
  }
  // T.172: only the messages that end a piece of the worker's work mean it is idle again (heard.ts's endsWork()). The
  // others come in the middle of one: status, progress and token, and the search for the number of threads (threads,
  // threads-compared), which runs inside a text, and gpu (T.148); read as idle, they gave the run button back and a
  // press started a second text on the same engine
  if (endsWork(data)) {
    progress.hidden = true;
    idle();
    showRun();
    run.disabled = !takesPrompts();
    select.disabled = files.disabled = false;
  }
};
// (T.367.4: a worker that did not load because the site was updated since this page came says that, and that alone)
worker.onerror = async (event) => {
  message("model error", (await unstarted(event, import.meta.url, `Failed to start the worker: ${event.message}`)).message);
};
