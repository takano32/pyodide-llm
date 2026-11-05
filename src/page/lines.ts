// src/page/lines.ts (T468.2, T475, T643.4): the page's words of a model, from what the page holds of the model and of
// its load (model.ts) and what a report says: the status line, the load's breakdown and the line under an answer, which
// draw.ts and receiver.ts write. Pure: Node imports this file as it is (tests/page-parts-check.ts), which reads every
// part of each line.
import { SLOWER } from "../runtime/updated.ts";
import type { Model } from "./model.ts";
import type { Done, GpuChose, Ready, Sampled } from "../runtime/contracts/messages.ts";

/** A time as the page says it: two decimals under 10 seconds, one above */
export const seconds = (value: number) => `${value.toFixed(value < 10 ? 2 : 1)} s`;

/** A ready model's line: its name, how it runs (and on how many threads, and what the GPU takes: gpuNote, with
 * ?gpuTest=on said where the tests' flag is on), Pyodide's version, and last (T439) that a reload would make it faster
 * where it runs slower for a file of this page's build that is gone */
export function readyStatus({ name, backend, threads, pyodide, gpuNote, gpuTest, outdated }:
  { name: string; backend: string; threads: number; pyodide: string; gpuNote?: string; gpuTest: boolean; outdated: boolean }) {
  // (?gpuTest=on, the tests' flag, says so as ?without= does)
  const tested = gpuNote && gpuTest ? " (gpuTest)" : "";
  return `${name} · ${backend}${threads > 1 ? `, ${threads} threads` : ""}${gpuNote ? `, ${gpuNote}${tested}` : ""} · Pyodide ${pyodide}${outdated ? ` · ${SLOWER}` : ""}`;
}

/** A load's line as it fetches and converts: what has arrived of how much, how fast, and the share converted. T119:
 * a conversion says what has arrived and how fast apart from what is converted (the share converted moves only as
 * fast as the bytes come, and alone it made a slow line look like a slow conversion). A file of the visitor's own
 * disk has nothing to fetch: the share alone. (The time left is remaining.ts's, said after it.) */
export function progressLine({ name, received, total, perSecond, converted }:
  { name: string; received: number; total: number; perSecond?: number; converted?: number }) {
  const megabytes = (bytes: number) => (bytes / 1e6).toFixed(1);
  const share = converted === undefined ? "" : `${Math.floor(converted * 100)} % converted`;
  return !received && share ? `${name}: ${share}`
    : `${name}: ${megabytes(received)} / ${megabytes(total)} MB` + (perSecond ? ` · ${megabytes(perSecond)} MB/s` : "") + (share ? ` · ${share}` : "");
}

/** How a model came to be loaded: converted from Hugging Face (or kept from that), read from the visitor's disk, or
 * fetched from the site */
const loadedHow = (data: Ready, model: Pick<Model, "hf" | "file">) => (model.hf ? (data.fromCache ? "from the cache of this browser" : "fetch and conversion") : model.file ? "read" : "download");

/** The load's breakdown under the status line, once the model is ready: Pyodide and the model's own fetch (and
 * whether the two ran at once), why the model is not kept for the next visit where it is not, and Llama() */
export function loadLines(data: Ready, model: Pick<Model, "hf" | "file">) {
  return [
    `Pyodide ${seconds(data.seconds.pyodide)} · ${loadedHow(data, model)} ${seconds(data.seconds.download)}${data.overlapped ? " (at the same time)" : ""}`,
    ...(data.notKept ? [`not kept for the next visit: ${data.notKept}`] : []),
    `Llama() ${seconds(data.seconds.construct)}`,
  ];
}

/** T148: the breakdown's line of the GPU once it is ready: how long it took, the forms it chose (remembered from an
 * earlier visit or timed now), and a token's form where the answers go on the GPU too */
export function gpuLine(data: GpuChose) {
  return `WebGPU ${seconds(data.seconds)} · ${data.matrices}${data.remembered ? " (remembered)" : ""} · ${data.attention}` +
    (data.tokens ? ` · a token by ${data.tokens}` : "");
}

// T135: how many of the prompt's tokens went through the GPU (the ones counted as the prompt: the one after them
// makes the first logits, on the CPU)
export const onGpu = (data: Done) => (!data.gpuTokens ? "" : data.gpuTokens >= data.prompt_tokens ? " on WebGPU"
  : ` (${data.gpuTokens} on WebGPU)`);
// T152: how many of the tokens it wrote the GPU sampled (the sampled steps: a stop token among them)
export const onGpuSampled = (data: Done) => (!data.gpuSampled ? "" : data.gpuSampled >= data.sampled ? " on WebGPU"
  : ` (${data.gpuSampled} on WebGPU)`);

/** The line under an answer: its summary (the model, the tokens, the speed, the threads where more than one, and the
 * settings this very answer was written with), the breakdown's 3 lines under it, and the seed where it sampled (with
 * the seed the same text can be asked for again). tests/e2e.ts reads "tok/s" in the summary and the lines that begin
 * "prompt <n>" and "generated <n>". */
export function answerMeta(data: Done, used: Sampled, name: string) {
  const how = used.temperature ? `temp ${used.temperature}` : "greedy";
  return {
    summary: `${name} · ${data.tokens} tokens · ${data.tokens_per_second.toFixed(1)} tok/s${data.threads > 1 ? ` · ${data.threads} threads` : ""} · ${how}`,
    breakdown: [
      `first token ${seconds(data.first_token_seconds)}`,
      `prompt ${data.prompt_tokens} tokens${onGpu(data)} · ${data.prompt_tokens_per_second.toFixed(1)} tok/s`,
      `generated ${data.tokens - data.prompt_tokens} tokens${onGpuSampled(data)} · ${data.tokens_per_second.toFixed(1)} tok/s · ${seconds(data.seconds)} in all`,
    ],
    seed: used.temperature ? used.seed : undefined,
  };
}
