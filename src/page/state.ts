// src/page/state.ts (T355): the one object the parts of the page's script share.
import { MODELS } from "../models.js";
import { fromUrl, requested } from "./address.ts";
import { recalled } from "./remembered.ts";

// What the page is doing now: what more than one part of this script reads or sets. A field is no variable, and a
// wrong name reads undefined without a word: tests/page-modules-check.mjs holds every page.<name> to this list.
export const page = {
  model: fromUrl ?? MODELS.find(({ id }) => id === requested) ?? recalled ?? MODELS[0],
  ready: false,
  answer: null as HTMLElement | null, // the bubble that is being written
  used: {} as Record<string, any>, // the generation settings of that run, for the line under it
  // the format of one turn the conversion read from the model's own chat_template, when there was one
  fromTemplate: undefined as string | undefined,
  benchmarking: false,  // the rounds are running: the run button stays off (the worker is busy loading)
  generating: false,
  longest: 256, // the context of the model, known once it is ready
  // the status line of a ready model: how it runs, and (T148) what the GPU takes of its prompts now
  readyLine: undefined as { backend: string; threads: number; pyodide: string } | undefined,
  gpuNote: undefined as string | undefined,
  // The model can be changed while one is loading: the worker cancels that download and starts the new one.
  // loads numbers them, and the worker's reports carry the number.
  loads: 1,
};

// ---- T378.5: what happens to it. The parts say these and set no field themselves.
type Model = typeof page.model;
/** The model the page opens with (the URL's, the one remembered). Once a model was chosen that one stays: false. */
export function opensWith(entry: Model) {
  if (page.loads !== 1) return false;
  page.model = entry;
  return true;
}
/** A model was chosen: nothing of the one before it holds any more. Its load's number. */
export function chose(entry: Model) {
  page.model = entry;
  page.ready = false;
  page.benchmarking = false;  // the worker cancels the benchmark's round for the new model, and never reports it
  page.readyLine = page.gpuNote = undefined;  // T148: the status line waits for the new model's
  page.loads += 1;
  return page.loads;
}
/** The worker's ready: the model takes prompts, as long as its context, in its own format, and runs like this. */
export function becameReady({ seq_len, template, backend, threads, pyodide }: { seq_len: number; template?: unknown; backend: string; threads: number; pyodide: string }) {
  page.ready = true;
  page.longest = seq_len;
  page.fromTemplate = typeof template === "string" ? template : undefined;
  page.readyLine = { backend, threads, pyodide };
}
/** The worker's error: a round that failed ends the benchmark (the run button must not stay off: the review of T76),
 * and a model the worker loads again on the CPU (T156: its GPU stopped while it was on the GPU alone) is not ready. */
export function failed(reloading: boolean) {
  page.benchmarking = false;
  if (reloading) page.ready = false;
}
export function benchBegan() {
  page.benchmarking = true;
}
export function benchEnded() {
  page.benchmarking = false;
}
/** What the GPU takes of the prompts now, when the worker says (nothing said keeps what was). Whether it is news. */
export function gpuSaid(note?: string) {
  const news = Boolean(note) && note !== page.gpuNote;
  page.gpuNote = note ?? page.gpuNote;
  return news;
}
/** The threads in use, as an answer's end says (T223's review: the search may change the count remembered). */
export function threadsNow(count: number) {
  if (page.readyLine) page.readyLine.threads = count;
}
/** Whether a prompt may be sent: one sent during the benchmark would find no model ready. */
export const takesPrompts = () => page.ready && !page.benchmarking;
/** An answer begins: its bubble, and the settings it is written with. A run that samples draws its seed here rather
 * than in the engine, so that the answer can say which seed wrote it; greedy models ignore it, and showing one would
 * suggest it mattered. */
export function answerBegan(bubble: HTMLElement, settings: Record<string, any>, seed: () => number) {
  page.generating = true;
  page.answer = bubble;
  page.used = { ...settings, steps: settings.steps > 0 ? Math.min(settings.steps, page.longest) : 0 };
  if (page.used.temperature) {
    page.used.seed ??= seed();
  } else {
    delete page.used.seed;
  }
  return page.used;
}
/** The worker is idle again: nothing is being written. */
export function idle() {
  page.generating = false;
}
