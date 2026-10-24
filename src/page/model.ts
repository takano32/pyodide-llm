// src/page/model.ts (T.378.5): the model the page has, and how far its load has come. The owner of that: the other
// parts of the page's script read these by the names they import (a static import of src/ is a live binding), and say
// what happened to the functions below; none of them sets one. Nothing of a page is read here (no element, no address,
// no storage): Node imports this file as it is (tests/page-parts-check.mjs).
import { DEFAULT_MODEL, MODELS } from "../models.js";

export type Model = typeof MODELS[number];
/** T.553: what the reports of a fetch say so far: when the first came and its bytes, when the last came and its bytes,
 * when the bytes last grew (ms of the page's clock), and the speed in bytes a second (none before two reports). */
export type Arriving = { began: number; first: number; at: number; bytes: number; moved: number; speed?: number };
// A first visit starts with the default (T.511.6); opening.ts says what this visit opens with instead.
export let model: Model = DEFAULT_MODEL;
// The model can be changed while one is loading: the worker cancels that download and starts the new one.
// loads numbers them, and the worker's reports carry the number.
export let loads = 1;
export let ready = false;
export let longest = 256; // the context of the model, known once it is ready
// the format of one turn the conversion read from the model's own chat_template, when there was one
export let fromTemplate: string | undefined = undefined;
// the status line of a ready model: how it runs, and (T.148) what the GPU takes of its prompts now
export let readyLine: { backend: string; threads: number; pyodide: string } | undefined = undefined;
export let gpuNote: string | undefined = undefined;
export let benchmarking = false;  // the rounds are running: the run button stays off (the worker is busy loading)
// T.439: the worker runs on a slower path because a file of this page's build is gone from the site (it was deployed
// again while the page was open). Of the page and not of a model: it holds for the rest of the visit
export let outdated = false;
// T.553: what the worker's reports of this load say of its fetch so far, for the time it has left (remaining.ts)
export let arriving: Arriving | undefined = undefined;

/** The model the page opens with (the URL's, the one remembered). Once a model was chosen that one stays: false. */
export function opensWith(entry: Model) {
  if (loads !== 1) return false;
  model = entry;
  return true;
}
/** A model was chosen: nothing of the one before it holds any more. Its load's number. */
export function chose(entry: Model) {
  model = entry;
  ready = false;
  benchmarking = false;  // the worker cancels the benchmark's round for the new model, and never reports it
  readyLine = gpuNote = undefined;  // T.148: the status line waits for the new model's
  arriving = undefined;
  loads += 1;
  return loads;
}
/** The worker's ready: the model takes prompts, as long as its context, in its own format, and runs like this. */
export function becameReady({ seq_len, template, backend, threads, pyodide }: { seq_len: number; template?: unknown; backend: string; threads: number; pyodide: string }) {
  ready = true;
  longest = seq_len;
  fromTemplate = typeof template === "string" ? template : undefined;
  readyLine = { backend, threads, pyodide };
  arriving = undefined;
}
/** The worker's error: a round that failed ends the benchmark (the run button must not stay off: the review of T.76),
 * and a model the worker loads again on the CPU (T.156: its GPU stopped while it was on the GPU alone) is not ready. */
export function failed(reloading: boolean) {
  benchmarking = false;
  arriving = undefined;
  if (reloading) ready = false;
}
/** The worker's report of the bytes of this load that have arrived, as remaining.ts's advanced() makes it of the one before. */
export function arrived(next: Arriving) {
  arriving = next;
}
export function benchBegan() {
  benchmarking = true;
}
export function benchEnded() {
  benchmarking = false;
}
/** What the GPU takes of the prompts now, when the worker says (nothing said keeps what was). Whether it is news. */
export function gpuSaid(note?: string) {
  const news = Boolean(note) && note !== gpuNote;
  gpuNote = note ?? gpuNote;
  return news;
}
/** The threads in use, as an answer's end says (T.223's review: the search may change the count remembered). */
export function threadsNow(count: number) {
  if (readyLine) readyLine.threads = count;
}
/** The worker's word that it went on without a file of its build that is gone (T.439). No choice of model undoes it. */
export function siteUpdated() {
  outdated = true;
}
/** Whether a prompt may be sent: one sent during the benchmark would find no model ready. */
export const takesPrompts = () => ready && !benchmarking;

