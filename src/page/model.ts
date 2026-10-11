// src/page/model.ts (T378.5): the model the page has, and how far its load has come. The owner of that: the other
// parts of the page's script read these by the names they import (a static import of src/ is a live binding), and say
// what happened to the functions below; none of them sets one. Nothing of a page is read here (no element, no address,
// no storage): Node imports this file as it is (tests/page-parts-check.mjs).
import { MODELS } from "../models.js";

export type Model = typeof MODELS[number];
// A first visit starts with the first of the list; opening.ts says what this visit opens with instead.
export let model: Model = MODELS[0];
// The model can be changed while one is loading: the worker cancels that download and starts the new one.
// loads numbers them, and the worker's reports carry the number.
export let loads = 1;
export let ready = false;
export let longest = 256; // the context of the model, known once it is ready
// the format of one turn the conversion read from the model's own chat_template, when there was one
export let fromTemplate: string | undefined = undefined;
// the status line of a ready model: how it runs, and (T148) what the GPU takes of its prompts now
export let readyLine: { backend: string; threads: number; pyodide: string } | undefined = undefined;
export let gpuNote: string | undefined = undefined;
export let benchmarking = false;  // the rounds are running: the run button stays off (the worker is busy loading)

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
  readyLine = gpuNote = undefined;  // T148: the status line waits for the new model's
  loads += 1;
  return loads;
}
/** The worker's ready: the model takes prompts, as long as its context, in its own format, and runs like this. */
export function becameReady({ seq_len, template, backend, threads, pyodide }: { seq_len: number; template?: unknown; backend: string; threads: number; pyodide: string }) {
  ready = true;
  longest = seq_len;
  fromTemplate = typeof template === "string" ? template : undefined;
  readyLine = { backend, threads, pyodide };
}
/** The worker's error: a round that failed ends the benchmark (the run button must not stay off: the review of T76),
 * and a model the worker loads again on the CPU (T156: its GPU stopped while it was on the GPU alone) is not ready. */
export function failed(reloading: boolean) {
  benchmarking = false;
  if (reloading) ready = false;
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
/** The threads in use, as an answer's end says (T223's review: the search may change the count remembered). */
export function threadsNow(count: number) {
  if (readyLine) readyLine.threads = count;
}
/** Whether a prompt may be sent: one sent during the benchmark would find no model ready. */
export const takesPrompts = () => ready && !benchmarking;
