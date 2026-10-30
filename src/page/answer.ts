// src/page/answer.ts (T.378.5): the answer that is being written. The owner of that: the other parts read these by the
// names they import and say what happened to the two functions below; none of them sets one. It imports nothing but a type and
// reads nothing of a page: Node imports this file as it is (tests/page-parts-check.mjs).
import type { Sampled } from "../runtime/contracts/messages.ts";

export let generating = false;
export let answer: HTMLElement | null = null; // the bubble that is being written
export let used: Sampled = {}; // the generation settings of that run, for the line under it
// T.554: what the answer says so far, the tokens not drawn yet, and when it was last drawn (the page's clock, in ms)
export let written = "";
export let undrawn = 0;
export let drawnAt: number | undefined = undefined;
// T.556: whether the answer is drawn as Markdown (a model that answers in a format: not one that goes on from the prompt),
// and whether it begins inside the model's thinking (a format that ends in <think>), which stays plain text
export let shape = { markdown: false, thinking: false };

/** An answer begins: its bubble, and the settings it is written with (no more steps than the model's context holds).
 * A run that samples draws its seed here rather than in the engine, so that the answer can say which seed wrote it;
 * greedy models ignore it, and showing one would suggest it mattered. The settings, as the worker is to get them.
 * T.468.3: none while an answer is being written (the worker writes one at a time: composer.ts asks no second, and
 * this one is not begun where it would), and nothing of the one being written changes. */
export function answerBegan(bubble: HTMLElement, settings: Sampled & { steps: number }, longest: number, seed: () => number,
  drawn = { markdown: false, thinking: false }): Sampled | undefined {
  if (generating) return undefined;
  generating = true;
  answer = bubble;
  shape = { ...drawn };
  written = "";
  undrawn = 0;
  drawnAt = undefined;
  used = { ...settings, steps: settings.steps > 0 ? Math.min(settings.steps, longest) : 0 };
  if (used.temperature) {
    used.seed ??= seed();
  } else {
    delete used.seed;
  }
  return used;
}
/** The worker is idle again: nothing is being written. (The bubble and its settings stay, for a late line under it.) */
export function idle() {
  generating = false;
}
/** T.554: a token of the answer came at `now`: whether it is to be drawn now. The first at once, then every 8 tokens or
 * 32 ms (naidan's form: createOutputPacing() of src/features/llama-cpp-browser/worker/output-pacing.ts at 852bdeab,
 * MIT, Ryo Ota; no line is copied). What is not drawn now is drawn by the page's timer, or at the answer's end. */
export function wrote(text: string, now: number) {
  written += text;
  undrawn += 1;
  return drawnAt === undefined || undrawn >= 8 || now - drawnAt >= 32;
}
/** What was written so far is drawn, at `now`. */
export function drew(now: number) {
  undrawn = 0;
  drawnAt = now;
}
