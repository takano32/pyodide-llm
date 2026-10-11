// src/page/answer.ts (T378.5): the answer that is being written. The owner of that: the other parts read these by the
// names they import and say what happened to the two functions below; none of them sets one. It imports nothing and
// reads nothing of a page: Node imports this file as it is (tests/page-parts-check.mjs).

export let generating = false;
export let answer: HTMLElement | null = null; // the bubble that is being written
export let used: Record<string, any> = {}; // the generation settings of that run, for the line under it

/** An answer begins: its bubble, and the settings it is written with (no more steps than the model's context holds).
 * A run that samples draws its seed here rather than in the engine, so that the answer can say which seed wrote it;
 * greedy models ignore it, and showing one would suggest it mattered. The settings, as the worker is to get them. */
export function answerBegan(bubble: HTMLElement, settings: Record<string, any>, longest: number, seed: () => number) {
  generating = true;
  answer = bubble;
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
