// T173: the page-memory section of /benchmark/: how much memory the page can hold before the browser refuses it or
// ends the tab. A plain module, so that Node tests it alone (tests/page-memory.ts); the page's worker is
// src/runtime/benchmark/memory.ts, and the section's Markdown is src/bench/memory.ts's (T301).
//
// iOS ends a tab past about 2 to 3 GB (jetsam) without an error, so the limit is found by a mark: before each step
// the page writes in sessionStorage what it held and what it is about to try; a tab the browser ended and loaded again
// finds the mark (readMark) and reports the last step it held. A normal end removes the mark. The section is never
// part of "Run all" (it may end the tab): only its own button or ?run=memory runs it, and a page that finds a mark
// does not run it again (a ?run=memory in the address would otherwise end the tab over and over).
//
// A page hidden (another app, the screen locked) or left (a reload, another page) stops the run at once (T173's
// review: the page's visibilitychange and pagehide settle `hidden`), and the run's finally removes the mark in the
// microtasks right after the event, before the page can be put away: a phone ends a hidden page far sooner than one in
// front, so a tab ended after that says nothing of the limit, and a page left is no tab ended. A mark found is then a
// tab that ended with no such event: the browser ended it (for its memory, most likely: a crash of another cause
// looks the same).
import type { Answer, MemoryGrown, Staged, ToMemory } from "../runtime/contracts/benchmark.ts";

// (T480.1: the section's data, what the worker answers a step with, and the tab's sessionStorage where there is one)
export type MemoryData = { stop: "limit" | "refused" | "hidden" | "error" | "none" | "reloaded"; held: number; trying?: number; limit: number;
  maximum?: number | null; why?: string; seconds?: number };
type Storing = Pick<Storage, "getItem" | "setItem" | "removeItem"> | null | undefined;

export const MEMORY_MARK = "benchmark-memory";
// the steps: 64 MiB of random bytes, a quarter of a second apart (time for the device to end the tab after a step, so
// that a step the page went on from was held for a while); up to 4 GiB, the most one 32-bit WebAssembly memory holds
export const MEMORY_STEP = 64 * 2 ** 20, MEMORY_MOST = 4 * 2 ** 30, MEMORY_PAUSE_MS = 250;

export const mib = (bytes: number) => `${Math.round(bytes / 2 ** 20)} MiB`;
const HIDDEN: unique symbol = Symbol("hidden");

/** The limit the page is asked for (?memoryMB=, for tests: a runner stops short of its own limit), in whole steps,
 * at least one step and at most MEMORY_MOST. */
export function memoryLimit(asked: string | null | undefined) {
  const bytes = Number(asked) > 0 ? Number(asked) * 2 ** 20 : MEMORY_MOST;
  return Math.min(MEMORY_MOST, Math.max(MEMORY_STEP, Math.floor(bytes / MEMORY_STEP) * MEMORY_STEP));
}

/** Remove the mark: a run that ended, or one the page stopped for its silence (it never reaches its own finally). */
export function forgetMark(storage: Storing) {
  try {
    storage?.removeItem(MEMORY_MARK);
  } catch {
    // nothing to remove it from
  }
}

/** Grow the worker's memory step by step up to limit, writing the mark before each step. ask(message) answers as the
 * worker does; storage is the tab's sessionStorage (null: none here, and nothing is grown: a tab ended without a
 * mark would report nothing). stage({stage, at, of}) is the page's progress (T177). hidden settles when the page is
 * hidden or left (the page has then ended the worker): the run stops and removes its mark. What it returns is the
 * section's data: { stop: "limit" | "refused" | "hidden" | "error" | "none", held, trying?, limit, maximum?, why?,
 * seconds }; maximum is the bytes the memory was made with room for, null where the browser refused that. */
export async function holdMemory({ ask, storage, limit = MEMORY_MOST, stage = () => {}, pause = MEMORY_PAUSE_MS,
                                   sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)), now = Date.now,
                                   hidden = new Promise(() => {}) }: { ask: (message: ToMemory) => Promise<Answer<MemoryGrown>>; storage: Storing; limit?: number;
                                   stage?: (stage: Staged) => void; pause?: number; sleep?: (ms: number) => Promise<unknown>; now?: () => number;
                                   hidden?: Promise<unknown> }): Promise<MemoryData> {
  if (!storage) return { stop: "none", held: 0, limit, why: "no sessionStorage in this tab: a tab the browser ended could not say how far it got" };
  const began = now(), of = limit / MEMORY_STEP;
  const or = <T>(promise: Promise<T>) => Promise.race([promise, hidden.then((): typeof HIDDEN => HIDDEN)]);
  let held = 0, maximum: number | null | undefined;
  const data = (fields: Partial<MemoryData> & Pick<MemoryData, "stop">): MemoryData => ({ held, limit, ...(maximum !== undefined && { maximum }), ...fields, seconds: (now() - began) / 1000 });
  try {
    while (held < limit) {
      const to = held + MEMORY_STEP;
      storage.setItem(MEMORY_MARK, JSON.stringify({ held, trying: to, limit, began }));
      stage({ stage: `holding ${mib(to)}`, at: to / MEMORY_STEP, of });
      const answer = await or(ask({ step: "grow", to, limit }));
      if (answer === HIDDEN) return data({ stop: "hidden", trying: to });
      if (answer.error) return data({ stop: "error", trying: to, why: answer.error });
      if (answer.result!.maximum !== undefined) maximum = answer.result!.maximum;
      if (answer.result!.refused) return data({ stop: "refused", held: answer.result!.held, trying: to, why: answer.result!.refused });
      held = answer.result!.held;
      if ((await or(sleep(pause))) === HIDDEN) return data({ stop: "hidden", trying: held + MEMORY_STEP });
    }
    return data({ stop: "limit" });
  } catch (error) {
    return data({ stop: "error", why: `${(error as Error | null)?.name ?? "Error"}: ${(error as Error | null)?.message ?? error}` });
  } finally {
    forgetMark(storage);
  }
}

/** A mark left by a run that never ended (the tab ended while it grew, with no hidden page or page left to stop it):
 * the section's data for it, and the mark removed. undefined when there is none. */
export function readMark(storage: Storing, now = Date.now): MemoryData | undefined {
  let text: string | null | undefined;
  try {
    text = storage?.getItem(MEMORY_MARK);
    if (text) storage!.removeItem(MEMORY_MARK);
  } catch {
    return undefined;
  }
  if (!text) return undefined;
  try {
    const m = JSON.parse(text);
    if (!(m.held >= 0 && m.trying > m.held)) return undefined;
    return { stop: "reloaded", held: m.held, trying: m.trying, limit: m.limit, seconds: Number.isFinite(m.began) ? (now() - m.began) / 1000 : undefined };
  } catch {
    return undefined;
  }
}
