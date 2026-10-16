// src/page/remaining.ts (T553): how long the fetch of a model has left, from the bytes the worker says have arrived and
// the times they were said at (model.ts holds what the reports so far say). The form is naidan's
// (src/features/llama-cpp-browser/hugging-face/download-estimate.ts at 852bdeab, MIT, Ryo Ota: no line of it is
// copied): the speed is an average of what each report adds, weighted by the time it covers so that it forgets the
// speed of about 10 seconds ago (a weight of 1 − e^(−seconds / 10)), and nothing is said for the first 3 seconds, nor
// while nothing arrived for 10 seconds, nor while the speed is not known. Pure: the clock is the caller's, and Node
// imports this file as it is (tests/page-parts-check.mjs).
import type { Arriving } from "./model.ts";

const MEMORY = 10, FIRST = 3, STILL = 10;  // seconds

/** A report of `bytes` arrived at `now` (ms). A first report, or one of fewer bytes than the last (a fetch begun again),
 * begins anew. For its first 10 seconds the speed is the plain average since the first report (a weighted one would
 * hold on to the first report's alone), and from then on the weighted one. */
export function advanced(was: Arriving | undefined, bytes: number, now: number): Arriving {
  if (!was || bytes < was.bytes) return { began: now, first: bytes, at: now, bytes, moved: now };
  const seconds = (now - was.at) / 1000;
  if (seconds <= 0) return was;  // (the bytes count in the next report's share)
  const moved = bytes > was.bytes ? now : was.moved;
  const speed = now - was.began <= MEMORY * 1000 || was.speed === undefined ? ((bytes - was.first) / (now - was.began)) * 1000
    : was.speed + (1 - Math.exp(-seconds / MEMORY)) * ((bytes - was.bytes) / seconds - was.speed);
  return { ...was, at: now, bytes, moved, speed };
}

/** The seconds left to fetch `total` bytes at `now`, or undefined where nothing is to be said yet (or any more). */
export function secondsLeft(arriving: Arriving | undefined, total: number, now: number) {
  if (!arriving || !arriving.speed || arriving.speed <= 0 || arriving.bytes >= total) return undefined;
  if (now - arriving.began < FIRST * 1000 || now - arriving.moved >= STILL * 1000) return undefined;
  return Math.max(1, (total - arriving.bytes) / arriving.speed);
}

/** How the status line says it (the owner's words, T553): " · about 40 s left", or nothing. Whole seconds under 10,
 * fives under a minute, tens above: the estimate moves with the line, and a number that jumps by one each report reads
 * as more than it knows. */
export function leftSaid(seconds: number | undefined) {
  if (seconds === undefined) return "";
  const step = seconds < 10 ? 1 : seconds < 60 ? 5 : 10;
  return ` · about ${Math.max(1, Math.round(seconds / step) * step)} s left`;
}

/** How long after the last report a line that says the time left no longer holds (nothing arrived meanwhile). */
export const STILL_MS = STILL * 1000;
