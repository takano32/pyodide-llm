// worker/clock.ts (T.350): the seconds since a moment and those of a load's parts, and a turn of the event loop.
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T.367.3).

import type { LoadSeconds } from "../contracts/messages.ts";

export const since = (started: number) => (performance.now() - started) / 1000;

// How long the load took, in seconds: Pyodide once per session, the others per model (what is not measured again
// stays as it was). The download of a model
// of this site runs while Pyodide loads and usually ends first, so its seconds are counted until the last byte
// arrives (not until the bytes reach Python, which has to wait for Pyodide). The page says that the two overlap
// instead of adding them up, but only for the model that was loaded while Pyodide was still coming.
const seconds = {} as LoadSeconds;
/** A part of the load (pyodide, download, convert, construct) took so many seconds */
export function took(part: keyof LoadSeconds, spent: number) {
  seconds[part] = spent;
}
/** The seconds of the load's parts as they stand, for the ready a load reports */
export const secondsSoFar = () => ({ ...seconds });

// Hand the event loop a turn, so that a message sent meanwhile is delivered. setTimeout would cost 4ms per
// call (the browsers clamp it), a MessageChannel comes back in the same millisecond.
export function breathe() {
  return new Promise<void>((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(0);
  });
}
