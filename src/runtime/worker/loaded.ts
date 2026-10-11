// worker/loaded.ts (T378.3): the model that is loaded now: Python's Llama and, where forward.js runs it, what
// forward.js made for it. The two are set together and let go of together, here and nowhere else.
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T367.3).

import { pyodide } from "./runtime.ts";
import type { Outside } from "../contracts/forward-js.ts";
import type { Llama } from "../contracts/python.ts";

// Python's Llama of the model loaded now (a PyProxy), or undefined
export let llama: Llama | undefined;
// T93: forward.js's external() of it (the forward pass in JavaScript), or undefined where NumPy runs the model
let outside: Outside | undefined;

/** forward.js's engine of the model loaded now, or undefined (no model, or NumPy's) */
export const engineNow = () => outside?.engine;

/** A model was built: made is what the place of its weights made of it ({ llama, outside }, weights.js) */
export function adopted(made: { llama: Llama; outside?: Outside }) {
  ({ llama, outside } = made);
}

/** Lets go of the model, so that two never have to fit in memory; nothing where there is none */
export async function letGo(signal: AbortSignal) {
  if (llama) {
    // what forward.js holds of Python's, and its software threads (T93); T205: and the GPU's worker, whose buffers and
    // device the next model waits for (up to forward.js's GPU_END_MS): an iPhone's tab went down in /benchmark/'s rounds
    // where the one before still held them as the next came from the cache
    const released = llama.release?.();
    llama.destroy();
    llama = undefined;
    outside = undefined;  // the engine goes; the memory stays for the next model (T96)
    // the engine's closures and the model refer to each other, so only the cycle collector frees the weights
    pyodide.runPython("import gc; gc.collect()");
    await released;
    signal.throwIfAborted();
  }
}
