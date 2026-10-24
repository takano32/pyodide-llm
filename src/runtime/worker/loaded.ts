// worker/loaded.ts (T.378.3): the model that is loaded now: Python's Llama and, where forward.ts runs it, what
// forward.ts made for it. The two are set together and let go of together, here and nowhere else.
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T.367.3).

import { pyodide } from "./runtime.ts";
import { rowsLetGo } from "./weights.ts";
import type { Outside } from "../forward.ts";
import type { Llama } from "../contracts/python.ts";

// Python's Llama of the model loaded now (a PyProxy), or undefined
export let llama: Llama | undefined;
// T.93: forward.ts's external() of it (the forward pass in JavaScript), or undefined where NumPy runs the model
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
    // what forward.ts holds of Python's, and its software threads (T.93); T.205: and the GPU's worker, whose buffers and
    // device the next model waits for (up to forward.ts's GPU_END_MS): an iPhone's tab went down in /benchmark/'s rounds
    // where the one before still held them as the next came from the cache
    const released = llama.release?.();
    llama.destroy();
    llama = undefined;
    outside = undefined;  // the engine goes; the memory stays for the next model (T.96)
    // the engine's closures and the model refer to each other, so only the cycle collector frees the weights
    pyodide.runPython("import gc; gc.collect()");
    await released;
    await rowsLetGo();
    signal.throwIfAborted();
    return;
  }
  // T.329.2.2: and the file its table of every layer's inputs was read from (the only handle the file can have), whether
  // or not a model was built on it (a load cancelled between the handle and the engine)
  await rowsLetGo();
}
