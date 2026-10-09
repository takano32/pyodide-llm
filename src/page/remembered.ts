// src/page/remembered.ts (T355): what the page finds in localStorage from an earlier visit, and what it leaves there
// under the same names: the model that was chosen, the number of threads and what the GPU chose for a model on this
// device. (The receiver writes the GPU's and the usage's entries where the worker's reports come: receiver.ts.)
import { MODELS } from "../models.js";
import { CPU_SPEED_KEY, USAGE_KEY, threadsKey as sharedThreadsKey } from "../bench.js";
import { fromUrl, parameters, requested } from "./address.ts";

// T93: the number of threads. ?threads=N fixes it (for measuring); else the worker finds it for this device and
// model and this page remembers the answer, so that the next visit starts with it.
const fixedThreads = Math.max(0, Math.floor(Number(parameters.get("threads")) || 0));
// (/benchmark/ reads the same key: T190)
export const threadsKey = (id: string) => sharedThreadsKey(id, navigator);
export const threadsFor = (entry: { id: string }) => {
  let remembered = 0;
  try {
    remembered = Number(localStorage.getItem(threadsKey(entry.id))) || 0;
  } catch {
    // no storage here: search again
  }
  return { fixed: fixedThreads, remembered, hint: navigator.hardwareConcurrency || 1 };
};
// T148: what the GPU chose for a model on this device (its shaders: T147 times them all the first time), for the next
// visit: the worker's GPU compiles only those, where the adapter and browser are the same (gpu.js's key)
// (a model of ?hf= or of the visitor's folder is "local" in the list: named by its repository or its name, the
// review of T148)
export const gpuKey = (entry: any) => `gpu:${entry.id !== "local" ? entry.id : entry.hf?.repo ? `hf:${entry.hf.repo}@${entry.hf.revision}` : `file:${entry.name}`}`;
export const gpuFor = (entry: { id: string }) => {
  try {
    // (T156: and the CPU's reading of the weights /benchmark/ measured here, which a model on the GPU alone is held against)
    return { remembered: JSON.parse(localStorage.getItem(gpuKey(entry)) ?? "null") ?? undefined,
      cpu: JSON.parse(localStorage.getItem(CPU_SPEED_KEY) ?? "null") ?? undefined,
      usage: JSON.parse(localStorage.getItem(USAGE_KEY) ?? "null") ?? undefined };
  } catch {
    return {};  // no storage here, or something else under the name: choose again
  }
};
// Whoever has chosen a model before starts with it again; a first visit starts with the first of the list. What
// the URL names comes first. A model that would be fetched from Hugging Face anew (more than 500 MB, and not kept
// in this browser when it was last used) is not started unasked.
function remembered(): typeof MODELS[number] | undefined {
  try {
    const { id, kept } = JSON.parse(localStorage.getItem("model") ?? "{}");
    const entry = MODELS.find((candidate) => candidate.id === id) as any;
    return entry && ((entry.download ?? 0) <= 500e6 || kept) ? entry : undefined;
  } catch {
    return undefined;  // no storage (some private modes), or something else wrote there
  }
}
export function remember(entry: typeof MODELS[number] | undefined, kept = false) {
  try {
    entry && entry.id !== "local" ? localStorage.setItem("model", JSON.stringify({ id: entry.id, kept })) : localStorage.removeItem("model");
  } catch {}
}
export const recalled = fromUrl || MODELS.some(({ id }) => id === requested) ? undefined : remembered();
