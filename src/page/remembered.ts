// src/page/remembered.ts (T355): what the page finds in localStorage from an earlier visit, and what it leaves there
// under the same names: the model that was chosen, the number of threads, what the GPU chose for a model on this
// device and how the page is used. T643.3: the page's reads and writes of its storage (localStorage, and receiver.ts's
// mark in sessionStorage) go through recall() and keep(), the only two that catch: the rest say what is kept, under
// which name, and the receiver calls them where the worker's reports come. (isolation.ts's marks in sessionStorage are
// under the try of the Service Worker's registration, whose catch stands for both: the page goes on unisolated.)
import { MODELS } from "../models.ts";
import { CPU_SPEED_KEY, USAGE_KEY, threadsKey as sharedThreadsKey, usedAfter } from "../bench.ts";
import { fromUrl, parameters, requested } from "./address.ts";
import { SAFETENSORS } from "./repository.ts";
import type { Model } from "./model.ts";
import type { Alone, GpuChose } from "../runtime/contracts/messages.ts";

/** where the page keeps a thing: localStorage, unless sessionStorage is asked for (reading either may throw) */
type Place = () => Storage;
const LOCAL: Place = () => localStorage;
export const SESSION: Place = () => sessionStorage;
/** What is kept under the name, as read() reads its text (null: nothing kept); `otherwise` where there is no storage here
 * (a private window, a policy) or read() refuses what stands there (something else wrote under the name) */
export function recall<T>(name: string, read: (text: string | null) => T, otherwise: T, place = LOCAL): T {
  try {
    return read(place().getItem(name));
  } catch {
    return otherwise;
  }
}
/** Keeps the text under the name (null takes it away); whether it was kept (no storage here, or no room: it was not) */
export function keep(name: string, text: string | null, place = LOCAL): boolean {
  try {
    text === null ? place().removeItem(name) : place().setItem(name, text);
    return true;
  } catch {
    return false;
  }
}
/** a kept JSON's value: undefined where nothing (or null) is kept, and a text that is no JSON throws (recall()'s otherwise) */
const parsed = (text: string | null) => JSON.parse(text ?? "null") ?? undefined;

// T93: the number of threads. ?threads=N fixes it (for measuring); else the worker finds it for this device and
// model and this page remembers the answer, so that the next visit starts with it.
const fixedThreads = Math.max(0, Math.floor(Number(parameters.get("threads")) || 0));
// (/benchmark/ reads the same key: T190)
export const threadsKey = (id: string) => sharedThreadsKey(id, navigator);
// a model as the page keeps what it found for it: by its id; one of ?hf= or of the visitor's folder is "local" in the
// list, and is named by its repository, revision and file (T508.3: the GGUF files of one repository apart, and the
// models of ?hf= apart for their threads, which all were "local"; a repository's model.safetensors as it was) or its name
export const keptAs = (entry: { id: string; name?: string; hf?: { repo?: string; revision?: string; weights?: string } }) => entry.id !== "local" ? entry.id
  : entry.hf?.repo ? `hf:${entry.hf.repo}@${entry.hf.revision}${entry.hf.weights && entry.hf.weights !== SAFETENSORS ? `/${entry.hf.weights}` : ""}` : `file:${entry.name}`;
// (none kept, or no storage here: the worker searches again)
export const threadsFor = (entry: { id: string }) =>
  ({ fixed: fixedThreads, remembered: recall(threadsKey(keptAs(entry)), (text) => Number(text) || 0, 0), hint: navigator.hardwareConcurrency || 1 });
/** what the worker found for this device and a model: the next visit starts with it. (T508.3: the model loaded now by
 * the name its next visit asks with: a model of ?hf= is "local" to the worker) */
export const rememberThreads = (loaded: Parameters<typeof keptAs>[0], named: string, count: number) =>
  keep(threadsKey(named === loaded.id ? keptAs(loaded) : named), String(count));
// T148: what the GPU chose for a model on this device (its shaders: T147 times them all the first time), for the next
// visit: the worker's GPU compiles only those, where the adapter and browser are the same (gpu.ts's key)
// (a model of ?hf= or of the visitor's folder: keptAs(), the review of T148)
export const gpuKey = (entry: Parameters<typeof keptAs>[0]) => `gpu:${keptAs(entry)}`;
// (none kept, no storage here, or something else under the name: the GPU's forms are chosen again)
export const gpuFor = (entry: { id: string }) => ({
  remembered: recall(gpuKey(entry), parsed, undefined),
  // (T156: and the CPU's reading of the weights /benchmark/ measured here, which a model on the GPU alone is held against)
  cpu: recall(CPU_SPEED_KEY, parsed, undefined),
  usage: recall(USAGE_KEY, parsed, undefined),
});
/** T148: what the GPU chose for the model (T152: and the layer of a token), for the next visit on this device (no
 * storage here: it times them again). T156: with what was kept of it on the GPU alone, whose own key says when it no
 * longer holds */
export function rememberGpu(entry: Parameters<typeof keptAs>[0], chose: Pick<GpuChose, "key" | "matrices" | "attention" | "tokens">) {
  const alone = recall(gpuKey(entry), parsed, undefined)?.alone;
  keep(gpuKey(entry), JSON.stringify({ key: chose.key, matrices: chose.matrices, attention: chose.attention, tokens: chose.tokens, alone }));
}
/** T156: the worker's verdict that the CPU is faster here than the model on the GPU alone, beside what the GPU chose
 * (no storage here: the next visit weighs the two again) */
export function rememberAlone(entry: Parameters<typeof keptAs>[0], alone: Alone) {
  keep(gpuKey(entry), JSON.stringify({ ...(recall(gpuKey(entry), parsed, undefined) ?? {}), alone }));
}
/** T156: how the page is used, after an answer of so many tokens of the prompt and so many written, for a model on the
 * GPU alone (no storage here: the tokens are weighed as many) */
export const rememberUsage = (prompt: number, written: number) =>
  keep(USAGE_KEY, JSON.stringify(usedAfter(recall(USAGE_KEY, parsed, undefined), prompt, written)));
// Whoever has chosen a model before starts with it again; a first visit starts with the default (T511.6). What
// the URL names comes first. A model that would be fetched from Hugging Face anew (more than 500 MB, and not kept
// in this browser when it was last used) is not started unasked.
// (none kept, no storage (some private modes), or something else wrote there: the default)
function remembered(): typeof MODELS[number] | undefined {
  const { id, kept } = recall("model", parsed, undefined) ?? {};
  const entry = MODELS.find((candidate) => candidate.id === id);
  return entry && ((entry.download ?? 0) <= 500e6 || kept) ? entry : undefined;
}
// (no storage here: the next visit opens the default)
export function remember(entry: Model | undefined, kept = false) {
  keep("model", entry && entry.id !== "local" ? JSON.stringify({ id: entry.id, kept }) : null);
}
export const recalled = fromUrl || MODELS.some(({ id }) => id === requested) ? undefined : remembered();
