// worker/pool.ts (T641.1, out of worker/weights.ts): the one memory of forward.ts kept from model to model (T96), and
// what a checkpoint and its forward pass need of it (footprint(), T115). It holds that memory and the one the model
// loaded now is in; the others say where a model went (pooled, weightsInPython) and ask (weightsBytes, memoryShared).
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T367.3).

import { llama2_numpy, forwardModule, jsKernels, sharedKernels, hasWebGpu } from "./runtime.ts";
import { threadsRequest, switchesNow } from "./request.ts";
import { told } from "../failure.ts";
import type { Dtype, Form } from "../contracts/form.ts";
import type { Options } from "../contracts/python.ts";
import type { Footprint, Header, Memory } from "../forward.ts";

// the one memory kept from model to model (pooledWeights()): asked: a shared one was asked for; shared: it is one;
// maximum: the pages it may grow to; limited: a new one would get no more
type Pool = { memory: Memory; base: number; wide: boolean; asked: boolean; shared: boolean; maximum: number; limited: boolean };
// What the memory's part alone holds (T378.3); the others say what happened (pooled, weightsInPython) and ask
// (weightsBytes, memoryShared).
// T96: the one memory of forward.ts that is kept from model to model (pooledWeights() says why), and the memory of
// the model loaded now
let weightsPool: Pool | undefined, weightsNow: WebAssembly.Memory | undefined;

// T96: one memory of forward.ts, kept from model to model. A browser reserves address space for every WebAssembly
// memory, shared or not and whatever its maximum, and Chromium refused the third one of a page: the benchmark's
// first round, or a visitor's second change of model, then found no memory at all. So a memory is kept and grown
// as long as the next model fits under its maximum, and made anew only for a larger one. The maximum comes from the
// model (weightsMemory: what it needs and a gigabyte): asking for 4 GB up front left a phone no room for Pyodide's
// own memory, and "Loading Pyodide" never ended (2026-09-25).
// "Fits" is the checkpoint and what the forward pass puts after it (after: forward.ts's footprint(), T115). Pythia
// 1B's checkpoint (1.1 GB) fitted the memory made for tiny-lm (1.2 GB) and its corrections did not ("Maximum memory
// size exceeded"; opened first, it ran; the review of T96). Where a new memory would get no more room than this one
// (the browser gave less than it asked for, or it is not shared and grows as far as any would), the checkpoint
// fitting is enough. shared is what was asked for, not what the browser gave: a device without shared memories
// made one for every model.
// T242: the switches (?without=) of each load the page says will follow on the model of its init, and no other model
// ({ ahead: [[...], ...] }: /benchmark/'s model section, whose worker loads one model, for the page's path and again
// for each round). The shared memory is then made for the largest of those loads and no more: the gigabyte that
// pooledWeights() keeps for the next model (T96) is for a model that never comes there, and it took the page down.
// Playwright's WebKit on Windows (bench.yml, 2026-10-01) ended the page's process in new WebAssembly.Memory() of that
// maximum (1.07 GiB for tiny-lm) in about one run in four, once the CPU section's worker, ended before, had had a
// shared memory of its own; with the maximum of the model alone (45 MB) it never did (0 of 39 runs), and neither
// section alone ends it. undefined on the model page: any model may follow.
// ahead: what the forward pass of the largest of loadsAhead puts after the checkpoint, or undefined
export function pooledWeights(size: number, after: number, shared: boolean, wide: boolean, ahead?: number) {
  const forward = forwardModule!;  // (T476: the forward module, which a load that places weights has: taken once)
  const pages = (bytes: number) => Math.ceil(bytes / 65536);
  const needs = (base: number) => pages(base + size + (weightsPool!.limited ? 0 : after)) + 1;
  // what a memory made now is for: the load going on, and the largest of those that follow it where the page said which do
  const largest = ahead === undefined ? after : Math.max(after, ahead);
  const fits = weightsPool && weightsPool.asked === shared && weightsPool.wide === wide && needs(weightsPool.base) <= weightsPool.maximum;
  if (!fits) {
    // nothing may hold the old memory while the new one is made (T96: Chromium refused a page's third)
    weightsPool = weightsNow = undefined;
    let memory: Memory | undefined, base!: number;
    if (shared) {
      try {
        // (a page more where nothing follows, as needs() counts one past what the model takes)
        ({ memory, base } = forward.weightsMemory(size, { shared: true, wide, after: largest, ...(ahead !== undefined && { spare: 65536 }) }));
      } catch (error) {
        // no shared memory here: one thread (T380.2: said, as the lowered one below is)
        console.warn(`memory: no shared memory for this model (${told(error)}): a memory that is not shared, and one thread`);
        memory = undefined;
      }
      // (T130's review) a shared memory the browser gave at a lowered maximum (weightsMemory's second and third try: the
      // checkpoint and a gigabyte, or a quarter of one) that the forward pass does not fit would run out of memory when
      // the cache grows, or at once where the corrections do not fit, after the whole checkpoint was read: a plain memory
      // grows as far as the browser allows. One thread, but the model reaches the end of its context. (T242's review: for the
      // loads that follow as well, where the page said which: a round of /benchmark/ that widens int8 needs the most)
      if (memory?.limited && memory.maximum! < pages(base + size + largest) + 1) {
        console.info(`memory: the browser gave a shared memory of ${Math.round(memory.maximum! * 65536 / 2 ** 20)} MiB, and this model needs ` +
          `${Math.round(pages(base + size + largest) * 65536 / 2 ** 20)} MiB: a memory that is not shared, and one thread`);
        memory = undefined;
      }
    }
    if (!memory) ({ memory, base } = forward.weightsMemory(size, { wide }));
    const isShared = shared && memory.buffer instanceof SharedArrayBuffer;
    // a memory without a maximum (not shared) grows as far as the browser allows: 4 GB of pages, 16 GB when wide
    const most = wide ? 262144 : 65536;
    weightsPool = { memory, base, wide, asked: shared, shared: isShared, maximum: isShared ? memory.maximum ?? most : most,
                    limited: !isShared || Boolean(memory.limited) };
  }
  const { memory, base } = weightsPool!;
  const more = pages(base + size) + 1 - memory.buffer.byteLength / 65536;
  if (more > 0) forward.growMemory(memory, more, wide);
  return weightsPool!;
}

// T115: what the forward pass of a checkpoint of size bytes puts after it, at most (forward.ts's footprint()): from
// its header (the 7 ints) and the options it is loaded with (their form: head_dim where a head is not dim / heads,
// T124: the keys and values of a Qwen3 0.6B are twice what the header says), on a shared memory or not (an int8
// model's keys and values may be float16; forward.ts's keysInHalf says whether they are, T160, T130).
// What footprint() takes (the worker asks keysInHalf the same).
// without: the switches of the load (those of the load going on, or of one that follows: loadsAhead, T242).
// T363.5, header: of a model whose form gives rows kinds of their own, Python counts the weights that are still the
// dtype's (dtype_weights(): footprint() has no rows to ask, and takes every weight of the file for one otherwise)
export function forwardOptions(header: Header, options: Options, shared: boolean, without = switchesNow()): Footprint {
  const { dtype = "float32" } = options;
  const int8 = !without.includes("int8");
  return {
    ...options, dtype, int8, relaxed: Boolean(jsKernels?.relaxed) && !without.includes("relaxed"),
    ...(options.kinds ? { weights: llama2_numpy.dtype_weights(header, dtype, options) } : {}),
    // (T375.5: of packed weights only, which footprint() knows by the dtype)
    halfKV: int8 && !without.includes("kv16"), shared,
    outliers: llama2_numpy.OUTLIER_CHANNELS, gpu: hasWebGpu,
  };
}
export const afterCheckpoint = (header: Header, size: number, options: Options, shared: boolean, without?: string[]) => forwardModule!.footprint(header, size, forwardOptions(header, options, shared, without));
// the page cross-origin isolated (stage 3), shared memories to be had, and not ?threads=1: the memory is shared
export const sharedWanted = () => Boolean(sharedKernels && self.crossOriginIsolated && threadsRequest?.fixed !== 1);
// llama2_numpy.external_tensors(): where every tensor of the checkpoint is, from its header alone
export function placesOf(header: Header, dtype: Dtype, form: Partial<Form>) {
  const proxy = llama2_numpy.external_tensors(header, dtype, form);
  try {
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    proxy.destroy();
  }
}
/** The bytes of forward.ts's memory of the model loaded now (0 where Python holds the weights, or nothing is loaded) */
export const weightsBytes = () => weightsNow?.buffer.byteLength ?? 0;
/** Whether the memory the weights are in is a shared one (software threads can be had on it) */
export const memoryShared = () => Boolean(weightsPool?.shared);
/** T641.1: the model loaded now is in this memory of forward.ts */
export function pooled(memory: WebAssembly.Memory) {
  weightsNow = memory;
}
/** T641.1: the model loaded now is in a Python buffer (NumPy computes): in no memory of forward.ts */
export function weightsInPython() {
  weightsNow = undefined;
}
