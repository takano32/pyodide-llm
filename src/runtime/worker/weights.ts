// worker/weights.ts (T.350): where the weights of a model go: the memory of forward.ts that is kept from model to model,
// a Python buffer where NumPy runs, or the GPU alone (T.156), and the sink a conversion writes them through.
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T.367.3).

import { pyodide, llama2_numpy, forwardModule, jsKernels, sharedKernels, wideKernels, wideUpdated, forceWide, gpuForce, benchPage, gpuAdapter, hasWebGpu,
  adapterAsked } from "./runtime.ts";
import { threadsRequest, gpuRequest, loadsAhead, benching, switchesNow } from "./request.ts";
import { workerWentWithout } from "./outdated.ts";
import { told } from "../failure.ts";
import type { keeper, Reader } from "../kept.ts";
import type { Dtype, Form } from "../contracts/form.ts";
import type { ToGpu } from "../contracts/gpu/protocol.ts";
import type { GpuAlone, Listed } from "../contracts/messages.ts";
import type { Bytes, Dict, List, Llama, Options, Sink } from "../contracts/python.ts";
import type { Apart, Direct, Engine, Footprint, GpuOnlyWeights, Header, Memory, Outside, Place, Thread } from "../forward.ts";
import type { Tensors } from "../contracts/plan.ts";

// T.368.3, the shapes this file makes and hands on.
// what writes a stretch of the checkpoint where it belongs
export type Write = (offset: number, chunk: Uint8Array<ArrayBuffer>) => void;
// a model built on its weights: Python's Llama and, where forward.ts runs it, what forward.ts made for it (loaded.ts)
export type Made = { llama: Llama; outside?: Outside };
// what an engine is made with here: the kernels' path for Python (the sampling) and the switches left out, over the options
type EngineOptions = Options & { kernels?: string; disable?: string[] };
// kept.ts's keeper(): the file a conversion is kept in as it comes (T.156)
export type Keeper = NonNullable<Awaited<ReturnType<typeof keeper>>>;
// T.156: a model on the GPU alone: what forward.ts reads of it (Direct), its GPU's worker, why the GPU stopped under
// it (or null), where the loops that write its weights wait (room) and its builder (drained), and end(): the worker
// stopped, resolved once it said so
type GpuOnly = Direct & { worker: Worker; lost: string | null; room: GpuOnlyWeights["room"]; drained: GpuOnlyWeights["drained"]; end(): Promise<void> };
// The place of a model's weights (weightsBuffer()): in a memory of forward.ts or in a Python buffer, where slice()
// copies a stretch out; or on the GPU alone (direct), where nothing holds them whole
// (T.329.2.2, hole: the stretch of the checkpoint the memory does not hold, a table of every layer's inputs read from
// where the checkpoint is kept: the first visit's file as it comes (the keeper), or, handed in by attach() before
// llama(), a kept one's; null where the memory holds all of it)
export type Weights =
  | { write: Write; slice(begin: number, end: number): Uint8Array<ArrayBuffer>; llama(tokenizer: Bytes, options: EngineOptions): Made; destroy(): void; direct?: undefined;
    hole: [number, number] | null; attach(reader: Reader): void }
  | { write: Write; llama(tokenizer: Bytes, options: EngineOptions): Made; destroy(): void; direct: GpuOnly; room: GpuOnlyWeights["room"]; drained: GpuOnlyWeights["drained"];
    hole: null; attach(reader: Reader): void };
// the one memory kept from model to model (pooledWeights()): asked: a shared one was asked for; shared: it is one;
// maximum: the pages it may grow to; limited: a new one would get no more
type Pool = { memory: Memory; base: number; wide: boolean; asked: boolean; shared: boolean; maximum: number; limited: boolean };
// where a model goes (gpuOnlyWeightsFor()): forward.ts's place and, where it may go on the GPU alone, its tensors,
// the bytes kept in the memory then and those with the forward pass after them
type Placed = Place & { tensors?: Tensors; stored?: number; gpuOnly?: number };

// What this file alone holds (T.378.3: they were fields of the object all the modules shared). The others say what
// happened (loadBegan, onCpuFromNow) and ask (weightsBytes, memoryShared, gpuOnlyLost, weightsRoom, weightsDrained).
// T.96: the one memory of forward.ts that is kept from model to model (pooledWeights() says why), and the memory of
// the model loaded now
let weightsPool: Pool | undefined, weightsNow: WebAssembly.Memory | undefined;
// T.156: the model on the GPU alone that is loading or loaded now (what gpuOnlyBuffer() made), or undefined
let gpuOnlyNow: GpuOnly | undefined;
// the GPU's worker of a model on the GPU alone let go before its engine was built (gpuOnlyBuffer()'s destroy()), or
// (T.461) of an engine whose Llama threw as it was built: the next load waits for it, as release() waits for a built
// one's (T.205)
let gpuOnlyEnding = Promise.resolve();
// T.461: a Llama that threw as it was built (Python's checks of the checkpoint and its options, after start()) lets go
// of the engine its external() built by then (the software threads, the GPU's worker): no model holds it, and nothing
// else would. The error goes on as it was
function builtOrLetGo(outside: Outside, build: () => Llama) {
  try {
    return build();
  } catch (thrown) {
    const engine = outside.engine;
    if (engine) {
      // (T.461's review: a release() that throws does not hide what Python said)
      try {
        gpuOnlyEnding = Promise.all([gpuOnlyEnding, engine.release()]).then(() => {});
      } catch (error) {
        console.warn(`engine: the engine of a model that failed as it was built could not be let go of (${told(error)}): its threads and GPU's worker may stay`);
      }
    }
    throw thrown;
  }
}
// the key of the model being loaded (cpuOnly)
let loadingKey: string | undefined;
// T.156: the models whose GPU failed while they were on it alone, loaded again on the CPU from then on (this visit)
const cpuOnly = new Set<string | undefined>();
// T.329.2.2: what the engine of the model loaded now reads its table of every layer's inputs from (a sync access handle
// of the kept checkpoint: the only one its file can have), or undefined; and the models whose table could not be read
// so, which keep it in the memory from then on (this visit)
let rowsNow: Reader | undefined, rowsWhole: (() => boolean) | undefined;
// T.503: the load now is of a model whose entry keeps llama.cpp's blocks, which this load widens (it does not run on
// the int8 kernels: convert.ts)
let blocksWidened = false;
const tableInMemory = new Set<string | undefined>();
const modelKey = (model: Listed) => model.hf ? `hf:${model.hf.repo}@${model.hf.revision}` : model.id ?? model.name;

/** A load begins: no model is on the GPU alone yet, and nothing of its weights is placed before the adapter is known
 * (T.156) and the GPU's worker of a model on the GPU alone that was let go before it was built has ended (its review) */
export async function loadBegan(model: Listed) {
  loadingKey = modelKey(model);
  gpuOnlyNow = undefined;
  blocksWidened = false;  // (T.503's review: a load that converts nothing does not say it: not the last load's)
  await adapterAsked;
  await gpuOnlyEnding;
}
/** T.329.2.2: the model let go of: the file its table of every layer's inputs was read from too (loaded.ts's letGo()) */
export async function rowsLetGo() {
  const reader = rowsNow;
  rowsNow = rowsWhole = undefined;
  await reader?.close();
}
/** T.329.2.2: after a model is built: whether the table of every layer's inputs it reads from the file it was kept in as
 * it came is all there (true; or it has none so read). Where a write failed before the table was whole, the model keeps
 * it in the memory from now on this visit, and is loaded again (false) */
export function rowsReady(model: Listed) {
  if (!rowsWhole || rowsWhole()) return true;
  console.warn("kept: the file this model was kept in as it came stopped taking its bytes before its table of every layer's inputs was whole: loaded again with the table in the memory");
  tableInMemory.add(modelKey(model));
  return false;
}
/** T.329.2.2: whether a model loading now may read its table of every layer's inputs from where it is kept (not where it
 * could not before, this visit) */
export const tableApart = () => !tableInMemory.has(loadingKey);
/** T.329.2.2: the model loading now keeps its table in the memory from now on (its kept file could not be opened) */
export function tableInMemoryFromNow(model: Listed) {
  tableInMemory.add(modelKey(model));
}
/** T.503: the model loading now keeps llama.cpp's blocks in its entry, and this load widens them (true), or not */
export function blocksWidenedNow(widened: boolean) {
  blocksWidened = widened;
}
/** T.156: the GPU stopped under this model while it was on the GPU alone: on the CPU from now on, this visit */
export function onCpuFromNow(model: Listed) {
  cpuOnly.add(modelKey(model));
}
/** The bytes of forward.js's memory of the model loaded now (0 where Python holds the weights, or nothing is loaded) */
export const weightsBytes = () => weightsNow?.buffer.byteLength ?? 0;
/** Whether the memory the weights are in is a shared one (software threads can be had on it) */
export const memoryShared = () => Boolean(weightsPool?.shared);
/** T.156: why the GPU stopped under the model on the GPU alone, or undefined (no such model, or its GPU goes on) */
export const gpuOnlyLost = () => gpuOnlyNow?.lost;

// a Python bytearray that JavaScript fills in place
export function pythonBuffer(size: number) {
  const buffer = pyodide.globals.get("bytearray")(size);
  const write: Write = (offset, chunk) => {
    // the view is taken anew every time: it dies when the WebAssembly memory grows
    const view = buffer.getBuffer("u8");
    view.data.set(chunk, offset);
    view.release();
  };
  return { buffer, write };
}

// Where the weights of a model go (T.93): the WebAssembly memory of forward.ts, where the forward pass runs, or,
// without the kernels (?without=kernels, or no WebAssembly SIMD), a Python bytearray for the NumPy engine.
// write(offset, bytes) fills it, slice(begin, end) copies a stretch out (to keep a conversion), llama() makes the
// engine on it ({ llama, outside }: Python's Llama and, where forward.ts runs it, what forward.ts made for it; loaded.ts
// takes the two), destroy() lets go of the Python buffer (a memory of forward.ts goes with the engine).
// a software thread of forward.ts (stage 2): a module worker of its own, ready once its kernels are warm
const spawnThread = (data: unknown) => new Promise<Thread>((resolve, reject) => {
  const worker = new Worker(new URL("../helper.ts", import.meta.url), { type: "module" });
  worker.onmessage = () => resolve({ terminate: () => worker.terminate() });
  worker.onerror = (event) => {
    void workerWentWithout(event, import.meta.url);  // (T.439: fewer threads, and the page is told where that is the site's update)
    reject(new Error(event.message ?? "a software thread did not start"));
  };
  worker.postMessage(data);
});

// T.135: the GPU's worker of forward.ts, a module worker of its own; forward.ts starts it
// (T.439: forward.ts holds its error, and the model stays on the CPU; beside that the page is told where it is the site's update)
function openGpu() {
  const worker = new Worker(new URL("../gpu.ts", import.meta.url), { type: "module" });
  worker.addEventListener("error", (event) => void workerWentWithout(event, import.meta.url));
  return worker;
}

// T.96: one memory of forward.ts, kept from model to model. A browser reserves address space for every WebAssembly
// memory, shared or not and whatever its maximum, and Chromium refused the third one of a page: the benchmark's
// first round, or a visitor's second change of model, then found no memory at all. So a memory is kept and grown
// as long as the next model fits under its maximum, and made anew only for a larger one. The maximum comes from the
// model (weightsMemory: what it needs and a gigabyte): asking for 4 GB up front left a phone no room for Pyodide's
// own memory, and "Loading Pyodide" never ended (2026-09-25).
// "Fits" is the checkpoint and what the forward pass puts after it (after: forward.ts's footprint(), T.115). Pythia
// 1B's checkpoint (1.1 GB) fitted the memory made for tiny-lm (1.2 GB) and its corrections did not ("Maximum memory
// size exceeded"; opened first, it ran; the review of T.96). Where a new memory would get no more room than this one
// (the browser gave less than it asked for, or it is not shared and grows as far as any would), the checkpoint
// fitting is enough. shared is what was asked for, not what the browser gave: a device without shared memories
// made one for every model.
// T.242: the switches (?without=) of each load the page says will follow on the model of its init, and no other model
// ({ ahead: [[...], ...] }: /benchmark/'s model section, whose worker loads one model, for the page's path and again
// for each round). The shared memory is then made for the largest of those loads and no more: the gigabyte that
// pooledWeights() keeps for the next model (T.96) is for a model that never comes there, and it took the page down.
// Playwright's WebKit on Windows (bench.yml, 2026-10-01) ended the page's process in new WebAssembly.Memory() of that
// maximum (1.07 GiB for tiny-lm) in about one run in four, once the CPU section's worker, ended before, had had a
// shared memory of its own; with the maximum of the model alone (45 MB) it never did (0 of 39 runs), and neither
// section alone ends it. undefined on the model page: any model may follow.
// ahead: what the forward pass of the largest of loadsAhead puts after the checkpoint, or undefined
function pooledWeights(size: number, after: number, shared: boolean, wide: boolean, ahead?: number) {
  const pages = (bytes: number) => Math.ceil(bytes / 65536);
  const needs = (base: number) => pages(base + size + (weightsPool!.limited ? 0 : after)) + 1;
  // what a memory made now is for: the load going on, and the largest of those that follow it where the page said which do
  const largest = ahead === undefined ? after : Math.max(after, ahead);
  const fits = weightsPool && weightsPool.asked === shared && weightsPool.wide === wide && needs(weightsPool.base) <= weightsPool.maximum;
  if (!fits) {
    // nothing may hold the old memory while the new one is made (T.96: Chromium refused a page's third)
    weightsPool = weightsNow = undefined;
    let memory: Memory | undefined, base!: number;
    if (shared) {
      try {
        // (a page more where nothing follows, as needs() counts one past what the model takes)
        ({ memory, base } = forwardModule!.weightsMemory(size, { shared: true, wide, after: largest, ...(ahead !== undefined && { spare: 65536 }) }));
      } catch (error) {
        // no shared memory here: one thread (T.380.2: said, as the lowered one below is)
        console.warn(`memory: no shared memory for this model (${told(error)}): a memory that is not shared, and one thread`);
        memory = undefined;
      }
      // (T.130's review) a shared memory the browser gave at a lowered maximum (weightsMemory's second and third try: the
      // checkpoint and a gigabyte, or a quarter of one) that the forward pass does not fit would run out of memory when
      // the cache grows, or at once where the corrections do not fit, after the whole checkpoint was read: a plain memory
      // grows as far as the browser allows. One thread, but the model reaches the end of its context. (T.242's review: for the
      // loads that follow as well, where the page said which: a round of /benchmark/ that widens int8 needs the most)
      if (memory?.limited && memory.maximum! < pages(base + size + largest) + 1) {
        console.info(`memory: the browser gave a shared memory of ${Math.round(memory.maximum! * 65536 / 2 ** 20)} MiB, and this model needs ` +
          `${Math.round(pages(base + size + largest) * 65536 / 2 ** 20)} MiB: a memory that is not shared, and one thread`);
        memory = undefined;
      }
    }
    if (!memory) ({ memory, base } = forwardModule!.weightsMemory(size, { wide }));
    const isShared = shared && memory.buffer instanceof SharedArrayBuffer;
    // a memory without a maximum (not shared) grows as far as the browser allows: 4 GB of pages, 16 GB when wide
    const most = wide ? 262144 : 65536;
    weightsPool = { memory, base, wide, asked: shared, shared: isShared, maximum: isShared ? memory.maximum ?? most : most,
                    limited: !isShared || Boolean(memory.limited) };
  }
  const { memory, base } = weightsPool!;
  const more = pages(base + size) + 1 - memory.buffer.byteLength / 65536;
  if (more > 0) forwardModule!.growMemory(memory, more, wide);
  return weightsPool!;
}

// T.115: what the forward pass of a checkpoint of size bytes puts after it, at most (forward.ts's footprint()): from
// its header (the 7 ints) and the options it is loaded with (their form: head_dim where a head is not dim / heads,
// T.124: the keys and values of a Qwen3 0.6B are twice what the header says), on a shared memory or not (an int8
// model's keys and values may be float16; forward.ts's keysInHalf says whether they are, T.160, T.130).
// What footprint() takes (the worker asks keysInHalf the same).
// without: the switches of the load (those of the load going on, or of one that follows: loadsAhead, T.242).
// T.363.5, header: of a model whose form gives rows kinds of their own, Python counts the weights that are still the
// dtype's (dtype_weights(): footprint() has no rows to ask, and takes every weight of the file for one otherwise)
function forwardOptions(header: Header, options: Options, shared: boolean, without = switchesNow()): Footprint {
  const { dtype = "float32" } = options;
  const int8 = !without.includes("int8");
  return {
    ...options, dtype, int8, relaxed: Boolean(jsKernels?.relaxed) && !without.includes("relaxed"),
    ...(options.kinds ? { weights: llama2_numpy.dtype_weights(header, dtype, options) } : {}),
    // (T.375.5: of packed weights only, which footprint() knows by the dtype)
    halfKV: int8 && !without.includes("kv16"), shared,
    outliers: llama2_numpy.OUTLIER_CHANNELS, gpu: hasWebGpu,
  };
}
const afterCheckpoint = (header: Header, size: number, options: Options, shared: boolean, without?: string[]) => forwardModule!.footprint(header, size, forwardOptions(header, options, shared, without));
/** T.363.5: whether a load now runs on the int8 kernels of forward.ts, which is where llama.cpp's blocks are multiplied
 * as they are: not where NumPy computes (?without=kernels, no WebAssembly SIMD) nor where the weights are widened to
 * float32 (?without=int8). A conversion keeps a GGUF's blocks only for such a load; any other gets them widened */
export const runsPacked = () => Boolean(jsKernels) && !switchesNow().some((left) => left === "kernels" || left === "int8");
// the page cross-origin isolated (stage 3), shared memories to be had, and not ?threads=1: the memory is shared
const sharedWanted = () => Boolean(sharedKernels && self.crossOriginIsolated && threadsRequest?.fixed !== 1);

// T.115: the bits of a model converted with none asked for (weightsFor() in src/models.ts asks for six bits where the
// device says it has too little memory): int8 unless its forward pass does not fit a 32-bit memory and this browser
// has no 64-bit one (T.133), then six bits (T.98: 7/9 of int8's memory, and about half as fast). The converter calls
// this once it knows the header: the size of either (sizes) and what the forward pass puts after them depend on it.
// form: what else sizes the forward pass (llama2_numpy.FORM, llama2_convert.Stream.form).
// T.329.2.2, apart: the table of every layer's inputs will not be in the memory (a file to keep the conversion in as it
// comes is open): int8 then where the rest fits
export function automaticBits(header: List<Header>, form: Dict<Partial<Form>>, sizes: Dict<{ int8: number }>, apart = false) {
  const ints = header.toJs(), int8 = sizes.toJs({ dict_converter: Object.fromEntries }).int8;
  const given = form.toJs({ dict_converter: Object.fromEntries });
  header.destroy();
  sizes.destroy();
  form.destroy();
  if (!forwardModule) return "int8";  // no forward.ts (no WebAssembly SIMD): NumPy widens every weight anyway
  const shared = sharedWanted();
  const hole = apart && given.per_layer && tableApart() ? forwardModule.tableHole(placesOf(ints, "int8", given)) : null;
  return forwardModule.automaticDtype(int8 - (hole ? hole[1] - hole[0] : 0), afterCheckpoint(ints, int8, { ...given, dtype: "int8" }, shared),
    Boolean(wideKernels?.plain));
}

// header: the checkpoint's 7 ints, options: what it is loaded with (its dtype and arch): what the forward pass puts
// after the checkpoint follows from them (T.115). T.156, keep: where a conversion is kept as it comes (kept.js's
// keeper()), for a model on the GPU alone, whose weights nothing holds whole to keep afterwards
// T.329.2.2, later: the checkpoint is read from where it is kept, which can be read from at once once it is all read
// (attach()): a table of every layer's inputs may stay there. keep: the file of the first visit, to read from as it comes
export function weightsBuffer(size: number, header: Header, options: Options, keep?: Keeper, later = false): Weights {
  if (jsKernels && !switchesNow().includes("kernels")) {
    // a shared memory where the page is cross-origin isolated (stage 3), unless ?threads=1; else one thread
    const wanted = sharedWanted();
    // T.101: a model past 4 GiB with its forward pass goes on a 64-bit memory (about a tenth slower: only when it has
    // to). T.130: where a shared one is refused after all, a model that fit a 32-bit one still does on the plain one
    // (footprint() keeps float16 keys and values there where only float32 would not fit), and a 64-bit one stays
    // 64-bit (with float32 keys and values, on a memory that has no maximum): the 64-bit question stays answered
    const after = afterCheckpoint(header, size, options, wanted);
    // T.329.2.2: a table of every layer's inputs stays where the checkpoint is kept (the file of this visit's conversion,
    // where it is kept as the dtype it is converted to; or a kept one): the memory holds the rest, held bytes. footprint()
    // was asked of the whole checkpoint, and counts the table for nothing
    const keeps = keep && keep.dtype === options.dtype;
    const hole = options.per_layer && tableApart() && (keeps || later) ? forwardModule!.tableHole(placesOf(header, options.dtype ?? "float32", options)) : null;
    const held = hole ? size - (hole[1] - hole[0]) : size;
    // T.129 (7): a model past even a 64-bit memory is refused here, before its weights are fetched (a Qwen2.5 32B of
    // ?hf=, about 37 GB as int8, began a 65 GB download and failed at 7.8 GB). The words are the owner's (2026-09-28)
    if (forwardModule!.pastWide(held, after)) {
      throw new Error(`This model is too large for a web page: it needs about ${Math.ceil((held + after) / 1e9)} GB of ` +
        "memory, and a browser gives a page 16 GB at most.");
    }
    // T.148: the layers on the GPU are a second copy of them, in the same memory where the GPU is a phone's or an
    // Apple's: both, with the rest of this model, within half of what the device says it has (as src/models.ts's
    // weightsFor asks for six bits past half). T.156: a model that does not fit so goes on the GPU alone where it can
    // (the owner's B, 2026-09-27: forward.ts's weightsPlace; the double copy held to 6.5 GiB where the device says 8),
    // decided before any memory is made for it (a memory of the whole checkpoint would be one too many, T.96)
    const deviceMemory = self.navigator?.deviceMemory ?? 4;
    const onGpu = gpuOnlyWeightsFor(held, header, options, after, deviceMemory);
    if (onGpu.mode === "gpu") return gpuOnlyBuffer(size, header, options, onGpu as Required<Placed>, keep);
    const gpuRoom = onGpu.gpuRoom;
    const wide = forceWide || forwardModule!.needsWide(held, after);
    if (wide && !wideKernels?.plain) {
      if (wideUpdated) throw wideUpdated;  // (T.475: the page is told the site was updated, not that this browser has none)
      throw new Error("This model needs more than 4 GB of memory, which this browser cannot give a web page (no 64-bit " +
        "WebAssembly memory: Safari has none yet). Chrome and Firefox can.");
    }
    // T.242: where the page said which loads follow on this model, the memory is made for the largest of them (a load
    // without the kernels takes none: NumPy's weights are Python's)
    const ahead = loadsAhead && Math.max(0, ...loadsAhead.filter((without) => !without.includes("kernels"))
      .map((without) => afterCheckpoint(header, size, options, wanted, without)));
    const { memory, base, shared } = pooledWeights(held, after, wanted && (!wide || Boolean(wideKernels!.shared)), wide, ahead);
    // T.160: the type of the keys and values that after counts, T.130: on the memory the browser gave
    const halfKeys = forwardModule!.keysInHalf(header, size, forwardOptions(header, options, shared));
    const kernels = (wide ? (shared ? wideKernels!.shared : wideKernels!.plain) : (shared ? sharedKernels : jsKernels))!;
    const spawn = shared ? spawnThread : undefined;
    weightsNow = memory;
    // T.329.2.2: where the table is apart, every byte to the file as it comes, and what is not the table to the memory
    // (after the table, as much lower as it is long); the table's rows are read from the file (a scratch of the widest
    // read, then into the memory: a shared one is not handed to the file system)
    let reader: Reader | undefined = hole && keeps ? keep : undefined;
    if (hole && keeps) keep.holding(hole);
    const place = hole ? forwardModule!.placer([hole]) : (offset: number) => offset;
    const write: Write = !hole ? (offset, chunk) => new Uint8Array(memory.buffer, base + offset, chunk.length).set(chunk) : (offset, chunk) => {
      const end = offset + chunk.length;
      if (offset < hole[0]) new Uint8Array(memory.buffer, base + offset, Math.min(end, hole[0]) - offset).set(chunk.subarray(0, Math.min(end, hole[0]) - offset));
      if (end > hole[1]) new Uint8Array(memory.buffer, base + place(Math.max(offset, hole[1])), end - Math.max(offset, hole[1])).set(chunk.subarray(Math.max(offset, hole[1]) - offset));
      if (keeps) keep.write(offset, chunk);
    };
    let scratch = new Uint8Array(0);
    const rows = (offset: number, bytes: number, to: number) => {
      if (scratch.length < bytes) scratch = new Uint8Array(bytes);
      const into = scratch.subarray(0, bytes);
      if (reader!.read(into, offset) !== bytes) throw new Error("The kept model gave fewer bytes than a row of its table has.");
      new Uint8Array(memory.buffer, to, bytes).set(into);
    };
    const apart: Apart | undefined = hole ? { place, stored: held, rows } : undefined;
    return {
      hole,
      attach(given) {
        reader = given;
        rowsNow = given;  // (T.329.2.2, review: let go of by the next load's letGo() even if llama() below never ran or threw)
      },
      write,
      slice: (begin, end) => new Uint8Array(memory.buffer, base + begin, end - begin).slice(),
      llama: (tokenizer, options) => {
        // (T.329.2.2: the file the table is read from is the loaded model's, let go of with it)
        if (apart) {
          rowsNow = reader;
          rowsWhole = reader === keep ? keep!.whole : undefined;
        }
        // (not on the benchmark's page, ?bench=, T.45: it times the CPU's combinations, which a GPU starting beside
        // them would slow down with its upload and compilation; its first load is one of them)
        // (nor under the rounds of /benchmark/'s model section, T.184: the same; its page path, timed on the first load
        // before the rounds, has the GPU, and T.205: the rounds' loads wait for that GPU to let go of its device)
        const outside = forwardModule!.external({ memory, base, size, kernels, spawn, gpu: hasWebGpu && !benchPage && !benching ? openGpu : undefined, gpuRoom,
          gpuRemembered: gpuRequest?.remembered, gpuForce, halfKeys, apart, oneThread: threadsRequest?.fixed === 1 });
        return { llama: builtOrLetGo(outside, () => llama2_numpy.Llama.callKwargs(null, tokenizer, { ...options, external: outside })), outside };
      },
      destroy() {},
    };
  }
  weightsNow = undefined;
  // T.503: a model of blocks widened past what Python's 32-bit memory holds runs on the SIMD kernels alone (Saluki:
  // about 32 GB widened). The words are the owner's (2026-10-15)
  if (blocksWidened && size > 2 ** 32) throw new Error("This model runs on the SIMD kernels only, which this browser (or ?without=kernels) leaves out.");
  const { buffer, write } = pythonBuffer(size);
  return {
    hole: null,
    attach() {},
    write,
    slice(begin, end) {
      const view = buffer.getBuffer("u8");
      const copy = view.data.slice(begin, end);
      view.release();
      return copy;
    },
    llama: (tokenizer, options) => ({ llama: llama2_numpy.Llama.callKwargs(buffer, tokenizer, options) }),
    destroy: () => buffer.destroy(),
  };
}

// T.156: what a model on the GPU alone needs of this page and device, whatever the model (gpuOnlyUnfit has the rest): a
// memory the device says (T.205), the GPU's worker (not the benchmark's rounds), a shared memory and the 64-bit kernels,
// the int8 kernels, an adapter that is no fallback (but for the tests), and no failure of the GPU under this model before
// (cpuOnly). convert() asks it too before it opens a file to keep a conversion in as it comes (the review of T.156)
// (the owner, 2026-09-27: and no verdict the page kept that the CPU is faster here: then the CPU at once)
// (T.375.5: and a dtype whose packing may be there at all, forward.ts's aloneTakes)
export const gpuOnlyPossible = (dtype: Dtype) => gpuOnlyAllowed() && forwardModule!.aloneTakes(dtype) && !aloneKept(dtype);
const gpuOnlyAllowed = () => hasWebGpu && self.navigator?.deviceMemory !== undefined && !benchPage && !benching && sharedWanted() &&
  Boolean(wideKernels?.shared) && !switchesNow().includes("int8") && !cpuOnly.has(loadingKey) &&
  Boolean(gpuAdapter) && (!gpuAdapter!.fallback || Boolean(gpuForce.fallback));
// T.156: whether the page kept that the CPU was faster here than this model on the GPU alone (forward.ts's aloneHolds)
// (dtype, T.232: the key is the one gpu.ts gave the verdict, a ternary model's its own)
// (T.547.3.7: a model whose rows have kinds of their own, Saluki's blocks, by the key of a model of blocks)
const aloneKept = (dtype: Dtype, blocks = false) => Boolean(gpuAdapter) &&
  forwardModule!.aloneHolds(gpuRequest?.remembered?.alone, gpuAdapter!.keys?.[dtype === "ternary" ? "ternary" : blocks ? "blocks" : "plain"], gpuRequest?.cpu);
// T.156: where a model goes ({ mode: "both" | "gpu" | "cpu", gpuRoom }, forward.ts's weightsPlace): the GPU alone only
// for a Llama of int8 (T.232: or of ternary weights) the GPU's steps take (gpuOnlyUnfit), where the page and the device
// may (gpuOnlyPossible)
function gpuOnlyWeightsFor(size: number, header: Header, options: Options, after: number, deviceMemory: number): Placed {
  const { dtype = "float32" } = options;
  const cpu = size + after, gpu = forwardModule!.gpuBytes(header, { ...options, dtype });
  // (T.232: with the form's rotated basis, which the GPU does not turn the inputs for (T.237): the worker did not hand it
  // on, and a Llama in a rotated basis would have gone on the GPU alone to be loaded again on the CPU once built.
  // T.375.5: the whole form as the options carry it: the layers that keep a state are read off it too)
  const fit = gpuOnlyAllowed() && !forwardModule!.gpuOnlyUnfit(header, dtype, options, gpuAdapter, gpuForce);
  const eligible = fit && !aloneKept(dtype, forwardModule!.blocksForm(options));
  // (the second review of T.156: the status line then says only the memory's reason where the layers do not fit beside
  // the CPU's copy; the console says the kept verdict, and what asks again)
  if (fit && !eligible) {
    console.info("gpu: the CPU as /benchmark/ measured it was faster here than this model on the GPU alone, as the page kept it: " +
      "on the CPU (a new run of /benchmark/'s CPU section, another browser version or new shaders weigh the two again)");
  }
  if (!eligible) return forwardModule!.weightsPlace({ cpu, gpu, deviceMemory });
  // (T.571: of the whole form as the options carry it, as gpuOnlyUnfit() is asked above: four keys of it went before, and a
  // Qwen3.5 on the GPU alone, whose linear-attention layers are in the form's linear, was refused by Python)
  const tensors = placesOf(header, dtype, options), stored = size - forwardModule!.gpuHoles(tensors).reduce((sum, [a, b]) => sum + b - a, 0);
  const gpuOnly = stored + afterCheckpoint(header, stored, { ...options, direct: true }, true);
  return { ...forwardModule!.weightsPlace({ cpu, gpuOnly, gpu, deviceMemory, eligible, forced: gpuForce.only }), tensors, stored, gpuOnly };
}
// llama2_numpy.external_tensors(): where every tensor of the checkpoint is, from its header alone
function placesOf(header: Header, dtype: Dtype, form: Partial<Form>) {
  const proxy = llama2_numpy.external_tensors(header, dtype, form);
  try {
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    proxy.destroy();
  }
}
// T.156: the weights of a model on the GPU alone: the layers' matrices to the GPU's worker as they come (forward.ts's
// gpuOnlyWeights), the rest into a memory of their size, and every byte to keep's file where the conversion is kept
function gpuOnlyBuffer(size: number, header: Header, options: Options, { tensors, stored, gpuOnly }: Required<Placed>, keep?: Keeper): Weights {
  // (T.576: the GPU's plan first: what makes it may throw (gpuPacking()), before a memory is taken or a worker opened)
  const plan = forwardModule!.gpuOnlyPlan(header, tensors, gpuForce, gpuRequest?.remembered, options);
  // (T.581: and what else is reckoned of the header alone, which may throw as well)
  const layerWeights = forwardModule!.layerWeightsOf(header, options);
  const after = gpuOnly - stored, wide = forwardModule!.needsWide(stored, after);
  const { memory, base, shared } = pooledWeights(stored, after, true, wide);
  if (!shared) throw new Error("This browser gave no shared memory for a model on the GPU alone.");
  const worker = openGpu();
  // (forward.ts listens from start() on. Here only "ended", beside it: T.205, the next load after the GPU's worker let
  // go of its device and its buffers, which it makes for every layer as it opens, before a byte comes)
  let ended: () => void;
  const ending = new Promise<void>((resolve) => { ended = resolve; });
  worker.addEventListener("message", ({ data }) => data?.type === "ended" && ended());
  worker.addEventListener("error", () => ended());
  // the GPU's worker stopped, and once it said "ended" (or after GPU_END_MS: terminated), resolved; as release() does
  const end = () => {
    worker.postMessage({ type: "stop" });
    let timer: ReturnType<typeof setTimeout>;
    const late = new Promise<void>((resolve) => { timer = setTimeout(resolve, forwardModule!.GPU_END_MS); });
    return Promise.race([ending, late]).then(() => {
      clearTimeout(timer);
      worker.terminate();
    });
  };
  // (T.581: a worker opened is let go of at once where what writes to it cannot be made or it cannot be told to open:
  // it makes its device and buffers once it opens, from a plan it was handed whole)
  let weights: GpuOnlyWeights;
  // T.377: a worker that says "error" before the engine is built (its script did not arrive, or it threw) takes no more
  // of the weights and counts none: the loops that write stop waiting for it at once (they waited 60 s, and the load
  // failed), and the engine, which listens from start() on, is handed the error it missed once it is built (failed)
  let failed: ErrorEvent | undefined;
  try {
    weights = forwardModule!.gpuOnlyWeights({ memory, base, size, tensors, worker });
    worker.addEventListener("error", (event) => {
      failed ??= event;
      weights.gone();
    });
    worker.postMessage({ type: "open", plan, flow: weights.flow } satisfies ToGpu);
  } catch (error) {
    worker.terminate();
    throw error;
  }
  // (T.156: the checkpoint's size, its layers' multiply-adds, /benchmark/'s CPU reading and the page's use, for the
  // estimate the GPU is held against: forward.ts's aloneVerdict)
  const direct: GpuOnly = gpuOnlyNow = { worker, lost: null, stored, size, place: weights.place,
    layerWeights, cpu: gpuRequest?.cpu, usage: gpuRequest?.usage ?? forwardModule!.USAGE_UNKNOWN,
    room: weights.room, drained: weights.drained, onLost: (why) => { direct.lost = why; }, end };
  weightsNow = memory;
  const kernels = (wide ? wideKernels!.shared : sharedKernels)!;
  let built = false;
  return {
    direct,
    hole: null,
    attach() {},
    write(offset, chunk) {
      weights.write(offset, chunk);
      keep?.write(offset, chunk);
    },
    room: weights.room,
    drained: weights.drained,  // every byte of the layers on the GPU (or the GPU failed: start() says so)
    llama: (tokenizer, engineOptions) => {
      // (no oneThread, T.371: a model on the GPU alone is one on a shared memory, which ?threads=1 does not ask for)
      const outside = forwardModule!.external({ memory, base, size, kernels, spawn: spawnThread, gpu: () => worker, gpuForce, halfKeys: true, direct,
        gpuRemembered: gpuRequest?.remembered });
      const made = builtOrLetGo(outside, () => llama2_numpy.Llama.callKwargs(null, tokenizer, { ...engineOptions, external: outside }));
      built = true;
      // (T.377: the engine's side of the GPU says why, as of a worker that failed under it, and the model is loaded
      // again on the CPU: gpuOnlyReady(). An engine that started no worker has said its own reason already)
      if (failed) worker.onerror?.(failed);
      return { llama: made, outside };
    },
    // (the review of T.156: a load cancelled or failed before its engine was built, or another try of the converter's
    // (sink.open() again), let go of the GPU's worker: nothing else would, and it held the device and a buffer for
    // every layer's matrices, 4.1 GB for Llama 3.2 3B, for the rest of the visit. The next load waits for it)
    destroy() {
      if (!built) gpuOnlyEnding = Promise.all([gpuOnlyEnding, end()]).then(() => {});  // (T.461: after a release() too)
    },
  };
}

// T.156: where the loops that write the weights can wait for the GPU's worker of a model on the GPU alone (room), and
// before its engine is built (drained)
export const weightsRoom = () => gpuOnlyNow?.room?.();
export const weightsDrained = () => gpuOnlyNow?.drained?.();
// T.156: after a model on the GPU alone is built: its GPU ready (true), or failed (false: the worker let go of it, and
// the model goes on the CPU from now on this visit)
// engine: forward.ts's engine of the model just built (loaded.ts's engineNow())
export async function gpuOnlyReady(model: Listed, id: number, engine: Engine | undefined) {
  const direct = gpuOnlyNow;
  if (!direct || engine === undefined) return true;
  await engine.gpu;
  if (!direct.lost) return true;
  console.warn(`gpu: ${direct.lost}: the model was on the GPU alone, and is loaded again on the CPU`);
  cpuOnly.add(modelKey(model));
  // (the CPU was faster: the page keeps it for this model and device once the load on the CPU is ready, and the next
  // visit loads on the CPU at once. The load's id: the page reads no word of a load another choice cancelled, the
  // second review of T.156: a verdict that came late was kept for the model chosen since)
  if (direct.verdict) postMessage({ type: "gpu-alone", load: id, alone: direct.verdict } satisfies GpuAlone);
  // (the GPU's worker let go of its device: stopped by forward.ts where it started it, and here where it did not, as a
  // model whose GPU forward.ts gave up as the engine was built, before start(): its release() has no "ended" to wait
  // for then. The review of T.156: the load on the CPU begins after it, T.205)
  await direct.end();
  gpuOnlyNow = undefined;
  return false;
}

// T.93: where the converter writes the checkpoint, piece by piece, straight into where the engine will read it (a
// Python buffer on the way would stay: Pyodide's memory never shrinks). sink is what the converter calls
// (llama2_convert.Writer), weights what sink.open() made (weightsBuffer), bytes its size; release() lets go of it once
// (T.145: a conversion that failed after sink.open() left a Python buffer of the model's size behind with
// ?without=kernels). Out of convert() so that tests/worker-sink-check.mjs can follow sink.open() to footprint().
export function checkpointSink(keep?: Keeper) {
  const into = {
    weights: undefined, bytes: 0,
    release() {
      into.weights?.destroy();
      into.weights = undefined;
    },
  } as { weights: Weights | undefined; bytes: number; release(): void; sink: Sink };
  into.sink = {
    // form: what lays out the checkpoint and sizes the forward pass besides the header (llama2_numpy.FORM, T.115, T.144)
    open(bytes, header, dtype, form) {
      into.release();  // an earlier try (another tokenizer) that got this far
      into.weights = weightsBuffer(bytes, header.toJs(), { dtype, ...form.toJs({ dict_converter: Object.fromEntries }) }, keep);
      header.destroy();
      form.destroy();
      into.bytes = bytes;
    },
    write(offset, array) {
      const view = array.getBuffer("u8");
      into.weights!.write(offset, view.data);
      view.release();
    },
  };
  return into;
}
