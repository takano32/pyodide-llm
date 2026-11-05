// worker/weights.ts (T350): where the weights of a model go: the memory of forward.ts that is kept from model to model,
// a Python buffer where NumPy runs, or the GPU alone (T156), and the sink a conversion writes them through.
// T641.1: the memory is worker/pool.ts's, the GPU alone worker/alone.ts's; this file holds the table of every layer's
// inputs read from where a model is kept (T329.2.2) and what a load widens (T503).
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T367.3).

import { pyodide, llama2_numpy, forwardModule, jsKernels, sharedKernels, wideKernels, wideUpdated, forceWide, gpuForce, benchPage, hasWebGpu,
  adapterAsked } from "./runtime.ts";
import { threadsRequest, gpuRequest, loadsAhead, benching, switchesNow } from "./request.ts";
import { pooledWeights, forwardOptions, afterCheckpoint, sharedWanted, placesOf, pooled, weightsInPython } from "./pool.ts";
import { gpuOnlyEnding, builtOrLetGo, loadingKey, modelKey, aloneBegan, spawnThread, openGpu, gpuOnlyWeightsFor, gpuOnlyBuffer } from "./alone.ts";
import type { GpuOnly, Placed } from "./alone.ts";
import type { keeper, Reader } from "../kept.ts";
import type { Form } from "../contracts/form.ts";
import type { Listed } from "../contracts/messages.ts";
import type { Bytes, Dict, List, Llama, Options, Sink } from "../contracts/python.ts";
import type { Apart, GpuOnlyWeights, Header, Outside } from "../forward.ts";

// T368.3, the shapes this file makes and hands on.
// what writes a stretch of the checkpoint where it belongs
export type Write = (offset: number, chunk: Uint8Array<ArrayBuffer>) => void;
// a model built on its weights: Python's Llama and, where forward.ts runs it, what forward.ts made for it (loaded.ts)
export type Made = { llama: Llama; outside?: Outside };
// what an engine is made with here: the kernels' path for Python (the sampling) and the switches left out, over the options
type EngineOptions = Options & { kernels?: string; disable?: string[] };
// kept.ts's keeper(): the file a conversion is kept in as it comes (T156)
export type Keeper = NonNullable<Awaited<ReturnType<typeof keeper>>>;
// The place of a model's weights (weightsBuffer()): in a memory of forward.ts or in a Python buffer, where slice()
// copies a stretch out; or on the GPU alone (direct), where nothing holds them whole
// (T329.2.2, hole: the stretch of the checkpoint the memory does not hold, a table of every layer's inputs read from
// where the checkpoint is kept: the first visit's file as it comes (the keeper), or, handed in by attach() before
// llama(), a kept one's; null where the memory holds all of it)
export type Weights =
  | { write: Write; slice(begin: number, end: number): Uint8Array<ArrayBuffer>; llama(tokenizer: Bytes, options: EngineOptions): Made; destroy(): void; direct?: undefined;
    hole: [number, number] | null; attach(reader: Reader): void }
  | { write: Write; llama(tokenizer: Bytes, options: EngineOptions): Made; destroy(): void; direct: GpuOnly; room: GpuOnlyWeights["room"]; drained: GpuOnlyWeights["drained"];
    hole: null; attach(reader: Reader): void };

// T329.2.2: what the engine of the model loaded now reads its table of every layer's inputs from (a sync access handle
// of the kept checkpoint: the only one its file can have), or undefined; and the models whose table could not be read
// so, which keep it in the memory from then on (this visit)
let rowsNow: Reader | undefined, rowsWhole: (() => boolean) | undefined;
// T503: the load now is of a model whose entry keeps llama.cpp's blocks, which this load widens (it does not run on
// the int8 kernels: convert.ts)
let blocksWidened = false;
const tableInMemory = new Set<string | undefined>();

/** A load begins: no model is on the GPU alone yet, and nothing of its weights is placed before the adapter is known
 * (T156) and the GPU's worker of a model on the GPU alone that was let go before it was built has ended (its review) */
export async function loadBegan(model: Listed) {
  aloneBegan(model);
  blocksWidened = false;  // (T503's review: a load that converts nothing does not say it: not the last load's)
  await adapterAsked;
  await gpuOnlyEnding;
}
/** T329.2.2: the model let go of: the file its table of every layer's inputs was read from too (loaded.ts's letGo()) */
export async function rowsLetGo() {
  const reader = rowsNow;
  rowsNow = rowsWhole = undefined;
  await reader?.close();
}
/** T329.2.2: after a model is built: whether the table of every layer's inputs it reads from the file it was kept in as
 * it came is all there (true; or it has none so read). Where a write failed before the table was whole, the model keeps
 * it in the memory from now on this visit, and is loaded again (false) */
export function rowsReady(model: Listed) {
  if (!rowsWhole || rowsWhole()) return true;
  console.warn("kept: the file this model was kept in as it came stopped taking its bytes before its table of every layer's inputs was whole: loaded again with the table in the memory");
  tableInMemory.add(modelKey(model));
  return false;
}
/** T329.2.2: whether a model loading now may read its table of every layer's inputs from where it is kept (not where it
 * could not before, this visit) */
export const tableApart = () => !tableInMemory.has(loadingKey);
/** T329.2.2: the model loading now keeps its table in the memory from now on (its kept file could not be opened) */
export function tableInMemoryFromNow(model: Listed) {
  tableInMemory.add(modelKey(model));
}
/** T503: the model loading now keeps llama.cpp's blocks in its entry, and this load widens them (true), or not */
export function blocksWidenedNow(widened: boolean) {
  blocksWidened = widened;
}

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

/** T363.5: whether a load now runs on the int8 kernels of forward.ts, which is where llama.cpp's blocks are multiplied
 * as they are: not where NumPy computes (?without=kernels, no WebAssembly SIMD) nor where the weights are widened to
 * float32 (?without=int8). A conversion keeps a GGUF's blocks only for such a load; any other gets them widened */
export const runsPacked = () => Boolean(jsKernels) && !switchesNow().some((left) => left === "kernels" || left === "int8");

// T115: the bits of a model converted with none asked for (weightsFor() in src/models.ts asks for six bits where the
// device says it has too little memory): int8 unless its forward pass does not fit a 32-bit memory and this browser
// has no 64-bit one (T133), then six bits (T98: 7/9 of int8's memory, and about half as fast). The converter calls
// this once it knows the header: the size of either (sizes) and what the forward pass puts after them depend on it.
// form: what else sizes the forward pass (llama2_numpy.FORM, llama2_convert.Stream.form).
// T329.2.2, apart: the table of every layer's inputs will not be in the memory (a file to keep the conversion in as it
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

// Where the weights of a model go (T93): the WebAssembly memory of forward.ts, where the forward pass runs, or,
// without the kernels (?without=kernels, or no WebAssembly SIMD), a Python bytearray for the NumPy engine.
// write(offset, bytes) fills it, slice(begin, end) copies a stretch out (to keep a conversion), llama() makes the
// engine on it ({ llama, outside }: Python's Llama and, where forward.ts runs it, what forward.ts made for it; loaded.ts
// takes the two), destroy() lets go of the Python buffer (a memory of forward.ts goes with the engine).
// header: the checkpoint's 7 ints, options: what it is loaded with (its dtype and arch): what the forward pass puts
// after the checkpoint follows from them (T115). T156, keep: where a conversion is kept as it comes (kept.js's
// keeper()), for a model on the GPU alone, whose weights nothing holds whole to keep afterwards
// T329.2.2, later: the checkpoint is read from where it is kept, which can be read from at once once it is all read
// (attach()): a table of every layer's inputs may stay there. keep: the file of the first visit, to read from as it comes
export function weightsBuffer(size: number, header: Header, options: Options, keep?: Keeper, later = false): Weights {
  const forward = forwardModule!;  // (T476: the forward module, which a load that places weights has: taken once)
  if (jsKernels && !switchesNow().includes("kernels")) {
    // a shared memory where the page is cross-origin isolated (stage 3), unless ?threads=1; else one thread
    const wanted = sharedWanted();
    // T101: a model past 4 GiB with its forward pass goes on a 64-bit memory (about a tenth slower: only when it has
    // to). T130: where a shared one is refused after all, a model that fit a 32-bit one still does on the plain one
    // (footprint() keeps float16 keys and values there where only float32 would not fit), and a 64-bit one stays
    // 64-bit (with float32 keys and values, on a memory that has no maximum): the 64-bit question stays answered
    const after = afterCheckpoint(header, size, options, wanted);
    // T329.2.2: a table of every layer's inputs stays where the checkpoint is kept (the file of this visit's conversion,
    // where it is kept as the dtype it is converted to; or a kept one): the memory holds the rest, held bytes. footprint()
    // was asked of the whole checkpoint, and counts the table for nothing
    const keeps = keep && keep.dtype === options.dtype;
    const hole = options.per_layer && tableApart() && (keeps || later) ? forward.tableHole(placesOf(header, options.dtype ?? "float32", options)) : null;
    const held = hole ? size - (hole[1] - hole[0]) : size;
    // T129 (7): a model past even a 64-bit memory is refused here, before its weights are fetched (a Qwen2.5 32B of
    // ?hf=, about 37 GB as int8, began a 65 GB download and failed at 7.8 GB). The words are the owner's (2026-09-28)
    if (forward.pastWide(held, after)) {
      throw new Error(`This model is too large for a web page: it needs about ${Math.ceil((held + after) / 1e9)} GB of ` +
        "memory, and a browser gives a page 16 GB at most.");
    }
    // T148: the layers on the GPU are a second copy of them, in the same memory where the GPU is a phone's or an
    // Apple's: both, with the rest of this model, within half of what the device says it has (as src/models.ts's
    // weightsFor asks for six bits past half). T156: a model that does not fit so goes on the GPU alone where it can
    // (the owner's B, 2026-09-27: forward.ts's weightsPlace; the double copy held to 6.5 GiB where the device says 8),
    // decided before any memory is made for it (a memory of the whole checkpoint would be one too many, T96)
    const deviceMemory = self.navigator?.deviceMemory ?? 4;
    const onGpu = gpuOnlyWeightsFor(held, header, options, after, deviceMemory);
    if (onGpu.mode === "gpu") return gpuOnlyBuffer(size, header, options, onGpu as Required<Placed>, keep);
    const gpuRoom = onGpu.gpuRoom;
    const wide = forceWide || forward.needsWide(held, after);
    if (wide && !wideKernels?.plain) {
      if (wideUpdated) throw wideUpdated;  // (T475: the page is told the site was updated, not that this browser has none)
      throw new Error("This model needs more than 4 GB of memory, which this browser cannot give a web page (no 64-bit " +
        "WebAssembly memory: Safari has none yet). Chrome and Firefox can.");
    }
    // T242: where the page said which loads follow on this model, the memory is made for the largest of them (a load
    // without the kernels takes none: NumPy's weights are Python's)
    const ahead = loadsAhead && Math.max(0, ...loadsAhead.filter((without) => !without.includes("kernels"))
      .map((without) => afterCheckpoint(header, size, options, wanted, without)));
    const { memory, base, shared } = pooledWeights(held, after, wanted && (!wide || Boolean(wideKernels!.shared)), wide, ahead);
    // T160: the type of the keys and values that after counts, T130: on the memory the browser gave
    const halfKeys = forward.keysInHalf(header, size, forwardOptions(header, options, shared));
    const kernels = (wide ? (shared ? wideKernels!.shared : wideKernels!.plain) : (shared ? sharedKernels : jsKernels))!;
    const spawn = shared ? spawnThread : undefined;
    pooled(memory);
    // T329.2.2: where the table is apart, every byte to the file as it comes, and what is not the table to the memory
    // (after the table, as much lower as it is long); the table's rows are read from the file (a scratch of the widest
    // read, then into the memory: a shared one is not handed to the file system)
    let reader: Reader | undefined = hole && keeps ? keep : undefined;
    if (hole && keeps) keep.holding(hole);
    const place = hole ? forward.placer([hole]) : (offset: number) => offset;
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
        rowsNow = given;  // (T329.2.2, review: let go of by the next load's letGo() even if llama() below never ran or threw)
      },
      write,
      slice: (begin, end) => new Uint8Array(memory.buffer, base + begin, end - begin).slice(),
      llama: (tokenizer, options) => {
        // (T329.2.2: the file the table is read from is the loaded model's, let go of with it)
        if (apart) {
          rowsNow = reader;
          rowsWhole = reader === keep ? keep!.whole : undefined;
        }
        // (not on the benchmark's page, ?bench=, T45: it times the CPU's combinations, which a GPU starting beside
        // them would slow down with its upload and compilation; its first load is one of them)
        // (nor under the rounds of /benchmark/'s model section, T184: the same; its page path, timed on the first load
        // before the rounds, has the GPU, and T205: the rounds' loads wait for that GPU to let go of its device)
        const outside = forward.external({ memory, base, size, kernels, spawn, gpu: hasWebGpu && !benchPage && !benching ? openGpu : undefined, gpuRoom,
          gpuRemembered: gpuRequest?.remembered, gpuForce, halfKeys, apart, oneThread: threadsRequest?.fixed === 1 });
        return { llama: builtOrLetGo(outside, () => llama2_numpy.Llama.callKwargs(null, tokenizer, { ...options, external: outside })), outside };
      },
      destroy() {},
    };
  }
  weightsInPython();
  // T503: a model of blocks widened past what Python's 32-bit memory holds runs on the SIMD kernels alone (Saluki:
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

// T93: where the converter writes the checkpoint, piece by piece, straight into where the engine will read it (a
// Python buffer on the way would stay: Pyodide's memory never shrinks). sink is what the converter calls
// (llama2_convert.Writer), weights what sink.open() made (weightsBuffer), bytes its size; release() lets go of it once
// (T145: a conversion that failed after sink.open() left a Python buffer of the model's size behind with
// ?without=kernels). Out of convert() so that tests/worker-sink-check.ts can follow sink.open() to footprint().
export function checkpointSink(keep?: Keeper) {
  const into = {
    weights: undefined, bytes: 0,
    release() {
      into.weights?.destroy();
      into.weights = undefined;
    },
  } as { weights: Weights | undefined; bytes: number; release(): void; sink: Sink };
  into.sink = {
    // form: what lays out the checkpoint and sizes the forward pass besides the header (llama2_numpy.FORM, T115, T144)
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
