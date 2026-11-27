// worker/alone.ts (T641.1, out of worker/weights.ts): a model on the GPU alone (T156): whether it may go there, the
// GPU's worker and the memory it is placed in, and what is told once it is built; the models back on the CPU this
// visit; and the workers an engine starts (its software threads, its GPU's worker) and lets go of where it fails as it
// is built. It holds the model on the GPU alone and the key of the model loading; the others say what happened
// (aloneBegan, onCpuFromNow) and ask (gpuOnlyLost, weightsRoom, weightsDrained, gpuOnlyReady).
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T367.3).

import { llama2_numpy, forwardModule, sharedKernels, wideKernels, gpuForce, benchPage, gpuAdapter, hasWebGpu } from "./runtime.ts";
import { gpuRequest, benching, switchesNow } from "./request.ts";
import { workerWentWithout } from "./outdated.ts";
import { pooledWeights, afterCheckpoint, sharedWanted, placesOf, pooled } from "./pool.ts";
import { told } from "../failure.ts";
import type { Keeper, Weights } from "./weights.ts";
import type { Dtype } from "../contracts/form.ts";
import type { ToGpu } from "../contracts/gpu/protocol.ts";
import type { GpuAlone, Listed } from "../contracts/messages.ts";
import type { Llama, Options } from "../contracts/python.ts";
import type { Direct, Engine, GpuOnlyWeights, Header, Outside, Place, Rows, Thread } from "../forward.ts";
import type { Reader } from "../kept.ts";
import type { Tensors } from "../contracts/plan.ts";

// T156: a model on the GPU alone: what forward.ts reads of it (Direct), its GPU's worker, why the GPU stopped under
// it (or null), where the loops that write its weights wait (room) and its builder (drained), and end(): the worker
// stopped, resolved once it said so. T491, gone(): why the worker went before the engine was built (an error, or
// nothing taken for 60 s: forward.ts's gpuOnlyWeights), or undefined
export type GpuOnly = Direct & { worker: Worker; lost: string | null; room: GpuOnlyWeights["room"]; drained: GpuOnlyWeights["drained"]; end(): Promise<void>;
  gone(): string | undefined };
// where a model goes (gpuOnlyWeightsFor()): forward.ts's place and, where it may go on the GPU alone, its tensors,
// the bytes kept in the memory then and those with the forward pass after them
export type Placed = Place & { tensors?: Tensors; stored?: number; gpuOnly?: number };
// T329.2.2, T334.3: a table of every layer's inputs that stays where the model is kept (weights.ts's tableInFile()): its
// stretch of the checkpoint (hole), what reads its rows into a memory (rows), a kept file's handle handed in (attach),
// and the engine built on it (built)
export type TableApart = { hole: [number, number]; rows(memory: WebAssembly.Memory): Rows; attach(reader: Reader): void; built(): void };

// T156: the model on the GPU alone that is loading or loaded now (what gpuOnlyBuffer() made), or undefined
let gpuOnlyNow: GpuOnly | undefined;
// the GPU's worker of a model on the GPU alone let go before its engine was built (gpuOnlyBuffer()'s destroy()), or
// (T461) of an engine whose Llama threw as it was built: the next load waits for it, as release() waits for a built
// one's (T205)
export let gpuOnlyEnding = Promise.resolve();
// T461: a Llama that threw as it was built (Python's checks of the checkpoint and its options, after start()) lets go
// of the engine its external() built by then (the software threads, the GPU's worker): no model holds it, and nothing
// else would. The error goes on as it was
export function builtOrLetGo(outside: Outside, build: () => Llama) {
  try {
    return build();
  } catch (thrown) {
    const engine = outside.engine;
    if (engine) {
      // (T461's review: a release() that throws does not hide what Python said)
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
export let loadingKey: string | undefined;
// T156: the models whose GPU failed while they were on it alone, loaded again on the CPU from then on (this visit)
const cpuOnly = new Set<string | undefined>();
export const modelKey = (model: Listed) => model.hf ? `hf:${model.hf.repo}@${model.hf.revision}` : model.id ?? model.name;
/** T641.1: a load begins: of this model, and none is on the GPU alone yet (loadBegan()) */
export function aloneBegan(model: Listed) {
  loadingKey = modelKey(model);
  gpuOnlyNow = undefined;
}
/** T156: the GPU stopped under this model while it was on the GPU alone: on the CPU from now on, this visit */
export function onCpuFromNow(model: Listed) {
  cpuOnly.add(modelKey(model));
}
/** T669: whether the model loaded now is on the GPU alone (no CPU of its own to draw a step on) */
export const onGpuAlone = () => gpuOnlyNow !== undefined;
/** T156: why the GPU stopped under the model on the GPU alone, or undefined (no such model, or its GPU goes on) */
export const gpuOnlyLost = () => gpuOnlyNow?.lost;

// a software thread of forward.ts (stage 2): a module worker of its own, ready once its kernels are warm
export const spawnThread = (data: unknown) => new Promise<Thread>((resolve, reject) => {
  const worker = new Worker(new URL("../helper.ts", import.meta.url), { type: "module" });
  worker.onmessage = () => resolve({ terminate: () => worker.terminate() });
  worker.onerror = (event) => {
    void workerWentWithout(event, import.meta.url);  // (T439: fewer threads, and the page is told where that is the site's update)
    reject(new Error(event.message ?? "a software thread did not start"));
  };
  worker.postMessage(data);
});

// T135: the GPU's worker of forward.ts, a module worker of its own; forward.ts starts it
// (T439: forward.ts holds its error, and the model stays on the CPU; beside that the page is told where it is the site's update)
export function openGpu() {
  const worker = new Worker(new URL("../gpu.ts", import.meta.url), { type: "module" });
  worker.addEventListener("error", (event) => void workerWentWithout(event, import.meta.url));
  return worker;
}

// T156: what a model on the GPU alone needs of this page and device, whatever the model (gpuOnlyUnfit has the rest): a
// memory the device says (T205), the GPU's worker (not the benchmark's rounds), a shared memory and the 64-bit kernels,
// the int8 kernels, an adapter that is no fallback (but for the tests), and no failure of the GPU under this model before
// (cpuOnly). convert() asks it too before it opens a file to keep a conversion in as it comes (the review of T156)
// (the owner, 2026-09-27: and no verdict the page kept that the CPU is faster here: then the CPU at once)
// (T375.5: and a dtype whose packing may be there at all, forward.ts's aloneTakes)
export const gpuOnlyPossible = (dtype: Dtype) => gpuOnlyAllowed() && forwardModule!.aloneTakes(dtype) && !aloneKept(dtype);
const gpuOnlyAllowed = () => hasWebGpu && self.navigator?.deviceMemory !== undefined && !benchPage && !benching && sharedWanted() &&
  Boolean(wideKernels?.shared) && !switchesNow().includes("int8") && !cpuOnly.has(loadingKey) &&
  Boolean(gpuAdapter) && (!gpuAdapter!.fallback || Boolean(gpuForce.fallback));
// T156: whether the page kept that the CPU was faster here than this model on the GPU alone (forward.ts's aloneHolds)
// (dtype, T232: the key is the one gpu.ts gave the verdict, a ternary model's its own)
// (T547.3.7: a model whose rows have kinds of their own, Saluki's blocks, by the key of a model of blocks; T634.3: a
// binary model's the ternary one's, whose shaders multiply its weights; T641.9: forward.ts's keyKindOf, by PACKED's gpu.as)
const aloneKept = (dtype: Dtype, blocks = false) => Boolean(gpuAdapter) &&
  forwardModule!.aloneHolds(gpuRequest?.remembered?.alone, gpuAdapter!.keys?.[forwardModule!.keyKindOf(dtype, blocks)], gpuRequest?.cpu);
// T156: where a model goes ({ mode: "both" | "gpu" | "cpu", gpuRoom }, forward.ts's weightsPlace): the GPU alone only
// for a Llama of int8 (T232: or of ternary weights) the GPU's steps take (gpuOnlyUnfit), where the page and the device
// may (gpuOnlyPossible)
export function gpuOnlyWeightsFor(size: number, header: Header, options: Options, after: number, deviceMemory: number): Placed {
  const forward = forwardModule!;  // (T476: the forward module, which a load that places weights has: taken once)
  const { dtype = "float32" } = options;
  const cpu = size + after, gpu = forward.gpuBytes(header, { ...options, dtype });
  // (T232: with the form's rotated basis, which the GPU does not turn the inputs for (T237): the worker did not hand it
  // on, and a Llama in a rotated basis would have gone on the GPU alone to be loaded again on the CPU once built.
  // T375.5: the whole form as the options carry it: the layers that keep a state are read off it too)
  const fit = gpuOnlyAllowed() && !forward.gpuOnlyUnfit(header, dtype, options, gpuAdapter, gpuForce);
  const eligible = fit && !aloneKept(dtype, forward.blocksForm(options));
  // (the second review of T156: the status line then says only the memory's reason where the layers do not fit beside
  // the CPU's copy; the console says the kept verdict, and what asks again)
  if (fit && !eligible) {
    console.info("gpu: the CPU as /benchmark/ measured it was faster here than this model on the GPU alone, as the page kept it: " +
      "on the CPU (a new run of /benchmark/'s CPU section, another browser version or new shaders weigh the two again)");
  }
  if (!eligible) {
    const place = forward.weightsPlace({ cpu, gpu, deviceMemory });
    // (T334.2's review: where the GPU would take the model but not both copies, the numbers the room was weighed by,
    // a line then alone: the status line says the GPU's bytes where they do not fit, not the room. Normal visits of
    // a model held twice, or of a page with no GPU, say nothing)
    if (place.mode === "cpu" && gpuAdapter && !benchPage) said(cpu, gpu, place.gpuRoom!, deviceMemory, `the GPU's room beside the CPU's is ${mb(place.gpuRoom!)} MB`);
    return place;
  }
  // (T571: of the whole form as the options carry it, as gpuOnlyUnfit() is asked above: four keys of it went before, and a
  // Qwen3.5 on the GPU alone, whose linear-attention layers are in the form's linear, was refused by Python)
  const tensors = placesOf(header, dtype, options), stored = size - forward.gpuHoles(tensors).reduce((sum, [a, b]) => sum + b - a, 0);
  const gpuOnly = stored + afterCheckpoint(header, stored, { ...options, direct: true }, true);
  const place = forward.weightsPlace({ cpu, gpuOnly, gpu, deviceMemory, eligible, forced: gpuForce.only });
  // (T682's review: and where the GPU alone takes it for that, the same numbers, a line too: the visit of a model too
  // large to hold twice, as the line above. A test that asks for the GPU alone (gpuForce.only) weighs no room: no line)
  if (place.mode === "gpu" && place.gpuRoom !== undefined) said(cpu, gpu, place.gpuRoom, deviceMemory, "on the GPU alone");
  return { ...place, tensors, stored, gpuOnly };
}
const mb = (bytes: number) => Math.round(bytes / 1e6);
// (T334.2's review, T682's: the numbers a model's place was weighed by, where both copies do not fit: the CPU's bytes,
// the GPU's, the room of the device, and where the model goes)
const said = (cpu: number, gpu: number, gpuRoom: number, deviceMemory: number, then: string) =>
  console.info(`gpu: both copies would not fit: the CPU's ${mb(cpu)} MB and the GPU's ${mb(gpu)} MB, in ${mb(cpu + gpuRoom)} MB of a device ` +
    `that says ${deviceMemory} GB: ${then}`);
// T156: the weights of a model on the GPU alone: the layers' matrices to the GPU's worker as they come (forward.ts's
// gpuOnlyWeights), the rest into a memory of their size, and every byte to keep's file where the conversion is kept
// (T334.3, table: a table of every layer's inputs that stays where the model is kept, neither in the memory nor on the
// GPU: its stretch is apart from both, and the engine reads its rows from there)
export function gpuOnlyBuffer(size: number, header: Header, options: Options, { tensors, stored, gpuOnly }: Required<Placed>, keep?: Keeper, table?: TableApart): Weights {
  const forward = forwardModule!;  // (T476: the forward module, which a load that places weights has: taken once)
  // (T576: the GPU's plan first: what makes it may throw (gpuPacking()), before a memory is taken or a worker opened)
  const plan = forward.gpuOnlyPlan(header, tensors, gpuForce, gpuRequest?.remembered, options);
  // (T581: and what else is reckoned of the header alone, which may throw as well)
  const layerWeights = forward.layerWeightsOf(header, options);
  const after = gpuOnly - stored, wide = forward.needsWide(stored, after);
  const { memory, base, shared } = pooledWeights(stored, after, true, wide);
  if (!shared) throw new Error("This browser gave no shared memory for a model on the GPU alone.");
  const worker = openGpu();
  // (forward.ts listens from start() on. Here only "ended", beside it: T205, the next load after the GPU's worker let
  // go of its device and its buffers, which it makes for every layer as it opens, before a byte comes)
  let ended: () => void;
  const ending = new Promise<void>((resolve) => { ended = resolve; });
  worker.addEventListener("message", ({ data }) => data?.type === "ended" && ended());
  worker.addEventListener("error", () => ended());
  // the GPU's worker stopped, and once it said "ended" (or after GPU_END_MS: terminated), resolved; as release() does
  const end = () => {
    worker.postMessage({ type: "stop" });
    let timer: ReturnType<typeof setTimeout>;
    const late = new Promise<void>((resolve) => { timer = setTimeout(resolve, forward.GPU_END_MS); });
    return Promise.race([ending, late]).then(() => {
      clearTimeout(timer);
      worker.terminate();
    });
  };
  // (T581: a worker opened is let go of at once where what writes to it cannot be made or it cannot be told to open:
  // it makes its device and buffers once it opens, from a plan it was handed whole)
  let weights: GpuOnlyWeights;
  // T377: a worker that says "error" before the engine is built (its script did not arrive, or it threw) takes no more
  // of the weights and counts none. T491: the loops that write throw why at their next wait for it, and the load gives
  // up there (gpuOnlyGaveUp()). An error past their last wait, before the engine is built, is handed to the engine,
  // which listens from start() on, once it is built (failed)
  let failed: ErrorEvent | undefined;
  try {
    weights = forward.gpuOnlyWeights({ memory, base, size, tensors, worker, apart: table?.hole });
    worker.addEventListener("error", (event) => {
      failed ??= event;
      weights.gone(event.message);
    });
    worker.postMessage({ type: "open", plan, flow: weights.flow } satisfies ToGpu);
  } catch (error) {
    worker.terminate();
    throw error;
  }
  // (T156: the checkpoint's size, its layers' multiply-adds, /benchmark/'s CPU reading and the page's use, for the
  // estimate the GPU is held against: forward.ts's aloneVerdict. T682's review: the size without a Gemma 4's table of
  // every layer's inputs, of which a token reads one row, wherever the table is: the CPU's token was reckoned to read
  // the E2B's 2.6 GB of it and the E4B's 3.2 GB, 2.0 and 1.6 times what it reads)
  const tableOf = forward.tableHole(tensors), reads = size - (tableOf ? tableOf[1] - tableOf[0] : 0);
  let built = false;
  const direct: GpuOnly = gpuOnlyNow = { worker, lost: null, stored, size: reads, place: weights.place,
    layerWeights, cpu: gpuRequest?.cpu, usage: gpuRequest?.usage ?? forward.USAGE_UNKNOWN,
    room: weights.room, drained: weights.drained, onLost: (why) => { direct.lost = why; }, end, gone: () => (built ? undefined : weights.lost()) };
  pooled(memory);
  const kernels = (wide ? wideKernels!.shared : sharedKernels)!;
  return {
    direct,
    hole: table?.hole ?? null,
    attach: (reader) => table?.attach(reader),
    write(offset, chunk) {
      weights.write(offset, chunk);
      keep?.write(offset, chunk);
    },
    room: weights.room,
    drained: weights.drained,  // every byte of the layers on the GPU (or the GPU failed: start() says so)
    llama: (tokenizer, engineOptions) => {
      // (T334.3: the table's rows read from where the model is kept, which the engine reads them by)
      table?.built();
      if (table) direct.rows = table.rows(memory);
      // (no oneThread, T371: a model on the GPU alone is one on a shared memory, which ?threads=1 does not ask for)
      const outside = forward.external({ memory, base, size, kernels, spawn: spawnThread, gpu: () => worker, gpuForce, halfKeys: true, direct,
        gpuRemembered: gpuRequest?.remembered });
      const made = builtOrLetGo(outside, () => llama2_numpy.Llama.callKwargs(null, tokenizer, { ...engineOptions, external: outside }));
      built = true;
      // (T377: the engine's side of the GPU says why, as of a worker that failed under it, and the model is loaded
      // again on the CPU: gpuOnlyReady(). An engine that started no worker has said its own reason already)
      if (failed) worker.onerror?.(failed);
      return { llama: made, outside };
    },
    // (the review of T156: a load cancelled or failed before its engine was built, or another try of the converter's
    // (sink.open() again), let go of the GPU's worker: nothing else would, and it held the device and a buffer for
    // every layer's matrices, 4.1 GB for Llama 3.2 3B, for the rest of the visit. The next load waits for it)
    destroy() {
      if (!built) gpuOnlyEnding = Promise.all([gpuOnlyEnding, end()]).then(() => {});  // (T461: after a release() too)
    },
  };
}

// T156: where the loops that write the weights can wait for the GPU's worker of a model on the GPU alone (room), and
// before its engine is built (drained)
export const weightsRoom = () => gpuOnlyNow?.room?.();
export const weightsDrained = () => gpuOnlyNow?.drained?.();
// T156: after a model on the GPU alone is built: its GPU ready (true), or failed (false: the worker let go of it, and
// the model goes on the CPU from now on this visit)
// engine: forward.ts's engine of the model just built (loaded.ts's engineNow())
export async function gpuOnlyReady(model: Listed, id: number, engine: Engine | undefined) {
  const direct = gpuOnlyNow;
  if (!direct || engine === undefined) return true;
  await engine.gpu;
  if (!direct.lost) return true;
  onCpu(model, direct.lost);
  // (the CPU was faster: the page keeps it for this model and device once the load on the CPU is ready, and the next
  // visit loads on the CPU at once. The load's id: the page reads no word of a load another choice cancelled, the
  // second review of T156: a verdict that came late was kept for the model chosen since)
  if (direct.verdict) postMessage({ type: "gpu-alone", load: id, alone: direct.verdict } satisfies GpuAlone);
  // (the GPU's worker let go of its device: stopped by forward.ts where it started it, and here where it did not, as a
  // model whose GPU forward.ts gave up as the engine was built, before start(): its release() has no "ended" to wait
  // for then. The review of T156: the load on the CPU begins after it, T205)
  await direct.end();
  gpuOnlyNow = undefined;
  return false;
}
/** T491 (the owner, 2026-11-26): after a load that failed: whether it failed because the GPU's worker of the model on
 * the GPU alone went before its engine was built (it said "error", or took nothing for 60 s), which its loops that
 * write threw at once: the model goes on the CPU from now on this visit, and is loaded again there (true). The worker
 * was let go of as the load ended (gpuOnlyBuffer()'s destroy()): the next load waits for it (gpuOnlyEnding). Any other
 * failure is the load's own (false) */
export function gpuOnlyGaveUp(model: Listed) {
  const why = gpuOnlyNow?.gone();
  if (!why) return false;
  onCpu(model, why);
  gpuOnlyNow = undefined;
  return true;
}
/** T491: the model loading now was given up on the GPU alone (gpuOnlyGaveUp()'s question, before the load ends: a
 * kept model's read, whose failures are otherwise read as a kept file that could not be read) */
export const gpuOnlyGone = () => Boolean(gpuOnlyNow?.gone());
// (said, and on the CPU from now on this visit)
function onCpu(model: Listed, why: string) {
  console.warn(`gpu: ${why}: the model was on the GPU alone, and is loaded again on the CPU`);
  cpuOnly.add(modelKey(model));
}
