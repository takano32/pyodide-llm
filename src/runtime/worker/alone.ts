// worker/alone.ts (T.641.1, out of worker/weights.ts): a model on the GPU alone (T.156): whether it may go there, the
// GPU's worker and the memory it is placed in, and what is told once it is built; the models back on the CPU this
// visit; and the workers an engine starts (its software threads, its GPU's worker) and lets go of where it fails as it
// is built. It holds the model on the GPU alone and the key of the model loading; the others say what happened
// (aloneBegan, onCpuFromNow) and ask (gpuOnlyLost, weightsRoom, weightsDrained, gpuOnlyReady).
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T.367.3).

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
import type { Direct, Engine, GpuOnlyWeights, Header, Outside, Place, Thread } from "../forward.ts";
import type { Tensors } from "../contracts/plan.ts";

// T.156: a model on the GPU alone: what forward.ts reads of it (Direct), its GPU's worker, why the GPU stopped under
// it (or null), where the loops that write its weights wait (room) and its builder (drained), and end(): the worker
// stopped, resolved once it said so
export type GpuOnly = Direct & { worker: Worker; lost: string | null; room: GpuOnlyWeights["room"]; drained: GpuOnlyWeights["drained"]; end(): Promise<void> };
// where a model goes (gpuOnlyWeightsFor()): forward.ts's place and, where it may go on the GPU alone, its tensors,
// the bytes kept in the memory then and those with the forward pass after them
export type Placed = Place & { tensors?: Tensors; stored?: number; gpuOnly?: number };

// T.156: the model on the GPU alone that is loading or loaded now (what gpuOnlyBuffer() made), or undefined
let gpuOnlyNow: GpuOnly | undefined;
// the GPU's worker of a model on the GPU alone let go before its engine was built (gpuOnlyBuffer()'s destroy()), or
// (T.461) of an engine whose Llama threw as it was built: the next load waits for it, as release() waits for a built
// one's (T.205)
export let gpuOnlyEnding = Promise.resolve();
// T.461: a Llama that threw as it was built (Python's checks of the checkpoint and its options, after start()) lets go
// of the engine its external() built by then (the software threads, the GPU's worker): no model holds it, and nothing
// else would. The error goes on as it was
export function builtOrLetGo(outside: Outside, build: () => Llama) {
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
export let loadingKey: string | undefined;
// T.156: the models whose GPU failed while they were on it alone, loaded again on the CPU from then on (this visit)
const cpuOnly = new Set<string | undefined>();
export const modelKey = (model: Listed) => model.hf ? `hf:${model.hf.repo}@${model.hf.revision}` : model.id ?? model.name;
/** T.641.1: a load begins: of this model, and none is on the GPU alone yet (loadBegan()) */
export function aloneBegan(model: Listed) {
  loadingKey = modelKey(model);
  gpuOnlyNow = undefined;
}
/** T.156: the GPU stopped under this model while it was on the GPU alone: on the CPU from now on, this visit */
export function onCpuFromNow(model: Listed) {
  cpuOnly.add(modelKey(model));
}
/** T.156: why the GPU stopped under the model on the GPU alone, or undefined (no such model, or its GPU goes on) */
export const gpuOnlyLost = () => gpuOnlyNow?.lost;

// a software thread of forward.ts (stage 2): a module worker of its own, ready once its kernels are warm
export const spawnThread = (data: unknown) => new Promise<Thread>((resolve, reject) => {
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
export function openGpu() {
  const worker = new Worker(new URL("../gpu.ts", import.meta.url), { type: "module" });
  worker.addEventListener("error", (event) => void workerWentWithout(event, import.meta.url));
  return worker;
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
// (T.547.3.7: a model whose rows have kinds of their own, Saluki's blocks, by the key of a model of blocks; T.634.3: a
// binary model's the ternary one's, whose shaders multiply its weights)
const aloneKept = (dtype: Dtype, blocks = false) => Boolean(gpuAdapter) &&
  forwardModule!.aloneHolds(gpuRequest?.remembered?.alone, gpuAdapter!.keys?.[dtype === "ternary" || dtype === "binary" ? "ternary" : blocks ? "blocks" : "plain"], gpuRequest?.cpu);
// T.156: where a model goes ({ mode: "both" | "gpu" | "cpu", gpuRoom }, forward.ts's weightsPlace): the GPU alone only
// for a Llama of int8 (T.232: or of ternary weights) the GPU's steps take (gpuOnlyUnfit), where the page and the device
// may (gpuOnlyPossible)
export function gpuOnlyWeightsFor(size: number, header: Header, options: Options, after: number, deviceMemory: number): Placed {
  const forward = forwardModule!;  // (T.476: the forward module, which a load that places weights has: taken once)
  const { dtype = "float32" } = options;
  const cpu = size + after, gpu = forward.gpuBytes(header, { ...options, dtype });
  // (T.232: with the form's rotated basis, which the GPU does not turn the inputs for (T.237): the worker did not hand it
  // on, and a Llama in a rotated basis would have gone on the GPU alone to be loaded again on the CPU once built.
  // T.375.5: the whole form as the options carry it: the layers that keep a state are read off it too)
  const fit = gpuOnlyAllowed() && !forward.gpuOnlyUnfit(header, dtype, options, gpuAdapter, gpuForce);
  const eligible = fit && !aloneKept(dtype, forward.blocksForm(options));
  // (the second review of T.156: the status line then says only the memory's reason where the layers do not fit beside
  // the CPU's copy; the console says the kept verdict, and what asks again)
  if (fit && !eligible) {
    console.info("gpu: the CPU as /benchmark/ measured it was faster here than this model on the GPU alone, as the page kept it: " +
      "on the CPU (a new run of /benchmark/'s CPU section, another browser version or new shaders weigh the two again)");
  }
  if (!eligible) return forward.weightsPlace({ cpu, gpu, deviceMemory });
  // (T.571: of the whole form as the options carry it, as gpuOnlyUnfit() is asked above: four keys of it went before, and a
  // Qwen3.5 on the GPU alone, whose linear-attention layers are in the form's linear, was refused by Python)
  const tensors = placesOf(header, dtype, options), stored = size - forward.gpuHoles(tensors).reduce((sum, [a, b]) => sum + b - a, 0);
  const gpuOnly = stored + afterCheckpoint(header, stored, { ...options, direct: true }, true);
  return { ...forward.weightsPlace({ cpu, gpuOnly, gpu, deviceMemory, eligible, forced: gpuForce.only }), tensors, stored, gpuOnly };
}
// T.156: the weights of a model on the GPU alone: the layers' matrices to the GPU's worker as they come (forward.ts's
// gpuOnlyWeights), the rest into a memory of their size, and every byte to keep's file where the conversion is kept
export function gpuOnlyBuffer(size: number, header: Header, options: Options, { tensors, stored, gpuOnly }: Required<Placed>, keep?: Keeper): Weights {
  const forward = forwardModule!;  // (T.476: the forward module, which a load that places weights has: taken once)
  // (T.576: the GPU's plan first: what makes it may throw (gpuPacking()), before a memory is taken or a worker opened)
  const plan = forward.gpuOnlyPlan(header, tensors, gpuForce, gpuRequest?.remembered, options);
  // (T.581: and what else is reckoned of the header alone, which may throw as well)
  const layerWeights = forward.layerWeightsOf(header, options);
  const after = gpuOnly - stored, wide = forward.needsWide(stored, after);
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
    const late = new Promise<void>((resolve) => { timer = setTimeout(resolve, forward.GPU_END_MS); });
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
    weights = forward.gpuOnlyWeights({ memory, base, size, tensors, worker });
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
    layerWeights, cpu: gpuRequest?.cpu, usage: gpuRequest?.usage ?? forward.USAGE_UNKNOWN,
    room: weights.room, drained: weights.drained, onLost: (why) => { direct.lost = why; }, end };
  pooled(memory);
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
      const outside = forward.external({ memory, base, size, kernels, spawn: spawnThread, gpu: () => worker, gpuForce, halfKeys: true, direct,
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
