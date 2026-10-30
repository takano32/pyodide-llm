// forward.ts (T.93): one token's forward pass in JavaScript, on the SIMD kernels of kernels/*.ts, on a WebAssembly
// memory of its own that holds the weights. Python (llama2_numpy.Llama with external=) still reads the header, says
// where every tensor is, tokenizes, samples and runs the generation loop; this file does what the engine's kernel_forward()
// did until T.93 retired it, in the same order with the same kernels. What it saves is the Python between the
// kernel calls (about 100 per token): 1.15 to 1.26 times the speed on int8 (T.93, stage 1a).
//
// A plain ES module: the worker imports it, and so does Node (tests/smoke.mjs).
//
//   const kernels = compileKernels(plainBytes, relaxedBytes);   // simdkernel_plain.wasm, simdkernel_relaxed_plain.wasm
//   const { memory, base } = weightsMemory(size);              // then write the checkpoint at base
//   const outside = external({ memory, base, size, kernels }); // what Llama(external=) takes

// T.349: this file is the window of the forward pass: compileKernels(), external(), and every name it exported as one
// file. The rest is in the modules of forward/: engine.ts (createForward(): one model's forward pass), threads.ts (its
// software threads and the search for their number), gpuside.ts (its GPU's side), choice.ts (the GPU or the CPU, and
// its times), paths.ts (what /benchmark/ times of the page's path), memory.ts (the memory a model takes) and alone.ts
// (a model on the GPU alone). Each is a static import, as jobs.ts is: a bundler links them into
// one file (T.367.3).
import { BATCH } from "./jobs.ts";
export { BATCH };
import { GPU_END_MS, GPU_BLOCK, GPU_TOKENS, promptTimes, tokenTimes, PROMPTS_UNTIMED, PROMPTS_GPU, PROMPTS_CPU, gpuLine } from "./forward/choice.ts";
import { PATH_ROUNDS, timePrompts, timedCell, SEARCH_SECONDS, endSearch } from "./forward/paths.ts";
import { memory64, footprint, keysInHalf, needsWide, pastWide, automaticDtype, growMemory, weightsMemory, takesBlocks, ringPositions } from "./forward/memory.ts";
import { gpuWeightBytes, aloneTakes, gpuBytes, gpuHoles, placer, BOTH_ON_8, weightsPlace, layerWeightsOf,
  USAGE_UNKNOWN, cpuReadBytes, aloneVerdict, aloneHolds, OUTSIDE_VOCABULARY, gpuOnlyUnfit, gpuOnlyWeights, gpuOnlyPlan, blocksForm,
  GPU_ALONE } from "./forward/alone.ts";
import { createForward } from "./forward/engine.ts";
import { LAYER_KINDS } from "./forward/kinds.ts";
import { tableHole } from "./forward/perlayer.ts";
import type { Blocks, Compiled, Engine, Made } from "./forward/engine.ts";
import type { Plan, TensorName } from "./contracts/plan.ts";
import type { Rows } from "./forward/perlayer.ts";
// (T.368.4) the types of what this window hands out, for the worker, which reads the module's own through
// `import type * as Forward from "../forward.ts"`
export type { Blocks, Compiled, Engine, Made, Trying } from "./forward/engine.ts";
export type { Core, Kind, Kinds, Layers, Room, Step } from "./forward/kinds.ts";
export type { Cache } from "./forward/cache.ts";
export type { Footprint, Header, Memory } from "./forward/memory.ts";
export type { Adapter, Direct, GpuForm, GpuOnlyWeights, Hole, Place } from "./forward/alone.ts";
export type { Chosen, GpuWorker, Side } from "./forward/gpuside.ts";
export type { Spawn, Thread } from "./forward/threads.ts";
export type { Rows } from "./forward/perlayer.ts";
// What Python's Llama(external=) takes (external()): engine is there once Python called start(plan). The plan comes as
// Python's dict (a PyProxy, which toJs() makes an object of) or, from the tests, as the object
export type Outside = { size: number; read(offset: number, length: number): Uint8Array; start(plan: Handed): Engine; engine?: Engine };
type Handed = Plan & { toJs?(options: { dict_converter: typeof Object.fromEntries }): Plan };
export { GPU_END_MS, GPU_BLOCK, GPU_TOKENS, promptTimes, tokenTimes, PROMPTS_UNTIMED, PROMPTS_GPU, PROMPTS_CPU, gpuLine,
  PATH_ROUNDS, timePrompts, timedCell, SEARCH_SECONDS, endSearch, memory64, footprint, keysInHalf, gpuWeightBytes,
  aloneTakes, gpuBytes, gpuHoles, placer, BOTH_ON_8, weightsPlace, layerWeightsOf, USAGE_UNKNOWN, cpuReadBytes,
  aloneVerdict, aloneHolds, OUTSIDE_VOCABULARY, gpuOnlyUnfit, gpuOnlyWeights, gpuOnlyPlan, blocksForm, needsWide, pastWide,
  automaticDtype, growMemory, weightsMemory, createForward, LAYER_KINDS, takesBlocks, tableHole, ringPositions };
// T.329.2.2: a model whose table of every layer's inputs is not in the memory (tableHole(): one stretch of the
// checkpoint), which holds the rest packed without it (stored bytes), a tensor at place(its offset in the checkpoint);
// rows reads the table's from where the checkpoint is kept (perlayer.ts's Rows)
export type Apart = { place(offset: number): number; stored: number; rows: Rows };

/** The kernels as WebAssembly modules. The relaxed one fails to compile where relaxed SIMD is missing (Safari):
 * then int8 runs on matmul_q8. wide (T.101): the build for a 64-bit memory (simdkernel_*64.wasm). */
export function compileKernels(plain: BufferSource, relaxed?: BufferSource | null, wide = false): Compiled {
  let relaxedModule: WebAssembly.Module | null = null;
  try {
    relaxedModule = relaxed ? new WebAssembly.Module(relaxed) : null;
  } catch {
    relaxedModule = null;
  }
  return { plain: new WebAssembly.Module(plain), relaxed: relaxedModule, wide };
}

/** T.363.5: the kernels of llama.cpp's blocks of one build (blockkernel_<kind>.wasm, blockkernel_relaxed_<kind>.wasm) as
 * modules: what a Compiled of the same build has as its blocks, for a model with such rows. The relaxed one is null
 * where relaxed SIMD is missing, as compileKernels()'s. */
export function compileBlocks(plain: BufferSource, relaxed?: BufferSource | null): Blocks {
  const compiled = compileKernels(plain, relaxed);
  return { plain: compiled.plain, relaxed: compiled.relaxed };
}

/** What Llama(external=) takes: the size of the checkpoint, read() for the few bytes Python looks at itself, and
 * start(plan), which builds the forward pass. */
/** T.156 direct ({ place, stored, onLost }, gpuOnlyWeights()): the layers' matrices are on the GPU alone, memory holds
 * the rest of the checkpoint (stored bytes) packed without them, a tensor at place(its offset in the checkpoint);
 * onLost(why) is told where the GPU stops (the model has to be loaded again on the CPU). */
export function external({ memory, base, size, kernels, spawn, gpu, gpuRoom, gpuRemembered, gpuForce, halfKeys, direct, trying, apart, oneThread }:
  Omit<Made, "plan" | "wrap" | "stalledMs" | "clock" | "kinds" | "rows"> & { apart?: Apart }) {
  const place = direct?.place ?? apart?.place ?? ((offset: number) => offset);
  const outside: Outside = {
    size,
    read: (offset, length) => new Uint8Array(memory.buffer, base + place(offset), length).slice(),
    start: (plan) => {
      plan = plan.toJs ? plan.toJs({ dict_converter: Object.fromEntries }) : plan;
      // (T.156: every tensor but those on the GPU alone where it is in memory; those keep their offsets in the
      // checkpoint, which only the GPU's worker reads)
      if (direct) {
        plan.tensors = Object.fromEntries(Object.entries(plan.tensors).map(([name, t]) => [name, GPU_ALONE.includes(name as TensorName) ? t
          : { ...t, offset: place(t.offset), ...(t.scales ? { scales: place(t.scales) } : {}) }] as const)) as Plan["tensors"];
      }
      // (T.329.2.2: every tensor where it is in memory; the table of every layer's inputs is the hole, whose offsets
      // place() leaves as they are in the checkpoint, which only apart.rows reads)
      if (apart) {
        plan.tensors = Object.fromEntries(Object.entries(plan.tensors).map(([name, t]) => [name, { ...t, offset: place(t.offset), ...(t.scales ? { scales: place(t.scales) } : {}),
          ...(t.layers ? { layers: t.layers.map((part) => ({ ...part, offset: place(part.offset), ...(part.scales ? { scales: place(part.scales) } : {}) })) } : {}) }] as const)) as Plan["tensors"];
      }
      outside.engine = createForward({ memory, base, size: direct ? direct.stored : apart ? apart.stored : size, kernels, spawn, gpu, gpuRoom,
        gpuRemembered, gpuForce, halfKeys, direct, trying, plan, rows: apart?.rows, oneThread });
      return outside.engine;
    },
  };
  return outside;
}
