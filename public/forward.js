// forward.js (T93): one token's forward pass in JavaScript, on the SIMD kernels of kernels/*.ts, on a WebAssembly
// memory of its own that holds the weights. Python (llama2_numpy.Llama with external=) still reads the header, says
// where every tensor is, tokenizes, samples and runs the generation loop; this file does what the engine's kernel_forward()
// did until T93 retired it, in the same order with the same kernels. What it saves is the Python between the
// kernel calls (about 100 per token): 1.15 to 1.26 times the speed on int8 (T93, stage 1a).
//
// A plain ES module: the worker imports it, and so does Node (tests/smoke.mjs).
//
//   const kernels = compileKernels(plainBytes, relaxedBytes);   // simdkernel_plain.wasm, simdkernel_relaxed_plain.wasm
//   const { memory, base } = weightsMemory(size);              // then write the checkpoint at base
//   const outside = external({ memory, base, size, kernels }); // what Llama(external=) takes

// T349: this file is the window of the forward pass: compileKernels(), external(), and every name it exported as one
// file. The rest is in the modules of forward/: engine.js (createForward(): one model's forward pass), threads.js (its
// software threads and the search for their number), gpuside.js (its GPU's side), choice.js (the GPU or the CPU, and
// its times), paths.js (what /benchmark/ times of the page's path), memory.js (the memory a model takes) and alone.js
// (a model on the GPU alone). Each is asked for with this file's own ?v=<build>, as jobs.js is, so that all come from
// one deployment, and all at once: one after another's end would add a round trip for each.
const modules = Object.fromEntries(["choice", "paths", "memory", "alone", "threads", "gpuside", "engine"].map((name) =>
  [name, import(new URL(`forward/${name}.js${new URL(import.meta.url).search}`, import.meta.url))]));
const { BATCH } = await import(new URL(`jobs.js${new URL(import.meta.url).search}`, import.meta.url));
export { BATCH };
const { GPU_END_MS, GPU_BLOCK, GPU_TOKENS, promptTimes, tokenTimes, PROMPTS_UNTIMED, PROMPTS_GPU, PROMPTS_CPU, gpuLine } = await modules.choice;
const { PATH_ROUNDS, timePrompts, timedCell, SEARCH_SECONDS, endSearch } = await modules.paths;
const { memory64, footprint, keysInHalf, needsWide, pastWide, automaticDtype, growMemory, weightsMemory } = await modules.memory;
const { GPU_WEIGHT_BYTES, GPU_TERNARY_BYTES, gpuBytes, gpuHoles, placer, BOTH_ON_8, weightsPlace, layerWeightsOf,
  USAGE_UNKNOWN, cpuReadBytes, aloneVerdict, aloneHolds, OUTSIDE_VOCABULARY, gpuOnlyUnfit, gpuOnlyWeights, gpuOnlyPlan,
  GPU_ALONE } = await modules.alone;
const { createForward } = await modules.engine;
export { GPU_END_MS, GPU_BLOCK, GPU_TOKENS, promptTimes, tokenTimes, PROMPTS_UNTIMED, PROMPTS_GPU, PROMPTS_CPU, gpuLine,
  PATH_ROUNDS, timePrompts, timedCell, SEARCH_SECONDS, endSearch, memory64, footprint, keysInHalf, GPU_WEIGHT_BYTES,
  GPU_TERNARY_BYTES, gpuBytes, gpuHoles, placer, BOTH_ON_8, weightsPlace, layerWeightsOf, USAGE_UNKNOWN, cpuReadBytes,
  aloneVerdict, aloneHolds, OUTSIDE_VOCABULARY, gpuOnlyUnfit, gpuOnlyWeights, gpuOnlyPlan, needsWide, pastWide,
  automaticDtype, growMemory, weightsMemory, createForward };

/** The kernels as WebAssembly modules. The relaxed one fails to compile where relaxed SIMD is missing (Safari):
 * then int8 runs on matmul_q8. wide (T101): the build for a 64-bit memory (simdkernel_*64.wasm). */
export function compileKernels(plain, relaxed, wide = false) {
  let relaxedModule = null;
  try {
    relaxedModule = relaxed ? new WebAssembly.Module(relaxed) : null;
  } catch {
    relaxedModule = null;
  }
  return { plain: new WebAssembly.Module(plain), relaxed: relaxedModule, wide };
}

/** What Llama(external=) takes: the size of the checkpoint, read() for the few bytes Python looks at itself, and
 * start(plan), which builds the forward pass. */
/** T156 direct ({ place, stored, onLost }, gpuOnlyWeights()): the layers' matrices are on the GPU alone, memory holds
 * the rest of the checkpoint (stored bytes) packed without them, a tensor at place(its offset in the checkpoint);
 * onLost(why) is told where the GPU stops (the model has to be loaded again on the CPU). */
export function external({ memory, base, size, kernels, spawn, gpu, gpuRoom, memoryUnsaid, gpuRemembered, gpuForce, halfKeys, direct }) {
  const place = direct?.place ?? ((offset) => offset);
  const outside = {
    size,
    read: (offset, length) => new Uint8Array(memory.buffer, base + place(offset), length).slice(),
    start: (plan) => {
      plan = plan.toJs ? plan.toJs({ dict_converter: Object.fromEntries }) : plan;
      // (T156: every tensor but those on the GPU alone where it is in memory; those keep their offsets in the
      // checkpoint, which only the GPU's worker reads)
      if (direct) {
        plan.tensors = Object.fromEntries(Object.entries(plan.tensors).map(([name, t]) => [name, GPU_ALONE.includes(name) ? t
          : { ...t, offset: place(t.offset), ...(t.scales ? { scales: place(t.scales) } : {}) }]));
      }
      outside.engine = createForward({ memory, base, size: direct ? direct.stored : size, kernels, spawn, gpu, gpuRoom,
        memoryUnsaid, gpuRemembered, gpuForce, halfKeys, direct, plan });
      return outside.engine;
    },
  };
  return outside;
}
