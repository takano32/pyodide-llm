// worker/state.js (T350): what the modules of the worker read and set together. A module cannot assign a variable of
// another, so what more than one of them sets is a field of this one object; below it, what this visit's worker found
// once and all of them read (WebGPU and its adapter, the models that left the GPU, the seconds of a load).
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const HF_CONNECTIONS = 6;
export const state = {
  pyodide: undefined, llama2_numpy: undefined, llama2_convert: undefined, llama: undefined, kernels: undefined,
  // the models kept from earlier conversions (kept.js, T99), imported when the first conversion comes
  keptModule: undefined,
  // T93: the forward pass in JavaScript (forward.js) and its kernels, compiled once; the memory of the model loaded now
  forwardModule: undefined, jsKernels: undefined, weightsNow: undefined,
  // T93 stage 2: the kernels for a shared memory (only where the page is cross-origin isolated), what the page asked
  // about the number of threads ({ fixed, remembered, hint }), and the forward pass of the model loaded now
  sharedKernels: undefined, threadsRequest: undefined, outsideNow: undefined,
  // T101: the kernels for a 64-bit memory, for a model past 4 GiB ({ plain, shared }), where the browser has Memory64
  wideKernels: undefined,
  // ?wide=on: a 64-bit memory for every model, to try that path on a small one (measuring, tests), as ?offline=on says
  forceWide: false,
  // ?gpuTest= (T148, T156; the tests only): what a fallback adapter is taken for (hasWebGpu's comment below)
  gpuForce: {},
  // T156: the adapter, asked for before a model is loaded ({ fallback, limits } or null): whether a model the device
  // cannot hold twice goes on the GPU alone is decided before its bytes come (forward.js's weightsPlace, gpuOnlyUnfit)
  gpuAdapter: null,
  // ?bench= (T45): the page measures the CPU's combinations, and no GPU starts beside them
  benchPage: false,
  // what the page kept of the GPU's shaders for the model asked for ({ remembered }), as threadsRequest
  gpuRequest: undefined,
  // the optimizations this session leaves out (T52): ?without=relaxed,sampler, and ?kernel=off as it always was
  disabled: [],
  // what the page's own URL said, to come back to after a benchmark has tried other combinations (T77)
  pageSwitches: [],
  // init() as a promise: every load waits for it, also the one that replaces the first
  initialized: undefined,
  // the AbortController of the load that is going on, and a promise that settles once it has cleaned up
  loading: undefined, unloaded: Promise.resolve(),
  // when Pyodide became usable, to tell that first model from the ones chosen afterwards
  pyodideAt: 0,
  // T96: the one memory of forward.js that is kept from model to model (pooledWeights() of weights.js says why)
  weightsPool: undefined,
  // T242: the switches of each load the page says will follow on the model of its init, or undefined (pooledWeights())
  loadsAhead: undefined,
  // T156: the model on the GPU alone that is loading or loaded now (what gpuOnlyBuffer() made), or undefined
  gpuOnlyNow: undefined,
  // the GPU's worker of a model on the GPU alone let go before its engine was built (gpuOnlyBuffer()'s destroy()): the next load
  // waits for it, as release() waits for a built one's (T205)
  gpuOnlyEnding: Promise.resolve(),
  // the key of the model being loaded (cpuOnly)
  loadingKey: undefined,
  // T107: ?hfParts=<MiB>&hfConnections=<N> fix the two, to measure; the page offers no way to them
  hfPartBytes: 0, hfConnections: HF_CONNECTIONS,  // 0: not fixed, decided per file from its first part
  // the run that is going on, and whether the page asked it to stop
  generating: undefined, stopped: false,
  lastLoad: undefined,  // the load message of the model now (T156)
  benching: false,  // the benchmark's rounds are running (T45); see the bench message
  pathing: false,  // T184: the benchmark's page path is being timed; see the paths message
};

// T148: a prompt's tokens through the layers on the GPU (gpu.js) by default, wherever this worker has WebGPU and
// forward.js can put the model there, and the GPU is faster than the CPU here (AGENTS.md's policy 9: no option).
// ?gpuTest=on, for the tests only: a fallback adapter (SwiftShader, the only WebGPU of CI) taken as a GPU, and every
// block of a prompt it can take given to it (its speed is no GPU's: the tests look at its numbers, not at its time),
// and the first right shader of the matrices taken untimed (SwiftShader timed Llama 3.2 1B's past gpu.js's 180 s)
export const hasWebGpu = Boolean(self.navigator?.gpu);
// (and its key, T148's: what the page kept of the model on this device holds only for the same, shaders.js's deviceKey)
export const adapterAsked = !hasWebGpu ? Promise.resolve() : navigator.gpu.requestAdapter().then(async (adapter) => {
  if (!adapter) return;
  const { maxStorageBufferBindingSize, maxBufferSize } = adapter.limits;
  // (T232, packed: WGSL's packed int8 dot, which ternary weights are multiplied with on the GPU)
  state.gpuAdapter = { fallback: Boolean(adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter),
    limits: { maxStorageBufferBindingSize, maxBufferSize },
    packed: Boolean(navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product")) };
  try {
    const wgsl = await import(new URL(`../shaders.js${self.location.search}`, import.meta.url));
    state.gpuAdapter.key = wgsl.deviceKey(adapter);
    // (T232: a model of ternary weights has a key of its own, with its shaders: gpu.js says that one)
    state.gpuAdapter.ternaryKey = wgsl.deviceKey(adapter, adapter, true);
  } catch {
    // no key: nothing the page kept of this device holds (a model on the GPU alone is weighed again)
  }
}).catch(() => {});
// T156: the models whose GPU failed while they were on it alone, loaded again on the CPU from then on (this visit)
export const cpuOnly = new Set();
export const modelKey = (model) => model.hf ? `hf:${model.hf.repo}@${model.hf.revision}` : model.id ?? model.name;

// how long the load took, in seconds: Pyodide once per session, the other two per model. The download of a model
// of this site runs while Pyodide loads and usually ends first, so its seconds are counted until the last byte
// arrives (not until the bytes reach Python, which has to wait for Pyodide). The page says that the two overlap
// instead of adding them up, but only for the model that was loaded while Pyodide was still coming.
export const loadSeconds = {};
