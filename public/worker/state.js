// worker/state.js (T350): what the modules of the worker read and set together. A module cannot assign a variable of
// another, so what more than one of them sets is a field of this one object.
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
  // ?gpuTest= (T148, T156; the tests only): what a fallback adapter is taken for (hasWebGpu's comment in worker.js)
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

// T156: where the loops that write the weights can wait for the GPU's worker of a model on the GPU alone (room), and
// before its engine is built (drained)
export const weightsRoom = () => state.gpuOnlyNow?.room?.();
export const weightsDrained = () => state.gpuOnlyNow?.drained?.();
