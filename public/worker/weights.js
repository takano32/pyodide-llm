// worker/weights.js (T350): where the weights of a model go: the memory of forward.js that is kept from model to model,
// a Python buffer where NumPy runs, or the GPU alone (T156), and the sink a conversion writes them through.
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const { state, hasWebGpu, cpuOnly, modelKey } = await import(new URL(`state.js${new URL(import.meta.url).search}`, import.meta.url));

// a Python bytearray that JavaScript fills in place
export function pythonBuffer(size) {
  const buffer = state.pyodide.globals.get("bytearray")(size);
  const write = (offset, chunk) => {
    // the view is taken anew every time: it dies when the WebAssembly memory grows
    const view = buffer.getBuffer("u8");
    view.data.set(chunk, offset);
    view.release();
  };
  return { buffer, write };
}

// Where the weights of a model go (T93): the WebAssembly memory of forward.js, where the forward pass runs, or,
// without the kernels (?without=kernels, or no WebAssembly SIMD), a Python bytearray for the NumPy engine.
// write(offset, bytes) fills it, slice(begin, end) copies a stretch out (to keep a conversion), llama() makes the
// engine on it, destroy() lets go of the Python buffer (a memory of forward.js goes with the engine).
// a software thread of forward.js (stage 2): a module worker of its own, ready once its kernels are warm
const spawnThread = (data) => new Promise((resolve, reject) => {
  const worker = new Worker(new URL(`../helper.js${self.location.search}`, import.meta.url), { type: "module" });
  worker.onmessage = () => resolve({ terminate: () => worker.terminate() });
  worker.onerror = (event) => reject(new Error(event.message ?? "a software thread did not start"));
  worker.postMessage(data);
});

// T135: the GPU's worker of forward.js, a module worker of its own; forward.js starts it
const openGpu = () => new Worker(new URL(`../gpu.js${self.location.search}`, import.meta.url), { type: "module" });

// T96: one memory of forward.js, kept from model to model. A browser reserves address space for every WebAssembly
// memory, shared or not and whatever its maximum, and Chromium refused the third one of a page: the benchmark's
// first round, or a visitor's second change of model, then found no memory at all. So a memory is kept and grown
// as long as the next model fits under its maximum, and made anew only for a larger one. The maximum comes from the
// model (weightsMemory: what it needs and a gigabyte): asking for 4 GB up front left a phone no room for Pyodide's
// own memory, and "Loading Pyodide" never ended (2026-09-25).
// "Fits" is the checkpoint and what the forward pass puts after it (after: forward.js's footprint(), T115). Pythia
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
function pooledWeights(size, after, shared, wide, ahead) {
  const pages = (bytes) => Math.ceil(bytes / 65536);
  const needs = (base) => pages(base + size + (state.weightsPool.limited ? 0 : after)) + 1;
  // what a memory made now is for: the load going on, and the largest of those that follow it where the page said which do
  const largest = ahead === undefined ? after : Math.max(after, ahead);
  const fits = state.weightsPool && state.weightsPool.asked === shared && state.weightsPool.wide === wide && needs(state.weightsPool.base) <= state.weightsPool.maximum;
  if (!fits) {
    // nothing may hold the old memory while the new one is made (T96: Chromium refused a page's third)
    state.weightsPool = state.weightsNow = undefined;
    let memory, base;
    if (shared) {
      try {
        // (a page more where nothing follows, as needs() counts one past what the model takes)
        ({ memory, base } = state.forwardModule.weightsMemory(size, { shared: true, wide, after: largest, ...(ahead !== undefined && { spare: 65536 }) }));
      } catch {
        memory = undefined;  // no shared memory here: one thread
      }
      // (T130's review) a shared memory the browser gave at a lowered maximum (weightsMemory's second and third try: the
      // checkpoint and a gigabyte, or a quarter of one) that the forward pass does not fit would run out of memory when
      // the cache grows, or at once where the corrections do not fit, after the whole checkpoint was read: a plain memory
      // grows as far as the browser allows. One thread, but the model reaches the end of its context. (T242's review: for the
      // loads that follow as well, where the page said which: a round of /benchmark/ that widens int8 needs the most)
      if (memory?.limited && memory.maximum < pages(base + size + largest) + 1) {
        console.info(`memory: the browser gave a shared memory of ${Math.round(memory.maximum * 65536 / 2 ** 20)} MiB, and this model needs ` +
          `${Math.round(pages(base + size + largest) * 65536 / 2 ** 20)} MiB: a memory that is not shared, and one thread`);
        memory = undefined;
      }
    }
    if (!memory) ({ memory, base } = state.forwardModule.weightsMemory(size, { wide }));
    const isShared = shared && memory.buffer instanceof SharedArrayBuffer;
    // a memory without a maximum (not shared) grows as far as the browser allows: 4 GB of pages, 16 GB when wide
    const most = wide ? 262144 : 65536;
    state.weightsPool = { memory, base, wide, asked: shared, shared: isShared, maximum: isShared ? memory.maximum ?? most : most,
                    limited: !isShared || Boolean(memory.limited) };
  }
  const { memory, base } = state.weightsPool;
  const more = pages(base + size) + 1 - memory.buffer.byteLength / 65536;
  if (more > 0) state.forwardModule.growMemory(memory, more, wide);
  return state.weightsPool;
}

// T115: what the forward pass of a checkpoint of size bytes puts after it, at most (forward.js's footprint()): from
// its header (the 7 ints) and the options it is loaded with (their form: head_dim where a head is not dim / heads,
// T124: the keys and values of a Qwen3 0.6B are twice what the header says), on a shared memory or not (an int8
// model's keys and values may be float16; forward.js's keysInHalf says whether they are, T160, T130).
// What footprint() takes (the worker asks keysInHalf the same).
// without: the switches of the load (those of the load going on, or of one that follows: loadsAhead, T242).
function forwardOptions(options, shared, without = state.disabled) {
  const { dtype = "float32" } = options;
  const int8 = !without.includes("int8"), quantized = ["int8", "int6", "ternary"].includes(dtype);
  return {
    ...options, dtype, int8, relaxed: Boolean(state.jsKernels?.relaxed) && !without.includes("relaxed"),
    halfKV: quantized && int8 && !without.includes("kv16"), shared,
    outliers: state.llama2_numpy.OUTLIER_CHANNELS, gpu: hasWebGpu,
  };
}
const afterCheckpoint = (header, size, options, shared, without) => state.forwardModule.footprint(header, size, forwardOptions(options, shared, without));
// the page cross-origin isolated (stage 3), shared memories to be had, and not ?threads=1: the memory is shared
const sharedWanted = () => Boolean(state.sharedKernels && self.crossOriginIsolated && state.threadsRequest?.fixed !== 1);

// T115: the bits of a model converted with none asked for (weightsFor() in src/models.js asks for six bits where the
// device says it has too little memory): int8 unless its forward pass does not fit a 32-bit memory and this browser
// has no 64-bit one (T133), then six bits (T98: 7/9 of int8's memory, and about half as fast). The converter calls
// this once it knows the header: the size of either (sizes) and what the forward pass puts after them depend on it.
// form: what else sizes the forward pass (llama2_numpy.FORM, llama2_convert.Stream.form).
export function automaticBits(header, form, sizes) {
  const ints = header.toJs(), int8 = sizes.toJs({ dict_converter: Object.fromEntries }).int8;
  const given = form.toJs({ dict_converter: Object.fromEntries });
  header.destroy();
  sizes.destroy();
  form.destroy();
  if (!state.forwardModule) return "int8";  // no forward.js (no WebAssembly SIMD): NumPy widens every weight anyway
  const shared = sharedWanted();
  return state.forwardModule.automaticDtype(int8, afterCheckpoint(ints, int8, { ...given, dtype: "int8" }, shared),
    Boolean(state.wideKernels?.plain));
}

// header: the checkpoint's 7 ints, options: what it is loaded with (its dtype and arch): what the forward pass puts
// after the checkpoint follows from them (T115). T156, keep: where a conversion is kept as it comes (kept.js's
// keeper()), for a model on the GPU alone, whose weights nothing holds whole to keep afterwards
export function weightsBuffer(size, header, options, keep) {
  if (state.jsKernels && !state.disabled.includes("kernels")) {
    // a shared memory where the page is cross-origin isolated (stage 3), unless ?threads=1; else one thread
    const wanted = sharedWanted();
    // T101: a model past 4 GiB with its forward pass goes on a 64-bit memory (about a tenth slower: only when it has
    // to). T130: where a shared one is refused after all, a model that fit a 32-bit one still does on the plain one
    // (footprint() keeps float16 keys and values there where only float32 would not fit), and a 64-bit one stays
    // 64-bit (with float32 keys and values, on a memory that has no maximum): the 64-bit question stays answered
    const after = afterCheckpoint(header, size, options, wanted);
    // T129 (7): a model past even a 64-bit memory is refused here, before its weights are fetched (a Qwen2.5 32B of
    // ?hf=, about 37 GB as int8, began a 65 GB download and failed at 7.8 GB). The words are the owner's (2026-09-28)
    if (state.forwardModule.pastWide(size, after)) {
      throw new Error(`This model is too large for a web page: it needs about ${Math.ceil((size + after) / 1e9)} GB of ` +
        "memory, and a browser gives a page 16 GB at most.");
    }
    // T148: the layers on the GPU are a second copy of them, in the same memory where the GPU is a phone's or an
    // Apple's: both, with the rest of this model, within half of what the device says it has (as src/models.js's
    // weightsFor asks for six bits past half). T156: a model that does not fit so goes on the GPU alone where it can
    // (the owner's B, 2026-09-27: forward.js's weightsPlace; the double copy held to 6.5 GiB where the device says 8),
    // decided before any memory is made for it (a memory of the whole checkpoint would be one too many, T96)
    const deviceMemory = self.navigator?.deviceMemory ?? 4;
    // T205: a browser that does not say (Safari, Firefox) keeps a generation's steps (the classifier and the embedding
    // on the GPU as well) on the CPU: nor does it put a model on the GPU alone, whose steps are there
    const memoryUnsaid = self.navigator?.deviceMemory === undefined;
    const onGpu = gpuOnlyWeightsFor(size, header, options, after, deviceMemory);
    if (onGpu.mode === "gpu") return gpuOnlyBuffer(size, header, options, onGpu, keep);
    const gpuRoom = onGpu.gpuRoom;
    const wide = state.forceWide || state.forwardModule.needsWide(size, after);
    if (wide && !state.wideKernels?.plain) {
      throw new Error("This model needs more than 4 GB of memory, which this browser cannot give a web page (no 64-bit " +
        "WebAssembly memory: Safari has none yet). Chrome and Firefox can.");
    }
    // T242: where the page said which loads follow on this model, the memory is made for the largest of them (a load
    // without the kernels takes none: NumPy's weights are Python's)
    const ahead = state.loadsAhead && Math.max(0, ...state.loadsAhead.filter((without) => !without.includes("kernels"))
      .map((without) => afterCheckpoint(header, size, options, wanted, without)));
    const { memory, base, shared } = pooledWeights(size, after, wanted && (!wide || Boolean(state.wideKernels.shared)), wide, ahead);
    // T160: the type of the keys and values that after counts, T130: on the memory the browser gave
    const halfKeys = state.forwardModule.keysInHalf(header, size, forwardOptions(options, shared));
    const kernels = wide ? (shared ? state.wideKernels.shared : state.wideKernels.plain) : (shared ? state.sharedKernels : state.jsKernels);
    const spawn = shared ? spawnThread : undefined;
    state.weightsNow = memory;
    return {
      write: (offset, chunk) => new Uint8Array(memory.buffer, base + offset, chunk.length).set(chunk),
      slice: (begin, end) => new Uint8Array(memory.buffer, base + begin, end - begin).slice(),
      llama: (tokenizer, options) => {
        // (not on the benchmark's page, ?bench=, T45: it times the CPU's combinations, which a GPU starting beside
        // them would slow down with its upload and compilation; its first load is one of them)
        // (nor under the rounds of /benchmark/'s model section, T184: the same; its page path, timed on the first load
        // before the rounds, has the GPU, and T205: the rounds' loads wait for that GPU to let go of its device)
        state.outsideNow = state.forwardModule.external({ memory, base, size, kernels, spawn, gpu: hasWebGpu && !state.benchPage && !state.benching ? openGpu : undefined, gpuRoom, memoryUnsaid,
          gpuRemembered: state.gpuRequest?.remembered, gpuForce: state.gpuForce, halfKeys });
        return state.llama2_numpy.Llama.callKwargs(null, tokenizer, { ...options, external: state.outsideNow });
      },
      destroy() {},
    };
  }
  state.weightsNow = undefined;
  state.outsideNow = undefined;
  const { buffer, write } = pythonBuffer(size);
  return {
    write,
    slice(begin, end) {
      const view = buffer.getBuffer("u8");
      const copy = view.data.slice(begin, end);
      view.release();
      return copy;
    },
    llama: (tokenizer, options) => state.llama2_numpy.Llama.callKwargs(buffer, tokenizer, options),
    destroy: () => buffer.destroy(),
  };
}

// T156: what a model on the GPU alone needs of this page and device, whatever the model (gpuOnlyUnfit has the rest): a
// memory the device says (T205), the GPU's worker (not the benchmark's rounds), a shared memory and the 64-bit kernels,
// the int8 kernels, an adapter that is no fallback (but for the tests), and no failure of the GPU under this model before
// (cpuOnly). convert() asks it too before it opens a file to keep a conversion in as it comes (the review of T156)
// (the owner, 2026-09-27: and no verdict the page kept that the CPU is faster here: then the CPU at once)
export const gpuOnlyPossible = (dtype) => gpuOnlyAllowed() && !aloneKept(dtype);
const gpuOnlyAllowed = () => hasWebGpu && self.navigator?.deviceMemory !== undefined && !state.benchPage && !state.benching && sharedWanted() &&
  Boolean(state.wideKernels?.shared) && !state.disabled.includes("int8") && !cpuOnly.has(state.loadingKey) &&
  Boolean(state.gpuAdapter) && (!state.gpuAdapter.fallback || Boolean(state.gpuForce.fallback));
// T156: whether the page kept that the CPU was faster here than this model on the GPU alone (forward.js's aloneHolds)
// (dtype, T232: the key is the one gpu.js gave the verdict, a ternary model's its own)
const aloneKept = (dtype) => Boolean(state.gpuAdapter) &&
  state.forwardModule.aloneHolds(state.gpuRequest?.remembered?.alone, dtype === "ternary" ? state.gpuAdapter.ternaryKey : state.gpuAdapter.key, state.gpuRequest?.cpu);
// T156: where a model goes ({ mode: "both" | "gpu" | "cpu", gpuRoom }, forward.js's weightsPlace): the GPU alone only
// for a Llama of int8 (T232: or of ternary weights) the GPU's steps take (gpuOnlyUnfit), where the page and the device
// may (gpuOnlyPossible)
function gpuOnlyWeightsFor(size, header, options, after, deviceMemory) {
  const { dtype = "float32" } = options, form = { arch: options.arch, bias: options.bias, qk_norm: options.qk_norm, head_dim: options.head_dim };
  const cpu = size + after, gpu = state.forwardModule.gpuBytes(header, { ...form, dtype });
  // (T232: with the form's rotated basis, which the GPU does not turn the inputs for (T237): the worker did not hand it
  // on, and a Llama in a rotated basis would have gone on the GPU alone to be loaded again on the CPU once built)
  const fit = gpuOnlyAllowed() && !state.forwardModule.gpuOnlyUnfit(header, dtype, { ...form, rotated: options.rotated }, state.gpuAdapter, state.gpuForce);
  const eligible = fit && !aloneKept(dtype);
  // (the second review of T156: the status line then says only the memory's reason where the layers do not fit beside
  // the CPU's copy; the console says the kept verdict, and what asks again)
  if (fit && !eligible) {
    console.info("gpu: the CPU as /benchmark/ measured it was faster here than this model on the GPU alone, as the page kept it: " +
      "on the CPU (a new run of /benchmark/'s CPU section, another browser version or new shaders weigh the two again)");
  }
  if (!eligible) return state.forwardModule.weightsPlace({ cpu, gpu, deviceMemory });
  const tensors = placesOf(header, dtype, form), stored = size - state.forwardModule.gpuHoles(tensors).reduce((sum, [a, b]) => sum + b - a, 0);
  const gpuOnly = stored + afterCheckpoint(header, stored, { ...options, direct: true }, true);
  return { ...state.forwardModule.weightsPlace({ cpu, gpuOnly, gpu, deviceMemory, eligible, forced: state.gpuForce.only }), tensors, stored, gpuOnly };
}
// llama2_numpy.external_tensors(): where every tensor of the checkpoint is, from its header alone
function placesOf(header, dtype, form) {
  const proxy = state.llama2_numpy.external_tensors(header, dtype, form);
  try {
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    proxy.destroy();
  }
}
// T156: the weights of a model on the GPU alone: the layers' matrices to the GPU's worker as they come (forward.js's
// gpuOnlyWeights), the rest into a memory of their size, and every byte to keep's file where the conversion is kept
function gpuOnlyBuffer(size, header, options, { tensors, stored, gpuOnly }, keep) {
  const after = gpuOnly - stored, wide = state.forwardModule.needsWide(stored, after);
  const { memory, base, shared } = pooledWeights(stored, after, true, wide);
  if (!shared) throw new Error("This browser gave no shared memory for a model on the GPU alone.");
  const worker = openGpu();
  // (forward.js listens from start() on. Here only "ended", beside it: T205, the next load after the GPU's worker let
  // go of its device and its buffers, which it makes for every layer as it opens, before a byte comes)
  let ended;
  const ending = new Promise((resolve) => { ended = resolve; });
  worker.addEventListener("message", ({ data }) => data?.type === "ended" && ended());
  worker.addEventListener("error", () => ended());
  // the GPU's worker stopped, and once it said "ended" (or after GPU_END_MS: terminated), resolved; as release() does
  const end = () => {
    worker.postMessage({ type: "stop" });
    let timer;
    const late = new Promise((resolve) => { timer = setTimeout(resolve, state.forwardModule.GPU_END_MS); });
    return Promise.race([ending, late]).then(() => {
      clearTimeout(timer);
      worker.terminate();
    });
  };
  const weights = state.forwardModule.gpuOnlyWeights({ memory, base, size, tensors, worker });
  worker.postMessage({ type: "open", plan: state.forwardModule.gpuOnlyPlan(header, tensors, state.gpuForce, state.gpuRequest?.remembered), flow: weights.flow });
  // (T156: the checkpoint's size, its layers' multiply-adds, /benchmark/'s CPU reading and the page's use, for the
  // estimate the GPU is held against: forward.js's aloneVerdict)
  const direct = state.gpuOnlyNow = { worker, lost: null, stored, size, place: weights.place,
    layerWeights: state.forwardModule.layerWeightsOf(header, options), cpu: state.gpuRequest?.cpu, usage: state.gpuRequest?.usage ?? state.forwardModule.USAGE_UNKNOWN,
    room: weights.room, drained: weights.drained, onLost: (why) => { direct.lost = why; }, end };
  state.weightsNow = memory;
  const kernels = wide ? state.wideKernels.shared : state.sharedKernels;
  let built = false;
  return {
    direct,
    write(offset, chunk) {
      weights.write(offset, chunk);
      keep?.write(offset, chunk);
    },
    room: weights.room,
    drained: weights.drained,  // every byte of the layers on the GPU (or the GPU failed: start() says so)
    llama: (tokenizer, engineOptions) => {
      state.outsideNow = state.forwardModule.external({ memory, base, size, kernels, spawn: spawnThread, gpu: () => worker, gpuForce: state.gpuForce, halfKeys: true, direct,
        gpuRemembered: state.gpuRequest?.remembered });
      const made = state.llama2_numpy.Llama.callKwargs(null, tokenizer, { ...engineOptions, external: state.outsideNow });
      built = true;
      return made;
    },
    // (the review of T156: a load cancelled or failed before its engine was built, or another try of the converter's
    // (sink.open() again), let go of the GPU's worker: nothing else would, and it held the device and a buffer for
    // every layer's matrices, 4.1 GB for Llama 3.2 3B, for the rest of the visit. The next load waits for it)
    destroy() {
      if (!built) state.gpuOnlyEnding = end();
    },
  };
}

// T156: where the loops that write the weights can wait for the GPU's worker of a model on the GPU alone (room), and
// before its engine is built (drained)
export const weightsRoom = () => state.gpuOnlyNow?.room?.();
export const weightsDrained = () => state.gpuOnlyNow?.drained?.();
// T156: after a model on the GPU alone is built: its GPU ready (true), or failed (false: the worker let go of it, and
// the model goes on the CPU from now on this visit)
export async function gpuOnlyReady(model, id) {
  const direct = state.gpuOnlyNow;
  if (!direct || state.outsideNow?.engine === undefined) return true;
  await state.outsideNow.engine.gpu;
  if (!direct.lost) return true;
  console.warn(`gpu: ${direct.lost}: the model was on the GPU alone, and is loaded again on the CPU`);
  cpuOnly.add(modelKey(model));
  // (the CPU was faster: the page keeps it for this model and device once the load on the CPU is ready, and the next
  // visit loads on the CPU at once. The load's id: the page reads no word of a load another choice cancelled, the
  // second review of T156: a verdict that came late was kept for the model chosen since)
  if (direct.verdict) postMessage({ type: "gpu-alone", load: id, alone: direct.verdict });
  // (the GPU's worker let go of its device: stopped by forward.js where it started it, and here where it did not, as a
  // model whose GPU forward.js gave up as the engine was built, before start(): its release() has no "ended" to wait
  // for then. The review of T156: the load on the CPU begins after it, T205)
  await direct.end();
  state.gpuOnlyNow = undefined;
  return false;
}

// T93: where the converter writes the checkpoint, piece by piece, straight into where the engine will read it (a
// Python buffer on the way would stay: Pyodide's memory never shrinks). sink is what the converter calls
// (llama2_convert.Writer), weights what sink.open() made (weightsBuffer), bytes its size; release() lets go of it once
// (T145: a conversion that failed after sink.open() left a Python buffer of the model's size behind with
// ?without=kernels). Out of convert() so that tests/worker-sink-check.mjs can follow sink.open() to footprint().
export function checkpointSink(keep) {
  const into = {
    weights: undefined, bytes: 0,
    release() {
      into.weights?.destroy();
      into.weights = undefined;
    },
  };
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
      into.weights.write(offset, view.data);
      view.release();
    },
  };
  return into;
}
