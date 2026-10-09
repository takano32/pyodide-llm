// worker/load.js (T350): the load of a model, whichever kind it is, up to the ready it reports, with the software
// threads and the GPU's report that follow it.
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const { state, adapterAsked, modelKey, loadSeconds } = await import(new URL(`state.js${new URL(import.meta.url).search}`, import.meta.url));
const { since } = await import(new URL(`clock.js${new URL(import.meta.url).search}`, import.meta.url));
const { pythonBuffer, weightsBuffer, weightsDrained, gpuOnlyReady } =
  await import(new URL(`weights.js${new URL(import.meta.url).search}`, import.meta.url));
const { sized, fetchRange } = await import(new URL(`ranges.js${new URL(import.meta.url).search}`, import.meta.url));
const { dropStaleParts, download, readFile, readUrl, HEADER_BYTES, headerInts } =
  await import(new URL(`sources.js${new URL(import.meta.url).search}`, import.meta.url));
const { convert } = await import(new URL(`convert.js${new URL(import.meta.url).search}`, import.meta.url));

// The legacy format carries no metadata, but its header fixes the size of a float32, a float16 and an int8 file.
// A file that is none of them is refused before it is read, and so is a tokenizer.bin of another vocabulary.
async function localOptions(model, vocabulary, head) {
  const header = state.pyodide.toPy(headerInts(head));
  const pieces = state.pyodide.toPy(vocabulary);
  try {
    // what the file cannot say and the settings may: a Qwen2 has biases, a GPT-2 or GPT-NeoX another set of tensors,
    // a Qwen3 the norms of q and k and maybe heads of another size than dim / heads (T124): the form of the options
    const dtype = state.llama2_numpy.checkpoint_dtype(header, model.bytes, model.options ?? {});
    state.llama2_numpy.check_tokenizer(pieces, header);
    return { ...model.options, dtype };
  } finally {
    header.destroy();
    pieces.destroy();
  }
}

export async function load(model, signal, id) {
  signal.throwIfAborted();
  state.loadingKey = modelKey(model);
  state.gpuOnlyNow = undefined;
  await adapterAsked;  // (T156: before any weights are placed)
  await state.gpuOnlyEnding;  // (the review of T156: the GPU's worker of a model on the GPU alone let go before it was built)
  // let go of the previous model first, so that two never have to fit in memory
  if (state.llama) {
    // what forward.js holds of Python's, and its software threads (T93); T205: and the GPU's worker, whose buffers and
    // device the next model waits for (up to forward.js's GPU_END_MS): an iPhone's tab went down in /benchmark/'s rounds
    // where the one before still held them as the next came from the cache
    const released = state.llama.release?.();
    state.llama.destroy();
    state.llama = undefined;
    state.outsideNow = undefined;  // the engine goes; the memory stays for the next model (T96)
    // the engine's closures and the model refer to each other, so only the cycle collector frees the weights
    state.pyodide.runPython("import gc; gc.collect()");
    await released;
    signal.throwIfAborted();
  }
  if (model.hf) {
    postMessage({ type: "status", load: id, text: `${model.name}: ${model.hf.repo ? "fetching from Hugging Face and converting" : "converting"}...` });
    await state.initialized;
    signal.throwIfAborted();
    const converted = await convert(model, signal, id);
    if (!(await gpuOnlyReady(model, id))) return load(model, signal, id);  // T156: the GPU failed on it alone
    await startThreads(model);
    postMessage({
      type: "ready", load: id, pyodide: state.pyodide.version, backend: state.llama.backend, seq_len: state.llama.seq_len,
      seconds: { ...loadSeconds }, heap: heapBytes(), threads: threadsNow(), gpu: watchGpu(id), ...converted,
    });
    return;
  }
  postMessage({ type: "status", load: id, text: `${model.file ? "Reading" : "Downloading"} ${model.name}...` });
  const downloadStarted = performance.now();
  let head;  // the checkpoint's header (T115); a download of this site's tells it when its first part comes
  if (model.url) {
    // the size of a file somewhere else is what its server says
    const first = await fetchRange(model.url.checkpoint, 0, HEADER_BYTES, signal);
    head = first.bytes;
    model.bytes = (await sized(model.url.checkpoint, first, signal)).total;
    if (!(model.bytes > 28)) {
      throw new Error(`${model.url.checkpoint} does not answer range requests, so its size is unknown.`);
    }
  }
  const checkpoint = model.file ? readFile(model, signal, id) : model.url ? readUrl(model, signal, id) : download(model, signal, id);
  const tokenizerBytes = model.file ? model.tokenizerFile.arrayBuffer()
    : fetch(model.url ? model.url.tokenizer : new URL(`../models/${model.tokenizer}`, import.meta.url), { signal }).then((res) => {
      if (!res.ok) {
        throw new Error(`Could not fetch ${model.url?.tokenizer ?? model.tokenizer}: ${res.status}`);
      }
      return res.arrayBuffer();
    });
  tokenizerBytes.catch(() => {});
  let options, weights;
  try {
    await state.initialized;
    signal.throwIfAborted();
    if (model.file) {
      head = new Uint8Array(await model.file.slice(0, HEADER_BYTES).arrayBuffer());
    }
    options = model.file || model.url ? await localOptions(model, new Uint8Array(await tokenizerBytes), head) : model.options;
    head ??= await checkpoint.header;
    signal.throwIfAborted();
    weights = weightsBuffer(model.bytes, headerInts(head), options);
  } catch (error) {
    // T129 (3): a load that ends before its weights have a place (the runtime never came, the file is none the engine
    // takes, the memory said no) fetches no more of the model, as a part that failed for good stops the others. What
    // was queued for the memory goes with the download
    checkpoint.stop?.(error);
    throw error;
  }
  let tokenizer;
  try {
    await checkpoint.into(weights.write);
    const vocabulary = new Uint8Array(await tokenizerBytes);
    await weightsDrained();  // (T156: a model on the GPU alone: every byte of its layers there)
    signal.throwIfAborted();
    // what the source itself measured: the bytes, without the wait for Pyodide that into() may have spent
    loadSeconds.download = checkpoint.seconds ?? since(downloadStarted);

    // from here to the end nothing waits, so no other message gets in between
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    try {
      state.llama = weights.llama(tokenizer.buffer, { kernels: state.kernels, disable: state.disabled, ...options });
    } catch (err) {
      if (!model.file) {
        throw err;
      }
      // the checkpoint has passed its check, so it is the second file or the settings
      const reason = String(err.message ?? err).trim().split("\n").pop();
      throw new Error(`${model.tokenizerFile.name} does not work as the tokenizer.bin of ${model.file.name} (${reason})`);
    }
    loadSeconds.construct = since(constructStarted);
  } finally {
    weights.destroy();
    tokenizer?.buffer.destroy();
  }
  if (!(await gpuOnlyReady(model, id))) return load(model, signal, id);  // T156: the GPU failed on it alone
  await startThreads(model);
  postMessage({
    type: "ready", load: id, pyodide: state.pyodide.version, backend: state.llama.backend, seq_len: state.llama.seq_len,
    seconds: { ...loadSeconds }, heap: heapBytes(), threads: threadsNow(), gpu: watchGpu(id),
    overlapped: checkpoint.overlapped === true && state.pyodideAt > downloadStarted,
  });
  if (!model.file && !model.url) {
    dropStaleParts(model);
  }
}

// The number of threads of the model just loaded (stage 2): ?threads=N fixes it; a count the page remembers for
// this device and model is used (and checked now and then); else forward.js finds it while generating, from the
// number of logical cores, and the page is told the answer to remember. The software threads of the starting count
// are started and warmed here, before the model is ready.
async function startThreads(model) {
  const engine = state.outsideNow?.engine;
  if (!engine?.findThreads) {
    return 1;
  }
  const { fixed = 0, remembered = 0, hint = 1 } = state.threadsRequest ?? {};
  try {
    if (fixed) {
      return await engine.setThreads(fixed);
    }
    return await engine.findThreads({ from: hint, remembered,
      chose: (count) => postMessage({ type: "threads", model: model.id, count, from: remembered ? "remembered" : "hint", hint }),
      compared: (verdict) => postMessage({ type: "threads-compared", model: model.id, ...verdict }) });
  } catch {
    engine.stopThreads?.();
    return 1;
  }
}
export const threadsNow = () => state.outsideNow?.engine?.threads ?? 1;

// T135, T148: what the status line says of the GPU as the model is ready (nothing waits for the GPU: the model runs on
// the CPU until the GPU is ready, and the prompts go where they are faster from then on, forward.js). Once the layers
// are on the GPU, or it is known that they will not be, the page is told { type: "gpu", note, ... } (what the GPU
// chose, and how long it took: the page shows it and remembers the shaders for the next visit). A load let go of
// meanwhile says it too; the page drops what is not of its latest load.
function watchGpu(id) {
  const engine = state.outsideNow?.engine;
  if (!engine) return "prompts on the CPU (the NumPy engine runs this model)";
  if (!engine.gpu) return `prompts on the CPU (${state.benchPage ? "the benchmark times the CPU" : "no WebGPU in a worker here"})`;
  engine.gpu.then((note) => state.outsideNow?.engine === engine && postMessage({ type: "gpu", load: id, note, ...(engine.gpuReady ?? {}) }));
  return engine.gpuStatus;
}

/** The size of Pyodide's WebAssembly memory, which only grows; undefined before Pyodide is there. */
export function heapBytes() {
  const python = state.pyodide?._module?.HEAPU8?.length;
  // T93: the weights and the forward pass have a memory of their own, outside Pyodide's
  return python === undefined ? undefined : python + (state.weightsNow?.buffer.byteLength ?? 0);
}
