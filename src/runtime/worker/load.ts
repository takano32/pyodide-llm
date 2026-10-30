// worker/load.ts (T.350): the load of a model, whichever kind it is, up to the ready it reports, with the software
// threads and the GPU's report that follow it.
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T.367.3).

import { pyodide, llama2_numpy, pythonKernels, pyodideAt, benchPage, runtimeReady, forwardModule, blockKernels } from "./runtime.ts";
import { threadsRequest, root, switchesNow } from "./request.ts";
import { llama, engineNow, adopted, letGo } from "./loaded.ts";
import { since, took, secondsSoFar, loadStarted } from "./clock.ts";
import { pythonBuffer, weightsBuffer, weightsDrained, gpuOnlyReady, loadBegan, weightsBytes, runsPacked, rowsReady } from "./weights.ts";
import { sized, fetchRange } from "./ranges.ts";
import { dropStaleParts, download, readFile, readUrl, HEADER_BYTES, headerInts } from "./sources.ts";
import { convert, converter } from "./convert.ts";
import type { Sized } from "./sources.ts";
import { lastLine, told } from "../failure.ts";
import type { Thrown } from "../failure.ts";
import type { GpuChose, GpuSaid, Listed, OfTwoFiles, OfTwoUrls, Ready, Status, Threads, ThreadsCompared } from "../contracts/messages.ts";

// The legacy format carries no metadata, but its header fixes the size of a float32, a float16 and an int8 file.
// A file that is none of them is refused before it is read, and so is a tokenizer.bin of another vocabulary.
async function localOptions(model: OfTwoFiles | OfTwoUrls, vocabulary: Uint8Array, head: Uint8Array) {
  const header = pyodide.toPy(headerInts(head));
  const pieces = pyodide.toPy(vocabulary);
  try {
    // what the file cannot say and the settings may: a Qwen2 has biases, a GPT-2 or GPT-NeoX another set of tensors,
    // a Qwen3 the norms of q and k and maybe heads of another size than dim / heads (T.124): the form of the options
    const dtype = llama2_numpy.checkpoint_dtype(header, model.bytes!, model.options ?? {});
    llama2_numpy.check_tokenizer(pieces, header);
    return { ...model.options, dtype };
  } finally {
    header.destroy();
    pieces.destroy();
  }
}

export async function load(model: Listed, signal: AbortSignal, id: number): Promise<void> {
  signal.throwIfAborted();
  loadStarted();  // (T.460)
  // (T.156: the adapter before any weights are placed, and the GPU's worker of a model on the GPU alone let go before it
  // was built: weights.ts)
  await loadBegan(model);
  // let go of the previous model first, so that two never have to fit in memory (loaded.ts)
  await letGo(signal);
  if (model.hf) {
    postMessage({ type: "status", load: id, text: `${model.name}: ${model.hf.repo ? "fetching from Hugging Face and converting" : "converting"}...` } satisfies Status);
    await runtimeReady();
    signal.throwIfAborted();
    const converted = await convert(model, signal, id);
    if (!(await gpuOnlyReady(model, id, engineNow()))) return load(model, signal, id);  // T.156: the GPU failed on it alone
    if (!rowsReady(model)) return load(model, signal, id);  // T.329.2.2: its table was not all kept as it came
    await startThreads(model);
    postMessage({
      type: "ready", load: id, pyodide: pyodide.version, backend: llama!.backend, seq_len: llama!.seq_len,
      seconds: secondsSoFar(), heap: heapBytes(), threads: threadsNow(), gpu: watchGpu(id), ...converted,
    } satisfies Ready);
    return;
  }
  postMessage({ type: "status", load: id, text: `${model.file ? "Reading" : "Downloading"} ${model.name}...` } satisfies Status);
  const downloadStarted = performance.now();
  let head;  // the checkpoint's header (T.115); a download of this site's tells it when its first part comes
  if (model.url) {
    // the size of a file somewhere else is what its server says
    const first = await fetchRange(model.url.checkpoint, 0, HEADER_BYTES, signal);
    head = first.bytes;
    model.bytes = (await sized(model.url.checkpoint, first, signal)).total;
    if (!(model.bytes > 28)) {
      throw new Error(`${model.url.checkpoint} does not answer range requests, so its size is unknown.`);
    }
  }
  const checkpoint = model.file ? readFile(model, signal, id) : model.url ? readUrl(model as Sized, signal, id) : download(model, signal, id);
  const tokenizerBytes = model.file ? model.tokenizerFile.arrayBuffer()
    : fetch(model.url ? model.url.tokenizer : new URL(`models/${model.tokenizer}`, root), { signal }).then((res) => {
      if (!res.ok) {
        throw new Error(`Could not fetch ${model.url?.tokenizer ?? model.tokenizer}: ${res.status}`);
      }
      return res.arrayBuffer();
    });
  tokenizerBytes.catch(() => {});
  let options, weights;
  try {
    await runtimeReady();
    signal.throwIfAborted();
    if (model.file) {
      head = new Uint8Array(await model.file.slice(0, HEADER_BYTES).arrayBuffer());
    }
    options = model.file || model.url ? await localOptions(model, new Uint8Array(await tokenizerBytes), head!) : model.options;
    head ??= await checkpoint.header;
    signal.throwIfAborted();
    // T.363.5: a checkpoint whose options give rows the kinds of llama.cpp's blocks (a visitor's file with its settings)
    // runs on the kernels of those, fetched for it; where NumPy computes, the readers of the blocks are the converter's
    if (forwardModule?.takesBlocks(options?.kinds)) await (runsPacked() ? blockKernels() : converter(signal));
    weights = weightsBuffer(model.bytes!, headerInts(head!), options);
  } catch (error) {
    // T.129 (3): a load that ends before its weights have a place (the runtime never came, the file is none the engine
    // takes, the memory said no) fetches no more of the model, as a part that failed for good stops the others. What
    // was queued for the memory goes with the download
    checkpoint.stop?.(error);
    throw error;
  }
  let tokenizer;
  try {
    await checkpoint.into(weights.write);
    const vocabulary = new Uint8Array(await tokenizerBytes);
    await weightsDrained();  // (T.156: a model on the GPU alone: every byte of its layers there)
    signal.throwIfAborted();
    // what the source itself measured: the bytes, without the wait for Pyodide that into() may have spent
    took("download", checkpoint.seconds ?? since(downloadStarted));

    // from here to the end nothing waits, so no other message gets in between
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    try {
      adopted(weights.llama(tokenizer.buffer, { kernels: pythonKernels, disable: switchesNow(), ...options }));
    } catch (err) {
      if (!model.file) {
        throw err;
      }
      // the checkpoint has passed its check, so it is the second file or the settings
      const reason = lastLine((err as Thrown).message ?? err);
      throw new Error(`${model.tokenizerFile.name} does not work as the tokenizer.bin of ${model.file.name} (${reason})`);
    }
    took("construct", since(constructStarted));
  } finally {
    weights.destroy();
    tokenizer?.buffer.destroy();
  }
  if (!(await gpuOnlyReady(model, id, engineNow()))) return load(model, signal, id);  // T.156: the GPU failed on it alone
  await startThreads(model);
  postMessage({
    type: "ready", load: id, pyodide: pyodide.version, backend: llama!.backend, seq_len: llama!.seq_len,
    seconds: secondsSoFar(), heap: heapBytes(), threads: threadsNow(), gpu: watchGpu(id),
    overlapped: checkpoint.overlapped === true && pyodideAt > downloadStarted,
  } satisfies Ready);
  if (!model.file && !model.url) {
    dropStaleParts(model);
  }
}

// The number of threads of the model just loaded (stage 2): ?threads=N fixes it; a count the page remembers for
// this device and model is used (and checked now and then); else forward.ts finds it while generating, from the
// number of logical cores, and the page is told the answer to remember. The software threads of the starting count
// are started and warmed here, before the model is ready.
async function startThreads(model: Listed) {
  const engine = engineNow();
  if (!engine?.findThreads) {
    return 1;
  }
  const { fixed = 0, remembered = 0, hint = 1 } = threadsRequest ?? {};
  try {
    if (fixed) {
      return await engine.setThreads(fixed);
    }
    return await engine.findThreads({ from: hint, remembered,
      chose: (count) => postMessage({ type: "threads", model: model.id, count, from: remembered ? "remembered" : "hint", hint } satisfies Threads),
      compared: (verdict) => postMessage({ type: "threads-compared", model: model.id, ...verdict } satisfies ThreadsCompared) });
  } catch (error) {
    // (T.380.2: said. The model is ready all the same, on one thread)
    console.warn(`threads: the software threads did not start (${told(error)}): one thread`);
    engine.stopThreads?.();
    return 1;
  }
}
export const threadsNow = () => engineNow()?.threads ?? 1;

// T.135, T.148: what the status line says of the GPU as the model is ready (nothing waits for the GPU: the model runs on
// the CPU until the GPU is ready, and the prompts go where they are faster from then on, forward.ts). Once the layers
// are on the GPU, or it is known that they will not be, the page is told { type: "gpu", note, ... } (what the GPU
// chose, and how long it took: the page shows it and remembers the shaders for the next visit). A load let go of
// meanwhile says it too; the page drops what is not of its latest load.
// (T.368.3: the message is cast, not held: tsc reads what is spread in or not as every key being there or not, each by itself)
function watchGpu(id: number) {
  const engine = engineNow();
  if (!engine) return "prompts on the CPU (the NumPy engine runs this model)";
  // (the words of an engine without a GPU's side are the engine's: forward/reasons.ts, which this file cannot read)
  if (!engine.gpu) return `prompts on the CPU (${benchPage ? "the benchmark times the CPU" : engine.gpuWhyNot})`;
  engine.gpu.then((note) => engineNow() === engine && postMessage({ type: "gpu", load: id, note, ...(engine.gpuReady ?? {}) } as GpuChose | GpuSaid));
  return engine.gpuStatus!;  // (a line wherever the engine has a GPU: engine.gpu, above)
}

/** The size of Pyodide's WebAssembly memory, which only grows; undefined before Pyodide is there. */
export function heapBytes() {
  const python = pyodide?._module?.HEAPU8?.length;
  // T.93: the weights and the forward pass have a memory of their own, outside Pyodide's
  return python === undefined ? undefined : python + weightsBytes();
}
