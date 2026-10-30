// worker/convert.ts (T.350): a Hugging Face model converted in here as it arrives (src/python/llama2_convert.py), and what
// a conversion made kept for the next visit and read back (kept.js).
// T.374.2.1: the conversion is conducted by Python (src/python/convert/conduct.py) and answered by conduct.ts: a model of
// huggingface.co from there, and (T.374.2.2) a folder of the visitor's disk from its Files.
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T.367.3).

import { pyodide, llama2_numpy, pythonKernels, blockKernels } from "./runtime.ts";
import { switchesNow } from "./request.ts";
import { adopted } from "./loaded.ts";
import { since, took } from "./clock.ts";
import { pythonBuffer, automaticBits, weightsBuffer, gpuOnlyPossible, weightsRoom, weightsDrained, checkpointSink, runsPacked, tableInMemoryFromNow, blocksWidenedNow } from "./weights.ts";
import { conductOf, answered, fromHub, fromFolder } from "./conduct.ts";
import { HEADER_BYTES, headerInts } from "./sources.ts";
import { templatePackage, placePython } from "./pyodide.ts";
import { built } from "../built.ts";
import { named } from "../updated.ts";
// (the kept models' module as a type; it is imported when a conversion first needs it, in convert())
import type * as Kept from "../kept.ts";
import type { Keeper, Weights } from "./weights.ts";
import { told } from "../failure.ts";
import type { Thrown } from "../failure.ts";
import type { Hf, OfHuggingFace, Progress, Status } from "../contracts/messages.ts";
import type { Progress as Arriving } from "../contracts/conduct.ts";
import type { Bytes, Conduct, Conversion, ConverterWindow, Dict, Options, Proxy, Widened } from "../contracts/python.ts";

// T.368.3: a model of huggingface.co, as what is kept reads it (kept.ts's Converted): its repository and revision
type Remote = OfHuggingFace & { hf: Hf & { repo: string; revision: string } };

// What this file alone holds (T.378.3): the converter's window, imported when the first conversion comes (the function of that name below),
// and the models kept from earlier conversions (kept.js, T.99), imported then too
let llama2_convert: ConverterWindow | undefined, keptModule: typeof Kept | undefined;

// What a conversion made is kept for the next visit (kept.js): in the origin private file system where there is one
// (T.99), else in the Cache API. The original is twice as large, and fetching and converting it again on every visit
// would be no way to use a model. The page lists what is kept and deletes it.
// Returns what the page needs of it, or { miss } with why nothing kept could be used (for tests/e2e.mjs).
async function loadConverted(model: Remote, signal: AbortSignal, id: number) {
  let kept;
  try {
    // under the bits asked for, or either the worker may choose (T.115), by this converter (T.116)
    kept = await keptModule!.openKept(model);
  } catch (error) {
    return { miss: `could not open what is kept: ${(error as Thrown).message ?? error}` };
  }
  if (!kept) {
    return { miss: "nothing kept for this model" };
  }
  const { manifest } = kept;
  const started = performance.now();
  // the first part holds the header, which the memory is chosen by (T.115)
  // (T.329.2.2: the stretch of a table of every layer's inputs that stays in the file is not read: skipping, once the
  // header says it)
  const skipping: { range?: [number, number] } = {};
  const parts = kept.parts(skipping)[Symbol.asyncIterator]();
  const unreadable = (error: unknown) => {
    if (signal.aborted) {
      throw error;
    }
    return { miss: `could not read what is kept: ${(error as Thrown).message ?? error}` };  // evicted: convert again
  };
  let part;
  try {
    part = await parts.next();
  } catch (error) {
    return unreadable(error);
  }
  if (part.done || part.value.length < HEADER_BYTES) {
    return { miss: "what is kept is empty" };
  }
  const weights = weightsBuffer(manifest.bytes, headerInts(part.value), manifest.options, undefined, kept.where === "opfs");
  if (weights.hole) skipping.range = weights.hole;
  let tokenizer;
  try {
    let offset = 0;
    try {
      for (; !part.done; part = await parts.next()) {
        signal.throwIfAborted();
        weights.write(offset, part.value);
        await weightsRoom();  // (T.156)
        offset += part.value.length;
        postMessage({ type: "progress", load: id, received: offset, total: manifest.bytes } satisfies Progress);
      }
    } catch (error) {
      return unreadable(error);
    }
    const vocabulary = await kept.tokenizer();
    // T.329.2.2: the table of every layer's inputs is read from the kept file from now on, at once: a handle of it (the
    // only one it can have: another tab of this site may hold it). Where there is none, the table is read into the
    // memory with the rest
    if (weights.hole) {
      const reader = await keptModule!.openRows(kept.name);
      if (!reader) {
        console.warn("kept: the kept model's file cannot be read from at once here (another tab of this site may have it open): its table of every layer's inputs is read into the memory");
        weights.destroy();
        tableInMemoryFromNow(model);
        return loadConverted(model, signal, id);
      }
      weights.attach(reader);
    }
    took("download", since(started));
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    // template is for the page, not for the engine (see convert())
    const engineOptions = { ...manifest.options };
    delete engineOptions.template;
    await weightsDrained();  // (T.156: a model on the GPU alone: every byte of its layers there)
    adopted(weights.llama(tokenizer.buffer, { kernels: pythonKernels, disable: switchesNow(), ...engineOptions, ...model.options }));
    took("construct", since(constructStarted));
    return { template: manifest.options.template, keptIn: kept.where };
  } finally {
    weights.destroy();
    tokenizer?.buffer.destroy();
  }
}

// checkpoint: the weights (weightsBuffer), bytes long; tokenizer: a PyProxy of the converted tokenizer. Returns why
// nothing was kept, or undefined.
// T.156, kept: the file of a model on the GPU alone, written as the conversion came (kept.js's keeper)
async function keepConverted(model: Remote, checkpoint: Weights, bytes: number, tokenizer: Bytes, options: Options, signal: AbortSignal, kept?: Keeper) {
  const view = tokenizer.getBuffer("u8");
  const vocabulary = view.data.slice();
  view.release();
  const manifest = { id: model.id, name: model.name, repo: model.hf.repo, revision: model.hf.revision, bytes, options, saved: Date.now() };
  // under the bits it was converted to, which the worker may have chosen (T.115)
  const converted = { ...model, conversion: { ...model.conversion, dtype: options.dtype } };
  // T.136: what this model was kept as before its source changed is never used again, and takes the room it needs
  for (const old of await keptModule!.replaced(model).catch(() => [])) await keptModule!.forget(old).catch(() => {});
  // slice() copies: the memory it comes from may grow (and so move) while an await waits
  if (checkpoint.direct) return kept ? kept.finish(manifest, vocabulary) : "the weights went to the GPU alone, and there is no file system here to keep them in as they came";
  // (T.329.2.2: and a model whose table of every layer's inputs is read from the file it is kept in as it came)
  if (checkpoint.hole && kept) return kept.finish(manifest, vocabulary);
  return keptModule!.keep(converted, manifest, (begin, end) => checkpoint.slice(begin, end), vocabulary, signal);
}

/** The converter's window (llama2_convert), fetched, placed and imported when a model is first converted: most
 * visitors never convert anything. (T.367.2: its Python, a window and its parts, is one archive, converter.zip.) */
export async function converter(signal?: AbortSignal) {
  if (!llama2_convert) {
    // (T.397) and jinja2 with it, for the model's chat template: the conversion goes on without it where it does not come
    const jinja = templatePackage(pyodide);
    const archive = "converter.zip";
    await placePython(pyodide, built(archive, signal).then((res) => res.arrayBuffer()), archive);
    await jinja;
    llama2_convert = pyodide.pyimport("llama2_convert");
  }
  return llama2_convert;
}

export async function convert(model: OfHuggingFace, signal: AbortSignal, id: number) {
  const remote = typeof model.hf.repo === "string";
  // T.363.5: a model whose conversion keeps its GGUF's blocks as they are (conversion.blocks) runs on the kernels of
  // those, fetched here for it before anything of it is placed (the memory is chosen in the middle of the conversion).
  // A load that does not run on the int8 kernels gets the blocks widened, as a model without the word: another
  // conversion, kept under another name (kept.ts)
  blocksWidenedNow(Boolean(model.conversion?.blocks) && !runsPacked());
  if (model.conversion?.blocks) {
    if (runsPacked()) await blockKernels();
    else model = { ...model, conversion: { ...model.conversion, blocks: false } };
  }
  // (asked for when a conversion first needs it: a module of its own, _astro/kept.<hash>.js. T.367.4: where it is not
  // there because the site was updated since this page was opened, that is what is said)
  keptModule ??= await named(import.meta.url, () => import("../kept.ts"));
  const kept = remote ? await loadConverted(model as Remote, signal, id) : undefined;
  if (kept && !kept.miss) {
    return { fromCache: true, keptIn: kept.keptIn, template: kept.template };
  }
  const keptMiss = kept?.miss;
  await converter(signal);
  const started = performance.now();
  // T.156: a model that goes on the GPU alone is kept as it comes (nothing holds its weights whole afterwards): a file
  // opened for its int8 conversion (T.232: or its ternary one, where the page asked for that) where it may (the choice
  // is the sink's, once the header is known), let go otherwise
  const keptDtype = model.conversion?.dtype ?? "int8";
  // (T.329.2.2: and on the kernels of forward.ts, where a table of every layer's inputs stays in that file, which the
  // engine reads its rows from: whether the model has one is known once its header is)
  const mayKeep = remote && (gpuOnlyPossible(keptDtype) || runsPacked());
  // (T.380.2: a file that could not be opened is said: the conversion goes on, and a model on the GPU alone is then not
  // kept; T.521: and a table of every layer's inputs is held in the memory)
  let keep = mayKeep ? await keptModule!.keeper({ ...(model as Remote), conversion: { ...model.conversion, dtype: keptDtype } }).catch((error) => {
    console.warn(`kept: no file to keep this model in as it comes (${told(error)}): it is not kept, and a model's table of every layer's inputs is held in the memory`);
    return undefined;
  }) : undefined;
  const into = checkpointSink(keep), { sink } = into;
  let keptAsItCame = false;  // keep went to keepConverted, which keeps it or lets it go
  let steps: Conduct | undefined, conversion: Conversion | undefined, quantizeRows: Proxy | undefined, readers: Proxy | undefined, failed;
  // T.403: whatever ends the conversion (its end, a file that is not there, the line, a refusal of the converter's, a
  // load cancelled at any moment), the one finally below lets go of the file opened to keep it in and of what the
  // converter holds. The file's handle is the only one its file can have: left open, the next conversion of the
  // model got none and was not kept.
  try {
    // T.115: no bits asked for (weightsFor() in src/models.ts asks for six only where the device says it has too little
    // memory): int8 where its forward pass fits a 32-bit memory or the browser has a 64-bit one, six bits where neither
    // (T.133), once the header is known
    // (T.329.2.2: a table of every layer's inputs not counted where it will stay in the file kept as it comes)
    const converting = { ...model.conversion, dtype: model.conversion?.dtype ?? ((header: Parameters<typeof automaticBits>[0], form: Parameters<typeof automaticBits>[1],
      sizes: Parameters<typeof automaticBits>[2]) => automaticBits(header, form, sizes, Boolean(keep))) };
    // T.89: quantize() on the SIMD kernels, the same bytes six times faster (none with ?without=kernels); and the
    // readers of the types a file stores its tensors in, the same float32 several times faster: whichever of them has
    // a kernel, by the type's name (the converter's table of them says which. T.123: bfloat16; T.136: GGUF's Q8_0;
    // T.273: the two ternary types, PQ2_0 and PTQ1_0)
    const onKernels = pythonKernels && !switchesNow().includes("kernels");
    quantizeRows = onKernels ? llama2_numpy.kernel_quantizer(pythonKernels!) : undefined;
    readers = onKernels ? llama2_convert!.kernel_readers(pythonKernels!) : undefined;
    const make = { ...converting, sink, quantize_rows: quantizeRows, readers };
    // T.119: the page hears both how much has arrived (and how fast) and how much is converted. With the converted
    // share alone, a slow line looked like a slow conversion: the share moves as fast as the bytes arrive (8 MB/s
    // from Japan, T.123), the conversion itself is about 500 MB/s
    let spent = 0, toldAt = 0, size: number, arrived = 0, converted = 0, firstAt = 0, firstBytes = 0;
    const tell = (now = performance.now()) => {
      if (now - toldAt < 250 && converted < 1) {
        return;
      }
      toldAt = now;
      const perSecond = firstAt && now > firstAt + 1000 ? ((arrived - firstBytes) / (now - firstAt)) * 1000 : undefined;
      postMessage({ type: "progress", load: id, received: arrived, total: size, converted, perSecond } satisfies Progress);
    };
    const progress: Arriving = {
      of(total) {
        size = total;
      },
      arriving(received) {
        if (!firstAt) {
          [firstAt, firstBytes] = [performance.now(), received];
        }
        arrived = received;
        tell();
      },
      converting(feed) {
        // T.84: the time Python spends converting, apart from the time spent waiting for the download
        const began = performance.now();
        converted = feed();
        spent += performance.now() - began;
        tell();
      },
    };
    // T.374.2.1: Python conducts (src/python/convert/conduct.py), the worker answers (conduct.ts): from huggingface.co, or
    // (T.374.2.2) from the Files of a folder of the visitor's disk, which the model has beside the names of its files
    const { files, ...listed } = model.hf;
    steps = conductOf(listed, make);
    conversion = await answered(steps, remote ? fromHub(model.hf, signal, progress) : fromFolder(files!, signal, progress), signal);
    took("download", since(started));
    took("convert", spent / 1000);

    const constructStarted = performance.now();
    // every one of these proxies keeps its Python object alive: none may be left behind (the checkpoint went to
    // weights, through the sink)
    const proxies: [Dict<Options>, Bytes, Dict<Record<string, Widened>>] = [conversion.options, conversion.tokenizer, conversion.widened];
    let kept, template, widened;
    try {
      const options = proxies[0].toJs({ dict_converter: Object.fromEntries });
      widened = proxies[2].toJs({ dict_converter: Object.fromEntries });
      // template is for the page (the format of one turn), not for the engine
      ({ template } = options);
      const engineOptions = { ...options };
      delete engineOptions.template;
      await weightsDrained();  // (T.156)
      adopted(into.weights!.llama(proxies[1], { kernels: pythonKernels, disable: switchesNow(), ...engineOptions, ...model.options }));
      took("construct", since(constructStarted));
      // (the review of T.156: a model not on the GPU alone is kept from its memory, as before (keep()): the file opened
      // for it as it came goes first. Its open handle refused keep()'s, and its drop() in the finally below removed what
      // keep() wrote: every conversion of a browser with a GPU adapter was kept no more, run 36344754761)
      if (keep && !into.weights!.direct && !into.weights!.hole) {
        await keep.drop();
        keep = undefined;
      }
      if (remote) {
        postMessage({ type: "status", load: id, text: `${model.name}: keeping the converted model...` } satisfies Status);
        keptAsItCame = Boolean((into.weights!.direct || into.weights!.hole) && keep);
        kept = await keepConverted(model as Remote, into.weights!, into.bytes, proxies[1], options, signal, into.weights!.direct || into.weights!.hole ? keep : undefined);
      }
    } finally {
      proxies.forEach((proxy) => proxy.destroy());
    }
    return { fromCache: false, notKept: kept, keptMiss, template, widened };
  } catch (error) {
    failed = { error };
    throw error;
  } finally {
    // T.407: every one of these, whatever the others throw (the file kept as it came, T.156, where it was not the one
    // kept; the place of the weights: the engine keeps what it needs of the checkpoint alive, the rest goes with it,
    // and a feed that failed leaves no Python buffer of the model's size behind, T.145; the conduct where it stands,
    // closed (it holds the conversion it was making), then its proxy; the conversion; the kernels' two). What the
    // conversion itself failed with is what is thrown, and never what letting go failed with after it (a cancelled
    // load stays one); after a conversion that ended well, the first of those failures is
    let wrong;
    for (const release of [() => (keptAsItCame ? undefined : keep?.drop()), () => into.release(), () => steps?.return(), () => steps?.destroy(),
      () => conversion?.destroy(), () => quantizeRows?.destroy(), () => readers?.destroy()]) {
      try {
        await release();
      } catch (error) {
        wrong ??= { error };
      }
    }
    if (wrong && !failed) {
      throw wrong.error;
    }
  }
}
