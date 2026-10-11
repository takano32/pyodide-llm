// worker/convert.js (T350): a Hugging Face model converted in here as it arrives (src/python/llama2_convert.py), and what
// a conversion made kept for the next visit and read back (kept.js).
// T374.2.1: the conversion is conducted by Python (src/python/convert/conduct.py) and answered by conduct.js: a model of
// huggingface.co from there, and (T374.2.2) a folder of the visitor's disk from its Files.
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const { state, loadSeconds } = await import(new URL(`state.js${new URL(import.meta.url).search}`, import.meta.url));
const { since } = await import(new URL(`clock.js${new URL(import.meta.url).search}`, import.meta.url));
const { pythonBuffer, automaticBits, weightsBuffer, gpuOnlyPossible, weightsRoom, weightsDrained, checkpointSink } =
  await import(new URL(`weights.js${new URL(import.meta.url).search}`, import.meta.url));
const { conductOf, answered, fromHub, fromFolder } = await import(new URL(`conduct.js${new URL(import.meta.url).search}`, import.meta.url));
const { HEADER_BYTES, headerInts } = await import(new URL(`sources.js${new URL(import.meta.url).search}`, import.meta.url));
const { templatePackage, pythonArchive, placePython } = await import(new URL(`pyodide.js${new URL(import.meta.url).search}`, import.meta.url));

// What a conversion made is kept for the next visit (kept.js): in the origin private file system where there is one
// (T99), else in the Cache API. The original is twice as large, and fetching and converting it again on every visit
// would be no way to use a model. The page lists what is kept and deletes it.
// Returns what the page needs of it, or { miss } with why nothing kept could be used (for tests/e2e.mjs).
async function loadConverted(model, signal, id) {
  let kept;
  try {
    // under the bits asked for, or either the worker may choose (T115), by this converter (T116)
    kept = await state.keptModule.openKept(model);
  } catch (error) {
    return { miss: `could not open what is kept: ${error.message ?? error}` };
  }
  if (!kept) {
    return { miss: "nothing kept for this model" };
  }
  const { manifest } = kept;
  const started = performance.now();
  // the first part holds the header, which the memory is chosen by (T115)
  const parts = kept.parts()[Symbol.asyncIterator]();
  const unreadable = (error) => {
    if (signal.aborted) {
      throw error;
    }
    return { miss: `could not read what is kept: ${error.message ?? error}` };  // evicted: convert again
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
  const weights = weightsBuffer(manifest.bytes, headerInts(part.value), manifest.options);
  let tokenizer;
  try {
    let offset = 0;
    try {
      for (; !part.done; part = await parts.next()) {
        signal.throwIfAborted();
        weights.write(offset, part.value);
        await weightsRoom();  // (T156)
        offset += part.value.length;
        postMessage({ type: "progress", load: id, received: offset, total: manifest.bytes });
      }
    } catch (error) {
      return unreadable(error);
    }
    const vocabulary = await kept.tokenizer();
    loadSeconds.download = since(started);
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    // template is for the page, not for the engine (see convert())
    const engineOptions = { ...manifest.options };
    delete engineOptions.template;
    await weightsDrained();  // (T156: a model on the GPU alone: every byte of its layers there)
    state.llama = weights.llama(tokenizer.buffer, { kernels: state.kernels, disable: state.disabled, ...engineOptions, ...model.options });
    loadSeconds.construct = since(constructStarted);
    return { template: manifest.options.template, keptIn: kept.where };
  } finally {
    weights.destroy();
    tokenizer?.buffer.destroy();
  }
}

// checkpoint: the weights (weightsBuffer), bytes long; tokenizer: a PyProxy of the converted tokenizer. Returns why
// nothing was kept, or undefined.
// T156, kept: the file of a model on the GPU alone, written as the conversion came (kept.js's keeper)
async function keepConverted(model, checkpoint, bytes, tokenizer, options, signal, kept) {
  const view = tokenizer.getBuffer("u8");
  const vocabulary = view.data.slice();
  view.release();
  const manifest = { id: model.id, name: model.name, repo: model.hf.repo, revision: model.hf.revision, bytes, options, saved: Date.now() };
  // under the bits it was converted to, which the worker may have chosen (T115)
  const converted = { ...model, conversion: { ...model.conversion, dtype: options.dtype } };
  // T136: what this model was kept as before its source changed is never used again, and takes the room it needs
  for (const old of await state.keptModule.replaced(model).catch(() => [])) await state.keptModule.forget(old).catch(() => {});
  // slice() copies: the memory it comes from may grow (and so move) while an await waits
  if (checkpoint.direct) return kept ? kept.finish(manifest, vocabulary) : "the weights went to the GPU alone, and there is no file system here to keep them in as they came";
  return state.keptModule.keep(converted, manifest, (begin, end) => checkpoint.slice(begin, end), vocabulary, signal);
}

/** The converter's window (state.llama2_convert), fetched, placed and imported when a model is first converted: most
 * visitors never convert anything. (T367.2: its Python, a window and its parts, is one archive, converter.zip.) */
export async function converter(signal) {
  if (!state.llama2_convert) {
    // (T397) and jinja2 with it, for the model's chat template: the conversion goes on without it where it does not come
    const jinja = templatePackage(state.pyodide);
    await placePython(state.pyodide, pythonArchive("converter.zip", signal));
    await jinja;
    state.llama2_convert = state.pyodide.pyimport("llama2_convert");
  }
  return state.llama2_convert;
}

export async function convert(model, signal, id) {
  const remote = typeof model.hf.repo === "string";
  // with the ?v=<build> of this worker, like every file it reads (AGENTS.md)
  state.keptModule ??= await import(new URL(`../kept.js${self.location.search}`, import.meta.url));
  const kept = remote ? await loadConverted(model, signal, id) : undefined;
  if (kept && !kept.miss) {
    return { fromCache: true, keptIn: kept.keptIn, template: kept.template };
  }
  const keptMiss = kept?.miss;
  await converter(signal);
  const started = performance.now();
  // T156: a model that goes on the GPU alone is kept as it comes (nothing holds its weights whole afterwards): a file
  // opened for its int8 conversion (T232: or its ternary one, where the page asked for that) where it may (the choice
  // is the sink's, once the header is known), let go otherwise
  const keptDtype = model.conversion?.dtype ?? "int8";
  const mayKeep = remote && ["int8", "ternary"].includes(keptDtype) && gpuOnlyPossible(keptDtype);
  let keep = mayKeep ? await state.keptModule.keeper({ ...model, conversion: { ...model.conversion, dtype: keptDtype } }).catch(() => undefined) : undefined;
  const into = checkpointSink(keep), { sink } = into;
  let keptAsItCame = false;  // keep went to keepConverted, which keeps it or lets it go
  let steps, conversion, quantizeRows, readers, failed;
  // T403: whatever ends the conversion (its end, a file that is not there, the line, a refusal of the converter's, a
  // load cancelled at any moment), the one finally below lets go of the file opened to keep it in and of what the
  // converter holds. The file's handle is the only one its file can have: left open, the next conversion of the
  // model got none and was not kept.
  try {
    // T115: no bits asked for (weightsFor() in src/models.js asks for six only where the device says it has too little
    // memory): int8 where its forward pass fits a 32-bit memory or the browser has a 64-bit one, six bits where neither
    // (T133), once the header is known
    const converting = { ...model.conversion, dtype: model.conversion?.dtype ?? automaticBits };
    // T89: quantize() on the SIMD kernels, the same bytes six times faster (none with ?without=kernels); and the
    // readers of the types a file stores its tensors in, the same float32 several times faster: whichever of them has
    // a kernel, by the type's name (the converter's table of them says which. T123: bfloat16; T136: GGUF's Q8_0;
    // T273: the two ternary types, PQ2_0 and PTQ1_0)
    const onKernels = state.kernels && !state.disabled.includes("kernels");
    quantizeRows = onKernels ? state.llama2_numpy.kernel_quantizer(state.kernels) : undefined;
    readers = onKernels ? state.llama2_convert.kernel_readers(state.kernels) : undefined;
    const make = { ...converting, sink, quantize_rows: quantizeRows, readers };
    // T119: the page hears both how much has arrived (and how fast) and how much is converted. With the converted
    // share alone, a slow line looked like a slow conversion: the share moves as fast as the bytes arrive (8 MB/s
    // from Japan, T123), the conversion itself is about 500 MB/s
    let spent = 0, told = 0, size, arrived = 0, converted = 0, firstAt = 0, firstBytes = 0;
    const tell = (now = performance.now()) => {
      if (now - told < 250 && converted < 1) {
        return;
      }
      told = now;
      const perSecond = firstAt && now > firstAt + 1000 ? ((arrived - firstBytes) / (now - firstAt)) * 1000 : undefined;
      postMessage({ type: "progress", load: id, received: arrived, total: size, converted, perSecond });
    };
    const progress = {
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
        // T84: the time Python spends converting, apart from the time spent waiting for the download
        const began = performance.now();
        converted = feed();
        spent += performance.now() - began;
        tell();
      },
    };
    // T374.2.1: Python conducts (src/python/convert/conduct.py), the worker answers (conduct.js): from huggingface.co, or
    // (T374.2.2) from the Files of a folder of the visitor's disk, which the model has beside the names of its files
    const { files, ...listed } = model.hf;
    steps = conductOf(listed, make);
    conversion = await answered(steps, remote ? fromHub(model.hf, signal, progress) : fromFolder(files, signal, progress), signal);
    loadSeconds.download = since(started);
    loadSeconds.convert = spent / 1000;

    const constructStarted = performance.now();
    // every one of these proxies keeps its Python object alive: none may be left behind (the checkpoint went to
    // weights, through the sink)
    const proxies = [conversion.options, conversion.tokenizer];
    let kept, template;
    try {
      const options = proxies[0].toJs({ dict_converter: Object.fromEntries });
      // template is for the page (the format of one turn), not for the engine
      ({ template } = options);
      const engineOptions = { ...options };
      delete engineOptions.template;
      await weightsDrained();  // (T156)
      state.llama = into.weights.llama(proxies[1], { kernels: state.kernels, disable: state.disabled, ...engineOptions, ...model.options });
      loadSeconds.construct = since(constructStarted);
      // (the review of T156: a model not on the GPU alone is kept from its memory, as before (keep()): the file opened
      // for it as it came goes first. Its open handle refused keep()'s, and its drop() in the finally below removed what
      // keep() wrote: every conversion of a browser with a GPU adapter was kept no more, run 36344754761)
      if (keep && !into.weights.direct) {
        await keep.drop();
        keep = undefined;
      }
      if (remote) {
        postMessage({ type: "status", load: id, text: `${model.name}: keeping the converted model...` });
        keptAsItCame = Boolean(into.weights.direct && keep);
        kept = await keepConverted(model, into.weights, into.bytes, proxies[1], options, signal, into.weights.direct ? keep : undefined);
      }
    } finally {
      proxies.forEach((proxy) => proxy.destroy());
    }
    return { fromCache: false, notKept: kept, keptMiss, template };
  } catch (error) {
    failed = { error };
    throw error;
  } finally {
    // T407: every one of these, whatever the others throw (the file kept as it came, T156, where it was not the one
    // kept; the place of the weights: the engine keeps what it needs of the checkpoint alive, the rest goes with it,
    // and a feed that failed leaves no Python buffer of the model's size behind, T145; the conduct where it stands,
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
