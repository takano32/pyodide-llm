// worker/convert.js (T350): a Hugging Face model converted in here as it arrives (public/llama2_convert.py), and what
// a conversion made kept for the next visit and read back (kept.js).
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const { state, loadSeconds } = await import(new URL(`state.js${new URL(import.meta.url).search}`, import.meta.url));
const { since } = await import(new URL(`clock.js${new URL(import.meta.url).search}`, import.meta.url));
const { pythonBuffer, automaticBits, weightsBuffer, gpuOnlyPossible, weightsRoom, weightsDrained, checkpointSink } =
  await import(new URL(`weights.js${new URL(import.meta.url).search}`, import.meta.url));
const { HF_HEADER_BYTES, sized, refused, fetchRange, inOrder } =
  await import(new URL(`ranges.js${new URL(import.meta.url).search}`, import.meta.url));
const { HEADER_BYTES, headerInts } = await import(new URL(`sources.js${new URL(import.meta.url).search}`, import.meta.url));
const { templatePackage } = await import(new URL(`pyodide.js${new URL(import.meta.url).search}`, import.meta.url));

// A model without a model.safetensors is split over several files (model-00001-of-00002.safetensors, ...), or
// published under the name of a shard even when there is only one (T78). model.safetensors.index.json says which
// file every tensor is in: the files, in the order of their names, which is the order they are fed in (T105).
function shardsOf(index) {
  try {
    const map = (typeof index === "string" ? JSON.parse(index) : index)?.weight_map;
    return [...new Set(Object.values(map ?? {}))].sort();
  } catch {
    return [];
  }
}

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

export async function convert(model, signal, id) {
  const remote = typeof model.hf.repo === "string";
  // with the ?v=<build> of this worker, like every file it reads (AGENTS.md)
  state.keptModule ??= await import(new URL(`../kept.js${self.location.search}`, import.meta.url));
  const kept = remote ? await loadConverted(model, signal, id) : undefined;
  if (kept && !kept.miss) {
    return { fromCache: true, keptIn: kept.keptIn, template: kept.template };
  }
  const keptMiss = kept?.miss;
  if (!state.llama2_convert) {
    // fetched when it is first needed: most visitors never convert anything
    // (T347: the converter is a window and its parts, python.js's list; each with this worker's ?v=<build>)
    const { placePython } = await import(new URL(`../python.js${self.location.search}`, import.meta.url));
    // (T397) and jinja2 with it, for the model's chat template: the conversion goes on without it where it does not come
    const jinja = templatePackage(state.pyodide);
    await placePython(state.pyodide, "llama2_convert", async (name) => {
      const res = await fetch(new URL(`../${name}${self.location.search}`, import.meta.url), { signal });
      if (!res.ok) {
        throw new Error(`Could not fetch ${name}: ${res.status}`);
      }
      return res.text();
    });
    await jinja;
    state.llama2_convert = state.pyodide.pyimport("llama2_convert");
  }
  const started = performance.now();
  const at = (name) => `https://huggingface.co/${model.hf.repo}/resolve/${model.hf.revision}/${name}`;
  const text = async (url) => {
    const res = await fetch(url, { signal });
    if (!res.ok) {
      throw refused(url, res);
    }
    return res;
  };
  let first, size, base, conversion, shards;
  // T156: a model that goes on the GPU alone is kept as it comes (nothing holds its weights whole afterwards): a file
  // opened for its int8 conversion (T232: or its ternary one, where the page asked for that) where it may (the choice
  // is the sink's, once the header is known), let go otherwise
  const keptDtype = model.conversion?.dtype ?? "int8";
  const mayKeep = remote && ["int8", "ternary"].includes(keptDtype) && gpuOnlyPossible(keptDtype);
  let keep = mayKeep ? await state.keptModule.keeper({ ...model, conversion: { ...model.conversion, dtype: keptDtype } }).catch(() => undefined) : undefined;
  const into = checkpointSink(keep), { sink } = into;
  let keptAsItCame = false;  // keep went to keepConverted, which keeps it or lets it go
  // T115: no bits asked for (weightsFor() in src/models.js asks for six only where the device says it has too little
  // memory): int8 where its forward pass fits a 32-bit memory or the browser has a 64-bit one, six bits where neither
  // (T133), once the header is known
  const converting = { ...model.conversion, dtype: model.conversion?.dtype ?? automaticBits };
  // T89: quantize() on the SIMD kernels, the same bytes six times faster (none with ?without=kernels); and the
  // readers of the types a file stores its tensors in, the same float32 several times faster: whichever of them has
  // a kernel, by the type's name (the converter's table of them says which. T123: bfloat16; T136: GGUF's Q8_0;
  // T273: the two ternary types, PQ2_0 and PTQ1_0)
  const onKernels = state.kernels && !state.disabled.includes("kernels");
  const quantizeRows = onKernels ? state.llama2_numpy.kernel_quantizer(state.kernels) : undefined;
  const readers = onKernels ? state.llama2_convert.kernel_readers(state.kernels) : undefined;
  // T136: a GGUF's weights with the vocabulary and config.json of the original repository (a sentencepiece vocabulary
  // in a GGUF says neither its kind nor its normalization): those files come from there, the weights from the GGUF
  const vocabulary = remote ? model.hf.vocabulary : undefined;
  const from = (name) => vocabulary ? `https://huggingface.co/${vocabulary.repo}/resolve/${vocabulary.revision}/${name}` : at(name);
  if (remote && model.hf.weights.endsWith(".gguf") && !vocabulary) {
    // T74: a GGUF holds the configuration and the vocabulary in its header, before the tensors: no config.json and
    // no tokenizer to fetch. The header is a few megabytes (the vocabulary), so it is fetched in growing pieces
    // until the converter can read all of it.
    for (let bytes = 4 * HF_HEADER_BYTES; ; bytes *= 4) {
      ({ bytes: first, total: size } = await sized(at(model.hf.weights), await fetchRange(at(model.hf.weights), 0, bytes, signal), signal));
      try {
        conversion = state.llama2_convert.Conversion.from_gguf.callKwargs(first, { ...converting, sink, quantize_rows: quantizeRows, readers });
        break;
      } catch (error) {
        if (error.type !== "Incomplete" || bytes >= size) {
          into.release();
          throw error;
        }
      }
    }
    base = conversion.base;
  } else {
    // the beginning of a file: 8 bytes that say how long the JSON header is, then the header
    const head = async (name) => {
      let { bytes, total } = remote ? await sized(at(name), await fetchRange(at(name), 0, HF_HEADER_BYTES, signal), signal)
        : { bytes: new Uint8Array(await name.slice(0, HF_HEADER_BYTES).arrayBuffer()), total: name.size };
      const headerBytes = bytes.length >= 8 ? Number(new DataView(bytes.buffer, bytes.byteOffset).getBigUint64(0, true)) : -1;
      if (!(headerBytes >= 2 && headerBytes <= 100e6)) {
        throw new Error("This is not a safetensors file.");
      }
      const start = 8 + headerBytes;
      if (start > bytes.length) {
        bytes = remote ? (await fetchRange(at(name), 0, start, signal)).bytes : new Uint8Array(await name.slice(0, start).arrayBuffer());
      }
      return { name, header: new TextDecoder().decode(bytes.subarray(8, start)), base: start, total };
    };
    let header;
    const config = remote ? await (await text(from(model.hf.config ?? "config.json"))).text() : await model.hf.config.text();
    if (vocabulary) {
      // the GGUF's header as a safetensors one, once the original's config.json agrees with it; the header is a few
      // megabytes (the GGUF's own vocabulary), fetched in growing pieces as above
      for (let bytes = 4 * HF_HEADER_BYTES; ; bytes *= 4) {
        ({ bytes: first, total: size } = await sized(at(model.hf.weights), await fetchRange(at(model.hf.weights), 0, bytes, signal), signal));
        try {
          const made = state.llama2_convert.gguf_weights(first, config);
          [header, base] = made.toJs();
          made.destroy();
          break;
        } catch (error) {
          if (error.type !== "Incomplete" || bytes >= size) {
            throw error;
          }
        }
      }
    } else {
      try {
        ({ header, base, total: size } = await head(model.hf.weights));
      } catch (error) {
        if (!remote) {
          throw error;
        }
        signal.throwIfAborted();
        const index = await text(at(`${model.hf.weights}.index.json`)).then((res) => res.text())
          .catch(() => { throw error; });
        const files = shardsOf(index);
        if (!files.length) {
          throw error;
        }
        if (files.length === 1) {
          model = { ...model, hf: { ...model.hf, weights: files[0] } };
          ({ header, base, total: size } = await head(files[0]));
        } else {
          // T105: the shards' headers joined into the header of one file made of their data one after another, which
          // the converter reads as it reads any file. Each shard is then fed from its own base, the next after it.
          shards = [];
          for (const name of files) {
            shards.push(await head(name));
          }
          const joined = state.llama2_convert.joined_shards(shards.map((shard) => shard.header));
          let lengths;
          [header, lengths] = joined.toJs();
          joined.destroy();
          shards.forEach((shard, i) => { shard.length = lengths[i]; });
          base = 0;
          size = shards.reduce((sum, shard) => sum + shard.length, 0);
        }
      }
    }
    // The format of one turn, when the model publishes a chat_template (T73). It is small, and a model without
    // one (or with one the converter cannot read) simply keeps the format src/models.js has for it.
    const tokenizerConfig = await (remote ? text(from("tokenizer_config.json")).then((r) => r.text())
      : model.hf.tokenizerConfig?.text() ?? Promise.resolve("")).catch(() => "");
    // T127: newer repositories keep the template in chat_template.jinja instead. Asked for only where
    // tokenizer_config.json has none: most repositories have no such file, and WebKit reports each 404 as an error
    const hasTemplate = (() => {
      try {
        return Boolean(JSON.parse(tokenizerConfig).chat_template);
      } catch {
        return false;
      }
    })();
    const chatTemplate = hasTemplate ? "" : await (remote ? text(from("chat_template.jinja")).then((r) => r.text())
      : model.hf.chatTemplate?.text() ?? Promise.resolve("")).catch(() => "");
    // For a repository nobody has looked at (?hf=), the tokenizer is whichever of these it has and the converter can read.
    // Where none will do, the converter's refusal of one that is there says why; a file that is not there (a 404 of the
    // first candidate) is said only where no other was there either (T144). Only a 404 moves on: a fetch that failed
    // otherwise (the line, 429, 5xx) must neither hide behind a later refusal nor let a later candidate be converted
    // and kept in its place (the review of T144)
    let refusal, missing;
    for (const candidate of [].concat(vocabulary?.tokenizer ?? model.hf.tokenizer)) {
      let tokenizer;
      try {
        tokenizer = new Uint8Array(remote ? await (await text(from(candidate))).arrayBuffer() : await candidate.arrayBuffer());
      } catch (error) {
        if (signal.aborted || !remote || error.status !== 404) {
          throw error;
        }
        missing ??= error;
        continue;
      }
      signal.throwIfAborted();
      try {
        conversion = state.llama2_convert.Conversion.callKwargs(header, base, config, tokenizer, remote ? candidate : candidate.name,
          { start: base, tokenizer_config: tokenizerConfig, chat_template: chatTemplate || null, ...converting, sink,
            quantize_rows: quantizeRows, readers });
        break;
      } catch (error) {
        refusal ??= error;
      }
    }
    if (!conversion) {
      into.release();
      throw refusal ?? missing;
    }
  }
  let template;
  try {
    // T119: the page hears both how much has arrived (and how fast) and how much is converted. With the converted
    // share alone, a slow line looked like a slow conversion: the share moves as fast as the bytes arrive (8 MB/s
    // from Japan, T123), the conversion itself is about 500 MB/s
    let converting = 0, told = 0, arrived = 0, converted = 0, firstAt = 0, firstBytes = 0;
    const tell = (now = performance.now()) => {
      if (now - told < 250 && converted < 1) {
        return;
      }
      told = now;
      const perSecond = firstAt && now > firstAt + 1000 ? ((arrived - firstBytes) / (now - firstAt)) * 1000 : undefined;
      postMessage({ type: "progress", load: id, received: arrived, total: size, converted, perSecond });
    };
    const arriving = (received) => {
      if (!firstAt) {
        [firstAt, firstBytes] = [performance.now(), received];
      }
      arrived = received;
      tell();
    };
    const feed = (bytes) => {
      // T84: the time Python spends converting, apart from the time spent waiting for the download
      const began = performance.now();
      converted = conversion.feed(bytes);
      converting += performance.now() - began;
      tell();
    };
    if (shards) {
      // what has arrived counts across the shards, one after another (the review of T119: it went back to 0 with each)
      let before = 0;
      for (const shard of shards) {
        await inOrder(at(shard.name), shard.base, shard.base + shard.length, feed, signal,
          (received) => arriving(before + received - shard.base));
        before += shard.length;
      }
    } else if (remote) {
      await inOrder(at(model.hf.weights), base, size, feed, signal, arriving);
    } else {
      const reader = model.hf.weights.slice(base).stream().getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (signal.aborted) {
          reader.cancel();
          signal.throwIfAborted();
        }
        feed(value);
        await weightsRoom();  // (T156)
      }
    }
    conversion.finish();
    loadSeconds.download = since(started);
    loadSeconds.convert = converting / 1000;

    const constructStarted = performance.now();
    // every one of these proxies keeps its Python object alive: none may be left behind (the checkpoint went to
    // weights, through the sink)
    const proxies = [conversion.options, conversion.tokenizer];
    let kept;
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
  } finally {
    // (T156: the file kept as it came is let go where it was not the one kept)
    if (!keptAsItCame) await keep?.drop();
    // the engine keeps what it needs of the checkpoint alive, the rest goes with this; and a feed that failed (the
    // line, a refusal on the way) leaves no Python buffer of the model's size behind (T145)
    into.release();
    conversion.destroy();
    quantizeRows?.destroy();
    readers?.destroy();
  }
}
