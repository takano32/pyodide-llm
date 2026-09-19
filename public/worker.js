// Pyodide lives in this worker, so the page stays responsive while the model is loading and generating.
// model is an entry of src/models.js, or one with {file, tokenizerFile}: two files of the visitor's own disk,
// which are read where they are and go nowhere. Or one with {hf: {weights, config, tokenizer}}: a Hugging Face
// model, of that disk (Files) or of huggingface.co ({repo, revision} and file names), which
// public/llama2_convert.py converts in here as it arrives.
// The page sends   {type: "init", search, model, load},  {type: "load", model, load},
//                  {type: "generate", prompt, ...options}  and  {type: "stop"}
// and receives     {type: "status" | "progress" | "ready" | "token" | "done" | "error", ...}
// load is a number the page counts up: a newer load cancels the one that is going on, and whatever this worker
// reports about a load carries its number, so that the page can tell a late report of a cancelled one.

// the version becomes part of a CDN URL, so accept nothing but a plain version number
const PYODIDE_VERSION_PATTERN = /^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$/;

// the latest release on npm (the "latest" tag never points at an alpha), or ?pyodide=<version> to force one
async function resolvePyodideVersion(search) {
  const forced = new URLSearchParams(search).get("pyodide");
  if (PYODIDE_VERSION_PATTERN.test(forced)) {
    return forced;
  }
  const res = await fetch("https://data.jsdelivr.com/v1/packages/npm/pyodide/resolved?specifier=latest");
  const version = (await res.json()).version;
  if (!PYODIDE_VERSION_PATTERN.test(version)) {
    throw new Error(`Unexpected Pyodide version: ${version}`);
  }
  return version;
}

// Checkpoints are deployed in parts of 8 MiB (see the Makefile). Several parts download at once, which is
// about twice as fast as one stream, and the download runs while Pyodide is still loading: until the Python
// buffer exists the chunks wait in a queue, after that every chunk is written straight into it.
const PART_BYTES = 8 * 1024 * 1024;
const CONNECTIONS = 8;

// GitHub Pages lets the browser keep a file for ten minutes only, so the parts also go into the Cache API: the
// next visit starts without downloading the model again. The size is part of the key, so a rebuilt model of
// another size is fetched anew. Without the Cache API (some private modes) this is a plain fetch.
// v2: llm-jp-3-150m got its whole context of 4096 tokens, which changed its header and not its size
const MODEL_CACHE = "models-v2";

// what an earlier version of this page stored
globalThis.caches?.delete("models-v1").catch(() => {});

async function fetchPart(url, model, signal) {
  const cache = await globalThis.caches?.open(MODEL_CACHE).catch(() => undefined);
  const key = `${url}?bytes=${model.bytes}`;
  const cached = await cache?.match(key);
  if (cached) {
    return cached;
  }
  const res = await fetch(url, { signal });
  if (res.ok && cache) {
    // stored while the other copy streams into Python; a full disk must not stop the download. A part that is
    // cancelled half way is not stored at all, the finished ones stay for the next time.
    cache.put(key, res.clone()).catch(() => {});
  }
  return res;
}

// parts of this checkpoint that were cached for another size are of no use any more
async function dropStaleParts(model) {
  const cache = await globalThis.caches?.open(MODEL_CACHE).catch(() => undefined);
  for (const request of (await cache?.keys()) ?? []) {
    const url = new URL(request.url);
    if (url.pathname.includes(`/models/${model.checkpoint}.`) && url.searchParams.get("bytes") !== String(model.bytes)) {
      cache.delete(request);
    }
  }
}

function download(model, signal, load) {
  const parts = Math.ceil(model.bytes / PART_BYTES);
  const queue = [];
  let sink, next = 0, received = 0, reported = -1;
  const connection = async () => {
    while (next < parts) {
      const part = next++;
      const res = await fetchPart(new URL(`models/${model.checkpoint}.${String(part).padStart(3, "0")}`, import.meta.url).href, model, signal);
      if (!res.ok) {
        throw new Error(`Could not fetch part ${part} of ${model.checkpoint}: ${res.status}`);
      }
      const reader = res.body.getReader();
      for (let offset = part * PART_BYTES; ;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        // a chunk that was already on its way when the load was cancelled: its buffer is gone
        signal.throwIfAborted();
        sink ? sink(offset, value) : queue.push([offset, value]);
        offset += value.length;
        received += value.length;
        // one message per percent is plenty
        const percent = Math.floor((received / model.bytes) * 100);
        if (percent !== reported) {
          reported = percent;
          postMessage({ type: "progress", load, received, total: model.bytes });
        }
      }
    }
  };
  const finished = Promise.all(Array.from({ length: Math.min(CONNECTIONS, parts) }, connection));
  // a load that is cancelled while Pyodide still loads never gets to into(): that is no unhandled rejection
  finished.catch(() => {});
  return {
    // write(offset, chunk) receives everything queued so far, and every later chunk
    async into(write) {
      sink = write;
      queue.splice(0).forEach(([offset, chunk]) => write(offset, chunk));
      await finished;
      if (received !== model.bytes) {
        throw new Error(`${model.checkpoint}: got ${received} bytes instead of ${model.bytes}`);
      }
    },
  };
}

// The same for a file of the visitor's own disk: read in chunks straight into the Python buffer, never as a whole.
function readFile(model, signal, load) {
  return {
    async into(write) {
      const reader = model.file.stream().getReader();
      let reported = -1;
      for (let offset = 0; ;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (signal.aborted) {
          reader.cancel();
          signal.throwIfAborted();
        }
        write(offset, value);
        offset += value.length;
        const percent = Math.floor((offset / model.bytes) * 100);
        if (percent !== reported) {
          reported = percent;
          postMessage({ type: "progress", load, received: offset, total: model.bytes });
        }
      }
    },
  };
}

// A llama2.c checkpoint at a URL (?checkpoint=&tokenizer=): range requests in parallel, written where they belong
function readUrl(model, signal, load) {
  return {
    async into(write) {
      let offset = 0, reported = -1;
      await inOrder(model.url.checkpoint, 0, model.bytes, (bytes) => {
        write(offset, bytes);
        offset += bytes.length;
        const percent = Math.floor((offset / model.bytes) * 100);
        if (percent !== reported) {
          reported = percent;
          postMessage({ type: "progress", load, received: offset, total: model.bytes });
        }
      }, signal);
    },
  };
}

// The legacy format carries no metadata, but its header fixes the size of a float32, a float16 and an int8 file.
// A file that is none of them is refused before it is read, and so is a tokenizer.bin of another vocabulary.
async function localOptions(model, vocabulary) {
  const first = model.file ? await model.file.slice(0, 28).arrayBuffer() : (await fetchRange(model.url.checkpoint, 0, 28, new AbortController().signal)).bytes.buffer;
  const header = pyodide.toPy([...new Int32Array(first)]);
  const pieces = pyodide.toPy(vocabulary);
  try {
    const dtype = llama2_numpy.checkpoint_dtype(header, model.bytes);
    llama2_numpy.check_tokenizer(pieces, header);
    return { ...model.options, dtype };
  } finally {
    header.destroy();
    pieces.destroy();
  }
}

let pyodide, llama2_numpy, llama2_convert, llama, kernels;
// init() as a promise: every load waits for it, also the one that replaces the first
let initialized;
// the AbortController of the load that is going on, and a promise that settles once it has cleaned up
let loading, unloaded = Promise.resolve();

// how long the load took, in seconds: Pyodide once per session, the other two per model. The download runs while
// Pyodide loads, so the two overlap and the page says so instead of adding them up.
const loadSeconds = {};
const since = (started) => (performance.now() - started) / 1000;

async function init(search) {
  const started = performance.now();
  const version = await resolvePyodideVersion(search);
  postMessage({ type: "status", text: `Loading Pyodide ${version}...` });
  const { loadPyodide } = await import(`https://cdn.jsdelivr.net/pyodide/v${version}/full/pyodide.mjs`);
  pyodide = await loadPyodide();
  await pyodide.loadPackage("numpy");

  // with the ?v=<build> of this worker, so that both always come from the same deployment
  const res = await fetch(new URL(`llama2_numpy.py${self.location.search}`, import.meta.url));
  if (!res.ok) {
    throw new Error(`Could not fetch llama2_numpy.py: ${res.status}`);
  }
  pyodide.FS.writeFile("llama2_numpy.py", await res.text());
  llama2_numpy = pyodide.pyimport("llama2_numpy");

  // The WASM SIMD kernels (kernels/*.ts), which llama2_numpy.py loads with ctypes. They are optional: without
  // them, or with ?kernel=off, NumPy does the math, several times slower.
  if (new URLSearchParams(search).get("kernel") !== "off") {
    for (const name of ["simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
      const kernel = await fetch(new URL(`${name}${self.location.search}`, import.meta.url)).catch(() => undefined);
      if (kernel?.ok) {
        pyodide.FS.writeFile(`/home/pyodide/${name}`, new Uint8Array(await kernel.arrayBuffer()));
        kernels = "/home/pyodide/simdkernel.so";
      }
    }
  }
  loadSeconds.pyodide = since(started);
}

// a Python bytearray that JavaScript fills in place
function pythonBuffer(size) {
  const buffer = pyodide.globals.get("bytearray")(size);
  const write = (offset, chunk) => {
    // the view is taken anew every time: it dies when the WebAssembly memory grows
    const view = buffer.getBuffer("u8");
    view.data.set(chunk, offset);
    view.release();
  };
  return { buffer, write };
}

// Every await in here may end with the AbortError of signal: a newer load has taken over, and this one must
// leave nothing behind, least of all a Python buffer as large as its model.
// A Hugging Face model, from the visitor's disk ({weights, config, tokenizer} are Files) or from huggingface.co
// ({repo, revision, weights, config, tokenizer} are names): model.safetensors arrives in the order of the file, a few
// megabytes at a time, and the Python code that builds the models of this site converts every tensor as it comes and
// writes it to its place in a buffer of the final size. Reading in the order of the output instead would mean
// hundreds of range requests, and each one takes a second.
const HF_PART_BYTES = 8 * 1024 * 1024;
const HF_CONNECTIONS = 6;
const HF_HEADER_BYTES = 512 * 1024;  // the JSON header of a safetensors file is a few dozen kilobytes

async function fetchRange(url, begin, end, signal) {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(url, { headers: { Range: `bytes=${begin}-${end - 1}` }, signal });
      if (res.status !== 206 && res.status !== 200) {
        throw new Error(`Could not fetch ${url}: ${res.status}`);
      }
      return { bytes: new Uint8Array(await res.arrayBuffer()), total: Number((res.headers.get("Content-Range") ?? "").split("/")[1]) };
    } catch (error) {
      if (signal.aborted || attempt === 2) {
        throw error;
      }
    }
  }
}

// feed(bytes) gets the file from position start to its end, in order, although the parts arrive as they like
async function inOrder(url, start, size, feed, signal) {
  const parts = Math.ceil((size - start) / HF_PART_BYTES);
  const arrived = new Map();
  let next = 0, fed = 0, waiting = [];
  const connection = async () => {
    while (next < parts) {
      // no more than two parts per connection wait in memory for an earlier one
      while (next - fed >= 2 * HF_CONNECTIONS) {
        await new Promise((resolve) => waiting.push(resolve));
      }
      if (next >= parts) {
        return;
      }
      const part = next++;
      const begin = start + part * HF_PART_BYTES;
      arrived.set(part, (await fetchRange(url, begin, Math.min(begin + HF_PART_BYTES, size), signal)).bytes);
      while (arrived.has(fed)) {
        signal.throwIfAborted();
        feed(arrived.get(fed));
        arrived.delete(fed++);
        // the conversion of a part takes a moment: let messages in
        await breathe();
      }
      waiting.splice(0).forEach((resolve) => resolve());
    }
  };
  await Promise.all(Array.from({ length: Math.min(HF_CONNECTIONS, parts) }, connection));
}

// What a conversion made is kept in the Cache API, in parts like the models of the site: the original is twice as
// large, and fetching and converting it again on every visit would be no way to use a model. One cache of its own,
// so that the page can list what is kept and delete it. The manifest is written last: parts without one are rubbish.
const CONVERTED_CACHE = "converted-v1";
const convertedKey = (model, name) => {
  const { dtype = "int8", max_seq_len = 4096 } = model.conversion ?? {};
  return `${self.location.origin}/converted/${encodeURIComponent(`${model.hf.repo}@${model.hf.revision}:${dtype}:${max_seq_len}`)}/${name}`;
};

async function loadConverted(model, signal, id) {
  const cache = await globalThis.caches?.open(CONVERTED_CACHE).catch(() => undefined);
  const manifest = await (await cache?.match(convertedKey(model, "manifest.json")))?.json();
  if (!manifest) {
    return false;
  }
  const started = performance.now();
  const weights = pythonBuffer(manifest.bytes);
  let tokenizer;
  try {
    for (let part = 0, offset = 0; part < manifest.parts; part++) {
      const stored = await cache.match(convertedKey(model, `part-${String(part).padStart(3, "0")}`));
      if (!stored) {
        return false;  // the browser has evicted a part: convert again
      }
      const bytes = new Uint8Array(await stored.arrayBuffer());
      signal.throwIfAborted();
      weights.write(offset, bytes);
      offset += bytes.length;
      postMessage({ type: "progress", load: id, received: offset, total: manifest.bytes });
    }
    const vocabulary = new Uint8Array(await (await cache.match(convertedKey(model, "tokenizer.bin"))).arrayBuffer());
    loadSeconds.download = since(started);
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    llama = llama2_numpy.Llama.callKwargs(weights.buffer, tokenizer.buffer, { kernels, ...manifest.options, ...model.options });
    loadSeconds.construct = since(constructStarted);
    return true;
  } finally {
    weights.buffer.destroy();
    tokenizer?.buffer.destroy();
  }
}

// checkpoint and tokenizer: PyProxies of the converted buffers. Returns why nothing was kept, or undefined.
async function keepConverted(model, checkpoint, tokenizer, options, signal) {
  const cache = await globalThis.caches?.open(CONVERTED_CACHE).catch(() => undefined);
  if (!cache) {
    return "this browser has no Cache API here";
  }
  const bytes = checkpoint.length;
  const { quota = 0, usage = 0 } = (await navigator.storage?.estimate?.().catch(() => ({}))) ?? {};
  if (quota && quota - usage < bytes * 1.1) {
    return `the browser leaves ${((quota - usage) / 1e6).toFixed(0)} MB, and the model needs ${(bytes / 1e6).toFixed(0)} MB`;
  }
  const parts = Math.ceil(bytes / PART_BYTES);
  try {
    for (let part = 0; part < parts; part++) {
      // the view is taken anew for every part: it dies when the WebAssembly memory grows, and an await may let it
      const view = checkpoint.getBuffer("u8");
      const copy = view.data.slice(part * PART_BYTES, Math.min((part + 1) * PART_BYTES, bytes));
      view.release();
      await cache.put(convertedKey(model, `part-${String(part).padStart(3, "0")}`), new Response(copy));
      signal.throwIfAborted();
    }
    const view = tokenizer.getBuffer("u8");
    const vocabulary = view.data.slice();
    view.release();
    await cache.put(convertedKey(model, "tokenizer.bin"), new Response(vocabulary));
    const manifest = { id: model.id, name: model.name, repo: model.hf.repo, revision: model.hf.revision, bytes, parts, options, saved: Date.now() };
    await cache.put(convertedKey(model, "manifest.json"), new Response(JSON.stringify(manifest), { headers: { "Content-Type": "application/json" } }));
  } catch (error) {
    // a full disk, or a change of mind: leave nothing half written
    for (const request of await cache.keys()) {
      if (request.url.startsWith(convertedKey(model, ""))) {
        await cache.delete(request);
      }
    }
    if (signal.aborted) {
      throw error;
    }
    return String(error.message ?? error);
  }
}

async function convert(model, signal, id) {
  const remote = typeof model.hf.repo === "string";
  if (remote && await loadConverted(model, signal, id)) {
    return { fromCache: true };
  }
  if (!llama2_convert) {
    // fetched when it is first needed: most visitors never convert anything
    const res = await fetch(new URL(`llama2_convert.py${self.location.search}`, import.meta.url), { signal });
    if (!res.ok) {
      throw new Error(`Could not fetch llama2_convert.py: ${res.status}`);
    }
    pyodide.FS.writeFile("llama2_convert.py", await res.text());
    llama2_convert = pyodide.pyimport("llama2_convert");
  }
  const started = performance.now();
  const at = (name) => `https://huggingface.co/${model.hf.repo}/resolve/${model.hf.revision}/${name}`;
  const text = async (url) => {
    const res = await fetch(url, { signal });
    if (!res.ok) {
      throw new Error(`Could not fetch ${url}: ${res.status}`);
    }
    return res;
  };
  // the beginning of the file: 8 bytes that say how long the JSON header is, then the header
  let first, size;
  if (remote) {
    ({ bytes: first, total: size } = await fetchRange(at(model.hf.weights), 0, HF_HEADER_BYTES, signal));
  } else {
    first = new Uint8Array(await model.hf.weights.slice(0, HF_HEADER_BYTES).arrayBuffer());
    size = model.hf.weights.size;
  }
  const headerBytes = first.length >= 8 ? Number(new DataView(first.buffer).getBigUint64(0, true)) : -1;
  if (!(headerBytes >= 2 && headerBytes <= 100e6)) {
    throw new Error("This is not a safetensors file.");
  }
  const base = 8 + headerBytes;
  if (base > first.length) {
    first = remote ? (await fetchRange(at(model.hf.weights), 0, base, signal)).bytes
      : new Uint8Array(await model.hf.weights.slice(0, base).arrayBuffer());
  }
  const header = new TextDecoder().decode(first.subarray(8, base));
  const config = remote ? await (await text(at(model.hf.config))).text() : await model.hf.config.text();
  // For a repository nobody has looked at (?hf=), the tokenizer is whichever of these it has and the converter can read
  let conversion, refusal;
  for (const candidate of [].concat(model.hf.tokenizer)) {
    try {
      const tokenizerName = remote ? candidate : candidate.name;
      const tokenizer = new Uint8Array(remote ? await (await text(at(candidate))).arrayBuffer() : await candidate.arrayBuffer());
      signal.throwIfAborted();
      conversion = llama2_convert.Conversion.callKwargs(header, base, config, tokenizer, tokenizerName, { start: base, ...model.conversion });
      break;
    } catch (error) {
      if (signal.aborted) {
        throw error;
      }
      refusal ??= error;
    }
  }
  if (!conversion) {
    throw refusal;
  }
  try {
    let reported = -1;
    const feed = (bytes) => {
      const percent = Math.floor(conversion.feed(bytes) * 100);
      if (percent !== reported) {
        reported = percent;
        postMessage({ type: "progress", load: id, received: Math.round((percent / 100) * size), total: size, converting: true });
      }
    };
    if (remote) {
      await inOrder(at(model.hf.weights), base, size, feed, signal);
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
      }
    }
    conversion.finish();
    loadSeconds.download = since(started);

    const constructStarted = performance.now();
    // every one of these proxies keeps its Python object alive, the checkpoint too: none may be left behind
    const proxies = [conversion.options, conversion.checkpoint, conversion.tokenizer];
    let kept;
    try {
      const options = proxies[0].toJs({ dict_converter: Object.fromEntries });
      llama = llama2_numpy.Llama.callKwargs(proxies[1], proxies[2], { kernels, ...options, ...model.options });
      loadSeconds.construct = since(constructStarted);
      if (remote) {
        postMessage({ type: "status", load: id, text: `${model.name}: keeping the converted model...` });
        kept = await keepConverted(model, proxies[1], proxies[2], options, signal);
      }
    } finally {
      proxies.forEach((proxy) => proxy.destroy());
    }
    return { fromCache: false, notKept: kept };
  } finally {
    // the engine keeps what it needs of the checkpoint alive, the rest goes with this
    conversion.destroy();
  }
}

async function load(model, signal, id) {
  signal.throwIfAborted();
  // let go of the previous model first, so that two never have to fit in memory
  if (llama) {
    llama.destroy();
    llama = undefined;
    // the engine's closures and the model refer to each other, so only the cycle collector frees the weights
    pyodide.runPython("import gc; gc.collect()");
  }
  if (model.hf) {
    postMessage({ type: "status", load: id, text: `${model.name}: ${model.hf.repo ? "fetching from Hugging Face and converting" : "converting"}...` });
    await initialized;
    signal.throwIfAborted();
    const converted = await convert(model, signal, id);
    postMessage({
      type: "ready", load: id, pyodide: pyodide.version, backend: llama.backend, seq_len: llama.seq_len,
      seconds: { ...loadSeconds }, ...converted,
    });
    return;
  }
  postMessage({ type: "status", load: id, text: `${model.file ? "Reading" : "Downloading"} ${model.name}...` });
  const downloadStarted = performance.now();
  if (model.url) {
    // the size of a file somewhere else is what its server says
    model.bytes = (await fetchRange(model.url.checkpoint, 0, 28, signal)).total;
    if (!(model.bytes > 28)) {
      throw new Error(`${model.url.checkpoint} does not answer range requests, so its size is unknown.`);
    }
  }
  const checkpoint = model.file ? readFile(model, signal, id) : model.url ? readUrl(model, signal, id) : download(model, signal, id);
  const tokenizerBytes = model.file ? model.tokenizerFile.arrayBuffer()
    : fetch(model.url ? model.url.tokenizer : new URL(`models/${model.tokenizer}`, import.meta.url), { signal }).then((res) => {
      if (!res.ok) {
        throw new Error(`Could not fetch ${model.url?.tokenizer ?? model.tokenizer}: ${res.status}`);
      }
      return res.arrayBuffer();
    });
  tokenizerBytes.catch(() => {});
  await initialized;
  signal.throwIfAborted();
  const options = model.file || model.url ? await localOptions(model, new Uint8Array(await tokenizerBytes)) : model.options;
  signal.throwIfAborted();

  const weights = pythonBuffer(model.bytes);
  let tokenizer;
  try {
    await checkpoint.into(weights.write);
    const vocabulary = new Uint8Array(await tokenizerBytes);
    signal.throwIfAborted();
    loadSeconds.download = since(downloadStarted);

    // from here to the end nothing waits, so no other message gets in between
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    try {
      llama = llama2_numpy.Llama.callKwargs(weights.buffer, tokenizer.buffer, { kernels, ...options });
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
    weights.buffer.destroy();
    tokenizer?.buffer.destroy();
  }
  postMessage({
    type: "ready", load: id, pyodide: pyodide.version, backend: llama.backend, seq_len: llama.seq_len,
    seconds: { ...loadSeconds },
  });
  if (!model.file && !model.url) {
    dropStaleParts(model);
  }
}

// the run that is going on, and whether the page asked it to stop
let generating, stopped = false;

// Hand the event loop a turn, so that a message sent meanwhile is delivered. setTimeout would cost 4ms per
// call (the browsers clamp it), a MessageChannel comes back in the same millisecond.
function breathe() {
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(0);
  });
}

async function generate({ type, prompt, ...options }) {
  // a Python generator: every step of the iteration runs one forward pass and hands over one piece of text
  const pieces = llama.generate.callKwargs(prompt, options);
  try {
    let breathed = performance.now();
    while (!stopped) {
      const { done, value } = pieces.next();
      if (done) {
        break;
      }
      postMessage({ type: "token", text: value });
      // by the clock and not by the token: often enough for the button to feel immediate with a model that
      // writes 50 tokens a second, rarely enough not to cost one that writes 900 a measurable tok/s
      if (performance.now() - breathed > 50) {
        await breathe();
        breathed = performance.now();
      }
    }
    // close the generator here, so that the engine has written its stats before the "done" below
    pieces.return();
  } finally {
    pieces.destroy();
  }
  postMessage({ type: "done", ...llama.stats.toJs({ dict_converter: Object.fromEntries }) });
}

self.onmessage = async ({ data }) => {
  let signal;
  try {
    if (data.type === "init" || data.type === "load") {
      // The latest choice wins: the download that is going on stops, and its parts that are complete stay in
      // the cache. Pyodide is loaded once, whatever happens to the model that was asked for first.
      loading?.abort();
      loading = new AbortController();
      signal = loading.signal;
      // the model downloads while Pyodide loads
      initialized ??= init(data.search);
      // never take the model away from a run that is going on
      stopped = generating !== undefined;
      // The cancelled load frees its buffer a few turns of the event loop after the abort. Without waiting for
      // that the next buffer is allocated first, and the WebAssembly memory, which never shrinks, grows by a
      // whole model with every change of mind (587 MB after four of them).
      const previous = unloaded;
      const current = previous.then(() => generating?.catch(() => {})).then(() => load(data.model, signal, data.load));
      unloaded = current.catch(() => {});
      await current;
    } else if (data.type === "generate") {
      if (!llama) {
        throw new Error("The model is not ready.");
      }
      // messages keep arriving while this runs, hence the promise the other branches look at
      stopped = false;
      generating = generate(data);
      try {
        await generating;
      } finally {
        generating = undefined;
      }
    } else if (data.type === "stop") {
      // a stop that arrives before a run starts, or after it ended, must not cut the next one short
      stopped = generating !== undefined;
    }
  } catch (err) {
    // a cancelled load has nothing to report: the one that replaced it speaks for itself
    if (!signal?.aborted) {
      // a ValueError of the engine is a message for the reader (wrong file, prompt too long): no traceback
      const message = err.type === "ValueError" ? err.message.trim().split("\n").pop().replace(/^ValueError: /, "")
        : err?.name === "Error" ? err.message : String(err);
      postMessage({ type: "error", load: data.load, message });
    }
  }
};
