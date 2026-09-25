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

// Pyodide asks for the NumPy wheel only once it has started, several seconds after its own files: until then it
// does not know the name. That name is in pyodide-lock.json, which Pyodide fetches anyway, so this reads the lock
// as well and puts the wheel into the HTTP cache while Pyodide is still coming up. Nothing is written down here:
// the version is the one resolved at run time, the file name comes from the lock. Anything unexpected (another
// shape of the lock, a CDN that says no) only means no head start, so every error is dropped.
async function prefetchNumpy(base) {
  try {
    const lock = await (await fetch(`${base}pyodide-lock.json`)).json();
    const name = lock.packages?.numpy?.file_name;
    if (typeof name !== "string" || !/^[A-Za-z0-9._+-]+\.whl$/.test(name)) {
      return;
    }
    const wheel = await fetch(base + name);
    // read it to the end so that the browser keeps it, and drop every chunk: this copy is never used
    await (wheel.body ? wheel.body.pipeTo(new WritableStream()) : wheel.arrayBuffer());
  } catch {
    // no head start
  }
}

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
  const started = performance.now();
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
  // The download runs while Pyodide loads, so it usually ends long before into() can write anything into Python.
  // Its own seconds are the time until the last byte arrived, not the time until the waiting was over as well.
  const source = { overlapped: true };
  const finished = Promise.all(Array.from({ length: Math.min(CONNECTIONS, parts) }, connection))
    .then(() => { source.seconds = since(started); });
  // a load that is cancelled while Pyodide still loads never gets to into(): that is no unhandled rejection
  finished.catch(() => {});
  // write(offset, chunk) receives everything queued so far, and every later chunk
  source.into = async (write) => {
    sink = write;
    queue.splice(0).forEach(([offset, chunk]) => write(offset, chunk));
    await finished;
    if (received !== model.bytes) {
      throw new Error(`${model.checkpoint}: got ${received} bytes instead of ${model.bytes}`);
    }
  };
  return source;
}

// The same for a file of the visitor's own disk: read in chunks straight into the Python buffer, never as a whole.
function readFile(model, signal, load) {
  // a file is read only once there is somewhere to put it, so these seconds begin here and not at the choice
  const source = {
    async into(write) {
      const started = performance.now();
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
      source.seconds = since(started);
    },
  };
  return source;
}

// A llama2.c checkpoint at a URL (?checkpoint=&tokenizer=): range requests in parallel, written where they belong
function readUrl(model, signal, load) {
  const source = {
    async into(write) {
      const started = performance.now();
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
      source.seconds = since(started);
    },
  };
  return source;
}

// The legacy format carries no metadata, but its header fixes the size of a float32, a float16 and an int8 file.
// A file that is none of them is refused before it is read, and so is a tokenizer.bin of another vocabulary.
async function localOptions(model, vocabulary) {
  const first = model.file ? await model.file.slice(0, 28).arrayBuffer() : (await fetchRange(model.url.checkpoint, 0, 28, new AbortController().signal)).bytes.buffer;
  const header = pyodide.toPy([...new Int32Array(first)]);
  const pieces = pyodide.toPy(vocabulary);
  try {
    // what the file cannot say and the settings may: a Qwen2 has biases, a GPT-2 or GPT-NeoX another set of tensors
    const { bias = false, arch = "llama" } = model.options ?? {};
    const dtype = llama2_numpy.checkpoint_dtype(header, model.bytes, bias, arch);
    llama2_numpy.check_tokenizer(pieces, header);
    return { ...model.options, dtype };
  } finally {
    header.destroy();
    pieces.destroy();
  }
}

let pyodide, llama2_numpy, llama2_convert, llama, kernels;
// the models kept from earlier conversions (kept.js, T99), imported when the first conversion comes
let keptModule;
// T93: the forward pass in JavaScript (forward.js) and its kernels, compiled once; the memory of the model loaded now
let forwardModule, jsKernels, weightsNow;
// T93 stage 2: the kernels for a shared memory (only where the page is cross-origin isolated), what the page asked
// about the number of threads ({ fixed, remembered, hint }), and the forward pass of the model loaded now
let sharedKernels, threadsRequest, outsideNow;
// the optimizations this session leaves out (T52): ?without=relaxed,sampler, and ?kernel=off as it always was
let disabled = [];
// what the page's own URL said, to come back to after a benchmark has tried other combinations (T77)
let pageSwitches = [];

// Only what the engine has a fallback for. A name it does not know is refused there, and the page says so.
function switchesOf(search) {
  const parameters = new URLSearchParams(search);
  const names = (parameters.get("without") ?? "").split(",").map((name) => name.trim()).filter(Boolean);
  if (parameters.get("kernel") === "off" && !names.includes("kernels")) {
    names.push("kernels");
  }
  return names;
}
// init() as a promise: every load waits for it, also the one that replaces the first
let initialized;
// the AbortController of the load that is going on, and a promise that settles once it has cleaned up
let loading, unloaded = Promise.resolve();

// how long the load took, in seconds: Pyodide once per session, the other two per model. The download of a model
// of this site runs while Pyodide loads and usually ends first, so its seconds are counted until the last byte
// arrives (not until the bytes reach Python, which has to wait for Pyodide). The page says that the two overlap
// instead of adding them up, but only for the model that was loaded while Pyodide was still coming.
const loadSeconds = {};
// when Pyodide became usable, to tell that first model from the ones chosen afterwards
let pyodideAt = 0;
const since = (started) => (performance.now() - started) / 1000;

async function init(search) {
  const started = performance.now();
  const asked = new URLSearchParams(search);
  const parts = Number(asked.get("hfParts")), connections = Number(asked.get("hfConnections"));
  if (parts >= 1 && parts <= 64) hfPartBytes = Math.round(parts * 1024 * 1024);
  if (connections >= 1 && connections <= 32) hfConnections = Math.floor(connections);
  const version = await resolvePyodideVersion(search);
  postMessage({ type: "status", text: `Loading Pyodide ${version}...` });
  const base = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;
  // while Pyodide starts, not after: loadPackage("numpy") below finds the wheel in the HTTP cache
  const numpy = prefetchNumpy(base);
  const { loadPyodide } = await import(`${base}pyodide.mjs`);
  pyodide = await loadPyodide();
  // a prefetch that is still running would otherwise be raced by loadPackage, and the wheel fetched twice
  await numpy;
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
  disabled = pageSwitches = switchesOf(search);
  if (!disabled.includes("kernels")) {
    for (const name of ["simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
      const kernel = await fetch(new URL(`${name}${self.location.search}`, import.meta.url)).catch(() => undefined);
      if (kernel?.ok) {
        pyodide.FS.writeFile(`/home/pyodide/${name}`, new Uint8Array(await kernel.arrayBuffer()));
        kernels = "/home/pyodide/simdkernel.so";
      }
    }
  }
  // T93: the forward pass runs in forward.js, on the plain build of the same kernels (simdkernel.so stays for the
  // sampling, which works on Python's logits). Without them (no WebAssembly SIMD) the engine runs NumPy.
  try {
    forwardModule = await import(new URL(`forward.js${self.location.search}`, import.meta.url));
    const [plain, relaxed] = await Promise.all(["simdkernel_plain.wasm", "simdkernel_relaxed_plain.wasm"].map((name) =>
      fetch(new URL(`${name}${self.location.search}`, import.meta.url)).then((res) => (res.ok ? res.arrayBuffer() : null)).catch(() => null)));
    jsKernels = plain ? forwardModule.compileKernels(plain, relaxed) : null;
    if (jsKernels && self.crossOriginIsolated) {
      const [sharedPlain, sharedRelaxed] = await Promise.all(["simdkernel_shared.wasm", "simdkernel_relaxed_shared.wasm"].map((name) =>
        fetch(new URL(`${name}${self.location.search}`, import.meta.url)).then((res) => (res.ok ? res.arrayBuffer() : null)).catch(() => null)));
      sharedKernels = sharedPlain ? forwardModule.compileKernels(sharedPlain, sharedRelaxed) : null;
    }
  } catch {
    jsKernels = null;
  }
  loadSeconds.pyodide = since(started);
  pyodideAt = performance.now();
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

// Where the weights of a model go (T93): the WebAssembly memory of forward.js, where the forward pass runs, or,
// without the kernels (?without=kernels, or no WebAssembly SIMD), a Python bytearray for the NumPy engine.
// write(offset, bytes) fills it, slice(begin, end) copies a stretch out (to keep a conversion), llama() makes the
// engine on it, destroy() lets go of the Python buffer (a memory of forward.js goes with the engine).
// a software thread of forward.js (stage 2): a module worker of its own, ready once its kernels are warm
const spawnThread = (data) => new Promise((resolve, reject) => {
  const worker = new Worker(new URL(`helper.js${self.location.search}`, import.meta.url), { type: "module" });
  worker.onmessage = () => resolve({ terminate: () => worker.terminate() });
  worker.onerror = (event) => reject(new Error(event.message ?? "a software thread did not start"));
  worker.postMessage(data);
});

// T96: one memory of forward.js, kept from model to model. A browser reserves address space for every WebAssembly
// memory, shared or not and whatever its maximum, and Chromium refused the third one of a page: the benchmark's
// first round, or a visitor's second change of model, then found no memory at all. So a memory is kept and grown
// as long as the next model fits under its maximum, and made anew only for a larger one. The maximum comes from the
// model (weightsMemory: four times the file and a gigabyte): asking for 4 GB up front left a phone no room for
// Pyodide's own memory, and "Loading Pyodide" never ended (2026-09-25).
let weightsPool;
function pooledWeights(size, shared) {
  const pages = (bytes) => Math.ceil(bytes / 65536);
  const fits = weightsPool && weightsPool.shared === shared && pages(weightsPool.base + size) + 1 <= weightsPool.maximum;
  if (!fits) {
    weightsPool = undefined;  // the old one goes with its engine; nothing else refers to it
    let memory, base;
    if (shared) {
      try {
        ({ memory, base } = forwardModule.weightsMemory(size, { shared: true }));
      } catch {
        memory = undefined;  // no shared memory here: one thread
      }
    }
    if (!memory) ({ memory, base } = forwardModule.weightsMemory(size));
    const isShared = shared && memory.buffer instanceof SharedArrayBuffer;
    // a memory without a maximum (not shared) grows as far as the browser allows: 4 GB of pages
    weightsPool = { memory, base, shared: isShared, maximum: isShared ? memory.maximum ?? 65536 : 65536 };
  }
  const { memory, base } = weightsPool;
  const more = pages(base + size) + 1 - memory.buffer.byteLength / 65536;
  if (more > 0) memory.grow(more);
  return weightsPool;
}

function weightsBuffer(size) {
  if (jsKernels && !disabled.includes("kernels")) {
    // a shared memory where the page is cross-origin isolated (stage 3), unless ?threads=1; else one thread
    const wanted = Boolean(sharedKernels && self.crossOriginIsolated && threadsRequest?.fixed !== 1);
    const { memory, base, shared } = pooledWeights(size, wanted);
    const kernels = shared ? sharedKernels : jsKernels, spawn = shared ? spawnThread : undefined;
    weightsNow = memory;
    return {
      write: (offset, chunk) => new Uint8Array(memory.buffer, base + offset, chunk.length).set(chunk),
      slice: (begin, end) => new Uint8Array(memory.buffer, base + begin, end - begin).slice(),
      llama: (tokenizer, options) => {
        outsideNow = forwardModule.external({ memory, base, size, kernels, spawn });
        return llama2_numpy.Llama.callKwargs(null, tokenizer, { ...options, external: outsideNow });
      },
      destroy() {},
    };
  }
  weightsNow = undefined;
  outsideNow = undefined;
  const { buffer, write } = pythonBuffer(size);
  return {
    write,
    slice(begin, end) {
      const view = buffer.getBuffer("u8");
      const copy = view.data.slice(begin, end);
      view.release();
      return copy;
    },
    llama: (tokenizer, options) => llama2_numpy.Llama.callKwargs(buffer, tokenizer, options),
    destroy: () => buffer.destroy(),
  };
}

// Every await in here may end with the AbortError of signal: a newer load has taken over, and this one must
// leave nothing behind, least of all a Python buffer as large as its model.
// A Hugging Face model, from the visitor's disk ({weights, config, tokenizer} are Files) or from huggingface.co
// ({repo, revision, weights, config, tokenizer} are names): model.safetensors arrives in the order of the file, a few
// megabytes at a time, and the Python code that builds the models of this site converts every tensor as it comes and
// writes it to its place in a buffer of the final size. Reading in the order of the output instead would mean
// hundreds of range requests, and each one takes a second.
// 16 MiB over 6 connections (T107, measured in CI against huggingface.co): parts of 8 MiB took 1.36 times as long
// for Qwen2.5 0.5B, of 4 MiB 2.8 times; more connections gained 6% at most. But the first bytes then come late on a
// slow line, and a phone has less room for what waits in the queue (two parts per connection: 192 MB at 16 MiB),
// so the first part is 8 MiB wherever the size is not fixed by the URL, and the rest follow what it measured
// (the owner's ask, 2026-09-25): 16 MiB where that part came in at 4 MB/s or more and the device says nothing of a
// small memory, 8 MiB otherwise.
const HF_PART_BYTES = 16 * 1024 * 1024;
const HF_SMALL_PART_BYTES = 8 * 1024 * 1024;
const HF_FAST_BYTES_PER_SECOND = 4e6;
const HF_CONNECTIONS = 6;
const HF_HEADER_BYTES = 512 * 1024;  // the JSON header of a safetensors file is a few dozen kilobytes
// T107: ?hfParts=<MiB>&hfConnections=<N> fix the two, to measure; the page offers no way to them
let hfPartBytes = 0, hfConnections = HF_CONNECTIONS;  // 0: not fixed, decided per file from its first part

// The size of a file, for the few places that need it (the whole of a model: how many parts to ask for). A range
// response says it in Content-Range, but that header is not one CORS shows by default: huggingface.co exposes it by
// name, its CDN by "*", and a browser that does not honour "*" (WebKit; T112) sees none and the fetch never began
// ("The file ended before all of its tensors were read"). Content-Length of a HEAD is always shown.
async function fileSize(url, signal) {
  const res = await fetch(url, { method: "HEAD", signal });
  const length = Number(res.headers.get("Content-Length"));
  if (!res.ok || !Number.isFinite(length) || length <= 0) {
    throw new Error(`Could not learn the size of ${url}: ${res.status}`);
  }
  return length;
}
// the size a range response reported, or the file's size asked for separately when it did not
const sized = async (url, result, signal) => (Number.isFinite(result.total) && result.total > 0 ? result : { ...result, total: await fileSize(url, signal) });

// arriving(count): told of every stretch of the body as it comes, for a progress line before a whole part is in
async function fetchRange(url, begin, end, signal, arriving) {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(url, { headers: { Range: `bytes=${begin}-${end - 1}` }, signal });
      if (res.status !== 206 && res.status !== 200) {
        throw new Error(`Could not fetch ${url}: ${res.status}`);
      }
      let bytes;
      if (arriving && res.body) {
        const pieces = [];
        let length = 0;
        for (const reader = res.body.getReader(); ;) {
          const { done, value } = await reader.read();
          if (done) break;
          pieces.push(value);
          length += value.length;
          arriving(value.length);
        }
        bytes = new Uint8Array(length);
        pieces.reduce((at, piece) => (bytes.set(piece, at), at + piece.length), 0);
      } else {
        bytes = new Uint8Array(await res.arrayBuffer());
      }
      let total = Number((res.headers.get("Content-Range") ?? "").split("/")[1]);
      if (res.status === 200) {
        // the server ignored the range and sent the whole file: what was asked for is cut out of it (slow, but
        // right), and the console says so (T112: a browser whose stack does this is one to know about)
        console.warn(`${url} answered a range request with the whole file (${bytes.length} bytes)`);
        total = bytes.length;
        bytes = bytes.subarray(begin, end);
      }
      return { bytes, total };
    } catch (error) {
      if (signal.aborted || attempt === 2) {
        throw error;
      }
    }
  }
}

// feed(bytes) gets the file from position start to its end, in order, although the parts arrive as they like.
// The parts are cut as they are asked for: the first small, the rest by what the first one measured (see above).
// arriving(bytes): how much of the file is in so far, told as it comes (the page shows it until the conversion of
// the first part gives it percentages: on a slow line the first part alone takes a while, and a line that says
// nothing looks stuck).
async function inOrder(url, start, size, feed, signal, arriving = () => {}) {
  const small = navigator.deviceMemory !== undefined && navigator.deviceMemory <= 4;
  let partBytes = hfPartBytes || HF_SMALL_PART_BYTES;
  const ranges = [];  // [begin, end] of every part asked for so far, in the order of the file
  const arrived = new Map();
  let scheduled = start, fed = 0, waiting = [], received = start;
  const connection = async () => {
    for (;;) {
      // no more than two parts per connection wait in memory for an earlier one
      while (ranges.length - fed >= 2 * hfConnections) {
        await new Promise((resolve) => waiting.push(resolve));
      }
      if (scheduled >= size) {
        return;
      }
      const part = ranges.length, begin = scheduled, end = Math.min(begin + partBytes, size);
      ranges.push([begin, end]);
      scheduled = end;
      const began = performance.now();
      arrived.set(part, (await fetchRange(url, begin, end, signal, (count) => { received += count; arriving(received); })).bytes);
      if (part === 0 && !hfPartBytes) {
        const rate = (end - begin) / ((performance.now() - began) / 1000);
        partBytes = rate >= HF_FAST_BYTES_PER_SECOND && !small ? HF_PART_BYTES : HF_SMALL_PART_BYTES;
      }
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
  await Promise.all(Array.from({ length: hfConnections }, connection));
}

// What a conversion made is kept for the next visit (kept.js): in the origin private file system where there is one
// (T99), else in the Cache API. The original is twice as large, and fetching and converting it again on every visit
// would be no way to use a model. The page lists what is kept and deletes it.
// Returns what the page needs of it, or { miss } with why nothing kept could be used (for tests/e2e.mjs).
async function loadConverted(model, signal, id) {
  let kept;
  try {
    kept = await keptModule.openKept(model);
  } catch (error) {
    return { miss: `could not open what is kept: ${error.message ?? error}` };
  }
  if (!kept) {
    return { miss: "nothing kept for this model" };
  }
  const { manifest } = kept;
  const started = performance.now();
  const weights = weightsBuffer(manifest.bytes);
  let tokenizer;
  try {
    let offset = 0;
    try {
      for await (const bytes of kept.parts()) {
        signal.throwIfAborted();
        weights.write(offset, bytes);
        offset += bytes.length;
        postMessage({ type: "progress", load: id, received: offset, total: manifest.bytes });
      }
    } catch (error) {
      if (signal.aborted) {
        throw error;
      }
      return { miss: `could not read what is kept: ${error.message ?? error}` };  // evicted: convert again
    }
    const vocabulary = await kept.tokenizer();
    loadSeconds.download = since(started);
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    // template is for the page, not for the engine (see convert())
    const engineOptions = { ...manifest.options };
    delete engineOptions.template;
    llama = weights.llama(tokenizer.buffer, { kernels, disable: disabled, ...engineOptions, ...model.options });
    loadSeconds.construct = since(constructStarted);
    return { template: manifest.options.template, keptIn: kept.where };
  } finally {
    weights.destroy();
    tokenizer?.buffer.destroy();
  }
}

// checkpoint: the weights (weightsBuffer), bytes long; tokenizer: a PyProxy of the converted tokenizer. Returns why
// nothing was kept, or undefined.
async function keepConverted(model, checkpoint, bytes, tokenizer, options, signal) {
  const view = tokenizer.getBuffer("u8");
  const vocabulary = view.data.slice();
  view.release();
  const manifest = { id: model.id, name: model.name, repo: model.hf.repo, revision: model.hf.revision, bytes, options, saved: Date.now() };
  // slice() copies: the memory it comes from may grow (and so move) while an await waits
  return keptModule.keep(model, manifest, (begin, end) => checkpoint.slice(begin, end), vocabulary, signal);
}

async function convert(model, signal, id) {
  const remote = typeof model.hf.repo === "string";
  // with the ?v=<build> of this worker, like every file it reads (AGENTS.md)
  keptModule ??= await import(new URL(`kept.js${self.location.search}`, import.meta.url));
  const kept = remote ? await loadConverted(model, signal, id) : undefined;
  if (kept && !kept.miss) {
    return { fromCache: true, keptIn: kept.keptIn, template: kept.template };
  }
  const keptMiss = kept?.miss;
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
  let first, size, base, conversion, shards;
  // T93: the converter writes the checkpoint here, piece by piece, straight into where the engine will read it.
  // A Python buffer on the way would stay: Pyodide's memory never shrinks.
  let weights, weightsSize = 0;
  const sink = {
    open(bytes) {
      weights?.destroy();  // an earlier try (another tokenizer) that got this far
      weights = weightsBuffer(bytes);
      weightsSize = bytes;
    },
    write(offset, array) {
      const view = array.getBuffer("u8");
      weights.write(offset, view.data);
      view.release();
    },
  };
  // T89: quantize() on the SIMD kernels, the same bytes six times faster (none with ?without=kernels)
  const quantizeRows = kernels && !disabled.includes("kernels") ? llama2_numpy.kernel_quantizer(kernels) : undefined;
  if (remote && model.hf.weights.endsWith(".gguf")) {
    // T74: a GGUF holds the configuration and the vocabulary in its header, before the tensors: no config.json and
    // no tokenizer to fetch. The header is a few megabytes (the vocabulary), so it is fetched in growing pieces
    // until the converter can read all of it.
    for (let bytes = 4 * HF_HEADER_BYTES; ; bytes *= 4) {
      ({ bytes: first, total: size } = await sized(at(model.hf.weights), await fetchRange(at(model.hf.weights), 0, bytes, signal), signal));
      try {
        conversion = llama2_convert.Conversion.from_gguf.callKwargs(first, { ...model.conversion, sink, quantize_rows: quantizeRows });
        break;
      } catch (error) {
        if (error.type !== "Incomplete" || bytes >= size) {
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
        const joined = llama2_convert.joined_shards(shards.map((shard) => shard.header));
        let lengths;
        [header, lengths] = joined.toJs();
        joined.destroy();
        shards.forEach((shard, i) => { shard.length = lengths[i]; });
        base = 0;
        size = shards.reduce((sum, shard) => sum + shard.length, 0);
      }
    }
    const config = remote ? await (await text(at(model.hf.config))).text() : await model.hf.config.text();
    // The format of one turn, when the model publishes a chat_template (T73). It is small, and a model without
    // one (or with one the converter cannot read) simply keeps the format src/models.js has for it.
    const tokenizerConfig = await (remote ? text(at("tokenizer_config.json")).then((r) => r.text())
      : model.hf.tokenizerConfig?.text() ?? Promise.resolve("")).catch(() => "");
    // For a repository nobody has looked at (?hf=), the tokenizer is whichever of these it has and the converter can read
    let refusal;
    for (const candidate of [].concat(model.hf.tokenizer)) {
      try {
        const tokenizerName = remote ? candidate : candidate.name;
        const tokenizer = new Uint8Array(remote ? await (await text(at(candidate))).arrayBuffer() : await candidate.arrayBuffer());
        signal.throwIfAborted();
        conversion = llama2_convert.Conversion.callKwargs(header, base, config, tokenizer, tokenizerName,
          { start: base, tokenizer_config: tokenizerConfig, ...model.conversion, sink, quantize_rows: quantizeRows });
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
  }
  let template;
  try {
    let reported = -1, converting = 0, fed = false, told = 0;
    // until the first part is converted, the page hears how much has arrived (a line that says nothing looks stuck)
    const arriving = (received) => {
      if (fed || performance.now() - told < 250) {
        return;
      }
      told = performance.now();
      postMessage({ type: "progress", load: id, received, total: size });
    };
    const feed = (bytes) => {
      fed = true;
      // T84: the time Python spends converting, apart from the time spent waiting for the download
      const began = performance.now();
      const percent = Math.floor(conversion.feed(bytes) * 100);
      converting += performance.now() - began;
      if (percent !== reported) {
        reported = percent;
        postMessage({ type: "progress", load: id, received: Math.round((percent / 100) * size), total: size, converting: true });
      }
    };
    if (shards) {
      for (const shard of shards) {
        await inOrder(at(shard.name), shard.base, shard.base + shard.length, feed, signal, arriving);
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
      llama = weights.llama(proxies[1], { kernels, disable: disabled, ...engineOptions, ...model.options });
      loadSeconds.construct = since(constructStarted);
      if (remote) {
        postMessage({ type: "status", load: id, text: `${model.name}: keeping the converted model...` });
        kept = await keepConverted(model, weights, weightsSize, proxies[1], options, signal);
      }
    } finally {
      proxies.forEach((proxy) => proxy.destroy());
      weights?.destroy();
    }
    return { fromCache: false, notKept: kept, keptMiss, template };
  } finally {
    // the engine keeps what it needs of the checkpoint alive, the rest goes with this
    conversion.destroy();
    quantizeRows?.destroy();
  }
}

async function load(model, signal, id) {
  signal.throwIfAborted();
  // let go of the previous model first, so that two never have to fit in memory
  if (llama) {
    llama.release?.();  // what forward.js holds of Python's, and its software threads (T93)
    llama.destroy();
    llama = undefined;
    outsideNow = undefined;  // the engine goes; the memory stays for the next model (T96)
    // the engine's closures and the model refer to each other, so only the cycle collector frees the weights
    pyodide.runPython("import gc; gc.collect()");
  }
  if (model.hf) {
    postMessage({ type: "status", load: id, text: `${model.name}: ${model.hf.repo ? "fetching from Hugging Face and converting" : "converting"}...` });
    await initialized;
    signal.throwIfAborted();
    const converted = await convert(model, signal, id);
    await startThreads(model);
    postMessage({
      type: "ready", load: id, pyodide: pyodide.version, backend: llama.backend, seq_len: llama.seq_len,
      seconds: { ...loadSeconds }, heap: heapBytes(), threads: threadsNow(), ...converted,
    });
    return;
  }
  postMessage({ type: "status", load: id, text: `${model.file ? "Reading" : "Downloading"} ${model.name}...` });
  const downloadStarted = performance.now();
  if (model.url) {
    // the size of a file somewhere else is what its server says
    model.bytes = (await sized(model.url.checkpoint, await fetchRange(model.url.checkpoint, 0, 28, signal), signal)).total;
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

  const weights = weightsBuffer(model.bytes);
  let tokenizer;
  try {
    await checkpoint.into(weights.write);
    const vocabulary = new Uint8Array(await tokenizerBytes);
    signal.throwIfAborted();
    // what the source itself measured: the bytes, without the wait for Pyodide that into() may have spent
    loadSeconds.download = checkpoint.seconds ?? since(downloadStarted);

    // from here to the end nothing waits, so no other message gets in between
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    try {
      llama = weights.llama(tokenizer.buffer, { kernels, disable: disabled, ...options });
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
  await startThreads(model);
  postMessage({
    type: "ready", load: id, pyodide: pyodide.version, backend: llama.backend, seq_len: llama.seq_len,
    seconds: { ...loadSeconds }, heap: heapBytes(), threads: threadsNow(), overlapped: checkpoint.overlapped === true && pyodideAt > downloadStarted,
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
  const engine = outsideNow?.engine;
  if (!engine?.findThreads) {
    return 1;
  }
  const { fixed = 0, remembered = 0, hint = 1 } = threadsRequest ?? {};
  try {
    if (fixed) {
      return await engine.setThreads(fixed);
    }
    return await engine.findThreads({ from: hint, remembered,
      chose: (count) => postMessage({ type: "threads", model: model.id, count }) });
  } catch {
    engine.stopThreads?.();
    return 1;
  }
}
const threadsNow = () => outsideNow?.engine?.threads ?? 1;

// the run that is going on, and whether the page asked it to stop
let generating, stopped = false;
let benching = false;  // the benchmark's rounds are running (T45); see the bench message

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
  outsideNow?.engine?.newGeneration?.();  // now and then the remembered number of threads is checked again
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
  postMessage({ type: "done", threads: threadsNow(), ...llama.stats.toJs({ dict_converter: Object.fromEntries }) });
}

/** The size of Pyodide's WebAssembly memory, which only grows; undefined before Pyodide is there. */
function heapBytes() {
  const python = pyodide?._module?.HEAPU8?.length;
  // T93: the weights and the forward pass have a memory of their own, outside Pyodide's
  return python === undefined ? undefined : python + (weightsNow?.buffer.byteLength ?? 0);
}

self.onmessage = async ({ data }) => {
  let signal;
  try {
    if (data.type === "init" || data.type === "load") {
      threadsRequest = data.threads;
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
    } else if (data.type === "bench") {
      // T45: the same model, measured again for every combination of switches the page asked for. The model is
      // built once per round from the checkpoint that is already in the Cache API, so only the engine changes.
      // One at a time: a second request while the rounds run would end a round's threads under its coordinator
      if (benching) {
        return;
      }
      benching = true;
      const rows = [];
      for (const round of data.rounds) {
        // every round is a load of its own, and it cancels whatever went before, exactly like a change of model
        loading?.abort();
        loading = new AbortController();
        signal = loading.signal;
        disabled = round.without;
        const started = performance.now();
        const previous = unloaded;
        const current = previous.then(() => generating?.catch(() => {})).then(() => load(data.model, signal, data.load));
        unloaded = current.catch(() => {});
        await current;
        const ready = since(started);
        // a warm-up, then the measured run: the same prompt, greedy, the same number of tokens
        const settings = { prompt: data.prompt, steps: data.steps, temperature: 0, echo: false };
        llama.generate.callKwargs(settings.prompt, { steps: 8, temperature: 0 }).return();
        const begin = performance.now();
        const pieces = llama.generate.callKwargs(settings.prompt, { steps: settings.steps, temperature: 0, echo: false });
        let tokens = 0;
        for (;;) {
          const { done } = pieces.next();
          if (done) {
            break;
          }
          tokens += 1;
        }
        pieces.destroy();
        const seconds = (performance.now() - begin) / 1000;
        rows.push({ name: round.name, without: round.without, tokens, speed: tokens / seconds,
                    backend: llama.backend, seconds: ready });
      }
      disabled = pageSwitches;  // not self.location.search: that is the worker's own URL (?v=hash)
      benching = false;
      postMessage({ type: "bench", load: data.load, rows, pyodide: pyodide.version });
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
      // T90: the memory ran out, in Python (MemoryError: malloc could not grow the WebAssembly memory) or in
      // JavaScript (RangeError: an ArrayBuffer or WebAssembly.Memory.grow was refused). The page says so in words
      // a visitor understands, with how much memory the page had when it happened.
      // (V8 also raises RangeError for a stack overflow, which is not this; Firefox says InternalError: out of memory)
      const memory = err?.type === "MemoryError" || err?.name === "InternalError" ||
        (err?.name === "RangeError" && !/call stack/i.test(err.message ?? ""));
      // where it happened goes to the page's console (T96): tests/e2e.mjs keeps the console of a failed run
      postMessage({ type: "error", load: data.load, message: memory ? String(err?.message ?? err).trim().split("\n").pop() : message,
                    stack: String(err?.stack ?? err), weights: weightsNow?.buffer.byteLength ?? 0,
                    ...(memory && { memory: true, heap: heapBytes() }) });
    }
  }
};
