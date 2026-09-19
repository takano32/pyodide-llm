// Pyodide lives in this worker, so the page stays responsive while the model is loading and generating.
// The page sends   {type: "init", search, model},  {type: "load", model},  {type: "generate", prompt, ...options}
//                  and {type: "stop"}
// and receives     {type: "status" | "progress" | "ready" | "token" | "done" | "error", ...}

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
const MODEL_CACHE = "models-v1";

async function fetchPart(url, model) {
  const cache = await globalThis.caches?.open(MODEL_CACHE).catch(() => undefined);
  const key = `${url}?bytes=${model.bytes}`;
  const cached = await cache?.match(key);
  if (cached) {
    return cached;
  }
  const res = await fetch(url);
  if (res.ok && cache) {
    // stored while the other copy streams into Python; a full disk must not stop the download
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

function download(model) {
  const parts = Math.ceil(model.bytes / PART_BYTES);
  const queue = [];
  let sink, next = 0, received = 0, reported = -1;
  const connection = async () => {
    while (next < parts) {
      const part = next++;
      const res = await fetchPart(new URL(`models/${model.checkpoint}.${String(part).padStart(3, "0")}`, import.meta.url).href, model);
      if (!res.ok) {
        throw new Error(`Could not fetch part ${part} of ${model.checkpoint}: ${res.status}`);
      }
      const reader = res.body.getReader();
      for (let offset = part * PART_BYTES; ;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        sink ? sink(offset, value) : queue.push([offset, value]);
        offset += value.length;
        received += value.length;
        // one message per percent is plenty
        const percent = Math.floor((received / model.bytes) * 100);
        if (percent !== reported) {
          reported = percent;
          postMessage({ type: "progress", received, total: model.bytes });
        }
      }
    }
  };
  const finished = Promise.all(Array.from({ length: Math.min(CONNECTIONS, parts) }, connection));
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

let pyodide, llama2_numpy, llama, kernels;

async function init(search) {
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

async function load(model, ready = Promise.resolve()) {
  // let go of the previous model first, so that two never have to fit in memory
  llama?.destroy();
  llama = undefined;
  postMessage({ type: "status", text: `Downloading ${model.name}...` });
  const checkpoint = download(model);
  const tokenizerBytes = fetch(new URL(`models/${model.tokenizer}`, import.meta.url)).then((res) => {
    if (!res.ok) {
      throw new Error(`Could not fetch ${model.tokenizer}: ${res.status}`);
    }
    return res.arrayBuffer();
  });
  await ready;

  const weights = pythonBuffer(model.bytes);
  await checkpoint.into(weights.write);
  const vocabulary = new Uint8Array(await tokenizerBytes);
  const tokenizer = pythonBuffer(vocabulary.length);
  tokenizer.write(0, vocabulary);
  llama = llama2_numpy.Llama.callKwargs(weights.buffer, tokenizer.buffer, { kernels, ...model.options });
  weights.buffer.destroy();
  tokenizer.buffer.destroy();
  postMessage({ type: "ready", pyodide: pyodide.version, backend: llama.backend });
  dropStaleParts(model);
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
  try {
    if (data.type === "init") {
      // the model downloads while Pyodide loads
      await load(data.model, init(data.search));
    } else if (data.type === "load") {
      // never take the model away from a run that is going on
      stopped = generating !== undefined;
      await generating?.catch(() => {});
      await load(data.model);
    } else if (data.type === "generate") {
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
    postMessage({ type: "error", message: String(err) });
  }
};
