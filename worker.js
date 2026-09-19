// Pyodide lives in this worker, so the page stays responsive while the model is loading and generating.
// The page sends   {type: "init", search, model},  {type: "load", model}  and  {type: "generate", prompt, ...options}
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
const CONNECTIONS = 4;

function download(model) {
  const parts = Math.ceil(model.bytes / PART_BYTES);
  const queue = [];
  let sink, next = 0, received = 0, reported = -1;
  const connection = async () => {
    while (next < parts) {
      const part = next++;
      const res = await fetch(new URL(`models/${model.checkpoint}.${String(part).padStart(3, "0")}`, import.meta.url));
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

let pyodide, llama2_numpy, llama;

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
  postMessage({ type: "status", text: `Downloading ${model.label}...` });
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
  llama = llama2_numpy.Llama.callKwargs(weights.buffer, tokenizer.buffer, model.options);
  weights.buffer.destroy();
  tokenizer.buffer.destroy();
  postMessage({ type: "ready", pyodide: pyodide.version });
}

function generate({ type, prompt, ...options }) {
  // a Python generator: every step of the iteration runs one forward pass and hands over one piece of text
  const pieces = llama.generate.callKwargs(prompt, options);
  try {
    for (const text of pieces) {
      postMessage({ type: "token", text });
    }
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
      await load(data.model);
    } else if (data.type === "generate") {
      generate(data);
    }
  } catch (err) {
    postMessage({ type: "error", message: String(err) });
  }
};
