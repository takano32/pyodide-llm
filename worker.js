// Pyodide lives in this worker, so the page stays responsive while the model is loading and generating.
// The page sends   {type: "init", search, model},  {type: "load", model}  and  {type: "generate", prompt, ...options}
// and receives     {type: "status" | "ready" | "token" | "done" | "error", ...}

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

let pyodide, llama2_numpy, llama;

async function init(search) {
  const version = await resolvePyodideVersion(search);
  postMessage({ type: "status", text: `Loading Pyodide ${version}...` });
  const { loadPyodide } = await import(`https://cdn.jsdelivr.net/pyodide/v${version}/full/pyodide.mjs`);
  pyodide = await loadPyodide();
  await pyodide.loadPackage("numpy");

  const res = await fetch(new URL("llama2_numpy.py", import.meta.url));
  if (!res.ok) {
    throw new Error(`Could not fetch llama2_numpy.py: ${res.status}`);
  }
  pyodide.FS.writeFile("llama2_numpy.py", await res.text());
  llama2_numpy = pyodide.pyimport("llama2_numpy");
}

async function load(model) {
  // let go of the previous model first, so that two never have to fit in memory
  llama?.destroy();
  llama = undefined;
  postMessage({ type: "status", text: `Downloading ${model.label}...` });
  llama = await llama2_numpy.load.callKwargs(
    new URL(model.checkpoint, import.meta.url).href,
    new URL(model.tokenizer, import.meta.url).href,
    model.options,
  );
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
      await init(data.search);
      await load(data.model);
    } else if (data.type === "load") {
      await load(data.model);
    } else if (data.type === "generate") {
      generate(data);
    }
  } catch (err) {
    postMessage({ type: "error", message: String(err) });
  }
};
