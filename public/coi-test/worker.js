// What the model page's worker would do under COEP: load the latest Pyodide and NumPy from the CDN, read a range of
// a file on Hugging Face, and take a shared WebAssembly memory from the page and signal on it with Atomics.
self.onmessage = async ({ data: { memory } }) => {
  const report = { workerIsolated: self.crossOriginIsolated };
  try {
    const shared = new Int32Array(memory.buffer);
    Atomics.store(shared, 0, 42);
    Atomics.notify(shared, 0);
    report.atomics = true;
  } catch (error) {
    report.atomics = String(error);
  }
  try {
    const version = (await (await fetch("https://data.jsdelivr.com/v1/packages/npm/pyodide/resolved?specifier=latest")).json()).version;
    const base = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;
    const { loadPyodide } = await import(`${base}pyodide.mjs`);
    const pyodide = await loadPyodide({ indexURL: base });
    await pyodide.loadPackage("numpy", { messageCallback: () => {} });
    report.pyodide = `${version}: numpy says ${pyodide.runPython("import numpy; str(int(numpy.arange(4).sum()))")}`;
  } catch (error) {
    report.pyodide = `failed: ${error}`;
  }
  try {
    const res = await fetch("https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q8_0.gguf",
      { headers: { Range: "bytes=0-3" } });
    report.huggingface = `${res.status} ${new TextDecoder().decode(await res.arrayBuffer())}`;
  } catch (error) {
    report.huggingface = `failed: ${error}`;
  }
  postMessage(report);
};
