// python.js (T347)
// The Python files Pyodide is given, by the module the worker imports: the window first, then its parts in the folder
// beside it. One list for the worker and for the tests that run the engine in Node; tests/python-files-check.mjs holds
// it to what is in public/.
export const PYTHON = {
  llama2_numpy: ["llama2_numpy.py", "engine/__init__.py", "engine/tokenizer.py", "engine/layers.py", "engine/kernels.py", "engine/packing.py",
    "engine/checkpoint.py", "engine/tensors.py", "engine/sampling.py", "engine/generation.py", "engine/model.py"],
  llama2_convert: ["llama2_convert.py", "convert/__init__.py", "convert/checkpoint.py", "convert/template.py", "convert/readers.py",
    "convert/sources.py", "convert/config.py", "convert/plan.py", "convert/stream.py", "convert/gguf.py", "convert/tokenizer.py",
    "convert/conversion.py"],
};

/** One file into Pyodide's file system, its folder made first */
export function placeFile(pyodide, name, content) {
  if (name.includes("/")) pyodide.FS.mkdirTree(name.slice(0, name.lastIndexOf("/")));
  pyodide.FS.writeFile(name, content);
}

/** Puts a module's files into Pyodide's file system, where `import` finds them. read(name) gives a file's text or
 * bytes (a promise of them): all are read before any is written, so a file that fails leaves nothing half placed. */
export async function placePython(pyodide, module, read) {
  const files = await Promise.all(PYTHON[module].map(async (name) => [name, await read(name)]));
  for (const [name, content] of files) placeFile(pyodide, name, content);
}
