// T648.3: tests/translation_check.py in Node's Pyodide, its affines on the SIMD kernel affine_q8 (--kernels): the
// kernels are Emscripten side modules, which only Pyodide's ctypes loads, so the check that the kernel's path writes
// fxtranslate's 100 sentences of each model runs here. The checkout, the models' folders (MARIAN_FILES) and
// bergamot-vs-shisa's sentences are mounted at their own paths; the arguments go to the check as they are.
//
//   node tests/translation-kernels.ts [--numpy] [--only <model>] [--limit <n>]       (after make kernels)
// --numpy: the check's NumPy affines in the same Pyodide (to set the kernel's seconds beside).
// Exit: the check's (1 where a translation differs; 0 with the models it does not find left out, as it says).
import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";
import { leave } from "./leave.ts";
import { placeKernels, treeOf } from "./tree.ts";

const root = path.resolve(fileURLToPath(new URL("../", import.meta.url)));
const home = path.join(os.homedir(), "GitHub", "bergamot-vs-shisa");
const folders = [root, ...(process.env.MARIAN_FILES ?? "").split(path.delimiter).filter(Boolean).map((folder) => path.resolve(folder)), home]
  .filter((folder) => fs.existsSync(folder));
const env = Object.fromEntries(["MARIAN_FILES", "SENTENCES"].filter((name) => process.env[name]).map((name) => [name, process.env[name]]));

const pyodide = await loadPyodide({ env: { HOME: os.homedir(), ...env } });
await pyodide.loadPackage("numpy", { messageCallback: () => {} });
placeKernels(pyodide, treeOf());
// (a folder under another mounted one is that one's already: mounting it again would hide nothing but costs nothing)
for (const folder of folders) {
  if (folders.some((other) => other !== folder && folder.startsWith(other + path.sep))) continue;
  pyodide.FS.mkdirTree(folder);
  pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: folder }, folder);
}
pyodide.globals.set("check", path.join(root, "tests", "translation_check.py"));
const given = process.argv.slice(2);
const numpy = given.includes("--numpy");
pyodide.globals.set("given", pyodide.toPy([...(numpy ? [] : ["--kernels"]), ...given.filter((arg) => arg !== "--numpy")]));
const code = pyodide.runPython(`
import os, runpy, sys
os.environ["TRANSLATION_KERNELS"] = os.path.abspath("simdkernel.so")
sys.argv = [check, *given]
try:
    runpy.run_path(check, run_name="__main__")
    code = 0
except SystemExit as stop:
    code = stop.code if isinstance(stop.code, int) else (print(stop.code) or 1) if stop.code else 0
code
`);
leave(code);
