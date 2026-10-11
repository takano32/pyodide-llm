// tests/tree.mjs (T367.1): where in a tree what is. The one place the tools ask; the rule itself is tests/tree.json
// (tests/tree.py reads the same file for the Python tools).
//
// A tree is this working tree or another commit's (tests/other-tree.mjs). Its files are of four kinds, and each kind is
// found by looking at the tree, so that a tool reads a tree of before the move to the build (T367: everything in
// public/) and one after it (src/runtime/, src/python/, src/runtime/built/) alike, and a tree in between (T367.2: the
// Python moved, the JavaScript not yet):
//
//   runtime   the JavaScript a browser runs: worker.js, forward.js, gpu.js, shaders.js, helper.js, jobs.js, kept.js,
//             their folders, benchmark/
//   python    the Python sources: llama2_numpy.py, engine/, llama2_convert.py, convert/
//   built     what `make kernels` builds (simdkernel*.wasm, simdkernel.so, *.wasmlib, ceilings*.wasm): beside the runtime
//   served    what is served as it is and stays in public/: coi.js, coi-test/, models/
//
//   const tree = treeOf(folder)        (no folder: this tree)
//   tree.runtime("forward.js")         a path; tree.runtimeUrl("forward.js") a file: URL, for import() and new Worker()
//   tree.python("engine/layout.py")    tree.python() is the folder Python's sys.path wants
//   tree.built("simdkernel_plain.wasm")   tree.served("models/tokenizer.bin")
//   tree.folders                       { runtime: "public", python: "public", built: "public", served: "public" }: the
//                                      folders as the tree names them, for a tool's words
//   tree.bundled                       whether a bundler links the runtime's files (static imports), or each is fetched
//                                      as it is and reads its neighbours with its own ?v= (tests/imports.mjs)
//   tree.pythonFiles("llama2_numpy")   the module's files under tree.python(): its window, then every .py of its
//                                      package's folder (a commit of before the packages has the window alone)
//   placePython(pyodide, tree, modules)   those files into Pyodide's file system, where `import` finds them
//   placeKernels(pyodide, tree)        simdkernel.so and simdkernel_relaxed.wasmlib of its built files, beside them
//
// and for this tree, by name:   import { runtime, runtimeUrl, python, built, served } from "./tree.mjs";
//
//   node tests/tree.mjs [--root <a tree>] runtime|python|built|served [<a file>]     the path, for the shell scripts
//
// What is assumed where T365's design leaves a detail open: the marks of tree.json (the runtime is where worker.js is,
// the Python where llama2_numpy.py is: both are moved whole), and that the built files are in built/ under the runtime
// once that is src/runtime/. A layout that differs is tree.json's to say, and tests/tree-check.mjs's to hold.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = fileURLToPath(new URL("..", import.meta.url));
export const RULE = JSON.parse(fs.readFileSync(new URL("tree.json", import.meta.url), "utf8"));

/** The folder (as the tree names it) of the kind whose rule is { mark, folders }: the one that holds the mark */
function found(root, kind, rule) {
  const holding = rule[kind].folders.filter((folder) => fs.existsSync(path.join(root, folder, rule[kind].mark)));
  if (holding.length !== 1) {
    throw new Error(holding.length ? `${root}: ${holding.join(" and ")} both hold ${rule[kind].mark}: a tree half moved`
      : `${root}: no ${rule[kind].mark} in ${rule[kind].folders.join(" or ")}: is this a tree of the project?`);
  }
  return holding[0];
}

export function treeOf(root = HERE, rule = RULE) {
  root = path.resolve(root);
  const folders = { runtime: found(root, "runtime", rule), python: found(root, "python", rule) };
  folders.built = rule.built.beside[folders.runtime];
  if (!folders.built) throw new Error(`tests/tree.json does not say where the built files are when the runtime is in ${folders.runtime}`);
  folders.served = rule.served.folder;
  const at = (kind) => (name = "") => path.join(root, folders[kind], name);
  const tree = { root, folders, bundled: rule.bundled.folders.includes(folders.runtime),
    runtime: at("runtime"), python: at("python"), built: at("built"), served: at("served"),
    runtimeUrl: (name = "") => pathToFileURL(path.join(root, folders.runtime, name)),
    pythonFiles(module) {
      const folder = rule.packages.modules[module];
      if (!folder) throw new Error(`tests/tree.json names no package of ${module}`);
      if (!fs.existsSync(tree.python(`${module}.py`))) throw new Error(`${tree.python(`${module}.py`)} is not there`);
      const parts = !fs.existsSync(tree.python(folder)) ? [] : fs.readdirSync(tree.python(folder), { recursive: true })
        .map((name) => `${folder}/${name.split(path.sep).join("/")}`).filter((name) => name.endsWith(".py") && !name.includes("__pycache__")).sort();
      return [`${module}.py`, ...parts];
    } };
  return tree;
}

/** The modules' Python files of a tree into Pyodide's file system (each folder made first). `under`, where two trees'
 * files are placed side by side: the folder in Pyodide to put them in. Returns the names placed. */
export function placePython(pyodide, tree, modules = Object.keys(RULE.packages.modules), under = "") {
  const names = modules.flatMap((module) => tree.pythonFiles(module));
  for (const name of names) {
    const to = under ? `${under}/${name}` : name;
    if (to.includes("/")) pyodide.FS.mkdirTree(to.slice(0, to.lastIndexOf("/")));
    pyodide.FS.writeFile(to, fs.readFileSync(tree.python(name)));
  }
  return names;
}

/** The kernels Pyodide loads as side modules (ctypes), of a tree's built files, into Pyodide's file system */
export function placeKernels(pyodide, tree, names = ["simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
  for (const name of names) pyodide.FS.writeFile(name, fs.readFileSync(tree.built(name)));
}

// this tree's, by name. (Lazily: a tool that only asks about another tree is not stopped by what this one lacks.)
let here;
const mine = (kind) => (...given) => (here ??= treeOf())[kind](...given);
export const runtime = mine("runtime"), runtimeUrl = mine("runtimeUrl"), python = mine("python"), built = mine("built"), served = mine("served");

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const root = args[0] === "--root" ? args.splice(0, 2)[1] : HERE;
  const [kind, name = ""] = args;
  if (!["runtime", "python", "built", "served"].includes(kind) || args.length > 2) {
    console.error("node tests/tree.mjs [--root <a tree>] runtime|python|built|served [<a file>]");
    process.exit(2);
  }
  console.log(treeOf(root)[kind](name));
}
