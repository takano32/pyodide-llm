// tests/tree-check.mjs (T367.1): tests/tree.mjs and tests/tree.py alone, on made-up trees under .tmp/: every layout the
// tools are to read (all in public/, the Python moved, both moved), the trees they are to refuse (half moved, no tree),
// the walk of a module's Python files, the placing into a stand-in Pyodide, and that the Python side answers what the
// JavaScript side does (one rule, tests/tree.json). Node and python, under a second:
//
//   node tests/tree-check.mjs       (PYTHON=.venv/bin/python)
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { RULE, built, placeKernels, placePython, python, runtime, runtimeUrl, served, treeOf } from "./tree.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const scratch = path.join(root, ".tmp", "tree-check", String(process.pid));
fs.rmSync(scratch, { recursive: true, force: true });
/** a made-up tree: its files (a name -> a text) under a folder of its own */
function madeUp(name, files) {
  const folder = path.join(scratch, name);
  for (const [file, text] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(folder, file)), { recursive: true });
    fs.writeFileSync(path.join(folder, file), text);
  }
  return folder;
}
const pythonOf = (...args) => spawnSync(process.env.PYTHON ?? "python3", [path.join(root, "tests/tree.py"), ...args], { encoding: "utf8" });
const nodeOf = (...args) => spawnSync(process.execPath, [path.join(root, "tests/tree.mjs"), ...args], { encoding: "utf8" });
const KINDS = ["runtime", "python", "built", "served"];

// ---- the layouts
const LAYOUTS = {
  "all in public": { files: { "public/worker.js": "", "public/llama2_numpy.py": "" },
    folders: { runtime: "public", python: "public", built: "public", served: "public" }, bundled: false },
  "the Python moved": { files: { "public/worker.js": "", "src/python/llama2_numpy.py": "" },
    folders: { runtime: "public", python: "src/python", built: "public", served: "public" }, bundled: false },
  "both moved": { files: { "src/runtime/worker.js": "", "src/python/llama2_numpy.py": "", "public/coi.js": "" },
    folders: { runtime: "src/runtime", python: "src/python", built: "src/runtime/built", served: "public" }, bundled: true },
};
for (const [name, { files, folders, bundled }] of Object.entries(LAYOUTS)) {
  const folder = madeUp(name, files), tree = treeOf(folder);
  assert.deepEqual(tree.folders, folders, `${name}: the folders`);
  assert.equal(tree.bundled, bundled, `${name}: whether a bundler links the runtime`);
  assert.equal(tree.root, folder);
  for (const kind of KINDS) {
    assert.equal(tree[kind](), path.join(folder, folders[kind]), `${name}: ${kind}'s folder`);
    assert.equal(tree[kind]("a/b.x"), path.join(folder, folders[kind], "a/b.x"), `${name}: a file of ${kind}`);
    // (the Python side and both command lines say the same, with a file and without)
    for (const file of [[], ["a/b.x"]]) {
      const [py, js] = [pythonOf("--root", folder, kind, ...file), nodeOf("--root", folder, kind, ...file)];
      assert.equal(py.status, 0, `${name}: tree.py ${kind}: ${py.stderr}`);
      assert.equal(py.stdout.trim(), tree[kind](...file), `${name}: tree.py's ${kind} ${file}`);
      assert.equal(js.stdout.trim(), tree[kind](...file), `${name}: tree.mjs's command line, ${kind} ${file}`);
    }
  }
  assert.equal(tree.runtimeUrl("worker.js").href, pathToFileURL(path.join(folder, folders.runtime, "worker.js")).href, `${name}: a URL of the runtime`);
}
// (a kind is found alone: the runtime's folder does not follow the Python's, nor the other way)
assert.notEqual(treeOf(path.join(scratch, "the Python moved")).folders.runtime, treeOf(path.join(scratch, "the Python moved")).folders.python);

// ---- the trees that are refused, by both sides
const REFUSED = {
  "half moved, the runtime": [{ "public/worker.js": "", "src/runtime/worker.js": "", "public/llama2_numpy.py": "" }, /src\/runtime and public both hold worker\.js/],
  "half moved, the Python": [{ "public/worker.js": "", "src/python/llama2_numpy.py": "", "public/llama2_numpy.py": "" }, /src\/python and public both hold llama2_numpy\.py/],
  "no runtime": [{ "public/llama2_numpy.py": "" }, /no worker\.js in src\/runtime or public/],
  "no Python": [{ "public/worker.js": "" }, /no llama2_numpy\.py in src\/python or public/],
  "the marks elsewhere": [{ "src/worker.js": "", "src/runtime/built/worker.js": "", "src/llama2_numpy.py": "" }, /no worker\.js/],
};
for (const [name, [files, words]] of Object.entries(REFUSED)) {
  const folder = madeUp(name, files);
  assert.throws(() => treeOf(folder), words, `${name}: refused`);
  const py = pythonOf("--root", folder, "python"), js = nodeOf("--root", folder, "python");
  assert.notEqual(py.status, 0, `${name}: tree.py refuses`);
  assert.match(py.stderr, words, `${name}: tree.py's words`);
  assert.notEqual(js.status, 0, `${name}: tree.mjs's command line refuses`);
}
for (const bad of [[], ["kernels"], ["python", "a", "b"]]) {
  assert.equal(nodeOf(...bad).status, 2, `tree.mjs ${bad}: usage`);
  assert.notEqual(pythonOf(...bad).status, 0, `tree.py ${bad}: usage`);
}

// ---- the rule is the JSON's: another rule, another answer (nothing in tree.mjs names a folder)
{
  const folder = madeUp("another rule", { "lib/main.js": "", "py/start.py": "", "py/pkg/a.py": "" });
  const rule = { runtime: { mark: "main.js", folders: ["lib"] }, python: { mark: "start.py", folders: ["py"] }, built: { beside: { lib: "out" } },
    served: { folder: "static" }, bundled: { folders: ["lib"] }, packages: { modules: { start: "pkg" } } };
  const tree = treeOf(folder, rule);
  assert.deepEqual(tree.folders, { runtime: "lib", python: "py", built: "out", served: "static" });
  assert.equal(tree.bundled, true);
  assert.deepEqual(tree.pythonFiles("start"), ["start.py", "pkg/a.py"]);
  assert.throws(() => treeOf(folder, { ...rule, built: { beside: {} } }), /does not say where the built files are when the runtime is in lib/);
  const source = fs.readFileSync(new URL("tree.mjs", import.meta.url), "utf8").split("\n").filter((line) => !line.startsWith("//")).join("\n");
  for (const folderName of ["public", "src/runtime", "src/python"]) assert.ok(!source.includes(`"${folderName}`), `tree.mjs names ${folderName} itself: the rule is tree.json's`);
  const pythonSource = fs.readFileSync(new URL("tree.py", import.meta.url), "utf8").split('"""').slice(2).join('"""');
  for (const folderName of ["public", "src/runtime", "src/python"]) assert.ok(!pythonSource.includes(folderName), `tree.py names ${folderName} itself: the rule is tree.json's`);
}

// ---- a module's Python files: the window, then every .py of its package's folder, in one order
{
  const folder = madeUp("python files", { "public/worker.js": "",
    "src/python/llama2_numpy.py": "window", "src/python/engine/__init__.py": "", "src/python/engine/model.py": "m", "src/python/engine/layout.py": "l",
    "src/python/engine/__pycache__/model.cpython-314.pyc": "x", "src/python/engine/__pycache__/stale.py": "x", "src/python/engine/notes.txt": "x",
    "src/python/llama2_convert.py": "c", "src/python/convert/__init__.py": "", "src/python/convert/families/__init__.py": "", "src/python/convert/families/llama.py": "f",
    "src/python/stray.py": "nobody's", "src/python/other/x.py": "nobody's" });
  const tree = treeOf(folder);
  assert.deepEqual(tree.pythonFiles("llama2_numpy"), ["llama2_numpy.py", "engine/__init__.py", "engine/layout.py", "engine/model.py"]);
  assert.deepEqual(tree.pythonFiles("llama2_convert"), ["llama2_convert.py", "convert/__init__.py", "convert/families/__init__.py", "convert/families/llama.py"]);
  assert.throws(() => tree.pythonFiles("numpy"), /names no package of numpy/);
  // a stand-in Pyodide: what is made and written
  const made = [], written = new Map();
  const pyodide = { FS: { mkdirTree: (name) => made.push(name), writeFile: (name, bytes) => written.set(name, Buffer.from(bytes).toString()) } };
  const names = placePython(pyodide, tree);
  assert.deepEqual(names, [...tree.pythonFiles("llama2_numpy"), ...tree.pythonFiles("llama2_convert")], "both modules by default, the engine first");
  assert.deepEqual([...written.keys()], names);
  assert.equal(written.get("llama2_numpy.py"), "window");
  assert.equal(written.get("convert/families/llama.py"), "f");
  assert.ok(made.indexOf("convert/families") >= 0 && !made.includes(""), "a folder is made before its file, and none for the windows");
  for (const name of names) if (name.includes("/")) assert.ok(made.includes(name.slice(0, name.lastIndexOf("/"))), `${name}'s folder was made`);
  written.clear();
  assert.deepEqual(placePython(pyodide, tree, ["llama2_numpy"], "before"), tree.pythonFiles("llama2_numpy"));
  assert.deepEqual([...written.keys()], tree.pythonFiles("llama2_numpy").map((name) => `before/${name}`), "under a folder of its own, where two trees are placed");
  assert.ok(made.includes("before") && made.includes("before/engine"));
  // the kernels Pyodide loads, from the tree's built files (here beside a runtime that is still in public/)
  for (const name of ["simdkernel.so", "simdkernel_relaxed.wasmlib"]) fs.writeFileSync(path.join(folder, "public", name), `built ${name}`);
  written.clear();
  placeKernels(pyodide, tree);
  assert.deepEqual([...written], [["simdkernel.so", "built simdkernel.so"], ["simdkernel_relaxed.wasmlib", "built simdkernel_relaxed.wasmlib"]]);
  assert.throws(() => placeKernels(pyodide, treeOf(path.join(scratch, "both moved"))), /src\/runtime\/built\/simdkernel\.so/, "from built/ beside a moved runtime, where nothing is built here");
  // a commit of before the packages: the window alone; a module whose window is missing is said
  const old = treeOf(madeUp("one file", { "public/worker.js": "", "public/llama2_numpy.py": "all of it" }));
  assert.deepEqual(old.pythonFiles("llama2_numpy"), ["llama2_numpy.py"]);
  assert.throws(() => old.pythonFiles("llama2_convert"), /llama2_convert\.py is not there/);
}

// ---- this tree, by name
{
  const tree = treeOf();
  assert.equal(tree.root, path.resolve(root));
  assert.equal(runtime("forward.js"), tree.runtime("forward.js"));
  assert.equal(python(), tree.python());
  assert.equal(built("simdkernel_plain.wasm"), tree.built("simdkernel_plain.wasm"));
  assert.equal(served("coi.js"), tree.served("coi.js"));
  assert.equal(runtimeUrl("jobs.js").href, tree.runtimeUrl("jobs.js").href);
  for (const file of [runtime("worker.js"), runtime("forward.js"), python("llama2_numpy.py"), python("llama2_convert.py"), served("coi.js")]) assert.ok(fs.existsSync(file), `${file} is there`);
  // every module the rule names is there, with its package, and no .py of the Python's folder is nobody's
  const all = Object.keys(RULE.packages.modules).flatMap((module) => tree.pythonFiles(module));
  for (const module of Object.keys(RULE.packages.modules)) assert.ok(tree.pythonFiles(module).length > 1, `${module} has a package here`);
  const there = fs.readdirSync(tree.python(), { recursive: true }).map((name) => name.split(path.sep).join("/"))
    .filter((name) => name.endsWith(".py") && !name.includes("__pycache__") && (tree.folders.python !== tree.folders.served || !name.startsWith("models/"))).sort();
  assert.deepEqual([...all].sort(), there, `the .py files of ${tree.folders.python}/ are the modules' and no other`);
  assert.equal(pythonOf("python").stdout.trim(), tree.python(), "tree.py's this tree is this tree");
  console.log(`tree-check: this tree: the runtime in ${tree.folders.runtime}/, the Python in ${tree.folders.python}/ (${all.length} files), ` +
    `the built files in ${tree.folders.built}/${tree.bundled ? ", bundled" : ""}`);
}
fs.rmSync(scratch, { recursive: true, force: true });
console.log(`tree-check: ok (${Object.keys(LAYOUTS).length} layouts and ${Object.keys(REFUSED).length} refused trees, by tree.mjs and tree.py alike)`);
