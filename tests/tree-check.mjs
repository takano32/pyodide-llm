// tests/tree-check.mjs (T367.1): tests/tree.mjs and tests/tree.py alone, on made-up trees under .tmp/. A file is found
// by looking at the tree asked about, wherever it lies: every layout the tools are to read (all in public/, the Python
// moved, both moved, and one nobody planned), what stops a search from taking the wrong file (found twice, found
// nowhere, the folders that are never entered, links, one walk a process, another tree searched under its own root),
// the walk of a module's Python files, the placing into a stand-in Pyodide, and that the Python side answers what the
// JavaScript side does. Node and python, about a second:
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
const AS_IT_IS = "const { BATCH } = await import(new URL(`jobs.js${new URL(import.meta.url).search}`, import.meta.url));\n", LINKED = 'import { BATCH } from "./jobs.js";\n';

// ---- the layouts: nothing of them is written in tree.mjs, tree.py or tree.json
const LAYOUTS = {
  "all in public": { files: { "public/forward.js": AS_IT_IS, "public/llama2_numpy.py": "", "public/simdkernel.so": "", "public/coi.js": "", "public/gpu.js": "", "public/gpu/device.js": "",
    "public/benchmark/gpu.js": "", "public/benchmark/gpu/device.js": "", "public/coi-test/worker.js": "", "public/worker.js": "" },
    folders: { runtime: "public", python: "public", built: "public", served: "public" }, bundled: false },
  "the Python moved": { files: { "public/forward.js": AS_IT_IS, "src/python/llama2_numpy.py": "", "public/simdkernel.so": "", "public/coi.js": "", "public/gpu.js": "" },
    folders: { runtime: "public", python: "src/python", built: "public", served: "public" }, bundled: false },
  "both moved": { files: { "src/runtime/forward.js": LINKED, "src/python/llama2_numpy.py": "", "src/runtime/built/simdkernel.so": "", "public/coi.js": "", "src/runtime/gpu.js": "",
    "src/runtime/benchmark/gpu.js": "" },
    folders: { runtime: "src/runtime", python: "src/python", built: "src/runtime/built", served: "public" }, bundled: true },
  "one nobody planned": { files: { "app/js/forward.js": LINKED, "py/llama2_numpy.py": "", "out/wasm/simdkernel.so": "", "coi.js": "", "app/js/gpu.js": "" },
    folders: { runtime: "app/js", python: "py", built: "out/wasm", served: "" }, bundled: true },
  // (nothing built: the place is the one the tree's Makefile builds the kernels in)
  "nothing built yet": { files: { "public/forward.js": AS_IT_IS, "public/llama2_numpy.py": "", "public/coi.js": "", "public/gpu.js": "", "Makefile": "kernels:\tlib/out/simdkernel.so\nlib/out/simdkernel.so:\tkernels/kernel.ts\n\tpython kernels/build.py\n" },
    folders: { runtime: "public", python: "public", built: "lib/out", served: "public" }, bundled: false },
};
for (const [name, { files, folders, bundled }] of Object.entries(LAYOUTS)) {
  const folder = madeUp(name, files), tree = treeOf(folder);
  assert.deepEqual(tree.folders, folders, `${name}: the folders`);
  assert.equal(tree.bundled, bundled, `${name}: whether a bundler links the runtime`);
  assert.equal(tree.root, folder);
  assert.equal(tree.runtime("gpu.js"), path.join(folder, folders.runtime, "gpu.js"), `${name}: a name under its kind, where another file of the tree ends the same`);
  for (const kind of KINDS) {
    assert.equal(tree[kind](), path.join(folder, folders[kind]), `${name}: ${kind}'s folder`);
    assert.equal(tree[kind]("a/b.x", { maybe: true }), path.join(folder, folders[kind], "a/b.x"), `${name}: where a file of ${kind} would be`);
    // (the Python side and both command lines say the same)
    const [py, js] = [pythonOf("--root", folder, kind), nodeOf("--root", folder, kind)];
    assert.equal(py.status, 0, `${name}: tree.py ${kind}: ${py.stderr}`);
    assert.equal(py.stdout.trim(), tree[kind](), `${name}: tree.py's ${kind}`);
    assert.equal(js.stdout.trim(), tree[kind](), `${name}: tree.mjs's command line, ${kind}`);
  }
  for (const [kind, file] of [["runtime", "gpu.js"], ["python", "llama2_numpy.py"], ["served", "coi.js"]]) {
    assert.equal(pythonOf("--root", folder, kind, file).stdout.trim(), tree[kind](file), `${name}: tree.py's ${file}`);
    assert.equal(nodeOf("--root", folder, kind, file).stdout.trim(), tree[kind](file), `${name}: tree.mjs's command line, ${file}`);
  }
  assert.equal(tree.runtimeUrl("forward.js").href, pathToFileURL(path.join(folder, folders.runtime, "forward.js")).href, `${name}: a URL of the runtime`);
  assert.equal(tree.walks, 1, `${name}: the tree was walked once for all of that`);
  assert.equal(treeOf(folder), tree, `${name}: and is remembered`);
}

// ---- what stops a search from taking the wrong file
{
  const said = (folder, kind, name, words) => {
    assert.throws(() => treeOf(folder)[kind](name), words, `${path.basename(folder)}: ${kind} ${name}`);
    const py = pythonOf("--root", folder, kind, ...(name ? [name] : [])), js = nodeOf("--root", folder, kind, ...(name ? [name] : []));
    assert.notEqual(py.status, 0, `tree.py refuses ${kind} ${name}`);
    assert.match(py.stderr, words, "tree.py's words");
    assert.equal(js.status, 1, `tree.mjs's command line refuses ${kind} ${name}`);
    assert.match(js.stderr, words, "the command line's words");
  };
  const whole = path.join(scratch, "all in public");
  // a name that is not under its kind is looked for in the whole tree: found once, found twice (both named), found nowhere
  assert.equal(treeOf(whole).runtime("coi-test/worker.js"), path.join(whole, "public/coi-test/worker.js"));
  assert.equal(treeOf(whole).python("gpu/device.js"), path.join(whole, "public/gpu/device.js"), "under the kind first: gpu/device.js is public/gpu/device.js, not benchmark's");
  said(whole, "runtime", "device.js", /device\.js is not in public\/ \(the runtime files\), and is in 2 places: public\/benchmark\/gpu\/device\.js and public\/gpu\/device\.js/);
  said(whole, "served", "nothing.bin", /no nothing\.bin in public\/ \(the served files\) nor anywhere in the tree \(looked in every folder but node_modules, dist, \.tmp/);
  const elsewhere = madeUp("a file elsewhere", { "public/forward.js": "", "public/llama2_numpy.py": "", "public/coi.js": "", "public/simdkernel.so": "", "extra/deep/only.js": "", "x/twice.js": "", "y/twice.js": "" });
  assert.equal(treeOf(elsewhere).runtime("only.js"), path.join(elsewhere, "extra/deep/only.js"), "by its name alone");
  assert.equal(treeOf(elsewhere).runtime("deep/only.js"), path.join(elsewhere, "extra/deep/only.js"), "by the end of its path");
  said(elsewhere, "runtime", "eep/only.js", /no eep\/only\.js/);  // (whole names: not the end of a name)
  said(elsewhere, "runtime", "twice.js", /is in 2 places: x\/twice\.js and y\/twice\.js/);
  assert.equal(treeOf(elsewhere).runtime("x/twice.js"), path.join(elsewhere, "x/twice.js"), "a path long enough to be one file");
  // a kind's anchor in two places (a tree half moved), or nowhere
  said(madeUp("half moved, the runtime", { "public/forward.js": "", "src/runtime/forward.js": "", "public/llama2_numpy.py": "" }), "runtime", "",
    /forward\.js \(what says where the runtime files are\) is in 2 places: public\/forward\.js and src\/runtime\/forward\.js/);
  said(madeUp("half moved, the Python", { "public/forward.js": "", "src/python/llama2_numpy.py": "", "public/llama2_numpy.py": "" }), "python", "engine/layout.py", /llama2_numpy\.py .* is in 2 places/);
  said(madeUp("no runtime", { "public/llama2_numpy.py": "" }), "runtime", "forward.js", /no forward\.js \(what says where the runtime files are\) anywhere in the tree/);
  said(madeUp("nothing built, no Makefile", { "public/forward.js": "" }), "built", "", /no simdkernel\.so anywhere in the tree, and no rule of its Makefile makes one/);
  // (a kind that is there is found whatever another kind lacks)
  assert.equal(treeOf(path.join(scratch, "half moved, the runtime")).python(), path.join(scratch, "half moved, the runtime", "public"));
  // the folders that are never entered, and links: a second anchor in any of them is not seen
  const hidden = madeUp("hidden copies", { "public/forward.js": "", "public/llama2_numpy.py": "", "public/coi.js": "", "public/simdkernel.so": "",
    ...Object.fromEntries(RULE.skip.flatMap((skipped) => [[`${skipped}/forward.js`, ""], [`public/${skipped}/llama2_numpy.py`, ""], [`${skipped}/deep/hidden.js`, ""]])),
    ".tmp/unchanged/0123abc/public/forward.js": "", ".tmp/unchanged/0123abc/public/llama2_numpy.py": "", ".tmp/unchanged/0123abc/public/coi.js": "", ".tmp/unchanged/0123abc/src/only-there.js": "" });
  fs.symlinkSync(path.join(scratch, "all in public"), path.join(hidden, "linked"));
  fs.symlinkSync(path.join(scratch, "all in public", "public", "forward.js"), path.join(hidden, "forward.js"));
  assert.deepEqual(treeOf(hidden).folders, { runtime: "public", python: "public", built: "public", served: "public" });
  said(hidden, "runtime", "hidden.js", /no hidden\.js/);
  said(hidden, "runtime", "only-there.js", /no only-there\.js/);
  for (const kind of ["runtime", "python", "served"]) assert.equal(pythonOf("--root", hidden, kind).stdout.trim(), treeOf(hidden)[kind](), `tree.py skips the same, ${kind}`);
  // another tree's copy under this one's .tmp is searched under its own root, when it is the tree asked about
  const other = path.join(hidden, ".tmp/unchanged/0123abc");
  assert.equal(treeOf(other).runtime("forward.js"), path.join(other, "public/forward.js"));
  assert.equal(treeOf(other).runtime("only-there.js"), path.join(other, "src/only-there.js"));
  assert.equal(pythonOf("--root", other, "python").stdout.trim(), path.join(other, "public"));
  for (const bad of [[], ["kernels"], ["python", "a", "b"]]) {
    assert.equal(nodeOf(...bad).status, 2, `tree.mjs ${bad}: usage`);
    assert.notEqual(pythonOf(...bad).status, 0, `tree.py ${bad}: usage`);
  }
}

// ---- no place is written down: neither file names a folder of the project, and another rule's anchors are found as well
{
  const code = (file, strip) => strip(fs.readFileSync(new URL(file, import.meta.url), "utf8"));
  const sources = { "tree.mjs": code("tree.mjs", (text) => text.split("\n").filter((line) => !line.trim().startsWith("//")).join("\n")),
    "tree.py": code("tree.py", (text) => text.split('"""').slice(2).join('"""')), "tree.json": code("tree.json", (text) => JSON.stringify({ ...JSON.parse(text), about: "" })) };
  for (const [file, text] of Object.entries(sources)) for (const place of ["public", "src/", "runtime/", "python/", "built/"]) assert.ok(!text.includes(`"${place}`) && !text.includes(`/${place}`), `tests/${file} names ${place}: a place is found, not written`);
  const folder = madeUp("another rule", { "lib/main.js": 'import { a } from "../b.js";\n', "py/start.py": "", "py/pkg/a.py": "", "o/k.so": "", "static/sw.js": "", "node_modules/main.js": "", "vendor/main.js": "" });
  const rule = { anchors: { runtime: "main.js", python: "start.py", built: "k.so", served: "sw.js" }, skip: ["node_modules", "vendor"], packages: { start: "pkg" } };
  const tree = treeOf(folder, rule);
  assert.deepEqual(tree.folders, { runtime: "lib", python: "py", built: "o", served: "static" });
  assert.equal(tree.bundled, true);
  assert.deepEqual(tree.pythonFiles("start"), ["start.py", "pkg/a.py"]);
  assert.throws(() => treeOf(folder, { ...rule, skip: ["node_modules"] }).runtime(), /main\.js .* is in 2 places: lib\/main\.js and vendor\/main\.js/);
}

// ---- a module's Python files: the window, then every .py of its package's folder, in one order
{
  const folder = madeUp("python files", { "public/forward.js": "", "public/coi.js": "",
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
  // the kernels Pyodide loads, from wherever the tree's built files are
  const builtTree = treeOf(madeUp("kernels", { "public/forward.js": "", "any/where/simdkernel.so": "built .so", "any/where/simdkernel_relaxed.wasmlib": "built .wasmlib" }));
  written.clear();
  placeKernels(pyodide, builtTree);
  assert.deepEqual([...written], [["simdkernel.so", "built .so"], ["simdkernel_relaxed.wasmlib", "built .wasmlib"]]);
  assert.throws(() => placeKernels(pyodide, builtTree, ["simdkernel_other.so"]), /no simdkernel_other\.so in any\/where\/ \(the built files\)/);
  // a commit of before the packages: the window alone; a module whose window is missing is said
  const old = treeOf(madeUp("one file", { "public/forward.js": "", "public/llama2_numpy.py": "all of it" }));
  assert.deepEqual(old.pythonFiles("llama2_numpy"), ["llama2_numpy.py"]);
  assert.throws(() => old.pythonFiles("llama2_convert"), /no llama2_convert\.py/);
}

// ---- this tree, by name
{
  const tree = treeOf();
  assert.equal(tree.root, path.resolve(root));
  assert.equal(runtime("forward.js"), tree.runtime("forward.js"));
  assert.equal(python(), tree.python());
  assert.equal(served("coi.js"), tree.served("coi.js"));
  assert.equal(built("simdkernel_plain.wasm", { maybe: true }), tree.built("simdkernel_plain.wasm", { maybe: true }));
  assert.equal(runtimeUrl("jobs.js").href, tree.runtimeUrl("jobs.js").href);
  for (const file of [runtime("worker.js"), runtime("gpu.js"), runtime("benchmark/gpu.js"), runtime("gpu/device.js"), python("llama2_convert.py"), python("convert/conduct.py")]) assert.ok(fs.existsSync(file), `${file} is there`);
  assert.notEqual(runtime("gpu.js"), runtime("benchmark/gpu.js"));
  // every module the rule names is there, with its package, and no .py beside them is nobody's
  const all = Object.keys(RULE.packages).flatMap((module) => tree.pythonFiles(module));
  for (const module of Object.keys(RULE.packages)) assert.ok(tree.pythonFiles(module).length > 1, `${module} has a package here`);
  const there = fs.readdirSync(tree.python(), { recursive: true }).map((name) => name.split(path.sep).join("/")).filter((name) => name.endsWith(".py") && !name.includes("__pycache__")).sort();
  assert.deepEqual([...all].sort(), there, `the .py files of ${tree.folders.python}/ are the modules' and no other`);
  assert.equal(pythonOf("python").stdout.trim(), tree.python(), "tree.py's this tree is this tree");
  assert.equal(tree.walks, 1, "this tree was walked once");
  // (the cost: one walk a process)
  const began = performance.now();
  treeOf(root, { ...RULE }).runtime();
  console.log(`tree-check: this tree: the runtime in ${tree.folders.runtime}/, the Python in ${tree.folders.python}/ (${all.length} files), ` +
    `the built files in ${tree.folders.built}/${tree.bundled ? ", bundled" : ""}; a walk of it takes ${(performance.now() - began).toFixed(1)} ms`);
}
fs.rmSync(scratch, { recursive: true, force: true });
console.log(`tree-check: ok (${Object.keys(LAYOUTS).length} layouts found by looking, by tree.mjs and tree.py alike)`);
