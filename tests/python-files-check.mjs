// python-files-check.mjs (T347, T367.1)
// The Python files Pyodide is given are the ones the tree holds, in either way a tree says them:
//   - by a list (python.js in the runtime's folder, today): the list is what the Python's folder holds. A part that
//     is not in the list is not fetched, and the page fails at the import where every test in Node passes (the tools
//     walk the folder: tests/tree.mjs);
//   - by no list (after T367.2, where an archive is made by walking the folder): every .py of the Python's folder is
//     some module's, by the rule the tools walk by (a module's window and every .py of its package: tests/tree.json).
//     A file that is nobody's would be in no archive.
// And in both: a module's window comes first, its parts are in one folder with an __init__.py, and every part of a
// package says where it comes from by the package's own name (`from convert.x import`), so that it is found the same
// in the browser, in Node and under pytest.   node tests/python-files-check.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { RULE, treeOf } from "./tree.mjs";

const tree = treeOf(), where = `${tree.folders.python}/`;
const there = fs.readdirSync(tree.python(), { recursive: true }).map((name) => name.split(path.sep).join("/"))
  .filter((name) => name.endsWith(".py") && !name.includes("__pycache__")).sort();
const walked = Object.fromEntries(Object.keys(RULE.packages).map((module) => [module, tree.pythonFiles(module)]));
const listing = fs.existsSync(tree.runtime("python.js", { maybe: true }));
const PYTHON = listing ? (await import(tree.runtimeUrl("python.js"))).PYTHON : walked;
const listed = Object.values(PYTHON).flat();
assert.deepEqual([...listed].sort(), there, listing ? `python.js's list is not the .py files of ${where}` : `a .py file of ${where} is no module's (a window, or a file of its package's folder: tests/tree.json)`);
assert.equal(new Set(listed).size, listed.length, "a file is listed twice");
// (the list and the walk say the same files of every module: a tool in Node places what the worker fetches)
if (listing) for (const module of Object.keys(PYTHON)) assert.deepEqual([...PYTHON[module]].sort(), [...(walked[module] ?? [])].sort(), `python.js's ${module} is not the window and its package's folder, which the tools walk`);
for (const [module, files] of Object.entries(PYTHON)) {
  assert.equal(files[0], `${module}.py`, `${module}'s window is not first`);
  const folders = new Set(files.slice(1).map((name) => name.split("/")[0]));
  assert.ok(folders.size <= 1, `${module}'s parts are in more than one folder`);
  for (const folder of folders) {
    assert.ok(files.includes(`${folder}/__init__.py`), `${folder}/ has no __init__.py${listing ? " in the list" : ""}`);
    for (const name of files.slice(1)) {
      const relative = fs.readFileSync(tree.python(name), "utf8").match(/^from \.+\w* import|^import \./m);
      assert.equal(relative, null, `${name} imports by a relative name`);
    }
  }
}
console.log(`ok: ${listing ? "python.js lists" : "the modules' windows and packages are"} the ${there.length} Python files of ${where} (${Object.entries(PYTHON).map(([module, files]) => `${module}: ${files.length}`).join(", ")})`);
