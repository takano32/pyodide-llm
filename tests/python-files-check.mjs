// python-files-check.mjs (T347)
// public/python.js's list of the Python files Pyodide is given is what public/ holds: a part that is not in the list
// is not fetched, and the page fails at the import where every test in Node passes (they read the same list, so a
// file the list lacks is missing there too, but one written to the folder and forgotten in the list only here).
// And every part of a package says where it comes from by the package's own name (`from convert.x import`), so that
// it is found the same in the browser, in Node and under pytest.   node tests/python-files-check.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { PYTHON } from "../public/python.js";

const root = fileURLToPath(new URL("../public/", import.meta.url));
const there = fs.readdirSync(root, { recursive: true }).map((name) => name.split("\\").join("/"))
  .filter((name) => name.endsWith(".py") && !name.includes("__pycache__")).sort();
const listed = Object.values(PYTHON).flat();
assert.deepEqual([...listed].sort(), there, "python.js's list is not the .py files of public/");
assert.equal(new Set(listed).size, listed.length, "a file is listed twice");
for (const [module, files] of Object.entries(PYTHON)) {
  assert.equal(files[0], `${module}.py`, `${module}'s window is not first`);
  const folders = new Set(files.slice(1).map((name) => name.split("/")[0]));
  assert.ok(folders.size <= 1, `${module}'s parts are in more than one folder`);
  for (const folder of folders) {
    assert.ok(files.includes(`${folder}/__init__.py`), `${folder}/ has no __init__.py in the list`);
    for (const name of files.slice(1)) {
      const relative = fs.readFileSync(root + name, "utf8").match(/^from \.+\w* import|^import \./m);
      assert.equal(relative, null, `${name} imports by a relative name`);
    }
  }
}
console.log(`ok: python.js lists the ${there.length} Python files of public/ (${Object.entries(PYTHON).map(([module, files]) => `${module}: ${files.length}`).join(", ")})`);
