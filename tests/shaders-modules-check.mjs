// tests/shaders-modules-check.mjs (T351): the names that public/shaders.js (the window) and its modules
// (public/shaders/*.js) hand one another.
//
// The window takes every name out of a module (`const { a, b } = await modules.x`) and a module takes a neighbour's the
// same way (`= await import(…)`): none of these is a variable until the line runs, a name that is not exported where it
// is taken from is undefined without a word (a shader's text with "undefined" in it, which only a GPU would refuse), and
// tests/device-key-check.mjs reads the modules as one text, where a missing import does not show. So this holds:
//   - the folder holds the modules the window asks for, each asked for with the asker's own ?v= and none by a static
//     import (GitHub Pages keeps a file for ten minutes: all must come from one deployment);
//   - every name a file uses is declared in it, taken from a neighbour, or one of the globals below; every name taken is
//     exported where it is taken from;
//   - the window's list has each module after those it takes from (so none waits for itself: a ring of awaited imports
//     never ends, and tests/shaders-source.mjs's oneSource() is in an order that runs);
//   - the window exports every name a module exports where it is declared, once, and nothing else: a name dropped from
//     the window is gone for gpu.js and /benchmark/, and a name a module hands only its neighbours (its last line's)
//     is no export of the window's;
//   - the window links in Node with every export defined, and what it exports is each module's own value.
// Node only, under a second:
//
//   node tests/shaders-modules-check.mjs
//
// The parser is @babel/parser, which Astro's packages bring (as tests/worker-modules-check.mjs, whose namesOf this uses).
import assert from "node:assert/strict";
import fs from "node:fs";
import { namesOf } from "./worker-modules-check.mjs";
import { shaderParts } from "./shaders-source.mjs";
import { treeOf } from "./tree.mjs";

const GLOBALS = new Set("Array ArrayBuffer Boolean Error Float32Array Float64Array Infinity Int8Array Math NaN Number Object Set URL Uint16Array Uint32Array Uint8Array globalThis undefined".split(" "));
// (T367.1: where the runtime is and which form its imports have is the tree's, tests/tree.mjs; that the folder holds the
// modules the window asks for, or, where a bundler links them, that the window reaches every file of it, is
// tests/imports.mjs's windowed(), which shaderParts() reads the files with)
const tree = treeOf();
const { window: windowText, linking, list, parts } = shaderParts(tree);
const there = fs.readdirSync(tree.runtime("shaders")).filter((name) => name.endsWith(".js")).map((name) => name.slice(0, -3)).sort();
assert.deepEqual([...list].sort(), there, `the modules shaders.js asks for are the files of ${tree.folders.runtime}/shaders/`);
assert.equal(new Set(list).size, list.length, "a module asked for twice");

// ---- the modules: every name used is declared, taken from a neighbour, or a global; what is taken is exported there
const byName = new Map(parts.map((part) => [part.name, part]));
let taken = 0;
for (const part of parts) {
  const unknown = [...namesOf(part.text).free].filter((used) => !GLOBALS.has(used));
  assert.deepEqual(unknown, [], `${part.file} uses ${unknown}, which it neither declares nor takes from a neighbour: a ReferenceError as the module loads, or where the line runs`);
  assert.equal(new Set([...part.declared, ...part.given]).size, part.declared.length + part.given.length, `${part.file} exports a name twice`);
  for (const [from, names] of part.takes) {
    assert.ok(byName.has(from), `${part.file} takes names from ${from}.js, which is no module of the window's list`);
    assert.ok(list.indexOf(from) < list.indexOf(part.name), `${part.file} takes from ${from}.js, which the window's list has after it (or the two wait for one another)`);
    const exported = new Set([...byName.get(from).declared, ...byName.get(from).given]);
    for (const wanted of names) {
      assert.ok(exported.has(wanted), `${part.file} takes ${wanted} from ${from}.js, which does not export it: undefined`);
      taken += 1;
    }
  }
}
// (a name a module's last line hands out is one a neighbour takes: the list is not a second window)
for (const part of parts) for (const name of part.given) {
  assert.ok(parts.some((other) => other.takes.get(part.name)?.includes(name)), `${part.file} exports ${name} for its neighbours, and none takes it`);
}
console.log(`shaders-modules-check: the ${parts.length} modules of ${tree.folders.runtime}/shaders/ use nothing they do not declare or take from a neighbour, ` +
  `the ${taken} names they take are exported, and each comes after those it takes from`);

// ---- the window: what it takes of each module, and what it exports
const unknown = [...namesOf(windowText).free].filter((used) => !GLOBALS.has(used));
assert.deepEqual(unknown, [], `shaders.js uses ${unknown}, which it neither declares nor takes from a module`);
const windowTakes = new Map();
for (const { from, names } of linking.takes) {
  const module = from.slice("shaders/".length, -".js".length);
  assert.ok(byName.has(module), `shaders.js takes names from ${from}, which is no module of its list`);
  assert.ok(!windowTakes.has(module), `shaders.js takes from ${from} twice`);
  windowTakes.set(module, names.map(({ imported, local }) => (assert.equal(imported, local, "shaders.js: a plain name"), imported)));
}
let exportsOfWindow = null;
for (const node of linking.program.body) {
  if (linking.left.includes(node)) continue;  // (the list of its modules, and what it takes of each)
  if (node.type === "ExportNamedDeclaration") {
    assert.ok(!node.declaration && !node.source && exportsOfWindow === null, "shaders.js exports one list of names, and declares nothing of its own");
    exportsOfWindow = node.specifiers.map((specifier) => (assert.equal(specifier.local.name, specifier.exported.name, "a name exported under another"), specifier.exported.name));
    continue;
  }
  assert.fail(`shaders.js: a ${node.type} at its top (\`${windowText.slice(node.start, node.end).slice(0, 80)}\`): the window takes and exports, and nothing else`);
}
assert.ok(exportsOfWindow, "shaders.js exports a list of names");
assert.equal(new Set(exportsOfWindow).size, exportsOfWindow.length, "shaders.js exports a name twice");
for (const part of parts) {
  // (what a module exports where it is declared is what the one file exported; its last line's names are its neighbours')
  assert.deepEqual([...(windowTakes.get(part.name) ?? [])].sort(), [...part.declared].sort(),
    `shaders.js takes of ${part.file} other names than the module exports where they are declared: one dropped is gone for gpu.js and /benchmark/, one more is undefined`);
}
assert.deepEqual([...exportsOfWindow].sort(), [...windowTakes.values()].flat().sort(), "shaders.js exports the names it takes of its modules, all of them and no other");
console.log(`shaders-modules-check: the window takes the ${exportsOfWindow.length} names its modules export where they are declared, and exports each once`);

// ---- the window in Node: it links, every export is defined and is the module's own
const real = await import(tree.runtimeUrl("shaders.js"));
assert.deepEqual(Object.keys(real).sort(), [...exportsOfWindow].sort(), "what shaders.js exports when it is imported");
for (const part of parts) {
  const module = await import(tree.runtimeUrl(part.file));
  for (const name of part.declared) {
    assert.notEqual(real[name], undefined, `shaders.js exports ${name} undefined`);
    assert.equal(real[name], module[name], `shaders.js's ${name} is not ${part.file}'s`);
  }
}
console.log(`shaders-modules-check: shaders.js links (${Object.keys(real).length} exports, each its module's own)`);
console.log("shaders-modules-check: ok");
