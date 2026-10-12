// tests/shaders-modules-check.mjs (T351): the names that src/runtime/shaders.js (the window) and its modules
// (src/runtime/shaders/*.js) hand one another.
//
// The window imports every name of every module and exports it; a module imports a neighbour's. /benchmark/'s GPU
// worker and the tools read the window as one object of names (shared.WGSL.<name>, wgsl.<name>): a name dropped from
// it is undefined there without a word. And tests/device-key-check.mjs reads the modules as one text, where a missing
// import does not show. So this holds:
//   - every file of shaders/ is a module the window takes names of;
//   - every name a file uses is declared in it, imported by it, or one of the globals below (a ReferenceError where
//     the line runs, otherwise: neither the build nor Node says it before);
//   - each module comes after those it takes from (tests/shaders-source.mjs's oneSource() is in an order that runs;
//     the build links a ring without a word);
//   - the window exports every name a module exports where it is declared, once, and nothing else: a name dropped from
//     the window is gone for gpu.js and /benchmark/, and a name a module hands only its neighbours (its last line's)
//     is no export of the window's;
//   - the window links in Node with every export defined, and what it exports is each module's own value.
// T367.5: what this held before and the build and Node hold by themselves now is gone from here: that a name taken is
// exported where it is taken from (the build: MISSING_EXPORT, an error; Node: a SyntaxError as the module is imported).
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
// (where the runtime is: tests/tree.mjs. That the window reaches every file of the folder is tests/imports.mjs's
// windowed(), which shaderParts() reads the files with)
const tree = treeOf();
const { window: windowText, linking, list, parts } = shaderParts(tree);
const there = fs.readdirSync(tree.runtime("shaders")).filter((name) => /\.[jt]s$/.test(name)).map((name) => name.slice(0, -3)).sort();
assert.deepEqual([...list].sort(), there, `the modules shaders.js asks for are the files of ${tree.folders.runtime}/shaders/`);
assert.equal(new Set(list).size, list.length, "a module asked for twice");

// ---- the modules: every name used is declared, imported, or a global; each after those it takes from
const byName = new Map(parts.map((part) => [part.name, part]));
let taken = 0;
for (const part of parts) {
  const unknown = [...namesOf(part.text).free].filter((used) => !GLOBALS.has(used));
  assert.deepEqual(unknown, [], `${part.file} uses ${unknown}, which it neither declares nor takes from a neighbour: a ReferenceError as the module loads, or where the line runs`);
  assert.equal(new Set([...part.declared, ...part.given]).size, part.declared.length + part.given.length, `${part.file} exports a name twice`);
  for (const [from, names] of part.takes) {
    assert.ok(byName.has(from), `${part.file} takes names from ${from}.js, which is no module of the window's list`);
    assert.ok(list.indexOf(from) < list.indexOf(part.name), `${part.file} takes from ${from}.js, which the window's list has after it (or the two wait for one another)`);
    taken += names.length;
  }
}
// (a name a module's last line hands out is one a neighbour takes: the list is not a second window)
for (const part of parts) for (const name of part.given) {
  assert.ok(parts.some((other) => other.takes.get(part.name)?.includes(name)), `${part.file} exports ${name} for its neighbours, and none takes it`);
}
console.log(`shaders-modules-check: the ${parts.length} modules of ${tree.folders.runtime}/shaders/ use nothing they do not declare or take from a neighbour, ` +
  `and each comes after those it takes from (${taken} names taken)`);

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
