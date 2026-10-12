// tests/shaders-modules-check.mjs (T351): what the type check does not hold of the names src/runtime/shaders.ts (the
// window) and its modules (src/runtime/shaders/*.ts) hand one another. Node only, under a second:
//
//   node tests/shaders-modules-check.mjs
//
// The window imports every name of every module and exports it; a module imports a neighbour's. /benchmark/'s GPU
// worker reads the window as one object of names (shared.WGSL.<name>), and so do the tools (wgsl.<name>).
// T368.8: the shaders are TypeScript, and tsc (tests/types.mjs) says what this check said before of a wrong name: a
// name used and declared nowhere, a name exported twice, and a name of the window that /benchmark/'s worker reads and
// the window has not. What is left here is what tsc, the build and Node all let pass, each with why:
//   - a file of shaders/ that the window takes nothing of, and a module that takes from one after it or from itself
//     (tests/shaders-source.mjs's oneSource() is in an order that runs; the build links a ring without a word);
//   - a name a module's last line hands its neighbours that none takes;
//   - a name a module exports where it is declared and the window drops: the tools read the window as JavaScript, where
//     it is undefined without a word (tests/unchanged.mjs's shaders, tests/gpu-check.mjs, tests/smoke.mjs), and tsc
//     says only a name that /benchmark/'s worker reads; and anything the window declares of its own (it takes and
//     exports, and nothing else);
//   - that the window links in Node with every export defined, and that what it exports is each module's own value.
// The parser is @babel/parser, which Astro's packages bring.
import assert from "node:assert/strict";
import fs from "node:fs";
import { shaderParts } from "./shaders-source.mjs";
import { treeOf } from "./tree.mjs";

// (where the runtime is: tests/tree.mjs. That the window reaches every file of the folder is tests/imports.mjs's
// windowed(), which shaderParts() reads the files with)
const tree = treeOf();
const { window: windowText, linking, list, parts } = shaderParts(tree);
const there = fs.readdirSync(tree.runtime("shaders")).filter((name) => /\.[jt]s$/.test(name)).map((name) => name.slice(0, -3)).sort();
assert.deepEqual([...list].sort(), there, `the modules shaders.js asks for are the files of ${tree.folders.runtime}/shaders/`);
assert.equal(new Set(list).size, list.length, "a module asked for twice");

// ---- the modules: each after those it takes from
const byName = new Map(parts.map((part) => [part.name, part]));
let taken = 0;
for (const part of parts) {
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
console.log(`shaders-modules-check: each of the ${parts.length} modules of ${tree.folders.runtime}/shaders/ comes after those it takes from (${taken} names taken), and what one hands its neighbours is taken`);

// ---- the window: what it takes of each module, and what it exports
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
