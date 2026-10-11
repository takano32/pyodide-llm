// tests/bench-modules-check.mjs (T353): the names that /benchmark/'s GPU worker (public/benchmark/gpu.js) and its modules
// (public/benchmark/gpu/*.js) hand one another, and the names the modules of src/bench/ use. None of these is held by
// anything else outside a browser with a GPU: the worker's steps run only there (CI's bench-check.mjs and bench-dawn.mjs).
//   - a name a file takes out of another (`const { a, b } = await import(...)`, `= await modules.x`) is exported there,
//     and every name a file uses is declared in it, taken from another, or a worker's global (a ReferenceError where
//     the line runs, otherwise);
//   - every shared.<name> is a field of the one object the steps share (gpu/device.js's shared: a field is no variable,
//     and a wrong name reads undefined and sets nothing a step reads), and every field is used;
//   - the folder holds the modules the window asks for, and what they take from one another goes round nowhere (a ring
//     of awaited imports never ends);
//   - the window links in Node with every export defined, and a message that comes at its first await (while the modules
//     load) reaches the steps' onmessage once (T109: a module worker loses what comes before its onmessage is set).
// And every module of src/bench/ uses only what it declares or imports (the tables are functions: a name that is
// nowhere is a ReferenceError only when that table is written).
// Node only, under a second:
//
//   node tests/bench-modules-check.mjs
//
// The parser is @babel/parser, which Astro's packages bring (as tests/worker-modules-check.mjs, whose namesOf this uses).
import assert from "node:assert/strict";
import fs from "node:fs";
import { windowed } from "./imports.mjs";
import { treeOf } from "./tree.mjs";
import { namesOf } from "./worker-modules-check.mjs";

const GLOBALS = new Set(("Array ArrayBuffer Atomics BigInt64Array Boolean Error Float32Array Float64Array Infinity Int32Array Int8Array Map Math NaN Number " +
  "Object Set String URL Uint16Array Uint32Array Uint8Array WebAssembly fetch navigator onmessage performance postMessage self undefined").split(" "));
// (T367.1: where the runtime is and which form its imports have is the tree's, tests/tree.mjs; the lines that take a
// neighbour's names are read by tests/imports.mjs, whose windowed() holds the folder to the window's list (each module
// with the window's own ?v=) or, where a bundler links the files, to what the window reaches, and says a ring)
const tree = treeOf();
const at = (name) => `benchmark/${name}`;  // (a file of this worker, by its name under the runtime's folder)
const linked = windowed(at("gpu.js"), { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), bundled: tree.bundled });
const files = new Map([...linked.files].map(([name, { text }]) => [name.slice(at("").length), text]));
const programs = new Map([...linked.files].map(([name, { program }]) => [name.slice(at("").length), program]));
const exported = new Map([...programs].map(([name, program]) => [name, new Set(program.body.flatMap((node) => {
  if (node.type !== "ExportNamedDeclaration") return [];
  if (!node.declaration) return node.specifiers.map((specifier) => specifier.exported.name);
  return node.declaration.id ? [node.declaration.id.name] : node.declaration.declarations.map((d) => (assert.equal(d.id.type, "Identifier"), d.id.name));
}))]));

// ---- every name used is declared, taken from another file, or a global; and what is taken is exported there
const keysOf = (pattern, what) => pattern.properties.map((property) => {
  assert.ok(property.type === "ObjectProperty" && !property.computed && property.key.type === "Identifier" && property.shorthand, `${what}: a plain name`);
  return property.key.name;
});
let taken = 0;
for (const [name, program] of programs) {
  const unknown = [...namesOf(files.get(name)).free].filter((used) => !GLOBALS.has(used));
  assert.deepEqual(unknown, [], `${name} uses ${unknown}, which it neither declares nor takes from another file: a ReferenceError where the line runs`);
  const { takes: taking, left } = linked.files.get(at(name));
  for (const node of program.body) {
    const d = node.type === "VariableDeclaration" ? node.declarations[0] : null;
    if (d?.init?.type !== "AwaitExpression") continue;
    assert.ok(left.includes(node), `${name}: \`${files.get(name).slice(d.init.start, d.init.end)}\`: what a file awaits at its top is another's names, taken apart`);
  }
  for (const { from, names, how } of taking) {
    const file = from.slice(at("").length);
    // (a module takes of a module of the folder; the window, where it has a list, of the list alone)
    assert.ok(from.startsWith(at("gpu/")) && exported.has(file), `${name} takes names from ${from}, which is no module of the worker`);
    assert.ok(tree.bundled || (how === "await modules") === (name === "gpu.js"), `${name}: a module is taken from with this file's own ?v=, and by the window from its list`);
    for (const { imported: wanted, local } of names) {
      assert.equal(wanted, local, `${name}: a plain name, taken under its own`);
      assert.ok(exported.get(file).has(wanted), `${name} takes ${wanted} from ${file}, which does not export it: undefined`);
      taken += 1;
    }
  }
}
// (that none waits for itself, and that the window takes from every module or a module it takes from does: windowed())
console.log(`bench-modules-check: the GPU worker's ${programs.size} files use nothing they do not declare or take from another, the ${taken} names they take are exported, and none waits for itself`);

// ---- the one object the steps share
const holders = [...programs].filter(([, program]) => program.body.some((node) => (node.declaration ?? node).type === "VariableDeclaration" && (node.declaration ?? node).declarations[0].id.name === "shared"));
assert.deepEqual(holders.map(([name]) => name), ["gpu/device.js"], "gpu/device.js declares shared, and nothing else does");
const declaration = holders[0][1].body.map((node) => node.declaration ?? node).find((node) => node.type === "VariableDeclaration" && node.declarations[0].id.name === "shared").declarations[0];
assert.equal(declaration.init.type, "ObjectExpression");
const fields = declaration.init.properties.map((property) => (assert.ok(property.type === "ObjectProperty" && !property.computed), property.key.name));
assert.equal(new Set(fields).size, fields.length, "a field of shared twice");
const used = new Set();
let said = 0;
const walk = (node, visit) => {
  if (!node || typeof node.type !== "string") return;
  visit(node);
  for (const [key, value] of Object.entries(node)) {
    if (key === "loc" || key.endsWith("Comments") || !value || typeof value !== "object") continue;
    (Array.isArray(value) ? value : [value]).forEach((child) => walk(child, visit));
  }
};
for (const [name, program] of programs) {
  walk(program, (node) => {
    if ((node.type !== "MemberExpression" && node.type !== "OptionalMemberExpression") || node.object.type !== "Identifier" || node.object.name !== "shared") return;
    assert.ok(!node.computed, `${name} reads shared[…]: this check knows shared.<name>`);
    assert.ok(fields.includes(node.property.name), `${name} says shared.${node.property.name}, which is no field of gpu/device.js's shared: undefined`);
    used.add(node.property.name);
    said += 1;
  });
}
assert.deepEqual(fields.filter((field) => !used.has(field)), [], "fields of shared that nothing reads or sets");
console.log(`bench-modules-check: the ${fields.length} fields of shared are the ones the steps use (${said} times)`);

// ---- the window in Node: its exports, and a message that comes while the modules load
const handlers = [], posted = [];
Object.defineProperty(globalThis, "onmessage", { configurable: true, get: () => handlers.at(-1), set(handler) {
  handlers.push(handler);
  // (the first one is set before the window's first await: a message at that moment, which a worker's port may deliver)
  if (handlers.length === 1) handler({ data: { step: "no such step" } });
} });
globalThis.postMessage = (message) => posted.push(message);
const window_ = await import(tree.runtimeUrl(at("gpu.js")));
// (where a bundler links the files the window awaits no module: the steps' onmessage is the one, set before anything can come)
if (tree.bundled) assert.equal(handlers.length, 1, "gpu.js sets the steps' onmessage once: there is no wait for its modules to keep a message through");
else assert.equal(handlers.length, 2, "gpu.js sets onmessage before its first await, and the steps' own at its end");
for (const [name, value] of Object.entries(window_)) assert.notEqual(value, undefined, `gpu.js exports ${name} undefined`);
for (const name of [...exported.get("gpu.js")]) assert.ok(name in window_, `gpu.js exports ${name}`);
for (let i = 0; i < 100 && !posted.length; i++) await new Promise((resolve) => setTimeout(resolve, 10));
assert.deepEqual(posted, [{ step: "no such step", result: undefined }], "the message that came at the window's first await was answered by the steps' onmessage, once");
console.log(`bench-modules-check: gpu.js links (${Object.keys(window_).length} exports), and a message at its first await is answered once`);

// ---- src/bench/: the tables' modules use what they declare or import
const TABLES = new URL("../src/bench/", import.meta.url);
const TABLE_GLOBALS = new Set("Boolean Map Math NaN Number Object Set String URLSearchParams encodeURIComponent globalThis undefined".split(" "));
const tables = fs.readdirSync(TABLES).filter((name) => name.endsWith(".js"));
for (const name of tables) {
  const unknown = [...namesOf(fs.readFileSync(new URL(name, TABLES), "utf8")).free].filter((used_) => !TABLE_GLOBALS.has(used_));
  assert.deepEqual(unknown, [], `src/bench/${name} uses ${unknown}, which it neither declares nor imports: a ReferenceError when that table is written`);
}
console.log(`bench-modules-check: the ${tables.length} modules of src/bench/ use nothing they do not declare or import`);
console.log("bench-modules-check: ok");
process.exit(0);
