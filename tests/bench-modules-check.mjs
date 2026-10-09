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
import { parse } from "@babel/parser";
import { namesOf } from "./worker-modules-check.mjs";

const GLOBALS = new Set(("Array ArrayBuffer Atomics BigInt64Array Boolean Error Float32Array Float64Array Infinity Int32Array Int8Array Map Math NaN Number " +
  "Object Set String URL Uint16Array Uint32Array Uint8Array WebAssembly fetch navigator onmessage performance postMessage self undefined").split(" "));
const FOLDER = new URL("../public/benchmark/", import.meta.url);
const read = (name) => fs.readFileSync(new URL(name, FOLDER), "utf8");
const asked = /\[([^\]]*)\]\.map\(\(name\) =>\n\s*\[name, import\(new URL\(`gpu\/\$\{name\}\.js\$\{new URL\(import\.meta\.url\)\.search\}`, import\.meta\.url\)\)\]\)\);/.exec(read("gpu.js"));
assert.ok(asked, "gpu.js asks for its modules in one list, each with its own ?v=");
const modules = [...asked[1].matchAll(/"(\w+)"/g)].map((match) => match[1]);
const there = fs.readdirSync(new URL("gpu/", FOLDER)).filter((name) => name.endsWith(".js")).map((name) => name.slice(0, -3)).sort();
assert.deepEqual([...modules].sort(), there, "the modules gpu.js asks for are the files of public/benchmark/gpu/");

const files = new Map([["gpu.js", read("gpu.js")], ...modules.map((name) => [`gpu/${name}.js`, read(`gpu/${name}.js`)])]);
const programs = new Map([...files].map(([name, text]) => [name, parse(text, { sourceType: "module" }).program]));
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
const takes = new Map();  // a file -> the files it takes names from
for (const [name, program] of programs) {
  const unknown = [...namesOf(files.get(name)).free].filter((used) => !GLOBALS.has(used));
  assert.deepEqual(unknown, [], `${name} uses ${unknown}, which it neither declares nor takes from another file: a ReferenceError where the line runs`);
  takes.set(name, []);
  for (const node of program.body) {
    const d = node.type === "VariableDeclaration" ? node.declarations[0] : null;
    if (d?.init?.type !== "AwaitExpression") continue;
    assert.ok(d.id.type === "ObjectPattern" && node.declarations.length === 1, `${name}: what a file awaits at its top is another's names, taken apart`);
    const from = files.get(name).slice(d.init.start, d.init.end);
    const source = /^await import\(new URL\(`(\w+\.js)\$\{new URL\(import\.meta\.url\)\.search\}`, import\.meta\.url\)\)$/.exec(from)?.[1] ?? /^await modules\.(\w+)$/.exec(from)?.[1];
    assert.ok(source && source.endsWith(".js") === (name !== "gpu.js"), `${name}: \`${from}\` is neither a module of this folder with this file's own ?v= nor one of the window's list`);
    const file = source.endsWith(".js") ? `gpu/${source}` : `gpu/${source}.js`;
    assert.ok(exported.has(file), `${name} takes names from ${file}, which is no module of the worker`);
    takes.get(name).push(file);
    for (const wanted of keysOf(d.id, name)) {
      assert.ok(exported.get(file).has(wanted), `${name} takes ${wanted} from ${file}, which does not export it: undefined`);
      taken += 1;
    }
  }
}
const open = [], ended = new Set();
(function visit(name) {
  assert.ok(!open.includes(name), `the modules wait for one another: ${[...open.slice(open.indexOf(name)), name].join(" → ")}`);
  if (ended.has(name)) return;
  open.push(name);
  takes.get(name).forEach(visit);
  open.pop();
  ended.add(name);
})("gpu.js");
assert.equal(ended.size, files.size, "the window takes from every module, or a module it takes from does");
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
const window_ = await import(new URL("gpu.js", FOLDER));
assert.equal(handlers.length, 2, "gpu.js sets onmessage before its first await, and the steps' own at its end");
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
