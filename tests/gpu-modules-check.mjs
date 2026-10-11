// tests/gpu-modules-check.mjs (T352): the names that the model's GPU worker (src/runtime/gpu.js) and its modules
// (src/runtime/gpu/*.js) use. Nothing else holds them outside a browser with a GPU: the worker runs only there
// (CI's gpu-check.mjs), and the checks of forward.js's choices make a GPU's worker of their own.
//   - every name a file uses is declared in it, imported by it, or a worker's global (a ReferenceError where the line
//     runs, otherwise: neither the build nor Node says it before);
//   - every common.<name> is a field of the one object the modules share (gpu/device.js's common: a field is no
//     variable, and a wrong name reads undefined and sets nothing another module reads), every field is used, and no
//     file binds a name `common` of its own (a parameter of that name would hide the object from the function's lines:
//     open() has one named shared, which is why the object is not);
//   - what the modules take from one another goes round nowhere (tests/imports.mjs's windowed(): the build links a
//     ring without a word, and Node only where a module uses the other's name as it loads), and every file of gpu/
//     is reached from the window;
//   - the window links in Node, and a start and the stop after it, given to its receiver as it is set, are answered
//     once each, in their order.
// T367.5: what this held before and the build and Node hold by themselves now is gone from here: that a name taken is
// exported where it is taken from (the build: MISSING_EXPORT, an error; Node: a SyntaxError as the window is imported
// below), and that no module exports a `let` (a static import is the live binding, no copy). What a module takes of
// the shaders is tests/device-key-check.mjs's (the import lines are the shaders the key must hold).
// Node only, under a second:
//
//   node tests/gpu-modules-check.mjs
//
// The parser is @babel/parser, which Astro's packages bring (as tests/worker-modules-check.mjs, whose namesOf this uses).
import assert from "node:assert/strict";
import fs from "node:fs";
import { windowed } from "./imports.mjs";
import { treeOf } from "./tree.mjs";
import { namesOf } from "./worker-modules-check.mjs";

const GLOBALS = new Set(("Array ArrayBuffer Atomics BigInt BigInt64Array Boolean Error Float32Array Float64Array Infinity Int32Array Int8Array Map Math " +
  "NaN Number Object Promise Set String URL Uint16Array Uint32Array Uint8Array clearInterval clearTimeout navigator onmessage performance postMessage " +
  "self setInterval setTimeout undefined").split(" "));
// (where the runtime is: tests/tree.mjs. tests/imports.mjs's windowed() reads the window and every file it reaches,
// says a ring, and holds every file of gpu/ to be reached; the shaders' parts a module takes names of are read with them)
const tree = treeOf();
const linked = windowed("gpu.js", { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), bundled: tree.bundled, outside: true });
const own = (name) => name === "gpu.js" || name.startsWith("gpu/");
const files = new Map([...linked.files].filter(([name]) => own(name)).map(([name, { text }]) => [name, text]));
const programs = new Map([...linked.files].filter(([name]) => own(name)).map(([name, { program }]) => [name, program]));
const top = (node) => (node.type === "ExportNamedDeclaration" && node.declaration ? node.declaration : node);

// ---- every name used is declared, imported, or a global
let names = 0;
for (const [name] of programs) {
  const { free } = namesOf(files.get(name)), unknown = [...free].filter((used) => !GLOBALS.has(used));
  assert.deepEqual(unknown, [], `${name} uses ${unknown}, which it neither declares nor imports: a ReferenceError where the line runs`);
  names += free.size;
  for (const { from } of linked.files.get(name).takes) {
    assert.ok(own(from) || /^shaders\/\w+\.js$/.test(from), `${name} takes names from ${from}: a module of the worker takes of the worker's modules and of the shaders' parts (shaders/<part>.js)`);
  }
}
console.log(`gpu-modules-check: the GPU worker's ${programs.size} files use nothing they do not declare or import (${names} names of a worker's globals), and none waits for itself`);

// ---- the one object the modules share
const holders = [...programs].filter(([, program]) => program.body.map(top).some((node) => node.type === "VariableDeclaration" && node.declarations.some((d) => d.id.name === "common")));
assert.deepEqual(holders.map(([name]) => name), ["gpu/device.js"], "gpu/device.js declares common, and nothing else does");
const declaration = holders[0][1].body.map(top).find((node) => node.type === "VariableDeclaration" && node.declarations[0].id.name === "common").declarations[0];
assert.equal(declaration.init.type, "ObjectExpression");
const fields = declaration.init.properties.map((property) => (assert.ok(property.type === "ObjectProperty" && !property.computed), property.key.name));
assert.equal(new Set(fields).size, fields.length, "a field of common twice");
const used = new Set();
let said = 0;
const walk = (node, visit, parent = null, key = "") => {
  if (!node || typeof node.type !== "string") return;
  visit(node, parent, key);
  for (const [k, value] of Object.entries(node)) {
    if (k === "loc" || k.endsWith("Comments") || !value || typeof value !== "object") continue;
    (Array.isArray(value) ? value : [value]).forEach((child) => walk(child, visit, node, k));
  }
};
for (const [name, program] of programs) {
  walk(program, (node, parent, key) => {
    if (node.type !== "Identifier" || node.name !== "common") return;
    const member = (parent.type === "MemberExpression" || parent.type === "OptionalMemberExpression") && !parent.computed;
    if (member && key === "property") return;  // (something's .common: no name)
    if (member && key === "object") {
      assert.ok(fields.includes(parent.property.name), `${name} says common.${parent.property.name}, which is no field of gpu/device.js's common: undefined`);
      used.add(parent.property.name);
      said += 1;
      return;
    }
    // the object's own declaration, a module taking it by its name, a module handing it on
    const itsOwn = parent === declaration && key === "id";
    const takenByName = parent.type === "ImportSpecifier";
    const handedOn = parent.type === "ExportSpecifier";
    assert.ok(itsOwn || takenByName || handedOn, `${name} uses the name common otherwise than as common.<field> (line ${node.loc.start.line}): a local of that name hides the object, and the object handed whole is not held here`);
  });
}
assert.deepEqual(fields.filter((field) => !used.has(field)), [], "fields of common that nothing reads or sets");
console.log(`gpu-modules-check: the ${fields.length} fields of common are the ones the modules use (${said} times)`);

// ---- the window in Node: it links, and what its receiver is given
const handlers = [], posted = [];
let closed = 0;
Object.defineProperty(globalThis, "onmessage", { configurable: true, get: () => handlers.at(-1), set(handler) {
  handlers.push(handler);
  // (two messages as the receiver is set: a start (no WebGPU in Node: "unusable", then "ended"), and the stop that
  // follows it, which ends nothing more while the start is still under way; played the other way round, the stop ends
  // the worker before the start does)
  if (handlers.length === 1) {
    handler({ data: { type: "start", memory: null, plan: { force: {}, matrices: {} } } });
    handler({ data: { type: "stop" } });
  }
} });
globalThis.self = globalThis;
globalThis.close = () => { closed += 1; };
globalThis.postMessage = (message) => posted.push(message);
assert.equal(globalThis.navigator?.gpu, undefined, "this check wants a Node without WebGPU");
const window_ = await import(tree.runtimeUrl("gpu.js"));
assert.equal(handlers.length, 1, "gpu.js sets its receiver once, as it loads: nothing is awaited before it");
for (const [name, value] of Object.entries(window_)) assert.notEqual(value, undefined, `gpu.js exports ${name} undefined`);
for (let i = 0; i < 100 && !closed; i++) await new Promise((resolve) => setTimeout(resolve, 10));
await new Promise((resolve) => setTimeout(resolve, 50));
assert.deepEqual(posted, [{ type: "unusable", reason: "no WebGPU in a worker here" }, { type: "ended" }],
  "the start and the stop that came as the receiver was set were answered once each, in their order");
assert.equal(closed, 1, "the worker ended once");
console.log(`gpu-modules-check: gpu.js links (${Object.keys(window_).length} export), and a start and its stop are answered once each, in order`);
console.log("gpu-modules-check: ok");
process.exit(0);
