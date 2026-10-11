// tests/gpu-modules-check.mjs (T352): the names that the model's GPU worker (public/gpu.js) and its modules
// (public/gpu/*.js) hand one another. Nothing else holds them outside a browser with a GPU: the worker runs only there
// (CI's gpu-check.mjs), and the checks of forward.js's choices make a GPU's worker of their own.
//   - a name a file takes out of another (`const { a, b } = await import(...)`, `= await modules.x`) is exported there,
//     under its own name, and every name a file uses is declared in it, taken from another, or a worker's global (a
//     ReferenceError where the line runs, otherwise);
//   - every common.<name> is a field of the one object the modules share (gpu/device.js's common: a field is no
//     variable, and a wrong name reads undefined and sets nothing another module reads), every field is used, and no
//     file binds a name `common` of its own (a parameter of that name would hide the object from the function's lines:
//     open() has one named shared, which is why the object is not);
//   - no module exports a `let` (what another takes of it through an awaited import is a copy of its value then);
//   - the folder holds the modules the window asks for, each with the window's own ?v=, and what they take from one
//     another goes round nowhere (a ring of awaited imports never ends); the shaders are asked for under one URL;
//   - the window links in Node with every export defined, and messages that come at its first await (while the modules
//     load) reach the receiver once each, in the order they came (T109: a module worker loses what comes before its
//     onmessage is set; a "stop" played before the "start" it followed would end the worker twice).
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
// (T367.1: where the runtime is and which form its imports have is the tree's, tests/tree.mjs; the lines that take a
// neighbour's names are read by tests/imports.mjs, whose windowed() holds the folder to the window's list (each module
// with the window's own ?v=) or, where a bundler links the files, to what the window reaches, and says a ring)
const tree = treeOf();
const at = (name) => `${name}`;  // (a file of this worker, by its name under the runtime's folder)
const linked = windowed(at("gpu.js"), { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), bundled: tree.bundled });
const files = new Map([...linked.files].map(([name, { text }]) => [name.slice(at("").length), text]));
const programs = new Map([...linked.files].map(([name, { program }]) => [name.slice(at("").length), program]));
const top = (node) => (node.type === "ExportNamedDeclaration" && node.declaration ? node.declaration : node);
const exported = new Map([...programs].map(([name, program]) => [name, new Set(program.body.flatMap((node) => {
  if (node.type !== "ExportNamedDeclaration") return [];
  if (!node.declaration) return node.specifiers.map((specifier) => (assert.equal(specifier.local.name, specifier.exported.name, `${name} exports ${specifier.local.name} under another name`), specifier.exported.name));
  return node.declaration.id ? [node.declaration.id.name] : node.declaration.declarations.map((d) => (assert.equal(d.id.type, "Identifier"), d.id.name));
}))]));
// (a `let` exported: the importer's name is a copy made when it was taken)
for (const [name, program] of programs) {
  const lets = new Set(program.body.map(top).filter((node) => node.type === "VariableDeclaration" && node.kind === "let").flatMap((node) => node.declarations.map((d) => d.id.name)));
  assert.deepEqual([...exported.get(name)].filter((exportedName) => lets.has(exportedName)), [], `${name} exports a let: another module's copy of it does not follow it`);
}

// ---- every name used is declared, taken from another file, or a global; and what is taken is exported there
const keysOf = (pattern, what) => pattern.properties.map((property) => {
  assert.ok(property.type === "ObjectProperty" && !property.computed && property.key.type === "Identifier" && property.shorthand, `${what}: a plain name, taken under its own`);
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
// the shaders: gpu/device.js's import of them, and the window's at once, which must be the same module (one URL)
// (where a bundler links the files there is one module by its path, and nothing to hold)
const SHADERS = (to) => `import(new URL(\`${to}shaders.js\${new URL(import.meta.url).search}\`, import.meta.url))`;
if (!tree.bundled) {
  assert.ok(files.get("gpu/device.js").includes(`const shaders = ${SHADERS("../")};`), "gpu/device.js imports ../shaders.js with its own ?v=");
  assert.ok(files.get("gpu.js").includes(`\n${SHADERS("")}.catch(() => {});\n`), "gpu.js asks for shaders.js at once, with its own ?v= (a failure is device.js's to say)");
  assert.equal([...files.values()].join("\n").split("shaders.js${").length - 1, 2, "the shaders are imported in those two places");
}
console.log(`gpu-modules-check: the GPU worker's ${programs.size} files use nothing they do not declare or take from another, the ${taken} names they take are exported, and none waits for itself`);

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
    const takenByName = parent.type === "ObjectProperty" && parent.shorthand;
    const handedOn = parent.type === "ExportSpecifier";
    assert.ok(itsOwn || takenByName || handedOn, `${name} uses the name common otherwise than as common.<field> (line ${node.loc.start.line}): a local of that name hides the object, and the object handed whole is not held here`);
  });
}
assert.deepEqual(fields.filter((field) => !used.has(field)), [], "fields of common that nothing reads or sets");
console.log(`gpu-modules-check: the ${fields.length} fields of common are the ones the modules use (${said} times)`);

// ---- the window in Node: its exports, and messages that come while the modules load
const handlers = [], posted = [];
let closed = 0;
Object.defineProperty(globalThis, "onmessage", { configurable: true, get: () => handlers.at(-1), set(handler) {
  handlers.push(handler);
  // (the first one is set before the window's first await: two messages at that moment, which a worker's port may
  // deliver: a start (no WebGPU in Node: "unusable", then "ended"), and the stop that follows it, which ends nothing more
  // while the start is still under way; played the other way round, the stop ends the worker before the start does)
  if (handlers.length === 1) {
    handler({ data: { type: "start", memory: null, plan: { force: {}, matrices: {} } } });
    handler({ data: { type: "stop" } });
  }
} });
globalThis.self = globalThis;
globalThis.close = () => { closed += 1; };
globalThis.postMessage = (message) => posted.push(message);
assert.equal(globalThis.navigator?.gpu, undefined, "this check wants a Node without WebGPU");
const window_ = await import(tree.runtimeUrl(at("gpu.js")));
// (where a bundler links the files the window awaits no module: the receiver is the one onmessage, set before anything can come)
if (tree.bundled) assert.equal(handlers.length, 1, "gpu.js sets its receiver once: there is no wait for its modules to keep messages through");
else assert.equal(handlers.length, 2, "gpu.js sets onmessage before its first await, and the receiver at its end");
for (const [name, value] of Object.entries(window_)) assert.notEqual(value, undefined, `gpu.js exports ${name} undefined`);
for (const name of [...exported.get("gpu.js")]) assert.ok(name in window_, `gpu.js exports ${name}`);
for (let i = 0; i < 100 && !closed; i++) await new Promise((resolve) => setTimeout(resolve, 10));
await new Promise((resolve) => setTimeout(resolve, 50));
assert.deepEqual(posted, [{ type: "unusable", reason: "no WebGPU in a worker here" }, { type: "ended" }],
  "the start and the stop that came at the window's first await were played to the receiver once each, in their order");
assert.equal(closed, 1, "the worker ended once");
console.log(`gpu-modules-check: gpu.js links (${Object.keys(window_).length} export), and what comes at its first await is played once, in order`);
console.log("gpu-modules-check: ok");
process.exit(0);
