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
import { parse } from "@babel/parser";
import { namesOf } from "./worker-modules-check.mjs";

const GLOBALS = new Set(("Array ArrayBuffer Atomics BigInt BigInt64Array Boolean Error Float32Array Float64Array Infinity Int32Array Int8Array Map Math " +
  "NaN Number Object Promise Set String URL Uint16Array Uint32Array Uint8Array clearInterval clearTimeout navigator onmessage performance postMessage " +
  "self setInterval setTimeout undefined").split(" "));
const FOLDER = new URL("../public/", import.meta.url);
const read = (name) => fs.readFileSync(new URL(name, FOLDER), "utf8");
const asked = /\[([^\]]*)\]\.map\(\(name\) =>\n\s*\[name, import\(new URL\(`gpu\/\$\{name\}\.js\$\{new URL\(import\.meta\.url\)\.search\}`, import\.meta\.url\)\)\]\)\);/.exec(read("gpu.js"));
assert.ok(asked, "gpu.js asks for its modules in one list, each with its own ?v=");
const modules = [...asked[1].matchAll(/"(\w+)"/g)].map((match) => match[1]);
const there = fs.readdirSync(new URL("gpu/", FOLDER)).filter((name) => name.endsWith(".js")).map((name) => name.slice(0, -3)).sort();
assert.deepEqual([...modules].sort(), there, "the modules gpu.js asks for are the files of public/gpu/");

const files = new Map([["gpu.js", read("gpu.js")], ...modules.map((name) => [`gpu/${name}.js`, read(`gpu/${name}.js`)])]);
const programs = new Map([...files].map(([name, text]) => [name, parse(text, { sourceType: "module" }).program]));
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
// the shaders: gpu/device.js's import of them, and the window's at once, which must be the same module (one URL)
const SHADERS = (to) => `import(new URL(\`${to}shaders.js\${new URL(import.meta.url).search}\`, import.meta.url))`;
assert.ok(files.get("gpu/device.js").includes(`const shaders = ${SHADERS("../")};`), "gpu/device.js imports ../shaders.js with its own ?v=");
assert.ok(files.get("gpu.js").includes(`\n${SHADERS("")};\n`), "gpu.js asks for shaders.js at once, with its own ?v=");
assert.equal([...files.values()].join("\n").split("shaders.js${").length - 1, 2, "the shaders are imported in those two places");
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
const window_ = await import(new URL("gpu.js", FOLDER));
assert.equal(handlers.length, 2, "gpu.js sets onmessage before its first await, and the receiver at its end");
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
