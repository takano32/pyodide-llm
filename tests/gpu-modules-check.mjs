// tests/gpu-modules-check.mjs (T352): the names that the model's GPU worker (src/runtime/gpu.js) and its modules
// (src/runtime/gpu/*.js) use. Nothing else holds them outside a browser with a GPU: the worker runs only there
// (CI's gpu-check.mjs), and the checks of forward.js's choices make a GPU's worker of their own.
//   - every name a file uses is declared in it, imported by it, or a worker's global (a ReferenceError where the line
//     runs, otherwise: neither the build nor Node says it before);
//   - T378.4: the model on the GPU, whether a stop was asked and why the device was lost are gpu/device.js's own
//     variables (model, stopping, lost): no other module sets them (an import cannot be assigned: a TypeError where
//     the line runs, which neither Node nor the build says before), and no file binds a name of theirs (a parameter or
//     a local would hide the variable from the function's lines);
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

// ---- what device.js holds and the others read (T378.4): model, stopping, lost
const walk = (node, visit, parent = null, key = "") => {
  if (!node || typeof node.type !== "string") return;
  visit(node, parent, key);
  for (const [k, value] of Object.entries(node)) {
    if (k === "loc" || k.endsWith("Comments") || !value || typeof value !== "object") continue;
    (Array.isArray(value) ? value : [value]).forEach((child) => walk(child, visit, node, k));
  }
};
const HELD = ["model", "stopping", "lost"], OWNER = "gpu/device.js";
const lets = programs.get(OWNER).body.map(top).filter((node) => node.type === "VariableDeclaration" && node.kind === "let").flatMap((node) => node.declarations.map((d) => d.id.name));
for (const name of HELD) assert.ok(lets.includes(name), `${OWNER} holds ${name} as a let of its own`);
assert.ok(![...programs.values()].some((program) => program.body.map(top).some((node) => node.type === "VariableDeclaration" && node.declarations.some((d) => d.id.name === "common"))),
  "no module declares common: what the modules read of one another is a name each, set where it is declared (T378.4)");
// a name written: the left of an assignment (a pattern's element too), ++ and --
const written = (node, parent, key) => (parent.type === "AssignmentExpression" && key === "left") || parent.type === "UpdateExpression" ||
  ((parent.type === "ArrayPattern" || (parent.type === "ObjectProperty" && key === "value") || parent.type === "RestElement" || parent.type === "AssignmentPattern") && key !== "right");
// a name bound: declared, a parameter, caught
const bound = (node, parent, key) => (parent.type === "VariableDeclarator" && key === "id") || (/Function/.test(parent.type) && (key === "params" || key === "id")) ||
  (parent.type === "CatchClause" && key === "param") || (parent.type === "ClassDeclaration" && key === "id");
let read = 0;
for (const [name, program] of programs) {
  const topLets = new Set(program.body.map(top).filter((node) => node.type === "VariableDeclaration").flatMap((node) => node.declarations));
  const inPattern = new Set();
  walk(program, (node, parent, key) => {
    // (a pattern's names are bound where the pattern is: a declarator's id, a parameter)
    if ((node.type === "ObjectPattern" || node.type === "ArrayPattern") && (bound(node, parent, key) || inPattern.has(parent) ||
      (parent.type === "ObjectProperty" && inPattern.has(parent)))) inPattern.add(node);
    if (node.type === "ObjectProperty" && inPattern.has(parent)) inPattern.add(node);
    if (node.type !== "Identifier" || !HELD.includes(node.name)) return;
    const member = (parent.type === "MemberExpression" || parent.type === "OptionalMemberExpression") && !parent.computed && key === "property";
    if (member || (parent.type === "ObjectProperty" && key === "key" && !parent.computed && !parent.shorthand) || parent.type === "ImportSpecifier" || parent.type === "ExportSpecifier") return;
    const binds = bound(node, parent, key) || (inPattern.has(parent) && (key === "value" || key === "elements" || parent.shorthand));
    if (name === OWNER && parent.type === "VariableDeclarator" && key === "id" && topLets.has(parent)) return;  // its own let
    assert.ok(!binds, `${name} binds a name ${node.name} of its own (line ${node.loc.start.line}): it hides ${OWNER}'s ${node.name} from the lines after it`);
    if (written(node, parent, key)) {
      assert.equal(name, OWNER, `${name} sets ${node.name} (line ${node.loc.start.line}): ${OWNER} holds it, and an import cannot be assigned (a TypeError where the line runs); say what happened to ${OWNER} instead`);
    } else read += 1;
  });
}
console.log(`gpu-modules-check: model, stopping and lost are gpu/device.js's, set there alone and read ${read} times; no module hides them`);

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
