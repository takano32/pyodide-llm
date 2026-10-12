// tests/bench-modules-check.mjs (T353): the names that /benchmark/'s GPU worker (src/runtime/benchmark/gpu.ts) and its modules
// (src/runtime/benchmark/gpu/*.ts; .js until T368.7) hand one another, and the names the modules of src/bench/ use. None of these is held by
// anything else outside a browser with a GPU: the worker's steps run only there (CI's bench-check.mjs and bench-dawn.mjs).
//   - every name a file uses is declared in it, imported by it, or a worker's global (a ReferenceError where the
//     line runs, otherwise: neither the build nor Node says it before);
//   - every shared.<name> is a field of the one object the steps share (gpu/device.js's shared: a field is no variable,
//     and a wrong name reads undefined and sets nothing a step reads), and every field is used;
//   - what the modules take from one another goes round nowhere (tests/imports.mjs's windowed(): the build links a ring
//     without a word), and every file of the folder is reached from the window;
//   - the window links in Node with every export defined, and a message given to the steps' onmessage as it is set is
//     answered once;
//   - (T368.7) every message a section's worker posts to the page says which type of contracts/benchmark.ts it is
//     (`satisfies`): a worker's own postMessage() takes anything, and tsc holds only a message that says so.
// T367.5: what this held before and the build and Node hold by themselves now is gone from here: that a name taken is
// exported where it is taken from (the build: MISSING_EXPORT, an error; Node: a SyntaxError as the module is imported).
// And every module of src/bench/ uses only what it declares or imports (the tables are functions: a name that is
// nowhere is a ReferenceError only when that table is written).
// Node only, under a second:
//
//   node tests/bench-modules-check.mjs
//
// The parser is @babel/parser, which Astro's packages bring (as tests/worker-modules-check.mjs, whose namesOf this uses).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { parse } from "@babel/parser";
import { contract } from "./contracts.mjs";
import { windowed } from "./imports.mjs";
import { treeOf } from "./tree.mjs";
import { typed, untyped } from "./untyped.mjs";
import { namesOf } from "./worker-modules-check.mjs";

const GLOBALS = new Set(("AbortController Array ArrayBuffer Atomics BigInt64Array Boolean Error Float32Array Float64Array Infinity Int32Array Int8Array Map Math NaN Number " +
  "Object Set String URL Uint16Array Uint32Array Uint8Array WebAssembly clearTimeout fetch globalThis navigator onmessage performance postMessage self setTimeout undefined").split(" "));
// (where the runtime is: tests/tree.mjs. tests/imports.mjs's windowed() reads the window and every file it reaches,
// says a ring, and holds every file of the folder to be reached)
const tree = treeOf();
const at = (name) => `benchmark/${name}`;  // (a file of this worker, by its name under the runtime's folder)
// (T368.7: the window by the name the tree has for it, gpu.ts: windowed() blanks a file's types by the name it is given,
// and the modules are then named by their own endings)
const WINDOW = path.basename(tree.runtime(at("gpu.js"))), END = path.extname(WINDOW);
const linked = windowed(at(WINDOW), { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), bundled: tree.bundled, outside: true });
// (a file by its name under the worker's folder; one outside it that a module takes names of, T367.3's table of the
// built files, by its way from there: "../built.js")
const short = (name) => (name.startsWith(at("")) ? name.slice(at("").length) : `../${name}`);
const full = new Map([...linked.files.keys()].map((name) => [short(name), name]));
const files = new Map([...linked.files].map(([name, { text }]) => [short(name), text]));
const programs = new Map([...linked.files].map(([name, { program }]) => [short(name), program]));

// ---- every name used is declared, imported, or a global
let names = 0;
for (const [name] of programs) {
  const { free } = namesOf(files.get(name)), unknown = [...free].filter((used) => !GLOBALS.has(used));
  assert.deepEqual(unknown, [], `${name} uses ${unknown}, which it neither declares nor imports: a ReferenceError where the line runs`);
  names += free.size;
  for (const { from } of linked.files.get(full.get(name)).takes) {
    assert.ok(from.startsWith(at("gpu/")) || linked.others.includes(from), `${name} takes names from ${from}, which is no module of the worker`);
  }
}
// (that none waits for itself, and that the window takes from every module or a module it takes from does: windowed())
console.log(`bench-modules-check: the GPU worker's ${programs.size} files use nothing they do not declare or import (${names} names of a worker's globals), and none waits for itself`);

// ---- the one object the steps share
const holders = [...programs].filter(([, program]) => program.body.some((node) => (node.declaration ?? node).type === "VariableDeclaration" && (node.declaration ?? node).declarations[0].id.name === "shared"));
assert.deepEqual(holders.map(([name]) => name), [`gpu/device${END}`], `gpu/device${END} declares shared, and nothing else does`);
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
    assert.ok(fields.includes(node.property.name), `${name} says shared.${node.property.name}, which is no field of gpu/device${END}'s shared: undefined`);
    used.add(node.property.name);
    said += 1;
  });
}
assert.deepEqual(fields.filter((field) => !used.has(field)), [], "fields of shared that nothing reads or sets");
console.log(`bench-modules-check: the ${fields.length} fields of shared are the ones the steps use (${said} times)`);

// ---- T368.7: the workers are TypeScript, and tsc holds a message one posts to the page to the type the message says
// it is: postMessage({ … } satisfies Alive). Every message of the section's workers says one of contracts/benchmark.ts
// (the sources as they are written: with the types blanked, as above, nothing of it is left). reader.ts is no
// section's worker: it answers the CPU section's worker, with one word.
if (END === ".ts") {
  const types = new Set(contract("benchmark.ts").names);
  const posting = [...[...linked.files.keys()].filter((name) => name.startsWith(at(""))), ...["sections", "storage", "memory", "waiter"].map((name) => at(`${name}${END}`))];
  let posts = 0;
  for (const name of posting) {
    walk(parse(fs.readFileSync(tree.runtime(name), "utf8"), { sourceType: "module", plugins: ["typescript"] }).program, (node) => {
      if (node.type !== "CallExpression" || node.callee.type !== "Identifier" || node.callee.name !== "postMessage") return;
      const [message] = node.arguments;
      assert.ok(message?.type === "TSSatisfiesExpression" && types.has(message.typeAnnotation.typeName?.name),
        `${name}, line ${node.loc.start.line}: a message is posted without \`satisfies\` a type of contracts/benchmark.ts: nothing holds its keys and values to what the page reads`);
      posts += 1;
    });
  }
  assert.ok(posts >= 30, `${posts} messages posted were found in ${posting.length} files: fewer than the workers post`);
  console.log(`bench-modules-check: the ${posts} messages the sections' workers post (${posting.length} files) each satisfy a type of contracts/benchmark.ts`);
}

// ---- the window in Node: it links, and what its onmessage is given
const handlers = [], posted = [];
Object.defineProperty(globalThis, "onmessage", { configurable: true, get: () => handlers.at(-1), set(handler) {
  handlers.push(handler);
  // (a message as the steps' onmessage is set)
  if (handlers.length === 1) handler({ data: { step: "no such step" } });
} });
globalThis.postMessage = (message) => posted.push(message);
const window_ = await import(tree.runtimeUrl(at("gpu.js")));
assert.equal(handlers.length, 1, `${WINDOW} sets the steps' onmessage once, as it loads: nothing is awaited before it`);
for (const [name, value] of Object.entries(window_)) assert.notEqual(value, undefined, `${WINDOW} exports ${name} undefined`);
for (let i = 0; i < 100 && !posted.length; i++) await new Promise((resolve) => setTimeout(resolve, 10));
assert.deepEqual(posted, [{ step: "no such step", result: undefined }], "the message given as the steps' onmessage was set was answered by it, once");
console.log(`bench-modules-check: ${WINDOW} links (${Object.keys(window_).length} exports), and a message given as its onmessage is set is answered once`);

// ---- src/bench/: the tables' modules use what they declare or import
const TABLES = new URL("../src/bench/", import.meta.url);
const TABLE_GLOBALS = new Set("Boolean Map Math NaN Number Object Set String URLSearchParams encodeURIComponent globalThis undefined".split(" "));
// (T368.7: either ending, a .ts file with its types blanked: a module made TypeScript is still read here)
const tables = fs.readdirSync(TABLES).filter((name) => name.endsWith(".js") || typed(name));
for (const name of tables) {
  const unknown = [...namesOf(untyped(fs.readFileSync(new URL(name, TABLES), "utf8"), name)).free].filter((used_) => !TABLE_GLOBALS.has(used_));
  assert.deepEqual(unknown, [], `src/bench/${name} uses ${unknown}, which it neither declares nor imports: a ReferenceError when that table is written`);
}
console.log(`bench-modules-check: the ${tables.length} modules of src/bench/ use nothing they do not declare or import`);
console.log("bench-modules-check: ok");
process.exit(0);
