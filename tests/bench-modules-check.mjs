// tests/bench-modules-check.mjs (T.353): what the type check does not hold of /benchmark/'s GPU worker
// (src/runtime/benchmark/gpu.ts) and its modules (src/runtime/benchmark/gpu/*.ts), and the names the modules of
// src/bench/ use. Nothing else reads the worker's lines outside a browser with a GPU: its steps run only there (CI's
// bench-check.mjs and bench-dawn.mjs). Node only, under a second:
//
//   node tests/bench-modules-check.mjs
//
// T.368.8: the worker is TypeScript, and tsc (tests/types.mjs) says what this check said before of a wrong name: a name
// used and declared nowhere. (A shared.<name> that is misspelt is tsc's too, and still said below: the same line holds
// a field that the type has as optional and the object is made without.)
// What is left here is what tsc, the build and Node all let pass, each with why:
//   - a shared.<name> that gpu/device.ts's object is made without (optional in its type, it is undefined without a
//     word), and a field of it that no step reads or sets (a correct object to tsc);
//   - a ring of imports, and a file of the folder that the window does not reach (tests/imports.mjs's windowed(): the
//     build links a ring without a word);
//   - (T.368.7) a message a section's worker posts to the page without `satisfies` a type of contracts/benchmark.ts: a
//     worker's own postMessage() takes anything, and tsc holds only a message that says so;
//   - that the window links in Node with every export defined, and that a message given to the steps' onmessage as it
//     is set is answered once;
//   - a name that a module of src/bench/ uses and neither declares nor imports: those are JavaScript (T.480), which tsc
//     reads for its shapes and does not check, and the tables are functions (a name that is nowhere is a
//     ReferenceError only when that table is written).
//   - (T.379.2, of T.379.1's review) the packing a step of this worker asks fusedDp4aMatVec for: "int8", written out.
//     The benchmark's matrices are int8, and "ternary" is as good a packing to tsc: the worker would bind int8 weights
//     to a text that reads ternary codes, and its steps run only where there is a GPU;
// What it does not see: an await before the steps' onmessage is set (the import below waits for the module whole).
// The parser is @babel/parser, which Astro's packages bring.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { parse } from "@babel/parser";
import { contract } from "./contracts.mjs";
import { windowed } from "./imports.mjs";
import { treeOf } from "./tree.mjs";
import { typed, untyped } from "./untyped.mjs";
import { namesOf } from "./names.mjs";

// (where the runtime is: tests/tree.mjs. tests/imports.mjs's windowed() reads the window and every file it reaches,
// says a ring, and holds every file of the folder to be reached)
const tree = treeOf();
const at = (name) => `benchmark/${name}`;  // (a file of this worker, by its name under the runtime's folder)
// (T.368.7: the window by the name the tree has for it, gpu.ts: windowed() blanks a file's types by the name it is given,
// and the modules are then named by their own endings)
const WINDOW = path.basename(tree.runtime(at("gpu.js"))), END = path.extname(WINDOW);
const linked = windowed(at(WINDOW), { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), outside: true });
// (a file by its name under the worker's folder; one outside it that a module takes names of, T.367.3's table of the
// built files, by its way from there: "../built.js")
const short = (name) => (name.startsWith(at("")) ? name.slice(at("").length) : `../${name}`);
const programs = new Map([...linked.files].map(([name, { program }]) => [short(name), program]));
// (that none waits for itself, and that the window takes from every module or a module it takes from does: windowed())
console.log(`bench-modules-check: the GPU worker's ${programs.size} files are reached from ${WINDOW}, and none waits for itself`);

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
// (T.379.2) the benchmark's DP4A matrix of a token is the int8 one, said at every call
let asked = 0, named = 0;
for (const [name, program] of programs) {
  walk(program, (node) => {
    if (node.type !== "CallExpression" || node.callee.type !== "MemberExpression" || node.callee.property.name !== "fusedDp4aMatVec") return;
    const given = node.arguments[0];
    const weights = given?.type === "ObjectExpression" ? given.properties.find((property) => property.type === "ObjectProperty" && !property.computed && (property.key.name ?? property.key.value) === "weights") : null;
    assert.ok(weights && weights.value.type === "StringLiteral" && weights.value.value === "int8",
      `${name} calls fusedDp4aMatVec without weights: "int8" written out: the benchmark's matrices are int8 (a ternary text would read them as codes, and nothing here runs it)`);
    asked += 1;
  });
  walk(program, (node) => { named += node.type === "Identifier" && node.name === "fusedDp4aMatVec" ? 1 : 0; });
}
assert.equal(named, asked, "/benchmark/'s worker names fusedDp4aMatVec otherwise than as <the shaders>.fusedDp4aMatVec({ … }): this check reads its calls");
assert.ok(asked >= 4, `the calls of fusedDp4aMatVec in /benchmark/'s worker: ${asked} found (4 when this was written)`);
console.log(`bench-modules-check: the ${fields.length} fields of shared are the ones the steps use (${said} times)`);
console.log(`bench-modules-check: ${asked} calls of fusedDp4aMatVec, each for int8 weights`);

// ---- T.368.7: the workers are TypeScript, and tsc holds a message one posts to the page to the type the message says
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
// (T.368.7: either ending, a .ts file with its types blanked: a module made TypeScript is still read here)
const tables = fs.readdirSync(TABLES).filter((name) => name.endsWith(".js") || typed(name));
for (const name of tables) {
  const unknown = [...namesOf(untyped(fs.readFileSync(new URL(name, TABLES), "utf8"), name)).free].filter((used_) => !TABLE_GLOBALS.has(used_));
  assert.deepEqual(unknown, [], `src/bench/${name} uses ${unknown}, which it neither declares nor imports: a ReferenceError when that table is written`);
}
console.log(`bench-modules-check: the ${tables.length} modules of src/bench/ use nothing they do not declare or import`);
console.log("bench-modules-check: ok");
process.exit(0);
