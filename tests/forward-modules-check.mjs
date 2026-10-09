// tests/forward-modules-check.mjs (T349): the names that public/forward.js and its modules (public/forward/*.js) hand
// one another. Three things there are no variables, so that a name spelled wrong is `undefined` without a word:
//   - a name a file takes out of another (`const { a, b } = await import(...)`, `= await modules.x`): every one is
//     exported there, and every name a file uses is declared in it, taken from another, or a global;
//   - what createForward() (forward/engine.js) hands a part (softwareThreads, gpuFit, gpuSide) in one object, which the
//     part takes apart by name: the names handed are the names taken, and what the engine takes out of the part's
//     object is in it;
//   - the parts' objects (pool, gpuPart, held): every pool.<name> is a member of what softwareThreads() returns, with a
//     setter where it is set (a member that is none would read undefined, and set nothing the part reads).
// And the folder holds the modules the window asks for, and the window links (Node imports it) with no export undefined.
// Node only, under a second:
//
//   node tests/forward-modules-check.mjs
//
// The parser is @babel/parser, which Astro's packages bring (as tests/worker-modules-check.mjs, whose namesOf this uses).
import assert from "node:assert/strict";
import fs from "node:fs";
import { parse } from "@babel/parser";
import { namesOf } from "./worker-modules-check.mjs";

const GLOBALS = new Set(("Array Atomics BigInt BigInt64Array Boolean Error Float32Array Float64Array Infinity Int32Array Int8Array Map Math NaN Number Object " +
  "Promise Set SharedArrayBuffer Uint16Array Uint8Array URL WebAssembly clearTimeout console performance setTimeout undefined").split(" "));
const PUBLIC = new URL("../public/", import.meta.url);
const read = (name) => fs.readFileSync(new URL(name, PUBLIC), "utf8");
const asked = /\[([^\]]*)\]\.map\(\(name\) =>\n\s*\[name, import\(new URL\(`forward\/\$\{name\}\.js/.exec(read("forward.js"));
assert.ok(asked, "forward.js asks for its modules in one list");
const modules = [...asked[1].matchAll(/"(\w+)"/g)].map((match) => match[1]);
const there = fs.readdirSync(new URL("forward/", PUBLIC)).filter((name) => name.endsWith(".js")).map((name) => name.slice(0, -3)).sort();
assert.deepEqual([...modules].sort(), there, "the modules forward.js asks for are the files of public/forward/");

const files = new Map([["forward.js", read("forward.js")], ["jobs.js", read("jobs.js")], ...modules.map((name) => [`forward/${name}.js`, read(`forward/${name}.js`)])]);
const programs = new Map([...files].map(([name, text]) => [name, parse(text, { sourceType: "module" }).program]));
const exported = new Map([...programs].map(([name, program]) => [name, new Set(program.body.flatMap((node) => {
  if (node.type !== "ExportNamedDeclaration") return [];
  if (!node.declaration) return node.specifiers.map((specifier) => specifier.exported.name);
  return node.declaration.id ? [node.declaration.id.name] : node.declaration.declarations.flatMap((d) => d.id.type === "Identifier" ? [d.id.name] : d.id.properties.map((p) => p.value.name));
}))]));

// ---- every name used is declared, taken from another file, or a global; and what is taken is exported there
const keysOf = (pattern, what) => pattern.properties.map((property) => {
  assert.ok(property.type === "ObjectProperty" && !property.computed && property.key.type === "Identifier", `${what}: a plain name`);
  return property.key.name;
});
let taken = 0;
for (const [name, program] of programs) {
  if (name === "jobs.js") continue;
  const unknown = [...namesOf(files.get(name)).free].filter((used) => !GLOBALS.has(used));
  assert.deepEqual(unknown, [], `${name} uses ${unknown}, which it neither declares nor takes from another file: a ReferenceError where the line runs`);
  for (const node of program.body) {
    const d = node.type === "VariableDeclaration" ? node.declarations[0] : null;
    if (d?.init?.type !== "AwaitExpression" || d.id.type !== "ObjectPattern") continue;
    const from = files.get(name).slice(d.init.start, d.init.end);
    const source = /import\(new URL\(`([\w./]+)\$\{new URL\(import\.meta\.url\)\.search\}`, import\.meta\.url\)\)$/.exec(from)?.[1] ?? /^await modules\.(\w+)$/.exec(from)?.[1];
    assert.ok(source, `${name}: \`${from}\` is neither a module of this deployment (its own ?v=) nor one of the window's list`);
    const file = source.endsWith(".js") ? new URL(source, new URL(name, PUBLIC)).href.slice(PUBLIC.href.length) : `forward/${source}.js`;
    assert.ok(exported.has(file), `${name} takes names from ${file}, which is no file of the forward pass`);
    for (const wanted of keysOf(d.id, name)) {
      assert.ok(exported.get(file).has(wanted), `${name} takes ${wanted} from ${file}, which does not export it: undefined`);
      taken += 1;
    }
  }
}
console.log(`forward-modules-check: ${programs.size - 1} files use nothing they do not declare or take from another, and the ${taken} names they take are exported`);

// ---- the parts: what the engine hands each is what it takes, and what a part's object is asked for is in it
const functionOf = (file, name) => programs.get(file).body.map((node) => node.declaration ?? node).find((node) => node.type === "FunctionDeclaration" && node.id.name === name);
const walk = (node, visit) => {
  if (!node || typeof node.type !== "string") return;
  visit(node);
  for (const [key, value] of Object.entries(node)) {
    if (key === "loc" || key.endsWith("Comments") || !value || typeof value !== "object") continue;
    (Array.isArray(value) ? value : [value]).forEach((child) => walk(child, visit));
  }
};
const membersOf = (object) => {
  const members = new Map();
  for (const property of object.properties) {
    assert.ok(!property.computed && property.key.type === "Identifier", "a part's object: plain names");
    const kinds = members.get(property.key.name) ?? new Set();
    kinds.add(property.type === "ObjectMethod" && property.kind !== "method" ? property.kind : "value");
    members.set(property.key.name, kinds);
  }
  return members;
};
const PARTS = { softwareThreads: "forward/threads.js", gpuFit: "forward/gpuside.js", gpuSide: "forward/gpuside.js" };
const objects = new Map();  // the name a part's object has in the engine -> its members
const create = functionOf("forward/engine.js", "createForward");
let handed = 0;
walk(create, (node) => {
  if (node.type !== "VariableDeclarator" || !node.init) return;
  const call = node.init.type === "CallExpression" && node.init.callee.type === "Identifier" && PARTS[node.init.callee.name] ? node.init : null;
  if (call) {
    const part = call.callee.name, made = functionOf(PARTS[part], part);
    assert.ok(made && made.params.length === 1 && made.params[0].type === "ObjectPattern", `${part} takes one object apart`);
    assert.ok(call.arguments.length === 1 && call.arguments[0].type === "ObjectExpression", `createForward() hands ${part} one object`);
    assert.deepEqual(keysOf(call.arguments[0], part).sort(), keysOf(made.params[0], part).sort(), `what createForward() hands ${part} is what it takes`);
    handed += call.arguments[0].properties.length;
    const returned = made.body.body.at(-1);
    assert.ok(returned.type === "ReturnStatement" && returned.argument.type === "ObjectExpression", `${part} returns one object`);
    const members = membersOf(returned.argument);
    if (node.id.type === "Identifier") objects.set(node.id.name, members);
    else keysOf(node.id, part).forEach((name) => assert.ok(members.has(name), `createForward() takes ${name} out of ${part}'s object, which has none`));
  } else if (node.id.type === "Identifier" && node.init.type === "ObjectExpression" && node.id.name === "held") {
    objects.set("held", membersOf(node.init));
  } else if (node.id.type === "ObjectPattern" && node.init.type === "Identifier" && objects.has(node.init.name)) {
    keysOf(node.id, node.init.name).forEach((name) => assert.ok(objects.get(node.init.name).has(name), `createForward() takes ${name} out of ${node.init.name}, which has none`));
  }
});
assert.deepEqual([...objects.keys()].sort(), ["gpuPart", "held", "pool"], "the parts' objects in createForward()");
let through = 0;
for (const [name, program] of programs) {
  walk(program, (node) => {
    const target = node.type === "AssignmentExpression" ? node.left : node.type === "UpdateExpression" ? node.argument : null;
    for (const [member, set] of [[target, true], [node, false]]) {
      if (member?.type !== "MemberExpression" && member?.type !== "OptionalMemberExpression") continue;
      if (member.object.type !== "Identifier" || !objects.has(member.object.name) || member.computed) continue;
      const kinds = objects.get(member.object.name).get(member.property.name);
      assert.ok(kinds, `${name} says ${member.object.name}.${member.property.name}, which is no member of that object: undefined`);
      if (set) assert.ok(kinds.has("set"), `${name} sets ${member.object.name}.${member.property.name}, which has no setter: nothing is set`);
      else through += 1;
    }
  });
}
console.log(`forward-modules-check: createForward() hands its parts the ${handed} names they take, and the ${through} places that go through pool, gpuPart and held name members of them`);

// ---- the window links, and exports nothing undefined
const window = await import("../public/forward.js");
for (const [name, value] of Object.entries(window)) assert.notEqual(value, undefined, `forward.js exports ${name} as undefined`);
console.log(`forward-modules-check: forward.js links its ${modules.length} modules and exports ${Object.keys(window).length} names`);
