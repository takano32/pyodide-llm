// tests/forward-modules-check.mjs (T349): the names that src/runtime/forward.js and its modules (src/runtime/forward/*.js) hand
// one another. Three things there are no variables, so that a name spelled wrong is `undefined` without a word:
//   - a name a file uses: every one is declared in it, imported by it, or a global (a ReferenceError where the line
//     runs, otherwise: neither the build nor Node says it before);
//   - what createForward() (forward/engine.js) hands a part (controlArea, softwareThreads, gpuFit, gpuSide) in one object, which the
//     part takes apart by name: the names handed are the names taken, and what the engine takes out of the part's
//     object is in it. T378.2: a part is handed a few things (shape, weights, cpuSide, ...), each an object it takes apart
//     once: the names taken out of each are keys of the object createForward() made, and no key is made for nothing;
//   - the parts' objects (threads, onGpu, fit, cpuSide): every threads.<name> is a member of what softwareThreads() returns, with a
//     setter where it is set (a member that is none would read undefined, and set nothing the part reads). T378.1: threads
//     has no setter at all, and three getters. T378.2: onGpu (gpuSide()'s) has none either.
// And every file of the folder is reached from the window, and the window links (Node imports it) with no export undefined.
// T367.5: what this held before and the build and Node hold by themselves now is gone from here: that a name taken is
// exported where it is taken from (the build: MISSING_EXPORT, an error; Node: a SyntaxError as the module is imported).
// Node only, under a second:
//
//   node tests/forward-modules-check.mjs
//
// The parser is @babel/parser, which Astro's packages bring (as tests/worker-modules-check.mjs, whose namesOf this uses).
import assert from "node:assert/strict";
import fs from "node:fs";
import { windowed } from "./imports.mjs";
import { treeOf } from "./tree.mjs";
import { namesOf } from "./worker-modules-check.mjs";

const GLOBALS = new Set(("Array Atomics BigInt BigInt64Array Boolean Error Float32Array Float64Array Infinity Int32Array Int8Array Map Math NaN Number Object " +
  "Promise Set SharedArrayBuffer Uint16Array Uint8Array URL WebAssembly clearTimeout console performance setTimeout undefined").split(" "));
// (where the runtime is: tests/tree.mjs. tests/imports.mjs's windowed() reads the window and every file it reaches,
// says a ring, and holds every file of the folder to be reached)
const tree = treeOf();
const linked = windowed("forward.js", { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), bundled: tree.bundled, more: ["jobs.ts"] });
const modules = linked.there;
const files = new Map([...linked.files].map(([name, { text }]) => [name, text]));
const programs = new Map([...linked.files].map(([name, { program }]) => [name, program]));

// ---- every name used is declared, imported, or a global
const keysOf = (pattern, what) => pattern.properties.map((property) => {
  assert.ok(property.type === "ObjectProperty" && !property.computed && property.key.type === "Identifier", `${what}: a plain name`);
  return property.key.name;
});
let names = 0;
for (const [name] of programs) {
  if (name === "jobs.ts") continue;
  const { free } = namesOf(files.get(name)), unknown = [...free].filter((used) => !GLOBALS.has(used));
  assert.deepEqual(unknown, [], `${name} uses ${unknown}, which it neither declares nor imports: a ReferenceError where the line runs`);
  names += free.size;
}
console.log(`forward-modules-check: ${programs.size - 1} files use nothing they do not declare or import (${names} names of the globals)`);

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
const PARTS = { controlArea: "forward/control.js", softwareThreads: "forward/threads.js", gpuFit: "forward/gpuside.js", gpuSide: "forward/gpuside.js" };
const objects = new Map();  // the name a part's object has in the engine -> its members
const create = functionOf("forward/engine.js", "createForward");
let handed = 0;
const calls = [], literals = new Map();  // T378.2: each part's call, and the objects createForward() makes by name
walk(create, (node) => {
  if (node.type === "VariableDeclarator" && node.id.type === "Identifier" && node.init?.type === "ObjectExpression") literals.set(node.id.name, node.init);
});
walk(create, (node) => {
  if (node.type !== "VariableDeclarator" || !node.init) return;
  const call = node.init.type === "CallExpression" && node.init.callee.type === "Identifier" && PARTS[node.init.callee.name] ? node.init : null;
  if (call) {
    const part = call.callee.name, made = functionOf(PARTS[part], part);
    assert.ok(made && made.params.length === 1 && made.params[0].type === "ObjectPattern", `${part} takes one object apart`);
    assert.ok(call.arguments.length === 1 && call.arguments[0].type === "ObjectExpression", `createForward() hands ${part} one object`);
    assert.deepEqual(keysOf(call.arguments[0], part).sort(), keysOf(made.params[0], part).sort(), `what createForward() hands ${part} is what it takes`);
    handed += call.arguments[0].properties.length;
    calls.push([part, made, call]);
    // T378.1: a model on the GPU alone (direct, T156) runs nothing on the CPU, and the one way the threads' part knows
    // is that it is handed no spawn (T450: tests/gpu-alone-check.mjs makes such an engine and sees that none is started)
    if (part === "softwareThreads") {
      const spawn = call.arguments[0].properties.find((property) => property.key.name === "spawn").value;
      assert.ok(spawn.type === "ConditionalExpression" && spawn.test.type === "Identifier" && spawn.test.name === "direct" && spawn.consequent.type === "NullLiteral" &&
        spawn.alternate.type === "Identifier" && spawn.alternate.name === "spawn", "createForward() hands softwareThreads no spawn where the model is on the GPU alone (spawn: direct ? null : spawn): software threads would be started there");
    }
    const returned = made.body.body.at(-1);
    assert.ok(returned.type === "ReturnStatement" && returned.argument.type === "ObjectExpression", `${part} returns one object`);
    const members = membersOf(returned.argument);
    if (node.id.type === "Identifier") objects.set(node.id.name, members);
    else keysOf(node.id, part).forEach((name) => assert.ok(members.has(name), `createForward() takes ${name} out of ${part}'s object, which has none`));
  } else if (node.id.type === "Identifier" && node.init.type === "ObjectExpression" && node.id.name === "cpuSide") {
    objects.set("cpuSide", membersOf(node.init));
  } else if (node.id.type === "ObjectPattern" && node.init.type === "Identifier" && objects.has(node.init.name)) {
    keysOf(node.id, node.init.name).forEach((name) => assert.ok(objects.get(node.init.name).has(name), `createForward() takes ${name} out of ${node.init.name}, which has none`));
  }
});
assert.deepEqual([...objects.keys()].sort(), ["control", "cpuSide", "fit", "onGpu", "threads"], "the parts' objects in createForward()");
// T378.2: a part is handed a few things, each an object it takes apart at its top (const { a, b } = shape): every name
// taken out of one is a key of the object createForward() made for it (a name that is none is undefined without a
// word, as before), and every key made is taken by some part (one that no part takes is a fact carried for nothing).
// NOT_HANDED: what a part takes with a default and only a test hands it
const NOT_HANDED = { gpuSide: { options: ["now"] } };
const made = new Map(), taken = new Map();  // an object of createForward() -> its keys; -> the keys the parts take
let grouped = 0;
for (const [part, declared, call] of calls) {
  const groups = new Map(call.arguments[0].properties.map((property) => [property.key.name, property.value]));
  for (const statement of declared.body.body) {
    if (statement.type !== "VariableDeclaration") continue;
    for (const { id, init } of statement.declarations) {
      if (id.type !== "ObjectPattern" || init?.type !== "Identifier" || !groups.has(init.name)) continue;
      const value = groups.get(init.name), literal = value.type === "Identifier" ? literals.get(value.name) : value.type === "ObjectExpression" ? value : null;
      if (value.type === "Identifier" && objects.has(value.name) && !literal) {
        // (another part's object: softwareThreads()'s, handed to gpuSide())
        keysOf(id, part).forEach((name) => assert.ok(objects.get(value.name).has(name), `${part} takes ${name} out of ${init.name}, which ${value.name} has not`));
        continue;
      }
      assert.ok(literal, `${part} takes ${init.name} apart, which createForward() makes as an object of its own`);
      const where = value.type === "Identifier" ? value.name : `${part}.${init.name}`, keys = [...membersOf(literal).keys()];
      made.set(where, keys);
      taken.set(where, taken.get(where) ?? new Set());
      for (const name of keysOf(id, part)) {
        if (NOT_HANDED[part]?.[init.name]?.includes(name)) {
          assert.ok(!keys.includes(name), `${part}'s ${init.name}.${name} is handed by createForward() now: take it off NOT_HANDED`);
          continue;
        }
        assert.ok(keys.includes(name), `${part} takes ${name} out of ${init.name}, which createForward() made without it: undefined`);
        taken.get(where).add(name);
        grouped += 1;
      }
    }
  }
}
// (cpuSide's four getters are read as cpuSide.<name> each time, not taken out once: the places below hold those)
const READ_THROUGH = { cpuSide: ["H", "capacity", "keys", "values"] };
for (const [where, keys] of made) {
  const unread = keys.filter((name) => !taken.get(where).has(name) && !READ_THROUGH[where]?.includes(name));
  assert.deepEqual(unread, [], `createForward() makes ${where} with ${unread}, which no part takes out of it`);
}
assert.deepEqual([...made.keys()].sort(), ["cpuSide", "gpuFit.device", "gpuSide.options", "shape", "taken", "weights"], "the things createForward() hands its parts");
// T378.1: the software threads' state has one owner: its object has no setter (what happens is told it by functions
// named for that), and the three things read of it are all it has that is no function of its own or a fact made once
const settable = [...objects.get("threads")].filter(([, kinds]) => kinds.has("set")).map(([name]) => name);
assert.deepEqual(settable, [], `softwareThreads()'s object has a setter (${settable}): its variables are its own, and what happens is told it by a function`);
assert.deepEqual([...objects.get("threads")].filter(([, kinds]) => kinds.has("get")).map(([name]) => name).sort(), ["count", "lost", "searching"], "what is read of the software threads' state");
// T378.2: the GPU's side's state the same: no setter, and what is read of it is what the page is shown
const assignable = [...objects.get("onGpu")].filter(([, kinds]) => kinds.has("set")).map(([name]) => name);
assert.deepEqual(assignable, [], `gpuSide()'s object has a setter (${assignable}): its variables are its own, and what happens is told it by a function`);
assert.deepEqual([...objects.get("onGpu")].filter(([, kinds]) => kinds.has("get")).map(([name]) => name).sort(),
  ["chosen", "promptTokens", "ready", "sampled", "status", "stepsWhyNot", "whyNot"], "what is read of the GPU's side's state");
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
console.log(`forward-modules-check: createForward() hands its parts the ${handed} things they take (${grouped} names taken out of ${made.size} of them), and the ${through} places that go through threads, onGpu, fit and cpuSide name members of them`);

// ---- the window links, and exports nothing undefined
const window = await import(tree.runtimeUrl("forward.js"));
for (const [name, value] of Object.entries(window)) assert.notEqual(value, undefined, `forward.js exports ${name} as undefined`);
console.log(`forward-modules-check: forward.js links its ${modules.length} modules and exports ${Object.keys(window).length} names`);
