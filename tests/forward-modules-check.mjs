// tests/forward-modules-check.mjs (T349): what the type check does not hold of what src/runtime/forward.ts and its
// modules (src/runtime/forward/*.ts) hand one another. Node only, under a second:
//
//   node tests/forward-modules-check.mjs
//
// T368.8: forward is TypeScript, and tsc (tests/types.mjs) says what this check said before of a wrong name: a name
// used and declared nowhere, and a name misspelt where createForward() hands a part its things or where the part
// takes them. And that a model on the GPU
// alone is handed no spawn (the syntax of one line here) is tests/gpu-alone-check.mjs's, which makes such an engine
// (T450); that threads.count and onGpu.chosen cannot be assigned, the two parts' own checks'.
// What is left here is what tsc, the build and Node all let pass, each with why:
//   - a ring of imports, and a file of forward/ that the window does not reach (tests/imports.mjs's windowed(): the
//     build links a ring without a word, and Node only where a module uses the other's name as it loads);
//   - a part that takes a thing createForward() never hands it, at the top or out of shape, weights, taken, …:
//     optional in the part's type, it is undefined without a word (the names handed are the names taken);
//   - a key createForward() makes of shape, weights, taken, cpuSide, … that no part takes out: a fact carried for
//     nothing, and a correct object to tsc;
//   - a name taken out of a part's object, or said of one (threads.<name>, onGpu.<name>, …), that is not in it, and
//     a member set that has no setter: tsc says the plain mistake, and not one made through a cast, which this reads
//     with the types blanked;
//   - what is read of the two parts' state: threads has three getters and onGpu seven, and one more is state shown
//     that was the part's own (T378.1, T378.2);
//   - that the window links in Node with no export undefined.
// The parser is @babel/parser, which Astro's packages bring.
import assert from "node:assert/strict";
import fs from "node:fs";
import { windowed } from "./imports.mjs";
import { treeOf } from "./tree.mjs";

// (where the runtime is: tests/tree.mjs. tests/imports.mjs's windowed() reads the window and every file it reaches,
// says a ring, and holds every file of the folder to be reached)
const tree = treeOf();
const linked = windowed("forward.ts", { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), bundled: tree.bundled, more: ["jobs.ts"] });
const modules = linked.there;
const programs = new Map([...linked.files].map(([name, { program }]) => [name, program]));

const keysOf = (pattern, what) => pattern.properties.map((property) => {
  assert.ok(property.type === "ObjectProperty" && !property.computed && property.key.type === "Identifier", `${what}: a plain name`);
  return property.key.name;
});

// ---- the parts: what the engine hands each is what it takes
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
const PARTS = { controlArea: "forward/control.ts", softwareThreads: "forward/threads.ts", gpuFit: "forward/gpuside.ts", gpuSide: "forward/gpuside.ts" };
const objects = new Map();  // the name a part's object has in the engine -> its members
const create = functionOf("forward/engine.ts", "createForward");
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
// taken out of one is a key createForward() made (a misspelt one is tsc's to say; one that is optional in the type and
// never made is undefined without a word), and every key made is taken by some part (one that no part takes is a fact
// carried for nothing). NOT_HANDED: what a part takes with a default and only a test hands it
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
// T378.1, T378.2: the state of the software threads and of the GPU's side has one owner each: what happens is told the
// part by functions named for that, and what is read of it is these getters and no other (that none can be assigned is
// tests/threads-part-check.mjs's and tests/gpu-part-check.mjs's to see, and that each is a member where it is read, tsc's)
const getters = (part) => [...objects.get(part)].filter(([, kinds]) => kinds.has("get")).map(([name]) => name).sort();
assert.deepEqual(getters("threads"), ["count", "lost", "searching"], "what is read of the software threads' state");
assert.deepEqual(getters("onGpu"), ["chosen", "promptTokens", "ready", "sampled", "status", "stepsWhyNot", "whyNot"], "what is read of the GPU's side's state");
// every threads.<name>, onGpu.<name>, fit.<name> and cpuSide.<name> is a member of that object, with a setter where it
// is set. tsc says one that is misspelt, and one set that has no setter (T368.8); these lines are kept for what it
// lets pass: a name said through a cast (the types are blanked here: `(threads as T).cuont` is threads.cuont)
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
console.log(`forward-modules-check: createForward() hands its parts the ${handed} things they take (${grouped} names taken out of ${made.size} of them), ` +
  `and the ${through} places that go through threads, onGpu, fit and cpuSide name members of them; what is read of threads and of onGpu is their getters`);

// ---- T375.3: what the engine hands a kind of layer and the FFN (core), and the arrays of a token's frame
// The kinds (attention.ts, linear.ts, convolution.ts: their make()) and the FFNs (ffn.ts) take core apart where they
// are declared. tsc says a name core has not. What it lets pass: a key of core that none of them takes (a thing
// carried for nothing), and a name taken out of `at` that nobody put in a frame (at is a record of numbers by any
// name: a name that is in no frame is undefined, and every address made of it NaN, without a word). The names of a
// frame are the first strings of the pairs in memory.ts's frameArrays() and in the kinds' frame().
const PARTS_OF_CORE = ["forward/attention.ts", "forward/linear.ts", "forward/convolution.ts", "forward/ffn.ts"];
const coreMade = [...membersOf(literals.get("core")).keys()], coreTaken = new Set();
assert.ok(coreMade.length, "createForward() makes core as an object");
let parts = 0;
for (const file of PARTS_OF_CORE) {
  walk(programs.get(file), (node) => {
    if (!["ObjectMethod", "FunctionDeclaration"].includes(node.type) || node.params[0]?.type !== "ObjectPattern") return;
    const keys = keysOf(node.params[0], file);
    if (!keys.includes("at")) return;  // (frame() takes the plan apart, not core)
    parts += 1;
    for (const name of keys) {
      assert.ok(coreMade.includes(name), `${file} takes ${name} out of core, which createForward() made without it: undefined`);
      coreTaken.add(name);
    }
  });
}
assert.equal(parts, 5, "the three kinds' make() and the two FFNs take core apart");
// (cache: the layers that keep keys and values alone take it; a kind that reads `core.<name>` otherwise is not counted)
assert.deepEqual(coreMade.filter((name) => !coreTaken.has(name)), [], "createForward() makes core with what no kind and no FFN takes out of it");
const framedNames = new Set();
for (const file of ["forward/memory.ts", ...PARTS_OF_CORE]) {
  walk(programs.get(file), (node) => {
    if (node.type === "ArrayExpression" && node.elements.length === 2 && node.elements[0]?.type === "StringLiteral") framedNames.add(node.elements[0].value);
  });
}
let framedTaken = 0;
for (const file of ["forward/engine.ts", ...PARTS_OF_CORE]) {
  walk(programs.get(file), (node) => {
    const names = node.type === "VariableDeclarator" && node.id.type === "ObjectPattern" && node.init?.type === "Identifier" && node.init.name === "at" ? keysOf(node.id, file)
      : node.type === "MemberExpression" && node.object.type === "Identifier" && node.object.name === "at" && !node.computed ? [node.property.name] : [];
    for (const name of names) {
      assert.ok(framedNames.has(name), `${file} takes ${name} out of a token's frame, where nothing puts an array of that name: undefined`);
      framedTaken += 1;
    }
  });
}
console.log(`forward-modules-check: the kinds of layers and the FFNs take the ${coreMade.length} things of core, and the ${framedTaken} arrays they and the engine take of a token's frame are ` +
  `among the ${framedNames.size} that are put in one`);

// ---- the window links, and exports nothing undefined
const window = await import(tree.runtimeUrl("forward.js"));
for (const [name, value] of Object.entries(window)) assert.notEqual(value, undefined, `forward.js exports ${name} as undefined`);
console.log(`forward-modules-check: forward.js links its ${modules.length} modules and exports ${Object.keys(window).length} names`);
