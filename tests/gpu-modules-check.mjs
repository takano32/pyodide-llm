// tests/gpu-modules-check.mjs (T352): the names that the model's GPU worker (src/runtime/gpu.js) and its modules
// (src/runtime/gpu/*.js) use. Nothing else holds them outside a browser with a GPU: the worker runs only there
// (CI's gpu-check.mjs), and the checks of forward.js's choices make a GPU's worker of their own.
//   - every name a file uses is declared in it, imported by it, or a worker's global (a ReferenceError where the line
//     runs, otherwise: neither the build nor Node says it before);
//   - T378.4: the model on the GPU, whether a stop was asked and why the device was lost are gpu/device.js's own
//     variables (model, stopping, lost): no other module sets them (an import cannot be assigned: a TypeError where
//     the line runs, which neither Node nor the build says before), and no file binds a name of theirs (a parameter or
//     a local would hide the variable from the function's lines);
//   - T378.4: the model's record (m, model) is device.js's fields and four records, m.weights, m.block, m.forms and
//     m.tokens, each made and set by one module: every field read is one its owner declares where it makes the record,
//     every write is in the owner's file, and no record is handed on whole (see the section below);
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
import path from "node:path";
import { windowed } from "./imports.mjs";
import { treeOf } from "./tree.mjs";
import { namesOf } from "./worker-modules-check.mjs";

const GLOBALS = new Set(("Array ArrayBuffer Atomics BigInt BigInt64Array Boolean Error Float32Array Float64Array Infinity Int32Array Int8Array Map Math " +
  "NaN Number Object Promise Set String URL Uint16Array Uint32Array Uint8Array clearInterval clearTimeout navigator onmessage performance postMessage " +
  "self setInterval setTimeout undefined").split(" "));
// (where the runtime is: tests/tree.mjs. tests/imports.mjs's windowed() reads the window and every file it reaches,
// says a ring, and holds every file of gpu/ to be reached; the shaders' parts a module takes names of are read with them)
const tree = treeOf();
// (T368.6: the window by the name the tree has for it, gpu.ts: windowed() blanks a file's types by the name it is given)
const WINDOW = path.basename(tree.runtime("gpu.js")), END = path.extname(WINDOW);
const linked = windowed(WINDOW, { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), bundled: tree.bundled, outside: true });
const own = (name) => name === WINDOW || name.startsWith("gpu/");
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
    assert.ok(own(from) || /^shaders\/\w+\.[jt]s$/.test(from), `${name} takes names from ${from}: a module of the worker takes of the worker's modules and of the shaders' parts (shaders/<part>.js)`);
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
const HELD = ["model", "stopping", "lost"], OWNER = `gpu/device${END}`;
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

// ---- the model's record (T378.4): what device.js puts in it, and the four records, each one module's
// A field is no variable: a wrong name reads undefined and sets nothing another module reads, and neither Node nor the
// build says so. Every m.<name> and model.<name> is a field device.js's openDevice() declares or one of the records;
// every field read of a record is one its owner declares where it makes the record (m.<record> = { ... }: the literal's
// keys, and the keys of what a builder of another module returns where the literal spreads it); a record and its
// fields are set in the owner's file alone; and a record goes nowhere whole (taken apart or under a local name in
// the function that reads it, where this follows it).
const RECORDS = { weights: `gpu/weights${END}`, block: `gpu/block${END}`, forms: `gpu/forms${END}`, tokens: `gpu/tokenforms${END}` };
const MODEL = ["m", "model"];
const isMember = (node) => node.type === "MemberExpression" || node.type === "OptionalMemberExpression";
const ofModel = (node) => isMember(node) && !node.computed && node.object.type === "Identifier" && MODEL.includes(node.object.name);
const keysOf = (literal, where) => literal.properties.filter((p) => p.type !== "SpreadElement").map((p) => (assert.ok(p.type === "ObjectProperty" && !p.computed, `${where}: a field by a computed name`), p.key.name));
// walk with the nodes above (path[0] is the parent)
const descend = (node, visit, path = [], key = "") => {
  if (!node || typeof node.type !== "string") return;
  visit(node, path, key);
  for (const [k, value] of Object.entries(node)) {
    if (k === "loc" || k.endsWith("Comments") || !value || typeof value !== "object") continue;
    (Array.isArray(value) ? value : [value]).forEach((child) => descend(child, visit, [node, ...path], k));
  }
};
const functions = new Map();  // every function declared at the top of a module, by its name
for (const [name, program] of programs) for (const node of program.body.map(top)) if (node.type === "FunctionDeclaration") functions.set(node.id.name, { node, file: name });
// what a builder returns: the keys of the literal it returns, or of the one it declares and returns by name (and what
// it sets of that one by name is one of them)
function builtBy(fn) {
  const { node, file } = functions.get(fn) ?? {};
  assert.ok(node, `no function ${fn} at the top of a module of the worker`);
  const literals = new Map(), keys = new Set();
  descend(node.body, (n) => { if (n.type === "VariableDeclarator" && n.id.type === "Identifier" && n.init?.type === "ObjectExpression") literals.set(n.id.name, n.init); });
  let local = null;
  descend(node.body, (n, path) => {
    if (n.type !== "ReturnStatement" || !n.argument || n.argument.type === "NullLiteral") return;
    if (path.some((above) => /Function/.test(above.type))) return;  // (a function inside it returns its own)
    const literal = n.argument.type === "ObjectExpression" ? n.argument : literals.get(n.argument.name);
    assert.ok(literal, `${file}: ${fn}() returns something that is no object written out (line ${n.loc.start.line})`);
    if (n.argument.type === "Identifier") local = n.argument.name;
    keysOf(literal, `${file}: ${fn}()`).forEach((key) => keys.add(key));
  });
  assert.ok(keys.size, `${file}: ${fn}() returns no object`);
  if (local) {
    descend(node.body, (n, path) => {
      if (isMember(n) && !n.computed && n.object.type === "Identifier" && n.object.name === local) {
        assert.ok(keys.has(n.property.name), `${file}: ${fn}() says ${local}.${n.property.name}, which is no field of the ${local} it declares (line ${n.loc.start.line})`);
      }
    });
  }
  return keys;
}
// model, stopping and lost are read where a function runs, never as a module loads: a value taken then (const gone =
// lost) is a copy that stays what it was, while the name itself is device.js's variable as it is now
for (const [name, program] of programs) {
  descend(program, (node, path, key) => {
    if (node.type !== "Identifier" || !HELD.includes(node.name) || path.some((above) => /Function/.test(above.type))) return;
    const parent = path[0];
    if ((isMember(parent) && key === "property" && !parent.computed) || (parent.type === "ObjectProperty" && key === "key") || parent.type === "ImportSpecifier" || parent.type === "ExportSpecifier") return;
    if (name === OWNER && parent.type === "VariableDeclarator" && key === "id") return;
    assert.fail(`${name} reads ${node.name} as it loads (line ${node.loc.start.line}): a copy of what it was then; read ${OWNER}'s ${node.name} where it is used`);
  });
}
// ---- the fields: device.js's own, and each record's
let base = null;
descend(programs.get(OWNER), (node) => {
  if (node.type === "AssignmentExpression" && node.left.type === "Identifier" && node.left.name === "model" && node.right.type === "ObjectExpression") {
    assert.equal(base, null, `${OWNER} makes the model's record in one place`);
    base = new Set(keysOf(node.right, OWNER));
  }
});
assert.ok(base, `${OWNER} makes the model's record (model = { ... })`);
for (const record of Object.keys(RECORDS)) assert.ok(!base.has(record), `${OWNER} declares ${record}, which is ${RECORDS[record]}'s record`);
const fieldsOf = {};
for (const [record, owner] of Object.entries(RECORDS)) {
  descend(programs.get(owner), (node, path) => {
    if (node.type !== "AssignmentExpression" || !ofModel(node.left) || node.left.property.name !== record || node.right.type === "NullLiteral") return;
    assert.equal(node.right.type, "ObjectExpression", `${owner} sets m.${record} to something that is no object written out (line ${node.loc.start.line}): its fields are declared where it is made`);
    assert.equal(fieldsOf[record], undefined, `${owner} makes m.${record} in one place`);
    const fields = new Set(keysOf(node.right, owner));
    for (const spread of node.right.properties.filter((p) => p.type === "SpreadElement")) {
      // (...made, where `made = await builder(m)` in the same function)
      assert.equal(spread.argument.type, "Identifier", `${owner}: m.${record} spreads something that is no name`);
      const scope = path.find((above) => /Function/.test(above.type));
      let builder = null;
      descend(scope.body, (n) => {
        if (n.type !== "VariableDeclarator" || n.id.name !== spread.argument.name) return;
        const call = n.init?.type === "AwaitExpression" ? n.init.argument : n.init;
        if (call?.type === "CallExpression" && call.callee.type === "Identifier") builder = call.callee.name;
      });
      assert.ok(builder, `${owner}: what m.${record} spreads (${spread.argument.name}) is not what a function returned`);
      builtBy(builder).forEach((key) => { assert.ok(!fields.has(key), `m.${record}.${key} twice`); fields.add(key); });
    }
    fieldsOf[record] = fields;
  });
  assert.ok(fieldsOf[record], `${owner} makes m.${record} (m.${record} = { ... })`);
}
// ---- every use
const reads = { base: new Set(), ...Object.fromEntries(Object.keys(RECORDS).map((record) => [record, new Set()])) };
let uses = 0;
const sets = (node, path, key) => (path[0].type === "AssignmentExpression" && key === "left") || path[0].type === "UpdateExpression";
// a field of a record used: node is <the record>.<field>
const fieldUsed = (name, record, node, path, key, how) => {
  assert.ok(fieldsOf[record].has(node.property.name), `${name} says ${how}.${node.property.name}, which is no field ${RECORDS[record]} declares of m.${record} (line ${node.loc.start.line}): undefined`);
  if (sets(node, path, key)) assert.equal(name, RECORDS[record], `${name} sets ${how}.${node.property.name} (line ${node.loc.start.line}): m.${record} is ${RECORDS[record]}'s to set`);
  else reads[record].add(node.property.name);
  uses += 1;
};
const pattern = (name, record, id, how) => {
  for (const property of id.properties) {
    assert.ok(property.type === "ObjectProperty" && !property.computed, `${name} takes ${how} apart otherwise than by its fields' names (line ${id.loc.start.line})`);
    assert.ok(fieldsOf[record].has(property.key.name), `${name} takes ${property.key.name} of ${how}, which is no field ${RECORDS[record]} declares of m.${record} (line ${property.loc.start.line}): undefined`);
    reads[record].add(property.key.name);
    uses += 1;
  }
};
for (const [name, program] of programs) {
  const aliases = [];  // { scope: the function it is declared in, local, record }
  const alias = (record, local, path) => aliases.push({ scope: path.find((above) => /Function/.test(above.type)) ?? program, local, record });
  descend(program, (node, path, key) => {
    // const { plan, tokens: g, block: { x } } = m
    if (node.type === "VariableDeclarator" && node.id.type === "ObjectPattern" && node.init?.type === "Identifier" && MODEL.includes(node.init.name)) {
      for (const property of node.id.properties) {
        assert.ok(property.type === "ObjectProperty" && !property.computed, `${name} takes the model's record apart otherwise than by names (line ${node.loc.start.line})`);
        const field = property.key.name;
        if (field in RECORDS) {
          if (property.value.type === "Identifier") alias(field, property.value.name, path);
          else if (property.value.type === "ObjectPattern") pattern(name, field, property.value, `m.${field}`);
          else assert.fail(`${name} takes m.${field} with a default (line ${property.loc.start.line})`);
        } else {
          assert.ok(base.has(field), `${name} takes ${field} of the model's record, which is neither a field ${OWNER} declares nor a record (line ${property.loc.start.line}): undefined`);
          reads.base.add(field);
        }
        uses += 1;
      }
    }
    if (!ofModel(node)) return;
    const field = node.property.name, parent = path[0];
    if (!(field in RECORDS)) {
      assert.ok(base.has(field), `${name} says ${node.object.name}.${field}, which is neither a field ${OWNER} declares of the model's record nor one of its records (${Object.keys(RECORDS).join(", ")}) (line ${node.loc.start.line}): undefined`);
      if (sets(node, path, key)) assert.equal(name, OWNER, `${name} sets ${node.object.name}.${field} (line ${node.loc.start.line}): ${OWNER} holds it; say what happened to ${OWNER} instead`);
      else reads.base.add(field);
      uses += 1;
      return;
    }
    const how = `${node.object.name}.${field}`;
    if (isMember(parent) && key === "object" && !parent.computed) return fieldUsed(name, field, parent, path.slice(1), Object.keys(path[1]).find((k) => path[1][k] === parent || (Array.isArray(path[1][k]) && path[1][k].includes(parent))), how);
    if (parent.type === "AssignmentExpression" && key === "left") return assert.equal(name, RECORDS[field], `${name} sets ${how} (line ${node.loc.start.line}): it is ${RECORDS[field]}'s to make`);
    if (parent.type === "VariableDeclarator" && key === "init" && parent.id.type === "Identifier") return alias(field, parent.id.name, path);
    if (parent.type === "VariableDeclarator" && key === "init" && parent.id.type === "ObjectPattern") return pattern(name, field, parent.id, how);
    assert.fail(`${name} hands ${how} on whole (line ${node.loc.start.line}): its fields are read by name, where this check sees them`);
  });
  // a record under a local name: every <local>.<field> in the function that declared it
  for (const { scope, local, record } of aliases) {
    assert.ok(!aliases.some((other) => other.local === local && other.record !== record), `${name} names two records ${local}`);
    descend(scope, (node, path, key) => {
      if (node.type !== "Identifier" || node.name !== local) return;
      const parent = path[0];
      if (isMember(parent) && key === "property" && !parent.computed) return;
      if ((parent.type === "ObjectProperty" && key === "key" && !parent.shorthand) || (parent.type === "ObjectProperty" && path[1].type === "ObjectPattern") || (parent.type === "VariableDeclarator" && key === "id")) return;
      // (g?.form, g.most, …: a field read; the name alone in a condition: whether there is a record)
      if (isMember(parent) && key === "object" && !parent.computed) {
        return fieldUsed(name, record, parent, path.slice(1), Object.keys(path[1]).find((k) => path[1][k] === parent || (Array.isArray(path[1][k]) && path[1][k].includes(parent))), local);
      }
      const asked = ["UnaryExpression", "LogicalExpression", "ConditionalExpression", "IfStatement"].includes(parent.type);
      assert.ok(asked, `${name} hands ${local} (m.${record}) on whole (line ${node.loc.start.line}): its fields are read by name, where this check sees them`);
    });
  }
}
for (const [record, fields] of Object.entries(fieldsOf)) {
  assert.deepEqual([...fields].filter((field) => !reads[record].has(field)), [], `fields of m.${record} that nothing reads`);
}
assert.deepEqual([...base].filter((field) => !reads.base.has(field)), [], `fields ${OWNER} declares of the model's record that nothing reads`);
console.log(`gpu-modules-check: the model's record: ${base.size} fields of gpu/device.js's and ${Object.entries(fieldsOf).map(([record, fields]) => `${fields.size} of m.${record}`).join(", ")}, ` +
  `each set by its owner alone and read by a name it declares (${uses} uses)`);

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
