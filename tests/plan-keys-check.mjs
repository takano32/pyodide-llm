// plan-keys-check.mjs (T357)
// The two plans that cross a line as plain objects, held to both of their ends: a key one side writes and the other
// never reads, or reads and is never written, is said. Nothing else holds them: a key that is not there is undefined
// without a word (`plan.rotary` spelled otherwise turns whole heads), and the tests that run forward.js alone are
// handed a plan written by hand (tests/plans.mjs's planOf(), held to Python's keys here too).
//
//   1. Python -> forward.js: the plan Llama(external=) hands start() (public/engine/external.py's ExternalForward), as
//      Python builds it for one made-up model of every layout and dtype (tests/engine_plans.py, the native Python):
//      its keys, the names of its tensors, each tensor's keys, the names of its derived tables. Against what
//      public/forward.js and public/forward/*.js read, by their syntax trees: `plan.<key>` and what is taken out of
//      `plan`; the names handed to matrix() and floats() and read off `T` (plan.tensors); the properties read off a
//      tensor (`t.<key>`, `T.<name>.<key>`, `tensors[...]`).
//   2. forward.js -> the GPU's worker: the plan of { type: "start" } (public/forward/gpuside.js's literal) and of
//      { type: "open" } (public/forward/alone.js's gpuOnlyPlan()), with what they hold one level down where that is
//      written as a literal (tokens, words, vectors, tables; a vector is read by its name as a property or in a text). Against what public/gpu.js and public/gpu/*.js read of
//      `plan` and of anything's `.plan`. Both ends by their syntax trees: the worker runs only where there is a GPU.
//      (force and remembered come from the worker and the page, and matrices is keyed by the tensors' names: their
//      insides are not looked at.)
//
//   node tests/plan-keys-check.mjs          (PYTHON=.venv/bin/python; a few seconds, no Pyodide, no kernels)
//
// What it does not see: a key read under a computed name (`plan[name]`), a plan passed on under another name than
// `plan`, and whether a value is what the reader takes it for. Exit 1 on a key with one end only.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "@babel/parser";
import { planOf } from "./plans.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
let failed = 0;
const ok = (line) => console.log(`ok: ${line}`);
const wrong = (line) => { failed++; console.log(`WRONG: ${line}`); };
const sorted = (names) => [...names].sort();
// both directions of one contract: written and read are sets; unread: { key: why nothing reads it } (said, not failed)
function held(what, written, read, unread = {}) {
  const neverWritten = sorted(read).filter((key) => !written.has(key));
  const neverRead = sorted(written).filter((key) => !read.has(key) && !(key in unread));
  const stale = Object.keys(unread).filter((key) => read.has(key) || !written.has(key));
  if (neverWritten.length) wrong(`${what}: read and never written: ${neverWritten.join(", ")}`);
  if (neverRead.length) wrong(`${what}: written and read nowhere: ${neverRead.join(", ")}`);
  if (stale.length) wrong(`${what}: listed as read nowhere, and it is read now or written no more: ${stale.join(", ")}`);
  for (const [key, why] of Object.entries(unread)) if (!stale.includes(key)) console.log(`     (${what}: ${key} is written and read nowhere: ${why})`);
  if (!neverWritten.length && !neverRead.length && !stale.length) ok(`${what}: ${written.size} written, ${read.size} read, each at both ends${Object.keys(unread).length ? ` but ${Object.keys(unread).join(", ")}` : ""}`);
}

// ---- the syntax trees
const treesOf = (files) => files.map((file) => ({ file, program: parse(fs.readFileSync(path.join(root, file), "utf8"), { sourceType: "module" }).program }));
const folder = (window) => [window, ...fs.readdirSync(path.join(root, window.replace(/\.js$/, ""))).filter((file) => file.endsWith(".js")).sort()
  .map((file) => `${window.replace(/\.js$/, "")}/${file}`)];
function walk(node, visit, parent = null) {
  if (!node || typeof node !== "object") return;
  if (Array.isArray(node)) { node.forEach((child) => walk(child, visit, parent)); return; }
  if (node.type) visit(node, parent);
  for (const key of Object.keys(node)) if (!["loc", "start", "end", "extra", "leadingComments", "trailingComments", "innerComments"].includes(key)) walk(node[key], visit, node.type ? node : parent);
}
const member = (node) => (node.type === "MemberExpression" || node.type === "OptionalMemberExpression") && !node.computed && node.property.type === "Identifier";
// the keys an object pattern takes ({ a, b: c, ...rest } takes a and b)
const taken = (pattern) => pattern.properties.filter((p) => p.type === "ObjectProperty" && !p.computed).map((p) => p.key.name ?? p.key.value);
// the keys of an object literal, through its spreads and the branches of a conditional or a logical expression
function literalKeys(node, into = new Set()) {
  if (!node) return into;
  if (node.type === "ObjectExpression") {
    for (const property of node.properties) {
      if (property.type === "SpreadElement") literalKeys(property.argument, into);
      else if (!property.computed) into.add(property.key.name ?? property.key.value);
    }
  } else if (node.type === "ConditionalExpression") { literalKeys(node.consequent, into); literalKeys(node.alternate, into); }
  else if (node.type === "LogicalExpression") { literalKeys(node.left, into); literalKeys(node.right, into); }
  return into;
}
const functionNamed = (trees, name) => {
  let found;
  for (const { program } of trees) walk(program, (node) => { if (node.type === "FunctionDeclaration" && node.id?.name === name) found = node; });
  assert.ok(found, `no function ${name}(): tests/plan-keys-check.mjs reads the plan it returns`);
  return found;
};
// the object literal a function returns (its one return of a literal)
const returned = (fn) => {
  const literals = [];
  walk(fn.body, (node) => { if (node.type === "ReturnStatement" && node.argument?.type === "ObjectExpression") literals.push(node.argument); });
  assert.equal(literals.length, 1, `${fn.id.name}() returns ${literals.length} object literals, not one`);
  return literals[0];
};

// ---- 1. Python -> forward.js
{
  const LINEAR = { every: 4, key_heads: 2, value_heads: 4, key_dim: 64, value_dim: 128, conv: 4 };
  const families = [
    { name: "llama", header: [256, 768, 4, 8, 8, -2048, 320], form: {}, more: { outliers: true, unturned: [1, 3] } },
    { name: "qwen2 and qwen3", header: [256, 768, 4, 4, 2, 2048, 320], form: { bias: true, qk_norm: true, head_dim: 128 }, more: {} },
    { name: "llama, rotated", header: [256, 768, 4, 8, 8, 2048, 320], form: { rotated: true }, more: {} },
    { name: "gpt2", header: [256, 1024, 3, 8, 8, 2048, 320], form: { arch: "gpt2" }, more: {} },
    { name: "neox", header: [256, 1024, 3, 8, 8, -2048, 320], form: { arch: "neox" }, more: { rotary: 8, parallel_residual: true } },
    { name: "qwen35", header: [256, 768, 8, 4, 2, 2048, 320], form: { arch: "qwen35", head_dim: 128, linear: LINEAR, rotated: true }, more: { rotary: 32 } },
    { name: "lfm2", header: [256, 768, 8, 8, 4, -2048, 320], form: { arch: "lfm2", convolution: { layers: "ccaccaca", taps: 3 } }, more: {} },
  ];
  const cases = families.flatMap((family) => ["float32", "float16", "int8", "int6", "ternary"].map((dtype) => ({ ...family, dtype })));
  const plans = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", [path.join(root, "tests/engine_plans.py"), root],
    { input: JSON.stringify(cases), maxBuffer: 1 << 28, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).toString());
  assert.equal(plans.length, cases.length);

  // what Python writes: every model has the same keys (a key that only some had would be undefined for the others)
  const keys = new Set(Object.keys(plans[0])), tensors = new Set(), entryKeys = new Set(), derived = new Set();
  plans.forEach((plan, index) => {
    const which = `${cases[index].name}, ${cases[index].dtype}`;
    assert.deepEqual(sorted(Object.keys(plan)), sorted(keys), `the plan of ${which} has other keys than the plan of ${cases[0].name}, ${cases[0].dtype}`);
    for (const [name, entry] of Object.entries(plan.tensors)) {
      tensors.add(name);
      Object.keys(entry).forEach((key) => entryKeys.add(key));
      assert.deepEqual(sorted(Object.keys(entry)), sorted(Object.keys(Object.values(plans[0].tensors)[0])), `the tensor ${name} of ${which} has other keys than the others`);
    }
    Object.keys(plan.derived).forEach((name) => derived.add(name.replace(/^signs\.\d+$/, "signs.<width>")));
  });
  ok(`Python's plan has the same ${keys.size} keys for ${plans.length} made-up models (${families.length} of the layouts' families in five dtypes), ${tensors.size} tensors among them`);

  // what forward.js reads
  const trees = treesOf(folder("public/forward.js"));
  const readKeys = new Set(), readTensors = new Set(), readEntry = new Set(), readDerived = new Set();
  // (toJs: the PyProxy's own method, where the plan is Python's object still)
  const NOT_KEYS = new Set(["toJs"]);
  // an expression that is a tensor's entry: `t` (the name these files give one), `T.<name>`, `tensors.<name>`, `tensors[...]`
  // and the two names an entry is kept under where it is read (the embedding's and the classifier's tables)
  const isTable = (node) => node.type === "Identifier" && ["T", "tensors"].includes(node.name);
  const isEntry = (node) => (node.type === "Identifier" && ["t", "embedding", "classifier"].includes(node.name)) ||
    ((node.type === "MemberExpression" || node.type === "OptionalMemberExpression") && isTable(node.object));
  for (const { program } of trees) {
    walk(program, (node) => {
      if (member(node) && node.object.type === "Identifier" && node.object.name === "plan" && !NOT_KEYS.has(node.property.name)) readKeys.add(node.property.name);
      if (node.type === "VariableDeclarator" && node.id.type === "ObjectPattern" && node.init?.type === "Identifier" && node.init.name === "plan") taken(node.id).forEach((key) => readKeys.add(key));
      // the tensors' names: matrix("wq"), floats("bq"), T.wo, tensors.wcls, and the lists a model on the GPU alone is cut by
      if (node.type === "CallExpression" && node.callee.type === "Identifier" && ["matrix", "floats"].includes(node.callee.name) && node.arguments[0]?.type === "StringLiteral") {
        readTensors.add(node.arguments[0].value);
        readDerived.add(node.arguments[0].value);
      }
      if (member(node) && isTable(node.object)) readTensors.add(node.property.name);
      if (node.type === "VariableDeclarator" && ["LAYER_MATRICES", "GPU_ALONE"].includes(node.id.name)) {
        walk(node.init, (inner) => { if (inner.type === "StringLiteral") readTensors.add(inner.value); });
      }
      if (member(node) && isEntry(node.object)) readEntry.add(node.property.name);
      if (node.type === "TemplateLiteral" && node.quasis[0].value.cooked === "signs.") readDerived.add("signs.<width>");
    });
  }
  // T359.5: every layer's facts and the widths, which engine/plan.py writes from the layout so that forward.js need not
  // work them out again from arch, linear and convolution; forward.js reads them from T375 (and planOf() has them once
  // it is Python's own plan, T357.7): until then they are written and read by nobody
  const ahead = { layers: "forward.js reads the layers' facts from T375", widths: "forward.js reads the widths from T375" };
  held("the plan's keys, Python to forward.js", keys, readKeys, ahead);
  // (a name forward.js asks for that no model of these has is a tensor of no architecture: floats() answers 0 for it)
  held("the tensors' names, Python to forward.js", tensors, readTensors);
  // (what else is read off those names: a table kept under `embedding` is also the GPU's plan's, { rows, n, ... })
  const entryRead = new Set([...readEntry].filter((key) => entryKeys.has(key) || !["rows", "n", "six", "ternary", "at", "slice", "length"].includes(key)));
  held("a tensor's keys, Python to forward.js", entryKeys, entryRead);
  held("the derived tables' names, Python to forward.js", derived, new Set([...readDerived].filter((name) => derived.has(name) || !tensors.has(name))));

  // tests/plans.mjs's planOf(): the plan memory-check and the net's calls hand createForward(), written by hand
  const standIn = planOf({ header: cases[0].header, form: { arch: "llama", linear: null, convolution: null, rotated: null }, dtype: "int8", head_size: 32,
    tensors: {}, derived: {}, keep_int8: true });
  held("the plan's keys, Python to tests/plans.mjs's planOf()", keys, new Set(Object.keys(standIn)), ahead);
}

// ---- 2. forward.js -> the GPU's worker
{
  const writers = treesOf(folder("public/forward.js"));
  // the plan of { type: "start", memory, plan: { ... } }
  let start;
  for (const { program } of writers) {
    walk(program, (node) => {
      if (node.type !== "ObjectExpression") return;
      const property = (name) => node.properties.find((p) => p.type === "ObjectProperty" && p.key.name === name);
      if (property("type")?.value.value === "start" && property("plan")?.value.type === "ObjectExpression") {
        assert.ok(!start, "two messages of type \"start\" carry a plan");
        start = property("plan").value;
      }
    });
  }
  assert.ok(start, "no { type: \"start\", plan: { ... } } in public/forward/: tests/plan-keys-check.mjs reads the GPU's plan there");
  const open = returned(functionNamed(writers, "gpuOnlyPlan"));
  const written = new Set([...literalKeys(start), ...literalKeys(open)]);
  const inside = {
    tokens: literalKeys(returned(functionNamed(writers, "gpuTokensPlan"))),
    vectors: literalKeys(returned(functionNamed(writers, "gpuVectors"))),
    words: literalKeys(start.properties.find((p) => p.key?.name === "words").value),
    tables: new Set(),
  };
  // (gpuOnlyPlan()'s tables: `const tables = { classifier, embedding }`)
  walk(functionNamed(writers, "gpuOnlyPlan").body, (node) => { if (node.type === "VariableDeclarator" && node.id.name === "tables") literalKeys(node.init, inside.tables); });
  for (const [name, keys] of Object.entries(inside)) assert.ok(keys.size, `nothing found of what the GPU's plan holds in ${name}`);

  const readers = treesOf(folder("public/gpu.js"));
  const read = new Set(), readInside = Object.fromEntries(Object.keys(inside).map((name) => [name, new Set()]));
  // a plan: `plan`, or anything's `.plan` (m.plan, common.model.plan)
  const isPlan = (node) => (node.type === "Identifier" && node.name === "plan") || (member(node) && node.property.name === "plan");
  // one level down: `<plan>.tokens`, `<plan>.vectors`, ... (which of them, or undefined)
  const insideOf = (node) => (member(node) && isPlan(node.object) && node.property.name in inside ? node.property.name : undefined);
  // (`plan.tables ?? plan.tokens`: either)
  const insidesOf = (node) => (node.type === "LogicalExpression" ? [...insidesOf(node.left), ...insidesOf(node.right)] : insideOf(node) ? [insideOf(node)] : []);
  for (const { program } of readers) {
    walk(program, (node) => {
      if (member(node) && isPlan(node.object)) read.add(node.property.name);
      if (member(node) && insideOf(node.object)) readInside[insideOf(node.object)].add(node.property.name);
      // the vectors go up as a list (m.vectors, by the plan's own names) and are read back as `V.<name>`, `m.vectors.<name>`
      // or by a name in a text (added(to, "bq")): a property is a read, and so is a text that is one of the written names
      if (member(node) && ((node.object.type === "Identifier" && node.object.name === "V") || (member(node.object) && node.object.property.name === "vectors"))) readInside.vectors.add(node.property.name);
      if (node.type === "StringLiteral" && inside.vectors.has(node.value)) readInside.vectors.add(node.value);
      if (node.type === "VariableDeclarator" && node.id.type === "ObjectPattern" && node.init) {
        if (isPlan(node.init)) taken(node.id).forEach((key) => read.add(key));
        for (const name of insidesOf(node.init)) taken(node.id).forEach((key) => readInside[name].add(key));
      }
    });
  }
  held("the GPU's plan, forward.js to the GPU's worker", written, read);
  // (tables and tokens are read through one expression, `plan.tables ?? plan.tokens`: a key of either is a key of both to the reader)
  const tables = new Set([...readInside.tables].filter((key) => inside.tables.has(key)));
  const tokens = new Set([...readInside.tokens].filter((key) => inside.tokens.has(key) || !inside.tables.has(key)));
  held("the GPU's plan's tokens", inside.tokens, tokens);
  held("the GPU's plan's tables (a model on the GPU alone)", inside.tables, tables);
  held("the GPU's plan's words", inside.words, readInside.words);
  // (what is read off a vector itself, { at, size }, is no name of one)
  held("the GPU's plan's vectors", inside.vectors, new Set([...readInside.vectors].filter((key) => !["at", "size"].includes(key))));
}

console.log(failed ? `plan-keys-check: FAILED (${failed})` : "plan-keys-check: ok");
process.exit(failed ? 1 : 0);
