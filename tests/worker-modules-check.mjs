// tests/worker-modules-check.mjs (T350): the names that public/worker.js and its modules (public/worker/*.js) use.
//
// The checks that run the worker in Node make one context's scripts of these files (tests/worker-source.mjs), where
// every module's top level is a global: a module that uses a name of another without importing it works there, and is
// a ReferenceError in a browser, at the moment the line runs. So this reads each file as the module it is and holds
// every name it uses to one of: declared in the file, imported by it (and exported where it is imported from), or one
// of the worker's own globals listed below. Node only, under a second:
//
//   node tests/worker-modules-check.mjs
//
// The parser is @babel/parser, which Astro's packages bring (npm ci installs it; this repository does not name it).
// It also holds every state.<name> to a field of worker/state.js's object.
// And it imports worker.js for real (Node links the modules), and plays a message that comes at its first await.
// What it does not see: a browser's own way with a module worker's first message (the browsers do: tests/e2e.mjs).
import assert from "node:assert/strict";
import { parse } from "@babel/parser";
import { treeOf } from "./tree.mjs";
import { workerScripts } from "./worker-source.mjs";
import fs from "node:fs";

// what a module worker has without declaring it, as far as these files use it
const GLOBALS = new Set(("AbortController Array Boolean DataView Date Error Int32Array JSON Map Math MessageChannel Number Object " +
  "PerformanceObserver Promise ReadableStream Response Set SharedArrayBuffer String Symbol TextDecoder TransformStream URL URLSearchParams WeakMap " +
  "Uint8Array Worker WritableStream clearInterval clearTimeout console fetch globalThis navigator performance postMessage self setInterval " +
  "setTimeout undefined").split(" "));

/** { declared: the file's top-level names, free: the names it uses and does not declare } */
export function namesOf(source) {
  const program = parse(source, { sourceType: "module" }).program;
  const scopes = [], free = new Set();
  const known = (name) => scopes.some((scope) => scope.has(name));
  // the names a pattern binds, and the expressions inside it (defaults, computed keys), which are read
  const bound = (node, names = [], inside = []) => {
    if (!node) return names;
    if (node.type === "Identifier") names.push(node.name);
    else if (node.type === "ObjectPattern") {
      for (const property of node.properties) {
        if (property.type !== "RestElement" && property.computed) inside.push(property.key);
        bound(property.type === "RestElement" ? property.argument : property.value, names, inside);
      }
    } else if (node.type === "ArrayPattern") node.elements.forEach((element) => bound(element, names, inside));
    else if (node.type === "AssignmentPattern") {
      inside.push(node.right);
      bound(node.left, names, inside);
    } else if (node.type === "RestElement") bound(node.argument, names, inside);
    else throw new Error(`a pattern this check does not know: ${node.type}`);
    return names;
  };
  const declarations = (statements) => statements.flatMap((statement) => {
    const s = statement.type === "ExportNamedDeclaration" && statement.declaration ? statement.declaration : statement;
    if (s.type === "VariableDeclaration") {
      assert.notEqual(s.kind, "var", "var: this check knows let and const");
      return s.declarations.flatMap((d) => bound(d.id));
    }
    if (s.type === "ImportDeclaration") return s.specifiers.map((specifier) => specifier.local.name);
    return s.type === "FunctionDeclaration" || s.type === "ClassDeclaration" ? [s.id.name] : [];
  });
  const scoped = (names, inside) => {
    scopes.push(new Set(names));
    inside();
    scopes.pop();
  };
  const block = (statements) => scoped(declarations(statements), () => statements.forEach((statement) => walk(statement)));
  const pattern = (node) => {
    const inside = [];
    const names = bound(node, [], inside);
    return { names, read: () => inside.forEach((expression) => walk(expression)) };
  };
  const callable = (node) => {
    const parameters = node.params.map(pattern);
    scoped([...parameters.flatMap((p) => p.names), "arguments", ...(node.type === "FunctionExpression" && node.id ? [node.id.name] : [])], () => {
      parameters.forEach((p) => p.read());
      if (node.body.type === "BlockStatement") block(node.body.body);
      else walk(node.body);
    });
  };
  function walk(node) {
    if (!node || typeof node.type !== "string") return;
    switch (node.type) {
      case "Identifier":
        if (!known(node.name)) free.add(node.name);
        return;
      case "MemberExpression": case "OptionalMemberExpression":
        walk(node.object);
        if (node.computed) walk(node.property);
        return;
      case "ObjectProperty": case "ClassProperty":
        if (node.computed) walk(node.key);
        return walk(node.value);
      case "ObjectMethod": case "ClassMethod": case "ClassPrivateMethod":
        if (node.computed) walk(node.key);
        return callable(node);
      case "FunctionDeclaration": case "FunctionExpression": case "ArrowFunctionExpression":
        return callable(node);
      case "ClassDeclaration": case "ClassExpression":
        return scoped(node.id ? [node.id.name] : [], () => {
          walk(node.superClass);
          node.body.body.forEach((member) => walk(member));
        });
      case "BlockStatement": case "StaticBlock":
        return block(node.body);
      case "VariableDeclaration":
        for (const d of node.declarations) {
          pattern(d.id).read();
          walk(d.init);
        }
        return;
      case "ForStatement": case "ForInStatement": case "ForOfStatement": {
        const head = node.init ?? node.left;
        return scoped(head?.type === "VariableDeclaration" ? head.declarations.flatMap((d) => bound(d.id)) : [],
          () => [node.init, node.left, node.right, node.test, node.update, node.body].forEach((part) => walk(part)));
      }
      case "CatchClause": {
        const caught = node.param ? pattern(node.param) : { names: [], read() {} };
        return scoped(caught.names, () => {
          caught.read();
          block(node.body.body);
        });
      }
      case "SwitchStatement":
        walk(node.discriminant);
        return scoped(declarations(node.cases.flatMap((c) => c.consequent)), () => node.cases.forEach((c) => {
          walk(c.test);
          c.consequent.forEach((statement) => walk(statement));
        }));
      case "LabeledStatement":
        return walk(node.body);
      case "BreakStatement": case "ContinueStatement": case "MetaProperty": case "ImportDeclaration":
        return;
      case "ExportNamedDeclaration":
        // (T349: `export { a, b };` uses the file's own a and b, tests/forward-modules-check.mjs)
        if (!node.declaration && !node.source) return node.specifiers.forEach((specifier) => walk(specifier.local));
        assert.ok(node.declaration, "an export from another file: this check knows `export const`, `export function` and `export { a }`");
        return walk(node.declaration);
      default:
        for (const [key, value] of Object.entries(node)) {
          if (key === "loc" || key.endsWith("Comments") || !value || typeof value !== "object") continue;
          (Array.isArray(value) ? value : [value]).forEach((child) => walk(child));
        }
    }
  }
  const declared = declarations(program.body);
  scoped(declared, () => program.body.forEach((statement) => walk(statement)));
  return { declared, free };
}

if (import.meta.url === new URL(process.argv[1], "file:").href) {
  const files = new Map(workerScripts().map(({ name, url }) => [name, namesOf(fs.readFileSync(url, "utf8"))]));
  let count = 0;
  for (const [name, { free }] of files) {
    const unknown = [...free].filter((used) => !GLOBALS.has(used));
    const whose = (used) => [...files].filter(([, other]) => other.declared.includes(used)).map(([other]) => other);
    assert.deepEqual(unknown, [], `${name} uses ${unknown.map((used) => `${used}${whose(used).length ? ` (declared in ${whose(used)}: not imported)` : ""}`).join(", ")}, ` +
      "which it neither declares nor imports: a ReferenceError in a browser");
    count += free.size;
  }
  // (an import of a name the other file does not export, and a module of the folder nobody asks for: workerScripts() threw)
  console.log(`worker-modules-check: ${files.size} files of the worker use ${count} names of the worker's globals, and nothing they do not declare or import`);

  // ---- the fields of the one object the modules share (worker/state.js): state.<name> is no variable, and a name
  // that is not a field reads undefined and is set without a word. Every one used is a field, and every field is used
  const texts = workerScripts().map(({ name, url }) => [name, fs.readFileSync(url, "utf8")]);
  const fields = /^export const state = \{\n([\s\S]*?)^\};$/m.exec(texts.find(([name]) => name === "state")[1])[1]
    .split("\n").filter((line) => !line.trim().startsWith("//")).flatMap((line) => [...line.matchAll(/(?:^  |, )(\w+): /g)].map((match) => match[1]));
  assert.equal(new Set(fields).size, fields.length, "a field of state twice");
  const used = new Set();
  for (const [name, text] of texts) {
    for (const [, field] of text.matchAll(/\bstate\.(?!js\b)(\w+)/g)) {
      assert.ok(fields.includes(field), `${name} says state.${field}, which is no field of worker/state.js's object`);
      used.add(field);
    }
  }
  assert.deepEqual(fields.filter((field) => !used.has(field)), [], "fields of state that nothing reads or sets");
  // (and what the checks that run the worker set in its context: a name that is no field would set nothing the worker reads)
  for (const check of ["worker-check.mjs", "worker-sink-check.mjs", "worker-fetches-check.mjs"]) {
    for (const [, field] of fs.readFileSync(new URL(check, import.meta.url), "utf8").matchAll(/\bstate\.(\w+)/g)) {
      assert.ok(fields.includes(field), `tests/${check} says state.${field}, which is no field of worker/state.js's object`);
    }
  }
  console.log(`worker-modules-check: the ${fields.length} fields of state are the ones the modules use`);

  // ---- the files as the modules they are, linked by Node: worker.js waits for its modules at its top level, and a
  // message that comes meanwhile is handled once they are there. A module worker's port opens at the module's first
  // await, and a message that finds no onmessage then is lost (AGENTS.md, helper.js: T109): played here by a message
  // that is given to whatever self.onmessage is in the microtask after worker.js first read its own URL's search,
  // which it does to ask for its modules, before its first await. A "generate" before any model: the worker's own
  // handler answers "The model is not ready."
  const posted = [];
  let delivered = false, handlerThen;
  Object.assign(globalThis, { self: globalThis, postMessage: (message) => posted.push(message) });
  globalThis.location = {
    get search() {
      if (!delivered) {
        delivered = true;
        queueMicrotask(() => {
          handlerThen = self.onmessage;
          self.onmessage?.({ data: { type: "generate", prompt: "early" } });
        });
      }
      return "";
    },
  };
  const tree = treeOf();
  await import(tree.runtimeUrl("worker.js"));
  await new Promise((resolve) => setTimeout(resolve, 50));
  if (tree.bundled) {
    // (T367.1: where a bundler links the files, the worker awaits no module: its handler is set before anything can
    // come, and the message is given to it now)
    assert.equal(typeof self.onmessage, "function", "worker.js set no onmessage as it loaded");
    if (!delivered) self.onmessage({ data: { type: "generate", prompt: "early" } });
    await new Promise((resolve) => setTimeout(resolve, 50));
  } else {
    assert.ok(delivered, "worker.js did not read its URL's search: how does it ask for its modules?");
    assert.equal(typeof handlerThen, "function", "worker.js had no onmessage at its first await: a message that comes then is lost in a browser");
  }
  assert.deepEqual(posted.map((message) => [message.type, message.message]), [["error", "The model is not ready."]],
    "a message that came while the worker's modules were fetched was not handled once, by the worker's own handler");
  console.log("worker-modules-check: the modules link, and a message that comes while they are fetched is handled after them");
}
