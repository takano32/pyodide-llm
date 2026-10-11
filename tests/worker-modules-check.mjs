// tests/worker-modules-check.mjs (T350): the names that src/runtime/worker.js and its modules (src/runtime/worker/*.js) use.
//
// The checks that run the worker in Node make one context's scripts of these files (tests/worker-source.mjs), where
// every module's top level is a global: a module that uses a name of another without importing it works there, and is
// a ReferenceError in a browser, at the moment the line runs. So this reads each file as the module it is and holds
// every name it uses to one of: declared in the file, imported by it, or one of the worker's own globals listed below
// (that what is imported is exported there is the build's and Node's to say: MISSING_EXPORT, a SyntaxError: T367.5). Node only, under a second:
//
//   node tests/worker-modules-check.mjs
//
// The parser is @babel/parser, which Astro's packages bring (npm ci installs it; this repository does not name it).
// It also holds what the modules keep to its owner (T378.3): no imported name assigned or hidden, no top-level name twice.
// And it imports worker.js for real (Node links the modules): its receiver is set as it loads, and answers.
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

/** { declared: the file's top-level names, free: the names it uses and does not declare, imported: the names it takes
 * from other files, set: the top-level names it assigns, inner: the names it declares below its top level, kinds: how
 * each top-level name is declared ("let", "const", "function", "class", "import") } */
export function namesOf(source) {
  const program = parse(source, { sourceType: "module" }).program;
  const scopes = [], free = new Set(), set = new Set(), inner = new Set();
  // (an assignment's target that no scope below the file's top level declares is one of the file's top level, or free)
  const assigned = (target) => {
    if (!target || target.type === "MemberExpression" || target.type === "OptionalMemberExpression") return;
    for (const name of bound(target)) if (!scopes.slice(1).some((scope) => scope.has(name))) set.add(name);
  };
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
    else if (node.type === "MemberExpression") inside.push(node);  // (an assignment's pattern: [a.b, c] = ...)
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
    if (scopes.length) names.forEach((name) => inner.add(name));
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
      case "AssignmentExpression":
        assigned(node.left);
        if (node.left.type === "Identifier" || node.left.type.endsWith("Pattern")) {
          const target = pattern(node.left);
          target.names.forEach((name) => !known(name) && free.add(name));
          target.read();
        } else walk(node.left);
        return walk(node.right);
      case "UpdateExpression":
        assigned(node.argument);
        return walk(node.argument);
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
        if (node.left && node.left.type !== "VariableDeclaration") assigned(node.left);
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
  const kinds = new Map(program.body.flatMap((statement) => {
    const s = statement.type === "ExportNamedDeclaration" && statement.declaration ? statement.declaration : statement;
    const kind = s.type === "VariableDeclaration" ? s.kind : s.type === "ImportDeclaration" ? "import" : s.type === "FunctionDeclaration" ? "function" : "class";
    return declarations([statement]).map((name) => [name, kind]);
  }));
  const imported = [...kinds].filter(([, kind]) => kind === "import").map(([name]) => name);
  return { declared, free, imported, set, inner, kinds };
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
  // (a module of the folder that the window does not reach, and a ring: workerScripts() threw)
  console.log(`worker-modules-check: ${files.size} files of the worker use ${count} names of the worker's globals, and nothing they do not declare or import`);

  // ---- T378.3: what the modules hold has owners: a module's own `let`, which the others read by the name they import
  // (a live binding) and never set. The scripts of one context that the checks run (worker-source.mjs) take the imports
  // out, and every top level is the context's there, so three mistakes pass those checks and fail in a browser, or the
  // other way round:
  //   - a module that assigns a name it imports: it sets the owner's variable in the one context, and is a TypeError in a
  //     browser ("Assignment to constant variable"), at the moment the line runs;
  //   - a name declared inside a function that is also one the file imports: the line below it reads the local, without
  //     a word (T367.4's pitfall: `let unstarted` beside the imported unstarted);
  //   - two modules that declare the same top-level name: two variables in a browser, one in the context (a `let` twice
  //     is a SyntaxError there; a function twice is the second one, silently).
  let owned = 0;
  for (const [name, { imported, set, inner }] of files) {
    const written = imported.filter((taken) => set.has(taken));
    assert.deepEqual(written, [], `${name} assigns ${written.join(", ")}, which it imports: only the module that declares a variable sets it ` +
      "(a TypeError in a browser; say what happened to a function of the owner instead)");
    const hidden = imported.filter((taken) => inner.has(taken));
    assert.deepEqual(hidden, [], `${name} declares ${hidden.join(", ")} inside a function, and imports the same name: the lines below read the local one`);
    owned += imported.length;
  }
  const owners = new Map();
  for (const [name, { kinds }] of files) {
    for (const [declared, kind] of kinds) {
      if (kind === "import") continue;
      assert.ok(!owners.has(declared), `${declared} is declared at the top of both ${owners.get(declared)} and ${name}: one name in the checks' one context, two in a browser`);
      owners.set(declared, name);
    }
  }
  // (and no module shares an object for the others to set fields of, as worker/state.js's was: what T378.3 took apart)
  assert.ok(!files.has("state"), "worker/state.js is back: what the modules hold has owners (runtime.js, request.js, loaded.js, and each module's own)");
  const lets = [...files].flatMap(([name, { kinds }]) => [...kinds].filter(([, kind]) => kind === "let").map(([declared]) => `${name}: ${declared}`));
  console.log(`worker-modules-check: ${lets.length} variables at the top of the worker's modules, each set by its own module alone; ` +
    `${owned} names imported, none assigned or hidden by a local; no top-level name twice`);

  // ---- the files as the modules they are, linked by Node: a name imported that is not exported where it is taken
  // from is a SyntaxError here. worker.js awaits nothing before its receiver is set (a module worker's port opens at the
  // module's first await, and a message that finds no onmessage then is lost: AGENTS.md, helper.js, T109). A "generate"
  // before any model: the worker's own handler answers "The model is not ready."
  const posted = [];
  Object.assign(globalThis, { self: globalThis, postMessage: (message) => posted.push(message) });
  globalThis.location = { search: "" };
  await import(treeOf().runtimeUrl("worker.js"));
  assert.equal(typeof self.onmessage, "function", "worker.js set no onmessage as it loaded");
  self.onmessage({ data: { type: "generate", prompt: "early" } });
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.deepEqual(posted.map((message) => [message.type, message.message]), [["error", "The model is not ready."]],
    "a message given to the receiver worker.js set as it loaded was not handled once, by the worker's own handler");
  console.log("worker-modules-check: the modules link, and the receiver set as they load answers");
}
