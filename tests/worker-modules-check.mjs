// tests/worker-modules-check.mjs (T350): the names that src/runtime/worker.ts and its modules (src/runtime/worker/*.ts) use.
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
import { treeOf, python } from "./tree.mjs";
import { workerScripts } from "./worker-source.mjs";
import { contract, differing, givenTo, handedTo, posted as postedBy, readOff, returnedBy, takenBy, told, treeOf as syntaxOf, walk } from "./contracts.mjs";
import fs from "node:fs";
import { untyped } from "./untyped.mjs";

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
  // (T368.3: each file as the JavaScript it is once its types are blanked, tests/untyped.mjs: a type's name is no name used)
  const plain = (url) => untyped(fs.readFileSync(url, "utf8"), url.pathname);
  const files = new Map(workerScripts().map(({ name, url }) => [name, namesOf(plain(url))]));
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

  // ---- T368.2: what the worker and a page say to one another is src/runtime/contracts/messages.ts's. The pages are
  // TypeScript, which tsc holds to those types; the worker is JavaScript still, which nothing reads for its shapes. So
  // here, by the syntax trees: the kinds the worker posts and the kinds it tells apart are the types', and so are the
  // keys of every message it posts. A key that is in no literal is followed to where it is written (the engine's
  // stats in Python, convert()'s and timedPaths()'s results, the threads' verdict, what the GPU chose); the type has
  // each of those, and nothing that none of them writes.
  // What this does not see: the values (a number where the type says a text), the keys inside a row (a round's, a
  // path's), a message posted under a computed type, and src/pages/index.astro's own first load (no .ts: tsc does not
  // read it; it is the same literal as src/page/choose.ts's, which tsc does).
  {
    const typed = contract("messages.ts");
    const trees = workerScripts().map(({ name, url }) => ({ file: name, program: parse(plain(url), { sourceType: "module" }).program }));
    const only = (...names) => trees.filter(({ file }) => names.includes(file));
    const forward = (file) => [syntaxOf(`forward/${file}`)];
    const same = (what, declared, written, more = {}) => {
      const lines = differing(what, declared, written, more);
      assert.deepEqual(lines, [], `${lines.join("; ")}: src/runtime/contracts/messages.ts and the worker say different things`);
    };
    const from = (keys, why) => Object.fromEntries([...keys].map((key) => [key, why]));
    // the engine's stats, which "done" spreads in: Python's dict (src/python/engine/generation.py)
    const generation = fs.readFileSync(python("engine/generation.py"), "utf8");
    const stats = new Set([.../model\.stats = \{([^}]*)\}/.exec(generation)[1].matchAll(/^\s*"(\w+)":/gm)].map(([, key]) => key));
    same("contracts/messages.ts's Stats, to the stats of Python's generate()", typed.keysOf("Stats"), stats);
    // the settings of an answer, which "generate" hands to Python as they are: Sampling's fields and generate()'s own
    const sampler = fs.readFileSync(python("engine/sampler.py"), "utf8");
    const sampling = /^class Sampling\b[^]*?(?=^\S)/m.exec(sampler)[0];
    const fields = new Set([...sampling.matchAll(/^    (\w+): \w+ = /gm)].map(([, key]) => key));
    assert.ok(fields.has("temperature") && fields.size >= 6, `Sampling's fields were not read out of engine/sampler.py (${[...fields]})`);
    const own = new Set(/^def generate\(model, ([^)]*)\)/m.exec(generation)[1].split(",").map((one) => one.trim().split("=")[0]).filter((one) => !one.startsWith("*")));
    assert.deepEqual([...own].sort(), ["echo", "prompt", "seed", "steps"], "generate()'s own arguments are others now: say here which of them a page sends");
    same("contracts/messages.ts's Sampled, to Sampling's fields and generate()'s steps and seed", typed.keysOf("Sampled"), new Set([...fields, "steps", "seed"]));

    // what the worker posts
    const declared = typed.membersOf("FromWorker"), sent = postedBy(trees, "owner");
    same("contracts/messages.ts's FromWorker, to the kinds the worker posts", new Set(declared.keys()), new Set(sent.keys()));
    const elsewhere = {
      done: from(stats, "the engine's stats"),
      ready: from(returnedBy(only("convert"), "convert"), "worker/convert.js's convert()"),
      paths: from(returnedBy(only("timing"), "timedPaths"), "worker/timing.js's timedPaths()"),
      "threads-compared": from(handedTo(forward("threads.js"), "onCompared"), "forward/threads.js's verdict"),
      gpu: from(givenTo(forward("gpuside.js"), "gpuChosen"), "forward/gpuside.js's gpuChosen"),
    };
    for (const [kind, keys] of declared) {
      // (a key that a literal has and another source has too is the literal's)
      const more = Object.fromEntries(Object.entries(elsewhere[kind] ?? {}).filter(([key]) => !sent.get(kind).keys.has(key)));
      assert.equal(Boolean(sent.get(kind).spreads), Boolean(elsewhere[kind]), `the worker's "${kind}" ${sent.get(kind).spreads ? "spreads something in that this check does not follow" : "spreads nothing in any more"}`);
      same(`contracts/messages.ts's FromWorker, the keys of "${kind}"`, keys, sent.get(kind).keys, more);
    }

    // T368.3: the worker is TypeScript, and tsc holds a message it posts to the type where the message says which:
    // postMessage({ … } satisfies <a member of FromWorker>). The worker's own postMessage() takes anything, so a
    // message without that is held by nothing but the keys above: every one says it (one is cast, and says why:
    // load.ts's "gpu", whose spread tsc reads otherwise).
    const members = new Set(["FromWorker", ...typed.names]);
    let held = 0;
    for (const { name, url } of workerScripts()) {
      const program = parse(fs.readFileSync(url, "utf8"), { sourceType: "module", plugins: ["typescript"] }).program;
      walk(program, (node) => {
        if (node.type !== "CallExpression" || node.callee.type !== "Identifier" || node.callee.name !== "postMessage") return;
        const [message] = node.arguments, said = message?.typeAnnotation;
        const names = !said ? [] : said.type === "TSUnionType" ? said.types.map((one) => one.typeName?.name) : [said.typeName?.name];
        assert.ok(["TSSatisfiesExpression", "TSAsExpression"].includes(message?.type) && names.length && names.every((one) => members.has(one)),
          `${name}, line ${node.loc.start.line}: a message is posted without \`satisfies\` a type of contracts/messages.ts: nothing holds its values to what the page reads`);
        held += 1;
      });
    }
    assert.ok(held >= declared.size, `${held} messages posted were found, fewer than the ${declared.size} kinds`);

    // what the worker is sent: the kinds its receiver tells apart, and the keys it reads off a message (worker.js's
    // receiver, request.js's loadAsked(), and what generate() and timedPaths() take of the message they are handed)
    const asked = typed.membersOf("ToWorker");
    same("contracts/messages.ts's ToWorker, to the kinds worker.js tells apart", new Set(asked.keys()), told(only("worker.ts")));
    const read = new Set([...readOff(only("worker.ts", "request"), "data"), ...takenBy(only("worker.ts"), "generate"), ...takenBy(only("timing"), "timedPaths")]);
    read.delete("type");
    const keys = new Set([...asked.values()].flatMap((one) => [...one]));
    // (steps is the benchmark's too, which the worker reads; the rest of an answer's settings are Python's to read)
    const handedOn = from([...typed.keysOf("Sampled"), "echo"].filter((key) => !read.has(key)), "handed to Python's generate() as it is");
    same("contracts/messages.ts's ToWorker, to the keys the worker reads off a message", keys, read, handedOn);
    console.log(`worker-modules-check: the worker posts the ${declared.size} kinds of contracts/messages.ts's FromWorker with their keys, ` +
      `and reads the ${asked.size} kinds of ToWorker and their ${keys.size} keys (${Object.keys(handedOn).length} of them Python's); each of the ${held} messages it posts says its type to tsc`);
  }

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
