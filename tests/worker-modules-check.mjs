// tests/worker-modules-check.mjs (T350): what the type check does not hold of src/runtime/worker.ts and its modules
// (src/runtime/worker/*.ts). Node only, under a second:
//
//   node tests/worker-modules-check.mjs
//
// T368.8: the worker is TypeScript, and tsc (tests/types.mjs) says what this check said before of a wrong name: a name
// used and neither declared nor imported (with the one rule tsc cannot hold, an ambient name: tests/types.mjs's own),
// a name imported and assigned (TS2632; the build refuses it too), a key or a kind of a message that its type has not.
// What is left here is what tsc, the build and Node all let pass, each with why:
//   - a name declared inside a function that the file also imports: the lines below it read the local, and the types
//     may well agree (T367.4's pitfall: `let unstarted` beside the imported unstarted);
//   - a name declared at the top of two modules: two variables to tsc and in a browser, one in the one context the
//     checks run the worker in (tests/worker-source.mjs: a `let` twice is a SyntaxError there, a function twice is the
//     second one, silently);
//   - worker/state.ts, the one object the modules shared, back (T378.3: what they hold has owners);
//   - what a type of src/runtime/contracts/messages.ts has and nobody writes or reads (a kind never posted, an optional
//     key no message has, a key of a load that the worker never reads), and its two ends that are Python's (the engine's
//     stats, the settings of an answer): to tsc a type with more than is used is a correct type;
//   - a message posted without `satisfies`: the worker's own postMessage() takes anything, and tsc holds a message to
//     its type only where the message says which;
//   - that the modules link in Node, and that the receiver worker.ts sets as it loads answers.
// The ring of imports and a file of worker/ that nothing reaches are tests/worker-source.mjs's (workerScripts() throws:
// the build links a ring without a word).
// What it does not see: a browser's own way with a module worker's first message (the browsers do: tests/e2e.mjs), and
// an await before the receiver is set (the import below waits for the module whole: tests/worker-check.mjs's one
// script cannot hold a top-level await, and falls).
import assert from "node:assert/strict";
import { parse } from "@babel/parser";
import { treeOf, python } from "./tree.mjs";
import { workerScripts } from "./worker-source.mjs";
import { contract, differing, givenTo, handedTo, posted as postedBy, readOff, returnedBy, takenBy, told, treeOf as syntaxOf, walk } from "./contracts.mjs";
import fs from "node:fs";
import { namesOf } from "./names.mjs";
import { untyped } from "./untyped.mjs";

// (T368.3: each file as the JavaScript it is once its types are blanked, tests/untyped.mjs: a type's name is no name used)
const plain = (url) => untyped(fs.readFileSync(url, "utf8"), url.pathname);
// (a module of the folder that the window does not reach, and a ring: workerScripts() throws)
const files = new Map(workerScripts().map(({ name, url }) => [name, namesOf(plain(url))]));

// ---- T378.3: what the modules hold has owners: a module's own `let`, which the others read by the name they import
// (a live binding) and never set (tsc: an import cannot be assigned). Two mistakes that tsc does not say:
let owned = 0;
for (const [name, { imported, inner }] of files) {
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
console.log(`worker-modules-check: ${files.size} files, ${lets.length} variables at their tops; ${owned} names imported, none hidden by a local; no top-level name twice`);

// ---- T368.2: what the worker and a page say to one another is src/runtime/contracts/messages.ts's. Both ends are
// TypeScript (the worker from T368.3), and tsc holds what each writes and reads to those types: a kind or a key that
// the type has not is its error (T368.8: broken both ways and seen). What tsc does not hold is the other direction,
// and that is what the comparisons below are kept for: the types have nothing that nobody writes or reads. By the
// syntax trees: the kinds the worker posts and the kinds it tells apart are the types', and so are the keys of every
// message it posts. A key that is in no literal is followed to where it is written (the engine's stats in Python,
// convert()'s and timedPaths()'s results, the threads' verdict, what the GPU chose); the type has each of those, and
// nothing that none of them writes.
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
