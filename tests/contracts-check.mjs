// tests/contracts-check.mjs (T368.2): the reader of the contracts (tests/contracts.mjs) on made-up texts, and what the
// files of src/runtime/contracts/ have to be for nothing of them to reach a browser.
//
//   node tests/contracts-check.mjs          (light suite; Python is not run, nothing is built; under a second)
//
// The contracts themselves are held to what is written across their lines elsewhere: plan-keys-check.mjs (the plan,
// the GPU's protocol), worker-modules-check.mjs (the messages), worker-sink-check.mjs (the form), and the conduct's
// here (it needs nothing but three texts). Here too: that the reader they share reads a type as it says it does, and
// refuses what it does not read rather than answering with fewer keys; that every file of contracts/ holds types alone;
// and that whatever reads one says `import type` (the file is then in no bundle: tests/dist-check.mjs counts the
// sources that are).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { parse } from "@babel/parser";
import { fileURLToPath, pathToFileURL } from "node:url";
import { contract, contractOf, differing, givenTo, handedTo, literalKeys, posted, readOff, returnedBy, takenBy, told, treeOf as syntaxOf, walk } from "./contracts.mjs";
import { python, runtime } from "./tree.mjs";

const sorted = (set) => [...set].sort();
const made = contractOf(`
import type { Elsewhere } from "./elsewhere.ts";
type Base = { load: number; root?: string };
export type Row = { a: number } | { a: number; b: string };
export type Whole = { one: number; two?: string; never?: undefined; inner: Inner | null; "quoted-key": 1; method(a: number): void; [signs: \`signs.\${number}\`]: Uint8Array };
export type Inner = { x: 1 };
export type Both = Base & { more: boolean };
export type Names = "a" | "b-c";
export type One = { type: "one"; load?: undefined; text: string };
export type Sent = One | ({ type: "two" } & Base) | { type: "three"; a: 1 } | { type: "three"; b?: 2 };
export type Asked = ["text", where: string, name: string] | ["done", unknown] | Stream;
type Stream = ["stream", string, number, number];
export type Listed = Record<string, number>;
`, "made-up.ts");
assert.deepEqual(sorted(made.keysOf("Whole")), ["inner", "method", "one", "quoted-key", "signs.<number>", "two"], "an object type's keys: not the one that can only be undefined");
assert.deepEqual(sorted(made.keysOf("Both")), ["load", "more", "root"], "an intersection's keys, through an alias");
assert.deepEqual(sorted(made.keysOf("Row")), ["a", "b"], "a union's keys are those of any of it");
assert.deepEqual(sorted(made.literalsOf("Names")), ["a", "b-c"]);
assert.deepEqual([...made.membersOf("Sent")].map(([tag, keys]) => [tag, sorted(keys)]), [["one", ["text"]], ["two", ["load", "root"]], ["three", ["a", "b"]]],
  "a union's members by their tag: two of one tag together, the tag and a key that is never there left out");
assert.deepEqual([...made.tuplesOf("Asked")], [["text", 3], ["done", 2], ["stream", 4]]);
for (const [what, read] of [["a type that is not there", () => made.keysOf("Nowhere")], ["a type this file imports", () => made.keysOf("Elsewhere")],
  ["a Record's keys", () => made.keysOf("Listed")], ["the texts of an object type", () => made.literalsOf("Inner")],
  ["the members of a union without a tag", () => made.membersOf("Row")], ["the tuples of an object type", () => made.tuplesOf("Inner")]]) {
  assert.throws(read, Error, `${what} was answered, not refused`);
}
assert.throws(() => contractOf("export const FORM = { a: 1 };\nexport type A = { a: number };", "a value"), /holds types alone/);
assert.throws(() => contractOf("export function f() {}", "a function"), /holds types alone/);

// what JavaScript writes and reads of a message
const tree = (text) => [{ file: "made-up.js", program: parse(text, { sourceType: "module" }).program }];
const worker = tree(`
  function result() { const inner = () => { return { hidden: 1 }; }; if (x) return { early: 1 }; return { late: 2, ...(y ? { maybe: 1 } : {}) }; }
  function taking({ type, prompt, ...rest }, other) {}
  let chosen; chosen = { a: 1 }; chosen.b = 2; other.c = 3;
  onCompared?.({ best, faster });
  postMessage({ type: "ready", load, ...(memory && { heap }), ...(engine.said ?? {}), ...result() });
  postMessage({ type: "ready", late: true });
  made.postMessage({ type: "start", plan });
  postMessage(notALiteral);
  self.onmessage = ({ data }) => { if (data.type === "load" || "stop" === data.type) use(data.model, data?.load); else if (other.type === "no") 0; };
`);
const both = posted(worker);
assert.deepEqual(sorted(both.keys()), ["ready", "start"]);
assert.deepEqual([sorted(both.get("ready").keys), both.get("ready").spreads], [["heap", "late", "load"], 2], "a message's keys through the spreads that are literals; the others counted");
assert.deepEqual(sorted(posted(worker, "owner").keys()), ["ready"], "a worker's own postMessage()");
assert.deepEqual(sorted(posted(worker, "worker").keys()), ["start"], "what is posted to a worker");
assert.deepEqual(sorted(told(worker)), ["load", "stop"]);
assert.deepEqual(sorted(readOff(worker, "data")), ["load", "model", "type"]);
assert.deepEqual(sorted(returnedBy(worker, "result")), ["early", "late", "maybe"], "a function's own returns, not those of a function inside it");
assert.deepEqual(sorted(takenBy(worker, "taking")), ["prompt", "type"]);
assert.deepEqual(sorted(handedTo(worker, "onCompared")), ["best", "faster"]);
assert.deepEqual(sorted(givenTo(worker, "chosen")), ["a", "b"]);
assert.throws(() => returnedBy(worker, "nowhere"), /0 functions named/);
assert.deepEqual(literalKeys(parse("({ a, ...b, ...(c || { d }) })").program.body[0].expression).unknown, 2);

// both ways, and what is listed as written elsewhere
const set = (...names) => new Set(names);
assert.deepEqual(differing("x", set("a", "b"), set("a", "b")), []);
assert.match(differing("x", set("a"), set("a", "b"))[0], /written and not in the type: b/);
assert.match(differing("x", set("a", "b"), set("a"))[0], /in the type and written nowhere: b/);
assert.deepEqual(differing("x", set("a", "b"), set("a"), { b: "spread in" }), []);
assert.match(differing("x", set("a", "b"), set("a", "b"), { b: "spread in" })[0], /listed as written elsewhere/);
assert.match(differing("x", set("a"), set("a"), { b: "spread in" })[0], /listed as written elsewhere/);

// ---- T368.2: the requests of a conduct are src/runtime/contracts/conduct.ts's Request, a tuple a kind. Three texts say
// them, in two languages, and no compiler reads two of them: the tuples src/python/convert/conduct.py yields (read here
// as text: each `yield (...)` and `needed((...))`, its kind and how long it is), the answers of the worker's two answerers
// and the two ends its loop knows (src/runtime/worker/conduct.js), and the tools' answerer (tests/conducting.mjs).
// What this does not see: what a place of a tuple holds (a begin where the type says a name), nor what an answer is.
{
  const typed = contract("conduct.ts"), requests = typed.tuplesOf("Request");
  const same = (what, declared, written) => {
    const lines = differing(what, declared, written);
    assert.deepEqual(lines, [], `${lines.join("; ")}: src/runtime/contracts/conduct.ts and the conduct say different things`);
  };
  // Python's tuples: the text from `("kind",` to its closing bracket, cut at the commas of its own level
  const text = fs.readFileSync(python("convert/conduct.py"), "utf8"), yielded = new Map();
  for (const match of text.matchAll(/(?:\byield |\bneeded\()\(\s*"(\w+)",/g)) {
    let depth = 0, count = 1, starred = false, at = match.index + match[0].indexOf("(", match[0].indexOf("(") + (match[0].startsWith("needed") ? 1 : 0));
    for (; at < text.length; at++) {
      const c = text[at];
      if ("([{".includes(c)) depth++;
      else if (")]}".includes(c) && --depth === 0) break;
      else if (c === "," && depth === 1) { count++; starred ||= /^\s*\*/.test(text.slice(at + 1)); }
    }
    // (a tuple that ends in *args is as long as whoever raised says: its kind alone is held)
    const lengths = yielded.get(match[1]) ?? new Set();
    if (!starred) lengths.add(count);
    yielded.set(match[1], lengths);
  }
  same("contracts/conduct.ts's Request, to the kinds conduct.py yields", new Set(requests.keys()), new Set(yielded.keys()));
  for (const [kind, lengths] of yielded) {
    assert.deepEqual([...lengths].filter((length) => length !== requests.get(kind)), [], `conduct.py yields ("${kind}", ...) of ${[...lengths]} places, and contracts/conduct.ts's Request says ${requests.get(kind)}`);
  }
  // the answerers: every `answers: { ... }` of the worker's and of the tools', and the type's Answers
  const answering = (tree) => {
    const found = [];
    walk(tree.program, (node) => {
      if (node.type === "ObjectProperty" && (node.key.name ?? node.key.value) === "answers" && node.value.type === "ObjectExpression") found.push(literalKeys(node.value).keys);
    });
    return found;
  };
  const worker = syntaxOf("worker/conduct.js"), tools = syntaxOf("conducting.mjs", (name) => fileURLToPath(new URL(name, import.meta.url)));
  const answerers = [...answering(worker), ...answering(tools)];
  assert.equal(answerers.length, 3, `${answerers.length} answerers found in worker/conduct.js and tests/conducting.mjs, not the worker's two and the tools' one`);
  // (the loop's own two: what it does not ask an answerer for)
  const ends = new Set([...fs.readFileSync(runtime("worker/conduct.js"), "utf8").matchAll(/\bkind === "(\w+)"/g)].map(([, kind]) => kind));
  for (const answers of answerers) {
    same("contracts/conduct.ts's Answers, to an answerer's answers", typed.keysOf("Answers"), answers);
    same("contracts/conduct.ts's Request, to an answerer's answers and the loop's two ends", new Set(requests.keys()), new Set([...answers, ...ends]));
  }
  console.log(`contracts-check: contracts/conduct.ts's ${requests.size} kinds of request are the ones conduct.py yields (each as long), the ${answerers.length} answerers answer and the loop ends on`);
}

// ---- T368.3: what the worker has of Python (contracts/python.ts) and of forward.js (contracts/forward-js.ts).
// Neither other end is read by a compiler with these types (Python; JavaScript until T368.4). What is held here is
// names, one way: every member a type says a module or an object has is a name its source has, so one renamed or
// dropped there fails here; and every member is one the worker uses (a type that lists what nothing calls is not what
// the worker has of it). What a call takes and gives is held by what runs the worker on the real ones (the smoke
// test, worker-fetches-check, the browsers).
{
  const read = (file) => fs.readFileSync(file, "utf8");
  const has = (text, name) => new RegExp(`(?<![\\w.])${name}\\b|\\.${name}\\b`).test(text);
  const worker = [runtime("worker.js"), ...fs.readdirSync(runtime("worker")).map((name) => runtime(`worker/${name}`))].map(read).join("\n");
  let held = 0;
  const holds = (typed, type, where, text) => {
    const names = [...typed.keysOf(type)].filter((name) => name !== "destroy");
    assert.ok(names.length > 0, `${type} has no members`);
    for (const name of names) {
      assert.ok(has(text, name), `${type} says ${name}, and ${where} has no such name`);
      assert.ok(new RegExp(`\\b${name}\\b`).test(worker), `${type} says ${name}, which nothing of the worker uses: a contract lists what the worker has, no more`);
      held += 1;
    }
  };
  const py = contract("python.ts");
  holds(py, "EngineWindow", "src/python/llama2_numpy.py", read(python("llama2_numpy.py")));
  holds(py, "ConverterWindow", "src/python/llama2_convert.py", read(python("llama2_convert.py")));
  holds(py, "ConductModule", "src/python/convert/conduct.py (def conduct)", read(python("convert/conduct.py")).match(/^def (\w+)\(/gm).join("\n"));
  holds(py, "Llama", "src/python/engine/model.py", read(python("engine/model.py")));
  holds(py, "Conversion", "src/python/convert/conversion.py", read(python("convert/conversion.py")));
  holds(py, "Sink", "src/python/convert/checkpoint.py (sink.<name>)", read(python("convert/checkpoint.py")).match(/\bsink\.\w+\(/g).join("\n"));
  // forward.js: the module's own exports, by importing it; the engine's members by forward/engine.js's text
  const forward = contract("forward-js.ts"), module = await import(pathToFileURL(runtime("forward.js")));
  holds(forward, "Forward", "src/runtime/forward.js's exports", Object.keys(module).join("\n"));
  holds(forward, "Engine", "src/runtime/forward/engine.js", read(runtime("forward/engine.js")));
  holds(forward, "Outside", "src/runtime/forward.js (external())", read(runtime("forward.js")));
  holds(forward, "GpuOnlyWeights", "src/runtime/forward/alone.js (gpuOnlyWeights())", read(runtime("forward/alone.js")));
  console.log(`contracts-check: contracts/python.ts's and contracts/forward-js.ts's ${held} members are names of Python's and forward.js's sources, each one the worker uses`);
}

// ---- the files of src/runtime/contracts/: types alone (contract() refuses a value), and read as types alone
const folder = path.dirname(runtime("contracts/plan.ts"));
const files = fs.readdirSync(folder, { recursive: true }).map((name) => name.split(path.sep).join("/")).filter((name) => name.endsWith(".ts")).sort();
assert.deepEqual(files, ["conduct.ts", "form.ts", "forward-js.ts", "gpu/protocol.ts", "messages.ts", "plan.ts", "python.ts"], "the contracts are others now: say in AGENTS.md what holds each to its two ends");
let types = 0;
for (const name of files) types += contract(name).names.length;
// (the runtime is under src/, where the pages that read a contract are too)
const src = path.dirname(path.dirname(folder));
let readers = 0;
for (const name of fs.readdirSync(src, { recursive: true })) {
  if (!/\.(ts|js|mjs|astro)$/.test(name)) continue;
  const text = fs.readFileSync(path.join(src, name), "utf8");
  for (const [statement] of text.matchAll(/^.*\bfrom\s*["'][^"']*contracts\/[^"']*["'].*$/gm)) {
    assert.match(statement, /^\s*import type\b/, `${name}: ${statement.trim()}\n    a contract is read with \`import type\`: nothing of it is to be in what a browser gets`);
    assert.match(statement, /\.ts["'];?\s*$/, `${name}: ${statement.trim()}\n    by its real name, .ts`);
    readers += 1;
  }
}
assert.ok(readers > 0, "nothing reads a contract");
console.log(`contracts-check: ok (the reader on made-up types and messages; ${files.length} files of contracts/, ${types} types and no value; ${readers} imports of them, each \`import type\`)`);
