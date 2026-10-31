// tests/failure-check.ts (T.380.1): how a failure is carried and told apart (src/runtime/failure.ts), by itself: no
// worker, no Pyodide, no GPU, no page. Node only, under a second:
//
//   node tests/failure-check.ts
//
// What is held:
//   - a Failure is an Error by name, said by its sentence alone, and known by its kind, also one that another copy of
//     the module made (each bundle has its own);
//   - failureOf() reads whatever is thrown here into one: the stand-ins below are what T.90, T.133 and T.242 met (Python's
//     ValueError and MemoryError as Pyodide hands them over, an error of JavaScript's that crossed Python, the three
//     engines' words for a memory that ran out, a stack overflow that is none, Emscripten's ExitStatus, a thing that is
//     no error at all), each with the kind and the very sentence the worker's one catch said of it before;
//   - in the sources: nothing puts a mark on an Error (Object.assign, or a property set on one that was caught), nothing
//     but failure.ts reads the words of Pyodide's exceptions, and every kind is made somewhere and read somewhere else.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { untyped } from "./untyped.ts";

const root = fileURLToPath(new URL("..", import.meta.url));
const { Failure, failureOf, kindOf, told, isError } = await import("../src/runtime/failure.ts");

// ---- a Failure
{
  const failure = new Failure("runtime", "Pyodide did not come");
  assert.ok(failure instanceof Error && isError(failure));
  assert.deepEqual([failure.name, failure.message, String(failure), failure.kind, failure.again, failure.status, failure.cause],
    ["Error", "Pyodide did not come", "Error: Pyodide did not come", "runtime", undefined, undefined, undefined]);
  assert.match(failure.stack, /^Error: Pyodide did not come\n\s+at /, "where it was made");
  assert.equal(told(failure), "Error: Pyodide did not come", "told as an Error is");
  const cause = new TypeError("Failed to fetch");
  const refused = new Failure("refused", "huggingface.co answered 404", { status: 404, again: false, cause });
  assert.deepEqual([refused.kind, refused.status, refused.again, refused.cause], ["refused", 404, false, cause]);
  assert.equal(new Failure("late", "x", { stack: "where it happened" }).stack, "where it happened");
  // its kind, and nothing of what is no Failure
  assert.equal(kindOf(failure), "runtime");
  for (const other of [new Error("runtime"), Object.assign(new Error("x"), { kind: "made up" }), { kind: "runtime" }, "runtime", null, undefined, 404]) {
    assert.equal(kindOf(other), undefined, `${told(other)} is no Failure`);
  }
  // a Failure of another copy of the module (a worker's bundle and the page's each have one) is one
  const copy = await import("../src/runtime/failure.ts?copy=1");
  assert.notEqual(copy.Failure, Failure);
  const theirs = new copy.Failure("updated", "the site was updated");
  assert.ok(!(theirs instanceof Failure));
  assert.equal(kindOf(theirs), "updated");
  assert.equal(failureOf(theirs), theirs, "and is read as itself");
  assert.equal(failureOf(failure), failure);
}
console.log("failure-check: a Failure is an Error by name with a kind, known by the kind whichever copy of the module made it");

// ---- failureOf(): what the others throw. Pyodide's PythonError: an Error whose name is PythonError, whose message is
// the traceback, and whose type is the name of Python's exception
const python = (type, traceback) => Object.assign(new Error(traceback), { name: "PythonError", type });
const named = (name, message) => Object.assign(new Error(message), { name });
const traceback = (last) => `Traceback (most recent call last):\n  File "/lib/python314.zip/_pyodide/_base.py", line 597, in eval_code_async\n  File "<exec>", line 3, in load\n${last}\n`;
const CASES = [
  // [what was thrown, its kind, the sentence]
  [new Error("The model is not ready."), "refused", "The model is not ready."],
  [new Error(""), "refused", ""],
  [python("ValueError", traceback("ValueError: The prompt is 259 tokens long, but only 255 fit.")), "refused", "The prompt is 259 tokens long, but only 255 fit."],
  [python("ValueError", traceback("ValueError: This is not a safetensors file.")), "refused", "This is not a safetensors file."],
  // (an error of this site's own that was thrown inside a call from Python: said as itself)
  [python("JsException", traceback("pyodide.ffi.JsException: Error: This model needs more than 4 GB of memory, which this browser cannot give a web page (no 64-bit WebAssembly memory: Safari has none yet). Chrome and Firefox can.")),
    "refused", "This model needs more than 4 GB of memory, which this browser cannot give a web page (no 64-bit WebAssembly memory: Safari has none yet). Chrome and Firefox can."],
  // T.90: the memory, in its three shapes and through Python
  [python("MemoryError", traceback("MemoryError")), "memory", "MemoryError"],
  [new RangeError("Invalid array buffer length"), "memory", "Invalid array buffer length"],
  [new RangeError("WebAssembly.Memory.grow(): Maximum memory size exceeded"), "memory", "WebAssembly.Memory.grow(): Maximum memory size exceeded"],
  [named("InternalError", "out of memory"), "memory", "out of memory"],
  [python("JsException", traceback("pyodide.ffi.JsException: RangeError: WebAssembly.Memory.grow(): Maximum memory size exceeded")), "memory", "WebAssembly.Memory.grow(): Maximum memory size exceeded"],
  [python("JsException", traceback("pyodide.ffi.JsException: InternalError: out of memory")), "memory", "out of memory"],
  // (V8's stack overflow is a RangeError and no memory that ran out)
  [new RangeError("Maximum call stack size exceeded"), "refused", "RangeError: Maximum call stack size exceeded"],
  // anything else by its name and words
  [new TypeError("Load failed"), "refused", "TypeError: Load failed"],
  [named("AbortError", "The operation was aborted."), "refused", "AbortError: The operation was aborted."],
  [python("KeyError", traceback("KeyError: 'dim'")), "refused", `PythonError: ${traceback("KeyError: 'dim'")}`],
  // (a JsException whose last line is not the shape: as the PythonError it is)
  [python("JsException", "pyodide.ffi.JsException"), "refused", "PythonError: pyodide.ffi.JsException"],
  // T.242: what is thrown is not always an Error
  [{ name: "ExitStatus", message: "Program terminated with exit(1)", status: 1 }, "refused", "ExitStatus: Program terminated with exit(1)"],
  [{ code: 7 }, "refused", 'something that is not an error was thrown: {"code":7}'],
  ["a string", "refused", "a string"],
  [undefined, "refused", "undefined"],
  [null, "refused", "null"],
];
for (const [thrown, kind, sentence] of CASES) {
  const failure = failureOf(thrown);
  assert.deepEqual([failure.kind, failure.message], [kind, sentence], `what is said of ${told(thrown).slice(0, 60)}`);
  assert.equal(failure.cause, thrown, "what was caught is kept");
  assert.equal(failure.stack, String(thrown?.stack ?? told(thrown)), "where it happened is the caught value's");
  assert.deepEqual([failure.again, failure.status], [undefined, undefined], "nothing is known of another try, nor of a server");
  assert.equal(kindOf(thrown), undefined);
}
console.log(`failure-check: failureOf() reads ${CASES.length} things that are thrown here, each to its kind and the sentence it was said by`);

// ---- the sources: one way, and no other
{
  const folders = ["src/runtime", "src/page", "src/benchmark", "src/bench", "src/models", "src"];
  const sources = [...new Set(folders.flatMap((folder) => fs.readdirSync(path.join(root, folder), { recursive: folder !== "src" })
    .filter((name) => /\.(?:ts|js)$/.test(name) && !/\.d\.ts$/.test(name)).map((name) => `${folder}/${String(name).split(path.sep).join("/")}`)))].sort();
  assert.ok(sources.length > 100 && sources.includes("src/runtime/failure.ts") && sources.includes("src/runtime/worker/sources.ts"));
  // (the code alone: the comments tell of how it was)
  const code = (name) => untyped(fs.readFileSync(path.join(root, name), "utf8"), name).replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "").replace(/\s\/\/ .*$/gm, "");
  const texts = new Map(sources.map((name) => [name, code(name)]));
  const made = new Map(), read = new Map();
  for (const [name, text] of texts) {
    assert.ok(!/Object\.assign\(\s*new \w*Error\b/.test(text), `${name} puts a mark on a new Error with Object.assign: a Failure of a kind says it (src/runtime/failure.ts)`);
    assert.ok(!/Object\.assign\(\s*(?:error|err|caught|failure)\b/.test(text), `${name} writes on an error it caught: a Failure of a kind says it (src/runtime/failure.ts)`);
    assert.ok(!/\b(?:error|err)\.(?:final|pyodide|late|cutShort|updated|directLost|status)\s*=[^=]/.test(text), `${name} sets a mark on an error: a Failure of a kind says it (src/runtime/failure.ts)`);
    if (name !== "src/runtime/failure.ts") {
      assert.ok(!/JsException|MemoryError|\.type === "ValueError"|InternalError|call stack/.test(text), `${name} reads the words of what Pyodide or the browser throws: failureOf() is the one place (src/runtime/failure.ts)`);
    }
    // (T.380.2: nor takes the last line of a message itself, which is how a traceback is read: failure.ts's lastLine())
    if (name !== "src/runtime/failure.ts") {
      assert.ok(!/\.split\(\s*["'`]\\n["'`]\s*\)\s*\.(?:pop\(\)|at\(-1\))/.test(text), `${name} takes the last line of a text itself: failure.ts reads what is thrown (failureOf(), or lastLine() for Python's own last line)`);
    }
    for (const [, kind] of text.matchAll(/new Failure\(\s*"([\w-]+)"/g)) made.set(kind, [...(made.get(kind) ?? []), name]);
    for (const [, kind] of text.matchAll(/\.?kind(?:Of\([^)]*\))? [!=]== "([\w-]+)"/g)) read.set(kind, [...(read.get(kind) ?? []), name]);
  }
  // every kind of the type is made by a file and read by one, and no other word is used as a kind
  const kinds = /const KINDS = \[([^\]]+)\] as const;/.exec(fs.readFileSync(path.join(root, "src/runtime/failure.ts"), "utf8"))[1].split(",").map((one) => one.trim().slice(1, -1));
  assert.deepEqual(kinds, ["refused", "memory", "runtime", "updated", "late", "cut-short"]);
  for (const kind of made.keys()) assert.ok(kinds.includes(kind), `${made.get(kind)[0]} makes a Failure of the kind "${kind}", which is none of ${kinds.join(", ")}`);
  for (const kind of kinds) {
    // (memory is made by failureOf() alone, of what the browser and Python throw: by a variable, not by a word)
    if (kind !== "memory") assert.ok(made.has(kind), `no file makes a Failure of the kind "${kind}": a kind nobody makes is not one`);
    // (refused is the kind of whatever has no other, which nothing asks for by name)
    if (kind !== "refused") assert.ok(read.has(kind), `no file reads the kind "${kind}": a kind nobody reads is not one`);
  }
  const lines = kinds.map((kind) => `${kind} (made ${(made.get(kind) ?? []).length || "by failureOf()"}, read ${(read.get(kind) ?? []).length})`);
  console.log(`failure-check: ${sources.length} sources put no mark on an Error and read no exception's words but failure.ts; the kinds: ${lines.join(", ")}`);
}
console.log("failure-check: ok");
