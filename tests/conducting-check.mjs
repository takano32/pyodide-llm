// tests/conducting-check.mjs (T374.4)
// tests/conducting.mjs alone: what the Node tools that convert a model in Pyodide (profile-convert.mjs, page-27b.mjs)
// answer the conduct of a conversion with. The conduct is a stand-in that plays a list of requests behind the face
// Pyodide gives a Python generator (next(answer), each request a proxy to be destroyed), the files are a folder under
// .tmp/, and there is no Pyodide. What is seen: every kind of request answered with what it asks for, a file that is
// not there answered with undefined (never null), a range wherever it begins and short at the file's end, a stream's
// parts fed to the feed the request brings, every proxy let go once, the hooks a tool times a conversion by, and what
// a path stands for.
//
//   node tests/conducting-check.mjs
//
// What this does not see: the real conduct (the tools themselves run it; tests/test_conducting.py runs the Python
// answerers with it).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { answered, fromFolder, listed } from "./conducting.mjs";

const room = path.join(fileURLToPath(new URL("..", import.meta.url)), ".tmp", `conducting-check-${process.pid}`);
fs.rmSync(room, { recursive: true, force: true });
const folderOf = (name, files) => {
  const folder = path.join(room, name);
  fs.mkdirSync(folder, { recursive: true });
  for (const [file, data] of Object.entries(files)) fs.writeFileSync(path.join(folder, file), data);
  return folder;
};
let passed = 0;
const ok = (name) => { passed++; console.log(`ok: ${name}`); };

// ---- the stand-ins. Every proxy counts how often it was destroyed
const proxies = [];
const proxy = (fields) => {
  const made = { destroyed: 0, destroy() { made.destroyed++; }, ...fields };
  proxies.push(made);
  return made;
};
/** A conduct that asks for requests, one after another: { steps, answers } (the answers it was sent). */
function played(requests) {
  const answers = [];
  let at = 0;
  const step = () => (at < requests.length
    ? { done: false, value: proxy({ toJs(options) { assert.deepEqual(options, { depth: 1 }); assert.equal(this.destroyed, 0); return requests[at++]; } }) }
    : { done: true });
  return { answers, steps: { next(answer) { if (at) answers.push(answer); return step(); } } };
}
const data = Uint8Array.from({ length: 1000 }, (_, at) => at % 251);
const same = (bytes, begin, end) => assert.deepEqual([...bytes], [...data.subarray(begin, end)]);

try {
  {
    const folder = folderOf("answers", { "config.json": "{\"a\": \"あ\"}", "tokenizer.json": "vocabulary", "model.gguf": data });
    const fed = [], feed = proxy({});
    const conversion = proxy({});
    const { steps, answers } = played([["text", "weights", "config.json"], ["text", "vocabulary", "chat_template.jinja"],
      ["bytes", "weights", "tokenizer.json"], ["bytes", "weights", "spiece.model"], ["range", "weights", "model.gguf", 0, 100],
      ["range", "weights", "model.gguf", 100, 300], ["range", "weights", "model.gguf", 900, 5000], ["range", "weights", "other.gguf", 0, 8],
      ["size", "weights", "model.gguf"],
      ["stream", "weights", "model.gguf", 300, 1000, 300, 1000, Object.assign((part) => { assert.equal(feed.destroyed, 0); fed.push(part); }, feed)],
      ["done", conversion]]);
    const starts = [], reads = [];
    const answerer = fromFolder(folder, { part: 256, starting: (total) => starts.push([total, fed.length]), reading: (ms) => reads.push(ms) });
    assert.equal(answered(steps, answerer), conversion);
    assert.equal(answers[0], "{\"a\": \"あ\"}");
    assert.equal(answers[1], undefined);  // (not null: null is not None in Python)
    assert.equal(Buffer.from(answers[2]).toString(), "vocabulary");
    assert.ok(answers[2] instanceof Uint8Array && !(answers[2] instanceof Buffer), "Pyodide refuses a Buffer where it takes a Uint8Array");
    assert.equal(answers[3], undefined);
    same(answers[4][0], 0, 100); assert.equal(answers[4][1], 1000);
    same(answers[5][0], 100, 300);  // T374.3: from where the bytes in hand end, not from the file's first byte
    same(answers[6][0], 900, 1000); assert.equal(answers[6][1], 1000);  // fewer where the file ends first
    assert.equal(answers[7], undefined);
    assert.equal(answers[8], 1000);
    assert.equal(answers[9], undefined);
    ok("every kind of request is answered with what it asks for, and a file that is not there with undefined");
    assert.deepEqual(fed.map((part) => part.length), [256, 256, 188]);
    same(Uint8Array.from(fed.flatMap((part) => [...part])), 300, 1000);
    assert.deepEqual(starts, [[1000, 0]]);
    assert.equal(reads.length, 3);
    ok("a stream's parts go to the feed in the order of the file, the clock started once before the first and each read timed");
    assert.ok(proxies.filter((made) => made !== conversion).every((made) => made.destroyed === 1), "a proxy was let go twice or never");
    assert.equal(conversion.destroyed, 0);
    answerer.close(); answerer.close();
    ok("every proxy a request brought is let go once, and the conversion is the caller's");
  }
  {
    const folder = folderOf("empty", {});
    const { steps } = played([["stream", "weights", "not-there.gguf", 64, 64, 64, 64, proxy({})], ["stream", "weights", "nor-this.gguf", 0, 0, 64, 64, proxy({})], ["done", null]]);
    const starts = [];
    answered(steps, fromFolder(folder, { starting: (total) => starts.push(total) }));
    assert.deepEqual(starts, [64]);
    ok("a stream with nothing in it opens no file, and still starts the clock, once");
  }
  {
    const folder = folderOf("short", { "model.gguf": data });
    const fed = [];
    const { steps } = played([["stream", "weights", "model.gguf", 0, 1500, 0, 1500, (part) => fed.push(part.length)], ["done", null]]);
    assert.throws(() => answered(steps, fromFolder(folder, { part: 400 })), /ends at 1000/);
    assert.deepEqual(fed, [400, 400]);
    ok("a file that ends before its stream does is an error, not a short conversion");
  }
  {
    const folder = folderOf("ends", {});
    const brought = proxy({});
    assert.throws(() => answered(played([["missing", "vocabulary", "config.json"]]).steps, fromFolder(folder)), new RegExp(`ends has no config.json`));
    assert.throws(() => answered(played([["folder", "weights", brought]]).steps, fromFolder(folder)), /asked for folder, which nothing here answers/);
    assert.equal(brought.destroyed, 1);
    assert.throws(() => answered(played([["constructor", "weights"]]).steps, fromFolder(folder)), /asked for constructor/);
    assert.throws(() => answered(played([["text", "weights", "config.json"]]).steps, fromFolder(folder)), /ended without a word/);
    ok("a conduct that ends for want of a file says where, and what nothing answers ends the loop");
  }
  {
    const gguf = folderOf("alone", { "model.Q8_0.gguf": "" });
    assert.deepEqual(listed(path.join(gguf, "model.Q8_0.gguf")), { hf: { weights: "model.Q8_0.gguf" }, folder: gguf });
    const beside = folderOf("with", { "b.gguf": "", "a.gguf": "", "config.json": "{}" });
    const found = listed(beside);
    assert.equal(found.hf.weights, "a.gguf");
    assert.ok(found.hf.vocabulary && Object.keys(found.hf.vocabulary).length && !("tokenizer" in found.hf.vocabulary) && !("tokenizer" in found.hf));
    assert.equal(found.folder, beside);
    assert.deepEqual(listed(folderOf("one", { "config.json": "{}" })).hf, { weights: "model.safetensors" });
    ok("what a path stands for: a GGUF alone, a GGUF beside its original's files (the first by its name), safetensors; no tokenizer named");
  }
} finally {
  fs.rmSync(room, { recursive: true, force: true });
}
console.log(`conducting-check: ${passed} passed`);
