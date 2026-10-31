// tests/fetching-check.ts (T.357.6): the Node tools' one place for fetching (tests/fetching.ts), by itself, and the
// tools that go through it or through Python's (tests/fetching.py) and that no other light check runs: the text of
// Wikipedia (tests/wikipedia.ts) and the files gguf.yml compares (tests/gguf_fetch.py). Nothing leaves this machine:
// the server is one of node:http on 127.0.0.1, which can do what a stand-in of fetch cannot (close a connection
// before the length it promised, cut a chunked body, never answer), and tests/fakes/fetch.ts is the network where
// the answer itself is wrong in a way no real server of node:http sends (a body that ends in order and short).
//
//   node tests/fetching-check.ts        (Node, and python for the last part: PYTHON=.venv/bin/python; a second or so)
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { madeUpNetwork } from "./fakes/fetch.ts";
import { Answered, download, fetched } from "./fetching.ts";
import { ARTICLES, wikipediaText } from "./wikipedia.ts";

const root = fileURLToPath(new URL("..", import.meta.url));
const room = path.join(root, ".tmp", `fetching-check-${process.pid}`);
fs.rmSync(room, { recursive: true, force: true });
fs.mkdirSync(room, { recursive: true });

let passed = 0;
const ok = (name) => { passed++; console.log(`ok: ${name}`); };
const failure = (promise) => promise.then(() => undefined, (error) => error);

// ---- a server whose every path answers as the check says: answers[path] is a list, one answer a request, the last
// one for every request after. An answer: { status, body, length (the Content-Length it says; none: chunked),
// cut (how many bytes go before the connection is destroyed), never (no answer at all) }
const asked = [];  // every request: "<path>" and its headers
let answers = {};
const sockets = new Set();
const server = http.createServer((request, response) => {
  const list = answers[request.url] ?? [{ status: 404, body: "" }];
  const n = asked.filter((one) => one.path === request.url).length;
  asked.push({ path: request.url, headers: request.headers });
  const { status = 200, body = "", length = Buffer.byteLength(body), cut, never } = list[Math.min(n, list.length - 1)];
  if (never) return;
  response.writeHead(status, length === null ? {} : { "Content-Length": length });
  if (cut === undefined) return response.end(body);
  response.write(Buffer.from(body).subarray(0, cut), () => setTimeout(() => request.socket.destroy(), 10));
});
server.on("connection", (socket) => {
  sockets.add(socket);
  socket.on("close", () => sockets.delete(socket));
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const serving = (table) => {
  answers = table;
  asked.length = 0;
};
const WHOLE = "y".repeat(100);
const placed = (name) => path.join(room, name);
const folder = (name) => fs.existsSync(placed(name)) ? fs.readdirSync(placed(name)).sort() : [];

// ---- download(): a file to its place
{
  serving({ "/file": [{ body: WHOLE }] });
  const target = placed("whole/deeper/file.bin");
  assert.equal(await download(`${base}/file`, target), target, "the place is what is answered");
  assert.equal(fs.readFileSync(target, "latin1"), WHOLE);
  assert.deepEqual(folder("whole/deeper"), ["file.bin"], "the folder is made, and no part of a file stays");
  assert.equal(asked.length, 1);
  assert.equal(asked[0].headers["accept-encoding"], "identity", "the body is asked for as it is: its length is then the length that comes");
  ok("a file is fetched to its place, through its folder, asked for unpacked");
}
{
  serving({ "/file": [{ body: "other bytes" }] });
  const target = placed("whole/deeper/file.bin");
  assert.equal(await download(`${base}/file`, target), target);
  assert.equal(fs.readFileSync(target, "latin1"), WHOLE, "the file that was there is the file");
  assert.equal(asked.length, 0, "and nothing was asked");
  ok("a file that is there is not fetched again");
}
{
  // (what huggingface.co did to six of 80 GGUFs in one run, T.307: Node's fetch fails here by itself)
  serving({ "/file": [{ body: WHOLE, cut: 40 }, { body: WHOLE, cut: 99 }, { body: WHOLE }] });
  const told = [];
  const target = placed("short/file.bin");
  await download(`${base}/file`, target, { said: (error, attempt) => told.push(attempt) });
  assert.equal(fs.readFileSync(target, "latin1"), WHOLE, "the whole file, and nothing of the short ones before it");
  assert.equal(asked.length, 3);
  assert.deepEqual(told, [1, 2], "each try that failed is told");
  assert.deepEqual(folder("short"), ["file.bin"]);
  ok("a connection that closes short of the promised length is asked for again");
}
{
  serving({ "/file": [{ body: WHOLE, cut: 40 }] });
  const told = [];
  const error = await failure(download(`${base}/file`, placed("never-whole/file.bin"), { said: (_, attempt) => told.push(attempt) }));
  assert.ok(error instanceof Error, "the third failure is thrown");
  assert.equal(asked.length, 3, "three tries by default");
  assert.deepEqual(told, [1, 2], "the last failure is thrown, not told");
  assert.deepEqual(folder("never-whole"), [], "neither a short file nor a part of one is left");
  serving({ "/file": [{ body: WHOLE, cut: 40 }] });
  await failure(download(`${base}/file`, placed("never-whole/file.bin"), { tries: 5 }));
  assert.equal(asked.length, 5, "as many tries as were asked for");
  ok("a file that never comes whole is an error, and leaves nothing");
}
{
  serving({ "/file": [{ body: WHOLE, length: null, cut: 40 }, { body: WHOLE, length: null }] });
  const target = placed("chunked/file.bin");
  await download(`${base}/file`, target);
  assert.equal(fs.readFileSync(target, "latin1"), WHOLE);
  assert.equal(asked.length, 2, "a chunked body that was cut is asked for again");
  ok("an answer that says no length is taken as it comes, and one that was cut is asked for again");
}
{
  serving({ "/file": [{ status: 404, body: "Entry not found" }] });
  const error = await failure(download(`${base}/file`, placed("refused/file.bin")));
  assert.ok(error instanceof Answered, "the server's answer is the error");
  assert.equal(error.status, 404);
  assert.match(error.message, /\/file: the server answered 404$/);
  assert.equal(asked.length, 1, "what is refused is not asked for again");
  assert.deepEqual(folder("refused"), [], "and the answer's body is no file");
  for (const status of [401, 403, 429]) {
    serving({ "/file": [{ status }] });
    assert.equal((await failure(download(`${base}/file`, placed("refused/file.bin")))).status, status);
    assert.equal(asked.length, 1, `a ${status} is asked once`);
  }
  ok("what the server refuses (4xx) is thrown at once, with its status");
}
{
  serving({ "/file": [{ status: 429 }, { status: 408 }, { status: 200, body: "whole" }] });
  const got = await download(`${base}/file`, placed("again/file.bin"), { again: [408, 429] });
  assert.equal(fs.readFileSync(got, "utf8"), "whole");
  assert.equal(asked.length, 3, "a status named in again is asked for again");
  ok("a 4xx named in `again` is asked for again (Wikipedia's 408 and 429)");
}
{
  serving({ "/file": [{ status: 404 }] });
  assert.equal(await download(`${base}/file`, placed("optional/file.bin"), { optional: true }), null);
  assert.equal(asked.length, 1);
  assert.deepEqual(folder("optional"), []);
  serving({ "/file": [{ status: 503 }] });
  const error = await failure(download(`${base}/file`, placed("optional/file.bin"), { optional: true }));
  assert.equal(error?.status, 503, "a server that failed did not say the file is not there");
  assert.equal(asked.length, 3);
  serving({ "/file": [{ body: WHOLE, cut: 40 }] });
  assert.ok(await failure(download(`${base}/file`, placed("optional/file.bin"), { optional: true })), "nor did a connection that broke");
  ok("an optional file that is refused is null; a failure is no such answer");
}
{
  serving({ "/file": [{ status: 500 }, { status: 502 }, { body: WHOLE }] });
  const target = placed("failed/file.bin");
  await download(`${base}/file`, target);
  assert.equal(fs.readFileSync(target, "latin1"), WHOLE);
  assert.equal(asked.length, 3);
  ok("a server's own failure (5xx) is asked for again");
}
{
  const waits = [];
  serving({ "/file": [{ status: 500 }, { status: 500 }, { status: 500 }, { body: WHOLE }] });
  await download(`${base}/file`, placed("waited/file.bin"), { tries: 4, wait: 10000, sleep: async (ms) => waits.push(ms) });
  assert.deepEqual(waits, [10000, 20000, 30000], "the wait once more before each next try");
  serving({ "/file": [{ status: 500 }, { body: WHOLE }] });
  await download(`${base}/file`, placed("waited/other.bin"), { sleep: async (ms) => waits.push(ms) });
  assert.equal(waits.length, 3, "and none where none was asked for");
  ok("the tries are as far apart as was asked");
}
{
  // an answer that ends in order, short of the length it said: no server of node:http sends one, a proxy may
  const answering = (list) => {
    const network = madeUpNetwork();
    network.answers((url, init, n) => list[Math.min(n, list.length - 1)]());
    return network;
  };
  const saying = (body, headers) => () => new Response(body, { headers });
  const target = placed("lied/file.bin");
  const between = [];  // what is in the folder as the second try is asked for
  let network = answering([saying("x".repeat(40), { "Content-Length": "100" }), () => (between.push(...folder("lied")), saying(WHOLE, { "Content-Length": "100" })())]);
  await download("https://hub.test/file", target, { fetch: network.fetch });
  assert.equal(fs.readFileSync(target, "latin1"), WHOLE);
  assert.equal(network.requests.length, 2, "40 bytes of a promised 100 are not the file");
  assert.deepEqual(between, ["file.bin.part"], "what has not all come is not under the file's name: a run that is stopped leaves no short file to be read");
  network = answering([saying("x".repeat(40), { "Content-Length": "100" })]);
  const error = await failure(download("https://hub.test/file", placed("lied/short.bin"), { fetch: network.fetch }));
  assert.match(error.message, /40 of 100 bytes came$/, "the error says what came");
  assert.equal(network.requests.length, 3);
  assert.deepEqual(folder("lied"), ["file.bin"], "and the short file is not left where the next run would read it");
  network = answering([saying("x".repeat(140), { "Content-Length": "100" }), saying(WHOLE, { "Content-Length": "100" })]);
  await download("https://hub.test/file", placed("lied/long.bin"), { fetch: network.fetch });
  assert.equal(network.requests.length, 2, "nor are 140 bytes");
  // (an encoded body's Content-Length is the length that travelled, not the length that is read)
  network = answering([saying(WHOLE, { "Content-Length": "31", "Content-Encoding": "gzip" })]);
  await download("https://hub.test/file", placed("lied/encoded.bin"), { fetch: network.fetch });
  assert.equal(network.requests.length, 1, "an encoded answer is not held to a length");
  network = answering([saying(WHOLE, {})]);
  await download("https://hub.test/file", placed("lied/unsaid.bin"), { fetch: network.fetch, headers: { "User-Agent": "a tool" } });
  assert.equal(network.requests.length, 1, "nor is one that says none");
  assert.deepEqual({ ...network.requests[0].init.headers }, { "Accept-Encoding": "identity", "User-Agent": "a tool" }, "the tool's headers go with the request");
  assert.ok(network.requests[0].signal instanceof AbortSignal, "every request has its time limit");
  ok("a body is held to the length its answer says, where it says one of the body as it comes");
}

{
  serving({ "/file": [{ never: true }, { body: WHOLE }] });
  const target = placed("silent/file.bin");
  await download(`${base}/file`, target, { timeout: 200 });
  assert.equal(fs.readFileSync(target, "latin1"), WHOLE);
  assert.equal(asked.length, 2);
  ok("a request that got no answer in its time is asked for again");
}

// ---- fetched(): an answer's body
{
  serving({ "/text": [{ status: 503 }, { body: WHOLE, cut: 10 }, { body: "héllo" }] });
  const body = await fetched(`${base}/text`);
  assert.ok(Buffer.isBuffer(body));
  assert.equal(body.toString(), "héllo", "the whole body, and nothing of the try that was cut");
  assert.equal(asked.length, 3);
  serving({ "/text": [{ status: 404 }] });
  const error = await failure(fetched(`${base}/text`));
  assert.equal(error.status, 404);
  assert.equal(asked.length, 1);
  serving({ "/text": [{ body: "" }] });
  assert.equal((await fetched(new URL(`${base}/text`))).length, 0, "an empty body is a body; an address may be a URL");
  ok("a body is fetched whole by the same rules");
}

// ---- tests/wikipedia.ts: the three articles of a language as one text
{
  const article = (title) => JSON.stringify({ query: { pages: { 7: { title, extract: `${title}: ${"文".repeat(7000)}` } } } });
  const network = madeUpNetwork();
  const lines = [], waits = [];
  const said = console.error;
  console.error = (line) => lines.push(line);
  let text;
  try {
    network.answers((url, init, n) => {
      const title = new URL(url).searchParams.get("titles");
      if (title === "夏目漱石" && n === 0) throw Object.assign(new TypeError("fetch failed"), { cause: { code: "ETIMEDOUT" } });
      if (title === "夏目漱石" && n === 1) return new Response("", { status: 503 });
      return new Response(article(title));
    });
    text = await wikipediaText("ja", undefined, { fetch: network.fetch, sleep: async (ms) => waits.push(ms) });
  } finally {
    console.error = said;
  }
  assert.deepEqual(network.requests.map(({ url }) => new URL(url).searchParams.get("titles")), ["富士山", "夏目漱石", "夏目漱石", "夏目漱石", "新幹線"]);
  assert.equal(new URL(network.requests[0].url).host, "ja.wikipedia.org");
  assert.equal(network.requests[0].init.headers["User-Agent"], "pyodide-llm perplexity measurement");
  assert.equal(text, ARTICLES.ja.map((title) => `${title}: ${"文".repeat(7000)}`.slice(0, 6000) + "\n").join(""), "the beginning of each article, a line after each");
  assert.deepEqual(lines, ["wikipedia.ts: attempt 1 failed (ETIMEDOUT); trying again",
                           `wikipedia.ts: attempt 2 failed (${network.requests[2].url}: the server answered 503); trying again`]);
  assert.deepEqual(waits, [10000, 20000], "10 s more before each next try (the review of T.236)");
  // five tries, and a refusal is not asked again
  network.requests.length = 0;
  network.answers(() => new Response("", { status: 500 }));
  console.error = () => {};
  try {
    assert.equal((await failure(wikipediaText("en", ["Shinkansen"], { fetch: network.fetch, sleep: async () => {} }))).status, 500);
    assert.equal(network.requests.length, 5, "five tries");
    network.requests.length = 0;
    network.answers(() => new Response("", { status: 403 }));
    assert.equal((await failure(wikipediaText("en", ["Shinkansen"], { fetch: network.fetch, sleep: async () => {} }))).status, 403);
    assert.equal(network.requests.length, 1);
  } finally {
    console.error = said;
  }
  ok("Wikipedia's text is fetched by the same place, five tries 10 s further apart each, and says each one that failed");
}

// ---- tests/gguf_fetch.py: what gguf.yml fetches before tests/gguf_check.py, by Python's download() (the real urllib,
// whose read() returns a body that closed short without a word: the server's `cut` is that)
{
  const python = process.env.PYTHON ?? "python";
  const run = (...args) => new Promise((resolve) => {
    const child = spawn(python, [path.join(root, "tests/gguf_fetch.py"), ...args], { env: { ...process.env, HF_ENDPOINT: base } });
    let out = "", err = "";
    child.stdout.on("data", (piece) => { out += piece; });
    child.stderr.on("data", (piece) => { err += piece; });
    child.on("close", (code) => resolve({ code, out, err }));
  });
  const at = (repo, revision, name) => `/${repo}/resolve/${revision}/${name}`;
  const GGUF = "g".repeat(3000), WEIGHTS = "w".repeat(2000);
  // stage 1: one safetensors, tokenizer.json; the GGUF closes short once, and its repository fails once
  serving({
    [at("owner/model", "abc", "config.json")]: [{ body: "{}" }],
    [at("owner/model", "abc", "tokenizer.json")]: [{ status: 502 }, { body: "{\"vocabulary\": 1}" }],
    [at("owner/model", "abc", "model.safetensors")]: [{ body: WEIGHTS }],
    [at("other/model-GGUF", "def", "folder/model-Q8_0.gguf")]: [{ body: GGUF, cut: 1000 }, { body: GGUF }],
  });
  let ran = await run("owner/model@abc", "other/model-GGUF@def/folder/model-Q8_0.gguf", placed("gguf-one"));
  assert.equal(ran.code, 0, ran.err);
  assert.deepEqual(folder("gguf-one"), ["config.json", "model.gguf", "model.safetensors", "tokenizer.json"]);
  assert.equal(fs.readFileSync(placed("gguf-one/model.gguf"), "latin1"), GGUF, "the GGUF whole, under the name the check reads");
  assert.equal(fs.readFileSync(placed("gguf-one/model.safetensors"), "latin1"), WEIGHTS);
  assert.deepEqual(asked.map((one) => one.path.split("/").pop()), ["config.json", "tokenizer.json", "tokenizer.json", "model.safetensors", "model-Q8_0.gguf", "model-Q8_0.gguf"]);
  const lines = ran.out.trim().split("\n");
  assert.equal(lines.length, 3, ran.out);
  assert.match(lines[0], /^tokenizer\.json is asked for again: HTTP Error 502/);
  assert.match(lines[1], /^model\.gguf is asked for again: .*1,000 of 3,000 bytes came$/, "a short download is a line of the log");
  assert.equal(lines[2], "['config.json', 'model.gguf', 'model.safetensors', 'tokenizer.json']");
  ok("gguf_fetch.py fetches an original of one file and a GGUF, each whole, and says each try that failed");

  // stage 2: the shards an index names, and whichever tokenizer there is
  serving({
    [at("owner/model", "abc", "config.json")]: [{ body: "{}" }],
    [at("owner/model", "abc", "tokenizer.model")]: [{ body: "pieces" }],
    [at("owner/model", "abc", "model.safetensors.index.json")]: [{ body: JSON.stringify({ weight_map: { a: "model-00002-of-00002.safetensors", b: "model-00001-of-00002.safetensors", c: "model-00002-of-00002.safetensors" } }) }],
    [at("owner/model", "abc", "model-00001-of-00002.safetensors")]: [{ body: "first" }],
    [at("owner/model", "abc", "model-00002-of-00002.safetensors")]: [{ body: "second" }],
    [at("other/model-GGUF", "def", "model.gguf")]: [{ body: GGUF }],
  });
  ran = await run("owner/model@abc", "other/model-GGUF@def/model.gguf", placed("gguf-two"), "--either-tokenizer");
  assert.equal(ran.code, 0, ran.err);
  assert.deepEqual(folder("gguf-two"), ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.gguf", "model.safetensors.index.json", "tokenizer.model"]);
  assert.deepEqual(asked.map((one) => one.path.split("/").pop()), ["config.json", "tokenizer.json", "tokenizer.model", "model.safetensors", "model.safetensors.index.json",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.gguf"], "each shard once, in order; what is not there is asked once");
  assert.equal(ran.out.trim().split("\n")[0], "tokenizer.json 404", "the tokenizer the original does not have is a line");
  ok("gguf_fetch.py fetches the shards an index names, and whichever tokenizer the original has");

  // what must be there and is not; a GGUF that never comes whole
  ran = await run("owner/model@abc", "other/model-GGUF@def/model.gguf", placed("gguf-three"));
  assert.notEqual(ran.code, 0, "without --either-tokenizer a tokenizer.json must be there");
  assert.match(ran.err, /HTTP Error 404/);
  serving({ ...answers, [at("other/model-GGUF", "def", "model.gguf")]: [{ body: GGUF, cut: 1000 }] });
  ran = await run("owner/model@abc", "other/model-GGUF@def/model.gguf", placed("gguf-four"), "--either-tokenizer");
  assert.notEqual(ran.code, 0);
  assert.equal(asked.filter((one) => one.path.endsWith("model.gguf")).length, 4, "four tries (T.307)");
  assert.ok(!fs.existsSync(placed("gguf-four/model.gguf")), "and no model.gguf for the check to read");
  assert.match(ran.err, /1,000 of 3,000 bytes came/);
  ran = await run("owner/model@abc");
  assert.notEqual(ran.code, 0);
  assert.match(ran.err, /^usage: gguf_fetch\.py /);
  ok("gguf_fetch.py fails where a file that must be there is not, and where one never comes whole");

  // T.517: an original with PyTorch's pickle alone (LINE's 1.7B instruction-sft): read without PyTorch and written as
  // model.safetensors. The pickle is made here by its opcodes: an OrderedDict of two float16 tensors on one storage,
  // with the attribute _metadata that torch.save gives it, as LINE's has them
  const made = placed("pickle.bin");
  const maker = spawn(python, ["-c", `
import io, struct, sys, zipfile
import numpy as np
text = lambda s: b"X" + struct.pack("<I", len(s.encode())) + s.encode()
number = lambda n: b"J" + struct.pack("<i", n)
def tensor(offset, size, stride):
    return (b"ctorch._utils\\n_rebuild_tensor_v2\\n(" + b"(" + text("storage") + b"ctorch\\nHalfStorage\\n" + text("0") + text("cpu")
            + number(6) + b"tQ" + number(offset) + b"(" + b"".join(map(number, size)) + b"t(" + b"".join(map(number, stride))
            + b"t\\x89ccollections\\nOrderedDict\\n)RtR")
pickle = (b"\\x80\\x02ccollections\\nOrderedDict\\n)R(" + text("a.weight") + tensor(0, (2, 2), (2, 1)) + text("b.bias")
          + tensor(4, (2,), (1,)) + b"u}" + text("_metadata") + b"}sb.")
out = io.BytesIO()
with zipfile.ZipFile(out, "w") as archive:
    archive.writestr("archive/data.pkl", pickle)
    archive.writestr("archive/data/0", np.arange(6, dtype=np.float16).tobytes())
open(sys.argv[1], "wb").write(out.getvalue())
`, made]);
  assert.equal(await new Promise((resolve) => maker.on("close", resolve)), 0, "the made-up pickle");
  serving({
    [at("owner/model", "abc", "config.json")]: [{ body: "{}" }],
    [at("owner/model", "abc", "tokenizer.json")]: [{ body: "{}" }],
    [at("owner/model", "abc", "pytorch_model.bin")]: [{ body: fs.readFileSync(made) }],
    [at("other/model-GGUF", "def", "model.gguf")]: [{ body: GGUF }],
  });
  ran = await run("owner/model@abc", "other/model-GGUF@def/model.gguf", placed("gguf-pickle"));
  assert.equal(ran.code, 0, ran.err);
  assert.deepEqual(folder("gguf-pickle"), ["config.json", "model.gguf", "model.safetensors", "tokenizer.json"], "the pickle gone, its safetensors there");
  const written = fs.readFileSync(placed("gguf-pickle/model.safetensors"));
  const length = Number(written.readBigUInt64LE(0));
  const header = JSON.parse(written.subarray(8, 8 + length).toString());
  assert.deepEqual(header["a.weight"], { dtype: "F16", shape: [2, 2], data_offsets: [0, 8] });
  assert.deepEqual(header["b.bias"], { dtype: "F16", shape: [2], data_offsets: [8, 12] });
  const values = Array.from(new Uint16Array(written.buffer.slice(written.byteOffset + 8 + length, written.byteOffset + written.length)));
  assert.deepEqual(values, [0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500], "0 to 3, then 4 and 5 of the storage's offset 4, as float16");
  ok("gguf_fetch.py writes an original of PyTorch's pickle alone as model.safetensors");
}

for (const socket of sockets) socket.destroy();
await new Promise((resolve) => server.close(resolve));
fs.rmSync(room, { recursive: true, force: true });
console.log(`fetching-check: ${passed} passed`);
