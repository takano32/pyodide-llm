// tests/fetching-check.mjs (T357.6): the Node tools' one place for fetching (tests/fetching.mjs), by itself. Nothing
// leaves this machine: the server is one of node:http on 127.0.0.1, which can do what a stand-in of fetch cannot (close
// a connection before the length it promised, cut a chunked body, never answer), and tests/fakes/fetch.mjs is the
// network where the answer itself is wrong in a way no real server of node:http sends (a body that ends in order and
// short).
//
//   node tests/fetching-check.mjs        (Node only, under a second)
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { madeUpNetwork } from "./fakes/fetch.mjs";
import { Answered, download, fetched } from "./fetching.mjs";

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
  // (what huggingface.co did to six of 80 GGUFs in one run, T307: Node's fetch fails here by itself)
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
  serving({ "/file": [{ never: true }, { body: WHOLE }] });
  const target = placed("silent/file.bin");
  await download(`${base}/file`, target, { timeout: 200 });
  assert.equal(fs.readFileSync(target, "latin1"), WHOLE);
  assert.equal(asked.length, 2);
  ok("a request that got no answer in its time is asked for again");
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
  let network = answering([saying("x".repeat(40), { "Content-Length": "100" }), saying(WHOLE, { "Content-Length": "100" })]);
  const target = placed("lied/file.bin");
  await download("https://hub.test/file", target, { fetch: network.fetch });
  assert.equal(fs.readFileSync(target, "latin1"), WHOLE);
  assert.equal(network.requests.length, 2, "40 bytes of a promised 100 are not the file");
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

for (const socket of sockets) socket.destroy();
await new Promise((resolve) => server.close(resolve));
fs.rmSync(room, { recursive: true, force: true });
console.log(`fetching-check: ${passed} passed`);
