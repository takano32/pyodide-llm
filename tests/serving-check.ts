// tests/serving-check.ts (T.391.6)
// The two things every tool that drives a browser has in one place, alone and with no browser: the server of the
// site (tests/serving.ts) and the end of the tool (closed() and leave() of tests/leave.ts). The site is a made-up
// folder under .tmp/ whose path has a space and Japanese in it, the requests are Node's own (written as they are, so
// that ".." reaches the server), and the browser that never closes is a function whose promise never settles.
//
//   node tests/serving-check.ts
//
// What is seen of the server: each kind of file with its type and length, a folder's index.html, what is not there,
// what is not under the base, every spelling of a path that leaves the folder, a path that cannot be decoded, a HEAD,
// a Range (not read), the headers it does not send, names with a space and Japanese, the tool's own say (a status, a
// dropped line, another body; what it is told), another base, two servers at once, a folder given as a URL, and the
// port free after close() with lines still open. Of the end: a close that returns, one that never does (given up on at the limit), one that
// throws, and a tool whose browser keeps the process alive ending by itself with all it wrote. Of the tools (their
// text): each that imports a browser's driver has no server of its own, closes through closed() and ends with leave(),
// so that the next tool written is told of the one place.
//
// What this does not see: a browser (the tools themselves, on this machine's Chrome and in CI), Windows' paths (CI).
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import { execFile } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { BASE, DIST, TYPES, serving } from "./serving.ts";
import { closed } from "./leave.ts";

const here = fileURLToPath(new URL(".", import.meta.url));
const room = path.join(here, "..", ".tmp", `serving check 試験 ${process.pid}`);
fs.rmSync(room, { recursive: true, force: true });
const folder = path.join(room, "site");
const FILES = {
  "index.html": "<!doctype html><title>the page</title>",
  "a.js": "export const a = 1;\n",
  "b.mjs": "export const b = 2;\n",
  "c.css": "body { margin: 0 }\n",
  "d.py": "print('d')\n",
  "k.wasm": Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]),
  "data.bin": Buffer.alloc(300000, 7),
  "benchmark/index.html": "<!doctype html><title>the benchmark</title>",
  "_astro/worker.abc123.js": "const LINE = 1e-3;\n",
  "名前 と空白.txt": "日本語の名前\n",
};
for (const [name, data] of Object.entries(FILES)) {
  fs.mkdirSync(path.dirname(path.join(folder, name)), { recursive: true });
  fs.writeFileSync(path.join(folder, name), data);
}
const SECRET = "what is beside the site, and not of it";
fs.writeFileSync(path.join(room, "secret.txt"), SECRET);
fs.writeFileSync(path.join(room, "site-too.txt"), SECRET);  // a neighbour whose name begins as the folder's does
fs.mkdirSync(path.join(room, "sitex"));
fs.writeFileSync(path.join(room, "sitex", "secret.txt"), SECRET);

let passed = 0;
const ok = (name) => { passed++; console.log(`ok: ${name}`); };
// a request written as it is (fetch() would fold ".." away before it left)
const asked = (port, target, { method = "GET", headers = {} } = {}) => new Promise((resolve, reject) => {
  const request = http.request({ host: "127.0.0.1", port, path: target, method, headers, agent: false }, (res) => {
    const parts = [];
    res.on("data", (part) => parts.push(part)).on("end", () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(parts) }));
    res.on("error", reject);
  });
  request.on("error", reject).end();
});
const enc = (name) => name.split("/").map(encodeURIComponent).join("/");

const site = await serving(folder);
try {
  assert.equal(site.url, `http://localhost:${site.port}/pyodide-llm/`);
  assert.equal(BASE, "/pyodide-llm/");
  assert.equal(DIST.href, new URL("../dist/", import.meta.url).href, "the default folder is this checkout's dist/, as a URL");
  ok("the address is localhost, the port the system's, under the site's base");

  // ---- the files
  for (const [name, data] of Object.entries(FILES)) {
    const got = await asked(site.port, `/pyodide-llm/${enc(name)}`);
    assert.equal(got.status, 200, name);
    assert.ok(got.body.equals(Buffer.from(data)), `${name}: its bytes`);
    assert.equal(got.headers["content-type"], TYPES[path.extname(name)] ?? "application/octet-stream", name);
    assert.equal(got.headers["content-length"], String(Buffer.from(data).length), name);
  }
  assert.deepEqual(TYPES, { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
    ".py": "text/plain", ".wasm": "application/wasm", ".zip": "application/x-zip-compressed" }, "the types, written a second time");
  ok(`${Object.keys(FILES).length} files with their bytes, types and lengths (a name with a space and Japanese among them)`);

  for (const target of ["/pyodide-llm/", "/pyodide-llm/?model=tiny-lm&bench=1", "/pyodide-llm/index.html?v=1#here"]) {
    const got = await asked(site.port, target);
    assert.equal(got.status, 200, target);
    assert.equal(got.body.toString(), FILES["index.html"], target);
    assert.equal(got.headers["content-type"], "text/html; charset=utf-8");
  }
  for (const target of ["/pyodide-llm/benchmark/", "/pyodide-llm/benchmark", "/pyodide-llm/benchmark/?run=all"]) {
    const got = await asked(site.port, target);
    assert.equal(got.status, 200, target);
    assert.equal(got.body.toString(), FILES["benchmark/index.html"], target);
  }
  ok("the base and a folder are their index.html, whatever the query");

  // ---- what is not there
  for (const target of ["/pyodide-llm/none.js", "/pyodide-llm/benchmark/none.html", "/pyodide-llm/_astro/", "/pyodide-llm/a.js/", "/pyodide-llm/a.js/x",
    "/pyodide-llm/a%00.js", "/pyodide-llm/index.html%00"]) {
    const got = await asked(site.port, target);
    assert.equal(got.status, 404, target);
    assert.equal(got.body.length, 0, target);
  }
  ok("404 with no body for a file that is not there (a folder without an index.html, a file taken as a folder, a zero byte)");

  for (const target of ["/", "/index.html", "/a.js", "/pyodide-llm", "/pyodide-llmx/a.js", "/other/pyodide-llm/a.js", "/favicon.ico", "//pyodide-llm/a.js",
    "/PYODIDE-LLM/a.js"]) {
    const got = await asked(site.port, target);
    assert.equal(got.status, 404, target);
  }
  ok("404 for what is not under the base");

  // ---- a path that leaves the folder, however it is spelt
  const LEAVING = ["/pyodide-llm/../secret.txt", "/pyodide-llm/%2e%2e/secret.txt", "/pyodide-llm/..%2fsecret.txt", "/pyodide-llm/..%2Fsecret.txt",
    "/pyodide-llm/%2e%2e%2fsecret.txt", "/pyodide-llm/benchmark/..%2f..%2fsecret.txt", "/pyodide-llm/benchmark/%2e%2e%2f%2e%2e%2fsecret.txt",
    "/pyodide-llm/..%5csecret.txt", "/pyodide-llm/%2fsecret.txt", "/pyodide-llm/..%2fsite-too.txt", "/pyodide-llm/..%2fsitex%2fsecret.txt",
    "/pyodide-llm/..%2fsite%2f..%2fsecret.txt", "/pyodide-llm/..%2f..%2f..%2f..%2f..%2f..%2f..%2f..%2f..%2fetc%2fpasswd",
    `/pyodide-llm/${encodeURIComponent(path.join(room, "secret.txt"))}`, "/pyodide-llm/..", "/pyodide-llm/..%2f", "/pyodide-llm/%2e%2e%2f"];
  for (const target of LEAVING) {
    const got = await asked(site.port, target);
    assert.equal(got.status, 404, `${target} was answered ${got.status}`);
    assert.equal(got.body.length, 0, target);
  }
  // (and one that goes out and comes back in is the file it names)
  assert.equal((await asked(site.port, "/pyodide-llm/benchmark/..%2fa.js")).body.toString(), FILES["a.js"]);
  ok(`${LEAVING.length} spellings of a path that leaves the folder: 404, and none of what is beside it`);

  for (const target of ["/pyodide-llm/%zz", "/pyodide-llm/a%2", "/pyodide-llm/%E5%90"]) {
    assert.equal((await asked(site.port, target)).status, 400, target);
  }
  assert.equal((await asked(site.port, "/pyodide-llm/a.js")).status, 200, "and the server goes on");
  ok("400 for a path that cannot be decoded, and the server goes on");

  // ---- the methods and the headers
  const head = await asked(site.port, "/pyodide-llm/data.bin", { method: "HEAD" });
  assert.deepEqual([head.status, head.headers["content-length"], head.body.length], [200, "300000", 0]);
  assert.equal((await asked(site.port, "/pyodide-llm/none", { method: "HEAD" })).status, 404);
  ok("a HEAD says the length and sends no body");

  const ranged = await asked(site.port, "/pyodide-llm/data.bin", { headers: { Range: "bytes=10-19" } });
  assert.deepEqual([ranged.status, ranged.body.length, ranged.headers["content-range"], ranged.headers["accept-ranges"]], [200, 300000, undefined, undefined]);
  ok("a Range is not read: the whole file, 200");

  const plain = await asked(site.port, "/pyodide-llm/a.js");
  assert.deepEqual(Object.keys(plain.headers).filter((name) => !["date", "connection", "keep-alive"].includes(name)).sort(), ["content-length", "content-type"]);
  ok("no header but the type and the length: no COOP, no COEP, nothing about keeping a file");
} finally {
  await site.close();
}

// ---- the tool's own say
{
  const told = [];
  let rule = () => undefined;
  const mine = await serving(folder, { instead: (request) => { told.push(request); return rule(request); } });
  try {
    assert.equal((await asked(mine.port, "/pyodide-llm/a.js?v=1")).body.toString(), FILES["a.js"]);
    await asked(mine.port, "/pyodide-llm/none.js", { method: "HEAD" });
    await asked(mine.port, "/pyodide-llm/benchmark/");
    await asked(mine.port, "/pyodide-llm/");
    await asked(mine.port, "/elsewhere/a.js");
    await asked(mine.port, "/pyodide-llm/%zz");
    assert.deepEqual(told, [{ name: "a.js", method: "GET", file: path.join(folder, "a.js") }, { name: "none.js", method: "HEAD", file: undefined },
      { name: "benchmark/", method: "GET", file: path.join(folder, "benchmark", "index.html") }, { name: "", method: "GET", file: path.join(folder, "index.html") }]);
    ok("it is told the name under the base, the method and the file there is; nothing of what is not under the base");

    rule = ({ name }) => (name === "a.js" ? 500 : name === "none.js" ? 204 : undefined);
    const failed = await asked(mine.port, "/pyodide-llm/a.js");
    assert.deepEqual([failed.status, failed.body.length], [500, 0]);
    assert.equal((await asked(mine.port, "/pyodide-llm/none.js")).status, 204, "a status for what is not there either");
    assert.equal((await asked(mine.port, "/pyodide-llm/b.mjs")).status, 200);
    rule = ({ name, method }) => (name === "a.js" && method === "HEAD" ? 404 : undefined);
    assert.equal((await asked(mine.port, "/pyodide-llm/a.js", { method: "HEAD" })).status, 404);
    assert.equal((await asked(mine.port, "/pyodide-llm/a.js")).status, 200);
    ok("a status answers in the file's place, by name and by method");

    rule = ({ name }) => (name === "a.js" ? "drop" : undefined);
    await assert.rejects(asked(mine.port, "/pyodide-llm/a.js"), /socket hang up|ECONNRESET/);
    assert.equal((await asked(mine.port, "/pyodide-llm/b.mjs")).status, 200, "the server goes on");
    ok("\"drop\" ends the line without an answer");

    rule = ({ name, file }) => (name.startsWith("_astro/worker.") ? { body: fs.readFileSync(file, "utf8").replace("1e-3", "1e-12"), type: ".js" } : undefined);
    const other = await asked(mine.port, "/pyodide-llm/_astro/worker.abc123.js");
    assert.deepEqual([other.status, other.body.toString(), other.headers["content-type"], other.headers["content-length"]], [200, "const LINE = 1e-12;\n", "text/javascript", "20"]);
    rule = () => ({ body: "何か" });
    const made = await asked(mine.port, "/pyodide-llm/made.css");
    assert.deepEqual([made.status, made.body.toString(), made.headers["content-type"], made.headers["content-length"]], [200, "何か", "text/css", "6"]);
    assert.equal((await asked(mine.port, "/pyodide-llm/made.css", { method: "HEAD" })).body.length, 0);
    rule = () => ({ body: "<p>", type: ".html" });
    assert.equal((await asked(mine.port, "/pyodide-llm/a.js")).headers["content-type"], "text/html; charset=utf-8", "the type it names, not the file's");
    ok("another body answers 200 with its type and its length in bytes");
  } finally {
    await mine.close();
  }
}

// ---- another base, two at once, and the end
{
  const root = await serving(folder, { base: "/" }), under = await serving(folder, { base: "/a/b/" });
  assert.notEqual(root.port, under.port);
  assert.equal(root.url, `http://localhost:${root.port}/`);
  assert.equal((await asked(root.port, "/a.js")).body.toString(), FILES["a.js"]);
  assert.equal((await asked(root.port, "/pyodide-llm/a.js")).status, 404);
  assert.equal((await asked(root.port, "/..%2fsecret.txt")).status, 404);
  assert.equal((await asked(under.port, "/a/b/benchmark/")).body.toString(), FILES["benchmark/index.html"]);
  assert.equal((await asked(under.port, "/a/a.js")).status, 404);
  ok("another base, and two servers at once");

  // the folder as a file: URL (as the default, dist/, is): read as a path, not as the URL's pathname ("%20", "%E8…")
  const byUrl = await serving(pathToFileURL(folder + path.sep));
  assert.ok(pathToFileURL(folder).pathname.includes("%20") && pathToFileURL(folder).pathname.includes("%E8"), "the made-up folder's URL is not its path");
  assert.equal((await asked(byUrl.port, "/pyodide-llm/benchmark/")).body.toString(), FILES["benchmark/index.html"]);
  assert.equal((await asked(byUrl.port, "/pyodide-llm/..%2fsecret.txt")).status, 404);
  await byUrl.close();
  ok("a folder given as a file: URL whose path has a space and Japanese");

  // a line a browser left open does not keep the port: one that is idle after an answer (keep-alive), and one in
  // the middle of a request (which a server's own close() waits for)
  const idle = net.connect(under.port, "127.0.0.1"), busy = net.connect(under.port, "127.0.0.1");
  await Promise.all([idle, busy].map((line) => new Promise((resolve) => line.once("connect", resolve))));
  idle.write("GET /a/b/a.js HTTP/1.1\r\nHost: localhost\r\n\r\n");
  await new Promise((resolve) => idle.once("data", resolve));
  busy.write("GET /a/b/data.bin HTTP/1.1\r\nHost: localhost\r\n");  // (its headers never end)
  await new Promise((resolve) => setTimeout(resolve, 50));
  const ended = Promise.all([idle, busy].map((line) => new Promise((resolve) => line.on("error", () => {}).once("close", resolve))));
  const began = Date.now();
  await under.close();
  await ended;
  assert.ok(Date.now() - began < 2000, `close() took ${Date.now() - began} ms with a line open`);
  await assert.rejects(asked(under.port, "/a/b/a.js"), /ECONNREFUSED/);
  const again = http.createServer();
  await new Promise((resolve, reject) => again.once("error", reject).listen(under.port, resolve));
  await new Promise((resolve) => again.close(resolve));
  await root.close();
  await root.close();  // (and a second close is no error)
  ok("after close() the port is free at once, with a line still open; a second close is nothing");
}

// ---- the end of a tool: closed()
{
  const timers = () => process.getActiveResourcesInfo().filter((kind) => kind === "Timeout").length;
  const before = timers();
  let calls = 0;
  assert.equal(await closed(async () => { calls++; }), true);
  assert.equal(await closed(() => { calls++; }), true, "a close that is no promise");
  assert.equal(calls, 2);
  assert.equal(timers(), before, "no timer is left to keep the process waiting 15 s");
  ok("a close that returns: true, and nothing left behind");

  let began = Date.now();
  assert.equal(await closed(() => new Promise(() => {}), 0.05), false);
  const took = Date.now() - began;
  assert.ok(took >= 40 && took < 1000, `given up on after ${took} ms of 50`);
  assert.equal(timers(), before);
  began = Date.now();
  assert.equal(await closed(() => new Promise((resolve) => setTimeout(resolve, 30)), 5), true);
  assert.ok(Date.now() - began < 1000, "a close that returns before the limit does not wait for the limit");
  ok("a close that never returns is given up on at the limit: false");

  assert.equal(await closed(() => Promise.reject(new Error("the browser is gone"))), false);
  assert.equal(await closed(() => { throw new Error("no browser"); }), false);
  assert.equal(await closed(undefined), false, "nothing to close (a browser that never started)");
  ok("a close that throws is false, and nothing is thrown");

  // a tool whose browser never closes and keeps the process alive: it ends by itself, with all it wrote
  const tool = path.join(room, "tool.mjs");
  fs.writeFileSync(tool, `import { closed, leave } from ${JSON.stringify(new URL("./leave.ts", import.meta.url).href)};
setInterval(() => {}, 1000);                                   // what a browser that did not close leaves behind
process.stdout.write("x".repeat(${4 * 2 ** 20}) + "\\n");            // more than a pipe takes at once
const done = await closed(() => new Promise(() => {}), 0.2);
console.log(done ? "closed" : "the browser did not close; going on");
console.error("to stderr");
await leave(3);
`);
  began = Date.now();
  const child = await new Promise((resolve) => execFile(process.execPath, [tool], { maxBuffer: 64 * 2 ** 20, timeout: 30000 },
    (error, stdout, stderr) => resolve({ code: error?.code, signal: error?.signal, stdout, stderr })));
  assert.deepEqual([child.code, child.signal ?? null, child.stderr], [3, null, "to stderr\n"]);
  assert.equal(child.stdout.length, 4 * 2 ** 20 + 1 + "the browser did not close; going on\n".length, "every byte it wrote");
  assert.ok(child.stdout.endsWith("\nthe browser did not close; going on\n"));
  assert.ok(Date.now() - began < 20000);
  ok("a tool whose browser never closes ends by itself, with its exit code and every byte it wrote");
}

// ---- the tools: every one that drives a browser has its server and its end from the one place
{
  // (tests/gpu-check/engines.ts is not among them: its server serves the sources and its own cases, and its browser's
  // close is its own line, which only CI's gpu-prompt.yml runs)
  const OWN_SERVER = { "slow-check.ts": "a proxy that throttles a deployed site: another thing, and the only one" };
  const tools = fs.readdirSync(here).filter((name) => name.endsWith(".ts") && /^import .* from "(playwright-core|selenium-webdriver)";$/m.test(fs.readFileSync(path.join(here, name), "utf8")));
  assert.ok(tools.length >= 10, `only ${tools.length} tools that drive a browser were found: ${tools}`);
  const wrong = [];
  for (const name of tools) {
    const text = fs.readFileSync(path.join(here, name), "utf8");
    if (/\bcreateServer\(/.test(text) && !OWN_SERVER[name]) wrong.push(`${name} has a server of its own: serve the site with serving() of tests/serving.ts`);
    if (!/\bcreateServer\(/.test(text) && OWN_SERVER[name]) wrong.push(`${name} has no server of its own any more: take it out of OWN_SERVER`);
    if (/\/pyodide-llm\/["`]/.test(text.replace(/https:\/\/takano32\.github\.io\/pyodide-llm\//g, ""))) wrong.push(`${name} writes the site's base itself: it is serving()'s (the address is its url)`);
    if (/os\.tmpdir\(\)/.test(text)) wrong.push(`${name} makes its folders in the system's temp folder: scratch() of tests/leave.ts puts them under .tmp/ and takes them away (T.512)`);
    for (const [at, line] of text.split("\n").entries()) {
      const closes = !line.trimStart().startsWith("//") && /\b(browser|context|driver)\.(close|quit)\(\)/.exec(line);
      if (closes && !line.includes(`closed(() => ${closes[0]})`)) wrong.push(`${name}:${at + 1} waits on ${closes[0]} itself: await closed(() => ${closes[0]}) of tests/leave.ts (T.141)`);
    }
    if (!/^import \{ closed, leave(, scratch)? \} from "\.\/leave\.ts";$/m.test(text)) wrong.push(`${name} does not take closed and leave of tests/leave.ts: a tool ends with await leave(code)`);
    const last = text.trimEnd().split("\n").at(-1);
    if (!/^await leave\(.*\);$/.test(last)) wrong.push(`${name} ends with ${JSON.stringify(last)}, not with await leave(code): what a browser left behind would keep the process`);
  }
  assert.deepEqual(wrong, []);
  ok(`${tools.length} tools that drive a browser: no server of their own (but ${Object.keys(OWN_SERVER)}), every close through closed(), every end leave()`);
}

fs.rmSync(room, { recursive: true, force: true });
console.log(`serving-check: ok (${passed} checks)`);
