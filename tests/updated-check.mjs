// T367.4: what says "the site was updated while this page was open" (src/runtime/updated.js and built.js's built()),
// alone and with stand-ins: a fetch that answers 404, one of the 500s, drops, is offline, never answers.
//
//   node tests/updated-check.mjs
//
// It sees: the sentence whole; which addresses are files of this site's build; that only a 404 of one of them is
// "gone" (no other status, no failed fetch, no silence, no model's part, no other site), asked by HEAD past the
// caches and once; that a failed import or a worker that did not start is "updated" only where its namer is gone, and
// is otherwise the failure it was; that built() tells a 404 of a built file so and everything else as before; and, in
// the sources, that every file fetched later is in the table below with what becomes of its failure (a new one that
// is not there falls here), that built files are fetched by built() alone, and that nothing of the token's path
// knows of any of this.
// It does not see: a browser (what a failed import or worker looks like there, the service worker's copies, a real
// deployment): tests/updated-in-browser.mjs.
import assert from "node:assert/strict";
import fs from "node:fs";
import { workerHarness } from "./worker-harness.mjs";
import { UPDATED, updated, isUpdated, saidOf, ofThisBuild, answeredGone, gone, named, unstarted } from "../src/runtime/updated.js";

// ---- the sentence, whole (the owner's: T367's decision 4)
assert.equal(UPDATED, "The site was updated while this page was open. Reload the page to go on.");
{
  const cause = new TypeError("Failed to fetch dynamically imported module");
  const error = updated(cause);
  // an Error by name: worker.js's report says such a one by its message alone, and String() has no other name in it
  assert.equal(error.name, "Error");
  assert.equal(error.message, UPDATED);
  assert.equal(String(error), `Error: ${UPDATED}`);
  assert.equal(error.cause, cause);
  assert.ok(isUpdated(error));
  for (const other of [new Error(UPDATED), new Error("x"), undefined, null, "updated", { updated: "yes" }]) assert.ok(!isUpdated(other));
  assert.equal(saidOf(error, "Error: something"), UPDATED);
  assert.equal(saidOf(new Error("x"), "Error: x"), "Error: x");
}
console.log("updated-check: the sentence is the owner's, whole, and an error that says it says nothing else");

// ---- which addresses are files of this site's build
const here = { origin: "https://takano32.github.io" };
const SITE = "https://takano32.github.io/pyodide-llm/";
const WORKER = `${SITE}_astro/worker.BwFFQ8AO.js`;
for (const [address, is] of [
  [WORKER, true],
  [`${SITE}_astro/converter.CreA44do.zip`, true],
  [`${SITE}_astro/simdkernel_plain.DfCGpm7G.wasm?x=1`, true],
  [new URL(WORKER), true],
  [`${SITE}models/tiny-lm.bin.000`, false],          // a model's part
  [`${SITE}models/tok512.bin`, false],
  [SITE, false],                                      // the page
  [`${SITE}coi.js`, false],
  ["https://cdn.jsdelivr.net/pyodide/v314.0.7/full/_astro/pyodide.mjs", false],  // another site, whatever its path
  ["https://huggingface.co/a/b/resolve/main/_astro/model.safetensors", false],
  ["http://takano32.github.io/pyodide-llm/_astro/worker.js", false],             // another origin (the scheme)
  ["http://localhost:8080/src/runtime/built/converter.zip", false],              // a development server's source
  ["file:///home/someone/pyodide-llm/src/runtime/worker.js", false],
  ["not an address", false], [undefined, false], ["", false],
]) {
  assert.equal(ofThisBuild(address, here), is, `${address} is ${is ? "" : "not "}a file of this build`);
}
assert.equal(ofThisBuild(WORKER, undefined), false, "nowhere (no location): nothing is of this build");
assert.equal(ofThisBuild(WORKER, { search: "" }), false, "a location without an origin: nothing is of this build");
assert.equal(ofThisBuild("http://localhost:8080/pyodide-llm/_astro/worker.x.js", { origin: "http://localhost:8080" }), true);
for (const [address, status, is] of [[WORKER, 404, true], [WORKER, 200, false], [WORKER, 500, false], [WORKER, 503, false], [WORKER, 403, false],
  [WORKER, 410, false], [WORKER, 0, false], [`${SITE}models/tiny-lm.bin.000`, 404, false], ["https://cdn.jsdelivr.net/x/_astro/y.js", 404, false]]) {
  assert.equal(answeredGone(address, status, here), is, `${status} for ${address}`);
}
console.log("updated-check: a file of this build is one of this origin under _astro/, and only its 404 says it is gone");

// ---- gone(): the server's 404 to a HEAD, and nothing else
const asked = [];
const answering = (answer) => async (address, init) => {
  asked.push({ address, ...init });
  return answer(address, init);
};
const status = (code) => answering(() => new Response(null, { status: code }));
const probe = (fetch, more = {}) => ({ fetch, here, ...more });
{
  asked.length = 0;
  assert.equal(await gone(WORKER, probe(status(404))), true);
  assert.equal(asked.length, 1, "asked once");
  assert.equal(asked[0].address, WORKER);
  assert.equal(asked[0].method, "HEAD", "by HEAD: the service worker keeps and answers GETs only, and nothing is downloaded");
  assert.equal(asked[0].cache, "no-store", "past the browser's own cache: a file it holds may be gone on the server");
  assert.ok(asked[0].signal instanceof AbortSignal);
  for (const code of [200, 204, 304, 400, 401, 403, 410, 429, 500, 502, 503]) {
    assert.equal(await gone(WORKER, probe(status(code))), false, `${code} is not "gone"`);
  }
  // a line that dropped, a page that is offline, a request something blocked: the fetch fails
  for (const failure of [new TypeError("Failed to fetch"), new TypeError("NetworkError when attempting to fetch resource."), new TypeError("Load failed"),
    new DOMException("blocked", "SecurityError")]) {
    assert.equal(await gone(WORKER, probe(answering(() => { throw failure; }))), false, `${failure.message} is not "gone"`);
  }
  // no answer at all: given up, and not called an update (the abort reaches the fetch, so nothing is left asking)
  let aborted = false;
  const silent = (address, init) => new Promise((resolve, reject) => init.signal.addEventListener("abort", () => { aborted = true; reject(new DOMException("aborted", "AbortError")); }));
  const began = performance.now();
  const waited = await Promise.race([gone(WORKER, probe(silent, { seconds: 0.05 })), new Promise((resolve) => setTimeout(() => resolve("still asking"), 2000))]);
  assert.equal(waited, false, "a HEAD that is not answered is given up, and is no update");
  assert.ok(aborted && performance.now() - began < 2000, "and its fetch is ended");
  // the limit it is given when no one says: ten seconds (a page waits that long, at most, before it shows the words it had)
  const delays = [];
  const setTimeoutWas = globalThis.setTimeout;
  globalThis.setTimeout = (fn, ms, ...rest) => (delays.push(ms), setTimeoutWas(fn, ms, ...rest));
  try {
    await gone(WORKER, probe(status(404)));
  } finally {
    globalThis.setTimeout = setTimeoutWas;
  }
  assert.deepEqual(delays, [10000], "a HEAD that is not answered is given up after ten seconds");
  // what is not of this build is not asked for at all: a model's part that is 404 is a missing part, as it was
  asked.length = 0;
  for (const address of [`${SITE}models/tiny-lm.bin.000`, "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs", "file:///x/src/runtime/worker.js", SITE]) {
    assert.equal(await gone(address, probe(status(404))), false, `${address} is not this build's to be gone`);
  }
  assert.equal(asked.length, 0, "nothing was asked of a server for them");
  // and by default it is the global fetch and location: a file: module (Node, the checks' vm) is never "updated"
  assert.equal(await gone(import.meta.url), false);
}
console.log("updated-check: gone() is the server's 404 to one HEAD past the caches; no other status, failure or silence is");

// ---- named(): a dynamic import by the module that names it
{
  const module = { kept: true };
  asked.length = 0;
  assert.equal(await named(WORKER, async () => module, probe(status(404))), module);
  assert.equal(asked.length, 0, "nothing is asked where nothing failed");
  const failure = new TypeError("Failed to fetch dynamically imported module: https://takano32.github.io/pyodide-llm/_astro/kept.BDa9_jbn.js");
  const failing = async () => { throw failure; };
  const told = await named(WORKER, failing, probe(status(404))).catch((error) => error);
  assert.ok(isUpdated(told) && told.message === UPDATED && told.cause === failure, "the namer is gone: the site was updated");
  assert.equal(asked.length, 1);
  assert.equal(asked[0].address, WORKER, "the namer is what is asked for");
  for (const [what, fetch] of [["the namer is there", status(200)], ["the server fails", status(500)], ["the server is busy", status(503)],
    ["the line dropped", answering(() => { throw new TypeError("Failed to fetch"); })]]) {
    assert.equal(await named(WORKER, failing, probe(fetch)).catch((error) => error), failure, `${what}: the failure it was`);
  }
  assert.equal(await named("file:///x/src/runtime/worker/convert.js", failing, probe(status(404))).catch((error) => error), failure, "not a build (a development server, Node): the failure it was");
  // a function that throws before it returns a promise is a failure like any other
  assert.equal(await named(WORKER, () => { throw failure; }, probe(status(200))).catch((error) => error), failure);
  // one that is updated() already (built()'s) is not asked about again
  asked.length = 0;
  const already = updated();
  assert.equal(await named(WORKER, async () => { throw already; }, probe(status(200))).catch((error) => error), already);
  assert.equal(asked.length, 0);
}
console.log("updated-check: a failed import is the site's update only where the file that names it is gone");

// ---- unstarted(): a worker's error event, for the module that made the worker
{
  asked.length = 0;
  // a script that threw: the event says what (an ErrorEvent), and that is no missing file whatever the server says
  const threw = await unstarted({ message: "Uncaught ReferenceError: x is not defined" }, WORKER, "the worker failed: Uncaught ReferenceError: x is not defined", probe(status(404)));
  assert.ok(!isUpdated(threw));
  assert.equal(threw.message, "the worker failed: Uncaught ReferenceError: x is not defined");
  assert.equal(asked.length, 0, "and nothing is asked");
  // one that did not load: an event without a message
  for (const event of [{}, { message: undefined }, { message: "" }, undefined]) {
    const told = await unstarted(event, WORKER, "the worker failed: error", probe(status(404)));
    assert.ok(isUpdated(told) && told.message === UPDATED);
  }
  for (const fetch of [status(200), status(500), answering(() => { throw new TypeError("Failed to fetch"); })]) {
    const told = await unstarted({}, WORKER, "the worker failed: error", probe(fetch));
    assert.ok(!isUpdated(told));
    assert.equal(told.message, "the worker failed: error");
  }
}
console.log("updated-check: a worker that did not start is the site's update only where it says nothing and its namer is gone");

// ---- built(): a built file's own answer. Its table is of URLs beside the module, so the module (and updated.js) is
// read from a folder named _astro, as a bundle of it is served, with a location of that origin
{
  const runtime = new URL("../src/runtime/", import.meta.url);
  const folder = new URL(`../.tmp/updated-check-${process.pid}/`, import.meta.url), astro = new URL("_astro/", folder);
  fs.mkdirSync(astro, { recursive: true });
  const globals = { fetch: globalThis.fetch, location: Object.getOwnPropertyDescriptor(globalThis, "location") };
  try {
    for (const name of ["built.js", "updated.js"]) fs.copyFileSync(new URL(name, runtime), new URL(name, astro));
    const served = await import(new URL("built.js", astro).href);
    const source = await import(new URL("built.js", runtime).href);
    const fetched = [];
    const answer = (make) => { globalThis.fetch = async (address, init) => { fetched.push({ address: String(address), ...init }); return make(); }; };
    const at = (origin) => Object.defineProperty(globalThis, "location", { configurable: true, value: origin === undefined ? undefined : { origin } });
    at("null");  // (a file: URL's origin)
    const signal = new AbortController().signal;
    answer(() => new Response("the bytes", { status: 200 }));
    assert.equal(await (await served.built("converter.zip", signal)).text(), "the bytes");
    assert.equal(fetched.at(-1).address, String(served.BUILT["converter.zip"]));
    assert.equal(fetched.at(-1).signal, signal, "a load that is cancelled cancels the fetch");
    assert.equal(fetched.at(-1).method, undefined, "a GET, which the service worker keeps");
    answer(() => new Response("", { status: 404 }));
    for (const name of Object.keys(served.BUILT)) {
      const told = await served.built(name).catch((error) => error);
      assert.ok(isUpdated(told) && told.message === UPDATED, `${name} gone from a build: the site was updated`);
    }
    for (const code of [500, 503, 403, 410]) {
      answer(() => new Response("", { status: code }));
      const told = await served.built("converter.zip").catch((error) => error);
      assert.ok(!isUpdated(told));
      assert.equal(told.message, `Could not fetch converter.zip: ${code}`);
    }
    const dropped = new TypeError("Failed to fetch");
    globalThis.fetch = async () => { throw dropped; };
    assert.equal(await served.built("converter.zip").catch((error) => error), dropped, "a fetch that fails fails as it did");
    // the same 404 where the file is not of a build (the sources as a development server or Node has them), and
    // where it is of another origin than the page: a file that could not be fetched, as before
    answer(() => new Response("", { status: 404 }));
    for (const [module, origin] of [[source, "null"], [served, "https://takano32.github.io"], [served, undefined]]) {
      at(origin);
      const told = await module.built("engine.zip").catch((error) => error);
      assert.ok(!isUpdated(told));
      assert.equal(told.message, "Could not fetch engine.zip: 404");
    }
  } finally {
    globalThis.fetch = globals.fetch;
    if (globals.location) Object.defineProperty(globalThis, "location", globals.location); else delete globalThis.location;
    fs.rmSync(folder, { recursive: true, force: true });
  }
}
console.log("updated-check: built() says a built file's 404 as the update, and any other answer or failure as before");

// ---- the model's worker reports it to the page as the sentence alone (worker.js and its modules in a vm, as
// tests/worker-check.mjs has them: whatever the worker was doing when a file of its build was found gone)
{
  const { run, messages } = workerHarness();
  run("adopted({ llama: { generate: { callKwargs() { throw updated(new TypeError('Failed to fetch dynamically imported module')); } } } });");
  await run("self.onmessage")({ data: { type: "generate", prompt: "Once" } });
  const report = messages.filter((message) => message.type === "error");
  assert.equal(report.length, 1);
  assert.equal(report[0].message, UPDATED, "the worker's report to the page is the sentence, with no name before it");
  assert.ok(!report[0].memory && !report[0].pyodide, "and it is no memory that ran out and no Pyodide that did not load");
  // and a built file that could not be fetched otherwise is reported as it was
  messages.length = 0;
  run("adopted({ llama: { generate: { callKwargs() { throw new Error('Could not fetch converter.zip: 500'); } } } });");
  await run("self.onmessage")({ data: { type: "generate", prompt: "Once" } });
  assert.equal(messages.find((message) => message.type === "error").message, "Could not fetch converter.zip: 500");
}
console.log("updated-check: the model's worker tells the page the sentence alone");

// ---- the sources: every file that is asked for later, and what becomes of its failure.
// said: the visitor is told the site was updated (the failure ends what was asked: a load, a conversion, a section).
// quiet: the engine goes on without the file, as it always did, and nothing is said: a model that answers is not
// interrupted, and the sentence ("Reload the page to go on") would not be true of it.
// A new line here is a decision: a file whose absence ends something is fetched by built(), named() or with
// unstarted() at its worker's error, and one the engine does without says why it is quiet.
const root = new URL("../", import.meta.url);
const LATER = {
  "src/runtime/worker/runtime.js": [
    ['import("../shaders/key.js")', "quiet: without the device's key nothing kept of this device holds, and the GPU is weighed again"],
    ['built("engine.zip").then((res) => res.arrayBuffer());', "said"],
    ["built(name).catch(() => undefined);", "quiet: without simdkernel.so the sampling runs on NumPy"],
    ['import("../forward.js")', "quiet: without forward.js or its kernels the engine runs on NumPy"],
    ["built(name).then((res) => res.arrayBuffer()).catch(() => null)));", "quiet: without the kernels the engine runs on NumPy"],
  ],
  "src/runtime/worker/convert.js": [
    ["built(archive, signal).then((res) => res.arrayBuffer()), archive);", "said"],
    ['named(import.meta.url, () => import("../kept.ts"))', "said"],
  ],
  "src/runtime/worker/weights.js": [
    ['new Worker(new URL("../helper.ts", import.meta.url)', "quiet: a software thread that does not start leaves fewer threads"],
    ['new Worker(new URL("../gpu.js", import.meta.url)', "quiet: a GPU worker that does not start leaves the model on the CPU (forward/gpuside.js)"],
  ],
  "src/runtime/benchmark/sections.js": [
    ["built(name).then((res) => res.arrayBuffer()).catch((error) => unlessUpdated(error) ?? null)));", "said"],
    ['named(import.meta.url, () => import("../forward.js"))', "said"],
    ['new Worker(new URL("../helper.ts", import.meta.url)', "said"],
    ['named(import.meta.url, () => import("../forward.js"))', "said"],
    ["built(name).then((res) => res.arrayBuffer()).catch((error) => unlessUpdated(error) ?? null)));", "said"],
    ['new Worker(new URL("../benchmark/reader.js", import.meta.url)', "said"],
  ],
  "src/runtime/benchmark/gpu/device.js": [
    ['named(import.meta.url, () => import("../../shaders.js"))', "said"],
  ],
  "src/runtime/benchmark/gpu/matvec.js": [
    ['built("simdkernel_plain.wasm")).arrayBuffer());', "said"],
  ],
  "src/page/receiver.ts": [
    ['new Worker(new URL("../runtime/worker.js", import.meta.url)', "said"],
  ],
  "src/benchmark/section.ts": [
    ['new Worker(new URL("../runtime/worker.js", import.meta.url)', "said"],
    ['new Worker(new URL("../runtime/benchmark/gpu.js", import.meta.url)', "said"],
    ['new Worker(new URL("../runtime/benchmark/memory.js", import.meta.url)', "said"],
    ['new Worker(new URL("../runtime/benchmark/sections.js", import.meta.url)', "said"],
    ['new Worker(new URL("../runtime/benchmark/storage.js", import.meta.url)', "said"],
    ['new Worker(new URL("../runtime/benchmark/waiter.js", import.meta.url)', "said"],
  ],
};
const sources = [];
const walk = (folder) => {
  for (const entry of fs.readdirSync(new URL(folder, root), { withFileTypes: true })) {
    if (entry.isDirectory()) { if (entry.name !== "built") walk(`${folder}${entry.name}/`); } else if (/\.(js|ts|astro)$/.test(entry.name)) sources.push(`${folder}${entry.name}`);
  }
};
["src/runtime/", "src/page/", "src/benchmark/", "src/pages/"].forEach(walk);
const code = (name) => fs.readFileSync(new URL(name, root), "utf8").split("\n").filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line)).join("\n");
// a fetch of a built file, a module asked for by a name of this site, a worker; with what is wrapped around each
const later = /(?:named\(import\.meta\.url, \(\) => )?import\("\.[^"]*"\)\)?(?:\.catch\(\(\) => \{\}\))?|new Worker\(new URL\("[^"]*", import\.meta\.url\)|\bbuilt\((?:"[^"]*"|\w+(?:, \w+)?)\)[^\n]*/g;
const tidy = (found) => found.replace(/\s*\/\/ .*$/, "");  // (a built file: the rest of its line, which is where its failure goes)
let said = 0, quiet = 0;
for (const name of sources) {
  const text = code(name);
  if (name === "src/runtime/updated.js") continue;
  const found = name === "src/runtime/built.js" ? [] : [...text.matchAll(later)].map((match) => tidy(match[0]));
  const listed = LATER[name] ?? [];
  assert.deepEqual(found, listed.map(([what]) => what),
    `${name}: the files it asks for later are not the ones tests/updated-check.mjs lists (LATER). One whose absence ends what was asked is fetched ` +
    "by built() (a built file), named(import.meta.url, () => import(…)) (a module) or has unstarted() at its worker's error, and is listed as said; " +
    "one the engine goes on without is listed as quiet, with why");
  // a built file's address is fetched by built() alone, and its table is read nowhere else
  if (name !== "src/runtime/built.js") assert.ok(!/\bBUILT\b/.test(text), `${name} reads the table of built files itself: built() is how one is fetched (a 404 is the site's update)`);
  for (const [what, why] of listed) {
    assert.ok(why === "said" || /^quiet: \S/.test(why), `${name}: ${what} is said, or quiet with a reason`);
    why === "said" ? said++ : quiet++;
    // a worker whose failure is said: the module that names it asks for its own address at the worker's error
    if (why === "said" && what.startsWith("new Worker(")) {
      assert.ok(/\bunstarted\(event, import\.meta\.url, /.test(text), `${name} makes a worker and does not hold its error to unstarted(event, import.meta.url, …)`);
    }
    if (why === "said" && what.includes("import(")) assert.ok(what.startsWith("named(import.meta.url, "), `${name}: ${what} is said by named(import.meta.url, …)`);
    if (why !== "said") assert.ok(!what.includes("named(") && !what.includes("unlessUpdated"), `${name}: ${what} is quiet`);
    // a built file whose failure is said: nothing on its line swallows it (a catch there lets updated() through)
    if (why === "said" && what.startsWith("built(")) assert.ok(!what.includes(".catch(") || what.includes(".catch((error) => unlessUpdated(error) ?? null)"), `${name}: ${what} is said`);
  }
  // in the runtime every worker has an error handler of its own: as many are held to unstarted() as are said
  const workersSaid = listed.filter(([what, why]) => why === "said" && what.startsWith("new Worker(")).length;
  if (name.startsWith("src/runtime/") && workersSaid) {
    assert.equal((text.match(/\.onerror = async \(event\) => reject\(await unstarted\(event, import\.meta\.url, /g) ?? []).length, workersSaid, `${name}: each worker it makes holds its error to unstarted()`);
  }
}
for (const name of Object.keys(LATER)) assert.ok(sources.includes(name), `${name} is listed and is not there`);
// a module that takes these names declares none of them again: sections.js had a `let unstarted` in the function that
// calls unstarted(), which then called the variable (undefined), and the worker's error went nowhere
for (const name of sources) {
  const text = code(name), taken = /import \{([^}]*)\} from "[./]*(?:runtime\/)?(?:updated|built)\.js";/g;
  for (const [, names] of text.matchAll(taken)) for (const one of names.split(",").map((each) => each.trim())) {
    assert.ok(!new RegExp(`\\b(?:let|const|var|function|class)\\s+${one}\\b|\\(${one}\\) =>|\\b${one} = `).test(text), `${name} takes ${one} and declares or assigns a ${one} of its own`);
  }
}
// the quiet ones' files know nothing of updated.js: no fallback was turned into an error
const knows = (name) => /updated\.js|\bunstarted\b|\bisUpdated\b|\bUPDATED\b/.test(code(name));
for (const name of ["src/runtime/worker/weights.js", "src/runtime/gpu.js", "src/runtime/gpu/device.js"]) {
  assert.ok(!knows(name), `${name} holds a fallback (LATER) and now knows of updated.js`);
}
// and neither does anything on a token's, a block's or a submit's path: the engine, its threads, the GPU's worker, the
// shaders, what is kept (the kernels and the Python are not JavaScript)
const path_ = sources.filter((name) => /^src\/runtime\/(forward|gpu|shaders)\/|^src\/runtime\/(forward|gpu|shaders|helper|jobs|kept)\.(js|ts)$/.test(name));
assert.ok(path_.length > 30, "the files of the token's path were found");
for (const name of path_) assert.ok(!knows(name), `${name} is on the token's path and knows of updated.js`);
// both pages hold a worker's error to it, and the model's worker reports it by its message alone (an Error by name)
for (const name of ["src/page/receiver.ts", "src/benchmark/section.ts"]) assert.ok(/import \{ unstarted \} from "\.\.\/runtime\/updated\.js";/.test(code(name)), name);
assert.ok(/w\.onerror = async \(event\) => resolve\(\{ error: await workerFailed\(event\) \}\);/.test(code("src/benchmark/model.ts")), "/benchmark/'s model section holds its worker's error to section.ts's workerFailed()");
assert.ok(/worker\.onerror = async \(event\) => resolve\(\{ error: await workerFailed\(event\) \}\);/.test(code("src/benchmark/section.ts")), "/benchmark/'s sections hold their workers' errors to workerFailed()");
// /benchmark/'s CPU and device worker says a caught value by one function, which says an update alone, and the rows
// that catch a failure of their own (the ceilings, a loop's attempt) let an update through to it
{
  const text = code("src/runtime/benchmark/sections.js");
  assert.ok(text.includes("const failed = (error) => saidOf(error, `${error?.name ?? \"Error\"}: ${error?.message ?? error}`);"), "sections.js's failed() says an update alone");
  assert.ok(text.includes("postMessage({ step: data.step, error: failed(error) });"), "a step that failed is said by failed()");
  assert.equal((text.match(/\n {4}unlessUpdated\(error\);\n/g) ?? []).length, 2, "the ceilings' and an attempt's catch let an update through");
  assert.ok(!/\$\{error\?\.name \?\? "Error"\}/.test(text.replace(/const failed = [^\n]*/, "")), "no other place of sections.js puts a name before a caught value's words");
}
// the sentence is written once
const writers = sources.filter((name) => fs.readFileSync(new URL(name, root), "utf8").includes("The site was updated"));
assert.deepEqual(writers, ["src/runtime/updated.js"], "the sentence is written in one file");
console.log(`updated-check: ${said} files asked for later say the update where they are gone, ${quiet} keep their fallback and say nothing; ${path_.length} files of the token's path know none of it`);
console.log("updated-check: ok");
