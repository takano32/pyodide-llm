// T.117: public/coi.js (the service worker) in Node, on a stand-in Cache API and fetch, the way a browser calls it:
// its fetch listener gets a request and answers through respondWith. What a browser does with it is for CI and the
// owner's devices (preview.yml's offline run); this holds the three things T.117 fixed:
//   1. offline, the page (a navigation) is answered by a copy whatever its query, a script of another build is not
//      (worker.js?v=<another> in a page of this build would mix two deployments)
//   2. a Cache API that fails to open leaves the network to answer, even for a file whose copy would answer first
//   3. a new wheel or a new file of /_astro/ puts the old one out of the copies
// and T.367.6's, what an activation does with the copies:
//   4. it drops this site's copies with another build's ?v= (the runtime's files of before T.367.3, which nobody asks
//      for again) and nothing else: not this build's, not /_astro/'s, not the page's, not another site's whatever
//      its address says; a second activation finds nothing to drop; without a network the page still answers
//   5. nothing in it stops the activation or the claim: a copy that cannot be deleted, a list that cannot be read,
//      a Cache API that does not open; and the claim does not wait for it
//   6. registered with offline=0 it throws the copies away whole, as before, and looks at none
//   7. the parts of coi.js that nothing may change without the owner's devices (AGENTS.md: how it treats other
//      sites' responses, the isolation headers, how a response is made again) are the text they were
//
//   node tests/coi-js-check.mjs
import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import vm from "node:vm";
import { cacheStorage } from "./fakes/cache.mjs";
import { failing, madeUpNetwork } from "./fakes/fetch.mjs";
import { served } from "./tree.mjs";

const SITE = "https://takano32.github.io/pyodide-llm/";
// 7's: the SHA-256 of the held parts of coi.js (as of T.367.6, the same text as before it)
const HELD = "88ad804c36acbbf740121bb54e654b22d9c0a1361655bf79e7409b5dc2eac9f5";
const source = fs.readFileSync(served("coi.js"), "utf8");

// the Cache API is tests/fakes/cache.mjs's: one cache, keyed by URL (ignoreSearch drops the query); broken: it does not
// open; stuck: the addresses whose copies cannot be deleted; unlisted: keys() fails. said: what was asked of it, in
// order ("keys", "delete <url>", "delete the cache <name>", "open")

// coi.js loaded afresh with these stand-ins; returns ask(url, mode) -> the answer's text, or an error.
// ask.activate() is the browser's activate event: it waits for what the listener asked it to wait for, and gives
// what that ended as (undefined, or the error it was rejected with). said: the cache's list, where the claim goes too
function worker({ caches, online, registered = "?v=new", said = [] }) {
  const listeners = {}, network = madeUpNetwork();
  const context = {
    self: { location: new URL(`${SITE}coi.js${registered}`), addEventListener: (type, f) => { listeners[type] = f; }, skipWaiting() {},
      clients: { claim: async () => { said.push("claim"); } } },
    caches, URL, Headers, Response,
    fetch: network.fetch,
  };
  // (tests/fakes/fetch.mjs's network: every address answers with its own name, or the line is down)
  network.answers(online ? (url) => new Response(`network: ${url}`, { status: 200 }) : failing(new TypeError("Failed to fetch")));
  context.self.caches = caches;
  vm.createContext(context);
  vm.runInContext(source, context);
  const ask = async (url, mode = "cors") => {
    let answer, waited = [];
    listeners.fetch({
      request: { url, method: "GET", headers: new Headers(), mode, cache: "default" },
      respondWith: (promise) => { answer = promise; },
      waitUntil: (promise) => waited.push(promise),
    });
    try {
      const text = await (await answer).text();
      await Promise.all(waited);
      return text;
    } catch (error) {
      return error;
    }
  };
  ask.activate = async () => {
    const waited = [];
    listeners.activate({ waitUntil: (promise) => waited.push(promise) });
    assert.equal(waited.length, 1, "the activation is waited for");
    return waited[0].then(() => undefined, (error) => error);
  };
  return ask;
}

// 1. offline: the page by any query, not a script of another build
{
  const { caches } = cacheStorage();
  const online = worker({ caches, online: true });
  await online(`${SITE}?model=tiny-lm`, "navigate");
  await online(`${SITE}worker.js?v=old`);
  const offline = worker({ caches, online: false });
  assert.equal(await offline(`${SITE}?model=stories260K`, "navigate"), `network: ${SITE}?model=tiny-lm`, "the page, whatever its query");
  assert.equal(await offline(`${SITE}worker.js?v=old`), `network: ${SITE}worker.js?v=old`, "the same build's file");
  assert.ok(await offline(`${SITE}worker.js?v=new`) instanceof Error, "another build's file is not answered by this one");
}
// 2. a Cache API that cannot be opened: the network answers, also for a file whose copy would answer first
{
  const { caches } = cacheStorage({ broken: true });
  const ask = worker({ caches, online: true });
  assert.equal(await ask(`${SITE}worker.js?v=new`), `network: ${SITE}worker.js?v=new`);
  assert.equal(await ask("https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.asm.wasm"), "network: https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.asm.wasm");
  assert.equal(await ask(`${SITE}?model=tiny-lm`, "navigate"), `network: ${SITE}?model=tiny-lm`);
}
// 3. one copy of a wheel and of a file of /_astro/: the newer puts the older out
{
  const { caches, map } = cacheStorage();
  const ask = worker({ caches, online: true });
  const wheel = (pyodide, numpy) => `https://cdn.jsdelivr.net/pyodide/v${pyodide}/full/numpy-${numpy}-cp314-cp314-pyodide_2025_0_wasm32.whl`;
  await ask(wheel("314.0.7", "2.2.5"));
  await ask(wheel("314.0.8", "2.3.0"));
  await ask(`${SITE}_astro/index.DQp1k2f3.css`);
  await ask(`${SITE}_astro/index.Zz9x8y7w.css`);
  await ask(`${SITE}_astro/hoisted.ab12cd34.js`);
  assert.deepEqual([...map.keys()].sort(), [`${SITE}_astro/hoisted.ab12cd34.js`, `${SITE}_astro/index.Zz9x8y7w.css`, wheel("314.0.8", "2.3.0")].sort());
}

// T.367.6. What a browser that visited before T.367.3 holds, and what one of today holds
const JSDELIVR = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const FORGOTTEN = [`${SITE}worker.js?v=old`, `${SITE}worker/state.js?v=old`, `${SITE}simdkernel_plain.wasm?v=old`, `${SITE}simdkernel.so?v=older`,
  `${SITE}engine/model.py?v=old`, `${SITE}python.js?v=old`, `${SITE}engine.zip?v=old`, `${SITE}benchmark/gpu.js?v=old&x=1`, `${SITE}?v=1&model=tiny-lm`];
const STAYING = [
  `${SITE}?model=tiny-lm`, `${SITE}benchmark/`,                                    // the pages
  `${SITE}_astro/index.DQp1k2f3.js`, `${SITE}_astro/simdkernel_plain.Zz9x8y7w.wasm`, `${SITE}_astro/engine.ab12cd34.zip`,
  `${SITE}_astro/index.DQp1k2f3.css`,
  `${SITE}models/tok32000.bin`,                                                    // a tokenizer: no version in its address
  `${SITE}served.js?v=new`,                                                        // a file served as it is, of this build
  `${JSDELIVR}pyodide.asm.wasm`, `${JSDELIVR}numpy-2.2.5-cp314-cp314-pyodide_2025_0_wasm32.whl`,
  "https://cdn.jsdelivr.net/npm/some@1/file.js?v=old", "https://data.jsdelivr.com/v1/packages/npm/pyodide/resolved?v=old",   // another site's, whatever it says
  "https://huggingface.co/owner/repo/resolve/main/config.json?v=old",
  SITE.replace("https://takano32.github.io", "https://takano32.github.io:8443") + "worker.js?v=old",      // another origin of the same name
];
const seeded = (options) => {
  const storage = cacheStorage(options);
  for (const url of [...FORGOTTEN, ...STAYING]) storage.map.set(url, new TextEncoder().encode(`copy: ${url}`).buffer);
  return storage;
};
const deleted = (said) => said.filter((line) => line.startsWith("delete ")).map((line) => line.slice("delete ".length));
const sorted = (list) => [...list].sort();

// 4. the copies with another build's ?v= go, and nothing else
{
  const { caches, map, said } = seeded();
  const ask = worker({ caches, online: true, said });
  assert.equal(await ask.activate(), undefined, "the activation ends well");
  assert.deepEqual(sorted(deleted(said)), sorted(FORGOTTEN), "what an activation deletes: this site's copies with another build's ?v=, each once, and no other");
  assert.deepEqual(sorted(map.keys()), sorted(STAYING), "what stays: this build's, /_astro/'s, the pages, the tokenizer, every other site's");
  assert.ok(said.includes("claim"), "the pages are claimed");
  assert.ok(!said.some((line) => line.startsWith("delete the cache")), "the copies are not thrown away whole");
  // a second activation (the next deployment's): one look, nothing to delete
  said.length = 0;
  assert.equal(await ask.activate(), undefined);
  assert.deepEqual(deleted(said), [], "a second activation deletes nothing");
  assert.deepEqual(said.filter((line) => line !== "claim"), ["open", "keys"], "and costs one list of the copies");
  // T.111: without a network, the page, its files and Pyodide still answer from the copies
  const offline = worker({ caches, online: false });
  assert.equal(await offline(`${SITE}?model=stories260K`, "navigate"), `copy: ${SITE}?model=tiny-lm`, "offline after it: the page");
  assert.equal(await offline(`${SITE}_astro/index.DQp1k2f3.js`), `copy: ${SITE}_astro/index.DQp1k2f3.js`, "offline after it: a file of /_astro/");
  assert.equal(await offline(`${JSDELIVR}pyodide.asm.wasm`), `copy: ${JSDELIVR}pyodide.asm.wasm`, "offline after it: Pyodide");
  assert.equal(await offline(`${SITE}models/tok32000.bin`), `copy: ${SITE}models/tok32000.bin`, "offline after it: the tokenizer");
  assert.equal(await offline(`${SITE}served.js?v=new`), `copy: ${SITE}served.js?v=new`, "offline after it: this build's file with ?v=");
}
// a Service Worker registered without a build: every ?v= of this site is another's
{
  const { caches, map, said } = seeded();
  assert.equal(await worker({ caches, online: true, said, registered: "" }).activate(), undefined);
  assert.deepEqual(sorted(map.keys()), sorted(STAYING.filter((url) => url !== `${SITE}served.js?v=new`)));
}
// a browser of today: nothing to delete on the first activation either
{
  const storage = cacheStorage();
  for (const url of STAYING) storage.map.set(url, new ArrayBuffer(1));
  assert.equal(await worker({ caches: storage.caches, online: true, said: storage.said }).activate(), undefined);
  assert.deepEqual(deleted(storage.said), [], "nothing of today's copies is deleted");
}
// 5. nothing in it stops the activation or the claim
{
  // one copy that cannot be deleted: the others go
  const stuck = [FORGOTTEN[1], FORGOTTEN[4]];
  const { caches, map, said } = seeded({ stuck });
  assert.equal(await worker({ caches, online: true, said }).activate(), undefined, "a copy that cannot be deleted does not fail the activation");
  assert.deepEqual(sorted(map.keys()), sorted([...STAYING, ...stuck]), "and the other copies still go");
  assert.ok(said.includes("claim"));
  // the claim does not wait for the deleting
  assert.ok(said.indexOf("claim") < said.findIndex((line) => line.startsWith("delete ")), `the pages are claimed before the first copy is deleted, not after the last: ${said.slice(0, 4).join(", ")}`);
}
for (const [what, options] of [["a list of the copies that cannot be read", { unlisted: true }], ["a Cache API that does not open", { broken: true }]]) {
  const { caches, map, said } = seeded(options);
  assert.equal(await worker({ caches, online: true, said }).activate(), undefined, `${what} does not fail the activation`);
  assert.ok(said.includes("claim"), `${what}: the pages are claimed`);
  assert.equal(map.size, FORGOTTEN.length + STAYING.length, `${what}: nothing is deleted`);
}
// 6. offline=0: the copies go whole, as before; no copy is looked at
{
  const { caches, said } = seeded();
  assert.equal(await worker({ caches, online: true, said, registered: "?v=new&offline=0" }).activate(), undefined);
  assert.deepEqual(said, ["delete the cache offline-v1", "claim"]);
  // (T.449) and a deletion that fails keeps the claim from nothing
  const refusing = { ...caches, delete: async () => { said.push("refused"); throw new Error("the deletion failed"); } };
  said.length = 0;
  assert.equal(await worker({ caches: refusing, online: true, said, registered: "?v=new&offline=0" }).activate(), undefined, "a deletion that fails does not fail the activation");
  assert.deepEqual(said, ["refused", "claim"], "the pages are claimed after a deletion that failed");
}
// 7. What this file may not change without the owner's devices first (AGENTS.md: 2026-09-25, a change to how
// another site's response passes was right in CI's browsers in the US and stopped the owner's phone and PC in Japan):
// how a response is made again with the two headers, the whole of the fetch listener, and what kept(), lasting()
// and family() say of another site's address. Which of this site's files are kept is free to change, and is not here.
{
  const part = (pattern, what) => {
    const found = pattern.exec(source)?.[0];
    assert.ok(found, `coi.js has no ${what}: tests/coi-js-check.mjs holds its text`);
    return found;
  };
  const held = [
    part(/^function isolated\(response\) \{\n[\s\S]*?\n\}$/m, "isolated(response)"),
    part(/^self\.addEventListener\("install"[^\n]*$/m, "install listener on one line"),
    part(/^self\.addEventListener\("fetch"[\s\S]*?\n\}\);$/m, "fetch listener"),
    part(/^  return url\.hostname === "cdn\.jsdelivr\.net" \|\| [^\n]*$/m, "line of kept() for another site"),
    part(/^  return url\.hostname === "cdn\.jsdelivr\.net" && [^\n]*$/m, "line of lasting() for another site"),
    part(/^  return url\.href\.replace\([^\n]*$/m, "line of family() for another site"),
    part(/^const quietly = [^\n]*$/m, "quietly()"),
  ];
  const hash = crypto.createHash("sha256").update(held.join("\n\u0000\n")).digest("hex");
  assert.equal(hash, HELD,
    "public/coi.js: how it treats another site's response, the isolation headers or the fetch listener changed. That is not a change to make on CI's word " +
    "(AGENTS.md, the pitfall of the Service Worker): the owner's devices first, and then the hash in tests/coi-js-check.mjs");
}
console.log("ok");
