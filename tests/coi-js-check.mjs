// T117: public/coi.js (the service worker) in Node, on a stand-in Cache API and fetch, the way a browser calls it:
// its fetch listener gets a request and answers through respondWith. What a browser does with it is for CI and the
// owner's devices (preview.yml's offline run); this holds the three things T117 fixed:
//   1. offline, the page (a navigation) is answered by a copy whatever its query, a script of another build is not
//      (worker.js?v=<another> in a page of this build would mix two deployments)
//   2. a Cache API that fails to open leaves the network to answer, even for a file whose copy would answer first
//   3. a new wheel or a new file of /_astro/ puts the old one out of the copies
//
//   node tests/coi-js-check.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const SITE = "https://takano32.github.io/pyodide-llm/";
const source = fs.readFileSync(new URL("../public/coi.js", import.meta.url), "utf8");

// a Cache API of one cache, keyed by URL (ignoreSearch drops the query)
function cacheStorage({ broken = false } = {}) {
  const map = new Map();
  const cache = {
    put: async (url, response) => { map.set(typeof url === "string" ? url : url.url, await response.arrayBuffer()); },
    match: async (url, { ignoreSearch = false } = {}) => {
      const key = typeof url === "string" ? url : url.url;
      const strip = (u) => u.split("?")[0];
      for (const [stored, body] of map) {
        if (stored === key || (ignoreSearch && strip(stored) === strip(key))) return new Response(body.slice(0));
      }
      return undefined;
    },
    keys: async () => [...map.keys()].map((url) => ({ url })),
    delete: async (request) => map.delete(request.url),
  };
  return { map, caches: { open: async () => { if (broken) throw new Error("SecurityError"); return cache; }, delete: async () => true } };
}

// coi.js loaded afresh with these stand-ins; returns ask(url, mode) -> the answer's text, or an error
function worker({ caches, online }) {
  const listeners = {};
  const context = {
    self: { location: new URL(`${SITE}coi.js?v=new`), addEventListener: (type, f) => { listeners[type] = f; }, skipWaiting() {}, clients: { claim: async () => {} } },
    caches, URL, Headers, Response,
    fetch: async (request) => {
      if (!online) throw new TypeError("Failed to fetch");
      return new Response(`network: ${request.url}`, { status: 200 });
    },
  };
  context.self.caches = caches;
  vm.createContext(context);
  vm.runInContext(source, context);
  return async (url, mode = "cors") => {
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
console.log("ok");
