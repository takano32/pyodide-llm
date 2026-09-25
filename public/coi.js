// coi.js (T93, stage 3): the service worker that makes this site cross-origin isolated, so that the page gets a
// SharedArrayBuffer and forward.js its software threads. GitHub Pages cannot send response headers, so this adds
// the two that isolation needs to every response it passes on: Cross-Origin-Opener-Policy: same-origin and
// Cross-Origin-Embedder-Policy: require-corp (credentialless is not known to WebKit; tests/coi-check.mjs). The
// cross-origin files this site reads (Pyodide on jsDelivr, models on huggingface.co) pass through here as well,
// so that what the page gets is a response of its own. It adds nothing else and, unless offline mode is on (below),
// keeps nothing: without it the page runs on one thread.
// ?coi=off on the page unregisters it.
//
// T111: offline. Registered with offline=1 (the page's ?offline=on, remembered), it also keeps a copy of what it
// passed on: this site's files and Pyodide's (jsDelivr), not the models' parts (the worker keeps those itself).
// the latest Pyodide is still resolved each visit (policy 2), and the page itself is asked for each time; only when
// that fails does the copy answer. A file whose address says its version (Pyodide's under /pyodide/v…/, this site's
// with ?v=<build> or under /_astro/) cannot change, so its copy answers without asking the network at all: on a
// mobile line a visit then costs the version check and the page, nothing more (the owner's wish; Safari does not
// say what line it is on, so this does not ask). One copy per file: a new build or a new Pyodide replaces the old
// one. Registered without the flag, it throws the copies away.
const OFFLINE = new URL(self.location.href).searchParams.get("offline") === "1";
const KEPT = "offline-v1";

/** Whether offline mode keeps a copy of this file. */
function kept(request) {
  const url = new URL(request.url);
  if (request.method !== "GET" || request.headers.has("range")) return false;
  // the model's 8 MiB parts (.000, .001, …) the worker keeps itself; its tokenizer is kept here
  if (url.origin === self.location.origin) return !/\/models\/.*\.\d{3}$/.test(url.pathname);
  return url.hostname === "cdn.jsdelivr.net" || url.hostname === "data.jsdelivr.com";
}
/** Whether the address names one version of the file, which then never changes. */
function lasting(request) {
  const url = new URL(request.url);
  if (url.origin === self.location.origin) return url.searchParams.has("v") || url.pathname.includes("/_astro/");
  return url.hostname === "cdn.jsdelivr.net" && /\/pyodide\/v[^/]+\//.test(url.pathname);
}
/** Files that are versions of one another: the same path of this site whatever ?v=, the same file of any Pyodide. */
function family(address) {
  const url = new URL(address);
  if (url.origin === self.location.origin) return url.origin + url.pathname;
  return url.href.replace(/\/pyodide\/v[^/]+\//, "/pyodide/*/");
}
async function keep(request, response) {
  const cache = await caches.open(KEPT);
  await cache.put(request.url, response);
  for (const old of await cache.keys()) {
    if (old.url !== request.url && family(old.url) === family(request.url)) await cache.delete(old);
  }
}
async function copyOf(request) {
  const cache = await caches.open(KEPT);
  // the page itself is asked for with whatever query the visitor wrote (?model=…): any of it will do
  return (await cache.match(request.url)) ?? (new URL(request.url).origin === self.location.origin
    ? await cache.match(request.url, { ignoreSearch: true }) : undefined);
}
function isolated(response) {
  if (response.status === 0) return response;  // opaque: nothing to add, and nothing may be read
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil((async () => {
  if (!OFFLINE) await caches.delete(KEPT);
  await self.clients.claim();
})()));
self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  // Cross-origin fetches (Pyodide on jsDelivr, a model on huggingface.co) go through here too, on purpose: the
  // response made here is the page's own, and the isolation checks let it through. Leaving them to the browser
  // (T97, for a day on 2026-09-25) stopped Pyodide's NumPy on iOS Safari and the model's fetch on the owner's
  // phone and PC, from Japan, while CI's browsers in the US went on: the checks fell on the CDN's real headers.
  event.respondWith((async () => {
    if (OFFLINE && kept(request) && lasting(request)) {
      const copy = await (await caches.open(KEPT)).match(request.url);
      if (copy) return isolated(copy);
    }
    let response;
    try {
      response = await fetch(request);
    } catch (error) {
      const copy = OFFLINE && kept(request) ? await copyOf(request) : undefined;
      if (copy) return isolated(copy);
      throw error;
    }
    if (OFFLINE && response.status === 200 && kept(request)) event.waitUntil(keep(request, response.clone()));
    return isolated(response);
  })());
});
