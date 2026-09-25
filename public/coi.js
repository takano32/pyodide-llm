// coi.js (T93, stage 3): the service worker that makes this site cross-origin isolated, so that the page gets a
// SharedArrayBuffer and forward.js its software threads. GitHub Pages cannot send response headers, so this adds
// the two that isolation needs to every response it passes on: Cross-Origin-Opener-Policy: same-origin and
// Cross-Origin-Embedder-Policy: require-corp (credentialless is not known to WebKit; tests/coi-check.mjs). The
// cross-origin files this site reads (Pyodide on jsDelivr, models on huggingface.co) answer with CORS, so
// require-corp lets them through, and this worker does not touch them (T97). It adds nothing else and keeps
// nothing: without it the page runs on one thread.
// ?coi=off on the page unregisters it.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  // T97: only what this site serves gets the headers. A cross-origin fetch (Pyodide on jsDelivr, a model on
  // huggingface.co) is left to the browser: the headers belong to the document and its scripts, and re-streaming
  // hundreds of megabytes of a model through here is what Firefox on Windows broke on ("Error in input stream").
  if (new URL(request.url).origin !== self.location.origin) return;
  event.respondWith((async () => {
    const response = await fetch(request);
    if (response.status === 0) return response;  // opaque: nothing to add, and nothing may be read
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  })());
});
