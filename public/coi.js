// coi.js (T93, stage 3): the service worker that makes this site cross-origin isolated, so that the page gets a
// SharedArrayBuffer and forward.js its software threads. GitHub Pages cannot send response headers, so this adds
// the two that isolation needs to every response it passes on: Cross-Origin-Opener-Policy: same-origin and
// Cross-Origin-Embedder-Policy: require-corp (credentialless is not known to WebKit; tests/coi-check.mjs). The
// cross-origin files this site reads (Pyodide on jsDelivr, models on huggingface.co) pass through here as well,
// so that what the page gets is a response of its own. It adds nothing else and keeps nothing: without it the page
// runs on one thread.
// ?coi=off on the page unregisters it.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  // Cross-origin fetches (Pyodide on jsDelivr, a model on huggingface.co) go through here too, on purpose: the
  // response made here is the page's own, and the isolation checks let it through. Leaving them to the browser
  // (T97, for a day on 2026-09-25) stopped Pyodide's NumPy on iOS Safari and the model's fetch on the owner's
  // phone and PC, from Japan, while CI's browsers in the US went on: the checks fell on the CDN's real headers.
  event.respondWith((async () => {
    const response = await fetch(request);
    if (response.status === 0) return response;  // opaque: nothing to add, and nothing may be read
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  })());
});
