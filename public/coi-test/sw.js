// T93, part 2: a service worker that adds the two headers that make a page cross-origin isolated, the way
// coi-serviceworker does, for GitHub Pages, which cannot send them. Its scope is this directory only
// (/coi-test/): the model page is never under it. ?coep= picks credentialless (the default) or require-corp.
const coep = new URL(self.location).searchParams.get("coep") === "require-corp" ? "require-corp" : "credentialless";
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  event.respondWith((async () => {
    const response = await fetch(request);
    if (response.status === 0) return response;  // opaque: nothing to add, and nothing may be read
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", coep);
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  })());
});
