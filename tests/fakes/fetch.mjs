// tests/fakes/fetch.mjs (T357.5; out of tests/worker-harness.mjs, where T129 wrote it)
// A stand-in for fetch(): a made-up network that writes down every request and answers it as the check says. For
// the checks of what asks a server for something: the worker on its made-up network (tests/worker-harness.mjs, which
// hands this on as its own), what says "the site was updated" (updated-check.mjs, worker-outdated-check.mjs) and the
// service worker (coi-js-check.mjs).
//
//   const { fetch, requests, answers } = madeUpNetwork({ now })
//
//   fetch(input, init)     what stands for the global: a promise of the answer. input is an address, a URL or anything
//                          with a url (a Request)
//   answers(route)         from here on route(url, init, n) answers a request with a Response, a promise of one, or
//                          HANG (no answer until the request is aborted); what it throws, or rejects with, is what the
//                          fetch fails with. n: how many requests of that address are written down already (0 for the
//                          first). Before any answers(): every request a 404
//   requests               every request, in order: { url, at, signal, range, method, init } (at: now() as it was
//                          made; range: its Range header; method: "GET" where the init says none; init: as it was
//                          given). The check's to empty (requests.length = 0), which starts the counts again
//   now                    the clock a request's `at` is read from (the check's own; by default the time stands at 0)
//
//   status(code), failing(error), hang     routes for the common answers: an empty response of that status, a fetch
//                          that fails with that error (a line that dropped: new TypeError("Failed to fetch")), no answer
//
// What it does as a browser's does, because the code under test leans on it: a request whose signal is aborted
// already fails with the signal's reason and the route is not asked; one aborted while it waits fails so then (a route
// that answered already is not undone: its body is the Response's own to break, see tests/worker-harness.mjs's body()).
// Where it is simpler: no redirects, no CORS, no cache, no Service Worker in between, no headers checked; the n-th
// request is counted by the address as a string (a query is part of it); an answer comes as soon as the route gives
// it, in the order the routes resolve.

/** No answer until the request is aborted: what a route returns for a server that never answers. */
export const HANG = "hang";
/** What an aborted request fails with: the signal's reason, as fetch() gives it. */
export const abortError = (signal) => signal.reason ?? new DOMException("aborted", "AbortError");

/** A route that answers every request with an empty response of that status. */
export const status = (code) => () => new Response(null, { status: code });
/** A route whose every request fails with that error. */
export const failing = (error) => () => { throw error; };
/** A route that never answers. */
export const hang = () => HANG;

export function madeUpNetwork({ now = () => 0 } = {}) {
  const requests = [];
  let route = () => new Response("", { status: 404 });
  function fetch(input, init = {}) {
    const url = String(input?.url ?? input);
    const n = requests.filter((r) => r.url === url).length;
    const request = { url, at: now(), signal: init.signal, range: init.headers?.Range, method: init.method ?? "GET", init };
    requests.push(request);
    return new Promise((resolve, reject) => {
      const signal = init.signal;
      if (signal?.aborted) return reject(abortError(signal));
      signal?.addEventListener("abort", () => reject(abortError(signal)), { once: true });
      Promise.resolve(route(url, init, n)).then((answer) => (answer === HANG ? undefined : resolve(answer)), reject);
    });
  }
  const answers = (routing) => {
    route = routing;
  };
  return { fetch, requests, answers };
}
