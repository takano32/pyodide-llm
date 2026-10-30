// tests/fakes/cache.mjs (T.357.5)
// A stand-in for the Cache API (a browser's `caches`), for the checks of what keeps copies in it: the service worker
// (coi-js-check.mjs), the kept conversions (kept-check.mjs) and the site's models' parts (worker-check.mjs).
//
//   const { caches, cache, map, said, stored } = cacheStorage({ broken, stuck, unlisted })
//
//   caches     what stands for globalThis.caches: open(name) gives the one cache whatever the name, delete(name) empties it
//   cache      that one cache: put, match, keys, delete
//   map        what it holds, address -> bytes: a check reads it, and may seed it (a Uint8Array or an ArrayBuffer)
//   said       what was asked of it, in order: "open", "keys", "delete <address>", "delete the cache <name>"
//   stored     every copy put() stored, in order: [address, its bytes' length] (T.495: a copy that was stored and
//              written over later is gone from map, and seen here)
//   broken     it does not open (a private window, storage turned off); stuck: the addresses whose copies cannot be
//              deleted; unlisted: keys() fails
//
// What it does as the specification says, because the code under test leans on it:
//   - put() reads the response's body to its end before anything is stored, and stores nothing where the body breaks
//     (it rejects with the body's error): a part that did not come whole leaves no copy;
//   - match() answers with a copy (reading it twice, or changing what was read, changes nothing held), undefined where
//     there is none, and with { ignoreSearch: true } the first copy whose address is the same but for its query;
//   - a key is an address or anything with a url (a Request).
// Where it is simpler than a browser's: one cache under every name; no Vary, no method, no status and no headers kept
// (a copy answers 200 whatever was put, and a 206 is not refused); no quota; keys() in the order they were put.

/** A Cache API of one cache. */
export function cacheStorage({ broken = false, stuck = [], unlisted = false } = {}) {
  const map = new Map(), said = [], stored = [];
  const addressOf = (key) => (typeof key === "string" ? key : key.url);
  const withoutQuery = (address) => address.split("?")[0];
  const cache = {
    async put(key, response) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      map.set(addressOf(key), bytes);
      stored.push([addressOf(key), bytes.length]);
    },
    async match(key, { ignoreSearch = false } = {}) {
      const address = addressOf(key);
      for (const [stored, body] of map) {
        if (stored === address || (ignoreSearch && withoutQuery(stored) === withoutQuery(address))) return new Response(body.slice(0), { status: 200 });
      }
      return undefined;
    },
    async keys() {
      said.push("keys");
      if (unlisted) throw new Error("UnknownError");
      return [...map.keys()].map((url) => ({ url }));
    },
    async delete(key) {
      const address = addressOf(key);
      said.push(`delete ${address}`);
      if (stuck.includes(address)) throw new Error("QuotaExceededError");
      return map.delete(address);
    },
  };
  const caches = {
    async open() {
      said.push("open");
      if (broken) throw new Error("SecurityError");
      return cache;
    },
    async delete(name) {
      said.push(`delete the cache ${name}`);
      map.clear();
      return true;
    },
  };
  return { map, said, stored, cache, caches };
}
