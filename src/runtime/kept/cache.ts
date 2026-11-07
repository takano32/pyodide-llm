// kept/cache.ts (T99, T641.7): the Cache API as a place a converted model is kept in (kept/place.ts), as before T99
// (converted-v1): parts of 8 MiB, the tokenizer and the manifest, under one URL prefix per model, the manifest put last.
// New conversions come here only where there is no OPFS (kept.ts's PLACES); what a visitor kept here before is read
// and deleted as it always was, never moved.

import type { Found, Keep, Kept, Manifest, Place, Slice } from "./place.ts";
import { PART_BYTES } from "./place.ts";

const CACHE_NAME = "converted-v1";

const cacheKey = (name: string, file: string) => `${self.location.origin}/converted/${name}/${file}`;
const partKey = (name: string, part: number) => cacheKey(name, `part-${String(part).padStart(3, "0")}`);
const openCache = () => globalThis.caches?.open(CACHE_NAME).catch(() => undefined) ?? Promise.resolve(undefined);

async function list() {
  const found: Kept[] = [];
  const cache = await openCache();
  for (const request of (await cache?.keys()) ?? []) {
    const match = request.url.match(/\/converted\/([^/]+)\/manifest\.json$/);
    if (match) found.push({ name: match[1], where: "cache", manifest: await (await cache!.match(request))!.json() });
  }
  return found;
}

async function open(name: string): Promise<Found | undefined> {
  const cache = await openCache();
  const manifest: Manifest | undefined = await (await cache?.match(cacheKey(name, "manifest.json")))?.json();
  if (!manifest) return undefined;
  return {
    manifest,
    where: "cache",
    name,
    // (a stretch to skip is read all the same: the parts are the Cache API's, and each is read whole)
    async *parts() {
      for (let part = 0; part < manifest.parts!; part++) {
        const stored = await cache!.match(partKey(name, part));
        if (!stored) throw new Error("the browser has evicted a part of the kept model");
        yield new Uint8Array(await stored.arrayBuffer());
      }
    },
    tokenizer: async () => new Uint8Array(await (await cache!.match(cacheKey(name, "tokenizer.bin")))!.arrayBuffer()),
  };
}

async function keeping(): Promise<Keep | undefined> {
  const cache = await openCache();
  return cache ? (name, manifest, slice, tokenizer, signal) => keep(cache, name, manifest, slice, tokenizer, signal) : undefined;
}
async function keep(cache: Cache, name: string, manifest: Manifest, slice: Slice, tokenizer: Uint8Array<ArrayBuffer>, signal?: AbortSignal) {
  try {
    const parts = Math.ceil(manifest.bytes / PART_BYTES);
    for (let part = 0; part < parts; part++) {
      const copy = slice(part * PART_BYTES, Math.min((part + 1) * PART_BYTES, manifest.bytes));
      await cache.put(partKey(name, part), new Response(copy));
      signal?.throwIfAborted();
    }
    await cache.put(cacheKey(name, "tokenizer.bin"), new Response(tokenizer));
    await cache.put(cacheKey(name, "manifest.json"), new Response(JSON.stringify({ ...manifest, parts }), { headers: { "Content-Type": "application/json" } }));
    return undefined;
  } catch (error) {
    await forget(name);
    if (signal?.aborted) throw error;
    return String((error as Error).message ?? error);
  }
}

async function forget(name: string) {
  const cache = await openCache();
  for (const request of (await cache?.keys()) ?? []) {
    if (request.url.startsWith(cacheKey(name, ""))) await cache!.delete(request);
  }
}

export const cache: Place = { where: "cache", list, open, keeping, forget };
