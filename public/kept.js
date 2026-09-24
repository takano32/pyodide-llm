// kept.js (T99): the models this browser converted from Hugging Face, kept for the next visit. The worker writes
// and reads them; the page lists them and deletes them. Two places, one interface:
//
//   the origin private file system (OPFS), where the browser has it: a folder per model with model.bin (the whole
//   checkpoint in one file), tokenizer.bin and manifest.json. The worker writes and reads with a sync access handle.
//   The Cache API, as before T99 (converted-v1): parts of 8 MiB, the tokenizer and the manifest, under one URL
//   prefix per model. New conversions go there only where there is no OPFS; what a visitor kept there before is read
//   and deleted as it always was, never moved.
//
// The manifest is written last in both: a folder without one is rubbish from a write that did not finish.
// A plain ES module: the worker imports it, and so does the page.

export const CACHE_NAME = "converted-v1";
const FOLDER = "converted-v1";
const PART_BYTES = 8 * 1024 * 1024;

// the name of a model's conversion: its repository, revision, dtype and context, as the Cache API key has it
export function keptName(model) {
  const { dtype = "int8", max_seq_len = 4096 } = model.conversion ?? {};
  return encodeURIComponent(`${model.hf.repo}@${model.hf.revision}:${dtype}:${max_seq_len}`);
}
const cacheKey = (name, file) => `${self.location.origin}/converted/${name}/${file}`;

async function folders(create = false) {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(FOLDER, { create });
  } catch {
    return undefined;  // no OPFS here (or not yet created)
  }
}
const openCache = () => globalThis.caches?.open(CACHE_NAME).catch(() => undefined) ?? Promise.resolve(undefined);
// OPFS with the synchronous access a worker needs to write a file of gigabytes piece by piece
const opfsWritable = () => typeof FileSystemFileHandle !== "undefined" && "createSyncAccessHandle" in FileSystemFileHandle.prototype;

/** Every kept model: { name, where: "opfs" | "cache", manifest }. */
export async function keptModels() {
  const found = [];
  const directory = await folders();
  if (directory) {
    for await (const [name, handle] of directory.entries()) {
      try {
        const file = await (await handle.getFileHandle("manifest.json")).getFile();
        found.push({ name, where: "opfs", manifest: JSON.parse(await file.text()) });
      } catch {
        // no manifest: not finished
      }
    }
  }
  const cache = await openCache();
  for (const request of (await cache?.keys()) ?? []) {
    const match = request.url.match(/\/converted\/([^/]+)\/manifest\.json$/);
    if (match && !found.some((kept) => kept.name === match[1])) {
      found.push({ name: match[1], where: "cache", manifest: await (await cache.match(request)).json() });
    }
  }
  return found;
}

/** A kept model to read: { manifest, where, parts(), tokenizer() }, or undefined. parts() yields its bytes in order. */
export async function openKept(model) {
  const name = keptName(model);
  const directory = await folders();
  const folder = await directory?.getDirectoryHandle(name).catch(() => undefined);
  if (folder) {
    try {
      const manifest = JSON.parse(await (await (await folder.getFileHandle("manifest.json")).getFile()).text());
      const file = await (await folder.getFileHandle("model.bin")).getFile();
      if (file.size === manifest.bytes) {
        return {
          manifest,
          where: "opfs",
          async *parts() {
            for (let offset = 0; offset < file.size; offset += PART_BYTES) {
              yield new Uint8Array(await file.slice(offset, offset + PART_BYTES).arrayBuffer());
            }
          },
          tokenizer: async () => new Uint8Array(await (await (await folder.getFileHandle("tokenizer.bin")).getFile()).arrayBuffer()),
        };
      }
    } catch {
      // half there: look in the Cache API, then convert again
    }
  }
  const cache = await openCache();
  const manifest = await (await cache?.match(cacheKey(name, "manifest.json")))?.json();
  if (!manifest) return undefined;
  return {
    manifest,
    where: "cache",
    async *parts() {
      for (let part = 0; part < manifest.parts; part++) {
        const stored = await cache.match(cacheKey(name, `part-${String(part).padStart(3, "0")}`));
        if (!stored) throw new Error("the browser has evicted a part of the kept model");
        yield new Uint8Array(await stored.arrayBuffer());
      }
    },
    tokenizer: async () => new Uint8Array(await (await cache.match(cacheKey(name, "tokenizer.bin"))).arrayBuffer()),
  };
}

/** Keep a conversion: slice(begin, end) gives the checkpoint's bytes (a copy), tokenizer is a Uint8Array.
 * Returns why nothing was kept, or undefined. Whatever was half written is removed again. Nothing asks the browser
 * beforehand how much room there is: its estimate said yes where the write then failed (T60). */
export async function keep(model, manifest, slice, tokenizer, signal) {
  const name = keptName(model);
  if (opfsWritable()) {
    const directory = await folders(true);
    if (directory) {
      try {
        const folder = await directory.getDirectoryHandle(name, { create: true });
        for (const [file, write] of [["model.bin", (handle) => {
          for (let offset = 0; offset < manifest.bytes; offset += PART_BYTES) {
            handle.write(slice(offset, Math.min(offset + PART_BYTES, manifest.bytes)), { at: offset });
          }
        }], ["tokenizer.bin", (handle) => handle.write(tokenizer, { at: 0 })],
        ["manifest.json", (handle) => handle.write(new TextEncoder().encode(JSON.stringify(manifest)), { at: 0 })]]) {
          const handle = await (await folder.getFileHandle(file, { create: true })).createSyncAccessHandle();
          try {
            handle.truncate(0);
            write(handle);
            handle.flush();
          } finally {
            handle.close();
          }
          signal?.throwIfAborted();
        }
        return undefined;
      } catch (error) {
        await directory.removeEntry(name, { recursive: true }).catch(() => {});
        if (signal?.aborted) throw error;
        return String(error.message ?? error);
      }
    }
  }
  const cache = await openCache();
  if (!cache) return "this browser has neither the origin private file system nor the Cache API here";
  try {
    const parts = Math.ceil(manifest.bytes / PART_BYTES);
    for (let part = 0; part < parts; part++) {
      const copy = slice(part * PART_BYTES, Math.min((part + 1) * PART_BYTES, manifest.bytes));
      await cache.put(cacheKey(name, `part-${String(part).padStart(3, "0")}`), new Response(copy));
      signal?.throwIfAborted();
    }
    await cache.put(cacheKey(name, "tokenizer.bin"), new Response(tokenizer));
    await cache.put(cacheKey(name, "manifest.json"), new Response(JSON.stringify({ ...manifest, parts }), { headers: { "Content-Type": "application/json" } }));
    return undefined;
  } catch (error) {
    await forget({ name, where: "cache" });
    if (signal?.aborted) throw error;
    return String(error.message ?? error);
  }
}

/** Delete a kept model (one of keptModels()). */
export async function forget({ name, where }) {
  if (where === "opfs") {
    await (await folders())?.removeEntry(name, { recursive: true }).catch(() => {});
    return;
  }
  const cache = await openCache();
  for (const request of (await cache?.keys()) ?? []) {
    if (request.url.startsWith(cacheKey(name, ""))) await cache.delete(request);
  }
}
