// kept-check.mjs (T99): public/kept.js in Node, on small stand-ins for the origin private file system and the Cache
// API (Node has neither). What a browser does with them is for CI (tests/e2e.mjs with E2E_TWICE); this holds the
// logic: a kept model reads back the same bytes, OPFS comes before the Cache API, the Cache API is the fallback and
// still read, a write that fails leaves nothing, and forget() removes what the list shows.
//
//   node tests/kept-check.mjs
import assert from "node:assert/strict";

// ---- stand-ins: a directory tree of files, and a cache of responses
class File {
  constructor() { this.bytes = new Uint8Array(0); }
}
class FileHandle {
  constructor(file, limit) { this.file = file; this.limit = limit; }
  async getFile() {
    const bytes = this.file.bytes;
    const blob = { size: bytes.length, slice: (a, b) => ({ arrayBuffer: async () => bytes.slice(a, b).buffer }),
      text: async () => new TextDecoder().decode(bytes), arrayBuffer: async () => bytes.slice().buffer };
    return blob;
  }
  async createSyncAccessHandle() {
    const file = this.file, limit = this.limit;
    return {
      truncate(n) { file.bytes = file.bytes.slice(0, n); },
      write(data, { at }) {
        const end = at + data.length;
        if (end > limit.bytes) throw new Error("QuotaExceededError");
        if (end > file.bytes.length) { const grown = new Uint8Array(end); grown.set(file.bytes); file.bytes = grown; }
        file.bytes.set(data, at);
        return data.length;
      },
      flush() {}, close() {},
    };
  }
}
class DirectoryHandle {
  constructor(limit) { this.children = new Map(); this.limit = limit; }
  async getDirectoryHandle(name, { create = false } = {}) {
    if (!this.children.has(name)) { if (!create) throw new Error("NotFoundError"); this.children.set(name, new DirectoryHandle(this.limit)); }
    return this.children.get(name);
  }
  async getFileHandle(name, { create = false } = {}) {
    if (!this.children.has(name)) { if (!create) throw new Error("NotFoundError"); this.children.set(name, new File()); }
    return new FileHandle(this.children.get(name), this.limit);
  }
  async removeEntry(name) { if (!this.children.delete(name)) throw new Error("NotFoundError"); }
  async *entries() { for (const [name, child] of this.children) if (child instanceof DirectoryHandle) yield [name, child]; }
}
class Cache {
  constructor() { this.map = new Map(); }
  async put(key, response) { this.map.set(key, new Uint8Array(await response.arrayBuffer())); }
  async match(key) { const bytes = this.map.get(typeof key === "string" ? key : key.url); return bytes && new Response(bytes); }
  async keys() { return [...this.map.keys()].map((url) => ({ url })); }
  async delete(request) { return this.map.delete(request.url); }
}

function browser({ opfs = true, room = Infinity } = {}) {
  const limit = { bytes: room };
  const root = new DirectoryHandle(limit), cache = new Cache();
  globalThis.self = { location: { origin: "https://example.test" } };
  globalThis.caches = { open: async () => cache };
  Object.defineProperty(globalThis, "navigator", { value: { storage: { getDirectory: async () => root } }, configurable: true });
  globalThis.FileSystemFileHandle = opfs ? FileHandle : undefined;
  if (!opfs) globalThis.navigator.storage.getDirectory = async () => { throw new Error("no OPFS"); };
  return { root, cache };
}

const kept = await import("../public/kept.js");
const model = { id: "m", name: "M", hf: { repo: "a/b", revision: "0123" }, conversion: {} };
const bytes = Uint8Array.from({ length: 20 * 1024 * 1024 + 5 }, (_, i) => (i * 7) & 255);  // three parts, the last short
const manifest = { id: "m", name: "M", repo: "a/b", revision: "0123", bytes: bytes.length, options: { dtype: "int8" } };
const vocabulary = Uint8Array.of(1, 2, 3);
const readBack = async (found) => {
  const out = new Uint8Array(found.manifest.bytes);
  let at = 0;
  for await (const part of found.parts()) { out.set(part, at); at += part.length; }
  return out;
};

// OPFS: kept, listed, read back the same, forgotten
{
  const { root, cache } = browser();
  assert.equal(await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary), undefined);
  assert.equal(cache.map.size, 0, "nothing goes to the Cache API where OPFS is");
  const list = await kept.keptModels();
  assert.deepEqual(list.map(({ where, manifest: m }) => [where, m.bytes]), [["opfs", bytes.length]]);
  const found = await kept.openKept(model);
  assert.deepEqual(await readBack(found), bytes);
  assert.deepEqual(await found.tokenizer(), vocabulary);
  await kept.forget(list[0]);
  assert.equal((await kept.keptModels()).length, 0);
  assert.equal((await root.getDirectoryHandle("converted-v1")).children.size, 0);
}
// no OPFS: the Cache API, as before T99, and read back the same
{
  const { cache } = browser({ opfs: false });
  assert.equal(await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary), undefined);
  assert.equal(cache.map.size, 3 + 2, "three parts, the tokenizer and the manifest");
  const list = await kept.keptModels();
  assert.deepEqual(list.map(({ where }) => where), ["cache"]);
  assert.deepEqual(await readBack(await kept.openKept(model)), bytes);
  await kept.forget(list[0]);
  assert.equal(cache.map.size, 0);
}
// a model kept in the Cache API before T99 is still found where OPFS is now, and not moved
{
  const { cache } = browser({ opfs: false });
  await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  const entries = [...cache.map];
  const { cache: later } = browser();
  entries.forEach(([key, value]) => later.map.set(key, value));
  assert.deepEqual((await kept.keptModels()).map(({ where }) => where), ["cache"]);
  assert.deepEqual(await readBack(await kept.openKept(model)), bytes);
}
// a write that runs out of room leaves nothing, and says why
{
  const { root } = browser({ room: 10 * 1024 * 1024 });
  const why = await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  assert.match(why, /Quota/);
  assert.equal((await root.getDirectoryHandle("converted-v1")).children.size, 0);
  assert.equal(await kept.openKept(model), undefined);
}
// a write the browser says it did but did not keep: found out, and nothing is left
{
  const { root } = browser();
  const original = FileHandle.prototype.createSyncAccessHandle;
  FileHandle.prototype.createSyncAccessHandle = async function () {
    const handle = await original.call(this);
    return { ...handle, write: (data) => data.length };  // says it wrote, holds nothing
  };
  const why = await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  FileHandle.prototype.createSyncAccessHandle = original;
  assert.match(why, /holds 0 of/);
  assert.equal((await root.getDirectoryHandle("converted-v1")).children.size, 0);
}
// a folder without its manifest (a write that did not finish) is not a kept model
{
  const { root } = browser();
  const folder = await (await root.getDirectoryHandle("converted-v1", { create: true })).getDirectoryHandle(kept.keptName(model), { create: true });
  await folder.getFileHandle("model.bin", { create: true });
  assert.equal((await kept.keptModels()).length, 0);
  assert.equal(await kept.openKept(model), undefined);
}
console.log("ok");
