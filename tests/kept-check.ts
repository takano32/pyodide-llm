// kept-check.ts (T.99): src/runtime/kept.js in Node, on small stand-ins for the origin private file system and the Cache
// API (Node has neither). What a browser does with them is for CI (tests/e2e.ts with E2E_TWICE); this holds the
// logic: a kept model reads back the same bytes, OPFS comes before the Cache API, the Cache API is the fallback and
// still read, a write that fails leaves nothing, and forget() removes what the list shows.
//
//   node tests/kept-check.ts
import assert from "node:assert/strict";
import { cacheStorage } from "./fakes/cache.ts";
import { runtimeUrl } from "./tree.ts";

// ---- stand-ins: a directory tree of files (the Cache API is tests/fakes/cache.ts's)
// (T.329.2.2: a file has one sync access handle at most, as in a browser: a second is refused while it is open, and a
// folder that holds an open one is not removed)
class File {
  constructor() { this.bytes = new Uint8Array(0); this.open = false; }
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
    if (file.open) throw new Error("NoModificationAllowedError: the file has a sync access handle open");
    file.open = true;
    return {
      read(data, { at }) {
        if (!file.open) throw new Error("InvalidStateError: closed");
        const got = file.bytes.subarray(at, at + data.length);
        data.set(got);
        return got.length;
      },
      getSize() { return file.bytes.length; },
      truncate(n) { file.bytes = file.bytes.slice(0, n); },
      write(data, { at }) {
        const end = at + data.length;
        if (end > limit.bytes) throw new Error("QuotaExceededError");
        if (end > file.bytes.length) { const grown = new Uint8Array(end); grown.set(file.bytes); file.bytes = grown; }
        file.bytes.set(data, at);
        return data.length;
      },
      flush() {}, close() { file.open = false; },
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
  async removeEntry(name) {
    const child = this.children.get(name);
    const busy = (entry) => (entry instanceof DirectoryHandle ? [...entry.children.values()].some(busy) : entry?.open);
    if (busy(child)) throw new Error("NoModificationAllowedError: a file in it is open");
    if (!this.children.delete(name)) throw new Error("NotFoundError");
  }
  async *entries() { for (const [name, child] of this.children) if (child instanceof DirectoryHandle) yield [name, child]; }
}
function browser({ opfs = true, room = Infinity } = {}) {
  const limit = { bytes: room };
  const root = new DirectoryHandle(limit), { caches, map } = cacheStorage();
  globalThis.self = { location: { origin: "https://example.test" } };
  globalThis.caches = caches;
  Object.defineProperty(globalThis, "navigator", { value: { storage: { getDirectory: async () => root } }, configurable: true });
  globalThis.FileSystemFileHandle = opfs ? FileHandle : undefined;
  if (!opfs) globalThis.navigator.storage.getDirectory = async () => { throw new Error("no OPFS"); };
  return { root, cache: { map } };
}

const kept = await import(runtimeUrl("kept.ts"));
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
// no OPFS: the Cache API, as before T.99, and read back the same
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
// a model kept in the Cache API before T.99 is still found where OPFS is now, and not moved
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
// a write that comes back short (the room ran out without an exception): found out, and nothing is left
{
  const { root } = browser();
  const original = FileHandle.prototype.createSyncAccessHandle;
  FileHandle.prototype.createSyncAccessHandle = async function () {
    const handle = await original.call(this);
    return { ...handle, write: (data, at) => { handle.write(data.subarray(0, data.length - 1), at); return data.length - 1; } };
  };
  const why = await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  FileHandle.prototype.createSyncAccessHandle = original;
  assert.match(why, /only part of a write/);
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
// T.116: the bits. A model kept as int8 serves one that asks for int8 or for no bits (the worker chooses, T.115), never
// one that asks for six bits: that one is converted again, and the page does not call it kept
{
  browser();
  const as = (dtype) => ({ ...model, conversion: dtype ? { dtype } : {} });
  assert.equal(await kept.keep(as("int8"), manifest, (a, b) => bytes.slice(a, b), vocabulary), undefined);
  const [one] = await kept.keptModels();
  assert.equal(one.manifest.converter, kept.CONVERTER, "the manifest says which converter made it");
  assert.ok(kept.serves(one, as("int8")) && kept.serves(one, as(undefined)) && !kept.serves(one, as("int6")));
  assert.equal(await kept.openKept(as("int6")), undefined);
  assert.deepEqual(await readBack(await kept.openKept(as(undefined))), bytes);
  // one the worker chose six bits for is found with no bits asked too
  await kept.forget(one);
  await kept.keep(as("int6"), manifest, (a, b) => bytes.slice(a, b), vocabulary);
  assert.deepEqual(await readBack(await kept.openKept(as(undefined))), bytes);
  assert.equal(await kept.openKept(as("int8")), undefined);
  // T.230: a ternary model is converted to the ternary dtype (the page asks for it by name: src/models.ts's
  // weightsFor), kept under that name. What was kept of it as int8 before serves ?bits=8 alone, never the ternary one
  const [six] = await kept.keptModels();
  assert.ok(!kept.serves(six, as("ternary")));
  assert.equal(await kept.openKept(as("ternary")), undefined);
  await kept.keep(as("ternary"), manifest, (a, b) => bytes.slice(a, b), vocabulary);
  assert.deepEqual(await readBack(await kept.openKept(as("ternary"))), bytes);
  const ternary = (await kept.keptModels()).find((one) => one.name === kept.keptName(as("ternary")));
  assert.ok(ternary && kept.serves(ternary, as("ternary")) && !kept.serves(ternary, as(undefined)) && !kept.serves(ternary, as("int8")));
}
// T.116: the converter. What an older one kept is not used and is deleted; what a newer one kept (a tab of an older
// page next to a newer one) is not used either, and is left alone
{
  const { root } = browser();
  const manifestOf = async () => (await (await root.getDirectoryHandle("converted-v1")).getDirectoryHandle(kept.keptName(model))).children.get("manifest.json");
  await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  (await manifestOf()).bytes = new TextEncoder().encode(JSON.stringify(manifest));  // as before T.116: no converter
  const [old] = await kept.keptModels();
  assert.ok(kept.outdated(old) && !kept.serves(old, model));
  assert.equal(await kept.openKept(model), undefined);
  assert.equal((await kept.keptModels()).length, 0, "the older one is deleted");
  await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  (await manifestOf()).bytes = new TextEncoder().encode(JSON.stringify({ ...manifest, converter: kept.CONVERTER + 1 }));
  const [newer] = await kept.keptModels();
  assert.ok(!kept.outdated(newer) && !kept.serves(newer, model));
  assert.equal(await kept.openKept(model), undefined);
  assert.equal((await kept.keptModels()).length, 1, "the newer one stays");
}
// T.136: a GGUF's weights with another repository's vocabulary and config.json is kept under both: the same GGUF with
// another vocabulary is another conversion. Without one, the name is what it always was
{
  const withVocabulary = { ...model, hf: { ...model.hf, vocabulary: { repo: "c/d", revision: "4567", tokenizer: "tokenizer.model" } } };
  assert.equal(decodeURIComponent(kept.keptName(model)), "a/b@0123:int8:4096");
  assert.equal(decodeURIComponent(kept.keptName(withVocabulary)), "a/b@0123+c/d@4567:int8:4096");
  // T.363.5: a conversion that kept a GGUF's blocks is another than the same file widened, and neither serves the other
  const blocks = { ...model, conversion: { dtype: "int8", blocks: true } };
  assert.equal(decodeURIComponent(kept.keptName(blocks)), "a/b@0123:int8:4096:blocks");
  assert.equal(kept.keptName({ ...model, conversion: { dtype: "int8", blocks: false } }), kept.keptName({ ...model, conversion: { dtype: "int8" } }));
  assert.deepEqual(kept.keptNames(blocks).map(decodeURIComponent), ["a/b@0123:int8:4096:blocks"]);
  const one = (name) => ({ name, where: "opfs", manifest: { converter: kept.CONVERTER } });
  assert.ok(kept.serves(one(kept.keptName(blocks)), blocks) && !kept.serves(one(kept.keptName(blocks)), { ...model, conversion: { dtype: "int8" } }) &&
    !kept.serves(one(kept.keptName(model)), blocks), "a model's blocks kept serve the one that asks for them, and no other");
}
// T.136's review: a model whose source changed (the 19 of stage 2) keeps its new conversion under a new name. What it
// was kept as before is replaced: never served, and in the way of the new one where the room is short. Its other
// bits are not, nor another model, nor what ?hf= and folders keep (all "local")
{
  browser();
  const moved = { ...model, hf: { repo: "a/b-GGUF", revision: "89ab", vocabulary: { repo: "a/b", revision: "0123" } } };
  const keepAs = (entry, dtype, id = entry.id) =>
    kept.keep({ ...entry, conversion: { dtype } }, { ...manifest, id }, (a, b) => bytes.slice(a, b), vocabulary);
  await keepAs(model, "int8");
  await keepAs(model, "int6");
  await keepAs({ ...model, hf: { repo: "x/y", revision: "1" } }, "int8", "other");
  await keepAs({ ...model, id: "local", hf: { repo: "p/q", revision: "2" } }, "int8");
  await keepAs(moved, "int6");
  const names = (list) => list.map((one) => decodeURIComponent(one.name)).sort();
  assert.deepEqual(names(await kept.replaced(moved)), ["a/b@0123:int6:4096", "a/b@0123:int8:4096"]);
  assert.deepEqual(names(await kept.replaced({ ...moved, conversion: { dtype: "int8" } })), names(await kept.replaced(moved)),
    "asking for eight bits does not make its own six-bit conversion replaced");
  assert.deepEqual(await kept.replaced({ ...model, id: "local", hf: { repo: "r/s", revision: "3" } }), []);
  // the review of T.230: a ternary model's ?bits=8 (or =6) visit converts it to int8, and must leave its ternary one:
  // the next visit without ?bits asks for the ternary one again (484 MB of Bonsai 1.7B and 2.3 GB of the 8B fetched and
  // converted again, for each ?bits=8 and plain visit in turn, when it was replaced). Nor does the ternary visit
  // replace the int8 and int6 conversions of the same source
  // T.375.5: kept.ts names the dtypes itself (the page reads it without the runtime): every packing of jobs.ts's PACKED
  // is one of them. A packing that is added there is added to replaced()'s names in src/runtime/kept.ts
  const { PACKED } = await import(runtimeUrl("jobs.js"));
  assert.ok(["ternary", "int8", "int6"].every((dtype) => Object.hasOwn(PACKED, dtype)));
  // (T.363.5: the packings a file's dtype may be: llama.cpp's blocks are a row's alone, kept under conversion.blocks)
  const dtypes = Object.keys(PACKED).filter((name) => PACKED[name].tabled === undefined);
  for (const dtype of dtypes) await keepAs(moved, dtype);
  // (and the same source with its blocks kept: its own too, whichever is asked for)
  await kept.keep({ ...moved, conversion: { dtype: "int8", blocks: true } }, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  for (const blocks of [true, false]) {
    assert.deepEqual(names(await kept.replaced({ ...moved, conversion: { dtype: "int8", blocks } })), ["a/b@0123:int6:4096", "a/b@0123:int8:4096"],
      `asking for the blocks ${blocks ? "kept" : "widened"} replaces neither of the two`);
  }
  for (const dtype of [...dtypes, undefined]) {
    assert.deepEqual(names(await kept.replaced({ ...moved, conversion: { dtype } })), ["a/b@0123:int6:4096", "a/b@0123:int8:4096"],
      `asking for ${dtype ?? "no bits"} replaces what its source kept before, and none of its own other bits ` +
      "(a packing of jobs.ts's PACKED that kept.ts's replaced() does not name is deleted here: add its name there)");
  }
}
// the review of T.203: a Qwen3's two forms share one conversion, kept under the id of the form that converted it. When
// their source changes, opening the other form first replaces it too (it left a 9 GB conversion of the 8B behind)
{
  browser();
  const pair = ["q-thinking", "q"];
  const before = { ...model, id: "q-thinking", hf: { repo: "a/q", revision: "1" } };
  const after = { ...model, id: "q", shares: pair, hf: { repo: "a/q-GGUF", revision: "2", vocabulary: { repo: "a/q", revision: "1" } } };
  await kept.keep({ ...before, conversion: { dtype: "int8" } }, { ...manifest, id: "q-thinking" }, (a, b) => bytes.slice(a, b), vocabulary);
  await kept.keep({ ...model, id: "r", conversion: { dtype: "int8" } }, { ...manifest, id: "r" }, (a, b) => bytes.slice(a, b), vocabulary);
  const names = (list) => list.map((one) => decodeURIComponent(one.name));
  assert.deepEqual(names(await kept.replaced(after)), ["a/q@1:int8:4096"], "the other form's conversion is replaced");
  assert.deepEqual(names(await kept.replaced({ ...after, id: "q-thinking" })), ["a/q@1:int8:4096"]);
  assert.deepEqual(await kept.replaced({ ...after, shares: undefined }), [], "without shares only its own id's");
}
// T.329.2.2: a conversion kept as it comes whose table of every layer's inputs the engine reads from the file: the file
// stays open past finish() and is read from at once; let go of, a kept one opened again (one handle at a time); one that
// ran out of room after the table was whole is read on and removed once let go of, one that ran out before is not whole
{
  const { root } = browser();
  const hold = [3 * 1024 * 1024, 9 * 1024 * 1024 + 11];
  const keeping = await kept.keeper(model);
  keeping.holding(hold);
  for (let at = 0; at < bytes.length; at += 1 << 20) keeping.write(at, bytes.subarray(at, Math.min(at + (1 << 20), bytes.length)));
  assert.equal(keeping.whole(), true);
  assert.equal(await keeping.finish(manifest, vocabulary), undefined);
  const row = new Uint8Array(100);
  assert.equal(keeping.read(row, hold[0] + 5), 100);
  assert.deepEqual(row, bytes.subarray(hold[0] + 5, hold[0] + 105), "read from the file at once, past finish()");
  // (kept: listed, and its file not to be opened again while the engine holds it)
  assert.deepEqual((await kept.keptModels()).map(({ where }) => where), ["opfs"]);
  const name = (await kept.keptModels())[0].name;
  assert.equal(await kept.openRows(name), undefined, "a second handle of the file while the first is open");
  await keeping.close();
  assert.throws(() => keeping.read(row, 0), /closed/);
  // the next visit: its parts, the stretch skipped (zeros where it is), and then the file opened to read from
  const found = await kept.openKept(model);
  assert.equal(found.name, name);
  const skipping = { range: [8 * 1024 * 1024, 16 * 1024 * 1024] };
  const parts = [];
  for await (const part of found.parts(skipping)) parts.push(part);
  assert.deepEqual(parts.map((part) => part.length), [8 << 20, 8 << 20, (4 << 20) + 5]);
  assert.ok(parts[1].every((byte) => byte === 0) && parts[0][1] === bytes[1] && parts[2][3] === bytes[(16 << 20) + 3], "the parts inside the stretch alone are zeros");
  const reader = await kept.openRows(name);
  assert.equal(reader.read(row, 77), 100);
  assert.deepEqual(row, bytes.subarray(77, 177));
  assert.equal(await kept.openRows(name), undefined, "one handle at a time");
  reader.close();
  const again = await kept.openRows(name);
  assert.ok(again, "the file opens again once let go of");
  again.close();
  assert.equal(await kept.openRows("nothing-of-that-name"), undefined);
  // room for the table and not the rest: not kept, but read on; the file goes once let go of
  for (const [room, whole] of [[12 * 1024 * 1024, true], [5 * 1024 * 1024, false]]) {
    const small = browser({ room });
    const short = await kept.keeper(model);
    short.holding(hold);
    for (let at = 0; at < bytes.length; at += 1 << 20) short.write(at, bytes.subarray(at, Math.min(at + (1 << 20), bytes.length)));
    assert.equal(short.whole(), whole);
    assert.match(await short.finish(manifest, vocabulary), /Quota/);
    assert.deepEqual(await kept.keptModels(), [], "nothing kept");
    if (whole) {
      assert.equal(short.read(row, hold[0]), 100);
      assert.deepEqual(row, bytes.subarray(hold[0], hold[0] + 100));
      assert.equal(small.root.children.get("converted-v1").children.size, 1, "its file is there while it is read");
      await short.close();
    }
    assert.equal(small.root.children.get("converted-v1").children.size, 0, "and gone once let go of");
  }
  void root;
}
// T.521 (the review of T.329.2.2): a conversion that opens again in the same keeper (another tokenizer's try) tells it
// the stretch again, and the bytes written again are not counted twice: half the table and then all of it is whole
{
  browser();
  const hold = [3 * 1024 * 1024, 9 * 1024 * 1024 + 11];
  const twice = await kept.keeper(model);
  twice.holding(hold);
  for (let at = 0; at < 6 << 20; at += 1 << 20) twice.write(at, bytes.subarray(at, at + (1 << 20)));
  assert.equal(twice.whole(), false);
  twice.holding(hold);
  for (let at = 0; at < bytes.length; at += 1 << 20) twice.write(at, bytes.subarray(at, Math.min(at + (1 << 20), bytes.length)));
  assert.equal(twice.whole(), true, "the stretch written again is whole, not counted twice");
  await twice.drop();
}
console.log("ok");
