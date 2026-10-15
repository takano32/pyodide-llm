// kept.ts (T99): the models this browser converted from Hugging Face, kept for the next visit. The worker writes
// and reads them; the page lists them and deletes them. Two places, one interface:
//
//   the origin private file system (OPFS), where the browser has it: a folder per model with model.bin (the whole
//   checkpoint in one file), tokenizer.bin and manifest.json. The worker writes and reads with a sync access handle.
//   The Cache API, as before T99 (converted-v1): parts of 8 MiB, the tokenizer and the manifest, under one URL
//   prefix per model. New conversions go there only where there is no OPFS; what a visitor kept there before is read
//   and deleted as it always was, never moved.
//
// The manifest is written last in both: a folder without one is rubbish from a write that did not finish.
// TypeScript of which only the types are erased (T368.1): the worker imports it, and so does the page.

import type { Options } from "./contracts/python.ts";

// (T368.1, the types. What is read here of a model of the list: where its weights come from and how they are converted.)
export type Kept = { name: string; where: "opfs" | "cache"; manifest: Manifest };
// what is written beside a kept model: what the worker says of the conversion (worker/convert.ts: the model's id and
// name, where it came from, the checkpoint's bytes, the options the engine gets, when), and this file's own two: the
// converter's version and, in the Cache API, the number of parts
export type Conversion = { id: string; name: string; repo: string; revision: string; bytes: number; options: Options; saved: number };
export type Manifest = Conversion & { converter?: number; parts?: number };
export type Converted = { id?: string; shares?: string[]; conversion?: { dtype?: string; max_seq_len?: number; blocks?: boolean };
  hf: { repo: string; revision: string; weights?: string; vocabulary?: { repo: string; revision: string } } };
// a file of the origin private file system as a worker has it, and its synchronous access (the page's types know neither)
type Access = { write(bytes: Uint8Array, options: { at: number }): number; truncate(size: number): void; flush(): void; close(): void;
  read(bytes: Uint8Array, options: { at: number }): number; getSize(): number };
// T329.2.2: what reads bytes of a kept checkpoint at once (a sync access handle's read()), and lets go of it
export type Reader = { read(bytes: Uint8Array, at: number): number; close(): void };
type WorkerFile = FileSystemFileHandle & { createSyncAccessHandle(): Promise<Access> };

export const CACHE_NAME = "converted-v1";
const FOLDER = "converted-v1";
const PART_BYTES = 8 * 1024 * 1024;

// the name of a model's conversion: its repository, revision, dtype and context, as the Cache API key has it, and
// where the vocabulary and config.json come from another repository than the weights (T136), that one too.
// T363.5: a conversion that kept a GGUF's blocks as they are (conversion.blocks) is another one than the same file
// widened, though both say the dtype int8 (how the rows that are no blocks are stored): ":blocks" after its name. Every
// other name is what it was.
// T505.3: a repository nobody listed (?hf=, the sheet: the id "local") may be opened by one file of it or by another
// (&file=: its Q8_0.gguf, its Q4_K_M.gguf), and each is another conversion: the file's name after the revision, unless
// it is the model.safetensors every such repository was opened by before. A model of the list is one file, and its
// name is what it was
export function keptName(model: Converted) {
  const { dtype = "int8", max_seq_len = 4096, blocks = false } = model.conversion ?? {};
  const { vocabulary, weights } = model.hf;
  const from = vocabulary ? `+${vocabulary.repo}@${vocabulary.revision}` : "";
  const file = model.id === "local" && weights && weights !== "model.safetensors" ? `/${weights}` : "";
  return encodeURIComponent(`${model.hf.repo}@${model.hf.revision}${file}${from}:${dtype}:${max_seq_len}${blocks ? ":blocks" : ""}`);
}
/** T116: the version of what llama2_convert.py writes, the checkpoint's bytes and the options the engine gets.
 * Raise it when either changes: a conversion kept by an older converter is then converted again, and deleted (the
 * options of T106, BOS and specials, stayed wrong in what was kept before). 1: the manifests without it. 3: T127,
 * the templates the converter reads now (selectattr, namespace(), chat_template.jinja, trim_blocks) in the options;
 * 4: and the control pieces of a sentencepiece model as their specials; 5: the nmt and collapse of a sentencepiece
 * model's normalizer (the review of T126: rinna's newlines); 6: a template's strftime_now() as {date:format}, filled
 * when the prompt is sent, and or / and, comments and escapes as Jinja reads them (the review of T127); 7: the
 * rms_norm_eps of a model whose is not 1e-5 (T124: the Qwen2.5 of the list, TinySwallow, DeepSeek-R1 and llm-jp-4, at 1e-6);
 * 8: {prompt:trim} for a template that trims what was typed (T138: Llama 3.1 and 3.2, Swallow 8B); 9: the batch the
 * owner held for one version (2026-09-28): the added tokens that are not special as specials, the BOS the tokenizer
 * names, a GGUF's padding as empty pieces, a byte-level tokenizer.bin's own U+2581 (T143), and a sentencepiece
 * model's charsmap after the pieces in place of the nfkc and nmt options (T216). 10: the round of T369 (2026-10-10): the
 * token a text begins with where the model's chat template was read (T264: the template's own first token, so that the
 * page sends the ids of apply_chat_template), lower case for a sentencepiece model that asks for it (T265), the dummy
 * prefix after a special token (T308), the templates jinja2 reads and the reader did not (T397, and T269's), and
 * qk_norm said by a Llama's file alone. */
export const CONVERTER = 10;
const converterOf = (manifest: Manifest) => manifest.converter ?? 1;
/** The names a model's conversion may be kept under: its bits, or with none asked for, either of the two the worker
 * may choose (T115) */
export function keptNames(model: Converted) {
  const asked = model.conversion?.dtype;
  return (asked ? [asked] : ["int8", "int6"]).map((dtype) => keptName({ ...model, conversion: { ...model.conversion, dtype } }));
}
/** Whether a kept conversion (one of keptModels()) serves a model: kept under one of its names, by this converter */
export const serves = (kept: Kept, model: Converted) => keptNames(model).includes(kept.name) && converterOf(kept.manifest) === CONVERTER;
/** Whether a kept conversion is of an older converter: not used, and deleted. (One of a newer converter is left
 * alone: a tab of the older page must not delete what the newer one kept.) */
export const outdated = (kept: Pick<Kept, "manifest">) => converterOf(kept.manifest) < CONVERTER;
/** The conversions kept for a model of the list under a name it has no more, of either bits (T136: its weights come
 * from another repository now, or its vocabulary): never served again, and in the way of the new one (a 7B's 8 GB
 * each). The worker deletes them before it keeps the new one. Not those of ?hf= and of folders, which all have the
 * id "local". shares: the ids of the list that share one conversion (a Qwen3's two forms, T124), kept under the id of
 * whichever converted it (the review of T203: the other form's left its 9 GB behind). */
export async function replaced(model: Converted) {
  if (model.id === "local") return [];
  // its own conversions of every bits stay, whichever one is asked for: the ternary one too (the review of T230: a
  // ?bits=8 visit of a ternary model deleted its ternary conversion, and the next plain visit fetched and converted it
  // again, for each visit in turn)
  // (T363.5: and both of a model whose blocks are kept: with its blocks, and widened, which a load without the int8
  // kernels asks for: ?without=kernels, a browser without WebAssembly SIMD)
  const names = [...keptNames(model), ...[false, true].flatMap((blocks) => ["int8", "int6", "ternary"].map((dtype) => keptName({ ...model, conversion: { ...model.conversion, dtype, blocks } })))];
  const ids = model.shares ?? [model.id];
  return (await keptModels()).filter((kept) => ids.includes(kept.manifest.id) && !names.includes(kept.name));
}
const cacheKey = (name: string, file: string) => `${self.location.origin}/converted/${name}/${file}`;

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
  const found: Kept[] = [];
  const directory = await folders();
  if (directory) {
    for await (const [name, handle] of directory.entries()) {
      try {
        const file = await (await (handle as FileSystemDirectoryHandle).getFileHandle("manifest.json")).getFile();
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
      found.push({ name: match[1], where: "cache", manifest: await (await cache!.match(request))!.json() });
    }
  }
  return found;
}

/** A kept model to read: { manifest, where, parts(), tokenizer() }, or undefined. parts() yields its bytes in order.
 * The first of its names (keptNames) kept by this converter; what an older one kept is deleted on the way (T116). */
export async function openKept(model: Converted) {
  for (const name of keptNames(model)) {
    const found = await openNamed(name);
    if (found && converterOf(found.manifest) === CONVERTER) return found;
    if (found && outdated(found)) await forget({ name, where: found.where });
  }
  return undefined;
}
async function openNamed(name: string) {
  const directory = await folders();
  const folder = await directory?.getDirectoryHandle(name).catch(() => undefined);
  if (folder) {
    try {
      const manifest = JSON.parse(await (await (await folder.getFileHandle("manifest.json")).getFile()).text());
      const file = await (await folder.getFileHandle("model.bin")).getFile();
      if (file.size === manifest.bytes) {
        return {
          manifest,
          where: "opfs" as const,
          name,
          // (T329.2.2, skip: a stretch nothing reads from the memory, whose parts are not read: zeros in their place)
          async *parts(skip: { range?: [number, number] } = {}) {
            for (let offset = 0; offset < file.size; offset += PART_BYTES) {
              const end = Math.min(offset + PART_BYTES, file.size), range = skip.range;
              yield range && range[0] <= offset && end <= range[1] ? new Uint8Array(end - offset) : new Uint8Array(await file.slice(offset, end).arrayBuffer());
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
    where: "cache" as const,
    name,
    async *parts(_skip?: { range?: [number, number] }) {
      for (let part = 0; part < manifest.parts; part++) {
        const stored = await cache!.match(cacheKey(name, `part-${String(part).padStart(3, "0")}`));
        if (!stored) throw new Error("the browser has evicted a part of the kept model");
        yield new Uint8Array(await stored.arrayBuffer());
      }
    },
    tokenizer: async () => new Uint8Array(await (await cache!.match(cacheKey(name, "tokenizer.bin")))!.arrayBuffer()),
  };
}

/** Keep a conversion: slice(begin, end) gives the checkpoint's bytes (a copy), tokenizer is a Uint8Array.
 * Returns why nothing was kept, or undefined. Whatever was half written is removed again. Nothing asks the browser
 * beforehand how much room there is: its estimate said yes where the write then failed (T60). */
export async function keep(model: Converted, kept: Conversion, slice: (begin: number, end: number) => Uint8Array<ArrayBuffer>, tokenizer: Uint8Array<ArrayBuffer>, signal?: AbortSignal) {
  const name = keptName(model), manifest = { ...kept, converter: CONVERTER };
  if (opfsWritable()) {
    const directory = await folders(true);
    if (directory) {
      try {
        const folder = await directory.getDirectoryHandle(name, { create: true });
        // a write may come back short instead of throwing when the room runs out: that is a failure too
        const put = (handle: Access, bytes: Uint8Array, at: number) => {
          if (handle.write(bytes, { at }) !== bytes.length) throw new Error("the origin private file system took only part of a write");
        };
        for (const [file, write] of [["model.bin", (handle) => {
          for (let offset = 0; offset < manifest.bytes; offset += PART_BYTES) {
            put(handle, slice(offset, Math.min(offset + PART_BYTES, manifest.bytes)), offset);
          }
        }], ["tokenizer.bin", (handle) => put(handle, tokenizer, 0)],
        ["manifest.json", (handle) => put(handle, new TextEncoder().encode(JSON.stringify(manifest)), 0)]] as [string, (handle: Access) => void][]) {
          const handle = await (await folder.getFileHandle(file, { create: true }) as WorkerFile).createSyncAccessHandle();
          try {
            handle.truncate(0);
            write(handle);
            handle.flush();
          } finally {
            handle.close();
          }
          signal?.throwIfAborted();
        }
        // read back what the browser says it holds now: a write that did not stay is not a kept model
        const held = await (await folder.getFileHandle("model.bin")).getFile();
        const written = await (await folder.getFileHandle("manifest.json")).getFile();
        if (held.size !== manifest.bytes || !written.size) {
          throw new Error(`the origin private file system holds ${held.size} of ${manifest.bytes} bytes after writing`);
        }
        return undefined;
      } catch (error) {
        await directory.removeEntry(name, { recursive: true }).catch(() => {});
        if (signal?.aborted) throw error;
        return String((error as Error).message ?? error);
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
    return String((error as Error).message ?? error);
  }
}

/** T156: a conversion kept as it comes, for a model whose weights go to the GPU alone (nothing holds them whole to
 * copy from afterwards, as keep() does): write(offset, bytes) as they come, in any order; finish(kept, tokenizer) writes
 * the tokenizer and the manifest last (a folder without one is rubbish, as keep()'s) and returns why nothing was kept,
 * or undefined; drop() lets go of it. On OPFS alone (a sync access handle): undefined where there is none. Whatever
 * was kept under the name before is removed first (openKept did not find it serving, or there would be no conversion) */
/** T329.2.2: holding(stretch), before the first write: a stretch of the checkpoint the engine reads from the file
 * ([start, end): a table of every layer's inputs, not in the memory). model.bin then stays open past finish(), for
 * read(); close() lets go of it. A conversion that could not be kept may still be read where the stretch was all
 * written before a write failed (whole()): the file is removed once it is let go of. dtype: what it is kept as (its
 * name says it) */
export async function keeper(model: Converted) {
  if (!opfsWritable()) return undefined;
  const directory = await folders(true);
  if (!directory) return undefined;
  const name = keptName(model);
  await directory.removeEntry(name, { recursive: true }).catch(() => {});
  const folder = await directory.getDirectoryHandle(name, { create: true });
  let handle: Access | null = await (await folder.getFileHandle("model.bin", { create: true }) as WorkerFile).createSyncAccessHandle();
  let failed: string | null = null, held = 0, rubbish = false, hold: [number, number] | undefined;
  const put = (target: Access, bytes: Uint8Array, at: number) => {
    if (target.write(bytes, { at }) !== bytes.length) throw new Error("the origin private file system took only part of a write");
  };
  const close = () => {
    handle?.close();
    handle = null;
  };
  // (T329.2.2: the bytes of the stretch held that were written)
  const within = (offset: number, length: number) => (hold ? Math.max(0, Math.min(offset + length, hold[1]) - Math.max(offset, hold[0])) : 0);
  const whole = () => Boolean(hold) && held === hold![1] - hold![0];
  const drop = async () => {
    close();
    await directory.removeEntry(name, { recursive: true }).catch(() => {});
  };
  return {
    dtype: model.conversion?.dtype ?? "int8",
    holding(stretch: [number, number]) {
      hold = stretch;
    },
    write(offset: number, bytes: Uint8Array) {
      if (failed || !handle) return;
      try {
        put(handle, bytes, offset);
        held += within(offset, bytes.length);
      } catch (error) {
        failed = String((error as Error).message ?? error);  // (the model goes on; it is not kept)
      }
    },
    async finish(kept: Conversion, tokenizer: Uint8Array) {
      const manifest = { ...kept, converter: CONVERTER };
      try {
        if (failed) throw new Error(failed);
        handle!.flush();
        // (T329.2.2: open still where the engine reads the stretch held from it)
        if (!hold) close();
        for (const [file, bytes] of [["tokenizer.bin", tokenizer], ["manifest.json", new TextEncoder().encode(JSON.stringify(manifest))]] as [string, Uint8Array][]) {
          const target = await (await folder.getFileHandle(file, { create: true }) as WorkerFile).createSyncAccessHandle();
          try {
            target.truncate(0);
            put(target, bytes, 0);
            target.flush();
          } finally {
            target.close();
          }
        }
        const size = handle ? handle.getSize() : (await (await folder.getFileHandle("model.bin")).getFile()).size;
        if (size !== manifest.bytes) throw new Error(`the origin private file system holds ${size} of ${manifest.bytes} bytes after writing`);
        return undefined;
      } catch (error) {
        // (T329.2.2: a stretch held that was all written is read on: its file goes once it is let go of)
        if (hold && handle && whole()) {
          rubbish = true;
          await directory.removeEntry(name, { recursive: true }).catch(() => {});
        } else {
          await drop();
        }
        return String((error as Error).message ?? error);
      }
    },
    drop,
    /** T329.2.2: bytes of the file at at, at once (the handle it is written by, open past finish() where a stretch is
     * held); how many were read */
    read(bytes: Uint8Array, at: number) {
      if (!handle) throw new Error("the kept checkpoint is closed");
      return handle.read(bytes, { at });
    },
    /** T329.2.2: whether every byte of the stretch held was written */
    whole,
    /** T329.2.2: lets go of the file, and of what is left of it where it was not kept */
    async close() {
      close();
      if (rubbish) await directory.removeEntry(name, { recursive: true }).catch(() => {});
    },
  };
}
/** T329.2.2: what reads a kept checkpoint (openKept()'s, of the origin private file system) at once: a sync access
 * handle of its model.bin, the only one its file can have while it is open (close() lets go of it); undefined where
 * it cannot be had (another tab of this site holds it, or the browser has no such handles) */
export async function openRows(name: string): Promise<Reader | undefined> {
  if (!opfsWritable()) return undefined;
  try {
    const file = await (await (await folders())!.getDirectoryHandle(name)).getFileHandle("model.bin") as WorkerFile;
    const handle = await file.createSyncAccessHandle();
    return { read: (bytes, at) => handle.read(bytes, { at }), close: () => handle.close() };
  } catch {
    return undefined;
  }
}
/** Delete a kept model (one of keptModels()). */
export async function forget({ name, where }: Pick<Kept, "name" | "where">) {
  if (where === "opfs") {
    await (await folders())?.removeEntry(name, { recursive: true }).catch(() => {});
    return;
  }
  const cache = await openCache();
  for (const request of (await cache?.keys()) ?? []) {
    if (request.url.startsWith(cacheKey(name, ""))) await cache!.delete(request);
  }
}
