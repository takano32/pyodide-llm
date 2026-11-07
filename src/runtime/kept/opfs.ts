// kept/opfs.ts (T99, T641.7): the origin private file system as a place a converted model is kept in (kept/place.ts):
// a folder per model with model.bin (the whole checkpoint in one file), tokenizer.bin and manifest.json, the manifest
// written last. The worker writes and reads with a sync access handle, and writes every file one way (written()).
// Here alone: keeper(), a conversion kept as it comes, and openRows(), a kept checkpoint read at once (T156, T329.2.2).

import type { Conversion, Found, Keep, Kept, Manifest, Place, Reader, Slice } from "./place.ts";
import { PART_BYTES } from "./place.ts";

// a file of the origin private file system as a worker has it, and its synchronous access (the page's types know neither)
type Access = { write(bytes: Uint8Array, options: { at: number }): number; truncate(size: number): void; flush(): void; close(): void;
  read(bytes: Uint8Array, options: { at: number }): number; getSize(): number };
type WorkerFile = FileSystemFileHandle & { createSyncAccessHandle(): Promise<Access> };

const FOLDER = "converted-v1";

async function folders(create = false) {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(FOLDER, { create });
  } catch {
    return undefined;  // no OPFS here (or not yet created)
  }
}
// OPFS with the synchronous access a worker needs to write a file of gigabytes piece by piece
const opfsWritable = () => typeof FileSystemFileHandle !== "undefined" && "createSyncAccessHandle" in FileSystemFileHandle.prototype;

const access = async (folder: FileSystemDirectoryHandle, file: string) =>
  (await folder.getFileHandle(file, { create: true }) as WorkerFile).createSyncAccessHandle();
// a write may come back short instead of throwing when the room runs out: that is a failure too
function put(handle: Access, bytes: Uint8Array, at: number) {
  if (handle.write(bytes, { at }) !== bytes.length) throw new Error("the origin private file system took only part of a write");
}
// a file written whole and let go of: emptied, written, flushed, closed
async function written(folder: FileSystemDirectoryHandle, file: string, write: (handle: Access) => void) {
  const handle = await access(folder, file);
  try {
    handle.truncate(0);
    write(handle);
    handle.flush();
  } finally {
    handle.close();
  }
}
const manifestBytes = (manifest: Manifest) => new TextEncoder().encode(JSON.stringify(manifest));

async function list() {
  const found: Kept[] = [];
  const directory = await folders();
  if (!directory) return found;
  for await (const [name, handle] of directory.entries()) {
    try {
      const file = await (await (handle as FileSystemDirectoryHandle).getFileHandle("manifest.json")).getFile();
      found.push({ name, where: "opfs", manifest: JSON.parse(await file.text()) });
    } catch {
      // no manifest: not finished
    }
  }
  return found;
}

async function open(name: string): Promise<Found | undefined> {
  const directory = await folders();
  const folder = await directory?.getDirectoryHandle(name).catch(() => undefined);
  if (!folder) return undefined;
  try {
    const manifest: Manifest = JSON.parse(await (await (await folder.getFileHandle("manifest.json")).getFile()).text());
    const file = await (await folder.getFileHandle("model.bin")).getFile();
    if (file.size !== manifest.bytes) return undefined;
    return {
      manifest,
      where: "opfs",
      name,
      async *parts(skip = {}) {
        for (let offset = 0; offset < file.size; offset += PART_BYTES) {
          const end = Math.min(offset + PART_BYTES, file.size), range = skip.range;
          yield range && range[0] <= offset && end <= range[1] ? new Uint8Array(end - offset) : new Uint8Array(await file.slice(offset, end).arrayBuffer());
        }
      },
      tokenizer: async () => new Uint8Array(await (await (await folder.getFileHandle("tokenizer.bin")).getFile()).arrayBuffer()),
    };
  } catch {
    return undefined;  // half there: look in the Cache API, then convert again
  }
}

async function keeping(): Promise<Keep | undefined> {
  if (!opfsWritable()) return undefined;
  const directory = await folders(true);
  return directory ? (name, manifest, slice, tokenizer, signal) => keep(directory, name, manifest, slice, tokenizer, signal) : undefined;
}
async function keep(directory: FileSystemDirectoryHandle, name: string, manifest: Manifest, slice: Slice, tokenizer: Uint8Array<ArrayBuffer>, signal?: AbortSignal) {
  try {
    const folder = await directory.getDirectoryHandle(name, { create: true });
    for (const [file, write] of [["model.bin", (handle) => {
      for (let offset = 0; offset < manifest.bytes; offset += PART_BYTES) {
        put(handle, slice(offset, Math.min(offset + PART_BYTES, manifest.bytes)), offset);
      }
    }], ["tokenizer.bin", (handle) => put(handle, tokenizer, 0)],
    ["manifest.json", (handle) => put(handle, manifestBytes(manifest), 0)]] as [string, (handle: Access) => void][]) {
      await written(folder, file, write);
      signal?.throwIfAborted();
    }
    // read back what the browser says it holds now: a write that did not stay is not a kept model
    const held = await (await folder.getFileHandle("model.bin")).getFile();
    const manifestWritten = await (await folder.getFileHandle("manifest.json")).getFile();
    if (held.size !== manifest.bytes || !manifestWritten.size) {
      throw new Error(`the origin private file system holds ${held.size} of ${manifest.bytes} bytes after writing`);
    }
    return undefined;
  } catch (error) {
    await directory.removeEntry(name, { recursive: true }).catch(() => {});
    if (signal?.aborted) throw error;
    return String((error as Error).message ?? error);
  }
}

async function forget(name: string) {
  await (await folders())?.removeEntry(name, { recursive: true }).catch(() => {});
}

export const opfs: Place = { where: "opfs", list, open, keeping, forget };

/** T156: a conversion kept as it comes, for a model whose weights go to the GPU alone (nothing holds them whole to
 * copy from afterwards, as keep() does): write(offset, bytes) as they come, in any order; finish(kept, tokenizer) writes
 * the tokenizer and the manifest last (a folder without one is rubbish, as keep()'s) and returns why nothing was kept,
 * or undefined; drop() lets go of it. On OPFS alone (a sync access handle): undefined where there is none. Whatever
 * was kept under the name before is removed first (openKept did not find it serving, or there would be no conversion) */
/** T329.2.2: holding(stretch), before the first write: a stretch of the checkpoint the engine reads from the file
 * ([start, end): a table of every layer's inputs, not in the memory). model.bin then stays open past finish(), for
 * read(); close() lets go of it. A conversion that could not be kept may still be read where the stretch was all
 * written before a write failed (whole()): the file is removed once it is let go of. dtype: what it is kept as (its
 * name says it); converter: the version its manifest says (kept.ts's CONVERTER) */
export async function keeper(name: string, dtype: string, converter: number) {
  if (!opfsWritable()) return undefined;
  const directory = await folders(true);
  if (!directory) return undefined;
  await directory.removeEntry(name, { recursive: true }).catch(() => {});
  const folder = await directory.getDirectoryHandle(name, { create: true });
  let handle: Access | null = await access(folder, "model.bin");
  let failed: string | null = null, held = 0, rubbish = false, hold: [number, number] | undefined;
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
    dtype,
    holding(stretch: [number, number]) {
      // (T521: told again where the conversion opens again, another tokenizer's try: what the try before wrote is
      // written again, and counted from nothing)
      hold = stretch;
      held = 0;
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
      const manifest = { ...kept, converter };
      try {
        if (failed) throw new Error(failed);
        handle!.flush();
        // (T329.2.2: open still where the engine reads the stretch held from it)
        if (!hold) close();
        await written(folder, "tokenizer.bin", (target) => put(target, tokenizer, 0));
        await written(folder, "manifest.json", (target) => put(target, manifestBytes(manifest), 0));
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
