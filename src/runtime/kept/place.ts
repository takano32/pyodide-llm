// kept/place.ts (T641.7): what a place a converted model is kept in is (kept.ts chooses one): the origin private file
// system (kept/opfs.ts) and the Cache API (kept/cache.ts), one shape and two of it. And the types both write and read.

import type { Options } from "../contracts/python.ts";

export type Where = "opfs" | "cache";
// (T368.1, the types)
export type Kept = { name: string; where: Where; manifest: Manifest };
// what is written beside a kept model: what the worker says of the conversion (worker/convert.ts: the model's id and
// name, where it came from, the checkpoint's bytes, the options the engine gets, when), and kept.ts's two: the
// converter's version and, in the Cache API, the number of parts
export type Conversion = { id: string; name: string; repo: string; revision: string; bytes: number; options: Options; saved: number };
export type Manifest = Conversion & { converter?: number; parts?: number };
// T329.2.2: what reads bytes of a kept checkpoint at once (a sync access handle's read()), and lets go of it
export type Reader = { read(bytes: Uint8Array, at: number): number; close(): void };
// a kept model to read: parts() yields its bytes in order (T329.2.2, skip: a stretch nothing reads from the memory,
// whose parts are not read where the place can leave them out: zeros in their place)
export type Found = {
  manifest: Manifest; where: Where; name: string;
  parts(skip?: { range?: [number, number] }): AsyncGenerator<Uint8Array<ArrayBuffer>>;
  tokenizer(): Promise<Uint8Array<ArrayBuffer>>;
};
// slice(begin, end): the checkpoint's bytes (a copy)
export type Slice = (begin: number, end: number) => Uint8Array<ArrayBuffer>;
// keeps a model under the name: why nothing was kept, or undefined. Whatever was half written is removed again
export type Keep = (name: string, manifest: Manifest, slice: Slice, tokenizer: Uint8Array<ArrayBuffer>, signal?: AbortSignal) => Promise<string | undefined>;
export type Place = {
  where: Where;
  /** Every model kept here, with its manifest (a folder without one is rubbish from a write that did not finish) */
  list(): Promise<Kept[]>;
  /** The model kept here under the name, whole, or undefined */
  open(name: string): Promise<Found | undefined>;
  /** What keeps a model here, where this browser has the place now; undefined where it has not */
  keeping(): Promise<Keep | undefined>;
  /** Delete what is kept here under the name */
  forget(name: string): Promise<void>;
};

export const PART_BYTES = 8 * 1024 * 1024;
