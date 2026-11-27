// kept.ts (T99): the models this browser converted from Hugging Face, kept for the next visit. The worker writes
// and reads them; the page lists them and deletes them. Here: the names a conversion is kept under, which are its
// own (keptName), which place it goes to, and (T326) the context a load asks for and a kept one is read at. T641.7: the two places are one shape (kept/place.ts), two of it:
//
//   the origin private file system (OPFS, kept/opfs.ts), where the browser has it: a folder per model with model.bin
//   (the whole checkpoint in one file), tokenizer.bin and manifest.json, written with a sync access handle;
//   the Cache API (kept/cache.ts), as before T99 (converted-v1): parts of 8 MiB, the tokenizer and the manifest. New
//   conversions go there only where there is no OPFS; what a visitor kept there before is read and deleted as it
//   always was, never moved.
//
// The manifest is written last in both: a folder without one is rubbish from a write that did not finish.
// TypeScript of which only the types are erased (T368.1): the worker imports it, and so does the page.

import { PYTHON_CONVERTER } from "./built/facts/converter.ts";
import { PYTHON_QUANTIZED } from "./built/facts/quantized.ts";
import type { ConfigOverrides } from "./contracts/messages.ts";
import { cache } from "./kept/cache.ts";
import { keeper as keeperIn, opfs, opfsOpens, openRows } from "./kept/opfs.ts";
import type { Conversion, Kept, Manifest, Slice } from "./kept/place.ts";

export type { Conversion, Kept, Manifest, Reader } from "./kept/place.ts";
export { openRows, opfsOpens };  // (T525: opfsOpens() for the page, src/models/memory.ts's modelBytes())
// (T368.1, the types. What is read here of a model of the list: where its weights come from and how they are converted.)
export type Converted = { id?: string; shares?: string[]; conversion?: { dtype?: string; max_seq_len?: number; blocks?: boolean };
  hf: { repo: string; revision: string; weights?: string; vocabulary?: { repo: string; revision: string; overrides?: ConfigOverrides } } };

// the places in the order they are looked in and kept in: OPFS first, and a name kept in both is OPFS's
const PLACES = [opfs, cache];

/** T326: the context a load asks of a model: its entry's (conversion.max_seq_len), or the page's most, 4096. The one
 * place it is chosen: a conversion is made at it (worker/convert.ts) and a kept one is read at it (contextRead()). */
export const contextOf = (model: Pick<Converted, "conversion">) => model.conversion?.max_seq_len ?? 4096;
// T326: the context every name said before (no entry asked another, and the page asked none), which still names what
// was kept then
const BEFORE = 4096;
// the dtypes of a file that holds no table as long as the context (Python's QUANTIZED: the engine makes the RoPE tables)
const quantized = (dtype: string) => (PYTHON_QUANTIZED as readonly string[]).includes(dtype);

// the name of a model's conversion: its repository, revision and dtype, as the Cache API key has it, and where the
// vocabulary and config.json come from another repository than the weights (T136), that one too.
// T363.5: a conversion that kept a GGUF's blocks as they are (conversion.blocks) is another one than the same file
// widened, though both say the dtype int8 (how the rows that are no blocks are stored): ":blocks" after its name. Every
// other name is what it was.
// T505.3: a repository nobody listed (?hf=, the sheet: the id "local") may be opened by one file of it or by another
// (&file=: its Q8_0.gguf, its Q4_K_M.gguf), and each is another conversion: the file's name after the revision, unless
// it is the model.safetensors every such repository was opened by before. A model of the list is one file, and its
// name is what it was.
// T684: what an entry sets over its vocabulary's config.json (vocabulary.overrides) is another conversion too, as the
// converter makes it: the keys and values after the vocabulary, in the order of the keys ({eos_token_id=2,...}), and
// nothing where it sets none (every name of before)
// T326: and its context only where the file holds tables as long as it (a float32 or float16 file's RoPE tables): a
// quantized file is the same at every context but for its header's seq_len, which is set as it is read
// (contextRead()). What was kept before under a name of its context (BEFORE) is read under that name (keptNames()),
// not moved: OPFS has no folder to rename, and the Cache API would copy every part. context: the one to name (BEFORE's)
export function keptName(model: Converted, context = quantized(model.conversion?.dtype ?? "int8") ? undefined : contextOf(model)) {
  const { dtype = "int8", blocks = false } = model.conversion ?? {};
  const { vocabulary, weights } = model.hf;
  const overrides = Object.entries(vocabulary?.overrides ?? {}).sort(([a], [b]) => (a < b ? -1 : 1));
  const set = overrides.length ? `{${overrides.map(([key, value]) => `${key}=${JSON.stringify(value)}`).join(",")}}` : "";
  const from = vocabulary ? `+${vocabulary.repo}@${vocabulary.revision}${set}` : "";
  const file = model.id === "local" && weights && weights !== "model.safetensors" ? `/${weights}` : "";
  return encodeURIComponent(`${model.hf.repo}@${model.hf.revision}${file}${from}:${dtype}${context === undefined ? "" : `:${context}`}${blocks ? ":blocks" : ""}`);
}
// T326: a conversion's two names, its own and the one it was kept under before (BEFORE said): one where they are the same
const named = (model: Converted) => [...new Set([keptName(model), keptName(model, BEFORE)])];
/** T326: the context a kept conversion is read at: the one asked for (contextOf()), no longer than the file may be read
 * at (manifest.longest, the converter's Conversion.longest: the model's own, or the header's own where the file holds
 * tables as long as it). A conversion kept before T326 says none: its header's own, unless that is BEFORE in a
 * quantized file, which the converter may have cut there (the context asked for is then taken for one the model has:
 * no entry asks more than its model's own). first: the kept file's first bytes, whose header's seq_len (its seventh
 * int, bytes 24 to 28) is set to that context here, before anything reads the header. Returns the context. */
export function contextRead(first: Uint8Array, manifest: Manifest, model: Pick<Converted, "conversion">) {
  // (an int32, little-endian, as Int32Array reads the header in every browser)
  const said = first[24] | (first[25] << 8) | (first[26] << 16) | (first[27] << 24);
  const cut = said === BEFORE && quantized(manifest.options.dtype ?? "int8");
  const context = Math.min(contextOf(model), manifest.longest ?? (cut ? Infinity : said));
  first.set([context & 255, (context >> 8) & 255, (context >> 16) & 255, context >>> 24], 24);
  return context;
}
/** T116: the version of what llama2_convert.py writes, the checkpoint's bytes and the options the engine gets: a
 * conversion kept by an older converter is converted again, and deleted. T628.4: Python's, the converter's own
 * (src/python/convert/version.py, where it is raised and its history is), written by python_facts.py */
export const CONVERTER = PYTHON_CONVERTER;
const converterOf = (manifest: Manifest) => manifest.converter ?? 1;
/** The names a model's conversion may be kept under: its bits, or with none asked for, either of the two the worker
 * may choose (T115) */
export function keptNames(model: Converted) {
  const asked = model.conversion?.dtype;
  // (T326: each under its own name first, then under the one it was kept under before)
  return (asked ? [asked] : ["int8", "int6"]).flatMap((dtype) => named({ ...model, conversion: { ...model.conversion, dtype } }));
}
/** Whether a kept conversion (one of keptModels()) serves a model: kept under one of its names, by this converter */
export const serves = (kept: Kept, model: Converted) => keptNames(model).includes(kept.name) && converterOf(kept.manifest) === CONVERTER;
/** Whether a kept conversion is of an older converter: not used, and deleted. (One of a newer converter is left
 * alone: a tab of the older page must not delete what the newer one kept.) */
export const outdated = (kept: Pick<Kept, "manifest">) => converterOf(kept.manifest) < CONVERTER;
/** The conversions kept for a model of the list under a name it has no more, of either bits (T136: its weights come
 * from another repository now, or its vocabulary): never served again, and in the way of the new one (a 7B's 8 GB
 * each). The worker deletes them before it keeps the new one. Not those of ?hf= and of folders, which all have the
 * id "local". shares: the ids of the list that share one conversion (a Qwen3's two forms, T124): one name, which the
 * source makes (keptName() reads no id), so either form reads what the other kept (T649.5); its manifest says the id of
 * whichever converted it, and so these are both ids' (the review of T203: the other form's left its 9 GB behind). */
export async function replaced(model: Converted) {
  if (model.id === "local") return [];
  // its own conversions of every bits stay, whichever one is asked for: the ternary one too (the review of T230: a
  // ?bits=8 visit of a ternary model deleted its ternary conversion, and the next plain visit fetched and converted it
  // again, for each visit in turn)
  // (T363.5: and both of a model whose blocks are kept: with its blocks, and widened, which a load without the int8
  // kernels asks for: ?without=kernels, a browser without WebAssembly SIMD)
  const names = [...keptNames(model), ...[false, true].flatMap((blocks) => PYTHON_QUANTIZED.flatMap((dtype) => named({ ...model, conversion: { ...model.conversion, dtype, blocks } })))];
  const ids = model.shares ?? [model.id];
  return (await keptModels()).filter((kept) => ids.includes(kept.manifest.id) && !names.includes(kept.name));
}

/** Every kept model: { name, where: "opfs" | "cache", manifest }. */
export async function keptModels() {
  const found: Kept[] = [];
  for (const place of PLACES) {
    for (const kept of await place.list()) if (!found.some((one) => one.name === kept.name)) found.push(kept);
  }
  return found;
}

/** A kept model to read: { manifest, where, name, parts(), tokenizer() }, or undefined. parts() yields its bytes in
 * order. The first of its names (keptNames) kept by this converter; what an older one kept is deleted on the way (T116). */
export async function openKept(model: Converted) {
  for (const name of keptNames(model)) {
    const found = await openNamed(name);
    if (found && converterOf(found.manifest) === CONVERTER) return found;
    if (found && outdated(found)) await forget({ name, where: found.where });
  }
  return undefined;
}
// the first place that holds the name whole (one half there in OPFS, cut off: the Cache API is looked in)
async function openNamed(name: string) {
  for (const place of PLACES) {
    const found = await place.open(name);
    if (found) return found;
  }
  return undefined;
}

/** Keep a conversion: slice(begin, end) gives the checkpoint's bytes (a copy), tokenizer is a Uint8Array.
 * Returns why nothing was kept, or undefined. Whatever was half written is removed again. Nothing asks the browser
 * beforehand how much room there is: its estimate said yes where the write then failed (T60). */
export async function keep(model: Converted, kept: Conversion, slice: Slice, tokenizer: Uint8Array<ArrayBuffer>, signal?: AbortSignal) {
  const manifest: Manifest = { ...kept, converter: CONVERTER };
  for (const place of PLACES) {
    const keeping = await place.keeping();
    if (keeping) return keeping(keptName(model), manifest, slice, tokenizer, signal);
  }
  return "this browser has neither the origin private file system nor the Cache API here";
}

/** T156: a conversion kept as it comes, in the origin private file system alone (kept/opfs.ts's keeper(), which says
 * the rest): undefined where there is none */
export const keeper = (model: Converted) => keeperIn(keptName(model), model.conversion?.dtype ?? "int8", CONVERTER);

/** Delete a kept model (one of keptModels()). */
export async function forget({ name, where }: Pick<Kept, "name" | "where">) {
  await PLACES.find((place) => place.where === where)!.forget(name);
}
