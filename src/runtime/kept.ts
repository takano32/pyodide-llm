// kept.ts (T99): the models this browser converted from Hugging Face, kept for the next visit. The worker writes
// and reads them; the page lists them and deletes them. Here: the names a conversion is kept under, which are its
// own (keptName), and which place it goes to. T641.7: the two places are one shape (kept/place.ts), two of it:
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
import { cache } from "./kept/cache.ts";
import { keeper as keeperIn, opfs, openRows } from "./kept/opfs.ts";
import type { Conversion, Kept, Manifest, Slice } from "./kept/place.ts";

export type { Conversion, Kept, Manifest, Reader } from "./kept/place.ts";
export { openRows };
// (T368.1, the types. What is read here of a model of the list: where its weights come from and how they are converted.)
export type Converted = { id?: string; shares?: string[]; conversion?: { dtype?: string; max_seq_len?: number; blocks?: boolean };
  hf: { repo: string; revision: string; weights?: string; vocabulary?: { repo: string; revision: string } } };

// the places in the order they are looked in and kept in: OPFS first, and a name kept in both is OPFS's
const PLACES = [opfs, cache];

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
/** T116: the version of what llama2_convert.py writes, the checkpoint's bytes and the options the engine gets: a
 * conversion kept by an older converter is converted again, and deleted. T628.4: Python's, the converter's own
 * (src/python/convert/version.py, where it is raised and its history is), written by python_facts.py */
export const CONVERTER = PYTHON_CONVERTER;
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
  const names = [...keptNames(model), ...[false, true].flatMap((blocks) => PYTHON_QUANTIZED.map((dtype) => keptName({ ...model, conversion: { ...model.conversion, dtype, blocks } })))];
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
