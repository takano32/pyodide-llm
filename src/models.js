
//
// This file is the window (T354): the list is put together here, in the order it had as one file, from its parts
// in src/models/, and everything the page, the worker and the tests take from the list is exported from here.
import { LICENSES } from "./models/licenses.js";
import { modelBytes } from "./models/memory.js";
import { BUILT } from "./models/built.js";
import { HF_JAPANESE } from "./models/hf-japanese.js";
import { HF_CLASSICS } from "./models/hf-classics.js";
import { HF_FAMILIES } from "./models/hf-families.js";

export { filled } from "./models/filled.js";
export { LICENSES } from "./models/licenses.js";
export { PAGE_MEMORY, modelBytes, SIX_OF_EIGHT, TERNARY_OF_EIGHT, DEVICE_MEMORY_CAP, weightsFor, memoryWarning, memoryFailure } from "./models/memory.js";

/** The Hugging Face repository a model comes from. */
export const sourceOf = (entry) => entry.hf?.repo ?? entry.source;
/** Every source once, in the order of the list, with its license and the names of the models taken from it. A
 * model fetched from a redistribution names both: where it comes from, and whose model it is. The redistribution's
 * line says which it is: "(GGUF)" for a GGUF (T74), "(copy)" for the same safetensors elsewhere (unsloth's Llama;
 * until 2026-09-26 it said "(GGUF)" for those too). A GGUF with a copy's vocabulary (T136: Llama 3.2's) is on all three. */
export function sources(models = MODELS) {
  const bySource = new Map();
  for (const entry of models) {
    // T136: the repository the vocabulary and config.json come from, where it is neither (unsloth's copy of Llama)
    for (const repo of new Set([entry.original, entry.hf?.vocabulary?.repo, sourceOf(entry)].filter(Boolean))) {
      if (!bySource.has(repo)) bySource.set(repo, { repo, license: LICENSES[repo], names: [] });
      const kind = repo !== sourceOf(entry) ? "copy" : entry.hf?.weights?.endsWith(".gguf") ? "GGUF" : "copy";
      bySource.get(repo).names.push(entry.original && repo !== entry.original ? `${entry.name} (${kind})` : entry.name);
    }
  }
  return [...bySource.values()];
}

// group: "site" (built with the site, the default), "original" or "hf"
export const GROUPS = { site: "Models of this site", original: "Unquantized originals", hf: "From Hugging Face, converted in this browser" };

// in the order they were added, more or less; MODELS below is the order of the list
const LISTED = [...BUILT, ...HF_JAPANESE, ...HF_CLASSICS, ...HF_FAMILIES];

/** T128: whether a model writes Japanese (its note says 日本語: Japanese alone, with English, or translating). */
export const writesJapanese = (entry) => (entry.note ?? "").includes("日本語");
/** The models as the list shows them (T128, the owner's order): the groups in the order of GROUPS, and within each
 * the ones that write Japanese from light to heavy, then the English-only ones from light to heavy, by modelBytes()
 * (int8 for a conversion, three times the file for a float16 original). The default, the first, is tiny-lm. */
export const MODELS = LISTED.map((entry) => ({ entry, key: [Object.keys(GROUPS).indexOf(entry.group ?? "site"), writesJapanese(entry) ? 0 : 1, modelBytes(entry)] }))
  .sort((a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1] || a.key[2] - b.key[2])
  .map(({ entry }) => entry);
