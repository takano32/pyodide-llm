
//
// This file is the window (T.354): the list is put together here, in the order it had as one file, from its parts
// in src/models/, and everything the page, the worker and the tests take from the list is exported from here.
import { LICENSES, NOTICES } from "./models/licenses.ts";
import { modelBytes, devicesOf } from "./models/memory.ts";
import { BUILT } from "./models/built.ts";
import { THOUGHT_BUDGET, thinks } from "./models/sampling.ts";
import { HF_JAPANESE } from "./models/hf-japanese.ts";
import { HF_CLASSICS } from "./models/hf-classics.ts";
import { HF_FAMILIES } from "./models/hf-families.ts";
import type { Entry } from "./models/builders.ts";

export { filled } from "./models/filled.ts";
export type { Entry } from "./models/builders.ts";
export { LICENSES, NOTICES } from "./models/licenses.ts";
export { PAGE_MEMORY, modelBytes, SIX_OF_EIGHT, TERNARY_OF_EIGHT, DEVICE_MEMORY_CAP, weightsFor, memoryWarning, memoryFailure } from "./models/memory.ts";

// a source of sources(): a repository, its license (and the notice it asks for), the names of the models taken from it
type Source = { repo: string; license: string; notice?: string; names: string[] };
/** The Hugging Face repository a model comes from. */
export const sourceOf = (entry: Entry) => entry.hf?.repo ?? entry.source;
/** Every source once, in the order of the list, with its license and the names of the models taken from it. A
 * model fetched from a redistribution names both: where it comes from, and whose model it is. The redistribution's
 * line says which it is: "(GGUF)" for a GGUF (T.74), "(copy)" for the same safetensors elsewhere (unsloth's Llama;
 * until 2026-09-26 it said "(GGUF)" for those too). A GGUF with a copy's vocabulary (T.136: Llama 3.2's) is on all three. */
export function sources(models = MODELS): Source[] {
  const bySource = new Map();
  for (const entry of models) {
    // T.136: the repository the vocabulary and config.json come from, where it is neither (unsloth's copy of Llama)
    for (const repo of new Set([entry.original, entry.hf?.vocabulary?.repo, sourceOf(entry)].filter(Boolean))) {
      if (!bySource.has(repo)) bySource.set(repo, { repo, license: LICENSES[repo!], notice: NOTICES[repo!], names: [] });
      const kind = repo !== sourceOf(entry) ? "copy" : entry.hf?.weights?.endsWith(".gguf") ? "GGUF" : "copy";
      bySource.get(repo).names.push(entry.original && repo !== entry.original ? `${entry.name} (${kind})` : entry.name);
    }
  }
  return [...bySource.values()];
}

// group: "site" (built with the site, the default), "original" or "hf"
export const GROUPS = { site: "Models of this site", original: "Unquantized originals", hf: "From Hugging Face, converted in this browser" };

// The default is DEFAULT_MODEL below (T.511.6), not the first of the list; ?model=<id> picks another one.
// Within each group the order is the ones that write Japanese from light to heavy, then the English-only ones from
// light to heavy (T.128): MODELS is sorted so below, so a model added to any part takes its place.
// in the order they were added, more or less; MODELS below is the order of the list
// T.511.4: "desktop only" and "Chrome and Firefox" are not written in the parts: devicesOf() adds them to every note.
// T.520: nor the budget of a thought: every entry that thinks has the same one (THOUGHT_BUDGET)
const LISTED: Entry[] = [...BUILT, ...HF_JAPANESE, ...HF_CLASSICS, ...HF_FAMILIES]
  .map((entry) => ({ ...entry, note: [entry.note, ...devicesOf(entry)].join(" · "),
    ...(thinks(entry) ? { generation: { ...entry.generation, thinking: THOUGHT_BUDGET } } : {}) }));

/** T.128: whether a model writes Japanese (its note says 日本語: Japanese alone, with English, or translating). */
export const writesJapanese = (entry: Entry) => (entry.note ?? "").includes("日本語");
/** The models as the list shows them (T.128, the owner's order): the groups in the order of GROUPS, and within each
 * the ones that write Japanese from light to heavy, then the English-only ones from light to heavy, by modelBytes()
 * (int8 for a conversion, three times the file for a float16 original). The first is tiny-lm, the lightest. */
export const MODELS = LISTED.map((entry) => ({ entry, key: [Object.keys(GROUPS).indexOf(entry.group ?? "site"), writesJapanese(entry) ? 0 : 1, modelBytes(entry)!] }))
  .sort((a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1] || a.key[2] - b.key[2])
  .map(({ entry }) => entry);

/** T.511.6 (the owner, 2026-10-15): the model a first visit opens, and the one the page falls back on. Shisa V2.1c LFM2
 * 350M answers questions (18 of T.511.5's 21 facts, tiny-lm none), for a first visit's 379 MB fetched from Hugging Face
 * and converted in the browser (kept for the next visit). It stays where the list's order puts it. Under 500 MB, so it
 * is started unasked (tests/models-check.mjs holds both). */
export const DEFAULT_MODEL = MODELS[MODELS.findIndex(({ id }) => id === "hf-shisa-v2.1c-lfm2-350m")];
