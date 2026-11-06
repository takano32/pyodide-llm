// The list put together (T643.21, from the window src/models.ts): the parts' entries with what every note says of
// the devices and every thinking entry's budget, in the list's order, the default model, and the sources the About
// names. src/models.ts only exports these names.
import { LICENSES, NOTICES } from "./licenses.ts";
import { modelBytes, devicesOf } from "./memory.ts";
import { BUILT } from "./built.ts";
import { THOUGHT_BUDGET, thinks } from "./sampling.ts";
import { HF_JAPANESE } from "./hf-japanese.ts";
import { HF_JAPANESE_LARGE } from "./hf-japanese-large.ts";
import { HF_CLASSICS } from "./hf-classics.ts";
import { HF_FAMILIES } from "./hf-families.ts";
import { TRANSLATIONS } from "./translations.ts";
import type { Entry } from "./builders.ts";

// group: "site" (built with the site, the default), "original" or "hf"
export const GROUPS = { site: "Models of this site", original: "Unquantized originals", hf: "From Hugging Face, converted in this browser" };

// T511.4: "desktop only" and "Chrome and Firefox" are not written in the parts: devicesOf() adds them to every note.
// T520: nor the budget of a thought: every entry that thinks has the same one (THOUGHT_BUDGET)
const LISTED: Entry[] = [...BUILT, ...TRANSLATIONS, ...HF_JAPANESE, ...HF_JAPANESE_LARGE, ...HF_CLASSICS, ...HF_FAMILIES]
  .map((entry) => ({ ...entry, note: [entry.note, ...devicesOf(entry)].join(" · "),
    ...(thinks(entry) ? { generation: { ...entry.generation, thinking: THOUGHT_BUDGET } } : {}) }));

/** T128: whether a model writes Japanese (T643.17: its own word, `japanese`; its note says 日本語 where it does, Japanese
 * alone, with English, or translating, which tests/models-check.ts holds to it). */
export const writesJapanese = (entry: Pick<Entry, "japanese">) => entry.japanese;

// the key an entry is sorted by: its group in the order of GROUPS, Japanese before English only, then light to heavy
const orderOf = (entry: Entry) => [Object.keys(GROUPS).indexOf(entry.group ?? "site"), writesJapanese(entry) ? 0 : 1, modelBytes(entry)!];
/** The models as the list shows them (T128, the owner's order): the groups in the order of GROUPS, and within each
 * the ones that write Japanese from light to heavy, then the English-only ones from light to heavy, by modelBytes()
 * (int8 for a conversion, three times the file for a float16 original), so a model added to any part takes its
 * place. The first is tiny-lm, the lightest. The default is DEFAULT_MODEL below (T511.6), not the first. */
export const MODELS = LISTED.map((entry) => ({ entry, key: orderOf(entry) }))
  .sort((a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1] || a.key[2] - b.key[2])
  .map(({ entry }) => entry);

/** T511.6 (the owner, 2026-10-15): the model a first visit opens, and the one the page falls back on. Shisa V2.1c LFM2
 * 350M answers questions (18 of T511.5's 21 facts, tiny-lm none), for a first visit's 379 MB fetched from Hugging Face
 * and converted in the browser (kept for the next visit). It stays where the list's order puts it. Under UNASKED_BYTES, so
 * it is started unasked (tests/models-check.ts holds both). */
export const DEFAULT_MODEL = listedAs("hf-shisa-v2.1c-lfm2-350m");
function listedAs(id: string): Entry {
  const entry = MODELS.find((entry) => entry.id === id);
  if (!entry) throw new Error(`The default model ${id} is not in the list.`);
  return entry;
}

// a source of sources(): a repository, its license (and the notice it asks for), the names of the models taken from it
type Source = { repo: string; license: string; notice?: string; names: string[] };
/** The Hugging Face repository a model comes from. */
export const sourceOf = (entry: Entry) => entry.hf?.repo ?? entry.source;
/** Every source once, in the order of the list, with its license and the names of the models taken from it. A
 * model fetched from a redistribution names both: where it comes from, and whose model it is. The redistribution's
 * line says which it is: "(GGUF)" for a GGUF (T74), "(copy)" for the same safetensors elsewhere (unsloth's Llama;
 * until 2026-09-26 it said "(GGUF)" for those too). A GGUF with a copy's vocabulary (T136: Llama 3.2's) is on all three. */
export function sources(models = MODELS): Source[] {
  const bySource = new Map<string, Source>();
  for (const entry of models) {
    const own = sourceOf(entry);
    // T136: the repository the vocabulary and config.json come from, where it is neither (unsloth's copy of Llama)
    for (const repo of new Set([entry.original, entry.hf?.vocabulary?.repo, own].filter((repo): repo is string => Boolean(repo)))) {
      if (!bySource.has(repo)) bySource.set(repo, { repo, license: LICENSES[repo], notice: NOTICES[repo], names: [] });
      const kind = repo === own && entry.hf?.weights?.endsWith(".gguf") ? "GGUF" : "copy";
      bySource.get(repo)!.names.push(entry.original && repo !== entry.original ? `${entry.name} (${kind})` : entry.name);
    }
  }
  return [...bySource.values()];
}
