// src/page/choosing.ts (T643.10): what choosing a model decides, apart from the elements that are chosen with. It reads
// no element, no address and no storage, so Node imports it (tests/page-parts-check.ts); choose.ts asks it and draws.
import { DEFAULT_MODEL, MODELS, asksBefore, memoryWarning } from "../models.ts";
import { REPOSITORY, REVISION } from "./repository.ts";
import type { Model } from "./model.ts";
import { type Converted, type Kept, serves } from "../runtime/kept.ts";

/** the select's last entry of the Hugging Face group: it opens the sheet that asks which repository (T88) */
export const OTHER = "hf-other";
/** what the select shows again as the sheet opens: the model the page has, or the default where that was one of the
 * disk whose entry is gone */
export const shownForOther = (current: Model, localListed: boolean) => (current.id === "local" && !localListed ? DEFAULT_MODEL.id : current.id);
/** the model an entry of the select names: "local" is the one of the disk or of a repository, the others the list's */
export const chosenOf = (value: string, local: Model | undefined): Model => (value === "local" ? local! : MODELS.find(({ id }) => id === value)!);

/** T643.5: the ids of the entries (weighed as they would be loaded now: their bits) that one of the kept conversions
 * serves (under this converter). The select says so in its words, and choosing such an entry asks for no download:
 * the fact is this set, not the words. */
export const keptOf = (stored: readonly Kept[], entries: readonly (Converted & { id: string })[]): ReadonlySet<string> =>
  new Set(entries.filter((entry) => stored.some((one) => serves(one, entry))).map(({ id }) => id));

/** What is asked before a model is loaded, or nothing: a large download (what is fetched, from whom, and that the
 * conversion happens here), unless the browser kept it; T90: and one that probably does not fit in a device that says
 * how much memory it has (Chromium only). `weighed` is the model as the worker gets it (address.ts's weighed()). */
export function askedFirst(weighed: Model, kept: boolean, deviceMemory: number | undefined) {
  const download = kept ? 0 : (weighed.download ?? 0);
  const asks = [asksBefore({ download }) && `${weighed.name}: ${(download / 1e9).toFixed(1)} GB will be fetched from huggingface.co and converted in this browser.`,
                memoryWarning(weighed, deviceMemory)].filter(Boolean);
  return asks.length ? `${asks.join(" ")} Go on?` : undefined;
}

/** The sheet's three fields as a repository, its revision (main where empty) and its file (none where empty, T505.3);
 * or the field that is wrong and the words for it (T119). The address of the model's page is what people copy:
 * owner/name is what is needed of it. */
export function sheetSays(name: string, revision: string, file: string):
    { repository: string; revision: string; file?: string; wrong?: undefined } | { wrong: "name" | "revision"; words: string } {
  const repository = name.trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/(tree|blob)\/.*$/, "").replace(/\/+$/, "");
  const at = revision.trim() || "main";
  return !REPOSITORY.test(repository) ? { wrong: "name", words: "owner/name, as huggingface.co spells it" }
    : !REVISION.test(at) ? { wrong: "revision", words: "a branch, a tag or a commit" }
    : { repository, revision: at, file: file.trim() || undefined };
}
