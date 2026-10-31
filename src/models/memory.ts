// What a model takes once loaded, the dtype it is converted to, and what the page says of a device's memory.
import { FACTS } from "../runtime/built/facts.ts";
import type { Entry } from "./builders.ts";

// T.90: memory. A device that runs out of it kills the worker's WebAssembly memory, so the page warns before it
// loads a model that probably does not fit, and says what happened when it did not.
/** What the page takes besides the model: Pyodide, NumPy and the engine (the margin T.90 asked for; the heap of
 * llm-jp-3 150M measures 112 MB above its 171 MB of weights, and the tab needs its own). */
export const PAGE_MEMORY = 300e6;
const megabytes = (bytes: number) => `${Math.round(bytes / 1e6).toLocaleString("en")} MB`;
/** The bytes of a model once loaded: `bytes` of a file of this site, or `loaded` of a conversion that says its size
 * itself (T.364.3: one kept as its file has it, llama.cpp's blocks, whose note is worded for a visitor and whose size no
 * ?bits= changes), or the "int8 N MB" its note gives for a conversion ("ternary N MB" for a ternary model, T.230: four
 * times that as int8, where ?bits= asks for it). undefined when none says (a file of the visitor's). */
export function modelBytes(entry: Entry): number | undefined {
  // a float16 original is widened to float32 when loaded, next to the file it came from: llm-jp-3 150M's 305 MB
  // file measures about 800 MB of heap (AGENTS.md), so three times the file is the honest estimate
  if (entry.bytes) return entry.options?.dtype === "float16" ? entry.bytes * 3 : entry.bytes;
  if (entry.loaded) return entry.loaded;
  const found = /(int8|ternary) ([\d.]+) (MB|GB)/.exec(entry.note ?? "");
  if (!found) return undefined;
  const said = Number(found[2]) * (found[3] === "GB" ? 1e9 : 1e6), asked = entry.conversion?.dtype;
  if (found[1] === "ternary" && !["int8", "int6"].includes(asked!)) return said;
  const int8 = found[1] === "ternary" ? said / TERNARY_OF_EIGHT : said;
  return asked === "int6" ? int8 * SIX_OF_EIGHT : int8;
}

// T.98: a model converted in the page can keep its weights in six bits instead of eight: 24 bytes and a scale per
// group of 32 against 32 and a scale, 7/9 of the size, at +1 to +3.4% of perplexity (measured on eight models), and
// slower on one thread (the groups are widened as they are read). So it is taken where int8 does not fit.
// (T.628.3: of Python's bytes of each, written by python_facts.py)
export const SIX_OF_EIGHT = FACTS.bytes.int6 / FACTS.bytes.int8;
// T.230: a ternary model keeps its weights as they are, two bits each: 32 bytes and a scale per group of 128 (T.533: a
// float16, 2 bytes), against int8's 128 bytes and four scales, with no loss at all (int8 is the same weights widened)
export const TERNARY_OF_EIGHT = FACTS.bytes.ternary / FACTS.bytes.int8;

// T.133: Chromium's navigator.deviceMemory stops at 8: a device that says 8 has 8 GB or more, as many as it likes
export const DEVICE_MEMORY_CAP = 8;
/** The dtype a model of Hugging Face is converted to: the entry's own when it has one (the settings of a visitor's
 * files); else asked is ?bits= (or a setting), "8", "6" or anything else for
 * automatic, which is the entry's own `weights` where it names them (T.230: "ternary", a ternary model's weights as
 * they are, smaller than six bits of them and exact), and else takes int6 where int8 would pass half of what the device says it has (deviceMemory, Chromium
 * only, and below its cap of 8: a device at the cap may have any more), and otherwise leaves the choice to the
 * worker (undefined): it knows the model's header once it converts, and with it what the forward pass needs, and
 * takes int6 where int8 would not fit a 32-bit memory and the browser has no 64-bit one (T.115, T.133).
 * undefined for a model that is not converted in the page. */
export function weightsFor(entry: Entry, asked: string | null | undefined, deviceMemory: number | undefined) {
  if (!entry.hf) return undefined;
  // a visitor's own files may come with settings that say it ({"conversion": {"dtype": ...}}): they win (T.119)
  if (entry.conversion?.dtype) return entry.conversion.dtype;
  if (asked === "6" || asked === "8") return `int${asked}`;
  if (entry.weights) return entry.weights;
  if (!deviceMemory || deviceMemory >= DEVICE_MEMORY_CAP) return undefined;
  const int8 = modelBytes({ ...entry, conversion: { ...entry.conversion, dtype: "int8" } });
  return int8 && int8 + PAGE_MEMORY > deviceMemory * 2 ** 30 / 2 ? "int6" : undefined;
}
/** A sentence for a device that says it has less memory than twice what the model needs, or "". deviceMemory is
 * navigator.deviceMemory (GB; only Chromium tells, and at most 8): without it nothing is guessed. A device at the cap
 * has 8 GB or more, so it is warned only of a model that needs more than 8 GB (T.133). */
export function memoryWarning(entry: Entry, deviceMemory: number | undefined): string {
  const bytes = modelBytes(entry);
  if (!deviceMemory || !bytes) return "";
  const capped = deviceMemory >= DEVICE_MEMORY_CAP;
  if (bytes + PAGE_MEMORY <= deviceMemory * 2 ** 30 / (capped ? 1 : 2)) return "";
  return `${entry.name} needs about ${megabytes(bytes + PAGE_MEMORY)} of memory, and this device has ` +
    `${capped ? `${DEVICE_MEMORY_CAP} GB or more` : `${deviceMemory} GB`}: it may run out of memory.`;
}
/** T.511.4: what the note of an entry of the list says of the devices it is for, by the page's own rules and not by
 * hand (the owner's choice of 2026-10-15): "desktop only" where a device that says 4 GB (a phone's) would be warned
 * by memoryWarning() (loaded, the model and PAGE_MEMORY pass 2 GiB: about 1.85 GB), and "Chrome and Firefox" where
 * what it loads passes 4 GiB, which only a 64-bit memory holds (Safari has none and refuses it). Safari would take
 * six bits there (weightsFor()), and six bits with what the forward pass keeps after them (the keys and values, the
 * corrections of relaxed SIMD) pass 4 GiB about where eight bits alone do; the list has no header to count it
 * exactly (footprint()). The words in the order the notes have them, after everything else. */
export function devicesOf(entry: Entry) {
  const bytes = modelBytes(entry);
  return [memoryWarning(entry, 4) && "desktop only", bytes! > 2 ** 32 && "Chrome and Firefox"].filter(Boolean);
}
/** What the page says when the worker ran out of memory (heap: the size of its WebAssembly memory then). */
export function memoryFailure(entry: Entry, heap: number | undefined, detail: string | undefined): string {
  const bytes = modelBytes(entry);
  return `This device ran out of memory for ${entry.name}` +
    (bytes ? ` (it needs about ${megabytes(bytes + PAGE_MEMORY)})` : "") +
    (heap ? `; the page was using ${megabytes(heap)} when it happened` : "") +
    `. A smaller model may fit, or closing other tabs may help.` + (detail ? ` (${detail})` : "");
}
