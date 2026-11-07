// src/page/parameters.ts (T643.9): the names of the model page's address, in one table, and the addresses the page
// writes with them. Nothing of a page is read here (no element, no address of its own: the caller hands it over), so
// Node imports it (tests/page-parts-check.ts). The names are what visitors' bookmarks, shared links and the README
// hold: none of them is renamed.
//
// The page reads its address through address.ts's `parameters`, which takes a name of this table and no other (tsc says
// a name that is not here); what it writes back is written here. The worker reads the address it is sent (search, of
// the load) with its own names (src/runtime/worker/runtime.ts, request.ts, pyodide.ts: ?gpuTest=, ?without=, ?pyodide=
// and the rest), which are the worker's and not this table's; /benchmark/ has its own (src/benchmark/).
import type { Sampled } from "../runtime/contracts/messages.ts";
import type { Model } from "./model.ts";
import { unlisted, unnamed } from "./repository.ts";

// What each name is for. "model": it names the model the page has (address.ts's named() reads them, addressOf()
// writes them, and only them, for the model chosen). "engine": how the model runs (settings.ts's switches and bits,
// which a reload keeps; remembered.ts's threads). "visit": what this visit does (measure once, the Service Worker of
// isolation.ts, which receiver.ts and remembered.ts hand ?coi= and ?offline=, the tests' GPU of draw.ts). The
// generation settings are names too: the keys of Sampled, which fields.ts's LIMITS reads (one row each there).
export const NAMES = {
  model: "model", hf: "model", revision: "model", file: "model", checkpoint: "model", tokenizer: "model", template: "model", prompt: "model",
  without: "engine", kernel: "engine", bits: "engine", threads: "engine",
  bench: "visit", coi: "visit", offline: "visit", gpuTest: "visit",
} as const;
/** a name the model page reads of its address */
export type Name = keyof typeof NAMES | keyof Sampled;
/** what the page reads of an address: URLSearchParams is one, held to the names of the table */
export type Address = { get(name: Name): string | null };
/** the names that name a model, in the table's order */
export const MODEL_NAMES = (Object.keys(NAMES) as (keyof typeof NAMES)[]).filter((name) => NAMES[name] === "model");

/** The address with the names set to their values, or taken away (null), in the order they are given. */
export function addressWith(href: string, changes: { [name in Name]?: string | null }): string {
  const url = new URL(href);
  for (const [name, value] of Object.entries(changes)) {
    value === null || value === undefined ? url.searchParams.delete(name) : url.searchParams.set(name, value);
  }
  return url.href;
}

// The URL names the model the page has, however it was chosen (the list, the sheet, the folder), so that a reload
// (every switch of the panel reloads) or a shared link opens that model and no other (the review of T75 and T88:
// after ?hf=, choosing a model of the list and then a switch reloaded the Hugging Face model). A file of the disk
// has no address: its URL names nothing, and a reload opens the model remembered before it.
/** The address `href` with the names of a model (MODEL_NAMES) saying the one the page has now, and the rest as they were. */
export function addressOf(current: Model, href: string): string {
  const said: { [name in typeof MODEL_NAMES[number]]?: string } = {};
  const { hf, url: address, template } = current;
  const named = unlisted(current);
  if (current.id !== "local") {
    said.model = current.id;
  } else if (named || address) {
    if (named) {
      said.hf = named.hf.repo;
      if (named.hf.revision !== "main") {
        said.revision = named.hf.revision;
      }
      // T505.3: and the file, where it is not the model.safetensors a repository is opened by: the one the visitor
      // named, or the GGUF that was the best there when the repository was looked at
      if (!unnamed(named)) {
        said.file = named.hf.weights;
      }
    } else if (address) {
      said.checkpoint = address.checkpoint;
      said.tokenizer = address.tokenizer;
    }
    for (const [name, value] of [["template", template], ["prompt", current.prompt]] as const) {
      if (value) {
        said[name] = value;
      }
    }
  }
  // (every name of a model taken away first, then the ones said set: the order a link was always written in)
  const cleared = addressWith(href, Object.fromEntries(MODEL_NAMES.map((name) => [name, null])));
  return addressWith(cleared, said);
}
