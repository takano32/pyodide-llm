// src/page/address.ts (T.355): what the page's address says: which model, how it is weighed (?bits=), whether to measure.
import { weightsFor } from "../models.ts";
import type { MODELS } from "../models.ts";
import { SAFETENSORS, noteOf } from "./repository.ts";

// A model that is in no list, named in the URL: ?hf=<owner>/<repository>[&revision=][&file=] is a Hugging Face model that
// the page converts (whatever the converter refuses, it refuses in words; T.505.3: &file= names the file that is the
// model, a GGUF, where it is not the repository's model.safetensors), ?checkpoint=<url>&tokenizer=<url> are
// files in llama2.c's format somewhere that allows cross-origin range requests. &template= wraps the prompt.
export const parameters = new URLSearchParams(location.search);
export const requested = parameters.get("model");
// ?bench=1 measures this model with and without the kernels, ?bench=full every step of T.52 (T.45)
export const benchmark = parameters.get("bench");
// T.98: ?bits=6 converts a model of Hugging Face to six bits a weight, ?bits=8 to int8; without it, six bits only
// where int8 would not fit (weightsFor). What the worker gets is the entry with that in its conversion.
export const weighed = (entry: any) => {
  const dtype = weightsFor(entry, parameters.get("bits"), (navigator as any).deviceMemory);
  return dtype ? { ...entry, conversion: { ...entry.conversion, dtype } } : entry;
};
export const REPOSITORY = /^[\w.-]+\/[\w.-]+$/, REVISION = /^[\w.-]+$/;
// (T.505.3) a file of a repository goes into an address too: a name, or names with / between them, none of them dots alone
// (the hyphens escaped: the sheet's field takes the same source as its pattern, which a browser reads more strictly;
// T.508.4: and FILE_MOST characters at the most, a name of an address and not a page of text)
export const FILE_MOST = 256;
export const FILE = /^(?=.{1,256}$)(?!\.+(\/|$))[\w.\-]+(\/(?!\.+(\/|$))[\w.\-]+)*$/;
// a repository nobody has looked at, as an entry of the list: the same for ?hf= and for the sheet (T.88). The
// template and prompt of a link belong to the repository the link names, not to one opened later in the sheet.
// T.374.2.3: it names no tokenizer. Which files are tried, and in which order, is the conduct's of the conversion to
// say (TOKENIZERS of src/python/convert/conduct.py)
// T.505.3: file is the file a link or the sheet names for the model (a GGUF, which needs nothing else of the repository).
// With none it is model.safetensors, until the repository is looked at (repository.ts's looked(): one without
// safetensors is opened by the best GGUF it has)
export type Unlisted = typeof MODELS[number] & { note: string; download?: number; hf: { repo: string; revision: string; weights: string; config: string } };
/** A model as one of a repository nobody listed, where it is one (and not a file of the disk, or of the list). */
export const unlisted = (entry: typeof MODELS[number] | undefined) => ((entry as Partial<Unlisted> | undefined)?.hf?.repo && entry!.id === "local" ? (entry as Unlisted) : undefined);
export function hfEntry(repository: string, revision: string, link: { template?: string; prompt?: string; file?: string } = {}): Unlisted {
  const common = { id: "local", options: {}, prompt: link.prompt ?? "", placeholder: "", template: link.template };
  const weights = link.file || SAFETENSORS;
  return { ...common, name: repository, note: noteOf(revision, weights),
    hf: { repo: repository, revision, weights, config: "config.json" },
    conversion: {}, generation: { steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty: 1.1 } } as any;
}
function named(): typeof MODELS[number] | undefined {
  const repository = parameters.get("hf"), checkpoint = parameters.get("checkpoint"), tokenizer = parameters.get("tokenizer");
  const common = { id: "local", options: {}, prompt: parameters.get("prompt") ?? "", placeholder: "", template: parameters.get("template") ?? undefined };
  // the revision goes into the file's address: only a name, as the sheet takes it (the review of T.88: a link with
  // ../ in it showed one repository and fetched another's files)
  const revision = parameters.get("revision") ?? "main";
  const file = parameters.get("file") ?? undefined;
  if (repository && REPOSITORY.test(repository) && REVISION.test(revision) && (!file || FILE.test(file))) {
    return hfEntry(repository, revision, { template: parameters.get("template") ?? undefined, prompt: parameters.get("prompt") ?? undefined, file });
  }
  if (checkpoint && tokenizer && [checkpoint, tokenizer].every((url) => /^https:\/\//.test(url))) {
    return { ...common, name: new URL(checkpoint).pathname.split("/").pop(), note: new URL(checkpoint).hostname,
      url: { checkpoint, tokenizer }, generation: { steps: 0, temperature: 0.0 }, prompt: parameters.get("prompt") ?? "Once upon a time" } as any;
  }
}
export const fromUrl = named();
