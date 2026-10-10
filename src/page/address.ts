// src/page/address.ts (T355): what the page's address says: which model, how it is weighed (?bits=), whether to measure.
import { weightsFor } from "../models.js";

// A model that is in no list, named in the URL: ?hf=<owner>/<repository>[&revision=] is a Hugging Face model that
// the page converts (whatever the converter refuses, it refuses in words), ?checkpoint=<url>&tokenizer=<url> are
// files in llama2.c's format somewhere that allows cross-origin range requests. &template= wraps the prompt.
export const parameters = new URLSearchParams(location.search);
export const requested = parameters.get("model");
// ?bench=1 measures this model with and without the kernels, ?bench=full every step of T52 (T45)
export const benchmark = parameters.get("bench");
// T98: ?bits=6 converts a model of Hugging Face to six bits a weight, ?bits=8 to int8; without it, six bits only
// where int8 would not fit (weightsFor). What the worker gets is the entry with that in its conversion.
export const weighed = (entry: any) => {
  const dtype = weightsFor(entry, parameters.get("bits"), (navigator as any).deviceMemory);
  return dtype ? { ...entry, conversion: { ...entry.conversion, dtype } } : entry;
};
export const REPOSITORY = /^[\w.-]+\/[\w.-]+$/, REVISION = /^[\w.-]+$/;
// a repository nobody has looked at, as an entry of the list: the same for ?hf= and for the sheet (T88). The
// template and prompt of a link belong to the repository the link names, not to one opened later in the sheet.
// T374.2.3: it names no tokenizer. Which files are tried, and in which order, is the conduct's of the conversion to
// say (TOKENIZERS of public/convert/conduct.py)
export function hfEntry(repository: string, revision: string, link: { template?: string; prompt?: string } = {}): typeof MODELS[number] {
  const common = { id: "local", options: {}, prompt: link.prompt ?? "", placeholder: "", template: link.template };
  return { ...common, name: repository, note: revision === "main" ? "Hugging Face" : `Hugging Face · ${revision.slice(0, 12)}`,
    hf: { repo: repository, revision, weights: "model.safetensors", config: "config.json" },
    conversion: {}, generation: { steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty: 1.1 } } as any;
}
function named(): typeof MODELS[number] | undefined {
  const repository = parameters.get("hf"), checkpoint = parameters.get("checkpoint"), tokenizer = parameters.get("tokenizer");
  const common = { id: "local", options: {}, prompt: parameters.get("prompt") ?? "", placeholder: "", template: parameters.get("template") ?? undefined };
  // the revision goes into the file's address: only a name, as the sheet takes it (the review of T88: a link with
  // ../ in it showed one repository and fetched another's files)
  const revision = parameters.get("revision") ?? "main";
  if (repository && REPOSITORY.test(repository) && REVISION.test(revision)) {
    return hfEntry(repository, revision, { template: parameters.get("template") ?? undefined, prompt: parameters.get("prompt") ?? undefined });
  }
  if (checkpoint && tokenizer && [checkpoint, tokenizer].every((url) => /^https:\/\//.test(url))) {
    return { ...common, name: new URL(checkpoint).pathname.split("/").pop(), note: new URL(checkpoint).hostname,
      url: { checkpoint, tokenizer }, generation: { steps: 0, temperature: 0.0 }, prompt: parameters.get("prompt") ?? "Once upon a time" } as any;
  }
}
export const fromUrl = named();
