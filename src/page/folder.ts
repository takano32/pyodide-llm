// src/page/folder.ts (T.374.2.2): the model the page makes of a folder of the visitor's disk that holds what Hugging Face
// publishes. Nothing of the page is read here (no element, no address): tests/worker-fetches-check.mjs makes the
// folders it hands the worker by this very function, in Node.

// The files Hugging Face publishes: model.safetensors, config.json and a tokenizer (tokenizer.json, or a
// sentencepiece tokenizer.model / spiece.model). The worker converts them in the browser, to int8 and with a
// context of 512 tokens unless the settings say otherwise ({"conversion": {"dtype": ..., "max_seq_len": ...}}).
import { PYTHON_TOKENIZERS } from "../runtime/built/facts/tokenizers.ts";

// The tokenizers a folder may come with: the conversion's own list, Python's (TOKENIZERS of
// src/python/convert/conduct.py: which are tried, and in which order). This page asks for one of them before Pyodide is
// there to say: python_facts.py writes the list into src/runtime/built/facts/tokenizers.ts, read here (T.628.2)
export const TOKENIZERS: readonly string[] = PYTHON_TOKENIZERS;
const HF_IGNORED = ["tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "model.safetensors.index.json"];
export async function openHuggingFace(chosen: File[]) {
  const named = (...names: string[]) => chosen.find(({ name }) => names.includes(name.toLowerCase()));
  const weights = chosen.filter(({ name }) => name.toLowerCase().endsWith(".safetensors"));
  // the tokenizers the folder has, of those the conversion tries (T.138: a folder with both, as RakutenAI 2.0 mini
  // publishes, stopped at the tokenizer.json the converter refuses). The page asks for one of them, and reads none
  const config = named("config.json"), tokenizers = TOKENIZERS.map((name) => named(name)).filter(Boolean);
  const settings = chosen.find(({ name }) => name.toLowerCase().endsWith(".json") && name !== config?.name && !tokenizers.some((file) => file!.name === name) && !HF_IGNORED.includes(name.toLowerCase()));
  if (weights.length !== 1 || !config || !tokenizers.length) {
    throw new Error(`A Hugging Face model needs three files together: one .safetensors file (a model in several shards is not supported), config.json, and ${TOKENIZERS.slice(0, -1).join(", ")} or ${TOKENIZERS[TOKENIZERS.length - 1]}.`);
  }
  const given = settings ? JSON.parse(await settings.text()) : {};
  return {
    id: "local", name: given.name ?? weights[0].name, note: `local · Hugging Face · ${(weights[0].size / 1e6).toFixed(0)} MB`,
    // T.374.2.2: the folder as it was chosen, and which of it is the weights. What else of it is read (config.json, the
    // chat template where it has one, T.127, the first tokenizer the converter can read, T.138) is asked for by name
    // by the conduct of the conversion (src/python/convert/conduct.py), and answered from these Files by the worker
    hf: { files: chosen, weights: weights[0].name },
    conversion: given.conversion ?? {}, options: given.options ?? {},
    // a model nobody has tuned this page for: sample, as such models loop when they decode greedily
    generation: given.generation ?? { steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty: 1.1 },
    prompt: given.prompt ?? "", placeholder: given.placeholder ?? "",
  };
}

// T.411: the model the page makes of a checkpoint of llama2.c's format and its tokenizer.bin of the visitor's disk (moved
// here from choose.ts as it was, so that Node tries it). The legacy format knows nothing about itself, so whatever
// differs from llama2.c's conventions comes as a third file: .json, shaped like an entry of src/models.ts ({options,
// generation, prompt, placeholder}). ask: the browser's confirm(); undefined where the visitor declines a large file
export async function openLegacy(chosen: File[], ask: (question: string) => boolean) {
  const settings = chosen.find(({ name }) => name.toLowerCase().endsWith(".json"));
  // the larger of the other two is the checkpoint
  const [checkpoint, tokenizer, ...more] = chosen.filter((file) => file !== settings).sort((a, b) => b.size - a.size);
  if (!tokenizer || more.length) {
    throw new Error("Choose two files together: the checkpoint and its tokenizer.bin (and, optionally, settings as .json).");
  }
  const given = settings ? JSON.parse(await settings.text()) : {};
  // WebAssembly addresses 32 bits, and a phone gives a tab far less than that
  if (checkpoint.size > 1e9 && !ask(`${checkpoint.name} has ${(checkpoint.size / 1e9).toFixed(1)} GB. That may be more than this browser can hold. Read it anyway?`)) {
    return undefined;
  }
  return {
    id: "local", name: given.name ?? checkpoint.name, note: `local · ${(checkpoint.size / 1e6).toFixed(0)} MB`,
    file: checkpoint, tokenizerFile: tokenizer, bytes: checkpoint.size, options: given.options ?? {},
    generation: given.generation ?? { steps: 0, temperature: 0.0 },
    prompt: given.prompt ?? "Once upon a time", placeholder: given.placeholder ?? "",
  };
}
