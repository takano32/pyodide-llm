// src/page/folder.ts (T374.2.2): the model the page makes of a folder of the visitor's disk that holds what Hugging Face
// publishes. Nothing of the page is read here (no element, no address): tests/worker-fetches-check.mjs makes the
// folders it hands the worker by this very function, in Node.

// The files Hugging Face publishes: model.safetensors, config.json and a tokenizer (tokenizer.json, or a
// sentencepiece tokenizer.model / spiece.model). The worker converts them in the browser, to int8 and with a
// context of 512 tokens unless the settings say otherwise ({"conversion": {"dtype": ..., "max_seq_len": ...}}).
// The tokenizers a folder may come with. The conversion's own list is Python's (TOKENIZERS of
// public/convert/conduct.py: which are tried, and in which order); this page asks for one of them before Pyodide is
// there to say, so it keeps the names too, and tests/worker-fetches-check.mjs holds the two lists to each other
export const TOKENIZERS = ["tokenizer.json", "tokenizer.model", "spiece.model"];
const HF_IGNORED = ["tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "model.safetensors.index.json"];
export async function openHuggingFace(chosen: File[]) {
  const named = (...names: string[]) => chosen.find(({ name }) => names.includes(name.toLowerCase()));
  const weights = chosen.filter(({ name }) => name.toLowerCase().endsWith(".safetensors"));
  // the tokenizers the folder has, of those the conversion tries (T138: a folder with both, as RakutenAI 2.0 mini
  // publishes, stopped at the tokenizer.json the converter refuses). The page asks for one of them, and reads none
  const config = named("config.json"), tokenizers = TOKENIZERS.map((name) => named(name)).filter(Boolean);
  const settings = chosen.find(({ name }) => name.toLowerCase().endsWith(".json") && name !== config?.name && !tokenizers.some((file) => file!.name === name) && !HF_IGNORED.includes(name.toLowerCase()));
  if (weights.length !== 1 || !config || !tokenizers.length) {
    throw new Error(`A Hugging Face model needs three files together: one .safetensors file (a model in several shards is not supported), config.json, and ${TOKENIZERS.slice(0, -1).join(", ")} or ${TOKENIZERS[TOKENIZERS.length - 1]}.`);
  }
  const given = settings ? JSON.parse(await settings.text()) : {};
  return {
    id: "local", name: given.name ?? weights[0].name, note: `local · Hugging Face · ${(weights[0].size / 1e6).toFixed(0)} MB`,
    // T374.2.2: the folder as it was chosen, and which of it is the weights. What else of it is read (config.json, the
    // chat template where it has one, T127, the first tokenizer the converter can read, T138) is asked for by name
    // by the conduct of the conversion (public/convert/conduct.py), and answered from these Files by the worker
    hf: { files: chosen, weights: weights[0].name },
    conversion: given.conversion ?? {}, options: given.options ?? {},
    // a model nobody has tuned this page for: sample, as such models loop when they decode greedily
    generation: given.generation ?? { steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty: 1.1 },
    prompt: given.prompt ?? "", placeholder: given.placeholder ?? "",
  };
}
