// The first model is the default: tiny-lm, the lightest one that writes Japanese. A public page should not make a
// phone fetch 171 MB unasked, and it is ready soonest; llm-jp-3 writes far better Japanese and is one choice away.
// Within each group the order is Japanese from light to heavy, then English from light to heavy.
// ?model=<id> picks another one. Every file of the first two groups is fetched when the site is built (see the
// Makefile): llm-jp-3 and tiny-lm are converted from their Hugging Face checkpoints by convert_hf.py, and the larger
// models are quantized to int8. bytes is the checkpoint size: it sizes the download buffer and the progress bar.
const JAPANESE = "文章の書き出しを入力（例: 富士山は、）";
const STORY = "Type the beginning of a story (e.g. Lily and Tom went to the park.)";
const unigram = { tokenizer_kind: "unigram" };
// greedy decoding makes small models loop, so the Japanese ones sample, and penalize repetition
// steps: 0 is as many tokens as the context of the model holds. Nothing here holds a model back by default.
const sampled = (repetition_penalty) => ({ steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty });
const greedy = { steps: 0, temperature: 0.0 };

// An instruction-tuned model answers instead of continuing, when its input has the form it was trained on: template
// wraps what the visitor typed ({prompt}), for one turn and no more. The forms are the chat_template of each model.
const LLM_JP_INSTRUCT = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{prompt}\n\n### 応答:\n";
const ASK_JAPANESE = "質問や指示を入力（例: 日本の首都は？）";
// ChatML. <|im_start|> and <|im_end|> are tokens of their own, so the engine is told to read them as such
const CHATML = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
const chatml = { specials: ["<|im_start|>", "<|im_end|>"], stop_tokens: [0, 2] };
// Models that huggingface.co serves and this page converts itself (public/llama2_convert.py, the code that builds
// the models above): plain Llama architecture, one safetensors file, a Unigram tokenizer.json or a sentencepiece
// model. revision pins the commit, so that nothing changes under the page. download is the size of model.safetensors.
const hf = (repo, revision, tokenizer = "tokenizer.json") => ({ repo, revision, weights: "model.safetensors", config: "config.json", tokenizer });
const llmJp = { stop_tokens: [1, 2, 7] };

// Where each model comes from, and under which license (T87). A model of this site names its source with
// `source`; one fetched from Hugging Face is its `hf.repo`. The page lists them from here, and
// tests/models-check.mjs fails when a model has no license, so a new model cannot be added without one.
const APACHE = "Apache License 2.0";
const MIT = "MIT License";
const LLAMA_32 = "Llama 3.2 Community License";
export const LICENSES = {
  "sbintuitions/tiny-lm": MIT, "llm-jp/llm-jp-3-150m": APACHE, "karpathy/tinyllamas": MIT, "ellishg/tinyllamas": MIT,
  "llm-jp/llm-jp-3-150m-instruct3": APACHE, "llm-jp/llm-jp-3-440m": APACHE, "llm-jp/llm-jp-3-440m-instruct3": APACHE,
  "llm-jp/llm-jp-3-980m-instruct3": APACHE, "rinna/japanese-gpt2-small": MIT, "rinna/japanese-gpt-neox-small": MIT,
  "Qwen/Qwen2.5-0.5B-Instruct": APACHE, "Qwen/Qwen2.5-Coder-0.5B-Instruct": APACHE, "Qwen/Qwen2.5-1.5B-Instruct": APACHE,
  "sbintuitions/sarashina2.2-0.5b": MIT, "sbintuitions/sarashina2.2-0.5b-instruct-v0.1": MIT,
  "EleutherAI/pythia-70m-deduped": APACHE, "EleutherAI/pythia-160m": APACHE, "EleutherAI/pythia-410m": APACHE,
  "EleutherAI/pythia-1b": APACHE, "EleutherAI/pythia-1.4b": APACHE,
  "HuggingFaceTB/SmolLM2-135M-Instruct": APACHE, "HuggingFaceTB/SmolLM2-360M-Instruct": APACHE,
  "bartowski/SmolLM2-135M-Instruct-GGUF": APACHE,
  "openai-community/gpt2": MIT, "TinyLlama/TinyLlama-1.1B-Chat-v1.0": APACHE,
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": MIT,
  "meta-llama/Llama-3.2-1B-Instruct": LLAMA_32, "unsloth/Llama-3.2-1B-Instruct": LLAMA_32,
};
/** The Hugging Face repository a model comes from. */
export const sourceOf = (entry) => entry.hf?.repo ?? entry.source;
/** Every source once, in the order of the list, with its license and the names of the models taken from it. A
 * model fetched from a redistribution (a GGUF, T74) names both: where it comes from, and whose model it is. */
export function sources(models = MODELS) {
  const bySource = new Map();
  for (const entry of models) {
    for (const repo of [entry.original, sourceOf(entry)].filter(Boolean)) {
      if (!bySource.has(repo)) bySource.set(repo, { repo, license: LICENSES[repo], names: [] });
      bySource.get(repo).names.push(entry.original && repo === sourceOf(entry) ? `${entry.name} (GGUF)` : entry.name);
    }
  }
  return [...bySource.values()];
}

// group: "site" (built with the site, the default), "original" or "hf"
export const GROUPS = { site: "Models of this site", original: "Unquantized originals", hf: "From Hugging Face, converted in this browser" };

export const MODELS = [
  { id: "tiny-lm", name: "tiny-lm 29M", note: "日本語 / English · int8 · 33 MB",
    source: "sbintuitions/tiny-lm", checkpoint: "tiny-lm.bin", bytes: 32891932, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "int8", ...unigram, nfkc: true, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "llm-jp-3-150m", name: "llm-jp-3 150M", note: "日本語 / English · int8 · 171 MB",
    source: "llm-jp/llm-jp-3-150m", checkpoint: "llm-jp-3-150m.bin", bytes: 171395100, tokenizer: "llm-jp-3-150m.tokenizer.bin",
    options: { dtype: "int8", ...unigram, stop_tokens: [1, 2, 7] },
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "stories260K", name: "TinyStories 260K", note: "English · float32 · 1 MB · tiny",
    source: "karpathy/tinyllamas", checkpoint: "stories260K.bin", bytes: 1056540, tokenizer: "tok512.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories3_5M", name: "TinyStories 3.5M", note: "English · float32 · 15 MB · fast",
    source: "ellishg/tinyllamas", checkpoint: "stories3_5M-v4k.bin", bytes: 14887004, tokenizer: "tok4096.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories15M", name: "TinyStories 15M", note: "English · int8 · 17 MB",
    source: "karpathy/tinyllamas", checkpoint: "stories15M.bin", bytes: 17101468, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories42M", name: "TinyStories 42M", note: "English · int8 · 47 MB · desktop only",
    source: "karpathy/tinyllamas", checkpoint: "stories42M.bin", bytes: 46925852, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  // the unquantized originals, to compare with int8
  { group: "original", id: "tiny-lm-f16", name: "tiny-lm 29M (original)", note: "日本語 / English · float16 · 59 MB",
    source: "sbintuitions/tiny-lm", checkpoint: "tiny-lm.f16", bytes: 58724892, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "float16", ...unigram, nfkc: true, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "original", id: "llm-jp-3-150m-f16", name: "llm-jp-3 150M (original)", note: "日本語 / English · float16 · 305 MB · desktop only",
    source: "llm-jp/llm-jp-3-150m", checkpoint: "llm-jp-3-150m.f16", bytes: 305161244, tokenizer: "llm-jp-3-150m.tokenizer.bin",
    options: { dtype: "float16", ...unigram, stop_tokens: [1, 2, 7] },
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "original", id: "stories15M-f32", name: "TinyStories 15M (original)", note: "English · float32 · 61 MB",
    source: "karpathy/tinyllamas", checkpoint: "stories15M.f32", bytes: 60816028, tokenizer: "tokenizer.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { group: "original", id: "stories42M-f32", name: "TinyStories 42M (original)", note: "English · float32 · 167 MB · desktop only",
    source: "karpathy/tinyllamas", checkpoint: "stories42M.f32", bytes: 167020572, tokenizer: "tokenizer.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  // Fetched from huggingface.co and converted in the page. Japanese from light to heavy, then English from light
  // to heavy, as everywhere else. What "fetches" says is the download; int8 is what it becomes here.
  { group: "hf", id: "hf-llm-jp-3-150m-instruct3", name: "llm-jp-3 150M instruct3", note: "answers instructions · 日本語 · fetches 305 MB → int8 171 MB",
    hf: hf("llm-jp/llm-jp-3-150m-instruct3", "5be263e1a3613cd5c163f41ad828c8de6a2aa6ec"), download: 304649360, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-japanese-gpt2-small", name: "japanese-gpt2 small", note: "日本語 · fetches 454 MB → int8 130 MB",
    hf: hf("rinna/japanese-gpt2-small", "f7fdefe2941d9629a7b2894564435e0e035df6a6", "spiece.model"), download: 454274094,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-gpt-neox-small", name: "japanese-gpt-neox small", note: "日本語 · fetches 663 MB → int8 193 MB",
    hf: hf("rinna/japanese-gpt-neox-small", "84d18c0fa8c9940a61cfc4e25bd9a5686898bac1", "spiece.model"), download: 663470088,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-qwen2.5-0.5b-instruct", name: "Qwen2.5 0.5B Instruct", note: "answers instructions · 日本語 / English · fetches 0.9 GB → int8 545 MB",
    hf: hf("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"), download: 988097824,
    conversion: {}, options: { ...chatml, stop_tokens: [151643, 151645] }, generation: sampled(1.1), template: CHATML,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m", name: "llm-jp-3 440M", note: "日本語 / English · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m", "0bfbf24efdcc5e4c57327e9c52e8cd832637adc2"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m-instruct3", name: "llm-jp-3 440M instruct3", note: "answers instructions · 日本語 · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m-instruct3", "a308f143bd5824c4033b3a2efaa1c00afbb3aa9e"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // sarashina2.2 (T79): the instruct one is published as a single shard with an index (T78)
  { group: "hf", id: "hf-sarashina2.2-0.5b", name: "sarashina2.2 0.5B", note: "日本語 · fetches 1.6 GB → int8 0.6 GB · desktop only",
    hf: hf("sbintuitions/sarashina2.2-0.5b", "5fb086c49f49824cfc93f09cc4ed5cd5917bef3d", "tokenizer.model"),
    download: 1586121792, conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-0.5b-instruct", name: "sarashina2.2 0.5B Instruct", note: "answers instructions · 日本語 · fetches 1.6 GB → int8 0.6 GB · desktop only",
    hf: hf("sbintuitions/sarashina2.2-0.5b-instruct-v0.1", "e4b9aacc3f644893d0179847946ef6c58d868f29", "tokenizer.model"),
    download: 1586121792, conversion: {}, options: {}, generation: sampled(1.1), template: CHATML,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-980m-instruct3", name: "llm-jp-3 980M instruct3", note: "answers instructions · 日本語 · fetches 2.0 GB → int8 1.1 GB · desktop only",
    hf: hf("llm-jp/llm-jp-3-980m-instruct3", "c079dbf3f88aa2ab702b9696231fc3336c46b1be"), download: 1980382824, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // English. Pythia is the same design at five sizes: a ladder for measuring (T80)
  { group: "hf", id: "hf-pythia-70m", name: "Pythia 70M", note: "English · fetches 166 MB → int8 96 MB",
    hf: hf("EleutherAI/pythia-70m-deduped", "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"), download: 166029852,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  // T74: from a GGUF (Q8_0): 145 MB instead of the 269 MB of model.safetensors, and the same int8 in the end (99.7%
  // of the most likely tokens and 0.13% of perplexity, tests/gguf_check.py). The GGUF is a redistribution; the
  // model and its license are HuggingFaceTB's (`original`)
  { group: "hf", id: "hf-smollm2-135m-instruct", name: "SmolLM2 135M Instruct", note: "answers instructions · English · fetches 145 MB (GGUF) → int8 145 MB",
    original: "HuggingFaceTB/SmolLM2-135M-Instruct",
    hf: { repo: "bartowski/SmolLM2-135M-Instruct-GGUF", revision: "09816acd5d99df7be770d85ea30822623dab342c",
          weights: "SmolLM2-135M-Instruct-Q8_0.gguf" }, download: 144811360,
    conversion: {}, options: chatml, generation: sampled(1.1), template: CHATML,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-pythia-160m", name: "Pythia 160M", note: "English · fetches 375 MB → int8 213 MB",
    hf: hf("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46"), download: 374998696,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-gpt2", name: "GPT-2 124M", note: "English · fetches 548 MB → int8 157 MB",
    hf: hf("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"), download: 548105171,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-smollm2-360m-instruct", name: "SmolLM2 360M Instruct", note: "answers instructions · English · fetches 724 MB → int8 390 MB",
    hf: hf("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"), download: 723674912,
    conversion: {}, options: chatml, generation: sampled(1.1), template: CHATML,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-pythia-410m", name: "Pythia 410M", note: "English · fetches 911 MB → int8 506 MB",
    hf: hf("EleutherAI/pythia-410m", "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"), download: 911373632,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-qwen2.5-coder-0.5b-instruct", name: "Qwen2.5 Coder 0.5B Instruct", note: "writes code · English · fetches 988 MB → int8 545 MB",
    hf: hf("Qwen/Qwen2.5-Coder-0.5B-Instruct", "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"), download: 988097824,
    conversion: {}, options: { ...chatml, stop_tokens: [151643, 151645] }, generation: sampled(1.1), template: CHATML,
    prompt: "Write a Python function that reverses a string.", placeholder: "Ask for code (e.g. Write a Python function that sorts a list.)" },
  { group: "hf", id: "hf-pythia-1b", name: "Pythia 1B", note: "English · fetches 2.1 GB → int8 1.1 GB · desktop only",
    hf: hf("EleutherAI/pythia-1b", "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"), download: 2090701528,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  // TinyLlama's template has </s> between the turns: specials makes the tokenizer read it as the token, not as text
  { group: "hf", id: "hf-tinyllama-1.1b-chat", name: "TinyLlama 1.1B Chat", note: "answers instructions · English · fetches 2.2 GB → int8 1.2 GB · desktop only",
    hf: hf("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6", "tokenizer.model"), download: 2200119864,
    conversion: {}, options: { specials: ["</s>"] }, generation: sampled(1.1), template: "<|user|>\n{prompt}</s>\n<|assistant|>\n",
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  // T106: Llama 3. The original (meta-llama) is gated, so the same weights come from unsloth's copy (`original`
  // names whose they are). The chat template and its special tokens are read from the model (T73)
  { group: "hf", id: "hf-llama-3.2-1b-instruct", name: "Llama 3.2 1B Instruct", note: "answers instructions · English · fetches 2.5 GB → int8 1.4 GB · desktop only",
    original: "meta-llama/Llama-3.2-1B-Instruct",
    hf: hf("unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c"), download: 2471645608,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-pythia-1.4b", name: "Pythia 1.4B", note: "English · fetches 2.9 GB → int8 1.5 GB · desktop only",
    hf: hf("EleutherAI/pythia-1.4b", "fedc38a16eea3bd36a96b906d78d11d2ce18ed79"), download: 2930002184,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-qwen2.5-1.5b-instruct", name: "Qwen2.5 1.5B Instruct", note: "answers instructions · 日本語 / English · fetches 3.1 GB → int8 1.6 GB · desktop only",
    hf: hf("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"), download: 3087467144,
    conversion: {}, options: { ...chatml, stop_tokens: [151643, 151645] }, generation: sampled(1.1), template: CHATML,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-deepseek-r1-qwen-1.5b", name: "DeepSeek-R1 Distill Qwen 1.5B", note: "thinks before it answers · English · fetches 3.6 GB → int8 1.6 GB · desktop only",
    hf: hf("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"), download: 3554214621,
    conversion: {}, options: { specials: ["<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"], stop_tokens: [151643] },
    generation: sampled(1.1), template: "<｜User｜>{prompt}<｜Assistant｜>",
    prompt: "What is 17 times 24? Think first.", placeholder: "Ask something that needs thinking" },
];

// T90: memory. A device that runs out of it kills the worker's WebAssembly memory, so the page warns before it
// loads a model that probably does not fit, and says what happened when it did not.
/** What the page takes besides the model: Pyodide, NumPy and the engine (the margin T90 asked for; the heap of
 * llm-jp-3 150M measures 112 MB above its 171 MB of weights, and the tab needs its own). */
export const PAGE_MEMORY = 300e6;
const megabytes = (bytes) => `${Math.round(bytes / 1e6).toLocaleString("en")} MB`;
/** The bytes of a model once loaded: `bytes` of a file of this site, or the "int8 N MB" its note gives for a
 * conversion. undefined when neither says (a file of the visitor's). */
export function modelBytes(entry) {
  // a float16 original is widened to float32 when loaded, next to the file it came from: llm-jp-3 150M's 305 MB
  // file measures about 800 MB of heap (AGENTS.md), so three times the file is the honest estimate
  if (entry.bytes) return entry.options?.dtype === "float16" ? entry.bytes * 3 : entry.bytes;
  const found = /int8 ([\d.]+) (MB|GB)/.exec(entry.note ?? "");
  const int8 = found ? Number(found[1]) * (found[2] === "GB" ? 1e9 : 1e6) : undefined;
  return int8 && entry.conversion?.dtype === "int6" ? int8 * SIX_OF_EIGHT : int8;
}

// T98: a model converted in the page can keep its weights in six bits instead of eight: 24 bytes and a scale per
// group of 32 against 32 and a scale, 7/9 of the size, at +1 to +3.4% of perplexity (measured on eight models), and
// slower on one thread (the groups are widened as they are read). So it is taken where int8 does not fit.
export const SIX_OF_EIGHT = 28 / 36;
/** The dtype a model of Hugging Face is converted to: the entry's own when it has one (the settings of a visitor's
 * files); else asked is ?bits= (or a setting), "8", "6" or anything else for
 * automatic, which takes int6 where int8 would pass half of what the device says it has (deviceMemory, Chromium
 * only), and otherwise leaves the choice to the worker (undefined): it knows the model's header once it converts,
 * and with it what the forward pass needs, and takes int6 where int8 would not fit a 32-bit memory (T115).
 * undefined for a model that is not converted in the page. */
export function weightsFor(entry, asked, deviceMemory) {
  if (!entry.hf) return undefined;
  // a visitor's own files may come with settings that say it ({"conversion": {"dtype": ...}}): they win (T119)
  if (entry.conversion?.dtype) return entry.conversion.dtype;
  if (asked === "6" || asked === "8") return `int${asked}`;
  const int8 = modelBytes({ ...entry, conversion: { ...entry.conversion, dtype: "int8" } });
  return int8 && deviceMemory && int8 + PAGE_MEMORY > deviceMemory * 2 ** 30 / 2 ? "int6" : undefined;
}
/** A sentence for a device that says it has less memory than twice what the model needs, or "". deviceMemory is
 * navigator.deviceMemory (GB; only Chromium tells, and at most 8): without it nothing is guessed. */
export function memoryWarning(entry, deviceMemory) {
  const bytes = modelBytes(entry);
  if (!deviceMemory || !bytes || bytes + PAGE_MEMORY <= deviceMemory * 2 ** 30 / 2) return "";
  return `${entry.name} needs about ${megabytes(bytes + PAGE_MEMORY)} of memory, and this device has ${deviceMemory} GB: it may run out of memory.`;
}
/** What the page says when the worker ran out of memory (heap: the size of its WebAssembly memory then). */
export function memoryFailure(entry, heap, detail) {
  const bytes = modelBytes(entry);
  return `This device ran out of memory for ${entry.name}` +
    (bytes ? ` (it needs about ${megabytes(bytes + PAGE_MEMORY)})` : "") +
    (heap ? `; the page was using ${megabytes(heap)} when it happened` : "") +
    `. A smaller model may fit, or closing other tabs may help.` + (detail ? ` (${detail})` : "");
}
