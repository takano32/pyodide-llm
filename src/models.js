// The first model is the default: tiny-lm, the lightest one that writes Japanese. A public page should not make a
// phone fetch 171 MB unasked, and it is ready soonest; llm-jp-3 writes far better Japanese and is one choice away.
// Within each group the order is the ones that write Japanese from light to heavy, then the English-only ones from
// light to heavy (T128): MODELS is sorted so at the end of this file, so a model added anywhere takes its place.
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
// sarashina2.2's chat_template (and CAT-Translate's, made from it) uses selectattr, which this project's template
// reader does not take (T73): one turn of it, as the real Jinja renders it, is this (T81; the 0.5B Instruct had
// ChatML here until 2026-09-26, whose <|im_start|> its vocabulary does not have). <|user|> (9), <|assistant|> (8)
// and </s> (2) are tokens of their own, and a model that writes the mark of a turn (7 to 9) has ended its answer
const SARASHINA = "<|user|>{prompt}</s><|assistant|>";
const sarashina = { specials: ["<|assistant|>", "<|user|>", "</s>"], stop_tokens: [1, 2, 7, 8, 9] };
const TRANSLATE = "Translate the following Japanese text into English.\n\n{日本語の文} (or English into Japanese)";
// llm-jp-4's chat_template is OpenAI's harmony (T132), with a system message of the model's name, its knowledge cutoff
// and the date ({date}: filled() writes today's). The template ends at "<|start|>assistant" and leaves the channel to
// the model; this one asks for the final channel, the answer, and so skips the analysis a harmony model may write
// first. Its tokenizer.json puts a "▁" before the text after each special token (a normalizer that replaces the start
// of every piece with it), which the engine does not: the space after each special token here makes the same tokens
// (the same IDs as the real Jinja and tokenizers for four prompts, T132). A turn ends with <|return|> (2), <|end|>
// (11) or <|call|> (13), and a new message would start with <|start|> (10)
const HARMONY = "<|start|> system<|message|> You are LLM-jp-4, a large language model trained by LLM-jp.\nKnowledge cutoff: " +
  "2025-12\nCurrent date: {date}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every " +
  "message.<|end|><|start|> user<|message|> {prompt}<|end|><|start|> assistant<|channel|> final<|message|>";
// T125: Mistral's formats, one turn as the real Jinja writes it (the same IDs as the real Jinja and tokenizers for
// four prompts, where the prompt has no space at either end: some templates trim it, the page does not). These
// models come with a sentencepiece tokenizer.model, which the engine reads (their tokenizer.json is a BPE of
// sentencepiece's kind, which it does not). v0.2 writes "<s> [INST]": after the BOS, the engine's dummy prefix is
// that space. v0.3's [INST] and [/INST] are tokens of their own (3 and 4)
const MISTRAL = "[INST] {prompt} [/INST]";
const MISTRAL_V3 = "[INST] {prompt}[/INST]";
// RakutenAI's (2.0 mini and 7B chat): no special tokens, a system sentence and USER / ASSISTANT
const RAKUTEN = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, " +
  "detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:";
// zephyr's tokenizer.json puts a "▁" before the text after </s> (a legacy Llama tokenizer), which the engine does
// not: the space after </s> makes the same tokens
const ZEPHYR = "<|user|>\n{prompt}</s> \n<|assistant|>\n";
const harmony = { specials: ["<|channel|>", "<|message|>", "<|start|>", "<|end|>"], stop_tokens: [1, 2, 10, 11, 13] };
/** What the page sends for a prompt in a model's template: {prompt} is what was typed, {date} today (YYYY-MM-DD, the
 * visitor's own day). What was typed goes in as it is: as a replacement string, its $$, $&, $` and $' were patterns
 * (a typed $' wrote the rest of the template, special tokens and all; the review of T132). */
export function filled(template, prompt, today = new Date()) {
  const date = [today.getFullYear(), today.getMonth() + 1, today.getDate()].map((n) => String(n).padStart(2, "0")).join("-");
  return template.replace("{date}", date).replace("{prompt}", () => prompt);
}
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
// TinySwallow's model card: "derived from Qwen (Apache 2.0) and trained on Gemma data (Gemma Terms, Prohibited Use).
// Use (including commercial) is permitted if you comply with both licenses/policies above."
const APACHE_GEMMA = "Apache License 2.0 (derived from Qwen), and the Gemma Terms of Use and Prohibited Use Policy (trained on Gemma data)";
// T132: Qwen2.5-3B's card names its own license (license_name: qwen-research), which is for research, not commercial
// use. Swallow's card says "META LLAMA 3.1 COMMUNITY LICENSE and Gemma Terms of Use" under License (its metadata says
// llama3.3 and gemma: the words of the card are copied)
const QWEN_RESEARCH = "Qwen Research License Agreement";
const SWALLOW = "Meta Llama 3.1 Community License and Gemma Terms of Use";
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
  // T81 (2026-09-26)
  "SakanaAI/TinySwallow-1.5B-Instruct": APACHE_GEMMA, "llm-jp/llm-jp-3.1-1.8b-instruct4": APACHE,
  "sbintuitions/sarashina2.2-1b-instruct-v0.1": MIT, "cyberagent/CAT-Translate-0.8b": MIT, "cyberagent/CAT-Translate-1.4b": MIT,
  "HuggingFaceTB/SmolLM2-1.7B-Instruct": APACHE,
  // T132 (2026-09-26)
  "Qwen/Qwen2.5-3B-Instruct": QWEN_RESEARCH, "sbintuitions/sarashina2.2-3b-instruct-v0.1": MIT,
  "Qwen/Qwen2.5-7B-Instruct": APACHE, "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5": SWALLOW,
  "llm-jp/llm-jp-4-8b-instruct": APACHE,
  // T126
  "rinna/japanese-gpt-1b": MIT,
  // T125
  "Rakuten/RakutenAI-2.0-mini-instruct": APACHE, "Rakuten/RakutenAI-7B-chat": APACHE,
  "tokyotech-llm/Swallow-MS-7b-instruct-v0.1": APACHE, "mistralai/Mistral-7B-Instruct-v0.2": APACHE,
  "mistralai/Mistral-7B-Instruct-v0.3": APACHE, "HuggingFaceH4/zephyr-7b-beta": MIT,
  "meta-llama/Llama-3.2-3B-Instruct": LLAMA_32, "unsloth/Llama-3.2-3B-Instruct": LLAMA_32,
};
/** The Hugging Face repository a model comes from. */
export const sourceOf = (entry) => entry.hf?.repo ?? entry.source;
/** Every source once, in the order of the list, with its license and the names of the models taken from it. A
 * model fetched from a redistribution names both: where it comes from, and whose model it is. The redistribution's
 * line says which it is: "(GGUF)" for a GGUF (T74), "(copy)" for the same safetensors elsewhere (unsloth's Llama;
 * until 2026-09-26 it said "(GGUF)" for those too). */
export function sources(models = MODELS) {
  const bySource = new Map();
  for (const entry of models) {
    for (const repo of [entry.original, sourceOf(entry)].filter(Boolean)) {
      if (!bySource.has(repo)) bySource.set(repo, { repo, license: LICENSES[repo], names: [] });
      const kind = entry.hf?.weights?.endsWith(".gguf") ? "GGUF" : "copy";
      bySource.get(repo).names.push(entry.original && repo === sourceOf(entry) ? `${entry.name} (${kind})` : entry.name);
    }
  }
  return [...bySource.values()];
}

// group: "site" (built with the site, the default), "original" or "hf"
export const GROUPS = { site: "Models of this site", original: "Unquantized originals", hf: "From Hugging Face, converted in this browser" };

// in the order they were added, more or less; MODELS below is the order of the list
const LISTED = [
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
  { group: "hf", id: "hf-sarashina2.2-0.5b", name: "sarashina2.2 0.5B", note: "日本語 · fetches 1.6 GB → int8 0.9 GB · desktop only",
    hf: hf("sbintuitions/sarashina2.2-0.5b", "5fb086c49f49824cfc93f09cc4ed5cd5917bef3d", "tokenizer.model"),
    download: 1586121792, conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-0.5b-instruct", name: "sarashina2.2 0.5B Instruct", note: "answers instructions · 日本語 · fetches 1.6 GB → int8 0.9 GB · desktop only",
    hf: hf("sbintuitions/sarashina2.2-0.5b-instruct-v0.1", "e4b9aacc3f644893d0179847946ef6c58d868f29", "tokenizer.model"),
    download: 1586121792, conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T81: a translator from sarashina2.2, Japanese to English and back, asked in its own words and greedy, as its
  // model card runs it (no generation_config: transformers' defaults; a repetition penalty bends a translation)
  { group: "hf", id: "hf-cat-translate-0.8b", name: "CAT-Translate 0.8B", note: "translates 日本語 ⇄ English · fetches 1.6 GB → int8 0.9 GB · desktop only",
    hf: hf("cyberagent/CAT-Translate-0.8b", "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9", "tokenizer.model"), download: 1586121792,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  { group: "hf", id: "hf-llm-jp-3-980m-instruct3", name: "llm-jp-3 980M instruct3", note: "answers instructions · 日本語 · fetches 2.0 GB → int8 1.1 GB · desktop only",
    hf: hf("llm-jp/llm-jp-3-980m-instruct3", "c079dbf3f88aa2ab702b9696231fc3336c46b1be"), download: 1980382824, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T126: a GPT-2 of 1.3B, whose activation is written gelu_fast (the same tanh approximation as gelu_new). Its
  // table of positions holds 1024, which is its context
  { group: "hf", id: "hf-japanese-gpt-1b", name: "japanese-gpt 1B", note: "日本語 · fetches 2.7 GB → int8 1.5 GB · desktop only",
    hf: hf("rinna/japanese-gpt-1b", "33fc2e4b4e97e229d24a2973073a1361157ecef6", "spiece.model"), download: 2655791788,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T81 (2026-09-26): the survey's Japanese models that convert as they are. TinySwallow reads its chat template
  // itself (T73); llm-jp-3.1 has its family's format with the system sentence its model card always passes (its
  // template alone leaves it out)
  { group: "hf", id: "hf-sarashina2.2-1b-instruct", name: "sarashina2.2 1B Instruct", note: "answers instructions · 日本語 · fetches 2.8 GB → int8 1.6 GB · desktop only",
    hf: hf("sbintuitions/sarashina2.2-1b-instruct-v0.1", "08cf5a8ae579be0fb5a9f802dda8a26acbc94951", "tokenizer.model"), download: 2815103168,
    conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-cat-translate-1.4b", name: "CAT-Translate 1.4B", note: "translates 日本語 ⇄ English · fetches 2.8 GB → int8 1.6 GB · desktop only",
    hf: hf("cyberagent/CAT-Translate-1.4b", "254120945fd9a61278ac2171ab07c831d56838fa", "tokenizer.model"), download: 2815103168,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  { group: "hf", id: "hf-tinyswallow-1.5b-instruct", name: "TinySwallow 1.5B Instruct", note: "answers instructions · 日本語 · fetches 3.1 GB → int8 1.7 GB · desktop only",
    hf: hf("SakanaAI/TinySwallow-1.5B-Instruct", "91e9fcc30f56d224aea84356c4d850cc4c5a3260"), download: 3087467144,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T125: a Mistral (a Llama by another name) of 1.5B, Japanese and English; its sliding window of 8192 is past the
  // context of 4096 the page gives it
  { group: "hf", id: "hf-rakutenai-2.0-mini-instruct", name: "RakutenAI 2.0 mini instruct", note: "answers instructions · 日本語 / English · fetches 3.1 GB → int8 1.7 GB · desktop only",
    hf: hf("Rakuten/RakutenAI-2.0-mini-instruct", "6d902489587d324b7d5e201299e4e1a169f3a40b", "tokenizer.model"), download: 3069389424,
    conversion: {}, options: {}, generation: sampled(1.1), template: RAKUTEN,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3.1-1.8b-instruct4", name: "llm-jp-3.1 1.8B instruct4", note: "answers instructions · 日本語 · fetches 3.7 GB → int8 2.1 GB · desktop only",
    hf: hf("llm-jp/llm-jp-3.1-1.8b-instruct4", "f19510db409090bb1737f24f868d17c4bdc86c8e"), download: 3735253776,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T132: the large ones, in shards (T105). Past 4 GiB with their forward pass they need a 64-bit memory (T101),
  // where the browser has one int8 (T133), else six bits (T98); the 7 to 8B ones do not fit a 32-bit memory even so
  { group: "hf", id: "hf-qwen2.5-3b-instruct", name: "Qwen2.5 3B Instruct", note: "answers instructions · 日本語 / English · fetches 6.2 GB → int8 3.5 GB · desktop only",
    hf: hf("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"), download: 6171926992,
    conversion: {}, options: { ...chatml, stop_tokens: [151643, 151645] }, generation: sampled(1.1), template: CHATML,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-3b-instruct", name: "sarashina2.2 3B Instruct", note: "answers instructions · 日本語 · fetches 6.7 GB → int8 3.8 GB · desktop only",
    hf: hf("sbintuitions/sarashina2.2-3b-instruct-v0.1", "4f3626fb1b64b3e97c908e67f27b2d627ba2a999", "tokenizer.model"), download: 6711252896,
    conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-qwen2.5-7b-instruct", name: "Qwen2.5 7B Instruct", note: "answers instructions · 日本語 / English · fetches 15.2 GB → int8 8.6 GB · desktop only · Chrome and Firefox",
    hf: hf("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"), download: 15231271888,
    conversion: {}, options: { ...chatml, stop_tokens: [151643, 151645] }, generation: sampled(1.1), template: CHATML,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T125: Mistral 7B's of Japanese, whose sliding window of 4096 is the page's context
  { group: "hf", id: "hf-rakutenai-7b-chat", name: "RakutenAI 7B chat", note: "answers instructions · 日本語 / English · fetches 14.7 GB → int8 8.3 GB · desktop only · Chrome and Firefox",
    hf: hf("Rakuten/RakutenAI-7B-chat", "7093167c61a0be6161cb68928c939c03fe0ab87d", "tokenizer.model"), download: 14745642040,
    conversion: {}, options: {}, generation: sampled(1.1), template: RAKUTEN,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-swallow-ms-7b-instruct", name: "Swallow-MS 7B instruct", note: "answers instructions · 日本語 / English · fetches 14.7 GB → int8 8.3 GB · desktop only · Chrome and Firefox",
    hf: hf("tokyotech-llm/Swallow-MS-7b-instruct-v0.1", "008d006f9065e37e39e31bf117ae8689390953e8", "tokenizer.model"), download: 14660445224,
    conversion: {}, options: {}, generation: sampled(1.1), template: `${MISTRAL} `,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // Swallow's own chat_template is read (T73): a Japanese system message, and a second BOS before the user's turn,
  // as the real Jinja writes it (the same IDs as the real Jinja and tokenizers, T132)
  { group: "hf", id: "hf-llama-3.1-swallow-8b-instruct", name: "Llama 3.1 Swallow 8B Instruct", note: "answers instructions · 日本語 / English · fetches 16.1 GB → int8 9.0 GB · desktop only · Chrome and Firefox",
    hf: hf("tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5", "b1f8317099a97e790ec872c1225ca155979b4816"), download: 16060556376,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-4-8b-instruct", name: "llm-jp-4 8B instruct", note: "answers instructions · 日本語 / English · fetches 17.2 GB → int8 9.7 GB · desktop only · Chrome and Firefox",
    hf: hf("llm-jp/llm-jp-4-8b-instruct", "098f2b2cf33021eba19a6d3582aa3d071ccc0aff"), download: 17180435544,
    conversion: {}, options: harmony, generation: sampled(1.1), template: HARMONY,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
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
  { group: "hf", id: "hf-pythia-1.4b", name: "Pythia 1.4B", note: "English · fetches 2.9 GB → int8 1.6 GB · desktop only",
    hf: hf("EleutherAI/pythia-1.4b", "fedc38a16eea3bd36a96b906d78d11d2ce18ed79"), download: 2930002184,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-qwen2.5-1.5b-instruct", name: "Qwen2.5 1.5B Instruct", note: "answers instructions · 日本語 / English · fetches 3.1 GB → int8 1.7 GB · desktop only",
    hf: hf("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"), download: 3087467144,
    conversion: {}, options: { ...chatml, stop_tokens: [151643, 151645] }, generation: sampled(1.1), template: CHATML,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-smollm2-1.7b-instruct", name: "SmolLM2 1.7B Instruct", note: "answers instructions · English · fetches 3.4 GB → int8 1.9 GB · desktop only",
    hf: hf("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674"), download: 3422777952,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-deepseek-r1-qwen-1.5b", name: "DeepSeek-R1 Distill Qwen 1.5B", note: "thinks before it answers · English · fetches 3.6 GB → int8 2.0 GB · desktop only",
    hf: hf("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"), download: 3554214621,
    conversion: {}, options: { specials: ["<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"], stop_tokens: [151643] },
    generation: sampled(1.1), template: "<｜User｜>{prompt}<｜Assistant｜>",
    prompt: "What is 17 times 24? Think first.", placeholder: "Ask something that needs thinking" },
  // T125: Mistral 7B, and zephyr made from it
  { group: "hf", id: "hf-mistral-7b-instruct-v0.2", name: "Mistral 7B Instruct v0.2", note: "answers instructions · English · fetches 14.5 GB → int8 8.2 GB · desktop only · Chrome and Firefox",
    hf: hf("mistralai/Mistral-7B-Instruct-v0.2", "63a8b081895390a26e140280378bc85ec8bce07a", "tokenizer.model"), download: 14483498016,
    conversion: {}, options: {}, generation: sampled(1.1), template: MISTRAL,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-mistral-7b-instruct-v0.3", name: "Mistral 7B Instruct v0.3", note: "answers instructions · English · fetches 14.5 GB → int8 8.2 GB · desktop only · Chrome and Firefox",
    hf: hf("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71", "tokenizer.model"), download: 14496080928,
    conversion: {}, options: { specials: ["[/INST]", "[INST]"] }, generation: sampled(1.1), template: MISTRAL_V3,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-zephyr-7b-beta", name: "zephyr 7B beta", note: "answers instructions · English · fetches 14.5 GB → int8 8.2 GB · desktop only · Chrome and Firefox",
    hf: hf("HuggingFaceH4/zephyr-7b-beta", "892b3d7a7b1cf10c7a701c60881cd93df615734c", "tokenizer.model"), download: 14483497952,
    conversion: {}, options: { specials: ["</s>"] }, generation: sampled(1.1), template: ZEPHYR,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-llama-3.2-3b-instruct", name: "Llama 3.2 3B Instruct", note: "answers instructions · English · fetches 6.4 GB → int8 3.6 GB · desktop only",
    original: "meta-llama/Llama-3.2-3B-Instruct",
    hf: hf("unsloth/Llama-3.2-3B-Instruct", "006f5dcd1393c3add266de40994ba96225e9689d"), download: 6425529048,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
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

/** T128: whether a model writes Japanese (its note says 日本語: Japanese alone, with English, or translating). */
export const writesJapanese = (entry) => (entry.note ?? "").includes("日本語");
/** The models as the list shows them (T128, the owner's order): the groups in the order of GROUPS, and within each
 * the ones that write Japanese from light to heavy, then the English-only ones from light to heavy, by modelBytes()
 * (int8 for a conversion, three times the file for a float16 original). The default, the first, is tiny-lm. */
export const MODELS = LISTED.map((entry) => ({ entry, key: [Object.keys(GROUPS).indexOf(entry.group ?? "site"), writesJapanese(entry) ? 0 : 1, modelBytes(entry)] }))
  .sort((a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1] || a.key[2] - b.key[2])
  .map(({ entry }) => entry);
// T133: Chromium's navigator.deviceMemory stops at 8: a device that says 8 has 8 GB or more, as many as it likes
export const DEVICE_MEMORY_CAP = 8;
/** The dtype a model of Hugging Face is converted to: the entry's own when it has one (the settings of a visitor's
 * files); else asked is ?bits= (or a setting), "8", "6" or anything else for
 * automatic, which takes int6 where int8 would pass half of what the device says it has (deviceMemory, Chromium
 * only, and below its cap of 8: a device at the cap may have any more), and otherwise leaves the choice to the
 * worker (undefined): it knows the model's header once it converts, and with it what the forward pass needs, and
 * takes int6 where int8 would not fit a 32-bit memory and the browser has no 64-bit one (T115, T133).
 * undefined for a model that is not converted in the page. */
export function weightsFor(entry, asked, deviceMemory) {
  if (!entry.hf) return undefined;
  // a visitor's own files may come with settings that say it ({"conversion": {"dtype": ...}}): they win (T119)
  if (entry.conversion?.dtype) return entry.conversion.dtype;
  if (asked === "6" || asked === "8") return `int${asked}`;
  if (!deviceMemory || deviceMemory >= DEVICE_MEMORY_CAP) return undefined;
  const int8 = modelBytes({ ...entry, conversion: { ...entry.conversion, dtype: "int8" } });
  return int8 && int8 + PAGE_MEMORY > deviceMemory * 2 ** 30 / 2 ? "int6" : undefined;
}
/** A sentence for a device that says it has less memory than twice what the model needs, or "". deviceMemory is
 * navigator.deviceMemory (GB; only Chromium tells, and at most 8): without it nothing is guessed. A device at the cap
 * has 8 GB or more, so it is warned only of a model that needs more than 8 GB (T133). */
export function memoryWarning(entry, deviceMemory) {
  const bytes = modelBytes(entry);
  if (!deviceMemory || !bytes) return "";
  const capped = deviceMemory >= DEVICE_MEMORY_CAP;
  if (bytes + PAGE_MEMORY <= deviceMemory * 2 ** 30 / (capped ? 1 : 2)) return "";
  return `${entry.name} needs about ${megabytes(bytes + PAGE_MEMORY)} of memory, and this device has ` +
    `${capped ? `${DEVICE_MEMORY_CAP} GB or more` : `${deviceMemory} GB`}: it may run out of memory.`;
}
/** What the page says when the worker ran out of memory (heap: the size of its WebAssembly memory then). */
export function memoryFailure(entry, heap, detail) {
  const bytes = modelBytes(entry);
  return `This device ran out of memory for ${entry.name}` +
    (bytes ? ` (it needs about ${megabytes(bytes + PAGE_MEMORY)})` : "") +
    (heap ? `; the page was using ${megabytes(heap)} when it happened` : "") +
    `. A smaller model may fit, or closing other tabs may help.` + (detail ? ` (${detail})` : "");
}
