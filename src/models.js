// The first model is the default: llm-jp-3 writes the best Japanese, and the SIMD kernels made it quick enough.
// ?model=<id> picks another one. Every file is fetched when the site is built (see the Makefile): llm-jp-3 and
// tiny-lm are converted from their Hugging Face checkpoints by convert_hf.py, and the larger models are quantized
// to int8 by quantize.py. bytes is the checkpoint size: it sizes the download buffer and the progress bar.
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
// Models that huggingface.co serves and this page converts itself (public/llama2_convert.py, the code that builds
// the models above): plain Llama architecture, one safetensors file, a Unigram tokenizer.json or a sentencepiece
// model. revision pins the commit, so that nothing changes under the page. download is the size of model.safetensors.
const hf = (repo, revision, tokenizer = "tokenizer.json") => ({ repo, revision, weights: "model.safetensors", config: "config.json", tokenizer });
const llmJp = { stop_tokens: [1, 2, 7] };

// group: "site" (built with the site, the default), "original" or "hf"
export const GROUPS = { site: "Models of this site", original: "Unquantized originals", hf: "From Hugging Face, converted in this browser" };

export const MODELS = [
  { id: "llm-jp-3-150m", name: "llm-jp-3 150M", note: "日本語 / English · int8 · 171 MB",
    checkpoint: "llm-jp-3-150m.bin", bytes: 171395100, tokenizer: "llm-jp-3-150m.tokenizer.bin",
    options: { dtype: "int8", ...unigram, stop_tokens: [1, 2, 7] },
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "tiny-lm", name: "tiny-lm 29M", note: "日本語 / English · int8 · 33 MB",
    checkpoint: "tiny-lm.bin", bytes: 32891932, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "int8", ...unigram, nfkc: true, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "stories15M", name: "TinyStories 15M", note: "English · int8 · 17 MB",
    checkpoint: "stories15M.bin", bytes: 17101468, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories3_5M", name: "TinyStories 3.5M", note: "English · float32 · 15 MB · fast",
    checkpoint: "stories3_5M-v4k.bin", bytes: 14887004, tokenizer: "tok4096.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories260K", name: "TinyStories 260K", note: "English · float32 · 1 MB · tiny",
    checkpoint: "stories260K.bin", bytes: 1056540, tokenizer: "tok512.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories42M", name: "TinyStories 42M", note: "English · int8 · 47 MB · desktop only",
    checkpoint: "stories42M.bin", bytes: 46925852, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  // the unquantized originals, to compare with int8
  { group: "original", id: "llm-jp-3-150m-f16", name: "llm-jp-3 150M (original)", note: "日本語 / English · float16 · 305 MB · desktop only",
    checkpoint: "llm-jp-3-150m.f16", bytes: 305161244, tokenizer: "llm-jp-3-150m.tokenizer.bin",
    options: { dtype: "float16", ...unigram, stop_tokens: [1, 2, 7] },
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "original", id: "tiny-lm-f16", name: "tiny-lm 29M (original)", note: "日本語 / English · float16 · 59 MB",
    checkpoint: "tiny-lm.f16", bytes: 58724892, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "float16", ...unigram, nfkc: true, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "original", id: "stories15M-f32", name: "TinyStories 15M (original)", note: "English · float32 · 61 MB",
    checkpoint: "stories15M.f32", bytes: 60816028, tokenizer: "tokenizer.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { group: "original", id: "stories42M-f32", name: "TinyStories 42M (original)", note: "English · float32 · 167 MB · desktop only",
    checkpoint: "stories42M.f32", bytes: 167020572, tokenizer: "tokenizer.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-llm-jp-3-150m-instruct3", name: "llm-jp-3 150M instruct3", note: "answers instructions · 日本語 · fetches 305 MB → int8 171 MB",
    hf: hf("llm-jp/llm-jp-3-150m-instruct3", "5be263e1a3613cd5c163f41ad828c8de6a2aa6ec"), download: 304649360, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m", name: "llm-jp-3 440M", note: "日本語 / English · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m", "0bfbf24efdcc5e4c57327e9c52e8cd832637adc2"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m-instruct3", name: "llm-jp-3 440M instruct3", note: "answers instructions · 日本語 · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m-instruct3", "a308f143bd5824c4033b3a2efaa1c00afbb3aa9e"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-980m-instruct3", name: "llm-jp-3 980M instruct3", note: "answers instructions · 日本語 · fetches 2.0 GB → int8 1.1 GB · desktop only",
    hf: hf("llm-jp/llm-jp-3-980m-instruct3", "c079dbf3f88aa2ab702b9696231fc3336c46b1be"), download: 1980382824, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // TinyLlama's template has </s> between the turns: specials makes the tokenizer read it as the token, not as text
  { group: "hf", id: "hf-tinyllama-1.1b-chat", name: "TinyLlama 1.1B Chat", note: "answers instructions · English · fetches 2.2 GB → int8 1.2 GB · desktop only",
    hf: hf("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6", "tokenizer.model"), download: 2200119864,
    conversion: {}, options: { specials: ["</s>"] }, generation: sampled(1.1), template: "<|user|>\n{prompt}</s>\n<|assistant|>\n",
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
];
