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

// group: "site" (built with the site, the default), "original" or "hf"
export const GROUPS = { site: "Models of this site", original: "Unquantized originals", hf: "From Hugging Face, converted in this browser" };

export const MODELS = [
  { id: "tiny-lm", name: "tiny-lm 29M", note: "日本語 / English · int8 · 33 MB",
    checkpoint: "tiny-lm.bin", bytes: 32891932, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "int8", ...unigram, nfkc: true, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "llm-jp-3-150m", name: "llm-jp-3 150M", note: "日本語 / English · int8 · 171 MB",
    checkpoint: "llm-jp-3-150m.bin", bytes: 171395100, tokenizer: "llm-jp-3-150m.tokenizer.bin",
    options: { dtype: "int8", ...unigram, stop_tokens: [1, 2, 7] },
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "stories260K", name: "TinyStories 260K", note: "English · float32 · 1 MB · tiny",
    checkpoint: "stories260K.bin", bytes: 1056540, tokenizer: "tok512.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories3_5M", name: "TinyStories 3.5M", note: "English · float32 · 15 MB · fast",
    checkpoint: "stories3_5M-v4k.bin", bytes: 14887004, tokenizer: "tok4096.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories15M", name: "TinyStories 15M", note: "English · int8 · 17 MB",
    checkpoint: "stories15M.bin", bytes: 17101468, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories42M", name: "TinyStories 42M", note: "English · int8 · 47 MB · desktop only",
    checkpoint: "stories42M.bin", bytes: 46925852, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  // the unquantized originals, to compare with int8
  { group: "original", id: "tiny-lm-f16", name: "tiny-lm 29M (original)", note: "日本語 / English · float16 · 59 MB",
    checkpoint: "tiny-lm.f16", bytes: 58724892, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "float16", ...unigram, nfkc: true, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "original", id: "llm-jp-3-150m-f16", name: "llm-jp-3 150M (original)", note: "日本語 / English · float16 · 305 MB · desktop only",
    checkpoint: "llm-jp-3-150m.f16", bytes: 305161244, tokenizer: "llm-jp-3-150m.tokenizer.bin",
    options: { dtype: "float16", ...unigram, stop_tokens: [1, 2, 7] },
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "original", id: "stories15M-f32", name: "TinyStories 15M (original)", note: "English · float32 · 61 MB",
    checkpoint: "stories15M.f32", bytes: 60816028, tokenizer: "tokenizer.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { group: "original", id: "stories42M-f32", name: "TinyStories 42M (original)", note: "English · float32 · 167 MB · desktop only",
    checkpoint: "stories42M.f32", bytes: 167020572, tokenizer: "tokenizer.bin", options: {},
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
  { group: "hf", id: "hf-smollm2-135m-instruct", name: "SmolLM2 135M Instruct", note: "answers instructions · English · fetches 269 MB → int8 145 MB",
    hf: hf("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"), download: 269060552,
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
