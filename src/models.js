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
// the same for a model whose BOS is <|im_start|> itself, which the page begins every text with (SmolLM2's, and a Qwen3's
// QWEN3_FROM_IM_START): the format begins after it (T250's review)
const CHATML_AFTER_START = "user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
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
// RakutenAI's (2.0 mini and 7B chat): no special tokens, a system sentence and USER / ASSISTANT; the template trims
// what was typed (T138)
const RAKUTEN = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, " +
  "detailed, and polite answers to the user's questions. USER: {prompt:trim} ASSISTANT:";
// T138: Swallow-MS's card always passes this system message, which its template puts in the user's turn; the template
// strips the whole turn, which trims what was typed at its end only (filled() trims both ends: a prompt that begins
// with spaces differs)
const SWALLOW_MS = "[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{prompt:trim} [/INST] ";
// zephyr's tokenizer.json puts a "▁" before the text after </s> (a legacy Llama tokenizer), which the engine does
// not: the space after </s> makes the same tokens
const ZEPHYR = "<|user|>\n{prompt}</s> \n<|assistant|>\n";
// T249: EuroLLM's chat_template is ChatML with a system turn that is empty unless one is given. Its tokenizer (a legacy
// Llama tokenizer, as zephyr's) puts a "▁" before the text after <|im_start|> and <|im_end|>, which the engine does
// not: the space after each makes the same tokens (the same IDs as the real Jinja and tokenizers for
// tests/format_check.py's prompts; the converter's own reading of the template, without the spaces, made none the same)
const EUROLLM = "<|im_start|> system\n<|im_end|> \n<|im_start|> user\n{prompt}<|im_end|> \n<|im_start|> assistant\n";
// T250: Llama-3-ELYZA-JP's template is Llama 3's (no date, the turns trimmed), and its card always passes this system
// message, as Swallow-MS's does: one turn of it as the real Jinja writes it with that message
const ELYZA = "<|start_header_id|>system<|end_header_id|>\n\nあなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、" +
  "常に日本語で回答してください。<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt:trim}<|eot_id|>" +
  "<|start_header_id|>assistant<|end_header_id|>\n\n";
// T124: Qwen3 thinks before it answers (<think>…</think>, then the answer), which is the form its chat_template writes
// and the converter reads. The same weights answer at once when the answer begins with an empty thought: the form of
// enable_thinking=false. <think> and </think> are tokens of the vocabulary that tokenizer.json does not call special:
// the converter names those as specials with the rest (T143: <tool_call>, <|fim_prefix|> ...), so the list names none
// for Qwen3. A list of its own would go over the converter's and spell the others out where a visitor types them (T221)
const QWEN3_AT_ONCE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
// T250's review: the real Qwen3 tokenizer puts nothing in front of a text and its format begins with <|im_start|>, but
// the converter's BOS is <|endoftext|> (QWEN3_OWN_BOS below, and what config.json names). A Qwen3 of 8 billion
// parameters is much worse with it in front: its own answers to five questions are 178% higher in perplexity (Qwen3 8B,
// plain text 144%; Shisa V2.1's fine-tune of it 67% and 54%), where Qwen's 0.6B and 1.7B are 1 to 4% off and CAT-Thinking
// 8B (Qwen3 Swallow's, continued on Japanese text) 0.4% (tests/answer_check.mjs, tests/start_check.mjs). For those the BOS
// is the format's own first token, <|im_start|> (151644), and the formats begin after it (as Qwen3.5's, T236): the page
// sends the very IDs the real template makes. The answer stops at <|im_end|>, at <|endoftext|> and at the mark of a new turn
const QWEN3_FROM_IM_START = { bos: 151644, stop_tokens: [151643, 151644, 151645] };
const QWEN3_THINKING_AFTER_START = CHATML_AFTER_START;
const QWEN3_AT_ONCE_AFTER_START = `${QWEN3_THINKING_AFTER_START}<think>\n\n</think>\n\n`;
// the sampling of Qwen3's model card for either form (its top-k and presence penalty the page's sampler has not)
const thinking = { steps: 0, temperature: 0.6, topp: 0.95, repetition_penalty: 1.0 };
const atOnce = { steps: 0, temperature: 0.7, topp: 0.8, repetition_penalty: 1.0 };
/** A Qwen3 twice (T124, the owner's "両方を別々に用意できないのか"): thinking first, and answering at once. The two
 * share their weights, and so a conversion kept in the browser; only the format differs. shares: both ids, for
 * kept.js's replaced() (what either kept before its source changed goes, whichever form is opened first) */
function thinkingAndNot(id, name, source, download, sizes, chat = {}, formats = {}) {
  const common = { group: "hf", ...source, download, conversion: {}, options: {}, shares: [`${id}-thinking`, id],
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE, ...chat };
  return [
    // formats.thinking: where the converter cannot read the model's chat_template (T236), else what it reads
    { ...common, id: `${id}-thinking`, name: `${name} (thinking)`, note: `thinks before it answers · 日本語 / English · ${sizes}`,
      generation: thinking, ...(formats.thinking ? { template: formats.thinking } : {}) },
    { ...common, id, name: `${name} (no thinking)`, note: `answers at once · 日本語 / English · ${sizes}`,
      generation: chat.generation ?? atOnce, template: formats.atOnce ?? QWEN3_AT_ONCE },
  ];
}
// T236: Qwen3.5's chat_template calls a macro (render_content, for the pictures of a message), which the converter's
// reader refuses: one turn of text by hand, as the real Jinja writes it with enable_thinking true and without (the
// same IDs as transformers' apply_chat_template for tests/format_check.py's prompts). The template trims what was
// typed. Without a template the converter read, its special tokens are not in the converter's specials, and a list of
// the entry's replaces the converter's (T221): so all of the converter's are here (the added tokens tokenizer.json
// does not call special, T143) with <|im_start|> and <|im_end|>, in the converter's order (the longest first).
// The BOS: the real tokenizer begins a text with none (bos_token null), and the page begins every text with one, the
// converter's being <|endoftext|> (248044, T229). That one costs this model much on plain text (on 299 tokens of
// Wikipedia the perplexity is 46% higher in English and 95% in Japanese with it in front, and 20% to 65% on four texts
// of 512 tokens in the review, lasting to the end: its linear-attention layers keep what they read in a state, where a
// Qwen3's attention looks past it, T131's ±3%). In chat form it does not change how likely an answer written by hand
// is (tests/chat_nll.py: -0.8%, worse on 12 of 24) but it changes what the model writes: along its own answers the
// next-token distributions move by 0.12 nats a token (0.18 thinking), the most likely token at 13% (9%) of the
// positions. So the BOS here is the format's own first token, <|im_start|> (248045), and the formats begin after it:
// the page then sends the very IDs the real template makes, none more, and the engine agrees with transformers on
// the most likely token at 98.9% (99.6%) of the positions, 86.7% (91.4%) the old way (an entry opened with ?hf= has no
// format and the converter's BOS). The answer stops at <|im_end|> (248046), at <|endoftext|> (all that config.json
// names) and at the mark of a new turn
const QWEN35_THINKING = "user\n{prompt:trim}<|im_end|>\n<|im_start|>assistant\n<think>\n";
const QWEN35_AT_ONCE = `${QWEN35_THINKING}\n</think>\n\n`;
const qwen35 = { bos: 248045, stop_tokens: [248044, 248045, 248046],
  specials: ["</tool_response>", "<tool_response>", "<|fim_middle|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|repo_name|>",
    "</tool_call>", "<|file_sep|>", "<|im_start|>", "<tool_call>", "<|fim_pad|>", "<|im_end|>", "</think>", "<think>"] };
// T253: IBM's Granite 4.2. Its chat_template defines a macro (tool_to_json), which the converter's reader refuses:
// one turn by hand, as the real Jinja writes it with enable_thinking true (its default) and false, with the empty
// system turn it always writes (the same IDs as transformers' apply_chat_template for tests/format_check.py's
// prompts). As for a Qwen3.5 (T236): the real tokenizer begins a text with no BOS (its post-processor adds none, and
// the template does not write the <s> config.json names), so the BOS here is the format's own first token,
// <|im_start|> (100256), and the formats begin after it: the page sends the very IDs the real template makes. The
// specials are the converter's (the added tokens tokenizer.json does not call special, T143) with <|im_start|> and
// <|im_end|>, in the converter's order. The answer stops at <|im_end|> (100257, the EOS), at the mark of a new turn
// and at <s> (100283)
const GRANITE_THINKING = "system\n<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n";
const GRANITE_AT_ONCE = "system\n<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>";
const granite = { bos: 100256, stop_tokens: [100256, 100257, 100283],
  specials: ["</tool_response>", "<tool_response>", "<|fim_middle|>", "<|fim_prefix|>", "<|fim_suffix|>", "</tool_call>",
    "<|filename|>", "<|im_start|>", "<|reponame|>", "<tool_call>", "<|fim_pad|>", "<|im_end|>", "</think>", "<think>"] };
// its card: "Use temperature=1.0 and top_p=0.95 across all tasks", thinking or not
const graniteSampling = { steps: 0, temperature: 1.0, topp: 0.95, repetition_penalty: 1.0 };
/** T254: OpenBMB's MiniCPM5 (a Llama; English and Chinese), twice as a Qwen3 is: its chat_template begins the answer
 * with "<think>\n" where enable_thinking is true and with an empty thought where it is false (and with neither where
 * nothing is said, which is the format the converter reads). The real tokenizer begins every text with <s>, the
 * converter's BOS. sampling: its card's for either form */
function miniCpm5(id, name, source, download, sizes, sampling) {
  const common = { group: "hf", ...source, download, conversion: {}, options: {}, shares: [`${id}-thinking`, id],
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" };
  const sampled = (temperature) => ({ steps: 0, temperature, topp: 0.95, repetition_penalty: 1.0 });
  return [
    { ...common, id: `${id}-thinking`, name: `${name} (thinking)`, note: `thinks before it answers · English / 中文 · ${sizes}`,
      generation: sampled(sampling.thinking), template: `${CHATML}<think>\n` },
    { ...common, id, name: `${name} (no thinking)`, note: `answers at once · English / 中文 · ${sizes}`,
      generation: sampled(sampling.atOnce), template: QWEN3_AT_ONCE },
  ];
}
/** T203 (T136's fourth stage): a Q8_0 GGUF's weights with the vocabulary and config.json of its original, which
 * tests/gguf_check.py tensors held them to (gguf.yml's candidates) */
const ggufOf = (repo, revision, weights, original, originalRevision, tokenizer = "tokenizer.json") =>
  ({ original, hf: { repo, revision, weights, vocabulary: { repo: original, revision: originalRevision, tokenizer } } });
// A Qwen3 whose config.json and tokenizer name no BOS (Ternary Bonsai, CAT-Thinking 8B; a Qwen3 of Qwen's own has
// bos_token_id in config.json): the converter would take token 1, '"'. The BOS here is Qwen3's own, <|endoftext|>
// (151643), as every Qwen3 of the list begins (the real tokenizer puts nothing in front: T131), and the answer stops at
// it and at <|im_end|> (151645). The converter could say this itself (a BOS that is named nowhere, and <|endoftext|> in
// the vocabulary: T248's survey, 7 (4)), at the next CONVERTER: then these lose their options. What it costs is the
// model's: QWEN3_FROM_IM_START (above) says which are worse for it. Where a model of this family is added,
// tests/start_check.mjs and tests/answer_check.mjs say whether this BOS is one it can bear
const QWEN3_OWN_BOS = { bos: 151643, stop_tokens: [151643, 151645] };
/** T235, T246: a Ternary Bonsai of Prism ML (a ternary Qwen3) in one of its sizes: its PQ2_0 GGUF's weights with the
 * vocabulary, config.json and chat template of its -unpacked original. What the sizes share is here alone, so that one
 * change covers them all (the owner's open choices of T235's review: config.json's yarn or a plain RoPE, which would
 * be rope_scaling: {} in these options; a word more in the note; Prism ML's attribution). The original's template
 * always begins the answer with an empty thought (Qwen3's enable_thinking=false): the model has one form, and the
 * converter reads it. It names no BOS (QWEN3_OWN_BOS, which the 1.7B and the 4B bear: 1.5% better than nothing in front
 * on plain text; the 8B does not: 70% worse, as a Qwen3 8B is, and begins at QWEN3_FROM_IM_START: `start`, T250's review).
 * The sampling is the originals' generation_config.json, the same file in the three (its top-k the page's sampler
 * has not) */
const ternaryBonsai = (size, revision, originalRevision, download, sizes, start = {}) => ({
  group: "hf", id: `hf-ternary-bonsai-${size.toLowerCase()}`, name: `Ternary Bonsai ${size}`,
  note: `answers at once · 日本語 / English · ternary weights · ${sizes}`,
  ...ggufOf(`prism-ml/Ternary-Bonsai-${size}-gguf`, revision, `Ternary-Bonsai-${size}-PQ2_0.gguf`,
    `prism-ml/Ternary-Bonsai-${size}-unpacked`, originalRevision), download,
  weights: "ternary", conversion: {}, options: QWEN3_OWN_BOS, ...start,
  generation: { steps: 0, temperature: 0.5, topp: 0.85, repetition_penalty: 1.0 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE });
const harmony = { specials: ["<|channel|>", "<|message|>", "<|start|>", "<|end|>"], stop_tokens: [1, 2, 10, 11, 13] };
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
  "November", "December"];
const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
/** The directives of strftime the chat templates use (%d %b %Y is Llama 3's), for a date */
function strftime(format, date) {
  const two = (n) => String(n).padStart(2, "0");
  const values = { d: two(date.getDate()), m: two(date.getMonth() + 1), Y: String(date.getFullYear()), y: two(date.getFullYear() % 100),
    b: MONTHS[date.getMonth()].slice(0, 3), B: MONTHS[date.getMonth()], a: DAYS[date.getDay()].slice(0, 3), A: DAYS[date.getDay()],
    H: two(date.getHours()), M: two(date.getMinutes()), S: two(date.getSeconds()), "%": "%" };
  return format.replace(/%(.)/g, (directive, letter) => values[letter] ?? directive);
}
// Jinja's trim is Python's str.strip(): the white space of str.isspace() at either end. JavaScript's trim() takes
// another set, U+FEFF too and not U+001C to U+001F nor U+0085 (the review of T138, T144: a prompt with one of them at
// an end differed from the real template's). A loop from either end, not a pattern: [...]+$ tries again from every
// white space in the middle, and a prompt of 100,000 spaces between two words held the page 45 seconds (the review of
// T144). Every one of them is a single UTF-16 unit.
const SPACE = /[\t-\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/;
function strip(text) {
  let start = 0, end = text.length;
  while (start < end && SPACE.test(text[start])) start++;
  while (end > start && SPACE.test(text[end - 1])) end--;
  return text.slice(start, end);
}
/** What the page sends for a prompt in a model's template: {prompt} is what was typed, {prompt:trim} the same
 * without the white space at either end (T138: what a template that pipes the message through Jinja's trim writes;
 * the converter says so), {date} today (YYYY-MM-DD, the visitor's own day), and {date:format} today in strftime's
 * format (what the converter writes for a template's strftime_now(): the review of T127, the day of the conversion
 * was kept with it). What was typed goes in as it is: as a replacement string, its $$, $&, $` and $' were patterns
 * (a typed $' wrote the rest of the template, special tokens and all; the review of T132). */
export function filled(template, prompt, today = new Date()) {
  return template.replace(/\{date(?::([^}]*))?\}/g, (_, format = "%Y-%m-%d") => strftime(format, today))
    .replace(/\{prompt(:trim)?\}/, (_, trim) => (trim ? strip(prompt) : prompt));
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
// T250: ELYZA's card says "Meta Llama 3 Community License" under License (its metadata: llama3)
const LLAMA_3 = "Meta Llama 3 Community License";
// T251: the cards of GPT-2's medium, large and XL: "License: Modified MIT License" (OpenAI's, github.com/openai/gpt-2)
const MODIFIED_MIT = "Modified MIT License";
// T252: the cards of unsloth's copy and of bartowski's GGUF say llama3.1 (the license's own title: "Llama 3.1 Community
// License Agreement"), and DeepSeek's card says what its Llama distill was made from
const LLAMA_31 = "Llama 3.1 Community License";
const MIT_OF_LLAMA_31 = "MIT License (derived from Llama 3.1 8B, originally under the Llama 3.1 Community License)";
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
  // T249 (2026-10-01): the Japanese ones of T248's survey that open as they are, and the Q8_0 GGUFs they are taken
  // from (the card's license is the original's)
  "rinna/japanese-gpt2-xsmall": MIT, "rinna/japanese-gpt2-medium": MIT,
  "sbintuitions/sarashina2.2-1b": MIT, "mradermacher/sarashina2.2-1b-GGUF": MIT,
  "stockmark/gpt-neox-japanese-1.4b": MIT,
  "line-corporation/japanese-large-lm-1.7b": APACHE, "mmnga/line-corp-japanese-large-lm-1.7b-gguf": APACHE,
  "utter-project/EuroLLM-1.7B-Instruct": APACHE, "mradermacher/EuroLLM-1.7B-Instruct-GGUF": APACHE,
  "llm-jp/llm-jp-3-1.8b-instruct3": APACHE, "mmnga/llm-jp-3-1.8b-instruct3-gguf": APACHE,
  "llm-jp/llm-jp-3-3.7b-instruct3": APACHE, "mmnga/llm-jp-3-3.7b-instruct3-gguf": APACHE,
  "shisa-ai/shisa-v2.1-llama3.2-3b": LLAMA_32, "mradermacher/shisa-v2.1-llama3.2-3b-GGUF": LLAMA_32,
  "cyberagent/CAT-Translate-3.3b": MIT,
  // T250 (2026-10-01): the 8B ones
  "elyza/Llama-3-ELYZA-JP-8B": LLAMA_3, "mmnga/Llama-3-ELYZA-JP-8B-gguf": LLAMA_3,
  "shisa-ai/shisa-v2.1-qwen3-8b": APACHE, "mradermacher/shisa-v2.1-qwen3-8b-GGUF": APACHE,
  "cyberagent/CAT-Thinking-8B": APACHE, "mmnga-o/CAT-Thinking-8B-gguf": APACHE,
  // T251 (2026-10-02): the English classics and the Q8_0 GGUFs they are taken from. The cards of GPT-2's medium, large
  // and XL say "License: Modified MIT License" (their metadata, and their GGUFs': mit); DistilGPT2's and Pythia's Apache 2.0
  "distilbert/distilgpt2": APACHE, "QuantFactory/distilgpt2-GGUF": APACHE,
  "openai-community/gpt2-medium": MODIFIED_MIT, "mradermacher/gpt2-medium-GGUF": MODIFIED_MIT,
  "openai-community/gpt2-large": MODIFIED_MIT, "mradermacher/gpt2-large-GGUF": MODIFIED_MIT,
  "openai-community/gpt2-xl": MODIFIED_MIT, "mradermacher/gpt2-xl-GGUF": MODIFIED_MIT,
  "EleutherAI/pythia-14m-deduped": APACHE, "mradermacher/pythia-14m-GGUF": APACHE,
  "EleutherAI/pythia-2.8b": APACHE, "mradermacher/pythia-2.8b-GGUF": APACHE,
  "EleutherAI/pythia-6.9b": APACHE, "mradermacher/pythia-6.9b-GGUF": APACHE,
  // T252 (2026-10-02): well-known ones, and the Q8_0 GGUFs they are taken from. Qwen2.5-Coder-3B's card names the
  // license of Qwen2.5-3B (license_name: qwen-research; its LICENSE: "Qwen RESEARCH LICENSE AGREEMENT"). Hermes 3's
  // card says llama3 (its base model is Llama 3.2 3B). DeepSeek's card, under License: "the model weights are licensed
  // under the MIT License ... DeepSeek-R1-Distill-Llama-8B is derived from Llama3.1-8B-Base and is originally licensed
  // under llama3.1 license" (its metadata, and its GGUF's: mit)
  "HuggingFaceTB/SmolLM2-135M": APACHE, "mradermacher/SmolLM2-135M-GGUF": APACHE,
  "HuggingFaceTB/SmolLM2-360M": APACHE, "mradermacher/SmolLM2-360M-GGUF": APACHE,
  "Qwen/Qwen2.5-Coder-1.5B-Instruct": APACHE, "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF": APACHE,
  "Qwen/Qwen2.5-Coder-3B-Instruct": QWEN_RESEARCH, "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF": QWEN_RESEARCH,
  "Qwen/Qwen2.5-Coder-7B-Instruct": APACHE, "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF": APACHE,
  "NousResearch/Hermes-3-Llama-3.2-3B": LLAMA_3, "NousResearch/Hermes-3-Llama-3.2-3B-GGUF": LLAMA_3,
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": MIT, "mradermacher/DeepSeek-R1-Distill-Qwen-7B-GGUF": MIT,
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": MIT_OF_LLAMA_31, "mradermacher/DeepSeek-R1-Distill-Llama-8B-GGUF": MIT_OF_LLAMA_31,
  "meta-llama/Llama-3.1-8B-Instruct": LLAMA_31, "unsloth/Meta-Llama-3.1-8B-Instruct": LLAMA_31,
  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF": LLAMA_31,
  // T125
  "Rakuten/RakutenAI-2.0-mini-instruct": APACHE, "Rakuten/RakutenAI-7B-chat": APACHE,
  "tokyotech-llm/Swallow-MS-7b-instruct-v0.1": APACHE, "mistralai/Mistral-7B-Instruct-v0.2": APACHE,
  "mistralai/Mistral-7B-Instruct-v0.3": APACHE, "HuggingFaceH4/zephyr-7b-beta": MIT,
  "meta-llama/Llama-3.2-3B-Instruct": LLAMA_32, "unsloth/Llama-3.2-3B-Instruct": LLAMA_32,
  // T124
  "Qwen/Qwen3-0.6B": APACHE, "Qwen/Qwen3-1.7B": APACHE, "Qwen/Qwen3-4B": APACHE, "Qwen/Qwen3-8B": APACHE,
  "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2": APACHE, "Qwen/Qwen3-4B-Instruct-2507": APACHE,
  "Qwen/Qwen3-4B-Thinking-2507": APACHE,
  // T136 (2026-09-26): the Q8_0 GGUFs the list fetches instead of the safetensors (tests/gguf_check.py tensors passed
  // against the originals), under the license of each card (TinySwallow's GGUF has the words of its original)
  "bartowski/Qwen2.5-0.5B-Instruct-GGUF": APACHE, "bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF": APACHE,
  "bartowski/Qwen2.5-1.5B-Instruct-GGUF": APACHE, "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF": APACHE,
  "SakanaAI/TinySwallow-1.5B-Instruct-GGUF": APACHE_GEMMA, "bartowski/SmolLM2-1.7B-Instruct-GGUF": APACHE,
  "bartowski/Qwen2.5-3B-Instruct-GGUF": QWEN_RESEARCH, "bartowski/Qwen2.5-7B-Instruct-GGUF": APACHE,
  // T136's second stage: the GGUF each takes its weights from (the card's license is the original's)
  "mradermacher/sarashina2.2-0.5b-GGUF": MIT, "mmnga/sarashina2.2-0.5b-instruct-v0.1-gguf": MIT,
  "mmnga/sarashina2.2-1b-instruct-v0.1-gguf": MIT, "mmnga/sarashina2.2-3b-instruct-v0.1-gguf": MIT,
  "mmnga-o/CAT-Translate-0.8b-gguf": MIT, "mmnga-o/CAT-Translate-1.4b-gguf": MIT,
  "mmnga/llm-jp-3-980m-instruct3-gguf": APACHE, "mmnga/llm-jp-3.1-1.8b-instruct4-gguf": APACHE,
  "mmnga-o/llm-jp-4-8b-instruct-gguf": APACHE, "mmnga/RakutenAI-2.0-mini-instruct-gguf": APACHE,
  "mradermacher/RakutenAI-7B-chat-GGUF": APACHE, "mmnga/tokyotech-llm-Swallow-MS-7b-instruct-v0.1-gguf": APACHE,
  "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF": APACHE, "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": APACHE,
  "bartowski/Mistral-7B-Instruct-v0.3-GGUF": APACHE, "TheBloke/zephyr-7B-beta-GGUF": MIT,
  "bartowski/Llama-3.2-1B-Instruct-GGUF": LLAMA_32, "bartowski/Llama-3.2-3B-Instruct-GGUF": LLAMA_32,
  "mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf": SWALLOW,
  // T136's third stage: GPT-2 and GPT-NeoX (the card's license is the original's)
  "mradermacher/pythia-70m-deduped-GGUF": APACHE, "mradermacher/pythia-160m-GGUF": APACHE,
  "mradermacher/pythia-410m-GGUF": APACHE, "mradermacher/pythia-1b-GGUF": APACHE, "mradermacher/pythia-1.4b-GGUF": APACHE,
  "mradermacher/gpt2-GGUF": MIT,
  // T203 (T136's fourth stage): Qwen3 and DeepSeek-R1 (the card's license is the original's)
  "unsloth/Qwen3-0.6B-GGUF": APACHE, "unsloth/Qwen3-1.7B-GGUF": APACHE, "Qwen/Qwen3-4B-GGUF": APACHE,
  "Qwen/Qwen3-8B-GGUF": APACHE, "mmnga-o/Qwen3-Swallow-8B-RL-v0.2-gguf": APACHE,
  "unsloth/Qwen3-4B-Instruct-2507-GGUF": APACHE, "unsloth/Qwen3-4B-Thinking-2507-GGUF": APACHE,
  "mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-GGUF": MIT,
  // T235: both cards say apache-2.0. Their NOTICE.txt: "copyright 2026-present Prism ML, Inc. ... built from Qwen3-1.7B,
  // Copyright 2024 Alibaba Cloud ... Apache 2.0", and asks for "Created using Bonsai by Prism ML." where it is deployed
  "prism-ml/Ternary-Bonsai-1.7B-gguf": APACHE, "prism-ml/Ternary-Bonsai-1.7B-unpacked": APACHE,
  // T246: the same of the 4B's and the 8B's cards and NOTICE.txt ("built from Qwen3-4B", "from Qwen3-8B")
  "prism-ml/Ternary-Bonsai-4B-gguf": APACHE, "prism-ml/Ternary-Bonsai-4B-unpacked": APACHE,
  "prism-ml/Ternary-Bonsai-8B-gguf": APACHE, "prism-ml/Ternary-Bonsai-8B-unpacked": APACHE,
  // T236: both cards say apache-2.0 (the GGUF's names the original's LICENSE as its license_link)
  "Qwen/Qwen3.5-0.8B": APACHE, "unsloth/Qwen3.5-0.8B-GGUF": APACHE,
  // T247: the same of the other sizes' cards
  "Qwen/Qwen3.5-2B": APACHE, "unsloth/Qwen3.5-2B-GGUF": APACHE,
  "Qwen/Qwen3.5-4B": APACHE, "unsloth/Qwen3.5-4B-GGUF": APACHE,
  "Qwen/Qwen3.5-9B": APACHE, "unsloth/Qwen3.5-9B-GGUF": APACHE,
  // T253: the four cards say apache-2.0
  "ibm-granite/granite-4.2-3b": APACHE, "ibm-granite/granite-4.2-3b-GGUF": APACHE,
  "ibm-granite/granite-4.2-8b": APACHE, "ibm-granite/granite-4.2-8b-GGUF": APACHE,
  // T254: the four cards say apache-2.0
  "openbmb/MiniCPM5-1B": APACHE, "openbmb/MiniCPM5-1B-GGUF": APACHE,
  "openbmb/MiniCPM5-2B": APACHE, "openbmb/MiniCPM5-2B-GGUF": APACHE,
};
/** The Hugging Face repository a model comes from. */
export const sourceOf = (entry) => entry.hf?.repo ?? entry.source;
/** Every source once, in the order of the list, with its license and the names of the models taken from it. A
 * model fetched from a redistribution names both: where it comes from, and whose model it is. The redistribution's
 * line says which it is: "(GGUF)" for a GGUF (T74), "(copy)" for the same safetensors elsewhere (unsloth's Llama;
 * until 2026-09-26 it said "(GGUF)" for those too). A GGUF with a copy's vocabulary (T136: Llama 3.2's) is on all three. */
export function sources(models = MODELS) {
  const bySource = new Map();
  for (const entry of models) {
    // T136: the repository the vocabulary and config.json come from, where it is neither (unsloth's copy of Llama)
    for (const repo of new Set([entry.original, entry.hf?.vocabulary?.repo, sourceOf(entry)].filter(Boolean))) {
      if (!bySource.has(repo)) bySource.set(repo, { repo, license: LICENSES[repo], names: [] });
      const kind = repo !== sourceOf(entry) ? "copy" : entry.hf?.weights?.endsWith(".gguf") ? "GGUF" : "copy";
      bySource.get(repo).names.push(entry.original && repo !== entry.original ? `${entry.name} (${kind})` : entry.name);
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
    options: { dtype: "int8", ...unigram, stop_tokens: [1, 2] },
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
    options: { dtype: "float16", ...unigram, stop_tokens: [1, 2] },
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
  // T249: rinna's GPT-2 below and above the small one. No GGUF of either was found: the safetensors
  { group: "hf", id: "hf-japanese-gpt2-xsmall", name: "japanese-gpt2 xsmall", note: "日本語 · fetches 156 MB → int8 42 MB",
    hf: hf("rinna/japanese-gpt2-xsmall", "8e91527b3276e0565154935e84a08bf0137ed99f", "spiece.model"), download: 155892312,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-gpt2-medium", name: "japanese-gpt2 medium", note: "日本語 · fetches 1.4 GB → int8 379 MB",
    hf: hf("rinna/japanese-gpt2-medium", "8ce2399c33e99013a593ea9389378fd86662b9c7", "spiece.model"), download: 1369713080,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-gpt2-small", name: "japanese-gpt2 small", note: "日本語 · fetches 454 MB → int8 130 MB",
    hf: hf("rinna/japanese-gpt2-small", "f7fdefe2941d9629a7b2894564435e0e035df6a6", "spiece.model"), download: 454274094,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-gpt-neox-small", name: "japanese-gpt-neox small", note: "日本語 · fetches 663 MB → int8 193 MB",
    hf: hf("rinna/japanese-gpt-neox-small", "84d18c0fa8c9940a61cfc4e25bd9a5686898bac1", "spiece.model"), download: 663470088,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-qwen2.5-0.5b-instruct", name: "Qwen2.5 0.5B Instruct", note: "answers instructions · 日本語 / English · fetches 531 MB (GGUF) → int8 545 MB",
    original: "Qwen/Qwen2.5-0.5B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-0.5B-Instruct-GGUF", revision: "41ba88dbac95fed2528c92514c131d73eb5a174b", weights: "Qwen2.5-0.5B-Instruct-Q8_0.gguf" }, download: 531068480,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m", name: "llm-jp-3 440M", note: "日本語 / English · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m", "0bfbf24efdcc5e4c57327e9c52e8cd832637adc2"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m-instruct3", name: "llm-jp-3 440M instruct3", note: "answers instructions · 日本語 · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m-instruct3", "a308f143bd5824c4033b3a2efaa1c00afbb3aa9e"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // sarashina2.2 (T79): the instruct one is published as a single shard with an index (T78)
  { group: "hf", id: "hf-sarashina2.2-0.5b", name: "sarashina2.2 0.5B", note: "日本語 · fetches 845 MB (GGUF) → int8 0.9 GB · desktop only",
    original: "sbintuitions/sarashina2.2-0.5b",
    hf: { repo: "mradermacher/sarashina2.2-0.5b-GGUF", revision: "2aed15b94f8c25b7582369f62d9801f3524e4721", weights: "sarashina2.2-0.5b.Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-0.5b", revision: "5fb086c49f49824cfc93f09cc4ed5cd5917bef3d", tokenizer: "tokenizer.model" } },
    download: 845360480, conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-0.5b-instruct", name: "sarashina2.2 0.5B Instruct", note: "answers instructions · 日本語 · fetches 845 MB (GGUF) → int8 0.9 GB · desktop only",
    original: "sbintuitions/sarashina2.2-0.5b-instruct-v0.1",
    hf: { repo: "mmnga/sarashina2.2-0.5b-instruct-v0.1-gguf", revision: "5c71186a7a57b8c0325dec70ba2f295303effcd8", weights: "sarashina2.2-0.5b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-0.5b-instruct-v0.1", revision: "e4b9aacc3f644893d0179847946ef6c58d868f29", tokenizer: "tokenizer.model" } },
    download: 845363072, conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T81: a translator from sarashina2.2, Japanese to English and back, asked in its own words and greedy, as its
  // model card runs it (no generation_config: transformers' defaults; a repetition penalty bends a translation)
  { group: "hf", id: "hf-cat-translate-0.8b", name: "CAT-Translate 0.8B", note: "translates 日本語 ⇄ English · fetches 845 MB (GGUF) → int8 0.9 GB · desktop only",
    original: "cyberagent/CAT-Translate-0.8b",
    hf: { repo: "mmnga-o/CAT-Translate-0.8b-gguf", revision: "c770a1944b55223ccbe766c770882c30c9866445", weights: "CAT-Translate-0.8b-Q8_0.gguf",
          vocabulary: { repo: "cyberagent/CAT-Translate-0.8b", revision: "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9", tokenizer: "tokenizer.model" } }, download: 845363136,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  { group: "hf", id: "hf-llm-jp-3-980m-instruct3", name: "llm-jp-3 980M instruct3", note: "answers instructions · 日本語 · fetches 1.1 GB (GGUF) → int8 1.1 GB · desktop only",
    original: "llm-jp/llm-jp-3-980m-instruct3",
    hf: { repo: "mmnga/llm-jp-3-980m-instruct3-gguf", revision: "5966bc9958a7d313da9b8cb4679cfdf6814b6d42", weights: "llm-jp-3-980m-instruct3-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-3-980m-instruct3", revision: "c079dbf3f88aa2ab702b9696231fc3336c46b1be", tokenizer: "tokenizer.json" } }, download: 1054638240, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T126: a GPT-2 of 1.3B, whose activation is written gelu_fast (the same tanh approximation as gelu_new). Its
  // table of positions holds 1024, which is its context
  { group: "hf", id: "hf-japanese-gpt-1b", name: "japanese-gpt 1B", note: "日本語 · fetches 2.7 GB → int8 1.5 GB · desktop only",
    hf: hf("rinna/japanese-gpt-1b", "33fc2e4b4e97e229d24a2973073a1361157ecef6", "spiece.model"), download: 2655791788,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T81 (2026-09-26): the survey's Japanese models that convert as they are. TinySwallow reads its chat template
  // itself (T73); llm-jp-3.1 has its family's format with the system sentence its model card always passes (its
  // template alone leaves it out)
  { group: "hf", id: "hf-sarashina2.2-1b-instruct", name: "sarashina2.2 1B Instruct", note: "answers instructions · 日本語 · fetches 1.5 GB (GGUF) → int8 1.6 GB · desktop only",
    original: "sbintuitions/sarashina2.2-1b-instruct-v0.1",
    hf: { repo: "mmnga/sarashina2.2-1b-instruct-v0.1-gguf", revision: "cd8b02dcc14f38a5101171f304ab269d514fafd4", weights: "sarashina2.2-1b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-1b-instruct-v0.1", revision: "08cf5a8ae579be0fb5a9f802dda8a26acbc94951", tokenizer: "tokenizer.model" } }, download: 1498333056,
    conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-cat-translate-1.4b", name: "CAT-Translate 1.4B", note: "translates 日本語 ⇄ English · fetches 1.5 GB (GGUF) → int8 1.6 GB · desktop only",
    original: "cyberagent/CAT-Translate-1.4b",
    hf: { repo: "mmnga-o/CAT-Translate-1.4b-gguf", revision: "fe0ab30f267cd976a4f03d218defa6151af88b87", weights: "CAT-Translate-1.4b-Q8_0.gguf",
          vocabulary: { repo: "cyberagent/CAT-Translate-1.4b", revision: "254120945fd9a61278ac2171ab07c831d56838fa", tokenizer: "tokenizer.model" } }, download: 1498333088,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  // T249: the base model of sarashina2.2 1B, and two Japanese base models of 2023. stockmark's GPT-NeoX rotates a
  // quarter of each head and has no parallel residual; mmnga's Q8_0 GGUF of it (2023) holds the very values, but with
  // q, k and v of a head in turns as Hugging Face has them, where today's llama.cpp writes all of q, then k, then v,
  // which is what the converter undoes (T136): read so, it wrote "のののの", so the safetensors. LINE's GPT-2, whose
  // table of positions holds 2048: its tokenizer puts nothing in front of a text, and the model writes nonsense after
  // <s> (1), the converter's BOS, and after nothing; after </s> (2) it writes Japanese (transformers on the original
  // says the same of all three), so that is the BOS here
  { group: "hf", id: "hf-sarashina2.2-1b", name: "sarashina2.2 1B", note: "日本語 · fetches 1.5 GB (GGUF) → int8 1.6 GB · desktop only",
    ...ggufOf("mradermacher/sarashina2.2-1b-GGUF", "9eaeb885b7b61d8ceb274bac21b9df4f42151e23", "sarashina2.2-1b.Q8_0.gguf",
      "sbintuitions/sarashina2.2-1b", "3bb836ad7475ba192926be66651e4730825df7da", "tokenizer.model"), download: 1498330464,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-gpt-neox-japanese-1.4b", name: "gpt-neox-japanese 1.4B", note: "日本語 · fetches 2.9 GB → int8 1.6 GB · desktop only",
    hf: hf("stockmark/gpt-neox-japanese-1.4b", "c8f1288a46ac11cf4445dfd18147605d9b692261"), download: 2852015168,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-large-lm-1.7b", name: "japanese-large-lm 1.7B", note: "日本語 · fetches 1.9 GB (GGUF) → int8 1.9 GB · desktop only",
    ...ggufOf("mmnga/line-corp-japanese-large-lm-1.7b-gguf", "d49108e627b7b6b7b6977184a6b81045b60a8582", "line-corp-japanese-large-lm-1.7b-q8_0.gguf",
      "line-corporation/japanese-large-lm-1.7b", "4288da0a536789f0615c730af0c6cbd9e475a7db", "spiece.model"), download: 1888727168,
    conversion: {}, options: { bos: 2 }, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T249: EuroLLM, of 35 languages with Japanese among them. Its tokenizer.json is a BPE of sentencepiece's kind, so
  // the tokenizer.model; the special tokens of its format are the converter's (it reads the template too)
  { group: "hf", id: "hf-eurollm-1.7b-instruct", name: "EuroLLM 1.7B Instruct", note: "answers instructions · 日本語 / English and 33 more languages · fetches 1.8 GB (GGUF) → int8 1.9 GB · desktop only",
    ...ggufOf("mradermacher/EuroLLM-1.7B-Instruct-GGUF", "2951f08f66429c934c8b01a94347161362430808", "EuroLLM-1.7B-Instruct.Q8_0.gguf",
      "utter-project/EuroLLM-1.7B-Instruct", "a25c7fa65fc2a644e6270b8940dbe295b51da681", "tokenizer.model"), download: 1763775712,
    conversion: {}, options: {}, generation: sampled(1.1), template: EUROLLM, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-1.8b-instruct3", name: "llm-jp-3 1.8B instruct3", note: "answers instructions · 日本語 · fetches 2.0 GB (GGUF) → int8 2.1 GB · desktop only",
    ...ggufOf("mmnga/llm-jp-3-1.8b-instruct3-gguf", "d908906be3bed7681e4d7269f5c441ea91d2fd56", "llm-jp-3-1.8b-instruct3-Q8_0.gguf",
      "llm-jp/llm-jp-3-1.8b-instruct3", "6b9b0bf051699e7ecffaa5e1166aa5008aa6534f"), download: 1987023136,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-tinyswallow-1.5b-instruct", name: "TinySwallow 1.5B Instruct", note: "answers instructions · 日本語 · fetches 1.6 GB (GGUF) → int8 1.7 GB · desktop only",
    original: "SakanaAI/TinySwallow-1.5B-Instruct",
    hf: { repo: "SakanaAI/TinySwallow-1.5B-Instruct-GGUF", revision: "38c003aaf8be9d17af11dece1fbabeb873c567fa", weights: "tinyswallow-1.5b-instruct-q8_0.gguf" }, download: 1646573920,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T125: a Mistral (a Llama by another name) of 1.5B, Japanese and English; its sliding window of 8192 is past the
  // context of 4096 the page gives it
  { group: "hf", id: "hf-rakutenai-2.0-mini-instruct", name: "RakutenAI 2.0 mini instruct", note: "answers instructions · 日本語 / English · fetches 1.6 GB (GGUF) → int8 1.7 GB · desktop only",
    original: "Rakuten/RakutenAI-2.0-mini-instruct",
    hf: { repo: "mmnga/RakutenAI-2.0-mini-instruct-gguf", revision: "9bd2900dd7bff11c248fb510d4297b3a88817b76", weights: "RakutenAI-2.0-mini-instruct-Q8_0.gguf",
          vocabulary: { repo: "Rakuten/RakutenAI-2.0-mini-instruct", revision: "6d902489587d324b7d5e201299e4e1a169f3a40b", tokenizer: "tokenizer.model" } }, download: 1631976352,
    conversion: {}, options: {}, generation: sampled(1.1), template: RAKUTEN,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3.1-1.8b-instruct4", name: "llm-jp-3.1 1.8B instruct4", note: "answers instructions · 日本語 · fetches 2.0 GB (GGUF) → int8 2.1 GB · desktop only",
    original: "llm-jp/llm-jp-3.1-1.8b-instruct4",
    hf: { repo: "mmnga/llm-jp-3.1-1.8b-instruct4-gguf", revision: "14ddbab20c68d8befdced79bd9daa3d7a1a29376", weights: "llm-jp-3.1-1.8b-instruct4-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-3.1-1.8b-instruct4", revision: "f19510db409090bb1737f24f868d17c4bdc86c8e", tokenizer: "tokenizer.json" } }, download: 1987023136,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T132: the large ones, in shards (T105). Past 4 GiB with their forward pass they need a 64-bit memory (T101),
  // where the browser has one int8 (T133), else six bits (T98); the 7 to 8B ones do not fit a 32-bit memory even so
  { group: "hf", id: "hf-qwen2.5-3b-instruct", name: "Qwen2.5 3B Instruct", note: "answers instructions · 日本語 / English · fetches 3.3 GB (GGUF) → int8 3.5 GB · desktop only",
    original: "Qwen/Qwen2.5-3B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-3B-Instruct-GGUF", revision: "f302c64a2269a69fb27b2f9473b362f5bb8e78d8", weights: "Qwen2.5-3B-Instruct-Q8_0.gguf" }, download: 3285476512,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-3b-instruct", name: "sarashina2.2 3B Instruct", note: "answers instructions · 日本語 · fetches 3.6 GB (GGUF) → int8 3.8 GB · desktop only",
    original: "sbintuitions/sarashina2.2-3b-instruct-v0.1",
    hf: { repo: "mmnga/sarashina2.2-3b-instruct-v0.1-gguf", revision: "31d771319b04032f33e0d9d860f3984ea4812154", weights: "sarashina2.2-3b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-3b-instruct-v0.1", revision: "4f3626fb1b64b3e97c908e67f27b2d627ba2a999", tokenizer: "tokenizer.model" } }, download: 3568393312,
    conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T249: the middle ones. Shisa V2.1's Llama 3.2 3B reads its chat template itself, as Llama 3.2 does (today's date
  // in the system turn); its card gives the model this name (the Llama license asks for it). CAT-Translate 3.3B is
  // the 0.8B's and 1.4B's larger sibling, from sarashina2.2 3B: no Q8_0 GGUF of it was found (only of its beta), so
  // the safetensors, in two shards
  { group: "hf", id: "hf-shisa-v2.1-llama3.2-3b", name: "Llama 3.2 Shisa V2.1 3B", note: "answers instructions · 日本語 / English · fetches 3.4 GB (GGUF) → int8 3.6 GB · desktop only",
    ...ggufOf("mradermacher/shisa-v2.1-llama3.2-3b-GGUF", "b8cb9e4b9c90657829c0547f091e591a3d24849c", "shisa-v2.1-llama3.2-3b.Q8_0.gguf",
      "shisa-ai/shisa-v2.1-llama3.2-3b", "5f4f59bbe65834daf86a38efd06ff96f7c94c8c3"), download: 3421900096,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-cat-translate-3.3b", name: "CAT-Translate 3.3B", note: "translates 日本語 ⇄ English · fetches 6.7 GB → int8 3.8 GB · desktop only",
    hf: hf("cyberagent/CAT-Translate-3.3b", "47e382331d005acd54a42cdf088a76aa88788e0c", "tokenizer.model"), download: 6711252920,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  { group: "hf", id: "hf-llm-jp-3-3.7b-instruct3", name: "llm-jp-3 3.7B instruct3", note: "answers instructions · 日本語 · fetches 4.0 GB (GGUF) → int8 4.3 GB · desktop only · Chrome and Firefox",
    ...ggufOf("mmnga/llm-jp-3-3.7b-instruct3-gguf", "7edef5a4f094ec8c1aed1e196c6a544675efbc2f", "llm-jp-3-3.7b-instruct3-Q8_0.gguf",
      "llm-jp/llm-jp-3-3.7b-instruct3", "f5d5466a3316e0c898b4347ece6557a756921220"), download: 4022249856,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-qwen2.5-7b-instruct", name: "Qwen2.5 7B Instruct", note: "answers instructions · 日本語 / English · fetches 8.1 GB (GGUF) → int8 8.6 GB · desktop only · Chrome and Firefox",
    original: "Qwen/Qwen2.5-7B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-7B-Instruct-GGUF", revision: "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031", weights: "Qwen2.5-7B-Instruct-Q8_0.gguf" }, download: 8098525888,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T125: Mistral 7B's of Japanese, whose sliding window of 4096 is the page's context
  { group: "hf", id: "hf-rakutenai-7b-chat", name: "RakutenAI 7B chat", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.3 GB · desktop only · Chrome and Firefox",
    original: "Rakuten/RakutenAI-7B-chat",
    hf: { repo: "mradermacher/RakutenAI-7B-chat-GGUF", revision: "d118d842e84cd542a48c6a89464d95b98d37e4e7", weights: "RakutenAI-7B-chat.Q8_0.gguf",
          vocabulary: { repo: "Rakuten/RakutenAI-7B-chat", revision: "7093167c61a0be6161cb68928c939c03fe0ab87d", tokenizer: "tokenizer.model" } }, download: 7835496736,
    conversion: {}, options: {}, generation: sampled(1.1), template: RAKUTEN,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-swallow-ms-7b-instruct", name: "Swallow-MS 7B instruct", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.3 GB · desktop only · Chrome and Firefox",
    original: "tokyotech-llm/Swallow-MS-7b-instruct-v0.1",
    hf: { repo: "mmnga/tokyotech-llm-Swallow-MS-7b-instruct-v0.1-gguf", revision: "cb9ab1c831cfdb8f56371f0ba94806df0fabd518", weights: "tokyotech-llm-Swallow-MS-7b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "tokyotech-llm/Swallow-MS-7b-instruct-v0.1", revision: "008d006f9065e37e39e31bf117ae8689390953e8", tokenizer: "tokenizer.model" } }, download: 7790109536,
    conversion: {}, options: {}, generation: sampled(1.1), template: SWALLOW_MS,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // Swallow's own chat_template is read (T73): a Japanese system message, and a second BOS before the user's turn,
  // as the real Jinja writes it (the same IDs as the real Jinja and tokenizers, T132)
  { group: "hf", id: "hf-llama-3.1-swallow-8b-instruct", name: "Llama 3.1 Swallow 8B Instruct", note: "answers instructions · 日本語 / English · fetches 8.5 GB (GGUF) → int8 9.0 GB · desktop only · Chrome and Firefox",
    original: "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
    hf: { repo: "mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf", revision: "dc00f584312c641eb7af415f7f44bcc4487350cb", weights: "Llama-3.1-Swallow-8B-Instruct-v0.5-Q8_0.gguf",
          vocabulary: { repo: "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5", revision: "b1f8317099a97e790ec872c1225ca155979b4816", tokenizer: "tokenizer.json" } }, download: 8540772672,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-4-8b-instruct", name: "llm-jp-4 8B instruct", note: "answers instructions · 日本語 / English · fetches 9.1 GB (GGUF) → int8 9.7 GB · desktop only · Chrome and Firefox",
    original: "llm-jp/llm-jp-4-8b-instruct",
    hf: { repo: "mmnga-o/llm-jp-4-8b-instruct-gguf", revision: "7ae4da12cee2f109509cb8e1d01cf8a0f1a5fbc1", weights: "llm-jp-4-8b-instruct-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-4-8b-instruct", revision: "098f2b2cf33021eba19a6d3582aa3d071ccc0aff", tokenizer: "tokenizer.json" } }, download: 9132708384,
    conversion: {}, options: harmony, generation: sampled(1.1), template: HARMONY,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T250: three more of 8B. ELYZA's Llama 3 with its card's system message. Shisa V2.1's Qwen3 8B, whose template
  // answers at once unless told to think (the other way round from Qwen3's); CAT-Thinking, from Qwen3 Swallow, which
  // thinks in Japanese before it answers (Qwen3's template). Neither of the two names a BOS. CAT-Thinking's options are
  // QWEN3_OWN_BOS and its format the converter's reading of the template. Shisa's sampling is its generation_config.json's;
  // CAT-Thinking's is its card's (0.8 and 0.95, and "to mitigate the probability of repetition, we find
  // repetition_penalty=1.05 or larger to be useful").
  // Shisa's BOS is QWEN3_FROM_IM_START's: with <|endoftext|> in front its own answers to five questions were 67% higher in
  // perplexity, the likeliest next token differed at one in four, and all five answers written again parted within the
  // first 12 tokens (the first hedged where the real IDs answer); plain text 54% higher (the review, tests/answer_check.mjs
  // and tests/start_check.mjs).
  // CAT-Thinking's GGUF is mmnga-o's: mradermacher's Q8_0 has 256 tensors 0.1 to 0.4% from the nearest of the original's
  // (tests/gguf_check.py tensors: not the pinned original's weights). The original was uploaded in float32 on 2026-05-28
  // and "converted to bf16 from float32" on 2026-05-29 (the pinned revision holds the bf16): mradermacher's GGUF is of
  // 2026-05-28, mmnga-o's of 2026-06-02, which is llama.cpp's Q8_0 of the pinned weights (0 off the nearest reference)
  { group: "hf", id: "hf-llama-3-elyza-jp-8b", name: "Llama-3-ELYZA-JP 8B", note: "answers instructions · 日本語 / English · fetches 8.5 GB (GGUF) → int8 9.0 GB · desktop only · Chrome and Firefox",
    ...ggufOf("mmnga/Llama-3-ELYZA-JP-8B-gguf", "1a5f8f625074ccb91568fa858402dc43c5170856", "Llama-3-ELYZA-JP-8B-Q8_0.gguf",
      "elyza/Llama-3-ELYZA-JP-8B", "e6c316496ee7d9a11710c50229e8cb39b6b0a4a3"), download: 8540770592,
    conversion: {}, options: {}, generation: sampled(1.1), template: ELYZA, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-shisa-v2.1-qwen3-8b", name: "Shisa V2.1 Qwen3 8B", note: "answers at once · 日本語 / English · fetches 8.7 GB (GGUF) → int8 9.2 GB · desktop only · Chrome and Firefox",
    ...ggufOf("mradermacher/shisa-v2.1-qwen3-8b-GGUF", "9b9187f69adca28b8e2b9490b2c151fcb85c0df6", "shisa-v2.1-qwen3-8b.Q8_0.gguf",
      "shisa-ai/shisa-v2.1-qwen3-8b", "0b0fe7c76dac910510ccd04fc807fdbdbc2fc16e"), download: 8709519392,
    conversion: {}, options: QWEN3_FROM_IM_START, template: QWEN3_AT_ONCE_AFTER_START,
    generation: { steps: 0, temperature: 0.6, topp: 0.95, repetition_penalty: 1.0 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-cat-thinking-8b", name: "CAT-Thinking 8B", note: "thinks in Japanese before it answers · 日本語 / English · fetches 8.7 GB (GGUF) → int8 9.2 GB · desktop only · Chrome and Firefox",
    ...ggufOf("mmnga-o/CAT-Thinking-8B-gguf", "d1747e658749aa7a67858914f0a60a2364172c2b", "CAT-Thinking-8B-Q8_0.gguf",
      "cyberagent/CAT-Thinking-8B", "0337f7bcf8d5e6dc08610e205bfe01d566e17669"), download: 8709518944,
    conversion: {}, options: QWEN3_OWN_BOS,
    generation: { steps: 0, temperature: 0.8, topp: 0.95, repetition_penalty: 1.05 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // English. Pythia is the same design at five sizes: a ladder for measuring (T80)
  { group: "hf", id: "hf-pythia-70m", name: "Pythia 70M", note: "English · fetches 77 MB (GGUF) → int8 96 MB",
    original: "EleutherAI/pythia-70m-deduped",
    hf: { repo: "mradermacher/pythia-70m-deduped-GGUF", revision: "de94e2866e79534ca28e66593069e941988ac217", weights: "pythia-70m-deduped.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-70m-deduped", revision: "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c", tokenizer: "tokenizer.json" } },
    download: 76722400,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  // T74: from a GGUF (Q8_0): 145 MB instead of the 269 MB of model.safetensors, and the same int8 in the end (99.7%
  // of the most likely tokens and 0.13% of perplexity, tests/gguf_check.py). The GGUF is a redistribution; the
  // model and its license are HuggingFaceTB's (`original`). Its BOS is <|im_start|> itself (bos_token_id 1), which the page
  // begins every text with: the format begins after it, where CHATML here made two (T250's review, tests/format_check.py's
  // same_ids(): its own answers to five questions were 5.5% higher in perplexity, the likeliest next token differed at 9 in 100)
  { group: "hf", id: "hf-smollm2-135m-instruct", name: "SmolLM2 135M Instruct", note: "answers instructions · English · fetches 145 MB (GGUF) → int8 145 MB",
    original: "HuggingFaceTB/SmolLM2-135M-Instruct",
    hf: { repo: "bartowski/SmolLM2-135M-Instruct-GGUF", revision: "09816acd5d99df7be770d85ea30822623dab342c",
          weights: "SmolLM2-135M-Instruct-Q8_0.gguf" }, download: 144811360,
    conversion: {}, options: chatml, generation: sampled(1.1), template: CHATML_AFTER_START,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-pythia-160m", name: "Pythia 160M", note: "English · fetches 175 MB (GGUF) → int8 213 MB",
    original: "EleutherAI/pythia-160m",
    hf: { repo: "mradermacher/pythia-160m-GGUF", revision: "7a79d8f693376ce9f1ba21e2c0586c3bc874e68c", weights: "pythia-160m.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-160m", revision: "50f5173d932e8e61f858120bcb800b97af589f46", tokenizer: "tokenizer.json" } },
    download: 174602272,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-gpt2", name: "GPT-2 124M", note: "English · fetches 178 MB (GGUF) → int8 157 MB",
    original: "openai-community/gpt2",
    hf: { repo: "mradermacher/gpt2-GGUF", revision: "0cda0c2b1459ccd32256c6ddde9d230934112c1c", weights: "gpt2.Q8_0.gguf",
          vocabulary: { repo: "openai-community/gpt2", revision: "607a30d783dfa663caf39e06633721c8d4cfcd7e", tokenizer: "tokenizer.json" } },
    download: 177669376,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-smollm2-360m-instruct", name: "SmolLM2 360M Instruct", note: "answers instructions · English · fetches 386 MB (GGUF) → int8 390 MB",
    original: "HuggingFaceTB/SmolLM2-360M-Instruct",
    hf: { repo: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", revision: "593b5a2e04c8f3e4ee880263f93e0bd2901ad47f", weights: "smollm2-360m-instruct-q8_0.gguf" }, download: 386404992,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-pythia-410m", name: "Pythia 410M", note: "English · fetches 433 MB (GGUF) → int8 506 MB",
    original: "EleutherAI/pythia-410m",
    hf: { repo: "mradermacher/pythia-410m-GGUF", revision: "29569dd6a296a653472b3086dbc8ded8479e9707", weights: "pythia-410m.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-410m", revision: "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f", tokenizer: "tokenizer.json" } },
    download: 433397664,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-qwen2.5-coder-0.5b-instruct", name: "Qwen2.5 Coder 0.5B Instruct", note: "writes code · English · fetches 531 MB (GGUF) → int8 545 MB",
    original: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF", revision: "69a2c192eed24297fb09a34d8ba948b8624cc3e2", weights: "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf" }, download: 531068576,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "Write a Python function that reverses a string.", placeholder: "Ask for code (e.g. Write a Python function that sorts a list.)" },
  { group: "hf", id: "hf-pythia-1b", name: "Pythia 1B", note: "English · fetches 1.1 GB (GGUF) → int8 1.1 GB · desktop only",
    original: "EleutherAI/pythia-1b",
    hf: { repo: "mradermacher/pythia-1b-GGUF", revision: "56d6c599008603650e8ee607d49b885bfd766b72", weights: "pythia-1b.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-1b", revision: "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2", tokenizer: "tokenizer.json" } },
    download: 1078061728,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  // TinyLlama's template has </s> between the turns: specials makes the tokenizer read it as the token, not as text
  { group: "hf", id: "hf-tinyllama-1.1b-chat", name: "TinyLlama 1.1B Chat", note: "answers instructions · English · fetches 1.2 GB (GGUF) → int8 1.2 GB · desktop only",
    original: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    hf: { repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", revision: "52e7645ba7c309695bec7ac98f4f005b139cf465", weights: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
          vocabulary: { repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", revision: "fe8a4ea1ffedaf415f4da2f062534de366a451e6", tokenizer: "tokenizer.model" } }, download: 1170781568,
    conversion: {}, options: { specials: ["</s>"] }, generation: sampled(1.1), template: "<|user|>\n{prompt}</s>\n<|assistant|>\n",
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  // T106: Llama 3. The original (meta-llama) is gated, so the same weights come from unsloth's copy (`original`
  // names whose they are). The chat template and its special tokens are read from the model (T73)
  { group: "hf", id: "hf-llama-3.2-1b-instruct", name: "Llama 3.2 1B Instruct", note: "answers instructions · English · fetches 1.3 GB (GGUF) → int8 1.4 GB · desktop only",
    original: "meta-llama/Llama-3.2-1B-Instruct",
    hf: { repo: "bartowski/Llama-3.2-1B-Instruct-GGUF", revision: "067b946cf014b7c697f3654f621d577a3e3afd1c", weights: "Llama-3.2-1B-Instruct-Q8_0.gguf",
          vocabulary: { repo: "unsloth/Llama-3.2-1B-Instruct", revision: "5a8abab4a5d6f164389b1079fb721cfab8d7126c", tokenizer: "tokenizer.json" } }, download: 1321083008,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-pythia-1.4b", name: "Pythia 1.4B", note: "English · fetches 1.5 GB (GGUF) → int8 1.6 GB · desktop only",
    original: "EleutherAI/pythia-1.4b",
    hf: { repo: "mradermacher/pythia-1.4b-GGUF", revision: "24c447e001d1db62d2af39a31084b71de854b24d", weights: "pythia-1.4b.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-1.4b", revision: "fedc38a16eea3bd36a96b906d78d11d2ce18ed79", tokenizer: "tokenizer.json" } },
    download: 1506738080,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-qwen2.5-1.5b-instruct", name: "Qwen2.5 1.5B Instruct", note: "answers instructions · 日本語 / English · fetches 1.6 GB (GGUF) → int8 1.7 GB · desktop only",
    original: "Qwen/Qwen2.5-1.5B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-1.5B-Instruct-GGUF", revision: "9eadc66189c7641e1ddd226b8267a9119b2ce2d4", weights: "Qwen2.5-1.5B-Instruct-Q8_0.gguf" }, download: 1646573312,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-smollm2-1.7b-instruct", name: "SmolLM2 1.7B Instruct", note: "answers instructions · English · fetches 1.8 GB (GGUF) → int8 1.9 GB · desktop only",
    original: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    hf: { repo: "bartowski/SmolLM2-1.7B-Instruct-GGUF", revision: "1f03464768bfcc0319fc50da8ff5fb20b6417ba2", weights: "SmolLM2-1.7B-Instruct-Q8_0.gguf" }, download: 1820414944,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-deepseek-r1-qwen-1.5b", name: "DeepSeek-R1 Distill Qwen 1.5B", note: "thinks before it answers · English · fetches 1.9 GB (GGUF) → int8 2.0 GB · desktop only",
    // T203: mradermacher's Q8_0 (its card names the original as its base model, and MIT)
    ...ggufOf("mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "dcc15b1cfd0973faf5dc043fd405661fe061e7aa",
      "DeepSeek-R1-Distill-Qwen-1.5B.Q8_0.gguf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"),
    download: 1894532384,
    // the BOS is the tokenizer's <｜begin▁of▁sentence｜> (151646): config.json says 151643, the end of a sentence, and
    // with it perplexity was 2.5 to 2.7 times higher and four answers of five fell apart; the real format opens the
    // thought with <think> (T138's review). The list's options go over what the kept conversion says: no new CONVERTER
    conversion: {}, options: { bos: 151646, specials: ["<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<think>"], stop_tokens: [151643] },
    // its model card's sampling: temperature 0.6 and top-p 0.95, and no penalty (it names none): Qwen3's thinking (T144)
    generation: thinking, template: "<｜User｜>{prompt}<｜Assistant｜><think>\n",
    prompt: "What is 17 times 24? Think first.", placeholder: "Ask something that needs thinking" },
  // T251: the English classics, each a ladder of one design. GPT-2 below and above the 124M one (DistilGPT2 is its
  // six-layer student), and Pythia's two ends (14M to 6.9B with the five above: T84's size against speed). Q8_0 GGUFs
  // which tests/gguf_check.py tensors held to the originals (gguf.yml's candidates), with the originals' vocabulary.
  // GPT-2's table of positions holds 1024, which is its context; Pythia's context is 2048. The 14M is the deduped one,
  // as the 70M is (mradermacher's GGUF of that name is of pythia-14m-deduped, its card says)
  { group: "hf", id: "hf-pythia-14m", name: "Pythia 14M", note: "English · fetches 17 MB (GGUF) → int8 16 MB",
    ...ggufOf("mradermacher/pythia-14m-GGUF", "6e0b616b2d66b8f2c6bb6f80fc661350a19d1dcf", "pythia-14m.Q8_0.gguf",
      "EleutherAI/pythia-14m-deduped", "7386d9a4ae45aef494a6e704910394def3037fc5"), download: 16750496,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-distilgpt2", name: "DistilGPT2 82M", note: "English · fetches 132 MB (GGUF) → int8 92 MB",
    ...ggufOf("QuantFactory/distilgpt2-GGUF", "b41ee4e4e4949dba1b3bed4f89a87198b82f851b", "distilgpt2.Q8_0.gguf",
      "distilbert/distilgpt2", "2290a62682d06624634c1f46a6ad5be0f47f38aa"), download: 132303872,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-gpt2-medium", name: "GPT-2 medium 355M", note: "English · fetches 437 MB (GGUF) → int8 400 MB",
    ...ggufOf("mradermacher/gpt2-medium-GGUF", "3b9897d67a84e967fbcc8d7de3db4c797e386740", "gpt2-medium.Q8_0.gguf",
      "openai-community/gpt2-medium", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"), download: 437487744,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-gpt2-large", name: "GPT-2 large 774M", note: "English · fetches 898 MB (GGUF) → int8 873 MB · desktop only",
    ...ggufOf("mradermacher/gpt2-large-GGUF", "c83630987bcee4945f3e947b1977c70e6b2760a1", "gpt2-large.Q8_0.gguf",
      "openai-community/gpt2-large", "32b71b12589c2f8d625668d2335a01cac3249519"), download: 898165824,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-gpt2-xl", name: "GPT-2 XL 1.5B", note: "English · fetches 1.7 GB (GGUF) → int8 1.8 GB · desktop only",
    ...ggufOf("mradermacher/gpt2-xl-GGUF", "a6a1b25a992d7541b185b0cd1227b47311f4eab6", "gpt2-xl.Q8_0.gguf",
      "openai-community/gpt2-xl", "15ea56dee5df4983c59b2538573817e1667135e2"), download: 1749953760,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-pythia-2.8b", name: "Pythia 2.8B", note: "English · fetches 3.0 GB (GGUF) → int8 3.1 GB · desktop only",
    ...ggufOf("mradermacher/pythia-2.8b-GGUF", "147586a405b950ff991ba909a3193ffafa98b014", "pythia-2.8b.Q8_0.gguf",
      "EleutherAI/pythia-2.8b", "2a259cdd96a4beb1cdf467512e3904197345f6a9"), download: 2953594016,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-pythia-6.9b", name: "Pythia 6.9B", note: "English · fetches 7.3 GB (GGUF) → int8 7.7 GB · desktop only · Chrome and Firefox",
    ...ggufOf("mradermacher/pythia-6.9b-GGUF", "d8e5050a875b7bfd5de881d30b33abc3d492cb60", "pythia-6.9b.Q8_0.gguf",
      "EleutherAI/pythia-6.9b", "c0e3eee36dc47af0c49f361c74cfe459c09f7f23"), download: 7292706720,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  // T125: Mistral 7B, and zephyr made from it
  { group: "hf", id: "hf-mistral-7b-instruct-v0.2", name: "Mistral 7B Instruct v0.2", note: "answers instructions · English · fetches 7.7 GB (GGUF) → int8 8.2 GB · desktop only · Chrome and Firefox",
    original: "mistralai/Mistral-7B-Instruct-v0.2",
    hf: { repo: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", revision: "3a6fbf4a41a1d52e415a4958cde6856d34b2db93", weights: "mistral-7b-instruct-v0.2.Q8_0.gguf",
          vocabulary: { repo: "mistralai/Mistral-7B-Instruct-v0.2", revision: "63a8b081895390a26e140280378bc85ec8bce07a", tokenizer: "tokenizer.model" } }, download: 7695857952,
    conversion: {}, options: {}, generation: sampled(1.1), template: MISTRAL,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-mistral-7b-instruct-v0.3", name: "Mistral 7B Instruct v0.3", note: "answers instructions · English · fetches 7.7 GB (GGUF) → int8 8.2 GB · desktop only · Chrome and Firefox",
    original: "mistralai/Mistral-7B-Instruct-v0.3",
    hf: { repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF", revision: "61fd4167fff3ab01ee1cfe0da183fa27a944db48", weights: "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
          vocabulary: { repo: "mistralai/Mistral-7B-Instruct-v0.3", revision: "c170c708c41dac9275d15a8fff4eca08d52bab71", tokenizer: "tokenizer.model" } }, download: 7702565088,
    conversion: {}, options: { specials: ["[/INST]", "[INST]"] }, generation: sampled(1.1), template: MISTRAL_V3,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-zephyr-7b-beta", name: "zephyr 7B beta", note: "answers instructions · English · fetches 7.7 GB (GGUF) → int8 8.2 GB · desktop only · Chrome and Firefox",
    original: "HuggingFaceH4/zephyr-7b-beta",
    hf: { repo: "TheBloke/zephyr-7B-beta-GGUF", revision: "e4714d14e9652aa9658fa937732cceadc63ac42e", weights: "zephyr-7b-beta.Q8_0.gguf",
          vocabulary: { repo: "HuggingFaceH4/zephyr-7b-beta", revision: "892b3d7a7b1cf10c7a701c60881cd93df615734c", tokenizer: "tokenizer.model" } }, download: 7695857344,
    conversion: {}, options: { specials: ["</s>"] }, generation: sampled(1.1), template: ZEPHYR,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-llama-3.2-3b-instruct", name: "Llama 3.2 3B Instruct", note: "answers instructions · English · fetches 3.4 GB (GGUF) → int8 3.6 GB · desktop only",
    original: "meta-llama/Llama-3.2-3B-Instruct",
    hf: { repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", revision: "5ab33fa94d1d04e903623ae72c95d1696f09f9e8", weights: "Llama-3.2-3B-Instruct-Q8_0.gguf",
          vocabulary: { repo: "unsloth/Llama-3.2-3B-Instruct", revision: "006f5dcd1393c3add266de40994ba96225e9689d", tokenizer: "tokenizer.json" } }, download: 3421899296,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  // T252: well-known ones the engine opens as it is, each from a Q8_0 GGUF that tests/gguf_check.py tensors held to its
  // original (gguf.yml's candidates), with the original's vocabulary, config.json and chat template.
  // SmolLM2's base models (the Instruct ones are above)
  { group: "hf", id: "hf-smollm2-135m", name: "SmolLM2 135M", note: "English · fetches 145 MB (GGUF) → int8 151 MB",
    ...ggufOf("mradermacher/SmolLM2-135M-GGUF", "bf92313aa80eb55329ae75ccce3743101784c802", "SmolLM2-135M.Q8_0.gguf",
      "HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"), download: 144810944,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-smollm2-360m", name: "SmolLM2 360M", note: "English · fetches 386 MB (GGUF) → int8 407 MB",
    ...ggufOf("mradermacher/SmolLM2-360M-GGUF", "630c4866d716e28f45a516ed00e2882149726c13", "SmolLM2-360M.Q8_0.gguf",
      "HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"), download: 386404864,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  // Qwen2.5 Coder above the 0.5B one
  { group: "hf", id: "hf-qwen2.5-coder-1.5b-instruct", name: "Qwen2.5 Coder 1.5B Instruct", note: "writes code · English · fetches 1.6 GB (GGUF) → int8 1.7 GB · desktop only",
    ...ggufOf("bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF", "1af47f78b1f9b0c242fabe43f7a365d5a67f3207", "Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf",
      "Qwen/Qwen2.5-Coder-1.5B-Instruct", "2e1fd397ee46e1388853d2af2c993145b0f1098a"), download: 1646573344,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "Write a Python function that reverses a string.", placeholder: "Ask for code (e.g. Write a Python function that sorts a list.)" },
  { group: "hf", id: "hf-qwen2.5-coder-3b-instruct", name: "Qwen2.5 Coder 3B Instruct", note: "writes code · English · fetches 3.3 GB (GGUF) → int8 3.5 GB · desktop only",
    ...ggufOf("bartowski/Qwen2.5-Coder-3B-Instruct-GGUF", "7c137640ef0332dfedb229f2504c58d83ed4307a", "Qwen2.5-Coder-3B-Instruct-Q8_0.gguf",
      "Qwen/Qwen2.5-Coder-3B-Instruct", "488639f1ff808d1d3d0ba301aef8c11461451ec5"), download: 3285476608,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "Write a Python function that reverses a string.", placeholder: "Ask for code (e.g. Write a Python function that sorts a list.)" },
  { group: "hf", id: "hf-qwen2.5-coder-7b-instruct", name: "Qwen2.5 Coder 7B Instruct", note: "writes code · English · fetches 8.1 GB (GGUF) → int8 8.6 GB · desktop only · Chrome and Firefox",
    ...ggufOf("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "1f629da0c8bed16b9e50cee91c70693650e66c35", "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
      "Qwen/Qwen2.5-Coder-7B-Instruct", "c03e6d358207e414f1eca0bb1891e29f1db0e242"), download: 8098525984,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "Write a Python function that reverses a string.", placeholder: "Ask for code (e.g. Write a Python function that sorts a list.)" },
  // Hermes 3 (Nous Research) on Llama 3.2 3B, in ChatML: its own Q8_0 GGUF
  { group: "hf", id: "hf-hermes-3-llama-3.2-3b", name: "Hermes 3 Llama 3.2 3B", note: "answers instructions · English · fetches 3.4 GB (GGUF) → int8 3.6 GB · desktop only",
    ...ggufOf("NousResearch/Hermes-3-Llama-3.2-3B-GGUF", "3cd927095d8cbab12c743f932aa63b6f7bbfa141", "Hermes-3-Llama-3.2-3B.Q8_0.gguf",
      "NousResearch/Hermes-3-Llama-3.2-3B", "7f1a6bec8cdce6551014fd5bbeb4cd8c0f1fbeab"), download: 3421895488,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  // DeepSeek-R1's larger distills, as the 1.5B above: the tokenizer's own BOS and the thought opened by the format
  { group: "hf", id: "hf-deepseek-r1-qwen-7b", name: "DeepSeek-R1 Distill Qwen 7B", note: "thinks before it answers · English · fetches 8.1 GB (GGUF) → int8 8.6 GB · desktop only · Chrome and Firefox",
    ...ggufOf("mradermacher/DeepSeek-R1-Distill-Qwen-7B-GGUF", "75e791d579161aa4082e642c8b84ea7814829698", "DeepSeek-R1-Distill-Qwen-7B.Q8_0.gguf",
      "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "916b56a44061fd5cd7d6a8fb632557ed4f724f60"), download: 8098525344,
    conversion: {}, options: {}, generation: thinking,
    prompt: "What is 17 times 24? Think first.", placeholder: "Ask something that needs thinking" },
  { group: "hf", id: "hf-deepseek-r1-llama-8b", name: "DeepSeek-R1 Distill Llama 8B", note: "thinks before it answers · English · fetches 8.5 GB (GGUF) → int8 9.0 GB · desktop only · Chrome and Firefox",
    ...ggufOf("mradermacher/DeepSeek-R1-Distill-Llama-8B-GGUF", "5c27c16fe2584d3a33b8633c8df944ce5f22a6b7", "DeepSeek-R1-Distill-Llama-8B.Q8_0.gguf",
      "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "6a6f4aa4197940add57724a7707d069478df56b1"), download: 8540773376,
    conversion: {}, options: {}, generation: thinking,
    prompt: "What is 17 times 24? Think first.", placeholder: "Ask something that needs thinking" },
  // Llama 3.1 8B Instruct, as Llama 3.2 above: the original is gated, so the vocabulary and config.json of unsloth's copy
  { group: "hf", id: "hf-llama-3.1-8b-instruct", name: "Llama 3.1 8B Instruct", note: "answers instructions · English · fetches 8.5 GB (GGUF) → int8 9.0 GB · desktop only · Chrome and Firefox",
    original: "meta-llama/Llama-3.1-8B-Instruct",
    hf: { repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", revision: "bf5b95e96dac0462e2a09145ec66cae9a3f12067", weights: "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
          vocabulary: { repo: "unsloth/Meta-Llama-3.1-8B-Instruct", revision: "a2856192dd7c25b842431f39c179a6c2c2f627d1", tokenizer: "tokenizer.json" } }, download: 8540775840,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  // T124: Qwen3, each twice (thinking and not), and Qwen3's 2507 4B, one of each form. T203: from Q8_0 GGUFs (Qwen's
  // own 0.6B and 1.7B are not the original's weights: tests/gguf_check.py tensors found every matrix 0.9 to 2.2% off
  // its Q8_0, so unsloth's)
  ...thinkingAndNot("hf-qwen3-0.6b", "Qwen3 0.6B",
    ggufOf("unsloth/Qwen3-0.6B-GGUF", "50968a4468ef4233ed78cd7c3de230dd1d61a56b", "Qwen3-0.6B-Q8_0.gguf",
      "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"), 639447744,
    "fetches 639 MB (GGUF) → int8 671 MB"),
  ...thinkingAndNot("hf-qwen3-1.7b", "Qwen3 1.7B",
    ggufOf("unsloth/Qwen3-1.7B-GGUF", "d7f544eead698dbd1f15126ef60b45a1e1933222", "Qwen3-1.7B-Q8_0.gguf",
      "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"), 1834426944,
    "fetches 1.8 GB (GGUF) → int8 1.9 GB · desktop only"),
  ...thinkingAndNot("hf-qwen3-4b", "Qwen3 4B",
    ggufOf("Qwen/Qwen3-4B-GGUF", "bc640142c66e1fdd12af0bd68f40445458f3869b", "Qwen3-4B-Q8_0.gguf",
      "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"), 4280404704,
    "fetches 4.3 GB (GGUF) → int8 4.5 GB · desktop only · Chrome and Firefox"),
  ...thinkingAndNot("hf-qwen3-8b", "Qwen3 8B",
    ggufOf("Qwen/Qwen3-8B-GGUF", "7c41481f57cb95916b40956ab2f0b139b296d974", "Qwen3-8B-Q8_0.gguf",
      "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"), 8709518112,
    "fetches 8.7 GB (GGUF) → int8 9.2 GB · desktop only · Chrome and Firefox",
    { options: QWEN3_FROM_IM_START }, { thinking: QWEN3_THINKING_AFTER_START, atOnce: QWEN3_AT_ONCE_AFTER_START }),
  // Qwen3 Swallow's card gives one sampling, the thinking one, for both
  ...thinkingAndNot("hf-qwen3-swallow-8b", "Qwen3 Swallow 8B RL",
    ggufOf("mmnga-o/Qwen3-Swallow-8B-RL-v0.2-gguf", "3fc755c6ab3780ebad6671130fc1b630bbd1575b", "Qwen3-Swallow-8B-RL-v0.2-Q8_0.gguf",
      "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", "9218f4843b6f93369a0b0999d8f58d61487ea71c"), 8709519968,
    "fetches 8.7 GB (GGUF) → int8 9.2 GB · desktop only · Chrome and Firefox", { generation: thinking }),
  { group: "hf", id: "hf-qwen3-4b-instruct-2507", name: "Qwen3 4B Instruct 2507", note: "answers instructions · 日本語 / English · fetches 4.3 GB (GGUF) → int8 4.5 GB · desktop only · Chrome and Firefox",
    ...ggufOf("unsloth/Qwen3-4B-Instruct-2507-GGUF", "a06e946bb6b655725eafa393f4a9745d460374c9", "Qwen3-4B-Instruct-2507-Q8_0.gguf",
      "Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"), download: 4280405600,
    conversion: {}, options: {}, generation: atOnce, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // its chat_template begins the answer with <think> itself
  { group: "hf", id: "hf-qwen3-4b-thinking-2507", name: "Qwen3 4B Thinking 2507", note: "thinks before it answers · 日本語 / English · fetches 4.3 GB (GGUF) → int8 4.5 GB · desktop only · Chrome and Firefox",
    ...ggufOf("unsloth/Qwen3-4B-Thinking-2507-GGUF", "f40adb104d4d44aee52f398b60597c5866a973a3", "Qwen3-4B-Thinking-2507-Q8_0.gguf",
      "Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea"), download: 4280405632,
    conversion: {}, options: {}, generation: thinking, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T235: Prism ML's ternary Qwen3 1.7B, every weight -1, 0 or 1 times a scale of its 128. Its PQ2_0 GGUF holds two
  // bits a weight, which the converter keeps as they are (T230: the ternary dtype, a quarter of int8's bytes, on kernels
  // of its own, T231; ?bits=8 widens them to int8 without loss of the values, as T235 did), with the vocabulary, config.json and
  // chat template of the float16 safetensors of the same weights (the card's base model). T246: and the 4B and the 8B,
  // the same in every file but the weights and config.json's sizes (ternaryBonsai() has what the three share). The
  // 4B's heads are not dim / heads wide and the 8B has a classifier of its own, as the Qwen3 4B and 8B they are built
  // from; as ternary they fit a 32-bit memory with their forward pass (2.2 and 3.3 GiB at 4096 positions; widened to
  // int8 they were 5.3 and 10.1 GiB, on a 64-bit one)
  ternaryBonsai("1.7B", "983b5dec2ff16aab79990711ba0f828a499a7e6a", "3aca840085293d026ce6f6b80fafdae937fd2eeb", 463290464,
    "fetches 463 MB (GGUF) → ternary 484 MB"),
  ternaryBonsai("4B", "a3eb42bafe873f9686bc97486c43b72ef7d75ec8", "4485fae7a00129467b9329b738110d88b2942a1a", 1074969344,
    "fetches 1.1 GB (GGUF) → ternary 1.1 GB · desktop only"),
  ternaryBonsai("8B", "c2aefbeb4b24469cd11579c3384b990404c17a30", "ac20f03fc62e872399218b659c8e949dfca05769", 2182184672,
    "fetches 2.2 GB (GGUF) → ternary 2.3 GB · desktop only", { options: QWEN3_FROM_IM_START, template: QWEN3_AT_ONCE_AFTER_START }),
  // T236: Qwen3.5 0.8B, the first of the list with hybrid attention (T229: three layers of four are Gated DeltaNet
  // layers, which keep a state of a fixed size where the fourth keeps keys and values), on the CPU (no GPU path yet).
  // A vision-language model, of which the page reads the language model. unsloth's Q8_0 GGUF, which
  // tests/gguf_check.py tensors held to the original (every Q8_0 tensor is llama.cpp's Q8_0 of the original's, every
  // F32 one the original's values). Its card: thinking is off unless asked for (the 0.8B "is more prone to entering
  // thinking loops"), and for the sampling without thinking it names temperature 1.0, top-p 1.0, top-k 20 and a
  // presence penalty of 2.0 for text, and 0.7, 0.8, 20 and 1.5 for pictures and in its benchmarks; with thinking 1.0,
  // 0.95, 20 and 1.5, or 0.6, 0.95, 20 and no penalty "for precise coding". The page's sampler has neither top-k nor a
  // presence penalty, and temperature 1.0 over the whole vocabulary leans on the top-k: so Qwen3's two, which are the
  // card's 0.7 and 0.8 without thinking and its 0.6 and 0.95 with
  ...thinkingAndNot("hf-qwen3.5-0.8b", "Qwen3.5 0.8B",
    ggufOf("unsloth/Qwen3.5-0.8B-GGUF", "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "Qwen3.5-0.8B-Q8_0.gguf",
      "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"), 811843840,
    "fetches 812 MB (GGUF) → int8 850 MB · desktop only", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }),
  // T247: the other sizes of Qwen3.5 whose int8 a 64-bit memory of 16 GiB holds (the 27B's Q8_0 is 28.6 GB, and the
  // larger ones are mixtures of experts, which the engine has not). The same vocabulary, format and form as the 0.8B.
  // The 4B and the 9B have two value heads to a key head in their linear-attention layers, which llama.cpp writes in
  // another order than Hugging Face (T245: the converter puts them back), and they think unless told not to (the 0.8B
  // and the 2B only when told to): either form is written out here, so the two entries are the same two. Their cards
  // name 0.7 and 0.8 without thinking and 0.6 and 0.95 "for precise coding" with (and 1.0 and 0.95 with a top-k and a
  // presence penalty, which the page's sampler has not). The 4B and the 9B are past a 32-bit memory as int8
  ...thinkingAndNot("hf-qwen3.5-2b", "Qwen3.5 2B",
    ggufOf("unsloth/Qwen3.5-2B-GGUF", "f6d5376be1edb4d416d56da11e5397a961aca8ae", "Qwen3.5-2B-Q8_0.gguf",
      "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"), 2012012800,
    "fetches 2.0 GB (GGUF) → int8 2.1 GB · desktop only", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }),
  ...thinkingAndNot("hf-qwen3.5-4b", "Qwen3.5 4B",
    ggufOf("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q8_0.gguf",
      "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"), 4482403488,
    "fetches 4.5 GB (GGUF) → int8 4.7 GB · desktop only · Chrome and Firefox", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }),
  ...thinkingAndNot("hf-qwen3.5-9b", "Qwen3.5 9B",
    ggufOf("unsloth/Qwen3.5-9B-GGUF", "3885219b6810b007914f3a7950a8d1b469d598a5", "Qwen3.5-9B-Q8_0.gguf",
      "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"), 9527502048,
    "fetches 9.5 GB (GGUF) → int8 10.1 GB · desktop only · Chrome and Firefox", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }),
  // T253: Granite 4.2 (IBM; Japanese is among the languages its card says it was tested in), a Llama whose attention
  // multiplies its scores by config.json's attention_multiplier, which the converter puts into q (llama2_convert's
  // query_scale()). IBM's own Q8_0 GGUFs, which tests/gguf_check.py tensors held to the originals. The 3B fits a
  // 32-bit memory in six bits (Safari); the 8B does not
  ...thinkingAndNot("hf-granite-4.2-3b", "Granite 4.2 3B",
    ggufOf("ibm-granite/granite-4.2-3b-GGUF", "c40945d71cd90f249a56985e8155551a9188dc30", "granite-4.2-3b-Q8_0.gguf",
      "ibm-granite/granite-4.2-3b", "e459acceac81e5fe67c07d9cfc72329a332e7eb1"), 3892651552,
    "fetches 3.9 GB (GGUF) → int8 4.1 GB · desktop only", { options: granite },
    { thinking: GRANITE_THINKING, atOnce: GRANITE_AT_ONCE }).map((entry) => ({ ...entry, generation: graniteSampling })),
  ...thinkingAndNot("hf-granite-4.2-8b", "Granite 4.2 8B",
    ggufOf("ibm-granite/granite-4.2-8b-GGUF", "93f3f6a8938ee922b784cf4e5b4203cd3428df8f", "granite-4.2-8b-Q8_0.gguf",
      "ibm-granite/granite-4.2-8b", "f8de16cdcdbc6c779ca517604e050d82cc119e44"), 9345613952,
    "fetches 9.3 GB (GGUF) → int8 9.9 GB · desktop only · Chrome and Firefox", { options: granite },
    { thinking: GRANITE_THINKING, atOnce: GRANITE_AT_ONCE }).map((entry) => ({ ...entry, generation: graniteSampling })),
  // T254: MiniCPM5, whose tokenizer.json cuts the numbers off before Llama 3's pattern runs (the engine's "minicpm5").
  // OpenBMB's own Q8_0 GGUFs, which tests/gguf_check.py tensors held to the originals; the vocabulary is the
  // original's (the 1B's GGUF calls its pre-tokenizer llama-bpe, which is not what its tokenizer.json does). The
  // cards: temperature 0.9 thinking and 0.7 without for the 1B, 1.0 for the 2B, top-p 0.95
  ...miniCpm5("hf-minicpm5-1b", "MiniCPM5 1B",
    ggufOf("openbmb/MiniCPM5-1B-GGUF", "3d55fac80935ae6456986ad2384b5cbcc4d6c948", "MiniCPM5-1B-Q8_0.gguf",
      "openbmb/MiniCPM5-1B", "87179e5c1f455ef22e6223592d2d61351b525bfc"), 1153529216,
    "fetches 1.2 GB (GGUF) → int8 1.2 GB · desktop only", { thinking: 0.9, atOnce: 0.7 }),
  ...miniCpm5("hf-minicpm5-2b", "MiniCPM5 2B",
    ggufOf("openbmb/MiniCPM5-2B-GGUF", "2079a22f3beaa4e306449978533478fe0522f4b3", "MiniCPM5-2B-Q8_0.gguf",
      "openbmb/MiniCPM5-2B", "f97400052a43d642bbc6e9975e2397e3ae6a6b52"), 2679710688,
    "fetches 2.7 GB (GGUF) → int8 2.8 GB · desktop only", { thinking: 1.0, atOnce: 1.0 }),
];

// T90: memory. A device that runs out of it kills the worker's WebAssembly memory, so the page warns before it
// loads a model that probably does not fit, and says what happened when it did not.
/** What the page takes besides the model: Pyodide, NumPy and the engine (the margin T90 asked for; the heap of
 * llm-jp-3 150M measures 112 MB above its 171 MB of weights, and the tab needs its own). */
export const PAGE_MEMORY = 300e6;
const megabytes = (bytes) => `${Math.round(bytes / 1e6).toLocaleString("en")} MB`;
/** The bytes of a model once loaded: `bytes` of a file of this site, or the "int8 N MB" its note gives for a
 * conversion ("ternary N MB" for a ternary model, T230: four times that as int8, where ?bits= asks for it). undefined
 * when neither says (a file of the visitor's). */
export function modelBytes(entry) {
  // a float16 original is widened to float32 when loaded, next to the file it came from: llm-jp-3 150M's 305 MB
  // file measures about 800 MB of heap (AGENTS.md), so three times the file is the honest estimate
  if (entry.bytes) return entry.options?.dtype === "float16" ? entry.bytes * 3 : entry.bytes;
  const found = /(int8|ternary) ([\d.]+) (MB|GB)/.exec(entry.note ?? "");
  if (!found) return undefined;
  const said = Number(found[2]) * (found[3] === "GB" ? 1e9 : 1e6), asked = entry.conversion?.dtype;
  if (found[1] === "ternary" && !["int8", "int6"].includes(asked)) return said;
  const int8 = found[1] === "ternary" ? said / TERNARY_OF_EIGHT : said;
  return asked === "int6" ? int8 * SIX_OF_EIGHT : int8;
}

// T98: a model converted in the page can keep its weights in six bits instead of eight: 24 bytes and a scale per
// group of 32 against 32 and a scale, 7/9 of the size, at +1 to +3.4% of perplexity (measured on eight models), and
// slower on one thread (the groups are widened as they are read). So it is taken where int8 does not fit.
export const SIX_OF_EIGHT = 28 / 36;
// T230: a ternary model keeps its weights as they are, two bits each: 32 bytes and a scale per group of 128, a quarter
// of int8's 128 bytes and four scales, with no loss at all (int8 is the same weights widened)
export const TERNARY_OF_EIGHT = 36 / 144;

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
 * automatic, which is the entry's own `weights` where it names them (T230: "ternary", a ternary model's weights as
 * they are, smaller than six bits of them and exact), and else takes int6 where int8 would pass half of what the device says it has (deviceMemory, Chromium
 * only, and below its cap of 8: a device at the cap may have any more), and otherwise leaves the choice to the
 * worker (undefined): it knows the model's header once it converts, and with it what the forward pass needs, and
 * takes int6 where int8 would not fit a 32-bit memory and the browser has no 64-bit one (T115, T133).
 * undefined for a model that is not converted in the page. */
export function weightsFor(entry, asked, deviceMemory) {
  if (!entry.hf) return undefined;
  // a visitor's own files may come with settings that say it ({"conversion": {"dtype": ...}}): they win (T119)
  if (entry.conversion?.dtype) return entry.conversion.dtype;
  if (asked === "6" || asked === "8") return `int${asked}`;
  if (entry.weights) return entry.weights;
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
