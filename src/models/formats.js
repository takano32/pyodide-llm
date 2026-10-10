// The formats the list's entries send their prompts in, and the engine's options that go with them (the tokens a
// format writes, where an answer stops): one turn of each model's chat template, as the real one renders it.

export const JAPANESE = "文章の書き出しを入力（例: 富士山は、）";
export const STORY = "Type the beginning of a story (e.g. Lily and Tom went to the park.)";
export const unigram = { tokenizer_kind: "unigram" };
// An instruction-tuned model answers instead of continuing, when its input has the form it was trained on: template
// wraps what the visitor typed ({prompt}), for one turn and no more. The forms are the chat_template of each model.
export const LLM_JP_INSTRUCT = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{prompt}\n\n### 応答:\n";
export const ASK_JAPANESE = "質問や指示を入力（例: 日本の首都は？）";
// ChatML. <|im_start|> and <|im_end|> are tokens of their own, so the engine is told to read them as such
export const CHATML = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
// the same for a model whose BOS is <|im_start|> itself, which the page begins every text with (SmolLM2's, a Qwen3's
// QWEN3_FROM_IM_START and Hermes 3's, T252's review): the format begins after it (T250's review)
export const CHATML_AFTER_START = "user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
export const chatml = { specials: ["<|im_start|>", "<|im_end|>"], stop_tokens: [0, 2] };
// sarashina2.2's chat_template (and CAT-Translate's, made from it) uses selectattr, which this project's template
// reader does not take (T73): one turn of it, as the real Jinja renders it, is this (T81; the 0.5B Instruct had
// ChatML here until 2026-09-26, whose <|im_start|> its vocabulary does not have). <|user|> (9), <|assistant|> (8)
// and </s> (2) are tokens of their own, and a model that writes the mark of a turn (7 to 9) has ended its answer
export const SARASHINA = "<|user|>{prompt}</s><|assistant|>";
export const sarashina = { specials: ["<|assistant|>", "<|user|>", "</s>"], stop_tokens: [1, 2, 7, 8, 9] };
export const TRANSLATE = "Translate the following Japanese text into English.\n\n{日本語の文} (or English into Japanese)";
// llm-jp-4's chat_template is OpenAI's harmony (T132), with a system message of the model's name, its knowledge cutoff
// and the date ({date}: filled() writes today's). The template ends at "<|start|>assistant" and leaves the channel to
// the model; this one asks for the final channel, the answer, and so skips the analysis a harmony model may write
// first. Its tokenizer.json puts a "▁" before the text after each special token (a normalizer that replaces the start
// of every piece with it): the converter says so to the engine (prefixed, T308; until T369 a space after each special
// token here made the same tokens: the same IDs as the real Jinja and tokenizers for four prompts, T132). A turn ends
// with <|return|> (2), <|end|> (11) or <|call|> (13), and a new message would start with <|start|> (10)
export const HARMONY = "<|start|>system<|message|>You are LLM-jp-4, a large language model trained by LLM-jp.\nKnowledge cutoff: " +
  "2025-12\nCurrent date: {date}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every " +
  "message.<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>";
// T125: Mistral's formats, one turn as the real Jinja writes it (the same IDs as the real Jinja and tokenizers for
// four prompts, where the prompt has no space at either end: some templates trim it, the page does not). These
// models come with a sentencepiece tokenizer.model, which the engine reads (their tokenizer.json is a BPE of
// sentencepiece's kind, which it does not). v0.2 writes "<s> [INST]": after the BOS, the engine's dummy prefix is
// that space. v0.3's [INST] and [/INST] are tokens of their own (3 and 4)
export const MISTRAL = "[INST] {prompt} [/INST]";
export const MISTRAL_V3 = "[INST] {prompt}[/INST]";
// RakutenAI's (2.0 mini and 7B chat): no special tokens, a system sentence and USER / ASSISTANT; the template trims
// what was typed (T138)
export const RAKUTEN = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, " +
  "detailed, and polite answers to the user's questions. USER: {prompt:trim} ASSISTANT:";
// T138: Swallow-MS's card always passes this system message, which its template puts in the user's turn; the template
// strips the whole turn, which trims what was typed at its end only (filled() trims both ends: a prompt that begins
// with spaces differs)
export const SWALLOW_MS = "[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{prompt:trim} [/INST] ";
// zephyr's tokenizer puts a "▁" before the text after </s> (a legacy Llama tokenizer): the converter says so to the
// engine (prefixed, T308; until T369 a space after </s> here made the same tokens)
export const ZEPHYR = "<|user|>\n{prompt}</s>\n<|assistant|>\n";
// T249: EuroLLM's chat_template is ChatML with a system turn that is empty unless one is given. Its tokenizer (a legacy
// Llama tokenizer, as zephyr's) puts a "▁" before the text after <|im_start|> and <|im_end|>: the converter says so to
// the engine (prefixed, T308; until T369 a space after each here made the same tokens: the same IDs as the real Jinja
// and tokenizers for tests/format_check.py's prompts)
export const EUROLLM = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
// T250: Llama-3-ELYZA-JP's template is Llama 3's (no date, the turns trimmed), and its card always passes this system
// message, as Swallow-MS's does: one turn of it as the real Jinja writes it with that message
export const ELYZA = "<|start_header_id|>system<|end_header_id|>\n\nあなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、" +
  "常に日本語で回答してください。<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt:trim}<|eot_id|>" +
  "<|start_header_id|>assistant<|end_header_id|>\n\n";
// T124: Qwen3 thinks before it answers (<think>…</think>, then the answer), which is the form its chat_template writes
// and the converter reads. The same weights answer at once when the answer begins with an empty thought: the form of
// enable_thinking=false. <think> and </think> are tokens of the vocabulary that tokenizer.json does not call special:
// the converter names those as specials with the rest (T143: <tool_call>, <|fim_prefix|> ...), so the list names none
// for Qwen3. A list of its own would go over the converter's and spell the others out where a visitor types them (T221)
export const QWEN3_AT_ONCE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
// T250's review: the real Qwen3 tokenizer puts nothing in front of a text and its format begins with <|im_start|>, but
// the converter's BOS is <|endoftext|> (QWEN3_OWN_BOS below, and what config.json names). A Qwen3 of 8 billion
// parameters is much worse with it in front: its own answers to five questions are 178% higher in perplexity (Qwen3 8B,
// plain text 144%; Shisa V2.1's fine-tune of it 67% and 54%), where Qwen's 0.6B and 1.7B are 1 to 4% off and CAT-Thinking
// 8B (Qwen3 Swallow's, continued on Japanese text) 0.4% (tests/answer_check.mjs, tests/start_check.mjs). For those the BOS
// is the format's own first token, <|im_start|> (151644), and the formats begin after it (as Qwen3.5's, T236): the page
// sends the very IDs the real template makes. The answer stops at <|im_end|>, at <|endoftext|> and at the mark of a new turn
export const QWEN3_FROM_IM_START = { bos: 151644, stop_tokens: [151643, 151644, 151645] };
export const QWEN3_THINKING_AFTER_START = CHATML_AFTER_START;
export const QWEN3_AT_ONCE_AFTER_START = `${QWEN3_THINKING_AFTER_START}<think>\n\n</think>\n\n`;
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
export const QWEN35_THINKING = "user\n{prompt:trim}<|im_end|>\n<|im_start|>assistant\n<think>\n";
export const QWEN35_AT_ONCE = `${QWEN35_THINKING}\n</think>\n\n`;
export const qwen35 = { bos: 248045, stop_tokens: [248044, 248045, 248046],
  specials: ["</tool_response>", "<tool_response>", "<|fim_middle|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|repo_name|>",
    "</tool_call>", "<|file_sep|>", "<|im_start|>", "<tool_call>", "<|fim_pad|>", "<|im_end|>", "</think>", "<think>"] };
// T255: SmolLM3's chat_template is past the converter's reader (it asks whether a text is in a variable it may not have
// set), so one turn by hand, as the real Jinja writes it with enable_thinking true and with false and no system turn
// (tests/format_check.py, strict). It writes a system turn of its own first: the day (strftime_now: {date:…}, the
// visitor's own), the mode, and one of two instructions, the long one where it thinks. That turn has no <|im_end|>
// unless there are tools (the template closes it inside that branch only), and the page sends what the template does.
// The real tokenizer begins a text with nothing (bos_token null) and the template with <|im_start|> (128011): that is
// the BOS, and the formats begin after it (as Hermes 3's). It stops at <|im_end|> (config.json's EOS), at
// <|begin_of_text|>, <|end_of_text|> and at the mark of a new turn. Without a template the converter read,
// <|im_start|> and <|im_end|> are not in its specials, and a list of the entry's replaces the converter's (T221): all
// of the converter's are here (the added tokens tokenizer.json does not call special, T143), in its order
const SMOLLM3_THINKS =
  "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves " +
  "thoroughly exploring questions through a systematic thinking process before providing the final precise and " +
  "accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, " +
  "reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please " +
  "structure your response into two main sections: Thought and Solution using the specified format: <think> " +
  "Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. " +
  "Each step should include detailed considerations such as analysing questions, summarizing relevant findings, " +
  "brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting " +
  "previous steps. In the Solution section, based on various attempts, explorations, and reflections from the " +
  "Thought section, systematically present the final solution that you deem correct. The Solution section " +
  "should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.";
const smollm3Format = (mode, instructions, answer) => "system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\n" +
  `Today Date: {date:%d %B %Y}\nReasoning Mode: ${mode}\n\n## Custom Instructions\n\n${instructions}\n\n` +
  `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n${answer}`;
export const SMOLLM3_THINKING = smollm3Format("/think", SMOLLM3_THINKS, "");
export const SMOLLM3_AT_ONCE = smollm3Format("/no_think", "You are a helpful AI assistant named SmolLM, trained by Hugging Face.",
  "<think>\n\n</think>\n");
export const smollm3 = { bos: 128011, stop_tokens: [128000, 128001, 128011, 128012],
  specials: ["</tool_response>", "<tool_response>", "</tool_call>", "<|im_start|>", "<tool_call>", "<|im_end|>", "</think>",
    "</code>", "<think>", "<code>"] };
// T337: Agents-A1-4B's chat_template is Qwen3.5's with one thing more: where the messages have no system turn it writes
// this one (the template's own default: the card's recommended one is the same but names tavily_search and is dated
// 2026-07-13. The date is fixed in the template). So the page sends it too,
// and its IDs are the real template's (tests/format_check.py, strict). It speaks of tools, which the page has none of:
// it also says to answer everyday questions directly
const AGENTS_A1_SYSTEM =
  "You are Intern-A1, a deep research assistant developed by InternAgent Team, Shanghai Artificial Intelligence Laboratory. 你是Intern-A1， 一个由上海人工智能实验室的InternAgent团队开发的深度研究人工智能助手。 You can have natural multi-turn conversations with users on any topic.\n" +
  "\n" +
  "## Daily Chat & Simple Questions\n" +
  "For everyday conversations, greetings, opinions, coding help, factual lookups, definitions, calculations, explanations, and any question you can confidently answer from your knowledge — just respond directly and naturally in the user's language as Intern-A1. Do NOT use any tools for these.\n" +
  "\n" +
  "## Research & Search Questions\n" +
  "Only when the user's question requires up-to-date information, in-depth investigation, multi-source verification, or involves recent events, niche topics, or anything you are uncertain about, use the available tools.\n" +
  "\n" +
  "Research strategy:\n" +
  "- Start with a focused search query to get an overview.\n" +
  "- If the initial search is insufficient, refine your query with more specific terms.\n" +
  "- Stop searching once you have enough information to provide a comprehensive answer. Do not over-research.\n" +
  "\n" +
  "Current date: 2026-07-14";
export const AGENTS_A1_THINKING = `system\n${AGENTS_A1_SYSTEM}<|im_end|>\n<|im_start|>${QWEN35_THINKING}`;
export const AGENTS_A1_AT_ONCE = `${AGENTS_A1_THINKING}\n</think>\n\n`;
// T253: IBM's Granite 4.2. Since T269 the converter's reader reads its chat_template (until then it stopped at the inline
// `a if b else c` and at the empty list `[]`), which gives ?hf= the thinking form; the list has both forms and its own
// BOS, so they stay by hand: one turn as the real Jinja writes it with enable_thinking true (its default) and false, with the empty
// system turn it always writes (the same IDs as transformers' apply_chat_template for tests/format_check.py's
// prompts). As for a Qwen3.5 (T236): the real tokenizer begins a text with no BOS (its post-processor adds none, and
// the template does not write the <s> config.json names), so the BOS here is the format's own first token,
// <|im_start|> (100256), and the formats begin after it: the page sends the very IDs the real template makes. The
// specials are the converter's (the added tokens tokenizer.json does not call special, T143) with <|im_start|> and
// <|im_end|>, in the converter's order. The answer stops at <|im_end|> (100257, the EOS), at the mark of a new turn
// and at <s> (100283)
export const GRANITE_THINKING = "system\n<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n";
export const GRANITE_AT_ONCE = "system\n<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>";
export const granite = { bos: 100256, stop_tokens: [100256, 100257, 100283],
  specials: ["</tool_response>", "<tool_response>", "<|fim_middle|>", "<|fim_prefix|>", "<|fim_suffix|>", "</tool_call>",
    "<|filename|>", "<|im_start|>", "<|reponame|>", "<tool_call>", "<|fim_pad|>", "<|im_end|>", "</think>", "<think>"] };
// T260: Liquid AI's LFM2 and LFM2.5 (convolution layers among attention layers; the engine's arch "lfm2"). The real
// tokenizer begins every text with <|startoftext|> (1), the converter's BOS, and the chat_template writes it first:
// the page sends the very ids the real template makes. The templates of the 230M, the 350M and the 1.2B Instruct call
// macros (parse_content, for the pictures and the tools of a message), which the converter's reader refuses, and the
// 700M's is past it too: one turn by hand, ChatML, as the real Jinja writes it (tests/format_check.py). Without a
// template the converter read, <|im_start|> and <|im_end|> are not in its specials, and a list of the entry's replaces
// the converter's (T221): so all of the converter's are here (the added tokens tokenizer.json does not call special,
// T143: "Mathias" and "python" are among them) with those two, in the converter's order (the longest first). The
// 1.2B JP's template the converter reads, and its entry says nothing. The answer stops at <|im_end|> (7, the EOS) and
// at the BOS. The cards: temperature 0.1 and a repetition penalty of 1.05 (and a top-k of 50, which the page's sampler
// has not; no top-p), the 700M's 0.3 (and a min-p it has not either)
export const lfm25 = { specials: ["<|tool_call_start|>", "<|tool_call_end|>", "<|im_start|>", "<|im_end|>", "</think>", "<think>",
  "Mathias", "python"] };
export const lfm2Old = { specials: ["<|im_start|>", "<|im_end|>", "Mathias", "python"] };
// A Qwen3 whose config.json and tokenizer name no BOS (Ternary Bonsai, CAT-Thinking 8B; a Qwen3 of Qwen's own has
// bos_token_id in config.json): the converter would take token 1, '"'. The BOS here is Qwen3's own, <|endoftext|>
// (151643), as every Qwen3 of the list begins (the real tokenizer puts nothing in front: T131), and the answer stops at
// it and at <|im_end|> (151645). The converter could say this itself (a BOS that is named nowhere, and <|endoftext|> in
// the vocabulary: T248's survey, 7 (4)), at the next CONVERTER: then these lose their options. What it costs is the
// model's: QWEN3_FROM_IM_START (above) says which are worse for it. Where a model of this family is added,
// tests/start_check.mjs and tests/answer_check.mjs say whether this BOS is one it can bear
export const QWEN3_OWN_BOS = { bos: 151643, stop_tokens: [151643, 151645] };
export const harmony = { specials: ["<|channel|>", "<|message|>", "<|start|>", "<|end|>"], stop_tokens: [1, 2, 10, 11, 13] };
export const llmJp = { stop_tokens: [1, 2, 7] };
