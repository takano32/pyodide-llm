// From Hugging Face, converted in the browser: the models that write Japanese (the first that were added).
import {
  ARROWPRO, ASK_JAPANESE, GLEIPNIR, OUMUAMUA, CHATML, ELYZA, EUROLLM, EVOLLM, LIGHTBLUE_R1, HARMONY_AFTER_START, HARMONY_THINKING, HARMONY_THINKING_AFTER_START, JAPANESE, LINE_SFT, LLM_JP_INSTRUCT, QWEN25, QWEN3_AT_ONCE_AFTER_START,
  QWEN3_FROM_IM_START, QWEN3_THINKING_AFTER_START, QWEN3_OWN_BOS, CALM2, ELYZA_LLAMA2, RAKUTEN, RAKUTEN_UNTRIMMED, SARASHINA, STABLELM_GAMMA, SWALLOW_MS,
  TINYSWALLOW, TRANSLATE, UNPHI4, UNPHI4_AFTER_START, eurollm, harmonyFromStart, harmonyThinking, harmonyThinkingFromStart, llmJp, sarashina,
} from "./formats.ts";
import { greedy, sampled, thinking } from "./sampling.ts";
import { ggufOf, hf, thinkingAndNot } from "./builders.ts";
import type { Entry } from "./builders.ts";

export const HF_JAPANESE: Entry[] = [
  // Fetched from huggingface.co and converted in the page. Japanese from light to heavy, then English from light
  // to heavy, as everywhere else. What "fetches" says is the download; int8 is what it becomes here.
  { group: "hf", id: "hf-llm-jp-3-150m-instruct3", name: "llm-jp-3 150M instruct3", note: "answers instructions · 日本語 / English · fetches 305 MB → int8 171 MB",
    hf: hf("llm-jp/llm-jp-3-150m-instruct3", "5be263e1a3613cd5c163f41ad828c8de6a2aa6ec"), download: 304649360, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.249: rinna's GPT-2 below and above the small one. No GGUF of either was found: the safetensors
  { group: "hf", id: "hf-japanese-gpt2-xsmall", name: "rinna japanese-gpt2 xsmall", note: "日本語 · fetches 156 MB → int8 42 MB",
    hf: hf("rinna/japanese-gpt2-xsmall", "8e91527b3276e0565154935e84a08bf0137ed99f", "spiece.model"), download: 155892312,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-gpt2-medium", name: "rinna japanese-gpt2 medium", note: "日本語 · fetches 1.4 GB → int8 379 MB",
    hf: hf("rinna/japanese-gpt2-medium", "8ce2399c33e99013a593ea9389378fd86662b9c7", "spiece.model"), download: 1369713080,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-gpt2-small", name: "rinna japanese-gpt2 small", note: "日本語 · fetches 454 MB → int8 130 MB",
    hf: hf("rinna/japanese-gpt2-small", "f7fdefe2941d9629a7b2894564435e0e035df6a6", "spiece.model"), download: 454274094,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-gpt-neox-small", name: "rinna japanese-gpt-neox small", note: "日本語 · fetches 663 MB → int8 193 MB",
    hf: hf("rinna/japanese-gpt-neox-small", "84d18c0fa8c9940a61cfc4e25bd9a5686898bac1", "spiece.model"), download: 663470088,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-qwen2.5-0.5b-instruct", name: "Alibaba Qwen2.5 0.5B Instruct", note: "answers instructions · 日本語 / English · fetches 531 MB (GGUF) → int8 545 MB",
    original: "Qwen/Qwen2.5-0.5B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-0.5B-Instruct-GGUF", revision: "41ba88dbac95fed2528c92514c131d73eb5a174b", weights: "Qwen2.5-0.5B-Instruct-Q8_0.gguf" }, download: 531068480,
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN25, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m", name: "llm-jp-3 440M", note: "日本語 / English · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m", "0bfbf24efdcc5e4c57327e9c52e8cd832637adc2"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m-instruct3", name: "llm-jp-3 440M instruct3", note: "answers instructions · 日本語 / English · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m-instruct3", "a308f143bd5824c4033b3a2efaa1c00afbb3aa9e"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // sarashina2.2 (T.79): the instruct one is published as a single shard with an index (T.78)
  { group: "hf", id: "hf-sarashina2.2-0.5b", name: "SB Intuitions sarashina2.2 0.5B", note: "日本語 / English · fetches 845 MB (GGUF) → int8 0.9 GB",
    original: "sbintuitions/sarashina2.2-0.5b",
    hf: { repo: "mradermacher/sarashina2.2-0.5b-GGUF", revision: "2aed15b94f8c25b7582369f62d9801f3524e4721", weights: "sarashina2.2-0.5b.Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-0.5b", revision: "5fb086c49f49824cfc93f09cc4ed5cd5917bef3d", tokenizer: "tokenizer.model" } },
    download: 845360480, conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-0.5b-instruct", name: "SB Intuitions sarashina2.2 0.5B Instruct", note: "answers instructions · 日本語 · fetches 845 MB (GGUF) → int8 0.9 GB",
    original: "sbintuitions/sarashina2.2-0.5b-instruct-v0.1",
    hf: { repo: "mmnga/sarashina2.2-0.5b-instruct-v0.1-gguf", revision: "5c71186a7a57b8c0325dec70ba2f295303effcd8", weights: "sarashina2.2-0.5b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-0.5b-instruct-v0.1", revision: "e4b9aacc3f644893d0179847946ef6c58d868f29", tokenizer: "tokenizer.model" } },
    download: 845363072, conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.81: a translator from sarashina2.2, Japanese to English and back, asked in its own words and greedy, as its
  // model card runs it (no generation_config: transformers' defaults; a repetition penalty bends a translation)
  { group: "hf", id: "hf-cat-translate-0.8b", name: "CyberAgent CAT-Translate 0.8B", note: "translates 日本語 ⇄ English · fetches 845 MB (GGUF) → int8 0.9 GB",
    original: "cyberagent/CAT-Translate-0.8b",
    hf: { repo: "mmnga-o/CAT-Translate-0.8b-gguf", revision: "c770a1944b55223ccbe766c770882c30c9866445", weights: "CAT-Translate-0.8b-Q8_0.gguf",
          vocabulary: { repo: "cyberagent/CAT-Translate-0.8b", revision: "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9", tokenizer: "tokenizer.model" } }, download: 845363136,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  { group: "hf", id: "hf-llm-jp-3-980m-instruct3", name: "llm-jp-3 980M instruct3", note: "answers instructions · 日本語 / English · fetches 1.1 GB (GGUF) → int8 1.1 GB",
    original: "llm-jp/llm-jp-3-980m-instruct3",
    hf: { repo: "mmnga/llm-jp-3-980m-instruct3-gguf", revision: "5966bc9958a7d313da9b8cb4679cfdf6814b6d42", weights: "llm-jp-3-980m-instruct3-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-3-980m-instruct3", revision: "c079dbf3f88aa2ab702b9696231fc3336c46b1be", tokenizer: "tokenizer.json" } }, download: 1054638240, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.126: a GPT-2 of 1.3B, whose activation is written gelu_fast (the same tanh approximation as gelu_new). Its
  // table of positions holds 1024, which is its context
  { group: "hf", id: "hf-japanese-gpt-1b", name: "rinna japanese-gpt 1B", note: "日本語 · fetches 2.7 GB → int8 1.5 GB",
    hf: hf("rinna/japanese-gpt-1b", "33fc2e4b4e97e229d24a2973073a1361157ecef6", "spiece.model"), download: 2655791788,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T.81 (2026-09-26): the survey's Japanese models that convert as they are. TinySwallow reads its chat template
  // itself (T.73); llm-jp-3.1 has its family's format with the system sentence its model card always passes (its
  // template alone leaves it out)
  { group: "hf", id: "hf-sarashina2.2-1b-instruct", name: "SB Intuitions sarashina2.2 1B Instruct", note: "answers instructions · 日本語 · fetches 1.5 GB (GGUF) → int8 1.6 GB",
    original: "sbintuitions/sarashina2.2-1b-instruct-v0.1",
    hf: { repo: "mmnga/sarashina2.2-1b-instruct-v0.1-gguf", revision: "cd8b02dcc14f38a5101171f304ab269d514fafd4", weights: "sarashina2.2-1b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-1b-instruct-v0.1", revision: "08cf5a8ae579be0fb5a9f802dda8a26acbc94951", tokenizer: "tokenizer.model" } }, download: 1498333056,
    conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-cat-translate-1.4b", name: "CyberAgent CAT-Translate 1.4B", note: "translates 日本語 ⇄ English · fetches 1.5 GB (GGUF) → int8 1.6 GB",
    original: "cyberagent/CAT-Translate-1.4b",
    hf: { repo: "mmnga-o/CAT-Translate-1.4b-gguf", revision: "fe0ab30f267cd976a4f03d218defa6151af88b87", weights: "CAT-Translate-1.4b-Q8_0.gguf",
          vocabulary: { repo: "cyberagent/CAT-Translate-1.4b", revision: "254120945fd9a61278ac2171ab07c831d56838fa", tokenizer: "tokenizer.model" } }, download: 1498333088,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  // T.249: the base model of sarashina2.2 1B, and two Japanese base models of 2023. stockmark's GPT-NeoX rotates a
  // quarter of each head and has no parallel residual; mmnga's Q8_0 GGUF of it (2023) holds the very values, but with
  // q, k and v of a head in turns as Hugging Face has them, where today's llama.cpp writes all of q, then k, then v,
  // which is what the converter undoes (T.136): read so, it wrote "のののの", so the safetensors. LINE's GPT-2, whose
  // table of positions holds 2048: its tokenizer puts nothing in front of a text, and the model writes nonsense after
  // <s> (1), the converter's BOS, and after nothing; after </s> (2) it writes Japanese (transformers on the original
  // says the same of all three), so that is the BOS here
  { group: "hf", id: "hf-sarashina2.2-1b", name: "SB Intuitions sarashina2.2 1B", note: "日本語 / English · fetches 1.5 GB (GGUF) → int8 1.6 GB",
    ...ggufOf("mradermacher/sarashina2.2-1b-GGUF", "9eaeb885b7b61d8ceb274bac21b9df4f42151e23", "sarashina2.2-1b.Q8_0.gguf",
      "sbintuitions/sarashina2.2-1b", "3bb836ad7475ba192926be66651e4730825df7da", "tokenizer.model"), download: 1498330464,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T.598: the rest of SB Intuitions' ladder of base models. sarashina2.2 3B from its safetensors (no Q8_0 of the base
  // one), sarashina2.1 1B (the Sarashina Model NonCommercial License) and sarashina2 7B (heads all with their own keys
  // and values; its original holds PyTorch's shards alone) from Q8_0 GGUFs with the originals' vocabularies
  { group: "hf", id: "hf-sarashina2.1-1b", name: "SB Intuitions sarashina2.1 1B", note: "日本語 / English · fetches 1.5 GB (GGUF) → int8 1.6 GB",
    ...ggufOf("mradermacher/sarashina2.1-1b-GGUF", "9f977c5c0615193a1b5fbcc2279a3e2496850ed1", "sarashina2.1-1b.Q8_0.gguf",
      "sbintuitions/sarashina2.1-1b", "30348c324bc3d1b7b33d9687ca20478b2e5b3dd9", "tokenizer.model"), download: 1498330496,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-3b", name: "SB Intuitions sarashina2.2 3B", note: "日本語 / English · fetches 6.7 GB → int8 3.8 GB",
    original: "sbintuitions/sarashina2.2-3b",
    hf: hf("sbintuitions/sarashina2.2-3b", "9a06e768a2156c21bccb96cf4c99ae4b98755577", "tokenizer.model"), download: 6711252920,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-sarashina2-7b", name: "SB Intuitions sarashina2 7B", note: "日本語 / English · fetches 7.8 GB (GGUF) → int8 8.2 GB",
    ...ggufOf("QuantFactory/sarashina2-7b-GGUF", "83d29cb7330a9067192b1a2986947388da144d1d", "sarashina2-7b.Q8_0.gguf",
      "sbintuitions/sarashina2-7b", "15b989b8a4eb6f68b990787fc1016def7dabf0bb", "tokenizer.model"), download: 7775676416,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-gpt-neox-japanese-1.4b", name: "Stockmark gpt-neox-japanese 1.4B", note: "日本語 · fetches 2.9 GB → int8 1.6 GB",
    hf: hf("stockmark/gpt-neox-japanese-1.4b", "c8f1288a46ac11cf4445dfd18147605d9b692261"), download: 2852015168,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-large-lm-1.7b", name: "LINE japanese-large-lm 1.7B", note: "日本語 · fetches 1.9 GB (GGUF) → int8 1.9 GB",
    ...ggufOf("mmnga/line-corp-japanese-large-lm-1.7b-gguf", "d49108e627b7b6b7b6977184a6b81045b60a8582", "line-corp-japanese-large-lm-1.7b-q8_0.gguf",
      "line-corporation/japanese-large-lm-1.7b", "4288da0a536789f0615c730af0c6cbd9e475a7db", "spiece.model"), download: 1888727168,
    conversion: {}, options: { bos: 2 }, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T.517: LINE's instruction-sft of the same 1.7B (mmnga's Q8_0 GGUF, as the base model's), in the card's format and
  // with the card's sampling (0.7, 0.9, a repetition penalty of 1.1)
  { group: "hf", id: "hf-japanese-large-lm-1.7b-instruction-sft", name: "LINE japanese-large-lm 1.7B instruction-sft", note: "answers instructions · 日本語 · fetches 1.9 GB (GGUF) → int8 1.9 GB",
    ...ggufOf("mmnga/line-corp-japanese-large-lm-1.7b-instruction-sft-gguf", "40b58a8229aee24065e22ba290994ee8066dcbe6", "line-corp-japanese-large-lm-1.7b-instruction-sft-q8_0.gguf",
      "line-corporation/japanese-large-lm-1.7b-instruction-sft", "6817da05bfe027aa82149fb32e1fc5bc2aa97398", "spiece.model"), download: 1888727232,
    conversion: {}, options: { bos: 2 }, generation: sampled(1.1), template: LINE_SFT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.249: EuroLLM, of 35 languages with Japanese among them. Its tokenizer.json is a BPE of sentencepiece's kind, so
  // the tokenizer.model; the special tokens of its format are the converter's (it reads the template too)
  { group: "hf", id: "hf-eurollm-1.7b-instruct", name: "EuroLLM 1.7B Instruct", note: "answers instructions · 日本語 / English and 33 more languages · fetches 1.8 GB (GGUF) → int8 1.9 GB",
    ...ggufOf("mradermacher/EuroLLM-1.7B-Instruct-GGUF", "2951f08f66429c934c8b01a94347161362430808", "EuroLLM-1.7B-Instruct.Q8_0.gguf",
      "utter-project/EuroLLM-1.7B-Instruct", "a25c7fa65fc2a644e6270b8940dbe295b51da681", "tokenizer.model"), download: 1763775712,
    conversion: {}, options: eurollm, generation: sampled(1.1), template: EUROLLM, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-1.8b-instruct3", name: "llm-jp-3 1.8B instruct3", note: "answers instructions · 日本語 / English · fetches 2.0 GB (GGUF) → int8 2.1 GB",
    ...ggufOf("mmnga/llm-jp-3-1.8b-instruct3-gguf", "d908906be3bed7681e4d7269f5c441ea91d2fd56", "llm-jp-3-1.8b-instruct3-Q8_0.gguf",
      "llm-jp/llm-jp-3-1.8b-instruct3", "6b9b0bf051699e7ecffaa5e1166aa5008aa6534f"), download: 1987023136,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-tinyswallow-1.5b-instruct", name: "Sakana AI TinySwallow 1.5B Instruct", note: "answers instructions · 日本語 · fetches 1.6 GB (GGUF) → int8 1.7 GB",
    original: "SakanaAI/TinySwallow-1.5B-Instruct",
    hf: { repo: "SakanaAI/TinySwallow-1.5B-Instruct-GGUF", revision: "38c003aaf8be9d17af11dece1fbabeb873c567fa", weights: "tinyswallow-1.5b-instruct-q8_0.gguf" }, download: 1646573920,
    conversion: {}, options: QWEN3_OWN_BOS, template: TINYSWALLOW, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.600: the base model of RakutenAI 2.0 mini, from its safetensors (no GGUF of it). Its tokenizer carries the chat
  // template of the instruct one, which the converter would read: the plain text instead, with the BOS and stops the
  // converter takes (the pair, as T.369 wants)
  { group: "hf", id: "hf-rakutenai-2.0-mini", name: "RakutenAI 2.0 mini", note: "日本語 / English · fetches 3.1 GB → int8 1.7 GB",
    original: "Rakuten/RakutenAI-2.0-mini",
    hf: hf("Rakuten/RakutenAI-2.0-mini", "3740b0473010eb05d59143858408358bead777ba", "tokenizer.model"), download: 3069389424,
    conversion: {}, options: { bos: 1, stop_tokens: [1, 2] }, generation: sampled(1.1), template: "{prompt}",
    prompt: "これからの流行りは", placeholder: JAPANESE },
  // T.125: a Mistral (a Llama by another name) of 1.5B, Japanese and English; its sliding window of 8192 is past the
  // context of 4096 the page gives it
  { group: "hf", id: "hf-rakutenai-2.0-mini-instruct", name: "RakutenAI 2.0 mini instruct", note: "answers instructions · 日本語 / English · fetches 1.6 GB (GGUF) → int8 1.7 GB",
    original: "Rakuten/RakutenAI-2.0-mini-instruct",
    hf: { repo: "mmnga/RakutenAI-2.0-mini-instruct-gguf", revision: "9bd2900dd7bff11c248fb510d4297b3a88817b76", weights: "RakutenAI-2.0-mini-instruct-Q8_0.gguf",
          vocabulary: { repo: "Rakuten/RakutenAI-2.0-mini-instruct", revision: "6d902489587d324b7d5e201299e4e1a169f3a40b", tokenizer: "tokenizer.model" } }, download: 1631976352,
    conversion: {}, options: {}, generation: sampled(1.1), template: RAKUTEN,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3.1-1.8b-instruct4", name: "llm-jp-3.1 1.8B instruct4", note: "answers instructions · 日本語 / English · fetches 2.0 GB (GGUF) → int8 2.1 GB",
    original: "llm-jp/llm-jp-3.1-1.8b-instruct4",
    hf: { repo: "mmnga/llm-jp-3.1-1.8b-instruct4-gguf", revision: "14ddbab20c68d8befdced79bd9daa3d7a1a29376", weights: "llm-jp-3.1-1.8b-instruct4-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-3.1-1.8b-instruct4", revision: "f19510db409090bb1737f24f868d17c4bdc86c8e", tokenizer: "tokenizer.json" } }, download: 1987023136,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.132: the large ones, in shards (T.105). Past 4 GiB with their forward pass they need a 64-bit memory (T.101),
  // where the browser has one int8 (T.133), else six bits (T.98); the 7 to 8B ones do not fit a 32-bit memory even so
  { group: "hf", id: "hf-qwen2.5-3b-instruct", name: "Alibaba Qwen2.5 3B Instruct", note: "answers instructions · 日本語 / English · fetches 3.3 GB (GGUF) → int8 3.5 GB",
    original: "Qwen/Qwen2.5-3B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-3B-Instruct-GGUF", revision: "f302c64a2269a69fb27b2f9473b362f5bb8e78d8", weights: "Qwen2.5-3B-Instruct-Q8_0.gguf" }, download: 3285476512,
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN25, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-3b-instruct", name: "SB Intuitions sarashina2.2 3B Instruct", note: "answers instructions · 日本語 · fetches 3.6 GB (GGUF) → int8 3.8 GB",
    original: "sbintuitions/sarashina2.2-3b-instruct-v0.1",
    hf: { repo: "mmnga/sarashina2.2-3b-instruct-v0.1-gguf", revision: "31d771319b04032f33e0d9d860f3984ea4812154", weights: "sarashina2.2-3b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-3b-instruct-v0.1", revision: "4f3626fb1b64b3e97c908e67f27b2d627ba2a999", tokenizer: "tokenizer.model" } }, download: 3568393312,
    conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.249: the middle ones. Shisa V2.1's Llama 3.2 3B reads its chat template itself, as Llama 3.2 does (today's date
  // in the system turn); its card gives the model this name (the Llama license asks for it). CAT-Translate 3.3B is
  // the 0.8B's and 1.4B's larger sibling, from sarashina2.2 3B: no Q8_0 GGUF of it was found (only of its beta), so
  // the safetensors, in two shards
  { group: "hf", id: "hf-shisa-v2.1-llama3.2-3b", name: "Llama 3.2 Shisa V2.1 3B", note: "answers instructions · 日本語 / English · fetches 3.4 GB (GGUF) → int8 3.6 GB",
    ...ggufOf("mradermacher/shisa-v2.1-llama3.2-3b-GGUF", "b8cb9e4b9c90657829c0547f091e591a3d24849c", "shisa-v2.1-llama3.2-3b.Q8_0.gguf",
      "shisa-ai/shisa-v2.1-llama3.2-3b", "5f4f59bbe65834daf86a38efd06ff96f7c94c8c3"), download: 3421900096,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.607.6: AXCXEPT's EZO Llama 3.2 3B Instruct dpoE, Llama 3.2 3B Instruct tuned further on Japanese and English. It
  // reads its chat template itself, as Llama 3.2 does (the same template); the sampling of its generation_config.json
  // (0.6, top-p 0.9). mradermacher's Q8_0 with the original's vocabulary
  { group: "hf", id: "hf-ezo-llama-3.2-3b-instruct-dpoe", name: "AXCXEPT EZO Llama 3.2 3B Instruct dpoE", note: "answers instructions · 日本語 / English · fetches 3.4 GB (GGUF) → int8 3.6 GB",
    ...ggufOf("mradermacher/EZO-Llama-3.2-3B-Instruct-dpoE-GGUF", "65ba0474ccc3c07a4ca4c383e5f3c0f174a81f48", "EZO-Llama-3.2-3B-Instruct-dpoE.Q8_0.gguf",
      "AXCXEPT/EZO-Llama-3.2-3B-Instruct-dpoE", "0b374d6e4b55192d833ec5b08ca842aaac4147eb"), download: 3421899936,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.6, topp: 0.9, repetition_penalty: 1.0 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-cat-translate-3.3b", name: "CyberAgent CAT-Translate 3.3B", note: "translates 日本語 ⇄ English · fetches 6.7 GB → int8 3.8 GB",
    hf: hf("cyberagent/CAT-Translate-3.3b", "47e382331d005acd54a42cdf088a76aa88788e0c", "tokenizer.model"), download: 6711252920,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  // T.599: the top of CAT-Translate's ladder, a Mistral of CyberAgent's own with a template of its own (ChatML with a
  // system turn, after <s>), which the converter reads, BOS and all; mradermacher's Q8_0 GGUF
  { group: "hf", id: "hf-cat-translate-7b", name: "CyberAgent CAT-Translate 7B", note: "translates 日本語 ⇄ English · fetches 8.0 GB (GGUF) → int8 8.4 GB",
    ...ggufOf("mradermacher/CAT-Translate-7b-GGUF", "cd7d970bc61657ad36b7f7f392c1bd9527dfe4ad", "CAT-Translate-7b.Q8_0.gguf",
      "cyberagent/CAT-Translate-7b", "e36edfc6858a9a6ba41f8d9fa34427f3d8461eca"), download: 7955662016,
    conversion: {}, options: {}, generation: greedy,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  { group: "hf", id: "hf-llm-jp-3-3.7b-instruct3", name: "llm-jp-3 3.7B instruct3", note: "answers instructions · 日本語 / English · fetches 4.0 GB (GGUF) → int8 4.3 GB",
    ...ggufOf("mmnga/llm-jp-3-3.7b-instruct3-gguf", "7edef5a4f094ec8c1aed1e196c6a544675efbc2f", "llm-jp-3-3.7b-instruct3-Q8_0.gguf",
      "llm-jp/llm-jp-3-3.7b-instruct3", "f5d5466a3316e0c898b4347ece6557a756921220"), download: 4022249856,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.607.7: AXCXEPT's two EZO of llm-jp-3 3.7B instruct, tuned further on instructions made from Japanese Wikipedia and
  // FineWeb (Common for general use, Humanities for the humanities). The chat template of llm-jp-3's instruct (the same
  // text), so its format and stops; the sampling of their cards' code (0.7, top-p 0.95, a penalty of 1.05). ineair's Q8_0
  // with the original's vocabulary
  { group: "hf", id: "hf-llm-jp-3-3.7b-instruct-ezo-common", name: "AXCXEPT llm-jp-3 3.7B instruct EZO Common", note: "answers instructions · 日本語 / English · fetches 4.0 GB (GGUF) → int8 4.3 GB",
    ...ggufOf("ineair/llm-jp-3-3.7b-instruct-EZO-Common-gguf", "31fe00046f1a04f12980d3e1f0d030c85b9a1cd7", "llm-jp-3-3.7b-instruct-EZO-Common-Q8_0.gguf",
      "AXCXEPT/llm-jp-3-3.7b-instruct-EZO-Common", "8cd47420b77e96061f4b1fa1f006eec7b0ff4609"), download: 4022249984,
    conversion: {}, options: llmJp, generation: { steps: 0, temperature: 0.7, topp: 0.95, repetition_penalty: 1.05 }, template: LLM_JP_INSTRUCT,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-3.7b-instruct-ezo-humanities", name: "AXCXEPT llm-jp-3 3.7B instruct EZO Humanities", note: "answers instructions · 日本語 / English · fetches 4.0 GB (GGUF) → int8 4.3 GB",
    ...ggufOf("ineair/llm-jp-3-3.7b-instruct-EZO-Humanities-gguf", "26cd20f98ba4217d5c9ca42b0e1b594371931f5c", "llm-jp-3-3.7b-instruct-EZO-Humanities-Q8_0.gguf",
      "AXCXEPT/llm-jp-3-3.7b-instruct-EZO-Humanities", "ade3da043ec830f0d4d54cfc0e9a91335dfe3f55"), download: 4022249984,
    conversion: {}, options: llmJp, generation: { steps: 0, temperature: 0.7, topp: 0.95, repetition_penalty: 1.05 }, template: LLM_JP_INSTRUCT,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.595: the next step of llm-jp-3's instruct3 ladder, with every head its own keys and values (MHA)
  { group: "hf", id: "hf-llm-jp-3-7.2b-instruct3", name: "llm-jp-3 7.2B instruct3", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.2 GB",
    ...ggufOf("mmnga/llm-jp-3-7.2b-instruct3-gguf", "368515ca8b89be62ee58cbdfbc225c762564c0c2", "llm-jp-3-7.2b-instruct3-Q8_0.gguf",
      "llm-jp/llm-jp-3-7.2b-instruct3", "cdd4c7f3296fdc7785423a864bd9a86ce4c15915"), download: 7750991872,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.517: LINE's 3.6B, a GPT-NeoX (the whole head turns, no parallel residual, the classifier shared with the
  // embedding), and its instruction-sft. The safetensors: mmnga's Q8_0 GGUFs of both (2023-09) hold the very values,
  // but q, k and v of a head in turns as Hugging Face has them (gguf_check.py tensors, run 38731171773), stockmark's
  // case of T.249. Their BOS is <s> (1), not the 1.7B's </s>: on the page's engine the first 1500 tokens of three Japanese
  // articles cost 3.0 to 3.2% more after <s> than after nothing, and 8.2 to 8.6% more after </s> (start_check.ts,
  // run 38736521555; nothing in front is no choice of the page's)
  { group: "hf", id: "hf-japanese-large-lm-3.6b", name: "LINE japanese-large-lm 3.6B", note: "日本語 · fetches 7.2 GB → int8 4.0 GB",
    hf: hf("line-corporation/japanese-large-lm-3.6b", "3ffb24daf746a96f715ff6b5e161c694b510ff03", "spiece.model"), download: 7237638964,
    conversion: {}, options: { bos: 1 }, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-large-lm-3.6b-instruction-sft", name: "LINE japanese-large-lm 3.6B instruction-sft", note: "answers instructions · 日本語 · fetches 7.2 GB → int8 4.0 GB",
    hf: hf("line-corporation/japanese-large-lm-3.6b-instruction-sft", "3d357b288f78beab8a3af7d962cf98d7d93bf72d", "spiece.model"), download: 7237638964,
    conversion: {}, options: { bos: 1 }, generation: sampled(1.1), template: LINE_SFT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-qwen2.5-7b-instruct", name: "Alibaba Qwen2.5 7B Instruct", note: "answers instructions · 日本語 / English · fetches 8.1 GB (GGUF) → int8 8.6 GB",
    original: "Qwen/Qwen2.5-7B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-7B-Instruct-GGUF", revision: "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031", weights: "Qwen2.5-7B-Instruct-Q8_0.gguf" }, download: 8098525888,
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN25, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.593: ELYZA's Qwen2.5 7B Instruct taught to answer without its steps of reasoning: the same template as the
  // Qwen2.5's (its system message and all), so the same BOS and format. The sampling of its card's example (0.6, 0.95)
  { group: "hf", id: "hf-elyza-shortcut-1.0-qwen-7b", name: "ELYZA-Shortcut-1.0 Qwen 7B", note: "answers instructions · 日本語 / English · fetches 8.1 GB (GGUF) → int8 8.6 GB",
    ...ggufOf("mmnga/ELYZA-Shortcut-1.0-Qwen-7B-gguf", "fc1fb43597cab7c9edbe71e41c109fadb83d3815", "ELYZA-Shortcut-1.0-Qwen-7B-Q8_0.gguf",
      "elyza/ELYZA-Shortcut-1.0-Qwen-7B", "d33448648d9b1b7d69cbadb3f1a8d6fb4c8a01ee"), download: 8098525440,
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN25, generation: { steps: 0, temperature: 0.6, topp: 0.95, repetition_penalty: 1.0 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.599: CyberAgent's calm2-7b-chat (Llama, every head with its own keys and values): TheBloke's Q8_0 GGUF of 2023
  // with the original's vocabulary (NFC and one digit a piece), its card's turns (CALM2) and temperature 0.8
  { group: "hf", id: "hf-calm2-7b-chat", name: "CyberAgent calm2 7B chat", note: "answers instructions · 日本語 / English · fetches 7.5 GB (GGUF) → int8 7.9 GB",
    ...ggufOf("TheBloke/calm2-7B-chat-GGUF", "381297d42dd175be349ad277fd3437aebc1d07c9", "calm2-7b-chat.Q8_0.gguf",
      "cyberagent/calm2-7b-chat", "3e110c7624ff7a940badebfd5f9a91569f98cc56"), download: 7450895200,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.8, topp: 1.0, repetition_penalty: 1.0 }, template: CALM2,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.125: Mistral 7B's of Japanese, whose sliding window of 4096 is the page's context
  { group: "hf", id: "hf-rakutenai-7b-chat", name: "RakutenAI 7B chat", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.3 GB",
    original: "Rakuten/RakutenAI-7B-chat",
    hf: { repo: "mradermacher/RakutenAI-7B-chat-GGUF", revision: "d118d842e84cd542a48c6a89464d95b98d37e4e7", weights: "RakutenAI-7B-chat.Q8_0.gguf",
          vocabulary: { repo: "Rakuten/RakutenAI-7B-chat", revision: "7093167c61a0be6161cb68928c939c03fe0ab87d", tokenizer: "tokenizer.model" } }, download: 7835496736,
    conversion: {}, options: {}, generation: sampled(1.1), template: RAKUTEN,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.601: rinna's Llama 3 Youko 8B (Llama 3 8B trained on Japanese, Meta's tokenizer): the instruct one reads its own
  // template (the BOS, then Llama 3's turns with no system turn) and samples as its card (0.6, 0.9, a penalty of 1.1);
  // and the base one. mradermacher's Q8_0 GGUFs with the originals' vocabularies
  { group: "hf", id: "hf-llama-3-youko-8b-instruct", name: "rinna Llama 3 Youko 8B Instruct", note: "answers instructions · 日本語 / English · fetches 8.5 GB (GGUF) → int8 9.0 GB",
    ...ggufOf("mradermacher/llama-3-youko-8b-instruct-GGUF", "3574c58616493522c7d94e2fbe0309342ec4056b", "llama-3-youko-8b-instruct.Q8_0.gguf",
      "rinna/llama-3-youko-8b-instruct", "681390a11903461d2066dbb68ab8bc7f66f3eb67"), download: 8540772032,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.6, topp: 0.9, repetition_penalty: 1.1 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llama-3-youko-8b", name: "rinna Llama 3 Youko 8B", note: "日本語 / English · fetches 8.5 GB (GGUF) → int8 9.0 GB",
    ...ggufOf("mradermacher/llama-3-youko-8b-GGUF", "22d2d9dc444414874ab405308c0d99a5f631dcc3", "llama-3-youko-8b.Q8_0.gguf",
      "rinna/llama-3-youko-8b", "6af890bc6294b8f311e25a1462fcd38554d95b8e"), download: 8540770592,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T.601: rinna's youri 7B chat (Llama 2 7B trained on Japanese, Llama 2's vocabulary: Japanese takes more tokens a
  // letter). No chat template: its card's turns, which are LINE's (LINE_SFT), with its temperature of 0.5
  { group: "hf", id: "hf-youri-7b-chat", name: "rinna youri 7B chat", note: "answers instructions · 日本語 / English · fetches 7.2 GB (GGUF) → int8 7.6 GB",
    ...ggufOf("mradermacher/youri-7b-chat-GGUF", "689eb667771650908f76850f08919148380d109d", "youri-7b-chat.Q8_0.gguf",
      "rinna/youri-7b-chat", "12c3e0e288efbdb525aa98abcab0c92d8445fcbc", "tokenizer.model"), download: 7161090816,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.5, topp: 1.0, repetition_penalty: 1.0 }, template: LINE_SFT,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.605: ELYZA's Llama 2 7B (2023), trained further on Japanese and taught to answer, and the fast one, whose
  // vocabulary has 13,043 more pieces for Japanese. No chat template: their card's turn (ELYZA_LLAMA2). mmnga's GGUFs,
  // the vocabulary of the originals (tokenizer.model; the fast one has tokenizer.json alone). The card's code is greedy;
  // the fast one's generation_config.json samples at 0.6 and a top-p of 0.9
  { group: "hf", id: "hf-elyza-japanese-llama-2-7b-instruct", name: "ELYZA-japanese-Llama-2 7B instruct",
    note: "answers instructions · 日本語 / English · fetches 7.2 GB (GGUF) → int8 7.6 GB",
    ...ggufOf("mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf", "2d708f9c52bde588049a494e95b986f5bedba76f", "ELYZA-japanese-Llama-2-7b-instruct-q8_0.gguf",
      "elyza/ELYZA-japanese-Llama-2-7b-instruct", "48fa08b3098a23d3671e09565499a4cfbaff1923", "tokenizer.model"), download: 7161089920,
    conversion: {}, options: {}, generation: sampled(1.1), template: ELYZA_LLAMA2,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-elyza-japanese-llama-2-7b-fast-instruct", name: "ELYZA-japanese-Llama-2 7B fast instruct",
    note: "answers instructions · 日本語 / English · fetches 7.3 GB (GGUF) → int8 7.7 GB",
    ...ggufOf("mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf", "0f0072a80fd0bebc6a49ee2f970df726513c10c6", "ELYZA-japanese-Llama-2-7b-fast-instruct-q8_0.gguf",
      "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct", "89de33d1ad568855853196802aeaecd799c6586f"), download: 7274877600,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.6, topp: 0.9, repetition_penalty: 1.0 }, template: ELYZA_LLAMA2,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.603: Lightblue's DeepSeek-R1 distills taught to think in the language they are asked in (the converter reads
  // their template and BOS, <｜begin▁of▁sentence｜>, as for DeepSeek's own; their specials are written: LIGHTBLUE_R1;
  // their cards' temperature of 0.5 and repetition penalty of 1.1; mradermacher's GGUFs: mmnga's of the 7B says MIT
  // where the original says Apache-2.0), and ABEJA's Qwen2.5 7B trained on Japanese (Qwen2.5's template, its
  // generation_config.json's sampling)
  { group: "hf", id: "hf-deepseek-r1-qwen-7b-japanese", name: "Lightblue DeepSeek-R1 Distill Qwen 7B Japanese",
    note: "thinks before it answers · 日本語 · fetches 8.1 GB (GGUF) → int8 8.6 GB",
    ...ggufOf("mradermacher/DeepSeek-R1-Distill-Qwen-7B-Japanese-GGUF", "5bf184bf1ef65f5ff18bc92aeb8e2305e76184b3", "DeepSeek-R1-Distill-Qwen-7B-Japanese.Q8_0.gguf",
      "lightblue/DeepSeek-R1-Distill-Qwen-7B-Japanese", "f5f497038d5da4221631058ff1e9b9563fe539d2"), download: 8098525856,
    conversion: {}, options: LIGHTBLUE_R1, generation: { steps: 0, temperature: 0.5, topp: 1.0, repetition_penalty: 1.1 },
    prompt: "17 × 24 はいくつですか？考えてから答えてください。", placeholder: ASK_JAPANESE },
  // T.610.3: CyberAgent's DeepSeek-R1 Distill Qwen 14B trained on Japanese (its int8 passes the 16 GiB a page gets:
  // T.610). mmnga's Q4_K_M with llama.cpp's blocks kept as they are (as Qwen3 14B's); rebuilt: four bits are not held to
  // the original within 1e-3. The converter reads the BOS and the template itself (as DeepSeek's own distills, in
  // hf-classics.ts); the sampling of its generation_config.json
  { group: "hf", id: "hf-deepseek-r1-qwen-14b-japanese", name: "CyberAgent DeepSeek-R1 Distill Qwen 14B Japanese",
    note: "thinks before it answers · 日本語 · 4-bit weights · fetches 9.0 GB (GGUF), kept mostly as it is: 10.8 GB",
    ...ggufOf("mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-gguf", "a8f469dc6c0da43dcb8fe8cc0d6702c7ea51733c",
      "cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-Q4_K_M.gguf",
      "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese", "cc1ebcc1e37521c3f4f69145ce922ca8f01e4cd0"), download: 8988110464,
    conversion: { dtype: "int8", blocks: true }, loaded: 10801599532, rebuilt: true, options: {}, generation: thinking,
    prompt: "17 × 24 はいくつですか？考えてから答えてください。", placeholder: ASK_JAPANESE },
  // T.610.4: ABEJA's Qwen3 14B trained further for agents (a context of 256k by its config's yarn, kept as it says), the
  // two forms of Qwen3 14B's (its template is Qwen3's, enable_thinking and all) with its options and formats; its card
  // says ja alone. mmnga-o's Q4_K_M with llama.cpp's blocks kept as they are; rebuilt as Qwen3 14B's
  ...thinkingAndNot("hf-abeja-qwen3-14b-agentic", "ABEJA Qwen3 14B Agentic 256k v0.1",
    ggufOf("mmnga-o/ABEJA-Qwen3-14B-Agentic-256k-v0.1-gguf", "4bda31b7b9c3c7b170e06d98fcff28dd16d8a8dc",
      "ABEJA-Qwen3-14B-Agentic-256k-v0.1-Q4_K_M.gguf",
      "abeja/ABEJA-Qwen3-14B-Agentic-256k-v0.1", "c18c202ab2deefede0e50eabb8f3ede832eb9cdb"), 9001753696,
    "4-bit weights · fetches 9.0 GB (GGUF), kept mostly as it is: 10.9 GB",
    { languages: "日本語", options: QWEN3_FROM_IM_START, conversion: { dtype: "int8", blocks: true }, loaded: 10869609516, rebuilt: true },
    { thinking: QWEN3_THINKING_AFTER_START, atOnce: QWEN3_AT_ONCE_AFTER_START }),
  // T.610.5: llm-jp-3.1's 13B instruct4, the next step of the 1.8B's ladder (every head its own keys and values, MHA: its
  // int8 passes the 16 GiB a page gets, T.610), with the 1.8B's options, format and sampling. mmnga's Q4_K_M with
  // llama.cpp's blocks kept as they are (half of v and w2 widened to int8: 10.2 GB, where the IQ4_XS keeps every matrix
  // in 7.6 GB but wrote at about half the speed); rebuilt: four bits are not held to the original within 1e-3
  { group: "hf", id: "hf-llm-jp-3.1-13b-instruct4", name: "llm-jp-3.1 13B instruct4",
    note: "answers instructions · 日本語 / English · 4-bit weights · fetches 8.3 GB (GGUF), kept mostly as it is: 10.2 GB",
    ...ggufOf("mmnga/llm-jp-3.1-13b-instruct4-gguf", "7809b3c67450cc2f7dadf0f4fdc8bc56debc96bb", "llm-jp-3.1-13b-instruct4-Q4_K_M.gguf",
      "llm-jp/llm-jp-3.1-13b-instruct4", "da86a55d7a754d1a45bc2d860f8548841a84f42f"), download: 8346102464,
    conversion: { dtype: "int8", blocks: true }, loaded: 10181317676, rebuilt: true, options: llmJp, generation: sampled(1.1),
    template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-deepseek-r1-qwen-1.5b-multilingual", name: "Lightblue DeepSeek-R1 Distill Qwen 1.5B Multilingual",
    note: "thinks before it answers · 日本語 / English and 36 more languages · fetches 1.9 GB (GGUF) → int8 2.0 GB",
    ...ggufOf("mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual-GGUF", "06eaa38aae801e83d2887cb2d06d4f84e8716f0e", "DeepSeek-R1-Distill-Qwen-1.5B-Multilingual.Q8_0.gguf",
      "lightblue/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual", "9c9c7b64e4d7e2154db8ee4cae0e015d7fac5e82"), download: 1894533312,
    conversion: {}, options: LIGHTBLUE_R1, generation: { steps: 0, temperature: 0.5, topp: 1.0, repetition_penalty: 1.1 },
    prompt: "17 × 24 はいくつですか？考えてから答えてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-abeja-qwen2.5-7b-japanese", name: "ABEJA Qwen2.5 7B Japanese v0.1", note: "answers instructions · 日本語 · fetches 8.1 GB (GGUF) → int8 8.6 GB",
    ...ggufOf("mmnga/ABEJA-Qwen2.5-7b-Japanese-v0.1-gguf", "e9edf6f2d3a4890dac56d6dde158e388f7d2690c", "ABEJA-Qwen2.5-7b-Japanese-v0.1-Q8_0.gguf",
      "abeja/ABEJA-Qwen2.5-7b-Japanese-v0.1", "b0027a80acb4052482d4b55499100f568e937644"), download: 8098525568,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.7, topp: 0.8, top_k: 20, repetition_penalty: 1.05 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.604: 2026's community 8Bs. DataPilot's ArrowCanaria (Llama 3.1 Swallow 8B v0.5 taught to chat, then GRPO: its
  // template, its card's sampling) and SousiOmine's Lumi v2 (llm-jp-4 8B base taught ChatML with thinking: its template,
  // whose default system turn it writes when there is none, and the maker's GGUF)
  { group: "hf", id: "hf-arrowcanaria-llama-8b-rl", name: "DataPilot ArrowCanaria Llama 8B RL v0.1",
    note: "answers instructions · 日本語 / English · fetches 8.5 GB (GGUF) → int8 9.0 GB",
    ...ggufOf("mradermacher/ArrowCanaria-Llama-8B-RL-v0.1-GGUF", "ec04d0610d5ef9a7d6c2e01e814023db6b3e7b9d", "ArrowCanaria-Llama-8B-RL-v0.1.Q8_0.gguf",
      "DataPilot/ArrowCanaria-Llama-8B-RL-v0.1", "2b487b422b5a30dbe5f4127cd6b60a15d25ad226"), download: 8540772256,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty: 1.05 },
    prompt: "最近ちょっと疲れてるんだよね。何かリフレッシュできる方法ない？", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-lumi-v2-llm-jp-4-8b", name: "SousiOmine Lumi v2 llm-jp-4 8B",
    note: "thinks before it answers · 日本語 / English · fetches 9.1 GB (GGUF) → int8 9.7 GB",
    ...ggufOf("SousiOmine/Lumi-v2-llm-jp-4-8b-GGUF", "820b4d43d7f89097d77073437cc53f11579793b1", "Lumi-v2-llm-jp-4-8b-Q8_0.gguf",
      "SousiOmine/Lumi-v2-llm-jp-4-8b", "0af15b25d14de36972c39501e883da7f07efbe61"), download: 9132701152,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "17 × 24 はいくつですか？考えてから答えてください。", placeholder: ASK_JAPANESE },
  // T.602: Sakana AI's other models. TinySwallow 1.5B's base one (its tokenizer has Qwen2.5's chat template: the plain
  // text, with the BOS and stops beside it), Karamaru (Llama-3-ELYZA-JP 8B taught to answer in the Japanese of the Edo
  // period: its template's system turn, and its card's sampling) and EvoLLM-JP 7B's two (merges of Mistral 7B's, the
  // card's format by hand: EVOLLM; mradermacher's GGUFs, whose cards name the originals' licenses: mmnga's say unknown).
  // Not TAID-LLM 1.5B: transformers' original as much as the page's never stops and wanders off the question
  { group: "hf", id: "hf-tinyswallow-1.5b", name: "Sakana AI TinySwallow 1.5B", note: "日本語 · fetches 1.6 GB (GGUF) → int8 1.7 GB",
    ...ggufOf("mmnga/TinySwallow-1.5B-gguf", "717ac6b609f797f36cac88055beff8d8520f8d70", "TinySwallow-1.5B-Q8_0.gguf",
      "SakanaAI/TinySwallow-1.5B", "535d1322615c9f8171c3d60b10ef6955cc896fe9"), download: 1646572992,
    conversion: {}, options: QWEN3_OWN_BOS, generation: sampled(1.1), template: "{prompt}", prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-llama-3-karamaru-v1", name: "Sakana AI Llama 3 Karamaru v1", note: "answers in the Japanese of the Edo period · 日本語 · fetches 8.5 GB (GGUF) → int8 9.0 GB",
    ...ggufOf("mradermacher/Llama-3-Karamaru-v1-GGUF", "84ad8f4021eaa476e25d4f943504cddd4260f966", "Llama-3-Karamaru-v1.Q8_0.gguf",
      "SakanaAI/Llama-3-Karamaru-v1", "17411929aa0ef76daa51e7c0d0b986d65c877a58"), download: 8540772576,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.6, topp: 0.9, top_k: 50, repetition_penalty: 1.1 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-evollm-jp-a-v1-7b", name: "Sakana AI EvoLLM-JP-A v1 7B", note: "answers instructions · 日本語 · fetches 7.7 GB (GGUF) → int8 8.2 GB",
    ...ggufOf("mradermacher/EvoLLM-JP-A-v1-7B-GGUF", "028368384460dd87549e789955894c369a93772d", "EvoLLM-JP-A-v1-7B.Q8_0.gguf",
      "SakanaAI/EvoLLM-JP-A-v1-7B", "a8b1356458c9883147c3aeffc1a2e51c99bfe70b", "tokenizer.model"), download: 7695858080,
    conversion: {}, options: {}, generation: sampled(1.1), template: EVOLLM, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-evollm-jp-v1-7b", name: "Sakana AI EvoLLM-JP v1 7B", note: "answers instructions · 日本語 · fetches 7.7 GB (GGUF) → int8 8.2 GB",
    ...ggufOf("mradermacher/EvoLLM-JP-v1-7B-GGUF", "a498d0ca5ec60a8ca0e48cfd177df004884b9351", "EvoLLM-JP-v1-7B.Q8_0.gguf",
      "SakanaAI/EvoLLM-JP-v1-7B", "a7acd2604d3e9f0fca2987d2448ff5dc19778c21", "tokenizer.model"), download: 7695858048,
    conversion: {}, options: {}, generation: sampled(1.1), template: EVOLLM, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.607.1: DataPilot's ArrowPro-7B-KUJIRA, from NTQAI's chatntq-ja-7b-v1.0 (Mistral 7B), made for chat and to hold
  // against prompt injection. No chat template: its card's turn (ARROWPRO) and its sampling (1.0, top-p 0.95). mmnga's
  // Q8_0 with the original's vocabulary
  { group: "hf", id: "hf-arrowpro-7b-kujira", name: "DataPilot ArrowPro 7B KUJIRA", note: "answers instructions · 日本語 · fetches 7.7 GB (GGUF) → int8 8.2 GB",
    ...ggufOf("mmnga/DataPilot-ArrowPro-7B-KUJIRA-gguf", "257e00a4e425aa143494ae12c54b4f9429b713f8", "DataPilot-ArrowPro-7B-KUJIRA-Q8_0.gguf",
      "DataPilot/ArrowPro-7B-KUJIRA", "755e478ac9404cdd551126e20ea055dfa42ddd1d"), download: 7695857376,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 1.0, topp: 0.95, repetition_penalty: 1.0 }, template: ARROWPRO,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.607.2: LocalNovelLLM-project's Vecteus v1, a Mistral 7B made to write novels (merges and LoRA on Japanese and
  // English, NSFW too by its card). Its tokenizer's template (USER: … ASSISTANT:) by the converter, with the BOS; the
  // card samples as transformers' default (1.0, no top-p). mmnga's Q8_0 with the original's vocabulary
  { group: "hf", id: "hf-vecteus-v1", name: "LocalNovelLLM-project Vecteus v1", note: "writes stories · 日本語 / English · fetches 7.7 GB (GGUF) → int8 8.2 GB",
    ...ggufOf("mmnga/Vecteus-v1-gguf", "59df491bb68ce3bc6879c3b292942e4c57274d9b", "Vecteus-v1-Q8_0.gguf",
      "Local-Novel-LLM-project/Vecteus-v1", "d3375ec5f7e149916e44cb0d47c8bda30b986f50", "tokenizer.model"), download: 7695857344,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 1.0, topp: 1.0, repetition_penalty: 1.0 },
    prompt: "猫が主人公の短い物語を書いてください。", placeholder: ASK_JAPANESE },
  // T.607.3: nitky's Oumuamua 7B instruct v2, a merge of Mistral 7B models over Swallow-MS's Japanese vocabulary (42,800).
  // No chat template: its card's turn (OUMUAMUA) and its sampling (temperature 0.1). mradermacher's Q8_0 with the
  // original's vocabulary
  { group: "hf", id: "hf-oumuamua-7b-instruct-v2", name: "nitky Oumuamua 7B instruct v2", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.3 GB",
    ...ggufOf("mradermacher/Oumuamua-7b-instruct-v2-GGUF", "b5ba02aa55bd31d2365354b492500229535bb592", "Oumuamua-7b-instruct-v2.Q8_0.gguf",
      "nitky/Oumuamua-7b-instruct-v2", "315cee7d10fd84a130ab0973942ff76bb1582c49"), download: 7790108960,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.1, topp: 1.0, repetition_penalty: 1.0 }, template: OUMUAMUA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.607.4: umiyuki's Umievo-itr012-Gleipnir-7B, an evolutionary merge of four Japanese Mistral 7B models (Vecteus v1
  // among them). No chat template: its card's turn (GLEIPNIR) and its sampling (1.0, top-p 0.95). mmnga's Q8_0 with the
  // original's vocabulary
  { group: "hf", id: "hf-umievo-gleipnir-7b", name: "umiyuki Umievo itr012 Gleipnir 7B", note: "answers instructions · 日本語 · fetches 7.7 GB (GGUF) → int8 8.2 GB",
    ...ggufOf("mmnga/umiyuki-Umievo-itr012-Gleipnir-7B-gguf", "322aacfb4932771d2e7520a90c3b1558ffb5d7a6", "umiyuki-Umievo-itr012-Gleipnir-7B-Q8_0.gguf",
      "umiyuki/Umievo-itr012-Gleipnir-7B", "5187cc9e441f3d876eda345f52f53eda949c5aee"), download: 7695857408,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 1.0, topp: 0.95, repetition_penalty: 1.0 }, template: GLEIPNIR,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.600: RakutenAI 7B's instruct (the chat's format untrimmed, as its card gives it by hand: no template) and base ones,
  // mradermacher's Q8_0 GGUFs as the chat's
  { group: "hf", id: "hf-rakutenai-7b-instruct", name: "RakutenAI 7B instruct", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.3 GB",
    ...ggufOf("mradermacher/RakutenAI-7B-instruct-GGUF", "3cc5940c04443a36791f79ffd8814cfb90d2ff97", "RakutenAI-7B-instruct.Q8_0.gguf",
      "Rakuten/RakutenAI-7B-instruct", "eb48a29043cda2dba50c088556f0ee411badcf2e", "tokenizer.model"), download: 7835495936,
    conversion: {}, options: {}, generation: sampled(1.1), template: RAKUTEN_UNTRIMMED,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-rakutenai-7b", name: "RakutenAI 7B", note: "日本語 / English · fetches 7.8 GB (GGUF) → int8 8.3 GB",
    ...ggufOf("mradermacher/RakutenAI-7B-GGUF", "1923f28f1f2fc603cd661999c93090a090296a0b", "RakutenAI-7B.Q8_0.gguf",
      "Rakuten/RakutenAI-7B", "eb61524c1ab1a7c656457bd68be4a423371cbacb", "tokenizer.model"), download: 7835495872,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T.600: Stability AI's Japanese StableLM Gamma 7B (a Mistral 7B trained on Japanese), its instruct one in its card's
  // format (STABLELM_GAMMA) and sampling, and its base one; Q8_0 GGUFs with the originals' vocabularies (the instruct one's
  // of 2024-11: TheBloke's of 2023-10 holds the weights before Stability AI's update of 2024-01-24)
  { group: "hf", id: "hf-japanese-stablelm-instruct-gamma-7b", name: "Stability AI Japanese StableLM Instruct Gamma 7B",
    note: "answers instructions · 日本語 · fetches 7.7 GB (GGUF) → int8 8.2 GB",
    ...ggufOf("featherless-ai-quants/stabilityai-japanese-stablelm-instruct-gamma-7b-GGUF", "91ce351b24bd0cf8cd4707d2b62bd20996364050",
      "stabilityai-japanese-stablelm-instruct-gamma-7b-Q8_0.gguf",
      "stabilityai/japanese-stablelm-instruct-gamma-7b", "bb4590d461327a6d63f52a5cdbaa64cfceabbb4a", "tokenizer.model"), download: 7695857856,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 1.0, topp: 0.95, repetition_penalty: 1.0 }, template: STABLELM_GAMMA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-japanese-stablelm-base-gamma-7b", name: "Stability AI Japanese StableLM Base Gamma 7B", note: "日本語 · fetches 7.7 GB (GGUF) → int8 8.2 GB",
    ...ggufOf("mmnga/japanese-stablelm-base-gamma-7b-gguf", "9d14a6b269fcb8190ec9c8b2a29803d24803f800", "japanese-stablelm-base-gamma-7b-q8_0.gguf",
      "stabilityai/japanese-stablelm-base-gamma-7b", "ffc474da0e4f5dbb9db54e96a94a22c081fc7fbf", "tokenizer.model"), download: 7695857280,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-swallow-ms-7b-instruct", name: "Science Tokyo Swallow-MS 7B instruct", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.3 GB",
    original: "tokyotech-llm/Swallow-MS-7b-instruct-v0.1",
    hf: { repo: "mmnga/tokyotech-llm-Swallow-MS-7b-instruct-v0.1-gguf", revision: "cb9ab1c831cfdb8f56371f0ba94806df0fabd518", weights: "tokyotech-llm-Swallow-MS-7b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "tokyotech-llm/Swallow-MS-7b-instruct-v0.1", revision: "008d006f9065e37e39e31bf117ae8689390953e8", tokenizer: "tokenizer.model" } }, download: 7790109536,
    conversion: {}, options: {}, generation: sampled(1.1), template: SWALLOW_MS,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // Swallow's own chat_template is read (T.73): a Japanese system message, and a second BOS before the user's turn,
  // as the real Jinja writes it (the same IDs as the real Jinja and tokenizers, T.132)
  { group: "hf", id: "hf-llama-3.1-swallow-8b-instruct", name: "Science Tokyo Llama 3.1 Swallow 8B Instruct", note: "answers instructions · 日本語 / English · fetches 8.5 GB (GGUF) → int8 9.0 GB",
    original: "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
    hf: { repo: "mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf", revision: "dc00f584312c641eb7af415f7f44bcc4487350cb", weights: "Llama-3.1-Swallow-8B-Instruct-v0.5-Q8_0.gguf",
          vocabulary: { repo: "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5", revision: "b1f8317099a97e790ec872c1225ca155979b4816", tokenizer: "tokenizer.json" } }, download: 8540772672,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-4-8b-instruct", name: "llm-jp-4 8B instruct", note: "answers instructions · 日本語 / English · fetches 9.1 GB (GGUF) → int8 9.7 GB",
    original: "llm-jp/llm-jp-4-8b-instruct",
    hf: { repo: "mmnga-o/llm-jp-4-8b-instruct-gguf", revision: "7ae4da12cee2f109509cb8e1d01cf8a0f1a5fbc1", weights: "llm-jp-4-8b-instruct-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-4-8b-instruct", revision: "098f2b2cf33021eba19a6d3582aa3d071ccc0aff", tokenizer: "tokenizer.json" } }, download: 9132708384,
    conversion: {}, options: harmonyFromStart, generation: sampled(1.1), template: HARMONY_AFTER_START,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.592.2: llm-jp-4 8B thinking, the instruct's sibling that thinks in harmony's analysis channel (T.592.1), with the
  // reasoning effort its template says by default (medium) and the sampling of the entries that think (its card names
  // none)
  { group: "hf", id: "hf-llm-jp-4-8b-thinking", name: "llm-jp-4 8B thinking", note: "thinks before it answers · 日本語 / English · fetches 9.1 GB (GGUF) → int8 9.7 GB",
    original: "llm-jp/llm-jp-4-8b-thinking",
    hf: { repo: "mmnga-o/llm-jp-4-8b-thinking-gguf", revision: "d10eb8c8470f89b2082e793a5ffa086a402a23fc", weights: "llm-jp-4-8b-thinking-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-4-8b-thinking", revision: "fc7c15c262710016c19bcda4372a96ff68875846", tokenizer: "tokenizer.json" } }, download: 9132708672,
    conversion: {}, options: harmonyThinkingFromStart, generation: thinking, template: HARMONY_THINKING_AFTER_START,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.304: llm-jp-4.1 8B thinking, of the same form, format and sampling as 4's. Its makers give a Q4_K_M and a BF16 alone
  // (no Q8_0 yet: the owner's choice of 2026-10-31 is to take the Q4_K_M now with llama.cpp's blocks kept as they are, and the Q8_0
  // when there is one); rebuilt: four bits are not held to the original within 1e-3
  { group: "hf", id: "hf-llm-jp-4.1-8b-thinking", name: "llm-jp-4.1 8B thinking",
    note: "thinks before it answers · 日本語 / English · 4-bit weights · fetches 5.5 GB (GGUF), kept mostly as it is: 6.4 GB",
    original: "llm-jp/llm-jp-4.1-8b-thinking",
    hf: { repo: "llm-jp/llm-jp-4.1-8b-thinking-gguf", revision: "7e48aef0e5bd12168494267608deff5e18302884", weights: "llm-jp-4.1-8b-thinking-Q4_K_M.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-4.1-8b-thinking", revision: "7c19bc90e49b717381e4f474ee95964c05217a28", tokenizer: "tokenizer.json" } }, download: 5499918304,
    conversion: { dtype: "int8", blocks: true }, loaded: 6418383916, rebuilt: true, options: harmonyThinking, generation: thinking,
    template: HARMONY_THINKING, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.250: three more of 8B. ELYZA's Llama 3 with its card's system message. Shisa V2.1's Qwen3 8B, whose template
  // answers at once unless told to think (the other way round from Qwen3's); CAT-Thinking, from Qwen3 Swallow, which
  // thinks in Japanese before it answers (Qwen3's template). Neither of the two names a BOS. CAT-Thinking's options are
  // QWEN3_OWN_BOS and its format the converter's reading of the template. Shisa's sampling is its generation_config.json's;
  // CAT-Thinking's is its card's (0.8 and 0.95, and "to mitigate the probability of repetition, we find
  // repetition_penalty=1.05 or larger to be useful").
  // Shisa's BOS is QWEN3_FROM_IM_START's: with <|endoftext|> in front its own answers to five questions were 67% higher in
  // perplexity, the likeliest next token differed at one in four, and all five answers written again parted within the
  // first 12 tokens (the first hedged where the real IDs answer); plain text 54% higher (the review, tests/answer_check.ts
  // and tests/start_check.ts).
  // CAT-Thinking's GGUF is mmnga-o's: mradermacher's Q8_0 has 256 tensors 0.1 to 0.4% from the nearest of the original's
  // (tests/gguf_check.py tensors: not the pinned original's weights). The original was uploaded in float32 on 2026-05-28
  // and "converted to bf16 from float32" on 2026-05-29 (the pinned revision holds the bf16): mradermacher's GGUF is of
  // 2026-05-28, mmnga-o's of 2026-06-02, which is llama.cpp's Q8_0 of the pinned weights (0 off the nearest reference)
  // T.594: Tanuki-8B, made from scratch by the volunteers of the University of Tokyo's Matsuo Lab (GENIAC), its SFT and
  // DPO. A Llama with llm-jp-3's form of Unigram tokenizer.json; the template (Alpaca in Japanese, with its default system
  // sentence) is read by the converter, jinja2 and the converter's own reader alike (full-width spaces inside its {{ }}).
  // The GGUF is the makers' team's Q8_0 (the card says its GGUFs may be worse: gguf_check.py tensors against the original)
  { group: "hf", id: "hf-tanuki-8b-dpo-v1.0", name: "Matsuo Lab Tanuki 8B", note: "answers instructions · 日本語 / English · fetches 8.0 GB (GGUF) → int8 8.5 GB",
    ...ggufOf("team-hatakeyama-phase2/Tanuki-8B-dpo-v1.0-GGUF", "d3e6436bf6256612bf9c7f8647826b326f7e07e8", "Tanuki-8B-dpo-v1.0-Q8_0.gguf",
      "weblab-GENIAC/Tanuki-8B-dpo-v1.0", "e2f8d17243f709043b18c24d88c3d7853749859c"), download: 7984193888,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.5, topp: 1.0, repetition_penalty: 1.0 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llama-3-elyza-jp-8b", name: "Llama-3-ELYZA-JP 8B", note: "answers instructions · 日本語 / English · fetches 8.5 GB (GGUF) → int8 9.0 GB",
    ...ggufOf("mmnga/Llama-3-ELYZA-JP-8B-gguf", "1a5f8f625074ccb91568fa858402dc43c5170856", "Llama-3-ELYZA-JP-8B-Q8_0.gguf",
      "elyza/Llama-3-ELYZA-JP-8B", "e6c316496ee7d9a11710c50229e8cb39b6b0a4a3"), download: 8540770592,
    conversion: {}, options: {}, generation: sampled(1.1), template: ELYZA, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-shisa-v2.1-qwen3-8b", name: "Shisa V2.1 Qwen3 8B", note: "answers at once · 日本語 / English · fetches 8.7 GB (GGUF) → int8 9.2 GB",
    ...ggufOf("mradermacher/shisa-v2.1-qwen3-8b-GGUF", "9b9187f69adca28b8e2b9490b2c151fcb85c0df6", "shisa-v2.1-qwen3-8b.Q8_0.gguf",
      "shisa-ai/shisa-v2.1-qwen3-8b", "0b0fe7c76dac910510ccd04fc807fdbdbc2fc16e"), download: 8709519392,
    conversion: {}, options: QWEN3_FROM_IM_START, template: QWEN3_AT_ONCE_AFTER_START,
    generation: { steps: 0, temperature: 0.6, topp: 0.95, repetition_penalty: 1.0 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T.607.5: AXCXEPT's Qwen3-EZO-8B-beta, Qwen3 8B trained further for multi-turn chat (its card: MT-Bench 9.08, JMT-Bench
  // 8.87). Its chat template is an older copy of Qwen3's (the same one turn), its generation_config.json Qwen3's thinking
  // sampling: twice as Qwen3 8B is, with the 8B's BOS (<|im_start|>) and formats. mradermacher's Q8_0 with the
  // original's vocabulary
  ...thinkingAndNot("hf-qwen3-ezo-8b-beta", "AXCXEPT Qwen3 EZO 8B beta",
    ggufOf("mradermacher/Qwen3-EZO-8B-beta-GGUF", "1867b17511f6bb9659fc7b6f585cae4a62bdcac7", "Qwen3-EZO-8B-beta.Q8_0.gguf",
      "AXCXEPT/Qwen3-EZO-8B-beta", "c7458182662d2fa15da5e41a17427319e6f265f5"), 8709519008,
    "fetches 8.7 GB (GGUF) → int8 9.2 GB",
    { options: QWEN3_FROM_IM_START }, { thinking: QWEN3_THINKING_AFTER_START, atOnce: QWEN3_AT_ONCE_AFTER_START }),
  // T.607.8: Aratako's Qwen3 8B RP v0.1, Qwen3 8B tuned for role-play in Japanese (its card sets the character in the
  // system turn, which the page does not send: the visitor sets it in the message). Its chat template is the one of
  // Qwen3 EZO 8B beta (an older copy of Qwen3's); its card's code leaves enable_thinking out and answers at once, with no
  // thought, so one entry with that format (the template's own, after <|im_start|>), the 8B's BOS, and the card's
  // temperature 0.5 (no top-p). Aratako's own Q8_0 with the original's vocabulary
  { group: "hf", id: "hf-qwen3-8b-rp-v0.1", name: "Aratako Qwen3 8B RP v0.1", note: "plays a role · 日本語 · fetches 8.7 GB (GGUF) → int8 9.2 GB",
    ...ggufOf("Aratako/Qwen3-8B-RP-v0.1-GGUF", "ff4bdc6cb369912f59bf369d3f6d47677dd83abe", "Qwen3-8B-RP-v0.1-Q8_0.gguf",
      "Aratako/Qwen3-8B-RP-v0.1", "aacfb07d1a7293054ad47ac3de98644ab237b7bd"), download: 8709518336,
    conversion: {}, options: QWEN3_FROM_IM_START, template: QWEN3_THINKING_AFTER_START,
    generation: { steps: 0, temperature: 0.5, topp: 1.0, repetition_penalty: 1.0 },
    prompt: "あなたは魔法学校の新入生「桜」です。桜として、初めて会ったクラスメイトに自己紹介してください。", placeholder: ASK_JAPANESE },
  // T.610.2: Shisa V2.1's 14B, Phi-4 made into a Llama (its int8 passes the 16 GiB a page gets: T.610). mradermacher's
  // Q4_K_M with llama.cpp's blocks kept as they are (as Qwen3 14B's); rebuilt: four bits are not held to the original
  // within 1e-3. Its format by hand, as the converter reads it but with the prompt trimmed (UNPHI4)
  { group: "hf", id: "hf-shisa-v2.1-unphi4-14b", name: "Shisa V2.1 unphi4 14B", note: "answers instructions · 日本語 / English · 4-bit weights · fetches 8.9 GB (GGUF), kept mostly as it is: 10.7 GB",
    ...ggufOf("mradermacher/shisa-v2.1-unphi4-14b-GGUF", "3f3044899c094d10d71e4a2c51c85c77cd330633", "shisa-v2.1-unphi4-14b.Q4_K_M.gguf",
      "shisa-ai/shisa-v2.1-unphi4-14b", "fb2ae1891b6296fc2ecb4c2399912ef2caed4719"), download: 8890306688,
    conversion: { dtype: "int8", blocks: true }, loaded: 10748285996, rebuilt: true, options: UNPHI4, template: UNPHI4_AFTER_START,
    generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-cat-thinking-8b", name: "CyberAgent CAT-Thinking 8B", note: "thinks in Japanese before it answers · 日本語 / English · fetches 8.7 GB (GGUF) → int8 9.2 GB",
    ...ggufOf("mmnga-o/CAT-Thinking-8B-gguf", "d1747e658749aa7a67858914f0a60a2364172c2b", "CAT-Thinking-8B-Q8_0.gguf",
      "cyberagent/CAT-Thinking-8B", "0337f7bcf8d5e6dc08610e205bfe01d566e17669"), download: 8709518944,
    conversion: {}, options: QWEN3_OWN_BOS, template: CHATML,
    generation: { steps: 0, temperature: 0.8, topp: 0.95, repetition_penalty: 1.05 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
];
