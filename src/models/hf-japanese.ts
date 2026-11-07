// From Hugging Face, converted in the browser: the models that write Japanese (the first that were added), below 7B
// (T643.24: from Qwen2.5 7B Instruct on, they are in hf-japanese-large.ts).
import {
  ASKED_JAPANESE, CONTINUES_JAPANESE, EUROLLM, LINE_SFT, LLM_JP_INSTRUCT, QWEN25, QWEN3_OWN_BOS, RAKUTEN,
  RINNA_INSTRUCTION, RINNA_NEWLINE, SARASHINA, TINYSWALLOW, TRANSLATES, eurollm, llmJp, sarashina,
} from "./formats.ts";
import { JAPANESE_SAMPLING, greedy, thinking } from "./sampling.ts";
import { ggufOf, hf } from "./builders.ts";
import type { Entry } from "./builders.ts";

export const HF_JAPANESE: Entry[] = [
  // Fetched from huggingface.co and converted in the page. Japanese from light to heavy, then English from light
  // to heavy, as everywhere else. What "fetches" says is the download; int8 is what it becomes here.
  { group: "hf", id: "hf-llm-jp-3-150m-instruct3", name: "llm-jp-3 150M instruct3", note: "answers instructions · 日本語 / English · fetches 305 MB → int8 171 MB", japanese: true, converted: 171e6,
    hf: hf("llm-jp/llm-jp-3-150m-instruct3", "5be263e1a3613cd5c163f41ad828c8de6a2aa6ec"), download: 304649360, conversion: {}, options: llmJp,
    generation: JAPANESE_SAMPLING, template: LLM_JP_INSTRUCT, ...ASKED_JAPANESE },
  // T249: rinna's GPT-2 below and above the small one. No GGUF of either was found: the safetensors
  { group: "hf", id: "hf-japanese-gpt2-xsmall", name: "rinna japanese-gpt2 xsmall", note: "日本語 · fetches 156 MB → int8 42 MB", japanese: true, converted: 42e6,
    hf: hf("rinna/japanese-gpt2-xsmall", "8e91527b3276e0565154935e84a08bf0137ed99f", "spiece.model"), download: 155892312,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-japanese-gpt2-medium", name: "rinna japanese-gpt2 medium", note: "日本語 · fetches 1.4 GB → int8 379 MB", japanese: true, converted: 379e6,
    hf: hf("rinna/japanese-gpt2-medium", "8ce2399c33e99013a593ea9389378fd86662b9c7", "spiece.model"), download: 1369713080,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-japanese-gpt2-small", name: "rinna japanese-gpt2 small", note: "日本語 · fetches 454 MB → int8 130 MB", japanese: true, converted: 130e6,
    hf: hf("rinna/japanese-gpt2-small", "f7fdefe2941d9629a7b2894564435e0e035df6a6", "spiece.model"), download: 454274094,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-japanese-gpt-neox-small", name: "rinna japanese-gpt-neox small", note: "日本語 · fetches 663 MB → int8 193 MB", japanese: true, converted: 193e6,
    hf: hf("rinna/japanese-gpt-neox-small", "84d18c0fa8c9940a61cfc4e25bd9a5686898bac1", "spiece.model"), download: 663470088,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-qwen2.5-0.5b-instruct", name: "Alibaba Qwen2.5 0.5B Instruct", note: "answers instructions · 日本語 / English · fetches 531 MB (GGUF) → int8 545 MB", japanese: true, converted: 545e6,
    original: "Qwen/Qwen2.5-0.5B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-0.5B-Instruct-GGUF", revision: "41ba88dbac95fed2528c92514c131d73eb5a174b", weights: "Qwen2.5-0.5B-Instruct-Q8_0.gguf" }, download: 531068480,
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN25, generation: JAPANESE_SAMPLING,
    ...ASKED_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m", name: "llm-jp-3 440M", note: "日本語 / English · fetches 0.9 GB → int8 503 MB", japanese: true, converted: 503e6,
    hf: hf("llm-jp/llm-jp-3-440m", "0bfbf24efdcc5e4c57327e9c52e8cd832637adc2"), download: 894519624, conversion: {}, options: llmJp,
    generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m-instruct3", name: "llm-jp-3 440M instruct3", note: "answers instructions · 日本語 / English · fetches 0.9 GB → int8 503 MB", japanese: true, converted: 503e6,
    hf: hf("llm-jp/llm-jp-3-440m-instruct3", "a308f143bd5824c4033b3a2efaa1c00afbb3aa9e"), download: 894519624, conversion: {}, options: llmJp,
    generation: JAPANESE_SAMPLING, template: LLM_JP_INSTRUCT, ...ASKED_JAPANESE },
  // sarashina2.2 (T79): the instruct one is published as a single shard with an index (T78)
  { group: "hf", id: "hf-sarashina2.2-0.5b", name: "SB Intuitions sarashina2.2 0.5B", note: "日本語 / English · fetches 845 MB (GGUF) → int8 0.9 GB", japanese: true, converted: 0.9e9,
    ...ggufOf("mradermacher/sarashina2.2-0.5b-GGUF", "2aed15b94f8c25b7582369f62d9801f3524e4721", "sarashina2.2-0.5b.Q8_0.gguf",
    "sbintuitions/sarashina2.2-0.5b", "5fb086c49f49824cfc93f09cc4ed5cd5917bef3d", "tokenizer.model"),
    download: 845360480, conversion: {}, options: {}, generation: JAPANESE_SAMPLING,
    ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-0.5b-instruct", name: "SB Intuitions sarashina2.2 0.5B Instruct", note: "answers instructions · 日本語 · fetches 845 MB (GGUF) → int8 0.9 GB", japanese: true, converted: 0.9e9,
    ...ggufOf("mmnga/sarashina2.2-0.5b-instruct-v0.1-gguf", "5c71186a7a57b8c0325dec70ba2f295303effcd8", "sarashina2.2-0.5b-instruct-v0.1-Q8_0.gguf",
    "sbintuitions/sarashina2.2-0.5b-instruct-v0.1", "e4b9aacc3f644893d0179847946ef6c58d868f29", "tokenizer.model"),
    download: 845363072, conversion: {}, options: sarashina, generation: JAPANESE_SAMPLING, template: SARASHINA,
    ...ASKED_JAPANESE },
  // T81: a translator from sarashina2.2, Japanese to English and back, asked in its own words and greedy, as its
  // model card runs it (no generation_config: transformers' defaults; a repetition penalty bends a translation)
  { group: "hf", id: "hf-cat-translate-0.8b", name: "CyberAgent CAT-Translate 0.8B", note: "translates 日本語 ⇄ English · fetches 845 MB (GGUF) → int8 0.9 GB", japanese: true, converted: 0.9e9,
    ...ggufOf("mmnga-o/CAT-Translate-0.8b-gguf", "c770a1944b55223ccbe766c770882c30c9866445", "CAT-Translate-0.8b-Q8_0.gguf",
    "cyberagent/CAT-Translate-0.8b", "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9", "tokenizer.model"), download: 845363136,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    ...TRANSLATES },
  { group: "hf", id: "hf-llm-jp-3-980m-instruct3", name: "llm-jp-3 980M instruct3", note: "answers instructions · 日本語 / English · fetches 1.1 GB (GGUF) → int8 1.1 GB", japanese: true, converted: 1.1e9,
    ...ggufOf("mmnga/llm-jp-3-980m-instruct3-gguf", "5966bc9958a7d313da9b8cb4679cfdf6814b6d42", "llm-jp-3-980m-instruct3-Q8_0.gguf",
    "llm-jp/llm-jp-3-980m-instruct3", "c079dbf3f88aa2ab702b9696231fc3336c46b1be"), download: 1054638240, conversion: {}, options: llmJp,
    generation: JAPANESE_SAMPLING, template: LLM_JP_INSTRUCT, ...ASKED_JAPANESE },
  // T126: a GPT-2 of 1.3B, whose activation is written gelu_fast (the same tanh approximation as gelu_new). Its
  // table of positions holds 1024, which is its context
  { group: "hf", id: "hf-japanese-gpt-1b", name: "rinna japanese-gpt 1B", note: "日本語 · fetches 2.7 GB → int8 1.5 GB", japanese: true, converted: 1.5e9,
    hf: hf("rinna/japanese-gpt-1b", "33fc2e4b4e97e229d24a2973073a1361157ecef6", "spiece.model"), download: 2655791788,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  // T81 (2026-09-26): the survey's Japanese models that convert as they are. TinySwallow reads its chat template
  // itself (T73); llm-jp-3.1 has its family's format with the system sentence its model card always passes (its
  // template alone leaves it out)
  { group: "hf", id: "hf-sarashina2.2-1b-instruct", name: "SB Intuitions sarashina2.2 1B Instruct", note: "answers instructions · 日本語 · fetches 1.5 GB (GGUF) → int8 1.6 GB", japanese: true, converted: 1.6e9,
    ...ggufOf("mmnga/sarashina2.2-1b-instruct-v0.1-gguf", "cd8b02dcc14f38a5101171f304ab269d514fafd4", "sarashina2.2-1b-instruct-v0.1-Q8_0.gguf",
    "sbintuitions/sarashina2.2-1b-instruct-v0.1", "08cf5a8ae579be0fb5a9f802dda8a26acbc94951", "tokenizer.model"), download: 1498333056,
    conversion: {}, options: sarashina, generation: JAPANESE_SAMPLING, template: SARASHINA,
    ...ASKED_JAPANESE },
  { group: "hf", id: "hf-cat-translate-1.4b", name: "CyberAgent CAT-Translate 1.4B", note: "translates 日本語 ⇄ English · fetches 1.5 GB (GGUF) → int8 1.6 GB", japanese: true, converted: 1.6e9,
    ...ggufOf("mmnga-o/CAT-Translate-1.4b-gguf", "fe0ab30f267cd976a4f03d218defa6151af88b87", "CAT-Translate-1.4b-Q8_0.gguf",
    "cyberagent/CAT-Translate-1.4b", "254120945fd9a61278ac2171ab07c831d56838fa", "tokenizer.model"), download: 1498333088,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    ...TRANSLATES },
  // T249: the base model of sarashina2.2 1B, and two Japanese base models of 2023. stockmark's GPT-NeoX rotates a
  // quarter of each head and has no parallel residual; mmnga's Q8_0 GGUF of it (2023) holds the very values, but with
  // q, k and v of a head in turns as Hugging Face has them, where today's llama.cpp writes all of q, then k, then v,
  // which is what the converter undoes (T136): read so, it wrote "のののの", so the safetensors. LINE's GPT-2, whose
  // table of positions holds 2048: its tokenizer puts nothing in front of a text, and the model writes nonsense after
  // <s> (1), the converter's BOS, and after nothing; after </s> (2) it writes Japanese (transformers on the original
  // says the same of all three), so that is the BOS here
  { group: "hf", id: "hf-sarashina2.2-1b", name: "SB Intuitions sarashina2.2 1B", note: "日本語 / English · fetches 1.5 GB (GGUF) → int8 1.6 GB", japanese: true, converted: 1.6e9,
    ...ggufOf("mradermacher/sarashina2.2-1b-GGUF", "9eaeb885b7b61d8ceb274bac21b9df4f42151e23", "sarashina2.2-1b.Q8_0.gguf",
      "sbintuitions/sarashina2.2-1b", "3bb836ad7475ba192926be66651e4730825df7da", "tokenizer.model"), download: 1498330464,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  // T598: the rest of SB Intuitions' ladder of base models. sarashina2.2 3B from its safetensors (no Q8_0 of the base
  // one), sarashina2.1 1B (the Sarashina Model NonCommercial License) and sarashina2 7B (heads all with their own keys
  // and values; its original holds PyTorch's shards alone) from Q8_0 GGUFs with the originals' vocabularies
  { group: "hf", id: "hf-sarashina2.1-1b", name: "SB Intuitions sarashina2.1 1B", note: "日本語 / English · fetches 1.5 GB (GGUF) → int8 1.6 GB", japanese: true, converted: 1.6e9,
    ...ggufOf("mradermacher/sarashina2.1-1b-GGUF", "9f977c5c0615193a1b5fbcc2279a3e2496850ed1", "sarashina2.1-1b.Q8_0.gguf",
      "sbintuitions/sarashina2.1-1b", "30348c324bc3d1b7b33d9687ca20478b2e5b3dd9", "tokenizer.model"), download: 1498330496,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-3b", name: "SB Intuitions sarashina2.2 3B", note: "日本語 / English · fetches 6.7 GB → int8 3.8 GB", japanese: true, converted: 3.8e9,
    original: "sbintuitions/sarashina2.2-3b",
    hf: hf("sbintuitions/sarashina2.2-3b", "9a06e768a2156c21bccb96cf4c99ae4b98755577", "tokenizer.model"), download: 6711252920,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-sarashina2-7b", name: "SB Intuitions sarashina2 7B", note: "日本語 / English · fetches 7.8 GB (GGUF) → int8 8.2 GB", japanese: true, converted: 8.2e9,
    ...ggufOf("QuantFactory/sarashina2-7b-GGUF", "83d29cb7330a9067192b1a2986947388da144d1d", "sarashina2-7b.Q8_0.gguf",
      "sbintuitions/sarashina2-7b", "15b989b8a4eb6f68b990787fc1016def7dabf0bb", "tokenizer.model"), download: 7775676416,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-gpt-neox-japanese-1.4b", name: "Stockmark gpt-neox-japanese 1.4B", note: "日本語 · fetches 2.9 GB → int8 1.6 GB", japanese: true, converted: 1.6e9,
    hf: hf("stockmark/gpt-neox-japanese-1.4b", "c8f1288a46ac11cf4445dfd18147605d9b692261"), download: 2852015168,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-japanese-large-lm-1.7b", name: "LINE japanese-large-lm 1.7B", note: "日本語 · fetches 1.9 GB (GGUF) → int8 1.9 GB", japanese: true, converted: 1.9e9,
    ...ggufOf("mmnga/line-corp-japanese-large-lm-1.7b-gguf", "d49108e627b7b6b7b6977184a6b81045b60a8582", "line-corp-japanese-large-lm-1.7b-q8_0.gguf",
      "line-corporation/japanese-large-lm-1.7b", "4288da0a536789f0615c730af0c6cbd9e475a7db", "spiece.model"), download: 1888727168,
    conversion: {}, options: { bos: 2 }, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  // T517: LINE's instruction-sft of the same 1.7B (mmnga's Q8_0 GGUF, as the base model's), in the card's format and
  // with the card's sampling (0.7, 0.9, a repetition penalty of 1.1)
  { group: "hf", id: "hf-japanese-large-lm-1.7b-instruction-sft", name: "LINE japanese-large-lm 1.7B instruction-sft", note: "answers instructions · 日本語 · fetches 1.9 GB (GGUF) → int8 1.9 GB", japanese: true, converted: 1.9e9,
    ...ggufOf("mmnga/line-corp-japanese-large-lm-1.7b-instruction-sft-gguf", "40b58a8229aee24065e22ba290994ee8066dcbe6", "line-corp-japanese-large-lm-1.7b-instruction-sft-q8_0.gguf",
      "line-corporation/japanese-large-lm-1.7b-instruction-sft", "6817da05bfe027aa82149fb32e1fc5bc2aa97398", "spiece.model"), download: 1888727232,
    conversion: {}, options: { bos: 2 }, generation: JAPANESE_SAMPLING, template: LINE_SFT, ...ASKED_JAPANESE },
  // T249: EuroLLM, of 35 languages with Japanese among them. Its tokenizer.json is a BPE of sentencepiece's kind, so
  // the tokenizer.model; the special tokens of its format are the converter's (it reads the template too)
  { group: "hf", id: "hf-eurollm-1.7b-instruct", name: "EuroLLM 1.7B Instruct", note: "answers instructions · 日本語 / English and 33 more languages · fetches 1.8 GB (GGUF) → int8 1.9 GB", japanese: true, converted: 1.9e9,
    ...ggufOf("mradermacher/EuroLLM-1.7B-Instruct-GGUF", "2951f08f66429c934c8b01a94347161362430808", "EuroLLM-1.7B-Instruct.Q8_0.gguf",
      "utter-project/EuroLLM-1.7B-Instruct", "a25c7fa65fc2a644e6270b8940dbe295b51da681", "tokenizer.model"), download: 1763775712,
    conversion: {}, options: eurollm, generation: JAPANESE_SAMPLING, template: EUROLLM, ...ASKED_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-1.8b-instruct3", name: "llm-jp-3 1.8B instruct3", note: "answers instructions · 日本語 / English · fetches 2.0 GB (GGUF) → int8 2.1 GB", japanese: true, converted: 2.1e9,
    ...ggufOf("mmnga/llm-jp-3-1.8b-instruct3-gguf", "d908906be3bed7681e4d7269f5c441ea91d2fd56", "llm-jp-3-1.8b-instruct3-Q8_0.gguf",
      "llm-jp/llm-jp-3-1.8b-instruct3", "6b9b0bf051699e7ecffaa5e1166aa5008aa6534f"), download: 1987023136,
    conversion: {}, options: llmJp, generation: JAPANESE_SAMPLING, template: LLM_JP_INSTRUCT, ...ASKED_JAPANESE },
  { group: "hf", id: "hf-tinyswallow-1.5b-instruct", name: "Sakana AI TinySwallow 1.5B Instruct", note: "answers instructions · 日本語 · fetches 1.6 GB (GGUF) → int8 1.7 GB", japanese: true, converted: 1.7e9,
    original: "SakanaAI/TinySwallow-1.5B-Instruct",
    hf: { repo: "SakanaAI/TinySwallow-1.5B-Instruct-GGUF", revision: "38c003aaf8be9d17af11dece1fbabeb873c567fa", weights: "tinyswallow-1.5b-instruct-q8_0.gguf" }, download: 1646573920,
    conversion: {}, options: QWEN3_OWN_BOS, template: TINYSWALLOW, generation: JAPANESE_SAMPLING, ...ASKED_JAPANESE },
  // T600: the base model of RakutenAI 2.0 mini, from its safetensors (no GGUF of it). Its tokenizer carries the chat
  // template of the instruct one, which the converter would read: the plain text instead, with the BOS and stops the
  // converter takes (the pair, as T369 wants)
  { group: "hf", id: "hf-rakutenai-2.0-mini", name: "RakutenAI 2.0 mini", note: "日本語 / English · fetches 3.1 GB → int8 1.7 GB", japanese: true, converted: 1.7e9,
    original: "Rakuten/RakutenAI-2.0-mini",
    hf: hf("Rakuten/RakutenAI-2.0-mini", "3740b0473010eb05d59143858408358bead777ba", "tokenizer.model"), download: 3069389424,
    conversion: {}, options: { bos: 1, stop_tokens: [1, 2] }, generation: JAPANESE_SAMPLING, template: "{prompt}",
    ...CONTINUES_JAPANESE },
  // T125: a Mistral (a Llama by another name) of 1.5B, Japanese and English; its sliding window of 8192 is past the
  // context of 4096 the page gives it
  { group: "hf", id: "hf-rakutenai-2.0-mini-instruct", name: "RakutenAI 2.0 mini instruct", note: "answers instructions · 日本語 / English · fetches 1.6 GB (GGUF) → int8 1.7 GB", japanese: true, converted: 1.7e9,
    ...ggufOf("mmnga/RakutenAI-2.0-mini-instruct-gguf", "9bd2900dd7bff11c248fb510d4297b3a88817b76", "RakutenAI-2.0-mini-instruct-Q8_0.gguf",
    "Rakuten/RakutenAI-2.0-mini-instruct", "6d902489587d324b7d5e201299e4e1a169f3a40b", "tokenizer.model"), download: 1631976352,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, template: RAKUTEN,
    ...ASKED_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3.1-1.8b-instruct4", name: "llm-jp-3.1 1.8B instruct4", note: "answers instructions · 日本語 / English · fetches 2.0 GB (GGUF) → int8 2.1 GB", japanese: true, converted: 2.1e9,
    ...ggufOf("mmnga/llm-jp-3.1-1.8b-instruct4-gguf", "14ddbab20c68d8befdced79bd9daa3d7a1a29376", "llm-jp-3.1-1.8b-instruct4-Q8_0.gguf",
    "llm-jp/llm-jp-3.1-1.8b-instruct4", "f19510db409090bb1737f24f868d17c4bdc86c8e"), download: 1987023136,
    conversion: {}, options: llmJp, generation: JAPANESE_SAMPLING, template: LLM_JP_INSTRUCT, ...ASKED_JAPANESE },
  // T132: the large ones, in shards (T105). Past 4 GiB with their forward pass they need a 64-bit memory (T101),
  // where the browser has one int8 (T133), else six bits (T98); the 7 to 8B ones do not fit a 32-bit memory even so
  { group: "hf", id: "hf-qwen2.5-3b-instruct", name: "Alibaba Qwen2.5 3B Instruct", note: "answers instructions · 日本語 / English · fetches 3.3 GB (GGUF) → int8 3.5 GB", japanese: true, converted: 3.5e9,
    original: "Qwen/Qwen2.5-3B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-3B-Instruct-GGUF", revision: "f302c64a2269a69fb27b2f9473b362f5bb8e78d8", weights: "Qwen2.5-3B-Instruct-Q8_0.gguf" }, download: 3285476512,
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN25, generation: JAPANESE_SAMPLING,
    ...ASKED_JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-3b-instruct", name: "SB Intuitions sarashina2.2 3B Instruct", note: "answers instructions · 日本語 · fetches 3.6 GB (GGUF) → int8 3.8 GB", japanese: true, converted: 3.8e9,
    ...ggufOf("mmnga/sarashina2.2-3b-instruct-v0.1-gguf", "31d771319b04032f33e0d9d860f3984ea4812154", "sarashina2.2-3b-instruct-v0.1-Q8_0.gguf",
    "sbintuitions/sarashina2.2-3b-instruct-v0.1", "4f3626fb1b64b3e97c908e67f27b2d627ba2a999", "tokenizer.model"), download: 3568393312,
    conversion: {}, options: sarashina, generation: JAPANESE_SAMPLING, template: SARASHINA,
    ...ASKED_JAPANESE },
  // T249: the middle ones. Shisa V2.1's Llama 3.2 3B reads its chat template itself, as Llama 3.2 does (today's date
  // in the system turn); its card gives the model this name (the Llama license asks for it). CAT-Translate 3.3B is
  // the 0.8B's and 1.4B's larger sibling, from sarashina2.2 3B: no Q8_0 GGUF of it was found (only of its beta), so
  // the safetensors, in two shards
  { group: "hf", id: "hf-shisa-v2.1-llama3.2-3b", name: "Llama 3.2 Shisa V2.1 3B", note: "answers instructions · 日本語 / English · fetches 3.4 GB (GGUF) → int8 3.6 GB", japanese: true, converted: 3.6e9,
    ...ggufOf("mradermacher/shisa-v2.1-llama3.2-3b-GGUF", "b8cb9e4b9c90657829c0547f091e591a3d24849c", "shisa-v2.1-llama3.2-3b.Q8_0.gguf",
      "shisa-ai/shisa-v2.1-llama3.2-3b", "5f4f59bbe65834daf86a38efd06ff96f7c94c8c3"), download: 3421900096,
    conversion: {}, options: {}, generation: JAPANESE_SAMPLING, ...ASKED_JAPANESE },
  // T607.6: AXCXEPT's EZO Llama 3.2 3B Instruct dpoE, Llama 3.2 3B Instruct tuned further on Japanese and English. It
  // reads its chat template itself, as Llama 3.2 does (the same template); the sampling of its generation_config.json
  // (0.6, top-p 0.9). mradermacher's Q8_0 with the original's vocabulary
  { group: "hf", id: "hf-ezo-llama-3.2-3b-instruct-dpoe", name: "AXCXEPT EZO Llama 3.2 3B Instruct dpoE", note: "answers instructions · 日本語 / English · fetches 3.4 GB (GGUF) → int8 3.6 GB", japanese: true, converted: 3.6e9,
    ...ggufOf("mradermacher/EZO-Llama-3.2-3B-Instruct-dpoE-GGUF", "65ba0474ccc3c07a4ca4c383e5f3c0f174a81f48", "EZO-Llama-3.2-3B-Instruct-dpoE.Q8_0.gguf",
      "AXCXEPT/EZO-Llama-3.2-3B-Instruct-dpoE", "0b374d6e4b55192d833ec5b08ca842aaac4147eb"), download: 3421899936,
    conversion: {}, options: {}, generation: { steps: 0, temperature: 0.6, topp: 0.9, repetition_penalty: 1.0 },
    ...ASKED_JAPANESE },
  { group: "hf", id: "hf-cat-translate-3.3b", name: "CyberAgent CAT-Translate 3.3B", note: "translates 日本語 ⇄ English · fetches 6.7 GB → int8 3.8 GB", japanese: true, converted: 3.8e9,
    hf: hf("cyberagent/CAT-Translate-3.3b", "47e382331d005acd54a42cdf088a76aa88788e0c", "tokenizer.model"), download: 6711252920,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    ...TRANSLATES },
  // T599: the top of CAT-Translate's ladder, a Mistral of CyberAgent's own with a template of its own (ChatML with a
  // system turn, after <s>), which the converter reads, BOS and all; mradermacher's Q8_0 GGUF
  { group: "hf", id: "hf-cat-translate-7b", name: "CyberAgent CAT-Translate 7B", note: "translates 日本語 ⇄ English · fetches 8.0 GB (GGUF) → int8 8.4 GB", japanese: true, converted: 8.4e9,
    ...ggufOf("mradermacher/CAT-Translate-7b-GGUF", "cd7d970bc61657ad36b7f7f392c1bd9527dfe4ad", "CAT-Translate-7b.Q8_0.gguf",
      "cyberagent/CAT-Translate-7b", "e36edfc6858a9a6ba41f8d9fa34427f3d8461eca"), download: 7955662016,
    conversion: {}, options: {}, generation: greedy,
    ...TRANSLATES },
  { group: "hf", id: "hf-llm-jp-3-3.7b-instruct3", name: "llm-jp-3 3.7B instruct3", note: "answers instructions · 日本語 / English · fetches 4.0 GB (GGUF) → int8 4.3 GB", japanese: true, converted: 4.3e9,
    ...ggufOf("mmnga/llm-jp-3-3.7b-instruct3-gguf", "7edef5a4f094ec8c1aed1e196c6a544675efbc2f", "llm-jp-3-3.7b-instruct3-Q8_0.gguf",
      "llm-jp/llm-jp-3-3.7b-instruct3", "f5d5466a3316e0c898b4347ece6557a756921220"), download: 4022249856,
    conversion: {}, options: llmJp, generation: JAPANESE_SAMPLING, template: LLM_JP_INSTRUCT, ...ASKED_JAPANESE },
  // T607.7: AXCXEPT's two EZO of llm-jp-3 3.7B instruct, tuned further on instructions made from Japanese Wikipedia and
  // FineWeb (Common for general use, Humanities for the humanities). The chat template of llm-jp-3's instruct (the same
  // text), so its format and stops; the sampling of their cards' code (0.7, top-p 0.95, a penalty of 1.05). ineair's Q8_0
  // with the original's vocabulary
  { group: "hf", id: "hf-llm-jp-3-3.7b-instruct-ezo-common", name: "AXCXEPT llm-jp-3 3.7B instruct EZO Common", note: "answers instructions · 日本語 / English · fetches 4.0 GB (GGUF) → int8 4.3 GB", japanese: true, converted: 4.3e9,
    ...ggufOf("ineair/llm-jp-3-3.7b-instruct-EZO-Common-gguf", "31fe00046f1a04f12980d3e1f0d030c85b9a1cd7", "llm-jp-3-3.7b-instruct-EZO-Common-Q8_0.gguf",
      "AXCXEPT/llm-jp-3-3.7b-instruct-EZO-Common", "8cd47420b77e96061f4b1fa1f006eec7b0ff4609"), download: 4022249984,
    conversion: {}, options: llmJp, generation: { steps: 0, temperature: 0.7, topp: 0.95, repetition_penalty: 1.05 }, template: LLM_JP_INSTRUCT,
    ...ASKED_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-3.7b-instruct-ezo-humanities", name: "AXCXEPT llm-jp-3 3.7B instruct EZO Humanities", note: "answers instructions · 日本語 / English · fetches 4.0 GB (GGUF) → int8 4.3 GB", japanese: true, converted: 4.3e9,
    ...ggufOf("ineair/llm-jp-3-3.7b-instruct-EZO-Humanities-gguf", "26cd20f98ba4217d5c9ca42b0e1b594371931f5c", "llm-jp-3-3.7b-instruct-EZO-Humanities-Q8_0.gguf",
      "AXCXEPT/llm-jp-3-3.7b-instruct-EZO-Humanities", "ade3da043ec830f0d4d54cfc0e9a91335dfe3f55"), download: 4022249984,
    conversion: {}, options: llmJp, generation: { steps: 0, temperature: 0.7, topp: 0.95, repetition_penalty: 1.05 }, template: LLM_JP_INSTRUCT,
    ...ASKED_JAPANESE },
  // T595: the next step of llm-jp-3's instruct3 ladder, with every head its own keys and values (MHA)
  { group: "hf", id: "hf-llm-jp-3-7.2b-instruct3", name: "llm-jp-3 7.2B instruct3", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.2 GB", japanese: true, converted: 8.2e9,
    ...ggufOf("mmnga/llm-jp-3-7.2b-instruct3-gguf", "368515ca8b89be62ee58cbdfbc225c762564c0c2", "llm-jp-3-7.2b-instruct3-Q8_0.gguf",
      "llm-jp/llm-jp-3-7.2b-instruct3", "cdd4c7f3296fdc7785423a864bd9a86ce4c15915"), download: 7750991872,
    conversion: {}, options: llmJp, generation: JAPANESE_SAMPLING, template: LLM_JP_INSTRUCT, ...ASKED_JAPANESE },
  // T517: LINE's 3.6B, a GPT-NeoX (the whole head turns, no parallel residual, the classifier shared with the
  // embedding), and its instruction-sft. The safetensors: mmnga's Q8_0 GGUFs of both (2023-09) hold the very values,
  // but q, k and v of a head in turns as Hugging Face has them (gguf_check.py tensors, run 38731171773), stockmark's
  // case of T249. Their BOS is <s> (1), not the 1.7B's </s>: on the page's engine the first 1500 tokens of three Japanese
  // articles cost 3.0 to 3.2% more after <s> than after nothing, and 8.2 to 8.6% more after </s> (start_check.ts,
  // run 38736521555; nothing in front is no choice of the page's)
  { group: "hf", id: "hf-japanese-large-lm-3.6b", name: "LINE japanese-large-lm 3.6B", note: "日本語 · fetches 7.2 GB → int8 4.0 GB", japanese: true, converted: 4.0e9,
    hf: hf("line-corporation/japanese-large-lm-3.6b", "3ffb24daf746a96f715ff6b5e161c694b510ff03", "spiece.model"), download: 7237638964,
    conversion: {}, options: { bos: 1 }, generation: JAPANESE_SAMPLING, ...CONTINUES_JAPANESE },
  { group: "hf", id: "hf-japanese-large-lm-3.6b-instruction-sft", name: "LINE japanese-large-lm 3.6B instruction-sft", note: "answers instructions · 日本語 · fetches 7.2 GB → int8 4.0 GB", japanese: true, converted: 4.0e9,
    hf: hf("line-corporation/japanese-large-lm-3.6b-instruction-sft", "3d357b288f78beab8a3af7d962cf98d7d93bf72d", "spiece.model"), download: 7237638964,
    conversion: {}, options: { bos: 1 }, generation: JAPANESE_SAMPLING, template: LINE_SFT, ...ASKED_JAPANESE },
  // T617: rinna's japanese-gpt-neox 3.6B instruction-ppo (a GPT-NeoX, its spiece.model; the safetensors: mmnga's GGUF of
  // 2023 may have attn_qkv in Hugging Face's order). Its card's turns, with "<NL>" for a newline both ways
  // (RINNA_NEWLINE), and its card's sampling (0.7 and a repetition penalty of 1.1, no top-p)
  { group: "hf", id: "hf-japanese-gpt-neox-3.6b-instruction-ppo", name: "rinna japanese-gpt-neox 3.6B instruction-ppo", note: "answers instructions · 日本語 · fetches 7.4 GB → int8 4.1 GB", japanese: true, converted: 4.1e9,
    hf: hf("rinna/japanese-gpt-neox-3.6b-instruction-ppo", "018e1f11cfe7ceeaad26c5c6f7dc570281516a51", "spiece.model"), download: 7365554624,
    conversion: {}, options: RINNA_NEWLINE, generation: { steps: 0, temperature: 0.7, topp: 1.0, repetition_penalty: 1.1 },
    template: RINNA_INSTRUCTION, ...ASKED_JAPANESE },
  // T614: PFN's PLaMo 3 610M Fin Instruct, a PLaMo 3 (plamo3: T649) trained further on finance in Japanese, its card
  // says ja alone. The original's safetensors: mradermacher's GGUFs of it say no yarn of the full layers' RoPE, which
  // its config.json has (the converter refuses them). As PLaMo 3 Translate's, PFN in front and PLaMo in the name and
  // "Built with PLaMo" in the note and About (the PLaMo Community License's Article 4: the owner's T649 (1)). The
  // converter reads its format (BOS 1, stops 1, 2 and 16; tests/format_check.py 9/9). Its card is greedy, which loops
  // in the thought on 4 of answers.ts's 12 questions to 1500 tokens (its marks are three tokens each: the engine's
  // budget closes none, T520): Qwen3's thinking set closed all 8 of those, with two seeds each
  { group: "hf", id: "hf-plamo-3-610m-fin-instruct", name: "PFN PLaMo 3 610M Fin Instruct",
    note: "thinks before it answers · 日本語 · fetches 1.8 GB → int8 1.0 GB, Built with PLaMo", japanese: true, thinks: true, converted: 1.0e9,
    hf: hf("pfnet/plamo-3-610m-fin-instruct", "a000b59209f1688322c0804009dc9f3fd67e62c1", "tokenizer.jsonl"), download: 1780709128,
    conversion: {}, options: {}, generation: thinking, ...ASKED_JAPANESE },
];
