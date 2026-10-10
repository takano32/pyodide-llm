// From Hugging Face, converted in the browser: the models that write Japanese (the first that were added).
import {
  ASK_JAPANESE, CHATML, ELYZA, EUROLLM, HARMONY, JAPANESE, LLM_JP_INSTRUCT, QWEN25, QWEN3_AT_ONCE_AFTER_START,
  QWEN3_FROM_IM_START, QWEN3_OWN_BOS, RAKUTEN, SARASHINA, SWALLOW_MS, TINYSWALLOW, TRANSLATE, eurollm, harmony, llmJp,
  sarashina,
} from "./formats.js";
import { greedy, sampled } from "./sampling.js";
import { ggufOf, hf } from "./builders.js";

export const HF_JAPANESE = [
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
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN25, generation: sampled(1.1),
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
    conversion: {}, options: eurollm, generation: sampled(1.1), template: EUROLLM, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-1.8b-instruct3", name: "llm-jp-3 1.8B instruct3", note: "answers instructions · 日本語 · fetches 2.0 GB (GGUF) → int8 2.1 GB · desktop only",
    ...ggufOf("mmnga/llm-jp-3-1.8b-instruct3-gguf", "d908906be3bed7681e4d7269f5c441ea91d2fd56", "llm-jp-3-1.8b-instruct3-Q8_0.gguf",
      "llm-jp/llm-jp-3-1.8b-instruct3", "6b9b0bf051699e7ecffaa5e1166aa5008aa6534f"), download: 1987023136,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-tinyswallow-1.5b-instruct", name: "TinySwallow 1.5B Instruct", note: "answers instructions · 日本語 · fetches 1.6 GB (GGUF) → int8 1.7 GB · desktop only",
    original: "SakanaAI/TinySwallow-1.5B-Instruct",
    hf: { repo: "SakanaAI/TinySwallow-1.5B-Instruct-GGUF", revision: "38c003aaf8be9d17af11dece1fbabeb873c567fa", weights: "tinyswallow-1.5b-instruct-q8_0.gguf" }, download: 1646573920,
    conversion: {}, options: QWEN3_OWN_BOS, template: TINYSWALLOW, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
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
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN25, generation: sampled(1.1),
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
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN25, generation: sampled(1.1),
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
    conversion: {}, options: QWEN3_OWN_BOS, template: CHATML,
    generation: { steps: 0, temperature: 0.8, topp: 0.95, repetition_penalty: 1.05 },
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
];
