// From Hugging Face, converted in the browser: the English classics and the well-known instruct models.
import {
  ASK_JAPANESE, CHATML_AFTER_START, MISTRAL, MISTRAL_V3, SMOLLM3_AT_ONCE, SMOLLM3_THINKING, STORY, ZEPHYR, chatml,
  smollm3,
} from "./formats.js";
import { sampled, thinking } from "./sampling.js";
import { ggufOf, thinkingAndNot } from "./builders.js";

export const HF_CLASSICS = [
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
  // original (gguf.yml's candidates), with the original's vocabulary, config.json and chat template: the formats and
  // the special tokens are the converter's reading, the same IDs as the real template's for tests/format_check.py's
  // prompts. What the page puts in front was measured for each (tests/start_check.mjs, tests/answer_check.mjs: T250's
  // review found a Qwen3 8B 178% worse for it).
  // SmolLM2's base models (the Instruct ones are above): <|endoftext|> in front is 1 to 1.5% better than nothing
  { group: "hf", id: "hf-smollm2-135m", name: "SmolLM2 135M", note: "English · fetches 145 MB (GGUF) → int8 151 MB",
    ...ggufOf("mradermacher/SmolLM2-135M-GGUF", "bf92313aa80eb55329ae75ccce3743101784c802", "SmolLM2-135M.Q8_0.gguf",
      "HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"), download: 144810944,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-smollm2-360m", name: "SmolLM2 360M", note: "English · fetches 386 MB (GGUF) → int8 407 MB",
    ...ggufOf("mradermacher/SmolLM2-360M-GGUF", "630c4866d716e28f45a516ed00e2882149726c13", "SmolLM2-360M.Q8_0.gguf",
      "HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"), download: 386404864,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  // Qwen2.5 Coder above the 0.5B one. The real tokenizer puts nothing in front of a text and the format begins with
  // <|im_start|>; the page begins with the converter's BOS, <|endoftext|>, as the Qwen2.5 of the list do: with it the
  // 1.5B's, the 3B's and the 7B's own answers to five questions are 0.5, 0.4 and 0.2% higher in perplexity, the
  // likeliest next token the same at 97.6, 98.4 and 99.6% of the positions (a Qwen2.5 is no Qwen3 8B), so it stays.
  // The 3B's license is Qwen2.5-3B's, for research
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
  // Hermes 3 (Nous Research) on Llama 3.2 3B, in ChatML: its own Q8_0 GGUF. The real template (apply_chat_template, which
  // the chat servers use) puts nothing in front and begins with <|im_start|>; its card's example code tokenizes a ChatML
  // string, which puts <|begin_of_text|> in front (tokenizer.json's post-processor, and llama.cpp does the same), and the
  // page began with that until T252's review measured what it does to a chat: on 24 answers written by hand it is 6 to 7%
  // higher in perplexity, 13 to 15% for the 13 English ones (four standard errors, with transformers' float32 products and
  // with this engine alike; tests/chat_nll.py, tests/chat_fluency.mjs), and the model's own answers move (KL 0.05 nats a
  // token, the likeliest token the same at 94%), where plain text is 5% better with it. So the BOS is the format's own first
  // token, <|im_start|> (128040), and the format begins after it: the ids are the real template's (a Qwen3 8B's
  // QWEN3_FROM_IM_START, T250's review). It stops at <|im_end|> (config.json's EOS), at <|begin_of_text|> and at the mark
  // of a new turn
  { group: "hf", id: "hf-hermes-3-llama-3.2-3b", name: "Hermes 3 Llama 3.2 3B", note: "answers instructions · English · fetches 3.4 GB (GGUF) → int8 3.6 GB · desktop only",
    ...ggufOf("NousResearch/Hermes-3-Llama-3.2-3B-GGUF", "3cd927095d8cbab12c743f932aa63b6f7bbfa141", "Hermes-3-Llama-3.2-3B.Q8_0.gguf",
      "NousResearch/Hermes-3-Llama-3.2-3B", "7f1a6bec8cdce6551014fd5bbeb4cd8c0f1fbeab"), download: 3421895488,
    conversion: {}, options: { bos: 128040, stop_tokens: [128000, 128039, 128040] }, template: CHATML_AFTER_START, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  // T255: Hugging Face's SmolLM3 3B, a Llama every fourth layer of which RoPE leaves alone (the conversion's options
  // name the layers: unturned). ggml-org's Q8_0 GGUF, which tests/gguf_check.py tensors held to the original
  // (gguf.yml's candidates), with the original's vocabulary and config.json. Six languages, Japanese not among them.
  // Twice, as a Qwen3 is: the two share their weights and a kept conversion. Its card's sampling for either form
  ...[["hf-smollm3-3b-thinking", "SmolLM3 3B (thinking)", "thinks before it answers", SMOLLM3_THINKING],
    ["hf-smollm3-3b", "SmolLM3 3B (no thinking)", "answers at once", SMOLLM3_AT_ONCE]].map(([id, name, what, template]) => ({
    group: "hf", id, name, note: `${what} · English · fetches 3.3 GB (GGUF) → int8 3.5 GB · desktop only`,
    ...ggufOf("ggml-org/SmolLM3-3B-GGUF", "4965cb60b150737b68a0408c36aeefb65078f894", "SmolLM3-Q8_0.gguf",
      "HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1"), download: 3275574624,
    conversion: {}, options: smollm3, shares: ["hf-smollm3-3b-thinking", "hf-smollm3-3b"], template,
    generation: { steps: 0, temperature: 0.6, topp: 0.95, repetition_penalty: 1.0 },
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" })),
  // DeepSeek-R1's larger distills, as the 1.5B above: the tokenizer's own BOS, <｜begin▁of▁sentence｜> (which the
  // real template writes first), and the thought opened by the format. The converter reads both itself now (T143: the
  // BOS tokenizer_config.json names, the template and its special tokens), so these have no options of their own. On
  // plain text the Qwen 7B is 6% worse with that BOS than with nothing in front and 26% worse with config.json's
  // (151643, the end of a sentence); the Llama 8B, a Llama 3.1 with DeepSeek's names for its tokens, 4% better
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
  // Llama 3.1 8B Instruct, as Llama 3.2 above: the original is gated, so the vocabulary and config.json of unsloth's
  // copy. Its template's system turn says "Today Date: 26 Jul 2024" unless a date is passed (Llama 3.2's asks the clock)
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
];
