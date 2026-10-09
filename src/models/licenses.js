// The license of every source of the list, as its model card names it (tests/models-check.mjs holds the list to it).

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
// T260: Liquid AI's own license, as the LICENSE file of every repository names itself (the cards say "other", lfm1.0)
const LFM_OPEN = "LFM Open License v1.0";
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
  "HuggingFaceTB/SmolLM2-1.7B-Instruct": APACHE, "HuggingFaceTB/SmolLM3-3B": APACHE, "ggml-org/SmolLM3-3B-GGUF": APACHE,
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
  // T335: the four cards say apache-2.0, and their LICENSE keeps "Copyright 2026 Alibaba Cloud" of the Qwen3.5 they
  // were trained from (config.json's modification_notice says what TokenRhythm changed)
  "TokenRhythm/NeoHorse-1-4B": APACHE, "TokenRhythm/NeoHorse-1-4B-GGUF": APACHE,
  "TokenRhythm/NeoHorse-1-9B": APACHE, "TokenRhythm/NeoHorse-1-9B-GGUF": APACHE,
  // T337: both cards say apache-2.0
  "InternScience/Agents-A1-4B": APACHE, "InternScience/Agents-A1-4B-Q8_0-GGUF": APACHE,
  // T233: both cards say apache-2.0. The GGUF's NOTICE.txt: "copyright 2026-present Prism ML, Inc. ... built from
  // Qwen3.8-27B, Copyright 2026 Alibaba Cloud ... Apache 2.0", and asks for "Created using Bonsai by Prism ML." where
  // it is deployed, as the smaller Ternary Bonsai's does
  "prism-ml/Ternary-Bonsai-2-27B-gguf": APACHE, "Qwen/Qwen3.8-27B": APACHE,
  // T253: the four cards say apache-2.0
  "ibm-granite/granite-4.2-3b": APACHE, "ibm-granite/granite-4.2-3b-GGUF": APACHE,
  "ibm-granite/granite-4.2-8b": APACHE, "ibm-granite/granite-4.2-8b-GGUF": APACHE,
  // T254: the four cards say apache-2.0
  "openbmb/MiniCPM5-1B": APACHE, "openbmb/MiniCPM5-1B-GGUF": APACHE,
  "openbmb/MiniCPM5-2B": APACHE, "openbmb/MiniCPM5-2B-GGUF": APACHE,
  // T260: the nine cards say license other, lfm1.0, and link the LICENSE file of their repository
  "LiquidAI/LFM2.5-230M": LFM_OPEN, "LiquidAI/LFM2.5-230M-GGUF": LFM_OPEN,
  "LiquidAI/LFM2.5-350M": LFM_OPEN, "LiquidAI/LFM2.5-350M-GGUF": LFM_OPEN,
  "LiquidAI/LFM2-700M": LFM_OPEN, "LiquidAI/LFM2-700M-GGUF": LFM_OPEN,
  "LiquidAI/LFM2.5-1.2B-Instruct": LFM_OPEN, "LiquidAI/LFM2.5-1.2B-Instruct-GGUF": LFM_OPEN,
  "LiquidAI/LFM2.5-1.2B-JP-202606": LFM_OPEN, "LiquidAI/LFM2.5-1.2B-JP-202606-GGUF": LFM_OPEN,
};
