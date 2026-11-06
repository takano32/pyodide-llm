// The license of every source of the list, as its model card names it (tests/models-check.ts holds the list to it).

// Where each model comes from, and under which license (T87). A model of this site names its source with
// `source`; one fetched from Hugging Face is its `hf.repo`. The page lists them from here, and
// tests/models-check.ts fails when a model has no license, so a new model cannot be added without one.
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
// T604: ArrowCanaria's card: "Llama 3.1 Community License + Gemma Terms of use License" (it is made from Swallow's)
const ARROWCANARIA = "Llama 3.1 Community License and Gemma Terms of Use";
// T250: ELYZA's card says "Meta Llama 3 Community License" under License (its metadata: llama3)
const LLAMA_3 = "Meta Llama 3 Community License";
// T601: rinna's youri (its card's metadata: llama2, Llama 2's community license)
const LLAMA_2 = "Llama 2 Community License";
// T602: EvoLLM-JP v1 7B's card: "MICROSOFT RESEARCH LICENSE TERMS (due to the inclusion of the WizardMath model)"
const MS_RESEARCH = "Microsoft Research License Terms";
// T251: the cards of GPT-2's medium, large and XL: "License: Modified MIT License" (OpenAI's, github.com/openai/gpt-2)
const MODIFIED_MIT = "Modified MIT License";
// T252: the cards of unsloth's copy and of bartowski's GGUF say llama3.1 (the license's own title: "Llama 3.1 Community
// License Agreement"), and DeepSeek's card says what its Llama distill was made from
const LLAMA_31 = "Llama 3.1 Community License";
const MIT_OF_LLAMA_31 = "MIT License (derived from Llama 3.1 8B, originally under the Llama 3.1 Community License)";
// T260: Liquid AI's own license, as the LICENSE file of every repository names itself (the cards say "other", lfm1.0)
const LFM_OPEN = "LFM Open License v1.0";
// T261.3: Google's terms for Gemma, as the cards of Gemma 3 name them ("**Terms of Use**: [Terms]", a link to
// ai.google.dev/gemma/terms; their metadata: gemma). unsloth's copies and GGUFs have the same words
const GEMMA = "Gemma Terms of Use";
const SARASHINA_NC = "Sarashina Model NonCommercial License Agreement";
// T648.4: Mozilla's Firefox Translations models: mozilla/translations' README ("The models are released under the
// MPL-2.0"), and the files' own LICENSE.txt this site serves beside them
const MPL = "Mozilla Public License 2.0";
// T649.3: PFN's license of PLaMo, as the cards of PLaMo 3 Translate name it ("released under the PLaMo Community
// License"; their metadata: other, plamo-community-license), its text in the repositories' LICENSE/en and LICENSE/ja
const PLAMO = "PLaMo Community License";
export const LICENSES: Record<string, string> = {
  "pfnet/plamo-3-translate-0.8b": PLAMO, "pfnet/plamo-3-translate-2b": PLAMO, "pfnet/plamo-3-translate-8b": PLAMO,
  // (T649.4: and PFN's GGUFs of them, under the same license and with the same LICENSE/en, to the byte)
  "pfnet/plamo-3-translate-0.8b-GGUF": PLAMO, "pfnet/plamo-3-translate-2b-GGUF": PLAMO, "pfnet/plamo-3-translate-8b-GGUF": PLAMO,
  "mozilla/translations": MPL,
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
  "Qwen/Qwen2.5-7B-Instruct": APACHE, "elyza/ELYZA-Shortcut-1.0-Qwen-7B": APACHE, "mmnga/ELYZA-Shortcut-1.0-Qwen-7B-gguf": APACHE, "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5": SWALLOW,
  "llm-jp/llm-jp-4-8b-instruct": APACHE,
  // T592.2: Apache-2.0 by both cards
  "llm-jp/llm-jp-4-8b-thinking": APACHE, "mmnga-o/llm-jp-4-8b-thinking-gguf": APACHE,
  // T607.1: Apache-2.0 by both cards
  "DataPilot/ArrowPro-7B-KUJIRA": APACHE, "mmnga/DataPilot-ArrowPro-7B-KUJIRA-gguf": APACHE,
  // T607.7: Apache-2.0 by the cards (the originals' say they follow llm-jp-3-3.7b-instruct's)
  "AXCXEPT/llm-jp-3-3.7b-instruct-EZO-Common": APACHE, "ineair/llm-jp-3-3.7b-instruct-EZO-Common-gguf": APACHE,
  "AXCXEPT/llm-jp-3-3.7b-instruct-EZO-Humanities": APACHE, "ineair/llm-jp-3-3.7b-instruct-EZO-Humanities-gguf": APACHE,
  // T607.6: the Llama 3.2 Community License by both cards (its card's legal notice too)
  "AXCXEPT/EZO-Llama-3.2-3B-Instruct-dpoE": LLAMA_32, "mradermacher/EZO-Llama-3.2-3B-Instruct-dpoE-GGUF": LLAMA_32,
  // T607.5: Apache-2.0 by both cards
  "AXCXEPT/Qwen3-EZO-8B-beta": APACHE, "mradermacher/Qwen3-EZO-8B-beta-GGUF": APACHE,
  // T607.4: Apache-2.0 by both cards
  "umiyuki/Umievo-itr012-Gleipnir-7B": APACHE, "mmnga/umiyuki-Umievo-itr012-Gleipnir-7B-gguf": APACHE,
  // T607.3: Apache-2.0 by both cards
  "nitky/Oumuamua-7b-instruct-v2": APACHE, "mradermacher/Oumuamua-7b-instruct-v2-GGUF": APACHE,
  // T607.2: Apache-2.0 by both cards
  "Local-Novel-LLM-project/Vecteus-v1": APACHE, "mmnga/Vecteus-v1-gguf": APACHE,
  // T607.8: MIT by both cards
  "Aratako/Qwen3-8B-RP-v0.1": MIT, "Aratako/Qwen3-8B-RP-v0.1-GGUF": MIT,
  // T304: Apache-2.0 by both cards
  "llm-jp/llm-jp-4.1-8b-thinking": APACHE, "llm-jp/llm-jp-4.1-8b-thinking-gguf": APACHE,
  // T126
  "rinna/japanese-gpt-1b": MIT,
  // T249 (2026-10-01): the Japanese ones of T248's survey that open as they are, and the Q8_0 GGUFs they are taken
  // from (the card's license is the original's)
  "rinna/japanese-gpt2-xsmall": MIT, "rinna/japanese-gpt2-medium": MIT,
  "sbintuitions/sarashina2.2-1b": MIT, "mradermacher/sarashina2.2-1b-GGUF": MIT,
  "sbintuitions/sarashina2.2-3b": MIT, "sbintuitions/sarashina2-7b": MIT, "QuantFactory/sarashina2-7b-GGUF": MIT,
  "sbintuitions/sarashina2.1-1b": SARASHINA_NC, "mradermacher/sarashina2.1-1b-GGUF": SARASHINA_NC,
  "stockmark/gpt-neox-japanese-1.4b": MIT,
  "line-corporation/japanese-large-lm-1.7b": APACHE, "mmnga/line-corp-japanese-large-lm-1.7b-gguf": APACHE,
  "utter-project/EuroLLM-1.7B-Instruct": APACHE, "mradermacher/EuroLLM-1.7B-Instruct-GGUF": APACHE,
  "llm-jp/llm-jp-3-1.8b-instruct3": APACHE, "mmnga/llm-jp-3-1.8b-instruct3-gguf": APACHE,
  "llm-jp/llm-jp-3-3.7b-instruct3": APACHE, "mmnga/llm-jp-3-3.7b-instruct3-gguf": APACHE,
  "llm-jp/llm-jp-3-7.2b-instruct3": APACHE, "mmnga/llm-jp-3-7.2b-instruct3-gguf": APACHE,
  "shisa-ai/shisa-v2.1-llama3.2-3b": LLAMA_32, "mradermacher/shisa-v2.1-llama3.2-3b-GGUF": LLAMA_32,
  "cyberagent/CAT-Translate-3.3b": MIT,
  "cyberagent/calm2-7b-chat": APACHE, "TheBloke/calm2-7B-chat-GGUF": APACHE, "cyberagent/CAT-Translate-7b": MIT,
  "mradermacher/CAT-Translate-7b-GGUF": MIT,
  // T517 (2026-10-15): the rest of LINE's japanese-large-lm (metadata and card body: Apache-2.0), and mmnga's Q8_0
  // GGUF of the 1.7B instruction-sft
  "line-corporation/japanese-large-lm-1.7b-instruction-sft": APACHE, "mmnga/line-corp-japanese-large-lm-1.7b-instruction-sft-gguf": APACHE,
  "line-corporation/japanese-large-lm-3.6b": APACHE, "line-corporation/japanese-large-lm-3.6b-instruction-sft": APACHE,
  // T250 (2026-10-01): the 8B ones
  "weblab-GENIAC/Tanuki-8B-dpo-v1.0": APACHE, "team-hatakeyama-phase2/Tanuki-8B-dpo-v1.0-GGUF": APACHE,
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
  "Rakuten/RakutenAI-7B-instruct": APACHE, "Rakuten/RakutenAI-7B": APACHE, "Rakuten/RakutenAI-2.0-mini": APACHE,
  "mradermacher/RakutenAI-7B-instruct-GGUF": APACHE, "mradermacher/RakutenAI-7B-GGUF": APACHE,
  // T601
  "rinna/llama-3-youko-8b-instruct": LLAMA_3, "rinna/llama-3-youko-8b": LLAMA_3,
  "mradermacher/llama-3-youko-8b-instruct-GGUF": LLAMA_3, "mradermacher/llama-3-youko-8b-GGUF": LLAMA_3,
  "rinna/youri-7b-chat": LLAMA_2, "mradermacher/youri-7b-chat-GGUF": LLAMA_2,
  // T605: ELYZA's card (metadata llama2; "Llama 2 is licensed under the LLAMA 2 Community License"), mmnga's the same
  "elyza/ELYZA-japanese-Llama-2-7b-instruct": LLAMA_2, "mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf": LLAMA_2,
  "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct": LLAMA_2, "mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf": LLAMA_2,
  // T604
  "DataPilot/ArrowCanaria-Llama-8B-RL-v0.1": ARROWCANARIA, "mradermacher/ArrowCanaria-Llama-8B-RL-v0.1-GGUF": ARROWCANARIA,
  "SousiOmine/Lumi-v2-llm-jp-4-8b": APACHE, "SousiOmine/Lumi-v2-llm-jp-4-8b-GGUF": APACHE,
  // T603
  "lightblue/DeepSeek-R1-Distill-Qwen-7B-Japanese": APACHE, "mradermacher/DeepSeek-R1-Distill-Qwen-7B-Japanese-GGUF": APACHE,
  "lightblue/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual": APACHE, "mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual-GGUF": APACHE,
  "abeja/ABEJA-Qwen2.5-7b-Japanese-v0.1": APACHE, "mmnga/ABEJA-Qwen2.5-7b-Japanese-v0.1-gguf": APACHE,
  // T610.4: Apache-2.0 by both cards
  "abeja/ABEJA-Qwen3-14B-Agentic-256k-v0.1": APACHE, "mmnga-o/ABEJA-Qwen3-14B-Agentic-256k-v0.1-gguf": APACHE,
  // T610.5: Apache-2.0 by both cards
  "llm-jp/llm-jp-3.1-13b-instruct4": APACHE, "mmnga/llm-jp-3.1-13b-instruct4-gguf": APACHE,
  // T602
  "SakanaAI/TinySwallow-1.5B": APACHE, "mmnga/TinySwallow-1.5B-gguf": APACHE,
  "SakanaAI/Llama-3-Karamaru-v1": LLAMA_3, "mradermacher/Llama-3-Karamaru-v1-GGUF": LLAMA_3,
  "SakanaAI/EvoLLM-JP-A-v1-7B": APACHE, "mradermacher/EvoLLM-JP-A-v1-7B-GGUF": APACHE,
  "SakanaAI/EvoLLM-JP-v1-7B": MS_RESEARCH, "mradermacher/EvoLLM-JP-v1-7B-GGUF": MS_RESEARCH,
  "stabilityai/japanese-stablelm-instruct-gamma-7b": APACHE, "stabilityai/japanese-stablelm-base-gamma-7b": APACHE,
  "featherless-ai-quants/stabilityai-japanese-stablelm-instruct-gamma-7b-GGUF": APACHE,
  "mmnga/japanese-stablelm-base-gamma-7b-gguf": APACHE,
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
  // T610.1
  "Qwen/Qwen3-14B": APACHE, "Qwen/Qwen3-14B-GGUF": APACHE,
  // T610.2
  "shisa-ai/shisa-v2.1-unphi4-14b": MIT, "mradermacher/shisa-v2.1-unphi4-14b-GGUF": MIT,
  // T610.3
  "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese": MIT, "mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-gguf": MIT,
  // T606: the cards and the GGUFs' metadata, apache-2.0
  "tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2": APACHE, "mmnga-o/Qwen3-Swallow-8B-SFT-v0.2-gguf": APACHE,
  "tokyotech-llm/Qwen3-Swallow-8B-CPT-v0.2": APACHE, "tatsuyaaaaaaa/Qwen3-Swallow-8B-CPT-v0.2-gguf": APACHE,
  "tokyotech-llm/Medical-Qwen3-Swallow-8B": APACHE,
  "unsloth/Qwen3-4B-Instruct-2507-GGUF": APACHE, "unsloth/Qwen3-4B-Thinking-2507-GGUF": APACHE,
  "mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-GGUF": MIT,
  // T235: both cards say apache-2.0. Their NOTICE.txt: "copyright 2026-present Prism ML, Inc. ... built from Qwen3-1.7B,
  // Copyright 2024 Alibaba Cloud ... Apache 2.0", and asks for "Created using Bonsai by Prism ML." where it is deployed
  "prism-ml/Ternary-Bonsai-1.7B-gguf": APACHE, "prism-ml/Ternary-Bonsai-1.7B-unpacked": APACHE,
  // T286.2: the 1-bit Bonsai, Apache-2.0 by both cards
  "prism-ml/Bonsai-1.7B-gguf": APACHE, "prism-ml/Bonsai-1.7B-unpacked": APACHE,
  "prism-ml/Bonsai-4B-gguf": APACHE, "prism-ml/Bonsai-4B-unpacked": APACHE,
  "prism-ml/Bonsai-8B-gguf": APACHE, "prism-ml/Bonsai-8B-unpacked": APACHE,
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
  // T511.2: both cards say apache-2.0 (and the original's: "Apache License 2.0", a derivative of Qwen/Qwen3.5-0.8B)
  "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2.5": APACHE, "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2.5-GGUF": APACHE,
  // T337: both cards say apache-2.0
  "InternScience/Agents-A1-4B": APACHE, "InternScience/Agents-A1-4B-Q8_0-GGUF": APACHE,
  // T233: both cards say apache-2.0. The GGUF's NOTICE.txt: "copyright 2026-present Prism ML, Inc. ... built from
  // Qwen3.8-27B, Copyright 2026 Alibaba Cloud ... Apache 2.0", and asks for "Created using Bonsai by Prism ML." where
  // it is deployed, as the smaller Ternary Bonsai's does
  "prism-ml/Ternary-Bonsai-2-27B-gguf": APACHE, "Qwen/Qwen3.8-27B": APACHE,
  // T286.5: both cards of the 1-bit Bonsai 27B say apache-2.0, and its NOTICE.txt (in both): "copyright 2026-present
  // Prism ML, Inc. ... built from Qwen3.6-27B, Copyright 2026 Alibaba Cloud ... Apache 2.0", with the same ask
  "prism-ml/Bonsai-27B-gguf": APACHE, "prism-ml/Bonsai-27B-unpacked": APACHE,
  // T364.3: the card says apache-2.0 (its metadata and its "Credits and license"), and LICENSE is Apache 2.0's text.
  // NOTICE: "Copyright 2026 Underdog (ConwayResearch) ... a modified version of" Qwen3.8-27B and of ISTA-DASLab's
  // Qwen3.8-27B-GSQ-RCO-GGUF, "both licensed under the Apache License, Version 2.0"
  "ConwayResearch/Underdog-Saluki-27B-1.0": APACHE,
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
  "LiquidAI/LFM2-350M-ENJP-MT": LFM_OPEN, "LiquidAI/LFM2-350M-ENJP-MT-GGUF": LFM_OPEN,
  "LiquidAI/LFM2.5-2.6B": LFM_OPEN, "LiquidAI/LFM2.5-2.6B-GGUF": LFM_OPEN,
  "Takenoko12345678/Spark-X2.5-4B-Japanese": APACHE, "Takenoko12345678/Spark-X2.5-4B-Japanese-GGUF": APACHE,
  // T511.1: the cards of Shisa's 350M and of its GGUF have no license (no field, no LICENSE file): written is the license
  // of their base_model, Liquid AI's LFM2-350M-ENJP-MT (itself from LFM2-350M), whose card says lfm1.0 (the owner's
  // decision, 2026-10-15)
  "shisa-ai/shisa-v2.1c-lfm2-350m": LFM_OPEN, "mradermacher/shisa-v2.1c-lfm2-350m-GGUF": LFM_OPEN,
  // T511.3: both cards say license other, lfm1.0 (the original's links its LICENSE file, Liquid AI's)
  "shisa-ai/shisa-v2.1-lfm2-1.2b": LFM_OPEN, "mradermacher/shisa-v2.1-lfm2-1.2b-GGUF": LFM_OPEN,
  // T261.3: Google's originals (gated: the list takes unsloth's copies and GGUFs of them)
  "google/gemma-3-270m-it": GEMMA, "unsloth/gemma-3-270m-it": GEMMA, "unsloth/gemma-3-270m-it-GGUF": GEMMA,
  "google/gemma-3-1b-it": GEMMA, "unsloth/gemma-3-1b-it": GEMMA, "unsloth/gemma-3-1b-it-GGUF": GEMMA,
  // T333: Gemma 4 is Apache 2.0, not the Gemma Terms of Use: the cards' metadata says apache-2.0 and their text "License:
  // Apache 2.0" (a link to ai.google.dev/gemma/docs/gemma_4_license); no notice is asked for
  "google/gemma-4-E2B-it": APACHE, "google/gemma-4-E4B-it": APACHE,
  // T529: llama.cpp's own GGUFs of them (ggml-org's; their cards say apache-2.0 and link Gemma 4's license)
  "ggml-org/gemma-4-E2B-it-GGUF": APACHE, "ggml-org/gemma-4-E4B-it-GGUF": APACHE,
  // T343: the four cards say apache-2.0, the originals' text "The Spark-X2.5 model series is licensed under the Apache 2.0
  // License", and their LICENSE is Apache 2.0's text
  "XHToken/Spark-X2.5-1.7B": APACHE, "XHToken/Spark-X2.5-1.7B-GGUF": APACHE,
  "XHToken/Spark-X2.5-4B": APACHE, "XHToken/Spark-X2.5-4B-GGUF": APACHE,
  "XHToken/Spark-X2.6-1.7B": APACHE, "XHToken/Spark-X2.6-1.7B-GGUF": APACHE,
  "XHToken/Spark-X2.6-4B": APACHE, "XHToken/Spark-X2.6-4B-GGUF": APACHE,
};

// T261.3: the notices a license asks whoever hosts the model to give, shown after the license on the line of the
// original in About (the owner's choice, 2026-10-14: not in the note, which a phone cuts short). Gemma's Terms of Use
// (section 3.1) ask for this one, word for word.
const GEMMA_NOTICE = "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms";
// T649.3: and the PLaMo Community License's Article 4 (1)(ii), "Built with PLaMo" (the owner's T649 (1): here and in
// the note of each of its entries)
const PLAMO_NOTICE = "Built with PLaMo";
export const NOTICES: Partial<Record<string, string>> = { "google/gemma-3-270m-it": GEMMA_NOTICE, "google/gemma-3-1b-it": GEMMA_NOTICE,
  "pfnet/plamo-3-translate-0.8b": PLAMO_NOTICE, "pfnet/plamo-3-translate-2b": PLAMO_NOTICE, "pfnet/plamo-3-translate-8b": PLAMO_NOTICE,
  "pfnet/plamo-3-translate-0.8b-GGUF": PLAMO_NOTICE, "pfnet/plamo-3-translate-2b-GGUF": PLAMO_NOTICE, "pfnet/plamo-3-translate-8b-GGUF": PLAMO_NOTICE };
// T649.3: where About's link of a license goes instead of the repository, for a license that asks for a copy of itself
// to be given with the model (the PLaMo Community License's Article 4 (1)(i)): its English text in the repository, at
// the revision the list pins (tests/models-check.ts holds them to the entries). T649.4: the line of the original (the
// vocabulary's) and that of the GGUF (the weights') each link their own, the same text
export const LICENSE_TEXTS: Partial<Record<string, string>> = {
  "pfnet/plamo-3-translate-0.8b": "https://huggingface.co/pfnet/plamo-3-translate-0.8b/blob/cc8e88169807d5a272e8ff06db12eb0129b67f81/LICENSE/en",
  "pfnet/plamo-3-translate-2b": "https://huggingface.co/pfnet/plamo-3-translate-2b/blob/355637cc04db67195ba87a97da85eaf1d52a22e8/LICENSE/en",
  "pfnet/plamo-3-translate-8b": "https://huggingface.co/pfnet/plamo-3-translate-8b/blob/681bd55cf01f5412aab8881e6a4a276c948fabc1/LICENSE/en",
  "pfnet/plamo-3-translate-0.8b-GGUF": "https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/blob/dbeb0856cb0b004b484f190bb71dd65876297013/LICENSE/en",
  "pfnet/plamo-3-translate-2b-GGUF": "https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF/blob/232fbd7916519edecddfdae0beae7e76fe70453f/LICENSE/en",
  "pfnet/plamo-3-translate-8b-GGUF": "https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF/blob/2234f168757f570714b333049c9d018161ec5277/LICENSE/en",
};
