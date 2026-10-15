// From Hugging Face, converted in the browser: the families that come as one that thinks and one that answers at
// once (Qwen3, Qwen3.5 and its fine-tunes, Granite, SmolLM3, Spark-X2.5), MiniCPM5, LFM2, Gemma 3 and Ternary Bonsai.
import {
  AGENTS_A1_AT_ONCE, AGENTS_A1_THINKING, ASK_JAPANESE, CHATML, GRANITE_AT_ONCE, GRANITE_THINKING, QWEN35_AT_ONCE,
  QWEN35_THINKING, QWEN3_AT_ONCE_AFTER_START, QWEN3_FROM_IM_START, QWEN3_OWN_BOS, QWEN3_THINKING_2507,
  QWEN3_THINKING_AFTER_START, SPARK_AT_ONCE, SPARK_THINKING, granite, lfm2Old, qwen35, spark,
} from "./formats.js";
import {
  AGENTS_A1_SAMPLING, BONSAI_2_SAMPLING, QWEN35_JA_SFT_SAMPLING, QWEN35_SAMPLING, SALUKI_SAMPLING, SPARK_SAMPLING, atOnce, graniteSampling, thinking,
} from "./sampling.js";
import { gemma3, gemma4, ggufOf, lfm2, miniCpm5, shisaLfm2, ternaryBonsai, thinkingAndNot } from "./builders.js";

export const HF_FAMILIES = [
  ...thinkingAndNot("hf-qwen3-4b", "Alibaba Qwen3 4B",
    ggufOf("Qwen/Qwen3-4B-GGUF", "bc640142c66e1fdd12af0bd68f40445458f3869b", "Qwen3-4B-Q8_0.gguf",
      "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"), 4280404704,
    "fetches 4.3 GB (GGUF) → int8 4.5 GB"),
  ...thinkingAndNot("hf-qwen3-8b", "Alibaba Qwen3 8B",
    ggufOf("Qwen/Qwen3-8B-GGUF", "7c41481f57cb95916b40956ab2f0b139b296d974", "Qwen3-8B-Q8_0.gguf",
      "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"), 8709518112,
    "fetches 8.7 GB (GGUF) → int8 9.2 GB",
    { options: QWEN3_FROM_IM_START }, { thinking: QWEN3_THINKING_AFTER_START, atOnce: QWEN3_AT_ONCE_AFTER_START }),
  // Qwen3 Swallow's card gives one sampling, the thinking one, for both
  ...thinkingAndNot("hf-qwen3-swallow-8b", "Science Tokyo Qwen3 Swallow 8B RL",
    ggufOf("mmnga-o/Qwen3-Swallow-8B-RL-v0.2-gguf", "3fc755c6ab3780ebad6671130fc1b630bbd1575b", "Qwen3-Swallow-8B-RL-v0.2-Q8_0.gguf",
      "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", "9218f4843b6f93369a0b0999d8f58d61487ea71c"), 8709519968,
    "fetches 8.7 GB (GGUF) → int8 9.2 GB", { generation: thinking }),
  { group: "hf", id: "hf-qwen3-4b-instruct-2507", name: "Alibaba Qwen3 4B Instruct 2507", note: "answers instructions · 日本語 / English · fetches 4.3 GB (GGUF) → int8 4.5 GB",
    ...ggufOf("unsloth/Qwen3-4B-Instruct-2507-GGUF", "a06e946bb6b655725eafa393f4a9745d460374c9", "Qwen3-4B-Instruct-2507-Q8_0.gguf",
      "Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"), download: 4280405600,
    conversion: {}, options: QWEN3_OWN_BOS, template: CHATML, generation: atOnce, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // its chat_template begins the answer with <think> itself
  { group: "hf", id: "hf-qwen3-4b-thinking-2507", name: "Alibaba Qwen3 4B Thinking 2507", note: "thinks before it answers · 日本語 / English · fetches 4.3 GB (GGUF) → int8 4.5 GB",
    ...ggufOf("unsloth/Qwen3-4B-Thinking-2507-GGUF", "f40adb104d4d44aee52f398b60597c5866a973a3", "Qwen3-4B-Thinking-2507-Q8_0.gguf",
      "Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea"), download: 4280405632,
    conversion: {}, options: QWEN3_OWN_BOS, template: QWEN3_THINKING_2507, generation: thinking, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
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
    "fetches 1.1 GB (GGUF) → ternary 1.1 GB"),
  ternaryBonsai("8B", "c2aefbeb4b24469cd11579c3384b990404c17a30", "ac20f03fc62e872399218b659c8e949dfca05769", 2182184672,
    "fetches 2.2 GB (GGUF) → ternary 2.3 GB", { options: QWEN3_FROM_IM_START, template: QWEN3_AT_ONCE_AFTER_START }),
  // T236: Qwen3.5 0.8B, the first of the list with hybrid attention (T229: three layers of four are Gated DeltaNet
  // layers, which keep a state of a fixed size where the fourth keeps keys and values), on the CPU (no GPU path yet).
  // A vision-language model, of which the page reads the language model. unsloth's Q8_0 GGUF, which
  // tests/gguf_check.py tensors held to the original (every Q8_0 tensor is llama.cpp's Q8_0 of the original's, every
  // F32 one the original's values). Its card: thinking is off unless asked for (the 0.8B "is more prone to entering
  // thinking loops"), and for the sampling without thinking it names temperature 1.0, top-p 1.0, top-k 20 and a
  // presence penalty of 2.0 for text, and 0.7, 0.8, 20 and 1.5 for pictures and in its benchmarks; with thinking 1.0,
  // 0.95, 20 and 1.5, or 0.6, 0.95, 20 and no penalty "for precise coding". Since T274 the page samples with
  // QWEN35_SAMPLING (the card's set with thinking, and its 0.7 and 0.8 without; until then Qwen3's two, without a top-k
  // or a presence penalty)
  ...thinkingAndNot("hf-qwen3.5-0.8b", "Alibaba Qwen3.5 0.8B",
    ggufOf("unsloth/Qwen3.5-0.8B-GGUF", "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "Qwen3.5-0.8B-Q8_0.gguf",
      "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"), 811843840,
    "fetches 812 MB (GGUF) → int8 850 MB", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  // T511.2: a person's unofficial Japanese fine-tune of the Qwen3.5 0.8B (Takenoko12345678's v2.5: fewer Chinese
  // words and refusals, a short Japanese thought). The maker's own Q8_0 GGUF, the vocabulary and config.json of the
  // original; its chat_template.jinja, tokenizer.json and tokenizer_config.json are Qwen3.5-0.8B's byte for byte (the
  // same blobs), so the 0.8B's formats, BOS, stops and specials (qwen35). The sampling is its card's
  ...thinkingAndNot("hf-qwen3.5-0.8b-japanese-sft-v2.5", "Atemoya Qwen3.5 0.8B Japanese SFT v2.5",
    ggufOf("Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2.5-GGUF", "101c43b482a00d8d7cb5f1fb0de9e7c1bf30e447",
      "Qwen3.5-0.8B-Japanese-SFT-v2.5-Q8_0.gguf",
      "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2.5", "b069f75cda9f72cb979be84a78411ee48f75ff50"), 811842912,
    "fetches 812 MB (GGUF) → int8 850 MB", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_JA_SFT_SAMPLING),
  // T247: the other sizes of Qwen3.5 whose int8 a 64-bit memory of 16 GiB holds (the 27B's Q8_0 is 28.6 GB, and the
  // larger ones are mixtures of experts, which the engine has not). The same vocabulary, format and form as the 0.8B.
  // The 4B and the 9B have two value heads to a key head in their linear-attention layers, which llama.cpp writes in
  // another order than Hugging Face (T245: the converter puts them back), and they think unless told not to (the 0.8B
  // and the 2B only when told to): either form is written out here, so the two entries are the same two. The sampling
  // is their cards' for general tasks (T274: QWEN35_SAMPLING, and the 2B's as the 0.8B's). The 4B and the 9B are past a
  // 32-bit memory as int8
  ...thinkingAndNot("hf-qwen3.5-2b", "Alibaba Qwen3.5 2B",
    ggufOf("unsloth/Qwen3.5-2B-GGUF", "f6d5376be1edb4d416d56da11e5397a961aca8ae", "Qwen3.5-2B-Q8_0.gguf",
      "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"), 2012012800,
    "fetches 2.0 GB (GGUF) → int8 2.1 GB", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  ...thinkingAndNot("hf-qwen3.5-4b", "Alibaba Qwen3.5 4B",
    ggufOf("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q8_0.gguf",
      "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"), 4482403488,
    "fetches 4.5 GB (GGUF) → int8 4.7 GB", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  ...thinkingAndNot("hf-qwen3.5-9b", "Alibaba Qwen3.5 9B",
    ggufOf("unsloth/Qwen3.5-9B-GGUF", "3885219b6810b007914f3a7950a8d1b469d598a5", "Qwen3.5-9B-Q8_0.gguf",
      "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"), 9527502048,
    "fetches 9.5 GB (GGUF) → int8 10.1 GB", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  // T335: TokenRhythm's NeoHorse-1, the Qwen3.5 4B and 9B trained further for tools, code and instructions (text
  // only: the language model saved alone, its tensors named "model.layers…" where Qwen3.5's are
  // "model.language_model.layers…", which the converter reads either way). The maker's own Q8_0 GGUFs, the vocabulary
  // and config.json of the originals. Their chat_template.jinja, tokenizer.json and tokenizer_config.json are
  // Qwen3.5's byte for byte: so the same two formats by hand, BOS, stops and specials (qwen35). The card measured with
  // thinking on, temperature 1.0, top-p 0.95, a top-k of 20 and a presence penalty of 1.5: Qwen3.5's sets (T274). The
  // page has no tools, so what this model was trained for
  // most (calling them) it cannot do here: it answers in chat form only
  ...thinkingAndNot("hf-neohorse-1-4b", "TokenRhythm NeoHorse-1 4B",
    ggufOf("TokenRhythm/NeoHorse-1-4B-GGUF", "3c5d58ca82e580b5b0b3ce6eeffd34ac7d0fd95a", "NeoHorse-1-4B-Q8_0.gguf",
      "TokenRhythm/NeoHorse-1-4B", "56f0584bb40578a2c33b1b40a08ccd17243ad710"), 4482403072,
    "fetches 4.5 GB (GGUF) → int8 4.7 GB", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  ...thinkingAndNot("hf-neohorse-1-9b", "TokenRhythm NeoHorse-1 9B",
    ggufOf("TokenRhythm/NeoHorse-1-9B-GGUF", "ddcb4c939b5392c86a9d2733c7c0ed30db2554fd", "NeoHorse-1-9B-Q8_0.gguf",
      "TokenRhythm/NeoHorse-1-9B", "ba5b6e40d88a6ddf4591e176738254a3bc715765"), 9527501632,
    "fetches 9.5 GB (GGUF) → int8 10.1 GB", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  // T337: InternScience's Agents-A1-4B, a Qwen3.5 4B trained further for agents' work (the same config.json, the
  // tensors named as Qwen3.5's, a vision tower the converter passes over). The maker's own Q8_0 GGUF (its mmproj, the
  // pictures, is another file and is not fetched), the vocabulary and config.json of the original: tokenizer.json has
  // Qwen3.5's vocabulary and merges and seven more special tokens (for sound). The formats are Qwen3.5's after the
  // template's own system turn (AGENTS_A1_SYSTEM): 241 tokens the page reads before every answer. The card's sampler
  // (temperature 0.85, top-p 0.95, top-k 20, presence penalty 1.1) is the page's for both forms since T274
  // (AGENTS_A1_SAMPLING). After a change to the system
  // text, run tests/format_check.py: nothing that runs by itself compares it with the real template
  ...thinkingAndNot("hf-agents-a1-4b", "Shanghai AI Lab Agents-A1 4B",
    ggufOf("InternScience/Agents-A1-4B-Q8_0-GGUF", "a5d63881e0ca8eee3c0f14663a5fa2a2c55e1b54", "Agents-A1-4B-Q8_0.gguf",
      "InternScience/Agents-A1-4B", "945c40a4aa6f534d434a353207b8d42ecf7a5293"), 4482404032,
    "fetches 4.5 GB (GGUF) → int8 4.7 GB", { options: qwen35 },
    { thinking: AGENTS_A1_THINKING, atOnce: AGENTS_A1_AT_ONCE }, AGENTS_A1_SAMPLING),
  // T233: Prism ML's Ternary Bonsai 2 27B, a ternary Qwen3.8 27B (a Qwen3.5 in its form: hybrid attention, three
  // value heads to a key head, T245), whose matrices are stored in a rotated basis (T237: the engine turns every
  // matrix's input by signs and a Walsh-Hadamard transform). Kept ternary (T230, T231: 7.66 GB; as int8 it would be
  // 30 GB), on a 64-bit memory (7.70 GiB with its forward pass at 4096 positions), on the CPU (a hybrid, ternary,
  // rotated model has no GPU path yet). The PTQ1_0 file, the smaller of the two that hold the same weights (5.95 GB
  // against PQ2_0's 7.21 GB; both convert to the same checkpoint, byte for byte): Pyodide converts it at 27 MB/s and
  // PQ2_0 at 53, so on a line slower than 33 MB/s it is ready sooner, by the 1.26 GB less it fetches (TODO.md's T233).
  // The vocabulary and config.json are those of the model it is built from, Qwen/Qwen3.8-27B, whose weights are not
  // these (rebuilt: no check holds this GGUF's tensors to that repository's; tests/unfold_27b.py and
  // tests/page_27b.sh hold them, to the base model in the turned-back basis and to Prism ML's fork of llama.cpp).
  // Its chat_template calls a macro, as a Qwen3.5's: the two formats are the Qwen3.5's by hand, and the BOS is the
  // format's own <|im_start|> (qwen35: this family is much worse with <|endoftext|> in front, T236). The one that
  // answers at once is what the real template writes with enable_thinking false. The one that thinks is what it
  // writes with reasoning_effort "medium" (no system turn), not the model's own default, "xhigh" (a system turn that
  // asks for careful thought, for which its card leaves room for 16384 tokens: hours at the speed of a CPU, and past
  // this context); with medium the thought ended after 59 to 267 tokens on three questions in CI. The sampling is the
  // card's (T274, BONSAI_2_SAMPLING): 0.7, 0.8, a top-k of 20 and a presence penalty of 1.5 without thinking; 1.0,
  // 0.95, the top-k and a min-p of 0.05 with
  ...thinkingAndNot("hf-ternary-bonsai-2-27b", "Prism ML Ternary Bonsai 2 27B",
    ggufOf("prism-ml/Ternary-Bonsai-2-27B-gguf", "b072e1d3b35a0a630cece372c2127528e0994386", "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
      "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"), 5946648928,
    "ternary weights · fetches 5.9 GB (GGUF) → ternary 7.7 GB",
    { options: qwen35, weights: "ternary", rebuilt: true }, { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, BONSAI_2_SAMPLING),
  // T364.3: Conway Research's Underdog Saluki 27B, the same Qwen3.8 27B at 2.3 bits a weight: a GGUF of llama.cpp's
  // K- and i-quantized blocks, ten types of them, a type to a matrix or to a layer of one (T361). Kept as the file
  // has them (conversion.blocks, T363: 7.96 GB, where int8 would be 30 GB) and multiplied by kernels of their own, on
  // a 64-bit memory (7.97 GiB with its forward pass at 4096 positions), on the CPU. What it takes once loaded is said
  // here (loaded: the checkpoint's bytes), for the note is worded for a visitor (the owner's, as the names). The
  // vocabulary, config.json and the two formats are Ternary Bonsai 2 27B's above, those of the base model (rebuilt:
  // tests/page_saluki.sh holds this GGUF, to llama.cpp with its activations left unrounded and to the engine's NumPy
  // pass). Its GGUF's chat template is that model's too, to the character: so the form that thinks is what it writes
  // with reasoning_effort "medium" (no system turn), as Bonsai's; it wrote with it in CI, where four thoughts of four
  // ended within 338 tokens. The sampling is its card's (SALUKI_SAMPLING)
  ...thinkingAndNot("hf-underdog-saluki-27b", "Underdog Saluki 27B",
    ggufOf("ConwayResearch/Underdog-Saluki-27B-1.0", "1336c0b5d74dfe6ad7f793577092f5effd9f97c6", "Underdog-Saluki-27B-1.0-IQ2-mix.gguf",
      "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"), 7898369152,
    "2.3-bit weights · fetches 7.9 GB (GGUF), kept as it is: 8.0 GB · about a word a second or less",
    { options: qwen35, conversion: { dtype: "int8", blocks: true }, loaded: 7956712492, rebuilt: true },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, SALUKI_SAMPLING),
  // T253: Granite 4.2 (IBM; Japanese is among the languages its card says it was tested in), a Llama whose attention
  // multiplies its scores by config.json's attention_multiplier, which the converter puts into q (llama2_convert's
  // query_scale()). IBM's own Q8_0 GGUFs, which tests/gguf_check.py tensors held to the originals. The 3B fits a
  // 32-bit memory in six bits (Safari); the 8B does not
  ...thinkingAndNot("hf-granite-4.2-3b", "IBM Granite 4.2 3B",
    ggufOf("ibm-granite/granite-4.2-3b-GGUF", "c40945d71cd90f249a56985e8155551a9188dc30", "granite-4.2-3b-Q8_0.gguf",
      "ibm-granite/granite-4.2-3b", "e459acceac81e5fe67c07d9cfc72329a332e7eb1"), 3892651552,
    "fetches 3.9 GB (GGUF) → int8 4.1 GB", { options: granite },
    { thinking: GRANITE_THINKING, atOnce: GRANITE_AT_ONCE }).map((entry) => ({ ...entry, generation: graniteSampling })),
  ...thinkingAndNot("hf-granite-4.2-8b", "IBM Granite 4.2 8B",
    ggufOf("ibm-granite/granite-4.2-8b-GGUF", "93f3f6a8938ee922b784cf4e5b4203cd3428df8f", "granite-4.2-8b-Q8_0.gguf",
      "ibm-granite/granite-4.2-8b", "f8de16cdcdbc6c779ca517604e050d82cc119e44"), 9345613952,
    "fetches 9.3 GB (GGUF) → int8 9.9 GB", { options: granite },
    { thinking: GRANITE_THINKING, atOnce: GRANITE_AT_ONCE }).map((entry) => ({ ...entry, generation: graniteSampling })),
  // T254: MiniCPM5, whose tokenizer.json cuts the numbers off before Llama 3's pattern runs (the engine's "minicpm5").
  // OpenBMB's own Q8_0 GGUFs, which tests/gguf_check.py tensors held to the originals; the vocabulary is the
  // original's (the 1B's GGUF calls its pre-tokenizer llama-bpe, which is not what its tokenizer.json does). The
  // cards: temperature 0.9 thinking and 0.7 without for the 1B, 1.0 for the 2B, top-p 0.95
  ...miniCpm5("hf-minicpm5-1b", "OpenBMB MiniCPM5 1B",
    ggufOf("openbmb/MiniCPM5-1B-GGUF", "3d55fac80935ae6456986ad2384b5cbcc4d6c948", "MiniCPM5-1B-Q8_0.gguf",
      "openbmb/MiniCPM5-1B", "87179e5c1f455ef22e6223592d2d61351b525bfc"), 1153529216,
    "fetches 1.2 GB (GGUF) → int8 1.2 GB", { thinking: 0.9, atOnce: 0.7 }),
  ...miniCpm5("hf-minicpm5-2b", "OpenBMB MiniCPM5 2B",
    ggufOf("openbmb/MiniCPM5-2B-GGUF", "2079a22f3beaa4e306449978533478fe0522f4b3", "MiniCPM5-2B-Q8_0.gguf",
      "openbmb/MiniCPM5-2B", "f97400052a43d642bbc6e9975e2397e3ae6a6b52"), 2679710688,
    "fetches 2.7 GB (GGUF) → int8 2.8 GB", { thinking: 1.0, atOnce: 1.0 }),
  // T260: Liquid AI's LFM2.5 (and the LFM2 700M), whose cards name Japanese among their languages; the 1.2B JP is
  // their Japanese chat model. Liquid AI's own Q8_0 GGUFs, which tests/gguf_check.py tensors held to the originals.
  // On the CPU alone (their convolution layers have no shader yet). The 230M's GGUF holds weights of a fuller
  // precision than the bfloat16 its original publishes (its float32 norms and taps round to the original's bfloat16
  // to the bit, and its Q8_0 matrices were rounded from the float32 too: 1.5e-3 from the Q8_0 of the original, where
  // the check allows 1e-3): since T307 the check holds every value of it to what such weights can give, and it is
  // taken (until then the 230M came from the original's safetensors, 459 MB)
  lfm2("hf-lfm2.5-230m", "Liquid AI LFM2.5 230M", "LFM2.5-230M", "03502067c64ce32ac4fe87b0cec0310a1a13d3e9",
    "40cb2ad3b3044d5a41eee083a6103c8b523afa45", 246598496, "fetches 247 MB (GGUF) → int8 259 MB"),
  lfm2("hf-lfm2.5-350m", "Liquid AI LFM2.5 350M", "LFM2.5-350M", "657e078c94084481950a2d555a941481f715536b",
    "9e6c6ccf47cd318696e137d381a7ded8fe4df09f", 379217632, "fetches 379 MB (GGUF) → int8 399 MB"),
  lfm2("hf-lfm2-700m", "Liquid AI LFM2 700M", "LFM2-700M", "fd39e80d7a5ac61494ffff577e61bbbfddbd0d02",
    "86f49fc9a3800c3a325b7320bde179c318062583", 791565248, "fetches 792 MB (GGUF) → int8 836 MB",
    { options: lfm2Old, template: CHATML }, 0.3),
  lfm2("hf-lfm2.5-1.2b-instruct", "Liquid AI LFM2.5 1.2B Instruct", "LFM2.5-1.2B-Instruct", "8ed288026e23958ad9dfa92d53ed773a8eee7125",
    "0f604ada3f766f9f257460c4c9f0b5d6f69d431b", 1246253888, "fetches 1.2 GB (GGUF) → int8 1.3 GB"),
  lfm2("hf-lfm2.5-1.2b-jp", "Liquid AI LFM2.5 1.2B JP", "LFM2.5-1.2B-JP-202606", "448ba3f7d408c2f5c32cec8038612f7c1ed9f054",
    "52b8b4475311a63bf839c6494f78f8ad59d13515", 1246253344, "fetches 1.2 GB (GGUF) → int8 1.3 GB", {}),
  // T511.1: Shisa.AI's LFM2 350M for Japanese chat (trained from Liquid AI's LFM2-350M-ENJP-MT, the translating one);
  // its card says it is the best at Japanese chat under 1B (Shaberi 4.51, the LFM2-350M's 3.76)
  shisaLfm2("hf-shisa-v2.1c-lfm2-350m", "Shisa V2.1c LFM2 350M", "shisa-v2.1c-lfm2-350m",
    "a902123e67f297bfce7e1961104ebb4fee2ba538", "38445ae70be89a0d3d870707696a8f967eb08d5e", 379216512,
    "fetches 379 MB (GGUF) → int8 399 MB"),
  // T528: and the same model lighter, beside it (the owner: "どっちも配ればいいんじゃないの？"): the same repository's
  // Q4_K_M, llama.cpp's blocks kept as they are (conversion.blocks, as Underdog Saluki 27B's) where the page has kernels
  // for them (Q4_K: q, k, the attention's and the convolutions' matrices, w1 and w3), the rest widened to int8 (the
  // embedding and the layers' w2 and v that are Q6_K). T513 measured it: 229 MB fetched where the int8 takes 379,
  // 282 + 129 MB of memory where it takes 399 + 155, 0.79 to 0.88 times its speed on CI's CPUs, 1.2% more nats a
  // character of Japanese. What it takes once loaded is said (loaded: the checkpoint's bytes, tests/quant_sizes.py).
  // rebuilt: tests/gguf_check.py tensors holds a Q8_0 to its original within 1e-3, which 4 bits are not (gguf.yml skips
  // it); what holds it is TODO.md's T528: that check read for its orders and errors, and T513's measures
  shisaLfm2("hf-shisa-v2.1c-lfm2-350m-q4", "Shisa V2.1c LFM2 350M (4-bit)", "shisa-v2.1c-lfm2-350m",
    "a902123e67f297bfce7e1961104ebb4fee2ba538", "38445ae70be89a0d3d870707696a8f967eb08d5e", 229311104,
    "4-bit weights · fetches 229 MB (GGUF), kept mostly as it is: 282 MB", "Q4_K_M",
    { conversion: { dtype: "int8", blocks: true }, loaded: 281641004, rebuilt: true }),
  // T511.3: and its 1.2B (from LFM2-1.2B), the 1B class's best at Japanese chat as Shisa's card measures (Shaberi 5.35)
  shisaLfm2("hf-shisa-v2.1-lfm2-1.2b", "Shisa V2.1 LFM2 1.2B", "shisa-v2.1-lfm2-1.2b",
    "292863c4c1d3b63357027b384f336a52498252b2", "4a433b919681b37f43088cf2c57cbe92a8d859d5", 1246254048,
    "fetches 1.2 GB (GGUF) → int8 1.3 GB"),
  // T261.3: Gemma 3's two smallest (the text models), on the CPU alone (no shader has their norms behind the layers, a
  // GELU in the gate or the window yet: T261.4). unsloth's Q8_0 GGUFs, which tests/gguf_check.py tensors held to
  // unsloth's copies of the originals
  gemma3("270M", "c90975dbd40c0c7b275fefaae758c3415c906238", "23cf460f6bb16954176b3ddcc8d4f250501458a9", 291546144,
    "fetches 292 MB (GGUF) → int8 302 MB"),
  gemma3("1B", "f0b45be0aac41bd6a100a4b5734cad5f67255bfb", "5b11413a10db4e486ef16a20101fd028f8f2499c", 1069306400,
    "fetches 1.1 GB (GGUF) → int8 1.1 GB"),
  // T333: Gemma 4's two smallest (E2B, E4B), on the CPU alone (no shader has their layers yet: T334). T329.3 held the
  // E2B to transformers (logits 1.10e-4, the most likely token 96 of 96) and ran it in three browsers from ?hf=. loaded:
  // the int8 checkpoint without the table of every layer's inputs, which stays in the kept file (the converter's sizes
  // of the real headers and forward.ts's tableHole()): E2B 5,207,962,792 bytes less 2,642,411,520, E4B 8,397,500,612 less 3,170,893,824
  // T529: the weights from ggml-org's Q8_0 GGUFs (llama.cpp's own), which tests/gguf_check.py tensors held to the originals
  gemma4("E2B", "3e22461f65e89153144f8adb70e3b8c2cc9845a7", "b4243c156154b6dca9324415f8c7ccc098b4aed1", 4967497152, 2565551272,
    "fetches 5.0 GB (GGUF) → int8 5.2 GB (2.6 GB in memory)"),
  gemma4("E4B", "ee0ef6023621cff504d758262d4e04895a5af4a2", "b8093469224f83f5c38f691eb906c380e9e63114", 8031242688, 5226606788,
    "fetches 8.0 GB (GGUF) → int8 8.4 GB (5.2 GB in memory)"),
  // T343: XHToken's Spark-X2.5 (T339 to T342: a Llama with a gate a head after the attention, an erf GELU in the FFN, and
  // three layers of four that see the last 512 positions alone, with a RoPE of their own; on the CPU alone, T344).
  // XHToken's own Q8_0 GGUFs, which tests/gguf_check.py tensors held to the originals (their q and k in the HF order,
  // which the family reads), with the vocabulary, config.json and chat template of the originals. Its template thinks
  // by default and answers at once with enable_thinking false: the two formats by hand (SPARK_THINKING,
  // SPARK_AT_ONCE: it calls a macro), with their BOS, stops and specials (spark). The card's sampling for every task
  ...thinkingAndNot("hf-spark-x2.5-1.7b", "Spark-X2.5 1.7B",
    ggufOf("XHToken/Spark-X2.5-1.7B-GGUF", "1f7fa33b1245c14730da39e125714ad3a327901b", "Spark-X2.5-1.7B-Q8_0.gguf",
      "XHToken/Spark-X2.5-1.7B", "14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8"), 1820112704,
    "fetches 1.8 GB (GGUF) → int8 1.9 GB", { options: spark },
    { thinking: SPARK_THINKING, atOnce: SPARK_AT_ONCE }, SPARK_SAMPLING),
  ...thinkingAndNot("hf-spark-x2.5-4b", "Spark-X2.5 4B",
    ggufOf("XHToken/Spark-X2.5-4B-GGUF", "9826e0be84e6e6e8b9668abc91421109a1df1e2d", "Spark-X2.5-4B-Q8_0.gguf",
      "XHToken/Spark-X2.5-4B", "0bcb35678590218655dff3765b9e61c83b35e9c4"), 4375021152,
    "fetches 4.4 GB (GGUF) → int8 4.6 GB", { options: spark },
    { thinking: SPARK_THINKING, atOnce: SPARK_AT_ONCE }, SPARK_SAMPLING),
];
