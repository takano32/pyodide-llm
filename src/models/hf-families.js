// From Hugging Face, converted in the browser: the families that come as one that thinks and one that answers at
// once (Qwen3, Qwen3.5 and its fine-tunes, Granite, SmolLM3), MiniCPM5, LFM2 and Ternary Bonsai.
import {
  AGENTS_A1_AT_ONCE, AGENTS_A1_THINKING, ASK_JAPANESE, CHATML, GRANITE_AT_ONCE, GRANITE_THINKING, QWEN35_AT_ONCE,
  QWEN35_THINKING, QWEN3_AT_ONCE_AFTER_START, QWEN3_FROM_IM_START, QWEN3_THINKING_AFTER_START, granite, lfm2Old,
  qwen35,
} from "./formats.js";
import {
  AGENTS_A1_SAMPLING, BONSAI_2_SAMPLING, QWEN35_SAMPLING, atOnce, graniteSampling, thinking,
} from "./sampling.js";
import { ggufOf, lfm2, miniCpm5, ternaryBonsai, thinkingAndNot } from "./builders.js";

export const HF_FAMILIES = [
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
  // 0.95, 20 and 1.5, or 0.6, 0.95, 20 and no penalty "for precise coding". Since T274 the page samples with
  // QWEN35_SAMPLING (the card's set with thinking, and its 0.7 and 0.8 without; until then Qwen3's two, without a top-k
  // or a presence penalty)
  ...thinkingAndNot("hf-qwen3.5-0.8b", "Qwen3.5 0.8B",
    ggufOf("unsloth/Qwen3.5-0.8B-GGUF", "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "Qwen3.5-0.8B-Q8_0.gguf",
      "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"), 811843840,
    "fetches 812 MB (GGUF) → int8 850 MB · desktop only", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  // T247: the other sizes of Qwen3.5 whose int8 a 64-bit memory of 16 GiB holds (the 27B's Q8_0 is 28.6 GB, and the
  // larger ones are mixtures of experts, which the engine has not). The same vocabulary, format and form as the 0.8B.
  // The 4B and the 9B have two value heads to a key head in their linear-attention layers, which llama.cpp writes in
  // another order than Hugging Face (T245: the converter puts them back), and they think unless told not to (the 0.8B
  // and the 2B only when told to): either form is written out here, so the two entries are the same two. The sampling
  // is their cards' for general tasks (T274: QWEN35_SAMPLING, and the 2B's as the 0.8B's). The 4B and the 9B are past a
  // 32-bit memory as int8
  ...thinkingAndNot("hf-qwen3.5-2b", "Qwen3.5 2B",
    ggufOf("unsloth/Qwen3.5-2B-GGUF", "f6d5376be1edb4d416d56da11e5397a961aca8ae", "Qwen3.5-2B-Q8_0.gguf",
      "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"), 2012012800,
    "fetches 2.0 GB (GGUF) → int8 2.1 GB · desktop only", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  ...thinkingAndNot("hf-qwen3.5-4b", "Qwen3.5 4B",
    ggufOf("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q8_0.gguf",
      "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"), 4482403488,
    "fetches 4.5 GB (GGUF) → int8 4.7 GB · desktop only · Chrome and Firefox", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  ...thinkingAndNot("hf-qwen3.5-9b", "Qwen3.5 9B",
    ggufOf("unsloth/Qwen3.5-9B-GGUF", "3885219b6810b007914f3a7950a8d1b469d598a5", "Qwen3.5-9B-Q8_0.gguf",
      "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"), 9527502048,
    "fetches 9.5 GB (GGUF) → int8 10.1 GB · desktop only · Chrome and Firefox", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  // T335: TokenRhythm's NeoHorse-1, the Qwen3.5 4B and 9B trained further for tools, code and instructions (text
  // only: the language model saved alone, its tensors named "model.layers…" where Qwen3.5's are
  // "model.language_model.layers…", which the converter reads either way). The maker's own Q8_0 GGUFs, the vocabulary
  // and config.json of the originals. Their chat_template.jinja, tokenizer.json and tokenizer_config.json are
  // Qwen3.5's byte for byte: so the same two formats by hand, BOS, stops and specials (qwen35). The card measured with
  // thinking on, temperature 1.0, top-p 0.95, a top-k of 20 and a presence penalty of 1.5: Qwen3.5's sets (T274). The
  // page has no tools, so what this model was trained for
  // most (calling them) it cannot do here: it answers in chat form only
  ...thinkingAndNot("hf-neohorse-1-4b", "NeoHorse-1 4B",
    ggufOf("TokenRhythm/NeoHorse-1-4B-GGUF", "3c5d58ca82e580b5b0b3ce6eeffd34ac7d0fd95a", "NeoHorse-1-4B-Q8_0.gguf",
      "TokenRhythm/NeoHorse-1-4B", "56f0584bb40578a2c33b1b40a08ccd17243ad710"), 4482403072,
    "fetches 4.5 GB (GGUF) → int8 4.7 GB · desktop only · Chrome and Firefox", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  ...thinkingAndNot("hf-neohorse-1-9b", "NeoHorse-1 9B",
    ggufOf("TokenRhythm/NeoHorse-1-9B-GGUF", "ddcb4c939b5392c86a9d2733c7c0ed30db2554fd", "NeoHorse-1-9B-Q8_0.gguf",
      "TokenRhythm/NeoHorse-1-9B", "ba5b6e40d88a6ddf4591e176738254a3bc715765"), 9527501632,
    "fetches 9.5 GB (GGUF) → int8 10.1 GB · desktop only · Chrome and Firefox", { options: qwen35 },
    { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, QWEN35_SAMPLING),
  // T337: InternScience's Agents-A1-4B, a Qwen3.5 4B trained further for agents' work (the same config.json, the
  // tensors named as Qwen3.5's, a vision tower the converter passes over). The maker's own Q8_0 GGUF (its mmproj, the
  // pictures, is another file and is not fetched), the vocabulary and config.json of the original: tokenizer.json has
  // Qwen3.5's vocabulary and merges and seven more special tokens (for sound). The formats are Qwen3.5's after the
  // template's own system turn (AGENTS_A1_SYSTEM): 241 tokens the page reads before every answer. The card's sampler
  // (temperature 0.85, top-p 0.95, top-k 20, presence penalty 1.1) is the page's for both forms since T274
  // (AGENTS_A1_SAMPLING). After a change to the system
  // text, run tests/format_check.py: nothing that runs by itself compares it with the real template
  ...thinkingAndNot("hf-agents-a1-4b", "Agents-A1 4B",
    ggufOf("InternScience/Agents-A1-4B-Q8_0-GGUF", "a5d63881e0ca8eee3c0f14663a5fa2a2c55e1b54", "Agents-A1-4B-Q8_0.gguf",
      "InternScience/Agents-A1-4B", "945c40a4aa6f534d434a353207b8d42ecf7a5293"), 4482404032,
    "fetches 4.5 GB (GGUF) → int8 4.7 GB · desktop only · Chrome and Firefox", { options: qwen35 },
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
  ...thinkingAndNot("hf-ternary-bonsai-2-27b", "Ternary Bonsai 2 27B",
    ggufOf("prism-ml/Ternary-Bonsai-2-27B-gguf", "b072e1d3b35a0a630cece372c2127528e0994386", "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
      "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"), 5946648928,
    "ternary weights · fetches 5.9 GB (GGUF) → ternary 7.7 GB · desktop only · Chrome and Firefox",
    { options: qwen35, weights: "ternary", rebuilt: true }, { thinking: QWEN35_THINKING, atOnce: QWEN35_AT_ONCE }, BONSAI_2_SAMPLING),
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
  // T260: Liquid AI's LFM2.5 (and the LFM2 700M), whose cards name Japanese among their languages; the 1.2B JP is
  // their Japanese chat model. Liquid AI's own Q8_0 GGUFs, which tests/gguf_check.py tensors held to the originals.
  // On the CPU alone (their convolution layers have no shader yet). The 230M's GGUF holds weights of a fuller
  // precision than the bfloat16 its original publishes (its float32 norms and taps round to the original's bfloat16
  // to the bit, and its Q8_0 matrices were rounded from the float32 too: 1.5e-3 from the Q8_0 of the original, where
  // the check allows 1e-3): since T307 the check holds every value of it to what such weights can give, and it is
  // taken (until then the 230M came from the original's safetensors, 459 MB)
  lfm2("hf-lfm2.5-230m", "LFM2.5 230M", "LFM2.5-230M", "03502067c64ce32ac4fe87b0cec0310a1a13d3e9",
    "40cb2ad3b3044d5a41eee083a6103c8b523afa45", 246598496, "fetches 247 MB (GGUF) → int8 259 MB"),
  lfm2("hf-lfm2.5-350m", "LFM2.5 350M", "LFM2.5-350M", "657e078c94084481950a2d555a941481f715536b",
    "9e6c6ccf47cd318696e137d381a7ded8fe4df09f", 379217632, "fetches 379 MB (GGUF) → int8 399 MB"),
  lfm2("hf-lfm2-700m", "LFM2 700M", "LFM2-700M", "fd39e80d7a5ac61494ffff577e61bbbfddbd0d02",
    "86f49fc9a3800c3a325b7320bde179c318062583", 791565248, "fetches 792 MB (GGUF) → int8 836 MB · desktop only",
    { options: lfm2Old, template: CHATML }, 0.3),
  lfm2("hf-lfm2.5-1.2b-instruct", "LFM2.5 1.2B Instruct", "LFM2.5-1.2B-Instruct", "8ed288026e23958ad9dfa92d53ed773a8eee7125",
    "0f604ada3f766f9f257460c4c9f0b5d6f69d431b", 1246253888, "fetches 1.2 GB (GGUF) → int8 1.3 GB · desktop only"),
  lfm2("hf-lfm2.5-1.2b-jp", "LFM2.5 1.2B JP", "LFM2.5-1.2B-JP-202606", "448ba3f7d408c2f5c32cec8038612f7c1ed9f054",
    "52b8b4475311a63bf839c6494f78f8ad59d13515", 1246253344, "fetches 1.2 GB (GGUF) → int8 1.3 GB · desktop only", {}),
];
