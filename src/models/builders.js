// The entries of the list that come in families: one function a family, which writes its entries from what differs.
import { ASK_JAPANESE, CHATML, PLAIN_ROPE, QWEN3_AT_ONCE, QWEN3_OWN_BOS, lfm25 } from "./formats.js";
import { atOnce, sampled, thinking } from "./sampling.js";

/** A Qwen3 twice (T.124, the owner's "両方を別々に用意できないのか"): thinking first, and answering at once. The two
 * share their weights, and so a conversion kept in the browser; only the format differs. shares: both ids, for
 * kept.js's replaced() (what either kept before its source changed goes, whichever form is opened first). sampling:
 * the two forms' generation settings where they are not Qwen3's. chat.languages: the note's languages where the card's
 * are not Japanese and English (T.537), with 日本語 where it answered Japanese questions in Japanese here (T.539) */
export function thinkingAndNot(id, name, source, download, sizes, { languages = "日本語 / English", ...chat } = {}, formats = {}, sampling = {}) {
  // T.369: where nothing else is said, a Qwen3 as the list has had it: <|endoftext|> in front of ChatML (QWEN3_OWN_BOS and
  // the whole format, formats.js's note on QWEN25). The converter alone would begin with the template's <|im_start|> now
  const common = { group: "hf", ...source, download, conversion: {}, options: QWEN3_OWN_BOS, shares: [`${id}-thinking`, id],
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE, ...chat };
  return [
    // formats.thinking: where the model's format is not Qwen3's (T.236)
    { ...common, id: `${id}-thinking`, name: `${name} (thinking)`, note: `thinks before it answers · ${languages} · ${sizes}`,
      generation: sampling.thinking ?? thinking, template: formats.thinking ?? CHATML },
    { ...common, id, name: `${name} (no thinking)`, note: `answers at once · ${languages} · ${sizes}`,
      generation: sampling.atOnce ?? chat.generation ?? atOnce, template: formats.atOnce ?? QWEN3_AT_ONCE },
  ];
}
/** T.254: OpenBMB's MiniCPM5 (a Llama; English and Chinese), twice as a Qwen3 is: its chat_template begins the answer
 * with "<think>\n" where enable_thinking is true and with an empty thought where it is false (and with neither where
 * nothing is said, which is the format the converter reads). The real tokenizer begins every text with <s>, the
 * converter's BOS. sampling: its card's for either form */
export function miniCpm5(id, name, source, download, sizes, sampling) {
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
/** T.203 (T.136's fourth stage): a Q8_0 GGUF's weights with the vocabulary and config.json of its original, which
 * tests/gguf_check.py tensors held them to (gguf.yml's candidates) */
export const ggufOf = (repo, revision, weights, original, originalRevision, tokenizer = "tokenizer.json") =>
  ({ original, hf: { repo, revision, weights, vocabulary: { repo: original, revision: originalRevision, tokenizer } } });
/** an LFM2 of the list: the maker's own Q8_0 GGUF's weights with the vocabulary and config.json of its original, or
 * (no revision of a GGUF: null) the original's safetensors */
export const lfm2 = (id, name, repo, revision, originalRevision, download, sizes, format = { options: lfm25, template: CHATML }, temperature = 0.1) => ({
  group: "hf", id, name, note: `answers instructions · 日本語 / English · ${sizes}`,
  ...(revision ? ggufOf(`LiquidAI/${repo}-GGUF`, revision, `${repo}-Q8_0.gguf`, `LiquidAI/${repo}`, originalRevision)
    : { hf: hf(`LiquidAI/${repo}`, originalRevision) }), download,
  conversion: {}, options: {}, ...format, generation: { steps: 0, temperature, topp: 1.0, repetition_penalty: 1.05 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE,
});
/** T.511.1, T.511.3: a Shisa V2.1 of Shisa.AI (an LFM2 trained further for Japanese chat): mradermacher's Q8_0 GGUF's
 * weights with the vocabulary, config.json and chat template of Shisa's original. Its template calls no macro (the
 * LFM2's of Liquid AI do), so the BOS (<|startoftext|>), the format and the specials are the converter's, as the
 * LFM2.5 1.2B JP's are. The cards name no sampling and generation_config.json says do_sample and nothing more: the
 * LFM2 700M's (temperature 0.3, Liquid AI's LFM2 cards; a choice of the coordinator, TASKS.md's T.511).
 * T.528: another GGUF of the same repository (quantization: the name's word, Q4_K_M) with what it takes besides (more:
 * conversion.blocks and loaded) */
export const shisaLfm2 = (id, name, repo, revision, ggufRevision, download, sizes, quantization = "Q8_0", more = {}) => ({
  group: "hf", id, name, note: `answers instructions · 日本語 / English · ${sizes}`,
  ...ggufOf(`mradermacher/${repo}-GGUF`, ggufRevision, `${repo}.${quantization}.gguf`, `shisa-ai/${repo}`, revision), download,
  conversion: {}, options: {}, generation: { steps: 0, temperature: 0.3, topp: 1.0, repetition_penalty: 1.05 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE, ...more,
});
/** T.261.3: a Gemma 3 of the list (the text model; Google's original is gated): unsloth's Q8_0 GGUF's weights with the
 * vocabulary, config.json and chat template of unsloth's copy of the original, as Llama 3.2's entries name all three.
 * The BOS (<bos>) and the format (<start_of_turn>user … <start_of_turn>model) are the converter's, read from the
 * template; the stops are the converter's <bos> and <end_of_turn> with <eos> besides, as generation_config.json's
 * eos_token_id (1 and 106) has the real generate() stop. The sampling is generation_config.json's: do_sample with a
 * top-k of 64 and a top-p of 0.95, and transformers' temperature of 1.0 where it names none.
 * The notice the Gemma Terms of Use ask for is not in the note: at a phone's width the select never shows it (the
 * owner chooses where it goes, TASKS.md's T.261.3) */
export const gemma3 = (size, revision, originalRevision, download, sizes) => ({
  group: "hf", id: `hf-gemma-3-${size.toLowerCase()}`, name: `Google Gemma 3 ${size}`,
  note: `answers instructions · 日本語 / English · ${sizes}`,
  ...ggufOf(`unsloth/gemma-3-${size.toLowerCase()}-it-GGUF`, revision, `gemma-3-${size.toLowerCase()}-it-Q8_0.gguf`,
    `unsloth/gemma-3-${size.toLowerCase()}-it`, originalRevision), original: `google/gemma-3-${size.toLowerCase()}-it`, download,
  conversion: {}, options: { stop_tokens: [1, 2, 106] },
  generation: { steps: 0, temperature: 1.0, topp: 0.95, top_k: 64, repetition_penalty: 1.0 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE,
});
/** T.333: a Gemma 4 of the list (E2B, E4B: the language model of Google's original, whose vision and audio towers the
 * converter leaves; the original is not gated and under Apache 2.0, so no copy and no notice). T.529: the weights of
 * llama.cpp's own Q8_0 GGUF of it (ggml-org's: half the original's download), with the vocabulary, config.json and chat
 * template of the original, which tests/gguf_check.py tensors held them to. Converted in the page; the table of every
 * layer's inputs stays in the file the conversion is kept in (OPFS) and is read a row a token (T.329.2), so what the
 * memory holds is the checkpoint without it: `loaded` says that, as the note
 * cannot (it gives the file's size). The BOS (<bos>) and the format (<|turn>user … <|turn>model, the real
 * apply_chat_template's without a thought) are the converter's. The stops are generation_config.json's eos_token_id,
 * <eos>, <turn|> and <|tool_response> (1, 106, 50): the converter names <bos> and <eos> alone (T.524), and an answer
 * then went on past <turn|>. The sampling is generation_config.json's: a temperature of 1.0, top-k 64, top-p 0.95 */
export const gemma4 = (size, revision, ggufRevision, download, loaded, sizes) => ({
  group: "hf", id: `hf-gemma-4-${size.toLowerCase()}`, name: `Google Gemma 4 ${size}`,
  note: `answers instructions · 日本語 / English · ${sizes}`,
  ...ggufOf(`ggml-org/gemma-4-${size}-it-GGUF`, ggufRevision, `gemma-4-${size}-it-Q8_0.gguf`, `google/gemma-4-${size}-it`,
    revision), download, loaded,
  conversion: {}, options: { stop_tokens: [1, 106, 50] },
  generation: { steps: 0, temperature: 1.0, topp: 0.95, top_k: 64, repetition_penalty: 1.0 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE,
});
/** T.235, T.246: a Ternary Bonsai of Prism ML (a ternary Qwen3) in one of its sizes: its PQ2_0 GGUF's weights with the
 * vocabulary, config.json and chat template of its -unpacked original. What the sizes share is here alone, so that one
 * change covers them all (the owner's open choices of T.235's review: a word more in the note; Prism ML's
 * attribution). The RoPE is plain (PLAIN_ROPE, the owner's choice of 2026-10-11) and not config.json's yarn: the page
 * cuts the context at 4096, inside the 8192 or 16384 positions yarn begins to stretch, and there yarn's perplexity was
 * 2 to 7% the higher in every size (transformers and the engine; Qwen3's card advises against a static yarn on short
 * texts as well). It is said last, so that a size's own options (`start`) keep it. The original's template
 * always begins the answer with an empty thought (Qwen3's enable_thinking=false): the model has one form, which the
 * converter reads; it is written here (QWEN3_AT_ONCE) since T.369, for the entries keep <|endoftext|> in front of it
 * (formats.js's note on QWEN25). It names no BOS (QWEN3_OWN_BOS, which the 1.7B and the 4B bear: 1.5% better than nothing in front
 * on plain text; the 8B does not: 70% worse, as a Qwen3 8B is, and begins at QWEN3_FROM_IM_START: `start`, T.250's review).
 * The sampling is the originals' generation_config.json, the same file in the three (its top-k the page's sampler
 * has not) */
export const ternaryBonsai = (size, revision, originalRevision, download, sizes, { options = QWEN3_OWN_BOS, ...start } = {}) => ({
  group: "hf", id: `hf-ternary-bonsai-${size.toLowerCase()}`, name: `Prism ML Ternary Bonsai ${size}`,
  note: `answers at once · 日本語 / English · ternary weights · ${sizes}`,
  ...ggufOf(`prism-ml/Ternary-Bonsai-${size}-gguf`, revision, `Ternary-Bonsai-${size}-PQ2_0.gguf`,
    `prism-ml/Ternary-Bonsai-${size}-unpacked`, originalRevision), download,
  weights: "ternary", conversion: {}, options: { ...options, ...PLAIN_ROPE }, template: QWEN3_AT_ONCE, ...start,
  generation: { steps: 0, temperature: 0.5, topp: 0.85, repetition_penalty: 1.0 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE });
// Models that huggingface.co serves and this page converts itself (src/python/llama2_convert.py, the code that builds
// the models above): plain Llama architecture, one safetensors file, a Unigram tokenizer.json or a sentencepiece
// model. revision pins the commit, so that nothing changes under the page. download is the size of model.safetensors.
export const hf = (repo, revision, tokenizer = "tokenizer.json") => ({ repo, revision, weights: "model.safetensors", config: "config.json", tokenizer });
