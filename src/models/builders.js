// The entries of the list that come in families: one function a family, which writes its entries from what differs.
import { ASK_JAPANESE, CHATML, QWEN3_AT_ONCE, QWEN3_OWN_BOS, lfm25 } from "./formats.js";
import { atOnce, sampled, thinking } from "./sampling.js";

/** A Qwen3 twice (T124, the owner's "両方を別々に用意できないのか"): thinking first, and answering at once. The two
 * share their weights, and so a conversion kept in the browser; only the format differs. shares: both ids, for
 * kept.js's replaced() (what either kept before its source changed goes, whichever form is opened first). sampling:
 * the two forms' generation settings where they are not Qwen3's */
export function thinkingAndNot(id, name, source, download, sizes, chat = {}, formats = {}, sampling = {}) {
  // T369: where nothing else is said, a Qwen3 as the list has had it: <|endoftext|> in front of ChatML (QWEN3_OWN_BOS and
  // the whole format, formats.js's note on QWEN25). The converter alone would begin with the template's <|im_start|> now
  const common = { group: "hf", ...source, download, conversion: {}, options: QWEN3_OWN_BOS, shares: [`${id}-thinking`, id],
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE, ...chat };
  return [
    // formats.thinking: where the model's format is not Qwen3's (T236)
    { ...common, id: `${id}-thinking`, name: `${name} (thinking)`, note: `thinks before it answers · 日本語 / English · ${sizes}`,
      generation: sampling.thinking ?? thinking, template: formats.thinking ?? CHATML },
    { ...common, id, name: `${name} (no thinking)`, note: `answers at once · 日本語 / English · ${sizes}`,
      generation: sampling.atOnce ?? chat.generation ?? atOnce, template: formats.atOnce ?? QWEN3_AT_ONCE },
  ];
}
/** T254: OpenBMB's MiniCPM5 (a Llama; English and Chinese), twice as a Qwen3 is: its chat_template begins the answer
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
/** T203 (T136's fourth stage): a Q8_0 GGUF's weights with the vocabulary and config.json of its original, which
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
/** T235, T246: a Ternary Bonsai of Prism ML (a ternary Qwen3) in one of its sizes: its PQ2_0 GGUF's weights with the
 * vocabulary, config.json and chat template of its -unpacked original. What the sizes share is here alone, so that one
 * change covers them all (the owner's open choices of T235's review: config.json's yarn or a plain RoPE, which would
 * be rope_scaling: {} in these options; a word more in the note; Prism ML's attribution). The original's template
 * always begins the answer with an empty thought (Qwen3's enable_thinking=false): the model has one form, which the
 * converter reads; it is written here (QWEN3_AT_ONCE) since T369, for the entries keep <|endoftext|> in front of it
 * (formats.js's note on QWEN25). It names no BOS (QWEN3_OWN_BOS, which the 1.7B and the 4B bear: 1.5% better than nothing in front
 * on plain text; the 8B does not: 70% worse, as a Qwen3 8B is, and begins at QWEN3_FROM_IM_START: `start`, T250's review).
 * The sampling is the originals' generation_config.json, the same file in the three (its top-k the page's sampler
 * has not) */
export const ternaryBonsai = (size, revision, originalRevision, download, sizes, start = {}) => ({
  group: "hf", id: `hf-ternary-bonsai-${size.toLowerCase()}`, name: `Ternary Bonsai ${size}`,
  note: `answers at once · 日本語 / English · ternary weights · ${sizes}`,
  ...ggufOf(`prism-ml/Ternary-Bonsai-${size}-gguf`, revision, `Ternary-Bonsai-${size}-PQ2_0.gguf`,
    `prism-ml/Ternary-Bonsai-${size}-unpacked`, originalRevision), download,
  weights: "ternary", conversion: {}, options: QWEN3_OWN_BOS, template: QWEN3_AT_ONCE, ...start,
  generation: { steps: 0, temperature: 0.5, topp: 0.85, repetition_penalty: 1.0 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE });
// Models that huggingface.co serves and this page converts itself (public/llama2_convert.py, the code that builds
// the models above): plain Llama architecture, one safetensors file, a Unigram tokenizer.json or a sentencepiece
// model. revision pins the commit, so that nothing changes under the page. download is the size of model.safetensors.
export const hf = (repo, revision, tokenizer = "tokenizer.json") => ({ repo, revision, weights: "model.safetensors", config: "config.json", tokenizer });
