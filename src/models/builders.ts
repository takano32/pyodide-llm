// The entries of the list that come in families: one function a family, which writes its entries from what differs.
import type { ListOptions } from "./formats.ts";
import { ASK_JAPANESE, CHATML, PLAIN_ROPE, QWEN3_AT_ONCE, QWEN3_OWN_BOS, lfm25, lfm2Old, plamo, plamoTranslate } from "./formats.ts";
import { atOnce, greedy, sampled, thinking } from "./sampling.ts";
import type { Hf, OfHuggingFace, Sampled, Translation } from "../runtime/contracts/messages.ts";
import type { Dtype } from "../runtime/contracts/form.ts";

// An entry of the list (src/models.ts, which exports this type): an id, a name and a note (the note of the list: the
// window adds the devices it is for; T643.17: words for a visitor, which nothing reads: what the page reads are the
// entry's own words, japanese: it writes Japanese, the list's order; thinks: it thinks before it answers, the budget
// of a thought; converted: below, which tests/models-check.ts holds the note to), the group it is listed in (none: "site"), and what the worker is handed of it
// (src/page/address.ts weighs it into the worker's Listed, src/runtime/contracts/messages.ts): a model of this site
// (source: the repository it was built from, checkpoint, tokenizer and bytes), or one of Hugging Face converted in the
// page (hf, original: the repository whose model it is, download, conversion, shares, weights: the dtype it keeps where
// it names one, converted: the bytes of its weights once converted, in that dtype (else int8), loaded: its bytes once
// loaded where they are not those (llama.cpp's blocks kept, a table read from the kept file), rebuilt: its weights are not the
// original's; T634.2, wide: its forward pass at the context passes a 32-bit memory where its weights alone do not),
// the engine's options, the template and the sampling of an answer, the prompt and the placeholder it
// opens with
// T643.1: the kinds are types (T648.4: three, with a translation model's; T653: four, with a checkpoint of Hugging Face's at its address), and an entry is one of them: what one kind has, the other says it has not
// (?: undefined), so that either is read by the same names. generation: steps and temperature, which every entry says
// (the page's settings start from them)
export type Generation = Sampled & { steps: number; temperature: number };
type Shared = {
  group?: "site" | "original" | "hf";
  id: string;
  name: string;
  note: string;
  japanese: boolean;
  thinks?: boolean;
  wide?: boolean;
  options: ListOptions;
  template?: string;
  generation: Generation;
  prompt: string;
  placeholder: string;
};
// an entry of a model of this site (`make models`)
export type Site = Shared & {
  source: string;
  checkpoint: string;
  tokenizer: string;
  bytes: number;
  hf?: undefined;
  original?: undefined;
  download?: undefined;
  conversion?: undefined;
  shares?: undefined;
  weights?: undefined;
  converted?: undefined;
  loaded?: undefined;
  rebuilt?: undefined;
  translation?: undefined;
  url?: undefined;
};
// an entry of a model of Hugging Face, converted in the page: its repository, revision and file are named
export type Remote = Shared & {
  hf: Hf & { repo: string; revision: string; weights: string };
  original?: string;
  download: number;
  conversion?: OfHuggingFace["conversion"];
  shares?: string[];
  weights?: Dtype;
  converted?: number;
  loaded?: number;
  rebuilt?: boolean;
  source?: undefined;
  checkpoint?: undefined;
  tokenizer?: undefined;
  bytes?: undefined;
  translation?: undefined;
  url?: undefined;
};
// T648.4: an entry of a translation model of this site (Firefox Translations', `make models`): its files under
// models/translations/ and its languages (translation), the bytes of its model's file, and the source of the list's
// line of licenses; it answers with the translation of what was typed (no template, greedy: its generation says so)
export type Translating = Shared & {
  source: string;
  translation: Translation;
  bytes: number;
  template?: undefined;
  checkpoint?: undefined;
  tokenizer?: undefined;
  hf?: undefined;
  original?: undefined;
  download?: undefined;
  conversion?: undefined;
  shares?: undefined;
  weights?: undefined;
  converted?: undefined;
  loaded?: undefined;
  rebuilt?: undefined;
  url?: undefined;
};
// T653: an entry of a llama2.c checkpoint of Hugging Face, fetched at its address as it is (the worker's kind of two
// URLs, sources.ts): the repository it is of (source), the checkpoint's address at a pinned revision and the name of
// the tokenizer.bin of this site's models/ it is read with (url), and the checkpoint's bytes
export type Linked = Shared & {
  source: string;
  url: { checkpoint: string; tokenizer: string };
  bytes: number;
  template?: undefined;
  checkpoint?: undefined;
  tokenizer?: undefined;
  hf?: undefined;
  original?: undefined;
  download?: undefined;
  conversion?: undefined;
  shares?: undefined;
  weights?: undefined;
  converted?: undefined;
  loaded?: undefined;
  rebuilt?: undefined;
  translation?: undefined;
};
export type Entry = Site | Remote | Translating | Linked;
// what the entries a function below writes share, before their own id, name, note and sampling
type Common = Omit<Remote, "id" | "name" | "note" | "generation">;
// the weights a function below is given: a GGUF's with its original's vocabulary (ggufOf()), or a repository's (hf())
type Source = Pick<Remote, "original" | "hf">;
// the two revisions a function below pins, by name: two hashes side by side were swapped unseen (gemma3() took the
// GGUF's first and gemma4() the original's), which no type tells apart
export type Revisions = { gguf: string; original: string };

/** A Qwen3 twice (T124, the owner's "両方を別々に用意できないのか"): thinking first, and answering at once. The two
 * share their weights, and so a conversion kept in the browser; only the format differs. shares: both ids, for
 * kept.ts's replaced() (what either kept before its source changed goes, whichever form is opened first). sampling:
 * the two forms' generation settings where they are not Qwen3's. chat.languages: the note's languages where the card's
 * are not Japanese and English (T537), with 日本語 where it answered Japanese questions in Japanese here (T539).
 * chat.generation: both forms' generation settings (T643.22: it was the at-once form's alone), where sampling names
 * neither form's */
export function thinkingAndNot(id: string, name: string, source: Source, download: number, sizes: string, { languages = "日本語 / English", ...chat }: { languages?: string } & Partial<Remote> = {},
  formats: { thinking?: string; atOnce?: string } = {}, sampling: { thinking?: Generation; atOnce?: Generation } = {}): Entry[] {
  // T369: where nothing else is said, a Qwen3 as the list has had it: <|endoftext|> in front of ChatML (QWEN3_OWN_BOS and
  // the whole format, formats.ts's note on QWEN25). The converter alone would begin with the template's <|im_start|> now
  const common: Common = { group: "hf", ...source, download, conversion: {}, options: QWEN3_OWN_BOS, shares: [`${id}-thinking`, id],
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE, japanese: true, ...chat };
  return [
    // formats.thinking: where the model's format is not Qwen3's (T236)
    { ...common, id: `${id}-thinking`, name: `${name} (thinking)`, note: `thinks before it answers · ${languages} · ${sizes}`,
      thinks: true, generation: sampling.thinking ?? chat.generation ?? thinking, template: formats.thinking ?? CHATML },
    { ...common, id, name: `${name} (no thinking)`, note: `answers at once · ${languages} · ${sizes}`,
      generation: sampling.atOnce ?? chat.generation ?? atOnce, template: formats.atOnce ?? QWEN3_AT_ONCE },
  ];
}
/** T254: OpenBMB's MiniCPM5 (a Llama; English and Chinese), twice as a Qwen3 is: its chat_template begins the answer
 * with "<think>\n" where enable_thinking is true and with an empty thought where it is false (and with neither where
 * nothing is said, which is the format the converter reads). The real tokenizer begins every text with <s>, the
 * converter's BOS. sampling: its card's for either form */
export function miniCpm5(id: string, name: string, source: Source, download: number, sizes: string, converted: number, sampling: { thinking: number; atOnce: number }): Entry[] {
  const common: Common = { group: "hf", ...source, download, converted, japanese: false, conversion: {}, options: {}, shares: [`${id}-thinking`, id],
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" };
  const sampled = (temperature: number) => ({ steps: 0, temperature, topp: 0.95, repetition_penalty: 1.0 });
  return [
    { ...common, id: `${id}-thinking`, name: `${name} (thinking)`, note: `thinks before it answers · English / 中文 · ${sizes}`, thinks: true,
      generation: sampled(sampling.thinking), template: `${CHATML}<think>\n` },
    { ...common, id, name: `${name} (no thinking)`, note: `answers at once · English / 中文 · ${sizes}`,
      generation: sampled(sampling.atOnce), template: QWEN3_AT_ONCE },
  ];
}
/** T203 (T136's fourth stage): a Q8_0 GGUF's weights with the vocabulary and config.json of its original, which
 * tests/gguf_check.py tensors held them to (gguf.yml's candidates) */
export const ggufOf = (repo: string, revision: string, weights: string, original: string, originalRevision: string, tokenizer = "tokenizer.json"): Source =>
  ({ original, hf: { repo, revision, weights, vocabulary: { repo: original, revision: originalRevision, tokenizer } } });
// what lfm2() writes of an LFM2's format, by name: the converter's (nothing), or ChatML with the options of LFM2.5 or LFM2
const LFM2_FORMATS = {
  "lfm2.5": { options: lfm25, template: CHATML },
  "lfm2": { options: lfm2Old, template: CHATML },
  "converter's": {},
} satisfies Record<string, Pick<Remote, "options" | "template"> | Record<string, never>>;
/** an LFM2 of the list: the maker's own Q8_0 GGUF's weights with the vocabulary and config.json of its original.
 * format (T643.22: named, where an empty object meant "the converter's" and leaving it out meant LFM2.5's): "lfm2.5"
 * and "lfm2", the ChatML written here with that family's options; "converter's", the BOS, the format
 * and the specials the converter reads from the original's template */
export const lfm2 = (id: string, name: string, repo: string, revisions: Revisions, download: number, sizes: string, converted: number,
  format: keyof typeof LFM2_FORMATS, temperature = 0.1): Entry => ({
  group: "hf", id, name, note: `answers instructions · 日本語 / English · ${sizes}`, japanese: true, converted,
  ...ggufOf(`LiquidAI/${repo}-GGUF`, revisions.gguf, `${repo}-Q8_0.gguf`, `LiquidAI/${repo}`, revisions.original), download,
  conversion: {}, options: {}, ...LFM2_FORMATS[format], generation: { steps: 0, temperature, topp: 1.0, repetition_penalty: 1.05 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE,
});
/** T511.1, T511.3: a Shisa V2.1 of Shisa.AI (an LFM2 trained further for Japanese chat): mradermacher's Q8_0 GGUF's
 * weights with the vocabulary, config.json and chat template of Shisa's original. Its template calls no macro (the
 * LFM2's of Liquid AI do), so the BOS (<|startoftext|>), the format and the specials are the converter's, as the
 * LFM2.5 1.2B JP's are. The cards name no sampling and generation_config.json says do_sample and nothing more: the
 * LFM2 700M's (temperature 0.3, Liquid AI's LFM2 cards; a choice of the coordinator, TASKS.md's T511).
 * T528: another GGUF of the same repository (quantization: the name's word, Q4_K_M) with what it takes besides (more:
 * conversion.blocks and loaded) */
export const shisaLfm2 = (id: string, name: string, repo: string, revisions: Revisions, download: number, sizes: string, converted: number | undefined, quantization = "Q8_0",
  more: Partial<Remote> = {}): Entry => ({
  group: "hf", id, name, note: `answers instructions · 日本語 / English · ${sizes}`, japanese: true, converted,
  ...ggufOf(`mradermacher/${repo}-GGUF`, revisions.gguf, `${repo}.${quantization}.gguf`, `shisa-ai/${repo}`, revisions.original), download,
  conversion: {}, options: {}, generation: { steps: 0, temperature: 0.3, topp: 1.0, repetition_penalty: 1.05 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE, ...more,
});
/** T261.3: a Gemma 3 of the list (the text model; Google's original is gated): unsloth's Q8_0 GGUF's weights with the
 * vocabulary, config.json and chat template of unsloth's copy of the original, as Llama 3.2's entries name all three.
 * The BOS (<bos>) and the format (<start_of_turn>user … <start_of_turn>model) are the converter's, read from the
 * template; the stops are the converter's <bos> and <end_of_turn> with <eos> besides, as generation_config.json's
 * eos_token_id (1 and 106) has the real generate() stop. The sampling is generation_config.json's: do_sample with a
 * top-k of 64 and a top-p of 0.95, and transformers' temperature of 1.0 where it names none.
 * The notice the Gemma Terms of Use ask for is not in the note: at a phone's width the select never shows it (the
 * owner chooses where it goes, TASKS.md's T261.3) */
export const gemma3 = (size: string, revisions: Revisions, download: number, sizes: string, converted: number): Entry => ({
  group: "hf", id: `hf-gemma-3-${size.toLowerCase()}`, name: `Google Gemma 3 ${size}`,
  note: `answers instructions · 日本語 / English · ${sizes}`, japanese: true, converted,
  ...ggufOf(`unsloth/gemma-3-${size.toLowerCase()}-it-GGUF`, revisions.gguf, `gemma-3-${size.toLowerCase()}-it-Q8_0.gguf`,
    `unsloth/gemma-3-${size.toLowerCase()}-it`, revisions.original), original: `google/gemma-3-${size.toLowerCase()}-it`, download,
  conversion: {}, options: { stop_tokens: [1, 2, 106] },
  generation: { steps: 0, temperature: 1.0, topp: 0.95, top_k: 64, repetition_penalty: 1.0 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE,
});
/** T333: a Gemma 4 of the list (E2B, E4B: the language model of Google's original, whose vision and audio towers the
 * converter leaves; the original is not gated and under Apache 2.0, so no copy and no notice). T529: the weights of
 * llama.cpp's own Q8_0 GGUF of it (ggml-org's: half the original's download), with the vocabulary, config.json and chat
 * template of the original, which tests/gguf_check.py tensors held them to. Converted in the page; the table of every
 * layer's inputs stays in the file the conversion is kept in (OPFS) and is read a row a token (T329.2), so what the
 * memory holds is the checkpoint without it: `loaded` says that, as the note
 * cannot (it gives the file's size). The BOS (<bos>) and the format (<|turn>user … <|turn>model, the real
 * apply_chat_template's without a thought) are the converter's. The stops are generation_config.json's eos_token_id,
 * <eos>, <turn|> and <|tool_response> (1, 106, 50): the converter names <bos> and <eos> alone (T524), and an answer
 * then went on past <turn|>. The sampling is generation_config.json's: a temperature of 1.0, top-k 64, top-p 0.95 */
export const gemma4 = (size: string, revisions: Revisions, download: number, loaded: number, sizes: string, converted: number): Entry => ({
  group: "hf", id: `hf-gemma-4-${size.toLowerCase()}`, name: `Google Gemma 4 ${size}`,
  note: `answers instructions · 日本語 / English · ${sizes}`, japanese: true, converted,
  ...ggufOf(`ggml-org/gemma-4-${size}-it-GGUF`, revisions.gguf, `gemma-4-${size}-it-Q8_0.gguf`, `google/gemma-4-${size}-it`,
    revisions.original), download, loaded,
  conversion: {}, options: { stop_tokens: [1, 106, 50] },
  generation: { steps: 0, temperature: 1.0, topp: 0.95, top_k: 64, repetition_penalty: 1.0 },
  prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE,
});
/** T235, T246: a Ternary Bonsai of Prism ML (a ternary Qwen3) in one of its sizes: its PQ2_0 GGUF's weights with the
 * vocabulary, config.json and chat template of its -unpacked original. What the sizes share is here alone, so that one
 * change covers them all (the owner's open choices of T235's review: a word more in the note; Prism ML's
 * attribution). The RoPE is plain (PLAIN_ROPE, the owner's choice of 2026-10-11) and not config.json's yarn: the page
 * cuts the context at 4096, inside the 8192 or 16384 positions yarn begins to stretch, and there yarn's perplexity was
 * 2 to 7% the higher in every size (transformers and the engine; Qwen3's card advises against a static yarn on short
 * texts as well). It is said last, so that a size's own options (`start`) keep it. The original's template
 * always begins the answer with an empty thought (Qwen3's enable_thinking=false): the model has one form, which the
 * converter reads; it is written here (QWEN3_AT_ONCE) since T369, for the entries keep <|endoftext|> in front of it
 * (formats.ts's note on QWEN25). It names no BOS (QWEN3_OWN_BOS, which the 1.7B and the 4B bear: 1.5% better than nothing in front
 * on plain text; the 8B does not: 70% worse, as a Qwen3 8B is, and begins at QWEN3_FROM_IM_START: `start`, T250's review).
 * The sampling is the originals' generation_config.json, the same file in the three (its top-k the page's sampler
 * has not) */
export const ternaryBonsai = (size: string, revisions: Revisions, download: number, sizes: string, converted: number,
  start: Partial<Remote> = {}): Entry =>
  bonsai("Ternary-Bonsai", "Ternary Bonsai", "PQ2_0", "ternary", "日本語 / English · ternary weights", size, revisions, download, sizes, converted, start,
    { japanese: true, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE });
/** T286: Prism ML's 1-bit Bonsai (1.7B ...), every weight -1 or +1 times a scale of its 128: a Q1_0 GGUF
 * (convert/readers.py's q1_0, T286.1), which the binary dtype holds as it is (T634.2: one bit a weight, Q1_0's 18
 * bytes a group of 128; the ternary dtype before, twice that). The rest as the ternary ones' above,
 * from its own -unpacked original. The card says English alone; `japanese` is for a size that answered tests/answers.ts's
 * Japanese questions in Japanese without breaking (T539: the 4B, T286.3). The 1.7B's Japanese answers looped and
 * named mountains that are not there (T286.2) */
export const binaryBonsai = (size: string, revisions: Revisions, download: number, sizes: string, converted: number,
  japanese: boolean, start: Partial<Remote> = {}): Entry =>
  japanese
    ? bonsai("Bonsai", "Bonsai", "Q1_0", "binary", "日本語 / English · 1-bit weights", size, revisions, download, sizes, converted, start,
      { japanese, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE })
    : bonsai("Bonsai", "Bonsai", "Q1_0", "binary", "English · 1-bit weights", size, revisions, download, sizes, converted, start,
      { japanese, prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" });
const bonsai = (repo: string, name: string, quantization: string, dtype: Dtype, weights: string, size: string, revisions: Revisions,
  download: number, sizes: string, converted: number, { options = QWEN3_OWN_BOS, ...start }: Partial<Remote>,
  asked: Pick<Remote, "japanese" | "prompt" | "placeholder">): Entry => ({
  group: "hf", id: `hf-${repo.toLowerCase()}-${size.toLowerCase()}`, name: `Prism ML ${name} ${size}`,
  note: `answers at once · ${weights} · ${sizes}`, converted,
  ...ggufOf(`prism-ml/${repo}-${size}-gguf`, revisions.gguf, `${repo}-${size}-${quantization}.gguf`,
    `prism-ml/${repo}-${size}-unpacked`, revisions.original), download,
  weights: dtype, conversion: {}, options: { ...options, ...PLAIN_ROPE }, template: QWEN3_AT_ONCE, ...start,
  generation: { steps: 0, temperature: 0.5, topp: 0.85, repetition_penalty: 1.0 }, ...asked });
// Models that huggingface.co serves and this page converts itself (src/python/llama2_convert.py, the code that builds
// the models above): plain Llama architecture, one safetensors file, a Unigram tokenizer.json or a sentencepiece
// model. revision pins the commit, so that nothing changes under the page. download is the size of model.safetensors.
export const hf = (repo: string, revision: string, tokenizer = "tokenizer.json"): Remote["hf"] => ({ repo, revision, weights: "model.safetensors", config: "config.json", tokenizer });
/** T649.3: PFN's PLaMo 3 Translate in one of its sizes, twice: English into Japanese and Japanese into English (the
 * owner's T649 (1): its maker PFN in front and PLaMo in the name, as the PLaMo Community License asks of a model built
 * with it, and "Built with PLaMo" in the note and on the line of its license in About). T649.4 (the owner's T649 (6)):
 * the weights from PFN's own Q4_K_M GGUF, llama.cpp's blocks kept as they are (conversion.blocks, as Underdog Saluki
 * 27B's: the embedding, wo, w1 and w3, about 58% of the weights) and the rest widened to int8 (wq, wk, wv, w2 and the
 * classifier, Q6_K or of a stack with a layer of it); a third of the int8's fetch, about 0.7 of its memory, 0.8 to
 * 0.87 times its speed of writing. The GGUF's first revisions are not served (their LFS copies answer 403): these are
 * the ones of 2026-11-06, which changed nothing but the chat template. The vocabulary (its own tokenizer.jsonl, T649.1),
 * config.json and chat template the original's (hf.vocabulary); rebuilt: tests/gguf_check.py tensors holds a Q8_0 to
 * its original within 1e-3, which 4 bits are not (Q4_K 0.07 to 0.08 of it, Q6_K 0.02: TASKS.md's T649.4). The format
 * is the request of its chat template, the BOS (1) in front (formats.ts's plamoTranslate); greedy, as every example of
 * its card translates (do_sample=False, temperature 0). The two share their weights (shares). download: the GGUF's
 * bytes; loaded: the checkpoint's (tests/quant_sizes.py) */
export function plamo3Translate(size: string, original: string, gguf: string, download: number, loaded: number): Entry[] {
  const lower = size.toLowerCase(), repo = `pfnet/plamo-3-translate-${lower}`, id = `hf-plamo-3-translate-${lower}`;
  const said = (bytes: number) => bytes < 1e9 ? `${Math.round(bytes / 1e6)} MB` : `${(bytes / 1e9).toFixed(1)} GB`;
  const note = `日本語, English, translation only, 4-bit weights, fetches ${said(download)} (GGUF), kept mostly as it is: ${said(loaded)}, Built with PLaMo`;
  const common: Omit<Common, "prompt" | "placeholder"> = { group: "hf",
    ...ggufOf(`${repo}-GGUF`, gguf, `plamo-3-translate-${lower}-Q4_K_M.gguf`, repo, original, "tokenizer.jsonl"), download,
    conversion: { dtype: "int8", blocks: true }, loaded, rebuilt: true, japanese: true, options: plamo, shares: [`${id}-en-ja`, `${id}-ja-en`] };
  return [
    { ...common, id: `${id}-en-ja`, name: `PFN PLaMo 3 Translate ${size} English → Japanese`, note, generation: greedy,
      template: plamoTranslate("en", "ja"), prompt: "The weather is nice today. Let's go for a walk.",
      placeholder: "Type text to translate into Japanese (e.g. The weather is nice today.)" },
    { ...common, id: `${id}-ja-en`, name: `PFN PLaMo 3 Translate ${size} Japanese → English`, note, generation: greedy,
      template: plamoTranslate("ja", "en"), prompt: "今日はいい天気ですね。散歩に行きましょう。",
      placeholder: "英語に訳す文を入力（例: 今日はいい天気ですね。）" },
  ];
}
