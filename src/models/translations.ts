// T648.4: Mozilla's Firefox Translations models of English and Japanese, which this site serves (`make models`:
// fetch_translations.py names each file, the Makefile cuts the models into parts of 8 MiB under
// models/translations/<folder>/). They translate what was typed, sentence by sentence, greedily as Firefox does: no
// template and no sampling (the worker's translate.ts, and src/python/engine/translating.py). bytes: the model's file.
// ja-en comes twice (the owner's T648 (2)): 3.0, the model of Firefox on the desktop ("base", 6 + 2 layers of 512),
// and 3.1, Android's ("base-memory", 6 + 4 of 384).
import type { Entry, Generation } from "./builders.ts";

// what a translation is written with: greedy, the whole bound of the decoder (the settings are not read: translating.py)
const GREEDY: Generation = { steps: 0, temperature: 0.0 };
const SOURCE = "mozilla/translations";
const INTO_JAPANESE = "Type text to translate into Japanese (e.g. The weather is nice today.)";
const INTO_ENGLISH = "英語に訳す文を入力（例: 今日はいい天気ですね。）";

export const TRANSLATIONS: Entry[] = [
  { id: "firefox-en-ja", name: "Mozilla Firefox Translations English → Japanese", note: "日本語, English, translation only, 44 MB",
    source: SOURCE, bytes: 43849787, options: {}, generation: GREEDY, prompt: "The weather is nice today. Let's go for a walk.", placeholder: INTO_JAPANESE,
    translation: { folder: "en-ja-3.0", model: "model.enja.intgemm.alphas.bin", source: "srcvocab.enja.spm", target: "trgvocab.enja.spm", from: "en", to: "ja" } },
  { id: "firefox-ja-en", name: "Mozilla Firefox Translations Japanese → English (desktop)", note: "日本語, English, translation only, 60 MB",
    source: SOURCE, bytes: 59504955, options: {}, generation: GREEDY, prompt: "今日はいい天気ですね。散歩に行きましょう。", placeholder: INTO_ENGLISH,
    translation: { folder: "ja-en-3.0", model: "model.jaen.intgemm.alphas.bin", source: "vocab.jaen.spm", target: "vocab.jaen.spm", from: "ja", to: "en" } },
  { id: "firefox-ja-en-android", name: "Mozilla Firefox Translations Japanese → English (Android)", note: "日本語, English, translation only, 44 MB",
    source: SOURCE, bytes: 43977787, options: {}, generation: GREEDY, prompt: "今日はいい天気ですね。散歩に行きましょう。", placeholder: INTO_ENGLISH,
    translation: { folder: "ja-en-3.1", model: "model.jaen.intgemm.alphas.bin", source: "vocab.jaen.spm", target: "vocab.jaen.spm", from: "ja", to: "en" } },
];
