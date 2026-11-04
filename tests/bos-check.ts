// T.401 (1): an entry's bos and its format go together. The converter takes a model's BOS from the model's own chat
// template (convert/conversion.py's beginning(): the tokenizer's BOS where the template begins with it or with text,
// the template's first token where it begins with another token the engine reads as one), and an entry's template
// replaces the model's format without changing that BOS. So an entry that writes one of the two and leaves the other
// to the converter can send <|im_start|><|im_start|>user… (the format's first token twice) or <|endoftext|>user…
// (the format begun after a mark nobody wrote): no error, a worse text (AGENTS.md's "モデルを一覧に足す", 3). What
// the entry's text alone says, with no fetch:
//
//   (a) an entry that writes a bos and no template is one that continues a text (its placeholder asks for the
//       beginning of one): a chat model's bos goes with the format it begins;
//   (b) where the template begins with a mark (a token such as <|im_start|>) or with a role (user, system, assistant:
//       the format of a model whose BOS is the mark the role follows), FIRST below says which token stands in front
//       of the format in that entry's vocabulary: the entry's own bos, or the converter's (null) where the entry
//       writes none. That token is not the format's first mark (the mark twice), and in front of a role it is the
//       mark that opens a turn;
//   (c) every row of FIRST is some entry's (a row left behind is a fact nobody holds any more).
//
// FIRST is a second copy (AGENTS.md: the facts of a table get a second copy in the tests): the token in front of each
// such entry's format, from its tokenizer and its chat template, which tests/format_check.py holds to the real
// apply_chat_template where the files are fetched. A new entry of those shapes needs its row; one whose template
// begins with text is not checked here (a text that the model's template begins with a token cannot be told from
// the entry: tests/format_check.py). What this does not see: a bos that names the wrong token of the right kind, and
// a format whose text the vocabulary reads otherwise.
// So it is more a net that says a change than a finder of errors: FIRST copies the list's own (bos, repository) pairs,
// and a bos taken off fails on the key, not on (b); the mark twice is met only where the converter's BOS or the
// entry's stands in front of a mark (a format that begins with a role is not), and a bos that names another token of
// the right kind passes. The texts of the rows were held to the real tokenizers on 2026-11-04 (TASKS.md's T.401)
//
//   node tests/bos-check.ts
import assert from "node:assert/strict";
import { MODELS } from "../src/models.ts";
import type { Entry } from "../src/models.ts";
import { JAPANESE, STORY } from "../src/models/formats.ts";

// a token in angle brackets at the start of a format: <|im_start|>, <｜User｜>, <think>
const MARK = /^(<\|[^|<>\s]+\|>|<｜[^｜<>\s]+｜>|<[A-Za-z_][A-Za-z0-9_]*>)/;
// a format that begins after the mark of its turn (CHATML_AFTER_START, QWEN35_THINKING, HARMONY_AFTER_START …)
const ROLE = /^(user|system|assistant)\b/;
// the marks a turn opens with, which a role follows
const OPENERS = ["<|im_start|>", "<|start|>"];

type Row = { bos: number | null; text: string; repos: string[] };
/** The token in front of the format, for the entries whose format begins with a mark or a role: bos is the entry's
 * (null: the entry writes none, and the converter's is the tokenizer's BOS that the model's template begins with),
 * text that token in the vocabulary of the repositories (the vocabulary's, hf.vocabulary or hf). */
const FIRST: Row[] = [
  // the converter's BOS: the model's template begins with the tokenizer's (LFM2's and MiniCPM5's {{ bos_token }},
  // Llama 3's <|begin_of_text|>), or with text that this vocabulary has no token of (TinyLlama's and zephyr's <|user|>,
  // after the <s> their tokenizer adds)
  { bos: null, text: "<|startoftext|>", repos: ["LiquidAI/LFM2-350M-ENJP-MT", "LiquidAI/LFM2-700M", "LiquidAI/LFM2.5-1.2B-Instruct",
    "LiquidAI/LFM2.5-2.6B", "LiquidAI/LFM2.5-230M", "LiquidAI/LFM2.5-350M"] },
  { bos: null, text: "<s>", repos: ["openbmb/MiniCPM5-1B", "openbmb/MiniCPM5-2B", "HuggingFaceH4/zephyr-7b-beta",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"] },
  { bos: null, text: "<|begin_of_text|>", repos: ["elyza/Llama-3-ELYZA-JP-8B"] },
  // SmolLM2's BOS is <|im_start|> itself (bos_token_id 1), which its template begins with
  { bos: null, text: "<|im_start|>", repos: ["bartowski/SmolLM2-135M-Instruct-GGUF"] },
  // the entry's own BOS in front of the template's mark (QWEN3_OWN_BOS, eurollm, harmony, spark, DeepSeek's)
  { bos: 151643, text: "<|endoftext|>", repos: ["Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507", "SakanaAI/TinySwallow-1.5B-Instruct-GGUF", "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
    "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "bartowski/Qwen2.5-3B-Instruct-GGUF", "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF", "cyberagent/CAT-Thinking-8B", "elyza/ELYZA-Shortcut-1.0-Qwen-7B",
    "prism-ml/Bonsai-1.7B-unpacked", "prism-ml/Bonsai-4B-unpacked", "prism-ml/Ternary-Bonsai-1.7B-unpacked",
    "prism-ml/Ternary-Bonsai-4B-unpacked", "tokyotech-llm/Medical-Qwen3-Swallow-8B", "tokyotech-llm/Qwen3-Swallow-8B-CPT-v0.2",
    "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", "tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2"] },
  { bos: 1, text: "<s>", repos: ["utter-project/EuroLLM-1.7B-Instruct"] },
  { bos: 1, text: "<|startoftext|>", repos: ["llm-jp/llm-jp-4.1-8b-thinking"] },
  { bos: 0, text: "<｜start▁of▁sentence｜>", repos: ["Takenoko12345678/Spark-X2.5-4B-Japanese", "XHToken/Spark-X2.5-1.7B",
    "XHToken/Spark-X2.5-4B", "XHToken/Spark-X2.6-1.7B", "XHToken/Spark-X2.6-4B"] },
  { bos: 151646, text: "<｜begin▁of▁sentence｜>", repos: ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"] },
  // the entry's BOS is the template's first mark, and the format begins after it (T.264's form)
  { bos: 151644, text: "<|im_start|>", repos: ["AXCXEPT/Qwen3-EZO-8B-beta", "Aratako/Qwen3-8B-RP-v0.1", "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B", "abeja/ABEJA-Qwen3-14B-Agentic-256k-v0.1", "prism-ml/Bonsai-8B-unpacked",
    "prism-ml/Ternary-Bonsai-8B-unpacked", "shisa-ai/shisa-v2.1-qwen3-8b"] },
  { bos: 248045, text: "<|im_start|>", repos: ["InternScience/Agents-A1-4B", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B", "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2.5", "TokenRhythm/NeoHorse-1-4B",
    "TokenRhythm/NeoHorse-1-9B", "prism-ml/Bonsai-27B-unpacked"] },
  { bos: 128011, text: "<|im_start|>", repos: ["HuggingFaceTB/SmolLM3-3B"] },
  { bos: 100256, text: "<|im_start|>", repos: ["ibm-granite/granite-4.2-3b", "ibm-granite/granite-4.2-8b"] },
  { bos: 100264, text: "<|im_start|>", repos: ["shisa-ai/shisa-v2.1-unphi4-14b"] },
  { bos: 10, text: "<|start|>", repos: ["llm-jp/llm-jp-4-8b-instruct", "llm-jp/llm-jp-4-8b-thinking"] },
];

const vocabularyOf = (entry: Entry) => entry.hf?.vocabulary?.repo ?? entry.hf?.repo ?? entry.source;

/** What is wrong with the entries' bos and format, one line each (an empty list: nothing), and the rows of FIRST used. */
export function unpaired(models: Entry[], first = FIRST): { said: string[]; used: Set<Row> } {
  const said: string[] = [];
  const used = new Set<Row>();
  for (const entry of models) {
    const bos = entry.options?.bos ?? null;
    const template = entry.template;
    if (!template) {
      // (a)
      if (bos !== null && ![JAPANESE, STORY].includes(entry.placeholder ?? "")) {
        said.push(`${entry.id} writes a bos (${bos}) and no template: a chat model's bos goes with the format it begins`);
      }
      continue;
    }
    const mark = template.match(MARK)?.[1];
    const role = ROLE.test(template);
    if (!mark && !role) continue;
    // (b)
    const repo = vocabularyOf(entry);
    const row = first.find((row) => row.bos === bos && row.repos.includes(repo!));
    if (!row) {
      const others = first.filter((row) => row.repos.includes(repo!)).map((row) => row.bos);
      said.push(`${entry.id}: its format begins with ${mark ?? "a role"} and ${bos === null ? "it leaves the bos to the converter" :
        `its bos is ${bos}`}, which no row of FIRST says for ${repo}` + (others.length ? ` (the rows say ${others.join(", ")}: ` +
        "the bos and the format are written together, or both left to the converter)" : " (add the token in front of its format)"));
      continue;
    }
    used.add(row);
    if (mark && row.text === mark) {
      said.push(`${entry.id}: ${row.text} in front of a format that begins with ${mark}: the mark twice ` +
        `(${bos === null ? "the converter's bos" : `bos ${bos}`}; the format begins after it)`);
    }
    if (role && !OPENERS.includes(row.text)) {
      said.push(`${entry.id}: its format begins with a role and ${row.text} is in front of it, not the mark of a turn`);
    }
  }
  return { said, used };
}

const { said, used } = unpaired(MODELS);
assert.deepEqual(said, [], said.join("\n"));
// (c)
const left = FIRST.filter((row) => !used.has(row));
assert.deepEqual(left, [], `rows of FIRST no entry is: ${JSON.stringify(left)}`);
for (const row of FIRST) assert.equal(new Set(row.repos).size, row.repos.length, `a repository twice in the row of ${row.text}`);

// the ways to break it, each one entry at a time: every entry that writes a bos with a checked format is said when its
// bos is taken off (the bos left to the converter, T.401's case), and every entry that leaves it is said when it is
// given the format's first mark as its BOS (the mark twice)
const checked = MODELS.filter((entry) => entry.template && (MARK.test(entry.template) || ROLE.test(entry.template)));
let broken = 0;
for (const entry of checked) {
  const bos = entry.options?.bos;
  const mark = entry.template!.match(MARK)?.[1];
  const changed = bos !== undefined ? { ...entry, options: { ...entry.options, bos: undefined } } : mark ? {
    ...entry, options: { ...entry.options, bos: -1 } } : undefined;
  if (!changed) continue;
  // the mark twice: a row that says the entry's -1 is the format's first mark
  const first = bos !== undefined ? FIRST : [...FIRST, { bos: -1, text: mark!, repos: [vocabularyOf(entry)!] }];
  assert.ok(unpaired([changed], first).said.length > 0, `${entry.id} with its bos ${bos !== undefined ? "taken off" : "made its mark"} is not said`);
  broken += 1;
}
// and every chat entry that writes both, with its template taken off (its bos left alone): the entries of a
// continuation that write "{prompt}" with their bos (RakutenAI 2.0 mini, TinySwallow 1.5B) are what (a) lets through
const chats = MODELS.filter((entry) => entry.options?.bos !== undefined && entry.template &&
  ![JAPANESE, STORY].includes(entry.placeholder ?? ""));
for (const entry of chats) {
  assert.ok(unpaired([{ ...entry, template: undefined }]).said.length > 0, `${entry.id} with its template taken off is not said`);
}
console.log(`bos-check: ${checked.length} formats that begin with a mark or a role, ${FIRST.length} rows; ` +
  `${broken} entries said with their bos broken, ${chats.length} with their template taken off`);
