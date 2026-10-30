// How the list's entries sample: the sets the models' cards recommend.

// greedy decoding makes small models loop, so the Japanese ones sample, and penalize repetition
// steps: 0 is as many tokens as the context of the model holds. Nothing here holds a model back by default.
export const sampled = (repetition_penalty) => ({ steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty });
export const greedy = { steps: 0, temperature: 0.0 };

// the sampling of Qwen3's model card for either form, without its top-k of 20: the engine has one since T.274, on the
// CPU alone, and these models' answers are the GPU's where it is faster (and the GPU's alone from 3B up)
export const thinking = { steps: 0, temperature: 0.6, topp: 0.95, repetition_penalty: 1.0 };
export const atOnce = { steps: 0, temperature: 0.7, topp: 0.8, repetition_penalty: 1.0 };
// T.274: the sampling of the cards of the Qwen3.5 family, whose models are on the CPU (hybrid attention has no GPU path):
// a top-k of 20 and a presence penalty with all of them. QWEN35_SAMPLING: the 4B's and the 9B's for general tasks, and
// NeoHorse-1's (its card measured with the thinking set). The 0.8B's and the 2B's cards name the same set with
// thinking; without, 1.0, top-p 1.0 and a presence penalty of 2.0 "for text" and these 0.7 and 0.8 "for VL tasks":
// the 0.8B wrote worse with the former (TASKS.md's T.274: a French sentence that means nothing, where it translated
// rightly with 0.7 and 0.8), so the latter. Agents-A1's card has one set, Bonsai 2's a min-p where it thinks
const sampling35 = (temperature, topp, more) => ({ steps: 0, temperature, topp, repetition_penalty: 1.0, top_k: 20, ...more });
export const QWEN35_SAMPLING = { thinking: sampling35(1.0, 0.95, { presence_penalty: 1.5 }), atOnce: sampling35(0.7, 0.8, { presence_penalty: 1.5 }) };
export const AGENTS_A1_SAMPLING = { thinking: sampling35(0.85, 0.95, { presence_penalty: 1.1 }), atOnce: sampling35(0.85, 0.95, { presence_penalty: 1.1 }) };
export const BONSAI_2_SAMPLING = { thinking: sampling35(1.0, 0.95, { min_p: 0.05 }), atOnce: QWEN35_SAMPLING.atOnce };
// T.511.2: the card of the Japanese SFT v2.5 of Qwen3.5 0.8B: 1.0, 0.95 and the top-k with thinking (no presence
// penalty), and for chat without thinking 1.0, top-p 1.0, the top-k and a presence penalty of 2.0
export const QWEN35_JA_SFT_SAMPLING = { thinking: sampling35(1.0, 0.95), atOnce: sampling35(1.0, 1.0, { presence_penalty: 2.0 }) };
// T.364.3: Underdog Saluki 27B's card: 0.6, 0.95 and the top-k where it thinks, greedy where it answers at once. Not a
// Qwen3.5's set: with its presence penalty one answer of four fell to doubting itself over and over (TASKS.md's T.364)
export const SALUKI_SAMPLING = { thinking: sampling35(0.6, 0.95), atOnce: greedy };
// its card: "Use temperature=1.0 and top_p=0.95 across all tasks", thinking or not
export const graniteSampling = { steps: 0, temperature: 1.0, topp: 0.95, repetition_penalty: 1.0 };
// T.343: Spark-X2.5's card ("temperature=1.0, top_p=0.95, and top_k=-1", no top-k) and its generation_config.json (the
// same, with presence and frequency penalties of 0), thinking or not
const spark = { steps: 0, temperature: 1.0, topp: 0.95, repetition_penalty: 1.0 };
export const SPARK_SAMPLING = { thinking: spark, atOnce: spark };

// T.520: the budget of a thought, in tokens, of every entry that thinks (its note says "thinks"): a thought that goes on
// this long is closed (its </think> written in place of a draw) and the model answers. The owner: as high as it can be,
// so that it holds no model back. Twelve questions (tests/answers.mjs) at the context's whole length, on CI: the
// thoughts that closed by themselves were 2,591 tokens at the longest outside the Qwen3.5 family, and within it 3,860;
// the ones that never closed ran to the end of the context and left no answer at all (docs/notes/
// t520-thinking-budget-2026-10-16.md). The context is 4096 for all of them, and 512 of it is left for the prompt and
// the answer: a thought of the Qwen3.5 family longer than this (3 of its 32 that closed: 3,701 to 3,860) is closed a little early.
export const THOUGHT_BUDGET = 4096 - 512;
/** T.520: whether an entry thinks before it answers (its note says so, as every one of them does). */
export const thinks = (entry) => /^thinks\b/.test(entry.note ?? "");
