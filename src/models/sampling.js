// How the list's entries sample: the sets the models' cards recommend.

// greedy decoding makes small models loop, so the Japanese ones sample, and penalize repetition
// steps: 0 is as many tokens as the context of the model holds. Nothing here holds a model back by default.
export const sampled = (repetition_penalty) => ({ steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty });
export const greedy = { steps: 0, temperature: 0.0 };

// the sampling of Qwen3's model card for either form, without its top-k of 20: the engine has one since T274, on the
// CPU alone, and these models' answers are the GPU's where it is faster (and the GPU's alone from 3B up)
export const thinking = { steps: 0, temperature: 0.6, topp: 0.95, repetition_penalty: 1.0 };
export const atOnce = { steps: 0, temperature: 0.7, topp: 0.8, repetition_penalty: 1.0 };
// T274: the sampling of the cards of the Qwen3.5 family, whose models are on the CPU (hybrid attention has no GPU path):
// a top-k of 20 and a presence penalty with all of them. QWEN35_SAMPLING: the 4B's and the 9B's for general tasks, and
// NeoHorse-1's (its card measured with the thinking set). The 0.8B's and the 2B's cards name the same set with
// thinking; without, 1.0, top-p 1.0 and a presence penalty of 2.0 "for text" and these 0.7 and 0.8 "for VL tasks":
// the 0.8B wrote worse with the former (TODO.md's T274: a French sentence that means nothing, where it translated
// rightly with 0.7 and 0.8), so the latter. Agents-A1's card has one set, Bonsai 2's a min-p where it thinks
const sampling35 = (temperature, topp, more) => ({ steps: 0, temperature, topp, repetition_penalty: 1.0, top_k: 20, ...more });
export const QWEN35_SAMPLING = { thinking: sampling35(1.0, 0.95, { presence_penalty: 1.5 }), atOnce: sampling35(0.7, 0.8, { presence_penalty: 1.5 }) };
export const AGENTS_A1_SAMPLING = { thinking: sampling35(0.85, 0.95, { presence_penalty: 1.1 }), atOnce: sampling35(0.85, 0.95, { presence_penalty: 1.1 }) };
export const BONSAI_2_SAMPLING = { thinking: sampling35(1.0, 0.95, { min_p: 0.05 }), atOnce: QWEN35_SAMPLING.atOnce };
// its card: "Use temperature=1.0 and top_p=0.95 across all tasks", thinking or not
export const graniteSampling = { steps: 0, temperature: 1.0, topp: 0.95, repetition_penalty: 1.0 };
