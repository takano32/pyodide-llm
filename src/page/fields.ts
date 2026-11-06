// src/page/fields.ts (T643.7): the generation settings' tables, which settings.ts goes round: what the URL may set and
// within what, and the fields of the settings' sheet. Nothing of a page is read here, so Node imports it
// (tests/page-parts-check.ts). A setting the worker is sent is a key of Sampled (src/runtime/contracts/messages.ts):
// LIMITS has every one of them, or tsc says which is missing.
import type { Sampled } from "../runtime/contracts/messages.ts";
import type { Model } from "./model.ts";

// the settings that have a field in the sheet (index.astro: an <input> of the id and an <output> of the id and "-value")
type Fielded = "temperature" | "steps" | "topp" | "repetition_penalty";
export type Settings = Sampled & Record<Fielded, number>;
// steps: 0 stands for as many tokens as the model's context holds, which is the default of every model
// T274: top_k, min_p and presence_penalty have no field: a model's own (its card's) or the link's
export const defaults = (current: Model): Settings => ({ topp: 0.9, repetition_penalty: 1.0, ...current.generation });

// The URL may set any of them (T43), so that a link reproduces an answer: ?temperature=0.7&seed=1234&steps=128
// (&topp= and &repetition_penalty= too, and &top_k=, &min_p= and &presence_penalty=; T520: &thinking=, the budget of a
// thought in tokens, 0 for none). A value that is not a number, or outside what the field allows, is
// ignored; a whole one is rounded.
export const LIMITS: Record<keyof Sampled, { low: number; high: number; whole?: true }> = {
  temperature: { low: 0, high: 1.5 }, topp: { low: 0.05, high: 1 }, steps: { low: 0, high: 1 << 20, whole: true },
  repetition_penalty: { low: 1, high: 2 }, seed: { low: 0, high: 0xffffffff, whole: true },
  top_k: { low: 0, high: 1 << 20, whole: true }, min_p: { low: 0, high: 1 }, presence_penalty: { low: 0, high: 2 },
  thinking: { low: 0, high: 1 << 20, whole: true },
};
/** What the link's parameters set of the settings. */
export function linkSettings(parameters: { get(name: string): string | null }): Sampled {
  const linked: Sampled = {};
  for (const [name, { low, high, whole }] of Object.entries(LIMITS) as [keyof Sampled, typeof LIMITS[keyof Sampled]][]) {
    const given = parameters.get(name);
    const value = given === null ? NaN : Number(given);
    if (given !== null && Number.isFinite(value) && value >= low && value <= high) {
      linked[name] = whole ? Math.round(value) : value;
    }
  }
  return linked;
}

// The sheet's fields: the element's id, the setting, and the words of its <output>. Steps are shown within the
// context (0, all of it, as its length), and a slider at the context's length is 0 again.
type Field = { id: string; key: Fielded; said: (value: number) => string;
               shown?: (value: number, longest: number) => number; taken?: (value: number, longest: number) => number };
export const FIELDS: readonly Field[] = [
  { id: "temperature", key: "temperature", said: (value) => (value ? value.toFixed(2) : "greedy") },
  { id: "steps", key: "steps", said: String, shown: (value, longest) => (value > 0 ? Math.min(value, longest) : longest),
    taken: (value, longest) => (value >= longest ? 0 : value) },
  { id: "topp", key: "topp", said: (value) => value.toFixed(2) },
  { id: "penalty", key: "repetition_penalty", said: (value) => value.toFixed(2) },
];

/** Whether the settings are other than the model's own ("all of the context" is the same setting whether it is
 * written as 0 or as the length of the context). */
export function changedFrom(settings: Settings, usual: Settings, longest: number): boolean {
  const whole = (count: number) => (count <= 0 || count >= longest ? 0 : count);
  return (Object.keys({ ...usual, ...settings }) as (keyof Settings)[])
    .some((key) => (key === "steps" ? whole(settings.steps) !== whole(usual.steps) : settings[key] !== usual[key]));
}
