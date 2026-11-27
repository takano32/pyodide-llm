// src/page/translating.ts (T669): what the page asks a model that translates with PLaMo 3 Translate's request (an entry's
// pfml: engine/document.py's page_translation()), and what it does with what the translation tells as it goes. The
// languages are Python's (engine/pfml.py's LANGUAGES, easy Japanese last, the forms of a translation into Japanese),
// which python_facts.py writes for the page before any Python runs. Nothing of a page is read here (no element, no
// address, no storage): Node imports it as it is (tests/page-parts-check.ts); target.ts draws the two fields.
import { PYTHON_TRANSLATION } from "../runtime/built/facts/translation.ts";
import type { Translate } from "../runtime/contracts/messages.ts";

const { languages, forms, formed, automatic } = PYTHON_TRANSLATION;
type Form = (typeof forms)[number];

/** The languages the field offers, in its order, with their words (the owner, 2026-11-26: the English name and the
 * language's own, "Japanese — 日本語"; one name where the two are the same): first the page's default, Japanese into
 * English and any other into Japanese (engine/document.py's AUTOMATIC: the model tells the text's language) */
export const TARGETS: readonly { code: string; said: string }[] = [
  { code: automatic, said: "Japanese ⇄ English (automatic)" },
  ...languages.map(([code, english, own]) => ({ code, said: english === own ? english : `${english} — ${own}` })),
];
/** The forms of a translation into Japanese the field offers: none (the model's own), polite, plain */
export const FORMS: readonly { code: Form | ""; said: string }[] = [
  { code: "", said: "Any form" },
  { code: "polite", said: "です・ます (polite)" },
  { code: "plain", said: "だ・である (plain)" },
];
/** Whether a target may have a form: Japanese, or the automatic one (Japanese for a text that is not) */
export const formable = (target: string) => target === formed || target === automatic;
/** What is asked for the target and the form said (the field's, the address's or the ones kept): a target the field
 * has, else the automatic one, and a form only where the target may have one and it is one of FORMS */
export function translateAsked(target: string | null | undefined, form: string | null | undefined): Translate {
  const code = TARGETS.some((one) => one.code === target) ? target! : automatic;
  return formable(code) && (forms as readonly string[]).includes(form ?? "") ? { target: code, form: form! } : { target: code };
}
/** The text without its last `characters` characters, counted as Python counts them (code points: a Withdrawn's; a
 * character past U+FFFF is two units of a JavaScript string, a surrogate pair, taken back whole) */
export function withoutLast(text: string, characters: number) {
  let at = text.length;
  for (let left = characters; left > 0 && at > 0; left -= 1) {
    const last = text.charCodeAt(at - 1);
    at -= at >= 2 && last >= 0xdc00 && last <= 0xdfff && (text.charCodeAt(at - 2) & 0xfc00) === 0xd800 ? 2 : 1;
  }
  return text.slice(0, at);
}
