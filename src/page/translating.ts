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

/** T670: what separates a glossary's term from its translation on a line, as tanakh/plamo-translate-desktop's glossary
 * reads it: "=", a tab, "=>" and "→" (where two are on a line, the first; "=>" before "=" where they begin at the
 * same place) */
export const GLOSSARY_SEPARATORS = ["\t", "=>", "→", "="] as const;
/** T670: a glossary's text read (the field's, kept as it was typed): a line "term = its translation" a pair, the two
 * sides without the whitespace around them; an empty line, or one that begins with "#" (a comment), is none; a line
 * with no separator, or with nothing on a side of it, is not read (unread: its number, the first line 1), and the
 * translation goes on without it. The one place a glossary is read: Python takes the pairs as they are
 * (engine/document.py's page_translation(), engine/pfml.py's pfml_pieces()) */
export function glossaryOf(text: string) {
  const pairs: [string, string][] = [], unread: number[] = [];
  text.split(/\r\n|\r|\n/).forEach((line, index) => {
    const said = line.trim();
    if (!said || said.startsWith("#")) return;
    let at = -1, width = 0;
    for (const separator of GLOSSARY_SEPARATORS) {
      const found = said.indexOf(separator);
      if (found >= 0 && (at < 0 || found < at)) [at, width] = [found, separator.length];
    }
    const term = at < 0 ? "" : said.slice(0, at).trim(), translation = at < 0 ? "" : said.slice(at + width).trim();
    if (term && translation) pairs.push([term, translation]);
    else unread.push(index + 1);
  });
  return { pairs, unread };
}
/** T670: the words under the glossary's field for its lines not read (none: "") */
export function unreadSaid(unread: readonly number[]) {
  if (!unread.length) return "";
  const lines = unread.length === 1 ? `Line ${unread[0]} is` : `Lines ${unread.slice(0, -1).join(", ")} and ${unread.at(-1)} are`;
  return `${lines} not read: write a term, then = and its translation.`;
}
/** T670: the words of the fold that holds the context and the glossary, with what of them a translation is asked with */
export function moreSaid(context: string, terms: number) {
  const said = [context.trim() ? "context" : "", terms ? `${terms} ${terms === 1 ? "term" : "terms"}` : ""].filter(Boolean);
  return `Context and glossary${said.length ? ` (${said.join(", ")})` : ""}`;
}
/** T670: what a translation is asked with besides its target and form: the context where it has more than whitespace
 * (as typed; not kept), and the glossary's pairs where it has any (none of either: no key) */
export function extrasAsked(context: string, glossary: string): Pick<Translate, "context" | "glossary"> {
  const { pairs } = glossaryOf(glossary);
  return { ...(context.trim() ? { context } : {}), ...(pairs.length ? { glossary: pairs } : {}) };
}
