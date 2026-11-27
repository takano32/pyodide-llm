// src/page/target.ts (T669): the two fields above the text field of a model that translates with PLaMo 3 Translate's
// request (an entry's pfml): the language to translate into and, where that may be Japanese, the form of the Japanese
// (translating.ts's tables and what is asked of them). What was chosen is kept for the next visit (remembered.ts's
// recall() and keep(), under "translate"); the address's ?target= and ?form= (parameters.ts) come first on this visit.
// T670: and, folded below them, a context and a glossary (translating.ts's glossaryOf() reads it). The glossary is kept
// as it was typed (under "glossary"), the context is not (it is about the text of now, as tanakh's); neither is in the
// address (long texts). What they say is asked with the next translation: one going on is not translated again.
import { $ } from "./dom.ts";
import { parameters } from "./address.ts";
import { keep, recall } from "./remembered.ts";
import { FORMS, TARGETS, extrasAsked, formable, glossaryOf, moreSaid, translateAsked, unreadSaid } from "./translating.ts";
import type { Model } from "./model.ts";
import type { Translate } from "../runtime/contracts/messages.ts";

const fields = $("translate"), target = $("target") as HTMLSelectElement, form = $("form") as HTMLSelectElement;
const more = $("translate-more"), context = $("context") as HTMLTextAreaElement, glossary = $("glossary") as HTMLTextAreaElement;
const unread = $("glossary-unread");
for (const { code, said } of TARGETS) target.add(new Option(said, code));
for (const { code, said } of FORMS) form.add(new Option(said, code));
// (none kept, no storage here, or something else wrote there: the automatic target)
const kept: Partial<Translate> = recall("translate", (text) => JSON.parse(text ?? "{}") ?? {}, {});
const first = translateAsked(parameters.get("target") ?? kept.target, parameters.get("form") ?? kept.form);
target.value = first.target;
form.value = first.form ?? "";
glossary.value = recall("glossary", (text) => text ?? "", "");

/** What a translation is asked with now: the fields' target and form (what is kept of them) */
export const targetAsked = () => translateAsked(target.value, form.value);
/** T670: and the context and the glossary besides: what a translation is asked with */
export const translationAsked = (): Translate => ({ ...targetAsked(), ...extrasAsked(context.value, glossary.value) });
function shown() {
  form.hidden = !formable(target.value);
}
function read() {
  const { pairs, unread: lines } = glossaryOf(glossary.value);
  unread.textContent = unreadSaid(lines);
  unread.hidden = !lines.length;
  more.textContent = moreSaid(context.value, pairs.length);
}
target.onchange = form.onchange = () => {
  shown();
  keep("translate", JSON.stringify(targetAsked()));
};
context.oninput = read;
glossary.oninput = () => {
  read();
  keep("glossary", glossary.value || null);
};
read();
/** The fields where the model translates with PLaMo 3 Translate's request, and none where it does not */
export function showTarget(current: Model) {
  fields.hidden = !current.pfml;
  shown();
}
