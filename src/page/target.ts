// src/page/target.ts (T669): the two fields above the text field of a model that translates with PLaMo 3 Translate's
// request (an entry's pfml): the language to translate into and, where that may be Japanese, the form of the Japanese
// (translating.ts's tables and what is asked of them). What was chosen is kept for the next visit (remembered.ts's
// recall() and keep(), under "translate"); the address's ?target= and ?form= (parameters.ts) come first on this visit.
import { $ } from "./dom.ts";
import { parameters } from "./address.ts";
import { keep, recall } from "./remembered.ts";
import { FORMS, TARGETS, formable, translateAsked } from "./translating.ts";
import type { Model } from "./model.ts";
import type { Translate } from "../runtime/contracts/messages.ts";

const fields = $("translate"), target = $("target") as HTMLSelectElement, form = $("form") as HTMLSelectElement;
for (const { code, said } of TARGETS) target.add(new Option(said, code));
for (const { code, said } of FORMS) form.add(new Option(said, code));
// (none kept, no storage here, or something else wrote there: the automatic target)
const kept: Partial<Translate> = recall("translate", (text) => JSON.parse(text ?? "{}") ?? {}, {});
const first = translateAsked(parameters.get("target") ?? kept.target, parameters.get("form") ?? kept.form);
target.value = first.target;
form.value = first.form ?? "";

/** What a translation is asked with now: the fields' */
export const targetAsked = () => translateAsked(target.value, form.value);
function shown() {
  form.hidden = !formable(target.value);
}
target.onchange = form.onchange = () => {
  shown();
  keep("translate", JSON.stringify(targetAsked()));
};
/** The fields where the model translates with PLaMo 3 Translate's request, and none where it does not */
export function showTarget(current: Model) {
  fields.hidden = !current.pfml;
  shown();
}
