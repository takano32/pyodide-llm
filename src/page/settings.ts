// src/page/settings.ts (T355): the generation settings and their sheet, the engine's switches and bits in it, and
// what the page shows of a model as it is chosen.
import { $, fit, prompt, select } from "./dom.ts";
import { parameters } from "./address.ts";
import { addressWith } from "./parameters.ts";
import { type Model, longest, model } from "./model.ts";
import { type Settings, FIELDS, changedFrom, defaults, linkSettings } from "./fields.ts";

// The generation settings. They start as the model's own and go back to them with every change of model, except
// the seed: "tap the seed of an answer, change the model, send" is how int8 is compared with the original.
// What the link sets (fields.ts) survives a change of model, like the seed: the link said what to run with.
const fromLink = linkSettings(parameters);
const defaultsWithLink = (current: Model): Settings => ({ ...defaults(current), ...fromLink });
export let settings = defaultsWithLink(model);
const field = (id: string) => $(id) as HTMLInputElement;
export function showSettings() {
  field("steps").max = String(longest);
  for (const { id, key, said, shown } of FIELDS) {
    const value = shown ? shown(settings[key], longest) : settings[key];
    field(id).value = String(value);
    $(`${id}-value`).textContent = said(value);
  }
  if (document.activeElement !== field("seed")) {
    field("seed").value = settings.seed === undefined ? "" : String(settings.seed);
  }
  $("settings-open").classList.toggle("changed", changedFrom(settings, defaults(model), longest));
}
export function fixSeed(seed: number) {
  settings.seed = seed;
  showSettings();
}
for (const { id, key, taken } of FIELDS) {
  field(id).oninput = () => {
    const value = Number(field(id).value);
    settings[key] = taken ? taken(value, longest) : value;
    showSettings();
  };
}
field("seed").oninput = () => {
  // what the engine's generator takes: a whole number of 32 bits. Anything else means a random one.
  const text = field("seed").value.trim();
  settings.seed = /^\d{1,10}$/.test(text) && Number(text) <= 0xffffffff ? Number(text) : undefined;
  showSettings();
};
field("seed").onblur = showSettings;
$("seed-clear").onclick = () => {
  settings.seed = undefined;
  showSettings();
};
$("settings-reset").onclick = () => {
  settings = defaults(model);
  showSettings();
};
// T75: the engine's switches. They are read from the URL (?without=, and ?kernel=off as it always was) and written
// back to it: the page reloads with the model it has, and the worker starts from the new URL.
const without = new Set((parameters.get("without") ?? "").split(",").map((name) => name.trim()).filter(Boolean));
if (parameters.get("kernel") === "off") {
  without.add("kernels");
}
// (the model of the list the page has, which the reload opens: the address of one of no list says it already)
const namedModel = () => (model.id !== "local" ? { model: model.id } : {});
const switches = [...document.querySelectorAll<HTMLInputElement>("#engine input[data-switch]")];
for (const box of switches) {
  const name = box.dataset.switch!;
  box.checked = !without.has(name);
  box.disabled = name !== "kernels" && without.has("kernels");  // the other three are ways of using the kernels
  box.onchange = () => {
    box.checked ? without.delete(name) : without.add(name);
    // the panel's four first, in their order, then whatever else the URL had (kv16, for measuring)
    const names = [...switches.map((each) => each.dataset.switch!).filter((each) => without.has(each)),
                   ...[...without].filter((each) => !switches.some((box) => box.dataset.switch === each))];
    // (T119: the reload is to run with the switches, not to measure again)
    location.assign(addressWith(location.href, { kernel: null, without: names.length ? names.join(",") : null, bench: null, ...namedModel() }));
  };
}
// T98: the bits of a converted model, from ?bits= (weighed() reads it) and back to it, the same way
const asked = ["6", "8"].includes(parameters.get("bits") ?? "") ? parameters.get("bits")! : "";
for (const choice of document.querySelectorAll<HTMLInputElement>('#engine input[name="bits"]')) {
  choice.checked = choice.value === asked;
  choice.onchange = () => {
    location.assign(addressWith(location.href, { bits: choice.value || null, bench: null, ...namedModel() }));
  };
}
const panel = $("settings");
panel.addEventListener("beforetoggle", (event) => {
  if ((event as ToggleEvent).newState !== "open") {
    return;
  }
  // a phone would keep its keyboard up, over the sheet
  prompt.blur();
  const button = $("settings-open").getBoundingClientRect();
  panel.style.left = `${button.left}px`;
  panel.style.bottom = `${innerHeight - button.top + 10}px`;
  // T119: no taller than the room above the button: with Engine and More both open, a phone lost its top
  panel.style.maxHeight = `${button.top - 26}px`;
});

export function show(current: Model) {
  settings = { ...defaultsWithLink(current), ...(settings.seed === undefined ? {} : { seed: settings.seed }) };
  showSettings();
  select.value = current.id;
  prompt.value = current.prompt;
  fit();
  prompt.placeholder = current.placeholder;
}
show(model);
