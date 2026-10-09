// src/page/settings.ts (T355): the generation settings and their sheet, the engine's switches and bits in it, and
// what the page shows of a model as it is chosen.
import { $, fit, prompt, select } from "./dom.ts";
import { parameters } from "./address.ts";
import { page } from "./state.ts";

// The generation settings. They start as the model's own and go back to them with every change of model, except
// the seed: "tap the seed of an answer, change the model, send" is how int8 is compared with the original.
// steps: 0 stands for as many tokens as the model's context holds, which is the default of every model
// T274: top_k, min_p and presence_penalty have no field: a model's own (its card's) or the link's
type Settings = { steps: number; temperature: number; topp: number; repetition_penalty: number; seed?: number;
                  top_k?: number; min_p?: number; presence_penalty?: number };
const defaults = (current: typeof page.model): Settings => ({ topp: 0.9, repetition_penalty: 1.0, ...(current.generation as any) });
// The URL may set any of them (T43), so that a link reproduces an answer: ?temperature=0.7&seed=1234&steps=128
// (&topp= and &repetition_penalty= too, and &top_k=, &min_p= and &presence_penalty=). A value that is not a number, or outside what the field allows, is
// ignored. They survive a change of model, like the seed: the link said what to run with.
const LIMITS: Record<string, [number, number]> = { temperature: [0, 1.5], topp: [0.05, 1], steps: [0, 1 << 20],
                                                   repetition_penalty: [1, 2], seed: [0, 0xffffffff],
                                                   top_k: [0, 1 << 20], min_p: [0, 1], presence_penalty: [0, 2] };
const fromLink: Partial<Settings> = {};
for (const [name, [low, high]] of Object.entries(LIMITS)) {
  const given = parameters.get(name);
  const value = given === null ? NaN : Number(given);
  if (given !== null && Number.isFinite(value) && value >= low && value <= high) {
    (fromLink as any)[name] = ["steps", "seed", "top_k"].includes(name) ? Math.round(value) : value;
  }
}
const defaultsWithLink = (current: typeof page.model): Settings => ({ ...defaults(current), ...fromLink });
export let settings = defaultsWithLink(page.model);
const field = (id: string) => $(id) as HTMLInputElement;
export function showSettings() {
  field("steps").max = String(page.longest);
  field("temperature").value = String(settings.temperature);
  const steps = settings.steps > 0 ? Math.min(settings.steps, page.longest) : page.longest;
  field("steps").value = String(steps);
  field("topp").value = String(settings.topp);
  field("penalty").value = String(settings.repetition_penalty);
  if (document.activeElement !== field("seed")) {
    field("seed").value = settings.seed === undefined ? "" : String(settings.seed);
  }
  $("temperature-value").textContent = settings.temperature ? settings.temperature.toFixed(2) : "greedy";
  $("steps-value").textContent = String(steps);
  $("topp-value").textContent = settings.topp.toFixed(2);
  $("penalty-value").textContent = settings.repetition_penalty.toFixed(2);
  const usual = defaults(page.model);
  // "all of the context" is the same setting whether it is written as 0 or as the length of the context
  const whole = (count: number) => (count <= 0 || count >= page.longest ? 0 : count);
  const changed = (Object.keys({ ...usual, ...settings }) as (keyof Settings)[])
    .some((key) => (key === "steps" ? whole(settings.steps) !== whole(usual.steps) : settings[key] !== usual[key]));
  $("settings-open").classList.toggle("changed", changed);
}
export function fixSeed(seed: number) {
  settings.seed = seed;
  showSettings();
}
for (const [id, key] of [["temperature", "temperature"], ["steps", "steps"], ["topp", "topp"], ["penalty", "repetition_penalty"]] as const) {
  field(id).oninput = () => {
    const value = Number(field(id).value);
    settings[key] = key === "steps" && value >= page.longest ? 0 : value;
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
  settings = defaults(page.model);
  showSettings();
};
// T75: the engine's switches. They are read from the URL (?without=, and ?kernel=off as it always was) and written
// back to it: the page reloads with the model it has, and the worker starts from the new URL.
const without = new Set((parameters.get("without") ?? "").split(",").map((name) => name.trim()).filter(Boolean));
if (parameters.get("kernel") === "off") {
  without.add("kernels");
}
const switches = [...document.querySelectorAll<HTMLInputElement>("#engine input[data-switch]")];
for (const box of switches) {
  const name = box.dataset.switch!;
  box.checked = !without.has(name);
  box.disabled = name !== "kernels" && without.has("kernels");  // the other three are ways of using the kernels
  box.onchange = () => {
    box.checked ? without.delete(name) : without.add(name);
    const url = new URL(location.href);
    url.searchParams.delete("kernel");
    // the panel's four first, in their order, then whatever else the URL had (kv16, for measuring)
    const names = [...switches.map((each) => each.dataset.switch!).filter((each) => without.has(each)),
                   ...[...without].filter((each) => !switches.some((box) => box.dataset.switch === each))];
    names.length ? url.searchParams.set("without", names.join(",")) : url.searchParams.delete("without");
    url.searchParams.delete("bench");  // T119: the reload is to run with the switches, not to measure again
    if (page.model.id !== "local") {
      url.searchParams.set("model", page.model.id);
    }
    location.assign(url);
  };
}
// T98: the bits of a converted model, from ?bits= (weighed() reads it) and back to it, the same way
const asked = ["6", "8"].includes(parameters.get("bits") ?? "") ? parameters.get("bits")! : "";
for (const choice of document.querySelectorAll<HTMLInputElement>('#engine input[name="bits"]')) {
  choice.checked = choice.value === asked;
  choice.onchange = () => {
    const url = new URL(location.href);
    choice.value ? url.searchParams.set("bits", choice.value) : url.searchParams.delete("bits");
    url.searchParams.delete("bench");
    if (page.model.id !== "local") {
      url.searchParams.set("model", page.model.id);
    }
    location.assign(url);
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

export function show(current: typeof page.model) {
  settings = { ...defaultsWithLink(current), ...(settings.seed === undefined ? {} : { seed: settings.seed }) };
  showSettings();
  select.value = current.id;
  prompt.value = current.prompt;
  fit();
  prompt.placeholder = current.placeholder;
}
show(page.model);
