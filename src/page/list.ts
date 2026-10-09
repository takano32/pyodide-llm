// src/page/list.ts (T355): the list of models in the select, and what this browser kept of them.
import { GROUPS, MODELS } from "../models.js";
import { $, select } from "./dom.ts";
import { fromUrl, weighed } from "./address.ts";
import { remember } from "./remembered.ts";
import { page } from "./state.ts";

declare const __BUILD__: string;

// What the worker kept of its conversions (public/kept.js: the origin private file system, or the Cache API): the
// select says which models need no download any more, and About lists them with a way to delete them.
const keptModule = import(/* @vite-ignore */ `${import.meta.env.BASE_URL}kept.js?v=${__BUILD__}`).catch(() => undefined);
export async function showKept() {
  const kept = await keptModule;
  let models = (await kept?.keptModels().catch(() => [])) ?? [];
  // T116: what an older converter kept is never used again, so it goes (the worker deletes it too when it looks)
  for (const one of models.filter((one: any) => kept.outdated(one))) await kept.forget(one);
  models = models.filter((one: any) => !kept.outdated(one));
  for (const option of select.querySelectorAll("option")) {
    const entry = MODELS.find(({ id }) => id === option.value) as any;
    if (entry?.hf) {
      // kept as it would be loaded now: under its bits (?bits=, or either the worker may choose), by this converter
      const found = models.some((one: any) => kept.serves(one, weighed(entry)));
      option.textContent = `${entry.name} — ${entry.note}${found ? " · kept in this browser" : ""}`;
    }
  }
  $("kept").hidden = models.length === 0;
  $("kept-list").textContent = "";
  for (const one of models) {
    const item = document.createElement("li");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "delete";
    remove.onclick = async () => {
      await kept.forget(one);
      if (page.model.id === one.manifest.id) {
        remember(page.model, false);
      }
      showKept();
    };
    item.append(`${one.manifest.name} (${(one.manifest.bytes / 1e6).toFixed(0)} MB) `, remove);
    $("kept-list").append(item);
  }
}
// T116: a remembered model that would be fetched anew starts unasked only because it was kept (remembered()): it
// must still be kept as it would be loaded now (its bits, this converter), or the first of the list starts instead
export async function stillKept(entry: any) {
  if ((entry.download ?? 0) <= 500e6) return true;
  const kept = await keptModule;
  const models = (await kept?.keptModels().catch(() => [])) ?? [];
  return models.some((one: any) => kept.serves(one, weighed(entry)));
}

// three groups: what the site was built with, the unquantized originals, and what comes from Hugging Face
for (const [group, label] of Object.entries(GROUPS)) {
  const options = document.createElement("optgroup");
  options.label = label;
  for (const { id, name, note } of MODELS.filter((entry) => ((entry as any).group ?? "site") === group)) {
    options.append(new Option(`${name} — ${note}`, id));
  }
  if (group === "hf") {
    options.append(new Option("Other repository… — any Llama, Mistral, Qwen2, Qwen3, GPT-2 or GPT-NeoX", "hf-other"));  // T88
  }
  select.append(options);
}
showKept();
if (fromUrl) {
  select.add(new Option(`${fromUrl.name} — ${fromUrl.note}`, "local"));
}
