// src/page/list.ts (T.355): the list of models in the select, and what this browser kept of them.
import { GROUPS, MODELS } from "../models.js";
import { $, select } from "./dom.ts";
import { fromUrl, weighed } from "./address.ts";
import { remember } from "./remembered.ts";
import { model } from "./model.ts";
import { forget, keptModels, outdated, serves } from "../runtime/kept.ts";

// What the worker kept of its conversions (src/runtime/kept.ts: the origin private file system, or the Cache API): the
// select says which models need no download any more, and About lists them with a way to delete them.
// (T.367.3: linked into the page's own script. A store that cannot be listed is one that kept nothing.)
// (T.380.2: one that cannot be listed says so on the console, and shows as before)
const listed = () => keptModels().catch((error): Awaited<ReturnType<typeof keptModels>> => {
  console.warn(`kept: what this browser kept could not be listed (${String(error)}): shown as nothing kept`);
  return [];
});
// T.490: what was kept, deleted. One that cannot be deleted says so on the console, and the list is drawn again as it is
// (an unhandled rejection left it as it was before the press)
const forgotten = (one: Parameters<typeof forget>[0]) => forget(one).catch((error) => {
  console.warn(`kept: ${one.name} could not be deleted (${String(error)}): it stays in the list`);
});
export async function showKept() {
  let models = await listed();
  // T.116: what an older converter kept is never used again, so it goes (the worker deletes it too when it looks)
  for (const one of models.filter((one) => outdated(one))) await forgotten(one);
  models = models.filter((one) => !outdated(one));
  for (const option of select.querySelectorAll("option")) {
    const entry = MODELS.find(({ id }) => id === option.value) as any;
    if (entry?.hf) {
      // kept as it would be loaded now: under its bits (?bits=, or either the worker may choose), by this converter
      const found = models.some((one) => serves(one, weighed(entry)));
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
      await forgotten(one);
      if (model.id === one.manifest.id) {
        remember(model, false);
      }
      showKept();
    };
    item.append(`${one.manifest.name} (${(one.manifest.bytes / 1e6).toFixed(0)} MB) `, remove);
    $("kept-list").append(item);
  }
}
// T.116: a remembered model that would be fetched anew starts unasked only because it was kept (remembered()): it
// must still be kept as it would be loaded now (its bits, this converter), or the default starts instead
export async function stillKept(entry: any) {
  if ((entry.download ?? 0) <= 500e6) return true;
  const models = await listed();
  return models.some((one) => serves(one, weighed(entry)));
}

// three groups: what the site was built with, the unquantized originals, and what comes from Hugging Face
for (const [group, label] of Object.entries(GROUPS)) {
  const options = document.createElement("optgroup");
  options.label = label;
  for (const { id, name, note } of MODELS.filter((entry) => ((entry as any).group ?? "site") === group)) {
    options.append(new Option(`${name} — ${note}`, id));
  }
  if (group === "hf") {
    options.append(new Option("Other repository… — any Llama, Mistral, Qwen2, Qwen3, GPT-2 or GPT-NeoX", "hf-other"));  // T.88
  }
  select.append(options);
}
showKept();
if (fromUrl) {
  select.add(new Option(`${fromUrl.name} — ${fromUrl.note}`, "local"));
}
