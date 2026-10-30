// src/page/choose.ts (T.355): choosing a model: the select, the files of the visitor's own disk, the sheet that asks
// for a repository of Hugging Face, the buttons of the intro. Each ends in the select's change, which tells the worker.
import { DEFAULT_MODEL, MODELS, memoryWarning } from "../models.js";
import { $, files, progress, run, select, statusText } from "./dom.ts";
import { FILE, REPOSITORY, REVISION, type Unlisted, fromUrl, hfEntry, unlisted, weighed } from "./address.ts";
import { looked, unnamed } from "./repository.ts";
import { gpuFor, threadsFor } from "./remembered.ts";
import { type Model, chose, loads, model, opensWith } from "./model.ts";
import { stillKept } from "./list.ts";
import { show } from "./settings.ts";
import { message } from "./draw.ts";
import { root, worker } from "./receiver.ts";
import { openHuggingFace, openLegacy } from "./folder.ts";

// A model of the visitor's own disk, from the folder button or dropped on the page (folder.ts makes it: what Hugging
// Face publishes, or a checkpoint of llama2.c's format and its tokenizer.bin).
let local: Model | undefined = fromUrl;
async function open(chosen: File[]) {
  if (chosen.some(({ name }) => name.toLowerCase().endsWith(".safetensors"))) {
    try {
      local = (await openHuggingFace(chosen)) as any;
    } catch (error) {
      message("model error", String((error as Error).message ?? error));
      return;
    }
    return showLocal();
  }
  try {
    const opened = await openLegacy(chosen, confirm);
    if (!opened) {
      return;
    }
    local = opened as any;
  } catch (error) {
    message("model error", String((error as Error).message ?? error));
    return;
  }
  showLocal();
}
function showLocal() {
  // (T.470: not while an answer is written, which keeps the select from a choice: the repository's sheet stays open
  // beside it, and its address may come back after that began)
  if (select.disabled) return;
  // the select shows what is running: one entry for the local model, replaced by the next one
  select.querySelector('option[value="local"]')?.remove();
  select.add(new Option(`${local!.name} — ${local!.note}`, "local"));
  select.value = "local";
  select.onchange!(new Event("change"));
}
files.onchange = () => {
  const chosen = [...(files.files ?? [])];
  // so that the same files can be chosen again
  files.value = "";
  if (chosen.length) {
    open(chosen);
  }
};
document.ondragover = (event) => event.preventDefault();
document.ondrop = (event) => {
  event.preventDefault();
  if (!files.disabled && event.dataTransfer?.files.length) {
    open([...event.dataTransfer.files]);
  }
};

for (const button of document.querySelectorAll<HTMLButtonElement>(".intro .choose")) {
  button.onclick = () => {
    if (!select.disabled && select.value !== button.dataset.model) {
      select.value = button.dataset.model!;
      select.onchange!(new Event("change"));
    }
  };
}

// The URL names the model the page has, however it was chosen (the list, the sheet, the folder), so that a reload
// (every switch of the panel reloads) or a shared link opens that model and no other (the review of T.75 and T.88:
// after ?hf=, choosing a model of the list and then a switch reloaded the Hugging Face model). A file of the disk
// has no address: its URL names nothing, and a reload opens the model remembered before it.
function nameInUrl(current: Model) {
  const url = new URL(location.href);
  for (const name of ["model", "hf", "revision", "file", "checkpoint", "tokenizer", "template", "prompt"]) {
    url.searchParams.delete(name);
  }
  const { hf, url: address, template } = current as any;
  if (current.id !== "local") {
    url.searchParams.set("model", current.id);
  } else if (hf?.repo || address) {
    if (hf?.repo) {
      url.searchParams.set("hf", hf.repo);
      if (hf.revision !== "main") {
        url.searchParams.set("revision", hf.revision);
      }
      // T.505.3: and the file, where it is not the model.safetensors a repository is opened by: the one the visitor
      // named, or the GGUF that was the best there when the repository was looked at
      if (!unnamed(unlisted(current)!)) {
        url.searchParams.set("file", hf.weights);
      }
    } else {
      url.searchParams.set("checkpoint", address.checkpoint);
      url.searchParams.set("tokenizer", address.tokenizer);
    }
    for (const [name, value] of [["template", template], ["prompt", current.prompt]]) {
      if (value) {
        url.searchParams.set(name, value);
      }
    }
  }
  history.replaceState(null, "", url);
}

// T.88: the last entry of the Hugging Face group asks which repository, in a sheet like the settings'
const repositorySheet = $("repository");
repositorySheet.addEventListener("beforetoggle", (event) => {
  if ((event as ToggleEvent).newState !== "open") {
    return;
  }
  const anchor = select.getBoundingClientRect();
  repositorySheet.style.left = `${Math.max(16, Math.min(anchor.left, innerWidth - 336))}px`;
  repositorySheet.style.top = `${anchor.bottom + 8}px`;
  repositorySheet.style.maxHeight = `${innerHeight - anchor.bottom - 24}px`;
});
repositorySheet.addEventListener("toggle", (event) => {
  if ((event as ToggleEvent).newState === "open") {
    ($("repository-name") as HTMLInputElement).focus();
  }
});
// T.505.3: the third field names the file that is the model, a GGUF; empty, the repository is looked at for one. What
// the field takes is what an address takes (the browser says so of anything else, in its own words)
const fileField = $("repository-file") as HTMLInputElement;
fileField.pattern = FILE.source;
($("repository-form") as HTMLFormElement).onsubmit = async (event) => {
  event.preventDefault();
  const field = $("repository-name") as HTMLInputElement, revisionField = $("repository-revision") as HTMLInputElement;
  const file = fileField.value.trim() || undefined;
  // the address of the model's page is what people copy: owner/name is what is needed of it
  const repository = field.value.trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/(tree|blob)\/.*$/, "").replace(/\/+$/, "");
  const revision = revisionField.value.trim() || "main";
  // T.119: the message goes on the field that is wrong, and goes away as that field is edited (on the name alone,
  // a wrong revision put right left the name marked, and the form would not open)
  const wrong: [HTMLInputElement, string] | undefined = !REPOSITORY.test(repository) ? [field, "owner/name, as huggingface.co spells it"]
    : !REVISION.test(revision) ? [revisionField, "a branch, a tag or a commit"] : undefined;
  if (wrong) {
    const [at, words] = wrong;
    at.setCustomValidity(words);
    at.reportValidity();
    at.oninput = () => at.setCustomValidity("");
    return;
  }
  // (T.470: as the intro's buttons, nothing while an answer is written: the sheet stays as it is, to be sent after it)
  if (select.disabled) return;
  repositorySheet.hidePopover();
  let entry: Unlisted;
  try {
    entry = await looked(hfEntry(repository, revision, { file }));
  } catch (error) {
    message("model error", String((error as Error).message ?? error));  // (a repository with nothing this page opens)
    return;
  }
  // a large GGUF is asked for before it is fetched, as a large model of the list is: not one this browser kept
  if ((entry.download ?? 0) > 500e6 && await stillKept(entry)) {
    entry = { ...entry, download: 0 };
  }
  local = entry;
  showLocal();  // the change of model writes the URL (nameInUrl), so that the link can be shared
};

/** T.505.3: the repository the URL names (?hf=) with no file of it is looked at before its load: one without
 * safetensors is opened by the best GGUF it has, and the URL says which from then on. Whether the page may go on to
 * load its model: not where the repository has nothing this page opens, which is said. (A model chosen while this
 * looked has started its own load, and wins.) */
export async function urlLooked() {
  const named = unlisted(fromUrl);
  if (!named || model !== named || !unnamed(named)) return true;
  let entry: Unlisted;
  try {
    entry = await looked(named);
  } catch (error) {
    if (loads === 1) {
      message("model error", String((error as Error).message ?? error));
      statusText.textContent = `Could not load ${named.name}`;
    }
    return false;
  }
  if (entry === named || !opensWith(entry)) return true;
  local = entry;
  select.querySelector('option[value="local"]')!.textContent = `${entry.name} — ${entry.note}`;
  nameInUrl(entry);
  show(entry);
  return true;
}

select.onchange = () => {
  if (select.value === "hf-other") {
    select.value = model.id === "local" && !select.querySelector('option[value="local"]') ? DEFAULT_MODEL.id : model.id;
    repositorySheet.showPopover();
    return;
  }
  const chosen = select.value === "local" ? local! : MODELS.find(({ id }) => id === select.value)!;
  // a large download is asked for first: what is fetched, from whom, and that the conversion happens here
  const download = select.selectedOptions[0]?.textContent?.includes("kept in this browser") ? 0 : ((chosen as any).download ?? 0);
  // T.90: and one that probably does not fit in a device that says how much memory it has (Chromium only)
  const asks = [download > 500e6 && `${chosen.name}: ${(download / 1e9).toFixed(1)} GB will be fetched from huggingface.co and converted in this browser.`,
                memoryWarning(weighed(chosen), (navigator as any).deviceMemory)].filter(Boolean);
  if (asks.length && !confirm(`${asks.join(" ")} Go on?`)) {
    select.value = model.id;
    return;
  }
  const load = chose(chosen);
  nameInUrl(model);
  show(model);
  run.disabled = true;
  progress.hidden = true;
  // the timings of the previous load
  $("load").classList.remove("measured");
  ($("load") as HTMLDetailsElement).open = false;
  // search: this may reach the worker before init does, while the page looks at what is kept (T.116)
  worker.postMessage({ type: "load", root, search: location.search, model: weighed(model), load, threads: threadsFor(model), gpu: gpuFor(model) });
};
