// src/page/choose.ts (T355): choosing a model: the select, the files of the visitor's own disk, the sheet that asks
// for a repository of Hugging Face, the buttons of the intro. Each sets the select and ends in chooseSelected(), which
// asks what is to be asked and tells the worker (T643.10: what is decided is choosing.ts's, which Node imports).
import { asksBefore } from "../models.ts";
import { $, files, progress, run, select, statusText } from "./dom.ts";
import { FILE, fromUrl, hfEntry, weighed } from "./address.ts";
import { looked, unlisted, unnamed } from "./repository.ts";
import { addressOf } from "./parameters.ts";
import { gpuFor, threadsFor } from "./remembered.ts";
import { type Model, type Unlisted, chose, loads, model, opensWith } from "./model.ts";
import { keptEntries, stillKept } from "./list.ts";
import { show } from "./settings.ts";
import { message } from "./draw.ts";
import { root, worker } from "./receiver.ts";
import { openHuggingFace, openLegacy } from "./folder.ts";
import { OTHER, askedFirst, chosenOf, sheetSays, shownForOther } from "./choosing.ts";
import type { ToWorker } from "../runtime/contracts/messages.ts";

// A model of the visitor's own disk, from the folder button or dropped on the page (folder.ts makes it: what Hugging
// Face publishes, or a checkpoint of llama2.c's format and its tokenizer.bin).
let local: Model | undefined = fromUrl;
async function open(chosen: File[]) {
  if (chosen.some(({ name }) => name.toLowerCase().endsWith(".safetensors"))) {
    try {
      local = await openHuggingFace(chosen);
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
    local = opened;
  } catch (error) {
    message("model error", String((error as Error).message ?? error));
    return;
  }
  showLocal();
}
function showLocal() {
  // (T470: not while an answer is written, which keeps the select from a choice: the repository's sheet stays open
  // beside it, and its address may come back after that began)
  if (select.disabled) return;
  // the select shows what is running: one entry for the local model, replaced by the next one
  select.querySelector('option[value="local"]')?.remove();
  select.add(new Option(`${local!.name} — ${local!.note}`, "local"));
  select.value = "local";
  chooseSelected();
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
      chooseSelected();
    }
  };
}

// The URL names the model the page has, however it was chosen (parameters.ts's addressOf(): a reload or a shared link
// opens that model and no other)
function nameInUrl(current: Model) {
  history.replaceState(null, "", addressOf(current, location.href));
}

// T88: the last entry of the Hugging Face group asks which repository, in a sheet like the settings'
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
// T505.3: the third field names the file that is the model, a GGUF; empty, the repository is looked at for one. What
// the field takes is what an address takes (the browser says so of anything else, in its own words)
const fileField = $("repository-file") as HTMLInputElement;
fileField.pattern = FILE.source;
($("repository-form") as HTMLFormElement).onsubmit = async (event) => {
  event.preventDefault();
  const field = $("repository-name") as HTMLInputElement, revisionField = $("repository-revision") as HTMLInputElement;
  const says = sheetSays(field.value, revisionField.value, fileField.value);
  // T119: the message goes on the field that is wrong, and goes away as that field is edited (on the name alone,
  // a wrong revision put right left the name marked, and the form would not open)
  if (says.wrong) {
    const at = says.wrong === "name" ? field : revisionField, words = says.words;
    at.setCustomValidity(words);
    at.reportValidity();
    at.oninput = () => at.setCustomValidity("");
    return;
  }
  // (T470: as the intro's buttons, nothing while an answer is written: the sheet stays as it is, to be sent after it)
  if (select.disabled) return;
  repositorySheet.hidePopover();
  let entry: Unlisted;
  try {
    entry = await looked(hfEntry(says.repository, says.revision, { file: says.file }));
  } catch (error) {
    message("model error", String((error as Error).message ?? error));  // (a repository with nothing this page opens)
    return;
  }
  // a large GGUF is asked for before it is fetched, as a large model of the list is: not one this browser kept
  if (asksBefore(entry) && await stillKept(entry)) {
    entry = { ...entry, download: 0 };
  }
  local = entry;
  showLocal();  // the change of model writes the URL (nameInUrl), so that the link can be shared
};

/** T505.3: the repository the URL names (?hf=) with no file of it is looked at before its load: one without
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

/** The select's entry, chosen: the sheet for the last one of the Hugging Face group, the model for any other. Each way
 * of choosing (the select, the intro's buttons, the disk's files, the sheet) sets the select and calls this. */
function chooseSelected() {
  if (select.value === OTHER) {
    select.value = shownForOther(model, !!select.querySelector('option[value="local"]'));
    repositorySheet.showPopover();
    return;
  }
  chooseModel(chosenOf(select.value, local), keptEntries.has(select.value));
}
/** A model chosen (`kept`: the browser kept it, and nothing is fetched): asked for first where choosing.ts says so,
 * then the page's model, its address, its settings, and the worker's load. */
function chooseModel(chosen: Model, kept: boolean) {
  const asked = askedFirst(weighed(chosen), kept, (navigator as Navigator & { deviceMemory?: number }).deviceMemory);
  if (asked && !confirm(asked)) {
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
  // search: this may reach the worker before init does, while the page looks at what is kept (T116)
  worker.postMessage({ type: "load", root, search: location.search, model: weighed(model), load, threads: threadsFor(model), gpu: gpuFor(model) } satisfies ToWorker);
}
select.onchange = chooseSelected;
