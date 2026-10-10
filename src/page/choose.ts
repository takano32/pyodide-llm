// src/page/choose.ts (T355): choosing a model: the select, the files of the visitor's own disk, the sheet that asks
// for a repository of Hugging Face, the buttons of the intro. Each ends in the select's change, which tells the worker.
import { MODELS, memoryWarning } from "../models.js";
import { $, files, progress, run, select } from "./dom.ts";
import { REPOSITORY, REVISION, fromUrl, hfEntry, weighed } from "./address.ts";
import { gpuFor, threadsFor } from "./remembered.ts";
import { page } from "./state.ts";
import { show } from "./settings.ts";
import { message } from "./draw.ts";
import { worker } from "./receiver.ts";

// A model of the visitor's own disk, from the folder button or dropped on the page. The legacy format knows
// nothing about itself, so whatever differs from llama2.c's conventions comes as a third file: .json, shaped
// like an entry of src/models.js ({options, generation, prompt, placeholder}).
let local: typeof page.model | undefined = fromUrl;
// The files Hugging Face publishes: model.safetensors, config.json and a tokenizer (tokenizer.json, or a
// sentencepiece tokenizer.model / spiece.model). The worker converts them in the browser, to int8 and with a
// context of 512 tokens unless the settings say otherwise ({"conversion": {"dtype": ..., "max_seq_len": ...}}).
const HF_IGNORED = ["tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "model.safetensors.index.json"];
async function openHuggingFace(chosen: File[]) {
  const named = (...names: string[]) => chosen.find(({ name }) => names.includes(name.toLowerCase()));
  const weights = chosen.filter(({ name }) => name.toLowerCase().endsWith(".safetensors"));
  // every tokenizer the folder has, in the order the worker tries them (T138: a folder with both, as RakutenAI 2.0
  // mini publishes, stopped at the tokenizer.json the converter refuses)
  const config = named("config.json"), tokenizers = [named("tokenizer.json"), named("tokenizer.model"), named("spiece.model")].filter(Boolean);
  const settings = chosen.find(({ name }) => name.toLowerCase().endsWith(".json") && name !== config?.name && !tokenizers.some((file) => file!.name === name) && !HF_IGNORED.includes(name.toLowerCase()));
  if (weights.length !== 1 || !config || !tokenizers.length) {
    throw new Error("A Hugging Face model needs three files together: one .safetensors file (a model in several shards is not supported), config.json, and tokenizer.json, tokenizer.model or spiece.model.");
  }
  const given = settings ? JSON.parse(await settings.text()) : {};
  return {
    id: "local", name: given.name ?? weights[0].name, note: `local · Hugging Face · ${(weights[0].size / 1e6).toFixed(0)} MB`,
    // T374.2.2: the folder as it was chosen, and which of it is the weights. What else of it is read (config.json, the
    // chat template where it has one, T127, the first tokenizer the converter can read, T138) is asked for by name
    // by the conduct of the conversion (public/convert/conduct.py), and answered from these Files by the worker
    hf: { files: chosen, weights: weights[0].name },
    conversion: given.conversion ?? {}, options: given.options ?? {},
    // a model nobody has tuned this page for: sample, as such models loop when they decode greedily
    generation: given.generation ?? { steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty: 1.1 },
    prompt: given.prompt ?? "", placeholder: given.placeholder ?? "",
  };
}

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
  const settings = chosen.find(({ name }) => name.toLowerCase().endsWith(".json"));
  // the larger of the other two is the checkpoint
  const [checkpoint, tokenizer, ...more] = chosen.filter((file) => file !== settings).sort((a, b) => b.size - a.size);
  try {
    if (!tokenizer || more.length) {
      throw new Error("Choose two files together: the checkpoint and its tokenizer.bin (and, optionally, settings as .json).");
    }
    const given = settings ? JSON.parse(await settings.text()) : {};
    // WebAssembly addresses 32 bits, and a phone gives a tab far less than that
    if (checkpoint.size > 1e9 && !confirm(`${checkpoint.name} has ${(checkpoint.size / 1e9).toFixed(1)} GB. That may be more than this browser can hold. Read it anyway?`)) {
      return;
    }
    local = {
      id: "local", name: given.name ?? checkpoint.name, note: `local · ${(checkpoint.size / 1e6).toFixed(0)} MB`,
      file: checkpoint, tokenizerFile: tokenizer, bytes: checkpoint.size, options: given.options ?? {},
      generation: given.generation ?? { steps: 0, temperature: 0.0 },
      prompt: given.prompt ?? "Once upon a time", placeholder: given.placeholder ?? "",
    } as any;
  } catch (error) {
    message("model error", String((error as Error).message ?? error));
    return;
  }
  showLocal();
}
function showLocal() {
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
// (every switch of the panel reloads) or a shared link opens that model and no other (the review of T75 and T88:
// after ?hf=, choosing a model of the list and then a switch reloaded the Hugging Face model). A file of the disk
// has no address: its URL names nothing, and a reload opens the model remembered before it.
function nameInUrl(current: typeof page.model) {
  const url = new URL(location.href);
  for (const name of ["model", "hf", "revision", "checkpoint", "tokenizer", "template", "prompt"]) {
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
($("repository-form") as HTMLFormElement).onsubmit = (event) => {
  event.preventDefault();
  const field = $("repository-name") as HTMLInputElement, revisionField = $("repository-revision") as HTMLInputElement;
  // the address of the model's page is what people copy: owner/name is what is needed of it
  const repository = field.value.trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/(tree|blob)\/.*$/, "").replace(/\/+$/, "");
  const revision = revisionField.value.trim() || "main";
  // T119: the message goes on the field that is wrong, and goes away as that field is edited (on the name alone,
  // a wrong revision put right left the name marked, and the form would not open)
  const wrong = !REPOSITORY.test(repository) ? [field, "owner/name, as huggingface.co spells it"]
    : !REVISION.test(revision) ? [revisionField, "a branch, a tag or a commit"] : undefined;
  if (wrong) {
    const [at, words] = wrong;
    at.setCustomValidity(words);
    at.reportValidity();
    at.oninput = () => at.setCustomValidity("");
    return;
  }
  repositorySheet.hidePopover();
  local = hfEntry(repository, revision);
  showLocal();  // the change of model writes the URL (nameInUrl), so that the link can be shared
};

select.onchange = () => {
  if (select.value === "hf-other") {
    select.value = page.model.id === "local" && !select.querySelector('option[value="local"]') ? MODELS[0].id : page.model.id;
    repositorySheet.showPopover();
    return;
  }
  const chosen = select.value === "local" ? local! : MODELS.find(({ id }) => id === select.value)!;
  // a large download is asked for first: what is fetched, from whom, and that the conversion happens here
  const download = select.selectedOptions[0]?.textContent?.includes("kept in this browser") ? 0 : ((chosen as any).download ?? 0);
  // T90: and one that probably does not fit in a device that says how much memory it has (Chromium only)
  const asks = [download > 500e6 && `${chosen.name}: ${(download / 1e9).toFixed(1)} GB will be fetched from huggingface.co and converted in this browser.`,
                memoryWarning(weighed(chosen), (navigator as any).deviceMemory)].filter(Boolean);
  if (asks.length && !confirm(`${asks.join(" ")} Go on?`)) {
    select.value = page.model.id;
    return;
  }
  page.model = chosen;
  nameInUrl(page.model);
  page.benchmarking = false;  // the worker cancels the benchmark's round for the new model, and never reports it
  show(page.model);
  page.ready = false;
  page.readyLine = page.gpuNote = undefined;  // T148: the status line waits for the new model's
  run.disabled = true;
  progress.hidden = true;
  // the timings of the previous load
  $("load").classList.remove("measured");
  ($("load") as HTMLDetailsElement).open = false;
  // search: this may reach the worker before init does, while the page looks at what is kept (T116)
  worker.postMessage({ type: "load", search: location.search, model: weighed(page.model), load: ++page.loads, threads: threadsFor(page.model), gpu: gpuFor(page.model) });
};
