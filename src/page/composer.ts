// src/page/composer.ts (T355): the text field and the round button: what is sent to the worker, and the stop.
import { filled } from "../models.js";
import { $, files, fit, prompt, run, select } from "./dom.ts";
import { answerBegan, page } from "./state.ts";
import { settings } from "./settings.ts";
import { message, showRun } from "./draw.ts";
import { worker } from "./receiver.ts";

// Enter breaks the line and Shift + Enter does too; Ctrl + Enter or Cmd + Enter sends. A phone has neither key
// and sends with the button.
prompt.onkeydown = (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) {
    event.preventDefault();
    ($("composer") as HTMLFormElement).requestSubmit();
  }
};
prompt.oninput = fit;
if (/Mac|iPhone|iPad/.test(navigator.platform)) {
  $("send-hint").textContent = "⌘ ⏎";
  run.title = "⌘ + Enter";
}

run.onclick = (event) => {
  // While generating the button stops the run instead of submitting the form. A textarea never submits a form
  // implicitly, so nothing but a press of the button itself arrives here.
  if (page.generating) {
    event.preventDefault();
    worker.postMessage({ type: "stop" });
  }
};

$("composer").onsubmit = (event) => {
  event.preventDefault();
  // Ctrl + Enter asks for this even while the button is disabled or a run is going on
  if (run.disabled || page.generating) {
    return;
  }
  select.disabled = files.disabled = true;
  message("user", prompt.value);
  const how = answerBegan(message("model"), settings, () => crypto.getRandomValues(new Uint32Array(1))[0]);
  showRun();
  // an instruction-tuned model gets what was typed inside the form it was trained on; the bubble shows what was typed
  // what src/models.js says for this model, or what its own chat_template gave (T73)
  const template = ((page.model as any).template ?? page.fromTemplate) as string | undefined;
  worker.postMessage({ type: "generate", prompt: template ? filled(template, prompt.value) : prompt.value, echo: !template, ...how });
};
