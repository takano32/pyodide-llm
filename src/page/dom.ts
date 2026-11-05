// src/page/dom.ts (T355): the elements of the model page (src/pages/index.astro) that its script's parts share, and
// what is decided once about the markup as the script starts.

export const $ = (id: string) => document.getElementById(id)!;
export const chat = $("chat"), select = $("model") as HTMLSelectElement, prompt = $("prompt") as HTMLTextAreaElement;
export const run = $("run") as HTMLButtonElement, statusText = $("status-text"), progress = $("progress") as HTMLProgressElement;
export const files = $("files") as HTMLInputElement;

// the text field is as tall as its text
export function fit() {
  prompt.style.height = "auto";
  prompt.style.height = `${prompt.scrollHeight}px`;
}

// the ribbon or the corner, whichever the dice say
document.querySelector(Math.random() < 0.5 ? ".ribbon" : ".github-corner")!.removeAttribute("hidden");

// The intro is written twice; a phone has no room for both. Japanese for a browser set to Japanese, English for
// everyone else.
document.querySelector<HTMLElement>(".intro")!.dataset.language = navigator.language.startsWith("ja") ? "ja" : "en";
