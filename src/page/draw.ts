// src/page/draw.ts (T355): what the page draws: the bubbles, the lines under an answer, the benchmark's table, the
// status line, the round button.
import { reportUrl } from "../bench.js";
import { chat, run, statusText } from "./dom.ts";
import { parameters } from "./address.ts";
import { page } from "./state.ts";
import { fixSeed } from "./settings.ts";

// The round button is the only control: an arrow that sends, a square that stops the run that is going on.
const RUN_ICON = "M3 20.5 21 12 3 3.5v6.6L15 12 3 13.9z", STOP_ICON = "M7 7h10v10H7z";
export function setGenerating(current: boolean) {
  page.generating = current;
  run.querySelector("path")!.setAttribute("d", current ? STOP_ICON : RUN_ICON);
  run.setAttribute("aria-label", current ? "Stop" : "Run");
}

export function message(kind: string, text = "") {
  const element = document.createElement("div");
  element.className = `message ${kind}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  element.append(bubble);
  chat.append(element);
  chat.scrollTop = chat.scrollHeight;
  return element;
}
// The numbers are the point of this experiment, so they stay on the page: a short line under the answer that
// opens into the breakdown. Tapping the summary is the only interaction, and closed it reads as one line.
export function rows(element: Element, lines: string[]) {
  for (const text of lines) {
    const row = document.createElement("div");
    row.textContent = text;
    element.append(row);
  }
}
// T76: the benchmark's bubble. The table is the Markdown's; the buttons under it are the way to send it on.
export function showBench(benchRows: any[], environment: any, markdown: string) {
  const element = message("model bench");
  const bubble = element.firstElementChild!;
  bubble.textContent = "";
  const title = document.createElement("div");
  title.className = "bench-title";
  title.textContent = [environment.model, environment.threads !== undefined && `${environment.threads} logical cores`,
    environment.pyodide && `Pyodide ${environment.pyodide}`].filter(Boolean).join(" · ");
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const text of ["what ran", "tok/s", "backend"]) {
    const cell = document.createElement("th");
    cell.textContent = text;
    head.append(cell);
  }
  const body = table.createTBody();
  for (const row of benchRows) {
    const line = body.insertRow();
    const skipped = row.skip !== undefined;  // T214: its speed is "skipped", and the last column says why
    for (const text of [row.name, skipped ? "skipped" : Number.isFinite(row.speed) ? row.speed.toFixed(1) : "?", skipped ? row.skip : row.backend ?? ""]) {
      line.insertCell().textContent = text;
    }
  }
  bubble.append(title, table);
  const line = document.createElement("div");
  line.className = "meta bench-actions";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "seed";
  copy.textContent = "copy as Markdown";
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(markdown);
      copy.textContent = "copied";
    } catch {
      copy.textContent = "could not copy";
    }
    setTimeout(() => { copy.textContent = "copy as Markdown"; }, 1500);
  };
  // T91: the issue template's questions (device, OS, browser) above the Markdown
  const report = document.createElement("a");
  report.href = reportUrl(markdown, environment);
  report.target = "_blank";
  report.rel = "noopener";
  report.textContent = "report it as an issue";
  line.append(copy, " · ", report);
  element.append(line);
  chat.scrollTop = chat.scrollHeight;
}
export function meta(element: Element, text: string, breakdown: string[], seed?: number) {
  const line = document.createElement("details");
  line.className = "meta";
  const head = document.createElement("summary");
  head.textContent = text;
  if (seed !== undefined) {
    // the seed is a button: the next run, of this model or another, uses it again
    const again = document.createElement("button");
    again.className = "seed";
    again.type = "button";
    again.title = "Use this seed for the next run";
    again.textContent = `seed ${seed}`;
    again.onclick = (event) => {
      event.preventDefault(); // or the line would open
      fixSeed(seed);
    };
    head.append(" · ", again);
  }
  line.append(head);
  rows(line, breakdown);
  element.append(line);
  chat.scrollTop = chat.scrollHeight;
}
export const seconds = (value: number) => `${value.toFixed(value < 10 ? 2 : 1)} s`;
export function showStatus(note?: string) {
  if (note && note !== page.gpuNote) console.info(`gpu: ${note}`);
  page.gpuNote = note ?? page.gpuNote;
  if (!page.readyLine || page.benchmarking) return;
  const { backend, threads, pyodide } = page.readyLine;
  // (?gpuTest=on, the tests' flag, says so as ?without= does)
  const tested = page.gpuNote && parameters.get("gpuTest") === "on" ? " (gpuTest)" : "";
  statusText.textContent = `${page.model.name} · ${backend}${threads > 1 ? `, ${threads} threads` : ""}${page.gpuNote ? `, ${page.gpuNote}${tested}` : ""} · Pyodide ${pyodide}`;
}
// T135: how many of the prompt's tokens went through the GPU (the ones counted as the prompt: the one after them
// makes the first logits, on the CPU)
export const onGpu = (data: any) => (!data.gpuTokens ? "" : data.gpuTokens >= data.prompt_tokens ? " on WebGPU"
  : ` (${data.gpuTokens} on WebGPU)`);
// T152: how many of the tokens it wrote the GPU sampled (the sampled steps: a stop token among them)
export const onGpuSampled = (data: any) => (!data.gpuSampled ? "" : data.gpuSampled >= data.sampled ? " on WebGPU"
  : ` (${data.gpuSampled} on WebGPU)`);
