// What every section runs on: its elements' record, Markdown as the page shows it, the stage it is at and its seconds
// (T177), a worker asked once, the watch for a section that says nothing, and the words the sections share.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import "./isolation.ts";
import { MODELS } from "../models.js";
import { BASE, V, $, type Name, type Result } from "./dom.ts";
import { page } from "./state.ts";

// ---- the sections' elements: a head with a Run button and the state, the progress while it runs (T177), and what
// the section found
type Progress = { box: HTMLElement; bar: HTMLElement; fill: HTMLElement; stage: HTMLElement; elapsed: HTMLElement };
export const parts: Record<string, { run: HTMLButtonElement; state: HTMLElement; progress: Progress; out: HTMLElement }> = {};

// Markdown as the page shows it: tables and paragraphs, which is all the sections write
const escape = (text: string) => text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]!));
const inline = (text: string) => escape(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/&lt;sub&gt;(.*?)&lt;\/sub&gt;/g, "<small>$1</small>");
export function rendered(markdown: string) {
  const html: string[] = [];
  const lines = markdown.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("|")) {
      // a \| in a cell is a | of its words, as GitHub reads it (T157's review: an error's words)
      const cells = (row: string) => row.split(/(?<!\\)\|/).slice(1, -1).map((c) => c.trim().replace(/\\\|/g, "|"));
      const head = cells(line);
      const rows: string[][] = [];
      for (i += 2; i < lines.length && lines[i].startsWith("|"); i++) rows.push(cells(lines[i]));
      i--;
      html.push(`<table${head.length === 2 ? ' class="pairs"' : ""}><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr>${rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</table>`);
    } else if (line.startsWith("#")) {
      continue;  // the section's own title is on the page already
    } else if (line.trim()) {
      html.push(`<p>${inline(line.replace(/^- /, "· "))}</p>`);
    }
  }
  return html.join("");
}

// ---- T177: the stage the running section is at. at / of where the section knows how many stages it has (the bar
// then fills stage by stage, and within a stage by part, 0 to 1, where that is known), else the stage's name only
export type Stage = { stage: string; at?: number; of?: number; part?: number };
export const clock = (ms: number) => {
  const s = Math.floor(ms / 1000);
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
};
export function staged(name: Name, { stage, at, of, part = 0 }: Stage) {
  const p = parts[name].progress;
  const counted = Boolean(at && of);
  p.bar.classList.toggle("unknown", !counted);
  p.fill.style.width = counted ? `${(100 * Math.min(1, (at! - 1 + Math.min(1, part)) / of!)).toFixed(1)}%` : "";
  if (counted) p.bar.setAttribute("aria-valuenow", String(Math.round((100 * (at! - 1)) / of!)));
  else p.bar.removeAttribute("aria-valuenow");
  p.stage.textContent = counted ? `${at} / ${of} · ${stage}` : stage;
}
export function ticked() {
  if (!page.current) return;
  const now = performance.now();
  parts[page.current].progress.elapsed.textContent = clock(now - page.sectionBegan);
  if (page.runNames.length > 1) {
    const title = document.querySelector(`section[data-section="${page.current}"] h2`)?.textContent;
    $("overall").textContent = `Section ${page.runNames.indexOf(page.current) + 1} / ${page.runNames.length} · ${title} · ${clock(now - page.runBegan)} in all`;
  }
}

// one worker, one question, one answer (the section workers answer every message once)
export function ask(worker: Worker, message: any): Promise<any> {
  return new Promise((resolve) => {
    worker.onmessage = ({ data }) => {
      // the GPU section's device-lost notice comes on its own, and so do the line's "still reading", the GPU
      // prompt's "still at it" and a section's stages (T177)
      if (data.lost || data.alive || data.stage !== undefined) return;
      resolve(data);
    };
    worker.onerror = (event) => resolve({ error: `the worker failed: ${event.message ?? "error"}` });
    worker.postMessage(message);
  });
}
// the workers of the running section, and when one of them last said anything: a section that says nothing for
// STALL_MS is stopped, its workers ended, and the run goes on (T134's review: a GPU without Atomics.waitAsync in its
// worker never ended the bridge, and the storage and the line never ran)
export const live = new Set<Worker>();
let heard = 0;
export function worker(file: string) {
  const w = new Worker(`${BASE}${file}${V}`, { type: "module" });
  live.add(w);
  w.addEventListener("message", ({ data }) => {
    heard = performance.now();
    if (page.current && typeof data?.stage === "string") staged(page.current, data);
  });
  return w;
}
const STALL_MS = 5 * 60_000, TICK_MS = 5000;
export function stalled(): { promise: Promise<Result>; stop: () => void } {
  heard = performance.now();
  let last = heard, timer = 0;
  const promise = new Promise<Result>((resolve) => {
    timer = setInterval(() => {
      const now = performance.now();
      // a tick far later than asked for is a page the device stopped (a switched app, a frozen tab): the section's
      // workers were stopped with it, and that time is no silence of theirs
      if (now - last > 2 * TICK_MS) heard += now - last - TICK_MS;
      last = now;
      if (now - heard > STALL_MS) resolve({ status: "error", markdown: "stopped: nothing came from this section for 5 minutes" });
    }, TICK_MS) as unknown as number;
  });
  return { promise, stop: () => clearInterval(timer) };
}
export const fixed = (value: any, digits = 1) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "?");
export const yes = (value: boolean) => (value ? "yes" : "none");
export const model = () => MODELS.find((entry: any) => entry.id === ($("model") as HTMLSelectElement).value) as any;
