// src/page/draw.ts (T.355): what the page draws: the bubbles, the lines under an answer, the benchmark's table, the
// status line, the round button.
import { reportUrl } from "../bench.js";
import { chat, run, statusText } from "./dom.ts";
import { parameters } from "./address.ts";
import { benchmarking, gpuNote, gpuSaid, model, outdated, readyLine } from "./model.ts";
import { generating } from "./answer.ts";
import { thoughtLine, thoughtOf } from "./thought.ts";
import { markdownBroke } from "./markdown.ts";
import type { Markdown } from "./markdown.ts";
import { fixSeed } from "./settings.ts";
import { SLOWER } from "../runtime/updated.ts";
import type { BenchRow, Done } from "../runtime/contracts/messages.ts";

// The round button is the only control: an arrow that sends, a square that stops the run that is going on.
const RUN_ICON = "M3 20.5 21 12 3 3.5v6.6L15 12 3 13.9z", STOP_ICON = "M7 7h10v10H7z";
export function showRun() {
  const current = generating;
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
/** T.554: the answer's bubble says `text`, all that was written so far, and the chat follows it only while the reader is
 * at its bottom. As plain text, the new part is added to its text node (the text before it is not made again). T.556: as
 * Markdown where the answer is drawn so and the libraries came: the thinking stays plain text, and of the rest only
 * the blocks that changed are made again (the last one as it grows). */
export function drawAnswer(bubble: Element, text: string, shape?: { markdown: boolean; thinking: boolean }, markdown?: Markdown) {
  const pinned = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 40;
  if (shape?.markdown) drawShaped(bubble, text, shape.thinking, markdown);
  else plainly(bubble, text);
  if (pinned) chat.scrollTop = chat.scrollHeight;
}
/** `text` as the one text node of `element` (what it held before is taken away where it was something else), the new
 * part added to it where the text before it is the same. */
function plainly(element: Element, text: string) {
  let node = element.firstChild as CharacterData | null;
  if (node?.nodeType !== 3 || element.childNodes.length > 1) {  // (3: a text node)
    element.textContent = "";
    element.append(node = document.createTextNode(""));
  }
  if (text.startsWith(node!.data)) node!.appendData(text.slice(node!.length));
  else node!.data = text;
}
// T.520: the folds of thoughts that have closed, and those a reader opened or closed: the page leaves those as they are
const folded = new WeakSet<Element>(), touched = new WeakSet<Element>();
/** An answer in a format: T.520, its thought (if any) in a fold above it, open while the model thinks and closed once
 * it answers (unless the reader opened or closed it), and the answer itself, as Markdown where the libraries came. */
function drawShaped(bubble: Element, text: string, opened: boolean, markdown?: Markdown) {
  const { thought, closed, answer } = thoughtOf(text, opened);
  let fold = bubble.querySelector(":scope > details.thought") as HTMLDetailsElement | null;
  let said = bubble.querySelector(":scope > div.answer") as HTMLElement | null;
  if (!said) {
    bubble.textContent = "";
    bubble.append(said = document.createElement("div"));
    said.className = "answer";
  }
  if (thought !== null) {
    if (!fold) {
      fold = document.createElement("details");
      fold.className = "thought";
      const line = document.createElement("summary"), words = document.createElement("div");
      line.textContent = thoughtLine(false);
      line.onclick = () => touched.add(fold!);
      fold.append(line, words);
      fold.open = !closed;
      bubble.prepend(fold);
    }
    plainly(fold.lastElementChild!, thought);
    if (closed && !folded.has(fold)) {
      folded.add(fold);
      fold.firstElementChild!.textContent = thoughtLine(true);
      if (!touched.has(fold)) fold.open = false;
    }
  }
  let drawn = false;
  if (markdown) {
    try {
      drawMarkdown(said, answer, markdown);
      drawn = true;
    } catch (error) {
      markdownBroke(error);
    }
  }
  if (!drawn) {
    said.classList.remove("markdown");
    plainly(said, answer);
  }
}
/** T.520: the line of an answer's fold, at its end: how long the thought was, by the engine's count. */
export function thoughtSaid(bubble: Element, stats: { thought: number | null; thought_cut: boolean }) {
  const line = bubble.querySelector(":scope > details.thought > summary");
  if (line) line.textContent = thoughtLine(folded.has(line.parentElement!), stats);
}
// the source of each block drawn, by its element
const sources = new WeakMap<Element, string>();
function drawMarkdown(list: Element, text: string, markdown: Markdown) {
  if (!list.classList.contains("markdown")) {
    list.textContent = "";  // (it was plain text until the libraries came)
    list.classList.add("markdown");
  }
  const all = markdown.blocks(text), links = all.links;  // (marked's references, which a block may name)
  const blocks = all.filter((block) => block.type !== "space");
  const drawnBlocks = [...list.children];
  let same = 0;
  while (same < blocks.length && same < drawnBlocks.length && sources.get(drawnBlocks[same]) === blocks[same].raw) same += 1;
  for (const gone of drawnBlocks.slice(same)) gone.remove();
  for (const block of blocks.slice(same)) {
    const element = document.createElement("div");
    element.append(markdown.html(block, links));
    sources.set(element, block.raw);
    list.append(element);
  }
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
// T.76: the benchmark's bubble. The table is the Markdown's; the buttons under it are the way to send it on.
export function showBench(benchRows: BenchRow[], environment: any, markdown: string) {
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
    const skipped = row.skip !== undefined;  // T.214: its speed is "skipped", and the last column says why
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
  // T.91: the issue template's questions (device, OS, browser) above the Markdown
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
  line.append(head);
  rows(line, breakdown);
  if (seed === undefined) element.append(line);
  else {
    // the seed is a button: the next run, of this model or another, uses it again. It stands beside the line and not
    // in its summary (T.372: a button inside a <summary> opened the breakdown too, and is no interactive content there)
    const again = document.createElement("button");
    again.className = "seed";
    again.type = "button";
    again.title = "Use this seed for the next run";
    again.textContent = `seed ${seed}`;
    again.onclick = () => fixSeed(seed);
    const beside = document.createElement("span");
    beside.className = "seed-beside";
    beside.append("· ", again);
    const both = document.createElement("div");
    both.className = "meta-line";
    both.append(line, beside);
    element.append(both);
  }
  chat.scrollTop = chat.scrollHeight;
}
export const seconds = (value: number) => `${value.toFixed(value < 10 ? 2 : 1)} s`;
export function showStatus(note?: string) {
  if (gpuSaid(note)) console.info(`gpu: ${note}`);
  if (!readyLine || benchmarking) return;
  const { backend, threads, pyodide } = readyLine;
  // (?gpuTest=on, the tests' flag, says so as ?without= does)
  const tested = gpuNote && parameters.get("gpuTest") === "on" ? " (gpuTest)" : "";
  // (T.439: and, last, that a reload would make it faster, where it runs slower for a file of this build that is gone)
  statusText.textContent = `${model.name} · ${backend}${threads > 1 ? `, ${threads} threads` : ""}${gpuNote ? `, ${gpuNote}${tested}` : ""} · Pyodide ${pyodide}${outdated ? ` · ${SLOWER}` : ""}`;
}
// T.135: how many of the prompt's tokens went through the GPU (the ones counted as the prompt: the one after them
// makes the first logits, on the CPU)
export const onGpu = (data: Done) => (!data.gpuTokens ? "" : data.gpuTokens >= data.prompt_tokens ? " on WebGPU"
  : ` (${data.gpuTokens} on WebGPU)`);
// T.152: how many of the tokens it wrote the GPU sampled (the sampled steps: a stop token among them)
export const onGpuSampled = (data: Done) => (!data.gpuSampled ? "" : data.gpuSampled >= data.sampled ? " on WebGPU"
  : ` (${data.gpuSampled} on WebGPU)`);
