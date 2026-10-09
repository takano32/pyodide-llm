// Running sections: the sections' elements and their Run buttons, the run of one or all, the clipboard, a tab that was
// loaded again by the page-memory section, and ?run=.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import { forgetMark, readMark, memoryResult } from "../page-memory.js";
import { cpuBaseline, CPU_SPEED_KEY } from "../bench.js";
import { $, parameters, SECTIONS, type Name, ALL, type Result, results } from "./dom.ts";
import { page } from "./state.ts";
import { parts, rendered, clock, staged, ticked, live, stalled } from "./section.ts";
import { device, cpu } from "./cpu.ts";
import { modelSection } from "./model.ts";
import { gpu, gpuMarkdown } from "./gpu.ts";
import { storage, line } from "./storage.ts";
import { tabStorage, pageMemory } from "./memory.ts";
import { STATE, report, awake } from "./report.ts";

for (const section of document.querySelectorAll<HTMLElement>("section[data-section]")) {
  const name = section.dataset.section as Name;
  const head = document.createElement("div");
  head.className = "head";
  const title = section.querySelector("h2")!;
  const run = document.createElement("button");
  run.type = "button";
  run.textContent = "Run";
  const state = document.createElement("span");
  state.className = "state";
  head.append(title, run, state);
  section.prepend(head);
  const box = document.createElement("div");
  box.className = "progress";
  box.hidden = true;
  box.innerHTML = `<div class="bar" role="progressbar" aria-label="${title.textContent}"><div></div></div><span class="stage"></span><span class="elapsed"></span>`;
  const progress = { box, bar: box.querySelector<HTMLElement>(".bar")!, fill: box.querySelector<HTMLElement>(".bar > div")!,
    stage: box.querySelector<HTMLElement>(".stage")!, elapsed: box.querySelector<HTMLElement>(".elapsed")! };
  const out = document.createElement("div");
  out.className = "out";
  section.append(box, out);
  parts[name] = { run, state, progress, out };
  run.onclick = () => runSections([name]);
}

const RUN: Record<Name, () => Promise<Result>> = { device, cpu, model: modelSection, gpu, storage, line, memory: pageMemory };
let running = false;
async function runSections(names: readonly Name[]) {
  if (running) return;
  running = true;
  const buttons = [$("run-all"), ...Object.values(parts).map((p) => p.run)] as HTMLButtonElement[];
  buttons.forEach((b) => (b.disabled = true));
  (window as any).__benchmark.done = false;
  page.runNames = names;
  page.runBegan = performance.now();
  $("overall").hidden = names.length < 2;
  const tick = setInterval(ticked, 1000);
  awake.start();
  for (const name of names) {
    const { state, progress, out } = parts[name];
    state.className = "state";
    state.textContent = "running...";
    out.innerHTML = "";
    page.current = name;
    page.sectionBegan = performance.now();
    staged(name, { stage: "starting" });
    progress.box.hidden = false;
    ticked();
    let result: Result;
    const stall = stalled();
    try {
      result = await Promise.race([RUN[name](), stall.promise]);
    } catch (error) {
      result = { status: "error", markdown: String((error as Error)?.message ?? error) };
    } finally {
      stall.stop();
      live.forEach((w) => w.terminate());  // the section's own finally has done it, unless it was stopped
      live.clear();
      // T173's review: a page-memory run stopped for its silence never reaches its own finally: its mark would
      // later read as a tab the browser ended
      if (name === "memory") forgetMark(tabStorage);
    }
    results[name] = result;
    // T156: the CPU's reading of the weights here, for the model page (a model on the GPU alone is held against it)
    const speed = name === "cpu" ? cpuBaseline(result).token : undefined;
    if (speed?.isolated) {
      try {
        // (and a prompt's G MAC/s at its fastest: its model's layers' multiply-adds a token, over its ms)
        const prompt = cpuBaseline(result).prompt, layerWeights = result.data?.layerWeights;
        const promptGMACs = prompt && layerWeights ? layerWeights / (prompt.msPerToken / 1000) / 1e9 : undefined;
        localStorage.setItem(CPU_SPEED_KEY, JSON.stringify({ GBps: speed.GBps, threads: speed.threads, promptGMACs }));
      } catch {
        // no storage here: a model on the GPU alone stays there
      }
    }
    progress.box.hidden = true;
    state.className = `state ${result.status}`;
    state.textContent = `${STATE[result.status]} · ${clock(performance.now() - page.sectionBegan)}`;
    out.innerHTML = rendered(result.markdown);
    // T157: the GPU section's tables hold the GPU against the CPU section; a CPU section run after it (on its own
    // button) writes them again, and the report with them
    const g = results.gpu;
    if (name === "cpu" && g?.data?.steps) {
      Object.assign(g, gpuMarkdown(g.data.steps, g.data.bridge, g.data.lost));
      parts.gpu.out.innerHTML = rendered(g.markdown);
    }
    report();
  }
  clearInterval(tick);
  awake.stop();
  page.current = undefined;
  if (names.length > 1) $("overall").textContent = `${names.length} sections in ${clock(performance.now() - page.runBegan)}`;
  buttons.forEach((b) => (b.disabled = false));
  (window as any).__benchmark.done = true;
  running = false;
}
$("run-all").onclick = () => runSections(ALL);
// T192: where the clipboard refuses (or there is none), say so and select the whole for copying by hand: on a
// device with a GPU the report is mostly too long for the link, and the issue then asks for it to be pasted
function toClipboard() {
  const said = $("clipboard");
  said.hidden = true;
  const refused = () => {
    said.hidden = false;
    const selection = getSelection();
    selection?.removeAllRanges();
    selection?.selectAllChildren($("markdown"));
    said.scrollIntoView({ block: "nearest" });
  };
  try {
    if (!navigator.clipboard?.writeText) return refused();
    navigator.clipboard.writeText($("markdown").textContent ?? "").catch(refused);
  } catch {
    refused();
  }
}
$("copy").onclick = toClipboard;
// a report too long for the link goes by the clipboard, in the click itself (Safari writes only there)
$("issue").addEventListener("click", () => {
  if (!$("long").hidden || !$("longer").hidden) toClipboard();
});
// T173: a mark of the page-memory section is a tab ended (or left) while it grew: what it held is shown, and the
// section is not run again by the ?run= below (a ?run=memory would end the tab over and over)
const ended = readMark(tabStorage);
if (ended) {
  const result = (results.memory = memoryResult(ended) as Result);
  parts.memory.state.className = `state ${result.status}`;
  parts.memory.state.textContent = `${STATE[result.status]} · the page was loaded again`;
  parts.memory.out.innerHTML = rendered(result.markdown);
  report();
}
(window as any).__benchmark.all = ALL;
// ?run=all runs every section at once but the page memory, ?run=cpu,line those (tests/bench-check.mjs)
const asked = parameters.get("run");
if (asked) runSections((asked === "all" ? ALL : SECTIONS.filter((name) => asked.split(",").includes(name))).filter((name) => !(ended && name === "memory")));
