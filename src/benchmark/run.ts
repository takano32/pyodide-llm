// Running sections: the sections' elements and their Run buttons, the run of one or all, the clipboard, a tab that was
// loaded again by the page-memory section, and ?run=.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import { forgetMark, readMark, memoryResult } from "./page-memory.ts";
import { cpuSpeed, CPU_SPEED_KEY, gpuMarkdown, STATE } from "../bench.ts";
import { $, parameters, ALL, told } from "./dom.ts";
import { SECTIONS } from "../bench.ts";
import type { Name, Result } from "../bench.ts";
import { gpuRewritten, results, runAt, runBegan, runEnded, sectionAt, sectionBegan, sectionEnded } from "./running.ts";
import { parts, rendered, clock, staged, ticked, endWorkers, stalled } from "./section.ts";
import { device, cpu } from "./cpu.ts";
import { modelSection } from "./model.ts";
import { gpu } from "./gpu.ts";
import { storage, line } from "./storage.ts";
import { tabStorage, pageMemory } from "./memory.ts";
import { report, awake } from "./report.ts";

for (const name of SECTIONS) parts[name].run.onclick = () => runSections([name]);

const RUN: { [Section in Name]: () => Promise<Result<Section>> } = { device, cpu, model: modelSection, gpu, storage, line, memory: pageMemory };
async function runSections(names: readonly Name[]) {
  if (!runBegan(names, performance.now())) return;
  const buttons = [$("run-all"), ...Object.values(parts).map((p) => p.run)] as HTMLButtonElement[];
  buttons.forEach((b) => (b.disabled = true));
  told({ done: false });
  $("overall").hidden = names.length < 2;
  const tick = setInterval(ticked, 1000);
  awake.start();
  for (const name of names) {
    const { state, progress, out } = parts[name];
    state.className = "state";
    state.textContent = "running...";
    out.innerHTML = "";
    sectionBegan(name, performance.now());
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
      endWorkers();
      // T173's review: a page-memory run stopped for its silence never reaches its own finally: its mark would
      // later read as a tab the browser ended
      if (name === "memory") forgetMark(tabStorage);
    }
    sectionEnded(name, result);
    // T156: the CPU's reading of the weights here, for the model page (a model on the GPU alone is held against it)
    const speed = name === "cpu" ? cpuSpeed(results.cpu) : undefined;
    if (speed) {
      try {
        localStorage.setItem(CPU_SPEED_KEY, JSON.stringify(speed));
      } catch {
        // no storage here: a model on the GPU alone stays there
      }
    }
    progress.box.hidden = true;
    state.className = `state ${result.status}`;
    state.textContent = `${STATE[result.status].shown} · ${clock(performance.now() - sectionAt)}`;
    out.innerHTML = rendered(result.markdown);
    // T157: the GPU section's tables hold the GPU against the CPU section; a CPU section run after it (on its own
    // button) writes them again, and the report with them
    const g = results.gpu;
    if (name === "cpu" && g?.data?.found) {
      const rewritten = { ...g, ...gpuMarkdown(g.data.found, g.data.bridge, g.data.lost, results.cpu) };
      gpuRewritten(rewritten);
      parts.gpu.out.innerHTML = rendered(rewritten.markdown);
    }
    report();
  }
  clearInterval(tick);
  awake.stop();
  if (names.length > 1) $("overall").textContent = `${names.length} sections in ${clock(performance.now() - runAt)}`;
  buttons.forEach((b) => (b.disabled = false));
  runEnded();
  told({ done: true });
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
  const result = memoryResult(ended);
  sectionEnded("memory", result);
  parts.memory.state.className = `state ${result.status}`;
  parts.memory.state.textContent = `${STATE[result.status].shown} · the page was loaded again`;
  parts.memory.out.innerHTML = rendered(result.markdown);
  report();
}
told({ all: ALL });
// ?run=all runs every section at once but the page memory, ?run=cpu,line those (tests/bench-check.ts)
const asked = parameters.get("run");
if (asked) runSections((asked === "all" ? ALL : SECTIONS.filter((name) => asked.split(",").includes(name))).filter((name) => !(ended && name === "memory")));
