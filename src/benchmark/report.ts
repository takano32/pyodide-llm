// The report: the sections' titles and states, a line a section (T185), the warnings at its head (T227), the link to
// the issue, and the screen kept on while sections run (T194).
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
declare const __BUILD__: string;
import { deviceSummary, cpuSummary, storageSummary, lineSummary, gpuSummary, cpuBaseline, tableCell, environmentOf,
  benchMarkdown, warnings, warningsBlock, shortReport, reportUrl, reportTooLong } from "../bench.js";
import { memorySummary } from "../page-memory.js";
import { hiddenLine, wakeKeeper } from "../wake.js";
import { $, SECTIONS, type Name, type Result, results } from "./dom.ts";
import { measured } from "./model.ts";

export const STATE = { ok: "done", none: "not in this browser", wrong: "WRONG", error: "failed" };

const TITLES: Record<Name, string> = { device: "This browser", cpu: "CPU", model: "Model", gpu: "GPU", storage: "Storage", line: "Line", memory: "Page memory" };
// T185: a section's lines of the summary (src/bench.js), from the data its tables are made of; a section that did not
// measure says its state and the first line of why. The GPU's CPU beside it is the CPU section's, as in its tables.
const SUMMARIES: Partial<Record<Name, (data: any) => string[]>> = {
  device: deviceSummary, cpu: cpuSummary, storage: storageSummary, line: lineSummary, memory: memorySummary,
  gpu: (data) => gpuSummary(data.steps, cpuBaseline(results.cpu), { lost: data.lost }),
};
function summaryOf(name: Name, result: Result): string[] {
  const lines = result.data && SUMMARIES[name] ? SUMMARIES[name]!(result.data) : [];
  if (result.status === "ok" && lines.length) return lines;
  const why = tableCell(result.markdown.split("\n")[0]).slice(0, 200);
  return [`${TITLES[name]}: ${STATE[result.status]}${lines.length || !why ? "" : `: ${why}`}`, ...lines];
}

// ---- one report: the model's table first (as ?bench=1 writes it, which tests/reports.mjs reads), then every section
export function report() {
  // the model and its table only when the model section measured: a report of the other sections names no model,
  // which tests/reports.mjs then leaves out (T134's review)
  const environment = environmentOf(navigator, { model: measured?.name, pyodide: measured?.pyodide, build: __BUILD__, site: location.origin + location.pathname });
  // the model section's table is the one on top; the section itself only where it did not measure
  const shown = SECTIONS.filter((name) => results[name] && (name !== "model" || results[name]!.status !== "ok"));
  const sections = shown.map((name) => `#### ${TITLES[name]}${results[name]!.status === "ok" ? "" : ` (${STATE[results[name]!.status]})`}\n\n${results[name]!.markdown}`);
  // T184: the model page's own path right under the rounds' table (a table of its own: parseReport() reads the first)
  // T194: the seconds the page was hidden while it ran, right under the machine (no line when it never was)
  const hidden = hiddenLine(awake.hidden());
  const head = [benchMarkdown(measured?.rows ?? [], environment), ...(hidden ? [hidden] : []), ...(measured?.paths ? [measured.paths] : [])].join("\n\n");
  // T227: all that came out WRONG, failed, unsteady or skipped, in one place under the top (parseReport() reads the
  // rounds' table above it), from the Markdown of the top and of the sections, in the page's own words
  const warned = warnings([{ title: TITLES.model, markdown: head, said: measured?.said }, ...shown.map((name) => ({ title: TITLES[name], ...results[name]! }))]);
  const markdown = [head, warningsBlock(warned), ...sections].filter(Boolean).join("\n\n");
  $("markdown").textContent = markdown;
  // T185: where the whole is too long for the link, the link holds the head, the warnings and a line a section
  const summary = shortReport(head, shown.flatMap((name) => summaryOf(name, results[name]!)), warned, environment);
  ($("issue") as HTMLAnchorElement).href = reportUrl(markdown, environment, summary);
  // T192: where the summary is too long as well, the link holds only the request to paste (TOO_LONG)
  const long = reportTooLong(markdown, environment), longer = long && reportTooLong(summary, environment);
  $("long").hidden = !long || longer;
  $("longer").hidden = !longer;
  (document.querySelector(".report") as HTMLElement).hidden = false;
  (window as any).__benchmark.markdown = markdown;
}

// T194: the screen kept on while sections run, and the seconds the page was hidden counted
export const awake = wakeKeeper({ document, navigator });
