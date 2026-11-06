// The whole report of /benchmark/: the model's table on top, a line a section (T185), the warnings at its head
// (T227), every section, and the link to the issue (T643.11: from the page's part, src/benchmark/report.ts, which
// writes what this returns into the page).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)
import { tableCell } from "./cells.ts";
import { benchMarkdown } from "./rounds.ts";
import { cpuBaseline } from "./cpu.ts";
import { deviceSummary, cpuSummary, storageSummary, lineSummary, gpuSummary } from "./summary.ts";
import { warnings, warningsBlock } from "./warnings.ts";
import { shortReport, reportUrl, reportTooLong } from "./report.ts";
import { SECTIONS, STATE, TITLES } from "./sections.ts";
import { memorySummary } from "../benchmark/page-memory.ts";
import type { Name, Result } from "./sections.ts";
import type { Environment } from "./rounds.ts";
import type { BenchRow } from "../runtime/contracts/messages.ts";

// what the model section measured, under the name of the model it ran (src/benchmark/model.ts's measured)
export type Measured = { name: string; rows: BenchRow[]; pyodide: string; paths?: string; said?: string[] };
// the report as the page writes it: its Markdown, the summary the link holds where that is too long (T185), the
// link, and whether the whole (long) and the summary as well (longer, T192) are too long for it
export type Whole = { markdown: string; summary: string; href: string; long: boolean; longer: boolean };

// T185: a section's lines of the summary, from the data its tables are made of; a section that did not measure says
// its state and the first line of why. The GPU's CPU beside it is the CPU section's, as in its tables.
const SUMMARIES: Partial<Record<Name, (data: any, results: Partial<Record<Name, Result>>) => string[]>> = {
  device: deviceSummary, cpu: cpuSummary, storage: storageSummary, line: lineSummary, memory: memorySummary,
  gpu: (data, results) => gpuSummary(data.found, cpuBaseline(results.cpu), { lost: data.lost }),
};
function summaryOf(name: Name, result: Result, results: Partial<Record<Name, Result>>): string[] {
  const lines = result.data && SUMMARIES[name] ? SUMMARIES[name]!(result.data, results) : [];
  if (result.status === "ok" && lines.length) return lines;
  const why = tableCell(result.markdown.split("\n")[0]).slice(0, 200);
  return [`${TITLES[name]}: ${STATE[result.status]}${lines.length || !why ? "" : `: ${why}`}`, ...lines];
}

/** One report: the model's table first (as ?bench=1 writes it, which tests/reports.ts reads), then every section.
 * results: the sections' results; measured: the model section's, where it measured; environment: the machine's
 * (environmentOf()); hidden: the line of the seconds the page was hidden (T194: "" where it never was). */
export function reportOf({ results, measured, environment, hidden }:
  { results: Partial<Record<Name, Result>>; measured?: Measured; environment: Environment; hidden: string }): Whole {
  // the model section's table is the one on top; the section itself only where it did not measure
  const shown = SECTIONS.filter((name) => results[name] && (name !== "model" || results[name]!.status !== "ok"));
  const sections = shown.map((name) => `#### ${TITLES[name]}${results[name]!.status === "ok" ? "" : ` (${STATE[results[name]!.status]})`}\n\n${results[name]!.markdown}`);
  // T184: the model page's own path right under the rounds' table (a table of its own: parseReport() reads the first)
  // T194: the seconds the page was hidden while it ran, right under the machine (no line when it never was)
  const head = [benchMarkdown(measured?.rows ?? [], environment), ...(hidden ? [hidden] : []), ...(measured?.paths ? [measured.paths] : [])].join("\n\n");
  // T227: all that came out WRONG, failed, unsteady or skipped, in one place under the top (parseReport() reads the
  // rounds' table above it), from the Markdown of the top and of the sections, in the page's own words
  const warned = warnings([{ title: TITLES.model, markdown: head, said: measured?.said }, ...shown.map((name) => ({ title: TITLES[name], ...results[name]! }))]);
  const markdown = [head, warningsBlock(warned), ...sections].filter(Boolean).join("\n\n");
  // T185: where the whole is too long for the link, the link holds the head, the warnings and a line a section
  const summary = shortReport(head, shown.flatMap((name) => summaryOf(name, results[name]!, results)), warned, environment);
  // T192: where the summary is too long as well, the link holds only the request to paste (TOO_LONG)
  const long = reportTooLong(markdown, environment), longer = long && reportTooLong(summary, environment);
  return { markdown, summary, href: reportUrl(markdown, environment, summary), long, longer };
}
