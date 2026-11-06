// The report: written into the page from src/bench/whole.ts's reportOf() (the sections' titles and states, a line a
// section (T185), the warnings at its head (T227), the link to the issue), and the screen kept on while sections run (T194).
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
declare const __BUILD__: string;
import { environmentOf, reportOf } from "../bench.ts";
import { hiddenLine, wakeKeeper } from "./wake.ts";
import { $, told } from "./dom.ts";
import { results } from "./running.ts";
import { measured } from "./model.ts";

// ---- one report: the model's table first (as ?bench=1 writes it, which tests/reports.ts reads), then every section
export function report() {
  // the model and its table only when the model section measured: a report of the other sections names no model,
  // which tests/reports.ts then leaves out (T134's review)
  const environment = environmentOf(navigator, { model: measured?.name, pyodide: measured?.pyodide, build: __BUILD__, site: location.origin + location.pathname });
  const whole = reportOf({ results, measured, environment, hidden: hiddenLine(awake.hidden()) });
  $("markdown").textContent = whole.markdown;
  ($("issue") as HTMLAnchorElement).href = whole.href;
  $("long").hidden = !whole.long || whole.longer;
  $("longer").hidden = !whole.longer;
  (document.querySelector(".report") as HTMLElement).hidden = false;
  told({ markdown: whole.markdown });
}

// T194: the screen kept on while sections run, and the seconds the page was hidden counted
export const awake = wakeKeeper({ document, navigator });
