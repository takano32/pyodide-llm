// The JSON lines tests/e2e.mjs appends (E2E_RESULTS) as one Markdown table, for the summary of a CI job (T82).
// Every row says under which conditions its number was measured: the browser and its version, and the backend
// line of the page (which kernels ran). A number whose conditions are unclear is worth nothing.
//
//   node tests/summary.mjs results.jsonl >> "$GITHUB_STEP_SUMMARY"
import fs from "node:fs";

const cell = (value) => String(value ?? "").replace(/\|/g, "\\|").replace(/\n/g, " ");
const number = (value, digits = 1) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "");

/** The lines of a results file, the broken ones left out (a run killed half way may leave half a line). */
export function readResults(text) {
  return text.split("\n").filter(Boolean).flatMap((line) => {
    try {
      return [JSON.parse(line)];
    } catch {
      return [];
    }
  });
}

/** One table of all the runs, then a count; the failures say why. */
export function summaryMarkdown(results) {
  if (!results.length) return "No results were recorded.\n";
  const rows = results.map((r) => {
    const outcome = r.timedOut ? "**timed out**" : r.ok ? "ok" : "**failed**";
    const why = r.ok ? "" : (r.failures ?? []).join("; ");
    return `| ${cell(r.model)} | ${cell(`${r.engine} ${r.browserVersion ?? ""}`.trim())} | ${number(r.readySeconds)} | ` +
      `${number(r.tokPerSecond)} | ${cell(r.backend)} | ${outcome} | ${cell(why)} |`;
  });
  const failed = results.filter((r) => !r.ok).length;
  const system = [...new Set(results.map((r) => r.os).filter(Boolean))].join(", ");
  return [
    `| model | browser | ready (s) | tok/s | backend | result | why |`,
    `|---|---|---|---|---|---|---|`,
    ...rows,
    "",
    `${results.length - failed} of ${results.length} ran${system ? ` on ${system}` : ""}.` +
      (failed ? ` ${failed} failed or timed out: the artifacts of this job hold their screenshots, DOM and console.` : ""),
    "",
  ].join("\n");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const file = process.argv[2];
  const text = file && fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
  process.stdout.write(summaryMarkdown(readResults(text)));
}
