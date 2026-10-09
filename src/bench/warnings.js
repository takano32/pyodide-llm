// T227: what came out WRONG, failed, unsteady or skipped in a run, listed once at the report's head.
// (T353: a part of src/bench.js, which is the window that exports every name of these)
import { STATES } from "./cpu.js";

// a | that tableCell() escaped stays in its cell (T214: a skipped round's reason is the words of a cell)
const cells = (line) => line.split(/(?<!\\)\|/).slice(1, -1).map((cell) => cell.trim().replace(/\\\|/g, "|"));

// T227: the words the page writes where something did not come out right
const MARKED = /\b(?:WRONG|FAILED|failed|unsteady|skipped)\b/;
// a row or a sentence that only repeats a verdict of the GPU's check, which is listed itself
const REPEATED = /\(WRONG in the check\)|The check found [^.]*WRONG\./g;
const marked = (text) => MARKED.test(text.replace(REPEATED, ""));
// how much a warning matters (T227's review), the worst first: 0 a result that cannot be right or a step that did not
// run (WRONG, FAILED, failed), 1 a round that was skipped, 2 a time that was rough (unsteady)
const severity = (text) => {
  const words = text.replace(REPEATED, "");
  return /\b(?:WRONG|FAILED|failed)\b/.test(words) ? 0 : /\bskipped\b/.test(words) ? 1 : 2;
};

/**
 * T227: all that came out WRONG, failed, unsteady or skipped in a run, a line each, in the words the page shows (the
 * owner picked them off the screen by hand: they were spread over the report's tables, and the summary had none of
 * the reasons). Read from the Markdown the sections wrote, so that what is listed is what the page says. sections:
 * [{title, status, markdown, said}], the top of the report (the rounds, the model page's path) as one of them. Of each:
 * a section that failed as a whole (its Markdown is why: no table, nothing it names itself); said, the lines the
 * section says went wrong itself (the GPU's verdicts of its check, a lost device, the CPU's logits not finite), as they
 * are; every row of a table with one of the words in a cell, its cells under their headers; and every sentence outside
 * the tables with one.
 * What a section says itself is taken out of its Markdown, wherever it runs over a line break (a device's error message
 * has some), so that it is listed once; nor are the marks that only repeat a verdict ("WRONG in the check" beside a
 * row) listed. A failure the page writes in none of these words is not here. The worst come first, within each kind in
 * the order of the sections (severity()): a summary that has room for a few (shortReport()) keeps the WRONG ones, not
 * the rough times of the page's path that open the report. [] where nothing did.
 */
export function warnings(sections) {
  const out = [];
  for (const { title, status, markdown = "", said = [] } of sections) {
    const before = out.length, add = (text, rank = severity(text)) => out.push({ rank, text: `${title}: ${text.replace(/\s*\n\s*/g, " ")}` });
    // a section that failed and wrote only why (no table, nothing it names itself)
    if (status === "error" && !said.length && !/^\|/m.test(markdown)) {
      add(`failed: ${markdown}`, 0);
      continue;
    }
    said.forEach((text) => add(text, 0));
    let rest = markdown;
    for (const one of said) rest = rest.split(one).join("");
    const lines = rest.split("\n");
    let head = [];
    lines.forEach((line, i) => {
      if (!line.startsWith("|")) {
        for (const sentence of line.split(/(?<=\.) (?=[A-Z"])/)) if (marked(sentence)) add(sentence.replace(/^- /, ""));
      } else if (lines[i + 1]?.startsWith("|---")) head = cells(line);
      else if (marked(line)) add(cells(line).map((cell, j) => cell && `${head[j] ? `${head[j]}: ` : ""}${cell}`).filter(Boolean).join("; "));
    });
    // a section the page calls WRONG or failed with none of the above: its state at the least
    if (out.length === before && (status === "wrong" || status === "error")) add(STATES[status], 0);
  }
  // (a sort keeps the order of equals: the sections' own)
  return [...new Set(out.sort((a, b) => a.rank - b.rank).map(({ text }) => text))];
}

/** T227: the warnings as the report holds them, under a heading of their own ("" where there are none); kept: how
 * many of them a summary has room for (shortReport()), the rest counted in a last line. */
export function warningsBlock(warned, kept = warned.length) {
  if (!warned.length) return "";
  const rest = warned.length - kept;
  const more = rest ? [`- ${kept ? `… and ${rest} more` : `${rest} of them`}, in the whole report below`] : [];
  return ["#### Warnings", "", ...warned.slice(0, kept).map((line) => `- ${line}`), ...more].join("\n");
}

export { cells };
