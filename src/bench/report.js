// The report: the issue's body and link, how long a link may be, the short report, and reading reports back (T91).
// (T353: a part of src/bench.js, which is the window that exports every name of these)
import { number } from "./cells.js";
import { cells, warningsBlock } from "./warnings.js";

// T91: where a visitor sends the result, and the one table the results make.
export const REPOSITORY = "takano32/pyodide-llm";
/** The questions of .github/ISSUE_TEMPLATE/benchmark.md that the page cannot answer. The answer goes after the colon;
 * what is in parentheses is an example, and is not an answer when it is left there. */
export const QUESTIONS = [
  ["Device", "e.g. Pixel 8, MacBook Air M2, a desktop with a Ryzen 7 7700"],
  ["OS", "e.g. Android 16, macOS 26, Windows 11"],
  ["Browser", "e.g. Chrome 148, Safari 26, Firefox 150"],
];

/** The body of a benchmark issue: the three questions, then the page's Markdown as it is. */
export function reportBody(markdown) {
  return [...QUESTIONS.map(([name, example]) => `**${name}**: (${example})`), "",
          "<!-- the page's Markdown, as the page wrote it: please leave it as it is -->", markdown].join("\n");
}

/** The longest address of GitHub's login a new issue may be sent through. A visitor who is not signed in is sent to
 * https://github.com/login?return_to=<the new issue's address, encoded once more: every % becomes %25>, and from
 * about 7,700 characters of that GitHub drops return_to without a word: the visitor signs in, lands on the dashboard,
 * and the report is gone (the second review of T134, 2026-09-26, by curl: an address of 4,485 characters of a
 * report's table was kept, one of 4,935 dropped; the 302 that "6,799 went through" once read was that drop). GitHub
 * also answers 500 from about 7,000 characters of the address itself, and 414 to anyone from about 8,100. A report
 * whose login address would pass this goes to the clipboard, and the issue asks for it to be pasted. */
export const REPORT_LIMIT = 7000;
export const TOO_LONG = "The results were too long for the link and are on your clipboard: please paste them here.";
/** T185: the last line of a summary (shortReport()), which the link holds where the whole report is too long */
export const PASTE = "The whole report is on your clipboard: please paste it below.";

const issueUrl = (body, environment) => `https://github.com/${REPOSITORY}/issues/new?${new URLSearchParams(
  { template: "benchmark.md", title: `Benchmark: ${environment.model ?? "this device"}`, body: reportBody(body) })}`;

const throughLogin = (url) => `https://github.com/login?return_to=${encodeURIComponent(url)}`;

/** Whether the results are too long for the address of a new issue (reportUrl() then leaves them out). */
export const reportTooLong = (markdown, environment) => throughLogin(issueUrl(markdown, environment)).length > REPORT_LIMIT;

/** The address of a new issue with the template, the title and the body filled in: the page's Markdown; where it is
 * too long, the summary (shortReport(): T185, a GPU's report is about four times the limit), which asks for the whole
 * from the clipboard; where that is too long as well, or there is none (the model page's), that request alone. */
export function reportUrl(markdown, environment, summary) {
  const body = !reportTooLong(markdown, environment) ? markdown
    : summary && !reportTooLong(summary, environment) ? summary : TOO_LONG;
  return issueUrl(body, environment);
}
/** reportUrl() as GitHub's login gets it (tests/bench.mjs) */
export const loginUrl = (markdown, environment, summary) => throughLogin(reportUrl(markdown, environment, summary));

/** T185: the report the link holds where the whole is too long: the top of the report as it is (the machine's line and
 * the rounds' table, which parseReport() reads, and T184's table of the model page's path), a line for each section,
 * and the request to paste the whole below. T227: the warnings (warnings()) between the top and the lines, as in the
 * whole report; where they make the link too long (environment: what the link is made with), the last of them are
 * left to the whole report, and a line says how many. */
export function shortReport(head, lines, warned = [], environment) {
  const summary = (kept) => [head, warningsBlock(warned, kept), "#### Summary", lines.map((line) => `- ${line}`).join("\n"), PASTE].filter(Boolean).join("\n\n");
  let kept = warned.length;
  while (kept && environment && reportTooLong(summary(kept), environment)) kept--;
  return summary(kept);
}

/** What one issue says: the answers, the model, the logical cores, and the rows of its table. undefined when the body
 * has no table of the page's. */
export function parseReport(body) {
  const lines = body.replace(/\r/g, "").split("\n");
  const answer = (name) => {
    const line = lines.find((text) => text.startsWith(`**${name}**:`));
    const value = line?.slice(`**${name}**:`.length).trim() ?? "";
    return /^\(e\.g\./.test(value) ? "" : value;  // the example left in place is no answer
  };
  const head = lines.findIndex((line) => line.startsWith("| what ran |"));
  if (head < 0) return undefined;
  const rows = [];
  for (const line of lines.slice(head + 2)) {
    if (!line.startsWith("|")) break;
    const [name, speed, ready, backend] = cells(line);
    // T214: a skipped round has no speed, and its last column is why
    rows.push(speed === "skipped" ? { name, speed: NaN, ready, backend: "", skip: backend } : { name, speed: Number(speed), ready, backend });
  }
  const machine = lines.find((line) => line.startsWith("**") && line.includes(" · ") && !line.includes("**:")) ?? "";
  return { device: answer("Device"), os: answer("OS"), browser: answer("Browser"),
           model: /^\*\*(.+?)\*\*/.exec(machine)?.[1] ?? "", cores: /(\d+) logical cores|(\d+) threads/.exec(machine)?.slice(1).find(Boolean) ?? "",
           rows };
}

/** The table of the visitors' reports (T83's 30-measurements.md): one row per issue, from its "everything" round. */
export function reportsTable(issues) {
  const out = ["| device | OS | browser | model | logical cores | tok/s | without the kernels | backend | report |",
               "|---|---|---|---|---:|---:|---:|---|---|"];
  for (const { number: id, url, body } of issues) {
    const report = parseReport(body ?? "");
    if (!report) continue;
    const row = (name) => report.rows.find((r) => r.name === name);
    const all = row("everything") ?? report.rows.at(-1), plain = row("without the kernels") ?? row("NumPy only");
    out.push(`| ${report.device || "?"} | ${report.os || "?"} | ${report.browser || "?"} | ${report.model || "?"} | ` +
             `${report.cores || "?"} | ${number(all?.speed)} | ${plain?.skip !== undefined ? "skipped" : number(plain?.speed)} | ${all?.backend ?? ""} | [#${id}](${url}) |`);
  }
  return out.join("\n");
}
