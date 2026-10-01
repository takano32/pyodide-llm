// E5: parseReport() and reportsTable() on the bodies an issue can hold with the Warnings block
import { parseReport, reportBody, reportsTable, warnings, warningsBlock, benchMarkdown, pathTable, MEMORY_UNSAID } from "../../src/bench.js";
import * as F from "./fixtures.mjs";

const { t225Whole, t225Summary, aWhole, android, rows, real } = F;
const answered = (body) => body.replace("**Device**: (e.g. Pixel 8, MacBook Air M2, a desktop with a Ryzen 7 7700)", "**Device**: Xiaomi 13T Pro")
  .replace("**OS**: (e.g. Android 16, macOS 26, Windows 11)", "**OS**: Android 15").replace("**Browser**: (e.g. Chrome 148, Safari 26, Firefox 150)", "**Browser**: Chrome 148");
const bodies = {
  "before T227 (no block), the whole": reportBody(aWhole),
  "the whole with the block": reportBody(t225Whole),
  "the summary alone": reportBody(t225Summary),
  "the summary, then the whole pasted below it": `${reportBody(t225Summary)}\n\n${t225Whole}`,
  "the summary, a blank, then the whole pasted with its own questions": `${reportBody(t225Summary)}\n\n${reportBody(t225Whole)}`,
  "the whole, the user wrote under the questions": answered(reportBody(t225Whole)),
};
const base = parseReport(bodies["before T227 (no block), the whole"]);
for (const [label, body] of Object.entries(bodies)) {
  const report = parseReport(body);
  const same = JSON.stringify({ ...report, device: "", os: "", browser: "" }) === JSON.stringify({ ...base, device: "", os: "", browser: "" });
  console.log(`${label}: ${report ? `${report.rows.length} rows, model ${JSON.stringify(report.model)}, cores ${report.cores}` : "undefined"}; same as before the change: ${same}`);
}
console.log(reportsTable(Object.entries(bodies).map(([label, body], i) => ({ number: i + 1, url: "u", body: answered(body) }))));

// a report from a browser that skips the NumPy round (Safari, Firefox): the skipped row is a warning, and reportsTable() reads it as before
const head = [benchMarkdown([rows[0], { name: "without the kernels", without: ["kernels"], skip: MEMORY_UNSAID }], android), pathTable(real, "llm-jp-3 150M")].join("\n\n");
const warned = warnings([{ title: "Model", markdown: head }]);
const whole = [head, warningsBlock(warned), "#### CPU", "x"].join("\n\n");
console.log(warned.map((w) => w.slice(0, 100)));
console.log(JSON.stringify(parseReport(reportBody(whole)).rows.map((r) => ({ name: r.name, speed: r.speed, skip: r.skip?.slice(0, 30) }))));
