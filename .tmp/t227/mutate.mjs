// Mutations of T227's code in src/bench.js, each run against a copy of tests/bench.mjs (the sandbox in .tmp/t227/mut)
import { spawnSync } from "node:child_process";
import fs from "node:fs";

const root = new URL("./mut/", import.meta.url).pathname;
const original = fs.readFileSync(new URL("../../src/bench.js", import.meta.url), "utf8");
const only = process.argv[2];
const M = [
  ["REPEATED not stripped", 'const marked = (text) => MARKED.test(text.replace(REPEATED, ""));', "const marked = (text) => MARKED.test(text);"],
  ["no dedupe", "return [...new Set(out.sort((a, b) => a.rank - b.rank).map(({ text }) => text))];", "return out.sort((a, b) => a.rank - b.rank).map(({ text }) => text);"],
  ["summary never cut", "while (kept && environment && reportTooLong(summary(kept), environment)) kept--;", ";"],
  ["failed section shortcut off", 'if (status === "error" && !said.length && !/^\\|/m.test(markdown)) {', "if (false) {"],
  ["said not listed", "said.forEach((text) => add(text, 0));", ""],
  ["no fallback state", 'if (out.length === before && (status === "wrong" || status === "error")) add(STATES[status], 0);', ""],
  ["no sentence split", 'for (const sentence of line.split(/(?<=\\.) (?=[A-Z"])/))', "for (const sentence of [line])"],
  ["no header detection", 'else if (lines[i + 1]?.startsWith("|---")) head = cells(line);', "else if (false) head = cells(line);"],
  ["said read again from the Markdown", "for (const one of said) rest = rest.split(one).join(\"\");", ""],
  ["unsteady not a mark", "WRONG|FAILED|failed|unsteady|skipped", "WRONG|FAILED|failed|skipped"],
  ["skipped not a mark", "WRONG|FAILED|failed|unsteady|skipped", "WRONG|FAILED|failed|unsteady"],
  ["failed not a mark", "WRONG|FAILED|failed|unsteady|skipped", "WRONG|FAILED|unsteady|skipped"],
  ["WRONG not a mark", "WRONG|FAILED|failed|unsteady|skipped", "FAILED|failed|unsteady|skipped"],
  ["no count line", 'const more = rest ? [`- ${kept ? `… and ${rest} more` : `${rest} of them`}, in the whole report below`] : [];', "const more = [];"],
  ["block keeps all", "...warned.slice(0, kept).map((line) => `- ${line}`)", "...warned.map((line) => `- ${line}`)"],
  ["unmeasured says no failed", "(error ? `failed: ${error}` : \"not measured\")", '(error ? `${error}` : "not measured")'],
  ["summary ignores the warnings", '[head, warningsBlock(warned, kept), "#### Summary"', '[head, "", "#### Summary"'],
  ["line breaks kept in a warning", '${text.replace(/\\s*\\n\\s*/g, " ")}', "${text}"],
  ["no header prefix in a row", "`${head[j] ? `${head[j]}: ` : \"\"}${cell}`", "`${cell}`"],
  ["a warning not titled", "text: `${title}: ${text.replace", "text: `${text.replace"],
  ["the warnings after the lines", '[head, warningsBlock(warned, kept), "#### Summary", lines.map((line) => `- ${line}`).join("\\n"), PASTE]', '[head, "#### Summary", lines.map((line) => `- ${line}`).join("\\n"), warningsBlock(warned, kept), PASTE]'],
  ["pipe unescape off", '.replace(/\\\\\\|/g, "|")', ""],
  ["checkVerdict WRONG as ok", 'const verdict = `${name} ${v.ok ? "ok" : "WRONG"}`;', 'const verdict = `${name} ok`;'],
  ["FAILED verdict as WRONG", "FAILED (${tableCell(v.error)})", "WRONG (${tableCell(v.error)})"],
  ["no severity sort", ".sort((a, b) => a.rank - b.rank)", ""],
  ["severity: unsteady worst", "? 0 : /\\bskipped\\b/.test(words) ? 1 : 2", "? 2 : /\\bskipped\\b/.test(words) ? 1 : 0"],
  ["severity: skipped before failed", "return /\\b(?:WRONG|FAILED|failed)\\b/.test(words) ? 0 : /\\bskipped\\b/.test(words) ? 1 : 2;", "return /\\bskipped\\b/.test(words) ? 0 : /\\b(?:WRONG|FAILED|failed)\\b/.test(words) ? 1 : 2;"],
  ["a section's own said sorted last", "said.forEach((text) => add(text, 0));", "said.forEach((text) => add(text, 2));"],
  ["a failed section sorted last", "add(`failed: ${markdown}`, 0);", "add(`failed: ${markdown}`, 2);"],
  ["the fallback state sorted last", "add(STATES[status], 0);", "add(STATES[status], 2);"],
  ["pathWarnings says nothing", "if (!paths || paths.error) return [];", "return [];"],
  ["pathWarnings lists no shared memory", "how.alone !== NO_SHARED_MEMORY && how.alone", "how.alone"],
  ["pathWarnings leaves the GPU out", "gpu.why === undefined && gpu.lost && GPU_STOPPED(gpu), ", ""],
  ["pathWarnings leaves unfinished out", "how.unfinished && NOT_ENDED(how), ", ""],
  ["pathWarnings leaves stopped out", ", how.stopped && STOPPED_WHILE_TIMED]", "]"],
  ["the CPU ceilings said as before", "tableCell(unmeasured(c.error))", "tableCell(`Not measured: ${c.error}`)"],
];
let caught = 0;
for (const [name, find, replace] of M) {
  if (only && !name.includes(only)) continue;
  if (!original.includes(find)) {
    console.log(`SKIP ${name}: the text to change is not there`);
    continue;
  }
  fs.writeFileSync(`${root}src/bench.js`, original.replace(find, replace));
  const run = spawnSync("node", ["tests/bench.mjs"], { cwd: root, encoding: "utf8", timeout: 60000 });
  const fell = run.status !== 0;
  caught += fell;
  const why = (run.stderr.split("\n").find((line) => /AssertionError|Error/.test(line)) ?? "").slice(0, 110);
  console.log(`${fell ? "caught " : "SURVIVED"} ${name}${fell ? `  (${why})` : ""}`);
}
fs.writeFileSync(`${root}src/bench.js`, original);
console.log(`${caught} of ${M.length} caught`);
