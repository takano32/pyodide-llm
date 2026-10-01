// mutants of public/forward.js's thread search for tests/thread-search-check.mjs: each must fail it
import fs from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";

const root = new URL("../../", import.meta.url).pathname;
const out = path.join(root, ".tmp/t223/mut");
fs.mkdirSync(out, { recursive: true });
fs.copyFileSync(path.join(root, "public/jobs.js"), path.join(out, "jobs.js"));
const source = fs.readFileSync(path.join(root, "public/forward.js"), "utf8");
const mutants = [
  ["no-margin", "const KEEP = 5, TIMED = 2, BETTER = 0.95;", "const KEEP = 5, TIMED = 2, BETTER = 1.0;"],
  ["margin-20", "const KEEP = 5, TIMED = 2, BETTER = 0.95;", "const KEEP = 5, TIMED = 2, BETTER = 0.80;"],
  ["upper-median-of-the-times", "const bestMs = lowerMedian(search.times[best]), candidateMs = lowerMedian(search.times[candidate]);",
    "const upper = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1]; const bestMs = upper(search.times[best]), candidateMs = upper(search.times[candidate]);"],
  ["no-first-check-of-a-remembered-count", "        chosen = remembered;\n        unchecked = true;  // T223: searched again from it on this visit", "        chosen = remembered;"],
  ["a-step-of-one", "const candidate = direction === \"down\" ? Math.floor(best / 2) : best * 2;", "const candidate = direction === \"down\" ? best - 1 : best + 1;"],
  ["always-down", "direction: from > 1 ? \"down\" : \"up\"", "direction: \"down\""],
  ["no-up-after-down", "    if (search.direction === \"down\" && !search.moved) {\n      search.direction = \"up\";\n      return nextCandidate();\n    }\n", ""],
  ["best-blocks-first", "const order = [search.best, search.candidate, search.candidate, search.best];", "const order = [search.best, search.best, search.candidate, search.candidate];"],
  ["one-block-each", "if (search.step < 4 * (BLOCK + 1)) return;", "if (search.step < 2 * (BLOCK + 1)) return;"],
  ["first-token-of-a-block-counted", "return [order[block], inBlock > 0];", "return [order[block], true];"],
];
const results = [];
for (const [name, find, replace] of mutants) {
  const count = source.split(find).length - 1;
  if (count !== 1) { console.log(`${name}: the text to change is there ${count} times`); continue; }
  const file = path.join(out, `forward-${name}.js`);
  fs.writeFileSync(file, source.replace(find, replace));
  let verdict;
  try {
    execFileSync("node", [path.join(root, "tests/thread-search-check.mjs"), "--forward", file], { cwd: root, stdio: ["ignore", "pipe", "pipe"], timeout: 120000 });
    verdict = "PASSED (not caught)";
  } catch (error) {
    const text = (error.stderr?.toString() ?? "") + (error.stdout?.toString() ?? "");
    const line = text.split("\n").find((l) => /AssertionError|Error:/.test(l)) ?? text.split("\n")[0];
    verdict = `fails: ${line.replace(/^.*?(AssertionError \[ERR_ASSERTION\]: |Error: )/, "").slice(0, 170)}`;
  }
  console.log(`${name.padEnd(40)} ${verdict}`);
  results.push([name, verdict]);
}
console.log(`\n${results.filter(([, v]) => v.startsWith("fails")).length} of ${results.length} mutants fail the check`);
