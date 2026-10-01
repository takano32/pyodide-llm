// Breakings of the page's wiring of the warnings: does tests/bench.mjs (the copy in mut/) notice?
import { spawnSync } from "node:child_process";
import fs from "node:fs";

const root = new URL("./mut/", import.meta.url).pathname;
const page = new URL("../../src/pages/benchmark.astro", import.meta.url);
const original = fs.readFileSync(page, "utf8");
fs.copyFileSync(new URL("../../tests/bench.mjs", import.meta.url), `${root}tests/bench.mjs`);
fs.copyFileSync(new URL("../../src/bench.js", import.meta.url), `${root}src/bench.js`);
const M = [
  ["the CPU section drops said", 'markdown: lines.join("\\n"), said };', 'markdown: lines.join("\\n") };'],
  ["the GPU section does not spread said", "...gpuMarkdown(steps, bridge, lost) };", "markdown: gpuMarkdown(steps, bridge, lost).markdown };"],
  ["the refresh writes a Markdown that is an object", "Object.assign(g, gpuMarkdown(g.data.steps, g.data.bridge, g.data.lost));", "g.markdown = gpuMarkdown(g.data.steps, g.data.bridge, g.data.lost).markdown;"],
  ["gpuMarkdown returns no said", 'return { markdown: lines.join("\\n"), said };', 'return { markdown: lines.join("\\n"), said: [] };'],
  ["the verdicts are not said", "said.push(...verdicts.filter(([, v]: any) => v.error || !v.ok).map(checkVerdict));", ""],
  ["the model section hands nothing over", ", said: pathWarnings(paths) };", " };"],
  ["the head's said is not handed on", "markdown: head, said: measured?.said }", "markdown: head }"],
  ["the summary is not cut to the link", "warned, environment);", "warned);"],
];
let caught = 0;
for (const [name, find, replace] of M) {
  if (!original.includes(find)) {
    console.log(`SKIP ${name}: the text to change is not there`);
    continue;
  }
  fs.mkdirSync(`${root}src/pages`, { recursive: true });
  fs.writeFileSync(`${root}src/pages/benchmark.astro`, original.replace(find, replace));
  const run = spawnSync("node", ["tests/bench.mjs"], { cwd: root, encoding: "utf8", timeout: 60000 });
  const fell = run.status !== 0;
  caught += fell;
  console.log(`${fell ? "caught " : "SURVIVED"} ${name}${fell ? `  (${(run.stderr.split("\n").find((l) => /AssertionError/.test(l)) ?? "").slice(0, 100)})` : ""}`);
}
fs.writeFileSync(`${root}src/pages/benchmark.astro`, original);
console.log(`${caught} of ${M.length} caught`);
