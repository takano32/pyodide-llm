// T239, T240: public/forward.js broken on purpose, one change a copy, and whether tests/thread-search-check.mjs and tests/gpu-default-check.mjs fail on it
//   node tests/t239-mutants.mjs (CI: tests.yml extra=)
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
const root = new URL("../", import.meta.url);
const tests = ["tests/thread-search-check.mjs", "tests/gpu-default-check.mjs"];
const source = fs.readFileSync(new URL("public/forward.js", root), "utf8");
const mutants = [
  ["T239: no quarter (the search as it was)", `    if (search.direction === "down" && !search.far) search.far = true;\n    else if`, "    if"],
  ["T239: the quarter is not let go after a count wins", `{ best: candidate, moved: true, far: false }`, `{ best: candidate, moved: true }`],
  ["T239: a count that won is not said to have moved", `{ best: candidate, moved: true, far: false }`, `{ best: candidate, far: false }`],
  ["T239: an eighth in place of the quarter", `Math.floor(best / (far ? 4 : 2))`, `Math.floor(best / (far ? 8 : 2))`],
  ["T239: the quarter before the half", `Math.floor(best / (far ? 4 : 2))`, `Math.floor(best / (far ? 2 : 4))`],
  ["T239: four times as many on the way up too", `: best * 2;`, `: best * (far ? 4 : 2);`],
  ["T239: no way up after the quarter lost", `else if (search.direction === "down" && !search.moved) search.direction = "up";`, `else if (false) search.direction = "up";`],
  ["T239: the way up after a count went down", `else if (search.direction === "down" && !search.moved) search.direction = "up";`, `else if (search.direction === "down") search.direction = "up";`],
  ["T240: no search begun inside a generation", `if (unchecked && generations && mayRecheck()) beginSearch(chosen);`, ``],
  ["T240: begun while the GPU gets ready", `if (unchecked && generations && mayRecheck()) beginSearch(chosen);`, `if (unchecked && generations && !search && chosen && recheckEvery) beginSearch(chosen);`],
  ["T240: begun where the page began no generation (/benchmark/)", `if (unchecked && generations && mayRecheck()) beginSearch(chosen);`, `if (unchecked && mayRecheck()) beginSearch(chosen);`],
  ["T240: begun whether the count is owed a search or not", `if (unchecked && generations && mayRecheck()) beginSearch(chosen);`, `if (generations && mayRecheck()) beginSearch(chosen);`],
];
let passed = 0, made = 0;
for (const [i, [name, from, to]] of mutants.entries()) {
  if (!source.includes(from)) {
    console.log(`- ${name}: NOT MADE (the line is not in forward.js)`);
    continue;
  }
  const dir = new URL(`.tmp/t239/mutant${i}/`, root);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(new URL("forward.js", dir), source.replace(from, to));
  for (const file of ["jobs.js", "shaders.js"]) fs.copyFileSync(new URL(`public/${file}`, root), new URL(file, dir));
  const results = tests.map((test) => {
    const run = spawnSync("node", [test, "--forward", fileURLToPath(new URL("forward.js", dir))], { cwd: fileURLToPath(root), encoding: "utf8", timeout: 600000 });
    const out = (run.stderr + run.stdout).split("\n");
    const why = out.filter((line) => /AssertionError|^- .*not /.test(line)).map((line) => line.trim().slice(0, 220)).slice(0, 3).join(" / ");
    return `${test.slice(6)} ${run.status === 0 ? "PASSES" : `fails (${why || out.filter(Boolean).at(-1)})`}`;
  });
  console.log(`- mutant ${name}: ${results.join("; ")}`);
  passed += results.every((r) => r.includes(" PASSES"));
  made += 1;
}
// the right forward.js through the same option: both must pass
for (const test of tests) {
  const run = spawnSync("node", [test, "--forward", fileURLToPath(new URL("public/forward.js", root))], { cwd: fileURLToPath(root), encoding: "utf8", timeout: 600000 });
  console.log(`- mutant none (public/forward.js), ${test.slice(6)}: ${run.status === 0 ? "passes" : "FAILS"}`);
}
console.log(`mutants: ${made - passed} of ${made} made fail a test`);
