// T239 (a throwaway branch): copies of public/forward.js, written to .tmp/t239/<form>/; node tests/t239-variants.mjs, then
// node tests/thread-search-check.mjs --table --forward .tmp/t239/<today|quarter|both|all>/forward.js
// copies of public/forward.js with the search in each form considered, for thread-search-check.mjs --table --forward
import fs from "node:fs";
const root = new URL("../", import.meta.url);
const source = fs.readFileSync(new URL("public/forward.js", root), "utf8");
const swap = (text, from, to) => {
  if (!text.includes(from)) throw new Error(`not found: ${from}`);
  return text.replace(from, to);
};
const FAR = `    if (search.direction === "down" && !search.far) search.far = true;\n    else if`;
const CANDIDATE = `const candidate = direction === "down" ? Math.floor(best / (far ? 4 : 2)) : best * 2;\n    if (candidate < 1) return passed();`;
const LIMIT = [`recheckEvery = recheck;`, `recheckEvery = recheck;\n      limit = Math.max(from, remembered);`];
const DECLARE = [`const BLOCK = 4;`, `const BLOCK = 4;\n  let limit = Infinity;`];
const forms = {
  today: (s) => swap(s, FAR, "    if"),
  quarter: (s) => s,
  both: (s) => [[FAR, `    if (!search.far) search.far = true;\n    else if`],
    [CANDIDATE, `const candidate = direction === "down" ? Math.floor(best / (far ? 4 : 2)) : best * (far ? 4 : 2);\n    if (candidate < 1 || (far && direction === "up" && candidate > limit)) return passed();`],
    [`search.direction = "up";`, `Object.assign(search, { direction: "up", far: false });`],
    LIMIT, DECLARE].reduce((t, [a, b]) => swap(t, a, b), s),
  all: (s) => [[CANDIDATE, `if (!search.list) {\n      search.list = [];\n      for (let n = Math.floor(best / 2); n >= 1; n = Math.floor(n / 2)) search.list.push(n);\n      for (let n = best * 2; n <= limit; n *= 2) search.list.push(n);\n    }\n    const candidate = search.list.shift() ?? 0;\n    if (candidate < 1) return finish();`],
    [FAR + ` (search.direction === "down" && !search.moved) search.direction = "up";\n    else return finish();\n    nextCandidate();`, `    nextCandidate();`],
    LIMIT, DECLARE].reduce((t, [a, b]) => swap(t, a, b), s),
};
for (const [name, form] of Object.entries(forms)) {
  fs.mkdirSync(new URL(`.tmp/t239/${name}/`, root), { recursive: true });
  fs.writeFileSync(new URL(`.tmp/t239/${name}/forward.js`, root), form(source));
  fs.copyFileSync(new URL("public/jobs.js", root), new URL(`.tmp/t239/${name}/jobs.js`, root));
}
console.log(Object.keys(forms).join(" "));
