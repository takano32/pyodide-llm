// tests/dist-same.ts (T.368.1): whether two builds are the same code. The proof of a step that only adds types (a file
// of the runtime made TypeScript, T.368): what a browser is given is what it was given before, so nothing of its speed or
// its behaviour is asked again.
//
//   node tests/dist-same.ts <a build> <another>      each a dist/ (everything in it but models/) or a dist/_astro/
//
// Two builds of the same code differ in two ways that are no difference, and these alone are set aside:
//   - a file's name carries a hash of its text (_astro/<name>.<hash>.<ext>), and a file that names another holds that
//     name: where one file's text differs at all, its name and every file that names it differ. Each build's own file
//     names are read as <name>.*.<ext>, wherever they are written;
//   - a bundler's "//#region <source>" and "//#endregion" lines (the workers are not minified), which say a source's
//     path: src/runtime/jobs.js became src/runtime/jobs.ts.
// Everything else is held to the byte: a comment that moved, a parenthesis a cast left behind, a file more or less.
// A file is paired with the other build's by its name without the hash. For each: "the same" (to the byte, name and
// all), "the same code" (but for the two things above), or "DIFFERENT" with the first line that differs; and the files
// only one build has. Ends 1 unless every file is one of the first two.
//
// What it does not see: the models (a dist/'s models/ is `make models`' and no build's: a tree built for the comparison
// has none), and whether either build works: tests/dist-check.ts is for that. Given two dist/, the pages and what is
// served as it is (coi.js) are compared too; given two _astro/, those folders alone.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const HASHED = /^(.+)\.([\w-]{8})\.(\w+)$/;  // (Vite's and Astro's hashes are eight characters)
const TEXT = /\.(js|mjs|css|html|json|txt|svg|map)$/;
const REGION = /^\s*\/\/#(end)?region\b.*$/;

/** a file's name as both builds spell it: without its hash */
export const unhashed = (name) => name.replace(HASHED, "$1.*.$3");

/** what a build holds: { root, files: [paths from the root], names: the hashed names of its _astro/ } */
function read(folder) {
  const root = path.resolve(folder);
  if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) throw new Error(`${folder} is no folder`);
  const inside = fs.existsSync(path.join(root, "_astro"));
  const files = fs.readdirSync(root, { recursive: true }).map((name) => name.split(path.sep).join("/"))
    .filter((file) => !(inside && file.startsWith("models/")) && fs.statSync(path.join(root, file)).isFile());
  return { root, files: files.sort(), names: fs.readdirSync(inside ? path.join(root, "_astro") : root).filter((name) => HASHED.test(name)) };
}

/** a text file as the comparison reads it: the build's own hashed names without their hashes, no region lines */
function plain(text, names) {
  let out = text;
  // (longest first: a name that ends another's is not cut out of it)
  for (const name of [...names].sort((a, b) => b.length - a.length)) out = out.split(name).join(unhashed(name));
  return out.split("\n").filter((line) => !REGION.test(line)).join("\n");
}

/** The two builds, file by file: [{ file, verdict: "same" | "same code" | "different" | "only before" | "only after",
 * line?, before?, after? }] in the order of the names. */
export function compared(before, after) {
  const [a, b] = [read(before), read(after)];
  const byName = (build) => {
    const map = new Map();
    for (const file of build.files) {
      const key = file.split("/").map(unhashed).join("/");
      if (map.has(key)) throw new Error(`${build.root}: ${map.get(key)} and ${file} are one name without their hashes`);
      map.set(key, file);
    }
    return map;
  };
  const [mapA, mapB] = [byName(a), byName(b)];
  const rows = [];
  for (const key of [...new Set([...mapA.keys(), ...mapB.keys()])].sort()) {
    const [fileA, fileB] = [mapA.get(key), mapB.get(key)];
    if (!fileA || !fileB) { rows.push({ file: key, verdict: fileA ? "only before" : "only after", named: fileA ?? fileB }); continue; }
    const [bytesA, bytesB] = [fs.readFileSync(path.join(a.root, fileA)), fs.readFileSync(path.join(b.root, fileB))];
    if (fileA === fileB && bytesA.equals(bytesB)) { rows.push({ file: key, verdict: "same" }); continue; }
    if (!TEXT.test(key)) {
      // a built file (a kernel, an archive): its bytes, and nothing is set aside but its name
      rows.push(bytesA.equals(bytesB) ? { file: key, verdict: "same code" } : { file: key, verdict: "different", line: 0, before: `${bytesA.length} bytes`, after: `${bytesB.length} bytes` });
      continue;
    }
    const [textA, textB] = [plain(bytesA.toString("utf8"), a.names), plain(bytesB.toString("utf8"), b.names)];
    if (textA === textB) { rows.push({ file: key, verdict: "same code" }); continue; }
    const [linesA, linesB] = [textA.split("\n"), textB.split("\n")];
    let at = 0;
    while (at < linesA.length && at < linesB.length && linesA[at] === linesB[at]) at++;
    rows.push({ file: key, verdict: "different", line: at + 1, before: around(linesA[at], linesB[at]), after: around(linesB[at], linesA[at]) });
  }
  return rows;
}

/** a line where it first differs from the other (a minified file is one line of a hundred thousand characters) */
function around(line, other) {
  if (line === undefined) return "(the file ends)";
  let at = 0;
  while (other !== undefined && at < line.length && at < other.length && line[at] === other[at]) at++;
  const from = Math.max(0, at - 60);
  return `${from ? "…" : ""}${line.slice(from, at + 100)}${at + 100 < line.length ? "…" : ""}`;
}

/** the report's lines, and whether the two builds are the same code */
export function report(rows) {
  const count = (verdict) => rows.filter((row) => row.verdict === verdict).length;
  const lines = [];
  for (const row of rows) {
    if (row.verdict === "different") {
      lines.push(`DIFFERENT  ${row.file}${row.line ? `, from line ${row.line} (without the region lines)` : ""}`, `    before: ${row.before}`, `    after:  ${row.after}`);
    } else if (row.verdict.startsWith("only")) lines.push(`${row.verdict.toUpperCase()}  ${row.named}`);
  }
  const same = count("different") + count("only before") + count("only after") === 0;
  lines.push(`dist-same: ${rows.length} files: ${count("same")} the same to the byte, ${count("same code")} the same code under another name or with other region lines, ` +
    `${count("different")} different, ${count("only before")} only in the first, ${count("only after")} only in the second`);
  lines.push(same ? "dist-same: the same code" : "dist-same: NOT the same");
  return { lines, same };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [before, after] = process.argv.slice(2);
  if (!before || !after || process.argv.length > 4) {
    console.error("node tests/dist-same.ts <a build: dist/ or dist/_astro/> <another>");
    process.exit(2);
  }
  let outcome;
  try { outcome = report(compared(before, after)); } catch (error) { console.error(`dist-same: ${error.message}`); process.exit(2); }
  console.log(outcome.lines.join("\n"));
  process.exitCode = outcome.same ? 0 : 1;
}
