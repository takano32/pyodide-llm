// tests/dist-check.mjs (T367.3.1): what a build wrote, held to what the Service Worker does with a file's name.
//
//   node tests/dist-check.mjs [<dist>] [--tree <the tree it was built of>]      after `npm run build` (dist/, this tree)
//
// public/coi.js keeps a copy of this site's files for a visit without a network (T111), and drops the old copy of a file
// when it keeps a new one: two addresses are "the same file" where family() gives them one name, and a file under
// /_astro/ is <name>.<hash>.<ext> to it. A bundler's names are not the Service Worker's business until they disagree
// with that, and then nothing fails where anyone looks (T365 measured both):
//   - a name family() does not take apart is a family of its own for ever: every deployment leaves one more copy
//     (Vite names a worker's files <name>-<hash>.js unless it is told otherwise: 27 files);
//   - two files of one build in one family put each other's copy out: one is kept, the other's copy is deleted, and
//     the page without a network finds a file missing (two entries called gpu.js, one of the model page and one of
//     /benchmark/; a module that the page and a worker both ask for).
// So, with coi.js's own family() (its text, run here: not a second copy of the rule):
//   1. every file of dist/_astro/ has a family that is not its own name;
//   2. no two files of dist/_astro/ have the same family;
// and what a bundler does without a word where a URL is written the wrong way:
//   3. no file of dist/_astro/ is a source of the runtime as it is (a URL with a variable in it copies its folder), and
//      no worker's text holds a file as a data: URL;
//   4. every built file the runtime names (src/runtime/built/) is in dist/_astro/ once, with the same bytes;
//   5. dist's root holds the pages, _astro/ and what is served as it is (the Service Worker and its test page, the
//      models), and nothing else: no source, no built file, no archive that nobody asks for.
// 3 to 5 are of a tree whose runtime a bundler links (tests/tree.mjs's tree.bundled). In a tree where each file is
// served as it is from the same folder as the Service Worker, the root is those files, and this says so and checks 1
// and 2 alone.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { pathToFileURL } from "node:url";
import { treeOf } from "./tree.mjs";

const ORIGIN = "https://site.test", BASE = "/pyodide-llm/";
/** what dist's root may hold: the pages, the bundler's folder, and what is served as it is */
export const ROOT = ["_astro", "index.html", "benchmark", "coi.js", "coi-test", "models"];

/** coi.js's own family(), from its text: the function as the Service Worker runs it, on this origin */
export function familyOf(coi) {
  const text = /^function family\(address\) \{\n[\s\S]*?\n\}$/m.exec(coi)?.[0];
  if (!text) throw new Error("coi.js has no family(address) at its top level: what says which copies are versions of one file?");
  return vm.runInNewContext(`(${text})`, { URL, self: { location: { origin: ORIGIN } } });
}

/** What is wrong with a build, as lines (none: nothing). files: the names in its _astro/; family: coi.js's. */
export function families(files, family) {
  const said = [], groups = new Map();
  for (const name of files) {
    const address = `${ORIGIN}${BASE}_astro/${name}`, f = family(address);
    if (f === address) said.push(`_astro/${name}: the Service Worker takes no name and hash apart in it (<name>.<hash>.<ext>): its copy is never dropped for a newer one`);
    groups.set(f, [...(groups.get(f) ?? []), name]);
  }
  for (const [f, names] of groups) {
    if (names.length > 1) said.push(`${names.map((name) => `_astro/${name}`).join(" and ")}: one family to the Service Worker (${f.slice(ORIGIN.length + BASE.length)}): keeping one drops the other's copy`);
  }
  return said;
}

/** every file under a folder, as its path from it with "/" between */
const under = (folder) => (!fs.existsSync(folder) ? [] : fs.readdirSync(folder, { recursive: true, withFileTypes: true }).filter((entry) => entry.isFile())
  .map((entry) => path.relative(folder, path.join(entry.parentPath, entry.name)).split(path.sep).join("/")).sort());

/** What is wrong with dist (a folder) as a build of tree, as lines; and what was looked at, in words. */
export function distSaid(dist, tree = treeOf()) {
  const astro = path.join(dist, "_astro");
  if (!fs.existsSync(astro)) throw new Error(`${dist}: no _astro/ in it: npm run build`);
  const files = fs.readdirSync(astro).sort();
  const said = families(files, familyOf(fs.readFileSync(tree.served("coi.js"), "utf8")));
  if (!tree.bundled) return { said, files, looked: `${files.length} files of _astro/, each a family of its own (the runtime is served as it is from ${tree.folders.runtime}/: the root is not looked at)` };
  // 3: a source as it is, a file as a data: URL
  const bytes = new Map(files.map((name) => [name, fs.readFileSync(path.join(astro, name))]));
  const sources = under(tree.runtime()).filter((name) => name.endsWith(".js") && !name.startsWith(`${path.relative(tree.runtime(), tree.built())}/`));
  for (const source of sources) {
    const text = fs.readFileSync(tree.runtime(source));
    for (const [name, got] of bytes) if (got.equals(text)) said.push(`_astro/${name} is ${tree.folders.runtime}/${source} as it is, not bundled: a URL made with a variable copies its whole folder (new URL("./literal", import.meta.url))`);
  }
  for (const [name, got] of bytes) {
    if (name.endsWith(".js") && /["'`]data:application\/(wasm|zip|octet-stream)/.test(got.toString("latin1"))) said.push(`_astro/${name} holds a file as a data: URL (vite.build.assetsInlineLimit: 0)`);
  }
  // 4: the built files, once each, the same bytes
  const built = under(tree.built());
  for (const file of built) {
    const { name: stem, ext } = path.parse(file);
    const there = files.filter((name) => name.startsWith(`${stem}.`) && name.endsWith(ext) && /^[\w-]{6,}$/.test(name.slice(stem.length + 1, name.length - ext.length)));
    if (there.length !== 1) said.push(`${tree.folders.built}/${file}: ${there.length ? `${there.length} files of _astro/ (${there.join(", ")})` : "no file of _astro/"}: every built file is named once, by its literal name`);
    else if (!bytes.get(there[0]).equals(fs.readFileSync(tree.built(file)))) said.push(`_astro/${there[0]} is not ${tree.folders.built}/${file} byte for byte: an old build of one of them`);
  }
  // 5: the root
  for (const name of fs.readdirSync(dist).sort()) {
    if (!ROOT.includes(name)) said.push(`${path.basename(dist)}/${name}: nothing asks for it there (the root holds ${ROOT.join(", ")}): a file of ${tree.folders.served}/ that is no longer served as it is?`);
  }
  for (const name of ["index.html", "benchmark/index.html", "coi.js", "_astro"]) if (!fs.existsSync(path.join(dist, name))) said.push(`${path.basename(dist)}/${name} is not there`);
  return { said, files, looked: `${files.length} files of _astro/, each a family of its own; ${built.length} built files in it once each; none of ${sources.length} sources as it is; the root holds nothing else` };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const at = args.indexOf("--tree");
  const tree = at < 0 ? treeOf() : treeOf(args.splice(at, 2)[1]);
  const dist = path.resolve(args[0] ?? path.join(tree.root, "dist"));
  const { said, looked } = distSaid(dist, tree);
  assert.deepEqual(said, [], `dist-check: ${said.length} thing${said.length > 1 ? "s" : ""} wrong with ${dist}:\n${said.join("\n")}\n`);
  console.log(`dist-check: ok (${looked})`);
}
