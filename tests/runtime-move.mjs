// tests/runtime-move.mjs (T367.3.1): the tool that takes the runtime's JavaScript from the folder it is served from as
// it is (public/) to the one a bundler links it from (src/runtime/), and rewrites how each file takes its neighbours.
// T365's throwaway codemod.mjs made again for the tree as it is now, on tests/imports.mjs's reader instead of regular
// expressions of its own, and committed: T367.3.2's commit of the mechanical rewrite is this tool's output and nothing
// else, so that whoever reads that commit can make it again and compare. It goes when that is landed (no tree is left
// to apply it to).
//
//   node tests/runtime-move.mjs move [--root <a tree>]       the files, as they are, to src/runtime/ (and what is built
//                                                            beside them to src/runtime/built/): renames only
//   node tests/runtime-move.mjs rewrite [--root <a tree>]    the linking of every file of src/runtime/, in place
//   node tests/runtime-move.mjs left [--root <a tree>]       what a hand has to finish: the lines that still carry a
//                                                            ?v=, a URL made from a variable, or the queue's remains
//
// What moves: every .js under the served folder but the Service Worker's own (STAYS: it is registered at a fixed URL of
// the site's root, and its test page with it). What is built (BUILT: the kernels and the Python's archives) moves when
// it is there; a tree where nothing is built has nothing to move.
//
// What `rewrite` does to a file, and nothing else (rewritten(), which tests/runtime-move-check.mjs holds to made-up
// texts and to this tree):
//   1. a neighbour's names awaited at the top level, and a window's module's, become a static import where they stood:
//        const { a, b: c } = await import(new URL(`x.js${…search}`, import.meta.url));   ->  import { a, b as c } from "./x.js";
//        const { a } = await modules.x;                                                   ->  import { a } from "./folder/x.js";
//      (the names keep their line breaks; the statement keeps its place, so the comment above it still stands above it)
//   2. the window's list of its modules goes (`const modules = Object.fromEntries([…].map(…));`): a module of the list
//      that the window takes no name of is reached through the others, or tests/imports.mjs's windowed() says it is lost
//   3. the queue of the messages that came while the modules were fetched goes (T350, T352, T353: a static import is
//      linked before the file's first statement): `const early = [];`, the first `onmessage = (event) => early.push(event);`
//      and the line that hands them on at the end
//   4. a module asked for later (a promise kept, or inside a function) keeps being asked for later, by its name:
//        import(new URL(`x.js${…search}`, import.meta.url))   ->  import("./x.js")
//   5. a worker is named by a literal URL:
//        new Worker(new URL(`x.js${…search}`, import.meta.url), { type: "module" })   ->  new Worker(new URL("./x.js", import.meta.url), { type: "module" })
//   6. benchmark/sections.js's at("x.js") (`../x.js` with the ?v=), where it names a module or a worker, likewise
// A line it does not know how to rewrite is left as it is and listed by `left`: the hand edits are those lines.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { importsOf } from "./imports.mjs";

export const TO = "src/runtime", BUILT_TO = "src/runtime/built";
/** what stays where it is served from: the Service Worker (a fixed URL of the site's root) and its test page */
export const STAYS = (name) => name === "coi.js" || name.startsWith("coi-test/");
/** what `make kernels` and python_archive.py write beside the runtime */
export const BUILT = (name) => !name.includes("/") && /^(simdkernel|ceilings)[\w]*\.(wasm|so|wasmlib)$|^(engine|converter)\.zip$/.test(name);

const SEARCH = String.raw`\$\{(?:new URL\(import\.meta\.url\)\.search|self\.location\.search|search)\}`;
const LATER = new RegExp(String.raw`\bimport\(new URL\(\`([\w./-]+\.js)${SEARCH}\`, import\.meta\.url\)\)`, "g");
const WORKER = new RegExp(String.raw`\bnew Worker\(new URL\(\`([\w./-]+\.js)${SEARCH}\`, import\.meta\.url\), \{ type: "module" \}\)`, "g");
const AT = /^const at = \(name\) => new URL\(`\.\.\/\$\{name\}\$\{search\}`, import\.meta\.url\);$/m;
const QUEUE = [/^const early = \[\];\n/m, /^(?:self\.)?onmessage = \(event\) => early\.push\(event\);\n/m,
  /^for \(const event of early\.splice\(0\)\) (?:self\.)?onmessage\(event\);\n/m, /^early\.splice\(0\)\.forEach\(\(event\) => (?:self\.)?onmessage\(event\)\);\n/m];
const relative = (specifier) => (specifier.startsWith(".") ? specifier : `./${specifier}`);

/** One file's text with its linking rewritten (the six rules at the head). file: its name under the runtime's folder. */
export function rewritten(text, file) {
  const counts = { static: 0, list: 0, queue: 0, later: 0, worker: 0 };
  // (a file that takes a neighbour by a static import is rewritten already: read as that, which refuses a ?v= beside it)
  const already = /^import\s[^;]*?\sfrom\s+"\.\.?\//m.test(text);
  const { asked, takes } = already ? { asked: null, takes: (importsOf(text, file, { bundled: true }), []) } : importsOf(text, file, { bundled: false });
  const edits = takes.map(({ from, node }) => {
    const pattern = node.declarations[0].id;
    const names = text.slice(pattern.start, pattern.end).replace(/(\w+): (\w+)/g, "$1 as $2");
    counts.static += 1;
    return { start: node.start, end: node.end, to: `import ${names} from "${relative(path.posix.relative(path.posix.dirname(file), from))}";` };
  });
  if (asked) {
    counts.list += 1;
    edits.push({ start: asked.node.start, end: asked.node.end + (text[asked.node.end] === "\n" ? 1 : 0), to: "" });
  }
  let now = edits.toSorted((a, b) => b.start - a.start).reduce((all, { start, end, to }) => all.slice(0, start) + to + all.slice(end), text);
  if (asked && /\bearly\b/.test(now)) {
    for (const line of QUEUE) now = now.replace(line, () => { counts.queue += 1; return ""; });
  }
  now = now.replace(LATER, (call, name) => { counts.later += 1; return `import("${relative(name)}")`; });
  now = now.replace(WORKER, (call, name) => { counts.worker += 1; return `new Worker(new URL("${relative(name)}", import.meta.url), { type: "module" })`; });
  if (AT.test(now)) {
    now = now.replace(/\bimport\(at\("([\w./-]+\.js)"\)\)/g, (call, name) => { counts.later += 1; return `import("../${name}")`; });
    now = now.replace(/\bnew Worker\(at\("([\w./-]+\.js)"\), \{ type: "module" \}\)/g, (call, name) => { counts.worker += 1; return `new Worker(new URL("../${name}", import.meta.url), { type: "module" })`; });
  }
  return { text: now, counts };
}

/** The lines of a rewritten file that a hand has to finish: code (no comment) that still reads a ?v=, makes a URL from
 * something other than one literal name, imports by a URL that is no literal, or holds a remain of the queue. */
export function left(text) {
  const literal = /new URL\("\.\.?\/[\w./-]+", import\.meta\.url\)/g;
  // (the code of a line: a comment at its end is words, like a line that is all comment)
  const code = (line) => line.replace(/\s\/\/ .*$/, "");
  return text.split("\n").map((line, at) => ({ line: line.trim(), at: at + 1 })).filter(({ line: whole }) => !whole.startsWith("//") && !whole.startsWith("*") && ((line) => (
    /location\.search|import\.meta\.url\)\.search|\$\{search\}/.test(line)                  // a ?v= read
    || /import\.meta\.url/.test(line.replace(literal, ""))                                    // a URL that is no one literal name
    || /\bimport\((?!["']|\/\* @vite-ignore \*\/)/.test(line) || /\bat\((?:name|")/.test(line)  // a module or a file by a variable
    || /\bearly\b/.test(line)))(code(whole)));                                                // the queue's remains
}

/** every file under a folder, as its path from it with "/" between */
function under(folder) {
  return fs.readdirSync(folder, { recursive: true, withFileTypes: true }).filter((entry) => entry.isFile())
    .map((entry) => path.relative(folder, path.join(entry.parentPath, entry.name)).split(path.sep).join("/")).sort();
}

/** [from, to] of every file `move` renames, as paths from the tree's root. served: the folder the runtime is in now. */
export function moves(root, served) {
  return under(path.join(root, served)).flatMap((name) => (
    name.endsWith(".js") && !STAYS(name) && !name.startsWith("models/") ? [[`${served}/${name}`, `${TO}/${name}`]]
      : BUILT(name) ? [[`${served}/${name}`, `${BUILT_TO}/${name}`]] : []));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const verb = args.shift();
  const root = path.resolve(args[0] === "--root" ? args[1] : fileURLToPath(new URL("..", import.meta.url)));
  if (!["move", "rewrite", "left"].includes(verb)) {
    console.error("node tests/runtime-move.mjs move|rewrite|left [--root <a tree>]");
    process.exit(2);
  }
  const to = path.join(root, TO);
  if (verb === "move") {
    // (the folder is read off the tree: where forward.js is, tests/tree.mjs's way, without its cache of this process)
    const { treeOf } = await import("./tree.mjs");
    const served = treeOf(root).folders.runtime;
    if (served === TO) throw new Error(`${root}: the runtime is in ${TO}/ already`);
    const all = moves(root, served);
    for (const [from, target] of all) {
      fs.mkdirSync(path.dirname(path.join(root, target)), { recursive: true });
      fs.renameSync(path.join(root, from), path.join(root, target));
    }
    // (the folders that the move emptied go: an empty folder of what is served as it is would be a folder of the site)
    for (const folder of [...new Set(all.flatMap(([from]) => { const parts = path.dirname(from).split("/"); return parts.map((_, i) => parts.slice(0, i + 1).join("/")); }))].sort().reverse()) {
      if (folder !== served && fs.existsSync(path.join(root, folder)) && !fs.readdirSync(path.join(root, folder)).length) fs.rmdirSync(path.join(root, folder));
    }
    const built = all.filter(([, target]) => target.startsWith(`${BUILT_TO}/`)).length;
    console.log(`runtime-move: ${all.length - built} files of ${served}/ to ${TO}/, ${built} built files to ${BUILT_TO}/`);
  } else {
    const files = under(to).filter((name) => name.endsWith(".js") && !name.startsWith("built/"));
    const total = { static: 0, list: 0, queue: 0, later: 0, worker: 0 }, lines = [];
    for (const file of files) {
      const before = fs.readFileSync(path.join(to, file), "utf8");
      if (verb === "left") { lines.push(...left(before).map(({ line, at }) => `${file}:${at}: ${line.slice(0, 200)}`)); continue; }
      const { text, counts } = rewritten(before, file);
      for (const key of Object.keys(total)) total[key] += counts[key];
      if (text !== before) fs.writeFileSync(path.join(to, file), text);
    }
    if (verb === "left") console.log(`${lines.join("\n")}\nruntime-move: ${lines.length} lines in ${new Set(lines.map((line) => line.split(":")[0])).size} files for a hand`);
    else console.log(`runtime-move: ${files.length} files: ${total.static} static imports, ${total.list} windows' lists and ${total.queue} lines of their queues gone, ${total.later} modules asked for later, ${total.worker} workers`);
  }
}
