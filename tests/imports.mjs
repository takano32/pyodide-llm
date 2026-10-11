// tests/imports.mjs (T367.1): how a file of the runtime takes its neighbours' names, read in either of the two forms.
// The one reader of the checks that read the imports as text (worker-source.mjs, shaders-source.mjs and the five
// *-modules-check.mjs): each asked its own regular expressions before, and all of them knew one form only.
//
// The form of a tree where each file is fetched as it is (public/): a module reads a neighbour with its own ?v=, so
// that all of a page's files are of one deployment (GitHub Pages keeps a file for ten minutes), and a window asks for
// all of its modules at once and then takes each one's names:
//
//   const { a, b } = await import(new URL(`x.js${new URL(import.meta.url).search}`, import.meta.url));
//   const modules = Object.fromEntries(["x", "y"].map((name) =>
//     [name, import(new URL(`folder/${name}.js${new URL(import.meta.url).search}`, import.meta.url))]));
//   const { a, b } = await modules.x;
//   (the ?v= is `new URL(import.meta.url).search`, or `self.location.search` where the file is a worker's first)
//
// The form of a tree whose files a bundler links (src/runtime/, T367.3): plain static imports of a neighbour's file,
//
//   import { a, b as c } from "./x.js";
//
// A tree has one form: `bundled` (tests/tree.mjs's tree.bundled) says which, and the other form in a file is refused
// (a static import in a file that is fetched as it is drops the ?v=; an import with a ?v= in a bundled file names a
// file that is not there under that name).
//
//   importsOf(text, file, { bundled })   file: the file's name under the runtime's folder ("forward/threads.js").
//     -> { program,                      the file parsed (@babel/parser, which Astro's packages bring)
//          asked,                        a window's list: { folder, names, node }, or null
//          takes,                        [{ from: the neighbour's name under the runtime's folder, names: [{ imported,
//                                        local }], node, how: "import" | "await import" | "await modules" }], in the
//                                        file's order
//          left }                        the statements that are the linking and nothing else (what a tool that reads
//                                        the files as one text leaves out), in the file's order
//   without(text, nodes, line)           text with those statements cut out (line: and the end of line after each)
//   reached(from, neighbours)            the files from `from` on, each after those it takes from: neighbours(name) ->
//                                        names. Throws where they go round (a ring of awaited imports never ends).
//
// What it refuses, wherever it stands in a file's top level: an awaited import() in another form, an `export … from`,
// a static import of something that is no file beside this one. What it does not look at: an import() that is not
// awaited at the top level (a promise kept, or one inside a function: what is fetched later).
import path from "node:path";
import { parse } from "@babel/parser";

const SEARCH = String.raw`\$\{(?:new URL\(import\.meta\.url\)\.search|self\.location\.search)\}`;
const NEIGHBOUR = new RegExp(String.raw`^await import\(new URL\(\`([\w./-]+\.js)${SEARCH}\`, import\.meta\.url\)\)$`);
const ASKED = new RegExp(String.raw`^const modules = Object\.fromEntries\(\[([^\]]*)\]\.map\(\(name\) =>\n\s*\[name, import\(new URL\(\`([\w-]+)/\$\{name\}\.js${SEARCH}\`, import\.meta\.url\)\)\]\)\);$`);
const WINDOW = /^await modules\.([\w-]+)$/;

/** a neighbour's name under the runtime's folder, from the file that names it */
const beside = (file, specifier) => path.posix.normalize(path.posix.join(path.posix.dirname(file), specifier));

export function importsOf(text, file, { bundled }) {
  if (typeof bundled !== "boolean") throw new Error("importsOf(): bundled, whether a bundler links the tree's files (tests/tree.mjs's tree.bundled)");
  const program = parse(text, { sourceType: "module" }).program;
  const takes = [], left = [];
  let asked = null;
  const fetchedAsItIs = (what) => {
    if (bundled) throw new Error(`${file}: ${what} with a ?v=, in a tree whose files a bundler links: a static import (import { … } from "./….js")`);
  };
  for (const node of program.body) {
    const source = text.slice(node.start, node.end);
    if (node.type === "ImportDeclaration") {
      const specifier = node.source.value;
      if (!/^\.\.?\/[\w./-]+\.js$/.test(specifier)) throw new Error(`${file}: an import this check does not know: ${source}`);
      if (!bundled) throw new Error(`${file}: a static import drops the ?v= (GitHub Pages keeps a file for ten minutes): await import(new URL(…))`);
      const names = node.specifiers.map((one) => {
        if (one.type !== "ImportSpecifier" || one.imported.type !== "Identifier") throw new Error(`${file}: an import this check does not know (the names, each by itself): ${source}`);
        return { imported: one.imported.name, local: one.local.name };
      });
      takes.push({ from: beside(file, specifier), names, node, how: "import" });
      left.push(node);
      continue;
    }
    if ((node.type === "ExportNamedDeclaration" || node.type === "ExportAllDeclaration") && node.source) throw new Error(`${file}: an export from another file: ${source}`);
    if (node.type !== "VariableDeclaration") continue;
    const list = ASKED.exec(source);
    if (list) {
      fetchedAsItIs("a window's list of its modules");
      asked = { folder: beside(file, list[2]), names: [...list[1].matchAll(/"([\w-]+)"/g)].map((match) => match[1]), node };
      left.push(node);
      continue;
    }
    for (const d of node.declarations) {
      if (d.init?.type !== "AwaitExpression") continue;
      const awaited = text.slice(d.init.start, d.init.end);
      const neighbour = NEIGHBOUR.exec(awaited)?.[1], ofWindow = WINDOW.exec(awaited)?.[1];
      if (!neighbour && !ofWindow) {
        if (/\bimport\(/.test(awaited)) throw new Error(`${file}: an import this check does not know: ${source}`);
        continue;  // (something else awaited at the top: no import)
      }
      fetchedAsItIs(neighbour ? "an awaited import" : "a window's module");
      if (node.declarations.length !== 1 || d.id.type !== "ObjectPattern") throw new Error(`${file}: what is taken of a neighbour is its names, taken apart: ${source}`);
      if (ofWindow && !asked) throw new Error(`${file}: \`${awaited}\` before the window's list of its modules`);
      if (ofWindow && !asked.names.includes(ofWindow)) throw new Error(`${file}: \`${awaited}\` is no module of the window's list`);
      const names = d.id.properties.map((property) => {
        if (property.type !== "ObjectProperty" || property.computed || property.key.type !== "Identifier" || property.value.type !== "Identifier") {
          throw new Error(`${file}: what is taken of a neighbour is plain names: ${source}`);
        }
        return { imported: property.key.name, local: property.value.name };
      });
      takes.push({ from: neighbour ? beside(file, neighbour) : `${asked.folder}/${ofWindow}.js`, names, node, how: neighbour ? "await import" : "await modules" });
      left.push(node);
    }
  }
  return { program, asked, takes, left };
}

/** text without the statements `nodes` (and, with `line`, without the end of line that follows each) */
export function without(text, nodes, line = false) {
  return nodes.toSorted((a, b) => b.start - a.start).reduce((now, node) => `${now.slice(0, node.start)}${now.slice(node.end + (line && now[node.end] === "\n" ? 1 : 0))}`, text);
}

/** The files reached from `from`, each after the ones it takes from (neighbours(name) -> [names]). ended: the files
 * placed already, where several are walked into one order. */
export function reached(from, neighbours, ended = []) {
  const open = [];
  (function visit(name) {
    if (open.includes(name)) throw new Error(`the modules wait for one another: ${[...open.slice(open.indexOf(name)), name].join(" → ")}`);
    if (ended.includes(name)) return;
    open.push(name);
    neighbours(name).forEach(visit);
    open.pop();
    ended.push(name);
  })(from);
  return ended;
}

/** The modules of a window: its file and every .js of the folder of its name ("gpu.js": gpu/), as { name: text },
 * read by `read(name)`, the folder listed by `list(folder)`. And that the window holds them all: in the form with a
 * list, the list is the folder's files; in the static form, every file of the folder is reached from the window. */
export function windowed(window, { read, list, bundled, more = [] }) {
  const folder = window.replace(/\.js$/, "");
  const there = list(folder).filter((name) => name.endsWith(".js")).sort().map((name) => `${folder}/${name}`);
  const files = new Map([window, ...more, ...there].map((name) => [name, { text: read(name) }]));
  for (const [name, entry] of files) Object.assign(entry, importsOf(entry.text, name, { bundled }));
  const { asked } = files.get(window);
  if (!bundled) {
    if (!asked) throw new Error(`${window} does not ask for its modules in one list, each with its own ?v=`);
    const listed = asked.names.map((name) => `${asked.folder}/${name}.js`);
    if (asked.folder !== folder || String([...listed].sort()) !== String(there)) {
      throw new Error(`${window} asks for [${asked.names}], and ${folder}/ holds [${there.map((name) => path.posix.basename(name, ".js"))}]`);
    }
    if (new Set(listed).size !== listed.length) throw new Error(`${window} asks for a module twice`);
  }
  for (const [name, { takes }] of files) for (const { from } of takes) {
    if (!files.has(from)) throw new Error(`${name} takes names from ${from}, which is no file of ${folder}/ (nor ${[window, ...more].join(", ")})`);
  }
  const order = reached(window, (name) => files.get(name).takes.map(({ from }) => from));
  const lost = there.filter((name) => !order.includes(name));
  if (lost.length) throw new Error(`${lost.join(", ")}: in ${folder}/, and neither ${window} nor a file it takes from takes anything of ${lost.length > 1 ? "them" : "it"}`);
  return { files, there, order, asked };
}
