// tests/imports.ts (T.367.1): how a file of the runtime takes its neighbours' names. The one reader of the checks that
// read the imports as text (worker-source.ts, shaders-source.ts and the *-modules-check.mjs of forward, the two GPU
// workers and, through those two, the worker and the shaders): each asked its own regular expressions before.
//
// The runtime's files are linked by a bundler (src/runtime/, T.367.3): plain static imports of a neighbour's file,
//
//   import { a, b as c } from "./x.js";      (T.368.1: or "./x.ts", a neighbour that is TypeScript)
//
// T.368.1: a file that is TypeScript is read with its types blanked (tests/untyped.ts: the same length, the same
// lines), so every tool that reads through here reads JavaScript, and a node's place is its place in the source. An
// `import type` is blanked with the rest: it links nothing.
//
// T.587: the form of before T.367, where each file was fetched as it is from public/ and read a neighbour by an awaited
// import with its own ?v=, is read no more (the oldest tree the tools compare with, f91d83f9, is bundled): such an
// import is one this check does not know.
//
//   importsOf(text, file)                file: the file's name under the runtime's folder ("forward/threads.js").
//     -> { program,                      the file parsed (@babel/parser, which Astro's packages bring)
//          takes,                        [{ from: the neighbour's name under the runtime's folder, names: [{ imported,
//                                        local }], node }], in the file's order
//          left }                        the statements that are the linking and nothing else (what a tool that reads
//                                        the files as one text leaves out), in the file's order
//   without(text, nodes, line)           text with those statements cut out (line: and the end of line after each)
//   reached(from, neighbours)            the files from `from` on, each after those it takes from: neighbours(name) ->
//                                        names. Throws where they go round.
//
// What it refuses, wherever it stands in a file's top level: an awaited import(), an `export … from`, a static import
// of something that is no file beside this one. What it does not look at: an import() that is not awaited at the top
// level (a promise kept, or one inside a function: what is fetched later).
import path from "node:path";
import { parse } from "@babel/parser";
import { typed, untyped } from "./untyped.ts";

/** a neighbour's name under the runtime's folder, from the file that names it */
const beside = (file, specifier) => path.posix.normalize(path.posix.join(path.posix.dirname(file), specifier));

export function importsOf(text, file) {
  // (T.368.1: a .ts file is parsed with its types blanked: a node's place is its place in the source, tests/untyped.ts)
  const program = parse(untyped(text, file), { sourceType: "module" }).program;
  const takes = [], left = [];
  for (const node of program.body) {
    const source = text.slice(node.start, node.end);
    if (node.type === "ImportDeclaration") {
      const specifier = node.source.value;
      if (!/^\.\.?\/[\w./-]+\.(js|ts)$/.test(specifier) || /\.d\.ts$/.test(specifier)) throw new Error(`${file}: an import this check does not know: ${source}`);
      const names = node.specifiers.map((one) => {
        if (one.type !== "ImportSpecifier" || one.imported.type !== "Identifier") throw new Error(`${file}: an import this check does not know (the names, each by itself): ${source}`);
        return { imported: one.imported.name, local: one.local.name };
      });
      takes.push({ from: beside(file, specifier), names, node });
      left.push(node);
      continue;
    }
    if ((node.type === "ExportNamedDeclaration" || node.type === "ExportAllDeclaration") && node.source) throw new Error(`${file}: an export from another file: ${source}`);
    if (node.type !== "VariableDeclaration") continue;
    for (const d of node.declarations) {
      if (d.init?.type !== "AwaitExpression") continue;
      // (something else awaited at the top is no import; an awaited import() is the form of before T.367, or none)
      if (/\bimport\(/.test(text.slice(d.init.start, d.init.end))) throw new Error(`${file}: an import this check does not know: ${source}`);
    }
  }
  return { program, takes, left };
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

/** The modules of a window: its file and every .js or .ts of the folder of its name ("gpu.js": gpu/), as { name: text },
 * read by `read(name)`, the folder listed by `list(folder)`. And that the window holds them all: every file of the
 * folder is reached from the window. others: with `outside`, the files outside the folder that were read because a
 * module takes names of them. */
export function windowed(window, { read, list, more = [], outside = false }) {
  const folder = window.replace(/\.(js|ts)$/, "");
  const there = list(folder).filter((name) => name.endsWith(".js") || typed(name)).sort().map((name) => `${folder}/${name}`);
  // (T.368.1: each file's text is JavaScript here, a .ts file's with its types blanked)
  const files = new Map([window, ...more, ...there].map((name) => [name, { text: untyped(read(name), name) }]));
  for (const [name, entry] of files) Object.assign(entry, importsOf(entry.text, name));
  // outside: a file that is no module of the folder and that one of them takes names of is read too, and what it takes
  // from in turn (T.367.3: the table of the built files, which a worker's modules and the benchmark's read alike). Without
  // it such a file has to be named in `more`: a module that reaches out of its folder is said, not followed.
  const others = [];
  for (const [name, { takes }] of files) for (const { from } of takes) {
    if (files.has(from)) continue;
    if (!outside) throw new Error(`${name} takes names from ${from}, which is no file of ${folder}/ (nor ${[window, ...more].join(", ")})`);
    const text = untyped(read(from), from);
    files.set(from, { text, ...importsOf(text, from) });  // (a Map takes what is added while it is walked)
    others.push(from);
  }
  const order = reached(window, (name) => files.get(name).takes.map(({ from }) => from));
  const lost = there.filter((name) => !order.includes(name));
  if (lost.length) throw new Error(`${lost.join(", ")}: in ${folder}/, and neither ${window} nor a file it takes from takes anything of ${lost.length > 1 ? "them" : "it"}`);
  return { files, there, order, others };
}
