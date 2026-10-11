// tests/worker-source.mjs (T350): src/runtime/worker.ts and its modules (src/runtime/worker/*.ts) as scripts of one vm context
// (T368.3: TypeScript, read with its types blanked, tests/untyped.mjs: the same lines, so a vm's error names the source's line).
//
// The checks that run the worker in Node (worker-check.mjs, worker-sink-check.mjs) reach its functions and variables as
// globals of a context: they call download() and set forwardModule. The worker's text was one script for that; it is a
// window and its modules now, which a browser links by their imports. Here each file becomes a script again: its
// imports (in either form a tree has: tests/imports.mjs) and the word export are taken out, import.meta.url is the file's own URL, and the
// scripts run in the order of what imports what, so that every module's top level is a global of the context when the
// next one runs, as it was when all of it was one file.
//
//   runWorker(context)   runs them all; the worker's top level is the context's afterwards
//
// What this does not see, since it takes the imports out: a module that uses a name it does not import (a global here,
// a ReferenceError in a browser) and the wait for the modules at the head of worker.js. tests/worker-modules-check.mjs
// reads the names; the wait is seen by the browsers alone (tests/e2e.mjs).
// What it does see and says: an import of a name the other file does not export, a module of the folder that the worker
// does not ask for (or one it asks for that is not there), an import in a form tests/imports.mjs does not know.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { reached, windowed, without } from "./imports.mjs";
import { treeOf } from "./tree.mjs";

/** [{ name, url, source, reads: [module names] }], the modules first, each after what it reads, the window last.
 * tree: tests/tree.mjs's (this tree's by default): where the runtime is, and which form its imports have. */
export function workerScripts(tree = treeOf()) {
  // (T368.3: the window by the name the tree has it under, worker.ts or worker.js: windowed() blanks a .ts file's types
  // by its name, and lists the folder's files of either ending)
  const WINDOW = path.basename(tree.runtime("worker.js"));
  const { files, there, others } = windowed(WINDOW, { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), bundled: tree.bundled, outside: true });
  // (the second name of "export const A = 1, B = 2;" and of "export let a, b;" too)
  const exported = (name, file) => new RegExp(`^export ((async )?(function\\*?|const|let|class) |(const|let) [^;\\n]*, )${name}\\b`, "m").test(files.get(file).text);
  // (a module by its name without the folder and the ending, as the checks that read these scripts know it; the
  // window by its file's name)
  // (a file outside worker/ that a module takes names of, T367.3's built.js: by its own name without the ending)
  const short = (file) => (file === WINDOW ? file : file.replace(/^worker\//, "").replace(/\.(js|ts)$/, ""));
  const scripts = new Map();
  for (const file of [...others, ...there, WINDOW]) {
    const { text, takes, left } = files.get(file), name = short(file), window = file === WINDOW;
    for (const { from, names } of takes) for (const { imported } of names) {
      if (!exported(imported, from)) throw new Error(`${name} imports ${imported} from ${from}, which does not export it`);
    }
    let source = without(text, left, true);
    if (!window) source = source.replace(/^export (?=(async )?(function\*?|const|let|class) )/gm, "");
    const unknown = /^(import|export)\b.*$|^.*\bawait import\(.*$/m.exec(source.replace(/^ .*$/gm, ""));
    if (unknown) throw new Error(`${name}: an import or export tests/worker-source.mjs does not know: ${unknown[0]}`);
    const url = tree.runtimeUrl(file);
    scripts.set(file, { name, url, source: source.replaceAll("import.meta.url", JSON.stringify(url.href)), reads: takes.map(({ from }) => short(from)) });
  }
  const ordered = [];
  try {
    for (const file of [...others, ...there, WINDOW]) reached(file, (name) => files.get(name).takes.map(({ from }) => from), ordered);
  } catch (error) {
    throw new Error(`the worker's modules import in a circle: ${error.message}`);
  }
  return ordered.map((file) => scripts.get(file));
}

/** Runs the worker's modules and then worker.js in context, each as a script: their top levels are the context's */
export function runWorker(context, tree = treeOf()) {
  for (const { url, source } of workerScripts(tree)) {
    vm.runInContext(source, context, { filename: fileURLToPath(url) });
  }
}
