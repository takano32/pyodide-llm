// tests/worker-source.mjs (T350): public/worker.js and its modules (public/worker/*.js) as scripts of one vm context.
//
// The checks that run the worker in Node (worker-check.mjs, worker-sink-check.mjs) reach its functions and variables as
// globals of a context: they call download() and set forwardModule. The worker's text was one script for that; it is a
// window and its modules now, which a browser links by their imports. Here each file becomes a script again: its
// imports (one form only, see below) and the word export are taken out, import.meta.url is the file's own URL, and the
// scripts run in the order of what imports what, so that every module's top level is a global of the context when the
// next one runs, as it was when all of it was one file.
//
//   runWorker(context)   runs them all; the worker's top level is the context's afterwards
//
// What this does not see, since it takes the imports out: a module that uses a name it does not import (a global here,
// a ReferenceError in a browser) and the wait for the modules at the head of worker.js. tests/worker-modules-check.mjs
// reads the names; the wait is seen by the browsers alone (tests/e2e.mjs).
// What it does see and says: an import of a name the other file does not export, a module of the folder that the worker
// does not ask for (or one it asks for that is not there), an import in another form than the two below.
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const WORKER = new URL("../public/worker.js", import.meta.url);
const FOLDER = new URL("../public/worker/", import.meta.url);
// a module reads a neighbour:   const { a, b } = await import(new URL(`x.js${new URL(import.meta.url).search}`, import.meta.url));
const NEIGHBOUR = /^const \{([^}]*)\} =\s+await import\(new URL\(`([\w-]+)\.js\$\{new URL\(import\.meta\.url\)\.search\}`, import\.meta\.url\)\);\n/gm;
// the worker asks for all of them at once, then takes each one's names:   const { a, b } = await modules.x;
const ASKED = /^const modules = Object\.fromEntries\(\[([^\]]*)\]\.map\(\(name\) =>\n  \[name, import\(new URL\(`worker\/\$\{name\}\.js\$\{self\.location\.search\}`, import\.meta\.url\)\)\]\)\);\n/m;
const TAKEN = /^const \{([^}]*)\} = await modules\.([\w-]+);\n/gm;
const namesOf = (list) => list.split(",").map((name) => name.trim()).filter(Boolean);

/** [{ name, url, source, reads: [module names] }], the modules first, each after what it reads, worker.js last */
export function workerScripts() {
  const texts = new Map(fs.readdirSync(FOLDER).filter((file) => file.endsWith(".js")).sort()
    .map((file) => [file.slice(0, -3), { url: new URL(file, FOLDER), text: fs.readFileSync(new URL(file, FOLDER), "utf8") }]));
  texts.set("worker.js", { url: WORKER, text: fs.readFileSync(WORKER, "utf8") });
  // (the second name of "export const A = 1, B = 2;" too)
  const exported = (name, module) => new RegExp(`^export ((async )?(function\\*?|const|let|class) |const [^;\\n]*, )${name}\\b`, "m").test(texts.get(module).text);
  const scripts = new Map();
  for (const [name, { url, text }] of texts) {
    const window = name === "worker.js", reads = [];
    let source = text;
    if (window) {
      const asked = ASKED.exec(source);
      if (!asked) throw new Error("worker.js does not ask for its modules in the form tests/worker-source.mjs knows");
      const list = namesOf(asked[1]).map((quoted) => JSON.parse(quoted)).sort();
      const there = [...texts.keys()].filter((module) => module !== "worker.js");
      if (String(list) !== String(there)) throw new Error(`worker.js asks for [${list}], and public/worker/ holds [${there}]`);
      source = source.replace(ASKED, "");
    }
    source = source.replace(window ? TAKEN : NEIGHBOUR, (line, names, module) => {
      if (!texts.has(module)) throw new Error(`${name} imports from ${module}.js, which is not in public/worker/`);
      for (const taken of namesOf(names)) {
        if (!exported(taken, module)) throw new Error(`${name} imports ${taken} from ${module}.js, which does not export it`);
      }
      reads.push(module);
      return "";
    });
    if (!window) source = source.replace(/^export (?=(async )?(function\*?|const|let|class) )/gm, "");
    const left = /^(import|export)\b.*$|^.*\bawait import\(.*$/m.exec(source.replace(/^ .*$/gm, ""));
    if (left) throw new Error(`${name}: an import or export tests/worker-source.mjs does not know: ${left[0]}`);
    scripts.set(name, { name, url, source: source.replaceAll("import.meta.url", JSON.stringify(url.href)), reads });
  }
  const ordered = [];
  const place = (name, path = []) => {
    if (path.includes(name)) throw new Error(`the worker's modules import in a circle: ${[...path, name].join(" -> ")}`);
    if (ordered.includes(scripts.get(name))) return;
    scripts.get(name).reads.forEach((module) => place(module, [...path, name]));
    ordered.push(scripts.get(name));
  };
  [...scripts.keys()].filter((name) => name !== "worker.js").forEach((name) => place(name));
  place("worker.js");
  return ordered;
}

/** Runs the worker's modules and then worker.js in context, each as a script: their top levels are the context's */
export function runWorker(context) {
  for (const { url, source } of workerScripts()) {
    vm.runInContext(source, context, { filename: fileURLToPath(url) });
  }
}
