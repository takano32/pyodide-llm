// tests/types.mjs (T368.1): the type check of what a browser runs, in the light suite and as `npm run types`.
//
//   node tests/types.mjs            the three programs, then the rule and the count below; ends 1 on any error
//
// Three programs, because a page's globals and a worker's cannot be in one (tsconfig.base.json has what they share:
// strict, no output, only syntax that is erased, a type's import said as one):
//   tsconfig.page.json     the pages' scripts (src/page/, src/benchmark/) and what they import
//   tsconfig.worker.json   the runtime's TypeScript (src/runtime/**/*.ts) and what it imports, with WebGPU's types
//   tsconfig.sw.json       public/coi.js as the JavaScript it is (// @ts-check), implicit any allowed
// A .js file that a .ts file imports is read for its shapes and not checked (checkJs is off): a file is checked from
// the day it is made .ts.
//
// The rule tsc does not hold: a type's import is `import type { X } from …`, never `import { type X } from …` with
// nothing else in the braces. Node runs a .ts file by blanking its types, so the second form is left as an import of
// the module (it is loaded and run, for nothing: T378.5's note), where a bundler drops it. With `import type` the two
// read the same.
//
// The count: every `any` that is written (and every @ts-expect-error, @ts-ignore), by file. Each step of T368 reports
// it; none is refused here. An `any` is counted where it stands in a type (what blanking the types removes), so one
// in a comment or a string is not.
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { typed, untyped } from "./untyped.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
export const PROGRAMS = ["tsconfig.page.json", "tsconfig.worker.json", "tsconfig.sw.json"];

/** the imports of a text that name types alone and are not `import type`: the statements, as written */
export function typeOnlyImports(text) {
  const found = [];
  for (const [statement, names] of text.matchAll(/^import\s*\{([^}]*)\}\s*from\s*["'][^"']+["'];?/gm)) {
    const each = names.split(",").map((name) => name.trim()).filter(Boolean);
    if (each.length && each.every((name) => /^type\s/.test(name))) found.push(statement.replace(/\s+/g, " "));
  }
  return found;
}

/** how many times a TypeScript text says any in a type, and how many checks it switches off by a comment */
export function anys(text, name) {
  const plain = untyped(text, name);
  let written = 0;
  for (const match of text.matchAll(/\bany\b/g)) if (plain.slice(match.index, match.index + 3) !== "any") written += 1;
  return { any: written, off: (text.match(/@ts-(expect-error|ignore|nocheck)\b/g) ?? []).length };
}

/** every TypeScript source under a folder of the tree (no declaration file), as paths from the root */
export function sources(folder = "src", tree = root) {
  return fs.readdirSync(path.join(tree, folder), { recursive: true }).map((name) => `${folder}/${name.split(path.sep).join("/")}`).filter(typed).sort();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  let failed = false;
  const tsc = path.join(root, "node_modules/typescript/bin/tsc");
  if (!fs.existsSync(tsc)) { console.error("types: no node_modules/typescript here: npm ci"); process.exit(1); }
  for (const program of PROGRAMS) {
    const began = performance.now();
    const run = spawnSync(process.execPath, [tsc, "--noEmit", "-p", path.join(root, program)], { encoding: "utf8", cwd: root });
    const out = `${run.stdout}${run.stderr}`.trim();
    const errors = (out.match(/\berror TS\d+/g) ?? []).length;
    if (run.status !== 0) { failed = true; console.log(out); }
    console.log(`types: ${program}: ${run.status === 0 ? "ok" : `${errors} errors`} (${((performance.now() - began) / 1000).toFixed(1)} s)`);
  }
  const files = sources(), counts = files.map((file) => ({ file, ...anys(fs.readFileSync(path.join(root, file), "utf8"), file) }));
  for (const file of files) for (const statement of typeOnlyImports(fs.readFileSync(path.join(root, file), "utf8"))) {
    failed = true;
    console.log(`types: ${file}: ${statement}\n    names types alone: write it \`import type { … }\` (Node would load the module for nothing, a bundler would not)`);
  }
  const total = (key) => counts.reduce((sum, count) => sum + count[key], 0);
  console.log(`types: ${files.length} TypeScript sources; any written ${total("any")} times (${counts.filter((count) => count.any).map((count) => `${count.file.replace(/^src\//, "")} ${count.any}`).join(", ") || "nowhere"}); ` +
    `${total("off")} checks switched off by a comment`);
  console.log(failed ? "types: FAILED" : "types: ok");
  process.exit(failed ? 1 : 0);
}
