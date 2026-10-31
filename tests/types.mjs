// tests/types.mjs (T.368.1): the type check of what a browser runs, in the light suite and as `npm run types`.
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
// Two rules tsc does not hold:
//   - a type's import is `import type { X } from …`, never `import { type X } from …` with nothing else in the braces.
//     Node runs a .ts file by blanking its types, so the second form is left as an import of the module (it is loaded
//     and run, for nothing: T.378.5's note), where a bundler drops it. With `import type` the two read the same.
//   - (T.368.8) a name that a file of the runtime uses and neither declares nor imports is one of AMBIENT, the names of
//     a worker that these files mean to use. tsc says a name that is nowhere (TS2304: a forgotten import, a misspelling;
//     the six *-modules-check.mjs each said that with a list of their own until T.368.8). It does not say one that
//     its library declares: `name` is a string in every worker, so a line moved out of the loop that bound its `name`
//     reads the worker's, and types the same. Nor one that a file without an import or an export declares at its top:
//     such a file is a script to tsc, and its names are every file's (benchmark/reader.ts's COUNT and TOTAL, which
//     forward/threads.ts imports from jobs.ts: left out of that import, tsc reads the reader's). A new ambient name is
//     added to the list where it is first used; one that no file uses any more is taken off (said here).
//     The pages' scripts have the same rule with a page's names in tests/page-modules-check.mjs (the .astro scripts
//     are in no program of tsc's), and src/bench/'s JavaScript in tests/bench-modules-check.mjs.
//
// The count: every `any` that is written (and every @ts-expect-error, @ts-ignore), by file. An `any` is counted where
// it stands in a type (what blanking the types removes), so one in a comment or a string is not.
// T.473: no more of them than ANY_MOST (what T.368 ended with). One more fails; where there are fewer, the line says to
// lower ANY_MOST to what is there, so that what was won is kept.
// And what else tells tsc to look away from a value, counted and held to nothing (T.476, T.477 and T.479 are to bring
// them down): the `!`s (a value said to be there: x!, and a `let x!: T`) and the casts (x as T, but for `as const`).
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parse } from "@babel/parser";
import { namesOf } from "./names.mjs";
import { typed, untyped } from "./untyped.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
export const PROGRAMS = ["tsconfig.page.json", "tsconfig.worker.json", "tsconfig.sw.json"];
// T.473: the most times `any` may be written in the TypeScript sources
export const ANY_MOST = 46;

// what a worker (and the page's main thread, for the two modules it runs too: kept.ts, updated.ts) has without
// declaring it, as far as the runtime's files use it
export const AMBIENT = new Set(("AbortController Array ArrayBuffer Atomics BigInt BigInt64Array Boolean Date Error FileSystemFileHandle Float32Array Float64Array " +
  "Infinity Int32Array Int8Array JSON Map Math MessageChannel NaN Number Object PerformanceObserver Promise ReadableStream Response Set SharedArrayBuffer String Symbol " +
  "TextEncoder TransformStream URL URLSearchParams Uint16Array Uint32Array Uint8Array WeakMap WebAssembly Worker WritableStream clearInterval clearTimeout console crypto " +
  "atob encodeURIComponent fetch globalThis navigator onmessage performance postMessage self setInterval setTimeout undefined").split(" "));

/** the names a text uses and neither declares nor imports (its types blanked first: a type's name is no name used) */
export const ambient = (text, name) => [...namesOf(untyped(text, name)).free];

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

/** how many times a TypeScript text says that a value is there (x!, let x!: T) and that it is of a type (x as T) */
export function lookedAway(text) {
  let sure = 0, cast = 0;
  (function visit(node) {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) return node.forEach(visit);
    if (node.type === "TSNonNullExpression" || (node.type === "VariableDeclarator" && node.definite)) sure += 1;
    if (node.type === "TSAsExpression" && !(node.typeAnnotation.type === "TSTypeReference" && node.typeAnnotation.typeName.name === "const")) cast += 1;
    for (const [key, value] of Object.entries(node)) if (key !== "loc" && !key.endsWith("Comments")) visit(value);
  })(parse(text, { sourceType: "module", plugins: ["typescript"] }).program);
  return { sure, cast };
}

/** what the count of the anys says of the most there may be: { failed, line } (line: nothing to say when it is the most) */
export function capped(written, most = ANY_MOST) {
  if (written > most) return { failed: true, line: `any is written ${written} times, and ${most} is the most (tests/types.mjs's ANY_MOST): say the type, or unknown and narrow it` };
  if (written < most) return { failed: false, line: `any is written ${written} times, fewer than the ${most} allowed: lower tests/types.mjs's ANY_MOST to ${written}` };
  return { failed: false, line: "" };
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
  const files = sources(), counts = files.map((file) => {
    const text = fs.readFileSync(path.join(root, file), "utf8");
    return { file, ...anys(text, file), ...lookedAway(text) };
  });
  for (const file of files) for (const statement of typeOnlyImports(fs.readFileSync(path.join(root, file), "utf8"))) {
    failed = true;
    console.log(`types: ${file}: ${statement}\n    names types alone: write it \`import type { … }\` (Node would load the module for nothing, a bundler would not)`);
  }
  const used = new Set();
  for (const file of sources("src/runtime")) {
    const free = ambient(fs.readFileSync(path.join(root, file), "utf8"), file), unknown = free.filter((name) => !AMBIENT.has(name));
    free.forEach((name) => used.add(name));
    if (!unknown.length) continue;
    failed = true;
    console.log(`types: ${file} uses ${unknown.join(", ")}, which it neither declares nor imports, and which is no ambient name the runtime uses (tests/types.mjs's AMBIENT):\n` +
      "    a forgotten import that tsc reads as its library's name or as a script's; or a new ambient name, to add to AMBIENT");
  }
  const unused = [...AMBIENT].filter((name) => !used.has(name));
  if (unused.length) { failed = true; console.log(`types: AMBIENT lists ${unused.join(", ")}, which no file of src/runtime uses any more: take ${unused.length > 1 ? "them" : "it"} off`); }
  console.log(`types: the runtime's ${sources("src/runtime").length} files use ${used.size} ambient names, each one of AMBIENT's ${AMBIENT.size}`);
  const total = (key) => counts.reduce((sum, count) => sum + count[key], 0);
  console.log(`types: ${files.length} TypeScript sources; any written ${total("any")} times (${counts.filter((count) => count.any).map((count) => `${count.file.replace(/^src\//, "")} ${count.any}`).join(", ") || "nowhere"}); ` +
    `${total("off")} checks switched off by a comment; ${total("sure")} values said to be there (!) and ${total("cast")} casts`);
  const cap = capped(total("any"));
  if (cap.failed) failed = true;
  if (cap.line) console.log(`types: ${cap.line}`);
  console.log(failed ? "types: FAILED" : "types: ok");
  process.exit(failed ? 1 : 0);
}
