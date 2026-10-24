// tests/page-modules-check.mjs (T.355): what the type check does not hold of the model page's script, in
// src/pages/index.astro and in its modules (src/page/*.ts); and (T.353) the same of /benchmark/'s script, in
// src/pages/benchmark.astro and src/benchmark/*.ts. Node only, under a second:
//
//   node tests/page-modules-check.mjs
//
// T.368.8: the modules are TypeScript, and tsc (tests/types.mjs) says a name of theirs that is declared nowhere and an
// imported name that is assigned (TS2632; in a page's own script, which tsc does not read, the build refuses the
// assignment). What is left here is what tsc, the build and Node all let pass, each with why:
//   1. every name a file uses is declared in it, imported by it, or a global of a page listed below. The script inside
//      a .astro page is in no program of tsc's, and the build bundles a name that is nowhere without a word (a
//      ReferenceError in the browser, at the moment the line runs; nothing opens the page outside CI's browsers). And
//      in the modules tsc takes for the page's own a name that a page has (prompt, open, status, name, …): a forgotten
//      import of dom.ts's prompt is window.prompt to it;
//   2. the imports go round nowhere (the build links a ring without a word), and the script reaches every file of the
//      folder;
//   3. (T.378.5) what the parts hold has owners: the modules that own the page's state (OWNERS, below) each declare
//      their own `export let`s, which the others read by the name they import and never set. No imported name is
//      hidden by a local of the same name (the lines below it read the local, without a word: T.367.4's pitfall), and
//      no name is declared at the top of two modules of one page (which of the two a reader means is then a matter of
//      an import's line). The owners read nothing of a page and import no other part, so that Node can import them
//      (tests/page-parts-check.mjs runs what it calls, and no more); every function of an owner is called by another
//      part (an event nobody says is a state that stays as it was); opening.ts runs before any part that reads the
//      model as it loads; and the one object the parts shared (`page`, of state.ts) is not back.
// The types come off with esbuild (with Node's own stripTypeScriptTypes for 3., which renames nothing) and the parser
// is @babel/parser; npm ci installs both for Astro.
// What it does not see: what the page does (tests/e2e.mjs, in CI's browsers).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";
import { parse } from "@babel/parser";
import { namesOf } from "./names.mjs";
import { stripTypeScriptTypes } from "node:module";

const root = fileURLToPath(new URL("..", import.meta.url));
// [the page, the folder of its script's modules]
const PAGES = [["src/pages/index.astro", "src/page"], ["src/pages/benchmark.astro", "src/benchmark"]];
// T.378.5: the owners of what a page's script holds, and all they may use without declaring it (nothing of a page)
export const OWNERS = { "src/page": ["model.ts", "answer.ts"], "src/benchmark": ["running.ts"] };
const OWNERS_GLOBALS = new Set("Boolean Math undefined".split(" "));

// what a page has without declaring it, as far as these files use it (__BUILD__: astro.config.mjs defines it)
const GLOBALS = new Set(("AbortSignal Array Boolean Error Event JSON Map Math NaN Number Object Option Promise RegExp Set String URL URLSearchParams Uint32Array Worker __BUILD__ confirm console " +
  "WeakMap WeakSet crypto document fetch history innerHeight innerWidth localStorage location navigator self sessionStorage setTimeout undefined window " +
  // (/benchmark/'s)
  "SharedArrayBuffer addEventListener clearInterval clearTimeout getSelection performance removeEventListener setInterval").split(" "));

/** The page's own script: what is between <script> and </script>, which Astro reads as a TypeScript module. */
export function scriptOf(text, PAGE = PAGES[0][0]) {
  const found = /<script>\n([\s\S]*?)<\/script>/.exec(text);
  assert.ok(found && text.split("<script").length === 2, `${PAGE} has one <script>, without attributes`);
  return found[1];
}

/** name (a path from the root) → { text, code (the types taken off, the imports kept), imports: [{ from, names }] } */
export function pageSources([PAGE, FOLDER] = PAGES[0]) {
  const files = new Map();
  const folder = path.join(root, FOLDER);
  const names = [PAGE, ...(fs.existsSync(folder) ? fs.readdirSync(folder).sort().map((file) => `${FOLDER}/${file}`) : [])];
  for (const name of names) {
    if (name !== PAGE) assert.ok(name.endsWith(".ts"), `${name}: the modules of the page are .ts files`);
    const whole = fs.readFileSync(path.join(root, name), "utf8");
    const text = name === PAGE ? scriptOf(whole, PAGE) : whole;
    const { code } = transformSync(text, { loader: "ts", sourcefile: name, tsconfigRaw: { compilerOptions: { verbatimModuleSyntax: true } } });
    const program = parse(code, { sourceType: "module" }).program;
    const imports = program.body.filter((statement) => statement.type === "ImportDeclaration").map((statement) => {
      assert.ok(statement.specifiers.every((specifier) => specifier.type === "ImportSpecifier") || !statement.specifiers.length,
        `${name}: this check knows \`import { a } from\` and \`import "…"\``);
      assert.ok(/^\.\.?\//.test(statement.source.value), `${name} imports ${statement.source.value}: this check knows the files of src/`);
      return { from: path.posix.join(path.posix.dirname(name), statement.source.value), names: statement.specifiers.map((specifier) => specifier.imported.name) };
    });
    // (the names a file exports: `export const`, `export let`, `export function`)
    const exported = program.body.filter((statement) => statement.type === "ExportNamedDeclaration").flatMap((statement) => {
      assert.ok(statement.declaration, `${name}: this check knows \`export const\`, \`export let\` and \`export function\``);
      return statement.declaration.type === "VariableDeclaration" ? statement.declaration.declarations.map((d) => (assert.equal(d.id.type, "Identifier"), d.id.name))
        : [statement.declaration.id.name];
    });
    assert.ok(!program.body.some((statement) => /^Export(Default|All)/.test(statement.type)), `${name}: an export this check does not know`);
    files.set(name, { text, code, imports, exported });
  }
  return files;
}

/** A module's text with its types blanked and nothing else changed (Node's own way with a .ts file). */
function written(text) {
  const { emitWarning } = process;
  process.emitWarning = () => {};  // (the function is experimental in Node 24, and says so every time)
  try {
    return stripTypeScriptTypes(text);
  } finally {
    process.emitWarning = emitWarning;
  }
}

if (import.meta.url === new URL(process.argv[1], "file:").href) for (const [PAGE, FOLDER] of PAGES) {
  const files = pageSources([PAGE, FOLDER]);
  assert.ok(files.size > 1, `${FOLDER}/ has the modules of ${PAGE}'s script`);

  // ---- 1. the names
  let count = 0;
  for (const [name, { code }] of files) {
    const { free } = namesOf(code);
    const unknown = [...free].filter((used) => !GLOBALS.has(used));
    const whose = (used) => [...files].filter(([, other]) => other.exported.includes(used)).map(([other]) => other);
    assert.deepEqual(unknown, [], `${name} uses ${unknown.map((used) => `${used}${whose(used).length ? ` (exported by ${whose(used)}: not imported)` : ""}`).join(", ")}, ` +
      "which it neither declares nor imports: a ReferenceError in a browser");
    count += free.size;
  }
  console.log(`page-modules-check: ${PAGE}: ${files.size} file${files.size > 1 ? "s" : ""} of the page's script use ${count} names of a page's globals, and nothing they do not declare or import`);

  // ---- 2. no ring, and the script reaches every file
  const reached = new Set(), open = [];
  (function visit(name) {
    assert.ok(!open.includes(name), `the imports go round: ${[...open.slice(open.indexOf(name)), name].join(" → ")}`);
    if (reached.has(name) || !files.has(name)) return;
    open.push(name);
    for (const { from } of files.get(name).imports) visit(from);
    open.pop();
    reached.add(name);
  })(PAGE);
  assert.deepEqual([...files.keys()].filter((name) => !reached.has(name)), [], `files of ${FOLDER}/ that the page's script never imports`);
  console.log(`page-modules-check: the imports go round nowhere; in the order they run: ${[...reached].map((name) => path.posix.basename(name)).join(", ")}`);

  // ---- 3. T.378.5: what the parts hold has owners
  // (the types off by Node itself, which leaves every name as it is written: esbuild gives a local that hides an
  // import another name (model2), and the line that reads the wrong one would pass)
  const names = new Map([...files].map(([name, { text }]) => [name, namesOf(written(text))]));
  let taken = 0;
  for (const [name, { imported, inner }] of names) {
    const hidden = imported.filter((one) => inner.has(one));
    assert.deepEqual(hidden, [], `${name} declares ${hidden.join(", ")} inside a function, and imports the same name: the lines below read the local one`);
    taken += imported.length;
  }
  const declaredBy = new Map();
  for (const [name, { kinds }] of names) {
    for (const [declared, kind] of kinds) {
      if (kind === "import") continue;
      assert.ok(!declaredBy.has(declared), `${declared} is declared at the top of both ${declaredBy.get(declared)} and ${name}: one name for two things of one page`);
      declaredBy.set(declared, name);
    }
  }
  assert.ok(!files.has(`${FOLDER}/state.ts`) && !declaredBy.has("page"), `${FOLDER}/: the one object the parts shared is back (state.ts, \`page\`): what they hold has owners (${OWNERS[FOLDER].join(", ")})`);
  // (a `page.<name>` left anywhere is a name used and not declared: 1. above says it)
  const lets = [];
  for (const owner of OWNERS[FOLDER]) {
    const name = `${FOLDER}/${owner}`;
    assert.ok(names.has(name), `${name}, an owner of the page's state, is not there`);
    const { free, kinds } = names.get(name);
    const reads = [...free].filter((used) => !OWNERS_GLOBALS.has(used));
    assert.deepEqual(reads, [], `${name} uses ${reads.join(", ")}: an owner of the page's state reads nothing of a page (Node imports it as it is)`);
    const own = [...kinds].filter(([, kind]) => kind === "let").map(([declared]) => declared);
    assert.ok(own.length > 0, `${name} declares no variable: it owns nothing`);
    // (and it takes nothing from a module that reads a page as it loads: its imports are ../models.js's at most)
    for (const { from } of files.get(name).imports) assert.ok(!from.startsWith(`${FOLDER}/`), `${name} imports ${from}: an owner imports no other part of the page's script`);
    lets.push(...own.map((declared) => `${owner}: ${declared}`));
  }
  // every function of an owner is called by another part of the script: an event nobody says is a state that stays as
  // it was, without a word (the worker's ready that no becameReady() follows is a page whose button never comes on)
  const calledIn = (code) => {
    const found = new Set();
    (function visit(node) {
      if (!node || typeof node !== "object") return;
      if (Array.isArray(node)) return node.forEach(visit);
      if (node.type === "CallExpression" && node.callee.type === "Identifier") found.add(node.callee.name);
      for (const key of Object.keys(node)) if (key !== "loc") visit(node[key]);
    })(parse(code, { sourceType: "module" }).program);
    return found;
  };
  const owners = OWNERS[FOLDER].map((owner) => `${FOLDER}/${owner}`);
  const called = new Set([...files].filter(([name]) => !owners.includes(name)).flatMap(([, { text }]) => [...calledIn(written(text))]));
  const events = owners.flatMap((name) => [...names.get(name).kinds].filter(([, kind]) => kind !== "let" && kind !== "import").map(([declared]) => declared));
  assert.deepEqual(events.filter((event) => !called.has(event)), [], `functions of ${OWNERS[FOLDER].join(", ")} that no other part of the script calls`);
  // (the model page: what the visit opens with is said before any other part reads the model as it loads)
  const order = [...reached];
  for (const [name, { imports }] of files) {
    if (!files.has(`${FOLDER}/opening.ts`) || name === `${FOLDER}/opening.ts` || owners.includes(name)) continue;
    if (!imports.some(({ from, names: taken }) => from === `${FOLDER}/model.ts` && taken.includes("model"))) continue;
    assert.ok(order.indexOf(`${FOLDER}/opening.ts`) < order.indexOf(name), `${name} reads the model and runs before opening.ts says which one the visit opens with`);
  }
  console.log(`page-modules-check: ${lets.length} variables of ${OWNERS[FOLDER].join(" and ")}, each set by its own module alone; ` +
    `${taken} names imported in ${FOLDER}/, none hidden by a local; no top-level name twice; ${events.length} events, each said somewhere`);
}
