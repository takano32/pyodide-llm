// tests/page-modules-check.mjs (T355): the names that the model page's script uses, in src/pages/index.astro and in its
// modules (src/page/*.ts).
//
// The build takes the types off and bundles: it does not say that a name is declared nowhere (a ReferenceError in the
// browser, at the moment the line runs), and no test opens the page outside CI's browsers. So this reads the script
// and each module as the module it is, and holds
//   1. every name it uses to one of: declared in the file, imported by it, a global of a page listed below;
//   2. every name it imports to an export of the file it names (src/models.js and src/bench.js: imported for real);
//   3. the imports to a graph without a ring, in which the script reaches every file of src/page/;
//   4. every page.<name> to a field of the one object the parts share (page is no variable's name but an object's: a
//      field that is not there reads undefined and is set without a word), and every field to a use.
// Node only, under a second:
//
//   node tests/page-modules-check.mjs
//
// The types come off with esbuild and the parser is @babel/parser, both of which npm ci installs for Astro.
// What it does not see: what the page does (tests/e2e.mjs, in CI's browsers).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";
import { parse } from "@babel/parser";
import { namesOf } from "./worker-modules-check.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const PAGE = "src/pages/index.astro", FOLDER = "src/page";

// what a page has without declaring it, as far as these files use it (__BUILD__: astro.config.mjs defines it)
const GLOBALS = new Set(("Boolean Error Event JSON Math NaN Number Object Option Promise Set String URL URLSearchParams Uint32Array Worker __BUILD__ confirm console " +
  "crypto document history innerHeight innerWidth localStorage location navigator self sessionStorage setTimeout undefined window").split(" "));

/** The page's own script: what is between <script> and </script>, which Astro reads as a TypeScript module. */
export function scriptOf(text) {
  const found = /<script>\n([\s\S]*?)<\/script>/.exec(text);
  assert.ok(found && text.split("<script").length === 2, `${PAGE} has one <script>, without attributes`);
  return found[1];
}

/** name (a path from the root) → { text, code (the types taken off, the imports kept), imports: [{ from, names }] } */
export function pageSources() {
  const files = new Map();
  const folder = path.join(root, FOLDER);
  const names = [PAGE, ...(fs.existsSync(folder) ? fs.readdirSync(folder).sort().map((file) => `${FOLDER}/${file}`) : [])];
  for (const name of names) {
    if (name !== PAGE) assert.ok(name.endsWith(".ts"), `${name}: the modules of the page are .ts files`);
    const whole = fs.readFileSync(path.join(root, name), "utf8");
    const text = name === PAGE ? scriptOf(whole) : whole;
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

if (import.meta.url === new URL(process.argv[1], "file:").href) {
  const files = pageSources();

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
  console.log(`page-modules-check: ${files.size} file${files.size > 1 ? "s" : ""} of the page's script use ${count} names of a page's globals, and nothing they do not declare or import`);

  // ---- 2. what is imported is exported there
  const outside = new Map();
  for (const [name, { imports }] of files) {
    for (const { from, names } of imports) {
      if (!files.has(from) && !outside.has(from)) {
        assert.ok(fs.existsSync(path.join(root, from)) && from.endsWith(".js"), `${name} imports ${from}, which is no file of the page's script and no module of src/`);
        outside.set(from, Object.keys(await import(path.join(root, from))));
      }
      const there = files.get(from)?.exported ?? outside.get(from);
      for (const one of names) assert.ok(there.includes(one), `${name} imports ${one} from ${from}, which does not export it`);
    }
  }

  // ---- 3. no ring, and the script reaches every file
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
  console.log(`page-modules-check: the imports name what is exported, without a ring; in the order they run: ${[...reached].map((name) => path.posix.basename(name)).join(", ")}`);

  // ---- 4. the fields of the one object the parts share
  const holders = [...files].filter(([, { code }]) => /^(export )?const page = \{$/m.test(code));
  assert.equal(holders.length, 1, "one file declares `const page = {`");
  const declaration = parse(holders[0][1].code, { sourceType: "module" }).program.body.map((statement) => statement.declaration ?? statement)
    .find((statement) => statement.type === "VariableDeclaration" && statement.declarations[0].id.name === "page").declarations[0];
  const fields = declaration.init.properties.map((property) => (assert.ok(property.type === "ObjectProperty" && !property.computed), property.key.name));
  assert.equal(new Set(fields).size, fields.length, "a field of page twice");
  const used = new Set();
  for (const [name, { text }] of files) {
    for (const [, field] of text.matchAll(/\bpage\.(\w+)/g)) {
      assert.ok(fields.includes(field), `${name} says page.${field}, which is no field of ${holders[0][0]}'s object`);
      used.add(field);
    }
    assert.ok(!/\bpage\s*\[/.test(text), `${name} reads page[…]: this check knows page.<name>`);
  }
  assert.deepEqual(fields.filter((field) => !used.has(field)), [], "fields of page that nothing reads or sets");
  console.log(`page-modules-check: the ${fields.length} fields of page (${holders[0][0]}) are the ones the script uses`);
}
