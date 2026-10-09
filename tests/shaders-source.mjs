// shaders-source.mjs (T351): public/shaders.js and its modules (public/shaders/*.js), read for the checks.
//
// shaders.js is the window; the texts and their makers are in the modules it lists. The checks that read the shaders'
// source (tests/device-key-check.mjs, tests/shaders-modules-check.mjs) take it from here:
//
//   shaderParts(folder)   { window, list, parts }: the window's text, the names it asks for in its order, and for
//                         each { name, file, text, program, takes: Map(neighbour -> names), declared: the names it
//                         exports where they are declared, given: the names its last line exports to its neighbours }
//   oneSource(folder)     the modules as the one module they were: each part's statements in the window's order, the
//                         lines that take a neighbour's names and the lists that hand them out left out. The names a
//                         part takes are then the declarations of the parts before it, by the same names, so this is
//                         what a bundler makes of them. (That every name taken is exported where it is taken from, and
//                         that the window's order has each part after those it takes from, is shaders-modules-check's.)
//
// folder: the tree's public/ as a URL (this tree's by default; tests/other-tree.mjs's for another commit's).
import assert from "node:assert/strict";
import fs from "node:fs";
import { parse } from "@babel/parser";

const PUBLIC = new URL("../public/", import.meta.url);
const ASKED = /\[([^\]]*)\]\.map\(\(name\) =>\n\s*\[name, import\(new URL\(`shaders\/\$\{name\}\.js\$\{new URL\(import\.meta\.url\)\.search\}`, import\.meta\.url\)\)\]\)\);/;
const TAKEN = /^await import\(new URL\(`(\w+)\.js\$\{new URL\(import\.meta\.url\)\.search\}`, import\.meta\.url\)\)$/;

/** the names an object pattern takes, each a plain one ({ a, b }: no renaming, no default) */
export const keysOf = (pattern, what) => pattern.properties.map((property) => {
  assert.ok(property.type === "ObjectProperty" && !property.computed && property.key.type === "Identifier" && property.shorthand, `${what}: a plain name`);
  return property.key.name;
});

export function shaderParts(folder = PUBLIC) {
  const window_ = fs.readFileSync(new URL("shaders.js", folder), "utf8");
  const asked = ASKED.exec(window_);
  assert.ok(asked, "shaders.js asks for its modules in one list, each with its own ?v=");
  const list = [...asked[1].matchAll(/"(\w+)"/g)].map((match) => match[1]);
  const parts = list.map((name) => {
    const file = `shaders/${name}.js`, text = fs.readFileSync(new URL(file, folder), "utf8");
    const program = parse(text, { sourceType: "module" }).program;
    const takes = new Map(), declared = [], given = [], left = [];  // left: the statements oneSource() leaves out
    for (const node of program.body) {
      assert.notEqual(node.type, "ImportDeclaration", `${file}: a static import drops the ?v= (GitHub Pages keeps a file for ten minutes): await import(new URL(…))`);
      if (node.type === "ExportNamedDeclaration" && !node.declaration) {
        assert.ok(!node.source, `${file}: an export from another file`);
        for (const specifier of node.specifiers) {
          assert.equal(specifier.local.name, specifier.exported.name, `${file}: a name exported under another`);
          given.push(specifier.exported.name);
        }
        left.push(node);
        continue;
      }
      const s = node.type === "ExportNamedDeclaration" ? node.declaration : node;
      if (node !== s) declared.push(...(s.id ? [s.id.name] : s.declarations.map((d) => (assert.equal(d.id.type, "Identifier", `${file}: an exported pattern`), d.id.name))));
      const d = s.type === "VariableDeclaration" ? s.declarations.find((one) => one.init?.type === "AwaitExpression") : null;
      if (!d) continue;
      assert.ok(node === s && s.declarations.length === 1 && d.id.type === "ObjectPattern", `${file}: what a part awaits at its top is a neighbour's names, taken apart`);
      const from = TAKEN.exec(text.slice(d.init.start, d.init.end))?.[1];
      assert.ok(from, `${file}: \`${text.slice(d.init.start, d.init.end)}\` is no module of this folder asked for with this file's own ?v=`);
      assert.ok(!takes.has(from), `${file} takes from ${from}.js twice`);
      takes.set(from, keysOf(d.id, file));
      left.push(node);
    }
    return { name, file, text, program, takes, declared, given, left };
  });
  return { window: window_, list, parts };
}

export function oneSource(folder = PUBLIC) {
  return shaderParts(folder).parts.map(({ text, left }) =>
    left.toSorted((a, b) => b.start - a.start).reduce((now, node) => `${now.slice(0, node.start)}${now.slice(node.end)}`, text)).join("\n");
}
