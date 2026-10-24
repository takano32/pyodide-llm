// shaders-source.mjs (T.351): src/runtime/shaders.js and its modules (src/runtime/shaders/*.js), read for the checks.
//
// shaders.js is the window; the texts and their makers are in the modules it lists. The checks that read the shaders'
// source (tests/device-key-check.mjs, tests/shaders-modules-check.mjs) take it from here:
//
//   shaderParts(tree)     { window, linking, list, parts }: the window's text and what tests/imports.mjs reads of it,
//                         the names of its modules in its order, and for
//                         each { name, file, text, program, takes: Map(neighbour -> names), declared: the names it
//                         exports where they are declared, given: the names its last line exports to its neighbours }
//   oneSource(tree)       the modules as the one module they were: each part's statements in the window's order, the
//                         lines that take a neighbour's names and the lists that hand them out left out. The names a
//                         part takes are then the declarations of the parts before it, by the same names, so this is
//                         what a bundler makes of them. (That every name taken is exported where it is taken from is the
//                         type check's, and that the window's order has each part after those it takes from,
//                         shaders-modules-check's.)
//
// tree: tests/tree.mjs's (this tree's by default; treeOf(<tests/other-tree.mjs's folder>) for another commit's). The
// lines that take a neighbour's names are read by tests/imports.mjs, in the form the tree has: awaited imports with the
// file's own ?v= and the window's one list where the files are fetched as they are, static imports where a bundler links
// them (there the window's order is the order of what takes from what).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { windowed, without } from "./imports.mjs";
import { treeOf } from "./tree.mjs";

/** the names an object pattern takes, each a plain one ({ a, b }: no renaming, no default) */
export const keysOf = (pattern, what) => pattern.properties.map((property) => {
  assert.ok(property.type === "ObjectProperty" && !property.computed && property.key.type === "Identifier" && property.shorthand, `${what}: a plain name`);
  return property.key.name;
});

export function shaderParts(tree = treeOf()) {
  // (T.368.5: the window by the name the tree has, shaders.ts or another commit's shaders.js: windowed() blanks the
  // types of a file by the name it is asked for, and the parts have the window's ending)
  const window = path.basename(tree.runtime("shaders.js")), ending = path.extname(window);
  const { files, order, asked } = windowed(window, { read: (name) => fs.readFileSync(tree.runtime(name), "utf8"), list: (folder) => fs.readdirSync(tree.runtime(folder)), bundled: tree.bundled });
  const short = (file) => file.slice("shaders/".length, -ending.length);
  const list = asked ? asked.names : order.filter((file) => file !== window).map(short);
  const parts = list.map((name) => {
    const file = `shaders/${name}${ending}`, { text, program, takes: taken, left: linking } = files.get(file);
    const takes = new Map(), declared = [], given = [], left = [...linking];  // left: the statements oneSource() leaves out
    for (const { from, names } of taken) {
      assert.ok(!takes.has(short(from)), `${file} takes from ${short(from)}${ending} twice`);
      takes.set(short(from), names.map(({ imported, local }) => (assert.equal(imported, local, `${file}: a plain name`), imported)));
    }
    for (const node of program.body) {
      if (node.type === "ExportNamedDeclaration" && !node.declaration) {
        for (const specifier of node.specifiers) {
          assert.equal(specifier.local.name, specifier.exported.name, `${file}: a name exported under another`);
          given.push(specifier.exported.name);
        }
        left.push(node);
        continue;
      }
      const s = node.type === "ExportNamedDeclaration" ? node.declaration : node;
      if (node !== s) declared.push(...(s.id ? [s.id.name] : s.declarations.map((d) => (assert.equal(d.id.type, "Identifier", `${file}: an exported pattern`), d.id.name))));
      const awaited = s.type === "VariableDeclaration" && s.declarations.some((one) => one.init?.type === "AwaitExpression");
      assert.ok(!awaited || linking.includes(node), `${file}: what a part awaits at its top is a neighbour's names, taken apart`);
    }
    return { name, file, text, program, takes, declared, given, left };
  });
  return { window: files.get(window).text, linking: files.get(window), list, parts };
}

export function oneSource(tree = treeOf()) {
  return shaderParts(tree).parts.map(({ text, left }) => without(text, left)).join("\n");
}
