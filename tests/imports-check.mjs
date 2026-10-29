// tests/imports-check.mjs (T.367.1): tests/imports.mjs alone, on made-up texts and made-up folders (stand-ins for the
// reading and the listing: nothing of the product is read). How a file takes its neighbours' names, what is refused
// (T.587: the awaited imports with a ?v= of before T.367 too), what is left out of a text, the order of what takes from
// what, and a window held to its folder. Node only, under a second:
//
//   node tests/imports-check.mjs
import assert from "node:assert/strict";
import { importsOf, reached, windowed, without } from "./imports.mjs";

const V = "${new URL(import.meta.url).search}", FIRST = "${self.location.search}";
const awaited = (names, file, search = V) => `const { ${names} } = await import(new URL(\`${file}${search}\`, import.meta.url));\n`;
const listOf = (folder, names, search = V) => `const modules = Object.fromEntries([${names.map((name) => `"${name}"`).join(", ")}].map((name) =>\n  [name, import(new URL(\`${folder}/\${name}.js${search}\`, import.meta.url))]));\n`;
const said = (takes) => takes.map(({ from, names }) => `${from}: ${names.map(({ imported, local }) => (imported === local ? imported : `${imported} as ${local}`)).join(" ")}`);
let checks = 0;
const check = (name, run) => { run(); checks += 1; };

check("static imports", () => {
  const text = `// head\nimport { a, b } from "./x.js";\nimport { WAKE as wake } from "../jobs.js";\nimport {\n  c,\n  d,\n} from "./deep/y.ts";\nexport const one = a + b + wake + c + d;\n`;
  const { takes, left, program } = importsOf(text, "forward/threads.js");
  assert.deepEqual(said(takes), ["forward/x.js: a b", "jobs.js: WAKE as wake", "forward/deep/y.ts: c d"]);
  assert.equal(left.length, 3);
  assert.equal(program.body.length, 4);
  assert.equal(without(text, left, true), "// head\nexport const one = a + b + wake + c + d;\n", "the lines that link are cut, with their ends of line");
  assert.equal(without(text, left), "// head\n\n\n\nexport const one = a + b + wake + c + d;\n", "or without them");
});

check("what is refused", () => {
  for (const [text, words] of [
    [`import fs from "node:fs";\n`, /an import this check does not know/],
    [`import { a } from "x.js";\n`, /an import this check does not know/],
    [`import { a } from "./x.mjs";\n`, /an import this check does not know/],
    [`import { a } from "./x.d.ts";\n`, /an import this check does not know/],
    [`import a from "./x.js";\n`, /the names, each by itself/],
    [`import * as all from "./x.js";\n`, /the names, each by itself/],
    [`export * from "./x.js";\n`, /an export from another file/],
    [`export { a } from "./x.js";\n`, /an export from another file/],
    [`const { a } = await import("./x.js");\n`, /an import this check does not know/],
    [`const { a } = await import(new URL("x.js", import.meta.url));\n`, /an import this check does not know/],
    // (T.587: the form of a tree served as it is, from before T.367, is read no more)
    [awaited("a", "x.js"), /an import this check does not know/],
    [awaited("a", "worker/state.js", FIRST), /an import this check does not know/],
  ]) assert.throws(() => importsOf(text, "f.js"), words, text);
});

check("what is not looked at: nothing awaited at the top, or no import", () => {
  const text = `const later = import("./python.js");\nimport("./shaders.js").catch(() => {});\nconst first = await started;\n` +
    `async function load() {\n  return (await import("./forward.js")).a;\n}\n`;
  const { takes, left } = importsOf(text, "worker.js");
  assert.deepEqual([takes, left], [[], []]);
});

check("the order of what takes from what, and a ring", () => {
  const graph = { w: ["b", "a"], a: [], b: ["a", "c"], c: [] };
  assert.deepEqual(reached("w", (name) => graph[name]), ["a", "c", "b", "w"]);
  const ended = ["c"];
  assert.equal(reached("b", (name) => graph[name], ended), ended);
  assert.deepEqual(reached("w", (name) => graph[name], ended), ["c", "a", "b", "w"], "into one order, what is placed already where it was");
  assert.throws(() => reached("w", (name) => ({ ...graph, c: ["b"] })[name]), /the modules wait for one another: b → c → b/);
  assert.throws(() => reached("a", () => ["a"]), /a → a/);
});

// ---- a window and its folder, on a made-up folder: { name: text }
const folderOf = (texts) => ({ read: (name) => { if (!(name in texts)) throw new Error(`no ${name}`); return texts[name]; },
  list: (folder) => Object.keys(texts).filter((name) => name.startsWith(`${folder}/`)).map((name) => name.slice(folder.length + 1)).concat(["notes.txt"]) });
const FOLDER = { "w.js": `import { one } from "./w/a.js";\nimport { two } from "./w/b.js";\nexport { one, two };\n`,
  "w/a.js": `export const one = 1;\n`, "w/b.js": `import { one } from "./a.js";\nimport { K } from "../k.js";\nexport const two = one + K;\n`, "k.js": "export const K = 1;\n" };

check("a window holds its folder", () => {
  const { files, there, order } = windowed("w.js", { ...folderOf(FOLDER), more: ["k.js"] });
  assert.deepEqual(there, ["w/a.js", "w/b.js"], "the folder's .js files, by their names under the runtime");
  assert.deepEqual([...files.keys()], ["w.js", "k.js", "w/a.js", "w/b.js"]);
  assert.deepEqual(order, ["w/a.js", "k.js", "w/b.js", "w.js"], "each after what it takes from, the window last");
  assert.deepEqual(said(files.get("w/b.js").takes), ["w/a.js: one", "k.js: K"]);
  // outside: a file out of the folder that a module takes names of is read and placed too, and what it takes from in
  // turn; one that is not there fails as the read does
  const deeper = ({ ...FOLDER, "k.js": `import { L } from "./l.js";\nexport const K = L;\n`, "l.js": "export const L = 1;\n" });
  const { "k.js": gone, ...without } = FOLDER;
  const outside = windowed("w.js", { ...folderOf(deeper), outside: true });
  assert.deepEqual(outside.others, ["k.js", "l.js"], "the files outside the folder, in the order they were met");
  assert.deepEqual(outside.there, ["w/a.js", "w/b.js"]);
  assert.deepEqual([...outside.files.keys()], ["w.js", "w/a.js", "w/b.js", "k.js", "l.js"]);
  assert.deepEqual(outside.order, ["w/a.js", "l.js", "k.js", "w/b.js", "w.js"]);
  assert.deepEqual(windowed("w.js", { ...folderOf(FOLDER), more: ["k.js"], outside: true }).others, [], "what `more` named is no other");
  assert.throws(() => windowed("w.js", { ...folderOf(without), outside: true }), /no k\.js/);
});

check("a window that does not hold its folder", () => {
  const but = (texts, changes) => folderOf(Object.fromEntries(Object.entries({ ...texts, ...changes }).filter(([, text]) => text !== null)));
  // a module that takes of a file that is not one of them (k.js is, only where the check says so)
  assert.throws(() => windowed("w.js", { ...folderOf(FOLDER) }), /w\/b\.js takes names from k\.js, which is no file of w\//);
  // a file of the folder nothing imports
  assert.throws(() => windowed("w.js", { ...but(FOLDER, { "w/c.js": "export const three = 3;\n" }), more: ["k.js"] }), /w\/c\.js: in w\/, and neither w\.js nor a file it takes from/);
  // a ring; and the old form's lines in a file
  assert.throws(() => windowed("w.js", { ...but(FOLDER, { "w/a.js": `import { two } from "./b.js";\nexport const one = two;\n` }), more: ["k.js"] }), /wait for one another: w\/a\.js → w\/b\.js → w\/a\.js/);
  assert.throws(() => windowed("w.js", { ...but(FOLDER, { "w/a.js": `${awaited("K", "../k.js")}export const one = K;\n` }), more: ["k.js"] }), /w\/a\.js: an import this check does not know/);
  // (a window's list of before T.367 is a promise kept, which is not looked at: its folder is then imported by nobody)
  assert.throws(() => windowed("w.js", { ...but(FOLDER, { "w.js": `${listOf("w", ["a", "b"])}export const none = 0;\n` }), more: ["k.js"] }), /w\/a\.js, w\/b\.js: in w\/, and neither w\.js nor a file it takes from takes anything of them/);
});

console.log(`imports-check: ok (${checks} checks of tests/imports.mjs, on made-up texts and folders)`);
