// tests/imports-check.mjs (T367.1): tests/imports.mjs alone, on made-up texts and made-up folders (stand-ins for the
// reading and the listing: nothing of the product is read). The two forms a file takes its neighbours' names in, what
// is refused in each, what is left out of a text, the order of what takes from what, and a window held to its folder.
// Node only, under a second:
//
//   node tests/imports-check.mjs
import assert from "node:assert/strict";
import { importsOf, reached, windowed, without } from "./imports.mjs";

const V = "${new URL(import.meta.url).search}", FIRST = "${self.location.search}";
const awaited = (names, file, search = V) => `const { ${names} } = await import(new URL(\`${file}${search}\`, import.meta.url));\n`;
const listOf = (folder, names, search = V) => `const modules = Object.fromEntries([${names.map((name) => `"${name}"`).join(", ")}].map((name) =>\n  [name, import(new URL(\`${folder}/\${name}.js${search}\`, import.meta.url))]));\n`;
const said = (takes) => takes.map(({ from, names, how }) => `${how} ${from}: ${names.map(({ imported, local }) => (imported === local ? imported : `${imported} as ${local}`)).join(" ")}`);
let checks = 0;
const check = (name, run) => { run(); checks += 1; };

check("a neighbour, with the file's own ?v=", () => {
  const text = `// head\n${awaited("a, b", "x.js")}${awaited("WAKE: wake", "../jobs.js")}const { c,\n  d } =\n  await import(new URL(\`deep/y.js${V}\`, import.meta.url));\nexport const one = a + b + wake + c + d;\n`;
  const { takes, left, asked, program } = importsOf(text, "forward/threads.js", { bundled: false });
  assert.deepEqual(said(takes), ["await import forward/x.js: a b", "await import jobs.js: WAKE as wake", "await import forward/deep/y.js: c d"]);
  assert.equal(asked, null);
  assert.equal(left.length, 3);
  assert.equal(program.body.length, 4);
  assert.equal(without(text, left, true), "// head\nexport const one = a + b + wake + c + d;\n", "the lines that link are cut, with their ends of line");
  assert.equal(without(text, left), "// head\n\n\n\nexport const one = a + b + wake + c + d;\n", "or without them");
  // (a worker's first file reads its ?v= from its location)
  assert.deepEqual(said(importsOf(awaited("a", "worker/state.js", FIRST), "worker.js", { bundled: false }).takes), ["await import worker/state.js: a"]);
});

check("a window: one list, then each module's names", () => {
  const text = `${listOf("gpu", ["device", "start"])}const { open } = await modules.start;\nconst { common, buffer } = await modules.device;\nexport { open };\n`;
  const { takes, asked, left } = importsOf(text, "benchmark/gpu.js", { bundled: false });
  assert.deepEqual([asked.folder, asked.names], ["benchmark/gpu", ["device", "start"]]);
  assert.deepEqual(said(takes), ["await modules benchmark/gpu/start.js: open", "await modules benchmark/gpu/device.js: common buffer"]);
  assert.equal(without(text, left, true), "export { open };\n");
  assert.deepEqual(importsOf(listOf("worker", ["a"], FIRST), "worker.js", { bundled: false }).asked.names, ["a"]);
});

check("static imports, where a bundler links the files", () => {
  const text = `// head\nimport { a, b } from "./x.js";\nimport { WAKE as wake } from "../jobs.js";\nimport {\n  c,\n  d,\n} from "./deep/y.js";\nexport const one = a + b + wake + c + d;\n`;
  const { takes, left, asked } = importsOf(text, "forward/threads.js", { bundled: true });
  assert.deepEqual(said(takes), ["import forward/x.js: a b", "import jobs.js: WAKE as wake", "import forward/deep/y.js: c d"]);
  assert.equal(asked, null);
  assert.equal(without(text, left, true), "// head\nexport const one = a + b + wake + c + d;\n");
});

check("the two forms of one module say the same", () => {
  const body = "export function f() { return a + c; }\n";
  const one = importsOf(`${awaited("a", "x.js")}${awaited("b: c", "../y.js")}${body}`, "m/n.js", { bundled: false });
  const other = importsOf(`import { a } from "./x.js";\nimport { b as c } from "../y.js";\n${body}`, "m/n.js", { bundled: true });
  const plain = (takes) => takes.map(({ from, names }) => [from, names]);
  assert.deepEqual(plain(one.takes), plain(other.takes));
  assert.equal(without(`${awaited("a", "x.js")}${awaited("b: c", "../y.js")}${body}`, one.left, true), body);
});

check("a tree has one form: the other one in a file is refused", () => {
  assert.throws(() => importsOf(`import { a } from "./x.js";\n`, "f.js", { bundled: false }), /f\.js: a static import drops the \?v=/);
  assert.throws(() => importsOf(awaited("a", "x.js"), "f.js", { bundled: true }), /f\.js: an awaited import with a \?v=, in a tree whose files a bundler links/);
  assert.throws(() => importsOf(listOf("f", ["a"]), "f.js", { bundled: true }), /a window's list of its modules with a \?v=/);
  assert.throws(() => importsOf("", "f.js", {}), /bundled/);
  assert.throws(() => importsOf("", "f.js", { bundled: undefined }), /bundled/);
});

check("what is refused in either form", () => {
  for (const bundled of [false, true]) {
    for (const [text, words] of [
      [`import fs from "node:fs";\n`, /an import this check does not know/],
      [`import { a } from "x.js";\n`, /an import this check does not know/],
      [`import { a } from "./x.mjs";\n`, /an import this check does not know/],
      [`export * from "./x.js";\n`, /an export from another file/],
      [`export { a } from "./x.js";\n`, /an export from another file/],
      [`const { a } = await import("./x.js");\n`, /an import this check does not know/],
      [`const { a } = await import(new URL("x.js", import.meta.url));\n`, /an import this check does not know/],
      ["const { a } = await import(new URL(`x.js`, import.meta.url));\n", /an import this check does not know/],  // (no ?v=: another deployment's file)
      [`const { a } = await import(new URL(\`x.js\${search}\`, import.meta.url));\n`, /an import this check does not know/],
      [`const m = await import(new URL(\`x.js\${somewhere.search}\`, import.meta.url));\n`, /an import this check does not know/],
    ]) assert.throws(() => importsOf(text, "f.js", { bundled }), words, `${bundled ? "bundled" : "as it is"}: ${text}`);
  }
  for (const [text, words] of [
    [`import a from "./x.js";\n`, /the names, each by itself/],
    [`import * as all from "./x.js";\n`, /the names, each by itself/],
  ]) assert.throws(() => importsOf(text, "f.js", { bundled: true }), words, text);
  for (const [text, words] of [
    [`const all = await import(new URL(\`x.js${V}\`, import.meta.url));\n`, /its names, taken apart/],
    [`const { a } = await import(new URL(\`x.js${V}\`, import.meta.url)), b = 1;\n`, /its names, taken apart/],
    [`const { a = 1 } = await import(new URL(\`x.js${V}\`, import.meta.url));\n`, /plain names/],
    [`const { ...rest } = await import(new URL(\`x.js${V}\`, import.meta.url));\n`, /plain names/],
    [`const { a } = await modules.x;\n`, /before the window's list/],
    [`${listOf("f", ["x"])}const { a } = await modules.y;\n`, /is no module of the window's list/],
  ]) assert.throws(() => importsOf(text, "f.js", { bundled: false }), words, text);
});

check("what is not looked at: nothing awaited at the top, or no import", () => {
  const text = `const later = import(new URL(\`python.js${FIRST}\`, import.meta.url));\nimport(new URL(\`shaders.js${V}\`, import.meta.url)).catch(() => {});\n` +
    `const first = await started;\nasync function load() {\n  const { a } = await import(new URL(\`forward.js${FIRST}\`, import.meta.url));\n  return a;\n}\n`;
  for (const bundled of [false, true]) {
    const { takes, left, asked } = importsOf(text, "worker.js", { bundled });
    assert.deepEqual([takes, left, asked], [[], [], null]);
  }
  const static_ = `const later = import("./python.js");\nasync function load() {\n  return (await import("./forward.js")).a;\n}\n`;
  assert.deepEqual(importsOf(static_, "worker.js", { bundled: true }).takes, []);
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
const AS_IT_IS = { "w.js": `${listOf("w", ["a", "b"])}const { one } = await modules.a;\nconst { two } = await modules.b;\nexport { one, two };\n`,
  "w/a.js": `export const one = 1;\n`, "w/b.js": `${awaited("one", "a.js")}${awaited("K", "../k.js")}export const two = one + K;\n`, "k.js": "export const K = 1;\n" };
const BUNDLED = { "w.js": `import { one } from "./w/a.js";\nimport { two } from "./w/b.js";\nexport { one, two };\n`,
  "w/a.js": `export const one = 1;\n`, "w/b.js": `import { one } from "./a.js";\nimport { K } from "../k.js";\nexport const two = one + K;\n`, "k.js": "export const K = 1;\n" };

check("a window holds its folder, in both forms", () => {
  for (const [texts, bundled] of [[AS_IT_IS, false], [BUNDLED, true]]) {
    const { files, there, order, asked } = windowed("w.js", { ...folderOf(texts), bundled, more: ["k.js"] });
    assert.deepEqual(there, ["w/a.js", "w/b.js"], "the folder's .js files, by their names under the runtime");
    assert.deepEqual([...files.keys()], ["w.js", "k.js", "w/a.js", "w/b.js"]);
    assert.deepEqual(order, ["w/a.js", "k.js", "w/b.js", "w.js"], "each after what it takes from, the window last");
    assert.equal(Boolean(asked), !bundled);
    assert.deepEqual(said(files.get("w/b.js").takes).map((line) => line.replace(/^(await )?import /, "")), ["w/a.js: one", "k.js: K"]);
  }
});

check("a window that does not hold its folder", () => {
  const but = (texts, changes) => folderOf(Object.fromEntries(Object.entries({ ...texts, ...changes }).filter(([, text]) => text !== null)));
  // a file of the folder the window's list lacks, one the list has and the folder lacks, one listed twice, no list
  assert.throws(() => windowed("w.js", { ...but(AS_IT_IS, { "w/c.js": "" }), bundled: false, more: ["k.js"] }), /w\.js asks for \[a,b\], and w\/ holds \[a,b,c\]/);
  assert.throws(() => windowed("w.js", { ...but(AS_IT_IS, { "w.js": AS_IT_IS["w.js"].replace('"a", "b"', '"a", "b", "c"') }), bundled: false, more: ["k.js"] }), /asks for \[a,b,c\], and w\/ holds \[a,b\]/);
  assert.throws(() => windowed("w.js", { ...but(AS_IT_IS, { "w.js": AS_IT_IS["w.js"].replace('"a", "b"', '"a", "b", "b"') }), bundled: false, more: ["k.js"] }), /asks for/);
  assert.throws(() => windowed("w.js", { ...but(AS_IT_IS, { "w.js": "export const nothing = 1;\n" }), bundled: false, more: ["k.js"] }), /does not ask for its modules in one list/);
  assert.throws(() => windowed("w.js", { ...but(AS_IT_IS, { "w.js": AS_IT_IS["w.js"].replace("`w/", "`v/") }), bundled: false, more: ["k.js"] }), /asks for/);
  // a module that takes of a file that is not one of them (k.js is, only where the check says so)
  assert.throws(() => windowed("w.js", { ...folderOf(AS_IT_IS), bundled: false }), /w\/b\.js takes names from k\.js, which is no file of w\//);
  assert.throws(() => windowed("w.js", { ...folderOf(BUNDLED), bundled: true }), /w\/b\.js takes names from k\.js/);
  // a file of the folder nothing takes from: listed and never taken, or (bundled) never imported
  assert.throws(() => windowed("w.js", { ...but(AS_IT_IS, { "w.js": AS_IT_IS["w.js"].replace("const { two } = await modules.b;\n", "").replace("one, two", "one") }), bundled: false, more: ["k.js"] }),
    /w\/b\.js: in w\/, and neither w\.js nor a file it takes from takes anything of it/);
  assert.throws(() => windowed("w.js", { ...but(BUNDLED, { "w/c.js": "export const three = 3;\n" }), bundled: true, more: ["k.js"] }), /w\/c\.js: in w\/, and neither w\.js nor a file it takes from/);
  // a ring; and the other form's lines in a file
  assert.throws(() => windowed("w.js", { ...but(BUNDLED, { "w/a.js": `import { two } from "./b.js";\nexport const one = two;\n` }), bundled: true, more: ["k.js"] }), /wait for one another: w\/a\.js → w\/b\.js → w\/a\.js/);
  assert.throws(() => windowed("w.js", { ...folderOf(BUNDLED), bundled: false, more: ["k.js"] }), /a static import drops the \?v=/);
  assert.throws(() => windowed("w.js", { ...folderOf(AS_IT_IS), bundled: true, more: ["k.js"] }), /with a \?v=, in a tree whose files a bundler links/);
});

console.log(`imports-check: ok (${checks} checks of tests/imports.mjs, on made-up texts and folders)`);
