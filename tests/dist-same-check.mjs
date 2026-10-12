// tests/dist-same-check.mjs (T368.1): tests/dist-same.mjs on made-up builds, and the two small tools the types stand on
// (tests/untyped.mjs, tests/types.mjs's rule and count), on made-up texts. Node alone, under a second.
//
// dist-same: two builds of one code are "the same code" whatever their hashes and region lines, and anything else is
// said: one character, a comment that moved, a parenthesis, a file more or less, a built file's bytes, a served file.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { compared, report, unhashed } from "./dist-same.mjs";
import { typed, untyped } from "./untyped.mjs";
import { AMBIENT, ambient, anys, typeOnlyImports } from "./types.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const scratch = path.join(root, ".tmp/dist-same-check");
fs.rmSync(scratch, { recursive: true, force: true });
function written(name, files) {
  const folder = path.join(scratch, name);
  for (const [file, text] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(folder, file)), { recursive: true });
    fs.writeFileSync(path.join(folder, file), text);
  }
  return folder;
}
const verdicts = (rows) => Object.fromEntries(rows.map((row) => [row.file, row.verdict]));

// ---- a build before, and the same code built after a source was renamed .ts: other hashes, other region lines
const WORKER = (helper, source) => `//#region src/runtime/${source}\nconst BATCH = 16;\n//#endregion\n//#region src/runtime/worker.js\nnew Worker(new URL("./${helper}", import.meta.url));\nfetch("simdkernel_plain.DfCGpm7G.wasm");\n//#endregion\n`;
const HELPER = (source) => `//#region src/runtime/${source}\nconst BATCH = 16;\n\t//#endregion\nself.postMessage("ready");\n`;
const before = { "index.html": '<script type="module" src="/_astro/index.AAAAAAAA.js"></script>', "benchmark/index.html": "<p>b</p>", "coi.js": "// coi\nself.x = 1;\n", "models/tiny.bin": "weights",
  "_astro/index.AAAAAAAA.js": 'new Worker("worker.BBBBBBBB.js")', "_astro/worker.BBBBBBBB.js": WORKER("helper.CCCCCCCC.js", "jobs.js"), "_astro/helper.CCCCCCCC.js": HELPER("jobs.js"),
  "_astro/simdkernel_plain.DfCGpm7G.wasm": Buffer.from([0, 97, 115, 109, 1, 255, 254]), "_astro/index.DDDDDDDD.css": "body{}" };
const after = { ...before, "index.html": '<script type="module" src="/_astro/index.EEEEEEEE.js"></script>', "_astro/index.AAAAAAAA.js": undefined, "_astro/worker.BBBBBBBB.js": undefined, "_astro/helper.CCCCCCCC.js": undefined,
  "_astro/index.EEEEEEEE.js": 'new Worker("worker.FFFFFFFF.js")', "_astro/worker.FFFFFFFF.js": WORKER("helper.GGGG-GG_.js", "jobs.ts"), "_astro/helper.GGGG-GG_.js": HELPER("jobs.ts") };
const build = (name, files) => written(name, Object.fromEntries(Object.entries(files).filter(([, text]) => text !== undefined)));
const A = build("before", before), B = build("after", after);
{
  const rows = compared(A, B), said = report(rows);
  assert.deepEqual(verdicts(rows), { "_astro/helper.*.js": "same code", "_astro/index.*.css": "same", "_astro/index.*.js": "same code", "_astro/simdkernel_plain.*.wasm": "same", "_astro/worker.*.js": "same code",
    "benchmark/index.html": "same", "coi.js": "same", "index.html": "same code" }, "the models are not compared; a page that names another script is the same code");
  assert.equal(said.same, true);
  assert.match(said.lines.at(-2), /^dist-same: 8 files: 4 the same to the byte, 4 the same code under another name or with other region lines, 0 different, 0 only in the first, 0 only in the second$/);
  assert.equal(said.lines.at(-1), "dist-same: the same code");
  assert.equal(report(compared(A, A)).lines.at(-2).includes("8 the same to the byte"), true, "a build against itself");
  // the folders of _astro/ alone: the pages and coi.js are not looked at
  assert.deepEqual(Object.keys(verdicts(compared(path.join(A, "_astro"), path.join(B, "_astro")))), ["helper.*.js", "index.*.css", "index.*.js", "simdkernel_plain.*.wasm", "worker.*.js"]);
  // the command line: 0 for the same code, and the words
  const run = spawnSync(process.execPath, [path.join(root, "tests/dist-same.mjs"), A, B], { encoding: "utf8" });
  assert.equal(run.status, 0, run.stderr);
  assert.match(run.stdout, /dist-same: the same code\n$/);
}

// ---- what is a difference, each alone
const differs = (name, change, file, { line, before: was, after: now } = {}, more = {}) => {
  const C = build(name, { ...after, ...change });
  const rows = compared(A, C), said = report(rows);
  const row = rows.find((one) => one.file === file);
  assert.ok(row, `${name}: ${file} is compared (${rows.map((one) => one.file)})`);
  assert.equal(said.same, false, name);
  assert.equal(said.lines.at(-1), "dist-same: NOT the same", name);
  for (const [other, verdict] of Object.entries({ [file]: "different", ...more })) assert.equal(rows.find((one) => one.file === other).verdict, verdict, `${name}: ${other}`);
  const others = rows.filter((one) => !(one.file in { [file]: 1, ...more }) && !["same", "same code"].includes(one.verdict));
  assert.deepEqual(others, [], `${name}: nothing else is said`);
  if (line !== undefined) assert.equal(row.line, line, `${name}: the first line that differs`);
  if (was) assert.match(row.before, was, name);
  if (now) assert.match(row.after, now, name);
  const run = spawnSync(process.execPath, [path.join(root, "tests/dist-same.mjs"), A, C], { encoding: "utf8" });
  assert.equal(run.status, 1, `${name}: the command line ends 1`);
  return said;
};
const worker = WORKER("helper.GGGG-GG_.js", "jobs.ts");
// (line 1: the region lines are not counted)
differs("one character", { "_astro/worker.FFFFFFFF.js": worker.replace("BATCH = 16", "BATCH = 17") }, "_astro/worker.*.js", { line: 1, before: /^const BATCH = 16;$/, after: /^const BATCH = 17;$/ });
differs("a parenthesis a cast left", { "_astro/index.EEEEEEEE.js": '(new Worker("worker.FFFFFFFF.js"))' }, "_astro/index.*.js", { line: 1, before: /^new Worker\("worker\.\*\.js"\)$/, after: /^\(new Worker\("worker\.\*\.js"\)\)$/ });
differs("a comment that moved", { "_astro/helper.GGGG-GG_.js": HELPER("jobs.ts").replace("const BATCH = 16;\n", "/** the most tokens */\nconst BATCH = 16;\n") }, "_astro/helper.*.js", { line: 1, after: /the most tokens/ });
differs("a line more at the end", { "_astro/helper.GGGG-GG_.js": `${HELPER("jobs.ts")}self.close();\n` }, "_astro/helper.*.js", { line: 3, before: /^$/, after: /^self\.close\(\);$/ });
differs("a built file's bytes", { "_astro/simdkernel_plain.DfCGpm7G.wasm": Buffer.from([0, 97, 115, 109, 1, 255, 253]) }, "_astro/simdkernel_plain.*.wasm", { line: 0, before: /^7 bytes$/ });
differs("a file served as it is", { "coi.js": "// @ts-check\n// coi\nself.x = 1;\n" }, "coi.js", { line: 1, before: /^\/\/ coi$/, after: /^\/\/ @ts-check$/ });
differs("a page", { "benchmark/index.html": "<p>c</p>" }, "benchmark/index.html", { line: 1 });
// a name that is written and is no file of the build stays as it is: a hash that is not this build's is a difference
differs("another build's file named", { "_astro/index.EEEEEEEE.js": 'new Worker("worker.BBBBBBBB.js")' }, "_astro/index.*.js", { before: /worker\.\*\.js/, after: /worker\.BBBBBBBB\.js/ });
// a region line is a line of its own: a line that only holds the word is code
differs("code that looks like a region line", { "_astro/helper.GGGG-GG_.js": HELPER("jobs.ts").replace('self.postMessage("ready");', 'self.postMessage("//#region");') }, "_astro/helper.*.js", { line: 2 });
differs("an empty line more at the end", { "_astro/helper.GGGG-GG_.js": `${HELPER("jobs.ts")}\n` }, "_astro/helper.*.js", { line: 4 });
{
  // (and a line that holds code before the word is code: what differs in it is said)
  const tail = (n) => HELPER("jobs.ts").replace('self.postMessage("ready");', `self.postMessage(${n}); //#region x`);
  const C = build("code before the word, before", { ...after, "_astro/helper.GGGG-GG_.js": tail(1) }), D = build("code before the word, after", { ...after, "_astro/helper.GGGG-GG_.js": tail(2) });
  assert.equal(compared(C, D).find((one) => one.file === "_astro/helper.*.js").verdict, "different");
}
// a long line (a minified page's script): the words are cut around where it differs
{
  const long = `${"a=1;".repeat(4000)}new Worker("worker.FFFFFFFF.js")${";b=2".repeat(4000)}`;
  const C = build("a long line, before", { ...after, "_astro/index.EEEEEEEE.js": long }), D = build("a long line, after", { ...after, "_astro/index.EEEEEEEE.js": long.replace(";b=2;b=2", ";b=3;b=2") });
  const row = compared(C, D).find((one) => one.verdict === "different");
  assert.ok(row.before.length < 200 && row.before.startsWith("…") && row.before.endsWith("…"), row.before);
  assert.ok(row.before.includes(";b=2;b=2") && row.after.includes(";b=3;b=2"));
}
// a file more, a file less (under their own names), and both at once
{
  const more = report(compared(A, build("a file more", { ...after, "_astro/extra.HHHHHHHH.js": "x" })));
  assert.equal(more.same, false);
  assert.ok(more.lines.includes("ONLY AFTER  _astro/extra.HHHHHHHH.js"), more.lines.join("\n"));
  const less = report(compared(A, build("a file less", { ...after, "_astro/index.DDDDDDDD.css": undefined })));
  assert.ok(less.lines.includes("ONLY BEFORE  _astro/index.DDDDDDDD.css"), less.lines.join("\n"));
  assert.match(less.lines.at(-2), /0 different, 1 only in the first, 0 only in the second$/);
  // (a chunk under another name is a file less and a file more: the names are what a Service Worker's families go by)
  const renamed = report(compared(A, build("a chunk renamed", { ...after, "_astro/helper.GGGG-GG_.js": undefined, "_astro/thread.GGGG-GG_.js": HELPER("jobs.ts") })));
  assert.ok(renamed.lines.includes("ONLY BEFORE  _astro/helper.CCCCCCCC.js") && renamed.lines.includes("ONLY AFTER  _astro/thread.GGGG-GG_.js"), renamed.lines.join("\n"));
}
// two files of one name without their hashes, a folder that is none, no arguments: said, with 2
{
  assert.throws(() => compared(A, build("two of one name", { ...after, "_astro/worker.ZZZZZZZZ.js": "x" })), /worker\.FFFFFFFF\.js and _astro\/worker\.ZZZZZZZZ\.js are one name without their hashes/);
  for (const args of [[], [A], [A, path.join(scratch, "nowhere")], [A, B, B]]) {
    assert.equal(spawnSync(process.execPath, [path.join(root, "tests/dist-same.mjs"), ...args], { encoding: "utf8" }).status, 2, `dist-same ${args.length} arguments`);
  }
  assert.equal(unhashed("benchmark-gpu.B-nxB-d4.js"), "benchmark-gpu.*.js");
  assert.equal(unhashed("coi.js"), "coi.js");
  assert.equal(unhashed("index.astro_astro_type_script_index_0_lang.Dx2kzFnp.js"), "index.astro_astro_type_script_index_0_lang.*.js");
}
console.log("dist-same-check: two builds of one code are the same code; 11 differences, a file more, a file less and a chunk renamed are each said");

// ---- tests/untyped.mjs: a .ts text with its types blanked, the same length and lines; a .js text as it is
{
  const ts = 'import type { Q } from "./q.ts";\nimport { a, type B } from "./x.ts";\ntype Job = Iterable<number>;\nexport function f(job: Job, n?: number): void {\n  return (job as any)!;\n}\n';
  const js = untyped(ts, "forward/x.ts");
  assert.equal(js.length, ts.length);
  assert.deepEqual(js.split("\n").map((line) => line.trimEnd()), ["", "import { a,        } from \"./x.ts\";", "", "export function f(job     , n         )       {", "  return (job       ) ;", "}", ""]);
  assert.equal(untyped(ts, "x.js"), ts, "a .js name: as it is");
  assert.equal(untyped("declare const a: number;\n", "x.d.ts"), "declare const a: number;\n", "a declaration file is nobody's source");
  assert.ok(typed("a/b.ts") && !typed("a/b.d.ts") && !typed("a/b.js") && !typed("a/b.tsx"));
  // what cannot be blanked is refused, as tsconfig.base.json's erasableSyntaxOnly refuses it
  for (const text of ["enum E { A }\n", "namespace N { export const a = 1; }\n", "class C { constructor(private x: number) {} }\n"]) assert.throws(() => untyped(text, "x.ts"), undefined, text);
  // (and Node says nothing of it being experimental into a tool's output)
  const run = spawnSync(process.execPath, ["-e", 'import("./tests/untyped.mjs").then((m) => console.log(m.untyped("const a: number = 1;", "a.ts")))'], { encoding: "utf8", cwd: root });
  assert.equal(run.stdout, "const a         = 1;\n");
  assert.equal(run.stderr, "", "no warning");
}
// ---- tests/types.mjs: an import that names types alone, and the anys that are written
{
  assert.deepEqual(typeOnlyImports('import { type A } from "./a.ts";\nimport { b, type C } from "./c.ts";\nimport type { D } from "./d.ts";\nimport {\n  type E,\n  type F,\n} from "./e.ts";\nimport "./g.ts";\nimport { h } from "./h.ts";\n'),
    ['import { type A } from "./a.ts";', 'import { type E, type F, } from "./e.ts";']);
  assert.deepEqual(anys('// any comment\nconst words = "any";\nconst a: any = 1, b = a as any;\nfunction f(x: any[], y: Record<string, any>): Promise<any> { return x as never; }\n// @ts-expect-error\nconst company = 1;\n', "x.ts"), { any: 5, off: 1 });
  assert.deepEqual(anys("const many = 1;\n", "x.ts"), { any: 0, off: 0 });
  // T368.8: the names a text uses and neither declares nor imports: a type's name is none, a name bound further in is none
  assert.deepEqual(ambient('import { a } from "./a.ts";\nimport type { T } from "./t.ts";\nlet own: T | Uint8Array;\nexport function f(given: T) {\n  for (const name of a) own = given;\n  return name + nowhere + fetch.length;\n}\n', "x.ts"),
    ["name", "nowhere", "fetch"]);
  assert.ok(AMBIENT.has("fetch") && !AMBIENT.has("name") && !AMBIENT.has("document"), "a worker's own names that the runtime means to use: not `name`, and nothing of a page");
}
console.log("dist-same-check: a .ts text is blanked to the same length and lines; a type's import, a written any and an ambient name are found");
fs.rmSync(scratch, { recursive: true, force: true });
console.log("dist-same-check: ok");
