// tests/runtime-move-check.mjs (T367.3.1): tests/runtime-move.mjs held to made-up texts, to a made-up tree, and to this
// tree as it is: what the tool would write of every file of the runtime, without writing it.
//
//   1. the six rules, each on a text of its own: what comes out, to the letter; what it does not know stays and is listed
//   2. a made-up tree moved and rewritten by the tool as it is run (node tests/runtime-move.mjs move|rewrite|left)
//   3. this tree, while its runtime is still served as it is: every file rewritten in memory. The result parses and
//      links in the form a bundler takes (tests/imports.mjs, bundled), every window reaches all of its folder with no
//      ring, each file takes the same names of the same neighbours as before, no line changed but a linking one, and
//      the lines left for a hand are the ones written down here (LEFT): a line more is a new thing for a hand, which
//      this check is then told of
// In a tree whose runtime a bundler links already there is nothing to rewrite: 3 says so (and the tool goes: T367.3.2).
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { importsOf, windowed } from "./imports.mjs";
import { BUILT, BUILT_TO, STAYS, TO, left, moves, rewritten } from "./runtime-move.mjs";
import { treeOf } from "./tree.mjs";

const TOOL = fileURLToPath(new URL("runtime-move.mjs", import.meta.url));
const ok = (said) => console.log(`ok: ${said}`);
const V = "${new URL(import.meta.url).search}", W = "${self.location.search}";

// ---- 1. the rules
{
  const one = (text, file = "w.js") => rewritten(text, file);
  // a neighbour's names, awaited: where they stood, the names and their line breaks kept, a rename as `as`
  assert.deepEqual(one(`// head\nconst { a, b: c,\n  d } = await import(new URL(\`x.js${V}\`, import.meta.url));\nexport const e = a + c + d;\n`),
    { text: `// head\nimport { a, b as c,\n  d } from "./x.js";\nexport const e = a + c + d;\n`, counts: { static: 1, list: 0, queue: 0, later: 0, worker: 0 } });
  assert.equal(one(`const { a } =\n  await import(new URL(\`../y/x.js${W}\`, import.meta.url));\n`, "w/part.js").text, `import { a } from "../y/x.js";\n`);
  assert.equal(one(`const { a } = await import(new URL(\`x.js${V}\`, import.meta.url));\n`, "deep/in/part.js").text, `import { a } from "./x.js";\n`, "a neighbour is beside the file that names it");
  // a window: the list goes, each module's names become an import of the folder's file, the queue goes
  const window = `// the window\nconst early = [];\nself.onmessage = (event) => early.push(event);\nconst modules = Object.fromEntries(["a", "b", "c"].map((name) =>\n  [name, import(new URL(\`w/\${name}.js${W}\`, import.meta.url))]));\n` +
    `const { one } = await modules.a;\n// (b's)\nconst { two, three: tre } = await modules.b;\nself.onmessage = ({ data }) => one(data, two, tre);\nearly.splice(0).forEach((event) => self.onmessage(event));\n`;
  assert.deepEqual(one(window), { text: `// the window\nimport { one } from "./w/a.js";\n// (b's)\nimport { two, three as tre } from "./w/b.js";\nself.onmessage = ({ data }) => one(data, two, tre);\n`,
    counts: { static: 2, list: 1, queue: 3, later: 0, worker: 0 } });
  assert.equal(one(window.replace("self.onmessage = (event)", "onmessage = (event)").replace("early.splice(0).forEach((event) => self.onmessage(event));", "for (const event of early.splice(0)) onmessage(event);")).counts.queue, 3, "the queue in its other spelling");
  // (a file with no list keeps a variable called early: it is the queue's only beside a window's list)
  assert.equal(one("const early = [];\nexport const e = early;\n").text, "const early = [];\nexport const e = early;\n");
  // a module asked for later, a worker: by a literal name
  assert.deepEqual(one(`const shaders = import(new URL(\`../s.js${V}\`, import.meta.url));\nasync function f() { return (await import(new URL(\`k.js${W}\`, import.meta.url))).k; }\n`, "g/d.js"),
    { text: `const shaders = import("../s.js");\nasync function f() { return (await import("./k.js")).k; }\n`, counts: { static: 0, list: 0, queue: 0, later: 2, worker: 0 } });
  assert.deepEqual(one(`const w = () => new Worker(new URL(\`../helper.js${W}\`, import.meta.url), { type: "module" });\n`, "w/weights.js"),
    { text: `const w = () => new Worker(new URL("../helper.js", import.meta.url), { type: "module" });\n`, counts: { static: 0, list: 0, queue: 0, later: 0, worker: 1 } });
  // at("…"), where the file has that helper and nowhere else
  const at = "const search = new URL(import.meta.url).search;\nconst at = (name) => new URL(`../${name}${search}`, import.meta.url);\n";
  const sections = `${at}const f = async () => import(at("forward.js"));\nconst w = new Worker(at("b/reader.js"), { type: "module" });\nconst k = fetch(at(name));\n`;
  assert.equal(one(sections, "b/sections.js").text, `${at}const f = async () => import("../forward.js");\nconst w = new Worker(new URL("../b/reader.js", import.meta.url), { type: "module" });\nconst k = fetch(at(name));\n`);
  assert.equal(one(`const f = async () => import(at("forward.js"));\n`, "b/other.js").text, `const f = async () => import(at("forward.js"));\n`, "at() of a file that does not define it is nobody's");
  // what a hand has to finish is listed, a comment is not, and a finished file has no line
  assert.deepEqual(left(one(sections, "b/sections.js").text).map(({ at: line }) => line), [1, 2, 5]);
  assert.deepEqual(left(`// fetch(new URL(\`x${W}\`, import.meta.url))\nfetch(new URL(\`\${name}${W}\`, import.meta.url));\nfetch(new URL(\`../models/\${m}\`, import.meta.url));\nimport(url);\nconst early = [];\n`).map(({ at: line }) => line), [2, 3, 4, 5]);
  assert.deepEqual(left(`import { a } from "./x.js";\nconst u = new URL("./built/k.wasm", import.meta.url);\nconst w = new Worker(new URL("../helper.js", import.meta.url), { type: "module" });\n` +
    `const m = await import("./forward.js");\nconst p = (url) => import(/* @vite-ignore */ url);\npool.search = null;\nif (pool.search) run();\nstate.initialized ??= init(data.search);\nstate.disabled = state.pageSwitches;  // not self.location.search: that is the worker's own URL\n`), []);
  // a form it does not know is an error of the reader's, not a file half rewritten
  assert.throws(() => one(`const m = await import(new URL("x.js", import.meta.url));\n`), /an import this check does not know/);
  // a file rewritten already is left as it is; one half rewritten is refused
  assert.equal(one(`import { a,\n  b } from "./x.js";\nexport const c = a + b;\n`).text, `import { a,\n  b } from "./x.js";\nexport const c = a + b;\n`);
  assert.throws(() => one(`import { a } from "./x.js";\nconst { b } = await import(new URL(\`y.js${V}\`, import.meta.url));\n`), /with a \?v=, in a tree whose files a bundler links/);
  // what moves, what stays, what is built
  for (const name of ["coi.js", "coi-test/sw.js", "coi-test/worker.js"]) assert.ok(STAYS(name), name);
  for (const name of ["worker.js", "worker/coi.js", "benchmark/gpu.js", "coi-tests/x.js"]) assert.ok(!STAYS(name), name);
  for (const name of ["simdkernel.so", "simdkernel_relaxed.wasmlib", "simdkernel_plain64.wasm", "ceilings_relaxed_shared.wasm", "engine.zip", "converter.zip"]) assert.ok(BUILT(name), name);
  for (const name of ["models/x.wasm", "other.zip", "worker.js", "simdkernel.txt", "coi.js"]) assert.ok(!BUILT(name), name);
  ok("the six rules on made-up texts: static imports where the awaited ones stood, the windows' lists and queues gone, later modules and workers by literal names; what is left is listed");
}

// ---- 2. the tool as it is run, on a made-up tree
{
  const scratch = path.join(fileURLToPath(new URL("..", import.meta.url)), ".tmp", "runtime-move-check", String(process.pid));
  fs.rmSync(scratch, { recursive: true, force: true });
  const files = {
    "public/forward.js": `const { J } = await import(new URL(\`jobs.js${V}\`, import.meta.url));\nexport const F = J;\n`,
    "public/jobs.js": "export const J = 1;\n", "public/deep/er/part.js": `const { J } = await import(new URL(\`../../jobs.js${V}\`, import.meta.url));\nexport const P = J;\nfetch(new URL(\`../x${W}\`, import.meta.url));\n`,
    "public/coi.js": "// the service worker\n", "public/coi-test/worker.js": "// its test\n", "public/models/m.js": "// a model's file that ends in .js\n", "public/models/t.bin": "",
    "public/simdkernel.so": "so", "public/simdkernel_plain.wasm": "wasm", "public/engine.zip": "zip", "public/llama2_numpy.py": "", "public/notes.txt": "",
  };
  for (const [name, text] of Object.entries(files)) { fs.mkdirSync(path.dirname(path.join(scratch, name)), { recursive: true }); fs.writeFileSync(path.join(scratch, name), text); }
  const tool = (...args) => execFileSync(process.execPath, [TOOL, ...args, "--root", scratch], { encoding: "utf8" });
  assert.deepEqual(moves(scratch, "public"), [["public/deep/er/part.js", `${TO}/deep/er/part.js`], ["public/engine.zip", `${BUILT_TO}/engine.zip`], ["public/forward.js", `${TO}/forward.js`], ["public/jobs.js", `${TO}/jobs.js`],
    ["public/simdkernel.so", `${BUILT_TO}/simdkernel.so`], ["public/simdkernel_plain.wasm", `${BUILT_TO}/simdkernel_plain.wasm`]]);
  assert.match(tool("move"), /3 files of public\/ to src\/runtime\/, 3 built files to src\/runtime\/built\//);
  const there = (folder) => fs.readdirSync(path.join(scratch, folder), { recursive: true }).filter((name) => fs.statSync(path.join(scratch, folder, name)).isFile()).map((name) => name.split(path.sep).join("/")).sort();
  assert.deepEqual(there("public"), ["coi-test/worker.js", "coi.js", "llama2_numpy.py", "models/m.js", "models/t.bin", "notes.txt"], "the Service Worker, its test page, the models and what is no runtime stay");
  assert.ok(!fs.existsSync(path.join(scratch, "public/deep")), "a folder the move emptied goes, and the folder above it");
  assert.deepEqual(there("src/runtime"), ["built/engine.zip", "built/simdkernel.so", "built/simdkernel_plain.wasm", "deep/er/part.js", "forward.js", "jobs.js"]);
  for (const [from, to] of [["public/forward.js", "src/runtime/forward.js"], ["public/simdkernel.so", "src/runtime/built/simdkernel.so"]]) assert.equal(fs.readFileSync(path.join(scratch, to), "utf8"), files[from], "moved as it is");
  assert.deepEqual(treeOf(scratch, undefined).folders, { runtime: "src/runtime", python: "public", built: "src/runtime/built", served: "public" }, "tests/tree.mjs finds the moved tree by looking");
  assert.equal(spawnSync(process.execPath, [TOOL, "move", "--root", scratch], { encoding: "utf8" }).status, 1, "a second move has nothing to move, and says so");
  assert.match(tool("left"), /^deep\/er\/part\.js:1: [^\n]*\ndeep\/er\/part\.js:3: [^\n]*\nforward\.js:1: [^\n]*\nruntime-move: 3 lines in 2 files for a hand\n$/, "before the rewrite every ?v= is a hand's");
  assert.match(tool("rewrite"), /3 files: 2 static imports, 0 windows' lists and 0 lines of their queues gone, 0 modules asked for later, 0 workers/);
  assert.equal(fs.readFileSync(path.join(scratch, "src/runtime/forward.js"), "utf8"), `import { J } from "./jobs.js";\nexport const F = J;\n`);
  assert.equal(fs.readFileSync(path.join(scratch, "src/runtime/deep/er/part.js"), "utf8"), `import { J } from "../../jobs.js";\nexport const P = J;\nfetch(new URL(\`../x${W}\`, import.meta.url));\n`);
  assert.match(tool("left"), /^deep\/er\/part\.js:3: fetch\(new URL\(`\.\.\/x\$\{self\.location\.search\}`, import\.meta\.url\)\);\nruntime-move: 1 lines in 1 files for a hand\n$/);
  assert.match(tool("rewrite"), /3 files: 0 static imports/, "a second rewrite finds nothing to rewrite");
  assert.equal(spawnSync(process.execPath, [TOOL, "tidy"], { encoding: "utf8" }).status, 2, "a verb it does not have: its usage");
  fs.rmSync(scratch, { recursive: true, force: true });
  ok("a made-up tree: the runtime's files and the built ones moved as they are, the Service Worker and the models left, the emptied folders gone, the rewrite in place and once");
}

// ---- 3. this tree
{
  const tree = treeOf();
  if (tree.bundled) {
    ok(`this tree's runtime is in ${tree.folders.runtime}/ and a bundler links it: nothing is left to move or rewrite (tests/runtime-move.mjs and this check go)`);
  } else {
    // the lines a hand has to finish after the rewrite, a file: how many (T367.3.2's commit of the hand edits is these,
    // the page's three places and the comments that the rewrite made untrue)
    const LEFT = { "benchmark/gpu/matvec.js": 1, "benchmark/sections.js": 4, "worker.js": 3, "worker/load.js": 1, "worker/pyodide.js": 1, "worker/sources.js": 1 };
    const served = tree.folders.runtime, all = moves(tree.root, served);
    const moved = all.filter(([, to]) => !to.startsWith(`${BUILT_TO}/`)).map(([from]) => from.slice(served.length + 1));
    assert.equal(moved.length, 63, `the runtime is ${moved.length} files (63 when this was written): a file more or less is a file the move takes or leaves`);
    assert.ok(moved.every((name) => name.endsWith(".js") && !STAYS(name)) && !moved.includes("coi.js"));
    const before = new Map(moved.map((name) => [name, fs.readFileSync(tree.runtime(name), "utf8")]));
    const total = { static: 0, list: 0, queue: 0, later: 0, worker: 0 }, after = new Map(), hand = {};
    const LINKING_OLD = /await import\(new URL\(|await modules\.\w+;|^const modules = Object\.fromEntries|\[name, import\(new URL\(|\bearly\b|import\(new URL\(`|new Worker\(new URL\(`|\bat\("|^const \{ [\w, :]+( \} =|,)?$|^  [\w, :]+\} =$|^  ("[\w-]+", )*"[\w-]+"\]\.map\(\(name\) =>$/;
    const LINKING_NEW = /^import \{|^  [\w, ]+\} from "\.\.?\/|^  [\w, ]+,$| from "\.\.?\/[\w./-]+\.js";$|import\("\.\.?\/[\w./-]+\.js"\)|new Worker\(new URL\("\.\.?\/[\w./-]+\.js", import\.meta\.url\)/;
    for (const [name, text] of before) {
      const { text: now, counts } = rewritten(text, name);
      for (const key of Object.keys(total)) total[key] += counts[key];
      after.set(name, now);
      // the same names of the same neighbours, under the same local names
      const names = (takes) => takes.flatMap(({ from, names: list }) => list.map(({ imported, local }) => `${from}: ${imported} as ${local}`)).sort();
      assert.deepEqual(names(importsOf(now, name, { bundled: true }).takes), names(importsOf(text, name, { bundled: false }).takes), `${name} takes other names once rewritten`);
      // no line changed but a linking one
      const gone = text.split("\n"), come = now.split("\n");
      for (const line of come) { const at = gone.indexOf(line); if (at >= 0) gone[at] = null; else assert.match(line, LINKING_NEW, `${name}: the rewrite wrote a line that links nothing: ${line}`); }
      for (const line of gone) if (line !== null) assert.match(line, LINKING_OLD, `${name}: the rewrite took out a line that links nothing: ${line}`);
      const lines = left(now);
      if (lines.length) hand[name] = lines.length;
    }
    assert.deepEqual(total, { static: 141, list: 5, queue: 9, later: 8, worker: 4 }, "what the rewrite does to this tree, counted (as T365 counted it: 106 neighbours and 35 of the windows' modules)");
    assert.deepEqual(hand, LEFT, "the lines of the rewritten tree that a hand has to finish are not the ones this check has written down");
    // every window reaches all of its folder, in the form a bundler takes, with no ring
    const read = (name) => { assert.ok(after.has(name), `${name} is no file of the runtime`); return after.get(name); };
    const list = (folder) => [...after.keys()].filter((name) => name.startsWith(`${folder}/`) && !name.slice(folder.length + 1).includes("/")).map((name) => name.slice(folder.length + 1));
    const windows = { "worker.js": {}, "forward.js": { more: ["jobs.js"] }, "shaders.js": {}, "gpu.js": {}, "benchmark/gpu.js": {} };
    let reached = 0;
    for (const [window, options] of Object.entries(windows)) reached += windowed(window, { read, list, bundled: true, ...options }).there.length;
    assert.equal(reached, 11 + 7 + 12 + 9 + 11, "the windows' modules");
    assert.ok(![...after.values()].some((text) => /^const modules = |await modules\./m.test(text)), "a window's list is left");
    const built = all.length - moved.length;
    ok(`this tree: ${moved.length} files of ${served}/ would move (and ${built} built files); rewritten in memory, they take the same names by ${total.static} static imports, ` +
      `${total.later} modules are asked for later and ${total.worker} workers started by literal names, the 5 windows reach their ${reached} modules, and ${Object.values(hand).reduce((a, b) => a + b, 0)} lines in ${Object.keys(hand).length} files are a hand's`);
  }
}
console.log("runtime-move-check: ok");
