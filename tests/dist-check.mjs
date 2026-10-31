// tests/dist-check.mjs (T.367.3.1): what a build wrote, held to what the Service Worker does with a file's name.
//
//   node tests/dist-check.mjs [<dist>] [--tree <the tree it was built of>]      after `npm run build` (dist/, this tree)
//
// public/coi.js keeps a copy of this site's files for a visit without a network (T.111), and drops the old copy of a file
// when it keeps a new one: two addresses are "the same file" where family() gives them one name, and a file under
// /_astro/ is <name>.<hash>.<ext> to it. A bundler's names are not the Service Worker's business until they disagree
// with that, and then nothing fails where anyone looks (T.365 measured both):
//   - a name family() does not take apart is a family of its own for ever: every deployment leaves one more copy
//     (Vite names a worker's files <name>-<hash>.js unless it is told otherwise: 27 files);
//   - two files of one build in one family put each other's copy out: one is kept, the other's copy is deleted, and
//     the page without a network finds a file missing (two entries called gpu.js, one of the model page and one of
//     /benchmark/; a module that the page and a worker both ask for).
// So, with coi.js's own family() (its text, run here: not a second copy of the rule):
//   1. every file of dist/_astro/ has a family that is not its own name;
//   2. no two files of dist/_astro/ have the same family;
// and what a bundler does without a word where a URL is written the wrong way:
//   3. no file of dist/_astro/ is a source of the runtime as it is (a URL with a variable in it copies its folder), and
//      no worker's text holds a file as a data: URL;
//   4. every built file the runtime names (src/runtime/built/) is in dist/_astro/ once, with the same bytes;
//   5. dist's root holds the pages, _astro/ and what is served as it is (the Service Worker and its test page, the
//      models), and nothing else: no source, no built file, no archive that nobody asks for.
//   6. every copyright line and licence text of a source (a /*! … */ comment: a bundler drops every other) is in each
//      file of dist/_astro/ that holds the source's code, whole; and no source keeps a copyright line in a // comment;
//   7. the built key module (the model's worker's), the built GPU worker and the built shaders module (/benchmark/'s)
//      give the same deviceKey() as the sources, for 192 made-up devices.
//   8. no file of dist/_astro/ asks for a module of Node's, or is the bundler's stand-in for one: the runtime is a
//      browser's (T.367.5: what Node's tests need of a worker is tests/helper-node.mjs's to give).
//   9. (T.445) no file of the runtime in dist/_astro/ reads a name it declares nowhere, other than the ambient names the
//      runtime uses (tests/types.mjs's AMBIENT): a name the bundler dropped from its declaration fails only in the
//      branch that reads it, and the bundled GPU worker's branches (ternary, int6, outlier columns) run on no GPU of
//      CI's. File-wide, not by scope (the bundles use var, which tests/names.mjs's scopes refuse): a name declared in
//      one function and read in another is not seen.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { pathToFileURL } from "node:url";
import { treeOf } from "./tree.mjs";
import { parse } from "@babel/parser";
import { builtinModules } from "node:module";
import { AMBIENT } from "./types.mjs";

const ORIGIN = "https://site.test", BASE = "/pyodide-llm/";
/** what dist's root may hold: the pages, the bundler's folder, and what is served as it is */
/** the chunks that are fetched only when they are used (dist-check's rule 6) */
export const LAZY = ["forward", "kept", "key", "gpu", "shaders"];
export const ROOT = ["_astro", "index.html", "benchmark", "coi.js", "coi-test", "models"];

/** coi.js's own family(), from its text: the function as the Service Worker runs it, on this origin */
export function familyOf(coi) {
  const text = /^function family\(address\) \{\n[\s\S]*?\n\}$/m.exec(coi)?.[0];
  if (!text) throw new Error("coi.js has no family(address) at its top level: what says which copies are versions of one file?");
  return vm.runInNewContext(`(${text})`, { URL, self: { location: { origin: ORIGIN } } });
}

/** What is wrong with a build, as lines (none: nothing). files: the names in its _astro/; family: coi.js's. */
export function families(files, family) {
  const said = [], groups = new Map();
  for (const name of files) {
    const address = `${ORIGIN}${BASE}_astro/${name}`, f = family(address);
    if (f === address) said.push(`_astro/${name}: the Service Worker takes no name and hash apart in it (<name>.<hash>.<ext>): its copy is never dropped for a newer one`);
    groups.set(f, [...(groups.get(f) ?? []), name]);
  }
  for (const [f, names] of groups) {
    if (names.length > 1) said.push(`${names.map((name) => `_astro/${name}`).join(" and ")}: one family to the Service Worker (${f.slice(ORIGIN.length + BASE.length)}): keeping one drops the other's copy`);
  }
  return said;
}

/** every file under a folder, as its path from it with "/" between */
const under = (folder) => (!fs.existsSync(folder) ? [] : fs.readdirSync(folder, { recursive: true, withFileTypes: true }).filter((entry) => entry.isFile())
  .map((entry) => path.relative(folder, path.join(entry.parentPath, entry.name)).split(path.sep).join("/")).sort());

// T.448: Node's own modules by name (worker_threads, fs, ...), as a pattern
const NODE_MODULES = builtinModules.filter((name) => !name.startsWith("_") && !name.includes("/")).map((name) => name.replace(/^node:/, "")).join("|");

/** T.445: the names a script reads that it declares nowhere, file-wide (a var, a let or a const, a function or a class
 * and its parameters, a catch's, an import's) */
export function undeclared(text) {
  const declared = new Set(), read = new Set();
  const declare = (pattern) => {
    if (!pattern) return;
    if (pattern.type === "Identifier") declared.add(pattern.name);
    else if (pattern.type === "ObjectPattern") pattern.properties.forEach((one) => declare(one.type === "RestElement" ? one.argument : one.value));
    else if (pattern.type === "ArrayPattern") pattern.elements.forEach(declare);
    else if (pattern.type === "AssignmentPattern") declare(pattern.left);
    else if (pattern.type === "RestElement") declare(pattern.argument);
  };
  // (an identifier that names no binding: a property, a key, a label, an exported or imported name, import.meta)
  const named = (parent, key) => ((parent.type === "MemberExpression" || parent.type === "OptionalMemberExpression") && key === "property" && !parent.computed)
    || (/^(ObjectProperty|ObjectMethod|ClassMethod|ClassProperty|ClassPrivateProperty)$/.test(parent.type) && key === "key" && !parent.computed)
    || /^(LabeledStatement|BreakStatement|ContinueStatement|MetaProperty)$/.test(parent.type)
    || (parent.type === "ExportSpecifier" && key === "exported") || (parent.type === "ImportSpecifier" && key === "imported");
  const walk = (node, parent, key) => {
    if (node.type === "VariableDeclarator") declare(node.id);
    else if (/^(FunctionDeclaration|FunctionExpression|ArrowFunctionExpression|ClassDeclaration|ClassExpression|ObjectMethod|ClassMethod)$/.test(node.type)) {
      if (node.id) declared.add(node.id.name);
      (node.params ?? []).forEach(declare);
    } else if (node.type === "CatchClause") declare(node.param);
    else if (/^Import(Default|Namespace)?Specifier$/.test(node.type)) declared.add(node.local.name);
    if (node.type === "Identifier" && !(parent && named(parent, key))) read.add(node.name);
    for (const [field, value] of Object.entries(node)) {
      if (field === "loc") continue;
      for (const child of Array.isArray(value) ? value : [value]) if (child && typeof child.type === "string") walk(child, node, field);
    }
  };
  // (errorRecovery: a name declared twice is no failure of this rule)
  walk(parse(text, { sourceType: "module", errorRecovery: true }).program, null, null);
  return [...read].filter((one) => !declared.has(one)).sort();
}

/** What is wrong with dist (a folder) as a build of tree, as lines; and what was looked at, in words. */
export function distSaid(dist, tree = treeOf()) {
  const astro = path.join(dist, "_astro");
  if (!fs.existsSync(astro)) throw new Error(`${dist}: no _astro/ in it: npm run build`);
  const files = fs.readdirSync(astro).sort();
  const said = families(files, familyOf(fs.readFileSync(tree.served("coi.js"), "utf8")));
  // 3: a source as it is, a file as a data: URL
  const bytes = new Map(files.map((name) => [name, fs.readFileSync(path.join(astro, name))]));
  const sources = under(tree.runtime()).filter((name) => /\.(js|ts)$/.test(name) && !name.endsWith(".d.ts") && !name.startsWith(`${path.relative(tree.runtime(), tree.built())}/`));
  for (const source of sources) {
    const text = fs.readFileSync(tree.runtime(source));
    for (const [name, got] of bytes) if (got.equals(text)) said.push(`_astro/${name} is ${tree.folders.runtime}/${source} as it is, not bundled: a URL made with a variable copies its whole folder (new URL("./literal", import.meta.url))`);
  }
  for (const [name, got] of bytes) {
    if (name.endsWith(".js") && /["'`]data:application\/(wasm|zip|octet-stream)/.test(got.toString("latin1"))) said.push(`_astro/${name} holds a file as a data: URL (vite.build.assetsInlineLimit: 0)`);
  }
  // (the workers are not minified: the owner's decision, 2026-10-10. A minified file is a few lines of thousands of
  // characters; what the bundler prints unminified has a statement a line)
  for (const [name, got] of bytes) {
    if (!/^(worker|forward|helper|gpu|benchmark-[\w-]+)\.[\w-]{6,}\.js$/.test(name)) continue;
    const lines = got.toString("utf8").split("\n").length;
    if (got.length / lines > 200) said.push(`_astro/${name} is minified (${Math.round(got.length / lines)} characters a line): the workers are served as the bundler prints them (vite.worker's output.minify: false)`);
  }
  // 6 (T.367.3's review): the parts that are read only when they are used stay chunks of their own. A static import of one
  // (or a bundler's choice to fold a small one in) puts it into the worker's text, and then every visit fetches and
  // evaluates it: the first request list is longer by it and nothing else fails. (key.js is the shaders' hash for the
  // device's key, read when a GPU is there; kept.js the store of converted models; gpu and shaders the GPU's worker.)
  for (const lazy of LAZY) {
    if (!files.some((name) => new RegExp(`^${lazy}\\.[\\w-]{6,}\\.js$`).test(name))) said.push(`_astro/${lazy}.<hash>.js is not there: ${lazy}.js was bundled into another file, and is fetched on every visit now, used or not (a static import of it, or a bundler's choice)`);
  }
  // 6: the notices. A bundler prints a file anew and drops its comments but the ones written /*! … */: the copyright
  // lines and licence texts of what was taken from other projects (policy 10; the shaders' parts) must be such
  // comments, or the site serves the code without them (the first bundled build served none of 25). Every notice of a
  // source is in every file of _astro/ that holds that source's code (the bundler marks it //#region <the source>).
  const words = (comment) => comment.replace(/^\/\*!/, "").replace(/\*\/$/, "").split("\n").map((line) => line.replace(/^\s*\* ?/, "")).join(" ").replace(/\s+/g, " ").trim();
  const noticesOf = (text) => (text.match(/\/\*![\s\S]*?\*\//g) ?? []).map(words);
  const chunks = [...bytes].filter(([name]) => name.endsWith(".js")).map(([name, got]) => ({ name, text: got.toString("utf8") })).map((chunk) => ({ ...chunk, notices: noticesOf(chunk.text), owed: new Map() }));
  let notices = 0;
  for (const source of sources) {
    const text = fs.readFileSync(tree.runtime(source), "utf8"), where = `${tree.folders.runtime}/${source}`;
    for (const line of text.replace(/\/\*![\s\S]*?\*\//g, "").split("\n")) {
      if (/^\s*(\/\/|\/\*|\*).*\bCopyright\b.*\b(19|20)\d\d\b/.test(line)) said.push(`${where}: a copyright line in a comment that a bundler drops: ${line.trim().slice(0, 80)} (the copyright lines and the licence text as /*! … */)`);
    }
    const mine = noticesOf(text);
    if (!mine.length) continue;
    const holding = chunks.filter((chunk) => chunk.text.includes(`//#region ${where}\n`));
    if (!holding.length) said.push(`${where} has ${mine.length} notices, and no file of _astro/ is marked as holding its code (//#region): are the workers minified, or the file no longer linked?`);
    for (const chunk of holding) for (const notice of mine) {
      notices += 1;
      chunk.owed.set(notice, [...(chunk.owed.get(notice) ?? []), where]);
    }
  }
  // (as many times as its sources have it: the same licence stands over several pieces of code, and one copy cut short
  // must not be covered by another's)
  for (const chunk of chunks) for (const [notice, owing] of chunk.owed) {
    const there = chunk.notices.filter((one) => one === notice).length;
    if (there < owing.length) said.push(`_astro/${chunk.name} holds the code of ${[...new Set(owing)].join(", ")} with ${there} of the ${owing.length} notices "${notice.slice(0, 60)}…" (${notice.length} characters): served code keeps the copyright line and the whole licence text`);
  }
  // 8 (T.367.5): Node's modules. A file of the runtime that imports one (helper.js had a branch for worker_threads) is
  // built with a warning and no error: the bundler puts an empty module in its place, a chunk of its own, and a
  // fragment of its runtime that the importer then fetches on every visit.
  for (const [name, got] of bytes) {
    if (!name.endsWith(".js")) continue;
    // (T.448: by its node: name, or by its bare name, which Node takes as well: worker_threads)
    const asked = new RegExp(`\\b(?:from|import)\\s*\\(?\\s*["']((?:node:)?(?:${NODE_MODULES})(?:/[\\w/]*)?)["']`).exec(got.toString("utf8"))?.[1];
    if (asked) said.push(`_astro/${name} asks for ${asked}: a module of Node's in a browser's file (what the tests need of Node is theirs to give: tests/helper-node.mjs)`);
    if (/^(__vite-browser-external|rolldown-runtime)\./.test(name)) said.push(`_astro/${name}: the bundler's stand-in for a module of Node's, or its own runtime for one: a file of the runtime imports a module a browser does not have`);
  }
  // 9 (T.445): names read and declared nowhere in a file of the runtime (the pages' scripts read the DOM's names)
  for (const [name, got] of bytes) {
    if (!name.endsWith(".js") || name.includes(".astro_astro_type_script")) continue;
    const unknown = undeclared(got.toString("utf8")).filter((one) => !AMBIENT.has(one));
    if (unknown.length) said.push(`_astro/${name} reads ${unknown.join(", ")}, declared nowhere in it and no ambient name of the runtime (tests/types.mjs's AMBIENT): a declaration the bundler dropped, or a new ambient name to add there`);
  }
  // 4: the built files, once each, the same bytes
  // (T.628.1: a module written into it, src/runtime/built/facts.ts, is bundled as a neighbour is, and fetched by no name)
  const built = under(tree.built()).filter((file) => file !== "facts.ts");
  for (const file of built) {
    const { name: stem, ext } = path.parse(file);
    const there = files.filter((name) => name.startsWith(`${stem}.`) && name.endsWith(ext) && /^[\w-]{6,}$/.test(name.slice(stem.length + 1, name.length - ext.length)));
    if (there.length !== 1) said.push(`${tree.folders.built}/${file}: ${there.length ? `${there.length} files of _astro/ (${there.join(", ")})` : "no file of _astro/"}: every built file is named once, by its literal name`);
    else if (!bytes.get(there[0]).equals(fs.readFileSync(tree.built(file)))) said.push(`_astro/${there[0]} is not ${tree.folders.built}/${file} byte for byte: an old build of one of them`);
  }
  // 5: the root
  for (const name of fs.readdirSync(dist).sort()) {
    if (!ROOT.includes(name)) said.push(`${path.basename(dist)}/${name}: nothing asks for it there (the root holds ${ROOT.join(", ")}): a file of ${tree.folders.served}/ that is no longer served as it is?`);
  }
  for (const name of ["index.html", "benchmark/index.html", "coi.js", "_astro"]) if (!fs.existsSync(path.join(dist, name))) said.push(`${path.basename(dist)}/${name} is not there`);
  // (T.383: and no bytecode Python left beside a source, anywhere: the zips leave it out, and a build that copies a
  // folder of Python would serve it)
  for (const name of under(dist)) if (/(^|\/)__pycache__\/|\.pyc$/.test(name)) said.push(`${path.basename(dist)}/${name}: Python's bytecode is served`);
  return { said, files, looked: `${files.length} files of _astro/, each a family of its own; ${built.length} built files in it once each; none of ${sources.length} sources as it is; ${notices} notices where their code is; the root holds nothing else, and no bytecode` };
}

/** The 192 made-up devices of tests/unchanged.mjs's shaders section: the device's key of each, by a module's deviceKey() */
export function keysOf(module) {
  const keys = {};
  for (const packed of [false, true]) {
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: { userAgent: "UA", gpu: { wgslLanguageFeatures: new Set(packed ? ["packed_4x8_integer_dot_product"] : []) } } });
    for (const f16 of [false, true]) for (const subgroups of [false, true]) for (const memory of [16384, 32768, 49152, 65536]) for (const threads of [128, 256, 1024]) for (const ternary of [false, true]) {
      const device = { info: { vendor: "v", architecture: "a", device: "d", description: "x" }, features: new Set([...(f16 ? ["shader-f16"] : []), ...(subgroups ? ["subgroups"] : [])]),
        limits: { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup: threads, maxComputeWorkgroupSizeX: threads } };
      keys[JSON.stringify([packed, f16, subgroups, memory, threads, ternary])] = module.deviceKey(device, device, ternary ? "ternary" : null);
    }
  }
  return keys;
}

/** 7: the model's worker reads the device's key from one built module (key) and the GPU's worker
 * computes it in its own file (gpu: T.367.5, the shaders' parts linked into it), each with its own copy of the shaders'
 * texts: what the page remembers of a device under the one's key is looked up under the other's, so the two and the
 * sources must give the same key for every device. And shaders, /benchmark/'s module of all of them, as the sources.
 * (gpu.<hash>.js is the worker itself, and a build keeps no export of a worker's entry: its text is imported here
 * with one line after it, which exports the deviceKey of its copy of shaders/key.js. It sets its receiver as it
 * loads and does nothing else.) Lines, as distSaid(). */
export async function keysSaid(dist, tree = treeOf()) {
  const { builtModule } = await import("./bundle.mjs");
  const said = [], NAMES = ["key", "gpu", "shaders"];
  const sources = keysOf(await import(tree.runtimeUrl("shaders.js")));
  // (a worker's global, which the worker's file assigns: a name Node does not have)
  const had = Object.getOwnPropertyDescriptor(globalThis, "onmessage");
  if (!had) Object.defineProperty(globalThis, "onmessage", { configurable: true, get: () => undefined, set() {} });
  const worker = async (url) => {
    const text = fs.readFileSync(url, "utf8"), region = /\/\/#region [^\n]*\/shaders\/key\.[jt]s\n([\s\S]*?)\n\/\/#endregion/.exec(text)?.[1];
    const name = /^function (deviceKey(?:\$\d+)?)\(/m.exec(region ?? "")?.[1];
    if (!name || /^\s*(import|export)\b/m.test(text)) return {};  // (no key.js in it, or a file that is not whole by itself)
    return import(`data:text/javascript;base64,${Buffer.from(`${text}\nexport { ${name} as deviceKey };\n`).toString("base64")}`);
  };
  // (T.448: the stand-in goes whatever happens: a module that throws as it loads leaves no global behind)
  try {
  for (const name of NAMES) {
    const url = builtModule(name, dist), module = await (name === "gpu" ? worker(url) : import(url));
    if (typeof module.deviceKey !== "function") {
      said.push(`_astro/${name}.<hash>.js has no deviceKey to ask: the key the built file gives a device cannot be held to the sources' (${name === "gpu" ? "the GPU's worker takes it from shaders/key.js, linked into its one file" : "an export of the module"})`);
      continue;
    }
    const built = keysOf(module), differ = Object.keys(sources).filter((device) => built[device] !== sources[device]);
    if (differ.length) said.push(`_astro/${name}.<hash>.js gives another deviceKey() than the sources for ${differ.length} of ${Object.keys(sources).length} made-up devices (the first: ${differ[0]}: ${built[differ[0]]} for ${sources[differ[0]]})`);
  }
  } finally {
    if (!had) delete globalThis.onmessage;
  }
  return { said, looked: `; the device's key the same from the built ${NAMES.join(", ")} modules and the sources, for ${Object.keys(sources).length} devices (${new Set(Object.values(sources)).size} keys)` };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const at = args.indexOf("--tree");
  const tree = at < 0 ? treeOf() : treeOf(args.splice(at, 2)[1]);
  const dist = path.resolve(args[0] ?? path.join(tree.root, "dist"));
  const first = distSaid(dist, tree), keys = await keysSaid(dist, tree);
  const said = [...first.said, ...keys.said], looked = `${first.looked}${keys.looked}`;
  assert.deepEqual(said, [], `dist-check: ${said.length} thing${said.length > 1 ? "s" : ""} wrong with ${dist}:\n${said.join("\n")}\n`);
  console.log(`dist-check: ok (${looked})`);
}
