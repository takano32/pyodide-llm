// tests/dist-rules-check.mjs (T367.3.1): tests/dist-check.mjs and tests/bundle.mjs held to made-up builds: what each
// must say of a build that is wrong in one way, and of one that is right. No build is needed (dist-check.mjs itself is
// run on the real one after `npm run build`).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { builtModule, forwardUrl } from "./bundle.mjs";
import { ROOT, distSaid, families, familyOf } from "./dist-check.mjs";
import { runtimeUrl, treeOf, RULE } from "./tree.mjs";

const scratch = path.join(fileURLToPath(new URL("..", import.meta.url)), ".tmp", "dist-rules-check", String(process.pid));
fs.rmSync(scratch, { recursive: true, force: true });
const written = (name, files) => {
  const folder = path.join(scratch, name);
  for (const [file, text] of Object.entries(files)) { fs.mkdirSync(path.dirname(path.join(folder, file)), { recursive: true }); fs.writeFileSync(path.join(folder, file), text); }
  return folder;
};
const ok = (said) => console.log(`ok: ${said}`);

// ---- the Service Worker's own rule, from its text
const coi = fs.readFileSync(treeOf().served("coi.js"), "utf8");
const family = familyOf(coi);
{
  const at = (name) => `https://site.test/pyodide-llm/_astro/${name}`;
  assert.equal(family(at("worker.DR4U5U7O.js")), at("worker.*.js"));
  assert.equal(family(at("worker.B-E8ZTYl.js")), at("worker.*.js"), "a hash with a - or a _ in it");
  assert.equal(family(at("simdkernel_relaxed_plain64.Dc20bOMt.wasm")), at("simdkernel_relaxed_plain64.*.wasm"));
  assert.equal(family(at("index.astro_astro_type_script_index_0_lang.BwOPHeO8.js")), at("index.astro_astro_type_script_index_0_lang.*.js"));
  assert.equal(family(at("worker-DR4U5U7O.js")), at("worker-DR4U5U7O.js"), "Vite's own name for a worker's file: no family");
  assert.throws(() => familyOf(coi.replace("function family(address)", "function kin(address)")), /coi\.js has no family\(address\)/);
  // (the function is the Service Worker's text and not a copy: another rule there is another answer here)
  assert.equal(familyOf(coi.replace(String.raw`/\/_astro\/(.+?)\.[\w-]{6,}\.(\w+)$/`, String.raw`/\/_astro\/(.+?)-[\w-]{6,}\.(\w+)$/`))(at("worker-DR4U5U7O.js")), at("worker.*.js"));
  assert.deepEqual(families(["worker.DR4U5U7O.js", "gpu.BNsfHYZd.js", "benchmark-gpu.DERyxEhn.js", "engine.D5queIE0.zip", "index.ChJdR91r.css", "index.BwOPHeO8.js"], family), []);
  assert.match(families(["worker-DR4U5U7O.js"], family).join("\n"), /^_astro\/worker-DR4U5U7O\.js: the Service Worker takes no name and hash apart in it[^\n]*never dropped for a newer one$/);
  const two = families(["gpu.BNsfHYZd.js", "gpu.DERyxEhn.js", "kept.BDa9_jbn.js"], family);
  assert.equal(two.length, 1);
  assert.match(two[0], /^_astro\/gpu\.BNsfHYZd\.js and _astro\/gpu\.DERyxEhn\.js: one family to the Service Worker \(_astro\/gpu\.\*\.js\): keeping one drops the other's copy$/);
  assert.equal(families(["shaders.C0Kxea-E.js", "shaders.C0Kxea-E.css"], family).length, 0, "the same name with another ending is another family");
  ok("coi.js's own family(), read from its text: a name without a hash apart has no family, two files of one name are one family");
}

// ---- a build of a tree whose runtime a bundler links
{
  const sources = { "src/runtime/forward.js": 'import { J } from "./jobs.js";\nexport const F = J;\n', "src/runtime/jobs.js": "export const J = 1;\n", "src/runtime/worker/part.js": "export const P = 2;\n",
    "src/runtime/built/simdkernel.so": "so", "src/runtime/built/simdkernel_plain.wasm": "wasm", "src/runtime/built/engine.zip": "zip", "src/python/llama2_numpy.py": "", "public/coi.js": coi, "public/coi-test/sw.js": "" };
  const good = { "index.html": "", "benchmark/index.html": "", "coi.js": coi, "coi-test/sw.js": "", "models/t.bin": "",
    "_astro/worker.DR4U5U7O.js": "bundled", "_astro/forward.Dqke8pTt.js": "export const F = 1;\n", "_astro/simdkernel.DtNlNAG4.so": "so", "_astro/simdkernel_plain.DfCGpm7G.wasm": "wasm", "_astro/engine.D5queIE0.zip": "zip", "_astro/index.ChJdR91r.css": "" };
  const build = (name, change = {}) => {
    const tree = written(name, { ...sources, ...Object.fromEntries(Object.entries({ ...good, ...change }).filter(([, text]) => text !== null).map(([file, text]) => [`dist/${file}`, text])) });
    return distSaid(path.join(tree, "dist"), treeOf(tree, { ...RULE }));
  };
  const right = build("right");
  assert.deepEqual(right.said, []);
  assert.match(right.looked, /^6 files of _astro\/, each a family of its own; 3 built files in it once each; none of 3 sources as it is; the root holds nothing else$/);
  const wrong = (name, change, ...patterns) => {
    const { said } = build(name, change);
    assert.equal(said.length, patterns.length, `${name}: ${said.join(" | ")}`);
    patterns.forEach((pattern, i) => assert.match(said[i], pattern, name));
  };
  wrong("a name without a family", { "_astro/helper-CZLQhea1.js": "x" }, /^_astro\/helper-CZLQhea1\.js: the Service Worker takes no name and hash apart/);
  wrong("two of a family", { "_astro/worker.AAAAAAAA.js": "another" }, /^_astro\/worker\.AAAAAAAA\.js and _astro\/worker\.DR4U5U7O\.js: one family/);
  wrong("a source as it is", { "_astro/part.B1B1B1B1.js": sources["src/runtime/worker/part.js"] }, /^_astro\/part\.B1B1B1B1\.js is src\/runtime\/worker\/part\.js as it is, not bundled: a URL made with a variable copies its whole folder/);
  wrong("a file as a data: URL", { "_astro/worker.DR4U5U7O.js": 'const u = "data:application/wasm;base64,AGFzbQ==";' }, /^_astro\/worker\.DR4U5U7O\.js holds a file as a data: URL \(vite\.build\.assetsInlineLimit: 0\)$/);
  wrong("a built file that is not there", { "_astro/engine.D5queIE0.zip": null }, /^src\/runtime\/built\/engine\.zip: no file of _astro\/: every built file is named once/);
  wrong("an old built file", { "_astro/simdkernel_plain.DfCGpm7G.wasm": "older" }, /^_astro\/simdkernel_plain\.DfCGpm7G\.wasm is not src\/runtime\/built\/simdkernel_plain\.wasm byte for byte/);
  // (simdkernel is no beginning of simdkernel_plain's name: each built file is looked for under its whole name)
  wrong("a built file's name that begins another's", { "_astro/simdkernel.DtNlNAG4.so": null, "_astro/simdkernel_x.DtNlNAG4.so": "so" }, /^src\/runtime\/built\/simdkernel\.so: no file of _astro\//);
  wrong("a runtime file at the root", { "worker.js": "as it was" }, /^dist\/worker\.js: nothing asks for it there \(the root holds _astro, index\.html, benchmark, coi\.js, coi-test, models\)/);
  wrong("an archive at the root", { "engine.zip": "zip" }, /^dist\/engine\.zip: nothing asks for it there/);
  wrong("an emptied folder at the root", { "worker/.keep": "" }, /^dist\/worker: nothing asks for it there/);
  wrong("no Service Worker", { "coi.js": null }, /^dist\/coi\.js is not there$/);
  wrong("no benchmark", { "benchmark/index.html": null }, /^dist\/benchmark\/index\.html is not there$/);
  assert.deepEqual(ROOT, ["_astro", "index.html", "benchmark", "coi.js", "coi-test", "models"]);
  assert.throws(() => distSaid(path.join(scratch, "right", "nothing"), treeOf(path.join(scratch, "right"), { ...RULE })), /no _astro\/ in it: npm run build/);
  ok("a bundled tree's build: right, and wrong in twelve ways, each said in a line of its own");

  // ---- a module as a build wrote it
  const dist = path.join(scratch, "right", "dist");
  assert.equal(builtModule("forward", dist).href, pathToFileURL(path.join(dist, "_astro/forward.Dqke8pTt.js")).href);
  assert.equal((await import(builtModule("forward", dist))).F, 1, "what is imported is the build's file, not the sources'");
  assert.throws(() => builtModule("forwar", dist), /neither _astro\/forwar\.<hash>\.js nor forwar\.js in it/);
  assert.throws(() => builtModule("kept", dist), /neither _astro\/kept\.<hash>\.js nor kept\.js/);
  assert.throws(() => builtModule("forward", path.join(scratch, "nowhere")), /no _astro\/ in it: npm run build/);
  const twice = path.join(written("two forwards", { "dist/_astro/forward.Dqke8pTt.js": "", "dist/_astro/forward.AAAAAAAA.js": "", "dist/_astro/forward-check.BBBBBBBB.js": "" }), "dist");
  assert.throws(() => builtModule("forward", twice), /2 files of _astro\/ are called forward \(forward\.AAAAAAAA\.js, forward\.Dqke8pTt\.js\)/);
  assert.equal(forwardUrl({}).href, runtimeUrl("forward.js").href, "no ENGINE_FROM: the sources");
  assert.equal(forwardUrl({ ENGINE_FROM: "sources" }).href, runtimeUrl("forward.js").href);
  assert.equal(forwardUrl({ ENGINE_FROM: "dist", ENGINE_DIST: dist }).href, pathToFileURL(path.join(dist, "_astro/forward.Dqke8pTt.js")).href);
  assert.throws(() => forwardUrl({ ENGINE_FROM: "bundle" }), /ENGINE_FROM=bundle: "dist"/);
  ok("a module as a build wrote it: the one chunk of its name, or why there is none");
}

// ---- a build of a tree whose runtime is served as it is: the families alone
{
  const tree = written("as it is", { "public/forward.js": "const { J } = await import(new URL(`jobs.js${new URL(import.meta.url).search}`, import.meta.url));\n", "public/coi.js": coi, "public/simdkernel.so": "so", "public/llama2_numpy.py": "",
    "dist/_astro/index.BwOPHeO8.js": "", "dist/_astro/index.ChJdR91r.css": "", "dist/forward.js": "export const F = 2;\n", "dist/worker.js": "", "dist/coi.js": coi, "dist/index.html": "" });
  const said = distSaid(path.join(tree, "dist"), treeOf(tree, { ...RULE }));
  assert.deepEqual(said.said, []);
  assert.match(said.looked, /^2 files of _astro\/, each a family of its own \(the runtime is served as it is from public\/: the root is not looked at\)$/);
  fs.writeFileSync(path.join(tree, "dist/_astro/index.AAAAAAAA.js"), "");
  assert.equal(distSaid(path.join(tree, "dist"), treeOf(tree, { ...RULE })).said.length, 1, "two of a family are said there too");
  assert.equal((await import(builtModule("forward", path.join(tree, "dist")))).F, 2, "a build that serves forward.js as it is: that file");
  ok("a tree that serves its runtime as it is: the families are held, the root is said not to be looked at");
}
fs.rmSync(scratch, { recursive: true, force: true });
console.log("dist-rules-check: ok");
