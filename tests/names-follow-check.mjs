// T.438: a file named in another's text is named again there when it changes. T.367.4's "the site was updated" rests on
// it (src/runtime/updated.ts: a module or a worker that fails to come is taken for an update where the file that names it,
// asked again, is gone): a built file's name is its text's hash, and a file that names another holds that name in its
// text, so a change anywhere moves every name up to the page's scripts. Nothing held that but the hand of T.367.4 (seven
// parts changed one at a time); a bundler or a naming that hashed a file without the names in it would leave an old page
// a namer that is still there, and the line would say nothing.
// So: the site built once as it is, and once more for every script of _astro/ that another file of _astro/ names, with that
// one's text changed after it is rendered (a plugin's renderChunk, in the page's build and in every worker's, puts a
// statement at its end: the page's minifier drops a comment). Each time:
//   - the changed file's name is another;
//   - and every file of _astro/ whose text names a file whose name changed has a name of its own that changed too, to the
//     page's scripts (the HTML, which no hash names, is not held);
//   - and no other name changed.
// Not held: the built files that are no scripts (the kernels, the Python archives, the CSS), whose text no plugin of the
// build writes: their names are their bytes' hash as Vite makes it, and their namers are held as any other's.
// About 15 builds of 12 s each: the full suite (and by hand, after a change to the bundler or the names: astro.config.mjs).
//   node tests/names-follow-check.mjs [--broken]     (needs `make kernels`; --broken: a namer's name put back as it was
//   in one of the changed builds, which is to fail)
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { build } from "astro";
import { leave, scratch } from "./leave.mjs";

const root = path.resolve(new URL("..", import.meta.url).pathname);
const broken = process.argv.includes("--broken");
const work = scratch("names-follow-");

// the site built into a folder of work's, with the script of the file touched (none: as it is); its _astro/ files by name.
// (A script is known by its chunk's name, what is before the hash, and its side: two are called gpu, the model's GPU worker
// and /benchmark/'s, whose name carries "benchmark-" before it: astro.config.mjs)
async function built(folder, touched) {
  const benchmark = touched?.startsWith("benchmark-"), name = touched?.replace(/^benchmark-/, "").replace(/\.[^.]+\.js$/, "");
  const touch = { name: "names-follow", renderChunk: (code, chunk) => (touched && chunk.name === name && Boolean(chunk.facadeModuleId?.includes("/runtime/benchmark/")) === benchmark
    ? `${code}\nglobalThis.touched = "T.438";\n` : null) };
  const outDir = path.join(work, folder);
  await build({ root, outDir, logLevel: "error", vite: { plugins: [touch], worker: { plugins: () => [touch] } } });
  const at = path.join(outDir, "_astro");
  return new Map(fs.readdirSync(at).sort().map((name) => [name, fs.readFileSync(path.join(at, name), "latin1")]));
}
// (a file's name without its hash)
const chunkOf = (file) => file.replace(/\.[^.]+\.[a-z]+$/, "");

const base = await built("base", null);
const scripts = [...base.keys()].filter((name) => name.endsWith(".js"));
const named = scripts.filter((name) => [...base].some(([other, text]) => other !== name && text.includes(name)));
assert.ok(named.length >= 10, `scripts named by another: ${named.join(", ")}`);
console.log(`names-follow-check: ${base.size} files built, ${named.length} scripts named by another: ${named.join(", ")}`);

let held = 0;
for (const [n, file] of named.entries()) {
  const now = await built(`touched-${n}`, file);
  // (--broken: the first namer of the touched file has the name it had, as a bundler that hashed it without its names
  // would leave it)
  if (broken && n === 0) {
    const namer = [...base].find(([other, text]) => other !== file && text.includes(file))[0];
    const moved = [...now.keys()].find((name) => !base.has(name) && name.startsWith(namer.replace(/\.[^.]+\.js$/, ".")));
    assert.ok(moved, `--broken: ${namer} kept its name, and nothing is broken`);
    now.set(namer, now.get(moved));
    now.delete(moved);
  }
  const gone = [...base.keys()].filter((name) => !now.has(name));
  assert.ok(gone.includes(file), `${file} touched: its name is the same`);
  assert.equal(now.size, base.size, `${file} touched: ${now.size} files built, not ${base.size}`);
  // every namer of a name that changed changed too; and nothing else did
  const follow = new Set([file]);
  for (let grew = true; grew;) {
    grew = false;
    for (const [other, text] of base) {
      if (!follow.has(other) && [...follow].some((name) => text.includes(name))) {
        follow.add(other);
        grew = true;
      }
    }
  }
  const left = [...follow].filter((name) => now.has(name));
  assert.deepEqual(left, [], `${file} touched: files that name a changed one kept their names: ${left.join(", ")}`);
  assert.deepEqual(gone.filter((name) => !follow.has(name)), [], `${file} touched: names that changed with nothing they name changed`);
  console.log(`ok: ${file} touched: ${gone.length} names changed, its own and its namers' (${[...follow].filter((name) => name !== file).map(chunkOf).join(", ")})`);
  fs.rmSync(path.join(work, `touched-${n}`), { recursive: true, force: true });
  held += 1;
}
console.log(`names-follow-check: ok, ${held} scripts touched one at a time`);
await leave(0);
