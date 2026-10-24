// tests/bundle.mjs (T.367.3.1): a module of the runtime as a build wrote it, for the tools that measure or check what a
// visitor's browser really runs and not the sources: a bundler rewrites constant expressions and prints every function
// anew even where it does not minify (T.365: `if (JOB_TABLE <= JOBS * 4 || JOB_TABLE % 8 || …)` came out as
// `if (2056 % 8 || false)`), so "the bundle is the sources' text" is nothing a tool may assume.
//
//   builtModule("forward")            the file: URL of dist/_astro/forward.<hash>.js of this tree's build: the chunk a
//   builtModule("forward", dist)      worker asks for later by `import("./forward.js")`, which keeps the module's
//                                     exports under their names. One file, or an error that says how many there are.
//                                     In a build of a tree whose runtime is served as it is: dist/forward.js itself.
//   ENGINE_FROM=dist [ENGINE_DIST=<a dist>]   tests/engine.mjs (forward-check, threads-check and every tool on it)
//                                     reads forward.js from there instead of from the sources: forwardUrl() below
//
// Only what a worker imports dynamically is a chunk with its own exports (forward, shaders, key, kept): a module that
// is linked statically is inside its importer's file, under whatever names the bundler gave it. A worker's entry is a
// file too (helper, gpu, worker), with its entry's exports: helper.<hash>.js is the software thread as a browser runs
// it, jobs.js linked into it, and tests/helper-node.mjs starts it in Node beside a bundled forward (T.367.5).
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { runtimeUrl, treeOf } from "./tree.mjs";

export function builtModule(name, dist = path.join(treeOf().root, "dist")) {
  dist = path.resolve(dist);
  const astro = path.join(dist, "_astro");
  if (!fs.existsSync(astro)) throw new Error(`${dist}: no _astro/ in it: npm run build`);
  const chunk = new RegExp(`^${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\.[\\w-]{6,}\\.js$`);
  const found = fs.readdirSync(astro).filter((file) => chunk.test(file));
  if (found.length > 1) throw new Error(`${dist}: ${found.length} files of _astro/ are called ${name} (${found.join(", ")}): which one a page runs is not said by the name`);
  if (found.length) return pathToFileURL(path.join(astro, found[0]));
  const asItIs = path.join(dist, `${name}.js`);
  if (fs.existsSync(asItIs)) return pathToFileURL(asItIs);
  throw new Error(`${dist}: neither _astro/${name}.<hash>.js nor ${name}.js in it: is ${name} still asked for by itself (import("./${name}.js")), and is the build of this tree?`);
}

/** forward.js as the tools on tests/engine.mjs read it: the sources', or with ENGINE_FROM=dist a build's */
export function forwardUrl(env = process.env) {
  if (!env.ENGINE_FROM || env.ENGINE_FROM === "sources") return runtimeUrl("forward.js");
  if (env.ENGINE_FROM !== "dist") throw new Error(`ENGINE_FROM=${env.ENGINE_FROM}: "dist" (a build's forward) or "sources"`);
  return builtModule("forward", env.ENGINE_DIST || undefined);
}
