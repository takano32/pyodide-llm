"""tests/runtime-move-hand.py (T367.3.1): what a hand did after tests/runtime-move.mjs in the throwaway copy the tools
were proven on, written down so that the copy can be made again (TODO.md, T367.3.1):

    node tests/runtime-move.mjs move && node tests/runtime-move.mjs rewrite
    python3 tests/runtime-move-hand.py .          # then `node tests/runtime-move.mjs left` lists no line
    npm run build && node tests/dist-check.mjs

It is not T367.3.2's commit of the hand edits: that one is made by a hand that reads each of these, and it also moves
where the kernels and the archives are built (Makefile, kernels/build.py, python_archive.py, .gitignore: here the move
took the built files along), and corrects the comments that the rewrite made untrue (about sixty lines say "with its
own ?v=<build>", about eighty say public/). What is here is what it takes for the moved tree to build and run:
the table of the built files (built.js), the site's root handed to the worker with a load (models/ is under it), the
worker's key from shaders/key.js, the page's three places (the worker, kept.js, /benchmark/'s workers), and Vite's
settings. Every edit must find its text exactly once (a tree that moved on says where). Goes with the tool.

    python3 tests/runtime-move-hand.py <a tree>"""
import sys, os
root = sys.argv[1]
def edit(file, pairs):
    p = os.path.join(root, file); s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (file, a, s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)
R = "src/runtime/"
open(os.path.join(root, R, "built.js"), "w", encoding="utf-8").write('''// built.js (T367.3): the files of this site that are built and not written (`make kernels`: kernels/build.py; `make
// python`: python_archive.py), in src/runtime/built/, each by its own literal name. That is how a bundler is told of a
// file it is to serve: Vite gives each a name with a hash of its bytes (_astro/simdkernel_plain.<hash>.wasm), so a file
// that did not change keeps its name from one deployment to the next, and a page never reads one of another build.
// One by one and never by a name put together (`./built/${name}`): a URL with a variable in it makes the bundler copy
// the whole folder, and files of less than 4 KB went into the worker's text as data: URLs (T365).
export const BUILT = {
  "simdkernel.so": new URL("./built/simdkernel.so", import.meta.url),
  "simdkernel_relaxed.wasmlib": new URL("./built/simdkernel_relaxed.wasmlib", import.meta.url),
  "simdkernel_plain.wasm": new URL("./built/simdkernel_plain.wasm", import.meta.url),
  "simdkernel_relaxed_plain.wasm": new URL("./built/simdkernel_relaxed_plain.wasm", import.meta.url),
  "simdkernel_shared.wasm": new URL("./built/simdkernel_shared.wasm", import.meta.url),
  "simdkernel_relaxed_shared.wasm": new URL("./built/simdkernel_relaxed_shared.wasm", import.meta.url),
  "simdkernel_plain64.wasm": new URL("./built/simdkernel_plain64.wasm", import.meta.url),
  "simdkernel_relaxed_plain64.wasm": new URL("./built/simdkernel_relaxed_plain64.wasm", import.meta.url),
  "simdkernel_shared64.wasm": new URL("./built/simdkernel_shared64.wasm", import.meta.url),
  "simdkernel_relaxed_shared64.wasm": new URL("./built/simdkernel_relaxed_shared64.wasm", import.meta.url),
  "ceilings_plain.wasm": new URL("./built/ceilings_plain.wasm", import.meta.url),
  "ceilings_relaxed_plain.wasm": new URL("./built/ceilings_relaxed_plain.wasm", import.meta.url),
  "ceilings_shared.wasm": new URL("./built/ceilings_shared.wasm", import.meta.url),
  "ceilings_relaxed_shared.wasm": new URL("./built/ceilings_relaxed_shared.wasm", import.meta.url),
  "engine.zip": new URL("./built/engine.zip", import.meta.url),
  "converter.zip": new URL("./built/converter.zip", import.meta.url),
};
''')
edit(R + "worker.js", [
    ("import { state, cpuOnly, modelKey, loadSeconds } from \"./worker/state.js\";", "import { BUILT } from \"./built.js\";\nimport { state, cpuOnly, modelKey, loadSeconds } from \"./worker/state.js\";"),
    ("(url) => import(url));", "(url) => import(/* @vite-ignore */ url));"),
    ("await fetch(new URL(`${name}${self.location.search}`, import.meta.url)).catch(() => undefined);", "await fetch(BUILT[name]).catch(() => undefined);"),
    ("fetch(new URL(`${name}${self.location.search}`, import.meta.url)).then((res)", "fetch(BUILT[name]).then((res)"),
    ("      state.loadsAhead = Array.isArray(data.ahead)", "      state.root = data.root ?? state.root;  // (T367.3: the site's root, where models/ is; the page knows it)\n      state.loadsAhead = Array.isArray(data.ahead)"),
])
edit(R + "worker/state.js", [
    ("  pageSwitches: [],", "  pageSwitches: [],\n  root: undefined,          // the URL of the site's root, which the page hands with a load (T367.3): models/ is under it"),
    ("const wgsl = await import(\"../shaders.js\");", "const wgsl = await import(\"../shaders/key.js\");"),
])
edit(R + "worker/pyodide.js", [
    ("await fetch(new URL(`../${name}${self.location.search}`, import.meta.url), { signal });", "await fetch(BUILT[name], { signal });"),
])
edit(R + "worker/sources.js", [("new URL(`../models/${model.checkpoint}.${String(part).padStart(3, \"0\")}`, import.meta.url).href", "new URL(`models/${model.checkpoint}.${String(part).padStart(3, \"0\")}`, state.root).href")])
edit(R + "worker/load.js", [("new URL(`../models/${model.tokenizer}`, import.meta.url)", "new URL(`models/${model.tokenizer}`, state.root)")])
edit(R + "benchmark/sections.js", [
    ("const search = new URL(import.meta.url).search;  // ?v=<commit>: every file of the same deployment\nconst at = (name) => new URL(`../${name}${search}`, import.meta.url);\n", "import { BUILT } from \"../built.js\";\n"),
])
p = os.path.join(root, R, "benchmark/sections.js"); s = open(p).read(); assert s.count("fetch(at(name))") == 2; open(p, "w").write(s.replace("fetch(at(name))", "fetch(BUILT[name])"))
edit(R + "benchmark/gpu/matvec.js", [("fetch(new URL(`../../simdkernel_plain.wasm${new URL(import.meta.url).search}`, import.meta.url))", "fetch(BUILT[\"simdkernel_plain.wasm\"])")])
edit("src/page/receiver.ts", [("new Worker(`${import.meta.env.BASE_URL}worker.js?v=${__BUILD__}`, { type: \"module\" });", "new Worker(new URL(\"../runtime/worker.js\", import.meta.url), { type: \"module\" });\n/** the site's root, which the worker is told with a load: models/ is under it, and the worker's own URL is not (T367.3) */\nexport const root = new URL(import.meta.env.BASE_URL, location.href).href;")])
edit("src/page/list.ts", [("declare const __BUILD__: string;\n\n", ""),
    ("const keptModule = import(/* @vite-ignore */ `${import.meta.env.BASE_URL}kept.js?v=${__BUILD__}`).catch(() => undefined);", "// (T367.3: linked into the page's own script. The page read it as a file of its own before, and went on without it\n// where that failed: what reads it still takes a promise of it.)\nconst keptModule = Promise.resolve({ forget, keptModels, outdated, serves });"),
    ("import { page } from \"./state.ts\";", "import { page } from \"./state.ts\";\nimport { forget, keptModels, outdated, serves } from \"../runtime/kept.js\";")])
edit("src/benchmark/section.ts", [("  const w = new Worker(`${BASE}${file}${V}`, { type: \"module\" });", "  const w = WORKERS[file]();"),
    ("export const live = new Set<Worker>();", '''// (T367.3: each by its literal URL, which is how the bundler is told of a worker)
const WORKERS = {
  "worker.js": () => new Worker(new URL("../runtime/worker.js", import.meta.url), { type: "module" }),
  "benchmark/gpu.js": () => new Worker(new URL("../runtime/benchmark/gpu.js", import.meta.url), { type: "module" }),
  "benchmark/memory.js": () => new Worker(new URL("../runtime/benchmark/memory.js", import.meta.url), { type: "module" }),
  "benchmark/sections.js": () => new Worker(new URL("../runtime/benchmark/sections.js", import.meta.url), { type: "module" }),
  "benchmark/storage.js": () => new Worker(new URL("../runtime/benchmark/storage.js", import.meta.url), { type: "module" }),
  "benchmark/waiter.js": () => new Worker(new URL("../runtime/benchmark/waiter.js", import.meta.url), { type: "module" }),
};
export const live = new Set<Worker>();'''),
    ("export function worker(file: string) {", "export function worker(file: keyof typeof WORKERS) {")])
edit("src/pages/index.astro", [("worker.postMessage({ type: \"init\", search: location.search,", "worker.postMessage({ type: \"init\", root, search: location.search,")])
edit("src/page/choose.ts", [("worker.postMessage({ type: \"load\", search: location.search,", "worker.postMessage({ type: \"load\", root, search: location.search,")])
edit("src/benchmark/model.ts", [("await request({ type: \"init\", search:", "await request({ type: \"init\", root: new URL(BASE, location.href).href, search:")])
edit(R + "worker/pyodide.js", [("import { isError, told } from \"./told.js\";", "import { isError, told } from \"./told.js\";\nimport { BUILT } from \"../built.js\";")])
edit(R + "benchmark/gpu/matvec.js", [("// the kernels of the same deployment as this file (?v=, GitHub Pages keeps a file for ten minutes)\n", "")])
p = os.path.join(root, R, "benchmark/gpu/matvec.js"); s = open(p).read(); i = s.index("import {"); open(p, "w").write(s[:i] + "import { BUILT } from \"../../built.js\";\n" + s[i:])
p = os.path.join(root, R, "worker/sources.js"); s = open(p).read()
if "import { state" not in s: s = s.replace("import { told } from \"./told.js\";", "import { state } from \"./state.js\";\nimport { told } from \"./told.js\";"); open(p, "w").write(s)
edit("src/page/choose.ts", [("import { worker } from \"./receiver.ts\";", "import { root, worker } from \"./receiver.ts\";")])
edit("src/pages/index.astro", [("import { worker } from \"../page/receiver.ts\";", "import { root, worker } from \"../page/receiver.ts\";")])
edit("src/benchmark/model.ts", [("import { $, parameters, type Result } from \"./dom.ts\";", "import { BASE, $, parameters, type Result } from \"./dom.ts\";")])
edit("src/benchmark/section.ts", [("import { BASE, V, $, type Name, type Result } from \"./dom.ts\";", "import { $, type Name, type Result } from \"./dom.ts\";")])
edit("astro.config.mjs", [("  vite: {\n", '''  vite: {
    // T367.3: the runtime (src/runtime/) is bundled, a graph for each worker. A built file is always a file of its own
    // (never a data: URL in a worker's text), and a worker's files are named as Astro names the page's,
    // _astro/<name>.<hash>.<ext>: the shape public/coi.js's family() knows, so that the Service Worker drops the old
    // copy of a file when it keeps the new one (Vite's own <name>-<hash> matched nothing: tests/dist-check.mjs).
    // /benchmark/'s workers carry their folder in the name (two entries are called gpu.js, and one family would put
    // the other's copy out). The workers are not minified (the owner, 2026-10-10: a visitor's report and the console
    // name functions as the sources do); format "es": a worker's module awaits at its top level.
    build: { assetsInlineLimit: 0 },
    worker: {
      format: "es",
      rolldownOptions: {
        output: {
          minify: false,
          entryFileNames: (chunk) => (chunk.facadeModuleId?.includes("/runtime/benchmark/") ? "_astro/benchmark-[name].[hash].js" : "_astro/[name].[hash].js"),
          chunkFileNames: "_astro/[name].[hash].js",
          assetFileNames: "_astro/[name].[hash][extname]",
        },
      },
    },
''')])
