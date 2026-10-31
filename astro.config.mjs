import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "astro/config";

// T.433: the site's Python as the two archives the workers fetch (python_archive.py, T.367.2: src/runtime/built/), made
// when a build or the development server starts (an `astro dev` or `astro build` called without npm had none: "Could not
// fetch engine.zip: 404"), and made again, and the page reloaded, whenever a .py of src/python/ changes under the
// development server (it served the archives of its start, and said nothing). Under 0.1 s: made whole every time.
// (by where this file is, whatever folder astro is called from)
const ROOT = path.dirname(fileURLToPath(import.meta.url)), PYTHON = path.join(ROOT, "src", "python");
const archives = () => execFileSync("python3", [path.join(ROOT, "python_archive.py")], { stdio: ["ignore", "ignore", "inherit"] });
const pythonArchives = () => ({
  name: "python-archives",
  buildStart: archives,
  configureServer(server) {
    server.watcher.add(PYTHON);
    server.watcher.on("all", (event, file) => {
      if (!file.startsWith(PYTHON + path.sep) || !file.endsWith(".py")) return;
      // (a .py saved broken is said, and the server goes on: the next save makes them again)
      try {
        archives();
      } catch (error) {
        server.config.logger.error(`python-archives: ${path.relative(PYTHON, file)} ${event}: the archives not made again (${error.message})`);
        return;
      }
      server.config.logger.info(`python-archives: ${path.relative(PYTHON, file)} ${event}: the archives made again`);
      server.ws.send({ type: "full-reload" });
    });
  },
});

// GitHub Pages serves the site under /pyodide-llm/
export default defineConfig({
  site: "https://takano32.github.io",
  base: "/pyodide-llm/",
  vite: {
    plugins: [pythonArchives()],
    // T.367.3: the runtime (src/runtime/) is bundled, a graph for each worker. A built file is always a file of its own
    // (never a data: URL in a worker's text), and a worker's files are named as Astro names the page's,
    // _astro/<name>.<hash>.<ext>: the shape public/coi.js's family() knows, so that the Service Worker drops the old
    // copy of a file when it keeps the new one (Vite's own <name>-<hash> matched nothing: tests/dist-check.ts).
    // /benchmark/'s workers carry their folder in the name (two entries are called gpu.js, and one family would put
    // the other's copy out). The workers are not minified (the owner, 2026-10-10: a visitor's report and the console
    // name functions as the sources do); format "es": a worker's module awaits at its top level.
    build: { assetsInlineLimit: 0 },
    // The development server only (`npm run dev`; a build has no server): it names a worker's module without the
    // site's base (/src/runtime/worker.js?worker_file), which is outside the Service Worker's scope, so that file
    // came without the isolation headers and the isolated page refused to start the worker (ERR_BLOCKED_BY_RESPONSE).
    // The server sends them itself. public/coi.js and the deployed site are as they were.
    server: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" } },
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
    define: {
      // The commit of the deployment: the ?v= of the Service Worker's registration (src/page/isolation.ts), so that a
      // new deployment installs its own, and the build a report names. The workers and what they read carry a hash
      // of their own text in their names instead (T.367.3).
      __BUILD__: JSON.stringify((process.env.GITHUB_SHA ?? "dev").slice(0, 7)),
    },
  },
});
