import { defineConfig } from "astro/config";

// GitHub Pages serves the site under /pyodide-llm/
export default defineConfig({
  site: "https://takano32.github.io",
  base: "/pyodide-llm/",
  vite: {
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
    define: {
      // The commit of the deployment. GitHub Pages lets browsers cache every file for ten minutes, and a new page
      // must never run with the worker or the Python code of the previous deployment: see the ?v= in src/page/.
      __BUILD__: JSON.stringify((process.env.GITHUB_SHA ?? "dev").slice(0, 7)),
    },
  },
});
