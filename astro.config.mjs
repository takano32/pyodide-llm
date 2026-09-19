import { defineConfig } from "astro/config";

// GitHub Pages serves the site under /pyodide-llama-py/
export default defineConfig({
  site: "https://takano32.github.io",
  base: "/pyodide-llama-py/",
  vite: {
    define: {
      // The commit of the deployment. GitHub Pages lets browsers cache every file for ten minutes, and a new page
      // must never run with the worker or the Python code of the previous deployment: see the ?v= in index.astro.
      __BUILD__: JSON.stringify((process.env.GITHUB_SHA ?? "dev").slice(0, 7)),
    },
  },
});
