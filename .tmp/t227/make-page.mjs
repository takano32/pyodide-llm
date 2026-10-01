// Cut the benchmark page's <script> out of src/pages/benchmark.astro as a TypeScript module that Node runs with the DOM,
// the workers and the storage stubbed (page-harness.mjs): the page's own report() and sections, end to end, without a browser
import fs from "node:fs";

const astro = fs.readFileSync(new URL("../../src/pages/benchmark.astro", import.meta.url), "utf8");
const start = astro.indexOf("<script>\n") + "<script>\n".length, end = astro.lastIndexOf("</script>");
let script = astro.slice(start, end);
script = script.replaceAll('from "../', 'from "../../src/').replace("import.meta.env.BASE_URL", '"/pyodide-llm/"');
script += `\n;(globalThis as any).__page = { results, runSections, report, device, cpu, gpu, gpuMarkdown, storage, line, SECTIONS, ALL, TITLES, parts };\n`;
fs.writeFileSync(new URL("./page-under-test.ts", import.meta.url), script);
console.log(`${script.split("\n").length} lines`);
