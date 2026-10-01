// P3: the page loaded again after its tab was ended (T173's mark in sessionStorage): the report of the page-memory section
import { install, load, byId } from "./page-harness.mjs";
import { base, use } from "./page-scenarios.mjs";

install();
const mark = { held: 1792 * 2 ** 20, trying: 1856 * 2 ** 20, limit: 4 * 2 ** 30, began: Date.now() - 20000 };
globalThis.sessionStorage.setItem("benchmark-memory", JSON.stringify(mark));
const page = await load();
use(base());
console.log(Object.fromEntries(Object.entries(page.results).map(([k, v]) => [k, v.status])));
console.log(window.__benchmark.markdown);
console.log("link", byId("issue").href.length);
process.exit(0);
