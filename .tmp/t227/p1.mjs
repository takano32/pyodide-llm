// P1: the page, all sections, the fake workers answering the fixtures (everything ok)
import { install, load, byId } from "./page-harness.mjs";
import { base, use } from "./page-scenarios.mjs";

install();
const page = await load();
const s = base();
use(s);
const t0 = Date.now();
await page.runSections(["device", "cpu", "model", "gpu", "storage", "line"]);
console.log(`ran in ${Date.now() - t0} ms; statuses:`, Object.fromEntries(Object.entries(page.results).map(([k, v]) => [k, v.status])));
const markdown = window.__benchmark.markdown;
console.log(`report ${markdown.length} chars`);
const at = markdown.indexOf("#### Warnings");
console.log(at < 0 ? "no Warnings block" : markdown.slice(at, markdown.indexOf("\n#### ", at + 5)).split("\n").map((l) => l.slice(0, 160)).join("\n"));
const href = byId("issue").href;
console.log("link", href.length, "login", `https://github.com/login?return_to=${encodeURIComponent(href)}`.length, "long hidden:", byId("long").hidden, "longer hidden:", byId("longer").hidden);
process.exit(0);
