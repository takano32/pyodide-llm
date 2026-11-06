// What the parts of /benchmark/'s script stand on: the deployment's ?v= (the Service Worker's), the elements by id, the
// URL's parameters, the sections the run-all runs, what the page tells the tools (window.__benchmark), and the choices
// the URL makes (the sections' names and what a section's result is are src/bench/sections.ts's, T643.11).
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)

declare const __BUILD__: string;
export const BASE = import.meta.env.BASE_URL;
// the Service Worker's registration carries it (isolation.ts). The workers do not any more (T367.3: a bundler names
// each file by a hash of its text)
export const V = `?v=${__BUILD__}`;
export const $ = (id: string) => document.getElementById(id)!;
export const parameters = new URLSearchParams(location.search);
import { SECTIONS } from "../bench.ts";
import type { Name } from "../bench.ts";
import { results } from "./running.ts";
// T173: "Run all" and ?run=all leave the page-memory section out (it may end the tab): its button or ?run=memory runs it
export const ALL = SECTIONS.filter((name) => name !== "memory");
// T643.14: what the page tells the tools that drive it (tests/bench-check.ts, page-memory-check.ts, updated-in-browser.ts,
// screenshots.ts), told here alone: the sections' results are running.ts's, read as they are now; whether a run is
// done (a tool sets it false itself before it presses a button), the sections of the run-all, and the report's Markdown
type Told = { done?: boolean; all?: readonly Name[]; markdown?: string };
const benchmark: Told & { readonly results: typeof results } = { get results() { return results; }, done: false };
Object.assign(window, { __benchmark: benchmark });
/** Tell the tools these. */
export const told = (fields: Told) => { Object.assign(benchmark, fields); };

for (const [name, value] of [["model", parameters.get("model")], ["size", parameters.get("size")]] as const) {
  const select = $(name) as HTMLSelectElement;
  if (value && [...select.options].some((option) => option.value === value)) select.value = value;
  else if (name === "size" && value && Number(value) > 0) { select.add(new Option(`${value} MB`, value)); select.value = value; }
}
if (parameters.has("large")) ($("large") as HTMLInputElement).checked = true;
if (parameters.has("full")) ($("full") as HTMLInputElement).checked = true;
