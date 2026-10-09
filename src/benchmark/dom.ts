// What the parts of /benchmark/'s script stand on: the deployment's ?v=, the elements by id, the URL's parameters, the
// sections' names, what a section's result is, the results, and the choices the URL makes.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)

declare const __BUILD__: string;
export const BASE = import.meta.env.BASE_URL;
// every worker of the same deployment as this page (GitHub Pages keeps a file for ten minutes)
export const V = `?v=${__BUILD__}`;
export const $ = (id: string) => document.getElementById(id)!;
export const parameters = new URLSearchParams(location.search);
export const SECTIONS = ["device", "cpu", "model", "gpu", "storage", "line", "memory"] as const;
export type Name = typeof SECTIONS[number];
// T173: "Run all" and ?run=all leave the page-memory section out (it may end the tab): its button or ?run=memory runs it
export const ALL = SECTIONS.filter((name) => name !== "memory");
// ok: measured; none: the browser has no such thing (said, and the run goes on); wrong: a result that cannot be
// right (a shader against JavaScript, a piece read back); error: a section that failed
// said (T227): the lines of the Markdown with which the section itself says what went wrong, for the report's warnings
export type Result = { status: "ok" | "none" | "wrong" | "error"; markdown: string; data?: any; said?: string[] };
export const results: Partial<Record<Name, Result>> = {};
(window as any).__benchmark = { results, done: false };

for (const [name, value] of [["model", parameters.get("model")], ["size", parameters.get("size")]] as const) {
  const select = $(name) as HTMLSelectElement;
  if (value && [...select.options].some((option) => option.value === value)) select.value = value;
  else if (name === "size" && value && Number(value) > 0) { select.add(new Option(`${value} MB`, value)); select.value = value; }
}
if (parameters.has("large")) ($("large") as HTMLInputElement).checked = true;
if (parameters.has("full")) ($("full") as HTMLInputElement).checked = true;
