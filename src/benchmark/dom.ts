// What the parts of /benchmark/'s script stand on: the deployment's ?v= (the Service Worker's), the elements by id, the URL's parameters, the
// sections the run-all runs, the results, and the choices the URL makes (the sections' names and what a section's result is are
// src/bench/sections.ts's, T643.11).
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)

declare const __BUILD__: string;
export const BASE = import.meta.env.BASE_URL;
// the Service Worker's registration carries it (isolation.ts). The workers do not any more (T367.3: a bundler names
// each file by a hash of its text)
export const V = `?v=${__BUILD__}`;
export const $ = (id: string) => document.getElementById(id)!;
export const parameters = new URLSearchParams(location.search);
import { SECTIONS } from "../bench.ts";
import type { Name, Result } from "../bench.ts";
// T173: "Run all" and ?run=all leave the page-memory section out (it may end the tab): its button or ?run=memory runs it
export const ALL = SECTIONS.filter((name) => name !== "memory");
export const results: Partial<Record<Name, Result>> = {};
(window as any).__benchmark = { results, done: false };

for (const [name, value] of [["model", parameters.get("model")], ["size", parameters.get("size")]] as const) {
  const select = $(name) as HTMLSelectElement;
  if (value && [...select.options].some((option) => option.value === value)) select.value = value;
  else if (name === "size" && value && Number(value) > 0) { select.add(new Option(`${value} MB`, value)); select.value = value; }
}
if (parameters.has("large")) ($("large") as HTMLInputElement).checked = true;
if (parameters.has("full")) ($("full") as HTMLInputElement).checked = true;
