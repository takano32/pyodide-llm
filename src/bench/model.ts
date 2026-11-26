// The section "Model": its result and what the report's head holds of it, from what the model page's worker answered
// (T301: from the page's part, src/benchmark/model.ts, so that the section's Markdown is made in Node as the others' are).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)
import { roundsTable, drift } from "./rounds.ts";
import { pathTable, pathWarnings } from "./path.ts";
import type { PathsFound } from "./path.ts";
import type { Result } from "./sections.ts";
import type { Measured } from "./whole.ts";
import type { BenchRow } from "../runtime/contracts/messages.ts";

/** The model section from the worker's answers: name, the model's; pyodide, the version that ran; rows, the rounds;
 * paths, the page's path. measured is what the report's head holds of it (said, T227's review: what the page's path
 * says went wrong in its first line, for the warnings). */
export function modelResult(name: string, pyodide: string, rows: BenchRow[], paths: PathsFound): { result: Result<"model">; measured: Measured } {
  const table = pathTable(paths, name);
  const lines = [...roundsTable(rows), ...[drift(rows)].filter(Boolean).flatMap((line) => ["", line])];
  return { measured: { name, rows, pyodide, paths: table, said: pathWarnings(paths) },
           result: { status: "ok", data: { rows, paths }, markdown: `${name} · Pyodide ${pyodide}\n\n${lines.join("\n")}\n\n${table}` } };
}
