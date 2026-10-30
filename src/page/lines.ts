// src/page/lines.ts (T.468.2, T.475): the status line's words, from what the page holds of the model and of its load
// (model.ts) and what a report says: what draw.ts and receiver.ts write there. Pure: Node imports this file as it is
// (tests/page-parts-check.mjs), which reads every part of each line.
import { SLOWER } from "../runtime/updated.ts";

/** A ready model's line: its name, how it runs (and on how many threads, and what the GPU takes: gpuNote, with
 * ?gpuTest=on said where the tests' flag is on), Pyodide's version, and last (T.439) that a reload would make it faster
 * where it runs slower for a file of this page's build that is gone */
export function readyStatus({ name, backend, threads, pyodide, gpuNote, gpuTest, outdated }:
  { name: string; backend: string; threads: number; pyodide: string; gpuNote?: string; gpuTest: boolean; outdated: boolean }) {
  // (?gpuTest=on, the tests' flag, says so as ?without= does)
  const tested = gpuNote && gpuTest ? " (gpuTest)" : "";
  return `${name} · ${backend}${threads > 1 ? `, ${threads} threads` : ""}${gpuNote ? `, ${gpuNote}${tested}` : ""} · Pyodide ${pyodide}${outdated ? ` · ${SLOWER}` : ""}`;
}

/** A load's line as it fetches and converts: what has arrived of how much, how fast, and the share converted. T.119:
 * a conversion says what has arrived and how fast apart from what is converted (the share converted moves only as
 * fast as the bytes come, and alone it made a slow line look like a slow conversion). A file of the visitor's own
 * disk has nothing to fetch: the share alone. (The time left is remaining.ts's, said after it.) */
export function progressLine({ name, received, total, perSecond, converted }:
  { name: string; received: number; total: number; perSecond?: number; converted?: number }) {
  const megabytes = (bytes: number) => (bytes / 1e6).toFixed(1);
  const share = converted === undefined ? "" : `${Math.floor(converted * 100)} % converted`;
  return !received && share ? `${name}: ${share}`
    : `${name}: ${megabytes(received)} / ${megabytes(total)} MB` + (perSecond ? ` · ${megabytes(perSecond)} MB/s` : "") + (share ? ` · ${share}` : "");
}
