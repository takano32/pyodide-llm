// src/page/heard.ts (T.468.1): what the page reads of a report of the model's worker before anything is done with it:
// whether it is of the load going on (a late report of a load that another choice of model cancelled is passed over:
// the reports of a load carry its number, contracts/messages.ts), and whether it ends a piece of the worker's work
// (T.172: a load ends with ready, the benchmark with bench, the page's path with paths, a text with done, any of them
// with error; the others come in the middle of one, and read as an end they gave the run button back while the worker
// was busy). Pure: Node imports this file as it is (tests/page-parts-check.mjs).
import type { FromWorker } from "../runtime/contracts/messages.ts";

const ENDS: readonly FromWorker["type"][] = ["ready", "bench", "paths", "done", "error"];

/** Whether a report is of the load going on, the page's loads-th (or of no load: the worker's own words) */
export const ofThisLoad = (data: FromWorker, loads: number) => data.load === undefined || data.load === loads;
/** Whether a report ends a piece of the worker's work: the page may ask for the next */
export const endsWork = (data: FromWorker) => ENDS.includes(data.type);
