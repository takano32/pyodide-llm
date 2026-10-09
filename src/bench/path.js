// The model section's table of the page's own path (T184, T190): its threads, its prompts and writing, and what it warns of.
// (T353: a part of src/bench.js, which is the window that exports every name of these)
import { number, times, tableCell } from "./cells.js";

/** T93: where the model page remembers the number of threads it found for a model on this device (localStorage).
 * T190: /benchmark/ reads the same, so that the page path is timed on the model page's count. nav: the navigator */
export const threadsKey = (id, nav) => `threads:${id}:${nav.hardwareConcurrency}:${nav.deviceMemory ?? ""}:${nav.userAgent}`;

// the words of the page path's first line for what stopped, which pathTable() writes and pathWarnings() lists (T227's
// review): software threads that stopped after the count was found, a search that did not end, the GPU stopped while the
// sides were timed; and (public/worker/timing.js) why a page that is not isolated has one thread, which is no failure
const STOPPED_WHILE_TIMED = "a software thread stopped while timed, and one thread went on";
const searchNotEnded = (how) => `the search had not ended after ${how.unfinished} s`;
const gpuStopped = (gpu) => `WebGPU stopped while timed: ${tableCell(gpu.lost)}`;
const NO_SHARED_MEMORY = "no shared memory here";

/** T190: how the page path's number of threads came about (worker/timing.js's timedPaths), in a few words */
export function threadsHow(how) {
  if (!how) return "";
  if (how.alone) return `: ${how.alone}`;
  // T190's review: a software thread that stopped after the count was found (T120): the later times are one thread's
  const stopped = how.stopped ? `; ${STOPPED_WHILE_TIMED}` : "";
  if (how.unfinished) return `: ${searchNotEnded(how)}${stopped}`;
  if (how.remembered) return `, as the model page remembers${stopped}`;
  const verdicts = (how.searched ?? []).map(([best, candidate, kept]) => `${best} or ${candidate}: ${kept}`);
  return `${verdicts.length ? `, searched here (${verdicts.join(", ")})` : ""}${stopped}`;
}

/** T190's review: the writing on each number of software threads (worker/timing.js's timedPaths: in turn, CPU only), the page's
 * marked; "" where there is one count or none. perCount: [{ threads, speed, low, high, unsteady }] */
export function threadsLine(perCount = [], page) {
  if (perCount.length < 2) return "";
  const rate = (value) => number(value, value >= 100 ? 0 : 1);
  const cells = perCount.map((c) => `${c.threads}${c.threads === page ? " (the page's)" : ""}: ${rate(c.speed)} tok/s` +
    ` (${rate(c.low)}–${rate(c.high)}${c.unsteady ? ", unsteady" : ""})`);
  return `Writing on each number of software threads (CPU only): ${cells.join(" · ")}`;
}

/** T184: the prompts the model page's path is timed on: one block of the GPU's (forward.js's GPU_BLOCK) and four; and
 * the tokens it writes after a prompt */
export const PATH_PROMPTS = [64, 256];
export const PATH_WRITES = 64;

/** T184: why a GPU cell of the page path is empty, in a few words: "not here" (no WebGPU, no adapter), "not on a
 * fallback adapter" (the CPU in a GPU's place: the page refuses it, T148, and CI's are all such, T182), else the words of
 * forward.js (a model the GPU does not take yet, too little memory, a failure). */
export function gpuSkipped(why = "") {
  if (/no WebGPU|no GPU adapter/.test(why)) return "not here";
  if (/fallback adapter/.test(why)) return "not on a fallback adapter";
  return why || "not measured";
}

/** T184: the model page's own path on this device, as one table (the model section times it on its first load,
 * worker/timing.js's timedPaths and forward.js's timePrompts): prompts as the page chooses between the GPU and the CPU (T148),
 * on the CPU only and on the GPU only, how many times faster the GPU is, and the writing after a prompt. paths:
 * { threads, how (threadsHow()), perCount (threadsLine()), gpu: { seconds, matrices, attention, lost? } or { why },
 * status (the status line's words of the GPU),
 * rows: [{ what: "prompt" | "generation", tokens, chosen, cpu, gpu }] } where a cell is { speed, low, high, gpuTokens,
 * unsteady }, { same: "cpu" } (one path: timed once) or { skip: why }; or { error }. The writing's GPU cells (T152: the
 * steps of a generation on the GPU) are timed where the GPU takes them, else the reason. A GPU that stopped while the sides were timed
 * (gpu.lost) leaves no ratio anywhere: its later times are no GPU's (T157). */
export function pathTable(paths, name = "") {
  const title = `**The model page's path**${name ? ` (${name})` : ""}`;
  if (!paths || paths.error) return `${title}: failed: ${tableCell(paths?.error ?? "no answer")}`;
  const { gpu = {}, rows = [] } = paths;
  const facts = [paths.threads !== undefined && `${paths.threads} software thread${paths.threads === 1 ? "" : "s"}${threadsHow(paths.how)}`];
  if (gpu.why !== undefined) facts.push(`WebGPU: ${tableCell(gpuSkipped(gpu.why))}`);
  else if (gpu.lost) facts.push(gpuStopped(gpu));
  else {
    facts.push(`WebGPU ready in ${number(gpu.seconds)} s`, `matrices by ${tableCell(gpu.matrices ?? "?")}`, `attention by ${tableCell(gpu.attention ?? "?")}`);
    if (paths.status) facts.push(tableCell(paths.status));
  }
  // a cell the GPU did not run: the few words of gpuSkipped(), or "not used" where the line above has the reason
  const skipped = (why) => (why === gpu.why && gpuSkipped(why) === why ? "not used" : tableCell(gpuSkipped(why)));
  const rate = (value) => number(value, value >= 100 ? 0 : 1);
  const speed = (cell) => {
    if (!cell) return "?";
    if (cell.skip !== undefined) return skipped(cell.skip);
    if (cell.same) return "same as CPU only";
    const spread = cell.low !== undefined ? ` (${rate(cell.low)}–${rate(cell.high)}${cell.unsteady ? ", unsteady" : ""})` : "";
    return `${rate(cell.speed)} tok/s${spread}`;
  };
  // where the page's choice put the tokens: all on one side, or a part on the GPU (a GPU, then the CPU: T148)
  const side = (cell, tokens) => (cell?.gpuTokens === undefined || cell.same ? "" : cell.gpuTokens >= tokens ? ", GPU"
    : cell.gpuTokens > 0 ? `, GPU ${cell.gpuTokens} of ${tokens}` : ", CPU");
  const lines = [`${title}: ${facts.filter(Boolean).join(" · ")}`, "",
    "| the page | as chosen | CPU only | GPU only | GPU ÷ CPU |", "|---|---|---|---|---|",
    ...rows.map((row) => {
      const what = row.what === "prompt" ? `a prompt of ${row.tokens} tokens` : `writing ${row.tokens} tokens`;
      const ratio = !gpu.lost && row.gpu?.speed && row.cpu?.speed ? times(row.gpu.speed / row.cpu.speed) : "";
      return `| ${what} | ${speed(row.chosen)}${side(row.chosen, row.tokens)} | ${speed(row.cpu)} | ${speed(row.gpu)} | ${ratio} |`;
    })];
  const counts = threadsLine(paths.perCount, paths.threads);
  if (counts) lines.push("", counts);
  return lines.join("\n");
}

/**
 * T227's review: what the model page's path says went wrong, as it says it in pathTable()'s first line, for warnings() to
 * list as the section's own (said). The GPU stopped while the sides were timed (forward.js gives a dozen reasons for
 * stopping it, and only some have "failed" in them: it said nothing, its worker stopped answering, its logits were not
 * finite), software threads that stopped or did not start, a search for their count that did not end. [] where none
 * did; a page that is not isolated has no software threads, and that is no failure.
 */
export function pathWarnings(paths) {
  if (!paths || paths.error) return [];  // (an error says "failed" itself)
  const { gpu = {}, how = {} } = paths;
  // (as threadsHow() and pathTable() write them: alone says why the count is one, and nothing else of the threads)
  const threads = how.alone ? [how.alone !== NO_SHARED_MEMORY && how.alone] : [how.unfinished && searchNotEnded(how), how.stopped && STOPPED_WHILE_TIMED];
  return [gpu.why === undefined && gpu.lost && gpuStopped(gpu), ...threads].filter(Boolean);
}
