// The benchmark of T.45: what this browser does with this model, as a table someone can paste into an issue.
// A plain module, so that Node can import it and test it (the page and tests/bench.ts both use it).

// T.353: this file is the window: every name it exported as one file, from the modules of bench/, a section each: rounds.js
// (the rounds and their table), report.js (the issue's body and link, the short report, reading reports back), warnings.js
// (what came out wrong, at the report's head), cpu.js, gpu.js, layersteps.js and generate.js (the CPU's and the GPU's
// tables), path.js (the page's own path), summary.js (a line a section) and cells.js (what the tables share).
export { threadsOf, times, timesFaster, noRatios, tableCell, unmeasured } from "./bench/cells.ts";
export { ROUNDS, FULL_ROUNDS, MEMORY_UNSAID, DRIFT_LINE, roundsHere, environmentOf, benchMarkdown, roundsTable, drift } from "./bench/rounds.ts";
export { threadCounts, cpuTable, CPU_SPEED_KEY, USAGE_KEY, USAGE_DECAY, usedAfter, cpuBaseline } from "./bench/cpu.ts";
export type { CpuMeasured, Baseline } from "./bench/cpu.ts";
export { warnings, warningsBlock } from "./bench/warnings.ts";
export { REPOSITORY, QUESTIONS, reportBody, REPORT_LIMIT, TOO_LONG, PASTE, reportTooLong, reportUrl, loginUrl,
  shortReport, parseReport, reportsTable } from "./bench/report.ts";
export { layerCheckNumbers, checkVerdict, matVecTable, tokenTable, layerTable } from "./bench/gpu.ts";
export { layerStepsTable } from "./bench/layersteps.ts";
export { generateTable } from "./bench/generate.ts";
export { threadsKey, threadsHow, threadsLine, PATH_PROMPTS, PATH_WRITES, gpuSkipped, pathTable, pathWarnings } from "./bench/path.ts";
export { deviceSummary, cpuSummary, gpuSummary, storageSummary, lineSummary } from "./bench/summary.ts";
