// The benchmark of T45: what this browser does with this model, as a table someone can paste into an issue.
// A plain module, so that Node can import it and test it (the page and tests/bench.mjs both use it).

// T353: this file is the window: every name it exported as one file, from the modules of bench/, a section each: rounds.js
// (the rounds and their table), report.js (the issue's body and link, the short report, reading reports back), warnings.js
// (what came out wrong, at the report's head), cpu.js, gpu.js, layersteps.js and generate.js (the CPU's and the GPU's
// tables), path.js (the page's own path), summary.js (a line a section) and cells.js (what the tables share).
export { threadsOf, times, timesFaster, noRatios, tableCell, unmeasured } from "./bench/cells.js";
export { ROUNDS, FULL_ROUNDS, MEMORY_UNSAID, roundsHere, environmentOf, benchMarkdown, roundsTable } from "./bench/rounds.js";
export { threadCounts, cpuTable, CPU_SPEED_KEY, USAGE_KEY, USAGE_DECAY, usedAfter, cpuBaseline } from "./bench/cpu.js";
export { warnings, warningsBlock } from "./bench/warnings.js";
export { REPOSITORY, QUESTIONS, reportBody, REPORT_LIMIT, TOO_LONG, PASTE, reportTooLong, reportUrl, loginUrl,
  shortReport, parseReport, reportsTable } from "./bench/report.js";
export { layerCheckNumbers, checkVerdict, matVecTable, tokenTable, layerTable } from "./bench/gpu.js";
export { layerStepsTable } from "./bench/layersteps.js";
export { generateTable } from "./bench/generate.js";
export { threadsKey, threadsHow, threadsLine, PATH_PROMPTS, PATH_WRITES, gpuSkipped, pathTable, pathWarnings } from "./bench/path.js";
export { deviceSummary, cpuSummary, gpuSummary, storageSummary, lineSummary } from "./bench/summary.js";
