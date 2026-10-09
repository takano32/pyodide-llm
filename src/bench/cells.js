// What the tables of /benchmark/ share: a number, a ratio, a cell's words, why there are no ratios, the threads' words.
// (T353: a part of src/bench.js, which is the window that exports every name of these)

const number = (value, digits = 1) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "?");

/** "4 software threads", "1 software thread (not cross-origin isolated)" */
export const threadsOf = ({ threads, isolated }) =>
  `${threads} software thread${threads === 1 ? "" : "s"}${isolated ? "" : " (not cross-origin isolated)"}`;

/** A ratio as the tables write it: to one decimal from 1 up, to two significant digits below ("0.50×", "0.071×"),
 * whole from 100; "" where it is no number. The measurements are one run each: more digits would be noise. */
export function times(value) {
  if (!Number.isFinite(value) || value <= 0) return "";
  return `${value >= 100 ? value.toFixed(0) : value >= 1 ? value.toFixed(1) : value.toPrecision(2)}×`;
}

/** How many times faster the GPU is than the CPU on the same work ("0.50×": the CPU is faster); "" where either is
 * missing. */
export const timesFaster = (cpuMs, gpuMs) => (Number.isFinite(cpuMs) && Number.isFinite(gpuMs) && cpuMs > 0 && gpuMs > 0
  ? times(cpuMs / gpuMs) : "");

/** Why the GPU section shows no ratios at all, whatever the CPU: a lost device (its later times are no GPU's, and
 * too fast: a lost device answers every wait at once) or a fallback adapter (the CPU in a GPU's place). undefined
 * where the ratios stand. gpu: { fallback, lost }. */
export function noRatios({ fallback, lost } = {}) {
  if (lost) return "none: the device was lost, and the times after it are no GPU's";
  if (fallback) return "none on a fallback adapter: its times are no GPU's";
  return undefined;
}

/** Words for one cell of a table: a | or a line break of them would break the row (the page and GitHub both read
 * \| as a | in a cell). */
export const tableCell = (text) => String(text).replace(/\|/g, "\\|").replace(/\s*\n\s*/g, " ");

/** Why a step of the GPU section has no table: it failed (T227: in that word, as the rows say it, so that warnings()
 * lists it), or it was not run. */
export const unmeasured = (error) => (error ? `failed: ${error}` : "not measured");

export { number };
