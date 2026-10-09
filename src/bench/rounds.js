// The rounds of the benchmark (T45: ?bench= and /benchmark/'s model section), the environment, and their table.
// (T353: a part of src/bench.js, which is the window that exports every name of these)
import { number, tableCell } from "./cells.js";

/** One row per switch combination the benchmark ran. */
export const ROUNDS = [
  { name: "everything", without: [] },
  { name: "without the kernels", without: ["kernels"] },
];

/** The steps of T52, for ?bench=full: each optimization added to the one before it (T110 added the sixth). */
export const FULL_ROUNDS = [
  { name: "NumPy only", without: ["kernels"] },
  { name: "the kernels, int8 widened", without: ["int8", "relaxed", "sampler", "kv16"] },
  { name: "int8 kept as int8", without: ["relaxed", "sampler", "kv16"] },
  { name: "relaxed SIMD", without: ["sampler", "kv16"] },
  { name: "sampling in the kernel", without: ["kv16"] },
  { name: "float16 keys and values", without: [] },
];

/** T214: why a round that widens the weights is skipped where the browser does not say how much memory the device has
 * (navigator.deviceMemory: Safari and Firefox never say). Without the kernels NumPy widens every weight to float32
 * (four bytes a weight, where the kernels keep one): with llm-jp-3 150M the worker's Pyodide grew to 958 MB and the
 * renderer by 0.81 GB in CI's Chromium, and an iPhone's tab went down in that round (T205's review). A page that is
 * not told the memory cannot tell whether it fits (the owner, 2026-09-27: such a device skips the NumPy round;
 * 2026-09-28: and the round of the kernels with int8 widened, which widens them as well). */
export const MEMORY_UNSAID = "this browser does not say how much memory the device has, and this round widens every weight " +
  "to float32 (four bytes a weight: about 1.2 GB in all with llm-jp-3 150M), which took an iPhone's tab down";

/** The rounds as this device runs them (T214): a round without the kernels carries skip (the words of its row) where
 * deviceMemory is not a number. The page decides (both ?bench= and /benchmark/), and the worker writes the row. */
export function roundsHere(rounds, deviceMemory) {
  const unsaid = typeof deviceMemory !== "number";
  // without the kernels NumPy widens them, and without int8 the kernels do (the owner, 2026-09-28: skip it too)
  const widens = (round) => round.without.includes("kernels") || round.without.includes("int8");
  return rounds.map((round) => (unsaid && widens(round) ? { ...round, skip: MEMORY_UNSAID } : round));
}

/** The environment the page can see. Whatever a browser does not tell is left out, never guessed. */
export function environmentOf(navigatorLike = globalThis.navigator, extra = {}) {
  const { hardwareConcurrency, deviceMemory, userAgent } = navigatorLike ?? {};
  return {
    userAgent: userAgent ?? "unknown",
    threads: typeof hardwareConcurrency === "number" ? hardwareConcurrency : undefined,
    memory: typeof deviceMemory === "number" ? deviceMemory : undefined,
    ...extra,
  };
}

/**
 * The Markdown of a benchmark: a line about the machine, then a row per round.
 * rows: [{ name, without, speed, backend, seconds }], environment: what environmentOf() made.
 */
export function benchMarkdown(rows, environment) {
  const machine = [
    environment.model && `**${environment.model}**`,
    // hardwareConcurrency counts logical cores: the software threads the engine used are in the backend column
    environment.threads !== undefined && `${environment.threads} logical cores`,
    environment.memory !== undefined && `${environment.memory} GB or more`,
    environment.pyodide && `Pyodide ${environment.pyodide}`,
    // T176: the site's version (the commit it was built from), so that a report says which shaders and kernels ran
    environment.build && `site ${environment.build}`,
    environment.site,
  ].filter(Boolean).join(" · ");
  // no rounds, no table: a /benchmark/ report without its model section (T134) is no row of reportsTable()
  const table = rows.length ? [...roundsTable(rows), ""] : [];
  return [`### Pyodide LLM benchmark`, "", machine, "", ...table, `<sub>${environment.userAgent}</sub>`].join("\n");
}

/** The rounds' table (its lines): a row per round; a skipped round (T214) says "skipped" for its speed, and why in
 * the last column. benchMarkdown() and /benchmark/'s model section both write it. */
export function roundsTable(rows) {
  return ["| what ran | tok/s | ready | backend |", "|---|---|---|---|", ...rows.map((row) => {
    if (row.skip !== undefined) return `| ${row.name} | skipped |  | ${tableCell(row.skip)} |`;
    const ready = row.seconds === undefined ? "" : `${number(row.seconds)} s`;
    return `| ${row.name} | ${number(row.speed)} | ${ready} | ${row.backend ?? ""} |`;
  })];
}
