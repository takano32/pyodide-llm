// The benchmark of T45: what this browser does with this model, as a table someone can paste into an issue.
// A plain module, so that Node can import it and test it (the page and tests/bench.mjs both use it).

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

const number = (value, digits = 1) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "?");

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
    environment.threads !== undefined && `${environment.threads} threads`,
    environment.memory !== undefined && `${environment.memory} GB or more`,
    environment.pyodide && `Pyodide ${environment.pyodide}`,
    environment.site,
  ].filter(Boolean).join(" · ");
  const head = ["| what ran | tok/s | ready | backend |", "|---|---|---|---|"];
  const body = rows.map((row) => {
    const ready = row.seconds === undefined ? "" : `${number(row.seconds)} s`;
    return `| ${row.name} | ${number(row.speed)} | ${ready} | ${row.backend ?? ""} |`;
  });
  return [`### Pyodide LLM benchmark`, "", machine, "", ...head, ...body, "",
          `<sub>${environment.userAgent}</sub>`].join("\n");
}
