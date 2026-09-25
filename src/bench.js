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
    // hardwareConcurrency counts logical cores: the software threads the engine used are in the backend column
    environment.threads !== undefined && `${environment.threads} logical cores`,
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

// T91: where a visitor sends the result, and the one table the results make.
export const REPOSITORY = "takano32/pyodide-llm";
/** The questions of .github/ISSUE_TEMPLATE/benchmark.md that the page cannot answer. The answer goes after the colon;
 * what is in parentheses is an example, and is not an answer when it is left there. */
export const QUESTIONS = [
  ["Device", "e.g. Pixel 8, MacBook Air M2, a desktop with a Ryzen 7 7700"],
  ["OS", "e.g. Android 16, macOS 26, Windows 11"],
  ["Browser", "e.g. Chrome 148, Safari 26, Firefox 150"],
];

/** The body of a benchmark issue: the three questions, then the page's Markdown as it is. */
export function reportBody(markdown) {
  return [...QUESTIONS.map(([name, example]) => `**${name}**: (${example})`), "",
          "<!-- the page's Markdown, as the page wrote it: please leave it as it is -->", markdown].join("\n");
}

/** The address of a new issue with the template, the title and the body filled in. */
export function reportUrl(markdown, environment) {
  const query = new URLSearchParams({ template: "benchmark.md", title: `Benchmark: ${environment.model ?? "a model"}`,
                                      body: reportBody(markdown) });
  return `https://github.com/${REPOSITORY}/issues/new?${query}`;
}

const cells = (line) => line.split("|").slice(1, -1).map((cell) => cell.trim());

/** What one issue says: the answers, the model, the logical cores, and the rows of its table. undefined when the body
 * has no table of the page's. */
export function parseReport(body) {
  const lines = body.replace(/\r/g, "").split("\n");
  const answer = (name) => {
    const line = lines.find((text) => text.startsWith(`**${name}**:`));
    const value = line?.slice(`**${name}**:`.length).trim() ?? "";
    return /^\(e\.g\./.test(value) ? "" : value;  // the example left in place is no answer
  };
  const head = lines.findIndex((line) => line.startsWith("| what ran |"));
  if (head < 0) return undefined;
  const rows = [];
  for (const line of lines.slice(head + 2)) {
    if (!line.startsWith("|")) break;
    const [name, speed, ready, backend] = cells(line);
    rows.push({ name, speed: Number(speed), ready, backend });
  }
  const machine = lines.find((line) => line.startsWith("**") && line.includes(" · ") && !line.includes("**:")) ?? "";
  return { device: answer("Device"), os: answer("OS"), browser: answer("Browser"),
           model: /^\*\*(.+?)\*\*/.exec(machine)?.[1] ?? "", cores: /(\d+) logical cores|(\d+) threads/.exec(machine)?.slice(1).find(Boolean) ?? "",
           rows };
}

/** The table of the visitors' reports (T83's 30-measurements.md): one row per issue, from its "everything" round. */
export function reportsTable(issues) {
  const out = ["| device | OS | browser | model | logical cores | tok/s | without the kernels | backend | report |",
               "|---|---|---|---|---:|---:|---:|---|---|"];
  for (const { number: id, url, body } of issues) {
    const report = parseReport(body ?? "");
    if (!report) continue;
    const row = (name) => report.rows.find((r) => r.name === name);
    const all = row("everything") ?? report.rows.at(-1), plain = row("without the kernels") ?? row("NumPy only");
    out.push(`| ${report.device || "?"} | ${report.os || "?"} | ${report.browser || "?"} | ${report.model || "?"} | ` +
             `${report.cores || "?"} | ${number(all?.speed)} | ${number(plain?.speed)} | ${all?.backend ?? ""} | [#${id}](${url}) |`);
  }
  return out.join("\n");
}
