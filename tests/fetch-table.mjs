// fetch-table.mjs (T107): the JSON lines of tests/e2e.mjs runs with E2E_QUERY=hfParts=..&hfConnections=.., as one
// table per model: for each setting the median of its runs, the fastest first. "fetch" is the download less the
// conversion (the worker converts each part as it arrives, so the download's seconds hold both).
//
//   node tests/fetch-table.mjs results.jsonl
import fs from "node:fs";

const median = (xs) => {
  const sorted = [...xs].sort((a, b) => a - b), middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const runs = fs.readFileSync(process.argv[2], "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
for (const model of [...new Set(runs.map((run) => run.model))]) {
  const settings = new Map();
  for (const run of runs.filter((run) => run.model === model)) {
    if (!settings.has(run.query)) settings.set(run.query, []);
    settings.get(run.query).push(run);
  }
  const rows = [...settings].map(([query, list]) => {
    const good = list.filter((run) => run.ok && run.load?.download);
    const fetch = good.map((run) => run.load.download - (run.load.convert ?? 0));
    return { query, runs: `${good.length} / ${list.length}`, fetch: good.length ? median(fetch) : NaN,
      download: good.length ? median(good.map((run) => run.load.download)) : NaN,
      ready: good.length ? median(good.map((run) => run.readySeconds)) : NaN,
      spread: good.length ? `${Math.min(...fetch).toFixed(1)}–${Math.max(...fetch).toFixed(1)}` : "" };
  }).sort((a, b) => (a.fetch || Infinity) - (b.fetch || Infinity));
  console.log(`\n### ${model}\n`);
  console.log("| setting | runs | fetch s (median) | fetch s (range) | download s | ready s |");
  console.log("|---|---:|---:|---:|---:|---:|");
  for (const row of rows) {
    console.log(`| ${row.query} | ${row.runs} | ${row.fetch.toFixed(1)} | ${row.spread} | ${row.download.toFixed(1)} | ${row.ready.toFixed(1)} |`);
  }
}
