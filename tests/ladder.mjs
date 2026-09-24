// T84: the Pythia ladder, one design at five sizes, from the JSON lines of the huggingface jobs of browsers.yml
// (tests/e2e.mjs writes them; the small job has 70M to 410M, the large one 1B and 1.4B, so give both files).
// If tok/s is bound by memory, int8 megabytes × tok/s (the weights read per second) stays about the same up the
// ladder; if it is bound by computing, it does too, since int8 is one multiply-add per byte (AGENTS.md). Where it
// falls is where something else costs: the attention, the calls from Python, the memory of the machine.
//
//   node tests/ladder.mjs huggingface-small/results.jsonl huggingface-large/results.jsonl
import fs from "node:fs";
import { MODELS, modelBytes } from "../src/models.js";
import { readResults } from "./summary.mjs";

export const LADDER = ["hf-pythia-70m", "hf-pythia-160m", "hf-pythia-410m", "hf-pythia-1b", "hf-pythia-1.4b"];

const fixed = (value, digits = 1) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "");

/** One row per rung that ran (the last run of each, if a model ran twice), as a Markdown table. */
export function ladderMarkdown(results) {
  const last = new Map(results.filter((r) => LADDER.includes(r.model)).map((r) => [r.model, r]));
  if (!last.size) return "No Pythia run was recorded.\n";
  const rows = LADDER.filter((id) => last.has(id)).map((id) => {
    const r = last.get(id), entry = MODELS.find((m) => m.id === id);
    const megabytes = modelBytes(entry) / 1e6;
    const convert = r.load?.convert, download = r.load?.download;
    return `| ${entry.name} | ${fixed(megabytes, 0)} | ${fixed(r.readySeconds)} | ${fixed(download)} | ${fixed(convert)} | ` +
      `${fixed(r.tokPerSecond)} | ${fixed(r.tokPerSecond * megabytes / 1000, 2)} | ${r.heapMB ?? ""} | ${r.ok ? "" : "failed"} |`;
  });
  const browser = [...new Set([...last.values()].map((r) => `${r.engine} ${r.browserVersion ?? ""}`.trim()))].join(", ");
  return [
    "| model | int8 MB | ready (s) | fetch and convert (s) | of which converting (s) | tok/s | GB/s of weights | heap MB | |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ...rows,
    "",
    `${browser}; ${[...new Set([...last.values()].map((r) => r.os))].join(", ")}.`,
    "",
  ].join("\n");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const results = process.argv.slice(2).flatMap((file) => readResults(fs.readFileSync(file, "utf8")));
  process.stdout.write(ladderMarkdown(results));
}
