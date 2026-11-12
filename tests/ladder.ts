// T84: the Pythia ladder, one design at eight sizes, from the JSON lines tests/e2e.ts writes (E2E_RESULTS): the
// results.jsonl of a models.yml run that was given the rungs (one job, so one runner and one CPU for every rung: the
// huggingface jobs of browsers.yml split the ladder over two runners, whose CPUs differ, AGENTS.md).
// If tok/s is bound by memory, int8 megabytes × tok/s (the weights read per second) stays about the same up the
// ladder; if it is bound by computing, it does too, since int8 is one multiply-add per byte (AGENTS.md). Where it
// falls is where something else costs: the attention, the calls from Python, the memory of the machine.
// One table for each system and CPU in the files (so the x86-64 run's and the arm64 run's can be given together).
//
//   node tests/ladder.ts models-x64/results.jsonl models-arm64/results.jsonl
import fs from "node:fs";
import { MODELS, modelBytes } from "../src/models.ts";
import { readResults } from "./summary.ts";

export const LADDER = ["hf-pythia-14m", "hf-pythia-70m", "hf-pythia-160m", "hf-pythia-410m", "hf-pythia-1b", "hf-pythia-1.4b",
  "hf-pythia-2.8b", "hf-pythia-6.9b"];

const fixed = (value, digits = 1) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "");

/** The software threads the line under the answer says (it names them only where there are more than one). */
export function threadsOf(meta) {
  if (typeof meta !== "string" || !/tok\/s/.test(meta)) return null;
  return Number(meta.match(/· (\d+) threads/)?.[1] ?? 1);
}

/** The software threads the URL asked for (?threads=N), or null: where the line says fewer, the table says so (T84's review:
 * the 70M of the x86-64 run asked for 4 and ran on 1, and the cell said only 1) */
export const threadsAsked = (query) => Number(/(?:^|&)threads=(\d+)/.exec(typeof query === "string" ? query : "")?.[1]) || null;

/** The tokens of the answer the tok/s is of (an answer that stopped early was timed at shorter positions). */
export const tokensOf = (meta) => (typeof meta === "string" ? Number(meta.match(/· (\d+) tokens ·/)?.[1]) || null : null);

function table(results) {
  const last = new Map(results.map((r) => [r.model, r]));
  const rows = LADDER.filter((id) => last.has(id)).map((id) => {
    const r = last.get(id), entry = MODELS.find((m) => m.id === id);
    const megabytes = modelBytes(entry) / 1e6;
    const convert = r.load?.convert, download = r.load?.download;
    const ran = threadsOf(r.meta), asked = threadsAsked(r.query);
    const threads = ran === null ? "" : asked && asked !== ran ? `${ran} (asked ${asked})` : `${ran}`;
    const share = typeof convert === "number" && r.readySeconds ? 100 * convert / r.readySeconds : null;
    return `| ${entry.name} | ${fixed(megabytes, 0)} | ${fixed(r.readySeconds)} | ${fixed(download)} | ${fixed(convert)} | ` +
      `${fixed(share, 0)} | ${tokensOf(r.meta) ?? ""} | ${fixed(r.tokPerSecond)} | ${threads} | ${fixed(r.tokPerSecond * megabytes / 1000, 2)} | ` +
      `${r.heapMB ?? ""} | ${r.ok ? "" : "failed"} |`;
  });
  const seen = [...last.values()];
  const browser = [...new Set(seen.map((r) => `${r.engine} ${r.browserVersion ?? ""}`.trim()))].join(", ");
  const gpu = seen.some((r) => /on WebGPU/.test(`${r.backend ?? ""} ${r.meta ?? ""}`)) ? "the GPU took a part" : "on the CPU (no WebGPU)";
  return [
    "| model | int8 MB | ready (s) | fetch and convert (s) | of which converting (s) | converting / ready (%) | tokens | tok/s | threads | GB/s of weights | WebAssembly memory MB | |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ...rows,
    "",
    `${browser}; ${seen[0].os}, ${seen[0].cpu ?? "the CPU not recorded"}; ${gpu}.`,
    "",
  ].join("\n");
}

/** One table per system and CPU, one row per rung that ran (the last run of each, if a model ran twice), as Markdown. */
export function ladderMarkdown(results) {
  const rungs = results.filter((r) => LADDER.includes(r.model));
  if (!rungs.length) return "No Pythia run was recorded.\n";
  const groups = new Map();
  for (const r of rungs) {
    const key = `${r.os} ${r.cpu ?? ""}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  return [...groups.values()].map(table).join("\n");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const results = process.argv.slice(2).flatMap((file) => readResults(fs.readFileSync(file, "utf8")));
  process.stdout.write(ladderMarkdown(results));
}
