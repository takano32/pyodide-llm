// The sections "Storage" (T137) and "Line" (T118).
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
// (T643.11: each asks its worker and hands the answers to src/bench/storage.ts's tables)
import { MODELS } from "../models.ts";
import { storageNone, storageResult, lineResult } from "../bench.ts";
import { BASE, $ } from "./dom.ts";
import type { Result } from "../bench.ts";
import { ask, worker, model } from "./section.ts";

export async function storage(): Promise<Result> {
  const w = worker("benchmark/storage.js");
  try {
    const none = storageNone(await ask(w, { step: "info" }));
    if (none) return none;
    return storageResult(await ask(w, { step: "run", mib: Number(($("size") as HTMLSelectElement).value) }));
  } finally {
    w.terminate();
  }
}

// the line: the first part of the chosen model of the site, and 32 MiB of a model on huggingface.co (the one of the
// list the line section names; any large file would do)
const LINE_MODEL = "hf-qwen2.5-0.5b-instruct", RATES = [1, 4, 8], PACED_SECONDS = 4;
export async function line(): Promise<Result> {
  const entry = model();
  const remote: any = MODELS.find((m: any) => m.id === LINE_MODEL) ?? MODELS.find((m: any) => m.hf && m.download > 64e6);
  const w = worker("benchmark/sections.js");
  try {
    return lineResult(await ask(w, { step: "line", rates: RATES, seconds: PACED_SECONDS,
      site: { url: `${location.origin}${BASE}models/${entry.checkpoint}.000`, bytes: Math.min(8 << 20, entry.bytes), range: false },
      hf: { url: `https://huggingface.co/${remote.hf.repo}/resolve/${remote.hf.revision}/${remote.hf.weights}`, bytes: 32 << 20 } }));
  } finally {
    w.terminate();
  }
}
