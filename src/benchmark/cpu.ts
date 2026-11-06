// The sections "This browser" and "CPU" (src/runtime/benchmark/sections.ts's steps).
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
// (T643.11: each asks its worker and hands the answer to src/bench/'s table, deviceResult() and cpuResult())
import { threadCounts, deviceResult, cpuResult } from "../bench.ts";
import type { Result } from "../bench.ts";
import { ask, worker } from "./section.ts";

// ---- the sections: each one starts its worker, asks, ends it, and says what it found as Markdown
export async function device(): Promise<Result> {
  const w = worker("benchmark/sections.js");
  try {
    return deviceResult(await ask(w, { step: "device" }));
  } finally {
    w.terminate();
  }
}

export async function cpu(): Promise<Result> {
  const w = worker("benchmark/sections.js");
  try {
    return cpuResult(await ask(w, { step: "cpu", threads: threadCounts(navigator.hardwareConcurrency) }));
  } finally {
    w.terminate();
  }
}
