// The sections "This browser" and "CPU" (public/benchmark/sections.js's steps).
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import { threadCounts, cpuTable } from "../bench.js";
import { type Result } from "./dom.ts";
import { ask, worker, yes } from "./section.ts";

// ---- the sections: each one starts its worker, asks, ends it, and says what it found as Markdown
export async function device(): Promise<Result> {
  const w = worker("benchmark/sections.js");
  try {
    const answer = await ask(w, { step: "device" });
    if (answer.error) return { status: "error", markdown: answer.error };
    const r = answer.result;
    const gb = (bytes: any) => (typeof bytes === "number" ? `${(bytes / 1e9).toFixed(1)} GB` : "?");
    return { status: "ok", data: r, markdown: [
      "| feature | here |", "|---|---|",
      `| logical cores | ${r.cores ?? "?"} |`,
      `| memory the browser says | ${r.memoryGB === null ? "not told" : `${r.memoryGB} GB or more`} |`,
      `| WebAssembly SIMD | ${yes(r.simd)} |`,
      `| relaxed SIMD | ${yes(r.relaxedSimd)} |`,
      `| 64-bit memory (models past 4 GB) | ${yes(r.memory64)} |`,
      `| cross-origin isolated (software threads) | ${yes(r.crossOriginIsolated && r.sharedMemory)} |`,
      `| WebGPU in a worker | ${yes(r.webgpu)} |`,
      `| private file system, writable in place | ${yes(r.opfs && r.syncHandle)} |`,
      `| storage the browser grants | ${gb(r.usage)} used of ${gb(r.quota)} |`,
    ].join("\n") };
  } finally {
    w.terminate();
  }
}

export async function cpu(): Promise<Result> {
  const w = worker("benchmark/sections.js");
  try {
    const answer = await ask(w, { step: "cpu", threads: threadCounts(navigator.hardwareConcurrency) });
    if (answer.error) return { status: "error", markdown: answer.error };
    const r = answer.result;
    if (r.none) return { status: "none", markdown: r.none };
    const c = r.ceilings ?? {};
    const cutShort = [...(c.read ?? []), c.dot, c.dotRegisters, c.fma].some((one: any) => one?.cutShort);
    const notFinite = r.rows.some((row: any) => row.finite === false);
    const said = [...(notFinite ? ["**The logits were not finite numbers: the forward pass computed something wrong.**"] : []),
      ...(cutShort ? ["**A ceiling's loop did not do its work (its checksum or its time): the compiler cut it short, and its number is no ceiling.**"] : [])];
    const lines = [...cpuTable(r), ...said.flatMap((line) => ["", line])];
    return { status: said.length ? "wrong" : "ok", data: r, markdown: lines.join("\n"), said };
  } finally {
    w.terminate();
  }
}
