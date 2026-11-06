// The section "This browser": what the browser has, as a table (T643.11: from the page's part, src/benchmark/cpu.ts).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)
import { yes } from "./cells.ts";
import type { Result } from "./sections.ts";
import type { Answer, DeviceFound } from "../runtime/contracts/benchmark.ts";

/** The section "This browser" from its worker's answer: the features, or what failed. */
export function deviceResult(answer: Answer<DeviceFound>): Result<"device"> {
  if (answer.error) return { status: "error", markdown: answer.error };
  const r = answer.result!;
  const gb = (bytes: number | undefined) => (typeof bytes === "number" ? `${(bytes / 1e9).toFixed(1)} GB` : "?");
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
}
