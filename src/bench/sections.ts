// /benchmark/'s sections: their names in the page's order, what a section's result is, their titles and the words of
// their states (T643.11: from the page's parts, src/benchmark/dom.ts and report.ts, so that the tables and the report
// that read them run in Node).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)
import type { CpuMeasured } from "./cpu.ts";
import type { Bridge, GpuFound } from "./gpusection.ts";
import type { MemoryData } from "../benchmark/page-memory.ts";
import type { DeviceFound, LineFound, StorageRun } from "../runtime/contracts/benchmark.ts";
import type { PathsFound } from "./path.ts";
import type { BenchRow } from "../runtime/contracts/messages.ts";

export const SECTIONS = ["device", "cpu", "model", "gpu", "storage", "line", "memory"] as const;
export type Name = typeof SECTIONS[number];
// ok: measured; none: the browser has no such thing (said, and the run goes on); wrong: a result that cannot be
// right (a shader against JavaScript, a piece read back); error: a section that failed
// said (T227): the lines of the Markdown with which the section itself says what went wrong, for the report's warnings
// data (T643.15): what the section found, which its tables and its line of the summary are made of, by its name
export type SectionData = { device: DeviceFound; cpu: CpuMeasured; model: { rows: BenchRow[]; paths: PathsFound };
  gpu: { found: GpuFound; bridge: Bridge; lost: string | undefined }; storage: StorageRun; line: LineFound; memory: MemoryData };
export type Result<Section extends Name = Name> = { status: "ok" | "none" | "wrong" | "error"; markdown: string; data?: SectionData[Section]; said?: string[] };
// what the sections found, by their names
export type Results = { [Section in Name]?: Result<Section> };

// the words of a state: shown, on the page's label and in the report's titles; said, after "the CPU section " (T157:
// why the GPU has no CPU beside it) and as a warning of a section that failed as a whole (T227)
export const STATE = { ok: { shown: "done" }, none: { shown: "not in this browser", said: "found nothing to run on here" },
  wrong: { shown: "WRONG", said: "computed something wrong" }, error: { shown: "failed", said: "failed" } };

export const TITLES: Record<Name, string> = { device: "This browser", cpu: "CPU", model: "Model", gpu: "GPU", storage: "Storage", line: "Line", memory: "Page memory" };
