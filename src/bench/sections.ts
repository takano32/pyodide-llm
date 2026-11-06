// /benchmark/'s sections: their names in the page's order, what a section's result is, their titles and the words of
// their states (T643.11: from the page's parts, src/benchmark/dom.ts and report.ts, so that the tables and the report
// that read them run in Node).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)

export const SECTIONS = ["device", "cpu", "model", "gpu", "storage", "line", "memory"] as const;
export type Name = typeof SECTIONS[number];
// ok: measured; none: the browser has no such thing (said, and the run goes on); wrong: a result that cannot be
// right (a shader against JavaScript, a piece read back); error: a section that failed
// said (T227): the lines of the Markdown with which the section itself says what went wrong, for the report's warnings
export type Result = { status: "ok" | "none" | "wrong" | "error"; markdown: string; data?: any; said?: string[] };

export const STATE = { ok: "done", none: "not in this browser", wrong: "WRONG", error: "failed" };

export const TITLES: Record<Name, string> = { device: "This browser", cpu: "CPU", model: "Model", gpu: "GPU", storage: "Storage", line: "Line", memory: "Page memory" };
