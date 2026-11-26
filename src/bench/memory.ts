// The section "Page memory" (T173): its result and its line of the summary, from what the run found (T301: from the
// page's part, src/benchmark/page-memory.ts, which holds the run itself).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)
import { tableCell } from "./cells.ts";
import { MEMORY_MOST, MEMORY_PAUSE_MS, MEMORY_STEP, mib } from "../benchmark/page-memory.ts";
import type { MemoryData } from "../benchmark/page-memory.ts";
import type { Result } from "./sections.ts";

// how a run ended (the owner chose the words that do not say more than the page knows, 2026-09-27: TASKS.md T173)
function ending(d: MemoryData) {
  switch (d.stop) {
    case "reloaded": return `the tab ended while it grew to ${mib(d.trying!)} and the page was loaded again: the browser ended it, most likely for its memory`;
    case "refused": return `the browser refused to grow it to ${mib(d.trying!)} (${d.why})`;
    case "limit": return d.limit >= MEMORY_MOST ? "it held the most one 32-bit WebAssembly memory holds" : "it held all it was asked to";
    default: return d.why ?? d.stop;
  }
}

/** The section's result for the page: { status, markdown, data } */
export function memoryResult(d: MemoryData): Result<"memory"> {
  if (d.stop === "none") return { status: "none", markdown: d.why!, data: d };
  if (d.stop === "error") return { status: "error", markdown: `${d.why} (after holding ${mib(d.held)})`, data: d };
  if (d.stop === "hidden") return { status: "error", data: d, markdown:
    `stopped: the page was hidden or left (another app, the screen locked, or another page) while it grew to ${mib(d.trying!)}, ` +
    `after holding ${mib(d.held)}. A phone ends a hidden page far sooner than one in front, so this is not the limit: ` +
    "run it again and keep this page in front until it ends." };
  const room = d.maximum === null ? ` The browser refused a memory with room for ${mib(d.limit)}, so it was made without a maximum.` : "";
  return { status: "ok", data: d, markdown: [
    `One WebAssembly memory grown ${mib(MEMORY_STEP)} at a time and filled with random bytes, ${MEMORY_PAUSE_MS / 1000} s apart, up to ${mib(d.limit)}. ` +
      `What the page used before it began is not counted.${room}`, "",
    "| held | how it ended |", "|---:|---|",
    `| ${mib(d.held)} | ${tableCell(ending(d))} |`,
  ].join("\n") };
}

/** The section's line of the report's summary (T185) */
export function memorySummary(d: MemoryData | undefined) {
  if (!d || d.stop === "none" || d.stop === "error" || d.stop === "hidden") return [];
  return [`Page memory: held ${mib(d.held)}; ${ending(d)}`];
}
