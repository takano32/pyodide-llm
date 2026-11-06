// The sections "Storage" (T137) and "Line" (T118): their tables, and what their workers' answers say (T643.11: from
// the page's part, src/benchmark/storage.ts).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)
import { number as fixed, tableCell, unmeasured } from "./cells.ts";
import type { Result } from "./sections.ts";
import type { Answer, LineFetched, LineFound, LinePaced, StorageInfo, StorageRun } from "../runtime/contracts/benchmark.ts";

// the storage section of T137's stage 0 (the download it is compared with: huggingface.co from Japan, T123, T136)
export const DOWNLOAD_MBPS = 8.3;

/** Why the storage section does not run here, from its worker's first answer (undefined: it runs). */
export function storageNone(info: Answer<StorageInfo>): Result | undefined {
  if (info.error) return { status: "error", markdown: info.error };
  if (!info.result!.syncHandle) return { status: "none", markdown: "no private file system that a worker can write in place here" };
  return undefined;
}

/** The storage section from its worker's answer to the run: the writes and the read back, or what failed. */
export function storageResult(answer: Answer<StorageRun>): Result {
  // T446 (2): a browser whose private file system has the call but refuses it (Windows' Playwright WebKit: "NotSupportedError:
  // Backend does not support this operation" as the file was opened) lacks the feature as one without the call does:
  // no failure. The model page keeps its conversions in the Cache API there (kept.ts)
  if (answer.error && /^NotSupportedError\b.*\(while opening the file \(createSyncAccessHandle\)\)/.test(answer.error)) {
    return { status: "none", markdown: `no private file system that a worker can write in place here: it refuses to open a file so (${answer.error})` };
  }
  if (answer.error) return { status: "error", markdown: answer.error };
  const r = answer.result!, mb = (r.mib * 2 ** 20) / 1e6;
  const row = (name: string, t: number, flush?: number) => `| ${name} | ${fixed(t, 2)} | ${flush === undefined ? "" : fixed(flush, 2)} | ${fixed(mb / t, 0)} | ${fixed(mb / t / DOWNLOAD_MBPS, 0)}× |`;
  return { status: r.read.wrong ? "wrong" : "ok", data: r, markdown: [
    `${r.mib} MiB in ${r.pieces} pieces of 8 MiB; opening a file ${fixed(r.openSeconds * 1000)} ms; a second handle on it: ${r.secondHandle}; opened again after close: ${r.reopen}`, "",
    `| writes | s | of it flushing, s | MB/s | × a download of ${DOWNLOAD_MBPS} MB/s |`, "|---|---:|---:|---:|---:|",
    row("in order, one flush", r.sequential.seconds, r.sequential.flushSeconds),
    row("far apart, one flush", r.scattered.seconds, r.scattered.flushSeconds),
    row("far apart, a flush and a record each piece", r.scatteredFlushEach.seconds, r.scatteredFlushEach.flushSeconds),
    row(`read back in order${r.read.wrong ? ` (${r.read.wrong} pieces WRONG)` : ", every piece right"}`, r.read.seconds),
  ].join("\n") };
}

/** The line section from its worker's answer: the two fetches and the paced reads, or what failed. */
export function lineResult(answer: Answer<LineFound>): Result {
  if (answer.error) return { status: "error", markdown: answer.error };
  const r = answer.result!;
  const fetched = (name: string, f: LineFetched | null) => f?.error ? `| ${name} | ${tableCell(unmeasured(f.error))} | | |`
    : `| ${name} | ${fixed(f!.bytes! / 1e6)} MB | ${fixed(f!.firstByteMs, 0)} ms | ${fixed(f!.MBps)} MB/s |`;
  const lines = ["| fetched | size | first byte | speed |", "|---|---:|---:|---:|",
    fetched("this site: a part of the model", r.site), fetched("huggingface.co: a range of a model", r.hf)];
  if (r.paced.length) {
    lines.push("", "| huggingface.co read no faster than | got | within 10% |", "|---:|---:|---|",
      ...r.paced.map((p: LinePaced) => p.slower ? `| ${p.rate} MB/s | the line is slower | |` : p.error ? `| ${p.rate} MB/s | ${tableCell(unmeasured(p.error))} | |`
        : `| ${p.rate} MB/s | ${fixed(p.MBps, 2)} MB/s | ${Math.abs(p.MBps! - p.rate) <= 0.1 * p.rate ? "yes" : "no"} |`));
  }
  const failed = r.site?.error && r.hf?.error;
  return { status: failed ? "error" : "ok", data: r, markdown: lines.join("\n") };
}
