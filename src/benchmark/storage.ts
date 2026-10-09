// The sections "Storage" (T137) and "Line" (T118).
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import { MODELS } from "../models.js";
import { tableCell, unmeasured } from "../bench.js";
import { BASE, $, type Result } from "./dom.ts";
import { ask, worker, fixed, model } from "./section.ts";

// the storage section of T137's stage 0 (the download it is compared with: huggingface.co from Japan, T123, T136)
const DOWNLOAD_MBPS = 8.3;
export async function storage(): Promise<Result> {
  const w = worker("benchmark/storage.js");
  try {
    const info = await ask(w, { step: "info" });
    if (info.error) return { status: "error", markdown: info.error };
    if (!info.result.syncHandle) return { status: "none", markdown: "no private file system that a worker can write in place here" };
    const answer = await ask(w, { step: "run", mib: Number(($("size") as HTMLSelectElement).value) });
    if (answer.error) return { status: "error", markdown: answer.error };
    const r = answer.result, mb = (r.mib * 2 ** 20) / 1e6;
    const row = (name: string, t: number, flush?: number) => `| ${name} | ${fixed(t, 2)} | ${flush === undefined ? "" : fixed(flush, 2)} | ${fixed(mb / t, 0)} | ${fixed(mb / t / DOWNLOAD_MBPS, 0)}× |`;
    return { status: r.read.wrong ? "wrong" : "ok", data: r, markdown: [
      `${r.mib} MiB in ${r.pieces} pieces of 8 MiB; opening a file ${fixed(r.openSeconds * 1000)} ms; a second handle on it: ${r.secondHandle}; opened again after close: ${r.reopen}`, "",
      `| writes | s | of it flushing, s | MB/s | × a download of ${DOWNLOAD_MBPS} MB/s |`, "|---|---:|---:|---:|---:|",
      row("in order, one flush", r.sequential.seconds, r.sequential.flushSeconds),
      row("far apart, one flush", r.scattered.seconds, r.scattered.flushSeconds),
      row("far apart, a flush and a record each piece", r.scatteredFlushEach.seconds, r.scatteredFlushEach.flushSeconds),
      row(`read back in order${r.read.wrong ? ` (${r.read.wrong} pieces WRONG)` : ", every piece right"}`, r.read.seconds),
    ].join("\n") };
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
    const answer = await ask(w, { step: "line", rates: RATES, seconds: PACED_SECONDS,
      site: { url: `${location.origin}${BASE}models/${entry.checkpoint}.000`, bytes: Math.min(8 << 20, entry.bytes), range: false },
      hf: { url: `https://huggingface.co/${remote.hf.repo}/resolve/${remote.hf.revision}/${remote.hf.weights}`, bytes: 32 << 20 } });
    if (answer.error) return { status: "error", markdown: answer.error };
    const r = answer.result;
    const fetched = (name: string, f: any) => f?.error ? `| ${name} | ${tableCell(unmeasured(f.error))} | | |`
      : `| ${name} | ${fixed(f.bytes / 1e6)} MB | ${fixed(f.firstByteMs, 0)} ms | ${fixed(f.MBps)} MB/s |`;
    const lines = ["| fetched | size | first byte | speed |", "|---|---:|---:|---:|",
      fetched("this site: a part of the model", r.site), fetched("huggingface.co: a range of a model", r.hf)];
    if (r.paced.length) {
      lines.push("", "| huggingface.co read no faster than | got | within 10% |", "|---:|---:|---|",
        ...r.paced.map((p: any) => p.slower ? `| ${p.rate} MB/s | the line is slower | |` : p.error ? `| ${p.rate} MB/s | ${tableCell(unmeasured(p.error))} | |`
          : `| ${p.rate} MB/s | ${fixed(p.MBps, 2)} MB/s | ${Math.abs(p.MBps - p.rate) <= 0.1 * p.rate ? "yes" : "no"} |`));
    }
    const failed = r.site?.error && r.hf?.error;
    return { status: failed ? "error" : "ok", data: r, markdown: lines.join("\n") };
  } finally {
    w.terminate();
  }
}
