// The GPU section's tables of what a token costs besides its weights, the device's ceilings and a prompt all at once
// (T643.11: from the page's part, src/benchmark/gpu.ts).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)
import { number as fixed, threadsOf, timesFaster, noRatios, tableCell, unmeasured } from "./cells.ts";
import type { Ratios, Step } from "./cells.ts";
import type { Baseline } from "./cpu.ts";
import type { Result } from "./sections.ts";
import type { CpuRow, GpuCeiling, GpuCeilings, Overhead, PromptFound, PromptRow, Verdict, Verdicts } from "../runtime/contracts/benchmark.ts";

/** What a token costs on the GPU besides reading its weights (T135: the phones' 18 to 20 ms, T94). */
export function overheadMarkdown(s?: Step<Overhead>): string[] {
  if (!s || s.error) return [`**Besides the weights**: ${unmeasured(s?.error)}`];
  const o = s.result!;
  return ["| besides the weights | GPU ms |", "|---|---:|",
    `| ${o.dispatches} dispatches that do nothing, one submission | ${fixed(o.emptyDispatches, 2)} |`,
    `| one submission, not waited for | ${fixed(o.submitOnly, 3)} |`,
    `| one submission, waited for | ${fixed(o.submitAndWait, 2)} |`,
    `| the token's id read back (4 bytes) | ${fixed(o.readToken, 2)} |`,
    `| every logit read back (${o.vocab} floats) | ${fixed(o.readLogits, 2)} |`];
}
// T168: the device's ceilings, each a loop of that alone (src/runtime/shaders.js, the method of clpeak): what the
// prompt's shaders are held against
const percent = (part: number, whole: number) => `${fixed((100 * part) / whole, 1)}%`;
/** T168: the device's ceilings as a table. */
export function ceilingsMarkdown(s?: Step<GpuCeilings>): string[] {
  if (!s || s.error) return [`**The device's ceilings**: ${unmeasured(s?.error)}`];
  const c = s.result!;
  // three digits where it is small (a fallback adapter's reads of the workgroup's memory are well under 1 GB/s); an
  // unsteady one (its 2n not about twice its n) says so, and the prompt is not held against it
  const cell = (v: GpuCeiling, key: "GFLOPS" | "GOPS" | "GBps", unit: string) => (v.none ?? (v.error ? tableCell(`failed: ${v.error}`)
    : `${v.unsteady ? "unsteady: " : ""}${v[key]! < 100 ? v[key]!.toPrecision(3) : fixed(v[key], 0)} ${unit}${v.shape ? ` (${v.shape})` : ""}`));
  return [`**The device's ceilings** (each a loop of that alone; the multiply-adds in two shapes, the faster shown)${c.fallback ? ". A fallback adapter: its ceilings are no GPU's, and the prompt is not held against them" : ""}:`, "",
    "| ceiling | GPU |", "|---|---:|",
    `| f32 multiply-adds | ${cell(c.f32, "GFLOPS", "GFLOPS")} |`,
    `| f16 multiply-adds | ${cell(c.f16, "GFLOPS", "GFLOPS")} |`,
    `| int8 dots (dot4I8Packed) | ${cell(c.dot4, "GOPS", "GOPS")} |`,
    `| reading the workgroup's memory (16-byte reads) | ${cell(c.shared, "GBps", "GB/s")} |`,
    `| reading a buffer${c.global.MiB ? ` (${fixed(c.global.MiB, 0)} MiB)` : ""} | ${cell(c.global, "GBps", "GB/s")} |`];
}
// what a row of the prompt is held against: the int8 dots for a packed shader, else the f32 multiply-adds (the f16
// tiles of llama.cpp hold halves in the workgroup's memory but multiply and add in f32), and at few tokens the
// buffer's reads, whichever is lower: an int8 weight and its share of a scale are 1.125 bytes, read once for the
// row's T tokens, 2T FLOPs. Past 100%: the ceiling was read low
function ceilingOf(row: PromptRow, c?: GpuCeilings) {
  if (!c || c.fallback) return "";
  const steady = (v: GpuCeiling | undefined, key: "GFLOPS" | "GOPS" | "GBps") => (v && !v.unsteady ? v[key] : undefined);
  const [name, compute] = row.packed ? ["int8 dots", steady(c.dot4, "GOPS")] : ["f32", steady(c.f32, "GFLOPS")];
  const reads = steady(c.global, "GBps"), bound = reads ? (reads * 2 * row.tokens!) / 1.125 : undefined;
  const [what, v] = bound && (!compute || bound < compute) ? ["reading", bound] : [name, compute];
  if (!v) return "";
  return `${percent(row.GFLOPS!, v)} of ${what}${row.GFLOPS! > v ? " (past it: the ceiling read low)" : ""}`;
}
/** A prompt's tokens at once on the GPU against the CPU section's blocks of 16 (the same made-up model: two layers of
 * Llama 3.2 1B's width, no classifier), by every shader of it (T146: the tiled ones beside T135's batched one), with
 * the GFLOPS of each (a multiply and an add for each weight and token) and the CPU's GOPS from the same count; and
 * how many times faster each is than the CPU's fastest count of threads (T157). cpu: the CPU section's result, whose
 * rows give its prompt's times. */
export function promptMarkdown(s?: Step<PromptFound>, check?: Verdicts, ceilings?: GpuCeilings, baseline?: Baseline, gpu?: Ratios, cpu?: Result<"cpu">): string[] {
  if (!s || s.error) return [`**A prompt all at once**: ${unmeasured(s?.error)}`];
  const r = s.result!, cpuPrompt = baseline?.prompt, none = noRatios(gpu);
  const cpuRows = (cpu?.data?.rows ?? []).filter((row: CpuRow) => row.promptMsPerToken);
  const cpuTimes = cpuRows.length ? cpuRows.map((row: CpuRow) => `${fixed(row.promptMsPerToken, 2)} ms with ${row.threads} (${fixed((2 * r.weights) / (row.promptMsPerToken! / 1000) / 1e9, 0)} GOPS)`).join(", ")
    : "run the CPU section for it";
  // a shader the check found WRONG (or could not check) says so on its rows, and is never the fastest (T146's review)
  const verdict = (row: PromptRow): Verdict | undefined => check?.[row.shader.replace(/, again at the end$/, "")];
  const wrong = (row: PromptRow) => { const v = verdict(row); return v !== undefined && !v.ok; };
  const fastest = (tokens: number) => {
    const best = r.rows.filter((row) => row.tokens === tokens && !row.again && !wrong(row)).sort((a, b) => a.msPerToken! - b.msPerToken!)[0];
    return best ? `at ${tokens} tokens ${best.shader}, ${fixed(best.msPerToken, 2)} ms a token (${fixed(best.GFLOPS, 0)} GFLOPS)` : "";
  };
  const name = (row: PromptRow) => (wrong(row) ? `${row.shader} (WRONG in the check)` : row.shader);
  return [`**A prompt** (the CPU section's model: two layers of Llama 3.2 1B's width, ${fixed(r.GB, 2)} GB, ${fixed(r.weights / 1e6, 0)} million weights, no classifier); ` +
    `the CPU, 16 tokens at once: ${cpuTimes}. A tiled shader's name says whose form it takes (llama.cpp's, TensorFlow.js's, ONNX Runtime's) and the rows × tokens of a workgroup's tile. ` +
    "The batched shader is measured again at the end: a device that slows down as it warms up shows it there. " +
    "Of the ceiling: its GFLOPS against the lower of the device's ceiling above for what the shader multiplies with (f32, or the int8 dots) and the buffer's reads at that many tokens. " +
    (!cpuPrompt ? `GPU ÷ CPU: not measured (${baseline?.why ?? "the CPU section measured no prompt"})`
      : none ? `GPU ÷ CPU: ${none}`
      : `GPU ÷ CPU: against the CPU's ${fixed(cpuPrompt.msPerToken, 2)} ms a token with ${threadsOf(cpuPrompt)}, 16 at once (its fastest). ` +
        "The GPU rows are the matrices only, the CPU's the whole forward pass. Above 1× the GPU is faster"), "",
    "| shader | tokens at once on the GPU | GPU ms | GPU ms a token | GFLOPS | of the ceiling | GPU ÷ CPU |", "|---|---:|---:|---:|---:|---:|---:|",
    ...r.rows.map((row) => (row.none || row.error ? `| ${name(row)} | ${tableCell(row.none ?? `failed: ${row.error}`)} | | | | | |`
      : `| ${name(row)} | ${row.tokens} | ${fixed(row.ms)} | ${fixed(row.msPerToken, 2)} | ${fixed(row.GFLOPS, 0)} | ${ceilingOf(row, ceilings)} | ${none || wrong(row) ? "" : timesFaster(cpuPrompt?.msPerToken, row.msPerToken)} |`)),
    "", `**Fastest on the GPU** (of the shaders the check found right): ${[fastest(16), fastest(64)].filter(Boolean).join("; ") || "nothing measured"}`];
}
