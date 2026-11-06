// tests/bench.ts's part (T643.13): the CPU section's table and the GPU ÷ CPU baseline (src/bench/cpu.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { cpuBaseline, cpuTable, threadCounts, times, timesFaster } from "../../src/bench.ts";

// T157: the GPU's token table against the CPU section's forward pass. The owner's Android (T134's first report): the
// CPU section 11.2, 10.3 and 7.3 ms a token with 1, 2 and 4 threads (18.9, 20.5, 28.7 GB/s), its prompt 5.80, 4.50
// and 2.77; the GPU's packed token of Llama 3.2 1B 48.4 ms, which the old column called 1.44×
const cpuSection = { status: "ok", data: { shared: true, rows: [
  { asked: 1, threads: 1, msPerToken: 11.2, GBps: 18.9, promptMsPerToken: 5.80 },
  { asked: 2, threads: 2, msPerToken: 10.3, GBps: 20.5, promptMsPerToken: 4.50 },
  { asked: 4, threads: 4, msPerToken: 7.3, GBps: 28.7, promptMsPerToken: 2.77 },
  { asked: 8, threads: 3, none: "the browser did not start that many software threads" }] } };
const baseline = cpuBaseline(cpuSection);
assert.deepEqual(baseline, { token: { threads: 4, msPerToken: 7.3, GBps: 28.7, isolated: true }, prompt: { threads: 4, msPerToken: 2.77, isolated: true } });
// the fastest of each on its own: a device whose prompt is fastest at another count than its token
assert.equal(cpuBaseline({ status: "ok", data: { rows: [{ threads: 1, msPerToken: 5, GBps: 40, promptMsPerToken: 3 },
  { threads: 2, msPerToken: 6, GBps: 35, promptMsPerToken: 2 }] } }).prompt.threads, 2);
// nothing to hold against: why, and no estimate in its place
assert.equal(cpuBaseline(undefined).why, "run the CPU section for it");
assert.equal(cpuBaseline({ status: "error", markdown: "stopped" }).why, "the CPU section failed");
assert.equal(cpuBaseline({ status: "wrong", data: cpuSection.data }).why, "the CPU section computed something wrong");
assert.equal(cpuBaseline({ status: "none", markdown: "no WebAssembly SIMD" }).why, "the CPU section found nothing to run on here");
assert.ok(cpuBaseline({ status: "ok", data: { rows: [{ asked: 1, none: "x" }] } }).why);
// T163: the CPU section's table with the ceilings beside it: a token's reads against reading alone with as many threads
// (211 MB of checkpoint and 23.4 MB of corrections: 18.9 GB/s of checkpoint reads 21.0, 56% of 37.8), the prompt's
// G MAC/s with one thread against relaxed_dot with its loads (121.6 M weights in the layers: 5.80 ms a token is 21.0)
const ceilingsOf = (read, extra = {}) => ({ read, dot: { GMACs: 43.4 }, dotRegisters: { GMACs: 67.9 }, fma: { GMACs: 8.3 }, ...extra });
const cpuResult = { backend: "SIMD kernels", shared: true, megabytes: 211, tokenMegabytes: 234.4, layerWeights: 121634816, rows: cpuSection.data.rows,
  ceilings: ceilingsOf([{ threads: 1, GBps: 37.8 }, { threads: 2, GBps: 41 }, { threads: 4, GBps: 57.4, unsteady: true }, { threads: 8, error: "x | y" }]) };
const cpuLines = cpuTable(cpuResult);
assert.ok(cpuLines.includes("| 1 | 11.2 | 18.9 (56%) | 89.3 | 5.80 | 21.0 (48%) |"), cpuLines.join("\n"));
assert.ok(cpuLines.includes("| 2 | 10.3 | 20.5 (56%) | 97.1 | 4.50 | 27.0 |"), "no share for the prompt past one thread");
assert.ok(cpuLines.includes("| 4 | 7.3 | 28.7 | 137.0 | 2.77 | 43.9 |"), "no share against an unsteady ceiling");
assert.ok(cpuLines.includes("| 8 | the browser did not start that many software threads | | | | |"));
assert.ok(cpuLines.includes("| reading alone | 4 | unsteady: 57.4 GB/s |"));
assert.ok(cpuLines.includes("| reading alone | 8 | failed: x \\| y |"), "a | in a cell");
assert.ok(cpuLines.includes("| relaxed_dot with its two loads (int8) | 1 | 43.4 G MAC/s |"));
assert.ok(cpuLines.includes("| relaxed_dot, registers only | 1 | 67.9 G MAC/s |"));
assert.ok(cpuLines.includes("| f32 multiply + add, registers only | 1 | 8.3 G MAC/s |"));
// Safari: no relaxed SIMD, so no dot ceilings, no share for the prompt, and no corrections read (matmul_q8)
const safari = cpuTable({ ...cpuResult, tokenMegabytes: 211, ceilings: ceilingsOf([{ threads: 1, GBps: 37.8 }], { dot: { none: "no relaxed SIMD in this browser" }, dotRegisters: { none: "no relaxed SIMD in this browser" } }) });
assert.ok(safari.includes("| relaxed_dot, registers only | 1 | not in this browser |"));
assert.ok(safari.includes("| relaxed_dot with its two loads (int8) | 1 | not in this browser |"));
assert.ok(safari.includes("| 1 | 11.2 | 18.9 (50%) | 89.3 | 5.80 | 21.0 |"), safari.join("\n"));
// the ceilings could not start: the forward pass stands, the ceilings say why (T227's review: "failed:", as the GPU
// section's steps say it, so that the report's warnings list it)
const unstarted = cpuTable({ ...cpuResult, ceilings: { error: "Error: Could not fetch ceilings_plain.wasm: 503" } });
assert.ok(unstarted.includes("| 1 | 11.2 | 18.9 | 89.3 | 5.80 | 21.0 |") && unstarted.at(-1).startsWith("failed: Error"), unstarted.join("\n"));
// a report from before T163 (no ceilings, no layerWeights) still makes a table
assert.ok(cpuTable({ ...cpuResult, layerWeights: undefined, ceilings: undefined }).includes("| 1 | 11.2 | 18.9 | 89.3 | 5.80 | ? |"));
// the counts of threads: doubling up to the logical cores, and the cores themselves
assert.deepEqual(threadCounts(8), [1, 2, 4, 8]);
assert.deepEqual(threadCounts(6), [1, 2, 4, 6]);
assert.deepEqual(threadCounts(1), [1]);
assert.deepEqual(threadCounts(undefined), [1, 2, 4]);
// ratios of one run each: one decimal from 1, two significant digits below
assert.equal(timesFaster(2.77, 5.51), "0.50×");
assert.equal(times(1.44), "1.4×");
assert.equal(times(0.0712), "0.071×");
assert.equal(times(277.3), "277×");
assert.equal(timesFaster(undefined, 5.51), "");
assert.equal(timesFaster(2.77, NaN), "");
export { cpuSection, baseline };
