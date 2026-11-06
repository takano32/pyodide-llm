// tests/bench.ts's part (T643.15): what /benchmark/ keeps of its CPU section for the model page (src/bench/cpu.ts's
// cpuSpeed(), which src/benchmark/run.ts writes under CPU_SPEED_KEY), and the line of quantizing of the matrix × vector
// table (src/bench/gpu.ts). Last, so that the calls before it are as they were.

import assert from "node:assert/strict";
import { cpuSpeed, matVecTable } from "../../src/bench.ts";
import { cpuSection } from "./cpu.ts";
import { aBandwidths, aCheck, aCeilings } from "./summary.ts";

// the fastest token's reading of the weights and its threads (T156), and a prompt's G MAC/s at its fastest: its model's
// layers' multiply-adds a token over its ms (2.77 ms with 4 threads)
const measured = { status: "ok", data: { ...cpuSection.data, layerWeights: 121634816 } };
assert.deepEqual(cpuSpeed(measured), { GBps: 28.7, threads: 4, promptGMACs: 121634816 / (2.77 / 1000) / 1e9 });
assert.deepEqual(cpuSpeed(cpuSection), { GBps: 28.7, threads: 4, promptGMACs: undefined }, "no layers' weights, no G MAC/s");
// nothing where the page was not isolated (one thread, no page's), the section did not measure a token, or did not run
assert.equal(cpuSpeed({ status: "ok", data: { ...measured.data, shared: false } }), undefined);
assert.equal(cpuSpeed({ status: "error", markdown: "stopped" }), undefined);
assert.equal(cpuSpeed(undefined), undefined);

// T643.15: the matrix × vector table's line of quantizing, whose model's numbers come from gpu/models.ts's table (16 layers
// of Llama 3.2 1B, each 8192 wide at the most): a token of it quantizes 16 × 4 + 1 vectors, at its w1's ms each
const quantizing = matVecTable(aBandwidths, aCheck, aCeilings, {}).find((line) => line.startsWith("**Quantizing the vector**"));
assert.equal(quantizing, "**Quantizing the vector** of a packed row (QUANTIZE, one dispatch; not in the packed rows): llm-jp-3 150M w1 0.041 ms, " +
  "Llama 3.2 1B w1 0.054 ms, Llama 3.2 1B classifier 0.067 ms; a token of Llama 3.2 1B quantizes 65 vectors (16 of them 8192 wide), 3.51 ms or more");
