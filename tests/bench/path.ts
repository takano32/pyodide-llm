// tests/bench.ts's part (T643.13): the model page's own path, and a check's verdicts (src/bench/path.ts).
// tests/bench.ts runs the parts in their order, and a part reads by name what an earlier part made.

import assert from "node:assert/strict";
import { parseReport, gpuSkipped, layerCheckNumbers, PATH_PROMPTS, PATH_WRITES, pathTable, threadsKey, threadsLine, checkVerdict } from "../../src/bench.ts";
import { markdown, pageText } from "./rounds.ts";
import { cellsOf } from "./gpu.ts";

// T184: the model page's own path, one table: a device whose GPU took the prompts, one whose GPU stopped while timed,
// CI's fallback adapter, a model the GPU does not take, a failure
const cell = (speed, low, high, gpuTokens, unsteady = false) => ({ speed, low, high, gpuTokens, unsteady });
const real = { threads: 4, gpu: { seconds: 4.23, matrices: "TF.js 64×64", attention: "llama.cpp flash" }, status: "prompts of 32 tokens and more on WebGPU, answers on the CPU (faster here)",
  rows: [{ what: "prompt", tokens: 64, chosen: cell(820.4, 800, 830, 64), cpu: cell(612, 600, 640, 0, true), gpu: cell(830, 810, 840, 64) },
         { what: "prompt", tokens: 256, chosen: cell(1300, 1280, 1310, 192), cpu: cell(650, 640, 655, 0), gpu: cell(1310, 1300, 1320, 256) },
         { what: "generation", tokens: 64, chosen: { same: "cpu" }, cpu: cell(80.44, 79.1, 81.2), gpu: { skip: "not on the GPU yet" } }] };
const pathLines = pathTable(real, "llm-jp-3 150M").split("\n");
assert.equal(pathLines[0], "**The model page's path** (llm-jp-3 150M): 4 software threads · WebGPU ready in 4.2 s · matrices by TF.js 64×64 · attention by llama.cpp flash · prompts of 32 tokens and more on WebGPU, answers on the CPU (faster here)");
assert.ok(pathLines.includes("| a prompt of 64 tokens | 820 tok/s (800–830), GPU | 612 tok/s (600–640, unsteady) | 830 tok/s (810–840) | 1.4× |"), pathLines.join("\n"));
assert.ok(pathLines.includes("| a prompt of 256 tokens | 1300 tok/s (1280–1310), GPU 192 of 256 | 650 tok/s (640–655) | 1310 tok/s (1300–1320) | 2.0× |"), pathLines.join("\n"));
assert.ok(pathLines.includes("| writing 64 tokens | same as CPU only | 80.4 tok/s (79.1–81.2) | not on the GPU yet |  |"), pathLines.join("\n"));
const pathRows = pathLines.filter((line) => line.startsWith("|"));
assert.equal(pathRows.length, 2 + real.rows.length, "a header, the separator, a row each: 3 to 6 rows (T185 keeps the report short)");
for (const line of pathRows) assert.equal(cellsOf(line).length, cellsOf(pathRows[0]).length, line);
// the review's must-fix: a GPU that stopped while the sides were timed says so, and no ratio stands (T157)
const stopped = pathTable({ ...real, gpu: { ...real.gpu, lost: "the GPU failed on a block of the prompt" },
  rows: [real.rows[0], { ...real.rows[1], chosen: cell(640, 630, 650, 0), gpu: { skip: "the GPU failed on a block of the prompt" } }] });
assert.ok(stopped.includes("WebGPU stopped while timed: the GPU failed on a block of the prompt") && !stopped.includes("ready in"), stopped);
assert.ok(!stopped.includes("×"), `no ratio where the GPU stopped\n${stopped}`);
assert.ok(stopped.includes("| 650 tok/s (640–655) | the GPU failed on a block of the prompt |  |"), stopped);
// the GPU's cells say why in a few words; the line has the reason where it is no such few words
const fallback = "a fallback adapter: the CPU in the GPU's place";
const onFallback = pathTable({ threads: 2, gpu: { why: fallback },
  rows: [{ what: "prompt", tokens: 64, chosen: { same: "cpu" }, cpu: cell(505, 500, 510, 0), gpu: { skip: fallback } }] });
assert.ok(onFallback.includes("WebGPU: not on a fallback adapter") && onFallback.includes("| a prompt of 64 tokens | same as CPU only | 505 tok/s (500–510) | not on a fallback adapter |  |"), onFallback);
assert.equal(gpuSkipped("no WebGPU in a worker here"), "not here");
assert.equal(gpuSkipped("no GPU adapter here"), "not here");
const float32 = pathTable({ threads: 1, gpu: { why: "float32 weights are not on the GPU yet" },
  rows: [{ what: "prompt", tokens: 64, chosen: { same: "cpu" }, cpu: cell(51, 50, 52, 0), gpu: { skip: "float32 weights are not on the GPU yet" } }] });
assert.ok(float32.includes("WebGPU: float32 weights are not on the GPU yet") && float32.includes("| 51.0 tok/s (50.0–52.0) | not used |  |"), float32);
assert.equal(pathTable({ error: "a | b" }, "x"), "**The model page's path** (x): failed: a \\| b");
assert.ok(!pathTable(undefined).includes("undefined"));
for (const text of [pathTable(real), stopped, onFallback, float32]) assert.ok(!text.includes("undefined") && !text.includes("NaN"), text);
// T190: the page path's threads are the model page's: the count it remembers (same key on both pages), or the search's
// verdicts here, or why there is one thread
const nav = { hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (Linux; Android 15)" };
assert.equal(threadsKey("llm-jp-3-150m", nav), "threads:llm-jp-3-150m:8:8:Mozilla/5.0 (Linux; Android 15)");
assert.equal(threadsKey("x", { hardwareConcurrency: 4, userAgent: "u" }), "threads:x:4::u", "no deviceMemory (Safari, Firefox)");
for (const page of ["index.astro", "benchmark.astro"]) {
  assert.ok(/threadsKey as sharedThreadsKey|threadsKey\(entry\.id, navigator\)/.test(pageText(page)),
    `${page} keys the remembered threads with src/bench.ts's threadsKey`);
}
const headOf = (how, threads = 4) => pathTable({ ...real, threads, how }).split("\n")[0];
assert.ok(headOf({ remembered: true }).includes("4 software threads, as the model page remembers ·"), headOf({ remembered: true }));
assert.ok(headOf({ searched: [[8, 4, 4], [4, 2, 4]] }).includes("4 software threads, searched here (8 or 4: 4, 4 or 2: 4) ·"));
assert.ok(headOf({ unfinished: 120 }).includes("4 software threads: the search had not ended after 120 s ·"));
assert.ok(headOf({ alone: "no shared memory here" }, 1).includes("1 software thread: no shared memory here ·"));
assert.ok(headOf({ searched: [] }).includes("4 software threads ·"), "no search: nothing more");
// T190's review: a software thread that stopped says so (worker/timing.js reads engine.lostThreads: found is 1 then, and the
// search's verdicts would name another count); one that stopped after the count was found, too
assert.ok(headOf({ alone: "a software thread stopped" }, 1).includes("1 software thread: a software thread stopped ·"));
assert.ok(headOf({ remembered: true, stopped: true }).includes("4 software threads, as the model page remembers; a software thread stopped while timed, and one thread went on ·"));
assert.ok(headOf({ searched: [[8, 4, 4]], stopped: true }).includes("searched here (8 or 4: 4); a software thread stopped while timed"));
// the writing on each number of threads, the page's marked, under the table; nothing with one count or none
const counted = pathTable({ ...real, perCount: [{ threads: 1, speed: 107.2, low: 105, high: 108 }, { threads: 2, speed: 151, low: 149, high: 160, unsteady: true },
  { threads: 4, speed: 163.4, low: 160, high: 165 }, { threads: 8, speed: 120, low: 118, high: 121 }] });
assert.equal(counted.split("\n").at(-1), "Writing on each number of software threads (CPU only): 1: 107 tok/s (105–108) · 2: 151 tok/s (149–160, unsteady) · " +
  "4 (the page's): 163 tok/s (160–165) · 8: 120 tok/s (118–121)");
assert.ok(!pathTable({ ...real, perCount: [{ threads: 1, speed: 107, low: 105, high: 108 }] }).includes("Writing on each"));
assert.ok(!pathTable(real).includes("Writing on each"));
assert.equal(threadsLine([]), "");
assert.deepEqual(PATH_PROMPTS, [64, 256]);
assert.equal(PATH_WRITES, 64);
// in the report, under the rounds' table: parseReport() still reads the rounds alone
const withPath = [markdown, pathTable(real, "tiny-lm 29M")].join("\n\n");
assert.deepEqual(parseReport(withPath).rows.map((row) => row.name), ["everything", "without the kernels"]);
// T186: a layer's verdict in one short line: the four quantized vectors summed, the fused form against the norms apart
const quantized = [{ point: "q, k, v", wrong: null, scale: 1.2e-7, apart: 3, of: 2112 }, { point: "o", wrong: null, scale: 2.5e-7, apart: 1, of: 2112 },
  { point: "gate, up", wrong: null, scale: 0, apart: 0, of: 2112 }, { point: "down", wrong: null, scale: 1e-8, apart: 2, of: 2080 }];
assert.equal(layerCheckNumbers({ quantized }), "quantized: scales 2.5e-7, 6 of 8416 off by 1");
assert.equal(layerCheckNumbers({ quantized, sameAsNormsApart: { ulps: 2, apart: 1, stream: 3e-7, bitForBit: false } }),
  "quantized: scales 2.5e-7, 6 of 8416 off by 1; norms apart: 2 ulp, 1 off by 1, stream 3.0e-7");
assert.equal(layerCheckNumbers({ sameAsNormsApart: { ulps: 0, apart: 0, stream: 0, bitForBit: true } }), "norms apart: bit for bit");
assert.equal(layerCheckNumbers({ quantized: [quantized[0], { ...quantized[1], wrong: "far from quantize_x's" }] }), "quantized: o far from quantize_x's");
assert.equal(layerCheckNumbers({ worstRelative: 1e-6, ok: true }), "");
// T225: the worker's line of the layer's stages comes last, as it is
assert.equal(layerCheckNumbers({ stages: "stages: q 2.1e-7, K and V 396 to the nearest float16" }), "stages: q 2.1e-7, K and V 396 to the nearest float16");
assert.equal(layerCheckNumbers({ quantized, stages: "stages: q 2.1e-7; first to depart: none" }), "quantized: scales 2.5e-7, 6 of 8416 off by 1; stages: q 2.1e-7; first to depart: none");
// and the line of the tokens' steps, in the verdict of the tokens (after the first problem, where there is one)
assert.equal(checkVerdict(["tokens on the GPU", { ok: true, worstRelative: 0, tokens: 12, edge: 0, steps: "steps: T 0: logits within 2.0e-6 of the largest" }]),
  "tokens on the GPU ok (12 tokens, 0 next to a border; steps: T 0: logits within 2.0e-6 of the largest)");
assert.equal(checkVerdict(["tokens on the GPU", { ok: false, worstRelative: 0, tokens: 9, edge: 0, problems: ["T 0.7, token 2: 48, the CPU 483"], steps: "steps: T 0.7: logits 1.0e-3 | 2.0e-3" }]),
  "tokens on the GPU WRONG (9 tokens, 0 next to a border: T 0.7, token 2: 48, the CPU 483; steps: T 0.7: logits 1.0e-3 \\| 2.0e-3)");
export { real };
