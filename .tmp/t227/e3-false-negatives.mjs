// E3: failures the page writes in none of warnings()'s words, built from the real table functions
import { warnings, pathTable, cpuTable, roundsTable, tableCell, benchMarkdown, checkVerdict, gpuSummary } from "../../src/bench.js";
import * as F from "./fixtures.mjs";

const { real, aCpu, android, rows } = F;
const show = (label, sections) => {
  const out = warnings(sections);
  console.log(`${label}: ${out.length ? "" : "NOTHING LISTED"}`);
  out.forEach((line) => console.log(`    ${line.slice(0, 200)}`));
};
const cell = (speed, low, high, gpuTokens, unsteady = false) => ({ speed, low, high, gpuTokens, unsteady });
const stopped = (lost) => pathTable({ ...real, gpu: { ...real.gpu, lost }, rows: [{ ...real.rows[0], cpu: cell(612, 600, 640, 0), chosen: cell(640, 630, 650, 0), gpu: { skip: lost } }, real.rows[2]] }, "tiny-lm");
const model = (head) => [{ title: "Model", markdown: head }];

console.log("-- the model page's path: the GPU stopped while it was timed (forward.js's stopGpu() reasons)");
for (const why of ["the GPU failed on a block of the prompt", "the GPU did not take a block or a step", "the GPU said nothing for 10 s", "the GPU's worker stopped answering for 10 s",
  "the GPU computed logits that are not finite numbers (NaN or infinity) at position 3", "the GPU sampled 3 of 4 tokens", "the GPU sampled 130000, outside the vocabulary of 128256",
  "the GPU's worker did not start (an error)"]) show(`  lost: ${why}`, model([benchMarkdown(rows, android), stopped(why)].join("\n\n")));

console.log("-- the model page's path: the threads (worker.js's timedPaths how)");
const threads = (how, n = 4) => [benchMarkdown(rows, android), pathTable({ ...real, threads: n, how }, "tiny-lm")].join("\n\n");
for (const how of [{ alone: "a software thread stopped" }, { stopped: true, remembered: true }, { unfinished: 120 }, { alone: "not the 4 asked for: its software threads did not start" },
  { alone: "no shared memory here" }]) show(`  how ${JSON.stringify(how)}`, model(threads(how, how.alone ? 1 : 4)));

console.log("-- the CPU section: a count the browser did not start, the ceilings failing as a whole, one ceiling failing");
const none = { ...aCpu, rows: [...aCpu.rows, { asked: 16, threads: undefined, none: "the browser did not start that many software threads" }] };
show("  a count not started", [{ title: "CPU", status: "ok", markdown: cpuTable(none).join("\n") }]);
show("  ceilings error", [{ title: "CPU", status: "ok", markdown: cpuTable({ ...aCpu, ceilings: { error: "out of memory" } }).join("\n") }]);
show("  one ceiling error", [{ title: "CPU", status: "ok", markdown: cpuTable({ ...aCpu, ceilings: { ...aCpu.ceilings, dot: { error: "x" } } }).join("\n") }]);

console.log("-- sections that are not ok");
show("  GPU 'none' with a worker's error as the why (benchmark.astro gpu(): steps[0].error becomes status none)", [{ title: "GPU", status: "none", markdown: "requestDevice failed: out of memory" }]);
show("  GPU 'none' (plain absence)", [{ title: "GPU", status: "none", markdown: "no WebGPU in a worker here" }]);
show("  watchdog", [{ title: "GPU", status: "error", markdown: "stopped: nothing came from this section for 5 minutes" }]);
show("  memory, tab ended (status ok)", [{ title: "Page memory", status: "ok", markdown: "One WebAssembly memory grown 64 MiB at a time ...\n\n| held | how it ended |\n|---:|---|\n| 1792 MiB | the tab ended while it grew to 1856 MiB and the page was loaded again: the browser ended it, most likely for its memory |" }]);
show("  line slower", [{ title: "Line", status: "ok", markdown: "| huggingface.co read no faster than | got | within 10% |\n|---:|---:|---|\n| 8 MB/s | the line is slower | |" }]);
show("  bridge error (the page's raw line)", [{ title: "GPU", status: "ok", markdown: "**Bridge** (Atomics.wait ↔ Atomics.waitAsync, 2000 round trips): Atomics.wait cannot be called in this context" }]);
show("  bridge not isolated", [{ title: "GPU", status: "ok", markdown: "**Bridge** (Atomics.wait ↔ Atomics.waitAsync, 2000 round trips): not measured: the page is not cross-origin isolated" }]);
