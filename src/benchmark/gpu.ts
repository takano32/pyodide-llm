// The section "GPU": the steps asked of public/benchmark/gpu.js one at a time, and the Markdown of what came back.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import { checkVerdict, matVecTable, cpuBaseline, tokenTable, layerTable, layerStepsTable, generateTable, unmeasured,
  tableCell, noRatios, threadsOf, timesFaster } from "../bench.js";
import { $, type Result, results } from "./dom.ts";
import { staged, ask, worker, fixed, yes } from "./section.ts";

// the GPU section of T94's stage 0, step by step as /gpu-test/ ran it
const SHAPES: Record<string, number[]> = { "llm-jp-3 150M w1": [2048, 512], "Llama 3.2 1B w1": [8192, 2048], "Llama 3.2 1B classifier": [128256, 2048] };
const BRIDGE_ROUNDS = 2000;
// T135's candidate for a token (T134): the next token chosen on the GPU (the fewer dispatches are T150's layer)
const GPU_WAYS: [string, any][] = [["chosen on the GPU", { sample: true }]];
const PROMPT_COUNTS = [1, 16, 64];
export async function gpu(): Promise<Result> {
  const w = worker("benchmark/gpu.js");
  const steps: any[] = [];
  let lost: string | undefined;
  // T177: the steps are this page's, counted once the adapter says whether it has the packed int8 dot
  const models = ["llm-jp-3 150M", "Llama 3.2 1B", ...(($("large") as HTMLInputElement).checked ? ["Llama 3.2 3B"] : [])];
  let of: number | undefined;
  const step = async (name: string, message: any) => {
    staged("gpu", { stage: name, at: of && steps.length + 1, of });
    const answer = await ask(w, message);
    steps.push({ name, ...(answer.error ? { error: answer.error } : { result: answer.result }) });
    return answer.result;
  };
  w.addEventListener("message", ({ data }) => { if (data.lost) lost = data.lost; });
  try {
    const info = await step("the adapter", { step: "info" });
    if (!info?.worker || !info.adapter) {
      const why = steps[0].error ?? (info?.worker ? "navigator.gpu gave no adapter" : "no WebGPU in a worker here");
      return { status: "none", markdown: why };
    }
    // the adapter, the check, the shapes, the tokens, the ways, a layer, its steps (T202), the tokens generated, besides the weights,
    // the ceilings, a prompt, the bridge (T192: the tokens generated were left out, and the last two read "N / N")
    of = 2 + Object.keys(SHAPES).length + models.length * (info.packed ? 2 : 1) + GPU_WAYS.length + 7;
    // T182: a fallback adapter (SwiftShader in CI: the CPU in a GPU's place) checks the shaders against JavaScript and
    // times nothing: its times are no GPU's, and they took about 4 minutes a browser (the owner, 2026-09-27: what CI
    // needs to know is that the GPU's path works). The timings are for a real GPU, on the owner's devices.
    if (info.fallback) of = 3; // the adapter, the check, the bridge
    const check = await step("the shaders against JavaScript", { step: "check" });
    if (!info.fallback) {
    for (const [name, shape] of Object.entries(SHAPES)) await step(`bandwidth: ${name}`, { step: "bandwidth", shape });
    for (const name of models) {
      await step(`a token of ${name}`, { step: "token", model: name, kind: "widen" });
      if (info.packed) await step(`a token of ${name}, packed int8`, { step: "token", model: name, kind: "packed" });
    }
    // T135's candidate on Llama 3.2 1B: the token chosen on the GPU
    for (const [label, how] of GPU_WAYS) await step(`a token of Llama 3.2 1B, ${label}`, { step: "token", model: "Llama 3.2 1B", kind: "widen", ...how });
    // T150: a layer of a token as separate steps and fused into fewer dispatches
    await step("a layer of a token", { step: "layer" });
    // T202: where a layer's time goes, step by step
    await step("the steps of a layer", { step: "layer steps" });
    // T151: tokens generated on the GPU, each read back as it comes against several a submission read back once
    await step("tokens generated on the GPU", { step: "generate" });
    await step("what a token costs besides the weights", { step: "overhead" });
    // T168: the ceilings just before the prompt, so that both are measured as warm as each other
    await step("the device's ceilings", { step: "ceilings" });
    await step("a prompt all at once", { step: "prompt", counts: PROMPT_COUNTS });
    }
    staged("gpu", { stage: "the bridge", at: of, of });
    const bridge = await gpuBridge(w);
    const wrong = check && Object.values(check).some((v: any) => !v.ok && !v.error);
    // not ok either when the shaders could not be checked (or a tiled one could not run, T146), or when the device
    // was lost: a lost device answers every later wait at once, and the times after it are no GPU's (T134's review)
    const unchecked = steps.find((s) => s.name === "the shaders against JavaScript")?.error
      ?? (check && Object.values(check).find((v: any) => v.error) as any)?.error;
    return { status: wrong ? "wrong" : unchecked || lost ? "error" : "ok", data: { steps, bridge, lost }, ...gpuMarkdown(steps, bridge, lost) };
  } finally {
    w.terminate();
  }
}
async function gpuBridge(w: Worker) {
  if (!self.crossOriginIsolated) return { error: "not measured: the page is not cross-origin isolated" };
  const memory = new SharedArrayBuffer(8);
  const waiter = worker("benchmark/waiter.js");
  const waited = new Promise<any>((resolve) => { waiter.onmessage = ({ data }) => resolve(data); });
  const answered = ask(w, { step: "bridge", memory, rounds: BRIDGE_ROUNDS });
  waiter.postMessage({ memory, rounds: BRIDGE_ROUNDS });
  try {
    // without Atomics.waitAsync nothing answers the waiter, which would wait for ever (T134's review)
    const answer = await answered;
    // (T227's review: a worker's error says "failed", as every step's does, so that the warnings list it; no waitAsync is no failure)
    if (!answer.result?.waitAsync) return { error: answer.error ? `failed: ${answer.error}` : "no Atomics.waitAsync here" };
    return { microseconds: (await waited).microseconds };
  } finally {
    waiter.terminate();
  }
}
// the section's Markdown, and of its lines those that say what went wrong (T227: a lost device, shaders not checked,
// the verdicts that are not ok)
export function gpuMarkdown(steps: any[], bridge: any, lost?: string) {
  const lines: string[] = [], said: string[] = [];
  const r = steps.find((s) => s.name === "the adapter").result;
  lines.push(`**Adapter**: ${r.adapter}${r.fallback ? " (a fallback adapter: the CPU in a GPU's place, each measured once, its speed no GPU's)" : ""}; max binding ${fixed(r.maxStorageBufferBindingSize / 2 ** 20, 0)} MiB, ` +
    `max buffer ${fixed(r.maxBufferSize / 2 ** 20, 0)} MiB, workgroup memory ${fixed(r.maxComputeWorkgroupStorageSize / 1024, 0)} KiB, ${r.maxComputeInvocationsPerWorkgroup} threads a workgroup, ` +
    `packed int8 dot ${yes(r.packed)}, shader-f16 ${yes(r.features?.includes("shader-f16"))}, subgroups ${yes(r.features?.includes("subgroups"))}${r.subgroupSizes ? ` (${r.subgroupSizes[0] === r.subgroupSizes[1] ? r.subgroupSizes[0] : r.subgroupSizes.join(" to ")} wide)` : ""}`, "");
  if (lost) said.push(`**The device was lost** (${lost}): the times measured after it are not the GPU's.`);
  const check = steps.find((s) => s.name === "the shaders against JavaScript");
  if (check?.error) said.push(`**Shaders against JavaScript**: not checked (${check.error})`);
  lines.push(...said.flatMap((line) => [line, ""]));
  // every verdict of the check in one line (src/bench.js's checkVerdict()); those not ok are the section's own warnings
  const verdicts = Object.entries(check?.result ?? {});
  if (check?.result) lines.push(`**Shaders against JavaScript**: ${verdicts.map(checkVerdict).join(", ")}`, "");
  said.push(...verdicts.filter(([, v]: any) => v.error || !v.ok).map(checkVerdict));
  const bandwidths = steps.filter((s) => s.name.startsWith("bandwidth"));
  const ceilings = steps.find((s) => s.name === "the device's ceilings")?.result;
  lines.push(...matVecTable(bandwidths, check?.result, ceilings, { lost }), "");
  // T157: the CPU beside the GPU is the CPU section's forward pass, rewritten when the CPU section runs after this one
  const baseline = cpuBaseline(results.cpu);
  const gpu = { fallback: r.fallback, lost, check: check?.result };
  lines.push(...tokenTable(steps.filter((x) => x.name.startsWith("a token")), baseline, gpu));
  lines.push("", ...layerTable(steps.find((s) => s.name === "a layer of a token"), check?.result, ceilings, gpu));
  lines.push("", ...layerStepsTable(steps.find((s) => s.name === "the steps of a layer"), check?.result, ceilings, gpu));
  lines.push("", ...generateTable(steps.find((s) => s.name === "tokens generated on the GPU"), check?.result, gpu));
  lines.push("", ...overheadMarkdown(steps.find((s) => s.name === "what a token costs besides the weights")),
    "", ...ceilingsMarkdown(steps.find((s) => s.name === "the device's ceilings")),
    "", ...promptMarkdown(steps.find((s) => s.name === "a prompt all at once"), check?.result, ceilings, baseline, gpu));
  lines.push("", `**Bridge** (Atomics.wait ↔ Atomics.waitAsync, ${BRIDGE_ROUNDS} round trips): ${bridge.error ?? `${fixed(bridge.microseconds)} µs each`}`);
  return { markdown: lines.join("\n"), said };
}

// what a token costs on the GPU besides reading its weights (T135: the phones' 18 to 20 ms, T94)
function overheadMarkdown(s: any) {
  if (!s || s.error) return [`**Besides the weights**: ${unmeasured(s?.error)}`];
  const o = s.result;
  return ["| besides the weights | GPU ms |", "|---|---:|",
    `| ${o.dispatches} dispatches that do nothing, one submission | ${fixed(o.emptyDispatches, 2)} |`,
    `| one submission, not waited for | ${fixed(o.submitOnly, 3)} |`,
    `| one submission, waited for | ${fixed(o.submitAndWait, 2)} |`,
    `| the token's id read back (4 bytes) | ${fixed(o.readToken, 2)} |`,
    `| every logit read back (${o.vocab} floats) | ${fixed(o.readLogits, 2)} |`];
}
// T168: the device's ceilings, each a loop of that alone (public/shaders.js, the method of clpeak): what the
// prompt's shaders are held against
const percent = (part: number, whole: number) => `${fixed((100 * part) / whole, 1)}%`;
function ceilingsMarkdown(s: any) {
  if (!s || s.error) return [`**The device's ceilings**: ${unmeasured(s?.error)}`];
  const c = s.result;
  // three digits where it is small (a fallback adapter's reads of the workgroup's memory are well under 1 GB/s); an
  // unsteady one (its 2n not about twice its n) says so, and the prompt is not held against it
  const cell = (v: any, key: string, unit: string) => (v.none ?? (v.error ? tableCell(`failed: ${v.error}`)
    : `${v.unsteady ? "unsteady: " : ""}${v[key] < 100 ? v[key].toPrecision(3) : fixed(v[key], 0)} ${unit}${v.shape ? ` (${v.shape})` : ""}`));
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
function ceilingOf(row: any, c: any) {
  if (!c || c.fallback) return "";
  const steady = (v: any, key: string) => (v && !v.unsteady ? v[key] : undefined);
  const [name, compute] = row.packed ? ["int8 dots", steady(c.dot4, "GOPS")] : ["f32", steady(c.f32, "GFLOPS")];
  const reads = steady(c.global, "GBps"), bound = reads ? (reads * 2 * row.tokens) / 1.125 : undefined;
  const [what, v] = bound && (!compute || bound < compute) ? ["reading", bound] : [name, compute];
  if (!v) return "";
  return `${percent(row.GFLOPS, v)} of ${what}${row.GFLOPS > v ? " (past it: the ceiling read low)" : ""}`;
}
// a prompt's tokens at once on the GPU against the CPU section's blocks of 16 (the same made-up model: two layers of
// Llama 3.2 1B's width, no classifier), by every shader of it (T146: the tiled ones beside T135's batched one), with
// the GFLOPS of each (a multiply and an add for each weight and token) and the CPU's GOPS from the same count; and
// how many times faster each is than the CPU's fastest count of threads (T157)
function promptMarkdown(s: any, check?: any, ceilings?: any, baseline?: any, gpu?: any) {
  if (!s || s.error) return [`**A prompt all at once**: ${unmeasured(s?.error)}`];
  const r = s.result, cpuPrompt = baseline?.prompt, none = noRatios(gpu);
  const cpuRows = (results.cpu?.data?.rows ?? []).filter((row: any) => row.promptMsPerToken);
  const cpu = cpuRows.length ? cpuRows.map((row: any) => `${fixed(row.promptMsPerToken, 2)} ms with ${row.threads} (${fixed((2 * r.weights) / (row.promptMsPerToken / 1000) / 1e9, 0)} GOPS)`).join(", ")
    : "run the CPU section for it";
  // a shader the check found WRONG (or could not check) says so on its rows, and is never the fastest (T146's review)
  const verdict = (row: any) => check?.[row.shader.replace(/, again at the end$/, "")];
  const wrong = (row: any) => { const v = verdict(row); return v !== undefined && !v.ok; };
  const fastest = (tokens: number) => {
    const best = r.rows.filter((row: any) => row.tokens === tokens && !row.again && !wrong(row)).sort((a: any, b: any) => a.msPerToken - b.msPerToken)[0];
    return best ? `at ${tokens} tokens ${best.shader}, ${fixed(best.msPerToken, 2)} ms a token (${fixed(best.GFLOPS, 0)} GFLOPS)` : "";
  };
  const name = (row: any) => (wrong(row) ? `${row.shader} (WRONG in the check)` : row.shader);
  return [`**A prompt** (the CPU section's model: two layers of Llama 3.2 1B's width, ${fixed(r.GB, 2)} GB, ${fixed(r.weights / 1e6, 0)} million weights, no classifier); ` +
    `the CPU, 16 tokens at once: ${cpu}. A tiled shader's name says whose form it takes (llama.cpp's, TensorFlow.js's, ONNX Runtime's) and the rows × tokens of a workgroup's tile. ` +
    "The batched shader is measured again at the end: a device that slows down as it warms up shows it there. " +
    "Of the ceiling: its GFLOPS against the lower of the device's ceiling above for what the shader multiplies with (f32, or the int8 dots) and the buffer's reads at that many tokens. " +
    (!cpuPrompt ? `GPU ÷ CPU: not measured (${baseline?.why ?? "the CPU section measured no prompt"})`
      : none ? `GPU ÷ CPU: ${none}`
      : `GPU ÷ CPU: against the CPU's ${fixed(cpuPrompt.msPerToken, 2)} ms a token with ${threadsOf(cpuPrompt)}, 16 at once (its fastest). ` +
        "The GPU rows are the matrices only, the CPU's the whole forward pass. Above 1× the GPU is faster"), "",
    "| shader | tokens at once on the GPU | GPU ms | GPU ms a token | GFLOPS | of the ceiling | GPU ÷ CPU |", "|---|---:|---:|---:|---:|---:|---:|",
    ...r.rows.map((row: any) => (row.none || row.error ? `| ${name(row)} | ${tableCell(row.none ?? `failed: ${row.error}`)} | | | | | |`
      : `| ${name(row)} | ${row.tokens} | ${fixed(row.ms)} | ${fixed(row.msPerToken, 2)} | ${fixed(row.GFLOPS, 0)} | ${ceilingOf(row, ceilings)} | ${none || wrong(row) ? "" : timesFaster(cpuPrompt?.msPerToken, row.msPerToken)} |`)),
    "", `**Fastest on the GPU** (of the shaders the check found right): ${[fastest(16), fastest(64)].filter(Boolean).join("; ") || "nothing measured"}`];
}
