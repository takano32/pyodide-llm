// The section "GPU": its Markdown from the steps the page asked of its worker, and its state (T643.11: from the
// page's part, src/benchmark/gpu.ts, which asks the steps and keeps them).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)
import { number as fixed, yes } from "./cells.ts";
import { cpuBaseline } from "./cpu.ts";
import { BANDWIDTHS, bandwidthName, checkVerdict, matVecTable, tokenTable, layerTable } from "./gpu.ts";
import { matrixOf } from "../runtime/benchmark/gpu/models.ts";
import { layerStepsTable } from "./layersteps.ts";
import { generateTable } from "./generate.ts";
import { overheadMarkdown, ceilingsMarkdown, promptMarkdown } from "./prompt.ts";
import { REASONS } from "../runtime/forward/reasons.ts";
import type { Step } from "./cells.ts";
import type { Result } from "./sections.ts";
import type { Bandwidth, BenchGpuResults, GpuInfo, GpuModel, ToBenchGpu, TokenFound, Verdicts } from "../runtime/contracts/benchmark.ts";

// the round trips the bridge is timed over (the page's gpuBridge() asks them)
export const BRIDGE_ROUNDS = 2000;
// what the bridge found: its µs a round trip, or why there is none
export type Bridge = { error?: string; microseconds?: number };
// T643.12: the steps the section keeps, by the step's key (contracts/benchmark.ts's BenchGpuResults), each with the name
// the page shows as it runs. The steps asked once are kept in their place; bandwidth (once a shape) and token (once a
// model and a way) in lists, in the order asked. The bridge is the section's own (Bridge).
type Once = Exclude<keyof BenchGpuResults, "bandwidth" | "token" | "bridge">;
export type GpuFound = { [Key in Once]?: Step<BenchGpuResults[Key]> } & { bandwidth: Step<Bandwidth>[]; token: Step<TokenFound>[] };
// a step's key (the bridge is asked on its own, by the page's gpuBridge())
export type GpuStepKey = Exclude<keyof BenchGpuResults, "bridge">;
// a step as the plan asks it: the name the page shows, and the message
export type GpuPlanned = { name: string; message: Exclude<ToBenchGpu, { step: "info" | "bridge" }> };

// T135's candidate for a token (T134): the next token chosen on the GPU (the fewer dispatches are T150's layer)
const GPU_WAYS: [string, { sample: boolean }][] = [["chosen on the GPU", { sample: true }]];
const PROMPT_COUNTS = [1, 16, 64];

/** T643.12: the steps the section asks after the adapter's, in order, from what the adapter said (packed: the packed
 * int8 dot; fallback: a fallback adapter) and the models asked for. The page counts them before it asks the first
 * (T192: counted by hand, the last two read "N / N"). T182: a fallback adapter (SwiftShader in CI: the CPU in a GPU's
 * place) checks the shaders against JavaScript and times nothing: its times are no GPU's, and they took about 4
 * minutes a browser (the owner, 2026-09-27: what CI needs to know is that the GPU's path works). The timings are for a
 * real GPU, on the owner's devices. */
export function gpuPlan({ packed, fallback }: Pick<GpuInfo, "packed" | "fallback">, models: GpuModel[]): GpuPlanned[] {
  const check: GpuPlanned = { name: "the shaders against JavaScript", message: { step: "check" } };
  if (fallback) return [check];
  return [check,
    // the GPU section of T94's stage 0, step by step as /gpu-test/ ran it
    ...BANDWIDTHS.map(([model, matrix]): GpuPlanned => ({ name: bandwidthName(model, matrix), message: { step: "bandwidth", shape: matrixOf(model, matrix) } })),
    ...models.flatMap((model): GpuPlanned[] => [{ name: `a token of ${model}`, message: { step: "token", model, kind: "widen" } },
      ...(packed ? [{ name: `a token of ${model}, packed int8`, message: { step: "token", model, kind: "packed" } } satisfies GpuPlanned] : [])]),
    // T135's candidate on Llama 3.2 1B: the token chosen on the GPU
    ...GPU_WAYS.map(([label, how]): GpuPlanned => ({ name: `a token of Llama 3.2 1B, ${label}`, message: { step: "token", model: "Llama 3.2 1B", kind: "widen", ...how } })),
    // T150: a layer of a token as separate steps and fused into fewer dispatches
    { name: "a layer of a token", message: { step: "layer" } },
    // T202: where a layer's time goes, step by step
    { name: "the steps of a layer", message: { step: "layer steps" } },
    // T151: tokens generated on the GPU, each read back as it comes against several a submission read back once
    { name: "tokens generated on the GPU", message: { step: "generate" } },
    { name: "what a token costs besides the weights", message: { step: "overhead" } },
    // T168: the ceilings just before the prompt, so that both are measured as warm as each other
    { name: "the device's ceilings", message: { step: "ceilings" } },
    { name: "a prompt all at once", message: { step: "prompt", counts: PROMPT_COUNTS } }];
}

/** found with a step kept under its key: bandwidth and token at the end of their lists, the others in their place */
export function gpuKept(found: GpuFound, key: GpuStepKey, step: Step<unknown>): void {
  if (key === "bandwidth" || key === "token") found[key].push(step as Step<never>);
  else found[key] = step as Step<never>;
}

/** Why the GPU section does not run here, from the adapter's step (its result, or what failed); undefined where it runs. */
export function gpuNone(adapter: Step<GpuInfo>): Result<"gpu"> | undefined {
  const info = adapter.result;
  if (info?.worker && info.adapter) return undefined;
  const why = adapter.error ?? (info?.worker ? "navigator.gpu gave no adapter" : REASONS.noWebGpu);
  return { status: "none", markdown: why };
}

/** The GPU section from its steps: WRONG where the check found a shader wrong, failed where the shaders could not be
 * checked (or a tiled one could not run, T146) or the device was lost (a lost device answers every later wait at
 * once, and the times after it are no GPU's: T134's review). cpu: the CPU section's result, which the tables hold the
 * GPU against (T157). */
export function gpuResult(found: GpuFound, bridge: Bridge, lost: string | undefined, cpu: Result<"cpu"> | undefined): Result<"gpu"> {
  const checked = found.check, check = checked?.result;
  const wrong = check && Object.values(check).some((v) => !v.ok && !v.error);
  const unchecked = checked?.error ?? (check && Object.values(check).find((v) => v.error))?.error;
  return { status: wrong ? "wrong" : unchecked || lost ? "error" : "ok", data: { found, bridge, lost }, ...gpuMarkdown(found, bridge, lost, cpu) };
}

/** The section's Markdown, and of its lines those that say what went wrong (T227: a lost device, shaders not checked,
 * the verdicts that are not ok). cpu: the CPU section's result (T157: the CPU beside the GPU is the CPU section's
 * forward pass, written again when the CPU section runs after this one). */
export function gpuMarkdown(found: GpuFound, bridge: Bridge, lost: string | undefined, cpu: Result<"cpu"> | undefined) {
  const lines: string[] = [], said: string[] = [];
  const r: GpuInfo = found.info?.result ?? { worker: false };
  lines.push(`**Adapter**: ${r.adapter}${r.fallback ? " (a fallback adapter: the CPU in a GPU's place, each measured once, its speed no GPU's)" : ""}; max binding ${fixed(r.maxStorageBufferBindingSize! / 2 ** 20, 0)} MiB, ` +
    `max buffer ${fixed(r.maxBufferSize! / 2 ** 20, 0)} MiB, workgroup memory ${fixed(r.maxComputeWorkgroupStorageSize! / 1024, 0)} KiB, ${r.maxComputeInvocationsPerWorkgroup} threads a workgroup, ` +
    `packed int8 dot ${yes(r.packed)}, shader-f16 ${yes(r.features?.includes("shader-f16"))}, subgroups ${yes(r.features?.includes("subgroups"))}${r.subgroupSizes ? ` (${r.subgroupSizes[0] === r.subgroupSizes[1] ? r.subgroupSizes[0] : r.subgroupSizes.join(" to ")} wide)` : ""}`, "");
  if (lost) said.push(`**The device was lost** (${lost}): the times measured after it are not the GPU's.`);
  const check: Step<Verdicts> | undefined = found.check;
  if (check?.error) said.push(`**Shaders against JavaScript**: not checked (${check.error})`);
  lines.push(...said.flatMap((line) => [line, ""]));
  // every verdict of the check in one line (checkVerdict()); those not ok are the section's own warnings
  const verdicts = Object.entries(check?.result ?? {});
  if (check?.result) lines.push(`**Shaders against JavaScript**: ${verdicts.map(checkVerdict).join(", ")}`, "");
  said.push(...verdicts.filter(([, v]) => v.error || !v.ok).map(checkVerdict));
  const ceilings = found.ceilings?.result;
  lines.push(...matVecTable(found.bandwidth, check?.result, ceilings, { lost }), "");
  const baseline = cpuBaseline(cpu);
  const gpu = { fallback: r.fallback, lost, check: check?.result };
  lines.push(...tokenTable(found.token, baseline, gpu));
  lines.push("", ...layerTable(found.layer, check?.result, ceilings, gpu));
  lines.push("", ...layerStepsTable(found["layer steps"], check?.result, ceilings, gpu));
  lines.push("", ...generateTable(found.generate, check?.result, gpu));
  lines.push("", ...overheadMarkdown(found.overhead),
    "", ...ceilingsMarkdown(found.ceilings),
    "", ...promptMarkdown(found.prompt, check?.result, ceilings, baseline, gpu, cpu));
  lines.push("", `**Bridge** (Atomics.wait ↔ Atomics.waitAsync, ${BRIDGE_ROUNDS} round trips): ${bridge.error ?? `${fixed(bridge.microseconds)} µs each`}`);
  return { markdown: lines.join("\n"), said };
}
