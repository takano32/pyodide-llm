// The section "GPU": its Markdown from the steps the page asked of its worker, and its state (T643.11: from the
// page's part, src/benchmark/gpu.ts, which asks the steps and keeps them).
// (T353: a part of src/bench.ts, which is the window that exports every name of these)
import { number as fixed, yes } from "./cells.ts";
import { cpuBaseline } from "./cpu.ts";
import { checkVerdict, matVecTable, tokenTable, layerTable } from "./gpu.ts";
import { layerStepsTable } from "./layersteps.ts";
import { generateTable } from "./generate.ts";
import { overheadMarkdown, ceilingsMarkdown, promptMarkdown } from "./prompt.ts";
import { REASONS } from "../runtime/forward/reasons.ts";
import type { Step } from "./cells.ts";
import type { Result } from "./sections.ts";
import type { GpuCeilings, GpuInfo, Verdicts } from "../runtime/contracts/benchmark.ts";

// the round trips the bridge is timed over (the page's gpuBridge() asks them)
export const BRIDGE_ROUNDS = 2000;
// what the bridge found: its µs a round trip, or why there is none
export type Bridge = { error?: string; microseconds?: number };

/** Why the GPU section does not run here, from the adapter's step (steps: the steps so far, the adapter's first);
 * undefined where it runs. */
export function gpuNone(info: GpuInfo | undefined, steps: Step<unknown>[]): Result | undefined {
  if (info?.worker && info.adapter) return undefined;
  const why = steps[0].error ?? (info?.worker ? "navigator.gpu gave no adapter" : REASONS.noWebGpu);
  return { status: "none", markdown: why };
}

/** The GPU section from its steps: WRONG where the check found a shader wrong, failed where the shaders could not be
 * checked (or a tiled one could not run, T146) or the device was lost (a lost device answers every later wait at
 * once, and the times after it are no GPU's: T134's review). cpu: the CPU section's result, which the tables hold the
 * GPU against (T157). */
export function gpuResult(steps: any[], bridge: Bridge, lost: string | undefined, cpu: Result | undefined): Result {
  const checked = steps.find((s) => s.name === "the shaders against JavaScript"), check: Verdicts | undefined = checked?.result;
  const wrong = check && Object.values(check).some((v) => !v.ok && !v.error);
  const unchecked = checked?.error ?? (check && Object.values(check).find((v) => v.error))?.error;
  return { status: wrong ? "wrong" : unchecked || lost ? "error" : "ok", data: { steps, bridge, lost }, ...gpuMarkdown(steps, bridge, lost, cpu) };
}

/** The section's Markdown, and of its lines those that say what went wrong (T227: a lost device, shaders not checked,
 * the verdicts that are not ok). cpu: the CPU section's result (T157: the CPU beside the GPU is the CPU section's
 * forward pass, written again when the CPU section runs after this one). */
export function gpuMarkdown(steps: any[], bridge: Bridge, lost: string | undefined, cpu: Result | undefined) {
  const lines: string[] = [], said: string[] = [];
  const r: GpuInfo = steps.find((s) => s.name === "the adapter").result;
  lines.push(`**Adapter**: ${r.adapter}${r.fallback ? " (a fallback adapter: the CPU in a GPU's place, each measured once, its speed no GPU's)" : ""}; max binding ${fixed(r.maxStorageBufferBindingSize! / 2 ** 20, 0)} MiB, ` +
    `max buffer ${fixed(r.maxBufferSize! / 2 ** 20, 0)} MiB, workgroup memory ${fixed(r.maxComputeWorkgroupStorageSize! / 1024, 0)} KiB, ${r.maxComputeInvocationsPerWorkgroup} threads a workgroup, ` +
    `packed int8 dot ${yes(r.packed)}, shader-f16 ${yes(r.features?.includes("shader-f16"))}, subgroups ${yes(r.features?.includes("subgroups"))}${r.subgroupSizes ? ` (${r.subgroupSizes[0] === r.subgroupSizes[1] ? r.subgroupSizes[0] : r.subgroupSizes.join(" to ")} wide)` : ""}`, "");
  if (lost) said.push(`**The device was lost** (${lost}): the times measured after it are not the GPU's.`);
  const check: Step<Verdicts> | undefined = steps.find((s) => s.name === "the shaders against JavaScript");
  if (check?.error) said.push(`**Shaders against JavaScript**: not checked (${check.error})`);
  lines.push(...said.flatMap((line) => [line, ""]));
  // every verdict of the check in one line (checkVerdict()); those not ok are the section's own warnings
  const verdicts = Object.entries(check?.result ?? {});
  if (check?.result) lines.push(`**Shaders against JavaScript**: ${verdicts.map(checkVerdict).join(", ")}`, "");
  said.push(...verdicts.filter(([, v]) => v.error || !v.ok).map(checkVerdict));
  const bandwidths = steps.filter((s) => s.name.startsWith("bandwidth"));
  const ceilings: GpuCeilings | undefined = steps.find((s) => s.name === "the device's ceilings")?.result;
  lines.push(...matVecTable(bandwidths, check?.result, ceilings, { lost }), "");
  const baseline = cpuBaseline(cpu);
  const gpu = { fallback: r.fallback, lost, check: check?.result };
  lines.push(...tokenTable(steps.filter((x) => x.name.startsWith("a token")), baseline, gpu));
  lines.push("", ...layerTable(steps.find((s) => s.name === "a layer of a token"), check?.result, ceilings, gpu));
  lines.push("", ...layerStepsTable(steps.find((s) => s.name === "the steps of a layer"), check?.result, ceilings, gpu));
  lines.push("", ...generateTable(steps.find((s) => s.name === "tokens generated on the GPU"), check?.result, gpu));
  lines.push("", ...overheadMarkdown(steps.find((s) => s.name === "what a token costs besides the weights")),
    "", ...ceilingsMarkdown(steps.find((s) => s.name === "the device's ceilings")),
    "", ...promptMarkdown(steps.find((s) => s.name === "a prompt all at once"), check?.result, ceilings, baseline, gpu, cpu));
  lines.push("", `**Bridge** (Atomics.wait ↔ Atomics.waitAsync, ${BRIDGE_ROUNDS} round trips): ${bridge.error ?? `${fixed(bridge.microseconds)} µs each`}`);
  return { markdown: lines.join("\n"), said };
}
