// The section "GPU": the steps asked of src/runtime/benchmark/gpu.ts one at a time, and the Markdown of what came back.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
// (T643.11: the section's Markdown and its state are src/bench/gpusection.ts's gpuResult(), from the steps kept here)
import { BRIDGE_ROUNDS, gpuKept, gpuNone, gpuPlan, gpuResult } from "../bench.ts";
import type { Bridge, GpuFound, GpuStepKey, Step } from "../bench.ts";
import { $, results } from "./dom.ts";
import type { Result } from "../bench.ts";
import { staged, ask, worker } from "./section.ts";
import type { Asking } from "./section.ts";
import type { GpuModel, ToBenchGpu, ToWaiter, Waited } from "../runtime/contracts/benchmark.ts";

// T643.12: the steps are src/bench/'s gpuPlan(), counted before the first is asked, and what each answers is kept by
// the step's key (gpuKept()) under the name the page shows as it runs
export async function gpu(): Promise<Result> {
  const w = worker("benchmark/gpu.js");
  const found: GpuFound = { bandwidth: [], token: [] };
  let lost: string | undefined;
  // T177: the steps are this page's, counted once the adapter says whether it has the packed int8 dot
  const models: GpuModel[] = ["llm-jp-3 150M", "Llama 3.2 1B", ...(($("large") as HTMLInputElement).checked ? ["Llama 3.2 3B"] satisfies GpuModel[] : [])];
  const step = async <Message extends ToBenchGpu & { step: GpuStepKey }>(name: string, message: Message, at?: number, of?: number) => {
    staged("gpu", { stage: name, at, of });
    const answer = await ask(w, message);
    const kept: Step<NonNullable<typeof answer.result>> = { name, ...(answer.error ? { error: answer.error } : { result: answer.result }) };
    gpuKept(found, message.step, kept);
    return kept;
  };
  w.addEventListener("message", ({ data }) => { if (data.lost) lost = data.lost; });
  try {
    const adapter = await step("the adapter", { step: "info" });
    const none = gpuNone(adapter);
    if (none) return none;
    // (gpuNone() answers undefined only where the adapter's step gave a worker and an adapter)
    const plan = gpuPlan(adapter.result!, models);
    // the adapter, the plan's steps and the bridge
    const of = plan.length + 2;
    for (const [i, { name, message }] of plan.entries()) await step(name, message, i + 2, of);
    staged("gpu", { stage: "the bridge", at: of, of });
    const bridge = await gpuBridge(w);
    return gpuResult(found, bridge, lost, results.cpu);
  } finally {
    w.terminate();
  }
}
async function gpuBridge(w: Asking<"benchmark/gpu.js">): Promise<Bridge> {
  if (!self.crossOriginIsolated) return { error: "not measured: the page is not cross-origin isolated" };
  const memory = new SharedArrayBuffer(8);
  const waiter = worker("benchmark/waiter.js");
  const waited = new Promise<Waited>((resolve) => { waiter.onmessage = ({ data }: MessageEvent<Waited>) => resolve(data); });
  const answered = ask(w, { step: "bridge", memory, rounds: BRIDGE_ROUNDS });
  waiter.postMessage({ memory, rounds: BRIDGE_ROUNDS } satisfies ToWaiter);
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
