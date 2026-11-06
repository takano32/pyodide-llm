// The section "GPU": the steps asked of src/runtime/benchmark/gpu.ts one at a time, and the Markdown of what came back.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
// (T643.11: the section's Markdown and its state are src/bench/gpusection.ts's gpuResult(), from the steps kept here)
import { BRIDGE_ROUNDS, gpuNone, gpuResult } from "../bench.ts";
import type { Bridge } from "../bench.ts";
import { $, results } from "./dom.ts";
import type { Result } from "../bench.ts";
import { staged, ask, worker } from "./section.ts";
import type { Asking } from "./section.ts";
import type { GpuModel, ToBenchGpu, ToWaiter, Waited } from "../runtime/contracts/benchmark.ts";
// T368.7: a step as the section keeps it, by the name the page gives it: its result (contracts/benchmark.ts's, by the
// step asked) or what failed. The list of them is read by name, which no type follows: each reader says what it takes

// the GPU section of T94's stage 0, step by step as /gpu-test/ ran it
const SHAPES: Record<string, [rows: number, n: number]> = { "llm-jp-3 150M w1": [2048, 512], "Llama 3.2 1B w1": [8192, 2048], "Llama 3.2 1B classifier": [128256, 2048] };
// T135's candidate for a token (T134): the next token chosen on the GPU (the fewer dispatches are T150's layer)
const GPU_WAYS: [string, { sample: boolean }][] = [["chosen on the GPU", { sample: true }]];
const PROMPT_COUNTS = [1, 16, 64];
export async function gpu(): Promise<Result> {
  const w = worker("benchmark/gpu.js");
  const steps: any[] = [];
  let lost: string | undefined;
  // T177: the steps are this page's, counted once the adapter says whether it has the packed int8 dot
  const models: GpuModel[] = ["llm-jp-3 150M", "Llama 3.2 1B", ...(($("large") as HTMLInputElement).checked ? ["Llama 3.2 3B"] satisfies GpuModel[] : [])];
  let of: number | undefined;
  const step = async <Message extends ToBenchGpu>(name: string, message: Message) => {
    staged("gpu", { stage: name, at: of && steps.length + 1, of });
    const answer = await ask(w, message);
    steps.push({ name, ...(answer.error ? { error: answer.error } : { result: answer.result }) });
    return answer.result;
  };
  w.addEventListener("message", ({ data }) => { if (data.lost) lost = data.lost; });
  try {
    const info = await step("the adapter", { step: "info" });
    const none = gpuNone(info, steps);
    if (none) return none;
    // (gpuNone() answers undefined only where the adapter's step gave a worker and an adapter)
    const adapter = info!;
    // the adapter, the check, the shapes, the tokens, the ways, a layer, its steps (T202), the tokens generated, besides the weights,
    // the ceilings, a prompt, the bridge (T192: the tokens generated were left out, and the last two read "N / N")
    of = 2 + Object.keys(SHAPES).length + models.length * (adapter.packed ? 2 : 1) + GPU_WAYS.length + 7;
    // T182: a fallback adapter (SwiftShader in CI: the CPU in a GPU's place) checks the shaders against JavaScript and
    // times nothing: its times are no GPU's, and they took about 4 minutes a browser (the owner, 2026-09-27: what CI
    // needs to know is that the GPU's path works). The timings are for a real GPU, on the owner's devices.
    if (adapter.fallback) of = 3; // the adapter, the check, the bridge
    await step("the shaders against JavaScript", { step: "check" });
    if (!adapter.fallback) {
    for (const [name, shape] of Object.entries(SHAPES)) await step(`bandwidth: ${name}`, { step: "bandwidth", shape });
    for (const name of models) {
      await step(`a token of ${name}`, { step: "token", model: name, kind: "widen" });
      if (adapter.packed) await step(`a token of ${name}, packed int8`, { step: "token", model: name, kind: "packed" });
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
    return gpuResult(steps, bridge, lost, results.cpu);
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
