// gpu/activations.ts (T344 review): a GPU's table of a gate by its activation (gpu/block.ts's GATE_SHADERS, gpu/tokens.ts's GLUS,
// gpu/reference.ts's GATES) read for a plan's activation. The activations the GPU's shaders have are forward/alone.ts's
// GATED, one list: each table is a Record of them, so tsc says where one lacks a row; an activation outside it
// (forward/gpufit.ts takes none such to the GPU) is refused here rather than taken as SwiGLU's
// (a module of the model's GPU worker, gpu.ts, which a bundler links it into)
import type { Gated } from "../forward/alone.ts";
import type { Activation } from "../contracts/plan.ts";

export function gatedBy<T>(table: Record<Gated, T>, activation: Activation): T {
  if (!Object.hasOwn(table, activation)) throw new Error(`The GPU has no gated FFN whose activation is ${JSON.stringify(activation)}.`);
  return table[activation as Gated];
}
