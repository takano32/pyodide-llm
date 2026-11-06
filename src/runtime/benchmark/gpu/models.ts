// The models the GPU section of /benchmark/ stands for, by their names: their shapes as the legacy header has them (dim,
// hidden, layers, heads, kv heads, vocab). T643.15: one table for both ends. The worker's steps read it through
// gpu/device.ts; the page's (src/bench/gpu.ts) asks the matrices of the bandwidth step by it and says what a token of
// one quantizes. It imports nothing, so that the page takes it without the worker.
import type { GpuModel } from "../../contracts/benchmark.ts";

// a model's shape, as the legacy header has it
export type ModelShape = { dim: number; hidden: number; layers: number; heads: number; kvHeads: number };
export const MODELS: Record<GpuModel, ModelShape & { vocab: number }> = {
  "llm-jp-3 150M": { dim: 512, hidden: 2048, layers: 12, heads: 8, kvHeads: 8, vocab: 99584 },
  "Llama 3.2 1B": { dim: 2048, hidden: 8192, layers: 16, heads: 32, kvHeads: 8, vocab: 128256 },
  "Llama 3.2 3B": { dim: 3072, hidden: 8192, layers: 28, heads: 24, kvHeads: 8, vocab: 128256 },
};

// a matrix of a model the bandwidth step times, by its name: w1 ([hidden, dim]) or the classifier ([vocab, dim])
export type ModelMatrix = "w1" | "classifier";
export const matrixOf = (model: GpuModel, matrix: ModelMatrix): [rows: number, n: number] => {
  const { dim, hidden, vocab } = MODELS[model];
  return [matrix === "w1" ? hidden : vocab, dim];
};
