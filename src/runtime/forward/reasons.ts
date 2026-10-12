// forward/reasons.ts (T380.1): why a model's prompts or answers are not on the GPU, or no longer: the one place each of
// these reasons is written. One cause, one sentence (or one function that fills in its numbers).
//
// A reason is a value, not an error: nothing is thrown, the model runs on the CPU, and the reason is what the status
// line says after "prompts on the CPU (", what the console says, and what /benchmark/ puts in a cell the GPU did not
// run. Whoever finds the cause names it here (REASONS.rotated); nobody writes its words a second time, and nobody
// tells two causes apart by their words (src/bench/path.js asks whether a reason IS one of these).
//
// Who reads this file: the GPU's side of the engine (gpuside.ts: what the model is, what stopped the GPU), alone.ts (a
// model on the GPU alone), paths.ts (/benchmark/'s cells), the GPU's worker for what it finds as it opens the device
// (gpu/device.ts), and the two pages' tables (src/bench/path.js, src/benchmark/gpu.ts). It imports nothing, and each
// bundle that reads it has its own copy. The model's worker does not read it (failure.ts says why a module of
// forward.ts's chunk is not one of the worker's entry too): it asks the engine (engine.gpuWhyNot), and has two
// reasons of its own for what it alone knows (worker/load.ts: the NumPy engine, the benchmark's page).
// What is not here: what a shader's check or the device says in its own words as it fails (gpu/*.ts: a sentence made
// once, of numbers, where it is thrown), which the GPU's worker hands over as the reason it is.
//
// The words are the owner's and the reviews': a text is changed only where its entry is, and the status line follows.
// Two causes are said by two sentences each, as they were before this table (the first is what a visitor may read,
// the second what gpuOnlyUnfit() answers, which only the choice between the places reads): a fallback adapter, and
// heads of a size that is no multiple of 4.
export const REASONS = {
  // ---- what the model is: its prompts stay on the CPU (gpuFit()'s unfit())
  linear: "linear-attention layers are not on the GPU yet",  // T229
  convolution: "convolution layers are not on the GPU yet",  // T260
  // (T375.2: a kind of layer that keeps a state which the engine comes to have: on the CPU until the GPU has it too)
  stateful: "layers that keep a state are not on the GPU yet",
  rotated: "a rotated basis is not on the GPU yet",  // T237
  // (T375.5, T484: the GPU's layers are of two pairs, RMSNorms with a gated FFN and LayerNorms with GPT-2's FFN)
  pair: "this pair of norms and FFN is not on the GPU yet",
  // (said wherever the memory is not a shared one: T371, ?threads=1 on an isolated page reads this too)
  notIsolated: "the page is not cross-origin isolated",
  heads: "heads of a size that is no multiple of 4 are not on the GPU",
  floats: "float32 weights are not on the GPU yet",
  layersRoom: (bytes: number) => `the layers on the GPU as well (${Math.round(bytes / 1e6)} MB) would not leave this device enough memory`,
  // ---- what the model is: its answers' steps stay on the CPU (gpuFit()'s tokensUnfit(), and a generation's stops)
  floatClassifier: "a classifier of float weights is not on the GPU's tokens",
  memoryUnsaid: "this browser does not say how much memory the device has",  // T205
  classifierRoom: (bytes: number) => `the classifier on the GPU as well (${Math.round(bytes / 1e6)} MB with the layers) would not leave this device enough memory`,
  stops: (most: number) => `more than ${most} stop tokens`,
  // ---- what the device is (the engine's side without a GPU; gpu/device.ts as it opens one; alone.ts)
  noWebGpu: "no WebGPU in a worker here",
  noAdapter: "no GPU adapter here",
  // T148: the CPU doing the GPU's work (SwiftShader, lavapipe)
  fallback: "a fallback adapter: the CPU in the GPU's place",
  // T232: ternary weights are multiplied as packed int8 alone
  packed: (weights: string) => `${weights} weights need the packed int8 dot product of WGSL, which this browser lacks`,
  // ---- a model on the GPU alone (alone.ts's gpuOnlyUnfit(): read as yes or no, and by the tests)
  aloneFallback: "a fallback adapter",
  // (T382: the words are older than what they say; kept as they are until that task)
  aloneArchitecture: "GPT-2 and GPT-NeoX (and a Qwen3.5 and an LFM2) are not placed on the GPU alone: only a Llama's tensors are, by external_tensors()",
  aloneDtype: (dtype: string) => `${dtype} weights stay on the CPU`,
  aloneHeads: "heads of a size that is no multiple of 4",
  aloneStarts: "q, k and v or gate and up would not start where this GPU binds a buffer",
  alonePast: "a layer's matrices are past a buffer of this GPU",
  // ---- what stopped the GPU (gpuside.ts's stopGpu(): the prompts and the tokens go on on the CPU)
  quiet: (ms: number) => `the GPU said nothing for ${ms / 1000} s`,
  stalled: (ms: number) => `the GPU's worker stopped answering for ${ms / 1000} s`,
  notStarted: (said: string | undefined) => `the GPU's worker did not start (${said ?? "an error"})`,
  letGo: "the model was let go",
  noSteps: "the GPU did not take the steps",
  notTaken: "the GPU did not take a block or a step",
  blockFailed: "the GPU failed on a block of the prompt",  // (the GPU's worker said why in the console)
  tokenFailed: "the GPU failed on a token",  // (the same)
  // T243: a key or a value that is no finite number
  keysInBlock: (position: number) => notFiniteKV(`in a block of the prompt at position ${position}`),
  keysInStep: (position: number) => notFiniteKV(`in a step at position ${position}`),
  // T219
  logits: (position: number) => `the GPU computed logits that are not finite numbers (NaN or infinity) at position ${position}`,
  sampledFew: (sampled: number, count: number) => `the GPU sampled ${sampled} of ${count} tokens`,
  sampledOutside: (id: number, vocab: number) => `the GPU sampled ${id}, outside the vocabulary of ${vocab}`,
  // T156: /benchmark/'s CPU against a model on the GPU alone (GBps: its reading, undefined where it measured none;
  // the two times in ms, the usage in tokens)
  cpuFaster: (GBps: number | undefined, cpu: number, gpu: number, prompt: unknown, written: unknown) =>
    `the CPU as /benchmark/ measured it (${GBps?.toFixed(1)} GB/s): ${cpu.toFixed(0)} ms ` +
    `against the GPU's ${gpu.toFixed(0)} ms for ${Number(prompt).toFixed(0)} tokens of prompts and ${Number(written).toFixed(0)} written`,
  // ---- a cell of /benchmark/'s page path the GPU did not run (paths.ts)
  notReady: "not ready",
  notEveryBlock: "the GPU did not take every block",
} as const;
const notFiniteKV = (where: string) => `the GPU computed keys or values that are not finite numbers (NaN or infinity) ${where}`;
