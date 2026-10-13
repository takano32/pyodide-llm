// contracts/benchmark.ts (T368.7): what /benchmark/'s page and the workers of its sections say to one another. Types
// alone: nothing of this file is in what a browser gets (it is read with `import type`).
//
// The page (src/benchmark/*.ts) starts a worker for a section and ends it after. The workers are
// src/runtime/benchmark/sections.ts (the browser, the CPU, the line), storage.ts, memory.ts, gpu.ts (and waiter.ts, its
// bridge's other end); each one's head says what its steps do. A step is asked by one message and answered by one
// (Answer: its result, or the words of what failed); while it runs a worker may say other things, which answer nothing
// (Aside: the stage it is at, that it is alive, that the GPU's device was lost) and which the page's ask() leaves out.
// Both ends are TypeScript and read these types: a key one end writes and the other does not read is the compiler's
// to say. What the tables of src/bench/ (JavaScript) read of a result is not held to them.

// ---- what answers nothing
// T177: the stage a step is at, for the page to show: at / of where the step knows how many it has, part (0 to 1) of
// the stage where that is known
export type Staged = { stage: string; at?: number; of?: number; part?: number; alive?: undefined; lost?: undefined };
// still at it (the page stops a section that says nothing for 5 minutes)
export type Alive = { alive: true; stage?: undefined; lost?: undefined };
// the GPU's device was lost, and why: "<the reason>: <the device's words>" (never empty, which is how the page tells it
// from an answer: by whether there is one)
export type Lost = { lost: `${string}: ${string}`; stage?: undefined; alive?: undefined };
export type Aside = Staged | Alive | Lost;

// ---- an answer: the step it answers (the workers that say it), and its result or what failed
type Answering = { step?: string; stage?: undefined; alive?: undefined; lost?: undefined };
export type Answered<Result> = Answering & { result: Result; error?: undefined };
export type Failed = Answering & { error: string; result?: undefined };
export type Answer<Result> = Answered<Result> | Failed;
// what a worker posts: an answer, or something that answers nothing
export type Said<Result> = Answer<Result> | Aside;

// ---- sections.ts: this browser, the CPU, the line
// a file fetched for the line: where it is, how many bytes of it, and (false) that a range of it cannot be asked for
export type Fetched = { url: string; bytes: number; range?: boolean };
export type ToSections = { step: "device" } | { step: "cpu"; threads?: number[] } | { step: "line"; site: Fetched; hf: Fetched; rates?: number[]; seconds?: number };
export type DeviceFound = { crossOriginIsolated: boolean; sharedMemory: boolean; simd: boolean; relaxedSimd: boolean; memory64: boolean; webgpu: boolean;
  opfs: boolean; syncHandle: boolean; waitAsync: boolean; cores: number | null; memoryGB: number | null; usage?: number; quota?: number };
// a ceiling of the CPU (T163): its rate, or why there is none; cutShort: the loop did not do its work
export type CpuCeiling<Rate extends string> = ({ [Key in Rate]: number } & { passes: number; ratio: number; unsteady?: boolean; error?: undefined; none?: undefined; cutShort?: undefined })
  | ({ [Key in Rate]?: undefined } & { error: string; cutShort?: boolean; none?: undefined })
  | ({ [Key in Rate]?: undefined } & { none: string; error?: undefined; cutShort?: undefined });
export type CpuCeilings = { read?: ({ threads: number } & CpuCeiling<"GBps">)[]; dot?: CpuCeiling<"GMACs">; dotRegisters?: CpuCeiling<"GMACs">; fma?: CpuCeiling<"GMACs">; error?: string };
// a row of the CPU's table: a count of software threads asked for, and what it measured (none: why it did not)
export type CpuRow = { asked: number; threads: number; none: string; msPerToken?: undefined; finite?: undefined; promptMsPerToken?: undefined }
  | { asked: number; threads: number; msPerToken: number; GBps: number; promptMsPerToken: number; finite: boolean; none?: undefined };
export type CpuFound = { none: string; rows?: undefined; ceilings?: undefined }
  | { backend: string; shared: boolean; megabytes: number; tokenMegabytes: number; layerWeights: number; rows: CpuRow[]; ceilings: CpuCeilings; none?: undefined };
// a fetch of the line, measured, or what failed
export type LineFetched = { status: number; headersMs: number; firstByteMs: number; bytes: number; MBps: number; error?: undefined }
  | { error: string; status?: undefined; headersMs?: undefined; firstByteMs?: undefined; bytes?: undefined; MBps?: undefined };
export type LinePaced = { rate: number; slower: true; error?: undefined } | ({ rate: number; slower?: undefined } & LineFetched);
export type LineFound = { site: LineFetched | null; hf: LineFetched | null; paced: LinePaced[] };
export type SectionsResults = { device: DeviceFound; cpu: CpuFound; line: LineFound };

// ---- storage.ts: the private file system
export type ToStorage = { step: "info" } | { step: "run"; mib: number };
export type StorageInfo = { opfs: boolean; syncHandle: boolean; usage?: number; quota?: number; estimate?: string };
export type StorageWrites = { seconds: number; flushSeconds: number };
export type StorageRun = { mib: number; pieces: number; openSeconds: number; secondHandle: string; truncateSeconds: number; sequential: StorageWrites;
  scattered: StorageWrites; scatteredFlushEach: StorageWrites; read: { seconds: number; wrong: number }; reopen: string };
export type StorageResults = { info: StorageInfo; run: StorageRun };

// ---- memory.ts: how much this page can hold (src/page-memory.js asks, which is JavaScript)
export type ToMemory = { step: "grow"; to: number; limit: number };
export type MemoryGrown = { held: number; maximum: number | null; refused?: string };
export type MemoryResults = { grow: MemoryGrown };

// ---- waiter.ts: the other end of the GPU worker's bridge; it is told once and answers once, with no step
export type ToWaiter = { memory: SharedArrayBuffer; rounds: number };
export type Waited = { microseconds: number };

// ---- gpu.ts: the GPU, a step at a time (its head says what each does)
export type GpuModel = "llm-jp-3 150M" | "Llama 3.2 1B" | "Llama 3.2 3B";
export type ToBenchGpu = { step: "info" } | { step: "check" } | { step: "bandwidth"; shape: [rows: number, n: number] }
  | { step: "token"; model: GpuModel; kind?: "widen" | "packed"; sample?: boolean } | { step: "layer" } | { step: "layer steps" } | { step: "generate" }
  | { step: "overhead" } | { step: "prompt"; counts?: number[] } | { step: "ceilings" } | { step: "bridge"; memory: SharedArrayBuffer; rounds: number };
// the adapter: whether a worker has WebGPU at all, and what the adapter says of itself
export type GpuAbout = { adapter: string; fallback: boolean | null; maxStorageBufferBindingSize: number; maxBufferSize: number; maxComputeWorkgroupsPerDimension: number;
  maxComputeWorkgroupStorageSize: number; maxComputeInvocationsPerWorkgroup: number; subgroupSizes: [number, number] | null; features: string[]; wgsl: string[]; packed: boolean };
export type GpuInfo = { worker: boolean } & Partial<GpuAbout>;
// a verdict of the check, of a shader or a form by its name: ok, how far off at the worst, and what its kind of check
// says besides (error: it could not be checked)
export type Verdict = { ok: boolean; worstRelative: number; error?: string;
  // a matrix × vector's (T149), a tiled shader's quantizing (T146)
  wrotePastTheRows?: boolean; apart?: number; far?: boolean;
  // a layer's (T175, T225)
  stages?: string; stream?: number; cache?: number; wroteOtherPositions?: boolean; sameAsNormsApart?: { ok: boolean; bitForBit: boolean; ulps: number; apart: number; stream: number };
  quantized?: { point: string; wrong: string | null; scale: number; apart: number; more: number; of: number }[];
  // a token's attention (T224): where it was farthest off
  at?: { headSize: number; positions: number };
  // the sampling's and the generated tokens' (T151, T225): the tokens checked, those right by an edge only, what was wrong,
  // the layer the tokens ran and how each step held
  tokens?: number; edge?: number; problems?: string[]; layer?: string; steps?: string };
export type Verdicts = Record<string, Verdict>;
// a row of the layer's table (T150, T175): a form, and its time where it ran (none, error: why it did not)
export type LayerRow = { form: string; check: string; base: string; fused: boolean; subgroups: boolean; normApart?: boolean; none?: string; error?: string;
  dispatches?: number; msPerLayer?: number; layers?: number; GBps?: number; ratio?: number; unsteady?: boolean };
// what a token costs besides its weights (ms each)
export type Overhead = { dispatches: number; emptyDispatches: number; submitOnly: number; submitAndWait: number; readToken: number; readLogits: number; vocab: number };
// a ceiling of the device (T168): its rate by its unit's name, or none (why it was not measured) or error
export type GpuCeiling = { none?: string; error?: string; unsteady?: boolean; GFLOPS?: number; GOPS?: number; GBps?: number; MiB?: number;
  shape?: string; shapes?: { shape: string; GFLOPS: number }[]; loops?: number; dispatches?: number; ratio?: number };
export type GpuCeilings = { fallback: boolean; f32: GpuCeiling; f16: GpuCeiling; dot4: GpuCeiling; shared: GpuCeiling; global: GpuCeiling };
// a prompt all at once: a row a shader and a count of tokens (none, error: why there is no time)
export type PromptRow = { shader: string; none?: string; error?: string; packed?: boolean; half?: boolean; again?: boolean;
  tokens?: number; ms?: number; msPerToken?: number; GFLOPS?: number };
export type PromptFound = { rows: PromptRow[]; layers: number; weights: number; GB: number };
// one matrix times a vector by every shader (T149), and the CPU's kernel on the same bytes
export type BandwidthRow = { shader: string; check: string; again?: boolean; none?: string; error?: string;
  GBps?: number; msEach?: number; dispatches?: number; ratio?: number; unsteady?: boolean };
export type CpuBandwidth = { GBps: number; msEach: number; threads: number };
export type Bandwidth = { rows: BandwidthRow[]; copies?: number; quantize?: { msEach?: number; unsteady?: boolean; error?: string };
  widen?: BandwidthRow; packed?: BandwidthRow; cpu: CpuBandwidth };
// a whole token's work of a model's shapes (error: the device did not hold or take the weights)
export type TokenFound = { model: GpuModel; error?: string; kind?: "widen" | "packed"; sample?: boolean; GB?: number; msPerToken?: number;
  tokPerSecond?: number; GBps?: number; dispatches?: number };
export type LayerFound = { model: string; pos: number; layers: number; copies: number; GB: number; rows: LayerRow[] };
// T202: a time of the steps' table: ms each of n, or why there is none
export type Timing = { ms?: number; n?: number; ratio?: number; unsteady?: boolean; error?: string; none?: string };
// a form whose layer the steps' table breaks down: how many of each step it dispatches, and the whole layer's time
export type StepsForm = Timing & { form: string; check: string; normApart: boolean; dispatches?: number; steps?: { step: string; count: number }[] };
// a step timed alone: its name, its kind ("matrix", "attention", "small", "alone", "floor") and the matrix it reads
export type StepTimed = Timing & { step: string; kind: string; matrix?: string };
// T208: the whole layer by the GPU's own clock
export type Timestamps = { none?: string; error?: string; layers?: number; rounds?: number; forms?: { form: string; ms: number; span: number }[] };
// T224: a token's attention alone, by the cache's length
export type AttentionRow = { attention: string; tiles?: boolean; times: Timing[] };
export type AttentionLengths = { error?: string; positions?: number[]; rows?: AttentionRow[]; base?: number; MB?: number; bytes?: number[] };
export type LayerStepsFound = { model: string; pos: number; layers: number; copies: number; GB: number; dp4a: boolean; chosen: { by: string; why?: string; fastest?: string };
  forms: StepsForm[]; steps: StepTimed[]; cycleMB?: number; spares?: number; caches?: { count: number; MB: number }; timestamps?: Timestamps; lengths: AttentionLengths };
// T151, T191: tokens generated on the GPU. A token's work without the reading back, by a sampler
export type TokenWork = { ms: number; tokens?: number; ratio?: number; unsteady?: boolean };
// the sampling alone: SAMPLE's one workgroup, and in chunks with its last stage (pick) alone; over: the tokens over the floor
export type SamplingTimed = { error?: string; msEach?: number; unsteady?: boolean };
export type SamplingAlone = SamplingTimed & { over: number; chunks: SamplingTimed & { pick: SamplingTimed } };
export type GenerateFound = { model: string; layer: string; layers: number; vocab: number; GB: number; dispatches: number; chunkDispatches: number; tokens: number;
  settings: { temperature: number; topp: number; penalty: number }; work?: TokenWork; chunkWork?: TokenWork;
  rows: { perSubmission: number; msPerToken: number; fixedMs?: number; chunks: number }[]; sampling?: SamplingAlone & { vocab: number; flat: SamplingAlone } };
// the bridge: whether this worker has Atomics.waitAsync to answer the waiter with
export type Bridged = { waitAsync: boolean };
export type BenchGpuResults = { info: GpuInfo; check: Verdicts; bandwidth: Bandwidth; token: TokenFound; layer: LayerFound; "layer steps": LayerStepsFound;
  generate: GenerateFound; overhead: Overhead; prompt: PromptFound; ceilings: GpuCeilings; bridge: Bridged };
// what a worker's steps are asked and answer, by the worker: for the page's ask()
export type Asked = { "benchmark/sections.js": { to: ToSections; results: SectionsResults }; "benchmark/storage.js": { to: ToStorage; results: StorageResults };
  "benchmark/memory.js": { to: ToMemory; results: MemoryResults }; "benchmark/gpu.js": { to: ToBenchGpu; results: BenchGpuResults } };
