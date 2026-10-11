// contracts/forward-js.ts (T368.3): what the model's worker has of forward.js, the engine outside Python. Types alone:
// nothing of this file is in what a browser gets (it is read with `import type`).
//
// UNTIL T368.4. forward.js and forward/ are JavaScript still. What tsc reads off them is too little to hold anything
// to: a parameter is any, one with a default of null is read as null alone (which a form's linear layers are not), and
// an object that is filled after it is made is read as it was made (a row of the page's path without its cpu and
// gpu). So the worker, which is TypeScript from T368.3, writes down here what it calls of that module and how: the
// members it reads and no others. The imported module is cast to this (worker/runtime.ts's forwardModule), through
// unknown: nothing holds these shapes to forward.js until forward/ is typed. Then Forward is
// `typeof import("../forward.ts")`, the shapes below move to where they are made, and this file goes. (Its name is not forward.ts: that is the name the tools find the runtime's folder by.)
import type { CpuSpeed, GpuAsked, GpuChose, PathCell, PathRow, ThreadsCompared, Usage } from "./messages.ts";
import type { Arch, Convolution, Dtype, Linear, Rotated } from "./form.ts";
import type { Force, Opened, Remembered } from "./gpu/protocol.ts";
import type { Tensor, TensorName } from "./plan.ts";

// a checkpoint's header: the 7 ints of the legacy format
export type Header = number[];
// where every tensor of a checkpoint is (Python's external_tensors())
export type Tensors = Partial<Record<TensorName, Tensor>>;

// The kernels of one build as modules (compileKernels()): relaxed: null where the browser has no relaxed SIMD
export type Kernels = { plain: WebAssembly.Module; relaxed: WebAssembly.Module | null; wide: boolean };

// A memory weightsMemory() made: a shared one says the pages it may grow to, and whether the browser gave fewer than
// were asked for (T96)
export type Memory = WebAssembly.Memory & { maximum?: number; limited?: boolean };

// What sizes the forward pass besides the header (footprint(), keysInHalf()): the dtype and the form's part of it,
// and how the engine is to run (int8: the int8 kernels; relaxed; halfKV: the keys and values may be float16; shared:
// on a shared memory; outliers: Python's OUTLIER_CHANNELS; gpu: the page has WebGPU; direct: on the GPU alone)
export type Footprint = {
  dtype?: Dtype;
  arch?: Arch;
  head_dim?: number;
  linear?: Linear | null;
  rotated?: Rotated | null;
  convolution?: Convolution | null;
  int8?: boolean;
  relaxed?: boolean;
  halfKV?: boolean;
  shared?: boolean;
  outliers?: number;
  gpu?: boolean;
  direct?: boolean;
};

// what of a form says whether a model can go on the GPU, and how many bytes it takes there
export type GpuForm = { arch?: Arch; bias?: boolean; qk_norm?: boolean; head_dim?: number };
// the adapter as the worker keeps it (worker/runtime.ts)
export type Adapter = { fallback: boolean; limits: { maxStorageBufferBindingSize: number; maxBufferSize: number }; packed: boolean; key?: string; ternaryKey?: string };

// T156: where a model goes (weightsPlace()): both the memory and the GPU, the GPU alone, or the CPU; gpuRoom: the
// bytes left for the GPU's copy where it is not on the GPU alone
export type Place = { mode: string; gpuRoom?: number };

// T156: the weights of a model on the GPU alone as they come (gpuOnlyWeights()): write() sends the layers' matrices
// to the GPU's worker and packs the rest into the memory; place(): where a byte of the checkpoint is in the memory;
// room(): resolves once the GPU's worker is no more than a stretch behind; drained(): once it has every byte
export type GpuOnlyWeights = {
  place(offset: number): number;
  flow: SharedArrayBuffer;
  write(offset: number, chunk: Uint8Array): void;
  room(): Promise<void>;
  drained(): Promise<void>;
};

// T156: a model on the GPU alone, as the worker makes it and forward.js reads it: the bytes kept in the memory
// (stored) of size, where each is (place), the multiply-adds of its layers, /benchmark/'s CPU reading and the page's
// use (what the GPU is held against), and onLost(why), told where the GPU stops under it. forward.js writes verdict:
// the CPU is faster here (the page keeps it)
export type Direct = {
  stored: number;
  size: number;
  place(offset: number): number;
  layerWeights: number;
  cpu: CpuSpeed | undefined;
  usage: Usage;
  onLost(why: string): void;
  verdict?: { key: string; cpu: CpuSpeed };
};

// a software thread, started and warm (the worker's spawn())
export type Thread = { terminate(): void };

// what the GPU chose, once it is ready (forward/gpuside.js's gpuChosen): what the page is told of it
export type GpuReady = Omit<GpuChose, "type" | "load" | "note">;

// The engine createForward() made of a model, as far as the worker reads it (Python calls its forward pass)
export type Engine = {
  threads: number;
  setThreads(count: number): Promise<number>;
  // from: the logical cores to search from; remembered: a count to use, checked now and then; chose and compared: told
  // the count found and every comparison on the way
  findThreads(asked: { from: number; remembered?: number; chose(count: number): void; compared(verdict: Omit<ThreadsCompared, "type" | "load" | "model">): void }): Promise<number>;
  stopThreads(): void;
  newGeneration(): void;
  // a software thread stopped, and the engine goes on with one (T120)
  lostThreads: boolean;
  searchLog: { best: number; candidate: number; faster: boolean }[];
  // undefined: no GPU for this engine; else it settles, with what the status line says, once the GPU is ready or will not be
  gpu: Promise<string> | undefined;
  gpuStatus: string;
  gpuReady: GpuReady | undefined | null;
  gpuWhyNot: string | undefined | null;
  // T152: where /benchmark/ puts a generation's steps ("cpu", "gpu"; null: as the engine chooses)
  gpuSide: string | null;
  // the prompt's tokens and the steps the GPU took, and why it takes no step
  gpuTokens: number;
  gpuSampled: number;
  gpuTokensWhyNot: string | undefined | null;
};

// What Python's Llama(external=) takes (external()): engine is there once Python called start(plan)
export type Outside = { size: number; read(offset: number, length: number): Uint8Array; start(plan: unknown): Engine; engine?: Engine };

export type Forward = {
  GPU_END_MS: number;
  SEARCH_SECONDS: number;
  USAGE_UNKNOWN: Usage;
  // relaxed: null where its file did not come
  compileKernels(plain: ArrayBuffer, relaxed: ArrayBuffer | null, wide?: boolean): Kernels;
  // whether this browser makes a 64-bit memory (T101)
  memory64(): boolean;
  // base: where the checkpoint begins in the memory (after the control area of a shared one)
  weightsMemory(size: number, options?: { shared?: boolean; wide?: boolean; after?: number; spare?: number }): { memory: Memory; base: number };
  growMemory(memory: WebAssembly.Memory, pages: number, wide: boolean): void;
  // the bytes the forward pass puts after a checkpoint of size bytes, at most (T115)
  footprint(header: Header, size: number, options?: Footprint): number;
  keysInHalf(header: Header, size: number, options?: Footprint): boolean;
  needsWide(size: number, after: number): boolean;
  pastWide(size: number, after: number): boolean;
  automaticDtype(int8: number, after: number, wide: boolean): Dtype;
  external(made: {
    memory: WebAssembly.Memory;
    base: number;
    size: number;
    kernels: Kernels;
    spawn: ((data: unknown) => Promise<Thread>) | undefined;
    gpu: (() => Worker) | undefined;
    gpuRoom?: number;
    memoryUnsaid?: boolean;
    gpuRemembered: GpuAsked["remembered"];
    gpuForce: Force;
    halfKeys: boolean;
    direct?: Direct;
  }): Outside;
  // the bytes of a model's layers on the GPU
  gpuBytes(header: Header, form?: GpuForm & { dtype?: Dtype }): number;
  // why a model cannot go on the GPU alone, or null
  gpuOnlyUnfit(header: Header, dtype: Dtype, form: (GpuForm & { rotated?: Rotated | null }) | undefined, adapter: Adapter | null, force?: Force): string | null;
  // whether what the page kept (the CPU was faster than this model on the GPU alone) still holds: the same device's
  // key and the same reading of the CPU
  aloneHolds(alone: Remembered["alone"], key: string | undefined, cpu: CpuSpeed | undefined): boolean;
  weightsPlace(sizes: { cpu: number; gpuOnly?: number; gpu: number; deviceMemory?: number; eligible?: boolean; forced?: boolean }): Place;
  // the stretches of a checkpoint that go to the GPU alone, [begin, end) each
  gpuHoles(tensors: Tensors): number[][];
  gpuOnlyWeights(model: { memory: WebAssembly.Memory; base: number; size: number; tensors: Tensors; worker: Worker }): GpuOnlyWeights;
  gpuOnlyPlan(header: Header, tensors: Tensors, force: Force | undefined, remembered: GpuAsked["remembered"]): Opened;
  layerWeightsOf(header: Header, options?: { head_dim?: number }): number;
  // T190: the search for the number of threads run to its end (write() writes some tokens): the count the engine
  // runs on, the count found, and whether the search ended in time
  endSearch(engine: Engine, write: () => unknown): Promise<{ threads: number; found: number; ended: boolean }>;
  // T184: prompts of counts tokens (of words, the prompt's ids) on each side, a row a count
  timePrompts(engine: Engine, asked: { words: number[]; counts: number[] }): PathRow[];
  // the runs of one side as a cell: count tokens in ms each, and the tokens the GPU took
  timedCell(runs: { ms: number; gpuTokens: number }[], count: number): PathCell;
};
