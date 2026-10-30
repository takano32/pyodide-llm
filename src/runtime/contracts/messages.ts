// contracts/messages.ts (T.368.2): what a page and the model's worker (src/runtime/worker.js) say to one another. Types
// alone: nothing of this file is in what a browser gets (it is read with `import type`).
//
// Two pages talk to that worker: the model page (src/page/, and the script of src/pages/index.astro, which sends the
// first load) and /benchmark/'s model section (src/benchmark/model.ts). The other workers of /benchmark/ have their own
// words (one question, one answer: src/benchmark/section.ts's ask()), which are not here.
//
// ToWorker: what a page sends. FromWorker: what the worker reports. Of the reports, only ready, bench, paths, done and
// error end a piece of the worker's work (T.172): worker.ts's head says which comes when.
// load: a number the page counts up. A newer load cancels the one going on, and what the worker reports about a load
// carries its number, so that the page can tell a late report of a cancelled one. A report that never carries one says
// `load?: undefined`.
//
// Both sides are TypeScript and are held to these by tsc: the page's by ModelWorker, below, the worker's (T.368.3) by
// what its receiver takes (MessageEvent<ToWorker>) and by `satisfies` at every postMessage().
// tests/worker-modules-check.mjs still holds the kinds and the keys the worker posts and reads to these types: it says
// a key that is in the type and written nowhere, and a postMessage() without its `satisfies`, which tsc does not.
import type { Dtype } from "./form.ts";
import type { Options } from "./python.ts";

// An item of src/models.ts as the page weighed it (src/page/address.ts's weighed()), or one the page made of a folder,
// of two files, of a repository or of two URLs: as far as the worker reads it (T.368.3; the list's own shape is that
// file's, which is JavaScript still, and has more). One of four kinds, told apart by hf, file and url:
export type Listed = OfThisSite | OfTwoFiles | OfTwoUrls | OfHuggingFace;
// what every kind has: an id (an item's of the list; "local" for one the page made), its name, and what its engine
// is made with (for a model that is converted, over what its conversion says: an item's options win)
type Named = { id: string; name: string; options?: Options };
// a model of this site: its parts under models/ (<checkpoint>.000, ...), their bytes together, its tokenizer's file
export type OfThisSite = Named & { checkpoint: string; tokenizer: string; bytes: number; options: Options; hf?: undefined; file?: undefined; url?: undefined };
// two files of the visitor's disk, a llama2.c checkpoint (bytes: its size) and its tokenizer.bin, read where they are
export type OfTwoFiles = Named & { file: File; tokenizerFile: File; bytes: number; hf?: undefined; url?: undefined };
// the same at two URLs (?checkpoint=&tokenizer=); bytes: the worker's own, what the server says the checkpoint's size is
export type OfTwoUrls = Named & { url: { checkpoint: string; tokenizer: string }; bytes?: number; tokenizer?: undefined; hf?: undefined; file?: undefined };
// A Hugging Face model, converted in the worker. hf: the names of its files, which Python's conduct reads
// (src/python/convert/conduct.py), and where they are: huggingface.co's repository and revision (vocabulary: another
// repository for the vocabulary, the config and the template, T.136), or files: the Files of a folder of the visitor's
// disk (T.374.2.2). conversion: the dtype asked for (none: the worker chooses, T.115) and the context it is cut to; blocks
// (T.363.5): a GGUF's K- and i-quantized tensors are kept as the file has them, llama.cpp's blocks, and not widened.
// shares: the other ids of the list that are the same conversion (kept.ts)
export type Hf = {
  repo?: string;
  revision?: string;
  weights?: string;
  config?: string;
  tokenizer?: string;
  vocabulary?: { repo: string; revision: string; config?: string; tokenizer?: string };
  files?: File[];
};
export type OfHuggingFace = Named & { hf: Hf; conversion?: { dtype?: Dtype; max_seq_len?: number; blocks?: boolean }; shares?: string[]; file?: undefined; url?: undefined };

// T.93 stage 2: what the page asks about the number of threads: fixed (?threads=N), the count it remembers for this
// device and model, and the logical cores to search from
export type ThreadsAsked = { fixed?: number; remembered?: number; hint?: number };
// what /benchmark/'s CPU section measured here, which a model on the GPU alone is held against (T.156)
export type CpuSpeed = { GBps?: number; threads?: number; promptGMACs?: number };
// how the page is used: the tokens of its prompts and those written, decayed (T.156)
export type Usage = { prompt: number; written: number };
// T.156: the worker's verdict that the CPU is faster here than this model on the GPU alone, under the device's key
export type Alone = { key: string; cpu: CpuSpeed };
// what the page kept of the GPU's choices for this model (T.148, T.152)
export type GpuRemembered = { key?: string; matrices?: string; attention?: string; tokens?: string; alone?: Alone };
export type GpuAsked = { remembered?: GpuRemembered; cpu?: CpuSpeed; usage?: Usage };

// What a load asks for: the model, the page's own address (its ?without=, ?kernel=, ?gpuTest= and the like), the
// site's root (T.367.3: models/ is under it), and T.242, ahead: the switches of each load that will follow on this
// model, where it is the only one this worker loads (/benchmark/)
export type Asked = { root?: string; search: string; model: Listed; load: number; threads?: ThreadsAsked; gpu?: GpuAsked; ahead?: string[][] };

// The settings of an answer (src/python/engine/sampler.py's Sampling and generate()'s own): the worker hands them to
// Python as they are. steps: 0 for as many as the context holds. echo: the prompt is written back first.
export type Sampled = {
  steps?: number;
  temperature?: number;
  topp?: number;
  repetition_penalty?: number;
  seed?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  // T.520: the budget of a thought, in tokens (generate()'s thinking: 0 or none, no budget)
  thinking?: number;
};

// T.45: a round of ?bench=1 (src/bench/rounds.ts): the switches it leaves out; skip: the words of its row where the
// device does not run it (T.214)
export type Round = { name: string; without: string[]; skip?: string };

export type ToWorker =
  // the first load of a worker, and every load after it
  | ({ type: "init" } & Asked)
  | ({ type: "load" } & Asked)
  // (load: neither is of a load; the worker's receiver reads data.load of whatever failed)
  | ({ type: "generate"; load?: undefined; prompt: string; echo: boolean } & Sampled)
  | { type: "stop"; load?: undefined }
  | { type: "bench"; model: Listed; load: number; rounds: Round[]; prompt: string; steps: number }
  // T.184: the model page's own path, timed: prompts of counts tokens, and sampled tokens written
  | { type: "paths"; load: number; prompt: string; counts: number[]; sampled: number };

// the seconds of a load's parts (worker/clock.ts): Pyodide once a session, the others per model
export type LoadSeconds = { pyodide: number; download: number; construct: number; convert?: number };

// the engine's stats of an answer (src/python/engine/generation.py)
export type Stats = {
  tokens: number;
  seconds: number;
  tokens_per_second: number;
  sampled: number;
  prompt_tokens: number;
  prompt_seconds: number;
  prompt_tokens_per_second: number;
  first_token_seconds: number;
  // T.520: the tokens of the answer's thought (null: none), and whether its budget closed it
  thought: number | null;
  thought_cut: boolean;
};

// a row of the rounds: one that ran (its tokens, tok/s, the engine's word of how it ran, the seconds of its load), or
// one skipped (T.214: why)
export type BenchRow =
  | { name: string; without: string[]; skip?: undefined; tokens: number; speed: number; backend: string; seconds: number }
  | { name: string; without: string[]; skip: string };

// T.184: runs of a side as a cell (forward/paths.ts's timedCell()), or why it has none, or that it is the CPU's
export type PathCell = { speed?: number; low?: number; high?: number; gpuTokens?: number; unsteady?: boolean; skip?: string; same?: string };
export type PathRow = { what?: string; tokens: number; chosen?: PathCell; cpu: PathCell; gpu: PathCell };
// how the number of threads the path ran on came about (T.190)
export type PathThreads = { alone?: string; unfinished?: number; remembered?: boolean; searched?: [number, number, number][]; stopped?: boolean };

// during a load (and, without a load's number, while Pyodide loads and while the page's path is timed)
export type Status = { type: "status"; load?: number; text: string };
// converted: the share of a conversion done; perSecond: the bytes arriving
export type Progress = { type: "progress"; load: number; received: number; total: number; converted?: number; perSecond?: number };
// A load ended (each round of the benchmark too). gpu: what the status line says of the GPU now. overlapped: a
// download of this site ran while Pyodide loaded. Of a conversion (worker/convert.ts): fromCache, keptIn (where it was
// kept), notKept (why not), keptMiss (why what was kept was not used), template (its chat_template's one turn)
export type Ready = {
  type: "ready";
  load: number;
  pyodide: string;
  backend: string;
  seq_len: number;
  seconds: LoadSeconds;
  heap: number | undefined;
  threads: number;
  gpu: string;
  overlapped?: boolean;
  fromCache?: boolean;
  keptIn?: string;
  notKept?: string;
  keptMiss?: string;
  template?: string;
  // T.385: of a conversion, how each stored type that has a kernel's reader was read ("kernel" or "NumPy": tests/e2e.mjs)
  widened?: Record<string, "kernel" | "NumPy">;
};
export type Token = { type: "token"; load?: undefined; text: string };
// an answer ended (also one stopped): the threads in use, the prompt's tokens and the steps the GPU took, its note
export type Done = { type: "done"; load?: undefined; threads: number; gpuTokens: number; gpuSampled: number; gpu: string | undefined } & Stats;
export type Bench = { type: "bench"; load: number; rows: BenchRow[]; pyodide: string };
// T.184: either error, or the rest (worker/timing.ts's timedPaths())
export type Paths = {
  type: "paths";
  load: number;
  error?: string;
  threads?: number;
  how?: PathThreads;
  perCount?: ({ threads: number } & PathCell)[];
  gpu?: { seconds?: number; matrices?: string; attention?: string; why?: string; lost?: string };
  status?: string;
  rows?: PathRow[];
};
// reloading (T.156): no error of its own, the model is loaded again on the CPU; else stack, weights (the bytes of the
// weights' memory) and pyodide (the runtime did not load); memory and heap where the memory ran out (T.90)
export type Failed = {
  type: "error";
  load: number | undefined;
  message: string;
  reloading?: boolean;
  stack?: string;
  weights?: number;
  pyodide?: boolean;
  memory?: boolean;
  heap?: number;
};
// T.93: the number of threads found for this device and model (model: its id), and T.114: every comparison on the way
export type Threads = { type: "threads"; load?: undefined; model: string; count: number; from: "remembered" | "hint"; hint: number };
export type ThreadsCompared = {
  type: "threads-compared";
  load?: undefined;
  model: string;
  best: number;
  candidate: number;
  bestMs: number;
  candidateMs: number;
  faster: boolean;
  whileGpu: boolean;
  tokens: number;
};
// T.148: the GPU is ready, with what it chose (the page shows it and remembers it), or it will not be
export type GpuChose = {
  type: "gpu";
  load: number;
  note: string;
  matrices: string;
  attention: string;
  key: string;
  remembered: boolean;
  seconds: number;
  tokens?: string;
  tokenAttention?: string;
  tablePieces?: number;
};
export type GpuSaid = { type: "gpu"; load: number; note: string; matrices?: undefined };
// T.156: the CPU is faster here than this model on the GPU alone (kept once its load on the CPU is ready)
export type GpuAlone = { type: "gpu-alone"; load: number; alone: Alone };
// T.439: the worker went on without a file of its build that is gone, on a slower path (worker/outdated.ts): once a
// worker at most, whatever else is going on, and of no load (the page is of a build the site no longer serves)
export type Updated = { type: "updated"; load?: undefined };

export type FromWorker = Status | Progress | Ready | Token | Done | Bench | Paths | Failed | Threads | ThreadsCompared | GpuChose | GpuSaid | GpuAlone | Updated;

// The model's worker as a page has it: a Worker that takes ToWorker and reports FromWorker
export type ModelWorker = Omit<Worker, "postMessage" | "onmessage"> & {
  postMessage(message: ToWorker): void;
  onmessage: ((event: MessageEvent<FromWorker>) => void) | null;
};
