// contracts/python.ts (T.368.3): what the model's worker has of Pyodide and of the site's Python. Types alone: nothing
// of this file is in what a browser gets (it is read with `import type`).
//
// Pyodide is loaded at run time, the latest release (policy 2: src/runtime/worker/pyodide.ts resolves the version), so
// no version of it is written anywhere, and none is here: these are the few members the worker calls, as it calls
// them, and not Pyodide's own declarations (npm's pyodide is what the tools run, and is in no bundle). A member the
// worker comes to call is added here.
// What Python hands JavaScript is a proxy: it keeps its Python object alive until destroy() (AGENTS.md: a proxy left
// behind held a model's weights). The engine's and the converter's windows (src/python/llama2_numpy.py,
// llama2_convert.py) and the conduct (convert/conduct.py) are read through such proxies, by the names below.
//
// The other end is Python, which no compiler reads with this: tests/contracts-check.ts holds every name of
// EngineWindow, ConverterWindow and ConductModule to a name of that module's file, and the members of Llama to
// src/python/engine/. The values (what a call takes and gives) are held by what runs the worker on the real Python:
// tests/smoke.ts, tests/worker-fetches-check.ts, the browsers.
import type { Stats } from "./messages.ts";
import type { Dtype, Form } from "./form.ts";
import type { Tensor, TensorName } from "./plan.ts";
import type { Request } from "./conduct.ts";

// what keeps a Python object alive until it is let go of
export type Proxy = { destroy(): void };
// Python's bytes as JavaScript sees them in place (getBuffer("u8")): a view that dies when Pyodide's memory grows,
// so it is taken anew for every use and released
export type View = { data: Uint8Array<ArrayBuffer>; release(): void };
// a bytearray, a bytes, a NumPy array of bytes
export type Bytes = Proxy & { getBuffer(type: "u8"): View };
// a dict, read as an object; a list or a tuple, read as an array
export type Dict<T> = Proxy & { toJs(options: { dict_converter: typeof Object.fromEntries }): T };
export type List<T> = Proxy & { toJs(options?: { depth: number }): T };

// A generator of Python's as JavaScript has it: next(sent) is send(sent); return() closes it (its finally runs)
export type Generator<Given, Sent = undefined> = Proxy & { next(sent?: Sent): { done?: boolean; value: Given }; return(): unknown };

// what an engine is made with besides its files, and what a conversion says of the model it made (its options): the
// form (contracts/form.ts), the dtype, and whatever else the two Pythons hand one another by name (the tokenizer's
// kind, the stop tokens, RoPE's scaling...), which JavaScript carries and does not read. template: a conversion's,
// for the page (the format of one turn), taken out before the engine gets the rest
export type Options = Partial<Form> & { dtype?: Dtype; template?: string } & Record<string, unknown>;

// Python's Llama (src/python/engine/model.py) of the model that is loaded
export type Llama = Proxy & {
  // the engine's word of how it runs ("NumPy", "int8, relaxed SIMD"...)
  backend: string;
  seq_len: number;
  bos: number;
  // generate(prompt, **settings): a piece of text a step (engine/generation.py)
  generate: { callKwargs(prompt: string, settings: object): Generator<string> };
  // the stats of the last generation, once its generator is closed
  stats: Dict<Stats>;
  tokenizer: { encode(text: string): List<number[]> };
  // what an engine outside Python holds (forward.ts's, its threads, the GPU's worker): a promise where there is a
  // GPU's worker to wait for; no such method where NumPy runs the model
  release?(): Promise<unknown> | undefined;
};

// src/python/llama2_numpy.py, the engine's window
export type EngineWindow = Proxy & {
  // T.92: how many channels of the classifier's input go as float32 at most
  OUTLIER_CHANNELS: number;
  // Llama(checkpoint, tokenizer, **options): checkpoint is None where the weights are outside Python (external=)
  Llama: { callKwargs(checkpoint: Bytes | null, tokenizer: Bytes, options: Options & { external?: unknown; kernels?: string; disable?: string[] }): Llama };
  // T.89: the quantizer on the SIMD kernels (a callable the converter takes as quantize_rows)
  kernel_quantizer(kernels: string): Proxy;
  // where every tensor of a checkpoint is, from its header (the 7 ints), dtype and form alone (T.156)
  external_tensors(header: number[], dtype: Dtype, form: Partial<Form>): Dict<Partial<Record<TensorName, Tensor>>>;
  // T.363.5: how many weights of a checkpoint's matrices its dtype itself packs, where its form gives rows kinds of their own
  dtype_weights(header: number[], dtype: Dtype, form: Partial<Form>): number;
  // the dtype of a legacy file of so many bytes, by its header and its form; ValueError where it is none
  checkpoint_dtype(header: Proxy, bytes: number, options: Options): Dtype;
  // ValueError where the tokenizer.bin is of another vocabulary than the header's
  check_tokenizer(pieces: Proxy, header: Proxy): void;
};

// src/python/llama2_convert.py, the converter's window
export type ConverterWindow = Proxy & {
  // T.273: the readers of the stored types that have a kernel, by the type's name
  kernel_readers(kernels: string): Proxy;
};

// What a conversion ends with (convert/conversion.py's Conversion): the checkpoint went to the sink as it came
// (T.385: widened, how each stored type that has a kernel's reader was read: "kernel" or "NumPy")
export type Conversion = Proxy & { options: Dict<Options>; tokenizer: Bytes; widened: Dict<Record<string, Widened>> };
export type Widened = "kernel" | "NumPy";

// src/python/convert/conduct.py: conduct(hf, **make), a generator of requests (contracts/conduct.ts), each a tuple
export type ConductModule = Proxy & { conduct: Proxy & { callKwargs(hf: Proxy, make: object): Conduct } };
export type Conduct = Generator<List<Request>, unknown>;

// Where the converter writes a checkpoint (convert/checkpoint.py's Writer, sink=): open() once the size is known (and
// again where another tokenizer is tried), then write() for every piece. header: the 7 ints; form: FORM as a dict
export type Sink = {
  open(bytes: number, header: List<number[]>, dtype: Dtype, form: Dict<Partial<Form>>): void;
  write(offset: number, array: Bytes): void;
};

// Pyodide, as far as the worker calls it
export type Pyodide = {
  version: string;
  pyimport(name: "llama2_numpy"): EngineWindow;
  pyimport(name: "llama2_convert"): ConverterWindow;
  pyimport(name: "convert.conduct"): ConductModule;
  pyimport(name: "jinja2"): Proxy;
  globals: { get(name: "bytearray"): (size: number) => Bytes };
  toPy(value: unknown): Proxy;
  runPython(code: string): unknown;
  // (an archive's bytes: a Node Buffer is refused)
  unpackArchive(bytes: ArrayBuffer, format: "zip"): void;
  loadPackage(name: string, options?: { checkIntegrity?: boolean; messageCallback?: (message: string) => void }): Promise<unknown>;
  FS: { writeFile(path: string, data: Uint8Array): void };
  // Emscripten's module: its memory, which only grows
  _module?: { HEAPU8?: { length: number } };
};
// pyodide.mjs, as the loader imports it
export type PyodideModule = { loadPyodide(): Promise<Pyodide> };
