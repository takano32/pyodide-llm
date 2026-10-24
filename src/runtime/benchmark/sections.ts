// The sections of /benchmark/ (T.134) that need neither Pyodide nor a GPU, in a module worker of their own: the page
// starts one for a section and ends it after, so that no WebAssembly memory of one section outlives it (T.96: Chromium
// refused a page its third memory; a phone has little room for the next section's).
//
//   { step: "device" }                          what this browser can do: SIMD, relaxed SIMD, 64-bit memories, shared
//                                               memory, WebGPU and OPFS in a worker, the storage the browser grants
//   { step: "cpu", threads }                    the forward pass of forward.ts, the one the model page runs, on a
//                                               made-up int8 model of Llama 3.2 1B's width (random weights, two
//                                               layers), one token at a time at each count of software threads, and
//                                               a prompt's tokens 16 at a time (T.108), as the GPU section does them;
//                                               and the CPU's ceilings beside them (T.163)
//   { step: "line", site, hf, rates, seconds }  the line: a part of the site's model and a range of a model on
//                                               huggingface.co, how long until the first byte and how fast; then
//                                               huggingface.co read no faster than each of rates (MB/s)
//
// The first message is claimed before anything is awaited (a module worker's port opens at its first await, and a
// message that comes before onmessage is set is lost: T.109). While a step runs, it tells the page each stage it starts
// ({stage, at, of}, T.177): the page shows it, and it answers none of the page's questions.
import { built } from "../built.ts";
import { isUpdated, saidOf, named, unstarted } from "../updated.ts";
import { Failure, kindOf } from "../failure.ts";
// T.375.2: the CPU section's made-up model and its plan (a module of its own, so that a test holds the plan to Python's)
import { GROUP, SHAPE, madeUpModel } from "./madeup.ts";
import type { MadeName, Model } from "./madeup.ts";
// T.368.7: what the page asks and is answered (contracts/benchmark.ts), and the engine's own types of what it is handed
import type * as Forward from "../forward.ts";
import type { Compiled, Spawn } from "../forward.ts";
import type { Built } from "../built.ts";
import type { Alive, Answer, CpuCeiling, CpuCeilings, CpuFound, CpuRow, DeviceFound, Failed, LineFetched, LineFound, SectionsResults, Staged, ToSections } from "../contracts/benchmark.ts";
// a caught value, where its words are read
type Thrown = { name?: string; message?: string } | null | undefined;
// a loop of the ceilings' modules, as it is called here: by numbers, for a checksum
type Loop = (...numbers: number[]) => number;
// what paired() measured, as a rate
type Rated = { rate: number; passes: number; ratio: number; unsteady?: boolean };
// what an attempt at a ceiling gives: what it measured, or the words of what failed
type Attempted<Measured> = Measured | { error: string; cutShort?: boolean };
// the ceilings' loops on a memory (ceilingsOf()), with what cpu() puts beside them as it measures
type Ceilings = { relaxed: boolean; read: NonNullable<CpuCeilings["read"]>; reading(count: number): Promise<{ GBps: number } & Omit<Rated, "rate">>;
  dotting(): { GMACs: number } & Omit<Rated, "rate">; dottingRegisters(): { GMACs: number } & Omit<Rated, "rate">; fmaing(): { GMACs: number } & Omit<Rated, "rate">; stop(): void;
  dot?: CpuCeiling<"GMACs">; dotRegisters?: CpuCeiling<"GMACs">; fma?: CpuCeiling<"GMACs"> };
const stage = (name: string, at: number, of: number) => postMessage({ stage: name, at, of } satisfies Staged);

self.onmessage = async ({ data }: MessageEvent<ToSections>) => {
  try {
    const result = data.step === "device" ? await device() : data.step === "cpu" ? await cpu(data.threads)
      : await line(data);
    postMessage({ step: data.step, result } satisfies Answer<SectionsResults[keyof SectionsResults]>);
  } catch (error) {
    postMessage({ step: data.step, error: failed(error) } satisfies Failed);
  }
};
// What a caught value is said as. T.367.4: a file of this build that is gone (the page was left open across a
// deployment) is said as that, alone, and ends the step wherever it is caught: a section measures this build, and what
// half of its files measure is nothing to report.
const failed = (error: unknown) => saidOf(error, `${(error as Thrown)?.name ?? "Error"}: ${(error as Thrown)?.message ?? error}`);
const unlessUpdated = (error: unknown): undefined => {
  if (isUpdated(error)) throw error;
};

// ---- the kernels, as the model's worker compiles them (src/runtime/worker/runtime.ts): the plain build, and the shared one where
// the page is cross-origin isolated. null where this browser cannot compile them (no WebAssembly SIMD)
async function kernelsOf(forward: typeof Forward, kind: "plain" | "shared"): Promise<Compiled | null> {
  const fetched = await Promise.all(([`simdkernel_${kind}.wasm`, `simdkernel_relaxed_${kind}.wasm`] satisfies Built[]).map((name) =>
    built(name).then((res) => res.arrayBuffer()).catch((error) => unlessUpdated(error) ?? null)));
  if (!fetched[0]) return null;
  try {
    return forward.compileKernels(fetched[0], fetched[1]);
  } catch {
    return null;
  }
}

async function device(): Promise<DeviceFound> {
  const forward = await named(import.meta.url, () => import("../forward.ts"));
  const plain = await kernelsOf(forward, "plain");
  const found: DeviceFound = {
    crossOriginIsolated: self.crossOriginIsolated,
    sharedMemory: typeof SharedArrayBuffer !== "undefined",
    simd: Boolean(plain),
    relaxedSimd: Boolean(plain?.relaxed),
    memory64: forward.memory64(),
    webgpu: Boolean(self.navigator?.gpu),
    opfs: Boolean(navigator.storage?.getDirectory),
    syncHandle: typeof FileSystemFileHandle !== "undefined" && typeof FileSystemFileHandle.prototype.createSyncAccessHandle === "function",
    waitAsync: typeof Atomics.waitAsync === "function",
    cores: navigator.hardwareConcurrency ?? null,
    memoryGB: navigator.deviceMemory ?? null,
  };
  try {
    const { usage, quota } = await navigator.storage.estimate();
    Object.assign(found, { usage, quota } satisfies Partial<DeviceFound>);
  } catch {
    // no navigator.storage here: said by opfs above
  }
  return found;
}

// ---- the CPU. A made-up model of Llama 3.2 1B's width (benchmark/madeup.ts). The forward pass is forward.ts's own
// (createForward, the kernels of simdkernel_*.wasm, the software threads of helper.js): what differs from the model
// page is only that the plan comes from there instead of from Python.
const TOKENS = 12, WARM = 3;
// T.108: a prompt's tokens go through the layers BATCH (16) at a time, without logits
const PROMPT = Array.from({ length: 16 }, (_, i) => 1 + i), PROMPT_RUNS = 5;

// random bytes for the int8 values, a scale that keeps the activations near 1, and norms of 1
function fillWeights(memory: WebAssembly.Memory, base: number, { plan }: Pick<Model, "plan">) {
  const U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const noise = new Uint8Array(1 << 20);
  for (let at = 0; at < noise.length; at += 65536) crypto.getRandomValues(noise.subarray(at, at + 65536));
  for (const t of Object.values(plan.tensors)) {
    const count = t.shape.reduce((a, b) => a * b, 1);
    if (t.kind === "int8") {
      for (let at = 0; at < count; at += noise.length) U.set(noise.subarray(0, Math.min(noise.length, count - at)), base + t.offset + at);
      F.fill(1 / (64 * Math.sqrt(t.shape.at(-1)!)), (base + t.scales) / 4, (base + t.scales) / 4 + count / GROUP);
    } else {
      F.fill(1, (base + t.offset) / 4, (base + t.offset) / 4 + count);
    }
  }
}

// a software thread of forward.ts, as the model's worker starts one (src/runtime/worker/weights.ts's spawnThread)
const spawn: Spawn = (data) => new Promise((resolve, reject) => {
  const worker = new Worker(new URL("../helper.ts", import.meta.url), { type: "module" });
  worker.onmessage = () => resolve({ terminate: () => worker.terminate() });
  worker.onerror = async (event) => reject(await unstarted(event, import.meta.url, event.message ?? "a software thread did not start"));
  worker.postMessage(data);
});

async function cpu(counts = [1, 2, 4]): Promise<CpuFound> {
  const forward = await named(import.meta.url, () => import("../forward.ts"));
  const shared = self.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined";
  const kernels = shared ? await kernelsOf(forward, "shared") : await kernelsOf(forward, "plain");
  if (!kernels) return { none: "no WebAssembly SIMD in this browser: the model page runs on NumPy here" };
  // T.177: making the model, then each count of software threads (one only without shared memory)
  const stages = 1 + counts.filter((asked) => shared || asked === 1).length;
  let done = 0;
  stage("making the model", ++done, stages);
  const model = madeUpModel();
  const after = forward.footprint(model.header, model.size, { dtype: "int8", relaxed: Boolean(kernels.relaxed), halfKV: true, shared });
  const { memory, base } = forward.weightsMemory(model.size, { shared, after });
  fillWeights(memory, base, model);
  const engine = forward.createForward({ memory, base, size: model.size, kernels, plan: model.plan, spawn: shared ? spawn : undefined });
  // T.163: the ceilings' loops on the same memory (no memory more: T.96); their failure leaves the forward pass alone
  let ceilings: Ceilings | null = null, ceilingsFailed: string | undefined;
  try {
    ceilings = await ceilingsOf(memory, base, model.size, shared);
  } catch (error) {
    unlessUpdated(error);
    ceilingsFailed = failed(error);
  }
  const rows: CpuRow[] = [];
  try {
    for (const asked of counts) {
      // without shared memory there are no software threads: one row, of one thread
      if (!shared && asked > 1) continue;
      stage(`${asked} software thread${asked > 1 ? "s" : ""}`, ++done, stages);
      const threads = await engine.setThreads(asked);
      if (threads !== asked) {
        rows.push({ asked, threads, none: "the browser did not start that many software threads" });
        continue;
      }
      for (let i = 0; i < WARM; i++) engine.forward(1, i, true);
      const times: number[] = [];
      for (let i = 0; i < TOKENS; i++) {
        const began = performance.now();
        engine.forward(1 + i, (WARM + i) % SHAPE.seqLen, true);
        times.push(performance.now() - began);
      }
      times.sort((a, b) => a - b);
      const ms = times[times.length >> 1];
      const logits = engine.logits();
      // a prompt of 16 tokens at positions 0 to 15, over and over (the keys and values of those positions are rewritten)
      engine.forwardMany(PROMPT, 0);
      const blocks: number[] = [];
      for (let i = 0; i < PROMPT_RUNS; i++) {
        const began = performance.now();
        engine.forwardMany(PROMPT, 0);
        blocks.push(performance.now() - began);
      }
      blocks.sort((a, b) => a - b);
      rows.push({ asked, threads, msPerToken: ms, GBps: model.size / (ms / 1000) / 1e9,
                  promptMsPerToken: blocks[blocks.length >> 1] / PROMPT.length, finite: logits.every(Number.isFinite) });
      // T.163: the ceilings right after the row they are held against (a phone that warms up slows both alike): reading
      // at the same count, and the compute loops after the row of one thread
      if (ceilings) ceilings.read.push({ threads, ...(await attempt(() => ceilings!.reading(threads))) });
      if (ceilings && threads === 1) {
        const none = { none: "no relaxed SIMD in this browser" };
        ceilings.dot = ceilings.relaxed ? await attempt(ceilings.dotting) : none;
        ceilings.dotRegisters = ceilings.relaxed ? await attempt(ceilings.dottingRegisters) : none;
        ceilings.fma = await attempt(ceilings.fmaing);
      }
    }
  } finally {
    engine.release();
    ceilings?.stop();
  }
  const { read, dot, dotRegisters, fma } = (ceilings ?? {}) as Partial<Ceilings>;
  return { backend: engine.backend, shared, megabytes: model.size / 1e6, tokenMegabytes: tokenBytes(model, kernels) / 1e6,
           layerWeights: layerWeights(model), rows, ceilings: ceilings ? { read, dot, dotRegisters, fma } : { error: ceilingsFailed } };
}

// what a token reads: the checkpoint, and with relaxed SIMD matmul_q8r's corrections too (an int32 for each group of
// every int8 matrix: 23.4 MB here). The GB/s column stays the checkpoint's (T.157 turns it into other models' speeds);
// its share of reading alone counts these bytes (T.163's review). Safari's matmul_q8 reads no corrections.
const tokenBytes = ({ plan, size }: Model, kernels: Compiled) => size + (kernels.relaxed ? Object.values(plan.tensors)
  .filter((t) => t.kind === "int8").reduce((sum, t) => sum + (t.shape.reduce((a, b) => a * b, 1) / t.group) * 4, 0) : 0);

// the weights of the layers' matrices: a multiply-add each for every token of a prompt (no classifier there)
const layerWeights = ({ plan }: Model) => (["wq", "wk", "wv", "wo", "w1", "w2", "w3"] satisfies MadeName[])
  .reduce((sum, name) => sum + plan.tensors[name].shape.reduce((a, b) => a * b, 1), 0);

// ---- T.163: the CPU's ceilings, loops of one kind of instruction (kernels/ceilings.ts, kernels/ceilings_relaxed.ts,
// the forms of T.158's audit): reading alone (GB/s) at each count of software threads, over the made-up model's weights
// (the bytes a token reads); relaxed_dot with its two loads from L1 and f32 multiply + add on registers (G MAC/s), one
// thread. Timed as the GPU section's ceilings are (T.168): passes doubled until one run takes PAIR_MS, then PAIRS runs of
// n and of 2n passes, the median of the differences; a ratio of 2n to n past STEADY takes the pairs once more, then is
// unsteady. The watch on a loop the compiler cut short: its checksum of n passes must be the one computed here, it
// must take PAIR_MS by MOST_PASSES, and 2n passes must take at least SHORT times n (a loop whose passes were dropped
// takes the same time for any count: T.163's review). cutShort: the section is then wrong, as for logits not finite.
const PAIR_MS = 20, PAIRS = 5, STEADY = [1.6, 2.2], SHORT = 1.3, MOST_PASSES = 2 ** 30;
const cutShort = (message: string) => new Failure("cut-short", message);
const middle = (values: number[]) => [...values].sort((a, b) => a - b)[values.length >> 1];
// run(n): the ms of n passes; right(n): whether the loop's checksum of n passes is the one computed here
function paired(run: (n: number) => number, right: (n: number) => boolean) {
  run(1);
  let n = 1, ms: number;
  while ((ms = run(n)) < PAIR_MS && n < MOST_PASSES) n *= 2;
  if (ms < PAIR_MS) throw cutShort(`${n} passes took ${ms.toFixed(3)} ms: the loop was compiled away`);
  if (!right(n)) throw cutShort(`the checksum of ${n} passes is not the loop's work: the compiler cut it short`);
  for (let tries = 0; ; tries++) {
    const differences: number[] = [], ratios: number[] = [];
    for (let i = 0; i < PAIRS; i++) {
      const once = run(n), twice = run(2 * n);
      differences.push(twice - once);
      ratios.push(twice / once);
    }
    const ratio = middle(ratios), took = middle(differences), steady = took > 0 && ratio >= STEADY[0] && ratio <= STEADY[1];
    if (steady) return { ms: took, passes: n, ratio };
    if (tries) {
      if (!(took > 0) || ratio < SHORT) throw cutShort(`${2 * n} passes took ${ratio.toFixed(2)} times as long as ${n}: the loop does not do its passes`);
      return { ms: took, passes: n, ratio, unsteady: true };
    }
  }
}
const attempt = async <Measured>(measure: () => Measured | Promise<Measured>): Promise<Attempted<Measured>> => {
  try {
    return await measure();
  } catch (error) {
    unlessUpdated(error);
    return { error: failed(error), ...(kindOf(error) === "cut-short" ? { cutShort: true } : {}) };
  }
};
const timed = (loop: (n: number) => unknown) => (n: number) => {
  const began = performance.now();
  loop(n);
  return performance.now() - began;
};
const rate = (work: number, r: ReturnType<typeof paired>): Rated => ({ rate: (work * r.passes) / (r.ms / 1000) / 1e9, passes: r.passes, ratio: r.ratio, ...(r.unsteady ? { unsteady: true } : {}) });
// the checksum of the int loops: the lane j of their sum is n times one pass's (lanes[j]), wrapped to 32 bits
const xorOf = (lanes: number[], n: number) => lanes.reduce((x, lane) => x ^ Math.imul(n, lane), 0);

// the control words of the reading threads, on a SharedArrayBuffer of their own (not a WebAssembly memory: T.96): as
// forward.ts cuts a matrix's rows (T.93), the weights are read a CHUNK at a time, each thread taking the next one from
// NEXT until TOTAL (passes × chunks), so that a slow core (a little one of a big.LITTLE phone) takes fewer instead of
// holding the others back (T.163's review: equal parts read at the slowest thread's speed)
const GO = 0, DONE = 1, COUNT = 2, TOTAL = 3, STOP = 4, NEXT = 5, CHUNK = 1 << 20, STARTING_MS = 30_000;

// The loops on the memory of the made-up model, its weights at base (size bytes): nothing new is reserved (T.96).
// dotting() makes 4 KB of them 7-bit for its loop (see ceilings_relaxed.ts) and puts them back after.
async function ceilingsOf(memory: WebAssembly.Memory, base: number, size: number, shared: boolean): Promise<Ceilings> {
  const kind = shared ? "shared" : "plain";
  const [plain, relaxed] = await Promise.all(([`ceilings_${kind}.wasm`, `ceilings_relaxed_${kind}.wasm`] satisfies Built[]).map((name) =>
    built(name).then((res) => res.arrayBuffer()).catch((error) => unlessUpdated(error) ?? null)));
  if (!plain) throw new Error("the ceilings' loops could not be fetched");
  const module = new WebAssembly.Module(plain);
  const loops = new WebAssembly.Instance(module, { env: { memory } }).exports as { read: Loop; fma: Loop };
  let dot: Loop | null = null, dotRegisters: Loop | null = null;
  try {
    if (relaxed) ({ dot, dotRegisters } = new WebAssembly.Instance(new WebAssembly.Module(relaxed), { env: { memory } }).exports as { dot: Loop; dotRegisters: Loop });
  } catch {
    // no relaxed SIMD here (Safari): the row says so
  }
  const chunks = Math.floor(size / CHUNK), bytes = chunks * CHUNK;
  const control = shared ? new Int32Array(new SharedArrayBuffer(8 * 4)) : null;
  const readers: Worker[] = [];
  let neverStarted: unknown;  // a reading thread that did not start: no count past it can be read
  const started = async (count: number) => {
    if (neverStarted && count > readers.length) throw neverStarted;
    while (readers.length < count - 1) {
      const worker = new Worker(new URL("../benchmark/reader.ts", import.meta.url), { type: "module" });
      readers.push(worker);  // stop() ends it even where it never answers
      let timer: ReturnType<typeof setTimeout> | undefined;
      await Promise.race([new Promise((resolve, reject) => {
        worker.onmessage = resolve;
        worker.onerror = async (event) => reject(await unstarted(event, import.meta.url, event.message ?? "a reading thread did not start"));
        worker.postMessage({ module, memory, control: control!.buffer, base, chunk: CHUNK, chunks, share: readers.length });
      }), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`a reading thread did not start in ${STARTING_MS / 1000} seconds`)), STARTING_MS);
      })]).catch((error) => {
        neverStarted = error;
        throw error;
      }).finally(() => clearTimeout(timer));
    }
  };
  // one pass's sums of the int32 lanes of the weights read, for the checksum of the reading loop
  const lanes = [0, 0, 0, 0];
  const ints = new Int32Array(memory.buffer, base, bytes / 4);
  for (let i = 0; i < ints.length; i++) lanes[i & 3] = (lanes[i & 3] + ints[i]) | 0;
  return {
    relaxed: Boolean(dot),
    read: [],
    // reading alone with count threads, CHUNKs taken in turn (this worker and the reading threads); the checksum is of
    // this worker's loop over all of them (the reading threads run the same loop)
    async reading(count: number) {
      await started(count);
      const r = paired((n) => {
        const began = performance.now();
        if (count === 1) {
          loops.read(base, bytes, n);
        } else {
          Atomics.store(control!, DONE, 0);
          Atomics.store(control!, NEXT, 0);
          Atomics.store(control!, TOTAL, n * chunks);
          Atomics.store(control!, COUNT, count);
          Atomics.add(control!, GO, 1);
          Atomics.notify(control!, GO);
          for (let k; (k = Atomics.add(control!, NEXT, 1)) < n * chunks;) loops.read(base + (k % chunks) * CHUNK, CHUNK, 1);
          for (let done; (done = Atomics.load(control!, DONE)) < count - 1;) Atomics.wait(control!, DONE, done);
        }
        return performance.now() - began;
      }, (n) => loops.read(base, bytes, n) === xorOf(lanes, n));
      const { rate: GBps, ...rest } = rate(bytes, r);
      return { GBps, ...rest };
    },
    dotting() {
      const w = new Int8Array(memory.buffer, base, 4096), x = new Uint8Array(memory.buffer, base + 4096, 4096);
      const kept = x.slice();
      try {
        for (let i = 0; i < x.length; i++) x[i] &= 0x7f;
        const sums = [0, 0, 0, 0];
        for (let i = 0; i < w.length; i++) sums[(i >> 2) & 3] += w[i] * x[i];
        const { rate: GMACs, ...rest } = rate(4096, paired(timed((n) => dot!(base, n)), (n) => dot!(base, n) === xorOf(sums, n)));
        return { GMACs, ...rest };
      } finally {
        x.set(kept);
      }
    },
    dottingRegisters() {
      const loop = (n: number) => dotRegisters!(n, 3, 5, 1);
      const { rate: GMACs, ...rest } = rate(128, paired(timed(loop), (n) => loop(n) === dotRegistersSum(n)));
      return { GMACs, ...rest };
    },
    fmaing() {
      const loop = (n: number) => loops.fma(n, 1, 0.9999, 1e-7);
      const { rate: GMACs, ...rest } = rate(32, paired(timed(loop), (n) => loop(n) === fmaSum(n)));
      return { GMACs, ...rest };
    },
    stop() {
      if (control) {
        Atomics.store(control, STOP, 1);
        Atomics.add(control, GO, 1);
        Atomics.notify(control, GO);
      }
      readers.forEach((worker) => worker.terminate());
    },
  };
}

// ceilings.ts's fma(n, 1, 0.9999, 1e-7) in float32, one lane (the four are alike) of the 8 accumulators, which start at
// 0 to 7, rounded as WebAssembly rounds each step
function fmaSum(n: number) {
  const f = Math.fround, d = f(1e-7), w = [1, 1.0009765625, 1.001953125, 1.0029296875];
  let x = f(0.9999), y = f(x + 0.25);
  const a = [0, 1, 2, 3, 4, 5, 6, 7];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 4; k++) {
      a[k] = f(a[k] + f(w[k] * x));
      a[4 + k] = f(a[4 + k] + f(w[k] * y));
    }
    x = f(x - d);
    y = f(y - d);
  }
  const lane = f(f(f(a[0] + a[1]) + f(a[2] + a[3])) + f(f(a[4] + a[5]) + f(a[6] + a[7])));  // as the loop sums them
  return f(f(f(lane + lane) + lane) + lane);
}

// ceilings_relaxed.ts's dotRegisters(n, 3, 5, 1): a lane of each dot adds 4 × a × 5, a running 3, 4, 5, … as int8
// (wrapping at 127); the 8 accumulators start at lanes 1000k + j, and the checksum sums the 4 lanes of all 8
function dotRegistersSum(n: number) {
  const int8 = (v: number) => ((v + 128) % 256 + 256) % 256 - 128;
  // the sum of a over n passes: whole turns of 256 values add -128 each
  let sum = Math.floor(n / 256) * -128;
  for (let i = 0; i < n % 256; i++) sum += int8(3 + i);
  const start = 4 * 28000 + 8 * 6;  // the accumulators' lanes: 1000k + j, k 0 to 7, j 0 to 3
  return (start + Math.imul(32 * 20, sum | 0)) | 0;
}

// ---- the line. A fetch read to its end, or read no faster than rate (MB/s): each piece is taken only when the bytes
// so far are due at that rate. Chromium and Firefox then stop taking from the connection, and the server's sending
// slows with it (the connection's flow control): what is paced is this page's download, not the whole line of the
// device. WebKit does not stop taking (T.134's review): there only this page's reading is paced.
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
// A fetch that brings nothing for STALL_MS is given up (as the model's worker gives up on Pyodide, T.118), and while
// bytes come the page hears of it once a second: its own deadline is for a section that says nothing at all.
const STALL_MS = 30_000;
// range: false for a part of the site's model, which GitHub Pages sends gzipped and answers a range of with a piece of
// the gzip stream (AGENTS.md): the whole part, read to its end
async function measure(url: string, { bytes, rate, range = true }: { bytes: number; rate?: number; range?: boolean }): Promise<LineFetched> {
  const began = performance.now();
  const abort = new AbortController();
  let stalled = false, told = began, timer: ReturnType<typeof setTimeout> | undefined;
  const alive = () => {
    clearTimeout(timer);
    timer = setTimeout(() => { stalled = true; abort.abort(); }, STALL_MS);
    if (performance.now() - told > 1000) {
      told = performance.now();
      postMessage({ alive: true } satisfies Alive);
    }
  };
  alive();
  let res: Response | undefined, headers: number | undefined, got = 0, first: number | undefined;
  try {
    res = await fetch(url, { headers: range ? { Range: `bytes=0-${bytes - 1}` } : {}, cache: "no-store", signal: abort.signal });
    if (!res.ok) throw new Error(`${url.split("?")[0]}: ${res.status}`);
    headers = performance.now() - began;
    const reader = res.body!.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      alive();
      if (done) break;
      first ??= performance.now();
      got += value.length;
      // the last piece waits too: WebKit takes the whole range whether it is read or not and hands it over in large
      // pieces, so there only the reading is paced, and a last piece taken at once made the rate look many times higher
      if (rate) {
        const due = first + (Math.min(got, bytes) / (rate * 1e6)) * 1000;
        if (due > performance.now()) await sleep(due - performance.now());
      }
      if (got >= bytes) {
        // a server that sends the whole file for a range (WebKit and Hugging Face's CDN, T.112) is read no further
        reader.cancel().catch(() => {});
        break;
      }
    }
  } catch (error) {
    throw stalled ? new Error("no bytes came for 30 s") : error;
  } finally {
    clearTimeout(timer);
  }
  const ended = performance.now();
  return { status: res!.status, headersMs: headers!, firstByteMs: (first ?? ended) - began, bytes: Math.min(got, bytes),
           MBps: Math.min(got, bytes) / 1e6 / ((ended - (first ?? began)) / 1000) };
}

async function line({ site, hf, rates = [], seconds = 4 }: Extract<ToSections, { step: "line" }>): Promise<LineFound> {
  const out: LineFound = { site: null, hf: null, paced: [] };
  const stages = 2 + rates.length;
  stage("this site: a part of the model", 1, stages);
  try {
    out.site = await measure(site.url, site);
  } catch (error) {
    out.site = { error: String((error as Error).message ?? error) };
  }
  stage("huggingface.co: a range of a model", 2, stages);
  try {
    out.hf = await measure(hf.url, { bytes: hf.bytes });
  } catch (error) {
    out.hf = { error: String((error as Error).message ?? error) };
    return out;
  }
  for (const [i, rate] of rates.entries()) {
    stage(`huggingface.co no faster than ${rate} MB/s`, 3 + i, stages);
    // a line slower than the rate asked for cannot be paced to it: said as such, not measured for ever
    if (out.hf!.MBps! < rate) {
      out.paced.push({ rate, slower: true });
      continue;
    }
    try {
      out.paced.push({ rate, ...(await measure(hf.url, { bytes: Math.round(rate * 1e6 * seconds), rate })) });
    } catch (error) {
      out.paced.push({ rate, error: String((error as Error).message ?? error) });
    }
  }
  return out;
}
