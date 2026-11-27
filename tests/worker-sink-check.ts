// T145, (7) and (8) of the review of T144 (2026-09-26): what lays out a checkpoint besides its header (its form,
// llama2_numpy.FORM: bias, arch, qk_norm, head_dim) goes from the converter's sink.open() through worker/weights.ts's
// weightsBuffer() to forward/memory.ts's footprint(), under the same names and with the same defaults. A name changed on
// one side only (head_dim, headDim) raises nothing: footprint() counts dim / heads, and a Qwen3 0.6B's keys and values
// come out 45% short (T124). Node only, with the native Python for FORM (numpy):
//
//   node tests/worker-sink-check.ts          (PYTHON=.venv/bin/python to take another Python)
//
// worker.js runs in a vm context with a few stand-ins (no Pyodide, no kernels, a memory of one page): its top level
// only declares, and its functions are what is called.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { workers as workersStandIn } from "./fakes/worker.ts";
import { python, runtimeUrl } from "./tree.ts";
const forward = await import(runtimeUrl("forward.js"));
const { STOPS_MOST } = await import(runtimeUrl("shaders.js"));  // (T681: the GPU sampling's, SAMPLE's)
// (T326: the header set to the context a load asks for; T281: and to fewer where the memory holds no more)
const { BEFORE, contextOf, contextRead, contextSet } = await import(runtimeUrl("kept.ts"));
import { runWorker } from "./worker-source.ts";
import { strictly } from "./worker-harness.ts";
import { contract, differing } from "./contracts.ts";

const root = new URL("..", import.meta.url);
const FORM = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c",
  "import json, sys; sys.path.insert(0, sys.argv[1]); import llama2_numpy; print(json.dumps(llama2_numpy.FORM))", python()],
{ cwd: fileURLToPath(root) }).toString());
// T229: "linear", the linear-attention layers of a Qwen3.5 (null where there are none); T237: "rotated", the basis the
// matrices are stored in (null: the model's own), whose signs and rotated inputs footprint() counts; T260: "convolution",
// the convolution layers of an LFM2 (null where there are none), which keep a few rows and no keys and values; T359:
// "kinds", the rows a file holds in a kind of their own (null where there are none; T363.5: of llama.cpp's blocks where
// a conversion kept a GGUF's, whose kernels' tables and activations footprint() counts, with Python's count of the
// weights that are still the dtype's: dtype_weights()); T363: "tiled", the order of the columns of a linear-attention
// layer's output matrix (null but in such a file: an array of a token's frame more, T363.5); T331: "wide" and "double", a
// Gemma 4's layers of wider heads and of an FFN twice as wide (null where there are none; footprint() reads them from
// T331.2, and until then no plan of a Gemma 4 is made: engine/plan.py); T330: "kv_shared", the layers that read another's
// keys and values (the same); T329: "per_layer", the width of every layer's input of its own table (0 where there is none)
assert.deepEqual(Object.keys(FORM).sort(), ["arch", "bias", "convolution", "double", "head_dim", "kinds", "kv_shared", "linear", "per_layer", "qk_norm", "rotated", "tiled", "wide"],
  "FORM has other keys now: say here which of them footprint() reads");

// T368.2: the type whatever is TypeScript reads a form by (src/runtime/contracts/form.ts) says the same keys, and the
// same architectures as Python's table of them (engine/layout.py's LAYOUTS); its two kinds of layers have the keys of
// Python's (plan-keys-check.ts holds those to the plan's)
{
  const typed = contract("form.ts");
  const archs = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c",
    "import json, sys; sys.path.insert(0, sys.argv[1]); from engine.layout import LAYOUTS; from engine.dtypes import DTYPES; print(json.dumps([list(LAYOUTS), list(DTYPES)]))", python()],
  { cwd: fileURLToPath(root) }).toString());
  const lines = [...differing("contracts/form.ts's Form, to Python's FORM", typed.keysOf("Form"), new Set(Object.keys(FORM))),
    ...differing("contracts/form.ts's Arch, to Python's LAYOUTS", typed.literalsOf("Arch"), new Set(archs[0])),
    ...differing("contracts/form.ts's Dtype, to Python's DTYPES", typed.literalsOf("Dtype"), new Set(archs[1]))];
  assert.deepEqual(lines, [], `${lines.join("; ")}: src/runtime/contracts/form.ts and src/python/engine/layout.py say different things`);
  console.log(`ok: contracts/form.ts's Form has FORM's ${Object.keys(FORM).length} keys, its Arch Python's ${archs[0].length} layouts, its Dtype Python's ${archs[1].length} dtypes`);
}

// (7) footprint()'s defaults are FORM's: a form without arch or head_dim (the options of a model converted before
// T124, a manifest) is counted as the engine reads it
const defaults = forward.footprint.toString();
assert.equal(/\barch = "(\w+)"/.exec(defaults)?.[1], FORM.arch, "footprint()'s default arch is not FORM's");
assert.equal(Number(/\bhead_dim = (\d+)/.exec(defaults)?.[1]), FORM.head_dim, "footprint()'s default head_dim is not FORM's");
const QWEN3 = [1024, 3072, 28, 16, 8, 151936, 4096];  // Qwen3 0.6B: heads of 128, not 1024 / 16
const GPT2 = [768, 3072, 12, 12, 12, 50257, 1024];
for (const header of [QWEN3, GPT2, [288, 768, 6, 6, 6, 32000, 256]]) {
  for (const dtype of ["int8", "float32"]) {
    assert.equal(forward.footprint(header, 600e6, { dtype }),
      forward.footprint(header, 600e6, { dtype, arch: FORM.arch, head_dim: FORM.head_dim }),
      `footprint() without a form is not footprint() with FORM's defaults (${header}, ${dtype})`);
  }
}

// (8) sink.open() -> weightsBuffer() -> footprint()
const context = vm.createContext({
  self: { navigator: {}, location: { search: "" }, crossOriginIsolated: false },
  console, performance, URL, TextDecoder, TextEncoder, setTimeout, clearTimeout, WebAssembly, Atomics, postMessage() {},
  // (this realm's, which the memories below are made in: worker.js asks whether a memory is shared with instanceof)
  SharedArrayBuffer,
});
runWorker(context);
const counted = [];
let destroyed = 0;
context.stand = {
  forward: {
    ...forward,
    footprint: (...args) => {
      counted.push(args);
      return forward.footprint(...args);
    },
    // a shared memory where one is asked for, unless the test refuses it (T130), or gives it at a lowered maximum (in pages: as
    // weightsMemory() marks a memory it made at its second or third try, the review of T130); what was made is written down
    weightsMemory: (size, { shared } = {}) => {
      (context.made ??= []).push(shared ? "shared" : "plain");
      if (shared && context.refuseShared) throw new Error("no shared memory here");
      if (!shared) return { memory: new WebAssembly.Memory({ initial: 1 }), base: 0 };
      const memory = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
      if (context.sharedMaximum !== undefined) Object.assign(memory, { maximum: context.sharedMaximum, limited: true });
      return { memory, base: 8192 };
    },
    growMemory() {},
  },
  // with ?without=kernels: the checkpoint in a Python bytearray
  pyodide: { globals: { get: () => () => ({ destroy: () => destroyed++, getBuffer: () => ({ data: new Uint8Array(8), release() {} }) }) } },
};
vm.runInContext(strictly("forwardModule = stand.forward; pyodide = stand.pyodide; jsKernels = { relaxed: true }; " +
  "llama2_numpy = { KV_START: 256, OUTLIER_CHANNELS: 8 }; roundBegan([]);"), context);
const proxy = (value) => ({ toJs: () => value, destroy() {} });
const opened = (header, form, dtype = "int8") => {
  counted.length = 0;
  const into = vm.runInContext(strictly("checkpointSink()"), context);
  into.sink.open(600e6, proxy(header), dtype, proxy(form));
  assert.equal(counted.length, 1, "sink.open() did not size the memory by footprint()");
  const [header2, size, options] = counted[0];
  assert.deepEqual([...header2], header);
  assert.equal(size, 600e6);
  assert.equal(options.dtype, dtype);
  for (const key of Object.keys(FORM)) assert.equal(options[key], form[key], `${key} of the form did not reach footprint()`);
  return into;
};
const qwen3 = { ...FORM, qk_norm: true, head_dim: 128 };
opened(QWEN3, qwen3);
// T363.5: a form with kinds: Python is asked how many weights the dtype itself still packs (with the header, the dtype
// and the whole form), and footprint() is handed its answer; a form without is not asked
{
  const asked = [];
  context.stand.weights = (header, dtype, options) => { asked.push([[...header], dtype, options.kinds]); return 4242; };
  vm.runInContext(strictly("llama2_numpy.dtype_weights = stand.weights"), context);
  const kinds = { w1: Array(QWEN3[2]).fill("IQ2_XXS"), wcls: "IQ3_S" };
  opened(QWEN3, { ...qwen3, kinds });
  // (once for footprint() and once for the type of the keys and values, which is asked the same way)
  assert.ok(asked.length > 0, "sink.open() did not ask Python for the weights of the dtype's own");
  for (const one of asked) assert.deepEqual(one, [QWEN3, "int8", kinds]);
  assert.equal(counted[0][2].weights, 4242, "Python's count did not reach footprint()");
  const times = asked.length;
  opened(QWEN3, qwen3);
  assert.ok(asked.length === times && !("weights" in counted[0][2]), "a form without kinds asks nothing");
  // what the test stands on: footprint() counts by them (the corrections of relaxed SIMD are of the dtype's own weights,
  // the blocks' kernels have tables, and a group of activations takes 16 bytes where it took 4)
  const plain = forward.footprint(QWEN3, 600e6, { ...qwen3, dtype: "int8" }), said = forward.footprint(QWEN3, 600e6, { ...qwen3, dtype: "int8", kinds, weights: 0 });
  assert.ok(forward.footprint(QWEN3, 600e6, { ...qwen3, dtype: "int8", kinds }) > plain + 34992 && said < plain - 600e6 / 9.5, "footprint() does not count a form's kinds");
  assert.ok(forward.takesBlocks(kinds) && !forward.takesBlocks(null) && !forward.takesBlocks({ wq: "int6", w1: ["int8", "float16"] }), "takesBlocks()");
}
// what the test stands on: the head's size changes what footprint() counts (a Qwen3 0.6B, T124)
assert.ok(forward.footprint(QWEN3, 600e6, { ...qwen3, dtype: "int8" }) > 1.5 * forward.footprint(QWEN3, 600e6, { ...FORM, dtype: "int8" }));
// T237: a rotated basis reaches footprint() as the options carry it (its block and the signs' texts by the width), and
// is counted: the signs of the three widths and a rotated input in each of the 16 frames
const rotated = { block: 1024, signs: { 1024: "00".repeat(128), 2048: "00".repeat(256), 3072: "00".repeat(384) } };
assert.ok(forward.footprint(QWEN3, 600e6, { ...qwen3, rotated, dtype: "int8" }) - forward.footprint(QWEN3, 600e6, { ...qwen3, dtype: "int8" })
  >= (1024 + 2048 + 3072) * 4 + 16 * 3072 * 4, "footprint() does not count a rotated basis");
// T260: an LFM2's convolution layers reach footprint() as the options carry them, and are counted: no keys and values
// for the 10 convolution layers of the 350M's 16 (2 kv dim floats a position a layer less), and their rows
const LFM2 = [1024, 4608, 16, 16, 8, 65536, 4096], lfm2 = { ...FORM, arch: "lfm2", convolution: { layers: "ccaccaccacacacac", taps: 3 } };
opened(LFM2, lfm2);
{
  const some = forward.footprint(LFM2, 600e6, { ...lfm2, dtype: "int8" });
  const all = forward.footprint(LFM2, 600e6, { ...lfm2, convolution: { layers: "a".repeat(16), taps: 3 }, dtype: "int8" });
  // (the keys and values of 10 layers more; the rows of 10 layers less, and, T375.5, the 16 frames' place for what a
  // convolution layer's matrix in gives, 3 dim floats, which a model with no such layer has not: nor has the engine)
  assert.equal(all - some, 10 * 4096 * 2 * 512 * 4 - 10 * 3 * 1024 * 4 - 16 * 3 * 1024 * 4, "footprint() does not count an LFM2's convolution layers");
}
opened(GPT2, { ...FORM, bias: true, arch: "gpt2" });
assert.notEqual(forward.footprint(GPT2, 600e6, { ...FORM, arch: "gpt2", dtype: "int8" }), forward.footprint(GPT2, 600e6, { ...FORM, dtype: "int8" }));

// T160 (the review): the type of the keys and values the worker sized the memory for (keysInHalf) is what it hands the
// engine (external's halfKeys). Without it createForward takes float32 for every grouped-query model: the same answer
// where float32 fits a 32-bit memory, and not for Llama 3.2 3B (64-bit either way: 0.46 GB more). No other test runs
// this path: forward-check and gpu-check make their engines themselves.
// T130: a shared memory asked for and refused. The plain one keeps float32 but where only float16 fits a 32-bit
// memory: llm-jp-3.1 1.8B float32 (3.66 GiB; the float16 of a shared memory would have left it 1.1 GiB short of
// counting it), sarashina2.2 3B in six bits float16 (4.36 GiB in float32). Before, the worker handed the engine the
// shared memory's answer and the engine took float32 on any plain memory: out of memory near the end of the context.
{
  const handed = [];
  context.stand.forward.external = (args) => {
    handed.push(args.halfKeys);
    return {};
  };
  vm.runInContext(strictly("roundBegan([]); sharedKernels = {}; wideKernels = { plain: {}, shared: {} }; threadsRequest = undefined; " +
    "llama2_numpy.Llama = { callKwargs: () => ({}) };"), context);
  const cases = [["llm-jp-3 150M", [512, 2048, 12, 8, 8, 99584, 4096], 160e6, {}, true],
    ["Qwen2.5 0.5B", [896, 4864, 24, 14, 2, 151936, 4096], 555992604, { bias: true }, false],
    ["Qwen2.5 3B", [2048, 11008, 36, 16, 2, 151936, 4096], 3472375836, { bias: true }, false],
    ["Llama 3.2 3B", [3072, 8192, 28, 24, 8, 128256, 4096], 3614847004, {}, true],
    ["llm-jp-3.1 1.8B", [2048, 7168, 24, 16, 16, -99584, 4096], 2101354524, {}, true],
    ["sarashina2.2 3B, six bits", [2560, 8960, 32, 16, 8, -102400, 4096], 2936678428, {}, true, "int6"],
    // T230: the ternary dtype is sized as int8 is but for its bytes: a grouped-query model's keys and values float32
    // (Ternary Bonsai 1.7B, 0.46 GB), float16 on a shared memory where the model is past 4 GiB (Ternary Bonsai 2 27B, 7.2 GB)
    ["Ternary Bonsai 1.7B, ternary", [2048, 6144, 28, 16, 8, 151936, 4096], 457490460, { qk_norm: true, head_dim: 128 }, false, "ternary"],
    ["Ternary Bonsai 2 27B, ternary", [5120, 17408, 64, 24, 4, -248320, 4096], 7242233884, { arch: "qwen35", head_dim: 256,
      linear: { every: 4, key_heads: 16, value_heads: 48, key_dim: 128, value_dim: 128, conv: 4 } }, true, "ternary"]];
  // on a plain memory: float32 but where float16 keeps on a 32-bit memory a model float32 would not (Llama 3.2 3B is
  // past it either way: float32, the owner, 2026-09-28)
  const plain = { "llm-jp-3 150M": false, "Qwen2.5 0.5B": false, "Qwen2.5 3B": false, "Llama 3.2 3B": false, "llm-jp-3.1 1.8B": false,
    "sarashina2.2 3B, six bits": true, "Ternary Bonsai 1.7B, ternary": false, "Ternary Bonsai 2 27B, ternary": false };
  const handedFor = (header, size, form, dtype) => {
    handed.length = 0;
    const into = vm.runInContext(strictly("checkpointSink()"), context);
    into.sink.open(size, proxy(header), dtype, proxy({ ...FORM, ...form }));
    into.weights.llama({}, {});
    assert.equal(handed.length, 1, "the engine was not made through external()");
    return handed[0];
  };
  // (T380.2: a shared memory that was refused is said on the console, once a memory that is made; nothing where none was)
  const warned = [];
  context.console = { ...console, warn: (line) => warned.push(line) };
  for (const [isolated, refused] of [[true, false], [false, false], [true, true]]) {
    context.self.crossOriginIsolated = isolated;
    context.refuseShared = refused;
    warned.length = 0;
    vm.runInContext(strictly("weightsPool = undefined"), context);  // a new memory for every case: the refusal is when one is made
    for (const [name, header, size, form, half, dtype = "int8"] of cases) {
      const want = isolated && !refused ? half : plain[name];
      const where = `${name}${isolated ? refused ? " (a shared memory refused)" : "" : " (not isolated)"}`;
      assert.equal(handedFor(header, size, form, dtype), want,
        `${where}: the worker hands the engine ${want ? "float16" : "float32"} keys and values`);
      // (T378.3: what timing.js asks of the memory: software threads can be had on a shared one alone)
      assert.equal(vm.runInContext(strictly("memoryShared()"), context), isolated && !refused, `${where}: whether the memory is shared`);
      vm.runInContext(strictly("weightsPool = undefined"), context);
      assert.equal(vm.runInContext(strictly("memoryShared()"), context), false, "no memory, and a shared one");
    }
    assert.deepEqual(warned, refused ? cases.map(() => "memory: no shared memory for this model (Error: no shared memory here): a memory that is not shared, and one thread") : [],
      `the console of ${isolated ? refused ? "a shared memory refused" : "a shared memory" : "a page that is not isolated"}`);
  }
  context.console = console;
  // (the review of T130) a shared memory the browser gave at a lowered maximum: where the forward pass does not fit it, the
  // worker makes a plain one instead (and hands the engine what a plain memory keeps); where it does, it keeps the shared one
  context.self.crossOriginIsolated = true;
  context.refuseShared = false;
  const GiB = 2 ** 30 / 65536;  // pages
  const said = context.console;
  context.console = { ...console, info() {} };  // (what the worker says of it)
  for (const [name, header, size, form, half, dtype = "int8"] of cases) {
    const need = Math.ceil((8192 + size + forward.footprint(header, size, { ...FORM, ...form, dtype, int8: true, relaxed: true, halfKV: true, shared: true, outliers: 8, gpu: false })) / 65536) + 1;
    for (const [maximum, shared] of [[need - 1, false], [need, true], [need + GiB, true]]) {
      context.sharedMaximum = maximum;
      context.made = [];
      vm.runInContext(strictly("weightsPool = undefined"), context);
      const want = shared ? half : plain[name], where = `${name}, a shared memory of ${maximum} pages where ${need} are needed`;
      assert.equal(handedFor(header, size, form, dtype), want, `${where}: the worker hands the engine ${want ? "float16" : "float32"} keys and values`);
      assert.deepEqual(context.made, shared ? ["shared"] : ["shared", "plain"], `${where}: ${shared ? "kept" : "a plain memory made instead"}`);
    }
  }
  context.console = said;
  context.sharedMaximum = undefined;
  context.self.crossOriginIsolated = false;
  context.refuseShared = false;
  vm.runInContext(strictly("sharedKernels = undefined; wideKernels = undefined; delete llama2_numpy.Llama;"), context);
  console.log("ok: the worker hands the engine the type of keys and values it sized the memory for");
}

// the Python buffer of ?without=kernels: let go once, by another open() (another tokenizer) or by release()
vm.runInContext(strictly('roundBegan(["kernels"]);'), context);
const into = vm.runInContext(strictly("checkpointSink()"), context);
assert.ok(vm.runInContext(strictly("weightsBytes()"), context) > 0, "the last model's memory of forward.js");
into.sink.open(600e6, proxy(QWEN3), "int8", proxy(qwen3));
// (T378.3: Python holds these weights: no memory of forward.js is the model's, which the page's heap would count twice)
assert.equal(vm.runInContext(strictly("weightsBytes()"), context), 0, "a model in a Python buffer still counts the last model's memory of forward.js");
assert.equal(destroyed, 0);
into.sink.open(600e6, proxy(QWEN3), "int8", proxy(qwen3));
assert.equal(destroyed, 1, "a second open() kept the first buffer");
into.release();
into.release();
assert.equal(destroyed, 2, "release() let go of the buffer not once");
// T503: a model of llama.cpp's blocks widened past Python's 32-bit memory is refused in the owner's words; a small one
// widened, and a large one that is no model of blocks, are not
vm.runInContext(strictly("blocksWidenedNow(true)"), context);
assert.throws(() => into.sink.open(5e9, proxy(QWEN3), "int8", proxy(qwen3)), /^Error: This model runs on the SIMD kernels only, which this browser \(or \?without=kernels\) leaves out\.$/);
into.sink.open(600e6, proxy(QWEN3), "int8", proxy(qwen3));
await vm.runInContext(strictly("loadBegan({ id: 'another' })"), context);
into.sink.open(5e9, proxy(QWEN3), "int8", proxy(qwen3));  // (a load after it, of no blocks: the word is not carried over)
into.release();
console.log("ok: FORM's keys and defaults reach footprint() from sink.open()");
// T281: a context asked past what the memory of a page holds (?context=): the header the converter hands sink.open() is
// placed at the longest context that fits (weights.ts's contextFitted(): here a 32-bit memory's, no 64-bit kernels), the
// memory sized for that, and its seq_len written as that as it passes; the converter's own bytes stay as they were
{
  vm.runInContext(strictly("roundBegan([]); blocksWidenedNow(false)"), context);
  const long = [...QWEN3.slice(0, 6), 262144];
  counted.length = 0;
  const into = vm.runInContext(strictly("checkpointSink"), context)(undefined, {}, 262144);
  into.sink.open(600e6, proxy(long), "int8", proxy(qwen3));
  const placed = counted[0][0][6];
  assert.ok(placed > 4096 && placed < 262144, `placed at ${placed} positions`);
  assert.ok(counted.every(([header]) => header[6] === placed), "the memory sized at that context, every time it was asked");
  const options = counted[0][2], fits = (at) => !forward.needsWide(600e6, forward.footprint([...long.slice(0, 6), at], 600e6, options));
  assert.ok(fits(placed) && !fits(placed + 1), "the longest that fits a 32-bit memory");
  const head = new Uint8Array(new Int32Array([...long, 77]).buffer);
  for (const [at, end] of [[0, 20], [20, 26], [26, 32]]) into.sink.write(at, { getBuffer: () => ({ data: head.subarray(at, end), release() {} }) });
  assert.deepEqual([...new Int32Array(vm.runInContext("weightsNow", context).buffer, 0, 8)], [...long.slice(0, 6), placed, 77], "its seq_len written as that, wherever the writes cut it");
  assert.deepEqual([...new Int32Array(head.buffer)], [...long, 77], "the converter's bytes stay its own");
  // and a context that fits is placed as it is, and the header written as the converter wrote it
  counted.length = 0;
  const fitting = vm.runInContext(strictly("checkpointSink"), context)(undefined, {}, 8192);
  fitting.sink.open(600e6, proxy([...QWEN3.slice(0, 6), 8192]), "int8", proxy(qwen3));
  assert.equal(counted[0][0][6], 8192);
  assert.equal(fitting.context, undefined);
}
console.log("ok: a context past what the memory holds is cut to the longest that fits, and the header written so as it passes (T281)");

// T329.2.2: a Gemma 4's table of every layer's inputs stays in the file the checkpoint is kept in. On the first visit
// sink.open() makes the memory without it (the bytes after it as much lower), every byte goes to the file as it comes,
// and the engine reads the table's rows from that file (the keeper); on the next, a kept one's file read from once it is
// opened (attach()). The model is let go of with its file; where the table was not all written, it is loaded again
// with the table in the memory, as it is where the file is kept as another dtype, or where its file cannot be opened
{
  const ask = (code) => vm.runInContext(strictly(code), context);
  ask("roundBegan([])");
  const GEMMA = [64, 128, 10, 4, 1, 100, 64], size = 40000, hole = [10000, 28000];
  const form = { ...FORM, arch: "gemma4", per_layer: 16 };
  // (the table as external_tensors() places it: 100 rows of 160 int8 values, then their scales, 20 bytes a row)
  const tensors = { per_layer_table: { kind: "int8", offset: 10000, shape: [100, 160], group: 32, scales: 26000 } };
  const bytes = Uint8Array.from({ length: size }, (_, i) => (i * 13 + 7) & 255);
  let handed;
  context.stand.forward = { ...context.stand.forward, external: (args) => { handed = args; return {}; } };
  context.stand.python = { external_tensors: () => ({ toJs: () => tensors, destroy() {} }), Llama: { callKwargs: () => ({}) } };
  ask("forwardModule = stand.forward; llama2_numpy.external_tensors = stand.python.external_tensors; llama2_numpy.Llama = stand.python.Llama");
  const keeperOf = (dtype = "int8", whole = true) => {
    const file = new Uint8Array(size), keep = { dtype, held: null, closed: 0, file,
      holding(stretch) { keep.held = stretch; }, write(offset, chunk) { file.set(chunk, offset); },
      read(into, at) { into.set(file.subarray(at, at + into.length)); return into.length; }, whole: () => whole, close() { keep.closed++; } };
    return keep;
  };
  const sunk = (keep, dtype = "int8", given = form) => {
    const into = ask("checkpointSink")(keep);
    into.sink.open(size, proxy(GEMMA), dtype, proxy(given));
    for (let at = 0; at < size; at += 7000) {
      const data = bytes.subarray(at, Math.min(at + 7000, size));
      into.sink.write(at, { getBuffer: () => ({ data, release() {} }) });
    }
    return into;
  };
  const inMemory = (length) => new Uint8Array(ask("weightsNow").buffer, 0, length);
  const without = Uint8Array.from([...bytes.subarray(0, hole[0]), ...bytes.subarray(hole[1])]);
  await ask("loadBegan({ id: 'gemma' })");
  // the first visit
  const keep = keeperOf(), into = sunk(keep);
  assert.deepEqual(into.weights.hole, hole);
  assert.deepEqual(keep.held, hole, "the keeper was told the stretch it holds");
  assert.deepEqual(keep.file, bytes, "every byte to the file as it came");
  assert.deepEqual(inMemory(without.length), without, "the memory holds the checkpoint without the table, the rest moved down");
  into.weights.llama({}, {});
  const { apart } = handed;
  assert.equal(handed.oneThread, false, "the engine is told no one thread was asked for (T371)");
  assert.deepEqual([apart.stored, apart.place(5), apart.place(hole[1]), apart.place(size - 1)], [size - (hole[1] - hole[0]), 5, hole[0], size - 1 - (hole[1] - hole[0])]);
  apart.rows(10000 + 3 * 160, 160, 30000);
  assert.deepEqual(new Uint8Array(ask("weightsNow").buffer, 30000, 160), bytes.subarray(10480, 10640), "a row read from the file into the memory");
  assert.equal(ask("rowsReady")({ id: "gemma" }), true);
  await ask("rowsLetGo")();
  await ask("rowsLetGo")();
  assert.equal(keep.closed, 1, "the file let go of with the model, once");
  // a table not all written: loaded again, with the table in the memory from then on (and its file let go of)
  const short = keeperOf("int8", false), again = sunk(short);
  again.weights.llama({}, {});
  assert.equal(ask("rowsReady")({ id: "gemma" }), false);
  await ask("rowsLetGo")();
  assert.equal(short.closed, 1);
  const third = sunk(keeperOf());
  assert.equal(third.weights.hole, null, "a model whose table was not all kept keeps it in the memory, this visit");
  assert.deepEqual(inMemory(size), bytes);
  // the next visit: a kept one, read whole but the table, and its file attached before the engine is made
  await ask("loadBegan({ id: 'gemma, kept' })");
  const kept = ask("weightsBuffer")(size, GEMMA, { ...form, dtype: "int8" }, undefined, true);
  assert.deepEqual(kept.hole, hole);
  for (let at = 0; at < size; at += 7000) kept.write(at, bytes.slice(at, Math.min(at + 7000, size)));
  assert.deepEqual(inMemory(without.length), without);
  const reader = keeperOf();
  reader.file.set(bytes);
  kept.attach(reader);
  kept.llama({}, {});
  handed.apart.rows(10000, 160, 30000);
  assert.deepEqual(new Uint8Array(ask("weightsNow").buffer, 30000, 160), bytes.subarray(10000, 10160));
  assert.equal(ask("rowsReady")({ id: "gemma, kept" }), true, "a kept file is whole");
  await ask("rowsLetGo")();
  assert.equal(reader.closed, 1);
  // (T521: the review of T329.2.2) and loadConverted() asks the kept parts to skip the table once the first part's
  // header says where it is (a kept file read whole is 2.6 GB too many on a real Gemma 4, and the same model after)
  const first = new Uint8Array(7000);
  first.set(new Uint8Array(new Int32Array(GEMMA).buffer));
  const skipped = [];
  context.stand.kept = { BEFORE, contextOf, contextRead, contextSet, openKept: async () => ({ manifest: { bytes: size, options: { ...form, dtype: "int8" } }, where: "opfs",
    async *parts(skipping) {
      skipped.push(skipping.range);
      yield first;
      skipped.push(skipping.range && [...skipping.range]);
      throw new Error("no more parts here");
    } }) };
  ask("keptModule = stand.kept");
  const missed = await ask("loadConverted")({ id: "gemma, kept" }, new AbortController().signal, 1);
  assert.match(missed.miss, /no more parts here/);
  assert.deepEqual(skipped, [undefined, hole], "the table skipped once the header said where it is");
  await ask("rowsLetGo")();
  // the table in the memory: a file kept as another dtype than the model is converted to, a model without the table, ?without=kernels
  for (const [keeping, dtype, given] of [[keeperOf("int8"), "int6", form], [keeperOf(), "int8", { ...form, per_layer: 0 }]]) {
    const plain = sunk(keeping, dtype, given);
    assert.equal(plain.weights.hole, null);
    assert.equal(keeping.held, null);
    assert.ok(keeping.file.every((byte) => byte === 0), "nothing to a file the table does not stay in");
  }
  ask('roundBegan(["kernels"])');
  assert.equal(ask("weightsBuffer")(size, GEMMA, { ...form, dtype: "int8" }, keeperOf(), true).hole, null);
  ask("roundBegan([])");
  // (T371) the engine is told whether the page asked for one thread, which is why an isolated page's memory is not shared
  ask("threadsRequest = { fixed: 1 }");
  sunk(keeperOf()).weights.llama({}, {});
  assert.equal(handed.oneThread, true, "?threads=1 told to the engine");
  ask("threadsRequest = undefined");
  await ask("rowsLetGo")();
  console.log("ok: a Gemma 4's table of every layer's inputs stays in the file it is kept in as it comes (or was kept in), the rest in the memory; " +
    "its rows are read from there, the file let go of with the model; where the table was not all written it is loaded again with the table in the memory");
}

// The review of T156: a model on the GPU alone. Its GPU's worker opens before a byte comes, and makes the device and a
// buffer for every layer's matrices then (4.1 GB of Llama 3.2 3B). A load let go before its engine was built (cancelled,
// failed, another try of the converter) must stop that worker, and the next load wait for its "ended": on the branch
// before the review nothing stopped it, and it held them for the rest of the visit. A built one whose GPU failed
// (gpuOnlyReady) is let go before its load on the CPU begins, whether forward.js had started its GPU or not (T205).
// worker.js runs in a context of its own here, with a WebGPU adapter as the owner's Android has (8 GB said, 256 MiB bound).
{
  const LLAMA3B = [3072, 8192, 28, 24, 8, 128256, 4096];
  const [dim, hidden, layers, heads, kvHeads, vocab] = LLAMA3B, kvDim = (kvHeads * dim) / heads;
  // its tensors in file order, int8, as llama2_numpy.external_tensors() places them
  const tensors = {};
  let end = 28;
  for (const [name, shape, int8] of [["token_embedding_table", [vocab, dim], true], ["rms_att_weight", [layers, dim], false],
    ["wq", [layers, dim, dim], true], ["wk", [layers, kvDim, dim], true], ["wv", [layers, kvDim, dim], true],
    ["wo", [layers, dim, dim], true], ["rms_ffn_weight", [layers, dim], false], ["w1", [layers, hidden, dim], true],
    ["w2", [layers, dim, hidden], true], ["w3", [layers, hidden, dim], true], ["rms_final_weight", [dim], false]]) {
    const count = shape.reduce((a, b) => a * b, 1);
    tensors[name] = { kind: int8 ? "int8" : "f32", offset: end, shape, group: int8 ? 32 : 0, scales: int8 ? end + count : 0 };
    end += int8 ? count + count / 8 : count * 4;
  }
  const size = end;
  // the GPU's worker (gpu.js; tests/fakes/worker.ts): what it was told, and "ended" a moment after a stop, as its end()
  // says it. (T377) worker.fail({ message }): the browser says the worker's script did not arrive, or that it threw
  const { Worker, made: workers } = workersStandIn({ answer(data, worker) {
    if (data.type === "stop") setTimeout(() => worker.say({ type: "ended" }), 5);
  } });
  const typesPosted = (worker) => worker.posted.map(({ type }) => type);
  const placedForms = [];
  let placing = tensors;  // (T682: the tensors the stand-in of Python places, a Gemma 4's below)
  const android = { deviceMemory: 8, gpu: { requestAdapter: async () => ({ info: { isFallbackAdapter: false },
    limits: { maxStorageBufferBindingSize: 2 ** 28, maxBufferSize: 2 ** 28, minStorageBufferOffsetAlignment: 256 } }) } };
  const gpuContext = vm.createContext({
    self: { navigator: android, location: { search: "" }, crossOriginIsolated: true }, navigator: android,
    console, performance, URL, TextDecoder, TextEncoder, setTimeout, clearTimeout, WebAssembly, Atomics, SharedArrayBuffer, Worker,
    postMessage() {},
  });
  runWorker(gpuContext);
  let engineGpu;
  gpuContext.stand = {
    forward: {
      ...forward,
      weightsMemory: () => ({ memory: new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true }), base: 64 }),
      growMemory() {},
      external: () => ({ engine: { gpu: engineGpu } }),
    },
    // (T571: the form it is asked with is written down: where the tensors are depends on all of it)
    llama2_numpy: { KV_START: 256, OUTLIER_CHANNELS: 8, external_tensors: (header, dtype, form) => { placedForms.push(form); return { toJs: () => placing, destroy() {} }; },
      Llama: { callKwargs: () => ({}) } },
  };
  vm.runInContext(strictly("forwardModule = stand.forward; llama2_numpy = stand.llama2_numpy; jsKernels = { relaxed: true }; " +
    "sharedKernels = {}; wideKernels = { plain: {}, shared: {} }; roundBegan([]);"), gpuContext);
  // (T641.8: asked for by init(), which this check does not run: asked here as init() asks)
  await vm.runInContext(strictly("adapterAsked = askAdapter()"), gpuContext);
  const opened = () => {
    const into = vm.runInContext(strictly("checkpointSink()"), gpuContext);
    into.sink.open(size, proxy(LLAMA3B), "int8", proxy(FORM));
    assert.ok(into.weights.direct, "Llama 3.2 3B on a device that says 8 GB did not go on the GPU alone");
    // T571: where its tensors are is asked of the whole form (FORM's every key, as the options carry it: a Qwen3.5's linear
    // too). Before, four keys of it went, and Python refused a Qwen3.5 on the GPU alone: "it has linear that keep a state,
    // and the form says none" (models.yml run 40051864570)
    const asked = placedForms.at(-1);
    const keys = new Set(Object.keys(asked?.toJs?.() ?? asked ?? {}));
    assert.deepEqual(Object.keys(FORM).filter((key) => !keys.has(key)), [], "the form where the tensors of a model on the GPU alone are asked: every key of FORM");
    const worker = workers.at(-1);
    assert.deepEqual(typesPosted(worker), ["open"]);
    return { into, worker };
  };
  // (1) let go before its engine was built: the worker is stopped, and the next load waits for its "ended"
  {
    const { into, worker } = opened();
    into.release();
    assert.deepEqual(typesPosted(worker), ["open", "stop"], "a model on the GPU alone let go before it was built kept its GPU's worker");
    assert.ok(!worker.terminated);
    await vm.runInContext(strictly("loadBegan({ id: 'the next load' })"), gpuContext);  // (T378.3: what load() says first)
    assert.ok(worker.terminated, "the next load did not wait for the GPU's worker to end");
  }
  // (2) another try of the converter (sink.open() again) lets go of the first one's worker
  {
    const { into, worker } = opened();
    into.sink.open(size, proxy(LLAMA3B), "int8", proxy(FORM));
    assert.deepEqual(typesPosted(worker), ["open", "stop"], "a second sink.open() kept the first GPU's worker");
    into.release();
    await vm.runInContext(strictly("gpuOnlyEnding"), gpuContext);
  }
  // T681: the converter's stop tokens come with the form (convert/stream.py's said): more than the GPU's sampling holds
  // keep the model off the GPU alone before a byte goes there, and the entry's own stop tokens win over them, as they do
  // where the engine is made
  {
    const many = { ...FORM, stop_tokens: Array.from({ length: STOPS_MOST + 1 }, (_, i) => i) };
    const into = vm.runInContext(strictly("checkpointSink()"), gpuContext);
    into.sink.open(size, proxy(LLAMA3B), "int8", proxy(many));
    assert.ok(!into.weights.direct, "T681: a model of more stop tokens than the GPU's sampling holds went on the GPU alone");
    into.release();
    gpuContext.own = { options: { stop_tokens: [1, 2] } };
    const own = vm.runInContext(strictly("checkpointSink(undefined, own)"), gpuContext);
    own.sink.open(size, proxy(LLAMA3B), "int8", proxy(many));
    assert.ok(own.weights.direct, "T681: the entry's own stop tokens did not win over the converter's");
    own.release();
    await vm.runInContext(strictly("loadBegan({ id: 'the next load' })"), gpuContext);
    await vm.runInContext(strictly("gpuOnlyEnding"), gpuContext);
  }
  // T682: a Gemma 4 on a device that says 8 GB, with no test asking for the GPU alone (the owner's choice, 2026-11-26: no
  // line of its own keeps a table of every layer's inputs off it). The place is weightsPlace()'s as for any model, of the
  // bytes the memory holds: where the table stays where the model is kept (the file kept as it comes on the first visit,
  // the kept one attached on the next), the memory holds the rest alone, and the E2B's 2.5 GB with the GPU's 2.6 GB fit
  // twice (both, as before); the E4B's do not, and it goes on the GPU alone, the engine reading the table's rows from that
  // file (direct.rows). Where nothing keeps it (no file: a visit with no origin private file system, T525), the table is
  // in the memory with the rest: the E2B does not fit twice then and goes on the GPU alone too, the engine reading the
  // table's rows in the memory (no direct.rows), which holds the table and the rest, less than the CPU's whole model.
  // (The tensors are Python's for the forms of google/gemma-4-E2B-it's and E4B-it's config.json, int8, a context of 4096)
  {
    const E2B = [1536, 6144, 35, 8, 1, 262144, 4096], E4B = [2560, 10240, 42, 8, 2, 262144, 4096];
    const formOf = (wide, shared, from, double) => ({ ...FORM, arch: "gemma4", head_dim: 256, wide: { layers: wide, head_dim: 512 },
      kv_shared: { layers: shared, from: shared.map((l) => (wide.includes(l) ? from[1] : from[0])) }, double, per_layer: 256 });
    const e2bShared = Array.from({ length: 20 }, (_, i) => 15 + i), e4bShared = Array.from({ length: 18 }, (_, i) => 24 + i);
    const models = {
      E2B: { header: E2B, form: formOf([4, 9, 14, 19, 24, 29, 34], e2bShared, [13, 14], e2bShared) },
      E4B: { header: E4B, form: formOf([5, 11, 17, 23, 29, 35, 41], e4bShared, [22, 23], null) },
    };
    for (const model of Object.values(models)) {
      model.placed = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c",
        "import json, sys; sys.path.insert(0, sys.argv[1]); import llama2_numpy; " +
        "print(json.dumps(llama2_numpy.external_tensors(json.loads(sys.argv[2]), 'int8', json.loads(sys.argv[3]))))", python(), JSON.stringify(model.header), JSON.stringify(model.form)],
      { cwd: fileURLToPath(root) }).toString());
      // (past the last tensor's start, a vector of floats: its length is not asked of here)
      model.size = Math.max(...Object.values(model.placed).flatMap((t) => [t.offset, t.scales || 0])) + 2 ** 20;
      model.hole = forward.tableHole(model.placed);
      model.without = (stretches) => model.size - stretches.reduce((sum, [a, b]) => sum + b - a, 0);
      model.holes = forward.gpuHoles(model.placed);
      // (T682's review: what a token reads of it, the CPU's side of aloneVerdict(): all but the table, of which it reads a row)
      model.reads = model.size - (model.hole[1] - model.hole[0]);
    }
    const keeperOf = () => {
      const keep = { dtype: "int8", held: null, closed: 0, holding(stretch) { keep.held = stretch; }, write() {}, read: (into) => into.length,
        whole: () => true, close() { keep.closed++; } };
      return keep;
    };
    const ask = (code) => vm.runInContext(strictly(code), gpuContext);
    const opened = async (name, keep) => {
      const { header, form, size, placed } = models[name];
      placing = placed;
      await ask(`loadBegan({ id: "${name}" })`);
      gpuContext.keep = keep;
      const into = ask("checkpointSink(keep)");
      into.sink.open(size, proxy(header), "int8", proxy(form));
      return into.weights;
    };
    // the engine built on a model on the GPU alone: what reads its table's rows (undefined: the memory, where they are)
    const built = async (weights) => {
      weights.llama({}, {});
      const { rows } = weights.direct;
      await weights.direct.end();
      await ask("rowsLetGo")();
      return rows;
    };
    // the E2B, its table in the file kept as it comes: held twice, as before
    {
      const keep = keeperOf(), weights = await opened("E2B", keep);
      assert.deepEqual(weights.hole, models.E2B.hole, "the E2B's table stays in the file kept as it comes");
      assert.ok(!weights.direct, "the E2B went on the GPU alone where both copies fit, its table in the kept file");
      weights.destroy();
    }
    // the E4B on the first visit: on the GPU alone, its table in the file kept as it comes
    {
      const keep = keeperOf(), weights = await opened("E4B", keep);
      assert.ok(weights.direct, "a Gemma 4 E4B on a device that says 8 GB did not go on the GPU alone");
      assert.deepEqual([weights.hole, keep.held], [models.E4B.hole, models.E4B.hole], "the E4B's table stays in the file kept as it comes");
      assert.equal(weights.direct.stored, models.E4B.without([...models.E4B.holes, models.E4B.hole]), "the memory holds neither the GPU's matrices nor the table");
      assert.equal(weights.direct.size, models.E4B.reads, "the CPU's token is reckoned to read all of the checkpoint but the table");
      assert.equal(typeof await built(weights), "function", "the engine reads the table's rows from the file kept as it came");
      assert.equal(keep.closed, 1, "the file let go of with the model");
    }
    // the next visit: the kept E4B, its file attached before the engine is built
    {
      await ask('loadBegan({ id: "E4B" })');
      placing = models.E4B.placed;
      gpuContext.form = { ...models.E4B.form, dtype: "int8" };
      const weights = ask("weightsBuffer")(models.E4B.size, E4B, gpuContext.form, undefined, true);
      assert.ok(weights.direct, "a kept Gemma 4 E4B did not go on the GPU alone");
      assert.deepEqual(weights.hole, models.E4B.hole, "a kept E4B on the GPU alone reads its table from the kept file");
      assert.equal(weights.direct.stored, models.E4B.without([...models.E4B.holes, models.E4B.hole]));
      const reader = keeperOf();
      weights.attach(reader);
      assert.equal(typeof await built(weights), "function");
      assert.equal(reader.closed, 1, "the kept file let go of with the model");
    }
    // nothing keeps them: the table in the memory, and both on the GPU alone
    for (const name of ["E2B", "E4B"]) {
      // (T682's review: the console says the numbers the place was weighed by, once, as where the CPU takes the model)
      const was = gpuContext.console, infos = [];
      gpuContext.console = { ...console, info: (line) => infos.push(line) };
      const weights = await opened(name, undefined);
      gpuContext.console = was;
      assert.deepEqual(infos.filter((line) => line.startsWith("gpu: both copies would not fit")).map((line) => line.replace(/\d+ MB/g, "N MB")),
        ["gpu: both copies would not fit: the CPU's N MB and the GPU's N MB, in N MB of a device that says 8 GB: on the GPU alone"], `the console of the ${name} on the GPU alone`);
      assert.ok(weights.direct, `a Gemma 4 ${name} whose table nothing keeps did not go on the GPU alone`);
      assert.equal(weights.hole, null);
      assert.equal(weights.direct.stored, models[name].without(models[name].holes), "where nothing keeps it, the memory holds the table with the rest");
      assert.equal(weights.direct.size, models[name].reads, "and the CPU's token is reckoned to read none of the table but a row, there too");
      assert.equal(await built(weights), undefined, "the engine reads the table's rows in the memory");
    }
    // (a device that says 4 GB keeps them off the GPU alone, as before: the GPU's copy alone is past half of it)
    android.deviceMemory = 4;
    for (const name of ["E2B", "E4B"]) {
      const weights = await opened(name, undefined);
      assert.ok(!weights.direct, `a Gemma 4 ${name} went on the GPU alone on a device that says 4 GB`);
      weights.destroy();
    }
    android.deviceMemory = 8;
    placing = tensors;
    await ask("loadBegan({ id: 'the next load' })");
    await ask("gpuOnlyEnding");
    console.log("ok: T682, a Gemma 4 on a device that says 8 GB: the E4B on the GPU alone, its table read from the file it is kept in as it comes or was kept in; " +
      "the E2B held twice where its table is kept so, on the GPU alone where nothing keeps it; both, their tables in the memory then; neither on one that says 4 GB");
  }
  // (3) built, its GPU ready: kept (destroy() lets nothing go); then its GPU fails: let go before the load on the CPU
  for (const lost of [null, "the GPU was lost (a test)"]) {
    const { into, worker } = opened();
    engineGpu = Promise.resolve("prompts and answers on WebGPU");
    gpuContext.made = into.weights.llama({}, {});
    into.release();
    assert.deepEqual(typesPosted(worker), ["open"], "a built model on the GPU alone lost its GPU's worker");
    gpuContext.lost = lost;
    vm.runInContext(strictly("gpuOnlyNow.lost = lost"), gpuContext);
    const ready = await vm.runInContext(strictly("gpuOnlyReady({ id: 'probe' }, undefined, made.outside.engine)"), gpuContext);
    assert.equal(ready, !lost);
    assert.equal(vm.runInContext(strictly("cpuOnly.has('probe')"), gpuContext), Boolean(lost));
    assert.deepEqual(typesPosted(worker), lost ? ["open", "stop"] : ["open"]);
    assert.equal(worker.terminated, Boolean(lost), "the load on the CPU began before the GPU's worker ended");
  }
  // (4) the second review of T156: the verdict that the CPU is faster goes to the page with its load's id (the page
  // reads no word of a load another choice cancelled: without it a late verdict was kept for the model chosen since),
  // and a verdict the page kept for this device and /benchmark/'s reading keeps the model off the GPU alone, and says so
  {
    const { into } = opened();
    engineGpu = Promise.resolve("prompts and answers on the CPU");
    gpuContext.made = into.weights.llama({}, {});
    into.release();
    const said = [], told = [];
    gpuContext.postMessage = (data) => said.push(JSON.parse(JSON.stringify(data)));
    gpuContext.console = { ...console, info: (line) => told.push(line) };
    gpuContext.verdict = { key: "arm|valhall||Mali-G615|a browser|0", cpu: { GBps: 28.7, threads: 4, promptGMACs: 40 } };
    vm.runInContext(strictly("gpuOnlyNow.lost = 'the CPU as /benchmark/ measured it'; gpuOnlyNow.verdict = verdict"), gpuContext);
    assert.equal(await vm.runInContext(strictly("gpuOnlyReady({ id: 'kept' }, 7, made.outside.engine)"), gpuContext), false);
    assert.deepEqual(said.filter((data) => data.type === "gpu-alone"), [{ type: "gpu-alone", load: 7, alone: gpuContext.verdict }],
      "the verdict went to the page without its load's id");
    const placed = (cpu) => {
      gpuContext.cpu = cpu;
      vm.runInContext(strictly("gpuAdapter.keys = { plain: verdict.key, ternary: \"\", blocks: \"\" }; gpuRequest = { remembered: { alone: verdict }, cpu }"), gpuContext);
      const sunk = vm.runInContext(strictly("checkpointSink()"), gpuContext);
      sunk.sink.open(size, proxy(LLAMA3B), "int8", proxy(FORM));
      const direct = Boolean(sunk.weights.direct);
      sunk.release();
      return direct;
    };
    assert.equal(placed({ GBps: 28.7, threads: 4, promptGMACs: 40 }), false, "a verdict the page kept left the model on the GPU alone");
    assert.ok(told.some((line) => /as the page kept it/.test(line)), "the console did not say the verdict it kept");
    assert.equal(placed({ GBps: 30.1, threads: 4, promptGMACs: 40 }), true, "/benchmark/'s CPU measured again did not weigh the two again");
    await vm.runInContext(strictly("gpuOnlyEnding"), gpuContext);
    gpuContext.console = console;
  }
  // (5) T378.3: what alone.js holds of a model on the GPU alone is its own; the others say what happened and ask.
  // The GPU is lost under the model (forward.js says so to the place's onLost): gpuOnlyLost() says why, to worker.js;
  // the next load begins with no model on the GPU alone (the reason does not outlive the model); a model said to be on
  // the CPU from now on (worker.js, where a run failed under it) goes there by its key, and another model does not
  {
    const ask = (code) => vm.runInContext(strictly(code), gpuContext);
    await ask("loadBegan({ id: 'llama-3b' })");
    const { into, worker } = opened();
    assert.deepEqual([ask("gpuOnlyLost()"), ask("memoryShared()"), ask("weightsBytes()")], [null, true, 65536], "a model on the GPU alone, its GPU going on");
    into.weights.direct.onLost("the GPU's device was lost (a test)");
    assert.equal(ask("gpuOnlyLost()"), "the GPU's device was lost (a test)");
    into.release();
    await ask("loadBegan({ id: 'llama-3b' })");
    assert.ok(worker.terminated);
    assert.deepEqual([ask("gpuOnlyLost()"), ask("weightsRoom()"), ask("weightsDrained()")], [undefined, undefined, undefined], "a new load began with the last one's model on the GPU alone");
    const direct = () => {
      const sunk = ask("checkpointSink()");
      sunk.sink.open(size, proxy(LLAMA3B), "int8", proxy(FORM));
      const alone = Boolean(sunk.weights.direct);
      sunk.release();
      return alone;
    };
    ask("gpuRequest = undefined");  // (no verdict kept, as (4) left one)
    assert.equal(direct(), true);
    await ask("loadBegan({ id: 'llama-3b' })");
    ask("onCpuFromNow({ id: 'llama-3b' })");
    assert.equal(direct(), false, "a model whose GPU failed under it went on the GPU alone again");
    await ask("loadBegan({ id: 'llama-3b' })");
    assert.equal(direct(), false, "and again at its next load: for the rest of the visit");
    await ask("loadBegan({ hf: { repo: 'someone/llama-3b', revision: 'abc' }, id: 'llama-3b' })");
    assert.equal(direct(), true, "a Hugging Face model is another model than the site's of the same id");
    ask("onCpuFromNow({ hf: { repo: 'someone/llama-3b', revision: 'abc' } })");
    assert.equal(direct(), false);
    await ask("loadBegan({ name: 'A file of mine' })");
    assert.equal(direct(), true, "a model with a name alone (a file of the visitor's)");
    // T375.5: by the model's facts, which the worker hands on as the whole form: the same header with layers that keep
    // a state (an LFM2's, whose architecture's line is a Llama's but for them) stays in this memory
    const formed = (form) => {
      const sunk = ask("checkpointSink()");
      sunk.sink.open(size, proxy(LLAMA3B), "int8", proxy({ ...FORM, ...form }));
      const alone = Boolean(sunk.weights.direct);
      sunk.release();
      return alone;
    };
    await ask("loadBegan({ name: 'A file with a form' })");
    assert.deepEqual([formed({}), formed({ arch: "lfm2", convolution: { layers: "cca".repeat(9) + "a", taps: 3 } }), formed({ arch: "lfm2", convolution: { layers: "a".repeat(28), taps: 3 } })],
      [true, false, true], "an LFM2 with convolution layers went on the GPU alone (the worker hands gpuOnlyUnfit() the form with its layers)");
    await ask("loadBegan({ id: 'the next load' })");
  }
  // (6) T377: the GPU's worker says "error" while the weights are written, before there is an engine to hear it (its
  // script did not arrive, or it threw). It takes nothing and counts nothing from then on, and nothing more is posted
  // to it. T491 (the owner, 2026-11-26): the loops that write throw why at their next wait for it (they waited 60 s, and
  // the load then failed with "the GPU took no weights for 60 s"; from T377, the load ran to its end and was loaded
  // again on the CPU after it): the load gives up there, gpuOnlyGaveUp() says why and puts the model on the CPU, and its
  // worker is let go of. An error past the last wait (all of it drained, the engine not built yet) is handed to the
  // engine once it is built: it says why in its own words (the place's onLost), and gpuOnlyReady() has the model loaded
  // again on the CPU, as before. forward.js's engine is a stand-in here that listens as the real one does (gpuside.js:
  // worker.onerror, from start() on)
  {
    const ask = (code) => vm.runInContext(strictly(code), gpuContext);
    const soon = (promise) => {
      let timer;
      return Promise.race([Promise.resolve(promise).then(() => "ended", (error) => `threw: ${error.message}`),
        new Promise((resolve) => { timer = setTimeout(() => resolve("still waiting"), 3000); })]).finally(() => clearTimeout(timer));
    };
    const listening = { ...gpuContext.stand.forward, external: ({ gpu, direct }) => {
      const worker = gpu();
      worker.onerror = (event) => direct.onLost(`the GPU's worker did not start (${event.message})`);
      return { engine: { gpu: engineGpu } };
    } };
    gpuContext.listening = listening;
    const inWq = new Uint8Array(1000), notStarted = "the GPU's worker did not start (no such script)";
    const line = "gpu: the GPU's worker did not start (no such script): the model was on the GPU alone, and is loaded again on the CPU";
    for (const [title, failsAt] of [["before a byte", 0], ["between two stretches", 1], ["while the weights wait for it", 2]]) {
      ask("gpuRequest = undefined; forwardModule = listening");
      await ask(`loadBegan({ id: 'unstarted ${failsAt}' })`);
      const { into, worker } = opened();
      const said = [];
      gpuContext.console = { ...console, warn: (text) => said.push(text) };
      if (failsAt === 0) worker.fail({ message: "no such script" });
      into.weights.write(tensors.wq.offset, inWq);
      if (failsAt === 1) worker.fail({ message: "no such script" });
      into.weights.write(tensors.wq.offset + 1000, inWq);
      // (the room is asked after every stretch, and all of it before the engine is built: sources.js, load.js)
      const waiting = ask("weightsDrained()");
      if (failsAt === 2) setTimeout(() => worker.fail({ message: "no such script" }), 20);
      assert.equal(await soon(waiting), `threw: ${notStarted}`, `a GPU's worker that failed ${title}: the weights went on waiting for it, or did not throw why`);
      assert.equal(await soon(ask("weightsRoom()")), `threw: ${notStarted}`);
      assert.deepEqual(typesPosted(worker), ["open", ...Array(Math.min(failsAt, 2)).fill("weights")], `a GPU's worker that failed ${title} was posted what it will not take`);
      // (the load ends: its weights let go of, as every loader's finally does; then load() asks why)
      into.release();
      assert.equal(ask(`gpuOnlyGaveUp({ id: 'unstarted ${failsAt}' })`), true, `a GPU's worker that failed ${title}: the load was not given up for the CPU`);
      assert.deepEqual([ask(`cpuOnly.has('unstarted ${failsAt}')`), ask("gpuOnlyGone()"), said], [true, false, [line]],
        `a GPU's worker that failed ${title}: not on the CPU from now on, or not said once`);
      await ask("gpuOnlyEnding");
      assert.ok(worker.terminated, `a GPU's worker that failed ${title} was not let go of before the load on the CPU`);
      gpuContext.console = console;
    }
    // a worker that goes on: waited for until it takes them, and a failure of the load besides is the load's own
    {
      ask("gpuRequest = undefined; forwardModule = listening");
      await ask("loadBegan({ id: 'going on' })");
      const { into, worker } = opened();
      into.weights.write(tensors.wq.offset, inWq);
      const waiting = ask("weightsDrained()");
      assert.equal(await Promise.race([waiting.then(() => "ended"), new Promise((resolve) => setTimeout(() => resolve("still waiting"), 300))]), "still waiting",
        "the weights did not wait for a GPU's worker that goes on and has not taken them yet");
      // (the worker takes them, as gpu.js's take() counts: the wait ends)
      const flow = new BigInt64Array(worker.posted.find(({ type }) => type === "open").flow);
      Atomics.add(flow, 0, 1000n);
      Atomics.notify(flow, 0);
      assert.equal(await soon(waiting), "ended");
      into.release();
      assert.deepEqual([ask("gpuOnlyGaveUp({ id: 'going on' })"), ask("cpuOnly.has('going on')")], [false, false], "a load that failed for another reason was given up for the CPU");
      await ask("gpuOnlyEnding");
    }
    // past the last wait, before the engine is built: the engine is handed the error, and the model is loaded again on
    // the CPU once built (gpuOnlyReady(), as from T377)
    {
      ask("gpuRequest = undefined; forwardModule = listening");
      await ask("loadBegan({ id: 'drained' })");
      const { into, worker } = opened();
      const said = [];
      gpuContext.console = { ...console, warn: (text) => said.push(text) };
      await ask("weightsDrained()");
      worker.fail({ message: "no such script" });
      engineGpu = Promise.resolve("prompts on the CPU (a stand-in)");
      gpuContext.made = into.weights.llama({}, {});
      into.release();
      assert.deepEqual([ask("gpuOnlyLost()"), ask("gpuOnlyGone()"), ask("gpuOnlyGaveUp({ id: 'drained' })")], [notStarted, false, false],
        "a GPU's worker that failed once all was drained: the engine was not handed its error, or a built model was given up as a load");
      assert.equal(await ask("gpuOnlyReady({ id: 'drained' }, 9, made.outside.engine)"), false);
      assert.deepEqual([ask("cpuOnly.has('drained')"), worker.terminated, said], [true, true, [line]]);
      gpuContext.console = console;
    }
    // T491: a kept model read under a GPU's worker that fails (its read failed and was taken for a kept file that could
    // not be read, and the model was converted again): what loadConverted() fails with is the GPU's, not a miss, and it
    // reads no more of the kept parts
    {
      ask("gpuRequest = undefined; forwardModule = listening");
      await ask("loadBegan({ id: 'read, kept' })");
      const PART = 28, head = new Uint8Array(PART);
      head.set(new Uint8Array(new Int32Array(LLAMA3B).buffer));
      let read = 0;
      gpuContext.stand.kept = { BEFORE, contextOf, contextRead, contextSet, openKept: async () => ({ manifest: { bytes: size, options: { ...FORM, dtype: "int8" } }, where: "opfs",
        async *parts() {
          for (let at = 0; at < size; at += PART) {
            read += 1;
            if (read === 2) workers.at(-1).fail({ message: "no such script" });
            yield at === 0 ? head : new Uint8Array(Math.min(PART, size - at));
          }
        } }) };
      ask("keptModule = stand.kept");
      const said = [];
      gpuContext.console = { ...console, warn: (text) => said.push(text) };
      const model = { id: "read, kept", hf: { repo: "someone/kept", revision: "abc" } };
      const failed = await ask("loadConverted")(model, new AbortController().signal, 1).then(
        (answer) => `answered ${JSON.stringify(answer)}`, (error) => `threw: ${error.message}`);
      assert.equal(failed, `threw: ${notStarted}`, "T491: a kept model's read that the GPU's worker failed was taken for a kept file that could not be read");
      assert.equal(read, 2, "the kept parts were read on past the GPU's worker that failed");
      assert.equal(ask("gpuOnlyGaveUp")(model), true);
      await ask("gpuOnlyEnding");
      gpuContext.console = console;
    }
    // T491: load() itself, its loader a stand-in that places the weights as checkpointLoaded() does: a GPU's worker that
    // fails gives the first load up at its next wait, and the model is loaded once more, on the CPU, and is ready; so
    // does one that takes nothing for 60 s (the owner, 2026-11-26: it failed the load), on this check's clock (forward.js
    // waits a second at a time, each at once here); a load that fails for another reason fails as it did, and a
    // cancelled one stays cancelled
    {
      const loads = [], said = [], posted = [];
      const silent = async (waiting) => {
        const real = { waitAsync: Atomics.waitAsync, now: performance.now };
        let clock = 0;
        performance.now = () => clock;
        Atomics.waitAsync = (array, index, value, ms) => ({ async: true, value: new Promise((resolve) => setImmediate(() => { clock += ms; resolve("timed-out"); })) });
        try {
          return await waiting();
        } finally {
          [Atomics.waitAsync, performance.now] = [real.waitAsync, real.now];
        }
      };
      gpuContext.loaderStandIn = async (model, kind, signal) => {
        const into = ask("checkpointSink()");
        into.sink.open(size, proxy(LLAMA3B), "int8", proxy(FORM));
        loads.push(into.weights.direct ? "the GPU alone" : "the CPU");
        try {
          if (into.weights.direct) into.weights.write(tensors.wq.offset, inWq);  // (a memory of a page here: nothing is written on the CPU)
          if (into.weights.direct && (gpuContext.failing === "gpu" || gpuContext.failing === "cancel")) workers.at(-1).fail({ message: "no such script" });
          if (gpuContext.failing === "cancel") gpuContext.cancel();
          await ask("weightsRoom()");
          if (into.weights.direct && gpuContext.failing === "silent") await silent(() => ask("weightsDrained()"));
          if (gpuContext.failing === "other") throw new Error("a part was refused");
          signal.throwIfAborted();
          return {};
        } finally {
          into.release();
        }
      };
      ask("gpuRequest = undefined; forwardModule = listening; checkpointLoaded = loaderStandIn; letGo = async () => {}; " +
        "startThreads = async () => 1; pyodide = { version: 'a stand-in' }; llama = { backend: 'a stand-in', seq_len: 1 }");
      gpuContext.console = { ...console, warn: (text) => said.push(text) };
      gpuContext.postMessage = (data) => posted.push(data.type);
      const loaded = (id, failing) => {
        const cancelled = new AbortController();
        Object.assign(gpuContext, { failing, cancel: () => cancelled.abort(new Error("cancelled")) });
        loads.length = said.length = posted.length = 0;
        return ask("load")({ id }, cancelled.signal, 3).then(() => "ready", (error) => `threw: ${error.message}`);
      };
      assert.equal(await loaded("loaded, gpu", "gpu"), "ready");
      assert.deepEqual([loads, said, posted.filter((type) => type === "ready").length], [["the GPU alone", "the CPU"], [line], 1],
        "T491: a load whose GPU's worker failed was not given up and loaded once more on the CPU");
      await ask("gpuOnlyEnding");
      assert.equal(await loaded("loaded, silent", "silent"), "ready");
      assert.deepEqual([loads, said, posted.filter((type) => type === "ready").length],
        [["the GPU alone", "the CPU"], ["gpu: the GPU said nothing for 60 s: the model was on the GPU alone, and is loaded again on the CPU"], 1],
        "T491: a load whose GPU's worker took nothing for 60 s was not loaded once more on the CPU");
      await ask("gpuOnlyEnding");
      assert.equal(await loaded("loaded, other", "other"), "threw: a part was refused");
      assert.deepEqual([loads, said, ask("cpuOnly.has('loaded, other')")], [["the GPU alone"], [], false], "a load that failed for another reason was loaded again on the CPU");
      await ask("gpuOnlyEnding");
      assert.match(await loaded("loaded, cancelled", "cancel"), /^threw: /, "a cancelled load was ready");
      assert.deepEqual([loads, said, ask("cpuOnly.has('loaded, cancelled')")], [["the GPU alone"], [], false], "a cancelled load whose GPU's worker failed was given up for the CPU");
      await ask("gpuOnlyEnding");
      gpuContext.console = console;
      gpuContext.postMessage = () => {};
    }
    // an error after the engine is built is the engine's alone to hear: said once
    {
      await ask("loadBegan({ id: 'late' })");
      const { into, worker } = opened();
      const { direct } = into.weights;
      gpuContext.made = into.weights.llama({}, {});
      into.release();
      const lost = [];
      direct.onLost = (why) => lost.push(why);
      worker.fail({ message: "it threw" });
      assert.deepEqual(lost, ["the GPU's worker did not start (it threw)"], "an error of the GPU's worker under a built engine was not said once");
      await direct.end();
    }
    ask("forwardModule = stand.forward");
    await ask("loadBegan({ id: 'the next load' })");
  }
  console.log("ok: a model on the GPU alone lets go of its GPU's worker where its load ends without it (T156's review); what alone.js holds of it is told and asked (T378.3); " +
    "a GPU's worker that fails before the engine is built is waited for no longer (T377), the load gives up at once and the model goes on the CPU (T491)");
}
