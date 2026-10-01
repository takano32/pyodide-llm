// T130 (the review of 2026-10-01): three things of the KV cache and of what createForward puts after the checkpoint that no
// other test holds to the byte, on forward.js's own createForward() with kernels that do nothing (no Pyodide, no built
// kernels: Node alone, with the native Python for the engine's own placing of the tensors, as worker-sink-check.mjs has it):
//   (1) the cache grows in place and loses no byte it holds. Every position of every layer's keys and values is written
//       with a number of its own as the engine writes a token (the kernel that adds the biases says it: add_inplace on
//       the key and the value), and after every token the cache is read back whole: through the doublings from a small
//       start, a last one that is no doubling (a context that is no power of two times the start), one layer or several,
//       keys and values in float32 or float16. A block moved to the wrong place, or written over before it moved, is read
//       back as another position's number.
//   (2) footprint() is an upper bound of what createForward allocates after the checkpoint at the end of its whole
//       context, and a close one (the megabyte it adds for alignment, and the page the memory grows by): for every kind
//       of model (a key for every head and grouped-query, biases, heads of another size than dim / heads, GPT-2, GPT-NeoX),
//       int8, six bits, float16 and float32, with and without relaxed SIMD, on a shared memory (its base, and the type of
//       the keys and values keysInHalf says) and on a plain one. The worker sizes the memory by footprint(); an engine
//       that put more after the checkpoint than it counted runs out of memory near the end of its context.
//   (3) the memory of a cache that doubles never holds more than the whole context's: it has the size footprint() counts
//       when the last doubling is done (a memory does not shrink, so what it held at its largest is what it holds). The cache
//       before T130 held 1.5 times the context at the last doubling, and a model at the edge of a 32-bit memory (Qwen2.5 3B:
//       3.89 GiB) ran out of memory there.
//   (4) a shared memory that the browser refuses leaves a plain one that holds what the worker sized the shared one for: of
//       the models of the list that are near an edge and two a visitor can open with ?hf=, the plain engine fits a 32-bit
//       memory wherever the shared one did, and a 64-bit memory (16 GiB) wherever pastWide(), asked with the shared size, let
//       it pass (T130: the type of the keys and values on the plain memory, in footprint()).
//   node tests/memory-check.mjs [--forward <another forward.js, to see a broken one fail>]     (PYTHON=.venv/bin/python)
// A forward.js of another place needs jobs.js beside it (it imports it by its own address).
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = new URL("..", import.meta.url);
const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : fileURLToPath(new URL("public/forward.js", root));
const { createForward, footprint, keysInHalf, needsWide, pastWide } = await import(forwardFile);
const { CONTROL_BYTES } = await import(new URL("public/jobs.js", root));
const PAGE = 65536, MiB = 2 ** 20;
let began = performance.now();
const seconds = () => { const was = began; began = performance.now(); return `${((began - was) / 1000).toFixed(1)} s`; };

// the plan Python hands createForward (llama2_numpy's own Places and tensor order), for a header, a form and a dtype:
// where every tensor is, and the bytes of the RoPE tables Python computes (the checkpoint's size is where the places end)
const python = `
import json, sys
import numpy as np
sys.path.insert(0, "public")
import llama2_convert, llama2_numpy as L

def plan_of(header, form, dtype):
    packing = dtype if dtype in L.PACKED else None
    npdtype = np.dtype(np.int8 if dtype == "int8" or packing else dtype)
    probe = L.Llama.__new__(L.Llama)
    (probe.dim, probe.hidden_dim, probe.n_layers, probe.n_heads, probe.n_kv_heads, vocab, probe.seq_len) = header
    probe.vocab_size = abs(vocab)
    probe.head_size = form["head_dim"] or probe.dim // probe.n_heads
    probe.q_dim = probe.n_heads * probe.head_size
    kv_dim = probe.n_kv_heads * probe.head_size
    probe.arch, probe.rotary = form["arch"], probe.head_size
    probe.rope_magnitude = 1.0  # the places, not the values (as external_tensors() sets it)
    # the engine's own condition for keeping int8: the int8 kernels work on groups of 32 only
    keep = npdtype == np.int8 and all(n % 32 == 0 for n in (probe.dim, probe.q_dim, kv_dim, probe.hidden_dim))
    places = L.Places(npdtype, packing)
    freq = lambda width: np.zeros(width // 2)
    if form["arch"] in ("gpt2", "neox"):
        probe.gpt2_tensors(places.take, vocab > 0, keep, kv_dim, places.dtype, freq)
    else:
        probe.llama_tensors(places.take, vocab > 0, keep, kv_dim, form["bias"], places.dtype, freq, form["qk_norm"])
    tensors = {name: getattr(probe, name).plan() for name in L.TENSOR_NAMES if isinstance(getattr(probe, name, None), L.Tensor)}
    table = probe.seq_len * (probe.head_size // 2) * 4
    derived = {"freq_cis_real": table, "freq_cis_imag": table} if form["arch"] == "gpt2" or npdtype != np.float32 else {}
    assert places.offset == llama2_convert.checkpoint_size(header, dtype, form), "the places do not end where the file does"
    return {"header": header, "form": form, "dtype": dtype, "size": places.offset, "tensors": tensors, "derived": derived,
            "keep_int8": bool(keep), "head_size": probe.head_size}

print(json.dumps([plan_of(**s) for s in json.loads(sys.stdin.read())]))
`;
const plansOf = (shapes) => JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c", python],
  { cwd: fileURLToPath(root), input: JSON.stringify(shapes), maxBuffer: 1 << 28 }).toString());
const FORM = { bias: false, arch: "llama", qk_norm: false, head_dim: 0 };

// what createForward is handed, from a plan of Python's: kv_start where the cache starts, and the outlier channels of the
// final norm there are (footprint() counts the most there can be for every int8 model: the second part asks for them)
const empty = new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
function planOf(p, { relaxed = true, kvStart = p.header[6], outliers = 0 } = {}) {
  const [dim, hidden, layers, heads, kvHeads, signedVocab, seqLen] = p.header;
  return {
    arch: p.form.arch, dim, hidden_dim: hidden, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: p.head_size,
    vocab_size: Math.abs(signedVocab), seq_len: seqLen, rotary: p.head_size, parallel_residual: p.form.arch === "neox",
    kv_start: kvStart, rms_norm_eps: 1e-5, shared_classifier: signedVocab > 0, int8: p.keep_int8, relaxed, tensors: p.tensors,
    derived: Object.fromEntries(Object.entries(p.derived).map(([name, bytes]) => [name, new Uint8Array(bytes)])),
    outliers: p.keep_int8 ? Array.from({ length: Math.min(outliers, dim) }, (_, c) => c) : [], half_kv: p.keep_int8,
  };
}
// (the kernels do nothing unless a part says what one does: wrap is how the profiler swaps them, tests/profile.mjs)
const nothing = () => new Proxy({}, { get: () => () => 0 });
const engineOn = (p, plan, { base, memory, halfKeys, kernels = nothing, gpu }) =>
  createForward({ memory, base, size: p.size, kernels: { plain: empty, relaxed: plan.relaxed ? empty : null, wide: false }, plan, halfKeys, gpu, wrap: kernels });
const plainMemory = (p, base) => new WebAssembly.Memory({ initial: Math.ceil((base + p.size) / PAGE) + 1 });
// a shared memory with room for the checkpoint and what the engine puts after it (the GPU's way into the engine wants one)
const sharedMemory = (p, bound) => new WebAssembly.Memory({ initial: Math.ceil((CONTROL_BYTES + p.size) / PAGE) + 1,
  maximum: Math.ceil((CONTROL_BYTES + p.size + bound + 2 * MiB) / PAGE) + 2, shared: true });
// the GPU's worker, that says nothing but that it ended when it was stopped: the engine puts aside the place its keys and
// values of a block come back through (and its rows) as it starts it
const silentGpu = () => {
  let listen;
  return { postMessage(data) { if (data.type === "stop") queueMicrotask(() => listen?.({ data: { type: "ended" } })); },
    set onmessage(handler) { listen = handler; }, set onerror(handler) {}, terminate() {} };
};

// half to float, exactly (forward.js's own)
function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// ---- (1) the cache grows in place
{
  const shapes = [];
  for (const layers of [1, 2, 3]) for (const kvHeads of [2, 4]) for (const seqLen of [19, 33, 70]) {
    for (const dtype of ["float32", "int8"]) shapes.push({ header: [64, 128, layers, 4, kvHeads, 100, seqLen], form: { ...FORM, bias: true }, dtype });
  }
  let runs = 0, positions = 0;
  for (const p of plansOf(shapes)) {
    const [, , layers, , kvHeads, , seqLen] = p.header, kvDim = kvHeads * p.head_size;
    // float32 keys and values in a float32 file's engine; float16 and float32 in an int8 file's (what the worker hands it)
    for (const halfKeys of p.keep_int8 ? [false, true] : [false]) {
      for (const start of [4, 8, 16]) {
        const memory = plainMemory(p, 64);
        // a number of its own for every (side, layer, position, value): below 31744 where float16 keeps it (3 layers of
        // 70 positions of 64 values, twice: 26880)
        const id = (side, l, pos, i) => ((side * layers + l) * seqLen + pos) * kvDim + i;
        let calls = 0, pos = 0;
        const kernels = () => new Proxy({
          // a layer's calls of add_inplace, as a token goes through it with biases: q, k, v, then the two residual adds. The
          // key and the value are written here, where the engine will take them into the cache
          add_inplace(dst, src, n) {
            const call = calls++, l = Math.floor(call / 5), j = call % 5;
            if (j === 1 || j === 2) {
              const F = new Float32Array(memory.buffer);
              for (let i = 0; i < n; i++) F[dst / 4 + i] = id(j - 1, l, pos, i);
            }
          },
          // the float16 of a key and a value: the number as it is, to read back as the half it is
          to_f16(dst, src, n) {
            const F = new Float32Array(memory.buffer), H = new Uint16Array(memory.buffer);
            for (let i = 0; i < n; i++) H[dst / 2 + i] = F[src / 4 + i];
          },
        }, { get: (target, name) => target[name] ?? (() => 0) });
        const engine = engineOn(p, planOf(p, { kvStart: start }), { base: 64, memory, halfKeys, kernels });
        for (pos = 0; pos < seqLen; pos++) {
          calls = 0;
          engine.forward(7, pos, false);
          const { keys, values } = engine.keysAndValues(0, pos + 1);
          const count = pos + 1;
          for (const [side, got] of [[0, keys], [1, values]]) {
            for (let l = 0; l < layers; l++) for (let t = 0; t < count; t++) for (let i = 0; i < kvDim; i++) {
              const want = halfKeys ? halfToFloat(id(side, l, t, i)) : id(side, l, t, i), at = (l * count + t) * kvDim + i;
              if (got[at] !== want) {
                assert.fail(`${p.dtype} file, ${halfKeys ? "float16" : "float32"} keys and values, ${layers} layers, ${kvHeads} keys and values of 16, a cache ` +
                  `starting at ${start} of ${seqLen} positions: after the token at ${pos}, the ${side ? "value" : "key"} of layer ${l} at position ${t} ` +
                  `(${i}) is ${got[at]}, not ${want}`);
              }
            }
          }
          positions++;
        }
        engine.release();
        runs++;
      }
    }
  }
  console.log(`ok: the cache grows in place without losing a byte (${runs} engines of 1 to 3 layers, ${positions} tokens each read back whole: ` +
    `starts of 4, 8 and 16, contexts of 19, 33 and 70, float32 and float16; ${seconds()})`);
}

// ---- (2) footprint() holds what createForward allocates
{
  const big = (name, header, form = {}, dtypes = ["int8", "int6"], loose = false) => dtypes.map((dtype) => ({ name, header, form: { ...FORM, ...form }, dtype, loose }));
  // models that are small in everything but their context and their vocabulary (20000 words: what grows with it, the logits,
  // the corrections of the classifier and the columns of its outlier channels, is 0.1 to 0.6 MiB each, not lost in the
  // megabyte footprint() allows for alignment): the keys and values outweigh the rest, as they do of a 3B
  const shapes = [
    ...big("llama, a key for every head, a classifier of its own", [256, 512, 8, 8, 8, -20000, 4096], {}, ["int8", "int6", "float32", "float16"]),
    ...big("llama, grouped-query, the classifier the embedding", [256, 512, 8, 8, 2, 20000, 4096]),
    ...big("qwen2, biases and grouped-query", [256, 512, 8, 8, 2, 20000, 4096], { bias: true }),
    ...big("qwen3, normalized heads of 64 in a dim of 256", [256, 512, 8, 8, 4, 20000, 4096], { qk_norm: true, head_dim: 64 }),
    // T230: ternary weights (rows of whole groups of 128): no corrections, and the activations' sums after their scales
    ...big("ternary qwen3, grouped-query", [256, 512, 8, 8, 4, 20000, 4096], { qk_norm: true, head_dim: 64 }, ["ternary"]),
    ...big("ternary, a key for every head, a classifier of its own", [256, 768, 8, 8, 8, -20000, 4096], {}, ["ternary"]),
    ...big("gpt2", [256, 1024, 4, 8, 8, 20000, 1024], { arch: "gpt2" }, ["int8", "int6", "float32"]),
    ...big("gpt-neox, a classifier of its own", [256, 1024, 4, 8, 8, -20000, 2048], { arch: "neox" }),
    // rows that are no whole groups of 32: the int8 kernels do not run, the weights are widened to float32, and the keys and
    // values are float32 whatever the worker says may be float16 (llama2_numpy's keep_int8, which half_kv follows). Its
    // groups are of 8, which footprint() takes for 32 (the scales then cost a quarter, not an eighth): more counted, never
    // less, so only the bound is held
    ...big("int8 kernels cannot run (dim 200)", [200, 400, 8, 4, 4, 1000, 4096], {}, ["int8"], true),
  ];
  let engines = 0, tightest = Infinity, loosest = 0;
  const plans = plansOf(shapes.map(({ name, loose, ...shape }) => shape));
  plans.forEach((p, n) => {
    const quantized = ["int8", "int6", "ternary"].includes(p.dtype);
    for (const relaxed of [true, false]) {
      for (const shared of [true, false]) {
        // (the worker's options: whether the int8 kernels run is footprint()'s to say)
        const options = { ...p.form, dtype: p.dtype, int8: true, relaxed, halfKV: quantized, outliers: 8, gpu: false, shared };
        const base = shared ? CONTROL_BYTES : 64, bound = footprint(p.header, p.size, options), halfKeys = keysInHalf(p.header, p.size, options);
        const memory = plainMemory(p, base);
        const engine = engineOn(p, planOf(p, { relaxed, outliers: 8 }), { base, memory, halfKeys });
        const used = engine.memoryBytes() - base - p.size;
        engine.release();
        const where = `${shapes[n].name}, ${p.dtype}, ${relaxed ? "relaxed SIMD" : "no relaxed SIMD"}, ${shared ? "shared" : "plain"} memory, ` +
          `${halfKeys ? "float16" : "float32"} keys and values`;
        assert.ok(used <= bound, `${where}: createForward put ${(used / MiB).toFixed(2)} MiB after the checkpoint, footprint() counted ${(bound / MiB).toFixed(2)}`);
        engines++;
        if (shapes[n].loose) continue;
        // close: the megabyte it allows for alignment less what it did not count (nothing, or a page), and no more
        assert.ok(bound - used <= 1.2 * MiB, `${where}: footprint() counted ${((bound - used) / MiB).toFixed(2)} MiB more than createForward put there`);
        assert.ok(bound - used >= 0.9 * MiB, `${where}: footprint() left ${((bound - used) / MiB).toFixed(2)} MiB over what createForward put there, of the megabyte it allows for alignment`);
        tightest = Math.min(tightest, bound - used);
        loosest = Math.max(loosest, bound - used);
      }
    }
  });
  // and where the page asks for the GPU (the prompt's blocks on it, T135: the engine puts aside the place the keys and values
  // of a block come back through, and its rows), for every model the GPU takes: int8 and six bits of whole groups
  let withGpu = 0;
  plans.forEach((p, n) => {
    if (!p.keep_int8 || shapes[n].loose || p.dtype === "ternary") return;  // (T231: ternary weights stay on the CPU, T232 is the GPU's)
    for (const relaxed of [true, false]) {
      const options = { ...p.form, dtype: p.dtype, int8: true, relaxed, halfKV: true, outliers: 8, gpu: true, shared: true };
      const bound = footprint(p.header, p.size, options), halfKeys = keysInHalf(p.header, p.size, options);
      const engine = engineOn(p, planOf(p, { relaxed, outliers: 8 }), { base: CONTROL_BYTES, memory: sharedMemory(p, bound), halfKeys, gpu: silentGpu });
      const used = engine.memoryBytes() - CONTROL_BYTES - p.size;
      assert.ok(engine.gpu, "no GPU's worker was asked for");
      engine.release();
      const where = `${shapes[n].name}, ${p.dtype}, ${relaxed ? "relaxed SIMD" : "no relaxed SIMD"}, a GPU asked for`;
      assert.ok(used <= bound && bound - used <= 1.2 * MiB && bound - used >= 0.9 * MiB,
        `${where}: createForward put ${(used / MiB).toFixed(2)} MiB after the checkpoint, footprint() counted ${(bound / MiB).toFixed(2)}`);
      withGpu++;
    }
  });
  console.log(`ok: footprint() holds what createForward allocates and no more than a megabyte over (${engines} engines: ${plans.length} models and dtypes, ` +
    `with and without relaxed SIMD, on a shared and a plain memory; ${(tightest / MiB).toFixed(2)} to ${(loosest / MiB).toFixed(2)} MiB over; ` +
    `and ${withGpu} with the GPU asked for; ${seconds()})`);
}

// ---- (3) the memory of a cache that doubles never holds more than the whole context's
{
  const shapes = [4096, 3000].map((seqLen) => ({ header: [256, 512, 8, 8, 8, 1000, seqLen], form: FORM, dtype: "int8" }));
  let engines = 0;
  for (const p of plansOf(shapes)) {
    const seqLen = p.header[6];
    for (const halfKeys of [false, true]) {
      // (float16 on a shared memory, float32 on a plain one: where the shape is a key for every head)
      const options = { ...p.form, dtype: p.dtype, int8: true, relaxed: true, halfKV: true, outliers: 8, gpu: false, shared: halfKeys };
      assert.equal(keysInHalf(p.header, p.size, options), halfKeys, "the options do not stand for the type of keys and values they are meant to");
      const memory = plainMemory(p, 64), engine = engineOn(p, planOf(p, { kvStart: 256 }), { base: 64, memory, halfKeys });
      // a token at each position where the cache is full: it doubles (256, 512, 1024, 2048, and the context's end where that is no doubling)
      for (const pos of [255, 256, 511, 512, 1023, 1024, 2047, 2048, seqLen - 1]) engine.forward(7, pos, false);
      const used = engine.memoryBytes() - 64 - p.size, bound = footprint(p.header, p.size, options);
      engine.release();
      const where = `a context of ${seqLen}, ${halfKeys ? "float16" : "float32"} keys and values`;
      assert.ok(used <= bound, `${where}: its memory held ${(used / MiB).toFixed(1)} MiB after the checkpoint at the last doubling, footprint() counts ${(bound / MiB).toFixed(1)}`);
      assert.ok(bound - used <= 1.2 * MiB, `${where}: its memory held ${(used / MiB).toFixed(1)} MiB, ${((bound - used) / MiB).toFixed(1)} under footprint()`);
      engines++;
    }
  }
  console.log(`ok: a cache that doubles up to its context holds the whole context and no more at its last doubling (${engines} engines of 4096 and 3000 positions, float32 and float16; ${seconds()})`);
}

// ---- (4) a shared memory refused leaves a plain one that holds what the shared one was sized for
{
  const GiB = 2 ** 30;
  // [name, header, size, form, dtype]: sizes from llama2_convert.checkpoint_size() of each model's config.json (the list's
  // headers; a 12B and a 13B model are what ?hf= can open that the list has none like)
  const listed = [
    ["llm-jp-3 440M", [1024, 3584, 16, 8, 8, -99584, 4096], 503255068, {}, "int8"],
    ["Qwen2.5 0.5B Instruct", [896, 4864, 24, 14, 2, 151936, 4096], 555992604, { bias: true }, "int8"],
    ["Qwen3 0.6B (no thinking)", [1024, 3072, 28, 16, 8, 151936, 4096], 670744604, { qk_norm: true, head_dim: 128 }, "int8"],
    ["llm-jp-3.1 1.8B instruct4", [2048, 7168, 24, 16, 16, -99584, 4096], 2101354524, {}, "int8"],
    ["SmolLM2 1.7B Instruct", [2048, 8192, 24, 32, 32, 49152, 4096], 1925586972, {}, "int8"],
    ["Qwen2.5 3B Instruct", [2048, 11008, 36, 16, 2, 151936, 4096], 3472375836, { bias: true }, "int8"],
    ["Llama 3.2 3B Instruct", [3072, 8192, 28, 24, 8, 128256, 4096], 3614847004, {}, "int8"],
    ["sarashina2.2 3B Instruct", [2560, 8960, 32, 16, 8, -102400, 4096], 2936678428, {}, "int6"],
    ["Qwen3 4B (no thinking)", [2560, 9728, 36, 32, 8, 151936, 4096], 4525840412, { qk_norm: true, head_dim: 128 }, "int8"],
    ["Qwen3 4B (no thinking)", [2560, 9728, 36, 32, 8, 151936, 4096], 3520272412, { qk_norm: true, head_dim: 128 }, "int6"],
    ["Llama 3.1 Swallow 8B Instruct", [4096, 14336, 32, 32, 8, -128256, 4096], 9034809372, {}, "int8"],
    ["llm-jp-4 8B instruct", [4096, 14336, 32, 32, 8, -196608, 4096], 9664741404, {}, "int8"],
    ["Qwen3 8B (no thinking)", [4096, 12288, 36, 32, 8, -151936, 4096], 9215463452, { qk_norm: true }, "int8"],
    ["japanese-gpt 1B", [2048, 8192, 24, 16, 16, 44928, 1024], 1467400220, { arch: "gpt2" }, "int8"],
    ["japanese-gpt-neox small", [768, 3072, 12, 12, 12, -44416, 2048], 172787740, { arch: "neox" }, "int8"],
    ["Pythia 12B (?hf=)", [5120, 20480, 36, 40, 40, -50688, 2048], 13333749788, { arch: "neox" }, "int8"],
    ["Llama 2 13B (?hf=)", [5120, 13824, 40, 40, 40, -32000, 4096], 11390177308, {}, "int6"],
    // T230: the ternary models there are, as ternary (Ternary Bonsai 1.7B, 4B, 8B: Qwen3s; Ternary Bonsai 2 27B: a Qwen3.5)
    ["Ternary Bonsai 1.7B", [2048, 6144, 28, 16, 8, 151936, 4096], 484372508, { qk_norm: true, head_dim: 128 }, "ternary"],
    ["Ternary Bonsai 4B", [2560, 9728, 36, 32, 8, 151936, 4096], 1132048412, { qk_norm: true, head_dim: 128 }, "ternary"],
    ["Ternary Bonsai 8B", [4096, 12288, 36, 32, 8, 151936, 4096], 2129760284, { qk_norm: true, head_dim: 128 }, "ternary"],
    ["Ternary Bonsai 2 27B", [5120, 17408, 64, 24, 4, -248320, 4096], 7662073884,
      { arch: "qwen35", head_dim: 256, linear: { every: 4, key_heads: 16, value_heads: 48, key_dim: 128, value_dim: 128, conv: 4 } }, "ternary"],
  ];
  let cases = 0, atTheEdge = 0;
  for (const [name, header, size, form, dtype] of listed) {
    for (const relaxed of [true, false]) {
      for (const gpu of [true, false]) {
        const options = { ...FORM, ...form, dtype, int8: true, relaxed, halfKV: true, outliers: 8, gpu };
        const shared = footprint(header, size, { ...options, shared: true }), plain = footprint(header, size, { ...options, shared: false });
        const where = `${name} as ${dtype}, ${relaxed ? "relaxed SIMD" : "no relaxed SIMD"}${gpu ? ", the GPU asked for" : ""}`;
        // a model the shared memory holds in 32 bits is held by the plain one, which the page makes where the browser refused
        // the shared (the worker sized by the shared one's, and chose 32 or 64 bits by it)
        assert.ok(needsWide(size, shared) || !needsWide(size, plain),
          `${where}: ${((8192 + size + shared) / GiB).toFixed(2)} GiB on a shared memory fits 32 bits, ${((64 + size + plain) / GiB).toFixed(2)} on a plain one does not`);
        // and pastWide(), asked with the shared size before the weights are fetched, lets through no model that the plain
        // memory a refusal leaves cannot hold (16 GiB of a 64-bit memory)
        assert.ok(pastWide(size, shared) || !pastWide(size, plain),
          `${where}: ${((8192 + size + shared) / GiB).toFixed(2)} GiB on a shared memory passes pastWide(), ${((64 + size + plain) / GiB).toFixed(2)} on a plain one is past 16 GiB`);
        if (needsWide(size, plain) !== needsWide(size, shared) || plain !== shared) atTheEdge++;
        cases++;
      }
    }
  }
  console.log(`ok: a shared memory refused leaves a plain one that holds what the shared one was sized for (${cases} cases of ${listed.length} models, ` +
    `the plain memory's keys and values a type of their own in ${atTheEdge})`);
}
