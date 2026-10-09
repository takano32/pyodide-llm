// forward.js (T93): one token's forward pass in JavaScript, on the SIMD kernels of kernels/*.ts, on a WebAssembly
// memory of its own that holds the weights. Python (llama2_numpy.Llama with external=) still reads the header, says
// where every tensor is, tokenizes, samples and runs the generation loop; this file does what the engine's kernel_forward()
// did until T93 retired it, in the same order with the same kernels. What it saves is the Python between the
// kernel calls (about 100 per token): 1.15 to 1.26 times the speed on int8 (T93, stage 1a).
//
// A plain ES module: the worker imports it, and so does Node (tests/smoke.mjs).
//
//   const kernels = compileKernels(plainBytes, relaxedBytes);   // simdkernel_plain.wasm, simdkernel_relaxed_plain.wasm
//   const { memory, base } = weightsMemory(size);              // then write the checkpoint at base
//   const outside = external({ memory, base, size, kernels }); // what Llama(external=) takes

// T349: this file is the window of the forward pass: createForward(), external() and compileKernels(), and every name it
// exported as one file. The rest is in the modules of forward/: choice.js (the GPU or the CPU, and its times), paths.js
// (what /benchmark/ times of the page's path), memory.js (the memory a model takes) and alone.js (a model on the GPU
// alone). Each is asked for with this file's own ?v=<build>, as jobs.js is, so that all come from one deployment, and
// all at once: one after another's end would add a round trip for each.
const modules = Object.fromEntries(["choice", "paths", "memory", "alone"].map((name) =>
  [name, import(new URL(`forward/${name}.js${new URL(import.meta.url).search}`, import.meta.url))]));

// what a job of a phase is, shared with the software threads (helper.js), from the same deployment as this file
const { CONTROL_BYTES, GEN, QUIT, COUNTER, FINISHED, ACTIVE, TOTAL, WAKE, JOBS, JOB, JOB_TABLE, BATCH, ROWS, COUNT, SIZE, FIRST,
  GPU_DONE, GPU_FAILED, GPU_BEAT, GPU_WANTED, addressed, runner } = await import(new URL(`jobs.js${new URL(import.meta.url).search}`, import.meta.url));
export { BATCH };
const { GPU_END_MS, GPU_BLOCK, GPU_RECHECK, GPU_TOKENS, TOKEN_RECHECK, STOPS_MOST, lowerMedian, BETTER, promptTimes,
  tokenTimes, PROMPTS_UNTIMED, PROMPTS_GPU, PROMPTS_CPU, gpuLine } = await modules.choice;
const { PATH_ROUNDS, timePrompts, timedCell, SEARCH_SECONDS, endSearch } = await modules.paths;
const { PAGE, align, memory64, layerSlots, attendingLayers, linearWidths, frameArrays, rotatedWidths, frameBytes,
  footprint, keysInHalf, needsWide, pastWide, automaticDtype, growMemory, weightsMemory } = await modules.memory;
const { GPU_WEIGHT_BYTES, GPU_TERNARY_BYTES, gpuBytes, GPU_ALONE, gpuHoles, placer, BOTH_ON_8, weightsPlace,
  layerWeightsOf, USAGE_UNKNOWN, cpuReadBytes, aloneVerdict, aloneHolds, OUTSIDE_VOCABULARY, gpuOnlyUnfit, gpuOnlyWeights,
  gpuOnlyPlan } = await modules.alone;
export { GPU_END_MS, GPU_BLOCK, GPU_TOKENS, promptTimes, tokenTimes, PROMPTS_UNTIMED, PROMPTS_GPU, PROMPTS_CPU, gpuLine,
  PATH_ROUNDS, timePrompts, timedCell, SEARCH_SECONDS, endSearch, memory64, footprint, keysInHalf, GPU_WEIGHT_BYTES,
  GPU_TERNARY_BYTES, gpuBytes, gpuHoles, placer, BOTH_ON_8, weightsPlace, layerWeightsOf, USAGE_UNKNOWN, cpuReadBytes,
  aloneVerdict, aloneHolds, OUTSIDE_VOCABULARY, gpuOnlyUnfit, gpuOnlyWeights, gpuOnlyPlan, needsWide, pastWide,
  automaticDtype, growMemory, weightsMemory };

// T120: no phase comes near this without progress (the longest token measured, Qwen2.5 7B's on CI, took 286 ms in
// all): a count that has not moved for so long means a software thread the browser stopped in the middle of its chunk
const STALLED_MS = 10000;
// T135: the GPU's worker says something at least this often while it puts a model on the GPU: after every step, each
// of which it gives up itself after 180 s (gpu.js's STEP_MS, T147: SwiftShader compiles a shader in up to 90 s). A
// worker quiet for longer than that is one the browser ended
const GPU_QUIET_MS = 200000;

// T147: the number of the GPU's requests, for this worker and every model it loads (the memory and its control area
// are kept from model to model, T96): a request of an engine let go is never one of the next engine's
let gpuRequests = 0;

/** The kernels as WebAssembly modules. The relaxed one fails to compile where relaxed SIMD is missing (Safari):
 * then int8 runs on matmul_q8. wide (T101): the build for a 64-bit memory (simdkernel_*64.wasm). */
export function compileKernels(plain, relaxed, wide = false) {
  let relaxedModule = null;
  try {
    relaxedModule = relaxed ? new WebAssembly.Module(relaxed) : null;
  } catch {
    relaxedModule = null;
  }
  return { plain: new WebAssembly.Module(plain), relaxed: relaxedModule, wide };
}

// ---- the helper threads (stage 2): the control area at the start of a shared memory (jobs.js has its layout)
// how many chunks per thread the rows of a matmul are cut into: whoever is free takes the next one, so a slow core
// (a little core of a big.LITTLE phone) simply takes fewer. 2 to 16 measured the same (T93); fewer does not steal.
const CHUNKS_PER_THREAD = 4;

/** What Llama(external=) takes: the size of the checkpoint, read() for the few bytes Python looks at itself, and
 * start(plan), which builds the forward pass. */
/** T156 direct ({ place, stored, onLost }, gpuOnlyWeights()): the layers' matrices are on the GPU alone, memory holds
 * the rest of the checkpoint (stored bytes) packed without them, a tensor at place(its offset in the checkpoint);
 * onLost(why) is told where the GPU stops (the model has to be loaded again on the CPU). */
export function external({ memory, base, size, kernels, spawn, gpu, gpuRoom, memoryUnsaid, gpuRemembered, gpuForce, halfKeys, direct }) {
  const place = direct?.place ?? ((offset) => offset);
  const outside = {
    size,
    read: (offset, length) => new Uint8Array(memory.buffer, base + place(offset), length).slice(),
    start: (plan) => {
      plan = plan.toJs ? plan.toJs({ dict_converter: Object.fromEntries }) : plan;
      // (T156: every tensor but those on the GPU alone where it is in memory; those keep their offsets in the
      // checkpoint, which only the GPU's worker reads)
      if (direct) {
        plan.tensors = Object.fromEntries(Object.entries(plan.tensors).map(([name, t]) => [name, GPU_ALONE.includes(name) ? t
          : { ...t, offset: place(t.offset), ...(t.scales ? { scales: place(t.scales) } : {}) }]));
      }
      outside.engine = createForward({ memory, base, size: direct ? direct.stored : size, kernels, spawn, gpu, gpuRoom,
        memoryUnsaid, gpuRemembered, gpuForce, halfKeys, direct, plan });
      return outside.engine;
    },
  };
  return outside;
}

// half to float, exactly (the same as NumPy's astype(float32))
function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

/** spawn (stage 2, a shared memory only): starts one helper thread with { memory, plain, relaxed } and resolves once
 * it is ready; the result has terminate(). Without it, or on a memory that is not shared, everything runs here. */
/** gpu (T135; T148: wherever the worker has WebGPU): makes the GPU's worker (gpu.js, not started: a Worker, or what
 * posts and listens as one). A prompt's blocks then go through the layers there where the model allows it and the GPU
 * is faster; see forwardMany. gpuRoom (T148): the bytes this device can give the GPU's copy of the layers (the
 * worker's reckoning from navigator.deviceMemory; none said: no limit). memoryUnsaid (T205): the browser does not say
 * how much memory the device has (no navigator.deviceMemory: Safari, Firefox); a generation's steps then stay on the
 * CPU. gpuRemembered: what the page kept of the GPU's shaders on an earlier visit ({ key, matrices, attention },
 * gpu.js). */
/** wrap (tests/profile.mjs only): gets the kernels' exports and returns what to call instead, to time the forward
 * pass with some kernels replaced by functions that do nothing. */
/** stalledMs (tests only): how long a phase may make no progress before its software threads are given up (T120) */
/** clock (tests only, T199): the time the threads' search reads, in ms (a made-up one slows a block of its choice) */
/** halfKeys (T160): keysInHalf's answer for this model on the memory it got (T130: a shared one refused, the
 * worker asks again for the plain one). Left out (the tests, the benchmark's own model): float16 on a shared memory
 * where every head has keys of its own, which keysInHalf answers for every model that fits a 32-bit memory with
 * float32 keys and values. */
/** gpuForce (tests only, T147): { matrices, attention }, the names of the GPU's shaders to take (shaders.js's
 * promptForms, gpu.js's attentions), without timing the others. T148: fallback, a fallback adapter taken as a GPU
 * (SwiftShader and lavapipe: the only WebGPU of CI and the development machine); always, every block the GPU can take
 * goes there, whatever the CPU's time (a fallback adapter is far slower than the CPU); quick (the page's tests), the
 * first right shader of the matrices untimed, and no block timed (SwiftShader timing Llama 3.2 1B's took more than
 * gpu.js's STEP_MS in CI, T147); pieceBytes (T155), the most bytes of a piece of a matrix on the GPU, so that a small
 * model goes in pieces as a matrix past a buffer of the device does */
/** direct (T156, external()): the layers' matrices (T210: and the tables) are on the GPU alone: nothing runs on the CPU,
 * every block and step goes to the GPU (as gpuForce.always), and one the GPU cannot take throws (the worker loads the
 * model again on the CPU). T210: no cache of keys and values here either; the GPU embeds a prompt's tokens itself */
export function createForward({ memory, base, size, kernels, plan, spawn, gpu, gpuRoom, memoryUnsaid, gpuRemembered, gpuForce = {},
  halfKeys, direct, wrap = (exports) => exports, stalledMs = STALLED_MS, clock = () => performance.now() }) {
  // every block and step the GPU can take goes there, whatever the CPU's time: the tests' fallback adapter, and a model
  // on the GPU alone (T156)
  const always = Boolean(gpuForce.always || direct);
  const { dim, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: headSize, vocab_size: vocab,
    seq_len: seqLen, rotary, arch } = plan;
  const hidden = plan.hidden_dim, kvDim = kvHeads * headSize, qDim = heads * headSize;
  const gpt2 = arch === "gpt2", layerNorm = arch === "gpt2" || arch === "neox", parallel = plan.parallel_residual;
  // T255: which layers RoPE turns q and k of: none of a GPT-2's, and not the ones a SmolLM3 leaves alone
  const unturned = Array.from(plan.unturned || []);
  const turns = Array.from({ length: layers }, (_, l) => !gpt2 && !unturned.includes(l));
  // T229: a Qwen3.5's linear-attention layers (null: none), which layers they are, and each layer's place among the
  // layers of its kind (the layer itself where all attend); attending: the layers with keys and values
  // T260: an LFM2's convolution layers (null: none), the same way
  const linear = plan.linear ?? null, convolution = plan.convolution ?? null, slots = layerSlots(layers, linear, convolution);
  // T237: the block of a rotated basis (0: the model's own basis). llama2_numpy.py has the definition above hadamard()
  const rotated = plan.rotated || 0;
  const lines = slots.map(([kind]) => kind), placeOf = slots.map(([, a]) => a), attending = attendingLayers(layers, linear, convolution);
  const imports = { env: { memory } };
  const wide = Boolean(kernels.wide);  // T101: a 64-bit memory, whose kernels take their addresses as BigInt
  // a page that is not cross-origin isolated has no SharedArrayBuffer to ask about: its memory is not shared
  const sharedMemory = typeof SharedArrayBuffer !== "undefined" && memory.buffer instanceof SharedArrayBuffer;
  const k = wrap(addressed(new WebAssembly.Instance(kernels.plain, imports).exports, wide));
  const relaxed = plan.int8 && plan.relaxed && kernels.relaxed ? wrap(addressed(new WebAssembly.Instance(kernels.relaxed, imports).exports, wide)) : null;
  const bias = relaxed ? 64 : 0;

  // ---- memory: the checkpoint at base, everything else after it; views are made again after the memory grows
  let top = align(base + size);
  let buffer, F, I, U, H;
  const views = () => {
    if (buffer !== memory.buffer) {
      buffer = memory.buffer;
      F = new Float32Array(buffer); I = new Int8Array(buffer); U = new Uint8Array(buffer); H = new Uint16Array(buffer);
    }
  };
  const alloc = (bytes) => {
    const at = top;
    top = align(at + bytes);
    if (top > memory.buffer.byteLength) growMemory(memory, Math.ceil((top - memory.buffer.byteLength) / PAGE), wide);
    views();
    return at;
  };
  views();

  const T = plan.tensors, derived = plan.derived ?? {};
  const count = (t) => t.shape.reduce((a, b) => a * b, 1);
  // weight i of an int8, int6 or ternary tensor, as the int8 it is: int6 (T98) unpacked from its group of 24 bytes,
  // the layout of llama2_numpy.pack6 (six bits, then two zero bits); ternary (T230) its two bits less one, the layout
  // of llama2_numpy.pack_ternary
  function weightAt(t, i) {
    if (t.kind === "int8") return I[base + t.offset + i];
    if (t.kind === "ternary") return ((U[base + t.offset + (i >> 2)] >> (2 * (i & 3))) & 3) - 1;
    const group = base + t.offset + ((i / 32) | 0) * 24, j = i % 32;
    const low = j < 16 ? U[group + j] & 15 : U[group + j - 16] >> 4;
    const top = (U[group + 16 + (j % 8)] >> (2 * ((j / 8) | 0))) & 3;
    return (((low | (top << 4)) << 2) << 24) >> 24;  // the byte as a signed int8
  }
  // a float32 copy of a tensor: float16 converted, int8, int6 and ternary times the scale of their group (Math.fround is the
  // float32 product NumPy computes)
  function widen(t) {
    const n = count(t), at = alloc(n * 4);
    if (t.kind === "f16") {
      for (let i = 0; i < n; i++) F[at / 4 + i] = halfToFloat(H[(base + t.offset) / 2 + i]);
    } else {
      const g = t.group;
      for (let i = 0; i < n; i++) F[at / 4 + i] = Math.fround(weightAt(t, i) * F[(base + t.scales) / 4 + ((i / g) | 0)]);
    }
    return at;
  }
  const widened = new Map();
  // the address of a float32 tensor, widened once if the file does not hold float32
  function floats(name) {
    if (name in derived) {
      if (!widened.has(name)) {
        const bytes = derived[name], at = alloc(bytes.length);
        U.set(bytes, at);
        widened.set(name, at);
      }
      return widened.get(name);
    }
    const t = T[name];
    if (!t) return 0;
    if (t.kind === "f32") return base + t.offset;
    if (!widened.has(name)) widened.set(name, widen(t));
    return widened.get(name);
  }
  // a matrix: int8 or int6 (values, scales, corrections) or ternary (values, scales) when plan.int8, else float32
  function matrix(name) {
    const source = plan.shared_classifier && name === "wcls" ? "token_embedding_table" : name;
    const t = T[source];
    if (!t) return null;
    const [rows, n] = t.shape.slice(-2);
    // T156: on the GPU alone: where each layer's values and scales start in the checkpoint (what gpu.js was opened
    // with), for the plan the GPU's worker is started with; the CPU never multiplies by it (T210: nor by the classifier)
    if (direct && GPU_ALONE.includes(source)) {
      const ternary = t.kind === "ternary";
      return { rows, n, int8: true, six: false, ternary, group: t.group, onGpu: true,
        layer: (l) => [t.offset + l * rows * (ternary ? n / 4 : n), t.scales + l * rows * (n / t.group) * 4] };
    }
    if ((t.kind === "int8" || t.kind === "int6" || t.kind === "ternary") && plan.int8) {
      const six = t.kind === "int6", ternary = t.kind === "ternary", rowBytes = ternary ? n / 4 : six ? n / 32 * 24 : n;
      const values = base + t.offset, scales = base + t.scales, groups = count(t) / t.group;
      let corrections = scales;
      if (relaxed && !ternary) {
        // relaxed SIMD multiplies by 7-bit unsigned activations with a bias of 64, which this takes out again:
        // dot(w, q - 64) = dot(w, q) - 64 * sum(w). -64 * sum of the group, an int32 (T197: the kernels add it to the
        // group's integer sum; before it the float32 scale * sum)
        corrections = alloc(groups * 4);
        // the same numbers as a sum in JavaScript, a kernel's speed (T98, T123: 7B spent 266 s here one value at a
        // time). Groups of 32: relaxed runs only where every row is whole groups
        (six ? k.six_sums : k.int8_sums)(corrections, values, groups);
      }
      const layer = (l) => [values + l * rows * rowBytes, scales + l * rows * (n / t.group) * 4, corrections + l * rows * (n / t.group) * 4];
      return { rows, n, int8: true, six, ternary, group: t.group, layer };
    }
    const w = floats(source);  // float32 as it is, or widened once (shared with the embedding when it is the same table)
    return { rows, n, int8: false, layer: (l) => [w + l * rows * n * 4] };
  }

  // ---- the activations: one frame per token, BATCH of them for a prompt (T108). A frame holds what one token needs,
  // in the order it always had, and the next token's frame follows: every array of token t is t * S bytes after token
  // 0's. (One block per array, BATCH tokens long, put token 0's arrays 32 KB apart for tiny-lm, and they fought for
  // the same lines of the cache: 3% slower on one token at a time.)
  // T110: an int8 model keeps its keys and values in float16 (half the bytes attention reads) where the memory is
  // shared, that is where there are software threads: several threads wait on the memory, and reading half of it
  // made a long context 1.2 times as fast with 4; one thread waits on the arithmetic, and widening every key and
  // value made it 1.6 to 1.8 times as slow. A float32 model, the one held to NumPy's numbers, stays in float32.
  // T160: a grouped-query model's too, unless that alone would not fit a 32-bit memory (keysInHalf, halfKeys). T130: and
  // on a memory that is not shared, float16 where float32 would not fit (halfKeys). The GPU's keys and values stay
  // float16 either way: cacheHalves widens them into a float32 cache.
  // KV: the bytes of one position's keys in the cache; KF: in float32.
  const halfKV = Boolean(plan.half_kv) && (halfKeys ?? (sharedMemory && kvHeads >= heads));
  const D = dim * 4, HD = hidden * 4, KF = kvDim * 4, QF = qDim * 4, KV = kvDim * (halfKV ? 2 : 4);
  const inFrame = frameArrays(dim, hidden, kvDim, qDim, linear, rotated, T.wo?.kind === "ternary", convolution), S = frameBytes(inFrame);
  const frames = alloc(BATCH * S), at = {};
  inFrame.reduce((offset, [name, bytes]) => { at[name] = frames + offset; return offset + align(bytes); }, 0);
  // kNow, vNow: this token's key and value in float32, before they go into the cache
  const { x, xb, xb2, q, kNow, vNow, before, hb, hb2, xq, xs, gate, mixed, conv, z, work, xr } = at;
  const A = seqLen * heads * 4;  // the scores of one token's attention
  const att = alloc(BATCH * A), logits = alloc(vocab * 4);
  const wq = matrix("wq"), wk = matrix("wk"), wv = matrix("wv"), wo = matrix("wo");
  const w1 = matrix("w1"), w2 = matrix("w2"), w3 = matrix("w3"), wcls = matrix("wcls");
  const attW = floats("rms_att_weight"), ffnW = floats("rms_ffn_weight"), finalW = floats("rms_final_weight");
  const attB = floats("ln_att_bias"), ffnB = floats("ln_ffn_bias"), finalB = floats("ln_final_bias");
  const bo = floats("bo"), b1 = floats("b1"), b2 = floats("b2"), bq = floats("bq"), bk = floats("bk"), bv = floats("bv");
  const qNorm = floats("q_norm"), kNorm = floats("k_norm");  // T124: Qwen3 normalizes every head of q and k
  const eps = plan.rms_norm_eps ?? 1e-5;  // the epsilon of every RMSNorm (T124: Qwen3's is 1e-6)
  const cosTable = floats("freq_cis_real"), sinTable = floats("freq_cis_imag");
  const positions = gpt2 ? floats("positions") : 0;
  const embedding = T.token_embedding_table;
  const embeddingRows = embedding.kind === "f16" ? floats("token_embedding_table") : 0;
  // T229: a Qwen3.5's gate of a full-attention layer's output, and its linear-attention layers' tensors: q, k and v
  // in one matrix, z, the output; in float32 the two small matrices of the gates, the taps of the convolution,
  // dt_bias, the decay and the norm of a value head
  const wg = matrix("wg"), wqkv = matrix("wqkv"), wz = matrix("wz"), wout = matrix("wout");
  const wb = floats("wb"), wa = floats("wa"), taps = floats("conv"), dtBias = floats("dt_bias"), decays = floats("decay");
  const deltaNorm = floats("delta_norm");
  // their state (linearStateBytes): for each such layer a matrix (keyDim, valueDim) a value head, twice (flips: which
  // of the two holds the state now; the delta rule writes the other; position 0 clears both), and the last conv tokens' q, k and v before the
  // convolution (the oldest first, the token under way last). stateAt: the position that comes next, -1 once a run
  // stopped half way. And the two weights rmsnorm takes the l2 norm of a head with: x / sqrt(sum(x * x) + 1e-6) is
  // rmsnorm's w * x / sqrt(mean(x * x) + eps) with w = 1 / sqrt(n) and eps = 1e-6 / n; q is divided by sqrt(n) more
  const keyHeads = linear?.key_heads, valueHeads = linear?.value_heads, keyDim = linear?.key_dim, valueDim = linear?.value_dim;
  const { mixed: mixedWidth = 0, read: readWidth = 0 } = linear ? linearWidths(linear) : {};
  // T260: an LFM2's convolution layers have a matrix in (win: 3 dim rows), their taps under the same name (conv) and
  // a matrix out (wout); their state is the convolution's rows alone, taps of dim values a layer
  const win = matrix("win");
  const stateBytes = linear ? valueHeads * keyDim * valueDim * 4 : 0;
  const convBytes = linear ? linear.conv * mixedWidth * 4 : convolution ? convolution.taps * dim * 4 : 0;
  const lineCount = layers - attending;
  const states = linear ? alloc(2 * lineCount * stateBytes) : 0, convRows = convBytes ? alloc(lineCount * convBytes) : 0;
  const qUnit = linear ? alloc(keyDim * 4) : 0, kUnit = linear ? alloc(keyDim * 4) : 0;
  const flips = new Uint8Array(lineCount);
  let stateAt = 0;
  if (linear) {
    F.fill(1 / keyDim, qUnit / 4, qUnit / 4 + keyDim);
    F.fill(1 / Math.sqrt(keyDim), kUnit / 4, kUnit / 4 + keyDim);
  }

  // T237: the signs of a rotated basis for every width a matrix reads (Python's, each times 1 / sqrt(block))
  const signs = rotated ? Object.fromEntries(rotatedWidths(dim, hidden, qDim, linear).map((width) => {
    if (!(`signs.${width}` in derived)) throw new Error(`The rotated basis has no signs for an input ${width} wide.`);
    return [width, floats(`signs.${width}`)];
  })) : null;

  // the outlier channels of the classifier's input (T92): their columns in float32, multiplied apart (T210: not of a
  // classifier on the GPU alone, which is not here; T226: the GPU multiplies a float classifier for a model with them,
  // so such a model stays on the GPU alone and needs no columns)
  const channels = plan.outliers ?? [];
  let columns = 0, picked = 0;
  if (channels.length && !direct) {
    const t = plan.shared_classifier ? T.token_embedding_table : T.wcls;
    columns = alloc(channels.length * vocab * 4);
    picked = alloc(channels.length * 4);
    const groups = dim / t.group;
    channels.forEach((c, i) => {
      for (let v = 0; v < vocab; v++) {
        F[columns / 4 + i * vocab + v] = Math.fround(weightAt(t, v * dim + c) * F[(base + t.scales) / 4 + v * groups + ((c / t.group) | 0)]);
      }
    });
  }
  // why this model's prompt stays on the CPU, or null: the first stage (T135) takes Llama's layers of int8 weights
  // (T153: with Qwen2's biases, Qwen3's norms of the heads and heads of another size than dim / heads as well; T154:
  // GPT-2's and GPT-NeoX's as well, whose FFN has no gate: w3 is null; T155: int6 weights and 64-bit memories too)
  // (a function, not a const: gpuUnfit runs before this line, AGENTS.md)
  function gpuMatrices() {
    return Object.fromEntries(Object.entries({ wq, wk, wv, wo, w1, w2, w3 }).filter(([, m]) => m));
  }
  // (T155: int6 weights too, widened to int8 on the GPU, and a model in a 64-bit memory: gpu.js; T232: ternary weights
  // as they are, in their groups of 128. The 27B, T233, is ternary in a rotated basis with linear-attention layers:
  // either of the first two lines keeps it on the CPU)
  function gpuUnfit() {
    if (linear) return "linear-attention layers are not on the GPU yet";  // T229
    if (convolution) return "convolution layers are not on the GPU yet";  // T260
    if (rotated) return "a rotated basis is not on the GPU yet";  // T237
    if (!sharedMemory) return "the page is not cross-origin isolated";
    if (headSize % 4) return "heads of a size that is no multiple of 4 are not on the GPU";
    if (!Object.values(gpuMatrices()).every((m) => m.int8 && m.group === (m.ternary ? 128 : 32))) return "float32 weights are not on the GPU yet";
    // T148: the layers twice, in this memory and on the GPU (T156 will keep one): a device with too little memory
    // for both keeps the CPU's alone (a phone or an Apple shares its memory between the two)
    // (T153, the review: with the GPU's own keys and values, float16, as the prompt may fill the whole context: Qwen3
    // 0.6B's are 0.47 GB at 4096 positions, 95% of its layers')
    const onGpu = layersOnGpu();
    if (gpuRoom !== undefined && onGpu > gpuRoom && !direct) {
      return `the layers on the GPU as well (${Math.round(onGpu / 1e6)} MB) would not leave this device enough memory`;
    }
    return null;
  }
  // the bytes of the layers on the GPU, and its own keys and values
  function layersOnGpu() {
    return Object.values(gpuMatrices()).reduce((bytes, m) => bytes + layers * m.rows * m.n * (m.ternary ? GPU_TERNARY_BYTES : GPU_WEIGHT_BYTES), 0) +
      Object.values(gpuVectors()).reduce((bytes, { size }) => bytes + layers * size * 4, 0) + 2 * layers * seqLen * kvDim * 2;
  }
  // T152: why a generation's steps stay on the CPU, or null. A step on the GPU is T150's and T175's fused layer (gpu.js):
  // Llama's (RMSNorm, RoPE on whole heads, SwiGLU), and T226: with what the prompt's blocks take besides (T153: Qwen2's
  // biases of q, k and v, Qwen3's norms of the heads and heads of another size than dim / heads; T154: GPT-2's and
  // GPT-NeoX's LayerNorm, biases, GELU, learned positions, RoPE on a part of a head, parallel residual), and T92's
  // outlier channels (the GPU's classifier multiplies floats for such a model, and needs no columns apart). The keys
  // and values are float16 as the GPU's, or float32 where the CPU keeps them so (T160, widened on the way back and
  // narrowed on the way up). What is left: a classifier
  // and an embedding of int8 or int6 in groups of 32 (T232: or ternary in groups of 128; of a ternary classifier the
  // GPU takes the outlier channels apart as this does, shaders.js's TAKE_OUTLIERS); and the memory for the classifier, the embedding where it is
  // another table, RoPE's table and the vocabulary's three arrays of the sampling, besides the layers
  function tokensUnfit() {
    const embedding = T.token_embedding_table, group = wcls?.ternary ? 128 : 32;
    if (!wcls?.int8 || wcls.group !== group || !(wcls.ternary ? ["ternary"] : ["int8", "int6"]).includes(embedding.kind) || embedding.group !== group) {
      return "a classifier of float weights is not on the GPU's tokens";
    }
    // T205: the classifier and the embedding on the GPU as well (llm-jp-3 150M's 73 MB of layers came to 189 MB) where
    // the browser does not say what the device has: an iPhone's tab went down in /benchmark/'s model section. The
    // prompts' blocks still go (their layers alone)
    if (memoryUnsaid) return "this browser does not say how much memory the device has";
    const table = vocab * dim * (wcls.ternary ? GPU_TERNARY_BYTES : GPU_WEIGHT_BYTES);
    // (GPT-2's positions on the GPU as well, a row a position)
    const onGpu = layersOnGpu() + table * (plan.shared_classifier ? 1 : 2) + seqLen * headSize * 4 + 3 * vocab * 4 + (positions ? seqLen * D : 0);
    if (gpuRoom !== undefined && onGpu > gpuRoom) {
      return `the classifier on the GPU as well (${Math.round(onGpu / 1e6)} MB with the layers) would not leave this device enough memory`;
    }
    return null;
  }
  // the vectors of every layer the GPU reads (gpu.js's plan.vectors): the norms' weights; T153: Qwen2's biases of q,
  // k and v, Qwen3's norms of a head of q and of k; T154: GPT-2's and GPT-NeoX's biases of the two LayerNorms, of q,
  // k and v and of o, w1 and w2. Each one's address here and its floats a layer
  function gpuVectors() {
    return { attention: { at: attW, size: dim }, ffn: { at: ffnW, size: dim },
      ...(layerNorm ? { attentionBias: { at: attB, size: dim }, ffnBias: { at: ffnB, size: dim } } : {}),
      ...(bq ? { bq: { at: bq, size: qDim }, bk: { at: bk, size: kvDim }, bv: { at: bv, size: kvDim } } : {}),
      ...(qNorm ? { qNorm: { at: qNorm, size: headSize }, kNorm: { at: kNorm, size: headSize } } : {}),
      ...(bo ? { bo: { at: bo, size: dim }, b1: { at: b1, size: hidden }, b2: { at: b2, size: dim } } : {}) };
  }

  // T135: where the GPU's worker puts the keys and values of a prompt's block (float16, T147: the keys
  // [layers][GPU_BLOCK][kvDim], then the values the same), for the cache below, and where it reads the block's rows
  // (dense). Only where the page asked for the GPU and it can take this model
  const gpuWhyNot = gpu ? gpuUnfit() : null;
  // (T210: a model on the GPU alone has its keys and values read back there only for the tests, keysAndValues; and the
  // GPU embeds its rows itself)
  const staging = gpu && !gpuWhyNot ? alloc(2 * layers * GPU_BLOCK * kvDim * 2) : 0;
  const gpuRows = staging && !direct ? alloc(GPU_BLOCK * D) : 0;
  // T152: why a generation's steps stay on the CPU where the prompt's blocks may go to the GPU (else null), and where
  // the GPU's worker writes the ids of the steps it took ([sampled, id, ...])
  const tokensWhyNot = staging ? tokensUnfit() : null;
  // (T219: and after the ids, the word that says the step after them was refused: its logits were not finite)
  const gpuIds = staging && !tokensWhyNot ? alloc((2 + GPU_TOKENS) * 4) : 0;

  // the KV cache: per layer [positions][kvDim], one block for the keys and one for the values, last in memory
  // so that growing it (KV_START, doubling) takes only the room it adds (T130): every layer's block moves up to where
  // it starts at the larger size, the last first, so that none is written over before it has moved (a block starts no
  // lower than it did, and past the end of the ones below it). Before, the larger blocks were made after the smaller
  // and moved down onto them: 1.5 times the context at the step to the whole of it, which a 32-bit memory at the edge
  // could not hold (Qwen2.5 3B's float32 keys and values on a memory that is not shared: 4.03 GiB, 3.89 now). T210:
  // none on the GPU alone, whose keys and values are the GPU's alone (no step runs here, and one that fails is loaded
  // again on the CPU)
  let capacity = direct ? 0 : Math.min(plan.kv_start, seqLen);
  // (T229: of the layers that attend; a Qwen3.5's linear-attention layers have none)
  let keys = alloc(2 * attending * capacity * KV), values = keys + attending * capacity * KV;
  function grow(pos) {
    const larger = Math.min(Math.max(2 * capacity, pos + 1), seqLen);
    const oldLayer = capacity * KV, newLayer = larger * KV;
    alloc(2 * attending * (newLayer - oldLayer));
    const newValues = keys + attending * newLayer;
    for (let l = attending - 1; l >= 0; l--) U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
    for (let l = attending - 1; l > 0; l--) U.copyWithin(keys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);
    values = newValues;
    capacity = larger;
  }
  const jobOf = (m, out, outStride, input, l, count) => {
    const [w, s, c] = m.layer(l);
    if (!m.int8) return [2, out, input, 0, w, 0, 0, m.n, 0, m.rows, count, outStride, S, 0];
    const kind = m.ternary ? (relaxed ? 8 : 9) : m.six ? (relaxed ? 5 : 6) : (relaxed ? 0 : 1);
    return [kind, out, xq, xs, w, s, relaxed && !m.ternary ? c : 0, m.n, 0, m.rows, count, outStride, S, S];
  };
  // the attention of token t of a run, at position pos (its scores have a place of their own: tokens run at once)
  const attentionJob = (t, pos, layerKeys, layerValues) =>
    [halfKV ? 4 : 3, xb + t * S, q + t * S, layerKeys, layerValues, att + t * A, pos, kvHeads, headSize, heads, 1, 0, 0, 0];
  const runRows = runner(k, relaxed);

  // ---- the matrix multiplications, in phases: the matmuls that read the same input (q, k and v; w1 and w3) go
  // out together. The input is quantized here, once; the rows are computed here and, with helper threads, by
  // whoever takes them. Every row is computed whole by one thread with the same kernel, so the numbers are the same
  // with any number of threads.
  //
  // A job is what jobs.js says: [kind, eight arguments, rows, count, out stride, a stride, b stride].
  const shared = sharedMemory && spawn;
  // the control area of a shared memory: the helpers' words, and the GPU's (T135: with or without helpers)
  const ctl = sharedMemory ? new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4) : null;
  const table = shared ? new Float64Array(memory.buffer, JOB_TABLE, BATCH * JOB) : null;  // the jobs (jobs.js)
  // the memory is kept from model to model (T96), and the control area with it: what the last engine's phases left
  // there (a helper counted in ACTIVE when it was ended, a generation) would hold this one's first phase for ever
  ctl?.fill(0);
  const helpers = [];
  let threads = 1, gen = 0;
  // stopThreads() runs on this thread too, never in the middle of a phase: a wait here ends when the helpers have done
  // their part. A helper the browser itself stopped (iOS may end a worker for its memory) never counts the chunk it
  // took: T120, the wait gives up when its count has not moved for stalledMs, and says false. (A check of a flag that
  // stopThreads() raised and lowered again stood here and could never be seen, the review of T96 found.)
  // A wait that took far longer than it asked for means this thread did not run either (a frozen tab, a phone that
  // suspended the page): the helpers were stopped with it, and the time from before does not count (the review of
  // T120: a stop of 10 s gave a thread up now and then as the page came back)
  // alive (T135): the word whose change counts as progress, where it is not the one waited on (the GPU's worker counts
  // one up while it works, and answers only at the end)
  const waitUntil = (index, done, alive = index) => {
    const tick = Math.min(1000, stalledMs);
    let moved = performance.now(), last = moved, beat = Atomics.load(ctl, alive);
    for (let seen = Atomics.load(ctl, index); !done(seen);) {
      Atomics.wait(ctl, index, seen, tick);
      const now = Atomics.load(ctl, index), beatNow = Atomics.load(ctl, alive), at = performance.now();
      if (at - last > 2 * tick) moved = at;
      last = at;
      if (now !== seen || beatNow !== beat) [seen, beat, moved] = [now, beatNow, at];
      else if (at - moved > stalledMs) return false;
    }
    return true;
  };
  // T120: every helper goes, and this engine keeps to one thread from here on. The phase is then run again here:
  // each chunk writes only its own rows, from inputs that no phase changes while it runs, so what the helpers did
  // before they stopped is written over with the same numbers
  let lost = false;
  function giveUp() {
    lost = true;
    console.warn("forward.js: a software thread stopped in the middle of its work; this model goes on with one thread");
    stopHelpers();
    search = null;
    chosen = 1;
  }
  function phase(jobs) {
    const alone = () => {
      for (const job of jobs) runRows(job, 0, job[ROWS]);
    };
    if (threads <= 1) return alone();
    // close the previous phase (odd), let every helper still awake leave it, then rewrite the jobs
    Atomics.store(ctl, GEN, gen + 1);
    if (!waitUntil(ACTIVE, (seen) => seen === 0)) {
      giveUp();
      return alone();
    }
    let total = 0;
    ctl[JOBS] = jobs.length;
    jobs.forEach((job, i) => {
      const at = i * JOB, rows = job[ROWS];
      const quad = job[COUNT] > 1 ? 4 : 1;  // T159: a prompt's chunks in fours of rows, the tiles of matmul_q8r_tile
      const size = quad * Math.ceil(rows / (threads * CHUNKS_PER_THREAD * quad));
      table.set(job, at);
      table[at + SIZE] = size;
      table[at + FIRST] = total;
      total += Math.ceil(rows / size);
    });
    ctl[TOTAL] = total;
    Atomics.store(ctl, COUNTER, 0);
    Atomics.store(ctl, FINISHED, 0);
    gen += 2;
    Atomics.store(ctl, GEN, gen);
    // wake helpers 1.. by name, no more than there are chunks for them and no more than this many threads
    for (let h = 1, wake = Math.min(threads - 1, total - 1); h <= wake; h++) {
      Atomics.store(ctl, WAKE + h, gen);
      Atomics.notify(ctl, WAKE + h, 1);
    }
    for (let c = Atomics.add(ctl, COUNTER, 1); c < total; c = Atomics.add(ctl, COUNTER, 1)) {
      let j = jobs.length - 1;
      while (table[j * JOB + FIRST] > c) j--;
      const at = j * JOB, size = table[at + SIZE], r0 = (c - table[at + FIRST]) * size;
      runRows(jobs[j], r0, Math.min(r0 + size, jobs[j][ROWS]));
      Atomics.add(ctl, FINISHED, 1);
    }
    if (!waitUntil(FINISHED, (seen) => seen === total)) {
      giveUp();
      alone();
    }
  }

  // ---- the number of threads (stage 2b): found by measuring, never written down. The search starts from a hint
  // (navigator.hardwareConcurrency, which counts the little cores of a big.LITTLE phone too) and compares the best
  // count so far with half of it and, if half is not faster, with twice as many; it goes on in that direction while
  // the other is faster by more than the noise of a run, and stops at the first that is not (T239: on the way down, at
  // the second in a row that is not: a quarter is compared where half was not faster). Only the tokens that
  // make logits are timed (a prompt's tokens skip the classifier). One comparison runs the two counts in blocks,
  // best-candidate-candidate-best, so that the growing cost of later positions falls on both alike, and drops the
  // first token of every block (the switch). Helpers that a count needs are started in the background; until they
  // are ready the tokens run on the best count and are not timed.
  // T199: a count's time is the lower median of its 8 (two blocks of 4). What else runs (a collection of the garbage,
  // another tab, the page on its way to the background) slows a block, never speeds one up, so one block slowed whole
  // leaves the other block's 4 below it and the verdict stands. The upper median it was took that block's time: twice
  // in CI 2 threads 1.24 to 1.34 times as fast as 1 lost to it (T190's review).
  // T223: a count is remembered only where a search timed it with nothing of the GPU's getting ready beside it (its
  // upload, its shaders checked against JavaScript and timed use the CPU and the memory: T148 does not time the CPU's
  // prompts then either). A search on the first texts goes on while the GPU gets ready (the first visit is not left on
  // the logical cores meanwhile), but its verdict is used only until the GPU is ready: the first generation after that
  // searches again from it, and that search's verdict is the one remembered. A count remembered from an earlier visit
  // is searched again the same way once a visit (the first generation with no GPU getting ready), not only every
  // recheck generations of one load: the owner's Android kept 4 threads for llm-jp-3 150M where 2 wrote 3.2 times as
  // fast (T223), and a visit seldom writes 8 answers. unchecked: the count in use is not such a search's verdict yet.
  // T239: half may be a dip with a faster count below it. The owner's PC (16 logical cores) stopped at 8 threads ("16 or
  // 8: 8, 8 or 4: 8") where 2 wrote faster, and the search of every later visit began from that 8 and ended on it. So a
  // count that half did not beat is compared with a quarter of it too (far), and the way down goes on by halves from a
  // quarter that is faster. A comparison more (20 tokens) where the best count is 4 or more and nothing below it is
  // faster; none more where it is 1 or 2. Not on the way up: a visit that remembers 2 would time 8 threads every time
  // (the owner's Android: 0.23 s), and no device's report has a dip above its count (TODO.md's T239 has the table).
  // T240: the search the count in use is owed (unchecked) does not wait for the next generation where the GPU is ready
  // inside one: it begins at the first token after that (forward() below), so that a long first answer is not written
  // to its end on a count timed beside the GPU's getting ready. Only where the page began a generation: /benchmark/
  // begins none and takes the count the model page remembers as it is (T190).
  const BLOCK = 4;
  let search = null, chosen = 0, generations = 0, recheckEvery = 0, onChosen = null, onCompared = null, unchecked = false;
  // whether a search from the count in use may begin now (none under way, nothing of the GPU's getting ready beside it)
  const mayRecheck = () => !search && chosen && recheckEvery && !gpuGettingReady();
  const searchLog = [];  // every comparison: the counts, their times in ms per token, and the verdict
  function beginSearch(from) {
    search = { best: Math.max(1, from), direction: from > 1 ? "down" : "up", moved: false, far: false, candidate: 0, times: null, step: 0, waiting: false, whileGpu: false };
    nextCandidate();
  }
  function nextCandidate() {
    if (lost) return finish();
    const { best, direction, far } = search;
    const candidate = direction === "down" ? Math.floor(best / (far ? 4 : 2)) : best * 2;
    if (candidate < 1) return passed();
    search.candidate = candidate;
    search.times = { [best]: [], [candidate]: [] };
    search.step = 0;
    if (helpers.length < candidate - 1) {
      search.waiting = true;
      ensureHelpers(candidate).then((complete) => {
        if (!search) return;
        if (complete) search.waiting = false;
        else finish();
      }, () => finish());
    }
  }
  // the candidate was not faster than the best, or there is no count there: the next one, or the end (T239: a quarter
  // after half; then twice as many, where the best is still the count the search began from)
  function passed() {
    if (search.direction === "down" && !search.far) search.far = true;
    else if (search.direction === "down" && !search.moved) search.direction = "up";
    else return finish();
    nextCandidate();
  }
  function finish() {
    chosen = lost ? 1 : search ? search.best : threads;
    threads = chosen;
    unchecked = !lost && Boolean(search?.whileGpu);  // T223: searched again once the GPU is ready, and remembered then
    search = null;
    // one thread after a give-up says nothing about the device: the page would start with it next time (the review of T120)
    if (!lost && !unchecked) onChosen?.(chosen);
  }
  // the count for the next token, and whether it is timed
  function countForToken() {
    if (lost) return [1, false];
    if (!search || search.waiting) return [search ? search.best : threads, false];
    const order = [search.best, search.candidate, search.candidate, search.best];
    const block = Math.floor(search.step / (BLOCK + 1)), inBlock = search.step % (BLOCK + 1);
    return [order[block], inBlock > 0];
  }
  function recordToken(count, milliseconds, timed) {
    if (!search || search.waiting) return;
    if (timed) {
      search.times[count].push(milliseconds);
      if (gpuGettingReady()) search.whileGpu = true;
    }
    search.step += 1;
    if (search.step < 4 * (BLOCK + 1)) return;
    const { best, candidate } = search;
    const bestMs = lowerMedian(search.times[best]), candidateMs = lowerMedian(search.times[candidate]);
    const faster = candidateMs < bestMs * BETTER;
    const { whileGpu } = search;
    searchLog.push({ best, candidate, times: search.times, faster, whileGpu });
    // T114: every verdict, so that a device's choice can be followed afterwards (the page writes it to the console)
    onCompared?.({ best, candidate, bestMs, candidateMs, faster, whileGpu, tokens: search.times[best].length + search.times[candidate].length });
    if (!faster) return passed();
    Object.assign(search, { best: candidate, moved: true, far: false });
    nextCandidate();
  }
  // the helpers in QUIT's hands: set, every one woken to see it, and each ended
  let stops = 0;  // stopHelpers() counts them: a helper whose start began before one is not kept
  function stopHelpers() {
    stops += 1;
    Atomics.store(ctl, QUIT, 1);
    for (let h = 1; h <= helpers.length; h++) {
      Atomics.add(ctl, WAKE + h, 2);
      Atomics.notify(ctl, WAKE + h);
    }
    // QUIT stays set until the next ensureHelpers(): a helper that wakes late must still see it
    helpers.splice(0).forEach((helper) => helper.terminate?.());
    threads = 1;
  }
  // Resolves to whether the n threads are there. A helper that becomes ready after this engine let its helpers go
  // (release(), stopThreads(), giveUp()) is ended at once: the review of T120 found such ones left alive when the
  // visitor chose another model while the search started more, and the next model's engine on the same memory
  // (T96) woke them, with the old engine's kernels (one threw holding a chunk: 10 s still, then one thread)
  async function ensureHelpers(n) {
    if (lost) return false;  // T120: none again after a helper stopped under this engine
    const since = stops;
    if (helpers.length < n - 1 && helpers.length === 0) Atomics.store(ctl, QUIT, 0);  // after stopThreads(): a fresh start
    while (helpers.length < n - 1) {
      const helper = await spawn({ memory, wide, plain: kernels.plain, relaxed: plan.int8 && plan.relaxed ? kernels.relaxed : null,
        share: helpers.length + 1 });
      if (lost || stops !== since) {
        helper.terminate?.();
        return false;
      }
      helpers.push(helper);
    }
    return true;
  }
  // T349: what the rest of the engine calls of the phases and the software threads, and reads and sets of their search
  const pool = {
    shared, ctl, waitUntil, phase, mayRecheck, searchLog, beginSearch, countForToken, recordToken, stopHelpers,
    ensureHelpers,
    get threads() { return threads; }, set threads(to) { threads = to; },
    get lost() { return lost; },
    get search() { return search; }, set search(to) { search = to; },
    get chosen() { return chosen; }, set chosen(to) { chosen = to; },
    get generations() { return generations; }, set generations(to) { generations = to; },
    get recheckEvery() { return recheckEvery; }, set recheckEvery(to) { recheckEvery = to; },
    get onChosen() { return onChosen; }, set onChosen(to) { onChosen = to; },
    get onCompared() { return onCompared; }, set onCompared(to) { onCompared = to; },
    get unchecked() { return unchecked; }, set unchecked(to) { unchecked = to; },
  };
  // matmuls of one input (count tokens of it, a frame apart): [matrix, output, output stride, layer] each
  // T237: in a rotated basis the matrices read R of the input, turned here once for all of them (a kernel on this
  // thread: 5 blocks of 1024 take a few microseconds, a thousandth of the rows they are multiplied by)
  function matmuls(input, count, list) {
    if (rotated) {
      const n = list[0][0].n;
      for (let t = 0; t < count; t++) k.rotate(xr + t * S, input + t * S, signs[n], n, rotated);
      input = xr;
    }
    const [first] = list[0];
    if (first.ternary) {
      // T231: the ternary kernels take the activations signed, in all 8 bits, and laid out as the weights' planes are
      for (let t = 0; t < count; t++) {
        k.quantize_x(xq + t * S, xs + t * S, input + t * S, first.n, 0);
        k.interleave(xq + t * S, xs + t * S, first.n);
      }
    } else if (first.int8) {
      for (let t = 0; t < count; t++) k.quantize_x(xq + t * S, xs + t * S, input + t * S, first.n, bias);
    }
    phase(list.map(([m, out, outStride, l]) => jobOf(m, out, outStride, input, l, count)));
  }

  // the embedding rows of tokens at positions pos0, pos0 + 1, ... into x, a frame apart (or into rows, stride apart)
  function embed(tokens, pos0, rows = x, stride = S) {
    for (let t = 0; t < tokens.length; t++) {
      const row = tokens[t] * dim, to = (rows + t * stride) / 4;
      if (embedding.kind === "int8" || embedding.kind === "int6" || embedding.kind === "ternary") {
        const g = embedding.group;
        for (let i = 0; i < dim; i++) {
          F[to + i] = Math.fround(weightAt(embedding, row + i) * F[(base + embedding.scales) / 4 + (((row + i) / g) | 0)]);
        }
      } else {
        const from = embedding.kind === "f32" ? base + embedding.offset : embeddingRows;
        F.copyWithin(to, from / 4 + row, from / 4 + row + dim);
      }
      if (positions) k.add_inplace(rows + t * stride, positions + (pos0 + t) * D, dim);
      // T237: the table of a rotated basis holds rotated rows
      if (rotated) k.unrotate(rows + t * stride, rows + t * stride, signs[dim], dim, rotated);
    }
  }
  // T147: a token's key and value in float16 (the GPU's) into the cache at keyAt and valueAt. T160: a float32 cache
  // (a grouped-query model's) widens them with a kernel, as attention_f16 reads them (halfToFloat one at a time took
  // 27 to 33 ns a value on CI's runners: 1.7 ms a token of Qwen3 0.6B's prompt)
  function cacheHalves(keyAt, valueAt, key, value) {
    if (halfKV) {
      U.copyWithin(keyAt, key, key + kvDim * 2);
      U.copyWithin(valueAt, value, value + kvDim * 2);
    } else {
      k.from_f16(keyAt, key, kvDim);
      k.from_f16(valueAt, value, kvDim);
    }
  }
  // T147: the keys and values of count positions from pos that the GPU's worker put into staging (float16, [keys,
  // values][layer][GPU_BLOCK positions]), into the cache
  function fromStaging(pos, count) {
    for (let l = 0; l < layers; l++) {
      const layerKeys = keys + l * capacity * KV, layerValues = values + l * capacity * KV;
      for (let t = 0; t < count; t++) {
        cacheHalves(layerKeys + (pos + t) * KV, layerValues + (pos + t) * KV,
          staging + (l * GPU_BLOCK + t) * kvDim * 2, staging + ((layers + l) * GPU_BLOCK + t) * kvDim * 2);
      }
    }
  }
  // T243: whether the keys and values of count positions in staging are all finite numbers. The CPU's readers of a
  // float16 (the kernels' halves4, and so from_f16 and attention_f16) make a finite number of a NaN and of an infinity
  // (65536 and more), so that what a GPU computed wrong would be read from the cache as numbers ever after, by the CPU's
  // steps and its logits (T195 sees nothing then). They are looked at by their bits (finite_f16: the exponent's five)
  // before anything of them is written: one pass of a kernel over what fromStaging reads, none in the attention's loops
  function stagingFinite(count) {
    for (let part = 0; part < 2 * layers; part++) {
      if (!k.finite_f16(staging + part * GPU_BLOCK * kvDim * 2, count * kvDim)) return false;
    }
    return true;
  }
  // T243: what the status line says of a GPU stopped for them (where: the request)
  const notFiniteKV = (where) => `the GPU computed keys or values that are not finite numbers (NaN or infinity) ${where}`;
  // a token's key and value (float32, at key and value) into the cache at keyAt and valueAt
  function cache(keyAt, valueAt, key, value) {
    if (halfKV) {
      k.to_f16(keyAt, key, kvDim);
      k.to_f16(valueAt, value, kvDim);
    } else {
      U.copyWithin(keyAt, key, key + KF);
      U.copyWithin(valueAt, value, value + KF);
    }
  }

  // one token (count 1) or up to BATCH tokens of a prompt at positions pos0, pos0 + 1, ... (T108). Every token is
  // computed as it would be alone: the same kernels on the same numbers, only the matmuls of a layer go out once for
  // all of them. The logits, if asked for, are the last token's.
  function run(tokens, pos0, needLogits) {
    // T156: the layers are on the GPU alone: a block or a step it did not take cannot run here
    if (direct) {
      if (!gpuPart.directLost) gpuPart.stopGpu("the GPU did not take a block or a step");  // (the worker loads the model again on the CPU)
      throw Object.assign(new Error(`The GPU stopped (${gpuPart.directLost}), and this model's weights were on it alone`), { directLost: true });
    }
    const count = tokens.length;
    if (pos0 + count - 1 >= capacity) grow(pos0 + count - 1);
    views();
    gpuPart.gpuEnd = Math.min(gpuPart.gpuEnd, pos0);  // T135: from here on the cache holds what the GPU does not
    if (convBytes) follow(pos0);
    embed(tokens, pos0);
    for (let l = 0; l < layers; l++) {
      // (a: the layer's place among the layers of its kind, T229: the layer itself where all attend)
      const a = placeOf[l];
      for (let t = 0; t < count; t++) {
        if (layerNorm) k.layernorm(xb + t * S, x + t * S, attW + l * D, attB + l * D, dim);
        else k.rmsnorm(xb + t * S, x + t * S, attW + l * D, dim, eps);
        if (parallel) F.copyWithin((before + t * S) / 4, (x + t * S) / 4, (x + t * S) / 4 + dim);  // GPT-NeoX reads this layer's input twice
      }
      if (lines[l]) {
        if (linear) linearAttention(a, count);
        else shortConvolution(a, count);
      } else {
        const layerKeys = keys + a * capacity * KV, layerValues = values + a * capacity * KV;
        const kp = layerKeys + pos0 * KV, vp = layerValues + pos0 * KV;
        // (a Qwen3.5's gate of the attention's output is one more matrix of the same input)
        matmuls(xb, count, wg ? [[wq, q, S, a], [wk, kNow, S, a], [wv, vNow, S, a], [wg, gate, S, a]]
          : [[wq, q, S, a], [wk, kNow, S, a], [wv, vNow, S, a]]);
        for (let t = 0; t < count; t++) {
          const qt = q + t * S, kt = kNow + t * S, vt = vNow + t * S, pos = pos0 + t;
          if (bq) {
            k.add_inplace(qt, bq + a * QF, qDim);
            k.add_inplace(kt, bk + a * KF, kvDim);
            k.add_inplace(vt, bv + a * KF, kvDim);
          }
          if (qNorm) {
            const HS = headSize * 4;
            for (let h = 0; h < heads; h++) k.rmsnorm(qt + h * HS, qt + h * HS, qNorm + a * HS, headSize, eps);
            for (let h = 0; h < kvHeads; h++) k.rmsnorm(kt + h * HS, kt + h * HS, kNorm + a * HS, headSize, eps);
          }
          if (turns[l]) {
            const cos = cosTable + pos * (headSize / 2) * 4, sin = sinTable + pos * (headSize / 2) * 4;
            k.rope(qt, cos, sin, heads, headSize, rotary);
            k.rope(kt, cos, sin, kvHeads, headSize, rotary);
          }
          cache(kp + t * KV, vp + t * KV, kt, vt);  // into the cache, at this token's position
        }
        // the keys and values of positions up to each token's are all there now: its own and the ones before it.
        // The heads of every token go out as one phase (T109).
        phase(tokens.map((_, t) => attentionJob(t, pos0 + t, layerKeys, layerValues)));
        if (wg) for (let t = 0; t < count; t++) k.gate(xb + t * S, xb + t * S, gate + t * S, qDim);
        matmuls(xb, count, [[wo, xb2, S, a]]);
      }
      for (let t = 0; t < count; t++) k.add_inplace(x + t * S, xb2 + t * S, dim);
      if (layerNorm) {
        for (let t = 0; t < count; t++) {
          k.add_inplace(x + t * S, bo + l * D, dim);
          k.layernorm(xb + t * S, (parallel ? before : x) + t * S, ffnW + l * D, ffnB + l * D, dim);
        }
        matmuls(xb, count, [[w1, hb, S, l]]);
        for (let t = 0; t < count; t++) k.gelu(hb + t * S, hb + t * S, b1 + l * HD, hidden);
        matmuls(hb, count, [[w2, xb2, S, l]]);
        for (let t = 0; t < count; t++) {
          k.add_inplace(x + t * S, xb2 + t * S, dim);
          k.add_inplace(x + t * S, b2 + l * D, dim);
        }
        continue;
      }
      for (let t = 0; t < count; t++) k.rmsnorm(xb + t * S, x + t * S, ffnW + l * D, dim, eps);
      matmuls(xb, count, [[w1, hb, S, l], [w3, hb2, S, l]]);
      for (let t = 0; t < count; t++) k.swiglu(hb + t * S, hb + t * S, hb2 + t * S, hidden);
      matmuls(hb, count, [[w2, xb2, S, l]]);
      for (let t = 0; t < count; t++) k.add_inplace(x + t * S, xb2 + t * S, dim);
    }
    if (convBytes) stateAt = pos0 + count;
    if (!needLogits) return;
    const last = x + (count - 1) * S;
    if (layerNorm) k.layernorm(xb, last, finalW, finalB, dim);
    else k.rmsnorm(xb, last, finalW, dim, eps);
    channels.forEach((c, i) => {  // T92: the outlier channels are multiplied apart
      F[picked / 4 + i] = F[xb / 4 + c];
      F[xb / 4 + c] = 0;
    });
    matmuls(xb, 1, [[wcls, logits, 0, 0]]);
    if (channels.length) k.add_columns(logits, columns, picked, channels.length, vocab);
  }
  const forward = (token, pos, needLogits) => run([token], pos, needLogits);

  // T229: the linear-attention layers' state is what the tokens before this position left: position 0 clears it, and
  // any other has to be the one that comes next (llama2_numpy's follow()). Keys and values could be written again at
  // any position; a state cannot, and tokens out of turn would compute on the wrong one without a word. A run that
  // stopped half way leaves it no token's: only position 0 goes on from there. T260: an LFM2's convolution layers' rows
  // the same.
  function follow(pos) {
    if (pos === 0) {
      if (linear) F.fill(0, states / 4, (states + 2 * lineCount * stateBytes) / 4);
      F.fill(0, convRows / 4, (convRows + lineCount * convBytes) / 4);
    } else if (pos !== stateAt) {
      throw new Error(`This model keeps a state from token to token: position ${stateAt < 0 ? 0 : stateAt} comes next ` +
        `(or 0, to begin again), not ${pos}.`);
    }
    stateAt = -1;
  }
  // T229: count tokens through the a-th Gated DeltaNet layer, from xb (the norm of x) into xb2, as the output
  // projection of an attending layer leaves it (llama2_numpy's linear_attention() has the rule). The matrices go out
  // once for all the tokens, as an attending layer's; the rest is a token at a time, for the state after a token is
  // what the next one reads. The value heads of a token are one phase, shared out as the heads of an attention.
  function linearAttention(a, count) {
    const C = mixedWidth * 4, rows = convRows + a * convBytes, newest = rows + (linear.conv - 1) * C;
    const KD = keyDim * 4, VD = valueDim * 4, l2 = 1e-6 / keyDim;
    matmuls(xb, count, [[wqkv, mixed, S, a], [wz, z, S, a]]);
    for (let t = 0; t < count; t++) {
      const xt = xb + t * S, ct = conv + t * S, wt = work + t * S;
      // beta = sigmoid(wb x) and decay = exp(decay * softplus(wa x + dt_bias)), one of each a value head, before
      // what the layer reads goes where its input is. A few numbers a layer: JavaScript's own exp and log
      k.matmul_f32(wt, xt, wb + a * valueHeads * D, dim, 0, valueHeads);
      k.matmul_f32(wt + valueHeads * 4, xt, wa + a * valueHeads * D, dim, 0, valueHeads);
      for (let h = 0; h < valueHeads; h++) {
        const b = wt / 4 + h, g = b + valueHeads, at = a * valueHeads + h;
        const step = F[g] + F[dtBias / 4 + at];
        F[b] = 1 / (1 + Math.exp(-F[b]));
        F[g] = Math.exp(F[decays / 4 + at] * (step > 20 ? step : Math.log1p(Math.exp(step))));
      }
      // this token's q, k and v behind those of the conv - 1 tokens before it, and the convolution over them
      F.copyWithin(rows / 4, (rows + C) / 4, (rows + convBytes) / 4);
      F.copyWithin(newest / 4, (mixed + t * S) / 4, (mixed + t * S + C) / 4);
      k.convolve(ct, taps + a * convBytes, rows, mixedWidth, linear.conv);
      for (let h = 0; h < keyHeads; h++) {
        k.rmsnorm(ct + h * KD, ct + h * KD, qUnit, keyDim, l2);
        k.rmsnorm(ct + (keyHeads + h) * KD, ct + (keyHeads + h) * KD, kUnit, keyDim, l2);
      }
      const now = states + (2 * a + flips[a]) * stateBytes, next = states + (2 * a + 1 - flips[a]) * stateBytes;
      phase([[7, xt, now, next, ct, wt, keyHeads, keyDim, valueDim, valueHeads, 1, 0, 0, 0]]);
      flips[a] ^= 1;
      // the norm of every value head, and z's gate
      for (let h = 0; h < valueHeads; h++) k.rmsnorm(xt + h * VD, xt + h * VD, deltaNorm + a * VD, valueDim, eps);
      k.swiglu(xt, z + t * S, xt, readWidth);
    }
    matmuls(xb, count, [[wout, xb2, S, a]]);
  }
  // T260: count tokens through the a-th convolution layer of an LFM2, from xb (the norm of x) into xb2
  // (llama2_numpy's short_convolution() has the rule). The two matrices go out once for all the tokens, as an attending
  // layer's; between them a token at a time, for the rows after a token are what the next one reads: the rows move up
  // by one, and the kernel writes this token's (B * z) and the convolution times C over xb. Only this thread touches
  // the rows, outside every phase: a phase that is run again (T120) reads what it read.
  function shortConvolution(a, count) {
    const rows = convRows + a * convBytes;
    matmuls(xb, count, [[win, mixed, S, a]]);
    for (let t = 0; t < count; t++) {
      F.copyWithin(rows / 4, (rows + D) / 4, (rows + convBytes) / 4);
      k.short_conv(xb + t * S, taps + a * convBytes, rows, mixed + t * S, dim, convolution.taps);
    }
    matmuls(xb, count, [[wout, xb2, S, a]]);
  }
  // T349: what the GPU's side reads of the views of this memory and of the cache of the keys and values, as they are now
  const held = {
    get H() { return H; },
    get capacity() { return capacity; },
    get keys() { return keys; },
    get values() { return values; },
  };

  // ---- T135: a prompt on the GPU. gpu() makes the GPU's worker (gpu.js), which puts this model's layers on the GPU
  // once (a second copy of them: the CPU keeps its own, for the tokens after the prompt), and then runs each block of a
  // prompt that forwardMany() hands it through all the layers in one submission, while this thread waits for its
  // answer in the control area (Python calls forwardMany() and cannot wait for a promise; the GPU's answer is one). The
  // GPU multiplies a block's tokens by each weight it reads once (a matrix by a matrix: T134's "A prompt"). It writes the
  // keys and values of the block back (staging), and they go into the cache here as if the CPU had computed them: the
  // prompt's last token, with its logits, and every token after it stay the CPU's (T94: one token at a time the GPU was
  // slower on the owner's three devices). A model it does not take, or any failure, leaves the prompt on the CPU, said
  // once. gpuEnd: the positions up to which the GPU's own keys and values are the cache's (a block that begins after it
  // goes to the CPU; run() ends it where the CPU writes).
  // T148: by default (AGENTS.md's policy 9), and without waiting for it: the model is ready on the CPU at once, and the
  // GPU takes the blocks of a prompt from the first one after it is ready (a block past gpuEnd stays on the CPU, so a
  // GPU that is ready in the middle of a prompt changes nothing of it). Which blocks it takes: promptTimes above.
  // T205: gpuEnded, settled once the last GPU's worker made has let go of its buffers and its device (or never began)
  let gpuEnded = Promise.resolve(), gpuLast = null, quietTimer;
  let gpuWorker = null, gpuOn = false, gpuEnd = 0, gpuSerial = 0, gpuTokens = 0, settleGpu = null, gpuChosen = null;
  const gpuGettingReady = () => settleGpu !== null;
  // T148: what the status line says of the GPU now, and whether this generation checks the side not chosen again
  // recheck: the side not chosen that a part of the next long prompt goes to, to time it again ("cpu": the last
  // 2 × BATCH tokens of the prompt, "gpu": its first GPU_BLOCK; a GPU, then the CPU, is the order a prompt may take),
  // every GPU_RECHECK generations after the first verdict (sinceCheck), until it is timed (the review of T148: every
  // block on the other side put a GPU faster from 17 to 64 tokens on blocks of 16, 1.47 times as long, and never
  // timed the CPU). written: the positions this generation has filled
  let gpuStatus = gpu ? null : undefined, recheck = null, sinceCheck = 0, written = 0;
  // T184: why the prompts stay on the CPU (the model, the device, a failure), once it is known; and the side the
  // benchmark puts every block of a prompt on ("cpu", "gpu"), or null: the choice above
  let gpuReason = null, gpuSide = null;
  // T156: why the GPU stopped under a model on the GPU alone, once it has
  let directLost = null;
  const times = promptTimes();
  // T152: a generation's steps. tokensOn: the GPU takes them where it is faster (steps: their times); tokenStatus: what
  // the status line says of them (gpuLine's answers); tokenRecheck, cpuRecheck, sinceTokens: the side not chosen timed again now and then
  // (the CPU's first TOKEN_RECHECK steps of a generation, or the GPU's first run), every GPU_RECHECK generations after
  // the first verdict, as a prompt's; gpuSampled: the steps the GPU took since the generation began
  let tokensOn = false, tokenStatus = null, tokensReason = tokensWhyNot, tokenRecheck = null, cpuRecheck = 0, sinceTokens = 0, gpuSampled = 0;
  const steps = tokenTimes();
  // what the status line says of the GPU: of the prompts, and of the answers where there is something to say (gpuLine)
  const statusNow = () => gpuLine(gpuStatus, tokenStatus);
  const gpuNote = !gpu ? undefined : new Promise((resolve) => {
    settleGpu = (note) => {
      settleGpu = null;
      gpuStatus = note;
      resolve(statusNow());
    };
    // (T156: a model on the GPU alone that the GPU cannot take, or whose steps it cannot, is lost at once)
    const why = gpuWhyNot ?? (direct ? tokensWhyNot : null);
    if (why) stopGpu(why);
    else startGpu();
  });
  // T152: what gpu.js takes for a generation's steps: the classifier and (where it is another table) the embedding,
  // { rows, n, six, ternary (T232), at: [values, scales] }, the final norm's weights, where the ids go, and the steps a submission.
  // T226: the final LayerNorm's bias and GPT-2's positions (float32, a row a position), 0 where the model has none; and
  // whether its classifier has outlier channels (T92), and (T232) which they are
  function gpuTokensPlan() {
    const embedding = T.token_embedding_table;
    return { classifier: { rows: wcls.rows, n: wcls.n, six: wcls.six, ternary: Boolean(wcls.ternary), at: wcls.layer(0).slice(0, 2) },
      embedding: plan.shared_classifier ? null
        : { rows: vocab, n: dim, six: embedding.kind === "int6", ternary: embedding.kind === "ternary", at: [base + embedding.offset, base + embedding.scales] },
      final: finalW, finalBias: finalB, positions, outliers: channels.length > 0, channels, ids: gpuIds, most: GPU_TOKENS };
  }
  function startGpu() {
    // the values of a head RoPE turns: all of them, GPT-NeoX's first rotary (T154), none of GPT-2's
    const turned = gpt2 ? 0 : plan.rotary > 0 && plan.rotary < headSize ? plan.rotary : headSize;
    // (six, T155: the values at a layer's address are int6, packed as llama2_numpy.pack6 packs them; ternary, T232:
    // two bits a weight, as llama2_numpy.pack_ternary packs them, which the GPU takes as they are)
    const matrices = Object.fromEntries(Object.entries(gpuMatrices()).map(([name, m]) =>
      [name, { rows: m.rows, n: m.n, six: m.six, ternary: Boolean(m.ternary), layers: Array.from({ length: layers }, (_, l) => m.layer(l).slice(0, 2)) }]));
    const listen = () => {
      clearTimeout(quietTimer);
      quietTimer = setTimeout(() => stopGpu(`the GPU said nothing for ${GPU_QUIET_MS / 1000} s`), GPU_QUIET_MS);
    };
    const worker = gpuLast = gpuWorker = gpu();
    let ended;
    gpuEnded = new Promise((resolve) => { ended = resolve; });
    gpuStatus = "prompts on the CPU while the GPU gets ready";
    worker.onmessage = ({ data }) => {
      // T205: its buffers and its device let go (gpu.js's end()), and nothing more from it after a stop
      if (data.type === "ended") return ended();
      if (worker !== gpuWorker) return;
      if (data.type === "progress") return listen();
      clearTimeout(quietTimer);
      if (data.type === "ready") {
        gpuOn = true;
        if (data.blocks.length) times.started(data.blocks);
        gpuChosen = { matrices: data.form, attention: data.attention, key: data.key, remembered: data.remembered, seconds: data.seconds };
        console.info(`gpu: ${data.adapter}: ${Math.round(data.bytes / 1e6)} MB of layers on it in ${data.seconds.toFixed(1)} s`);
        // T147: the matrices' shader this device runs fastest of those that are right here, and what the others came to
        const forms = data.forms.map((f) => `${f.name} ${f.none ?? (f.remembered ? "remembered" : f.ms ? `${f.ms.toFixed(1)} ms` : "untimed")}`).join("; ");
        const blocks = data.blocks.map(({ count, ms }) => `${count} tokens ${ms.toFixed(1)} ms`).join(", ");
        console.info(`gpu: the matrices by ${data.form}, the attention by ${data.attention} (a pass of the first layer by ${GPU_BLOCK} tokens: ${forms}; a whole block: ${blocks})`);
        // T152: a generation's steps, where gpu.js made them (else why not)
        if (data.tokens) {
          tokensOn = true;
          gpuChosen.tokens = data.tokens.form;
          gpuChosen.tokenAttention = data.tokens.attention;  // T224: the attention of a token chosen here (tests)
          gpuChosen.tablePieces = data.tokens.pieces;  // T209: the classifier in pieces past what the device binds
          if (data.tokens.ms !== undefined) steps.gpu(data.tokens.ms);
          tokenStatus = always ? "gpu" : "untimed";
          const kinds = data.tokens.forms.map((f) => `${f.name} ${f.none ?? (f.remembered ? "remembered" : f.ms ? `${f.ms.toFixed(2)} ms` : "untimed")}`).join("; ");
          const attentions = (data.tokens.attentions ?? []).map((a) => `${a.name} ${a.none ?? (a.ms ? `${a.ms.toFixed(2)} ms` : "untimed")}`).join("; ");
          console.info(`gpu: a token by ${data.tokens.form} (a step of a run of ${GPU_TOKENS}: ${kinds}), its attention by ${data.tokens.attention} (${attentions})`);
        } else if (data.tokensWhy) {
          tokensReason = data.tokensWhy;
          tokenStatus = "why";
          console.info(`gpu: answers on the CPU (${tokensReason})`);
        } else if (tokensReason) {
          // (T152's review: a model whose steps were not asked of the GPU says why too, in the console alone: the
          // owner's words have the status line say no reason for the answers)
          tokenStatus = "why";
          console.info(`gpu: answers on the CPU (${tokensReason})`);
        }
        // T156: a model on the GPU alone needs its steps there too; and where /benchmark/ measured the CPU here, its
        // step (the checkpoint read once at the CPU section's fastest, T157's estimate) must not be faster by more
        // than BETTER's margin (the owner's B: the CPU's side estimated, not measured, for a model that does not fit twice)
        if (direct && !data.tokens) return stopGpu(data.tokensWhy ?? tokensReason ?? "the GPU did not take the steps");
        // (the owner, 2026-09-27: the prompts as well, which the GPU runs about five times as fast, weighed by use)
        if (direct) {
          const block = data.blocks.find((b) => b.count === GPU_BLOCK);
          const verdict = aloneVerdict({ size: cpuReadBytes(direct.size, wo.ternary ? "ternary" : "int8"), layerWeights: direct.layerWeights, cpu: direct.cpu, usage: direct.usage,
            gpu: { stepMs: data.tokens.ms, promptMs: block ? block.ms / block.count : undefined } });
          if (verdict.cpuFaster) {
            direct.verdict = { key: data.key, cpu: direct.cpu };  // the page keeps it: the next load goes on the CPU at once
            return stopGpu(`the CPU as /benchmark/ measured it (${direct.cpu.GBps?.toFixed(1)} GB/s): ${verdict.cpu.toFixed(0)} ms ` +
              `against the GPU's ${verdict.gpu.toFixed(0)} ms for ${Number(direct.usage.prompt).toFixed(0)} tokens of prompts and ${Number(direct.usage.written).toFixed(0)} written`);
          }
        }
        settleGpu?.(always ? PROMPTS_GPU : PROMPTS_UNTIMED);
      } else if (data.type === "unusable") {
        stopGpu(data.reason);
      } else if (data.type === "failed") {
        console.warn(`gpu: ${data.reason}`);
      }
    };
    worker.onerror = (event) => {
      ended();  // a worker that did not start holds nothing
      if (worker === gpuWorker) stopGpu(`the GPU's worker did not start (${event.message ?? "an error"})`);
    };
    listen();
    // T154: LayerNorm's epsilon is the CPU's layernorm kernel's and NumPy's, 1e-5 (GPT-2's and GPT-NeoX's
    // layer_norm_epsilon); parallel: GPT-NeoX's parallel residual
    worker.postMessage({ type: "start", memory, plan: { dim, hidden, layers, heads, kvHeads, headSize, turned, unturned, seqLen,
      kvStart: plan.kv_start, eps: layerNorm ? 1e-5 : eps, layerNorm, parallel: Boolean(parallel), batch: GPU_BLOCK, matrices,
      vectors: gpuVectors(), rows: gpuRows, tokens: gpuIds ? gpuTokensPlan() : null,
      cos: cosTable, sin: sinTable, staging, force: gpuForce, remembered: gpuRemembered, direct: Boolean(direct),
      words: { done: GPU_DONE, failed: GPU_FAILED, beat: GPU_BEAT, wanted: GPU_WANTED } } });
  }
  // the prompt stays on the CPU from here on (why: what the console says, where it was on the GPU); the GPU's worker
  // lets go of the device and ends
  function stopGpu(why) {
    clearTimeout(quietTimer);
    if (gpuOn && why) console.warn(`gpu: ${why}: the prompts and the tokens go on on the CPU`);
    // T156: the layers were on the GPU alone: the worker loads the model again on the CPU (not when it is let go)
    if (direct && why && !directLost) {
      directLost = why;
      direct.onLost?.(why);
    }
    gpuOn = false;
    tokensOn = false;
    tokenStatus = null;
    if (ctl) Atomics.store(ctl, GPU_WANTED, 0);  // T147: a request still under way writes nothing now
    gpuWorker?.postMessage({ type: "stop" });
    gpuWorker = null;
    gpuReason = why ?? "the model was let go";
    const note = `prompts on the CPU (${gpuReason})`;
    if (settleGpu) settleGpu(note);
    else if (gpu) gpuStatus = note;
  }
  // T148: whether a block of count tokens at pos0 goes to the GPU: where the GPU holds the keys and values before it,
  // and is faster for count tokens than the CPU on the threads in use now (promptTimes); and the first whole block of
  // a prompt where the GPU is to be timed again (recheck)
  function gpuTakes(count, pos0) {
    if (!gpuOn || pos0 > gpuEnd || gpuSide === "cpu") return false;
    if (always || gpuSide === "gpu") return true;
    const known = times.of(count, pool.threads);
    if (!known) return false;  // the CPU is timed first, on this prompt
    return known.faster || (recheck === "gpu" && pos0 === 0 && count === GPU_BLOCK);
  }
  // T148: the verdict for a whole block, as a prompt begins: the status line has it (the fewest tokens the GPU takes,
  // rounded up to a block of the CPU's, and none said up to one: the line does not move with every prompt), and the
  // console says it where it changes. Until the CPU is timed, the line stays as the GPU left it
  function verdict() {
    const most = times.of(GPU_BLOCK, pool.threads), from = times.threshold(GPU_BLOCK, pool.threads);
    if (!most || always) return;
    const status = from > GPU_BLOCK ? PROMPTS_CPU
      : from > BATCH ? `prompts of ${Math.ceil(from / BATCH) * BATCH} tokens and more on WebGPU` : PROMPTS_GPU;
    if (status !== gpuStatus) {
      console.info(`gpu: a block of ${GPU_BLOCK} tokens: ${most.gpu.toFixed(1)} ms on the GPU, ${most.cpu.toFixed(1)} ms on the ` +
        `CPU (${pool.threads} thread${pool.threads > 1 ? "s" : ""}), the GPU from ${from > GPU_BLOCK ? "no count" : `${from} tokens`}: ${status}`);
    }
    gpuStatus = status;
  }
  // A block of a prompt (up to GPU_BLOCK tokens at pos0, pos0 + 1, ...) through the layers on the GPU: false where it
  // must go to the CPU instead (no GPU, keys and values the GPU does not have, a failure)
  function promptOnGpu(tokens, pos0) {
    const count = tokens.length, began = performance.now();
    if (!direct && pos0 + count - 1 >= held.capacity) grow(pos0 + count - 1);
    views();
    // the GPU reads the rows from there (T210: on the GPU alone, it embeds the tokens itself)
    if (!direct) embed(tokens, pos0, gpuRows, D);
    gpuSerial = ++gpuRequests;
    Atomics.store(ctl, GPU_WANTED, gpuSerial);
    gpuWorker.postMessage({ type: "prompt", serial: gpuSerial, count, pos: pos0, ...(direct ? { tokens } : {}) });
    if (!waitUntil(GPU_DONE, (seen) => seen === gpuSerial, GPU_BEAT)) {
      stopGpu(`the GPU's worker stopped answering for ${stalledMs / 1000} s`);
      return false;
    }
    if (Atomics.load(ctl, GPU_FAILED)) {
      stopGpu("the GPU failed on a block of the prompt");  // the GPU's worker said why in the console
      return false;
    }
    views();
    // T243: a key or a value that is no finite number: the block is refused as a step with an id outside the vocabulary
    // is (T219): nothing of it written, the GPU's positions not counted as the cache's, the GPU stopped, and the CPU
    // takes the block from its own keys and values (forwardMany). T210: a model on the GPU alone reads none back (its
    // keys and values stay on the GPU, where T219's flag on the logits of the steps is what guards them)
    if (!direct && !stagingFinite(count)) {
      stopGpu(notFiniteKV(`in a block of the prompt at position ${pos0}`));
      return false;
    }
    if (!direct) fromStaging(pos0, count);
    gpuEnd = pos0 + count;
    gpuTokens += count;
    times.gpu(count, performance.now() - began);
    if (recheck === "gpu") recheck = null;
    return true;
  }
  // T210: the GPU's own keys and values of count positions from pos (a model on the GPU alone keeps none here), read
  // back GPU_BLOCK positions a request through staging: in float32, [layers][count][kvDim] each, as keysAndValues
  function gpuKeysAndValues(pos, count) {
    const out = [0, 1].map(() => new Float32Array(layers * count * kvDim));
    for (let at = 0; at < count; at += GPU_BLOCK) {
      const part = Math.min(GPU_BLOCK, count - at);
      if (!gpuOn) throw new Error(`The GPU stopped (${directLost ?? gpuReason}), and this model's keys and values were on it alone`);
      gpuSerial = ++gpuRequests;
      Atomics.store(ctl, GPU_WANTED, gpuSerial);
      gpuWorker.postMessage({ type: "keys", serial: gpuSerial, count: part, pos: pos + at });
      if (!waitUntil(GPU_DONE, (seen) => seen === gpuSerial, GPU_BEAT) || Atomics.load(ctl, GPU_FAILED)) {
        throw new Error("The GPU did not read its keys and values back");
      }
      views();
      for (let side = 0; side < 2; side++) {
        for (let l = 0; l < layers; l++) {
          for (let t = 0; t < part; t++) {
            const from = (staging + ((side * layers + l) * GPU_BLOCK + t) * kvDim * 2) / 2, to = (l * count + at + t) * kvDim;
            for (let i = 0; i < kvDim; i++) out[side][to + i] = halfToFloat(held.H[from + i]);
          }
        }
      }
    }
    return { keys: out[0], values: out[1] };
  }
  // T148: a block of a prompt on the CPU, BATCH at a time (T108), each whole one timed where the GPU is there to
  // weigh against (not while it starts: its upload and compilation share the CPU and the memory)
  function promptOnCpu(tokens, pos0) {
    for (let i = 0; i < tokens.length; i += BATCH) {
      const piece = tokens.slice(i, i + BATCH), began = performance.now();
      run(piece, pos0 + i, false);
      if (gpuOn && piece.length === BATCH) {
        times.cpu(pool.threads, (performance.now() - began) / BATCH);
        if (recheck === "cpu") recheck = null;
      }
    }
  }

  // T152: the verdict for a generation's steps, as a generation begins: the status line has it (it changes only there),
  // and the console says it where it changes
  function tokenVerdict() {
    const known = tokensOn && !always ? steps.of(pool.threads) : null;
    if (!known) return;
    const status = known.faster ? "gpu" : "cpu";
    if (status !== tokenStatus) {
      console.info(`gpu: a step of a generation: ${known.gpu.toFixed(2)} ms on the GPU, ${known.cpu.toFixed(2)} ms on the CPU ` +
        `(${pool.threads} thread${pool.threads > 1 ? "s" : ""}): answers on ${known.faster ? "WebGPU" : "the CPU"}`);
    }
    tokenStatus = status;
  }
  // T152: whether the steps of a generation go to the GPU now (see tokenBlock)
  function gpuSteps() {
    if (!tokensOn || !gpuOn || gpuSide === "cpu") return false;
    if (always || gpuSide === "gpu") return true;
    if (pool.search) return false;
    if (tokenRecheck === "gpu") return true;
    if (tokenRecheck === "cpu") return false;
    return Boolean(steps.of(pool.threads)?.faster);
  }
  /** T152: count steps of generate() on the GPU from token at pos: the forward pass of each token and the sampling of
   * the next (the penalty over the last of history, whose length is length; temperature, topp; randoms: a number for
   * each step, the CPU's generator's, none where greedy; stops: the stop tokens). Returns the ids sampled (a stop
   * token last where one came), whose keys and values are in the cache then as the CPU would have written them; or
   * undefined where the GPU did not take them (the CPU takes the step instead: nothing of it was written). Not null:
   * Pyodide makes JavaScript's null jsnull, which is not None (T152's review), and undefined None */
  function generateMany(token, pos, history, length, count, temperature, topp, penalty, randoms, stops) {
    const list = (x) => (x?.toJs ? x.toJs() : [...(x ?? [])]);
    if (!gpuSteps() || count < 1 || count > GPU_TOKENS) return undefined;
    const stopList = list(stops);
    if (stopList.length > STOPS_MOST) {
      tokensOn = false;
      tokensReason = `more than ${STOPS_MOST} stop tokens`;
      tokenStatus = "why";
      console.info(`gpu: answers on the CPU (${tokensReason})`);
      return undefined;
    }
    const began = performance.now();
    if (!direct && pos + count - 1 >= held.capacity) grow(pos + count - 1);
    views();
    gpuSerial = ++gpuRequests;
    Atomics.store(ctl, GPU_WANTED, gpuSerial);
    // (T210: on the GPU alone, no cache here to go up from, nor any position the GPU does not hold)
    gpuWorker.postMessage({ type: "tokens", serial: gpuSerial, count, pos, from: direct ? pos : Math.min(gpuEnd, pos), token, history: list(history), length,
      cache: direct ? null : { keys: held.keys, values: held.values, capacity: held.capacity, row: KV, half: halfKV }, settings: { temperature, topp, penalty, stops: stopList }, randoms: list(randoms) });
    if (!waitUntil(GPU_DONE, (seen) => seen === gpuSerial, GPU_BEAT)) {
      stopGpu(`the GPU's worker stopped answering for ${stalledMs / 1000} s`);
      return undefined;
    }
    if (Atomics.load(ctl, GPU_FAILED)) {
      stopGpu("the GPU failed on a token");  // the GPU's worker said why in the console
      return undefined;
    }
    views();
    const words = new Int32Array(memory.buffer, gpuIds, 2 + GPU_TOKENS), sampled = words[0], notFinite = words[1 + GPU_TOKENS];
    if (notFinite) {
      // T219: the sampler refused the step after the `sampled` ones: its logits held a NaN or +inf, or none over
      // -3.4e38 (the State's not_finite word, set by the bits: WGSL lets a GPU take NaN and infinities as absent). The
      // whole request is refused as one with an id outside the vocabulary below is (nothing of it written, the ids
      // before it not taken), the GPU stops, and the CPU takes the step again and stops where its own logits are not
      // finite either (T195's NOT_FINITE). A model on the GPU alone cannot: it stops, said in words
      if (direct) throw new Error(OUTSIDE_VOCABULARY);
      stopGpu(`the GPU computed logits that are not finite numbers (NaN or infinity) at position ${pos + sampled}`);
      return undefined;
    }
    if (!(sampled >= 1 && sampled <= count)) {
      stopGpu(`the GPU sampled ${sampled} of ${count} tokens`);
      return undefined;
    }
    // T219: an id outside the vocabulary is what SAMPLE gives for logits that are not finite numbers (NONE, -1: T195,
    // WGSL lets a GPU take NaN and infinities as absent). The step is refused, nothing of it written: the CPU takes it
    // again and stops where its own logits are not finite either (T195's NOT_FINITE), and the GPU, whose numbers are
    // no longer trusted, stops here. A model on the GPU alone cannot take the step on the CPU: it stops, said in words
    const ids = Array.from(words.subarray(1, 1 + sampled));
    const outside = ids.find((id) => !(id >= 0 && id < vocab));
    if (outside !== undefined) {
      if (direct) throw new Error(OUTSIDE_VOCABULARY);
      stopGpu(`the GPU sampled ${outside}, outside the vocabulary of ${vocab}`);
      return undefined;
    }
    // the keys and values of the positions sampled, float16 in the staging place as a prompt's block's (T147), into the
    // cache (T160's review of T152: a float32 cache, a grouped-query model's, widens them as a prompt's). T210: none
    // on the GPU alone. T243: where one of them is no finite number, the whole request is refused as above (nothing
    // written, none of its ids taken): the CPU takes the step, from keys and values that are its own
    if (!direct && !stagingFinite(sampled)) {
      stopGpu(notFiniteKV(`in a step at position ${pos}`));
      return undefined;
    }
    if (!direct) fromStaging(pos, sampled);
    gpuEnd = pos + sampled;
    gpuSampled += sampled;
    if (sampled === count) steps.gpu((performance.now() - began) / count);
    if (tokenRecheck === "gpu") tokenRecheck = null;
    return ids;
  }
  // T349: what the rest of the engine calls of the GPU's side, and reads and sets of where it is
  const gpuPart = {
    gpuGettingReady, times, steps, statusNow, gpuNote, stopGpu, gpuTakes, verdict, promptOnGpu, gpuKeysAndValues,
    promptOnCpu, tokenVerdict, gpuSteps, generateMany,
    get gpuEnded() { return gpuEnded; },
    get gpuLast() { return gpuLast; },
    get gpuWorker() { return gpuWorker; },
    get gpuOn() { return gpuOn; },
    get gpuEnd() { return gpuEnd; }, set gpuEnd(to) { gpuEnd = to; },
    get gpuTokens() { return gpuTokens; }, set gpuTokens(to) { gpuTokens = to; },
    get gpuChosen() { return gpuChosen; },
    get recheck() { return recheck; }, set recheck(to) { recheck = to; },
    get sinceCheck() { return sinceCheck; }, set sinceCheck(to) { sinceCheck = to; },
    get written() { return written; }, set written(to) { written = to; },
    get gpuReason() { return gpuReason; },
    get gpuSide() { return gpuSide; }, set gpuSide(to) { gpuSide = to; },
    get directLost() { return directLost; },
    get tokensOn() { return tokensOn; },
    get tokensReason() { return tokensReason; },
    get tokenRecheck() { return tokenRecheck; }, set tokenRecheck(to) { tokenRecheck = to; },
    get cpuRecheck() { return cpuRecheck; }, set cpuRecheck(to) { cpuRecheck = to; },
    get sinceTokens() { return sinceTokens; }, set sinceTokens(to) { sinceTokens = to; },
    get gpuSampled() { return gpuSampled; }, set gpuSampled(to) { gpuSampled = to; },
  };

  let bound = null;
  const backend = (plan.int8 ? `SIMD kernels, ${["int6", "ternary"].includes(T.wo?.kind) ? T.wo.kind : "int8"}${relaxed ? ", relaxed SIMD" : ""}` : "SIMD kernels, float32") +
    (wide ? ", 64-bit memory" : "");  // T101: the status line says so, as it says every other way the model runs
  return {
    backend,
    /** Use n threads from the next token on (stage 2): starts the helpers that are missing. 1 on a memory that is
     * not shared. Resolves to the number in use. */
    async setThreads(n) {
      if (!shared || pool.lost || direct) return (pool.threads = 1);
      pool.search = null;
      if (!(await ensureHelpers(n))) return (pool.threads = 1);
      pool.threads = Math.max(1, n);
      return pool.threads;
    },
    /** Find the number of threads while generating (see above): from a hint, or from a count remembered from an
     * earlier visit, which is then only checked against its neighbours now and then (every recheck generations).
     * chose(count) is told the answer, compared(verdict) every comparison on the way. The helpers of the starting count are started (and warmed) before this
     * resolves, so the first tokens do not wait for them. */
    async findThreads({ from, remembered = 0, recheck = 8, chose, compared }) {
      if (!shared || pool.lost || direct) return 1;
      pool.onChosen = chose;
      pool.onCompared = compared;
      pool.recheckEvery = recheck;
      const start = Math.max(1, remembered || from);
      if (!(await ensureHelpers(start))) return (pool.threads = 1);
      pool.threads = start;
      if (remembered) {
        pool.chosen = remembered;
        pool.unchecked = true;  // T223: searched again from it on this visit
      } else {
        beginSearch(start);
      }
      return pool.threads;
    },
    /** the page starts a generation: now and then the remembered count is checked against its neighbours again (T223:
     * and first on the first generation of a load with no GPU getting ready, where the count in use is remembered from
     * an earlier visit or was found while the GPU got ready) */
    newGeneration() {
      gpuPart.gpuTokens = 0;
      pool.generations += 1;
      if (mayRecheck() && (pool.unchecked || pool.generations % pool.recheckEvery === 0)) beginSearch(pool.chosen);
      // T148: halfway between the threads' checks, a prompt goes to the side not chosen, so that its time stays
      // today's (a device that heats up, a GPU timed while the CPU was busy)
      gpuPart.written = 0;
      const whole = gpuPart.gpuOn && !always ? times.of(GPU_BLOCK, pool.threads) : null;
      if (whole && ++gpuPart.sinceCheck >= GPU_RECHECK && !gpuPart.recheck && !pool.search) {
        gpuPart.recheck = whole.faster ? "cpu" : "gpu";
        gpuPart.sinceCheck = 0;
        console.info(`gpu: the ${gpuPart.recheck === "cpu" ? "CPU on the end" : "GPU on the beginning"} of the next long prompt, to time it again`);
      }
      // T152: the same for the steps of a generation, the side not chosen taking its first ones
      gpuPart.gpuSampled = 0;
      const stepsNow = gpuPart.tokensOn && !always ? steps.of(pool.threads) : null;
      if (stepsNow && ++gpuPart.sinceTokens >= GPU_RECHECK && !gpuPart.tokenRecheck && !pool.search) {
        gpuPart.tokenRecheck = stepsNow.faster ? "cpu" : "gpu";
        gpuPart.cpuRecheck = TOKEN_RECHECK;
        gpuPart.sinceTokens = 0;
        console.info(`gpu: the ${gpuPart.tokenRecheck === "cpu" ? `CPU on the first ${TOKEN_RECHECK} steps` : "GPU on the first steps"} of the next generation, to time it again`);
      }
      tokenVerdict();
    },
    get searching() {
      return pool.search !== null;
    },
    searchLog,
    get threads() {
      return pool.threads;
    },
    /** the helper threads end; this engine runs on its own again */
    stopThreads() {
      if (shared) stopHelpers();
    },
    /** T120: whether a software thread stopped under this engine, which then went on with one */
    get lostThreads() {
      return pool.lost;
    },
    /** the float32 array of Python's that forward() fills with the logits */
    bind(array) {
      bound = array.copy ? array.copy() : array;
    },
    /** T108: tokens of a prompt at positions pos, pos + 1, ...: the same as forward() for each of them in turn without
     * logits, BATCH at a time through the layers. T135: on the GPU where it is on (see above), GPU_BLOCK at a time */
    forwardMany(tokens, pos) {
      const list = tokens.toJs ? tokens.toJs() : [...tokens];
      for (let at = 0; at < list.length;) {
        const block = list.slice(at, at + (gpuPart.gpuOn ? GPU_BLOCK : BATCH)), from = pos + at;
        if (from === 0) verdict();
        // T148: the CPU timed again on the last 2 × BATCH tokens of a prompt (its last block is the short one), the
        // rest of that block where it goes
        const tail = gpuPart.recheck === "cpu" && block.length < GPU_BLOCK ? Math.min(block.length, 2 * BATCH) : 0;
        const head = block.slice(0, block.length - tail);
        if (head.length && (!gpuTakes(head.length, from) || !promptOnGpu(head, from))) promptOnCpu(head, from);
        if (tail) promptOnCpu(block.slice(head.length), from + head.length);
        at += block.length;
      }
      gpuPart.written = pos + list.length;
    },
    /** T147: how many tokens of a prompt Python hands forwardMany() at once: GPU_BLOCK where the GPU takes a whole
     * block (T148: where it is faster), else BATCH (a longer call keeps the worker from answering for longer, and the
     * CPU gains nothing from it) */
    get promptBlock() {
      if (!gpuPart.gpuOn || gpuPart.gpuSide === "cpu") return BATCH;
      if (always || gpuPart.gpuSide === "gpu") return GPU_BLOCK;
      const whole = times.of(GPU_BLOCK, pool.threads);
      return whole && (whole.faster || (gpuPart.recheck === "gpu" && gpuPart.written === 0)) ? GPU_BLOCK : BATCH;
    },
    /** T147: the shaders the GPU multiplies a prompt's matrices and runs its attention with (tests) */
    get gpuForm() {
      return gpuPart.gpuChosen?.matrices;
    },
    get gpuAttention() {
      return gpuPart.gpuChosen?.attention;
    },
    /** T135: where the worker has WebGPU, a promise of what the status line says of the GPU (gpuLine: "WebGPU where it
     * is faster than the CPU", or on the CPU and why), settled once the layers are on the GPU or it is known that they
     * will not be. T148: nothing waits for it */
    gpu: gpuNote,
    /** T148: what the status line says of the GPU now (it changes with the times of the prompts), and what the GPU
     * chose as it started ({ matrices, attention, key, remembered, seconds }: the page shows it and remembers it) */
    get gpuStatus() {
      return statusNow();
    },
    get gpuReady() {
      return gpuPart.gpuOn ? gpuPart.gpuChosen : undefined;
    },
    /** T184: why the prompts stay on the CPU (no WebGPU, a fallback adapter, a model the GPU does not take yet, a
     * failure), or null while none is known */
    get gpuWhyNot() {
      return gpu ? gpuPart.gpuReason : "no WebGPU in a worker here";
    },
    /** T184 (the benchmark's page path): every block of a prompt on "cpu" or on "gpu" (where the GPU is on and holds
     * the keys and values before it, as gpuForce.always), or null: each where it is faster (promptTimes) */
    set gpuSide(side) {
      gpuPart.gpuSide = side === "cpu" || side === "gpu" ? side : null;
    },
    /** T135: the tokens of a prompt that went through the GPU since the generation began */
    get gpuTokens() {
      return gpuPart.gpuTokens;
    },
    /** T152: the steps of a generation the GPU took since it began */
    get gpuSampled() {
      return gpuPart.gpuSampled;
    },
    /** T152: why a generation's steps stay on the CPU (as gpuWhyNot, and the model's layer of a token, or the device's
     * memory for the classifier), or null */
    get gpuTokensWhyNot() {
      return gpu ? gpuPart.gpuReason ?? gpuPart.tokensReason : "no WebGPU in a worker here";
    },
    /** T152: whether this model's steps were asked of the GPU (the model and the memory allow them: tests) */
    get gpuTokensPlanned() {
      return Boolean(gpuIds);
    },
    /** T152: how many steps of a generation Python hands generateMany() at once now: GPU_TOKENS where the GPU takes
     * them (it is faster, or it is timed again), 0 where the CPU takes the step (the forward pass, and the sampling in
     * Python). Not while the threads are searched: that times the CPU's tokens */
    get tokenBlock() {
      return gpuSteps() ? GPU_TOKENS : 0;
    },
    /** T152: count steps of generate() on the GPU, or undefined where it did not take them (generateMany above) */
    generateMany,
    forward(token, pos, needLogits = true) {
      if (pool.unchecked && pool.generations && mayRecheck()) beginSearch(pool.chosen);  // T240
      if (pool.search && needLogits) {
        const [count, timed] = countForToken();
        pool.threads = count;
        const began = clock();
        forward(token, pos, needLogits);
        recordToken(count, clock() - began, timed);
        if (pool.search) pool.threads = pool.search.best;
      } else if (needLogits && gpuPart.tokensOn && gpuPart.gpuOn) {
        // T152: a step on the CPU, timed against the GPU's
        const began = performance.now();
        forward(token, pos, needLogits);
        steps.cpu(pool.threads, performance.now() - began);
        if (gpuPart.tokenRecheck === "cpu" && --gpuPart.cpuRecheck <= 0) gpuPart.tokenRecheck = null;
      } else {
        forward(token, pos, needLogits);
      }
      if (!needLogits || !bound) return;
      const view = bound.getBuffer ? bound.getBuffer("f32") : { data: bound, release() {} };
      view.data.set(new Float32Array(memory.buffer, logits, vocab));
      view.release();
    },
    /** Python's array goes back (Llama.release()), the helper threads end, and the GPU's worker (T135). T205: resolves
     * once the GPU's worker has let go of its buffers and its device, or after GPU_END_MS (the worker is terminated
     * then), true where it said so: the next model is read after it (an iPhone's tab went down where the GPU of the
     * one before still held up to 189 MB as the next came from the cache) */
    release() {
      bound?.destroy?.();
      bound = null;
      this.stopThreads();
      if (gpuPart.gpuWorker) stopGpu();
      const worker = gpuPart.gpuLast;
      let timer;
      const late = new Promise((resolve) => { timer = setTimeout(() => resolve(false), GPU_END_MS); });
      return Promise.race([gpuPart.gpuEnded.then(() => true), late]).then((ended) => {
        clearTimeout(timer);
        if (!ended) {
          console.warn(`gpu: the GPU's worker did not end within ${GPU_END_MS / 1000} s: terminated`);
          worker?.terminate?.();
        }
        return ended;
      });
    },
    /** the logits in this memory, for callers without Python (tests) */
    logits: () => new Float32Array(memory.buffer, logits, vocab),
    /** the keys and values in the cache at positions from .. from + count - 1, in float32, [layers][count][kvDim] each
     * (T229: the layers that attend)
     * (tests: T135 holds what the GPU wrote back to what the CPU computes). T210: on the GPU alone, the GPU's own, read
     * back from it (GPU_BLOCK positions a request) */
    keysAndValues(from, count) {
      if (direct) return gpuKeysAndValues(from, count);
      views();
      const read = (block) => {
        const out = new Float32Array(attending * count * kvDim);
        for (let l = 0; l < attending; l++) {
          for (let t = 0; t < count; t++) {
            const at = block + l * capacity * KV + (from + t) * KV;
            for (let i = 0; i < kvDim; i++) out[(l * count + t) * kvDim + i] = halfKV ? halfToFloat(H[at / 2 + i]) : F[at / 4 + i];
          }
        }
        return out;
      };
      return { keys: read(keys), values: read(values) };
    },
    memoryBytes: () => memory.buffer.byteLength,
  };
}
