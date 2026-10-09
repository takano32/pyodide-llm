// forward/engine.js (T349): createForward(), one model's forward pass: its memory after the checkpoint, its tensors,
// the frames of the activations, the cache of the keys and values, the layers of every architecture, and the object
// Python and the worker call. The software threads are forward/threads.js's and the GPU's side forward/gpuside.js's;
// each is handed what it reads of this function and gives back what this one calls of it.
// A module of public/forward.js, asked for with its ?v=<build>; it reads jobs.js and its neighbours the same way.

const { BATCH, addressed, runner } = await import(new URL(`../jobs.js${new URL(import.meta.url).search}`, import.meta.url));
const { GPU_END_MS, GPU_BLOCK, GPU_TOKENS, GPU_RECHECK, TOKEN_RECHECK } = await import(new URL(`choice.js${new URL(import.meta.url).search}`, import.meta.url));
const { growMemory, PAGE, align, layerSlots, attendingLayers, linearWidths, frameArrays, rotatedWidths, frameBytes } = await import(new URL(`memory.js${new URL(import.meta.url).search}`, import.meta.url));
const { GPU_ALONE } = await import(new URL(`alone.js${new URL(import.meta.url).search}`, import.meta.url));
const { softwareThreads } = await import(new URL(`threads.js${new URL(import.meta.url).search}`, import.meta.url));
const { gpuFit, gpuSide } = await import(new URL(`gpuside.js${new URL(import.meta.url).search}`, import.meta.url));

// T120: no phase comes near this without progress (the longest token measured, Qwen2.5 7B's on CI, took 286 ms in
// all): a count that has not moved for so long means a software thread the browser stopped in the middle of its chunk
const STALLED_MS = 10000;

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
  // T349: what the GPU would hold of this model, and why it would not take it (forward/gpuside.js)
  const { gpuMatrices, gpuUnfit, tokensUnfit, gpuVectors } = gpuFit({ plan, gpuRoom, memoryUnsaid, direct, dim, layers,
    headSize, vocab, seqLen, hidden, kvDim, qDim, layerNorm, linear, convolution, rotated, sharedMemory, T, D, wq, wk, wv,
    wo, w1, w2, w3, wcls, attW, ffnW, attB, ffnB, bo, b1, b2, bq, bk, bv, qNorm, kNorm, positions });

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

  // T349: the phases and the software threads (forward/threads.js): their functions by name, their variables through pool
  const pool = softwareThreads({ memory, kernels, plan, spawn, stalledMs, wide, sharedMemory, runRows,
    gpuGettingReady: () => gpuPart.gpuGettingReady() });
  const { phase, ctl, waitUntil, shared, ensureHelpers, beginSearch, mayRecheck, searchLog, stopHelpers, countForToken,
    recordToken } = pool;
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

  // T349: the GPU's side (forward/gpuside.js): its functions by name, its variables through gpuPart
  const gpuPart = gpuSide({ memory, base, plan, gpu, gpuRemembered, gpuForce, direct, stalledMs, always, dim, layers, heads,
    kvHeads, headSize, vocab, seqLen, hidden, kvDim, gpt2, layerNorm, parallel, unturned, views, T, halfKV, D, KV, wo, wcls,
    finalW, finalB, eps, cosTable, sinTable, positions, channels, gpuMatrices, gpuVectors, gpuWhyNot, staging, gpuRows,
    tokensWhyNot, gpuIds, grow, ctl, waitUntil, pool, embed, fromStaging, stagingFinite, notFiniteKV, run, held, halfToFloat });
  const { stopGpu, times, steps, tokenVerdict, verdict, gpuTakes, promptOnGpu, promptOnCpu, gpuNote, statusNow, gpuSteps,
    generateMany, gpuKeysAndValues } = gpuPart;

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
