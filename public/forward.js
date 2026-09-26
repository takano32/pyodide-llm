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

// what a job of a phase is, shared with the software threads (helper.js), from the same deployment as this file
const { CONTROL_BYTES, GEN, QUIT, COUNTER, FINISHED, ACTIVE, TOTAL, WAKE, JOBS, JOB, JOB_TABLE, BATCH, ROWS, SIZE, FIRST, addressed, runner } =
  await import(new URL(`jobs.js${new URL(import.meta.url).search}`, import.meta.url));
export { BATCH };

const PAGE = 65536;
// T120: no phase comes near this without progress (the longest token measured, Qwen2.5 7B's on CI, took 286 ms in
// all): a count that has not moved for so long means a software thread the browser stopped in the middle of its chunk
const STALLED_MS = 10000;
const align = (n, to = 64) => Math.ceil(n / to) * to;

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

// T101: a 64-bit memory (Memory64) holds more than 4 GiB. Chrome and Firefox have it, shipping Safari not. Its sizes
// are BigInt; the pages of a 32-bit memory stop at 65536, those of a 64-bit one here at 262144 (16 GiB, Chrome's).
const PAGES_32 = 65536, PAGES_64 = 262144;
/** Whether this browser makes 64-bit memories. */
export function memory64() {
  try {
    new WebAssembly.Memory({ initial: 1n, address: "i64" });
    return true;
  } catch {
    return false;
  }
}

// The arrays of one token's frame (see createForward), in their order, and the bytes of each. qDim: the width of q
// and of the attention's output (into xb), heads times the head size: dim, except where a head has another size (T124)
const frameArrays = (dim, hidden, kvDim, qDim = dim) => {
  const D = dim * 4, HD = hidden * 4, KF = kvDim * 4, QD = Math.max(dim, qDim) * 4, XQ = Math.max(dim, hidden, qDim);
  return [["x", D], ["xb", QD], ["xb2", D], ["q", qDim * 4], ["kNow", KF], ["vNow", KF], ["before", D], ["hb", HD],
    ["hb2", HD], ["xq", XQ], ["xs", Math.ceil(XQ / 32) * 4]];
};
const frameBytes = (arrays) => arrays.reduce((size, [, bytes]) => size + align(bytes), 0);

/** T115: the most bytes the forward pass puts after a checkpoint of size bytes: at the end of its whole context,
 * while the KV cache grows to it (the old blocks and the new ones are both there then). An upper bound, a little
 * above what createForward allocates (tests/forward-check.mjs holds the two together).
 * header: the 7 ints of the legacy format. dtype: the file's ("float32", "float16", "int8", "int6"). int8: the int8
 * kernels compute on the weights (not with ?without=int8, which widens them to float32); relaxed: with relaxed SIMD
 * (a float32 correction a group); halfKV: keys and values in float16 (T110: an int8 model on a shared memory).
 * kvStart and outliers are llama2_numpy's KV_START and OUTLIER_CHANNELS. headDim: the size of a head where it is not
 * dim / heads (T124: the converter's options say head_dim then; 0 where they do not). */
export function footprint(header, size, { dtype = "float32", arch = "llama", int8 = true, relaxed = true, halfKV = false,
  kvStart = 256, outliers = 8, headDim = 0 } = {}) {
  const [dim, hidden, layers, heads, kvHeads, signedVocab, seqLen] = header;
  const vocab = Math.abs(signedVocab), headSize = headDim || dim / heads, kvDim = kvHeads * headSize, qDim = heads * headSize;
  const quantized = dtype === "int8" || dtype === "int6", six = dtype === "int6";
  // the int8 kernels take rows of whole groups of 32 (llama2_numpy widens the others)
  const onInt8 = int8 && dim % 32 === 0 && qDim % 32 === 0 && kvDim % 32 === 0 && hidden % 32 === 0;
  let bytes = 0;
  // what the file holds in another form, for its matrices (not the tables that are no matrix multiplied: an
  // embedding apart from the classifier, GPT-2's positions): the corrections of relaxed SIMD, one float32 a group,
  // as many as the scales (a ninth of an int8 file, a seventh of an int6 one), and the float32 columns of the
  // outlier channels (T92); or, off the int8 kernels, every weight widened to float32
  const tables = (signedVocab < 0 ? vocab * dim : 0) + (arch === "gpt2" ? seqLen * dim : 0);
  const weights = quantized ? size * (six ? 32 / 28 : 32 / 36) - tables : 0;
  if (quantized && onInt8) bytes += (relaxed ? weights / 8 : 0) + Math.min(outliers, dim) * (vocab + 1) * 4;
  else if (quantized) bytes += weights * 4;
  else if (dtype === "float16") bytes += size * 2;
  // what a quantized file leaves out: GPT-2's positions widened, the RoPE tables Python computes
  if (quantized) bytes += arch === "gpt2" ? seqLen * dim * 4 : seqLen * headSize * 4;
  // the frames of BATCH tokens, their attention scores, the logits
  bytes += BATCH * (frameBytes(frameArrays(dim, hidden, kvDim, qDim)) + align(seqLen * heads * 4)) + vocab * 4;
  // the KV cache doubles from kvStart: at its largest step, the smaller blocks are still there next to the larger
  let capacity = Math.min(kvStart, seqLen), most = capacity;
  while (capacity < seqLen) {
    const larger = Math.min(2 * capacity, seqLen);
    most = Math.max(most, capacity + larger);
    capacity = larger;
  }
  bytes += most * layers * 2 * kvDim * (halfKV ? 2 : 4);
  return Math.ceil(bytes) + 2 ** 20;  // and a megabyte for the alignment of every array
}
/** Whether a checkpoint of size bytes and the forward pass after it (footprint) pass the 4 GiB of a 32-bit memory.
 * A model that fits stays there: a 64-bit memory runs the kernels about a tenth slower (T101, measured). */
export const needsWide = (size, after) => CONTROL_BYTES + size + after > PAGES_32 * PAGE;
/** T133: the dtype of a model converted with none asked for, from its int8 size and what the forward pass puts after
 * it (footprint, as int8): int8 where that fits a 32-bit memory, or where the browser has a 64-bit one (wide: Chrome
 * and Firefox; about a tenth slower, against six bits' half the speed and +1.4 to 1.7% of perplexity, T98); else six
 * bits (7/9 of int8's memory: Safari). */
export const automaticDtype = (int8, after, wide) => (wide || !needsWide(int8, after) ? "int8" : "int6");
/** memory.grow(pages), in the number type of the memory (wide: 64-bit) */
export function growMemory(memory, pages, wide) {
  memory.grow(wide ? BigInt(pages) : pages);
}

/** A memory with room for a checkpoint of size bytes at base; the forward pass allocates after it. shared (stage 2):
 * a SharedArrayBuffer for the helper threads, only where the page is cross-origin isolated; its first 8 KiB are the
 * control area of the helpers. A shared memory needs a maximum: as much as the browser grants, less if it refuses.
 * maximum (pages): what to ask for first; else what this model needs (after: what the forward pass puts after the
 * checkpoint, footprint(); three times the file where it is not said) and a gigabyte more. The worker keeps the
 * memory for the models that fit under that maximum (T96): a browser reserves address space for each WebAssembly
 * memory whatever its maximum, and Chromium refused the third one of a page. */
export function weightsMemory(size, { shared = false, maximum, wide = false, after = 3 * size } = {}) {
  const base = shared ? CONTROL_BYTES : 64;
  const initial = Math.ceil((base + size) / PAGE) + 1;
  // a 64-bit memory (T101) says its sizes in BigInt
  const describe = (pages) => (wide ? { initial: BigInt(initial), ...(pages ? { maximum: BigInt(pages) } : {}), address: "i64" }
    : { initial, ...(pages ? { maximum: pages } : {}) });
  if (!shared) return { memory: new WebAssembly.Memory(describe()), base };
  // what the model needs, and a gigabyte for the next one to fit as well (T96); less if the browser refuses
  const most = maximum ?? Math.min(wide ? PAGES_64 : PAGES_32, Math.ceil((base + size + after + 2 ** 30) / PAGE));
  for (const pages of [most, initial + 16384, initial + 4096]) {
    try {
      const memory = new WebAssembly.Memory({ ...describe(Math.max(pages, initial)), shared: true });
      memory.maximum = Math.max(pages, initial);  // the worker keeps the memory as long as the next model fits (T96)
      memory.limited = pages < most;  // the browser gave less than was asked for: a new memory would not get more
      return { memory, base };
    } catch {
      // too much address space for this browser: ask for less
    }
  }
  throw new Error("This browser gives no shared WebAssembly memory for this model.");
}

// ---- the helper threads (stage 2): the control area at the start of a shared memory (jobs.js has its layout)
// how many chunks per thread the rows of a matmul are cut into: whoever is free takes the next one, so a slow core
// (a little core of a big.LITTLE phone) simply takes fewer. 2 to 16 measured the same (T93); fewer does not steal.
const CHUNKS_PER_THREAD = 4;

/** What Llama(external=) takes: the size of the checkpoint, read() for the few bytes Python looks at itself, and
 * start(plan), which builds the forward pass. */
export function external({ memory, base, size, kernels, spawn }) {
  const outside = {
    size,
    read: (offset, length) => new Uint8Array(memory.buffer, base + offset, length).slice(),
    start: (plan) => {
      outside.engine = createForward({ memory, base, size, kernels, spawn, plan: plan.toJs ? plan.toJs({ dict_converter: Object.fromEntries }) : plan });
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
/** wrap (tests/profile.mjs only): gets the kernels' exports and returns what to call instead, to time the forward
 * pass with some kernels replaced by functions that do nothing. */
/** stalledMs (tests only): how long a phase may make no progress before its software threads are given up (T120) */
export function createForward({ memory, base, size, kernels, plan, spawn, wrap = (exports) => exports, stalledMs = STALLED_MS }) {
  const { dim, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: headSize, vocab_size: vocab,
    seq_len: seqLen, rotary, arch } = plan;
  const hidden = plan.hidden_dim, kvDim = kvHeads * headSize, qDim = heads * headSize;
  const gpt2 = arch === "gpt2", layerNorm = arch === "gpt2" || arch === "neox", parallel = plan.parallel_residual;
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
  // weight i of an int8 or int6 tensor, as the int8 it is: int6 (T98) unpacked from its group of 24 bytes, the
  // layout of llama2_numpy.pack6 (six bits, then two zero bits)
  function weightAt(t, i) {
    if (t.kind === "int8") return I[base + t.offset + i];
    const group = base + t.offset + ((i / 32) | 0) * 24, j = i % 32;
    const low = j < 16 ? U[group + j] & 15 : U[group + j - 16] >> 4;
    const top = (U[group + 16 + (j % 8)] >> (2 * ((j / 8) | 0))) & 3;
    return (((low | (top << 4)) << 2) << 24) >> 24;  // the byte as a signed int8
  }
  // a float32 copy of a tensor: float16 converted, int8 and int6 times the scale of their group (Math.fround is the
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
  // a matrix: int8 or int6 (values, scales, corrections) when plan.int8, else float32
  function matrix(name) {
    const source = plan.shared_classifier && name === "wcls" ? "token_embedding_table" : name;
    const t = T[source];
    if (!t) return null;
    const [rows, n] = t.shape.slice(-2);
    if ((t.kind === "int8" || t.kind === "int6") && plan.int8) {
      const six = t.kind === "int6", rowBytes = six ? n / 32 * 24 : n;
      const values = base + t.offset, scales = base + t.scales, groups = count(t) / t.group;
      let corrections = scales;
      if (relaxed) {
        // relaxed SIMD multiplies by 7-bit unsigned activations with a bias of 64, which this takes out again:
        // dot(w, q - 64) = dot(w, q) - 64 * sum(w). scale * sum of the group, in float32
        corrections = alloc(groups * 4);
        // the same numbers as a sum in JavaScript, a kernel's speed (T98, T123: 7B spent 266 s here one value at a
        // time). Groups of 32: relaxed runs only where every row is whole groups
        (six ? k.six_sums : k.int8_sums)(corrections, values, scales, groups);
      }
      const layer = (l) => [values + l * rows * rowBytes, scales + l * rows * (n / t.group) * 4, corrections + l * rows * (n / t.group) * 4];
      return { rows, n, int8: true, six, layer };
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
  // KV: the bytes of one position's keys in the cache; KF: in float32.
  const halfKV = Boolean(plan.half_kv) && sharedMemory;
  const D = dim * 4, HD = hidden * 4, KF = kvDim * 4, QF = qDim * 4, KV = kvDim * (halfKV ? 2 : 4);
  const inFrame = frameArrays(dim, hidden, kvDim, qDim), S = frameBytes(inFrame);
  const frames = alloc(BATCH * S), at = {};
  inFrame.reduce((offset, [name, bytes]) => { at[name] = frames + offset; return offset + align(bytes); }, 0);
  // kNow, vNow: this token's key and value in float32, before they go into the cache
  const { x, xb, xb2, q, kNow, vNow, before, hb, hb2, xq, xs } = at;
  const A = seqLen * heads * 4;  // the scores of one token's attention
  const att = alloc(BATCH * A), logits = alloc(vocab * 4);
  const wq = matrix("wq"), wk = matrix("wk"), wv = matrix("wv"), wo = matrix("wo");
  const w1 = matrix("w1"), w2 = matrix("w2"), w3 = matrix("w3"), wcls = matrix("wcls");
  const attW = floats("rms_att_weight"), ffnW = floats("rms_ffn_weight"), finalW = floats("rms_final_weight");
  const attB = floats("ln_att_bias"), ffnB = floats("ln_ffn_bias"), finalB = floats("ln_final_bias");
  const bo = floats("bo"), b1 = floats("b1"), b2 = floats("b2"), bq = floats("bq"), bk = floats("bk"), bv = floats("bv");
  const qNorm = floats("q_norm"), kNorm = floats("k_norm");  // T124: Qwen3 normalizes every head of q and k
  const cosTable = floats("freq_cis_real"), sinTable = floats("freq_cis_imag");
  const positions = gpt2 ? floats("positions") : 0;
  const embedding = T.token_embedding_table;
  const embeddingRows = embedding.kind === "f16" ? floats("token_embedding_table") : 0;

  // the outlier channels of the classifier's input (T92): their columns in float32, multiplied apart
  const channels = plan.outliers ?? [];
  let columns = 0, picked = 0;
  if (channels.length) {
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

  // the KV cache: per layer [positions][kvDim], one block for the keys and one for the values, last in memory
  // so that growing it (KV_START, doubling) can take the space of the smaller one
  let capacity = Math.min(plan.kv_start, seqLen);
  let keys = alloc(layers * capacity * KV), values = alloc(layers * capacity * KV);
  function grow(pos) {
    const larger = Math.min(Math.max(2 * capacity, pos + 1), seqLen);
    const oldLayer = capacity * KV, newLayer = larger * KV;
    const newKeys = alloc(layers * newLayer), newValues = alloc(layers * newLayer);
    for (let l = 0; l < layers; l++) {
      U.copyWithin(newKeys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);
      U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
    }
    // move both down onto the old blocks, which were the last thing in memory
    const start = keys;
    U.copyWithin(start, newKeys, newValues + layers * newLayer);
    keys = start;
    values = start + (newValues - newKeys);
    top = align(values + layers * newLayer);
    capacity = larger;
  }

  // ---- the matrix multiplications, in phases: the matmuls that read the same input (q, k and v; w1 and w3) go
  // out together. The input is quantized here, once; the rows are computed here and, with helper threads, by
  // whoever takes them. Every row is computed whole by one thread with the same kernel, so the numbers are the same
  // with any number of threads.
  //
  // A job is what jobs.js says: [kind, eight arguments, rows, count, out stride, a stride, b stride].
  const shared = sharedMemory && spawn;
  const ctl = shared ? new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4) : null;
  const table = shared ? new Float64Array(memory.buffer, JOB_TABLE, BATCH * JOB) : null;  // the jobs (jobs.js)
  // the memory is kept from model to model (T96), and the control area with it: what the last engine's phases left
  // there (a helper counted in ACTIVE when it was ended, a generation) would hold this one's first phase for ever
  ctl?.fill(0);
  const helpers = [];
  let threads = 1, gen = 0;
  const jobOf = (m, out, outStride, input, l, count) => {
    const [w, s, c] = m.layer(l);
    if (!m.int8) return [2, out, input, 0, w, 0, 0, m.n, 0, m.rows, count, outStride, S, 0];
    const kind = m.six ? (relaxed ? 5 : 6) : (relaxed ? 0 : 1);
    return [kind, out, xq, xs, w, s, relaxed ? c : 0, m.n, 0, m.rows, count, outStride, S, S];
  };
  // the attention of token t of a run, at position pos (its scores have a place of their own: tokens run at once)
  const attentionJob = (t, pos, layerKeys, layerValues) =>
    [halfKV ? 4 : 3, xb + t * S, q + t * S, layerKeys, layerValues, att + t * A, pos, kvHeads, headSize, heads, 1, 0, 0, 0];
  const runRows = runner(k, relaxed);
  // stopThreads() runs on this thread too, never in the middle of a phase: a wait here ends when the helpers have done
  // their part. A helper the browser itself stopped (iOS may end a worker for its memory) never counts the chunk it
  // took: T120, the wait gives up when its count has not moved for stalledMs, and says false. (A check of a flag that
  // stopThreads() raised and lowered again stood here and could never be seen, the review of T96 found.)
  // A wait that took far longer than it asked for means this thread did not run either (a frozen tab, a phone that
  // suspended the page): the helpers were stopped with it, and the time from before does not count (the review of
  // T120: a stop of 10 s gave a thread up now and then as the page came back)
  const waitUntil = (index, done) => {
    const tick = Math.min(1000, stalledMs);
    let moved = performance.now(), last = moved;
    for (let seen = Atomics.load(ctl, index); !done(seen);) {
      Atomics.wait(ctl, index, seen, tick);
      const now = Atomics.load(ctl, index), at = performance.now();
      if (at - last > 2 * tick) moved = at;
      last = at;
      if (now !== seen) [seen, moved] = [now, at];
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
      const size = Math.max(1, Math.ceil(rows / (threads * CHUNKS_PER_THREAD)));
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
  // matmuls of one input (count tokens of it, a frame apart): [matrix, output, output stride, layer] each
  function matmuls(input, count, list) {
    if (list[0][0].int8) {
      for (let t = 0; t < count; t++) k.quantize_x(xq + t * S, xs + t * S, input + t * S, list[0][0].n, bias);
    }
    phase(list.map(([m, out, outStride, l]) => jobOf(m, out, outStride, input, l, count)));
  }

  // one token (count 1) or up to BATCH tokens of a prompt at positions pos0, pos0 + 1, ... (T108). Every token is
  // computed as it would be alone: the same kernels on the same numbers, only the matmuls of a layer go out once for
  // all of them. The logits, if asked for, are the last token's.
  function run(tokens, pos0, needLogits) {
    const count = tokens.length;
    if (pos0 + count - 1 >= capacity) grow(pos0 + count - 1);
    views();
    // the embedding rows
    for (let t = 0; t < count; t++) {
      const row = tokens[t] * dim, to = (x + t * S) / 4;
      if (embedding.kind === "int8" || embedding.kind === "int6") {
        const g = embedding.group;
        for (let i = 0; i < dim; i++) {
          F[to + i] = Math.fround(weightAt(embedding, row + i) * F[(base + embedding.scales) / 4 + (((row + i) / g) | 0)]);
        }
      } else {
        const from = embedding.kind === "f32" ? base + embedding.offset : embeddingRows;
        F.copyWithin(to, from / 4 + row, from / 4 + row + dim);
      }
      if (positions) k.add_inplace(x + t * S, positions + (pos0 + t) * D, dim);
    }
    for (let l = 0; l < layers; l++) {
      const layerKeys = keys + l * capacity * KV, layerValues = values + l * capacity * KV;
      const kp = layerKeys + pos0 * KV, vp = layerValues + pos0 * KV;
      for (let t = 0; t < count; t++) {
        if (layerNorm) k.layernorm(xb + t * S, x + t * S, attW + l * D, attB + l * D, dim);
        else k.rmsnorm(xb + t * S, x + t * S, attW + l * D, dim);
        if (parallel) F.copyWithin((before + t * S) / 4, (x + t * S) / 4, (x + t * S) / 4 + dim);  // GPT-NeoX reads this layer's input twice
      }
      matmuls(xb, count, [[wq, q, S, l], [wk, kNow, S, l], [wv, vNow, S, l]]);
      for (let t = 0; t < count; t++) {
        const qt = q + t * S, kt = kNow + t * S, vt = vNow + t * S, pos = pos0 + t;
        if (bq) {
          k.add_inplace(qt, bq + l * QF, qDim);
          k.add_inplace(kt, bk + l * KF, kvDim);
          k.add_inplace(vt, bv + l * KF, kvDim);
        }
        if (qNorm) {
          const HS = headSize * 4;
          for (let h = 0; h < heads; h++) k.rmsnorm(qt + h * HS, qt + h * HS, qNorm + l * HS, headSize);
          for (let h = 0; h < kvHeads; h++) k.rmsnorm(kt + h * HS, kt + h * HS, kNorm + l * HS, headSize);
        }
        if (!gpt2) {
          const cos = cosTable + pos * (headSize / 2) * 4, sin = sinTable + pos * (headSize / 2) * 4;
          k.rope(qt, cos, sin, heads, headSize, rotary);
          k.rope(kt, cos, sin, kvHeads, headSize, rotary);
        }
        // into the cache, at this token's position
        if (halfKV) {
          k.to_f16(kp + t * KV, kt, kvDim);
          k.to_f16(vp + t * KV, vt, kvDim);
        } else {
          U.copyWithin(kp + t * KV, kt, kt + KF);
          U.copyWithin(vp + t * KV, vt, vt + KF);
        }
      }
      // the keys and values of positions up to each token's are all there now: its own and the ones before it.
      // The heads of every token go out as one phase (T109).
      phase(tokens.map((_, t) => attentionJob(t, pos0 + t, layerKeys, layerValues)));
      matmuls(xb, count, [[wo, xb2, S, l]]);
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
      for (let t = 0; t < count; t++) k.rmsnorm(xb + t * S, x + t * S, ffnW + l * D, dim);
      matmuls(xb, count, [[w1, hb, S, l], [w3, hb2, S, l]]);
      for (let t = 0; t < count; t++) k.swiglu(hb + t * S, hb + t * S, hb2 + t * S, hidden);
      matmuls(hb, count, [[w2, xb2, S, l]]);
      for (let t = 0; t < count; t++) k.add_inplace(x + t * S, xb2 + t * S, dim);
    }
    if (!needLogits) return;
    const last = x + (count - 1) * S;
    if (layerNorm) k.layernorm(xb, last, finalW, finalB, dim);
    else k.rmsnorm(xb, last, finalW, dim);
    channels.forEach((c, i) => {  // T92: the outlier channels are multiplied apart
      F[picked / 4 + i] = F[xb / 4 + c];
      F[xb / 4 + c] = 0;
    });
    matmuls(xb, 1, [[wcls, logits, 0, 0]]);
    if (channels.length) k.add_columns(logits, columns, picked, channels.length, vocab);
  }
  const forward = (token, pos, needLogits) => run([token], pos, needLogits);

  // ---- the number of threads (stage 2b): found by measuring, never written down. The search starts from a hint
  // (navigator.hardwareConcurrency, which counts the little cores of a big.LITTLE phone too) and compares the best
  // count so far with half of it and, if half is not faster, with twice as many; it goes on in that direction while
  // the other is faster by more than the noise of a run, and stops at the first that is not. Only the tokens that
  // make logits are timed (a prompt's tokens skip the classifier). One comparison runs the two counts in blocks,
  // best-candidate-candidate-best, so that the growing cost of later positions falls on both alike, and drops the
  // first token of every block (the switch). Helpers that a count needs are started in the background; until they
  // are ready the tokens run on the best count and are not timed.
  const BLOCK = 4, BETTER = 0.95;
  let search = null, chosen = 0, generations = 0, recheckEvery = 0, onChosen = null, onCompared = null;
  const searchLog = [];  // every comparison: the counts, their times in ms per token, and the verdict
  const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
  function beginSearch(from) {
    search = { best: Math.max(1, from), direction: from > 1 ? "down" : "up", moved: false, candidate: 0, times: null, step: 0, waiting: false };
    nextCandidate();
  }
  function nextCandidate() {
    if (lost) return finish();
    const { best, direction } = search;
    const candidate = direction === "down" ? Math.floor(best / 2) : best * 2;
    if (candidate < 1) return finish();
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
  function finish() {
    chosen = lost ? 1 : search ? search.best : threads;
    threads = chosen;
    search = null;
    // one thread after a give-up says nothing about the device: the page would start with it next time (the review of T120)
    if (!lost) onChosen?.(chosen);
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
    if (timed) search.times[count].push(milliseconds);
    search.step += 1;
    if (search.step < 4 * (BLOCK + 1)) return;
    const { best, candidate } = search;
    const faster = median(search.times[candidate]) < median(search.times[best]) * BETTER;
    searchLog.push({ best, candidate, times: search.times, faster });
    // T114: every verdict, so that a device's choice can be followed afterwards (the page writes it to the console)
    onCompared?.({ best, candidate, bestMs: median(search.times[best]), candidateMs: median(search.times[candidate]),
                   faster, tokens: search.times[best].length + search.times[candidate].length });
    if (faster) {
      search.best = candidate;
      search.moved = true;
      return nextCandidate();
    }
    if (search.direction === "down" && !search.moved) {
      search.direction = "up";
      return nextCandidate();
    }
    finish();
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

  let bound = null;
  const backend = (plan.int8 ? `SIMD kernels, ${T.wq?.kind === "int6" ? "int6" : "int8"}${relaxed ? ", relaxed SIMD" : ""}` : "SIMD kernels, float32") +
    (wide ? ", 64-bit memory" : "");  // T101: the status line says so, as it says every other way the model runs
  return {
    backend,
    /** Use n threads from the next token on (stage 2): starts the helpers that are missing. 1 on a memory that is
     * not shared. Resolves to the number in use. */
    async setThreads(n) {
      if (!shared || lost) return (threads = 1);
      search = null;
      if (!(await ensureHelpers(n))) return (threads = 1);
      threads = Math.max(1, n);
      return threads;
    },
    /** Find the number of threads while generating (see above): from a hint, or from a count remembered from an
     * earlier visit, which is then only checked against its neighbours now and then (every recheck generations).
     * chose(count) is told the answer, compared(verdict) every comparison on the way. The helpers of the starting count are started (and warmed) before this
     * resolves, so the first tokens do not wait for them. */
    async findThreads({ from, remembered = 0, recheck = 8, chose, compared }) {
      if (!shared || lost) return 1;
      onChosen = chose;
      onCompared = compared;
      recheckEvery = recheck;
      const start = Math.max(1, remembered || from);
      if (!(await ensureHelpers(start))) return (threads = 1);
      threads = start;
      if (remembered) {
        chosen = remembered;
      } else {
        beginSearch(start);
      }
      return threads;
    },
    /** the page starts a generation: now and then the remembered count is checked against its neighbours again */
    newGeneration() {
      generations += 1;
      if (!search && chosen && recheckEvery && generations % recheckEvery === 0) beginSearch(chosen);
    },
    get searching() {
      return search !== null;
    },
    searchLog,
    get threads() {
      return threads;
    },
    /** the helper threads end; this engine runs on its own again */
    stopThreads() {
      if (shared) stopHelpers();
    },
    /** T120: whether a software thread stopped under this engine, which then went on with one */
    get lostThreads() {
      return lost;
    },
    /** the float32 array of Python's that forward() fills with the logits */
    bind(array) {
      bound = array.copy ? array.copy() : array;
    },
    /** T108: tokens (up to BATCH) of a prompt at positions pos, pos + 1, ...: the same as forward() for each of them
     * in turn without logits, in one pass through the layers */
    forwardMany(tokens, pos) {
      const list = tokens.toJs ? tokens.toJs() : [...tokens];
      for (let at = 0; at < list.length; at += BATCH) run(list.slice(at, at + BATCH), pos + at, false);
    },
    forward(token, pos, needLogits = true) {
      if (search && needLogits) {
        const [count, timed] = countForToken();
        threads = count;
        const began = performance.now();
        forward(token, pos, needLogits);
        recordToken(count, performance.now() - began, timed);
        if (search) threads = search.best;
      } else {
        forward(token, pos, needLogits);
      }
      if (!needLogits || !bound) return;
      const view = bound.getBuffer ? bound.getBuffer("f32") : { data: bound, release() {} };
      view.data.set(new Float32Array(memory.buffer, logits, vocab));
      view.release();
    },
    /** Python's array goes back (Llama.release()), and the helper threads end */
    release() {
      bound?.destroy?.();
      bound = null;
      this.stopThreads();
    },
    /** the logits in this memory, for callers without Python (tests) */
    logits: () => new Float32Array(memory.buffer, logits, vocab),
    memoryBytes: () => memory.buffer.byteLength,
  };
}
