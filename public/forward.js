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

const PAGE = 65536;
const align = (n, to = 64) => Math.ceil(n / to) * to;

/** The kernels as WebAssembly modules. The relaxed one fails to compile where relaxed SIMD is missing (Safari):
 * then int8 runs on matmul_q8. */
export function compileKernels(plain, relaxed) {
  let relaxedModule = null;
  try {
    relaxedModule = relaxed ? new WebAssembly.Module(relaxed) : null;
  } catch {
    relaxedModule = null;
  }
  return { plain: new WebAssembly.Module(plain), relaxed: relaxedModule };
}

/** A memory with room for a checkpoint of size bytes at base; the forward pass allocates after it. shared (stage 2):
 * a SharedArrayBuffer for the helper threads, only where the page is cross-origin isolated; its first 4 KiB are the
 * control area of the helpers. A shared memory needs a maximum: as much as the browser grants, less if it refuses. */
export function weightsMemory(size, { shared = false } = {}) {
  const base = shared ? CONTROL_BYTES : 64;
  const initial = Math.ceil((base + size) / PAGE) + 1;
  if (!shared) return { memory: new WebAssembly.Memory({ initial }), base };
  // A shared memory reserves its maximum up front, and a page that loads model after model (the benchmark does)
  // ran out of address space with 4 GB each (T93). So: what this model can need at most, the checkpoint widened
  // to float32 (four times an int8 file, with the int8 switch off) and a gigabyte for the KV cache and the rest.
  const most = Math.min(65536, Math.ceil((base + 4 * size + 2 ** 30) / PAGE));
  for (const maximum of [most, initial + 16384, initial + 4096]) {
    try {
      return { memory: new WebAssembly.Memory({ initial, maximum: Math.max(maximum, initial), shared: true }), base };
    } catch {
      // too much address space for this browser: ask for less
    }
  }
  throw new Error("This browser gives no shared WebAssembly memory for this model.");
}

// ---- the helper threads (stage 2): the control area at the start of a shared memory, as helper.js reads it
const CONTROL_BYTES = 4096;
// WAKE + share: each helper's own word, so that a phase wakes exactly helpers 1..threads-1 (see helper.js)
const GEN = 0, QUIT = 1, COUNTER = 2, FINISHED = 3, ACTIVE = 4, TOTAL = 5, WAKE = 256, JOBS = 512, JOB = 12;
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
export function createForward({ memory, base, size, kernels, plan, spawn, wrap = (exports) => exports }) {
  const { dim, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: headSize, vocab_size: vocab,
    seq_len: seqLen, rotary, arch } = plan;
  const hidden = plan.hidden_dim, kvDim = kvHeads * headSize;
  const gpt2 = arch === "gpt2", layerNorm = arch === "gpt2" || arch === "neox", parallel = plan.parallel_residual;
  const imports = { env: { memory } };
  const k = wrap(new WebAssembly.Instance(kernels.plain, imports).exports);
  const relaxed = plan.int8 && plan.relaxed && kernels.relaxed ? wrap(new WebAssembly.Instance(kernels.relaxed, imports).exports).matmul_q8r : null;
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
    if (top > memory.buffer.byteLength) memory.grow(Math.ceil((top - memory.buffer.byteLength) / PAGE));
    views();
    return at;
  };
  views();

  const T = plan.tensors, derived = plan.derived ?? {};
  const count = (t) => t.shape.reduce((a, b) => a * b, 1);
  // a float32 copy of a tensor: float16 converted, int8 times the scale of its group (Math.fround is the float32
  // product NumPy computes)
  function widen(t) {
    const n = count(t), at = alloc(n * 4);
    if (t.kind === "f16") {
      for (let i = 0; i < n; i++) F[at / 4 + i] = halfToFloat(H[(base + t.offset) / 2 + i]);
    } else {
      const g = t.group;
      for (let i = 0; i < n; i++) F[at / 4 + i] = Math.fround(I[base + t.offset + i] * F[(base + t.scales) / 4 + ((i / g) | 0)]);
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
  // a matrix: int8 (values, scales, corrections) when plan.int8, else float32
  function matrix(name) {
    const source = plan.shared_classifier && name === "wcls" ? "token_embedding_table" : name;
    const t = T[source];
    if (!t) return null;
    const [rows, n] = t.shape.slice(-2);
    if (t.kind === "int8" && plan.int8) {
      const values = base + t.offset, scales = base + t.scales, groups = count(t) / t.group;
      let corrections = scales;
      if (relaxed) {
        // relaxed SIMD multiplies by 7-bit unsigned activations with a bias of 64, which this takes out again:
        // dot(w, q - 64) = dot(w, q) - 64 * sum(w). scale * sum of the group, in float32
        corrections = alloc(groups * 4);
        for (let g = 0; g < groups; g++) {
          let sum = 0;
          for (let i = 0; i < t.group; i++) sum += I[values + g * t.group + i];
          F[corrections / 4 + g] = Math.fround(F[scales / 4 + g] * sum);
        }
      }
      const layer = (l) => [values + l * rows * n, scales + l * rows * (n / t.group) * 4, corrections + l * rows * (n / t.group) * 4];
      return { rows, n, int8: true, layer };
    }
    const w = floats(source);  // float32 as it is, or widened once (shared with the embedding when it is the same table)
    return { rows, n, int8: false, layer: (l) => [w + l * rows * n * 4] };
  }

  // ---- the activations
  const x = alloc(dim * 4), xb = alloc(dim * 4), xb2 = alloc(dim * 4), q = alloc(dim * 4), before = alloc(dim * 4);
  const hb = alloc(hidden * 4), hb2 = alloc(hidden * 4), att = alloc(seqLen * heads * 4), logits = alloc(vocab * 4);
  const xq = alloc(Math.max(dim, hidden)), xs = alloc((Math.max(dim, hidden) / 32) * 4);
  const wq = matrix("wq"), wk = matrix("wk"), wv = matrix("wv"), wo = matrix("wo");
  const w1 = matrix("w1"), w2 = matrix("w2"), w3 = matrix("w3"), wcls = matrix("wcls");
  const attW = floats("rms_att_weight"), ffnW = floats("rms_ffn_weight"), finalW = floats("rms_final_weight");
  const attB = floats("ln_att_bias"), ffnB = floats("ln_ffn_bias"), finalB = floats("ln_final_bias");
  const bo = floats("bo"), b1 = floats("b1"), b2 = floats("b2"), bq = floats("bq"), bk = floats("bk"), bv = floats("bv");
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
        F[columns / 4 + i * vocab + v] = Math.fround(I[base + t.offset + v * dim + c] * F[(base + t.scales) / 4 + v * groups + ((c / t.group) | 0)]);
      }
    });
  }

  // the KV cache: per layer [positions][kvDim], one block for the keys and one for the values, last in memory
  // so that growing it (KV_START, doubling) can take the space of the smaller one
  let capacity = Math.min(plan.kv_start, seqLen);
  let keys = alloc(layers * capacity * kvDim * 4), values = alloc(layers * capacity * kvDim * 4);
  function grow(pos) {
    const larger = Math.min(Math.max(2 * capacity, pos + 1), seqLen);
    const oldLayer = capacity * kvDim * 4, newLayer = larger * kvDim * 4;
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
  const shared = typeof SharedArrayBuffer !== "undefined" && memory.buffer instanceof SharedArrayBuffer && spawn;
  const ctl = shared ? new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4) : null;
  const helpers = [];
  let threads = 1, gen = 0;
  const jobOf = (m, out, input, l) => {
    const [w, s, c] = m.layer(l);
    if (!m.int8) return [2, out, input, 0, w, 0, 0, m.n, m.rows];
    return relaxed ? [0, out, xq, xs, w, s, c, m.n, m.rows] : [1, out, xq, xs, w, s, 0, m.n, m.rows];
  };
  const runRows = (job, r0, r1) => {
    const [kind, out, a, b, w, s, c, n] = job;
    if (kind === 0) relaxed(out, a, b, w, s, c, n, r0, r1);
    else if (kind === 1) k.matmul_q8(out, a, b, w, s, n, r0, r1);
    else k.matmul_f32(out, a, w, n, r0, r1);
  };
  const waitUntil = (index, done) => {
    for (let seen = Atomics.load(ctl, index); !done(seen); seen = Atomics.load(ctl, index)) Atomics.wait(ctl, index, seen);
  };
  function phase(jobs) {
    if (threads <= 1) {
      for (const job of jobs) runRows(job, 0, job[8]);
      return;
    }
    // close the previous phase (odd), let every helper still awake leave it, then rewrite the jobs
    Atomics.store(ctl, GEN, gen + 1);
    waitUntil(ACTIVE, (seen) => seen === 0);
    let total = 0;
    ctl[JOBS] = jobs.length;
    jobs.forEach((job, i) => {
      const at = JOBS + 1 + i * JOB, rows = job[8];
      const size = Math.max(1, Math.ceil(rows / (threads * CHUNKS_PER_THREAD)));
      job.forEach((value, n) => { ctl[at + n] = value; });
      ctl[at + 9] = size;
      ctl[at + 10] = total;
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
      while (ctl[JOBS + 1 + j * JOB + 10] > c) j--;
      const at = JOBS + 1 + j * JOB, size = ctl[at + 9], r0 = (c - ctl[at + 10]) * size;
      runRows(jobs[j], r0, Math.min(r0 + size, jobs[j][8]));
      Atomics.add(ctl, FINISHED, 1);
    }
    waitUntil(FINISHED, (seen) => seen === total);
  }
  // matmuls of one input: [matrix, output, layer] each
  function matmuls(input, list) {
    if (list[0][0].int8) k.quantize_x(xq, xs, input, list[0][0].n, bias);
    phase(list.map(([m, out, l]) => jobOf(m, out, input, l)));
  }

  function forward(token, pos, needLogits) {
    if (pos >= capacity) grow(pos);
    views();
    // the embedding row
    const row = token * dim;
    if (embedding.kind === "int8") {
      const g = embedding.group;
      for (let i = 0; i < dim; i++) {
        F[x / 4 + i] = Math.fround(I[base + embedding.offset + row + i] * F[(base + embedding.scales) / 4 + (((row + i) / g) | 0)]);
      }
    } else {
      const from = embedding.kind === "f32" ? base + embedding.offset : embeddingRows;
      F.copyWithin(x / 4, from / 4 + row, from / 4 + row + dim);
    }
    if (positions) k.add_inplace(x, positions + pos * dim * 4, dim);
    const cos = cosTable + pos * (headSize / 2) * 4, sin = sinTable + pos * (headSize / 2) * 4;
    for (let l = 0; l < layers; l++) {
      const layerKeys = keys + l * capacity * kvDim * 4, layerValues = values + l * capacity * kvDim * 4;
      const kp = layerKeys + pos * kvDim * 4, vp = layerValues + pos * kvDim * 4;
      if (layerNorm) k.layernorm(xb, x, attW + l * dim * 4, attB + l * dim * 4, dim);
      else k.rmsnorm(xb, x, attW + l * dim * 4, dim);
      if (parallel) F.copyWithin(before / 4, x / 4, x / 4 + dim);  // GPT-NeoX reads this layer's input twice
      matmuls(xb, [[wq, q, l], [wk, kp, l], [wv, vp, l]]);
      if (bq) {
        k.add_inplace(q, bq + l * dim * 4, dim);
        k.add_inplace(kp, bk + l * kvDim * 4, kvDim);
        k.add_inplace(vp, bv + l * kvDim * 4, kvDim);
      }
      if (!gpt2) {
        k.rope(q, cos, sin, heads, headSize, rotary);
        k.rope(kp, cos, sin, kvHeads, headSize, rotary);
      }
      k.attention(xb, q, layerKeys, layerValues, att, pos, heads, kvHeads, headSize);
      matmuls(xb, [[wo, xb2, l]]);
      k.add_inplace(x, xb2, dim);
      if (layerNorm) {
        k.add_inplace(x, bo + l * dim * 4, dim);
        k.layernorm(xb, parallel ? before : x, ffnW + l * dim * 4, ffnB + l * dim * 4, dim);
        matmuls(xb, [[w1, hb, l]]);
        k.gelu(hb, hb, b1 + l * hidden * 4, hidden);
        matmuls(hb, [[w2, xb2, l]]);
        k.add_inplace(x, xb2, dim);
        k.add_inplace(x, b2 + l * dim * 4, dim);
        continue;
      }
      k.rmsnorm(xb, x, ffnW + l * dim * 4, dim);
      matmuls(xb, [[w1, hb, l], [w3, hb2, l]]);
      k.swiglu(hb, hb, hb2, hidden);
      matmuls(hb, [[w2, xb2, l]]);
      k.add_inplace(x, xb2, dim);
    }
    if (!needLogits) return;
    if (layerNorm) k.layernorm(xb, x, finalW, finalB, dim);
    else k.rmsnorm(xb, x, finalW, dim);
    if (channels.length) {
      channels.forEach((c, i) => {
        F[picked / 4 + i] = F[xb / 4 + c];
        F[xb / 4 + c] = 0;
      });
      matmuls(xb, [[wcls, logits, 0]]);
      k.add_columns(logits, columns, picked, channels.length, vocab);
    } else {
      matmuls(xb, [[wcls, logits, 0]]);
    }
  }

  // ---- the number of threads (stage 2b): found by measuring, never written down. The search starts from a hint
  // (navigator.hardwareConcurrency, which counts the little cores of a big.LITTLE phone too) and compares the best
  // count so far with half of it and, if half is not faster, with twice as many; it goes on in that direction while
  // the other is faster by more than the noise of a run, and stops at the first that is not. Only the tokens that
  // make logits are timed (a prompt's tokens skip the classifier). One comparison runs the two counts in blocks,
  // best-candidate-candidate-best, so that the growing cost of later positions falls on both alike, and drops the
  // first token of every block (the switch). Helpers that a count needs are started in the background; until they
  // are ready the tokens run on the best count and are not timed.
  const BLOCK = 4, BETTER = 0.95;
  let search = null, chosen = 0, generations = 0, recheckEvery = 0, onChosen = null;
  const searchLog = [];  // every comparison: the counts, their times in ms per token, and the verdict
  const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
  function beginSearch(from) {
    search = { best: Math.max(1, from), direction: from > 1 ? "down" : "up", moved: false, candidate: 0, times: null, step: 0, waiting: false };
    nextCandidate();
  }
  function nextCandidate() {
    const { best, direction } = search;
    const candidate = direction === "down" ? Math.floor(best / 2) : best * 2;
    if (candidate < 1) return finish();
    search.candidate = candidate;
    search.times = { [best]: [], [candidate]: [] };
    search.step = 0;
    if (helpers.length < candidate - 1) {
      search.waiting = true;
      ensureHelpers(candidate).then(() => { if (search) search.waiting = false; }, () => finish());
    }
  }
  function finish() {
    chosen = search ? search.best : threads;
    threads = chosen;
    search = null;
    onChosen?.(chosen);
  }
  // the count for the next token, and whether it is timed
  function countForToken() {
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
  async function ensureHelpers(n) {
    while (helpers.length < n - 1) {
      helpers.push(await spawn({ memory, plain: kernels.plain, relaxed: plan.int8 && plan.relaxed ? kernels.relaxed : null,
        share: helpers.length + 1 }));
    }
  }

  let bound = null;
  const backend = plan.int8 ? `SIMD kernels, int8${relaxed ? ", relaxed SIMD" : ""}` : "SIMD kernels, float32";
  return {
    backend,
    /** Use n threads from the next token on (stage 2): starts the helpers that are missing. 1 on a memory that is
     * not shared. Resolves to the number in use. */
    async setThreads(n) {
      if (!shared) return (threads = 1);
      search = null;
      await ensureHelpers(n);
      threads = Math.max(1, n);
      return threads;
    },
    /** Find the number of threads while generating (see above): from a hint, or from a count remembered from an
     * earlier visit, which is then only checked against its neighbours now and then (every recheck generations).
     * chose(count) is told the answer. The helpers of the starting count are started (and warmed) before this
     * resolves, so the first tokens do not wait for them. */
    async findThreads({ from, remembered = 0, recheck = 8, chose }) {
      if (!shared) return 1;
      onChosen = chose;
      recheckEvery = recheck;
      const start = Math.max(1, remembered || from);
      await ensureHelpers(start);
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
      if (!shared) return;
      Atomics.store(ctl, QUIT, 1);
      for (let h = 1; h <= helpers.length; h++) {
        Atomics.add(ctl, WAKE + h, 2);
        Atomics.notify(ctl, WAKE + h);
      }
      helpers.splice(0).forEach((helper) => helper.terminate?.());
      Atomics.store(ctl, QUIT, 0);
      threads = 1;
    },
    /** the float32 array of Python's that forward() fills with the logits */
    bind(array) {
      bound = array.copy ? array.copy() : array;
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
