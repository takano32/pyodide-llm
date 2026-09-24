// forward.js (T93): one token's forward pass in JavaScript, on the SIMD kernels of kernels/*.ts, on a WebAssembly
// memory of its own that holds the weights. Python (llama2_numpy.Llama with external=) still reads the header, says
// where every tensor is, tokenizes, samples and runs the generation loop; this file does what kernel_forward() did,
// in the same order with the same kernels, so the numbers are the same. What it saves is the Python between the
// kernel calls (about 100 per token): 1.16 to 1.42 times the speed, measured with tests/threads-prototype.
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

/** A memory with room for a checkpoint of size bytes at base; the forward pass allocates after it. */
export function weightsMemory(size, base = 64) {
  const memory = new WebAssembly.Memory({ initial: Math.ceil((base + size) / PAGE) + 1 });
  return { memory, base };
}

/** What Llama(external=) takes: the size of the checkpoint, read() for the few bytes Python looks at itself, and
 * start(plan), which builds the forward pass. */
export function external({ memory, base, size, kernels }) {
  return {
    size,
    read: (offset, length) => new Uint8Array(memory.buffer, base + offset, length).slice(),
    start: (plan) => createForward({ memory, base, size, kernels, plan: plan.toJs ? plan.toJs({ dict_converter: Object.fromEntries }) : plan }),
  };
}

// half to float, exactly (the same as NumPy's astype(float32))
function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

export function createForward({ memory, base, size, kernels, plan }) {
  const { dim, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: headSize, vocab_size: vocab,
    seq_len: seqLen, rotary, arch } = plan;
  const hidden = plan.hidden_dim, kvDim = kvHeads * headSize;
  const gpt2 = arch === "gpt2", layerNorm = arch === "gpt2" || arch === "neox", parallel = plan.parallel_residual;
  const imports = { env: { memory } };
  const k = new WebAssembly.Instance(kernels.plain, imports).exports;
  const relaxed = plan.int8 && plan.relaxed && kernels.relaxed ? new WebAssembly.Instance(kernels.relaxed, imports).exports.matmul_q8r : null;
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
        // dot(w, q - 64) = dot(w, q) - 64 * sum(w). As kernel_forward(): scale * sum of the group, in float32
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

  // ---- the activations, as kernel_forward() has them
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

  function matmul(m, out, input, l, sameInput = false) {
    const [w, s, c] = m.layer(l);
    if (!m.int8) {
      k.matmul_f32(out, input, w, m.n, 0, m.rows);
      return;
    }
    if (!sameInput) k.quantize_x(xq, xs, input, m.n, bias);  // q, k, v (and w1, w3) share their input
    if (relaxed) relaxed(out, xq, xs, w, s, c, m.n, 0, m.rows);
    else k.matmul_q8(out, xq, xs, w, s, m.n, 0, m.rows);
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
      matmul(wq, q, xb, l);
      matmul(wk, kp, xb, l, true);
      matmul(wv, vp, xb, l, true);
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
      matmul(wo, xb2, xb, l);
      k.add_inplace(x, xb2, dim);
      if (layerNorm) {
        k.add_inplace(x, bo + l * dim * 4, dim);
        k.layernorm(xb, parallel ? before : x, ffnW + l * dim * 4, ffnB + l * dim * 4, dim);
        matmul(w1, hb, xb, l);
        k.gelu(hb, hb, b1 + l * hidden * 4, hidden);
        matmul(w2, xb2, hb, l);
        k.add_inplace(x, xb2, dim);
        k.add_inplace(x, b2 + l * dim * 4, dim);
        continue;
      }
      k.rmsnorm(xb, x, ffnW + l * dim * 4, dim);
      matmul(w1, hb, xb, l);
      matmul(w3, hb2, xb, l, true);
      k.swiglu(hb, hb, hb2, hidden);
      matmul(w2, xb2, hb, l);
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
      matmul(wcls, logits, xb, 0);
      k.add_columns(logits, columns, picked, channels.length, vocab);
    } else {
      matmul(wcls, logits, xb, 0);
    }
  }

  let bound = null;
  return {
    backend: plan.int8 ? `SIMD kernels, int8${relaxed ? ", relaxed SIMD" : ""}` : "SIMD kernels, float32",
    /** the float32 array of Python's that forward() fills with the logits */
    bind(array) {
      bound = array.copy ? array.copy() : array;
    },
    forward(token, pos, needLogits = true) {
      forward(token, pos, needLogits);
      if (!needLogits || !bound) return;
      const view = bound.getBuffer ? bound.getBuffer("f32") : { data: bound, release() {} };
      view.data.set(new Float32Array(memory.buffer, logits, vocab));
      view.release();
    },
    /** the logits in this memory, for callers without Python (tests) */
    logits: () => new Float32Array(memory.buffer, logits, vocab),
    memoryBytes: () => memory.buffer.byteLength,
  };
}
