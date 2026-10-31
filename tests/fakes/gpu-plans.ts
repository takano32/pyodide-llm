// tests/fakes/gpu-plans.ts (T.465.1, moved out of tests/unchanged-gpusteps.mjs)
// Made-up models as the model's GPU worker (src/runtime/gpu/) is handed them: a plan as forward/gpuside.ts hands one (a
// Started) over a memory it lays out, small (dim 128, heads of 16). For the checks that run that worker's parts on
// tests/fakes/webgpu.mjs: unchanged-gpusteps.mjs and gpu-forms-check.ts.
//
//   const { plan, bytes, cache } = planned(MODELS.llama, direct)
//
// The same as a real plan: every key the worker reads, every matrix int8 with its scales, tables and vectors where
// their bytes are. Simpler: no model's numbers (the memory is whatever a check writes there), one size of everything.
export const DIM = 128, HIDDEN = 256, HEADS = 2, KV_HEADS = 1, HEAD = 16, BATCH = 8, CONTEXT = 64, VOCAB = 96, GROUP = 32, MOST = 4;
export const Q_DIM = HEADS * HEAD, KV_DIM = KV_HEADS * HEAD, KV_ROW = KV_DIM * 2;
export const LINEAR = { keyHeads: 1, valueHeads: 2, keyDim: 16, valueDim: 16, taps: 4 }, KEYS = LINEAR.keyHeads * LINEAR.keyDim, READ = LINEAR.valueHeads * LINEAR.valueDim;
export const MIXED = 2 * KEYS + READ, CONV_TAPS = 3;

// (T.547.1: a matrix's and a table's packing by name; six and ternary besides, which the trees before it read: they can
// go once the net's other tree is never one before 351fc6cb)
/** A plan as forward/gpuside.ts hands one (a Started), over a memory it lays out: kinds, every layer's kind; more, the
 * model's own; matrices and vectors, [rows, n, of] and [size, of] by name. */
// a made-up model: every layer's kind, its matrices [rows, n, the kind of the layers it is in] and vectors [size, kind]
// by name, and the plan's keys of its own (positions: a GPT-2's table of learned positions)
type Made = { kinds: string[]; matrices: Record<string, (number | string | undefined)[]>; vectors: Record<string, (number | string | undefined)[]>;
  more?: Record<string, unknown> & { positions?: boolean } };
export function planned({ kinds, matrices, vectors, more: { positions, ...more } = {} }: Made, direct = false) {
  // (T.465.6.5: the widths of a model whose more says its own: the wide Llama's)
  const dim = (more.dim as number | undefined) ?? DIM, hidden = (more.hidden as number | undefined) ?? HIDDEN;
  let next = 4096;
  const take = (bytes) => { const here = next; next += Math.ceil(bytes / 256) * 256; return here; };
  const count = (of) => (of ? kinds.filter((kind) => kind === of).length : kinds.length);
  const places = kinds.map((kind, l) => kinds.slice(0, l).filter((other) => other === kind).length);
  const plan = {
    dim, hidden, layers: kinds.length, heads: HEADS, kvHeads: KV_HEADS, headSize: HEAD, turned: HEAD, unturned: [], seqLen: CONTEXT, kvStart: 8,
    eps: 1e-5, layerNorm: false, parallel: false, activation: "silu", windows: kinds.map(() => 0), local: [], localCos: 0, localSin: 0, kinds, places,
    convolution: null, linear: null, batch: BATCH,
    matrices: Object.fromEntries(Object.entries(matrices).map(([name, [rows, n, of]]) =>
      [name, { rows, n, packing: "int8", six: false, ternary: false, layers: Array.from({ length: count(of) }, () => [take(rows * n), take((rows * n / GROUP) * 4)]), ...(of ? { of } : {}) }])),
    vectors: Object.fromEntries(Object.entries(vectors).map(([name, [size, of]]) => [name, { at: take(count(of) * size * 4), size, ...(of ? { of } : {}) }])),
    rows: take(BATCH * dim * 4), cos: take(CONTEXT * (HEAD / 2) * 4), sin: take(CONTEXT * (HEAD / 2) * 4),
    tokens: { classifier: { rows: VOCAB, n: dim, packing: "int8", six: false, ternary: false, at: [take(VOCAB * dim), take((VOCAB * dim / GROUP) * 4)] }, embedding: null, final: take(dim * 4),
      finalBias: 0, positions: 0, outliers: false, channels: [], ids: take(4096), most: MOST },
    staging: take(1 << 20), force: {}, remembered: undefined, direct: false, words: { done: 3, failed: 2, beat: 0, wanted: 1 }, rotated: null, ...more,
  };
  if (plan.localCos === true) Object.assign(plan, { localCos: take(CONTEXT * (HEAD / 2) * 4), localSin: take(CONTEXT * (HEAD / 2) * 4) });
  if (positions) plan.tokens.positions = take(CONTEXT * dim * 4);
  if (plan.layerNorm) plan.tokens.finalBias = take(dim * 4);
  // (T.271.1: a rotated basis's signs of each width a matrix reads, in blocks of 32)
  if (plan.rotated === true) plan.rotated = { block: 32, signs: [...new Set([DIM, Q_DIM, HIDDEN])].map((width) => ({ width, at: take(width * 4) })) };
  // (T.210: on the GPU alone, the tables come with the checkpoint, and a block embeds its tokens there)
  if (direct) Object.assign(plan, { direct: true, tables: { classifier: plan.tokens.classifier, embedding: null } });
  // (the CPU's cache a step's keys and values go up from: float32 rows, the widest, for each attending layer, of the
  // model's own keys: T.465.6.5, a model whose more says other heads)
  const kvDim = plan.kvHeads * plan.headSize;
  const cache = kinds.filter((kind) => kind === "attention").map(() => [take(CONTEXT * kvDim * 4), take(CONTEXT * kvDim * 4)]);
  return { plan, bytes: next, cache };
}
const ffn = { w1: [HIDDEN, DIM], w3: [HIDDEN, DIM], w2: [DIM, HIDDEN] };
const attending = (of) => ({ wq: [Q_DIM, DIM, of], wk: [KV_DIM, DIM, of], wv: [KV_DIM, DIM, of], wo: [DIM, Q_DIM, of] });
const norms = { attention: [DIM], ffn: [DIM] };
// (GPT-2's: LayerNorms with their biases, biases of every matrix, no gate, learned positions, nothing turned; GPT-NeoX's:
// a part of a head turned and the parallel residual)
const gpt2 = { matrices: { ...attending(), w1: [HIDDEN, DIM], w2: [DIM, HIDDEN] },
  vectors: { ...norms, attentionBias: [DIM], ffnBias: [DIM], bq: [Q_DIM], bk: [KV_DIM], bv: [KV_DIM], bo: [DIM], b1: [HIDDEN], b2: [DIM] } };
export const MODELS = {
  llama: { kinds: ["attention", "attention"], matrices: { ...attending(), ...ffn }, vectors: norms },
  qwen3: { kinds: ["attention", "attention"], matrices: { ...attending(), ...ffn },
    vectors: { ...norms, bq: [Q_DIM], bk: [KV_DIM], bv: [KV_DIM], qNorm: [HEAD], kNorm: [HEAD] } },
  gemma3: { kinds: ["attention", "attention"], matrices: { ...attending(), ...ffn }, vectors: { ...norms, attentionPost: [DIM], ffnPost: [DIM], qNorm: [HEAD], kNorm: [HEAD] },
    more: { activation: "gelu", windows: [4, 0], local: [0], localCos: true } },
  lfm2: { kinds: ["convolution", "attention", "convolution"], matrices: { ...attending("attention"), win: [3 * DIM, DIM, "convolution"], wout: [DIM, DIM, "convolution"], ...ffn },
    vectors: { ...norms, conv: [CONV_TAPS * DIM, "convolution"], qNorm: [HEAD, "attention"], kNorm: [HEAD, "attention"] }, more: { convolution: { taps: CONV_TAPS } } },
  qwen35: { kinds: ["linear", "linear", "attention"],
    matrices: { wqkv: [MIXED, DIM, "linear"], wz: [READ, DIM, "linear"], wout: [DIM, READ, "linear"], ...attending("attention"), wg: [Q_DIM, DIM, "attention"], ...ffn },
    vectors: { ...norms, conv: [LINEAR.taps * MIXED, "linear"], wb: [LINEAR.valueHeads * DIM, "linear"], wa: [LINEAR.valueHeads * DIM, "linear"],
      dtBias: [LINEAR.valueHeads, "linear"], decay: [LINEAR.valueHeads, "linear"], deltaNorm: [LINEAR.valueDim, "linear"], qNorm: [HEAD, "attention"], kNorm: [HEAD, "attention"] },
    more: { linear: LINEAR } },
  gpt2: { kinds: ["attention", "attention"], ...gpt2, more: { layerNorm: true, turned: 0, activation: "gelu", positions: true } },
  neox: { kinds: ["attention", "attention"], ...gpt2, more: { layerNorm: true, turned: HEAD / 2, parallel: true, activation: "gelu" } },
  // T.271.1: a Llama in a rotated basis (every matrix's input turned as it is quantized, a token's embedded row turned back)
  rotated: { kinds: ["attention", "attention"], matrices: { ...attending(), ...ffn }, vectors: norms, more: { rotated: true } },
};
// T.565: the Qwen3.5 with its linear layers' states in bundles of one place each (as the 27B's go past what a phone's
// GPU binds)
MODELS["qwen35 (states in bundles)"] = { ...MODELS.qwen35, more: { linear: LINEAR, force: { stateBytes: LINEAR.valueHeads * LINEAR.keyDim * LINEAR.valueDim * 4 } } };
// T.465.6.3: a Llama of 8 heads on 4 of keys and values (q 128 rows, k and v 64 each), for ternary weights: a ternary row is
// whole groups of 128 (o reads q's width: the 32 above are a quarter of one), and takes 4 bytes of scales (a float32 a
// group), so that the 32 rows of q, k and v above end off where a device binds a buffer (every 256 bytes) and a token's
// layers are not joined (m.weights.unjoined); a real model's are far more
MODELS["llama (8 heads)"] = { kinds: ["attention", "attention"],
  matrices: { wq: [128, DIM], wk: [64, DIM], wv: [64, DIM], wo: [DIM, 128], ...ffn }, vectors: norms, more: { heads: 8, kvHeads: 4 } };
// T.465.6.5: a wide Llama (dim 256, an FFN of 512, 8 heads of 32 on 4 of keys and values), whose every matrix and table is
// rows of 256 and more, which llama.cpp's blocks take (each matrix of its own type: q, k and v, gate and up apart)
MODELS["llama (wide)"] = { kinds: ["attention", "attention"],
  matrices: { wq: [256, 256], wk: [128, 256], wv: [128, 256], wo: [256, 256], w1: [512, 256], w3: [512, 256], w2: [256, 512] },
  vectors: { attention: [256], ffn: [256] }, more: { dim: 256, hidden: 512, heads: 8, kvHeads: 4, headSize: 32, turned: 32 } };
// (T.156, T.271.4: on the GPU alone, its weights the checkpoint's as they come: a Llama's and a Qwen3.5's)
export const ALONE = ["llama", "qwen35", "rotated", "qwen35 (states in bundles)"];


// T.465.4: numbers a model could have, into the memory a plan lays out (whose bytes are arbitrary where nothing writes:
// a float32 of arbitrary bits may be NaN or 1e38, and a check's own reference is then NaN too): every int8 matrix's
// and table's scales small (1e-2 to 2e-2), every vector near 1 (norms; a bias near 0 would be as good), and RoPE's
// tables the cos and sin of a position's angles (base 10000). The int8 values stay the bytes they are.
export function plausible(plan, memory: { buffer: ArrayBufferLike }) {
  const floats = (at: number, n: number, value: (i: number) => number) => {
    const view = new Float32Array(memory.buffer, at, n);
    for (let i = 0; i < n; i++) view[i] = value(i);
  };
  const small = (i: number) => 0.01 + 0.01 * (((i * 2654435761) >>> 0) % 1000) / 1000;
  // (T.465.6.3: a ternary matrix's scales are float16s, one a group of 128: 1/64 to 1/32, as halves)
  const halves = (at: number, n: number) => {
    const view = new Uint16Array(memory.buffer, at, n);
    for (let i = 0; i < n; i++) view[i] = 0x2400 + (((i * 2654435761) >>> 0) % 1024);
  };
  for (const matrix of Object.values(plan.matrices) as { rows: number; n: number; packing: string; layers: [number, number][] }[]) {
    for (const [, scales] of matrix.layers) {
      if (matrix.packing === "ternary") halves(scales, (matrix.rows * matrix.n) / (4 * GROUP));
      else floats(scales, (matrix.rows * matrix.n) / GROUP, small);
    }
  }
  for (const table of [plan.tokens.classifier, plan.tokens.embedding].filter(Boolean)) {
    if (table.packing === "ternary") halves(table.at[1], (table.rows * table.n) / (4 * GROUP));
    else floats(table.at[1], (table.rows * table.n) / GROUP, small);
  }
  // (each vector its own values, T.465.6.2's review: vectors of one size all alike let one be read for another, a norm's
  // weight for the other's or one bias for another, and the check passed)
  (Object.values(plan.vectors) as { at: number; size: number }[]).forEach((vector, k) => floats(vector.at, vector.size * kindsOf(plan, vector), (i) => 1 + 0.01 * ((i + 3 * k) % 7) + 0.02 * k));
  floats(plan.tokens.final, plan.dim, (i) => 1 + 0.01 * (i % 5));
  // (T.465.6.2: a LayerNorm's final bias, small; GPT-2's learned positions as large as a row of the embedding, ±0.5:
  // positions of 1e-2 left out of a token's pass passed the check)
  if (plan.tokens.finalBias) floats(plan.tokens.finalBias, plan.dim, (i) => 0.01 * ((i % 9) - 4));
  if (plan.tokens.positions) floats(plan.tokens.positions, CONTEXT * plan.dim, (i) => 100 * small(i) - 1.5);
  const half = plan.headSize / 2;
  // (and a Gemma 3's local layers' own, where the plan has them: of another base, any whose angles differ)
  for (const [at, f, base] of [[plan.cos, Math.cos, 10000], [plan.sin, Math.sin, 10000], [plan.localCos, Math.cos, 100], [plan.localSin, Math.sin, 100]] as const) {
    if (at) floats(at, CONTEXT * half, (i) => f(Math.floor(i / half) * base ** (-(i % half) / half)));
  }
}
// (the layers a vector has a copy of: of its kind's, or all)
const kindsOf = (plan, vector) => (vector.of ? plan.kinds.filter((kind) => kind === vector.of).length : plan.layers);
