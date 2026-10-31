// tests/fakes/gpu-plans.mjs (T.465.1, moved out of tests/unchanged-gpusteps.mjs)
// Made-up models as the model's GPU worker (src/runtime/gpu/) is handed them: a plan as forward/gpuside.ts hands one (a
// Started) over a memory it lays out, small (dim 128, heads of 16). For the checks that run that worker's parts on
// tests/fakes/webgpu.mjs: unchanged-gpusteps.mjs and gpu-forms-check.mjs.
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
export function planned({ kinds, matrices, vectors, more: { positions, ...more } = {} }, direct = false) {
  let next = 4096;
  const take = (bytes) => { const here = next; next += Math.ceil(bytes / 256) * 256; return here; };
  const count = (of) => (of ? kinds.filter((kind) => kind === of).length : kinds.length);
  const places = kinds.map((kind, l) => kinds.slice(0, l).filter((other) => other === kind).length);
  const plan = {
    dim: DIM, hidden: HIDDEN, layers: kinds.length, heads: HEADS, kvHeads: KV_HEADS, headSize: HEAD, turned: HEAD, unturned: [], seqLen: CONTEXT, kvStart: 8,
    eps: 1e-5, layerNorm: false, parallel: false, activation: "silu", windows: kinds.map(() => 0), local: [], localCos: 0, localSin: 0, kinds, places,
    convolution: null, linear: null, batch: BATCH,
    matrices: Object.fromEntries(Object.entries(matrices).map(([name, [rows, n, of]]) =>
      [name, { rows, n, packing: "int8", six: false, ternary: false, layers: Array.from({ length: count(of) }, () => [take(rows * n), take((rows * n / GROUP) * 4)]), ...(of ? { of } : {}) }])),
    vectors: Object.fromEntries(Object.entries(vectors).map(([name, [size, of]]) => [name, { at: take(count(of) * size * 4), size, ...(of ? { of } : {}) }])),
    rows: take(BATCH * DIM * 4), cos: take(CONTEXT * (HEAD / 2) * 4), sin: take(CONTEXT * (HEAD / 2) * 4),
    tokens: { classifier: { rows: VOCAB, n: DIM, packing: "int8", six: false, ternary: false, at: [take(VOCAB * DIM), take((VOCAB * DIM / GROUP) * 4)] }, embedding: null, final: take(DIM * 4),
      finalBias: 0, positions: 0, outliers: false, channels: [], ids: take(4096), most: MOST },
    staging: take(1 << 20), force: {}, remembered: undefined, direct: false, words: { done: 3, failed: 2, beat: 0, wanted: 1 }, rotated: null, ...more,
  };
  if (plan.localCos === true) Object.assign(plan, { localCos: take(CONTEXT * (HEAD / 2) * 4), localSin: take(CONTEXT * (HEAD / 2) * 4) });
  if (positions) plan.tokens.positions = take(CONTEXT * DIM * 4);
  if (plan.layerNorm) plan.tokens.finalBias = take(DIM * 4);
  // (T.271.1: a rotated basis's signs of each width a matrix reads, in blocks of 32)
  if (plan.rotated === true) plan.rotated = { block: 32, signs: [...new Set([DIM, Q_DIM, HIDDEN])].map((width) => ({ width, at: take(width * 4) })) };
  // (T.210: on the GPU alone, the tables come with the checkpoint, and a block embeds its tokens there)
  if (direct) Object.assign(plan, { direct: true, tables: { classifier: plan.tokens.classifier, embedding: null } });
  // (the CPU's cache a step's keys and values go up from: float32 rows, the widest, for each attending layer)
  const cache = kinds.filter((kind) => kind === "attention").map(() => [take(CONTEXT * KV_DIM * 4), take(CONTEXT * KV_DIM * 4)]);
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
  gemma3: { kinds: ["attention", "attention"], matrices: { ...attending(), ...ffn }, vectors: { ...norms, attentionPost: [DIM], ffnPost: [DIM] },
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
// (T.156, T.271.4: on the GPU alone, its weights the checkpoint's as they come: a Llama's and a Qwen3.5's)
export const ALONE = ["llama", "qwen35", "rotated", "qwen35 (states in bundles)"];

