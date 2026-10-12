// T357.3: the models of tests/gpu-check.mjs that are made up, a part of that tool: what each is and why (below, and
// beside each: a form is the arguments of tests/made_up.py's quantized(), the options are the engine's), which of them
// an engine runs for "made-up", and where the rounds with a NaN or an infinity run. Data and one function of the
// engine's name: nothing here reads the arguments, a file, Pyodide or a browser.
//
// "synthetic": a made-up int8 model with grouped-query attention (4 heads, 2 of keys and values; none of the models
// of this directory has it). T153: "synthetic-qwen2", the same with biases of q, k and v (Qwen2's; T187: drawn around
// 0, 0.3·N, as GPT-2's below) and an epsilon of 1e-6; "synthetic-qwen3", the norms of every head of q and k (Qwen3's), heads of 32 where dim / heads is 16 (q and
// the attention's output 128 wide, dim 64), and an epsilon of 0.5, near mean(x²) (T150: an epsilon far below it
// hides a wrong one); both of three layers. The review of T235: "synthetic-yarn", "synthetic-qwen3" with yarn's RoPE (its
// tables are 1.1386 times a unit turn, which no other model here has). T154: "synthetic-gpt2", GPT-2's form (LayerNorm with biases, a bias after
// every matrix, an FFN of two matrices and GELU, learned positions and no RoPE), and "synthetic-neox", GPT-NeoX's (the
// same with RoPE on the first quarter of every head, as Pythia's rotary_pct 0.25, and the parallel residual), both of
// three layers and 4 heads of keys and values (neither has grouped-query attention), their biases drawn around 0 (the
// review of T154: a bias of 1 ± 0.03 is nearly the same number everywhere, and LayerNorm's mean takes most of it out);
// "synthetic-neox-256", GPT-NeoX with Pythia 1B's heads of 256 (dim 512, 2 heads), 64 of them turned. T155:
// "synthetic-6bit", int6 weights (T98's, which the GPU widens to int8 as it takes them: the shader is shaders.js's
// WIDEN_SIX), and "synthetic-wide", int8 in a 64-bit memory with the checkpoint 4 GiB up (T101: every address past
// 2^32; the pages below are never touched), both of dim 128 and hidden 320 with the GPU's matrices in pieces of
// 20480 bytes at most (gpuForce.pieceBytes: w1 and w3 in three, w2 in two, the last of w1's rows short), as a matrix
// past a buffer of the device goes. The others are the models of this directory (make models kernels), or <prefix>.json: a
// model tests/perplexity_prepare.py converted (<prefix>.bin, <prefix>.tokenizer.bin, and the options in <prefix>.json;
// gpu-prompt.yml's input real= fetches and converts models of src/models.js so, T183), whose NumPy answer comes from
// the native Python ($PYTHON, python3 by default).
//
// T226's review: "synthetic-gpt2-calm" and "synthetic-qwen-calm" (Qwen2's biases and Qwen3's norms of the heads in one
// model: no real model has both, the engine takes them as they come), whose layers' matrices are at 0.1 where the
// others' are at 0.3, and the norms' weights 1 ± 0.3 where the others' are 1 ± 0.03. A matrix of 64 weights a row
// multiplies the size of a vector by 8 × 0.3 = 2.4, or 8 × 0.1 = 0.8: on the others the 8-bit rounding of an
// activation grows from layer to layer (Q8 of the made-up GPT-2's keys and values, 1.1e-2, 5.1e-2 and 7.9e-2 at layers
// 0 to 2), and the packed shaders' line, K_PACKED × Q8, lay above the distance of a GPU that reads another layer's
// vector. NumPy's own answers (this file's references, the layers' vectors taken from layer 0 in every layer, 149
// positions; TODO.md's T226) put such a GPU at 0.4 to 3.8 of the DP4A line on the made-up GPT-2, 0.4 to 6.9 on
// GPT-NeoX, 0.4 to 2.1 on Qwen2 and 0.3 to 1.2 on Qwen3 (a float form's 3.7 to 111), and on the calm ones at 4.8 and
// more on GPT-2 and 9.2 and more on the Qwen, but for the vectors of q alone (q's bias and norm change the attention's
// weights, and so the next layer's keys, which a calm model's flat attention hardly shows: 1.1 and 0.8 on the DP4A
// line, 4 to 9 on a float form's; q's bias goes by one ADD with k's and v's, and its norm by the code of k's). CI's
// Dawn, the layers' vectors of a step read from layer 0 (gpu.js's tokenPass; runs 36884705557, 36884706717, 36884707864
// and 36884707646): the calm ones' DP4A rows 5.7 to 18.7 of their lines (a correct GPU 0.27 to 0.37), the others' 0.34
// to 1.17 (an RMSNorm model's packed shaders read the norm weights through a uniform of their own, g.u.norm, which
// none of its float forms reads: the FFN's from layer 0 reads 0.36 and 0.57 on Qwen2 and Qwen3, 5.7 to 5.9 on the calm
// Qwen; on GPT-2, whose LayerNorm every form runs apart, 0.85 and 6.8 on the packed shaders).
// T232: "synthetic-ternary", ternary weights (T230: two bits a weight and a scale a group of 128, which the GPU takes
// as they are and multiplies by its packed shaders alone: shaders.js's TERNARY_PACKED) in Ternary Bonsai's form, a
// Qwen3 (the norms of the heads of q and k, 8 heads of 32 in a dim of 128: q and the attention's output 256 wide,
// grouped-query attention, an epsilon of 0.5), every row whole groups of 128, three layers, and eight weights of the
// final norm at 12 (outlier channels, T92, as Ternary Bonsai 1.7B's and 8B's final norms have: the GPU takes them out of
// the classifier's input and adds their columns, shaders.js's TAKE_OUTLIERS and TERNARY_COLUMNS); "synthetic-ternary-calm",
// the same without them (as the 4B) and calm (T226's review: its packed shaders' lines show a layer's weights or vectors read from another layer);
// "synthetic-ternary-wide", the same in a 64-bit memory with the checkpoint 4 GiB up and the GPU's matrices in pieces
// of 8192 bytes at most (w1 and w3 in two, the second short; w2 in two); and (the review) "synthetic-ternary-untied",
// a dim of 256 (two groups of 128 a row) and a hidden size of 1152 (a token's matrix loops twice), its classifier apart from
// the embedding (as the 8B's), outlier channels in both groups, calm. NumPy's answer widens the ternary weights to
// float32, and Q8 is the arithmetic of the ternary shaders as it is of ORT's DP4A: the matrices' inputs in 8 bits a
// group of 32. Each is also put on the GPU alone (T156, T210), but for the one in pieces.
// T241's review: the rounds that put a NaN or an infinity in a block's row, a step's values and the GPU's weights (the
// quantizers alone run everywhere: one tiny dispatch). T241 took the full suite's Edge job from 23 to 35 minutes: run
// 36889437902 against 36884862122 (the same models before T241), the seconds after the GPU is ready lined up by the
// runner's pace (the ratio of the seconds it took to get ready: 1.27, 0.79 and 0.83 for Edge, Chrome and Chromium) say 3 to
// 6 minutes more in the runs of a SwiftShader job (the made-up model of 512 wide the most: 84 to 143 s), and the 30
// engines of the weights' rounds at about 6 s each, 3.5 minutes (the rest of the 12 minutes was Edge's runner). On
// Dawn's lavapipe a run takes 0.4 s more and the whole growth was 1.7 minutes. A browser's SwiftShader is the same
// compiler in Chromium, Chrome and Edge, and the shaders do not depend on the model (QUANTIZE and NORM_QUANTIZE are one
// each; the three made-up models below differ in the steps around them: Llama's fused layer, Qwen3's norms of the heads,
// GPT-2's LayerNorm and GELU), so: Dawn runs every round on the made-up models, Chromium the rows and the values on those
// three (the weights are lavapipe's), Chrome and Edge none. --nan all|made-up|small|none chooses otherwise (all: the
// models of src/models.js too)
const NAN_MODELS = ["synthetic", "synthetic-qwen3", "synthetic-gpt2"];
const NAN_CHOICES = { all: { models: "all", weights: true }, "made-up": { models: "made-up", weights: true }, small: { models: NAN_MODELS, weights: false },
  none: { models: [], weights: false } };
// T153: the made-up models of another form (see above). Three layers: a layer's vectors are read at l × their size,
// which a second layer alone would not tell from 0 + size
const TERNARY = { dim: 128, hidden: 384, layers: 3, heads: 8, kv_heads: 4, qk_norm: true, head_dim: 32, dtype: "ternary" };
const TERNARY_OPTIONS = { dtype: "ternary", qk_norm: true, head_dim: 32, rms_norm_eps: 0.5 };
const SYNTHETIC = { "synthetic": [{}, {}], "synthetic-qwen2": [{ layers: 3, bias: true }, { bias: true, rms_norm_eps: 1e-6 }],
  "synthetic-qwen3": [{ layers: 3, qk_norm: true, head_dim: 32 }, { qk_norm: true, head_dim: 32, rms_norm_eps: 0.5 }],
  // the review of T235: Qwen3 with yarn's RoPE (Ternary Bonsai's): the cos and sin of every pair are 1 + 0.1 ln 4 = 1.1386
  // times a unit turn, and the pairs turn as plain at first and slower after (original context 32 of heads of 32: the
  // pairs 0 to 3 a ramp, the rest four times slower). The shaders turn by the table as Python gives it, and a shader that
  // takes the table for a unit turn is wrong on this model only
  "synthetic-yarn": [{ layers: 3, qk_norm: true, head_dim: 32 },
    { qk_norm: true, head_dim: 32, rms_norm_eps: 0.5, rope_scaling: { rope_type: "yarn", factor: 4.0, original_max_position_embeddings: 32 } }],
  // T255: a SmolLM3's layers that RoPE leaves alone (the second and the last of five: the first layer, which the
  // loading's check computes, turns, as the published 3B's does, and its last, 35, leaves alone: the prompt stops at
  // that layer's keys and values, the step goes on to the classifier from it): the shaders are the Llama's, and a
  // layer's count of turned values is 0 there. A GPU that turns every layer, leaves the wrong ones alone, or treats
  // the last differently, is far from NumPy's in those layers' keys (T255's review: with the second and the fourth
  // alone, a GPU that turned the last layer anyway was as right as the one that left it, the fifth layer turning)
  "synthetic-smollm3": [{ layers: 5 }, { unturned: [1, 4] }],
  "synthetic-gpt2": [{ layers: 3, kv_heads: 4, arch: "gpt2", outliers: 8 }, { arch: "gpt2" }],
  "synthetic-neox": [{ layers: 3, kv_heads: 4, arch: "neox" }, { arch: "neox", rotary: 4, parallel_residual: true }],
  "synthetic-neox-256": [{ dim: 512, hidden: 1024, layers: 3, heads: 2, kv_heads: 2, arch: "neox" }, { arch: "neox", rotary: 64, parallel_residual: true }],
  // T226's review (calm: see above): a vector read from the wrong layer is far from NumPy's on the packed shaders too
  "synthetic-gpt2-calm": [{ layers: 3, kv_heads: 4, arch: "gpt2", outliers: 8, calm: true }, { arch: "gpt2" }],
  "synthetic-qwen-calm": [{ layers: 3, bias: true, qk_norm: true, head_dim: 32, calm: true }, { bias: true, qk_norm: true, head_dim: 32, rms_norm_eps: 0.5 }],
  // T155: [form, options, the run's: the GPU's pieces, a 64-bit memory]
  "synthetic-6bit": [{ dim: 128, hidden: 320, layers: 3, dtype: "int6" }, { dtype: "int6" }, { force: { pieceBytes: 20480 } }],
  "synthetic-wide": [{ dim: 128, hidden: 320, layers: 3 }, {}, { force: { pieceBytes: 20480 }, wide: true }],
  // T232: ternary weights, in Ternary Bonsai's form (see above)
  "synthetic-ternary": [{ ...TERNARY, outliers: 8 }, TERNARY_OPTIONS],
  "synthetic-ternary-calm": [{ ...TERNARY, calm: true }, TERNARY_OPTIONS],
  "synthetic-ternary-wide": [TERNARY, TERNARY_OPTIONS, { force: { pieceBytes: 8192 }, wide: true }],
  // the review of T232: what the three above cannot tell. Their dim is 128, one group of 128 a row, so the scale of the
  // wrong group of a row, or of a row's stride (n / 128 scales a row: 1 there), in the embedding's shaders and the outlier
  // columns' read the same scale as the right one (mutants of TERNARY_COLUMNS' `channel / 128u` and of the embedding's group
  // passed all three; only the real 1.7B on Dawn, by hand, would catch them); every model is tied, where the 8B's
  // embedding and classifier are two tables (a ternary one each, the embedding on EMBED_TERNARY); and no row is more
  // than 12 groups of the vector's 32, so a token's matrix loops once (32 groups a pass) where the real ones loop 2 to 6
  // times, and a scale indexed by the thread's own column and not by the pass's reads the same. Here: dim 256 (two groups
  // a row, as wo's 256), the classifier apart (vocab -320), eight outlier channels at 0, 37, ... 259 mod 256 (some in each
  // group), a hidden size of 1152 (w2's rows are 9 groups of 128, 36 of 32: a second pass of 4), and calm (T226's review),
  // so that the packed lines also show a layer's weights read from another layer
  "synthetic-ternary-untied": [{ ...TERNARY, dim: 256, hidden: 1152, vocab: -320, outliers: 8, spread: 37, calm: true }, TERNARY_OPTIONS] };
// which made-up models an engine runs for "made-up"
// T241's review (the full suite's time: Chrome 38 and Edge 42 minutes in main's run 36902097346, Edge 23 before T241): a run
// of a model on SwiftShader costs 5 to 7 s of its shaders' compiling before anything is checked, a made-up model has 14
// runs, and the three browsers run the same SwiftShader. So "made-up" is every made-up model on Dawn (lavapipe, 0.5 s a
// run); on Chromium all but the ones that only change the numbers a form is held to (DAWN_ONLY: yarn's tables, the calm
// models' lines); on Chrome and Edge, which differ from Chromium by the build, the three whose shaders differ (NAN_MODELS:
// Llama's fused layer, Qwen3's norms of the heads, GPT-2's LayerNorm and GELU). A model named is run wherever it is named.
// TODO.md's T241 has the minutes (the review of 2026-10-01)
// (T232: the ternary models' calm one and the one in pieces in a 64-bit memory are Dawn's too: their shaders are the first's;
// the review's untied one too: the tables' shaders are the first's, what differs is the numbers they are given)
const DAWN_ONLY = ["synthetic-yarn", "synthetic-smollm3", "synthetic-gpt2-calm", "synthetic-qwen-calm", "synthetic-ternary-calm", "synthetic-ternary-wide", "synthetic-ternary-untied"];
const madeUpOn = (engine) => Object.keys(SYNTHETIC).filter((name) => engine === "dawn" || (engine === "chromium" ? !DAWN_ONLY.includes(name) : NAN_MODELS.includes(name)));

export { NAN_MODELS, NAN_CHOICES, SYNTHETIC, DAWN_ONLY, madeUpOn };
