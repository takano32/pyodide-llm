// T.357.3: the models of tests/gpu-check.mjs that are made up, a part of that tool: what each is and why (below, and
// beside each: a form is the arguments of tests/made_up.py's quantized(), the options are the engine's), which of them
// an engine runs for "made-up", and where the rounds with a NaN or an infinity run. Data and one function of the
// engine's name: nothing here reads the arguments, a file, Pyodide or a browser.
//
// "synthetic": a made-up int8 model with grouped-query attention (4 heads, 2 of keys and values; none of the models
// of this directory has it). T.153: "synthetic-qwen2", the same with biases of q, k and v (Qwen2's; T.187: drawn around
// 0, 0.3·N, as GPT-2's below) and an epsilon of 1e-6; "synthetic-qwen3", the norms of every head of q and k (Qwen3's), heads of 32 where dim / heads is 16 (q and
// the attention's output 128 wide, dim 64), and an epsilon of 0.5, near mean(x²) (T.150: an epsilon far below it
// hides a wrong one); both of three layers. The review of T.235: "synthetic-yarn", "synthetic-qwen3" with yarn's RoPE (its
// tables are 1.1386 times a unit turn, which no other model here has). T.154: "synthetic-gpt2", GPT-2's form (LayerNorm with biases, a bias after
// every matrix, an FFN of two matrices and GELU, learned positions and no RoPE), and "synthetic-neox", GPT-NeoX's (the
// same with RoPE on the first quarter of every head, as Pythia's rotary_pct 0.25, and the parallel residual), both of
// three layers and 4 heads of keys and values (neither has grouped-query attention), their biases drawn around 0 (the
// review of T.154: a bias of 1 ± 0.03 is nearly the same number everywhere, and LayerNorm's mean takes most of it out);
// "synthetic-neox-256", GPT-NeoX with Pythia 1B's heads of 256 (dim 512, 2 heads), 64 of them turned. T.155:
// "synthetic-6bit", int6 weights (T.98's, which the GPU widens to int8 as it takes them: the shader is shaders.js's
// WIDEN_SIX), and "synthetic-wide", int8 in a 64-bit memory with the checkpoint 4 GiB up (T.101: every address past
// 2^32; the pages below are never touched), both of dim 128 and hidden 320 with the GPU's matrices in pieces of
// 20480 bytes at most (gpuForce.pieceBytes: w1 and w3 in three, w2 in two, the last of w1's rows short), as a matrix
// past a buffer of the device goes. The others are the models of this directory (make models kernels), or <prefix>.json: a
// model tests/perplexity_prepare.py converted (<prefix>.bin, <prefix>.tokenizer.bin, and the options in <prefix>.json;
// gpu-prompt.yml's input real= fetches and converts models of src/models.js so, T.183), whose NumPy answer comes from
// the native Python ($PYTHON, python3 by default).
//
// T.226's review: "synthetic-gpt2-calm" and "synthetic-qwen-calm" (Qwen2's biases and Qwen3's norms of the heads in one
// model: no real model has both, the engine takes them as they come), whose layers' matrices are at 0.1 where the
// others' are at 0.3, and the norms' weights 1 ± 0.3 where the others' are 1 ± 0.03. A matrix of 64 weights a row
// multiplies the size of a vector by 8 × 0.3 = 2.4, or 8 × 0.1 = 0.8: on the others the 8-bit rounding of an
// activation grows from layer to layer (Q8 of the made-up GPT-2's keys and values, 1.1e-2, 5.1e-2 and 7.9e-2 at layers
// 0 to 2), and the packed shaders' line, K_PACKED × Q8, lay above the distance of a GPU that reads another layer's
// vector. NumPy's own answers (reference.mjs's answers, the layers' vectors taken from layer 0 in every layer, 149
// positions; TASKS.md's T.226) put such a GPU at 0.4 to 3.8 of the DP4A line on the made-up GPT-2, 0.4 to 6.9 on
// GPT-NeoX, 0.4 to 2.1 on Qwen2 and 0.3 to 1.2 on Qwen3 (a float form's 3.7 to 111), and on the calm ones at 4.8 and
// more on GPT-2 and 9.2 and more on the Qwen, but for the vectors of q alone (q's bias and norm change the attention's
// weights, and so the next layer's keys, which a calm model's flat attention hardly shows: 1.1 and 0.8 on the DP4A
// line, 4 to 9 on a float form's; q's bias goes by one ADD with k's and v's, and its norm by the code of k's). CI's
// Dawn, the layers' vectors of a step read from layer 0 (gpu.js's tokenPass; runs 36884705557, 36884706717, 36884707864
// and 36884707646): the calm ones' DP4A rows 5.7 to 18.7 of their lines (a correct GPU 0.27 to 0.37), the others' 0.34
// to 1.17 (an RMSNorm model's packed shaders read the norm weights through a uniform of their own, g.u.norm, which
// none of its float forms reads: the FFN's from layer 0 reads 0.36 and 0.57 on Qwen2 and Qwen3, 5.7 to 5.9 on the calm
// Qwen; on GPT-2, whose LayerNorm every form runs apart, 0.85 and 6.8 on the packed shaders).
// T.232: "synthetic-ternary", ternary weights (T.230: two bits a weight and a scale a group of 128, which the GPU takes
// as they are and multiplies by its packed shaders alone: shaders.js's TERNARY_PACKED) in Ternary Bonsai's form, a
// Qwen3 (the norms of the heads of q and k, 8 heads of 32 in a dim of 128: q and the attention's output 256 wide,
// grouped-query attention, an epsilon of 0.5), every row whole groups of 128, three layers, and eight weights of the
// final norm at 12 (outlier channels, T.92, as Ternary Bonsai 1.7B's and 8B's final norms have: the GPU takes them out of
// the classifier's input and adds their columns, shaders.js's TAKE_OUTLIERS and TERNARY_COLUMNS); "synthetic-ternary-calm",
// the same without them (as the 4B) and calm (T.226's review: its packed shaders' lines show a layer's weights or vectors read from another layer);
// "synthetic-ternary-wide", the same in a 64-bit memory with the checkpoint 4 GiB up and the GPU's matrices in pieces
// of 8192 bytes at most (w1 and w3 in two, the second short; w2 in two); and (the review) "synthetic-ternary-untied",
// a dim of 256 (two groups of 128 a row) and a hidden size of 1152 (a token's matrix loops twice), its classifier apart from
// the embedding (as the 8B's), outlier channels in both groups, calm. NumPy's answer widens the ternary weights to
// float32, and Q8 is the arithmetic of the ternary shaders as it is of ORT's DP4A: the matrices' inputs in 8 bits a
// group of 32. Each is also put on the GPU alone (T.156, T.210), but for the one in pieces.
// T.241's review: the rounds that put a NaN or an infinity in a block's row, a step's values and the GPU's weights (the
// quantizers alone run everywhere: one tiny dispatch). T.241 took the full suite's Edge job from 23 to 35 minutes: run
// 36889437902 against 36884862122 (the same models before T.241), the seconds after the GPU is ready lined up by the
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
// T.153: the made-up models of another form (see above). Three layers: a layer's vectors are read at l × their size,
// which a second layer alone would not tell from 0 + size
const TERNARY = { dim: 128, hidden: 384, layers: 3, heads: 8, kv_heads: 4, qk_norm: true, head_dim: 32, dtype: "ternary" };
const TERNARY_OPTIONS = { dtype: "ternary", qk_norm: true, head_dim: 32, rms_norm_eps: 0.5 };
// T.271.1: a rotated basis (T.237) as the options carry one: its block, and the signs of each width a matrix reads as
// llama2_numpy.sign_bits() writes them (a bit a value, 1 for -1, the first value the highest bit of the first byte, in
// hexadecimal), here drawn from a seed (a simple generator: what matters is that they are not all one sign)
const signBits = (width, seed) => {
  let state = seed;
  return Array.from({ length: width / 8 }, () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return ((state >> 8) & 255).toString(16).padStart(2, "0");
  }).join("");
};
const rotatedBasis = (block, widths) => ({ block, signs: Object.fromEntries(widths.map((width, i) => [String(width), signBits(width, 7 + i)])) });
// (TERNARY's widths: dim 128, the attention's output 8 heads of 32, the FFN's 384; blocks of 64, two to six a row)
const ROTATED = rotatedBasis(64, [128, 256, 384]);
// (the untied one's: dim 256, the attention's output 256, the FFN's 1152; blocks of 128, nine of the FFN's)
const ROTATED_UNTIED = rotatedBasis(128, [256, 1152]);
// (T.271.1's review: blocks of 1024 as the 27B's, which the two above do not reach: a thread's second turn over a block's
// values and over its pairs, and 32 threads quantizing. dim 1024, the attention's output 8 heads of 128, the FFN's 2048)
const ROTATED_1024 = rotatedBasis(1024, [1024, 2048]);
const QWEN35_LINEAR = { every: 2, key_heads: 2, value_heads: 6, key_dim: 32, value_dim: 64, conv: 4 };
// (T.547.2.3: the matrices kept as Q4_K blocks; the others are int8)
const Q4K_KINDS = { wq: "Q4_K", wk: "Q4_K", wo: "Q4_K", w1: "Q4_K", w2: "Q4_K", w3: "Q4_K" };
const LFM2_Q4K_KINDS = { wq: "Q4_K", wk: "Q4_K", win: "Q4_K", wout: "Q4_K", w1: "Q4_K", w3: "Q4_K" };
// (T.547.3.1: a stack's layers each of a packing of its own, as Saluki's are: int8 and Q4_K by layer, every matrix's
// layers of both, and in a layer q, k and v, gate and up of two packings)
const Q4K_LAYERS = { wq: ["Q4_K", "int8", "Q4_K"], wk: ["int8", "Q4_K", "Q4_K"], wv: ["int8", "int8", "Q4_K"], wo: ["Q4_K", "Q4_K", "int8"],
  w1: ["int8", "Q4_K", "int8"], w2: ["Q4_K", "int8", "Q4_K"], w3: ["Q4_K", "Q4_K", "int8"] };
// (T.547.3.2's review: whole stacks of two packings in a layer, apart where they differ: q and k apart with gate and up
// one; q and k one with v, gate and up apart)
const Q4K_QK_APART = { wq: "Q4_K", wk: "int8", wv: "int8", wo: "Q4_K", w1: "Q4_K", w2: "Q4_K", w3: "Q4_K" };
const Q4K_UP_APART = { wq: "Q4_K", wk: "Q4_K", wo: "Q4_K", w1: "Q4_K", w2: "Q4_K" };
// (T.547.3.3: Q2_K's blocks, every matrix but v; and stacks of the three packings by layer, Saluki's way)
// (the review: v of Q2_K too, so that q, k and v are one matrix of blocks and a token's Q2_K text writes them turned)
const Q2K_KINDS = { wq: "Q2_K", wk: "Q2_K", wv: "Q2_K", wo: "Q2_K", w1: "Q2_K", w2: "Q2_K", w3: "Q2_K" };
const BLOCK_LAYERS = { wq: ["Q2_K", "Q4_K", "int8"], wk: ["Q4_K", "Q2_K", "Q2_K"], wv: ["int8", "Q2_K", "Q4_K"], wo: ["Q2_K", "int8", "Q4_K"],
  w1: ["Q4_K", "Q2_K", "int8"], w2: ["Q2_K", "Q2_K", "Q4_K"], w3: ["Q2_K", "int8", "Q2_K"] };
// (T.547.3.4: the IQ2 types, which read the codebooks: q, k and v one of IQ2_XS, gate and up one of IQ2_S, o and down
// IQ2_XXS; and stacks of them by layer with the K types and int8)
const IQ2_KINDS = { wq: "IQ2_XS", wk: "IQ2_XS", wv: "IQ2_XS", wo: "IQ2_XXS", w1: "IQ2_S", w2: "IQ2_XXS", w3: "IQ2_S" };
const IQ2_LAYERS = { wq: ["IQ2_XXS", "Q4_K", "IQ2_S"], wk: ["IQ2_XS", "IQ2_XXS", "int8"], wv: ["IQ2_S", "Q2_K", "IQ2_XXS"], wo: ["IQ2_XXS", "IQ2_XS", "Q2_K"],
  w1: ["IQ2_S", "int8", "IQ2_XXS"], w2: ["IQ2_XXS", "IQ2_S", "IQ2_XS"], w3: ["IQ2_XS", "IQ2_XXS", "IQ2_S"] };
// (the review: and an LFM2's, whose stacks of a kind's layers are by their places among that kind's: two attending
// layers' q and k, four convolution layers' matrices in and out)
const LFM2_Q4K_LAYERS = { wq: ["int8", "Q4_K"], wk: ["Q4_K", "int8"], win: ["Q4_K", "int8", "int8", "Q4_K"], wout: ["int8", "Q4_K", "Q4_K", "int8"],
  w1: ["Q4_K", "int8", "Q4_K", "int8", "Q4_K", "int8"], w3: ["int8", "int8", "Q4_K", "Q4_K", "int8", "Q4_K"] };
// (a place's state: value heads × key dim × value dim floats)
const QWEN35_STATE = 6 * 32 * 64 * 4;
const SYNTHETIC = { "synthetic": [{}, {}], "synthetic-qwen2": [{ layers: 3, bias: true }, { bias: true, rms_norm_eps: 1e-6 }],
  "synthetic-qwen3": [{ layers: 3, qk_norm: true, head_dim: 32 }, { qk_norm: true, head_dim: 32, rms_norm_eps: 0.5 }],
  // the review of T.235: Qwen3 with yarn's RoPE (Ternary Bonsai's): the cos and sin of every pair are 1 + 0.1 ln 4 = 1.1386
  // times a unit turn, and the pairs turn as plain at first and slower after (original context 32 of heads of 32: the
  // pairs 0 to 3 a ramp, the rest four times slower). The shaders turn by the table as Python gives it, and a shader that
  // takes the table for a unit turn is wrong on this model only
  "synthetic-yarn": [{ layers: 3, qk_norm: true, head_dim: 32 },
    { qk_norm: true, head_dim: 32, rms_norm_eps: 0.5, rope_scaling: { rope_type: "yarn", factor: 4.0, original_max_position_embeddings: 32 } }],
  // T.255: a SmolLM3's layers that RoPE leaves alone (the second and the last of five: the first layer, which the
  // loading's check computes, turns, as the published 3B's does, and its last, 35, leaves alone: the prompt stops at
  // that layer's keys and values, the step goes on to the classifier from it): the shaders are the Llama's, and a
  // layer's count of turned values is 0 there. A GPU that turns every layer, leaves the wrong ones alone, or treats
  // the last differently, is far from NumPy's in those layers' keys (T.255's review: with the second and the fourth
  // alone, a GPU that turned the last layer anyway was as right as the one that left it, the fifth layer turning)
  "synthetic-smollm3": [{ layers: 5 }, { unturned: [1, 4] }],
  // T.261.4: a Gemma 3 (GEGLU, a norm behind the attention and the FFN, norms of the heads, heads of 32 where dim / heads
  // is 16, one head of keys and values): three local layers of a window of 64 positions with their own base of RoPE
  // (100, where the others' is 10000), then one that attends every position. The 86 tokens of the prompt past the 64th
  // leave positions out, a block of 64 straddles the window's edge and the next is all past it, each row of a tile from
  // its own position; a GPU that attends every position, or turns the local layers by the other tables, or lacks a norm
  // behind or the GELU, is off in the keys of that layer or of the ones after (a window one too wide or too narrow is a
  // 64th of the weights: the loading's check of the attention, a window of 13 on random scores, is what sees that). A
  // context of 160 positions: the CPU keeps every one (a ring, T.510.1, is for a context more than twice window + 16, and
  // a ring's keys and values of the positions past it are no more to read back). The last layer attends every
  // position: the CPU's step after the prompt reads its keys and values
  "synthetic-gemma3": [{ layers: 4, kv_heads: 1, arch: "gemma3", head_dim: 32, seq_len: 160 },
    { arch: "gemma3", head_dim: 32, local: { layers: [0, 1, 2], window: 64, rope_theta: 100 } }],
  // T.545: an LFM2 (convolution layers among attending ones, an LFM2's norms of the heads of q and k), of six layers:
  // two of convolution first (the first, whose matrices the loading's check reads (T.545.2), and one whose rows the second
  // block goes on from), then attending and convolution in turn, the last a convolution layer (a prompt's block leaves
  // its rows alone and the CPU's step after the prompt goes on from them: CONV_STATE without SHORT_CONV). The prompt's
  // 149 tokens are blocks of 64, 64 and 21: each goes on from the rows the one before left, which come back to the CPU
  // and go up again; a GPU that moves the rows by a token too many or too few, reads the taps backwards, or does not
  // clear them at position 0 is off in the keys of the attending layers after (the third layer's and the fifth's), and
  // the CPU's last token and steps, which go on from the rows that came back, are off in the logits
  "synthetic-lfm2": [{ layers: 6, arch: "lfm2", convolution: { layers: "ccacac", taps: 3 } }, { arch: "lfm2", convolution: { layers: "ccacac", taps: 3 } }],
  // T.560: the same LFM2 with a prompt whose last block is shorter than the taps (129 and 130 tokens: blocks of 64, 64
  // and 1 or 2): CONV_STATE keeps the rows the block before left for the rows the block has no token of, and the
  // steps go on from those (count: the tokens of the run, the prompt and the first step's; COUNT where none is said)
  "synthetic-lfm2-short": [{ layers: 6, arch: "lfm2", convolution: { layers: "ccacac", taps: 3 } }, { arch: "lfm2", convolution: { layers: "ccacac", taps: 3 } }, { count: 130 }],
  "synthetic-lfm2-short2": [{ layers: 6, arch: "lfm2", convolution: { layers: "ccacac", taps: 3 } }, { arch: "lfm2", convolution: { layers: "ccacac", taps: 3 } }, { count: 131 }],
  // T.271.2: a Qwen3.5 (linear-attention layers among full-attention ones), of four layers: linear, full, linear, full (the
  // first the loading's check reads, the last attends: the CPU's step after the prompt reads its keys and values), three
  // value heads to a key head (a key head read by the wrong value heads is off), a key dim of 32 and a value dim of 64
  // (the delta rule's threads past the key dim load nothing, and every column of a 64-wide value is one), taps of 4, a
  // gate of the attention's output, RoPE on 8 of a head of 32. The prompt's 149 tokens are blocks of 64, 64 and 21, each
  // going on from the states and rows the one before left (they come back to the CPU and go up again, or the GPU's own:
  // fresh); a GPU that reads the state's rows the wrong way, moves the rows by a token too many or too few, leaves out
  // the decay or a norm, or does not clear the state at position 0 is off in the keys of the full layers after it, and
  // the CPU's last token and steps, which go on from what came back, in the logits
  "synthetic-qwen35": [{ layers: 4, arch: "qwen35", head_dim: 32, linear: QWEN35_LINEAR }, { arch: "qwen35", head_dim: 32, rotary: 8, linear: QWEN35_LINEAR }],
  // the same with a prompt whose last block is shorter than the taps less one (130 tokens: blocks of 64, 64 and 2): the
  // rows keep the block before's for the tokens the block has none of
  "synthetic-qwen35-short": [{ layers: 4, arch: "qwen35", head_dim: 32, linear: QWEN35_LINEAR }, { arch: "qwen35", head_dim: 32, rotary: 8, linear: QWEN35_LINEAR },
    { count: 131 }],
  // T.565: the same with the states in bundles of one place each (as the 27B's 48 places go in bundles past what a
  // phone's GPU binds): the second linear layer's state is the second bundle's, at 0 of it (a place past the first of a
  // bundle is the one-bundle way's, every other model's); a GPU that reads a place's state from the wrong bundle is off
  // from the third layer's on. (Six layers of these values leave the f16 tiles past their line with the states in one
  // buffer or in bundles alike, the numbers the same: run 40071238751; six calm ones too: run 40071452165)
  "synthetic-qwen35-bundles": [{ layers: 4, arch: "qwen35", head_dim: 32, linear: QWEN35_LINEAR }, { arch: "qwen35", head_dim: 32, rotary: 8, linear: QWEN35_LINEAR },
    { force: { stateBytes: QWEN35_STATE } }],
  // T.271.5: Ternary Bonsai 2 27B's form (T.233), small: a Qwen3.5 of ternary weights in a rotated basis, three value
  // heads to a key head. Its linear layers' matrices read R of their input (wqkv and wz of the stream, wout of z's gate
  // of 384), the gates' small matrices the stream as it is (T.271.1)
  "synthetic-qwen35-ternary-rotated": [{ ...TERNARY, layers: 4, qk_norm: false, arch: "qwen35", linear: QWEN35_LINEAR, rotated: ROTATED },
    { dtype: "ternary", arch: "qwen35", head_dim: 32, rotary: 8, linear: QWEN35_LINEAR, rotated: ROTATED }],
  // T.547.2.3: a model of int8 matrices and Q4_K blocks, as a 4-bit Q4_K_M is kept (the 4-bit Shisa: q, k, o and the
  // FFN's gate and up Q4_K, v int8, widened from Q6_K): every row whole blocks of 256 (dim 256, hidden 768), and down
  // Q4_K too (the review: three blocks to its rows, so that a row's blocks after the first and every group of a block
  // are read in the layers as well as in the forms' check). A
  // prompt's matrices go to the GPU by the DP4A tiles, the Q4_K ones by their Q4_K text; its steps stay on the CPU.
  // The blocks are values of 0.3 N quantized (made_up.valued_blocks(): random bytes are weights of one sign mostly)
  "synthetic-q4k": [{ dim: 256, hidden: 768, layers: 3, heads: 4, kv_heads: 2, kinds: Q4K_KINDS, valued: true }, { kinds: Q4K_KINDS }],
  // (T.547.3.1: the same of layers each of a packing of its own, Q4K_LAYERS: a prompt's matrix by its layer's text; its
  // steps on the CPU until T.547.3.2)
  "synthetic-q4k-layers": [{ dim: 256, hidden: 768, layers: 3, heads: 4, kv_heads: 2, kinds: Q4K_LAYERS, valued: true }, { kinds: Q4K_LAYERS }],
  // (T.547.2.4's review: the same wider, rows of 1280 and 1536 weights, 40 and 48 groups of 32: a token's matrix goes
  // round its loop over the row twice, the second at blocks of the row past the first round's; calm, as rows this wide
  // grow the stream past the logits' line otherwise, on the CPU too. Seed 3: of seed 0, NumPy's greedy steps in float32
  // and in the GPU's arithmetic (8-bit inputs, the cache in float16) part at step 2, by 0.14 in log-probability (the
  // GPU took the second's, run 40088736192), as of most seeds of a model this wide; of seed 3 they are the same eight
  // steps, the least margin between the first two 1.38 (.tmp's search, T.547.2.4's review))
  "synthetic-q4k-wide": [{ dim: 1280, hidden: 1536, layers: 2, heads: 10, kv_heads: 2, kinds: Q4K_KINDS, valued: true, calm: true, seed: 3 }, { kinds: Q4K_KINDS }],
  // (and an LFM2 of Q4_K blocks, the 4-bit Shisa's shape: its convolution layers' matrices in and out of Q4_K, q and k
  // of Q4_K with v int8 apart and the norms of the heads, attending layers among the others)
  "synthetic-q2k": [{ dim: 256, hidden: 768, layers: 3, heads: 4, kv_heads: 2, kinds: Q2K_KINDS, valued: true }, { kinds: Q2K_KINDS }],
  "synthetic-blocks-layers": [{ dim: 256, hidden: 768, layers: 3, heads: 4, kv_heads: 2, kinds: BLOCK_LAYERS, valued: true }, { kinds: BLOCK_LAYERS }],
  "synthetic-iq2": [{ dim: 256, hidden: 768, layers: 3, heads: 4, kv_heads: 2, kinds: IQ2_KINDS, valued: true }, { kinds: IQ2_KINDS }],
  "synthetic-iq2-layers": [{ dim: 256, hidden: 768, layers: 3, heads: 4, kv_heads: 2, kinds: IQ2_LAYERS, valued: true }, { kinds: IQ2_LAYERS }],
  "synthetic-q4k-qk-apart": [{ dim: 256, hidden: 768, layers: 3, heads: 4, kv_heads: 2, kinds: Q4K_QK_APART, valued: true }, { kinds: Q4K_QK_APART }],
  "synthetic-q4k-up-apart": [{ dim: 256, hidden: 768, layers: 3, heads: 4, kv_heads: 2, kinds: Q4K_UP_APART, valued: true }, { kinds: Q4K_UP_APART }],
  "synthetic-lfm2-q4k-layers": [{ dim: 256, hidden: 768, layers: 6, heads: 4, kv_heads: 2, arch: "lfm2", convolution: { layers: "ccacac", taps: 3 }, kinds: LFM2_Q4K_LAYERS, valued: true },
    { arch: "lfm2", convolution: { layers: "ccacac", taps: 3 }, kinds: LFM2_Q4K_LAYERS }],
  "synthetic-lfm2-q4k": [{ dim: 256, hidden: 768, layers: 6, heads: 4, kv_heads: 2, arch: "lfm2", convolution: { layers: "ccacac", taps: 3 }, kinds: LFM2_Q4K_KINDS, valued: true },
    { arch: "lfm2", convolution: { layers: "ccacac", taps: 3 }, kinds: LFM2_Q4K_KINDS }],
  "synthetic-gpt2": [{ layers: 3, kv_heads: 4, arch: "gpt2", outliers: 8 }, { arch: "gpt2" }],
  "synthetic-neox": [{ layers: 3, kv_heads: 4, arch: "neox" }, { arch: "neox", rotary: 4, parallel_residual: true }],
  "synthetic-neox-256": [{ dim: 512, hidden: 1024, layers: 3, heads: 2, kv_heads: 2, arch: "neox" }, { arch: "neox", rotary: 64, parallel_residual: true }],
  // T.226's review (calm: see above): a vector read from the wrong layer is far from NumPy's on the packed shaders too
  "synthetic-gpt2-calm": [{ layers: 3, kv_heads: 4, arch: "gpt2", outliers: 8, calm: true }, { arch: "gpt2" }],
  "synthetic-qwen-calm": [{ layers: 3, bias: true, qk_norm: true, head_dim: 32, calm: true }, { bias: true, qk_norm: true, head_dim: 32, rms_norm_eps: 0.5 }],
  // T.155: [form, options, the run's: the GPU's pieces, a 64-bit memory]
  "synthetic-6bit": [{ dim: 128, hidden: 320, layers: 3, dtype: "int6" }, { dtype: "int6" }, { force: { pieceBytes: 20480 } }],
  "synthetic-wide": [{ dim: 128, hidden: 320, layers: 3 }, {}, { force: { pieceBytes: 20480 }, wide: true }],
  // T.232: ternary weights, in Ternary Bonsai's form (see above)
  "synthetic-ternary": [{ ...TERNARY, outliers: 8 }, TERNARY_OPTIONS],
  "synthetic-ternary-calm": [{ ...TERNARY, calm: true }, TERNARY_OPTIONS],
  "synthetic-ternary-wide": [TERNARY, TERNARY_OPTIONS, { force: { pieceBytes: 8192 }, wide: true }],
  // the review of T.232: what the three above cannot tell. Their dim is 128, one group of 128 a row, so the scale of the
  // wrong group of a row, or of a row's stride (n / 128 scales a row: 1 there), in the embedding's shaders and the outlier
  // columns' read the same scale as the right one (mutants of TERNARY_COLUMNS' `channel / 128u` and of the embedding's group
  // passed all three; only the real 1.7B on Dawn, by hand, would catch them); every model is tied, where the 8B's
  // embedding and classifier are two tables (a ternary one each, the embedding on EMBED_TERNARY); and no row is more
  // than 12 groups of the vector's 32, so a token's matrix loops once (32 groups a pass) where the real ones loop 2 to 6
  // times, and a scale indexed by the thread's own column and not by the pass's reads the same. Here: dim 256 (two groups
  // a row, as wo's 256), the classifier apart (vocab -320), eight outlier channels at 0, 37, ... 259 mod 256 (some in each
  // group), a hidden size of 1152 (w2's rows are 9 groups of 128, 36 of 32: a second pass of 4), and calm (T.226's review),
  // so that the packed lines also show a layer's weights read from another layer
  "synthetic-ternary-untied": [{ ...TERNARY, dim: 256, hidden: 1152, vocab: -320, outliers: 8, spread: 37, calm: true }, TERNARY_OPTIONS],
  // T.271.1: the same in a rotated basis, as Ternary Bonsai 2 27B's file is (T.237): every matrix reads R of its input, which
  // the GPU turns as it quantizes it (ROTATE_QUANTIZE), and a token's embedded row is turned back (UNROTATE). No outlier
  // channels (the CPU uses none in a rotated basis). The first: several blocks to a row; the second: a classifier apart,
  // and an FFN of nine blocks
  "synthetic-ternary-rotated": [{ ...TERNARY, rotated: ROTATED }, { ...TERNARY_OPTIONS, rotated: ROTATED }],
  "synthetic-ternary-rotated-untied": [{ ...TERNARY, dim: 256, hidden: 1152, vocab: -320, spread: 37, calm: true, rotated: ROTATED_UNTIED },
    { ...TERNARY_OPTIONS, rotated: ROTATED_UNTIED }],
  "synthetic-ternary-rotated-1024": [{ ...TERNARY, dim: 1024, hidden: 2048, layers: 2, head_dim: 128, rotated: ROTATED_1024 },
    { ...TERNARY_OPTIONS, head_dim: 128, rotated: ROTATED_1024 }] };
// which made-up models an engine runs for "made-up"
// T.241's review (the full suite's time: Chrome 38 and Edge 42 minutes in main's run 36902097346, Edge 23 before T.241): a run
// of a model on SwiftShader costs 5 to 7 s of its shaders' compiling before anything is checked, a made-up model has 14
// runs, and the three browsers run the same SwiftShader. So "made-up" is every made-up model on Dawn (lavapipe, 0.5 s a
// run); on Chromium all but the ones that only change the numbers a form is held to (DAWN_ONLY: yarn's tables, the calm
// models' lines); on Chrome and Edge, which differ from Chromium by the build, the three whose shaders differ (NAN_MODELS:
// Llama's fused layer, Qwen3's norms of the heads, GPT-2's LayerNorm and GELU). A model named is run wherever it is named.
// TASKS.md's T.241 has the minutes (the review of 2026-10-01)
// (T.232: the ternary models' calm one and the one in pieces in a 64-bit memory are Dawn's too: their shaders are the first's;
// the review's untied one too: the tables' shaders are the first's, what differs is the numbers they are given)
const DAWN_ONLY = ["synthetic-yarn", "synthetic-smollm3", "synthetic-gpt2-calm", "synthetic-qwen-calm", "synthetic-ternary-calm", "synthetic-ternary-wide", "synthetic-ternary-untied",
  "synthetic-ternary-rotated-untied", "synthetic-ternary-rotated-1024", "synthetic-qwen35-bundles", "synthetic-qwen35-ternary-rotated", "synthetic-q4k-wide",
  "synthetic-lfm2-q4k", "synthetic-q4k-layers",
  "synthetic-lfm2-q4k-layers", "synthetic-q4k-qk-apart", "synthetic-q4k-up-apart",
  "synthetic-blocks-layers", "synthetic-iq2-layers"];
const madeUpOn = (engine) => Object.keys(SYNTHETIC).filter((name) => engine === "dawn" || (engine === "chromium" ? !DAWN_ONLY.includes(name) : NAN_MODELS.includes(name)));

export { NAN_MODELS, NAN_CHOICES, SYNTHETIC, DAWN_ONLY, madeUpOn };
