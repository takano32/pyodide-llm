// T.357.3: the lines of tests/gpu-check.ts, a part of that tool: the sizes of the prompt and of the steps, the status
// lines of a refusal, and the numbers a run's keys and values, scale and logits are held to, each with where it
// comes from. Constants only.
// T.147: 150 tokens, so that the GPU's blocks of 64 are two and a part (the tiles' ends), and the caches grow to 256
const COUNT = 150, KV_START = 8;
// T.152: the greedy steps NumPy takes after the prompt
const GEN = 8;
// T.219: the status line of a step refused for its logits (forward.js's generateMany)
const NOT_FINITE_STATUS = /^prompts on the CPU \(the GPU computed logits that are not finite numbers/;
// T.243: and of a block of a prompt refused for its keys and values (forward/gpuside.js's promptOnGpu)
const NOT_FINITE_KV_STATUS = /^prompts on the CPU \(the GPU computed keys or values that are not finite numbers \(NaN or infinity\) in a block of the prompt at position 0\)/;
// The worst row of the keys and values against NumPy's, by what the matrices' shader computes in (T.147, measured on
// Dawn's lavapipe and this machine's SwiftShader, 149 tokens: two blocks of 64 and a part). The CPU's forward.js: 4.1e-2
// to 3.2e-1 (its 7-bit activations; the made-up model's random weights the most). A GPU that is wrong lands far past
// the lines (broken on purpose: RoPE at the next position 1.13 to 1.16, no causal mask 1.86 to 6.97, the keys written
// back in the order [token][layer] 2.50 to 4.62, T.135; T.147's six in TASKS.md).
//   float32 (TF.js's tiles, llama.cpp's f32): 8.1e-4 to 2.4e-3 for the models here, 3.90e-3 for the made-up one: the
//     float16 of the cache (2^-11 = 4.9e-4 of a value) and the attention on float16 keys and values from the first
//     layer on, as the CPU's (T.135's GPU read its own float32 ones: 4.5e-4 to 4.7e-4 at 39 tokens), growing with the
//     positions (the made-up model 1.56e-3 at 39 tokens). The line: 8e-3.
//   f16 (llama.cpp's f16: every weight × scale and activation rounded to 11 bits): 9.5e-4 to 6.1e-3. The line: 1.5e-2.
//   8 bits (ORT's DP4A: the activations quantized as the CPU's matmul_q8 takes them, 8 bits where the CPU's relaxed
//     SIMD takes 7): held to Q8 (T.187, below).
// T.153: those lines hold for the first layer of every model and for all layers of a shallow one. Over the layers of a
// deep one the float16 of the cache grows by itself, the more so with Qwen's large activations: E16 (NumPy's answer
// with nothing but its cache rounded to float16, against NumPy's) is its measure. A GPU that computes as it should is a
// few E16 away, whatever the CPU's forward.js does (a line of a quarter of the CPU's error loosened the made-up models'
// lines 1.3 to 10 times, and a GPU that used layer 0's biases in every layer passed: T.153's review). So every layer of
// a float32 or f16 form is held to the shader's line or K × E16, whichever is larger, and the first layer to the
// shader's line alone. K (CI's Dawn on lavapipe, run 36308518805, 149 tokens): the float32 shaders are 0.9 to 1.6 E16
// away (Qwen3 0.6B the most), llama.cpp's f16 0.9 to 3.2 and 5.5 (Qwen3 0.6B with the f16 attention), while a GPU that
// reads layer 0's biases in every layer (synthetic-qwen2) is 13.9 E16 away on float32 and 14.4 on f16: K is 4 and 8 (T.562: the f16 forms are held to H16 now, below).
// T.187: a layer at a time, the line of layer l max(shader's line, K × E16 of layer l): against the largest E16 of all
// layers, a deep model's early layers had a line of the late ones (Qwen3 0.6B's DP4A 2.78 where its layers 0 to 7 read
// 0.026 to 0.28 by their own). A GPU as it should be went to 0.75 of the lines a layer (runs 36325111865 and
// 36325960845, and on lavapipe's subgroups of 16, 36325111922: Qwen3 0.6B's f16 with the f16 attention at layer 12; the made-up models 0.66
// at most). The made-up Qwen2 with its biases around 0 (T.187): a GPU that reads layer 0's biases in every layer is 158
// E16 away on float32 and f16, 39.6 and 19.8 times its layer's line (run 36325111772).
const GPU_LINE = 8e-3, HALF_LINE = 1.5e-2;
const K = { float32: 4 };
// T.187: none of the lines is a ratio to the CPU's error (T.147 to T.153 held DP4A to 0.75 of it and the logits to 1.5
// times it: a correct DP4A went to 0.98 of its line on Qwen2.5 0.5B, one that read layer 0's biases in every layer
// passed at 0.55, and GPT-2's logits went past 1.5 on keys and values that were NumPy's own; they moved wherever the
// CPU's arithmetic did). Each is held to a yardstick of its own. The numbers are CI's Dawn on lavapipe (runs
// 36314674862 and 36314674849, 149 tokens, the made-up models, Qwen2.5 0.5B, Qwen3 0.6B, llm-jp-3 150M, SmolLM2 360M,
// and with T.154 GPT-2 124M and Pythia 160M; TASKS.md's T.187):
//   - the packed shaders (8-bit activations) to Q8: NumPy's answer with the inputs of the layers' matrices quantized
//     as quantize_x does it (8 bits, 32 a group, half to even) and its cache in float16, which is ORT's DP4A's
//     arithmetic. Their first layer against Q8's: 8.1e-5 to 2.9e-3 (Qwen3 0.6B the most), the line GPU_LINE (a
//     LayerNorm's mean over n − 1: 1.57e-2 and 1.93e-2). Past the first layer the 8-bit roundings that fall on the
//     other side of a boundary part them from Q8 as far as Q8 is from NumPy's (a GPU as it should be 0.25 to 2.6 Q8
//     from Q8), so all their layers are held against NumPy's: 0.49 to 1.40 times Q8's distance from it (GPT-2 124M
//     the most, its outlier channels). A layer at a time (the review of T.187), the line of layer l is K_PACKED × Q8's
//     distance at layer l: the review read a GPU as it should be at 2.0 times it on lavapipe's subgroups of 16 from its
//     tables' two digits (Qwen2.5 0.5B's layer 17, Qwen3 0.6B's layer 27), so K_PACKED is 3; the runs read 1.44 at
//     most (0.48 of the line, SmolLM2 360M's layer 13, run 36325960845; 1.17 on subgroups of 16, run 36325111922), the
//     made-up models 1.02. A bias read from
//     layer 0 in every layer went under a line of 2 on the made-up Qwen2 while its biases were near 1 in every layer
//     (0.98), and goes past its layer's line 2.34 times (7.02 Q8 over all layers) with biases drawn around 0 (0.3·N; a
//     correct one 1.01 Q8, 0.34 of its lines), and 2534 times on Qwen2.5 0.5B (run 36325111772).
//   - the scale of the first layer's keys and values (the norm's scale wrong by a little, as a mean of squares or a
//     variance over n − 1 (1 + 1/2n: 7.8e-3 at dim 64, 5.2e-4 at 960) or a wrong epsilon, stays under the rows' lines,
//     and the fused norm's quantized integers do not see it at all, T.175): the least-squares factor s of the GPU's
//     against NumPy's (the packed shaders' against Q8's, whose own scale moves by up to 4.4e-4), sum(g·w) / sum(w·w),
//     which the rounding of single values leaves at 1 within its noise over thousands of values. |s − 1|: 3.8e-9 to
//     3.3e-5 as it should be; RMSNorm over n − 1 5.2e-4 to 1.0e-2, LayerNorm's variance over n − 1 5.4e-4 to 7.4e-3.
//     The line SCALE_LINE: n − 1 goes past it up to a dim of 5000. Only where f32 rounds to f16 to the nearest, as
//     SwiftShader and lavapipe do (the review, 2026-09-27): WGSL (15.7.6) leaves the direction open, and a GPU that
//     rounds toward zero shrinks every value, which does not average out. A throwaway branch that truncates the
//     cache's pack2x16float and the f16 tiles' operands (run 36320147208) read s − 1 = −3.4e-4 to −3.9e-4 on the
//     float32 and packed shaders and −9.5e-4 to −1.1e-3 on the f16 ones, every model, with every other line held: on
//     such a GPU this line fails a GPU as it should be (then hold the scale against NumPy's rounded toward zero too,
//     and the f16 shaders to about 2e-3).
//   - the logits of the prompt's last token (on the CPU in every run, on the GPU's keys and values in the GPU's): the
//     most likely token NumPy's, or one NumPy gives at least half the probability of its most likely (a near tie:
//     T.147's stories15M at 149 tokens). Their KL divergence from NumPy's is printed and held to no line (the review,
//     2026-09-27): the CPU computes that last token with its 7-bit activations, so the KL is the CPU's arithmetic,
//     not the GPU's. The CPU alone is 9.1e-5 to 9.8e-2 on the made-up models (synthetic-6bit the most), and on keys and
//     values that are 1.0 E16 (TF.js) synthetic-wide read 0.22 to 0.27 on every job of run 36320963317, where a line
//     of 0.05 failed it. A wrong bias on Qwen2.5 0.5B reads 9.3 to 11.7, and fails the keys and values' lines 1569
//     times over and the most likely token as well. Printed beside it: the largest difference of the logits of
//     NumPy's ten most likely tokens over NumPy's largest |logit| (the old measure took all of them: GPT-2's went past
//     on a token some 50 logits below the top).
// T.562: K_PACKED 3 went past a GPU as it should be on the real models' deep layers: Qwen3 0.6B's DP4A at layer 19,
// 1.39 of its line (T.546's run 38881685551, 4.2 Q8 of the layer), Qwen2.5 0.5B's at layer 14, 1.06 (run 40157762775,
// 3.2 Q8), each layer and run another (the same Qwen3 read 0.96 at layer 19 in run 40153360370): the 8-bit roundings
// that fall on the other side of a boundary, as above, more of them over 24 to 28 layers. A wrong GPU is far past it
// (the layer 0's biases in every layer: 2222 times K_PACKED 3's line on Qwen2.5 0.5B, run 40157832633; 7.02 Q8 on the made-up
// Qwen2, the weakest one, 2.34 times K_PACKED 3's line). K_PACKED 5: those correct runs 0.83 of it at the most, the
// made-up Qwen2's wrong one 1.40 past it
const K_PACKED = 5, SCALE_LINE = 1e-4, TIE = Math.LN2;
// T.562: the f16 tiles held to K × E16 went past it on Qwen3 0.6B, a GPU as it should be (llama.cpp's f16 tiles with the
// float32 attention, layer 8 at 1.04 of 8 × E16, run 40153360370, where the same tiles with the f16 attention read 0.80):
// they round every weight × scale and activation to f16, which E16 (the cache alone) does not, and which grows with
// the depth and Qwen's large activations as Q8's 8 bits do. So their layers are held, as the packed shaders' are, to
// H16's distance from NumPy's (NumPy's answer with the layers' matrices and their inputs rounded to float16 and its
// cache in float16, reference.ts's H16), K_HALF times it, and to HALF_LINE where that is larger. K_HALF (CI's Dawn on
// lavapipe, runs 40157762788 and 40157762775): a GPU as it should be is 0.79 to 1.42 H16 from NumPy's (the made-up
// synthetic-yarn the most; Qwen3 0.6B 1.00, Qwen2.5 0.5B 0.96 to 1.08, SmolLM2 360M 0.79 to 1.00), at 0.58 of its line
// at the most; one that reads layer 0's biases in every layer is 98 H16 away on the made-up Qwen2 (32.8 times its
// line) and 17477 on Qwen2.5 0.5B (run 40157832633). K_HALF 3, as K_PACKED was
const K_HALF = 3;
// T.562 (the review): the steps' keys and values keep K_PACKED 3 × Q8, for their line is also the run's own prompt's
// distance at the layer (steps.ts), which carries a deep layer's 8-bit roundings; the weak wrong GPUs of T.226 (a layer
// that reads layer 0's FFN norm) were measured against 3
const K_PACKED_STEPS = 3;

export { COUNT, KV_START, GEN, NOT_FINITE_STATUS, NOT_FINITE_KV_STATUS, GPU_LINE, HALF_LINE, K, K_PACKED, K_PACKED_STEPS, K_HALF, SCALE_LINE, TIE };
