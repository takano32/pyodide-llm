// shaders/steps.js (T351): the quantizing and the small steps of a layer: the activations to int8 (QUANTIZE, T241's
// scale word, which fused.js's NORM_QUANTIZE takes too), int6 weights widened (T155), and RMSNorm, LayerNorm, the
// residual add, RoPE, SwiGLU and GELU for the tokens of a prompt.
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).
//
// The notice of what T241's scale_word() takes its test of a float's bits from (isnan() of TensorFlow.js,
// tfjs-backend-webgpu/src/webgpu_program.ts: `(floatToUint & 0x7fffffffu) > 0x7f800000u`; said above SCALE_WORD, and
// with what was changed above sample.js's SAMPLER_COMMON, where this notice stands too; copied here, as this file
// holds lines of that form). Copyright 2022 Google LLC. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
const { GROUP, STEP } = await import(new URL(`common.js${new URL(import.meta.url).search}`, import.meta.url));

// The activations of the packed shaders, as the CPU's quantize_x makes them (kernels/kernel.ts): per token and group
// of 32, the scale is the largest |value| / 127 and a value round(value × (1 / scale)) (half to even), clamped to
// ±127, four to a u32 with the first in the lowest byte. ORT's dp4a_quantize is not taken: its pack4x8snorm rounds
// as ⌊0.5 + 127 × value⌋ (half up, not the CPU's half to even), and its groups are 128. One thread a group; the tokens are the dispatch's y. x holds
// the tokens xStride floats apart, xq the same bytes apart and xs the scales xStride / 32 floats apart.
//
// ---- T241: a group with a value that is no finite number (a NaN, an infinity) gets a NaN for its scale, so that the
// 8 bits do not hide it. WGSL lets an implementation take NaN and infinities as absent (§15.7): on lavapipe max(NaN, v)
// is v whatever the order, i32(NaN) is the least integer and the clamp makes it -127, so the float max these quantizers
// had turned a NaN of the keys and values into finite numbers, the logits stayed finite, T219's flag of the sampler saw
// nothing, and the DP4A forms returned ids where the float forms refused the step (the review of T219, run
// 36876165803). The CPU's quantize_x keeps a NaN (f32x4.max), and stops by T195's rule.
// No public implementation was found to take this part from (the forms of these quantizers are ORT's dp4a_quantize's
// and vLLM's per-token scales', which take a largest of floats; their sources were not read again for a NaN's fate),
// so it is written apart, with what it is given:
//   magnitude: u32   the largest of bitcast<u32>(value) & FLOAT_MAGNITUDE over the group's 32 values: their bits
//                    without the sign, taken with the integer max
//   scale: f32       bitcast<f32>(magnitude) / 127.0, the group's scale where every value is finite
//   scale_word(magnitude, scale) -> u32   the word the quantizer stores for the group in xs (declared array<u32> in
//                    the quantizers; the matrix reads the same buffer as array<f32>): bitcast<u32>(scale) where
//                    magnitude < 0x7f800000 (every value finite), else magnitude | 0x00400000, a NaN (the exponent
//                    all ones, a bit of the fraction set: 0x7fc00000 for an infinity, the NaN's own bits with that
//                    bit for a NaN)
// Why so. (1) The bits of floats that are not negative are in the order of their values, so the integer max of the
// magnitudes is the bits of the float max of the |values| for every finite input (-0 is 0, a denormal its own bits):
// the scale and the values of a finite group are what they were, to the bit. An infinity's magnitude (0x7f800000) is
// over every finite one's and a NaN's over that (isnan's form of TensorFlow.js, T219's: its notice is at the head of this file and above sample.js's
// SAMPLER_COMMON), and an integer max drops neither. (2) What carries it on is the scale itself, no flag: the matrix
// (the tiles' and fusedDp4aMatVec's SDP8AI) multiplies each group's integer dot by scale_a × scale_b and adds the
// groups of a row, so every row of its output is a NaN, whatever the integers of that group are; the residual stream
// is a NaN from there, each quantizer after it finds it again by its bits (NORM_QUANTIZE's values are weight × (s × x),
// a NaN where x or the sum of x² is one), and the classifier's logits are NaN, which T219's flag refuses by their
// bits. A multiplication and an addition by a NaN are the hardware's own. (T241's review: there are floats' max,
// comparisons and a clamp on the way, where a device may drop a NaN: the attention's softmax (the max of the scores,
// exp_sum != 0), GELU's clamp, the quantizers' own scale > 0. None is the one thing that carries the NaN. A score that is
// a NaN makes p = exp(score - m) a NaN whatever max did with it, and then p × V, the sum and the output; GELU and SwiGLU
// multiply the NaN input itself; the quantizers' comparison only picks the integers of a group whose scale is a NaN
// already. The tests on lavapipe and SwiftShader, where max drops a NaN, hold that for those two; a device where exp(NaN)
// or a half's NaN is another number is not tried.) A flag word would need a binding in every quantizer, a place in the
// State and a reader in gpu.js and forward.js for what the logits already say. (3) An infinity becomes a NaN too: a scale of
// infinity makes a row +inf or -inf by the sign of its dot (and NaN where the dot is 0), and the way from there to the
// logits then rests on the next norm's 0 × inf (a NaN) and on T195's rule for logits that are all -inf (refused: no token
// over -3.4e38; some -inf among finite ones are tokens that cannot be drawn, which no scale of one infinity makes).
// lavapipe and SwiftShader do both, so the review's mutants that keep an infinity as it is ("no-or", "no-select") pass
// every layer's round there: the scale word is tested alone (tests/gpu-check.mjs's quantizerProbe), and this word makes
// the way short and independent of what a device does with an infinity. (4) The word is chosen by a comparison of integers and stored as an
// integer: no float of the device's is asked whether it is a NaN. In a prompt's block (the same QUANTIZE) the keys
// and values written back are NaN then. The GPU's next step finds them (by the above), and forward.js looks at the
// halves it reads back by their bits before they go into the CPU's cache and refuses the block (T243: the CPU's own
// reading of a float16 NaN, kernels/kernel.ts's halves4, is a finite number, so T195's rule alone would not see it).
// The cost: a value's AND and integer max where its abs and float max were (the abs is a source modifier on most
// devices, so one instruction a value more: 32 a group), and a comparison, an OR and a select a group of 32.
const SCALE_WORD = /* wgsl */ `
const FLOAT_MAGNITUDE = 0x7fffffffu;
fn scale_word(magnitude: u32, scale: f32) -> u32 {
  return select(bitcast<u32>(scale), magnitude | 0x00400000u, magnitude >= 0x7f800000u);
}`;
export const QUANTIZE = /* wgsl */ `
struct Quantize { n: u32, xStride: u32, unused0: u32, unused1: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xs: array<u32>;       // a scale's bits (scale_word, T241)
@group(0) @binding(3) var<uniform> quantize: Quantize;
@group(0) @binding(4) var<uniform> step: Step;
fn packed(v: vec4<i32>) -> u32 {
  let b = bitcast<vec4<u32>>(v) & vec4<u32>(0xffu);
  return b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u);
}
${SCALE_WORD}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let token = id.y;
  if (g >= quantize.n / ${GROUP}u || token >= step.tokens) { return; }
  let at = (token * quantize.xStride) / 4u + g * 8u;
  var magnitude = 0u;
  for (var k = 0u; k < 8u; k++) {
    let v = bitcast<vec4<u32>>(x[at + k]) & vec4<u32>(FLOAT_MAGNITUDE);
    magnitude = max(magnitude, max(max(v.x, v.y), max(v.z, v.w)));
  }
  let scale = bitcast<f32>(magnitude) / 127.0;
  let inverse = select(0.0, 1.0 / scale, scale > 0.0);
  for (var k = 0u; k < 8u; k++) {
    xq[at + k] = packed(clamp(vec4<i32>(round(x[at + k] * inverse)), vec4<i32>(-127), vec4<i32>(127)));
  }
  xs[token * (quantize.xStride / ${GROUP}u) + g] = scale_word(magnitude, scale);
}`;

// ---- T155: int6 weights (T98) on the GPU. The model's GPU worker widens every int6 matrix once, as it puts it on the
// GPU, into the int8 the tiled shaders read (and keeps no int6 there): an int6 value is an int8 with its two low bits
// zero and its scale is a quarter (llama2_numpy.pack6), so the widened values and the checkpoint's own scales are an
// int8 matrix, the same products to the bit, and every tiled shader, check and timing stays as it is.
//
// WIDEN_SIX: the WGSL of that widening, one dispatch a piece of a matrix (gpu.js's upload). No public implementation
// has this packing (llama.cpp's Q6_K packs otherwise), so it is written apart (T155: Fable high). What gpu.js gives it:
//   @group(0) @binding(0) var<storage, read> packed: array<u32>;         the groups as llama2_numpy.pack6 writes them,
//                                                                        24 bytes (6 words) a group, one after another,
//                                                                        from word 0 (the piece alone, uploaded apart)
//   @group(0) @binding(1) var<storage, read_write> values: array<u32>;   the same groups widened, 32 int8 (8 words) a
//                                                                        group, value j of a group at its byte j (four
//                                                                        to a word, the lowest byte first), from word 0
//   @group(0) @binding(2) var<uniform> widen: vec4<u32>;                 x: the groups; y, z, w: 0
// dispatched as @workgroup_size(WIDEN_SIX_WORKGROUP) with sixDispatch(groups, the device's
// maxComputeWorkgroupsPerDimension): a thread a group, group number (workgroup_id.y × num_workgroups.x +
// workgroup_id.x) × WIDEN_SIX_WORKGROUP + local_invocation_index, those past widen.x doing nothing. gpu.js checks it
// against sixValues() below as it starts (every group of random bytes: any 24 bytes are a group), to the bit.
//
// The form (T155, Fable): four values a word, as the bytes lie. A group's six words are its 24 bytes: low[0..15] in
// words 0..3 (byte j of the group at byte j % 4 of word j / 4) and top[0..7] in words 4 and 5. An int6 value's int8 is
// its six bits shifted up twice, so it is a byte with nothing to sign-extend: bits 2-5 from a nibble of low[j] (value
// j its low nibble, value j + 16 its high one) and bits 6-7 from a pair of bits of top[j % 8] (values k, k + 8,
// k + 16, k + 24 at bits 0, 2, 4, 6). Because value j sits in the same byte of its word as low[j] and top[j % 8] do
// (word i of the output takes low from word i and top from word 4 + i % 2), each output word is two masked shifts
// of two input words: the nibbles of four values to bits 2-5 at once (0x0f0f0f0f, or the high nibbles as word >> 4),
// and their pairs to bits 6-7 at once (top << (6 - pair's bit), masked to 0xc0c0c0c0: what a shift carries across a
// byte lands under the mask). Masks and shifts by constants only, as kernels/six.ts. The packing is this project's,
// but the form is llama.cpp's for Q6_K, whose six bits lie as nibbles and pairs of bits too: four values a 32-bit
// word, masks by the byte (0x0F0F0F0F, 0x30303030, 0xC0C0C0C0) and shifts (https://github.com/ggml-org/llama.cpp,
// commit 95887577, MIT: ggml-cuda/vecdotq.cuh's vec_dot_q6_K_q8_1_impl_mmvq and ggml-metal/kernels/dequantize.h's
// dequantize_q6_K; no line copied: Q6_K's values are unsigned with 32 taken off, where pack6's are their int8's). A
// thread a group: it reads 6 words and writes 8, the 64 threads of a workgroup reading 1536 bytes in a row and writing
// 2048, every line of the cache used whole; the traffic is the 56 bytes a group either way, and the widening runs once
// as a model is put on the GPU (Llama 3.2 1B's layers: 0.73 GB read and 0.97 GB written by it; their scales, 0.12 GB,
// go up as an int8 model's do). Splitting a group over two threads would read the top words twice for 4 fewer stores
// a thread: not taken.
export const WIDEN_SIX_WORKGROUP = 64;
export const WIDEN_SIX = /* wgsl */ `
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> values: array<u32>;
@group(0) @binding(2) var<uniform> widen: vec4<u32>;
// four int8 from the low nibbles of low's bytes and the pair of bits at bit (bits 0, 2, 4 or 6) of top's bytes
fn widened(low: u32, top: u32, bit: u32) -> u32 {
  return ((low & 0x0f0f0f0fu) << 2u) | ((top << (6u - bit)) & 0xc0c0c0c0u);
}
@compute @workgroup_size(${WIDEN_SIX_WORKGROUP})
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let g = (id.y * count.x + id.x) * ${WIDEN_SIX_WORKGROUP}u + t;
  if (g >= widen.x) { return; }
  let at = g * 6u;
  let low = vec4<u32>(packed[at], packed[at + 1u], packed[at + 2u], packed[at + 3u]);
  let top = vec2<u32>(packed[at + 4u], packed[at + 5u]);
  let out = g * 8u;
  values[out] = widened(low.x, top.x, 0u);            // values 0..3: low nibbles, top bits 0-1 of top[0..3]
  values[out + 1u] = widened(low.y, top.y, 0u);       // 4..7: top[4..7]
  values[out + 2u] = widened(low.z, top.x, 2u);       // 8..11: bits 2-3 of top[0..3]
  values[out + 3u] = widened(low.w, top.y, 2u);       // 12..15
  values[out + 4u] = widened(low.x >> 4u, top.x, 4u); // 16..19: high nibbles, bits 4-5
  values[out + 5u] = widened(low.y >> 4u, top.y, 4u); // 20..23
  values[out + 6u] = widened(low.z >> 4u, top.x, 6u); // 24..27: bits 6-7
  values[out + 7u] = widened(low.w >> 4u, top.y, 6u); // 28..31
}`;
/** the workgroups [x, y] of WIDEN_SIX for groups: a second dimension past the device's most of one */
export const sixDispatch = (groups, most) => {
  const workgroups = Math.ceil(groups / WIDEN_SIX_WORKGROUP), x = Math.min(workgroups, most);
  return [Math.max(1, x), Math.ceil(workgroups / Math.max(1, x))];
};
/** JavaScript's widening (the check's answer): the int8 values of groups of 24 packed bytes (a Uint8Array), as
 * llama2_numpy.unpack6 and public/forward/engine.js's weightAt read them */
export function sixValues(packed) {
  const groups = packed.length / 24, out = new Int8Array(groups * 32);
  for (let g = 0; g < groups; g++) {
    for (let j = 0; j < 32; j++) {
      const low = j < 16 ? packed[g * 24 + j] & 15 : packed[g * 24 + j - 16] >> 4;
      const top = (packed[g * 24 + 16 + (j % 8)] >> (2 * ((j / 8) | 0))) & 3;
      out[g * 32 + j] = ((low | (top << 4)) << 2) << 24 >> 24;
    }
  }
  return out;
}

// ---- the steps of a layer besides its matrices, for the tokens of a prompt (the model's GPU worker). Each does for
// every token what the CPU's kernel of the same name (kernels/kernel.ts) does for one, in float32.

// RMSNorm: one workgroup per row, out = weight * (x / sqrt(mean(x²) + eps)), weight this layer's from float at. The
// rows of a token are the dispatch's x (one for the layer's norms, a row of dim; T153: Qwen3's norms of q and k, a row
// a head of headSize, heads of them), the tokens its y. inPlace (T153): x is written over (a head's q or k), as
// llama.cpp's rms_norm_mul.wgsl has it (INPLACE: the norm and the weight's product in one dispatch, the weight's row
// the same for every row, mul_src_ne1 1); else out is another buffer (xb, the layer's norms). T226: norm.first, the rows
// before the dispatch's first (0 for a prompt's): a generated token's q, k and v are one buffer as their matrix writes
// them, and the heads of k are the rows after q's heads.
//
// The row of a norm, its weight broadcast over the rows and the in-place form adapted from llama.cpp,
// ggml/src/ggml-webgpu/wgsl-shaders/rms_norm_mul.wgsl and binary.wgsl (OP_ADD, INPLACE; ADD below), and (T154)
// row_norm.wgsl (NORM: LAYER_NORM below) and unary.wgsl (GELU: GELU below), and the fusion of LAYER_NORM's last line
// from ggml/src/ggml-metal/kernels/norm.metal (kernel_norm_mul_add_f32, F == 3; ggml-metal-fusion.cpp's NORM_MUL_ADD)
// (https://github.com/ggml-org/llama.cpp, commit 2145525a, 2026-09-26), under the MIT License:
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
const rmsNorm = (inPlace) => /* wgsl */ `
struct Norm { size: u32, at: u32, eps: f32, first: u32 }
${STEP}
@group(0) @binding(0) var<storage, ${inPlace ? "read_write" : "read"}> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
${inPlace ? "" : "@group(0) @binding(2) var<storage, read_write> out: array<f32>;\n"}@group(0) @binding(${inPlace ? 2 : 3}) var<uniform> norm: Norm;
@group(0) @binding(${inPlace ? 3 : 4}) var<uniform> step: Step;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) rows: vec3u, @builtin(local_invocation_index) t: u32) {
  if (id.y >= step.tokens) { return; }
  let row = (id.y * rows.x + id.x + norm.first) * norm.size;
  var squares = 0.0;
  for (var i = t; i < norm.size; i += 64u) { squares += x[row + i] * x[row + i]; }
  partial[t] = squares;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let s = 1.0 / sqrt(partial[0] / f32(norm.size) + norm.eps);
  for (var i = t; i < norm.size; i += 64u) { ${inPlace ? "x" : "out"}[row + i] = weight[norm.at + i] * (s * x[row + i]); }
}`;
export const RMSNORM = rmsNorm(false);
export const HEAD_NORM = rmsNorm(true);

// T154: LayerNorm (GPT-2's and GPT-NeoX's), out = weight * ((x - mean) / sqrt(variance + eps)) + bias, as the CPU's
// layernorm kernel computes it: llama.cpp's row_norm.wgsl with NORM (the notice above: the sum of the row, its mean,
// then the sum of the squares about the mean, a reduction of the workgroup each), in the frame of RMSNORM above: one
// workgroup a row, the tokens the dispatch's y. The weight's MUL and the bias's ADD, which llama.cpp's WebGPU runs as
// two dispatches of binary.wgsl after the NORM, are the last line of this one, as llama.cpp's Metal fuses NORM, MUL and
// ADD (kernel_norm_mul_add_f32: (y * scale) * f0 + f1, y = x - mean, the rounding in the same order as here and as the
// CPU's kernel); both are this layer's, from float at.
export const LAYER_NORM = /* wgsl */ `
struct Norm { size: u32, at: u32, eps: f32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> norm: Norm;
@group(0) @binding(5) var<uniform> step: Step;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) rows: vec3u, @builtin(local_invocation_index) t: u32) {
  if (id.y >= step.tokens) { return; }
  let row = (id.y * rows.x + id.x) * norm.size;
  var sum = 0.0;
  for (var i = t; i < norm.size; i += 64u) { sum += x[row + i]; }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let mean = partial[0] / f32(norm.size);
  var squares = 0.0;
  for (var i = t; i < norm.size; i += 64u) {
    let d = x[row + i] - mean;
    squares += d * d;
  }
  workgroupBarrier();
  partial[t] = squares;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let s = 1.0 / sqrt(partial[0] / f32(norm.size) + norm.eps);
  for (var i = t; i < norm.size; i += 64u) { out[row + i] = weight[norm.at + i] * (s * (x[row + i] - mean)) + bias[norm.at + i]; }
}`;

// T153: a bias added to every token of a matrix's output (Qwen2's q, k and v), llama.cpp's binary.wgsl with OP_ADD and
// INPLACE (the notice above): y += bias, the bias (this layer's, from float at) the same for every token (b_ne1 1).
// Changed: the token is the dispatch's y and the element its x (as SWIGLU here), where llama.cpp numbers every element
// of the tensor along x and y and finds its place in either by strides. T226: byPos, the floats a position of the
// bias takes (0: the same vector for every token, as all the biases are): GPT-2's learned positions added to a
// generated token's row of the embedding, the table's row of the token's position (llama.cpp adds the rows of
// position_embd to the embedding with the same ADD, after a get_rows of them: here the row is found by the Step)
export const ADD = /* wgsl */ `
struct Bias { n: u32, at: u32, byPos: u32, unused1: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> shape: Bias;
@group(0) @binding(3) var<uniform> step: Step;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.y >= step.tokens || id.x >= shape.n) { return; }
  y[id.y * shape.n + id.x] += bias[shape.at + (step.pos + id.y) * shape.byPos + id.x];
}`;

// RoPE on q and k, and the keys and values of every token into this layer's cache at its position (step.pos + the
// token): one workgroup per token. Pairs of neighbours turn (llama2.c's order), the first turned of every head (all
// of it but for GPT-NeoX, T154: whose converter puts the pairs of its rotary part in this order; none of it for GPT-2,
// whose positions are learned and added to the rows by the CPU: its keys go into the cache as they are); angles holds, per token, the cos of its headSize / 2 angles and then their sin. The cache
// holds float16 (T147: as the CPU's cache does, T110, and as llama.cpp's flash attention reads its K and V), a pair of
// neighbours to a u32 (pack2x16float: no shader-f16 needed), the pair RoPE turns together.
export const ROPE = /* wgsl */ `
struct Rope { heads: u32, kvHeads: u32, headSize: u32, turned: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> values: array<u32>;
@group(0) @binding(5) var<storage, read> angles: array<f32>;
@group(0) @binding(6) var<uniform> rope: Rope;
@group(0) @binding(7) var<uniform> step: Step;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let token = id.x;
  if (token >= step.tokens) { return; }
  let size = rope.headSize;
  let half = size / 2u;
  let pairs = rope.turned / 2u;
  let angle = token * size;
  // q in place, a pair per thread
  for (var p = t; p < rope.heads * pairs; p += 64u) {
    let i = p % pairs;
    let at = token * rope.heads * size + (p / pairs) * size + 2u * i;
    let c = angles[angle + i];
    let s = angles[angle + half + i];
    let v0 = q[at];
    let v1 = q[at + 1u];
    q[at] = v0 * c - v1 * s;
    q[at + 1u] = v0 * s + v1 * c;
  }
  // k turned on its way into the cache, v as it is: a pair of neighbours per thread
  let kvDim = rope.kvHeads * size;
  let row = (step.pos + token) * kvDim / 2u;
  for (var j = t; j < kvDim / 2u; j += 64u) {
    let at = token * kvDim + 2u * j;
    var key = vec2<f32>(k[at], k[at + 1u]);
    let inHead = (2u * j) % size;
    if (inHead < rope.turned) {
      let c = angles[angle + inHead / 2u];
      let s = angles[angle + half + inHead / 2u];
      key = vec2<f32>(key.x * c - key.y * s, key.x * s + key.y * c);
    }
    keys[row + j] = pack2x16float(key);
    values[row + j] = pack2x16float(vec2<f32>(v[at], v[at + 1u]));
  }
}`;

// SwiGLU: gate = silu(gate) * up, for every token (the second dimension of the dispatch)
export const SWIGLU = /* wgsl */ `
struct Size { n: u32, unused0: u32, unused1: u32, unused2: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> size: Size;
@group(0) @binding(3) var<uniform> step: Step;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let i = id.x * 64u + t;
  if (id.y >= step.tokens || i >= size.n) { return; }
  let at = id.y * size.n + i;
  let v = gate[at];
  gate[at] = v / (1.0 + exp(-v)) * up[at];
}`;

// T154: GELU (GPT-2's gelu_new and GPT-NeoX's, the tanh approximation), gate = gelu(gate) for every token (the dispatch's
// y), in place: llama.cpp's unary.wgsl with GELU (the notice above: its formula, the argument of tanh clamped as it
// has it). Changed: the element is the dispatch's x and the token its y (as SWIGLU); the bias of the projection before
// it is ADD's, as llama.cpp adds it (the CPU's gelu kernel adds it itself: the same sum, v + b, rounded once)
export const GELU = /* wgsl */ `
struct Size { n: u32, unused0: u32, unused1: u32, unused2: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<uniform> size: Size;
@group(0) @binding(2) var<uniform> step: Step;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let i = id.x * 64u + t;
  if (id.y >= step.tokens || i >= size.n) { return; }
  let at = id.y * size.n + i;
  let v = gate[at];
  gate[at] = 0.5 * v * (1.0 + tanh(clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -9.010913, 9.010913)));
}`;

export { SCALE_WORD };
