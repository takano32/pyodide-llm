// shaders/fused.js (T351): a layer of a generated token in five dispatches (T150): llama.cpp's matrix × vector with
// the norm on its read and RoPE, the residual add or SwiGLU on its write, the same on ONNX Runtime's DP4A (T175, with
// NORM_QUANTIZE) and on ternary weights (T232).
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).
const { GROUP, STEP } = await import(new URL(`common.js${new URL(import.meta.url).search}`, import.meta.url));
const { TERNARY_PACKED, sdp8ai } = await import(new URL(`prompt.js${new URL(import.meta.url).search}`, import.meta.url));
const { SCALE_WORD } = await import(new URL(`steps.js${new URL(import.meta.url).search}`, import.meta.url));
const { MUL_MAT_VEC_ROWS, ORT_DP4A_MATVEC_ROWS } = await import(new URL(`matvec.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T150: a layer of a generated token in five dispatches where the separate steps take fourteen (RMSNorm; q, k and
// v; RoPE and the cache; the attention; o; the residual add; RMSNorm; gate and up; SwiGLU; down; the residual add):
//   1. q, k and v as one matrix (their rows one after the other in one buffer), the norm on its read of the input, and
//      RoPE and the cache on its write: q turned into q, k turned and v as they are into the cache at step.pos
//   2. the attention (flashTile, as the prompt's)
//   3. o, its product added to the residual stream on its write
//   4. gate and up in one workgroup (gate's rows, then up's in the same buffer), the norm on the read, SwiGLU on the
//      write
//   5. down, added to the residual stream on its write
// For /benchmark/'s GPU section first (T150), and for the model's GPU worker when it generates on the GPU (T152).
// The matrix × vector is llama.cpp's mul_mat_vec in its widened form (T149's mulMatVec), and each fold takes its form
// from a public implementation:
//   - gate and up in one kernel with the GLU on the write: llama.cpp's CUDA mul_mat_vec_q with its fusion (mmvq.cu:
//     vgate beside vx, tmp_gate beside tmp, reduced alike, and result *= silu(gate) where a row is written; commit
//     2145525a). Here the two matrices are one buffer, up's rows `second` rows after gate's.
//   - the residual added on the write: the same kernel's x_bias (result += x_biases[j], an ADD after the MUL_MAT fused
//     into it), and MLC LLM's fused matmul and add (FuseDequantizeMatmulEwise, Apache-2.0)
//   - q, k and v (and gate and up) as one matrix: MLC LLM's Llama (qkv_proj and gate_up_proj, Apache-2.0)
//   - RoPE and the cache on the write: llama.cpp's CUDA fuses rope and set_rows (ggml-cuda.cu, rope_set_rows_ops) into
//     one kernel after the matrix; here into the matrix's own write, since a workgroup's 4 rows hold whole pairs (the
//     neighbours RoPE turns together, and the pairs of float16 the cache holds a u32). The turning is ROPE's (steps.js).
//   - RMSNorm on the read: the norm's scale 1 / sqrt(mean(x²) + eps) is one number for the whole row, so it comes out of
//     the sum: W·(g ⊙ x·s) = s × W·(g ⊙ x). Every workgroup reads all of x once anyway, and adds up x² beside the rows'
//     sums in the same reduction (FlashNorm, Graef et al. 2024, arXiv 2407.09577: the scale deferred past the matrix).
//     No public WGSL does this (llama.cpp's WebGPU fuses the norm with its weight only, rms_norm_mul.wgsl): the lines
//     are this project's, written as llama.cpp's rms_norm_mul computes it (eps inside the sqrt with the mean, as the
//     engine's rmsnorm and steps.js's RMSNORM). Checked (T150, Fable): the deferred scale rounds once at the end where the
//     separate form rounds every x·s, so both are within a few float32 ulp of float64 (4e-7 of the largest row at
//     worst, with channels at 1000× the rest too: the sum of x² adds the small terms among themselves before the tree
//     meets a large one). What it costs a workgroup: reading g (4·dim bytes from the cache, beside x) and 2·dim more
//     ALU next to the rows' 4·dim·1.125 bytes from DRAM; what it saves a layer: two dispatches and a vector written and
//     read. Which is cheaper is the device's (the layer table's fused row against the separate one).
//     A norm of a head (Qwen3's q and k, T153) cannot go on the write: a head's rows span headSize / 4 workgroups and
//     WGSL has no sum across them; the read side stays as it is, the write side gets a small dispatch after the matrix.
// Changed from mul_mat_vec besides: the sums are SUMS (the rows', gate's rows' too, and x²'s), reduced as llama.cpp
// reduces its rows' (the workgroup's tree, or subgroupAdd), into totals that the write (the epilogue) reads; where a
// row is written differs by the output. The weights' layout, the scales and Params' first four are T149's.
//
// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl and mul_mat_vec_acc.tmpl (MUL_ACC_Q8_0),
// and ggml/src/ggml-cuda/mmvq.cu (the fusion of gate, bias and GLU) (https://github.com/ggml-org/llama.cpp, commit
// 2145525a, 2026-09-26), under the MIT License:
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
//
// input: "norm" (x is the residual stream, normed on the read with norm_weight from params.normAt) or "plain".
// output: "rope" (rows of q, then kvRows of k, then of v: q into q, k and v into the cache at step.pos), "add" (added
// to dst: the residual stream), "swiglu" (silu(gate) × up into dst, up's rows params.second after gate's) or "write"
// (into dst: the classifier's logits, T151).
// Bindings: 0 the weights, 1 their scales, 2 x, 3 params; 4 norm_weight (norm); 5 dst (add, swiglu, write), or 5 q,
// 6 the keys, 7 the values, 8 the angles and 9 the step (rope). T151 (T150's review, (c)): the position is the Step's
// (a uniform of its own, which a run of tokens on the GPU copies from the sampler's state after each token), and the
// angles are the whole table of RoPE on the GPU, a row a position: the cos of its headSize / 2 angles, then their sin
// (the rows of the CPU's table: the model's own, Llama 3's scaling too).
const FUSED_PARAMS = /* wgsl */ `struct Params { rows: u32, words: u32, perRow: u32, second: u32, eps: f32, normAt: u32, qRows: u32, kvRows: u32,
                headSize: u32, turned: u32, unused0: u32, unused1: u32 }`;
// where the write of an output goes (bindings 5 on): T150's, and T175's DP4A form's alike. T226: the write of a pair
// is a function (write_pair), which TOKEN_ROPE below calls too: the same lines where the matrix's write cannot turn
// the pair yet (a bias or a head's norm comes between)
const fusedOutputs = (output) => (output === "rope" ? `@group(0) @binding(5) var<storage, read_write> q: array<f32>;
@group(0) @binding(6) var<storage, read_write> keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> values: array<u32>;
@group(0) @binding(8) var<storage, read> angles: array<f32>;
@group(0) @binding(9) var<uniform> step: Step;

// a pair of neighbouring rows (row even, v0 and v1 their values): q's turned into q; k's turned and v's as they are
// into the cache at pos
fn write_pair(row: u32, v0: f32, v1: f32) {
    let size = params.headSize;
    let half = size / 2u;
    let at = step.pos * size;  // the position's row of the table: its cos, then its sin
    if (row < params.qRows) {
        let in_head = row % size;
        if (in_head < params.turned) {
            let c = angles[at + in_head / 2u];
            let s = angles[at + half + in_head / 2u];
            q[row] = v0 * c - v1 * s;
            q[row + 1u] = v0 * s + v1 * c;
        } else {
            q[row] = v0;
            q[row + 1u] = v1;
        }
    } else if (row < params.qRows + params.kvRows) {
        let j = row - params.qRows;
        var key = vec2<f32>(v0, v1);
        let in_head = j % size;
        if (in_head < params.turned) {
            let c = angles[at + in_head / 2u];
            let s = angles[at + half + in_head / 2u];
            key = vec2<f32>(key.x * c - key.y * s, key.x * s + key.y * c);
        }
        keys[step.pos * params.kvRows / 2u + j / 2u] = pack2x16float(key);
    } else {
        let j = row - params.qRows - params.kvRows;
        values[step.pos * params.kvRows / 2u + j / 2u] = pack2x16float(vec2<f32>(v0, v1));
    }
}` : "@group(0) @binding(5) var<storage, read_write> dst: array<f32>;");
// The write (the epilogue), T150's and T175's DP4A form's alike: a workgroup's OUTPUTS_PER_WG rows from row_base on,
// their sums in totals (up's rows' after them, for SwiGLU), each times scale (the norm's, or 1), by thread thread_id
const fusedWrite = (output) => (output === "rope" ? `    // a pair of neighbouring rows a thread
    if (thread_id < OUTPUTS_PER_WG / 2u) {
        let row = row_base + 2u * thread_id;
        if (row < params.rows) {
            write_pair(row, totals[2u * thread_id] * scale, totals[2u * thread_id + 1u] * scale);
        }
    }` : `    if (thread_id < OUTPUTS_PER_WG) {
        let row = row_base + thread_id;
        if (row < params.rows) {
            let value = totals[thread_id] * scale;${output === "swiglu" ? `
            // llama.cpp's result *= silu(gate_value), as the GLU writes it (glu.wgsl's OP_SWIGLU)
            let gate = value;
            let up = totals[OUTPUTS_PER_WG + thread_id] * scale;
            dst[row] = gate / (1.0 + exp(-gate)) * up;` : output === "write" ? `
            dst[row] = value;` : `
            dst[row] = dst[row] + value;`}
        }
    }`);
// T226: RoPE and the cache of a generated token whose q, k and v cannot be turned on their matrix's write: Qwen2's
// biases (ADD) and Qwen3's norms of the heads (HEAD_NORM) come between, as on the CPU and in a prompt's block (T153:
// the biases, the norms of the heads, RoPE). The matrix then writes q, k and v as they are into one buffer (output
// "write": the rows of q, then k's, then v's), those dispatches change them in place, and this one does with every
// pair what the matrix's write would have done (write_pair above: ROPE's turning, the keys and values as pairs of
// float16 into the cache at step.pos). One workgroup, a thread a pair. Bindings: 2 the rows, 3 the matrix's Params
// (rows, qRows, kvRows, headSize, turned), 5 to 9 as the fused "rope"
export const TOKEN_ROPE = /* wgsl */ `
${FUSED_PARAMS}
${STEP}
@group(0) @binding(2) var<storage, read> src1: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
${fusedOutputs("rope")}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32) {
    for (var row = 2u * t; row < params.rows; row += 128u) {
        write_pair(row, src1[row], src1[row + 1u]);
    }
}`;
export const fusedMatVec =({ input, output, subgroups }) => {
  const norm = input === "norm", glu = output === "swiglu";
  const matrices = glu ? 2 : 1, sums = MUL_MAT_VEC_ROWS * matrices + (norm ? 1 : 0);
  return /* wgsl */ `${subgroups ? "enable subgroups;\nrequires subgroup_id;\n" : ""}
${FUSED_PARAMS}
${STEP}
@group(0) @binding(0) var<storage, read> src0: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> src1: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
${norm ? "@group(0) @binding(4) var<storage, read> norm_weight: array<f32>;" : ""}
${fusedOutputs(output)}

const WG_SIZE = 256u;
const OUTPUTS_PER_WG = ${MUL_MAT_VEC_ROWS}u;
const BLOCK_SIZE = 32u;
const THREADS_PER_BLOCK = 4u;
const ELEMS_PER_THREAD = BLOCK_SIZE / THREADS_PER_BLOCK;
const MATRICES = ${matrices}u;
const SUMS = ${sums}u;
const SQUARES = OUTPUTS_PER_WG * MATRICES;  // where the sum of x² is, when the norm is on the read

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

// a row's sum at the thread's part of a block: llama.cpp's inner loop, for one row of either matrix
fn block_dot(weight_row: u32, block: u32, thread_within_block: u32, x_block: array<f32, ELEMS_PER_THREAD>) -> f32 {
    let d = scales[weight_row * params.perRow + block];
    let block_word_base = weight_row * params.words + block * (BLOCK_SIZE / 4u);
    var q_packed: array<u32, ELEMS_PER_THREAD / 4u>;
    for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
        q_packed[packed_idx] = src0[block_word_base + thread_within_block * 2u + packed_idx];
    }
    var row_sum = 0.0;
    for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
        for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
            let q_val = f32(get_byte_i32(q_packed[packed_idx], byte_idx)) * d;
            row_sum += q_val * x_block[packed_idx * 4u + byte_idx];
        }
    }
    return row_sum;
}

// the sums as the reduction takes them: the rows' (acc), then up's rows' (acc_up, beside acc as mmvq.cu's tmp_gate beside tmp), then x²
fn accumulate_vec_dot(thread_id: u32, row_base: u32) -> array<f32, SUMS> {
    var acc: array<f32, OUTPUTS_PER_WG>;${glu ? `
    var acc_up: array<f32, OUTPUTS_PER_WG>;` : ""}${norm ? `
    var squares = 0.0;` : ""}

    let num_blocks = params.perRow;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = block * BLOCK_SIZE + thread_within_block * ELEMS_PER_THREAD;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
            x_block[i] = src1[x_base + i];${norm ? `
            // the workgroup reads each value of x once: x² for the norm's scale, and the norm's weight now
            squares += x_block[i] * x_block[i];
            x_block[i] *= norm_weight[params.normAt + x_base + i];` : ""}
        }
        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.rows) {
                acc[row] += block_dot(output_row, block, thread_within_block, x_block);${glu ? `
                // up's row, params.second rows after gate's (mmvq.cu's vgate beside vx)
                acc_up[row] += block_dot(output_row + params.second, block, thread_within_block, x_block);` : ""}
            }
        }
    }

    var sums: array<f32, SUMS>;
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        sums[row] = acc[row];${glu ? `
        sums[OUTPUTS_PER_WG + row] = acc_up[row];` : ""}
    }${norm ? `
    sums[SQUARES] = squares;` : ""}
    return sums;
}

// Flattened as [sum][thread] to keep each sum's reduction contiguous in memory.
var<workgroup> partial_sums: array<f32, SUMS * WG_SIZE>;
// what the reduction leaves: each sum of the workgroup, for the write
var<workgroup> totals: array<f32, SUMS>;

fn partial_index(sum: u32, thread: u32) -> u32 {
    return sum * WG_SIZE + thread;
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>${subgroups ? `,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_size) subgroup_size: u32` : ""}
) {
    let thread_id = local_id.x;

    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let output_groups = (params.rows + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    if (wg_linear >= output_groups) {
        return;
    }

    let row_base = wg_linear * OUTPUTS_PER_WG;

    let acc = accumulate_vec_dot(thread_id, row_base);
${subgroups ? `
    for (var sum = 0u; sum < SUMS; sum++) {
        let subgroup_total = subgroupAdd(acc[sum]);
        if (subgroup_invocation_id == 0u) {
            partial_sums[partial_index(sum, subgroup_id)] = subgroup_total;
        }
    }

    workgroupBarrier();

    for (var sum = subgroup_id; sum < SUMS; sum += num_subgroups) {
        var sum_acc = 0.0f;
        for (var k = subgroup_invocation_id; k < num_subgroups; k += subgroup_size) {
            sum_acc += partial_sums[partial_index(sum, k)];
        }
        let sum_total = subgroupAdd(sum_acc);
        if (subgroup_invocation_id == 0) {
            totals[sum] = sum_total;
        }
    }` : `
    for (var sum = 0u; sum < SUMS; sum++) {
        partial_sums[partial_index(sum, thread_id)] = acc[sum];
    }

    workgroupBarrier();

    var stride = WG_SIZE / 2u;

    while (stride > 0) {
        if (thread_id < stride) {
            for (var sum = 0u; sum < SUMS; sum++) {
                partial_sums[partial_index(sum, thread_id)] += partial_sums[partial_index(sum, thread_id + stride)];
            }
        }

        workgroupBarrier();
        stride = stride / 2;
    }

    if (thread_id < SUMS) {
        totals[thread_id] = partial_sums[partial_index(thread_id, 0)];
    }`}

    workgroupBarrier();

    // the write: the norm's scale on every sum (s × W·(g ⊙ x)), then what the output does with a row
    let scale = ${norm ? "1.0 / sqrt(totals[SQUARES] / f32(params.perRow * BLOCK_SIZE) + params.eps)" : "1.0"};
${fusedWrite(output)}
}`;
};

// ---- T175: T150's layer on ONNX Runtime's DP4A for small M (T149's ortDp4aMatVec), the form that read Llama 3.2 1B's
// w1 at 96.8% of the buffer's reads on the owner's Android where llama.cpp's mul_mat_vec (T150's) read a layer at
// 18.5%. ORT runs it after a dispatch of its own that quantizes the vector (dp4a_matmul_nbits.cc:
// DP4AMatMulQuantizeProgram, then DP4AMatMulNBitsSmallMProgram), and adds the bias on the write. So here: the vector
// quantized first (QUANTIZE, T146's, as the CPU's quantize_x makes it), then the matrix, and on its write what
// T150's fusedMatVec does (fusedWrite: RoPE and the cache, the residual's add, SwiGLU, or the logits). Four
// quantizations a layer, one where a new vector comes in: the normed stream before q, k and v (one for the three),
// the attention's output before o, the normed stream before gate and up (one for both), silu(gate) × up before down;
// the classifier's normed stream a fifth, once a token. The norm cannot go on the matrix's read as in T150: the
// matrix reads the quantized vector, and a group's scale is of the values after the norm, so the norm goes with the
// quantizer (NORM_QUANTIZE), one dispatch where RMSNORM and QUANTIZE are two.
//
// RMSNorm with its output quantized as QUANTIZE quantizes it: one workgroup of 64 a token, the sum of x² as RMSNORM
// adds it, then a group of 32 a thread at a time, its values weight × (s × x) as RMSNORM writes them, the largest
// |value| / 127 its scale and each value rounded (half to even) and clamped to ±127, four to a u32 (QUANTIZE's). The
// form is vLLM's rms_norm_per_block_quant (csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu
// with layernorm_utils.cuh, https://github.com/vllm-project/vllm, commit 24c9772d, Apache-2.0): the row's rms in one
// block, then each group's scale from its normed values, then the values normed and quantized (compute_rms,
// compute_dynamic_per_token_scales, norm_and_quant). No lines are taken from it: these are RMSNORM's and QUANTIZE's
// in one workgroup (vLLM's block reduction, fp8 and residual paths are not here). T241: the group's largest by the
// bits and its scale's word as QUANTIZE's (scale_word, above steps.js's QUANTIZE: a NaN where a value of the group is no finite
// number; here a value is weight × (s × x), so a NaN of x, of the sum of x² or of the weight, and an infinity of x,
// whose s is 0 and 0 × inf a NaN).
export const NORM_QUANTIZE = /* wgsl */ `
struct Norm { size: u32, at: u32, eps: f32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<u32>;       // a scale's bits (scale_word, T241)
@group(0) @binding(4) var<uniform> norm: Norm;
@group(0) @binding(5) var<uniform> step: Step;
var<workgroup> partial: array<f32, 64>;
fn packed(v: vec4<i32>) -> u32 {
  let b = bitcast<vec4<u32>>(v) & vec4<u32>(0xffu);
  return b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u);
}
${SCALE_WORD}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let token = id.x;
  if (token >= step.tokens) { return; }
  let row = token * norm.size;
  var squares = 0.0;
  for (var i = t; i < norm.size; i += 64u) { squares += x[row + i] * x[row + i]; }
  partial[t] = squares;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let s = 1.0 / sqrt(partial[0] / f32(norm.size) + norm.eps);
  for (var g = t; g < norm.size / ${GROUP}u; g += 64u) {
    let at = g * ${GROUP}u;
    var magnitude = 0u;
    for (var i = 0u; i < ${GROUP}u; i++) {
      magnitude = max(magnitude, bitcast<u32>(weight[norm.at + at + i] * (s * x[row + at + i])) & FLOAT_MAGNITUDE);
    }
    let scale = bitcast<f32>(magnitude) / 127.0;
    let inverse = select(0.0, 1.0 / scale, scale > 0.0);
    for (var k = 0u; k < ${GROUP / 4}u; k++) {
      let i = at + 4u * k;
      let v = vec4<f32>(weight[norm.at + i] * (s * x[row + i]), weight[norm.at + i + 1u] * (s * x[row + i + 1u]),
                        weight[norm.at + i + 2u] * (s * x[row + i + 2u]), weight[norm.at + i + 3u] * (s * x[row + i + 3u]));
      xq[(row + i) / 4u] = packed(clamp(vec4<i32>(round(v * inverse)), vec4<i32>(-127), vec4<i32>(127)));
    }
    xs[token * (norm.size / ${GROUP}u) + g] = scale_word(magnitude, scale);
  }
}`;

// The matrix: ORT's DP4A for small M as ortDp4aMatVec has it (128 threads for 4 rows, 32 of them along the width,
// each a group of 32 as two vec4<u32> against the quantized vector in the workgroup's memory, the partial sums of a
// row reduced in inter_results), on T150's Params and bindings (the scales of the vector at 4, where T150's norm
// weight is), and T150's write on its rows. Changed from ortDp4aMatVec besides: for SwiGLU a thread multiplies up's row
// (params.second rows after gate's) beside gate's with the same registers of the vector, into rows of inter_results of
// its own (llama.cpp's CUDA mul_mat_vec_q fuses gate so, T150: vgate beside vx); the reduced sums go to totals, and
// the write reads them there (ORT writes a row where it reduces it, and RoPE needs a pair).
//
// Adapted from ONNX Runtime, onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_small_m.wgsl.template (n_bits
// 8) with the parameters dp4a_matmul_nbits.cc gives it (https://github.com/microsoft/onnxruntime, commit 3756d4dc,
// 2026-09-26), under the MIT License:
//
// Copyright (c) Microsoft Corporation
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
//
// output: as fusedMatVec's ("rope", "add", "swiglu" or "write"). Bindings: 0 the weights, 1 their scales, 2 the
// quantized vector (xq), 3 params, 4 its scales (xs); 5 on as fusedMatVec's.
export const fusedDp4aMatVec = ({ output }) => /* wgsl */ `requires packed_4x8_integer_dot_product;
${FUSED_PARAMS}
${STEP}
@group(0) @binding(0) var<storage, read> b: array<vec4<u32>>;        // the weights, 16 int8 to a vec4<u32>
@group(0) @binding(1) var<storage, read> scales_b: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<vec4<u32>>;        // the quantized vector (xq)
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> scales_a: array<f32>;       // its scales (xs), one a group of 32
${fusedOutputs(output)}
${sdp8ai}

const workgroup_size_x = 128u;
const tile_size = ${ORT_DP4A_MATVEC_ROWS}u;
const tile_size_k_vec = 32u;
const sub_tile_count = workgroup_size_x / tile_size_k_vec;
const MATRICES = ${output === "swiglu" ? 2 : 1}u;
// fusedWrite's name for the rows of a workgroup
const OUTPUTS_PER_WG = tile_size;

const double_tile_size_k_vec = 2 * tile_size_k_vec;

var<workgroup> inter_results: array<array<f32, tile_size_k_vec>, tile_size * MATRICES>;
var<workgroup> tile_A : array<vec4<u32>, double_tile_size_k_vec>;
const scale_a_size_in_tile_a = double_tile_size_k_vec / 2;
var<workgroup> scale_A : array<f32, scale_a_size_in_tile_a>;
// what the reduction leaves: each row's sum (up's rows' after gate's), for the write
var<workgroup> totals: array<f32, tile_size * MATRICES>;

fn loadSHMA(kidx_v: u32, col: u32)
{
    let K16 = params.words / 4u;
    let k_offset = kidx_v + col;
    if (k_offset >= K16) {
    return;
    }

    tile_A[col] = a[k_offset];
    if (col < scale_a_size_in_tile_a && kidx_v / 2u + col < params.perRow)
    {
    // kidx_v - covers 16 values of k in input_a
    scale_A[col] = scales_a[kidx_v / 2u + col];
    }
}

@compute @workgroup_size(workgroup_size_x)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32) {
    let workgroup_idx = wg_id.y * num_wg.x + wg_id.x;
    let num_N_tile = (params.rows + tile_size - 1u) / tile_size;
    if (workgroup_idx >= num_N_tile) {
        return;
    }
    let K32 = params.perRow;
    let b_global_base = workgroup_idx * tile_size;
    // Handle each workgroup threads as a block of [sub_tile_count][tile_size_k_vec]
    let local_col = local_idx % tile_size_k_vec;
    let local_row = local_idx / tile_size_k_vec;

    for (var kidx_v:u32 = 0; kidx_v < K32; kidx_v += tile_size_k_vec)
    {
        // Load Phase: Populate shared memory for the workgroup.
        if (local_idx < double_tile_size_k_vec)
        {
        loadSHMA(kidx_v * 2, local_idx);
        }
        workgroupBarrier();
        var own_a: vec4<u32> = tile_A[local_col * 2];
        var own_a1: vec4<u32> = tile_A[local_col * 2 + 1];
        var own_scale_a = scale_A[local_col];
        let k_offset = kidx_v + local_col;
        // k_offset - covers 32 values of k in input_b
        let block_idx = k_offset;
        // calculate intermediate results into inter_results.
        for (var row_offset = 0u; row_offset < tile_size; row_offset += sub_tile_count) {
            let b_global = b_global_base + row_offset + local_row;
            if (b_global < params.rows && k_offset < K32)
            {
                let b_offset = b_global * K32 + k_offset;
                let own_scale_b = scales_b[b_global * params.perRow + block_idx];
                let own_b = b[b_offset * 2];
                let own_b1 = b[b_offset * 2 + 1];
                inter_results[row_offset + local_row][local_col] += SDP8AI(own_a, own_b, own_a1, own_b1, own_scale_a * own_scale_b);${output === "swiglu" ? `
                // up's row, params.second rows after gate's (mmvq.cu's vgate beside vx), with the same own_a
                let up_global = b_global + params.second;
                let up_offset = up_global * K32 + k_offset;
                let up_scale_b = scales_b[up_global * params.perRow + block_idx];
                inter_results[tile_size + row_offset + local_row][local_col] += SDP8AI(own_a, b[up_offset * 2], own_a1, b[up_offset * 2 + 1], own_scale_a * up_scale_b);` : ""}
            }
        }
        workgroupBarrier();
    }

    if (local_idx < tile_size * MATRICES) {
      // Do reduce sum to get final output.
      var output_value = f32(0);
      for (var b = 0u; b < tile_size_k_vec; b++) {
        output_value += inter_results[local_idx][b];
      }
      totals[local_idx] = output_value;
    }

    workgroupBarrier();

    let thread_id = local_idx;
    let row_base = b_global_base;
    let scale = 1.0;  // the norm is the quantizer's (NORM_QUANTIZE)
${fusedWrite(output)}
}`;

// ---- T232: fusedDp4aMatVec for ternary weights (see prompt.js's TERNARY_PACKED, above dp4a: the layout, the unpacking and why).
// The same Params, bindings, dispatch and write; the weights at binding 0 are the codes, 16 to a u32, and a row's scales
// one a group of 128 weights (params.perRow / 4, where perRow counts the vector's groups of 32). A thread's group of 32
// is two words of codes, at the index its two vec4<u32> of int8 have in an int8 matrix (a row is n / 16 words either
// way), unpacked to those two vec4<u32>; the scale is its group of 128's, the same for four threads of a row. Every
// other line is fusedDp4aMatVec's: it became a function of its own because deviceKey() hashed that one's source (until
// T366: a parameter more changed every device's key). The key holds the text a maker makes now, so the two may be one
// maker whose int8 text stays what it is (tests/gpu-choice-check.mjs holds every line but those that read the weights
// the same in the two meanwhile).
//
// Adapted from ONNX Runtime, onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul_small_m.wgsl.template (n_bits
// 8: the loop fusedDp4aMatVec is. For 2-bit weights ORT has the same template's n_bits == 2 path, whose reads are this
// one's: the two u32 of 16 codes a thread's 32 weights are (a vec2<u32> at b_global * K32 + k_offset, K32 counting 32
// weights; the tiles' loadSHMB reads a u32 at b_global * K16 + kidx_v + col), the scale of block_idx = k_offset * 32 /
// block_size, k_offset / 4 for blocks of 128; where this unpacks by ALU, TERNARY_PACKED, ORT's table lookup is the
// other form, noted there) with the parameters dp4a_matmul_nbits.cc gives it (https://github.com/microsoft/onnxruntime,
// commit 3756d4dc, 2026-09-26), under the MIT License:
//
// Copyright (c) Microsoft Corporation
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
//
// (ternary_packed: the notice of Prism ML's fork of llama.cpp is above prompt.js's TERNARY_PACKED.)
export const ternaryMatVec = ({ output }) => /* wgsl */ `requires packed_4x8_integer_dot_product;
${FUSED_PARAMS}
${STEP}
@group(0) @binding(0) var<storage, read> b: array<u32>;              // the weights' codes, 16 of two bits to a u32
@group(0) @binding(1) var<storage, read> scales_b: array<f32>;       // a scale a group of 128 weights
@group(0) @binding(2) var<storage, read> a: array<vec4<u32>>;        // the quantized vector (xq)
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> scales_a: array<f32>;       // its scales (xs), one a group of 32
${fusedOutputs(output)}
${sdp8ai}${TERNARY_PACKED}

const workgroup_size_x = 128u;
const tile_size = ${ORT_DP4A_MATVEC_ROWS}u;
const tile_size_k_vec = 32u;
const sub_tile_count = workgroup_size_x / tile_size_k_vec;
const MATRICES = ${output === "swiglu" ? 2 : 1}u;
// fusedWrite's name for the rows of a workgroup
const OUTPUTS_PER_WG = tile_size;

const double_tile_size_k_vec = 2 * tile_size_k_vec;

var<workgroup> inter_results: array<array<f32, tile_size_k_vec>, tile_size * MATRICES>;
var<workgroup> tile_A : array<vec4<u32>, double_tile_size_k_vec>;
const scale_a_size_in_tile_a = double_tile_size_k_vec / 2;
var<workgroup> scale_A : array<f32, scale_a_size_in_tile_a>;
// what the reduction leaves: each row's sum (up's rows' after gate's), for the write
var<workgroup> totals: array<f32, tile_size * MATRICES>;

fn loadSHMA(kidx_v: u32, col: u32)
{
    let K16 = params.words / 4u;
    let k_offset = kidx_v + col;
    if (k_offset >= K16) {
    return;
    }

    tile_A[col] = a[k_offset];
    if (col < scale_a_size_in_tile_a && kidx_v / 2u + col < params.perRow)
    {
    // kidx_v - covers 16 values of k in input_a
    scale_A[col] = scales_a[kidx_v / 2u + col];
    }
}

@compute @workgroup_size(workgroup_size_x)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32) {
    let workgroup_idx = wg_id.y * num_wg.x + wg_id.x;
    let num_N_tile = (params.rows + tile_size - 1u) / tile_size;
    if (workgroup_idx >= num_N_tile) {
        return;
    }
    let K32 = params.perRow;
    // the scales of a row of the weights: one a group of 128
    let K128 = params.perRow / 4u;
    let b_global_base = workgroup_idx * tile_size;
    // Handle each workgroup threads as a block of [sub_tile_count][tile_size_k_vec]
    let local_col = local_idx % tile_size_k_vec;
    let local_row = local_idx / tile_size_k_vec;

    for (var kidx_v:u32 = 0; kidx_v < K32; kidx_v += tile_size_k_vec)
    {
        // Load Phase: Populate shared memory for the workgroup.
        if (local_idx < double_tile_size_k_vec)
        {
        loadSHMA(kidx_v * 2, local_idx);
        }
        workgroupBarrier();
        var own_a: vec4<u32> = tile_A[local_col * 2];
        var own_a1: vec4<u32> = tile_A[local_col * 2 + 1];
        var own_scale_a = scale_A[local_col];
        let k_offset = kidx_v + local_col;
        // k_offset - covers 32 values of k in input_b: two words of codes
        // calculate intermediate results into inter_results.
        for (var row_offset = 0u; row_offset < tile_size; row_offset += sub_tile_count) {
            let b_global = b_global_base + row_offset + local_row;
            if (b_global < params.rows && k_offset < K32)
            {
                let b_offset = b_global * K32 + k_offset;
                let own_scale_b = scales_b[b_global * K128 + k_offset / 4u];
                let own_b = ternary_packed(b[b_offset * 2]);
                let own_b1 = ternary_packed(b[b_offset * 2 + 1]);
                inter_results[row_offset + local_row][local_col] += SDP8AI(own_a, own_b, own_a1, own_b1, own_scale_a * own_scale_b);${output === "swiglu" ? `
                // up's row, params.second rows after gate's (mmvq.cu's vgate beside vx), with the same own_a
                let up_global = b_global + params.second;
                let up_offset = up_global * K32 + k_offset;
                let up_scale_b = scales_b[up_global * K128 + k_offset / 4u];
                inter_results[tile_size + row_offset + local_row][local_col] += SDP8AI(own_a, ternary_packed(b[up_offset * 2]), own_a1, ternary_packed(b[up_offset * 2 + 1]), own_scale_a * up_scale_b);` : ""}
            }
        }
        workgroupBarrier();
    }

    if (local_idx < tile_size * MATRICES) {
      // Do reduce sum to get final output.
      var output_value = f32(0);
      for (var b = 0u; b < tile_size_k_vec; b++) {
        output_value += inter_results[local_idx][b];
      }
      totals[local_idx] = output_value;
    }

    workgroupBarrier();

    let thread_id = local_idx;
    let row_base = b_global_base;
    let scale = 1.0;  // the norm is the quantizer's (NORM_QUANTIZE)
${fusedWrite(output)}
}`;
