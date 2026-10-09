// shaders/matvec.js (T351): T149's matrix times one vector in the forms of public implementations (llama.cpp's
// mul_mat_vec, ONNX Runtime's MatMulNBits and its DP4A for small M, each under its notice below). /benchmark/'s GPU
// section alone runs these three; the fused layer of a generated token (fused.js) is made from them and takes the rows a
// workgroup has from here.
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).
const { sdp8ai } = await import(new URL(`prompt.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T149: a matrix times one vector (a generated token's), in the forms of public implementations, for the GPU
// section of /benchmark/ to measure beside WIDEN and PACKED (the model's GPU worker is to take the fastest on each
// device, T152). WIDEN and PACKED give a row a workgroup of 64 threads that read one u32 each at a time; on the owner's
// Android they read Llama 3.2 1B's classifier at 27 to 36 GB/s but its w1 (16.8 MB) at 3.3 to 15.7 (T134). These read
// more at once and give a workgroup several rows:
//   mulMatVec({ packed, subgroups }): llama.cpp's WebGPU mul_mat_vec (mul_mat_vec.wgsl with mul_mat_vec_acc.tmpl's
//     MUL_ACC_Q8_0, or with mul_mat_vec_q_acc.tmpl's MMVQ path for Q8_0 where packed): 256 threads for OUTPUTS_PER_WG
//     rows, four threads to a group of 32 (8 weights, two u32, a thread), the vector's 8 values held in registers for
//     every row of the workgroup; the sums added up in the workgroup's memory, or with subgroupAdd where subgroups.
//     llama.cpp takes MMVQ (the int8 dot) only on AMD, Intel and NVIDIA; the benchmark measures both everywhere.
//   ortMatVec: ONNX Runtime's MatMulNBits for 8 bits (matmul_nbits.wgsl.template, MatMulNBitsProgram: the form it takes
//     for a token where DP4A is not taken): 128 threads for tile_size 8 rows, 32 of them along the width, each reading
//     a vec4<u32> (16 weights) at a time; the vector's 512 values of a step in the workgroup's memory.
//   ortDp4aMatVec: ONNX Runtime's DP4A MatMulNBits for small M (dp4a_matmul_small_m.wgsl.template, which ORT takes for a
//     token on devices with subgroups other than Apple's when the output is f32): 128 threads for 4 rows, 32 along the
//     width, each a group of 32 as two vec4<u32> by dot4I8Packed against the quantized vector in the workgroup's memory.
// Changed from the sources for this project's weights, and why: the int8 values and their float32 scales are two
// buffers (llama2_numpy's layout), so a group's scale is read from its own buffer (llama.cpp's Q8_0 blocks carry an f16
// scale in 34 bytes; ORT's scales are their own buffer already); the weights are signed (ORT's 8 bits are unsigned
// about 128: no zero point is taken off, and the widened form reads the bytes signed with extractBits rather than
// unpack4xU8, which also needs no language feature); the vector's quantized scales are one a group of 32 (ORT's are
// one a 128, llama.cpp's q8_1 blocks carry theirs); one vector (llama.cpp's NUM_COLS 1, ORT's M 1, no batches, no bias,
// no weight index: those loops and offsets are left out); the output goes to y from row first on (a matrix cut in
// chunks of rows). The bindings and the Shape are WIDEN's and PACKED's (x, or xq with its scales xs at 5), so that
// the benchmark binds them all alike.
export const MUL_MAT_VEC_ROWS = 4;  // llama.cpp's WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG
export const ORT_MATVEC_ROWS = 8;  // ORT's tile_size for MatMulNBitsProgram
export const ORT_DP4A_MATVEC_ROWS = 4;  // ORT's tile_size_n for DP4AMatMulNBitsSmallMProgram
const MATVEC_SHAPE = /* wgsl */ `struct Shape { rows: u32, words: u32, perRow: u32, first: u32 }`;

// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl, mul_mat_vec_acc.tmpl (MUL_ACC_Q8_0),
// mul_mat_vec_q_acc.tmpl (MMVQ, LEGACY_QUANTS, MUL_ACC_Q8_0) and common_decls.tmpl (get_byte_i32), with the defaults
// of ggml-webgpu-shader-lib.hpp (https://github.com/ggml-org/llama.cpp, commit 2145525a, 2026-09-26), under the MIT
// License:
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
// The preprocessor's defines are the template's parameters here (a function-scope array cannot take an override's
// size): WG_SIZE 256, OUTPUTS_PER_WG 4, NUM_COLS 1 (its loop and the barrier after each column left out).
const mulMatVecAcc = (packed) => (packed ? /* wgsl */ `
fn accumulate_vec_q_dot(thread_id: u32, row_base: u32) -> array<f32, OUTPUTS_PER_WG> {
    var acc: array<f32, OUTPUTS_PER_WG>;

    let num_blocks = params.perRow;

    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let inner_id = thread_id % THREADS_PER_BLOCK;
        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.rows) {
                // repack_a: the block's two words of this thread; get_dm: the block's scale, from its own buffer
                let block_word_base = output_row * params.words + block * (BLOCK_SIZE / 4u);
                let a_repacked = vec2<u32>(src0[block_word_base + inner_id * 2u], src0[block_word_base + inner_id * 2u + 1u]);
                let da = scales[output_row * params.perRow + block];
                // repack_b_qs and repack_b_dm: the quantized vector's two words and its group's scale
                let b_repacked = vec2<u32>(src1[block * (BLOCK_SIZE / 4u) + inner_id * 2u], src1[block * (BLOCK_SIZE / 4u) + inner_id * 2u + 1u]);
                let b_ds = src1_scales[block];

                let row_sum = dot4I8Packed(a_repacked[0], b_repacked[0]) + dot4I8Packed(a_repacked[1], b_repacked[1]);

                acc[row] += f32(row_sum) * (da * b_ds);
            }
        }
    }

    return acc;
}` : /* wgsl */ `
fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

fn accumulate_vec_dot(thread_id: u32, row_base: u32) -> array<f32, OUTPUTS_PER_WG> {
    var acc: array<f32, OUTPUTS_PER_WG>;

    let num_blocks = params.perRow;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = block * BLOCK_SIZE + thread_within_block * ELEMS_PER_THREAD;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
            x_block[i] = src1[x_base + i];
        }
        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.rows) {
                // the block's scale from its own buffer, its values the row's words from block * 8 on
                let d = scales[output_row * params.perRow + block];
                let block_word_base = output_row * params.words + block * (BLOCK_SIZE / 4u);
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
                acc[row] += row_sum;
            }
        }
    }

    return acc;
}`);
export const mulMatVec = ({ packed, subgroups }) => /* wgsl */ `${subgroups ? "enable subgroups;\nrequires subgroup_id;\n" : ""}${packed ? "requires packed_4x8_integer_dot_product;\n" : ""}
${MATVEC_SHAPE}
@group(0) @binding(0) var<storage, read> src0: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> src1: array<${packed ? "u32" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: Shape;
${packed ? "@group(0) @binding(5) var<storage, read> src1_scales: array<f32>;" : ""}

const WG_SIZE = 256u;
const OUTPUTS_PER_WG = ${MUL_MAT_VEC_ROWS}u;
const BLOCK_SIZE = 32u;
const THREADS_PER_BLOCK = 4u;
const ELEMS_PER_THREAD = BLOCK_SIZE / THREADS_PER_BLOCK;
${mulMatVecAcc(packed)}

// Flattened as [row][thread] to keep each row's reduction contiguous in memory.
var<workgroup> partial_sums: array<f32, OUTPUTS_PER_WG * WG_SIZE>;

fn partial_index(row: u32, thread: u32) -> u32 {
    return row * WG_SIZE + thread;
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
    let dst_idx_base = params.first + row_base;

    let acc = ${packed ? "accumulate_vec_q_dot" : "accumulate_vec_dot"}(thread_id, row_base);
${subgroups ? `
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        let subgroup_total = subgroupAdd(acc[row]);
        if (subgroup_invocation_id == 0u) {
            partial_sums[partial_index(row, subgroup_id)] = subgroup_total;
        }
    }

    workgroupBarrier();

    for (var row = subgroup_id; (row < OUTPUTS_PER_WG) && (row_base + row < params.rows); row += num_subgroups) {
        var row_acc = 0.0f;
        for (var k = subgroup_invocation_id; k < num_subgroups; k += subgroup_size) {
            row_acc += partial_sums[partial_index(row, k)];
        }
        let row_total = subgroupAdd(row_acc);
        if (subgroup_invocation_id == 0) {
            dst[dst_idx_base + row] = row_total;
        }
    }` : `
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        partial_sums[partial_index(row, thread_id)] = acc[row];
    }

    workgroupBarrier();

    var stride = WG_SIZE / 2u;

    while (stride > 0) {
        if (thread_id < stride) {
            for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
                partial_sums[partial_index(row, thread_id)] += partial_sums[partial_index(row, thread_id + stride)];
            }
        }

        workgroupBarrier();
        stride = stride / 2;
    }

    if (thread_id < OUTPUTS_PER_WG) {
        let output_row = row_base + thread_id;
        if (output_row < params.rows) {
            dst[dst_idx_base + thread_id] = partial_sums[partial_index(thread_id, 0)];
        }
    }`}
}`;

// Adapted from ONNX Runtime, onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.wgsl.template (n_bits 8,
// component_a 4, component_b 4, no zero points) and dp4a_matmul_small_m.wgsl.template (n_bits 8) with the parameters
// matmul_nbits.cc and dp4a_matmul_nbits.cc give them (https://github.com/microsoft/onnxruntime, commit 3756d4dc,
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
// A is the vector (M = 1, a_global 0), B the weights (N = the rows); workgroup_idx is linear over x and then y.
export const ortMatVec = /* wgsl */ `
${MATVEC_SHAPE}
@group(0) @binding(0) var<storage, read> b: array<vec4<u32>>;       // the weights, 16 int8 to a vec4<u32>
@group(0) @binding(1) var<storage, read> scales_b: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: Shape;

const workgroup_size_x = 128u;
const tile_size = ${ORT_MATVEC_ROWS}u;
const tile_size_k_vec = 32u;
const sub_tile_count = workgroup_size_x / tile_size_k_vec;
const component_a = 4u;
const component_b = 4u;
const elements_in_value_b = component_b * (32u / 8u);
const tile_size_k = tile_size_k_vec * elements_in_value_b;
const a_length_per_tile = tile_size_k / component_a;
const block_size = 32u;

// four signed int8 of a u32 as floats (ORT's unpack4xU8 less the zero point of its unsigned 8 bits)
fn unpacked(v: u32) -> vec4<f32> {
  let w = bitcast<i32>(v);
  return vec4<f32>(vec4<i32>(extractBits(w, 0u, 8u), extractBits(w, 8u, 8u), extractBits(w, 16u, 8u), extractBits(w, 24u, 8u)));
}

// Shared memory
var<workgroup> tile_A : array<vec4<f32>, a_length_per_tile>;
var<workgroup> inter_results: array<array<f32, tile_size_k_vec>, tile_size>;

fn loadSHMA(kidx: u32, col: u32)
{
    let k_offset = kidx / component_a + col;
    if (k_offset < uniforms.words) {
        tile_A[col] = a[k_offset];
    } else {
        tile_A[col] = vec4<f32>(0);
    }
}

@compute @workgroup_size(workgroup_size_x)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32) {
  let workgroup_idx = wg_id.y * num_wg.x + wg_id.x;
  let num_N_tile = (uniforms.rows + tile_size - 1u) / tile_size;
  if (workgroup_idx >= num_N_tile) {
    return;
  }
  let K = uniforms.words * 4u;
  let K_of_b = K / elements_in_value_b;
  let b_global_base = workgroup_idx * tile_size;

  let idx = local_idx % tile_size_k_vec;
  let idy = local_idx / tile_size_k_vec;

  for (var kidx = 0u; kidx < K; kidx += tile_size_k)
  {
    for (var id = local_idx; id < a_length_per_tile; id += workgroup_size_x)
    {
      loadSHMA(kidx, id);
    }
    workgroupBarrier();

    for (var local_row_offset = 0u; local_row_offset < tile_size; local_row_offset += sub_tile_count)
    {
      var b_global = b_global_base + local_row_offset + idy;
      var k_offset = kidx / elements_in_value_b + idx;
      if (b_global < uniforms.rows && k_offset < K_of_b)
      {
        let block_idx = (kidx + idx * elements_in_value_b) / block_size;
        let scale_b = scales_b[b_global * uniforms.perRow + block_idx];
        var b_value = b[b_global * K_of_b + k_offset];

        var sum = f32(0);
        var a_offset = idx * (4u / component_a) * component_b;
        for (var i = 0u; i < component_b; i++) {
            let b_value_unpacked = unpacked(b_value[i]) * scale_b;
            sum += dot(tile_A[a_offset], b_value_unpacked);
            a_offset += 1;
        }

        inter_results[local_row_offset + idy][idx] += sum;
      }
    }
    workgroupBarrier();
  }

  if (local_idx < tile_size) {
    var output_value = f32(0);
    for (var b = 0u; b < tile_size_k_vec; b++) {
      output_value += inter_results[local_idx][b];
    }
    let b_global =  b_global_base + local_idx;
    if (b_global < uniforms.rows) {
      output[uniforms.first + b_global] = output_value;
    }
  }
}`;

// ORT's scale_A holds a scale for each 128 of the vector (8 a step): here one for each group of 32 (32 a step), loaded
// with a bound of its own (the tile's bound is in 16s)
export const ortDp4aMatVec = /* wgsl */ `requires packed_4x8_integer_dot_product;
${MATVEC_SHAPE}
@group(0) @binding(0) var<storage, read> b: array<vec4<u32>>;        // the weights, 16 int8 to a vec4<u32>
@group(0) @binding(1) var<storage, read> scales_b: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<vec4<u32>>;        // the quantized vector (xq)
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: Shape;
@group(0) @binding(5) var<storage, read> scales_a: array<f32>;       // its scales (xs), one a group of 32
${sdp8ai}

const workgroup_size_x = 128u;
const tile_size = ${ORT_DP4A_MATVEC_ROWS}u;
const tile_size_k_vec = 32u;
const sub_tile_count = workgroup_size_x / tile_size_k_vec;

const double_tile_size_k_vec = 2 * tile_size_k_vec;

var<workgroup> inter_results: array<array<f32, tile_size_k_vec>, tile_size>;
var<workgroup> tile_A : array<vec4<u32>, double_tile_size_k_vec>;
const scale_a_size_in_tile_a = double_tile_size_k_vec / 2;
var<workgroup> scale_A : array<f32, scale_a_size_in_tile_a>;

fn loadSHMA(kidx_v: u32, col: u32)
{
    let K16 = uniforms.words / 4u;
    let k_offset = kidx_v + col;
    if (k_offset >= K16) {
    return;
    }

    tile_A[col] = a[k_offset];
    if (col < scale_a_size_in_tile_a && kidx_v / 2u + col < uniforms.perRow)
    {
    // kidx_v - covers 16 values of k in input_a
    scale_A[col] = scales_a[kidx_v / 2u + col];
    }
}

@compute @workgroup_size(workgroup_size_x)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32) {
    let workgroup_idx = wg_id.y * num_wg.x + wg_id.x;
    let num_N_tile = (uniforms.rows + tile_size - 1u) / tile_size;
    if (workgroup_idx >= num_N_tile) {
        return;
    }
    let K32 = uniforms.perRow;
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
            if (b_global < uniforms.rows && k_offset < K32)
            {
                let b_offset = b_global * K32 + k_offset;
                let own_scale_b = scales_b[b_global * uniforms.perRow + block_idx];
                let own_b = b[b_offset * 2];
                let own_b1 = b[b_offset * 2 + 1];
                inter_results[row_offset + local_row][local_col] += SDP8AI(own_a, own_b, own_a1, own_b1, own_scale_a * own_scale_b);
            }
        }
        workgroupBarrier();
    }

    if (local_idx < tile_size) {
      // Do reduce sum to get final output.
      var output_value = f32(0);
      for (var b = 0u; b < tile_size_k_vec; b++) {
        output_value += inter_results[local_idx][b];
      }
      let b_global =  b_global_base + local_idx;
      if (b_global < uniforms.rows) {
        output[uniforms.first + b_global] = output_value;
      }
    }
}`;
