// shaders/matvec.ts (T.351): T.149's matrix times one vector in the forms of public implementations that the engine has
// no text of: llama.cpp's mul_mat_vec in its MMVQ form (the int8 dot) and ONNX Runtime's MatMulNBits, each under its
// notice below. /benchmark/'s GPU section alone runs these two. (T.624.2: the other two forms T.149 measured, llama.cpp's
// widened mul_mat_vec and ORT's DP4A for small M, were copies here of what the fused layer of a generated token,
// fused.ts, is made from; /benchmark/ runs the engine's own texts of them now, fusedMatVec and fusedDp4aMatVec with the
// output "write", and this part takes from fused.ts the start of mul_mat_vec's main() and the rows a workgroup has.
// The engine's part reads nothing of this one: T.391.4.)
// A part of src/runtime/shaders.ts, which is the window: everything outside src/runtime/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part takes its neighbours' names by static imports (T.367.3:
// a bundler links them).
import { MUL_MAT_VEC_ROWS, mulMatVecMain } from "./fused.ts";

// ---- T.149: a matrix times one vector (a generated token's), in the forms of public implementations, for the GPU
// section of /benchmark/ to measure beside WIDEN and PACKED and the engine's two forms (T.624.2). WIDEN and PACKED give
// a row a workgroup of 64 threads that read one u32 each at a time; on the owner's Android they read Llama 3.2 1B's
// classifier at 27 to 36 GB/s but its w1 (16.8 MB) at 3.3 to 15.7 (T.134). These read more at once and give a
// workgroup several rows:
//   mulMatVec({ subgroups }): llama.cpp's WebGPU mul_mat_vec (mul_mat_vec.wgsl with mul_mat_vec_q_acc.tmpl's MMVQ path
//     for Q8_0): 256 threads for OUTPUTS_PER_WG rows, four threads to a group of 32 (two u32 of the weights against two
//     of the quantized vector by dot4I8Packed, a thread), for every row of the workgroup; the sums added up in the
//     workgroup's memory, or with subgroupAdd where subgroups. llama.cpp takes MMVQ only on AMD, Intel and NVIDIA; the
//     benchmark measures it everywhere.
//   ortMatVec: ONNX Runtime's MatMulNBits for 8 bits (matmul_nbits.wgsl.template, MatMulNBitsProgram: the form it takes
//     for a token where DP4A is not taken): 128 threads for tile_size 8 rows, 32 of them along the width, each reading
//     a vec4<u32> (16 weights) at a time; the vector's 512 values of a step in the workgroup's memory.
// Changed from the sources for this project's weights, and why: the int8 values and their float32 scales are two
// buffers (llama2_numpy's layout), so a group's scale is read from its own buffer (llama.cpp's Q8_0 blocks carry an f16
// scale in 34 bytes; ORT's scales are their own buffer already); the weights are signed (ORT's 8 bits are unsigned
// about 128: no zero point is taken off, and the widened form reads the bytes signed with extractBits rather than
// unpack4xU8, which also needs no language feature); the vector's quantized scales are one a group of 32 (llama.cpp's
// q8_1 blocks carry theirs); one vector (llama.cpp's NUM_COLS 1, ORT's M 1, no batches, no bias, no weight index:
// those loops and offsets are left out); the output goes to y from row first on (a matrix cut in chunks of rows). The
// bindings and the Shape are WIDEN's and PACKED's (x, or xq with its scales xs at 5), so that the benchmark binds them
// all alike (the engine's two forms bind as the engine does: benchmark/gpu/device.ts's bound()).
// (MUL_MAT_VEC_ROWS, the rows a workgroup of the first has, is fused.ts's)
export const ORT_MATVEC_ROWS = 8;  // ORT's tile_size for MatMulNBitsProgram
const MATVEC_SHAPE = /* wgsl */ `struct Shape { rows: u32, words: u32, perRow: u32, first: u32 }`;

// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl, mul_mat_vec_acc.tmpl (MUL_ACC_Q8_0),
// mul_mat_vec_q_acc.tmpl (MMVQ, LEGACY_QUANTS, MUL_ACC_Q8_0), with the defaults
// of ggml-webgpu-shader-lib.hpp (https://github.com/ggml-org/llama.cpp, commit 2145525a, 2026-09-26), under the MIT
// License:
//
/*! Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
// The preprocessor's defines are the template's parameters here (a function-scope array cannot take an override's
// size): WG_SIZE 256, OUTPUTS_PER_WG 4, NUM_COLS 1 (its loop and the barrier after each column left out).
const MUL_MAT_VEC_Q_ACC = /* wgsl */ `
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
}`;
export const mulMatVec = ({ subgroups }: { subgroups: boolean }): string => /* wgsl */ `${subgroups ? "enable subgroups;\nrequires subgroup_id;\n" : ""}requires packed_4x8_integer_dot_product;\n
${MATVEC_SHAPE}
@group(0) @binding(0) var<storage, read> src0: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> src1: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: Shape;
@group(0) @binding(5) var<storage, read> src1_scales: array<f32>;

const WG_SIZE = 256u;
const OUTPUTS_PER_WG = ${MUL_MAT_VEC_ROWS}u;
const BLOCK_SIZE = 32u;
const THREADS_PER_BLOCK = 4u;
const ELEMS_PER_THREAD = BLOCK_SIZE / THREADS_PER_BLOCK;
${MUL_MAT_VEC_Q_ACC}

// Flattened as [row][thread] to keep each row's reduction contiguous in memory.
var<workgroup> partial_sums: array<f32, OUTPUTS_PER_WG * WG_SIZE>;

fn partial_index(row: u32, thread: u32) -> u32 {
    return row * WG_SIZE + thread;
}

@compute @workgroup_size(WG_SIZE)
${mulMatVecMain(subgroups)}
    let dst_idx_base = params.first + row_base;

    let acc = accumulate_vec_q_dot(thread_id, row_base);
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
// component_a 4, component_b 4, no zero points) with the parameters matmul_nbits.cc gives it
// (https://github.com/microsoft/onnxruntime, commit 3756d4dc, 2026-09-26), under the MIT License:
//
/*! Copyright (c) Microsoft Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
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
