// shaders/prompt.js (T351): a matrix times the tokens of a prompt by tiles (T146): llama.cpp's register tiles,
// TensorFlow.js's packed vec4 and ONNX Runtime's DP4A (each under its notice below), ternary weights unpacked for the last
// (T232), the forms a device can make of them (T147's promptForms) and the check of one against JavaScript (tiledOff).
// SDP8AI and ternary_packed() are also what the DP4A forms of one vector (matvec.js, fused.js) are made of.
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).
const { GROUP, STEP } = await import(new URL(`common.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T146: a matrix times the tokens of a prompt by tiles (the benchmark measures them; the model's GPU worker is to
// take the fastest on each device, T147). BATCHED reads each weight once for TILE tokens but loads an activation for
// every multiply-add, and 64 threads add up every sum. These three take their form from public implementations
// instead (the owner, 2026-09-26: take the best public one rather than invent one):
//   regTile(half): llama.cpp's WebGPU register tiling (mul_mat_reg_tile.wgsl with mul_mat_decls.tmpl's Q8_0 and float
//     loaders). A workgroup owns TILE_M × WORKGROUP_SIZE_M rows by TILE_N × WORKGROUP_SIZE_N tokens; each step of
//     TILE_K = 32 (one group) widens the step's weights (× their scale) and copies the tokens' activations into the
//     workgroup's memory, then each thread multiplies its 4 rows by its 4 tokens with the sums in registers. half: the
//     workgroup's memory holds f16 as llama.cpp's does (shader-f16); the f32 form (no shader-f16) is this project's,
//     llama.cpp's register tiling is f16 only. The sums are f32 either way.
//   tfjsTile: TensorFlow.js's WebGPU makeMatMulPackedVec4Source (matmul_packed_webgpu.ts): the same classic tiles
//     (32 × 32, 8 × 8 threads, 4 × 4 a thread, 32 of the width a step), but the workgroup's memory is read as vec4 and
//     a thread's 4 tokens are one vec4 of sums (fma), where llama.cpp reads scalars (T146's review: a Mali GPU is
//     likely bound by the workgroup memory's reads, and on Apple llama.cpp's src0 rows 512 bytes apart share a bank).
//   dp4a(subgroups): ONNX Runtime Web's DP4A MatMulNBits (dp4a_matmul.wgsl.template, 8 bits, no zero points): a tile
//     of 64 tokens × 64 rows, 256 threads, 32 of the width a step as packed int8 in the workgroup's memory; each thread
//     one token × 16 rows, a group's int sum by dot4I8Packed times the two scales, as the CPU's matmul_q8 sums them.
//     The activations are quantized first (QUANTIZE). subgroups: the same with ORT's subgroupShuffle path where the
//     device's subgroups are 16 wide (Arm Valhall's), which reads the rows from the registers of the subgroup's lanes.
// Changed from the sources for this project's weights, and why: the int8 values and their float32 scales are two
// buffers (llama2_numpy's layout), not Q8_0's 34-byte blocks of f16 scale and 32 values, so the loaders read a group's
// scale from its own buffer; the weights are signed already (ORT's 8-bit ones are unsigned about 128); a group is 32
// for the activations too (ORT's scales_a are per 128); the outputs are written one float at a time, with each row and
// token checked and added to y where shape.add (a matrix cut in chunks of rows of any count, the residual stream),
// where ORT writes a vec4 and asks N % 16 == 0, llama.cpp a vec4 of rows and TensorFlow.js a vec4 of its columns; the workgroups are numbered as each
// source numbers them, over x and then y (a dispatch's dimension holds at most 65535; TensorFlow.js dispatches in two
// dimensions, here numbered as the others). The Shape, the Step and the
// bindings are BATCHED's (dp4a reads x as xq and adds the activations' scales, 6).

// llama.cpp's defaults (ggml-webgpu-shader-lib.hpp: WEBGPU_MUL_MAT_WG_SIZE_M/N 8, TILE_M/N 4, REG_TILE_K_QUANT 32)
// and a workgroup of 256 threads for a tile of 64 tokens (the benchmark measures both; the overrides are the pipeline's)
export const REG_TILES = [{ m: 8, n: 8 }, { m: 16, n: 16 }];
export const regTileShape = ({ m, n }) => ({ rows: 4 * m, tokens: 4 * n, threads: m * n });
// the workgroup's memory of a regTile: a step's weights and activations, 2 or 4 bytes each
export const regTileBytes = ({ m, n }, half) => 32 * 4 * (m + n) * (half ? 2 : 4);
export const DP4A_SHAPE = { rows: 64, tokens: 64, threads: 256 };

// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl, mul_mat_decls.tmpl and
// quant_inner_loops.tmpl (https://github.com/ggml-org/llama.cpp, commit 2145525a, 2026-09-26), under the MIT License:
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
export const regTile = (half) => /* wgsl */ `${half ? "enable f16;\n" : ""}
struct Shape { rows: u32, words: u32, perRow: u32, first: u32, xStride: u32, yStride: u32, add: u32, unused: u32 }
${STEP}
alias shmem_t = ${half ? "f16" : "f32"};
@group(0) @binding(0) var<storage, read> w: array<u32>;             // M rows, K columns: 4 int8 to a u32
@group(0) @binding(1) var<storage, read> scales: array<f32>;        // a scale a row and group of 32
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;       // N tokens, K columns (xStride floats apart)
@group(0) @binding(3) var<storage, read_write> y: array<f32>;       // N tokens, M rows (yStride floats apart)
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<uniform> step: Step;

override WORKGROUP_SIZE_M: u32 = 8u;
override WORKGROUP_SIZE_N: u32 = 8u;
const TILE_M = 4u;
const TILE_N = 4u;
const TILE_K = 32u;
const BLOCK_SIZE = 32u;
const BLOCKS_K = TILE_K / BLOCK_SIZE;
const NQ = 16u;
const BYTES_PER_THREAD = 16u;  // NQ(16) weights use 16 bytes of q
const BYTES_PER_INNER_LOOP = 4u;
override TOTAL_WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
override TILE_SRC0_SHMEM: u32 = TILE_K * WORKGROUP_SIZE_M * TILE_M;
override TILE_SRC1_SHMEM: u32 = TILE_K * WORKGROUP_SIZE_N * TILE_N;
override TILE_SHMEM: u32 = TILE_SRC0_SHMEM + TILE_SRC1_SHMEM;
var<workgroup> shmem: array<shmem_t, TILE_SHMEM>;

fn get_byte_i32(value: u32, index: u32) -> i32 {
  return bitcast<i32>(((value >> (index * 8u)) & 0xFFu) << 24u) >> 24u;
}
// Q8_0's loader: NQ weights a thread, widened with their scale (here a float32 of its own buffer, the product rounded
// once into the memory's type)
fn init_shmem_src0(thread_id: u32, offset_m: u32, k_outer: u32) {
  for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
    let block_idx = i / BLOCK_SIZE;
    let block_offset = (i % BLOCK_SIZE) / NQ;
    let shmem_idx = block_idx * BLOCK_SIZE + block_offset * BYTES_PER_THREAD;
    let tile_m = block_idx / BLOCKS_K;
    let global_m = offset_m + tile_m;
    let block_k = block_idx % BLOCKS_K;
    let global_block_k = k_outer / BLOCK_SIZE + block_k;
    if (global_m < shape.rows && global_block_k < shape.perRow) {
      let d = scales[global_m * shape.perRow + global_block_k];
      for (var j = 0u; j < BYTES_PER_THREAD / BYTES_PER_INNER_LOOP; j += 1u) {
        let q_packed = w[global_m * shape.words + global_block_k * (BLOCK_SIZE / 4u) + (block_offset * BYTES_PER_THREAD) / 4u + j];
        for (var k = 0u; k < 4u; k++) {
          shmem[shmem_idx + j * BYTES_PER_INNER_LOOP + k] = shmem_t(f32(get_byte_i32(q_packed, k)) * d);
        }
      }
    }
  }
}
// the activations' loader, four at a time: llama.cpp's VEC loader, which llama.cpp itself takes only for F32 and F16
// weights (Q8_0 takes its SCALAR one; x here is float32 and four aligned). A token past the request or a column past
// the width reads 0
fn init_shmem_src1(thread_id: u32, offset_n: u32, k_outer: u32) {
  let k = shape.words * 4u;
  for (var elem_idx = thread_id * 4u; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * 4u) {
    let tile_n = elem_idx / TILE_K;
    let tile_k = elem_idx % TILE_K;
    let global_n = offset_n + tile_n;
    let global_k = k_outer + tile_k;
    let src1_idx = global_n * shape.xStride + global_k;
    let src1_val = select(vec4<f32>(0.0), x[src1_idx / 4u], global_n < step.tokens && global_k < k);
    let at = TILE_SRC0_SHMEM + elem_idx;
    shmem[at] = shmem_t(src1_val.x);
    shmem[at + 1u] = shmem_t(src1_val.y);
    shmem[at + 2u] = shmem_t(src1_val.z);
    shmem[at + 3u] = shmem_t(src1_val.w);
  }
}

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
  let thread_id = local_id.x;
  let local_m = thread_id % WORKGROUP_SIZE_M;
  let local_n = thread_id / WORKGROUP_SIZE_M;

  let wg_n_count = (step.tokens + WORKGROUP_SIZE_N * TILE_N - 1u) / (WORKGROUP_SIZE_N * TILE_N);
  let wg_m_count = (shape.rows + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);
  let wg_linear = wg_id.y * num_wg.x + wg_id.x;
  if (wg_linear >= wg_m_count * wg_n_count) {
    return;
  }
  let wg_m = wg_linear % wg_m_count;
  let wg_n = wg_linear / wg_m_count;

  let output_row_base = wg_m * WORKGROUP_SIZE_M * TILE_M + local_m * TILE_M;
  let output_col_base = wg_n * WORKGROUP_SIZE_N * TILE_N + local_n * TILE_N;
  let offset_m = wg_m * WORKGROUP_SIZE_M * TILE_M;
  let offset_n = wg_n * WORKGROUP_SIZE_N * TILE_N;

  var acc: array<array<f32, TILE_N>, TILE_M>;
  let k = shape.words * 4u;
  for (var k_outer = 0u; k_outer < k; k_outer += TILE_K) {
    init_shmem_src0(thread_id, offset_m, k_outer);
    init_shmem_src1(thread_id, offset_n, k_outer);
    workgroupBarrier();
    let k_end = min(TILE_K, k - k_outer);
    for (var k_inner = 0u; k_inner < k_end; k_inner++) {
      var src0_tile: array<shmem_t, TILE_M>;
      for (var tm = 0u; tm < TILE_M; tm++) {
        let src0_m = local_m * TILE_M + tm;
        let src0_idx = k_inner + src0_m * TILE_K;
        src0_tile[tm] = shmem[src0_idx];
      }
      for (var tn = 0u; tn < TILE_N; tn++) {
        let src1_n = local_n * TILE_N + tn;
        let src1_idx = src1_n * TILE_K + k_inner;
        let src1_val = shmem[TILE_SRC0_SHMEM + src1_idx];
        for (var tm = 0u; tm < TILE_M; tm++) {
          acc[tm][tn] += f32(src0_tile[tm]) * f32(src1_val);
        }
      }
    }
    workgroupBarrier();
  }

  for (var tn = 0u; tn < TILE_N; tn++) {
    let global_col = output_col_base + tn;
    if (global_col < step.tokens) {
      for (var tm = 0u; tm < TILE_M; tm++) {
        let global_row = output_row_base + tm;
        if (global_row < shape.rows) {
          let at = global_col * shape.yStride + shape.first + global_row;
          y[at] = select(0.0, y[at], shape.add != 0u) + acc[tm][tn];
        }
      }
    }
  }
}`;

// TensorFlow.js's tile: 32 rows × 32 tokens, 8 × 8 threads, 4 rows × 4 tokens a thread
export const TFJS_SHAPE = { rows: 32, tokens: 32, threads: 64 };

// Adapted from TensorFlow.js, tfjs-backend-webgpu/src/matmul_packed_webgpu.ts (makeMatMulPackedVec4Source,
// matMulReadFnSource and matMulReadWriteFnSource; https://github.com/tensorflow/tfjs, 2026-09-26).
// Copyright 2019 Google LLC. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
// Changed: A is the weights (M = the rows), read as 4 int8 of a u32 widened with their group's scale; B is the
// activations transposed (N = the tokens, a vec4 of 4 tokens at one column of the width), so that a thread's vec4 of
// sums is 4 tokens of a row; the tile's number is linear over x and then y as the other tiled shaders'; mm_write
// writes the 4 tokens one float at a time with shape.add; workPerThread [4, 4], workgroupSize [8, 8, 1] and
// tileInner 32 are TensorFlow.js's for large products (computeWorkgroupInfoForMatMul), and not transposed.
export const tfjsTile = /* wgsl */ `
struct Shape { rows: u32, words: u32, perRow: u32, first: u32, xStride: u32, yStride: u32, add: u32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<uniform> step: Step;

const rowPerThread = 4;
const colPerThread = 4;
const tileInner = 32;
const innerElementSize = 4;
const rowPerThreadB = 4;  // tileInner / workgroupSize[1]
const tileAOuter = 32;
const tileBOuter = 32;

var<workgroup> mm_Asub : array<array<vec4<f32>, 8>, 32>;
var<workgroup> mm_Bsub : array<array<vec4<f32>, 8>, 32>;

// four weights of a row (columns col to col + 3) times their group's scale
fn mm_readA(row: i32, col: i32) -> vec4<f32> {
  var value = vec4<f32>(0.0);
  if (row < i32(shape.rows) && col < i32(shape.words * 4u)) {
    let word = w[u32(row) * shape.words + u32(col) / 4u];
    let q = vec4<i32>(bitcast<i32>(word << 24u), bitcast<i32>(word << 16u), bitcast<i32>(word << 8u), bitcast<i32>(word)) >> vec4<u32>(24u);
    value = vec4<f32>(q) * scales[u32(row) * shape.perRow + u32(col) / ${GROUP}u];
  }
  return value;
}
// the activations of four tokens (col to col + 3) at one column of the width (row)
fn mm_readB(row: i32, col: i32) -> vec4<f32> {
  var value = vec4<f32>(0.0);
  if (row < i32(shape.words * 4u)) {
    for (var i = 0; i < 4; i++) {
      if (col + i < i32(step.tokens)) {
        value[i] = x[u32(col + i) * shape.xStride + u32(row)];
      }
    }
  }
  return value;
}
fn mm_write(row: i32, col: i32, valueIn: vec4<f32>) {
  if (row < i32(shape.rows)) {
    for (var i = 0; i < 4; i++) {
      if (col + i < i32(step.tokens)) {
        let at = u32(col + i) * shape.yStride + shape.first + u32(row);
        y[at] = select(0.0, y[at], shape.add != 0u) + valueIn[i];
      }
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) localId: vec3<u32>, @builtin(workgroup_id) workgroupId: vec3<u32>,
        @builtin(num_workgroups) numWorkgroups: vec3<u32>) {
  let tilesA = (shape.rows + 31u) / 32u;
  let tilesB = (step.tokens + 31u) / 32u;
  let linear = workgroupId.y * numWorkgroups.x + workgroupId.x;
  if (linear >= tilesA * tilesB) {
    return;
  }
  let localRow = i32(localId.y);
  let tileRow = localRow * rowPerThread;
  let tileCol = i32(localId.x);

  let globalRow = i32(linear % tilesA) * tileAOuter + tileRow;
  let globalCol = i32(linear / tilesA) * tileBOuter + tileCol * colPerThread;

  let numTiles = (i32(shape.words * 4u) - 1) / tileInner + 1;
  var kStart = 0;

  var acc: array<vec4<f32>, rowPerThread>;

  // Loop over shared dimension.
  let tileRowB = localRow * rowPerThreadB;
  for (var t = 0; t < numTiles; t++) {
      // Load one tile of A into local memory.
      for (var innerRow = 0; innerRow < rowPerThread; innerRow++) {
          let inputRow = tileRow + innerRow;
          let inputCol = tileCol;
          mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, kStart + inputCol * innerElementSize);
      }

      // Load one tile of B into local memory.
      for (var innerRow = 0; innerRow < rowPerThreadB; innerRow++) {
          let inputRow = tileRowB + innerRow;
          let inputCol = tileCol;
          mm_Bsub[inputRow][inputCol] = mm_readB(kStart + inputRow, globalCol);
      }
      kStart = kStart + tileInner;
      workgroupBarrier();

      // Compute acc values for a single thread.
      for (var k = 0; k < tileInner / innerElementSize; k++) {
        let BCached0 = mm_Bsub[k * innerElementSize + 0][tileCol];
        let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];
        let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];
        let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];
        for (var i = 0; i < rowPerThread; i++) {
          let ACached = mm_Asub[tileRow + i][k];
          acc[i] = fma(BCached0, vec4<f32>(ACached[0]), acc[i]);
          acc[i] = fma(BCached1, vec4<f32>(ACached[1]), acc[i]);
          acc[i] = fma(BCached2, vec4<f32>(ACached[2]), acc[i]);
          acc[i] = fma(BCached3, vec4<f32>(ACached[3]), acc[i]);
        }
      }
      workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < rowPerThread; innerRow++) {
      mm_write(globalRow + innerRow, globalCol, acc[innerRow]);
  }
}`;

// ---- T232: ternary weights (T230: Prism ML's Ternary Bonsai) on the GPU, as the checkpoint holds them: PQ2_0's
// codes, two bits a weight (weight j of a row at bits 2 (j % 4) of byte j / 4: 0, 1, 2 for -1, 0, +1 times the scale
// of its group of 128 along the row; 3, which no ternary file has, is +2, as public/forward/engine.js's weightAt and
// llama2_numpy.unpack_ternary read it), 16 weights a u32, and a float32 scale a group of 128 in a buffer of its own.
// Nothing is widened as it goes up (the 8B is 2.3 GB so and 9.2 GB as int8): a row of n weights is n / 4 bytes and
// n / 128 scales, which is the layout of an int8 row of n / 4 weights (32 bytes of values to a scale), so gpu.js's
// buffers, pieces and joined matrices count bytes as they did.
//
// The form is the packed-integer one the engine has from ONNX Runtime (the tiles of dp4a above for a prompt's block,
// fusedDp4aMatVec for a generated token: the activations quantized to int8 first by QUANTIZE or NORM_QUANTIZE, a scale
// a group of 32), with the weights unpacked from their codes to packed int8 where ORT loads them: ternary_packed(), a
// word of 16 codes to the vec4<u32> of 16 int8 that dot4I8Packed takes. Its first two lines are the fork's that Prism
// ML ships the models with (unpack_pq2_0 of its Vulkan backend: a byte's four pairs of bits to the low bits of the four
// bytes of a word), on the four bytes of the word at once. Then a code less one in every byte: the fork keeps the codes
// as they are and takes the activations' sum off afterwards (its q8_1 blocks carry that sum: mul_q8_1), as this
// project's CPU kernel does (kernels/ternary.ts, T231); here the activations have no sum beside them (QUANTIZE and
// NORM_QUANTIZE are the int8 models' too, and their text is not to change: deviceKey()), so the codes are made signed,
// two instructions a word of 16 weights more (an add and an xor of constants; see the count below), and every line
// after the load is ORT's as it was, the scales' NaN of T241 with it (a group's dot times scale_a × scale_b).
// The inner loop of a generated token, a thread's 32 weights of a row against a group of 32 of the vector:
//   2 loads of codes; twice unpack4xU8, 2 shifts, 2 ors, 2 ands, an add and an xor on a vec4<u32> (18 instructions where
//   a vec4 is one, 72 where it is four); 8 dot4I8Packed and 7 adds; a conversion, the two scales' product and the
//   product; the store: about 40, or 95 where vectors are scalar (int8's DP4A: 2 loads of vec4, 8 dots, 7 adds and the
//   same 4: 21, on four times the bytes). Kept unsigned with the sum taken off (the fork's and the CPU's form) it is 4
//   vector instructions fewer and 2 reads of the workgroup's memory, an add and a subtraction more, with a sum made by
//   each thread that loads the vector: fewer instructions where vectors are scalar, as many where they are not. Which
//   is faster is a device's to say, and none has measured it (TODO.md's T232 has what would overturn this choice).
// (The T232 review.) ONNX Runtime has these weights too: the n_bits == 2 path of its templates unpacks a word by a table
// of 256 words in the workgroup's memory (dp4a_matmul_common.wgsl.template's DequantizedFrom2BitsTo8Bits and
// LoadDequantizationTable; the quarter of its table with zero point 1 is the code less one, this project's mapping): 4
// reads of the table a word of 16 codes where this is 9 vector instructions (36 where vectors are scalar). Not taken: the
// fork's is the form for these files, and a wave reading a table at random meets its banks (simulated: 2.3 cycles a read
// for 16 lanes, 3.2 for 32, on 32 banks); where an instruction is what bounds a device the table may be faster. Nothing
// is measured: a device's to say, as T146's forms are (TODO.md's T232 review proposes giving it both).
// In the tiles the unpacking is in the load of the workgroup's memory, once a row for the tile's 64 tokens.
// A device without the packed int8 dot (packed_4x8_integer_dot_product) keeps a ternary model on the CPU: the float
// forms (llama.cpp's tiles, TensorFlow.js's) would unpack every weight to a float and multiply it, where the model's
// point is that it need not.
//
// ternary_packed() is adapted from PrismML-Eng/llama.cpp, ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl
// (unpack_pq2_0 and repack4; https://github.com/PrismML-Eng/llama.cpp, commit 88c4bc60), under the MIT License:
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
// Changed: the four bytes of a word at once (unpack4xU8 and vec4<u32>, where the fork calls it a byte at a time); the
// codes made signed (the last line), which is this project's: 0x7f added to a byte of 0 to 3 carries nothing into the
// next byte, and the top bit flipped leaves 0xff, 0x00, 0x01 (and 0x02): the code less one as an int8.
const TERNARY_PACKED = /* wgsl */ `
// T232: a word of 16 ternary codes (two bits each, the first in the lowest bits) as 16 int8 (the code less one), four
// to a u32 with the first in the lowest byte
fn ternary_packed(codes: u32) -> vec4<u32> {
  // Move bit pairs [1:0], [3:2], [5:4], [7:6] to [1:0], [9:8], [17:16], [25:24].
  var bits = unpack4xU8(codes);
  bits = (bits | (bits << vec4<u32>(12u))) & vec4<u32>(0x000f000fu);
  bits = (bits | (bits << vec4<u32>(6u))) & vec4<u32>(0x03030303u);
  return (bits + vec4<u32>(0x7f7f7f7fu)) ^ vec4<u32>(0x80808080u);
}`;
/** JavaScript's unpacking (the checks' answer): the int8 values of ternary codes (a Uint8Array, four weights a byte),
 * as llama2_numpy.unpack_ternary and public/forward/engine.js's weightAt read them */
export function ternaryValues(packed) {
  const out = new Int8Array(packed.length * 4);
  for (let i = 0; i < out.length; i++) out[i] = ((packed[i >> 2] >> (2 * (i & 3))) & 3) - 1;
  return out;
}

// Adapted from ONNX Runtime, onnxruntime/contrib_ops/webgpu/quantization/dp4a_matmul.wgsl.template and
// dp4a_matmul_common.wgsl.template (https://github.com/microsoft/onnxruntime, commit 3756d4dc, 2026-09-26), under the
// MIT License:
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
// A is the activations (M = the tokens), B the weights (N = the rows): "a_global" is a token and "b_global" a row.
const sdp8ai = /* wgsl */ `
// Scaled dot product of 8 packed integers.
fn SDP8AI(a1: vec4<u32>, b1: vec4<u32>, a2: vec4<u32>, b2: vec4<u32>, scale: f32) -> f32 {
  var local_sum = dot4I8Packed(a1[0], b1[0]);
  local_sum += dot4I8Packed(a1[1], b1[1]);
  local_sum += dot4I8Packed(a1[2], b1[2]);
  local_sum += dot4I8Packed(a1[3], b1[3]);
  local_sum += dot4I8Packed(a2[0], b2[0]);
  local_sum += dot4I8Packed(a2[1], b2[1]);
  local_sum += dot4I8Packed(a2[2], b2[2]);
  local_sum += dot4I8Packed(a2[3], b2[3]);
  return f32(local_sum) * scale;
}`;
// ORT's step 2, one line a row of the subtile: from the workgroup's memory, or from the lanes of a subgroup of 16
const dp4aLines = (subgroup) => Array.from({ length: 16 }, (_, i) => `    lane_output${(i >> 2) + 1}[${i & 3}] += ` + (subgroup
  ? `SDP8AI(own_a0, subgroupShuffle(own_b0, ${i}u), own_a1, subgroupShuffle(own_b1, ${i}u), subgroupShuffle(own_scale_b, ${i}u) * own_scale_a);`
  : `SDP8AI(own_a0, tile_B[0][base_B + ${i}u], own_a1, tile_B[1][base_B + ${i}u], own_scale_a * scale_B[base_B + ${i}u]);`)).join("\n");
// T232, ternary: the weights are PQ2_0's codes as the checkpoint holds them (TERNARY_PACKED below, and what is changed
// for them there); the text without it is what it was, to the byte (deviceKey() hashes it: a device keeps the shader it
// remembers)
export const dp4a = (subgroups, ternary = false) => /* wgsl */ `requires packed_4x8_integer_dot_product;
${subgroups ? "enable subgroups;\n" : ""}
struct Shape { rows: u32, words: u32, perRow: u32, first: u32, xStride: u32, yStride: u32, add: u32, unused: u32 }
${STEP}
${ternary ? "@group(0) @binding(0) var<storage, read> b: array<u32>;                  // the weights' codes, 16 of two bits to a u32"
    : "@group(0) @binding(0) var<storage, read> b: array<vec4<u32>>;        // the weights, 16 int8 to a vec4<u32>"}
@group(0) @binding(1) var<storage, read> scales_b: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<vec4<u32>>;        // the quantized activations (QUANTIZE's xq)
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<uniform> step: Step;
@group(0) @binding(6) var<storage, read> scales_a: array<f32>;       // QUANTIZE's xs: a scale a token and group of 32
${sdp8ai}${ternary ? TERNARY_PACKED : ""}

const tile_size = 64u;
const subtile_size = 16u;
const tile_size_k_vec = 2u;

// Shared memory
var<workgroup> tile_A: array<array<vec4<u32>, tile_size>, tile_size_k_vec>;  // 64 x 32
var<workgroup> scale_A: array<f32, tile_size>;                                // 64 x 1
var<workgroup> tile_B: array<array<vec4<u32>, tile_size>, tile_size_k_vec>;  // 64 x 32
var<workgroup> scale_B: array<f32, tile_size>;                                // 64 x 1

fn loadSHMA(a_global_base: u32, kidx_v: u32, row: u32, col: u32) {
  let a_global = a_global_base + row;
  if (a_global >= step.tokens) {
    return;
  }
  tile_A[col][row] = a[a_global * (shape.xStride / 16u) + kidx_v + col];
  if (col == 0u) {
    // kidx_v covers 16 values of k: a group of 32 is two
    scale_A[row] = scales_a[a_global * (shape.xStride / 32u) + kidx_v / 2u];
  }
}
fn loadSHMB(b_global_base: u32, kidx_v: u32, row: u32, col: u32) {
  let b_global = b_global_base + row;
  if (b_global >= shape.rows) {
    return;
  }
${ternary ? `  // (a row of n weights is n / 16 words of codes, as many as an int8 row's vec4s: the same index)
  tile_B[col][row] = ternary_packed(b[b_global * (shape.words / 4u) + kidx_v + col]);
  if (col == 0u) {
    // a scale a group of 128 weights: four of the activations' groups of 32
    scale_B[row] = scales_b[b_global * (shape.perRow / 4u) + kidx_v / 8u];
  }` : `  tile_B[col][row] = b[b_global * (shape.words / 4u) + kidx_v + col];
  if (col == 0u) {
    scale_B[row] = scales_b[b_global * shape.perRow + kidx_v / 2u];
  }`}
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32${subgroups ? `,
        @builtin(subgroup_size) sg_size: u32, @builtin(subgroup_invocation_id) sg_id: u32` : ""}) {
  let num_M_tile = (step.tokens + tile_size - 1u) / tile_size;
  let num_N_tile = (shape.rows + tile_size - 1u) / tile_size;
  let workgroup_idx = wg_id.y * num_wg.x + wg_id.x;
  if (workgroup_idx >= num_M_tile * num_N_tile) {
    return;
  }
  // During the load phase we use all 256 threads to load 64 rows of A/B.
  // For each row we load tile_size_k_vec (2) vectorized elements, which are 32 elements of K.
  let a_global_base = (workgroup_idx / num_N_tile) * tile_size;
  let b_global_base = (workgroup_idx % num_N_tile) * tile_size;
  let load_AorB = local_idx / 128u;
  let load_row = (local_idx % 128u) / 2u;
  let load_col = local_idx % 2u;

  // During the compute phase, we have the 64x64 tile split into subtiles of 16x16. We have a grid of 4x4 subtiles.
  let subtile_id = local_idx / subtile_size;
  let subtile_idx = subtile_id / 4u;
  let subtile_idy = subtile_id % 4u;
  let base_A = subtile_idx * 16u;
  let base_B = subtile_idy * 16u;
  // For each subtile we have 16 threads assigned.
  let a_idx = local_idx % subtile_size;

  var lane_output1: vec4<f32>;
  var lane_output2: vec4<f32>;
  var lane_output3: vec4<f32>;
  var lane_output4: vec4<f32>;
  // K's vectorization is 16 items per index; tile_size_k_vec (2) is the k tile of 32 in it.
  let K16 = shape.words / 4u;
  for (var kidx_v = 0u; kidx_v < K16; kidx_v += tile_size_k_vec) {
    // Load Phase: Populate shared memory for the workgroup.
    if (load_AorB == 0u) {
      loadSHMA(a_global_base, kidx_v, load_row, load_col);
    } else {
      loadSHMB(b_global_base, kidx_v, load_row, load_col);
    }
    workgroupBarrier();

    // Compute phase: Perform matmul for this subtile 16 x 32 x 16.
    // Step 1: Load from shared memory into registers across entire subgroup.
    let own_a0: vec4<u32> = tile_A[0][base_A + a_idx];
    let own_a1: vec4<u32> = tile_A[1][base_A + a_idx];
    let own_scale_a: f32 = scale_A[base_A + a_idx];
${subgroups ? `    if (sg_size == 16u) {
      let own_b0: vec4<u32> = tile_B[0][base_B + sg_id];
      let own_b1: vec4<u32> = tile_B[1][base_B + sg_id];
      let own_scale_b: f32 = scale_B[base_B + sg_id];
      // Step 2: Access registers across the subgroup using subgroupShuffle and perform the matmul.
${dp4aLines(true).replace(/^/gm, "  ")}
    } else {
      // Code for other subgroup sizes, simply doesn't use subgroups at all.
${dp4aLines(false).replace(/^/gm, "  ")}
    }` : `    // Relies on reads from single location tile_B[][base_B + col] by all being optimized by the hardware.
${dp4aLines(false)}`}
    workgroupBarrier();
  }
  let a_global = a_global_base + base_A + a_idx;
  let b_global = b_global_base + base_B;
  if (a_global < step.tokens) {
    let outputs = array<vec4<f32>, 4>(lane_output1, lane_output2, lane_output3, lane_output4);
    for (var i = 0u; i < 16u; i++) {
      if (b_global + i < shape.rows) {
        let at = a_global * shape.yStride + shape.first + b_global + i;
        y[at] = select(0.0, y[at], shape.add != 0u) + outputs[i / 4u][i % 4u];
      }
    }
  }
}`;

// ---- T147: the tiled shaders a device may run a prompt's matrices with (T146's), for the model's GPU worker, which
// checks each against JavaScript on a small matrix (tiledOff) and times the right ones on the model's own weights, and
// takes the fastest: which is fastest differs from GPU to GPU (T146), and only the device can say. none: why a shape
// is not made here (the device's threads or workgroup memory), as the benchmark says it (public/benchmark/gpu.js).
// T232, ternary: the forms of a model of ternary weights, ORT's DP4A alone with the weights unpacked from their codes
// (dp4a's ternary: the float forms would widen every weight to a float); none where there is no packed int8 dot
export const promptForms = ({ half, subgroups, packed, memory, threads, ternary = false }) => {
  const past = ({ threads: wanted }, bytes) => (wanted > threads ? `${wanted} threads, the device ${threads}`
    : bytes > memory ? `${bytes} bytes of workgroup memory, the device ${memory}` : undefined);
  if (ternary) {
    const none = packed ? past(DP4A_SHAPE, 4608) : "no packed int8 dot here";
    return [false, ...(subgroups ? [true] : [])].map((lanes) => ({ name: `ORT DP4A 64×64${lanes ? ", subgroups" : ""}, ternary`,
      tile: DP4A_SHAPE, packed: true, half: false, ternary: true, code: dp4a(lanes, true), none }));
  }
  const forms = REG_TILES.map((tile) => {
    const shape = regTileShape(tile);
    return { name: `llama.cpp tiles ${shape.rows}×${shape.tokens}, ${half ? "f16" : "f32"}`, tile: shape, packed: false, half,
      code: regTile(half), constants: { WORKGROUP_SIZE_M: tile.m, WORKGROUP_SIZE_N: tile.n }, none: past(shape, regTileBytes(tile, half)) };
  });
  forms.push({ name: "TF.js tiles 32×32, vec4", tile: TFJS_SHAPE, packed: false, half: false, code: tfjsTile, none: past(TFJS_SHAPE, 2 * 32 * 32 * 4) });
  const dp4aNone = packed ? past(DP4A_SHAPE, 4608) : "no packed int8 dot here";
  forms.push({ name: "ORT DP4A 64×64", tile: DP4A_SHAPE, packed: true, half: false, code: dp4a(false), none: dp4aNone });
  if (subgroups) forms.push({ name: "ORT DP4A 64×64, subgroups", tile: DP4A_SHAPE, packed: true, half: false, code: dp4a(true), none: dp4aNone });
  return forms;
};

// x (groups of GROUP values) quantized as the CPU's quantize_x does it (and QUANTIZE): the largest |value| / 127,
// round half to even
export function quantizedLikeCpu(x) {
  const xq = new Int8Array(x.length), xs = new Float32Array(x.length / GROUP);
  for (let g = 0; g < xs.length; g++) {
    let largest = 0;
    for (let i = 0; i < GROUP; i++) largest = Math.max(largest, Math.abs(x[g * GROUP + i]));
    xs[g] = Math.fround(largest / 127);
    for (let i = 0; i < GROUP; i++) {
      const v = x[g * GROUP + i] / xs[g], r = Math.round(v);
      xq[g * GROUP + i] = Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 ? r - 1 : r;
    }
  }
  return { xq, xs };
}

// How far a tiled shader's products (got: y after the product twice, the second added: 2 × W·x) are from
// JavaScript's, the check of T146's review (public/benchmark/gpu/check.js's checkTiled): w, int8 [rows][n] with the float32
// scales s [rows][n / 32]; x, the tokens xStride apart; got, yStride apart. A packed form multiplies what the GPU
// quantized (xq, xs, xStride apart), held to JavaScript's quantize_x first: a scale may differ in its last bits (WGSL's
// division is not rounded exactly) and a value then by 1, a wrong index by far more. f32 forms: no more than 1e-4 of
// the row's and token's sum of |products| (a float32 sum in another order may differ by 544 × 2^-24 = 3.2e-5 of it at
// most; a wrong index, scale or group by about 1 / sqrt(544) = 4e-2). f16 forms: WGSL leaves the direction of the
// rounding to f16 to the device, so each weight × scale and activation is within 1 ulp (2^-10, or 2^-24 where it is
// subnormal): no more than |products| × (2^-9 + 2^-20 + (n + 1) × 2^-24) + 2^-24 × Σ(|weight| + |activation|).
// Returns { worst, wrong, far, apart, values }: the worst difference over the sum of |products|, why it is wrong or
// null, and of a packed form whether a quantized value is far from quantize_x's and how many of values differ by 1.
// the f32 forms' line of tiledOff, a share of the sum of |products| (the matrix × vector checks of the benchmark hold theirs to it too)
// T232: group, the weights a scale of s (32; 128 for ternary weights, w then their values of -1 to 2: ternaryValues)
export const TILED_LINE = 1e-4;
export function tiledOff({ w, s, x, got, xq, xs, rows, n, tokens, xStride, yStride, half, group = GROUP }) {
  const perRow = n / GROUP, scaleOf = (r, g) => s[r * (n / group) + Math.floor((g * GROUP) / group)];
  let worst = 0, over = false, far = false, apart = 0, values = 0;
  for (let t = 0; t < tokens; t++) {
    const at = t * xStride, groups = t * (xStride / GROUP);
    const mine = xq ? quantizedLikeCpu(x.subarray(at, at + n)) : null;
    if (mine) {
      for (let g = 0; g < perRow; g++) far ||= Math.abs(xs[groups + g] - mine.xs[g]) > 1e-6 * mine.xs[g];
      for (let i = 0; i < n; i++) {
        far ||= Math.abs(xq[at + i] - mine.xq[i]) > 1;
        apart += xq[at + i] !== mine.xq[i];
      }
      values += n;
    }
    for (let r = 0; r < rows; r++) {
      let want = 0, size = 0, small = 0;
      for (let g = 0; g < perRow; g++) {
        const scale = scaleOf(r, g) * (mine ? xs[groups + g] : 1);
        for (let i = g * GROUP; i < (g + 1) * GROUP; i++) {
          const weight = Math.fround(w[r * n + i] * scaleOf(r, g)), value = x[at + i];
          const product = mine ? w[r * n + i] * xq[at + i] * scale : half ? weight * value : w[r * n + i] * value * scaleOf(r, g);
          want += product;
          size += Math.abs(product);
          small += Math.abs(weight) + Math.abs(value);
        }
      }
      const off = Math.abs(got[t * yStride + r] / 2 - want);
      worst = Math.max(worst, off / size);
      // (T232: a difference that is no number is wrong too: a NaN is neither over a line nor under it, and passed)
      over ||= !(half ? off <= size * (2 ** -9 + 2 ** -20 + (n + 1) * 2 ** -24) + small * 2 ** -24 : off < TILED_LINE * size);
    }
  }
  const wrong = far ? "the quantized activations are far from quantize_x's" : apart > 0.01 * values
    ? `${apart} of ${values} quantized activations are not quantize_x's` : over ? `products ${worst.toExponential(2)} from JavaScript's` : null;
  return { worst, wrong, far, apart, values };
}

export { TERNARY_PACKED, sdp8ai };
