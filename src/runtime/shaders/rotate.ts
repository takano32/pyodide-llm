// shaders/rotate.ts (T271.1): a rotated basis on the GPU (T237: Ternary Bonsai 2 27B's matrices are stored as W R^-1,
// R = H S, H the normalized Walsh-Hadamard transform of every block values, S a sign a value). A matrix then reads R x:
// ROTATE_QUANTIZE turns its input and quantizes it for the packed matrices in one dispatch (in the place of QUANTIZE,
// so that a token takes no more dispatches than it did); a row of the embedding is R^-1 of what the model reads, and
// UNROTATE turns it back where the GPU embeds a token. Both are the CPU's kernels (kernels/kernel/activations.ts's
// rotate and unrotate), the same float32 operations in the same order: the signs (whose magnitude is the transform's
// 1 / sqrt(block), multiplied in first as NumPy does), then sums and differences of values 1, 2, 4, ... apart, so a
// device whose float32 is IEEE's gives the CPU's numbers; and the 8 bits are QUANTIZE's (steps.ts: its packing and its
// scale word, T241's NaN of a group's scale).
//
// The form is the fork's Vulkan shader's for the transform as T232's spec of T271 reads it (https://github.com/PrismML-Eng/llama.cpp,
// commit 88c4bc60, MIT, ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp: a block a workgroup, its values in the
// workgroup's memory (its FWHT_SHMEM way), a barrier a stage of butterflies). No line is taken: that shader keeps a block in registers and swaps them
// over subgroups or shared memory by stages, where these keep it in the workgroup's memory throughout (a phone's
// subgroups are 16 wide, the fork's are 32), and the quantizing of its groups after the last stage has no public form
// (T232's spec of T271: QUANTIZE's lines and the transform's in one workgroup).
import { PACKED_WORD, SCALE_WORD } from "./steps.ts";

// the largest block the workgroup holds (1024: the 27B's), and its threads
export const ROTATE_MOST = 1024;
const ROTATE_THREADS = 256;

// the butterflies of a block of the workgroup's values (rotate.block of them): every stage after a barrier, a thread
// a pair or more
const BUTTERFLIES = /* wgsl */ `
  for (var span = 1u; span < shape.block; span *= 2u) {
    workgroupBarrier();
    for (var p = lid; p < shape.block / 2u; p += ${ROTATE_THREADS}u) {
      let i = (p / span) * 2u * span + p % span;
      let a = held[i];
      let b = held[i + span];
      held[i] = a + b;
      held[i + span] = a - b;
    }
  }
  workgroupBarrier();`;

// ROTATE_QUANTIZE: R x of every block of a token's n values, quantized as QUANTIZE quantizes x. Dispatched (n / block,
// tokens): a workgroup a block of a token. x holds the tokens xStride floats apart, xq the same bytes apart and xs the
// scales xStride / 32 floats apart (QUANTIZE's layout); signs are the width's n (Python's, each times 1 / sqrt(block))
export const ROTATE_QUANTIZE = /* wgsl */ `
struct Shape { n: u32, xStride: u32, block: u32, unused: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xs: array<u32>;       // a scale's bits (scale_word, T241)
@group(0) @binding(3) var<storage, read> signs: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
var<workgroup> held: array<f32, ${ROTATE_MOST}>;
${PACKED_WORD}
${SCALE_WORD}
@compute @workgroup_size(${ROTATE_THREADS})
fn main(@builtin(workgroup_id) block: vec3u, @builtin(local_invocation_index) lid: u32) {
  let first = block.x * shape.block;
  let token = block.y;
  let at = token * shape.xStride + first;
  for (var i = lid; i < shape.block; i += ${ROTATE_THREADS}u) {
    held[i] = x[at + i] * signs[first + i];
  }
  ${BUTTERFLIES}
  // a thread a group of 32 of the block (at most 32 of them)
  if (lid < shape.block / 32u) {
    let base = lid * 32u;
    var magnitude = 0u;
    for (var k = 0u; k < 32u; k++) {
      magnitude = max(magnitude, bitcast<u32>(held[base + k]) & FLOAT_MAGNITUDE);
    }
    let scale = bitcast<f32>(magnitude) / 127.0;
    let inverse = select(0.0, 1.0 / scale, scale > 0.0);
    let word = (at + base) / 4u;
    for (var k = 0u; k < 8u; k++) {
      let v = vec4<f32>(held[base + 4u * k], held[base + 4u * k + 1u], held[base + 4u * k + 2u], held[base + 4u * k + 3u]);
      xq[word + k] = packed(clamp(vec4<i32>(round(v * inverse)), vec4<i32>(-127), vec4<i32>(127)));
    }
    xs[token * (shape.xStride / 32u) + (first + base) / 32u] = scale_word(magnitude, scale);
  }
}`;

// UNROTATE: R^-1 z of every block of a token's n values, in place: z times |signs|, the butterflies, then the signs'
// sign bit on the result (the CPU's unrotate). Dispatched (n / block, tokens), x the tokens xStride floats apart
export const UNROTATE = /* wgsl */ `
struct Shape { n: u32, xStride: u32, block: u32, unused: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> signs: array<f32>;
@group(0) @binding(2) var<uniform> shape: Shape;
var<workgroup> held: array<f32, ${ROTATE_MOST}>;
@compute @workgroup_size(${ROTATE_THREADS})
fn main(@builtin(workgroup_id) block: vec3u, @builtin(local_invocation_index) lid: u32) {
  let first = block.x * shape.block;
  let at = block.y * shape.xStride + first;
  for (var i = lid; i < shape.block; i += ${ROTATE_THREADS}u) {
    held[i] = x[at + i] * abs(signs[first + i]);
  }
  ${BUTTERFLIES}
  for (var i = lid; i < shape.block; i += ${ROTATE_THREADS}u) {
    x[at + i] = bitcast<f32>(bitcast<u32>(held[i]) ^ (bitcast<u32>(signs[first + i]) & 0x80000000u));
  }
}`;

// the same two in JavaScript, in float32 (the check of a token's first layer against them, gpu/tokenforms.ts): the
// CPU's kernels, operation by operation
function butterfliesLikeCpu(y: Float32Array, block: number) {
  for (let b = 0; b < y.length; b += block) {
    for (let half = 1; half < block; half *= 2) {
      for (let i = b; i < b + block; i += 2 * half) {
        for (let j = i; j < i + half; j++) {
          const a = y[j], c = y[j + half];
          y[j] = a + c;
          y[j + half] = a - c;
        }
      }
    }
  }
  return y;
}
export const rotateLikeCpu = (x: ArrayLike<number>, signs: ArrayLike<number>, block: number) =>
  butterfliesLikeCpu(Float32Array.from(x, (value, i) => Math.fround(value * signs[i])), block);
export function unrotateLikeCpu(z: ArrayLike<number>, signs: ArrayLike<number>, block: number) {
  const y = butterfliesLikeCpu(Float32Array.from(z, (value, i) => Math.fround(value * Math.abs(signs[i]))), block);
  return y.map((value, i) => (signs[i] < 0 || Object.is(signs[i], -0) ? -value : value));
}
// the workgroups of either over n values of count tokens (a workgroup a block of a token)
export const rotateGroups = (n: number, block: number, count: number): [number, number] => [n / block, count];
