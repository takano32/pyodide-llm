// shaders/bench.js (T351): the shaders that /benchmark/'s GPU section alone runs, but for T149's matrix × vector
// (matvec.js) and the sampling in chunks (stages.js): T135's first matrix products (WIDEN, PACKED, BATCHED), the argmax,
// the empty and the small dispatch, and T168's ceilings of the device (clpeak's method; no line of it is copied).
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).
const { STEP } = await import(new URL(`common.js${new URL(import.meta.url).search}`, import.meta.url));

// the tokens a workgroup of the batched matrix multiplies at once: each weight is read once for all of them
export const TILE = 8;
// a matrix times one vector, the weights widened to float32 on the way (the benchmark's)
export const WIDEN = /* wgsl */ `
struct Shape { rows: u32, words: u32, perRow: u32, first: u32 }
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let row = id.x + id.y * count.x;
  if (row >= shape.rows) { return; }
  var sum = 0.0;
  for (var i = t; i < shape.words; i += 64u) {
    let word = bitcast<i32>(w[row * shape.words + i]);
    let at = i * 4u;
    let dot = f32(extractBits(word, 0u, 8u)) * x[at] + f32(extractBits(word, 8u, 8u)) * x[at + 1u]
            + f32(extractBits(word, 16u, 8u)) * x[at + 2u] + f32(extractBits(word, 24u, 8u)) * x[at + 3u];
    sum += dot * scales[row * shape.perRow + i / 8u];
  }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  if (t == 0u) { y[shape.first + row] = partial[0]; }
}`;

// the same with the activations quantized to int8 as the CPU's matmul_q8 takes them (a float32 scale per group of
// 32), and WGSL's packed dot product: where the language feature is there (the benchmark's)
export const PACKED = /* wgsl */ `
requires packed_4x8_integer_dot_product;
struct Shape { rows: u32, words: u32, perRow: u32, first: u32 }
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<storage, read> xs: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let row = id.x + id.y * count.x;
  if (row >= shape.rows) { return; }
  var sum = 0.0;
  for (var i = t; i < shape.words; i += 64u) {
    sum += f32(dot4I8Packed(w[row * shape.words + i], xq[i])) * scales[row * shape.perRow + i / 8u] * xs[i / 8u];
  }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  if (t == 0u) { y[shape.first + row] = partial[0]; }
}`;

// A matrix times the tokens of a prompt (step.tokens of them), TILE at a time: one workgroup per row and tile of
// tokens (the third dimension of the dispatch). x holds the tokens xStride floats apart, y their outputs yStride
// apart, from row first on (a matrix cut into chunks of rows). add: the result is added to what y holds (the residual
// stream: x + W·v, as the CPU adds it after its matmul), else it replaces it.
export const BATCHED = /* wgsl */ `
struct Shape { rows: u32, words: u32, perRow: u32, first: u32, xStride: u32, yStride: u32, add: u32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
@group(0) @binding(5) var<uniform> step: Step;
const TILE = ${TILE}u;
var<workgroup> partial: array<f32, ${TILE * 64}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(num_workgroups) count: vec3u, @builtin(local_invocation_index) t: u32) {
  let row = id.x + id.y * count.x;
  if (row >= shape.rows) { return; }
  let first = id.z * TILE;
  var sums: array<f32, ${TILE}>;
  for (var i = t; i < shape.words; i += 64u) {
    let word = bitcast<i32>(w[row * shape.words + i]);
    let scale = scales[row * shape.perRow + i / 8u];
    let w0 = f32(extractBits(word, 0u, 8u)) * scale;
    let w1 = f32(extractBits(word, 8u, 8u)) * scale;
    let w2 = f32(extractBits(word, 16u, 8u)) * scale;
    let w3 = f32(extractBits(word, 24u, 8u)) * scale;
    for (var k = 0u; k < TILE; k++) {
      if (first + k < step.tokens) {
        let at = (first + k) * shape.xStride + i * 4u;
        sums[k] += w0 * x[at] + w1 * x[at + 1u] + w2 * x[at + 2u] + w3 * x[at + 3u];
      }
    }
  }
  for (var k = 0u; k < TILE; k++) { partial[k * 64u + t] = sums[k]; }
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) {
      for (var k = 0u; k < TILE; k++) { partial[k * 64u + t] += partial[k * 64u + t + half]; }
    }
    workgroupBarrier();
  }
  if (t < TILE && first + t < step.tokens) {
    let at = (first + t) * shape.yStride + shape.first + row;
    y[at] = select(0.0, y[at], shape.add != 0u) + partial[t * 64u];
  }
}`;

// the most likely token: the first index of the largest logit, in one workgroup, so that only 4 bytes come back
export const ARGMAX = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> chosen: array<u32>;
@group(0) @binding(2) var<uniform> count: vec4u;
var<workgroup> best: array<f32, 256>;
var<workgroup> index: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
  var value = -3.4e38;
  var at = 0u;
  for (var i = t; i < count.x; i += 256u) {
    if (logits[i] > value) { value = logits[i]; at = i; }
  }
  best[t] = value;
  index[t] = at;
  workgroupBarrier();
  for (var half = 128u; half > 0u; half >>= 1u) {
    if (t < half && (best[t + half] > best[t] || (best[t + half] == best[t] && index[t + half] < index[t]))) {
      best[t] = best[t + half];
      index[t] = index[t + half];
    }
    workgroupBarrier();
  }
  if (t == 0u) { chosen[0] = index[0]; }
}`;

// a dispatch that does nothing: what a dispatch costs by itself (the benchmark's)
export const EMPTY = /* wgsl */ `@compute @workgroup_size(1) fn main() {}`;

// the benchmark's stand-in for the small steps of a layer (norms, RoPE, a short attention, SwiGLU, the residual adds):
// what they cost is mostly that they are dispatches of their own, so one that adds a vector stands for each
export const SMALL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = arrayLength(&x);
  for (var i = id.x; i < n; i += 64u) { y[i] = y[i] + x[i]; }
}`;

// ---- T168: the device's ceilings, for the GPU section of /benchmark/ to say what share of them the prompt's shaders
// reach: the multiply-adds of f32 and f16 (GFLOPS), WGSL's dot4I8Packed (GOPS: a device that emulates it shows it
// here), and reading the workgroup's memory and a storage buffer (GB/s). The shapes are clpeak's
// (https://github.com/krrishnarraj/clpeak, src/opencl/kernels/: compute_sp, compute_int8_dp, global_bandwidth), taken
// as a method only and written anew here: clpeak is under the GPL-3.0, and no line of it is copied. What the method
// keeps a compiler from making the loops cheaper than they look:
//   - the multiply-adds in two shapes, as clpeak races two (mad_chain.cl) and keeps the faster, since no one shape is
//     the fastest on every device: "square", the recurrence x = x·x + c (c differs by lane), which no algebra folds,
//     two chains of vec4 (8 independent multiply-adds a thread); and "affine", x = x·a + b with a and b the same for
//     every lane (from the uniform), one chain of vec4 (clpeak: "N is 1 from width 4 up": the vector is the
//     parallelism), whose three operands are distinct registers (clpeak: Intel's GPUs halve a multiply-add that reads
//     one register twice, as x·x does). Floating point does not reassociate, so the affine chain is not folded either.
//     c is in [-1.55, -1], where the square stays in [-2, 2] (the real axis of the Mandelbrot set), and a = 0.999,
//     b = 0.001 draw the affine one to 1: no infinities and no subnormals to time. Both do FMA_PER_LOOP a loop;
//   - a dot4I8Packed chain is two accumulators feeding each other, a = dot(x, b) + a, b = dot(x, a) + b (clpeak's
//     compute_int8_dp: with both operands loop-invariant a compiler may turn the chain into one multiply); four pairs;
//     WGSL's dot has no accumulating form, so the add is part of each dot here as in the DP4A shader of the prompt;
//   - every thread writes what its chains or reads came to, and the loop count comes from a uniform;
//   - the workgroup's memory is read at addresses that move with the loop (nothing to hoist out of it), 16 bytes a
//     thread with neighbours on neighbouring vec4s; the storage buffer as clpeak's global_offset kernels read it, each
//     read a dispatch's threads apart, so that neighbours read neighbouring vec4s and the buffer once a dispatch.
// Every ceiling runs CEILING_WORKGROUP threads a workgroup and writes one u32 a thread to out; the uniform (plan)
// holds the loops, a seed and the affine chain's a and b. *_PER_LOOP: what one thread does in one pass of its loop, in FLOPs, ops or bytes
export const CEILING_WORKGROUP = 256;
export const FMA_PER_LOOP = 32 * 4 * 2;
export const DOT4_PER_LOOP = 8 * 4 * 2 * 8;
export const SHARED_PER_LOOP = 16 * 16;
export const GLOBAL_PER_THREAD = 16 * 16;
const CEILING_HEAD = /* wgsl */ `
struct Plan { loops: u32, seed: u32, a: f32, b: f32 }
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> plan: Plan;`;
export const FMA_SHAPES = ["square", "affine"];
export const fmaCeiling = (half, shape) => {
  const T = half ? "f16" : "f32";
  const body = shape === "square"
    ? `  let c = vec4<${T}>(${T}(-1.0 - f32(lane) / 1024.0)) - vec4<${T}>(0.0, 0.1, 0.2, 0.3);
  var x = vec4<${T}>(${T}(f32(plan.seed & 255u) / 256.0));
  var y = x - vec4<${T}>(0.5);
  for (var i = 0u; i < plan.loops; i++) {
${"    x = fma(x, x, c);\n    y = fma(y, y, c);\n".repeat(16)}  }
  out[id.x] = bitcast<u32>(dot(vec4<f32>(x + y), vec4<f32>(1.0)));`
    : `  let a = vec4<${T}>(${T}(plan.a));
  let b = vec4<${T}>(${T}(plan.b));
  var x = vec4<${T}>(${T}(f32(lane) / 256.0)) + vec4<${T}>(0.0, 0.1, 0.2, 0.3);
  for (var i = 0u; i < plan.loops; i++) {
${"    x = fma(x, a, b);\n".repeat(32)}  }
  out[id.x] = bitcast<u32>(dot(vec4<f32>(x), vec4<f32>(1.0)));`;
  return /* wgsl */ `${half ? "enable f16;" : ""}
${CEILING_HEAD}
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
${body}
}`;
};
export const DOT4_CEILING = /* wgsl */ `requires packed_4x8_integer_dot_product;
${CEILING_HEAD}
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let x = plan.seed ^ (id.x * 0x9e3779b9u);
  var a0 = i32(id.x); var a1 = a0 + 1; var a2 = a0 + 2; var a3 = a0 + 3;
  var b0 = i32(x); var b1 = b0 ^ 1; var b2 = b0 ^ 2; var b3 = b0 ^ 3;
  for (var i = 0u; i < plan.loops; i++) {
${[...Array(8)].map(() => [0, 1, 2, 3].map((k) =>
    `    a${k} = dot4I8Packed(x, bitcast<u32>(b${k})) + a${k};\n    b${k} = dot4I8Packed(x, bitcast<u32>(a${k})) + b${k};\n`).join("")).join("")}  }
  out[id.x] = bitcast<u32>(a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3);
}`;
// 16 KiB of the workgroup's memory (every device has that much: WebGPU's default limit), filled once, then read
export const SHARED_CEILING = /* wgsl */ `${CEILING_HEAD}
var<workgroup> held: array<vec4<f32>, 1024>;
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
  for (var j = 0u; j < 4u; j++) { held[lane + j * ${CEILING_WORKGROUP}u] = vec4<f32>(f32(lane + j), f32(plan.seed), 1.0, 2.0); }
  workgroupBarrier();
  var s0 = vec4<f32>(); var s1 = vec4<f32>(); var s2 = vec4<f32>(); var s3 = vec4<f32>();
  for (var i = 0u; i < plan.loops; i++) {
    let at = ((i * 64u) & 511u) + lane;
${[...Array(16)].map((_, k) => `    s${k % 4} += held[at + ${k * 16}u];\n`).join("")}  }
  out[id.x] = bitcast<u32>(dot(s0 + s1 + s2 + s3, vec4<f32>(1.0)));
}`;
// the buffer read once a dispatch (plan.loops unused: its size is the work): 16 vec4s a thread, each a dispatch's threads after the one before
export const GLOBAL_CEILING = /* wgsl */ `${CEILING_HEAD}
@group(0) @binding(2) var<storage, read> data: array<vec4<u32>>;
@compute @workgroup_size(${CEILING_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let apart = groups.x * ${CEILING_WORKGROUP}u;
  var s0 = vec4<u32>(); var s1 = vec4<u32>(); var s2 = vec4<u32>(); var s3 = vec4<u32>();
${[...Array(16)].map((_, k) => `  s${k % 4} += data[id.x + ${k}u * apart];\n`).join("")}  let s = s0 + s1 + s2 + s3;
  out[id.x] = s.x ^ s.y ^ s.z ^ s.w ^ plan.seed;
}`;
