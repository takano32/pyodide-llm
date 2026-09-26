// shaders.js (T135): the WGSL of this project in one place. The GPU section of /benchmark/ (public/benchmark/gpu.js,
// T134) measures with some of them; the model's GPU worker (public/gpu.js) runs a prompt's tokens through the layers
// with the batched matrix and the steps of a layer below it. A plain ES module: both import it with the ?v= of their
// own URL (GitHub Pages keeps a file for ten minutes: all must come from the same deployment).
//
// The weights are this project's int8: values in groups of GROUP with one float32 scale each (llama2_numpy's layout),
// four values to a u32 as the shaders read them. Every matrix times vector: one workgroup of 64 per row, each thread a
// word (4 weights) at a time with the stride of the workgroup, so that neighbours read neighbouring words; the partial
// sums add up in the workgroup's memory. Rows past the most workgroups of a dimension go to its second one.

export const GROUP = 32;
// the tokens a workgroup of the batched matrix multiplies at once: each weight is read once for all of them
export const TILE = 8;

// the tokens of a request and the position of the first, written once per request: every step of a layer reads it
const STEP = /* wgsl */ `struct Step { tokens: u32, pos: u32, unused0: u32, unused1: u32 }`;

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

// ---- the steps of a layer besides its matrices, for the tokens of a prompt (the model's GPU worker). Each does for
// every token what the CPU's kernel of the same name (kernels/kernel.ts) does for one, in float32.

// RMSNorm: one workgroup per token. out = weight * (x / sqrt(mean(x²) + eps)); weight: this layer's, from float at
export const RMSNORM = /* wgsl */ `
struct Norm { size: u32, at: u32, eps: f32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> norm: Norm;
@group(0) @binding(4) var<uniform> step: Step;
var<workgroup> partial: array<f32, 64>;
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
  for (var i = t; i < norm.size; i += 64u) { out[row + i] = weight[norm.at + i] * (s * x[row + i]); }
}`;

// RoPE on q and k, and the keys and values of every token into this layer's cache at its position (step.pos + the
// token): one workgroup per token. Pairs of neighbours turn (llama2.c's order), the first turned of every head (all
// of it but for GPT-NeoX); angles holds, per token, the cos of its headSize / 2 angles and then their sin.
export const ROPE = /* wgsl */ `
struct Rope { heads: u32, kvHeads: u32, headSize: u32, turned: u32 }
${STEP}
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> keys: array<f32>;
@group(0) @binding(4) var<storage, read_write> values: array<f32>;
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
  // k turned on its way into the cache (the first of a pair from both, the second from both), v as it is
  let kvDim = rope.kvHeads * size;
  let row = (step.pos + token) * kvDim;
  for (var j = t; j < kvDim; j += 64u) {
    let at = token * kvDim + j;
    var key = k[at];
    let inHead = j % size;
    if (inHead < rope.turned) {
      let c = angles[angle + inHead / 2u];
      let s = angles[angle + half + inHead / 2u];
      if (inHead % 2u == 0u) { key = k[at] * c - k[at + 1u] * s; } else { key = k[at - 1u] * s + k[at] * c; }
    }
    keys[row + j] = key;
    values[row + j] = v[at];
  }
}`;

// Attention: one workgroup per head and token. The token at position step.pos + token sees the positions up to its own
// (causal: none after it, also where the cache holds the later tokens of the same request). The scores go to scores
// ([tokens][heads][positions]), then their softmax, then the weighted sum of the values into out. Grouped-query
// attention: heads / kvHeads query heads share a head of keys and values.
export const ATTENTION = /* wgsl */ `
struct Attention { heads: u32, kvHeads: u32, headSize: u32, positions: u32, scale: f32, unused0: u32, unused1: u32, unused2: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> attention: Attention;
@group(0) @binding(6) var<uniform> step: Step;
var<workgroup> head: array<f32, 256>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) t: u32) {
  let h = id.x;
  let token = id.y;
  if (token >= step.tokens) { return; }
  let size = attention.headSize;
  let kvDim = attention.kvHeads * size;
  let kv = (h / (attention.heads / attention.kvHeads)) * size;
  let at = token * attention.heads * size + h * size;
  let seen = step.pos + token + 1u;
  let first = (token * attention.heads + h) * attention.positions;
  for (var j = t; j < size; j += 64u) { head[j] = q[at + j]; }
  workgroupBarrier();
  // 1. the scores, and the largest
  var largest = -3.4e38;
  for (var p = t; p < seen; p += 64u) {
    var dot = 0.0;
    for (var j = 0u; j < size; j++) { dot += head[j] * keys[p * kvDim + kv + j]; }
    let score = dot * attention.scale;
    scores[first + p] = score;
    largest = max(largest, score);
  }
  partial[t] = largest;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] = max(partial[t], partial[t + half]); }
    workgroupBarrier();
  }
  let most = partial[0];
  workgroupBarrier();
  // 2. the exponentials and their sum
  var sum = 0.0;
  for (var p = t; p < seen; p += 64u) {
    let e = exp(scores[first + p] - most);
    scores[first + p] = e;
    sum += e;
  }
  partial[t] = sum;
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) { partial[t] += partial[t + half]; }
    workgroupBarrier();
  }
  let inverse = 1.0 / partial[0];
  // every thread reads every position's weight now, written by the others
  storageBarrier();
  // 3. the weighted sum of the values, a value of the head per thread
  for (var d = t; d < size; d += 64u) {
    var total = 0.0;
    for (var p = 0u; p < seen; p++) { total += scores[first + p] * values[p * kvDim + kv + d]; }
    out[at + d] = total * inverse;
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
