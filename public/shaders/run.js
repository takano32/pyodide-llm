// shaders/run.js (T351): a run of generated tokens on the GPU (T151) but for its sampling (sample.js): the state a
// pass hands the next and the settings of the sampling (with the JavaScript that writes them), the embedding of a token
// or of a prompt's rows, and the outlier channels of a ternary classifier (T232).
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).

// ---- T151: a run of generated tokens on the GPU, the ids read back once for all of them. Each token of the run is
// one compute pass: EMBED (the token's row of the embedding into the residual stream), the layers (fusedMatVec, T150),
// the final norm and the classifier (fusedMatVec, "write"), then SAMPLE (the repetition penalty, softmax, top-p and
// the draw), which writes the token into `chosen` and the state the next pass starts from; between passes the state's
// first four words are copied into the Step uniform that fusedMatVec and flashTile read (copyBufferToBuffer: a
// uniform is not a shader's to write). What the CPU gets back: chosen[0 .. sampled) and the state, after the run.
// For /benchmark/'s GPU section first (T151), and for the model's GPU worker when it generates on the GPU (T152).
//
// What the GPU does is the CPU's sampling (kernels/kernel.ts's penalize() and sample(), the engine's generate()),
// token for token: the same random number (the CPU draws them, in order, one a sampled token: randoms[sampled]) picks
// the same token but where a float32 sum in another order moves a border (likecpu.js's sampleLikeCpu is that CPU in
// JavaScript; tests/smoke.mjs holds it to the kernel, /benchmark/'s check holds SAMPLE to it). Their forms:
//   - the penalty: MLC LLM's apply_penalty_inplace (a thread a token of the window, the logit divided where positive
//     and multiplied where not; mlc_llm/compiler_pass/attach_logit_processor.py, Apache-2.0), with the CPU's window
//     (the latest REPETITION_WINDOW tokens of the history, the prompt and BOS in it) and its once for each distinct
//     token, where MLC counts them (its presence and frequency penalties are not the engine's)
//   - the largest logit and its first index, and softmax: llama.cpp's WebGPU argmax.wgsl (the pairs reduced in the
//     workgroup's memory; here the smaller index of two equal, as NumPy's argmax and bench.js's ARGMAX) and soft_max.wgsl
//     (exp(value - max), summed by the workgroup's tree), with kernel.ts's temperature and its floor of the nucleus
//     (a token under a ten millionth of the best one's probability is left out before exp())
//   - top-p without sorting: MLC LLM's top_p_pivot (mlc_llm/op/top_p_pivot.py, Apache-2.0): pivots between a bound
//     known to keep at least top-p of the mass and one known not to, the sum of the probabilities at or over each
//     pivot added up in one pass, the bounds moved to the pivots, until the smallest probability of the nucleus is
//     found. Changed: the pivots are spaced over the bits of the float (a positive float orders as its bits), so the
//     search ends on that probability itself, where MLC's ends within 1e-7 of it; the probabilities are those left
//     after the floor, gathered in the order of their index (below) where MLC reads the whole vocabulary each round
//   - the draw: where the CPU sorts the nucleus and walks it from the most probable token to the random number times
//     its mass (WebLLM's sample_with_top_p does the same after an argsort on the GPU: TVM's
//     sample_top_p_top_k_from_sorted_prob, Apache-2.0), SAMPLE finds the token by the same pivots: the probability of
//     the token the walk stops at is the largest p whose mass at or over it passes the random number (tokens of equal
//     probability are taken in the order of their index: the CPU's quicksort takes them in no set order). Without a
//     nucleus the CPU walks the tokens in the order of their index, and so does SAMPLE: llama.cpp's WebGPU
//     cumsum.wgsl (a thread a run of consecutive tokens, their sums scanned in the workgroup's memory), the first token
//     whose running sum passes the random number
//   - gathering the tokens over the floor in the order of their index (for the pivots, and the order of equals):
//     cumsum.wgsl's scan again, of the counts
//   - the token's row of the embedding: llama.cpp's WebGPU get_rows.wgsl (copy_elements of Q8_0: each weight times
//     its block's scale), for this project's int8 in groups of 32
// This project's: the state that carries the loop from one token to the next (the input token, the position, the
// window, the stop), and stopping: a token of settings.stop is written into chosen and ends the run (the passes after
// it change nothing: the position and the state stay, so a pass recomputes the same position and writes the same
// cache row), as the CPU's generate() breaks at a stop token.
//
// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/argmax.wgsl, soft_max.wgsl, cumsum.wgsl and get_rows.wgsl
// (https://github.com/ggml-org/llama.cpp, commit 95887577, 2026-09-26), under the MIT License:
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
// Adapted from MLC LLM, python/mlc_llm/op/top_p_pivot.py and python/mlc_llm/compiler_pass/attach_logit_processor.py
// (and, for T191's sampling in chunks in stages.js, python/mlc_llm/compiler_pass/attach_softmax_with_temperature.py)
// (https://github.com/mlc-ai/mlc-llm, commit 9fa644f5, 2026-08-17), and from the sampling of WebLLM
// (https://github.com/mlc-ai/web-llm, src/llm_chat.ts) and Apache TVM (python/tvm/relax/frontend/nn/op.py,
// https://github.com/apache/tvm, commit e0ed4aad), under the Apache License, Version 2.0. Changed as described above.
// Copyright (c) 2023-2025 by MLC LLM Contributors (MLC LLM's NOTICE)
// Copyright 2019-2023 The Apache Software Foundation (Apache TVM's NOTICE)
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// the engine's REPETITION_WINDOW (llama2_numpy.py): the penalty looks at this many of the latest tokens
export const REPETITION_WINDOW = 64;
// the most stop tokens settings hold (the list's models have up to 5, src/models.js)
export const STOPS_MOST = 8;
// State: Step's four words first (copied into the Step uniform after each token), then the loop's own
const STATE = /* wgsl */ `struct State {
  tokens: u32, pos: u32, unused0: u32,
  not_finite: u32, // T219: 1 once a step's logits held a NaN or +inf, or none over -3.4e38 (all -inf): the sampler
                   // refused that step (nothing sampled, stopped set), and the CPU takes it by T195's rule. Word 3,
                   // the last of the four the Step uniform gets a copy of (which reads only tokens and pos)
  token: u32,     // the input of the next pass: EMBED's row
  sampled: u32,   // the tokens sampled so far in this run: the index of the next random number and of chosen[]
  history: u32,   // how long the history is (BOS, the prompt, the sampled tokens): recent[history % WINDOW] is next
  stopped: u32,   // 1 once a stop token was sampled, or a step was refused (not_finite): nothing changes after
  recent: array<u32, ${REPETITION_WINDOW}>,
}`;
/** T219: the State's word that says a step was refused (read back with the ids: gpu.js, forward.js) */
export const STATE_NOT_FINITE = 3;
export const STATE_BYTES = 32 + 4 * REPETITION_WINDOW;
/** T152's review (T160): float32 values into float16 bits, rounded to the nearest (ties to even), as the GPU's own keys
 * and values are (pack2x16float) where it rounds so: a float32 cache's keys and values going up to the GPU. out: a
 * Uint16Array as long as floats. Infinities and NaN stay so, what is past float16's largest becomes an infinity, and
 * what is below half its least subnormal 0. */
const floatWord = new Float32Array(1), floatBits = new Uint32Array(floatWord.buffer);
export function halvesOf(floats, out) {
  for (let i = 0; i < floats.length; i++) {
    floatWord[0] = floats[i];
    const bits = floatBits[0], sign = (bits >>> 16) & 0x8000, exponent = (bits >>> 23) & 0xff, fraction = bits & 0x7fffff;
    let half;
    if (exponent === 0xff) half = 0x7c00 | (fraction ? 0x200 : 0);
    else {
      const e = exponent - 112;  // float16's biased exponent
      if (e >= 0x1f) half = 0x7c00;
      else if (e <= 0) {
        // a subnormal (or 0): the fraction with its leading 1, shifted to units of 2^-24
        const shift = 14 - e;
        if (shift > 24) half = 0;
        else {
          const whole = fraction | 0x800000, rest = whole & ((1 << shift) - 1), middle = 1 << (shift - 1);
          half = whole >>> shift;
          if (rest > middle || (rest === middle && (half & 1))) half += 1;
        }
      } else {
        const rest = fraction & 0x1fff;
        half = (e << 10) | (fraction >>> 13);
        // (a carry into the exponent is right: to the next power of two, or to the infinity at the top)
        if (rest > 0x1000 || (rest === 0x1000 && (half & 1))) half += 1;
      }
    }
    out[i] = sign | half;
  }
  return out;
}
/** The state a run starts from: the token it feeds first, at pos, and the history before it (BOS, the prompt and
 * the tokens sampled before, the fed token last: the penalty's window is its latest REPETITION_WINDOW). length (T152:
 * the engine hands the latest of a longer history over): how long the history is, of which history is the end. */
export function samplingState({ token, pos, history, length = history.length }) {
  const words = new Uint32Array(STATE_BYTES / 4), first = length - history.length;
  words.set([1, pos, 0, 0, token, 0, length, 0]);
  for (let at = Math.max(first, length - REPETITION_WINDOW); at < length; at++) {
    words[8 + (at % REPETITION_WINDOW)] = history[at - first];
  }
  return words;
}
// Sampling, a uniform: the settings of the engine's generate()
const SAMPLING = /* wgsl */ `struct Sampling {
  vocab: u32, stops: u32, unused0: u32, unused1: u32,
  temperature: f32, topp: f32, penalty: f32, unused2: f32,
  stop: array<vec4<u32>, ${STOPS_MOST / 4}>,
}`;
export const SAMPLING_BYTES = 32 + 4 * STOPS_MOST;
/** Sampling's bytes: the engine's settings (temperature 0: the most likely token; top-p outside (0, 1): no nucleus;
 * penalty 1: none) and its stop tokens. */
export function samplingSettings({ vocab, temperature, topp, penalty = 1, stops = [] }) {
  if (stops.length > STOPS_MOST) throw new Error(`more than ${STOPS_MOST} stop tokens`);
  const bytes = new ArrayBuffer(SAMPLING_BYTES), words = new Uint32Array(bytes), values = new Float32Array(bytes);
  words.set([vocab, stops.length]);
  values.set([temperature, topp, penalty], 4);
  words.set(stops, 8);
  return new Uint8Array(bytes);
}

// the token's row of the embedding (int8, groups of 32 with a float32 scale each, 4 to a u32) into the stream x.
// Bindings: 0 the table, 1 its scales, 2 the state (its token), 3 x, 4 the row's width (x of a vec4). T209: a table in
// pieces of rows (past what the device binds: Llama 3.2 3B's 394 MB on the owner's Android, which binds 256 MiB) is
// one dispatch a piece, y its first row and z its rows: the piece that holds the token writes its row, the others
// nothing (z 0: the whole table, as before). llama.cpp's get_rows reads one table; the range is ours (T209)
export const EMBED = /* wgsl */ `
${STATE}
@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> state: State;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> shape: vec4<u32>;

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = shape.x;
    let words = n / 4u;
    let row = state.token - shape.y;
    if (shape.z != 0u && row >= shape.z) {
        return;
    }
    for (var word = lid.x; word < words; word += 256u) {
        let q_packed = table[row * words + word];
        let d = scales[(row * n + word * 4u) / 32u];
        for (var k = 0u; k < 4u; k++) {
            dst[word * 4u + k] = f32(get_byte_i32(q_packed, k)) * d;
        }
    }
}`;
// T210: the rows of a prompt's block from the embedding, for a model on the GPU alone (the CPU holds no embedding
// then): EMBED's loop, a workgroup a token (workgroup_id.y) of ids, into its row of dst. Bindings: 0 the table, 1 its
// scales, 2 the block's ids, 3 x (the block's rows, dense), 4 the row's width and (T209) the piece's first row and rows.
// llama.cpp's get_rows.wgsl reads its row's index from idx and writes a row of dst for each (i_dst1: a row of the
// output an index); the range of a piece is T209's
export const EMBED_ROWS = /* wgsl */ `
@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> shape: vec4<u32>;

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = shape.x;
    let words = n / 4u;
    let row = ids[wid.y] - shape.y;
    if (shape.z != 0u && row >= shape.z) {
        return;
    }
    let out = wid.y * n;
    for (var word = lid.x; word < words; word += 256u) {
        let q_packed = table[row * words + word];
        let d = scales[(row * n + word * 4u) / 32u];
        for (var k = 0u; k < 4u; k++) {
            dst[out + word * 4u + k] = f32(get_byte_i32(q_packed, k)) * d;
        }
    }
}`;
// T232: EMBED and EMBED_ROWS from a table of ternary weights (see prompt.js's TERNARY_PACKED, above dp4a): a word is 16 codes of
// two bits and a scale covers 128 weights; a value is its code less one times the scale, one float32 product, as the
// CPU's embed() has it (public/forward/engine.js's weightAt). The bindings, the shape and the dispatch are theirs (block: EMBED_ROWS',
// a workgroup a token of ids; else EMBED's, the state's token). Their loop is llama.cpp's get_rows, as theirs
const embedTernary = (block) => /* wgsl */ `${block ? "" : STATE}
@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> ${block ? "ids: array<u32>" : "state: State"};
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> shape: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = shape.x;
    let words = n / 16u;
    let row = ${block ? "ids[wid.y]" : "state.token"} - shape.y;
    if (shape.z != 0u && row >= shape.z) {
        return;
    }
    let out = ${block ? "wid.y * n" : "0u"};
    for (var word = lid.x; word < words; word += 256u) {
        let codes = table[row * words + word];
        let d = scales[(row * n + word * 16u) / 128u];
        for (var k = 0u; k < 16u; k++) {
            dst[out + word * 16u + k] = f32(i32((codes >> (2u * k)) & 3u) - 1) * d;
        }
    }
}`;
export const EMBED_TERNARY = embedTernary(false), EMBED_ROWS_TERNARY = embedTernary(true);

// ---- T232: the outlier channels of a ternary classifier's input (T92: a few weights of the final norm are several
// times the others (Ternary Bonsai 1.7B's largest is 5.6 times its median, the 8B's 5.2), and a group of 32 quantized
// to 8 bits with one of them loses the other 31). The CPU takes them out of the normed stream before it quantizes it
// and multiplies their columns of the classifier in float32 (public/forward/engine.js's picked and the kernel add_columns); an int8
// model's GPU multiplies a classifier of floats instead (T226: fusedMatVec), which reads int8 weights. For ternary
// weights the GPU does what the CPU does, in two small dispatches around the classifier's matrix, the columns read
// from the table of codes as it is. No public implementation has this (ONNX Runtime and llama.cpp quantize the input
// as it comes), so it is written apart, with what gpu.js gives it:
//   struct Outliers { count: u32, rows: u32, n: u32, unused: u32, channels: array<vec4<u32>, 2> }
//                        count: the channels (OUTLIERS_MOST at most), channel k at channels[k / 4][k % 4], no two the
//                        same; rows and n: the table's piece (TERNARY_COLUMNS alone)
//   TAKE_OUTLIERS    0 x: the final norm's output, n float32 (read and written); 1 picked: count float32 (written);
//                    2 the Outliers. One workgroup of OUTLIERS_MOST threads, thread k: picked[k] = x[channel k], then
//                    x[channel k] = 0. Dispatched between the norm and QUANTIZE (the norm is apart for such a model).
//   TERNARY_COLUMNS  0 the table's piece (codes, 16 to a u32, a row n / 16 words) and 1 its scales (n / 128 a row);
//                    2 picked; 3 dst: the piece's rows of the logits, float32 (read and written); 4 the Outliers.
//                    A thread a row, row (workgroup_id.y × num_workgroups.x + workgroup_id.x) × 256 +
//                    local_invocation_index, those past rows doing nothing: dst[row] += the sum over k of picked[k] ×
//                    ((the code of weight (row, channel k) less one) × the scale of its group of 128). Dispatched after
//                    the classifier's matrix has written the piece's logits.
// gpu.js checks the two with the classifier, on the model's own table and final norm, against JavaScript's (checkTokens:
// the logits of rows of every piece).
export const OUTLIERS_MOST = 8;
const OUTLIERS = /* wgsl */ `struct Outliers { count: u32, rows: u32, n: u32, unused: u32, channels: array<vec4<u32>, 2> }`;
export const TAKE_OUTLIERS = /* wgsl */ `
${OUTLIERS}
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> picked: array<f32>;
@group(0) @binding(2) var<uniform> outliers: Outliers;

@compute @workgroup_size(${OUTLIERS_MOST})
fn main(@builtin(local_invocation_index) k: u32) {
    if (k >= outliers.count) {
        return;
    }
    let channel = outliers.channels[k / 4u][k % 4u];
    picked[k] = x[channel];
    x[channel] = 0.0;
}`;
export const TERNARY_COLUMNS = /* wgsl */ `
${OUTLIERS}
@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> picked: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> outliers: Outliers;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(num_workgroups) count: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = (id.y * count.x + id.x) * 256u + t;
    if (row >= outliers.rows) {
        return;
    }
    let words = outliers.n / 16u;
    var sum = 0.0;
    for (var k = 0u; k < outliers.count; k++) {
        let channel = outliers.channels[k / 4u][k % 4u];
        let code = (table[row * words + channel / 16u] >> (2u * (channel % 16u))) & 3u;
        sum += picked[k] * (f32(i32(code) - 1) * scales[row * (outliers.n / 128u) + channel / 128u]);
    }
    dst[row] = dst[row] + sum;
}`;
/** the Outliers of channels (OUTLIERS_MOST at most) for a piece of rows by n of the table, as bytes */
export function outliersOf(channels, rows = 0, n = 0) {
  const words = new Uint32Array(4 + OUTLIERS_MOST);
  words.set([channels.length, rows, n, 0]);
  words.set(channels, 4);
  return words;
}

export { STATE, SAMPLING };
