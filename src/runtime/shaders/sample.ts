// shaders/sample.ts (T.351): the sampling of a generated token on the GPU (T.151's SAMPLE: the repetition penalty,
// the softmax, top-p by pivots and the draw, T.219's refusal of logits that are not finite), and what it shares with the
// sampling in chunks (stages.ts).
// A part of src/runtime/shaders.ts, which is the window: everything outside src/runtime/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part takes its neighbours' names by static imports (T.367.3:
// a bundler links them).
//
// The notices of what SAMPLER_COMMON, SAMPLER_DRAW and SAMPLE are adapted from (the forms and what was changed are described at the head of
// run.ts, above REPETITION_WINDOW, where these notices stand too; they are copied here, as this file holds the lines):
//
// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/argmax.wgsl, soft_max.wgsl, cumsum.wgsl and get_rows.wgsl
// (https://github.com/ggml-org/llama.cpp, commit 95887577, 2026-09-26), under the MIT License:
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
//
// Adapted from MLC LLM, python/mlc_llm/op/top_p_pivot.py and python/mlc_llm/compiler_pass/attach_logit_processor.py
// (and, for T.191's sampling in chunks in stages.ts, python/mlc_llm/compiler_pass/attach_softmax_with_temperature.py)
// (https://github.com/mlc-ai/mlc-llm, commit 9fa644f5, 2026-08-17), and from the sampling of WebLLM
// (https://github.com/mlc-ai/web-llm, src/llm_chat.ts) and Apache TVM (python/tvm/relax/frontend/nn/op.py,
// https://github.com/apache/tvm, commit e0ed4aad), under the Apache License, Version 2.0. Changed as described above.
/*! Copyright (c) 2023-2025 by MLC LLM Contributors (MLC LLM's NOTICE)
 * Copyright 2019-2023 The Apache Software Foundation (Apache TVM's NOTICE)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */
import { REPETITION_WINDOW, STATE, SAMPLING } from "./run.ts";

// What SAMPLE and the stages of the sampling in chunks (T.191, stages.ts) share: the constants, the workgroup's memory of
// the reductions, and cumsum.wgsl's scan.
// T.219: is_nan_magnitude() below takes the form of isnan() in TensorFlow.js, tfjs-backend-webgpu/src/webgpu_program.ts
// (https://github.com/tensorflow/tfjs, master of 2026-09-28, the file's last commit d45c6af3 of 2023-07-17, which has
// no NOTICE file: `(floatToUint & 0x7fffffffu) > 0x7f800000u` of the bitcast<u32> of the value; the review of T.219
// read it there, and its header is this one's).
/*! Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */
// Changed: +inf counts too (TensorFlow.js's isinf compares the value with a uniform INFINITY in floating point, which
// an implementation may fold: not taken); how the flag travels (the State's word, the chunks' partial results) is
// this project's.
const SAMPLER_COMMON = /* wgsl */ `
${STATE}
${SAMPLING}
const WG_SIZE = 256u;
const WINDOW = ${REPETITION_WINDOW}u;
// MLC's num_pivots: three pivots a round, the fourth lane of the sums unused
const PIVOTS = 3u;
// the bits of 1.0: the most probable token's probability is exp(0), and none is larger
const ONE_BITS = 0x3F800000u;
// the floor of the nucleus, the CPU's sample kernel's: ln 1e7
const NUCLEUS_FLOOR = 16.118095;
// the largest float below 1: a random number of 1.0 (a float64 just under 1 rounded) would pass every sum
const BELOW_ONE = 0.99999994;
const NONE = 0xFFFFFFFFu;

var<workgroup> best_value: array<f32, WG_SIZE>;
var<workgroup> best_index: array<u32, WG_SIZE>;
var<workgroup> shared_sum: array<f32, WG_SIZE>;
var<workgroup> uniform_word: u32;
var<workgroup> any_word: atomic<u32>;

// T219: the logits the engine draws no token from (T195's rule: the largest logit is no finite number), by their bits.
// WGSL lets an implementation assume that no NaN nor infinity occurs (§15.7), so 'x != x' and comparisons with an
// infinity may be folded away (lavapipe folds 'x != x': the broken copy of T219); the bits of a u32 are not. isnan's
// form is TensorFlow.js's: the magnitude (the bits without the sign) over 0x7f800000, the exponent all ones and a
// fraction. Here over the largest magnitude a thread saw (an AND and a max a logit, no branch): it is over 0x7f800000
// iff some logit's is. +inf is caught apart, as the largest logit (its bits INFINITY_BITS); -inf (the sign set) is
// left alone, as the CPU's sampler leaves it (a token it never draws is no error)
const FLOAT_MAGNITUDE = 0x7fffffffu;
const INFINITY_BITS = 0x7f800000u;
fn is_nan_magnitude(largest_magnitude: u32) -> bool {
    return largest_magnitude > INFINITY_BITS;
}
// whether any thread's flag is set, to every thread as a uniform value (a barrier; through workgroupUniformLoad, so
// that a branch on it may hold the barriers of the reductions after: an atomicLoad's value is not uniform to WGSL)
fn any_of(flag: bool, t: u32) -> bool {
    if (t == 0u) {
        atomicStore(&any_word, 0u);
    }
    workgroupBarrier();
    if (flag) {
        atomicStore(&any_word, 1u);
    }
    workgroupBarrier();
    if (t == 0u) {
        uniform_word = atomicLoad(&any_word);
    }
    return workgroupUniformLoad(&uniform_word) != 0u;
}

// cumsum.wgsl's scan: (the sum of the values of the threads before t, the sum of all), in the same order every run
fn scan(value: f32, t: u32) -> vec2<f32> {
    shared_sum[t] = value;
    workgroupBarrier();
    // upsweep
    var offset = 1u;
    while (offset < WG_SIZE) {
        let idx = (t + 1u) * offset * 2u - 1u;
        if (idx < WG_SIZE) {
            shared_sum[idx] = shared_sum[idx] + shared_sum[idx - offset];
        }
        workgroupBarrier();
        offset <<= 1u;
    }
    let all = shared_sum[WG_SIZE - 1u];
    workgroupBarrier();
    // set last to 0 for exclusive sum
    if (t == 0u) {
        shared_sum[WG_SIZE - 1u] = 0.0;
    }
    workgroupBarrier();
    // downsweep
    offset = WG_SIZE >> 1u;
    while (offset > 0u) {
        let idx = (t + 1u) * offset * 2u - 1u;
        if (idx < WG_SIZE) {
            let x = shared_sum[idx - offset];
            shared_sum[idx - offset] = shared_sum[idx];
            shared_sum[idx] = shared_sum[idx] + x;
        }
        workgroupBarrier();
        offset >>= 1u;
    }
    let before = shared_sum[t];
    workgroupBarrier();
    return vec2<f32>(before, all);
}

// argmax.wgsl's pairs, reduced in the workgroup's memory: the largest value and its first index (of two equal, the
// smaller index)
struct Best { value: f32, at: u32 }
fn best_of(value: f32, at: u32, t: u32) -> Best {
    best_value[t] = value;
    best_index[t] = at;
    workgroupBarrier();
    var offset = WG_SIZE / 2u;
    while (offset > 0u) {
        if (t < offset) {
            let b = best_value[t + offset];
            let bi = best_index[t + offset];
            if (b > best_value[t] || (b == best_value[t] && bi < best_index[t])) {
                best_value[t] = b;
                best_index[t] = bi;
            }
        }
        workgroupBarrier();
        offset >>= 1u;
    }
    let best = Best(best_value[0], best_index[0]);
    workgroupBarrier();
    return best;
}
`;

// The nucleus and the draw from the probabilities gathered over the floor (probs[0 .. count), their tokens in order[]
// in the order of their index), and the token into chosen and the state: SAMPLE's and the chunks' last stage's. The
// shader that takes it binds probs, order, state (read_write), chosen and settings
const SAMPLER_DRAW = /* wgsl */ `
var<workgroup> shared_sums: array<vec4<f32>, WG_SIZE>;
// the search's bounds, in bits, and the sum at lo: thread 0 writes them, every thread reads them by one
// workgroupUniformLoad a round (a barrier, and the value uniform for the loop's own barriers)
struct Bound { lo: u32, hi: u32, sum: f32 }
var<workgroup> bound: Bound;
var<workgroup> found: atomic<u32>;

// soft_max.wgsl's tree, four sums at once (a pivot's each)
fn total4(value: vec4<f32>, t: u32) -> vec4<f32> {
    shared_sums[t] = value;
    workgroupBarrier();
    var offset = WG_SIZE / 2u;
    while (offset > 0u) {
        if (t < offset) {
            shared_sums[t] += shared_sums[t + offset];
        }
        offset = offset / 2u;
        workgroupBarrier();
    }
    let sums = shared_sums[0];
    workgroupBarrier();
    return sums;
}

// top_p_pivot's search, over the bits of the probabilities gathered (probs[0 .. count)): from lo, whose sum S(lo)
// (the sum of the probabilities at or over it) is lo_sum and passes, to hi, which does not, the largest bits whose
// sum passes: at or over limit (the nucleus: strict false), or over it (the draw: strict true). S only changes at a
// probability that is present, so the largest passing bits are one (and the draw's is where the CPU's walk stops:
// S(w) counts every token the sorted walk passes before w's last, and S of the next larger probability is what it
// passed before w's first). Each round narrows (lo, hi) to a quarter or so: 15 rounds from bits 0 to 1.0, 10 to 14
// for the draw from the nucleus's probability, each a read of the count gathered. Returns (those bits, that sum).
// T546: counting, S counts the tokens at or over a probability instead of adding them up (the k-th largest of a top-k)
fn pivot(lo_start: u32, lo_sum_start: f32, limit: f32, strict: bool, counting: bool, count: u32, t: u32) -> vec2<f32> {
    if (t == 0u) {
        bound = Bound(lo_start, ONE_BITS + 1u, lo_sum_start);
    }
    var b = workgroupUniformLoad(&bound);
    while (b.hi - b.lo > 1u) {
        // three pivots spaced over (lo, hi), in bits: a positive float orders as its bits
        let spacing = max((b.hi - b.lo) / (PIVOTS + 1u), 1u);
        let u = min(vec3<u32>(b.lo + spacing, b.lo + 2u * spacing, b.lo + 3u * spacing), vec3<u32>(b.hi - 1u));
        let f = bitcast<vec3<f32>>(u);
        var acc = vec4<f32>(0.0);
        for (var k = t; k < count; k += WG_SIZE) {
            let p = probs[k];
            acc += select(vec4<f32>(0.0), vec4<f32>(select(p, 1.0, counting)), vec4<bool>(p >= f.x, p >= f.y, p >= f.z, false));
        }
        let sums = total4(acc, t);
        if (t == 0u) {
            var next = b;
            for (var j = 0u; j < PIVOTS; j++) {
                let passes = select(sums[j] >= limit, sums[j] > limit, strict);
                if (passes && u[j] > next.lo) {
                    next.lo = u[j];
                    next.sum = sums[j];
                } else if (!passes && u[j] < next.hi) {
                    next.hi = u[j];
                }
            }
            bound = next;
        }
        b = workgroupUniformLoad(&bound);
    }
    return vec2<f32>(bitcast<f32>(b.lo), b.sum);
}

// the token into chosen and the state: the next pass's input at the next position, into the window (T546: a sampled
// one, the presence penalty's); or the end
fn finish(token: u32) {
    let at = state.sampled;
    chosen[at] = token;
    state.sampled = at + 1u;
    var stop = false;
    for (var j = 0u; j < settings.stops; j++) {
        stop = stop || settings.stop[j / 4u][j % 4u] == token;
    }
    if (stop) {
        state.stopped = 1u;
    } else {
        state.token = token;
        state.pos = state.pos + 1u;
        state.recent[state.history % WINDOW] = token;
        state.history = state.history + 1u;
        state.written = state.written + 1u;
    }
}

// the nucleus of the count gathered (total: the mass of all over the floor), the draw in it for the random number r,
// and the token into chosen and the state (argmax where no token was found). T546: the CPU's order of the samplers
// (kernel sample(), NumPy's drawing()): a top-k first, then the nucleus of what it kept (topp of its mass: the
// probabilities add up to one again), then min-p; the tokens walked from the most probable
fn draw_nucleus(count: u32, total: f32, r: f32, argmax: u32, t: u32) {
    // the probability the tokens kept are at or over (its bits), and their mass
    var lo = 0u;
    var mass = total;
    // T546: a top-k (where it leaves some out): the k-th largest probability, by the pivots counting the tokens at or
    // over each, and of the tokens of that probability the first ones in the order of their index (the CPU's sort
    // takes equal ones in that order: likecpu.ts), keep of them
    var kth = 0.0;
    var keep = NONE;
    if (settings.top_k != 0u && any_of(settings.top_k < count, t)) {
        kth = pivot(0u, f32(count), f32(settings.top_k), false, true, count, t).x;
        var acc = vec4<f32>(0.0);
        for (var k = t; k < count; k += WG_SIZE) {
            let p = probs[k];
            acc += select(vec4<f32>(0.0), vec4<f32>(p, 1.0, 0.0, 0.0), p > kth);
        }
        let over = total4(acc, t);
        keep = settings.top_k - u32(over.y);
        lo = bitcast<u32>(kth);
        mass = over.x + f32(keep) * kth;
    }
    // the nucleus: the smallest probability p whose tokens at or over it hold topp of the mass
    if (settings.topp > 0.0 && settings.topp < 1.0) {
        let nucleus_found = pivot(lo, mass, settings.topp * mass, false, false, count, t);
        lo = bitcast<u32>(nucleus_found.x);
        mass = nucleus_found.y;
    }
    // T546: min-p, what is less than min_p times as probable as the most probable left out (the most probable's
    // probability is exp(0), 1: ONE_BITS), where that is more than the nucleus left out
    let least = min(settings.min_p, 1.0);
    if (settings.min_p > 0.0 && bitcast<u32>(least) > lo) {
        lo = bitcast<u32>(least);
        var acc = 0.0;
        for (var k = t; k < count; k += WG_SIZE) {
            let p = probs[k];
            acc += select(0.0, p, p >= least);
        }
        mass = total4(vec4<f32>(acc, 0.0, 0.0, 0.0), t).x;
    }
    // the draw: the largest probability whose tokens at or over it hold more than r × the mass kept
    let goal = r * mass;
    let drawn = pivot(lo, mass, goal, true, false, count, t);
    let w = drawn.x;

    // the tokens of probability w, the mass over it, and which of the equal ones: the order of their index
    let run = (count + WG_SIZE - 1u) / WG_SIZE;
    let first = min(t * run, count);
    let last = min(first + run, count);
    var equal = 0u;
    var over = 0.0;
    for (var k = first; k < last; k++) {
        let p = probs[k];
        if (p == w) {
            equal++;
        } else if (p > w) {
            over += p;
        }
    }
    let equals = scan(f32(equal), t);
    let above = scan(over, t).y;
    let ties = min(u32(equals.y), select(NONE, keep, w == kth));
    let nth = min(u32(max(floor((goal - above) / w), 0.0)), max(ties, 1u) - 1u);
    var rank = u32(equals.x);
    for (var k = first; k < last; k++) {
        if (probs[k] == w) {
            if (rank == nth) {
                atomicStore(&found, order[k]);
            }
            rank++;
        }
    }
    workgroupBarrier();
    if (t == 0u) {
        let hit = atomicLoad(&found);
        finish(select(hit, argmax, hit == NONE));
    }
}
`;

// (T.391.4) Two loops over a thread's own run of the vocabulary, from index `from` to before `to` (the names the text
// around has for them), which SAMPLE below and the sampling in chunks (stages.ts's SAMPLE_MAX and SAMPLE_SUM) have
// alike: what one finds or sums, the other must, token for token.
// The largest logit of the run and its first index (argmax.wgsl's pairs), into value and at, and (T.219) the largest
// of the logits' bits without their signs into magnitude, by which a NaN among them is seen
const largestLogit = (from: string, to: string): string => /* wgsl */ `    var value = -3.4e38;
    var at = NONE;
    var magnitude = 0u;
    for (var i = ${from}; i < ${to}; i++) {
        let v = logits[i];
        magnitude = max(magnitude, bitcast<u32>(v) & FLOAT_MAGNITUDE);
        if (v > value) {
            value = v;
            at = i;
        }
    }`;
// softmax's exp(value - max) of the run's logits, at the temperature (inverse) and, with a nucleus, only of those
// over the floor (lowest; the others are not written then): how many are kept into kept, their sum into sum, and
// without a nucleus each token's into probs
const softmaxTerms = (from: string, to: string): string => /* wgsl */ `    var sum = 0.0;
    var kept = 0u;
    for (var i = ${from}; i < ${to}; i++) {
        let v = logits[i];
        if (v >= lowest) {
            let p = exp((v - best) * inverse);
            kept++;
            sum += p;
            if (!nucleus) {
                probs[i] = p;
            }
        } else if (!nucleus) {
            probs[i] = 0.0;
        }
    }`;

// The penalty, softmax, top-p and the draw of one token, in one workgroup: see the head of run.ts. Bindings: 0 the logits (the
// penalty is applied to them in place, as the CPU does), 1 and 2 scratch of the vocabulary's size (probabilities and
// the indices of those gathered), 3 the state, 4 chosen, 5 the random numbers, 6 the settings.
// T.546: the presence penalty, a top-k and min-p too (the CPU's: kernels/kernel/sample.ts): see draw_nucleus above and
// the penalty and the draw without a nucleus below
export const SAMPLE = /* wgsl */ `
${SAMPLER_COMMON}
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> order: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: State;
@group(0) @binding(4) var<storage, read_write> chosen: array<u32>;
@group(0) @binding(5) var<storage, read> randoms: array<f32>;
@group(0) @binding(6) var<uniform> settings: Sampling;
${SAMPLER_DRAW}
// T546: the last token min-p keeps, in the order of the index (the draw without a nucleus ends on it where no running
// sum passes)
var<workgroup> last_kept: atomic<u32>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // a run that stopped changes nothing more
    if (t == 0u) {
        uniform_word = state.stopped;
        atomicStore(&found, NONE);
        atomicStore(&last_kept, 0u);
    }
    if (workgroupUniformLoad(&uniform_word) != 0u) {
        return;
    }

    // the repetition penalty (apply_penalty_inplace's thread a token): each distinct token of the window once. T546:
    // then the presence penalty taken off each distinct one of those sampled in this generation (the latest
    // state.written of the history, a slot's age the tokens written after it: the CPU's penalize() over its written
    // tokens, after the repetition penalty's over the history; the prompt's are not counted)
    let window = min(state.history, WINDOW);
    let written = min(state.written, WINDOW);
    if ((settings.penalty != 1.0 || settings.presence != 0.0) && t < window) {
        let token = state.recent[t];
        var seen = false;
        var sampled = false;
        for (var j = 0u; j < window; j++) {
            let same = state.recent[j] == token;
            seen = seen || (same && j < t);
            sampled = sampled || (same && (state.history - 1u - j) % WINDOW < written);
        }
        if (!seen && token < settings.vocab) {
            var value = logits[token];
            if (settings.penalty != 1.0) {
                value = select(value * settings.penalty, value / settings.penalty, value > 0.0);
            }
            if (sampled) {
                value = value - settings.presence;
            }
            logits[token] = value;
        }
    }
    storageBarrier();

    // a run of consecutive tokens a thread (cumsum.wgsl's): the largest logit and its first index (argmax.wgsl's pairs),
    // and (T219) whether any logit is a NaN or +inf, by its bits
    let vocab = settings.vocab;
    let chunk = (vocab + WG_SIZE - 1u) / WG_SIZE;
    let begin = min(t * chunk, vocab);
    let end = min(begin + chunk, vocab);
${largestLogit("begin", "end")}
    let largest = best_of(value, at, t);
    let best = largest.value;
    let argmax = largest.at;

    // T219: the step is refused where the largest logit is no finite number (T195's rule: a NaN or +inf anywhere,
    // or nothing over -3.4e38): the flag into the state, the run stopped, no token written; the CPU takes the step.
    // (argmax's NONE goes in as a flag: a value from the workgroup's memory is not uniform to WGSL's analysis either)
    if (any_of(is_nan_magnitude(magnitude) || bitcast<u32>(best) == INFINITY_BITS || argmax == NONE, t)) {
        if (t == 0u) {
            state.not_finite = 1u;
            state.stopped = 1u;
        }
        return;
    }

    if (settings.temperature == 0.0) {
        if (t == 0u) {
            finish(argmax);
        }
        return;
    }

    // softmax's exp(value - max), at the temperature; with a nucleus only over the floor (the CPU's sample kernel's)
    let nucleus = settings.topp > 0.0 && settings.topp < 1.0;
    let inverse = 1.0 / settings.temperature;
    let lowest = select(-3.4e38, best - settings.temperature * NUCLEUS_FLOOR, nucleus);
${softmaxTerms("begin", "end")}
    let sums = scan(sum, t);
    let total = sums.y;
    let r = min(randoms[state.sampled], BELOW_ONE);

    // T546: a top-k that leaves some out walks the tokens from the most probable, as a nucleus does (the CPU's)
    var walked = nucleus;
    if (!nucleus && settings.top_k != 0u) {
        walked = any_of(settings.top_k < u32(scan(f32(kept), t).y), t);
    }
    if (!walked) {
        // the draw in the order of the index: the first token whose running sum passes r × total, else the last.
        // T546: min-p first leaves out, in that order, what is less than min_p times as probable as the most probable
        // (whose probability is exp(0), 1): the mass and the running sums are of what it keeps
        var goal = r * total;
        var running = sums.x;
        var least = 0.0;
        var last = vocab - 1u;
        if (settings.min_p > 0.0) {
            least = min(settings.min_p, 1.0);
            var stays = 0.0;
            for (var i = begin; i < end; i++) {
                let p = probs[i];
                if (p >= least) {
                    stays += p;
                    atomicMax(&last_kept, i);
                }
            }
            let kept_sums = scan(stays, t);
            goal = r * kept_sums.y;
            running = kept_sums.x;
            last = atomicLoad(&last_kept);
        }
        for (var i = begin; i < end; i++) {
            let p = probs[i];
            if (p >= least) {
                running += p;
                if (running > goal) {
                    atomicMin(&found, i);
                    break;
                }
            }
        }
        workgroupBarrier();
        if (t == 0u) {
            let hit = atomicLoad(&found);
            finish(select(hit, last, hit == NONE));
        }
        return;
    }

    // the tokens over the floor gathered in the order of their index (their counts scanned): their indices into order,
    // then their probabilities into probs, the same exp() of the same logit as above
    let counts = scan(f32(kept), t);
    let count = u32(counts.y);
    var into = u32(counts.x);
    for (var i = begin; i < end; i++) {
        if (logits[i] >= lowest) {
            order[into] = i;
            into++;
        }
    }
    storageBarrier();
    for (var k = t; k < count; k += WG_SIZE) {
        probs[k] = exp((logits[order[k]] - best) * inverse);
    }
    storageBarrier();
    draw_nucleus(count, total, r, argmax, t);
}`;

export { SAMPLER_COMMON, SAMPLER_DRAW, largestLogit, softmaxTerms };
