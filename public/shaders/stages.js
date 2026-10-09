// shaders/stages.js (T351): the sampling in chunks of the vocabulary, many workgroups (T191's SAMPLER_STAGES).
// /benchmark/'s GPU section alone runs them.
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The lines are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).
//
// The notices of what the stages are adapted from (the forms and what was changed are described at the head of
// run.js, above REPETITION_WINDOW, where these notices stand too; they are copied here, as this file holds the lines):
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
// (and, for T191's sampling in chunks below, python/mlc_llm/compiler_pass/attach_softmax_with_temperature.py)
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
const { SAMPLER_COMMON, SAMPLER_DRAW } = await import(new URL(`sample.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T191: the sampling in chunks of the vocabulary, many workgroups. SAMPLE's one workgroup reads the vocabulary on
// one of the GPU's cores (the owner's Android: 5.9 ms of 11 a token of the CPU section's model, Llama 3's 128256
// tokens), and each of its threads a run of 501 consecutive logits. Here a workgroup a chunk of SAMPLE_CHUNK tokens,
// four consecutive ones a thread (a subgroup's loads side by side), in four dispatches: SAMPLE_MAX (the penalty on the
// window's tokens in the chunk, the chunk's largest logit and its first index), SAMPLE_SUM (every workgroup reduces
// the chunks' largest to the vocabulary's, then its chunk's softmax sum and how many tokens are over the floor; without
// a nucleus its probabilities into probs), SAMPLE_GATHER (with a nucleus: the chunk's tokens over the floor into
// order and probs, after those of the chunks before it, so in the order of the index as SAMPLE gathers them), and
// SAMPLE_PICK (one workgroup: SAMPLE's nucleus and draw on what was gathered, SAMPLER_DRAW; without a nucleus the draw
// in the order of the index, the chunk first by the chunks' sums, then the token in it). The same random number picks
// the same token as SAMPLE and the CPU (sampleLikeCpu) but where a float32 sum in another order moves a border.
// Their forms:
//   - the chunks, and each workgroup reducing all the chunks' partial results again before its own chunk: MLC LLM's
//     two-stage softmax (mlc_llm/compiler_pass/attach_softmax_with_temperature.py, Apache-2.0, above: chunk_lse, the
//     max and sum of each chunk of 4096, and softmax_with_chunked_sum, which merges the chunks' in every block), here
//     with the max of the vocabulary merged before the sums (SAMPLE's exp(value - max) and floor, whose max is the
//     vocabulary's) where MLC merges the sums' log-sum-exp
//   - the largest and its first index: argmax.wgsl's pairs, over a chunk and over the chunks; the sums, the counts and
//     the order of the index: soft_max.wgsl's tree and cumsum.wgsl's scan, as SAMPLE's
// This project's: where a chunk's tokens go in order[] (the tokens over the floor of the chunks before it, summed by
// every workgroup), the penalty applied by the workgroup whose chunk holds the token, and the draw without a nucleus in
// two steps (the chunk whose running sum passes, then the token in it; where the rounding of the two scans leaves none
// passing in that chunk, its last token). What one stage computes and a later one needs to agree with is handed on in
// the partial results, never computed again in the later pipeline (the floor: SAMPLE_GATHER's comment).
// Bindings, of the same numbers as SAMPLE's (each stage binds those it reads: samplerStages): 0 the logits, 1 probs, 2
// order, 3 the state, 4 chosen, 5 the random numbers, 6 the settings, and 7 the chunks' partial results
// (samplePartsBytes: CHUNKS_COMMON's counted() and flagged()).
// T219: SAMPLE_MAX (the one stage that reads every logit whatever the settings) flags each chunk that holds a NaN in
// the partial results, and SAMPLE_PICK, which alone writes the state, folds the chunks' flags (and sees +inf as the
// largest logit) and refuses the
// step as SAMPLE does (the stages between run on such logits harmlessly: a NaN is under any floor, +inf over it, and
// what they count and gather stays within the vocabulary).
export const SAMPLE_CHUNK = 1024;
export const sampleChunks = (vocab) => Math.ceil(vocab / SAMPLE_CHUNK);
export const samplePartsBytes = (vocab) => 8 * (1 + 3 * sampleChunks(vocab));
const CHUNKS_COMMON = /* wgsl */ `
${SAMPLER_COMMON}
const CHUNK = ${SAMPLE_CHUNK}u;
const EACH = CHUNK / WG_SIZE;

// a run that stopped changes nothing more (the state's word read by all: a barrier, and uniform)
fn stopped(t: u32) -> bool {
    if (t == 0u) {
        uniform_word = state.stopped;
    }
    return workgroupUniformLoad(&uniform_word) != 0u;
}

fn chunk_count() -> u32 {
    return (settings.vocab + CHUNK - 1u) / CHUNK;
}
// the partial results: [0] the vocabulary's largest logit and the floor (SAMPLE_SUM's), [1 + j] chunk j's largest logit
// and its index (SAMPLE_MAX's), [1 + chunks + j] how many of its tokens are over the floor and their sum (SAMPLE_SUM's),
// [1 + 2 × chunks + j] whether chunk j holds a NaN (T219, SAMPLE_MAX's; .y unused)
fn counted(j: u32) -> vec2<u32> {
    return parts[1u + chunk_count() + j];
}
fn flagged(j: u32) -> bool {
    return parts[1u + 2u * chunk_count() + j].x != 0u;
}

// the vocabulary's largest logit and its first index, from the chunks' (SAMPLE_MAX's)
fn largest(t: u32) -> Best {
    let chunks = chunk_count();
    var value = -3.4e38;
    var at = NONE;
    for (var j = t; j < chunks; j += WG_SIZE) {
        let part = parts[1u + j];
        let v = bitcast<f32>(part.x);
        if (v > value) {
            value = v;
            at = part.y;
        }
    }
    return best_of(value, at, t);
}

// soft_max.wgsl's tree: the sum of all the threads' values
fn sum_all(value: f32, t: u32) -> f32 {
    shared_sum[t] = value;
    workgroupBarrier();
    var offset = WG_SIZE / 2u;
    while (offset > 0u) {
        if (t < offset) {
            shared_sum[t] += shared_sum[t + offset];
        }
        offset = offset / 2u;
        workgroupBarrier();
    }
    let all = shared_sum[0];
    workgroupBarrier();
    return all;
}
`;
const CHUNK_MAIN = "@compute @workgroup_size(WG_SIZE)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>)";
export const SAMPLE_MAX = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<storage, read> state: State;
@group(0) @binding(6) var<uniform> settings: Sampling;
@group(0) @binding(7) var<storage, read_write> parts: array<vec2<u32>>;
${CHUNKS_COMMON}
${CHUNK_MAIN} {
    let t = lid.x;
    if (stopped(t)) {
        return;
    }
    let begin = wid.x * CHUNK;
    let end = min(begin + CHUNK, settings.vocab);
    // the repetition penalty (apply_penalty_inplace's thread a token) on the window's tokens in this chunk: each
    // distinct token once, by the one workgroup whose chunk holds it
    let window = min(state.history, WINDOW);
    if (settings.penalty != 1.0 && t < window) {
        let token = state.recent[t];
        if (token >= begin && token < end) {
            var seen = false;
            for (var j = 0u; j < t; j++) {
                seen = seen || state.recent[j] == token;
            }
            if (!seen) {
                let value = logits[token];
                logits[token] = select(value * settings.penalty, value / settings.penalty, value > 0.0);
            }
        }
    }
    storageBarrier();
    let first = min(begin + t * EACH, end);
    let last = min(first + EACH, end);
    var value = -3.4e38;
    var at = NONE;
    var magnitude = 0u;
    for (var i = first; i < last; i++) {
        let v = logits[i];
        magnitude = max(magnitude, bitcast<u32>(v) & FLOAT_MAGNITUDE);
        if (v > value) {
            value = v;
            at = i;
        }
    }
    let best = best_of(value, at, t);
    // (T219) and whether the chunk holds a NaN, for SAMPLE_PICK (+inf shows in the largest logit there)
    let chunk_bad = any_of(is_nan_magnitude(magnitude), t);
    if (t == 0u) {
        parts[1u + wid.x] = vec2<u32>(bitcast<u32>(best.value), best.at);
        parts[1u + 2u * chunk_count() + wid.x] = vec2<u32>(u32(chunk_bad), 0u);
    }
}`;
export const SAMPLE_SUM = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(3) var<storage, read> state: State;
@group(0) @binding(6) var<uniform> settings: Sampling;
@group(0) @binding(7) var<storage, read_write> parts: array<vec2<u32>>;
${CHUNKS_COMMON}
${CHUNK_MAIN} {
    let t = lid.x;
    if (stopped(t)) {
        return;
    }
    if (settings.temperature == 0.0) {
        return;
    }
    let best = largest(t).value;
    // SAMPLE's softmax: exp(value - max), at the temperature; with a nucleus only over the floor (kernel.ts)
    let nucleus = settings.topp > 0.0 && settings.topp < 1.0;
    let inverse = 1.0 / settings.temperature;
    let lowest = select(-3.4e38, best - settings.temperature * NUCLEUS_FLOOR, nucleus);
    let begin = wid.x * CHUNK;
    let end = min(begin + CHUNK, settings.vocab);
    let first = min(begin + t * EACH, end);
    let last = min(first + EACH, end);
    var sum = 0.0;
    var kept = 0u;
    for (var i = first; i < last; i++) {
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
    }
    // the chunk's sum by the scan SAMPLE_PICK sums its probabilities with, in the same order
    let chunk_sum = scan(sum, t).y;
    let chunk_kept = sum_all(f32(kept), t);
    if (t == 0u) {
        parts[1u + chunk_count() + wid.x] = vec2<u32>(u32(chunk_kept), bitcast<u32>(chunk_sum));
        if (wid.x == 0u) {
            parts[0] = vec2<u32>(bitcast<u32>(best), bitcast<u32>(lowest));
        }
    }
}`;
export const SAMPLE_GATHER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> order: array<u32>;
@group(0) @binding(3) var<storage, read> state: State;
@group(0) @binding(6) var<uniform> settings: Sampling;
@group(0) @binding(7) var<storage, read> parts: array<vec2<u32>>;
${CHUNKS_COMMON}
${CHUNK_MAIN} {
    let t = lid.x;
    if (stopped(t)) {
        return;
    }
    let nucleus = settings.topp > 0.0 && settings.topp < 1.0;
    if (settings.temperature == 0.0 || !nucleus) {
        return;
    }
    // the largest and the floor as SAMPLE_SUM counted the chunks by: read, never computed here again. The same
    // expression in another pipeline may round 1 ulp apart (WGSL lets an implementation fuse the multiply into the
    // subtraction), and a token on that ulp would be counted by SUM and not gathered here, or the other way: the
    // chunks' places in order[] and probs[] would overlap, or keep a token of the draw before. No check sees it (lavapipe
    // rounds both pipelines alike: Fable's broken copy that computes it again passed), so this read is what keeps it
    let head = parts[0];
    let best = bitcast<f32>(head.x);
    let lowest = bitcast<f32>(head.y);
    let inverse = 1.0 / settings.temperature;
    // where this chunk's go: after the tokens over the floor of the chunks before it
    var before = 0.0;
    for (var j = t; j < wid.x; j += WG_SIZE) {
        before += f32(counted(j).x);
    }
    let offset = u32(sum_all(before, t));
    let begin = wid.x * CHUNK;
    let end = min(begin + CHUNK, settings.vocab);
    let first = min(begin + t * EACH, end);
    let last = min(first + EACH, end);
    var kept = 0u;
    for (var i = first; i < last; i++) {
        if (logits[i] >= lowest) {
            kept++;
        }
    }
    var into = offset + u32(scan(f32(kept), t).x);
    for (var i = first; i < last; i++) {
        let v = logits[i];
        if (v >= lowest) {
            order[into] = i;
            probs[into] = exp((v - best) * inverse);
            into++;
        }
    }
}`;
export const SAMPLE_PICK = /* wgsl */ `
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: State;
@group(0) @binding(4) var<storage, read_write> chosen: array<u32>;
@group(0) @binding(5) var<storage, read> randoms: array<f32>;
@group(0) @binding(6) var<uniform> settings: Sampling;
@group(0) @binding(7) var<storage, read> parts: array<vec2<u32>>;
${CHUNKS_COMMON}
${SAMPLER_DRAW}
var<workgroup> found_chunk: atomic<u32>;
var<workgroup> chunk_before: f32;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    if (t == 0u) {
        atomicStore(&found, NONE);
        atomicStore(&found_chunk, NONE);
    }
    if (stopped(t)) {
        return;
    }
    let vocab = settings.vocab;
    let chunks = (vocab + CHUNK - 1u) / CHUNK;
    let top = largest(t);
    let argmax = top.at;
    // T219: the step is refused where the largest logit is no finite number (a NaN: SAMPLE_MAX's flags of the chunks;
    // +inf: the largest; or no chunk with a logit over -3.4e38), as SAMPLE refuses it
    var bad = false;
    for (var j = t; j < chunks; j += WG_SIZE) {
        bad = bad || flagged(j);
    }
    if (any_of(bad || bitcast<u32>(top.value) == INFINITY_BITS || argmax == NONE, t)) {
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
    let nucleus = settings.topp > 0.0 && settings.topp < 1.0;
    let r = min(randoms[state.sampled], BELOW_ONE);

    if (!nucleus) {
        // the draw in the order of the index: the chunk whose running sum passes r × total (a run of consecutive
        // chunks a thread, their sums scanned), then the token in it whose does, else the last
        let run = (chunks + WG_SIZE - 1u) / WG_SIZE;
        let lo_chunk = min(t * run, chunks);
        let hi_chunk = min(lo_chunk + run, chunks);
        var own = 0.0;
        for (var j = lo_chunk; j < hi_chunk; j++) {
            own += bitcast<f32>(counted(j).y);
        }
        let sums = scan(own, t);
        let goal = r * sums.y;
        var running = sums.x;
        for (var j = lo_chunk; j < hi_chunk; j++) {
            running += bitcast<f32>(counted(j).y);
            if (running > goal) {
                atomicMin(&found_chunk, j);
                break;
            }
        }
        workgroupBarrier();
        if (t == 0u) {
            uniform_word = atomicLoad(&found_chunk);
        }
        let c = workgroupUniformLoad(&uniform_word);
        if (c == NONE) {
            if (t == 0u) {
                finish(vocab - 1u);
            }
            return;
        }
        // the sum of the chunks before it, from the thread whose run holds it
        if (c >= lo_chunk && c < hi_chunk) {
            var prefix = sums.x;
            for (var j = lo_chunk; j < c; j++) {
                prefix += bitcast<f32>(counted(j).y);
            }
            chunk_before = prefix;
        }
        let prefix = workgroupUniformLoad(&chunk_before);
        let begin = c * CHUNK;
        let end = min(begin + CHUNK, vocab);
        let first = min(begin + t * EACH, end);
        let last = min(first + EACH, end);
        var mine = 0.0;
        for (var i = first; i < last; i++) {
            mine += probs[i];
        }
        var inside = prefix + scan(mine, t).x;
        for (var i = first; i < last; i++) {
            inside += probs[i];
            if (inside > goal) {
                atomicMin(&found, i);
                break;
            }
        }
        workgroupBarrier();
        if (t == 0u) {
            let hit = atomicLoad(&found);
            finish(select(hit, end - 1u, hit == NONE));
        }
        return;
    }

    // how many tokens the chunks gathered over the floor, and their mass
    var kept = 0.0;
    var mass = 0.0;
    for (var j = t; j < chunks; j += WG_SIZE) {
        let part = counted(j);
        kept += f32(part.x);
        mass += bitcast<f32>(part.y);
    }
    let count = u32(sum_all(kept, t));
    draw_nucleus(count, sum_all(mass, t), r, argmax, t);
}`;
/** The sampling in chunks (T191): its four stages in order, each with the bindings it takes (of SAMPLE's numbers and
 * 7 the partial results) and whether it runs a workgroup a chunk (else one workgroup) */
export const SAMPLER_STAGES = [
  { name: "max", code: SAMPLE_MAX, bindings: [0, 3, 6, 7], chunks: true },
  { name: "sum", code: SAMPLE_SUM, bindings: [0, 1, 3, 6, 7], chunks: true },
  { name: "gather", code: SAMPLE_GATHER, bindings: [0, 1, 2, 3, 6, 7], chunks: true },
  { name: "pick", code: SAMPLE_PICK, bindings: [1, 2, 3, 4, 5, 6, 7], chunks: false },
];
