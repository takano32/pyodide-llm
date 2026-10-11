// shaders/attention.js (T351): the attention: llama.cpp's flash attention by tiles for the tokens of a prompt
// (T147) and its flash_attn_vec for one generated token (T224), each under its notice below, and the made-up numbers and
// the JavaScript the checks hold a token's attention to.
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).
const { STEP } = await import(new URL(`common.js${new URL(import.meta.url).search}`, import.meta.url));

// T147: the attention of a prompt's tokens, llama.cpp's flash attention with tiles (flash_attn_tile.wgsl), where
// T135's (one workgroup a head and token) wrote every score to memory and read it three times, and read the same keys
// and values again for every head of q that shares them. A workgroup takes Q_TILE (4) tokens of one head: their q
// (scaled) in its memory, then KV_TILE positions of the head's keys at a time in its memory as f16 (or f32 where there
// is no shader-f16), one row of q a subgroup, a position a lane; the softmax online (the largest so far and the sum
// rescaled as a tile brings a larger one: subgroupMax and subgroupAdd), and the values of the tile the same way into
// each lane's vec4s of the output. Changed from the source, and why: the causal mask is the positions' order (a
// position past the row's own is not seen; llama.cpp adds a mask tensor of -inf), so the tiles stop at the last
// token's position; the keys and values are this project's cache (float16 pairs in u32, [positions][kvHeads ×
// headSize], a head's row from its offset), read four at a time as llama.cpp's vec4 loader (flash_attn_staging.tmpl);
// q and the output are float32 [tokens][heads × headSize]; no ALiBi, soft-cap or sinks. Where there are no subgroups
// (or no subgroup_id), LANES threads of the workgroup stand for a row's subgroup and add up in the workgroup's memory
// (this project's: llama.cpp's tile path needs subgroups). The shape is llama.cpp's choice
// (ggml-webgpu-shader-lib.hpp): Q_TILE 4, KV_TILE at most 64 and what the workgroup's memory holds, WG_SIZE the larger
// of 128 and 4 subgroups; MIN_SUBGROUP_SIZE sizes the registers.
export const FLASH_Q_TILE = 4;
export const flashShape = ({ headSize, half, subgroups, memory, threads, subgroupMin = 4, subgroupMax = 128 }) => {
  const bytes = half ? 2 : 4;
  const wgSize = subgroups ? Math.min(threads, Math.max(128, FLASH_Q_TILE * subgroupMax)) : 128;
  // llama.cpp's ggml_webgpu_flash_attn_wg_mem_bytes, for this shader's arrays: q, then per position its keys or values
  // and a weight of each row
  const base = FLASH_Q_TILE * headSize * 4 + (subgroups ? 0 : wgSize * 4), perPosition = headSize * bytes + FLASH_Q_TILE * bytes;
  const kvTile = Math.min(64, Math.floor((memory - base) / perPosition));
  return { headSize, half, subgroups, wgSize, kvTile, minSubgroup: subgroups ? subgroupMin : wgSize / FLASH_Q_TILE,
    none: subgroups && wgSize < FLASH_Q_TILE * subgroupMax ? `subgroups of ${subgroupMax} are more than ${threads} threads / 4`
      : kvTile < 1 ? `a head of ${headSize} is more than the workgroup's memory (${memory} bytes)` : undefined };
};

// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_tile.wgsl, flash_attn_decls.tmpl and
// flash_attn_staging.tmpl (https://github.com/ggml-org/llama.cpp, commit 2145525a, 2026-09-26), under the MIT License:
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
export const flashTile = ({ headSize, half, subgroups, wgSize, kvTile, minSubgroup }) => /* wgsl */ `${half ? "enable f16;\n" : ""}${subgroups ? "enable subgroups;\n" : ""}
struct Params { heads: u32, kvHeads: u32, scale: f32, unused: u32 }
${STEP}
@group(0) @binding(0) var<storage, read> Q: array<f32>;             // [tokens][heads × HEAD_DIM]
@group(0) @binding(1) var<storage, read> K: array<vec2<u32>>;       // [positions][kvHeads × HEAD_DIM]: 4 f16 a vec2<u32>
@group(0) @binding(2) var<storage, read> V: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;  // [tokens][heads × HEAD_DIM]
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<uniform> step: Step;

alias shmem_t = ${half ? "f16" : "f32"};
const HEAD_DIM_QK: u32 = ${headSize}u;
const HEAD_DIM_V: u32 = ${headSize}u;
const Q_TILE: u32 = ${FLASH_Q_TILE}u;
const KV_TILE: u32 = ${kvTile}u;
const WG_SIZE: u32 = ${wgSize}u;
const MIN_SUBGROUP_SIZE: u32 = ${minSubgroup}u;
// Just a very small float value.
const FLOAT_MIN: f32 = -1.0e9;

const Q_CHUNKS: u32 = HEAD_DIM_QK / 4u;
const V_CHUNKS: u32 = HEAD_DIM_V / 4u;
const SCORE_REGS_PER_LANE: u32 = (KV_TILE + MIN_SUBGROUP_SIZE - 1u) / MIN_SUBGROUP_SIZE;
const OUT_REGS_PER_LANE: u32 = (V_CHUNKS + MIN_SUBGROUP_SIZE - 1u) / MIN_SUBGROUP_SIZE;

const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
var<workgroup> kv_shmem: array<shmem_t, kv_shmem_size>;
var<workgroup> q_shmem: array<f32, Q_TILE * HEAD_DIM_QK>;
var<workgroup> p_shmem: array<shmem_t, Q_TILE * KV_TILE>;

fn halves(pair: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(unpack2x16float(pair.x), unpack2x16float(pair.y));
}
fn load_k_tile_block(local_x: u32, kv_count: u32, kv_tile: u32, k_head_offset: u32) {
    let stride_k1 = params.kvHeads * HEAD_DIM_QK;
    for (var vec_idx_local = local_x; vec_idx_local < kv_count * Q_CHUNKS; vec_idx_local += WG_SIZE) {
        let kv_local = vec_idx_local / Q_CHUNKS;
        let chunk = vec_idx_local % Q_CHUNKS;
        let global_k_row = kv_tile + kv_local;
        let k_vec_index = (k_head_offset + global_k_row * stride_k1 + chunk * 4u) >> 2u;
        let k4 = halves(K[k_vec_index]);
        let kv_off = kv_local * HEAD_DIM_QK + chunk * 4u;
        kv_shmem[kv_off + 0u] = shmem_t(k4.x);
        kv_shmem[kv_off + 1u] = shmem_t(k4.y);
        kv_shmem[kv_off + 2u] = shmem_t(k4.z);
        kv_shmem[kv_off + 3u] = shmem_t(k4.w);
    }
}
fn load_v_tile_block(local_x: u32, kv_count: u32, kv_tile: u32, v_head_offset: u32) {
    let stride_v1 = params.kvHeads * HEAD_DIM_V;
    for (var vec_idx_local = local_x; vec_idx_local < kv_count * V_CHUNKS; vec_idx_local += WG_SIZE) {
        let kv_local = vec_idx_local / V_CHUNKS;
        let chunk = vec_idx_local % V_CHUNKS;
        let global_v_row = kv_tile + kv_local;
        let v_vec_index = (v_head_offset + global_v_row * stride_v1 + chunk * 4u) >> 2u;
        let v4 = halves(V[v_vec_index]);
        let kv_off = kv_local * HEAD_DIM_V + chunk * 4u;
        kv_shmem[kv_off + 0u] = shmem_t(v4.x);
        kv_shmem[kv_off + 1u] = shmem_t(v4.y);
        kv_shmem[kv_off + 2u] = shmem_t(v4.z);
        kv_shmem[kv_off + 3u] = shmem_t(v4.w);
    }
}
${subgroups ? `fn row_max(value: f32) -> f32 { return subgroupMax(value); }
fn row_sum(value: f32) -> f32 { return subgroupAdd(value); }` : `// a row's LANES threads stand for its subgroup: their largest and their sum through the workgroup's memory
const LANES: u32 = WG_SIZE / Q_TILE;
var<workgroup> lanes: array<f32, WG_SIZE>;
var<private> lane_x: u32;
fn row_max(value: f32) -> f32 {
  lanes[lane_x] = value;
  workgroupBarrier();
  for (var half = LANES / 2u; half > 0u; half >>= 1u) {
    if (lane_x % LANES < half) { lanes[lane_x] = max(lanes[lane_x], lanes[lane_x + half]); }
    workgroupBarrier();
  }
  let result = lanes[lane_x - lane_x % LANES];
  workgroupBarrier();
  return result;
}
fn row_sum(value: f32) -> f32 {
  lanes[lane_x] = value;
  workgroupBarrier();
  for (var half = LANES / 2u; half > 0u; half >>= 1u) {
    if (lane_x % LANES < half) { lanes[lane_x] += lanes[lane_x + half]; }
    workgroupBarrier();
  }
  let result = lanes[lane_x - lane_x % LANES];
  workgroupBarrier();
  return result;
}`}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>${subgroups ? `,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(num_subgroups) num_subgroups: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {
    if (subgroup_size == 0u || num_subgroups < Q_TILE) {
        return;
    }` : `) {
    lane_x = local_id.x;
    let subgroup_id = local_id.x / LANES;
    let subgroup_size = LANES;
    let sg_inv_id = local_id.x % LANES;`}

    let wg_per_head = (step.tokens + Q_TILE - 1u) / Q_TILE;
    let head_idx = wg_id.x / wg_per_head;
    let k_head_idx = head_idx / (params.heads / params.kvHeads);
    let k_head_offset = k_head_idx * HEAD_DIM_QK;
    let v_head_offset = k_head_idx * HEAD_DIM_V;
    let stride_q1 = params.heads * HEAD_DIM_QK;

    let wg_in_head = wg_id.x % wg_per_head;
    let q_row_start = wg_in_head * Q_TILE;
    let global_q_row = q_row_start + subgroup_id;
    let row_active = subgroup_id < Q_TILE && global_q_row < step.tokens;
    // causal: the tile's positions up to its last token's, each row's up to its own
    let seq_len_kv = step.pos + min(q_row_start + Q_TILE, step.tokens);
    let row_position = step.pos + global_q_row;

    for (var elem_idx = local_id.x; elem_idx < Q_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
        let q_tile_row = elem_idx / HEAD_DIM_QK;
        let q_col = elem_idx % HEAD_DIM_QK;
        let head_q_row = q_row_start + q_tile_row;
        let global_q_row_offset = head_q_row * stride_q1 + head_idx * HEAD_DIM_QK;
        q_shmem[elem_idx] = select(
            0.0,
            Q[global_q_row_offset + q_col] * params.scale,
            head_q_row < step.tokens);
    }

    workgroupBarrier();

    var row_max_now = FLOAT_MIN;
    var exp_sum = 0.0;
    var out_regs: array<vec4<f32>, OUT_REGS_PER_LANE>;
    for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
        out_regs[reg_idx] = vec4<f32>(0.0);
    }

    let q_base = subgroup_id * HEAD_DIM_QK;
    let subgroup_p_offset = subgroup_id * KV_TILE;

    for (var kv_tile = 0u; kv_tile < seq_len_kv; kv_tile += KV_TILE) {
        let kv_count = min(KV_TILE, seq_len_kv - kv_tile);
        let score_slots = min(SCORE_REGS_PER_LANE, (kv_count + subgroup_size - 1u) / subgroup_size);
        let out_slots = min(OUT_REGS_PER_LANE, (V_CHUNKS + subgroup_size - 1u) / subgroup_size);
        var local_scores: array<f32, SCORE_REGS_PER_LANE>;
        for (var slot = 0u; slot < SCORE_REGS_PER_LANE; slot += 1u) {
            local_scores[slot] = FLOAT_MIN;
        }

        load_k_tile_block(local_id.x, kv_count, kv_tile, k_head_offset);

        workgroupBarrier();

        var local_max = FLOAT_MIN;
        if (row_active) {
            for (var slot = 0u; slot < score_slots; slot += 1u) {
                let kv_local = sg_inv_id + slot * subgroup_size;
                if (kv_local >= kv_count) {
                    continue;
                }

                let global_k_row = kv_tile + kv_local;
                var dot_val = 0.0;
                for (var chunk = 0u; chunk < Q_CHUNKS; chunk += 1u) {
                    let q_off = q_base + chunk * 4u;
                    let qv = vec4<f32>(
                        q_shmem[q_off + 0u],
                        q_shmem[q_off + 1u],
                        q_shmem[q_off + 2u],
                        q_shmem[q_off + 3u]);
                    let kv_off = kv_local * HEAD_DIM_QK + chunk * 4u;
                    let kv = vec4<shmem_t>(
                        kv_shmem[kv_off + 0u],
                        kv_shmem[kv_off + 1u],
                        kv_shmem[kv_off + 2u],
                        kv_shmem[kv_off + 3u]);
                    dot_val += dot(qv, vec4<f32>(kv));
                }
                // the causal mask: no position after the row's own
                if (global_k_row > row_position) {
                    dot_val = FLOAT_MIN;
                }
                local_scores[slot] = dot_val;
                local_max = max(local_max, dot_val);
            }
        }

        let tile_max = row_max(local_max);
        let new_max = max(row_max_now, tile_max);
        let cur_exp = exp(row_max_now - new_max);
        exp_sum *= cur_exp;
        for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
            out_regs[reg_idx] *= cur_exp;
        }

        var local_sum = 0.0;
        for (var slot = 0u; slot < score_slots; slot += 1u) {
            let kv_local = sg_inv_id + slot * subgroup_size;
            if (row_active && kv_local < kv_count) {
                let p = exp(local_scores[slot] - new_max);
                p_shmem[subgroup_p_offset + kv_local] = shmem_t(p);
                local_sum += p;
            }
        }

        workgroupBarrier();

        load_v_tile_block(local_id.x, kv_count, kv_tile, v_head_offset);

        workgroupBarrier();

        let tile_sum = row_sum(local_sum);
        exp_sum += tile_sum;
        row_max_now = new_max;

        if (row_active) {
            for (var reg_idx = 0u; reg_idx < out_slots; reg_idx += 1u) {
                let chunk = sg_inv_id + reg_idx * subgroup_size;
                if (chunk >= V_CHUNKS) {
                    continue;
                }

                var acc = out_regs[reg_idx];
                for (var kv_local = 0u; kv_local < kv_count; kv_local += 1u) {
                    let p = f32(p_shmem[subgroup_p_offset + kv_local]);
                    let kv_off = kv_local * HEAD_DIM_V + chunk * 4u;
                    let v4 = vec4<shmem_t>(
                        kv_shmem[kv_off + 0u],
                        kv_shmem[kv_off + 1u],
                        kv_shmem[kv_off + 2u],
                        kv_shmem[kv_off + 3u]);
                    acc += p * vec4<f32>(v4);
                }
                out_regs[reg_idx] = acc;
            }
        }

        workgroupBarrier();
    }

    if (row_active) {
        let inv_exp_sum = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
        let row_base = global_q_row * stride_q1 + head_idx * HEAD_DIM_V;
        let out_slots = min(OUT_REGS_PER_LANE, (V_CHUNKS + subgroup_size - 1u) / subgroup_size);
        for (var reg_idx = 0u; reg_idx < out_slots; reg_idx += 1u) {
            let chunk = sg_inv_id + reg_idx * subgroup_size;
            if (chunk >= V_CHUNKS) {
                continue;
            }
            let dst_vec_index = (row_base + chunk * 4u) >> 2u;
            dst[dst_vec_index] = out_regs[reg_idx] * inv_exp_sum;
        }
    }
}`;

// T224: the attention of one generated token, llama.cpp's decode form (flash_attn_vec_split.wgsl and
// flash_attn_vec_reduce.wgsl, chosen by ggml-webgpu.cpp's ggml_webgpu_flash_attn_vec where a query has fewer than 20
// rows): where the prompt's tiles (flashTile) leave three of their four rows of q empty for one token and run a
// workgroup a head, this runs nwg workgroups a head, each over every nwg-th KV_TILE of the positions (one subgroup:
// D_SPLIT lanes a position for q·k, the softmax online by subgroupMax and subgroupAdd, the values the same way into
// the workgroup's output), each writing its output unscaled with its sum and its largest score; a second dispatch
// (the reduce) then takes the largest of the heads' largest, rescales each part by exp(its largest − that) and
// divides by the sum of the rescaled sums. nwg doubles while 2 × nwg × KV_TILE positions fall short of the positions
// read, up to the least subgroup (llama.cpp's ggml_webgpu_flash_attn_vec_nwg; its reduce holds a part a lane): at
// position 127 two parts a head, at 1023 and 4095 as many as the least subgroup (16 on the owner's Mali). With one part
// the first dispatch writes the output itself and there is no reduce.
// Changed from the source, and why: the one row of q is the token's (no mask: it reads the positions up to its own,
// step.pos + 1), no ALiBi, soft-cap, sinks or blocks; the keys and values are this project's cache (float16 pairs in
// u32, [positions][kvHeads × headSize]) read straight from it four at a time (llama.cpp's K_DIRECT and V_DIRECT for
// f16 read vec4<f16> from the cache so); q and the output float32 [heads × headSize], the parts float32 in their own
// buffer (llama.cpp's scratch past dst); the parts' place (nwg and where the sums and largests start) a uniform of the
// run's nwg (llama.cpp's tmp_stats_base is the host's too); the values' loop over the output's vec4s runs from a
// uniform base (col_base + tx_pv: every lane goes round it as often, D_SPLIT dividing headSize / 4) and its reduction
// of the four sums is one subgroupShuffleDown of the vec4 (llama.cpp shuffles the four floats one by one), as the
// reduce's subgroupAdd; an index into the workgroup's scores is kept within them where the source reads past them.
// The shape is llama.cpp's choice (ggml-webgpu-shader-lib.hpp's get_flash_attn_vec_pipeline): KV_TILE 32
// (GGML_WEBGPU_FLASH_ATTN_VEC_MAX_KV_TILE), a workgroup of the largest subgroup, D_SPLIT the least of the least
// subgroup, 4 and the lowest bit of headSize / 4; the reduce's workgroup the larger of the largest subgroup and nwg of
// them (at most the device's threads), where llama.cpp makes a reduce pipeline for the nwg of the call and this makes
// one for the most nwg a head can have (splits: the least subgroup), which every nwg runs: for two parts a head on a
// Mali (16..16) a workgroup of 256 threads where llama.cpp's is 32 (T224's review: the same sums, more subgroups to
// share the output's vec4s).
// Where there are no subgroups (or no subgroup_id), FLASH_VEC_LANES threads stand for one subgroup, its shuffles, largest
// and sum through the workgroup's memory (a butterfly: each pair of lanes adds the same two values, so every lane holds
// the same sum), and nwg goes up to FLASH_VEC_LANES (this project's, as flashTile's LANES: llama.cpp's vec path needs
// subgroups).
export const FLASH_VEC_KV_TILE = 32, FLASH_VEC_LANES = 32;
export const flashVecShape = ({ headSize, subgroups, threads, subgroupMin = 4, subgroupMax = 128 }) => {
  const least = subgroups ? subgroupMin : FLASH_VEC_LANES, wgSize = subgroups ? subgroupMax : FLASH_VEC_LANES;
  const lowest = headSize & -headSize, dSplit = Math.min(least, 4, Math.max(lowest / 4, 1));
  return { headSize, subgroups, wgSize, kvTile: FLASH_VEC_KV_TILE, dSplit, splits: least,
    reduceSize: subgroups ? Math.max(subgroupMax, Math.min(least * subgroupMax, threads)) : FLASH_VEC_LANES,
    none: headSize % 4 ? `a head of ${headSize} is not of four values` : wgSize > threads ? `subgroups of ${wgSize} are more than ${threads} threads` : undefined };
};
// Adapted from llama.cpp, ggml/src/ggml-webgpu/ggml-webgpu.cpp (ggml_webgpu_flash_attn_vec_nwg), commit 95887577, under
// the MIT License (the notice is the one above flashVec): the workgroups a head for positions
export function flashVecSplits({ splits, kvTile }, positions) {
  let nwg = 1;
  while (2 * nwg * kvTile < positions && nwg < splits) nwg <<= 1;
  return Math.min(nwg, splits);
}
// the bytes of the parts a dispatch of heads writes (nwg of the most a head, headSize values and two numbers each)
export const flashVecPartsBytes = ({ splits, headSize }, heads) => heads * splits * (headSize + 2) * 4;
// the Params both dispatches read: heads, kvHeads, the scale of a score, nwg, where the sums and largests start
export function flashVecParams({ headSize, splits }, heads, kvHeads, nwg) {
  const bytes = new ArrayBuffer(32);
  new Uint32Array(bytes).set([heads, kvHeads, 0, nwg, heads * nwg * headSize, 0, 0, 0]);
  new Float32Array(bytes, 8, 1)[0] = 1 / Math.sqrt(headSize);
  return new Uint8Array(bytes);
}
// the WGSL head and helpers both take: a subgroup's shuffles, largest and sum, the subgroup's or (none) the workgroup's
// lanes' through its memory
const flashVecLanes = (subgroups, wgSize) => (subgroups ? /* wgsl */ `
fn shuffle_down(value: f32, delta: u32) -> f32 { return subgroupShuffleDown(value, delta); }
fn shuffle_down4(value: vec4<f32>, delta: u32) -> vec4<f32> { return subgroupShuffleDown(value, delta); }
fn shuffle(value: f32, source: u32) -> f32 { return subgroupShuffle(value, source); }
fn lanes_max(value: f32) -> f32 { return subgroupMax(value); }
fn lanes_add(value: f32) -> f32 { return subgroupAdd(value); }
fn lanes_add4(value: vec4<f32>) -> vec4<f32> { return subgroupAdd(value); }` : /* wgsl */ `
const LANES: u32 = ${wgSize}u;
var<workgroup> lanes: array<vec4<f32>, LANES>;
var<private> lane: u32;
fn exchange(value: vec4<f32>, source: u32) -> vec4<f32> {
  lanes[lane] = value;
  workgroupBarrier();
  let got = lanes[min(source, LANES - 1u)];
  workgroupBarrier();
  return got;
}
fn shuffle_down(value: f32, delta: u32) -> f32 { return exchange(vec4<f32>(value), lane + delta).x; }
fn shuffle_down4(value: vec4<f32>, delta: u32) -> vec4<f32> { return exchange(value, lane + delta); }
fn shuffle(value: f32, source: u32) -> f32 { return exchange(vec4<f32>(value), source).x; }
fn lanes_max(value: f32) -> f32 {
  var v = value;
  for (var delta = LANES / 2u; delta > 0u; delta >>= 1u) { v = max(v, exchange(vec4<f32>(v), lane ^ delta).x); }
  return v;
}
fn lanes_add4(value: vec4<f32>) -> vec4<f32> {
  var v = value;
  for (var delta = LANES / 2u; delta > 0u; delta >>= 1u) { v += exchange(v, lane ^ delta); }
  return v;
}
fn lanes_add(value: f32) -> f32 { return lanes_add4(vec4<f32>(value)).x; }`);
const FLASH_VEC_PARAMS = /* wgsl */ `struct Params { heads: u32, kvHeads: u32, scale: f32, nwg: u32, stats: u32, unused0: u32, unused1: u32, unused2: u32 }`;
const flashVecHead = (subgroups) => (subgroups ? "diagnostic(off, subgroup_uniformity);\nenable subgroups;\nrequires subgroup_id;\n" : "");

// Adapted from llama.cpp, ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl and flash_attn_vec_reduce.wgsl
// (and, for flashVecShape's and flashVecSplits' numbers, ggml-webgpu.cpp and ggml-webgpu-shader-lib.hpp; commit
// 95887577, committed 2026-09-26 20:05 UTC, https://github.com/ggml-org/llama.cpp), under the MIT License:
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
// The first dispatch: heads × nwg workgroups (the head wg_id.x / nwg, its part wg_id.x % nwg)
export const flashVec = ({ headSize, subgroups, wgSize, kvTile, dSplit }) => {
  // only the first subgroup works where the workgroup holds more than one (llama.cpp's); the lanes are one subgroup
  const first = subgroups ? "if (subgroup_id == 0u) {" : "{";
  return /* wgsl */ `${flashVecHead(subgroups)}
${FLASH_VEC_PARAMS}
${STEP}
@group(0) @binding(0) var<storage, read> Q: array<f32>;             // [heads × HEAD_DIM]: the token's q
@group(0) @binding(1) var<storage, read> K: array<vec2<u32>>;       // [positions][kvHeads × HEAD_DIM]: 4 f16 a vec2<u32>
@group(0) @binding(2) var<storage, read> V: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> tmp: array<f32>;     // the parts: [heads][nwg][HEAD_DIM], then from
                                                                    // params.stats [heads][nwg][the sum, the largest]
@group(0) @binding(4) var<storage, read_write> dst: array<vec4<f32>>;  // nwg 1: the output, [heads × HEAD_DIM]
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<uniform> step: Step;

const HEAD_DIM_QK: u32 = ${headSize}u;
const HEAD_DIM_V: u32 = ${headSize}u;
const KV_TILE: u32 = ${kvTile}u;
const WG_SIZE: u32 = ${wgSize}u;
const D_SPLIT: u32 = ${dSplit}u;
const FLOAT_MIN: f32 = -1.0e9;
const Q_CHUNKS: u32 = HEAD_DIM_QK / 4u;
const V_CHUNKS: u32 = HEAD_DIM_V / 4u;

var<workgroup> q_shmem: array<f32, HEAD_DIM_QK>;
var<workgroup> o_shmem: array<f32, HEAD_DIM_V>;
var<workgroup> inter_shmem: array<f32, KV_TILE>;
${flashVecLanes(subgroups, wgSize)}

fn halves(pair: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(unpack2x16float(pair.x), unpack2x16float(pair.y));
}
fn calc_softmax_term(kv_idx: u32) -> f32 {
    return select(FLOAT_MIN, inter_shmem[min(kv_idx, KV_TILE - 1u)] * params.scale, kv_idx < KV_TILE);
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>${subgroups ? `,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {` : `) {
    lane = local_id.x;
    let subgroup_size = WG_SIZE;
    let sg_inv_id = local_id.x;`}
    var row_max = FLOAT_MIN;
    var exp_sum = 0.0;

    for (var i = local_id.x; i < HEAD_DIM_V; i += WG_SIZE) {
        o_shmem[i] = 0.0;
    }

    let iwg = wg_id.x % params.nwg;
    let head_idx = wg_id.x / params.nwg;
    let k_head_idx = head_idx / (params.heads / params.kvHeads);
    let stride_k1 = params.kvHeads * HEAD_DIM_QK;
    let k_head_offset = k_head_idx * HEAD_DIM_QK;
    let v_head_offset = k_head_idx * HEAD_DIM_V;
    // the token's positions: up to its own
    let seq_len_kv = step.pos + 1u;

    // load the single Q row into shared memory
    for (var elem_idx = local_id.x; elem_idx < HEAD_DIM_QK; elem_idx += WG_SIZE) {
        q_shmem[elem_idx] = Q[head_idx * HEAD_DIM_QK + elem_idx];
    }

    for (var kv_tile = iwg * KV_TILE; kv_tile < seq_len_kv; kv_tile += KV_TILE * params.nwg) {
        for (var elem_idx = local_id.x; elem_idx < KV_TILE; elem_idx += WG_SIZE) {
            inter_shmem[elem_idx] = 0.0;
        }

        workgroupBarrier();

        // accumulate q block * k block into registers across the entire KV tile
        let num_of_threads: u32 = D_SPLIT;
        let tx = sg_inv_id % num_of_threads;
        let ty = sg_inv_id / num_of_threads;
        ${first}
            for (var kv_base: u32 = 0u; kv_base < KV_TILE; kv_base += subgroup_size / D_SPLIT) {
                let kv_idx = kv_base + ty;
                var partial_sum: f32 = 0.0;
                let kv_valid = kv_idx < KV_TILE && (kv_tile + kv_idx) < seq_len_kv;
                if (kv_valid) {
                    for (var i = tx; i < Q_CHUNKS; i += num_of_threads) {
                        let q_off = i * 4u;
                        let qv = vec4<f32>(
                            q_shmem[q_off + 0u],
                            q_shmem[q_off + 1u],
                            q_shmem[q_off + 2u],
                            q_shmem[q_off + 3u]);
                        let idx = k_head_offset + (kv_tile + kv_idx) * stride_k1 + (i * 4u);
                        let kv = halves(K[idx >> 2u]);
                        partial_sum += dot(qv, kv);
                    }
                }
                var sum = partial_sum;
                // Reduce over tx threads (NL) for this ty stripe.
                var tx_delta = num_of_threads >> 1u;
                loop {
                    if (tx_delta == 0u) {
                        break;
                    }
                    let sh = shuffle_down(sum, tx_delta);
                    if (tx < tx_delta) {
                        sum += sh;
                    }
                    tx_delta >>= 1u;
                }

                let sum_bcast = shuffle(sum, num_of_threads * ty);
                if (tx == 0u && kv_valid) {
                    inter_shmem[kv_idx] = sum_bcast;
                }
            }
        }

        workgroupBarrier();

        // online softmax
        ${first}
            let prev_max = row_max;
            var final_max = prev_max;
            // pass 1: compute final max across the full KV tile in chunks
            for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
                let kv_idx = kv_offset + sg_inv_id;
                let kv_valid = kv_tile + kv_idx < seq_len_kv && kv_idx < KV_TILE;
                let softmax_term = select(FLOAT_MIN, calc_softmax_term(kv_idx), kv_valid);
                final_max = lanes_max(max(final_max, softmax_term));
            }

            var total_exp_term: f32 = 0.0;
            // pass 2: compute exp sum and write P using final_max
            for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
                let kv_idx = kv_offset + sg_inv_id;
                let softmax_term = calc_softmax_term(kv_idx);
                let cur_p = select(0.0,
                                   exp(softmax_term - final_max),
                                   kv_tile + kv_idx < seq_len_kv && kv_idx < KV_TILE);
                total_exp_term += lanes_add(cur_p);
                if (kv_idx < KV_TILE) {
                    inter_shmem[kv_idx] = cur_p;
                }
            }

            let cur_exp = exp(prev_max - final_max);

            row_max = final_max;
            exp_sum = exp_sum * cur_exp + total_exp_term;

            for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
                o_shmem[elem_idx] = o_shmem[elem_idx] * cur_exp;
            }
        }

        workgroupBarrier();

        // we have P (KV_TILE) in inter_shmem and V (KV_TILE x head_dim_v) in the cache
        // we want to compute O += P * V across the full KV tile
        let ne_threads : u32 = subgroup_size / D_SPLIT;
        let nl_threads = max(1u, subgroup_size / ne_threads);
        let tx_pv = sg_inv_id % nl_threads;
        let ty_pv = sg_inv_id / nl_threads;
        ${first}
            for (var col_base = 0u; col_base < V_CHUNKS; col_base += nl_threads) {
                let vec_col = col_base + tx_pv;
                var lo = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                for (var cc = 0u; cc * ne_threads < KV_TILE; cc += 1u) {
                    let kv_idx = cc * ne_threads + ty_pv;
                    if (kv_idx >= KV_TILE) {
                        continue;
                    }
                    let v_row = kv_tile + kv_idx;
                    if (v_row >= seq_len_kv) {
                        continue;
                    }

                    let p = inter_shmem[kv_idx];
                    let v_idx = v_head_offset + v_row * stride_k1 + vec_col * 4u;
                    lo += p * halves(V[v_idx >> 2u]);
                }

                // Reduce over ty threads (NE) for this tx thread.
                var ty_delta = ne_threads >> 1u;
                loop {
                    if (ty_delta == 0u) {
                        break;
                    }
                    let sh = shuffle_down4(lo, ty_delta * nl_threads);
                    if (ty_pv < ty_delta) {
                        lo += sh;
                    }
                    ty_delta >>= 1u;
                }

                if (ty_pv == 0u) {
                    let elem_base = vec_col * 4u;
                    o_shmem[elem_base + 0u] = o_shmem[elem_base + 0u] + lo.x;
                    o_shmem[elem_base + 1u] = o_shmem[elem_base + 1u] + lo.y;
                    o_shmem[elem_base + 2u] = o_shmem[elem_base + 2u] + lo.z;
                    o_shmem[elem_base + 3u] = o_shmem[elem_base + 3u] + lo.w;
                }
            }
        }

        workgroupBarrier();
    }

    ${first}
        if (params.nwg == 1u) {
            let scale = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
            let row_base: u32 = head_idx * HEAD_DIM_V;
            for (var elem_base = sg_inv_id * 4u; elem_base < HEAD_DIM_V; elem_base += subgroup_size * 4u) {
                let v = vec4<f32>(
                    o_shmem[elem_base + 0u] * scale,
                    o_shmem[elem_base + 1u] * scale,
                    o_shmem[elem_base + 2u] * scale,
                    o_shmem[elem_base + 3u] * scale
                );
                dst[(row_base + elem_base) >> 2u] = v;
            }
        } else {
            let rid = head_idx;
            let tmp_row_data_base = rid * (HEAD_DIM_V * params.nwg) + iwg * HEAD_DIM_V;
            let tmp_row_stats_base = params.stats + rid * (2u * params.nwg) + 2u * iwg;
            for (var elem_base = sg_inv_id * 4u; elem_base < HEAD_DIM_V; elem_base += subgroup_size * 4u) {
                let tbase = tmp_row_data_base + elem_base;
                tmp[tbase + 0u] = o_shmem[elem_base + 0u];
                tmp[tbase + 1u] = o_shmem[elem_base + 1u];
                tmp[tbase + 2u] = o_shmem[elem_base + 2u];
                tmp[tbase + 3u] = o_shmem[elem_base + 3u];
            }
            if (sg_inv_id == 0u) {
                tmp[tmp_row_stats_base + 0u] = exp_sum;
                tmp[tmp_row_stats_base + 1u] = row_max;
            }
        }
    }
}`;
};
// The second dispatch (where nwg is more than 1): a workgroup a head, a lane a part (the reduce's subgroup, or the
// lanes)
export const flashVecReduce = ({ headSize, subgroups, reduceSize }) => /* wgsl */ `${flashVecHead(subgroups)}
${FLASH_VEC_PARAMS}
@group(0) @binding(0) var<storage, read> tmp: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const HEAD_DIM_V: u32 = ${headSize}u;
const WG_SIZE: u32 = ${reduceSize}u;
const FLOAT_MIN: f32 = -1.0e9;
${flashVecLanes(subgroups, reduceSize)}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>${subgroups ? `,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(num_subgroups) num_subgroups: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {` : `) {
    lane = local_id.x;
    let subgroup_id = 0u;
    let num_subgroups = 1u;
    let subgroup_size = WG_SIZE;
    let sg_inv_id = local_id.x;`}
    let rid = wg_id.x;
    let row_base = rid * HEAD_DIM_V;

    let thread = sg_inv_id;
    if (params.nwg > subgroup_size) {
        return;
    }

    let stats_base = params.stats + rid * (2u * params.nwg);
    let active_thread = thread < params.nwg;
    let part = min(thread, params.nwg - 1u);
    let si = select(0.0, tmp[stats_base + 2u * part + 0u], active_thread);
    let mi = select(FLOAT_MIN, tmp[stats_base + 2u * part + 1u], active_thread);
    let m = lanes_max(mi);
    let ms = select(0.0, exp(mi - m), active_thread);
    let s = lanes_add(si * ms);
    let inv_s = select(0.0, 1.0 / s, s != 0.0);

    let row_tmp_base = rid * (HEAD_DIM_V * params.nwg);
    for (var elem_base = subgroup_id * 4u; elem_base < HEAD_DIM_V; elem_base += num_subgroups * 4u) {
        var weighted = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (active_thread) {
            let src = row_tmp_base + thread * HEAD_DIM_V + elem_base;
            weighted = vec4<f32>(tmp[src + 0u], tmp[src + 1u], tmp[src + 2u], tmp[src + 3u]) * ms;
        }

        let sum = lanes_add4(weighted);

        if (thread == 0u) {
            dst[(row_base + elem_base) >> 2u] = sum * inv_s;
        }
    }
}`;
/** T224: the made-up numbers of a check of a token's attention (the engine's, gpu.js, and /benchmark/'s): q of heads
 * × size (±1; the heads in steep TOKEN_ATTENTION_STEEP times that, whose scores then spread over hundreds: a largest of
 * a softmax or of the reduce that is not the largest of all makes exp() go past float32 there, where a largest taken
 * wrong cancels out of any softmax whose scores are within 88 of each other), and the keys and values of positions
 * and TOKEN_ATTENTION_PAST more ([position][kvHeads × size], float16 of random bits between 2^-3 and 4 in size: those
 * past the token's must not be read). A steep head's key at TOKEN_ATTENTION_PEAK (in the second KV_TILE: flash_attn_vec's
 * second part where there are two or more) is 4 in the sign of its q, so that its score is hundreds above every other:
 * random keys alone leave every part's largest within 88 of the largest of all (the CI's mutations, T224: the reduce
 * taking part 0's largest passed), and a part's largest must be that far from another's to be seen. */
export const TOKEN_ATTENTION_PAST = 8, TOKEN_ATTENTION_STEEP = 40, TOKEN_ATTENTION_PEAK = 33;
export function tokenAttentionData({ heads, kvHeads, size, positions, steep = [] }) {
  const q = new Float32Array(heads * size).map((_, i) => (Math.random() * 2 - 1) * (steep.includes(Math.floor(i / size)) ? TOKEN_ATTENTION_STEEP : 1));
  const halfBits = () => ((Math.random() < 0.5 ? 0x8000 : 0) | ((12 + ((Math.random() * 5) | 0)) << 10) | ((Math.random() * 1024) | 0));
  const rows = (positions + TOKEN_ATTENTION_PAST) * kvHeads * size;
  const keys = new Uint16Array(rows).map(halfBits);
  for (const h of steep.filter(() => positions > TOKEN_ATTENTION_PEAK)) {
    const kv = Math.floor(h / (heads / kvHeads)) * size, at = TOKEN_ATTENTION_PEAK * kvHeads * size + kv;
    for (let d = 0; d < size; d++) keys[at + d] = q[h * size + d] < 0 ? 0xc400 : 0x4400;  // ±4 in float16
  }
  return { q, keys, values: new Uint16Array(rows).map(halfBits) };
}
const fromHalf = (h) => (h & 0x8000 ? -1 : 1) * ((h >> 10) & 31 ? 2 ** (((h >> 10) & 31) - 15) * (1 + (h & 1023) / 1024) : 2 ** -14 * ((h & 1023) / 1024));
/** T224: how far got (a token's attention output, heads × size, from the GPU) is from the attention of data
 * (tokenAttentionData's, heads of q on kvHeads, each head of q to kvHeads' own, the token reading positions) in float64:
 * the largest difference of a head's output over the largest |value| of its head. NaN where got holds one. */
export function tokenAttentionOff(got, { q, keys, values }, { heads, kvHeads, size, positions }) {
  const kvDim = kvHeads * size, scale = 1 / Math.sqrt(size);
  let worst = 0;
  for (let h = 0; h < heads; h++) {
    const kv = Math.floor(h / (heads / kvHeads)) * size, row = h * size, weights = new Float64Array(positions);
    for (let p = 0; p < positions; p++) {
      for (let d = 0; d < size; d++) weights[p] += q[row + d] * fromHalf(keys[p * kvDim + kv + d]) * scale;
    }
    const most = weights.reduce((a, b) => Math.max(a, b), -Infinity);
    let sum = 0, largest = 0;
    for (let p = 0; p < positions; p++) sum += (weights[p] = Math.exp(weights[p] - most));
    for (let p = 0; p < positions; p++) for (let d = 0; d < size; d++) largest = Math.max(largest, Math.abs(fromHalf(values[p * kvDim + kv + d])));
    for (let d = 0; d < size; d++) {
      let want = 0;
      for (let p = 0; p < positions; p++) want += weights[p] * fromHalf(values[p * kvDim + kv + d]);
      // (Math.max keeps a NaN, as the comment above says: `if (!(off <= worst)) worst = off` let the next value take its
      // place, and a NaN in one head, or in the first values of the last, passed: T224's review)
      worst = Math.max(worst, Math.abs(got[row + d] - want / sum) / largest);
    }
  }
  return worst;
}
