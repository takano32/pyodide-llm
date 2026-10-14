// forward/attention.ts (T375.3): the layers that attend over every position before them, and keep keys and values
// (cache.ts): every layer of a Llama, a Qwen2, a Qwen3, a GPT-2 and a GPT-NeoX, the full-attention layers of a Qwen3.5
// and the attention layers of an LFM2. A kind of layer of forward/kinds.ts's table; a module of src/runtime/forward.ts.
//
// What one of them has beside q, k, v and the output is read off the model's tensors: Qwen2's biases of q, k and v
// (bq), Qwen3's norms of every head of q and k (q_norm), a Qwen3.5's gate of the output (wg). Whether RoPE turns a
// layer's q and k is the plan's fact of that layer (T255: none of a GPT-2's, not the ones a SmolLM3 leaves alone).
// T261: and so are two more, which a Gemma 3's sliding-attention layers have: how many of the last positions a layer
// attends (window; 0: all of them), and whether RoPE turns it by the model's local tables. A layer with a window is
// handed the cache from its window's first position on, and the kernels that attend know of no window. T510.1: where
// the context is long enough, such a layer keeps only its window's positions, in a ring of the cache (cache.ts: each
// position twice, so that a window is one run of the ring, oldest first), and is handed that run.
import { BATCH, KIND } from "../jobs.ts";
import type { Core, Kind, Layers, Product } from "./kinds.ts";
import type { Layer } from "../contracts/plan.ts";

export const attention: Kind = {
  // a Qwen3.5's gate of the output beside q; and the scores of one token's attention, apart from its frame (tokens run
  // at once, each with scores of its own)
  frame: ({ gate, widths, seqLen, heads }) => ({ arrays: gate ? [["gate", widths.q * 4]] : [], apart: [["att", seqLen * heads * 4]] }),
  cached: true,
  make({ plan, eps, k, floats, matrix, at, S, matmuls, phase, cache }: Core, count: number, facts: Layer[] = []): Layers {
    const { n_heads: heads, n_kv_heads: kvHeads, head_size: headSize, seq_len: seqLen, rotary } = plan, { q: qDim, kv: kvDim } = plan.widths;
    // kNow, vNow: this token's key and value in float32, before they go into the cache
    const { xb, xb2, q, kNow, vNow, gate, att } = at;
    const A = seqLen * heads * 4, KF = kvDim * 4, QF = qDim * 4, HS = headSize * 4, { KV } = cache;
    const wq = matrix("wq"), wk = matrix("wk"), wv = matrix("wv"), wo = matrix("wo");
    const bq = floats("bq"), bk = floats("bk"), bv = floats("bv");
    const qNorm = floats("q_norm"), kNorm = floats("k_norm");  // T124: Qwen3 normalizes every head of q and k
    const cosTable = floats("freq_cis_real"), sinTable = floats("freq_cis_imag");
    const wg = matrix("wg");  // T229
    // T261: the tables of the local layers (0: the model has none, and no layer of its plan says it is one)
    const localCos = floats("local_cis_real"), localSin = floats("local_cis_imag");
    // where these layers' keys and values are in the cache: the place-th layer's are the (first + place)-th (T510.1:
    // with each one's window, which a ring may hold)
    const windows: number[] = [];
    for (const { place, window } of facts) windows[place] = window;
    const first = cache.hold(count, windows);
    // the job (jobs.ts) of the attention of token t of a run, at position pos (T110: over float16 keys and values, or float32)
    const scores = cache.half ? KIND.attention_f16 : KIND.attention;
    return {
      layer(a, { rope, window, local }) {
        if (local && !localCos) throw new Error("The plan has a local layer and no RoPE tables of the local layers.");
        const cosOf = local ? localCos : cosTable, sinOf = local ? localSin : sinTable;
        // (a Qwen3.5's gate of the attention's output is one more matrix of the same input)
        const into: Product[] = wg ? [[wq, q, S, a], [wk, kNow, S, a], [wv, vNow, S, a], [wg, gate, S, a]] : [[wq, q, S, a], [wk, kNow, S, a], [wv, vNow, S, a]];
        const out: Product[] = [[wo, xb2, S, a]];
        // this layer's ring (T510.1), or its index among the blocks of the cache that keep every position
        const ring = cache.ring(first + a), index = cache.index(first + a);
        // token t's q, key and value at position pos, after the matmuls: biases, norms of the heads, RoPE
        const ready = (t: number, pos: number) => {
          const qt = q + t * S, kt = kNow + t * S, vt = vNow + t * S;
          if (bq) {
            k.add_inplace(qt, bq + a * QF, qDim);
            k.add_inplace(kt, bk + a * KF, kvDim);
            k.add_inplace(vt, bv + a * KF, kvDim);
          }
          if (qNorm) {
            for (let h = 0; h < heads; h++) k.rmsnorm(qt + h * HS, qt + h * HS, qNorm + a * HS, headSize, eps);
            for (let h = 0; h < kvHeads; h++) k.rmsnorm(kt + h * HS, kt + h * HS, kNorm + a * HS, headSize, eps);
          }
          if (rope) {
            const cos = cosOf + pos * (headSize / 2) * 4, sin = sinOf + pos * (headSize / 2) * 4;
            k.rope(qt, cos, sin, heads, headSize, rotary);
            k.rope(kt, cos, sin, kvHeads, headSize, rotary);
          }
        };
        if (ring) {
          // T510.1: the positions of the ring: position p at p mod size and at p mod size + size. A run may begin at 0,
          // or where the window of its first token is still in the ring: no further back than a block before the last
          // position written (what came after it has written over the ones before)
          const { size } = ring, twice = size * KV;
          let latest = -1;
          return (count, pos0) => {
            if (pos0 !== 0 && pos0 < latest - BATCH) {
              throw new Error(`A layer of this model keeps the keys and values of its last ${window} positions alone: position ${pos0} is too far back after ${latest}.`);
            }
            latest = pos0 === 0 ? pos0 + count - 1 : Math.max(latest, pos0 + count - 1);
            const { keys, values } = ring;
            matmuls(xb, count, into);
            for (let t = 0; t < count; t++) {
              const pos = pos0 + t, slot = (pos % size) * KV;
              ready(t, pos);
              cache.put(keys + slot, values + slot, kNow + t * S, vNow + t * S);
              cache.put(keys + slot + twice, values + slot + twice, kNow + t * S, vNow + t * S);
            }
            // every token's window is in the ring now, from where its first position is
            const jobs = [];
            for (let t = 0; t < count; t++) {
              const from = Math.max(0, pos0 + t + 1 - window), slot = (from % size) * KV;
              jobs.push([scores, xb + t * S, q + t * S, keys + slot, values + slot, att + t * A, pos0 + t - from, kvHeads, headSize, heads, 1, 0, 0, 0]);
            }
            phase(jobs);
            if (wg) for (let t = 0; t < count; t++) k.gate(xb + t * S, xb + t * S, gate + t * S, qDim);
            matmuls(xb, count, out);
          };
        }
        return (count, pos0) => {
          const { capacity } = cache, layerKeys = cache.keys + index * capacity * KV, layerValues = cache.values + index * capacity * KV;
          const kp = layerKeys + pos0 * KV, vp = layerValues + pos0 * KV;
          matmuls(xb, count, into);
          for (let t = 0; t < count; t++) {
            ready(t, pos0 + t);
            cache.put(kp + t * KV, vp + t * KV, kNow + t * S, vNow + t * S);  // into the cache, at this token's position
          }
          // the keys and values of positions up to each token's are all there now: its own and the ones before it.
          // The heads of every token go out as one phase (T109).
          const jobs = [];
          for (let t = 0; t < count; t++) {
            // (the first position this token attends: 0, or where its window begins)
            const from = window ? Math.max(0, pos0 + t + 1 - window) : 0;
            jobs.push([scores, xb + t * S, q + t * S, layerKeys + from * KV, layerValues + from * KV, att + t * A, pos0 + t - from, kvHeads, headSize, heads, 1, 0, 0, 0]);
          }
          phase(jobs);
          if (wg) for (let t = 0; t < count; t++) k.gate(xb + t * S, xb + t * S, gate + t * S, qDim);
          matmuls(xb, count, out);
        };
      },
    };
  },
};
