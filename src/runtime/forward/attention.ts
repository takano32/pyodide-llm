// forward/attention.ts (T375.3): the layers that attend over every position before them, and keep keys and values
// (cache.ts): every layer of a Llama, a Qwen2, a Qwen3, a GPT-2 and a GPT-NeoX, the full-attention layers of a Qwen3.5
// and the attention layers of an LFM2. A kind of layer of forward/kinds.ts's table; a module of src/runtime/forward.ts.
//
// What one of them has beside q, k, v and the output is read off the model's tensors: Qwen2's biases of q, k and v
// (bq), Qwen3's norms of every head of q and k (q_norm), a Qwen3.5's gate of the output (wg). Whether RoPE turns a
// layer's q and k is the plan's fact of that layer (T255: none of a GPT-2's, not the ones a SmolLM3 leaves alone).
import type { Core, Kind, Layers, Product } from "./kinds.ts";

export const attention: Kind = {
  // a Qwen3.5's gate of the output beside q; and the scores of one token's attention, apart from its frame (tokens run
  // at once, each with scores of its own)
  frame: ({ tensors, widths, seq_len: seqLen, n_heads: heads }) =>
    ({ arrays: tensors.wg ? [["gate", widths.q * 4]] : [], apart: [["att", seqLen * heads * 4]] }),
  make({ plan, eps, k, floats, matrix, at, S, matmuls, phase, cache }: Core, count: number): Layers {
    const { n_heads: heads, n_kv_heads: kvHeads, head_size: headSize, seq_len: seqLen, rotary } = plan, { q: qDim, kv: kvDim } = plan.widths;
    // kNow, vNow: this token's key and value in float32, before they go into the cache
    const { xb, xb2, q, kNow, vNow, gate, att } = at;
    const A = seqLen * heads * 4, KF = kvDim * 4, QF = qDim * 4, HS = headSize * 4, { KV } = cache;
    const wq = matrix("wq"), wk = matrix("wk"), wv = matrix("wv"), wo = matrix("wo");
    const bq = floats("bq"), bk = floats("bk"), bv = floats("bv");
    const qNorm = floats("q_norm"), kNorm = floats("k_norm");  // T124: Qwen3 normalizes every head of q and k
    const cosTable = floats("freq_cis_real"), sinTable = floats("freq_cis_imag");
    const wg = matrix("wg");  // T229
    // where these layers' keys and values are in the cache: the place-th layer's are the (first + place)-th
    const first = cache.hold(count);
    // the job (jobs.ts) of the attention of token t of a run, at position pos (T110: over float16 keys and values, or float32)
    const scores = cache.half ? 4 : 3;
    return {
      layer(a, { rope }) {
        // (a Qwen3.5's gate of the attention's output is one more matrix of the same input)
        const into: Product[] = wg ? [[wq, q, S, a], [wk, kNow, S, a], [wv, vNow, S, a], [wg, gate, S, a]] : [[wq, q, S, a], [wk, kNow, S, a], [wv, vNow, S, a]];
        const out: Product[] = [[wo, xb2, S, a]];
        const slot = first + a;
        return (count, pos0) => {
          const { capacity } = cache, layerKeys = cache.keys + slot * capacity * KV, layerValues = cache.values + slot * capacity * KV;
          const kp = layerKeys + pos0 * KV, vp = layerValues + pos0 * KV;
          matmuls(xb, count, into);
          for (let t = 0; t < count; t++) {
            const qt = q + t * S, kt = kNow + t * S, vt = vNow + t * S, pos = pos0 + t;
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
              const cos = cosTable + pos * (headSize / 2) * 4, sin = sinTable + pos * (headSize / 2) * 4;
              k.rope(qt, cos, sin, heads, headSize, rotary);
              k.rope(kt, cos, sin, kvHeads, headSize, rotary);
            }
            cache.put(kp + t * KV, vp + t * KV, kt, vt);  // into the cache, at this token's position
          }
          // the keys and values of positions up to each token's are all there now: its own and the ones before it.
          // The heads of every token go out as one phase (T109).
          const jobs = [];
          for (let t = 0; t < count; t++) {
            jobs.push([scores, xb + t * S, q + t * S, layerKeys, layerValues, att + t * A, pos0 + t, kvHeads, headSize, heads, 1, 0, 0, 0]);
          }
          phase(jobs);
          if (wg) for (let t = 0; t < count; t++) k.gate(xb + t * S, xb + t * S, gate + t * S, qDim);
          matmuls(xb, count, out);
        };
      },
    };
  },
};
