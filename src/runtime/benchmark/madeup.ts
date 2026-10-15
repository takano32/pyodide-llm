// benchmark/madeup.ts (T375.2): the made-up model /benchmark/'s CPU section runs, and its plan. A module of
// src/runtime/benchmark/sections.ts, with nothing of a worker in it, so that a test reads it too.
//
// A model of Llama 3.2 1B's width (dim 2048, FFN 8192, 32 heads of 64, 8 of them for keys and values) with two layers
// and a vocabulary of 32000: 211 MB of int8 weights and their scales, which is what a token reads, whatever the
// numbers in them.
//
// The plan is what Python makes for a model (src/python/engine/plan.py's forward_plan()), written here by hand: the
// section runs without Pyodide. It is a whole Plan (contracts/plan.ts), every key of it, and
// tests/plan-keys-check.mjs holds it to the plan Python makes for the same header: a key or a fact the engine comes
// to read that is not here fails the type check, and one that is here and wrong fails that check. It differs from
// Python's in two things: the classifier is under the embedding's name alone (Python has the one table under "wcls"
// too; the engine reads the embedding's for a shared classifier), and the RoPE tables are computed here.
import type { Plan, Tensor } from "../contracts/plan.ts";

export const SHAPE = { dim: 2048, hidden: 8192, layers: 2, heads: 32, kvHeads: 8, vocab: 32000, seqLen: 64 };
export const GROUP = 32;

export type MadeName = "token_embedding_table" | "rms_att_weight" | "wq" | "wk" | "wv" | "wo" | "rms_ffn_weight" | "w1" | "w2" | "w3" | "rms_final_weight";
type MadeUp = Omit<Plan, "tensors"> & { tensors: Record<MadeName, Tensor> };
export type Model = ReturnType<typeof madeUpModel>;

export function madeUpModel() {
  const { dim, hidden, layers, heads, kvHeads, vocab, seqLen } = SHAPE;
  const headSize = dim / heads, kvDim = kvHeads * headSize;
  const tensors = {} as MadeUp["tensors"];
  let offset = 28;  // after the legacy header, as in a checkpoint file
  const int8 = (name: MadeName, shape: number[]) => {
    const count = shape.reduce((a, b) => a * b, 1);
    tensors[name] = { kind: "int8", offset, shape, group: GROUP, scales: offset + count };
    offset += count + (count / GROUP) * 4;
  };
  // (a float32 tensor has no groups and no scales)
  const f32 = (name: MadeName, shape: number[]) => {
    tensors[name] = { kind: "f32", offset, shape, group: 0, scales: 0 };
    offset += shape.reduce((a, b) => a * b, 1) * 4;
  };
  int8("token_embedding_table", [vocab, dim]);
  f32("rms_att_weight", [layers, dim]);
  int8("wq", [layers, dim, dim]);
  int8("wk", [layers, kvDim, dim]);
  int8("wv", [layers, kvDim, dim]);
  int8("wo", [layers, dim, dim]);
  f32("rms_ffn_weight", [layers, dim]);
  int8("w1", [layers, hidden, dim]);
  int8("w2", [layers, dim, hidden]);
  int8("w3", [layers, hidden, dim]);
  f32("rms_final_weight", [dim]);
  // RoPE's tables, which Python computes for an int8 file (derived: bytes of float32)
  const cos = new Float32Array(seqLen * headSize / 2), sin = new Float32Array(seqLen * headSize / 2);
  for (let pos = 0; pos < seqLen; pos++) {
    for (let i = 0; i < headSize / 2; i++) {
      const angle = pos / 10000 ** ((2 * i) / headSize);
      cos[pos * headSize / 2 + i] = Math.cos(angle);
      sin[pos * headSize / 2 + i] = Math.sin(angle);
    }
  }
  const plan: MadeUp = { arch: "llama", dim, hidden_dim: hidden, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads,
    head_size: headSize, vocab_size: vocab, seq_len: seqLen, rotary: headSize, parallel_residual: false, kv_start: seqLen,
    rms_norm_eps: 1e-5, shared_classifier: true, int8: true, relaxed: true, tensors, outliers: [], half_kv: true,
    derived: { freq_cis_real: new Uint8Array(cos.buffer), freq_cis_imag: new Uint8Array(sin.buffer) },
    // a Llama's facts: every layer attends, the i-th of them, and RoPE turns its q and k; RMSNorms and a gated FFN; no
    // layer that keeps a state, none RoPE leaves alone, the model's own basis (rotated: the widths its matrices read)
    linear: null, convolution: null, unturned: [], rotated: 0,
    // (T331: every layer's heads and FFN the model's, at its place in the stacks)
    layers: Array.from({ length: layers }, (_, place) => ({ kind: "attention" as const, place, rope: true, window: 0, local: false,
      head: headSize, rotary: headSize, wide: false, stack: place, double: false, ffn: place })),
    widths: { q: heads * headSize, kv: kvDim, linear: null, rotated: [...new Set([dim, heads * headSize, hidden])].sort((a, b) => a - b) },
    layer_norm: false, gated_ffn: true, activation: "silu" as const, tiled: false, own_kinds: false, value_norm: false, softcap: 0 };
  return { plan, size: offset, header: [dim, hidden, layers, heads, kvHeads, vocab, seqLen] };
}
