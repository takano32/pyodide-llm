// contracts/plan.ts (T368.2): the plan Python hands forward.ts, what a model is to an engine outside Python. Types
// alone: nothing of this file is in what a browser gets (it is read with `import type`).
//
// Python makes it in one place (src/python/engine/plan.py's forward_plan(), a dict; Llama(external=) hands it to
// start(plan), which is forward.ts's external()), and createForward() (src/runtime/forward/engine.ts) reads it. The two
// ends are in two languages, so no compiler holds them together: tests/plan-keys-check.mjs does, and it holds the keys
// written here to Python's too (a key added or dropped on one side alone fails the light suite).
//
// A key that is added: plan.py's dict, this file, and whatever reads it in forward/ (AGENTS.md, "計画を値にして渡すとき").
import type { Arch, Convolution, Linear } from "./form.ts";

// Where a tensor of the checkpoint is (engine/checkpoint.py's Tensor.plan()), counted from the start of the file:
// kind: how its values are stored (int8, int6 and ternary: values, then a float32 scale a group at scales); group: the
// values a scale is for (0: no scales); scales: where the scales are
export type Stored = "f32" | "f16" | "int8" | "int6" | "ternary";
export type Tensor = { kind: Stored; offset: number; shape: number[]; group: number; scales: number };

// The rows of the four layouts (engine/layout.py), by their names. A model has those of its own layout; the classifier
// of a model that has none of its own is its embedding under both names.
export type TensorName =
  | "token_embedding_table" | "wcls" | "rms_final_weight" | "freq_cis_real" | "freq_cis_imag"
  | "rms_att_weight" | "rms_ffn_weight" | "wq" | "wk" | "wv" | "wo" | "w1" | "w2" | "w3"
  // Qwen2's biases and Qwen3's norms of a head
  | "bq" | "bk" | "bv" | "q_norm" | "k_norm"
  // GPT-2 and GPT-NeoX: the positions, LayerNorm's biases, a bias after every matrix
  | "positions" | "ln_att_bias" | "ln_ffn_bias" | "ln_final_bias" | "bo" | "b1" | "b2"
  // Qwen3.5: a full-attention layer's gate, and a linear-attention layer's matrices, convolution and numbers (T229)
  | "wg" | "wqkv" | "wz" | "wb" | "wa" | "conv" | "dt_bias" | "decay" | "delta_norm"
  // LFM2: a convolution layer's two matrices (T260; its taps are under conv too)
  | "win" | "wout";

// where every tensor of a model is: the plan's, and what Python's external_tensors() places before a model is built
export type Tensors = Partial<Record<TensorName, Tensor>>;

// The few arrays Python computes, as bytes: the RoPE tables of a file that has none of its own (float32), and the
// signs of a rotated basis for every width a matrix reads (T237)
export type Derived = { freq_cis_real?: Uint8Array; freq_cis_imag?: Uint8Array; [signs: `signs.${number}`]: Uint8Array };

// T359.5: what an engine has to know of a layer: its kind ("attention", or the layout's kind of a layer that keeps a
// state), its place among the layers of its kind, whether RoPE turns its q and k
export type LayerKind = "attention" | "linear" | "convolution";
export type Layer = { kind: LayerKind; place: number; rope: boolean };

// the widths of a linear-attention layer: what the convolution runs over, q or k, and v (which the output matrix reads)
export type LinearWidths = { mixed: number; keys: number; read: number };
// T359.5: the widths an engine sizes its buffers by, which are not in the header
export type Widths = { q: number; kv: number; linear: LinearWidths | null; rotated: number[] };

export type Plan = {
  // the architecture's name, for a person who looks at a plan: no engine reads it (T375.2: its facts are below)
  arch: Arch;
  // the header's numbers (head_size: the form's head_dim, or dim / n_heads)
  dim: number;
  hidden_dim: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  head_size: number;
  vocab_size: number;
  seq_len: number;
  // how many values of a head RoPE turns
  rotary: number;
  parallel_residual: boolean;
  // the positions the keys and values begin with room for
  kv_start: number;
  rms_norm_eps: number;
  shared_classifier: boolean;
  // the int8 kernels compute on the matrices as they are
  int8: boolean;
  relaxed: boolean;
  tensors: Tensors;
  derived: Derived;
  // T92: the channels of the classifier's input that go as float32
  outliers: number[];
  linear: Linear | null;
  // T255: the layers RoPE leaves alone
  unturned: number[];
  convolution: Convolution | null;
  // T237: the block of a rotated basis; 0: the model's own basis
  rotated: number;
  // T110: the keys and values may be float16
  half_kv: boolean;
  // T359.5: every layer's facts, and the widths (forward.ts reads them from T375.2, and works none of them out)
  layers: Layer[];
  widths: Widths;
  // T375.2: the facts of the architecture the layers differ by, as engine/layout.py's Layout names them: the norms are
  // LayerNorms (else RMSNorms), the FFN is gated (three matrices; else GPT-2's two, with their biases and a GELU)
  layer_norm: boolean;
  gated_ffn: boolean;
};
