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
// values a scale is for (0: no scales); scales: where the scales are (of llama.cpp's blocks, T363: where the blocks end)
// T363.5: a row may be of a kind of its own (the form's kinds): one of llama.cpp's blocks by its type's name, 256
// weights a group with the scales inside; and "u8", the bytes of the codebooks a file with such rows carries. A stack
// with a kind for each layer says "layers" and, under layers, each layer's kind and place, one after the other (a key
// of such a stack alone)
export type Stored = "f32" | "f16" | "int8" | "int6" | "ternary" | "u8" | "layers"
  | "Q2_K" | "Q4_K" | "IQ2_XXS" | "IQ2_XS" | "IQ3_XXS" | "IQ1_S" | "IQ3_S" | "IQ2_S" | "IQ4_XS" | "IQ1_M";
export type Part = { kind: Stored; offset: number; group: number; scales: number };
export type Tensor = { kind: Stored; offset: number; shape: number[]; group: number; scales: number; layers?: Part[] };

// The rows of the layouts (engine/layout.py), by their names. A model has those of its own layout; the classifier
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
  | "win" | "wout"
  // Gemma 3: the norms behind the attention and behind the FFN (T261)
  | "rms_att_post_weight" | "rms_ffn_post_weight"
  // Gemma 4 (T331): the attention of the layers whose heads are wide, the FFN of those whose FFN is twice as wide, and
  // the number every layer ends by multiplying the stream by
  | "wq_wide" | "wk_wide" | "wv_wide" | "wo_wide" | "q_norm_wide" | "k_norm_wide" | "w1_double" | "w2_double" | "w3_double"
  | "layer_scale"
  // T329: a Gemma 4's embedding of every layer: the table (a row of it every token), the projection of the embedded
  // token and its norm, and every layer's gate, projection and norm of what it makes of its input
  | "per_layer_table" | "per_layer_projection" | "per_layer_norm" | "ple_gate" | "ple_proj" | "ple_post"
  // T363: the codebooks of llama.cpp's blocks, of a file with rows of such types
  | "codebooks";

// where every tensor of a model is: the plan's, and what Python's external_tensors() places before a model is built
export type Tensors = Partial<Record<TensorName, Tensor>>;

// The few arrays Python computes, as bytes: the RoPE tables of a file that has none of its own (float32), those of a
// model's local layers (T261, which no file has), and the signs of a rotated basis for every width a matrix reads (T237)
export type Derived = { freq_cis_real?: Uint8Array; freq_cis_imag?: Uint8Array; local_cis_real?: Uint8Array; local_cis_imag?: Uint8Array;
  [signs: `signs.${number}`]: Uint8Array };

// T359.5: what an engine has to know of a layer: its kind ("attention", or the layout's kind of a layer that keeps a
// state), its place among the layers of its kind, whether RoPE turns its q and k. T261: how many of the last
// positions it attends, its own among them (0: all of them), and whether RoPE turns it by the model's local tables
// (local_cis_real and local_cis_imag of the derived arrays) in place of the others: two facts, which a Gemma 3's
// sliding-attention layers have together
export type LayerKind = "attention" | "linear" | "convolution";
// T331: and the size of its heads (head), how many values of a head RoPE turns (rotary), whether its attention's tensors
// are the wide ones (wide: the stacks of the names with "_wide") and its place in its attention's stacks (stack); whether
// its FFN's are the ones twice as wide (double: "_double") and its place in its FFN's stacks (ffn). A layer of a model
// whose layers are all alike has the model's head and rotary, its place and its number
export type Layer = { kind: LayerKind; place: number; rope: boolean; window: number; local: boolean;
  head: number; rotary: number; wide: boolean; stack: number; double: boolean; ffn: number;
  // T330: its place in the stacks of k, v and the norm of k's heads (-1: it reads another layer's keys and values), and
  // the place among the layers of its kind of the layer whose keys and values it attends over (its own, or that one's)
  kept: number; reads: number };

// the widths of a linear-attention layer: what the convolution runs over, q or k, and v (which the output matrix reads)
export type LinearWidths = { mixed: number; keys: number; read: number };
// T359.5: the widths an engine sizes its buffers by, which are not in the header (T331: q and kv of the widest heads)
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
  // T363.5: a linear-attention layer's output matrix reads its value heads in llama.cpp's order (the form's tiled)
  tiled: boolean;
  // T363.5: some rows are stored in kinds of their own (the form's kinds): the matrices of one input may then take
  // their activations in different forms, and a stack may be its layers, each of a kind of its own
  own_kinds: boolean;
  // T359.5: every layer's facts, and the widths (forward.ts reads them from T375.2, and works none of them out)
  layers: Layer[];
  widths: Widths;
  // T375.2: the facts of the architecture the layers differ by, as engine/layout.py's Layout names them: the norms are
  // LayerNorms (else RMSNorms), the FFN is gated (three matrices; else GPT-2's two, with their biases); T261: and the
  // FFN's activation (a Llama's gate is a SiLU's, a Gemma's a GELU's, as GPT-2's one matrix's output)
  activation: "silu" | "gelu";
  layer_norm: boolean;
  gated_ffn: boolean;
  // T331: and the heads of v are normalized (a Gemma 4's, an RMSNorm without a weight); the logits are capped, softcap
  // * tanh(logits / softcap) (0: as they are)
  value_norm: boolean;
  softcap: number;
};
