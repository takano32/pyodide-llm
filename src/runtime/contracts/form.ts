// contracts/form.ts (T.368.2): what a checkpoint's file cannot say of itself, its form. Types alone: nothing of this
// file is in what a browser gets (it is read with `import type`).
//
// The form is Python's (src/python/engine/layout.py's FORM, with each key's default there): the converter makes it
// (convert/plan.py's checkpoint_form()), a conversion's options and a kept model's manifest carry it, the worker takes it
// from the converter's sink.open() (worker/weights.ts) and hands it on by the same names to forward.ts's footprint()
// and to external_tensors(). tests/worker-sink-check.mjs holds these keys to FORM's and follows them to footprint().
//
// A key that is added here is added in layout.py's FORM first (AGENTS.md, "形式を変えるとき").

// how a row of the file is stored (src/python/engine/dtypes.py's DTYPES; a file has one for its matrices)
export type Dtype = "float32" | "float16" | "int8" | "int6" | "ternary";

// T.229: the linear-attention layers of a Qwen3.5 (layout.py's LINEAR): every "every"-th layer attends over all the
// positions, the others keep a state of key_heads and value_heads heads of these sizes, after a convolution of conv taps
export type Linear = { every: number; key_heads: number; value_heads: number; key_dim: number; value_dim: number; conv: number };

// T.260: the layers of an LFM2: a letter a layer ("c" a convolution layer, "a" one that attends) and the taps
export type Convolution = { layers: string; taps: number };

// T.237: a rotated basis: the block of the Walsh-Hadamard transform and, for every width a matrix reads, its signs
// (bits; the plan carries them as bytes under derived, and of this only the block)
export type Rotated = { block: number; signs: Record<string, unknown> };

// the architecture's name, a key of layout.py's LAYOUTS
export type Arch = "llama" | "gpt2" | "neox" | "qwen35" | "lfm2" | "gemma3" | "gemma4" | "spark";

// The numbers of the layers that keep a state, each under its kind's name (forward/kinds.ts): what a form says of
// them, and a plan after it by the same names (contracts/plan.ts)
// (tiled, T.363: of the linear-attention layers, whose output matrix then reads its value heads in llama.cpp's order)
export type Stateful = { linear?: Linear | null; convolution?: Convolution | null; tiled?: boolean | null };

export type Form = {
  // Qwen2: q, k and v have a bias (and every matrix of a GPT-2)
  bias: boolean;
  arch: Arch;
  // Qwen3: a norm of every head of q and of k
  qk_norm: boolean;
  // the size of a head where it is not dim / heads (T.124); 0 where it is
  head_dim: number;
  linear: Linear | null;
  rotated: Rotated | null;
  convolution: Convolution | null;
  // T.359: the rows stored otherwise than the file's dtype stores a row of their role, each by its kind's name (a
  // dtype's, or T.363 one of llama.cpp's blocks: engine/dtypes.py's KINDS), a stack's layers each by a list of them
  kinds: Record<string, string | string[]> | null;
  // T.363: a linear-attention layer's output matrix has its columns in llama.cpp's order of the value heads (the same)
  tiled: boolean | null;
  // T.331: the layers whose heads are of a size of their own (a Gemma 4's full-attention layers), and those whose FFN is
  // twice as wide: layout.py's wide_form() and double_form()
  wide: { layers: number[]; head_dim: number } | null;
  double: number[] | null;
  // T.330: the layers that read the keys and values of another, and which each reads: layout.py's shared_form()
  kv_shared: { layers: number[]; from: number[] } | null;
  // T.329: the width of every layer's input of a table of its own (a Gemma 4's per-layer embedding); 0 where none
  per_layer: number;
};
