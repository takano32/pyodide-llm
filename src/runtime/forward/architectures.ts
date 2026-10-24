// forward/architectures.ts (T.375.5): what an architecture's name says of a model's memory before the model is loaded,
// as one table. A module of src/runtime/forward.ts.
//
// The engine reads no architecture's name (T.375.2): Python's plan says the facts of the model it is made for. But what a
// model will take is asked before there is a plan, of its header and its form alone (forward/memory.ts's footprint(),
// forward/alone.ts: the worker as a conversion begins, /benchmark/, the tools), where no Python runs yet. So the facts
// that size a model are here a second time: this table is a copy of Python's (src/python/engine/layout.py's LAYOUTS),
// and tests/plan-keys-check.mjs holds every line to what Python's plans of that architecture say. The table of Python
// is the owner: where the two differ, this one is put right.
//
// A line says only what the form does not: which layers keep a state is the form's (its "linear", its "convolution":
// the kinds of forward/kinds.ts read them), and a bias or a norm of the heads weighs nothing here.
//   positions  the file has a learned table of positions (a row a position, no matrix), which a packed file leaves out
//   gate       the layers that attend gate their output (the row wg; T.342: head_gate, a value a head): a value more
//              a token's frame
//   rope       RoPE turns q and k: a float32 file has its two tables (else Python makes them, of zeros, at any dtype)
//   gatedFfn   the FFN is gated (three matrices); else GPT-2's (two)
//   layerNorm  the norms are LayerNorms; else RMSNorms
//   activation the FFN's activation (T.261): a SiLU's, or a GELU's (GPT-2's one matrix's output, a Gemma's gate), or
//              (T.342) the exact GELU's (a Spark-X2.5's gate)
//   local      some layers may be local ones (T.261: a window of the last positions, RoPE tables of their own)
import type { Arch } from "../contracts/form.ts";
import type { Activation } from "../contracts/plan.ts";

export type Architecture = { positions: boolean; gate: boolean; rope: boolean; gatedFfn: boolean; layerNorm: boolean; activation: Activation; local: boolean };
const LLAMA: Architecture = { positions: false, gate: false, rope: true, gatedFfn: true, layerNorm: false, activation: "silu", local: false };
export const ARCHITECTURES: Record<Arch, Architecture> = {
  llama: LLAMA,
  gpt2: { ...LLAMA, positions: true, rope: false, gatedFfn: false, layerNorm: true, activation: "gelu" },
  neox: { ...LLAMA, gatedFfn: false, layerNorm: true, activation: "gelu" },
  qwen35: { ...LLAMA, gate: true },
  lfm2: LLAMA,
  gemma3: { ...LLAMA, activation: "gelu", local: true },
  // (T.331: what sizes a Gemma 4 beyond a Gemma 3, its wide heads and its FFN twice as wide, is T.331.2's)
  gemma4: { ...LLAMA, activation: "gelu", local: true },
  spark: { ...LLAMA, gate: true, activation: "gelu_erf", local: true },
};
/** An architecture's line. A name that is none of the table's is sized as a Llama, as it always was: Python refuses it
 * by its own words once the model is built (layout_of()), and those are the words a visitor is to read. */
export const architectureOf = (arch: string): Architecture => (Object.hasOwn(ARCHITECTURES, arch) ? ARCHITECTURES[arch as Arch] : LLAMA);
