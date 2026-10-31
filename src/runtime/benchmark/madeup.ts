// benchmark/madeup.ts (T.375.2): the made-up model /benchmark/'s CPU section runs, and its plan. A module of
// src/runtime/benchmark/sections.ts, with nothing of a worker in it, so that a test reads it too.
//
// A model of Llama 3.2 1B's width (dim 2048, FFN 8192, 32 heads of 64, 8 of them for keys and values) with two layers
// and a vocabulary of 32000: 211 MB of int8 weights and their scales, which is what a token reads, whatever the
// numbers in them.
//
// The plan is what Python makes for a model (src/python/engine/plan.py's forward_plan()). The section runs without
// Pyodide, so T.628.5: python_facts.py makes it at build time, of the header there (python_facts.py's MADEUP), and
// writes it into src/runtime/built/facts/madeup.ts (the classifier under the embedding's name alone, the RoPE tables'
// bytes in base64). What the section chooses for itself is laid over it here: kv_start, its context (Python's plan
// grows the keys and values from 256). tests/plan-keys-check.ts still holds it to the plan Python makes.
import { PYTHON_MADEUP } from "../built/facts/madeup.ts";
import type { Plan, Tensor } from "../contracts/plan.ts";

const [dim, hidden, layers, heads, kvHeads, vocab, seqLen] = PYTHON_MADEUP.header;
export const SHAPE = { dim, hidden, layers, heads, kvHeads, vocab, seqLen };
export const GROUP = PYTHON_MADEUP.plan.tensors.wq.group;

export type MadeName = keyof typeof PYTHON_MADEUP.plan.tensors;
type MadeUp = Omit<Plan, "tensors"> & { tensors: Record<MadeName, Tensor> };
export type Model = ReturnType<typeof madeUpModel>;

const bytesOf = (text: string) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0));

export function madeUpModel() {
  const { derived, ...plain } = PYTHON_MADEUP.plan;
  const plan = { ...plain, kv_start: seqLen,
    derived: { freq_cis_real: bytesOf(derived.freq_cis_real), freq_cis_imag: bytesOf(derived.freq_cis_imag) } } as unknown as MadeUp;
  return { plan, size: PYTHON_MADEUP.size, header: [...PYTHON_MADEUP.header] };
}
