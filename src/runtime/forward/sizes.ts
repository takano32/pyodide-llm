// forward/sizes.ts (T.375.5): what sizes a model's memory, as one value, made of a plan or of a header and a form. A
// module of src/runtime/forward.ts.
//
// The engine sizes a token's frame and its layers' states as it is made of Python's plan; footprint() (memory.ts) says
// the same before the model is loaded, of the header and the form alone. Both ask the kinds of layers (kinds.ts's
// table: frame(), state()) with a Sizes, and differ only in where it comes from:
//   sizesOfPlan()  says again what the plan says (nothing is worked out a second time, T.375.2)
//   sizesOfForm()  works it out as Python does: which layer is of which kind and each kind's widths are the kinds' own
//                  (layers(), widths()), the gate is the architecture's line (architectures.ts)
// tests/plan-keys-check.mjs holds the two together for Python's plans: the same Sizes either way.
import { architectureOf } from "./architectures.ts";
import type { Kinds } from "./kinds.ts";
import type { Header } from "./memory.ts";
import type { Stateful } from "../contracts/form.ts";
import type { Plan, Widths } from "../contracts/plan.ts";

// dim, hidden (the FFN's inside: T.331, of the widest layer's), heads, seqLen: the header's. kinds: every layer's kind, by its name. widths: the
// plan's (contracts/plan.ts). gate: the layers that attend gate their output. numbers: of the kinds that keep a state,
// each under its kind's name (the form's "linear" and "convolution", which the plan carries by the same names)
export type Sizes = { dim: number; hidden: number; heads: number; seqLen: number; kinds: string[]; widths: Widths; gate: boolean; numbers: Stateful };
// what a form says of a model's sizes besides its header (contracts/form.ts's Form has each)
// (T.331: and the layers of wide heads, of an FFN twice as wide)
export type Formed = { arch?: string; head_dim?: number; wide?: { layers: number[]; head_dim: number } | null; double?: number[] | null } & Stateful;

// the kind of a layer that no other kind says is its own: it attends over every position before it
const ATTENTION = "attention";

export const sizesOfPlan = (plan: Plan): Sizes => ({ dim: plan.dim, hidden: plan.layers.some(({ double }) => double) ? 2 * plan.hidden_dim : plan.hidden_dim,
  heads: plan.n_heads, seqLen: plan.seq_len,
  kinds: plan.layers.map(({ kind }) => kind), widths: plan.widths, gate: Boolean(plan.tensors.wg ?? plan.tensors.head_gate), numbers: plan });

/** the kinds of the table a model has layers of, in the table's order (the order they are made in, and so of what they
 * take of the memory), each with the number of its layers */
export const kindsHad = (sizes: Sizes, kinds: Kinds) =>
  Object.keys(kinds).map((kind) => ({ kind, count: sizes.kinds.filter((had) => had === kind).length })).filter(({ count }) => count);

export function sizesOfForm(header: Header, form: Formed, kinds: Kinds): Sizes {
  const [dim, inside, layers, heads, kvHeads, , seqLen] = header;
  // (T.331: of the widest heads and the widest FFN, where the form says some layers' are wider)
  const headSize = Math.max(form.head_dim || dim / heads, form.wide?.head_dim ?? 0), q = heads * headSize, kv = kvHeads * headSize;
  const hidden = form.double?.length ? 2 * inside : inside;
  // every layer is of the first kind that says so of this form, and attends where none does
  const said = Object.entries(kinds).flatMap(([name, kind]): [string, boolean[]][] => {
    const own = kind.layers?.(form, layers);
    return own ? [[name, own]] : [];
  });
  const widths: Widths = Object.assign({ q, kv, linear: null, rotated: [] }, ...Object.values(kinds).map((kind) => kind.widths?.(form) ?? {}));
  const sizes: Sizes = { dim, hidden, heads, seqLen, kinds: Array.from({ length: layers }, (_, l) => said.find(([, own]) => own[l])?.[0] ?? ATTENTION),
    widths, gate: architectureOf(form.arch ?? "llama").gate, numbers: form };
  // T.237: the widths of what the model's matrices read, each of which a rotated basis has signs for: the residual
  // stream, the attention's output, the FFN's inside, and what a kind's own matrices read (its frame says)
  const read = kindsHad(sizes, kinds).flatMap(({ kind }) => kinds[kind].frame(sizes).widest ?? []);
  widths.rotated = [...new Set([dim, q, hidden, ...read])].sort((a, b) => a - b);
  return sizes;
}

export { ATTENTION };
