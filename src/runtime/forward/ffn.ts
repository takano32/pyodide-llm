// forward/ffn.ts (T.375.3): the two FFNs a layer ends with, one of which the engine takes as it is made (the plan's
// gated_ffn, as engine/layout.py's Layout names the fact). Neither keeps a state: each is made of what the engine
// hands a kind of layer (kinds.ts's Core) and is the function a token's path calls for layer l: count tokens from
// at.xb (the norm of the stream) through the FFN, added to the stream (at.x). A module of src/runtime/forward.ts.
import type { Core } from "./kinds.ts";

export type Ffn = (l: number, count: number) => void;

/** w2(act(w1 x) * w3 x): every architecture but GPT-2's two. The activation of the gate is the plan's (a SiLU, or
 * T.261 a Gemma's GELU, or T.342 a Spark-X2.5's exact GELU), and what the FFN gives is normalized before it joins the stream where the model has the norm
 * for that (rms_ffn_post_weight: a Gemma's). T.331: a layer's FFN may be twice as wide, its matrices in stacks of their
 * own (the plan's facts of the layer: double, and its place among those, ffn) */
export function gatedFfn({ plan, eps, k, floats, matrix, at, S, matmuls }: Core): Ffn {
  const { dim, hidden_dim: inside } = plan, { x, xb, xb2, hb, hb2 } = at, D = dim * 4;
  if (!Object.hasOwn(GATES, plan.activation)) throw new Error(`This engine has no gated FFN whose activation is ${JSON.stringify(plan.activation)}.`);
  const gate = k[GATES[plan.activation]], post = floats("rms_ffn_post_weight");
  // every layer's matrices, its place in their stacks and its width, chosen here once
  const single = [matrix("w1"), matrix("w2"), matrix("w3")] as const;
  const double = plan.layers.some(({ double }) => double) ? [matrix("w1_double"), matrix("w2_double"), matrix("w3_double")] as const : null;
  const layers = plan.layers.map(({ double: twice, ffn }) => ({ m: twice ? double! : single, at: ffn, hidden: twice ? 2 * inside : inside }));
  return (l, count) => {
    const { m: [w1, w2, w3], at: a, hidden } = layers[l];
    matmuls(xb, count, [[w1, hb, S, a], [w3, hb2, S, a]]);
    for (let t = 0; t < count; t++) gate(hb + t * S, hb + t * S, hb2 + t * S, hidden);
    matmuls(hb, count, [[w2, xb2, S, a]]);
    for (let t = 0; t < count; t++) {
      if (post) k.rmsnorm(xb2 + t * S, xb2 + t * S, post + l * D, dim, eps);
      k.add_inplace(x + t * S, xb2 + t * S, dim);
    }
  };
}
// the kernel of a gated FFN's gate, by the activation's name (engine/layout.py's SILU, GELU and GELU_ERF)
const GATES = { silu: "swiglu", gelu: "geglu", gelu_erf: "geglu_erf" } as const;
/** GPT-2's and GPT-NeoX's: w2(gelu(w1 x + b1)) + b2 */
export function plainFfn({ plan, k, floats, matrix, at, S, matmuls }: Core): Ffn {
  const { dim, hidden_dim: hidden } = plan, { x, xb, xb2, hb } = at, D = dim * 4, HD = hidden * 4;
  if (plan.activation !== "gelu") throw new Error(`This engine has no FFN without a gate whose activation is ${JSON.stringify(plan.activation)}.`);
  const w1 = matrix("w1"), w2 = matrix("w2"), b1 = floats("b1"), b2 = floats("b2");
  return (l, count) => {
    matmuls(xb, count, [[w1, hb, S, l]]);
    for (let t = 0; t < count; t++) k.gelu(hb + t * S, hb + t * S, b1 + l * HD, hidden);
    matmuls(hb, count, [[w2, xb2, S, l]]);
    for (let t = 0; t < count; t++) {
      k.add_inplace(x + t * S, xb2 + t * S, dim);
      k.add_inplace(x + t * S, b2 + l * D, dim);
    }
  };
}
