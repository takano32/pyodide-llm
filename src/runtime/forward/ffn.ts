// forward/ffn.ts (T375.3): the two FFNs a layer ends with, one of which the engine takes as it is made (the plan's
// gated_ffn, as engine/layout.py's Layout names the fact). Neither keeps a state: each is made of what the engine
// hands a kind of layer (kinds.ts's Core) and is the function a token's path calls for layer l: count tokens from
// at.xb (the norm of the stream) through the FFN, added to the stream (at.x). A module of src/runtime/forward.ts.
import type { Core } from "./kinds.ts";

export type Ffn = (l: number, count: number) => void;

/** w2(silu(w1 x) * w3 x): every architecture but GPT-2's two */
export function gatedFfn({ plan, k, matrix, at, S, matmuls }: Core): Ffn {
  const { dim, hidden_dim: hidden } = plan, { x, xb, xb2, hb, hb2 } = at;
  const w1 = matrix("w1"), w2 = matrix("w2"), w3 = matrix("w3");
  return (l, count) => {
    matmuls(xb, count, [[w1, hb, S, l], [w3, hb2, S, l]]);
    for (let t = 0; t < count; t++) k.swiglu(hb + t * S, hb + t * S, hb2 + t * S, hidden);
    matmuls(hb, count, [[w2, xb2, S, l]]);
    for (let t = 0; t < count; t++) k.add_inplace(x + t * S, xb2 + t * S, dim);
  };
}
/** GPT-2's and GPT-NeoX's: w2(gelu(w1 x + b1)) + b2 */
export function plainFfn({ plan, k, floats, matrix, at, S, matmuls }: Core): Ffn {
  const { dim, hidden_dim: hidden } = plan, { x, xb, xb2, hb } = at, D = dim * 4, HD = hidden * 4;
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
