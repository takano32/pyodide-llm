// forward/linear.ts (T375.3; the layer is T229's): a Qwen3.5's linear-attention layers (Gated DeltaNet), which keep
// no keys and values but a state of a fixed size. llama2_numpy's linear_attention() has the rule. A kind of layer of
// forward/kinds.ts's table; a module of src/runtime/forward.ts.
//
// Their tensors: q, k and v in one matrix (wqkv), z, the output (wout); in float32 the two small matrices of the gates
// (wb, wa), the taps of the convolution, dt_bias, the decay and the norm of a value head.
// Their state: for each layer a matrix (keyDim, valueDim) a value head, twice (flips: which of the two holds the
// state now; the delta rule reads it and writes the other, T120: a phase that is run again after a software thread
// stopped then reads what it read; position 0 clears both), and the last conv tokens' q, k and v before the
// convolution (the oldest first, the token under way last). And the two weights rmsnorm takes the l2 norm of a head
// with: x / sqrt(sum(x * x) + 1e-6) is rmsnorm's w * x / sqrt(mean(x * x) + eps) with w = 1 / sqrt(n) and
// eps = 1e-6 / n; q is divided by sqrt(n) more.
import type { Kind, Layers } from "./kinds.ts";
import type { Plan } from "../contracts/plan.ts";

// the numbers of the linear-attention layers and their widths (the plan's: mixed, what the convolution runs over;
// read, what the output matrix reads)
const linearNumbers = ({ linear, widths }: Plan) => {
  if (!linear || !widths.linear) throw new Error("The plan has a layer that keeps a state, and not the numbers of the layers of its kind.");
  return { ...linear, ...widths.linear };
};

export const linear: Kind = {
  // q, k and v before and after the convolution, z, and the delta rule's work (beta and decay of every value head,
  // then its delta); xb holds what the delta rule reads (as wide as v)
  frame(plan) {
    const { mixed, read, value_heads: valueHeads } = linearNumbers(plan);
    return { arrays: [["mixed", mixed * 4], ["conv", mixed * 4], ["z", read * 4], ["work", (2 * valueHeads + read) * 4]], widest: read };
  },
  make({ plan, eps, k, room, floats, matrix, at, S, matmuls, phase }, count): Layers {
    const { key_heads: keyHeads, value_heads: valueHeads, key_dim: keyDim, value_dim: valueDim, conv: tapsCount, mixed: mixedWidth, read: readWidth } = linearNumbers(plan);
    const { dim } = plan, { xb, xb2, mixed, conv, z, work } = at, D = dim * 4;
    const wqkv = matrix("wqkv"), wz = matrix("wz"), wout = matrix("wout");
    const wb = floats("wb"), wa = floats("wa"), taps = floats("conv"), dtBias = floats("dt_bias"), decays = floats("decay");
    const deltaNorm = floats("delta_norm");
    const stateBytes = valueHeads * keyDim * valueDim * 4, convBytes = tapsCount * mixedWidth * 4;
    const states = room.alloc(2 * count * stateBytes), convRows = room.alloc(count * convBytes);
    const qUnit = room.alloc(keyDim * 4), kUnit = room.alloc(keyDim * 4);
    const flips = new Uint8Array(count);
    room.F.fill(1 / keyDim, qUnit / 4, qUnit / 4 + keyDim);
    room.F.fill(1 / Math.sqrt(keyDim), kUnit / 4, kUnit / 4 + keyDim);
    const C = mixedWidth * 4, KD = keyDim * 4, VD = valueDim * 4, l2 = 1e-6 / keyDim;
    return {
      clear() {
        const { F } = room;
        F.fill(0, states / 4, (states + 2 * count * stateBytes) / 4);
        F.fill(0, convRows / 4, (convRows + count * convBytes) / 4);
      },
      // count tokens through the a-th layer, from xb (the norm of x) into xb2, as the output projection of an attending
      // layer leaves it. The matrices go out once for all the tokens, as an attending layer's; the rest is a token at a
      // time, for the state after a token is what the next one reads. The value heads of a token are one phase, shared
      // out as the heads of an attention.
      layer: (a) => (count) => {
        const { F } = room;
        const rows = convRows + a * convBytes, newest = rows + (tapsCount - 1) * C;
        matmuls(xb, count, [[wqkv, mixed, S, a], [wz, z, S, a]]);
        for (let t = 0; t < count; t++) {
          const xt = xb + t * S, ct = conv + t * S, wt = work + t * S;
          // beta = sigmoid(wb x) and decay = exp(decay * softplus(wa x + dt_bias)), one of each a value head, before
          // what the layer reads goes where its input is. A few numbers a layer: JavaScript's own exp and log
          k.matmul_f32(wt, xt, wb + a * valueHeads * D, dim, 0, valueHeads);
          k.matmul_f32(wt + valueHeads * 4, xt, wa + a * valueHeads * D, dim, 0, valueHeads);
          for (let h = 0; h < valueHeads; h++) {
            const b = wt / 4 + h, g = b + valueHeads, head = a * valueHeads + h;
            const step = F[g] + F[dtBias / 4 + head];
            F[b] = 1 / (1 + Math.exp(-F[b]));
            F[g] = Math.exp(F[decays / 4 + head] * (step > 20 ? step : Math.log1p(Math.exp(step))));
          }
          // this token's q, k and v behind those of the conv - 1 tokens before it, and the convolution over them
          F.copyWithin(rows / 4, (rows + C) / 4, (rows + convBytes) / 4);
          F.copyWithin(newest / 4, (mixed + t * S) / 4, (mixed + t * S + C) / 4);
          k.convolve(ct, taps + a * convBytes, rows, mixedWidth, tapsCount);
          for (let h = 0; h < keyHeads; h++) {
            k.rmsnorm(ct + h * KD, ct + h * KD, qUnit, keyDim, l2);
            k.rmsnorm(ct + (keyHeads + h) * KD, ct + (keyHeads + h) * KD, kUnit, keyDim, l2);
          }
          const now = states + (2 * a + flips[a]) * stateBytes, next = states + (2 * a + 1 - flips[a]) * stateBytes;
          phase([[7, xt, now, next, ct, wt, keyHeads, keyDim, valueDim, valueHeads, 1, 0, 0, 0]]);
          flips[a] ^= 1;
          // the norm of every value head, and z's gate
          for (let h = 0; h < valueHeads; h++) k.rmsnorm(xt + h * VD, xt + h * VD, deltaNorm + a * VD, valueDim, eps);
          k.swiglu(xt, z + t * S, xt, readWidth);
        }
        matmuls(xb, count, [[wout, xb2, S, a]]);
      },
    };
  },
};
