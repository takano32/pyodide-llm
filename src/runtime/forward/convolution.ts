// forward/convolution.ts (T375.3; the layer is T260's): an LFM2's convolution layers, which keep no keys and values
// but the last taps tokens' rows. llama2_numpy's short_convolution() has the rule. A kind of layer of
// forward/kinds.ts's table; a module of src/runtime/forward.ts.
//
// Their tensors: a matrix in (win: 3 dim rows), the taps (conv) and a matrix out (wout). Their state is the
// convolution's rows alone, taps of dim values a layer, which position 0 clears.
import type { Kind, Layers } from "./kinds.ts";
import type { Plan } from "../contracts/plan.ts";

const convolutionNumbers = ({ convolution }: Plan) => {
  if (!convolution) throw new Error("The plan has a layer that keeps a state, and not the numbers of the layers of its kind.");
  return convolution;
};

export const convolution: Kind = {
  // what the matrix in gives (mixed: the gates B and C and what B multiplies, 3 dim)
  frame(plan) {
    convolutionNumbers(plan);
    return { arrays: [["mixed", 3 * plan.dim * 4]] };
  },
  make({ plan, k, room, floats, matrix, at, S, matmuls }, count): Layers {
    const { dim } = plan, { taps: tapsCount } = convolutionNumbers(plan), D = dim * 4;
    const { xb, xb2, mixed } = at;
    const wout = matrix("wout"), taps = floats("conv"), win = matrix("win");
    const convBytes = tapsCount * dim * 4, convRows = room.alloc(count * convBytes);
    return {
      clear() {
        room.F.fill(0, convRows / 4, (convRows + count * convBytes) / 4);
      },
      // count tokens through the a-th layer, from xb (the norm of x) into xb2. The two matrices go out once for all the
      // tokens, as an attending layer's; between them a token at a time, for the rows after a token are what the next
      // one reads: the rows move up by one, and the kernel writes this token's (B * z) and the convolution times C over
      // xb. Only this thread touches the rows, outside every phase: a phase that is run again (T120) reads what it read.
      layer: (a) => (count) => {
        const { F } = room, rows = convRows + a * convBytes;
        matmuls(xb, count, [[win, mixed, S, a]]);
        for (let t = 0; t < count; t++) {
          F.copyWithin(rows / 4, (rows + D) / 4, (rows + convBytes) / 4);
          k.short_conv(xb + t * S, taps + a * convBytes, rows, mixed + t * S, dim, tapsCount);
        }
        matmuls(xb, count, [[wout, xb2, S, a]]);
      },
    };
  },
};
