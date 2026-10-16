// forward/convolution.ts (T375.3; the layer is T260's): an LFM2's convolution layers, which keep no keys and values
// but the last taps tokens' rows. llama2_numpy's short_convolution() has the rule. A kind of layer of
// forward/kinds.ts's table; a module of src/runtime/forward.ts.
//
// Their tensors: a matrix in (win: 3 dim rows), the taps (conv) and a matrix out (wout). Their state is the
// convolution's rows alone, taps of dim values a layer, which position 0 clears.
import type { Kind, Layers } from "./kinds.ts";
import type { Sizes } from "./sizes.ts";

const convolutionNumbers = ({ numbers: { convolution } }: Sizes) => {
  if (!convolution) throw new Error("The plan has a layer that keeps a state, and not the numbers of the layers of its kind.");
  return convolution;
};
// the bytes of one layer's rows: the last taps tokens' values before the convolution, dim of them a token
const rowsBytes = (sizes: Sizes) => convolutionNumbers(sizes).taps * sizes.dim * 4;
// what count of them keep (the bytes of each): their rows
const kept = (sizes: Sizes, count: number) => [count * rowsBytes(sizes)];

export const convolution: Kind = {
  // an LFM2's form has a letter a layer: "c" for one of these, "a" for one that attends
  layers: ({ convolution }, count) => (convolution ? Array.from({ length: count }, (_, l) => convolution.layers[l] === "c") : null),
  // what the matrix in gives (mixed: the gates B and C and what B multiplies, 3 dim)
  frame(sizes) {
    convolutionNumbers(sizes);
    return { arrays: [["mixed", 3 * sizes.dim * 4]] };
  },
  state: kept,
  // T545: the GPU's worker has these layers too (make()'s rows)
  staged: true,
  make({ plan, sizes, k, room, floats, matrix, at, S, matmuls }, count): Layers {
    const { dim } = plan, { taps: tapsCount } = convolutionNumbers(sizes), D = dim * 4;
    const { xb, xb2, mixed } = at;
    const wout = matrix("wout"), taps = floats("conv"), win = matrix("win");
    const convBytes = rowsBytes(sizes), [convRows] = kept(sizes, count).map((bytes) => room.alloc(bytes));
    return {
      // T545: the GPU's worker takes these layers too (gpu/block.ts): a block goes on there from these rows, and gives
      // its own back here
      rows: { at: convRows, bytes: count * convBytes },
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
