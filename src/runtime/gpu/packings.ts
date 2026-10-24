// gpu/packings.ts (T.547.2.1): how the model's GPU worker holds each packing a matrix or a table of its plan may be of
// (contracts/gpu/protocol.ts's GpuPacking, jobs.ts's PACKED names), as one table: what device.ts, weights.ts and
// tokenforms.ts compared the packing's name for. A packing the GPU takes is a line here (and its line of PACKED's gpu).
// (a module of the model's GPU worker, gpu.ts, which a bundler links it into)
import { sixValues } from "../shaders/steps.ts";
import { ternaryValues } from "../shaders/prompt.ts";
import type { GpuPacking } from "../contracts/gpu/protocol.ts";

// What a packing is to the GPU:
//   row()     the bytes of a row of n weights on the GPU: a byte a weight (int8, and int6 widened), or two bits
//             (ternary: the codes as the checkpoint holds them, 16 weights a u32). A float32 scale goes with every
//             GROUP (32) of those bytes either way: an int8 group of 32 weights, a ternary group of 128
//   stored()  the bytes of the row in the checkpoint (int6: 24 a group of 32)
//   scale     the bytes of a group's scale in the checkpoint: a float32, or (T.533: ternary's) a float16, which goes up
//             widened to a float32
//   widened   whether its values are widened to int8 as they go up (int6, T.155: WIDEN_SIX)
//   values()  the bytes of a row as the checkpoint holds them, as its int8 values (the checks' answer)
export type HeldAs = { row(n: number): number; stored(n: number): number; scale: 2 | 4; widened: boolean; values(bytes: Uint8Array): Int8Array };
const asBytes = (bytes: Uint8Array) => new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length);
export const PACKINGS: Record<GpuPacking, HeldAs> = {
  int8: { row: (n) => n, stored: (n) => n, scale: 4, widened: false, values: asBytes },
  int6: { row: (n) => n, stored: (n) => (n * 3) / 4, scale: 4, widened: true, values: sixValues },
  ternary: { row: (n) => n / 4, stored: (n) => n / 4, scale: 2, widened: false, values: ternaryValues },
};
