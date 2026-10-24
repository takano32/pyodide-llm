// gpu/packings.ts (T.547.2.1): how the model's GPU worker holds each packing a matrix or a table of its plan may be of
// (contracts/gpu/protocol.ts's GpuPacking, jobs.ts's PACKED names), as one table: what device.ts, weights.ts and
// tokenforms.ts compared the packing's name for. A packing the GPU takes is a line here (and its line of PACKED's gpu).
// (a module of the model's GPU worker, gpu.ts, which a bundler links it into)
import { GROUP } from "../shaders/common.ts";
import { sixValues } from "../shaders/steps.ts";
import { ternaryValues, BLOCK_READS, BLOCK_WEIGHTS } from "../shaders/prompt.ts";
import { BLOCK_PACKINGS } from "../shaders/common.ts";
import type { BlockPacking } from "../shaders/common.ts";
import type { GpuPacking, Matrix } from "../contracts/gpu/protocol.ts";

// What a packing is to the GPU:
//   row()     the bytes of a row of n weights on the GPU: a byte a weight (int8, and int6 widened), two bits (ternary:
//             the codes as the checkpoint holds them, 16 weights a u32), or (T.547.2.3) Q4_K's blocks of 256 weights in
//             144 bytes as the checkpoint holds them (n a multiple of 256: the CPU keeps a matrix as blocks only then)
//   scales()  the float32 scales of the row on the GPU (T.547.2.2), in a buffer of their own: one with every GROUP bytes
//             of row() (a group's, whatever its packing: an int8 group of 32 weights, a ternary group of 128), or none
//             where a block holds its own (Q4_K)
//   stored()  the bytes of the row in the checkpoint (int6: 24 a group of 32)
//   scale     the bytes of a group's scale in the checkpoint: a float32, or (T.533: ternary's) a float16, which goes up
//             widened to a float32; 0: none apart from the values (Q4_K)
//   widened   whether its values are widened to int8 as they go up (int6, T.155: WIDEN_SIX)
//   dot       whether its shaders need WGSL's packed int8 dot (PACKED's gpu.dot: a device without it keeps the model on
//             the CPU)
//   values()  the bytes of a row as the checkpoint holds them, as its int8 values (the checks' answer), which a group's
//             scale times is the row's weights; (T.547.2.4) a packing of no scales apart, its weights themselves (Q4_K's
//             d s q - dmin m), which its one scale of 1 times is
export type HeldAs = { row(n: number): number; scales(n: number): number; stored(n: number): number; scale: 0 | 2 | 4; widened: boolean; dot: boolean;
  values(bytes: Uint8Array): ArrayLike<number> };
const asBytes = (bytes: Uint8Array) => new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length);
// (T.547.3.3: llama.cpp's blocks of a type of prompt.ts's BLOCK_READS, as the checkpoint holds them)
const blocks = (packing: BlockPacking): HeldAs => {
  const { bytes } = BLOCK_READS[packing], { weights } = BLOCK_WEIGHTS[packing], row = (n: number) => (n / 256) * bytes;
  return { row, scales: () => 0, stored: row, scale: 0, widened: false, dot: true, values: weights };
};
export const PACKINGS: Record<GpuPacking, HeldAs> = {
  int8: { row: (n) => n, scales: (n) => n / GROUP, stored: (n) => n, scale: 4, widened: false, dot: false, values: asBytes },
  int6: { row: (n) => n, scales: (n) => n / GROUP, stored: (n) => (n * 3) / 4, scale: 4, widened: true, dot: false, values: sixValues },
  ternary: { row: (n) => n / 4, scales: (n) => n / 4 / GROUP, stored: (n) => n / 4, scale: 2, widened: false, dot: true, values: ternaryValues },
  Q4_K: blocks("Q4_K"),
  Q2_K: blocks("Q2_K"),
};
// T.547.3.1: the packing of a matrix's layer by its place (a stack whose layers are each of a packing of its own, packing
// "layers": its packings'), and the packings a matrix has
type Packed = Pick<Matrix, "packing" | "packings">;
export const packingAt = ({ packing, packings }: Packed, place: number): GpuPacking => (packing === "layers" ? packings![place] : packing);
export const packingsOf = ({ packing, packings }: Packed): GpuPacking[] => (packing === "layers" ? [...new Set(packings!)] : [packing]);
// T.547.3.3: whether a packing is one of llama.cpp's blocks the GPU reads (a type of BLOCK_READS)
export const isBlock = (packing: GpuPacking): packing is BlockPacking => (BLOCK_PACKINGS as readonly string[]).includes(packing);
// T.547.3.3: the types of llama.cpp's blocks the matrices have, in BLOCK_PACKINGS' order (the texts a model compiles)
export const blocksOf = (matrices: Record<string, Packed>): BlockPacking[] =>
  BLOCK_PACKINGS.filter((packing) => Object.values(matrices).some((matrix) => packingsOf(matrix).includes(packing)));
