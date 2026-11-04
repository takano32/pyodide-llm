// gpu/packings.ts (T.547.2.1): how the model's GPU worker holds each packing a matrix or a table of its plan may be of
// (contracts/gpu/protocol.ts's GpuPacking, jobs.ts's PACKED names), as one table: what device.ts, weights.ts and
// tokenforms.ts compared the packing's name for. A packing the GPU takes is a line here (and its line of PACKED's gpu).
// (a module of the model's GPU worker, gpu.ts, which a bundler links it into)
import { GROUP } from "../shaders/common.ts";
import { sixValues } from "../shaders/steps.ts";
import { ternaryValues, BLOCK_READS, BLOCK_WEIGHTS, readsGrid } from "../shaders/prompt.ts";
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
//             d s q - dmin m), which its one scale of 1 times is; (T.547.3.4) of an IQ type, of the codebooks too
//   grid      (T.547.3.4) whether its shaders read the codebooks, bound where an int8 matrix's scales are (the IQ types)
//   on        (T.634.3) the packing it is on the GPU, whose shaders multiply it and whose bytes a read back is in: its
//             own, but binary's, which goes up as ternary codes (spread)
//   spread    (T.634.3) whether its bytes are spread on the CPU as they go up (binary's bits to ternary's codes: spreadBits())
export type HeldAs = { row(n: number): number; scales(n: number): number; stored(n: number): number; scale: 0 | 2 | 4; widened: boolean; dot: boolean;
  grid: boolean; values(bytes: Uint8Array, codebooks?: Uint8Array): ArrayLike<number>; on: GpuPacking; spread: boolean };
const asBytes = (bytes: Uint8Array) => new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length);
// (T.547.3.3: llama.cpp's blocks of a type of prompt.ts's BLOCK_READS, as the checkpoint holds them)
const blocks = (packing: BlockPacking): HeldAs => {
  const { bytes } = BLOCK_READS[packing], { weights } = BLOCK_WEIGHTS[packing], row = (n: number) => (n / 256) * bytes;
  return { row, scales: () => 0, stored: row, scale: 0, widened: false, dot: true, grid: readsGrid(packing), values: weights, on: packing, spread: false };
};
// T.634.3: binary (jobs.ts's PACKED's binary, llama2_numpy.pack_binary): a group of 128 weights in 16 bytes, weight
// 64 h + 4 c + p at bit 4 h + p of byte c, set for +1. On the GPU it is ternary's codes (PQ2_0's order: weight j's code,
// the weight + 1, in byte j >> 2 at bits 2 (j & 3)), which ternary's shaders multiply as they are: byte c's low four bits
// are weights 4 c to 4 c + 3, ternary's byte c, and its high four weights 64 + 4 c on, ternary's byte 16 + c. A bit b is the
// code 2 b (+1: 2, -1: 0), so a nibble's four codes are a table of 16 (CODES). The scales are ternary's, float16s.
export const CODES = Uint8Array.from({ length: 16 }, (_, x) => [0, 1, 2, 3].reduce((byte, p) => byte | ((((x >> p) & 1) * 2) << (2 * p)), 0));
/** whole groups of binary's bytes (16 a group) as ternary's codes (32 a group), into out (twice as long) */
export function spreadBits(bits: Uint8Array, out = new Uint8Array(bits.length * 2)) {
  for (let g = 0; g < bits.length; g += 16) {
    for (let c = 0; c < 16; c++) {
      const v = bits[g + c];
      out[2 * g + c] = CODES[v & 15];
      out[2 * g + 16 + c] = CODES[v >> 4];
    }
  }
  return out;
}
// (the bytes of a binary row as its int8 values, -1 and +1: the checks' answer)
const binaryValues = (bytes: Uint8Array) => Int8Array.from({ length: bytes.length * 8 }, (_, i) => (((bytes[((i >> 7) << 4) + ((i >> 2) & 15)] >> ((((i >> 6) & 1) << 2) | (i & 3))) & 1) << 1) - 1);
export const PACKINGS: Record<GpuPacking, HeldAs> = {
  int8: { row: (n) => n, scales: (n) => n / GROUP, stored: (n) => n, scale: 4, widened: false, dot: false, grid: false, values: asBytes, on: "int8", spread: false },
  int6: { row: (n) => n, scales: (n) => n / GROUP, stored: (n) => (n * 3) / 4, scale: 4, widened: true, dot: false, grid: false, values: sixValues, on: "int6", spread: false },
  ternary: { row: (n) => n / 4, scales: (n) => n / 4 / GROUP, stored: (n) => n / 4, scale: 2, widened: false, dot: true, grid: false, values: ternaryValues, on: "ternary", spread: false },
  binary: { row: (n) => n / 4, scales: (n) => n / 4 / GROUP, stored: (n) => n / 8, scale: 2, widened: false, dot: true, grid: false, values: binaryValues, on: "ternary", spread: true },
  ...(Object.fromEntries(BLOCK_PACKINGS.map((packing) => [packing, blocks(packing)] as const)) as Record<BlockPacking, HeldAs>),
};
// T.547.3.1: the packing of a matrix's layer by its place (a stack whose layers are each of a packing of its own, packing
// "layers": its packings'), and the packings a matrix has
type Packed = Pick<Matrix, "packing" | "packings">;
export const packingAt = ({ packing, packings }: Packed, place: number): GpuPacking => (packing === "layers" ? packings![place] : packing);
export const packingsOf = ({ packing, packings }: Packed): GpuPacking[] => (packing === "layers" ? [...new Set(packings!)] : [packing]);
// T.634.3: whether a packing is multiplied by ternary's shaders on the GPU (ternary, and binary as its codes)
export const asTernary = (packing: GpuPacking) => PACKINGS[packing].on === "ternary";
// T.547.3.3: whether a packing is one of llama.cpp's blocks the GPU reads (a type of BLOCK_READS)
export const isBlock = (packing: GpuPacking): packing is BlockPacking => (BLOCK_PACKINGS as readonly string[]).includes(packing);
// T.547.3.3: the types of llama.cpp's blocks the matrices have, in BLOCK_PACKINGS' order (the texts a model compiles)
export const blocksOf = (matrices: Record<string, Packed>): BlockPacking[] =>
  BLOCK_PACKINGS.filter((packing) => Object.values(matrices).some((matrix) => packingsOf(matrix).includes(packing)));
