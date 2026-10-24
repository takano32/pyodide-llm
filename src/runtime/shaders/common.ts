// shaders/common.ts (T.351): what the shaders of every family share: the size of a group of weights, and the Step
// struct that every step of a layer reads.
// A part of src/runtime/shaders.ts, which is the window: everything outside src/runtime/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part takes its neighbours' names by static imports (T.367.3:
// a bundler links them).

export const GROUP = 32;
// (T.379.2) how the weights of a matrix are packed, where a shader's text reads them packed: int8 (four to a u32, a
// scale a group of 32; int6 is widened to it as it goes up, T.155) or ternary codes (sixteen to a u32, a scale a group
// of 128, T.232). The name of the choice for every maker that has both: the prompt's DP4A tile and a token's DP4A matrix
// (T.547.2.4: and Q4_K's blocks of 256, as they are; T.547.3.3: and every type of llama.cpp's
// blocks the GPU reads, BLOCK_PACKINGS, in the order of the catalogue: prompt.ts's BLOCK_READS)
export const BLOCK_PACKINGS = ["Q4_K", "Q2_K"] as const;
export type BlockPacking = (typeof BLOCK_PACKINGS)[number];
export type Packing = "int8" | "ternary" | BlockPacking;

// the tokens of a request and the position of the first, written once per request: every step of a layer reads it
const STEP = /* wgsl */ `struct Step { tokens: u32, pos: u32, unused0: u32, unused1: u32 }`;

export { STEP };
