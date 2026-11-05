// blocks.ts (T363.4): the window of the kernels for rows kept as a GGUF has them, llama.cpp's blocks of 256 weights
// (the ten types Underdog Saluki 27B is made of). A module of their own (blockkernel_*.wasm), not kernel.ts's: they
// are several times the size of every other kernel together, and only a model with such rows needs them.
// blocks_relaxed.ts is the window of those that need relaxed SIMD. What this file exports is what the module exports.
//   blocks/types.ts   what each type is (the head of that file: the types, the arithmetic, the tables) and sb_tables
//   blocks/token.ts   one token's rows (matmul_sb here, matmul_sbr there), and sums_x, what its activations need
//   blocks/tile.ts    a prompt's tokens, four against one reading of a row (blocks_relaxed.ts's)
//   blocks/row.ts     a row as float32 (sb_row: a token's row of the embedding)
// A file added to blocks/ is added to the Makefile's KERNEL_PARTS too, or make kernels does not see it change.

export { sb_tables } from "./blocks/types";
export { sums_x, matmul_sb } from "./blocks/token";
export { sb_row } from "./blocks/row";
