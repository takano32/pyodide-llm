// blocks_relaxed.ts (T.363.4): the kernels of blocks.ts that need the relaxed-SIMD proposal, in a module of their own
// (blockkernel_relaxed_*.wasm) for the reason kernel_relaxed.ts is one: a browser without relaxed SIMD rejects the
// whole module as it compiles it, and then has blocks.ts's alone. matmul_sbr is matmul_sb's numbers to the bit.

export { matmul_sbr } from "./blocks/token";
export { matmul_sbr_tile, matmul_sbw_tile, signed_dots } from "./blocks/tile";
