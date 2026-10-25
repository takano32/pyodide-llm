// forward/blocks.ts (T.502.2): what an engine makes once for the kernels of llama.cpp's blocks (T.363.5), apart from
// createForward() (engine.ts), which calls it where the model has rows of such a type. A module of src/runtime/forward.ts.

import { BLOCK_TABLES, SPREAD_THREADS, packingOf, spreadBytes } from "../jobs.ts";
import type { Kernels } from "../jobs.ts";
import type { Part, Plan, Tensor } from "../contracts/plan.ts";

/** T.363.5: what the kernels of llama.cpp's blocks read besides a matrix: their tables, made here once of the
 * checkpoint's codebooks; and for the tile of the signed dot product, taken where this engine's relaxed SIMD multiplies
 * signed bytes by signed bytes exactly (every pair of bytes is asked, once: signed, the tests' answer instead), the bytes
 * each thread spreads a row into, for the widest row there is (0: none). alloc() takes the engine's memory. */
export function blockTables({ k, relaxed, alloc, memory, base, tensors, partsOf, signed }:
  { k: Kernels; relaxed: Kernels | null; alloc(bytes: number): number; memory: WebAssembly.Memory; base: number; tensors: Plan["tensors"];
    partsOf(t: Tensor): Part[]; signed: boolean | undefined }) {
  const tables = alloc(BLOCK_TABLES);
  if (!tensors.codebooks || k.sb_tables(tables, base + tensors.codebooks.offset) !== BLOCK_TABLES) throw new Error("The kernels of llama.cpp's blocks did not make their tables of this checkpoint.");
  let spread = 0;
  if (relaxed?.matmul_sbw_tile && (signed ?? signedDots(memory, relaxed, alloc(64)))) {
    const widest = Math.max(...Object.values(tensors).filter((t) => partsOf(t).some(({ kind }) => packingOf(kind)?.tabled !== undefined)).map((t) => t.shape[t.shape.length - 1]));
    spread = alloc(SPREAD_THREADS * spreadBytes(widest));
  }
  return { tables, spread };
}

// Whether the relaxed dot product of this engine multiplies signed bytes by signed bytes exactly, for every pair of
// bytes: what matmul_sbw_tile needs (kernels/README.md; arm64's does, x86's takes the second for unsigned). at: 48 bytes
function signedDots(memory: WebAssembly.Memory, r: Kernels, at: number) {
  const I8 = new Int8Array(memory.buffer), I32 = new Int32Array(memory.buffer);
  for (let x = -128; x < 128; x++) {
    for (let y0 = -128; y0 < 128; y0 += 16) {
      let want = 0;
      for (let j = 0; j < 16; j++) {
        I8[at + j] = x;
        I8[at + 16 + j] = y0 + j;
        want += x * (y0 + j);
      }
      r.signed_dots(at + 32, at, at + 16);
      if (I32[at / 4 + 8] + I32[at / 4 + 9] + I32[at / 4 + 10] + I32[at / 4 + 11] !== want) return false;
    }
  }
  return true;
}
