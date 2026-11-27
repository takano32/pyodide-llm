// forward/pairs.ts (T686): whether an engine of a ternary model multiplies with the int16 sums (matmul_t2x: kernels/
// relaxed/signs.ts's Pairs) or the int32 ones (matmul_t2r), chosen once as the engine is made, apart from createForward()
// (engine.ts), as forward/blocks.ts chooses the blocks' signed tile. A module of src/runtime/forward.ts.
//
// The two make the same numbers to the bit; which is faster is the device's: on AMD's Zen 3 and 4 the int16 form is 1.33
// to 1.36 times as fast in V8, on arm64's Neoverse-N2 0.63 to 0.65 (T288's tests/ternary-browser-bench.ts --relaxed;
// kernels/README.md). So it is asked of x86's relaxed dot product alone (signed_dots: x86 takes the second operand for
// unsigned, arm64 for signed: an arm64 engine, the owner's Android's, times nothing), and there timed: the model's own
// largest ternary matrix (at most MOST_BYTES of its rows) by either kernel in turn, the order turned every round, the
// fastest round of each; the int16 form where it is faster by more than the noise of a run (choice.ts's BETTER, the
// threads' margin). Once an engine, nothing on a token's path: the matrices take the kind chosen (tensors.ts).
// The kernels are cold here (the engine's first calls of them): an engine compiles a function again, optimized, in the
// background after it has run a while, and uses that from a later call on, so the first rounds may set one form's
// unoptimized code against the other's optimized one. The CI's EPYC 9V74 chose the int32 sums so once in four engines
// with the lower median of seven rounds after three calls each. So both are first called often on one row (as
// jobs.ts's warmUp() warms a thread), and the fastest of more rounds is read (a round is slowed by what else runs, never
// sped up).

import { packingOf } from "../jobs.ts";
import { BETTER } from "./choice.ts";
import type { Kernels } from "../jobs.ts";
import type { Plan } from "../contracts/plan.ts";

// the rounds of the timing (each kernel once a round, after WARM calls of each), and the most bytes of a matrix's rows
// a call reads (a 2048-wide ternary row is 512 bytes: 4096 rows. 1.7B's w1 whole is 3 MiB, a 27B's 22 MiB)
const ROUNDS = 11, WARM = 2, TINY = 1000, MOST_BYTES = 2 << 20;

/** T686: whether this engine takes matmul_t2x for its ternary matrices. relaxed: the relaxed module's kernels (null:
 * none, or the model does not run on them); direct: the matrices are on the GPU alone (none multiplied here); pairs:
 * the tests' answer instead (Trying); clock: the engine's (a made-up one in the tests). alloc() takes the engine's memory. */
export function pairsChosen({ relaxed, memory, alloc, base, tensors, direct, clock, pairs }:
  { relaxed: Kernels | null; memory: WebAssembly.Memory; alloc(bytes: number): number; base: number; tensors: Plan["tensors"];
    direct: boolean; clock: () => number; pairs: boolean | undefined }) {
  if (!relaxed?.matmul_t2x || direct) return false;
  // the ternary matrices as the file holds them (a stack of matrices of one packing: its first layer)
  const ternary = Object.values(tensors).filter((t) => t && packingOf(t.kind)?.jobs.pairs !== undefined && t.shape.length >= 2);
  if (!ternary.length) return false;
  if (pairs !== undefined) return pairs;
  if (!unsignedSecond(memory, relaxed, alloc(48))) return false;
  const widest = ternary.reduce((a, b) => (count(b) > count(a) ? b : a));
  const [rows, n] = widest.shape.slice(-2), rowBytes = packingOf(widest.kind)!.rowBytes(n);
  const taken = Math.max(1, Math.min(rows, Math.floor(MOST_BYTES / rowBytes)));
  // the activations (their codes, and after them their scales and sums: what interleave() leaves) and the outputs: zero,
  // which the kernels take as fast as any
  const xq = alloc(n), xs = alloc((n / 32) * 8), out = alloc(taken * 4), w = base + widest.offset, s = base + widest.scales;
  const kernels = [relaxed.matmul_t2r, relaxed.matmul_t2x];
  const call = (which: number) => kernels[which](out, xq, xs, w, s, n, 0, taken, 3);
  for (let i = 0; i < TINY; i++) for (const kernel of kernels) kernel(out, xq, xs, w, s, 128, 0, 1, 3);
  for (let i = 0; i < WARM; i++) { call(0); call(1); }
  const times: number[][] = [[], []];
  for (let round = 0; round < ROUNDS; round++) {
    for (const which of round % 2 ? [1, 0] : [0, 1]) {
      const began = clock();
      call(which);
      times[which].push(clock() - began);
    }
  }
  return Math.min(...times[1]) < BETTER * Math.min(...times[0]);
}
const count = (t: { shape: number[] }) => t.shape.slice(-2).reduce((a, b) => a * b, 1);

// Whether this engine's relaxed dot product takes its second operand for unsigned, as x86's PMADDUBSW does: 16 bytes of
// 1 by 16 of 255 (-1 as a signed byte) are 4 x 255 a lane there and -4 where it multiplies signed by signed (arm64's
// SDOT). Anything else (a kernel that does nothing, in the tests) is neither, and no x86. at: 48 bytes
function unsignedSecond(memory: WebAssembly.Memory, r: Kernels, at: number) {
  const U = new Uint8Array(memory.buffer), I32 = new Int32Array(memory.buffer);
  U.fill(1, at, at + 16);
  U.fill(255, at + 16, at + 32);
  r.signed_dots(at + 32, at, at + 16);
  return [0, 1, 2, 3].every((lane) => I32[at / 4 + 8 + lane] === 4 * 255);
}
